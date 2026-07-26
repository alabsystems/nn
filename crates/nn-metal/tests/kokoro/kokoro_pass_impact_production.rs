// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Production Kokoro PeepholeConfig optimizer analysis (D=512).
//!
//! Runs the optimizer search with a 60-second budget per segment on the full
//! production Kokoro model. Reports per-segment baseline vs optimized dispatch
//! counts and saves a durable summary to `kokoro_optimizer_summary.json` in
//! the project root.
//!
//! Key finding from initial runs: PeepholeConfig optimization yields ZERO
//! dispatch savings for Kokoro. All 26 peephole passes are orthogonal for
//! this model -- each pass either matches a pattern in a segment or doesn't,
//! and no combination produces fewer dispatches than all-passes-enabled.
//! Dispatch reduction must come from NativeOp fusion or new compiler fusion
//! patterns, not from PeepholeConfig tuning.
//!
//! Gated behind `KOKORO_WEIGHTS` env var. Skips gracefully when unset.
//!
//! Part of #4264.

use std::time::Duration;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

/// Helper: load production Kokoro and prepare test inputs.
fn load_production_kokoro() -> Option<(
    nn_metal::compiled_kokoro::CompiledKokoro,
    nn_metal::PipelineCache,
    DynTensor,
    DynTensor,
)> {
    let weights_path =
        super::kokoro_test_env::require_kokoro_weights("production pass-impact analysis not run.")?;

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let mut hb = nn_tts_verify::HardBoundsConfig::default();
    hb.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;

    // SAFETY: safetensors file not modified while alive.
    let kokoro = unsafe {
        nn_metal::compiled_kokoro::CompiledKokoro::load_with_hard_bounds(&weights_path, hb)
            .expect("failed to load production Kokoro weights")
    };

    let input_ids = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, 8],
        &Device::Cpu,
    )
    .unwrap();
    let style = DynTensor::full(&[1, 256], 0.01, DType::F32, &Device::Cpu).unwrap();

    Some((kokoro, cache, input_ids, style))
}

/// Production optimizer search with 60s budget per segment.
///
/// Reports per-segment dispatch counts before/after optimization and saves
/// a machine-readable summary JSON for tracking across commits.
#[test]
fn kokoro_optimizer_search_d512() {
    let (mut kokoro, cache, input_ids, style) = match load_production_kokoro() {
        Some(v) => v,
        None => return,
    };

    // Synthesize once to compile all segments.
    let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache);

    // Run optimizer search with 60s budget per segment.
    let results = kokoro
        .segment_optimizer_search(&input_ids, &style, 1.0, &cache, Duration::from_mins(1))
        .unwrap();

    // --- Report table ---
    eprintln!("\n=== PRODUCTION KOKORO OPTIMIZER SEARCH (D=512, 60s/segment) ===");
    eprintln!(
        "{:<16} {:>10} {:>10} {:>10} {:>12} {:>12} {:>8}",
        "Segment", "Baseline", "Optimized", "Saved", "Base Cost", "Best Cost", "Configs"
    );
    eprintln!("{}", "-".repeat(84));

    let mut total_baseline = 0usize;
    let mut total_optimized = 0usize;
    let mut total_base_cost = 0.0_f64;
    let mut total_best_cost = 0.0_f64;
    let mut segments_improved = 0usize;

    for seg in &results {
        let opt = &seg.optimization;
        let saved = opt
            .baseline_dispatch_count
            .saturating_sub(opt.dispatch_count);
        eprintln!(
            "{:<16} {:>10} {:>10} {:>10} {:>10.0}us {:>10.0}us {:>8}",
            seg.segment_name,
            opt.baseline_dispatch_count,
            opt.dispatch_count,
            saved,
            opt.baseline_cost_ns / 1e3,
            opt.best_cost_ns / 1e3,
            opt.configs_explored,
        );
        total_baseline += opt.baseline_dispatch_count;
        total_optimized += opt.dispatch_count;
        total_base_cost += opt.baseline_cost_ns;
        total_best_cost += opt.best_cost_ns;

        if saved > 0 {
            segments_improved += 1;
            eprintln!("  -> Winning PeepholeConfig: {:?}", opt.config);
        }
    }

    let total_saved = total_baseline.saturating_sub(total_optimized);
    let cost_reduction_pct = if total_base_cost > 0.0 {
        (total_base_cost - total_best_cost) / total_base_cost * 100.0
    } else {
        0.0
    };
    eprintln!("{}", "-".repeat(84));
    eprintln!(
        "{:<16} {:>10} {:>10} {:>10} {:>10.0}us {:>10.0}us",
        "TOTAL",
        total_baseline,
        total_optimized,
        total_saved,
        total_base_cost / 1e3,
        total_best_cost / 1e3,
    );
    eprintln!(
        "Cost reduction: {cost_reduction_pct:.1}%, Segments improved: {segments_improved}/{}",
        results.len(),
    );

    let gen_failed = !results.iter().any(|r| r.segment_name == "generator");
    if gen_failed {
        eprintln!("NOTE: Generator segment trace failed (shape mismatch, known issue)");
    }
    eprintln!("===============================================================\n");

    // --- Save durable summary ---
    let summary_segments: Vec<serde_json::Value> = results
        .iter()
        .map(|seg| {
            serde_json::json!({
                "segment": seg.segment_name,
                "baseline_dispatches": seg.optimization.baseline_dispatch_count,
                "optimized_dispatches": seg.optimization.dispatch_count,
                "saved": seg.optimization.baseline_dispatch_count
                    .saturating_sub(seg.optimization.dispatch_count),
                "baseline_cost_us": format!("{:.1}", seg.optimization.baseline_cost_ns / 1e3),
                "best_cost_us": format!("{:.1}", seg.optimization.best_cost_ns / 1e3),
                "configs_explored": seg.optimization.configs_explored,
            })
        })
        .collect();

    let summary = serde_json::json!({
        "model": "kokoro_v1_0",
        "d_en": 512,
        "input_tokens": 8,
        "budget_per_segment_secs": 60,
        "peephole_field_count": 26,
        "total_config_space": 1u64 << 26,
        "segments": summary_segments,
        "totals": {
            "baseline_dispatches": total_baseline,
            "optimized_dispatches": total_optimized,
            "saved_dispatches": total_saved,
            "baseline_cost_us": format!("{:.1}", total_base_cost / 1e3),
            "best_cost_us": format!("{:.1}", total_best_cost / 1e3),
            "cost_reduction_pct": format!("{:.1}", cost_reduction_pct),
            "segments_improved": segments_improved,
            "segments_traced": results.len(),
            "generator_trace_failed": gen_failed,
        },
        "finding": "PeepholeConfig optimization yields zero dispatch savings for Kokoro D=512. \
                    All 26 peephole passes are orthogonal: each pass independently matches (or \
                    does not match) patterns in each segment. No combination of disabled passes \
                    produces fewer dispatches than all-enabled. Dispatch reduction from 251 to \
                    target 60 requires: (1) NativeOp fusion to merge adjacent GPU dispatches, \
                    (2) fixing generator segment trace (shape mismatch), (3) new compiler \
                    fusion patterns for unfused dispatch sequences.",
    });

    let summary_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("kokoro_optimizer_summary.json");

    std::fs::write(
        &summary_path,
        serde_json::to_string_pretty(&summary).unwrap(),
    )
    .expect("failed to write optimizer summary");
    eprintln!("Summary saved to: {}", summary_path.display());

    // --- Assertions ---

    // At least 7 segments must trace.
    assert!(
        results.len() >= 7,
        "Expected at least 7 segments, got {}",
        results.len(),
    );

    // Optimizer must never regress.
    for seg in &results {
        assert!(
            seg.optimization.dispatch_count <= seg.optimization.baseline_dispatch_count,
            "Segment {}: optimizer regressed! {} > {} dispatches",
            seg.segment_name,
            seg.optimization.dispatch_count,
            seg.optimization.baseline_dispatch_count,
        );
    }

    // Each segment must explore at least 2 configs.
    for seg in &results {
        assert!(
            seg.optimization.configs_explored >= 2,
            "Segment {}: only {} config(s) explored",
            seg.segment_name,
            seg.optimization.configs_explored,
        );
    }

    // All costs must be finite and non-negative.
    for seg in &results {
        let opt = &seg.optimization;
        assert!(
            opt.baseline_cost_ns.is_finite() && opt.baseline_cost_ns >= 0.0,
            "Segment {}: invalid baseline cost: {}",
            seg.segment_name,
            opt.baseline_cost_ns,
        );
        assert!(
            opt.best_cost_ns.is_finite() && opt.best_cost_ns >= 0.0,
            "Segment {}: invalid best cost: {}",
            seg.segment_name,
            opt.best_cost_ns,
        );
    }
}
