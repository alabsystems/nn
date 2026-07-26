// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fusion gap analysis gate for all 8 Kokoro segments.
//!
//! Traces each segment, compiles to a plan, and runs `analyze_fusion_gaps` +
//! `CostModel::apple_m4().estimate()`. Prints per-segment dispatch counts,
//! theoretical minimums, cost estimates, and blocker distributions.
//!
//! Part of #3836.

/// Gate: fusion gap analysis for all compiled Kokoro segments.
///
/// Ensures that gap analysis runs on all traceable segments and that
/// `theoretical_minimum <= dispatch_count` for every segment (sanity).
///
/// Prints a dashboard for manual inspection of optimization opportunities.
///
/// Part of #3836.
#[test]
fn gate_fusion_gap_analysis() {
    let (mut kokoro, cache) = super::kokoro_gates::build_kokoro();
    let (input_ids, style) = super::kokoro_gates::test_inputs();

    // Synthesize first to compile all segments and populate caches.
    let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache).unwrap();

    // Run gap analysis on all 8 segments.
    let results = kokoro
        .segment_gap_analysis(&input_ids, &style, 1.0, &cache)
        .unwrap();

    // Print dashboard.
    eprintln!("\n=== FUSION GAP ANALYSIS (all segments) ===");
    let mut total_dispatches = 0usize;
    let mut total_theoretical_min = 0usize;
    for seg in &results {
        eprintln!(
            "  [{:>14}] dispatches: {:>3}, theoretical_min: {:>3}, gaps: {:>3}, cost: {:>8.0} ns",
            seg.segment_name,
            seg.dispatch_count,
            seg.theoretical_minimum,
            seg.gap_analysis.gaps.len(),
            seg.cost_estimate.total_ns,
        );
        // Print blocker distribution.
        let counts = seg.gap_analysis.blocker_counts();
        if !counts.is_empty() {
            for (blocker, count) in &counts {
                eprintln!("    {blocker}: {count}");
            }
        }
        total_dispatches += seg.dispatch_count;
        total_theoretical_min += seg.theoretical_minimum;
    }
    let gap_pct = if total_dispatches > 0 {
        (total_dispatches - total_theoretical_min) as f64 / total_dispatches as f64 * 100.0
    } else {
        0.0
    };
    eprintln!(
        "  TOTAL: dispatches={total_dispatches}, theoretical_min={total_theoretical_min}, \
         gap={gap_pct:.1}%"
    );
    eprintln!("  Segments analyzed: {}/8", results.len());
    eprintln!("==========================================\n");

    // -- Assertions --

    // All 8 segments should be analyzable: plbert, text, prosody, regulate,
    // f0, sinegen_pre, sinegen_post, generator. (#4309 fixed generator trace.)
    assert!(
        results.len() >= 8,
        "Expected all 8 analyzed segments, got {}. \
         Segments: {:?}",
        results.len(),
        results
            .iter()
            .map(|s| s.segment_name.as_str())
            .collect::<Vec<_>>(),
    );

    // Generator segment must be present (was previously failing due to
    // transposed f0 causing Conv1d in_channels mismatch, #4309).
    assert!(
        results.iter().any(|s| s.segment_name == "generator"),
        "Generator segment missing from gap analysis results. \
         Segments: {:?}",
        results
            .iter()
            .map(|s| s.segment_name.as_str())
            .collect::<Vec<_>>(),
    );

    // Sanity: theoretical_minimum <= dispatch_count for every segment.
    for seg in &results {
        assert!(
            seg.theoretical_minimum <= seg.dispatch_count,
            "{}: theoretical_min ({}) > dispatch_count ({})",
            seg.segment_name,
            seg.theoretical_minimum,
            seg.dispatch_count,
        );
    }

    // Sanity: total dispatches should be > 0.
    assert!(
        total_dispatches > 0,
        "Total dispatches across all segments is 0 -- gap analysis may be broken",
    );
}

/// Production gap analysis: loads full D=512 Kokoro from safetensors,
/// synthesizes once to compile all segments, then runs `segment_gap_analysis()`
/// and reports per-segment dispatch counts, theoretical minimums, and top
/// blocker distributions.
///
/// Gated behind `KOKORO_WEIGHTS` env var. Skips gracefully when unset.
///
/// Part of #3828.
#[test]
fn production_gap_analysis() {
    let weights_path =
        match super::kokoro_test_env::require_kokoro_weights("production gap analysis not run.") {
            Some(p) => p,
            None => return,
        };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Use Warn policy: test tokens produce click artifacts with production
    // weights that fail the no_clicks hard bound. Part of #4262.
    let mut hb = nn_tts_verify::HardBoundsConfig::default();
    hb.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;

    // SAFETY: safetensors file not modified while alive.
    let mut kokoro = unsafe {
        nn_metal::compiled_kokoro::CompiledKokoro::load_with_hard_bounds(&weights_path, hb)
            .expect("failed to load production Kokoro weights")
    };

    // Standard test utterance: 8 phoneme tokens.
    let input_ids = nn_core::dyn_tensor::DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, 8],
        &nn_core::Device::Cpu,
    )
    .unwrap();
    let style = nn_core::dyn_tensor::DynTensor::full(
        &[1, 256],
        0.01,
        nn_core::DType::F32,
        &nn_core::Device::Cpu,
    )
    .unwrap();

    // Synthesize once to compile all segments and populate caches.
    let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache);

    // Run gap analysis on all segments.
    let results = kokoro
        .segment_gap_analysis(&input_ids, &style, 1.0, &cache)
        .unwrap();

    // --- Per-segment dashboard ---
    eprintln!("\n=== PRODUCTION FUSION GAP ANALYSIS ===");
    eprintln!(
        "{:<16} {:>10} {:>10} {:>6}   Top Blockers",
        "Segment", "Dispatches", "TheoMin", "Gaps",
    );
    eprintln!("{}", "-".repeat(72));

    let mut total_dispatches = 0usize;
    let mut total_theoretical_min = 0usize;

    for seg in &results {
        let counts = seg.gap_analysis.blocker_counts();
        // Format top blockers sorted by count descending.
        let mut sorted_blockers: Vec<_> = counts.into_iter().collect();
        sorted_blockers.sort_by_key(|x| std::cmp::Reverse(x.1));
        let blocker_str: String = sorted_blockers
            .iter()
            .map(|(name, count)| format!("{name}({count})"))
            .collect::<Vec<_>>()
            .join(", ");

        eprintln!(
            "{:<16} {:>10} {:>10} {:>6}   {}",
            seg.segment_name,
            seg.dispatch_count,
            seg.theoretical_minimum,
            seg.gap_analysis.gaps.len(),
            blocker_str,
        );
        total_dispatches += seg.dispatch_count;
        total_theoretical_min += seg.theoretical_minimum;
    }

    let gap_pct = if total_dispatches > 0 {
        (total_dispatches - total_theoretical_min) as f64 / total_dispatches as f64 * 100.0
    } else {
        0.0
    };
    eprintln!("{}", "-".repeat(72));
    eprintln!(
        "{:<16} {:>10} {:>10}   gap={:.1}%",
        "TOTAL", total_dispatches, total_theoretical_min, gap_pct,
    );
    eprintln!("Segments analyzed: {}/8", results.len());
    eprintln!("======================================\n");

    // --- Assertions ---

    // At least 2 segments must be analyzable with production weights.
    // plbert and text are always traceable; others depend on device-matching
    // of GPU-resident intermediates.
    assert!(
        results.len() >= 2,
        "Expected at least 2 analyzed production segments, got {}. \
         Segments: {:?}",
        results.len(),
        results
            .iter()
            .map(|s| s.segment_name.as_str())
            .collect::<Vec<_>>(),
    );

    // Sanity: theoretical_minimum <= dispatch_count for every segment.
    for seg in &results {
        assert!(
            seg.theoretical_minimum <= seg.dispatch_count,
            "{}: theoretical_min ({}) > dispatch_count ({})",
            seg.segment_name,
            seg.theoretical_minimum,
            seg.dispatch_count,
        );
    }

    // Total dispatches must not exceed the current gate.
    // Measured 2026-03-31 with all 8 segments including generator:
    //   plbert=112, text=15, prosody=37, regulate=4, f0=61,
    //   sinegen_pre=11, sinegen_post=13, generator=234. Total=487.
    // Note: this is the per-segment re-traced count (higher than production
    // because segment_gap_analysis re-compiles each segment independently
    // without cross-segment optimization — generator alone is 234 here
    // vs 46 logical dispatches in the production pipeline).
    assert!(
        total_dispatches <= 500,
        "Total production dispatches ({total_dispatches}) exceeds gate (500)",
    );
}
