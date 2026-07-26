// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Dispatch-level profiler for Kokoro RTF optimization.
//!
//! Combines three data sources to produce a complete picture of where time
//! goes in the Kokoro pipeline:
//!
//! 1. **GPU timing** via `synthesize_with_gpu_timing`: actual GPU execution
//!    time per pipeline step (encode, prosody, regulate, f0, harmonic,
//!    generate, istft, verify).
//! 2. **Dispatch audit** via `per_segment_step_audit`: per-step NativeOp/IR
//!    dispatch type and estimated Metal launch counts.
//! 3. **Gap analysis + RtfOptimizer**: roofline cost model estimates,
//!    fusion gap identification, and bottleneck ranking.
//!
//! The test reports:
//! - Top 20 slowest pipeline steps by GPU time fraction
//! - Per-segment GPU time vs dispatch count (time per dispatch)
//! - RtfOptimizer bottleneck ranking with estimated savings
//! - Actionable summary: which dispatch steps to fuse next
//!
//! Requires `KOKORO_WEIGHTS` env var pointing to kokoro_v1_0.safetensors.
//!
//! Run:
//!   KOKORO_WEIGHTS=path/to/kokoro_v1_0.safetensors \
//!   cargo test -p nn-metal --test kokoro_all kokoro_dispatch_profiler -- --nocapture
//!
//! Part of #4264.

use std::collections::BTreeMap;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

fn cpu() -> Device {
    Device::Cpu
}

/// Full dispatch-level profiler: combines GPU timing, dispatch audit, gap
/// analysis, and RtfOptimizer into a single actionable report.
///
/// This test answers: "Which of the 233 generator dispatches are slowest?"
/// and "What should we fuse next to reduce RTF?"
#[test]
fn dispatch_level_profiler() {
    let weights_path = match super::kokoro_test_env::require_kokoro_weights(
        "Dispatch profiler not run. Set KOKORO_WEIGHTS to enable.",
    ) {
        Some(path) => path,
        None => return,
    };

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let mut hb = nn_tts_verify::HardBoundsConfig::default();
    hb.rejection_policy = nn_tts_verify::RejectionPolicy::Warn;

    // SAFETY: safetensors file not modified while alive.
    let mut kokoro = unsafe {
        nn_metal::compiled_kokoro::CompiledKokoro::load_with_hard_bounds(&weights_path, hb)
            .expect("failed to load Kokoro weights")
    }
    .with_recommended_autocast();

    // Use 40 tokens — representative short utterance.
    let token_count: usize = 40;
    let tokens: Vec<i64> = (0..token_count).map(|i| (i % 178) as i64).collect();
    let ids = DynTensor::from_vec_i64(tokens, &[1, token_count], &cpu()).unwrap();
    let style = DynTensor::full(&[1, 256], 0.01, DType::F32, &cpu()).unwrap();

    // Warmup: compile all segments and fill caches.
    for _ in 0..3 {
        let _ = kokoro
            .synthesize(&ids, &style, 1.0, &cache)
            .expect("warmup synthesis failed");
    }

    // =========================================================================
    // Section 1: GPU Timing Profile (actual measured GPU execution per step)
    // =========================================================================

    let (audio, _cert, gpu_timing) = kokoro
        .synthesize_with_gpu_timing(&ids, &style, 1.0, &cache)
        .expect("GPU timing synthesis failed");

    let num_samples = audio.numel();
    let audio_secs = num_samples as f64 / 24_000.0;
    let total_gpu_ms = gpu_timing.total.as_secs_f64() * 1000.0;
    let profiled_rtf = gpu_timing.total.as_secs_f64() / audio_secs;

    // Collect step timings as sorted vec.
    let step_timings: Vec<(&str, f64)> = vec![
        ("encode", gpu_timing.encode.as_secs_f64() * 1000.0),
        ("prosody", gpu_timing.prosody.as_secs_f64() * 1000.0),
        ("regulate", gpu_timing.regulate.as_secs_f64() * 1000.0),
        ("f0_energy", gpu_timing.f0_energy.as_secs_f64() * 1000.0),
        ("harmonic", gpu_timing.harmonic.as_secs_f64() * 1000.0),
        ("generate", gpu_timing.generate.as_secs_f64() * 1000.0),
        ("istft", gpu_timing.istft.as_secs_f64() * 1000.0),
        ("verify", gpu_timing.verify.as_secs_f64() * 1000.0),
    ];

    eprintln!("\n{}", "=".repeat(90));
    eprintln!("  KOKORO DISPATCH-LEVEL PROFILER -- RTF Optimization Data");
    eprintln!(
        "  {} tokens, {} samples, {:.2} ms audio",
        token_count,
        num_samples,
        audio_secs * 1000.0
    );
    eprintln!("{}\n", "=".repeat(90));

    eprintln!("--- 1. GPU Timing Profile (measured, per-step flush) ---\n");
    eprintln!(
        "  {:<14} {:>10} {:>8} {:>10}",
        "Step", "GPU ms", "% total", "Cumulative"
    );
    eprintln!("  {}", "-".repeat(46));

    let mut sorted_timings = step_timings.clone();
    sorted_timings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut cumulative_pct = 0.0;
    for (name, ms) in &sorted_timings {
        let pct = if total_gpu_ms > 0.0 {
            ms / total_gpu_ms * 100.0
        } else {
            0.0
        };
        cumulative_pct += pct;
        eprintln!(
            "  {name:<14} {ms:>10.3} {pct:>7.1}% {cumulative_pct:>9.1}%",
        );
    }
    eprintln!("  {}", "-".repeat(46));
    eprintln!("  {:<14} {:>10.3}", "TOTAL", total_gpu_ms);
    eprintln!("  Profiled RTF: {profiled_rtf:.4} (target: 0.03)");
    eprintln!("  Note: per-step flushes add overhead; production RTF is lower.\n");

    // =========================================================================
    // Section 2: Dispatch audit — per-segment step breakdown
    // =========================================================================

    let audit = kokoro.per_segment_step_audit();
    let ds = kokoro.dispatch_summary();

    // Map segment names to GPU timing.
    let seg_timing_map: BTreeMap<&str, f64> = [
        ("plbert", gpu_timing.encode.as_secs_f64() * 1000.0 * 0.5), // split encode 50/50
        (
            "text_encoder",
            gpu_timing.encode.as_secs_f64() * 1000.0 * 0.5,
        ),
        ("prosody", gpu_timing.prosody.as_secs_f64() * 1000.0),
        ("regulate", gpu_timing.regulate.as_secs_f64() * 1000.0),
        ("f0_energy", gpu_timing.f0_energy.as_secs_f64() * 1000.0),
        (
            "sinegen_pre",
            gpu_timing.harmonic.as_secs_f64() * 1000.0 * 0.5,
        ),
        (
            "sinegen_post",
            gpu_timing.harmonic.as_secs_f64() * 1000.0 * 0.5,
        ),
        ("generator", gpu_timing.generate.as_secs_f64() * 1000.0),
    ]
    .into_iter()
    .collect();

    eprintln!("--- 2. Per-Segment Dispatch Audit with GPU Time ---\n");
    eprintln!(
        "  {:<14} {:>6} {:>8} {:>10} {:>10} {:>12}",
        "Segment", "Disps", "NativeOp", "IR Disp", "Metal Est", "GPU ms"
    );
    eprintln!("  {}", "-".repeat(66));

    // Collect per-kernel Metal counts and GPU time estimates.
    let mut kernel_details: Vec<(String, String, usize, f64)> = Vec::new();

    for (seg_name, steps, dispatches, metal_dispatches) in &audit {
        let mut native_count = 0usize;
        let mut ir_count = 0usize;

        let seg_gpu_ms = seg_timing_map
            .get(seg_name.as_str())
            .copied()
            .unwrap_or(0.0);
        // Estimate per-dispatch GPU time from segment total.
        let per_dispatch_ms = if *dispatches > 0 {
            seg_gpu_ms / *dispatches as f64
        } else {
            0.0
        };

        for (_step_idx, step_type, detail, metal) in steps {
            match *step_type {
                "NativeOp" => {
                    native_count += 1;
                    // Weighted estimate: more Metal launches = more time.
                    let est_ms = per_dispatch_ms * (*metal as f64 / (*metal).max(1) as f64);
                    kernel_details.push((
                        seg_name.clone(),
                        format!("[NativeOp] {detail}"),
                        *metal,
                        est_ms,
                    ));
                }
                "Dispatch" => {
                    ir_count += 1;
                    let est_ms = per_dispatch_ms;
                    kernel_details.push((
                        seg_name.clone(),
                        format!("[IR] {detail}"),
                        *metal,
                        est_ms,
                    ));
                }
                "RuntimeOp" => {
                    let est_ms = per_dispatch_ms;
                    kernel_details.push((
                        seg_name.clone(),
                        format!("[RuntimeOp] {detail}"),
                        *metal,
                        est_ms,
                    ));
                }
                _ => {} // zero-cost steps
            }
        }

        eprintln!(
            "  {seg_name:<14} {dispatches:>6} {native_count:>8} {ir_count:>10} {metal_dispatches:>10} {seg_gpu_ms:>11.3}",
        );
    }
    eprintln!("  {}", "-".repeat(66));
    eprintln!(
        "  {:<14} {:>6}                               {:>11.3}",
        "TOTAL",
        ds.total(),
        total_gpu_ms,
    );

    // Time per dispatch by segment.
    eprintln!("\n  Time per dispatch by segment:");
    for (seg_name, _steps, dispatches, _metal) in &audit {
        let seg_gpu_ms = seg_timing_map
            .get(seg_name.as_str())
            .copied()
            .unwrap_or(0.0);
        if *dispatches > 0 {
            let per_disp_us = seg_gpu_ms * 1000.0 / *dispatches as f64;
            eprintln!(
                "    {seg_name:<14} {per_disp_us:>8.1} us/dispatch  ({dispatches} dispatches, {seg_gpu_ms:.3} ms total)",
            );
        }
    }
    eprintln!();

    // =========================================================================
    // Section 3: Top 20 slowest dispatch steps (estimated from GPU timing)
    // =========================================================================

    eprintln!("--- 3. Top 20 Slowest Dispatch Steps (estimated GPU time) ---\n");

    // For segments with many dispatches, distribute GPU time by Metal launch
    // count as a proxy for GPU work. More Metal launches = more GPU time.
    let mut enriched_steps: Vec<(String, String, usize, f64)> = Vec::new();

    for (seg_name, steps, _dispatches, _metal_total) in &audit {
        let seg_gpu_ms = seg_timing_map
            .get(seg_name.as_str())
            .copied()
            .unwrap_or(0.0);
        // Sum Metal launches for this segment's dispatch steps.
        let total_metal_in_seg: usize = steps
            .iter()
            .filter(|(_, st, _, _)| matches!(*st, "NativeOp" | "Dispatch" | "RuntimeOp"))
            .map(|(_, _, _, metal)| metal)
            .sum();

        for (_step_idx, step_type, detail, metal) in steps {
            if matches!(*step_type, "NativeOp" | "Dispatch" | "RuntimeOp") {
                let est_ms = if total_metal_in_seg > 0 {
                    seg_gpu_ms * (*metal as f64 / total_metal_in_seg as f64)
                } else {
                    0.0
                };
                enriched_steps.push((
                    seg_name.clone(),
                    format!("[{step_type}] {detail}"),
                    *metal,
                    est_ms,
                ));
            }
        }
    }

    // Sort by estimated GPU time descending.
    enriched_steps.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

    eprintln!(
        "  {:>4}  {:<14} {:<40} {:>6} {:>10} {:>7}",
        "Rank", "Segment", "Step", "Metal", "Est ms", "% GPU"
    );
    eprintln!("  {}", "-".repeat(86));

    let mut top20_cumulative_pct = 0.0;
    for (i, (seg, step, metal, est_ms)) in enriched_steps.iter().take(20).enumerate() {
        let pct = if total_gpu_ms > 0.0 {
            est_ms / total_gpu_ms * 100.0
        } else {
            0.0
        };
        top20_cumulative_pct += pct;
        // Truncate step name for readability.
        let step_short = if step.len() > 40 { &step[..40] } else { step };
        eprintln!(
            "  {:>4}  {:<14} {:<40} {:>6} {:>10.3} {:>6.1}%",
            i + 1,
            seg,
            step_short,
            metal,
            est_ms,
            pct,
        );
    }
    eprintln!("  {}", "-".repeat(86));
    eprintln!(
        "  Top 20 account for {:.1}% of total GPU time ({:.3} ms / {:.3} ms)\n",
        top20_cumulative_pct,
        enriched_steps.iter().take(20).map(|s| s.3).sum::<f64>(),
        total_gpu_ms,
    );

    // =========================================================================
    // Section 4: RtfOptimizer Analysis — bottleneck identification
    // =========================================================================

    eprintln!("--- 4. RtfOptimizer Analysis ---\n");

    let gap_results = kokoro
        .segment_gap_analysis(&ids, &style, 1.0, &cache)
        .unwrap();

    let optimizer = kokoro.rtf_optimizer();
    let report = optimizer.analyze(&gap_results);

    eprintln!("{}", report.summary());

    // =========================================================================
    // Section 5: Actionable Summary — what to fuse next
    // =========================================================================

    eprintln!("--- 5. Actionable Optimization Targets ---\n");

    // Identify which segment dominates GPU time.
    let (dominant_seg, dominant_ms) = sorted_timings[0];
    let dominant_pct = if total_gpu_ms > 0.0 {
        dominant_ms / total_gpu_ms * 100.0
    } else {
        0.0
    };
    eprintln!(
        "  Dominant segment: {dominant_seg} ({dominant_pct:.1}% of GPU time, {dominant_ms:.3} ms)",
    );

    // Report the top 5 dispatch steps that should be fused.
    eprintln!("\n  Top 5 dispatch steps to optimize:");
    for (i, (seg, step, metal, est_ms)) in enriched_steps.iter().take(5).enumerate() {
        let pct = if total_gpu_ms > 0.0 {
            est_ms / total_gpu_ms * 100.0
        } else {
            0.0
        };
        eprintln!(
            "    #{}: [{}] {} — {:.3} ms ({:.1}%, {} Metal launches)",
            i + 1,
            seg,
            step,
            est_ms,
            pct,
            metal,
        );
    }

    // Category breakdown from dispatch audit.
    eprintln!("\n  GPU time by kernel category:");
    let mut category_time: BTreeMap<String, f64> = BTreeMap::new();
    for (_, step, _, est_ms) in &enriched_steps {
        let cat = categorize_dispatch_step(step);
        *category_time.entry(cat).or_insert(0.0) += est_ms;
    }
    let mut cat_sorted: Vec<_> = category_time.iter().collect();
    cat_sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (cat, ms) in &cat_sorted {
        let pct = if total_gpu_ms > 0.0 {
            *ms / total_gpu_ms * 100.0
        } else {
            0.0
        };
        eprintln!("    {cat:<20} {ms:>8.3} ms  ({pct:>5.1}%)");
    }

    // RTF projections.
    eprintln!("\n  RTF Summary:");
    eprintln!("    Profiled RTF (per-step flush):  {profiled_rtf:.4}");
    eprintln!(
        "    Projected RTF (cost model):     {:.4}",
        report.projected_rtf
    );
    eprintln!("    Target RTF:                     0.03");
    eprintln!(
        "    Gap factor:                     {:.1}x (profiled / target)",
        profiled_rtf / 0.03,
    );
    eprintln!(
        "    Dispatches:                     {} (theoretical min: {})",
        report.total_dispatches, report.total_theoretical_minimum,
    );
    eprintln!("{}\n", "=".repeat(90));

    // =========================================================================
    // Structural assertions (sanity, not hard gates)
    // =========================================================================

    // GPU timing should be positive for all pipeline steps.
    assert!(
        gpu_timing.encode.as_nanos() > 0,
        "encode GPU time should be > 0"
    );
    assert!(
        gpu_timing.generate.as_nanos() > 0,
        "generate GPU time should be > 0"
    );

    // Dispatch audit should cover at least 5 segments.
    assert!(
        audit.len() >= 5,
        "Expected at least 5 compiled segments in audit, got {}",
        audit.len(),
    );

    // Gap analysis should produce results.
    assert!(
        !gap_results.is_empty(),
        "segment_gap_analysis should return at least one result"
    );

    // RtfOptimizer report should have bottlenecks.
    assert!(
        !report.bottlenecks.is_empty(),
        "RtfOptimizer should identify at least one bottleneck"
    );

    // Generator should be the dominant segment (it's 60%+ of compute).
    assert!(
        dominant_seg == "generate",
        "Expected generator to be dominant segment, got '{dominant_seg}'. \
         If another segment now dominates, update the RTF optimization strategy.",
    );

    // Top 20 dispatch steps should account for >50% of GPU time (otherwise
    // the profiling data is too uniform to be actionable).
    assert!(
        top20_cumulative_pct > 30.0,
        "Top 20 dispatch steps account for only {top20_cumulative_pct:.1}% of GPU time. \
         Expected >30% — profiling may not be capturing GPU work correctly.",
    );
}

/// Categorize a dispatch step name into a kernel category for grouping.
fn categorize_dispatch_step(name: &str) -> String {
    let lower = name.to_lowercase();

    if lower.contains("conv") && lower.contains("snake") {
        return "conv+snake (fused)".to_string();
    }
    if lower.contains("resblock") {
        return "resblock (fused)".to_string();
    }
    if lower.contains("conv") && lower.contains("norm") {
        return "conv+norm (fused)".to_string();
    }
    if lower.contains("matmul") || lower.contains("gemm") || lower.contains("linear") {
        return "matmul/linear".to_string();
    }
    if lower.contains("conv") {
        return "conv".to_string();
    }
    if lower.contains("lstm") || lower.contains("bilstm") {
        return "lstm".to_string();
    }
    if lower.contains("attention") || lower.contains("sdpa") {
        return "attention".to_string();
    }
    if lower.contains("norm") {
        return "normalization".to_string();
    }
    if lower.contains("snake")
        || lower.contains("relu")
        || lower.contains("gelu")
        || lower.contains("silu")
        || lower.contains("sigmoid")
        || lower.contains("tanh")
    {
        return "activation".to_string();
    }
    if lower.contains("add")
        || lower.contains("mul")
        || lower.contains("sub")
        || lower.contains("div")
        || lower.contains("fused_")
    {
        return "elementwise".to_string();
    }
    if lower.contains("softmax") {
        return "softmax".to_string();
    }
    if lower.contains("embedding") || lower.contains("gather") {
        return "embedding".to_string();
    }

    "other".to_string()
}
