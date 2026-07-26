// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-step Metal dispatch decomposition — identifies where the 190 eager
//! dispatch gap comes from.
//!
//! W11 measured 424 actual vs 234 estimated Metal dispatches (95bb032).
//! The 190-dispatch gap could be compiled segment underestimates (H1) or
//! untracked eager dispatches (H2). This test measures each pipeline step
//! individually using `reset_counters()` / `dispatch_stats()` to decompose
//! the total into per-step actuals.
//!
//! Run: `cargo test -p nn-metal --test kokoro_all kokoro_dispatch_decomposition -- --nocapture`
//!
//! Part of #3192 (extracted from #1815 epic).

use nn_core::layers::NanCheckPolicy;

/// Per-step Metal dispatch decomposition.
///
/// Wraps each pipeline step with counter reset/read to determine actual
/// per-step GPU dispatch counts. Prints a table comparing planner estimates
/// to actual runtime dispatches.
///
/// Part of #3192, #1815.
#[test]
fn gate_dispatch_decomposition() {
    let (mut kokoro, cache) = super::kokoro_gates::build_kokoro();
    let (input_ids, style) = super::kokoro_gates::test_inputs();

    // Warmup: compile all segments (cold path JIT).
    let _ = kokoro.synthesize(&input_ids, &style, 1.0, &cache).unwrap();

    // Split style once (minimal GPU work — narrow is a view).
    let style_split = kokoro.split_style(&style).unwrap();
    let decoder_style = style_split
        .decoder_style
        .to_device(&nn_core::Device::Metal { device_id: 0 })
        .unwrap();
    let prosody_style = style_split
        .prosody_style
        .to_device(&nn_core::Device::Metal { device_id: 0 })
        .unwrap();

    // Per-step measurement inside NaN-skip scope (matches synthesize()).
    let mut step_counts: Vec<(&str, usize)> = Vec::new();

    nn_core::layers::with_nan_check_policy(NanCheckPolicy::Skip, || {
        // Step 1+2: PlBert + TextEncoder (compiled segments 0+1)
        nn_metal::reset_counters();
        let enc = kokoro.step_encode(&input_ids, &cache).unwrap();
        let encode_dispatches = nn_metal::dispatch_stats().compute_encodings;
        step_counts.push(("step_encode (PlBert+TextEncoder)", encode_dispatches));

        // Step 3: ProsodyPredictor (compiled segment 2)
        nn_metal::reset_counters();
        let pros = kokoro
            .step_predict_prosody(&enc.bert_features, &prosody_style, enc.seq_len, &cache)
            .unwrap();
        let prosody_dispatches = nn_metal::dispatch_stats().compute_encodings;
        step_counts.push(("step_predict_prosody", prosody_dispatches));

        // Step 4: Duration + length_regulate (compiled segment 5 + eager scatter)
        nn_metal::reset_counters();
        let reg = kokoro
            .step_regulate(
                &pros.dur_logits,
                &pros.features,
                &enc.text_features,
                1.0,
                &cache,
            )
            .unwrap();
        let regulate_dispatches = nn_metal::dispatch_stats().compute_encodings;
        step_counts.push(("step_regulate (compiled+eager)", regulate_dispatches));

        // Step 5: F0EnergyPredictor (compiled segment 3)
        nn_metal::reset_counters();
        let f0e = kokoro
            .step_predict_f0_energy(&reg.aligned_dur, &prosody_style, reg.t_mel, &cache)
            .unwrap();
        let f0_dispatches = nn_metal::dispatch_stats().compute_encodings;
        step_counts.push(("step_predict_f0_energy", f0_dispatches));

        // Step 6: Harmonic source / SineGen (eager)
        nn_metal::reset_counters();
        let har_source = kokoro
            .step_harmonic_source(&f0e.f0, &f0e.energy, reg.t_mel, &cache)
            .unwrap();
        let sinegen_dispatches = nn_metal::dispatch_stats().compute_encodings;
        step_counts.push(("step_harmonic_source (SineGen, eager)", sinegen_dispatches));

        // Step 7: Generator / FullDecoder (compiled segment 4)
        nn_metal::reset_counters();
        let generator = kokoro
            .step_generate(
                &reg.regulated,
                &f0e.f0,
                &f0e.energy,
                &decoder_style,
                &har_source,
                reg.t_mel,
                &cache,
            )
            .unwrap();
        let generator_dispatches = nn_metal::dispatch_stats().compute_encodings;
        step_counts.push(("step_generate (Generator)", generator_dispatches));

        // Step 8: GPU iSTFT → PCM audio (eager)
        nn_metal::reset_counters();
        let _audio = kokoro
            .step_istft(&generator.magnitude, &generator.phase, &cache)
            .unwrap();
        let istft_dispatches = nn_metal::dispatch_stats().compute_encodings;
        step_counts.push(("step_istft (eager)", istft_dispatches));

        Ok::<(), nn_core::TensorError>(())
    })
    .unwrap();

    // Print decomposition table.
    let total_per_step: usize = step_counts.iter().map(|(_, n)| n).sum();

    eprintln!("\n=== PER-STEP DISPATCH DECOMPOSITION (#3192) ===");
    eprintln!("{:<45} {:>10}", "Step", "Actual Metal");
    eprintln!("{}", "-".repeat(57));
    for (name, count) in &step_counts {
        eprintln!("{name:<45} {count:>10}");
    }
    eprintln!("{}", "-".repeat(57));
    eprintln!("{:<45} {:>10}", "TOTAL (sum of per-step)", total_per_step);
    eprintln!();

    // Compare to pipeline-wide measurement.
    // D5.1: use total_encoding_events() (compute + blits) instead of
    // total_metal_dispatches() (plan-expanded kernel count). See #1815.
    let planner_estimate = kokoro.total_encoding_events();
    nn_metal::reset_counters();
    let (_audio2, _cert2, pipeline_stats) = kokoro
        .synthesize_with_stats(&input_ids, &style, 1.0, &cache)
        .unwrap();
    let pipeline_actual = pipeline_stats.compute_encodings + pipeline_stats.blits;

    eprintln!("Pipeline-wide:  estimated={planner_estimate}, actual={pipeline_actual} (compute={}, blits={})",
        pipeline_stats.compute_encodings, pipeline_stats.blits);
    eprintln!("Per-step total: {total_per_step}");
    eprintln!(
        "Accounting gap: {} (pipeline_actual - per_step_total)",
        pipeline_actual as isize - total_per_step as isize
    );
    eprintln!(
        "Eager overhead: {} (actual - estimated)",
        pipeline_actual.saturating_sub(planner_estimate)
    );
    eprintln!("================================================\n");

    // Sanity check: per-step compute total should approximately equal pipeline compute.
    // Per-step measurements only count compute_encodings (not blits), so compare
    // against pipeline compute-only. D5.1: prior code compared per-step compute
    // against pipeline total (compute + blits) — apples-to-oranges.
    let pipeline_compute = pipeline_stats.compute_encodings;
    let gap = (pipeline_compute as isize - total_per_step as isize).unsigned_abs();
    assert!(
        gap < 20,
        "Per-step total ({total_per_step}) differs from pipeline compute ({pipeline_compute}) \
         by {gap} dispatches — accounting mismatch. Expected < 20.",
    );
}
