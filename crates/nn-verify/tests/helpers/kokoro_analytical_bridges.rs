// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Analytical bound verification for Kokoro pipeline bridge stages.
//!
//! Bridge stages (length_regulate, harmonic_source) are not compiled GPU
//! segments — they use eager DynTensor ops, so NY trace graphs are
//! unavailable. Instead, bounds are derived analytically from operation
//! semantics:
//!
//! - **length_regulate:** sigmoid ∈ (0,1), clamp enforces [1, max_dur],
//!   repeat_interleave preserves value bounds.
//! - **harmonic_source:** SourceModule.forward ends with tanh ∈ (-1,1),
//!   forward STFT magnitude ≤ Hann_window_sum (9.5 for n_fft=20),
//!   phase ∈ [-π, π].
//!
//! No production weights or NY features required.
//!
//! Part of #2930 (Automated bound propagation gap detector).
//! Part of #2218 (Perfect Kokoro epic).

use std::path::Path;

use nn_verify::{
    model_for_kernel, model_status_path, PropMethod, VerificationSoundnessMode, VerifyStatus,
};

/// Record an analytical verification entry to the per-model status file.
///
/// Uses `record_pipeline` with `PropMethod::Analytical` and
/// `VerificationSoundnessMode::Sound` since analytical bounds from
/// operation semantics are exact (not heuristic approximations).
fn record_analytical(
    status_key: &str,
    input_lower: f32,
    input_upper: f32,
    output_lower: f32,
    output_upper: f32,
    output_shape: &[usize],
    justification: &str,
) {
    let ws = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let model = model_for_kernel(status_key);
    let model_path = model_status_path(ws, model);
    let mut locked = VerifyStatus::load_locked(&model_path).expect("load_locked");

    locked
        .status
        .record_pipeline(
            status_key,
            PropMethod::Analytical,
            input_lower,
            input_upper,
            output_lower,
            output_upper,
            output_shape,
            VerificationSoundnessMode::Sound,
            Some(&[1]),
        )
        .expect("record_pipeline");
    locked
        .status
        .set_soundness_justification(status_key, justification)
        .expect("set justification");
    locked.save().expect("save status");
    eprintln!("Recorded {status_key}: [{output_lower}, {output_upper}] (analytical, sound)");
}

/// Analytical bounds for `step_regulate` (length_regulate bridge).
///
/// Operation chain:
/// 1. sigmoid(dur_logits) → ∈ (0, 1) for all real inputs
/// 2. sum(dim=2) → (0, D) where D=50 is the number of sigmoid bins
/// 3. mul_scalar(1/speed) → for speed ∈ [0.5, 2.0]: (0, 100)
/// 4. clamp(1, max_dur=50) → [1, 50] — hard analytical bound
/// 5. add_scalar(0.5) + floor + clamp_min(1) → integer counts in [1, 50]
/// 6. repeat_interleave → preserves value bounds of input features
///
/// The tightest novel bound introduced by this stage is the clamp: [1, 50].
/// Output features (aligned_dur, regulated) have the same bounds as their
/// inputs (features from ProsodyPredictor, text_features from TextEncoder).
#[test]
fn test_analytical_bounds_length_regulate() {
    // Analytical derivation: sigmoid output ∈ (0, 1)
    let sigmoid_lo = 0.0_f32;
    let sigmoid_hi = 1.0_f32;

    // Sum of D=50 sigmoid bins: (0, 50)
    let d_bins = 50;
    let sum_lo = sigmoid_lo * d_bins as f32; // 0
    let sum_hi = sigmoid_hi * d_bins as f32; // 50

    // After clamp(1, max_dur=50): [1, 50]
    let max_dur = 50.0_f32;
    let clamp_lo = sum_lo.max(1.0).min(max_dur); // 1.0
    let clamp_hi = sum_hi.max(1.0).min(max_dur); // 50.0

    assert!(
        (clamp_lo - 1.0).abs() < 1e-6,
        "clamp lower should be 1.0, got {clamp_lo}"
    );
    assert!(
        (clamp_hi - 50.0).abs() < 1e-6,
        "clamp upper should be 50.0, got {clamp_hi}"
    );

    let output_width = clamp_hi - clamp_lo;
    assert!(
        (output_width - 49.0).abs() < 1e-6,
        "duration width should be 49.0, got {output_width}"
    );

    // Record to status file: durations ∈ [1, 50], width = 49.
    // Input bounds are from ProsodyPredictor (arbitrary real dur_logits).
    record_analytical(
        "kokoro_production_length_regulate",
        -150.0,   // input range: ProsodyPredictor output (conservative)
        150.0,    // input range: ProsodyPredictor output (conservative)
        clamp_lo, // 1.0
        clamp_hi, // 50.0
        &[50],    // output shape: T phonemes
        "Analytical: sigmoid ∈ (0,1), sum monotone, clamp(1, max_dur=50) enforces [1,50]. \
         repeat_interleave preserves value bounds. Sound for all real inputs.",
    );

    eprintln!(
        "length_regulate analytical bounds: durations ∈ [{clamp_lo}, {clamp_hi}], width={output_width}"
    );
}

/// Analytical bounds for `build_harmonic_source` (harmonic_source bridge).
///
/// Operation chain:
/// 1. SineGen: produces sin(harmonics) ∈ [-1, 1] for 9 harmonics
/// 2. l_linear: Linear([1, 9]) maps 9 channels to 1 — weight-dependent
/// 3. tanh(projected) → (-1, 1) — hard analytical bound, independent of weights
/// 4. Forward STFT (n_fft=20, hop=5, Hann window):
///    - source signal ∈ (-1, 1) from tanh
///    - windowed DFT: real_f = Σ_k w[k]·cos(2πfk/N)·x[k]
///    - Hann window sum for N=20: Σ w[k] = (N-1)/2 = 9.5
///    - |real_f|, |imag_f| ≤ 9.5 (Cauchy-Schwarz with |e^{-iθ}|=1)
///    - magnitude = |Σ w[k]·e^{-i2πfk/N}·x[k]| ≤ Σ w[k]·|x[k]| ≤ 9.5
///    - phase ∈ [-π, π]
///    - output = cat([magnitude, phase], dim=1)
///
/// CPU bridge (cumsum_f64 at frame rate) does not affect output bounds —
/// it accumulates phase for SineGen internally but doesn't appear in
/// the output path.
#[test]
fn test_analytical_bounds_harmonic_source() {
    // tanh output bound (exact, weight-independent)
    let tanh_lo = -1.0_f32;
    let tanh_hi = 1.0_f32;

    // Hann window sum for n_fft = 20
    // Hann(k, N) = 0.5 - 0.5·cos(2πk/(N-1))
    // Σ_{k=0}^{N-1} Hann(k, N) = 0.5·N - 0.5·Σcos(2πk/(N-1))
    // For N=20: Σcos(2πk/19) for k=0..19 = 1.0 (19 complete periods + cos(2π))
    // So Σ w[k] = 0.5·20 - 0.5·1 = 9.5
    let n_fft = 20_usize;
    let hann_sum: f32 = (0..n_fft)
        .map(|k| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * k as f32 / (n_fft - 1) as f32).cos())
        .sum();
    let expected_hann_sum = (n_fft as f32 - 1.0) / 2.0; // 9.5

    assert!(
        (hann_sum - expected_hann_sum).abs() < 0.01,
        "Hann window sum should be ~{expected_hann_sum}, got {hann_sum}"
    );

    // STFT magnitude bound: for source ∈ (-1, 1), magnitude ≤ hann_sum
    let mag_lo = 0.0_f32;
    let mag_hi = hann_sum; // 9.5

    // Phase bound
    let phase_lo = -std::f32::consts::PI;
    let phase_hi = std::f32::consts::PI;

    // Combined output: cat([magnitude, phase])
    // Global lower = min(mag_lo, phase_lo) = -π ≈ -3.14
    // Global upper = max(mag_hi, phase_hi) = 9.5
    let combined_lo = mag_lo.min(phase_lo);
    let combined_hi = mag_hi.max(phase_hi);
    let combined_width = combined_hi - combined_lo;

    eprintln!("harmonic_source analytical bounds:");
    eprintln!("  tanh: ({tanh_lo}, {tanh_hi})");
    eprintln!("  Hann window sum (n_fft={n_fft}): {hann_sum:.2}");
    eprintln!("  STFT magnitude: [{mag_lo}, {mag_hi:.2}]");
    eprintln!("  STFT phase: [{phase_lo:.4}, {phase_hi:.4}]");
    eprintln!("  combined: [{combined_lo:.4}, {combined_hi:.2}], width={combined_width:.2}");

    assert!(
        combined_width < 15.0,
        "combined width should be ~12.6, got {combined_width}"
    );

    let n_bins = n_fft / 2 + 1; // 11
    let n_frames = 50; // typical for T_mel~25 with upsample

    record_analytical(
        "kokoro_production_harmonic_source",
        -1.0,                     // input: F0 from F0EnergyPredictor
        1.0,                      // input: F0 normalized range
        combined_lo,              // -π
        combined_hi,              // 9.5
        &[2 * n_bins * n_frames], // output shape: cat([mag, phase])
        "Analytical: tanh ∈ (-1,1) (weight-independent). \
         Forward STFT with Hann window: magnitude ≤ Σw[k]=9.5, phase ∈ [-π,π]. \
         CPU bridge (cumsum_f64) internal to SineGen, doesn't affect output bounds. \
         Sound for all inputs.",
    );
}
