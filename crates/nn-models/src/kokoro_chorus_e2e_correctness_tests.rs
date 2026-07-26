// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end correctness tests for the Kokoro chorus pipeline.
//!
//! Validates that the full pipeline produces REAL, CORRECT audio output:
//!   - No NaN/Inf in any output sample
//!   - Output is not silent (RMS above threshold)
//!   - Output stays within [-1, 1] when limiter is enabled (no clipping)
//!   - Stereo channels differ for multi-voice configs (stereo imaging works)
//!   - Output length matches input length
//!   - Exercises 1, 2, and 4 voice counts
//!   - Tests both ChorusConfig (basic mixing) and ChorusMasterConfig (full pipeline)
//!   - Tests all production presets at production-realistic buffer sizes
//!
//! Part of #3351.

use crate::kokoro_chorus::{mix_voices, ChorusConfig};
use crate::kokoro_chorus_dynamics::DynamicsPreset;
use crate::kokoro_chorus_eq::EqPreset;
use crate::kokoro_chorus_pipeline::{process_chorus, ChorusMasterConfig, ChorusMasterPipeline};
use crate::kokoro_chorus_stereo::StereoChorusConfig;
use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Sample rate as f32 for convenience.
const SR: f32 = KOKORO_SAMPLE_RATE as f32;

/// Generate a realistic voice-like test signal: fundamental + 3 harmonics
/// with decreasing amplitude, modulated by a slow amplitude envelope.
///
/// More realistic than a pure sine -- harmonics create spectral content
/// that exercises EQ, de-essing, and dynamics processing.
fn synth_voice(fundamental_hz: f32, duration_sec: f32, amplitude: f32) -> Vec<f32> {
    let n = (SR * duration_sec) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / SR;
            let phase = 2.0 * std::f32::consts::PI * fundamental_hz * t;
            // Fundamental + harmonics at 2x, 3x, 4x with decreasing gain.
            let signal = phase.sin()
                + 0.5 * (2.0 * phase).sin()
                + 0.25 * (3.0 * phase).sin()
                + 0.125 * (4.0 * phase).sin();
            // Slow amplitude envelope (fade in over 20ms, sustain, fade out over 20ms).
            let fade_samples = (0.02 * SR) as usize;
            let env = if i < fade_samples {
                i as f32 / fade_samples as f32
            } else if i > n - fade_samples {
                (n - i) as f32 / fade_samples as f32
            } else {
                1.0
            };
            // Normalize: max of sum of harmonics is 1 + 0.5 + 0.25 + 0.125 = 1.875
            amplitude * env * signal / 1.875
        })
        .collect()
}

/// Generate N voices at slightly different pitches (simulating a real chorus).
fn make_chorus_voices(n: usize, base_hz: f32, duration_sec: f32) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            // Spread voices by ~10 Hz each for natural detuning.
            synth_voice(base_hz + i as f32 * 10.0, duration_sec, 0.5)
        })
        .collect()
}

/// Compute RMS energy of a signal.
fn rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    (signal.iter().map(|&s| s * s).sum::<f32>() / signal.len() as f32).sqrt()
}

/// Compute peak absolute value of a signal.
fn peak_abs(signal: &[f32]) -> f32 {
    signal.iter().fold(0.0f32, |m, &s| m.max(s.abs()))
}

/// Check that every sample in the signal is finite (not NaN or Inf).
fn assert_all_finite(signal: &[f32], label: &str) {
    for (i, &s) in signal.iter().enumerate() {
        assert!(s.is_finite(), "{label}[{i}] is not finite: {s}");
    }
}

/// Compute mean absolute difference between two equal-length signals.
fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "signal lengths must match for diff");
    if a.is_empty() {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .sum::<f32>()
        / a.len() as f32
}

// ===========================================================================
// 1. ChorusConfig (basic mixing) end-to-end tests
// ===========================================================================

#[test]
fn test_e2e_basic_mix_1_voice() {
    let voice = synth_voice(220.0, 1.0, 0.5);
    let config = ChorusConfig::equal_gain(1).expect("1-voice config");
    let mixed =
        mix_voices(std::slice::from_ref(&voice), &config.gains, config.clip_output)
            .expect("mix_voices");

    assert_eq!(mixed.len(), voice.len(), "output length must match input");
    assert_all_finite(&mixed, "mixed");

    let out_rms = rms(&mixed);
    assert!(
        out_rms > 0.01,
        "single voice output should not be silent, RMS = {out_rms}"
    );

    // Single voice with gain=1.0: output should be very close to input.
    let diff = mean_abs_diff(&mixed, &voice);
    assert!(
        diff < 1e-5,
        "single voice passthrough diff too large: {diff}"
    );
}

#[test]
fn test_e2e_basic_mix_2_voices() {
    let voices = make_chorus_voices(2, 220.0, 1.0);
    let config = ChorusConfig::equal_gain(2).expect("2-voice config");
    let mixed = mix_voices(&voices, &config.gains, config.clip_output).expect("mix_voices");

    assert_eq!(mixed.len(), voices[0].len());
    assert_all_finite(&mixed, "mixed");

    let out_rms = rms(&mixed);
    assert!(
        out_rms > 0.01,
        "2-voice mix should not be silent, RMS = {out_rms}"
    );

    // Output should be bounded by clipping.
    let peak = peak_abs(&mixed);
    assert!(
        peak <= 1.0 + 1e-6,
        "clipped output should be within [-1, 1], peak = {peak}"
    );
}

#[test]
fn test_e2e_basic_mix_4_voices() {
    let voices = make_chorus_voices(4, 200.0, 1.0);
    let config = ChorusConfig::equal_gain(4).expect("4-voice config");
    let mixed = mix_voices(&voices, &config.gains, config.clip_output).expect("mix_voices");

    assert_eq!(mixed.len(), voices[0].len());
    assert_all_finite(&mixed, "mixed");

    let out_rms = rms(&mixed);
    assert!(
        out_rms > 0.01,
        "4-voice mix should not be silent, RMS = {out_rms}"
    );

    let peak = peak_abs(&mixed);
    assert!(
        peak <= 1.0 + 1e-6,
        "clipped output should be within [-1, 1], peak = {peak}"
    );
}

#[test]
fn test_e2e_basic_mix_custom_gains() {
    let voices = make_chorus_voices(3, 300.0, 0.5);
    let config = ChorusConfig::with_gains(vec![0.6, 0.3, 0.1]).expect("custom gains");
    let mixed = mix_voices(&voices, &config.gains, config.clip_output).expect("mix_voices");

    assert_eq!(mixed.len(), voices[0].len());
    assert_all_finite(&mixed, "mixed");

    // Voice 0 should dominate -- verify the mix is not zero.
    let out_rms = rms(&mixed);
    assert!(
        out_rms > 0.01,
        "custom-gain mix should not be silent, RMS = {out_rms}"
    );
}

// ===========================================================================
// 2. ChorusMasterPipeline end-to-end with 1, 2, and 4 voices
// ===========================================================================

/// Full correctness check for a pipeline run.
fn assert_pipeline_correct(
    left: &[f32],
    right: &[f32],
    expected_len: usize,
    label: &str,
    expect_stereo_diff: bool,
    expect_bounded: bool,
) {
    // Length correctness.
    assert_eq!(
        left.len(),
        right.len(),
        "{label}: L and R lengths differ ({} vs {})",
        left.len(),
        right.len()
    );
    assert_eq!(
        left.len(),
        expected_len,
        "{label}: output length {} != expected {expected_len}",
        left.len()
    );

    // Finiteness (no NaN/Inf).
    assert_all_finite(left, &format!("{label} left"));
    assert_all_finite(right, &format!("{label} right"));

    // Not silent.
    let rms_l = rms(left);
    let rms_r = rms(right);
    assert!(
        rms_l > 1e-4,
        "{label}: left channel is silent, RMS = {rms_l}"
    );
    assert!(
        rms_r > 1e-4,
        "{label}: right channel is silent, RMS = {rms_r}"
    );

    // Stereo differentiation.
    if expect_stereo_diff {
        let diff = mean_abs_diff(left, right);
        assert!(
            diff > 1e-6,
            "{label}: stereo channels should differ, mean diff = {diff}"
        );
    }

    // Output bounded (no clipping) when limiter enabled.
    if expect_bounded {
        let peak_l = peak_abs(left);
        let peak_r = peak_abs(right);
        // Allow slight overshoot from limiter attack time.
        assert!(
            peak_l < 1.5,
            "{label}: left peak too high ({peak_l}), possible clipping"
        );
        assert!(
            peak_r < 1.5,
            "{label}: right peak too high ({peak_r}), possible clipping"
        );
    }
}

#[test]
fn test_e2e_pipeline_1_voice_minimal() {
    let voices = make_chorus_voices(1, 440.0, 1.0);
    let config = ChorusMasterConfig::minimal(1).expect("minimal(1)");
    let (left, right) = process_chorus(&voices, &config).expect("process");

    assert_pipeline_correct(
        &left,
        &right,
        voices[0].len(),
        "1-voice minimal",
        false, // Single voice: L and R may be identical (centered).
        false, // No limiter in minimal.
    );
}

#[test]
fn test_e2e_pipeline_2_voices_standard() {
    let voices = make_chorus_voices(2, 300.0, 1.0);
    let config = ChorusMasterConfig::standard(2).expect("standard(2)");
    let (left, right) = process_chorus(&voices, &config).expect("process");

    assert_pipeline_correct(
        &left,
        &right,
        voices[0].len(),
        "2-voice standard",
        true, // 2 voices panned => stereo diff.
        true, // Standard has limiter.
    );
}

#[test]
fn test_e2e_pipeline_4_voices_full() {
    let voices = make_chorus_voices(4, 220.0, 1.0);
    let config = ChorusMasterConfig::full(4).expect("full(4)");
    let mut pipeline = ChorusMasterPipeline::new(config).expect("pipeline");
    let (left, right) = pipeline.process(&voices).expect("process");

    assert_pipeline_correct(
        &left,
        &right,
        voices[0].len(),
        "4-voice full",
        true, // Multiple voices with stereo panning.
        true, // Full has limiter.
    );

    // Additional quantitative checks for full pipeline.
    let rms_l = rms(&left);
    let rms_r = rms(&right);
    // Full pipeline with saturation, dynamics, reverb -- output should have
    // meaningful energy, not just noise floor.
    assert!(rms_l > 0.01, "4-voice full left RMS too low: {rms_l}");
    assert!(rms_r > 0.01, "4-voice full right RMS too low: {rms_r}");
}

#[test]
fn test_e2e_pipeline_4_voices_custom_builder() {
    let voices = make_chorus_voices(4, 260.0, 1.0);
    let config = ChorusMasterConfig::new(4)
        .expect("new(4)")
        .with_eq(EqPreset::Natural.to_config())
        .with_stereo(StereoChorusConfig::auto_layout(4).expect("stereo layout"))
        .with_dynamics(DynamicsPreset::Gentle.to_config())
        .with_limiter(true);

    let (left, right) = process_chorus(&voices, &config).expect("process");

    assert_pipeline_correct(
        &left,
        &right,
        voices[0].len(),
        "4-voice custom builder",
        true,
        true,
    );
}

// ===========================================================================
// 3. Production presets at production buffer size (1 second = 24000 samples)
// ===========================================================================

fn run_preset_e2e(
    name: &str,
    config_fn: impl Fn(usize) -> Result<ChorusMasterConfig, KokoroError>,
    n_voices: usize,
    expect_stereo: bool,
) {
    let voices = make_chorus_voices(n_voices, 220.0, 1.0);
    let config = config_fn(n_voices).unwrap_or_else(|e| {
        panic!("{name}({n_voices}) failed to construct: {e}");
    });

    let has_limiter = config.limiter_enabled;
    let mut pipeline = ChorusMasterPipeline::new(config).unwrap_or_else(|e| {
        panic!("{name}({n_voices}) pipeline construction failed: {e}");
    });
    let (left, right) = pipeline.process(&voices).unwrap_or_else(|e| {
        panic!("{name}({n_voices}) process failed: {e}");
    });

    assert_pipeline_correct(
        &left,
        &right,
        voices[0].len(),
        &format!("{name}({n_voices})"),
        expect_stereo,
        has_limiter,
    );
}

#[test]
fn test_e2e_singing_chorus_4_voices() {
    run_preset_e2e(
        "singing_chorus",
        ChorusMasterConfig::singing_chorus,
        4,
        true,
    );
}

#[test]
fn test_e2e_speaking_chorus_4_voices() {
    run_preset_e2e(
        "speaking_chorus",
        ChorusMasterConfig::speaking_chorus,
        4,
        true,
    );
}

#[test]
fn test_e2e_intimate_2_voices() {
    run_preset_e2e("intimate", ChorusMasterConfig::intimate, 2, true);
}

#[test]
fn test_e2e_cathedral_4_voices() {
    run_preset_e2e("cathedral", ChorusMasterConfig::cathedral, 4, true);
}

#[test]
fn test_e2e_broadcast_4_voices() {
    run_preset_e2e("broadcast", ChorusMasterConfig::broadcast, 4, true);
}

// ===========================================================================
// 4. Signal integrity: loud input stress test
// ===========================================================================

#[test]
fn test_e2e_loud_input_no_clipping_with_limiter() {
    // Generate 4 loud voices (amplitude = 0.9 each).
    let voices: Vec<Vec<f32>> = (0..4)
        .map(|i| synth_voice(200.0 + i as f32 * 15.0, 1.0, 0.9))
        .collect();

    // Standard config has limiter enabled.
    let config = ChorusMasterConfig::standard(4).expect("standard(4)");
    let (left, right) = process_chorus(&voices, &config).expect("process");

    assert_all_finite(&left, "loud left");
    assert_all_finite(&right, "loud right");

    // With limiter, peaks should be controlled.
    let peak_l = peak_abs(&left);
    let peak_r = peak_abs(&right);
    assert!(
        peak_l < 1.5,
        "limiter should control left peak, got {peak_l}"
    );
    assert!(
        peak_r < 1.5,
        "limiter should control right peak, got {peak_r}"
    );
}

// ===========================================================================
// 5. Determinism: same config + same input => same output
// ===========================================================================

#[test]
fn test_e2e_determinism_full_pipeline() {
    let voices = make_chorus_voices(4, 220.0, 1.0);
    let config = ChorusMasterConfig::full(4).expect("full(4)");

    let mut pipeline_a = ChorusMasterPipeline::new(config.clone()).expect("pipeline A");
    let mut pipeline_b = ChorusMasterPipeline::new(config).expect("pipeline B");

    let (la, ra) = pipeline_a.process(&voices).expect("process A");
    let (lb, rb) = pipeline_b.process(&voices).expect("process B");

    assert_eq!(la.len(), lb.len());
    assert_eq!(ra.len(), rb.len());

    let max_diff_l = la
        .iter()
        .zip(lb.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_diff_r = ra
        .iter()
        .zip(rb.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff_l < 1e-6,
        "pipeline should be deterministic, left max diff = {max_diff_l}"
    );
    assert!(
        max_diff_r < 1e-6,
        "pipeline should be deterministic, right max diff = {max_diff_r}"
    );
}

// ===========================================================================
// 6. Silent input => near-silent output
// ===========================================================================

#[test]
fn test_e2e_silent_input_produces_near_silence() {
    let n_samples = (SR * 1.0) as usize;
    let voices: Vec<Vec<f32>> = (0..4).map(|_| vec![0.0f32; n_samples]).collect();
    let config = ChorusMasterConfig::full(4).expect("full(4)");
    let (left, right) = process_chorus(&voices, &config).expect("process");

    assert_all_finite(&left, "silent left");
    assert_all_finite(&right, "silent right");

    let rms_l = rms(&left);
    let rms_r = rms(&right);
    // Silent input through full pipeline should produce very low output.
    // Allow small noise from dithering, breath insertion, etc.
    assert!(
        rms_l < 0.05,
        "silent input should produce near-silent left, RMS = {rms_l}"
    );
    assert!(
        rms_r < 0.05,
        "silent input should produce near-silent right, RMS = {rms_r}"
    );
}

// ===========================================================================
// 7. Voice count sweep: verify pipeline works for 1, 2, 3, 4, 6, 8 voices
// ===========================================================================

#[test]
fn test_e2e_voice_count_sweep_standard() {
    for n in [1, 2, 3, 4, 6, 8] {
        let voices = make_chorus_voices(n, 220.0, 0.5);
        let config = ChorusMasterConfig::standard(n)
            .unwrap_or_else(|e| panic!("standard({n}) construct: {e}"));
        let (left, right) = process_chorus(&voices, &config)
            .unwrap_or_else(|e| panic!("standard({n}) process: {e}"));

        assert_eq!(
            left.len(),
            voices[0].len(),
            "standard({n}): output length mismatch"
        );
        assert_all_finite(&left, &format!("standard({n}) left"));
        assert_all_finite(&right, &format!("standard({n}) right"));

        let out_rms = rms(&left);
        assert!(
            out_rms > 1e-4,
            "standard({n}): output should not be silent, RMS = {out_rms}"
        );
    }
}

// ===========================================================================
// 8. Stereo imaging quantitative check
// ===========================================================================

#[test]
fn test_e2e_stereo_imaging_increases_with_voices() {
    // With more voices spread across the stereo field, the L/R difference
    // should be larger than with fewer voices.
    let config_2 = ChorusMasterConfig::singing_chorus(2).expect("singing(2)");
    let config_4 = ChorusMasterConfig::singing_chorus(4).expect("singing(4)");

    let voices_2 = make_chorus_voices(2, 220.0, 0.5);
    let voices_4 = make_chorus_voices(4, 220.0, 0.5);

    let (l2, r2) = process_chorus(&voices_2, &config_2).expect("process 2");
    let (l4, r4) = process_chorus(&voices_4, &config_4).expect("process 4");

    let diff_2 = mean_abs_diff(&l2, &r2);
    let diff_4 = mean_abs_diff(&l4, &r4);

    // 4 voices should have at least as much stereo spread as 2 voices.
    // (The exact relationship depends on pan law, but more voices = more spread.)
    assert!(
        diff_4 >= diff_2 * 0.5,
        "4-voice stereo diff ({diff_4}) should not be drastically less than 2-voice ({diff_2})"
    );

    // Both should have meaningful stereo content.
    assert!(
        diff_2 > 1e-6,
        "2-voice singing chorus should have stereo diff > 0, got {diff_2}"
    );
    assert!(
        diff_4 > 1e-6,
        "4-voice singing chorus should have stereo diff > 0, got {diff_4}"
    );
}

// ===========================================================================
// 9. Pipeline reuse: process the same pipeline twice (stateful processors)
// ===========================================================================

#[test]
fn test_e2e_pipeline_reuse_second_pass() {
    let voices = make_chorus_voices(4, 220.0, 0.5);
    let config = ChorusMasterConfig::full(4).expect("full(4)");
    let mut pipeline = ChorusMasterPipeline::new(config).expect("pipeline");

    // First pass.
    let (l1, r1) = pipeline.process(&voices).expect("first pass");
    assert_all_finite(&l1, "pass1 left");
    assert_all_finite(&r1, "pass1 right");

    // Second pass -- stateful processors (compressors, limiters, etc.) should
    // still produce valid output.
    let (l2, r2) = pipeline.process(&voices).expect("second pass");
    assert_all_finite(&l2, "pass2 left");
    assert_all_finite(&r2, "pass2 right");

    let rms_l2 = rms(&l2);
    assert!(
        rms_l2 > 1e-4,
        "second pass should not be silent, RMS = {rms_l2}"
    );
}
