// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Numerical audio quality gate — SNR, spectral convergence, max error, cosine.
//!
//! Extends L3 parity with signal-level quality metrics that catch regressions
//! invisible to per-stage correctness tests. Runs the same synthesis path as
//! [`kokoro_l3_parity`] and evaluates the result with `nn-tts-verify` metrics.
//!
//! Gated on environment variables:
//!   - `KOKORO_WEIGHTS`: path to kokoro_v1_0.safetensors model weights
//!   - `KOKORO_REFERENCE`: path to kokoro_reference.safetensors (PyTorch output)
//!
//! Part of #2927, #2218.

use nn_tts_verify::{compute_multi_res_stft, MultiResStftConfig};

// -- Tests --------------------------------------------------------------------

/// SNR gate: Signal-to-Noise Ratio vs PyTorch reference.
///
/// SNR = 10 * log10(signal_power / noise_power) where noise = nn − ref.
/// Higher is better. Threshold: > 40 dB.
///
/// Note: End-to-end SNR may be limited by STFT atan2 phase wrapping (~0.15
/// max error). If this test fails, check #2928 (GPU STFT regression) and
/// the per-stage parity tests for root cause localization.
#[test]
fn test_audio_quality_snr() {
    let (weights, reference) = match super::kokoro_l3_parity::require_paths() {
        Some(p) => p,
        None => return,
    };

    let (nn_audio, ref_audio) =
        super::kokoro_l3_parity::synthesize_and_load_ref(&weights, &reference);

    let snr = nn_tts_verify::quality::compute_snr(&nn_audio, &ref_audio, 40.0)
        .expect("SNR computation failed");

    eprintln!(
        "SNR: {:.2} dB (threshold: > {:.1} dB) — {}",
        snr.value,
        snr.threshold,
        if snr.passed { "PASSED" } else { "FAILED" },
    );
    assert!(
        snr.passed,
        "Audio quality gate FAILED: SNR={:.2} dB (min {:.1})",
        snr.value, snr.threshold,
    );
}

/// Spectral convergence gate: multi-resolution STFT magnitude distance.
///
/// Computes spectral convergence + log spectral distance at FFT sizes
/// [512, 1024, 2048], averaged across resolutions. Lower is better.
/// Threshold: < 1.0 (default).
///
/// Citation: Yamamoto et al. 2020, "Parallel WaveGAN", ICASSP.
#[test]
fn test_audio_quality_spectral_convergence() {
    let (weights, reference) = match super::kokoro_l3_parity::require_paths() {
        Some(p) => p,
        None => return,
    };

    let (nn_audio, ref_audio) =
        super::kokoro_l3_parity::synthesize_and_load_ref(&weights, &reference);

    let mut config = MultiResStftConfig::default();
    config.max_loss = 0.1;

    let stft_loss = compute_multi_res_stft(&nn_audio, &ref_audio, 24000, &config)
        .expect("Multi-res STFT computation failed");

    eprintln!(
        "Multi-res STFT loss: {:.6} (threshold: < {:.2}) — {}",
        stft_loss.value,
        stft_loss.threshold,
        if stft_loss.passed { "PASSED" } else { "FAILED" },
    );
    assert!(
        stft_loss.passed,
        "Audio quality gate FAILED: spectral_loss={:.6} (max {:.2})",
        stft_loss.value, stft_loss.threshold,
    );
}

/// Max absolute error gate.
///
/// End-to-end max |nn − ref|. Threshold: < 1e-3.
/// Note: This is tighter than L3 parity AC1 (informational only, ~0.15).
/// Expected to fail until STFT phase wrapping is resolved (#2928).
#[test]
fn test_audio_quality_max_error() {
    let (weights, reference) = match super::kokoro_l3_parity::require_paths() {
        Some(p) => p,
        None => return,
    };

    let (nn_audio, ref_audio) =
        super::kokoro_l3_parity::synthesize_and_load_ref(&weights, &reference);

    let max_err = nn_audio
        .iter()
        .zip(ref_audio.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    let threshold = 1e-3_f32;

    eprintln!(
        "Max absolute error: {:.6e} (threshold: < {:.0e}) — {}",
        max_err,
        threshold,
        if max_err < threshold {
            "PASSED"
        } else {
            "FAILED"
        },
    );
    assert!(
        max_err < threshold,
        "Audio quality gate FAILED: max_error={max_err:.6e} (max {threshold:.0e})",
    );
}

/// Cosine similarity gate: waveform-level similarity.
///
/// Cosine similarity in [-1, 1], higher is better. Threshold: > 0.999.
/// This is tighter than L3 parity AC3 (> 0.85).
#[test]
fn test_audio_quality_cosine() {
    let (weights, reference) = match super::kokoro_l3_parity::require_paths() {
        Some(p) => p,
        None => return,
    };

    let (nn_audio, ref_audio) =
        super::kokoro_l3_parity::synthesize_and_load_ref(&weights, &reference);

    let cosine = nn_tts_verify::quality::compute_cosine_similarity(&nn_audio, &ref_audio, 0.999)
        .expect("Cosine similarity computation failed");

    eprintln!(
        "Cosine similarity: {:.6} (threshold: > {:.3}) — {}",
        cosine.value,
        cosine.threshold,
        if cosine.passed { "PASSED" } else { "FAILED" },
    );
    assert!(
        cosine.passed,
        "Audio quality gate FAILED: cosine={:.6} (min {:.3})",
        cosine.value, cosine.threshold,
    );
}

/// SDR gate: Signal-to-Distortion Ratio (BSS_EVAL).
///
/// SDR accounts for signal scaling differences via orthogonal projection.
/// More robust than SNR for comparing outputs with slight gain differences.
/// Higher is better. Threshold: > 30 dB.
///
/// Citation: Vincent et al. 2006, IEEE TASLP.
#[test]
fn test_audio_quality_sdr() {
    let (weights, reference) = match super::kokoro_l3_parity::require_paths() {
        Some(p) => p,
        None => return,
    };

    let (nn_audio, ref_audio) =
        super::kokoro_l3_parity::synthesize_and_load_ref(&weights, &reference);

    let sdr = nn_tts_verify::quality::compute_sdr(&nn_audio, &ref_audio, 30.0)
        .expect("SDR computation failed");

    eprintln!(
        "SDR: {:.2} dB (threshold: > {:.1} dB) — {}",
        sdr.value,
        sdr.threshold,
        if sdr.passed { "PASSED" } else { "FAILED" },
    );
    assert!(
        sdr.passed,
        "Audio quality gate FAILED: SDR={:.2} dB (min {:.1})",
        sdr.value, sdr.threshold,
    );
}
