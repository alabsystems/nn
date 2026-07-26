// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Audio quality metric scalar functions: cosine similarity, SNR, SDR.
//!
//! These are the pure mathematical functions underlying dvoice V1/V2 gate
//! pass conditions. Each uses f64 accumulators for numerical stability and
//! handles edge cases (zero vectors, perfect reconstruction, silent reference).
//!
//! Kani proofs verify: finite output for finite input, correct output ranges,
//! division-by-zero guards, and normalization correctness.

use crate::error::TtsVerifyError;

/// Compute cosine similarity between two equal-length f32 slices.
///
/// Returns a value in [-1.0, 1.0]. Returns 0.0 if either vector is zero.
/// Uses f64 accumulators for numerical stability.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f64, TtsVerifyError> {
    if a.is_empty() || b.is_empty() {
        return Err(TtsVerifyError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(TtsVerifyError::LengthMismatch {
            candidate: a.len(),
            reference: b.len(),
        });
    }

    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;

    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = f64::from(x);
        let yf = f64::from(y);
        dot += xf * yf;
        norm_a += xf * xf;
        norm_b += yf * yf;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        return Ok(0.0); // Zero-vector → undefined, return 0.
    }

    // Clamp to [-1, 1] to handle floating-point rounding.
    Ok((dot / denom).clamp(-1.0, 1.0))
}

/// Compute Signal-to-Noise Ratio in dB.
///
/// SNR = 10 * log10(signal_power / noise_power)
/// where noise = candidate - reference.
/// Returns +Infinity if noise is zero (perfect reconstruction).
/// Returns 0 dB if reference is silent.
pub fn snr_db(candidate: &[f32], reference: &[f32]) -> Result<f64, TtsVerifyError> {
    if candidate.is_empty() || reference.is_empty() {
        return Err(TtsVerifyError::EmptyInput);
    }
    if candidate.len() != reference.len() {
        return Err(TtsVerifyError::LengthMismatch {
            candidate: candidate.len(),
            reference: reference.len(),
        });
    }

    let mut signal_power = 0.0_f64;
    let mut noise_power = 0.0_f64;

    for (&c, &r) in candidate.iter().zip(reference.iter()) {
        let rf = f64::from(r);
        let diff = f64::from(c) - rf;
        signal_power += rf * rf;
        noise_power += diff * diff;
    }

    if signal_power == 0.0 {
        return Ok(0.0); // Silent reference → SNR is 0 dB.
    }
    if noise_power == 0.0 {
        return Ok(f64::INFINITY); // Perfect reconstruction.
    }

    Ok(10.0 * (signal_power / noise_power).log10())
}

/// Compute Signal-to-Distortion Ratio in dB (BSS_EVAL definition).
///
/// SDR = 10 * log10(||s_target||^2 / ||e_total||^2)
/// where s_target is the orthogonal projection of candidate onto reference,
/// and e_total = candidate - s_target.
///
/// Citation: Vincent et al. 2006, "Performance measurement in blind audio
/// source separation", IEEE TASLP.
pub fn sdr_db(candidate: &[f32], reference: &[f32]) -> Result<f64, TtsVerifyError> {
    if candidate.is_empty() || reference.is_empty() {
        return Err(TtsVerifyError::EmptyInput);
    }
    if candidate.len() != reference.len() {
        return Err(TtsVerifyError::LengthMismatch {
            candidate: candidate.len(),
            reference: reference.len(),
        });
    }

    // s_target = <candidate, reference> / <reference, reference> * reference
    let mut dot_cr = 0.0_f64;
    let mut dot_rr = 0.0_f64;

    for (&c, &r) in candidate.iter().zip(reference.iter()) {
        let cf = f64::from(c);
        let rf = f64::from(r);
        dot_cr += cf * rf;
        dot_rr += rf * rf;
    }

    if dot_rr == 0.0 {
        return Ok(0.0); // Silent reference.
    }

    let scale = dot_cr / dot_rr;

    // s_target_power = scale^2 * ||reference||^2
    let s_target_power = scale * scale * dot_rr;

    // e_total = candidate - scale * reference
    let mut e_power = 0.0_f64;
    for (&c, &r) in candidate.iter().zip(reference.iter()) {
        let residual = f64::from(c) - scale * f64::from(r);
        e_power += residual * residual;
    }

    if e_power == 0.0 {
        return Ok(f64::INFINITY); // Perfect reconstruction.
    }

    Ok(10.0 * (s_target_power / e_power).log10())
}

// -- Scalar helpers for Kani verification and tests --------------------------
// Used by #[cfg(kani)] proofs in dsp_kani_proofs.rs and #[cfg(test)] in
// dsp_quality_metrics_tests.rs. Not called from production code paths.

#[allow(dead_code)]
pub(crate) fn cosine_similarity_scalar(a: f32, b: f32) -> f64 {
    let af = f64::from(a);
    let bf = f64::from(b);
    let dot = af * bf;
    let denom = (af * af * bf * bf).sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    (dot / denom).clamp(-1.0, 1.0)
}

#[allow(dead_code)]
pub(crate) fn snr_scalar(signal: f32, noise: f32) -> f64 {
    let sig_power = f64::from(signal) * f64::from(signal);
    let noise_power = f64::from(noise) * f64::from(noise);
    if sig_power == 0.0 {
        return 0.0;
    }
    if noise_power == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (sig_power / noise_power).log10()
}

#[allow(dead_code)]
pub(crate) fn rms_scalar(x: f32) -> f64 {
    let xf = f64::from(x);
    (xf * xf).sqrt()
}

#[allow(dead_code)]
pub(crate) fn power_to_db(power: f64) -> f64 {
    10.0 * power.log10()
}

// Used by dsp_kani_proofs.rs and dsp_quality_metrics_tests.rs
#[allow(unused_imports)]
pub(crate) use nn_core::audio::hz_to_mel_htk as hz_to_mel;
#[allow(unused_imports)]
pub(crate) use nn_core::audio::mel_to_hz_htk as mel_to_hz;

#[cfg(test)]
#[path = "dsp_quality_metrics_tests.rs"]
mod tests;
