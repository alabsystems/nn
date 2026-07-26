// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CROWN-verified per-phoneme acoustic feature bounds.
//!
//! Extends the [`DurationPositivityCertificate`](crate::monotonicity::DurationPositivityCertificate)
//! pattern to per-phoneme F0 and energy features. The key insight: Kokoro's
//! `F0EnergyPredictor` outputs `[f0..., energy...]` at phoneme resolution.
//! CROWN bounds on this output give per-phoneme F0/energy bounds.
//!
//! # Interpretation
//!
//! Output bounds are in the model's internal representation. For F0 in Hz,
//! apply the denormalization function: `f0_hz = exp(f0_logits) * base_freq`.
//! Default `base_freq = 200 Hz` (Kokoro convention).
//!
//! # Usage
//!
//! ```rust,ignore
//! use nn_tts_verify::phoneme_crown::{
//!     PhonemeFeatureCertificate, interpret_phoneme_features,
//! };
//!
//! let cert = interpret_phoneme_features(
//!     &output_lower,  // from CROWN propagation
//!     &output_upper,
//!     2,              // seq_len (number of phonemes)
//!     1.0, 1.0,       // input_bound, style_bound
//!     200.0,          // f0_base_freq_hz
//!     "CROWN",
//! );
//! assert!(cert.is_proven);
//! ```
//!
//! Part of #1737: Certified Pronunciation and Phoneme Realization.

use crate::error::{InvalidConfigKind, TtsVerifyError};

/// CROWN-proven bounds on per-phoneme acoustic features (F0 and energy).
///
/// Each phoneme gets a proven F0 range (in Hz) and energy range. These bounds
/// hold for all inputs within the specified perturbation regions.
///
/// `is_proven` is true when all bounds are finite (non-vacuous). Vacuous bounds
/// (±infinity from IBP over-approximation) indicate that the CROWN propagation
/// failed to produce meaningful bounds for some phonemes.
#[derive(Debug, Clone)]
pub struct PhonemeFeatureCertificate {
    /// Per-phoneme F0 lower bound (Hz) — CROWN-proven.
    pub f0_lower_hz: Vec<f64>,
    /// Per-phoneme F0 upper bound (Hz) — CROWN-proven.
    pub f0_upper_hz: Vec<f64>,
    /// Per-phoneme energy lower bound — CROWN-proven.
    pub energy_lower: Vec<f64>,
    /// Per-phoneme energy upper bound — CROWN-proven.
    pub energy_upper: Vec<f64>,
    /// Text feature input bound (symmetric: `[-bound, +bound]`).
    pub input_bound: f64,
    /// Style embedding input bound (symmetric: `[-bound, +bound]`).
    pub style_bound: f64,
    /// Sequence length (number of phonemes).
    pub sequence_length: usize,
    /// Propagation mode used (`"IBP"`, `"CROWN"`, or `"alpha-CROWN"`).
    pub propagation_mode: String,
    /// Are all bounds finite and non-vacuous?
    pub is_proven: bool,
}

/// Interpret CROWN output bounds as per-phoneme F0/energy certificates.
///
/// The output layout follows the F0EnergyPredictor convention:
/// `[f0_0, f0_1, ..., f0_{T-1}, energy_0, energy_1, ..., energy_{T-1}]`
/// where `T = sequence_length`.
///
/// F0 values are in log-space (model output). Conversion to Hz uses:
/// `f0_hz = exp(f0_logit) * base_freq_hz`.
///
/// # Arguments
///
/// * `output_lower` — per-element lower bounds from CROWN (length = 2 * seq_len)
/// * `output_upper` — per-element upper bounds from CROWN (length = 2 * seq_len)
/// * `sequence_length` — number of phonemes (T)
/// * `input_bound` — symmetric bound on text features (`[-B, B]`)
/// * `style_bound` — symmetric bound on style embedding (`[-B, B]`)
/// * `f0_base_freq_hz` — base frequency for F0 denormalization (default 200.0 Hz)
/// * `propagation_mode` — `"IBP"`, `"CROWN"`, etc.
///
/// # Errors
///
/// Returns [`TtsVerifyError::InvalidConfig`] if:
/// - `sequence_length` is zero
/// - output bounds length doesn't match `2 * sequence_length`
/// - `f0_base_freq_hz` is non-positive or non-finite
pub fn interpret_phoneme_features(
    output_lower: &[f32],
    output_upper: &[f32],
    sequence_length: usize,
    input_bound: f64,
    style_bound: f64,
    f0_base_freq_hz: f64,
    propagation_mode: &str,
) -> Result<PhonemeFeatureCertificate, TtsVerifyError> {
    if sequence_length == 0 {
        return Err(TtsVerifyError::InvalidConfig(
            InvalidConfigKind::NonPositive {
                param: "sequence_length",
            },
        ));
    }

    let expected_len = 2 * sequence_length;
    if output_lower.len() != expected_len || output_upper.len() != expected_len {
        return Err(TtsVerifyError::InvalidConfig(
            InvalidConfigKind::Constraint {
                what: "output bounds length must equal 2 * sequence_length",
            },
        ));
    }

    if !f0_base_freq_hz.is_finite() || f0_base_freq_hz <= 0.0 {
        return Err(TtsVerifyError::InvalidConfig(
            InvalidConfigKind::NonPositive {
                param: "f0_base_freq_hz",
            },
        ));
    }

    // Split output bounds: [0..T) = F0 logits, [T..2T) = energy
    let f0_lo_logits = &output_lower[..sequence_length];
    let f0_hi_logits = &output_upper[..sequence_length];
    let energy_lo = &output_lower[sequence_length..];
    let energy_hi = &output_upper[sequence_length..];

    // Convert F0 from log-space to Hz: f0_hz = exp(logit) * base_freq
    let mut f0_lower_hz = Vec::with_capacity(sequence_length);
    let mut f0_upper_hz = Vec::with_capacity(sequence_length);
    let mut all_finite = true;

    for i in 0..sequence_length {
        let lo = f64::from(f0_lo_logits[i]);
        let hi = f64::from(f0_hi_logits[i]);

        if !lo.is_finite() || !hi.is_finite() {
            all_finite = false;
        }

        // exp is monotonically increasing, so bounds are preserved
        let lo_hz = lo.exp() * f0_base_freq_hz;
        let hi_hz = hi.exp() * f0_base_freq_hz;
        f0_lower_hz.push(lo_hz);
        f0_upper_hz.push(hi_hz);
    }

    let energy_lower: Vec<f64> = energy_lo.iter().map(|&v| f64::from(v)).collect();
    let energy_upper: Vec<f64> = energy_hi.iter().map(|&v| f64::from(v)).collect();

    // Check energy bounds are finite too
    for (lo, hi) in energy_lower.iter().zip(energy_upper.iter()) {
        if !lo.is_finite() || !hi.is_finite() {
            all_finite = false;
        }
    }

    // Check Hz bounds are finite (exp can overflow for large logits)
    for (lo, hi) in f0_lower_hz.iter().zip(f0_upper_hz.iter()) {
        if !lo.is_finite() || !hi.is_finite() {
            all_finite = false;
        }
    }

    Ok(PhonemeFeatureCertificate {
        f0_lower_hz,
        f0_upper_hz,
        energy_lower,
        energy_upper,
        input_bound,
        style_bound,
        sequence_length,
        propagation_mode: propagation_mode.to_string(),
        is_proven: all_finite,
    })
}

/// Compute the F0 range width (Hz) for a given phoneme from the certificate.
///
/// Returns `None` if the phoneme index is out of range.
pub fn f0_range_hz(cert: &PhonemeFeatureCertificate, phoneme_idx: usize) -> Option<f64> {
    if phoneme_idx >= cert.sequence_length {
        return None;
    }
    Some(cert.f0_upper_hz[phoneme_idx] - cert.f0_lower_hz[phoneme_idx])
}

/// Compute the energy range width for a given phoneme from the certificate.
///
/// Returns `None` if the phoneme index is out of range.
pub fn energy_range(cert: &PhonemeFeatureCertificate, phoneme_idx: usize) -> Option<f64> {
    if phoneme_idx >= cert.sequence_length {
        return None;
    }
    Some(cert.energy_upper[phoneme_idx] - cert.energy_lower[phoneme_idx])
}

/// Maximum F0 range across all phonemes (Hz). Larger = more variation allowed.
pub fn max_f0_range_hz(cert: &PhonemeFeatureCertificate) -> f64 {
    crate::stats::fold_max_propagate_nan(
        (0..cert.sequence_length).filter_map(|i| f0_range_hz(cert, i)),
        0.0_f64,
    )
}

/// Maximum energy range across all phonemes. Larger = more variation allowed.
pub fn max_energy_range(cert: &PhonemeFeatureCertificate) -> f64 {
    crate::stats::fold_max_propagate_nan(
        (0..cert.sequence_length).filter_map(|i| energy_range(cert, i)),
        0.0_f64,
    )
}

#[cfg(test)]
#[path = "phoneme_crown_tests.rs"]
mod tests;
