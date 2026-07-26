// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Biquad IIR filter kernel — Direct Form II Transposed.
//!
//! Implements the fundamental audio DSP building block per the Audio EQ Cookbook
//! (Robert Bristow-Johnson). Three filter types: peaking EQ, high-shelf, and
//! bandpass. Coefficient constructors validate parameters and normalize by a0.
//!
//! Part of #956 (Audio DSP kernel support).

use crate::kernel_error::KernelError;
use crate::kernel_util::{checked_scalar_output, validate_finite_inputs};

/// Minimum Q value to prevent near-zero denominators in coefficient computation.
pub const BIQUAD_MIN_Q: f32 = 0.001;

/// Biquad filter coefficients (normalized so a0 = 1.0).
///
/// Feedforward: `b0, b1, b2`. Feedback: `a1, a2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiquadCoeffs {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
}

/// Output of a single biquad `process_sample` step: sample output + updated state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiquadSampleOutput {
    pub y: f32,
    pub z1: f32,
    pub z2: f32,
}

impl BiquadCoeffs {
    /// Check BIBO stability via Jury/Schur-Cohn conditions.
    ///
    /// A second-order IIR filter is stable iff all three hold:
    /// - `|a2| < 1`
    /// - `1 + a1 + a2 > 0`
    /// - `1 - a1 + a2 > 0`
    #[must_use]
    pub fn is_stable(&self) -> bool {
        self.a2.abs() < 1.0 && (1.0 + self.a1 + self.a2) > 0.0 && (1.0 - self.a1 + self.a2) > 0.0
    }

    /// Check if coefficients produce an identity (passthrough) filter
    /// within the given tolerance.
    ///
    /// Identity: `b = [1, 0, 0]`, `a = [0, 0]`.
    #[must_use]
    pub fn is_identity(&self, tol: f32) -> bool {
        (self.b0 - 1.0).abs() < tol
            && self.b1.abs() < tol
            && self.b2.abs() < tol
            && self.a1.abs() < tol
            && self.a2.abs() < tol
    }

    /// Check if the filter is transparent: `H(z) = 1` for all `z`.
    ///
    /// A filter is transparent when `b0 = 1` and `b1 = a1` and `b2 = a2`,
    /// meaning numerator equals denominator. This is the general form —
    /// `is_identity` is the special case where `a1 = a2 = 0`.
    ///
    /// 0 dB peaking EQ produces a transparent filter with non-zero a1, a2.
    #[must_use]
    pub fn is_transparent(&self, tol: f32) -> bool {
        (self.b0 - 1.0).abs() < tol
            && (self.b1 - self.a1).abs() < tol
            && (self.b2 - self.a2).abs() < tol
    }

    /// DC gain: `H(z=1) = (b0 + b1 + b2) / (1 + a1 + a2)`.
    ///
    /// Returns `None` if the denominator is zero (marginally stable at DC).
    #[must_use]
    pub fn dc_gain(&self) -> Option<f32> {
        let denom = 1.0 + self.a1 + self.a2;
        if denom.abs() < f32::EPSILON {
            return None;
        }
        let gain = (self.b0 + self.b1 + self.b2) / denom;
        if gain.is_finite() {
            Some(gain)
        } else {
            None
        }
    }

    /// Nyquist gain: `H(z=-1) = (b0 - b1 + b2) / (1 - a1 + a2)`.
    ///
    /// Returns `None` if the denominator is zero.
    #[must_use]
    pub fn nyquist_gain(&self) -> Option<f32> {
        let denom = 1.0 - self.a1 + self.a2;
        if denom.abs() < f32::EPSILON {
            return None;
        }
        let gain = (self.b0 - self.b1 + self.b2) / denom;
        if gain.is_finite() {
            Some(gain)
        } else {
            None
        }
    }
}

/// Compute peaking EQ biquad coefficients.
///
/// Boost/cut at `freq` with bandwidth `Q`. Audio EQ Cookbook formula.
///
/// # Errors
///
/// Returns [`KernelError`] if parameters are non-finite or out of range.
pub fn biquad_peaking(
    sample_rate: f32,
    freq: f32,
    gain_db: f32,
    q: f32,
) -> Result<BiquadCoeffs, KernelError> {
    validate_finite_inputs(&[
        ("sample_rate", sample_rate),
        ("freq", freq),
        ("gain_db", gain_db),
        ("q", q),
    ])?;
    validate_biquad_params(sample_rate, freq, q)?;

    let a = 10.0_f32.powf(gain_db / 40.0);
    let w0 = 2.0 * std::f32::consts::PI * freq / sample_rate;
    let safe_q = q.max(BIQUAD_MIN_Q);
    let sin_w0 = w0.sin();
    let cos_w0 = w0.cos();
    let alpha = sin_w0 / (2.0 * safe_q);

    let b0 = 1.0 + alpha * a;
    let b1 = -2.0 * cos_w0;
    let b2 = 1.0 - alpha * a;
    let a0 = 1.0 + alpha / a;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha / a;

    normalize_coeffs(b0, b1, b2, a0, a1, a2)
}

/// Compute high-shelf biquad coefficients.
///
/// Boost/cut above `freq`. Audio EQ Cookbook formula.
///
/// # Errors
///
/// Returns [`KernelError`] if parameters are non-finite or out of range.
pub fn biquad_high_shelf(
    sample_rate: f32,
    freq: f32,
    gain_db: f32,
    q: f32,
) -> Result<BiquadCoeffs, KernelError> {
    validate_finite_inputs(&[
        ("sample_rate", sample_rate),
        ("freq", freq),
        ("gain_db", gain_db),
        ("q", q),
    ])?;
    validate_biquad_params(sample_rate, freq, q)?;

    let a = 10.0_f32.powf(gain_db / 40.0);
    let w0 = 2.0 * std::f32::consts::PI * freq / sample_rate;
    let safe_q = q.max(BIQUAD_MIN_Q);
    let sin_w0 = w0.sin();
    let cos_w0 = w0.cos();
    let alpha = sin_w0 / (2.0 * safe_q);
    let sqrt_a = a.sqrt();
    let two_sqrt_a_alpha = 2.0 * sqrt_a * alpha;

    let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
    let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
    let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
    let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
    let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
    let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha;

    normalize_coeffs(b0, b1, b2, a0, a1, a2)
}

/// Compute bandpass biquad coefficients.
///
/// Pass frequencies near `freq` with bandwidth `Q`. Audio EQ Cookbook formula.
///
/// # Errors
///
/// Returns [`KernelError`] if parameters are non-finite or out of range.
pub fn biquad_bandpass(sample_rate: f32, freq: f32, q: f32) -> Result<BiquadCoeffs, KernelError> {
    validate_finite_inputs(&[("sample_rate", sample_rate), ("freq", freq), ("q", q)])?;
    validate_biquad_params(sample_rate, freq, q)?;

    let w0 = 2.0 * std::f32::consts::PI * freq / sample_rate;
    let safe_q = q.max(BIQUAD_MIN_Q);
    let sin_w0 = w0.sin();
    let cos_w0 = w0.cos();
    let alpha = sin_w0 / (2.0 * safe_q);

    let b0 = alpha;
    let b1 = 0.0;
    let b2 = -alpha;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;

    normalize_coeffs(b0, b1, b2, a0, a1, a2)
}

/// Process one sample through a biquad filter (Direct Form II Transposed).
///
/// Pure function: takes current state `(z1, z2)`, returns output and new state.
///
/// ```text
/// y  = b0 * x + z1
/// z1 = b1 * x - a1 * y + z2
/// z2 = b2 * x - a2 * y
/// ```
///
/// # Errors
///
/// Returns [`KernelError`] if any input or output is non-finite.
pub fn biquad_process_sample_scalar(
    x: f32,
    coeffs: &BiquadCoeffs,
    z1: f32,
    z2: f32,
) -> Result<BiquadSampleOutput, KernelError> {
    validate_finite_inputs(&[
        ("x", x),
        ("b0", coeffs.b0),
        ("b1", coeffs.b1),
        ("b2", coeffs.b2),
        ("a1", coeffs.a1),
        ("a2", coeffs.a2),
        ("z1", z1),
        ("z2", z2),
    ])?;

    let y = checked_scalar_output(coeffs.b0 * x + z1)?;
    let new_z1 = checked_scalar_output(coeffs.b1 * x - coeffs.a1 * y + z2)?;
    let new_z2 = checked_scalar_output(coeffs.b2 * x - coeffs.a2 * y)?;

    Ok(BiquadSampleOutput {
        y,
        z1: new_z1,
        z2: new_z2,
    })
}

// --- Internal helpers ---

fn validate_biquad_params(sample_rate: f32, freq: f32, q: f32) -> Result<(), KernelError> {
    if sample_rate <= 0.0 {
        return Err(KernelError::InvalidParam {
            name: "sample_rate",
            constraint: "strictly positive",
            value: sample_rate,
        });
    }
    let nyquist = sample_rate / 2.0;
    if freq <= 0.0 || freq >= nyquist {
        return Err(KernelError::InvalidParam {
            name: "freq",
            constraint: "in (0, Nyquist)",
            value: freq,
        });
    }
    if q <= 0.0 {
        return Err(KernelError::InvalidParam {
            name: "q",
            constraint: "strictly positive",
            value: q,
        });
    }
    Ok(())
}

fn normalize_coeffs(
    b0: f32,
    b1: f32,
    b2: f32,
    a0: f32,
    a1: f32,
    a2: f32,
) -> Result<BiquadCoeffs, KernelError> {
    if !a0.is_finite() || a0.abs() < f32::EPSILON {
        return Err(KernelError::NonFiniteOutput {
            name: "a0",
            value: a0,
        });
    }
    let inv_a0 = 1.0 / a0;
    let coeffs = BiquadCoeffs {
        b0: checked_scalar_output(b0 * inv_a0)?,
        b1: checked_scalar_output(b1 * inv_a0)?,
        b2: checked_scalar_output(b2 * inv_a0)?,
        a1: checked_scalar_output(a1 * inv_a0)?,
        a2: checked_scalar_output(a2 * inv_a0)?,
    };
    Ok(coeffs)
}

#[cfg(test)]
#[path = "biquad_tests.rs"]
mod tests;

#[cfg(kani)]
#[path = "biquad_kani.rs"]
mod kani_proofs;
