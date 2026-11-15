// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Reverb filter kernels — comb and allpass filter stability.
//!
//! Implements the two building blocks of Freeverb (Jezar's public-domain
//! reverb algorithm): lowpass-feedback comb filters and Schroeder allpass
//! filters. Both are stateful per-sample kernels whose stability depends
//! on feedback coefficient bounds.
//!
//! Invariants:
//! - Comb: `|feedback| < 1` and `|damp| ≤ 1` → no divergence
//! - Allpass: `|feedback| < 1` → output bounded for bounded input
//!
//! Part of #956 D4 (Audio DSP kernel support).

use crate::kernel_error::KernelError;
use crate::kernel_util::{checked_scalar_output, validate_finite_inputs};

// ---------------------------------------------------------------------------
// Comb filter (lowpass-feedback comb, Freeverb style)
// ---------------------------------------------------------------------------

/// Comb filter configuration (time-invariant).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CombCoeffs {
    /// Feedback gain in `(-1, 1)`. Controls decay time.
    pub feedback: f32,
    /// Damping coefficient in `[0, 1]`. 0 = no damping, 1 = full damping.
    pub damp: f32,
}

/// Comb filter state (time-varying, carried between samples).
///
/// `filterstore` is the one-pole lowpass state applied to the feedback path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CombState {
    /// One-pole lowpass state in the feedback path.
    pub filterstore: f32,
}

/// Output of a single comb filter step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CombOutput {
    /// The value to write back into the delay buffer at the current position.
    pub write_back: f32,
    /// Updated filterstore state.
    pub filterstore: f32,
}

impl CombState {
    /// Initial state: zero filterstore.
    #[must_use]
    pub fn new() -> Self {
        Self { filterstore: 0.0 }
    }
}

impl Default for CombState {
    fn default() -> Self {
        Self::new()
    }
}

/// Process one comb filter sample.
///
/// Freeverb comb filter: reads from delay buffer, applies one-pole lowpass
/// in the feedback path, then writes back into the delay buffer.
///
/// The caller manages the delay buffer; this function computes the feedback
/// path for one sample.
///
/// ```text
/// output = delay_read
/// lp = damp1 * delay_read + damp2 * filterstore   (one-pole lowpass)
/// write_back = lp * feedback + input
/// ```
///
/// where `damp1 = 1 - damp` and `damp2 = damp`.
///
/// # Errors
///
/// Returns [`KernelError`] if any input/output is non-finite.
pub fn comb_process_sample_scalar(
    input: f32,
    delay_read: f32,
    state: &CombState,
    coeffs: &CombCoeffs,
) -> Result<CombOutput, KernelError> {
    validate_finite_inputs(&[
        ("input", input),
        ("delay_read", delay_read),
        ("filterstore", state.filterstore),
        ("feedback", coeffs.feedback),
        ("damp", coeffs.damp),
    ])?;

    // One-pole lowpass in feedback path
    let damp1 = 1.0 - coeffs.damp;
    let damp2 = coeffs.damp;
    let new_filterstore = checked_scalar_output(damp1 * delay_read + damp2 * state.filterstore)?;

    // Feedback + input → write back to delay line
    let write_back = checked_scalar_output(new_filterstore * coeffs.feedback + input)?;

    Ok(CombOutput {
        write_back,
        filterstore: new_filterstore,
    })
}

/// Validate comb filter configuration.
///
/// # Errors
///
/// Returns [`KernelError::InvalidParam`] if parameters are out of range.
pub fn validate_comb_config(coeffs: &CombCoeffs) -> Result<(), KernelError> {
    validate_finite_inputs(&[("feedback", coeffs.feedback), ("damp", coeffs.damp)])?;
    if coeffs.feedback.abs() >= 1.0 {
        return Err(KernelError::InvalidParam {
            name: "feedback",
            constraint: "|feedback| < 1",
            value: coeffs.feedback,
        });
    }
    if coeffs.damp < 0.0 || coeffs.damp > 1.0 {
        return Err(KernelError::InvalidParam {
            name: "damp",
            constraint: "in [0, 1]",
            value: coeffs.damp,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Allpass filter (Schroeder allpass, Freeverb style)
// ---------------------------------------------------------------------------

/// Allpass filter configuration (time-invariant).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AllpassCoeffs {
    /// Feedback coefficient in `(-1, 1)`.
    pub feedback: f32,
}

/// Output of a single allpass filter step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AllpassOutput {
    /// Filter output sample.
    pub y: f32,
    /// Value to write back into the delay buffer.
    pub write_back: f32,
}

/// Process one allpass filter sample.
///
/// Schroeder allpass structure:
/// ```text
/// write_back = input + feedback * delay_read
/// output     = delay_read - feedback * write_back
///            = delay_read - feedback * (input + feedback * delay_read)
///            = delay_read * (1 - feedback²) - feedback * input
/// ```
///
/// The caller manages the delay buffer; this function computes one sample.
///
/// # Errors
///
/// Returns [`KernelError`] if any input/output is non-finite.
pub fn allpass_process_sample_scalar(
    input: f32,
    delay_read: f32,
    coeffs: &AllpassCoeffs,
) -> Result<AllpassOutput, KernelError> {
    validate_finite_inputs(&[
        ("input", input),
        ("delay_read", delay_read),
        ("feedback", coeffs.feedback),
    ])?;

    let write_back = checked_scalar_output(input + coeffs.feedback * delay_read)?;
    let y = checked_scalar_output(delay_read - coeffs.feedback * write_back)?;

    Ok(AllpassOutput { y, write_back })
}

/// Validate allpass filter configuration.
///
/// # Errors
///
/// Returns [`KernelError::InvalidParam`] if feedback is out of range.
pub fn validate_allpass_config(coeffs: &AllpassCoeffs) -> Result<(), KernelError> {
    validate_finite_inputs(&[("feedback", coeffs.feedback)])?;
    if coeffs.feedback.abs() >= 1.0 {
        return Err(KernelError::InvalidParam {
            name: "feedback",
            constraint: "|feedback| < 1",
            value: coeffs.feedback,
        });
    }
    Ok(())
}

#[cfg(test)]
#[path = "reverb_tests.rs"]
mod tests;
