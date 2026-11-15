// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Side-chain ducker kernel — envelope-following gain reduction.
//!
//! Tracks a sidechain signal's envelope and reduces gain when it exceeds
//! a threshold. Invariants: envelope >= 0, gain in [0, 1], never amplifies.
//!
//! Part of #956 D2 (Audio DSP kernel support).

use crate::kernel_error::KernelError;
use crate::kernel_util::{checked_scalar_output, validate_finite_inputs};

/// Ducker configuration (time-invariant).
///
/// Coefficients are one-pole smoothing factors in `(0, 1)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DuckerCoeffs {
    /// Attack smoothing coefficient in `(0, 1)`.
    pub attack_coeff: f32,
    /// Release smoothing coefficient in `(0, 1)`.
    pub release_coeff: f32,
    /// Envelope threshold (positive). Ducking engages above this level.
    pub threshold: f32,
    /// Ducking ratio in `[0, 1)`. Lower = more reduction. 0.2 = 5:1 ducking.
    pub ratio: f32,
}

/// Ducker state (time-varying, carried between samples).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DuckerState {
    pub envelope: f32,
    pub gain: f32,
}

/// Output of a single ducker step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DuckerOutput {
    pub y: f32,
    pub envelope: f32,
    pub gain: f32,
}

impl DuckerState {
    /// Initial state: zero envelope, unity gain.
    #[must_use]
    pub fn new() -> Self {
        Self {
            envelope: 0.0,
            gain: 1.0,
        }
    }
}

impl Default for DuckerState {
    fn default() -> Self {
        Self::new()
    }
}

/// Process one sample through the ducker.
///
/// Tracks `sidechain` envelope and reduces `x` gain when envelope exceeds
/// threshold.
///
/// Invariants:
/// - `envelope >= 0`
/// - `gain in [0, 1]`
/// - `|output| <= |input|`
///
/// # Errors
///
/// Returns [`KernelError`] if any input/output is non-finite.
pub fn ducker_process_sample_scalar(
    x: f32,
    sidechain: f32,
    state: &DuckerState,
    config: &DuckerCoeffs,
) -> Result<DuckerOutput, KernelError> {
    validate_finite_inputs(&[
        ("x", x),
        ("sidechain", sidechain),
        ("envelope", state.envelope),
        ("gain", state.gain),
        ("attack_coeff", config.attack_coeff),
        ("release_coeff", config.release_coeff),
        ("threshold", config.threshold),
        ("ratio", config.ratio),
    ])?;

    // Envelope follower: one-pole smoothing of |sidechain|
    let abs_sc = sidechain.abs();
    let coeff = if abs_sc > state.envelope {
        config.attack_coeff
    } else {
        config.release_coeff
    };
    let raw_env = coeff * abs_sc + (1.0 - coeff) * state.envelope;
    let new_envelope = checked_scalar_output(raw_env)?.max(0.0);

    // Gain computation (no transcendentals)
    // When envelope > threshold: reduce gain proportionally
    // gain = ratio + (threshold / envelope) * (1 - ratio)
    // At threshold: gain = ratio + 1*(1-ratio) = 1.0
    // As envelope → ∞: gain → ratio
    let new_gain = if new_envelope > config.threshold && config.threshold > 0.0 {
        let t = (config.threshold / new_envelope).clamp(0.0, 1.0);
        let gain = config.ratio + t * (1.0 - config.ratio);
        gain.clamp(0.0, 1.0)
    } else {
        1.0
    };

    let output = checked_scalar_output(x * new_gain)?;

    Ok(DuckerOutput {
        y: output,
        envelope: new_envelope,
        gain: new_gain,
    })
}

/// Validate ducker configuration parameters.
///
/// # Errors
///
/// Returns [`KernelError::InvalidParam`] if any parameter is out of range.
pub fn validate_ducker_config(config: &DuckerCoeffs) -> Result<(), KernelError> {
    validate_finite_inputs(&[
        ("attack_coeff", config.attack_coeff),
        ("release_coeff", config.release_coeff),
        ("threshold", config.threshold),
        ("ratio", config.ratio),
    ])?;
    if config.attack_coeff <= 0.0 || config.attack_coeff >= 1.0 {
        return Err(KernelError::InvalidParam {
            name: "attack_coeff",
            constraint: "in (0, 1)",
            value: config.attack_coeff,
        });
    }
    if config.release_coeff <= 0.0 || config.release_coeff >= 1.0 {
        return Err(KernelError::InvalidParam {
            name: "release_coeff",
            constraint: "in (0, 1)",
            value: config.release_coeff,
        });
    }
    if config.threshold <= 0.0 {
        return Err(KernelError::InvalidParam {
            name: "threshold",
            constraint: "strictly positive",
            value: config.threshold,
        });
    }
    if config.ratio < 0.0 || config.ratio >= 1.0 {
        return Err(KernelError::InvalidParam {
            name: "ratio",
            constraint: "in [0, 1)",
            value: config.ratio,
        });
    }
    Ok(())
}

#[cfg(test)]
#[path = "ducker_tests.rs"]
mod tests;
