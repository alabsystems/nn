// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro TTS junction contract bounds — dvoice runtime contracts for nn-verify.
//!
//! Mirrors the 6 junction contracts defined in dvoice's
//! `crates/dvoice-tts/src/kokoro/junction_contracts.rs`.
//! These constants define the verified bounds at zone crossing points
//! in the Kokoro TTS pipeline:
//!
//! - **Zone A**: CPU F32 front-end (tokens -> conditioning tensors)
//! - **Zone B**: BF16 Metal decoder (conditioning -> spectrogram)
//! - **Zone C**: F32 islands (SourceModule sine, magnitude exp, iSTFT)
//!
//! Used by compose tests to verify NY proven output bounds
//! are contained within junction input bounds at each crossing.
//!
//! Part of #2478.

use crate::pipeline::VerifiedStage;

// ── J2: Decoder -> SourceModule boundary ──────────────────────────

/// F0 lower bound (Hz) with GPU float tolerance.
///
/// The F0 predictor outputs ReLU-activated values (non-negative), but
/// GPU float accumulation can produce tiny negatives (observed: -0.24 Hz).
/// dvoice uses -5.0 as the tolerance floor.
pub const J2_F0_LOWER: f64 = -5.0;

/// F0 upper bound (Hz). Soprano ceiling + headroom.
pub const J2_F0_UPPER: f64 = 800.0;

/// Energy feature lower bound.
///
/// Energy from ProsodyPredictor is a raw Linear layer output (no activation).
/// Observed real-weight values include -9.8 (#1590). Bounds set with 5x headroom.
pub const J2_ENERGY_LOWER: f64 = -50.0;

/// Energy feature upper bound.
pub const J2_ENERGY_UPPER: f64 = 50.0;

// ── J3: Generator post_conv -> magnitude exp() ───────────────────

/// Pre-exp magnitude lower clamp.
pub const J3_MAGNITUDE_LOWER: f64 = -80.0;

/// Pre-exp magnitude upper clamp. exp(80) ~ 5.5e34, within F32.
pub const J3_MAGNITUDE_UPPER: f64 = 80.0;

// ── J3b: Phase channels post_conv ────────────────────────────────

/// Phase absolute maximum. 2*pi*1000. Large values indicate accumulation
/// drift and produce numerically unstable sin() output.
pub const J3B_PHASE_LOWER: f64 = -6283.2;

/// Phase absolute maximum (positive).
pub const J3B_PHASE_UPPER: f64 = 6283.2;

// ── J4: F32 -> BF16 downcast precision ───────────────────────────

/// BF16 safe absolute maximum.
///
/// BF16 has 7 mantissa bits. At |x| in [128, 256): ULP = 1.0.
/// Typical pipeline features |x| < 100, so ULP < 1.0.
pub const J4_BF16_LOWER: f64 = -128.0;

/// BF16 safe absolute maximum (positive).
pub const J4_BF16_UPPER: f64 = 128.0;

// ── J5: iSTFT -> audio output ────────────────────────────────────

/// Audio output lower bound. PCM convention.
pub const J5_AUDIO_LOWER: f64 = -1.0;

/// Audio output upper bound. PCM convention.
pub const J5_AUDIO_UPPER: f64 = 1.0;

/// A named junction contract with bounds.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct JunctionContract {
    /// Junction identifier (e.g., "J2_F0", "J3_MAGNITUDE").
    pub name: &'static str,
    /// Human-readable description of the zone crossing.
    pub zone: &'static str,
    /// Lower bound.
    pub lower: f64,
    /// Upper bound.
    pub upper: f64,
}

impl JunctionContract {
    /// Create a new junction contract.
    pub const fn new(name: &'static str, zone: &'static str, lower: f64, upper: f64) -> Self {
        Self {
            name,
            zone,
            lower,
            upper,
        }
    }
}

/// A junction contract with an attached constructive composition proof.
///
/// Wraps a `JunctionContract` with an optional `CrownCompositionProofExport`
/// Lean4 source proving that NY output bounds at this junction are
/// contained within the contract bounds. Used for pipeline-level certification
/// where each zone crossing carries machine-checkable evidence (#4315).
#[derive(Debug, Clone)]
pub struct VerifiedJunctionContract {
    /// The underlying junction contract bounds.
    pub contract: JunctionContract,
    /// Lean4 source from `CrownCompositionProofExport` proving containment
    /// of the stage output bounds within the junction contract bounds.
    ///
    /// `None` when the composition proof was not generated or failed.
    pub composition_proof_lean4: Option<String>,
    /// Theorem name from the composition proof.
    pub composition_theorem_name: Option<String>,
    /// Whether the proven output bounds are contained within the contract.
    pub bounds_verified: bool,
}

impl VerifiedJunctionContract {
    /// Create a verified junction contract from a base contract.
    ///
    /// Initially unverified (`bounds_verified = false`, no composition proof).
    /// Use [`with_composition_proof`] to attach proof evidence.
    #[must_use]
    pub fn new(contract: JunctionContract) -> Self {
        Self {
            contract,
            composition_proof_lean4: None,
            composition_theorem_name: None,
            bounds_verified: false,
        }
    }

    /// Attach a composition proof Lean4 source and mark as verified.
    #[must_use]
    pub fn with_composition_proof(mut self, lean4_source: String, theorem_name: String) -> Self {
        self.composition_proof_lean4 = Some(lean4_source);
        self.composition_theorem_name = Some(theorem_name);
        self.bounds_verified = true;
        self
    }

    /// Whether this junction has a machine-checkable composition proof.
    #[must_use]
    pub fn has_composition_proof(&self) -> bool {
        self.composition_proof_lean4.is_some()
    }
}

/// All 6 Kokoro junction contracts.
#[must_use]
pub fn all_contracts() -> [JunctionContract; 6] {
    [
        JunctionContract::new("J2_F0", "Decoder -> SourceModule", J2_F0_LOWER, J2_F0_UPPER),
        JunctionContract::new(
            "J2_ENERGY",
            "Decoder -> SourceModule",
            J2_ENERGY_LOWER,
            J2_ENERGY_UPPER,
        ),
        JunctionContract::new(
            "J3_MAGNITUDE",
            "Generator post_conv",
            J3_MAGNITUDE_LOWER,
            J3_MAGNITUDE_UPPER,
        ),
        JunctionContract::new(
            "J3B_PHASE",
            "Generator post_conv",
            J3B_PHASE_LOWER,
            J3B_PHASE_UPPER,
        ),
        JunctionContract::new(
            "J4_BF16",
            "F32 -> BF16 downcast",
            J4_BF16_LOWER,
            J4_BF16_UPPER,
        ),
        JunctionContract::new("J5_AUDIO", "iSTFT output", J5_AUDIO_LOWER, J5_AUDIO_UPPER),
    ]
}

/// Check whether proven output bounds are contained within a junction contract.
///
/// Returns `true` if every element of `proven_lower..proven_upper` lies within
/// `contract.lower..contract.upper`.
#[must_use]
pub fn bounds_within_contract(
    contract: &JunctionContract,
    proven_lower: &[f64],
    proven_upper: &[f64],
) -> bool {
    if proven_lower.len() != proven_upper.len() {
        return false;
    }
    for (lo, hi) in proven_lower.iter().zip(proven_upper.iter()) {
        if !lo.is_finite() || !hi.is_finite() {
            return false;
        }
        if *lo < contract.lower || *hi > contract.upper {
            return false;
        }
    }
    true
}

/// Maximum violation of proven bounds against a junction contract.
///
/// Returns 0.0 if fully contained, positive value for the worst violation.
#[must_use]
pub fn max_contract_violation(
    contract: &JunctionContract,
    proven_lower: &[f64],
    proven_upper: &[f64],
) -> f64 {
    let mut worst = 0.0_f64;
    for (lo, hi) in proven_lower.iter().zip(proven_upper.iter()) {
        if !lo.is_finite() || !hi.is_finite() {
            return f64::MAX;
        }
        let lower_gap = contract.lower - lo; // positive if proven_lower < contract_lower
        let upper_gap = hi - contract.upper; // positive if proven_upper > contract_upper
        worst = worst.max(lower_gap).max(upper_gap);
    }
    worst.max(0.0)
}

/// Build a `VerifiedStage` with uniform junction contract bounds.
///
/// Creates a stage where all input elements share the `input` contract bounds
/// and all output elements share the `output` contract bounds. Used for
/// composing Kokoro pipeline stages in `verify_pipeline()`.
pub fn contract_stage(
    name: &str,
    input_shape: &[usize],
    output_shape: &[usize],
    input_contract: &JunctionContract,
    output_contract: &JunctionContract,
    method: &str,
    is_sound: bool,
) -> VerifiedStage {
    let in_elements: usize = input_shape.iter().product();
    let out_elements: usize = output_shape.iter().product();

    VerifiedStage::new(
        name,
        input_shape.to_vec(),
        output_shape.to_vec(),
        vec![input_contract.lower; in_elements],
        vec![input_contract.upper; in_elements],
        vec![output_contract.lower; out_elements],
        vec![output_contract.upper; out_elements],
        method,
        is_sound,
    )
}

#[cfg(test)]
#[path = "kokoro_contracts_tests.rs"]
mod tests;
