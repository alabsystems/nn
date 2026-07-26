// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Dead-neuron elimination equivalence proof — nn-side wrapper around
//! NY's [`eliminate_and_verify`] (upstream commit `1ed64542f`).
//!
//! [`eliminate_and_verify`] performs dead/constant neuron elimination on a
//! sequential [`ny_propagate::Network`] and then formally verifies the
//! optimized network is equivalent to the original via a difference network
//! (Linf equivalence within an input region). The bundle it returns contains
//! the optimized network, an [`EliminationCertificate`], and an
//! [`EquivalenceResult`].
//!
//! This module exposes a small, serializable [`DeadNeuronEliminationProof`]
//! summary suitable for attaching to nn-verify's certificate bundle and for
//! the Moonshot `Certificate` field in `nn-tts-verify`.
//!
//! # Design reference
//!
//! `designs/2026-04-19-NY-f57811-adoption.md` §3
//! (P2 issue #3 — *Wire eliminate_and_verify into edit_verify and certify bundle*).
//!
//! # Availability
//!
//! `eliminate_and_verify` requires a sequential [`ny_propagate::Network`].
//! nn's graph translation produces `GraphNetwork` values, so this helper is a
//! stand-alone entry point: callers construct a sequential `Network` (e.g., a
//! subnet extracted from a trace, or a manually-built test fixture) and pass it
//! in. When a caller operates only on `GraphNetwork` (the common case), they
//! simply skip this step and leave the proof field `None`.

use ny_api::BoundedTensor;
use ny_core::Bound;
use ny_propagate::{
    analyze_neurons, eliminate_and_verify, EliminationCertificate, EliminationVerification,
    EquivalenceResult, Network, PropagationConfig,
};

use crate::error::VerifyError;

/// Serializable summary of a dead-neuron-elimination equivalence proof.
///
/// Built from NY's [`EliminationVerification`] bundle. Captures the
/// three pieces of evidence a deployment auditor needs:
///
/// 1. How many neurons were eliminated ([`neurons_before`] / [`neurons_after`]).
/// 2. Whether the optimized network is formally equivalent to the original
///    ([`equivalent`]).
/// 3. The worst-case output difference bound proven by CROWN ([`worst_case_bound`]).
///
/// [`EliminationVerification`]: ny_propagate::EliminationVerification
/// [`neurons_before`]: DeadNeuronEliminationProof::neurons_before
/// [`neurons_after`]: DeadNeuronEliminationProof::neurons_after
/// [`equivalent`]: DeadNeuronEliminationProof::equivalent
/// [`worst_case_bound`]: DeadNeuronEliminationProof::worst_case_bound
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[must_use]
pub struct DeadNeuronEliminationProof {
    /// Total neurons (at ReLU layers) in the original sequential network.
    pub neurons_before: usize,
    /// Neurons remaining after eliminating dead/constant/always-active neurons.
    pub neurons_after: usize,
    /// Fraction of neurons eliminated (0.0 = none, 1.0 = all).
    ///
    /// Computed as `1 - (neurons_after / neurons_before)`.
    pub elimination_fraction: f32,
    /// Number of layers in the original network.
    pub layers_before: usize,
    /// Number of layers in the optimized network.
    pub layers_after: usize,
    /// Whether the optimized network was proven Linf-equivalent to the
    /// original within `epsilon` by [`verify_equivalence`].
    ///
    /// [`verify_equivalence`]: ny_propagate::verify_equivalence
    pub equivalent: bool,
    /// Maximum absolute output difference proven by the equivalence verifier.
    ///
    /// For `EquivalenceResult::Equivalent { bound }` this is `bound`. For
    /// `NotEquivalent { worst_case_bound }` or `Unknown { best_bound }` this
    /// is the respective best-effort value.
    pub worst_case_bound: f64,
    /// Epsilon budget used for the equivalence check.
    pub epsilon: f32,
    /// Human-readable label for the equivalence outcome:
    /// `"equivalent"`, `"not_equivalent"`, or `"unknown"`.
    pub equivalence_label: String,
}

impl DeadNeuronEliminationProof {
    /// Whether this proof attests a deployment-safe optimization — the
    /// equivalence verifier proved the optimized network matches the original
    /// within epsilon.
    #[must_use]
    pub fn is_deployment_safe(&self) -> bool {
        self.equivalent
    }

    /// Whether any neurons were actually removed.
    #[must_use]
    pub fn eliminated_any(&self) -> bool {
        self.neurons_after < self.neurons_before
    }
}

/// Run NY's dead-neuron elimination and equivalence verification on
/// a sequential [`Network`] and return a [`DeadNeuronEliminationProof`].
///
/// Runs the full upstream pipeline (see `ny_propagate::elimination`):
///
/// 1. [`analyze_neurons`] computes per-neuron IBP classifications.
/// 2. [`eliminate_and_verify`] eliminates dead/constant/always-active neurons
///    and verifies equivalence via CROWN on the difference network.
/// 3. The summary is packaged into a [`DeadNeuronEliminationProof`].
///
/// # Errors
///
/// Returns [`VerifyError::Ny`] for any analysis/elimination/equivalence
/// failure surfaced by NY.
/// Returns [`VerifyError::InvalidThreshold`] if `epsilon` is NaN, Inf, or
/// negative.
pub fn run_dead_neuron_elimination(
    network: &Network,
    input: &BoundedTensor,
    epsilon: f32,
) -> Result<DeadNeuronEliminationProof, VerifyError> {
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(VerifyError::InvalidThreshold { value: epsilon });
    }

    // 1. Classify neurons at ReLU layers via IBP.
    let analysis = analyze_neurons(network, input)?;

    // 2. Convert input BoundedTensor to a flat &[Bound] slice for the
    //    equivalence verifier.
    let input_bounds = flatten_bounds(input)?;

    // 3. Eliminate and formally verify equivalence.
    let verification: EliminationVerification = eliminate_and_verify(
        network,
        &analysis,
        &input_bounds,
        epsilon,
        PropagationConfig::default(),
    )?;

    Ok(summarize(&verification, epsilon))
}

/// Flatten a [`BoundedTensor`] into a `Vec<Bound>` per-element slice.
fn flatten_bounds(input: &BoundedTensor) -> Result<Vec<Bound>, VerifyError> {
    let lower = input.lower();
    let upper = input.upper();
    let lo_slice = lower
        .as_slice()
        .ok_or_else(|| VerifyError::InvalidInput("input bounds not contiguous".into()))?;
    let hi_slice = upper
        .as_slice()
        .ok_or_else(|| VerifyError::InvalidInput("input bounds not contiguous".into()))?;
    if lo_slice.len() != hi_slice.len() {
        return Err(VerifyError::InvalidInput(format!(
            "bounds length mismatch: lower={}, upper={}",
            lo_slice.len(),
            hi_slice.len()
        )));
    }
    let mut out = Vec::with_capacity(lo_slice.len());
    for (lo, hi) in lo_slice.iter().zip(hi_slice.iter()) {
        out.push(
            Bound::try_new(*lo, *hi)
                .map_err(|e| VerifyError::InvalidInput(format!("invalid input bound: {e}")))?,
        );
    }
    Ok(out)
}

/// Package an [`EliminationVerification`] into a serializable proof summary.
fn summarize(verification: &EliminationVerification, epsilon: f32) -> DeadNeuronEliminationProof {
    let cert: &EliminationCertificate = &verification.certificate;
    let (equivalent, worst_case_bound, label) = match &verification.equivalence {
        EquivalenceResult::Equivalent { bound } => (true, *bound, "equivalent".to_string()),
        EquivalenceResult::NotEquivalent { worst_case_bound } => {
            (false, *worst_case_bound, "not_equivalent".to_string())
        }
        EquivalenceResult::Unknown { best_bound } => (false, *best_bound, "unknown".to_string()),
        // `EquivalenceResult` is `#[non_exhaustive]` — default to the
        // conservative "unknown" interpretation for future variants.
        _ => (false, f64::INFINITY, "unknown".to_string()),
    };

    DeadNeuronEliminationProof {
        neurons_before: cert.neurons_before,
        neurons_after: cert.neurons_after,
        elimination_fraction: cert.elimination_fraction(),
        layers_before: cert.layers_before,
        layers_after: cert.layers_after,
        equivalent,
        worst_case_bound,
        epsilon,
        equivalence_label: label,
    }
}

#[cfg(test)]
#[path = "dead_neuron_proof_tests.rs"]
mod tests;
