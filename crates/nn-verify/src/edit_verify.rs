// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Edit verification via NY dual-path diamond DAG.
//!
//! Verifies that weight edits (LoRA overlays, ROME updates, direct writes)
//! produce bounded output changes: `|model(x; W+ΔW) - model(x; W)| < epsilon`.
//!
//! Uses the same diamond DAG pattern as fusion verification (`fusion.rs`):
//! two copies of the model subgraph share the same input, with a `SubLayer`
//! computing the diff at the output.
//!
//! ```text
//!          input [bounds]
//!         /             \
//!   model(W)       model(W+ΔW)
//!         \             /
//!          SubLayer (diff)
//!              |
//!       output [diff bounds]
//! ```
//!
//! # Design reference
//!
//! `designs/2026-03-11-verified-weight-surgery-nn-side.md` (D5)

use std::collections::HashMap;

use ny_api::BoundedTensor;
use ny_core::VerificationSoundnessMode;
use ndarray::ArrayD;

use crate::dead_neuron_proof::{run_dead_neuron_elimination, DeadNeuronEliminationProof};
use crate::error::{StructuralError, VerifyError};
use crate::fusion::propagate_with_crown_fallback;
use crate::graph_tensor::{tensor_kernel_to_graph, TensorParamBinding};
use crate::soundness::soundness_for_graph;
use crate::verify::PropMethod;

use ny_propagate::Network;
use nn_dsl::tensor_ir::TensorKernelDef;

/// Specification for a weight edit verification.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EditVerificationSpec {
    /// Model subgraph (TensorKernelDef) to verify.
    pub model: TensorKernelDef,
    /// Original weight values, keyed by parameter index.
    pub original_weights: HashMap<usize, ArrayD<f32>>,
    /// Edited weight values, keyed by parameter index (same keys as original).
    pub edited_weights: HashMap<usize, ArrayD<f32>>,
    /// Input bounds for the variable parameters.
    pub input_bounds: BoundedTensor,
    /// Maximum acceptable output difference.
    pub epsilon: f32,
}

/// Result of weight edit verification.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[must_use]
pub struct EditVerification {
    /// Lower bound on output difference (original - edited).
    pub diff_lower: f32,
    /// Upper bound on output difference (original - edited).
    pub diff_upper: f32,
    /// Maximum absolute difference bound: max(|diff_lower|, |diff_upper|).
    pub max_abs_diff: f32,
    /// Whether the diff is provably within the epsilon budget.
    pub within_epsilon: bool,
    /// The epsilon threshold used.
    pub epsilon: f32,
    /// Propagation method that produced these bounds.
    pub method: PropMethod,
    /// If CROWN failed and we fell back to IBP, the error reason.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub crown_fallback_reason: Option<String>,
    /// Soundness classification of the propagation result.
    #[serde(default = "default_heuristic")]
    pub soundness_mode: VerificationSoundnessMode,
    /// Optional dead-neuron elimination equivalence proof for the original
    /// network (upstream NY commit `1ed64542f`).
    ///
    /// Populated by [`verify_edit_with_elimination`] when the caller supplies
    /// a sequential form of the original network. Left `None` by the default
    /// [`verify_edit`] entry point because nn's graph translator produces
    /// `GraphNetwork` values and `eliminate_and_verify` requires a sequential
    /// [`ny_propagate::Network`].
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub dead_neuron_proof: Option<DeadNeuronEliminationProof>,
}

fn default_heuristic() -> VerificationSoundnessMode {
    VerificationSoundnessMode::Heuristic
}

impl EditVerification {
    /// Whether this verification result is conclusive.
    ///
    /// IBP produces vacuously wide Minkowski-difference bounds for diamond
    /// DAGs (same limitation as fusion verification). Only CROWN results
    /// capture input correlation tightly enough to be conclusive.
    pub fn is_conclusive(&self) -> bool {
        self.method.is_tight()
    }
}

/// Validate the edit verification spec: epsilon, key consistency, shape consistency.
fn validate_spec(spec: &EditVerificationSpec) -> Result<(), VerifyError> {
    if !spec.epsilon.is_finite() || spec.epsilon < 0.0 {
        return Err(VerifyError::InvalidThreshold {
            value: spec.epsilon,
        });
    }
    if spec
        .original_weights
        .keys()
        .collect::<std::collections::HashSet<_>>()
        != spec
            .edited_weights
            .keys()
            .collect::<std::collections::HashSet<_>>()
    {
        return Err(VerifyError::InvalidInput(
            "original_weights and edited_weights must have the same parameter indices".into(),
        ));
    }
    for (idx, original) in &spec.original_weights {
        let edited = spec.edited_weights.get(idx).ok_or_else(|| {
            VerifyError::InvalidInput(format!("missing edited weight for parameter index {idx}"))
        })?;
        if original.shape() != edited.shape() {
            return Err(VerifyError::InvalidInput(format!(
                "weight shape mismatch at parameter {idx}: original {:?} vs edited {:?}",
                original.shape(),
                edited.shape(),
            )));
        }
    }
    Ok(())
}

/// Compute worst-case diff bounds from two pairs of output bounds slices.
///
/// Returns `(diff_lower, diff_upper)` where diff = original - edited.
fn compute_diff_bounds(
    orig_output: &BoundedTensor,
    edit_output: &BoundedTensor,
) -> Result<(f32, f32), VerifyError> {
    let orig_lower_arr = orig_output.lower();
    let orig_upper_arr = orig_output.upper();
    let edit_lower_arr = edit_output.lower();
    let edit_upper_arr = edit_output.upper();

    let orig_lo = orig_lower_arr
        .as_slice()
        .ok_or_else(|| VerifyError::InvalidInput("original lower not contiguous".into()))?;
    let orig_hi = orig_upper_arr
        .as_slice()
        .ok_or_else(|| VerifyError::InvalidInput("original upper not contiguous".into()))?;
    let edit_lo = edit_lower_arr
        .as_slice()
        .ok_or_else(|| VerifyError::InvalidInput("edited lower not contiguous".into()))?;
    let edit_hi = edit_upper_arr
        .as_slice()
        .ok_or_else(|| VerifyError::InvalidInput("edited upper not contiguous".into()))?;

    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for i in 0..orig_lo.len().min(edit_lo.len()) {
        let d_lo = orig_lo[i] - edit_hi[i];
        let d_hi = orig_hi[i] - edit_lo[i];
        if d_lo < lo {
            lo = d_lo;
        }
        if d_hi > hi {
            hi = d_hi;
        }
    }

    if !lo.is_finite() || !hi.is_finite() {
        return Err(StructuralError::NonFiniteBounds {
            lower: lo,
            upper: hi,
        }
        .into());
    }
    Ok((lo, hi))
}

/// Verify that a weight edit preserves output bounds.
///
/// Builds two graph copies (original weights, edited weights), propagates
/// bounds independently, then computes the maximum output difference via
/// interval arithmetic.
///
/// # Errors
///
/// Returns [`VerifyError::InvalidThreshold`] if epsilon is NaN, Inf, or negative.
/// Returns [`VerifyError::InvalidInput`] if weight keys don't match or shapes mismatch.
pub fn verify_edit(spec: &EditVerificationSpec) -> Result<EditVerification, VerifyError> {
    validate_spec(spec)?;

    let param_count = spec
        .model
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, nn_dsl::tensor_ir::TensorOpKind::Input { .. }))
        .count();
    let original_bindings = build_bindings(param_count, &spec.original_weights);
    let edited_bindings = build_bindings(param_count, &spec.edited_weights);

    let original_graph = tensor_kernel_to_graph(&spec.model, &original_bindings)?;
    let edited_graph = tensor_kernel_to_graph(&spec.model, &edited_bindings)?;

    let (orig_method, orig_output, orig_fallback) =
        propagate_with_crown_fallback(&original_graph, &spec.input_bounds)?;
    let (edit_method, edit_output, edit_fallback) =
        propagate_with_crown_fallback(&edited_graph, &spec.input_bounds)?;

    let (diff_lower, diff_upper) = compute_diff_bounds(&orig_output, &edit_output)?;
    let max_abs_diff = diff_lower.abs().max(diff_upper.abs());

    let method = if orig_method.is_tight() && edit_method.is_tight() {
        orig_method
    } else {
        PropMethod::Ibp
    };

    let crown_fallback_reason = match (&orig_fallback, &edit_fallback) {
        (None, None) => None,
        (Some(r), None) | (None, Some(r)) => Some(r.clone()),
        (Some(r1), Some(r2)) => Some(format!("original: {r1}; edited: {r2}")),
    };

    let provenance =
        soundness_for_graph(&original_graph, &method, Some(&spec.input_bounds), false)?;

    Ok(EditVerification {
        diff_lower,
        diff_upper,
        max_abs_diff,
        within_epsilon: max_abs_diff <= spec.epsilon,
        epsilon: spec.epsilon,
        method,
        crown_fallback_reason,
        soundness_mode: provenance.mode(),
        dead_neuron_proof: None,
    })
}

/// Verify a weight edit and also produce a dead-neuron elimination equivalence
/// proof for the original (un-edited) network.
///
/// Runs [`verify_edit`] first, then composes the result with a call to
/// [`run_dead_neuron_elimination`] on the caller-supplied sequential form of
/// the original network. The result is an [`EditVerification`] whose
/// `dead_neuron_proof` field is `Some(_)`.
///
/// # Why a separate entry point?
///
/// nn's graph translator (`tensor_kernel_to_graph`) produces a
/// [`ny_propagate::GraphNetwork`]; upstream's `eliminate_and_verify`
/// operates on a sequential [`Network`]. Callers that already have (or can
/// build) a sequential form — e.g. a manually-constructed subnet or a linear
/// extraction from a larger graph — can use this entry point to attach a
/// formal equivalence proof to their edit verification result.
///
/// # Errors
///
/// Returns [`VerifyError::InvalidThreshold`] if `elimination_epsilon` is
/// NaN, Inf, or negative. Propagates all errors from [`verify_edit`] and
/// [`run_dead_neuron_elimination`].
pub fn verify_edit_with_elimination(
    spec: &EditVerificationSpec,
    original_sequential: &Network,
    elimination_epsilon: f32,
) -> Result<EditVerification, VerifyError> {
    let mut result = verify_edit(spec)?;
    let proof =
        run_dead_neuron_elimination(original_sequential, &spec.input_bounds, elimination_epsilon)?;
    result.dead_neuron_proof = Some(proof);
    Ok(result)
}

/// Build `TensorParamBinding` array from a parameter count and weight map.
///
/// Parameters in `weights` get `ConstantTensor` bindings; all others get `Variable`.
fn build_bindings(
    param_count: usize,
    weights: &HashMap<usize, ArrayD<f32>>,
) -> Vec<TensorParamBinding> {
    (0..param_count)
        .map(|i| {
            if let Some(arr) = weights.get(&i) {
                TensorParamBinding::ConstantTensor(arr.clone())
            } else {
                TensorParamBinding::Variable
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "edit_verify_tests.rs"]
mod tests;
