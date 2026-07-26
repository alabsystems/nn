// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Overlay composition verification API.
//!
//! When stacking multiple low-rank overlays on a base model, verifies that the
//! composition is safe — the combined perturbation preserves output bounds.
//! Individual overlay verification is necessary but not sufficient: overlays
//! targeting the same weight matrices can interfere.
//!
//! # Design
//!
//! 1. **Non-overlapping targets** — overlays targeting different weight matrices
//!    compose trivially. No additional verification needed.
//! 2. **Overlapping targets** — compute accumulated perturbation on shared weights
//!    and verify via dual-path propagation.
//! 3. **Certificate** — records which overlays were verified together, accumulated
//!    perturbation norms, and propagation results.
//!
//! # References
//!
//! - `designs/2026-03-11-verified-weight-surgery-nn-side.md` (D5)
//! - Issue #1846 (overlay composition verification API)

use std::collections::{HashMap, HashSet};

use ny_api::BoundedTensor;
use ny_core::VerificationSoundnessMode;
use ndarray::ArrayD;

use crate::edit_verify::{verify_edit, EditVerification, EditVerificationSpec};
use crate::error::VerifyError;
use crate::verify::PropMethod;

use nn_dsl::tensor_ir::TensorKernelDef;

/// A verified overlay with known weight deltas and target layers.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct VerifiedOverlay {
    /// Name of this overlay (e.g., "speaker_style_lora").
    pub name: String,
    /// Which parameter indices this overlay modifies.
    pub target_params: HashSet<usize>,
    /// Weight deltas keyed by parameter index: ΔW for each targeted weight.
    pub deltas: HashMap<usize, ArrayD<f32>>,
    /// Frobenius norm of each delta: ‖ΔWᵢ‖_F.
    pub delta_norms: HashMap<usize, f32>,
}

impl VerifiedOverlay {
    /// Create a new overlay from weight deltas.
    ///
    /// Computes Frobenius norms automatically from the delta tensors.
    ///
    /// # Errors
    ///
    /// Returns [`VerifyError::InvalidInput`] if `deltas` is empty or contains
    /// non-finite values.
    pub fn new(name: String, deltas: HashMap<usize, ArrayD<f32>>) -> Result<Self, VerifyError> {
        if deltas.is_empty() {
            return Err(VerifyError::InvalidInput(
                "overlay must have at least one weight delta".into(),
            ));
        }

        let target_params: HashSet<usize> = deltas.keys().copied().collect();
        let mut delta_norms = HashMap::new();

        for (&idx, delta) in &deltas {
            // Validate finiteness of delta values.
            let norm_sq: f32 = delta.iter().map(|v| v * v).sum();
            if !norm_sq.is_finite() {
                return Err(VerifyError::NonFiniteInputMetadata {
                    context: format!("delta norm for parameter {idx} is non-finite"),
                });
            }
            delta_norms.insert(idx, norm_sq.sqrt());
        }

        Ok(Self {
            name,
            target_params,
            deltas,
            delta_norms,
        })
    }
}

/// Specification for overlay preservation bounds.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BoundSpec {
    /// Maximum acceptable output difference (epsilon).
    pub epsilon: f32,
    /// Description of what this bound preserves (e.g., "intelligibility").
    pub description: String,
}

impl BoundSpec {
    /// Create a new bound specification.
    pub fn new(epsilon: f32, description: impl Into<String>) -> Self {
        Self {
            epsilon,
            description: description.into(),
        }
    }
}

/// Certificate proving overlay composition is safe.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct CompositionCertificate {
    /// Names of overlays verified together.
    pub overlay_names: Vec<String>,
    /// Parameter indices where overlays interact (overlap).
    pub overlapping_params: Vec<usize>,
    /// Parameter indices with no interaction (disjoint).
    pub disjoint_params: Vec<usize>,
    /// Accumulated perturbation norm per overlapping parameter: ‖Σᵢ ΔWᵢ‖_F.
    pub accumulated_norms: HashMap<usize, f32>,
    /// Edit verification result for the accumulated perturbation.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub verification: Option<EditVerification>,
    /// Whether the composition passes all preservation specs.
    pub all_specs_pass: bool,
    /// Per-spec pass/fail results.
    pub spec_results: Vec<SpecResult>,
    /// Propagation method used.
    pub method: PropMethod,
    /// Soundness classification.
    #[serde(default = "default_heuristic")]
    pub soundness_mode: VerificationSoundnessMode,
}

fn default_heuristic() -> VerificationSoundnessMode {
    VerificationSoundnessMode::Heuristic
}

/// Result for a single preservation spec.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct SpecResult {
    /// Description from the BoundSpec.
    pub description: String,
    /// Epsilon threshold.
    pub epsilon: f32,
    /// Whether this spec was satisfied.
    pub passed: bool,
    /// Actual max absolute diff (None if no overlapping params needed verification).
    pub max_abs_diff: Option<f32>,
}

/// Compute the overlay interaction matrix.
///
/// Returns a mapping from parameter index to the set of overlay indices that
/// target that parameter. Parameters targeted by only one overlay are "disjoint"
/// (no composition verification needed). Parameters targeted by 2+ overlays
/// are "overlapping" (need composition verification).
pub fn overlay_interaction_matrix(overlays: &[VerifiedOverlay]) -> HashMap<usize, Vec<usize>> {
    let mut param_to_overlays: HashMap<usize, Vec<usize>> = HashMap::new();
    for (overlay_idx, overlay) in overlays.iter().enumerate() {
        for &param_idx in &overlay.target_params {
            param_to_overlays
                .entry(param_idx)
                .or_default()
                .push(overlay_idx);
        }
    }
    param_to_overlays
}

/// Verify that a set of overlays composes safely on a base model.
///
/// 1. Computes overlay interaction matrix to find overlapping targets.
/// 2. For non-overlapping targets, composition is trivially safe.
/// 3. For overlapping targets, accumulates deltas and runs edit verification
///    on the combined perturbation.
///
/// # Arguments
///
/// * `model` — The model subgraph to verify.
/// * `original_weights` — Base model weights keyed by parameter index.
/// * `overlays` — Set of overlays to compose.
/// * `input_bounds` — Input bounds for verification.
/// * `preservation_specs` — Output bound specifications that must be preserved.
///
/// # Errors
///
/// Returns [`VerifyError::InvalidInput`] if overlays list is empty, weight
/// shapes don't match, or overlays target parameters not in `original_weights`.
pub fn verify_overlay_composition(
    model: &TensorKernelDef,
    original_weights: &HashMap<usize, ArrayD<f32>>,
    overlays: &[VerifiedOverlay],
    input_bounds: &BoundedTensor,
    preservation_specs: &[BoundSpec],
) -> Result<CompositionCertificate, VerifyError> {
    if overlays.is_empty() {
        return Err(VerifyError::InvalidInput(
            "at least one overlay is required".into(),
        ));
    }

    // Compute interaction matrix.
    let interactions = overlay_interaction_matrix(overlays);

    // Partition parameters into disjoint and overlapping.
    let mut disjoint_params = Vec::new();
    let mut overlapping_params = Vec::new();
    for (&param_idx, overlay_indices) in &interactions {
        if overlay_indices.len() == 1 {
            disjoint_params.push(param_idx);
        } else {
            overlapping_params.push(param_idx);
        }
    }
    disjoint_params.sort_unstable();
    overlapping_params.sort_unstable();

    // Accumulate deltas for all targeted parameters.
    let mut accumulated_deltas: HashMap<usize, ArrayD<f32>> = HashMap::new();
    let mut accumulated_norms: HashMap<usize, f32> = HashMap::new();

    for overlay in overlays {
        for (&param_idx, delta) in &overlay.deltas {
            // Validate that the original weight exists.
            let original = original_weights.get(&param_idx).ok_or_else(|| {
                VerifyError::InvalidInput(format!(
                    "overlay '{}' targets parameter {param_idx} which is not in original_weights",
                    overlay.name,
                ))
            })?;

            // Validate shape match.
            if delta.shape() != original.shape() {
                return Err(VerifyError::InvalidInput(format!(
                    "overlay '{}' delta shape {:?} != original weight shape {:?} at param {param_idx}",
                    overlay.name,
                    delta.shape(),
                    original.shape(),
                )));
            }

            // Accumulate: ΔW_total = Σᵢ ΔWᵢ
            accumulated_deltas
                .entry(param_idx)
                .and_modify(|acc| *acc = &*acc + delta)
                .or_insert_with(|| delta.clone());
        }
    }

    // Compute norms of accumulated deltas.
    for (&param_idx, acc_delta) in &accumulated_deltas {
        let norm_sq: f32 = acc_delta.iter().map(|v| v * v).sum();
        if !norm_sq.is_finite() {
            return Err(VerifyError::NonFiniteInputMetadata {
                context: format!("accumulated delta norm for parameter {param_idx} is non-finite"),
            });
        }
        accumulated_norms.insert(param_idx, norm_sq.sqrt());
    }

    // Build edited weights: W + ΔW_total for each targeted parameter.
    let mut edited_weights = original_weights.clone();
    for (&param_idx, acc_delta) in &accumulated_deltas {
        let original = &original_weights[&param_idx];
        edited_weights.insert(param_idx, original + acc_delta);
    }

    let overlay_names: Vec<String> = overlays.iter().map(|o| o.name.clone()).collect();

    // If no overlapping params, composition is trivially safe for all specs.
    // We still run verification on the accumulated perturbation for the certificate.
    let max_epsilon = preservation_specs
        .iter()
        .map(|s| s.epsilon)
        .fold(f32::INFINITY, f32::min);
    let epsilon = if max_epsilon.is_finite() && max_epsilon >= 0.0 {
        max_epsilon
    } else if preservation_specs.is_empty() {
        // Default epsilon when no specs provided — just verify boundedness.
        1.0
    } else {
        return Err(VerifyError::InvalidThreshold { value: max_epsilon });
    };

    let edit_spec = EditVerificationSpec {
        model: model.clone(),
        original_weights: original_weights.clone(),
        edited_weights,
        input_bounds: input_bounds.clone(),
        epsilon,
    };

    let verification = verify_edit(&edit_spec)?;

    // Check each preservation spec.
    let spec_results: Vec<SpecResult> = preservation_specs
        .iter()
        .map(|spec| SpecResult {
            description: spec.description.clone(),
            epsilon: spec.epsilon,
            passed: verification.max_abs_diff <= spec.epsilon,
            max_abs_diff: Some(verification.max_abs_diff),
        })
        .collect();

    let all_specs_pass = spec_results.iter().all(|r| r.passed);

    Ok(CompositionCertificate {
        overlay_names,
        overlapping_params,
        disjoint_params,
        accumulated_norms,
        verification: Some(verification.clone()),
        all_specs_pass,
        spec_results,
        method: verification.method,
        soundness_mode: verification.soundness_mode,
    })
}

#[cfg(test)]
#[path = "overlay_composition_tests.rs"]
mod tests;
