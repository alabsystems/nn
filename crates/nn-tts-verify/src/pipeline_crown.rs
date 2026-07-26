// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! NY integration for the pipeline verification framework.
//!
//! These functions bridge NY's `BoundedTensor` results into
//! the pipeline composition framework, enabling end-to-end CROWN verification
//! of multi-stage TTS models.

use crate::error::{InvalidConfigKind, TtsVerifyError};

use super::{verify_pipeline, VerifiedStage};

const CROWN_PIPELINE_NORM_MODE: nn_verify::NormBoundsMode =
    nn_verify::NormBoundsMode::Conservative;

fn propagation_method_name(method: &nn_verify::PropMethod) -> &'static str {
    match method {
        nn_verify::PropMethod::Crown => "CROWN",
        nn_verify::PropMethod::AlphaCrown => "AlphaCrown",
        nn_verify::PropMethod::BetaCrown => "BetaCrown",
        nn_verify::PropMethod::Analytical => "Analytical",
        nn_verify::PropMethod::Ibp => "IBP",
        nn_verify::PropMethod::MixedIbpCrown => "mixed_IBP_CROWN",
        _ => "unknown",
    }
}

fn crown_error_must_propagate(error: &ny_propagate::prelude::NyError) -> bool {
    matches!(
        error,
        ny_propagate::prelude::NyError::SoundnessRefusal(_)
            | ny_propagate::prelude::NyError::InternalError(_)
    )
}

fn propagate_with_tight_crown_fallback(
    graph: &nn_verify::GraphNetwork,
    input_bounds: &nn_verify::BoundedTensor,
) -> Result<
    (
        nn_verify::PropMethod,
        nn_verify::BoundedTensor,
        Option<String>,
    ),
    nn_verify::VerifyError,
> {
    match graph.propagate_alpha_crown(input_bounds) {
        Ok(alpha_output) => Ok((nn_verify::PropMethod::AlphaCrown, alpha_output, None)),
        Err(alpha_err) => {
            if crown_error_must_propagate(&alpha_err) {
                return Err(alpha_err.into());
            }

            match graph.propagate_crown_with_provenance(input_bounds) {
                Ok(crown_result) => {
                    if crown_result.is_fallback() {
                        Ok((
                            nn_verify::PropMethod::Ibp,
                            crown_result.bounds,
                            Some(format!(
                                "alpha-CROWN failed: {alpha_err}; fixed-slope CROWN fell back to IBP internally"
                            )),
                        ))
                    } else {
                        Ok((nn_verify::PropMethod::Crown, crown_result.bounds, None))
                    }
                }
                Err(crown_err) => {
                    if crown_error_must_propagate(&crown_err) {
                        return Err(crown_err.into());
                    }

                    let ibp_output = graph.propagate_ibp(input_bounds)?;
                    Ok((
                        nn_verify::PropMethod::Ibp,
                        ibp_output,
                        Some(format!(
                            "alpha-CROWN failed: {alpha_err}; fixed-slope CROWN failed: {crown_err}"
                        )),
                    ))
                }
            }
        }
    }
}

/// Create a [`VerifiedStage`] from NY `BoundedTensor` results.
///
/// Extracts bounds from NY input/output `BoundedTensor` after
/// CROWN/IBP propagation, converting f32 bounds to f64. This bridges
/// existing per-model NY verification into the pipeline framework.
pub fn stage_from_bounds(
    name: &str,
    input_bounds: &nn_verify::BoundedTensor,
    output_bounds: &nn_verify::BoundedTensor,
    method: &str,
    is_sound: bool,
) -> VerifiedStage {
    VerifiedStage {
        name: name.to_string(),
        input_lower: input_bounds.lower().iter().map(|&v| f64::from(v)).collect(),
        input_upper: input_bounds.upper().iter().map(|&v| f64::from(v)).collect(),
        output_lower: output_bounds
            .lower()
            .iter()
            .map(|&v| f64::from(v))
            .collect(),
        output_upper: output_bounds
            .upper()
            .iter()
            .map(|&v| f64::from(v))
            .collect(),
        input_shape: input_bounds.shape().to_vec(),
        output_shape: output_bounds.shape().to_vec(),
        method: method.to_string(),
        is_sound,
    }
}

/// Create a [`VerifiedStage`] from a [`nn_verify::PropMethod`] result.
///
/// Convenience wrapper over [`stage_from_bounds`] that extracts the method
/// name from the propagation method enum.
pub fn stage_from_propagation(
    name: &str,
    input_bounds: &nn_verify::BoundedTensor,
    output_bounds: &nn_verify::BoundedTensor,
    method: &nn_verify::PropMethod,
) -> VerifiedStage {
    let soundness_mode = if method.is_tight() {
        nn_verify::VerificationSoundnessMode::Sound
    } else {
        nn_verify::VerificationSoundnessMode::Heuristic
    };
    stage_from_propagation_with_soundness(name, input_bounds, output_bounds, method, soundness_mode)
}

/// Create a [`VerifiedStage`] from a propagation result plus explicit soundness.
pub fn stage_from_propagation_with_soundness(
    name: &str,
    input_bounds: &nn_verify::BoundedTensor,
    output_bounds: &nn_verify::BoundedTensor,
    method: &nn_verify::PropMethod,
    soundness_mode: nn_verify::VerificationSoundnessMode,
) -> VerifiedStage {
    let is_sound =
        method.is_tight() && matches!(soundness_mode, nn_verify::VerificationSoundnessMode::Sound);
    stage_from_bounds(
        name,
        input_bounds,
        output_bounds,
        propagation_method_name(method),
        is_sound,
    )
}

/// Verify a model layer-by-layer using per-layer CROWN on pre-built GraphNetworks.
///
/// Same approach as [`verify_layerwise`] but accepts `GraphNetwork` directly,
/// skipping the `tensor_kernel_to_graph()` conversion step. This enables
/// trace-based verification where layers are captured via `trace_graph()` and
/// converted to `GraphNetwork` via `trace_to_graph_model()`.
pub fn verify_layerwise_from_graphs(
    graphs: &[nn_verify::GraphNetwork],
    initial_bounds: &nn_verify::BoundedTensor,
) -> Result<super::PipelineCertificate, TtsVerifyError> {
    if graphs.len() < 2 {
        return Err(TtsVerifyError::InsufficientStages {
            count: graphs.len(),
        });
    }

    let mut stages = Vec::with_capacity(graphs.len());
    let mut current_bounds = initial_bounds.clone();

    for (i, graph) in graphs.iter().enumerate() {
        let (method, output_bounds, _fallback_reason) =
            propagate_with_tight_crown_fallback(graph, &current_bounds).map_err(|e| {
                TtsVerifyError::OperationFailed {
                    context: "CROWN propagation",
                    source: Box::new(e),
                }
            })?;
        let soundness_mode =
            nn_verify::soundness_mode_for_graph(graph, &method, Some(&current_bounds)).map_err(
                |e| TtsVerifyError::OperationFailed {
                    context: "soundness classification",
                    source: Box::new(e),
                },
            )?;

        let stage = stage_from_propagation_with_soundness(
            &format!("layer_{i}"),
            &current_bounds,
            &output_bounds,
            &method,
            soundness_mode,
        );
        stages.push(stage);
        current_bounds = output_bounds;
    }

    verify_pipeline(&stages)
}

/// Grouping specification for multi-layer CROWN sub-graphs (#2592).
///
/// Each group is a set of layer indices that will be merged into a single
/// `GraphNetwork` for CROWN propagation. Normalization layers should still be
/// natural group boundaries for performance, but grouped CROWN now uses
/// `NormBoundsMode::Conservative` so norm-containing groups use sound
/// `IbpValidated` CROWN linearization (gc#4399).
///
/// # Example
///
/// For a 5-layer pipeline where layer 3 is InstanceNorm:
/// ```text
/// LayerwiseGrouping { groups: vec![vec![0, 1, 2], vec![3], vec![4]] }
/// ```
#[derive(Debug, Clone)]
pub struct LayerwiseGrouping {
    /// Groups of layer indices. Each group is merged into one `GraphNetwork`.
    pub groups: Vec<Vec<usize>>,
}

/// Validate grouping indices against layer count.
///
/// Checks: non-empty groups, in-range indices, strictly increasing within
/// groups, monotonic across groups, full layer coverage. Shared by
/// `verify_layerwise_grouped` and `verify_layerwise_mixed`.
fn validate_grouping(
    grouping: &LayerwiseGrouping,
    num_layers: usize,
) -> Result<(), TtsVerifyError> {
    if grouping.groups.len() < 2 {
        return Err(TtsVerifyError::InsufficientStages {
            count: grouping.groups.len(),
        });
    }
    let mut prev_max: Option<usize> = None;
    for group in grouping.groups.iter() {
        if group.is_empty() {
            return Err(TtsVerifyError::InvalidConfig(
                InvalidConfigKind::Constraint {
                    what: "grouping contains empty group",
                },
            ));
        }
        for (j, &idx) in group.iter().enumerate() {
            if idx >= num_layers {
                return Err(TtsVerifyError::InvalidConfig(
                    InvalidConfigKind::Constraint {
                        what: "group index out of range",
                    },
                ));
            }
            if j > 0 && idx <= group[j - 1] {
                return Err(TtsVerifyError::InvalidConfig(
                    InvalidConfigKind::Constraint {
                        what: "group indices not strictly increasing",
                    },
                ));
            }
        }
        if let Some(prev) = prev_max {
            if group[0] <= prev {
                return Err(TtsVerifyError::InvalidConfig(
                    InvalidConfigKind::Constraint {
                        what: "group not after previous group",
                    },
                ));
            }
        }
        prev_max = group.last().copied();
    }
    let mut covered = vec![false; num_layers];
    for group in &grouping.groups {
        for &idx in group {
            covered[idx] = true;
        }
    }
    if let Some(_missing) = covered.iter().position(|&c| !c) {
        return Err(TtsVerifyError::InvalidConfig(
            InvalidConfigKind::Constraint {
                what: "layer index not covered by any group",
            },
        ));
    }
    Ok(())
}

/// Build grouped graph from a subset of layers.
fn build_group_graph(
    layers: &[(
        nn_dsl::tensor_ir::TensorKernelDef,
        Vec<nn_verify::TensorParamBinding>,
    )],
    group: &[usize],
    _gi: usize,
) -> Result<nn_verify::GraphNetwork, TtsVerifyError> {
    let group_layers: Vec<_> = group
        .iter()
        .map(|&idx| (layers[idx].0.clone(), layers[idx].1.clone()))
        .collect();
    nn_verify::tensor_kernels_to_grouped_graph(&group_layers, CROWN_PIPELINE_NORM_MODE).map_err(
        |e| TtsVerifyError::OperationFailed {
            context: "grouped graph build",
            source: Box::new(e),
        },
    )
}

/// Verify a layerwise pipeline with explicit grouping for multi-layer CROWN.
///
/// Layers within the same group are merged into a single `GraphNetwork` via
/// [`nn_verify::tensor_kernels_to_grouped_graph`] and verified with CROWN as
/// a unit. Groups are chained via output→input bounds propagation.
///
/// # Arguments
///
/// * `layers` - Full layer sequence (TensorKernelDef + parameter bindings).
/// * `initial_bounds` - Input bounds for the first group.
/// * `grouping` - Which layers form each group (see [`LayerwiseGrouping`]).
pub fn verify_layerwise_grouped(
    layers: &[(
        nn_dsl::tensor_ir::TensorKernelDef,
        Vec<nn_verify::TensorParamBinding>,
    )],
    initial_bounds: &nn_verify::BoundedTensor,
    grouping: &LayerwiseGrouping,
) -> Result<super::PipelineCertificate, TtsVerifyError> {
    validate_grouping(grouping, layers.len())?;

    let mut stages = Vec::with_capacity(grouping.groups.len());
    let mut current_bounds = initial_bounds.clone();

    for (gi, group) in grouping.groups.iter().enumerate() {
        let graph = build_group_graph(layers, group, gi)?;
        let (method, output_bounds, _) =
            propagate_with_tight_crown_fallback(&graph, &current_bounds).map_err(|e| {
                TtsVerifyError::OperationFailed {
                    context: "CROWN propagation",
                    source: Box::new(e),
                }
            })?;
        let soundness_mode =
            nn_verify::soundness_mode_for_graph(&graph, &method, Some(&current_bounds)).map_err(
                |e| TtsVerifyError::OperationFailed {
                    context: "soundness classification",
                    source: Box::new(e),
                },
            )?;
        let stage = stage_from_propagation_with_soundness(
            &format!("group_{gi}"),
            &current_bounds,
            &output_bounds,
            &method,
            soundness_mode,
        );
        stages.push(stage);
        current_bounds = output_bounds;
    }

    verify_pipeline(&stages)
}

/// Per-group verification mode for mixed IBP/CROWN pipelines (#2599).
///
/// At production dimensions (D=512), some sub-blocks have Conv1d weights too
/// large for CROWN (512×512×3 = 786K elements). These groups use IBP only.
/// Smaller groups (128-channel Stage 1 sub-blocks, 49K-180K elements) are
/// CROWN-tractable and benefit from tighter bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupVerifyMode {
    /// IBP only — skip CROWN attempt (for intractable groups).
    Ibp,
    /// Alpha-first CROWN escalation with strict verifier-error propagation.
    Crown,
}

/// Verify a layerwise pipeline with per-group verification mode (#2599 Level 2).
///
/// Like [`verify_layerwise_grouped`] but each group can use a different
/// verification strategy. Groups marked [`GroupVerifyMode::Ibp`] use IBP
/// directly, saving time on intractable sub-blocks. Groups marked
/// [`GroupVerifyMode::Crown`] use the same alpha-first CROWN escalation as
/// [`verify_layerwise`], but refuse to silently mask verifier
/// `SoundnessRefusal` and `InternalError` failures behind IBP.
///
/// # Arguments
///
/// * `layers` - Full layer sequence (TensorKernelDef + parameter bindings).
/// * `initial_bounds` - Input bounds for the first group.
/// * `grouping` - Which layers form each group (see [`LayerwiseGrouping`]).
/// * `modes` - Per-group verification mode. Must have same length as `grouping.groups`.
pub fn verify_layerwise_mixed(
    layers: &[(
        nn_dsl::tensor_ir::TensorKernelDef,
        Vec<nn_verify::TensorParamBinding>,
    )],
    initial_bounds: &nn_verify::BoundedTensor,
    grouping: &LayerwiseGrouping,
    modes: &[GroupVerifyMode],
) -> Result<super::PipelineCertificate, TtsVerifyError> {
    if modes.len() != grouping.groups.len() {
        return Err(TtsVerifyError::InvalidConfig(
            InvalidConfigKind::Constraint {
                what: "modes length does not match groups length",
            },
        ));
    }
    validate_grouping(grouping, layers.len())?;

    let mut stages = Vec::with_capacity(grouping.groups.len());
    let mut current_bounds = initial_bounds.clone();

    for (gi, group) in grouping.groups.iter().enumerate() {
        let graph = build_group_graph(layers, group, gi)?;
        let (method, output_bounds) = match modes[gi] {
            GroupVerifyMode::Ibp => {
                let out = graph.propagate_ibp(&current_bounds).map_err(|e| {
                    TtsVerifyError::OperationFailed {
                        context: "IBP propagation",
                        source: Box::new(e),
                    }
                })?;
                (nn_verify::PropMethod::Ibp, out)
            }
            GroupVerifyMode::Crown => {
                let (m, out, _) = propagate_with_tight_crown_fallback(&graph, &current_bounds)
                    .map_err(|e| TtsVerifyError::OperationFailed {
                        context: "CROWN propagation",
                        source: Box::new(e),
                    })?;
                (m, out)
            }
        };
        let soundness_mode = if method.is_tight() {
            nn_verify::soundness_mode_for_graph(&graph, &method, Some(&current_bounds)).map_err(
                |e| TtsVerifyError::OperationFailed {
                    context: "soundness classification",
                    source: Box::new(e),
                },
            )?
        } else {
            nn_verify::VerificationSoundnessMode::Heuristic
        };
        let stage = stage_from_propagation_with_soundness(
            &format!("group_{gi}"),
            &current_bounds,
            &output_bounds,
            &method,
            soundness_mode,
        );
        stages.push(stage);
        current_bounds = output_bounds;
    }

    verify_pipeline(&stages)
}

/// Verify a model layer-by-layer at production dimensions using per-layer CROWN.
///
/// Each layer is verified independently via NY CROWN propagation.
/// Output bounds from layer N become input bounds for layer N+1. The resulting
/// `PipelineCertificate` proves end-to-end bounds by composing per-layer
/// certificates through junction containment.
///
/// Tensor kernels are translated with `NormBoundsMode::Conservative` so
/// normalization-heavy stages (InstanceNorm, LayerNorm, AdaIN, RMSNorm) use
/// sound `IbpValidated` CROWN linearization (gc#4399 closed).
pub fn verify_layerwise(
    layers: &[(
        nn_dsl::tensor_ir::TensorKernelDef,
        Vec<nn_verify::TensorParamBinding>,
    )],
    initial_bounds: &nn_verify::BoundedTensor,
) -> Result<super::PipelineCertificate, TtsVerifyError> {
    if layers.len() < 2 {
        return Err(TtsVerifyError::InsufficientStages {
            count: layers.len(),
        });
    }

    let mut stages = Vec::with_capacity(layers.len());
    let mut current_bounds = initial_bounds.clone();

    for (i, (layer, bindings)) in layers.iter().enumerate() {
        let graph = nn_verify::tensor_kernel_to_graph_with_norm_mode(
            layer,
            bindings,
            CROWN_PIPELINE_NORM_MODE,
        )
        .map_err(|e| TtsVerifyError::OperationFailed {
            context: "layer graph build",
            source: Box::new(e),
        })?;

        let (method, output_bounds, _fallback_reason) =
            propagate_with_tight_crown_fallback(&graph, &current_bounds).map_err(|e| {
                TtsVerifyError::OperationFailed {
                    context: "CROWN propagation",
                    source: Box::new(e),
                }
            })?;
        let soundness_mode =
            nn_verify::soundness_mode_for_graph(&graph, &method, Some(&current_bounds)).map_err(
                |e| TtsVerifyError::OperationFailed {
                    context: "soundness classification",
                    source: Box::new(e),
                },
            )?;

        let stage = stage_from_propagation_with_soundness(
            &format!("layer_{i}"),
            &current_bounds,
            &output_bounds,
            &method,
            soundness_mode,
        );
        stages.push(stage);
        current_bounds = output_bounds;
    }

    verify_pipeline(&stages)
}

#[cfg(test)]
mod tests {
    use super::crown_error_must_propagate;

    #[test]
    fn test_crown_error_must_propagate_soundness_refusal() {
        let error = ny_propagate::prelude::NyError::SoundnessRefusal("refused".into());
        assert!(crown_error_must_propagate(&error));
    }

    #[test]
    fn test_crown_error_must_propagate_internal_error() {
        let error = ny_propagate::prelude::NyError::InternalError("bug".into());
        assert!(crown_error_must_propagate(&error));
    }

    #[test]
    fn test_crown_error_must_propagate_keeps_nonfatal_errors_fallback_eligible() {
        let error = ny_propagate::prelude::NyError::UnsupportedOp("softmax".into());
        assert!(!crown_error_must_propagate(&error));
    }
}
