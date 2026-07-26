// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Coupled CROWN + cost propagation through NY GraphNetwork.
//!
//! AC5 of #1739: propagate cost bounds alongside numerical bounds through
//! the per-layer CROWN verification pipeline. Each CROWN-verified layer is
//! directly paired with its cost profile derived from the *same*
//! `TensorKernelDef`, closing the gap between "bounds are correct" and
//! "cost is bounded."
//!
//! # Architecture
//!
//! The existing `verify_layerwise_with_timing` takes a separate
//! `dispatch_plan: &[DispatchStep]` that is decoupled from the layers
//! being CROWN-verified. This module generates the dispatch plan from
//! each layer's `TensorKernelDef` during CROWN propagation, guaranteeing
//! 1:1 coupling between verified bounds and cost profiles.
//!
//! # Usage
//!
//! ```text
//! let cert = verify_layerwise_coupled(
//!     &layers,
//!     &initial_bounds,
//!     &HardwareCostModel::m4_max(),
//!     100_000.0,
//! )?;
//! assert!(cert.timing.overall_passed);
//! assert!(cert.all_layers_coupled());
//! ```
//!
//! Part of #1739 Phase 2 — AC5: CROWN propagation of cost bounds.

use crate::cost_model::LayerCostProfile;
use crate::pipeline::{TimingCertificate, VerifiedStage};

#[cfg(feature = "ny")]
use crate::cost_model::HardwareCostModel;

/// Per-layer coupled verification result: CROWN bounds + cost profile.
///
/// Each entry proves that a single layer has both correct output bounds
/// (via CROWN) and bounded computational cost (via roofline model),
/// derived from the same `TensorKernelDef`.
#[derive(Debug, Clone)]
pub struct CoupledLayerResult {
    /// The CROWN-verified stage (bounds, shapes, method, soundness).
    pub stage: VerifiedStage,
    /// Cost profile for this layer from the roofline model.
    pub cost_profile: LayerCostProfile,
    /// Number of dispatch steps generated for this layer.
    pub dispatch_step_count: usize,
}

/// Certificate from coupled CROWN + cost propagation.
///
/// Extends `TimingCertificate` with per-layer coupling evidence: each
/// layer's cost profile is derived from the same `TensorKernelDef` that
/// was CROWN-verified, not from a separate dispatch plan.
#[derive(Debug, Clone)]
pub struct CoupledTimingCertificate {
    /// The underlying timing certificate (bounds + roofline timing).
    pub timing: TimingCertificate,
    /// Per-layer coupled results showing bounds + cost for each layer.
    pub coupled_layers: Vec<CoupledLayerResult>,
    /// Total dispatch steps across all layers.
    pub total_dispatch_steps: usize,
}

impl CoupledTimingCertificate {
    /// Whether all layers have both verified bounds and cost profiles.
    pub fn all_layers_coupled(&self) -> bool {
        !self.coupled_layers.is_empty()
            && self
                .coupled_layers
                .iter()
                .all(|l| l.dispatch_step_count > 0)
    }

    /// Generate a human-readable coupled verification report.
    pub fn report(&self) -> String {
        let mut out = self.timing.report();
        out.push_str("\n--- Per-Layer Coupled Verification ---\n\n");

        for (i, layer) in self.coupled_layers.iter().enumerate() {
            out.push_str(&format!(
                "Layer {}: {} (method={}, sound={})\n",
                i, layer.stage.name, layer.stage.method, layer.stage.is_sound,
            ));
            out.push_str(&format!(
                "  Bounds: input [{:.4}, {:.4}], output [{:.4}, {:.4}]\n",
                crate::stats::fold_min_propagate_nan(
                    layer.stage.input_lower.iter().copied(),
                    f64::INFINITY,
                ),
                crate::stats::fold_max_propagate_nan(
                    layer.stage.input_upper.iter().copied(),
                    f64::NEG_INFINITY,
                ),
                crate::stats::fold_min_propagate_nan(
                    layer.stage.output_lower.iter().copied(),
                    f64::INFINITY,
                ),
                crate::stats::fold_max_propagate_nan(
                    layer.stage.output_upper.iter().copied(),
                    f64::NEG_INFINITY,
                ),
            ));
            out.push_str(&format!(
                "  Cost: {:.1} FLOPs, {:.1} KB memory, {:.1} μs estimated\n",
                layer.cost_profile.flops as f64,
                layer.cost_profile.memory_bytes as f64 / 1024.0,
                layer.cost_profile.estimated_time_us,
            ));
            out.push_str(&format!(
                "  Dispatch steps: {}\n",
                layer.dispatch_step_count,
            ));
        }

        out.push_str(&format!(
            "\nTotal dispatch steps: {}\n",
            self.total_dispatch_steps,
        ));
        out.push_str(&format!(
            "All layers coupled: {}\n",
            self.all_layers_coupled(),
        ));

        out
    }
}

impl std::fmt::Display for CoupledTimingCertificate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CoupledTimingCertificate({} layers, coupled={}, timing={})",
            self.coupled_layers.len(),
            self.all_layers_coupled(),
            self.timing,
        )
    }
}

/// Verify a model layer-by-layer with coupled CROWN bounds + cost profiling.
///
/// For each layer, this function:
/// 1. Runs CROWN propagation via `tensor_kernel_to_graph` + `propagate_with_crown_fallback`.
/// 2. Generates the dispatch plan via `build_dispatch_plan` from the *same* `TensorKernelDef`.
/// 3. Profiles the dispatch plan through the roofline cost model.
/// 4. Pairs the CROWN-verified `VerifiedStage` with its `LayerCostProfile`.
///
/// The result is a `CoupledTimingCertificate` where each layer's bounds
/// and cost are derived from the same source, guaranteeing correctness
/// of the cost attribution.
///
/// # Arguments
///
/// * `layers` — Sequence of (TensorKernelDef, parameter bindings) for each layer.
/// * `initial_bounds` — Input bounds for the first layer.
/// * `hardware_model` — Target hardware for roofline timing estimates.
/// * `timing_bound_us` — Maximum acceptable inference time in microseconds.
///
/// # Errors
///
/// Returns `TtsVerifyError` if:
/// - CROWN propagation fails for any layer.
/// - Dispatch plan generation fails for any layer (treated as non-fatal:
///   the layer gets zero cost, and `all_layers_coupled()` returns false).
/// - Fewer than 2 layers.
/// - `timing_bound_us` is non-positive or non-finite.
///
/// Part of #1739 Phase 2 — AC5.
#[cfg(feature = "ny")]
pub fn verify_layerwise_coupled(
    layers: &[(
        nn_dsl::tensor_ir::TensorKernelDef,
        Vec<nn_verify::TensorParamBinding>,
    )],
    initial_bounds: &nn_verify::BoundedTensor,
    hardware_model: &HardwareCostModel,
    timing_bound_us: f64,
) -> Result<CoupledTimingCertificate, crate::error::TtsVerifyError> {
    use crate::cost_model::{
        profile_dispatch_plan, total_estimated_time_us, total_flops, total_memory_bytes,
    };
    use crate::error::{InvalidConfigKind, TtsVerifyError};
    use crate::pipeline::{stage_from_propagation_with_soundness, verify_pipeline};
    use nn_dsl::ScalarType;

    if !timing_bound_us.is_finite() || timing_bound_us <= 0.0 {
        return Err(TtsVerifyError::InvalidConfig(
            InvalidConfigKind::NonPositive {
                param: "timing_bound_us",
            },
        ));
    }

    if layers.len() < 2 {
        return Err(TtsVerifyError::InsufficientStages {
            count: layers.len(),
        });
    }

    let mut stages = Vec::with_capacity(layers.len());
    let mut coupled_layers = Vec::with_capacity(layers.len());
    let mut current_bounds = initial_bounds.clone();

    for (i, (layer, bindings)) in layers.iter().enumerate() {
        // 1. Run CROWN propagation for this layer.
        let graph = nn_verify::tensor_kernel_to_graph_with_norm_mode(
            layer,
            bindings,
            nn_verify::NormBoundsMode::CrownSampling,
        )
        .map_err(|e| TtsVerifyError::OperationFailed {
            context: "CROWN layer graph build",
            source: Box::new(e),
        })?;

        let (method, output_bounds, _fallback_reason) =
            nn_verify::propagate_with_crown_fallback(&graph, &current_bounds).map_err(|e| {
                TtsVerifyError::OperationFailed {
                    context: "CROWN layer propagation",
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

        // 2. Generate dispatch plan from the SAME TensorKernelDef.
        let (layer_cost, step_count) = match nn_dsl::build_dispatch_plan(layer, ScalarType::F32) {
            Ok((steps, _output_id)) => {
                let profiles = profile_dispatch_plan(&steps, hardware_model);
                let step_count = steps.len();
                // Aggregate per-layer cost from all dispatch steps.
                let layer_cost = aggregate_layer_cost(&profiles, &format!("layer_{i}"));
                (layer_cost, step_count)
            }
            Err(_) => {
                // Dispatch plan generation failed — layer gets zero cost.
                // This is non-fatal: the CROWN verification still holds,
                // but the cost profile is incomplete.
                let empty_cost = LayerCostProfile {
                    layer_name: format!("layer_{i}"),
                    flops: 0,
                    memory_bytes: 0,
                    estimated_time_us: 0.0,
                    measured_time_us: None,
                };
                (empty_cost, 0)
            }
        };

        coupled_layers.push(CoupledLayerResult {
            stage: stage.clone(),
            cost_profile: layer_cost,
            dispatch_step_count: step_count,
        });

        stages.push(stage);
        current_bounds = output_bounds;
    }

    // Compose pipeline from all verified stages.
    let bounds_cert = verify_pipeline(&stages)?;

    // Aggregate cost profiles: one per layer (aggregated from dispatch steps).
    let cost_profiles: Vec<LayerCostProfile> = coupled_layers
        .iter()
        .map(|cl| cl.cost_profile.clone())
        .collect();
    let worst_case_time = total_estimated_time_us(&cost_profiles);
    let flops = total_flops(&cost_profiles);
    let mem_bytes = total_memory_bytes(&cost_profiles);

    let timing_met = worst_case_time <= timing_bound_us;
    let total_dispatch_steps: usize = coupled_layers.iter().map(|cl| cl.dispatch_step_count).sum();

    let timing_cert = TimingCertificate {
        overall_passed: bounds_cert.is_valid && bounds_cert.is_sound && timing_met,
        bounds_cert,
        cost_profiles,
        worst_case_time_us: worst_case_time,
        total_flops: flops,
        total_memory_bytes: mem_bytes,
        hardware_name: format!(
            "peak={:.1} TFLOPS, bw={:.0} GB/s, dispatch={:.1} μs",
            hardware_model.peak_tflops_f32,
            hardware_model.peak_bandwidth_gbs,
            hardware_model.dispatch_overhead_us,
        ),
        timing_bound_us,
        timing_bound_met: timing_met,
        peak_memory: None,
    };

    Ok(CoupledTimingCertificate {
        timing: timing_cert,
        coupled_layers,
        total_dispatch_steps,
    })
}

/// Aggregate multiple dispatch step profiles into a single layer cost profile.
#[cfg_attr(not(feature = "ny"), allow(dead_code))]
pub(crate) fn aggregate_layer_cost(
    profiles: &[LayerCostProfile],
    layer_name: &str,
) -> LayerCostProfile {
    LayerCostProfile {
        layer_name: layer_name.to_string(),
        flops: profiles.iter().map(|p| p.flops).sum(),
        memory_bytes: profiles.iter().map(|p| p.memory_bytes).sum(),
        estimated_time_us: profiles.iter().map(|p| p.estimated_time_us).sum(),
        measured_time_us: None,
    }
}

#[cfg(test)]
#[path = "cost_propagation_tests.rs"]
mod tests;
