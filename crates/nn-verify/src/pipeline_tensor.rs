// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tensor-level verification pipeline (#703).
//!
//! Extracted from `pipeline.rs` to stay within the 500-line file limit.

use ny_api::BoundedTensor;
use nn_dsl::tensor_ir::TensorKernelDef;

use crate::bound_analysis::{
    analyze_layer_bounds, layers_needing_crown, AnalysisConfig, BoundAnalysisReport,
};
use crate::error::VerifyError;
use crate::graph_tensor::{tensor_kernel_to_graph_with_norm_mode, TensorParamBinding};
use crate::status::{ParamInputRecord, VerifyStatus};
use crate::verify::{run_escalation, KernelVerification, PropMethod, VerifyConfig};

/// Result of the tensor verification pipeline.
#[derive(Debug)]
#[non_exhaustive]
pub struct TensorPipelineResult {
    /// NY bounds verification result (scalar summary).
    pub verification: KernelVerification,
    /// Output bounds as a tensor (full per-element bounds).
    pub output_bounds: BoundedTensor,
    /// Number of variable inputs in the kernel.
    pub num_variables: usize,
    /// Per-layer bound trace for proof certificates (AC3, #802).
    ///
    /// Populated when `config.collect_layer_bounds()` is true. Contains one
    /// `LayerBoundRecord` per graph node in topological order with per-element
    /// input/output bounds and propagation method provenance (CROWN vs IBP).
    ///
    /// When `None`, layer bounds were not collected (default for backward compat).
    pub layer_bounds: Option<Vec<crate::certificate_types::LayerBoundRecord>>,
    /// Per-layer bound analysis with explosion detection and recommendations.
    ///
    /// Populated when `layer_bounds` is present. Contains derived metrics
    /// (width, expansion ratio, explosion points) and machine-readable
    /// `TighteningRecommendation`s for progressive tightening compilation.
    ///
    /// This is a diagnostic artifact, not a proof artifact — it belongs
    /// in the pipeline result (which drives tightening), not in
    /// `ProofCertificate` (which is shipped to auditors).
    pub bound_analysis: Option<BoundAnalysisReport>,
    /// Error message when layer bounds extraction was requested but failed.
    ///
    /// When `Some`, `layer_bounds` is `None` due to an extraction error (not
    /// because collection was disabled). Callers should check this field to
    /// distinguish "not requested" from "failed".
    pub layer_bounds_error: Option<String>,
}

/// Run the tensor verification pipeline for a `TensorKernelDef`:
/// translate → NY bounds (IBP/CROWN escalation) → record.
///
/// Replaces the manual 4-step wiring pattern:
/// 1. `tensor_kernel_to_graph(kernel, bindings)` → `GraphNetwork`
/// 2. `graph.propagate_ibp(input_bounds)` → `BoundedTensor`
/// 3. Validate output bounds
/// 4. `status.record_with_variable_inputs(...)`
///
/// `bindings` maps each `Input` node in the `TensorKernelDef` to a
/// `TensorParamBinding`. `input_bounds` provides the `BoundedTensor` for
/// all `Variable` inputs (stacked along axis 0 when >1 variable, matching
/// the `tensor_kernel_to_graph` convention).
///
/// `status_key` overrides the key in `nn_verify_status.json`. When `None`,
/// `kernel.name` is used.
///
/// # Errors
///
/// Returns an error if graph translation, bounds propagation, or status
/// recording fails.
#[must_use = "returns a Result that may contain an error"]
pub fn verify_tensor_and_record(
    status: &mut VerifyStatus,
    kernel: &TensorKernelDef,
    bindings: &[TensorParamBinding],
    input_bounds: &BoundedTensor,
    status_key: Option<&str>,
) -> Result<TensorPipelineResult, VerifyError> {
    verify_tensor_and_record_with_config(
        status,
        kernel,
        bindings,
        input_bounds,
        status_key,
        &VerifyConfig::default(),
    )
}

/// Tensor verification pipeline with explicit config for normalization mode.
///
/// Same as [`verify_tensor_and_record`] but accepts a [`VerifyConfig`] to
/// control normalization layer bounds mode (see [`NormBoundsMode`]). Use
/// `config.with_norm_mode(NormBoundsMode::ForwardMode)` for tighter bounds
/// through normalization layers (see #744).
#[must_use = "returns a Result that may contain an error"]
pub fn verify_tensor_and_record_with_config(
    status: &mut VerifyStatus,
    kernel: &TensorKernelDef,
    bindings: &[TensorParamBinding],
    input_bounds: &BoundedTensor,
    status_key: Option<&str>,
    config: &VerifyConfig,
) -> Result<TensorPipelineResult, VerifyError> {
    let key = status_key.unwrap_or(&kernel.name);

    // 1. Translate tensor kernel to NY graph with norm mode.
    let graph = tensor_kernel_to_graph_with_norm_mode(kernel, bindings, config.norm_mode())?;

    // 2. Run IBP → CROWN escalation to get bounds.
    let (verification, output_bounds) =
        run_escalation(&graph, input_bounds, &kernel.name, config, false)?;

    // 3. Build per-variable input metadata for status recording.
    //    For tensor variables, record the scalar min(lower) / max(upper)
    //    across all elements — the scalar summary of tensor bounds.
    let num_variables = bindings
        .iter()
        .filter(|b| matches!(b, TensorParamBinding::Variable))
        .count();

    let variable_inputs: Vec<ParamInputRecord> = bindings
        .iter()
        .enumerate()
        .filter(|(_, b)| matches!(b, TensorParamBinding::Variable))
        .enumerate()
        .map(|(var_idx, (param_idx, _))| {
            // Extract scalar bounds for this variable from the stacked input tensor.
            // Single variable: input_bounds shape matches the variable shape.
            // Multi variable: variables are stacked along axis 0.
            let (lo, hi) = if num_variables == 1 {
                let (lower, upper) = input_bounds.lower_upper();
                (
                    lower.iter().copied().fold(f32::INFINITY, f32::min),
                    upper.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                )
            } else {
                // Slice axis 0 at var_idx for this variable's bounds.
                let (lower, upper) = input_bounds.lower_upper();
                let lo_slice = lower.index_axis(ndarray::Axis(0), var_idx);
                let hi_slice = upper.index_axis(ndarray::Axis(0), var_idx);
                (
                    lo_slice.iter().copied().fold(f32::INFINITY, f32::min),
                    hi_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                )
            };
            ParamInputRecord {
                param_index: param_idx,
                lower: lo,
                upper: hi,
            }
        })
        .collect();

    let constant_params: Vec<f32> = bindings
        .iter()
        .filter_map(|b| match b {
            TensorParamBinding::ConstantScalar(v) => Some(*v),
            TensorParamBinding::Variable | TensorParamBinding::ConstantTensor(_) => None,
        })
        .collect();

    // 4. Record to status file with actual input tensor shape.
    let input_shape_raw = input_bounds.shape();
    status.record_with_variable_inputs(
        &verification,
        &variable_inputs,
        &constant_params,
        Some(key),
        Some(input_shape_raw),
    )?;

    // 5. Optionally collect per-layer bound trace for certificates (#802 AC3).
    let (layer_bounds, layer_bounds_error) = if config.collect_layer_bounds() {
        match crate::layer_bounds::extract_layer_bounds(&graph, input_bounds) {
            Ok(lb) => (Some(lb), None),
            Err(e) => (None, Some(format!("{e}"))),
        }
    } else {
        (None, None)
    };

    // 6. Run bound analysis on collected layer bounds (progressive tightening).
    let bound_analysis = layer_bounds
        .as_ref()
        .map(|lb| analyze_layer_bounds(key, lb, &AnalysisConfig::default()));

    // 7. Return the BoundedTensor from run_escalation (IBP or CROWN).
    //    When CROWN escalated, output_bounds contains the tighter CROWN bounds.
    //    The scalar summary in `verification` matches the same propagation method.
    Ok(TensorPipelineResult {
        verification,
        output_bounds,
        num_variables,
        layer_bounds,
        bound_analysis,
        layer_bounds_error,
    })
}

// ---------------------------------------------------------------------------
// Selective CROWN escalation (Phase 2, #2454)
// ---------------------------------------------------------------------------

/// Result of selective CROWN escalation guided by BoundAnalysisReport.
#[derive(Debug)]
#[non_exhaustive]
pub struct SelectiveCrownResult {
    /// Per-node output bounds after selective CROWN tightening.
    /// Nodes identified as explosion points have CROWN bounds; others keep IBP.
    pub node_bounds: std::collections::HashMap<String, BoundedTensor>,
    /// Layer indices that were targeted for CROWN tightening (from EscalateToCrown).
    pub crown_layer_indices: Vec<usize>,
    /// Number of layers that actually received CROWN bounds (may be less than
    /// `crown_layer_indices.len()` if CROWN failed at some nodes).
    pub crown_tightened_count: usize,
    /// IBP-only output bounds (before CROWN tightening) for comparison.
    pub ibp_output_bounds: BoundedTensor,
    /// CROWN-tightened output bounds from the final layer.
    pub output_bounds: BoundedTensor,
    /// Bound analysis report from the initial IBP pass.
    pub ibp_analysis: BoundAnalysisReport,
    /// Bound analysis report after selective CROWN tightening.
    pub crown_analysis: BoundAnalysisReport,
}

/// Run selective per-layer CROWN escalation guided by BoundAnalysisReport.
///
/// This is the Phase 2 implementation of progressive tightening compilation
/// (#2454). Instead of running CROWN on the entire graph (O(L^2) backward
/// passes), it:
///
/// 1. Runs a fast IBP forward pass to get per-layer bounds.
/// 2. Analyzes the bounds with [`analyze_layer_bounds`] to find explosion points.
/// 3. Extracts layers needing CROWN via [`layers_needing_crown`].
/// 4. Re-runs CROWN-IBP with `min_width_to_tighten` set to the analysis config's
///    `crown_escalation_width`, so only wide-IBP nodes get CROWN backward passes.
///
/// This is O(K * L) where K = number of explosion-point layers, vs O(L^2) for
/// full CROWN. For typical models, K << L.
///
/// # Arguments
///
/// * `graph` — Pre-translated NY `GraphNetwork`.
/// * `input_bounds` — Input bounds for the network.
/// * `analysis_config` — Thresholds controlling which layers are flagged for
///   CROWN (via `crown_escalation_width`).
///
/// # Errors
///
/// Returns `VerifyError::Ny` if IBP propagation or CROWN tightening fails.
pub fn verify_with_selective_crown(
    graph: &ny_propagate::GraphNetwork,
    input_bounds: &BoundedTensor,
    analysis_config: &AnalysisConfig,
) -> Result<SelectiveCrownResult, VerifyError> {
    // 1. IBP forward pass: fast, sound, potentially loose.
    let ibp_output = graph.propagate_ibp(input_bounds)?;

    // 2. Collect per-node IBP bounds for analysis and as precomputed input.
    let ibp_node_bounds = graph.collect_crown_ibp_bounds_dag(input_bounds)?;

    // 3. Extract per-layer bounds from the IBP-only pass for analysis.
    let ibp_layer_records = crate::layer_bounds::extract_layer_bounds(graph, input_bounds)?;

    // 4. Analyze: find explosion points and generate EscalateToCrown recommendations.
    let ibp_analysis = analyze_layer_bounds("selective_crown", &ibp_layer_records, analysis_config);
    let crown_layer_indices = layers_needing_crown(&ibp_analysis);

    // 5. If no layers need CROWN, return IBP result directly (fast path).
    if crown_layer_indices.is_empty() {
        let crown_analysis = ibp_analysis.clone();
        return Ok(SelectiveCrownResult {
            node_bounds: ibp_node_bounds,
            crown_layer_indices,
            crown_tightened_count: 0,
            ibp_output_bounds: ibp_output.clone(),
            output_bounds: ibp_output,
            ibp_analysis,
            crown_analysis,
        });
    }

    // 6. Selective CROWN: use min_width_to_tighten so only nodes with IBP width
    //    above the escalation threshold get CROWN backward passes. This is O(K*L)
    //    where K = number of wide nodes, not O(L^2) for full CROWN.
    let crown_result = graph
        .collect_crown_ibp_bounds_dag_with_precomputed_ibp_and_width_threshold(
            input_bounds,
            ibp_node_bounds,
            crate::verify::crown_deadline(), // bound the CROWN backward passes
            analysis_config.crown_escalation_width,
        )?;

    // 7. Count how many nodes were actually CROWN-tightened (non-fallback provenance).
    let crown_tightened_count = crown_result
        .provenance
        .values()
        .filter(|prov| !prov.is_fallback())
        .count();

    // 8. Extract the tightened output bounds from the graph's output node.
    let topo_order = graph.topological_sort()?;
    let output_node_name = topo_order.last().ok_or(VerifyError::EmptyGraph)?;
    let output_bounds = crown_result
        .bounds
        .get(output_node_name)
        .cloned()
        .unwrap_or_else(|| ibp_output.clone());

    // 9. Re-extract layer bounds from the tightened result for the post-analysis.
    //    Use the tightened bounds to build LayerBoundRecords with CROWN provenance.
    let crown_layer_records =
        build_layer_records_from_crown_result(graph, &topo_order, &crown_result, input_bounds);

    let crown_analysis = analyze_layer_bounds(
        "selective_crown_tightened",
        &crown_layer_records,
        analysis_config,
    );

    Ok(SelectiveCrownResult {
        node_bounds: crown_result.bounds,
        crown_layer_indices,
        crown_tightened_count,
        ibp_output_bounds: ibp_output,
        output_bounds,
        ibp_analysis,
        crown_analysis,
    })
}

/// Build `LayerBoundRecord`s from a `GraphCrownIbpBoundsResult` for analysis.
///
/// Maps per-node bounds + provenance into the same format that
/// `extract_layer_bounds` produces, allowing `analyze_layer_bounds` to
/// compare IBP-only vs. CROWN-tightened reports.
fn build_layer_records_from_crown_result(
    graph: &ny_propagate::GraphNetwork,
    topo_order: &[String],
    crown_result: &ny_propagate::types::GraphCrownIbpBoundsResult,
    network_input: &BoundedTensor,
) -> Vec<crate::certificate_types::LayerBoundRecord> {
    let mut records = Vec::with_capacity(topo_order.len());

    for (layer_index, node_name) in topo_order.iter().enumerate() {
        let node = match graph.node(node_name) {
            Some(n) => n,
            None => continue,
        };

        let layer_type = node.layer().layer_type().to_string();

        let output_bt = match crown_result.bounds.get(node_name) {
            Some(bt) => bt,
            None => continue,
        };

        // Resolve input bounds from predecessors.
        let inputs = node.inputs();
        let input_bt = if inputs.is_empty() || inputs[0] == ny_propagate::NETWORK_INPUT {
            network_input
        } else {
            crown_result.bounds.get(&inputs[0]).unwrap_or(network_input)
        };

        // Provenance: CROWN if not fallback, IBP otherwise.
        let method = match crown_result.provenance.get(node_name) {
            Some(prov) if !prov.is_fallback() => PropMethod::Crown,
            _ => PropMethod::Ibp,
        };

        let input_pairs: Vec<(f32, f32)> = {
            let (lower, upper) = input_bt.lower_upper();
            lower
                .iter()
                .zip(upper.iter())
                .map(|(&lo, &hi)| (lo, hi))
                .collect()
        };
        let output_pairs: Vec<(f32, f32)> = {
            let (lower, upper) = output_bt.lower_upper();
            lower
                .iter()
                .zip(upper.iter())
                .map(|(&lo, &hi)| (lo, hi))
                .collect()
        };

        records.push(crate::certificate_types::LayerBoundRecord {
            layer_index,
            layer_type,
            input_bounds: input_pairs,
            output_bounds: output_pairs,
            method,
            node_name: Some(node_name.clone()),
            input_sources: None,
        });
    }

    records
}
