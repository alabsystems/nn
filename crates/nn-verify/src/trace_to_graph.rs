// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Translates a DynTensor `ComputationGraph` to a NY `GraphNetwork`.
//!
//! This bridges nn's imperative DynTensor tracing (captured via `trace_graph()`)
//! with NY's formal verification (IBP/CROWN bound propagation).
//!
//! Translation is NY-OWNED via `ny-trace-bridge`: every entry point serializes
//! the trace into the bridge schema ([`crate::trace_to_schema::to_schema`]) and
//! lowers it with `ny_trace_bridge::translate` — the single op→LayerSpec source
//! of truth, soundness-classified per op (`ny coverage`). The resulting
//! `ny_build::GraphModel` is built into a `GraphNetwork` with nn's strict
//! output-resolution policy. The legacy in-crate translator was deleted after
//! the INC-FINAL cutover (parity + full-corpus gates recorded in
//! ny/docs/TRACE_BRIDGE_MIGRATION.md).
//!
//! The owned boundary exposed here is limited to gamma-build / gamma-propagate's
//! `GraphModel` -> `GraphBuildInputs` -> `GraphNetwork` handoff. It does not
//! by itself wire traced models into other upstream NY subsystems.
//!
//! # Usage
//!
//! ```rust,no_run
//! use nn_core::dyn_tensor::trace::trace_graph;
//! use nn_verify::trace_to_graph_model;
//!
//! let (output, graph) = trace_graph(|| model.forward(&input))?;
//! let network = trace_to_graph_model(&graph)?;
//! // network is ready for IBP/CROWN propagation
//! ```

use ny_build::{GraphBuildInputs, GraphModel, GraphNetworkOptions, MissingOutputPolicy};
use ny_propagate::GraphNetwork;
use nn_core::dyn_tensor::trace::{ComputationGraph, TraceOp};

use crate::error::VerifyError;

/// Result of translating a `ComputationGraph` to gamma-build's owned graph
/// producer boundary plus the built `GraphNetwork`.
///
/// This exposes the reusable `ny_build::GraphModel` contract so future
/// gamma-build / NY consumers can inspect or rebuild from the traced
/// producer data without retracing DynTensor execution. The boundary is scoped
/// to the graph-builder / propagator contract used by nn today; it does not
/// imply integration with dormant upstream subsystems outside that path.
#[derive(Debug)]
pub struct TraceGraphBoundaryResult {
    /// Owned gamma-build producer contract for this traced graph.
    pub graph_model: GraphModel,
    /// The NY graph network built from `graph_model` using nn's
    /// strict output resolution policy.
    pub graph: GraphNetwork,
    /// Number of F16/BF16 downcast points modeled as Clip layers.
    pub dtype_cast_count: usize,
}

impl TraceGraphBoundaryResult {
    /// Reborrow the owned producer contract as graph-builder inputs.
    ///
    /// This borrowed view is sufficient for generic `ny_build` consumers to
    /// call `build_graph_network()` themselves without reconstructing traced
    /// weights, tensor provenance, or shape metadata.
    #[must_use]
    pub fn graph_build_inputs(&self) -> GraphBuildInputs<'_> {
        self.graph_model.graph_build_inputs()
    }

    /// Rebuild the graph network from the owned producer contract without
    /// retracing the source computation graph.
    ///
    /// Rebuilds with the same strict output-resolution policy used during the
    /// eager translation path (`MissingOutputPolicy::Error`).
    pub fn build_graph_network(&self) -> Result<GraphNetwork, VerifyError> {
        Ok(self
            .graph_model
            .build_graph_network(trace_graph_network_options())?)
    }
}

/// Result of translating a `ComputationGraph` to a NY `GraphNetwork`.
///
/// Contains the network for verification plus metadata about the translation
/// (e.g., dtype cast count for certificate auto-population). Use
/// [`trace_to_graph_model_with_boundary`] when callers also need the owned
/// `ny_build::GraphModel` producer artifact.
#[derive(Debug)]
pub struct TraceTranslateResult {
    /// The NY graph network ready for IBP/CROWN propagation.
    pub graph: GraphNetwork,
    /// Number of F16/BF16 downcast points modeled as Clip layers.
    /// Zero means the graph is pure F32. Part of #3023.
    pub dtype_cast_count: usize,
}

impl From<TraceGraphBoundaryResult> for TraceTranslateResult {
    fn from(result: TraceGraphBoundaryResult) -> Self {
        Self {
            graph: result.graph,
            dtype_cast_count: result.dtype_cast_count,
        }
    }
}

#[path = "trace_to_graph_predicates.rs"]
mod predicates;
use predicates::{is_variable_input, reachable_nodes};

fn trace_graph_network_options() -> GraphNetworkOptions {
    GraphNetworkOptions {
        // Strict mode: output tensor name must resolve to a graph node.
        // WarnAndFallback silently picks the last-added node on mismatch,
        // which can verify bounds on the WRONG output tensor (#2400).
        missing_output_policy: MissingOutputPolicy::Error,
        ..GraphNetworkOptions::default()
    }
}

fn validate_single_input_mode(graph: &ComputationGraph) -> Result<(), VerifyError> {
    // Guard: reject graphs with multiple variable inputs (#2425).
    // Single-input mode aliases ALL Input nodes to the same NETWORK_INPUT,
    // which produces unsound bounds when inputs are genuinely independent.
    let reachable = reachable_nodes(graph);
    let variable_input_count = graph
        .nodes()
        .iter()
        .filter(|n| reachable.contains(&n.id()) && matches!(n.op(), TraceOp::Input))
        .filter(|n| is_variable_input(graph, n.id(), &reachable))
        .count();
    if variable_input_count > 1 {
        return Err(VerifyError::MultipleVariableInputs {
            count: variable_input_count,
        });
    }

    Ok(())
}

/// Route one traced graph through the NY-owned bridge translator and package
/// the result in the [`TraceGraphBoundaryResult`] contract.
///
/// `multi_input` selects `ny_trace_bridge::translate::translate_multi_input`
/// (stacked-1D input + Slice/Reshape splits, mirroring
/// [`trace_to_graph_model_multi_input`]) versus the single-input
/// `translate_with_metadata`. Both return the bridge's `Translation`; its
/// `dtype_cast_count` metadata is wired through unchanged.
///
/// The `GraphNetwork` is built with nn's strict output-resolution policy
/// ([`trace_graph_network_options`]).
fn bridge_boundary_result(
    graph: &ComputationGraph,
    multi_input: bool,
) -> Result<TraceGraphBoundaryResult, VerifyError> {
    // Run NN's own single-input guard BEFORE delegating, so multi-variable
    // graphs are refused with the canonical
    // [`VerifyError::MultipleVariableInputs`]. The bridge applies the
    // semantically identical guard internally
    // (`NyError::UnsupportedConfiguration`), but callers and tests match on
    // NN's error contract. #seams.
    if !multi_input {
        validate_single_input_mode(graph)?;
    }
    let schema_graph = crate::trace_to_schema::to_schema(graph);
    let translation = if multi_input {
        ny_trace_bridge::translate::translate_multi_input(&schema_graph)?
    } else {
        ny_trace_bridge::translate::translate_with_metadata(&schema_graph)?
    };
    let network = translation
        .model
        .build_graph_network(trace_graph_network_options())?;
    Ok(TraceGraphBoundaryResult {
        graph_model: translation.model,
        graph: network,
        dtype_cast_count: translation.dtype_cast_count,
    })
}

/// Translate a DynTensor computation graph to a NY `GraphNetwork`.
///
/// Translation is NY-owned via `ny-trace-bridge` (see the module docs): the
/// trace is serialized into the bridge schema and lowered by
/// `ny_trace_bridge::translate`, then built with gamma-build's declarative
/// `build_graph_network()` API under nn's strict output-resolution policy.
///
/// Single-input mode: all `TraceOp::Input` nodes alias to the same
/// `NETWORK_INPUT` tensor. Use this for single-variable models.
///
/// The `ComputationGraph` is obtained via `trace_graph(|| model.forward(input))`.
/// Returns a `TraceTranslateResult` with the `GraphNetwork` and translation metadata.
///
/// Fail-closed by design:
/// * ops the bridge does not translate — unsupported ops AND deliberate
///   soundness refusals (e.g. MoeGating, WhereCond, ScatterAdd/IndexAdd,
///   data-dependent ops) — error with `NyError::UnsupportedOp` (surfaced as
///   [`VerifyError::Ny`]) instead of lowering vacuously;
/// * graphs with more than one variable input are refused with the canonical
///   [`VerifyError::MultipleVariableInputs`](crate::error::VerifyError::MultipleVariableInputs)
///   (nn's own guard runs before delegating; the bridge's semantically
///   identical internal guard is never the caller-visible error) — use
///   [`trace_to_graph_model_multi_input`] for genuinely independent inputs.
pub fn trace_to_graph_model(graph: &ComputationGraph) -> Result<TraceTranslateResult, VerifyError> {
    trace_to_graph_model_with_boundary(graph).map(Into::into)
}

/// Like [`trace_to_graph_model`], but also returns the owned
/// `ny_build::GraphModel` producer contract used to build the graph.
///
/// Callers can reborrow the result as `GraphBuildInputs` or rebuild a fresh
/// `GraphNetwork` later without retracing the source DynTensor computation.
/// This boundary is intentionally limited to the traced graph-model handoff
/// consumed by gamma-build / gamma-propagate in nn today. The returned
/// `graph_model` is the bridge-lowered `GraphModel` and `dtype_cast_count`
/// comes from the bridge's translation metadata.
pub fn trace_to_graph_model_with_boundary(
    graph: &ComputationGraph,
) -> Result<TraceGraphBoundaryResult, VerifyError> {
    bridge_boundary_result(graph, false)
}

/// Multi-input variant: each distinct `TraceOp::Input` node gets its own
/// variable slice from a stacked 1D input tensor (#2377).
///
/// Use this when the model has genuinely independent input variables (e.g.,
/// ProsodyPredictor with `bert_output` + `style`). The caller must provide
/// IBP bounds as a flat 1D tensor of shape `[sum of all input elements]`.
///
/// Weight-only Input nodes (consumed only as parameters by composite ops
/// like Conv1d/Linear) are automatically filtered out and do not require
/// slots in the stacked bounds.
///
/// Translation is NY-owned via
/// `ny_trace_bridge::translate::translate_multi_input`, which implements this
/// stacked-1D `multi_in` + Slice/Reshape producer contract.
pub fn trace_to_graph_model_multi_input(
    graph: &ComputationGraph,
) -> Result<TraceTranslateResult, VerifyError> {
    trace_to_graph_model_multi_input_with_boundary(graph).map(Into::into)
}

/// Like [`trace_to_graph_model_multi_input`], but also returns the owned
/// `ny_build::GraphModel` producer contract used to build the graph.
///
/// This preserves the stacked-input producer boundary (`multi_in` plus the
/// generated Slice/Reshape split layers) for future gamma-build consumers.
/// It does not automatically extend traced models into other upstream
/// NY subsystems beyond this graph-model boundary.
pub fn trace_to_graph_model_multi_input_with_boundary(
    graph: &ComputationGraph,
) -> Result<TraceGraphBoundaryResult, VerifyError> {
    bridge_boundary_result(graph, true)
}

// -- Segmented translation (#2378) --------------------------------------------

/// Translation result for a single segment of a segmented graph.
#[derive(Debug)]
pub struct SegmentTranslation {
    /// The NY graph network for this segment.
    pub result: TraceTranslateResult,
    /// Index of this segment (0-based).
    pub segment_index: usize,
    /// Reason for the preceding boundary (e.g., "length_regulate").
    /// `None` for the first segment.
    pub boundary_reason: Option<String>,
    /// Optional (lower, upper) bounds hint from the preceding boundary.
    pub boundary_bounds: Option<(f32, f32)>,
}

/// Result of translating a segmented computation graph.
///
/// Contains one `SegmentTranslation` per segment. Output bounds from
/// segment N feed as input bounds to segment N+1 during verification.
#[derive(Debug)]
pub struct SegmentedTranslateResult {
    /// Per-segment translations in order.
    pub segments: Vec<SegmentTranslation>,
}

/// Translate a computation graph with `SegmentBoundary` markers into
/// independently verifiable segments (#2378).
///
/// Data-dependent operations (e.g., `length_regulate` with runtime-dependent
/// `repeat_interleave`) break single-graph verification because the output
/// shape depends on tensor *values*. This function:
///
/// 1. Splits the graph at `SegmentBoundary` markers
/// 2. Translates each segment to a NY `GraphNetwork`
/// 3. Returns `SegmentedTranslateResult` with per-segment results
///
/// Each segment is a self-contained graph. The first segment uses the
/// original model inputs; subsequent segments use synthetic inputs at
/// the boundary shape. Callers compose bounds across segments by feeding
/// output bounds from segment N as input bounds to segment N+1.
///
/// If the graph has no `SegmentBoundary` markers, returns a single segment
/// equivalent to `trace_to_graph_model()`.
///
/// Translation is NY-owned via `ny-trace-bridge`: each segment goes through
/// the same single-input translation as [`trace_to_graph_model`] (per-segment
/// `translate_with_metadata`, mirroring what the bridge's
/// `translate_segmented` does internally while additionally keeping the
/// per-segment `dtype_cast_count` metadata the bridge's bare segmented entry
/// drops). Consequently the single-input guard now applies PER SEGMENT: a
/// segment with more than one variable input is refused with the canonical
/// [`VerifyError::MultipleVariableInputs`](crate::error::VerifyError::MultipleVariableInputs).
/// The deleted legacy segmented path skipped that guard and silently ALIASED
/// independent per-segment inputs to one network input — unsound bounds; the
/// sound refusal is deliberate (a flagged legacy nn gap, closed here).
pub fn trace_to_graph_segmented(
    graph: &ComputationGraph,
) -> Result<SegmentedTranslateResult, VerifyError> {
    let segmented = graph.split_at_segment_boundaries();

    let mut translations = Vec::with_capacity(segmented.segments.len());
    for (i, segment) in segmented.segments.into_iter().enumerate() {
        let result = bridge_boundary_result(&segment.graph, false).map(Into::into)?;
        translations.push(SegmentTranslation {
            result,
            segment_index: i,
            boundary_reason: segment.boundary_reason,
            boundary_bounds: segment.boundary_bounds,
        });
    }

    Ok(SegmentedTranslateResult {
        segments: translations,
    })
}
