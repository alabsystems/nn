// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compile a DynTensor `ComputationGraph` (from `trace_graph()`) into
//! pre-built `TensorKernelDef` dispatch plans.
//!
//! This is the bridge between World A (DynTensor runtime traces) and
//! World B (TensorBlockBuilder → Metal dispatch). Each `TraceNode` in
//! the computation graph is lowered to a `CompiledStep` which is either
//! a GPU-dispatchable `TensorKernelDef` or a shape-only passthrough.
//!
//! # Usage
//!
//! ```rust,ignore
//! use nn_core::dyn_tensor::trace::{trace_graph, ComputationGraph};
//! use nn_dsl::trace_compile::compile_trace;
//!
//! let (output, graph) = trace_graph(|| model.forward(&input))?;
//! let steps = compile_trace(&graph)?;
//! // steps: Vec<CompiledStep> — pre-compiled dispatch plans
//! ```
//!
//! See `designs/2026-03-13-compile-time-graph-execution.md` for full design.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};

use crate::tensor_ir::TensorIRError;

#[path = "trace_compile_types.rs"]
mod trace_compile_types;
pub use trace_compile_types::{
    CompiledKernel, CompiledPlan, CompiledStep, FusionStats, PeepholeStats, RuntimeOpKind,
};

#[path = "trace_compile_native_ops.rs"]
mod trace_compile_native_ops;
pub use trace_compile_native_ops::{
    AttentionLayout, ConvActivation, FusedNormKind, GemmActivation, NativeOpKind,
    NormActivConv1dParams, NormActivation, ResBlockChainEntry, StyleBatchOffset,
    StyleProjectionParams,
};

#[path = "trace_compile_ops.rs"]
mod trace_compile_ops;
use trace_compile_ops::{add_weight, build_single_op, graph_input_ids};

#[path = "trace_compile_misc.rs"]
mod trace_compile_misc;

#[path = "trace_compile_selection.rs"]
mod trace_compile_selection;

#[path = "trace_compile_attention.rs"]
pub(crate) mod trace_compile_attention;

#[path = "trace_compile_unfold.rs"]
mod trace_compile_unfold;

#[path = "trace_compile_resblock.rs"]
mod trace_compile_resblock;

#[path = "trace_compile_peephole.rs"]
pub(crate) mod peephole;
pub use peephole::PeepholeConfig;

#[path = "trace_compile_dispatch.rs"]
mod dispatch;
use dispatch::compile_node;

/// Compile a traced computation graph into pre-built dispatch plans.
///
/// Walks the `ComputationGraph` in topological order. For each node,
/// creates a `CompiledStep`:
/// - Compute ops (matmul, conv, norm, etc.) → `CompiledStep::Dispatch`
/// - Shape ops (reshape, squeeze) → `CompiledStep::Passthrough`
/// - Graph inputs → `CompiledStep::InputForward`
/// - Identity ops (dropout) → `CompiledStep::IdentityPassthrough`
///
/// # Errors
///
/// Returns `TensorIRError::UnsupportedTraceOp` for ops that have no
/// TensorBlockBuilder mapping (e.g., PixelShuffle, custom ops).
pub fn compile_trace(graph: &ComputationGraph) -> Result<Vec<CompiledStep>, TensorIRError> {
    validate_topology(graph)?;

    let mut steps = Vec::with_capacity(graph.len());

    for node in graph.nodes() {
        let step = compile_node(node, graph)?;
        steps.push(step);
    }

    Ok(steps)
}

/// Build a [`CompiledPlan`] from pre-compiled steps and graph metadata.
fn build_plan(steps: Vec<CompiledStep>, graph: &ComputationGraph) -> CompiledPlan {
    let input_shapes: Vec<Vec<usize>> = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::Input))
        .map(|n| n.output_shape().to_vec())
        .collect();

    let output_step = if steps.is_empty() { 0 } else { steps.len() - 1 };

    let mut weight_names: Vec<String> = steps
        .iter()
        .filter_map(|s| match s {
            CompiledStep::Dispatch { weight_data, .. }
            | CompiledStep::NativeOp { weight_data, .. } => Some(weight_data),
            _ => None,
        })
        .flat_map(|wd| wd.keys().cloned())
        .collect();
    weight_names.sort();
    weight_names.dedup();

    CompiledPlan {
        steps,
        input_shapes,
        output_step,
        weight_names,
    }
}

/// Compile a traced computation graph into a [`CompiledPlan`] (no fusion).
pub fn compile_trace_to_plan(graph: &ComputationGraph) -> Result<CompiledPlan, TensorIRError> {
    let steps = compile_trace(graph)?;
    Ok(build_plan(steps, graph))
}

/// Compile a traced computation graph into a [`CompiledPlan`].
///
/// Uses constant folding, partition-driven elementwise fusion, and peephole
/// optimization. Recommended for production — used by `CompiledModel::builder().build()`.
///
/// Pipeline:
/// 1. Constant fold the graph
/// 2. Sequential elementwise chain fusion (`compile_trace_with_fusion`)
/// 3. Partition-driven overlay: fuse Elementwise-dominant groups that
///    sequential chain fusion missed (non-adjacent chains, fan-in patterns).
///    Passthrough (Opaque/Reduction/Native) groups keep their chain-fused steps.
/// 4. Peephole passes (NormActivConv1d, SiluMul, BiLstmCat, etc.)
/// 5. Demote zero-copy Dispatch to Passthrough
///
/// Partition codegen reduces dispatch count by ~7 on Kokoro (153 → 146).
/// The fused steps' `external_node_ids` are remapped to avoid referencing
/// `IdentityPassthrough` steps (fix for #4345 edge_map resolution failure).
pub fn compile_trace_to_plan_with_fusion(
    graph: &ComputationGraph,
) -> Result<CompiledPlan, TensorIRError> {
    // Pre-pass: fold constant subgraphs and simplify identity patterns (#3083).
    let folded_graph = constant_fold::constant_fold(graph);
    // Step 2: sequential chain fusion as base — preserves peephole patterns
    // like adjacent Silu+Mul for SwiGLU, and NormActivConv1d adjacency.
    let chain_fused_steps = compile_trace_with_fusion(&folded_graph)?;
    // Step 3: partition-driven overlay — only fuses Elementwise-dominant groups.
    // Passthrough groups keep chain-fused steps intact for peephole adjacency.
    let partition = trace_compile_partition::partition_graph(&folded_graph);
    let mut steps = trace_compile_partition_codegen::compile_partition_groups(
        &partition.groups,
        &folded_graph,
        &chain_fused_steps,
    )?;
    // Step 4: peephole passes (operate on adjacency patterns).
    peephole::apply_peephole(&mut steps, &folded_graph);
    // Step 5: demote zero-copy Dispatch to Passthrough. Part of D3 (#3351).
    simplify_zero_compute_dispatches(&mut steps);
    Ok(build_plan(steps, &folded_graph))
}

/// Compile with per-pass peephole configuration.
///
/// Same as [`compile_trace_to_plan_with_fusion`] but allows disabling
/// individual peephole passes via [`PeepholeConfig`]. Use this to diagnose
/// performance regressions by selectively disabling passes.
pub fn compile_trace_to_plan_configured(
    graph: &ComputationGraph,
    peephole_config: &PeepholeConfig,
) -> Result<CompiledPlan, TensorIRError> {
    let folded_graph = constant_fold::constant_fold(graph);
    let chain_fused_steps = compile_trace_with_fusion(&folded_graph)?;
    let partition = trace_compile_partition::partition_graph(&folded_graph);
    let mut steps = trace_compile_partition_codegen::compile_partition_groups(
        &partition.groups,
        &folded_graph,
        &chain_fused_steps,
    )?;
    peephole::apply_peephole_with_config(&mut steps, &folded_graph, peephole_config);
    simplify_zero_compute_dispatches(&mut steps);
    Ok(build_plan(steps, &folded_graph))
}

/// Run partition analysis on a computation graph and return dispatch counts.
///
/// Returns `(pre_partition_dispatches, post_partition_dispatches)` — the number
/// of GPU dispatches before and after the partition algorithm merges fusible ops
/// into groups. This is a diagnostic entry point for external callers (e.g.
/// `CompiledKokoro` diagnostics) to measure theoretical fusion reduction without
/// altering codegen.
pub fn partition_analysis(graph: &ComputationGraph) -> (usize, usize) {
    let folded = constant_fold::constant_fold(graph);
    let result = trace_compile_partition::partition_graph(&folded);
    (
        result.pre_partition_dispatches,
        result.post_partition_dispatches,
    )
}

/// Demote `Dispatch` steps whose dispatch plans are entirely zero-copy to `Passthrough`.
///
/// Some `CompiledStep::Dispatch` entries produce TensorKernelDefs that, when
/// planned, resolve to all `DispatchStep::Reshape` (buffer alias) operations.
/// These perform no GPU compute and can be represented as `Passthrough` steps,
/// removing them from the logical dispatch count. Part of D3 (#3351).
fn simplify_zero_compute_dispatches(steps: &mut [CompiledStep]) {
    use crate::codegen_msl_tensor::build_dispatch_plan;
    use crate::codegen_msl_tensor::DispatchStep;
    use crate::ir::ScalarType;

    for step in steps.iter_mut() {
        let (kernel_def, kernel_name) = match step {
            CompiledStep::Dispatch { kernel, .. } => (kernel.def(), kernel.name().to_string()),
            _ => continue,
        };

        // Build the dispatch plan at F32 — dtype doesn't affect whether steps are
        // shape-only. If plan building fails, skip this step (conservative).
        let plan = match build_dispatch_plan(kernel_def, ScalarType::F32) {
            Ok((plan, _output)) => plan,
            Err(_) => continue,
        };

        // A dispatch with no compute steps OR only Reshape steps is zero-copy.
        let all_zero_copy = plan
            .iter()
            .all(|s| matches!(s, DispatchStep::Reshape { .. }));
        if all_zero_copy && !plan.is_empty() {
            // Extract output shape from the kernel def's output node.
            // Use iter().find() for safety — node ID may not match array index.
            let output_shape = kernel_def
                .nodes
                .iter()
                .find(|n| n.id == kernel_def.output)
                .map(|n| n.shape.clone())
                .unwrap_or_default();
            *step = CompiledStep::Passthrough {
                op_name: kernel_name,
                output_shape,
            };
        }
    }
}

/// Resolve the shape of a trace node's i-th input from the computation graph.
///
/// # Errors
///
/// Returns `TensorIRError::MissingInputNode` when:
/// - The node has no input at `input_idx`, or
/// - The referenced input node does not exist in the graph.
///
/// This catches malformed graphs (from incomplete tracing or bugs in trace
/// recording) at compile time rather than silently producing wrong shapes.
fn resolve_input_shape<'a>(
    node: &'a TraceNode,
    input_idx: usize,
    graph: &'a ComputationGraph,
) -> Result<&'a [usize], TensorIRError> {
    let &input_id = node.inputs().get(input_idx).ok_or_else(|| {
        TensorIRError::MissingInputNode {
            node_name: node.name().to_string(),
            input_idx,
            input_id: 0, // no input_id available when index is out of bounds
        }
    })?;
    graph
        .node(input_id)
        .map(TraceNode::output_shape)
        .ok_or_else(|| TensorIRError::MissingInputNode {
            node_name: node.name().to_string(),
            input_idx,
            input_id,
        })
}

/// Verify that all nodes in the graph are in topological order (each node's
/// inputs reference only nodes that appear earlier). This catches out-of-order
/// or dangling references before compilation begins, preventing silent wrong
/// results in the executor.
fn validate_topology(graph: &ComputationGraph) -> Result<(), TensorIRError> {
    use std::collections::HashSet;
    let mut seen: HashSet<u64> = HashSet::with_capacity(graph.len());
    for node in graph.nodes() {
        for (input_idx, &input_id) in node.inputs().iter().enumerate() {
            if !seen.contains(&input_id) {
                return Err(TensorIRError::MissingInputNode {
                    node_name: node.name().to_string(),
                    input_idx,
                    input_id,
                });
            }
        }
        seen.insert(node.id());
    }
    Ok(())
}

// -- Serialization (JSON round-trip) ------------------------------------------

#[path = "compiled_plan_serde.rs"]
mod plan_serde;
pub use plan_serde::CompiledPlanSerdeError;

// -- MSL export for metallib precompilation -----------------------------------

#[path = "compiled_plan_msl.rs"]
mod plan_msl;
pub use plan_msl::{ExportMslError, MslSource};

// -- Diagnostics (summary + diff) ---------------------------------------------

#[path = "compiled_plan_diagnostics.rs"]
mod diagnostics;
pub use diagnostics::{BufferPlanMetrics, PlanDiff, PlanSummary};

// -- Fusion gap analysis (#3829) ----------------------------------------------

#[path = "fusion_gap_analyzer.rs"]
mod fusion_gap_analyzer;
pub use fusion_gap_analyzer::{
    analyze_fusion_gaps, theoretical_minimum_dispatches, FusionBlocker, FusionGap,
    FusionGapAnalysis,
};

// -- NativeOp fusion opportunity analyzer (#4252) -----------------------------

#[path = "fusion_opportunity_analyzer.rs"]
mod fusion_opportunity_analyzer;
pub use fusion_opportunity_analyzer::{analyze_fusion_opportunities, FusionOpportunity};

// -- Data-driven fusion scanner (#4264) ----------------------------------------

#[path = "trace_compile_fusion_scanner.rs"]
pub(crate) mod fusion_scanner;
pub use fusion_scanner::{
    scan_fusion_opportunities, FusionCategory as ScannerFusionCategory,
    FusionOpportunity as ScannerOpportunity, FusionScanResult,
};

// -- PeepholeConfig exhaustive search (#3828 Phase 4) -------------------------

#[path = "optimize_plan.rs"]
pub mod optimize_plan;
pub use optimize_plan::{
    analyze_pass_impact, count_dispatches, optimize_plan, optimize_plan_with_cost,
    optimize_segments, OptimizationResult, PassImpactEntry, SegmentOptimizationResult,
};

// -- Constant folding ---------------------------------------------------------

#[path = "trace_compile_constant_fold.rs"]
mod constant_fold;

// -- Op classification for graph-global fusion --------------------------------

#[path = "trace_compile_classify.rs"]
pub(crate) mod trace_compile_classify;

// -- DAG-aware graph partition ------------------------------------------------

#[path = "trace_compile_partition.rs"]
pub(crate) mod trace_compile_partition;

// -- Partition-driven codegen -------------------------------------------------

#[path = "trace_compile_partition_codegen.rs"]
mod trace_compile_partition_codegen;

// -- Elementwise chain fusion -------------------------------------------------

#[path = "trace_compile_fusion.rs"]
mod fusion;
pub use fusion::{compile_trace_with_fusion, detect_fusion_chains, FusionChainInfo, FusionPair};

#[cfg(kani)]
#[path = "kani_trace_compile_misc.rs"]
mod kani_trace_compile_misc;

#[cfg(kani)]
#[path = "kani_trace_compile_fusion.rs"]
mod kani_trace_compile_fusion;

#[cfg(kani)]
#[path = "kani_trace_compile_fusion_3745.rs"]
mod kani_trace_compile_fusion_3745;

#[cfg(kani)]
#[path = "kani_fusion_3738.rs"]
mod kani_fusion_3738;

#[cfg(test)]
#[path = "trace_compile_test_index.rs"]
mod test_index;

#[cfg(test)]
#[path = "trace_compile_swiglu_tests.rs"]
mod swiglu_tests;

#[cfg(test)]
#[path = "tests_peephole_fusion_native_ops.rs"]
mod peephole_fusion_native_ops_tests;
