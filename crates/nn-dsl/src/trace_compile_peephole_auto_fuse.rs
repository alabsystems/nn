// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Peephole pass 15: auto-fuse elementwise chains.
//!
//! Finds maximal chains of consecutive `CompiledStep::Dispatch` entries
//! that are single elementwise ops (each containing one `TensorOpKind::Elementwise`
//! node with a scalar `KernelDef`). Composes chains of length >= 2 into a
//! single fused `CompiledStep::Dispatch` by inlining the scalar IR nodes.
//!
//! This pass catches elementwise ops that survived as separate dispatches
//! after the pre-compilation fusion in `trace_compile_fusion.rs` — for
//! example, single-element chains (fan-out > 1 at the TraceNode level)
//! that became adjacent after other peephole passes absorbed intervening
//! materialization ops.
//!
//! Runs AFTER all named peephole passes (passes 1-11) so that specific
//! named patterns (AddLayerNorm, LinearActivation, etc.) match first.
//! Part of #3517.

use std::collections::HashMap;

use crate::ir::{IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType};
use crate::tensor_block_builder::TensorBlockBuilder;
use crate::tensor_ir::TensorOpKind;

use super::super::{CompiledKernel, CompiledStep};

/// Statistics from the auto-fuse pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct AutoFuseStats {
    /// Number of chains fused (each of length >= 2).
    pub(super) chains_fused: usize,
    /// Total number of ops absorbed into fused chains.
    pub(super) ops_fused: usize,
    /// Number of chains skipped (too short or composition failed).
    pub(super) chains_skipped: usize,
}

/// Run the auto-fuse elementwise chain pass on a compiled step sequence.
///
/// Identifies maximal chains of consecutive single-elementwise `Dispatch`
/// steps where each intermediate step has fan-out == 1, and fuses chains
/// of length >= 2 into a single `Dispatch` with a composed scalar kernel.
pub(super) fn fuse_elementwise_chains(
    steps: &mut [CompiledStep],
    use_counts: &[usize],
) -> AutoFuseStats {
    let mut stats = AutoFuseStats::default();
    let len = steps.len();

    // Detect chains: walk the step list, building maximal chains of
    // consecutive single-elementwise Dispatch steps.
    let chains = detect_chains(steps, use_counts, len);

    // Fuse each chain of length >= 2.
    for chain in chains {
        if chain.len() < 2 {
            stats.chains_skipped += 1;
            continue;
        }

        match compose_chain(steps, &chain) {
            Some(fused_step) => {
                stats.chains_fused += 1;
                stats.ops_fused += chain.len();

                // Place fused step at the last position in the chain.
                // Earlier positions become IdentityPassthrough.
                let last = chain[chain.len() - 1];
                for &idx in &chain[..chain.len() - 1] {
                    steps[idx] = CompiledStep::IdentityPassthrough;
                }
                steps[last] = fused_step;
            }
            None => {
                stats.chains_skipped += 1;
            }
        }
    }

    stats
}

/// Detect maximal chains of consecutive single-elementwise Dispatch steps.
///
/// A chain extends when:
/// 1. The step is a single-elementwise Dispatch.
/// 2. The previous chain member has fan-out of exactly 1.
/// 3. No intervening non-passthrough steps break the data dependency.
///
/// IdentityPassthrough steps between chain members are skipped.
fn detect_chains(steps: &[CompiledStep], use_counts: &[usize], len: usize) -> Vec<Vec<usize>> {
    let mut chains: Vec<Vec<usize>> = Vec::new();
    let mut in_chain = vec![false; len];

    let mut i = 0;
    while i < len {
        if in_chain[i] || !is_single_elementwise_dispatch(&steps[i]) {
            i += 1;
            continue;
        }

        // Start a new chain at position i.
        let mut chain = vec![i];
        let mut cur = i;

        // Extend the chain forward.
        let mut j = cur + 1;
        while j < len {
            // Skip IdentityPassthrough (fusion placeholders).
            if matches!(steps[j], CompiledStep::IdentityPassthrough) {
                j += 1;
                continue;
            }

            // Fan-out check: the current tail must have exactly 1 consumer.
            if use_counts.get(cur).copied().unwrap_or(0) != 1 {
                break;
            }

            if is_single_elementwise_dispatch(&steps[j]) && !in_chain[j] {
                chain.push(j);
                cur = j;
                j += 1;
            } else {
                // Hit a non-elementwise step (materialization point).
                break;
            }
        }

        if chain.len() >= 2 {
            for &idx in &chain {
                in_chain[idx] = true;
            }
            chains.push(chain);
        }

        i += 1;
    }

    chains
}

/// Check if a `CompiledStep` is a single-elementwise Dispatch.
///
/// The step must have:
/// - One or more `TensorOpKind::Input` nodes
/// - Optionally `TensorOpKind::Broadcast` nodes
/// - Exactly one `TensorOpKind::Elementwise` node (with scalar `KernelDef`)
/// - No other compute nodes
fn is_single_elementwise_dispatch(step: &CompiledStep) -> bool {
    let kernel = match step {
        CompiledStep::Dispatch { kernel, .. } => kernel,
        _ => return false,
    };

    let def = kernel.def();
    let mut elementwise_count = 0;
    for node in &def.nodes {
        match &node.kind {
            TensorOpKind::Input { .. } | TensorOpKind::Broadcast { .. } => {}
            TensorOpKind::Elementwise { .. } => elementwise_count += 1,
            _ => return false,
        }
    }
    elementwise_count == 1
}

/// Extracted info about a single-elementwise Dispatch step.
struct ElementwiseInfo {
    /// The scalar kernel (from `TensorOpKind::Elementwise`).
    scalar_kernel: KernelDef,
    /// Output shape of the elementwise op.
    output_shape: Vec<usize>,
    /// Per tensor-level input: (tensor_node_id, shape).
    input_shapes: Vec<Vec<usize>>,
    /// External graph node IDs (from CompiledStep), if available.
    external_node_ids: Option<Vec<u64>>,
}

/// Extract the scalar `KernelDef` and metadata from a single-elementwise
/// Dispatch step.
fn extract_elementwise_info(step: &CompiledStep) -> Option<ElementwiseInfo> {
    let (kernel, external_node_ids) = match step {
        CompiledStep::Dispatch {
            kernel,
            external_node_ids,
            ..
        } => (kernel, external_node_ids),
        _ => return None,
    };

    let def = kernel.def();
    let mut scalar_kernel = None;
    let mut output_shape: Vec<usize> = Vec::new();

    for node in &def.nodes {
        if let TensorOpKind::Elementwise { kernel, .. } = &node.kind {
            scalar_kernel = Some(kernel.clone());
            output_shape = node.shape.clone();
        }
    }

    let scalar_kernel = scalar_kernel?;

    // Collect tensor-level input shapes in order.
    let input_shapes: Vec<Vec<usize>> = def
        .nodes
        .iter()
        .filter_map(|n| match &n.kind {
            TensorOpKind::Input { shape, .. } => Some(shape.clone()),
            _ => None,
        })
        .collect();

    Some(ElementwiseInfo {
        scalar_kernel,
        output_shape,
        input_shapes,
        external_node_ids: external_node_ids.clone(),
    })
}

/// Compose a chain of elementwise steps into a single fused `CompiledStep`.
fn compose_chain(steps: &[CompiledStep], chain: &[usize]) -> Option<CompiledStep> {
    if chain.len() < 2 {
        return None;
    }

    let infos: Vec<ElementwiseInfo> = chain
        .iter()
        .map(|&idx| extract_elementwise_info(&steps[idx]))
        .collect::<Option<Vec<_>>>()?;

    let (composed_kernel, ext_inputs) = compose_scalar_kernels(&infos)?;

    // Build the tensor-level kernel def with the composed scalar kernel.
    let out_shape = &infos.last()?.output_shape;
    let mut b = TensorBlockBuilder::new(&composed_kernel.name);

    let mut tensor_inputs = Vec::with_capacity(ext_inputs.len());
    for (i, ext) in ext_inputs.iter().enumerate() {
        let input_node = b.add_input(&format!("input_{i}"), &ext.shape);
        if ext.shape != *out_shape {
            tensor_inputs.push(b.add_broadcast(input_node, out_shape));
        } else {
            tensor_inputs.push(input_node);
        }
    }

    let elem_out = b.add_elementwise(composed_kernel, &tensor_inputs, out_shape);
    let def = b.build(elem_out).ok()?;

    let merged_ext_ids: Vec<u64> = ext_inputs.iter().map(|e| e.graph_node_id).collect();

    Some(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: Some(merged_ext_ids),
    })
}

/// An external input to the composed kernel.
struct ExternalInput {
    graph_node_id: u64,
    shape: Vec<usize>,
}

/// Compose multiple scalar `KernelDef`s into a single fused kernel.
///
/// For a chain [A, B, C]:
/// - A's params become the first N params of the composed kernel.
/// - B's primary input (param 0) is wired to A's output node.
/// - B's remaining params become additional composed params.
/// - C's primary input is wired to B's output node.
///
/// Returns the composed kernel and the ordered list of external inputs.
fn compose_scalar_kernels(infos: &[ElementwiseInfo]) -> Option<(KernelDef, Vec<ExternalInput>)> {
    if infos.is_empty() {
        return None;
    }

    let mut composed_params: Vec<Param> = Vec::new();
    let mut composed_nodes: Vec<IRNode> = Vec::new();
    let mut external_inputs: Vec<ExternalInput> = Vec::new();

    // Track the IR node ID of the previous step's output.
    let mut prev_output: Option<NodeId> = None;

    for (step_idx, info) in infos.iter().enumerate() {
        let kernel = &info.scalar_kernel;

        // Build mapping from old Param node IDs to composed node IDs.
        // For subsequent kernels, param 0 maps to the previous output.
        let mut old_to_new: HashMap<NodeId, NodeId> = HashMap::new();

        for (param_idx, _param) in kernel.params.iter().enumerate() {
            // Find the Param node in the kernel's IR nodes.
            let old_param_node = kernel
                .nodes
                .iter()
                .find(|n| matches!(n.kind, IRNodeKind::Param(idx) if idx == param_idx));
            let old_param_id = match old_param_node {
                Some(node) => node.id,
                None => continue, // Defensive: skip if param node not found.
            };

            if step_idx > 0 && param_idx == 0 {
                // Primary input wired to previous step's output.
                old_to_new.insert(old_param_id, prev_output?);
            } else {
                // External input: emit a new Param node in the composed kernel.
                let composed_param_idx = composed_params.len();
                composed_params.push(Param::new(
                    format!("p{composed_param_idx}"),
                    ScalarType::F32,
                ));
                let new_id = NodeId::new(composed_nodes.len());
                composed_nodes.push(IRNode::new(new_id, IRNodeKind::Param(composed_param_idx)));
                old_to_new.insert(old_param_id, new_id);

                // Record external input info.
                let ext_graph_id = resolve_external_graph_id(info, param_idx);
                let ext_shape = resolve_external_shape(info, param_idx);
                external_inputs.push(ExternalInput {
                    graph_node_id: ext_graph_id,
                    shape: ext_shape,
                });
            }
        }

        // Inline non-Param IR nodes with remapped references.
        for node in &kernel.nodes {
            if matches!(node.kind, IRNodeKind::Param(_)) {
                // Already handled above.
                continue;
            }

            let new_kind = remap_ir_node_kind(&node.kind, &old_to_new);
            let new_id = NodeId::new(composed_nodes.len());
            old_to_new.insert(node.id, new_id);
            composed_nodes.push(IRNode::new(new_id, new_kind));
        }

        // Track this step's output.
        prev_output = old_to_new.get(&kernel.output).copied();
    }

    let output = prev_output?;

    // Name: "fused_{first_kernel_name}_x{chain_len}".
    let base_name = infos[0]
        .scalar_kernel
        .name
        .strip_prefix("fused_")
        .unwrap_or(&infos[0].scalar_kernel.name);
    let name = format!("fused_{base_name}_x{}", infos.len());

    let composed = KernelDef::new(
        name,
        composed_params,
        ScalarType::F32,
        composed_nodes,
        output,
    );
    if composed.validate().is_err() {
        return None;
    }

    Some((composed, external_inputs))
}

/// Resolve the graph-level node ID for an external parameter.
fn resolve_external_graph_id(info: &ElementwiseInfo, param_idx: usize) -> u64 {
    if let Some(ext_ids) = &info.external_node_ids {
        if let Some(&id) = ext_ids.get(param_idx) {
            return id;
        }
    }
    // Fallback: use param_idx as a placeholder.
    param_idx as u64
}

/// Resolve the shape of an external parameter.
fn resolve_external_shape(info: &ElementwiseInfo, param_idx: usize) -> Vec<usize> {
    if let Some(shape) = info.input_shapes.get(param_idx) {
        return shape.clone();
    }
    // Fallback to output shape.
    info.output_shape.clone()
}

/// Remap an IR node's kind, substituting old node IDs for new ones.
///
/// Exhaustive match on `IRNodeKind` — adding a new variant will cause a
/// compile error here, forcing explicit handling.
fn remap_ir_node_kind(kind: &IRNodeKind, old_to_new: &HashMap<NodeId, NodeId>) -> IRNodeKind {
    match kind {
        IRNodeKind::Param(idx) => IRNodeKind::Param(*idx),
        IRNodeKind::Literal(val) => IRNodeKind::Literal(*val),
        IRNodeKind::BinOp { op, lhs, rhs } => IRNodeKind::BinOp {
            op: *op,
            lhs: remap_id(*lhs, old_to_new),
            rhs: remap_id(*rhs, old_to_new),
        },
        IRNodeKind::UnaryFn { op, input } => IRNodeKind::UnaryFn {
            op: *op,
            input: remap_id(*input, old_to_new),
        },
        IRNodeKind::MinMax { op, lhs, rhs } => IRNodeKind::MinMax {
            op: *op,
            lhs: remap_id(*lhs, old_to_new),
            rhs: remap_id(*rhs, old_to_new),
        },
        IRNodeKind::Compare { op, lhs, rhs } => IRNodeKind::Compare {
            op: *op,
            lhs: remap_id(*lhs, old_to_new),
            rhs: remap_id(*rhs, old_to_new),
        },
        IRNodeKind::Select {
            cond,
            then_val,
            else_val,
        } => IRNodeKind::Select {
            cond: remap_id(*cond, old_to_new),
            then_val: remap_id(*then_val, old_to_new),
            else_val: remap_id(*else_val, old_to_new),
        },
        IRNodeKind::BinaryFn { op, lhs, rhs } => IRNodeKind::BinaryFn {
            op: *op,
            lhs: remap_id(*lhs, old_to_new),
            rhs: remap_id(*rhs, old_to_new),
        },
        IRNodeKind::Powi { base, exp } => IRNodeKind::Powi {
            base: remap_id(*base, old_to_new),
            exp: *exp,
        },
        IRNodeKind::Clamp { input, min, max } => IRNodeKind::Clamp {
            input: remap_id(*input, old_to_new),
            min: remap_id(*min, old_to_new),
            max: remap_id(*max, old_to_new),
        },
        IRNodeKind::SumReduce { inputs } => IRNodeKind::SumReduce {
            inputs: inputs.iter().map(|id| remap_id(*id, old_to_new)).collect(),
        },
    }
}

/// Remap a single NodeId via the mapping, falling back to the original.
fn remap_id(id: NodeId, old_to_new: &HashMap<NodeId, NodeId>) -> NodeId {
    old_to_new.get(&id).copied().unwrap_or(id)
}

#[cfg(test)]
#[path = "trace_compile_peephole_auto_fuse_tests.rs"]
mod tests;
