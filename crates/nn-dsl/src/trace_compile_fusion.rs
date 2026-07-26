// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Elementwise chain fusion for compiled trace graphs.
//!
//! Detects runs of consecutive elementwise ops on the same shape and
//! composes them into a single fused scalar `KernelDef`. This reduces
//! GPU dispatch count by collapsing chains like `exp → mul → add → relu`
//! into a **single GPU kernel launch**.
//!
//! The scalar-level composition (in [`kernel_compose`]) inlines each op's
//! IR nodes into one composed `KernelDef`, which is wrapped in a single
//! `TensorOpKind::Elementwise` node → single `DispatchStep::Elementwise`.
//!
//! Called from [`compile_trace_with_fusion()`] as a replacement for
//! the 1-to-1 [`compile_node()`](super::compile_node) mapping.

use std::collections::{HashMap, HashSet};

use nn_core::dyn_tensor::trace::{ComputationGraph, NodeId, TraceNode, TraceOp};

use crate::ir::KernelDef;
use crate::tensor_block_builder::TensorBlockBuilder;
use crate::tensor_ir::TensorIRError;

use super::{compile_node, CompiledKernel, CompiledStep};

#[path = "kernel_compose.rs"]
pub(super) mod kernel_compose;

/// Returns true if a `TraceOp` is a fusible elementwise operation.
///
/// Fusible ops: unary math/activations and binary arithmetic that
/// preserve shape (or broadcast to the output shape).
pub(super) fn is_fusible_elementwise(op: &TraceOp) -> bool {
    matches!(
        op,
        // Unary math
        TraceOp::Exp
            | TraceOp::Log
            | TraceOp::Sqrt
            | TraceOp::Sqr
            | TraceOp::Abs
            | TraceOp::Neg
            | TraceOp::Recip
            | TraceOp::Sin
            | TraceOp::Cos
            | TraceOp::Floor
            | TraceOp::Round
            | TraceOp::Fract
            | TraceOp::Tanh
            // Activations (Silu decomposes to sigmoid+mul in the builder)
            | TraceOp::Relu
            | TraceOp::Gelu
            | TraceOp::GeluErf
            | TraceOp::Sigmoid
            | TraceOp::Silu
            // Parameterized activations (decomposed in scalar IR)
            | TraceOp::LeakyRelu { .. }
            | TraceOp::Elu { .. }
            // Clamp (decomposed to MinMax for optional bounds)
            | TraceOp::Clamp { .. }
            // Power: decomposed to exp(exponent * log(x))
            | TraceOp::Powf { .. }
            // Binary arithmetic
            | TraceOp::Add
            | TraceOp::Sub
            | TraceOp::Mul
            | TraceOp::Div
            // Binary min/max
            | TraceOp::Maximum
            | TraceOp::Minimum
            // Binary trigonometric
            | TraceOp::Atan2
    )
}

/// Returns the number of external inputs for a `TraceOp`.
pub(super) fn op_input_count(op: &TraceOp) -> usize {
    match op {
        TraceOp::Add
        | TraceOp::Sub
        | TraceOp::Mul
        | TraceOp::Div
        | TraceOp::Maximum
        | TraceOp::Minimum
        | TraceOp::Atan2 => 2,
        _ => 1,
    }
}

/// Build a use-count map: for each `NodeId`, how many downstream nodes
/// reference it as an input.
fn build_use_counts(graph: &ComputationGraph) -> HashMap<NodeId, usize> {
    let mut counts: HashMap<NodeId, usize> = HashMap::new();
    for node in graph.nodes() {
        for &input_id in node.inputs() {
            *counts.entry(input_id).or_insert(0) += 1;
        }
    }
    counts
}

/// Detect fusible elementwise chains, allowing non-consecutive node indices.
///
/// Chains may span non-consecutive indices: non-fusible nodes between
/// chain members (Input, ConstantWeight, Dropout) are correctly skipped.
/// A candidate extends the chain when:
/// 1. It is a fusible elementwise op
/// 2. It has the same output shape as the chain
/// 3. The current chain tail has fan-out of exactly 1
/// 4. The candidate consumes the current chain tail as an input
fn detect_fusible_chains(
    nodes: &[TraceNode],
    use_counts: &HashMap<NodeId, usize>,
) -> Vec<Vec<usize>> {
    let mut chains = Vec::new();
    let mut in_chain = HashSet::new();

    for i in 0..nodes.len() {
        if !is_fusible_elementwise(nodes[i].op()) || in_chain.contains(&i) {
            continue;
        }
        let chain_shape = nodes[i].output_shape();
        let mut chain = vec![i];
        let mut cur = i;

        for j in (cur + 1)..nodes.len() {
            // Check consumer relationship first: once we find cur's
            // consumer, either extend the chain or stop scanning.
            // This avoids O(n²) when the consumer is non-fusible (#3243 F11).
            if nodes[j].inputs().contains(&nodes[cur].id()) {
                if use_counts.get(&nodes[cur].id()).copied().unwrap_or(0) != 1
                    || in_chain.contains(&j)
                    || !is_fusible_elementwise(nodes[j].op())
                    || nodes[j].output_shape() != chain_shape
                {
                    break; // Consumer found but can't extend chain.
                }
                chain.push(j);
                cur = j;
            }
        }

        if chain.len() >= 2 {
            // Truncate trailing [Add, Mul(scalar)] from chains to preserve
            // the pattern for resblock peephole pass (Pass 2). The peephole
            // detects `add + mul_scalar` and absorbs it into
            // FusedResBlock::residual_scale. Without truncation, the chain
            // fusion eats the `add`, hiding it from Pass 2.
            //
            // For length-2 [Add, Mul(scalar)]: truncates to [] → no chain.
            // For [Exp, Add, Mul(scalar)]: truncates to [Exp] → too short.
            // For [Add, Add, Add, Mul(scalar)]: truncates to [Add, Add] → fused pair.
            // For [Add, Add, Mul(non-scalar)]: no truncation → fused as-is.
            let chain = truncate_trailing_add_scalar_mul(chain, nodes);

            if chain.len() >= 2 {
                for &idx in &chain {
                    in_chain.insert(idx);
                }
                chains.push(chain);
            }
        }
    }

    chains
}

/// If the chain ends with `[..., Add, Mul(scalar)]`, truncate to remove
/// the trailing Add and Mul. The `Add + Mul(scalar)` pattern must remain
/// separate for resblock peephole fusion (Pass 2) to detect and absorb
/// into `FusedResBlock::residual_scale`.
///
/// Returns the (possibly shortened) chain. If the trailing pattern is not
/// found, returns the chain unmodified.
fn truncate_trailing_add_scalar_mul(chain: Vec<usize>, nodes: &[TraceNode]) -> Vec<usize> {
    if chain.len() < 2 {
        return chain;
    }
    let last = chain.len() - 1;
    let penultimate = chain.len() - 2;

    // Last element must be Mul, second-to-last must be Add.
    if !matches!(nodes[chain[last]].op(), TraceOp::Mul) {
        return chain;
    }
    if !matches!(nodes[chain[penultimate]].op(), TraceOp::Add) {
        return chain;
    }

    // Check if the Mul's non-chain input is a scalar constant.
    // mul_scalar creates TraceOp::Constant (via DynTensor::full(&[], val, ...)),
    // while auto-registered weights create TraceOp::ConstantWeight.
    let mul_inputs = nodes[chain[last]].inputs();
    let add_id = nodes[chain[penultimate]].id();
    let other_id = if mul_inputs.len() == 2 {
        if mul_inputs[0] == add_id {
            mul_inputs[1]
        } else {
            mul_inputs[0]
        }
    } else {
        return chain;
    };

    let is_scalar = nodes
        .iter()
        .any(|n| n.id() == other_id && is_scalar_constant(n.op()));

    if is_scalar {
        // Truncate: remove the trailing Add and Mul(scalar).
        chain[..penultimate].to_vec()
    } else {
        chain
    }
}

/// Returns true if the op is a scalar constant (Constant or single-element ConstantWeight).
pub(super) fn is_scalar_constant(op: &TraceOp) -> bool {
    match op {
        TraceOp::Constant { .. } => true,
        TraceOp::ConstantWeight { weight } => weight.data().len() == 1,
        _ => false,
    }
}

/// Compile a traced graph with elementwise chain fusion.
///
/// Detects fusible elementwise chains (which may span non-consecutive
/// node indices, skipping interleaved Input/ConstantWeight/Dropout nodes)
/// and fuses them into single `CompiledStep::Dispatch` steps. Non-fusible
/// ops and chain-of-1 ops are compiled individually via [`compile_node()`].
pub fn compile_trace_with_fusion(
    graph: &ComputationGraph,
) -> Result<Vec<CompiledStep>, TensorIRError> {
    // Verify topological order up front — fusion depends on nodes being
    // in dependency order.
    super::validate_topology(graph)?;

    let nodes = graph.nodes();
    let use_counts = build_use_counts(graph);
    let chains = detect_fusible_chains(nodes, &use_counts);

    // Build membership map: node_index -> is_last_in_chain.
    // Also map last-member index to chain index for retrieval.
    let mut chain_member: HashMap<usize, bool> = HashMap::new();
    let mut chain_by_last: HashMap<usize, usize> = HashMap::new();
    for (ci, chain) in chains.iter().enumerate() {
        let last_pos = chain.len() - 1;
        for (pos, &idx) in chain.iter().enumerate() {
            chain_member.insert(idx, pos == last_pos);
        }
        chain_by_last.insert(chain[last_pos], ci);
    }

    let mut steps = Vec::with_capacity(nodes.len());
    for i in 0..nodes.len() {
        match chain_member.get(&i) {
            Some(false) => {
                // Intermediate chain member — placeholder keeps step indices
                // aligned with node indices (required by CompiledModel edge_map).
                steps.push(CompiledStep::IdentityPassthrough);
            }
            Some(true) => {
                // Last chain member — compile the fused chain.
                let ci = chain_by_last[&i];
                let chain_nodes: Vec<TraceNode> =
                    chains[ci].iter().map(|&idx| nodes[idx].clone()).collect();
                let fused = compile_fused_chain(&chain_nodes, graph)?;
                steps.push(fused);
            }
            None => {
                // Not in any chain — compile individually.
                steps.push(compile_node(&nodes[i], graph)?);
            }
        }
    }

    Ok(steps)
}

/// Compile a chain of elementwise nodes into a single fused `CompiledStep`.
///
/// Uses [`kernel_compose::build_fused_scalar_kernel`] to build a single
/// composed scalar `KernelDef`, then wraps it in one `TensorOpKind::Elementwise`
/// node → single `DispatchStep::Elementwise` → single GPU kernel launch.
fn compile_fused_chain(
    chain: &[TraceNode],
    graph: &ComputationGraph,
) -> Result<CompiledStep, TensorIRError> {
    // The caller guarantees chain.len() >= 2. Verify in release builds to
    // prevent silent wrong results from degenerate single-node "fusion".
    if chain.len() < 2 {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: format!("fused chain too short ({})", chain.len()),
        });
    }

    let out_shape = chain[0].output_shape();

    // Build a single composed scalar KernelDef from the chain.
    let (composed_kernel, external_ids) = kernel_compose::build_fused_scalar_kernel(chain, graph)?;

    // Wrap the composed scalar kernel in a single TensorOpKind::Elementwise
    // node via TensorBlockBuilder. This produces one dispatch step.
    let mut b = TensorBlockBuilder::new(&composed_kernel.name);
    let mut tensor_inputs = Vec::with_capacity(external_ids.len());
    for (i, &ext_id) in external_ids.iter().enumerate() {
        let input_shape = graph
            .node(ext_id)
            .map(TraceNode::output_shape)
            .ok_or_else(|| TensorIRError::MissingInputNode {
                node_name: "fused_chain".into(),
                input_idx: i,
                input_id: ext_id,
            })?;
        let input_node = b.add_input(&format!("input_{i}"), input_shape);
        // Insert broadcast for external inputs with shapes smaller than the
        // chain output (e.g., [1,1,8] → [1,4,8] for broadcast binary ops).
        if input_shape != out_shape {
            tensor_inputs.push(b.add_broadcast(input_node, out_shape));
        } else {
            tensor_inputs.push(input_node);
        }
    }
    let elem_out = b.add_elementwise(composed_kernel, &tensor_inputs, out_shape);
    let def = b.build(elem_out)?;

    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: Some(external_ids),
    })
}

// -- Fusion chain detection for verification ----------------------------------

/// A pairwise fusion specification ready for NY verification.
///
/// Contains the fused and individual scalar `KernelDef`s with pre-computed
/// parameter mappings. Corresponds to one consecutive pair in a fused chain.
#[derive(Debug, Clone)]
pub struct FusionPair {
    /// The 2-op fused kernel.
    pub fused: KernelDef,
    /// First individual kernel.
    pub first: KernelDef,
    /// Second individual kernel.
    pub second: KernelDef,
    /// Maps first's params to fused's param indices.
    pub first_param_indices: Vec<usize>,
    /// Maps second's params to fused's param indices.
    /// The entry at `second_input_from_first` is a placeholder (ignored by FusionSpec).
    pub second_param_indices: Vec<usize>,
    /// Which param of second takes first's output.
    pub second_input_from_first: usize,
}

/// Information about a detected fusion chain for verification.
///
/// For a chain of N ops, contains N-1 pairwise [`FusionPair`]s. Each pair
/// proves that fusing two consecutive ops is equivalent to running them
/// sequentially. The full chain correctness follows by induction.
#[derive(Debug, Clone)]
pub struct FusionChainInfo {
    /// Pairwise fusion data for each consecutive pair in the chain.
    pub pairs: Vec<FusionPair>,
    /// Chain name (e.g., "fused_exp_x3").
    pub chain_name: String,
    /// Number of ops in the chain.
    pub chain_len: usize,
}

/// Detect fusible elementwise chains and extract kernel info for verification.
///
/// Uses the same chain detection algorithm as [`compile_trace_with_fusion`],
/// but instead of compiling to GPU dispatch steps, extracts the scalar
/// `KernelDef`s and parameter mappings needed for NY fusion
/// equivalence proofs.
///
/// # Errors
///
/// Returns `TensorIRError` if the graph fails topology validation or
/// kernel building fails for any chain.
pub fn detect_fusion_chains(
    graph: &ComputationGraph,
) -> Result<Vec<FusionChainInfo>, TensorIRError> {
    super::validate_topology(graph)?;

    let nodes = graph.nodes();
    let use_counts = build_use_counts(graph);
    let detected = detect_fusible_chains(nodes, &use_counts);
    let mut chains = Vec::new();

    for chain_indices in &detected {
        let chain_nodes: Vec<TraceNode> = chain_indices
            .iter()
            .map(|&idx| nodes[idx].clone())
            .collect();
        let info = build_chain_info(&chain_nodes, graph)?;
        chains.push(info);
    }

    Ok(chains)
}

/// Build [`FusionChainInfo`] for a detected chain of elementwise ops.
fn build_chain_info(
    chain: &[TraceNode],
    graph: &ComputationGraph,
) -> Result<FusionChainInfo, TensorIRError> {
    let chain_name = format!("fused_{}_x{}", chain[0].op().canonical_name(), chain.len());
    let mut pairs = Vec::with_capacity(chain.len() - 1);

    for j in 0..chain.len() - 1 {
        let pair_chain = &chain[j..=j + 1];
        let first_chain = std::slice::from_ref(&chain[j]);
        let second_chain = std::slice::from_ref(&chain[j + 1]);

        // Build the 2-op fused kernel and individual single-op kernels.
        let (fused_kernel, fused_ext_ids) =
            kernel_compose::build_fused_scalar_kernel(pair_chain, graph)?;
        let (first_kernel, first_ext_ids) =
            kernel_compose::build_fused_scalar_kernel(first_chain, graph)?;
        let (second_kernel, second_ext_ids) =
            kernel_compose::build_fused_scalar_kernel(second_chain, graph)?;

        // Map TraceNodeId -> fused param index for quick lookup.
        let fused_ext_map: HashMap<NodeId, usize> = fused_ext_ids
            .iter()
            .enumerate()
            .map(|(idx, &nid)| (nid, idx))
            .collect();

        // first_param_indices: each of first's params -> fused param index.
        let first_param_indices: Vec<usize> = first_ext_ids
            .iter()
            .map(|nid| {
                fused_ext_map
                    .get(nid)
                    .copied()
                    .ok_or_else(|| TensorIRError::UnsupportedTraceOp {
                        name: format!(
                            "first kernel external input not in fused kernel for {chain_name}"
                        ),
                    })
            })
            .collect::<Result<_, _>>()?;

        // second_param_indices + second_input_from_first.
        let first_output_id = chain[j].id();
        let mut second_input_from_first = None;
        let mut second_param_indices: Vec<usize> = Vec::with_capacity(second_ext_ids.len());

        for (k, &nid) in second_ext_ids.iter().enumerate() {
            if nid == first_output_id {
                second_input_from_first = Some(k);
                // Placeholder — ignored at this index by FusionSpec.
                second_param_indices.push(0);
            } else {
                let fused_idx = fused_ext_map.get(&nid).copied().ok_or_else(|| {
                    TensorIRError::UnsupportedTraceOp {
                        name: format!(
                            "second kernel external input not in fused kernel for {chain_name}"
                        ),
                    }
                })?;
                second_param_indices.push(fused_idx);
            }
        }

        let second_input_from_first =
            second_input_from_first.ok_or_else(|| TensorIRError::UnsupportedTraceOp {
                name: format!("second kernel has no input from first in chain {chain_name}"),
            })?;

        pairs.push(FusionPair {
            fused: fused_kernel,
            first: first_kernel,
            second: second_kernel,
            first_param_indices,
            second_param_indices,
            second_input_from_first,
        });
    }

    Ok(FusionChainInfo {
        pairs,
        chain_name,
        chain_len: chain.len(),
    })
}

#[cfg(test)]
#[path = "trace_compile_fusion_tests.rs"]
mod tests;
