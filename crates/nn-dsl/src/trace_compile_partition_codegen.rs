// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Partition-driven codegen: compiles `PartitionGroup`s into fused `CompiledStep`s.
//!
//! Takes the output of [`partition_graph`](super::trace_compile_partition::partition_graph)
//! and generates actual compiled steps, replacing the per-node 1:1 compilation
//! with fused group compilation where possible.
//!
//! Codegen strategy per group category:
//! - **Elementwise**: Chain 2+ fusible element-wise ops into a single fused scalar
//!   kernel via `kernel_compose::build_fused_scalar_kernel`.
//! - **Opaque/Reduction/Broadcast/Native**: Passthrough — all base steps preserved.
//!   The peephole passes that run after partition codegen handle Opaque fusion
//!   patterns (NormActivConv1d, LinearActivation, etc.) more safely.
//!
//! Also provides [`PartitionPlan`] for batch-encodable dispatch grouping and
//! [`resolve_partition_edge_map`] for cross-partition edge resolution using
//! `external_node_ids`. Part of #4283.
//!
//! Design reference: `designs/2026-03-22-graph-partitioning-beyond-peephole.md`

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode};

use crate::tensor_block_builder::TensorBlockBuilder;
use crate::tensor_ir::TensorIRError;

use super::trace_compile_classify::FusionCategory;
use super::trace_compile_partition::{PartitionGroup, PartitionResult};
use super::{CompiledKernel, CompiledStep};

use super::fusion::is_fusible_elementwise;

/// A batch of partition groups that can be encoded in parallel.
///
/// All groups in a batch have no dependencies on each other — only
/// on groups in earlier batches.
#[derive(Debug)]
pub(crate) struct PartitionBatch {
    /// Indices into `PartitionResult::groups` for groups in this batch.
    pub(crate) group_indices: Vec<usize>,
}

/// A plan for batch-encodable dispatch of partition groups.
///
/// Groups are organized into sequential batches where each batch's groups
/// depend only on groups in earlier batches (topological layering).
#[derive(Debug)]
pub(crate) struct PartitionPlan {
    /// Batches in dependency order (batch 0 has no deps, batch 1 depends
    /// only on batch 0, etc.).
    pub(crate) batches: Vec<PartitionBatch>,
    /// Total number of dispatches across all batches.
    pub(crate) total_dispatches: usize,
}

/// Build a `PartitionPlan` from a `PartitionResult` using Kahn's algorithm
/// for topological layering.
///
/// Groups with no dependencies go into batch 0. Groups whose dependencies
/// are all in earlier batches go into the earliest possible batch.
pub(crate) fn build_partition_plan(partition: &PartitionResult) -> PartitionPlan {
    let n = partition.groups.len();
    if n == 0 {
        return PartitionPlan {
            batches: vec![],
            total_dispatches: 0,
        };
    }

    // Compute in-degree for each group: number of dependencies (incoming edges).
    let mut in_degree: Vec<usize> = partition
        .group_edge_map
        .iter()
        .map(Vec::len)
        .collect();

    // Build reverse adjacency: rdeps[i] = groups that depend on i.
    let mut rdeps: Vec<Vec<usize>> = vec![vec![]; n];
    for (gi, deps) in partition.group_edge_map.iter().enumerate() {
        for &dep in deps {
            rdeps[dep].push(gi);
        }
    }

    // Kahn's algorithm with layer tracking.
    let mut batches: Vec<PartitionBatch> = Vec::new();
    let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();

    while !queue.is_empty() {
        let batch_indices = std::mem::take(&mut queue);
        for &gi in &batch_indices {
            for &dependent in &rdeps[gi] {
                in_degree[dependent] -= 1;
                if in_degree[dependent] == 0 {
                    queue.push(dependent);
                }
            }
        }
        batches.push(PartitionBatch {
            group_indices: batch_indices,
        });
    }

    PartitionPlan {
        total_dispatches: n,
        batches,
    }
}

/// Resolve cross-partition edges into a step-level edge map.
///
/// Uses `external_node_ids` from `CompiledStep::Dispatch` and `NativeOp`
/// variants to map graph-level input edges to step indices. For fused
/// groups (where intermediate nodes become `IdentityPassthrough`), edges
/// are redirected to the group's output step.
///
/// Returns a `Vec<Vec<usize>>` where `result[step_i]` lists the step
/// indices that step `step_i` depends on (its inputs).
pub(crate) fn resolve_partition_edge_map(
    graph: &ComputationGraph,
    steps: &[CompiledStep],
    partition: &PartitionResult,
) -> Vec<Vec<usize>> {
    let nodes = graph.nodes();
    let n = nodes.len();

    // Build node_id -> node_index map.
    let id_to_idx: HashMap<u64, usize> =
        nodes.iter().enumerate().map(|(i, n)| (n.id(), i)).collect();

    // Build node_index -> group_index map.
    let mut node_to_group: HashMap<usize, usize> = HashMap::new();
    for (gi, group) in partition.groups.iter().enumerate() {
        for &node_idx in &group.nodes {
            node_to_group.insert(node_idx, gi);
        }
    }

    // For each group, find the output step (last node in topological order).
    let group_output_step: Vec<usize> = partition
        .groups
        .iter()
        .map(|g| *g.nodes.last().expect("non-empty group"))
        .collect();

    // Build edge map for each step.
    let mut edge_map: Vec<Vec<usize>> = Vec::with_capacity(n);

    for (step_idx, step) in steps.iter().enumerate() {
        let ext_ids: Option<&[u64]> = match step {
            CompiledStep::Dispatch {
                external_node_ids, ..
            } => external_node_ids.as_deref(),
            _ => None,
        };

        if let Some(ids) = ext_ids {
            let mut deps: Vec<usize> = Vec::new();
            for &node_id in ids {
                if let Some(&node_idx) = id_to_idx.get(&node_id) {
                    // If the referenced node was absorbed into a fused group
                    // (its step became IdentityPassthrough), redirect to the
                    // group's output step. For passthrough groups where base
                    // steps are preserved, use node_idx directly.
                    let target = if node_idx < steps.len()
                        && matches!(steps[node_idx], CompiledStep::IdentityPassthrough)
                    {
                        if let Some(&gi) = node_to_group.get(&node_idx) {
                            group_output_step[gi]
                        } else {
                            node_idx
                        }
                    } else {
                        node_idx
                    };
                    if target != step_idx && !deps.contains(&target) {
                        deps.push(target);
                    }
                }
            }
            deps.sort_unstable();
            edge_map.push(deps);
        } else {
            // Steps without external_node_ids: fall back to graph topology.
            if step_idx < nodes.len() {
                let mut deps: Vec<usize> = Vec::new();
                for &input_id in nodes[step_idx].inputs() {
                    if let Some(&input_idx) = id_to_idx.get(&input_id) {
                        let target = if input_idx < steps.len()
                            && matches!(steps[input_idx], CompiledStep::IdentityPassthrough)
                        {
                            if let Some(&gi) = node_to_group.get(&input_idx) {
                                group_output_step[gi]
                            } else {
                                input_idx
                            }
                        } else {
                            input_idx
                        };
                        if target != step_idx && !deps.contains(&target) {
                            deps.push(target);
                        }
                    }
                }
                deps.sort_unstable();
                edge_map.push(deps);
            } else {
                edge_map.push(vec![]);
            }
        }
    }

    edge_map
}

/// Compile partition groups into a sequence of `CompiledStep`s.
///
/// For each partition group, determines the codegen strategy based on its
/// `FusionCategory`. Nodes within a multi-node group that are not the
/// "output" step are replaced with `IdentityPassthrough` to keep step
/// indices aligned with graph node indices (required by the executor's
/// edge_map).
///
/// # Arguments
///
/// - `groups`: Partition groups from `partition_graph()`, in topological order.
/// - `graph`: The (constant-folded) computation graph.
/// - `base_steps`: Per-node compiled steps from `compile_trace()`.
///
/// # Errors
///
/// Returns `TensorIRError` if kernel building fails for any fused group.
pub(crate) fn compile_partition_groups(
    groups: &[PartitionGroup],
    graph: &ComputationGraph,
    base_steps: &[CompiledStep],
) -> Result<Vec<CompiledStep>, TensorIRError> {
    // Build a map: node_index -> group_index.
    let mut node_to_group: HashMap<usize, usize> = HashMap::new();
    for (gi, group) in groups.iter().enumerate() {
        for &node_idx in &group.nodes {
            node_to_group.insert(node_idx, gi);
        }
    }

    let nodes = graph.nodes();

    // Phase 1: pre-compute fused steps for groups that can actually fuse.
    // Groups that can't fuse (or fail fusion) are marked as passthrough
    // so their base steps are preserved in phase 2.
    let mut fused_group_set: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut fused_steps: HashMap<usize, CompiledStep> = HashMap::new();
    for (gi, group) in groups.iter().enumerate() {
        if group.nodes.len() <= 1 {
            continue; // Single-node groups always passthrough.
        }
        if let Some(step) = try_fuse_group(group, graph, nodes, base_steps)? {
            fused_steps.insert(gi, step);
            fused_group_set.insert(gi);
        }
        // If try_fuse_group returns None, this group is passthrough.
    }

    // Phase 2: emit steps. For fused groups, non-last nodes become
    // IdentityPassthrough and the last node emits the fused step.
    // For passthrough groups, every node preserves its base step.
    let mut steps: Vec<CompiledStep> = Vec::with_capacity(nodes.len());

    for node_idx in 0..nodes.len() {
        let gi = match node_to_group.get(&node_idx) {
            Some(&g) => g,
            None => {
                steps.push(base_step_or_identity(node_idx, base_steps));
                continue;
            }
        };

        let group = &groups[gi];

        if group.nodes.len() == 1 || !fused_group_set.contains(&gi) {
            // Single-node or passthrough group: preserve base step.
            steps.push(base_step_or_identity(node_idx, base_steps));
            continue;
        }

        // Fused group: last node gets the fused step, others get identity.
        let is_last = node_idx == *group.nodes.last().expect("non-empty group");
        if is_last {
            // Move the fused step out (only consumed once).
            steps.push(
                fused_steps
                    .remove(&gi)
                    .unwrap_or(CompiledStep::IdentityPassthrough),
            );
        } else {
            steps.push(CompiledStep::IdentityPassthrough);
        }
    }

    // Phase 3: remap external_node_ids in fused steps so they never reference
    // IdentityPassthrough steps. When partition codegen fuses a group, non-last
    // nodes become IdentityPassthrough. If another fused group's external_node_ids
    // references one of these nodes, compute_edge_map would resolve the edge to
    // the IdentityPassthrough step (no buffer), causing "input is neither a weight
    // nor a graph edge" in the executor. Fix: redirect each external ID to the
    // group's output node ID if the original target is an IdentityPassthrough step.
    // Part of #4345.
    let id_to_idx: HashMap<u64, usize> =
        nodes.iter().enumerate().map(|(i, n)| (n.id(), i)).collect();

    // Build redirect: step_idx of IdentityPassthrough -> step_idx of group output.
    let mut identity_redirect: HashMap<usize, usize> = HashMap::new();
    for (gi, group) in groups.iter().enumerate() {
        if !fused_group_set.contains(&gi) {
            continue;
        }
        let output_idx = *group.nodes.last().expect("non-empty group");
        for &node_idx in &group.nodes {
            if node_idx != output_idx {
                identity_redirect.insert(node_idx, output_idx);
            }
        }
    }

    if !identity_redirect.is_empty() {
        for step in steps.iter_mut() {
            let ext_ids = match step {
                CompiledStep::Dispatch {
                    external_node_ids: Some(ids),
                    ..
                } => ids,
                _ => continue,
            };
            let mut changed = false;
            for id in ext_ids.iter_mut() {
                if let Some(&step_idx) = id_to_idx.get(id) {
                    if let Some(&redirect_idx) = identity_redirect.get(&step_idx) {
                        *id = nodes[redirect_idx].id();
                        changed = true;
                    }
                }
            }
            // Deduplicate: two external IDs that were in the same group now
            // both point to the group output. Keep unique IDs only.
            if changed {
                ext_ids.dedup();
            }
        }
    }

    Ok(steps)
}

/// Get the base step for a node index, or IdentityPassthrough if out of bounds.
fn base_step_or_identity(node_idx: usize, base_steps: &[CompiledStep]) -> CompiledStep {
    base_steps
        .get(node_idx)
        .cloned()
        .unwrap_or(CompiledStep::IdentityPassthrough)
}

/// Try to fuse a multi-node group. Returns `Some(fused_step)` on success,
/// `None` if the group should be treated as passthrough (all base steps preserved).
fn try_fuse_group(
    group: &PartitionGroup,
    graph: &ComputationGraph,
    nodes: &[TraceNode],
    _base_steps: &[CompiledStep],
) -> Result<Option<CompiledStep>, TensorIRError> {
    match group.category {
        FusionCategory::Elementwise => try_fuse_elementwise_group(group, graph, nodes),
        // Opaque, Reduction, Broadcast, Native: passthrough (preserve all base steps).
        // The peephole passes that run after partition codegen handle Opaque
        // fusion patterns (NormActivConv1d, LinearActivation, etc.) better
        // than we can here without losing non-fused nodes.
        _ => Ok(None),
    }
}

// Compile a single partition group into a `CompiledStep`.
// ---------------------------------------------------------------------------
// Elementwise group codegen
// ---------------------------------------------------------------------------

/// Try to fuse an Elementwise-dominant group into a single fused kernel.
///
/// Filters the group's nodes to only fusible elementwise ops, then uses
/// `kernel_compose::build_fused_scalar_kernel` to produce a composed scalar
/// kernel. Returns `None` if fewer than 2 fusible ops (group becomes passthrough).
fn try_fuse_elementwise_group(
    group: &PartitionGroup,
    graph: &ComputationGraph,
    nodes: &[TraceNode],
) -> Result<Option<CompiledStep>, TensorIRError> {
    // Collect the elementwise nodes in this group (skip broadcast/metadata).
    let ew_nodes: Vec<&TraceNode> = group
        .nodes
        .iter()
        .map(|&idx| &nodes[idx])
        .filter(|n| is_fusible_elementwise(n.op()))
        .collect();

    if ew_nodes.len() < 2 {
        // Not enough fusible ops — treat group as passthrough.
        return Ok(None);
    }

    // Use the existing chain fusion infrastructure.
    let chain_nodes: Vec<TraceNode> = ew_nodes.into_iter().cloned().collect();
    compile_fused_elementwise_chain(&chain_nodes, graph).map(Some)
}

/// Compile a chain of elementwise `TraceNode`s into a single fused `CompiledStep`.
///
/// Mirrors `compile_fused_chain` from `trace_compile_fusion.rs` but takes
/// pre-filtered nodes (from a partition group, not from sequential chain
/// detection).
fn compile_fused_elementwise_chain(
    chain: &[TraceNode],
    graph: &ComputationGraph,
) -> Result<CompiledStep, TensorIRError> {
    if chain.len() < 2 {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: format!("partition fused chain too short ({})", chain.len()),
        });
    }

    let out_shape = chain.last().expect("non-empty").output_shape();

    // Build a single composed scalar KernelDef from the chain.
    let (composed_kernel, external_ids) =
        super::fusion::kernel_compose::build_fused_scalar_kernel(chain, graph)?;

    // Wrap the composed scalar kernel in a single TensorOpKind::Elementwise
    // node via TensorBlockBuilder.
    let mut b = TensorBlockBuilder::new(&composed_kernel.name);
    let mut tensor_inputs = Vec::with_capacity(external_ids.len());
    for (i, &ext_id) in external_ids.iter().enumerate() {
        let input_shape = graph
            .node(ext_id)
            .map(TraceNode::output_shape)
            .ok_or_else(|| TensorIRError::MissingInputNode {
                node_name: "partition_fused_chain".into(),
                input_idx: i,
                input_id: ext_id,
            })?;
        let input_node = b.add_input(&format!("input_{i}"), input_shape);
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

// Opaque, Reduction, Broadcast, and Native groups are all passthrough
// (handled by is_passthrough_group check in compile_partition_groups).
// The peephole passes that run after partition codegen handle Opaque fusion
// patterns (NormActivConv1d, LinearActivation, etc.) more safely.

/// Post-chain-fusion partition pass: fuse elementwise groups that chain fusion missed.
///
/// Runs after `compile_trace_with_fusion()` to find partition groups of fusible
/// elementwise ops that remain as individual Dispatch steps. Skips any group
/// where chain fusion already absorbed a member (indicated by IdentityPassthrough),
/// avoiding conflicts with already-fused multi-op kernels. Part of #4283.
pub(crate) fn apply_partition_elementwise_fusion(
    steps: &mut [CompiledStep],
    graph: &ComputationGraph,
) {
    let partition = super::trace_compile_partition::partition_graph(graph);
    let nodes = graph.nodes();

    for group in &partition.groups {
        if group.nodes.len() <= 1 {
            continue;
        }
        if group.category != FusionCategory::Elementwise {
            continue;
        }

        // Safety check: if chain fusion already absorbed any fusible elementwise
        // member of this group (turning it into IdentityPassthrough), skip the
        // entire group. The remaining Dispatch steps may be multi-op fused kernels
        // that don't match the graph node's original single-op semantics.
        let any_absorbed = group.nodes.iter().any(|&idx| {
            idx < nodes.len()
                && is_fusible_elementwise(nodes[idx].op())
                && idx < steps.len()
                && matches!(steps[idx], CompiledStep::IdentityPassthrough)
        });
        if any_absorbed {
            continue;
        }

        // Collect fusible elementwise nodes that are still individual Dispatch steps.
        let ew_indices: Vec<usize> = group
            .nodes
            .iter()
            .copied()
            .filter(|&idx| {
                idx < nodes.len()
                    && is_fusible_elementwise(nodes[idx].op())
                    && idx < steps.len()
                    && matches!(steps[idx], CompiledStep::Dispatch { .. })
            })
            .collect();

        if ew_indices.len() < 2 {
            continue;
        }

        let chain_nodes: Vec<TraceNode> =
            ew_indices.iter().map(|&idx| nodes[idx].clone()).collect();

        let fused_step = match compile_fused_elementwise_chain(&chain_nodes, graph) {
            Ok(step) => step,
            Err(_) => continue,
        };

        // Replace: last index gets the fused step, others become IdentityPassthrough.
        let last_idx = *ew_indices.last().expect("non-empty");
        for &idx in &ew_indices[..ew_indices.len() - 1] {
            steps[idx] = CompiledStep::IdentityPassthrough;
        }
        steps[last_idx] = fused_step;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nn_core::dyn_tensor::trace::{TraceNode, TraceOp, WeightRef};
    use nn_core::DType;

    use super::super::trace_compile_partition::partition_graph;

    fn build_graph(nodes: Vec<(TraceOp, Vec<usize>, Vec<usize>)>) -> ComputationGraph {
        let mut trace_nodes = Vec::new();
        for (i, (op, input_indices, shape)) in nodes.into_iter().enumerate() {
            let id = i as u64;
            let input_ids: Vec<u64> = input_indices.iter().map(|&idx| idx as u64).collect();
            let node = TraceNode::new(id, format!("node_{i}"), op, input_ids, shape, DType::F32);
            trace_nodes.push(node);
        }
        ComputationGraph::from_nodes(trace_nodes)
    }

    fn compile_base_steps(graph: &ComputationGraph) -> Vec<CompiledStep> {
        super::super::compile_trace(graph).expect("base compile should succeed")
    }

    // -- Codegen tests --------------------------------------------------------

    #[test]
    fn test_partition_elementwise_chain_fusion() {
        // Input -> Relu -> Exp -> Sigmoid: should produce single fused dispatch
        let graph = build_graph(vec![
            (TraceOp::Input, vec![], vec![1, 64]),
            (TraceOp::Relu, vec![0], vec![1, 64]),
            (TraceOp::Exp, vec![1], vec![1, 64]),
            (TraceOp::Sigmoid, vec![2], vec![1, 64]),
        ]);
        let base_steps = compile_base_steps(&graph);
        let partition = partition_graph(&graph);
        let steps = compile_partition_groups(&partition.groups, &graph, &base_steps)
            .expect("partition codegen should succeed");

        // Count non-passthrough dispatches.
        let dispatch_count = steps
            .iter()
            .filter(|s| {
                matches!(
                    s,
                    CompiledStep::Dispatch { .. } | CompiledStep::NativeOp { .. }
                )
            })
            .count();
        // Should be 1 fused dispatch (Relu+Exp+Sigmoid fused).
        assert_eq!(
            dispatch_count, 1,
            "expected 1 fused dispatch, got {dispatch_count}"
        );

        // The fused kernel name should indicate fusion.
        let fused_name = steps.iter().find_map(|s| match s {
            CompiledStep::Dispatch { kernel, .. } => {
                let name = kernel.name().to_string();
                if name.starts_with("fused_") {
                    Some(name)
                } else {
                    None
                }
            }
            _ => None,
        });
        assert!(fused_name.is_some(), "expected a fused kernel name");
    }

    #[test]
    fn test_partition_opaque_passthrough() {
        // Input -> Linear -> Relu: Opaque groups are now passthrough.
        // Base steps preserved — peephole handles LinearActivation fusion.
        let w = WeightRef::new(vec![1.0; 4], vec![2, 2]).unwrap();
        let graph = build_graph(vec![
            (TraceOp::Input, vec![], vec![1, 2]),
            (
                TraceOp::Linear {
                    weight: w,
                    bias: None,
                },
                vec![0],
                vec![1, 2],
            ),
            (TraceOp::Relu, vec![1], vec![1, 2]),
        ]);
        let base_steps = compile_base_steps(&graph);
        let partition = partition_graph(&graph);
        let steps = compile_partition_groups(&partition.groups, &graph, &base_steps)
            .expect("partition codegen should succeed");

        // Opaque group is passthrough: both Linear and Relu base steps preserved.
        assert_eq!(steps.len(), base_steps.len());
        let dispatch_count = steps
            .iter()
            .filter(|s| matches!(s, CompiledStep::Dispatch { .. }))
            .count();
        // Linear + Relu = 2 dispatches (peephole will fuse later).
        assert_eq!(
            dispatch_count, 2,
            "expected 2 dispatches for passthrough opaque"
        );
    }

    #[test]
    fn test_partition_reduction_passthrough() {
        // Input -> Mul -> ReduceSum: reduction group uses last base step
        let graph = build_graph(vec![
            (TraceOp::Input, vec![], vec![1, 64]),
            (TraceOp::Mul, vec![0, 0], vec![1, 64]),
            (
                TraceOp::ReduceSum {
                    dim: 1,
                    keepdim: false,
                },
                vec![1],
                vec![1],
            ),
        ]);
        let base_steps = compile_base_steps(&graph);
        let partition = partition_graph(&graph);
        let steps = compile_partition_groups(&partition.groups, &graph, &base_steps)
            .expect("partition codegen should succeed");

        // The reduction step should be present.
        let has_dispatch = steps
            .iter()
            .any(|s| matches!(s, CompiledStep::Dispatch { .. }));
        assert!(
            has_dispatch,
            "expected at least one dispatch for reduction group"
        );
    }

    #[test]
    fn test_partition_mixed_linear_relu_exp_linear() {
        // Input -> Linear -> Relu -> Exp -> Linear
        let w = WeightRef::new(vec![1.0; 4], vec![2, 2]).unwrap();
        let graph = build_graph(vec![
            (TraceOp::Input, vec![], vec![1, 2]),
            (
                TraceOp::Linear {
                    weight: w.clone(),
                    bias: None,
                },
                vec![0],
                vec![1, 2],
            ),
            (TraceOp::Relu, vec![1], vec![1, 2]),
            (TraceOp::Exp, vec![2], vec![1, 2]),
            (
                TraceOp::Linear {
                    weight: w,
                    bias: None,
                },
                vec![3],
                vec![1, 2],
            ),
        ]);
        let base_steps = compile_base_steps(&graph);
        let partition = partition_graph(&graph);
        let steps = compile_partition_groups(&partition.groups, &graph, &base_steps)
            .expect("partition codegen should succeed");

        // Count real dispatches (non-passthrough, non-identity).
        let real_steps: Vec<_> = steps
            .iter()
            .filter(|s| {
                !matches!(
                    s,
                    CompiledStep::IdentityPassthrough
                        | CompiledStep::InputForward
                        | CompiledStep::Passthrough { .. }
                )
            })
            .collect();
        // Should be at most 4: e.g. LinearActivation(Linear+Relu) + Exp + Linear
        assert!(
            real_steps.len() <= 4,
            "expected at most 4 real steps, got {}",
            real_steps.len()
        );
    }

    #[test]
    fn test_partition_broadcast_passthrough() {
        // Input -> Reshape -> Reshape: pure broadcast group
        let graph = build_graph(vec![
            (TraceOp::Input, vec![], vec![1, 64]),
            (
                TraceOp::Reshape {
                    target_shape: vec![64],
                },
                vec![0],
                vec![64],
            ),
            (
                TraceOp::Reshape {
                    target_shape: vec![1, 64],
                },
                vec![1],
                vec![1, 64],
            ),
        ]);
        let base_steps = compile_base_steps(&graph);
        let partition = partition_graph(&graph);
        let steps = compile_partition_groups(&partition.groups, &graph, &base_steps)
            .expect("partition codegen should succeed");

        // No GPU dispatches -- all Passthrough or IdentityPassthrough.
        let dispatch_count = steps
            .iter()
            .filter(|s| {
                matches!(
                    s,
                    CompiledStep::Dispatch { .. } | CompiledStep::NativeOp { .. }
                )
            })
            .count();
        assert_eq!(
            dispatch_count, 0,
            "expected 0 dispatches for broadcast-only graph"
        );
    }

    // -- PartitionPlan tests --------------------------------------------------

    #[test]
    fn test_partition_plan_linear_chain() {
        // Input -> Linear -> Linear: 3 groups in a chain.
        let w = WeightRef::new(vec![1.0; 4], vec![2, 2]).unwrap();
        let graph = build_graph(vec![
            (TraceOp::Input, vec![], vec![1, 2]),
            (
                TraceOp::Linear {
                    weight: w.clone(),
                    bias: None,
                },
                vec![0],
                vec![1, 2],
            ),
            (
                TraceOp::Linear {
                    weight: w,
                    bias: None,
                },
                vec![1],
                vec![1, 2],
            ),
        ]);
        let partition = partition_graph(&graph);
        let plan = build_partition_plan(&partition);

        // 3 groups in a linear chain -> 3 batches of 1 each.
        assert_eq!(plan.batches.len(), 3);
        assert_eq!(plan.total_dispatches, 3);
        for batch in &plan.batches {
            assert_eq!(batch.group_indices.len(), 1);
        }
    }

    #[test]
    fn test_partition_plan_parallel_groups() {
        // Two independent chains from one input:
        // Input -> Relu (group A) and Input -> Exp (group B)
        // Both depend only on Input, so they can be in the same batch.
        let graph = build_graph(vec![
            (TraceOp::Input, vec![], vec![1, 64]),
            (TraceOp::Relu, vec![0], vec![1, 64]),
            (TraceOp::Exp, vec![0], vec![1, 64]),
        ]);
        let partition = partition_graph(&graph);
        let plan = build_partition_plan(&partition);

        // Relu and Exp may or may not be in the same group depending on
        // partition algorithm. But the plan should have at most 2 batches.
        assert!(
            plan.batches.len() <= 2,
            "expected at most 2 batches for parallel groups, got {}",
            plan.batches.len()
        );
    }

    #[test]
    fn test_partition_plan_empty() {
        let graph = ComputationGraph::from_nodes(vec![]);
        let partition = partition_graph(&graph);
        let plan = build_partition_plan(&partition);

        assert_eq!(plan.batches.len(), 0);
        assert_eq!(plan.total_dispatches, 0);
    }

    // -- Edge map resolution tests --------------------------------------------

    #[test]
    fn test_partition_edge_map_resolution_fused_chain() {
        // Input -> Relu -> Exp -> Sigmoid: fused group.
        // The fused step should depend on step 0 (Input).
        let graph = build_graph(vec![
            (TraceOp::Input, vec![], vec![1, 64]),
            (TraceOp::Relu, vec![0], vec![1, 64]),
            (TraceOp::Exp, vec![1], vec![1, 64]),
            (TraceOp::Sigmoid, vec![2], vec![1, 64]),
        ]);
        let base_steps = compile_base_steps(&graph);
        let partition = partition_graph(&graph);
        let steps = compile_partition_groups(&partition.groups, &graph, &base_steps)
            .expect("partition codegen should succeed");

        let edge_map = resolve_partition_edge_map(&graph, &steps, &partition);

        // Step 0 (Input) has no deps.
        assert!(edge_map[0].is_empty(), "Input should have no deps");
        // The fused output step (step 3, last in the fused group) should
        // depend on step 0 (Input).
        assert!(
            edge_map[3].contains(&0),
            "fused step should depend on Input, got {:?}",
            edge_map[3]
        );
    }

    #[test]
    fn test_partition_edge_map_resolution_passthrough() {
        // Input -> Linear -> Relu: opaque passthrough (no actual fusion).
        // Relu (step 2) should depend on Linear (step 1).
        let w = WeightRef::new(vec![1.0; 4], vec![2, 2]).unwrap();
        let graph = build_graph(vec![
            (TraceOp::Input, vec![], vec![1, 2]),
            (
                TraceOp::Linear {
                    weight: w,
                    bias: None,
                },
                vec![0],
                vec![1, 2],
            ),
            (TraceOp::Relu, vec![1], vec![1, 2]),
        ]);
        let base_steps = compile_base_steps(&graph);
        let partition = partition_graph(&graph);
        let steps = compile_partition_groups(&partition.groups, &graph, &base_steps)
            .expect("partition codegen should succeed");

        let edge_map = resolve_partition_edge_map(&graph, &steps, &partition);

        // Step 0 (Input) has no deps.
        assert!(edge_map[0].is_empty(), "Input should have no deps");
        // Step 1 (Linear) depends on step 0 (Input).
        assert_eq!(
            edge_map[1],
            vec![0],
            "Linear should depend on Input, got {:?}",
            edge_map[1]
        );
        // Step 2 (Relu) depends on step 1 (Linear) — NOT redirected
        // because this is a passthrough group.
        assert_eq!(
            edge_map[2],
            vec![1],
            "Relu should depend on Linear, got {:?}",
            edge_map[2]
        );
    }

    #[test]
    fn test_partition_cross_group_redirect() {
        // Two fusible chains separated by a non-fusible op.
        // Group A: [Relu, Exp] (fused at step 2)
        // Group B (passthrough): [Linear]
        // Group C: [Sigmoid, Tanh] consuming Linear output (fused at step 5)
        //
        // Without Phase 3 redirection, group C's external_node_ids would
        // point to node 3 (Linear), which is correct (passthrough, not
        // IdentityPassthrough). This test verifies the basic flow works.
        let w = WeightRef::new(vec![1.0; 4096], vec![64, 64]).unwrap();
        let graph = build_graph(vec![
            (TraceOp::Input, vec![], vec![1, 64]), // 0: Input
            (TraceOp::Relu, vec![0], vec![1, 64]), // 1: Relu
            (TraceOp::Exp, vec![1], vec![1, 64]),  // 2: Exp
            (
                TraceOp::Linear {
                    weight: w,
                    bias: None,
                },
                vec![2],
                vec![1, 64],
            ), // 3: Linear
            (TraceOp::Sigmoid, vec![3], vec![1, 64]), // 4: Sigmoid
            (TraceOp::Tanh, vec![4], vec![1, 64]), // 5: Tanh
        ]);
        let base_steps = compile_base_steps(&graph);
        let partition = partition_graph(&graph);
        let steps = compile_partition_groups(&partition.groups, &graph, &base_steps)
            .expect("partition codegen should succeed");

        // Verify no fused step has external_node_ids pointing to IdentityPassthrough.
        for (i, step) in steps.iter().enumerate() {
            if let CompiledStep::Dispatch {
                external_node_ids: Some(ids),
                ..
            } = step
            {
                let nodes = graph.nodes();
                let id_to_idx: HashMap<u64, usize> = nodes
                    .iter()
                    .enumerate()
                    .map(|(idx, n)| (n.id(), idx))
                    .collect();
                for &ext_id in ids {
                    if let Some(&idx) = id_to_idx.get(&ext_id) {
                        assert!(
                            !matches!(steps[idx], CompiledStep::IdentityPassthrough),
                            "Step {i} has external_node_id pointing to \
                             IdentityPassthrough at step {idx}. Phase 3 \
                             redirection failed.",
                        );
                    }
                }
            }
        }

        // The edge_map from compute_edge_map should resolve correctly.
        let edge_map = crate::compute_edge_map(&graph, &steps);
        // Every non-Input step with edges should point to a step that
        // actually produces a buffer (not IdentityPassthrough).
        for (i, edges) in edge_map.iter().enumerate() {
            if matches!(
                steps[i],
                CompiledStep::IdentityPassthrough | CompiledStep::InputForward
            ) {
                continue;
            }
            for &src in edges {
                assert!(
                    !matches!(steps[src], CompiledStep::IdentityPassthrough),
                    "Step {i} edge_map references IdentityPassthrough step {src}",
                );
            }
        }
    }
}
