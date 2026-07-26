// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared edge_map builder for compiled model execution plans.
//!
//! Builds step-to-step edge mapping from graph topology, then applies
//! patches for fused operations. Used by both the buffer planner (nn-dsl)
//! and the Metal executor (nn-metal). Part of #3261.
//!
//! Callers may apply additional patches after calling [`compute_edge_map`]
//! for backend-specific needs (e.g., FusedResBlock/BatchedStyleProjection
//! for buffer lifetime analysis in the buffer planner).

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::ComputationGraph;

use crate::trace_compile::CompiledStep;

/// Compute edge_map: for each step index, the step indices that produce its inputs.
///
/// Handles:
/// - Base edges from graph topology (`TraceNode.inputs`)
/// - `external_node_ids` override for fused Dispatch and NativeOp steps (#3261)
///
/// All fused NativeOps (NormActivConv1d, AdainSnake, AdainLeakyRelu) set
/// `external_node_ids` at creation time, so no per-NativeOp-variant patches
/// are needed. The generic `external_node_ids()` method handles all of them.
///
/// **FusedResBlock and BatchedStyleProjection** bypass `resolve_input_slice`
/// (direct buffer access via `input_steps`/`style_step`). Their edges are
/// patched below so that `cast_autocast_inputs` correctly casts buffers at
/// F16↔F32 boundaries. Without this, the executor wraps F32 bytes as F16
/// (NaN/garbage). Part of #3261, #3299.
///
/// **Handled here:** `NarrowView { source_step }` and `ProjectionSlice`
/// steps from batched QKV projections (#3269). Although `ProjectionSlice`
/// reads from a thread-local temp at runtime, the edge_map must reflect
/// the source dependency for correct buffer lifetime analysis.
///
/// Unknown node IDs are silently dropped (consistent with buffer planner
/// behavior — graph inconsistencies are caught at construction time).
pub fn compute_edge_map(graph: &ComputationGraph, steps: &[CompiledStep]) -> Vec<Vec<usize>> {
    let nodes = graph.nodes();
    let id_to_idx: HashMap<u64, usize> =
        nodes.iter().enumerate().map(|(i, n)| (n.id(), i)).collect();

    let mut edge_map: Vec<Vec<usize>> = nodes
        .iter()
        .map(|node| {
            node.inputs()
                .iter()
                .filter_map(|&input_id| id_to_idx.get(&input_id).copied())
                .collect()
        })
        .collect();

    // Patch steps that carry their own external_node_ids, overriding the
    // graph-topology-based edges. Handles fused Dispatch steps (from
    // elementwise chain fusion) and NativeOps that set external_node_ids
    // at creation time (NormActivConv1d, AdainSnake, AdainLeakyRelu).
    // Part of #3261: all fused NativeOps use this generic path.
    for (step_idx, step) in steps.iter().enumerate() {
        let ext_ids: Option<&[u64]> = match step {
            CompiledStep::Dispatch {
                external_node_ids: Some(ids),
                ..
            } => Some(ids),
            CompiledStep::NativeOp { op, .. } => op.external_node_ids(),
            _ => None,
        };
        if let Some(ids) = ext_ids {
            if step_idx < edge_map.len() {
                edge_map[step_idx] = ids
                    .iter()
                    .filter_map(|&id| id_to_idx.get(&id).copied())
                    .collect();
            }
        }
    }

    // Patch NarrowView with explicit source_step and ProjectionSlice
    // steps from batched QKV projections (#3269). These read from a
    // specific source step rather than following graph topology.
    for (step_idx, step) in steps.iter().enumerate() {
        if step_idx >= edge_map.len() {
            continue;
        }
        match step {
            CompiledStep::NarrowView {
                source_step: Some(src),
                ..
            } => {
                edge_map[step_idx] = vec![*src];
            }
            CompiledStep::NativeOp {
                op: crate::NativeOpKind::ProjectionSlice { source_step, .. },
                ..
            } => {
                edge_map[step_idx] = vec![*source_step];
            }
            _ => {}
        }
    }

    // Patch FusedResBlock and BatchedStyleProjection: their executors read
    // buffers directly via `input_steps`/`style_step`, bypassing
    // `resolve_input_slice`. Without these edges, `cast_autocast_inputs`
    // won't cast the buffers at F16↔F32 boundaries. Part of #3299.
    for (step_idx, step) in steps.iter().enumerate() {
        if step_idx >= edge_map.len() {
            continue;
        }
        match step {
            CompiledStep::NativeOp {
                op:
                    crate::NativeOpKind::FusedResBlock {
                        input_steps,
                        shortcut_step,
                        pool_step,
                        ..
                    },
                ..
            } => {
                for &s in input_steps {
                    if !edge_map[step_idx].contains(&s) {
                        edge_map[step_idx].push(s);
                    }
                }
                if let Some(sc) = shortcut_step {
                    if !edge_map[step_idx].contains(sc) {
                        edge_map[step_idx].push(*sc);
                    }
                }
                if let Some(ps) = pool_step {
                    if !edge_map[step_idx].contains(ps) {
                        edge_map[step_idx].push(*ps);
                    }
                }
            }
            CompiledStep::NativeOp {
                op: crate::NativeOpKind::BatchedStyleProjection { style_step, .. },
                ..
            } => {
                if !edge_map[step_idx].contains(style_step) {
                    edge_map[step_idx].push(*style_step);
                }
            }
            // FusedResBlockChain: same direct buffer access pattern as FusedResBlock.
            // The chain reads from `input_steps` [x_step, style_step] and optionally
            // `first_shortcut_step`. Without this patch, the buffer planner frees the
            // style projection buffer before the chain executes. Part of #4264.
            CompiledStep::NativeOp {
                op:
                    crate::NativeOpKind::FusedResBlockChain {
                        input_steps,
                        first_shortcut_step,
                        ..
                    },
                ..
            } => {
                for &s in input_steps {
                    if !edge_map[step_idx].contains(&s) {
                        edge_map[step_idx].push(s);
                    }
                }
                if let Some(sc) = first_shortcut_step {
                    if !edge_map[step_idx].contains(sc) {
                        edge_map[step_idx].push(*sc);
                    }
                }
            }
            _ => {}
        }
    }

    edge_map
}
