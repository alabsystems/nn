// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Static buffer planner for compiled model execution.
//!
//! Analyzes the lifetime of intermediate buffers in a [`CompiledPlan`] and
//! assigns byte offsets into a single contiguous GPU allocation. Buffers
//! whose lifetimes don't overlap share the same memory region, reducing
//! peak GPU memory usage.
//!
//! # Algorithm
//!
//! Linear-scan register allocation adapted for GPU buffers:
//! 1. Compute the output byte size for each step.
//! 2. Determine which steps need a dedicated allocation (Dispatch,
//!    ConstantValue) vs. aliasing an existing buffer (InputForward,
//!    Passthrough, IdentityPassthrough).
//! 3. For each allocating step, find the last step that consumes its
//!    output (last-use analysis via edge_map).
//! 4. Greedily assign byte offsets, reusing freed slots when a prior
//!    buffer's last consumer has already been processed.
//!
//! # Example
//!
//! ```rust,ignore
//! let plan = compile_trace_to_plan_with_fusion(&graph)?;
//! let buffer_plan = plan_buffers(&plan, &graph);
//! // buffer_plan.total_bytes < naive sum of all intermediate sizes
//! ```

use nn_core::dyn_tensor::trace::ComputationGraph;

use crate::ir::ScalarType;
use crate::trace_compile::{CompiledPlan, CompiledStep};

// Byte-size computation extracted to buffer_planner_bytes.rs (500-line limit).
#[path = "buffer_planner_bytes.rs"]
mod bytes;
#[cfg(any(test, kani))]
use bytes::native_op_output_bytes;
use bytes::{step_output_bytes, step_output_bytes_typed};

/// A static buffer allocation plan for a compiled model.
///
/// Maps each step that needs a dedicated GPU buffer to a byte offset
/// within a single contiguous allocation of `total_bytes`.
#[derive(Clone, Debug)]
pub struct BufferPlan {
    /// Total bytes needed for the single backing allocation.
    pub total_bytes: usize,
    /// Per-step byte offset. `None` means the step aliases another
    /// step's buffer (InputForward, Passthrough, IdentityPassthrough).
    pub step_offsets: Vec<Option<usize>>,
    /// Per-step output byte size. `0` for non-allocating steps.
    pub step_sizes: Vec<usize>,
    /// Sum of all individual buffer sizes (before reuse).
    /// `total_bytes < naive_total` proves buffer reuse is working.
    pub naive_total: usize,
    /// Per-step last consumer index. `last_use[i]` is the highest step
    /// index that reads step `i`'s output. Used for eager buffer release:
    /// once a step's last consumer has executed, the intermediate buffer
    /// can be dropped to reduce peak live memory.
    pub last_use: Vec<usize>,
}

/// Build an edge map for the buffer planner's lifetime analysis.
///
/// Delegates to [`crate::edge_map::compute_edge_map`] for the shared logic
/// (base edges, external_node_ids, NormActivConv1d, AdaIN), then applies
/// buffer-planner-specific patches for NativeOps that bypass the edge_map
/// at execution time (FusedResBlock, BatchedStyleProjection). These ops
/// read buffers directly via step indices; the planner needs the edges
/// to keep those buffers alive until this step executes.
fn build_edge_map_simple(graph: &ComputationGraph, steps: &[CompiledStep]) -> Vec<Vec<usize>> {
    let mut edge_map = crate::edge_map::compute_edge_map(graph, steps);

    // Override edges for NativeOps that bypass resolve_input_slice at runtime
    // (direct buffer access via step indices). The shared compute_edge_map
    // APPENDS these edges (for autocast casting at F16/F32 boundaries), but
    // the buffer planner REPLACES with only the direct-access edges. This is
    // intentional: graph-topology edges from the fused position point to
    // IdentityPassthrough steps (zero allocation), so they don't affect
    // buffer lifetime. Only the actual buffer reads matter. (#3348 analysis)
    for (step_idx, step) in steps.iter().enumerate() {
        match step {
            // FusedResBlock reads inputs directly from buffers via input_steps.
            // Also include shortcut_step if present (conv1x1 residual path).
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
                if step_idx < edge_map.len() {
                    let mut edges = input_steps.clone();
                    if let Some(sc) = shortcut_step {
                        edges.push(*sc);
                    }
                    if let Some(ps) = pool_step {
                        edges.push(*ps);
                    }
                    edge_map[step_idx] = edges;
                }
            }
            // BatchedStyleProjection reads style embedding directly via style_step.
            CompiledStep::NativeOp {
                op: crate::NativeOpKind::BatchedStyleProjection { style_step, .. },
                ..
            } => {
                if step_idx < edge_map.len() {
                    edge_map[step_idx] = vec![*style_step];
                }
            }
            // FusedResBlockChain: same direct buffer access as FusedResBlock.
            // Reads from input_steps [x_step, style_step] and optionally
            // first_shortcut_step. Without this patch, the buffer planner
            // computes incorrect last-use for the style projection step,
            // freeing it before the chain executes. Part of #4264.
            CompiledStep::NativeOp {
                op:
                    crate::NativeOpKind::FusedResBlockChain {
                        input_steps,
                        first_shortcut_step,
                        ..
                    },
                ..
            }
                if step_idx < edge_map.len() => {
                    let mut edges = input_steps.clone();
                    if let Some(sc) = first_shortcut_step {
                        edges.push(*sc);
                    }
                    edge_map[step_idx] = edges;
                }
            _ => {}
        }
    }

    edge_map
}

/// Compute the last step index that consumes each step's output.
///
/// For step `i`, `last_use[i]` is the highest step index `j` where
/// `j > i` and step `j` has step `i` as an input. If no downstream
/// step consumes `i`'s output, `last_use[i] = i` (the step is its
/// own last user, i.e., it's a graph output or dead).
fn compute_last_use(edge_map: &[Vec<usize>], num_steps: usize) -> Vec<usize> {
    let mut last_use: Vec<usize> = (0..num_steps).collect();

    for (consumer_idx, inputs) in edge_map.iter().enumerate() {
        for &producer_idx in inputs {
            if consumer_idx > last_use[producer_idx] {
                last_use[producer_idx] = consumer_idx;
            }
        }
    }

    last_use
}

/// A free slot in the allocation pool.
#[derive(Debug)]
struct FreeSlot {
    offset: usize,
    size: usize,
}

/// Plan buffer allocation for a compiled model.
///
/// Analyzes buffer lifetimes and assigns byte offsets to minimize peak
/// GPU memory usage. Steps that don't need dedicated allocations
/// (InputForward, Passthrough, IdentityPassthrough) get `None` offsets.
///
/// # Arguments
///
/// * `plan` - The compiled execution plan.
/// * `graph` - The computation graph (for edge/dependency info).
///
/// # Returns
///
/// A [`BufferPlan`] with byte offsets and total allocation size.
pub fn plan_buffers(plan: &CompiledPlan, graph: &ComputationGraph) -> BufferPlan {
    let num_steps = plan.steps.len();
    if num_steps == 0 {
        return BufferPlan {
            total_bytes: 0,
            step_offsets: Vec::new(),
            step_sizes: Vec::new(),
            naive_total: 0,
            last_use: Vec::new(),
        };
    }

    let step_sizes: Vec<usize> = plan.steps.iter().map(step_output_bytes).collect();
    let naive_total: usize = step_sizes.iter().fold(0usize, |a, &b| a.saturating_add(b));

    let edge_map = build_edge_map_simple(graph, &plan.steps);
    let last_use = compute_last_use(&edge_map, num_steps);

    let (step_offsets, high_water_mark) = linear_scan_alloc(&step_sizes, &last_use);

    BufferPlan {
        total_bytes: high_water_mark,
        step_offsets,
        step_sizes,
        naive_total,
        last_use,
    }
}

/// Plan buffer allocation with per-step dtype awareness.
///
/// Like [`plan_buffers`], but uses `dtypes` to compute correct byte sizes
/// per step. F16/BF16 steps use 2 bytes per element, F32 steps use 4 bytes.
/// NativeOp sizes are also scaled: mixed-precision executors cast NativeOp
/// F32 output to the target dtype before storing back to the planned buffer.
///
/// `dtypes` must have the same length as `plan.steps`.
pub fn plan_buffers_with_dtypes(
    plan: &CompiledPlan,
    graph: &ComputationGraph,
    dtypes: &[ScalarType],
) -> BufferPlan {
    let num_steps = plan.steps.len();
    if num_steps == 0 {
        return BufferPlan {
            total_bytes: 0,
            step_offsets: Vec::new(),
            step_sizes: Vec::new(),
            naive_total: 0,
            last_use: Vec::new(),
        };
    }

    let step_sizes: Vec<usize> = plan
        .steps
        .iter()
        .enumerate()
        .map(|(i, step)| step_output_bytes_typed(step, dtypes.get(i).copied()))
        .collect();
    let naive_total: usize = step_sizes.iter().fold(0usize, |a, &b| a.saturating_add(b));

    let edge_map = build_edge_map_simple(graph, &plan.steps);
    let last_use = compute_last_use(&edge_map, num_steps);

    let (step_offsets, high_water_mark) = linear_scan_alloc(&step_sizes, &last_use);

    BufferPlan {
        total_bytes: high_water_mark,
        step_offsets,
        step_sizes,
        naive_total,
        last_use,
    }
}

/// Linear-scan allocation: process steps in order, assign byte offsets,
/// and free slots when a buffer's last consumer has been processed.
///
/// Returns `(step_offsets, high_water_mark)`.
fn linear_scan_alloc(step_sizes: &[usize], last_use: &[usize]) -> (Vec<Option<usize>>, usize) {
    let num_steps = step_sizes.len();
    let mut step_offsets: Vec<Option<usize>> = vec![None; num_steps];
    let mut free_slots: Vec<FreeSlot> = Vec::new();
    let mut high_water_mark: usize = 0;

    // Pre-build release map: release_at[j] lists steps whose last consumer
    // is step j. O(n) to build, O(1) lookup per step. Replaces the prior
    // O(n²) inner loop that scanned all prior steps per allocation.
    let mut release_at: Vec<Vec<usize>> = (0..num_steps).map(|_| Vec::new()).collect();
    for (step, &consumer) in last_use.iter().enumerate() {
        if consumer > step && consumer < num_steps && step_sizes[step] > 0 {
            release_at[consumer].push(step);
        }
    }

    for step_idx in 0..num_steps {
        let size = step_sizes[step_idx];
        if size == 0 {
            continue;
        }

        let offset = alloc_or_reuse(&mut free_slots, &mut high_water_mark, size);
        step_offsets[step_idx] = Some(offset);

        // Free prior buffers whose last consumer is this step.
        // Uses pre-built release_at for O(1) lookup instead of O(n) scan.
        for &prior_idx in &release_at[step_idx] {
            if let Some(prior_offset) = step_offsets[prior_idx] {
                free_slots.push(FreeSlot {
                    offset: prior_offset,
                    size: step_sizes[prior_idx],
                });
            }
        }
    }

    (step_offsets, high_water_mark)
}

/// Try to reuse a free slot (best-fit), or allocate at the high water mark.
fn alloc_or_reuse(
    free_slots: &mut Vec<FreeSlot>,
    high_water_mark: &mut usize,
    size: usize,
) -> usize {
    let best_fit = free_slots
        .iter()
        .enumerate()
        .filter(|(_, slot)| slot.size >= size)
        .min_by_key(|(_, slot)| slot.size)
        .map(|(idx, _)| idx);

    if let Some(slot_idx) = best_fit {
        let slot = free_slots.swap_remove(slot_idx);
        let remainder = slot.size - size;
        if remainder > 0 {
            free_slots.push(FreeSlot {
                offset: slot.offset.saturating_add(size),
                size: remainder,
            });
        }
        slot.offset
    } else {
        let offset = *high_water_mark;
        *high_water_mark = high_water_mark.saturating_add(size);
        offset
    }
}

#[cfg(test)]
#[path = "buffer_planner_tests.rs"]
mod tests;

#[cfg(kani)]
#[path = "kani_buffer_planner_native_ops.rs"]
mod kani_buffer_planner_native_ops;

#[cfg(kani)]
#[path = "kani_buffer_planner_step_proofs.rs"]
mod kani_buffer_planner_step_proofs;
