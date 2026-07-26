// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Dispatch plan optimizer with dependency-aware reordering.
//!
//! Analyzes a [`CompiledPlan`] to build a dependency graph between steps,
//! then topologically sorts with a priority heuristic that schedules steps
//! freeing buffers earliest. This reduces peak memory by allowing buffer
//! reuse sooner.
//!
//! # Segment ordering constraint
//!
//! Steps can be annotated with a segment id. Steps within the same segment
//! maintain their relative order (GPU command buffer ordering requirement).
//! Only steps in *different* segments (or unassigned) may be reordered
//! relative to each other, subject to data dependencies.
//!
//! # Algorithm
//!
//! 1. Build adjacency lists from the edge_map (step i depends on step j
//!    if j is in edge_map[i]).
//! 2. Add segment-ordering edges: within each segment, step[k] depends
//!    on step[k-1] (enforces original relative order).
//! 3. Topological sort using a priority queue. Priority: prefer steps
//!    whose execution frees the most buffer bytes (i.e., steps that are
//!    the last consumer of large buffers).
//! 4. Compute peak memory before and after reordering.
//!
//! Part of #4264 (RTF optimization).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fmt;

use nn_core::dyn_tensor::trace::ComputationGraph;

use crate::trace_compile::{CompiledPlan, CompiledStep};

/// A step identifier in the optimizer's dependency graph.
pub type StepId = usize;

/// Dependency edge: step `from` must execute before step `to`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepEdge {
    pub from: StepId,
    pub to: StepId,
}

/// Dependency graph for dispatch plan steps.
#[derive(Debug, Clone)]
pub struct DepGraph {
    /// Number of steps.
    pub num_steps: usize,
    /// Forward adjacency: successors[i] = steps that depend on step i.
    pub successors: Vec<Vec<StepId>>,
    /// Reverse adjacency: predecessors[i] = steps that step i depends on.
    pub predecessors: Vec<Vec<StepId>>,
    /// In-degree for each step (number of unresolved predecessors).
    pub in_degree: Vec<usize>,
}

impl DepGraph {
    /// Build a dependency graph from an edge map.
    ///
    /// `edge_map[i]` contains the step indices that produce step i's inputs.
    /// `segments[i]` is an optional segment id for step i. Steps within the
    /// same segment maintain relative order via synthetic edges.
    pub fn from_edge_map(edge_map: &[Vec<usize>], segments: &[Option<u32>]) -> Self {
        let n = edge_map.len();
        let mut successors = vec![Vec::new(); n];
        let mut predecessors = vec![Vec::new(); n];

        // Data-dependency edges from edge_map.
        for (step, deps) in edge_map.iter().enumerate() {
            for &dep in deps {
                if dep < n && dep != step {
                    successors[dep].push(step);
                    predecessors[step].push(dep);
                }
            }
        }

        // Segment-ordering edges: within same segment, maintain relative order.
        // Group steps by segment, then chain them.
        let mut segment_groups: HashMap<u32, Vec<StepId>> = HashMap::new();
        for (i, seg) in segments.iter().enumerate() {
            if let Some(s) = seg {
                segment_groups.entry(*s).or_default().push(i);
            }
        }
        for steps in segment_groups.values() {
            // Steps are already in original order (indices are ascending).
            let mut sorted_steps = steps.clone();
            sorted_steps.sort_unstable();
            for window in sorted_steps.windows(2) {
                let prev = window[0];
                let next = window[1];
                // prev must execute before next (segment ordering).
                if !predecessors[next].contains(&prev) {
                    successors[prev].push(next);
                    predecessors[next].push(prev);
                }
            }
        }

        // Deduplicate adjacency lists.
        for list in &mut successors {
            list.sort_unstable();
            list.dedup();
        }
        for list in &mut predecessors {
            list.sort_unstable();
            list.dedup();
        }

        let in_degree: Vec<usize> = predecessors.iter().map(Vec::len).collect();

        Self {
            num_steps: n,
            successors,
            predecessors,
            in_degree,
        }
    }
}

/// Priority entry for the topological sort heap.
///
/// Higher `freed_bytes` means this step should be scheduled first
/// (it unblocks the most memory for reuse).
#[derive(Debug, Clone, Eq, PartialEq)]
struct PriorityEntry {
    step: StepId,
    /// Bytes freed when this step completes (sum of input buffers
    /// whose last consumer is this step).
    freed_bytes: usize,
    /// Original index for stability (lower = earlier in original plan).
    original_index: usize,
}

impl Ord for PriorityEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Primary: more freed bytes first.
        self.freed_bytes
            .cmp(&other.freed_bytes)
            // Secondary: earlier original index first (stability).
            .then_with(|| other.original_index.cmp(&self.original_index))
    }
}

impl PartialOrd for PriorityEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Result of the dispatch plan optimization.
#[derive(Debug, Clone)]
pub struct OptimizedPlan {
    /// Reordered step indices (index into original plan's steps).
    pub reordered_indices: Vec<StepId>,
    /// Number of steps that changed position.
    pub reorder_count: usize,
    /// Peak live memory (bytes) in the original order.
    pub peak_memory_before: usize,
    /// Peak live memory (bytes) in the optimized order.
    pub peak_memory_after: usize,
    /// Total steps in the plan.
    pub total_steps: usize,
}

impl OptimizedPlan {
    /// Memory savings in bytes (0 if optimization increased memory).
    #[must_use]
    pub fn memory_saved(&self) -> usize {
        self.peak_memory_before
            .saturating_sub(self.peak_memory_after)
    }

    /// Memory savings as a percentage of original peak.
    #[must_use]
    pub fn savings_pct(&self) -> f64 {
        if self.peak_memory_before == 0 {
            return 0.0;
        }
        let saved = self
            .peak_memory_before
            .saturating_sub(self.peak_memory_after);
        (saved as f64 / self.peak_memory_before as f64) * 100.0
    }
}

impl fmt::Display for OptimizedPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Dispatch Plan Optimization:")?;
        writeln!(f, "  steps: {}", self.total_steps)?;
        writeln!(f, "  reordered: {}", self.reorder_count)?;
        writeln!(
            f,
            "  peak memory before: {:.2} MB",
            self.peak_memory_before as f64 / (1024.0 * 1024.0)
        )?;
        writeln!(
            f,
            "  peak memory after:  {:.2} MB",
            self.peak_memory_after as f64 / (1024.0 * 1024.0)
        )?;
        write!(f, "  savings: {:.1}%", self.savings_pct())
    }
}

/// Estimate output buffer size for a step (in bytes, assuming f32).
///
/// Returns 0 for non-allocating steps (Passthrough, InputForward,
/// IdentityPassthrough, NarrowView).
fn estimate_step_bytes(step: &CompiledStep) -> usize {
    match step {
        CompiledStep::Dispatch { kernel, .. } => kernel
            .output_shape()
            .map(|s| s.iter().product::<usize>() * 4)
            .unwrap_or(0),
        CompiledStep::NativeOp { op, .. } => {
            // NativeOps have varying output sizes. Use a conservative estimate.
            op.estimated_metal_dispatches() * 4096
        }
        CompiledStep::ConstantValue { shape, .. } => shape.iter().product::<usize>() * 4,
        // Non-allocating steps.
        CompiledStep::Passthrough { .. }
        | CompiledStep::InputForward
        | CompiledStep::IdentityPassthrough
        | CompiledStep::NarrowView { .. }
        | CompiledStep::RuntimeOp { .. } => 0,
    }
}

/// Compute last-use map: for each step, the set of steps for which it
/// is the last consumer.
fn compute_last_use_map(edge_map: &[Vec<usize>], order: &[StepId]) -> Vec<usize> {
    let n = edge_map.len();
    // last_consumer[i] = the step index (in `order`) that is the last
    // to consume step i's output.
    let mut last_consumer = vec![0usize; n];

    // Build position map: step -> position in execution order.
    let mut position = vec![0usize; n];
    for (pos, &step) in order.iter().enumerate() {
        if step < n {
            position[step] = pos;
        }
    }

    // For each step in `order`, update last_consumer for its inputs.
    for &step in order {
        if step >= n {
            continue;
        }
        for &dep in &edge_map[step] {
            if dep < n && position[step] > position[dep] {
                if position[step] > last_consumer[dep] {
                    last_consumer[dep] = position[step];
                }
            }
        }
    }

    last_consumer
}

/// Compute peak live memory for a given execution order.
fn compute_peak_memory(steps: &[CompiledStep], edge_map: &[Vec<usize>], order: &[StepId]) -> usize {
    let n = steps.len();
    if n == 0 {
        return 0;
    }

    let step_bytes: Vec<usize> = steps.iter().map(estimate_step_bytes).collect();
    let last_use = compute_last_use_map(edge_map, order);

    let mut position = vec![0usize; n];
    for (pos, &step) in order.iter().enumerate() {
        if step < n {
            position[step] = pos;
        }
    }

    let mut live_bytes = 0usize;
    let mut peak_bytes = 0usize;
    let mut alive = vec![false; n];

    for (exec_pos, &step) in order.iter().enumerate() {
        if step >= n {
            continue;
        }

        // Allocate this step's output.
        if step_bytes[step] > 0 {
            alive[step] = true;
            live_bytes = live_bytes.saturating_add(step_bytes[step]);
        }

        if live_bytes > peak_bytes {
            peak_bytes = live_bytes;
        }

        // Free buffers whose last consumer is at this execution position.
        for buf_idx in 0..n {
            if alive[buf_idx] && last_use[buf_idx] <= exec_pos {
                alive[buf_idx] = false;
                live_bytes = live_bytes.saturating_sub(step_bytes[buf_idx]);
            }
        }
    }

    peak_bytes
}

/// Compute bytes freed by executing a given step.
///
/// A buffer is freed when its last consumer executes. This function
/// returns the total bytes of buffers for which `step` is the last
/// consumer in the given edge_map.
fn compute_freed_bytes(
    step: StepId,
    edge_map: &[Vec<usize>],
    step_bytes: &[usize],
    last_consumer: &[StepId],
) -> usize {
    let n = edge_map.len();
    let mut freed = 0usize;

    // Check all buffers: if `step` is the last consumer, we free them.
    for buf in 0..n {
        if last_consumer[buf] == step && step_bytes[buf] > 0 {
            freed = freed.saturating_add(step_bytes[buf]);
        }
    }

    freed
}

/// Build an edge map from a `CompiledPlan` and its source `ComputationGraph`.
///
/// Delegates to [`crate::edge_map::compute_edge_map`].
pub fn build_optimizer_edge_map(plan: &CompiledPlan, graph: &ComputationGraph) -> Vec<Vec<usize>> {
    crate::edge_map::compute_edge_map(graph, &plan.steps)
}

/// Optimize a dispatch plan by reordering steps to minimize peak memory.
///
/// Uses a topological sort with a priority heuristic: prefer scheduling
/// steps that free the most buffer bytes (last consumer of large buffers).
///
/// # Arguments
///
/// * `plan` - The compiled plan to optimize.
/// * `graph` - The computation graph that produced the plan.
/// * `segments` - Optional segment assignment per step. Steps within the
///   same segment maintain relative order. `None` entries are unconstrained.
///
/// The plan's `output_step` is always scheduled last.
pub fn optimize_dispatch_plan(
    plan: &CompiledPlan,
    graph: &ComputationGraph,
    segments: &[Option<u32>],
) -> OptimizedPlan {
    let n = plan.steps.len();
    if n == 0 {
        return OptimizedPlan {
            reordered_indices: Vec::new(),
            reorder_count: 0,
            peak_memory_before: 0,
            peak_memory_after: 0,
            total_steps: 0,
        };
    }

    let edge_map = build_optimizer_edge_map(plan, graph);
    let dep_graph = DepGraph::from_edge_map(&edge_map, segments);

    let step_bytes: Vec<usize> = plan.steps.iter().map(estimate_step_bytes).collect();

    // Original order for peak-memory baseline.
    let original_order: Vec<StepId> = (0..n).collect();
    let peak_before = compute_peak_memory(&plan.steps, &edge_map, &original_order);

    // Compute last-consumer for each buffer (in original order, for the
    // heuristic). We'll recompute for actual freed bytes in the priority.
    let mut last_consumer_of = vec![0usize; n];
    for step in 0..n {
        for &dep in &edge_map[step] {
            if dep < n && step > last_consumer_of[dep] {
                last_consumer_of[dep] = step;
            }
        }
    }

    // Priority-based topological sort.
    let mut in_degree = dep_graph.in_degree.clone();
    let mut heap = BinaryHeap::new();
    let mut result_order = Vec::with_capacity(n);

    // Seed the heap with steps that have no predecessors.
    for step in 0..n {
        if in_degree[step] == 0 {
            let freed = compute_freed_bytes(step, &edge_map, &step_bytes, &last_consumer_of);
            heap.push(PriorityEntry {
                step,
                freed_bytes: freed,
                original_index: step,
            });
        }
    }

    while let Some(entry) = heap.pop() {
        let step = entry.step;
        result_order.push(step);

        // Release successors.
        for &succ in &dep_graph.successors[step] {
            in_degree[succ] = in_degree[succ].saturating_sub(1);
            if in_degree[succ] == 0 {
                let freed = compute_freed_bytes(succ, &edge_map, &step_bytes, &last_consumer_of);
                heap.push(PriorityEntry {
                    step: succ,
                    freed_bytes: freed,
                    original_index: succ,
                });
            }
        }
    }

    // If we didn't schedule all steps, there's a cycle — fall back to
    // original order for unscheduled steps.
    if result_order.len() < n {
        let scheduled: std::collections::HashSet<StepId> = result_order.iter().copied().collect();
        for step in 0..n {
            if !scheduled.contains(&step) {
                result_order.push(step);
            }
        }
    }

    let peak_after = compute_peak_memory(&plan.steps, &edge_map, &result_order);

    // Count reordered steps.
    let reorder_count = result_order
        .iter()
        .enumerate()
        .filter(|(pos, &step)| *pos != step)
        .count();

    OptimizedPlan {
        reordered_indices: result_order,
        reorder_count,
        peak_memory_before: peak_before,
        peak_memory_after: peak_after,
        total_steps: n,
    }
}

/// Optimize a dispatch plan without segment constraints.
///
/// Convenience wrapper for [`optimize_dispatch_plan`] with all steps
/// unconstrained (no segment ordering requirement).
pub fn optimize_dispatch_plan_unconstrained(
    plan: &CompiledPlan,
    graph: &ComputationGraph,
) -> OptimizedPlan {
    let segments = vec![None; plan.steps.len()];
    optimize_dispatch_plan(plan, graph, &segments)
}

#[cfg(test)]
#[path = "dispatch_plan_optimizer_tests.rs"]
mod tests;
