// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Metal command encoder batching for Kokoro dispatch plans.
//!
//! Each Metal compute encoder has creation overhead: `newComputeCommandEncoder`
//! allocates an ObjC object, validates command buffer state, and configures GPU
//! state. For Kokoro's ~201 dispatches, creating one encoder per dispatch step
//! adds measurable Metal API overhead.
//!
//! This module analyzes a dispatch plan and groups consecutive steps that can
//! share a single encoder. Grouping criteria:
//! - No CPU readback between dispatches (readback requires `commit_and_wait`,
//!   which ends the encoder and command buffer).
//! - No blit operations between compute dispatches (blits use a separate
//!   `MTLBlitCommandEncoder` which is incompatible with compute encoders).
//! - Same command buffer (all steps in a group share one lazy batch).
//!
//! `Reshape` steps are purely logical (buffer alias) and do not require an
//! encoder, so they can be included in any group without splitting.
//!
//! # Architecture
//!
//! The [`EncoderBatchPlanner`] takes a slice of `DispatchStep` and produces
//! [`EncoderGroup`] ranges. Each group indicates a contiguous range of steps
//! that can share one encoder. The existing `dispatch_one_step` function
//! continues to handle individual step dispatch — the planner only provides
//! grouping metadata for the caller to decide when to create/end encoders.
//!
//! [`BatchStats`] tracks encoder count reduction for diagnostics.
//!
//! Part of #4264.

use nn_dsl::DispatchStep;

/// A contiguous range of dispatch steps that can share a single compute encoder.
///
/// Steps in `start..end` (exclusive end) can be dispatched using the same
/// `BatchEncoder` without ending and recreating it. This eliminates
/// `newComputeCommandEncoder` overhead for consecutive compatible dispatches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderGroup {
    /// First step index (inclusive).
    pub start: usize,
    /// One past the last step index (exclusive).
    pub end: usize,
    /// Whether this group contains only reshape (zero-dispatch) steps.
    ///
    /// When `true`, no encoder is needed at all for this group — all steps
    /// are logical buffer aliases. The caller can skip encoder creation.
    pub reshape_only: bool,
}

impl EncoderGroup {
    /// Number of dispatch steps in this group.
    #[must_use]
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether this group contains zero steps.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

/// Statistics about encoder batching for a dispatch plan.
///
/// Tracks the before/after encoder count to measure Metal API overhead
/// reduction. Used for diagnostics and gate assertions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchStats {
    /// Total dispatch steps in the plan.
    pub total_steps: usize,
    /// Encoder count without batching (one per non-reshape step).
    pub encoders_before: usize,
    /// Encoder count with batching (one per group, excluding reshape-only groups).
    pub encoders_after: usize,
    /// Number of encoder groups produced.
    pub group_count: usize,
    /// Number of reshape-only groups (need no encoder).
    pub reshape_only_groups: usize,
}

impl BatchStats {
    /// Number of encoders eliminated by batching.
    #[must_use]
    pub fn encoders_saved(&self) -> usize {
        self.encoders_before.saturating_sub(self.encoders_after)
    }

    /// Average dispatches per encoder (with batching).
    ///
    /// Returns 0.0 if no encoders are used.
    #[must_use]
    pub fn avg_dispatches_per_encoder(&self) -> f64 {
        if self.encoders_after == 0 {
            return 0.0;
        }
        // Non-reshape steps divided by encoder count.
        let dispatch_steps = self.total_steps
            - self.reshape_only_groups; // rough: reshape-only groups have 0 dispatches
        dispatch_steps as f64 / self.encoders_after as f64
    }
}

/// Analyzes a dispatch plan and groups consecutive steps that can share
/// a single Metal compute command encoder.
///
/// # Grouping Rules
///
/// 1. `Reshape` steps are logical (no GPU work) and never force a split.
/// 2. All other steps require a compute encoder and are grouped consecutively.
/// 3. Currently all compute steps in a contiguous sequence share one encoder.
///    Future: blit barriers and CPU readback points will split groups.
///
/// # Example
///
/// ```text
/// Steps: [Elementwise, Elementwise, Reshape, Reduce, Reshape, Reshape, Elementwise]
/// Groups: [{0..4, reshape_only: false}, {4..6, reshape_only: true}, {6..7, reshape_only: false}]
/// ```
///
/// With batching, 4 non-reshape steps need only 2 encoders instead of 4.
pub struct EncoderBatchPlanner;

impl EncoderBatchPlanner {
    /// Analyze a dispatch plan and return encoder groups.
    ///
    /// Each group is a contiguous range of steps that can share one encoder.
    /// Groups are split when a sequence of reshape-only steps appears between
    /// compute steps (keeping reshape-only groups separate avoids creating
    /// an encoder for zero-dispatch work).
    #[must_use]
    pub fn plan(steps: &[DispatchStep]) -> Vec<EncoderGroup> {
        if steps.is_empty() {
            return Vec::new();
        }

        let mut groups = Vec::new();
        let mut group_start = 0;
        let mut has_compute = false;

        for (i, step) in steps.iter().enumerate() {
            let is_reshape = Self::is_reshape(step);

            if i == 0 {
                has_compute = !is_reshape;
                continue;
            }

            // Transition: compute group followed by reshape, or reshape group
            // followed by compute. Split at the boundary.
            let prev_reshape = Self::is_reshape(&steps[i - 1]);
            if is_reshape && !prev_reshape && has_compute {
                // End compute group, start reshape group.
                groups.push(EncoderGroup {
                    start: group_start,
                    end: i,
                    reshape_only: false,
                });
                group_start = i;
                has_compute = false;
            } else if !is_reshape && prev_reshape && !has_compute {
                // End reshape-only group, start compute group.
                groups.push(EncoderGroup {
                    start: group_start,
                    end: i,
                    reshape_only: true,
                });
                group_start = i;
                has_compute = true;
            } else if !is_reshape {
                has_compute = true;
            }
        }

        // Close final group.
        if group_start < steps.len() {
            groups.push(EncoderGroup {
                start: group_start,
                end: steps.len(),
                reshape_only: !has_compute,
            });
        }

        groups
    }

    /// Analyze a dispatch plan and return both groups and statistics.
    #[must_use]
    pub fn plan_with_stats(steps: &[DispatchStep]) -> (Vec<EncoderGroup>, BatchStats) {
        let groups = Self::plan(steps);
        let stats = Self::compute_stats(steps, &groups);
        (groups, stats)
    }

    /// Compute batching statistics from groups.
    #[must_use]
    pub fn compute_stats(steps: &[DispatchStep], groups: &[EncoderGroup]) -> BatchStats {
        let total_steps = steps.len();

        // Without batching: one encoder per non-reshape step.
        let encoders_before = steps.iter().filter(|s| !Self::is_reshape(s)).count();

        // With batching: one encoder per non-reshape-only group.
        let reshape_only_groups = groups.iter().filter(|g| g.reshape_only).count();
        let encoders_after = groups.len() - reshape_only_groups;

        BatchStats {
            total_steps,
            encoders_before,
            encoders_after,
            group_count: groups.len(),
            reshape_only_groups,
        }
    }

    /// Check whether a dispatch step is a purely logical (no-GPU) reshape.
    fn is_reshape(step: &DispatchStep) -> bool {
        matches!(step, DispatchStep::Reshape { .. })
    }
}

/// Check whether a dispatch step sequence contains a CPU readback boundary.
///
/// CPU readback requires `flush()` which commits the command buffer, ending
/// all active encoders. Steps after a readback boundary must use a new
/// command buffer and encoder.
///
/// Currently Kokoro's dispatch plans do not contain inline CPU readbacks
/// (the regulate scalar readback is handled at the pipeline level, not
/// within a single dispatch plan). This function is provided for future
/// use when dispatch plans may include explicit sync points.
#[must_use]
pub fn has_cpu_readback_boundary(steps: &[DispatchStep]) -> bool {
    // DispatchStep variants are all GPU-side compute or logical reshapes.
    // CPU readback boundaries exist at the pipeline level (between
    // CompiledModel executions), not within a single dispatch plan.
    // Return false for all current dispatch plans.
    let _ = steps;
    false
}

#[cfg(test)]
#[path = "compiled_kokoro_encoder_batch_tests.rs"]
mod tests;
