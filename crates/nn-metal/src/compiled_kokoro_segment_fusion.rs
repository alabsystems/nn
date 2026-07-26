// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Segment fusion planner for Kokoro Metal submit reduction.
//!
//! The Kokoro pipeline has 8 GPU segments. Between adjacent segments,
//! `GpuFence::submit_current()` submits a command buffer and the GPU
//! begins executing. When two adjacent segments have no CPU readback
//! boundary between them, they can be merged into a single fused segment
//! with one command buffer submission instead of two.
//!
//! The only hard CPU readback in the Kokoro pipeline is `step_regulate`
//! (step 4): a 4-byte prefix-sum scalar readback that determines `t_mel`.
//! All other step boundaries are soft fences (GPU-to-GPU ordering, no
//! CPU wait). This means most adjacent segment pairs are fusible.
//!
//! # Architecture
//!
//! [`SegmentFusionPlanner`] analyzes a list of [`SegmentInfo`] descriptors
//! and produces a [`FusionPlan`] grouping adjacent fusible segments. Each
//! [`FusedGroup`] in the plan represents one or more segments that can
//! share a single command buffer submission.
//!
//! # Example
//!
//! Given the Kokoro pipeline:
//! ```text
//! seg0: encode      (no readback after)
//! seg1: prosody     (no readback after)
//! seg2: regulate    (CPU readback after — hard sync)
//! seg3: f0_energy   (no readback after)
//! seg4: harmonic    (no readback after)
//! seg5: generator   (no readback after)
//! seg6: istft       (terminal)
//! ```
//!
//! The planner produces 2 fused groups:
//! - Group 0: [encode, prosody, regulate] — 1 submit (the regulate sync
//!   is internal to the group; the submit happens at the readback)
//! - Group 1: [f0_energy, harmonic, generator, istft] — 1 submit
//!
//! This reduces submits from 7 (one per segment boundary) to 2.
//!
//! Part of #4264.

use std::fmt;

/// Metadata about a single pipeline segment for fusion analysis.
///
/// The planner uses `has_cpu_readback_after` to determine whether two
/// adjacent segments can be merged. When the current segment has a CPU
/// readback after it, the next segment cannot be fused into the same
/// group because the CPU must wait for the readback result before
/// encoding the next segment's dispatches.
#[derive(Debug, Clone)]
pub struct SegmentInfo {
    /// Human-readable segment name (e.g., "encode", "regulate", "generator").
    pub name: String,
    /// Number of GPU dispatches in this segment.
    pub dispatch_count: usize,
    /// Whether this segment requires a CPU readback after execution.
    ///
    /// When `true`, the GPU must flush and the CPU must read back a result
    /// before the next segment can be encoded. This creates a hard boundary
    /// that prevents fusion with the following segment.
    ///
    /// In Kokoro, only `step_regulate` has this property (4-byte prefix-sum
    /// scalar readback for `t_mel`).
    pub has_cpu_readback_after: bool,
}

impl SegmentInfo {
    /// Create a new segment info descriptor.
    #[must_use]
    pub fn new(name: impl Into<String>, dispatch_count: usize, has_cpu_readback_after: bool) -> Self {
        Self {
            name: name.into(),
            dispatch_count,
            has_cpu_readback_after,
        }
    }
}

/// A group of adjacent segments that can share a single command buffer
/// submission.
///
/// All segments in the group are dispatched into the same lazy batch.
/// Only one `GpuFence::submit_current()` (or equivalent) is needed at
/// the end of the group instead of one per segment.
#[derive(Debug, Clone)]
pub struct FusedGroup {
    /// Indices into the original segment list (inclusive range).
    pub start_idx: usize,
    /// One past the last segment index (exclusive).
    pub end_idx: usize,
    /// Names of the segments in this group.
    pub segment_names: Vec<String>,
    /// Total dispatch count across all segments in this group.
    pub total_dispatches: usize,
    /// Whether this group ends with a CPU readback boundary.
    ///
    /// `true` when the last segment in the group has `has_cpu_readback_after`.
    /// `false` for the terminal group or groups followed by a readback-free
    /// boundary.
    pub ends_with_readback: bool,
}

impl FusedGroup {
    /// Number of segments in this group.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.end_idx - self.start_idx
    }
}

/// Result of segment fusion analysis.
///
/// Contains the fused groups and before/after metrics for diagnostic
/// reporting. The plan does not mutate the pipeline — it provides
/// metadata that the dispatcher can use to batch command buffer
/// submissions.
#[derive(Debug, Clone)]
pub struct FusionPlan {
    /// Fused segment groups. Each group represents segments that can
    /// share one command buffer submission.
    pub groups: Vec<FusedGroup>,
    /// Number of segments before fusion (original count).
    pub segments_before: usize,
    /// Number of fused groups after fusion.
    pub groups_after: usize,
    /// Expected submit reduction: `segments_before - groups_after`.
    /// This is the number of `GpuFence::submit_current()` calls
    /// eliminated by fusion.
    pub submit_reduction: usize,
}

impl FusionPlan {
    /// Whether any fusion was possible (at least one group contains
    /// more than one segment).
    #[must_use]
    pub fn has_fusion(&self) -> bool {
        self.groups.iter().any(|g| g.segment_count() > 1)
    }

    /// Total dispatch count across all groups (should equal original total).
    #[must_use]
    pub fn total_dispatches(&self) -> usize {
        self.groups.iter().map(|g| g.total_dispatches).sum()
    }
}

impl fmt::Display for FusionPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Segment Fusion Plan")?;
        writeln!(
            f,
            "  Segments: {} -> {} groups (submit reduction: {})",
            self.segments_before, self.groups_after, self.submit_reduction,
        )?;
        for (i, group) in self.groups.iter().enumerate() {
            let names = group.segment_names.join(" + ");
            let readback = if group.ends_with_readback {
                " [CPU readback]"
            } else {
                ""
            };
            writeln!(
                f,
                "  Group {i}: [{names}] ({} segments, {} dispatches){readback}",
                group.segment_count(),
                group.total_dispatches,
            )?;
        }
        Ok(())
    }
}

/// Determines whether two adjacent segments can be fused.
///
/// Fusion is possible when `seg_a` does NOT have a CPU readback after it.
/// If `seg_a` requires a CPU readback, the GPU must flush and the CPU must
/// read the result before `seg_b` can be encoded, creating a hard boundary.
///
/// # Arguments
///
/// * `seg_a` - The earlier segment (closer to pipeline input).
/// * `seg_b` - The later segment (closer to pipeline output).
///
/// # Returns
///
/// `true` if the two segments can be merged into a single fused group.
#[must_use]
pub fn can_fuse(seg_a: &SegmentInfo, _seg_b: &SegmentInfo) -> bool {
    !seg_a.has_cpu_readback_after
}

/// Analyze a sequence of segments and produce a fusion plan.
///
/// Scans the segment list left-to-right, accumulating adjacent fusible
/// segments into groups. A new group starts whenever a segment has
/// `has_cpu_readback_after == true` (the current segment becomes the
/// last in its group, and the next segment starts a new group).
///
/// # Arguments
///
/// * `segments` - Ordered pipeline segments to analyze.
///
/// # Returns
///
/// A [`FusionPlan`] with the optimal grouping. If `segments` is empty,
/// returns an empty plan with zero groups.
#[must_use]
pub fn plan_segment_fusion(segments: &[SegmentInfo]) -> FusionPlan {
    if segments.is_empty() {
        return FusionPlan {
            groups: Vec::new(),
            segments_before: 0,
            groups_after: 0,
            submit_reduction: 0,
        };
    }

    let mut groups = Vec::new();
    let mut group_start = 0;

    for i in 0..segments.len() {
        // If this segment has a CPU readback after it, or it's the last
        // segment, close the current group.
        let is_last = i == segments.len() - 1;
        let has_readback = segments[i].has_cpu_readback_after;

        if has_readback || is_last {
            let segment_names: Vec<String> = segments[group_start..=i]
                .iter()
                .map(|s| s.name.clone())
                .collect();
            let total_dispatches: usize = segments[group_start..=i]
                .iter()
                .map(|s| s.dispatch_count)
                .sum();

            groups.push(FusedGroup {
                start_idx: group_start,
                end_idx: i + 1,
                segment_names,
                total_dispatches,
                ends_with_readback: has_readback,
            });

            group_start = i + 1;
        }
    }

    let segments_before = segments.len();
    let groups_after = groups.len();
    let submit_reduction = segments_before.saturating_sub(groups_after);

    FusionPlan {
        groups,
        segments_before,
        groups_after,
        submit_reduction,
    }
}

/// Planner that wraps the Kokoro pipeline's known segment layout.
///
/// Provides a convenience method to build [`SegmentInfo`] descriptors
/// from a [`CompiledKokoro`] instance's cached segment state and
/// produce a [`FusionPlan`] reflecting the actual pipeline readback
/// boundaries.
pub struct SegmentFusionPlanner;

impl SegmentFusionPlanner {
    /// Build the standard Kokoro pipeline segment descriptors.
    ///
    /// The Kokoro pipeline has the following segments and readback points:
    ///
    /// ```text
    /// Step 1-2: encode (PlBert + TextEncoder)        -> no readback
    /// Step 3:   prosody (ProsodyPredictor)            -> no readback
    /// Step 4:   regulate (Duration + length_regulate) -> CPU readback (prefix-sum)
    /// Step 5:   f0_energy (F0EnergyPredictor)         -> no readback
    /// Step 6:   harmonic (SineGen + STFT)             -> no readback
    /// Step 7:   generator (FullDecoder)               -> no readback
    /// Step 8:   istft (GPU iSTFT -> PCM)              -> terminal
    /// ```
    ///
    /// Segments without compiled models use dispatch_count 0.
    ///
    /// # Arguments
    ///
    /// * `dispatch_counts` - Per-segment dispatch counts in pipeline order:
    ///   `[encode, prosody, regulate, f0_energy, harmonic, generator, istft]`.
    ///   Use `CompiledKokoro::dispatch_summary()` to obtain these.
    #[must_use]
    pub fn kokoro_segments(dispatch_counts: &[usize; 7]) -> Vec<SegmentInfo> {
        vec![
            SegmentInfo::new("encode", dispatch_counts[0], false),
            SegmentInfo::new("prosody", dispatch_counts[1], false),
            // regulate has the only CPU readback in the pipeline:
            // 4-byte prefix-sum scalar for t_mel determination.
            SegmentInfo::new("regulate", dispatch_counts[2], true),
            SegmentInfo::new("f0_energy", dispatch_counts[3], false),
            SegmentInfo::new("harmonic", dispatch_counts[4], false),
            SegmentInfo::new("generator", dispatch_counts[5], false),
            SegmentInfo::new("istft", dispatch_counts[6], false),
        ]
    }

    /// Analyze the Kokoro pipeline and produce a fusion plan.
    ///
    /// Convenience wrapper that builds segment descriptors from dispatch
    /// counts and runs the fusion planner.
    ///
    /// # Arguments
    ///
    /// * `dispatch_counts` - Per-segment dispatch counts (see [`kokoro_segments`]).
    #[must_use]
    pub fn plan_kokoro(dispatch_counts: &[usize; 7]) -> FusionPlan {
        let segments = Self::kokoro_segments(dispatch_counts);
        plan_segment_fusion(&segments)
    }
}

#[cfg(test)]
#[path = "compiled_kokoro_segment_fusion_tests.rs"]
mod tests;
