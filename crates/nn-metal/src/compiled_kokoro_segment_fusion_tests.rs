// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for segment fusion planner.
//!
//! Part of #4264.

use super::*;

/// All segments fusible (no CPU readbacks) -> 1 fused group.
#[test]
fn test_all_segments_fusible_single_group() {
    let segments = vec![
        SegmentInfo::new("a", 10, false),
        SegmentInfo::new("b", 20, false),
        SegmentInfo::new("c", 15, false),
        SegmentInfo::new("d", 25, false),
    ];

    let plan = plan_segment_fusion(&segments);

    assert_eq!(plan.segments_before, 4);
    assert_eq!(plan.groups_after, 1);
    assert_eq!(plan.submit_reduction, 3);
    assert!(plan.has_fusion());

    assert_eq!(plan.groups.len(), 1);
    let g = &plan.groups[0];
    assert_eq!(g.start_idx, 0);
    assert_eq!(g.end_idx, 4);
    assert_eq!(g.segment_count(), 4);
    assert_eq!(g.total_dispatches, 70);
    assert!(!g.ends_with_readback);
    assert_eq!(g.segment_names, vec!["a", "b", "c", "d"]);
}

/// No segments fusible (all have readbacks) -> each segment is its own group.
#[test]
fn test_no_segments_fusible_all_readbacks() {
    let segments = vec![
        SegmentInfo::new("a", 10, true),
        SegmentInfo::new("b", 20, true),
        SegmentInfo::new("c", 15, true),
        SegmentInfo::new("d", 25, true),
    ];

    let plan = plan_segment_fusion(&segments);

    assert_eq!(plan.segments_before, 4);
    assert_eq!(plan.groups_after, 4);
    assert_eq!(plan.submit_reduction, 0);
    assert!(!plan.has_fusion());

    for (i, g) in plan.groups.iter().enumerate() {
        assert_eq!(g.start_idx, i);
        assert_eq!(g.end_idx, i + 1);
        assert_eq!(g.segment_count(), 1);
        assert!(g.ends_with_readback);
    }
}

/// Partial fusion: segments 0-2 fusible, seg 2 has readback, segments 3-6 fusible.
/// This mirrors the Kokoro pipeline: encode+prosody+regulate form one group
/// (regulate has readback), then f0+harmonic+generator+istft form another.
#[test]
fn test_partial_fusion_kokoro_pattern() {
    let segments = vec![
        SegmentInfo::new("encode", 30, false),
        SegmentInfo::new("prosody", 20, false),
        SegmentInfo::new("regulate", 8, true), // CPU readback
        SegmentInfo::new("f0_energy", 15, false),
        SegmentInfo::new("harmonic", 12, false),
        SegmentInfo::new("generator", 45, false),
        SegmentInfo::new("istft", 10, false),
    ];

    let plan = plan_segment_fusion(&segments);

    assert_eq!(plan.segments_before, 7);
    assert_eq!(plan.groups_after, 2);
    assert_eq!(plan.submit_reduction, 5);
    assert!(plan.has_fusion());
    assert_eq!(plan.total_dispatches(), 140);

    // Group 0: encode + prosody + regulate
    let g0 = &plan.groups[0];
    assert_eq!(g0.start_idx, 0);
    assert_eq!(g0.end_idx, 3);
    assert_eq!(g0.segment_count(), 3);
    assert_eq!(g0.total_dispatches, 58);
    assert!(g0.ends_with_readback);
    assert_eq!(g0.segment_names, vec!["encode", "prosody", "regulate"]);

    // Group 1: f0_energy + harmonic + generator + istft
    let g1 = &plan.groups[1];
    assert_eq!(g1.start_idx, 3);
    assert_eq!(g1.end_idx, 7);
    assert_eq!(g1.segment_count(), 4);
    assert_eq!(g1.total_dispatches, 82);
    assert!(!g1.ends_with_readback);
    assert_eq!(
        g1.segment_names,
        vec!["f0_energy", "harmonic", "generator", "istft"]
    );
}

/// Multiple readback boundaries create multiple groups.
/// Pattern: [A, B] | readback | [C] | readback | [D, E, F, G, H]
#[test]
fn test_multiple_readback_boundaries() {
    let segments = vec![
        SegmentInfo::new("seg1", 10, false),
        SegmentInfo::new("seg2", 10, true),  // readback
        SegmentInfo::new("seg3", 10, true),  // readback
        SegmentInfo::new("seg4", 10, false),
        SegmentInfo::new("seg5", 10, false),
        SegmentInfo::new("seg6", 10, false),
        SegmentInfo::new("seg7", 10, false),
        SegmentInfo::new("seg8", 10, false),
    ];

    let plan = plan_segment_fusion(&segments);

    assert_eq!(plan.segments_before, 8);
    assert_eq!(plan.groups_after, 3);
    assert_eq!(plan.submit_reduction, 5);

    // Group 0: seg1+seg2 (seg2 has readback)
    assert_eq!(plan.groups[0].segment_count(), 2);
    assert!(plan.groups[0].ends_with_readback);

    // Group 1: seg3 (has readback, standalone)
    assert_eq!(plan.groups[1].segment_count(), 1);
    assert!(plan.groups[1].ends_with_readback);

    // Group 2: seg4+seg5+seg6+seg7+seg8 (no readbacks)
    assert_eq!(plan.groups[2].segment_count(), 5);
    assert!(!plan.groups[2].ends_with_readback);
}

/// Empty input produces empty plan.
#[test]
fn test_empty_segments() {
    let plan = plan_segment_fusion(&[]);

    assert_eq!(plan.segments_before, 0);
    assert_eq!(plan.groups_after, 0);
    assert_eq!(plan.submit_reduction, 0);
    assert!(!plan.has_fusion());
    assert!(plan.groups.is_empty());
}

/// Single segment produces a single group.
#[test]
fn test_single_segment() {
    let segments = vec![SegmentInfo::new("only", 42, false)];

    let plan = plan_segment_fusion(&segments);

    assert_eq!(plan.segments_before, 1);
    assert_eq!(plan.groups_after, 1);
    assert_eq!(plan.submit_reduction, 0);
    assert!(!plan.has_fusion());
    assert_eq!(plan.groups[0].segment_count(), 1);
    assert_eq!(plan.groups[0].total_dispatches, 42);
}

/// Single segment with readback.
#[test]
fn test_single_segment_with_readback() {
    let segments = vec![SegmentInfo::new("only", 42, true)];

    let plan = plan_segment_fusion(&segments);

    assert_eq!(plan.segments_before, 1);
    assert_eq!(plan.groups_after, 1);
    assert_eq!(plan.submit_reduction, 0);
    assert!(plan.groups[0].ends_with_readback);
}

/// can_fuse returns true when seg_a has no readback.
#[test]
fn test_can_fuse_no_readback() {
    let a = SegmentInfo::new("a", 10, false);
    let b = SegmentInfo::new("b", 20, false);
    assert!(can_fuse(&a, &b));
}

/// can_fuse returns false when seg_a has a readback.
#[test]
fn test_can_fuse_with_readback() {
    let a = SegmentInfo::new("a", 10, true);
    let b = SegmentInfo::new("b", 20, false);
    assert!(!can_fuse(&a, &b));
}

/// SegmentFusionPlanner::kokoro_segments produces the expected layout.
#[test]
fn test_kokoro_segments_layout() {
    let counts = [30, 20, 8, 15, 12, 45, 10];
    let segments = SegmentFusionPlanner::kokoro_segments(&counts);

    assert_eq!(segments.len(), 7);
    assert_eq!(segments[0].name, "encode");
    assert!(!segments[0].has_cpu_readback_after);
    assert_eq!(segments[2].name, "regulate");
    assert!(segments[2].has_cpu_readback_after);
    assert_eq!(segments[6].name, "istft");
    assert!(!segments[6].has_cpu_readback_after);
}

/// SegmentFusionPlanner::plan_kokoro produces 2 groups matching pipeline phases.
#[test]
fn test_kokoro_plan_two_phases() {
    let counts = [30, 20, 8, 15, 12, 45, 10];
    let plan = SegmentFusionPlanner::plan_kokoro(&counts);

    assert_eq!(plan.segments_before, 7);
    assert_eq!(plan.groups_after, 2);
    assert_eq!(plan.submit_reduction, 5);
    assert!(plan.has_fusion());

    // Phase 1: encode + prosody + regulate
    assert_eq!(plan.groups[0].segment_count(), 3);
    assert!(plan.groups[0].ends_with_readback);

    // Phase 2: f0_energy + harmonic + generator + istft
    assert_eq!(plan.groups[1].segment_count(), 4);
    assert!(!plan.groups[1].ends_with_readback);
}

/// FusionPlan Display output is well-formed.
#[test]
fn test_fusion_plan_display() {
    let counts = [30, 20, 8, 15, 12, 45, 10];
    let plan = SegmentFusionPlanner::plan_kokoro(&counts);
    let display = format!("{plan}");

    assert!(display.contains("Segment Fusion Plan"));
    assert!(display.contains("7 -> 2 groups"));
    assert!(display.contains("submit reduction: 5"));
    assert!(display.contains("encode + prosody + regulate"));
    assert!(display.contains("f0_energy + harmonic + generator + istft"));
    assert!(display.contains("[CPU readback]"));
}

/// Total dispatches is preserved through fusion (no dispatches lost).
#[test]
fn test_dispatch_count_preservation() {
    let segments = vec![
        SegmentInfo::new("a", 10, false),
        SegmentInfo::new("b", 20, true),
        SegmentInfo::new("c", 30, false),
        SegmentInfo::new("d", 40, false),
    ];

    let total_before: usize = segments.iter().map(|s| s.dispatch_count).sum();
    let plan = plan_segment_fusion(&segments);
    assert_eq!(plan.total_dispatches(), total_before);
}

/// Readback at the very first segment isolates it.
#[test]
fn test_readback_at_first_segment() {
    let segments = vec![
        SegmentInfo::new("first", 5, true),  // readback isolates this
        SegmentInfo::new("second", 10, false),
        SegmentInfo::new("third", 15, false),
    ];

    let plan = plan_segment_fusion(&segments);

    assert_eq!(plan.groups_after, 2);
    assert_eq!(plan.groups[0].segment_count(), 1);
    assert_eq!(plan.groups[0].segment_names, vec!["first"]);
    assert_eq!(plan.groups[1].segment_count(), 2);
    assert_eq!(plan.groups[1].segment_names, vec!["second", "third"]);
}

/// Readback at the very last segment: all prior segments fuse, last is included.
#[test]
fn test_readback_at_last_segment() {
    let segments = vec![
        SegmentInfo::new("a", 10, false),
        SegmentInfo::new("b", 20, false),
        SegmentInfo::new("c", 30, true), // readback at end
    ];

    let plan = plan_segment_fusion(&segments);

    // All three fuse into one group because readback is *after* c,
    // and c is the last segment, so the group closes at c.
    assert_eq!(plan.groups_after, 1);
    assert_eq!(plan.groups[0].segment_count(), 3);
    assert!(plan.groups[0].ends_with_readback);
}
