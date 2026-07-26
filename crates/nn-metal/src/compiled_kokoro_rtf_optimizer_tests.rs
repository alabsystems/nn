// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the RTF optimizer module.

use super::*;
use nn_dsl::{CostEstimate, FusionBlocker, FusionGap, FusionGapAnalysis};

/// Build a minimal `SegmentGapAnalysis` for testing.
fn make_segment(
    name: &str,
    dispatch_count: usize,
    theoretical_minimum: usize,
    cost_ns: f64,
    gaps: Vec<FusionGap>,
) -> SegmentGapAnalysis {
    SegmentGapAnalysis {
        segment_name: name.to_string(),
        gap_analysis: FusionGapAnalysis {
            gaps,
            total_dispatches: dispatch_count,
            theoretical_minimum,
        },
        cost_estimate: CostEstimate {
            total_ns: cost_ns,
            per_step_ns: vec![(0, cost_ns)],
            dispatch_count,
        },
        dispatch_count,
        theoretical_minimum,
    }
}

fn make_gap(blocker: FusionBlocker, savings: usize) -> FusionGap {
    FusionGap {
        step_a: 0,
        step_b: 1,
        kernel_a: "a".into(),
        kernel_b: "b".into(),
        reason: blocker,
        savings,
    }
}

#[test]
fn test_rtf_optimizer_empty_segments() {
    let optimizer = RtfOptimizer::apple_m4_max();
    let report = optimizer.analyze(&[]);
    assert_eq!(report.total_dispatches, 0);
    assert_eq!(report.total_estimated_ns, 0.0);
    assert!(report.meets_target, "0 cost should meet any target");
    assert!(report.bottlenecks.is_empty());
}

#[test]
fn test_rtf_optimizer_single_segment() {
    let optimizer = RtfOptimizer::new(CostModel::apple_m4_max(), 0.03);
    let segments = vec![make_segment("generator", 100, 60, 50_000.0, vec![])];
    let report = optimizer.analyze(&segments);

    assert_eq!(report.total_dispatches, 100);
    assert_eq!(report.total_theoretical_minimum, 60);
    assert_eq!(report.segments.len(), 1);
    assert!(
        (report.segments[0].cost_fraction - 1.0).abs() < 1e-6,
        "single segment should have 100% cost fraction"
    );
}

#[test]
fn test_rtf_optimizer_multi_segment_cost_fractions() {
    let optimizer = RtfOptimizer::apple_m4_max();
    let segments = vec![
        make_segment("plbert", 30, 25, 10_000.0, vec![]),
        make_segment("generator", 100, 60, 90_000.0, vec![]),
    ];
    let report = optimizer.analyze(&segments);

    assert_eq!(report.total_dispatches, 130);
    assert_eq!(report.total_estimated_ns, 100_000.0);

    // plbert should have 10% cost fraction.
    let plbert = &report.segments[0];
    assert!(
        (plbert.cost_fraction - 0.1).abs() < 1e-6,
        "plbert cost fraction should be 0.1, got {}",
        plbert.cost_fraction
    );

    // generator should have 90% cost fraction.
    let generator = &report.segments[1];
    assert!(
        (generator.cost_fraction - 0.9).abs() < 1e-6,
        "generator cost fraction should be 0.9, got {}",
        generator.cost_fraction
    );
}

#[test]
fn test_bottleneck_high_cost_fraction() {
    let optimizer = RtfOptimizer::apple_m4_max();
    let segments = vec![
        make_segment("plbert", 10, 10, 1_000.0, vec![]),
        make_segment("generator", 100, 60, 99_000.0, vec![]),
    ];
    let report = optimizer.analyze(&segments);

    // Generator consumes 99% of cost — should be flagged.
    let high_cost: Vec<_> = report
        .bottlenecks
        .iter()
        .filter(|b| b.category == "high_cost_fraction")
        .collect();
    assert!(
        !high_cost.is_empty(),
        "generator should be flagged as high cost fraction"
    );
    assert_eq!(high_cost[0].segment_name, "generator");
}

#[test]
fn test_bottleneck_fusion_gap() {
    let optimizer = RtfOptimizer::apple_m4_max();
    let gaps = vec![
        make_gap(FusionBlocker::FanOut, 1),
        make_gap(FusionBlocker::ShapeMismatch, 1),
    ];
    // 20 dispatches, 18 theoretical min => 10% opportunity
    let segments = vec![make_segment("text", 20, 17, 10_000.0, gaps)];
    let report = optimizer.analyze(&segments);

    let fusion_gaps: Vec<_> = report
        .bottlenecks
        .iter()
        .filter(|b| b.category == "fusion_gap")
        .collect();
    assert!(
        !fusion_gaps.is_empty(),
        "text should be flagged for fusion gaps"
    );
}

#[test]
fn test_bottleneck_dispatch_overhead() {
    let optimizer = RtfOptimizer::apple_m4_max();
    // 50 dispatches, 30 minimum => 20 excess > threshold of 5.
    let segments = vec![make_segment("prosody", 50, 30, 25_000.0, vec![])];
    let report = optimizer.analyze(&segments);

    let dispatch_overhead: Vec<_> = report
        .bottlenecks
        .iter()
        .filter(|b| b.category == "dispatch_overhead")
        .collect();
    assert!(
        !dispatch_overhead.is_empty(),
        "prosody should be flagged for dispatch overhead"
    );
}

#[test]
fn test_bottlenecks_sorted_by_savings() {
    let optimizer = RtfOptimizer::apple_m4_max();
    let segments = vec![
        make_segment("plbert", 50, 30, 10_000.0, vec![]),
        make_segment("generator", 100, 60, 90_000.0, vec![]),
    ];
    let report = optimizer.analyze(&segments);

    // Verify bottlenecks are sorted by potential_savings_ns descending.
    for w in report.bottlenecks.windows(2) {
        assert!(
            w[0].potential_savings_ns >= w[1].potential_savings_ns,
            "bottlenecks should be sorted by savings: {:.1} < {:.1}",
            w[0].potential_savings_ns,
            w[1].potential_savings_ns,
        );
    }
}

#[test]
fn test_aggregate_blockers() {
    let optimizer = RtfOptimizer::apple_m4_max();
    let segments = vec![
        make_segment(
            "plbert",
            20,
            18,
            5_000.0,
            vec![make_gap(FusionBlocker::NonFusibleOp, 0)],
        ),
        make_segment(
            "generator",
            100,
            80,
            50_000.0,
            vec![
                make_gap(FusionBlocker::NonFusibleOp, 0),
                make_gap(FusionBlocker::FanOut, 1),
            ],
        ),
    ];
    let report = optimizer.analyze(&segments);

    assert_eq!(report.aggregate_blockers.get("NonFusibleOp"), Some(&2));
    assert_eq!(report.aggregate_blockers.get("FanOut"), Some(&1));
}

#[test]
fn test_projected_rtf_meets_target() {
    let optimizer = RtfOptimizer::new(CostModel::apple_m4_max(), 1.0);
    // Very small cost => very low RTF => should meet target.
    let segments = vec![make_segment("generator", 5, 5, 100.0, vec![])];
    let report = optimizer.analyze(&segments);
    assert!(
        report.meets_target,
        "100ns inference for 3s audio should meet RTF 1.0"
    );
    assert!(report.projected_rtf < 1.0);
}

#[test]
fn test_report_summary_contains_key_info() {
    let optimizer = RtfOptimizer::apple_m4_max();
    let segments = vec![
        make_segment("plbert", 30, 25, 10_000.0, vec![]),
        make_segment("generator", 100, 60, 90_000.0, vec![]),
    ];
    let report = optimizer.analyze(&segments);
    let summary = report.summary();

    assert!(summary.contains("RTF Optimization Report"));
    assert!(summary.contains("Projected RTF"));
    assert!(summary.contains("plbert"));
    assert!(summary.contains("generator"));
    assert!(summary.contains("Dispatches:"));
    assert!(summary.contains("Per-Segment Breakdown"));
}

#[test]
fn test_display_matches_summary() {
    let optimizer = RtfOptimizer::apple_m4_max();
    let segments = vec![make_segment("text", 10, 10, 5_000.0, vec![])];
    let report = optimizer.analyze(&segments);
    assert_eq!(format!("{report}"), report.summary());
}

#[test]
fn test_optimization_opportunity_pct() {
    let optimizer = RtfOptimizer::apple_m4_max();
    let segments = vec![
        make_segment("plbert", 20, 15, 10_000.0, vec![]),
        make_segment("generator", 80, 50, 40_000.0, vec![]),
    ];
    let report = optimizer.analyze(&segments);

    // Total: 100 dispatches, 65 minimum => 35% opportunity.
    assert!(
        (report.optimization_opportunity_pct - 35.0).abs() < 0.1,
        "expected 35% opportunity, got {:.1}%",
        report.optimization_opportunity_pct
    );
}

#[test]
fn test_constructor_accessors() {
    let optimizer = RtfOptimizer::new(CostModel::apple_m4(), 0.05);
    assert_eq!(optimizer.target_rtf(), 0.05);
    assert_eq!(optimizer.cost_model().simd_width, 32);
}

#[test]
fn test_apple_m4_max_preset() {
    let optimizer = RtfOptimizer::apple_m4_max();
    assert_eq!(optimizer.target_rtf(), 0.03);
    assert_eq!(optimizer.cost_model().launch_overhead_ns, 1500.0);
}

#[cfg(feature = "plan-serde")]
#[test]
fn test_closed_loop_skips_warmup_when_target_is_met_without_gaps() {
    let optimizer = RtfOptimizer::new(CostModel::apple_m4_max(), 1.0);
    let baseline = vec![make_segment("generator", 5, 5, 100.0, vec![])];
    let mut gap_calls = 0usize;
    let mut warmup_calls = 0usize;

    let report = optimizer
        .run_closed_loop(
            || {
                gap_calls += 1;
                Ok(baseline.clone())
            },
            || {
                warmup_calls += 1;
                Ok(RtfWarmupSummary {
                    loaded_from_cache: false,
                    configs_applied: 1,
                    segments_compiled: 8,
                })
            },
            || Some("unused".to_string()),
        )
        .expect("closed loop should succeed");

    assert_eq!(gap_calls, 1, "baseline only should be analyzed once");
    assert_eq!(warmup_calls, 0, "warmup should be skipped");
    assert!(report.warmup.is_none(), "no warmup summary expected");
    assert_eq!(
        report.baseline.total_dispatches,
        report.final_report.total_dispatches
    );
    assert!(
        report.actions_taken[0].contains("Skipped optimizer warmup"),
        "expected skip action, got {:?}",
        report.actions_taken
    );
}

#[cfg(feature = "plan-serde")]
#[test]
fn test_closed_loop_runs_warmup_and_reanalyzes() {
    let optimizer = RtfOptimizer::new(CostModel::apple_m4_max(), 1e-9);
    let baseline = vec![make_segment("generator", 100, 60, 90_000.0, vec![])];
    let optimized = vec![make_segment("generator", 60, 60, 40_000.0, vec![])];
    let mut gap_calls = 0usize;
    let mut warmup_calls = 0usize;

    let report = optimizer
        .run_closed_loop(
            || {
                gap_calls += 1;
                if gap_calls == 1 {
                    Ok(baseline.clone())
                } else {
                    Ok(optimized.clone())
                }
            },
            || {
                warmup_calls += 1;
                Ok(RtfWarmupSummary {
                    loaded_from_cache: true,
                    configs_applied: 3,
                    segments_compiled: 12,
                })
            },
            || Some("optimizer search summary".to_string()),
        )
        .expect("closed loop should succeed");

    assert_eq!(gap_calls, 2, "baseline and final analysis should run");
    assert_eq!(warmup_calls, 1, "warmup should run once");
    assert_eq!(report.dispatches_saved(), 40);
    assert!(report.estimated_cost_saved_ns() > 0.0);
    assert_eq!(
        report.warmup,
        Some(RtfWarmupSummary {
            loaded_from_cache: true,
            configs_applied: 3,
            segments_compiled: 12,
        })
    );
    assert_eq!(
        report.optimizer_summary.as_deref(),
        Some("optimizer search summary")
    );
    assert!(
        report.summary().contains("Actions Taken"),
        "closed-loop summary should include action log"
    );
    assert_eq!(format!("{report}"), report.summary());
}
