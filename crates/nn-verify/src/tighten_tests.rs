// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::{
    classify_tightening, compare_report_sequence, compare_reports, format_tightening_diff,
    format_tightening_sequence, kokoro_runner_unavailable, TighteningOutcome,
};
use crate::bound_analysis::BoundAnalysisReport;

fn report(
    model_name: &str,
    output_width: f32,
    explosion_points: Vec<usize>,
    crown_coverage: f32,
    precision_drift_ratio: Option<f32>,
) -> BoundAnalysisReport {
    BoundAnalysisReport {
        model_name: model_name.to_string(),
        total_layers: 0,
        layers: Vec::new(),
        explosion_points,
        output_width,
        output_is_finite: output_width.is_finite(),
        crown_coverage,
        recommendations: Vec::new(),
        analyzed_at: "2026-01-01T00:00:00Z".to_string(),
        chained_norm_depth: 0,
        precision_drift_ratio,
        drift_per_layer: None,
    }
}

#[test]
fn test_compare_reports_classifies_improvement() {
    let previous = report("m", 10.0, vec![1, 2], 0.25, Some(0.90));
    let current = report("m", 8.0, vec![1], 0.50, Some(0.95));

    let diff = compare_reports(&previous, &current).expect("compare reports");

    assert_eq!(diff.outcome, TighteningOutcome::Improved);
    assert!(diff.is_improvement());
    assert_eq!(diff.output_width_delta, Some(-2.0));
    assert_eq!(diff.explosion_points_delta, -1);
    assert_eq!(diff.crown_coverage_delta, 0.25);
    let drift_delta = diff.precision_drift_delta.expect("precision drift delta");
    assert!(
        (drift_delta - 0.05).abs() < 1e-6,
        "drift_delta={drift_delta}"
    );
    assert_eq!(
        classify_tightening(&previous, &current).expect("classify tightening"),
        TighteningOutcome::Improved
    );
}

#[test]
fn test_compare_reports_regression_wins_over_partial_improvement() {
    let previous = report("m", 10.0, vec![1], 0.40, Some(0.90));
    let current = report("m", 8.0, vec![1, 2], 0.60, Some(0.95));

    let diff = compare_reports(&previous, &current).expect("compare reports");

    assert_eq!(diff.outcome, TighteningOutcome::Regressed);
    assert!(diff.is_regression());
    assert_eq!(diff.output_width_delta, Some(-2.0));
    assert_eq!(diff.explosion_points_delta, 1);
}

#[test]
fn test_compare_reports_non_finite_output_regresses() {
    let previous = report("m", 10.0, vec![1], 0.40, Some(0.90));
    let current = report("m", f32::INFINITY, vec![1], 0.40, Some(0.90));

    let diff = compare_reports(&previous, &current).expect("compare reports");

    assert_eq!(diff.outcome, TighteningOutcome::Regressed);
    assert!(diff.output_is_finite_before);
    assert!(!diff.output_is_finite_after);
    assert_eq!(diff.output_width_delta, None);
}

#[test]
fn test_compare_reports_classifies_stall() {
    let previous = report("m", 10.0, vec![1, 2], 0.40, Some(0.90));
    let current = report("m", 10.0, vec![1, 2], 0.40, Some(0.90));

    let diff = compare_reports(&previous, &current).expect("compare reports");

    assert_eq!(diff.outcome, TighteningOutcome::Stalled);
    assert!(diff.is_stall());
    assert_eq!(diff.output_width_delta, Some(0.0));
    assert_eq!(diff.explosion_points_delta, 0);
    assert_eq!(diff.crown_coverage_delta, 0.0);
    assert_eq!(diff.precision_drift_delta, Some(0.0));
}

#[test]
fn test_compare_reports_ignores_missing_precision_drift() {
    let previous = report("m", 10.0, vec![1], 0.50, Some(0.90));
    let current = report("m", 10.0, vec![1], 0.50, None);

    let diff = compare_reports(&previous, &current).expect("compare reports");

    assert_eq!(diff.outcome, TighteningOutcome::Stalled);
    assert_eq!(diff.precision_drift_before, Some(0.90));
    assert_eq!(diff.precision_drift_after, None);
    assert_eq!(diff.precision_drift_delta, None);
}

#[test]
fn test_compare_report_sequence_tracks_loop_outcomes() {
    let baseline = report("kokoro", 10.0, vec![1, 2], 0.25, Some(0.90));
    let improved = report("kokoro", 8.0, vec![1], 0.50, Some(0.95));
    let stalled = report("kokoro", 8.0, vec![1], 0.50, Some(0.95));
    let regressed = report("kokoro", 9.0, vec![1, 3], 0.45, Some(0.92));

    let sequence = compare_report_sequence(&[baseline, improved, stalled, regressed])
        .expect("compare sequence");

    assert_eq!(sequence.overall_outcome, TighteningOutcome::Improved);
    assert_eq!(sequence.terminal_outcome, TighteningOutcome::Regressed);
    assert_eq!(sequence.transition_count, 3);
    assert_eq!(sequence.improvement_count, 1);
    assert_eq!(sequence.stall_count, 1);
    assert_eq!(sequence.regression_count, 1);
    assert!(!sequence.monotonic);
    assert_eq!(
        sequence
            .transitions
            .iter()
            .map(|diff| diff.outcome)
            .collect::<Vec<_>>(),
        vec![
            TighteningOutcome::Improved,
            TighteningOutcome::Stalled,
            TighteningOutcome::Regressed,
        ]
    );
}

#[test]
fn test_compare_report_sequence_overall_outcome_uses_baseline_to_final_result() {
    let baseline = report("kokoro", 10.0, vec![1, 2], 0.25, Some(0.90));
    let regressed_mid = report("kokoro", 12.0, vec![1, 2, 3], 0.20, Some(0.88));
    let final_improved = report("kokoro", 8.0, vec![1], 0.50, Some(0.95));

    let sequence = compare_report_sequence(&[baseline, regressed_mid, final_improved])
        .expect("compare sequence");

    assert_eq!(sequence.overall_outcome, TighteningOutcome::Improved);
    assert_eq!(sequence.terminal_outcome, TighteningOutcome::Improved);
    assert_eq!(sequence.improvement_count, 1);
    assert_eq!(sequence.regression_count, 1);
    assert!(!sequence.monotonic);
}

#[test]
fn test_compare_report_sequence_requires_baseline_and_candidate() {
    let err = compare_report_sequence(&[report("kokoro", 10.0, vec![1, 2], 0.25, Some(0.90))])
        .expect_err("single report should fail");

    assert!(err.to_string().contains("baseline"), "{err}");
    assert!(err.to_string().contains("candidate"), "{err}");
}

#[test]
fn test_compare_reports_rejects_incompatible_model_identity() {
    let previous = report("kokoro", 10.0, vec![1], 0.25, Some(0.90));
    let current = report("whisper", 8.0, vec![1], 0.50, Some(0.95));

    let err = compare_reports(&previous, &current).expect_err("model mismatch should fail");
    let err_text = err.to_string();
    assert!(
        err_text.contains("incompatible tightening reports"),
        "{err_text}"
    );
    assert!(err_text.contains("model_name"), "{err_text}");
    assert!(err_text.contains("kokoro"), "{err_text}");
    assert!(err_text.contains("whisper"), "{err_text}");

    let err =
        classify_tightening(&previous, &current).expect_err("classification should fail closed");
    let err_text = err.to_string();
    assert!(err_text.contains("model_name"), "{err_text}");
}

#[test]
fn test_compare_reports_accepts_selective_crown_model_name_variants() {
    let previous = report("kokoro", 10.0, vec![1], 0.25, Some(0.90));
    let current = report("kokoro_selective_crown", 8.0, vec![1], 0.50, Some(0.95));

    let diff = compare_reports(&previous, &current).expect("selective crown suffix should pass");
    assert_eq!(diff.previous_model_name, "kokoro");
    assert_eq!(diff.current_model_name, "kokoro_selective_crown");
    assert_eq!(diff.outcome, TighteningOutcome::Improved);

    let generic_previous = report("selective_crown", 10.0, vec![1], 0.25, Some(0.90));
    let generic_current = report("selective_crown_tightened", 8.0, vec![1], 0.50, Some(0.95));

    assert_eq!(
        classify_tightening(&generic_previous, &generic_current)
            .expect("generic selective crown variant should pass"),
        TighteningOutcome::Improved
    );
}

#[test]
fn test_compare_reports_rejects_structural_mismatch_before_model_alias_check() {
    let previous = report("kokoro", 10.0, vec![1], 0.25, Some(0.90));
    let mut current = report("kokoro_selective_crown", 8.0, vec![1], 0.50, Some(0.95));
    current.total_layers = 1;

    let err = compare_reports(&previous, &current)
        .expect_err("structural mismatch should fail before model aliasing");
    let err_text = err.to_string();

    assert!(err_text.contains("total_layers"), "{err_text}");
    assert!(!err_text.contains("model_name"), "{err_text}");
}

#[test]
fn test_compare_report_sequence_rejects_structural_counter_mismatch() {
    let baseline = report("kokoro", 10.0, vec![1], 0.25, Some(0.90));
    let mut candidate = report("kokoro", 8.0, vec![1], 0.50, Some(0.95));
    candidate.total_layers = 1;

    let err = compare_report_sequence(&[baseline, candidate])
        .expect_err("structural mismatch should fail");

    let err_text = err.to_string();
    assert!(err_text.contains("report-diff pass 1"), "{err_text}");
    assert!(err_text.contains("total_layers"), "{err_text}");
    assert!(err_text.contains("(0 vs 1)"), "{err_text}");
}

#[test]
fn test_format_tightening_diff_mentions_key_signals() {
    let previous = report("kokoro", 10.0, vec![1, 2], 0.25, Some(0.90));
    let current = report("kokoro", 8.0, vec![1], 0.50, Some(0.95));
    let diff = compare_reports(&previous, &current).expect("compare reports");

    let formatted = format_tightening_diff(&diff);

    assert!(formatted.contains("Outcome: improved"), "{formatted}");
    assert!(formatted.contains("Model: kokoro -> kokoro"), "{formatted}");
    assert!(
        formatted.contains("Output width: 10.000000 -> 8.000000"),
        "{formatted}"
    );
    assert!(
        formatted.contains("Explosion points: 2 -> 1"),
        "{formatted}"
    );
    assert!(
        formatted.contains("CROWN coverage: 0.250000 -> 0.500000"),
        "{formatted}"
    );
}

#[test]
fn test_format_tightening_sequence_mentions_summary_and_passes() {
    let baseline = report("kokoro", 10.0, vec![1, 2], 0.25, Some(0.90));
    let improved = report("kokoro", 8.0, vec![1], 0.50, Some(0.95));
    let stalled = report("kokoro", 8.0, vec![1], 0.50, Some(0.95));
    let sequence =
        compare_report_sequence(&[baseline, improved, stalled]).expect("compare sequence");

    let formatted = format_tightening_sequence(&sequence);

    assert!(
        formatted.contains("Overall outcome: improved"),
        "{formatted}"
    );
    assert!(
        formatted.contains("Latest transition: stalled"),
        "{formatted}"
    );
    assert!(formatted.contains("Transitions: 2 (improved 1, stalled 1, regressed 0)"));
    assert!(formatted.contains("Pass 1: improved"), "{formatted}");
    assert!(formatted.contains("Pass 2: stalled"), "{formatted}");
    assert!(
        formatted.contains("Model path: kokoro -> kokoro"),
        "{formatted}"
    );
}

#[test]
fn test_format_tightening_sequence_distinguishes_overall_from_latest_transition() {
    let baseline = report("kokoro", 10.0, vec![1, 2], 0.25, Some(0.90));
    let improved = report("kokoro", 8.0, vec![1], 0.50, Some(0.95));
    let stalled = report("kokoro", 8.0, vec![1], 0.50, Some(0.95));
    let sequence = compare_report_sequence(&[baseline, improved, stalled]).expect("sequence");

    let formatted = format_tightening_sequence(&sequence);

    assert!(
        formatted.contains("Overall outcome: improved"),
        "{formatted}"
    );
    assert!(
        formatted.contains("Latest transition: stalled"),
        "{formatted}"
    );
}

#[test]
fn test_kokoro_runner_unavailable_mentions_report_diff() {
    let err = kokoro_runner_unavailable().to_string();

    assert!(err.contains("report-diff"), "{err}");
    assert!(err.contains("not wired"), "{err}");
}
