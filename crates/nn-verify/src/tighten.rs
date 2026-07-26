// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Reusable report-diff helpers for progressive tightening.
//!
//! Compares two [`BoundAnalysisReport`] values and classifies the transition
//! as an improvement, regression, or stall. The comparison is intentionally
//! small: it focuses on the core signals already carried by the report rather
//! than introducing another orchestration layer.
//!
//! Part of #2456.

use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use crate::bound_analysis::BoundAnalysisReport;
use crate::error::VerifyError;

const FLOAT_REL_TOLERANCE: f32 = 1e-6;

/// Coarse outcome when comparing two bound-analysis reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TighteningOutcome {
    /// The new report is strictly better on at least one tracked signal and
    /// does not worsen any tracked signal.
    Improved,
    /// The new report worsens at least one tracked signal.
    Regressed,
    /// The new report is effectively unchanged across the tracked signals.
    Stalled,
}

impl std::fmt::Display for TighteningOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Improved => "improved",
            Self::Regressed => "regressed",
            Self::Stalled => "stalled",
        })
    }
}

/// Structured diff between two [`BoundAnalysisReport`] values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TighteningDiff {
    /// Overall classification for the transition.
    pub outcome: TighteningOutcome,
    /// Previous report model name.
    pub previous_model_name: String,
    /// Current report model name.
    pub current_model_name: String,
    /// Previous final-layer output width.
    pub output_width_before: f32,
    /// Current final-layer output width.
    pub output_width_after: f32,
    /// Delta in output width when both values are finite.
    pub output_width_delta: Option<f32>,
    /// Previous explosion-point count.
    pub explosion_points_before: usize,
    /// Current explosion-point count.
    pub explosion_points_after: usize,
    /// Signed delta in explosion-point count.
    pub explosion_points_delta: isize,
    /// Previous CROWN coverage.
    pub crown_coverage_before: f32,
    /// Current CROWN coverage.
    pub crown_coverage_after: f32,
    /// Delta in CROWN coverage.
    pub crown_coverage_delta: f32,
    /// Previous precision-drift ratio, if available.
    pub precision_drift_before: Option<f32>,
    /// Current precision-drift ratio, if available.
    pub precision_drift_after: Option<f32>,
    /// Delta in precision-drift ratio when both values are finite.
    pub precision_drift_delta: Option<f32>,
    /// Previous output finiteness flag.
    pub output_is_finite_before: bool,
    /// Current output finiteness flag.
    pub output_is_finite_after: bool,
}

/// Structured comparison across a baseline and one or more tightening passes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TighteningSequence {
    /// Aggregate outcome across the full pass sequence.
    pub overall_outcome: TighteningOutcome,
    /// Outcome of the final transition in the sequence.
    pub terminal_outcome: TighteningOutcome,
    /// Baseline report model name.
    pub baseline_model_name: String,
    /// Final report model name.
    pub final_model_name: String,
    /// Number of pairwise transitions in the sequence.
    pub transition_count: usize,
    /// Number of improving transitions.
    pub improvement_count: usize,
    /// Number of regressing transitions.
    pub regression_count: usize,
    /// Number of stalled transitions.
    pub stall_count: usize,
    /// Whether every transition avoided regression.
    pub monotonic: bool,
    /// Per-transition diffs in order.
    pub transitions: Vec<TighteningDiff>,
}

impl TighteningDiff {
    /// Returns `true` when the current report is a tightening improvement.
    #[must_use]
    pub fn is_improvement(&self) -> bool {
        matches!(self.outcome, TighteningOutcome::Improved)
    }

    /// Returns `true` when the current report regressed.
    #[must_use]
    pub fn is_regression(&self) -> bool {
        matches!(self.outcome, TighteningOutcome::Regressed)
    }

    /// Returns `true` when the comparison stalled.
    #[must_use]
    pub fn is_stall(&self) -> bool {
        matches!(self.outcome, TighteningOutcome::Stalled)
    }
}

/// Compare two reports and return a structured diff.
///
/// Returns [`VerifyError::InvalidInput`] when the reports do not appear to
/// describe the same model structure.
pub fn compare_reports(
    previous: &BoundAnalysisReport,
    current: &BoundAnalysisReport,
) -> Result<TighteningDiff, VerifyError> {
    validate_report_compatibility(previous, current)?;

    let output_width_delta = match (previous.output_width, current.output_width) {
        (prev, curr) if prev.is_finite() && curr.is_finite() => Some(curr - prev),
        _ => None,
    };

    let explosion_points_delta =
        current.explosion_points.len() as isize - previous.explosion_points.len() as isize;

    let crown_coverage_delta = current.crown_coverage - previous.crown_coverage;

    let precision_drift_delta = match (
        previous.precision_drift_ratio,
        current.precision_drift_ratio,
    ) {
        (Some(prev), Some(curr)) if prev.is_finite() && curr.is_finite() => Some(curr - prev),
        _ => None,
    };

    let outcome = classify_transition(
        previous.output_width,
        current.output_width,
        previous.explosion_points.len(),
        current.explosion_points.len(),
        previous.crown_coverage,
        current.crown_coverage,
        previous.precision_drift_ratio,
        current.precision_drift_ratio,
        previous.output_is_finite,
        current.output_is_finite,
    );

    Ok(TighteningDiff {
        outcome,
        previous_model_name: previous.model_name.clone(),
        current_model_name: current.model_name.clone(),
        output_width_before: previous.output_width,
        output_width_after: current.output_width,
        output_width_delta,
        explosion_points_before: previous.explosion_points.len(),
        explosion_points_after: current.explosion_points.len(),
        explosion_points_delta,
        crown_coverage_before: previous.crown_coverage,
        crown_coverage_after: current.crown_coverage,
        crown_coverage_delta,
        precision_drift_before: previous.precision_drift_ratio,
        precision_drift_after: current.precision_drift_ratio,
        precision_drift_delta,
        output_is_finite_before: previous.output_is_finite,
        output_is_finite_after: current.output_is_finite,
    })
}

/// Classify the comparison between two reports without materializing the diff.
///
/// Returns [`VerifyError::InvalidInput`] when the reports do not appear to
/// describe the same model structure.
pub fn classify_tightening(
    previous: &BoundAnalysisReport,
    current: &BoundAnalysisReport,
) -> Result<TighteningOutcome, VerifyError> {
    Ok(compare_reports(previous, current)?.outcome)
}

/// Load a bound-analysis report from a JSON file.
pub fn load_bound_analysis_report(
    path: impl AsRef<Path>,
) -> Result<BoundAnalysisReport, VerifyError> {
    let path = path.as_ref();
    let contents = fs::read_to_string(path)?;
    serde_json::from_str(&contents).map_err(VerifyError::from)
}

/// Compare two JSON report files and return the structured diff.
pub fn compare_report_paths(
    previous: impl AsRef<Path>,
    current: impl AsRef<Path>,
) -> Result<TighteningDiff, VerifyError> {
    let previous = load_bound_analysis_report(previous)?;
    let current = load_bound_analysis_report(current)?;
    compare_reports(&previous, &current)
}

/// Compare a baseline and one or more tightening passes.
pub fn compare_report_sequence(
    reports: &[BoundAnalysisReport],
) -> Result<TighteningSequence, VerifyError> {
    if reports.len() < 2 {
        return Err(VerifyError::InvalidInput(
            "report-diff sequence requires at least a baseline and one candidate report"
                .to_string(),
        ));
    }

    let transitions: Vec<TighteningDiff> = reports
        .windows(2)
        .enumerate()
        .map(|(idx, pair)| {
            compare_reports(&pair[0], &pair[1])
                .map_err(|err| annotate_sequence_transition_error(idx + 1, err))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut improvement_count = 0usize;
    let mut regression_count = 0usize;
    let mut stall_count = 0usize;

    for diff in &transitions {
        match diff.outcome {
            TighteningOutcome::Improved => improvement_count += 1,
            TighteningOutcome::Regressed => regression_count += 1,
            TighteningOutcome::Stalled => stall_count += 1,
        }
    }

    // "Overall" answers the question users actually care about: did the final
    // candidate end up better than the baseline, regardless of whether the
    // intermediate path was monotonic? The separate `monotonic` flag and
    // transition counters preserve the step-by-step story.
    let overall_outcome = compare_reports(
        reports.first().expect("validated report sequence length"),
        reports.last().expect("validated report sequence length"),
    )?
    .outcome;

    let terminal_outcome = transitions
        .last()
        .map(|diff| diff.outcome)
        .expect("validated report sequence length");

    Ok(TighteningSequence {
        overall_outcome,
        terminal_outcome,
        baseline_model_name: reports
            .first()
            .expect("validated report sequence length")
            .model_name
            .clone(),
        final_model_name: reports
            .last()
            .expect("validated report sequence length")
            .model_name
            .clone(),
        transition_count: transitions.len(),
        improvement_count,
        regression_count,
        stall_count,
        monotonic: regression_count == 0,
        transitions,
    })
}

/// Compare a baseline file and one or more candidate report files.
pub fn compare_report_path_sequence<P>(paths: &[P]) -> Result<TighteningSequence, VerifyError>
where
    P: AsRef<Path>,
{
    let reports = paths
        .iter()
        .map(load_bound_analysis_report)
        .collect::<Result<Vec<_>, _>>()?;
    compare_report_sequence(&reports)
}

/// Format a diff for human-readable CLI output.
pub fn format_tightening_diff(diff: &TighteningDiff) -> String {
    let mut out = String::new();
    let width_ratio = match diff.output_width_delta {
        Some(delta)
            if diff.output_width_before.is_finite()
                && diff.output_width_before.abs() > f32::EPSILON =>
        {
            Some(delta / diff.output_width_before)
        }
        _ => None,
    };

    writeln!(out, "Outcome: {}", diff.outcome).expect("write to string");
    writeln!(
        out,
        "Model: {} -> {}",
        diff.previous_model_name, diff.current_model_name
    )
    .expect("write to string");
    writeln!(
        out,
        "Output width: {:.6} -> {:.6}",
        diff.output_width_before, diff.output_width_after
    )
    .expect("write to string");
    match diff.output_width_delta {
        Some(delta) => {
            if let Some(ratio) = width_ratio {
                writeln!(out, "Width delta: {delta:+.6} ({ratio:+.6} of baseline)")
                    .expect("write to string");
            } else {
                writeln!(out, "Width delta: {delta:+.6}").expect("write to string");
            }
        }
        None => {
            writeln!(
                out,
                "Width delta: unavailable (non-finite baseline or candidate)"
            )
            .expect("write to string");
        }
    }
    writeln!(
        out,
        "Explosion points: {} -> {} (delta {:+})",
        diff.explosion_points_before, diff.explosion_points_after, diff.explosion_points_delta
    )
    .expect("write to string");
    writeln!(
        out,
        "CROWN coverage: {:.6} -> {:.6} (delta {:+.6})",
        diff.crown_coverage_before, diff.crown_coverage_after, diff.crown_coverage_delta
    )
    .expect("write to string");
    match diff.precision_drift_delta {
        Some(delta) => {
            let before = diff
                .precision_drift_before
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "n/a".to_string());
            let after = diff
                .precision_drift_after
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "n/a".to_string());
            writeln!(
                out,
                "Precision drift: {before} -> {after} (delta {delta:+.6})"
            )
            .expect("write to string");
        }
        None => {
            writeln!(out, "Precision drift: unavailable").expect("write to string");
        }
    }
    writeln!(
        out,
        "Finiteness: {} -> {}",
        diff.output_is_finite_before, diff.output_is_finite_after
    )
    .expect("write to string");

    out
}

/// Format a sequence of diffs for human-readable CLI output.
pub fn format_tightening_sequence(sequence: &TighteningSequence) -> String {
    let mut out = String::new();
    writeln!(out, "Overall outcome: {}", sequence.overall_outcome).expect("write to string");
    writeln!(out, "Latest transition: {}", sequence.terminal_outcome).expect("write to string");
    writeln!(
        out,
        "Model path: {} -> {}",
        sequence.baseline_model_name, sequence.final_model_name
    )
    .expect("write to string");
    writeln!(
        out,
        "Transitions: {} (improved {}, stalled {}, regressed {})",
        sequence.transition_count,
        sequence.improvement_count,
        sequence.stall_count,
        sequence.regression_count
    )
    .expect("write to string");
    writeln!(out, "Monotonic: {}", sequence.monotonic).expect("write to string");

    for (idx, diff) in sequence.transitions.iter().enumerate() {
        writeln!(out).expect("write to string");
        writeln!(out, "Pass {}: {}", idx + 1, diff.outcome).expect("write to string");
        for line in format_tightening_diff(diff).lines() {
            writeln!(out, "  {line}").expect("write to string");
        }
    }

    out
}

/// Clear error for the not-yet-wired live Kokoro runner.
pub fn kokoro_runner_unavailable() -> VerifyError {
    VerifyError::InvalidInput(
        "Kokoro live tightening runner is not wired in nn-verify yet; use report-diff on precomputed BoundAnalysisReport JSON instead".to_string(),
    )
}

fn validate_report_compatibility(
    previous: &BoundAnalysisReport,
    current: &BoundAnalysisReport,
) -> Result<(), VerifyError> {
    ensure_matching_usize("total_layers", previous.total_layers, current.total_layers)?;
    ensure_matching_usize("layers.len()", previous.layers.len(), current.layers.len())?;
    ensure_matching_usize(
        "chained_norm_depth",
        previous.chained_norm_depth,
        current.chained_norm_depth,
    )?;

    for (slot, (previous_layer, current_layer)) in
        previous.layers.iter().zip(&current.layers).enumerate()
    {
        ensure_matching_usize(
            &format!("layers[{slot}].layer_index"),
            previous_layer.layer_index,
            current_layer.layer_index,
        )?;
        ensure_matching_string(
            &format!("layers[{slot}].layer_type"),
            &previous_layer.layer_type,
            &current_layer.layer_type,
        )?;
        if previous_layer.node_name != current_layer.node_name {
            return Err(incompatible_reports(format!(
                "layers[{slot}].node_name differs ({:?} vs {:?})",
                previous_layer.node_name, current_layer.node_name
            )));
        }
    }

    ensure_compatible_model_name(&previous.model_name, &current.model_name)?;

    Ok(())
}

fn ensure_matching_string(field: &str, previous: &str, current: &str) -> Result<(), VerifyError> {
    if previous == current {
        return Ok(());
    }

    Err(incompatible_reports(format!(
        "{field} differs ({previous:?} vs {current:?})"
    )))
}

fn ensure_compatible_model_name(previous: &str, current: &str) -> Result<(), VerifyError> {
    if model_names_are_compatible(previous, current) {
        return Ok(());
    }

    Err(incompatible_reports(format!(
        "model_name differs ({previous:?} vs {current:?})"
    )))
}

fn model_names_are_compatible(previous: &str, current: &str) -> bool {
    if previous == current {
        return true;
    }

    match (
        selective_crown_model_root(previous),
        selective_crown_model_root(current),
    ) {
        (Some(previous_root), Some(current_root)) => previous_root == current_root,
        (Some(previous_root), None) => previous_root == current,
        (None, Some(current_root)) => previous == current_root,
        (None, None) => false,
    }
}

fn selective_crown_model_root(name: &str) -> Option<&str> {
    if matches!(name, "selective_crown" | "selective_crown_tightened") {
        return Some("selective_crown");
    }

    name.strip_suffix("_selective_crown_tightened")
        .or_else(|| name.strip_suffix("_selective_crown"))
}

fn ensure_matching_usize(field: &str, previous: usize, current: usize) -> Result<(), VerifyError> {
    if previous == current {
        return Ok(());
    }

    Err(incompatible_reports(format!(
        "{field} differs ({previous} vs {current})"
    )))
}

fn incompatible_reports(reason: String) -> VerifyError {
    VerifyError::InvalidInput(format!("incompatible tightening reports: {reason}"))
}

fn annotate_sequence_transition_error(pass: usize, err: VerifyError) -> VerifyError {
    match err {
        VerifyError::InvalidInput(message) => {
            VerifyError::InvalidInput(format!("report-diff pass {pass}: {message}"))
        }
        other => other,
    }
}

fn classify_transition(
    previous_width: f32,
    current_width: f32,
    previous_explosions: usize,
    current_explosions: usize,
    previous_coverage: f32,
    current_coverage: f32,
    previous_drift: Option<f32>,
    current_drift: Option<f32>,
    previous_finite: bool,
    current_finite: bool,
) -> TighteningOutcome {
    let mut saw_improvement = false;

    if previous_finite != current_finite {
        if current_finite {
            saw_improvement = true;
        } else {
            return TighteningOutcome::Regressed;
        }
    }

    match compare_lower_is_better(previous_width, current_width) {
        MetricTrend::Improved => saw_improvement = true,
        MetricTrend::Regressed => return TighteningOutcome::Regressed,
        MetricTrend::Stalled => {}
    }

    match current_explosions.cmp(&previous_explosions) {
        std::cmp::Ordering::Less => saw_improvement = true,
        std::cmp::Ordering::Greater => return TighteningOutcome::Regressed,
        std::cmp::Ordering::Equal => {}
    }

    match compare_higher_is_better(previous_coverage, current_coverage) {
        MetricTrend::Improved => saw_improvement = true,
        MetricTrend::Regressed => return TighteningOutcome::Regressed,
        MetricTrend::Stalled => {}
    }

    match compare_optional_higher_is_better(previous_drift, current_drift) {
        MetricTrend::Improved => saw_improvement = true,
        MetricTrend::Regressed => return TighteningOutcome::Regressed,
        MetricTrend::Stalled => {}
    }

    if saw_improvement {
        TighteningOutcome::Improved
    } else {
        TighteningOutcome::Stalled
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetricTrend {
    Improved,
    Regressed,
    Stalled,
}

fn compare_lower_is_better(previous: f32, current: f32) -> MetricTrend {
    compare_with_direction(previous, current, false)
}

fn compare_higher_is_better(previous: f32, current: f32) -> MetricTrend {
    compare_with_direction(previous, current, true)
}

fn compare_optional_higher_is_better(previous: Option<f32>, current: Option<f32>) -> MetricTrend {
    match (previous, current) {
        (Some(prev), Some(curr)) => compare_higher_is_better(prev, curr),
        _ => MetricTrend::Stalled,
    }
}

fn compare_with_direction(previous: f32, current: f32, higher_is_better: bool) -> MetricTrend {
    if previous.is_finite() != current.is_finite() {
        return if current.is_finite() {
            MetricTrend::Improved
        } else {
            MetricTrend::Regressed
        };
    }

    if !previous.is_finite() && !current.is_finite() {
        return MetricTrend::Stalled;
    }

    let tolerance = float_tolerance(previous, current);
    if higher_is_better {
        if current > previous + tolerance {
            MetricTrend::Improved
        } else if current + tolerance < previous {
            MetricTrend::Regressed
        } else {
            MetricTrend::Stalled
        }
    } else if current + tolerance < previous {
        MetricTrend::Improved
    } else if current > previous + tolerance {
        MetricTrend::Regressed
    } else {
        MetricTrend::Stalled
    }
}

fn float_tolerance(previous: f32, current: f32) -> f32 {
    FLOAT_REL_TOLERANCE * previous.abs().max(current.abs()).max(1.0)
}

#[cfg(test)]
#[path = "tighten_tests.rs"]
mod tests;
