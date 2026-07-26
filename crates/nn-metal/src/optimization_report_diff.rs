// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Report diff and stall detection for the progressive tightening loop.
//!
//! Compares two [`OptimizationReport`]s (prev vs. curr) and produces a
//! [`ReportDelta`] with numeric deltas, violation changes, and a verdict.
//!
//! Part of #2456, #2218.

use serde::{Deserialize, Serialize};

use super::OptimizationReport;

/// Numeric and qualitative diff between two optimization iterations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ReportDelta {
    /// Previous iteration number.
    pub prev_iteration: usize,
    /// Current iteration number.
    pub curr_iteration: usize,
    /// Dispatch count change (negative = improvement).
    pub dispatch_delta: i64,
    /// Metal dispatch count change (negative = improvement).
    pub metal_dispatch_delta: i64,
    /// Buffer memory change in bytes (negative = improvement).
    pub memory_delta: i64,
    /// GPU flush count change (negative = improvement). Part of #2739.
    pub flush_delta: i64,
    /// GPU submit count change (negative = improvement). Part of #2739.
    pub submit_delta: i64,
    /// New violations introduced in this iteration.
    pub new_violations: Vec<String>,
    /// Violations resolved in this iteration.
    pub resolved_violations: Vec<String>,
    /// Overall verdict for this iteration.
    pub verdict: IterationVerdict,
}

/// High-level verdict for an optimization iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum IterationVerdict {
    /// Net improvement across metrics.
    Improved {
        /// Human-readable summary.
        summary: String,
    },
    /// Net regression across metrics.
    Regressed {
        /// Human-readable summary.
        summary: String,
    },
    /// No meaningful change (potential stall).
    Stalled {
        /// Number of consecutive stalled iterations (caller tracks).
        consecutive_stalls: usize,
    },
    /// Mixed: some metrics improved, some regressed.
    Mixed {
        /// Metrics that improved.
        improved: Vec<String>,
        /// Metrics that regressed.
        regressed: Vec<String>,
    },
}

/// Compare two reports and produce a delta.
///
/// Extracts `total_dispatches`, `total_metal_dispatches`, and memory from
/// each report's performance section (via `serde_json::Value` traversal).
#[must_use]
pub fn diff_reports(prev: &OptimizationReport, curr: &OptimizationReport) -> ReportDelta {
    let prev_dispatches = extract_usize(&prev.performance, "total_dispatches");
    let curr_dispatches = extract_usize(&curr.performance, "total_dispatches");
    let dispatch_delta = curr_dispatches as i64 - prev_dispatches as i64;

    let prev_metal = extract_usize(&prev.performance, "total_metal_dispatches");
    let curr_metal = extract_usize(&curr.performance, "total_metal_dispatches");
    let metal_dispatch_delta = curr_metal as i64 - prev_metal as i64;

    let prev_mem = extract_usize_path(&prev.performance, &["memory", "total_buffer_bytes"]);
    let curr_mem = extract_usize_path(&curr.performance, &["memory", "total_buffer_bytes"]);
    let memory_delta = curr_mem as i64 - prev_mem as i64;

    let prev_flushes = extract_usize(&prev.performance, "total_flushes");
    let curr_flushes = extract_usize(&curr.performance, "total_flushes");
    let flush_delta = curr_flushes as i64 - prev_flushes as i64;

    let prev_submits = extract_usize(&prev.performance, "total_submits");
    let curr_submits = extract_usize(&curr.performance, "total_submits");
    let submit_delta = curr_submits as i64 - prev_submits as i64;

    let (new_violations, resolved_violations) = diff_violations(prev, curr);
    let verdict = compute_verdict(
        dispatch_delta,
        metal_dispatch_delta,
        memory_delta,
        flush_delta,
        submit_delta,
        prev_dispatches,
        curr_dispatches,
        prev_metal,
        curr_metal,
        prev_mem,
        curr_mem,
        prev_flushes,
        curr_flushes,
        prev_submits,
        curr_submits,
        &new_violations,
        &resolved_violations,
    );

    ReportDelta {
        prev_iteration: prev.iteration,
        curr_iteration: curr.iteration,
        dispatch_delta,
        metal_dispatch_delta,
        memory_delta,
        flush_delta,
        submit_delta,
        new_violations,
        resolved_violations,
        verdict,
    }
}

/// Compute violation diff between two reports.
fn diff_violations(
    prev: &OptimizationReport,
    curr: &OptimizationReport,
) -> (Vec<String>, Vec<String>) {
    let prev_violations: Vec<String> = prev
        .contract_status
        .as_ref()
        .map(|s| s.violations.clone())
        .unwrap_or_default();
    let curr_violations: Vec<String> = curr
        .contract_status
        .as_ref()
        .map(|s| s.violations.clone())
        .unwrap_or_default();

    let new: Vec<String> = curr_violations
        .iter()
        .filter(|v| !prev_violations.contains(v))
        .cloned()
        .collect();
    let resolved: Vec<String> = prev_violations
        .iter()
        .filter(|v| !curr_violations.contains(v))
        .cloned()
        .collect();
    (new, resolved)
}

/// Classify metric deltas into an [`IterationVerdict`].
#[allow(clippy::too_many_arguments)]
fn compute_verdict(
    dispatch_delta: i64,
    metal_dispatch_delta: i64,
    memory_delta: i64,
    flush_delta: i64,
    submit_delta: i64,
    prev_dispatches: usize,
    curr_dispatches: usize,
    prev_metal: usize,
    curr_metal: usize,
    prev_mem: usize,
    curr_mem: usize,
    prev_flushes: usize,
    curr_flushes: usize,
    prev_submits: usize,
    curr_submits: usize,
    new_violations: &[String],
    resolved_violations: &[String],
) -> IterationVerdict {
    let mut improved = Vec::new();
    let mut regressed = Vec::new();

    if dispatch_delta < 0 {
        improved.push(format!(
            "dispatches: {prev_dispatches} -> {curr_dispatches}"
        ));
    } else if dispatch_delta > 0 {
        regressed.push(format!(
            "dispatches: {prev_dispatches} -> {curr_dispatches}"
        ));
    }

    if metal_dispatch_delta < 0 {
        improved.push(format!("metal_dispatches: {prev_metal} -> {curr_metal}"));
    } else if metal_dispatch_delta > 0 {
        regressed.push(format!("metal_dispatches: {prev_metal} -> {curr_metal}"));
    }

    if memory_delta < 0 {
        improved.push(format!("memory: {prev_mem} -> {curr_mem} bytes"));
    } else if memory_delta > 0 {
        regressed.push(format!("memory: {prev_mem} -> {curr_mem} bytes"));
    }

    // GPU sync point deltas (#2739) — only count when at least one side measured.
    if flush_delta < 0 && (prev_flushes > 0 || curr_flushes > 0) {
        improved.push(format!("flushes: {prev_flushes} -> {curr_flushes}"));
    } else if flush_delta > 0 && (prev_flushes > 0 || curr_flushes > 0) {
        regressed.push(format!("flushes: {prev_flushes} -> {curr_flushes}"));
    }

    if submit_delta < 0 && (prev_submits > 0 || curr_submits > 0) {
        improved.push(format!("submits: {prev_submits} -> {curr_submits}"));
    } else if submit_delta > 0 && (prev_submits > 0 || curr_submits > 0) {
        regressed.push(format!("submits: {prev_submits} -> {curr_submits}"));
    }

    if !new_violations.is_empty() {
        regressed.push(format!("{} new violations", new_violations.len()));
    }
    if !resolved_violations.is_empty() {
        improved.push(format!("{} violations resolved", resolved_violations.len()));
    }

    if improved.is_empty() && regressed.is_empty() {
        IterationVerdict::Stalled {
            consecutive_stalls: 1,
        }
    } else if regressed.is_empty() {
        IterationVerdict::Improved {
            summary: improved.join("; "),
        }
    } else if improved.is_empty() {
        IterationVerdict::Regressed {
            summary: regressed.join("; "),
        }
    } else {
        IterationVerdict::Mixed {
            improved,
            regressed,
        }
    }
}

/// Extract a `usize` value from a top-level JSON key.
fn extract_usize(val: &serde_json::Value, key: &str) -> usize {
    val.get(key)
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as usize
}

/// Extract a `usize` from a nested JSON path.
fn extract_usize_path(val: &serde_json::Value, path: &[&str]) -> usize {
    let mut current = val;
    for key in path {
        match current.get(*key) {
            Some(v) => current = v,
            None => return 0,
        }
    }
    current.as_u64().unwrap_or(0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContractStatus, OptimizationReport};
    use nn_dsl::{PerformanceReport, SegmentPerformance};

    fn make_report(iteration: usize, dispatches: usize) -> OptimizationReport {
        let mut seg = SegmentPerformance::new("generator");
        seg.dispatches = dispatches;
        seg.buffer_bytes = dispatches * 1000;
        seg.buffer_naive_bytes = dispatches * 2000;
        let perf = PerformanceReport::from_segments("kokoro", vec![seg]);
        OptimizationReport::new(iteration, "kokoro", &perf).expect("create report")
    }

    #[test]
    fn test_diff_improved() {
        let prev = make_report(0, 100);
        let curr = make_report(1, 80);
        let delta = diff_reports(&prev, &curr);
        assert_eq!(delta.dispatch_delta, -20);
        assert!(matches!(delta.verdict, IterationVerdict::Improved { .. }));
    }

    #[test]
    fn test_diff_regressed() {
        let prev = make_report(0, 80);
        let curr = make_report(1, 100);
        let delta = diff_reports(&prev, &curr);
        assert_eq!(delta.dispatch_delta, 20);
        assert!(matches!(delta.verdict, IterationVerdict::Regressed { .. }));
    }

    #[test]
    fn test_diff_stalled() {
        let prev = make_report(0, 96);
        let curr = make_report(1, 96);
        let delta = diff_reports(&prev, &curr);
        assert_eq!(delta.dispatch_delta, 0);
        assert!(matches!(
            delta.verdict,
            IterationVerdict::Stalled {
                consecutive_stalls: 1
            }
        ));
    }

    #[test]
    fn test_diff_violations() {
        let mut prev = make_report(0, 96);
        prev.contract_status = Some(ContractStatus {
            all_bounds_satisfied: false,
            violations: vec!["output_non_finite".into()],
            tightened_bounds: Vec::new(),
        });
        let mut curr = make_report(1, 96);
        curr.contract_status = Some(ContractStatus {
            all_bounds_satisfied: true,
            violations: Vec::new(),
            tightened_bounds: Vec::new(),
        });
        let delta = diff_reports(&prev, &curr);
        assert_eq!(delta.resolved_violations.len(), 1);
        assert!(delta.new_violations.is_empty());
        // Resolved violations count as improvement.
        assert!(matches!(delta.verdict, IterationVerdict::Improved { .. }));
    }

    #[test]
    fn test_diff_roundtrip() {
        let prev = make_report(0, 100);
        let curr = make_report(1, 80);
        let delta = diff_reports(&prev, &curr);
        let json = serde_json::to_string_pretty(&delta).expect("serialize");
        let parsed: ReportDelta = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.dispatch_delta, -20);
    }
}
