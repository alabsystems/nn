// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Converter feedback loop: structured parity diagnostics.
//!
//! Provides `verify_parity()` which runs L0 (structural) and L3 (numerical)
//! checks on an imported graph, returning a structured `ParityReport`.
//! L2 (NY bounds) is deferred pending #4350.
//!
//! Part of #4349.

use crate::graph_build::ImportedGraph;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Structured parity diagnostic report from `verify_parity()`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ParityReport {
    /// Human-readable model name for the report.
    pub model_name: String,
    /// Whether all non-skipped checks passed.
    pub overall_pass: bool,
    /// Individual check results (both passing and failing).
    pub checks: Vec<ParityCheck>,
}

impl ParityReport {
    /// Create a new report, computing `overall_pass` from the checks.
    pub fn new(model_name: String, checks: Vec<ParityCheck>) -> Self {
        let overall_pass = checks
            .iter()
            .all(|c| matches!(c.status, CheckStatus::Passed | CheckStatus::Skipped));
        Self {
            model_name,
            overall_pass,
            checks,
        }
    }

    /// Return only the failing checks.
    #[must_use]
    pub fn failures(&self) -> Vec<&ParityCheck> {
        self.checks
            .iter()
            .filter(|c| matches!(c.status, CheckStatus::Failed))
            .collect()
    }

    /// Print a human-readable summary to stderr.
    pub fn print(&self) {
        let pass_str = if self.overall_pass { "PASS" } else { "FAIL" };
        eprintln!("Parity report for '{}': {}", self.model_name, pass_str);
        for check in &self.checks {
            let icon = match check.status {
                CheckStatus::Passed => "OK",
                CheckStatus::Failed => "FAIL",
                CheckStatus::Skipped => "SKIP",
            };
            eprint!("  [{icon}] {} (L{})", check.name, check.level as u8);
            if let Some(ref metric) = check.metric {
                eprint!(
                    " cosine={:.6} max_abs={:.6} rms={:.6}",
                    metric.cosine_similarity, metric.max_abs_diff, metric.rms_diff
                );
            }
            if let Some(ref err) = check.detail {
                eprint!(" -- {err}");
            }
            eprintln!();
        }
    }
}

/// Status of a single parity check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckStatus {
    /// Check ran and passed.
    Passed,
    /// Check ran and failed.
    Failed,
    /// Check was skipped (e.g., no reference data provided).
    Skipped,
}

/// A single parity check result.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ParityCheck {
    /// Human-readable name of the check.
    pub name: String,
    /// Which level of the equivalence proof this check belongs to.
    pub level: ParityLevel,
    /// Whether the check passed, failed, or was skipped.
    pub status: CheckStatus,
    /// Numerical metrics (only for L3 numerical parity checks).
    pub metric: Option<ParityMetric>,
    /// Human-readable detail (error message on failure, reason on skip).
    pub detail: Option<String>,
}

impl ParityCheck {
    /// Convenience: create a passing check.
    pub fn passed(name: impl Into<String>, level: ParityLevel) -> Self {
        Self {
            name: name.into(),
            level,
            status: CheckStatus::Passed,
            metric: None,
            detail: None,
        }
    }

    /// Convenience: create a failing check with a detail message.
    pub fn failed(name: impl Into<String>, level: ParityLevel, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            level,
            status: CheckStatus::Failed,
            metric: None,
            detail: Some(detail.into()),
        }
    }

    /// Convenience: create a skipped check with a reason.
    pub fn skipped(name: impl Into<String>, level: ParityLevel, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            level,
            status: CheckStatus::Skipped,
            metric: None,
            detail: Some(reason.into()),
        }
    }
}

/// Which level of the equivalence proof this check belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParityLevel {
    /// L0: Structural -- graph shape, op count, input/output names.
    Structure = 0,
    /// L1: Kani kernel safety (offline, not checked at runtime).
    KernelSafety = 1,
    /// L2: NY IBP composition bounds.
    Bounds = 2,
    /// L3: Numerical parity against reference.
    NumericalParity = 3,
}

/// Numerical metrics from a parity comparison.
#[derive(Debug, Clone)]
pub struct ParityMetric {
    /// Cosine similarity between candidate and reference (1.0 = identical direction).
    pub cosine_similarity: f64,
    /// Maximum absolute difference across all elements.
    pub max_abs_diff: f64,
    /// Root mean square difference across all elements.
    pub rms_diff: f64,
    /// Total elements compared.
    pub element_count: usize,
}

/// Configurable thresholds for parity checks.
#[derive(Debug, Clone)]
pub struct ParityThresholds {
    /// Minimum acceptable cosine similarity (default: 0.999).
    pub cosine_min: f64,
    /// Maximum acceptable absolute difference (default: 0.02).
    pub max_abs_max: f64,
    /// Maximum acceptable RMS difference (default: 0.001).
    pub rms_max: f64,
}

impl Default for ParityThresholds {
    fn default() -> Self {
        Self {
            cosine_min: 0.999,
            max_abs_max: 0.02,
            rms_max: 0.001,
        }
    }
}

// ---------------------------------------------------------------------------
// L0: Structural checks
// ---------------------------------------------------------------------------

/// Expected structural properties for L0 validation.
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct StructuralExpectation {
    /// Expected number of graph nodes (None = skip check).
    pub expected_op_count: Option<usize>,
    /// Expected user input names (None = skip check).
    pub expected_input_names: Option<Vec<String>>,
    /// Expected output names (None = skip check).
    pub expected_output_names: Option<Vec<String>>,
}


/// Run L0 structural checks on an imported graph.
fn check_structure(
    imported: &ImportedGraph,
    expectation: &StructuralExpectation,
) -> Vec<ParityCheck> {
    use nn_core::dyn_tensor::trace::TraceOp;

    let mut checks = Vec::new();

    // Check 1: graph is non-empty.
    let node_count = imported.graph.len();
    if node_count == 0 {
        checks.push(ParityCheck::failed(
            "graph_non_empty",
            ParityLevel::Structure,
            "graph has 0 nodes",
        ));
    } else {
        checks.push(ParityCheck::passed(
            "graph_non_empty",
            ParityLevel::Structure,
        ));
    }

    // Check 2: at least one user input.
    if imported.num_user_inputs == 0 {
        checks.push(ParityCheck::failed(
            "has_user_inputs",
            ParityLevel::Structure,
            "graph has 0 user inputs",
        ));
    } else {
        checks.push(ParityCheck::passed(
            "has_user_inputs",
            ParityLevel::Structure,
        ));
    }

    // Check 3: at least one output.
    if imported.graph.output_node().is_none() {
        checks.push(ParityCheck::failed(
            "has_output",
            ParityLevel::Structure,
            "graph has no output node",
        ));
    } else {
        checks.push(ParityCheck::passed("has_output", ParityLevel::Structure));
    }

    // Check 4: compute op count (non-Input, non-Constant nodes).
    let compute_ops = imported
        .graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op(), TraceOp::Input | TraceOp::Constant { .. }))
        .count();
    if compute_ops == 0 {
        checks.push(ParityCheck::failed(
            "has_compute_ops",
            ParityLevel::Structure,
            "graph has 0 compute ops (only Input/Constant nodes)",
        ));
    } else {
        checks.push(ParityCheck::passed(
            "has_compute_ops",
            ParityLevel::Structure,
        ));
    }

    // Check 5: expected op count (if specified).
    if let Some(expected) = expectation.expected_op_count {
        if node_count == expected {
            checks.push(ParityCheck::passed(
                "op_count_match",
                ParityLevel::Structure,
            ));
        } else {
            checks.push(ParityCheck::failed(
                "op_count_match",
                ParityLevel::Structure,
                format!("expected {expected} nodes, got {node_count}"),
            ));
        }
    }

    // Check 6: input names match (if specified).
    if let Some(ref expected_names) = expectation.expected_input_names {
        if imported.user_input_names == *expected_names {
            checks.push(ParityCheck::passed(
                "input_names_match",
                ParityLevel::Structure,
            ));
        } else {
            checks.push(ParityCheck::failed(
                "input_names_match",
                ParityLevel::Structure,
                format!(
                    "expected {:?}, got {:?}",
                    expected_names, imported.user_input_names
                ),
            ));
        }
    }

    // Check 7: output names match (if specified).
    if let Some(ref expected_names) = expectation.expected_output_names {
        if imported.output_names == *expected_names {
            checks.push(ParityCheck::passed(
                "output_names_match",
                ParityLevel::Structure,
            ));
        } else {
            checks.push(ParityCheck::failed(
                "output_names_match",
                ParityLevel::Structure,
                format!(
                    "expected {:?}, got {:?}",
                    expected_names, imported.output_names
                ),
            ));
        }
    }

    checks
}

// ---------------------------------------------------------------------------
// L3: Numerical parity checks
// ---------------------------------------------------------------------------

/// Compute parity metrics between a candidate and reference f32 slice.
///
/// Returns `None` if the slices have different lengths or are empty.
pub fn compute_parity_metric(candidate: &[f32], reference: &[f32]) -> Option<ParityMetric> {
    if candidate.len() != reference.len() || candidate.is_empty() {
        return None;
    }

    let n = candidate.len();

    // Cosine similarity.
    let mut dot = 0.0_f64;
    let mut norm_c = 0.0_f64;
    let mut norm_r = 0.0_f64;
    let mut sum_sq_diff = 0.0_f64;
    let mut max_abs = 0.0_f64;

    for i in 0..n {
        let c = f64::from(candidate[i]);
        let r = f64::from(reference[i]);
        dot += c * r;
        norm_c += c * c;
        norm_r += r * r;
        let diff = (c - r).abs();
        sum_sq_diff += diff * diff;
        if diff > max_abs {
            max_abs = diff;
        }
    }

    let denom = norm_c.sqrt() * norm_r.sqrt();
    let cosine_similarity = if denom > 0.0 { dot / denom } else { 1.0 };
    let rms_diff = (sum_sq_diff / n as f64).sqrt();

    Some(ParityMetric {
        cosine_similarity,
        max_abs_diff: max_abs,
        rms_diff,
        element_count: n,
    })
}

/// Run L3 numerical parity checks between candidate and reference outputs.
///
/// `outputs` maps output name to (candidate_data, reference_data) pairs.
fn check_numerical_parity(
    outputs: &[(&str, &[f32], &[f32])],
    thresholds: &ParityThresholds,
) -> Vec<ParityCheck> {
    let mut checks = Vec::new();

    for &(name, candidate, reference) in outputs {
        let metric = match compute_parity_metric(candidate, reference) {
            Some(m) => m,
            None => {
                let detail = if candidate.len() != reference.len() {
                    format!(
                        "shape mismatch: candidate has {} elements, reference has {}",
                        candidate.len(),
                        reference.len()
                    )
                } else {
                    "empty tensors".to_string()
                };
                checks.push(ParityCheck::failed(
                    format!("numerical_parity:{name}"),
                    ParityLevel::NumericalParity,
                    detail,
                ));
                continue;
            }
        };

        let mut failures = Vec::new();

        if metric.cosine_similarity < thresholds.cosine_min {
            failures.push(format!(
                "cosine {:.6} < {:.6}",
                metric.cosine_similarity, thresholds.cosine_min
            ));
        }
        if metric.max_abs_diff > thresholds.max_abs_max {
            failures.push(format!(
                "max_abs {:.6} > {:.6}",
                metric.max_abs_diff, thresholds.max_abs_max
            ));
        }
        if metric.rms_diff > thresholds.rms_max {
            failures.push(format!(
                "rms {:.6} > {:.6}",
                metric.rms_diff, thresholds.rms_max
            ));
        }

        let mut check = if failures.is_empty() {
            ParityCheck::passed(
                format!("numerical_parity:{name}"),
                ParityLevel::NumericalParity,
            )
        } else {
            ParityCheck::failed(
                format!("numerical_parity:{name}"),
                ParityLevel::NumericalParity,
                failures.join("; "),
            )
        };
        check.metric = Some(metric);
        checks.push(check);
    }

    checks
}

// ---------------------------------------------------------------------------
// Top-level verify_parity()
// ---------------------------------------------------------------------------

/// Run the parity diagnostic pipeline on an imported graph.
///
/// Checks (in order):
/// 1. L0 Structure: graph has expected ops, inputs, outputs
/// 2. L3 Parity: if reference outputs provided, computes cosine_similarity and max_abs_diff
///
/// L2 (NY bounds) is deferred pending #4350.
///
/// Returns a `ParityReport` with all check results. The report includes
/// both passing and failing checks for complete diagnostic visibility.
pub fn verify_parity(
    imported: &ImportedGraph,
    model_name: &str,
    structural: &StructuralExpectation,
    reference_outputs: Option<&[(&str, &[f32], &[f32])]>,
    thresholds: &ParityThresholds,
) -> ParityReport {
    let mut checks = Vec::new();

    // L0: Structural checks.
    checks.extend(check_structure(imported, structural));

    // L3: Numerical parity (if reference data is provided).
    match reference_outputs {
        Some(outputs) if !outputs.is_empty() => {
            checks.extend(check_numerical_parity(outputs, thresholds));
        }
        _ => {
            checks.push(ParityCheck::skipped(
                "numerical_parity",
                ParityLevel::NumericalParity,
                "no reference outputs provided",
            ));
        }
    }

    ParityReport::new(model_name.to_string(), checks)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "convert_parity_tests.rs"]
mod tests;
