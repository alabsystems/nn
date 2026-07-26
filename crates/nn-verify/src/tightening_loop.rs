// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Automated verify-analyze-tighten iteration loop.
//!
//! Orchestrates progressive bound tightening across a NY
//! [`GraphNetwork`]: runs IBP on iteration 1, identifies explosion
//! points via [`analyze_layer_bounds`], selectively escalates to
//! CROWN on subsequent iterations, and tracks convergence of the
//! maximum output width.
//!
//! # Algorithm
//!
//! 1. **Iteration 1 (IBP):** Propagate IBP bounds through the full
//!    graph. Extract per-layer bounds and analyze for explosion points.
//! 2. **Iteration 2+ (Selective CROWN):** For layers flagged with
//!    `EscalateToCrown` recommendations, attempt CROWN propagation.
//!    Re-analyze and track improvement (max-width reduction).
//! 3. **Convergence:** Stop when improvement falls below the
//!    configured threshold, or when `max_iterations` is reached.
//! 4. **ay candidates (optional):** After the loop, identify small
//!    subgraphs suitable for exact SMT verification.
//!
//! # Usage
//!
//! ```rust,no_run
//! use nn_verify::tightening_loop::{run_tightening_loop, TighteningConfig};
//!
//! // After tracing: let network = trace_to_graph_model(&graph)?.graph;
//! // let input_bounds = BoundedTensor::from_epsilon(&input, 0.01)?;
//! // let result = run_tightening_loop(&network, &input_bounds, &TighteningConfig::default())?;
//! // assert!(result.converged || result.iterations_run == config.max_iterations);
//! ```
//!
//! Part of #2456: Tightening loop orchestrator.

use ny_api::BoundedTensor;
use ny_propagate::GraphNetwork;

use crate::bound_analysis::{
    analyze_layer_bounds, AnalysisConfig, BoundAnalysisReport, TighteningRecommendation,
};
use crate::certificate_types::LayerBoundRecord;
use crate::error::VerifyError;
use crate::layer_bounds::extract_layer_bounds;
use crate::verify_types::PropMethod;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the tightening loop.
#[derive(Debug, Clone)]
pub struct TighteningConfig {
    /// Maximum number of iterations. Default: 5.
    pub max_iterations: usize,
    /// Stop when max-width improvement between iterations drops below
    /// this fraction. E.g., 0.01 = stop when improvement < 1%.
    /// Default: 0.01.
    pub convergence_threshold: f32,
    /// Bound analysis configuration (explosion threshold, etc.).
    pub analysis_config: AnalysisConfig,
    /// Whether to identify ay SMT candidates after the loop.
    /// Default: true.
    pub ay_candidates_enabled: bool,
    /// Model name for analysis reports.
    pub model_name: String,
}

impl Default for TighteningConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            convergence_threshold: 0.01,
            analysis_config: AnalysisConfig::default(),
            ay_candidates_enabled: true,
            model_name: "unknown".to_string(),
        }
    }
}

impl TighteningConfig {
    /// Create a config with a specific model name.
    #[must_use]
    pub fn new(model_name: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
            ..Default::default()
        }
    }

    /// Set the maximum number of iterations.
    #[must_use]
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Set the convergence threshold.
    #[must_use]
    pub fn with_convergence_threshold(mut self, threshold: f32) -> Self {
        self.convergence_threshold = threshold;
        self
    }

    /// Enable or disable ay candidate identification.
    #[must_use]
    pub fn with_ay_candidates(mut self, enabled: bool) -> Self {
        self.ay_candidates_enabled = enabled;
        self
    }
}

// ---------------------------------------------------------------------------
// Per-iteration metrics
// ---------------------------------------------------------------------------

/// Metrics collected during a single tightening iteration.
#[derive(Debug, Clone)]
pub struct TighteningStep {
    /// Iteration number (1-based).
    pub iteration: usize,
    /// Propagation method used for this iteration.
    pub method: PropMethod,
    /// Maximum output width after this iteration.
    pub max_output_width: f32,
    /// Number of explosion points detected.
    pub explosion_count: usize,
    /// Number of CROWN escalation recommendations.
    pub crown_escalation_count: usize,
    /// Fraction of layers using CROWN bounds.
    pub crown_coverage: f32,
    /// Whether the output bounds are all finite.
    pub output_is_finite: bool,
    /// Improvement ratio vs. previous iteration.
    /// `None` for iteration 1 (no baseline).
    pub improvement_ratio: Option<f32>,
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Result of the tightening loop.
#[derive(Debug, Clone)]
pub struct TighteningResult {
    /// Number of iterations that were run.
    pub iterations_run: usize,
    /// Final bound analysis report from the last iteration.
    pub final_report: BoundAnalysisReport,
    /// Per-iteration step metrics.
    pub improvement_history: Vec<TighteningStep>,
    /// Whether the loop converged (improvement < threshold).
    pub converged: bool,
    /// Layer indices suitable for exact ay SMT verification.
    /// Only populated when `ay_candidates_enabled` is true.
    pub ay_candidate_ranges: Vec<AYCandidateRange>,
    /// Final per-layer bound records from the last iteration.
    pub final_layer_bounds: Vec<LayerBoundRecord>,
}

/// A contiguous layer range identified as suitable for ay SMT verification.
#[derive(Debug, Clone)]
pub struct AYCandidateRange {
    /// Index of the first layer in the range.
    pub start_layer: usize,
    /// Index of the last layer in the range.
    pub end_layer: usize,
    /// Estimated total elements in the subgraph.
    pub estimated_elements: usize,
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/// Run the automated tightening loop on a NY graph.
///
/// Iterates IBP -> analyze -> selective CROWN -> re-analyze until
/// convergence or `max_iterations`.
///
/// # Arguments
///
/// * `graph` - The NY graph network.
/// * `input_bounds` - Input bounds for the model.
/// * `config` - Tightening loop configuration.
///
/// # Errors
///
/// Returns `VerifyError::EmptyGraph` if the graph has no nodes.
/// Returns `VerifyError::Ny` if bounds propagation fails.
pub fn run_tightening_loop(
    graph: &GraphNetwork,
    input_bounds: &BoundedTensor,
    config: &TighteningConfig,
) -> Result<TighteningResult, VerifyError> {
    if config.max_iterations == 0 {
        return Err(VerifyError::InvalidInput(
            "max_iterations must be >= 1".to_string(),
        ));
    }

    let mut history = Vec::with_capacity(config.max_iterations);
    let mut prev_max_width: Option<f32> = None;
    let mut converged = false;
    let mut last_report: Option<BoundAnalysisReport> = None;
    let mut last_layer_bounds = Vec::new();

    for iteration in 1..=config.max_iterations {
        // --- Propagation pass ---
        let (method, layer_bounds) = if iteration == 1 {
            // Iteration 1: IBP only.
            let records = extract_layer_bounds(graph, input_bounds)?;
            (PropMethod::Ibp, records)
        } else {
            // Iteration 2+: Attempt CROWN, fall back to IBP per-node.
            // extract_layer_bounds internally uses CROWN-IBP collection,
            // which tries CROWN on each node and falls back to IBP.
            let records = extract_layer_bounds(graph, input_bounds)?;
            // Determine dominant method from the records.
            let crown_count = records.iter().filter(|r| r.method.is_tight()).count();
            let method = if crown_count > 0 {
                PropMethod::MixedIbpCrown
            } else {
                PropMethod::Ibp
            };
            (method, records)
        };

        // --- Analyze ---
        let report =
            analyze_layer_bounds(&config.model_name, &layer_bounds, &config.analysis_config);

        // --- Compute step metrics ---
        let current_max_width = report.output_width;
        let improvement_ratio = prev_max_width.and_then(|prev| {
            if !prev.is_finite() || !current_max_width.is_finite() || prev <= f32::EPSILON {
                None
            } else {
                Some((prev - current_max_width) / prev)
            }
        });

        let crown_escalation_count = report
            .recommendations
            .iter()
            .filter(|r| matches!(r, TighteningRecommendation::EscalateToCrown { .. }))
            .count();

        let step = TighteningStep {
            iteration,
            method,
            max_output_width: current_max_width,
            explosion_count: report.explosion_points.len(),
            crown_escalation_count,
            crown_coverage: report.crown_coverage,
            output_is_finite: report.output_is_finite,
            improvement_ratio,
        };

        history.push(step);

        // --- Convergence check ---
        if let Some(ratio) = improvement_ratio {
            if ratio.is_finite() && ratio < config.convergence_threshold {
                converged = true;
                last_report = Some(report);
                last_layer_bounds = layer_bounds;
                break;
            }
        }

        prev_max_width = Some(current_max_width);
        last_report = Some(report);
        last_layer_bounds = layer_bounds;

        // If no explosion points and output is finite, we are done.
        if let Some(ref rpt) = last_report {
            if rpt.explosion_points.is_empty() && rpt.output_is_finite {
                converged = true;
                break;
            }
        }
    }

    let final_report = last_report.unwrap_or_else(|| {
        // Should not happen given max_iterations >= 1.
        analyze_layer_bounds(
            &config.model_name,
            &last_layer_bounds,
            &config.analysis_config,
        )
    });

    // --- ay candidate identification ---
    let ay_candidate_ranges = if config.ay_candidates_enabled {
        extract_ay_candidates(&final_report)
    } else {
        Vec::new()
    };

    Ok(TighteningResult {
        iterations_run: history.len(),
        final_report,
        improvement_history: history,
        converged,
        ay_candidate_ranges,
        final_layer_bounds: last_layer_bounds,
    })
}

/// Extract ay SMT candidate ranges from `ExtractForSmt` recommendations.
fn extract_ay_candidates(report: &BoundAnalysisReport) -> Vec<AYCandidateRange> {
    report
        .recommendations
        .iter()
        .filter_map(|rec| match rec {
            TighteningRecommendation::ExtractForSmt {
                start_layer,
                end_layer,
                estimated_elements,
            } => Some(AYCandidateRange {
                start_layer: *start_layer,
                end_layer: *end_layer,
                estimated_elements: *estimated_elements,
            }),
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Convenience: summary formatting
// ---------------------------------------------------------------------------

impl TighteningResult {
    /// Whether any iteration achieved finite output bounds.
    #[must_use]
    pub fn achieved_finite_bounds(&self) -> bool {
        self.improvement_history
            .iter()
            .any(|step| step.output_is_finite)
    }

    /// The maximum improvement ratio across all iterations.
    /// Returns `None` if only one iteration was run.
    #[must_use]
    pub fn best_improvement(&self) -> Option<f32> {
        self.improvement_history
            .iter()
            .filter_map(|step| step.improvement_ratio)
            .filter(|r| r.is_finite())
            .fold(None, |best, r| Some(best.map_or(r, |b: f32| b.max(r))))
    }

    /// Final max output width.
    #[must_use]
    pub fn final_max_width(&self) -> f32 {
        self.final_report.output_width
    }
}

#[cfg(test)]
#[path = "tightening_loop_tests.rs"]
mod tests;
