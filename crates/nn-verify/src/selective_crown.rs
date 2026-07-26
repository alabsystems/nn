// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Selective per-layer CROWN escalation guided by [`BoundAnalysisReport`] explosion points.
//!
//! Instead of running CROWN on the entire graph (O(L^2) backward passes) or
//! using IBP everywhere, this module identifies "explosion point" layers where
//! IBP bounds blow up and escalates only those layers to CROWN.
//!
//! # Architecture
//!
//! 1. Run a fast IBP pass to get per-layer bounds.
//! 2. Analyze bounds via [`analyze_layer_bounds`] to find explosion points.
//! 3. Select layers for CROWN via [`select_crown_layers`] using configurable strategy.
//! 4. Re-run with CROWN on selected layers only (O(K * L) where K << L).
//!
//! # Usage
//!
//! ```rust,no_run
//! use nn_verify::selective_crown::{
//!     SelectiveCrownConfig, EscalationStrategy, select_crown_layers,
//! };
//! use nn_verify::bound_analysis::{analyze_layer_bounds, AnalysisConfig};
//!
//! // After IBP pass, analyze bounds:
//! // let report = analyze_layer_bounds("model", &records, &AnalysisConfig::default());
//! // let config = SelectiveCrownConfig::default();
//! // let crown_layers = select_crown_layers(&report, &config);
//! ```
//!
//! Issue: #2454.

use serde::{Deserialize, Serialize};

use crate::bound_analysis::{
    analyze_layer_bounds, layers_needing_crown, AnalysisConfig, BoundAnalysisReport,
};
use crate::certificate_types::LayerBoundRecord;
use crate::verify_types::PropMethod;

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Strategy for selecting which layers get CROWN escalation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[derive(Default)]
pub enum EscalationStrategy {
    /// Escalate all layers with output width above `min_width_to_tighten`,
    /// up to `max_crown_layers` (widest first if capped).
    #[default]
    WidestFirst,
    /// Escalate all layers above threshold without a cap on count.
    /// May be slow for models with many wide layers.
    AllAboveThreshold,
}

/// Configuration for selective CROWN escalation.
///
/// Controls which layers are promoted from IBP to CROWN based on their
/// output interval width from an initial IBP pass.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SelectiveCrownConfig {
    /// Layers with maximum output width above this threshold are candidates
    /// for CROWN escalation. Default: 5.0.
    ///
    /// Lower values escalate more layers (tighter bounds, slower).
    /// Higher values escalate fewer layers (looser bounds, faster).
    pub min_width_to_tighten: f32,
    /// Maximum number of layers to escalate to CROWN. Default: 10.
    ///
    /// Only applies when `escalation_strategy` is [`EscalationStrategy::WidestFirst`].
    /// When the number of candidate layers exceeds this cap, only the widest
    /// are selected.
    pub max_crown_layers: usize,
    /// Strategy for selecting layers when more candidates than `max_crown_layers`.
    /// Default: [`EscalationStrategy::WidestFirst`].
    pub escalation_strategy: EscalationStrategy,
}

impl Default for SelectiveCrownConfig {
    fn default() -> Self {
        Self {
            min_width_to_tighten: 5.0,
            max_crown_layers: 10,
            escalation_strategy: EscalationStrategy::WidestFirst,
        }
    }
}

impl SelectiveCrownConfig {
    /// Create a config with custom width threshold.
    #[must_use]
    pub fn with_min_width(mut self, min_width: f32) -> Self {
        self.min_width_to_tighten = min_width;
        self
    }

    /// Set the maximum number of CROWN layers.
    #[must_use]
    pub fn with_max_crown_layers(mut self, max: usize) -> Self {
        self.max_crown_layers = max;
        self
    }

    /// Set the escalation strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: EscalationStrategy) -> Self {
        self.escalation_strategy = strategy;
        self
    }

    /// Build an [`AnalysisConfig`] with `crown_escalation_width` set to
    /// this config's `min_width_to_tighten`.
    ///
    /// This bridges `SelectiveCrownConfig` to the existing analysis pipeline,
    /// which uses `AnalysisConfig::crown_escalation_width` to generate
    /// `EscalateToCrown` recommendations.
    #[must_use]
    pub fn to_analysis_config(&self) -> AnalysisConfig {
        AnalysisConfig {
            crown_escalation_width: self.min_width_to_tighten,
            ..AnalysisConfig::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Layer selection
// ---------------------------------------------------------------------------

/// Select layer indices that should use CROWN instead of IBP.
///
/// Analyzes a [`BoundAnalysisReport`] (from an initial IBP pass) and returns
/// the indices of layers whose IBP bounds are wider than
/// [`SelectiveCrownConfig::min_width_to_tighten`] and that currently use IBP.
///
/// # Strategy behavior
///
/// - [`EscalationStrategy::WidestFirst`]: Returns up to `max_crown_layers`
///   indices, sorted by descending output width. The returned `Vec` is sorted
///   by layer index for deterministic downstream processing.
///
/// - [`EscalationStrategy::AllAboveThreshold`]: Returns all qualifying layers
///   regardless of count. `max_crown_layers` is ignored.
///
/// # Returns
///
/// Sorted, deduplicated layer indices. Empty if no layers need escalation.
#[must_use]
pub fn select_crown_layers(
    report: &BoundAnalysisReport,
    config: &SelectiveCrownConfig,
) -> Vec<usize> {
    // Collect candidate layers: IBP-only with output width above threshold.
    let mut candidates: Vec<(usize, f32)> = report
        .layers
        .iter()
        .filter(|layer| {
            !layer.method.is_tight()
                && layer.max_output_width.is_finite()
                && layer.max_output_width > config.min_width_to_tighten
        })
        .map(|layer| (layer.layer_index, layer.max_output_width))
        .collect();

    match config.escalation_strategy {
        EscalationStrategy::WidestFirst => {
            // Sort by width descending, then take top N.
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(config.max_crown_layers);
        }
        EscalationStrategy::AllAboveThreshold => {
            // No cap — all qualifying layers.
        }
    }

    // Sort by layer index for deterministic processing order.
    let mut indices: Vec<usize> = candidates.into_iter().map(|(idx, _)| idx).collect();
    indices.sort_unstable();
    indices.dedup();
    indices
}

/// Convenience: analyze layer records and select CROWN layers in one call.
///
/// Equivalent to running [`analyze_layer_bounds`] with an [`AnalysisConfig`]
/// derived from `crown_config`, then [`select_crown_layers`] on the result.
///
/// Returns `(report, crown_layer_indices)`.
#[must_use]
pub fn analyze_and_select(
    model_name: &str,
    records: &[LayerBoundRecord],
    crown_config: &SelectiveCrownConfig,
) -> (BoundAnalysisReport, Vec<usize>) {
    let analysis_config = crown_config.to_analysis_config();
    let report = analyze_layer_bounds(model_name, records, &analysis_config);
    let crown_layers = select_crown_layers(&report, crown_config);
    (report, crown_layers)
}

// ---------------------------------------------------------------------------
// Selective CROWN verification result
// ---------------------------------------------------------------------------

/// Result of selective CROWN escalation guided by BoundAnalysisReport.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SelectiveCrownAnalysis {
    /// Layer indices that were targeted for CROWN tightening.
    pub crown_layer_indices: Vec<usize>,
    /// Number of layers that actually received CROWN bounds (may be less than
    /// `crown_layer_indices.len()` if CROWN failed at some nodes).
    pub crown_tightened_count: usize,
    /// Bound analysis report from the initial IBP pass.
    pub ibp_analysis: BoundAnalysisReport,
    /// Bound analysis report after selective CROWN tightening.
    pub crown_analysis: BoundAnalysisReport,
    /// Per-layer records from the IBP pass (input to analysis).
    pub ibp_records: Vec<LayerBoundRecord>,
    /// Per-layer records after CROWN tightening (simulated or actual).
    pub crown_records: Vec<LayerBoundRecord>,
}

/// Apply simulated CROWN tightening to layer records for analysis comparison.
///
/// Given IBP layer records and a set of layer indices to "tighten", produces
/// new records where selected layers have their method changed to CROWN.
/// The actual bounds are not changed (real CROWN would tighten them); this is
/// for pipeline analysis and strategy validation without requiring a live
/// NY graph.
///
/// For actual NY integration, see [`verify_with_selective_crown`] in
/// `pipeline_tensor.rs` which calls `propagate_crown` on the real graph.
#[must_use]
pub fn simulate_crown_tightening(
    records: &[LayerBoundRecord],
    crown_layers: &[usize],
) -> Vec<LayerBoundRecord> {
    records
        .iter()
        .map(|record| {
            if crown_layers.contains(&record.layer_index) && !record.method.is_tight() {
                let mut tightened = record.clone();
                tightened.method = PropMethod::Crown;
                tightened
            } else {
                record.clone()
            }
        })
        .collect()
}

/// Run selective CROWN escalation analysis on pre-computed layer records.
///
/// This is the analysis-only path that does not require a live NY
/// graph. It:
///
/// 1. Analyzes IBP records to find explosion points.
/// 2. Selects layers for CROWN via the configured strategy.
/// 3. Simulates CROWN tightening (marks selected layers as CROWN method).
/// 4. Re-analyzes to produce a comparison report.
///
/// For actual NY-backed selective CROWN verification, use
/// `pipeline_tensor::verify_with_selective_crown` which calls `propagate_crown`
/// on the real graph.
#[must_use]
pub fn analyze_selective_crown(
    model_name: &str,
    records: &[LayerBoundRecord],
    config: &SelectiveCrownConfig,
) -> SelectiveCrownAnalysis {
    let analysis_config = config.to_analysis_config();

    // 1. Analyze IBP records.
    let ibp_analysis = analyze_layer_bounds(model_name, records, &analysis_config);

    // 2. Select layers for CROWN escalation.
    let crown_layer_indices = select_crown_layers(&ibp_analysis, config);

    // 3. If no layers need CROWN, fast-path return.
    if crown_layer_indices.is_empty() {
        return SelectiveCrownAnalysis {
            crown_layer_indices,
            crown_tightened_count: 0,
            ibp_analysis: ibp_analysis.clone(),
            crown_analysis: ibp_analysis,
            ibp_records: records.to_vec(),
            crown_records: records.to_vec(),
        };
    }

    // 4. Simulate CROWN tightening.
    let crown_records = simulate_crown_tightening(records, &crown_layer_indices);
    let crown_tightened_count = crown_layer_indices.len();

    // 5. Re-analyze with CROWN method annotations.
    let crown_model_name = format!("{model_name}_selective_crown");
    let crown_analysis = analyze_layer_bounds(&crown_model_name, &crown_records, &analysis_config);

    SelectiveCrownAnalysis {
        crown_layer_indices,
        crown_tightened_count,
        ibp_analysis,
        crown_analysis,
        ibp_records: records.to_vec(),
        crown_records,
    }
}

// ---------------------------------------------------------------------------
// Integration with layers_needing_crown (backward compat bridge)
// ---------------------------------------------------------------------------

/// Bridge from [`BoundAnalysisReport`] recommendations to selective CROWN config.
///
/// Uses the existing [`layers_needing_crown`] function (which reads
/// `EscalateToCrown` recommendations) and applies the `SelectiveCrownConfig`
/// cap/strategy on top.
///
/// This allows code that already uses `layers_needing_crown` to add strategy
/// control without changing the recommendation generation pipeline.
#[must_use]
pub fn select_from_recommendations(
    report: &BoundAnalysisReport,
    config: &SelectiveCrownConfig,
) -> Vec<usize> {
    let mut candidates = layers_needing_crown(report);

    match config.escalation_strategy {
        EscalationStrategy::WidestFirst if candidates.len() > config.max_crown_layers => {
            // Sort candidates by their output width (widest first).
            candidates.sort_by(|&a, &b| {
                let width_a = report
                    .layers
                    .iter()
                    .find(|l| l.layer_index == a)
                    .map_or(0.0, |l| l.max_output_width);
                let width_b = report
                    .layers
                    .iter()
                    .find(|l| l.layer_index == b)
                    .map_or(0.0, |l| l.max_output_width);
                width_b
                    .partial_cmp(&width_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(config.max_crown_layers);
            candidates.sort_unstable();
        }
        _ => {}
    }

    candidates
}

#[cfg(test)]
#[path = "selective_crown_tests.rs"]
mod tests;
