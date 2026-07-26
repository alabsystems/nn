// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! One-pass bound analysis for progressive tightening compilation.
//!
//! Analyzes per-layer bound records from a verification run and produces a
//! structured [`BoundAnalysisReport`] with per-layer diagnostics, explosion
//! point detection, and machine-readable [`TighteningRecommendation`]s.
//!
//! This module is the Phase 1 implementation of the progressive tightening
//! architecture described in `designs/2026-03-15-progressive-tightening-compilation.md`.
//!
//! # Usage
//!
//! ```rust,no_run
//! use nn_verify::bound_analysis::{analyze_layer_bounds, AnalysisConfig};
//!
//! // After verification, given Vec<LayerBoundRecord> from extract_layer_bounds():
//! // let report = analyze_layer_bounds("nn_model", &records, &AnalysisConfig::default());
//! ```

#[path = "bound_analysis_types.rs"]
mod types;
pub use types::{
    AnalysisConfig, BoundAnalysisReport, LayerAnalysis, TighteningRecommendation, TighteningTarget,
};

use crate::certificate::now_iso8601;
use crate::certificate_types::LayerBoundRecord;

// ---------------------------------------------------------------------------
// Norm layer type detection
// ---------------------------------------------------------------------------

/// Returns `true` if the layer type string indicates a normalization layer.
pub(super) fn is_norm_layer(layer_type: &str) -> bool {
    matches!(
        layer_type,
        "LayerNorm"
            | "RMSNorm"
            | "InstanceNorm"
            | "BatchNorm"
            | "GroupNorm"
            | "InstanceNorm1d"
            | "InstanceNorm2d"
    )
}

/// Returns `true` if the layer type is an unbounded exponential-family op.
pub(super) fn is_exp_family(layer_type: &str) -> bool {
    matches!(layer_type, "Exp" | "Pow" | "Softmax" | "LogSoftmax")
}

// ---------------------------------------------------------------------------
// Width computation
// ---------------------------------------------------------------------------

/// Compute the maximum interval width from a slice of (lower, upper) pairs.
///
/// Returns `f32::INFINITY` if any bound is non-finite.
pub(super) fn max_width(bounds: &[(f32, f32)]) -> f32 {
    if bounds.is_empty() {
        return 0.0;
    }
    let mut max = 0.0f32;
    for &(lo, hi) in bounds {
        if !lo.is_finite() || !hi.is_finite() {
            return f32::INFINITY;
        }
        let w = hi - lo;
        if !w.is_finite() {
            return f32::INFINITY;
        }
        if w > max {
            max = w;
        }
    }
    max
}

/// Compute the average interval width from a slice of (lower, upper) pairs.
///
/// Returns `f32::INFINITY` if any bound is non-finite.
fn avg_width(bounds: &[(f32, f32)]) -> f32 {
    if bounds.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for &(lo, hi) in bounds {
        if !lo.is_finite() || !hi.is_finite() {
            return f32::INFINITY;
        }
        let w = f64::from(hi - lo);
        if !w.is_finite() {
            return f32::INFINITY;
        }
        sum += w;
    }
    (sum / bounds.len() as f64) as f32
}

/// Check if any bound pair contains non-finite values.
fn has_non_finite(bounds: &[(f32, f32)]) -> bool {
    bounds
        .iter()
        .any(|&(lo, hi)| !lo.is_finite() || !hi.is_finite())
}

/// Compute the longest contiguous run of normalization layers.
fn longest_norm_chain(records: &[LayerBoundRecord]) -> usize {
    let mut max_len = 0;
    let mut current_len = 0;
    for record in records {
        if is_norm_layer(&record.layer_type) {
            current_len += 1;
            if current_len > max_len {
                max_len = current_len;
            }
        } else {
            current_len = 0;
        }
    }
    max_len
}

// ---------------------------------------------------------------------------
// Core analysis
// ---------------------------------------------------------------------------

/// Analyze per-layer bounds from a verification run.
///
/// Computes per-layer width metrics, detects explosion points, and generates
/// machine-readable tightening recommendations.
///
/// # Arguments
///
/// * `model_name` — Name of the model/kernel for the report.
/// * `records` — Per-layer bound records from `extract_layer_bounds()`.
/// * `config` — Analysis thresholds and cost budgets.
#[must_use]
pub fn analyze_layer_bounds(
    model_name: &str,
    records: &[LayerBoundRecord],
    config: &AnalysisConfig,
) -> BoundAnalysisReport {
    let mut layers = Vec::with_capacity(records.len());
    let mut explosion_points = Vec::new();
    let mut crown_count = 0usize;

    for record in records {
        let out_max_w = max_width(&record.output_bounds);
        let out_avg_w = avg_width(&record.output_bounds);
        let in_max_w = max_width(&record.input_bounds);

        // Expansion ratio: guard against zero/non-finite input width.
        let expansion_ratio = if !in_max_w.is_finite() || !out_max_w.is_finite() {
            f32::INFINITY
        } else if in_max_w <= f32::EPSILON {
            if out_max_w <= f32::EPSILON {
                1.0
            } else {
                f32::INFINITY
            }
        } else {
            out_max_w / in_max_w
        };

        let is_explosion = expansion_ratio.is_finite()
            && expansion_ratio > config.explosion_threshold
            || !expansion_ratio.is_finite();

        if is_explosion {
            explosion_points.push(record.layer_index);
        }

        if record.method.is_tight() {
            crown_count += 1;
        }

        layers.push(LayerAnalysis {
            layer_index: record.layer_index,
            layer_type: record.layer_type.clone(),
            node_name: record.node_name.clone(),
            avg_output_width: out_avg_w,
            max_output_width: out_max_w,
            expansion_ratio,
            method: record.method,
            is_explosion_point: is_explosion,
            has_non_finite_bounds: has_non_finite(&record.output_bounds),
        });
    }

    // Output summary from last layer.
    let (output_width, output_is_finite) = layers
        .last()
        .map(|l| (l.max_output_width, !l.has_non_finite_bounds))
        .unwrap_or((0.0, true));

    let crown_coverage = if records.is_empty() {
        0.0
    } else {
        crown_count as f32 / records.len() as f32
    };

    // Generate recommendations.
    let recommendations = generate_recommendations(records, &layers, config);

    let chained_norm_depth = longest_norm_chain(records);

    BoundAnalysisReport {
        model_name: model_name.to_string(),
        total_layers: records.len(),
        layers,
        explosion_points,
        output_width,
        output_is_finite,
        crown_coverage,
        recommendations,
        analyzed_at: now_iso8601(),
        chained_norm_depth,
        precision_drift_ratio: None,
        drift_per_layer: None,
    }
}

// -- Recommendation generation extracted to bound_analysis_recommendations.rs (Wave 4 D3b) --

#[path = "bound_analysis_recommendations.rs"]
mod recommendations;
use recommendations::generate_recommendations;

// ---------------------------------------------------------------------------
// Selective CROWN escalation helpers (Phase 2, #2454)
// ---------------------------------------------------------------------------

/// Extract layer indices that need CROWN escalation from a report's recommendations.
///
/// Scans [`BoundAnalysisReport::recommendations`] for [`TighteningRecommendation::EscalateToCrown`]
/// entries and returns their `layer_index` values as a sorted, deduplicated set.
///
/// This is the bridge between Phase 1 (analysis + recommendations) and Phase 2
/// (selective CROWN re-run): after an initial IBP pass, call `analyze_layer_bounds`
/// to get a report, then `layers_needing_crown` to identify which layers should
/// be re-verified with CROWN. Remaining layers keep their cheaper IBP bounds.
///
/// Returns an empty `Vec` when no layers need escalation (IBP was sufficient
/// everywhere, or all wide layers already used a tight method).
///
/// # Example
///
/// ```rust,no_run
/// use nn_verify::bound_analysis::{analyze_layer_bounds, layers_needing_crown, AnalysisConfig};
///
/// // let report = analyze_layer_bounds("model", &records, &AnalysisConfig::default());
/// // let crown_layers = layers_needing_crown(&report);
/// // assert!(crown_layers.is_empty() || crown_layers.iter().all(|&i| i < report.total_layers));
/// ```
#[must_use]
pub fn layers_needing_crown(report: &BoundAnalysisReport) -> Vec<usize> {
    let mut indices: Vec<usize> = report
        .recommendations
        .iter()
        .filter_map(|rec| match rec {
            TighteningRecommendation::EscalateToCrown { layer_index, .. } => Some(*layer_index),
            _ => None,
        })
        .collect();
    indices.sort_unstable();
    indices.dedup();
    indices
}

/// Serialize a `BoundAnalysisReport` to pretty-printed JSON.
///
/// # Errors
///
/// Returns `serde_json::Error` if serialization fails (should not happen
/// for well-formed reports).
pub fn report_to_json(report: &BoundAnalysisReport) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(report)
}

#[path = "bound_analysis_drift.rs"]
mod drift;
pub use drift::estimate_norm_chain_precision_drift;

#[cfg(kani)]
#[path = "kani_bound_widening.rs"]
mod kani_bound_widening;

#[cfg(test)]
#[path = "bound_analysis_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "bound_analysis_perf_tests.rs"]
mod perf_tests;
