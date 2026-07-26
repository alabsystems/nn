// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tightening recommendation generation for bound analysis.
//!
//! Extracted from `bound_analysis.rs` (Wave 4 D3b). Generates machine-readable
//! recommendations from per-layer analysis results.

use crate::certificate_types::LayerBoundRecord;
use crate::verify_types::PropMethod;

use super::{
    is_exp_family, is_norm_layer, max_width, AnalysisConfig, LayerAnalysis,
    TighteningRecommendation, TighteningTarget,
};

/// Generate tightening recommendations from layer analysis.
pub(super) fn generate_recommendations(
    records: &[LayerBoundRecord],
    layers: &[LayerAnalysis],
    config: &AnalysisConfig,
) -> Vec<TighteningRecommendation> {
    let mut recs = Vec::new();

    for (i, (record, analysis)) in records.iter().zip(layers.iter()).enumerate() {
        // Non-finite bounds: immediate TightenLayer recommendation.
        if analysis.has_non_finite_bounds {
            recs.push(TighteningRecommendation::TightenLayer {
                layer_index: record.layer_index,
                node_name: record.node_name.clone(),
                layer_type: record.layer_type.clone(),
                current_width: analysis.max_output_width,
                expansion_ratio: analysis.expansion_ratio,
                target: TighteningTarget::Framework,
                suggestion: "Non-finite bounds detected. Check numerical stability.".to_string(),
            });
            continue;
        }

        // Norm layers with high expansion: suggest ForwardMode.
        if is_norm_layer(&record.layer_type) && analysis.expansion_ratio > 50.0 {
            recs.push(TighteningRecommendation::SwitchNormMode {
                layer_index: record.layer_index,
                node_name: record.node_name.clone(),
                layer_type: record.layer_type.clone(),
                current_width: analysis.max_output_width,
                suggested_mode: "ForwardMode".to_string(),
                target: TighteningTarget::Framework,
            });
            continue;
        }

        // IBP-only layers with wide bounds: suggest CROWN escalation.
        if record.method == PropMethod::Ibp
            && analysis.max_output_width > config.crown_escalation_width
        {
            recs.push(TighteningRecommendation::EscalateToCrown {
                layer_index: record.layer_index,
                node_name: record.node_name.clone(),
                layer_type: record.layer_type.clone(),
                ibp_width: analysis.max_output_width,
            });
        }

        // Layers following unbounded ops: suggest clamp constraint.
        if i > 0 && is_exp_family(&records[i - 1].layer_type) && analysis.is_explosion_point {
            let suggested_lo = -10.0;
            let suggested_hi = 10.0;
            recs.push(TighteningRecommendation::AnnotateConstraint {
                layer_index: record.layer_index,
                node_name: record.node_name.clone(),
                suggested_range: (suggested_lo, suggested_hi),
                reason: format!(
                    "Following unbounded {} layer with {:.1}x expansion",
                    records[i - 1].layer_type,
                    analysis.expansion_ratio
                ),
            });
        }
    }

    // Norm chain explosion: detect >threshold growth through consecutive norms.
    detect_norm_chain_explosions(records, layers, &mut recs, config);

    // SMT extraction: find small consecutive subgraphs within cost budget.
    recommend_smt_extraction(records, &mut recs, config);

    recs
}

/// Detect chains of consecutive normalization layers where cumulative bounds
/// growth exceeds the configured threshold.
///
/// Walks the layer list, identifies maximal runs of norm layers, and emits a
/// `NormChainExplosion` recommendation for each chain of length >=
/// `config.norm_chain_min_length` whose total expansion exceeds
/// `config.norm_chain_explosion_ratio`.
fn detect_norm_chain_explosions(
    records: &[LayerBoundRecord],
    layers: &[LayerAnalysis],
    recs: &mut Vec<TighteningRecommendation>,
    config: &AnalysisConfig,
) {
    let n = records.len();
    let mut i = 0;

    while i < n {
        if !is_norm_layer(&records[i].layer_type) {
            i += 1;
            continue;
        }

        // Found start of a norm chain — extend to find maximal run.
        let chain_start = i;
        while i < n && is_norm_layer(&records[i].layer_type) {
            i += 1;
        }
        let chain_end = i - 1; // inclusive
        let chain_depth = chain_end - chain_start + 1;

        if chain_depth < config.norm_chain_min_length {
            continue;
        }

        // Compute total expansion: output width of last / input width of first.
        let chain_input_width = max_width(&records[chain_start].input_bounds);
        let chain_output_width = layers[chain_end].max_output_width;

        let total_expansion = if !chain_input_width.is_finite() || !chain_output_width.is_finite() {
            f32::INFINITY
        } else if chain_input_width <= f32::EPSILON {
            if chain_output_width <= f32::EPSILON {
                1.0
            } else {
                f32::INFINITY
            }
        } else {
            chain_output_width / chain_input_width
        };

        if !total_expansion.is_finite() || total_expansion > config.norm_chain_explosion_ratio {
            let per_layer_expansions: Vec<f32> = layers[chain_start..=chain_end]
                .iter()
                .map(|l| l.expansion_ratio)
                .collect();
            let layer_types: Vec<String> = records[chain_start..=chain_end]
                .iter()
                .map(|r| r.layer_type.clone())
                .collect();

            recs.push(TighteningRecommendation::NormChainExplosion {
                start_layer: records[chain_start].layer_index,
                end_layer: records[chain_end].layer_index,
                chain_depth,
                total_expansion,
                per_layer_expansions,
                layer_types,
            });
        }
    }
}

/// Identify small subgraphs suitable for ay SMT verification.
///
/// Sliding window finds the longest consecutive run of layers whose total
/// output elements fit within `config.smt_max_elements`. Emits at most one
/// `ExtractForSmt` recommendation for the largest qualifying window.
fn recommend_smt_extraction(
    records: &[LayerBoundRecord],
    recs: &mut Vec<TighteningRecommendation>,
    config: &AnalysisConfig,
) {
    if records.is_empty() {
        return;
    }

    // Single sliding-window pass to find the largest qualifying window.
    let mut best_start = 0;
    let mut best_end = 0;
    let mut best_size = 0;
    let mut start = 0;
    let mut total_elements = 0usize;

    for (end, record) in records.iter().enumerate() {
        let layer_elements = record.output_bounds.len();
        total_elements = total_elements.saturating_add(layer_elements);

        // Shrink window from left until within budget.
        while total_elements > config.smt_max_elements && start <= end {
            total_elements = total_elements.saturating_sub(records[start].output_bounds.len());
            start += 1;
        }

        let window_len = end.saturating_sub(start) + 1;
        if window_len >= 2 && total_elements <= config.smt_max_elements && window_len > best_size {
            best_start = start;
            best_end = end;
            best_size = window_len;
        }
    }

    if best_size >= 2 {
        let estimated: usize = records[best_start..=best_end]
            .iter()
            .map(|r| r.output_bounds.len())
            .sum();
        // Use record.layer_index (semantic layer ID), not array position.
        // Array positions only equal layer_index when indices are contiguous
        // from 0 — sparse graphs (graph-aware verification) diverge.
        recs.push(TighteningRecommendation::ExtractForSmt {
            start_layer: records[best_start].layer_index,
            end_layer: records[best_end].layer_index,
            estimated_elements: estimated,
        });
    }
}
