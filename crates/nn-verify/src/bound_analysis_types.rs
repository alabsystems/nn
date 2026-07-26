// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Type definitions for bound analysis: [`LayerAnalysis`], [`BoundAnalysisReport`],
//! [`TighteningRecommendation`], [`TighteningTarget`], and [`AnalysisConfig`].

use serde::{Deserialize, Serialize};

use crate::verify_types::PropMethod;

/// Per-layer derived metrics from bound analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LayerAnalysis {
    /// Index of this layer in the network graph (0-based).
    pub layer_index: usize,
    /// Layer type name (e.g. "Linear", "ReLU", "LayerNorm").
    pub layer_type: String,
    /// Graph node name for mapping back to NY.
    pub node_name: Option<String>,
    /// Average output interval width across all elements.
    pub avg_output_width: f32,
    /// Maximum output interval width across all elements.
    pub max_output_width: f32,
    /// Expansion ratio: output_width / input_width.
    /// `f32::INFINITY` when input width is zero or non-finite.
    pub expansion_ratio: f32,
    /// Propagation method used for this layer.
    pub method: PropMethod,
    /// Whether this layer is flagged as a bound explosion point.
    pub is_explosion_point: bool,
    /// Whether any bound in this layer is non-finite (NaN or Inf).
    pub has_non_finite_bounds: bool,
}

/// Full model bound analysis report — the primary output of one pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BoundAnalysisReport {
    /// Name of the model or kernel that was analyzed.
    pub model_name: String,
    /// Total number of layers analyzed.
    pub total_layers: usize,
    /// Per-layer analysis results.
    pub layers: Vec<LayerAnalysis>,
    /// Indices of layers flagged as bound explosion points.
    pub explosion_points: Vec<usize>,
    /// Maximum output interval width of the final layer.
    pub output_width: f32,
    /// Whether the final layer's output bounds are all finite.
    pub output_is_finite: bool,
    /// Fraction of layers using CROWN (vs IBP) bounds.
    pub crown_coverage: f32,
    /// Machine-readable recommendations for tightening.
    pub recommendations: Vec<TighteningRecommendation>,
    /// ISO 8601 timestamp of when the analysis was performed.
    pub analyzed_at: String,
    /// Longest chain of consecutive normalization layers in the model graph.
    #[serde(default)]
    pub chained_norm_depth: usize,
    /// max(f32_output / f64_output) across all output elements.
    /// `None` until an F64 reference forward pass is available.
    #[serde(default)]
    pub precision_drift_ratio: Option<f32>,
    /// Estimated per-layer drift: `1.0 - ratio^(1/depth)`.
    /// `None` until `precision_drift_ratio` is populated.
    #[serde(default)]
    pub drift_per_layer: Option<f32>,
}

/// Machine-readable recommendation targeting model, nn, or NY.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TighteningRecommendation {
    /// A specific layer has wide bounds that need tightening.
    TightenLayer {
        layer_index: usize,
        node_name: Option<String>,
        layer_type: String,
        current_width: f32,
        expansion_ratio: f32,
        target: TighteningTarget,
        suggestion: String,
    },
    /// Layer uses IBP but bounds are wide enough to warrant CROWN.
    EscalateToCrown {
        layer_index: usize,
        node_name: Option<String>,
        layer_type: String,
        ibp_width: f32,
    },
    /// Switch normalization bounds mode for tighter results.
    SwitchNormMode {
        layer_index: usize,
        node_name: Option<String>,
        layer_type: String,
        current_width: f32,
        suggested_mode: String,
        target: TighteningTarget,
    },
    /// Small subgraph suitable for exact ay SMT verification.
    ExtractForSmt {
        start_layer: usize,
        end_layer: usize,
        estimated_elements: usize,
    },
    /// Suggest inserting a constraint (clamp) at a specific layer.
    AnnotateConstraint {
        layer_index: usize,
        node_name: Option<String>,
        suggested_range: (f32, f32),
        reason: String,
    },
    /// F32 vs F64 precision drift exceeds threshold for a deep norm chain.
    /// Flagged when `chained_norm_depth > precision_risk_depth_threshold`
    /// AND `precision_drift_ratio < precision_risk_drift_threshold`.
    PrecisionRisk {
        /// Longest normalization chain depth in the model.
        chained_norm_depth: usize,
        /// Measured F32/F64 output ratio (< 1.0 means F32 attenuates).
        precision_drift_ratio: f32,
        /// Per-layer drift estimate: `1.0 - ratio^(1/depth)`.
        drift_per_layer: f32,
    },
    /// A chain of consecutive normalization layers has cumulative bounds
    /// growth exceeding the configured threshold (default: 10x).
    NormChainExplosion {
        /// Index of the first norm layer in the chain.
        start_layer: usize,
        /// Index of the last norm layer in the chain.
        end_layer: usize,
        /// Number of normalization layers in the chain.
        chain_depth: usize,
        /// Total output_width / input_width across the entire chain.
        total_expansion: f32,
        /// Per-layer expansion factors within the chain.
        per_layer_expansions: Vec<f32>,
        /// Layer type names within the chain.
        layer_types: Vec<String>,
    },
}

/// Which codebase a recommendation targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TighteningTarget {
    /// Modify model source code (insert constraints, replace unbounded ops).
    Model,
    /// Modify nn nn layers / bound propagation rules.
    Framework,
    /// Modify NY Layer implementations.
    Verifier,
}

/// Configuration for bound analysis.
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Expansion ratio threshold above which a layer is flagged as an
    /// explosion point. Default: 100.0.
    pub explosion_threshold: f32,
    /// Output width threshold above which IBP-only layers get an
    /// `EscalateToCrown` recommendation. Default: 1e4.
    pub crown_escalation_width: f32,
    /// Maximum total elements for `ExtractForSmt` recommendations.
    /// Default: 256.
    pub smt_max_elements: usize,
    /// Minimum number of consecutive normalization layers to qualify as a
    /// "norm chain" for explosion detection. Default: 5.
    pub norm_chain_min_length: usize,
    /// Total expansion ratio threshold across a norm chain above which a
    /// `NormChainExplosion` recommendation is generated. Default: 10.0.
    pub norm_chain_explosion_ratio: f32,
    /// Minimum chained norm depth to qualify for `PrecisionRisk`.
    /// Default: 20.
    pub precision_risk_depth_threshold: usize,
    /// F32/F64 ratio below which `PrecisionRisk` is flagged.
    /// Default: 0.95.
    pub precision_risk_drift_threshold: f32,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            explosion_threshold: 100.0,
            crown_escalation_width: 1e4,
            smt_max_elements: 256,
            norm_chain_min_length: 5,
            norm_chain_explosion_ratio: 10.0,
            precision_risk_depth_threshold: 20,
            precision_risk_drift_threshold: 0.95,
        }
    }
}
