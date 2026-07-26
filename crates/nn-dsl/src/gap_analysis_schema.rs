// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! JSON schema and agent protocol types for [`FusionGapAnalysis`].
//!
//! Provides:
//! - A JSON Schema definition for the `FusionGapAnalysis` output format.
//! - Convenience serialization/deserialization methods on `FusionGapAnalysis`.
//! - Agent protocol types (`OptimizationRequest`, `OptimizationResponse`,
//!   `OptimizationSuggestion`) for AI-agent-driven optimization.
//!
//! Part of #3833 (Self-Optimizing ML Compiler, Phase 5).

use crate::cost_model::CostEstimate;
use crate::trace_compile::FusionGapAnalysis;

/// Protocol version for the optimization agent interface.
pub const PROTOCOL_VERSION: &str = "0.1.0";

/// Returns the JSON schema for [`FusionGapAnalysis`] as a `serde_json::Value`.
///
/// Used by the `nn optimize` CLI to document the expected output format
/// for AI agent consumption.
#[must_use]
pub fn fusion_gap_analysis_schema() -> serde_json::Value {
    serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "FusionGapAnalysis",
        "description": "Result of analyzing all fusion gaps in a compiled execution plan.",
        "type": "object",
        "required": ["gaps", "total_dispatches", "theoretical_minimum"],
        "properties": {
            "gaps": {
                "type": "array",
                "description": "All identified fusion gaps.",
                "items": { "$ref": "#/$defs/FusionGap" }
            },
            "total_dispatches": {
                "type": "integer",
                "minimum": 0,
                "description": "Total dispatch steps in the plan (Dispatch + NativeOp)."
            },
            "theoretical_minimum": {
                "type": "integer",
                "minimum": 0,
                "description": "Theoretical minimum dispatches if all closable gaps were fused."
            }
        },
        "$defs": {
            "FusionGap": {
                "type": "object",
                "description": "A single fusion gap between two adjacent plan steps.",
                "required": ["step_a", "step_b", "kernel_a", "kernel_b", "reason", "savings"],
                "properties": {
                    "step_a": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Index of the first step in the plan."
                    },
                    "step_b": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Index of the second step in the plan."
                    },
                    "kernel_a": {
                        "type": "string",
                        "description": "Kernel name of step A (if Dispatch)."
                    },
                    "kernel_b": {
                        "type": "string",
                        "description": "Kernel name of step B (if Dispatch)."
                    },
                    "reason": {
                        "$ref": "#/$defs/FusionBlocker"
                    },
                    "savings": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Estimated dispatch savings if this gap were closed (usually 1)."
                    }
                }
            },
            "FusionBlocker": {
                "type": "string",
                "description": "Why a pair of adjacent dispatches was not fused.",
                "enum": [
                    "FanOut",
                    "ShapeMismatch",
                    "NonFusibleOp",
                    "NotDispatch",
                    "AlreadyOptimal",
                    "NoPeepholePattern",
                    "NoDependency"
                ]
            }
        }
    })
}

impl FusionGapAnalysis {
    /// Serialize this analysis to a `serde_json::Value`.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("FusionGapAnalysis is always serializable")
    }

    /// Serialize this analysis to a pretty-printed JSON string.
    #[must_use]
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("FusionGapAnalysis is always serializable")
    }

    /// Deserialize a `FusionGapAnalysis` from a `serde_json::Value`.
    ///
    /// # Errors
    ///
    /// Returns `serde_json::Error` if the value does not match the expected
    /// schema.
    pub fn from_json(value: &serde_json::Value) -> Result<Self, serde_json::Error> {
        serde_json::from_value(value.clone())
    }
}

/// Request sent to an AI agent for optimization suggestions.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OptimizationRequest {
    /// The gap analysis results.
    pub gap_analysis: FusionGapAnalysis,
    /// Model name (e.g., "kokoro").
    pub model_name: String,
    /// Protocol version.
    pub protocol_version: String,
    /// Maximum number of suggestions requested.
    pub max_suggestions: usize,
}

/// A single optimization suggestion from an AI agent.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OptimizationSuggestion {
    /// Which gap this suggestion addresses (index into gaps array).
    pub gap_index: usize,
    /// Type of optimization: "peephole_pass", "native_op", "scheduling_directive".
    pub optimization_type: String,
    /// Human-readable description.
    pub description: String,
    /// Estimated dispatch savings.
    pub estimated_savings: usize,
}

/// Response from an AI agent.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OptimizationResponse {
    /// Protocol version.
    pub protocol_version: String,
    /// Suggestions, sorted by estimated_savings descending.
    pub suggestions: Vec<OptimizationSuggestion>,
}

/// A single segment entry within a [`GapAnalysisReport`].
///
/// Captures the fusion gap analysis, cost estimate, and dispatch counts for
/// one named segment of a model (e.g., "plbert", "generator").
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GapAnalysisSegment {
    /// Segment name (e.g., "plbert", "text", "generator").
    pub segment_name: String,
    /// Fusion gap analysis with per-gap blocker classification.
    pub gap_analysis: FusionGapAnalysis,
    /// Roofline-based cost estimate.
    pub cost_estimate: CostEstimate,
    /// Total dispatch steps in the compiled plan.
    pub dispatch_count: usize,
    /// Theoretical minimum dispatches if all closable gaps were fused.
    pub theoretical_minimum: usize,
}

/// Multi-segment gap analysis report for JSON persistence and cross-run comparison.
///
/// Wraps per-segment fusion gap results with model-level metadata. Designed for
/// the self-optimizing loop: serialize after analysis, deserialize to compare
/// against prior runs and detect regressions or improvements.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GapAnalysisReport {
    /// Protocol version for forward-compatible parsing.
    pub protocol_version: String,
    /// Model name (e.g., "kokoro", "htdemucs").
    pub model_name: String,
    /// Per-segment analysis results.
    pub segments: Vec<GapAnalysisSegment>,
    /// Total dispatch count across all segments.
    pub total_dispatches: usize,
    /// Sum of theoretical minimums across all segments.
    pub total_theoretical_minimum: usize,
}

impl GapAnalysisReport {
    /// Build a report from per-segment results.
    ///
    /// Computes `total_dispatches` and `total_theoretical_minimum` from the
    /// segment entries.
    #[must_use]
    pub fn new(model_name: impl Into<String>, segments: Vec<GapAnalysisSegment>) -> Self {
        let total_dispatches = segments.iter().map(|s| s.dispatch_count).sum();
        let total_theoretical_minimum = segments.iter().map(|s| s.theoretical_minimum).sum();
        Self {
            protocol_version: PROTOCOL_VERSION.to_string(),
            model_name: model_name.into(),
            segments,
            total_dispatches,
            total_theoretical_minimum,
        }
    }

    /// Serialize this report to a `serde_json::Value`.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("GapAnalysisReport is always serializable")
    }

    /// Serialize this report to a pretty-printed JSON string.
    #[must_use]
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("GapAnalysisReport is always serializable")
    }

    /// Deserialize a `GapAnalysisReport` from a `serde_json::Value`.
    ///
    /// # Errors
    ///
    /// Returns `serde_json::Error` if the value does not match the expected
    /// schema.
    pub fn from_json(value: &serde_json::Value) -> Result<Self, serde_json::Error> {
        serde_json::from_value(value.clone())
    }

    /// Deserialize a `GapAnalysisReport` from a JSON string.
    ///
    /// Convenience method for loading persisted reports from disk.
    ///
    /// # Errors
    ///
    /// Returns `serde_json::Error` if the string is not valid JSON or does
    /// not match the expected schema.
    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace_compile::{FusionBlocker, FusionGap};

    fn sample_analysis() -> FusionGapAnalysis {
        FusionGapAnalysis {
            gaps: vec![
                FusionGap {
                    step_a: 0,
                    step_b: 1,
                    kernel_a: "relu".into(),
                    kernel_b: "exp".into(),
                    reason: FusionBlocker::FanOut,
                    savings: 1,
                },
                FusionGap {
                    step_a: 3,
                    step_b: 4,
                    kernel_a: "matmul".into(),
                    kernel_b: "softmax".into(),
                    reason: FusionBlocker::NonFusibleOp,
                    savings: 0,
                },
            ],
            total_dispatches: 10,
            theoretical_minimum: 9,
        }
    }

    fn sample_cost_estimate() -> CostEstimate {
        CostEstimate {
            total_ns: 42_500.0,
            per_step_ns: vec![(0, 12_500.0), (1, 30_000.0)],
            dispatch_count: 2,
        }
    }

    fn sample_segment(name: &str) -> GapAnalysisSegment {
        GapAnalysisSegment {
            segment_name: name.to_string(),
            gap_analysis: sample_analysis(),
            cost_estimate: sample_cost_estimate(),
            dispatch_count: 10,
            theoretical_minimum: 9,
        }
    }

    fn sample_report() -> GapAnalysisReport {
        GapAnalysisReport::new(
            "kokoro",
            vec![sample_segment("plbert"), sample_segment("generator")],
        )
    }

    #[test]
    fn test_round_trip_to_json_from_json() {
        let original = sample_analysis();
        let json = original.to_json();
        let restored =
            FusionGapAnalysis::from_json(&json).expect("round-trip deserialization should succeed");

        // Compare via re-serialization since FusionGapAnalysis does not derive PartialEq.
        assert_eq!(original.to_json(), restored.to_json());
    }

    #[test]
    fn test_round_trip_preserves_all_gap_fields() {
        let original = sample_analysis();
        let json = original.to_json();
        let restored = FusionGapAnalysis::from_json(&json).expect("round-trip should succeed");

        assert_eq!(restored.total_dispatches, 10);
        assert_eq!(restored.theoretical_minimum, 9);
        assert_eq!(restored.gaps.len(), 2);

        assert_eq!(restored.gaps[0].step_a, 0);
        assert_eq!(restored.gaps[0].step_b, 1);
        assert_eq!(restored.gaps[0].kernel_a, "relu");
        assert_eq!(restored.gaps[0].kernel_b, "exp");
        assert_eq!(restored.gaps[0].reason, FusionBlocker::FanOut);
        assert_eq!(restored.gaps[0].savings, 1);

        assert_eq!(restored.gaps[1].step_a, 3);
        assert_eq!(restored.gaps[1].kernel_b, "softmax");
        assert_eq!(restored.gaps[1].reason, FusionBlocker::NonFusibleOp);
        assert_eq!(restored.gaps[1].savings, 0);
    }

    #[test]
    fn test_round_trip_via_json_string() {
        let original = sample_analysis();
        let json_str = original.to_json_pretty();
        let parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("pretty JSON should parse");
        let restored =
            FusionGapAnalysis::from_json(&parsed).expect("round-trip from string should succeed");
        assert_eq!(original.to_json(), restored.to_json());
    }

    #[test]
    fn test_to_json_pretty_is_valid_json() {
        let analysis = sample_analysis();
        let pretty = analysis.to_json_pretty();
        let parsed: serde_json::Value =
            serde_json::from_str(&pretty).expect("pretty JSON should be valid");
        assert!(parsed.is_object());
        assert_eq!(parsed["total_dispatches"], 10);
        assert_eq!(parsed["theoretical_minimum"], 9);
        assert_eq!(parsed["gaps"].as_array().expect("gaps is array").len(), 2);
    }

    #[test]
    fn test_schema_has_required_fields() {
        let schema = fusion_gap_analysis_schema();
        let required = schema["required"]
            .as_array()
            .expect("schema should have required array");
        let required_strs: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(required_strs.contains(&"gaps"));
        assert!(required_strs.contains(&"total_dispatches"));
        assert!(required_strs.contains(&"theoretical_minimum"));
    }

    #[test]
    fn test_schema_has_defs() {
        let schema = fusion_gap_analysis_schema();
        let defs = schema["$defs"]
            .as_object()
            .expect("schema should have $defs");
        assert!(defs.contains_key("FusionGap"));
        assert!(defs.contains_key("FusionBlocker"));
    }

    #[test]
    fn test_schema_fusion_blocker_enum_variants() {
        let schema = fusion_gap_analysis_schema();
        let variants = schema["$defs"]["FusionBlocker"]["enum"]
            .as_array()
            .expect("FusionBlocker should have enum array");
        let variant_strs: Vec<&str> = variants.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(variant_strs.contains(&"FanOut"));
        assert!(variant_strs.contains(&"ShapeMismatch"));
        assert!(variant_strs.contains(&"NonFusibleOp"));
        assert!(variant_strs.contains(&"NotDispatch"));
        assert!(variant_strs.contains(&"AlreadyOptimal"));
        assert!(variant_strs.contains(&"NoPeepholePattern"));
        assert!(variant_strs.contains(&"NoDependency"));
    }

    #[test]
    fn test_optimization_request_serializes() {
        let request = OptimizationRequest {
            gap_analysis: sample_analysis(),
            model_name: "kokoro".into(),
            protocol_version: PROTOCOL_VERSION.into(),
            max_suggestions: 5,
        };
        let json = serde_json::to_value(&request).expect("request should serialize");
        assert_eq!(json["model_name"], "kokoro");
        assert_eq!(json["protocol_version"], PROTOCOL_VERSION);
        assert_eq!(json["max_suggestions"], 5);
        assert!(json["gap_analysis"].is_object());
    }

    #[test]
    fn test_optimization_response_deserializes() {
        let json = serde_json::json!({
            "protocol_version": "0.1.0",
            "suggestions": [
                {
                    "gap_index": 0,
                    "optimization_type": "peephole_pass",
                    "description": "Add relu-exp fusion peephole pass",
                    "estimated_savings": 1
                },
                {
                    "gap_index": 2,
                    "optimization_type": "native_op",
                    "description": "Wrap matmul+softmax as NativeOp",
                    "estimated_savings": 3
                }
            ]
        });
        let response: OptimizationResponse =
            serde_json::from_value(json).expect("response should deserialize");
        assert_eq!(response.protocol_version, "0.1.0");
        assert_eq!(response.suggestions.len(), 2);
        assert_eq!(response.suggestions[0].gap_index, 0);
        assert_eq!(response.suggestions[0].optimization_type, "peephole_pass");
        assert_eq!(response.suggestions[1].estimated_savings, 3);
    }

    #[test]
    fn test_protocol_version_constant() {
        assert_eq!(PROTOCOL_VERSION, "0.1.0");
    }

    // -- GapAnalysisReport tests --

    #[test]
    fn test_report_new_computes_totals() {
        let report = sample_report();
        assert_eq!(report.model_name, "kokoro");
        assert_eq!(report.protocol_version, PROTOCOL_VERSION);
        assert_eq!(report.segments.len(), 2);
        // Each segment has dispatch_count=10, theoretical_minimum=9
        assert_eq!(report.total_dispatches, 20);
        assert_eq!(report.total_theoretical_minimum, 18);
    }

    #[test]
    fn test_report_round_trip_json_value() {
        let original = sample_report();
        let json = original.to_json();
        let restored = GapAnalysisReport::from_json(&json).expect("round-trip should succeed");

        assert_eq!(restored.model_name, "kokoro");
        assert_eq!(restored.protocol_version, PROTOCOL_VERSION);
        assert_eq!(restored.segments.len(), 2);
        assert_eq!(restored.total_dispatches, 20);
        assert_eq!(restored.total_theoretical_minimum, 18);

        // Verify segment-level fields survive
        assert_eq!(restored.segments[0].segment_name, "plbert");
        assert_eq!(restored.segments[0].dispatch_count, 10);
        assert_eq!(restored.segments[0].theoretical_minimum, 9);
        assert_eq!(restored.segments[0].gap_analysis.gaps.len(), 2);
        assert!((restored.segments[0].cost_estimate.total_ns - 42_500.0).abs() < f64::EPSILON);
        assert_eq!(restored.segments[0].cost_estimate.per_step_ns.len(), 2);
        assert_eq!(restored.segments[0].cost_estimate.dispatch_count, 2);

        assert_eq!(restored.segments[1].segment_name, "generator");
    }

    #[test]
    fn test_report_round_trip_json_string() {
        let original = sample_report();
        let json_str = original.to_json_pretty();
        let restored =
            GapAnalysisReport::from_json_str(&json_str).expect("string round-trip should succeed");

        // Full re-serialization equality
        assert_eq!(original.to_json(), restored.to_json());
    }

    #[test]
    fn test_report_empty_segments() {
        let report = GapAnalysisReport::new("empty_model", vec![]);
        assert_eq!(report.total_dispatches, 0);
        assert_eq!(report.total_theoretical_minimum, 0);
        assert!(report.segments.is_empty());

        let json = report.to_json();
        let restored =
            GapAnalysisReport::from_json(&json).expect("empty report round-trip should succeed");
        assert!(restored.segments.is_empty());
        assert_eq!(restored.total_dispatches, 0);
    }

    #[test]
    fn test_report_from_json_rejects_invalid() {
        let bad_json = serde_json::json!({"not": "a report"});
        let result = GapAnalysisReport::from_json(&bad_json);
        assert!(result.is_err());
    }

    #[test]
    fn test_report_from_json_str_rejects_invalid() {
        let result = GapAnalysisReport::from_json_str("not valid json");
        assert!(result.is_err());
    }

    #[test]
    fn test_cost_estimate_round_trip() {
        let original = sample_cost_estimate();
        let json = serde_json::to_value(&original).expect("CostEstimate should serialize");
        let restored: CostEstimate =
            serde_json::from_value(json).expect("CostEstimate should deserialize");
        assert!((restored.total_ns - 42_500.0).abs() < f64::EPSILON);
        assert_eq!(restored.per_step_ns.len(), 2);
        assert_eq!(restored.per_step_ns[0], (0, 12_500.0));
        assert_eq!(restored.per_step_ns[1], (1, 30_000.0));
        assert_eq!(restored.dispatch_count, 2);
    }
}
