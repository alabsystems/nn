// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the fusion gap analyzer and GapAnalysisReport infrastructure.
//!
//! Validates:
//! - GapAnalysisReport creation, serialization, and deserialization
//! - Tier classification of gaps (FusionBlocker variants)
//! - Blocker distribution counting
//! - Gap analysis on simple graphs with known unfused patterns
//! - Empty graph produces empty report
//! - Optimization opportunity percentage calculations
//! - FusionGapAnalysis.summarize() format

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use crate::cost_model::CostEstimate;
use crate::gap_analysis_schema::{
    GapAnalysisReport, GapAnalysisSegment, OptimizationRequest, OptimizationResponse,
    OptimizationSuggestion, PROTOCOL_VERSION,
};
use crate::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use crate::trace_compile::{
    analyze_fusion_gaps, CompiledKernel, CompiledPlan, CompiledStep, FusionBlocker, FusionGap,
    FusionGapAnalysis,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_dispatch(name: &str) -> CompiledStep {
    let node_id = TensorNodeId::new(0);
    let input_node = TensorNode::new(
        node_id,
        TensorOpKind::Input {
            name: "input_0".into(),
            shape: vec![1, 4],
        },
        vec![1, 4],
    );
    let def = TensorKernelDef::new(name, vec![input_node], node_id);
    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: None,
    }
}

fn make_trace_node(
    id: u64,
    name: &str,
    op: TraceOp,
    inputs: Vec<u64>,
    shape: Vec<usize>,
) -> TraceNode {
    TraceNode::new(id, name.to_string(), op, inputs, shape, DType::F32)
}

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
        total_ns: 50_000.0,
        per_step_ns: vec![(0, 25_000.0), (1, 25_000.0)],
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

// ---------------------------------------------------------------------------
// Empty graph produces empty analysis
// ---------------------------------------------------------------------------

#[test]
fn test_empty_graph_produces_empty_analysis() {
    let plan = CompiledPlan {
        steps: vec![],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    let graph = ComputationGraph::from_nodes(vec![]);
    let analysis = analyze_fusion_gaps(&plan, &graph);

    assert!(analysis.gaps.is_empty());
    assert_eq!(analysis.total_dispatches, 0);
    assert_eq!(analysis.theoretical_minimum, 0);
}

#[test]
fn test_empty_analysis_optimization_opportunity_is_zero() {
    let analysis = FusionGapAnalysis {
        gaps: vec![],
        total_dispatches: 0,
        theoretical_minimum: 0,
    };
    assert!((analysis.optimization_opportunity_pct() - 0.0).abs() < f64::EPSILON);
}

// ---------------------------------------------------------------------------
// Blocker distribution counting
// ---------------------------------------------------------------------------

#[test]
fn test_blocker_counts_empty() {
    let analysis = FusionGapAnalysis {
        gaps: vec![],
        total_dispatches: 5,
        theoretical_minimum: 5,
    };
    assert!(analysis.blocker_counts().is_empty());
}

#[test]
fn test_blocker_counts_single_type() {
    let analysis = FusionGapAnalysis {
        gaps: vec![
            FusionGap {
                step_a: 0,
                step_b: 1,
                kernel_a: "a".into(),
                kernel_b: "b".into(),
                reason: FusionBlocker::FanOut,
                savings: 1,
            },
            FusionGap {
                step_a: 2,
                step_b: 3,
                kernel_a: "c".into(),
                kernel_b: "d".into(),
                reason: FusionBlocker::FanOut,
                savings: 1,
            },
        ],
        total_dispatches: 10,
        theoretical_minimum: 8,
    };
    let counts = analysis.blocker_counts();
    assert_eq!(counts.len(), 1);
    assert_eq!(counts.get("FanOut"), Some(&2));
}

#[test]
fn test_blocker_counts_multiple_types() {
    let analysis = FusionGapAnalysis {
        gaps: vec![
            FusionGap {
                step_a: 0,
                step_b: 1,
                kernel_a: "a".into(),
                kernel_b: "b".into(),
                reason: FusionBlocker::NonFusibleOp,
                savings: 0,
            },
            FusionGap {
                step_a: 1,
                step_b: 2,
                kernel_a: "b".into(),
                kernel_b: "c".into(),
                reason: FusionBlocker::FanOut,
                savings: 1,
            },
            FusionGap {
                step_a: 2,
                step_b: 3,
                kernel_a: "c".into(),
                kernel_b: "d".into(),
                reason: FusionBlocker::ShapeMismatch,
                savings: 1,
            },
            FusionGap {
                step_a: 3,
                step_b: 4,
                kernel_a: "d".into(),
                kernel_b: "e".into(),
                reason: FusionBlocker::NonFusibleOp,
                savings: 0,
            },
        ],
        total_dispatches: 20,
        theoretical_minimum: 18,
    };
    let counts = analysis.blocker_counts();
    assert_eq!(counts.len(), 3);
    assert_eq!(counts.get("NonFusibleOp"), Some(&2));
    assert_eq!(counts.get("FanOut"), Some(&1));
    assert_eq!(counts.get("ShapeMismatch"), Some(&1));
}

// ---------------------------------------------------------------------------
// Optimization opportunity percentage
// ---------------------------------------------------------------------------

#[test]
fn test_optimization_opportunity_pct_normal() {
    let analysis = FusionGapAnalysis {
        gaps: vec![],
        total_dispatches: 200,
        theoretical_minimum: 80,
    };
    // (200 - 80) / 200 * 100 = 60.0%
    assert!((analysis.optimization_opportunity_pct() - 60.0).abs() < 0.01);
}

#[test]
fn test_optimization_opportunity_pct_no_savings() {
    let analysis = FusionGapAnalysis {
        gaps: vec![],
        total_dispatches: 50,
        theoretical_minimum: 50,
    };
    assert!((analysis.optimization_opportunity_pct() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_optimization_opportunity_pct_all_saveable() {
    let analysis = FusionGapAnalysis {
        gaps: vec![],
        total_dispatches: 100,
        theoretical_minimum: 0,
    };
    assert!((analysis.optimization_opportunity_pct() - 100.0).abs() < 0.01);
}

// ---------------------------------------------------------------------------
// FusionBlocker Display
// ---------------------------------------------------------------------------

#[test]
fn test_fusion_blocker_display_all_variants() {
    assert_eq!(format!("{}", FusionBlocker::FanOut), "FanOut");
    assert_eq!(format!("{}", FusionBlocker::ShapeMismatch), "ShapeMismatch");
    assert_eq!(format!("{}", FusionBlocker::NonFusibleOp), "NonFusibleOp");
    assert_eq!(format!("{}", FusionBlocker::NotDispatch), "NotDispatch");
    assert_eq!(
        format!("{}", FusionBlocker::AlreadyOptimal),
        "AlreadyOptimal"
    );
    assert_eq!(
        format!("{}", FusionBlocker::NoPeepholePattern),
        "NoPeepholePattern"
    );
    assert_eq!(format!("{}", FusionBlocker::NoDependency), "NoDependency");
}

// ---------------------------------------------------------------------------
// Summarize format
// ---------------------------------------------------------------------------

#[test]
fn test_summarize_empty_gaps_no_top_blockers() {
    let analysis = FusionGapAnalysis {
        gaps: vec![],
        total_dispatches: 5,
        theoretical_minimum: 5,
    };
    let summary = analysis.summarize();
    assert!(summary.contains("5 dispatches"));
    assert!(summary.contains("0.0% optimization opportunity"));
    assert!(!summary.contains("Top blockers:"));
}

#[test]
fn test_summarize_with_gaps_includes_blockers() {
    let analysis = sample_analysis();
    let summary = analysis.summarize();
    assert!(summary.contains("10 dispatches"));
    assert!(summary.contains("theoretical min: 9"));
    assert!(summary.contains("Top blockers:"));
    assert!(summary.contains("NonFusibleOp"));
    assert!(summary.contains("FanOut"));
}

#[test]
fn test_display_matches_summarize() {
    let analysis = sample_analysis();
    assert_eq!(format!("{analysis}"), analysis.summarize());
}

// ---------------------------------------------------------------------------
// GapAnalysisReport creation and serialization
// ---------------------------------------------------------------------------

#[test]
fn test_report_new_computes_totals() {
    let report = GapAnalysisReport::new(
        "test_model",
        vec![sample_segment("seg_a"), sample_segment("seg_b")],
    );
    assert_eq!(report.model_name, "test_model");
    assert_eq!(report.protocol_version, PROTOCOL_VERSION);
    assert_eq!(report.segments.len(), 2);
    assert_eq!(report.total_dispatches, 20);
    assert_eq!(report.total_theoretical_minimum, 18);
}

#[test]
fn test_report_empty_segments_zero_totals() {
    let report = GapAnalysisReport::new("empty", vec![]);
    assert_eq!(report.total_dispatches, 0);
    assert_eq!(report.total_theoretical_minimum, 0);
    assert!(report.segments.is_empty());
}

#[test]
fn test_report_json_roundtrip() {
    let original = GapAnalysisReport::new(
        "kokoro",
        vec![sample_segment("plbert"), sample_segment("generator")],
    );
    let json = original.to_json();
    let restored = GapAnalysisReport::from_json(&json).expect("round-trip should succeed");

    assert_eq!(restored.model_name, "kokoro");
    assert_eq!(restored.segments.len(), 2);
    assert_eq!(restored.total_dispatches, 20);
    assert_eq!(restored.total_theoretical_minimum, 18);
    assert_eq!(restored.segments[0].segment_name, "plbert");
    assert_eq!(restored.segments[1].segment_name, "generator");
}

#[test]
fn test_report_json_string_roundtrip() {
    let original = GapAnalysisReport::new("test", vec![sample_segment("s1")]);
    let json_str = original.to_json_pretty();
    let restored = GapAnalysisReport::from_json_str(&json_str).expect("string round-trip");
    assert_eq!(original.to_json(), restored.to_json());
}

#[test]
fn test_report_from_json_rejects_invalid() {
    let bad = serde_json::json!({"not": "a report"});
    assert!(GapAnalysisReport::from_json(&bad).is_err());
}

#[test]
fn test_report_from_json_str_rejects_invalid() {
    assert!(GapAnalysisReport::from_json_str("not json").is_err());
}

// ---------------------------------------------------------------------------
// FusionGapAnalysis serde roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_fusion_gap_analysis_serde_roundtrip() {
    let original = sample_analysis();
    let json_str = serde_json::to_string_pretty(&original).expect("serialize");
    let restored: FusionGapAnalysis = serde_json::from_str(&json_str).expect("deserialize");

    assert_eq!(restored.total_dispatches, original.total_dispatches);
    assert_eq!(restored.theoretical_minimum, original.theoretical_minimum);
    assert_eq!(restored.gaps.len(), original.gaps.len());
    for (o, r) in original.gaps.iter().zip(restored.gaps.iter()) {
        assert_eq!(o.step_a, r.step_a);
        assert_eq!(o.step_b, r.step_b);
        assert_eq!(o.kernel_a, r.kernel_a);
        assert_eq!(o.kernel_b, r.kernel_b);
        assert_eq!(o.reason, r.reason);
        assert_eq!(o.savings, r.savings);
    }
}

// ---------------------------------------------------------------------------
// Gap analysis on simple graphs with known patterns
// ---------------------------------------------------------------------------

#[test]
fn test_single_dispatch_no_gaps() {
    let node = TraceNode::new(
        0,
        "relu".into(),
        TraceOp::Relu,
        vec![],
        vec![1, 4],
        DType::F32,
    );
    let graph = ComputationGraph::from_nodes(vec![node]);
    let plan = CompiledPlan {
        steps: vec![make_dispatch("relu")],
        input_shapes: vec![vec![1, 4]],
        output_step: 0,
        weight_names: vec![],
    };
    let analysis = analyze_fusion_gaps(&plan, &graph);
    assert!(analysis.gaps.is_empty());
    assert_eq!(analysis.total_dispatches, 1);
}

#[test]
fn test_non_fusible_ops_detected() {
    let n0 = make_trace_node(0, "matmul", TraceOp::MatMul, vec![], vec![1, 4]);
    let n1 = make_trace_node(
        1,
        "softmax",
        TraceOp::Softmax { dim: 1 },
        vec![0],
        vec![1, 4],
    );
    let graph = ComputationGraph::from_nodes(vec![n0, n1]);

    let plan = CompiledPlan {
        steps: vec![make_dispatch("matmul"), make_dispatch("softmax")],
        input_shapes: vec![],
        output_step: 1,
        weight_names: vec![],
    };
    let analysis = analyze_fusion_gaps(&plan, &graph);
    assert_eq!(analysis.gaps.len(), 1);
    assert_eq!(analysis.gaps[0].reason, FusionBlocker::NonFusibleOp);
}

#[test]
fn test_fan_out_gap_detected() {
    // Node 0: Relu with 2 consumers
    let n0 = make_trace_node(0, "relu", TraceOp::Relu, vec![], vec![1, 4]);
    let n1 = make_trace_node(1, "exp", TraceOp::Exp, vec![0], vec![1, 4]);
    let n2 = make_trace_node(2, "neg", TraceOp::Neg, vec![0], vec![1, 4]);
    let graph = ComputationGraph::from_nodes(vec![n0, n1, n2]);

    let plan = CompiledPlan {
        steps: vec![
            make_dispatch("relu"),
            make_dispatch("exp"),
            make_dispatch("neg"),
        ],
        input_shapes: vec![],
        output_step: 2,
        weight_names: vec![],
    };
    let analysis = analyze_fusion_gaps(&plan, &graph);
    let fan_out_gaps: Vec<_> = analysis
        .gaps
        .iter()
        .filter(|g| g.reason == FusionBlocker::FanOut)
        .collect();
    assert_eq!(fan_out_gaps.len(), 1);
    assert_eq!(fan_out_gaps[0].savings, 1);
}

#[test]
fn test_already_optimal_fused_kernel() {
    let n0 = make_trace_node(0, "exp", TraceOp::Exp, vec![], vec![1, 4]);
    let n1 = make_trace_node(1, "relu", TraceOp::Relu, vec![0], vec![1, 4]);
    let graph = ComputationGraph::from_nodes(vec![n0, n1]);

    let plan = CompiledPlan {
        steps: vec![make_dispatch("fused_exp_relu"), make_dispatch("linear")],
        input_shapes: vec![],
        output_step: 1,
        weight_names: vec![],
    };
    let analysis = analyze_fusion_gaps(&plan, &graph);
    let optimal: Vec<_> = analysis
        .gaps
        .iter()
        .filter(|g| g.reason == FusionBlocker::AlreadyOptimal)
        .collect();
    assert_eq!(optimal.len(), 1);
    assert_eq!(optimal[0].savings, 0);
}

// ---------------------------------------------------------------------------
// OptimizationRequest / OptimizationResponse serde
// ---------------------------------------------------------------------------

#[test]
fn test_optimization_request_roundtrip() {
    let request = OptimizationRequest {
        gap_analysis: sample_analysis(),
        model_name: "kokoro".into(),
        protocol_version: PROTOCOL_VERSION.into(),
        max_suggestions: 5,
    };
    let json = serde_json::to_value(&request).expect("serialize");
    assert_eq!(json["model_name"], "kokoro");
    assert_eq!(json["max_suggestions"], 5);

    let restored: OptimizationRequest = serde_json::from_value(json).expect("deserialize");
    assert_eq!(restored.model_name, "kokoro");
    assert_eq!(restored.max_suggestions, 5);
}

#[test]
fn test_optimization_response_roundtrip() {
    let response = OptimizationResponse {
        protocol_version: PROTOCOL_VERSION.into(),
        suggestions: vec![
            OptimizationSuggestion {
                gap_index: 0,
                optimization_type: "peephole_pass".into(),
                description: "Add relu-exp fusion".into(),
                estimated_savings: 1,
            },
            OptimizationSuggestion {
                gap_index: 2,
                optimization_type: "native_op".into(),
                description: "Wrap matmul+softmax".into(),
                estimated_savings: 3,
            },
        ],
    };
    let json = serde_json::to_string_pretty(&response).expect("serialize");
    let restored: OptimizationResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored.suggestions.len(), 2);
    assert_eq!(restored.suggestions[0].gap_index, 0);
    assert_eq!(restored.suggestions[1].estimated_savings, 3);
}

// ---------------------------------------------------------------------------
// GapAnalysisSegment preserves fields through serde
// ---------------------------------------------------------------------------

#[test]
fn test_segment_roundtrip() {
    let seg = sample_segment("encoder");
    let json = serde_json::to_value(&seg).expect("serialize segment");
    let restored: GapAnalysisSegment = serde_json::from_value(json).expect("deserialize segment");
    assert_eq!(restored.segment_name, "encoder");
    assert_eq!(restored.dispatch_count, 10);
    assert_eq!(restored.theoretical_minimum, 9);
    assert!((restored.cost_estimate.total_ns - 50_000.0).abs() < f64::EPSILON);
    assert_eq!(restored.gap_analysis.gaps.len(), 2);
}
