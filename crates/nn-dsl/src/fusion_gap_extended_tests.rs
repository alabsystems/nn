// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for fusion gap analysis, cost model, and optimization search.
//!
//! Coverage areas:
//! 1. Fusion gap analysis — `analyze_fusion_gaps` identifies fusible sequences
//! 2. FusionChainInfo — chain detection for unary/binary elementwise patterns
//! 3. CostModel — bandwidth/compute estimates via roofline model
//! 4. Cost model calibration — roofline predictions match expected arithmetic intensity
//! 5. optimize_plan_with_cost — exhaustive PeepholeConfig search finds optimal
//! 6. Gap analysis JSON — GapAnalysisReport serialization/deserialization
//! 7. Blocker distribution — categorizing blockers by type
//! 8. Dispatch count extraction — plan analysis utilities
//!
//! Part of #4186.

use std::collections::HashMap;
use std::time::Duration;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use crate::cost_model::{CostEstimate, CostModel};
use crate::gap_analysis_schema::{
    GapAnalysisReport, GapAnalysisSegment, OptimizationRequest, OptimizationResponse,
    OptimizationSuggestion, PROTOCOL_VERSION,
};
use crate::optimize_plan::{
    config_from_bitmask, count_dispatches, optimize_plan, optimize_plan_with_cost,
    PEEPHOLE_FIELD_COUNT,
};
use crate::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use crate::trace_compile::{
    analyze_fusion_gaps, compile_trace_to_plan, compile_trace_to_plan_configured,
    detect_fusion_chains, CompiledKernel, CompiledPlan, CompiledStep, FusionBlocker, FusionGap,
    FusionGapAnalysis, PeepholeConfig,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_dispatch(name: &str, shape: &[usize]) -> CompiledStep {
    let node_id = TensorNodeId::new(0);
    let input_node = TensorNode::new(
        node_id,
        TensorOpKind::Input {
            name: "input_0".into(),
            shape: shape.to_vec(),
        },
        shape.to_vec(),
    );
    let def = TensorKernelDef::new(name, vec![input_node], node_id);
    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: None,
    }
}

fn test_node(id: u64, name: &str, op: TraceOp, inputs: Vec<u64>, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(id, name.to_string(), op, inputs, shape, DType::F32)
}

fn input_node(id: u64, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(
        id,
        format!("input_{id}"),
        TraceOp::Input,
        vec![],
        shape,
        DType::F32,
    )
}

fn sample_gap_analysis(total: usize, min: usize, gaps: Vec<FusionGap>) -> FusionGapAnalysis {
    FusionGapAnalysis {
        gaps,
        total_dispatches: total,
        theoretical_minimum: min,
    }
}

fn sample_cost_estimate() -> CostEstimate {
    CostEstimate {
        total_ns: 50_000.0,
        per_step_ns: vec![(0, 20_000.0), (1, 30_000.0)],
        dispatch_count: 2,
    }
}

fn sample_segment(name: &str, dispatches: usize, minimum: usize) -> GapAnalysisSegment {
    GapAnalysisSegment {
        segment_name: name.to_string(),
        gap_analysis: sample_gap_analysis(dispatches, minimum, vec![]),
        cost_estimate: sample_cost_estimate(),
        dispatch_count: dispatches,
        theoretical_minimum: minimum,
    }
}

// ===========================================================================
// 1. Fusion gap analysis: analyze_fusion_gaps identifies fusible sequences
// ===========================================================================

#[test]
fn test_gap_analysis_identifies_non_fusible_pair() {
    // MatMul -> Softmax: both non-fusible elementwise.
    let n0 = test_node(0, "matmul", TraceOp::MatMul, vec![], vec![4, 64]);
    let n1 = test_node(
        1,
        "softmax",
        TraceOp::Softmax { dim: 1 },
        vec![0],
        vec![4, 64],
    );
    let graph = ComputationGraph::from_nodes(vec![n0, n1]);
    let plan = CompiledPlan {
        steps: vec![
            make_dispatch("matmul", &[4, 64]),
            make_dispatch("softmax", &[4, 64]),
        ],
        input_shapes: vec![vec![4, 64]],
        output_step: 1,
        weight_names: vec![],
    };

    let analysis = analyze_fusion_gaps(&plan, &graph);
    assert_eq!(analysis.total_dispatches, 2);
    assert!(
        !analysis.gaps.is_empty(),
        "should identify at least one gap"
    );
    assert_eq!(analysis.gaps[0].reason, FusionBlocker::NonFusibleOp);
}

#[test]
fn test_gap_analysis_shape_mismatch_detection() {
    // Two fusible elementwise ops with different output shapes.
    let n0 = test_node(0, "relu_a", TraceOp::Relu, vec![], vec![1, 256]);
    let n1 = test_node(1, "exp_b", TraceOp::Exp, vec![0], vec![1, 512]);
    let graph = ComputationGraph::from_nodes(vec![n0, n1]);
    let plan = CompiledPlan {
        steps: vec![
            make_dispatch("relu", &[1, 256]),
            make_dispatch("exp", &[1, 512]),
        ],
        input_shapes: vec![vec![1, 256]],
        output_step: 1,
        weight_names: vec![],
    };

    let analysis = analyze_fusion_gaps(&plan, &graph);
    let shape_gaps: Vec<_> = analysis
        .gaps
        .iter()
        .filter(|g| g.reason == FusionBlocker::ShapeMismatch)
        .collect();
    assert_eq!(shape_gaps.len(), 1, "should detect shape mismatch");
    assert_eq!(shape_gaps[0].savings, 1);
}

#[test]
fn test_gap_analysis_fan_out_with_three_consumers() {
    // Node 0 feeds 3 consumers -- fan-out should be detected.
    let n0 = test_node(0, "relu", TraceOp::Relu, vec![], vec![1, 128]);
    let n1 = test_node(1, "exp", TraceOp::Exp, vec![0], vec![1, 128]);
    let n2 = test_node(2, "neg", TraceOp::Neg, vec![0], vec![1, 128]);
    let n3 = test_node(3, "log", TraceOp::Log, vec![0], vec![1, 128]);
    let graph = ComputationGraph::from_nodes(vec![n0, n1, n2, n3]);
    let plan = CompiledPlan {
        steps: vec![
            make_dispatch("relu", &[1, 128]),
            make_dispatch("exp", &[1, 128]),
            make_dispatch("neg", &[1, 128]),
            make_dispatch("log", &[1, 128]),
        ],
        input_shapes: vec![vec![1, 128]],
        output_step: 3,
        weight_names: vec![],
    };

    let analysis = analyze_fusion_gaps(&plan, &graph);
    let fan_out_gaps: Vec<_> = analysis
        .gaps
        .iter()
        .filter(|g| g.reason == FusionBlocker::FanOut)
        .collect();
    assert!(
        !fan_out_gaps.is_empty(),
        "should detect fan-out when node has 3 consumers"
    );
}

#[test]
fn test_gap_analysis_not_dispatch_mixed() {
    // A Dispatch step adjacent to a Passthrough step.
    let n0 = test_node(0, "relu", TraceOp::Relu, vec![], vec![1, 64]);
    let graph = ComputationGraph::from_nodes(vec![n0]);
    let plan = CompiledPlan {
        steps: vec![
            make_dispatch("relu", &[1, 64]),
            CompiledStep::Passthrough {
                op_name: "reshape".to_string(),
                output_shape: vec![1, 64],
            },
        ],
        input_shapes: vec![vec![1, 64]],
        output_step: 1,
        weight_names: vec![],
    };

    let analysis = analyze_fusion_gaps(&plan, &graph);
    let not_dispatch_gaps: Vec<_> = analysis
        .gaps
        .iter()
        .filter(|g| g.reason == FusionBlocker::NotDispatch)
        .collect();
    assert_eq!(
        not_dispatch_gaps.len(),
        1,
        "should identify NotDispatch gap"
    );
    assert_eq!(not_dispatch_gaps[0].savings, 0);
}

#[test]
fn test_gap_analysis_already_optimal_fused_prefix() {
    // Kernels named "fused_*" should be marked AlreadyOptimal.
    let n0 = test_node(0, "relu", TraceOp::Relu, vec![], vec![1, 64]);
    let n1 = test_node(1, "gelu", TraceOp::Gelu, vec![0], vec![1, 64]);
    let graph = ComputationGraph::from_nodes(vec![n0, n1]);
    let plan = CompiledPlan {
        steps: vec![
            make_dispatch("fused_relu_gelu", &[1, 64]),
            make_dispatch("linear", &[1, 64]),
        ],
        input_shapes: vec![vec![1, 64]],
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
}

// ===========================================================================
// 2. FusionChainInfo: chain detection for elementwise patterns
// ===========================================================================

#[test]
fn test_chain_detection_relu_exp_log() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 256]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![1, 256]),
        test_node(2, "exp", TraceOp::Exp, vec![1], vec![1, 256]),
        test_node(3, "log", TraceOp::Log, vec![2], vec![1, 256]),
    ]);
    let chains = detect_fusion_chains(&graph).unwrap();
    assert!(
        !chains.is_empty(),
        "relu -> exp -> log should be detected as a chain"
    );
    assert_eq!(chains[0].chain_len, 3);
    assert_eq!(chains[0].pairs.len(), 2, "3-op chain should have 2 pairs");
}

#[test]
fn test_chain_detection_gelu_sigmoid() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 512]),
        test_node(1, "gelu", TraceOp::Gelu, vec![0], vec![1, 512]),
        test_node(2, "sigmoid", TraceOp::Sigmoid, vec![1], vec![1, 512]),
    ]);
    let chains = detect_fusion_chains(&graph).unwrap();
    assert!(!chains.is_empty(), "gelu -> sigmoid should be fusible");
    assert_eq!(chains[0].chain_len, 2);
}

#[test]
fn test_chain_detection_not_across_matmul() {
    // Relu -> MatMul -> Gelu: MatMul breaks the chain.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![4, 768]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![4, 768]),
        input_node(2, vec![768, 3072]),
        test_node(3, "matmul", TraceOp::MatMul, vec![1, 2], vec![4, 3072]),
        test_node(4, "gelu", TraceOp::Gelu, vec![3], vec![4, 3072]),
    ]);
    let chains = detect_fusion_chains(&graph).unwrap();
    // No chain should span across the matmul.
    for chain in &chains {
        assert!(
            !chain.chain_name.contains("matmul"),
            "MatMul should not be part of any fusion chain"
        );
    }
}

#[test]
fn test_chain_detection_fan_out_prevents_chain() {
    // exp -> (log AND neg): fan-out breaks the chain.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 128]),
        test_node(1, "exp", TraceOp::Exp, vec![0], vec![1, 128]),
        test_node(2, "log", TraceOp::Log, vec![1], vec![1, 128]),
        test_node(3, "neg", TraceOp::Neg, vec![1], vec![1, 128]),
    ]);
    let chains = detect_fusion_chains(&graph).unwrap();
    assert!(chains.is_empty(), "fan-out should prevent chain detection");
}

#[test]
fn test_chain_detection_empty_graph() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let chains = detect_fusion_chains(&graph).unwrap();
    assert!(chains.is_empty());
}

// ===========================================================================
// 3. CostModel: bandwidth/compute estimates
// ===========================================================================

#[test]
fn test_cost_model_m4_bandwidth_and_compute() {
    let model = CostModel::apple_m4();
    assert!(
        model.bandwidth_bytes_per_sec > 100e9,
        "M4 bandwidth should exceed 100 GB/s"
    );
    assert!(
        model.bandwidth_bytes_per_sec <= 500e9,
        "M4 bandwidth should be at most 500 GB/s"
    );
    assert_eq!(model.simd_width, 32);
    assert!(model.launch_overhead_ns > 0.0);
}

#[test]
fn test_cost_model_estimate_scales_with_elements() {
    let model = CostModel::apple_m4();
    let plan_small = CompiledPlan {
        steps: vec![make_dispatch("relu", &[1, 32])],
        input_shapes: vec![vec![1, 32]],
        output_step: 0,
        weight_names: vec![],
    };
    let plan_large = CompiledPlan {
        steps: vec![make_dispatch("relu", &[64, 4096])],
        input_shapes: vec![vec![64, 4096]],
        output_step: 0,
        weight_names: vec![],
    };
    let est_small = model.estimate(&plan_small);
    let est_large = model.estimate(&plan_large);
    assert!(
        est_large.total_ns > est_small.total_ns,
        "larger tensor ({:.0} ns) should cost more than smaller ({:.0} ns)",
        est_large.total_ns,
        est_small.total_ns,
    );
}

#[test]
fn test_cost_model_estimate_multi_step_sums_costs() {
    let model = CostModel::apple_m4();
    let plan_one = CompiledPlan {
        steps: vec![make_dispatch("gelu", &[4, 768])],
        input_shapes: vec![vec![4, 768]],
        output_step: 0,
        weight_names: vec![],
    };
    let plan_two = CompiledPlan {
        steps: vec![
            make_dispatch("gelu", &[4, 768]),
            make_dispatch("gelu", &[4, 768]),
        ],
        input_shapes: vec![vec![4, 768]],
        output_step: 1,
        weight_names: vec![],
    };
    let est_one = model.estimate(&plan_one);
    let est_two = model.estimate(&plan_two);
    // Two identical dispatches should cost approximately 2x.
    assert!(
        est_two.total_ns > est_one.total_ns,
        "two dispatches should cost more than one"
    );
    assert_eq!(est_two.dispatch_count, 2);
    assert_eq!(est_one.dispatch_count, 1);
}

#[test]
fn test_cost_model_passthrough_steps_are_free() {
    let model = CostModel::apple_m4();
    let plan_dispatch_only = CompiledPlan {
        steps: vec![make_dispatch("relu", &[4, 256])],
        input_shapes: vec![vec![4, 256]],
        output_step: 0,
        weight_names: vec![],
    };
    let plan_with_passthrough = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            make_dispatch("relu", &[4, 256]),
            CompiledStep::IdentityPassthrough,
        ],
        input_shapes: vec![vec![4, 256]],
        output_step: 1,
        weight_names: vec![],
    };
    let est_dispatch = model.estimate(&plan_dispatch_only);
    let est_with_pass = model.estimate(&plan_with_passthrough);
    // Total cost should be the same (passthroughs add zero cost).
    assert!(
        (est_dispatch.total_ns - est_with_pass.total_ns).abs() < f64::EPSILON,
        "passthrough steps should not add cost"
    );
}

#[test]
fn test_cost_model_empty_plan_zero_cost() {
    let model = CostModel::apple_m4();
    let plan = CompiledPlan {
        steps: vec![],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    let est = model.estimate(&plan);
    assert_eq!(est.total_ns, 0.0);
    assert_eq!(est.dispatch_count, 0);
    assert!(est.per_step_ns.is_empty());
}

#[test]
fn test_cost_model_m4_max_higher_throughput_entries() {
    let model = CostModel::apple_m4_max();
    assert!(!model.op_throughput.is_empty());
    // M4 Max should have higher throughput for matmul than default (1 TFLOP/s).
    let matmul_throughput = model.op_throughput.get("matmul").copied().unwrap_or(0.0);
    assert!(
        matmul_throughput > 1e12,
        "M4 Max matmul throughput ({:.0} TFLOP/s) should exceed 1 TFLOP/s",
        matmul_throughput / 1e12,
    );
}

// ===========================================================================
// 4. Cost model calibration: roofline predictions vs arithmetic intensity
// ===========================================================================

#[test]
fn test_calibration_plan_only_dispatches() {
    let model = CostModel::apple_m4();
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            make_dispatch("relu", &[4, 256]),
            CompiledStep::IdentityPassthrough,
            make_dispatch("gelu", &[4, 256]),
        ],
        input_shapes: vec![vec![4, 256]],
        output_step: 3,
        weight_names: vec![],
    };
    let records = model.calibration_plan(&plan);
    assert_eq!(
        records.len(),
        2,
        "only Dispatch steps get calibration records"
    );
    assert_eq!(records[0].step_index, 1);
    assert_eq!(records[0].op_name, "relu");
    assert_eq!(records[1].step_index, 3);
    assert_eq!(records[1].op_name, "gelu");
    for record in &records {
        assert!(record.estimated_ns > 0.0, "estimated time must be positive");
        assert!(record.actual_ns.is_none(), "actual_ns starts unset");
    }
}

#[test]
fn test_calibrate_matched_entries() {
    let predictions = vec![
        ("relu".to_string(), 100.0),
        ("gelu".to_string(), 200.0),
        ("matmul".to_string(), 500.0),
    ];
    let actuals = vec![
        ("relu".to_string(), 110.0),
        ("gelu".to_string(), 180.0),
        ("matmul".to_string(), 550.0),
    ];
    let report = CostModel::calibrate(&predictions, &actuals).expect("calibration should succeed");
    assert!(report.mean_absolute_error_ns > 0.0);
    assert_eq!(report.entries.len(), 3);
    // Mean error ratio should be near 1.0 (predictions are close to actuals).
    assert!(
        report.mean_error_ratio > 0.5 && report.mean_error_ratio < 2.0,
        "mean_error_ratio {:.3} should be near 1.0",
        report.mean_error_ratio,
    );
}

#[test]
fn test_calibrate_no_matching_steps_errors() {
    let predictions = vec![("relu".to_string(), 100.0)];
    let actuals = vec![("matmul".to_string(), 200.0)];
    let result = CostModel::calibrate(&predictions, &actuals);
    assert!(result.is_err(), "no matching steps should return error");
}

#[test]
fn test_calibration_report_from_records_no_actuals() {
    use crate::cost_model::{CalibrationRecord, CalibrationReport};
    let records = vec![CalibrationRecord {
        step_index: 0,
        estimated_ns: 100.0,
        actual_ns: None,
        op_name: "relu".to_string(),
        is_memory_bound: true,
    }];
    let report = CalibrationReport::from_records(&records);
    assert_eq!(report.mean_absolute_error_ns, 0.0);
    assert!(report.correlation.is_nan());
}

#[test]
fn test_cost_model_roofline_memory_bound_small_tensor() {
    // A small tensor doing a simple elementwise op should be memory-bound
    // (bandwidth time dominates compute time for small element counts).
    let model = CostModel::apple_m4_max();
    let plan = CompiledPlan {
        steps: vec![make_dispatch("relu", &[1, 32])],
        input_shapes: vec![vec![1, 32]],
        output_step: 0,
        weight_names: vec![],
    };
    let records = model.calibration_plan(&plan);
    assert_eq!(records.len(), 1);
    // 32 elements at 4 bytes * 2 (read+write) = 256 bytes.
    // With 400 GB/s bandwidth, that's ~0.64 ns memory time.
    // With 12 TFLOP/s throughput, compute is ~0.003 ns.
    // Memory time > compute time => memory bound.
    assert!(
        records[0].is_memory_bound,
        "small tensor relu should be memory-bound"
    );
}

// ===========================================================================
// 5. optimize_plan_with_cost: exhaustive PeepholeConfig search
// ===========================================================================

#[test]
fn test_optimize_plan_with_cost_empty_graph() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let model = CostModel::apple_m4();
    let result = optimize_plan_with_cost(&graph, &model, Duration::from_secs(5))
        .expect("optimize on empty graph should succeed");
    assert_eq!(result.dispatch_count, 0);
    assert_eq!(result.baseline_dispatch_count, 0);
    assert!(result.configs_explored >= 1);
}

#[test]
fn test_optimize_plan_with_cost_zero_budget_returns_baseline() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let model = CostModel::apple_m4();
    let result = optimize_plan_with_cost(&graph, &model, Duration::ZERO)
        .expect("zero-budget optimize should succeed");
    assert_eq!(result.configs_explored, 1, "only baseline explored");
    assert_eq!(result.dispatch_count, 0);
}

#[test]
fn test_optimize_plan_result_summarize() {
    use crate::optimize_plan::OptimizationResult;
    let result = OptimizationResult {
        plan: CompiledPlan {
            steps: vec![],
            input_shapes: vec![],
            output_step: 0,
            weight_names: vec![],
        },
        config: PeepholeConfig::default(),
        dispatch_count: 150,
        configs_explored: 32768,
        baseline_dispatch_count: 200,
        best_cost_ns: 8000.0,
        baseline_cost_ns: 10000.0,
    };
    let summary = result.summarize();
    assert!(summary.contains("200"), "should mention baseline count");
    assert!(summary.contains("150"), "should mention best count");
    assert!(summary.contains("25.0%"), "should show 25% reduction");
    assert!(summary.contains("32768"), "should mention configs explored");
    assert!(
        summary.contains("Baseline cost"),
        "should include cost info"
    );
}

#[test]
fn test_optimize_plan_best_le_baseline() {
    // Even on an empty graph, best should be <= baseline.
    let graph = ComputationGraph::from_nodes(vec![]);
    let result =
        optimize_plan(&graph, Duration::from_millis(100)).expect("optimize should succeed");
    assert!(
        result.dispatch_count <= result.baseline_dispatch_count,
        "best ({}) should be <= baseline ({})",
        result.dispatch_count,
        result.baseline_dispatch_count,
    );
}

// ===========================================================================
// 6. Gap analysis JSON: GapAnalysisReport serialization/deserialization
// ===========================================================================

#[test]
fn test_gap_analysis_report_roundtrip_json() {
    let report = GapAnalysisReport::new(
        "test_model",
        vec![
            sample_segment("encoder", 15, 12),
            sample_segment("decoder", 25, 20),
        ],
    );
    assert_eq!(report.total_dispatches, 40);
    assert_eq!(report.total_theoretical_minimum, 32);

    let json_str = report.to_json_pretty();
    let restored =
        GapAnalysisReport::from_json_str(&json_str).expect("JSON string roundtrip should succeed");
    assert_eq!(restored.model_name, "test_model");
    assert_eq!(restored.segments.len(), 2);
    assert_eq!(restored.total_dispatches, 40);
    assert_eq!(restored.total_theoretical_minimum, 32);
    assert_eq!(restored.segments[0].segment_name, "encoder");
    assert_eq!(restored.segments[1].segment_name, "decoder");
}

#[test]
fn test_gap_analysis_report_roundtrip_json_value() {
    let report = GapAnalysisReport::new("kokoro", vec![sample_segment("plbert", 10, 8)]);
    let json_val = report.to_json();
    let restored =
        GapAnalysisReport::from_json(&json_val).expect("JSON value roundtrip should succeed");
    assert_eq!(restored.model_name, "kokoro");
    assert_eq!(restored.segments.len(), 1);
    assert_eq!(restored.protocol_version, PROTOCOL_VERSION);
}

#[test]
fn test_gap_analysis_report_empty_segments_roundtrip() {
    let report = GapAnalysisReport::new("empty", vec![]);
    assert_eq!(report.total_dispatches, 0);
    let json_str = report.to_json_pretty();
    let restored = GapAnalysisReport::from_json_str(&json_str).unwrap();
    assert!(restored.segments.is_empty());
    assert_eq!(restored.total_dispatches, 0);
}

#[test]
fn test_gap_analysis_report_rejects_invalid_json() {
    let result = GapAnalysisReport::from_json_str("{\"invalid\": true}");
    assert!(result.is_err());
}

#[test]
fn test_fusion_gap_analysis_to_json_roundtrip() {
    let analysis = FusionGapAnalysis {
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
                step_a: 5,
                step_b: 6,
                kernel_a: "conv".into(),
                kernel_b: "bn".into(),
                reason: FusionBlocker::ShapeMismatch,
                savings: 1,
            },
        ],
        total_dispatches: 20,
        theoretical_minimum: 18,
    };

    let json_val = analysis.to_json();
    let restored = FusionGapAnalysis::from_json(&json_val).unwrap();
    assert_eq!(restored.total_dispatches, 20);
    assert_eq!(restored.theoretical_minimum, 18);
    assert_eq!(restored.gaps.len(), 2);
    assert_eq!(restored.gaps[0].reason, FusionBlocker::FanOut);
    assert_eq!(restored.gaps[1].reason, FusionBlocker::ShapeMismatch);
}

#[test]
fn test_optimization_request_response_roundtrip() {
    let request = OptimizationRequest {
        gap_analysis: FusionGapAnalysis {
            gaps: vec![],
            total_dispatches: 5,
            theoretical_minimum: 5,
        },
        model_name: "whisper".into(),
        protocol_version: PROTOCOL_VERSION.into(),
        max_suggestions: 3,
    };
    let json = serde_json::to_value(&request).expect("request serialization");
    assert_eq!(json["model_name"], "whisper");
    assert_eq!(json["max_suggestions"], 3);

    let response = OptimizationResponse {
        protocol_version: PROTOCOL_VERSION.into(),
        suggestions: vec![OptimizationSuggestion {
            gap_index: 0,
            optimization_type: "peephole_pass".into(),
            description: "Add relu-exp fusion".into(),
            estimated_savings: 2,
        }],
    };
    let resp_json = serde_json::to_value(&response).expect("response serialization");
    let restored: OptimizationResponse =
        serde_json::from_value(resp_json).expect("response deserialization");
    assert_eq!(restored.suggestions.len(), 1);
    assert_eq!(restored.suggestions[0].estimated_savings, 2);
}

// ===========================================================================
// 7. Blocker distribution: categorizing blockers by type
// ===========================================================================

#[test]
fn test_blocker_distribution_all_types() {
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
                step_a: 1,
                step_b: 2,
                kernel_a: "b".into(),
                kernel_b: "c".into(),
                reason: FusionBlocker::ShapeMismatch,
                savings: 1,
            },
            FusionGap {
                step_a: 2,
                step_b: 3,
                kernel_a: "c".into(),
                kernel_b: "d".into(),
                reason: FusionBlocker::NonFusibleOp,
                savings: 0,
            },
            FusionGap {
                step_a: 3,
                step_b: 4,
                kernel_a: "d".into(),
                kernel_b: "e".into(),
                reason: FusionBlocker::NotDispatch,
                savings: 0,
            },
            FusionGap {
                step_a: 4,
                step_b: 5,
                kernel_a: "e".into(),
                kernel_b: "f".into(),
                reason: FusionBlocker::AlreadyOptimal,
                savings: 0,
            },
            FusionGap {
                step_a: 5,
                step_b: 6,
                kernel_a: "f".into(),
                kernel_b: "g".into(),
                reason: FusionBlocker::NoPeepholePattern,
                savings: 1,
            },
            FusionGap {
                step_a: 6,
                step_b: 7,
                kernel_a: "g".into(),
                kernel_b: "h".into(),
                reason: FusionBlocker::NoDependency,
                savings: 0,
            },
        ],
        total_dispatches: 30,
        theoretical_minimum: 27,
    };
    let counts = analysis.blocker_counts();
    assert_eq!(
        counts.len(),
        7,
        "should have exactly 7 distinct blocker types"
    );
    assert_eq!(counts.get("FanOut"), Some(&1));
    assert_eq!(counts.get("ShapeMismatch"), Some(&1));
    assert_eq!(counts.get("NonFusibleOp"), Some(&1));
    assert_eq!(counts.get("NotDispatch"), Some(&1));
    assert_eq!(counts.get("AlreadyOptimal"), Some(&1));
    assert_eq!(counts.get("NoPeepholePattern"), Some(&1));
    assert_eq!(counts.get("NoDependency"), Some(&1));
}

#[test]
fn test_blocker_distribution_duplicates() {
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
            FusionGap {
                step_a: 4,
                step_b: 5,
                kernel_a: "e".into(),
                kernel_b: "f".into(),
                reason: FusionBlocker::FanOut,
                savings: 1,
            },
            FusionGap {
                step_a: 6,
                step_b: 7,
                kernel_a: "g".into(),
                kernel_b: "h".into(),
                reason: FusionBlocker::NonFusibleOp,
                savings: 0,
            },
        ],
        total_dispatches: 20,
        theoretical_minimum: 17,
    };
    let counts = analysis.blocker_counts();
    assert_eq!(counts.get("FanOut"), Some(&3));
    assert_eq!(counts.get("NonFusibleOp"), Some(&1));
    assert_eq!(counts.len(), 2);
}

#[test]
fn test_blocker_distribution_empty() {
    let analysis = FusionGapAnalysis {
        gaps: vec![],
        total_dispatches: 5,
        theoretical_minimum: 5,
    };
    let counts = analysis.blocker_counts();
    assert!(counts.is_empty());
}

#[test]
fn test_optimization_opportunity_pct_various() {
    // 50% opportunity.
    let a = FusionGapAnalysis {
        gaps: vec![],
        total_dispatches: 100,
        theoretical_minimum: 50,
    };
    assert!((a.optimization_opportunity_pct() - 50.0).abs() < 0.01);

    // 0% opportunity.
    let b = FusionGapAnalysis {
        gaps: vec![],
        total_dispatches: 100,
        theoretical_minimum: 100,
    };
    assert!((b.optimization_opportunity_pct() - 0.0).abs() < f64::EPSILON);

    // 100% opportunity.
    let c = FusionGapAnalysis {
        gaps: vec![],
        total_dispatches: 100,
        theoretical_minimum: 0,
    };
    assert!((c.optimization_opportunity_pct() - 100.0).abs() < 0.01);
}

#[test]
fn test_summarize_includes_top_blockers_sorted() {
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
                reason: FusionBlocker::NonFusibleOp,
                savings: 0,
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
        theoretical_minimum: 9,
    };
    let summary = analysis.summarize();
    assert!(summary.contains("Top blockers:"));
    // NonFusibleOp (2) should appear before FanOut (1) in the summary.
    let nfo_pos = summary.find("NonFusibleOp").unwrap();
    let fo_pos = summary.find("FanOut").unwrap();
    assert!(
        nfo_pos < fo_pos,
        "NonFusibleOp (2) should appear before FanOut (1) in sorted output"
    );
}

// ===========================================================================
// 8. Dispatch count extraction: plan analysis utilities
// ===========================================================================

#[test]
fn test_count_dispatches_only_dispatch_steps() {
    let plan = CompiledPlan {
        steps: vec![
            make_dispatch("relu", &[4, 256]),
            make_dispatch("gelu", &[4, 256]),
            make_dispatch("softmax", &[4, 256]),
        ],
        input_shapes: vec![vec![4, 256]],
        output_step: 2,
        weight_names: vec![],
    };
    assert_eq!(count_dispatches(&plan), 3);
}

#[test]
fn test_count_dispatches_with_passthrough_and_constant() {
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            make_dispatch("relu", &[4, 256]),
            CompiledStep::Passthrough {
                op_name: "reshape".to_string(),
                output_shape: vec![4, 256],
            },
            CompiledStep::IdentityPassthrough,
            CompiledStep::ConstantValue {
                value: 0.0,
                shape: vec![1],
            },
            make_dispatch("gelu", &[4, 256]),
        ],
        input_shapes: vec![vec![4, 256]],
        output_step: 5,
        weight_names: vec![],
    };
    assert_eq!(
        count_dispatches(&plan),
        2,
        "only Dispatch steps should be counted"
    );
}

#[test]
fn test_count_dispatches_empty_plan() {
    let plan = CompiledPlan {
        steps: vec![],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    assert_eq!(count_dispatches(&plan), 0);
}

#[test]
fn test_count_dispatches_no_dispatch_steps() {
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            CompiledStep::IdentityPassthrough,
        ],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    assert_eq!(count_dispatches(&plan), 0);
}

#[test]
fn test_compile_empty_graph_zero_dispatches() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let plan = compile_trace_to_plan(&graph).unwrap();
    assert_eq!(count_dispatches(&plan), 0);
}

#[test]
fn test_compile_all_disabled_config_zero_dispatches() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let config = config_from_bitmask(0);
    let plan = compile_trace_to_plan_configured(&graph, &config).unwrap();
    assert_eq!(count_dispatches(&plan), 0);
}

#[test]
fn test_compile_default_config_zero_dispatches_on_empty() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let all_on = (1u32 << PEEPHOLE_FIELD_COUNT) - 1;
    let config = config_from_bitmask(all_on);
    let plan = compile_trace_to_plan_configured(&graph, &config).unwrap();
    assert_eq!(count_dispatches(&plan), 0);
}

#[test]
fn test_cost_estimate_per_step_indices_match_dispatch_positions() {
    let model = CostModel::apple_m4();
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,        // index 0
            make_dispatch("relu", &[4, 256]),  // index 1
            CompiledStep::IdentityPassthrough, // index 2
            make_dispatch("gelu", &[4, 256]),  // index 3
        ],
        input_shapes: vec![vec![4, 256]],
        output_step: 3,
        weight_names: vec![],
    };
    let est = model.estimate(&plan);
    assert_eq!(est.dispatch_count, 2);
    assert_eq!(est.per_step_ns.len(), 2);
    // per_step_ns indices should be the plan step indices of Dispatch steps.
    assert_eq!(est.per_step_ns[0].0, 1, "first dispatch is at step index 1");
    assert_eq!(
        est.per_step_ns[1].0, 3,
        "second dispatch is at step index 3"
    );
    // Each should have positive cost.
    assert!(est.per_step_ns[0].1 > 0.0);
    assert!(est.per_step_ns[1].1 > 0.0);
}
