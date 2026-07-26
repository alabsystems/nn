// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end integration tests for the self-optimizing compiler pipeline.
//!
//! Exercises all phases together on synthetic computation graphs:
//! compile -> gap analysis -> cost model -> optimization search -> JSON protocol.
//!
//! Part of #3835 (Self-Optimizing ML Compiler, Phase 7).

use std::time::Duration;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;
use nn_dsl::{
    analyze_fusion_gaps, compile_trace_to_plan_with_fusion, fusion_gap_analysis_schema,
    optimize_plan, optimize_plan_with_cost, optimize_segments, theoretical_minimum_dispatches,
    CostModel, FusionGapAnalysis, OptimizationRequest, OptimizationResponse,
    OptimizationSuggestion, SegmentOptimizationResult, PROTOCOL_VERSION,
};

/// Build a 5-node synthetic graph: Input -> Relu -> Exp -> Neg -> Add(Neg, Input).
///
/// Node 0: Input [1, 64]
/// Node 1: Relu(0) [1, 64]
/// Node 2: Exp(1)  [1, 64]
/// Node 3: Neg(2)  [1, 64]
/// Node 4: Add(3, 0) [1, 64]  -- binary op consuming Neg output and Input
fn build_five_node_graph() -> ComputationGraph {
    let shape = vec![1, 64];
    let nodes = vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            1,
            "relu_0".into(),
            TraceOp::Relu,
            vec![0],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            2,
            "exp_0".into(),
            TraceOp::Exp,
            vec![1],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            3,
            "neg_0".into(),
            TraceOp::Neg,
            vec![2],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            4,
            "add_0".into(),
            TraceOp::Add,
            vec![3, 0],
            shape,
            DType::F32,
        ),
    ];
    ComputationGraph::from_nodes(nodes)
}

/// Build a simple 3-node linear chain: Input -> Relu -> Neg.
fn build_three_node_graph() -> ComputationGraph {
    let shape = vec![1, 32];
    let nodes = vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            1,
            "relu_0".into(),
            TraceOp::Relu,
            vec![0],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            2,
            "neg_0".into(),
            TraceOp::Neg,
            vec![1],
            shape,
            DType::F32,
        ),
    ];
    ComputationGraph::from_nodes(nodes)
}

/// Build a larger graph for cost comparison: Input -> 8x Relu chain.
fn build_large_chain_graph(chain_len: usize) -> ComputationGraph {
    let shape = vec![1, 1024];
    let mut nodes = vec![TraceNode::new(
        0,
        "input_0".into(),
        TraceOp::Input,
        vec![],
        shape.clone(),
        DType::F32,
    )];
    for i in 1..=chain_len {
        nodes.push(TraceNode::new(
            i as u64,
            format!("relu_{}", i - 1),
            TraceOp::Relu,
            vec![(i - 1) as u64],
            shape.clone(),
            DType::F32,
        ));
    }
    ComputationGraph::from_nodes(nodes)
}

/// Build a graph that produces multiple gap types: mixed fusible and non-fusible ops.
///
/// Node 0: Input [1, 64]
/// Node 1: Relu(0) [1, 64]              -- fusible elementwise
/// Node 2: Softmax(1, dim=1) [1, 64]    -- non-fusible (reduction-based)
/// Node 3: Exp(2) [1, 64]               -- fusible elementwise
/// Node 4: Neg(3) [1, 64]               -- fusible elementwise
fn build_mixed_ops_graph() -> ComputationGraph {
    let shape = vec![1, 64];
    let nodes = vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            1,
            "relu_0".into(),
            TraceOp::Relu,
            vec![0],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            2,
            "softmax_0".into(),
            TraceOp::Softmax { dim: 1 },
            vec![1],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            3,
            "exp_0".into(),
            TraceOp::Exp,
            vec![2],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            4,
            "neg_0".into(),
            TraceOp::Neg,
            vec![3],
            shape,
            DType::F32,
        ),
    ];
    ComputationGraph::from_nodes(nodes)
}

// =============================================================================
// Test 1: Full Pipeline on Synthetic Graph
// =============================================================================

#[test]
fn test_full_pipeline_on_synthetic_graph() {
    let graph = build_five_node_graph();

    // Step 1: Compile with fusion.
    let plan = compile_trace_to_plan_with_fusion(&graph)
        .expect("compile_trace_to_plan_with_fusion should succeed on synthetic graph");

    assert!(
        !plan.steps.is_empty(),
        "compiled plan should have at least one step"
    );

    // Step 2: Analyze fusion gaps.
    let analysis = analyze_fusion_gaps(&plan, &graph);
    assert!(
        analysis.total_dispatches > 0,
        "5-node graph should produce at least 1 dispatch"
    );
    assert!(
        analysis.theoretical_minimum <= analysis.total_dispatches,
        "theoretical minimum ({}) should be <= total dispatches ({})",
        analysis.theoretical_minimum,
        analysis.total_dispatches,
    );

    // The theoretical_minimum convenience function should match.
    assert_eq!(
        theoretical_minimum_dispatches(&analysis),
        analysis.theoretical_minimum
    );

    // Step 3: Cost model estimate.
    let cost = CostModel::apple_m4().estimate(&plan);
    assert!(
        cost.total_ns > 0.0,
        "cost estimate should be > 0 for a non-empty plan"
    );
    assert!(
        cost.dispatch_count > 0,
        "dispatch_count should be > 0 for a plan with compute ops"
    );

    // Step 4: PlanSummary with fusion gap.
    let summary = plan.summary().with_fusion_gap(&analysis);
    assert!(
        summary.fusion_gap_summary.is_some(),
        "with_fusion_gap should populate fusion_gap_summary"
    );
    let display_output = format!("{summary}");
    assert!(
        display_output.contains("Fusion Gap Analysis"),
        "Display output should contain 'Fusion Gap Analysis', got: {display_output}"
    );
}

// =============================================================================
// Test 2: JSON Round-Trip
// =============================================================================

#[test]
fn test_json_round_trip() {
    let graph = build_five_node_graph();
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compilation should succeed");
    let analysis = analyze_fusion_gaps(&plan, &graph);

    // to_json -> from_json -> compare via re-serialization.
    let json_val = analysis.to_json();
    let restored =
        FusionGapAnalysis::from_json(&json_val).expect("from_json should succeed on valid JSON");
    assert_eq!(
        analysis.to_json(),
        restored.to_json(),
        "round-trip should produce identical JSON"
    );

    // to_json_pretty -> parse as serde_json::Value -> verify fields exist.
    let pretty = analysis.to_json_pretty();
    let parsed: serde_json::Value =
        serde_json::from_str(&pretty).expect("pretty JSON should be valid");
    assert!(parsed.is_object(), "top-level should be an object");
    assert!(parsed.get("gaps").is_some(), "should have 'gaps' field");
    assert!(
        parsed.get("total_dispatches").is_some(),
        "should have 'total_dispatches' field"
    );
    assert!(
        parsed.get("theoretical_minimum").is_some(),
        "should have 'theoretical_minimum' field"
    );

    // Verify total_dispatches matches.
    assert_eq!(
        parsed["total_dispatches"].as_u64().unwrap(),
        analysis.total_dispatches as u64,
    );
}

// =============================================================================
// Test 3: Cost Model Ranking
// =============================================================================

#[test]
fn test_cost_model_ranking() {
    let small_graph = build_three_node_graph();
    let large_graph = build_large_chain_graph(8);

    let small_plan = compile_trace_to_plan_with_fusion(&small_graph)
        .expect("small graph compilation should succeed");
    let large_plan = compile_trace_to_plan_with_fusion(&large_graph)
        .expect("large graph compilation should succeed");

    let model = CostModel::apple_m4();
    let small_cost = model.estimate(&small_plan);
    let large_cost = model.estimate(&large_plan);

    // The larger graph (more dispatch steps, larger tensors) should cost more.
    assert!(
        large_cost.total_ns >= small_cost.total_ns,
        "larger plan ({:.1} ns, {} dispatches) should cost >= smaller plan ({:.1} ns, {} dispatches)",
        large_cost.total_ns,
        large_cost.dispatch_count,
        small_cost.total_ns,
        small_cost.dispatch_count,
    );

    // Both should have positive cost.
    assert!(small_cost.total_ns > 0.0, "small plan cost should be > 0");
    assert!(large_cost.total_ns > 0.0, "large plan cost should be > 0");
}

// =============================================================================
// Test 4: Optimization Search on Simple Graph
// =============================================================================

#[test]
fn test_optimization_search_on_simple_graph() {
    let graph = build_three_node_graph();

    let result = optimize_plan(&graph, Duration::from_secs(10))
        .expect("optimize_plan should succeed on simple graph");

    // No regression: best dispatch count <= baseline.
    assert!(
        result.dispatch_count <= result.baseline_dispatch_count,
        "optimized dispatch count ({}) should be <= baseline ({})",
        result.dispatch_count,
        result.baseline_dispatch_count,
    );

    // Should have explored at least the baseline config.
    assert!(
        result.configs_explored >= 1,
        "should explore at least 1 config, got {}",
        result.configs_explored,
    );

    // Summary should be non-empty.
    let summary = result.summarize();
    assert!(
        !summary.is_empty(),
        "OptimizationResult::summarize() should not be empty"
    );
    assert!(
        summary.contains("Baseline dispatches"),
        "summary should mention baseline dispatches"
    );
}

// =============================================================================
// Test 5: Schema Validation
// =============================================================================

#[test]
fn test_schema_validation() {
    let schema = fusion_gap_analysis_schema();

    // Should be a valid JSON object with a title.
    assert!(schema.is_object(), "schema should be a JSON object");
    assert_eq!(
        schema["title"].as_str(),
        Some("FusionGapAnalysis"),
        "schema title should be 'FusionGapAnalysis'"
    );

    // Should have required fields.
    let required = schema["required"]
        .as_array()
        .expect("schema should have 'required' array");
    let required_strs: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
    assert!(required_strs.contains(&"gaps"));
    assert!(required_strs.contains(&"total_dispatches"));
    assert!(required_strs.contains(&"theoretical_minimum"));

    // All FusionBlocker variants should be listed in the schema enum.
    let variants = schema["$defs"]["FusionBlocker"]["enum"]
        .as_array()
        .expect("FusionBlocker should have enum array");
    let variant_strs: Vec<&str> = variants.iter().filter_map(|v| v.as_str()).collect();

    // Cross-check against all known FusionBlocker variants.
    let expected_variants = [
        "FanOut",
        "ShapeMismatch",
        "NonFusibleOp",
        "NotDispatch",
        "AlreadyOptimal",
        "NoPeepholePattern",
        "NoDependency",
    ];
    for expected in &expected_variants {
        assert!(
            variant_strs.contains(expected),
            "schema should contain FusionBlocker variant '{expected}', got: {variant_strs:?}"
        );
    }
    assert_eq!(
        variant_strs.len(),
        expected_variants.len(),
        "schema should have exactly {} FusionBlocker variants",
        expected_variants.len()
    );
}

// =============================================================================
// Test 6: Protocol Types Serde
// =============================================================================

#[test]
fn test_protocol_types_serde() {
    let graph = build_five_node_graph();
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compilation should succeed");
    let analysis = analyze_fusion_gaps(&plan, &graph);

    // Build an OptimizationRequest from a real gap analysis.
    let request = OptimizationRequest {
        gap_analysis: analysis.clone(),
        model_name: "test_synthetic".into(),
        protocol_version: PROTOCOL_VERSION.into(),
        max_suggestions: 3,
    };

    // Serialize to JSON.
    let json = serde_json::to_value(&request).expect("request should serialize");
    assert_eq!(json["model_name"], "test_synthetic");
    assert_eq!(json["protocol_version"], PROTOCOL_VERSION);
    assert_eq!(json["max_suggestions"], 3);
    assert!(json["gap_analysis"].is_object());

    // Deserialize back and verify fields match.
    let restored: OptimizationRequest =
        serde_json::from_value(json).expect("request should deserialize");
    assert_eq!(restored.model_name, "test_synthetic");
    assert_eq!(restored.protocol_version, PROTOCOL_VERSION);
    assert_eq!(restored.max_suggestions, 3);
    assert_eq!(
        restored.gap_analysis.total_dispatches,
        analysis.total_dispatches
    );

    // OptimizationResponse round-trip.
    let response = OptimizationResponse {
        protocol_version: PROTOCOL_VERSION.into(),
        suggestions: vec![
            OptimizationSuggestion {
                gap_index: 0,
                optimization_type: "peephole_pass".into(),
                description: "Fuse relu-exp chain".into(),
                estimated_savings: 1,
            },
            OptimizationSuggestion {
                gap_index: 2,
                optimization_type: "native_op".into(),
                description: "Wrap matmul+add as NativeOp".into(),
                estimated_savings: 2,
            },
        ],
    };
    let resp_json = serde_json::to_value(&response).expect("response should serialize");
    let resp_restored: OptimizationResponse =
        serde_json::from_value(resp_json).expect("response should deserialize");
    assert_eq!(resp_restored.protocol_version, PROTOCOL_VERSION);
    assert_eq!(resp_restored.suggestions.len(), 2);
    assert_eq!(resp_restored.suggestions[0].gap_index, 0);
    assert_eq!(resp_restored.suggestions[1].estimated_savings, 2);
}

// =============================================================================
// Test 7: Blocker Distribution
// =============================================================================

#[test]
fn test_blocker_distribution() {
    let graph = build_mixed_ops_graph();
    let plan = compile_trace_to_plan_with_fusion(&graph)
        .expect("compilation should succeed on mixed ops graph");
    let analysis = analyze_fusion_gaps(&plan, &graph);

    let counts = analysis.blocker_counts();

    // With MatMul in the graph, we expect at least NonFusibleOp blockers.
    // The exact distribution depends on the compiler's fusion decisions, but
    // the blocker_counts map should be non-empty for a graph with non-fusible ops.
    if !analysis.gaps.is_empty() {
        assert!(
            !counts.is_empty(),
            "blocker_counts should be non-empty when gaps exist"
        );

        // Every gap's reason should appear in the counts.
        for gap in &analysis.gaps {
            let key = gap.reason.to_string();
            assert!(
                counts.contains_key(&key),
                "blocker_counts should contain key '{key}'"
            );
        }
    }

    // The summarize() output should be a valid non-empty string.
    let summary = analysis.summarize();
    assert!(!summary.is_empty(), "summarize() should not be empty");
    assert!(
        summary.contains("dispatches"),
        "summary should mention dispatches"
    );
    assert!(
        summary.contains("theoretical min"),
        "summary should mention theoretical minimum"
    );
}

// =============================================================================
// Test 8: Cost-Guided Optimization
// =============================================================================

#[test]
fn test_cost_guided_optimization() {
    let graph = build_five_node_graph();
    let cost_model = CostModel::apple_m4();

    let result = optimize_plan_with_cost(&graph, &cost_model, Duration::from_secs(10))
        .expect("cost-guided optimization should succeed");

    assert!(result.best_cost_ns >= 0.0);
    assert!(result.baseline_cost_ns >= 0.0);
    assert!(result.dispatch_count <= result.baseline_dispatch_count);
}

// =============================================================================
// Test 9: optimize_plan Also Populates Cost Fields
// =============================================================================

#[test]
fn test_optimize_plan_populates_cost_fields() {
    let graph = build_five_node_graph();

    let result =
        optimize_plan(&graph, Duration::from_secs(10)).expect("optimize_plan should succeed");

    assert!(
        result.baseline_cost_ns >= 0.0,
        "baseline_cost_ns should be non-negative, got {}",
        result.baseline_cost_ns
    );
    assert!(
        result.best_cost_ns >= 0.0,
        "best_cost_ns should be non-negative, got {}",
        result.best_cost_ns
    );
}

// =============================================================================
// Test 10: Multi-Segment Optimization
// =============================================================================

#[test]
fn test_multi_segment_optimization() {
    let graph_a = build_three_node_graph();
    let graph_b = build_five_node_graph();
    let cost_model = CostModel::apple_m4();

    let segments: Vec<(&str, &ComputationGraph)> =
        vec![("encoder", &graph_a), ("decoder", &graph_b)];

    let results: Vec<SegmentOptimizationResult> =
        optimize_segments(&segments, &cost_model, Duration::from_secs(5));

    assert_eq!(results.len(), 2, "should have results for both segments");
    assert_eq!(results[0].segment_name, "encoder");
    assert_eq!(results[1].segment_name, "decoder");

    for seg in &results {
        assert!(
            seg.result.dispatch_count <= seg.result.baseline_dispatch_count,
            "segment '{}': optimized ({}) should be <= baseline ({})",
            seg.segment_name,
            seg.result.dispatch_count,
            seg.result.baseline_dispatch_count,
        );
    }
}

// =============================================================================
// Test 11: Top Expensive Steps
// =============================================================================

#[test]
fn test_top_expensive_steps() {
    let graph = build_large_chain_graph(8);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compilation should succeed");
    let cost_model = CostModel::apple_m4();
    let estimate = cost_model.estimate(&plan);

    let top3 = estimate.top_expensive_steps(3);

    assert!(
        top3.len() <= 3,
        "top_expensive_steps(3) should return <= 3, got {}",
        top3.len()
    );

    for window in top3.windows(2) {
        assert!(
            window[0].1 >= window[1].1,
            "top_expensive_steps should be sorted descending: {:.1} >= {:.1}",
            window[0].1,
            window[1].1,
        );
    }

    let top100 = estimate.top_expensive_steps(100);
    assert_eq!(
        top100.len(),
        estimate.per_step_ns.len(),
        "requesting N > available should return all steps"
    );
}
