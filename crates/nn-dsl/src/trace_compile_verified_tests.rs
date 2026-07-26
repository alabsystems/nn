// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `compile_trace_to_plan_verified()` — the compilation gate
//! that rejects unverifiable learned operations.
//!
//! Part of #2218: compiler-enforced verifiability.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use crate::tensor_ir::TensorIRError;
use crate::trace_compile::compile_trace_to_plan_verified;
use crate::verifiability::VerifiabilityClass;

// -- Helpers ------------------------------------------------------------------

fn graph_from_nodes(nodes: Vec<TraceNode>) -> ComputationGraph {
    ComputationGraph::from_nodes(nodes)
}

fn input_node(id: u64, shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        format!("input_{id}"),
        TraceOp::Input,
        vec![],
        shape.to_vec(),
        DType::F32,
    )
}

fn unary_node(id: u64, name: &str, op: TraceOp, input_id: u64, shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        op,
        vec![input_id],
        shape.to_vec(),
        DType::F32,
    )
}

fn binary_node(
    id: u64,
    name: &str,
    op: TraceOp,
    inputs: &[u64],
    shape: &[usize],
) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        op,
        inputs.to_vec(),
        shape.to_vec(),
        DType::F32,
    )
}

// -- Tests: verifiable graphs should compile ------------------------------------

/// A simple relu graph (fully verifiable) compiles successfully.
#[test]
fn test_verified_relu_compiles() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 4, 16]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[1, 4, 16]),
    ]);
    let plan = compile_trace_to_plan_verified(&graph).expect("relu is verifiable");
    assert!(!plan.steps.is_empty());
}

/// Linear + ReLU (both verifiable) compiles successfully.
#[test]
fn test_verified_linear_relu_compiles() {
    let weight = WeightRef::new(vec![1.0; 16], vec![4, 4]).expect("test weight");
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 4]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear {
                weight,
                bias: None,
            },
            vec![0],
            vec![1, 4],
            DType::F32,
        ),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[1, 4]),
    ]);
    let plan = compile_trace_to_plan_verified(&graph).expect("linear+relu is verifiable");
    assert!(plan.steps.len() >= 2);
}

/// Add (verifiable binary op) compiles successfully.
#[test]
fn test_verified_add_compiles() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 4, 16]),
        input_node(1, &[1, 4, 16]),
        binary_node(2, "add_0", TraceOp::Add, &[0, 1], &[1, 4, 16]),
    ]);
    let plan = compile_trace_to_plan_verified(&graph).expect("add is verifiable");
    assert!(!plan.steps.is_empty());
}

// -- Tests: unverifiable graphs should fail ------------------------------------

/// SDPA (unverifiable learned) should fail compilation.
#[test]
fn test_verified_sdpa_fails() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 8, 16, 64]),
        input_node(1, &[1, 8, 16, 64]),
        input_node(2, &[1, 8, 16, 64]),
        TraceNode::new(
            3,
            "sdpa_0".into(),
            TraceOp::Sdpa { scale: 0.125 },
            vec![0, 1, 2],
            vec![1, 8, 16, 64],
            DType::F32,
        ),
    ]);
    let err = compile_trace_to_plan_verified(&graph).unwrap_err();
    match err {
        TensorIRError::UnverifiableOperation { class, .. } => {
            assert_eq!(class, VerifiabilityClass::UnverifiableLearned);
        }
        other => panic!("expected UnverifiableOperation, got: {other}"),
    }
}

/// Custom op (unverifiable learned) should fail compilation.
#[test]
fn test_verified_custom_op_fails() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 4, 16]),
        unary_node(
            1,
            "custom_0",
            TraceOp::Custom {
                name: "mystery_op".to_string(),
            },
            0,
            &[1, 4, 16],
        ),
    ]);
    let err = compile_trace_to_plan_verified(&graph).unwrap_err();
    match err {
        TensorIRError::UnverifiableOperation { op_name, class, .. } => {
            assert!(op_name.contains("mystery_op"), "op_name should contain 'mystery_op', got: {op_name}");
            assert_eq!(class, VerifiabilityClass::UnverifiableLearned);
        }
        other => panic!("expected UnverifiableOperation, got: {other}"),
    }
}

/// Powf with exponent=3 (unverifiable learned) should fail compilation.
#[test]
fn test_verified_powf_cubic_fails() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 4, 16]),
        unary_node(1, "powf_0", TraceOp::Powf { exponent: 3.0 }, 0, &[1, 4, 16]),
    ]);
    let err = compile_trace_to_plan_verified(&graph).unwrap_err();
    assert!(matches!(err, TensorIRError::UnverifiableOperation { .. }));
}

// -- Tests: safe unverifiable ops should compile --------------------------------

/// Atan2 (unverifiable safe) should compile — it's in signal processing, not learned path.
#[test]
fn test_verified_atan2_compiles() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 4, 16]),
        input_node(1, &[1, 4, 16]),
        binary_node(2, "atan2_0", TraceOp::Atan2, &[0, 1], &[1, 4, 16]),
    ]);
    // Atan2 is UnverifiableSafe — compilation gate should allow it.
    let plan = compile_trace_to_plan_verified(&graph).expect("atan2 is safe, should compile");
    assert!(!plan.steps.is_empty());
}

/// Dropout (passthrough) should compile.
#[test]
fn test_verified_dropout_compiles() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 4, 16]),
        unary_node(1, "dropout_0", TraceOp::Dropout, 0, &[1, 4, 16]),
    ]);
    let plan = compile_trace_to_plan_verified(&graph).expect("dropout is passthrough");
    assert!(!plan.steps.is_empty());
}

/// Reshape (shape-only) should compile.
#[test]
fn test_verified_reshape_compiles() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 4, 16]),
        unary_node(
            1,
            "reshape_0",
            TraceOp::Reshape {
                target_shape: vec![1, 64],
            },
            0,
            &[1, 64],
        ),
    ]);
    let plan = compile_trace_to_plan_verified(&graph).expect("reshape is shape-only");
    assert!(!plan.steps.is_empty());
}

// -- Test: error includes useful diagnostics -----------------------------------

#[test]
fn test_unverifiable_error_has_node_id() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 8, 16, 64]),
        input_node(1, &[1, 8, 16, 64]),
        input_node(2, &[1, 8, 16, 64]),
        TraceNode::new(
            42,
            "sdpa_bad".into(),
            TraceOp::Sdpa { scale: 0.125 },
            vec![0, 1, 2],
            vec![1, 8, 16, 64],
            DType::F32,
        ),
    ]);
    let err = compile_trace_to_plan_verified(&graph).unwrap_err();
    match err {
        TensorIRError::UnverifiableOperation { node_id, .. } => {
            assert_eq!(node_id, 42, "error should include correct node_id");
        }
        other => panic!("expected UnverifiableOperation, got: {other}"),
    }
}

// -- Tests: allow_unverifiable annotation opt-out ------------------------------

/// SDPA with allow_unverifiable annotation should compile despite being UnverifiableLearned.
#[test]
fn test_allow_unverifiable_sdpa_compiles() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 8, 16, 64]),
        input_node(1, &[1, 8, 16, 64]),
        input_node(2, &[1, 8, 16, 64]),
        TraceNode::new(
            3,
            "sdpa_annotated".into(),
            TraceOp::Sdpa { scale: 0.125 },
            vec![0, 1, 2],
            vec![1, 8, 16, 64],
            DType::F32,
        )
        .with_allow_unverifiable("SDPA in attention head — bounded by softmax output in [0,1]"),
    ]);
    let plan =
        compile_trace_to_plan_verified(&graph).expect("annotated SDPA should compile");
    assert!(!plan.steps.is_empty());
}

/// Powf(3) with allow_unverifiable annotation compiles (normally rejected as UnverifiableLearned).
#[test]
fn test_allow_unverifiable_powf_cubic_compiles() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 4, 16]),
        unary_node(
            1,
            "powf_annotated",
            TraceOp::Powf { exponent: 3.0 },
            0,
            &[1, 4, 16],
        )
        .with_allow_unverifiable("cubic activation in non-learned path"),
    ]);
    let plan =
        compile_trace_to_plan_verified(&graph).expect("annotated powf(3) should compile");
    assert!(!plan.steps.is_empty());
}

/// Annotation accessor returns the reason string.
#[test]
fn test_allow_unverifiable_accessor() {
    let node = TraceNode::new(
        0,
        "test".into(),
        TraceOp::Relu,
        vec![],
        vec![1],
        DType::F32,
    )
    .with_allow_unverifiable("test reason");
    assert_eq!(node.allow_unverifiable(), Some("test reason"));

    let plain = TraceNode::new(
        1,
        "plain".into(),
        TraceOp::Relu,
        vec![],
        vec![1],
        DType::F32,
    );
    assert_eq!(plain.allow_unverifiable(), None);
}
