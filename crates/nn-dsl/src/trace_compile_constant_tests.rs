// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `CompiledStep::ConstantValue` and powf finiteness validation.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use crate::trace_compile::{compile_trace, compile_trace_to_plan, CompiledStep};

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

fn binary_node(
    id: u64,
    name: &str,
    op: TraceOp,
    lhs_id: u64,
    rhs_id: u64,
    shape: &[usize],
) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        op,
        vec![lhs_id, rhs_id],
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

// -- ConstantValue tests ---------------------------------------------------

#[test]
fn test_compile_constant_produces_constant_value() {
    // A Constant node should produce ConstantValue, NOT InputForward.
    let graph = ComputationGraph::from_nodes(vec![TraceNode::new(
        0,
        "const_0".into(),
        TraceOp::Constant { value: 1.0 },
        vec![],
        vec![4, 8],
        DType::F32,
    )]);
    let steps = compile_trace(&graph).expect("constant should compile");
    assert_eq!(steps.len(), 1);
    match &steps[0] {
        CompiledStep::ConstantValue { value, shape } => {
            assert_eq!(*value, 1.0);
            assert_eq!(shape, &[4, 8]);
        }
        other => panic!("expected ConstantValue, got {other:?}"),
    }
}

#[test]
fn test_compile_constant_does_not_count_as_input() {
    // Graph: Input + Constant + Add. The plan should have 1 input shape
    // (not 2), because Constant is not an external input.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        TraceNode::new(
            1,
            "const_ones".into(),
            TraceOp::Constant { value: 1.0 },
            vec![],
            vec![4],
            DType::F32,
        ),
        binary_node(2, "add_0", TraceOp::Add, 0, 1, &[4]),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("should compile");
    assert_eq!(
        plan.input_shapes.len(),
        1,
        "only Input nodes count as inputs"
    );
    assert_eq!(plan.input_shapes[0], vec![4]);
    assert!(matches!(plan.steps[0], CompiledStep::InputForward));
    assert!(matches!(plan.steps[1], CompiledStep::ConstantValue { .. }));
    assert!(matches!(plan.steps[2], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_constant_zero_value() {
    // zeros() during tracing produces Constant { value: 0.0 }.
    let graph = ComputationGraph::from_nodes(vec![TraceNode::new(
        0,
        "zeros".into(),
        TraceOp::Constant { value: 0.0 },
        vec![],
        vec![2, 3],
        DType::F32,
    )]);
    let steps = compile_trace(&graph).expect("zero constant should compile");
    match &steps[0] {
        CompiledStep::ConstantValue { value, shape } => {
            assert_eq!(*value, 0.0);
            assert_eq!(shape, &[2, 3]);
        }
        other => panic!("expected ConstantValue, got {other:?}"),
    }
}

// -- Powf finiteness tests ----------------------------------------------------

#[test]
fn test_compile_powf_rejects_nan_exponent() {
    // NaN exponent must be rejected — it would silently produce wrong results.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "powf_nan", TraceOp::Powf { exponent: f64::NAN }, 0, &[4]),
    ]);
    let err = compile_trace(&graph).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("NonFiniteConstant"),
        "expected NonFiniteConstant error for NaN exponent, got: {msg}"
    );
}

#[test]
fn test_compile_powf_rejects_inf_exponent() {
    // Infinity exponent must be rejected.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(
            1,
            "powf_inf",
            TraceOp::Powf {
                exponent: f64::INFINITY,
            },
            0,
            &[4],
        ),
    ]);
    let err = compile_trace(&graph).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("NonFiniteConstant"),
        "expected NonFiniteConstant error for Inf exponent, got: {msg}"
    );
}

#[test]
fn test_compile_powf_accepts_finite_exponent() {
    // Normal finite exponent should compile successfully.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "powf_2", TraceOp::Powf { exponent: 2.0 }, 0, &[4]),
    ]);
    let steps = compile_trace(&graph).expect("finite exponent should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[0], CompiledStep::InputForward));
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
}
