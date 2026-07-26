// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for #656: Sigmoid MSL emission via dedicated DispatchStep.

use nn_dsl::ir::ScalarType;
use nn_dsl::test_kernels::square_kernel;
use nn_dsl::{
    build_dispatch_plan, emit_tensor_msl, DispatchStep, TensorKernelDef, TensorNode, TensorNodeId,
    TensorOpKind,
};

/// AC2: Sigmoid tensor op produces compilable MSL with the correct kernel.
#[test]
fn test_sigmoid_msl_emission() {
    let def = TensorKernelDef::new(
        "sig_emit",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: vec![4, 32],
                },
                vec![4, 32],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Sigmoid {
                    input: TensorNodeId::new(0),
                },
                vec![4, 32],
            ),
        ],
        TensorNodeId::new(1),
    );
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("Sigmoid MSL emission");
    assert!(
        msl.contains("sig_emit_sigmoid_n1"),
        "kernel name must appear in MSL:\n{msl}"
    );
    assert!(
        msl.contains("metal::precise::exp(-x)"),
        "sigmoid formula must use metal::precise::exp(-x) for numerical accuracy:\n{msl}"
    );
    assert!(
        msl.contains("128u"),
        "total_elements guard (4*32=128) must appear:\n{msl}"
    );
    assert!(
        msl.starts_with("#include <metal_stdlib>"),
        "MSL prelude must be present"
    );
}

/// AC1: Sigmoid dispatches as DispatchStep::Sigmoid, not Elementwise.
#[test]
fn test_sigmoid_dispatches_as_sigmoid_step() {
    let def = TensorKernelDef::new(
        "sig_plan",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: vec![2, 4],
                },
                vec![2, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Sigmoid {
                    input: TensorNodeId::new(0),
                },
                vec![2, 4],
            ),
        ],
        TensorNodeId::new(1),
    );
    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32).expect("plan");
    assert_eq!(plan.len(), 1, "should have exactly 1 dispatch step");
    assert!(
        matches!(
            &plan[0],
            DispatchStep::Sigmoid {
                total_elements: 8,
                ..
            }
        ),
        "expected Sigmoid dispatch step, got {:?}",
        &plan[0]
    );
}

/// Elementwise nodes emit inline scalar MSL via the `scalar_kernel` field (#656).
#[test]
fn test_elementwise_emits_scalar_msl() {
    let def = TensorKernelDef::new(
        "ew_emit",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: vec![2, 4],
                },
                vec![2, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Elementwise {
                    kernel: square_kernel(),
                    inputs: vec![TensorNodeId::new(0)],
                },
                vec![2, 4],
            ),
        ],
        TensorNodeId::new(1),
    );
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("Elementwise MSL emission");
    assert!(
        msl.starts_with("#include <metal_stdlib>"),
        "prelude must be present"
    );
    // After #656, Elementwise steps emit their scalar kernel inline.
    assert!(
        msl.contains("[[kernel]]"),
        "emitted MSL must contain a kernel entry point:\n{msl}"
    );
}

/// AC3: Mixed graph (Elementwise + Reduce) emits MSL for both dispatch steps.
#[test]
fn test_elementwise_reduce_mixed_graph() {
    use nn_dsl::ReduceOp;

    let def = TensorKernelDef::new(
        "ew_reduce",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: vec![4, 32],
                },
                vec![4, 32],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Elementwise {
                    kernel: square_kernel(),
                    inputs: vec![TensorNodeId::new(0)],
                },
                vec![4, 32],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Reduce {
                    op: ReduceOp::Sum,
                    input: TensorNodeId::new(1),
                    axis: 1,
                    keepdim: false,
                },
                vec![4],
            ),
        ],
        TensorNodeId::new(2),
    );
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("mixed graph MSL");
    // Elementwise scalar kernel should be emitted.
    assert!(
        msl.contains("[[kernel]]"),
        "elementwise kernel must appear in MSL:\n{msl}"
    );
    // Reduce kernel should also be emitted.
    assert!(
        msl.contains("ew_reduce_reduce_sum_n2"),
        "reduce kernel must appear:\n{msl}"
    );
}

/// AC3: Mixed graph (Sigmoid + Reduce) emits MSL for both dispatch steps.
#[test]
fn test_sigmoid_reduce_mixed_graph() {
    use nn_dsl::ReduceOp;

    let def = TensorKernelDef::new(
        "sig_reduce",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: vec![4, 32],
                },
                vec![4, 32],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Sigmoid {
                    input: TensorNodeId::new(0),
                },
                vec![4, 32],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Reduce {
                    op: ReduceOp::Sum,
                    input: TensorNodeId::new(1),
                    axis: 1,
                    keepdim: false,
                },
                vec![4],
            ),
        ],
        TensorNodeId::new(2),
    );
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("mixed graph MSL");
    assert!(
        msl.contains("sig_reduce_sigmoid_n1"),
        "sigmoid kernel must appear:\n{msl}"
    );
    assert!(
        msl.contains("sig_reduce_reduce_sum_n2"),
        "reduce kernel must appear:\n{msl}"
    );
}
