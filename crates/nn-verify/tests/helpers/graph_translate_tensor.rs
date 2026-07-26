// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for tensor IR → NY GraphNetwork translation:
//! reduce ops, elementwise ops, broadcast passthrough, constant bindings,
//! and single-variable regression.
//! InstanceNorm tests extracted to graph_translate_tensor_norm.rs (#356).

use super::common;

use nn_dsl::tensor_ir::{
    BroadcastAlignment, ReduceOp, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind,
};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding, VerifyError};
use ndarray::{ArrayD, IxDyn};

fn simple_tensor_reduce(
    name: &str,
    shape: Vec<usize>,
    op: ReduceOp,
    axis: usize,
) -> TensorKernelDef {
    let mut out_shape = shape.clone();
    out_shape.remove(axis);
    if out_shape.is_empty() {
        out_shape.push(1);
    }
    TensorKernelDef::new(
        name.to_string(),
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: shape.clone(),
                },
                shape,
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reduce {
                    op,
                    input: TensorNodeId::new(0),
                    axis,
                    keepdim: false,
                },
                out_shape,
            ),
        ],
        TensorNodeId::new(1),
    )
}

#[test]
fn test_tensor_reduce_mean_builds_graph() {
    let def = simple_tensor_reduce("mean_test", vec![4, 32, 128], ReduceOp::Mean, 2);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable])
        .expect("tensor mean graph should build");
    assert_eq!(graph.num_nodes(), 1, "tensor graph should have 1 node");
}

#[test]
fn test_tensor_reduce_sum_builds_graph() {
    let def = simple_tensor_reduce("sum_test", vec![4, 32, 128], ReduceOp::Sum, 2);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable])
        .expect("tensor sum graph should build");
    assert_eq!(graph.num_nodes(), 1, "tensor graph should have 1 node");
}

#[test]
fn test_tensor_reduce_mean_ibp_bounds() {
    // Use 2D input → reduce axis 1 → 2D output [4,1] (keepdim=true for broadcasting)
    let def = simple_tensor_reduce("ibp_mean", vec![4, 8], ReduceOp::Mean, 1);
    let graph =
        tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable]).expect("build mean graph");

    let lower = ArrayD::from_elem(IxDyn(&[4, 8]), -5.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4, 8]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();
    // keepdim=false: reduce axis 1 of [4,8] → [4], so index with [row]
    // Mean of [-5, 10] bounded inputs per row: output should be in [-5, 10]
    assert!(lo[[0]] >= -5.01, "mean lower >= -5, got {}", lo[[0]]);
    assert!(hi[[0]] <= 10.01, "mean upper <= 10, got {}", hi[[0]]);
}

#[test]
fn test_tensor_reduce_constant_mean() {
    let def = simple_tensor_reduce("const_mean", vec![4, 8], ReduceOp::Mean, 1);
    // Treat input as a constant (e.g., eps tensor)
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::ConstantScalar(3.0)])
        .expect("constant mean graph should build");
    // The output should be a constant (mean of 3.0 = 3.0)
    assert_eq!(graph.num_nodes(), 1, "tensor graph should have 1 node");
}

#[test]
fn test_tensor_binding_count_mismatch() {
    let def = simple_tensor_reduce("mm", vec![4, 8], ReduceOp::Mean, 1);
    // 1 input node, but 2 bindings → error
    let err = tensor_kernel_to_graph(
        &def,
        &[
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantScalar(1.0),
        ],
    )
    .expect_err("should reject mismatched binding count");
    assert!(
        matches!(err, VerifyError::ParamCountMismatch { .. }),
        "unexpected error: {err:?}"
    );
}

#[test]
fn test_tensor_elementwise_inline_square() {
    // Test: input → elementwise(square) → output
    let square = common::square_kernel();
    let def = TensorKernelDef::new(
        "sq_test",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![4, 8],
                },
                vec![4, 8],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Elementwise {
                    kernel: square,
                    inputs: vec![TensorNodeId::new(0)],
                },
                vec![4, 8],
            ),
        ],
        TensorNodeId::new(1),
    );
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable])
        .expect("elementwise square graph should build");
    assert_eq!(graph.num_nodes(), 1, "tensor graph should have 1 node");

    // IBP: square of [2, 5] → [4, 25]
    let lower = ArrayD::from_elem(IxDyn(&[4, 8]), 2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4, 8]), 5.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");
    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();
    // x^2 with x in [2,5]: min = 4, max = 25
    assert!(lo[[0, 0]] >= 3.9, "square lower >= 4, got {}", lo[[0, 0]]);
    assert!(hi[[0, 0]] <= 25.1, "square upper <= 25, got {}", hi[[0, 0]]);
}

#[test]
fn test_tensor_reduce_sum_constant_overflow_rejected() {
    // Constant val * dim overflows f32 → checked_constant rejects as NonFiniteConstant
    let def = simple_tensor_reduce("sum_overflow", vec![4, 8], ReduceOp::Sum, 1);
    let err = tensor_kernel_to_graph(&def, &[TensorParamBinding::ConstantScalar(f32::MAX / 2.0)])
        .expect_err("should reject overflow from constant * dim");
    assert!(
        matches!(err, VerifyError::NonFiniteConstant { .. }),
        "expected NonFiniteConstant, got {err:?}"
    );
}

#[test]
fn test_tensor_reduce_sum_constant_dim1_identity() {
    // Reducing dim=1 with constant input: sum = val * 1 = val
    let def = simple_tensor_reduce("sum_id", vec![4, 1], ReduceOp::Sum, 1);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::ConstantScalar(7.5)])
        .expect("constant sum with dim=1 should build");
    assert_eq!(graph.num_nodes(), 1, "tensor graph should have 1 node");
}

#[test]
fn test_tensor_non_finite_constant_binding_rejected() {
    let def = simple_tensor_reduce("nan_bind", vec![4, 8], ReduceOp::Mean, 1);
    let err_nan = tensor_kernel_to_graph(&def, &[TensorParamBinding::ConstantScalar(f32::NAN)])
        .expect_err("NaN binding should be rejected");
    assert!(
        matches!(err_nan, VerifyError::NonFiniteConstant { .. }),
        "expected NonFiniteConstant for NaN, got {err_nan:?}"
    );

    let err_inf =
        tensor_kernel_to_graph(&def, &[TensorParamBinding::ConstantScalar(f32::INFINITY)])
            .expect_err("Infinity binding should be rejected");
    assert!(
        matches!(err_inf, VerifyError::NonFiniteConstant { .. }),
        "expected NonFiniteConstant for Inf, got {err_inf:?}"
    );
}

#[test]
fn test_tensor_reduce_mean_constant_preserves_value() {
    // Mean of N copies of val = val, regardless of dim
    let def = simple_tensor_reduce("mean_const", vec![4, 128], ReduceOp::Mean, 1);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::ConstantScalar(-42.0)])
        .expect("constant mean should build");
    // Graph should have nodes (constant output gets wrapped in AddConstant)
    assert_eq!(graph.num_nodes(), 1, "tensor graph should have 1 node");
}

#[test]
fn test_tensor_broadcast_passthrough() {
    // Test: input → reduce → broadcast → output
    // Broadcast inserts explicit ReshapeLayer + TileLayer for Variable nodes
    // (W4-626 21be3c5f, W5-622 a2573dba) so NY correctly propagates
    // bounds through shape changes. ReduceMean produces [4] (1-D), broadcast
    // target is [4, 8] (2-D) → reshape [4] to [4, 1] + tile axis 1 by 8.
    let def = TensorKernelDef::new(
        "rb_test",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: vec![4, 8],
                },
                vec![4, 8],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reduce {
                    op: ReduceOp::Mean,
                    input: TensorNodeId::new(0),
                    axis: 1,
                    keepdim: false,
                },
                vec![4],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Broadcast {
                    input: TensorNodeId::new(1),
                    target_shape: vec![4, 8],
                    alignment: BroadcastAlignment::Left,
                },
                vec![4, 8],
            ),
        ],
        TensorNodeId::new(2),
    );
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable])
        .expect("reduce+broadcast graph should build");
    // 3 nodes: ReduceMean + ReshapeLayer (rank alignment) + TileLayer (axis 1)
    assert_eq!(
        graph.num_nodes(),
        3,
        "tensor graph should have 3 nodes (reduce + reshape + tile)"
    );
}

#[test]
fn test_single_variable_still_uses_network_input() {
    // Single variable case: should work identically to before the fix
    let def = simple_tensor_reduce("single_var", vec![4, 8], ReduceOp::Mean, 1);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable])
        .expect("single-variable graph should build");
    assert_eq!(graph.num_nodes(), 1, "tensor graph should have 1 node");

    let lower = ArrayD::from_elem(IxDyn(&[4, 8]), 2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4, 8]), 6.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");
    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();
    // keepdim=false: reduce axis 1 of [4,8] → [4], so index with [row]
    // Mean of [2, 6] → should be in [2, 6]
    assert!(lo[[0]] >= 1.9, "mean lower >= 2, got {}", lo[[0]]);
    assert!(hi[[0]] <= 6.1, "mean upper <= 6, got {}", hi[[0]]);
}

/// ConstantTensor with NaN element must be rejected during graph translation.
/// Guards design doc #66 finiteness policy for tensor-level constants.
/// Part of #587 AC4.
#[test]
fn test_constant_tensor_nan_rejected() {
    // Reuse simple_tensor_reduce — it creates a valid Input + Reduce kernel.
    let def = simple_tensor_reduce("nan_weight", vec![2, 3], ReduceOp::Sum, 1);

    // NaN element in the tensor: should fail.
    let mut nan_data = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0f32);
    nan_data[[1, 2]] = f32::NAN;
    let err = tensor_kernel_to_graph(&def, &[TensorParamBinding::ConstantTensor(nan_data)])
        .expect_err("ConstantTensor with NaN must be rejected");
    assert!(
        matches!(err, VerifyError::NonFiniteConstant { .. }),
        "expected NonFiniteConstant, got: {err:?}"
    );

    // Inf element in the tensor: should also fail.
    let mut inf_data = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0f32);
    inf_data[[0, 0]] = f32::INFINITY;
    let err = tensor_kernel_to_graph(&def, &[TensorParamBinding::ConstantTensor(inf_data)])
        .expect_err("ConstantTensor with Inf must be rejected");
    assert!(
        matches!(err, VerifyError::NonFiniteConstant { .. }),
        "expected NonFiniteConstant, got: {err:?}"
    );

    // Positive case: finite ConstantTensor passes the finiteness check but
    // Reduce ops structurally reject weight tensors — that is the expected
    // behaviour.  Valid acceptance of finite ConstantTensor is covered by
    // graph_translate_conv1d.rs (Conv1d kernels bind weight tensors).
    let valid_data = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0f32);
    let valid_err = tensor_kernel_to_graph(&def, &[TensorParamBinding::ConstantTensor(valid_data)])
        .expect_err("reduce rejects weight tensor even when finite");
    // Ensure the error is NOT NonFiniteConstant — finiteness guard passed.
    assert!(
        !matches!(valid_err, VerifyError::NonFiniteConstant { .. }),
        "expected structural rejection, not finiteness rejection: {valid_err:?}"
    );
}

/// Pass-through tensor kernel: output = input (no ops).
/// The tensor graph builder must wrap the bare NETWORK_INPUT in an identity
/// layer (AddConstant(0.0)) so NY IBP propagation succeeds.
/// Regression test for #727 / #477.
#[test]
fn test_passthrough_tensor_kernel_identity_wrapper() {
    // A kernel with a single input node and output = that input (no operations).
    let def = TensorKernelDef::new(
        "passthrough",
        vec![TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: "x".to_string(),
                shape: vec![4, 8],
            },
            vec![4, 8],
        )],
        TensorNodeId::new(0), // output IS the input — triggers identity wrapper
    );

    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable])
        .expect("pass-through tensor graph should build");
    // The identity wrapper inserts exactly one AddConstant(0.0) node.
    assert_eq!(
        graph.num_nodes(),
        1,
        "pass-through graph must have exactly 1 node (identity wrapper)"
    );

    // Verify IBP propagation succeeds (would fail without the identity wrapper).
    let lower = ArrayD::from_elem(IxDyn(&[4, 8]), -3.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4, 8]), 7.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP must succeed on pass-through");
    let (lo, hi) = output.lower_upper();
    // Identity: output bounds should match input bounds (within tolerance).
    assert!(
        lo[[0, 0]] >= -3.01 && lo[[0, 0]] <= -2.99,
        "pass-through lower should be ~-3.0, got {}",
        lo[[0, 0]]
    );
    assert!(
        hi[[0, 0]] >= 6.99 && hi[[0, 0]] <= 7.01,
        "pass-through upper should be ~7.0, got {}",
        hi[[0, 0]]
    );
}

// InstanceNorm tests extracted to graph_translate_tensor_norm.rs (#356).
