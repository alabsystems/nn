// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-variable tensor kernel → NY translation tests (#70).
//!
//! Tests for multi-variable stacking via SliceLayer, element-wise add,
//! and reduce-then-add patterns. Extracted from `graph_translate_tensor_multi.rs`
//! to stay under the 500-line file limit (#542).

use nn_dsl::ir::{BinOpKind, IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType};
use nn_dsl::tensor_ir::{ReduceOp, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Scalar kernel: a + b
fn scalar_add_kernel() -> KernelDef {
    KernelDef::new(
        "add",
        vec![
            Param::new("a", ScalarType::F32),
            Param::new("b", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    )
}

/// Build a tensor kernel with two variable inputs: x + y (element-wise add).
fn two_variable_add(shape: Vec<usize>) -> TensorKernelDef {
    TensorKernelDef::new(
        "two_var_add",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: shape.clone(),
                },
                shape.clone(),
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "y".to_string(),
                    shape: shape.clone(),
                },
                shape.clone(),
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Elementwise {
                    kernel: scalar_add_kernel(),
                    inputs: vec![TensorNodeId::new(0), TensorNodeId::new(1)],
                },
                shape,
            ),
        ],
        TensorNodeId::new(2),
    )
}

/// Two variable inputs → reduce_mean each → add results.
fn two_variable_reduce_add(shape: Vec<usize>, reduce_axis: usize) -> TensorKernelDef {
    let mut out_shape = shape.clone();
    out_shape.remove(reduce_axis);
    if out_shape.is_empty() {
        out_shape.push(1);
    }
    TensorKernelDef::new(
        "multi_reduce_add",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".to_string(),
                    shape: shape.clone(),
                },
                shape.clone(),
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "y".to_string(),
                    shape: shape.clone(),
                },
                shape,
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::Reduce {
                    op: ReduceOp::Mean,
                    input: TensorNodeId::new(0),
                    axis: reduce_axis,
                    keepdim: false,
                },
                out_shape.clone(),
            ),
            TensorNode::new(
                TensorNodeId::new(3),
                TensorOpKind::Reduce {
                    op: ReduceOp::Mean,
                    input: TensorNodeId::new(1),
                    axis: reduce_axis,
                    keepdim: false,
                },
                out_shape.clone(),
            ),
            TensorNode::new(
                TensorNodeId::new(4),
                TensorOpKind::Elementwise {
                    kernel: scalar_add_kernel(),
                    inputs: vec![TensorNodeId::new(2), TensorNodeId::new(3)],
                },
                out_shape,
            ),
        ],
        TensorNodeId::new(4),
    )
}

#[test]
fn test_multi_variable_tensor_builds_graph() {
    let def = two_variable_add(vec![4, 8]);
    let graph = tensor_kernel_to_graph(
        &def,
        &[TensorParamBinding::Variable, TensorParamBinding::Variable],
    )
    .expect("two-variable tensor graph should build");
    // Must have at least the slice nodes + elementwise nodes
    assert!(
        graph.num_nodes() >= 3,
        "multi-variable graph needs slice + compute nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_multi_variable_tensor_ibp_distinct_bounds() {
    let def = two_variable_add(vec![4, 8]);
    let graph = tensor_kernel_to_graph(
        &def,
        &[TensorParamBinding::Variable, TensorParamBinding::Variable],
    )
    .expect("build two-variable graph");

    // Input: [2, 4, 8] — two variables stacked along dim 0.
    // Variable 0 (x): bounds [1, 3], Variable 1 (y): bounds [10, 20].
    // Output (x + y): bounds should be [11, 23].
    let mut lower = ArrayD::zeros(IxDyn(&[2, 4, 8]));
    let mut upper = ArrayD::zeros(IxDyn(&[2, 4, 8]));
    // Variable 0: [1, 3]
    lower.slice_mut(ndarray::s![0, .., ..]).fill(1.0f32);
    upper.slice_mut(ndarray::s![0, .., ..]).fill(3.0f32);
    // Variable 1: [10, 20]
    lower.slice_mut(ndarray::s![1, .., ..]).fill(10.0f32);
    upper.slice_mut(ndarray::s![1, .., ..]).fill(20.0f32);

    let input = BoundedTensor::new(lower, upper).expect("valid stacked bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    let (lo, hi) = output.lower_upper();

    // x in [1,3] + y in [10,20] → output in [11,23]
    assert!(
        lo.iter().all(|&v| v >= 10.9),
        "output lower should be >= 11, got {lo:?}"
    );
    assert!(
        hi.iter().all(|&v| v <= 23.1),
        "output upper should be <= 23, got {hi:?}"
    );
}

#[test]
fn test_multi_variable_reduce_then_add() {
    let def = two_variable_reduce_add(vec![4, 8], 1);
    let bindings = [TensorParamBinding::Variable, TensorParamBinding::Variable];
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("multi-variable reduce+add should build");

    // Input: [2, 4, 8] — two variables stacked along dim 0
    let mut lower = ArrayD::zeros(IxDyn(&[2, 4, 8]));
    let mut upper = ArrayD::zeros(IxDyn(&[2, 4, 8]));
    // x in [0, 4], y in [10, 14]
    lower.slice_mut(ndarray::s![0, .., ..]).fill(0.0f32);
    upper.slice_mut(ndarray::s![0, .., ..]).fill(4.0f32);
    lower.slice_mut(ndarray::s![1, .., ..]).fill(10.0f32);
    upper.slice_mut(ndarray::s![1, .., ..]).fill(14.0f32);

    let input = BoundedTensor::new(lower, upper).expect("valid bounds");
    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();

    // mean(x) in [0, 4], mean(y) in [10, 14] → sum in [10, 18]
    for &v in lo.iter() {
        assert!(v >= 9.9, "output lower should be >= 10, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 18.1, "output upper should be <= 18, got {v}");
    }
}
