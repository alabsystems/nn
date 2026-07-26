// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for scalar-level KernelDef composition.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use crate::ir::{CompareOpKind, IRNodeKind, UnaryFnKind};

use super::build_fused_scalar_kernel;

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

#[test]
fn test_compose_two_unary_ops() {
    // exp → relu should produce a single KernelDef with 1 param.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[4]),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[4]),
    ]);
    let chain: Vec<_> = graph.nodes()[1..=2].to_vec();
    let (kernel, ext_ids) = build_fused_scalar_kernel(&chain, &graph).unwrap();

    assert_eq!(kernel.params.len(), 1, "exp→relu has 1 external input");
    assert_eq!(ext_ids.len(), 1);
    assert!(kernel.name.starts_with("fused_"));
    assert!(kernel.name.contains("x2"));
    // Validate: nodes should be Param, UnaryFn(Exp), Literal(0), MinMax(Max)
    assert!(
        kernel.nodes.len() >= 4,
        "exp + relu(=max(x,0)) needs ≥4 nodes"
    );
}

#[test]
fn test_compose_three_unary_ops() {
    // exp → sqrt → tanh: 1 param, 3 unary ops inlined.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[4]),
        unary_node(2, "sqrt_0", TraceOp::Sqrt, 1, &[4]),
        unary_node(3, "tanh_0", TraceOp::Tanh, 2, &[4]),
    ]);
    let chain: Vec<_> = graph.nodes()[1..=3].to_vec();
    let (kernel, ext_ids) = build_fused_scalar_kernel(&chain, &graph).unwrap();

    assert_eq!(kernel.params.len(), 1);
    assert_eq!(ext_ids.len(), 1);
    assert!(kernel.name.contains("x3"));
    // 1 Param + 3 UnaryFn = 4 nodes minimum
    assert!(kernel.nodes.len() >= 4);
}

#[test]
fn test_compose_binary_then_unary() {
    // add(x, y) → relu: 2 params (x, y from external inputs).
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        binary_node(2, "add_0", TraceOp::Add, 0, 1, &[4]),
        unary_node(3, "relu_0", TraceOp::Relu, 2, &[4]),
    ]);
    let chain: Vec<_> = graph.nodes()[2..=3].to_vec();
    let (kernel, ext_ids) = build_fused_scalar_kernel(&chain, &graph).unwrap();

    assert_eq!(
        kernel.params.len(),
        2,
        "add(x,y)→relu has 2 external inputs"
    );
    assert_eq!(ext_ids.len(), 2);
}

#[test]
fn test_compose_neg_no_weight_data() {
    // neg → relu: neg uses Literal(0.0), not a weight tensor.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "neg_0", TraceOp::Neg, 0, &[4]),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[4]),
    ]);
    let chain: Vec<_> = graph.nodes()[1..=2].to_vec();
    let (kernel, ext_ids) = build_fused_scalar_kernel(&chain, &graph).unwrap();

    assert_eq!(kernel.params.len(), 1, "neg→relu has 1 external input");
    assert_eq!(ext_ids.len(), 1);
    // Neg = Literal(0) + Sub, Relu = Literal(0) + MinMax(Max)
    // Param(0), Literal(0), Sub, Literal(0), MinMax(Max) = 5 nodes
    assert!(kernel.nodes.len() >= 5);
}

#[test]
fn test_compose_sigmoid() {
    // exp → sigmoid: sigmoid decomposes to 1/(1+exp(-x)) in scalar IR.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[4]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 1, &[4]),
    ]);
    let chain: Vec<_> = graph.nodes()[1..=2].to_vec();
    let (kernel, ext_ids) = build_fused_scalar_kernel(&chain, &graph).unwrap();

    assert_eq!(kernel.params.len(), 1);
    assert_eq!(ext_ids.len(), 1);
    // exp(1) + sigmoid(~6 nodes) = many nodes
    assert!(kernel.nodes.len() >= 7);
}

#[test]
fn test_compose_silu() {
    // exp → silu: silu = x * sigmoid(x), many scalar IR nodes.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[8]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[8]),
        unary_node(2, "silu_0", TraceOp::Silu, 1, &[8]),
    ]);
    let chain: Vec<_> = graph.nodes()[1..=2].to_vec();
    let (kernel, ext_ids) = build_fused_scalar_kernel(&chain, &graph).unwrap();

    assert_eq!(kernel.params.len(), 1);
    assert_eq!(ext_ids.len(), 1);
}

#[test]
fn test_compose_gelu() {
    // relu → gelu: gelu approximation has ~12 scalar IR nodes.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
        unary_node(2, "gelu_0", TraceOp::Gelu, 1, &[4]),
    ]);
    let chain: Vec<_> = graph.nodes()[1..=2].to_vec();
    let (kernel, ext_ids) = build_fused_scalar_kernel(&chain, &graph).unwrap();

    assert_eq!(kernel.params.len(), 1);
    assert_eq!(ext_ids.len(), 1);
    // relu(2 nodes) + gelu(~12 nodes) + param(1) = many nodes
    assert!(kernel.nodes.len() >= 14);
}

#[test]
fn test_compose_chain_with_external_mid_input() {
    // exp(x) → add(exp_out, y) → relu: y is external, referenced mid-chain.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        unary_node(2, "exp_0", TraceOp::Exp, 0, &[4]),
        binary_node(3, "add_0", TraceOp::Add, 2, 1, &[4]),
        unary_node(4, "relu_0", TraceOp::Relu, 3, &[4]),
    ]);
    let chain: Vec<_> = graph.nodes()[2..=4].to_vec();
    let (kernel, ext_ids) = build_fused_scalar_kernel(&chain, &graph).unwrap();

    assert_eq!(
        kernel.params.len(),
        2,
        "exp→add(_, y)→relu has 2 external inputs"
    );
    assert_eq!(ext_ids.len(), 2);
    assert_eq!(ext_ids[0], 0, "first external is input_0 (feeds exp)");
    assert_eq!(ext_ids[1], 1, "second external is input_1 (feeds add)");
}

#[test]
fn test_compose_powf_odd_integer_uses_abs_and_sign_select() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "pow_0", TraceOp::Powf { exponent: 3.0 }, 0, &[4]),
    ]);
    let chain: Vec<_> = graph.nodes()[1..=1].to_vec();
    let (kernel, ext_ids) = build_fused_scalar_kernel(&chain, &graph).unwrap();

    assert_eq!(kernel.params.len(), 1);
    assert_eq!(ext_ids.len(), 1);
    assert!(kernel.nodes.iter().any(|n| matches!(
        n.kind,
        IRNodeKind::UnaryFn {
            op: UnaryFnKind::Abs,
            ..
        }
    )));
    assert!(kernel.nodes.iter().any(|n| matches!(
        n.kind,
        IRNodeKind::Compare {
            op: CompareOpKind::Lt,
            ..
        }
    )));
    assert!(kernel
        .nodes
        .iter()
        .any(|n| matches!(n.kind, IRNodeKind::Select { .. })));
}

#[test]
fn test_compose_powf_fractional_uses_nan_select() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "pow_0", TraceOp::Powf { exponent: 1.5 }, 0, &[4]),
    ]);
    let chain: Vec<_> = graph.nodes()[1..=1].to_vec();
    let (kernel, _ext_ids) = build_fused_scalar_kernel(&chain, &graph).unwrap();

    assert!(kernel
        .nodes
        .iter()
        .any(|n| matches!(n.kind, IRNodeKind::Select { .. })));
    assert!(kernel
        .nodes
        .iter()
        .any(|n| matches!(n.kind, IRNodeKind::Literal(v) if (v + 1.0).abs() < f64::EPSILON)));
    assert!(kernel.nodes.iter().any(|n| matches!(
        n.kind,
        IRNodeKind::UnaryFn {
            op: UnaryFnKind::Log,
            ..
        }
    )));
}

#[test]
fn test_composed_kernel_validates() {
    // All fusible ops should produce a kernel that passes validate().
    let all_unary_ops = vec![
        TraceOp::Exp,
        TraceOp::Log,
        TraceOp::Sqrt,
        TraceOp::Sqr,
        TraceOp::Abs,
        TraceOp::Neg,
        TraceOp::Recip,
        TraceOp::Sin,
        TraceOp::Cos,
        TraceOp::Floor,
        TraceOp::Round,
        TraceOp::Fract,
        TraceOp::Tanh,
        TraceOp::Relu,
        TraceOp::Gelu,
        TraceOp::GeluErf,
        TraceOp::Sigmoid,
        TraceOp::Silu,
    ];
    for (i, op) in all_unary_ops.into_iter().enumerate() {
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[4]),
            unary_node(1, &format!("op_{i}"), op.clone(), 0, &[4]),
            unary_node(2, "relu_tail", TraceOp::Relu, 1, &[4]),
        ]);
        let chain: Vec<_> = graph.nodes()[1..=2].to_vec();
        let result = build_fused_scalar_kernel(&chain, &graph);
        assert!(
            result.is_ok(),
            "scalar composition failed for unary op index {i}: {:?}",
            result.err()
        );
    }
}

#[test]
fn test_compose_all_binary_ops() {
    let binary_ops = vec![
        TraceOp::Add,
        TraceOp::Sub,
        TraceOp::Mul,
        TraceOp::Div,
        TraceOp::Maximum,
        TraceOp::Minimum,
    ];
    for (i, op) in binary_ops.into_iter().enumerate() {
        let graph = ComputationGraph::from_nodes(vec![
            input_node(0, &[4]),
            input_node(1, &[4]),
            binary_node(2, &format!("op_{i}"), op, 0, 1, &[4]),
            unary_node(3, "relu_tail", TraceOp::Relu, 2, &[4]),
        ]);
        let chain: Vec<_> = graph.nodes()[2..=3].to_vec();
        let result = build_fused_scalar_kernel(&chain, &graph);
        assert!(
            result.is_ok(),
            "scalar composition failed for binary op index {i}: {:?}",
            result.err()
        );
    }
}

#[test]
fn test_compose_gelu_erf() {
    // GeluErf decomposes to A&S 7.1.26 polynomial (~30 IR nodes).
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
        unary_node(2, "gelu_erf_0", TraceOp::GeluErf, 1, &[4]),
    ]);
    let chain: Vec<_> = graph.nodes()[1..=2].to_vec();
    let (kernel, ext_ids) = build_fused_scalar_kernel(&chain, &graph).unwrap();

    assert_eq!(kernel.params.len(), 1);
    assert_eq!(ext_ids.len(), 1);
    // relu(2 nodes) + gelu_erf(~30 nodes: erf polynomial + gelu wrapper) + param(1)
    assert!(
        kernel.nodes.len() >= 30,
        "gelu_erf decomposition should produce ≥30 IR nodes, got {}",
        kernel.nodes.len()
    );
}

#[test]
fn test_compose_gelu_erf_then_mul() {
    // gelu_erf → mul: both should compose into one kernel with 2 params.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        unary_node(2, "gelu_erf_0", TraceOp::GeluErf, 0, &[4]),
        binary_node(3, "mul_0", TraceOp::Mul, 2, 1, &[4]),
    ]);
    let chain: Vec<_> = graph.nodes()[2..=3].to_vec();
    let (kernel, ext_ids) = build_fused_scalar_kernel(&chain, &graph).unwrap();

    assert_eq!(kernel.params.len(), 2, "gelu_erf→mul has 2 external inputs");
    assert_eq!(ext_ids.len(), 2);
    assert_eq!(ext_ids[0], 0, "first external is input_0 (feeds gelu_erf)");
    assert_eq!(ext_ids[1], 1, "second external is input_1 (feeds mul)");
}
