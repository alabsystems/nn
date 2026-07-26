// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for fusion chain detection: elementwise chains, normalization+affine
//! patterns, residual patterns, activation fusions, and chain-breaking boundaries.
//!
//! Part of #4186.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use crate::trace_compile::{
    compile_trace_to_plan, compile_trace_to_plan_configured, count_dispatches,
    detect_fusion_chains, PeepholeConfig,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

/// Build a 2-element elementwise chain: input -> op_a -> op_b.
fn two_op_chain(op_a: TraceOp, name_a: &str, op_b: TraceOp, name_b: &str) -> ComputationGraph {
    ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 256]),
        test_node(1, name_a, op_a, vec![0], vec![1, 256]),
        test_node(2, name_b, op_b, vec![1], vec![1, 256]),
    ])
}

/// Build a 3-element elementwise chain: input -> op_a -> op_b -> op_c.
fn three_op_chain(
    op_a: TraceOp,
    name_a: &str,
    op_b: TraceOp,
    name_b: &str,
    op_c: TraceOp,
    name_c: &str,
) -> ComputationGraph {
    ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 256]),
        test_node(1, name_a, op_a, vec![0], vec![1, 256]),
        test_node(2, name_b, op_b, vec![1], vec![1, 256]),
        test_node(3, name_c, op_c, vec![2], vec![1, 256]),
    ])
}

// ===========================================================================
// Section 1: Elementwise chain detection
// ===========================================================================

#[test]
fn test_detect_chain_mul_add() {
    // input -> Mul(scalar) -> Add(scalar): binary ops may or may not chain.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 256]),
        input_node(1, vec![1, 256]),
        test_node(2, "mul", TraceOp::Mul, vec![0, 1], vec![1, 256]),
        input_node(3, vec![1, 256]),
        test_node(4, "add", TraceOp::Add, vec![2, 3], vec![1, 256]),
    ]);
    let chains = detect_fusion_chains(&graph).unwrap();
    // Key invariant: function does not panic and all chains have length >= 2.
    assert!(
        chains.iter().all(|c| c.chain_len >= 2),
        "all detected chains should have length >= 2"
    );
}

#[test]
fn test_detect_chain_relu_mul() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 256]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![1, 256]),
        input_node(2, vec![1, 256]),
        test_node(3, "mul", TraceOp::Mul, vec![1, 2], vec![1, 256]),
    ]);
    let chains = detect_fusion_chains(&graph).unwrap();
    for chain in &chains {
        assert!(chain.chain_len >= 2);
    }
}

#[test]
fn test_detect_chain_exp_log() {
    let graph = two_op_chain(TraceOp::Exp, "exp", TraceOp::Log, "log");
    let chains = detect_fusion_chains(&graph).unwrap();
    assert!(
        !chains.is_empty(),
        "exp -> log should be detected as a fusible chain"
    );
    assert_eq!(chains[0].chain_len, 2);
}

#[test]
fn test_detect_chain_unary_sequence_sin_cos_exp() {
    let graph = three_op_chain(
        TraceOp::Sin,
        "sin",
        TraceOp::Cos,
        "cos",
        TraceOp::Exp,
        "exp",
    );
    let chains = detect_fusion_chains(&graph).unwrap();
    assert!(!chains.is_empty(), "sin -> cos -> exp should chain");
    assert_eq!(chains[0].chain_len, 3);
}

#[test]
fn test_detect_chain_gelu_sigmoid() {
    let graph = two_op_chain(TraceOp::Gelu, "gelu", TraceOp::Sigmoid, "sigmoid");
    let chains = detect_fusion_chains(&graph).unwrap();
    assert!(!chains.is_empty());
    assert_eq!(chains[0].chain_len, 2);
}

#[test]
fn test_detect_chain_sqrt_recip() {
    let graph = two_op_chain(TraceOp::Sqrt, "sqrt", TraceOp::Recip, "recip");
    let chains = detect_fusion_chains(&graph).unwrap();
    assert!(!chains.is_empty(), "sqrt -> recip should be fusible");
    assert_eq!(chains[0].chain_len, 2);
}

#[test]
fn test_detect_chain_neg_abs_exp() {
    let graph = three_op_chain(
        TraceOp::Neg,
        "neg",
        TraceOp::Abs,
        "abs",
        TraceOp::Exp,
        "exp",
    );
    let chains = detect_fusion_chains(&graph).unwrap();
    assert!(!chains.is_empty());
    assert_eq!(chains[0].chain_len, 3);
}

// ===========================================================================
// Section 2: Non-fusible boundaries break chains
// ===========================================================================

#[test]
fn test_no_chain_for_single_op() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 256]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![1, 256]),
    ]);
    let chains = detect_fusion_chains(&graph).unwrap();
    assert!(chains.is_empty(), "single op should not form a chain");
}

#[test]
fn test_no_chain_for_matmul_boundary() {
    // MatMul is NOT fusible elementwise -- should break any chain.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![4, 768]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![4, 768]),
        input_node(2, vec![768, 3072]),
        test_node(3, "matmul", TraceOp::MatMul, vec![1, 2], vec![4, 3072]),
        test_node(4, "gelu", TraceOp::Gelu, vec![3], vec![4, 3072]),
    ]);
    let chains = detect_fusion_chains(&graph).unwrap();
    // MatMul should break the chain. No chain should span across it.
    for chain in &chains {
        let has_matmul = chain.chain_name.contains("matmul");
        assert!(!has_matmul, "MatMul should not appear in a fusion chain");
    }
}

#[test]
fn test_no_chain_for_softmax_boundary() {
    // Softmax is a reduction -- not fusible elementwise.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![4, 768]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![4, 768]),
        test_node(
            2,
            "softmax",
            TraceOp::Softmax { dim: 1 },
            vec![1],
            vec![4, 768],
        ),
        test_node(3, "exp", TraceOp::Exp, vec![2], vec![4, 768]),
    ]);
    let chains = detect_fusion_chains(&graph).unwrap();
    // No chain should span across softmax.
    for chain in &chains {
        assert!(
            chain.chain_len <= 2,
            "no chain should span across softmax reduction"
        );
    }
}

#[test]
fn test_chain_breaks_on_fanout() {
    // When an op has 2 consumers, the chain should not form.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 256]),
        test_node(1, "exp", TraceOp::Exp, vec![0], vec![1, 256]),
        // Node 1 has 2 consumers -> fan-out breaks chain.
        test_node(2, "log", TraceOp::Log, vec![1], vec![1, 256]),
        test_node(3, "neg", TraceOp::Neg, vec![1], vec![1, 256]),
    ]);
    let chains = detect_fusion_chains(&graph).unwrap();
    assert!(
        chains.is_empty(),
        "fan-out > 1 should prevent chain formation"
    );
}

#[test]
fn test_detect_empty_graph_no_chains() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let chains = detect_fusion_chains(&graph).unwrap();
    assert!(chains.is_empty());
}

#[test]
fn test_detect_input_only_no_chains() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 256]),
        input_node(1, vec![1, 256]),
    ]);
    let chains = detect_fusion_chains(&graph).unwrap();
    assert!(chains.is_empty());
}

// ===========================================================================
// Section 3: Compile-level fusion detection via public API
// ===========================================================================

#[test]
fn test_compile_empty_graph() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let plan = compile_trace_to_plan(&graph).unwrap();
    assert_eq!(count_dispatches(&plan), 0);
}

#[test]
fn test_compile_all_passes_disabled() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let config = PeepholeConfig {
        norm_activ_conv1d: false,
        fused_resblock: false,
        linear_activation: false,
        add_layer_norm: false,
        norm_linear: false,
        attention_transpose: false,
        flip_lstm: false,
        batched_linear_projection: false,
        channels_first_layer_norm: false,
        silu_mul: false,
        auto_fuse_elementwise: false,
        bilstm_cat: false,
        add_norm_linear: false,
        fuse_adain_snake: false,
        fuse_upsample_conv1d: false,
        fuse_instance_norm_mul_add: false,
        fuse_conv1d_activation: false,
        fuse_snake_instance_norm: false,
        fuse_conv1d_snake_norm: false,
        fuse_conv1d_snake_norm_resblock: false,
        fuse_add_instance_norm_conv1x1: false,
        fuse_conv_transpose1d_activation: false,
        norm_activ_conv_transpose1d: false,
        fuse_instance_norm_conv1d: false,
        fuse_conv1d_instance_norm: false,
        fuse_linear_layer_norm: false,
        fuse_resblock_chain: false,
        fuse_activation_conv1d: false,
    };
    let plan = compile_trace_to_plan_configured(&graph, &config).unwrap();
    assert_eq!(count_dispatches(&plan), 0);
}

#[test]
fn test_compile_single_pass_disabled() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let config = PeepholeConfig {
        linear_activation: false,
        ..Default::default()
    };
    let plan = compile_trace_to_plan_configured(&graph, &config).unwrap();
    assert_eq!(count_dispatches(&plan), 0);
}

#[test]
fn test_compile_with_several_passes_disabled() {
    // Verify that disabling selected passes still compiles on empty graph.
    let graph = ComputationGraph::from_nodes(vec![]);

    let config = PeepholeConfig {
        norm_activ_conv1d: false,
        silu_mul: false,
        bilstm_cat: false,
        ..Default::default()
    };
    let plan = compile_trace_to_plan_configured(&graph, &config).unwrap();
    assert_eq!(count_dispatches(&plan), 0);

    let config2 = PeepholeConfig {
        attention_transpose: false,
        auto_fuse_elementwise: false,
        ..Default::default()
    };
    let plan2 = compile_trace_to_plan_configured(&graph, &config2).unwrap();
    assert_eq!(count_dispatches(&plan2), 0);
}

// ===========================================================================
// Section 4: Chain detection properties
// ===========================================================================

#[test]
fn test_all_chains_have_length_at_least_2() {
    let graph = three_op_chain(
        TraceOp::Exp,
        "exp",
        TraceOp::Log,
        "log",
        TraceOp::Neg,
        "neg",
    );
    let chains = detect_fusion_chains(&graph).unwrap();
    for chain in &chains {
        assert!(
            chain.chain_len >= 2,
            "chain '{}' has length {} < 2",
            chain.chain_name,
            chain.chain_len,
        );
    }
}

#[test]
fn test_chain_name_contains_op_name() {
    let graph = two_op_chain(TraceOp::Exp, "exp", TraceOp::Log, "log");
    let chains = detect_fusion_chains(&graph).unwrap();
    assert!(!chains.is_empty());
    // Chain name should reference the first op.
    assert!(
        chains[0].chain_name.contains("exp") || chains[0].chain_name.contains("fused"),
        "chain name '{}' should reference op or 'fused'",
        chains[0].chain_name,
    );
}

#[test]
fn test_chain_pairs_count_is_chain_len_minus_1() {
    let graph = three_op_chain(
        TraceOp::Tanh,
        "tanh",
        TraceOp::Sigmoid,
        "sigmoid",
        TraceOp::Exp,
        "exp",
    );
    let chains = detect_fusion_chains(&graph).unwrap();
    for chain in &chains {
        assert_eq!(
            chain.pairs.len(),
            chain.chain_len - 1,
            "chain '{}': pairs.len() ({}) should be chain_len ({}) - 1",
            chain.chain_name,
            chain.pairs.len(),
            chain.chain_len,
        );
    }
}

#[test]
fn test_detect_chain_with_leaky_relu() {
    let graph = two_op_chain(
        TraceOp::LeakyRelu { slope: 0.2 },
        "leaky_relu",
        TraceOp::Exp,
        "exp",
    );
    let chains = detect_fusion_chains(&graph).unwrap();
    assert!(!chains.is_empty(), "leaky_relu -> exp should be fusible");
}
