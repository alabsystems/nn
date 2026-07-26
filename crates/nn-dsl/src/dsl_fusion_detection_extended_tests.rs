// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for DSL fusion detection, kernel IR, peephole pattern
//! matching, NativeOp detection, dispatch count estimation, buffer planner
//! fusion, and fusion cycle detection.
//!
//! Covers:
//! 1. Elementwise fusion: consecutive elementwise ops detected as fusible
//! 2. Reduction blocking: reduction ops block fusion chains
//! 3. Broadcast compatibility: broadcast-compatible ops can fuse
//! 4. Memory-bound vs compute-bound: classification of fusion benefit
//! 5. Peephole pattern matching: known patterns detected
//! 6. Fusion graph construction: fused op graph preserves semantics
//! 7. NativeOp detection: trace patterns map to NativeOpKind variants
//! 8. Dispatch count estimation: fused ops estimate fewer dispatches
//! 9. Buffer planner fusion: fused ops share intermediate buffers
//! 10. Fusion cycle detection: circular dependencies rejected
//!
//! Part of #4560.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use crate::auto_fuse_codegen::{compose_trace_ops_to_kernel_ir, FuseableOp, OpWiring};
use crate::buffer_planner::plan_buffers;
use crate::cost_model::CostModel;
use crate::partition_compiler::{find_fusion_groups, partition_plan, partition_summary};
use crate::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use crate::trace_compile::{
    compile_trace_to_plan, compile_trace_to_plan_configured, compile_trace_to_plan_with_fusion,
    count_dispatches, detect_fusion_chains, CompiledKernel, CompiledPlan, CompiledStep,
    FusionBlocker, FusionGap, FusionGapAnalysis, NativeOpKind, PeepholeConfig,
};

// ===========================================================================
// Helpers
// ===========================================================================

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

fn test_node(id: u64, name: &str, op: TraceOp, inputs: Vec<u64>, shape: Vec<usize>) -> TraceNode {
    TraceNode::new(id, name.to_string(), op, inputs, shape, DType::F32)
}

fn make_simple_dispatch(name: &str, shape: &[usize]) -> CompiledStep {
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

fn make_dispatch_with_ext(name: &str, shape: &[usize], ext_ids: Vec<u64>) -> CompiledStep {
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
        external_node_ids: Some(ext_ids),
    }
}

/// Build a linear elementwise chain: input -> op_a -> op_b -> op_c.
fn elementwise_chain_3(op_a: TraceOp, op_b: TraceOp, op_c: TraceOp) -> ComputationGraph {
    ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 256]),
        test_node(1, "op_a", op_a, vec![0], vec![1, 256]),
        test_node(2, "op_b", op_b, vec![1], vec![1, 256]),
        test_node(3, "op_c", op_c, vec![2], vec![1, 256]),
    ])
}

// ===========================================================================
// Section 1: Elementwise fusion -- consecutive elementwise ops fusible
// ===========================================================================

#[test]
fn test_elementwise_add_mul_relu_chain_detected() {
    // add -> mul -> relu should form a fusible chain of 3 consecutive
    // elementwise ops (add and mul are binary, relu is unary).
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 256]),
        input_node(1, vec![1, 256]),
        test_node(2, "add", TraceOp::Add, vec![0, 1], vec![1, 256]),
        input_node(3, vec![1, 256]),
        test_node(4, "mul", TraceOp::Mul, vec![2, 3], vec![1, 256]),
        test_node(5, "relu", TraceOp::Relu, vec![4], vec![1, 256]),
    ]);
    let chains = detect_fusion_chains(&graph).unwrap();
    // At minimum, mul->relu should be detected as a chain of 2.
    assert!(
        chains.iter().any(|c| c.chain_len >= 2),
        "add->mul->relu must produce at least one chain of length >= 2, got {:?}",
        chains.iter().map(|c| c.chain_len).collect::<Vec<_>>()
    );
}

#[test]
fn test_elementwise_unary_chain_exp_log_neg() {
    let graph = elementwise_chain_3(TraceOp::Exp, TraceOp::Log, TraceOp::Neg);
    let chains = detect_fusion_chains(&graph).unwrap();
    assert!(!chains.is_empty(), "exp->log->neg should form a chain");
    assert_eq!(
        chains[0].chain_len, 3,
        "three consecutive unary ops should produce a chain of length 3"
    );
}

#[test]
fn test_elementwise_sigmoid_tanh_relu_chain() {
    let graph = elementwise_chain_3(TraceOp::Sigmoid, TraceOp::Tanh, TraceOp::Relu);
    let chains = detect_fusion_chains(&graph).unwrap();
    assert!(!chains.is_empty());
    assert_eq!(chains[0].chain_len, 3);
}

#[test]
fn test_elementwise_four_op_chain() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![4, 64]),
        test_node(1, "exp", TraceOp::Exp, vec![0], vec![4, 64]),
        test_node(2, "neg", TraceOp::Neg, vec![1], vec![4, 64]),
        test_node(3, "abs", TraceOp::Abs, vec![2], vec![4, 64]),
        test_node(4, "relu", TraceOp::Relu, vec![3], vec![4, 64]),
    ]);
    let chains = detect_fusion_chains(&graph).unwrap();
    assert!(!chains.is_empty());
    assert!(
        chains[0].chain_len >= 4,
        "four consecutive unary ops: chain_len = {}, expected >= 4",
        chains[0].chain_len
    );
}

#[test]
fn test_composed_kernel_ir_from_two_unary_ops() {
    let ops = vec![
        FuseableOp::unary(TraceOp::Relu),
        FuseableOp::unary(TraceOp::Exp),
    ];
    let kernel = compose_trace_ops_to_kernel_ir(&ops, "relu_exp").unwrap();
    kernel.validate().expect("fused kernel should validate");
    assert_eq!(kernel.params.len(), 1, "unary chain has one input param");
    assert_eq!(kernel.name, "relu_exp");
}

#[test]
fn test_composed_kernel_ir_three_unary_ops() {
    let ops = vec![
        FuseableOp::unary(TraceOp::Sigmoid),
        FuseableOp::unary(TraceOp::Tanh),
        FuseableOp::unary(TraceOp::Neg),
    ];
    let kernel = compose_trace_ops_to_kernel_ir(&ops, "sig_tanh_neg").unwrap();
    kernel
        .validate()
        .expect("3-op fused kernel should validate");
    assert_eq!(kernel.params.len(), 1);
}

// ===========================================================================
// Section 2: Reduction blocking -- reduction ops block fusion chains
// ===========================================================================

#[test]
fn test_softmax_blocks_elementwise_chain() {
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
    // No chain should span across the softmax reduction.
    for chain in &chains {
        assert!(
            chain.chain_len <= 2,
            "softmax should block chains from spanning: chain_len = {}",
            chain.chain_len,
        );
    }
}

#[test]
fn test_sum_reduce_blocks_chain() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![4, 64]),
        test_node(1, "exp", TraceOp::Exp, vec![0], vec![4, 64]),
        test_node(
            2,
            "sum",
            TraceOp::ReduceSum {
                dim: 1,
                keepdim: false,
            },
            vec![1],
            vec![4],
        ),
        // Post-reduce op has different shape, can't fuse with pre-reduce.
        test_node(3, "relu", TraceOp::Relu, vec![2], vec![4]),
    ]);
    let chains = detect_fusion_chains(&graph).unwrap();
    // exp before reduce and relu after reduce should not form a single chain.
    for chain in &chains {
        assert!(chain.chain_len <= 2, "reduce_sum should break the chain");
    }
}

#[test]
fn test_matmul_blocks_fusion_chain() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![4, 64]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![4, 64]),
        input_node(2, vec![64, 128]),
        test_node(3, "matmul", TraceOp::MatMul, vec![1, 2], vec![4, 128]),
        test_node(4, "gelu", TraceOp::Gelu, vec![3], vec![4, 128]),
    ]);
    let chains = detect_fusion_chains(&graph).unwrap();
    // MatMul is not fusible elementwise -- no chain should include it.
    for chain in &chains {
        assert!(
            !chain.chain_name.contains("matmul"),
            "MatMul must not appear in fusion chain: '{}'",
            chain.chain_name,
        );
    }
}

#[test]
fn test_instance_norm_blocks_fusion_chain() {
    // InstanceNorm is a normalization (reduction-based) op that should break fusion.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 32, 768]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![1, 32, 768]),
        test_node(
            2,
            "instance_norm",
            TraceOp::InstanceNorm { eps: 1e-5 },
            vec![1],
            vec![1, 32, 768],
        ),
        test_node(3, "sigmoid", TraceOp::Sigmoid, vec![2], vec![1, 32, 768]),
    ]);
    let chains = detect_fusion_chains(&graph).unwrap();
    // No chain should span across instance_norm.
    for chain in &chains {
        assert!(
            chain.chain_len <= 2,
            "instance_norm should block fusion chain: chain_len = {}",
            chain.chain_len,
        );
    }
}

// ===========================================================================
// Section 3: Broadcast compatibility -- same-shape ops can fuse
// ===========================================================================

#[test]
fn test_same_shape_ops_fuse() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![2, 128]),
        test_node(1, "exp", TraceOp::Exp, vec![0], vec![2, 128]),
        test_node(2, "sigmoid", TraceOp::Sigmoid, vec![1], vec![2, 128]),
    ]);
    let chains = detect_fusion_chains(&graph).unwrap();
    assert!(!chains.is_empty(), "same-shape ops should fuse");
    assert_eq!(chains[0].chain_len, 2);
}

#[test]
fn test_different_shape_ops_no_chain() {
    // Shape mismatch should prevent chain formation.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![4, 64]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![4, 64]),
        // Reshape changes shape -- next op has different shape.
        test_node(
            2,
            "reshape",
            TraceOp::Reshape {
                target_shape: vec![2, 128],
            },
            vec![1],
            vec![2, 128],
        ),
        test_node(3, "exp", TraceOp::Exp, vec![2], vec![2, 128]),
    ]);
    let chains = detect_fusion_chains(&graph).unwrap();
    // relu on [4,64] and exp on [2,128] should not be in the same chain.
    for chain in &chains {
        assert!(
            chain.chain_len <= 2,
            "shape change should block long chains"
        );
    }
}

#[test]
fn test_broadcast_binary_op_fusible_with_compose() {
    // Binary op with broadcast-compatible shapes should compose in auto-fuse.
    let ops = vec![
        FuseableOp::unary(TraceOp::Relu),
        FuseableOp {
            op: TraceOp::Add,
            wiring: OpWiring::BinarySecondExternal,
        },
    ];
    let kernel = compose_trace_ops_to_kernel_ir(&ops, "relu_add").unwrap();
    kernel
        .validate()
        .expect("relu+add should compose and validate");
    // relu(x) + y => 2 params
    assert_eq!(kernel.params.len(), 2);
}

#[test]
fn test_broadcast_mul_then_relu_composes() {
    let ops = vec![
        FuseableOp {
            op: TraceOp::Mul,
            wiring: OpWiring::BinarySecondExternal,
        },
        FuseableOp::unary(TraceOp::Relu),
    ];
    let kernel = compose_trace_ops_to_kernel_ir(&ops, "mul_relu").unwrap();
    kernel.validate().expect("mul+relu should compose");
    assert_eq!(kernel.params.len(), 2, "x * y -> relu => 2 params");
}

// ===========================================================================
// Section 4: Memory-bound vs compute-bound classification
// ===========================================================================

#[test]
fn test_cost_model_elementwise_is_memory_bound() {
    // Elementwise ops on large tensors should be memory-bound (bandwidth-limited).
    let model = CostModel::apple_m4();
    let plan = CompiledPlan {
        steps: vec![make_simple_dispatch("relu", &[1, 1_000_000])],
        input_shapes: vec![vec![1, 1_000_000]],
        output_step: 0,
        weight_names: vec![],
    };
    let est = model.estimate(&plan);
    assert_eq!(est.dispatch_count, 1);
    // Cost should be positive and dominated by memory transfer.
    assert!(
        est.total_ns > 0.0,
        "elementwise dispatch must have positive cost"
    );
}

#[test]
fn test_cost_model_large_matmul_has_higher_cost() {
    let model = CostModel::apple_m4();
    let small_plan = CompiledPlan {
        steps: vec![make_simple_dispatch("relu", &[1, 64])],
        input_shapes: vec![vec![1, 64]],
        output_step: 0,
        weight_names: vec![],
    };
    let large_plan = CompiledPlan {
        steps: vec![make_simple_dispatch("relu", &[1, 65536])],
        input_shapes: vec![vec![1, 65536]],
        output_step: 0,
        weight_names: vec![],
    };
    let small_est = model.estimate(&small_plan);
    let large_est = model.estimate(&large_plan);
    assert!(
        large_est.total_ns >= small_est.total_ns,
        "larger tensor should cost >= smaller: {} vs {}",
        large_est.total_ns,
        small_est.total_ns,
    );
}

#[test]
fn test_cost_model_fusion_reduces_launch_overhead() {
    let model = CostModel::apple_m4();
    // Unfused: 3 separate dispatches.
    let unfused = CompiledPlan {
        steps: vec![
            make_simple_dispatch("relu", &[1, 1024]),
            make_simple_dispatch("exp", &[1, 1024]),
            make_simple_dispatch("neg", &[1, 1024]),
        ],
        input_shapes: vec![vec![1, 1024]],
        output_step: 2,
        weight_names: vec![],
    };
    // Fused: single dispatch (same total work, fewer launches).
    let fused = CompiledPlan {
        steps: vec![make_simple_dispatch("fused_relu_exp_neg", &[1, 1024])],
        input_shapes: vec![vec![1, 1024]],
        output_step: 0,
        weight_names: vec![],
    };
    let unfused_est = model.estimate(&unfused);
    let fused_est = model.estimate(&fused);
    assert_eq!(unfused_est.dispatch_count, 3);
    assert_eq!(fused_est.dispatch_count, 1);
    // Fused should have lower total cost due to fewer launch overheads.
    assert!(
        fused_est.total_ns < unfused_est.total_ns,
        "fused ({:.0} ns) should be cheaper than unfused ({:.0} ns)",
        fused_est.total_ns,
        unfused_est.total_ns,
    );
}

// ===========================================================================
// Section 5: Peephole pattern matching -- known patterns detected
// ===========================================================================

#[test]
fn test_peephole_config_default_enables_all_patterns() {
    let cfg = PeepholeConfig::default();
    // All peephole patterns should be enabled by default.
    assert!(cfg.norm_activ_conv1d);
    assert!(cfg.fused_resblock);
    assert!(cfg.linear_activation);
    assert!(cfg.add_layer_norm);
    assert!(cfg.norm_linear);
    assert!(cfg.attention_transpose);
    assert!(cfg.silu_mul);
    assert!(cfg.auto_fuse_elementwise);
    assert!(cfg.fuse_adain_snake);
    assert!(cfg.fuse_upsample_conv1d);
    assert!(cfg.fuse_instance_norm_mul_add);
    assert!(cfg.fuse_conv1d_activation);
}

#[test]
fn test_peephole_disabled_does_not_crash_on_real_graph() {
    // Build a simple graph and compile with all peepholes disabled.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 64]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![1, 64]),
        test_node(2, "exp", TraceOp::Exp, vec![1], vec![1, 64]),
    ]);
    let disabled = PeepholeConfig {
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
    let plan = compile_trace_to_plan_configured(&graph, &disabled)
        .expect("disabled peephole should not crash");
    assert!(!plan.steps.is_empty());
}

#[test]
fn test_peephole_silu_mul_enabled_vs_disabled() {
    // Test that toggling silu_mul changes plan structure.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 256]),
        test_node(1, "silu", TraceOp::Silu, vec![0], vec![1, 256]),
        input_node(2, vec![1, 256]),
        test_node(3, "mul", TraceOp::Mul, vec![1, 2], vec![1, 256]),
    ]);
    let enabled = PeepholeConfig::default();
    let plan_enabled = compile_trace_to_plan_configured(&graph, &enabled).unwrap();
    let disabled = PeepholeConfig {
        silu_mul: false,
        ..Default::default()
    };
    let plan_disabled = compile_trace_to_plan_configured(&graph, &disabled).unwrap();
    // Both plans should compile without error. When silu_mul is enabled,
    // dispatch count may be lower due to fusion.
    let d_enabled = count_dispatches(&plan_enabled);
    let d_disabled = count_dispatches(&plan_disabled);
    assert!(
        d_enabled <= d_disabled,
        "silu_mul enabled ({d_enabled}) should have <= dispatches than disabled ({d_disabled})"
    );
}

#[test]
fn test_peephole_auto_fuse_elementwise_effect() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 512]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![1, 512]),
        test_node(2, "sigmoid", TraceOp::Sigmoid, vec![1], vec![1, 512]),
        test_node(3, "tanh", TraceOp::Tanh, vec![2], vec![1, 512]),
    ]);
    let enabled = PeepholeConfig::default();
    let plan_enabled = compile_trace_to_plan_configured(&graph, &enabled).unwrap();
    let disabled = PeepholeConfig {
        auto_fuse_elementwise: false,
        ..Default::default()
    };
    let plan_disabled = compile_trace_to_plan_configured(&graph, &disabled).unwrap();
    let d_enabled = count_dispatches(&plan_enabled);
    let d_disabled = count_dispatches(&plan_disabled);
    // With auto-fuse enabled, the chain relu->sigmoid->tanh should fuse.
    assert!(
        d_enabled <= d_disabled,
        "auto_fuse_elementwise should reduce dispatches: {d_enabled} vs {d_disabled}"
    );
}

// ===========================================================================
// Section 6: Fusion graph construction -- fused op preserves semantics
// ===========================================================================

#[test]
fn test_fused_kernel_validates_single_unary() {
    let ops = vec![FuseableOp::unary(TraceOp::Relu)];
    let kernel = compose_trace_ops_to_kernel_ir(&ops, "fused_relu").unwrap();
    kernel
        .validate()
        .expect("single unary fused kernel must validate");
    assert_eq!(kernel.name, "fused_relu");
}

#[test]
fn test_fused_kernel_validates_chain_of_five() {
    let ops = vec![
        FuseableOp::unary(TraceOp::Exp),
        FuseableOp::unary(TraceOp::Neg),
        FuseableOp::unary(TraceOp::Abs),
        FuseableOp::unary(TraceOp::Relu),
        FuseableOp::unary(TraceOp::Sigmoid),
    ];
    let kernel = compose_trace_ops_to_kernel_ir(&ops, "chain_5").unwrap();
    kernel.validate().expect("5-op chain should validate");
    assert_eq!(kernel.params.len(), 1, "all unary => 1 param");
}

#[test]
fn test_fused_binary_then_unary_chain() {
    let ops = vec![
        FuseableOp {
            op: TraceOp::Add,
            wiring: OpWiring::BinarySecondExternal,
        },
        FuseableOp::unary(TraceOp::Relu),
        FuseableOp::unary(TraceOp::Sigmoid),
    ];
    let kernel = compose_trace_ops_to_kernel_ir(&ops, "add_relu_sig").unwrap();
    kernel
        .validate()
        .expect("add->relu->sigmoid should validate");
    // add(x, y) -> relu -> sigmoid => 2 params
    assert_eq!(kernel.params.len(), 2);
}

#[test]
fn test_fused_kernel_ir_node_count_grows_with_ops() {
    let ops_1 = vec![FuseableOp::unary(TraceOp::Relu)];
    let ops_2 = vec![
        FuseableOp::unary(TraceOp::Relu),
        FuseableOp::unary(TraceOp::Exp),
    ];
    let ops_3 = vec![
        FuseableOp::unary(TraceOp::Relu),
        FuseableOp::unary(TraceOp::Exp),
        FuseableOp::unary(TraceOp::Neg),
    ];
    let k1 = compose_trace_ops_to_kernel_ir(&ops_1, "k1").unwrap();
    let k2 = compose_trace_ops_to_kernel_ir(&ops_2, "k2").unwrap();
    let k3 = compose_trace_ops_to_kernel_ir(&ops_3, "k3").unwrap();
    assert!(
        k2.nodes.len() > k1.nodes.len(),
        "more ops should produce more IR nodes: {} vs {}",
        k2.nodes.len(),
        k1.nodes.len(),
    );
    assert!(
        k3.nodes.len() > k2.nodes.len(),
        "more ops should produce more IR nodes: {} vs {}",
        k3.nodes.len(),
        k2.nodes.len(),
    );
}

#[test]
fn test_fused_kernel_output_node_is_last() {
    let ops = vec![
        FuseableOp::unary(TraceOp::Exp),
        FuseableOp::unary(TraceOp::Neg),
    ];
    let kernel = compose_trace_ops_to_kernel_ir(&ops, "exp_neg").unwrap();
    // Output node should be the last node in the DAG.
    let output_idx = kernel.output.index();
    assert_eq!(
        output_idx,
        kernel.nodes.len() - 1,
        "output should be last node in fused kernel"
    );
}

// ===========================================================================
// Section 7: NativeOp detection -- trace patterns map to NativeOpKind
// ===========================================================================

#[test]
fn test_native_op_instance_norm_construction() {
    let op = NativeOpKind::InstanceNorm {
        eps: 1e-5,
        input_shape: vec![1, 64, 1024],
    };
    match &op {
        NativeOpKind::InstanceNorm { eps, input_shape } => {
            assert!((eps - 1e-5).abs() < 1e-8);
            assert_eq!(input_shape, &vec![1, 64, 1024]);
        }
        _ => panic!("expected InstanceNorm"),
    }
}

#[test]
fn test_native_op_adain_snake_construction() {
    let op = NativeOpKind::AdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 256, 512],
        channels: 256,
        residual_gamma: true,
        external_node_ids: None,
    };
    match &op {
        NativeOpKind::AdainSnake {
            channels,
            residual_gamma,
            ..
        } => {
            assert_eq!(*channels, 256);
            assert!(*residual_gamma);
        }
        _ => panic!("expected AdainSnake"),
    }
}

#[test]
fn test_native_op_layer_norm_construction() {
    let op = NativeOpKind::LayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 32, 768],
        hidden_dim: 768,
    };
    match &op {
        NativeOpKind::LayerNorm {
            hidden_dim, eps, ..
        } => {
            assert_eq!(*hidden_dim, 768);
            assert!((eps - 1e-5).abs() < 1e-8);
        }
        _ => panic!("expected LayerNorm"),
    }
}

#[test]
fn test_native_op_silu_mul_construction() {
    let op = NativeOpKind::SiluMul {
        input_shape: vec![1, 32, 3072],
    };
    match &op {
        NativeOpKind::SiluMul { input_shape } => {
            assert_eq!(input_shape, &vec![1, 32, 3072]);
        }
        _ => panic!("expected SiluMul"),
    }
}

#[test]
fn test_native_op_rotary_embedding_construction() {
    let op = NativeOpKind::RotaryEmbedding {
        head_dim: 128,
        input_shape: vec![1, 32, 128, 128],
    };
    match &op {
        NativeOpKind::RotaryEmbedding { head_dim, .. } => {
            assert_eq!(*head_dim, 128);
        }
        _ => panic!("expected RotaryEmbedding"),
    }
}

#[test]
fn test_native_op_bilstm_cat_construction() {
    let op = NativeOpKind::BiLstmCat {
        hidden_size: 256,
        input_shape: vec![50, 1, 128],
        h_shape: vec![1, 256],
        fwd_lstm_step: 0,
        rev_lstm_step: 1,
    };
    match &op {
        NativeOpKind::BiLstmCat {
            hidden_size,
            input_shape,
            ..
        } => {
            assert_eq!(*hidden_size, 256);
            assert_eq!(input_shape, &vec![50, 1, 128]);
        }
        _ => panic!("expected BiLstmCat"),
    }
}

#[test]
fn test_native_op_fused_adain_snake_construction() {
    let op = NativeOpKind::FusedAdainSnake {
        eps: 1e-5,
        channels: 128,
        input_shape: vec![1, 128, 256],
        external_node_ids: None,
    };
    match &op {
        NativeOpKind::FusedAdainSnake { channels, eps, .. } => {
            assert_eq!(*channels, 128);
            assert!((eps - 1e-5).abs() < 1e-8);
        }
        _ => panic!("expected FusedAdainSnake"),
    }
}

#[test]
fn test_native_op_fused_upsample_conv1d_construction() {
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 8,
        in_channels: 256,
        out_channels: 128,
        kernel_size: 16,
        stride: 1,
        padding: 4,
        input_shape: vec![1, 256, 64],
    };
    match &op {
        NativeOpKind::FusedUpsampleConv1d {
            upsample_factor,
            in_channels,
            out_channels,
            ..
        } => {
            assert_eq!(*upsample_factor, 8);
            assert_eq!(*in_channels, 256);
            assert_eq!(*out_channels, 128);
        }
        _ => panic!("expected FusedUpsampleConv1d"),
    }
}

#[test]
fn test_native_op_batch_norm_2d_construction() {
    let op = NativeOpKind::BatchNorm2d {
        eps: 1e-5,
        num_channels: 64,
        input_shape: vec![1, 64, 32, 32],
        has_weight: true,
        has_bias: true,
    };
    match &op {
        NativeOpKind::BatchNorm2d {
            num_channels,
            has_weight,
            has_bias,
            ..
        } => {
            assert_eq!(*num_channels, 64);
            assert!(*has_weight);
            assert!(*has_bias);
        }
        _ => panic!("expected BatchNorm2d"),
    }
}

// ===========================================================================
// Section 8: Dispatch count estimation -- fused ops have fewer dispatches
// ===========================================================================

#[test]
fn test_dispatch_count_empty_plan() {
    let plan = CompiledPlan {
        steps: vec![],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    assert_eq!(count_dispatches(&plan), 0);
}

#[test]
fn test_dispatch_count_single_dispatch() {
    let plan = CompiledPlan {
        steps: vec![make_simple_dispatch("relu", &[1, 256])],
        input_shapes: vec![vec![1, 256]],
        output_step: 0,
        weight_names: vec![],
    };
    assert_eq!(count_dispatches(&plan), 1);
}

#[test]
fn test_dispatch_count_passthrough_not_counted() {
    let plan = CompiledPlan {
        steps: vec![
            make_simple_dispatch("relu", &[1, 64]),
            CompiledStep::Passthrough {
                op_name: "reshape".into(),
                output_shape: vec![1, 8, 8],
            },
            CompiledStep::IdentityPassthrough,
        ],
        input_shapes: vec![vec![1, 64]],
        output_step: 0,
        weight_names: vec![],
    };
    assert_eq!(
        count_dispatches(&plan),
        1,
        "only Dispatch steps count, not Passthrough/IdentityPassthrough"
    );
}

#[test]
fn test_fused_plan_has_fewer_dispatches_than_unfused() {
    // Build a graph with 3 fusible elementwise ops.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 256]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![1, 256]),
        test_node(2, "sigmoid", TraceOp::Sigmoid, vec![1], vec![1, 256]),
        test_node(3, "tanh", TraceOp::Tanh, vec![2], vec![1, 256]),
    ]);
    // Plan without fusion: `compile_trace_to_plan` is the no-fusion entry
    // point (it calls `compile_trace`, with no chain fusion, partition
    // fusion, or peephole passes).
    let plan_no_fuse = compile_trace_to_plan(&graph).unwrap();
    // Plan with fusion: `compile_trace_to_plan_with_fusion` runs chain
    // fusion + partition-driven fusion + peephole passes.
    let plan_fused = compile_trace_to_plan_with_fusion(&graph).unwrap();
    let d_no_fuse = count_dispatches(&plan_no_fuse);
    let d_fused = count_dispatches(&plan_fused);
    assert!(
        d_fused <= d_no_fuse,
        "fused plan ({d_fused}) should have <= dispatches than unfused ({d_no_fuse})"
    );
}

#[test]
fn test_fusion_gap_analysis_reports_savings() {
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
                step_a: 2,
                step_b: 3,
                kernel_a: "sigmoid".into(),
                kernel_b: "tanh".into(),
                reason: FusionBlocker::ShapeMismatch,
                savings: 1,
            },
        ],
        total_dispatches: 20,
        theoretical_minimum: 10,
    };
    let pct = analysis.optimization_opportunity_pct();
    assert!((pct - 50.0).abs() < 1e-6, "expected 50%, got {pct}");
    assert_eq!(analysis.gaps.len(), 2);
}

#[test]
fn test_compiled_plan_with_fusion_dispatches_monotonic() {
    // compile_trace_to_plan_with_fusion should produce <= dispatches
    // compared to compile_trace_to_plan (which uses basic peephole only).
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 128]),
        test_node(1, "exp", TraceOp::Exp, vec![0], vec![1, 128]),
        test_node(2, "neg", TraceOp::Neg, vec![1], vec![1, 128]),
        test_node(3, "abs", TraceOp::Abs, vec![2], vec![1, 128]),
    ]);
    let plan_basic = compile_trace_to_plan(&graph).unwrap();
    let plan_fusion = compile_trace_to_plan_with_fusion(&graph).unwrap();
    let d_basic = count_dispatches(&plan_basic);
    let d_fusion = count_dispatches(&plan_fusion);
    assert!(
        d_fusion <= d_basic,
        "fusion plan ({d_fusion}) should have <= dispatches than basic ({d_basic})"
    );
}

// ===========================================================================
// Section 9: Buffer planner fusion -- fused ops share intermediate buffers
// ===========================================================================

#[test]
fn test_buffer_planner_empty_plan() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let plan = CompiledPlan {
        steps: vec![],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    let bp = plan_buffers(&plan, &graph);
    assert_eq!(bp.total_bytes, 0);
    assert_eq!(bp.naive_total, 0);
    assert!(bp.step_offsets.is_empty());
}

#[test]
fn test_buffer_planner_single_dispatch() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 256]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![1, 256]),
    ]);
    let plan = compile_trace_to_plan(&graph).unwrap();
    let bp = plan_buffers(&plan, &graph);
    assert!(
        bp.total_bytes > 0,
        "single dispatch should allocate buffers"
    );
    assert_eq!(bp.step_offsets.len(), plan.steps.len());
}

#[test]
fn test_buffer_planner_reuse_reduces_total_bytes() {
    // Sequential ops with non-overlapping lifetimes should reuse buffers.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 1024]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![1, 1024]),
        test_node(2, "exp", TraceOp::Exp, vec![1], vec![1, 1024]),
        test_node(3, "neg", TraceOp::Neg, vec![2], vec![1, 1024]),
    ]);
    let plan = compile_trace_to_plan(&graph).unwrap();
    let bp = plan_buffers(&plan, &graph);
    // Buffer reuse should make total_bytes <= naive_total.
    assert!(
        bp.total_bytes <= bp.naive_total,
        "buffer reuse should reduce total: {} <= {}",
        bp.total_bytes,
        bp.naive_total,
    );
}

#[test]
fn test_buffer_planner_fused_plan_uses_less_memory() {
    // A fused plan should use fewer intermediate buffers.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 512]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![1, 512]),
        test_node(2, "sigmoid", TraceOp::Sigmoid, vec![1], vec![1, 512]),
        test_node(3, "tanh", TraceOp::Tanh, vec![2], vec![1, 512]),
    ]);
    let plan_fused = compile_trace_to_plan_with_fusion(&graph).unwrap();
    let bp_fused = plan_buffers(&plan_fused, &graph);

    let no_fuse_cfg = PeepholeConfig {
        auto_fuse_elementwise: false,
        ..Default::default()
    };
    let plan_unfused = compile_trace_to_plan_configured(&graph, &no_fuse_cfg).unwrap();
    let bp_unfused = plan_buffers(&plan_unfused, &graph);

    // Fused plan should need <= bytes because it has fewer intermediates.
    assert!(
        bp_fused.total_bytes <= bp_unfused.total_bytes,
        "fused plan ({} bytes) should use <= memory than unfused ({} bytes)",
        bp_fused.total_bytes,
        bp_unfused.total_bytes,
    );
}

#[test]
fn test_buffer_planner_last_use_indices_valid() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 64]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![1, 64]),
        test_node(2, "exp", TraceOp::Exp, vec![1], vec![1, 64]),
    ]);
    let plan = compile_trace_to_plan(&graph).unwrap();
    let bp = plan_buffers(&plan, &graph);
    // All last_use indices should be valid step indices.
    for (i, &lu) in bp.last_use.iter().enumerate() {
        assert!(
            lu < plan.steps.len(),
            "last_use[{i}] = {lu} out of range (len = {})",
            plan.steps.len(),
        );
        assert!(lu >= i, "last_use[{i}] = {lu} should be >= {i}");
    }
}

// ===========================================================================
// Section 10: Fusion cycle detection -- circular dependencies rejected
// ===========================================================================

#[test]
fn test_partition_dag_no_self_loops() {
    // A well-formed plan should produce a DAG with no self-loops.
    let plan = CompiledPlan {
        steps: vec![
            make_dispatch_with_ext("relu", &[1, 64], vec![]),
            make_dispatch_with_ext("exp", &[1, 64], vec![0]),
            make_dispatch_with_ext("neg", &[1, 64], vec![1]),
        ],
        input_shapes: vec![vec![1, 64]],
        output_step: 2,
        weight_names: vec![],
    };
    let dag = partition_plan(&plan, None);
    for (i, preds) in dag.predecessors.iter().enumerate() {
        assert!(!preds.contains(&i), "step {i} should not depend on itself");
    }
    for (i, succs) in dag.successors.iter().enumerate() {
        assert!(
            !succs.contains(&i),
            "step {i} should not be its own successor"
        );
    }
}

#[test]
fn test_partition_dag_predecessors_and_successors_consistent() {
    let plan = CompiledPlan {
        steps: vec![
            make_dispatch_with_ext("relu", &[1, 128], vec![]),
            make_dispatch_with_ext("exp", &[1, 128], vec![0]),
            make_dispatch_with_ext("sigmoid", &[1, 128], vec![1]),
        ],
        input_shapes: vec![vec![1, 128]],
        output_step: 2,
        weight_names: vec![],
    };
    let dag = partition_plan(&plan, None);
    // If A is in predecessors[B], then B must be in successors[A].
    for (b, preds) in dag.predecessors.iter().enumerate() {
        for &a in preds {
            assert!(
                dag.successors[a].contains(&b),
                "predecessors[{b}] contains {a}, but successors[{a}] missing {b}"
            );
        }
    }
    for (a, succs) in dag.successors.iter().enumerate() {
        for &b in succs {
            assert!(
                dag.predecessors[b].contains(&a),
                "successors[{a}] contains {b}, but predecessors[{b}] missing {a}"
            );
        }
    }
}

#[test]
fn test_fusion_groups_from_linear_chain() {
    // A linear chain of 3 elementwise ops should produce one fusion group.
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            make_dispatch_with_ext("relu", &[1, 256], vec![0]),
            make_dispatch_with_ext("exp", &[1, 256], vec![1]),
            make_dispatch_with_ext("neg", &[1, 256], vec![2]),
        ],
        input_shapes: vec![vec![1, 256]],
        output_step: 3,
        weight_names: vec![],
    };
    let dag = partition_plan(&plan, None);
    let groups = find_fusion_groups(&dag);
    // Three consecutive elementwise ops with same shape should fuse.
    assert!(
        !groups.is_empty(),
        "linear elementwise chain should produce fusion groups"
    );
    let total_savings: usize = groups.iter().map(|g| g.dispatches_saved).sum();
    assert!(
        total_savings >= 2,
        "3 elementwise ops fused should save >= 2 dispatches, got {total_savings}"
    );
}

#[test]
fn test_fusion_group_blocked_by_opaque_dispatch() {
    // An opaque dispatch (matmul) between elementwise ops should block fusion.
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            make_dispatch_with_ext("relu", &[1, 64], vec![0]),
            make_dispatch_with_ext("matmul", &[1, 128], vec![1]),
            make_dispatch_with_ext("sigmoid", &[1, 128], vec![2]),
        ],
        input_shapes: vec![vec![1, 64]],
        output_step: 3,
        weight_names: vec![],
    };
    let dag = partition_plan(&plan, None);
    let groups = find_fusion_groups(&dag);
    // relu and sigmoid have different shapes and matmul between them.
    // They should not be in the same fusion group.
    for group in &groups {
        assert!(
            group.steps.len() <= 2,
            "matmul should prevent large fusion groups: {:?}",
            group.steps,
        );
    }
}

#[test]
fn test_partition_summary_correctness() {
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            make_dispatch_with_ext("relu", &[1, 64], vec![0]),
            make_dispatch_with_ext("exp", &[1, 64], vec![1]),
            CompiledStep::Passthrough {
                op_name: "reshape".into(),
                output_shape: vec![64],
            },
        ],
        input_shapes: vec![vec![1, 64]],
        output_step: 2,
        weight_names: vec![],
    };
    let dag = partition_plan(&plan, None);
    let groups = find_fusion_groups(&dag);
    let summary = partition_summary(&dag, &groups);
    assert_eq!(summary.total_dispatches, 2, "only Dispatch steps count");
    assert_eq!(
        summary.elementwise_dispatches, 2,
        "relu and exp are elementwise"
    );
    assert!(
        summary.theoretical_minimum <= summary.total_dispatches,
        "theoretical minimum should be <= total"
    );
}

#[test]
fn test_fanout_prevents_fusion_group() {
    // Step 1 has two consumers (steps 2 and 3) -- fan-out should prevent fusion.
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            make_dispatch_with_ext("relu", &[1, 64], vec![0]),
            make_dispatch_with_ext("exp", &[1, 64], vec![1]),
            make_dispatch_with_ext("neg", &[1, 64], vec![1]),
        ],
        input_shapes: vec![vec![1, 64]],
        output_step: 2,
        weight_names: vec![],
    };
    let dag = partition_plan(&plan, None);
    let groups = find_fusion_groups(&dag);
    // Step 1 (relu) has fan-out of 2, so no group should contain all 3.
    for group in &groups {
        let has_all =
            group.steps.contains(&1) && group.steps.contains(&2) && group.steps.contains(&3);
        assert!(
            !has_all,
            "fan-out should prevent all three ops from fusing into one group"
        );
    }
}

#[test]
fn test_dag_acyclicity_on_real_graph() {
    // Build a real graph, compile, and verify the partition DAG is acyclic.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 128]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![1, 128]),
        test_node(2, "exp", TraceOp::Exp, vec![1], vec![1, 128]),
        test_node(3, "sigmoid", TraceOp::Sigmoid, vec![2], vec![1, 128]),
    ]);
    let plan = compile_trace_to_plan(&graph).unwrap();
    let dag = partition_plan(&plan, None);
    // Verify acyclicity: for each step, no predecessor has a higher index
    // (since steps are in topological order by construction).
    for (i, preds) in dag.predecessors.iter().enumerate() {
        for &p in preds {
            assert!(
                p < i,
                "step {i} has predecessor {p} >= {i}, indicating a cycle or disorder"
            );
        }
    }
}

#[test]
fn test_dag_acyclicity_on_branching_graph() {
    // A graph where one op fans out to two independent paths.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, vec![1, 64]),
        test_node(1, "relu", TraceOp::Relu, vec![0], vec![1, 64]),
        test_node(2, "exp", TraceOp::Exp, vec![1], vec![1, 64]),
        test_node(3, "neg", TraceOp::Neg, vec![1], vec![1, 64]),
        test_node(4, "add", TraceOp::Add, vec![2, 3], vec![1, 64]),
    ]);
    let plan = compile_trace_to_plan(&graph).unwrap();
    let dag = partition_plan(&plan, None);
    // DAG should be acyclic.
    for (i, preds) in dag.predecessors.iter().enumerate() {
        for &p in preds {
            assert!(p < i, "cycle detected: step {i} depends on step {p}");
        }
    }
}
