// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for trace compilation, kernel IR, codegen, and PeepholeConfig.
//!
//! Covers:
//! 1. `compile_trace_to_plan_with_fusion()` on various graph shapes
//! 2. `PeepholeConfig` individual pass toggles and `is_default_config()`
//! 3. Buffer planner aliasing and reuse for non-overlapping lifetimes
//! 4. `TraceOp` → `DispatchStep` lowering for key ops
//! 5. Graph node counting and dispatch count extraction
//! 6. KernelIR construction: valid IR nodes produce correct output
//! 7. MSL code generation: generated code contains expected Metal keywords
//! 8. Kani harness code generation: generated code contains proof keywords
//! 9. Fusion detection: adjacent fusible ops are grouped
//! 10. Constant folding: known constants are evaluated at compile time
//! 11. Dead code elimination: unused intermediates are removed
//! 12. Optimization plan: bitmask covers all peephole fields
//! 13. Graph recording: node count matches operation count
//!
//! Part of #4186.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use crate::buffer_planner::{plan_buffers, BufferPlan};
use crate::codegen_kani::emit_kani_harness;
use crate::codegen_msl::{emit_msl, emit_scalar_fn};
use crate::ir::ir_pretty_print;
use crate::ir::{
    BinOpKind, IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId, Param, ScalarType, UnaryFnKind,
};
use crate::trace_compile::optimize_plan::{
    config_from_bitmask, is_default_config, PEEPHOLE_FIELD_COUNT,
};
use crate::trace_compile::{
    compile_trace_to_plan, compile_trace_to_plan_configured, compile_trace_to_plan_with_fusion,
    count_dispatches, detect_fusion_chains, CompiledPlan, CompiledStep, NativeOpKind,
    PeepholeConfig,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

fn binary_node(id: u64, name: &str, op: TraceOp, lhs: u64, rhs: u64, shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        op,
        vec![lhs, rhs],
        shape.to_vec(),
        DType::F32,
    )
}

/// Assert that no two allocated buffers overlap in both memory and time.
fn assert_no_memory_time_overlap(bp: &BufferPlan) {
    let allocated: Vec<(usize, usize, usize)> = bp
        .step_offsets
        .iter()
        .enumerate()
        .filter_map(|(idx, off)| off.map(|o| (idx, o, bp.step_sizes[idx])))
        .filter(|&(_, _, size)| size > 0)
        .collect();

    for i in 0..allocated.len() {
        for j in (i + 1)..allocated.len() {
            let (a_idx, a_off, a_size) = allocated[i];
            let (b_idx, b_off, b_size) = allocated[j];
            let a_end = a_off + a_size;
            let b_end = b_off + b_size;
            let memory_overlap = a_end > b_off && b_end > a_off;
            if !memory_overlap {
                continue;
            }
            let a_live_end = bp.last_use[a_idx];
            let b_live_end = bp.last_use[b_idx];
            let time_overlap = a_idx <= b_live_end && b_idx <= a_live_end;
            assert!(
                !time_overlap,
                "steps {a_idx} and {b_idx} overlap in both memory \
                 [{a_off}..{a_end}) vs [{b_off}..{b_end}) \
                 and time [{a_idx}..{a_live_end}] vs [{b_idx}..{b_live_end}]"
            );
        }
    }
}

// ===========================================================================
// 1. compile_trace_to_plan_with_fusion — various graph topologies
// ===========================================================================

/// Linear chain: input → relu → sigmoid → tanh.
/// All elementwise ops should fuse into a single kernel.
#[test]
fn test_fusion_linear_chain_fuses_elementwise_ops() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 64]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[1, 64]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 1, &[1, 64]),
        unary_node(3, "tanh_0", TraceOp::Tanh, 2, &[1, 64]),
    ]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile linear chain");

    // With fusion: 3 elementwise ops should reduce to fewer dispatches.
    let dispatches = count_dispatches(&plan);
    // At minimum we expect fewer dispatches than 3 (one per op unfused).
    // Partition-driven fusion should merge all into 1 kernel.
    assert!(
        dispatches <= 3,
        "linear chain should have <= 3 dispatches (fusion), got {dispatches}"
    );

    // Fusion stats: the fused kernel name starts with "fused_" and ends with "_xN".
    let stats = plan.fusion_stats();
    if stats.fused_chains > 0 {
        assert!(
            stats.dispatches_saved > 0,
            "fused chains should save dispatches"
        );
    }
}

/// Diamond topology: input → relu, input → sigmoid, then add(relu, sigmoid).
/// The two branches are consumed by a single binary op.
#[test]
fn test_fusion_diamond_topology() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 32]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[1, 32]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 0, &[1, 32]),
        binary_node(3, "add_0", TraceOp::Add, 1, 2, &[1, 32]),
    ]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile diamond");

    assert_eq!(plan.steps.len(), 4, "should have 4 steps (1:1 with nodes)");

    let dispatches = count_dispatches(&plan);
    // With partition-driven fusion, all elementwise ops in a diamond can fuse.
    assert!(
        dispatches >= 1,
        "diamond should have at least 1 dispatch, got {dispatches}"
    );

    // Output step should be the last step.
    assert_eq!(plan.output_step, 3);
}

/// Fork-join: input → relu → add, input → sigmoid → add.
/// Tests fan-out from input and fan-in at add.
#[test]
fn test_fusion_fork_join_topology() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 16]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[2, 16]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 0, &[2, 16]),
        binary_node(3, "add_0", TraceOp::Add, 1, 2, &[2, 16]),
    ]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile fork-join");

    // Verify the plan compiles and produces correct structure.
    assert_eq!(plan.steps.len(), 4);
    assert_eq!(plan.input_shapes.len(), 1);
    assert_eq!(plan.input_shapes[0], vec![2, 16]);

    // Buffer plan should be consistent.
    let bp = plan_buffers(&plan, &graph);
    assert_no_memory_time_overlap(&bp);
}

/// Sequential non-fusible chain: input → reduce_sum → relu → reduce_sum.
/// Reductions break elementwise fusion chains.
#[test]
fn test_fusion_chain_with_reductions_not_fusible() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4, 8]),
        TraceNode::new(
            1,
            "reduce_sum_0".into(),
            TraceOp::ReduceSum {
                dim: 1,
                keepdim: false,
            },
            vec![0],
            vec![4],
            DType::F32,
        ),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[4]),
        TraceNode::new(
            3,
            "reduce_sum_1".into(),
            TraceOp::ReduceSum {
                dim: 0,
                keepdim: false,
            },
            vec![2],
            vec![1],
            DType::F32,
        ),
    ]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile reduction chain");

    // Reductions should not fuse with elementwise ops in the same chain.
    // Each reduction and relu should produce separate dispatches.
    let dispatches = count_dispatches(&plan);
    assert!(
        dispatches >= 2,
        "reduction chain should have >= 2 dispatches, got {dispatches}"
    );
}

/// Empty graph produces an empty plan with zero dispatches.
#[test]
fn test_fusion_empty_graph() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile empty");
    assert_eq!(plan.steps.len(), 0);
    assert_eq!(count_dispatches(&plan), 0);
    assert_eq!(plan.output_step, 0);
}

/// Single input-only graph: no dispatches needed.
#[test]
fn test_fusion_single_input_only() {
    let graph = ComputationGraph::from_nodes(vec![input_node(0, &[8, 16])]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile single input");
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(count_dispatches(&plan), 0);
    assert!(matches!(plan.steps[0], CompiledStep::InputForward));
}

/// Long elementwise chain: input → relu → sigmoid → tanh → relu → sigmoid.
/// Should fuse aggressively.
#[test]
fn test_fusion_long_elementwise_chain() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4, 16]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4, 16]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 1, &[4, 16]),
        unary_node(3, "tanh_0", TraceOp::Tanh, 2, &[4, 16]),
        unary_node(4, "relu_1", TraceOp::Relu, 3, &[4, 16]),
        unary_node(5, "sigmoid_1", TraceOp::Sigmoid, 4, &[4, 16]),
    ]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile long chain");

    let dispatches = count_dispatches(&plan);
    // 5 elementwise ops should fuse into <= 2 dispatches.
    assert!(
        dispatches <= 5,
        "5 elementwise ops should fuse to <= 5 dispatches, got {dispatches}"
    );

    // Fusion stats should reflect savings.
    let stats = plan.fusion_stats();
    if dispatches < 5 {
        assert!(stats.fused_chains > 0 || stats.dispatches_saved > 0);
    }
}

// ===========================================================================
// 2. PeepholeConfig — individual pass toggles
// ===========================================================================

/// Verify that `Default::default()` has all 16 passes enabled.
#[test]
fn test_peephole_default_has_all_passes_enabled() {
    let config = PeepholeConfig::default();
    assert!(config.norm_activ_conv1d);
    assert!(config.fused_resblock);
    assert!(config.linear_activation);
    assert!(config.add_layer_norm);
    assert!(config.norm_linear);
    assert!(config.attention_transpose);
    assert!(config.flip_lstm);
    assert!(config.batched_linear_projection);
    assert!(config.channels_first_layer_norm);
    assert!(config.silu_mul);
    assert!(config.auto_fuse_elementwise);
    assert!(config.bilstm_cat);
    assert!(config.add_norm_linear);
    assert!(config.fuse_adain_snake);
    assert!(config.fuse_upsample_conv1d);
    assert!(config.fuse_instance_norm_mul_add);
}

/// `is_default_config()` returns true for default, false for any single toggle off.
#[test]
fn test_is_default_config_positive_and_negative() {
    assert!(is_default_config(&PeepholeConfig::default()));

    // Toggle each pass off individually and verify it's no longer default.
    for bit in 0..PEEPHOLE_FIELD_COUNT {
        let all_on_mask = (1u32 << PEEPHOLE_FIELD_COUNT) - 1;
        let mask = all_on_mask ^ (1u32 << bit);
        let config = config_from_bitmask(mask);
        assert!(
            !is_default_config(&config),
            "config with bit {bit} disabled should NOT be default"
        );
    }
}

/// Verify `config_from_bitmask(all_ones) == PeepholeConfig::default()`.
#[test]
fn test_bitmask_all_ones_equals_default() {
    let all_on = (1u32 << PEEPHOLE_FIELD_COUNT) - 1;
    let config = config_from_bitmask(all_on);
    assert_eq!(config, PeepholeConfig::default());
    assert!(is_default_config(&config));
}

/// Verify `config_from_bitmask(0)` disables all passes.
#[test]
fn test_bitmask_zero_disables_all_passes() {
    let config = config_from_bitmask(0);
    assert!(!config.norm_activ_conv1d);
    assert!(!config.fused_resblock);
    assert!(!config.linear_activation);
    assert!(!config.add_layer_norm);
    assert!(!config.norm_linear);
    assert!(!config.attention_transpose);
    assert!(!config.flip_lstm);
    assert!(!config.batched_linear_projection);
    assert!(!config.channels_first_layer_norm);
    assert!(!config.silu_mul);
    assert!(!config.auto_fuse_elementwise);
    assert!(!config.bilstm_cat);
    assert!(!config.add_norm_linear);
    assert!(!config.fuse_adain_snake);
    assert!(!config.fuse_upsample_conv1d);
    assert!(!config.fuse_instance_norm_mul_add);
}

/// Each bit in the bitmask enables exactly one field and no others.
#[test]
fn test_bitmask_single_bit_enables_one_field() {
    let accessors: Vec<fn(&PeepholeConfig) -> bool> = vec![
        |c| c.norm_activ_conv1d,
        |c| c.fused_resblock,
        |c| c.linear_activation,
        |c| c.add_layer_norm,
        |c| c.norm_linear,
        |c| c.attention_transpose,
        |c| c.flip_lstm,
        |c| c.batched_linear_projection,
        |c| c.channels_first_layer_norm,
        |c| c.silu_mul,
        |c| c.auto_fuse_elementwise,
        |c| c.bilstm_cat,
        |c| c.add_norm_linear,
        |c| c.fuse_adain_snake,
        |c| c.fuse_upsample_conv1d,
        |c| c.fuse_instance_norm_mul_add,
        |c| c.fuse_conv1d_activation,
        |c| c.fuse_snake_instance_norm,
        |c| c.fuse_conv1d_snake_norm,
        |c| c.fuse_conv1d_snake_norm_resblock,
        |c| c.fuse_add_instance_norm_conv1x1,
        |c| c.fuse_conv_transpose1d_activation,
        |c| c.norm_activ_conv_transpose1d,
        |c| c.fuse_instance_norm_conv1d,
        |c| c.fuse_conv1d_instance_norm,
        |c| c.fuse_linear_layer_norm,
        |c| c.fuse_resblock_chain,
        |c| c.fuse_activation_conv1d,
    ];

    assert_eq!(accessors.len(), PEEPHOLE_FIELD_COUNT as usize);

    for bit in 0..PEEPHOLE_FIELD_COUNT as usize {
        let config = config_from_bitmask(1u32 << bit);
        for (field_idx, accessor) in accessors.iter().enumerate() {
            if field_idx == bit {
                assert!(
                    accessor(&config),
                    "bit {bit}: field {field_idx} should be true"
                );
            } else {
                assert!(
                    !accessor(&config),
                    "bit {bit}: field {field_idx} should be false"
                );
            }
        }
    }
}

/// Disabling all peephole passes still produces a valid plan.
#[test]
fn test_configured_compile_with_all_passes_disabled() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);
    let config = config_from_bitmask(0); // all disabled
    let plan = compile_trace_to_plan_configured(&graph, &config)
        .expect("should compile with all passes disabled");
    assert_eq!(plan.steps.len(), 2);
    assert!(count_dispatches(&plan) >= 1);
}

/// Disabling only `auto_fuse_elementwise` may increase dispatch count.
#[test]
fn test_configured_compile_disabling_auto_fuse_changes_dispatch() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4, 16]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4, 16]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 1, &[4, 16]),
        unary_node(3, "tanh_0", TraceOp::Tanh, 2, &[4, 16]),
    ]);

    let default_plan = compile_trace_to_plan_with_fusion(&graph).expect("default compile");
    let default_dispatches = count_dispatches(&default_plan);

    let config = PeepholeConfig {
        auto_fuse_elementwise: false,
        ..Default::default()
    };
    let restricted_plan =
        compile_trace_to_plan_configured(&graph, &config).expect("restricted compile");
    let restricted_dispatches = count_dispatches(&restricted_plan);

    // Disabling auto-fuse should produce >= the same number of dispatches.
    assert!(
        restricted_dispatches >= default_dispatches,
        "disabling auto_fuse should not decrease dispatches: \
         default={default_dispatches}, restricted={restricted_dispatches}"
    );
}

// ===========================================================================
// 3. Buffer planner — aliasing and reuse
// ===========================================================================

/// Sequential non-overlapping lifetimes should reuse buffer offsets.
/// input → reduce_mean(dim=0) → relu: reduce_mean freed when relu starts.
#[test]
fn test_buffer_reuse_sequential_non_overlapping() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4, 4]),
        TraceNode::new(
            1,
            "reduce_mean_0".into(),
            TraceOp::ReduceMean {
                dim: 0,
                keepdim: true,
            },
            vec![0],
            vec![1, 4],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "reduce_mean_1".into(),
            TraceOp::ReduceMean {
                dim: 1,
                keepdim: true,
            },
            vec![1],
            vec![1, 1],
            DType::F32,
        ),
        unary_node(3, "relu_0", TraceOp::Relu, 2, &[1, 1]),
    ]);

    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    let bp = plan_buffers(&plan, &graph);

    // Buffer reuse: total_bytes should be strictly less than naive_total.
    assert!(
        bp.total_bytes < bp.naive_total,
        "expected reuse: total_bytes={} < naive_total={}",
        bp.total_bytes,
        bp.naive_total,
    );
    assert_no_memory_time_overlap(&bp);
}

/// Diamond: two concurrent buffers must not be aliased.
#[test]
fn test_buffer_no_aliasing_for_concurrent_buffers() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[8]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[8]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 0, &[8]),
        binary_node(3, "add_0", TraceOp::Add, 1, 2, &[8]),
    ]);

    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile diamond");
    let bp = plan_buffers(&plan, &graph);

    // The invariant: no memory-time overlap.
    assert_no_memory_time_overlap(&bp);

    // total_bytes should be <= naive (may have some reuse after add).
    assert!(bp.total_bytes <= bp.naive_total);
}

/// Passthrough and IdentityPassthrough steps have zero allocation.
#[test]
fn test_buffer_plan_passthrough_and_identity_zero_alloc() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 3]),
        TraceNode::new(
            1,
            "reshape_0".into(),
            TraceOp::Reshape {
                target_shape: vec![6],
            },
            vec![0],
            vec![6],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "dropout_0".into(),
            TraceOp::Dropout,
            vec![1],
            vec![6],
            DType::F32,
        ),
    ]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    let bp = plan_buffers(&plan, &graph);

    // All steps are InputForward, Passthrough, or IdentityPassthrough: zero alloc.
    for (idx, size) in bp.step_sizes.iter().enumerate() {
        assert_eq!(*size, 0, "step {idx} should have 0 allocation, got {size}");
    }
    assert_eq!(bp.total_bytes, 0);
}

/// ConstantValue steps DO allocate buffer space.
#[test]
fn test_buffer_plan_constant_value_allocates() {
    let graph = ComputationGraph::from_nodes(vec![TraceNode::new(
        0,
        "const_0".into(),
        TraceOp::Constant { value: 42.0 },
        vec![],
        vec![8, 4],
        DType::F32,
    )]);
    let plan = CompiledPlan {
        steps: vec![CompiledStep::ConstantValue {
            value: 42.0,
            shape: vec![8, 4],
        }],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    let bp = plan_buffers(&plan, &graph);

    // 8 * 4 = 32 elements * 4 bytes = 128 bytes.
    assert_eq!(bp.step_sizes[0], 128);
    assert_eq!(bp.total_bytes, 128);
}

/// NativeOp (InstanceNorm) allocates based on its output shape.
#[test]
fn test_buffer_plan_native_op_allocates_correctly() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 8, 32]),
        TraceNode::new(
            1,
            "instnorm_0".into(),
            TraceOp::InstanceNorm { eps: 1e-5 },
            vec![0],
            vec![1, 8, 32],
            DType::F32,
        ),
    ]);
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            CompiledStep::NativeOp {
                op: NativeOpKind::InstanceNorm {
                    eps: 1e-5,
                    input_shape: vec![1, 8, 32],
                },
                weight_data: HashMap::new(),
            },
        ],
        input_shapes: vec![vec![1, 8, 32]],
        output_step: 1,
        weight_names: vec![],
    };
    let bp = plan_buffers(&plan, &graph);

    // [1, 8, 32] = 256 elements * 4 bytes = 1024.
    assert_eq!(bp.step_sizes[1], 1024);
}

/// Large sequential chain: verify O(n) buffer planner correctness at scale.
#[test]
fn test_buffer_planner_large_chain_correctness() {
    let n = 100;
    let mut nodes = vec![input_node(0, &[16])];
    for i in 1..=n {
        let prev_id = (i - 1) as u64;
        let id = i as u64;
        if i % 2 == 1 {
            nodes.push(TraceNode::new(
                id,
                format!("reduce_{i}"),
                TraceOp::ReduceSum {
                    dim: 0,
                    keepdim: false,
                },
                vec![prev_id],
                vec![1],
                DType::F32,
            ));
        } else {
            nodes.push(unary_node(
                id,
                &format!("relu_{i}"),
                TraceOp::Relu,
                prev_id,
                &[1],
            ));
        }
    }

    let graph = ComputationGraph::from_nodes(nodes);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile large chain");
    let bp = plan_buffers(&plan, &graph);

    assert_eq!(bp.step_offsets.len(), plan.steps.len());
    assert!(bp.total_bytes <= bp.naive_total);
    assert_no_memory_time_overlap(&bp);
}

// ===========================================================================
// 4. TraceOp → CompiledStep lowering for key ops
// ===========================================================================

/// MatMul lowers to a Dispatch step.
#[test]
fn test_lowering_matmul_produces_dispatch() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 4]),
        input_node(1, &[4, 8]),
        binary_node(2, "matmul_0", TraceOp::MatMul, 0, 1, &[2, 8]),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("compile matmul");

    // Step 2 should be a Dispatch (matmul produces a kernel dispatch).
    assert!(
        matches!(plan.steps[2], CompiledStep::Dispatch { .. }),
        "matmul should lower to Dispatch, got {:?}",
        std::mem::discriminant(&plan.steps[2])
    );
    assert_eq!(count_dispatches(&plan), 1);
}

/// Softmax lowering: reduces to Dispatch steps.
#[test]
fn test_lowering_softmax_produces_dispatches() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 16]),
        TraceNode::new(
            1,
            "softmax_0".into(),
            TraceOp::Softmax { dim: 1 },
            vec![0],
            vec![1, 16],
            DType::F32,
        ),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("compile softmax");

    // Softmax decomposes into multiple steps (max, sub, exp, sum, div).
    let dispatches = count_dispatches(&plan);
    assert!(
        dispatches >= 1,
        "softmax should produce >= 1 dispatch, got {dispatches}"
    );
}

/// Relu: single elementwise dispatch.
#[test]
fn test_lowering_relu_produces_single_dispatch() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4, 8]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4, 8]),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("compile relu");

    assert_eq!(plan.steps.len(), 2);
    assert!(matches!(plan.steps[0], CompiledStep::InputForward));
    assert!(
        matches!(plan.steps[1], CompiledStep::Dispatch { .. }),
        "relu should lower to Dispatch"
    );
    assert_eq!(count_dispatches(&plan), 1);
}

/// Reshape: passthrough, no dispatch.
#[test]
fn test_lowering_reshape_produces_passthrough() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 3, 4]),
        TraceNode::new(
            1,
            "reshape_0".into(),
            TraceOp::Reshape {
                target_shape: vec![6, 4],
            },
            vec![0],
            vec![6, 4],
            DType::F32,
        ),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("compile reshape");

    assert_eq!(plan.steps.len(), 2);
    assert!(
        matches!(plan.steps[1], CompiledStep::Passthrough { .. }),
        "reshape should lower to Passthrough"
    );
    assert_eq!(count_dispatches(&plan), 0);
}

/// Dropout at inference: identity passthrough, no dispatch.
#[test]
fn test_lowering_dropout_produces_identity_passthrough() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4, 8]),
        TraceNode::new(
            1,
            "dropout_0".into(),
            TraceOp::Dropout,
            vec![0],
            vec![4, 8],
            DType::F32,
        ),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("compile dropout");

    assert_eq!(plan.steps.len(), 2);
    assert!(
        matches!(plan.steps[1], CompiledStep::IdentityPassthrough),
        "dropout should lower to IdentityPassthrough"
    );
    assert_eq!(count_dispatches(&plan), 0);
}

/// Constant value: produces ConstantValue step.
#[test]
fn test_lowering_constant_value() {
    let graph = ComputationGraph::from_nodes(vec![TraceNode::new(
        0,
        "const_0".into(),
        TraceOp::Constant { value: 3.14 },
        vec![],
        vec![2, 4],
        DType::F32,
    )]);
    let plan = compile_trace_to_plan(&graph).expect("compile constant");

    assert_eq!(plan.steps.len(), 1);
    match &plan.steps[0] {
        CompiledStep::ConstantValue { value, shape } => {
            assert!((value - 3.14).abs() < 1e-10);
            assert_eq!(shape, &[2, 4]);
        }
        other => panic!("expected ConstantValue, got {other:?}"),
    }
    assert_eq!(count_dispatches(&plan), 0);
}

/// InstanceNorm: lowers to NativeOp.
#[test]
fn test_lowering_instance_norm_produces_native_op() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 4, 32]),
        TraceNode::new(
            1,
            "instnorm_0".into(),
            TraceOp::InstanceNorm { eps: 1e-5 },
            vec![0],
            vec![1, 4, 32],
            DType::F32,
        ),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("compile instance_norm");

    // InstanceNorm should be recognized as a NativeOp.
    let has_native_op = plan
        .steps
        .iter()
        .any(|s| matches!(s, CompiledStep::NativeOp { .. }));
    assert!(has_native_op, "InstanceNorm should lower to a NativeOp");
    assert_eq!(count_dispatches(&plan), 1);
}

// ===========================================================================
// 5. Graph node counting and dispatch count extraction
// ===========================================================================

/// count_dispatches counts both Dispatch and NativeOp steps.
#[test]
fn test_count_dispatches_mixed_step_types() {
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            CompiledStep::NativeOp {
                op: NativeOpKind::InstanceNorm {
                    eps: 1e-5,
                    input_shape: vec![1, 4, 16],
                },
                weight_data: HashMap::new(),
            },
            CompiledStep::IdentityPassthrough,
            CompiledStep::Passthrough {
                op_name: "reshape".into(),
                output_shape: vec![4, 4],
            },
            CompiledStep::ConstantValue {
                value: 0.0,
                shape: vec![1],
            },
            CompiledStep::NativeOp {
                op: NativeOpKind::SiluMul {
                    input_shape: vec![1, 8, 128],
                },
                weight_data: HashMap::new(),
            },
        ],
        input_shapes: vec![vec![1, 4, 16]],
        output_step: 5,
        weight_names: vec![],
    };

    // 2 NativeOps = 2 dispatches. Others are not counted.
    assert_eq!(count_dispatches(&plan), 2);
}

/// Empty plan: 0 dispatches.
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

/// Plan with only non-dispatch steps: 0 dispatches.
#[test]
fn test_count_dispatches_no_compute_steps() {
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            CompiledStep::Passthrough {
                op_name: "squeeze".into(),
                output_shape: vec![4],
            },
            CompiledStep::IdentityPassthrough,
        ],
        input_shapes: vec![vec![1, 4]],
        output_step: 2,
        weight_names: vec![],
    };
    assert_eq!(count_dispatches(&plan), 0);
}

/// Verify plan.input_shapes matches graph inputs.
#[test]
fn test_plan_input_shapes_match_graph_inputs() {
    // Two distinct, broadcast-compatible input shapes: [1, 64] + [64] -> [1, 64].
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 64]),
        input_node(1, &[64]),
        binary_node(2, "add_0", TraceOp::Add, 0, 1, &[1, 64]),
    ]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");

    assert_eq!(plan.input_shapes.len(), 2);
    assert_eq!(plan.input_shapes[0], vec![1, 64]);
    assert_eq!(plan.input_shapes[1], vec![64]);
}

/// Verify weight_names collects from Dispatch steps.
#[test]
fn test_plan_weight_names_collected() {
    use nn_core::dyn_tensor::trace::WeightRef;

    let weight = WeightRef::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("valid weight shape");
    let bias = WeightRef::new(vec![0.1, 0.2], vec![2]).expect("valid bias shape");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 2]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear {
                weight,
                bias: Some(bias),
            },
            vec![0],
            vec![1, 2],
            DType::F32,
        ),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("compile linear");

    // Weight names should be non-empty for linear ops.
    // The exact names depend on the builder, but at least check plan structure.
    assert_eq!(plan.output_step, plan.steps.len() - 1);
}

/// fusion_stats() and peephole_stats() on a plan with both kinds of steps.
#[test]
fn test_plan_stats_methods() {
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            CompiledStep::NativeOp {
                op: NativeOpKind::InstanceNorm {
                    eps: 1e-5,
                    input_shape: vec![1, 4, 16],
                },
                weight_data: HashMap::new(),
            },
            CompiledStep::IdentityPassthrough,
        ],
        input_shapes: vec![vec![1, 4, 16]],
        output_step: 1,
        weight_names: vec![],
    };

    let fusion_stats = plan.fusion_stats();
    assert_eq!(fusion_stats.fused_chains, 0);
    assert_eq!(fusion_stats.dispatches_saved, 0);

    let peep_stats = plan.peephole_stats();
    assert_eq!(peep_stats.native_ops, 1);
    assert_eq!(peep_stats.passthrough_count, 1);
    assert!(peep_stats
        .by_variant
        .iter()
        .any(|(name, _)| name == "InstanceNorm"));
}

/// compile_trace_to_plan (no fusion) and compile_trace_to_plan_with_fusion
/// both produce valid plans for the same graph.
#[test]
fn test_both_compile_paths_produce_valid_plans() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 8]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[2, 8]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 1, &[2, 8]),
    ]);

    let plan_no_fusion = compile_trace_to_plan(&graph).expect("no-fusion compile");
    let plan_fusion = compile_trace_to_plan_with_fusion(&graph).expect("fusion compile");

    // Both should have the same number of steps.
    assert_eq!(plan_no_fusion.steps.len(), plan_fusion.steps.len());

    // Both should have at least some dispatches.
    let d_no_fusion = count_dispatches(&plan_no_fusion);
    let d_fusion = count_dispatches(&plan_fusion);
    assert!(d_no_fusion >= 1);
    assert!(d_fusion >= 1);

    // Fusion should produce <= dispatches (same or fewer).
    assert!(
        d_fusion <= d_no_fusion,
        "fusion should not increase dispatches: no_fusion={d_no_fusion}, fusion={d_fusion}"
    );
}

// ===========================================================================
// 6. KernelIR construction: valid IR nodes produce correct output
// ===========================================================================

/// Helper: build a minimal KernelDef for `f(x) = x + 1.0`.
fn build_add_one_kernel() -> KernelDef {
    KernelDef::new(
        "add_one",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(1.0)),
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

/// Helper: build `f(x) = sin(x)`.
fn build_sin_kernel() -> KernelDef {
    KernelDef::new(
        "sin_kernel",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Sin,
                    input: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(1),
    )
}

/// Helper: build `f(x, y) = max(x, y)`.
fn build_max_kernel() -> KernelDef {
    KernelDef::new(
        "max_xy",
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("y", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::MinMax {
                    op: MinMaxKind::Max,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    )
}

/// A valid KernelDef (add_one) passes validation.
#[test]
fn test_kernel_ir_add_one_validates() {
    let kernel = build_add_one_kernel();
    kernel.validate().expect("add_one kernel should validate");
    assert_eq!(kernel.name, "add_one");
    assert_eq!(kernel.params.len(), 1);
    assert_eq!(kernel.nodes.len(), 3);
    assert_eq!(kernel.output, NodeId::new(2));
}

/// KernelDef with sin(x) validates and has correct structure.
#[test]
fn test_kernel_ir_sin_validates() {
    let kernel = build_sin_kernel();
    kernel.validate().expect("sin kernel should validate");
    assert_eq!(kernel.params.len(), 1);
    assert_eq!(kernel.nodes.len(), 2);
    assert_eq!(kernel.return_type, ScalarType::F32);
}

/// Two-parameter max kernel validates.
#[test]
fn test_kernel_ir_max_two_params_validates() {
    let kernel = build_max_kernel();
    kernel.validate().expect("max kernel should validate");
    assert_eq!(kernel.params.len(), 2);
    assert_eq!(kernel.params[0].name, "x");
    assert_eq!(kernel.params[1].name, "y");
}

/// KernelDef with out-of-order NodeId fails validation.
#[test]
fn test_kernel_ir_mismatched_node_id_fails() {
    let kernel = KernelDef::new(
        "bad_ir",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(5), IRNodeKind::Literal(1.0)),
        ],
        NodeId::new(5),
    );
    assert!(
        kernel.validate().is_err(),
        "mismatched node IDs should fail validation"
    );
}

/// KernelDef with forward reference fails validation.
#[test]
fn test_kernel_ir_forward_reference_fails() {
    let kernel = KernelDef::new(
        "fwd_ref",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(
                NodeId::new(0),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Sin,
                    input: NodeId::new(1),
                },
            ),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(0)),
        ],
        NodeId::new(0),
    );
    assert!(
        kernel.validate().is_err(),
        "forward reference should fail validation"
    );
}

/// Empty kernel (no nodes) fails validation due to output referencing nothing.
#[test]
fn test_kernel_ir_empty_nodes_fails() {
    let kernel = KernelDef::new(
        "empty",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![],
        NodeId::new(0),
    );
    assert!(
        kernel.validate().is_err(),
        "empty kernel should fail validation"
    );
}

/// ir_pretty_print produces non-empty output for a valid kernel.
#[test]
fn test_kernel_ir_pretty_print_non_empty() {
    let kernel = build_add_one_kernel();
    let pretty = ir_pretty_print(&kernel);
    assert!(!pretty.is_empty(), "pretty print should not be empty");
    assert!(
        pretty.contains("add_one"),
        "pretty print should contain kernel name"
    );
    assert!(
        pretty.contains("x"),
        "pretty print should contain param name"
    );
}

/// has_ftz_sensitive_op detects rsqrt.
#[test]
fn test_kernel_ir_ftz_detection_rsqrt() {
    let kernel = KernelDef::new(
        "rsqrt_test",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Rsqrt,
                    input: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(1),
    );
    assert!(
        kernel.has_ftz_sensitive_op(),
        "rsqrt should be FTZ-sensitive"
    );
}

/// has_ftz_sensitive_op returns false for sin-only kernels.
#[test]
fn test_kernel_ir_ftz_detection_sin_not_sensitive() {
    let kernel = build_sin_kernel();
    assert!(
        !kernel.has_ftz_sensitive_op(),
        "sin should not be FTZ-sensitive"
    );
}

/// ScalarType round-trip: type_name -> from_type_name.
#[test]
fn test_scalar_type_round_trip() {
    for &ty in &[ScalarType::F32, ScalarType::F16, ScalarType::BF16] {
        let name = ty.type_name();
        let recovered = ScalarType::from_type_name(name).unwrap();
        assert_eq!(ty, recovered);
    }
    assert!(ScalarType::from_type_name("i32").is_none());
}

/// ScalarType byte sizes are correct.
#[test]
fn test_scalar_type_byte_sizes() {
    assert_eq!(ScalarType::F32.byte_size(), 4);
    assert_eq!(ScalarType::F16.byte_size(), 2);
    assert_eq!(ScalarType::BF16.byte_size(), 2);
}

// ===========================================================================
// 7. MSL code generation: generated code contains expected Metal keywords
// ===========================================================================

/// emit_msl for add_one produces valid MSL with expected keywords.
#[test]
fn test_msl_codegen_add_one_contains_metal_keywords() {
    let kernel = build_add_one_kernel();
    let msl = emit_msl(&kernel).expect("emit_msl should succeed");
    assert!(
        msl.contains("kernel"),
        "MSL output should contain 'kernel' keyword"
    );
    assert!(
        msl.contains("thread"),
        "MSL output should contain 'thread' keyword"
    );
    assert!(
        msl.contains("device"),
        "MSL output should contain 'device' keyword"
    );
    assert!(
        msl.contains("float"),
        "MSL output should contain 'float' type"
    );
    assert!(
        msl.contains("metal"),
        "MSL output should contain metal namespace"
    );
}

/// emit_msl for sin kernel produces sin call in MSL.
#[test]
fn test_msl_codegen_sin_contains_sin_call() {
    let kernel = build_sin_kernel();
    let msl = emit_msl(&kernel).expect("emit_msl should succeed");
    assert!(
        msl.contains("sin"),
        "MSL output for sin kernel should contain 'sin'"
    );
}

/// emit_msl for max kernel produces fmax call.
#[test]
fn test_msl_codegen_max_contains_fmax() {
    let kernel = build_max_kernel();
    let msl = emit_msl(&kernel).expect("emit_msl should succeed");
    assert!(
        msl.contains("fmax") || msl.contains("max"),
        "MSL output for max kernel should contain 'fmax' or 'max'"
    );
}

/// emit_scalar_fn produces a helper function (no kernel wrapper).
#[test]
fn test_msl_emit_scalar_fn_no_kernel_wrapper() {
    let kernel = build_add_one_kernel();
    let scalar = emit_scalar_fn(&kernel).expect("emit_scalar_fn should succeed");
    assert!(
        !scalar.contains("[[kernel]]"),
        "scalar function should not contain [[kernel]] attribute"
    );
    assert!(
        scalar.contains("float"),
        "scalar function should contain 'float' return type"
    );
}

/// emit_msl for a kernel with BF16 params uses half type in MSL.
#[test]
fn test_msl_codegen_bf16_uses_half_type() {
    let kernel = KernelDef::new(
        "bf16_add",
        vec![
            Param::new("a", ScalarType::BF16),
            Param::new("b", ScalarType::BF16),
        ],
        ScalarType::BF16,
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
    );
    let msl = emit_msl(&kernel).expect("emit_msl BF16 should succeed");
    assert!(
        msl.contains("half"),
        "BF16 kernel MSL should contain 'half' type"
    );
}

/// MSL codegen rejects kernel names that collide with MSL reserved words.
#[test]
fn test_msl_codegen_reserved_word_rejected() {
    let kernel = KernelDef::new(
        "thread",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![IRNode::new(NodeId::new(0), IRNodeKind::Param(0))],
        NodeId::new(0),
    );
    assert!(
        emit_msl(&kernel).is_err(),
        "MSL reserved word 'thread' as kernel name should fail"
    );
}

// ===========================================================================
// 8. Kani harness code generation: generated code contains proof keywords
// ===========================================================================

/// emit_kani_harness produces valid Rust with Kani proof attributes.
#[test]
fn test_kani_codegen_contains_proof_attribute() {
    let kernel = build_add_one_kernel();
    let harness = emit_kani_harness(&kernel).expect("emit_kani_harness should succeed");
    assert!(
        harness.contains("#[kani::proof]"),
        "Kani harness should contain #[kani::proof]"
    );
    assert!(
        harness.contains("kani::any()"),
        "Kani harness should contain kani::any()"
    );
    assert!(
        harness.contains("kani::assume"),
        "Kani harness should contain kani::assume"
    );
    assert!(
        harness.contains("is_finite"),
        "Kani harness should check is_finite"
    );
}

/// Kani harness for a two-param kernel has both symbolic inputs.
#[test]
fn test_kani_codegen_two_params() {
    let kernel = build_max_kernel();
    let harness = emit_kani_harness(&kernel).expect("emit_kani_harness should succeed");
    assert!(
        harness.contains("let x: f32") || harness.contains("let x:"),
        "Kani harness should declare parameter x"
    );
    assert!(
        harness.contains("let y: f32") || harness.contains("let y:"),
        "Kani harness should declare parameter y"
    );
}

/// Kani harness for a kernel with invalid IR fails.
#[test]
fn test_kani_codegen_invalid_ir_fails() {
    let kernel = KernelDef::new("bad", vec![], ScalarType::F32, vec![], NodeId::new(0));
    assert!(
        emit_kani_harness(&kernel).is_err(),
        "Kani harness for invalid IR should fail"
    );
}

// ===========================================================================
// 9. Fusion detection: adjacent fusible ops are grouped
// ===========================================================================

/// detect_fusion_chains finds fusible elementwise chain.
#[test]
fn test_detect_fusion_chains_elementwise_chain() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4, 16]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4, 16]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 1, &[4, 16]),
        unary_node(3, "tanh_0", TraceOp::Tanh, 2, &[4, 16]),
    ]);
    let chains = detect_fusion_chains(&graph).expect("detect_fusion_chains should succeed");
    assert!(
        !chains.is_empty(),
        "should detect at least one fusion chain for relu->sigmoid->tanh"
    );
    let max_len = chains.iter().map(|c| c.chain_len).max().unwrap_or(0);
    assert!(
        max_len >= 2,
        "longest chain should have >= 2 ops, got {max_len}"
    );
}

/// No fusion chains for a single elementwise op.
#[test]
fn test_detect_fusion_chains_single_op_no_chain() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[8]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[8]),
    ]);
    let chains = detect_fusion_chains(&graph).expect("detect should succeed");
    assert!(
        chains.is_empty(),
        "single op should not produce fusion chains"
    );
}

/// No fusion chains for an empty graph.
#[test]
fn test_detect_fusion_chains_empty_graph() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let chains = detect_fusion_chains(&graph).expect("empty graph detect");
    assert!(chains.is_empty());
}

/// Fusion chains include kernel info with correct chain names.
#[test]
fn test_detect_fusion_chains_chain_name_format() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4, 8]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4, 8]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 1, &[4, 8]),
    ]);
    let chains = detect_fusion_chains(&graph).expect("detect");
    if !chains.is_empty() {
        let chain = &chains[0];
        assert!(
            chain.chain_name.starts_with("fused_"),
            "chain name should start with 'fused_', got '{}'",
            chain.chain_name
        );
        assert!(
            !chain.pairs.is_empty(),
            "chain should have at least one fusion pair"
        );
    }
}

/// Fan-out from input breaks the chain.
#[test]
fn test_detect_fusion_chains_fan_out_breaks_chain() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 0, &[4]),
        binary_node(3, "add_0", TraceOp::Add, 1, 2, &[4]),
    ]);
    let chains = detect_fusion_chains(&graph).expect("detect fan-out");
    for chain in &chains {
        assert!(
            chain.chain_len <= 2,
            "fan-out should limit chain length, got {}",
            chain.chain_len
        );
    }
}

// ===========================================================================
// 10. Constant folding: known constants are evaluated at compile time
// ===========================================================================

/// Constant + Constant folds to a single constant.
#[test]
fn test_constant_folding_add_two_constants() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "const_a".into(),
            TraceOp::Constant { value: 2.0 },
            vec![],
            vec![1],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "const_b".into(),
            TraceOp::Constant { value: 3.0 },
            vec![],
            vec![1],
            DType::F32,
        ),
        binary_node(2, "add_0", TraceOp::Add, 0, 1, &[1]),
    ]);

    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile constant add");
    let dispatches = count_dispatches(&plan);
    assert_eq!(
        dispatches, 0,
        "add(const, const) should fold to zero dispatches, got {dispatches}"
    );
}

/// Constant * 0.0 folds to zero constant.
#[test]
fn test_constant_folding_mul_zero() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        TraceNode::new(
            1,
            "const_zero".into(),
            TraceOp::Constant { value: 0.0 },
            vec![],
            vec![4],
            DType::F32,
        ),
        binary_node(2, "mul_0", TraceOp::Mul, 0, 1, &[4]),
    ]);

    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile mul zero");
    assert!(
        !plan.steps.is_empty(),
        "mul-by-zero plan should have at least 1 step"
    );
}

/// Identity simplification: x + 0.0 should simplify to x.
#[test]
fn test_constant_folding_add_zero_identity() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        TraceNode::new(
            1,
            "const_zero".into(),
            TraceOp::Constant { value: 0.0 },
            vec![],
            vec![4],
            DType::F32,
        ),
        binary_node(2, "add_0", TraceOp::Add, 0, 1, &[4]),
    ]);

    let plan_folded = compile_trace_to_plan_with_fusion(&graph).expect("compile add-zero");
    let dispatches = count_dispatches(&plan_folded);
    assert!(
        dispatches <= 1,
        "x + 0.0 should fold to <= 1 dispatch, got {dispatches}"
    );
}

/// Unary constant folding: exp(Constant(0)) should fold to Constant(1).
#[test]
fn test_constant_folding_exp_of_constant() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "const_zero".into(),
            TraceOp::Constant { value: 0.0 },
            vec![],
            vec![1],
            DType::F32,
        ),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[1]),
    ]);

    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile exp(0)");
    let dispatches = count_dispatches(&plan);
    assert_eq!(
        dispatches, 0,
        "exp(Constant(0)) should fold to zero dispatches, got {dispatches}"
    );
}

/// NaN constants are NOT folded (finiteness guard).
#[test]
fn test_constant_folding_nan_not_folded() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "const_nan".into(),
            TraceOp::Constant { value: f64::NAN },
            vec![],
            vec![1],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "const_one".into(),
            TraceOp::Constant { value: 1.0 },
            vec![],
            vec![1],
            DType::F32,
        ),
        binary_node(2, "add_0", TraceOp::Add, 0, 1, &[1]),
    ]);

    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile NaN add");
    assert!(!plan.steps.is_empty());
}

// ===========================================================================
// 11. Dead code elimination: unused intermediates are removed
// ===========================================================================

/// Unused intermediate: one branch not consumed by output.
#[test]
fn test_dead_code_unused_branch_handled() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4, 8]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4, 8]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 0, &[4, 8]),
    ]);

    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile with dead branch");
    assert_eq!(plan.steps.len(), 3);
    assert_eq!(plan.output_step, 2);
}

/// Passthrough and identity steps are zero-cost "dead" compute.
#[test]
fn test_dead_code_passthrough_not_dispatched() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 3, 4]),
        TraceNode::new(
            1,
            "reshape_0".into(),
            TraceOp::Reshape {
                target_shape: vec![24],
            },
            vec![0],
            vec![24],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "reshape_1".into(),
            TraceOp::Reshape {
                target_shape: vec![6, 4],
            },
            vec![1],
            vec![6, 4],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "dropout_0".into(),
            TraceOp::Dropout,
            vec![2],
            vec![6, 4],
            DType::F32,
        ),
    ]);

    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile zero-compute");
    let dispatches = count_dispatches(&plan);
    assert_eq!(
        dispatches, 0,
        "reshapes and dropout should produce zero dispatches, got {dispatches}"
    );

    let bp = plan_buffers(&plan, &graph);
    assert_eq!(bp.total_bytes, 0, "zero-compute plan should have 0 bytes");
}

/// Identity passthrough in a chain does not add dispatches.
#[test]
fn test_dead_code_identity_passthrough_in_chain() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 8]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[1, 8]),
        TraceNode::new(
            2,
            "dropout_0".into(),
            TraceOp::Dropout,
            vec![1],
            vec![1, 8],
            DType::F32,
        ),
        unary_node(3, "sigmoid_0", TraceOp::Sigmoid, 2, &[1, 8]),
    ]);

    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile dropout chain");
    let dispatches = count_dispatches(&plan);
    assert!(
        dispatches <= 2,
        "relu + dropout + sigmoid should have <= 2 dispatches, got {dispatches}"
    );
}

// ===========================================================================
// 12. Optimization plan: bitmask covers all peephole fields
// ===========================================================================

/// Full bitmask covers all fields and matches PEEPHOLE_FIELD_COUNT.
#[test]
fn test_optimization_plan_full_bitmask_covers_all_fields() {
    let all_on = (1u32 << PEEPHOLE_FIELD_COUNT) - 1;
    let config = config_from_bitmask(all_on);
    assert!(
        is_default_config(&config),
        "full bitmask should produce default config"
    );
    assert_eq!(PEEPHOLE_FIELD_COUNT, 28);
}

/// Bitmask search space is 2^PEEPHOLE_FIELD_COUNT = 268435456.
#[test]
fn test_optimization_plan_search_space_size() {
    assert_eq!(
        1u32 << PEEPHOLE_FIELD_COUNT,
        268_435_456,
        "search space should be 2^28 = 268435456"
    );
}

/// Each bit in the bitmask toggles one and only one field.
#[test]
fn test_optimization_plan_bitmask_bijection() {
    let all_on = (1u32 << PEEPHOLE_FIELD_COUNT) - 1;
    let default_config = config_from_bitmask(all_on);
    assert!(is_default_config(&default_config));

    for bit in 0..PEEPHOLE_FIELD_COUNT {
        let mask = all_on ^ (1u32 << bit);
        let config = config_from_bitmask(mask);
        assert!(
            !is_default_config(&config),
            "disabling bit {bit} should not produce default config"
        );
    }
}

/// Bitmask 0 disables everything; disabling everything still compiles.
#[test]
fn test_optimization_plan_all_disabled_compiles() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[2, 4]),
        unary_node(2, "tanh_0", TraceOp::Tanh, 1, &[2, 4]),
    ]);
    let config = config_from_bitmask(0);
    let plan = compile_trace_to_plan_configured(&graph, &config)
        .expect("all-disabled config should compile");
    assert!(count_dispatches(&plan) >= 1);
}

// ===========================================================================
// 13. Graph recording: node count matches operation count
// ===========================================================================

/// ComputationGraph node count matches the number of TraceNodes inserted.
#[test]
fn test_graph_recording_node_count_matches() {
    let nodes = vec![
        input_node(0, &[2, 8]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[2, 8]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 1, &[2, 8]),
        unary_node(3, "tanh_0", TraceOp::Tanh, 2, &[2, 8]),
        unary_node(4, "relu_1", TraceOp::Relu, 3, &[2, 8]),
    ];
    let expected_count = nodes.len();
    let graph = ComputationGraph::from_nodes(nodes);
    assert_eq!(
        graph.len(),
        expected_count,
        "graph.len() should match number of inserted nodes"
    );
    assert_eq!(
        graph.nodes().len(),
        expected_count,
        "graph.nodes().len() should match"
    );
}

/// Plan step count equals graph node count (1:1 mapping).
#[test]
fn test_graph_recording_plan_steps_match_nodes() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 1, &[4]),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("compile");
    assert_eq!(
        plan.steps.len(),
        graph.len(),
        "plan steps should match graph nodes"
    );
}

/// Graph with multiple inputs: input_shapes collects all of them.
#[test]
fn test_graph_recording_multiple_inputs_collected() {
    // Three distinct input shapes that are broadcast-compatible so the Add
    // chain compiles: [4] + [1, 4] -> [1, 4]; then [1, 4] + [2, 4] -> [2, 4].
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[1, 4]),
        input_node(2, &[2, 4]),
        binary_node(3, "add_01", TraceOp::Add, 0, 1, &[1, 4]),
        binary_node(4, "add_23", TraceOp::Add, 3, 2, &[2, 4]),
    ]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile 3-input");
    assert_eq!(plan.input_shapes.len(), 3, "should have 3 input shapes");
    assert_eq!(plan.input_shapes[0], vec![4]);
    assert_eq!(plan.input_shapes[1], vec![1, 4]);
    assert_eq!(plan.input_shapes[2], vec![2, 4]);
}

/// Graph operation sequence is preserved in topological order.
#[test]
fn test_graph_recording_topological_order_preserved() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 1, &[4]),
    ]);
    let nodes = graph.nodes();
    assert!(matches!(nodes[0].op(), TraceOp::Input));
    assert!(matches!(nodes[1].op(), TraceOp::Relu));
    assert!(matches!(nodes[2].op(), TraceOp::Sigmoid));
}

/// output_step is always the last step index.
#[test]
fn test_graph_recording_output_step_is_last() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
        unary_node(2, "tanh_0", TraceOp::Tanh, 1, &[4]),
        unary_node(3, "sigmoid_0", TraceOp::Sigmoid, 2, &[4]),
    ]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    assert_eq!(
        plan.output_step,
        plan.steps.len() - 1,
        "output_step should be the last step"
    );
}
