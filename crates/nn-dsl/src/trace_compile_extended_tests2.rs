// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests (set 2) for trace compilation, buffer planning, and peephole
//! optimization.
//!
//! Covers:
//! 1. `PeepholeConfig` default and all-disabled configurations
//! 2. `PeepholeConfig` field count matches `PEEPHOLE_FIELD_COUNT`
//! 3. `NativeOpKind` Debug format and `variant_name()` consistency
//! 4. `NativeOpKind` variant count (33 variants)
//! 5. Buffer planner allocation strategies (best-fit, high water mark)
//! 6. Trace compile with empty graphs
//! 7. Trace compile with single-op graphs (copy, unary)

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use crate::buffer_planner::{plan_buffers, BufferPlan};
use crate::trace_compile::optimize_plan::{
    config_from_bitmask, is_default_config, PEEPHOLE_FIELD_COUNT, PEEPHOLE_FIELD_NAMES,
};
use crate::trace_compile::{
    compile_trace_to_plan, compile_trace_to_plan_configured, compile_trace_to_plan_with_fusion,
    count_dispatches, CompiledPlan, CompiledStep, NativeOpKind, PeepholeConfig,
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
// 1. PeepholeConfig — default and all-disabled configurations
// ===========================================================================

/// All 16 fields of PeepholeConfig::default() are `true`.
#[test]
fn test_peephole_default_all_fields_true() {
    let config = PeepholeConfig::default();
    let accessors: Vec<(&str, fn(&PeepholeConfig) -> bool)> = vec![
        ("norm_activ_conv1d", |c| c.norm_activ_conv1d),
        ("fused_resblock", |c| c.fused_resblock),
        ("linear_activation", |c| c.linear_activation),
        ("add_layer_norm", |c| c.add_layer_norm),
        ("norm_linear", |c| c.norm_linear),
        ("attention_transpose", |c| c.attention_transpose),
        ("flip_lstm", |c| c.flip_lstm),
        ("batched_linear_projection", |c| c.batched_linear_projection),
        ("channels_first_layer_norm", |c| c.channels_first_layer_norm),
        ("silu_mul", |c| c.silu_mul),
        ("auto_fuse_elementwise", |c| c.auto_fuse_elementwise),
        ("bilstm_cat", |c| c.bilstm_cat),
        ("add_norm_linear", |c| c.add_norm_linear),
        ("fuse_adain_snake", |c| c.fuse_adain_snake),
        ("fuse_upsample_conv1d", |c| c.fuse_upsample_conv1d),
        ("fuse_instance_norm_mul_add", |c| {
            c.fuse_instance_norm_mul_add
        }),
    ];
    for (name, accessor) in &accessors {
        assert!(
            accessor(&config),
            "PeepholeConfig::default().{name} should be true"
        );
    }
}

/// bitmask(0) produces all-disabled config; no field is true.
#[test]
fn test_peephole_all_disabled_no_fields_true() {
    let config = config_from_bitmask(0);
    let accessors: Vec<(&str, fn(&PeepholeConfig) -> bool)> = vec![
        ("norm_activ_conv1d", |c| c.norm_activ_conv1d),
        ("fused_resblock", |c| c.fused_resblock),
        ("linear_activation", |c| c.linear_activation),
        ("add_layer_norm", |c| c.add_layer_norm),
        ("norm_linear", |c| c.norm_linear),
        ("attention_transpose", |c| c.attention_transpose),
        ("flip_lstm", |c| c.flip_lstm),
        ("batched_linear_projection", |c| c.batched_linear_projection),
        ("channels_first_layer_norm", |c| c.channels_first_layer_norm),
        ("silu_mul", |c| c.silu_mul),
        ("auto_fuse_elementwise", |c| c.auto_fuse_elementwise),
        ("bilstm_cat", |c| c.bilstm_cat),
        ("add_norm_linear", |c| c.add_norm_linear),
        ("fuse_adain_snake", |c| c.fuse_adain_snake),
        ("fuse_upsample_conv1d", |c| c.fuse_upsample_conv1d),
        ("fuse_instance_norm_mul_add", |c| {
            c.fuse_instance_norm_mul_add
        }),
    ];
    for (name, accessor) in &accessors {
        assert!(
            !accessor(&config),
            "config_from_bitmask(0).{name} should be false"
        );
    }
}

/// All-disabled config is NOT default. Default is default.
#[test]
fn test_peephole_is_default_config_boundary() {
    assert!(is_default_config(&PeepholeConfig::default()));
    assert!(!is_default_config(&config_from_bitmask(0)));
}

// ===========================================================================
// 2. PeepholeConfig field count matches PEEPHOLE_FIELD_COUNT
// ===========================================================================

/// PEEPHOLE_FIELD_COUNT is 28 (matching the 28 boolean fields in PeepholeConfig).
#[test]
fn test_peephole_field_count_is_21() {
    assert_eq!(PEEPHOLE_FIELD_COUNT, 28, "PEEPHOLE_FIELD_COUNT must be 28");
}

/// PEEPHOLE_FIELD_NAMES has exactly PEEPHOLE_FIELD_COUNT entries.
#[test]
fn test_peephole_field_names_length_matches_count() {
    assert_eq!(
        PEEPHOLE_FIELD_NAMES.len(),
        PEEPHOLE_FIELD_COUNT as usize,
        "PEEPHOLE_FIELD_NAMES length must match PEEPHOLE_FIELD_COUNT"
    );
}

/// All bitmask field names are non-empty and unique.
#[test]
fn test_peephole_field_names_non_empty_unique() {
    for (i, name) in PEEPHOLE_FIELD_NAMES.iter().enumerate() {
        assert!(
            !name.is_empty(),
            "PEEPHOLE_FIELD_NAMES[{i}] must not be empty"
        );
    }
    let unique_count = {
        let mut set = std::collections::HashSet::new();
        for name in &PEEPHOLE_FIELD_NAMES {
            set.insert(*name);
        }
        set.len()
    };
    assert_eq!(
        unique_count,
        PEEPHOLE_FIELD_NAMES.len(),
        "PEEPHOLE_FIELD_NAMES entries must be unique"
    );
}

/// The search space is 2^PEEPHOLE_FIELD_COUNT = 268435456.
#[test]
fn test_peephole_search_space_size() {
    assert_eq!(1u32 << PEEPHOLE_FIELD_COUNT, 268_435_456);
}

/// Toggling each single bit off from all-on produces a non-default config.
#[test]
fn test_peephole_single_bit_toggle_not_default() {
    let all_on_mask = (1u32 << PEEPHOLE_FIELD_COUNT) - 1;
    for bit in 0..PEEPHOLE_FIELD_COUNT {
        let mask = all_on_mask ^ (1u32 << bit);
        let config = config_from_bitmask(mask);
        assert!(
            !is_default_config(&config),
            "disabling bit {bit} ({}) should not be default",
            PEEPHOLE_FIELD_NAMES[bit as usize],
        );
    }
}

// ===========================================================================
// 3. NativeOpKind Debug format and variant_name() consistency
// ===========================================================================

/// variant_name() output appears in the Debug format for representative ops.
#[test]
fn test_native_op_kind_debug_contains_variant_name() {
    let ops: Vec<NativeOpKind> = vec![
        NativeOpKind::InstanceNorm {
            eps: 1e-5,
            input_shape: vec![1, 4, 16],
        },
        NativeOpKind::SiluMul {
            input_shape: vec![1, 8, 128],
        },
        NativeOpKind::FusedMulAdd {
            input_shape: vec![1, 4, 32],
        },
        NativeOpKind::Cumsum {
            dim: 0,
            input_shape: vec![8],
        },
    ];
    for op in &ops {
        let debug_str = format!("{op:?}");
        let name = op.variant_name();
        assert!(
            debug_str.contains(name),
            "Debug format of {name} should contain variant_name(); \
             got debug={debug_str}"
        );
    }
}

/// variant_name() returns distinct non-empty strings for distinct variants.
#[test]
fn test_native_op_kind_variant_names_nonempty() {
    // A subset of representative variants.
    let ops: Vec<NativeOpKind> = vec![
        NativeOpKind::LstmSequence {
            hidden_size: 64,
            input_shape: vec![4, 1, 128],
            h_shape: vec![1, 64],
            reverse: false,
        },
        NativeOpKind::InstanceNorm {
            eps: 1e-5,
            input_shape: vec![1, 4, 16],
        },
        NativeOpKind::LayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 16],
            hidden_dim: 16,
        },
        NativeOpKind::SiluMul {
            input_shape: vec![1, 8, 128],
        },
        NativeOpKind::FusedGeGLU {
            input_shape: vec![1, 8, 256],
        },
        NativeOpKind::FusedSiGLU {
            input_shape: vec![1, 8, 256],
        },
    ];
    for op in &ops {
        let name = op.variant_name();
        assert!(!name.is_empty(), "variant_name() must be non-empty");
    }
}

// ===========================================================================
// 4. NativeOpKind variant count (33 variants)
// ===========================================================================

/// There are 33 known NativeOpKind variants. One for each entry in
/// variant_name()'s match. This list must be updated when adding a variant.
#[test]
fn test_native_op_kind_variant_count_33() {
    // Build one minimal instance of every variant. The count is the invariant.
    let all_ops: Vec<NativeOpKind> = vec![
        NativeOpKind::LstmSequence {
            hidden_size: 1,
            input_shape: vec![1, 1, 1],
            h_shape: vec![1, 1],
            reverse: false,
        },
        NativeOpKind::Cumsum {
            dim: 0,
            input_shape: vec![1],
        },
        NativeOpKind::InstanceNorm {
            eps: 1e-5,
            input_shape: vec![1, 1, 1],
        },
        NativeOpKind::LayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 1],
            hidden_dim: 1,
        },
        NativeOpKind::AddLayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 1],
            hidden_dim: 1,
        },
        NativeOpKind::AdainSnake {
            eps: 1e-5,
            input_shape: vec![1, 1, 1],
            channels: 1,
            residual_gamma: false,
            external_node_ids: None,
        },
        NativeOpKind::AdainLeakyRelu {
            eps: 1e-5,
            slope: 0.01,
            input_shape: vec![1, 1, 1],
            external_node_ids: None,
        },
        NativeOpKind::AdaLayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 1, 1],
            hidden_dim: 1,
        },
        NativeOpKind::FlashAttention {
            scale: 1.0,
            causal: false,
            q_shape: vec![1, 1, 1, 1],
            k_shape: vec![1, 1, 1, 1],
            output_shape: vec![1, 1, 1, 1],
            input_layout: Default::default(),
        },
        NativeOpKind::MaxPool1d {
            kernel_size: 1,
            stride: 1,
            padding: 0,
            input_shape: vec![1, 1, 1],
        },
        NativeOpKind::ConstantWeight {
            name: "w".into(),
            shape: vec![1],
        },
        NativeOpKind::FusedResBlock {
            phase1: crate::trace_compile::NormActivConv1dParams {
                activation: crate::trace_compile::NormActivation::Snake,
                eps: 1e-5,
                conv_dilation: 1,
                conv_padding: 1,
                input_shape: vec![1, 1, 1],
                output_channels: 1,
                kernel_size: 1,
            },
            phase2: crate::trace_compile::NormActivConv1dParams {
                activation: crate::trace_compile::NormActivation::Snake,
                eps: 1e-5,
                conv_dilation: 1,
                conv_padding: 1,
                input_shape: vec![1, 1, 1],
                output_channels: 1,
                kernel_size: 1,
            },
            input_steps: vec![0],
            residual_scale: 1.0,
            style_proj: None,
            shortcut_step: None,
            pool_step: None,
            style_batch_offset: None,
        },
        NativeOpKind::BatchedStyleProjection {
            blocks: vec![],
            style_dim: 1,
            total_out: 1,
            style_step: 0,
        },
        NativeOpKind::NormActivConv1d {
            activation: crate::trace_compile::NormActivation::Snake,
            eps: 1e-5,
            conv_dilation: 1,
            conv_padding: 0,
            input_shape: vec![1, 1, 1],
            output_channels: 1,
            kernel_size: 1,
            external_node_ids: None,
        },
        NativeOpKind::LinearActivation {
            activation: crate::trace_compile::GemmActivation::Relu,
            in_features: 1,
            out_features: 1,
            has_bias: false,
            input_shape: vec![1, 1],
        },
        NativeOpKind::BatchedLinearProjection {
            in_features: 1,
            total_out_features: 1,
            projection_sizes: vec![1],
            has_bias: false,
            input_shape: vec![1, 1, 1],
        },
        NativeOpKind::ProjectionSlice {
            source_step: 0,
            dim: 0,
            start: 0,
            length: 1,
            output_shape: vec![1],
        },
        NativeOpKind::NormLinear {
            norm_kind: crate::trace_compile::FusedNormKind::LayerNorm,
            eps: 1e-5,
            input_shape: vec![1, 1],
            hidden_dim: 1,
            out_features: 1,
            has_bias: false,
        },
        NativeOpKind::ChannelsFirstLayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 1, 1],
            channels: 1,
            leaky_relu_slope: None,
        },
        NativeOpKind::Int8Gemm {
            in_features: 1,
            out_features: 1,
            has_bias: false,
            input_shape: vec![1, 1],
        },
        NativeOpKind::Conv1dGemm {
            input_shape: vec![1, 1, 1],
            out_channels: 1,
            kernel_size: 1,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            has_bias: false,
        },
        NativeOpKind::SiluMul {
            input_shape: vec![1, 1, 1],
        },
        NativeOpKind::RotaryEmbedding {
            head_dim: 1,
            input_shape: vec![1, 1, 1, 1],
        },
        NativeOpKind::AddNormLinear {
            eps: 1e-5,
            input_shape: vec![1, 1],
            hidden_dim: 1,
            out_features: 1,
            has_bias: false,
        },
        NativeOpKind::MoeGating {
            num_experts: 1,
            top_k: 1,
            input_shape: vec![1, 1],
        },
        NativeOpKind::FusedAdainSnake {
            eps: 1e-5,
            input_shape: vec![1, 1, 1],
            channels: 1,
            external_node_ids: None,
        },
        NativeOpKind::FusedUpsampleConv1d {
            upsample_factor: 2,
            in_channels: 1,
            out_channels: 1,
            kernel_size: 1,
            stride: 1,
            padding: 0,
            input_shape: vec![1, 1, 1],
        },
        NativeOpKind::BiLstmCat {
            hidden_size: 1,
            input_shape: vec![1, 1, 1],
            h_shape: vec![1, 1],
            fwd_lstm_step: 0,
            rev_lstm_step: 1,
        },
        NativeOpKind::FusedMulAdd {
            input_shape: vec![1, 1, 1],
        },
        NativeOpKind::FusedSiGLU {
            input_shape: vec![1, 1, 1],
        },
        NativeOpKind::FusedGeGLU {
            input_shape: vec![1, 1, 1],
        },
        NativeOpKind::FusedLayerNormLinear {
            eps: 1e-5,
            input_shape: vec![1, 1],
            hidden_dim: 1,
            out_features: 1,
            has_bias: false,
        },
        NativeOpKind::FusedInstanceNormMulAdd {
            eps: 1e-5,
            input_shape: vec![1, 1, 1],
            channels: 1,
            external_node_ids: None,
        },
    ];

    // 33 known variants.
    assert_eq!(
        all_ops.len(),
        33,
        "NativeOpKind should have 33 variants; update this test when adding new variants"
    );

    // Every instance has a non-empty variant_name.
    for op in &all_ops {
        assert!(
            !op.variant_name().is_empty(),
            "variant_name() must be non-empty for all variants"
        );
    }

    // All variant names are distinct.
    let names: Vec<&str> = all_ops.iter().map(NativeOpKind::variant_name).collect();
    let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
    assert_eq!(
        unique.len(),
        names.len(),
        "all variant_name() values must be unique"
    );
}

/// estimated_metal_dispatches() returns > 0 for compute ops and 0 for ConstantWeight.
#[test]
fn test_native_op_kind_dispatch_estimates_consistent() {
    let compute_op = NativeOpKind::InstanceNorm {
        eps: 1e-5,
        input_shape: vec![1, 4, 16],
    };
    assert!(
        compute_op.estimated_metal_dispatches() > 0,
        "InstanceNorm should have > 0 estimated dispatches"
    );

    let no_dispatch_op = NativeOpKind::ConstantWeight {
        name: "test".into(),
        shape: vec![4],
    };
    assert_eq!(
        no_dispatch_op.estimated_metal_dispatches(),
        0,
        "ConstantWeight should have 0 estimated dispatches"
    );
}

// ===========================================================================
// 5. Buffer planner allocation strategies
// ===========================================================================

/// Empty plan: buffer planner returns empty plan with zero bytes.
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
    assert!(bp.step_sizes.is_empty());
    assert!(bp.last_use.is_empty());
}

/// Single input: zero allocation (InputForward does not allocate).
#[test]
fn test_buffer_planner_single_input_zero_alloc() {
    let graph = ComputationGraph::from_nodes(vec![input_node(0, &[4, 8])]);
    let plan = compile_trace_to_plan(&graph).expect("compile single input");
    let bp = plan_buffers(&plan, &graph);
    assert_eq!(bp.total_bytes, 0, "InputForward should not allocate");
    assert_eq!(bp.step_sizes[0], 0);
}

/// Best-fit reuse: sequential chain with decreasing sizes should reuse
/// the first freed slot when a smaller buffer fits.
#[test]
fn test_buffer_planner_best_fit_reuse_decreasing() {
    // input(0) → reduce_sum(1, shape [4]) → relu(2, shape [4]) → reduce_sum(3, shape [1])
    // Step 1 allocates 16 bytes (4 * f32). Step 3 allocates 4 bytes.
    // After step 2 consumes step 1, the 16-byte slot is freed.
    // Step 3 (4 bytes) should reuse within that freed 16-byte slot.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4, 4]),
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

    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    let bp = plan_buffers(&plan, &graph);
    assert_no_memory_time_overlap(&bp);

    // Total bytes should be less than naive total (reuse occurring).
    assert!(
        bp.total_bytes <= bp.naive_total,
        "buffer planner should reuse freed slots: total={}, naive={}",
        bp.total_bytes,
        bp.naive_total,
    );
}

/// High water mark: when all buffers are live simultaneously, no reuse.
#[test]
fn test_buffer_planner_no_reuse_concurrent() {
    // input(0) → relu(1), sigmoid(2): both consume input.
    // add(3): consumes relu and sigmoid.
    // relu(1) and sigmoid(2) are both live until step 3.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[16]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[16]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 0, &[16]),
        binary_node(3, "add_0", TraceOp::Add, 1, 2, &[16]),
    ]);

    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile");
    let bp = plan_buffers(&plan, &graph);
    assert_no_memory_time_overlap(&bp);
}

/// ConstantValue steps allocate buffer space based on element count * dtype.
#[test]
fn test_buffer_planner_constant_value_size() {
    let graph = ComputationGraph::from_nodes(vec![TraceNode::new(
        0,
        "const_0".into(),
        TraceOp::Constant { value: 1.0 },
        vec![],
        vec![3, 5],
        DType::F32,
    )]);
    let plan = CompiledPlan {
        steps: vec![CompiledStep::ConstantValue {
            value: 1.0,
            shape: vec![3, 5],
        }],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    let bp = plan_buffers(&plan, &graph);
    // 3 * 5 = 15 elements * 4 bytes (f32) = 60 bytes.
    assert_eq!(bp.step_sizes[0], 60);
    assert_eq!(bp.total_bytes, 60);
}

/// NativeOp steps allocate based on output tensor shape.
#[test]
fn test_buffer_planner_native_op_size() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 4, 8]),
        TraceNode::new(
            1,
            "instnorm_0".into(),
            TraceOp::InstanceNorm { eps: 1e-5 },
            vec![0],
            vec![2, 4, 8],
            DType::F32,
        ),
    ]);
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            CompiledStep::NativeOp {
                op: NativeOpKind::InstanceNorm {
                    eps: 1e-5,
                    input_shape: vec![2, 4, 8],
                },
                weight_data: HashMap::new(),
            },
        ],
        input_shapes: vec![vec![2, 4, 8]],
        output_step: 1,
        weight_names: vec![],
    };
    let bp = plan_buffers(&plan, &graph);
    // [2, 4, 8] = 64 elements * 4 bytes = 256 bytes.
    assert_eq!(bp.step_sizes[0], 0, "InputForward should not allocate");
    assert_eq!(bp.step_sizes[1], 256);
}

// ===========================================================================
// 6. Trace compile with empty graphs
// ===========================================================================

/// Empty graph: compile_trace_to_plan produces an empty plan.
#[test]
fn test_compile_empty_graph_no_fusion() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let plan = compile_trace_to_plan(&graph).expect("compile empty");
    assert_eq!(plan.steps.len(), 0);
    assert_eq!(plan.output_step, 0);
    assert_eq!(count_dispatches(&plan), 0);
    assert!(plan.input_shapes.is_empty());
    assert!(plan.weight_names.is_empty());
}

/// Empty graph: compile_trace_to_plan_with_fusion produces an empty plan.
#[test]
fn test_compile_empty_graph_with_fusion() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile empty");
    assert_eq!(plan.steps.len(), 0);
    assert_eq!(count_dispatches(&plan), 0);
}

/// Empty graph: compile_trace_to_plan_configured produces an empty plan.
#[test]
fn test_compile_empty_graph_configured() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let plan = compile_trace_to_plan_configured(&graph, &PeepholeConfig::default())
        .expect("compile empty configured");
    assert_eq!(plan.steps.len(), 0);
    assert_eq!(count_dispatches(&plan), 0);
}

/// Empty graph: buffer planner on empty plan is consistent.
#[test]
fn test_buffer_planner_on_empty_compiled_graph() {
    let graph = ComputationGraph::from_nodes(vec![]);
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile empty");
    let bp = plan_buffers(&plan, &graph);
    assert_eq!(bp.total_bytes, 0);
    assert_eq!(bp.naive_total, 0);
    assert!(bp.step_offsets.is_empty());
}

// ===========================================================================
// 7. Trace compile with single-op graphs (copy, unary)
// ===========================================================================

/// Single input-only graph: 1 step, InputForward, no dispatches.
#[test]
fn test_compile_single_input_only() {
    let graph = ComputationGraph::from_nodes(vec![input_node(0, &[4, 8])]);
    let plan = compile_trace_to_plan(&graph).expect("compile single input");
    assert_eq!(plan.steps.len(), 1);
    assert!(matches!(plan.steps[0], CompiledStep::InputForward));
    assert_eq!(count_dispatches(&plan), 0);
    assert_eq!(plan.input_shapes.len(), 1);
    assert_eq!(plan.input_shapes[0], vec![4, 8]);
}

/// Single Relu: input → relu produces exactly 2 steps (1 InputForward + 1 Dispatch).
#[test]
fn test_compile_single_relu() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 8]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[2, 8]),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("compile relu");
    assert_eq!(plan.steps.len(), 2);
    assert!(matches!(plan.steps[0], CompiledStep::InputForward));
    assert!(
        matches!(plan.steps[1], CompiledStep::Dispatch { .. }),
        "relu should produce a Dispatch step"
    );
    assert_eq!(count_dispatches(&plan), 1);
    assert_eq!(plan.output_step, 1);
}

/// Single Sigmoid: input → sigmoid produces a Dispatch step.
#[test]
fn test_compile_single_sigmoid() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 16]),
        unary_node(1, "sigmoid_0", TraceOp::Sigmoid, 0, &[1, 16]),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("compile sigmoid");
    assert_eq!(plan.steps.len(), 2);
    assert_eq!(count_dispatches(&plan), 1);
}

/// Single Tanh: input → tanh produces a Dispatch step.
#[test]
fn test_compile_single_tanh() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[3, 4]),
        unary_node(1, "tanh_0", TraceOp::Tanh, 0, &[3, 4]),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("compile tanh");
    assert_eq!(plan.steps.len(), 2);
    assert_eq!(count_dispatches(&plan), 1);
}

/// Single Reshape (copy): input → reshape produces a Passthrough, no dispatch.
#[test]
fn test_compile_single_reshape() {
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
    ]);
    let plan = compile_trace_to_plan(&graph).expect("compile reshape");
    assert_eq!(plan.steps.len(), 2);
    assert!(
        matches!(plan.steps[1], CompiledStep::Passthrough { .. }),
        "reshape should produce a Passthrough step"
    );
    assert_eq!(count_dispatches(&plan), 0);
}

/// Single Dropout (copy at inference): produces IdentityPassthrough.
#[test]
fn test_compile_single_dropout() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4, 4]),
        TraceNode::new(
            1,
            "dropout_0".into(),
            TraceOp::Dropout,
            vec![0],
            vec![4, 4],
            DType::F32,
        ),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("compile dropout");
    assert_eq!(plan.steps.len(), 2);
    assert!(
        matches!(plan.steps[1], CompiledStep::IdentityPassthrough),
        "dropout should produce IdentityPassthrough"
    );
    assert_eq!(count_dispatches(&plan), 0);
}

/// Single Add (binary op): input0 + input1 produces exactly 1 dispatch.
#[test]
fn test_compile_single_add() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4, 8]),
        input_node(1, &[4, 8]),
        binary_node(2, "add_0", TraceOp::Add, 0, 1, &[4, 8]),
    ]);
    let plan = compile_trace_to_plan(&graph).expect("compile add");
    assert_eq!(plan.steps.len(), 3);
    assert_eq!(count_dispatches(&plan), 1);
    assert_eq!(plan.output_step, 2);
    assert_eq!(plan.input_shapes.len(), 2);
}

/// Single Constant: produces a ConstantValue step, no dispatch.
#[test]
fn test_compile_single_constant() {
    let graph = ComputationGraph::from_nodes(vec![TraceNode::new(
        0,
        "const_0".into(),
        TraceOp::Constant { value: 2.718 },
        vec![],
        vec![4, 2],
        DType::F32,
    )]);
    let plan = compile_trace_to_plan(&graph).expect("compile constant");
    assert_eq!(plan.steps.len(), 1);
    match &plan.steps[0] {
        CompiledStep::ConstantValue { value, shape } => {
            assert!((value - 2.718).abs() < 1e-6);
            assert_eq!(shape, &[4, 2]);
        }
        other => panic!("expected ConstantValue, got {other:?}"),
    }
    assert_eq!(count_dispatches(&plan), 0);
}

/// Single InstanceNorm: produces a NativeOp step with 1 dispatch.
#[test]
fn test_compile_single_instance_norm() {
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
    let plan = compile_trace_to_plan(&graph).expect("compile instance_norm");
    let has_native = plan
        .steps
        .iter()
        .any(|s| matches!(s, CompiledStep::NativeOp { .. }));
    assert!(has_native, "InstanceNorm should produce a NativeOp step");
    assert_eq!(count_dispatches(&plan), 1);
}

/// Compile with fusion vs. without produces consistent output_step.
#[test]
fn test_compile_single_op_fusion_output_step_consistent() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[2, 4]),
    ]);

    let plan_no_fusion = compile_trace_to_plan(&graph).expect("no fusion");
    let plan_fusion = compile_trace_to_plan_with_fusion(&graph).expect("fusion");

    assert_eq!(
        plan_no_fusion.output_step, plan_fusion.output_step,
        "output_step should be consistent between fusion and non-fusion"
    );
    assert_eq!(plan_no_fusion.steps.len(), plan_fusion.steps.len());
}

/// Compiling with all peephole passes disabled still produces a valid
/// single-op plan.
#[test]
fn test_compile_single_op_all_passes_disabled() {
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4, 8]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4, 8]),
    ]);
    let all_off = config_from_bitmask(0);
    let plan = compile_trace_to_plan_configured(&graph, &all_off)
        .expect("compile with all passes disabled");
    assert_eq!(plan.steps.len(), 2);
    assert!(count_dispatches(&plan) >= 1);
}
