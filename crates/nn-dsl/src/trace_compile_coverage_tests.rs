// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Trace compilation pipeline coverage tests.
//!
//! End-to-end coverage for the trace → graph → peephole → native ops →
//! buffer plan → execution plan pipeline. Part of #4186.

use std::collections::HashMap;

use nn_core::DType;

use crate::buffer_planner::{plan_buffers, plan_buffers_with_dtypes};
use crate::ir::ScalarType;
use crate::trace_compile::optimize_plan::PEEPHOLE_FIELD_COUNT;
use crate::trace_compile::{
    count_dispatches, CompiledPlan, CompiledStep, NativeOpKind, PeepholeConfig, RuntimeOpKind,
};

// ---------------------------------------------------------------------------
// 1. CompiledPlan structural tests
// ---------------------------------------------------------------------------

#[test]
fn test_empty_plan_has_zero_dispatches() {
    let plan = CompiledPlan {
        steps: vec![],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    assert_eq!(count_dispatches(&plan), 0);
}

#[test]
fn test_plan_with_only_passthrough_steps_has_zero_dispatches() {
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            CompiledStep::Passthrough {
                op_name: "reshape".into(),
                output_shape: vec![2, 3],
            },
            CompiledStep::IdentityPassthrough,
        ],
        input_shapes: vec![vec![6]],
        output_step: 2,
        weight_names: vec![],
    };
    assert_eq!(count_dispatches(&plan), 0);
}

#[test]
fn test_plan_with_native_op_counts_as_dispatch() {
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
        ],
        input_shapes: vec![vec![1, 4, 16]],
        output_step: 1,
        weight_names: vec![],
    };
    assert_eq!(count_dispatches(&plan), 1);
}

#[test]
fn test_plan_with_constant_value_does_not_count_as_dispatch() {
    let plan = CompiledPlan {
        steps: vec![CompiledStep::ConstantValue {
            value: 0.0,
            shape: vec![1, 4],
        }],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    assert_eq!(count_dispatches(&plan), 0);
}

#[test]
fn test_plan_with_runtime_op_counts_as_dispatch() {
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            CompiledStep::InputForward,
            CompiledStep::RuntimeOp {
                op: RuntimeOpKind::RepeatInterleave {
                    dim: 1,
                    input_shape: vec![1, 4, 8],
                    counts_shape: vec![4],
                },
            },
        ],
        input_shapes: vec![vec![1, 4, 8], vec![4]],
        output_step: 2,
        weight_names: vec![],
    };
    // `count_dispatches` counts only Dispatch + NativeOp steps; per its
    // documented contract InputForward and RuntimeOp are both excluded
    // (RuntimeOp is also treated as zero-cost by the cost model and buffer
    // planner). This plan has no Dispatch/NativeOp steps, so the count is 0.
    assert_eq!(count_dispatches(&plan), 0);
}

#[test]
fn test_fusion_stats_empty_plan() {
    let plan = CompiledPlan {
        steps: vec![],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    let stats = plan.fusion_stats();
    assert_eq!(stats.fused_chains, 0);
    assert_eq!(stats.fused_ops, 0);
    assert_eq!(stats.dispatches_saved, 0);
}

#[test]
fn test_peephole_stats_empty_plan() {
    let plan = CompiledPlan {
        steps: vec![],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    let stats = plan.peephole_stats();
    assert_eq!(stats.native_ops, 0);
    assert_eq!(stats.native_dispatches, 0);
    assert_eq!(stats.passthrough_count, 0);
}

#[test]
fn test_peephole_stats_counts_native_ops_and_passthrough() {
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
            CompiledStep::NativeOp {
                op: NativeOpKind::SiluMul {
                    input_shape: vec![1, 8, 256],
                },
                weight_data: HashMap::new(),
            },
        ],
        input_shapes: vec![vec![1, 4, 16]],
        output_step: 3,
        weight_names: vec![],
    };
    let stats = plan.peephole_stats();
    assert_eq!(stats.native_ops, 2);
    assert_eq!(stats.passthrough_count, 1);
    assert_eq!(stats.native_dispatches, 2);
    assert_eq!(stats.by_variant.len(), 2);
}

#[test]
fn test_plan_output_step_consistency() {
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            CompiledStep::Passthrough {
                op_name: "relu".into(),
                output_shape: vec![1, 4],
            },
        ],
        input_shapes: vec![vec![1, 4]],
        output_step: 1,
        weight_names: vec![],
    };
    assert!(plan.output_step < plan.steps.len());
}

// ---------------------------------------------------------------------------
// 2. NativeOpKind dispatch count coverage tests
// ---------------------------------------------------------------------------

#[test]
fn test_native_op_instance_norm_dispatch_count() {
    let op = NativeOpKind::InstanceNorm {
        eps: 1e-5,
        input_shape: vec![1, 4, 16],
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
    assert_eq!(op.variant_name(), "InstanceNorm");
}

#[test]
fn test_native_op_layer_norm_dispatch_count() {
    let op = NativeOpKind::LayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 16, 256],
        hidden_dim: 256,
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
    assert_eq!(op.variant_name(), "LayerNorm");
}

#[test]
fn test_native_op_lstm_sequence_dispatch_count() {
    let op = NativeOpKind::LstmSequence {
        hidden_size: 128,
        input_shape: vec![16, 1, 64],
        h_shape: vec![1, 128],
        reverse: false,
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 2);
    assert_eq!(op.variant_name(), "LstmSequence");
}

#[test]
fn test_native_op_cumsum_single_pass() {
    let op = NativeOpKind::Cumsum {
        dim: 0,
        input_shape: vec![128],
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

#[test]
fn test_native_op_cumsum_multi_pass() {
    let op = NativeOpKind::Cumsum {
        dim: 0,
        input_shape: vec![512],
    };
    assert_eq!(op.estimated_metal_dispatches(), 3);
    assert_eq!(op.estimated_encoding_events(), 1);
}

#[test]
fn test_native_op_flash_attention_dispatch_count() {
    let op = NativeOpKind::FlashAttention {
        scale: 0.125,
        causal: true,
        q_shape: vec![1, 8, 64, 64],
        k_shape: vec![1, 8, 64, 64],
        output_shape: vec![1, 8, 64, 64],
        input_layout: Default::default(),
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.variant_name(), "FlashAttention");
}

#[test]
fn test_native_op_constant_weight_zero_dispatches() {
    let op = NativeOpKind::ConstantWeight {
        name: "arange_data".into(),
        shape: vec![256],
    };
    assert_eq!(op.estimated_metal_dispatches(), 0);
    assert_eq!(op.estimated_encoding_events(), 0);
}

#[test]
fn test_native_op_max_pool1d_dispatch_and_encoding() {
    let op = NativeOpKind::MaxPool1d {
        kernel_size: 3,
        stride: 2,
        padding: 1,
        input_shape: vec![1, 64, 128],
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 0);
}

#[test]
fn test_native_op_conv1d_gemm_with_bias() {
    // K=3, stride=1, dilation=1 → direct K=3 path: 1 conv + 1 bias = 2.
    let op = NativeOpKind::Conv1dGemm {
        input_shape: vec![1, 128, 256],
        out_channels: 256,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
        has_bias: true,
    };
    assert_eq!(op.estimated_metal_dispatches(), 2);
    assert_eq!(op.estimated_encoding_events(), 2);
}

#[test]
fn test_native_op_conv1d_gemm_without_bias() {
    // K=3, stride=1, dilation=1 → direct K=3 path: 1 conv dispatch.
    let op = NativeOpKind::Conv1dGemm {
        input_shape: vec![1, 128, 256],
        out_channels: 256,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
        has_bias: false,
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

#[test]
fn test_native_op_silu_mul_single_dispatch() {
    let op = NativeOpKind::SiluMul {
        input_shape: vec![1, 8, 256],
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

#[test]
fn test_native_op_rotary_embedding_dispatch_count() {
    let op = NativeOpKind::RotaryEmbedding {
        head_dim: 64,
        input_shape: vec![1, 8, 16, 64],
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

#[test]
fn test_native_op_bilstm_cat_dispatch_count() {
    let op = NativeOpKind::BiLstmCat {
        hidden_size: 64,
        input_shape: vec![16, 1, 128],
        h_shape: vec![1, 64],
        fwd_lstm_step: 0,
        rev_lstm_step: 1,
    };
    assert_eq!(op.estimated_metal_dispatches(), 3);
    assert_eq!(op.estimated_encoding_events(), 5);
}

#[test]
fn test_native_op_fused_mul_add_dispatch() {
    let op = NativeOpKind::FusedMulAdd {
        input_shape: vec![1, 8, 256],
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

#[test]
fn test_native_op_fused_siglu_dispatch() {
    let op = NativeOpKind::FusedSiGLU {
        input_shape: vec![1, 8, 256],
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

#[test]
fn test_native_op_fused_geglu_dispatch() {
    let op = NativeOpKind::FusedGeGLU {
        input_shape: vec![1, 8, 256],
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

#[test]
fn test_native_op_fused_instance_norm_mul_add_dispatch() {
    let op = NativeOpKind::FusedInstanceNormMulAdd {
        eps: 1e-5,
        input_shape: vec![1, 64, 128],
        channels: 64,
        external_node_ids: None,
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

#[test]
fn test_native_op_fused_upsample_conv1d_dispatch() {
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 2,
        in_channels: 32,
        out_channels: 64,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        input_shape: vec![1, 32, 128],
    };
    // True single-kernel fusion: upsample + conv in one MSL kernel (#4310).
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

#[test]
fn test_native_op_fused_layer_norm_linear_dispatch_small() {
    let op = NativeOpKind::FusedLayerNormLinear {
        eps: 1e-5,
        input_shape: vec![1, 4],
        hidden_dim: 4,
        out_features: 8,
        has_bias: true,
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

#[test]
fn test_native_op_external_node_ids_for_adain_snake() {
    let op = NativeOpKind::AdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 4, 16],
        channels: 4,
        residual_gamma: false,
        external_node_ids: Some(vec![10, 20, 30]),
    };
    assert_eq!(op.external_node_ids(), Some(&[10u64, 20, 30][..]));
}

#[test]
fn test_native_op_external_node_ids_none_for_instance_norm() {
    let op = NativeOpKind::InstanceNorm {
        eps: 1e-5,
        input_shape: vec![1, 4, 16],
    };
    assert_eq!(op.external_node_ids(), None);
}

// ---------------------------------------------------------------------------
// 3. PeepholeConfig coverage tests
// ---------------------------------------------------------------------------

#[test]
fn test_peephole_config_default_all_enabled() {
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

#[test]
fn test_peephole_config_field_count_is_17() {
    assert_eq!(PEEPHOLE_FIELD_COUNT, 28);
}

#[test]
fn test_peephole_config_bitmask_all_zeros_produces_all_disabled() {
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

#[test]
fn test_peephole_config_bitmask_all_ones_produces_default() {
    let all_on_mask = (1u32 << PEEPHOLE_FIELD_COUNT) - 1;
    let config = config_from_bitmask(all_on_mask);
    assert_eq!(config, PeepholeConfig::default());
}

#[test]
fn test_peephole_config_individual_bits_isolation() {
    // Each bit maps to exactly one field; verify isolation.
    let field_accessors: Vec<Box<dyn Fn(&PeepholeConfig) -> bool>> = vec![
        Box::new(|c| c.norm_activ_conv1d),
        Box::new(|c| c.fused_resblock),
        Box::new(|c| c.linear_activation),
        Box::new(|c| c.add_layer_norm),
        Box::new(|c| c.norm_linear),
        Box::new(|c| c.attention_transpose),
        Box::new(|c| c.flip_lstm),
        Box::new(|c| c.batched_linear_projection),
        Box::new(|c| c.channels_first_layer_norm),
        Box::new(|c| c.silu_mul),
        Box::new(|c| c.auto_fuse_elementwise),
        Box::new(|c| c.bilstm_cat),
        Box::new(|c| c.add_norm_linear),
        Box::new(|c| c.fuse_adain_snake),
        Box::new(|c| c.fuse_upsample_conv1d),
        Box::new(|c| c.fuse_instance_norm_mul_add),
        Box::new(|c| c.fuse_conv1d_activation),
        Box::new(|c| c.fuse_snake_instance_norm),
        Box::new(|c| c.fuse_conv1d_snake_norm),
        Box::new(|c| c.fuse_conv1d_snake_norm_resblock),
        Box::new(|c| c.fuse_add_instance_norm_conv1x1),
        Box::new(|c| c.fuse_conv_transpose1d_activation),
        Box::new(|c| c.norm_activ_conv_transpose1d),
        Box::new(|c| c.fuse_instance_norm_conv1d),
        Box::new(|c| c.fuse_conv1d_instance_norm),
        Box::new(|c| c.fuse_linear_layer_norm),
        Box::new(|c| c.fuse_resblock_chain),
        Box::new(|c| c.fuse_activation_conv1d),
    ];

    assert_eq!(
        field_accessors.len(),
        PEEPHOLE_FIELD_COUNT as usize,
        "field_accessors count must match PEEPHOLE_FIELD_COUNT"
    );

    for (bit, accessor) in field_accessors.iter().enumerate() {
        let mask = 1u32 << bit;
        let config = config_from_bitmask(mask);
        assert!(
            accessor(&config),
            "bit {bit} should enable its corresponding field"
        );

        for (other_bit, other_accessor) in field_accessors.iter().enumerate() {
            if other_bit != bit {
                assert!(
                    !other_accessor(&config),
                    "bit {bit} should not enable field at bit {other_bit}"
                );
            }
        }
    }
}

#[test]
fn test_peephole_config_eq_impl() {
    let a = PeepholeConfig::default();
    let b = PeepholeConfig::default();
    assert_eq!(a, b);

    let c = PeepholeConfig {
        silu_mul: false,
        ..Default::default()
    };
    assert_ne!(a, c);
}

#[test]
fn test_peephole_config_clone() {
    let original = PeepholeConfig::default();
    let cloned = original.clone();
    assert_eq!(original, cloned);
}

#[test]
fn test_peephole_config_debug_format() {
    let config = PeepholeConfig::default();
    let debug = format!("{config:?}");
    assert!(debug.contains("PeepholeConfig"));
    assert!(debug.contains("norm_activ_conv1d: true"));
}

// ---------------------------------------------------------------------------
// 4. Buffer planner coverage tests
// ---------------------------------------------------------------------------

#[test]
fn test_buffer_plan_empty_plan() {
    use nn_core::dyn_tensor::trace::ComputationGraph;

    let plan = CompiledPlan {
        steps: vec![],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    let graph = ComputationGraph::from_nodes(vec![]);
    let buffer_plan = plan_buffers(&plan, &graph);
    assert_eq!(buffer_plan.total_bytes, 0);
    assert_eq!(buffer_plan.naive_total, 0);
    assert!(buffer_plan.step_offsets.is_empty());
    assert!(buffer_plan.step_sizes.is_empty());
}

#[test]
fn test_buffer_plan_passthrough_steps_have_zero_size() {
    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};

    let nodes = vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![6],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "reshape".into(),
            TraceOp::Reshape {
                target_shape: vec![2, 3],
            },
            vec![0],
            vec![2, 3],
            DType::F32,
        ),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            CompiledStep::Passthrough {
                op_name: "reshape".into(),
                output_shape: vec![2, 3],
            },
        ],
        input_shapes: vec![vec![6]],
        output_step: 1,
        weight_names: vec![],
    };
    let buffer_plan = plan_buffers(&plan, &graph);
    assert_eq!(buffer_plan.step_sizes[0], 0);
    assert_eq!(buffer_plan.step_sizes[1], 0);
}

#[test]
fn test_buffer_plan_constant_value_has_nonzero_size() {
    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};

    let nodes = vec![TraceNode::new(
        0,
        "const".into(),
        TraceOp::Constant { value: 1.0 },
        vec![],
        vec![4, 8],
        DType::F32,
    )];
    let graph = ComputationGraph::from_nodes(nodes);
    let plan = CompiledPlan {
        steps: vec![CompiledStep::ConstantValue {
            value: 1.0,
            shape: vec![4, 8],
        }],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };
    let buffer_plan = plan_buffers(&plan, &graph);
    // 4 * 8 * 4 bytes (F32) = 128
    assert_eq!(buffer_plan.step_sizes[0], 128);
}

#[test]
fn test_buffer_plan_native_op_instance_norm_size() {
    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};

    let nodes = vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![1, 4, 16],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "instnorm".into(),
            TraceOp::InstanceNorm { eps: 1e-5 },
            vec![0],
            vec![1, 4, 16],
            DType::F32,
        ),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
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
        ],
        input_shapes: vec![vec![1, 4, 16]],
        output_step: 1,
        weight_names: vec![],
    };
    let buffer_plan = plan_buffers(&plan, &graph);
    // [1,4,16] = 64 elements * 4 bytes = 256
    assert_eq!(buffer_plan.step_sizes[1], 256);
}

#[test]
fn test_buffer_plan_identity_passthrough_zero_size() {
    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};

    let nodes = vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![8],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "dropout".into(),
            TraceOp::Dropout,
            vec![0],
            vec![8],
            DType::F32,
        ),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            CompiledStep::IdentityPassthrough,
        ],
        input_shapes: vec![vec![8]],
        output_step: 1,
        weight_names: vec![],
    };
    let buffer_plan = plan_buffers(&plan, &graph);
    assert_eq!(buffer_plan.step_sizes[1], 0);
}

#[test]
fn test_buffer_plan_naive_total_sums_all_sizes() {
    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};

    let nodes = vec![
        TraceNode::new(
            0,
            "c1".into(),
            TraceOp::Constant { value: 0.0 },
            vec![],
            vec![4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "c2".into(),
            TraceOp::Constant { value: 1.0 },
            vec![],
            vec![8],
            DType::F32,
        ),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::ConstantValue {
                value: 0.0,
                shape: vec![4],
            },
            CompiledStep::ConstantValue {
                value: 1.0,
                shape: vec![8],
            },
        ],
        input_shapes: vec![],
        output_step: 1,
        weight_names: vec![],
    };
    let buffer_plan = plan_buffers(&plan, &graph);
    assert_eq!(buffer_plan.step_sizes[0], 16);
    assert_eq!(buffer_plan.step_sizes[1], 32);
    assert_eq!(buffer_plan.naive_total, 48);
}

#[test]
fn test_buffer_plan_total_bytes_lte_naive_total() {
    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};

    let nodes = vec![
        TraceNode::new(
            0,
            "c1".into(),
            TraceOp::Constant { value: 0.0 },
            vec![],
            vec![100],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "c2".into(),
            TraceOp::Constant { value: 1.0 },
            vec![],
            vec![200],
            DType::F32,
        ),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::ConstantValue {
                value: 0.0,
                shape: vec![100],
            },
            CompiledStep::ConstantValue {
                value: 1.0,
                shape: vec![200],
            },
        ],
        input_shapes: vec![],
        output_step: 1,
        weight_names: vec![],
    };
    let buffer_plan = plan_buffers(&plan, &graph);
    assert!(buffer_plan.total_bytes <= buffer_plan.naive_total);
}

#[test]
fn test_buffer_plan_with_dtypes_f16_halves_size() {
    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};

    let nodes = vec![TraceNode::new(
        0,
        "c1".into(),
        TraceOp::Constant { value: 0.0 },
        vec![],
        vec![16],
        DType::F32,
    )];
    let graph = ComputationGraph::from_nodes(nodes);
    let plan = CompiledPlan {
        steps: vec![CompiledStep::ConstantValue {
            value: 0.0,
            shape: vec![16],
        }],
        input_shapes: vec![],
        output_step: 0,
        weight_names: vec![],
    };

    let plan_f32 = plan_buffers(&plan, &graph);
    let dtypes = vec![ScalarType::F16];
    let plan_f16 = plan_buffers_with_dtypes(&plan, &graph, &dtypes);

    // F32: 16 * 4 = 64. F16: 16 * 2 = 32.
    assert_eq!(plan_f32.step_sizes[0], 64);
    assert_eq!(plan_f16.step_sizes[0], 32);
}

// ---------------------------------------------------------------------------
// 5. NativeOpKind collect_direct_step_deps
// ---------------------------------------------------------------------------

#[test]
fn test_collect_direct_step_deps_fused_resblock() {
    use crate::trace_compile::NormActivConv1dParams;

    let params = NormActivConv1dParams {
        activation: crate::trace_compile::NormActivation::Snake,
        eps: 1e-5,
        conv_dilation: 1,
        conv_padding: 1,
        input_shape: vec![1, 4, 8],
        output_channels: 4,
        kernel_size: 3,
    };
    let op = NativeOpKind::FusedResBlock {
        phase1: params.clone(),
        phase2: params,
        input_steps: vec![0, 1, 2, 3, 4],
        residual_scale: 1.0,
        style_proj: None,
        shortcut_step: Some(10),
        pool_step: Some(20),
        style_batch_offset: None,
    };

    let mut deps = vec![];
    op.collect_direct_step_deps(&mut deps);
    assert_eq!(deps, vec![0, 1, 2, 3, 4, 10, 20]);
}

#[test]
fn test_collect_direct_step_deps_batched_style_projection() {
    let op = NativeOpKind::BatchedStyleProjection {
        blocks: vec![],
        style_dim: 128,
        total_out: 256,
        style_step: 5,
    };
    let mut deps = vec![];
    op.collect_direct_step_deps(&mut deps);
    assert_eq!(deps, vec![5]);
}

#[test]
fn test_collect_direct_step_deps_projection_slice() {
    let op = NativeOpKind::ProjectionSlice {
        source_step: 7,
        dim: 2,
        start: 0,
        length: 64,
        output_shape: vec![1, 4, 64],
    };
    let mut deps = vec![];
    op.collect_direct_step_deps(&mut deps);
    assert_eq!(deps, vec![7]);
}

#[test]
fn test_collect_direct_step_deps_instance_norm_empty() {
    let op = NativeOpKind::InstanceNorm {
        eps: 1e-5,
        input_shape: vec![1, 4, 16],
    };
    let mut deps = vec![];
    op.collect_direct_step_deps(&mut deps);
    assert!(deps.is_empty());
}

#[test]
fn test_collect_direct_step_deps_fused_resblock_no_optional() {
    use crate::trace_compile::NormActivConv1dParams;

    let params = NormActivConv1dParams {
        activation: crate::trace_compile::NormActivation::Snake,
        eps: 1e-5,
        conv_dilation: 1,
        conv_padding: 1,
        input_shape: vec![1, 4, 8],
        output_channels: 4,
        kernel_size: 3,
    };
    let op = NativeOpKind::FusedResBlock {
        phase1: params.clone(),
        phase2: params,
        input_steps: vec![0, 1],
        residual_scale: 1.0,
        style_proj: None,
        shortcut_step: None,
        pool_step: None,
        style_batch_offset: None,
    };

    let mut deps = vec![];
    op.collect_direct_step_deps(&mut deps);
    assert_eq!(deps, vec![0, 1]);
}

// ---------------------------------------------------------------------------
// 6. NativeOpKind serialization round-trip (serde)
// ---------------------------------------------------------------------------

#[test]
fn test_native_op_serde_round_trip_instance_norm() {
    let op = NativeOpKind::InstanceNorm {
        eps: 1e-5,
        input_shape: vec![1, 4, 16],
    };
    let json = serde_json::to_string(&op).expect("serialize");
    let deserialized: NativeOpKind = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.variant_name(), "InstanceNorm");
}

#[test]
fn test_native_op_serde_round_trip_fused_mul_add() {
    let op = NativeOpKind::FusedMulAdd {
        input_shape: vec![2, 8, 64],
    };
    let json = serde_json::to_string(&op).expect("serialize");
    let deserialized: NativeOpKind = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.variant_name(), "FusedMulAdd");
}

#[test]
fn test_compiled_step_serde_round_trip_identity_passthrough() {
    let step = CompiledStep::IdentityPassthrough;
    let json = serde_json::to_string(&step).expect("serialize");
    let deserialized: CompiledStep = serde_json::from_str(&json).expect("deserialize");
    assert!(matches!(deserialized, CompiledStep::IdentityPassthrough));
}

#[test]
fn test_compiled_step_serde_round_trip_constant_value() {
    let step = CompiledStep::ConstantValue {
        value: 3.14,
        shape: vec![2, 3],
    };
    let json = serde_json::to_string(&step).expect("serialize");
    let deserialized: CompiledStep = serde_json::from_str(&json).expect("deserialize");
    match deserialized {
        CompiledStep::ConstantValue { value, shape } => {
            assert!((value - 3.14).abs() < 1e-10);
            assert_eq!(shape, vec![2, 3]);
        }
        _ => panic!("expected ConstantValue"),
    }
}

#[test]
fn test_compiled_plan_serde_round_trip() {
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            CompiledStep::ConstantValue {
                value: 1.0,
                shape: vec![4],
            },
            CompiledStep::IdentityPassthrough,
        ],
        input_shapes: vec![vec![4]],
        output_step: 2,
        weight_names: vec!["w1".into()],
    };
    let json = serde_json::to_string(&plan).expect("serialize");
    let deserialized: CompiledPlan = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.steps.len(), 3);
    assert_eq!(deserialized.output_step, 2);
    assert_eq!(deserialized.weight_names, vec!["w1"]);
}

// ---------------------------------------------------------------------------
// 7. OptimizationResult summarize
// ---------------------------------------------------------------------------

#[test]
fn test_optimization_result_summarize_contains_key_info() {
    use crate::trace_compile::OptimizationResult;

    let result = OptimizationResult {
        plan: CompiledPlan {
            steps: vec![],
            input_shapes: vec![],
            output_step: 0,
            weight_names: vec![],
        },
        config: PeepholeConfig::default(),
        dispatch_count: 150,
        configs_explored: 65536,
        baseline_dispatch_count: 200,
        best_cost_ns: 5000.0,
        baseline_cost_ns: 10000.0,
    };
    let summary = result.summarize();
    assert!(
        summary.contains("200"),
        "should contain baseline dispatch count"
    );
    assert!(
        summary.contains("150"),
        "should contain best dispatch count"
    );
    assert!(summary.contains("65536"), "should contain configs explored");
    assert!(
        summary.contains("25.0%"),
        "should contain reduction percentage"
    );
}

#[test]
fn test_optimization_result_summarize_zero_baseline() {
    use crate::trace_compile::OptimizationResult;

    let result = OptimizationResult {
        plan: CompiledPlan {
            steps: vec![],
            input_shapes: vec![],
            output_step: 0,
            weight_names: vec![],
        },
        config: PeepholeConfig::default(),
        dispatch_count: 0,
        configs_explored: 1,
        baseline_dispatch_count: 0,
        best_cost_ns: 0.0,
        baseline_cost_ns: 0.0,
    };
    let summary = result.summarize();
    assert!(summary.contains("baseline is 0 dispatches"));
}

// ---------------------------------------------------------------------------
// 8. Multiple NativeOps in a plan
// ---------------------------------------------------------------------------

#[test]
fn test_plan_with_multiple_native_ops_dispatch_sum() {
    // Verify count_dispatches sums across all NativeOp steps.
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
            CompiledStep::NativeOp {
                op: NativeOpKind::LayerNorm {
                    eps: 1e-5,
                    input_shape: vec![1, 4, 16],
                    hidden_dim: 16,
                },
                weight_data: HashMap::new(),
            },
            CompiledStep::NativeOp {
                op: NativeOpKind::SiluMul {
                    input_shape: vec![1, 4, 16],
                },
                weight_data: HashMap::new(),
            },
        ],
        input_shapes: vec![vec![1, 4, 16]],
        output_step: 3,
        weight_names: vec![],
    };
    assert_eq!(count_dispatches(&plan), 3);
}

#[test]
fn test_narrow_view_step_zero_allocation() {
    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};

    let nodes = vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![1, 8, 256],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "narrow".into(),
            TraceOp::Narrow {
                dim: 2,
                start: 0,
                length: 128,
            },
            vec![0],
            vec![1, 8, 128],
            DType::F32,
        ),
    ];
    let graph = ComputationGraph::from_nodes(nodes);
    let plan = CompiledPlan {
        steps: vec![
            CompiledStep::InputForward,
            CompiledStep::NarrowView {
                byte_offset: 0,
                output_shape: vec![1, 8, 128],
                source_step: None,
            },
        ],
        input_shapes: vec![vec![1, 8, 256]],
        output_step: 1,
        weight_names: vec![],
    };
    let buffer_plan = plan_buffers(&plan, &graph);
    // NarrowView has zero allocation
    assert_eq!(buffer_plan.step_sizes[1], 0);
}

// ---------------------------------------------------------------------------
// Helper: config_from_bitmask (local copy for tests)
// ---------------------------------------------------------------------------

fn config_from_bitmask(mask: u32) -> PeepholeConfig {
    PeepholeConfig {
        norm_activ_conv1d: mask & (1 << 0) != 0,
        fused_resblock: mask & (1 << 1) != 0,
        linear_activation: mask & (1 << 2) != 0,
        add_layer_norm: mask & (1 << 3) != 0,
        norm_linear: mask & (1 << 4) != 0,
        attention_transpose: mask & (1 << 5) != 0,
        flip_lstm: mask & (1 << 6) != 0,
        batched_linear_projection: mask & (1 << 7) != 0,
        channels_first_layer_norm: mask & (1 << 8) != 0,
        silu_mul: mask & (1 << 9) != 0,
        auto_fuse_elementwise: mask & (1 << 10) != 0,
        bilstm_cat: mask & (1 << 11) != 0,
        add_norm_linear: mask & (1 << 12) != 0,
        fuse_adain_snake: mask & (1 << 13) != 0,
        fuse_upsample_conv1d: mask & (1 << 14) != 0,
        fuse_instance_norm_mul_add: mask & (1 << 15) != 0,
        fuse_conv1d_activation: mask & (1 << 16) != 0,
        fuse_snake_instance_norm: mask & (1 << 17) != 0,
        fuse_conv1d_snake_norm: mask & (1 << 18) != 0,
        fuse_conv1d_snake_norm_resblock: mask & (1 << 19) != 0,
        fuse_add_instance_norm_conv1x1: mask & (1 << 20) != 0,
        fuse_conv_transpose1d_activation: mask & (1 << 21) != 0,
        norm_activ_conv_transpose1d: mask & (1 << 22) != 0,
        fuse_instance_norm_conv1d: mask & (1 << 23) != 0,
        fuse_conv1d_instance_norm: mask & (1 << 24) != 0,
        fuse_linear_layer_norm: mask & (1 << 25) != 0,
        fuse_resblock_chain: mask & (1 << 26) != 0,
        fuse_activation_conv1d: mask & (1 << 27) != 0,
    }
}
