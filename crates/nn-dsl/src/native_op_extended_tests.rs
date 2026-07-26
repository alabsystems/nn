// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Part of #4186.
//!
//! Extended tests for `NativeOpKind`, `PeepholeConfig`, `CompiledStep`,
//! and related optimization types in nn-dsl.

use crate::trace_compile::optimize_plan::PEEPHOLE_FIELD_COUNT;
use crate::trace_compile::{
    AttentionLayout, CompiledStep, ConvActivation, FusedNormKind, GemmActivation, NativeOpKind,
    NormActivConv1dParams, NormActivation, PeepholeConfig, RuntimeOpKind, StyleBatchOffset,
    StyleProjectionParams,
};

// ===========================================================================
// 1. NativeOpKind — enum construction, Debug, variant_name(), dispatch count
// ===========================================================================

#[test]
fn test_native_op_kind_lstm_sequence_construction() {
    let op = NativeOpKind::LstmSequence {
        hidden_size: 256,
        input_shape: vec![100, 1, 512],
        h_shape: vec![1, 256],
        reverse: false,
    };
    assert_eq!(op.variant_name(), "LstmSequence");
}

#[test]
fn test_native_op_kind_cumsum_construction() {
    let op = NativeOpKind::Cumsum {
        dim: 2,
        input_shape: vec![1, 8, 1024],
    };
    assert_eq!(op.variant_name(), "Cumsum");
}

#[test]
fn test_native_op_kind_instance_norm_construction() {
    let op = NativeOpKind::InstanceNorm {
        eps: 1e-5,
        input_shape: vec![1, 64, 256],
    };
    assert_eq!(op.variant_name(), "InstanceNorm");
}

#[test]
fn test_native_op_kind_layer_norm_construction() {
    let op = NativeOpKind::LayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 128, 768],
        hidden_dim: 768,
    };
    assert_eq!(op.variant_name(), "LayerNorm");
}

#[test]
fn test_native_op_kind_add_layer_norm_construction() {
    let op = NativeOpKind::AddLayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 128, 768],
        hidden_dim: 768,
    };
    assert_eq!(op.variant_name(), "AddLayerNorm");
}

#[test]
fn test_native_op_kind_adain_snake_construction() {
    let op = NativeOpKind::AdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 128, 512],
        channels: 128,
        residual_gamma: true,
        external_node_ids: None,
    };
    assert_eq!(op.variant_name(), "AdainSnake");
}

#[test]
fn test_native_op_kind_adain_leaky_relu_construction() {
    let op = NativeOpKind::AdainLeakyRelu {
        eps: 1e-5,
        slope: 0.2,
        input_shape: vec![1, 64, 256],
        external_node_ids: Some(vec![10, 20, 30]),
    };
    assert_eq!(op.variant_name(), "AdainLeakyRelu");
}

#[test]
fn test_native_op_kind_ada_layer_norm_construction() {
    let op = NativeOpKind::AdaLayerNorm {
        eps: 1e-6,
        input_shape: vec![1, 32, 256],
        hidden_dim: 256,
    };
    assert_eq!(op.variant_name(), "AdaLayerNorm");
}

#[test]
fn test_native_op_kind_flash_attention_construction() {
    let op = NativeOpKind::FlashAttention {
        scale: 0.125,
        causal: true,
        q_shape: vec![1, 8, 128, 64],
        k_shape: vec![1, 8, 128, 64],
        output_shape: vec![1, 8, 128, 64],
        input_layout: AttentionLayout::HeadsFirst,
    };
    assert_eq!(op.variant_name(), "FlashAttention");
}

#[test]
fn test_native_op_kind_max_pool1d_construction() {
    let op = NativeOpKind::MaxPool1d {
        kernel_size: 3,
        stride: 2,
        padding: 1,
        input_shape: vec![1, 64, 256],
    };
    assert_eq!(op.variant_name(), "MaxPool1d");
}

#[test]
fn test_native_op_kind_constant_weight_construction() {
    let op = NativeOpKind::ConstantWeight {
        name: "arange_pos".to_string(),
        shape: vec![128],
    };
    assert_eq!(op.variant_name(), "ConstantWeight");
}

#[test]
fn test_native_op_kind_linear_activation_construction() {
    let op = NativeOpKind::LinearActivation {
        activation: GemmActivation::Gelu,
        in_features: 768,
        out_features: 3072,
        has_bias: true,
        input_shape: vec![1, 128, 768],
    };
    assert_eq!(op.variant_name(), "LinearActivation");
}

#[test]
fn test_native_op_kind_batched_linear_projection_construction() {
    let op = NativeOpKind::BatchedLinearProjection {
        in_features: 768,
        total_out_features: 2304,
        projection_sizes: vec![768, 768, 768],
        has_bias: true,
        input_shape: vec![1, 128, 768],
    };
    assert_eq!(op.variant_name(), "BatchedLinearProjection");
}

#[test]
fn test_native_op_kind_projection_slice_construction() {
    let op = NativeOpKind::ProjectionSlice {
        source_step: 5,
        dim: 2,
        start: 768,
        length: 768,
        output_shape: vec![1, 128, 768],
    };
    assert_eq!(op.variant_name(), "ProjectionSlice");
}

#[test]
fn test_native_op_kind_norm_linear_construction() {
    let op = NativeOpKind::NormLinear {
        norm_kind: FusedNormKind::LayerNorm,
        eps: 1e-5,
        input_shape: vec![1, 128, 768],
        hidden_dim: 768,
        out_features: 3072,
        has_bias: true,
    };
    assert_eq!(op.variant_name(), "NormLinear");
}

#[test]
fn test_native_op_kind_channels_first_layer_norm_construction() {
    let op = NativeOpKind::ChannelsFirstLayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 256, 512],
        channels: 256,
        leaky_relu_slope: Some(0.2),
    };
    assert_eq!(op.variant_name(), "ChannelsFirstLayerNorm");
}

#[test]
fn test_native_op_kind_int8_gemm_construction() {
    let op = NativeOpKind::Int8Gemm {
        in_features: 768,
        out_features: 768,
        has_bias: false,
        input_shape: vec![1, 128, 768],
    };
    assert_eq!(op.variant_name(), "Int8Gemm");
}

#[test]
fn test_native_op_kind_conv1d_gemm_construction() {
    let op = NativeOpKind::Conv1dGemm {
        input_shape: vec![1, 128, 512],
        out_channels: 256,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
        has_bias: true,
    };
    assert_eq!(op.variant_name(), "Conv1dGemm");
}

#[test]
fn test_native_op_kind_silu_mul_construction() {
    let op = NativeOpKind::SiluMul {
        input_shape: vec![1, 128, 3072],
    };
    assert_eq!(op.variant_name(), "SiluMul");
}

#[test]
fn test_native_op_kind_rotary_embedding_construction() {
    let op = NativeOpKind::RotaryEmbedding {
        head_dim: 64,
        input_shape: vec![1, 8, 128, 64],
    };
    assert_eq!(op.variant_name(), "RotaryEmbedding");
}

#[test]
fn test_native_op_kind_add_norm_linear_construction() {
    let op = NativeOpKind::AddNormLinear {
        eps: 1e-5,
        input_shape: vec![1, 128, 768],
        hidden_dim: 768,
        out_features: 3072,
        has_bias: true,
    };
    assert_eq!(op.variant_name(), "AddNormLinear");
}

#[test]
fn test_native_op_kind_moe_gating_construction() {
    let op = NativeOpKind::MoeGating {
        num_experts: 8,
        top_k: 2,
        input_shape: vec![1, 128, 768],
    };
    assert_eq!(op.variant_name(), "MoeGating");
}

#[test]
fn test_native_op_kind_fused_adain_snake_construction() {
    let op = NativeOpKind::FusedAdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 128, 512],
        channels: 128,
        external_node_ids: None,
    };
    assert_eq!(op.variant_name(), "FusedAdainSnake");
}

#[test]
fn test_native_op_kind_fused_upsample_conv1d_construction() {
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 2,
        in_channels: 128,
        out_channels: 256,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        input_shape: vec![1, 128, 256],
    };
    assert_eq!(op.variant_name(), "FusedUpsampleConv1d");
}

#[test]
fn test_native_op_kind_bilstm_cat_construction() {
    let op = NativeOpKind::BiLstmCat {
        hidden_size: 256,
        input_shape: vec![100, 1, 512],
        h_shape: vec![1, 256],
        fwd_lstm_step: 3,
        rev_lstm_step: 5,
    };
    assert_eq!(op.variant_name(), "BiLstmCat");
}

#[test]
fn test_native_op_kind_fused_mul_add_construction() {
    let op = NativeOpKind::FusedMulAdd {
        input_shape: vec![1, 128, 512],
    };
    assert_eq!(op.variant_name(), "FusedMulAdd");
}

#[test]
fn test_native_op_kind_fused_siglu_construction() {
    let op = NativeOpKind::FusedSiGLU {
        input_shape: vec![1, 128, 3072],
    };
    assert_eq!(op.variant_name(), "FusedSiGLU");
}

#[test]
fn test_native_op_kind_fused_geglu_construction() {
    let op = NativeOpKind::FusedGeGLU {
        input_shape: vec![1, 128, 3072],
    };
    assert_eq!(op.variant_name(), "FusedGeGLU");
}

#[test]
fn test_native_op_kind_fused_layer_norm_linear_construction() {
    let op = NativeOpKind::FusedLayerNormLinear {
        eps: 1e-5,
        input_shape: vec![1, 128, 768],
        hidden_dim: 768,
        out_features: 3072,
        has_bias: true,
    };
    assert_eq!(op.variant_name(), "FusedLayerNormLinear");
}

#[test]
fn test_native_op_kind_fused_instance_norm_mul_add_construction() {
    let op = NativeOpKind::FusedInstanceNormMulAdd {
        eps: 1e-5,
        input_shape: vec![1, 128, 512],
        channels: 128,
        external_node_ids: Some(vec![1, 2, 3]),
    };
    assert_eq!(op.variant_name(), "FusedInstanceNormMulAdd");
}

#[test]
fn test_native_op_kind_debug_format_contains_variant() {
    let op = NativeOpKind::SiluMul {
        input_shape: vec![1, 128],
    };
    let dbg = format!("{op:?}");
    assert!(dbg.contains("SiluMul"), "Debug output: {dbg}");
}

#[test]
fn test_native_op_kind_clone_equality() {
    let op = NativeOpKind::InstanceNorm {
        eps: 1e-5,
        input_shape: vec![1, 64, 256],
    };
    let cloned = op.clone();
    let dbg_orig = format!("{op:?}");
    let dbg_clone = format!("{cloned:?}");
    assert_eq!(dbg_orig, dbg_clone);
}

#[test]
fn test_native_op_kind_estimated_dispatches_single_kernel_ops() {
    // Most fused ops are single Metal dispatches.
    let single_dispatch_ops = vec![
        NativeOpKind::InstanceNorm {
            eps: 1e-5,
            input_shape: vec![1, 64, 256],
        },
        NativeOpKind::AdainSnake {
            eps: 1e-5,
            input_shape: vec![1, 128, 512],
            channels: 128,
            residual_gamma: true,
            external_node_ids: None,
        },
        NativeOpKind::SiluMul {
            input_shape: vec![1, 128, 3072],
        },
        NativeOpKind::FusedMulAdd {
            input_shape: vec![1, 128, 512],
        },
        NativeOpKind::FusedSiGLU {
            input_shape: vec![1, 128, 3072],
        },
        NativeOpKind::FusedGeGLU {
            input_shape: vec![1, 128, 3072],
        },
    ];
    for op in &single_dispatch_ops {
        assert_eq!(
            op.estimated_metal_dispatches(),
            1,
            "Expected 1 dispatch for {}, got {}",
            op.variant_name(),
            op.estimated_metal_dispatches()
        );
    }
}

// ===========================================================================
// 2. PeepholeConfig — Default, field access, field count
// ===========================================================================

#[test]
fn test_peephole_config_default_all_true() {
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
    assert!(config.fuse_conv1d_activation);
}

#[test]
fn test_peephole_config_field_count_matches_constant() {
    // PEEPHOLE_FIELD_COUNT must match the number of boolean fields.
    // The struct has 28 boolean fields.
    assert_eq!(PEEPHOLE_FIELD_COUNT, 28);
}

#[test]
fn test_peephole_config_clone_equality() {
    let config = PeepholeConfig::default();
    let cloned = config.clone();
    assert_eq!(config, cloned);
}

#[test]
fn test_peephole_config_individual_disable() {
    let config = PeepholeConfig {
        norm_activ_conv1d: false,
        ..Default::default()
    };
    assert!(!config.norm_activ_conv1d);
    // Other fields remain true.
    assert!(config.fused_resblock);
    assert!(config.linear_activation);
}

#[test]
fn test_peephole_config_all_disabled() {
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
    assert!(!config.fuse_conv1d_activation);
}

#[test]
fn test_peephole_config_ne_when_different() {
    let a = PeepholeConfig::default();
    let b = PeepholeConfig {
        silu_mul: false,
        ..Default::default()
    };
    assert_ne!(a, b);
}

#[test]
fn test_peephole_config_debug_format() {
    let config = PeepholeConfig::default();
    let dbg = format!("{config:?}");
    assert!(dbg.contains("PeepholeConfig"), "Debug output: {dbg}");
    assert!(
        dbg.contains("norm_activ_conv1d: true"),
        "Debug output: {dbg}"
    );
}

// ===========================================================================
// 3. NormActivation enum
// ===========================================================================

#[test]
fn test_norm_activation_leaky_relu_construction() {
    let act = NormActivation::LeakyRelu { slope: 0.2 };
    let dbg = format!("{act:?}");
    assert!(dbg.contains("LeakyRelu"), "Debug: {dbg}");
    assert!(dbg.contains("0.2"), "Debug: {dbg}");
}

#[test]
fn test_norm_activation_snake_construction() {
    let act = NormActivation::Snake;
    let dbg = format!("{act:?}");
    assert!(dbg.contains("Snake"), "Debug: {dbg}");
}

#[test]
fn test_norm_activation_clone_eq() {
    let a = NormActivation::LeakyRelu { slope: 0.1 };
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn test_norm_activation_ne_different_variants() {
    let a = NormActivation::Snake;
    let b = NormActivation::LeakyRelu { slope: 0.2 };
    assert_ne!(a, b);
}

// ===========================================================================
// 4. GemmActivation enum
// ===========================================================================

#[test]
fn test_gemm_activation_all_variants_debug() {
    let variants = [
        GemmActivation::Relu,
        GemmActivation::Gelu,
        GemmActivation::GeluErf,
        GemmActivation::Sigmoid,
        GemmActivation::Silu,
        GemmActivation::Tanh,
    ];
    let names = ["Relu", "Gelu", "GeluErf", "Sigmoid", "Silu", "Tanh"];
    for (v, name) in variants.iter().zip(names.iter()) {
        let dbg = format!("{v:?}");
        assert!(dbg.contains(name), "Expected '{name}' in Debug: {dbg}");
    }
}

#[test]
fn test_gemm_activation_copy_eq() {
    let a = GemmActivation::Gelu;
    let b = a; // Copy
    assert_eq!(a, b);
}

// ===========================================================================
// 5. AttentionLayout enum
// ===========================================================================

#[test]
fn test_attention_layout_default_is_heads_first() {
    let layout = AttentionLayout::default();
    assert_eq!(layout, AttentionLayout::HeadsFirst);
}

#[test]
fn test_attention_layout_seq_first_ne_heads_first() {
    assert_ne!(AttentionLayout::HeadsFirst, AttentionLayout::SeqFirst);
}

#[test]
fn test_attention_layout_copy_semantics() {
    let a = AttentionLayout::SeqFirst;
    let b = a; // Copy
    assert_eq!(a, b);
}

// ===========================================================================
// 6. FusedNormKind enum
// ===========================================================================

#[test]
fn test_fused_norm_kind_variants() {
    assert_ne!(FusedNormKind::LayerNorm, FusedNormKind::RmsNorm);
    let a = FusedNormKind::LayerNorm;
    let b = a; // Copy
    assert_eq!(a, b);
}

#[test]
fn test_fused_norm_kind_debug() {
    let dbg = format!("{:?}", FusedNormKind::RmsNorm);
    assert!(dbg.contains("RmsNorm"), "Debug: {dbg}");
}

// ===========================================================================
// 7. StyleProjectionParams and StyleBatchOffset (constructor-based)
// ===========================================================================

#[test]
fn test_style_projection_params_new() {
    let params = StyleProjectionParams::new(128, 256, 512);
    assert_eq!(params.channels1, 128);
    assert_eq!(params.channels2, 256);
    assert_eq!(params.style_dim, 512);
}

#[test]
fn test_style_batch_offset_new() {
    let offset = StyleBatchOffset::new(0, 128, 256);
    assert_eq!(offset.offset, 0);
    assert_eq!(offset.channels1, 128);
    assert_eq!(offset.channels2, 256);
}

// ===========================================================================
// 8. NormActivConv1dParams (constructor-based, #[non_exhaustive])
// ===========================================================================

#[test]
fn test_norm_activ_conv1d_params_new() {
    let params = NormActivConv1dParams::new(
        NormActivation::Snake,
        1e-5,
        3, // dilation
        1, // padding
        vec![1, 128, 512],
        256, // output_channels
        3,   // kernel_size
    );
    assert_eq!(params.activation, NormActivation::Snake);
    assert!((params.eps - 1e-5).abs() < 1e-10);
    assert_eq!(params.conv_dilation, 3);
    assert_eq!(params.conv_padding, 1);
    assert_eq!(params.input_shape, vec![1, 128, 512]);
    assert_eq!(params.output_channels, 256);
    assert_eq!(params.kernel_size, 3);
}

// ===========================================================================
// 9. CompiledStep variants
// ===========================================================================

#[test]
fn test_compiled_step_identity_passthrough_debug() {
    let step = CompiledStep::IdentityPassthrough;
    let dbg = format!("{step:?}");
    assert!(dbg.contains("IdentityPassthrough"), "Debug: {dbg}");
}

#[test]
fn test_compiled_step_input_forward_debug() {
    let step = CompiledStep::InputForward;
    let dbg = format!("{step:?}");
    assert!(dbg.contains("InputForward"), "Debug: {dbg}");
}

#[test]
fn test_compiled_step_constant_value_construction() {
    let step = CompiledStep::ConstantValue {
        value: 1.0,
        shape: vec![1, 128, 768],
    };
    let dbg = format!("{step:?}");
    assert!(dbg.contains("ConstantValue"), "Debug: {dbg}");
}

#[test]
fn test_compiled_step_passthrough_construction() {
    let step = CompiledStep::Passthrough {
        op_name: "reshape".to_string(),
        output_shape: vec![1, 128, 768],
    };
    let dbg = format!("{step:?}");
    assert!(dbg.contains("Passthrough"), "Debug: {dbg}");
    assert!(dbg.contains("reshape"), "Debug: {dbg}");
}

#[test]
fn test_compiled_step_narrow_view_construction() {
    let step = CompiledStep::NarrowView {
        byte_offset: 1024,
        output_shape: vec![1, 64, 768],
        source_step: Some(3),
    };
    let dbg = format!("{step:?}");
    assert!(dbg.contains("NarrowView"), "Debug: {dbg}");
    assert!(dbg.contains("1024"), "Debug: {dbg}");
}

#[test]
fn test_compiled_step_native_op_wraps_kind() {
    let step = CompiledStep::NativeOp {
        op: NativeOpKind::FusedMulAdd {
            input_shape: vec![1, 128],
        },
        weight_data: std::collections::HashMap::new(),
    };
    let dbg = format!("{step:?}");
    assert!(dbg.contains("NativeOp"), "Debug: {dbg}");
    assert!(dbg.contains("FusedMulAdd"), "Debug: {dbg}");
}

#[test]
fn test_compiled_step_runtime_op_construction() {
    let step = CompiledStep::RuntimeOp {
        op: RuntimeOpKind::RepeatInterleave {
            dim: 1,
            input_shape: vec![1, 128, 768],
            counts_shape: vec![128],
        },
    };
    let dbg = format!("{step:?}");
    assert!(dbg.contains("RuntimeOp"), "Debug: {dbg}");
    assert!(dbg.contains("RepeatInterleave"), "Debug: {dbg}");
}

// ===========================================================================
// 10. NativeOpKind variant_name exhaustive coverage (all 34 variants)
// ===========================================================================

#[test]
fn test_native_op_kind_variant_name_exhaustive() {
    // Construct one of each variant and verify variant_name returns non-empty.
    let ops: Vec<NativeOpKind> = vec![
        NativeOpKind::LstmSequence {
            hidden_size: 256,
            input_shape: vec![10, 1, 512],
            h_shape: vec![1, 256],
            reverse: false,
        },
        NativeOpKind::Cumsum {
            dim: 0,
            input_shape: vec![10],
        },
        NativeOpKind::InstanceNorm {
            eps: 1e-5,
            input_shape: vec![1, 64, 256],
        },
        NativeOpKind::LayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 128, 768],
            hidden_dim: 768,
        },
        NativeOpKind::AddLayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 128, 768],
            hidden_dim: 768,
        },
        NativeOpKind::AdainSnake {
            eps: 1e-5,
            input_shape: vec![1, 128, 512],
            channels: 128,
            residual_gamma: false,
            external_node_ids: None,
        },
        NativeOpKind::AdainLeakyRelu {
            eps: 1e-5,
            slope: 0.2,
            input_shape: vec![1, 64, 256],
            external_node_ids: None,
        },
        NativeOpKind::AdaLayerNorm {
            eps: 1e-6,
            input_shape: vec![1, 32, 256],
            hidden_dim: 256,
        },
        NativeOpKind::FlashAttention {
            scale: 0.125,
            causal: false,
            q_shape: vec![1, 8, 128, 64],
            k_shape: vec![1, 8, 128, 64],
            output_shape: vec![1, 8, 128, 64],
            input_layout: AttentionLayout::default(),
        },
        NativeOpKind::MaxPool1d {
            kernel_size: 3,
            stride: 2,
            padding: 1,
            input_shape: vec![1, 64, 256],
        },
        NativeOpKind::ConstantWeight {
            name: "test".into(),
            shape: vec![128],
        },
        NativeOpKind::FusedResBlock {
            phase1: NormActivConv1dParams::new(
                NormActivation::Snake,
                1e-5,
                1,
                1,
                vec![1, 128, 512],
                128,
                3,
            ),
            phase2: NormActivConv1dParams::new(
                NormActivation::Snake,
                1e-5,
                1,
                1,
                vec![1, 128, 512],
                128,
                3,
            ),
            input_steps: vec![0, 1, 2, 3, 4],
            residual_scale: 1.0,
            style_proj: None,
            shortcut_step: None,
            pool_step: None,
            style_batch_offset: None,
        },
        NativeOpKind::BatchedStyleProjection {
            blocks: vec![],
            style_dim: 128,
            total_out: 0,
            style_step: 0,
        },
        NativeOpKind::NormActivConv1d {
            activation: NormActivation::Snake,
            eps: 1e-5,
            conv_dilation: 1,
            conv_padding: 1,
            input_shape: vec![1, 128, 512],
            output_channels: 256,
            kernel_size: 3,
            external_node_ids: None,
        },
        NativeOpKind::LinearActivation {
            activation: GemmActivation::Relu,
            in_features: 768,
            out_features: 3072,
            has_bias: true,
            input_shape: vec![1, 128, 768],
        },
        NativeOpKind::BatchedLinearProjection {
            in_features: 768,
            total_out_features: 2304,
            projection_sizes: vec![768, 768, 768],
            has_bias: true,
            input_shape: vec![1, 128, 768],
        },
        NativeOpKind::ProjectionSlice {
            source_step: 0,
            dim: 2,
            start: 0,
            length: 768,
            output_shape: vec![1, 128, 768],
        },
        NativeOpKind::NormLinear {
            norm_kind: FusedNormKind::RmsNorm,
            eps: 1e-5,
            input_shape: vec![1, 128, 768],
            hidden_dim: 768,
            out_features: 3072,
            has_bias: false,
        },
        NativeOpKind::ChannelsFirstLayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 256, 512],
            channels: 256,
            leaky_relu_slope: None,
        },
        NativeOpKind::Int8Gemm {
            in_features: 768,
            out_features: 768,
            has_bias: false,
            input_shape: vec![1, 128, 768],
        },
        NativeOpKind::Conv1dGemm {
            input_shape: vec![1, 128, 512],
            out_channels: 256,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            has_bias: true,
        },
        NativeOpKind::SiluMul {
            input_shape: vec![1, 128, 3072],
        },
        NativeOpKind::RotaryEmbedding {
            head_dim: 64,
            input_shape: vec![1, 8, 128, 64],
        },
        NativeOpKind::AddNormLinear {
            eps: 1e-5,
            input_shape: vec![1, 128, 768],
            hidden_dim: 768,
            out_features: 3072,
            has_bias: true,
        },
        NativeOpKind::MoeGating {
            num_experts: 8,
            top_k: 2,
            input_shape: vec![1, 128, 768],
        },
        NativeOpKind::FusedAdainSnake {
            eps: 1e-5,
            input_shape: vec![1, 128, 512],
            channels: 128,
            external_node_ids: None,
        },
        NativeOpKind::FusedUpsampleConv1d {
            upsample_factor: 2,
            in_channels: 128,
            out_channels: 256,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            input_shape: vec![1, 128, 256],
        },
        NativeOpKind::BiLstmCat {
            hidden_size: 256,
            input_shape: vec![100, 1, 512],
            h_shape: vec![1, 256],
            fwd_lstm_step: 3,
            rev_lstm_step: 5,
        },
        NativeOpKind::FusedMulAdd {
            input_shape: vec![1, 128, 512],
        },
        NativeOpKind::FusedSiGLU {
            input_shape: vec![1, 128, 3072],
        },
        NativeOpKind::FusedGeGLU {
            input_shape: vec![1, 128, 3072],
        },
        NativeOpKind::FusedLayerNormLinear {
            eps: 1e-5,
            input_shape: vec![1, 128, 768],
            hidden_dim: 768,
            out_features: 3072,
            has_bias: true,
        },
        NativeOpKind::FusedInstanceNormMulAdd {
            eps: 1e-5,
            input_shape: vec![1, 128, 512],
            channels: 128,
            external_node_ids: None,
        },
        NativeOpKind::FusedConv1dActivation {
            activation: ConvActivation::Relu,
            out_channels: 256,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            has_bias: true,
            input_shape: vec![1, 128, 512],
            pre_activation: false,
        },
    ];

    // Verify each variant_name is non-empty.
    for op in &ops {
        let name = op.variant_name();
        assert!(!name.is_empty(), "variant_name() should be non-empty");
    }

    // Verify no duplicate variant names (each variant is represented once).
    let mut seen = std::collections::HashSet::new();
    for op in &ops {
        let name = op.variant_name();
        assert!(seen.insert(name), "Duplicate variant_name: {name}");
    }
}

// ===========================================================================
// 11. NativeOpKind serialization round-trip (serde)
// ===========================================================================

#[test]
fn test_native_op_kind_serde_round_trip() {
    let op = NativeOpKind::SiluMul {
        input_shape: vec![1, 128, 3072],
    };
    let json = serde_json::to_string(&op).expect("serialize NativeOpKind");
    let deserialized: NativeOpKind = serde_json::from_str(&json).expect("deserialize NativeOpKind");
    assert_eq!(deserialized.variant_name(), "SiluMul");
}

#[test]
fn test_native_op_kind_serde_flash_attention_round_trip() {
    let op = NativeOpKind::FlashAttention {
        scale: 0.125,
        causal: true,
        q_shape: vec![1, 8, 128, 64],
        k_shape: vec![1, 8, 128, 64],
        output_shape: vec![1, 8, 128, 64],
        input_layout: AttentionLayout::SeqFirst,
    };
    let json = serde_json::to_string(&op).expect("serialize");
    let deserialized: NativeOpKind = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.variant_name(), "FlashAttention");
    // Verify the JSON contains the layout.
    assert!(json.contains("SeqFirst"), "JSON: {json}");
}

#[test]
fn test_compiled_step_serde_round_trip() {
    let step = CompiledStep::ConstantValue {
        value: 42.0,
        shape: vec![1, 10],
    };
    let json = serde_json::to_string(&step).expect("serialize CompiledStep");
    let deserialized: CompiledStep = serde_json::from_str(&json).expect("deserialize");
    let dbg = format!("{deserialized:?}");
    assert!(dbg.contains("42"), "Debug: {dbg}");
}

// ===========================================================================
// 12. NativeOpKind — reverse field on LstmSequence
// ===========================================================================

#[test]
fn test_native_op_kind_lstm_reverse_field() {
    let fwd = NativeOpKind::LstmSequence {
        hidden_size: 128,
        input_shape: vec![50, 1, 256],
        h_shape: vec![1, 128],
        reverse: false,
    };
    let rev = NativeOpKind::LstmSequence {
        hidden_size: 128,
        input_shape: vec![50, 1, 256],
        h_shape: vec![1, 128],
        reverse: true,
    };
    // Both have same variant name but different Debug output.
    assert_eq!(fwd.variant_name(), rev.variant_name());
    let fwd_dbg = format!("{fwd:?}");
    let rev_dbg = format!("{rev:?}");
    assert!(fwd_dbg.contains("reverse: false"), "fwd Debug: {fwd_dbg}");
    assert!(rev_dbg.contains("reverse: true"), "rev Debug: {rev_dbg}");
}

// ===========================================================================
// 13. ChannelsFirstLayerNorm with and without leaky_relu_slope
// ===========================================================================

#[test]
fn test_channels_first_layer_norm_optional_leaky_relu() {
    let without = NativeOpKind::ChannelsFirstLayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 256, 512],
        channels: 256,
        leaky_relu_slope: None,
    };
    let with = NativeOpKind::ChannelsFirstLayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 256, 512],
        channels: 256,
        leaky_relu_slope: Some(0.1),
    };
    let dbg_without = format!("{without:?}");
    let dbg_with = format!("{with:?}");
    assert!(dbg_without.contains("None"), "Debug: {dbg_without}");
    assert!(dbg_with.contains("0.1"), "Debug: {dbg_with}");
}

// ===========================================================================
// 14. FusedResBlock with style projection params
// ===========================================================================

#[test]
fn test_fused_resblock_with_style_projection() {
    let params = StyleProjectionParams::new(128, 256, 512);
    let op = NativeOpKind::FusedResBlock {
        phase1: NormActivConv1dParams::new(
            NormActivation::LeakyRelu { slope: 0.2 },
            1e-5,
            1,
            1,
            vec![1, 128, 512],
            128,
            3,
        ),
        phase2: NormActivConv1dParams::new(
            NormActivation::LeakyRelu { slope: 0.2 },
            1e-5,
            1,
            1,
            vec![1, 128, 512],
            128,
            3,
        ),
        input_steps: vec![0, 1],
        residual_scale: std::f32::consts::FRAC_1_SQRT_2,
        style_proj: Some(params),
        shortcut_step: Some(7),
        pool_step: None,
        style_batch_offset: None,
    };
    assert_eq!(op.variant_name(), "FusedResBlock");
    let dbg = format!("{op:?}");
    assert!(dbg.contains("StyleProjectionParams"), "Debug: {dbg}");
    assert!(dbg.contains("shortcut_step: Some(7)"), "Debug: {dbg}");
}

// ===========================================================================
// 15. FusedConv1dActivation — construction, serde, dispatch count (#4252)
// ===========================================================================

#[test]
fn test_fused_conv1d_activation_relu_construction() {
    let op = NativeOpKind::FusedConv1dActivation {
        activation: ConvActivation::Relu,
        out_channels: 256,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
        has_bias: true,
        input_shape: vec![1, 128, 512],
        pre_activation: false,
    };
    assert_eq!(op.variant_name(), "FusedConv1dActivation");
}

#[test]
fn test_fused_conv1d_activation_snake_construction() {
    let op = NativeOpKind::FusedConv1dActivation {
        activation: ConvActivation::Snake,
        out_channels: 128,
        kernel_size: 7,
        stride: 1,
        padding: 3,
        dilation: 1,
        groups: 1,
        has_bias: true,
        input_shape: vec![1, 128, 1024],
        pre_activation: false,
    };
    assert_eq!(op.variant_name(), "FusedConv1dActivation");
    let dbg = format!("{op:?}");
    assert!(dbg.contains("Snake"), "Debug: {dbg}");
}

#[test]
fn test_fused_conv1d_activation_leaky_relu_construction() {
    let op = NativeOpKind::FusedConv1dActivation {
        activation: ConvActivation::LeakyRelu { slope: 0.2 },
        out_channels: 64,
        kernel_size: 3,
        stride: 2,
        padding: 1,
        dilation: 1,
        groups: 1,
        has_bias: false,
        input_shape: vec![1, 32, 256],
        pre_activation: false,
    };
    let dbg = format!("{op:?}");
    assert!(dbg.contains("LeakyRelu"), "Debug: {dbg}");
    assert!(dbg.contains("0.2"), "Debug: {dbg}");
}

#[test]
fn test_fused_conv1d_activation_silu_construction() {
    let op = NativeOpKind::FusedConv1dActivation {
        activation: ConvActivation::Silu,
        out_channels: 512,
        kernel_size: 1,
        stride: 1,
        padding: 0,
        dilation: 1,
        groups: 1,
        has_bias: true,
        input_shape: vec![1, 256, 128],
        pre_activation: false,
    };
    let dbg = format!("{op:?}");
    assert!(dbg.contains("Silu"), "Debug: {dbg}");
}

#[test]
fn test_fused_conv1d_activation_serde_round_trip() {
    let op = NativeOpKind::FusedConv1dActivation {
        activation: ConvActivation::Snake,
        out_channels: 128,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
        has_bias: true,
        input_shape: vec![1, 128, 512],
        pre_activation: false,
    };
    let json = serde_json::to_string(&op).expect("serialize FusedConv1dActivation");
    let deserialized: NativeOpKind = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.variant_name(), "FusedConv1dActivation");
    assert!(json.contains("Snake"), "JSON: {json}");
    assert!(json.contains("FusedConv1dActivation"), "JSON: {json}");
}

#[test]
fn test_fused_conv1d_activation_dispatch_count() {
    // FusedConv1dActivation fuses conv1d + activation into 1 dispatch.
    let op = NativeOpKind::FusedConv1dActivation {
        activation: ConvActivation::Relu,
        out_channels: 256,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
        has_bias: true,
        input_shape: vec![1, 128, 512],
        pre_activation: false,
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

// ===========================================================================
// 16. ConvActivation enum tests (#4252)
// ===========================================================================

#[test]
fn test_conv_activation_all_variants_debug() {
    let variants = [
        ConvActivation::Snake,
        ConvActivation::Relu,
        ConvActivation::LeakyRelu { slope: 0.01 },
        ConvActivation::Silu,
    ];
    let names = ["Snake", "Relu", "LeakyRelu", "Silu"];
    for (v, name) in variants.iter().zip(names.iter()) {
        let dbg = format!("{v:?}");
        assert!(dbg.contains(name), "Expected '{name}' in Debug: {dbg}");
    }
}

#[test]
fn test_conv_activation_copy_eq() {
    let a = ConvActivation::Silu;
    let b = a; // Copy
    assert_eq!(a, b);
}

#[test]
fn test_conv_activation_ne_different_variants() {
    assert_ne!(ConvActivation::Snake, ConvActivation::Relu);
    assert_ne!(ConvActivation::Relu, ConvActivation::Silu);
    assert_ne!(
        ConvActivation::LeakyRelu { slope: 0.1 },
        ConvActivation::LeakyRelu { slope: 0.2 }
    );
}
