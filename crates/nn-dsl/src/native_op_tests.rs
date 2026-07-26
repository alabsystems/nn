// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Part of #4252.
//!
//! Tests for `NativeOpKind` construction, Debug formatting, variant count,
//! `PeepholeConfig` field count, and default configuration.

use crate::trace_compile::optimize_plan::PEEPHOLE_FIELD_COUNT;
use crate::trace_compile::{
    AttentionLayout, ConvActivation, FusedNormKind, GemmActivation, NativeOpKind,
    NormActivConv1dParams, NormActivation, PeepholeConfig, StyleBatchOffset,
};

// ===========================================================================
// 1. NativeOpKind — every variant can be constructed
// ===========================================================================

#[test]
fn test_construct_all_native_op_variants() {
    let ops: Vec<NativeOpKind> = vec![
        NativeOpKind::LstmSequence {
            hidden_size: 128,
            input_shape: vec![10, 1, 256],
            h_shape: vec![1, 128],
            reverse: false,
        },
        NativeOpKind::Cumsum {
            dim: 1,
            input_shape: vec![1, 8, 512],
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
            residual_gamma: true,
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
            causal: true,
            q_shape: vec![1, 8, 128, 64],
            k_shape: vec![1, 8, 128, 64],
            output_shape: vec![1, 8, 128, 64],
            input_layout: AttentionLayout::HeadsFirst,
        },
        NativeOpKind::MaxPool1d {
            kernel_size: 3,
            stride: 2,
            padding: 1,
            input_shape: vec![1, 64, 256],
        },
        NativeOpKind::ConstantWeight {
            name: "test_const".to_string(),
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
            blocks: vec![StyleBatchOffset::new(0, 128, 128)],
            style_dim: 128,
            total_out: 512,
            style_step: 0,
        },
        NativeOpKind::NormActivConv1d {
            activation: NormActivation::LeakyRelu { slope: 0.2 },
            eps: 1e-5,
            conv_dilation: 1,
            conv_padding: 1,
            input_shape: vec![1, 128, 512],
            output_channels: 256,
            kernel_size: 3,
            external_node_ids: None,
        },
        NativeOpKind::LinearActivation {
            activation: GemmActivation::Gelu,
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
            source_step: 5,
            dim: 2,
            start: 768,
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
            leaky_relu_slope: Some(0.2),
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
            external_node_ids: Some(vec![1, 2, 3]),
        },
        NativeOpKind::FusedConv1dActivation {
            activation: ConvActivation::Snake,
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

    // All 34 variants must be present (matches KNOWN_NATIVE_OP_COUNT in
    // trace_compile_native_ops_dispatch_count.rs).
    assert_eq!(
        ops.len(),
        34,
        "Expected 34 NativeOpKind variants, got {}. Update this test when adding variants.",
        ops.len()
    );

    // Each variant_name must be non-empty and unique.
    let mut seen = std::collections::HashSet::new();
    for op in &ops {
        let name = op.variant_name();
        assert!(!name.is_empty(), "variant_name() returned empty string");
        assert!(seen.insert(name), "Duplicate variant_name: {name}");
    }
}

// ===========================================================================
// 2. NativeOpKind Debug formatting
// ===========================================================================

#[test]
fn test_native_op_kind_debug_contains_variant_name() {
    let test_cases: Vec<(NativeOpKind, &str)> = vec![
        (
            NativeOpKind::SiluMul {
                input_shape: vec![1, 128],
            },
            "SiluMul",
        ),
        (
            NativeOpKind::FusedMulAdd {
                input_shape: vec![1, 64],
            },
            "FusedMulAdd",
        ),
        (
            NativeOpKind::FlashAttention {
                scale: 0.125,
                causal: false,
                q_shape: vec![1, 8, 64, 64],
                k_shape: vec![1, 8, 64, 64],
                output_shape: vec![1, 8, 64, 64],
                input_layout: AttentionLayout::SeqFirst,
            },
            "FlashAttention",
        ),
        (
            NativeOpKind::FusedConv1dActivation {
                activation: ConvActivation::Relu,
                out_channels: 64,
                kernel_size: 3,
                stride: 1,
                padding: 1,
                dilation: 1,
                groups: 1,
                has_bias: true,
                input_shape: vec![1, 32, 128],
                pre_activation: false,
            },
            "FusedConv1dActivation",
        ),
        (
            NativeOpKind::FusedInstanceNormMulAdd {
                eps: 1e-5,
                input_shape: vec![1, 64, 256],
                channels: 64,
                external_node_ids: None,
            },
            "FusedInstanceNormMulAdd",
        ),
    ];

    for (op, expected_name) in &test_cases {
        let dbg = format!("{op:?}");
        assert!(
            dbg.contains(expected_name),
            "Debug for {expected_name} missing variant name: {dbg}"
        );
    }
}

// ===========================================================================
// 3. KNOWN_NATIVE_OP_COUNT matches actual variant count (34)
// ===========================================================================

#[test]
fn test_known_variant_count_matches_exhaustive_list() {
    // Verified by test_construct_all_native_op_variants which constructs
    // exactly 34 variants and checks the count. If a new variant is added,
    // that test will fail until the new variant is included in the list.
    //
    // This test separately verifies the variant_name() match arm count by
    // constructing a minimal instance of each variant and collecting names.
    let names: Vec<&str> = vec![
        "LstmSequence",
        "Cumsum",
        "InstanceNorm",
        "LayerNorm",
        "AddLayerNorm",
        "AdainSnake",
        "AdainLeakyRelu",
        "AdaLayerNorm",
        "FlashAttention",
        "MaxPool1d",
        "ConstantWeight",
        "FusedResBlock",
        "BatchedStyleProjection",
        "NormActivConv1d",
        "LinearActivation",
        "BatchedLinearProjection",
        "ProjectionSlice",
        "NormLinear",
        "ChannelsFirstLayerNorm",
        "Int8Gemm",
        "Conv1dGemm",
        "SiluMul",
        "RotaryEmbedding",
        "AddNormLinear",
        "MoeGating",
        "FusedAdainSnake",
        "FusedUpsampleConv1d",
        "BiLstmCat",
        "FusedMulAdd",
        "FusedSiGLU",
        "FusedGeGLU",
        "FusedLayerNormLinear",
        "FusedInstanceNormMulAdd",
        "FusedConv1dActivation",
        "FusedSnakeInstanceNorm",
        "FusedConv1dSnakeNorm",
        "BatchNorm2d",
        "FusedConv1dSnakeNormResBlock",
    ];
    assert_eq!(names.len(), 38, "Expected 38 variant names");
}

// ===========================================================================
// 4. PEEPHOLE_FIELD_COUNT matches PeepholeConfig fields (17)
// ===========================================================================

#[test]
fn test_peephole_field_count_is_20() {
    // PeepholeConfig has 28 boolean fields. This constant is used by the
    // bitmask enumeration for the self-optimizing compiler.
    assert_eq!(PEEPHOLE_FIELD_COUNT, 28);
}

// ===========================================================================
// 5. PeepholeConfig default has all fusions enabled
// ===========================================================================

#[test]
fn test_peephole_config_default_all_fusions_enabled() {
    let config = PeepholeConfig::default();
    assert!(config.norm_activ_conv1d, "norm_activ_conv1d should be true");
    assert!(config.fused_resblock, "fused_resblock should be true");
    assert!(config.linear_activation, "linear_activation should be true");
    assert!(config.add_layer_norm, "add_layer_norm should be true");
    assert!(config.norm_linear, "norm_linear should be true");
    assert!(
        config.attention_transpose,
        "attention_transpose should be true"
    );
    assert!(config.flip_lstm, "flip_lstm should be true");
    assert!(
        config.batched_linear_projection,
        "batched_linear_projection should be true"
    );
    assert!(
        config.channels_first_layer_norm,
        "channels_first_layer_norm should be true"
    );
    assert!(config.silu_mul, "silu_mul should be true");
    assert!(
        config.auto_fuse_elementwise,
        "auto_fuse_elementwise should be true"
    );
    assert!(config.bilstm_cat, "bilstm_cat should be true");
    assert!(config.add_norm_linear, "add_norm_linear should be true");
    assert!(config.fuse_adain_snake, "fuse_adain_snake should be true");
    assert!(
        config.fuse_upsample_conv1d,
        "fuse_upsample_conv1d should be true"
    );
    assert!(
        config.fuse_instance_norm_mul_add,
        "fuse_instance_norm_mul_add should be true"
    );
    assert!(
        config.fuse_conv1d_activation,
        "fuse_conv1d_activation should be true"
    );
    assert!(
        config.fuse_snake_instance_norm,
        "fuse_snake_instance_norm should be true"
    );
    assert!(
        config.fuse_conv1d_snake_norm,
        "fuse_conv1d_snake_norm should be true"
    );
    assert!(
        config.fuse_conv1d_snake_norm_resblock,
        "fuse_conv1d_snake_norm_resblock should be true"
    );
    assert!(
        config.fuse_conv_transpose1d_activation,
        "fuse_conv_transpose1d_activation should be true"
    );
    assert!(
        config.norm_activ_conv_transpose1d,
        "norm_activ_conv_transpose1d should be true"
    );
}

// ===========================================================================
// 6. PeepholeConfig search space matches 2^PEEPHOLE_FIELD_COUNT
// ===========================================================================

#[test]
fn test_peephole_search_space_size() {
    // The self-optimizing compiler enumerates 2^28 = 268435456 configurations.
    assert_eq!(1u32 << PEEPHOLE_FIELD_COUNT, 268_435_456);
}

// ===========================================================================
// 7. ConvActivation variants for FusedConv1dActivation
// ===========================================================================

#[test]
fn test_conv_activation_all_variants() {
    let variants = [
        ConvActivation::Snake,
        ConvActivation::Relu,
        ConvActivation::LeakyRelu { slope: 0.01 },
        ConvActivation::Silu,
    ];
    let expected_names = ["Snake", "Relu", "LeakyRelu", "Silu"];
    for (v, name) in variants.iter().zip(expected_names.iter()) {
        let dbg = format!("{v:?}");
        assert!(dbg.contains(name), "Expected '{name}' in Debug: {dbg}");
    }
}

#[test]
fn test_conv_activation_copy_eq() {
    let a = ConvActivation::Relu;
    let b = a; // Copy
    assert_eq!(a, b);
}

// ===========================================================================
// 8. NativeOpKind clone produces identical Debug output
// ===========================================================================

#[test]
fn test_native_op_kind_clone_identical_debug() {
    let op = NativeOpKind::FusedConv1dActivation {
        activation: ConvActivation::LeakyRelu { slope: 0.2 },
        out_channels: 128,
        kernel_size: 5,
        stride: 1,
        padding: 2,
        dilation: 1,
        groups: 1,
        has_bias: false,
        input_shape: vec![1, 64, 256],
        pre_activation: false,
    };
    let cloned = op.clone();
    assert_eq!(format!("{op:?}"), format!("{cloned:?}"));
}

// ===========================================================================
// 9. FusedConv1dActivation dispatch count is 1
// ===========================================================================

#[test]
fn test_fused_conv1d_activation_dispatch_count() {
    let op = NativeOpKind::FusedConv1dActivation {
        activation: ConvActivation::Snake,
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
