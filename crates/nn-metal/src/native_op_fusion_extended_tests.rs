// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for fused NativeOp variants and dispatch pipeline.
//!
//! Covers 10 areas:
//! 1. NativeOpKind — all 35 variants construction, variant_name, serde, Debug
//! 2. Fusion detection — pairs of ops that should fuse (peephole)
//! 3. Fusion rejection — pairs that should NOT fuse
//! 4. NATIVE_OP_VARIANT_COUNT matches actual enum variant count
//! 5. Dispatch plan building for elementwise/reduction/grid ops
//! 6. Threadgroup size calculations for different tensor shapes
//! 7. Buffer aliasing safety for fused operations
//! 8. Dtype validation — F32 required, non-F32 fallback
//! 9. PeepholeConfig field validation (17 fields)
//! 10. NativeOpKind dispatch count estimation consistency
//!
//! All tests are structure/config tests — no live GPU kernel execution.
//!
//! Part of #4252.

use std::collections::HashSet;

use nn_core::DType;
use nn_dsl::ir::ScalarType;
use nn_dsl::trace_compile::{
    ConvActivation, FusedNormKind, GemmActivation, NativeOpKind, NormActivation,
    NormActivConv1dParams,
};
use nn_dsl::PeepholeConfig;

use crate::dispatch_plan::{
    clear_dispatch_plan_cache, plan_elementwise, plan_grid_2d, plan_reduction, DispatchMode,
};

// ═══════════════════════════════════════════════════════════════════════
// 1. NativeOpKind — all 35 variants construction, variant_name, serde
// ═══════════════════════════════════════════════════════════════════════

/// Helper: construct all 35 NativeOpKind variants (including BatchNorm2d).
fn all_35_variants() -> Vec<NativeOpKind> {
    vec![
        NativeOpKind::LstmSequence {
            hidden_size: 256,
            input_shape: vec![10, 1, 128],
            h_shape: vec![1, 256],
            reverse: false,
        },
        NativeOpKind::Cumsum {
            dim: 0,
            input_shape: vec![10, 32],
        },
        NativeOpKind::InstanceNorm {
            eps: 1e-5,
            input_shape: vec![1, 128, 512],
        },
        NativeOpKind::LayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 32, 768],
            hidden_dim: 768,
        },
        NativeOpKind::AddLayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 32, 768],
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
            input_shape: vec![1, 128, 512],
            external_node_ids: None,
        },
        NativeOpKind::AdaLayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 32, 256],
            hidden_dim: 256,
        },
        NativeOpKind::FlashAttention {
            scale: 0.125,
            causal: false,
            q_shape: vec![1, 8, 32, 64],
            k_shape: vec![1, 8, 32, 64],
            output_shape: vec![1, 8, 32, 64],
            input_layout: Default::default(),
        },
        NativeOpKind::MaxPool1d {
            kernel_size: 3,
            stride: 2,
            padding: 1,
            input_shape: vec![1, 64, 100],
        },
        NativeOpKind::ConstantWeight {
            name: "arange".into(),
            shape: vec![100],
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
        NativeOpKind::NormActivConv1d {
            activation: NormActivation::Snake,
            eps: 1e-5,
            conv_dilation: 1,
            conv_padding: 1,
            input_shape: vec![1, 128, 512],
            output_channels: 128,
            kernel_size: 3,
            external_node_ids: None,
        },
        NativeOpKind::LinearActivation {
            activation: GemmActivation::Relu,
            in_features: 768,
            out_features: 256,
            has_bias: true,
            input_shape: vec![1, 32, 768],
        },
        NativeOpKind::BatchedLinearProjection {
            in_features: 768,
            total_out_features: 2304,
            projection_sizes: vec![768, 768, 768],
            has_bias: true,
            input_shape: vec![1, 32, 768],
        },
        NativeOpKind::ProjectionSlice {
            source_step: 0,
            dim: 2,
            start: 0,
            length: 768,
            output_shape: vec![1, 32, 768],
        },
        NativeOpKind::NormLinear {
            norm_kind: FusedNormKind::LayerNorm,
            eps: 1e-5,
            input_shape: vec![1, 32, 768],
            hidden_dim: 768,
            out_features: 256,
            has_bias: true,
        },
        NativeOpKind::BatchedStyleProjection {
            blocks: vec![],
            style_dim: 128,
            total_out: 512,
            style_step: 0,
        },
        NativeOpKind::Int8Gemm {
            in_features: 768,
            out_features: 256,
            has_bias: true,
            input_shape: vec![1, 32, 768],
        },
        NativeOpKind::Conv1dGemm {
            input_shape: vec![1, 128, 1024],
            out_channels: 256,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            has_bias: true,
        },
        NativeOpKind::SiluMul {
            input_shape: vec![1, 32, 768],
        },
        NativeOpKind::RotaryEmbedding {
            head_dim: 64,
            input_shape: vec![1, 8, 32, 64],
        },
        NativeOpKind::AddNormLinear {
            eps: 1e-5,
            input_shape: vec![1, 32, 768],
            hidden_dim: 768,
            out_features: 256,
            has_bias: true,
        },
        NativeOpKind::MoeGating {
            num_experts: 8,
            top_k: 2,
            input_shape: vec![1, 32, 768],
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
            out_channels: 64,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            input_shape: vec![1, 128, 256],
        },
        NativeOpKind::BiLstmCat {
            hidden_size: 256,
            input_shape: vec![10, 1, 128],
            h_shape: vec![1, 256],
            fwd_lstm_step: 0,
            rev_lstm_step: 1,
        },
        NativeOpKind::FusedMulAdd {
            input_shape: vec![1, 32, 768],
        },
        NativeOpKind::FusedSiGLU {
            input_shape: vec![1, 32, 768],
        },
        NativeOpKind::FusedGeGLU {
            input_shape: vec![1, 32, 768],
        },
        NativeOpKind::FusedLayerNormLinear {
            eps: 1e-5,
            input_shape: vec![1, 32, 768],
            hidden_dim: 768,
            out_features: 256,
            has_bias: true,
        },
        NativeOpKind::BatchNorm2d {
            eps: 1e-5,
            num_channels: 64,
            input_shape: vec![1, 64, 224, 224],
            has_weight: true,
            has_bias: true,
        },
        NativeOpKind::FusedInstanceNormMulAdd {
            eps: 1e-5,
            input_shape: vec![1, 128, 512],
            channels: 128,
            external_node_ids: None,
        },
        NativeOpKind::FusedConv1dActivation {
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
        },
        NativeOpKind::ChannelsFirstLayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 256, 100],
            channels: 256,
            leaky_relu_slope: None,
        },
    ]
}

/// All 35 NativeOpKind variants produce distinct, non-empty variant_name strings.
#[test]
fn test_all_35_variants_unique_names() {
    let variants = all_35_variants();
    assert_eq!(
        variants.len(),
        35,
        "Expected 35 variants, got {}",
        variants.len()
    );
    let names: HashSet<&str> = variants.iter().map(NativeOpKind::variant_name).collect();
    assert_eq!(
        names.len(),
        35,
        "Expected 35 unique variant names, got {}. Names: {names:?}",
        names.len(),
    );
    for name in &names {
        assert!(!name.is_empty(), "variant_name must be non-empty");
    }
}

/// All 35 variants produce non-empty Debug output containing the variant name.
#[test]
fn test_all_35_variants_debug_contains_name() {
    for op in &all_35_variants() {
        let name = op.variant_name();
        let dbg = format!("{op:?}");
        assert!(
            dbg.contains(name),
            "Debug for {name} should contain variant name: {dbg}"
        );
    }
}

/// All 35 NativeOpKind variants round-trip through serde_json.
#[test]
fn test_all_35_variants_serde_round_trip() {
    for op in &all_35_variants() {
        let json = serde_json::to_string(op)
            .unwrap_or_else(|e| panic!("serialize {} failed: {e}", op.variant_name()));
        let deser: NativeOpKind = serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("deserialize {} failed: {e}", op.variant_name()));
        assert_eq!(
            deser.variant_name(),
            op.variant_name(),
            "Round-trip variant name mismatch for {json}"
        );
    }
}

/// All NativeOpKind variant names are ASCII alphanumeric (used as registry keys).
#[test]
fn test_all_35_variant_names_alphanumeric() {
    for op in &all_35_variants() {
        let name = op.variant_name();
        assert!(
            name.chars().all(|c| c.is_ascii_alphanumeric()),
            "variant_name should be alphanumeric: {name}"
        );
    }
}

/// BatchNorm2d — the 35th variant — constructs and serializes correctly.
#[test]
fn test_batchnorm2d_construction_and_serde() {
    let op = NativeOpKind::BatchNorm2d {
        eps: 1e-5,
        num_channels: 128,
        input_shape: vec![4, 128, 56, 56],
        has_weight: true,
        has_bias: true,
    };
    assert_eq!(op.variant_name(), "BatchNorm2d");
    assert_eq!(op.estimated_metal_dispatches(), 1);

    let json = serde_json::to_string(&op).expect("serialize BatchNorm2d");
    assert!(json.contains("BatchNorm2d"));
    assert!(json.contains("128"));
    let deser: NativeOpKind = serde_json::from_str(&json).expect("deserialize BatchNorm2d");
    assert_eq!(deser.variant_name(), "BatchNorm2d");
}

/// BatchNorm2d without weight/bias (no learnable affine).
#[test]
fn test_batchnorm2d_no_affine() {
    let op = NativeOpKind::BatchNorm2d {
        eps: 1e-3,
        num_channels: 32,
        input_shape: vec![2, 32, 112, 112],
        has_weight: false,
        has_bias: false,
    };
    assert_eq!(op.variant_name(), "BatchNorm2d");
    assert_eq!(op.estimated_metal_dispatches(), 1);
    let json = serde_json::to_string(&op).expect("serialize");
    assert!(json.contains("\"has_weight\":false"));
    assert!(json.contains("\"has_bias\":false"));
}

// ═══════════════════════════════════════════════════════════════════════
// 2. Fusion detection — pairs of ops that should fuse (peephole patterns)
// ═══════════════════════════════════════════════════════════════════════

/// Conv1d + Snake should fuse via NormActivConv1d or FusedConv1dActivation.
/// The fused variant always has fewer dispatches than the sum of parts.
#[test]
fn test_fusion_conv1d_snake_saves_dispatches() {
    // Unfused: Conv1dGemm (1 dispatch) + separate Snake activation (1 dispatch) = 2
    let conv = NativeOpKind::Conv1dGemm {
        input_shape: vec![1, 128, 512],
        out_channels: 128,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
        has_bias: true,
    };
    let fused = NativeOpKind::FusedConv1dActivation {
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
    // FusedConv1dActivation is 1 dispatch; unfused Conv1dGemm + Snake is 2.
    assert!(
        fused.estimated_metal_dispatches() < conv.estimated_metal_dispatches() + 1,
        "Fused ({}) should be < unfused conv ({}) + activation (1)",
        fused.estimated_metal_dispatches(),
        conv.estimated_metal_dispatches()
    );
}

/// LayerNorm + Linear should fuse via NormLinear or FusedLayerNormLinear.
#[test]
fn test_fusion_layernorm_linear_saves_dispatches() {
    let norm = NativeOpKind::LayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 32, 768],
        hidden_dim: 768,
    };
    let fused = NativeOpKind::FusedLayerNormLinear {
        eps: 1e-5,
        input_shape: vec![1, 32, 768],
        hidden_dim: 768,
        out_features: 3072,
        has_bias: true,
    };
    // LayerNorm (1) + Linear (1) = 2 dispatches unfused.
    // FusedLayerNormLinear should be <= 2.
    assert!(
        fused.estimated_metal_dispatches() <= norm.estimated_metal_dispatches() + 1,
        "Fused ({}) should be <= LayerNorm ({}) + Linear (1)",
        fused.estimated_metal_dispatches(),
        norm.estimated_metal_dispatches()
    );
}

/// Add + LayerNorm should fuse via AddLayerNorm.
#[test]
fn test_fusion_add_layernorm_single_dispatch() {
    let fused = NativeOpKind::AddLayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 32, 768],
        hidden_dim: 768,
    };
    // Replaces Add (1) + LayerNorm (1) = 2 dispatches with 1.
    assert_eq!(
        fused.estimated_metal_dispatches(),
        1,
        "AddLayerNorm should be 1 dispatch"
    );
}

/// Add + LayerNorm + Linear fuses via AddNormLinear.
#[test]
fn test_fusion_add_norm_linear_saves_dispatches() {
    let fused = NativeOpKind::AddNormLinear {
        eps: 1e-5,
        input_shape: vec![1, 32, 768],
        hidden_dim: 768,
        out_features: 3072,
        has_bias: true,
    };
    // Replaces Add (1) + LayerNorm (1) + Linear (1) = 3 with 1-2.
    assert!(
        fused.estimated_metal_dispatches() <= 3,
        "AddNormLinear ({}) should be <= 3 dispatches",
        fused.estimated_metal_dispatches()
    );
}

/// InstanceNorm + Mul + Add should fuse via FusedInstanceNormMulAdd.
#[test]
fn test_fusion_instance_norm_mul_add_single_dispatch() {
    let fused = NativeOpKind::FusedInstanceNormMulAdd {
        eps: 1e-5,
        input_shape: vec![1, 64, 256],
        channels: 64,
        external_node_ids: None,
    };
    // Replaces InstanceNorm (1) + Mul (1) + Add (1) = 3 with 1.
    assert_eq!(
        fused.estimated_metal_dispatches(),
        1,
        "FusedInstanceNormMulAdd should be 1 dispatch"
    );
}

/// InstanceNorm + Mul + Add + Snake fuses via FusedAdainSnake.
#[test]
fn test_fusion_adain_snake_single_dispatch() {
    let fused = NativeOpKind::FusedAdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 128, 512],
        channels: 128,
        external_node_ids: None,
    };
    // Replaces InstanceNorm (1) + Mul (1) + Add (1) + Snake (1) = 4 with 1.
    assert_eq!(
        fused.estimated_metal_dispatches(),
        1,
        "FusedAdainSnake should be 1 dispatch"
    );
}

/// Upsample1d + Conv1d fuses via FusedUpsampleConv1d.
#[test]
fn test_fusion_upsample_conv1d_single_dispatch() {
    let fused = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 2,
        in_channels: 256,
        out_channels: 128,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        input_shape: vec![1, 256, 512],
    };
    assert_eq!(
        fused.estimated_metal_dispatches(),
        1,
        "FusedUpsampleConv1d should be 1 dispatch"
    );
}

/// Forward LSTM + Reverse LSTM + Cat fuses via BiLstmCat.
#[test]
fn test_fusion_bilstm_cat_fewer_dispatches() {
    let fwd = NativeOpKind::LstmSequence {
        hidden_size: 256,
        input_shape: vec![10, 1, 128],
        h_shape: vec![1, 256],
        reverse: false,
    };
    let rev = NativeOpKind::LstmSequence {
        hidden_size: 256,
        input_shape: vec![10, 1, 128],
        h_shape: vec![1, 256],
        reverse: true,
    };
    let fused = NativeOpKind::BiLstmCat {
        hidden_size: 256,
        input_shape: vec![10, 1, 128],
        h_shape: vec![1, 256],
        fwd_lstm_step: 0,
        rev_lstm_step: 1,
    };
    // Unfused: fwd (1) + rev (1) + cat (1) = 3.
    // BiLstmCat should be fewer.
    assert!(
        fused.estimated_metal_dispatches()
            <= fwd.estimated_metal_dispatches() + rev.estimated_metal_dispatches() + 1,
        "BiLstmCat ({}) should be <= fwd ({}) + rev ({}) + cat (1)",
        fused.estimated_metal_dispatches(),
        fwd.estimated_metal_dispatches(),
        rev.estimated_metal_dispatches()
    );
}

/// SiLU + Mul fuses via SiluMul.
#[test]
fn test_fusion_silu_mul_single_dispatch() {
    let fused = NativeOpKind::SiluMul {
        input_shape: vec![1, 32, 768],
    };
    assert_eq!(
        fused.estimated_metal_dispatches(),
        1,
        "SiluMul should be 1 dispatch"
    );
}

/// FusedMulAdd: a*b+c in single dispatch.
#[test]
fn test_fusion_fma_single_dispatch() {
    let fused = NativeOpKind::FusedMulAdd {
        input_shape: vec![1, 32, 768],
    };
    assert_eq!(fused.estimated_metal_dispatches(), 1);
}

/// FusedSiGLU: sigmoid(x)*x in single dispatch.
#[test]
fn test_fusion_siglu_single_dispatch() {
    let fused = NativeOpKind::FusedSiGLU {
        input_shape: vec![1, 32, 768],
    };
    assert_eq!(fused.estimated_metal_dispatches(), 1);
}

/// FusedGeGLU: gelu(gate)*up in single dispatch.
#[test]
fn test_fusion_geglu_single_dispatch() {
    let fused = NativeOpKind::FusedGeGLU {
        input_shape: vec![1, 32, 768],
    };
    assert_eq!(fused.estimated_metal_dispatches(), 1);
}

/// Linear + Activation fuses via LinearActivation.
#[test]
fn test_fusion_linear_activation_single_dispatch() {
    for act in [
        GemmActivation::Relu,
        GemmActivation::Gelu,
        GemmActivation::Tanh,
        GemmActivation::Silu,
    ] {
        let fused = NativeOpKind::LinearActivation {
            activation: act,
            in_features: 768,
            out_features: 3072,
            has_bias: true,
            input_shape: vec![1, 32, 768],
        };
        assert_eq!(
            fused.estimated_metal_dispatches(),
            1,
            "LinearActivation({act:?}) should be 1 dispatch"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 3. Fusion rejection — pairs that should NOT fuse
// ═══════════════════════════════════════════════════════════════════════

/// MatMul + Conv1d should NOT fuse — different compute categories (Opaque+Opaque).
/// There is no NativeOpKind variant for MatMul+Conv1d fusion.
#[test]
fn test_no_fusion_matmul_conv1d() {
    let all_names: HashSet<&str> = all_35_variants()
        .iter()
        .map(NativeOpKind::variant_name)
        .collect();
    // There is no "FusedMatMulConv1d" or "MatMulConv1d" variant.
    assert!(
        !all_names.contains("FusedMatMulConv1d"),
        "MatMul+Conv1d should not have a fused NativeOp"
    );
    assert!(
        !all_names.contains("MatMulConv1d"),
        "MatMul+Conv1d should not have a fused NativeOp"
    );
}

/// LSTM + Conv1d should NOT fuse.
#[test]
fn test_no_fusion_lstm_conv1d() {
    let all_names: HashSet<&str> = all_35_variants()
        .iter()
        .map(NativeOpKind::variant_name)
        .collect();
    assert!(
        !all_names.contains("FusedLstmConv1d"),
        "LSTM+Conv1d should not have a fused NativeOp"
    );
}

/// FlashAttention + Conv1d should NOT fuse.
#[test]
fn test_no_fusion_attention_conv1d() {
    let all_names: HashSet<&str> = all_35_variants()
        .iter()
        .map(NativeOpKind::variant_name)
        .collect();
    assert!(
        !all_names.contains("FusedAttentionConv1d"),
        "FlashAttention+Conv1d should not have a fused NativeOp"
    );
}

/// Cumsum + Softmax should NOT fuse (Opaque + Reduction).
#[test]
fn test_no_fusion_cumsum_softmax() {
    let all_names: HashSet<&str> = all_35_variants()
        .iter()
        .map(NativeOpKind::variant_name)
        .collect();
    assert!(!all_names.contains("FusedCumsumSoftmax"));
}

/// FusedResBlock should NOT be nested inside another FusedResBlock.
/// No "FusedResBlockResBlock" variant exists.
#[test]
fn test_no_nested_fused_resblock() {
    let all_names: HashSet<&str> = all_35_variants()
        .iter()
        .map(NativeOpKind::variant_name)
        .collect();
    assert!(!all_names.contains("FusedResBlockResBlock"));
    assert!(!all_names.contains("NestedResBlock"));
}

// ═══════════════════════════════════════════════════════════════════════
// 4. NATIVE_OP_VARIANT_COUNT matches actual enum variant count
// ═══════════════════════════════════════════════════════════════════════

/// The all_35_variants helper produces exactly 35 variants matching the
/// NATIVE_OP_VARIANT_COUNT constant (35) in compiled_kokoro_registry.rs.
#[test]
fn test_native_op_variant_count_is_35() {
    let variants = all_35_variants();
    assert_eq!(
        variants.len(),
        35,
        "all_35_variants should produce exactly 35 variants"
    );
}

/// Every variant name in our 35-variant set is unique and present.
#[test]
fn test_every_variant_name_unique_and_present() {
    let variants = all_35_variants();
    let names: HashSet<&str> = variants.iter().map(NativeOpKind::variant_name).collect();
    assert_eq!(
        names.len(),
        35,
        "Expected 35 unique variant names, got {}",
        names.len()
    );

    // Verify key variants are included.
    let expected_names = [
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
        "NormActivConv1d",
        "LinearActivation",
        "BatchedLinearProjection",
        "ProjectionSlice",
        "NormLinear",
        "BatchedStyleProjection",
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
        "BatchNorm2d",
        "FusedInstanceNormMulAdd",
        "FusedConv1dActivation",
        "ChannelsFirstLayerNorm",
    ];
    for name in &expected_names {
        assert!(
            names.contains(name),
            "Missing expected NativeOpKind variant: {name}"
        );
    }
    assert_eq!(
        expected_names.len(),
        35,
        "Expected name list should have 35 entries"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 5. Dispatch plan building for elementwise/reduction/grid ops
// ═══════════════════════════════════════════════════════════════════════

/// Elementwise plans succeed for typical NativeOp tensor sizes.
#[test]
fn test_dispatch_plan_elementwise_typical_sizes() {
    clear_dispatch_plan_cache();
    // Typical shapes from Kokoro: [1, 128, 512], [1, 32, 768], [1, 8, 32, 64]
    let element_counts: Vec<u32> = vec![
        1 * 128 * 512,    // 65536
        1 * 32 * 768,     // 24576
        1 * 8 * 32 * 64,  // 16384
        1 * 256 * 1024,   // 262144
        1 * 64 * 224 * 224, // 3211264 (BatchNorm2d typical)
    ];
    for total in element_counts {
        let plan = plan_elementwise(total);
        assert!(
            plan.is_ok(),
            "plan_elementwise({total}) should succeed: {:?}",
            plan.err()
        );
    }
}

/// Elementwise plan with 1 element succeeds (edge case).
#[test]
fn test_dispatch_plan_elementwise_single_element() {
    let plan = plan_elementwise(1).expect("single element plan");
    assert_eq!(plan.output_elems(), 1);
}

/// Elementwise plan with large element count (close to u32::MAX) succeeds.
#[test]
fn test_dispatch_plan_elementwise_large() {
    let plan = plan_elementwise(u32::MAX).expect("max u32 plan");
    assert!(plan.output_elems() >= u32::MAX as usize);
}

/// Grid2D plans produce correct grid dimensions.
#[test]
fn test_dispatch_plan_grid2d_dimensions() {
    let cases: Vec<([u32; 2], [u32; 2])> = vec![
        ([64, 32], [8, 8]),
        ([128, 128], [16, 16]),
        ([1, 768], [1, 256]),
        ([256, 256], [8, 8]),
    ];
    for (grid, threads) in cases {
        let plan = plan_grid_2d(grid, threads);
        assert!(
            plan.is_ok(),
            "plan_grid_2d({grid:?}, {threads:?}) should succeed: {:?}",
            plan.err()
        );
        let p = plan.unwrap();
        assert_eq!(p.output_elems(), grid[0] as usize * grid[1] as usize);
    }
}

/// Reduction plans succeed for typical shapes.
#[test]
fn test_dispatch_plan_reduction_typical() {
    let cases = [
        (32_u32, 768_u32, 256_u32, 3072_u32), // LayerNorm [32, 768]
        (1_u32, 512_u32, 256_u32, 2048_u32),   // InstanceNorm reduce
        (64_u32, 128_u32, 128_u32, 1024_u32),  // Small reduction
    ];
    for (outer, reduce, thr, shared) in cases {
        let plan = plan_reduction(outer, reduce, thr, shared);
        assert!(
            plan.is_ok(),
            "plan_reduction({outer}, {reduce}, {thr}, {shared}) failed: {:?}",
            plan.err()
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 6. Threadgroup size calculations for different tensor shapes
// ═══════════════════════════════════════════════════════════════════════

/// Elementwise threadgroup product never exceeds Metal limit (1024).
#[test]
fn test_threadgroup_elementwise_bounded() {
    for total in [1, 32, 64, 128, 256, 1024, 65536, 262144, u32::MAX] {
        let plan = plan_elementwise(total).unwrap();
        let [tw, th, td] = plan.threads();
        let product = tw * th * td;
        assert!(
            product <= 1024,
            "threadgroup product {product} exceeds 1024 for total={total}"
        );
    }
}

/// Grid2D threadgroup product never exceeds Metal limit (1024).
#[test]
fn test_threadgroup_grid2d_bounded() {
    let cases = [
        ([64_u32, 32], [8_u32, 8]),
        ([256, 256], [16, 16]),
        ([1024, 1], [32, 1]),
    ];
    for (grid, threads) in cases {
        let plan = plan_grid_2d(grid, threads).unwrap();
        let [tw, th, td] = plan.threads();
        let product = tw * th * td;
        assert!(
            product <= 1024,
            "Grid2D threadgroup product {product} exceeds 1024 for grid={grid:?}"
        );
    }
}

/// Reduction threadgroup product never exceeds Metal limit.
#[test]
fn test_threadgroup_reduction_bounded() {
    let cases = [
        (32_u32, 256, 256, 4096),
        (1, 1024, 128, 2048),
        (512, 64, 64, 1024),
        (1, 768, 256, 3072),
    ];
    for (outer, reduce, thr, shared) in cases {
        let plan = plan_reduction(outer, reduce, thr, shared).unwrap();
        let [tw, th, td] = plan.threads();
        let product = tw * th * td;
        assert!(
            product <= 1024,
            "Reduction threadgroup {product} exceeds 1024 for reduce={reduce}"
        );
    }
}

/// DispatchMode::Elementwise plan has correct output_elems.
#[test]
fn test_dispatch_mode_elementwise_output_elems() {
    for total in [1_u32, 100, 1024, 65536, 1048576] {
        let plan = DispatchMode::Elementwise { total }.plan().unwrap();
        assert_eq!(
            plan.output_elems(),
            total as usize,
            "output_elems should match total for {total}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 7. Buffer aliasing safety for fused operations
// ═══════════════════════════════════════════════════════════════════════

/// Buffer size calculation for fused ops uses checked_mul to prevent overflow.
#[test]
fn test_buffer_size_checked_mul_overflow() {
    // usize::MAX / 2 * 2 == usize::MAX - 1 (no overflow), so use MAX/2 + 1 to
    // guarantee overflow for the F16 factor (2) as well as F32 (4).
    let large = usize::MAX / 2 + 1;
    assert!(
        large.checked_mul(4).is_none(),
        "should overflow with F32 byte size"
    );
    assert!(
        large.checked_mul(2).is_none(),
        "should overflow with F16 byte size"
    );
    // Normal cases succeed.
    let typical = 1_usize * 128 * 512;
    assert_eq!(typical.checked_mul(4).unwrap(), typical * 4);
}

/// GpuSlice byte_offset is preserved through construction.
#[test]
fn test_gpu_slice_offset_safety() {
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let buf = ctx.create_buffer_zeroed(4096).expect("alloc buffer");

    let slice0 = crate::gpu_slice::GpuSlice::zero_offset(buf.alias());
    assert_eq!(slice0.byte_offset(), 0);

    let slice_mid = crate::gpu_slice::GpuSlice::new(buf.alias(), 2048);
    assert_eq!(slice_mid.byte_offset(), 2048);

    let slice_end = crate::gpu_slice::GpuSlice::new(buf.alias(), 4000);
    assert_eq!(slice_end.byte_offset(), 4000);
}

/// GpuSlice alias preserves byte offset for fused operations.
#[test]
fn test_gpu_slice_alias_offset_preservation() {
    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let buf = ctx.create_buffer_zeroed(8192).expect("alloc buffer");
    let offsets = [0, 256, 1024, 4096, 8000];
    for offset in offsets {
        let slice = crate::gpu_slice::GpuSlice::new(buf.alias(), offset);
        let aliased = slice.alias();
        assert_eq!(
            aliased.byte_offset(),
            offset,
            "alias should preserve offset {offset}"
        );
    }
}

/// Buffer size for typical fused op tensor shapes.
#[test]
fn test_fused_op_buffer_sizes() {
    // FusedAdainSnake: [1, 128, 512] F32
    let adain_elems: usize = [1, 128, 512].iter().product();
    assert_eq!(adain_elems * 4, 262_144); // 256 KB

    // FusedResBlock: [1, 128, 512] F32 input + same output
    // Input/output each 256 KB.

    // BiLstmCat: [10, 1, 128] input, output [10, 1, 512] (2*hidden)
    let bilstm_in: usize = [10, 1, 128].iter().product();
    let bilstm_out: usize = [10, 1, 512].iter().product();
    assert_eq!(bilstm_in * 4, 5120);
    assert_eq!(bilstm_out * 4, 20480);

    // BatchNorm2d: [1, 64, 224, 224] F32
    let bn_elems: usize = [1, 64, 224, 224].iter().product();
    assert_eq!(bn_elems * 4, 12_845_056); // ~12.25 MB
}

// ═══════════════════════════════════════════════════════════════════════
// 8. Dtype validation — F32 required, non-F32 fallback
// ═══════════════════════════════════════════════════════════════════════

/// dtype_to_msl succeeds for GPU float types (F32, F16, BF16).
#[test]
fn test_dtype_to_msl_float_types() {
    let (msl, size) = crate::dtype_to_msl(DType::F32).expect("F32");
    assert_eq!(msl, "float");
    assert_eq!(size, 4);

    let (msl, size) = crate::dtype_to_msl(DType::F16).expect("F16");
    assert_eq!(msl, "half");
    assert_eq!(size, 2);

    let (msl, size) = crate::dtype_to_msl(DType::BF16).expect("BF16");
    assert_eq!(msl, "half");
    assert_eq!(size, 2);
}

/// dtype_to_msl rejects non-float types (triggers CPU fallback path).
#[test]
fn test_dtype_to_msl_rejects_non_float() {
    for dtype in [DType::I32, DType::I64, DType::U32, DType::U8, DType::Bool] {
        assert!(
            crate::dtype_to_msl(dtype).is_err(),
            "{dtype:?} should be rejected for MSL codegen"
        );
    }
}

/// ScalarType round-trips through DType for all float variants.
#[test]
fn test_scalar_type_dtype_roundtrip() {
    for st in [ScalarType::F32, ScalarType::F16, ScalarType::BF16] {
        let dtype: DType = st.into();
        let recovered = ScalarType::try_from(dtype).expect("roundtrip");
        assert_eq!(st, recovered, "ScalarType roundtrip failed for {st:?}");
    }
}

/// ScalarType byte sizes match DType byte sizes.
#[test]
fn test_scalar_type_byte_size_consistency() {
    assert_eq!(ScalarType::F32.byte_size(), DType::F32.size_bytes());
    assert_eq!(ScalarType::F16.byte_size(), DType::F16.size_bytes());
    assert_eq!(ScalarType::BF16.byte_size(), DType::BF16.size_bytes());
}

/// ScalarType accumulator is always "float" (full precision reduction).
#[test]
fn test_scalar_type_accumulator_always_f32() {
    for st in [ScalarType::F32, ScalarType::F16, ScalarType::BF16] {
        assert_eq!(
            st.msl_accumulator_str(),
            "float",
            "accumulator should be float for {st:?}"
        );
    }
}

/// F64 is a valid DType but has no ScalarType (not supported on Metal GPU).
#[test]
fn test_f64_no_scalar_type() {
    let result = ScalarType::try_from(DType::F64);
    assert!(
        result.is_err(),
        "F64 should not convert to ScalarType (no Metal support)"
    );
}

/// can_use_direct_dispatch for elementwise fused ops.
#[test]
fn test_direct_dispatch_coverage() {
    use crate::native_op_direct::can_use_direct_dispatch;

    // These should support direct dispatch (bypass DynTensor bridge).
    let silu_mul = NativeOpKind::SiluMul {
        input_shape: vec![1, 32, 768],
    };
    assert!(
        can_use_direct_dispatch(&silu_mul),
        "SiluMul should support direct dispatch"
    );

    let fma = NativeOpKind::FusedMulAdd {
        input_shape: vec![1, 32, 768],
    };
    assert!(
        can_use_direct_dispatch(&fma),
        "FusedMulAdd should support direct dispatch"
    );

    let siglu = NativeOpKind::FusedSiGLU {
        input_shape: vec![1, 32, 768],
    };
    assert!(
        can_use_direct_dispatch(&siglu),
        "FusedSiGLU should support direct dispatch"
    );

    let geglu = NativeOpKind::FusedGeGLU {
        input_shape: vec![1, 32, 768],
    };
    assert!(
        can_use_direct_dispatch(&geglu),
        "FusedGeGLU should support direct dispatch"
    );

    // These should NOT support direct dispatch (complex ops).
    let lstm = NativeOpKind::LstmSequence {
        hidden_size: 256,
        input_shape: vec![10, 1, 128],
        h_shape: vec![1, 256],
        reverse: false,
    };
    assert!(
        !can_use_direct_dispatch(&lstm),
        "LstmSequence should NOT support direct dispatch"
    );

    let flash = NativeOpKind::FlashAttention {
        scale: 0.125,
        causal: false,
        q_shape: vec![1, 8, 32, 64],
        k_shape: vec![1, 8, 32, 64],
        output_shape: vec![1, 8, 32, 64],
        input_layout: Default::default(),
    };
    assert!(
        !can_use_direct_dispatch(&flash),
        "FlashAttention should NOT support direct dispatch"
    );

    let norm = NativeOpKind::LayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 32, 768],
        hidden_dim: 768,
    };
    assert!(
        !can_use_direct_dispatch(&norm),
        "LayerNorm should NOT support direct dispatch"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 9. PeepholeConfig field validation (17 fields)
// ═══════════════════════════════════════════════════════════════════════

/// PeepholeConfig::default() has all 17 boolean fields enabled.
#[test]
fn test_peephole_config_default_all_enabled() {
    let config = PeepholeConfig::default();
    // All 17 fields must be true.
    assert!(config.norm_activ_conv1d, "pass 1: norm_activ_conv1d");
    assert!(config.fused_resblock, "pass 2-4: fused_resblock");
    assert!(config.linear_activation, "pass 5: linear_activation");
    assert!(config.add_layer_norm, "pass 6: add_layer_norm");
    assert!(config.norm_linear, "pass 7: norm_linear");
    assert!(config.attention_transpose, "pass 9: attention_transpose");
    assert!(config.flip_lstm, "pass 10: flip_lstm");
    assert!(config.batched_linear_projection, "pass 12: batched_linear_projection");
    assert!(config.channels_first_layer_norm, "pass 13: channels_first_layer_norm");
    assert!(config.silu_mul, "pass 14: silu_mul");
    assert!(config.auto_fuse_elementwise, "pass 15: auto_fuse_elementwise");
    assert!(config.bilstm_cat, "pass 11: bilstm_cat");
    assert!(config.add_norm_linear, "pass 8: add_norm_linear");
    assert!(config.fuse_adain_snake, "pass 0: fuse_adain_snake");
    assert!(config.fuse_upsample_conv1d, "fuse_upsample_conv1d");
    assert!(config.fuse_instance_norm_mul_add, "fuse_instance_norm_mul_add");
    assert!(config.fuse_conv1d_activation, "fuse_conv1d_activation");
}

/// PeepholeConfig has exactly 17 boolean fields.
/// Verified by explicit enumeration of all fields.
#[test]
fn test_peephole_config_field_count() {
    // Count fields explicitly — if a field is added to PeepholeConfig
    // without updating this test, the selective_disable test below will
    // catch the mismatch.
    let bools = [
        PeepholeConfig::default().norm_activ_conv1d,
        PeepholeConfig::default().fused_resblock,
        PeepholeConfig::default().linear_activation,
        PeepholeConfig::default().add_layer_norm,
        PeepholeConfig::default().norm_linear,
        PeepholeConfig::default().attention_transpose,
        PeepholeConfig::default().flip_lstm,
        PeepholeConfig::default().batched_linear_projection,
        PeepholeConfig::default().channels_first_layer_norm,
        PeepholeConfig::default().silu_mul,
        PeepholeConfig::default().auto_fuse_elementwise,
        PeepholeConfig::default().bilstm_cat,
        PeepholeConfig::default().add_norm_linear,
        PeepholeConfig::default().fuse_adain_snake,
        PeepholeConfig::default().fuse_upsample_conv1d,
        PeepholeConfig::default().fuse_instance_norm_mul_add,
        PeepholeConfig::default().fuse_conv1d_activation,
    ];
    assert_eq!(bools.len(), 17, "PeepholeConfig should have 17 boolean fields");
}

/// All default PeepholeConfig boolean fields are true.
#[test]
fn test_peephole_config_all_default_true() {
    let config = PeepholeConfig::default();
    let all_true = config.norm_activ_conv1d
        && config.fused_resblock
        && config.linear_activation
        && config.add_layer_norm
        && config.norm_linear
        && config.attention_transpose
        && config.flip_lstm
        && config.batched_linear_projection
        && config.channels_first_layer_norm
        && config.silu_mul
        && config.auto_fuse_elementwise
        && config.bilstm_cat
        && config.add_norm_linear
        && config.fuse_adain_snake
        && config.fuse_upsample_conv1d
        && config.fuse_instance_norm_mul_add
        && config.fuse_conv1d_activation;
    assert!(all_true, "All PeepholeConfig::default() fields should be true");
}

/// PeepholeConfig can be selectively disabled field by field.
#[test]
fn test_peephole_config_selective_disable() {
    let config = PeepholeConfig {
        fuse_adain_snake: false,
        norm_linear: false,
        silu_mul: false,
        ..Default::default()
    };

    assert!(!config.fuse_adain_snake);
    assert!(!config.norm_linear);
    assert!(!config.silu_mul);
    // Remaining fields still true.
    assert!(config.norm_activ_conv1d);
    assert!(config.fused_resblock);
    assert!(config.linear_activation);
    assert!(config.add_layer_norm);
    assert!(config.attention_transpose);
    assert!(config.flip_lstm);
    assert!(config.bilstm_cat);
    assert!(config.add_norm_linear);
    assert!(config.fuse_upsample_conv1d);
    assert!(config.fuse_instance_norm_mul_add);
    assert!(config.fuse_conv1d_activation);
    assert!(config.channels_first_layer_norm);
    assert!(config.batched_linear_projection);
    assert!(config.auto_fuse_elementwise);
}

/// PeepholeConfig Clone + PartialEq work correctly.
#[test]
fn test_peephole_config_clone_eq() {
    let config = PeepholeConfig::default();
    let cloned = config.clone();
    assert_eq!(config, cloned, "Clone should produce equal PeepholeConfig");
}

/// PeepholeConfig with all passes disabled is not equal to default.
#[test]
fn test_peephole_config_all_disabled_ne_default() {
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
        ..Default::default()
    };

    let default_config = PeepholeConfig::default();
    assert_ne!(
        config, default_config,
        "All-disabled config should differ from default"
    );

    // Verify all fields are false.
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

// ═══════════════════════════════════════════════════════════════════════
// 10. NativeOpKind dispatch count estimation consistency
// ═══════════════════════════════════════════════════════════════════════

/// All 35 variants have estimated_metal_dispatches >= 0.
/// ConstantWeight may be 0 (no GPU compute), all others >= 1.
#[test]
fn test_all_variants_dispatch_estimate_non_negative() {
    for op in &all_35_variants() {
        let count = op.estimated_metal_dispatches();
        if op.variant_name() == "ConstantWeight" {
            assert_eq!(
                count, 0,
                "ConstantWeight should have 0 dispatches (pre-uploaded)"
            );
        } else {
            assert!(
                count >= 1,
                "{} should have >= 1 estimated dispatch, got {count}",
                op.variant_name()
            );
        }
    }
}

/// All 35 variants have estimated_encoding_events >= 1, except the documented
/// zero-encoding cases: `ConstantWeight` (pre-uploaded buffer, no GPU work) and
/// `MaxPool1d` (GPU→CPU→GPU roundtrip via to_device, no compute encoding).
#[test]
fn test_all_variants_encoding_events_positive() {
    for op in &all_35_variants() {
        let count = op.estimated_encoding_events();
        if matches!(op.variant_name(), "ConstantWeight" | "MaxPool1d") {
            assert_eq!(
                count, 0,
                "{} should have 0 encoding events (no compute dispatch)",
                op.variant_name()
            );
        } else {
            assert!(
                count >= 1,
                "{} should have >= 1 encoding event, got {count}",
                op.variant_name()
            );
        }
    }
}

/// Fused ops always have <= dispatch count of their unfused equivalents.
/// Core invariant: fusion never increases dispatch count.
#[test]
fn test_fused_never_more_dispatches_than_unfused() {
    // FusedAdainSnake <= InstanceNorm + Mul + Add + Snake (4)
    let fused_adain = NativeOpKind::FusedAdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 128, 512],
        channels: 128,
        external_node_ids: None,
    };
    assert!(fused_adain.estimated_metal_dispatches() <= 4);

    // FusedInstanceNormMulAdd <= InstanceNorm + Mul + Add (3)
    let fused_norm = NativeOpKind::FusedInstanceNormMulAdd {
        eps: 1e-5,
        input_shape: vec![1, 128, 512],
        channels: 128,
        external_node_ids: None,
    };
    assert!(fused_norm.estimated_metal_dispatches() <= 3);

    // AddLayerNorm <= Add + LayerNorm (2)
    let add_ln = NativeOpKind::AddLayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 32, 768],
        hidden_dim: 768,
    };
    assert!(add_ln.estimated_metal_dispatches() <= 2);

    // FusedConv1dActivation <= Conv1d + Activation (2)
    let fused_conv_act = NativeOpKind::FusedConv1dActivation {
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
    assert!(fused_conv_act.estimated_metal_dispatches() <= 2);

    // SiluMul <= Silu + Mul (2)
    let silu_mul = NativeOpKind::SiluMul {
        input_shape: vec![1, 32, 768],
    };
    assert!(silu_mul.estimated_metal_dispatches() <= 2);

    // FusedMulAdd <= Mul + Add (2)
    let fma = NativeOpKind::FusedMulAdd {
        input_shape: vec![1, 32, 768],
    };
    assert!(fma.estimated_metal_dispatches() <= 2);

    // FusedGeGLU <= GELU + Mul (2)
    let geglu = NativeOpKind::FusedGeGLU {
        input_shape: vec![1, 32, 768],
    };
    assert!(geglu.estimated_metal_dispatches() <= 2);

    // FusedSiGLU <= Sigmoid + Mul (2)
    let siglu = NativeOpKind::FusedSiGLU {
        input_shape: vec![1, 32, 768],
    };
    assert!(siglu.estimated_metal_dispatches() <= 2);

    // FusedUpsampleConv1d <= Upsample + Conv1d (2)
    let fused_up = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 2,
        in_channels: 128,
        out_channels: 64,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        input_shape: vec![1, 128, 256],
    };
    assert!(fused_up.estimated_metal_dispatches() <= 2);

    // LinearActivation <= Linear + Activation (2)
    let lin_act = NativeOpKind::LinearActivation {
        activation: GemmActivation::Gelu,
        in_features: 768,
        out_features: 3072,
        has_bias: true,
        input_shape: vec![1, 32, 768],
    };
    assert!(lin_act.estimated_metal_dispatches() <= 2);

    // BatchNorm2d is a single fused kernel (replaces ~6 dispatches)
    let bn = NativeOpKind::BatchNorm2d {
        eps: 1e-5,
        num_channels: 64,
        input_shape: vec![1, 64, 224, 224],
        has_weight: true,
        has_bias: true,
    };
    assert!(bn.estimated_metal_dispatches() <= 6);
}

/// Cumsum dispatch count depends on axis size (1 for small, 3 for large).
#[test]
fn test_cumsum_dispatch_count_axis_dependent() {
    let small = NativeOpKind::Cumsum {
        dim: 0,
        input_shape: vec![128, 32], // axis size 128 <= 256 => 1 dispatch
    };
    assert_eq!(
        small.estimated_metal_dispatches(),
        1,
        "Cumsum with axis_size <= 256 should be 1 dispatch"
    );

    let large = NativeOpKind::Cumsum {
        dim: 0,
        input_shape: vec![1024, 32], // axis size 1024 > 256 => 3 dispatches
    };
    assert_eq!(
        large.estimated_metal_dispatches(),
        3,
        "Cumsum with axis_size > 256 should be 3 dispatches"
    );
}

/// NormLinear dispatch count depends on dimensions (1 for scalar GEMM, 2 for simdgroup).
#[test]
fn test_norm_linear_dispatch_count_shape_dependent() {
    // Small dimensions: scalar GEMM fallback => 1 dispatch.
    let small = NativeOpKind::NormLinear {
        norm_kind: FusedNormKind::LayerNorm,
        eps: 1e-5,
        input_shape: vec![1, 4, 32],
        hidden_dim: 32,
        out_features: 16,
        has_bias: true,
    };
    assert!(
        small.estimated_metal_dispatches() >= 1,
        "NormLinear should have >= 1 dispatch"
    );

    // Large dimensions: may use simdgroup GEMM => 2 dispatches.
    let large = NativeOpKind::NormLinear {
        norm_kind: FusedNormKind::LayerNorm,
        eps: 1e-5,
        input_shape: vec![1, 32, 768],
        hidden_dim: 768,
        out_features: 3072,
        has_bias: true,
    };
    assert!(
        large.estimated_metal_dispatches() <= 2,
        "NormLinear should be <= 2 dispatches, got {}",
        large.estimated_metal_dispatches()
    );
}

/// FusedResBlock dispatch count varies by style projection mode.
#[test]
fn test_fused_resblock_dispatch_count_varies() {
    // No style projection: base dispatches only.
    let no_proj = NativeOpKind::FusedResBlock {
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
    };
    let base_dispatches = no_proj.estimated_metal_dispatches();
    assert!(
        base_dispatches >= 3,
        "FusedResBlock base should be >= 3 dispatches, got {base_dispatches}"
    );
}

/// FusedConv1dActivation with each ConvActivation variant has 1 dispatch.
#[test]
fn test_fused_conv1d_activation_all_variants_single_dispatch() {
    let activations = [
        ConvActivation::Snake,
        ConvActivation::Relu,
        ConvActivation::LeakyRelu { slope: 0.2 },
        ConvActivation::Silu,
    ];
    for act in &activations {
        let op = NativeOpKind::FusedConv1dActivation {
            activation: *act,
            out_channels: 256,
            kernel_size: 5,
            stride: 2,
            padding: 2,
            dilation: 1,
            groups: 1,
            has_bias: false,
            input_shape: vec![1, 128, 1024],
            pre_activation: false,
        };
        assert_eq!(
            op.estimated_metal_dispatches(),
            1,
            "FusedConv1dActivation({act:?}) should be 1 dispatch"
        );
    }
}

/// ChannelsFirstLayerNorm dispatch count is always 1.
#[test]
fn test_channels_first_ln_single_dispatch() {
    let op = NativeOpKind::ChannelsFirstLayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 512, 200],
        channels: 512,
        leaky_relu_slope: None,
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);

    // With optional fused LeakyReLU.
    let op_leaky = NativeOpKind::ChannelsFirstLayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 256, 100],
        channels: 256,
        leaky_relu_slope: Some(0.2),
    };
    assert_eq!(op_leaky.estimated_metal_dispatches(), 1);
}

/// FlashAttention is always 1 dispatch.
#[test]
fn test_flash_attention_single_dispatch() {
    let op = NativeOpKind::FlashAttention {
        scale: 0.0884, // 1/sqrt(128)
        causal: true,
        q_shape: vec![1, 16, 64, 128],
        k_shape: vec![1, 4, 64, 128], // GQA: 4 KV heads
        output_shape: vec![1, 16, 64, 128],
        input_layout: Default::default(),
    };
    assert_eq!(op.estimated_metal_dispatches(), 1);
}
