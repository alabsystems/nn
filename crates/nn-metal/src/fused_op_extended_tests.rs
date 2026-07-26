// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Part of #4252.
//!
//! Extended tests for fused NativeOp operations: FusedAdainSnake,
//! FusedInstanceNormMulAdd, FusedConv1dActivation, FusedUpsampleConv1d.
//!
//! These tests verify construction, serialization, dispatch counting,
//! and variant classification for fused operations used in the Kokoro
//! compiled model pipeline.

use nn_dsl::trace_compile::{ConvActivation, NativeOpKind};

// ===========================================================================
// 1. FusedAdainSnake — construction, dispatch, serde
// ===========================================================================

#[test]
fn fused_adain_snake_construction_no_external_ids() {
    let op = NativeOpKind::FusedAdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 128, 512],
        channels: 128,
        external_node_ids: None,
    };
    assert_eq!(op.variant_name(), "FusedAdainSnake");
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

#[test]
fn fused_adain_snake_construction_with_external_ids() {
    let op = NativeOpKind::FusedAdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 256, 1024],
        channels: 256,
        external_node_ids: Some(vec![10, 20, 30]),
    };
    assert_eq!(op.variant_name(), "FusedAdainSnake");
    let dbg = format!("{op:?}");
    assert!(dbg.contains("256"), "Debug: {dbg}");
    assert!(dbg.contains("external_node_ids: Some"), "Debug: {dbg}");
}

#[test]
fn fused_adain_snake_serde_round_trip() {
    let op = NativeOpKind::FusedAdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 128, 512],
        channels: 128,
        external_node_ids: Some(vec![1, 2, 3]),
    };
    let json = serde_json::to_string(&op).expect("serialize FusedAdainSnake");
    let deserialized: NativeOpKind = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.variant_name(), "FusedAdainSnake");
    assert!(json.contains("FusedAdainSnake"), "JSON: {json}");
}

#[test]
fn fused_adain_snake_saves_dispatches_vs_unfused() {
    // Unfused AdaIN+Snake: InstanceNorm(~7) + Mul(1) + Add(1) + Snake(~2) = ~11 dispatches.
    // FusedAdainSnake: 1 dispatch. Verifying it is indeed 1.
    let fused = NativeOpKind::FusedAdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 128, 512],
        channels: 128,
        external_node_ids: None,
    };
    let unfused_instance_norm = NativeOpKind::InstanceNorm {
        eps: 1e-5,
        input_shape: vec![1, 128, 512],
    };
    // Fused should use fewer dispatches than InstanceNorm alone.
    assert!(
        fused.estimated_metal_dispatches() <= unfused_instance_norm.estimated_metal_dispatches(),
        "Fused ({}) should be <= InstanceNorm alone ({})",
        fused.estimated_metal_dispatches(),
        unfused_instance_norm.estimated_metal_dispatches()
    );
}

// ===========================================================================
// 2. FusedInstanceNormMulAdd — construction, dispatch, serde
// ===========================================================================

#[test]
fn fused_instance_norm_mul_add_construction_no_external_ids() {
    let op = NativeOpKind::FusedInstanceNormMulAdd {
        eps: 1e-5,
        input_shape: vec![1, 128, 512],
        channels: 128,
        external_node_ids: None,
    };
    assert_eq!(op.variant_name(), "FusedInstanceNormMulAdd");
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

#[test]
fn fused_instance_norm_mul_add_construction_with_external_ids() {
    let op = NativeOpKind::FusedInstanceNormMulAdd {
        eps: 1e-6,
        input_shape: vec![2, 64, 256],
        channels: 64,
        external_node_ids: Some(vec![5, 10, 15]),
    };
    assert_eq!(op.variant_name(), "FusedInstanceNormMulAdd");
    let dbg = format!("{op:?}");
    assert!(dbg.contains("64"), "Debug: {dbg}");
    assert!(dbg.contains("1e-6") || dbg.contains("0.000001"), "Debug: {dbg}");
}

#[test]
fn fused_instance_norm_mul_add_serde_round_trip() {
    let op = NativeOpKind::FusedInstanceNormMulAdd {
        eps: 1e-5,
        input_shape: vec![1, 128, 512],
        channels: 128,
        external_node_ids: None,
    };
    let json = serde_json::to_string(&op).expect("serialize FusedInstanceNormMulAdd");
    let deserialized: NativeOpKind = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.variant_name(), "FusedInstanceNormMulAdd");
    assert!(json.contains("FusedInstanceNormMulAdd"), "JSON: {json}");
}

#[test]
fn fused_instance_norm_mul_add_saves_dispatches_vs_unfused() {
    // Unfused: InstanceNorm(1) + Mul(1) + Add(1) = 3 dispatches.
    // FusedInstanceNormMulAdd: 1 dispatch.
    let fused = NativeOpKind::FusedInstanceNormMulAdd {
        eps: 1e-5,
        input_shape: vec![1, 128, 512],
        channels: 128,
        external_node_ids: None,
    };
    assert_eq!(fused.estimated_metal_dispatches(), 1);
}

// ===========================================================================
// 3. FusedConv1dActivation — construction, dispatch, serde, all activations
// ===========================================================================

#[test]
fn fused_conv1d_activation_relu_construction() {
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
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

#[test]
fn fused_conv1d_activation_snake_construction() {
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
fn fused_conv1d_activation_leaky_relu_construction() {
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
fn fused_conv1d_activation_silu_construction() {
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
fn fused_conv1d_activation_serde_round_trip_all_activations() {
    let activations = [
        ConvActivation::Snake,
        ConvActivation::Relu,
        ConvActivation::LeakyRelu { slope: 0.01 },
        ConvActivation::Silu,
    ];
    for act in &activations {
        let op = NativeOpKind::FusedConv1dActivation {
            activation: *act,
            out_channels: 128,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            has_bias: true,
            input_shape: vec![1, 64, 512],
            pre_activation: false,
        };
        let json = serde_json::to_string(&op).expect("serialize");
        let deserialized: NativeOpKind = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            deserialized.variant_name(),
            "FusedConv1dActivation",
            "Round-trip failed for {act:?}"
        );
    }
}

#[test]
fn fused_conv1d_activation_saves_dispatch_vs_separate() {
    // Separate: Conv1dGemm(1) + activation(1) = 2 dispatches.
    // FusedConv1dActivation: 1 dispatch.
    let fused = NativeOpKind::FusedConv1dActivation {
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
    let separate_conv = NativeOpKind::Conv1dGemm {
        input_shape: vec![1, 128, 512],
        out_channels: 256,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
        has_bias: true,
    };
    // Fused should use strictly fewer dispatches than conv alone + 1 activation.
    let separate_total = separate_conv.estimated_metal_dispatches() + 1;
    assert!(
        fused.estimated_metal_dispatches() < separate_total,
        "Fused ({}) should be < separate ({} + 1 = {})",
        fused.estimated_metal_dispatches(),
        separate_conv.estimated_metal_dispatches(),
        separate_total
    );
}

#[test]
fn fused_conv1d_activation_dilated_construction() {
    // Non-trivial dilation and stride configuration.
    let op = NativeOpKind::FusedConv1dActivation {
        activation: ConvActivation::Snake,
        out_channels: 128,
        kernel_size: 3,
        stride: 1,
        padding: 3,
        dilation: 3,
        groups: 1,
        has_bias: true,
        input_shape: vec![1, 128, 512],
        pre_activation: false,
    };
    assert_eq!(op.variant_name(), "FusedConv1dActivation");
    assert_eq!(op.estimated_metal_dispatches(), 1);
}

// ===========================================================================
// 4. FusedUpsampleConv1d — construction, dispatch, serde
// ===========================================================================

#[test]
fn fused_upsample_conv1d_construction() {
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 2,
        in_channels: 128,
        out_channels: 64,
        kernel_size: 3,
        stride: 1,
        padding: 1,
        input_shape: vec![1, 128, 256],
    };
    assert_eq!(op.variant_name(), "FusedUpsampleConv1d");
    assert_eq!(op.estimated_metal_dispatches(), 1);
    assert_eq!(op.estimated_encoding_events(), 1);
}

#[test]
fn fused_upsample_conv1d_serde_round_trip() {
    let op = NativeOpKind::FusedUpsampleConv1d {
        upsample_factor: 4,
        in_channels: 256,
        out_channels: 128,
        kernel_size: 5,
        stride: 1,
        padding: 2,
        input_shape: vec![1, 256, 128],
    };
    let json = serde_json::to_string(&op).expect("serialize FusedUpsampleConv1d");
    let deserialized: NativeOpKind = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.variant_name(), "FusedUpsampleConv1d");
    assert!(json.contains("FusedUpsampleConv1d"), "JSON: {json}");
    assert!(json.contains("\"upsample_factor\":4"), "JSON: {json}");
}

#[test]
fn fused_upsample_conv1d_various_factors() {
    for factor in [2, 4, 8] {
        let op = NativeOpKind::FusedUpsampleConv1d {
            upsample_factor: factor,
            in_channels: 128,
            out_channels: 64,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            input_shape: vec![1, 128, 256],
        };
        assert_eq!(op.variant_name(), "FusedUpsampleConv1d");
        assert_eq!(
            op.estimated_metal_dispatches(),
            1,
            "Upsample factor {factor} should still be 1 dispatch"
        );
    }
}

// ===========================================================================
// 5. Cross-variant dispatch count comparisons
// ===========================================================================

#[test]
fn all_fused_ops_single_dispatch() {
    // All fused ops should be single-dispatch kernels.
    let fused_ops: Vec<NativeOpKind> = vec![
        NativeOpKind::FusedAdainSnake {
            eps: 1e-5,
            input_shape: vec![1, 128, 512],
            channels: 128,
            external_node_ids: None,
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
        NativeOpKind::FusedUpsampleConv1d {
            upsample_factor: 2,
            in_channels: 128,
            out_channels: 64,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            input_shape: vec![1, 128, 256],
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
        // FusedLayerNormLinear with small dims (scalar GEMM fallback = 1 dispatch).
        // Large dims (hidden_dim=768, out_features=3072) would route to simdgroup = 2.
        NativeOpKind::FusedLayerNormLinear {
            eps: 1e-5,
            input_shape: vec![1, 4, 32],
            hidden_dim: 32,
            out_features: 16,
            has_bias: true,
        },
    ];
    for op in &fused_ops {
        assert_eq!(
            op.estimated_metal_dispatches(),
            1,
            "{} should be 1 dispatch, got {}",
            op.variant_name(),
            op.estimated_metal_dispatches()
        );
        assert_eq!(
            op.estimated_encoding_events(),
            1,
            "{} should be 1 encoding event, got {}",
            op.variant_name(),
            op.estimated_encoding_events()
        );
    }
}

// ===========================================================================
// 5b. FusedLayerNormLinear dispatch count is shape-dependent
// ===========================================================================

#[test]
fn fused_layer_norm_linear_dispatch_count_shape_dependent() {
    // Small dims: scalar GEMM fallback => 1 dispatch.
    let small = NativeOpKind::FusedLayerNormLinear {
        eps: 1e-5,
        input_shape: vec![1, 4, 32],
        hidden_dim: 32,
        out_features: 16,
        has_bias: true,
    };
    assert_eq!(
        small.estimated_metal_dispatches(),
        1,
        "Small dims should use scalar fallback (1 dispatch)"
    );

    // Large dims: simdgroup GEMM path => 2 dispatches (norm + matmul).
    // M=128, K=768, N=3072: all mult-of-8, M*N >= 16384, K >= 128.
    let large = NativeOpKind::FusedLayerNormLinear {
        eps: 1e-5,
        input_shape: vec![1, 128, 768],
        hidden_dim: 768,
        out_features: 3072,
        has_bias: true,
    };
    assert_eq!(
        large.estimated_metal_dispatches(),
        2,
        "Large dims should use simdgroup (2 dispatches)"
    );
}

// ===========================================================================
// 6. Fused op clone and Debug consistency
// ===========================================================================

#[test]
fn fused_ops_clone_debug_consistency() {
    let ops: Vec<NativeOpKind> = vec![
        NativeOpKind::FusedAdainSnake {
            eps: 1e-5,
            input_shape: vec![1, 128, 512],
            channels: 128,
            external_node_ids: None,
        },
        NativeOpKind::FusedInstanceNormMulAdd {
            eps: 1e-5,
            input_shape: vec![1, 64, 256],
            channels: 64,
            external_node_ids: Some(vec![1, 2, 3]),
        },
        NativeOpKind::FusedConv1dActivation {
            activation: ConvActivation::LeakyRelu { slope: 0.1 },
            out_channels: 128,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            has_bias: true,
            input_shape: vec![1, 64, 512],
            pre_activation: false,
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
    ];
    for op in &ops {
        let cloned = op.clone();
        let dbg_orig = format!("{op:?}");
        let dbg_clone = format!("{cloned:?}");
        assert_eq!(
            dbg_orig, dbg_clone,
            "Clone mismatch for {}",
            op.variant_name()
        );
    }
}
