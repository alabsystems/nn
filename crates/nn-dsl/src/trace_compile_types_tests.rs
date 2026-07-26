// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `NativeOpKind::estimated_metal_dispatches()` and
//! `estimated_encoding_events()`.

use super::*;
use crate::trace_compile::{
    AttentionLayout, NormActivConv1dParams, NormActivation, StyleBatchOffset, StyleProjectionParams,
};

#[test]
fn test_estimated_metal_dispatches_fused_ops_return_one() {
    let lstm = NativeOpKind::LstmSequence {
        hidden_size: 256,
        input_shape: vec![50, 1, 512],
        h_shape: vec![1, 256],
        reverse: false,
    };
    assert_eq!(lstm.estimated_metal_dispatches(), 1);

    let inorm = NativeOpKind::InstanceNorm {
        eps: 1e-5,
        input_shape: vec![1, 256, 100],
    };
    assert_eq!(inorm.estimated_metal_dispatches(), 1);

    let flash = NativeOpKind::FlashAttention {
        scale: 0.125,
        causal: false,
        q_shape: vec![1, 12, 50, 64],
        k_shape: vec![1, 12, 50, 64],
        output_shape: vec![1, 12, 50, 64],
        input_layout: AttentionLayout::HeadsFirst,
    };
    assert_eq!(flash.estimated_metal_dispatches(), 1);

    let adain_snake = NativeOpKind::AdainSnake {
        eps: 1e-5,
        input_shape: vec![1, 256, 100],
        channels: 256,
        residual_gamma: true,
        external_node_ids: None,
    };
    assert_eq!(adain_snake.estimated_metal_dispatches(), 1);

    let adain_lr = NativeOpKind::AdainLeakyRelu {
        eps: 1e-5,
        slope: 0.2,
        input_shape: vec![1, 256, 100],
        external_node_ids: None,
    };
    assert_eq!(adain_lr.estimated_metal_dispatches(), 1);

    let adaln = NativeOpKind::AdaLayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 50, 768],
        hidden_dim: 768,
    };
    assert_eq!(adaln.estimated_metal_dispatches(), 1);

    let pool = NativeOpKind::MaxPool1d {
        kernel_size: 3,
        stride: 2,
        padding: 1,
        input_shape: vec![1, 64, 100],
    };
    assert_eq!(pool.estimated_metal_dispatches(), 1);
}

#[test]
fn test_estimated_metal_dispatches_layer_norm_is_1() {
    // Fused single-dispatch kernel (#2937).
    let ln = NativeOpKind::LayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 50, 768],
        hidden_dim: 768,
    };
    assert_eq!(ln.estimated_metal_dispatches(), 1);
}

#[test]
fn test_estimated_metal_dispatches_constant_weight_is_zero() {
    let cw = NativeOpKind::ConstantWeight {
        name: "arange".into(),
        shape: vec![100],
    };
    assert_eq!(cw.estimated_metal_dispatches(), 0);
}

#[test]
fn test_estimated_metal_dispatches_fused_resblock() {
    let params = NormActivConv1dParams {
        activation: NormActivation::Snake,
        eps: 1e-5,
        conv_dilation: 1,
        conv_padding: 1,
        input_shape: vec![1, 256, 100],
        output_channels: 256,
        kernel_size: 3,
    };
    // Conv-stats fusion (#1815 Tier 2): phase 1 conv writes output stats
    // in its epilogue, phase 2 conv uses precomputed stats.
    // 3 base dispatches: p1_stats + p1_conv_with_stats + p2_conv_precomputed.
    let rb_no_scale = NativeOpKind::FusedResBlock {
        phase1: params.clone(),
        phase2: params.clone(),
        input_steps: vec![0, 1, 2, 3, 4],
        residual_scale: 1.0,
        style_proj: None,
        shortcut_step: None,
        pool_step: None,
        style_batch_offset: None,
    };
    assert_eq!(rb_no_scale.estimated_metal_dispatches(), 3);

    // scale = 1/√2 → residual fused in phase 2, still 3 dispatches.
    let rb_scaled = NativeOpKind::FusedResBlock {
        phase1: params.clone(),
        phase2: params.clone(),
        input_steps: vec![0, 1, 2, 3, 4],
        residual_scale: std::f32::consts::FRAC_1_SQRT_2,
        style_proj: None,
        shortcut_step: None,
        pool_step: None,
        style_batch_offset: None,
    };
    assert_eq!(rb_scaled.estimated_metal_dispatches(), 3);

    // style_proj → 3 base + 4 proj = 7.
    let rb_style = NativeOpKind::FusedResBlock {
        phase1: params.clone(),
        phase2: params.clone(),
        input_steps: vec![0, 1],
        residual_scale: std::f32::consts::FRAC_1_SQRT_2,
        style_proj: Some(StyleProjectionParams {
            channels1: 256,
            channels2: 256,
            style_dim: 128,
        }),
        shortcut_step: None,
        pool_step: None,
        style_batch_offset: None,
    };
    assert_eq!(rb_style.estimated_metal_dispatches(), 7);

    // style_batch_offset → 3 base + 0 proj = 3 (zero-copy narrow).
    let rb_batched = NativeOpKind::FusedResBlock {
        phase1: params.clone(),
        phase2: params,
        input_steps: vec![0, 99],
        residual_scale: std::f32::consts::FRAC_1_SQRT_2,
        style_proj: None,
        shortcut_step: None,
        pool_step: None,
        style_batch_offset: Some(StyleBatchOffset::new(0, 256, 256)),
    };
    assert_eq!(rb_batched.estimated_metal_dispatches(), 3);
}

#[test]
fn test_estimated_metal_dispatches_norm_activ_conv1d() {
    let nac = NativeOpKind::NormActivConv1d {
        activation: NormActivation::LeakyRelu { slope: 0.2 },
        eps: 1e-5,
        conv_dilation: 1,
        conv_padding: 1,
        input_shape: vec![1, 256, 100],
        output_channels: 256,
        kernel_size: 3,
        external_node_ids: None,
    };
    assert_eq!(nac.estimated_metal_dispatches(), 2);
}

#[test]
fn test_estimated_metal_dispatches_cumsum_depends_on_axis() {
    let small = NativeOpKind::Cumsum {
        dim: 0,
        input_shape: vec![200],
    };
    assert_eq!(small.estimated_metal_dispatches(), 1);

    let large = NativeOpKind::Cumsum {
        dim: 0,
        input_shape: vec![1000],
    };
    assert_eq!(large.estimated_metal_dispatches(), 3);
}

// --- estimated_encoding_events tests (#1815 D5.2) ---

#[test]
fn test_encoding_events_lstm_includes_bias_combine() {
    // LSTM always has bias_ih + bias_hh in PyTorch = +1 GPU add encoding event.
    let lstm = NativeOpKind::LstmSequence {
        hidden_size: 256,
        input_shape: vec![50, 1, 512],
        h_shape: vec![1, 256],
        reverse: false,
    };
    assert_eq!(
        lstm.estimated_encoding_events(),
        2,
        "LSTM = 1 bias combine + 1 kernel"
    );
    // Metal dispatches only counts the kernel, not the bias combine.
    assert_eq!(lstm.estimated_metal_dispatches(), 1);
}

#[test]
fn test_encoding_events_cumsum_always_one_batch() {
    // Multi-pass cumsum uses 3 sub-encoders in 1 get_or_create_batch().
    let small = NativeOpKind::Cumsum {
        dim: 0,
        input_shape: vec![200],
    };
    assert_eq!(small.estimated_encoding_events(), 1);

    let large = NativeOpKind::Cumsum {
        dim: 0,
        input_shape: vec![1000],
    };
    assert_eq!(
        large.estimated_encoding_events(),
        1,
        "multi-pass = 1 batch, not 3"
    );
    assert_eq!(
        large.estimated_metal_dispatches(),
        3,
        "but 3 metal dispatches"
    );
}

#[test]
fn test_encoding_events_norm_activ_conv1d_is_one() {
    // 1 get_or_create_batch with 2 sub-encoders (stats + conv).
    let nac = NativeOpKind::NormActivConv1d {
        activation: NormActivation::LeakyRelu { slope: 0.2 },
        eps: 1e-5,
        conv_dilation: 1,
        conv_padding: 1,
        input_shape: vec![1, 256, 100],
        output_channels: 256,
        kernel_size: 3,
        external_node_ids: None,
    };
    assert_eq!(
        nac.estimated_encoding_events(),
        1,
        "1 batch, 2 sub-encoders"
    );
    assert_eq!(nac.estimated_metal_dispatches(), 2);
}

#[test]
fn test_encoding_events_fused_resblock() {
    let params = NormActivConv1dParams {
        activation: NormActivation::Snake,
        eps: 1e-5,
        conv_dilation: 1,
        conv_padding: 1,
        input_shape: vec![1, 256, 100],
        output_channels: 256,
        kernel_size: 3,
    };
    // Base: 2 encoding events (phase 1 + phase 2), not 3.
    let rb = NativeOpKind::FusedResBlock {
        phase1: params.clone(),
        phase2: params.clone(),
        input_steps: vec![0, 1, 2, 3, 4],
        residual_scale: 1.0,
        style_proj: None,
        shortcut_step: None,
        pool_step: None,
        style_batch_offset: None,
    };
    assert_eq!(rb.estimated_encoding_events(), 2, "2 phases = 2 batches");
    assert_eq!(rb.estimated_metal_dispatches(), 3, "but 3 metal dispatches");

    // style_proj adds 4 encoding events (2 projections × 2).
    let rb_style = NativeOpKind::FusedResBlock {
        phase1: params.clone(),
        phase2: params.clone(),
        input_steps: vec![0, 1],
        residual_scale: std::f32::consts::FRAC_1_SQRT_2,
        style_proj: Some(StyleProjectionParams {
            channels1: 256,
            channels2: 256,
            style_dim: 128,
        }),
        shortcut_step: None,
        pool_step: None,
        style_batch_offset: None,
    };
    assert_eq!(rb_style.estimated_encoding_events(), 6, "2 base + 4 proj");

    // Batched style offset = 0 proj encoding events.
    let rb_batched = NativeOpKind::FusedResBlock {
        phase1: params.clone(),
        phase2: params,
        input_steps: vec![0, 99],
        residual_scale: std::f32::consts::FRAC_1_SQRT_2,
        style_proj: None,
        shortcut_step: None,
        pool_step: None,
        style_batch_offset: Some(StyleBatchOffset::new(0, 256, 256)),
    };
    assert_eq!(rb_batched.estimated_encoding_events(), 2, "2 base + 0 proj");
}

#[test]
fn test_encoding_events_max_pool1d_is_zero() {
    // MaxPool1d does CPU roundtrip — no GPU compute dispatch.
    let pool = NativeOpKind::MaxPool1d {
        kernel_size: 3,
        stride: 2,
        padding: 1,
        input_shape: vec![1, 64, 100],
    };
    assert_eq!(
        pool.estimated_encoding_events(),
        0,
        "CPU roundtrip = 0 encoding events"
    );
    assert_eq!(
        pool.estimated_metal_dispatches(),
        1,
        "but counted as 1 metal dispatch"
    );
}

#[test]
fn test_encoding_events_single_dispatch_ops() {
    // All fused single-dispatch NativeOps should be 1 encoding event.
    let ops: Vec<NativeOpKind> = vec![
        NativeOpKind::InstanceNorm {
            eps: 1e-5,
            input_shape: vec![1, 256, 100],
        },
        NativeOpKind::LayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 50, 768],
            hidden_dim: 768,
        },
        NativeOpKind::AddLayerNorm {
            eps: 1e-5,
            input_shape: vec![1, 50, 768],
            hidden_dim: 768,
        },
        NativeOpKind::FlashAttention {
            scale: 0.125,
            causal: false,
            q_shape: vec![1, 12, 50, 64],
            k_shape: vec![1, 12, 50, 64],
            output_shape: vec![1, 12, 50, 64],
            input_layout: AttentionLayout::HeadsFirst,
        },
        NativeOpKind::NormLinear {
            norm_kind: crate::trace_compile::FusedNormKind::LayerNorm,
            eps: 1e-5,
            input_shape: vec![1, 50, 768],
            hidden_dim: 768,
            out_features: 3072,
            has_bias: true,
        },
    ];
    for op in &ops {
        assert_eq!(
            op.estimated_encoding_events(),
            1,
            "{} should be 1 encoding event",
            op.variant_name()
        );
    }
}
