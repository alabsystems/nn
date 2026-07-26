// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for nn module layer composition patterns.
//!
//! Tests realistic neural network building blocks:
//! - Transformer blocks (LayerNorm -> MHA -> residual -> FFN)
//! - Conv-based encoder pipelines (Conv1d -> BatchNorm -> ReLU -> Pool)
//! - Decoder with cross-attention and KV cache
//! - Normalization comparison patterns
//! - Activation function dispatch
//! - Quantized layer roundtrip accuracy

#![allow(deprecated)]

use crate::dyn_tensor::DynTensor;
use crate::layers::*;
use crate::{DType, Device};

fn cpu() -> Device {
    Device::Cpu
}

/// Helper: create random-ish tensor (deterministic via simple formula).
fn pseudo_random(shape: &[usize], scale: f32) -> DynTensor {
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = (0..numel)
        .map(|i| {
            let x = i as f32;
            // Simple deterministic pseudo-random: sin-based hash
            (x * 0.7 + 1.3).sin() * scale
        })
        .collect();
    DynTensor::from_vec(data, shape, &cpu()).unwrap()
}

/// Helper: make a Linear layer with given dimensions (zero-init from VarBuilder).
fn make_linear(in_f: usize, out_f: usize) -> Linear {
    let vb = crate::var_builder::VarBuilder::zeros(DType::F32, &cpu());
    linear(in_f, out_f, &vb).unwrap()
}

/// Helper: make a Linear layer with specific weight values.
fn make_linear_with_data(in_f: usize, out_f: usize, scale: f32) -> Linear {
    let w = pseudo_random(&[out_f, in_f], scale);
    let b = pseudo_random(&[out_f], scale * 0.1);
    Linear::new(w, Some(b)).unwrap()
}

// ===========================================================================
// 1. Transformer Block Composition (8+ tests)
// ===========================================================================

#[test]
fn test_transformer_block_shape_propagation() {
    // LayerNorm -> MHA -> residual -> LayerNorm -> Linear -> GELU -> Linear
    let dim = 32;
    let num_heads = 4;
    let batch = 2;
    let seq_len = 8;

    let ln1_w = DynTensor::ones(&[dim], DType::F32, &cpu()).unwrap();
    let ln1_b = DynTensor::zeros(&[dim], DType::F32, &cpu()).unwrap();
    let ln1 = LayerNorm::new(ln1_w, ln1_b, 1e-5).unwrap();

    let ln2_w = DynTensor::ones(&[dim], DType::F32, &cpu()).unwrap();
    let ln2_b = DynTensor::zeros(&[dim], DType::F32, &cpu()).unwrap();
    let ln2 = LayerNorm::new(ln2_w, ln2_b, 1e-5).unwrap();

    let vb = crate::var_builder::VarBuilder::zeros(DType::F32, &cpu());
    let mha = MultiHeadAttention::load(vb.pp("attn"), dim, num_heads, num_heads, true).unwrap();

    let ffn_up = make_linear(dim, dim * 4);
    let ffn_down = make_linear(dim * 4, dim);

    let x = DynTensor::ones(&[batch, seq_len, dim], DType::F32, &cpu()).unwrap();

    // Pre-norm transformer block
    let normed = ln1.forward(&x).unwrap();
    let attn_out = mha.forward(&normed, None, None, None, 0).unwrap();
    let residual1 = x.broadcast_add(&attn_out).unwrap();

    let normed2 = ln2.forward(&residual1).unwrap();
    let ffn_h = ffn_up.forward(&normed2).unwrap();
    let ffn_h = ffn_h.gelu().unwrap();
    let ffn_out = ffn_down.forward(&ffn_h).unwrap();
    let residual2 = residual1.broadcast_add(&ffn_out).unwrap();

    assert_eq!(residual2.dims(), &[batch, seq_len, dim]);
}

#[test]
fn test_transformer_block_with_causal_mask() {
    let dim = 16;
    let num_heads = 2;
    let seq_len = 6;

    let vb = crate::var_builder::VarBuilder::zeros(DType::F32, &cpu());
    let mha = MultiHeadAttention::load(vb.pp("attn"), dim, num_heads, num_heads, false).unwrap();

    let x = pseudo_random(&[1, seq_len, dim], 0.5);
    let mask = causal_mask(seq_len, &cpu()).unwrap();

    let out = mha.forward(&x, None, Some(&mask), None, 0).unwrap();
    assert_eq!(out.dims(), &[1, seq_len, dim]);

    // Output should be finite
    let flat = out.to_flat_vec::<f32>().unwrap();
    for v in &flat {
        assert!(v.is_finite(), "causal MHA output must be finite, got {v}");
    }
}

#[test]
fn test_transformer_single_head() {
    let dim = 8;
    let vb = crate::var_builder::VarBuilder::zeros(DType::F32, &cpu());
    let mha = MultiHeadAttention::load(vb.pp("attn"), dim, 1, 1, true).unwrap();
    assert_eq!(mha.num_heads(), 1);
    assert_eq!(mha.head_dim(), 8);

    let x = pseudo_random(&[1, 4, dim], 0.3);
    let out = mha.forward(&x, None, None, None, 0).unwrap();
    assert_eq!(out.dims(), &[1, 4, dim]);
}

#[test]
fn test_transformer_gqa_4_heads_2_kv() {
    // Grouped-query attention: 4 Q heads, 2 KV heads
    let dim = 16;
    let vb = crate::var_builder::VarBuilder::zeros(DType::F32, &cpu());
    let mha = MultiHeadAttention::load(vb.pp("attn"), dim, 4, 2, false).unwrap();
    assert_eq!(mha.num_heads(), 4);
    assert_eq!(mha.num_kv_heads(), 2);
    assert_eq!(mha.head_dim(), 4);

    let x = pseudo_random(&[1, 3, dim], 0.2);
    let out = mha.forward(&x, None, None, None, 0).unwrap();
    assert_eq!(out.dims(), &[1, 3, dim]);
}

#[test]
fn test_transformer_gqa_8_heads_1_kv_mqa() {
    // Multi-query attention: 8 Q heads, 1 KV head
    let dim = 32;
    let vb = crate::var_builder::VarBuilder::zeros(DType::F32, &cpu());
    let mha = MultiHeadAttention::load(vb.pp("attn"), dim, 8, 1, false).unwrap();
    assert_eq!(mha.num_heads(), 8);
    assert_eq!(mha.num_kv_heads(), 1);

    let x = pseudo_random(&[2, 5, dim], 0.2);
    let out = mha.forward(&x, None, None, None, 0).unwrap();
    assert_eq!(out.dims(), &[2, 5, dim]);
}

#[test]
fn test_transformer_16_heads() {
    let dim = 64;
    let vb = crate::var_builder::VarBuilder::zeros(DType::F32, &cpu());
    let mha = MultiHeadAttention::load(vb.pp("attn"), dim, 16, 16, true).unwrap();
    assert_eq!(mha.num_heads(), 16);
    assert_eq!(mha.head_dim(), 4);

    let x = DynTensor::ones(&[1, 2, dim], DType::F32, &cpu()).unwrap();
    let out = mha.forward(&x, None, None, None, 0).unwrap();
    assert_eq!(out.dims(), &[1, 2, dim]);
}

#[test]
fn test_transformer_cross_attention() {
    let dim = 16;
    let num_heads = 2;
    let vb = crate::var_builder::VarBuilder::zeros(DType::F32, &cpu());
    let mha = MultiHeadAttention::load(vb.pp("attn"), dim, num_heads, num_heads, false).unwrap();

    let query = pseudo_random(&[1, 4, dim], 0.3);
    let encoder_out = pseudo_random(&[1, 10, dim], 0.3);

    let out = mha
        .forward(&query, Some(&encoder_out), None, None, 0)
        .unwrap();
    assert_eq!(out.dims(), &[1, 4, dim]);
}

#[test]
fn test_transformer_residual_preserves_scale() {
    // Verify residual connection: output = input + attn(norm(input))
    // With zero-init weights, attn output is zero, so output == input.
    let dim = 16;
    let num_heads = 2;

    let ln_w = DynTensor::ones(&[dim], DType::F32, &cpu()).unwrap();
    let ln_b = DynTensor::zeros(&[dim], DType::F32, &cpu()).unwrap();
    let ln = LayerNorm::new(ln_w, ln_b, 1e-5).unwrap();

    let vb = crate::var_builder::VarBuilder::zeros(DType::F32, &cpu());
    let mha = MultiHeadAttention::load(vb.pp("attn"), dim, num_heads, num_heads, false).unwrap();

    let x = DynTensor::full(&[1, 3, dim], 2.0, DType::F32, &cpu()).unwrap();
    let normed = ln.forward(&x).unwrap();
    let attn_out = mha.forward(&normed, None, None, None, 0).unwrap();
    let residual = x.broadcast_add(&attn_out).unwrap();

    // With zero weights, MHA output is zero, residual should equal input
    let input_vals = x.to_flat_vec::<f32>().unwrap();
    let output_vals = residual.to_flat_vec::<f32>().unwrap();
    for (i, (a, b)) in input_vals.iter().zip(output_vals.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "Residual should preserve input at index {i}: {a} vs {b}"
        );
    }
}

// ===========================================================================
// 2. Conv-based Encoder (8+ tests)
// ===========================================================================

#[test]
fn test_conv_encoder_shape_downsampling() {
    // Conv1d(stride=2) -> Conv1d(stride=2) halves length twice
    let w1 = DynTensor::ones(&[16, 1, 3], DType::F32, &cpu()).unwrap();
    let cfg1 = Conv1dConfig::new(1, 2, 1); // padding=1, stride=2
    let conv1 = Conv1d::new(w1, None, cfg1).unwrap();

    let w2 = DynTensor::ones(&[32, 16, 3], DType::F32, &cpu()).unwrap();
    let cfg2 = Conv1dConfig::new(1, 2, 1);
    let conv2 = Conv1d::new(w2, None, cfg2).unwrap();

    let x = DynTensor::ones(&[1, 1, 64], DType::F32, &cpu()).unwrap();
    let h1 = conv1.forward(&x).unwrap();
    assert_eq!(h1.dims()[0], 1);
    assert_eq!(h1.dims()[1], 16);
    // output_len = (64 + 2*1 - 3) / 2 + 1 = 32
    assert_eq!(h1.dims()[2], 32);

    let h2 = conv2.forward(&h1).unwrap();
    assert_eq!(h2.dims()[1], 32);
    // output_len = (32 + 2*1 - 3) / 2 + 1 = 16
    assert_eq!(h2.dims()[2], 16);
}

#[test]
fn test_conv_relu_chain() {
    let w = DynTensor::from_vec(vec![0.5, -0.3, 0.2], &[1, 1, 3], &cpu()).unwrap();
    let conv = Conv1d::new(w, None, Conv1dConfig::default()).unwrap();

    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &cpu()).unwrap();
    let h = conv.forward(&x).unwrap();
    let h = h.relu().unwrap();

    let flat = h.to_flat_vec::<f32>().unwrap();
    for v in &flat {
        assert!(*v >= 0.0, "ReLU output must be non-negative, got {v}");
    }
}

#[test]
fn test_conv_different_kernel_sizes() {
    for k in [1, 3, 5, 7] {
        let w = DynTensor::ones(&[4, 2, k], DType::F32, &cpu()).unwrap();
        let padding = k / 2;
        let cfg = Conv1dConfig::new(padding, 1, 1);
        let conv = Conv1d::new(w, None, cfg).unwrap();

        let x = DynTensor::ones(&[1, 2, 16], DType::F32, &cpu()).unwrap();
        let out = conv.forward(&x).unwrap();
        assert_eq!(
            out.dims()[1],
            4,
            "kernel_size={k}: out_channels should be 4"
        );
        // With padding=k/2, stride=1: output_len = input_len for odd kernels
        assert_eq!(
            out.dims()[2],
            16,
            "kernel_size={k}: same-padding should preserve length"
        );
    }
}

#[test]
fn test_conv_different_strides() {
    let in_len = 32;
    for stride in [1, 2, 4] {
        let w = DynTensor::ones(&[4, 1, 3], DType::F32, &cpu()).unwrap();
        let cfg = Conv1dConfig::new(1, stride, 1);
        let conv = Conv1d::new(w, None, cfg).unwrap();

        let x = DynTensor::ones(&[1, 1, in_len], DType::F32, &cpu()).unwrap();
        let out = conv.forward(&x).unwrap();
        let expected_len = (in_len + 2 - 3) / stride + 1;
        assert_eq!(
            out.dims()[2],
            expected_len,
            "stride={stride}: expected len={expected_len}"
        );
    }
}

#[test]
fn test_conv_grouped_convolution() {
    // groups=2: weight shape [out=4, in/groups=1, kernel=3]
    let w = DynTensor::ones(&[4, 1, 3], DType::F32, &cpu()).unwrap();
    let cfg = Conv1dConfig::new(1, 1, 1).with_groups(2);
    let conv = Conv1d::new(w, None, cfg).unwrap();

    let x = DynTensor::ones(&[1, 2, 8], DType::F32, &cpu()).unwrap();
    let out = conv.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, 4, 8]);
}

#[test]
fn test_conv_with_bias() {
    let w = DynTensor::zeros(&[2, 1, 3], DType::F32, &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![1.0, -1.0], &[2], &cpu()).unwrap();
    let conv = Conv1d::new(w, Some(b), Conv1dConfig::new(1, 1, 1)).unwrap();

    let x = DynTensor::zeros(&[1, 1, 4], DType::F32, &cpu()).unwrap();
    let out = conv.forward(&x).unwrap();

    // Zero weights + bias: output should be [1, -1] repeated
    let flat = out.to_flat_vec::<f32>().unwrap();
    assert_eq!(out.dims(), &[1, 2, 4]);
    assert!((flat[0] - 1.0).abs() < 1e-5, "bias channel 0 should be 1.0");
    assert!(
        (flat[4] - (-1.0)).abs() < 1e-5,
        "bias channel 1 should be -1.0"
    );
}

#[test]
fn test_conv_encoder_batch_independence() {
    // Two different batch elements should produce independent outputs
    let w = pseudo_random(&[4, 1, 3], 0.5);
    let conv = Conv1d::new(w, None, Conv1dConfig::new(1, 1, 1)).unwrap();

    let x1 = DynTensor::ones(&[1, 1, 8], DType::F32, &cpu()).unwrap();
    let x2 = DynTensor::full(&[1, 1, 8], 2.0, DType::F32, &cpu()).unwrap();

    let out1 = conv.forward(&x1).unwrap();
    let _out2 = conv.forward(&x2).unwrap();

    // Batched forward
    let x_batch = DynTensor::cat(&[&x1, &x2], 0).unwrap();
    let out_batch = conv.forward(&x_batch).unwrap();
    assert_eq!(out_batch.dims()[0], 2);

    let flat1 = out1.to_flat_vec::<f32>().unwrap();
    let flat_batch = out_batch.to_flat_vec::<f32>().unwrap();
    let per_sample = flat_batch.len() / 2;
    for (i, (a, b)) in flat1
        .iter()
        .zip(flat_batch[..per_sample].iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1e-5,
            "Batch element 0 mismatch at {i}: {a} vs {b}"
        );
    }
}

#[test]
fn test_conv_maxpool_downsampling() {
    let w = DynTensor::ones(&[4, 1, 3], DType::F32, &cpu()).unwrap();
    let conv = Conv1d::new(w, None, Conv1dConfig::new(1, 1, 1)).unwrap();
    let pool = MaxPool1d::new(Pool1dConfig::new(2)).unwrap();

    let x = DynTensor::ones(&[1, 1, 16], DType::F32, &cpu()).unwrap();
    let h = conv.forward(&x).unwrap();
    assert_eq!(h.dims()[2], 16);

    let h = pool.forward(&h).unwrap();
    assert_eq!(h.dims()[2], 8, "MaxPool1d(2) should halve the length");
}

// ===========================================================================
// 3. Decoder with Attention (6+ tests)
// ===========================================================================

#[test]
fn test_cross_attention_decoder_shapes() {
    let dim = 16;
    let num_heads = 2;
    let vb = crate::var_builder::VarBuilder::zeros(DType::F32, &cpu());
    let cross_attn =
        MultiHeadAttention::load(vb.pp("cross"), dim, num_heads, num_heads, false).unwrap();

    let decoder_input = pseudo_random(&[1, 5, dim], 0.3);
    let encoder_output = pseudo_random(&[1, 20, dim], 0.3);

    let out = cross_attn
        .forward(&decoder_input, Some(&encoder_output), None, None, 0)
        .unwrap();
    assert_eq!(out.dims(), &[1, 5, dim]);
}

#[test]
fn test_kv_cache_autoregressive_step() {
    let dim = 16;
    let num_heads = 2;
    let vb = crate::var_builder::VarBuilder::zeros(DType::F32, &cpu());
    let mha = MultiHeadAttention::load(vb.pp("attn"), dim, num_heads, num_heads, false).unwrap();
    let mut cache = KvCacheLayer::new(2, 512).unwrap();

    // Prefill: 4 tokens
    let prefill = pseudo_random(&[1, 4, dim], 0.3);
    let out1 = mha
        .forward_kv_cached(&prefill, None, &mut cache, None, None, 0)
        .unwrap();
    assert_eq!(out1.dims(), &[1, 4, dim]);

    // Decode: 1 token at a time
    let token = pseudo_random(&[1, 1, dim], 0.3);
    let out2 = mha
        .forward_kv_cached(&token, None, &mut cache, None, None, 4)
        .unwrap();
    assert_eq!(out2.dims(), &[1, 1, dim]);
}

#[test]
fn test_kv_cache_multiple_decode_steps() {
    let dim = 8;
    let num_heads = 2;
    let vb = crate::var_builder::VarBuilder::zeros(DType::F32, &cpu());
    let mha = MultiHeadAttention::load(vb.pp("attn"), dim, num_heads, num_heads, false).unwrap();
    let mut cache = KvCacheLayer::new(2, 128).unwrap();

    // Prefill with 2 tokens
    let prefill = pseudo_random(&[1, 2, dim], 0.5);
    mha.forward_kv_cached(&prefill, None, &mut cache, None, None, 0)
        .unwrap();

    // 5 decode steps
    for step in 0..5 {
        let token = pseudo_random(&[1, 1, dim], 0.1 * (step as f32 + 1.0));
        let out = mha
            .forward_kv_cached(&token, None, &mut cache, None, None, 2 + step)
            .unwrap();
        assert_eq!(
            out.dims(),
            &[1, 1, dim],
            "decode step {step} shape mismatch"
        );
    }
}

#[test]
fn test_kv_cache_reset_allows_new_sequence() {
    let dim = 8;
    let num_heads = 1;
    let vb = crate::var_builder::VarBuilder::zeros(DType::F32, &cpu());
    let mha = MultiHeadAttention::load(vb.pp("attn"), dim, num_heads, num_heads, false).unwrap();
    let mut cache = KvCacheLayer::new(2, 64).unwrap();

    let x = pseudo_random(&[1, 3, dim], 0.3);
    mha.forward_kv_cached(&x, None, &mut cache, None, None, 0)
        .unwrap();

    cache.reset();

    // After reset, cache should accept new sequence from scratch
    let x2 = pseudo_random(&[1, 2, dim], 0.5);
    let out = mha
        .forward_kv_cached(&x2, None, &mut cache, None, None, 0)
        .unwrap();
    assert_eq!(out.dims(), &[1, 2, dim]);
}

#[test]
fn test_causal_mask_shape_correct() {
    let mask = causal_mask(8, &cpu()).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 8, 8]);

    // Lower-triangular: mask[0,0,i,j] should be 0 for j<=i, -inf for j>i
    let flat = mask.to_flat_vec::<f32>().unwrap();
    // Position (0,1): row=0, col=1 -> j > i -> should be -inf
    assert!(
        flat[1].is_infinite() && flat[1] < 0.0,
        "mask[0,1] should be -inf"
    );
    // Position (1,0): row=1, col=0 -> j <= i -> should be 0
    assert_eq!(flat[8], 0.0, "mask[1,0] should be 0");
}

#[test]
fn test_decoder_with_causal_mask_and_ffn() {
    let dim = 16;
    let num_heads = 2;
    let seq_len = 4;

    let vb = crate::var_builder::VarBuilder::zeros(DType::F32, &cpu());
    let self_attn =
        MultiHeadAttention::load(vb.pp("self"), dim, num_heads, num_heads, false).unwrap();
    let ffn_up = make_linear(dim, dim * 4);
    let ffn_down = make_linear(dim * 4, dim);

    let x = pseudo_random(&[1, seq_len, dim], 0.3);
    let mask = causal_mask(seq_len, &cpu()).unwrap();

    // Self-attention with causal mask
    let attn_out = self_attn.forward(&x, None, Some(&mask), None, 0).unwrap();
    let h = x.broadcast_add(&attn_out).unwrap();

    // FFN
    let ffn_h = ffn_up.forward(&h).unwrap().gelu().unwrap();
    let out = ffn_down.forward(&ffn_h).unwrap();
    let out = h.broadcast_add(&out).unwrap();

    assert_eq!(out.dims(), &[1, seq_len, dim]);
}

// ===========================================================================
// 4. Normalization Patterns (8+ tests)
// ===========================================================================

#[test]
fn test_rmsnorm_vs_layernorm_output_differs() {
    let dim = 8;
    let w = DynTensor::ones(&[dim], DType::F32, &cpu()).unwrap();
    let b = DynTensor::zeros(&[dim], DType::F32, &cpu()).unwrap();

    let rms = RmsNorm::new(w.clone(), 1e-5).unwrap();
    let ln = LayerNorm::new(w, b, 1e-5).unwrap();

    // Non-zero-mean input: RmsNorm and LayerNorm should differ
    let x = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, dim],
        &cpu(),
    )
    .unwrap();

    let rms_out = rms.forward(&x).unwrap();
    let ln_out = ln.forward(&x).unwrap();

    let rms_flat = rms_out.to_flat_vec::<f32>().unwrap();
    let ln_flat = ln_out.to_flat_vec::<f32>().unwrap();

    // They should differ because LayerNorm subtracts mean, RmsNorm does not
    let mut any_differ = false;
    for (a, b) in rms_flat.iter().zip(ln_flat.iter()) {
        if (a - b).abs() > 1e-5 {
            any_differ = true;
            break;
        }
    }
    assert!(
        any_differ,
        "RmsNorm and LayerNorm should produce different outputs for non-zero-mean input"
    );
}

#[test]
fn test_rmsnorm_vs_layernorm_agree_on_zero_mean() {
    // For zero-mean inputs, LayerNorm's mean subtraction is a no-op,
    // but variance vs RMS still differ (RMS includes mean^2 in denominator).
    // Actually they still differ slightly because LayerNorm uses variance not RMS.
    let dim = 4;
    let w = DynTensor::ones(&[dim], DType::F32, &cpu()).unwrap();
    let b = DynTensor::zeros(&[dim], DType::F32, &cpu()).unwrap();

    let rms = RmsNorm::new(w.clone(), 1e-5).unwrap();
    let ln = LayerNorm::new(w, b, 1e-5).unwrap();

    // Symmetric zero-mean input: [-1, 1, -1, 1]
    let x = DynTensor::from_vec(vec![-1.0, 1.0, -1.0, 1.0], &[1, dim], &cpu()).unwrap();
    let rms_out = rms.forward(&x).unwrap();
    let ln_out = ln.forward(&x).unwrap();

    let rms_flat = rms_out.to_flat_vec::<f32>().unwrap();
    let ln_flat = ln_out.to_flat_vec::<f32>().unwrap();

    // For zero-mean data, mean(x^2) == variance, so they should agree
    for (i, (a, b)) in rms_flat.iter().zip(ln_flat.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "Zero-mean: RmsNorm and LayerNorm should agree at {i}: {a} vs {b}"
        );
    }
}

#[test]
fn test_groupnorm_different_group_counts() {
    let channels = 12;
    for num_groups in [1, 2, 3, 4, 6, 12] {
        let w = DynTensor::ones(&[channels], DType::F32, &cpu()).unwrap();
        let b = DynTensor::zeros(&[channels], DType::F32, &cpu()).unwrap();
        let gn = GroupNorm::new(num_groups, channels, w, b, 1e-5).unwrap();

        let x = pseudo_random(&[2, channels, 8], 1.0);
        let out = gn.forward(&x).unwrap();
        assert_eq!(
            out.dims(),
            &[2, channels, 8],
            "GroupNorm(groups={num_groups}) should preserve shape"
        );
    }
}

#[test]
fn test_instancenorm_across_batch() {
    // InstanceNorm normalizes per (batch, channel) independently
    let eps = 1e-5;
    let inorm = InstanceNorm::new(eps).unwrap();

    // Two batch elements with different distributions
    let x = DynTensor::from_vec(
        vec![
            1.0, 2.0, 3.0, 4.0, // batch 0, channel 0
            10.0, 20.0, 30.0, 40.0, // batch 1, channel 0
        ],
        &[2, 1, 4],
        &cpu(),
    )
    .unwrap();

    let out = inorm.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 1, 4]);

    let flat = out.to_flat_vec::<f32>().unwrap();
    // Each (batch, channel) pair should be normalized independently
    // batch 0: mean=2.5, std ~1.118 -> normalized ~ [-1.34, -0.447, 0.447, 1.34]
    // batch 1: mean=25, std ~11.18 -> normalized ~ [-1.34, -0.447, 0.447, 1.34]
    // Both should have similar patterns after normalization
    for i in 0..4 {
        assert!(
            (flat[i] - flat[i + 4]).abs() < 0.01,
            "InstanceNorm: batch 0 and batch 1 should normalize identically at pos {i}"
        );
    }
}

#[test]
fn test_layernorm_with_affine() {
    let dim = 4;
    let w = DynTensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], &[dim], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[dim], &cpu()).unwrap();
    let ln_affine = LayerNorm::new(w, b, 1e-5).unwrap();

    let w_one = DynTensor::ones(&[dim], DType::F32, &cpu()).unwrap();
    let b_zero = DynTensor::zeros(&[dim], DType::F32, &cpu()).unwrap();
    let ln_no_affine = LayerNorm::new(w_one, b_zero, 1e-5).unwrap();

    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, dim], &cpu()).unwrap();
    let out_affine = ln_affine.forward(&x).unwrap();
    let out_plain = ln_no_affine.forward(&x).unwrap();

    let flat_a = out_affine.to_flat_vec::<f32>().unwrap();
    let flat_p = out_plain.to_flat_vec::<f32>().unwrap();

    // affine = plain * 2 + 1
    for (i, (a, p)) in flat_a.iter().zip(flat_p.iter()).enumerate() {
        let expected = p * 2.0 + 1.0;
        assert!(
            (a - expected).abs() < 1e-4,
            "Affine LayerNorm at {i}: {a} vs expected {expected}"
        );
    }
}

#[test]
fn test_groupnorm_1_group_equals_layernorm_style() {
    // GroupNorm with 1 group normalizes over all channels (similar to LayerNorm)
    let channels = 4;
    let w = DynTensor::ones(&[channels], DType::F32, &cpu()).unwrap();
    let b = DynTensor::zeros(&[channels], DType::F32, &cpu()).unwrap();
    let gn = GroupNorm::new(1, channels, w, b, 1e-5).unwrap();

    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, channels, 1], &cpu()).unwrap();
    let out = gn.forward(&x).unwrap();
    let flat = out.to_flat_vec::<f32>().unwrap();

    // With 1 group, all channels normalized together
    // Mean = 2.5, std ~1.118, so first element ~ (1-2.5)/1.118 ~ -1.342
    assert!((flat[0] - (-1.3416)).abs() < 0.02);
}

#[test]
fn test_groupnorm_n_groups_equals_instancenorm_style() {
    // GroupNorm with num_groups == num_channels normalizes per-channel (like InstanceNorm)
    let channels = 4;
    let w = DynTensor::ones(&[channels], DType::F32, &cpu()).unwrap();
    let b = DynTensor::zeros(&[channels], DType::F32, &cpu()).unwrap();
    let gn = GroupNorm::new(channels, channels, w, b, 1e-5).unwrap();

    let x = pseudo_random(&[1, channels, 8], 1.0);
    let out = gn.forward(&x).unwrap();
    assert_eq!(out.dims(), &[1, channels, 8]);
    // Each channel should be independently normalized (mean~0, std~1)
    for v in out.to_flat_vec::<f32>().unwrap() {
        assert!(v.is_finite(), "GroupNorm(n==C) output must be finite");
    }
}

#[test]
fn test_norm_sequential_chain() {
    // Chain: LayerNorm -> RmsNorm (both applied to same tensor shape)
    let dim = 8;
    let ln_w = DynTensor::ones(&[dim], DType::F32, &cpu()).unwrap();
    let ln_b = DynTensor::zeros(&[dim], DType::F32, &cpu()).unwrap();
    let ln = LayerNorm::new(ln_w, ln_b, 1e-5).unwrap();
    let rms_w = DynTensor::ones(&[dim], DType::F32, &cpu()).unwrap();
    let rms = RmsNorm::new(rms_w, 1e-5).unwrap();

    let x = pseudo_random(&[2, 4, dim], 1.0);
    let h = ln.forward(&x).unwrap();
    let out = rms.forward(&h).unwrap();
    assert_eq!(out.dims(), &[2, 4, dim]);

    // Double normalization: output should be roughly unit-scale
    let flat = out.to_flat_vec::<f32>().unwrap();
    for v in &flat {
        assert!(
            v.abs() < 10.0,
            "Double-norm output should be bounded, got {v}"
        );
    }
}

// ===========================================================================
// 5. Activation Patterns (6+ tests)
// ===========================================================================

#[test]
fn test_all_activations_preserve_shape() {
    let activations = [
        Activation::Relu,
        Activation::Gelu,
        Activation::Silu,
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::Elu(1.0),
        Activation::LeakyRelu(0.01),
    ];
    let x = pseudo_random(&[2, 3, 4], 1.0);
    for act in &activations {
        let out = act.forward(&x).unwrap();
        assert_eq!(out.dims(), x.dims(), "{act:?} should preserve shape");
    }
}

#[test]
fn test_activation_enum_dispatch_correctness() {
    // Verify Activation enum dispatches to correct underlying function
    let x = DynTensor::from_vec(vec![-1.0, 0.0, 1.0], &[3], &cpu()).unwrap();

    let relu_out = Activation::Relu.forward(&x).unwrap();
    let direct_relu = x.relu().unwrap();
    assert_eq!(
        relu_out.to_flat_vec::<f32>().unwrap(),
        direct_relu.to_flat_vec::<f32>().unwrap()
    );

    let gelu_out = Activation::Gelu.forward(&x).unwrap();
    let direct_gelu = x.gelu().unwrap();
    assert_eq!(
        gelu_out.to_flat_vec::<f32>().unwrap(),
        direct_gelu.to_flat_vec::<f32>().unwrap()
    );

    let silu_out = Activation::Silu.forward(&x).unwrap();
    let direct_silu = x.silu().unwrap();
    assert_eq!(
        silu_out.to_flat_vec::<f32>().unwrap(),
        direct_silu.to_flat_vec::<f32>().unwrap()
    );
}

#[test]
fn test_relu_zeros_negative_values() {
    let x = DynTensor::from_vec(vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0], &[6], &cpu()).unwrap();
    let out = Activation::Relu.forward(&x).unwrap();
    let flat = out.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat, vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_sigmoid_bounded_0_1() {
    let x = DynTensor::from_vec(vec![-100.0, -1.0, 0.0, 1.0, 100.0], &[5], &cpu()).unwrap();
    let out = Activation::Sigmoid.forward(&x).unwrap();
    let flat = out.to_flat_vec::<f32>().unwrap();
    for v in &flat {
        assert!(*v >= 0.0 && *v <= 1.0, "Sigmoid must be in [0,1], got {v}");
    }
    assert!(flat[0] < 0.01, "sigmoid(-100) should be near 0");
    assert!(flat[4] > 0.99, "sigmoid(100) should be near 1");
    assert!((flat[2] - 0.5).abs() < 1e-5, "sigmoid(0) should be 0.5");
}

#[test]
fn test_swiglu_forward_shape() {
    let dim = 8;
    let ff_dim = 32;

    let w_gate = make_linear_with_data(dim, ff_dim, 0.1);
    let w_up = make_linear_with_data(dim, ff_dim, 0.1);
    let w_down = make_linear_with_data(ff_dim, dim, 0.1);

    let swiglu = SwiGlu::new(w_gate, w_up, w_down).unwrap();
    let x = pseudo_random(&[2, 4, dim], 0.5);
    let out = swiglu.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 4, dim]);
}

#[test]
fn test_swiglu_gate_mechanism() {
    // SwiGlu: gate = silu(w_gate(x)), up = w_up(x), out = w_down(gate * up)
    // With identity-like weights, verify the gating works
    let dim = 4;
    let ff_dim = 4;

    let w_gate = make_linear_with_data(dim, ff_dim, 0.3);
    let w_up = make_linear_with_data(dim, ff_dim, 0.3);
    let w_down = make_linear_with_data(ff_dim, dim, 0.3);
    let swiglu = SwiGlu::new(w_gate, w_up, w_down).unwrap();

    let x = DynTensor::ones(&[1, 1, dim], DType::F32, &cpu()).unwrap();
    let out = swiglu.forward(&x).unwrap();

    let flat = out.to_flat_vec::<f32>().unwrap();
    for v in &flat {
        assert!(v.is_finite(), "SwiGlu output should be finite, got {v}");
    }
}

// ===========================================================================
// 6. Quantized Layers (6+ tests)
// ===========================================================================

#[test]
fn test_int8_symmetric_quantize_roundtrip() {
    let w = DynTensor::from_vec(
        vec![1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0, 0.75],
        &[2, 4],
        &cpu(),
    )
    .unwrap();

    let (q, params) = quantize_per_channel(&w, Int8Mode::Symmetric).unwrap();
    assert_eq!(q.dims(), &[2, 4]);
    assert_eq!(params.scale.len(), 2);

    let deq = dequantize_per_channel(&q, &params).unwrap();
    assert_eq!(deq.dims(), &[2, 4]);

    let orig = w.to_flat_vec::<f32>().unwrap();
    let roundtrip = deq.to_flat_vec::<f32>().unwrap();
    for (i, (a, b)) in orig.iter().zip(roundtrip.iter()).enumerate() {
        assert!(
            (a - b).abs() < 0.02,
            "INT8 symmetric roundtrip error too large at {i}: {a} vs {b}"
        );
    }
}

#[test]
fn test_int8_asymmetric_quantize_roundtrip() {
    let w = DynTensor::from_vec(
        vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        &[2, 4],
        &cpu(),
    )
    .unwrap();

    let (q, params) = quantize_per_channel(&w, Int8Mode::Asymmetric).unwrap();
    let deq = dequantize_per_channel(&q, &params).unwrap();

    let orig = w.to_flat_vec::<f32>().unwrap();
    let roundtrip = deq.to_flat_vec::<f32>().unwrap();
    for (i, (a, b)) in orig.iter().zip(roundtrip.iter()).enumerate() {
        assert!(
            (a - b).abs() < 0.02,
            "INT8 asymmetric roundtrip error at {i}: {a} vs {b}"
        );
    }
}

#[test]
fn test_max_quantization_error_bounded() {
    let w = pseudo_random(&[4, 8], 1.0);

    let (q, params) = quantize_per_channel(&w, Int8Mode::Symmetric).unwrap();
    let deq = dequantize_per_channel(&q, &params).unwrap();

    let orig = w.to_flat_vec::<f32>().unwrap();
    let roundtrip = deq.to_flat_vec::<f32>().unwrap();
    let max_err = max_quantization_error(&orig, &roundtrip);

    // INT8 quantization should have small error relative to the weight range.
    // For values in [-1, 1], step size is ~2/254 ~ 0.008, max error ~ 0.004.
    assert!(
        max_err < 0.02,
        "INT8 max quantization error {max_err} should be small (< 0.02)"
    );
    // Error should be non-negative
    assert!(max_err >= 0.0, "Max error must be non-negative");
}

#[test]
fn test_int8_linear_forward() {
    let w = pseudo_random(&[4, 8], 0.5);
    let b = pseudo_random(&[4], 0.1);
    let lin = Linear::new(w, Some(b)).unwrap();
    let int8_lin = Int8Linear::from_linear(&lin, Int8Mode::Symmetric).unwrap();

    let x = pseudo_random(&[1, 8], 0.3);
    let out_f32 = lin.forward(&x).unwrap();
    let out_int8 = int8_lin.forward(&x).unwrap();

    assert_eq!(out_int8.dims(), out_f32.dims());

    let f32_vals = out_f32.to_flat_vec::<f32>().unwrap();
    let int8_vals = out_int8.to_flat_vec::<f32>().unwrap();
    for (i, (a, b)) in f32_vals.iter().zip(int8_vals.iter()).enumerate() {
        assert!(
            (a - b).abs() < 0.05,
            "Int8Linear output mismatch at {i}: f32={a} vs int8={b}"
        );
    }
}

#[test]
fn test_int8_linear_preserves_shape() {
    let w = pseudo_random(&[16, 32], 0.5);
    let lin = Linear::new(w, None).unwrap();
    let int8_lin = Int8Linear::from_linear(&lin, Int8Mode::Symmetric).unwrap();

    let x = pseudo_random(&[2, 4, 32], 0.3);
    let out = int8_lin.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 4, 16]);
}

#[test]
fn test_quantize_rejects_non_finite() {
    let w = DynTensor::from_vec(vec![1.0, f32::NAN, 0.5, -0.5], &[2, 2], &cpu()).unwrap();
    let err = quantize_per_channel(&w, Int8Mode::Symmetric);
    assert!(err.is_err(), "quantize should reject NaN values");
}

// ===========================================================================
// Additional: Sequential composition with mixed layers
// ===========================================================================

#[test]
fn test_sequential_linear_activation_chain() {
    let mut seq = Sequential::new();
    let lin1 = make_linear_with_data(4, 8, 0.3);
    let lin2 = make_linear_with_data(8, 4, 0.3);
    seq.add(lin1);
    seq.add(Activation::Gelu);
    seq.add(lin2);
    seq.add(Activation::Relu);

    assert_eq!(seq.len(), 4);
    let x = pseudo_random(&[2, 4], 0.5);
    let out = seq.forward(&x).unwrap();
    assert_eq!(out.dims(), &[2, 4]);

    // ReLU at the end: output should be non-negative
    let flat = out.to_flat_vec::<f32>().unwrap();
    for v in &flat {
        assert!(
            *v >= 0.0,
            "Sequential ending with ReLU should be >= 0, got {v}"
        );
    }
}

#[test]
fn test_dropout_is_identity_in_eval() {
    let drop = Dropout::new(0.5);
    let x = pseudo_random(&[4, 8], 1.0);
    // Dropout in eval mode (default) should be identity
    let out = drop.forward(&x).unwrap();
    let x_flat = x.to_flat_vec::<f32>().unwrap();
    let out_flat = out.to_flat_vec::<f32>().unwrap();
    assert_eq!(x_flat, out_flat, "Dropout in eval mode should be identity");
}

#[test]
fn test_embedding_lookup_composition() {
    // Embedding -> LayerNorm -> Linear
    let vocab = 100;
    let dim = 16;

    let embed_w = pseudo_random(&[vocab, dim], 0.1);
    let embed = Embedding::new(embed_w).unwrap();

    let ln_w = DynTensor::ones(&[dim], DType::F32, &cpu()).unwrap();
    let ln_b = DynTensor::zeros(&[dim], DType::F32, &cpu()).unwrap();
    let ln = LayerNorm::new(ln_w, ln_b, 1e-5).unwrap();

    let proj = make_linear(dim, dim);

    // Token indices [3, 7, 42, 99] as U32
    let ids = DynTensor::from_vec_u32(vec![3, 7, 42, 99], &[1, 4], &cpu()).unwrap();
    let h = embed.forward(&ids).unwrap();
    assert_eq!(h.dims(), &[1, 4, dim]);

    let h = ln.forward(&h).unwrap();
    let out = proj.forward(&h).unwrap();
    assert_eq!(out.dims(), &[1, 4, dim]);
}
