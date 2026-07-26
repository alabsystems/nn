// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`SlidingWindowAttention`] and [`sliding_window_mask`].

use super::{sliding_window_mask, SlidingWindowAttention};
use crate::dyn_tensor::DynTensor;
use crate::layers::Linear;
use crate::{DType, Device};

// -- Helper -------------------------------------------------------------------

fn make_linear(out_features: usize, in_features: usize, seed: f32) -> Linear {
    let n = out_features * in_features;
    let w_data: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + seed) * 0.001).sin() * 0.1)
        .collect();
    let b_data: Vec<f32> = (0..out_features)
        .map(|i| ((i as f32 + seed * 2.0) * 0.003).sin() * 0.01)
        .collect();
    let w = DynTensor::from_vec(w_data, &[out_features, in_features], &Device::Cpu).unwrap();
    let b = DynTensor::from_vec(b_data, &[out_features], &Device::Cpu).unwrap();
    Linear::new(w, Some(b)).unwrap()
}

fn make_sliding_window_attn(
    embed_dim: usize,
    num_heads: usize,
    window_size: usize,
) -> SlidingWindowAttention {
    let qkv = make_linear(3 * embed_dim, embed_dim, 1.0);
    let out_proj = make_linear(embed_dim, embed_dim, 2.0);
    SlidingWindowAttention::new(qkv, out_proj, num_heads, window_size).unwrap()
}

// -- sliding_window_mask tests ------------------------------------------------

#[test]
fn test_mask_shape() {
    let mask = sliding_window_mask(8, 3, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 8, 8]);
}

#[test]
fn test_mask_window3_seq4() {
    // window_size=3, half=1: each position sees itself and 1 neighbor each side.
    // Expected visible pattern for seq_len=4:
    //   [0,0] [0,1]  -inf  -inf
    //   [1,0] [1,1] [1,2]  -inf
    //    -inf [2,1] [2,2] [2,3]
    //    -inf  -inf [3,2] [3,3]
    let mask = sliding_window_mask(4, 3, &Device::Cpu).unwrap();
    let data = mask.to_flat_vec::<f32>().unwrap();
    // 4x4 = 16 elements, row-major within [1,1,4,4]
    let neginf = f32::NEG_INFINITY;

    // row 0: visible at j=0,1; masked at j=2,3
    assert_eq!(data[0], 0.0); // [0,0]
    assert_eq!(data[1], 0.0); // [0,1]
    assert_eq!(data[2], neginf); // [0,2]
    assert_eq!(data[3], neginf); // [0,3]

    // row 1: visible at j=0,1,2; masked at j=3
    assert_eq!(data[4], 0.0); // [1,0]
    assert_eq!(data[5], 0.0); // [1,1]
    assert_eq!(data[6], 0.0); // [1,2]
    assert_eq!(data[7], neginf); // [1,3]

    // row 2: masked at j=0; visible at j=1,2,3
    assert_eq!(data[8], neginf); // [2,0]
    assert_eq!(data[9], 0.0); // [2,1]
    assert_eq!(data[10], 0.0); // [2,2]
    assert_eq!(data[11], 0.0); // [2,3]

    // row 3: masked at j=0,1; visible at j=2,3
    assert_eq!(data[12], neginf); // [3,0]
    assert_eq!(data[13], neginf); // [3,1]
    assert_eq!(data[14], 0.0); // [3,2]
    assert_eq!(data[15], 0.0); // [3,3]
}

#[test]
fn test_mask_window1_self_only() {
    // window_size=1: half=0, each token only attends to itself.
    let mask = sliding_window_mask(4, 1, &Device::Cpu).unwrap();
    let data = mask.to_flat_vec::<f32>().unwrap();
    let neginf = f32::NEG_INFINITY;

    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 0.0 } else { neginf };
            assert_eq!(
                data[i * 4 + j],
                expected,
                "mask[{i},{j}]: expected {expected}, got {}",
                data[i * 4 + j]
            );
        }
    }
}

#[test]
fn test_mask_large_window_is_full_attention() {
    // window_size >= 2*seq_len means everything is visible (full attention).
    let seq_len = 6;
    let window_size = 2 * seq_len;
    let mask = sliding_window_mask(seq_len, window_size, &Device::Cpu).unwrap();
    let data = mask.to_flat_vec::<f32>().unwrap();

    // All entries should be 0.0 (no masking).
    for (idx, val) in data.iter().enumerate() {
        assert_eq!(
            *val, 0.0,
            "mask element {idx} should be 0.0 for full attention, got {val}"
        );
    }
}

#[test]
fn test_mask_seq_len_zero_error() {
    let err = sliding_window_mask(0, 3, &Device::Cpu);
    assert!(err.is_err());
    let msg = format!("{:?}", err.unwrap_err());
    assert!(msg.contains("seq_len"), "msg: {msg}");
}

#[test]
fn test_mask_window_size_zero_error() {
    let err = sliding_window_mask(4, 0, &Device::Cpu);
    assert!(err.is_err());
    let msg = format!("{:?}", err.unwrap_err());
    assert!(msg.contains("window_size"), "msg: {msg}");
}

#[test]
fn test_mask_symmetry() {
    // The sliding window mask is symmetric: mask[i,j] == mask[j,i].
    let mask = sliding_window_mask(8, 5, &Device::Cpu).unwrap();
    let data = mask.to_flat_vec::<f32>().unwrap();
    for i in 0..8 {
        for j in 0..8 {
            let ij = data[i * 8 + j];
            let ji = data[j * 8 + i];
            // Both are either 0.0 or -inf; comparing bitwise is valid.
            assert_eq!(
                ij.to_bits(),
                ji.to_bits(),
                "mask[{i},{j}]={ij} != mask[{j},{i}]={ji}"
            );
        }
    }
}

// -- SlidingWindowAttention construction tests --------------------------------

#[test]
fn test_new_valid() {
    let attn = make_sliding_window_attn(16, 2, 3);
    assert_eq!(attn.num_heads(), 2);
    assert_eq!(attn.head_dim(), 8);
    assert_eq!(attn.window_size(), 3);
}

#[test]
fn test_new_zero_heads_error() {
    let qkv = make_linear(48, 16, 1.0);
    let out = make_linear(16, 16, 2.0);
    let err = SlidingWindowAttention::new(qkv, out, 0, 3);
    assert!(err.is_err());
}

#[test]
fn test_new_zero_window_error() {
    let qkv = make_linear(48, 16, 1.0);
    let out = make_linear(16, 16, 2.0);
    let err = SlidingWindowAttention::new(qkv, out, 2, 0);
    assert!(err.is_err());
    let msg = format!("{:?}", err.unwrap_err());
    assert!(msg.contains("window_size"), "msg: {msg}");
}

#[test]
fn test_new_bad_qkv_shape_error() {
    // QKV out_features not divisible by 3
    let qkv = make_linear(17, 16, 1.0);
    let out = make_linear(16, 16, 2.0);
    let err = SlidingWindowAttention::new(qkv, out, 2, 3);
    assert!(err.is_err());
    let msg = format!("{:?}", err.unwrap_err());
    assert!(msg.contains("divisible by 3"), "msg: {msg}");
}

#[test]
fn test_new_embed_dim_not_divisible_by_heads() {
    // embed_dim=15 (QKV out=45), num_heads=4: 15 % 4 != 0
    let qkv = make_linear(45, 15, 1.0);
    let out = make_linear(15, 15, 2.0);
    let err = SlidingWindowAttention::new(qkv, out, 4, 3);
    assert!(err.is_err());
    let msg = format!("{:?}", err.unwrap_err());
    assert!(msg.contains("divisible by num_heads"), "msg: {msg}");
}

#[test]
fn test_debug_format() {
    let attn = make_sliding_window_attn(16, 2, 5);
    let dbg = format!("{attn:?}");
    assert!(dbg.contains("SlidingWindowAttention"));
    assert!(dbg.contains("window_size"));
}

// -- Forward pass tests -------------------------------------------------------

#[test]
fn test_forward_output_shape() {
    let embed_dim = 16;
    let num_heads = 2;
    let window_size = 3;
    let (batch, seq_len) = (1, 8);

    let attn = make_sliding_window_attn(embed_dim, num_heads, window_size);
    let x = DynTensor::ones(&[batch, seq_len, embed_dim], DType::F32, &Device::Cpu).unwrap();
    let out = attn.forward_t(&x).unwrap();

    assert_eq!(out.dims(), &[batch, seq_len, embed_dim]);
}

#[test]
fn test_forward_batch2() {
    let embed_dim = 16;
    let num_heads = 2;
    let window_size = 3;
    let (batch, seq_len) = (2, 6);

    let attn = make_sliding_window_attn(embed_dim, num_heads, window_size);
    let x = DynTensor::ones(&[batch, seq_len, embed_dim], DType::F32, &Device::Cpu).unwrap();
    let out = attn.forward_t(&x).unwrap();

    assert_eq!(out.dims(), &[batch, seq_len, embed_dim]);
}

#[test]
fn test_forward_window_ge_seq_len_is_full_attention() {
    // When window_size >= seq_len, the mask is all-visible, equivalent to
    // standard (unmasked) attention.
    let embed_dim = 16;
    let num_heads = 2;
    let seq_len = 4;
    let window_size = 2 * seq_len; // larger than seq_len

    let attn = make_sliding_window_attn(embed_dim, num_heads, window_size);
    let data: Vec<f32> = (0..(seq_len * embed_dim))
        .map(|i| ((i as f32) * 0.01).sin() * 0.1)
        .collect();
    let x = DynTensor::from_vec(data, &[1, seq_len, embed_dim], &Device::Cpu).unwrap();
    let out = attn.forward_t(&x).unwrap();

    assert_eq!(out.dims(), &[1, seq_len, embed_dim]);
    // Output should be finite (no NaN/Inf from unmasked attention).
    let out_data = out.to_flat_vec::<f32>().unwrap();
    for (i, v) in out_data.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] is not finite: {v}");
    }
}

#[test]
fn test_forward_window1_self_attention_only() {
    // With window_size=1, each token only attends to itself.
    // Output should still be finite and have correct shape.
    let embed_dim = 16;
    let num_heads = 2;
    let window_size = 1;
    let (batch, seq_len) = (1, 6);

    let attn = make_sliding_window_attn(embed_dim, num_heads, window_size);
    let data: Vec<f32> = (0..(batch * seq_len * embed_dim))
        .map(|i| ((i as f32) * 0.007).sin() * 0.1)
        .collect();
    let x = DynTensor::from_vec(data, &[batch, seq_len, embed_dim], &Device::Cpu).unwrap();
    let out = attn.forward_t(&x).unwrap();

    assert_eq!(out.dims(), &[batch, seq_len, embed_dim]);
    let out_data = out.to_flat_vec::<f32>().unwrap();
    for (i, v) in out_data.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] is not finite: {v}");
    }
}

#[test]
fn test_module_trait() {
    use crate::layers::Module;

    let embed_dim = 16;
    let attn = make_sliding_window_attn(embed_dim, 2, 3);
    let x = DynTensor::ones(&[1, 4, embed_dim], DType::F32, &Device::Cpu).unwrap();

    // Module::forward should produce the same result as forward_t.
    let out = Module::forward(&attn, &x).unwrap();
    assert_eq!(out.dims(), &[1, 4, embed_dim]);
}

#[test]
fn test_forward_seq_len_1() {
    // Edge case: single-token sequence.
    let embed_dim = 16;
    let attn = make_sliding_window_attn(embed_dim, 2, 3);
    let x = DynTensor::ones(&[1, 1, embed_dim], DType::F32, &Device::Cpu).unwrap();
    let out = attn.forward_t(&x).unwrap();

    assert_eq!(out.dims(), &[1, 1, embed_dim]);
}

// -- Mask correctness via attention weight inspection -------------------------

#[test]
fn test_attention_weights_zero_outside_window() {
    // Manually compute attention weights and verify they are zero outside
    // the window. We apply the mask directly to inspect weights.
    let seq_len = 6;
    let head_dim = 4;
    let num_heads = 1;
    let window_size = 3; // half = 1

    // Random-ish Q, K, V: [1, 1, 6, 4]
    let q_data: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i as f32) * 0.1).sin())
        .collect();
    let k_data: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i as f32) * 0.07 + 1.0).sin())
        .collect();
    let v_data: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i as f32) * 0.03 + 2.0).sin())
        .collect();

    let q = DynTensor::from_vec(q_data, &[1, num_heads, seq_len, head_dim], &Device::Cpu).unwrap();
    let k = DynTensor::from_vec(k_data, &[1, num_heads, seq_len, head_dim], &Device::Cpu).unwrap();
    let _v = DynTensor::from_vec(v_data, &[1, num_heads, seq_len, head_dim], &Device::Cpu).unwrap();

    let mask = sliding_window_mask(seq_len, window_size, &Device::Cpu).unwrap();

    // Compute attention scores manually to inspect weights.
    let scale = 1.0 / (head_dim as f64).sqrt();
    let k_t = k.transpose(2, 3).unwrap();
    let scores = q.matmul(&k_t).unwrap().mul_scalar(scale).unwrap();
    let scores = scores.broadcast_add(&mask).unwrap();
    let attn_weights = scores.softmax(3).unwrap();

    let weights = attn_weights.to_flat_vec::<f32>().unwrap();
    let half = window_size / 2;

    for i in 0..seq_len {
        for j in 0..seq_len {
            let dist = i.abs_diff(j);
            let w = weights[i * seq_len + j];
            if dist > half {
                // Outside window: weight should be 0 (exp(-inf) = 0).
                assert!(
                    w.abs() < 1e-6,
                    "weight[{i},{j}] should be ~0 outside window, got {w}"
                );
            }
        }
    }
}
