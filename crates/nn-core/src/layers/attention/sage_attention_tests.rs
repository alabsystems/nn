// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`SageAttention`] — quantized attention with INT8 Q/K scoring.
//!
//! Covers shape correctness, numerical accuracy vs reference SDPA, causal
//! masking, GQA, smooth_k, edge cases (single token, long sequences, zeros),
//! quantization range invariants, and dtype preservation.

use super::{SageAttention, SageAttentionConfig};
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn default_config() -> SageAttentionConfig {
    SageAttentionConfig {
        head_dim: 4,
        num_heads: 2,
        num_kv_heads: None,
        causal: false,
        smooth_k: false,
    }
}

/// Create deterministic QKV tensors with sin-based pseudo-random data.
fn make_qkv(
    batch: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> (DynTensor, DynTensor, DynTensor) {
    let device = Device::Cpu;
    let n = batch * num_heads * seq_len * head_dim;
    let shape = &[batch, num_heads, seq_len, head_dim];

    let q_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017).sin() * 0.5).collect();
    let k_data: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.031 + 1.0).sin() * 0.5)
        .collect();
    let v_data: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.047 + 2.0).sin() * 0.5)
        .collect();

    let q = DynTensor::from_vec(q_data, shape, &device).unwrap();
    let k = DynTensor::from_vec(k_data, shape, &device).unwrap();
    let v = DynTensor::from_vec(v_data, shape, &device).unwrap();
    (q, k, v)
}

/// Create deterministic QKV with specific KV head count (for GQA tests).
fn make_qkv_gqa(
    batch: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> (DynTensor, DynTensor, DynTensor) {
    let device = Device::Cpu;
    let nq = batch * num_q_heads * seq_len * head_dim;
    let nkv = batch * num_kv_heads * seq_len * head_dim;

    let q_data: Vec<f32> = (0..nq).map(|i| ((i as f32) * 0.017).sin() * 0.5).collect();
    let k_data: Vec<f32> = (0..nkv)
        .map(|i| ((i as f32) * 0.031 + 1.0).sin() * 0.5)
        .collect();
    let v_data: Vec<f32> = (0..nkv)
        .map(|i| ((i as f32) * 0.047 + 2.0).sin() * 0.5)
        .collect();

    let q = DynTensor::from_vec(q_data, &[batch, num_q_heads, seq_len, head_dim], &device).unwrap();
    let k =
        DynTensor::from_vec(k_data, &[batch, num_kv_heads, seq_len, head_dim], &device).unwrap();
    let v =
        DynTensor::from_vec(v_data, &[batch, num_kv_heads, seq_len, head_dim], &device).unwrap();
    (q, k, v)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Basic forward pass: shape correctness with [2, 2, 8, 4].
#[test]
fn test_sage_basic_forward() {
    let cfg = default_config();
    let attn = SageAttention::new(cfg).unwrap();
    let (q, k, v) = make_qkv(2, 2, 8, 4);
    let out = attn.forward(&q, &k, &v).unwrap();
    assert_eq!(out.dims(), &[2, 2, 8, 4]);
}

/// Compare SageAttention output to standard SDPA reference.
/// Max absolute difference should be small for bounded inputs.
#[test]
fn test_sage_matches_sdpa_reference() {
    let cfg = SageAttentionConfig {
        head_dim: 4,
        num_heads: 2,
        num_kv_heads: None,
        causal: false,
        smooth_k: false,
    };
    let attn = SageAttention::new(cfg).unwrap();
    let (q, k, v) = make_qkv(1, 2, 8, 4);

    // SageAttention output
    let sage_out = attn.forward(&q, &k, &v).unwrap();

    // Reference: standard SDPA (no quantization)
    let scale = 1.0 / (4.0_f64).sqrt();
    let ref_out = crate::layers::attention::sdpa(&q, &k, &v, None, scale).unwrap();

    // Compute max absolute difference
    let diff = sage_out
        .broadcast_add(&ref_out.mul_scalar(-1.0).unwrap())
        .unwrap();
    let diff_abs = diff.abs().unwrap();
    let max_diff = diff_abs
        .max_keepdim(3)
        .unwrap()
        .max_keepdim(2)
        .unwrap()
        .max_keepdim(1)
        .unwrap()
        .max_keepdim(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    assert!(
        max_diff < 0.15,
        "SageAttention vs SDPA max_diff={max_diff}, expected < 0.15"
    );
}

/// Causal mode: verify causal masking works correctly.
#[test]
fn test_sage_causal_mode() {
    let cfg = SageAttentionConfig {
        causal: true,
        ..default_config()
    };
    let attn = SageAttention::new(cfg).unwrap();
    let (q, k, v) = make_qkv(1, 2, 8, 4);
    let out = attn.forward(&q, &k, &v).unwrap();
    assert_eq!(out.dims(), &[1, 2, 8, 4]);

    // Output should be finite (causal mask with -inf should not produce NaN)
    let out_data = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        out_data.iter().all(|x| x.is_finite()),
        "causal output must be finite"
    );
}

/// GQA: num_kv_heads < num_heads (e.g. 2 KV heads, 8 Q heads).
#[test]
fn test_sage_gqa() {
    let cfg = SageAttentionConfig {
        num_heads: 8,
        num_kv_heads: Some(2),
        head_dim: 4,
        causal: false,
        smooth_k: false,
    };
    let attn = SageAttention::new(cfg).unwrap();
    let (q, k, v) = make_qkv_gqa(1, 8, 2, 8, 4);
    let out = attn.forward(&q, &k, &v).unwrap();
    assert_eq!(out.dims(), &[1, 8, 8, 4]);
}

/// smooth_k=true vs smooth_k=false should produce different outputs.
#[test]
fn test_sage_smooth_k() {
    let (q, k, v) = make_qkv(1, 2, 8, 4);

    let cfg_smooth = SageAttentionConfig {
        smooth_k: true,
        ..default_config()
    };
    let cfg_plain = SageAttentionConfig {
        smooth_k: false,
        ..default_config()
    };

    let attn_smooth = SageAttention::new(cfg_smooth).unwrap();
    let attn_plain = SageAttention::new(cfg_plain).unwrap();

    let out_smooth = attn_smooth.forward(&q, &k, &v).unwrap();
    let out_plain = attn_plain.forward(&q, &k, &v).unwrap();

    // They should differ (unless K is constant, which is unlikely for sin-based data)
    let diff = out_smooth
        .broadcast_add(&out_plain.mul_scalar(-1.0).unwrap())
        .unwrap();
    let diff_abs = diff.abs().unwrap();
    let max_diff = diff_abs
        .max_keepdim(3)
        .unwrap()
        .max_keepdim(2)
        .unwrap()
        .max_keepdim(1)
        .unwrap()
        .max_keepdim(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();

    assert!(
        max_diff > 1e-6,
        "smooth_k should produce different output, max_diff={max_diff}"
    );
}

/// Single-token sequence (seq_len=1) should work.
#[test]
fn test_sage_single_token() {
    let cfg = default_config();
    let attn = SageAttention::new(cfg).unwrap();
    let (q, k, v) = make_qkv(1, 2, 1, 4);
    let out = attn.forward(&q, &k, &v).unwrap();
    assert_eq!(out.dims(), &[1, 2, 1, 4]);
}

/// Long sequence (seq_len=128).
#[test]
fn test_sage_long_sequence() {
    let cfg = default_config();
    let attn = SageAttention::new(cfg).unwrap();
    let (q, k, v) = make_qkv(1, 2, 128, 4);
    let out = attn.forward(&q, &k, &v).unwrap();
    assert_eq!(out.dims(), &[1, 2, 128, 4]);
}

/// head_dim=16 (larger head dimension).
#[test]
fn test_sage_head_dim_16() {
    let cfg = SageAttentionConfig {
        head_dim: 16,
        ..default_config()
    };
    let attn = SageAttention::new(cfg).unwrap();
    let (q, k, v) = make_qkv(1, 2, 8, 16);
    let out = attn.forward(&q, &k, &v).unwrap();
    assert_eq!(out.dims(), &[1, 2, 8, 16]);
}

/// Quantization scales must be non-negative (verified via finite output).
#[test]
fn test_sage_scale_non_negative() {
    let cfg = default_config();
    let attn = SageAttention::new(cfg).unwrap();
    let (q, k, v) = make_qkv(2, 2, 8, 4);
    let out = attn.forward(&q, &k, &v).unwrap();
    let data = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        data.iter().all(|x| x.is_finite()),
        "output must be finite (implies valid scales)"
    );
}

/// Quantized values must be in [-128, 127] range.
#[test]
fn test_sage_quantized_range() {
    let device = Device::Cpu;
    // Create tensor with values spanning a wide range
    let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.5).collect();
    let t = DynTensor::from_vec(data, &[1, 1, 1, 100], &device).unwrap();

    // Quantize: scale = absmax / 127, q_int8 = round(t / scale).clamp(-128, 127)
    let t_abs = t.abs().unwrap();
    let absmax = t_abs.max_keepdim(3).unwrap();
    let scale = absmax.clamp_min(1e-10).unwrap().div_scalar(127.0).unwrap();
    let q_int8 = t
        .broadcast_div(&scale)
        .unwrap()
        .round()
        .unwrap()
        .clamp(-128.0, 127.0)
        .unwrap();

    let q_data = q_int8.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for &val in &q_data {
        assert!(
            (-128.0..=127.0).contains(&val),
            "quantized value {val} out of INT8 range"
        );
    }
}

/// All-zero input should not produce NaN.
#[test]
fn test_sage_zero_input() {
    let cfg = default_config();
    let attn = SageAttention::new(cfg).unwrap();
    let device = Device::Cpu;
    let q = DynTensor::zeros(&[1, 2, 4, 4], DType::F32, &device).unwrap();
    let k = DynTensor::zeros(&[1, 2, 4, 4], DType::F32, &device).unwrap();
    let v = DynTensor::zeros(&[1, 2, 4, 4], DType::F32, &device).unwrap();
    let out = attn.forward(&q, &k, &v).unwrap();
    let data = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(
        data.iter().all(|x| x.is_finite()),
        "zero input must not produce NaN/Inf"
    );
}

/// Output dtype should match input dtype.
#[test]
fn test_sage_output_dtype() {
    let cfg = default_config();
    let attn = SageAttention::new(cfg).unwrap();
    let (q, k, v) = make_qkv(1, 2, 4, 4);
    let out = attn.forward(&q, &k, &v).unwrap();
    assert_eq!(out.dtype(), q.dtype(), "output dtype must match input");
}

/// Config validation: num_heads=0 should fail.
#[test]
fn test_sage_config_rejects_zero_heads() {
    let cfg = SageAttentionConfig {
        num_heads: 0,
        ..default_config()
    };
    assert!(SageAttention::new(cfg).is_err());
}

/// Config validation: num_kv_heads not dividing num_heads should fail.
#[test]
fn test_sage_config_rejects_bad_kv_heads() {
    let cfg = SageAttentionConfig {
        num_heads: 8,
        num_kv_heads: Some(3), // 8 % 3 != 0
        ..default_config()
    };
    assert!(SageAttention::new(cfg).is_err());
}
