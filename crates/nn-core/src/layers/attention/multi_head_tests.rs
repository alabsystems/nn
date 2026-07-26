#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`MultiHeadAttention`], [`causal_mask`], and [`repeat_kv`].

use super::MultiHeadAttention;
use crate::dyn_tensor::test_helpers::make_linear_seeded as make_linear;
use crate::dyn_tensor::DynTensor;
use crate::layers::attention::{causal_mask, causal_mask_dtype, causal_mask_with_offset, repeat_kv};
use crate::layers::Module;
use crate::{DType, Device};

/// Helper: create a standard MHA (no GQA) with dim=64, 4 heads.
fn make_mha_4h_64d() -> MultiHeadAttention {
    let dim = 64;
    let num_heads = 4;
    MultiHeadAttention::new(
        make_linear(dim, dim, 1.0),
        make_linear(dim, dim, 2.0),
        make_linear(dim, dim, 3.0),
        make_linear(dim, dim, 4.0),
        num_heads,
        num_heads,
    )
    .expect("valid MHA")
}

/// Helper: create a random input tensor.
fn make_input(batch: usize, seq: usize, dim: usize, seed: f32) -> DynTensor {
    let n = batch * seq * dim;
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + seed) * 0.017).sin() * 0.5)
        .collect();
    DynTensor::from_vec(data, &[batch, seq, dim], &Device::Cpu).unwrap()
}

// -- Constructor validation ---------------------------------------------------

#[test]
fn test_new_zero_heads_rejected() {
    let l = make_linear(64, 64, 0.0);
    let r = MultiHeadAttention::new(
        make_linear(64, 64, 1.0),
        make_linear(64, 64, 2.0),
        make_linear(64, 64, 3.0),
        l,
        0,
        4,
    );
    assert!(r.is_err());
    let msg = r.unwrap_err().to_string();
    assert!(msg.contains("num_heads"), "error: {msg}");
}

#[test]
fn test_new_zero_kv_heads_rejected() {
    let r = MultiHeadAttention::new(
        make_linear(64, 64, 1.0),
        make_linear(64, 64, 2.0),
        make_linear(64, 64, 3.0),
        make_linear(64, 64, 4.0),
        4,
        0,
    );
    assert!(r.is_err());
    let msg = r.unwrap_err().to_string();
    // validate_heads error says "num_heads must be > 0" — it doesn't include
    // the parameter name "num_kv_heads" because the shared validator uses a
    // generic description. Match on the actual error content.
    assert!(
        msg.contains("num_heads"),
        "error should mention num_heads: {msg}"
    );
}

#[test]
fn test_new_heads_not_divisible_rejected() {
    // 4 heads not divisible by 3 kv_heads
    let r = MultiHeadAttention::new(
        make_linear(64, 64, 1.0),
        make_linear(64, 64, 2.0),
        make_linear(64, 64, 3.0),
        make_linear(64, 64, 4.0),
        4,
        3,
    );
    assert!(r.is_err());
    let msg = r.unwrap_err().to_string();
    assert!(msg.contains("divisible"), "error: {msg}");
}

#[test]
fn test_new_valid_standard_mha() {
    let mha = make_mha_4h_64d();
    assert_eq!(mha.num_heads(), 4);
    assert_eq!(mha.num_kv_heads(), 4);
    assert_eq!(mha.head_dim(), 16); // 64 / 4
}

#[test]
fn test_new_valid_gqa() {
    // 8 query heads, 2 kv heads -> GQA with 4x repeat
    let dim = 64;
    let mha = MultiHeadAttention::new(
        make_linear(dim, dim, 1.0),
        make_linear(16, dim, 2.0), // 2 kv heads * 8 head_dim = 16
        make_linear(16, dim, 3.0),
        make_linear(dim, dim, 4.0),
        8,
        2,
    )
    .expect("valid GQA");
    assert_eq!(mha.num_heads(), 8);
    assert_eq!(mha.num_kv_heads(), 2);
    assert_eq!(mha.head_dim(), 8); // 64 / 8
}

// -- Self-attention -----------------------------------------------------------

#[test]
fn test_self_attention_output_shape() {
    let mha = make_mha_4h_64d();
    let x = make_input(2, 5, 64, 0.0);
    let out = mha.forward(&x, None, None, None, 0).unwrap();
    assert_eq!(out.dims(), &[2, 5, 64]);
}

#[test]
fn test_self_attention_single_token() {
    let mha = make_mha_4h_64d();
    let x = make_input(1, 1, 64, 0.0);
    let out = mha.forward(&x, None, None, None, 0).unwrap();
    assert_eq!(out.dims(), &[1, 1, 64]);
}

#[test]
fn test_self_attention_deterministic() {
    let mha = make_mha_4h_64d();
    let x = make_input(1, 3, 64, 0.0);
    let out1 = mha.forward(&x, None, None, None, 0).unwrap();
    let out2 = mha.forward(&x, None, None, None, 0).unwrap();
    let v1 = out1.to_flat_vec::<f32>().unwrap();
    let v2 = out2.to_flat_vec::<f32>().unwrap();
    assert_eq!(v1.len(), v2.len());
    for (a, b) in v1.iter().zip(v2.iter()) {
        assert!((a - b).abs() < 1e-6, "non-deterministic: {a} vs {b}");
    }
}

#[test]
fn test_module_trait_self_attention() {
    let mha = make_mha_4h_64d();
    let x = make_input(1, 4, 64, 0.0);
    // Module::forward should give same result as forward(x, None, None, None, 0)
    let out_module = Module::forward(&mha, &x).unwrap();
    let out_explicit = mha.forward(&x, None, None, None, 0).unwrap();
    let v1 = out_module.to_flat_vec::<f32>().unwrap();
    let v2 = out_explicit.to_flat_vec::<f32>().unwrap();
    for (a, b) in v1.iter().zip(v2.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

// -- Cross-attention ----------------------------------------------------------

#[test]
fn test_cross_attention_output_shape() {
    let mha = make_mha_4h_64d();
    let x = make_input(2, 3, 64, 0.0);
    let enc = make_input(2, 10, 64, 1.0);
    let out = mha.forward(&x, Some(&enc), None, None, 0).unwrap();
    // Output seq_len matches query, not encoder
    assert_eq!(out.dims(), &[2, 3, 64]);
}

#[test]
fn test_cross_attention_different_kv_len() {
    let mha = make_mha_4h_64d();
    let q = make_input(1, 1, 64, 0.0);
    let kv = make_input(1, 20, 64, 1.0);
    let out = mha.forward(&q, Some(&kv), None, None, 0).unwrap();
    assert_eq!(out.dims(), &[1, 1, 64]);
}

// -- GQA (grouped-query attention) -------------------------------------------

#[test]
fn test_gqa_output_shape() {
    let dim = 64;
    let mha = MultiHeadAttention::new(
        make_linear(dim, dim, 1.0),
        make_linear(16, dim, 2.0), // 2 kv heads * 8 head_dim
        make_linear(16, dim, 3.0),
        make_linear(dim, dim, 4.0),
        8,
        2,
    )
    .unwrap();
    let x = make_input(1, 5, 64, 0.0);
    let out = mha.forward(&x, None, None, None, 0).unwrap();
    assert_eq!(out.dims(), &[1, 5, 64]);
}

#[test]
fn test_mqa_single_kv_head() {
    // MQA: 4 query heads, 1 kv head
    let dim = 64;
    let head_dim = dim / 4;
    let mha = MultiHeadAttention::new(
        make_linear(dim, dim, 1.0),
        make_linear(head_dim, dim, 2.0), // 1 kv head
        make_linear(head_dim, dim, 3.0),
        make_linear(dim, dim, 4.0),
        4,
        1,
    )
    .unwrap();
    let x = make_input(2, 3, 64, 0.0);
    let out = mha.forward(&x, None, None, None, 0).unwrap();
    assert_eq!(out.dims(), &[2, 3, 64]);
}

// -- Causal mask --------------------------------------------------------------

#[test]
fn test_causal_mask_shape() {
    let mask = causal_mask(4, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 4, 4]);
}

#[test]
fn test_causal_mask_values() {
    let mask = causal_mask(3, &Device::Cpu).unwrap();
    let flat = mask.to_flat_vec::<f32>().unwrap();
    // Row 0: [0, -inf, -inf]
    assert_eq!(flat[0], 0.0);
    assert!(flat[1].is_infinite() && flat[1] < 0.0);
    assert!(flat[2].is_infinite() && flat[2] < 0.0);
    // Row 1: [0, 0, -inf]
    assert_eq!(flat[3], 0.0);
    assert_eq!(flat[4], 0.0);
    assert!(flat[5].is_infinite() && flat[5] < 0.0);
    // Row 2: [0, 0, 0]
    assert_eq!(flat[6], 0.0);
    assert_eq!(flat[7], 0.0);
    assert_eq!(flat[8], 0.0);
}

#[test]
fn test_causal_mask_seq1() {
    let mask = causal_mask(1, &Device::Cpu).unwrap();
    let flat = mask.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat, vec![0.0]);
}

#[test]
fn test_causal_mask_zero_rejected() {
    assert!(causal_mask(0, &Device::Cpu).is_err());
}

#[test]
fn test_causal_mask_overflow_rejected() {
    // seq_len where seq_len * seq_len overflows usize (sqrt(usize::MAX) + 1).
    let huge = (usize::MAX as f64).sqrt() as usize + 2;
    let result = causal_mask(huge, &Device::Cpu);
    assert!(result.is_err(), "overflow should be rejected");
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("overflow"), "error: {msg}");
}

#[test]
fn test_self_attention_with_causal_mask() {
    let mha = make_mha_4h_64d();
    let x = make_input(1, 4, 64, 0.0);
    let mask = causal_mask(4, &Device::Cpu).unwrap();
    let out = mha.forward(&x, None, Some(&mask), None, 0).unwrap();
    assert_eq!(out.dims(), &[1, 4, 64]);
    // Output should be finite
    let flat = out.to_flat_vec::<f32>().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()), "output has NaN/Inf");
}

// -- repeat_kv ----------------------------------------------------------------

#[test]
fn test_repeat_kv_noop() {
    let x = make_input(1, 4, 16, 0.0).reshape([1, 2, 2, 16]).unwrap();
    let out = repeat_kv(&x, 1).unwrap();
    assert_eq!(out.dims(), x.dims());
}

#[test]
fn test_repeat_kv_2x() {
    let x = DynTensor::from_vec(
        (0..24).map(|i| i as f32).collect(),
        &[1, 2, 3, 4],
        &Device::Cpu,
    )
    .unwrap();
    let out = repeat_kv(&x, 2).unwrap();
    assert_eq!(out.dims(), &[1, 4, 3, 4]);
    // First two heads should be copies of head 0, next two copies of head 1
    let flat = out.to_flat_vec::<f32>().unwrap();
    // head 0 data (first 12 elements)
    let h0: Vec<f32> = (0..12).map(|i| i as f32).collect();
    assert_eq!(&flat[0..12], &h0[..]);
    assert_eq!(&flat[12..24], &h0[..]);
}

// -- Finiteness validation (#1202) --------------------------------------------

#[test]
fn test_mha_nan_input_returns_error() {
    let mha = make_mha_4h_64d();
    let mut data = vec![0.1f32; 2 * 64];
    data[0] = f32::NAN;
    let x = DynTensor::from_vec(data, &[1, 2, 64], &Device::Cpu).unwrap();
    let result = mha.forward(&x, None, None, None, 0);
    assert!(result.is_err(), "NaN input should produce an error");
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Non-finite") || msg.contains("NaN"),
        "error should mention non-finite: {msg}"
    );
}

// -- sdpa scale validation (#1590) --------------------------------------------

#[test]
fn test_sdpa_nan_scale_rejected() {
    use crate::layers::attention::sdpa;
    let q = DynTensor::ones(&[1, 1, 2, 4], DType::F32, &Device::Cpu).unwrap();
    let k = q.clone();
    let v = q.clone();
    let result = sdpa(&q, &k, &v, None, f64::NAN);
    assert!(result.is_err(), "NaN scale should be rejected");
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("scale"), "error should mention scale: {msg}");
}

#[test]
fn test_sdpa_inf_scale_rejected() {
    use crate::layers::attention::sdpa;
    let q = DynTensor::ones(&[1, 1, 2, 4], DType::F32, &Device::Cpu).unwrap();
    let k = q.clone();
    let v = q.clone();
    let result = sdpa(&q, &k, &v, None, f64::INFINITY);
    assert!(result.is_err(), "Inf scale should be rejected");
}

// -- causal_mask_dtype and causal_mask_with_offset dtype tests (#1710) ---

#[test]
fn test_causal_mask_dtype_bf16_shape_and_values() {
    let mask = causal_mask_dtype(3, DType::BF16, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 3, 3]);
    assert_eq!(mask.dtype(), DType::BF16);
    // Convert to f32 for value inspection.
    let f32_mask = mask.to_dtype(DType::F32).unwrap();
    let flat = f32_mask.to_flat_vec::<f32>().unwrap();
    // Row 0: [0, -inf, -inf]
    assert_eq!(flat[0], 0.0);
    assert!(
        flat[1].is_infinite() && flat[1] < 0.0,
        "bf16 -inf must survive: {}",
        flat[1]
    );
    assert!(flat[2].is_infinite() && flat[2] < 0.0);
    // Row 1: [0, 0, -inf]
    assert_eq!(flat[3], 0.0);
    assert_eq!(flat[4], 0.0);
    assert!(flat[5].is_infinite() && flat[5] < 0.0);
    // Row 2: [0, 0, 0]
    assert_eq!(flat[6], 0.0);
    assert_eq!(flat[7], 0.0);
    assert_eq!(flat[8], 0.0);
}

#[test]
fn test_causal_mask_dtype_f16_preserves_dtype() {
    let mask = causal_mask_dtype(2, DType::F16, &Device::Cpu).unwrap();
    assert_eq!(mask.dtype(), DType::F16);
    assert_eq!(mask.dims(), &[1, 1, 2, 2]);
}

#[test]
fn test_causal_mask_with_offset_bf16() {
    // Decode step: 1 new token attending to 4 total tokens.
    let mask = causal_mask_with_offset(1, 4, DType::BF16, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 1, 4]);
    assert_eq!(mask.dtype(), DType::BF16);
    // Single row should be all zeros (token at position 3 can attend to 0,1,2,3).
    let f32_mask = mask.to_dtype(DType::F32).unwrap();
    let flat = f32_mask.to_flat_vec::<f32>().unwrap();
    assert!(
        flat.iter().all(|&v| v == 0.0),
        "decode mask should be all-zero: {flat:?}"
    );
}

#[test]
fn test_causal_mask_with_offset_bf16_prefill() {
    // Prefill: 3 new tokens, 5 total tokens (2 already cached).
    let mask = causal_mask_with_offset(3, 5, DType::BF16, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 3, 5]);
    assert_eq!(mask.dtype(), DType::BF16);
    let f32_mask = mask.to_dtype(DType::F32).unwrap();
    let flat = f32_mask.to_flat_vec::<f32>().unwrap();
    // Row 0 (abs pos 2): attend to 0,1,2; mask 3,4
    assert_eq!(flat[0], 0.0);
    assert_eq!(flat[1], 0.0);
    assert_eq!(flat[2], 0.0);
    assert!(flat[3].is_infinite() && flat[3] < 0.0);
    assert!(flat[4].is_infinite() && flat[4] < 0.0);
    // Row 1 (abs pos 3): attend to 0,1,2,3; mask 4
    assert_eq!(flat[5], 0.0);
    assert_eq!(flat[6], 0.0);
    assert_eq!(flat[7], 0.0);
    assert_eq!(flat[8], 0.0);
    assert!(flat[9].is_infinite() && flat[9] < 0.0);
    // Row 2 (abs pos 4): attend to all
    assert!(flat[10..15].iter().all(|&v| v == 0.0));
}

// -- KV cache, RoPE, VarBuilder, Bias tests (extracted to integration file) ---
#[path = "multi_head_tests_integration.rs"]
mod integration;
