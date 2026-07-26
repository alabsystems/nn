// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended SDPA and attention infrastructure tests.
//!
//! Covers: MultiHeadAttention configuration, scaled dot-product attention,
//! causal masks, ALiBi bias, KvCache operations, RotaryEmbedding variants,
//! repeat_kv for GQA, AttentionMode, and edge cases.
//!
//! Part of #4186.

use crate::dyn_tensor::test_helpers::make_linear_seeded;
use crate::dyn_tensor::DynTensor;
use crate::layers::attention::{
    alibi_bias, alibi_bias_scaled, alibi_slopes, causal_mask, causal_mask_dtype,
    causal_mask_with_offset, repeat_kv, sdpa, sdpa_causal, AttentionMode, HalfRotaryEmbedding,
    MultiHeadAttention, RotaryEmbedding, RotaryEmbedding2d, YarnScaling,
};
use crate::layers::{KvCacheLayer, Module};
use crate::test_prng::rand_f32_vec;
use crate::{DType, Device};

/// Helper: create a DynTensor with deterministic random data.
fn rand_tensor(seed: u64, dims: &[usize]) -> DynTensor {
    let numel: usize = dims.iter().product();
    let data = rand_f32_vec(seed, numel, -1.0, 1.0);
    DynTensor::from_vec(data, dims, &Device::Cpu).unwrap()
}

/// Helper: create a standard MHA with dim=64, given heads config.
fn make_mha(dim: usize, num_heads: usize, num_kv_heads: usize) -> MultiHeadAttention {
    let kv_dim = dim / num_heads * num_kv_heads;
    MultiHeadAttention::new(
        make_linear_seeded(dim, dim, 1.0),
        make_linear_seeded(kv_dim, dim, 2.0),
        make_linear_seeded(kv_dim, dim, 3.0),
        make_linear_seeded(dim, dim, 4.0),
        num_heads,
        num_kv_heads,
    )
    .expect("valid MHA")
}

// =============================================================================
// MultiHeadAttention configuration
// =============================================================================

#[test]
fn test_mha_config_num_heads_head_dim() {
    let mha = make_mha(128, 8, 8);
    assert_eq!(mha.num_heads(), 8);
    assert_eq!(mha.num_kv_heads(), 8);
    assert_eq!(mha.head_dim(), 16); // 128 / 8
}

#[test]
fn test_mha_config_gqa_4_to_2() {
    let mha = make_mha(64, 4, 2);
    assert_eq!(mha.num_heads(), 4);
    assert_eq!(mha.num_kv_heads(), 2);
    assert_eq!(mha.head_dim(), 16); // 64 / 4
}

#[test]
fn test_mha_config_gqa_8_to_1_mqa() {
    // Multi-query attention: 8 query heads, 1 kv head
    let mha = make_mha(64, 8, 1);
    assert_eq!(mha.num_heads(), 8);
    assert_eq!(mha.num_kv_heads(), 1);
    assert_eq!(mha.head_dim(), 8); // 64 / 8
}

#[test]
fn test_mha_config_embed_dim_consistency() {
    // embed_dim = num_heads * head_dim
    let dim = 256;
    let mha = make_mha(dim, 16, 4);
    assert_eq!(mha.num_heads() * mha.head_dim(), dim);
}

#[test]
fn test_mha_zero_heads_error() {
    let result = MultiHeadAttention::new(
        make_linear_seeded(64, 64, 1.0),
        make_linear_seeded(64, 64, 2.0),
        make_linear_seeded(64, 64, 3.0),
        make_linear_seeded(64, 64, 4.0),
        0,
        4,
    );
    assert!(result.is_err());
}

#[test]
fn test_mha_indivisible_heads_error() {
    // 5 query heads, 3 kv heads -- not divisible
    let result = MultiHeadAttention::new(
        make_linear_seeded(64, 64, 1.0),
        make_linear_seeded(64, 64, 2.0),
        make_linear_seeded(64, 64, 3.0),
        make_linear_seeded(64, 64, 4.0),
        5,
        3,
    );
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("divisible"), "error: {msg}");
}

// =============================================================================
// Scaled dot-product attention (sdpa, sdpa_causal)
// =============================================================================

#[test]
fn test_sdpa_basic_output_shape() {
    let b = 2;
    let h = 4;
    let s = 8;
    let d = 16;
    let scale = 1.0 / (d as f64).sqrt();
    let q = rand_tensor(100, &[b, h, s, d]);
    let k = rand_tensor(101, &[b, h, s, d]);
    let v = rand_tensor(102, &[b, h, s, d]);
    let out = sdpa(&q, &k, &v, None, scale).unwrap();
    assert_eq!(out.dims(), &[b, h, s, d]);
}

#[test]
fn test_sdpa_with_mask() {
    let b = 1;
    let h = 2;
    let s = 4;
    let d = 8;
    let scale = 1.0 / (d as f64).sqrt();
    let q = rand_tensor(200, &[b, h, s, d]);
    let k = rand_tensor(201, &[b, h, s, d]);
    let v = rand_tensor(202, &[b, h, s, d]);
    let mask = causal_mask(s, &Device::Cpu).unwrap();
    let out = sdpa(&q, &k, &v, Some(&mask), scale).unwrap();
    assert_eq!(out.dims(), &[b, h, s, d]);
    let flat = out.to_flat_vec::<f32>().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()), "output has NaN/Inf");
}

#[test]
fn test_sdpa_output_finite() {
    let b = 1;
    let h = 1;
    let s = 3;
    let d = 4;
    let scale = 1.0 / (d as f64).sqrt();
    let q = rand_tensor(300, &[b, h, s, d]);
    let k = rand_tensor(301, &[b, h, s, d]);
    let v = rand_tensor(302, &[b, h, s, d]);
    let out = sdpa(&q, &k, &v, None, scale).unwrap();
    let flat = out.to_flat_vec::<f32>().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "sdpa output must be finite"
    );
}

#[test]
fn test_sdpa_nan_scale_error() {
    let q = DynTensor::ones(&[1, 1, 2, 4], DType::F32, &Device::Cpu).unwrap();
    let k = q.clone();
    let v = q.clone();
    assert!(sdpa(&q, &k, &v, None, f64::NAN).is_err());
}

#[test]
fn test_sdpa_inf_scale_error() {
    let q = DynTensor::ones(&[1, 1, 2, 4], DType::F32, &Device::Cpu).unwrap();
    let k = q.clone();
    let v = q.clone();
    assert!(sdpa(&q, &k, &v, None, f64::INFINITY).is_err());
}

#[test]
fn test_sdpa_causal_output_shape_and_finite() {
    let b = 2;
    let h = 4;
    let s = 6;
    let d = 16;
    let scale = 1.0 / (d as f64).sqrt();
    let q = rand_tensor(400, &[b, h, s, d]);
    let k = rand_tensor(401, &[b, h, s, d]);
    let v = rand_tensor(402, &[b, h, s, d]);
    let out = sdpa_causal(&q, &k, &v, scale).unwrap();
    assert_eq!(out.dims(), &[b, h, s, d]);
    let flat = out.to_flat_vec::<f32>().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "sdpa_causal output must be finite"
    );
}

#[test]
fn test_sdpa_causal_nan_scale_error() {
    let q = DynTensor::ones(&[1, 1, 2, 4], DType::F32, &Device::Cpu).unwrap();
    let k = q.clone();
    let v = q.clone();
    assert!(sdpa_causal(&q, &k, &v, f64::NAN).is_err());
}

#[test]
fn test_sdpa_causal_single_token() {
    // Single-token attention: seq_len=1 should succeed (no future tokens to mask)
    let b = 1;
    let h = 2;
    let d = 8;
    let scale = 1.0 / (d as f64).sqrt();
    let q = rand_tensor(500, &[b, h, 1, d]);
    let k = rand_tensor(501, &[b, h, 1, d]);
    let v = rand_tensor(502, &[b, h, 1, d]);
    let out = sdpa_causal(&q, &k, &v, scale).unwrap();
    assert_eq!(out.dims(), &[b, h, 1, d]);
}

#[test]
fn test_sdpa_deterministic() {
    let b = 1;
    let h = 2;
    let s = 4;
    let d = 8;
    let scale = 1.0 / (d as f64).sqrt();
    let q = rand_tensor(600, &[b, h, s, d]);
    let k = rand_tensor(601, &[b, h, s, d]);
    let v = rand_tensor(602, &[b, h, s, d]);
    let out1 = sdpa(&q, &k, &v, None, scale).unwrap();
    let out2 = sdpa(&q, &k, &v, None, scale).unwrap();
    let v1 = out1.to_flat_vec::<f32>().unwrap();
    let v2 = out2.to_flat_vec::<f32>().unwrap();
    for (a, b) in v1.iter().zip(v2.iter()) {
        assert!((a - b).abs() < 1e-6, "non-deterministic: {a} vs {b}");
    }
}

// =============================================================================
// Causal mask generation
// =============================================================================

#[test]
fn test_causal_mask_seq1_all_zero() {
    let mask = causal_mask(1, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 1, 1]);
    let flat = mask.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat, vec![0.0]);
}

#[test]
fn test_causal_mask_diagonal_is_zero() {
    let s = 6;
    let mask = causal_mask(s, &Device::Cpu).unwrap();
    let data = mask.to_f32_array().unwrap();
    for i in 0..s {
        let val = data[[0, 0, i, i]];
        assert_eq!(val, 0.0, "Diagonal at ({i},{i}) should be 0.0, got {val}");
    }
}

#[test]
fn test_causal_mask_lower_triangle_zero() {
    let s = 5;
    let mask = causal_mask(s, &Device::Cpu).unwrap();
    let data = mask.to_f32_array().unwrap();
    for i in 0..s {
        for j in 0..=i {
            let val = data[[0, 0, i, j]];
            assert_eq!(
                val, 0.0,
                "Lower triangle at ({i},{j}) should be 0.0, got {val}"
            );
        }
    }
}

#[test]
fn test_causal_mask_upper_triangle_neg_inf() {
    let s = 5;
    let mask = causal_mask(s, &Device::Cpu).unwrap();
    let data = mask.to_f32_array().unwrap();
    for i in 0..s {
        for j in (i + 1)..s {
            let val = data[[0, 0, i, j]];
            assert!(
                val == f32::NEG_INFINITY,
                "Upper triangle at ({i},{j}) should be -inf, got {val}"
            );
        }
    }
}

#[test]
fn test_causal_mask_with_offset_decode_step() {
    // Simulating autoregressive decode: 1 new token, 8 total tokens
    let mask = causal_mask_with_offset(1, 8, DType::F32, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 1, 8]);
    // Token at position 7 can attend to all 8 positions
    let flat = mask.to_flat_vec::<f32>().unwrap();
    assert!(
        flat.iter().all(|&v| v == 0.0),
        "decode mask should be all zero"
    );
}

#[test]
fn test_causal_mask_with_offset_partial_prefill() {
    // 3 new tokens, 6 total (3 already cached)
    let mask = causal_mask_with_offset(3, 6, DType::F32, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 3, 6]);
    let data = mask.to_f32_array().unwrap();
    // Row 0 (abs pos 3): attend 0..=3, mask 4,5
    for j in 0..=3 {
        assert_eq!(data[[0, 0, 0, j]], 0.0, "row0 col{j}");
    }
    assert_eq!(data[[0, 0, 0, 4]], f32::NEG_INFINITY);
    assert_eq!(data[[0, 0, 0, 5]], f32::NEG_INFINITY);
    // Row 2 (abs pos 5): attend to all
    for j in 0..6 {
        assert_eq!(data[[0, 0, 2, j]], 0.0, "row2 col{j}");
    }
}

#[test]
fn test_causal_mask_with_offset_total_lt_new_error() {
    let result = causal_mask_with_offset(5, 3, DType::F32, &Device::Cpu);
    assert!(result.is_err(), "total_tokens < new_tokens should error");
}

#[test]
fn test_causal_mask_with_offset_zero_tokens_error() {
    assert!(causal_mask_with_offset(0, 5, DType::F32, &Device::Cpu).is_err());
    assert!(causal_mask_with_offset(3, 0, DType::F32, &Device::Cpu).is_err());
}

#[test]
fn test_causal_mask_dtype_bf16() {
    let mask = causal_mask_dtype(4, DType::BF16, &Device::Cpu).unwrap();
    assert_eq!(mask.dtype(), DType::BF16);
    assert_eq!(mask.dims(), &[1, 1, 4, 4]);
}

// =============================================================================
// ALiBi bias computation
// =============================================================================

#[test]
fn test_alibi_slopes_power_of_2() {
    let slopes = alibi_slopes(8).unwrap();
    assert_eq!(slopes.len(), 8);
    // slopes[h] = 2^(-8*(h+1)/8) = 2^(-(h+1))
    for (h, &s) in slopes.iter().enumerate() {
        let expected = 2f32.powf(-((h + 1) as f32));
        assert!(
            (s - expected).abs() < 1e-6,
            "h={h}: expected {expected}, got {s}"
        );
    }
}

#[test]
fn test_alibi_slopes_non_power_of_2() {
    let slopes = alibi_slopes(6).unwrap();
    assert_eq!(slopes.len(), 6);
    // Should be strictly decreasing
    for i in 1..6 {
        assert!(slopes[i] < slopes[i - 1], "slopes not decreasing at {i}");
    }
}

#[test]
fn test_alibi_slopes_single_head() {
    let slopes = alibi_slopes(1).unwrap();
    assert_eq!(slopes.len(), 1);
    // 2^(-8*1/1) = 2^-8
    let expected = 2f32.powf(-8.0);
    assert!((slopes[0] - expected).abs() < 1e-6);
}

#[test]
fn test_alibi_slopes_zero_heads_error() {
    assert!(alibi_slopes(0).is_err());
}

#[test]
fn test_alibi_bias_shape() {
    let bias = alibi_bias(4, 8, &Device::Cpu).unwrap();
    assert_eq!(bias.dims(), &[1, 4, 8, 8]);
}

#[test]
fn test_alibi_bias_diagonal_zero() {
    let num_heads = 4;
    let seq_len = 6;
    let bias = alibi_bias(num_heads, seq_len, &Device::Cpu).unwrap();
    let data = bias.to_f32_array().unwrap();
    for h in 0..num_heads {
        for i in 0..seq_len {
            let val = data[[0, h, i, i]];
            assert!(
                val.abs() < 1e-6,
                "diagonal at h={h}, i={i} should be 0, got {val}"
            );
        }
    }
}

#[test]
fn test_alibi_bias_antisymmetric() {
    let num_heads = 2;
    let seq_len = 4;
    let bias = alibi_bias(num_heads, seq_len, &Device::Cpu).unwrap();
    let data = bias.to_f32_array().unwrap();
    for h in 0..num_heads {
        for i in 0..seq_len {
            for j in 0..seq_len {
                let bij = data[[0, h, i, j]];
                let bji = data[[0, h, j, i]];
                let sum = bij + bji;
                assert!(
                    sum.abs() < 1e-5,
                    "antisymmetry violated: h={h} b[{i}][{j}]={bij} + b[{j}][{i}]={bji} = {sum}"
                );
            }
        }
    }
}

#[test]
fn test_alibi_bias_scaled_identity_scale() {
    let num_heads = 4;
    let seq_len = 6;
    let scale = DynTensor::ones(&[num_heads], DType::F32, &Device::Cpu).unwrap();
    let unscaled = alibi_bias(num_heads, seq_len, &Device::Cpu).unwrap();
    let scaled = alibi_bias_scaled(num_heads, seq_len, &scale, &Device::Cpu).unwrap();
    let u = unscaled.to_flat_vec::<f32>().unwrap();
    let s = scaled.to_flat_vec::<f32>().unwrap();
    for (a, b) in u.iter().zip(s.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "scale=1 should give identical bias: {a} vs {b}"
        );
    }
}

#[test]
fn test_alibi_bias_scaled_double() {
    let num_heads = 2;
    let seq_len = 3;
    let scale_data = vec![2.0f32; num_heads];
    let scale = DynTensor::from_vec(scale_data, &[num_heads], &Device::Cpu).unwrap();
    let unscaled = alibi_bias(num_heads, seq_len, &Device::Cpu).unwrap();
    let scaled = alibi_bias_scaled(num_heads, seq_len, &scale, &Device::Cpu).unwrap();
    let u = unscaled.to_flat_vec::<f32>().unwrap();
    let s = scaled.to_flat_vec::<f32>().unwrap();
    for (a, b) in u.iter().zip(s.iter()) {
        assert!((a * 2.0 - b).abs() < 1e-5, "scale=2 mismatch: {a}*2 vs {b}");
    }
}

// =============================================================================
// KvCache operations
// =============================================================================

#[test]
fn test_kvcache_empty_initial_state() {
    let cache = KvCacheLayer::empty();
    assert_eq!(cache.seq_len(), 0);
    assert_eq!(cache.current_seq_len(), 0);
    assert_eq!(cache.dim(), 2);
    assert!(cache.is_empty());
    assert!(cache.key().unwrap().is_none());
    assert!(cache.value().unwrap().is_none());
}

#[test]
fn test_kvcache_new_dim2() {
    let cache = KvCacheLayer::new(2, 512).unwrap();
    assert!(cache.is_empty());
}

#[test]
fn test_kvcache_new_wrong_dim_error() {
    assert!(KvCacheLayer::new(0, 512).is_err());
    assert!(KvCacheLayer::new(1, 512).is_err());
    assert!(KvCacheLayer::new(3, 512).is_err());
}

#[test]
fn test_kvcache_append_single() {
    let mut cache = KvCacheLayer::empty();
    let k = rand_tensor(700, &[1, 4, 1, 16]); // [batch, heads, seq=1, head_dim]
    let v = rand_tensor(701, &[1, 4, 1, 16]);
    let (full_k, full_v) = cache.append(&k, &v).unwrap();
    assert_eq!(cache.seq_len(), 1);
    assert_eq!(full_k.dims(), &[1, 4, 1, 16]);
    assert_eq!(full_v.dims(), &[1, 4, 1, 16]);
}

#[test]
fn test_kvcache_append_multiple() {
    let mut cache = KvCacheLayer::empty();
    // First append: 3 tokens
    let k1 = rand_tensor(710, &[1, 2, 3, 8]);
    let v1 = rand_tensor(711, &[1, 2, 3, 8]);
    let (fk1, fv1) = cache.append(&k1, &v1).unwrap();
    assert_eq!(cache.seq_len(), 3);
    assert_eq!(fk1.dims(), &[1, 2, 3, 8]);
    assert_eq!(fv1.dims(), &[1, 2, 3, 8]);

    // Second append: 1 token
    let k2 = rand_tensor(712, &[1, 2, 1, 8]);
    let v2 = rand_tensor(713, &[1, 2, 1, 8]);
    let (fk2, fv2) = cache.append(&k2, &v2).unwrap();
    assert_eq!(cache.seq_len(), 4);
    assert_eq!(fk2.dims(), &[1, 2, 4, 8]);
    assert_eq!(fv2.dims(), &[1, 2, 4, 8]);
}

#[test]
fn test_kvcache_clear_preserves_capacity() {
    let mut cache = KvCacheLayer::empty();
    let k = rand_tensor(720, &[1, 2, 5, 8]);
    let v = rand_tensor(721, &[1, 2, 5, 8]);
    cache.append(&k, &v).unwrap();
    let cap_before = cache.buffer_capacity();
    assert!(cap_before > 0);

    cache.clear();
    assert_eq!(cache.seq_len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.buffer_capacity(), cap_before); // capacity preserved
}

#[test]
fn test_kvcache_reset_drops_buffers() {
    let mut cache = KvCacheLayer::empty();
    let k = rand_tensor(730, &[1, 2, 5, 8]);
    let v = rand_tensor(731, &[1, 2, 5, 8]);
    cache.append(&k, &v).unwrap();
    assert!(!cache.is_empty());

    cache.reset();
    assert!(cache.is_empty());
    assert_eq!(cache.buffer_capacity(), 0); // fully dropped
}

#[test]
fn test_kvcache_invalidate_increments_generation() {
    let mut cache = KvCacheLayer::empty();
    assert_eq!(cache.weight_generation(), 0);
    let k = rand_tensor(740, &[1, 2, 3, 8]);
    let v = rand_tensor(741, &[1, 2, 3, 8]);
    cache.append(&k, &v).unwrap();

    cache.invalidate();
    assert_eq!(cache.weight_generation(), 1);
    assert!(cache.is_empty());

    cache.invalidate();
    assert_eq!(cache.weight_generation(), 2);
}

#[test]
fn test_kvcache_candle_compat_aliases() {
    let cache = KvCacheLayer::empty();
    assert!(cache.k().unwrap().is_none());
    assert!(cache.v().unwrap().is_none());
    assert_eq!(cache.current_seq_len(), cache.seq_len());
}

// =============================================================================
// RotaryEmbedding (standard)
// =============================================================================

#[test]
fn test_rope_basic_shape() {
    let head_dim = 16;
    let max_seq = 64;
    let rope = RotaryEmbedding::new(head_dim, max_seq, 10000.0, &Device::Cpu).unwrap();
    assert_eq!(rope.head_dim(), head_dim);
    assert_eq!(rope.max_seq_len(), max_seq);

    let x = rand_tensor(800, &[1, 4, 8, head_dim]);
    let out = rope.apply(&x, 0).unwrap();
    assert_eq!(out.dims(), x.dims());
}

#[test]
fn test_rope_offset() {
    let head_dim = 8;
    let max_seq = 32;
    let rope = RotaryEmbedding::new(head_dim, max_seq, 10000.0, &Device::Cpu).unwrap();
    let x = rand_tensor(810, &[1, 2, 4, head_dim]);
    // Apply at offset 0
    let out0 = rope.apply(&x, 0).unwrap();
    // Apply at offset 10
    let out10 = rope.apply(&x, 10).unwrap();
    // Outputs should differ (different positions = different rotations)
    let v0 = out0.to_flat_vec::<f32>().unwrap();
    let v10 = out10.to_flat_vec::<f32>().unwrap();
    let any_differ = v0.iter().zip(v10.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(
        any_differ,
        "RoPE at different offsets should produce different outputs"
    );
}

#[test]
fn test_rope_exceeds_max_seq_error() {
    let head_dim = 8;
    let rope = RotaryEmbedding::new(head_dim, 16, 10000.0, &Device::Cpu).unwrap();
    let x = rand_tensor(820, &[1, 2, 10, head_dim]);
    // offset 10 + seq_len 10 = 20 > max_seq 16
    let result = rope.apply(&x, 10);
    assert!(result.is_err(), "exceeding max_seq_len should error");
}

#[test]
fn test_rope_odd_head_dim_error() {
    assert!(RotaryEmbedding::new(7, 16, 10000.0, &Device::Cpu).is_err());
}

#[test]
fn test_rope_zero_head_dim_error() {
    assert!(RotaryEmbedding::new(0, 16, 10000.0, &Device::Cpu).is_err());
}

#[test]
fn test_rope_zero_max_seq_error() {
    assert!(RotaryEmbedding::new(8, 0, 10000.0, &Device::Cpu).is_err());
}

#[test]
fn test_rope_apply_pair() {
    let head_dim = 8;
    let rope = RotaryEmbedding::new(head_dim, 64, 10000.0, &Device::Cpu).unwrap();
    let q = rand_tensor(830, &[1, 4, 3, head_dim]);
    let k = rand_tensor(831, &[1, 2, 3, head_dim]);
    let positions = vec![0, 1, 2];
    let (q_rot, k_rot) = rope.apply_pair(&q, &k, &positions).unwrap();
    assert_eq!(q_rot.dims(), q.dims());
    assert_eq!(k_rot.dims(), k.dims());
}

#[test]
fn test_rope_preserves_norm() {
    // RoPE is a rotation, so it should approximately preserve L2 norm
    let head_dim = 8;
    let rope = RotaryEmbedding::new(head_dim, 64, 10000.0, &Device::Cpu).unwrap();
    let x = rand_tensor(840, &[1, 1, 1, head_dim]);
    let out = rope.apply(&x, 0).unwrap();
    let x_flat = x.to_flat_vec::<f32>().unwrap();
    let o_flat = out.to_flat_vec::<f32>().unwrap();
    let x_norm: f32 = x_flat.iter().map(|v| v * v).sum::<f32>().sqrt();
    let o_norm: f32 = o_flat.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (x_norm - o_norm).abs() < 1e-4,
        "RoPE should preserve norm: {x_norm} vs {o_norm}"
    );
}

// =============================================================================
// RotaryEmbedding2d
// =============================================================================

#[test]
fn test_rope_2d_basic() {
    let head_dim = 8; // must be divisible by 4
    let rope2d = RotaryEmbedding2d::new(head_dim, 32, 10000.0, &Device::Cpu).unwrap();
    assert_eq!(rope2d.head_dim(), head_dim);
    assert_eq!(rope2d.max_position(), 32);

    let seq_len = 4;
    let x = rand_tensor(850, &[1, seq_len, head_dim]);
    let h_pos = vec![0, 0, 1, 1];
    let w_pos = vec![0, 1, 0, 1];
    let out = rope2d.apply(&x, &h_pos, &w_pos).unwrap();
    assert_eq!(out.dims(), x.dims());
}

#[test]
fn test_rope_2d_head_dim_not_mult_4_error() {
    assert!(RotaryEmbedding2d::new(6, 32, 10000.0, &Device::Cpu).is_err());
    assert!(RotaryEmbedding2d::new(0, 32, 10000.0, &Device::Cpu).is_err());
}

#[test]
fn test_rope_2d_position_exceeds_max_error() {
    let rope2d = RotaryEmbedding2d::new(8, 4, 10000.0, &Device::Cpu).unwrap();
    let x = rand_tensor(860, &[1, 2, 8]);
    let h_pos = vec![0, 5]; // 5 >= max_position=4
    let w_pos = vec![0, 0];
    assert!(rope2d.apply(&x, &h_pos, &w_pos).is_err());
}

// =============================================================================
// HalfRotaryEmbedding
// =============================================================================

#[test]
fn test_half_rope_basic() {
    let head_dim = 16; // must be divisible by 4
    let half_rope = HalfRotaryEmbedding::new(head_dim, 64, 10000.0, &Device::Cpu).unwrap();
    assert_eq!(half_rope.head_dim(), head_dim);
    assert_eq!(half_rope.rope_dim(), head_dim / 2);

    let x = rand_tensor(870, &[1, 2, 4, head_dim]);
    let out = half_rope.apply(&x, 0).unwrap();
    assert_eq!(out.dims(), x.dims());
}

#[test]
fn test_half_rope_second_half_unchanged() {
    let head_dim = 8;
    let half_rope = HalfRotaryEmbedding::new(head_dim, 64, 10000.0, &Device::Cpu).unwrap();
    let x = rand_tensor(880, &[1, 1, 2, head_dim]);
    let out = half_rope.apply(&x, 5).unwrap();

    // Extract second half of last dim (the pass-through portion)
    let x_pass = x.narrow(3, head_dim / 2, head_dim / 2).unwrap();
    let o_pass = out.narrow(3, head_dim / 2, head_dim / 2).unwrap();
    let xv = x_pass.to_flat_vec::<f32>().unwrap();
    let ov = o_pass.to_flat_vec::<f32>().unwrap();
    for (a, b) in xv.iter().zip(ov.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "pass-through half should be unchanged: {a} vs {b}"
        );
    }
}

#[test]
fn test_half_rope_head_dim_not_mult_4_error() {
    assert!(HalfRotaryEmbedding::new(6, 64, 10000.0, &Device::Cpu).is_err());
}

#[test]
fn test_half_rope_apply_pair() {
    let head_dim = 8;
    let half_rope = HalfRotaryEmbedding::new(head_dim, 64, 10000.0, &Device::Cpu).unwrap();
    let q = rand_tensor(890, &[1, 4, 3, head_dim]);
    let k = rand_tensor(891, &[1, 2, 3, head_dim]);
    let positions = vec![0, 1, 2];
    let (q_rot, k_rot) = half_rope.apply_pair(&q, &k, &positions).unwrap();
    assert_eq!(q_rot.dims(), q.dims());
    assert_eq!(k_rot.dims(), k.dims());
}

// =============================================================================
// YaRN scaling
// =============================================================================

#[test]
fn test_yarn_rope_basic() {
    let head_dim = 16;
    let max_seq = 128;
    let yarn = YarnScaling::new(4.0, 1.0, 32.0, 1.0, 32);
    let rope = RotaryEmbedding::new_yarn(head_dim, max_seq, 10000.0, &yarn, &Device::Cpu).unwrap();
    assert_eq!(rope.head_dim(), head_dim);
    assert_eq!(rope.max_seq_len(), max_seq);

    let x = rand_tensor(900, &[1, 2, 8, head_dim]);
    let out = rope.apply(&x, 0).unwrap();
    assert_eq!(out.dims(), x.dims());
}

#[test]
fn test_yarn_rope_different_from_standard() {
    let head_dim = 8;
    let max_seq = 32;
    let std_rope = RotaryEmbedding::new(head_dim, max_seq, 10000.0, &Device::Cpu).unwrap();
    let yarn = YarnScaling::new(4.0, 1.0, 32.0, 1.0, 16);
    let yarn_rope =
        RotaryEmbedding::new_yarn(head_dim, max_seq, 10000.0, &yarn, &Device::Cpu).unwrap();

    let x = rand_tensor(910, &[1, 1, 4, head_dim]);
    let out_std = std_rope.apply(&x, 0).unwrap();
    let out_yarn = yarn_rope.apply(&x, 0).unwrap();
    let v_std = out_std.to_flat_vec::<f32>().unwrap();
    let v_yarn = out_yarn.to_flat_vec::<f32>().unwrap();
    let any_differ = v_std
        .iter()
        .zip(v_yarn.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(
        any_differ,
        "YaRN and standard RoPE should produce different outputs"
    );
}

#[test]
fn test_yarn_invalid_factor_error() {
    let yarn = YarnScaling::new(0.0, 1.0, 32.0, 1.0, 32);
    assert!(RotaryEmbedding::new_yarn(8, 32, 10000.0, &yarn, &Device::Cpu).is_err());
}

#[test]
fn test_yarn_negative_factor_error() {
    let yarn = YarnScaling::new(-1.0, 1.0, 32.0, 1.0, 32);
    assert!(RotaryEmbedding::new_yarn(8, 32, 10000.0, &yarn, &Device::Cpu).is_err());
}

// =============================================================================
// repeat_kv for GQA
// =============================================================================

#[test]
fn test_repeat_kv_identity() {
    let x = rand_tensor(950, &[2, 4, 6, 16]);
    let out = repeat_kv(&x, 1).unwrap();
    assert_eq!(out.dims(), x.dims());
    let xv = x.to_flat_vec::<f32>().unwrap();
    let ov = out.to_flat_vec::<f32>().unwrap();
    assert_eq!(xv, ov);
}

#[test]
fn test_repeat_kv_triple() {
    let x = rand_tensor(960, &[1, 2, 4, 8]);
    let out = repeat_kv(&x, 3).unwrap();
    assert_eq!(out.dims(), &[1, 6, 4, 8]); // 2 * 3 = 6 heads
}

#[test]
fn test_repeat_kv_quadruple() {
    let x = rand_tensor(970, &[1, 1, 3, 4]);
    let out = repeat_kv(&x, 4).unwrap();
    assert_eq!(out.dims(), &[1, 4, 3, 4]);
}

#[test]
fn test_repeat_kv_preserves_values() {
    // With 1 kv head repeated 2x, both output heads should have identical data
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data, &[1, 1, 3, 4], &Device::Cpu).unwrap();
    let out = repeat_kv(&x, 2).unwrap();
    assert_eq!(out.dims(), &[1, 2, 3, 4]);
    let flat = out.to_flat_vec::<f32>().unwrap();
    // Head 0 and head 1 should be identical
    assert_eq!(&flat[0..12], &flat[12..24]);
}

// =============================================================================
// Attention mode selection
// =============================================================================

#[test]
fn test_attention_mode_variants() {
    let global = AttentionMode::Global;
    let window = AttentionMode::Window;
    assert_ne!(global, window);
    assert_eq!(global, AttentionMode::Global);
    assert_eq!(window, AttentionMode::Window);
}

#[test]
fn test_attention_mode_copy() {
    let mode = AttentionMode::Global;
    let mode2 = mode; // Copy
    assert_eq!(mode, mode2);
}

// =============================================================================
// Edge cases
// =============================================================================

#[test]
fn test_sdpa_batch_size_1() {
    let scale = 1.0 / 4.0_f64.sqrt();
    let q = rand_tensor(1000, &[1, 1, 1, 4]);
    let k = rand_tensor(1001, &[1, 1, 1, 4]);
    let v = rand_tensor(1002, &[1, 1, 1, 4]);
    let out = sdpa(&q, &k, &v, None, scale).unwrap();
    assert_eq!(out.dims(), &[1, 1, 1, 4]);
    // With seq=1, attention is just softmax over a single element = 1.0
    // so output should be exactly v
    let ov = out.to_flat_vec::<f32>().unwrap();
    let vv = v.to_flat_vec::<f32>().unwrap();
    for (a, b) in ov.iter().zip(vv.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "single-element attention should return v: {a} vs {b}"
        );
    }
}

#[test]
fn test_mha_self_attention_batch_1_seq_1() {
    let mha = make_mha(64, 4, 4);
    let x = rand_tensor(1010, &[1, 1, 64]);
    let out = <MultiHeadAttention as Module>::forward(&mha, &x).unwrap();
    assert_eq!(out.dims(), &[1, 1, 64]);
    let flat = out.to_flat_vec::<f32>().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "output should be finite"
    );
}

#[test]
fn test_kvcache_append_then_clear_then_append() {
    let mut cache = KvCacheLayer::empty();
    let k1 = rand_tensor(1020, &[1, 2, 3, 8]);
    let v1 = rand_tensor(1021, &[1, 2, 3, 8]);
    cache.append(&k1, &v1).unwrap();
    assert_eq!(cache.seq_len(), 3);

    cache.clear();
    assert_eq!(cache.seq_len(), 0);

    let k2 = rand_tensor(1022, &[1, 2, 2, 8]);
    let v2 = rand_tensor(1023, &[1, 2, 2, 8]);
    let (fk, fv) = cache.append(&k2, &v2).unwrap();
    assert_eq!(cache.seq_len(), 2);
    assert_eq!(fk.dims(), &[1, 2, 2, 8]);
    assert_eq!(fv.dims(), &[1, 2, 2, 8]);
}

#[test]
fn test_rope_2d_rank_2_input() {
    // Minimal rank: [seq_len, head_dim]
    let rope2d = RotaryEmbedding2d::new(8, 16, 10000.0, &Device::Cpu).unwrap();
    let x = rand_tensor(1030, &[3, 8]);
    let h_pos = vec![0, 1, 2];
    let w_pos = vec![0, 0, 0];
    let out = rope2d.apply(&x, &h_pos, &w_pos).unwrap();
    assert_eq!(out.dims(), &[3, 8]);
}

#[test]
fn test_causal_mask_large_seq() {
    // Verify we can generate a mask for a moderately large sequence without overflow
    let s = 128;
    let mask = causal_mask(s, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, s, s]);
}

#[test]
fn test_alibi_bias_large_heads() {
    let bias = alibi_bias(16, 4, &Device::Cpu).unwrap();
    assert_eq!(bias.dims(), &[1, 16, 4, 4]);
}
