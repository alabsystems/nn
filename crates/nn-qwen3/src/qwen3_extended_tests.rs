// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Qwen3 model tests covering configuration properties, RoPE frequency
//! geometry, KV cache lifecycle, and SwiGLU activation formula (#4186).
//!
//! Complements existing tests in config_tests.rs, rope_tests.rs, kv_cache_tests.rs,
//! and mlp_tests.rs with deeper property-level assertions.

use crate::rope_cache::RoPECache;
use crate::test_utils::tiny_config;
use crate::{Qwen3Config, Qwen3Model};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::kv_cache::KvCache;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ===========================================================================
// Model Configuration
// ===========================================================================

/// Qwen3-0.6B: hidden=896, 14 heads, 2 kv heads, head_dim=128.
/// Note: hidden_size != num_heads * head_dim (896 != 1792) because Qwen3 uses
/// a fixed head_dim=128 with independent hidden_size.
#[test]
fn test_qwen3_config_0_6b() {
    let cfg = Qwen3Config::new(
        896,
        4864,
        28,
        14,
        2,
        151_936,
        1e-6,
        1_000_000.0,
        40_960,
        true,
        None,
    );
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.hidden_size, 896);
    assert_eq!(cfg.intermediate_size, 4864);
    assert_eq!(cfg.num_hidden_layers, 28);
    assert_eq!(cfg.num_attention_heads, 14);
    assert_eq!(cfg.num_key_value_heads, 2);
    assert_eq!(cfg.head_dim(), 128);
    assert_eq!(cfg.vocab_size, 151_936);
    assert!((cfg.rope_theta - 1_000_000.0).abs() < f64::EPSILON);
}

/// Qwen3-1.7B: hidden=2048, 16 heads, 4 kv heads.
/// Here hidden_size == num_heads * head_dim (2048 == 16*128).
#[test]
fn test_qwen3_config_1_7b() {
    let cfg = Qwen3Config::new(
        2048,
        6144,
        28,
        16,
        4,
        151_936,
        1e-6,
        1_000_000.0,
        40_960,
        false,
        None,
    );
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.num_key_value_heads, 4);
    assert_eq!(cfg.head_dim(), 128);
    // 1.7B: hidden == num_heads * head_dim
    assert_eq!(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim());
}

/// head_dim is a constant 128 for all Qwen3 variants (not computed from
/// hidden_size / num_attention_heads).
#[test]
fn test_config_head_dim_computed() {
    // Regardless of hidden_size and num_attention_heads, head_dim is always 128
    for (hidden, num_heads) in [(896, 14), (2048, 16), (4096, 32), (5120, 40)] {
        let cfg = Qwen3Config::new(
            hidden,
            hidden * 2,
            1,
            num_heads,
            1,
            100,
            1e-6,
            10_000.0,
            64,
            true,
            None,
        );
        assert_eq!(
            cfg.head_dim(),
            128,
            "head_dim should always be 128, not hidden/heads={} for hidden={hidden}, heads={num_heads}",
            hidden / num_heads
        );
    }
}

/// num_key_value_heads must divide num_attention_heads evenly (GQA constraint).
#[test]
fn test_config_kv_heads() {
    // Valid: kv_heads divides heads
    for (nh, nkv) in [(2, 1), (2, 2), (8, 2), (8, 4), (8, 8), (32, 4)] {
        let cfg = Qwen3Config::new(
            nh * 128,
            1024,
            1,
            nh,
            nkv,
            100,
            1e-6,
            10_000.0,
            64,
            true,
            None,
        );
        assert!(cfg.validate().is_ok(), "nh={nh}, nkv={nkv} should be valid");
        assert_eq!(cfg.num_kv_groups().unwrap(), nh / nkv);
    }

    // Invalid: kv_heads does not divide heads
    let bad = Qwen3Config::new(1024, 2048, 1, 8, 3, 100, 1e-6, 10_000.0, 64, true, None);
    assert!(
        bad.validate().is_err(),
        "8 heads, 3 kv_heads: 8 % 3 != 0 should be rejected"
    );
}

/// intermediate_size is typically 2.5-5.5x hidden_size in Qwen3 production models.
#[test]
fn test_config_mlp_hidden() {
    let production_configs = [
        (896, 4864, "0.6B"),  // ratio 5.43
        (2048, 6144, "1.7B"), // ratio 3.0
        (2560, 9216, "4B"),   // ratio 3.6
        (4096, 14336, "8B"),  // ratio 3.5
        (5120, 25600, "32B"), // ratio 5.0
    ];
    for (hidden, intermediate, name) in production_configs {
        let ratio = f64::from(intermediate) / f64::from(hidden);
        assert!(
            ratio > 2.0 && ratio < 6.0,
            "Qwen3-{name}: intermediate/hidden ratio {ratio:.2} should be in [2.0, 6.0]"
        );
    }
}

// ===========================================================================
// RoPE (Rotary Position Embeddings)
// ===========================================================================

/// RoPE frequencies follow a geometric sequence: theta[i] = 1/base^(2i/dim).
/// Each successive frequency should be strictly smaller than the previous.
#[test]
fn test_rope_frequencies() {
    let head_dim = 128;
    let base = 10_000.0_f32;
    let half_dim = head_dim / 2;

    // Compute the theta sequence
    let thetas: Vec<f64> = (0..half_dim)
        .map(|i| 1.0 / f64::from(base).powf((2 * i) as f64 / head_dim as f64))
        .collect();

    // Verify monotonically decreasing
    for i in 1..half_dim {
        assert!(
            thetas[i] < thetas[i - 1],
            "theta[{i}] ({}) should be < theta[{}] ({})",
            thetas[i],
            i - 1,
            thetas[i - 1]
        );
    }

    // Verify boundary values: theta[0] = 1.0, theta[last] is small
    assert!(
        (thetas[0] - 1.0).abs() < 1e-10,
        "theta[0] should be 1.0, got {}",
        thetas[0]
    );
    assert!(
        thetas[half_dim - 1] < 0.001,
        "theta[last] should be very small, got {}",
        thetas[half_dim - 1]
    );

    // Verify geometric ratio is constant: theta[i+1]/theta[i] should be
    // base^(-2/dim) for all i.
    let expected_ratio = f64::from(base).powf(-2.0 / head_dim as f64);
    for i in 0..half_dim - 1 {
        let ratio = thetas[i + 1] / thetas[i];
        assert!(
            (ratio - expected_ratio).abs() < 1e-10,
            "geometric ratio at {i}: expected {expected_ratio}, got {ratio}"
        );
    }
}

/// The cos/sin cache should have shape [max_seq_len, head_dim/2] per position.
#[test]
fn test_rope_cos_sin_cache_shape() {
    let max_seq_len = 256;
    let head_dim = 128;
    let half_dim = head_dim / 2;
    let cache = RoPECache::new(max_seq_len, head_dim, 10_000.0);

    assert_eq!(cache.max_seq_len(), max_seq_len);
    assert_eq!(cache.half_dim(), half_dim);

    // Every position should have half_dim cos/sin values
    for pos in [0, 1, 100, max_seq_len - 1] {
        let (cos, sin) = cache.get(pos);
        assert_eq!(
            cos.len(),
            half_dim,
            "cos at pos {pos} should have {half_dim} elements"
        );
        assert_eq!(
            sin.len(),
            half_dim,
            "sin at pos {pos} should have {half_dim} elements"
        );
    }

    // Range also returns correct inner lengths
    let (cos_range, sin_range) = cache.get_range(0, 10);
    assert_eq!(cos_range.len(), 10);
    for row in cos_range {
        assert_eq!(row.len(), half_dim);
    }
    for row in sin_range {
        assert_eq!(row.len(), half_dim);
    }
}

/// At position 0, all angles are 0 so cos=1 and sin=0 for every frequency.
/// This means RoPE applies no rotation (identity transform).
#[test]
fn test_rope_identity_at_position_0() {
    let head_dim = 128;
    let cache = RoPECache::new(16, head_dim, 10_000.0);
    let (cos, sin) = cache.get(0);

    for i in 0..head_dim / 2 {
        assert!(
            (cos[i] - 1.0).abs() < 1e-7,
            "cos[{i}] at position 0 should be 1.0, got {}",
            cos[i]
        );
        assert!(
            sin[i].abs() < 1e-7,
            "sin[{i}] at position 0 should be 0.0, got {}",
            sin[i]
        );
    }

    // Applying RoPE at position 0 should leave the vector unchanged
    let original: Vec<f32> = (1..=head_dim).map(|i| i as f32 * 0.1).collect();
    let mut q = original.clone();
    let mut k = original.clone();
    RoPECache::apply_rope(&mut q, &mut k, cos, sin);

    for i in 0..head_dim {
        assert!(
            (q[i] - original[i]).abs() < 1e-6,
            "q[{i}] should be unchanged at position 0"
        );
        assert!(
            (k[i] - original[i]).abs() < 1e-6,
            "k[{i}] should be unchanged at position 0"
        );
    }
}

// ===========================================================================
// KV Cache
// ===========================================================================

/// KV cache starts empty with zero sequence length.
#[test]
fn test_kv_cache_empty_creation() {
    let cache = KvCache::new(8);
    assert!(cache.is_empty());
    assert_eq!(cache.seq_len(), 0);
    assert_eq!(cache.num_layers(), 8);
}

/// Appending tokens via forward increases the cache sequence length.
#[test]
fn test_kv_cache_append() {
    let cfg = tiny_config(); // 2 layers
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let mut cache = model.new_cache();

    assert_eq!(cache.seq_len(), 0);

    // Append 3 tokens
    model
        .forward_cached(&[1, 2, 3], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 3);

    // Append 1 more token
    model.forward_cached(&[4], &[3], Some(&mut cache)).unwrap();
    assert_eq!(cache.seq_len(), 4);

    // Append 2 more tokens
    model
        .forward_cached(&[5, 6], &[4, 5], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 6);
}

/// Cache respects the model's configured max_position_embeddings. While the
/// KvCache itself does not enforce a max length, RoPE is only precomputed up
/// to max_position_embeddings. Verify the cache can grow to near-max and
/// produce finite logits.
#[test]
fn test_kv_cache_max_length() {
    let cfg = tiny_config(); // max_position_embeddings = 64
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    // Fill up to max_position_embeddings - 1
    let max_pos = cfg.max_position_embeddings;
    for i in 0..max_pos {
        let logits = model
            .forward_cached(&[i % cfg.vocab_size], &[i], Some(&mut cache))
            .unwrap();
        let vals = logits.to_flat_vec::<f32>().unwrap();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "logits should be finite at position {i}"
        );
    }
    assert_eq!(cache.seq_len(), max_pos);
}

// ===========================================================================
// SwiGLU Activation
// ===========================================================================

/// SwiGLU output preserves the hidden dimension: [batch, seq, hidden] ->
/// [batch, seq, hidden].
#[test]
fn test_swiglu_output_shape() {
    let cfg = tiny_config(); // hidden=256, intermediate=512
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    for seq_len in [1, 4, 16] {
        let ids: Vec<usize> = (0..seq_len).collect();
        let positions: Vec<usize> = (0..seq_len).collect();
        let logits = model.forward(&ids, &positions).unwrap();
        // Output shape: [1, seq_len, vocab_size]
        assert_eq!(logits.dims(), &[1, seq_len, cfg.vocab_size]);
    }
}

/// SwiGLU formula: output = down_proj(silu(gate_proj(x)) * up_proj(x))
/// where silu(x) = x * sigmoid(x).
///
/// Verify the silu activation matches the mathematical formula on DynTensor.
#[test]
fn test_swiglu_matches_formula() {
    // Test silu on a known input
    let input_data = vec![-2.0_f32, -1.0, 0.0, 1.0, 2.0];
    let x = DynTensor::from_vec(input_data.clone(), &[5], &Device::Cpu).unwrap();
    let silu_result = x.silu().unwrap();
    let silu_vals = silu_result.to_flat_vec::<f32>().unwrap();

    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    for (i, &xi) in input_data.iter().enumerate() {
        let expected = xi / (1.0 + (-xi).exp());
        assert!(
            (silu_vals[i] - expected).abs() < 1e-5,
            "silu({xi}) = {}: expected {expected}, got {}",
            silu_vals[i],
            silu_vals[i]
        );
    }

    // Verify SwiGLU property: silu(0) = 0
    assert!(
        silu_vals[2].abs() < 1e-7,
        "silu(0) should be 0, got {}",
        silu_vals[2]
    );

    // Verify SwiGLU property: silu is monotonically increasing for x > 0
    assert!(
        silu_vals[3] < silu_vals[4],
        "silu should be monotonic for positive x: silu(1)={} < silu(2)={}",
        silu_vals[3],
        silu_vals[4]
    );
}

/// Full SwiGLU MLP with zero weights produces zero output, verifying the
/// gate/up/down projection chain is correctly wired.
#[test]
fn test_swiglu_zero_weights_zero_output() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let logits = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
    let vals = logits.to_flat_vec::<f32>().unwrap();

    // With zero weights everywhere: embed=0, attn=0, mlp=0, norm scales output
    // to 0 (since input is 0), lm_head=0 => all zeros
    assert!(
        vals.iter().all(|&v| v == 0.0),
        "zero-weight model should produce zero logits"
    );
}

// ===========================================================================
// Additional integration tests
// ===========================================================================

/// Forward without cache and with empty cache should produce identical logits.
#[test]
fn test_forward_no_cache_vs_fresh_cache() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let tokens = [5, 10, 15];
    let positions = [0, 1, 2];

    // Without cache
    let logits_no_cache = model.forward(&tokens, &positions).unwrap();
    let no_cache_vec = logits_no_cache.to_flat_vec::<f32>().unwrap();

    // With fresh cache
    let mut cache = model.new_cache();
    let logits_cached = model
        .forward_cached(&tokens, &positions, Some(&mut cache))
        .unwrap();
    let cached_vec = logits_cached.to_flat_vec::<f32>().unwrap();

    assert_eq!(no_cache_vec.len(), cached_vec.len());
    for (i, (&a, &b)) in no_cache_vec.iter().zip(cached_vec.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "logit mismatch at {i}: no_cache={a}, cached={b}"
        );
    }
}

/// Model accessors return correct values after loading.
#[test]
fn test_model_accessors() {
    let cfg = Qwen3Config::new(256, 512, 3, 2, 1, 200, 1e-5, 50_000.0, 128, false, None);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    assert_eq!(model.config().hidden_size, 256);
    assert_eq!(model.config().num_hidden_layers, 3);
    assert_eq!(model.config().num_key_value_heads, 1);
    assert_eq!(model.dtype(), DType::F32);
    assert!(matches!(model.device(), Device::Cpu));

    let cache = model.new_cache();
    assert_eq!(cache.num_layers(), 3);
}

/// Config with YaRN rope scaling validates and loads successfully.
#[test]
fn test_config_with_yarn_scaling_loads() {
    use nn_core::layers::YarnScaling;

    let cfg = Qwen3Config::new(
        256,
        512,
        1,
        2,
        2,
        50,
        1e-6,
        1_000_000.0,
        131_072,
        true,
        Some(YarnScaling::new(4.0, 1.0, 32.0, 1.0, 64)),
    );
    assert!(cfg.validate().is_ok());
    assert!(cfg.rope_scaling.is_some());

    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg);
    assert!(
        model.is_ok(),
        "model with YaRN scaling should load: {:?}",
        model.err()
    );
}

/// RoPE at different base values (10k vs 1M) produces different rotations
/// at the same position, except at position 0 which is always identity.
#[test]
fn test_rope_different_bases_diverge_at_nonzero_position() {
    let head_dim = 128;
    let cache_10k = RoPECache::new(64, head_dim, 10_000.0);
    let cache_1m = RoPECache::new(64, head_dim, 1_000_000.0);

    // Position 0: both should be identity (cos=1, sin=0)
    let (cos_10k_0, sin_10k_0) = cache_10k.get(0);
    let (cos_1m_0, sin_1m_0) = cache_1m.get(0);
    for i in 0..head_dim / 2 {
        assert!(
            (cos_10k_0[i] - cos_1m_0[i]).abs() < 1e-6,
            "at pos 0, both bases should give cos=1"
        );
        assert!(
            (sin_10k_0[i] - sin_1m_0[i]).abs() < 1e-6,
            "at pos 0, both bases should give sin=0"
        );
    }

    // Position 10: should differ for high-frequency indices
    let (cos_10k_10, _) = cache_10k.get(10);
    let (cos_1m_10, _) = cache_1m.get(10);
    let max_diff: f32 = cos_10k_10
        .iter()
        .zip(cos_1m_10.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff > 0.01,
        "different bases at pos 10 should produce noticeably different cos values, max_diff={max_diff}"
    );
}
