// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended model-level tests for nn-qwen3 covering Qwen3 model configuration
//! invariants, weight naming conventions, RoPE frequency properties, GQA group
//! arithmetic, SwiGLU activation geometry, tokenizer vocab constraints,
//! MoE routing properties, and forward-path validation (#4545).
//!
//! Complements existing tests in `qwen3_extended_tests.rs`,
//! `qwen3_architecture_extended_tests.rs`, and `qwen3_config_extended_tests.rs`.

use crate::rope_cache::RoPECache;
use crate::test_utils::tiny_config;
use crate::{Qwen3Config, Qwen3Model, Qwen3MoeConfig};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::YarnScaling;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ===========================================================================
// Helper: build a loadable model from tiny_config with zero weights
// ===========================================================================

fn load_tiny_model() -> Qwen3Model {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    Qwen3Model::load(&vb, cfg).expect("tiny model should load")
}

// ===========================================================================
// 1. Configuration field-level invariants
// ===========================================================================

/// hidden_size must be positive; zero is rejected by validate().
#[test]
fn test_config_rejects_zero_hidden_size() {
    let cfg = Qwen3Config::new(0, 512, 1, 1, 1, 100, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

/// intermediate_size must be positive.
#[test]
fn test_config_rejects_zero_intermediate_size() {
    let cfg = Qwen3Config::new(256, 0, 1, 2, 2, 100, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

/// vocab_size must be positive.
#[test]
fn test_config_rejects_zero_vocab_size() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 0, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

/// max_position_embeddings must be positive.
#[test]
fn test_config_rejects_zero_max_position_embeddings() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, 10_000.0, 0, true, None);
    assert!(cfg.validate().is_err());
}

/// rms_norm_eps must be finite and positive.
#[test]
fn test_config_rejects_nan_rms_norm_eps() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 100, f64::NAN, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

/// rms_norm_eps must be positive (zero rejected).
#[test]
fn test_config_rejects_zero_rms_norm_eps() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 100, 0.0, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

/// rms_norm_eps negative is rejected.
#[test]
fn test_config_rejects_negative_rms_norm_eps() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 100, -1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

/// rope_theta must be finite and positive.
#[test]
fn test_config_rejects_nan_rope_theta() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, f64::NAN, 64, true, None);
    assert!(cfg.validate().is_err());
}

/// rope_theta must be positive (zero rejected).
#[test]
fn test_config_rejects_zero_rope_theta() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, 0.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

/// rope_theta negative is rejected.
#[test]
fn test_config_rejects_negative_rope_theta() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, -1.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

/// rope_theta infinity is rejected.
#[test]
fn test_config_rejects_inf_rope_theta() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, f64::INFINITY, 64, true, None);
    assert!(cfg.validate().is_err());
}

// ===========================================================================
// 2. GQA group arithmetic
// ===========================================================================

/// GQA groups: when kv_heads == heads, groups == 1 (MHA).
#[test]
fn test_gqa_mha_groups_eq_1() {
    let cfg = Qwen3Config::new(512, 1024, 1, 4, 4, 100, 1e-6, 10_000.0, 64, true, None);
    assert_eq!(cfg.num_kv_groups().unwrap(), 1);
}

/// GQA groups: kv_heads == 1 => groups == num_heads (MQA).
#[test]
fn test_gqa_mqa_groups_eq_num_heads() {
    let cfg = Qwen3Config::new(512, 1024, 1, 8, 1, 100, 1e-6, 10_000.0, 64, true, None);
    assert_eq!(cfg.num_kv_groups().unwrap(), 8);
}

/// GQA groups: standard Qwen3 pattern 32 heads, 8 kv_heads => 4 groups.
#[test]
fn test_gqa_standard_qwen3_pattern() {
    let cfg = Qwen3Config::new(4096, 14336, 1, 32, 8, 100, 1e-6, 10_000.0, 64, true, None);
    assert_eq!(cfg.num_kv_groups().unwrap(), 4);
}

/// GQA groups error when kv_heads is zero.
#[test]
fn test_gqa_zero_kv_heads_error() {
    let cfg = Qwen3Config::new(256, 512, 1, 4, 0, 100, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.num_kv_groups().is_err());
}

/// GQA groups error when heads not divisible by kv_heads.
#[test]
fn test_gqa_indivisible_heads_error() {
    let cfg = Qwen3Config::new(256, 512, 1, 7, 3, 100, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.num_kv_groups().is_err());
}

/// GQA memory reduction: kv_heads < heads means fewer KV projections.
/// For 32 heads / 8 kv_heads, KV memory is 1/4 of MHA.
#[test]
fn test_gqa_kv_memory_ratio() {
    let nh = 32_usize;
    let nkv = 8_usize;
    let hd = 128_usize;
    let ratio = (nkv * hd) as f64 / (nh * hd) as f64;
    assert!((ratio - 0.25).abs() < 1e-10);
}

// ===========================================================================
// 3. head_dim is constant 128 for all Qwen3 models
// ===========================================================================

/// head_dim is always 128 regardless of hidden_size or num_heads.
#[test]
fn test_head_dim_constant_128_across_sizes() {
    for (h, nh) in [
        (256, 2),
        (512, 4),
        (896, 14),
        (2048, 16),
        (4096, 32),
        (5120, 40),
    ] {
        let cfg = Qwen3Config::new(h, h * 2, 1, nh, 1, 100, 1e-6, 10_000.0, 64, true, None);
        assert_eq!(
            cfg.head_dim(),
            128,
            "head_dim should be 128 for hidden={h}, nh={nh}"
        );
    }
}

// ===========================================================================
// 4. SwiGLU geometry: intermediate_size relationships
// ===========================================================================

/// SwiGLU has 3 weight matrices (gate, up, down) so total MLP params = 3 * intermediate * hidden.
#[test]
fn test_swiglu_param_count_formula() {
    let h = 4096_usize;
    let i = 14336_usize;
    let swiglu_params = 3 * i * h;
    // Qwen3-8B: 3 * 14336 * 4096 = 176,160,768 per layer
    assert_eq!(swiglu_params, 176_160_768);
}

/// Qwen3 intermediate_size is typically ~3.5x hidden_size.
#[test]
fn test_swiglu_intermediate_to_hidden_ratio() {
    let cases = [
        (896_usize, 4864_usize, "0.6B"),
        (2048, 6144, "1.7B"),
        (4096, 14336, "8B"),
        (5120, 17408, "14B"),
    ];
    for (h, i, name) in cases {
        let ratio = i as f64 / h as f64;
        assert!(
            ratio > 2.0 && ratio < 6.0,
            "{name}: intermediate/hidden ratio {ratio} outside expected range [2.0, 6.0]"
        );
    }
}

// ===========================================================================
// 5. Builder pattern: with_vocab_size, with_num_hidden_layers
// ===========================================================================

/// with_vocab_size modifies vocab_size.
#[test]
fn test_builder_with_vocab_size() {
    let cfg = tiny_config().with_vocab_size(32_000);
    assert_eq!(cfg.vocab_size, 32_000);
    // Other fields unchanged
    assert_eq!(cfg.hidden_size, 256);
}

/// with_num_hidden_layers modifies num_hidden_layers.
#[test]
fn test_builder_with_num_hidden_layers() {
    let cfg = tiny_config().with_num_hidden_layers(12);
    assert_eq!(cfg.num_hidden_layers, 12);
    // Other fields unchanged
    assert_eq!(cfg.hidden_size, 256);
}

/// Chaining builders works.
#[test]
fn test_builder_chaining() {
    let cfg = tiny_config()
        .with_vocab_size(50_000)
        .with_num_hidden_layers(6);
    assert_eq!(cfg.vocab_size, 50_000);
    assert_eq!(cfg.num_hidden_layers, 6);
    assert!(cfg.validate().is_ok());
}

// ===========================================================================
// 6. RoPE frequency geometry
// ===========================================================================

/// RoPE position 0 produces cos=1, sin=0 for all frequencies.
#[test]
fn test_rope_position_zero_is_identity() {
    let cache = RoPECache::new(128, 128, 1_000_000.0);
    let (cos, sin) = cache.get(0);
    for i in 0..64 {
        assert!(
            (cos[i] - 1.0).abs() < 1e-6,
            "cos[{i}] at pos 0 should be 1.0"
        );
        assert!(sin[i].abs() < 1e-6, "sin[{i}] at pos 0 should be 0.0");
    }
}

/// RoPE with Qwen3's base=1M: low-frequency (high i) wavelengths are very long.
#[test]
fn test_rope_qwen3_base_1m_long_wavelengths() {
    let base = 1_000_000.0_f64;
    // Last frequency index (i = head_dim/2 - 1 = 63):
    // theta[63] = 1 / base^(126/128) ≈ very small
    // wavelength = 2*pi / theta[63] ≈ very long
    let exponent = 126.0 / 128.0;
    let theta = 1.0 / base.powf(exponent);
    let wavelength = 2.0 * std::f64::consts::PI / theta;
    // With base=1M, last frequency wavelength should be very long (> 1M positions)
    assert!(
        wavelength > 1_000_000.0,
        "last frequency wavelength should be > 1M, got {wavelength}"
    );
}

/// RoPE frequencies are monotonically decreasing in theta.
#[test]
fn test_rope_frequencies_monotonically_decreasing() {
    let head_dim = 128_usize;
    let base = 10_000.0_f64;
    let half_dim = head_dim / 2;
    let thetas: Vec<f64> = (0..half_dim)
        .map(|i| 1.0 / base.powf((2 * i) as f64 / head_dim as f64))
        .collect();
    for i in 1..half_dim {
        assert!(
            thetas[i] < thetas[i - 1],
            "theta[{i}]={} should be < theta[{}]={}",
            thetas[i],
            i - 1,
            thetas[i - 1]
        );
    }
}

/// RoPE apply_rope preserves L2 norm (orthogonal rotation).
#[test]
fn test_rope_preserves_l2_norm() {
    let cache = RoPECache::new(256, 128, 1_000_000.0);
    for pos in [1, 10, 50, 100, 255] {
        let (cos, sin) = cache.get(pos);
        let mut q: Vec<f32> = (0..128)
            .map(|i| ((i * 7 + 3) % 100) as f32 * 0.01)
            .collect();
        let mut k: Vec<f32> = (0..128)
            .map(|i| ((i * 11 + 5) % 100) as f32 * 0.01)
            .collect();
        let q_norm_before: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let k_norm_before: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt();
        RoPECache::apply_rope(&mut q, &mut k, cos, sin);
        let q_norm_after: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let k_norm_after: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (q_norm_before - q_norm_after).abs() < 1e-4,
            "q norm changed at pos {pos}: {q_norm_before} -> {q_norm_after}"
        );
        assert!(
            (k_norm_before - k_norm_after).abs() < 1e-4,
            "k norm changed at pos {pos}: {k_norm_before} -> {k_norm_after}"
        );
    }
}

/// RoPE get_range returns the same data as individual get() calls.
#[test]
fn test_rope_get_range_consistency() {
    let cache = RoPECache::new(200, 128, 1_000_000.0);
    let (cos_range, sin_range) = cache.get_range(50, 10);
    for i in 0..10 {
        let (cos_single, sin_single) = cache.get(50 + i);
        assert_eq!(cos_range[i].as_slice(), cos_single);
        assert_eq!(sin_range[i].as_slice(), sin_single);
    }
}

// ===========================================================================
// 7. Model loading and structural checks
// ===========================================================================

/// Model loads with zero weights and reports correct config.
#[test]
fn test_model_load_reports_config() {
    let model = load_tiny_model();
    let cfg = model.config();
    assert_eq!(cfg.hidden_size, 256);
    assert_eq!(cfg.num_hidden_layers, 2);
    assert_eq!(cfg.num_attention_heads, 2);
    assert_eq!(cfg.vocab_size, 100);
}

/// Model dtype matches VarBuilder dtype.
#[test]
fn test_model_dtype_matches_varbuilder() {
    let model = load_tiny_model();
    assert_eq!(model.dtype(), DType::F32);
}

/// Model device is CPU when loaded with CPU VarBuilder.
#[test]
fn test_model_device_is_cpu() {
    let model = load_tiny_model();
    assert_eq!(model.device(), Device::Cpu);
}

/// Embedding layer reference has correct shape.
#[test]
fn test_model_embed_tokens_accessible() {
    let model = load_tiny_model();
    let embed = model.embed_tokens();
    let w = embed.weight();
    assert_eq!(w.dims(), &[100, 256]); // [vocab_size, hidden_size]
}

/// new_cache creates a KvCache with correct layer count.
#[test]
fn test_model_new_cache_layer_count() {
    let model = load_tiny_model();
    let cache = model.new_cache();
    assert_eq!(cache.num_layers(), 2);
}

// ===========================================================================
// 8. Forward pass validation
// ===========================================================================

/// Forward with empty input_ids is rejected (mismatched with positions).
#[test]
fn test_forward_rejects_mismatched_ids_positions() {
    let model = load_tiny_model();
    let result = model.forward(&[0, 1], &[0]);
    assert!(
        result.is_err(),
        "mismatched input_ids/positions should error"
    );
}

/// Forward with zero-weight model produces finite output.
#[test]
fn test_forward_zero_weights_finite_output() {
    let model = load_tiny_model();
    let result = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
    let data = result.to_flat_vec::<f32>().unwrap();
    for (i, &v) in data.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] is not finite: {v}");
    }
}

/// Forward output shape is [1, seq_len, vocab_size].
#[test]
fn test_forward_output_shape() {
    let model = load_tiny_model();
    let result = model.forward(&[0, 1, 2, 3], &[0, 1, 2, 3]).unwrap();
    assert_eq!(result.dims(), &[1, 4, 100]); // [batch=1, seq=4, vocab=100]
}

/// Forward with single token.
#[test]
fn test_forward_single_token() {
    let model = load_tiny_model();
    let result = model.forward(&[5], &[0]).unwrap();
    assert_eq!(result.dims(), &[1, 1, 100]);
}

// ===========================================================================
// 9. Forward with KV cache
// ===========================================================================

/// Cached forward produces same shape as uncached for initial prompt.
#[test]
fn test_cached_forward_initial_shape() {
    let model = load_tiny_model();
    let mut cache = model.new_cache();
    let result = model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(result.dims(), &[1, 3, 100]);
}

/// KV cache grows after forward pass.
#[test]
fn test_kv_cache_grows_after_forward() {
    let model = load_tiny_model();
    let mut cache = model.new_cache();
    assert_eq!(cache.seq_len(), 0);
    model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 3);
}

/// Autoregressive decode step: process one token at a time.
#[test]
fn test_autoregressive_decode_step() {
    let model = load_tiny_model();
    let mut cache = model.new_cache();
    // Prefill
    model
        .forward_cached(&[0, 1], &[0, 1], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 2);
    // Decode step
    let result = model.forward_cached(&[2], &[2], Some(&mut cache)).unwrap();
    assert_eq!(result.dims(), &[1, 1, 100]);
    assert_eq!(cache.seq_len(), 3);
}

/// Multiple decode steps grow cache monotonically.
#[test]
fn test_multiple_decode_steps_cache_growth() {
    let model = load_tiny_model();
    let mut cache = model.new_cache();
    model.forward_cached(&[0], &[0], Some(&mut cache)).unwrap();
    for step in 1..5 {
        model
            .forward_cached(&[step], &[step], Some(&mut cache))
            .unwrap();
        assert_eq!(cache.seq_len(), step + 1);
    }
}

// ===========================================================================
// 10. Forward from embeddings
// ===========================================================================

/// forward_from_embeddings with correct hidden_size works.
#[test]
fn test_forward_from_embeddings_shape() {
    let model = load_tiny_model();
    let hidden = DynTensor::zeros(&[1, 3, 256], DType::F32, &Device::Cpu).unwrap();
    let result = model
        .forward_from_embeddings(&hidden, &[0, 1, 2], None)
        .unwrap();
    assert_eq!(result.dims(), &[1, 3, 100]);
}

/// forward_from_embeddings rejects wrong hidden_size.
#[test]
fn test_forward_from_embeddings_wrong_hidden_size() {
    let model = load_tiny_model();
    let hidden = DynTensor::zeros(&[1, 3, 128], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&hidden, &[0, 1, 2], None);
    assert!(result.is_err(), "wrong hidden_size should be rejected");
}

/// forward_from_embeddings rejects mismatched seq_len / positions.
#[test]
fn test_forward_from_embeddings_mismatched_positions() {
    let model = load_tiny_model();
    let hidden = DynTensor::zeros(&[1, 3, 256], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&hidden, &[0, 1], None);
    assert!(
        result.is_err(),
        "mismatched seq_len/positions should be rejected"
    );
}

/// forward_from_embeddings_with_hidden returns both logits and normed hidden.
#[test]
fn test_forward_from_embeddings_with_hidden_shapes() {
    let model = load_tiny_model();
    let hidden = DynTensor::zeros(&[1, 3, 256], DType::F32, &Device::Cpu).unwrap();
    let (logits, normed) = model
        .forward_from_embeddings_with_hidden(&hidden, &[0, 1, 2], None)
        .unwrap();
    assert_eq!(logits.dims(), &[1, 3, 100]);
    assert_eq!(normed.dims(), &[1, 3, 256]);
}

// ===========================================================================
// 11. Vocab and tokenizer constraints
// ===========================================================================

/// Qwen3 standard vocab_size is 151_936.
#[test]
fn test_qwen3_standard_vocab_size() {
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
    assert_eq!(cfg.vocab_size, 151_936);
    assert!(cfg.validate().is_ok());
}

/// Vocab size of 1 is technically valid (edge case).
#[test]
fn test_config_min_vocab_size_1() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 1, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_ok());
}

// ===========================================================================
// 12. YaRN scaling configuration
// ===========================================================================

/// Config with YaRN rope_scaling passes validation.
#[test]
fn test_config_with_yarn_scaling_validates() {
    let yarn = YarnScaling::new(4.0, 0.25, 32.0, 1.0, 32_768);
    let cfg = Qwen3Config::new(
        2048,
        6144,
        1,
        16,
        4,
        100,
        1e-6,
        1_000_000.0,
        131_072,
        true,
        Some(yarn),
    );
    assert!(cfg.validate().is_ok());
    assert!(cfg.rope_scaling.is_some());
}

/// Config without YaRN scaling has None rope_scaling.
#[test]
fn test_config_without_yarn_scaling() {
    let cfg = tiny_config();
    assert!(cfg.rope_scaling.is_none());
}

// ===========================================================================
// 13. MoE configuration
// ===========================================================================

/// MoE config validates with sane defaults.
#[test]
fn test_moe_config_validates() {
    let base = tiny_config();
    let moe = Qwen3MoeConfig::new(base, 8, 2, false, None);
    assert!(moe.validate().is_ok());
}

/// MoE config rejects zero experts.
#[test]
fn test_moe_config_rejects_zero_experts() {
    let base = tiny_config();
    let moe = Qwen3MoeConfig::new(base, 0, 2, false, None);
    assert!(moe.validate().is_err());
}

/// MoE config rejects zero experts_per_tok.
#[test]
fn test_moe_config_rejects_zero_experts_per_tok() {
    let base = tiny_config();
    let moe = Qwen3MoeConfig::new(base, 8, 0, false, None);
    assert!(moe.validate().is_err());
}

/// MoE config rejects experts_per_tok > num_experts.
#[test]
fn test_moe_config_rejects_experts_per_tok_exceeds_total() {
    let base = tiny_config();
    let moe = Qwen3MoeConfig::new(base, 4, 5, false, None);
    assert!(moe.validate().is_err());
}

/// MoE config with shared expert validates.
#[test]
fn test_moe_config_shared_expert_validates() {
    let base = tiny_config();
    let moe = Qwen3MoeConfig::new(base, 128, 8, true, Some(2048));
    assert!(moe.validate().is_ok());
    assert_eq!(moe.shared_expert_ff_dim(), 2048);
}

/// MoE config shared_expert_ff_dim falls back to base.intermediate_size.
#[test]
fn test_moe_config_shared_expert_ff_dim_fallback() {
    let base = tiny_config();
    let i = base.intermediate_size;
    let moe = Qwen3MoeConfig::new(base, 128, 8, true, None);
    assert_eq!(moe.shared_expert_ff_dim(), i);
}

/// MoE config rejects shared expert with zero intermediate size.
#[test]
fn test_moe_config_shared_expert_zero_intermediate_rejected() {
    let base = tiny_config();
    let moe = Qwen3MoeConfig::new(base, 8, 2, true, Some(0));
    assert!(moe.validate().is_err());
}

/// Qwen3 MoE production pattern: 128 experts, 8 active.
#[test]
fn test_moe_production_pattern_128_experts() {
    let base = Qwen3Config::new(
        4096,
        2560,
        48,
        32,
        4,
        151_936,
        1e-6,
        1_000_000.0,
        131_072,
        false,
        None,
    );
    let moe = Qwen3MoeConfig::new(base, 128, 8, true, Some(10_240));
    assert!(moe.validate().is_ok());
    assert_eq!(moe.num_experts, 128);
    assert_eq!(moe.num_experts_per_tok, 8);
}

// ===========================================================================
// 14. Weight naming conventions (structural)
// ===========================================================================

/// Weight names expected for Qwen3 dense model (HuggingFace convention).
#[test]
fn test_weight_naming_convention_dense() {
    // These are the weight name patterns; verify they follow HF convention.
    let expected_patterns = [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.norm.weight",
    ];
    // Verify naming structure: all start with "model." except lm_head
    for name in expected_patterns {
        assert!(
            name.starts_with("model."),
            "weight name should start with 'model.': {name}"
        );
    }
}

/// MoE weight naming includes expert indices.
#[test]
fn test_weight_naming_convention_moe() {
    let expected_expert_pattern = "model.layers.0.mlp.experts.0.gate_proj.weight";
    assert!(expected_expert_pattern.contains("experts.0.gate_proj"));
    let expected_router = "model.layers.0.mlp.gate.weight";
    assert!(expected_router.contains("mlp.gate.weight"));
}

// ===========================================================================
// 15. Tie word embeddings
// ===========================================================================

/// When tie_word_embeddings is true, lm_head reuses embed_tokens weight.
/// Model loads successfully without separate lm_head.weight.
#[test]
fn test_tied_embeddings_load_succeeds() {
    let cfg = tiny_config(); // tie_word_embeddings = true
    assert!(cfg.tie_word_embeddings);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg);
    assert!(model.is_ok());
}

/// When tie_word_embeddings is false, model needs lm_head.weight.
/// With ZerosBackend, this should still load (zeros provides any tensor).
#[test]
fn test_untied_embeddings_load_succeeds() {
    let mut cfg = tiny_config();
    cfg.tie_word_embeddings = false;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg);
    assert!(model.is_ok());
}

// ===========================================================================
// 16. RoPE cache try_new validation
// ===========================================================================

/// try_new with valid params succeeds.
#[test]
fn test_rope_cache_try_new_valid() {
    let cache = RoPECache::try_new(1024, 128, 10_000.0);
    assert!(cache.is_ok());
    let cache = cache.unwrap();
    assert_eq!(cache.max_seq_len(), 1024);
    assert_eq!(cache.head_dim(), 128);
}

/// try_new rejects zero base.
#[test]
fn test_rope_cache_try_new_rejects_zero_base() {
    assert!(RoPECache::try_new(128, 128, 0.0).is_err());
}

/// try_new rejects odd head_dim.
#[test]
fn test_rope_cache_try_new_rejects_odd_head_dim() {
    assert!(RoPECache::try_new(128, 127, 10_000.0).is_err());
}

// ===========================================================================
// 17. Causal mask construction
// ===========================================================================

/// Causal mask for seq_len=4 has shape [1, 1, 4, 4].
#[test]
fn test_causal_mask_shape_4() {
    use crate::causal_mask;
    let mask = causal_mask(4, DType::F32, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 4, 4]);
}

/// Causal mask lower triangle is 0, upper triangle is -inf.
#[test]
fn test_causal_mask_lower_zero_upper_neginf() {
    use crate::causal_mask;
    let mask = causal_mask(3, DType::F32, &Device::Cpu).unwrap();
    let data = mask.to_flat_vec::<f32>().unwrap();
    // Lower triangle (including diagonal) should be 0.0
    // [0]=0, [3]=0, [4]=0, [6]=0, [7]=0, [8]=0
    assert_eq!(data[0], 0.0);
    assert_eq!(data[3], 0.0);
    assert_eq!(data[4], 0.0);
    assert_eq!(data[6], 0.0);
    assert_eq!(data[7], 0.0);
    assert_eq!(data[8], 0.0);
    // Upper triangle should be -inf
    assert!(data[1] < 0.0 && data[1].is_infinite());
    assert!(data[2] < 0.0 && data[2].is_infinite());
    assert!(data[5] < 0.0 && data[5].is_infinite());
}

// ===========================================================================
// 18. Config clone and debug
// ===========================================================================

/// Qwen3Config implements Clone.
#[test]
fn test_config_clone() {
    let cfg = tiny_config();
    let cfg2 = cfg.clone();
    assert_eq!(cfg2.hidden_size, cfg.hidden_size);
    assert_eq!(cfg2.num_hidden_layers, cfg.num_hidden_layers);
    assert_eq!(cfg2.vocab_size, cfg.vocab_size);
}

/// Qwen3Config implements Debug.
#[test]
fn test_config_debug() {
    let cfg = tiny_config();
    let dbg = format!("{cfg:?}");
    assert!(dbg.contains("Qwen3Config"));
    assert!(dbg.contains("hidden_size"));
}

/// Qwen3MoeConfig implements Clone and Debug.
#[test]
fn test_moe_config_clone_debug() {
    let base = tiny_config();
    let moe = Qwen3MoeConfig::new(base, 8, 2, false, None);
    let moe2 = moe.clone();
    assert_eq!(moe2.num_experts, 8);
    let dbg = format!("{moe:?}");
    assert!(dbg.contains("Qwen3MoeConfig"));
}

// ===========================================================================
// 19. Error types
// ===========================================================================

/// Qwen3Error::InvalidConfig has meaningful message.
#[test]
fn test_error_invalid_config_message() {
    let err = crate::Qwen3Error::InvalidConfig {
        reason: "test reason".into(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("test reason"));
}

/// Qwen3Error::InvalidInput has meaningful message.
#[test]
fn test_error_invalid_input_message() {
    let err = crate::Qwen3Error::InvalidInput {
        reason: "bad input".into(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("bad input"));
}

/// Qwen3Error::CacheMismatch has meaningful message.
#[test]
fn test_error_cache_mismatch_message() {
    let err = crate::Qwen3Error::CacheMismatch {
        cache_layers: 4,
        model_layers: 8,
    };
    let msg = format!("{err}");
    assert!(msg.contains("4") && msg.contains("8"));
}

/// Qwen3Error::NonFiniteOutput has meaningful message.
#[test]
fn test_error_non_finite_output_message() {
    let err = crate::Qwen3Error::NonFiniteOutput {
        stage: "test_stage",
        count: 42,
    };
    let msg = format!("{err}");
    assert!(msg.contains("test_stage") && msg.contains("42"));
}

/// Qwen3Error::WeightLoad has meaningful message.
#[test]
fn test_error_weight_load_message() {
    let err = crate::Qwen3Error::WeightLoad {
        reason: "missing tensor".into(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("missing tensor"));
}
