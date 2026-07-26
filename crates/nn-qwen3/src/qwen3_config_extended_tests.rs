// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Qwen3 tests covering model sizing, RoPE wavelength analysis, GQA
//! memory reduction, SwiGLU geometry, builder pattern, forward validation errors,
//! causal mask construction, KV cache mismatch detection, MoE production configs,
//! and minimum viable config edge cases (#4186).
//!
//! Complements existing tests in `config_tests.rs`, `qwen3_extended_tests.rs`,
//! `qwen3_architecture_extended_tests.rs`, `rope_tests.rs`, and
//! `attention_tests.rs`.

use crate::rope_cache::RoPECache;
use crate::test_utils::tiny_config;
use crate::{Qwen3Config, Qwen3Model, Qwen3MoeConfig};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::kv_cache::KvCache;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ===========================================================================
// 1. Model parameter counting for all Qwen3 dense variants
// ===========================================================================

/// Compute the total parameter count for a Qwen3 dense model.
fn dense_param_count(cfg: &Qwen3Config) -> usize {
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let hd = 128; // Qwen3 constant head_dim
    let v = cfg.vocab_size;
    let n = cfg.num_hidden_layers;

    // Embedding: [vocab, hidden]
    let embed = v * h;
    // Per-layer attention: q[nh*hd, h] + k[nkv*hd, h] + v[nkv*hd, h] + o[h, nh*hd]
    let attn_proj = (nh * hd * h) + 2 * (nkv * hd * h) + (h * nh * hd);
    // Per-layer QK-Norm: 2 * [hd]
    let qk_norm = 2 * hd;
    // Per-layer RMSNorm: 2 * [h] (input + post_attn)
    let layer_norm = 2 * h;
    // Per-layer SwiGLU MLP: gate[i, h] + up[i, h] + down[h, i]
    let mlp = 3 * i * h;
    // Final RMSNorm: [h]
    let final_norm = h;
    // lm_head: [vocab, hidden] (or tied)
    let lm_head = if cfg.tie_word_embeddings { 0 } else { v * h };

    embed + n * (attn_proj + qk_norm + layer_norm + mlp) + final_norm + lm_head
}

/// Qwen3-0.6B total parameter count is approximately 600M.
#[test]
fn test_param_count_0_6b() {
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
    let count = dense_param_count(&cfg);
    // 0.6B model: ~580-630M params. Tied embeddings.
    assert!(
        count > 500_000_000 && count < 700_000_000,
        "Qwen3-0.6B should have ~600M params, got {count}"
    );
}

/// Qwen3-1.7B total parameter count.
#[test]
fn test_param_count_1_7b() {
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
    let count = dense_param_count(&cfg);
    assert!(
        count > 1_500_000_000 && count < 2_000_000_000,
        "Qwen3-1.7B should have ~1.7B params, got {count}"
    );
}

/// Qwen3-4B total parameter count.
#[test]
fn test_param_count_4b() {
    let cfg = Qwen3Config::new(
        2560,
        9216,
        36,
        32,
        8,
        151_936,
        1e-6,
        1_000_000.0,
        131_072,
        false,
        None,
    );
    let count = dense_param_count(&cfg);
    assert!(
        count > 3_500_000_000 && count < 4_500_000_000,
        "Qwen3-4B should have ~4B params, got {count}"
    );
}

/// Qwen3-8B total parameter count.
#[test]
fn test_param_count_8b() {
    let cfg = Qwen3Config::new(
        4096,
        14336,
        36,
        32,
        8,
        151_936,
        1e-6,
        1_000_000.0,
        131_072,
        false,
        None,
    );
    let count = dense_param_count(&cfg);
    assert!(
        count > 7_000_000_000 && count < 10_000_000_000,
        "Qwen3-8B should have ~8-9B params, got {count}"
    );
}

/// Qwen3-14B total parameter count.
#[test]
fn test_param_count_14b() {
    let cfg = Qwen3Config::new(
        5120,
        17408,
        40,
        40,
        8,
        151_936,
        1e-6,
        1_000_000.0,
        131_072,
        false,
        None,
    );
    let count = dense_param_count(&cfg);
    assert!(
        count > 13_000_000_000 && count < 16_000_000_000,
        "Qwen3-14B should have ~14B params, got {count}"
    );
}

/// Qwen3-32B total parameter count.
#[test]
fn test_param_count_32b() {
    let cfg = Qwen3Config::new(
        5120,
        25600,
        64,
        40,
        8,
        151_936,
        1e-6,
        1_000_000.0,
        131_072,
        false,
        None,
    );
    let count = dense_param_count(&cfg);
    assert!(
        count > 30_000_000_000 && count < 35_000_000_000,
        "Qwen3-32B should have ~32B params, got {count}"
    );
}

// ===========================================================================
// 2. RoPE wavelength analysis
// ===========================================================================

/// The longest wavelength in the RoPE spectrum is 2*pi / theta[last].
/// With base=1M, head_dim=128: theta[63] = 1/1M^(126/128), wavelength > 1M positions.
/// This enables 128K+ context without YaRN.
#[test]
fn test_rope_longest_wavelength_1m_base() {
    let base = 1_000_000.0_f64;
    let head_dim = 128;
    let last_idx = head_dim / 2 - 1; // 63
    let theta_last = 1.0 / base.powf(f64::from(2 * last_idx) / f64::from(head_dim));
    let wavelength = 2.0 * std::f64::consts::PI / theta_last;

    // With base=1M, the longest wavelength should be enormous (> 10^5 positions)
    assert!(
        wavelength > 100_000.0,
        "longest wavelength with base=1M should be > 100K positions, got {wavelength:.0}"
    );
}

/// The shortest wavelength is 2*pi / theta[0] = 2*pi (theta[0] = 1.0).
/// This is constant regardless of base.
#[test]
fn test_rope_shortest_wavelength_is_2pi() {
    let theta_0 = 1.0_f64; // 1 / base^(0/dim) = 1.0
    let wavelength = 2.0 * std::f64::consts::PI / theta_0;
    assert!(
        (wavelength - 2.0 * std::f64::consts::PI).abs() < 1e-10,
        "shortest wavelength should be 2*pi, got {wavelength}"
    );
}

/// RoPE frequency at index i follows: freq[i] = 1 / base^(2i/dim).
/// Verify this produces a logarithmically-spaced frequency spectrum.
#[test]
fn test_rope_log_spaced_frequencies() {
    let base = 10_000.0_f64;
    let head_dim = 128;
    let half = head_dim / 2;

    let log_freqs: Vec<f64> = (0..half)
        .map(|i| {
            let freq = 1.0 / base.powf((2 * i) as f64 / head_dim as f64);
            freq.ln()
        })
        .collect();

    // ln(freq[i]) = -(2i/dim) * ln(base)
    // The log-space spacing should be constant: delta = -2/dim * ln(base)
    let expected_delta = -2.0 / head_dim as f64 * base.ln();
    for i in 0..half - 1 {
        let delta = log_freqs[i + 1] - log_freqs[i];
        assert!(
            (delta - expected_delta).abs() < 1e-10,
            "log-space delta at {i}: expected {expected_delta}, got {delta}"
        );
    }
}

/// Higher base = slower frequency decay = longer effective context.
/// base=1M should have 100x smaller frequencies than base=10K at the same index.
#[test]
fn test_rope_base_comparison_frequency_ratio() {
    let head_dim = 128;
    // At index 32 (midpoint): freq = 1/sqrt(base)
    let freq_10k = 1.0 / (10_000.0_f64).sqrt();
    let freq_1m = 1.0 / (1_000_000.0_f64).sqrt();
    let ratio = freq_10k / freq_1m;
    assert!(
        (ratio - 10.0).abs() < 1e-10,
        "base=10K should have 10x higher mid-band freq than base=1M, got ratio {ratio}"
    );

    // At index 0: both are 1.0 (no difference)
    let freq_10k_0 = 1.0 / (10_000.0_f64).powf(0.0 / f64::from(head_dim));
    let freq_1m_0 = 1.0 / (1_000_000.0_f64).powf(0.0 / f64::from(head_dim));
    assert!(
        (freq_10k_0 - freq_1m_0).abs() < 1e-10,
        "at index 0, both bases should give freq=1.0"
    );
}

// ===========================================================================
// 3. GQA memory savings calculations
// ===========================================================================

/// GQA KV cache memory is proportional to num_key_value_heads, not num_attention_heads.
/// Verify the memory reduction factor matches the GQA group count.
#[test]
fn test_gqa_kv_memory_reduction_factor() {
    // For each production config, KV cache memory is reduced by factor of
    // num_attention_heads / num_key_value_heads compared to MHA.
    let configs = [
        ("0.6B", 14, 2, 7),
        ("1.7B", 16, 4, 4),
        ("4B", 32, 8, 4),
        ("8B", 32, 8, 4),
        ("14B", 40, 8, 5),
        ("32B", 40, 8, 5),
    ];
    for (name, nh, nkv, expected_factor) in configs {
        let kv_reduction = nh / nkv;
        assert_eq!(
            kv_reduction, expected_factor,
            "{name}: GQA memory reduction should be {expected_factor}x, got {kv_reduction}x"
        );
    }
}

/// KV cache size per token per layer = 2 * num_kv_heads * head_dim * sizeof(f32).
/// Verify actual values for production configs.
#[test]
fn test_kv_cache_bytes_per_token_per_layer() {
    let head_dim = 128;
    let sizeof_f32 = 4_usize;

    let configs = [
        ("0.6B", 2_usize, 2 * 2 * head_dim * sizeof_f32),
        ("1.7B", 4, 2 * 4 * head_dim * sizeof_f32),
        ("8B", 8, 2 * 8 * head_dim * sizeof_f32),
    ];
    for (name, nkv, expected_bytes) in configs {
        let bytes = 2 * nkv * head_dim * sizeof_f32;
        assert_eq!(
            bytes, expected_bytes,
            "{name}: KV bytes/token/layer should be {expected_bytes}, got {bytes}"
        );
    }
}

/// Total KV cache size for 4K context on Qwen3-8B:
/// 36 layers * 2 (K+V) * 8 kv_heads * 128 head_dim * 4 bytes * 4096 tokens = 1152 MiB.
#[test]
fn test_kv_cache_total_size_8b_4k_context() {
    let layers = 36_usize;
    let nkv = 8_usize;
    let head_dim = 128_usize;
    let seq_len = 4096_usize;
    let bytes_per_elem = 4_usize;

    let total_bytes = layers * 2 * nkv * head_dim * bytes_per_elem * seq_len;
    let mib = total_bytes as f64 / (1024.0 * 1024.0);

    assert!(
        mib > 1100.0 && mib < 1200.0,
        "Qwen3-8B 4K context KV cache should be ~1152 MiB, got {mib:.0} MiB"
    );
}

// ===========================================================================
// 4. SwiGLU intermediate size formula
// ===========================================================================

/// SwiGLU "8/3" rule: optimal intermediate_size ~= (8/3) * hidden_size,
/// typically rounded to a multiple of 256 for hardware alignment.
/// Verify production configs follow this pattern approximately.
#[test]
fn test_swiglu_intermediate_size_follows_8_3_rule() {
    let configs = [
        ("0.6B", 896, 4864),
        ("1.7B", 2048, 6144),
        ("4B", 2560, 9216),
        ("8B", 4096, 14336),
        ("14B", 5120, 17408),
        ("32B", 5120, 25600),
    ];
    for (name, hidden, intermediate) in configs {
        let ratio = f64::from(intermediate) / f64::from(hidden);
        // The 8/3 rule gives ratio ~ 2.67. Production models range from ~2.67 to ~5.43
        // because they also optimize for tensor core alignment and model capacity.
        // All should be multiples of 128.
        assert_eq!(
            intermediate % 128,
            0,
            "{name}: intermediate_size {intermediate} should be 128-aligned"
        );
        assert!(
            ratio > 2.5,
            "{name}: intermediate/hidden ratio {ratio:.2} should be > 2.5 (SwiGLU minimum)"
        );
    }
}

/// SwiGLU effective parameter count is 3 * hidden * intermediate per layer
/// (gate + up + down projections). This is ~12 * hidden^2 at 8/3 ratio.
#[test]
fn test_swiglu_params_per_layer() {
    let hidden = 4096_usize;
    let intermediate = 14336_usize; // Qwen3-8B
    let mlp_params = 3 * hidden * intermediate;
    // 3 * 4096 * 14336 = 176,160,768 ~ 176M params per MLP layer
    assert_eq!(mlp_params, 176_160_768);

    // Compare to standard FFN: 2 * hidden * 4*hidden = 8 * hidden^2
    let standard_ffn = 8 * hidden * hidden;
    // SwiGLU ratio = 3 * intermediate / (8 * hidden)
    let swiglu_ratio = (3.0 * intermediate as f64) / (8.0 * hidden as f64);
    assert!(
        swiglu_ratio > 1.0 && swiglu_ratio < 1.5,
        "SwiGLU vs standard FFN param ratio should be ~1.32, got {swiglu_ratio:.3}"
    );
    let _ = standard_ffn; // suppress unused warning
}

// ===========================================================================
// 5. Config builder pattern
// ===========================================================================

/// with_vocab_size creates a new config with updated vocab_size.
#[test]
fn test_builder_with_vocab_size() {
    let cfg = tiny_config().with_vocab_size(50_000);
    assert_eq!(cfg.vocab_size, 50_000);
    // Other fields unchanged
    assert_eq!(cfg.hidden_size, 256);
    assert_eq!(cfg.num_hidden_layers, 2);
    assert!(cfg.validate().is_ok());
}

/// with_num_hidden_layers creates a new config with updated layer count.
#[test]
fn test_builder_with_num_hidden_layers() {
    let cfg = tiny_config().with_num_hidden_layers(16);
    assert_eq!(cfg.num_hidden_layers, 16);
    // Other fields unchanged
    assert_eq!(cfg.hidden_size, 256);
    assert_eq!(cfg.vocab_size, 100);
    assert!(cfg.validate().is_ok());
}

/// Builder methods can be chained.
#[test]
fn test_builder_chaining() {
    let cfg = tiny_config()
        .with_vocab_size(32_000)
        .with_num_hidden_layers(12);
    assert_eq!(cfg.vocab_size, 32_000);
    assert_eq!(cfg.num_hidden_layers, 12);
    assert!(cfg.validate().is_ok());
}

/// Config Clone produces an independent copy.
#[test]
fn test_config_clone_independence() {
    let cfg1 = tiny_config();
    let cfg2 = cfg1.clone().with_vocab_size(999);
    assert_eq!(cfg1.vocab_size, 100);
    assert_eq!(cfg2.vocab_size, 999);
}

/// Config Debug representation includes field names.
#[test]
fn test_config_debug_format() {
    let cfg = tiny_config();
    let debug = format!("{cfg:?}");
    assert!(
        debug.contains("hidden_size"),
        "Debug should include hidden_size"
    );
    assert!(
        debug.contains("num_attention_heads"),
        "Debug should include num_attention_heads"
    );
    assert!(
        debug.contains("rope_theta"),
        "Debug should include rope_theta"
    );
}

// ===========================================================================
// 6. Forward validation errors
// ===========================================================================

/// Mismatched input_ids and positions lengths should produce an error.
#[test]
fn test_forward_rejects_mismatched_ids_positions() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let result = model.forward(&[0, 1, 2], &[0, 1]); // 3 ids, 2 positions
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("input_ids") || msg.contains("positions"),
        "error should mention input/positions mismatch: {msg}"
    );
}

/// Empty input should produce an error (positions is empty => mask creation fails
/// or embedding lookup fails).
#[test]
fn test_forward_empty_input_is_error() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let result = model.forward(&[], &[]);
    // Empty input should fail at embedding or reshape
    assert!(result.is_err(), "empty input should produce an error");
}

/// forward_from_embeddings with wrong hidden_size should error.
#[test]
fn test_forward_from_embeddings_wrong_hidden_size() {
    let cfg = tiny_config(); // hidden=256
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    // Create embedding with wrong hidden_size (128 instead of 256)
    let bad_embed = DynTensor::zeros(&[1, 3, 128], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&bad_embed, &[0, 1, 2], None);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("hidden_size"),
        "error should mention hidden_size mismatch: {msg}"
    );
}

/// forward_from_embeddings with mismatched seq_len and positions should error.
#[test]
fn test_forward_from_embeddings_mismatched_seq_positions() {
    let cfg = tiny_config(); // hidden=256
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    // 3-token embedding but 2 positions
    let embed = DynTensor::zeros(&[1, 3, 256], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&embed, &[0, 1], None);
    assert!(result.is_err());
}

// ===========================================================================
// 7. KV cache mismatch detection
// ===========================================================================

/// KV cache with wrong layer count should be detected at forward time.
#[test]
fn test_kv_cache_layer_mismatch_detected() {
    let cfg = tiny_config(); // 2 layers
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    // Create cache with 5 layers (model has 2)
    let mut bad_cache = KvCache::new(5);
    let result = model.forward_cached(&[0], &[0], Some(&mut bad_cache));
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("cache") || msg.contains("layer"),
        "error should mention cache/layer mismatch: {msg}"
    );
}

/// KV cache with 1 layer when model has 2 should fail.
#[test]
fn test_kv_cache_too_few_layers_detected() {
    let cfg = tiny_config(); // 2 layers
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let mut bad_cache = KvCache::new(1);
    let result = model.forward_cached(&[0], &[0], Some(&mut bad_cache));
    assert!(result.is_err());
}

// ===========================================================================
// 8. Causal mask construction
// ===========================================================================

/// build_causal_mask returns None for single-token decode (optimization).
#[test]
fn test_causal_mask_none_for_single_token() {
    use crate::forward_common::build_causal_mask;

    let mask = build_causal_mask(1, None, DType::F32, &Device::Cpu).unwrap();
    assert!(mask.is_none(), "single token should produce no mask");
}

/// build_causal_mask returns None for single-token with cache.
#[test]
fn test_causal_mask_none_for_single_token_with_cache() {
    use crate::forward_common::build_causal_mask;

    // Single token decode step with 10 cached tokens
    let cache = KvCache::new(2);
    // Note: cache seq_len is 0 because we haven't put anything in it.
    // With seq_len=1 and total_seq=1: returns None.
    let mask = build_causal_mask(1, Some(&cache), DType::F32, &Device::Cpu).unwrap();
    assert!(
        mask.is_none(),
        "single token with empty cache should produce no mask"
    );
}

/// build_causal_mask returns Some for multi-token prefill.
#[test]
fn test_causal_mask_some_for_multi_token() {
    use crate::forward_common::build_causal_mask;

    let mask = build_causal_mask(4, None, DType::F32, &Device::Cpu).unwrap();
    assert!(mask.is_some(), "multi-token prefill should produce a mask");
    let m = mask.unwrap();
    // Causal mask shape: [1, 1, 4, 4]
    assert_eq!(m.dims(), &[1, 1, 4, 4]);
}

// ===========================================================================
// 9. Edge cases: minimum viable config
// ===========================================================================

/// Single-layer, single-head model should load and run forward.
#[test]
fn test_minimum_viable_config_single_layer_single_head() {
    let cfg = Qwen3Config::new(128, 256, 1, 1, 1, 10, 1e-6, 10_000.0, 16, true, None);
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.num_kv_groups().unwrap(), 1);

    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
}

/// Single-layer model with GQA (2 heads, 1 kv_head).
#[test]
fn test_minimum_config_with_gqa() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 1, 20, 1e-6, 10_000.0, 16, true, None);
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.num_kv_groups().unwrap(), 2);

    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let logits = model.forward(&[0, 1], &[0, 1]).unwrap();
    assert_eq!(logits.dims(), &[1, 2, cfg.vocab_size]);
}

/// Config with very large num_hidden_layers (128) should validate.
#[test]
fn test_config_many_layers_validates() {
    let cfg = tiny_config().with_num_hidden_layers(128);
    assert!(cfg.validate().is_ok());
}

/// Config with very large vocab should validate.
#[test]
fn test_config_large_vocab_validates() {
    let cfg = tiny_config().with_vocab_size(1_000_000);
    assert!(cfg.validate().is_ok());
}

/// Config with hidden_size=0 should fail validation.
#[test]
fn test_config_zero_hidden_size_rejected() {
    let cfg = Qwen3Config::new(0, 512, 2, 2, 2, 100, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

/// Config with intermediate_size=0 should fail validation.
#[test]
fn test_config_zero_intermediate_size_rejected() {
    let cfg = Qwen3Config::new(256, 0, 2, 2, 2, 100, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

/// Config with NaN rms_norm_eps should fail validation.
#[test]
fn test_config_nan_rms_norm_eps_rejected() {
    let cfg = Qwen3Config::new(256, 512, 2, 2, 2, 100, f64::NAN, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

/// Config with Infinity rms_norm_eps should fail validation.
#[test]
fn test_config_inf_rms_norm_eps_rejected() {
    let cfg = Qwen3Config::new(
        256,
        512,
        2,
        2,
        2,
        100,
        f64::INFINITY,
        10_000.0,
        64,
        true,
        None,
    );
    assert!(cfg.validate().is_err());
}

// ===========================================================================
// 10. MoE production configs (Qwen3-30B-A3B, Qwen3-235B-A22B)
// ===========================================================================

/// Qwen3-30B-A3B MoE config: 128 experts, 8 active, with shared expert.
#[test]
fn test_moe_config_30b_a3b() {
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
    let moe = Qwen3MoeConfig::new(base, 128, 8, true, Some(2560));
    assert!(moe.validate().is_ok());
    assert_eq!(moe.num_experts, 128);
    assert_eq!(moe.num_experts_per_tok, 8);
    assert!(moe.shared_expert);
    assert_eq!(moe.shared_expert_ff_dim(), 2560);
}

/// Qwen3-235B-A22B MoE config: 128 experts, 8 active.
#[test]
fn test_moe_config_235b_a22b() {
    let base = Qwen3Config::new(
        6144,
        3072,
        94,
        64,
        4,
        151_936,
        1e-6,
        1_000_000.0,
        131_072,
        false,
        None,
    );
    let moe = Qwen3MoeConfig::new(base, 128, 8, true, Some(3072));
    assert!(moe.validate().is_ok());
    assert_eq!(moe.num_experts, 128);
    assert_eq!(moe.num_experts_per_tok, 8);
}

/// MoE effective parameter count: total_experts * expert_params + shared_expert.
/// Active params per token = active_experts * expert_params + shared_expert.
#[test]
fn test_moe_effective_parameter_ratio() {
    // 128 experts, 8 active => 6.25% expert utilization per token
    let total_experts = 128_usize;
    let active = 8_usize;
    let utilization = active as f64 / total_experts as f64;
    assert!(
        (utilization - 0.0625).abs() < 1e-10,
        "utilization should be 6.25%, got {:.2}%",
        utilization * 100.0
    );
}

/// MoE config with experts_per_tok > num_experts should fail.
#[test]
fn test_moe_config_active_exceeds_total_rejected() {
    let base = tiny_config();
    let moe = Qwen3MoeConfig::new(base, 4, 8, false, None);
    assert!(moe.validate().is_err());
}

/// MoE config inherits base config validation.
#[test]
fn test_moe_config_inherits_base_validation() {
    // Base config with zero hidden_size should fail MoE validation too
    let bad_base = Qwen3Config::new(0, 512, 2, 2, 2, 100, 1e-6, 10_000.0, 64, true, None);
    let moe = Qwen3MoeConfig::new(bad_base, 8, 4, false, None);
    assert!(moe.validate().is_err());
}

// ===========================================================================
// 11. Forward from embeddings with correct dimensions
// ===========================================================================

/// forward_from_embeddings with correct dimensions should succeed.
#[test]
fn test_forward_from_embeddings_correct_dims() {
    let cfg = tiny_config(); // hidden=256
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let embed = DynTensor::zeros(&[1, 3, 256], DType::F32, &Device::Cpu).unwrap();
    let logits = model
        .forward_from_embeddings(&embed, &[0, 1, 2], None)
        .unwrap();
    assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
}

/// forward_from_embeddings_with_hidden returns both logits and hidden states.
#[test]
fn test_forward_from_embeddings_with_hidden_returns_both() {
    let cfg = tiny_config(); // hidden=256
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let embed = DynTensor::zeros(&[1, 2, 256], DType::F32, &Device::Cpu).unwrap();
    let (logits, hidden) = model
        .forward_from_embeddings_with_hidden(&embed, &[0, 1], None)
        .unwrap();
    assert_eq!(logits.dims(), &[1, 2, cfg.vocab_size]);
    assert_eq!(hidden.dims(), &[1, 2, cfg.hidden_size]);
}

// ===========================================================================
// 12. RoPE cache memory estimation
// ===========================================================================

/// RoPE cache memory = 2 * max_seq_len * (head_dim/2) * sizeof(f32).
/// For Qwen3 defaults (32K seq, head_dim=128): ~8 MiB.
#[test]
fn test_rope_cache_memory_estimate() {
    let max_seq = 32_768_usize;
    let head_dim = 128_usize;
    let half_dim = head_dim / 2;
    let sizeof_f32 = 4_usize;

    let expected_bytes = 2 * max_seq * half_dim * sizeof_f32;
    let mib = expected_bytes as f64 / (1024.0 * 1024.0);

    // 2 * 32768 * 64 * 4 = 16,777,216 = 16 MiB
    assert_eq!(expected_bytes, 16_777_216);
    assert!((mib - 16.0).abs() < 0.01, "should be 16 MiB, got {mib:.2}");
}

/// RoPE cache for small head_dim=4 has correct memory footprint.
#[test]
fn test_rope_cache_small_memory() {
    let cache = RoPECache::new(16, 4, 10_000.0);
    assert_eq!(cache.max_seq_len(), 16);
    assert_eq!(cache.half_dim(), 2);
    // 16 positions * 2 half_dim = 32 f32 values per table (cos and sin)
    // Total: 64 f32 values
    let (cos, sin) = cache.get(0);
    assert_eq!(cos.len(), 2);
    assert_eq!(sin.len(), 2);
}

// ===========================================================================
// 13. Attention scale factor
// ===========================================================================

/// Attention scale = 1/sqrt(head_dim) is constant across all Qwen3 variants.
#[test]
fn test_attention_scale_constant() {
    let scale = 1.0 / (128.0_f64).sqrt();
    // ~0.08838834764831843
    assert!(
        (scale - 0.08838834764831843).abs() < 1e-15,
        "scale should be 1/sqrt(128) = 0.08839, got {scale}"
    );

    // Scale is independent of hidden_size, num_heads, etc.
    for hidden in [896, 2048, 4096, 5120] {
        let cfg = Qwen3Config::new(
            hidden,
            hidden * 2,
            1,
            2,
            2,
            100,
            1e-6,
            10_000.0,
            64,
            true,
            None,
        );
        let s = 1.0 / (cfg.head_dim() as f64).sqrt();
        assert!(
            (s - scale).abs() < 1e-15,
            "scale should be constant for hidden={hidden}"
        );
    }
}

// ===========================================================================
// 14. Tied vs untied embedding parameter count
// ===========================================================================

/// Tied embeddings save vocab_size * hidden_size parameters.
#[test]
fn test_tied_embeddings_parameter_savings() {
    let hidden = 4096_usize;
    let vocab = 151_936_usize;
    let savings = vocab * hidden; // params saved by tying
                                  // 151936 * 4096 = 622,329,856 ~ 622M params
    assert_eq!(savings, 622_329_856);

    // For small models (0.6B tied), this saves ~37% of embedding params
    let hidden_06b = 896_usize;
    let savings_06b = vocab * hidden_06b;
    let total_06b_approx = 600_000_000_usize;
    let pct = savings_06b as f64 / total_06b_approx as f64 * 100.0;
    assert!(
        pct > 15.0 && pct < 30.0,
        "0.6B tied savings should be ~22%, got {pct:.1}%"
    );
}

/// Model with tie_word_embeddings=true loads with fewer parameters.
#[test]
fn test_tied_vs_untied_model_loads() {
    let cfg_tied = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, 10_000.0, 32, true, None);
    let cfg_untied = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, 10_000.0, 32, false, None);

    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model_tied = Qwen3Model::load(&vb, cfg_tied.clone()).unwrap();
    let model_untied = Qwen3Model::load(&vb, cfg_untied.clone()).unwrap();

    // Both should produce same-shaped output
    let logits_tied = model_tied.forward(&[0], &[0]).unwrap();
    let logits_untied = model_untied.forward(&[0], &[0]).unwrap();
    assert_eq!(logits_tied.dims(), logits_untied.dims());

    // But parameter counts differ
    let params_tied = dense_param_count(&cfg_tied);
    let params_untied = dense_param_count(&cfg_untied);
    assert!(
        params_untied > params_tied,
        "untied ({params_untied}) should have more params than tied ({params_tied})"
    );
    // The difference should be exactly vocab_size * hidden_size
    assert_eq!(
        params_untied - params_tied,
        cfg_tied.vocab_size * cfg_tied.hidden_size
    );
}
