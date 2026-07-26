// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for GLM-5 model configuration, shape algebra, and
//! architectural invariants.
//!
//! Part of #3525. Covers:
//!
//! 1. Config validation boundary conditions (each invalid field produces
//!    correct error diagnostics)
//! 2. Parametric model-size sweeps (1.5B, 9B, 130B class dimensions)
//! 3. KV cache shape algebra (flat KVCache dimensions vs config)
//! 4. Vocabulary projection alignment and embedding/output shape consistency
//! 5. SwiGLU intermediate dimension ratio bounds across configs
//! 6. Rotary embedding config consistency (rope_dim, kv_channels, head_dim)
//! 7. GQA repeat factor exhaustive divisor enumeration
//! 8. Config Debug/Clone trait independence
//! 9. Error type coverage (Glm5Error variant construction and Display)
//! 10. Multi-step KV cache shape accumulation
//! 11. QKV dimension algebra across attention configs
//! 12. Config new() exhaustive field preservation
//! 13. Parameter count estimation across GLM model sizes
//! 14. Half-RoPE wavelength analysis (GLM-specific partial rotation)
//! 15. GQA KV cache memory calculations (byte-level)
//! 16. Attention scale factor consistency
//! 17. KV cache total memory for production configs
//! 18. Config hidden_size / heads / kv_channels relationship
//! 19. Layernorm epsilon precision requirements across variants
//! 20. Weight memory estimation per layer

use super::*;
use crate::kv_cache::KVCache;
use crate::test_utils::tiny_config;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::kv_cache::KvCache;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_tiny_model() -> Glm5Model {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    Glm5Model::load(&vb, cfg).unwrap()
}

fn load_model_with_config(cfg: Glm5Config) -> Glm5Model {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    Glm5Model::load(&vb, cfg).unwrap()
}

/// Compute the total parameter count for a GLM model configuration.
///
/// Accounts for:
/// - Embedding: [padded_vocab_size, hidden_size]
/// - Per-layer: QKV (fused), QKV bias (optional), dense, MLP (SwiGLU), norms
/// - Final layernorm: [hidden_size]
/// - Output layer: [padded_vocab_size, hidden_size] (always untied in GLM)
/// - Output bias (optional): [padded_vocab_size]
fn glm_param_count(cfg: &Glm5Config) -> usize {
    let h = cfg.hidden_size;
    let ffn = cfg.ffn_hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.multi_query_group_num;
    let hd = cfg.kv_channels;
    let v = cfg.padded_vocab_size;

    // Embedding: [vocab, hidden]
    let embedding = v * h;

    // Per-layer attention: fused QKV [(nh + 2*nkv)*hd, h]
    let qkv_weight = (nh + 2 * nkv) * hd * h;
    let qkv_bias = if cfg.add_qkv_bias {
        (nh + 2 * nkv) * hd
    } else {
        0
    };
    // Dense output projection: [h, nh*hd]
    let dense_weight = h * nh * hd;
    let dense_bias = if cfg.add_bias_linear { h } else { 0 };

    // MLP: dense_h_to_4h [ffn*2, h], dense_4h_to_h [h, ffn]
    let mlp_h_to_4h = ffn * 2 * h;
    let mlp_h_to_4h_bias = if cfg.add_bias_linear { ffn * 2 } else { 0 };
    let mlp_4h_to_h = h * ffn;
    let mlp_4h_to_h_bias = if cfg.add_bias_linear { h } else { 0 };

    // RMSNorm: 2 * [h] per layer (input_layernorm + post_attention_layernorm)
    let layer_norms = 2 * h;

    let per_layer = qkv_weight
        + qkv_bias
        + dense_weight
        + dense_bias
        + mlp_h_to_4h
        + mlp_h_to_4h_bias
        + mlp_4h_to_h
        + mlp_4h_to_h_bias
        + layer_norms;

    // Final layernorm: [h]
    let final_norm = h;

    // Output layer: [vocab, hidden] (always separate in GLM)
    let output_weight = v * h;
    let output_bias = if cfg.add_bias_linear { v } else { 0 };

    embedding + cfg.num_layers * per_layer + final_norm + output_weight + output_bias
}

// ===========================================================================
// 1. Config validation: each invalid field produces a diagnostic error
// ===========================================================================

#[test]
fn test_validate_rejects_zero_kv_channels() {
    let mut cfg = tiny_config();
    cfg.kv_channels = 0;
    let err = cfg.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("kv_channels"),
        "error should mention kv_channels: {msg}"
    );
}

#[test]
fn test_validate_rejects_kv_channels_not_multiple_of_4_odd() {
    let mut cfg = tiny_config();
    cfg.kv_channels = 5;
    let err = cfg.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("multiple of 4"),
        "error should mention multiple-of-4 constraint: {msg}"
    );
}

#[test]
fn test_validate_rejects_kv_channels_not_multiple_of_4_even() {
    let mut cfg = tiny_config();
    cfg.kv_channels = 6;
    let err = cfg.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("multiple of 4"),
        "error should mention multiple-of-4 constraint for kv_channels=6: {msg}"
    );
}

#[test]
fn test_validate_rejects_zero_padded_vocab_size() {
    let mut cfg = tiny_config();
    cfg.padded_vocab_size = 0;
    let err = cfg.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("padded_vocab_size"),
        "error should mention padded_vocab_size: {msg}"
    );
}

#[test]
fn test_validate_rejects_zero_seq_length() {
    let mut cfg = tiny_config();
    cfg.seq_length = 0;
    let err = cfg.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("seq_length"),
        "error should mention seq_length: {msg}"
    );
}

#[test]
fn test_validate_rejects_nan_rope_theta() {
    let mut cfg = tiny_config();
    cfg.rope_theta = f64::NAN;
    let err = cfg.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("rope_theta"),
        "error should mention rope_theta: {msg}"
    );
}

#[test]
fn test_validate_rejects_negative_rope_theta() {
    let mut cfg = tiny_config();
    cfg.rope_theta = -500.0;
    let err = cfg.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("rope_theta") && msg.contains("positive"),
        "error should explain rope_theta must be positive: {msg}"
    );
}

#[test]
fn test_validate_rejects_inf_rope_theta() {
    let mut cfg = tiny_config();
    cfg.rope_theta = f64::INFINITY;
    assert!(cfg.validate().is_err(), "infinite rope_theta must fail");
}

#[test]
fn test_validate_rejects_nan_layernorm_epsilon() {
    let mut cfg = tiny_config();
    cfg.layernorm_epsilon = f64::NAN;
    assert!(cfg.validate().is_err(), "NaN epsilon must fail");
}

#[test]
fn test_validate_rejects_zero_layernorm_epsilon() {
    let mut cfg = tiny_config();
    cfg.layernorm_epsilon = 0.0;
    let err = cfg.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("layernorm_epsilon") && msg.contains("positive"),
        "error should explain epsilon must be positive: {msg}"
    );
}

#[test]
fn test_validate_rejects_indivisible_heads() {
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 7;
    cfg.multi_query_group_num = 3;
    let err = cfg.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("divisible"),
        "error should mention divisibility: {msg}"
    );
}

#[test]
fn test_validate_rejects_zero_num_layers() {
    let mut cfg = tiny_config();
    cfg.num_layers = 0;
    let err = cfg.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("num_layers"),
        "error should mention num_layers: {msg}"
    );
}

// ===========================================================================
// 2. Parametric model-size sweeps
// ===========================================================================

/// Hypothetical GLM model sizes: verify config validates and derived
/// dimensions are consistent for each size class.
#[test]
fn test_parametric_model_sizes_validate_and_derive_consistently() {
    // (hidden, ffn, layers, heads, kv_heads, vocab, kv_ch, name)
    let model_sizes: Vec<(usize, usize, usize, usize, usize, usize, usize, &str)> = vec![
        // ~1.5B class (smaller)
        (2048, 5504, 24, 16, 2, 65536, 128, "1.5B"),
        // ~9B class (GLM-4-9B)
        (4096, 13696, 40, 32, 2, 151552, 128, "9B"),
        // ~130B class (hypothetical large)
        (8192, 28672, 80, 64, 8, 151552, 128, "130B"),
    ];

    for (hidden, ffn, layers, heads, kv_heads, vocab, kv_ch, name) in &model_sizes {
        let cfg = Glm5Config::new(
            *hidden, *ffn, *layers, *heads, *kv_heads, *vocab, *kv_ch, 1e-5, 8192, true, true,
            false, 10_000.0,
        );
        assert!(cfg.validate().is_ok(), "{name} config should validate");

        // Derived dimension checks
        assert_eq!(
            cfg.head_dim(),
            *kv_ch,
            "{name}: head_dim should equal kv_channels"
        );

        let groups = cfg.num_kv_groups().unwrap();
        assert_eq!(
            groups,
            heads / kv_heads,
            "{name}: num_kv_groups = heads/kv_heads"
        );

        // QKV fused dimension
        let qkv_dim = (*heads + 2 * *kv_heads) * *kv_ch;
        assert!(qkv_dim > 0, "{name}: QKV fused dim should be positive");
        assert_eq!(
            qkv_dim,
            *heads * *kv_ch + 2 * *kv_heads * *kv_ch,
            "{name}: QKV = Q + K + V"
        );

        // SwiGLU fused dim must be even
        assert_eq!((ffn * 2) % 2, 0, "{name}: SwiGLU fused dim must be even");
    }
}

/// Verify parameter count estimates scale with model size.
#[test]
fn test_parameter_count_increases_with_model_size() {
    let small = Glm5Config::new(
        2048, 5504, 24, 16, 2, 65536, 128, 1e-5, 4096, true, true, false, 10_000.0,
    );
    let large = Glm5Config::default(); // 9B

    let small_params = glm_param_count(&small);
    let large_params = glm_param_count(&large);
    assert!(
        large_params > small_params,
        "9B ({large_params}) should have more params than small ({small_params})"
    );
}

// ===========================================================================
// 3. KV cache shape algebra
// ===========================================================================

#[test]
fn test_flat_kv_cache_token_size_matches_config() {
    let cfg = tiny_config();
    let nkv = cfg.multi_query_group_num;
    let hd = cfg.kv_channels;
    let token_size = nkv * hd;

    let mut cache = KVCache::new(cfg.num_layers, nkv, hd, cfg.seq_length);
    let key = vec![0.0_f32; token_size];
    let val = vec![0.0_f32; token_size];

    for layer in 0..cfg.num_layers {
        cache.append(layer, &key, &val).unwrap();
    }
    assert_eq!(cache.len(), 1);

    // After 1 token, each layer's key buffer should have exactly token_size floats
    for layer in 0..cfg.num_layers {
        assert_eq!(
            cache.get_keys(layer).len(),
            token_size,
            "layer {layer}: keys should have {token_size} floats after 1 token"
        );
        assert_eq!(
            cache.get_values(layer).len(),
            token_size,
            "layer {layer}: values should have {token_size} floats after 1 token"
        );
    }
}

#[test]
fn test_flat_kv_cache_multi_token_accumulation() {
    let num_layers = 3;
    let num_heads = 2;
    let head_dim = 8;
    let max_seq = 10;
    let token_size = num_heads * head_dim;

    let mut cache = KVCache::new(num_layers, num_heads, head_dim, max_seq);

    // Append 5 tokens
    for _tok in 0..5 {
        for layer in 0..num_layers {
            let key: Vec<f32> = (0..token_size).map(|i| i as f32).collect();
            let val: Vec<f32> = (0..token_size).map(|i| i as f32 + 100.0).collect();
            cache.append(layer, &key, &val).unwrap();
        }
    }
    assert_eq!(cache.len(), 5);

    // Each layer should have 5 * token_size floats
    for layer in 0..num_layers {
        assert_eq!(
            cache.get_keys(layer).len(),
            5 * token_size,
            "layer {layer}: keys should accumulate 5 * token_size"
        );
    }
}

#[test]
fn test_flat_kv_cache_max_seq_exactly_at_limit() {
    // Create cache with max_seq=3 and fill it exactly
    let mut cache = KVCache::new(1, 2, 4, 3);
    let token = vec![0.0_f32; 2 * 4];

    for _tok in 0..3 {
        cache.append(0, &token, &token).unwrap();
        // Single-layer cache increments on layer 0 (last layer)
    }
    assert_eq!(cache.len(), 3);

    // The 4th append should fail
    let result = cache.append(0, &token, &token);
    assert!(result.is_err(), "should reject append beyond max_seq=3");
}

#[test]
fn test_dyntensor_kv_cache_new_matches_model_layers() {
    let model = load_tiny_model();
    let cache = model.new_cache();
    assert_eq!(
        cache.num_layers(),
        model.config().num_layers,
        "DynTensor KvCache layers should match model"
    );
    assert_eq!(cache.seq_len(), 0, "new cache should be empty");
}

#[test]
fn test_dyntensor_kv_cache_seq_len_grows_through_decode() {
    let model = load_tiny_model();
    let mut cache = model.new_cache();

    // Prefill 4 tokens
    let _ = model
        .forward_cached(&[0, 1, 2, 3], &[0, 1, 2, 3], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 4);

    // 3 decode steps
    for step in 0..3 {
        let pos = 4 + step;
        let _ = model
            .forward_cached(&[0], &[pos], Some(&mut cache))
            .unwrap();
        assert_eq!(
            cache.seq_len(),
            pos + 1,
            "after decode step {step}: cache should hold {} tokens",
            pos + 1
        );
    }
}

// ===========================================================================
// 4. Vocabulary projection alignment and embedding/output shape consistency
// ===========================================================================

#[test]
fn test_vocab_size_padded_alignment() {
    // GLM-4-9B pads to 151552 which is divisible by 64 for GPU alignment
    let cfg = Glm5Config::default();
    assert_eq!(
        cfg.padded_vocab_size % 64,
        0,
        "padded_vocab_size={} should be aligned to 64",
        cfg.padded_vocab_size
    );

    // 151552 = 2368 * 64 = 37 * 64 * 64
    assert_eq!(cfg.padded_vocab_size, 151552);
}

#[test]
fn test_embedding_and_output_use_same_vocab_dimension() {
    // Both embedding and output_layer use padded_vocab_size as their first dim.
    // Verify by loading and checking forward output shape.
    let model = load_tiny_model();
    let cfg = model.config().clone();

    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(
        logits.dims()[2],
        cfg.padded_vocab_size,
        "output projection dim should equal padded_vocab_size"
    );
}

#[test]
fn test_output_logits_cover_full_vocab() {
    let model = load_tiny_model();
    let cfg = model.config().clone();
    let logits = model.forward(&[0], &[0]).unwrap();
    let flat = logits.to_flat_vec::<f32>().unwrap();

    // Should have exactly padded_vocab_size logits for 1 token
    assert_eq!(
        flat.len(),
        cfg.padded_vocab_size,
        "single-token forward should produce padded_vocab_size logits"
    );
}

#[test]
fn test_multi_token_logits_shape_is_seq_times_vocab() {
    let model = load_tiny_model();
    let cfg = model.config().clone();
    let seq_len = 7;
    let ids: Vec<usize> = (0..seq_len).collect();
    let pos: Vec<usize> = (0..seq_len).collect();
    let logits = model.forward(&ids, &pos).unwrap();
    let flat = logits.to_flat_vec::<f32>().unwrap();

    assert_eq!(
        flat.len(),
        seq_len * cfg.padded_vocab_size,
        "logits total elements should be seq_len * vocab_size"
    );
}

// ===========================================================================
// 5. SwiGLU intermediate dimension ratio bounds
// ===========================================================================

#[test]
fn test_swiglu_fused_dim_is_double_ffn() {
    // dense_h_to_4h outputs ffn_hidden_size * 2 (gate + up fused)
    let cfgs = vec![
        ("tiny", tiny_config()),
        ("9B", Glm5Config::default()),
        ("chat", Glm5Config::glm4_9b_chat()),
    ];
    for (name, cfg) in &cfgs {
        let fused = cfg.ffn_hidden_size * 2;
        assert_eq!(fused % 2, 0, "{name}: fused SwiGLU dim must be even");
        assert_eq!(
            fused / 2,
            cfg.ffn_hidden_size,
            "{name}: each SwiGLU half should equal ffn_hidden_size"
        );
    }
}

#[test]
fn test_swiglu_mlp_expansion_ratio_in_reasonable_range() {
    // The MLP expansion ratio (ffn_hidden_size / hidden_size) typically
    // ranges from 2.0 to 5.0 for SwiGLU models.
    let cfgs = vec![("tiny", tiny_config()), ("9B", Glm5Config::default())];
    for (name, cfg) in &cfgs {
        let ratio = cfg.ffn_hidden_size as f64 / cfg.hidden_size as f64;
        assert!(
            (1.5..=6.0).contains(&ratio),
            "{name}: MLP expansion ratio {ratio} outside expected range [1.5, 6.0]"
        );
    }
}

#[test]
fn test_swiglu_dense_4h_to_h_projects_back_to_hidden() {
    // dense_4h_to_h: [hidden_size, ffn_hidden_size]
    // The output dim is hidden_size, matching the residual connection.
    let cfg = tiny_config();
    // This invariant is checked indirectly: forward succeeds because
    // the residual add requires matching dimensions.
    let model = load_model_with_config(cfg.clone());
    let emb = DynTensor::ones(&[1, 2, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits = model.forward_from_embeddings(&emb, &[0, 1], None).unwrap();
    assert_eq!(
        logits.dims(),
        &[1, 2, cfg.padded_vocab_size],
        "MLP output must match hidden_size for residual to work"
    );
}

// ===========================================================================
// 6. Rotary embedding config consistency
// ===========================================================================

#[test]
fn test_rope_dim_is_half_of_head_dim() {
    // GLM uses HalfRotaryEmbedding: rotates first head_dim/2 dims
    let cfg = tiny_config();
    let rope_dim = cfg.kv_channels / 2;
    assert_eq!(
        rope_dim,
        cfg.head_dim() / 2,
        "rope_dim should be half of head_dim"
    );
    assert!(rope_dim > 0, "rope_dim must be positive");
    // rope_dim must be even (each rotation operates on pairs)
    assert_eq!(
        rope_dim % 2,
        0,
        "rope_dim={rope_dim} must be even for paired rotation"
    );
}

#[test]
fn test_kv_channels_equals_head_dim_across_configs() {
    let cfgs = vec![
        ("tiny", tiny_config()),
        ("9B", Glm5Config::default()),
        ("chat", Glm5Config::glm4_9b_chat()),
    ];
    for (name, cfg) in &cfgs {
        assert_eq!(
            cfg.kv_channels,
            cfg.head_dim(),
            "{name}: kv_channels and head_dim() should be identical"
        );
    }
}

#[test]
fn test_rope_theta_base_vs_chat_relationship() {
    let base = Glm5Config::default();
    let chat = Glm5Config::glm4_9b_chat();

    // Chat uses rope_ratio=500, so theta_chat = theta_base * 500
    let expected_chat_theta = base.rope_theta * 500.0;
    assert!(
        (chat.rope_theta - expected_chat_theta).abs() < 1.0,
        "chat theta ({}) should be base theta ({}) * 500 = {expected_chat_theta}",
        chat.rope_theta,
        base.rope_theta,
    );
}

#[test]
fn test_kv_channels_divisible_by_4_for_half_rope() {
    // HalfRotaryEmbedding requires head_dim (= kv_channels) divisible by 4
    for kv_ch in [4, 8, 16, 32, 64, 128, 256] {
        assert_eq!(kv_ch % 4, 0, "kv_channels={kv_ch} must be divisible by 4");
        // Inner rope dim is kv_ch / 2, and that must also support pairing
        let inner_dim = kv_ch / 2;
        assert_eq!(inner_dim % 2, 0, "inner rope dim={inner_dim} must be even");
    }
}

// ===========================================================================
// 7. GQA repeat factor exhaustive divisor enumeration
// ===========================================================================

#[test]
fn test_gqa_repeat_factor_for_all_valid_divisors_of_32() {
    // GLM-4-9B has 32 Q heads. Valid kv_head counts are divisors of 32.
    let num_heads = 32_usize;
    let divisors: Vec<usize> = (1..=num_heads).filter(|d| num_heads.is_multiple_of(*d)).collect();

    // Expected divisors of 32: 1, 2, 4, 8, 16, 32
    assert_eq!(divisors, vec![1, 2, 4, 8, 16, 32], "divisors of 32");

    for &kv_heads in &divisors {
        let cfg = Glm5Config {
            multi_query_group_num: kv_heads,
            ..Default::default()
        };
        assert!(
            cfg.validate().is_ok(),
            "kv_heads={kv_heads} should be valid for 32 Q heads"
        );
        let repeat = cfg.num_kv_groups().unwrap();
        assert_eq!(
            repeat,
            num_heads / kv_heads,
            "kv_heads={kv_heads}: repeat should be {}/{}={}",
            num_heads,
            kv_heads,
            num_heads / kv_heads
        );
    }
}

#[test]
fn test_gqa_repeat_factor_mqa_extreme() {
    // MQA: single KV head, maximum sharing
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 16;
    cfg.multi_query_group_num = 1;
    cfg.hidden_size = 16 * cfg.kv_channels; // Maintain invariant
    assert!(cfg.validate().is_ok());
    assert_eq!(
        cfg.num_kv_groups().unwrap(),
        16,
        "MQA: repeat factor = num_heads"
    );
}

#[test]
fn test_gqa_repeat_factor_mha_no_sharing() {
    // MHA: every head has its own KV head, no sharing
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 4;
    cfg.multi_query_group_num = 4;
    assert!(cfg.validate().is_ok());
    assert_eq!(
        cfg.num_kv_groups().unwrap(),
        1,
        "MHA: repeat factor = 1 (no sharing)"
    );
}

#[test]
fn test_gqa_non_divisor_kv_heads_rejected() {
    // Non-divisors of num_attention_heads should fail validation
    let non_divisors_of_4 = vec![3, 5, 6, 7];
    for kv in non_divisors_of_4 {
        let mut cfg = tiny_config(); // num_attention_heads = 4
        cfg.multi_query_group_num = kv;
        assert!(
            cfg.validate().is_err(),
            "kv_heads={kv} should be rejected for 4 Q heads"
        );
        assert!(
            cfg.num_kv_groups().is_err(),
            "num_kv_groups() should also fail for kv_heads={kv}"
        );
    }
}

// ===========================================================================
// 8. Config Debug/Clone trait independence
// ===========================================================================

#[test]
fn test_config_clone_is_independent() {
    let original = tiny_config();
    let cloned = original.clone();

    // Mutate a fresh copy to prove clone is independent
    let mut mutated = original;
    mutated.hidden_size = 9999;
    mutated.num_layers = 100;
    mutated.rope_theta = 1.0;

    // Clone should be unaffected by mutations to `mutated`
    assert_eq!(cloned.hidden_size, 256);
    assert_eq!(cloned.num_layers, 2);
    assert_eq!(cloned.rope_theta, 10_000.0);

    // Verify mutated is actually different
    assert_ne!(cloned.hidden_size, mutated.hidden_size);
    assert_ne!(cloned.num_layers, mutated.num_layers);
    assert_ne!(cloned.rope_theta, mutated.rope_theta);
}

#[test]
fn test_config_debug_contains_field_names() {
    let cfg = tiny_config();
    let debug_str = format!("{cfg:?}");
    assert!(
        debug_str.contains("hidden_size"),
        "Debug output should contain hidden_size: {debug_str}"
    );
    assert!(
        debug_str.contains("num_attention_heads"),
        "Debug output should contain num_attention_heads: {debug_str}"
    );
    assert!(
        debug_str.contains("multi_query_group_num"),
        "Debug output should contain multi_query_group_num: {debug_str}"
    );
}

// ===========================================================================
// 9. Error type coverage
// ===========================================================================

#[test]
fn test_error_invalid_config_display() {
    let err = Glm5Error::InvalidConfig {
        reason: "test reason".into(),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("invalid config") && msg.contains("test reason"),
        "InvalidConfig display: {msg}"
    );
}

#[test]
fn test_error_invalid_input_display() {
    let err = Glm5Error::InvalidInput {
        reason: "bad input".into(),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("invalid input") && msg.contains("bad input"),
        "InvalidInput display: {msg}"
    );
}

#[test]
fn test_error_cache_mismatch_display() {
    let err = Glm5Error::CacheMismatch {
        cache_layers: 3,
        model_layers: 5,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("3") && msg.contains("5") && msg.contains("cache"),
        "CacheMismatch display: {msg}"
    );
}

#[test]
fn test_error_non_finite_output_display() {
    let err = Glm5Error::NonFiniteOutput {
        stage: "test_stage",
        count: 42,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("test_stage") && msg.contains("42"),
        "NonFiniteOutput display: {msg}"
    );
}

#[test]
fn test_error_weight_load_display() {
    let err = Glm5Error::WeightLoad {
        reason: "missing tensor".into(),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("weight load") && msg.contains("missing tensor"),
        "WeightLoad display: {msg}"
    );
}

#[test]
fn test_error_conversion_to_tensor_error() {
    use nn_core::TensorError;
    let err = Glm5Error::InvalidConfig {
        reason: "test".into(),
    };
    let te: TensorError = err.into();
    let msg = te.to_string();
    assert!(
        msg.contains("test"),
        "TensorError conversion should preserve message: {msg}"
    );
}

// ===========================================================================
// 10. Forward pass input validation edge cases
// ===========================================================================

#[test]
fn test_forward_empty_input_returns_error() {
    let model = load_tiny_model();
    // Empty input (0 tokens) triggers a zero-length dimension error in
    // attention softmax — this is expected behavior, not a bug.
    let result = model.forward(&[], &[]);
    assert!(
        result.is_err(),
        "empty input should return an error (zero-length softmax)"
    );
}

#[test]
fn test_forward_from_embeddings_seq_len_mismatch_rejected() {
    let model = load_tiny_model();
    let cfg = model.config().clone();

    // 3 tokens in embeddings but 2 positions
    let emb = DynTensor::ones(&[1, 3, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&emb, &[0, 1], None);
    assert!(
        result.is_err(),
        "seq_len mismatch between embeddings and positions should be rejected"
    );
}

#[test]
fn test_forward_cached_rejects_cache_with_fewer_layers() {
    let model = load_tiny_model();
    // Model has 2 layers, cache has 1
    let mut cache = KvCache::new(1);
    let result = model.forward_cached(&[0], &[0], Some(&mut cache));
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("1") && msg.contains("2"),
        "error should mention layer counts: {msg}"
    );
}

#[test]
fn test_forward_cached_rejects_cache_with_more_layers() {
    let model = load_tiny_model();
    // Model has 2 layers, cache has 5
    let mut cache = KvCache::new(5);
    let result = model.forward_cached(&[0], &[0], Some(&mut cache));
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("5") && msg.contains("2"),
        "error should mention layer counts: {msg}"
    );
}

// ===========================================================================
// 11. QKV dimension algebra across attention configs
// ===========================================================================

#[test]
fn test_qkv_fused_dimension_for_various_gqa_configs() {
    // QKV fused weight: [(nh + 2*nkv) * hd, hidden_size]
    let cases = vec![
        // (nh, nkv, hd, expected_qkv_dim)
        (4, 2, 64, (4 + 2 * 2) * 64),     // tiny: 512
        (32, 2, 128, (32 + 2 * 2) * 128), // 9B: 4608
        (8, 1, 128, (8 + 2) * 128),   // MQA-8: 1280
        (4, 4, 64, (4 + 2 * 4) * 64),     // MHA-4: 768
        (64, 8, 128, (64 + 2 * 8) * 128), // 130B: 10240
    ];

    for (nh, nkv, hd, expected) in cases {
        let q_dim = nh * hd;
        let k_dim = nkv * hd;
        let v_dim = nkv * hd;
        let total = q_dim + k_dim + v_dim;
        assert_eq!(
            total, expected,
            "nh={nh}, nkv={nkv}, hd={hd}: Q+K+V={total}, expected={expected}"
        );

        // Verify Q >> K == V for GQA (nh > nkv)
        if nh > nkv {
            assert!(
                q_dim > k_dim,
                "GQA: Q dim ({q_dim}) should be larger than K dim ({k_dim})"
            );
        }
        assert_eq!(k_dim, v_dim, "K and V dimensions always match");
    }
}

#[test]
fn test_dense_output_projection_dim_equals_hidden_size() {
    // dense.weight: [hidden_size, nh * hd]
    // Since hidden_size = nh * hd for standard configs, this is square
    let cfg = tiny_config();
    let dense_rows = cfg.hidden_size;
    let dense_cols = cfg.num_attention_heads * cfg.kv_channels;
    assert_eq!(
        dense_rows, dense_cols,
        "for standard configs, dense projection should be [hidden, hidden]"
    );
}

// ===========================================================================
// 12. Config new() exhaustive field preservation
// ===========================================================================

#[test]
fn test_config_new_bool_fields_independently_settable() {
    // Test all 4 combinations of rmsnorm and add_qkv_bias
    for rmsnorm in [true, false] {
        for add_qkv_bias in [true, false] {
            for add_bias_linear in [true, false] {
                let cfg = Glm5Config::new(
                    256,
                    512,
                    2,
                    4,
                    2,
                    100,
                    64,
                    1e-5,
                    64,
                    rmsnorm,
                    add_qkv_bias,
                    add_bias_linear,
                    10_000.0,
                );
                assert_eq!(cfg.rmsnorm, rmsnorm);
                assert_eq!(cfg.add_qkv_bias, add_qkv_bias);
                assert_eq!(cfg.add_bias_linear, add_bias_linear);
                assert!(cfg.validate().is_ok());
            }
        }
    }
}

#[test]
fn test_config_new_with_extreme_valid_values() {
    // Large but valid values
    let cfg = Glm5Config::new(
        16384,     // very large hidden_size
        65536,     // very large ffn
        200,       // many layers
        128,       // many heads
        8,         // 8 kv heads
        262144,    // large vocab
        128,       // standard head dim
        1e-12,     // very small epsilon
        2_097_152, // 2M seq length
        true, true, true, 1e12, // very large theta
    );
    assert!(
        cfg.validate().is_ok(),
        "extreme but valid config should pass validation"
    );
    assert_eq!(cfg.head_dim(), 128);
    assert_eq!(cfg.num_kv_groups().unwrap(), 16);
}

// ===========================================================================
// 13. Parameter count estimation across GLM model sizes
// ===========================================================================

/// GLM-4-9B parameter count should be approximately 9 billion.
#[test]
fn test_param_count_glm4_9b() {
    let cfg = Glm5Config::default();
    let count = glm_param_count(&cfg);
    // GLM-4-9B: ~8.5-9.5B params
    assert!(
        count > 8_000_000_000 && count < 10_000_000_000,
        "GLM-4-9B should have ~9B params, got {count}"
    );
}

/// GLM-4-9B-chat has the same parameter count as the base model
/// (only inference config differs: rope_theta, seq_length, epsilon).
#[test]
fn test_param_count_chat_equals_base() {
    let base_count = glm_param_count(&Glm5Config::default());
    let chat_count = glm_param_count(&Glm5Config::glm4_9b_chat());
    assert_eq!(
        base_count, chat_count,
        "chat and base should have identical parameter counts"
    );
}

/// Hypothetical 1.5B model parameter count.
#[test]
fn test_param_count_hypothetical_1_5b() {
    let cfg = Glm5Config::new(
        2048, 5504, 24, 16, 2, 65536, 128, 1e-5, 8192, true, true, false, 10_000.0,
    );
    let count = glm_param_count(&cfg);
    assert!(
        count > 1_000_000_000 && count < 2_500_000_000,
        "hypothetical 1.5B should have ~1.5B params, got {count}"
    );
}

/// Hypothetical 130B model parameter count.
///
/// Uses authentic GLM-130B-class dimensions: hidden=12288, ffn=32768, 80 layers,
/// 96 query heads / 8 KV groups, head_dim=128. The previous config (hidden=8192,
/// ffn=28672, 64 heads) only yields ~70.9B params (verified against the formula),
/// so it could never satisfy the >100B assertion below.
#[test]
fn test_param_count_hypothetical_130b() {
    let cfg = Glm5Config::new(
        12288, 32768, 80, 96, 8, 151552, 128, 1e-5, 8192, true, true, false, 10_000.0,
    );
    let count = glm_param_count(&cfg);
    assert!(
        count > 100_000_000_000 && count < 200_000_000_000,
        "hypothetical 130B should have ~130B params, got {count}"
    );
}

/// Parameter count strictly increases with model size.
#[test]
fn test_param_count_monotonically_increases() {
    let sizes = [Glm5Config::new(
            2048, 5504, 24, 16, 2, 65536, 128, 1e-5, 8192, true, true, false, 10_000.0,
        ),
        Glm5Config::default(),
        Glm5Config::new(
            8192, 28672, 80, 64, 8, 151552, 128, 1e-5, 8192, true, true, false, 10_000.0,
        )];
    let counts: Vec<usize> = sizes.iter().map(glm_param_count).collect();
    for i in 0..counts.len() - 1 {
        assert!(
            counts[i] < counts[i + 1],
            "param count should increase: {} < {}",
            counts[i],
            counts[i + 1]
        );
    }
}

/// QKV bias adds approximately (nh + 2*nkv) * hd * num_layers parameters.
#[test]
fn test_param_count_bias_contribution() {
    let mut cfg_no_bias = tiny_config();
    cfg_no_bias.add_qkv_bias = false;
    cfg_no_bias.add_bias_linear = false;

    let mut cfg_qkv_bias = tiny_config();
    cfg_qkv_bias.add_qkv_bias = true;
    cfg_qkv_bias.add_bias_linear = false;

    let no_bias_count = glm_param_count(&cfg_no_bias);
    let qkv_bias_count = glm_param_count(&cfg_qkv_bias);

    let expected_qkv_bias = (cfg_qkv_bias.num_attention_heads
        + 2 * cfg_qkv_bias.multi_query_group_num)
        * cfg_qkv_bias.kv_channels
        * cfg_qkv_bias.num_layers;
    assert_eq!(
        qkv_bias_count - no_bias_count,
        expected_qkv_bias,
        "QKV bias should add exactly (nh+2*nkv)*hd*layers params"
    );
}

// ===========================================================================
// 14. Half-RoPE wavelength analysis (GLM-specific partial rotation)
// ===========================================================================

/// GLM uses HalfRotaryEmbedding: only the first head_dim/2 dimensions
/// are rotated. The inner RoPE dimension is head_dim/2, so frequencies
/// span half the indices compared to full-RoPE models like Llama/Qwen.
#[test]
fn test_half_rope_inner_dimension() {
    let cfg = Glm5Config::default();
    let head_dim = cfg.kv_channels; // 128
    let rope_inner_dim = head_dim / 2; // 64
    assert_eq!(rope_inner_dim, 64);

    // Number of frequency pairs = rope_inner_dim / 2 = 32
    let freq_pairs = rope_inner_dim / 2;
    assert_eq!(freq_pairs, 32);
}

/// The longest wavelength in GLM's half-RoPE spectrum.
/// With base=10K, inner_dim=64: theta[31] = 1/10000^(62/64).
#[test]
fn test_half_rope_longest_wavelength_base_10k() {
    let base = 10_000.0_f64;
    let inner_dim = 64; // head_dim/2 for GLM with kv_channels=128
    let last_idx = inner_dim / 2 - 1; // 31
    let theta_last = 1.0 / base.powf(f64::from(2 * last_idx) / f64::from(inner_dim));
    let wavelength = 2.0 * std::f64::consts::PI / theta_last;

    // With base=10K and inner_dim=64, the longest wavelength is significant
    assert!(
        wavelength > 10_000.0,
        "longest half-RoPE wavelength with base=10K should be > 10K positions, got {wavelength:.0}"
    );
}

/// Chat model's rope_theta=5M extends the longest wavelength dramatically.
#[test]
fn test_half_rope_longest_wavelength_chat_5m() {
    let base = 5_000_000.0_f64;
    let inner_dim = 64;
    let last_idx = inner_dim / 2 - 1; // 31
    let theta_last = 1.0 / base.powf(f64::from(2 * last_idx) / f64::from(inner_dim));
    let wavelength = 2.0 * std::f64::consts::PI / theta_last;

    // With base=5M, the longest wavelength should support 128K+ context
    assert!(
        wavelength > 100_000.0,
        "chat half-RoPE longest wavelength should support 128K context, got {wavelength:.0}"
    );
}

/// The shortest wavelength is always 2*pi regardless of base (theta[0]=1).
#[test]
fn test_half_rope_shortest_wavelength_is_2pi() {
    let theta_0 = 1.0_f64; // 1 / base^(0/dim) = 1.0
    let wavelength = 2.0 * std::f64::consts::PI / theta_0;
    assert!(
        (wavelength - 2.0 * std::f64::consts::PI).abs() < 1e-10,
        "shortest wavelength should be 2*pi, got {wavelength}"
    );
}

/// Half-RoPE frequencies are logarithmically spaced within the inner dimension.
#[test]
fn test_half_rope_log_spaced_frequencies() {
    let base = 10_000.0_f64;
    let inner_dim = 64; // GLM's half-RoPE inner dim
    let half = inner_dim / 2;

    let log_freqs: Vec<f64> = (0..half)
        .map(|i| {
            let freq = 1.0 / base.powf((2 * i) as f64 / inner_dim as f64);
            freq.ln()
        })
        .collect();

    // ln(freq[i]) = -(2i/dim) * ln(base)
    // The log-space spacing should be constant
    let expected_delta = -2.0 / inner_dim as f64 * base.ln();
    for i in 0..half - 1 {
        let delta = log_freqs[i + 1] - log_freqs[i];
        assert!(
            (delta - expected_delta).abs() < 1e-10,
            "log-space delta at {i}: expected {expected_delta}, got {delta}"
        );
    }
}

/// GLM's half-RoPE has fewer frequency components than full-RoPE with the
/// same head_dim. Full-RoPE uses head_dim/2 pairs; half-RoPE uses head_dim/4 pairs.
#[test]
fn test_half_rope_fewer_frequencies_than_full_rope() {
    let head_dim = 128;

    // Full RoPE (Llama/Qwen): head_dim/2 = 64 frequency pairs
    let full_rope_pairs = head_dim / 2;

    // Half RoPE (GLM): inner_dim = head_dim/2 = 64, then inner_dim/2 = 32 pairs
    let half_rope_inner = head_dim / 2;
    let half_rope_pairs = half_rope_inner / 2;

    assert_eq!(full_rope_pairs, 64);
    assert_eq!(half_rope_pairs, 32);
    assert_eq!(
        full_rope_pairs,
        2 * half_rope_pairs,
        "full RoPE uses 2x the frequency pairs of half RoPE"
    );
}

// ===========================================================================
// 15. GQA KV cache memory calculations (byte-level)
// ===========================================================================

/// GQA KV cache memory is proportional to multi_query_group_num, not num_attention_heads.
/// Verify the memory reduction factor for GLM-4-9B.
#[test]
fn test_gqa_kv_memory_reduction_factor_9b() {
    let cfg = Glm5Config::default();
    let kv_reduction = cfg.num_attention_heads / cfg.multi_query_group_num;
    assert_eq!(
        kv_reduction, 16,
        "GLM-4-9B: 32 Q heads / 2 KV heads = 16x memory reduction"
    );
}

/// KV cache bytes per token per layer = 2 * nkv * hd * sizeof(f32).
#[test]
fn test_kv_cache_bytes_per_token_per_layer() {
    let sizeof_f32 = 4_usize;

    let configs = [
        ("tiny", 2_usize, 64_usize), // 2 kv heads, 64 head_dim
        ("9B", 2, 128),              // 2 kv heads, 128 head_dim
        ("130B-hyp", 8, 128),        // 8 kv heads, 128 head_dim
    ];
    for (name, nkv, hd) in configs {
        let bytes = 2 * nkv * hd * sizeof_f32;
        // K + V = 2 tensors, each nkv * hd floats, each float is 4 bytes
        assert_eq!(
            bytes,
            2 * nkv * hd * sizeof_f32,
            "{name}: KV bytes/token/layer"
        );
        assert!(bytes > 0);
    }
}

/// Total KV cache size for GLM-4-9B at 8K context (base model):
/// 40 layers * 2 (K+V) * 2 kv_heads * 128 head_dim * 4 bytes * 8192 tokens.
#[test]
fn test_kv_cache_total_size_9b_8k_context() {
    let cfg = Glm5Config::default();
    let seq_len = cfg.seq_length; // 8192
    let nkv = cfg.multi_query_group_num; // 2
    let hd = cfg.kv_channels; // 128
    let sizeof_f32 = 4_usize;

    let total_bytes = cfg.num_layers * 2 * nkv * hd * sizeof_f32 * seq_len;
    let mib = total_bytes as f64 / (1024.0 * 1024.0);

    // 40 * 2 * 2 * 128 * 4 * 8192 = 671,088,640 bytes = 640 MiB
    assert_eq!(total_bytes, 671_088_640);
    assert!(
        (mib - 640.0).abs() < 1.0,
        "GLM-4-9B 8K context KV cache should be ~640 MiB, got {mib:.0} MiB"
    );
}

/// Total KV cache size for GLM-4-9B-chat at 128K context.
/// Much larger due to extended context window.
#[test]
fn test_kv_cache_total_size_9b_128k_context() {
    let cfg = Glm5Config::glm4_9b_chat();
    let seq_len = cfg.seq_length; // 131072
    let nkv = cfg.multi_query_group_num; // 2
    let hd = cfg.kv_channels; // 128
    let sizeof_f32 = 4_usize;

    let total_bytes = cfg.num_layers * 2 * nkv * hd * sizeof_f32 * seq_len;
    let gib = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    // 128K context = 16x the 8K cache
    assert!(
        gib > 9.0 && gib < 11.0,
        "GLM-4-9B-chat 128K context KV cache should be ~10 GiB, got {gib:.1} GiB"
    );
}

/// GQA provides massive memory savings. Compare MHA vs GQA KV cache.
#[test]
fn test_gqa_vs_mha_kv_cache_savings() {
    let cfg = Glm5Config::default();
    let seq_len = 4096_usize;
    let sizeof_f32 = 4_usize;

    // GQA: 2 KV heads
    let gqa_bytes =
        cfg.num_layers * 2 * cfg.multi_query_group_num * cfg.kv_channels * sizeof_f32 * seq_len;
    // MHA: 32 KV heads (= num_attention_heads)
    let mha_bytes =
        cfg.num_layers * 2 * cfg.num_attention_heads * cfg.kv_channels * sizeof_f32 * seq_len;

    let savings_factor = mha_bytes / gqa_bytes;
    assert_eq!(
        savings_factor, 16,
        "GQA should save 16x memory vs MHA for GLM-4-9B"
    );

    let gqa_mib = gqa_bytes as f64 / (1024.0 * 1024.0);
    let mha_mib = mha_bytes as f64 / (1024.0 * 1024.0);
    assert!(
        gqa_mib < 400.0,
        "GQA KV cache for 4K context should be < 400 MiB"
    );
    assert!(
        mha_mib > 4000.0,
        "MHA KV cache for 4K context would be > 4 GiB"
    );
}

// ===========================================================================
// 16. Attention scale factor consistency
// ===========================================================================

/// Attention scale = 1/sqrt(head_dim). For GLM with kv_channels=128, this is constant.
#[test]
fn test_attention_scale_factor_128() {
    let scale = 1.0 / (128.0_f64).sqrt();
    // ~0.08838834764831843
    assert!(
        (scale - 0.08838834764831843).abs() < 1e-15,
        "scale should be 1/sqrt(128) = 0.08839, got {scale}"
    );
}

/// Attention scale varies with head_dim (kv_channels).
#[test]
fn test_attention_scale_varies_with_head_dim() {
    let scales: Vec<(usize, f64)> = vec![
        (32, 1.0 / (32.0_f64).sqrt()),
        (64, 1.0 / (64.0_f64).sqrt()),
        (128, 1.0 / (128.0_f64).sqrt()),
        (256, 1.0 / (256.0_f64).sqrt()),
    ];

    for (hd, expected_scale) in &scales {
        let mut cfg = tiny_config();
        cfg.kv_channels = *hd;
        cfg.hidden_size = cfg.num_attention_heads * *hd;
        let computed_scale = 1.0 / (cfg.head_dim() as f64).sqrt();
        assert!(
            (computed_scale - expected_scale).abs() < 1e-15,
            "head_dim={hd}: scale should be {expected_scale}, got {computed_scale}"
        );
    }
}

/// Scale factor is independent of model size (hidden_size, num_layers, etc.).
#[test]
fn test_attention_scale_independent_of_model_size() {
    let tiny_scale = 1.0 / (tiny_config().head_dim() as f64).sqrt();
    let big_scale = 1.0 / (Glm5Config::default().head_dim() as f64).sqrt();

    // tiny has kv_channels=64, 9B has kv_channels=128 -- different
    assert_ne!(
        tiny_config().kv_channels,
        Glm5Config::default().kv_channels,
        "tiny and 9B have different head dims"
    );
    // So their scales differ
    assert!(
        (tiny_scale - big_scale).abs() > 0.01,
        "different head_dims produce different scales"
    );
}

// ===========================================================================
// 17. Hidden_size / heads / kv_channels relationship
// ===========================================================================

/// For standard GLM configs, hidden_size = num_attention_heads * kv_channels.
#[test]
fn test_standard_configs_satisfy_hidden_heads_kv_relationship() {
    let cfgs = vec![
        ("tiny", tiny_config()),
        ("9B", Glm5Config::default()),
        ("chat", Glm5Config::glm4_9b_chat()),
    ];
    for (name, cfg) in &cfgs {
        assert_eq!(
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.kv_channels,
            "{name}: hidden_size = nh * hd"
        );
    }
}

/// GLM validation does NOT enforce hidden_size == nh * kv_channels.
/// This is by design (allows non-standard configs).
#[test]
fn test_non_standard_hidden_size_accepted() {
    let mut cfg = tiny_config();
    cfg.hidden_size = 300; // 300 != 4 * 64 = 256
    assert!(
        cfg.validate().is_ok(),
        "non-standard hidden_size should still validate"
    );
}

/// Total Q projection dimension = num_attention_heads * kv_channels.
/// This should equal hidden_size for standard configs (square dense matrix).
#[test]
fn test_q_projection_dim_standard_configs() {
    let cfg = Glm5Config::default();
    let q_dim = cfg.num_attention_heads * cfg.kv_channels;
    assert_eq!(
        q_dim, cfg.hidden_size,
        "Q projection dim should equal hidden_size for GLM-4-9B"
    );
    assert_eq!(q_dim, 4096);
}

// ===========================================================================
// 18. Layernorm epsilon precision requirements across variants
// ===========================================================================

/// Base and chat use different epsilon values by two orders of magnitude.
#[test]
fn test_layernorm_epsilon_precision_difference() {
    let base_eps = Glm5Config::default().layernorm_epsilon;
    let chat_eps = Glm5Config::glm4_9b_chat().layernorm_epsilon;

    assert_eq!(base_eps, 1.5625e-5);
    assert_eq!(chat_eps, 1.5625e-7);

    let ratio = base_eps / chat_eps;
    assert!(
        (ratio - 100.0).abs() < 0.01,
        "base epsilon should be 100x chat epsilon, got ratio {ratio}"
    );
}

/// Very small epsilon values are valid for high-precision inference.
#[test]
fn test_layernorm_epsilon_range_boundaries() {
    let valid_epsilons = [1e-15, 1e-12, 1e-8, 1e-6, 1e-5, 1e-3, 0.1, 0.99];
    for eps in valid_epsilons {
        let mut cfg = tiny_config();
        cfg.layernorm_epsilon = eps;
        assert!(cfg.validate().is_ok(), "epsilon={eps} should validate");
    }

    let invalid_epsilons = [
        0.0,
        -1.0,
        -1e-10,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    for eps in invalid_epsilons {
        let mut cfg = tiny_config();
        cfg.layernorm_epsilon = eps;
        assert!(cfg.validate().is_err(), "epsilon={eps} should be rejected");
    }
}

// ===========================================================================
// 19. Weight memory estimation per layer
// ===========================================================================

/// Per-layer weight memory in bytes for GLM-4-9B.
#[test]
fn test_per_layer_weight_memory_9b() {
    let cfg = Glm5Config::default();
    let h = cfg.hidden_size;
    let ffn = cfg.ffn_hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.multi_query_group_num;
    let hd = cfg.kv_channels;
    let sizeof_f32 = 4_usize;

    // QKV weight: (nh + 2*nkv) * hd * h
    let qkv = (nh + 2 * nkv) * hd * h;
    // QKV bias: (nh + 2*nkv) * hd (since add_qkv_bias=true)
    let qkv_bias = (nh + 2 * nkv) * hd;
    // Dense: h * nh * hd
    let dense = h * nh * hd;
    // MLP: ffn*2*h + h*ffn
    let mlp = ffn * 2 * h + h * ffn;
    // Norms: 2*h
    let norms = 2 * h;

    let per_layer_params = qkv + qkv_bias + dense + mlp + norms;
    let per_layer_bytes = per_layer_params * sizeof_f32;
    let per_layer_mib = per_layer_bytes as f64 / (1024.0 * 1024.0);

    // Each layer should be ~200-300 MiB in fp32
    assert!(
        per_layer_mib > 500.0 && per_layer_mib < 1200.0,
        "GLM-4-9B per-layer weight memory should be ~800 MiB in fp32, got {per_layer_mib:.0} MiB"
    );
}

/// Total model weight memory (fp32) for GLM-4-9B.
#[test]
fn test_total_weight_memory_9b_fp32() {
    let cfg = Glm5Config::default();
    let total_params = glm_param_count(&cfg);
    let total_bytes = total_params * 4; // f32
    let gib = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    // 9B params * 4 bytes = ~36 GiB in fp32
    assert!(
        gib > 30.0 && gib < 45.0,
        "GLM-4-9B should require ~36 GiB in fp32, got {gib:.1} GiB"
    );
}

/// BF16/F16 halves the weight memory requirement.
#[test]
fn test_weight_memory_bf16_half_of_fp32() {
    let cfg = Glm5Config::default();
    let total_params = glm_param_count(&cfg);
    let fp32_gib = (total_params * 4) as f64 / (1024.0 * 1024.0 * 1024.0);
    let bf16_gib = (total_params * 2) as f64 / (1024.0 * 1024.0 * 1024.0);

    assert!(
        (fp32_gib / bf16_gib - 2.0).abs() < 0.01,
        "BF16 should be exactly half of FP32: fp32={fp32_gib:.1}, bf16={bf16_gib:.1}"
    );
    assert!(bf16_gib < 20.0, "BF16 should be under 20 GiB for 9B model");
}

// ===========================================================================
// 20. Forward pass shape consistency across config variations
// ===========================================================================

/// Forward pass with add_bias_linear=true still produces correct output shape.
#[test]
fn test_forward_with_bias_linear_correct_shape() {
    let mut cfg = tiny_config();
    cfg.add_bias_linear = true;
    let model = load_model_with_config(cfg.clone());
    let logits = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
    assert_eq!(logits.dims(), &[1, 3, cfg.padded_vocab_size]);
}

/// Forward pass with MQA (1 KV head) produces correct output shape.
#[test]
fn test_forward_mqa_correct_shape() {
    let mut cfg = tiny_config();
    cfg.multi_query_group_num = 1;
    let model = load_model_with_config(cfg.clone());
    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.padded_vocab_size]);
}

/// Forward pass with MHA (4 KV heads = num_attention_heads) produces correct output shape.
#[test]
fn test_forward_mha_correct_shape() {
    let mut cfg = tiny_config();
    cfg.multi_query_group_num = 4; // equals num_attention_heads
    let model = load_model_with_config(cfg.clone());
    let logits = model.forward(&[0, 1], &[0, 1]).unwrap();
    assert_eq!(logits.dims(), &[1, 2, cfg.padded_vocab_size]);
}

/// Forward pass with add_qkv_bias=false produces correct output shape.
#[test]
fn test_forward_no_qkv_bias_correct_shape() {
    let mut cfg = tiny_config();
    cfg.add_qkv_bias = false;
    let model = load_model_with_config(cfg.clone());
    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.padded_vocab_size]);
}

/// Forward pass with all biases enabled produces correct output shape.
#[test]
fn test_forward_all_biases_correct_shape() {
    let mut cfg = tiny_config();
    cfg.add_qkv_bias = true;
    cfg.add_bias_linear = true;
    let model = load_model_with_config(cfg.clone());
    let logits = model.forward(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4]).unwrap();
    assert_eq!(logits.dims(), &[1, 5, cfg.padded_vocab_size]);
}

/// input_ids/positions length mismatch rejected.
#[test]
fn test_forward_ids_positions_length_mismatch() {
    let model = load_tiny_model();
    let result = model.forward(&[0, 1, 2], &[0, 1]); // 3 ids, 2 positions
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("input_ids") || msg.contains("positions"),
        "error should mention input/positions mismatch: {msg}"
    );
}

/// forward_from_embeddings with wrong hidden_size errors.
#[test]
fn test_forward_from_embeddings_wrong_hidden_size() {
    let model = load_tiny_model();
    let bad_emb = DynTensor::ones(&[1, 2, 128], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&bad_emb, &[0, 1], None);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("hidden_size"),
        "error should mention hidden_size mismatch: {msg}"
    );
}
