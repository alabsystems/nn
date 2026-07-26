// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Architecture validation tests for GLM-4/5 model (#3942).
//!
//! Validates structural invariants across GLM production configs: layer counts,
//! parameter count estimates, GQA group arithmetic, hidden_size consistency,
//! shape propagation, KV cache sizing, and config properties.

use super::*;
use crate::test_utils::tiny_config;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// Production config constructors
// ---------------------------------------------------------------------------

/// GLM-4-9B from HuggingFace THUDM/glm-4-9b config.json
fn glm4_9b() -> Glm5Config {
    Glm5Config::new(
        4096, 13696, 40, 32, 2, 151552, 128, 1.5625e-5, 8192, true, true, false, 10_000.0,
    )
}

/// GLM-4-9B-chat-1M (extended context) — same architecture, different seq_length
fn glm4_9b_1m() -> Glm5Config {
    Glm5Config::new(
        4096, 13696, 40, 32, 2, 151552, 128, 1.5625e-5, 1_048_576, true, true, false, 500_000.0,
    )
}

// ---------------------------------------------------------------------------
// All production configs validate
// ---------------------------------------------------------------------------

#[test]
fn test_glm4_9b_validates() {
    glm4_9b().validate().expect("GLM-4-9B should validate");
}

#[test]
fn test_glm4_9b_1m_validates() {
    glm4_9b_1m()
        .validate()
        .expect("GLM-4-9B-1M should validate");
}

#[test]
fn test_default_config_is_glm4_9b() {
    let default = Glm5Config::default();
    let expected = glm4_9b();
    assert_eq!(default.hidden_size, expected.hidden_size);
    assert_eq!(default.ffn_hidden_size, expected.ffn_hidden_size);
    assert_eq!(default.num_layers, expected.num_layers);
    assert_eq!(default.num_attention_heads, expected.num_attention_heads);
    assert_eq!(
        default.multi_query_group_num,
        expected.multi_query_group_num
    );
    assert_eq!(default.padded_vocab_size, expected.padded_vocab_size);
    assert_eq!(default.kv_channels, expected.kv_channels);
}

// ---------------------------------------------------------------------------
// Layer counts
// ---------------------------------------------------------------------------

#[test]
fn test_glm4_9b_layer_count() {
    assert_eq!(glm4_9b().num_layers, 40, "GLM-4-9B has 40 layers");
}

#[test]
fn test_tiny_config_layer_count() {
    assert_eq!(tiny_config().num_layers, 2, "tiny config has 2 layers");
}

// ---------------------------------------------------------------------------
// Head dimension and GQA group calculations
// ---------------------------------------------------------------------------

#[test]
fn test_glm4_9b_head_dim() {
    let cfg = glm4_9b();
    assert_eq!(cfg.head_dim(), 128, "GLM-4-9B kv_channels = 128");
}

#[test]
fn test_glm4_9b_gqa_groups() {
    let cfg = glm4_9b();
    // 32 attention heads / 2 multi_query_group_num = 16 GQA groups
    assert_eq!(cfg.num_kv_groups().unwrap(), 16);
}

#[test]
fn test_tiny_config_gqa_groups() {
    let cfg = tiny_config();
    // 4 attention heads / 2 multi_query_group_num = 2 GQA groups
    assert_eq!(cfg.num_kv_groups().unwrap(), 2);
}

// ---------------------------------------------------------------------------
// hidden_size = num_attention_heads * kv_channels for GLM
// ---------------------------------------------------------------------------

#[test]
fn test_hidden_size_equals_heads_times_kv_channels() {
    let configs: Vec<(&str, Glm5Config)> = vec![
        ("tiny", tiny_config()),
        ("GLM-4-9B", glm4_9b()),
        ("GLM-4-9B-1M", glm4_9b_1m()),
    ];
    for (name, cfg) in &configs {
        assert_eq!(
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.kv_channels,
            "{name}: hidden_size should equal num_attention_heads * kv_channels"
        );
    }
}

// ---------------------------------------------------------------------------
// Parameter count estimation and monotonicity
// ---------------------------------------------------------------------------

/// Rough parameter count: embedding + per-layer (QKV fused + dense + MLP + norms).
fn estimate_glm5_params(c: &Glm5Config) -> usize {
    let embed = c.padded_vocab_size * c.hidden_size;
    // QKV fused: [hidden_size, (num_attention_heads + 2 * multi_query_group_num) * kv_channels]
    let qkv_out = (c.num_attention_heads + 2 * c.multi_query_group_num) * c.kv_channels;
    let per_layer_attn = c.hidden_size * qkv_out       // QKV fused
        + c.hidden_size * c.hidden_size; // dense (output proj)
                                         // MLP: dense_h_to_4h is [hidden, 2 * ffn_hidden] (gate+up fused), dense_4h_to_h is [ffn_hidden, hidden]
    let per_layer_mlp = c.hidden_size * 2 * c.ffn_hidden_size + c.ffn_hidden_size * c.hidden_size;
    let per_layer_norms = 2 * c.hidden_size; // 2 RMSNorm weights
    let per_layer = per_layer_attn + per_layer_mlp + per_layer_norms;
    embed + per_layer * c.num_layers
}

#[test]
fn test_tiny_config_fewer_params_than_production() {
    let tiny_params = estimate_glm5_params(&tiny_config());
    let prod_params = estimate_glm5_params(&glm4_9b());
    assert!(
        tiny_params < prod_params,
        "tiny config ({tiny_params}) should have fewer params than GLM-4-9B ({prod_params})"
    );
}

// ---------------------------------------------------------------------------
// Shape propagation: forward pass output
// ---------------------------------------------------------------------------

#[test]
fn test_forward_output_shape_single_token() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg.clone()).unwrap();
    let logits = model.forward(&[42], &[0]).unwrap();
    assert_eq!(
        logits.dims(),
        &[1, 1, cfg.padded_vocab_size],
        "single token -> [1, 1, padded_vocab_size]"
    );
}

#[test]
fn test_forward_output_shape_multi_token() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg.clone()).unwrap();
    let logits = model.forward(&[0, 1, 2, 3], &[0, 1, 2, 3]).unwrap();
    assert_eq!(
        logits.dims(),
        &[1, 4, cfg.padded_vocab_size],
        "4 tokens -> [1, 4, padded_vocab_size]"
    );
}

// ---------------------------------------------------------------------------
// KV cache sizing matches model config
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_num_layers_matches_config() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg.clone()).unwrap();
    let cache = model.new_cache();
    assert_eq!(
        cache.num_layers(),
        cfg.num_layers,
        "KV cache should have num_layers entries"
    );
}

// ---------------------------------------------------------------------------
// Config accessor roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_model_config_accessor_returns_original() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg.clone()).unwrap();
    let model_cfg = model.config();
    assert_eq!(model_cfg.hidden_size, cfg.hidden_size);
    assert_eq!(model_cfg.num_layers, cfg.num_layers);
    assert_eq!(model_cfg.padded_vocab_size, cfg.padded_vocab_size);
    assert_eq!(model_cfg.kv_channels, cfg.kv_channels);
}

// ---------------------------------------------------------------------------
// Model dtype accessor
// ---------------------------------------------------------------------------

#[test]
fn test_model_dtype_matches_vb_f32() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    assert_eq!(model.dtype(), DType::F32);
}

#[test]
fn test_model_dtype_matches_vb_bf16() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    assert_eq!(model.dtype(), DType::BF16);
}

// ---------------------------------------------------------------------------
// Config clone independence
// ---------------------------------------------------------------------------

#[test]
fn test_config_clone_independence() {
    let c1 = glm4_9b();
    let mut c2 = c1.clone();
    c2.hidden_size = 9999;
    assert_eq!(c1.hidden_size, 4096, "original should be unchanged");
    assert_eq!(c2.hidden_size, 9999);
}

// ---------------------------------------------------------------------------
// Config Debug format includes key fields
// ---------------------------------------------------------------------------

#[test]
fn test_config_debug_contains_key_fields() {
    let c = tiny_config();
    let debug = format!("{c:?}");
    assert!(
        debug.contains("hidden_size"),
        "Debug should contain hidden_size"
    );
    assert!(
        debug.contains("num_layers"),
        "Debug should contain num_layers"
    );
    assert!(
        debug.contains("num_attention_heads"),
        "Debug should contain num_attention_heads"
    );
    assert!(
        debug.contains("kv_channels"),
        "Debug should contain kv_channels"
    );
}

// ---------------------------------------------------------------------------
// kv_channels must be multiple of 4 (HalfRotaryEmbedding requirement)
// ---------------------------------------------------------------------------

#[test]
fn test_kv_channels_multiple_of_4_for_all_configs() {
    let configs: Vec<(&str, Glm5Config)> = vec![
        ("tiny", tiny_config()),
        ("GLM-4-9B", glm4_9b()),
        ("GLM-4-9B-1M", glm4_9b_1m()),
    ];
    for (name, cfg) in &configs {
        assert!(
            cfg.kv_channels.is_multiple_of(4),
            "{name}: kv_channels ({}) must be a multiple of 4",
            cfg.kv_channels
        );
    }
}

// ---------------------------------------------------------------------------
// Partial RoPE: GLM uses half rotation (architectural difference from Qwen3)
// ---------------------------------------------------------------------------

#[test]
fn test_glm_rope_theta_10k_default() {
    let cfg = glm4_9b();
    assert!(
        (cfg.rope_theta - 10_000.0).abs() < 1e-3,
        "GLM-4-9B default rope_theta is 10,000"
    );
}

#[test]
fn test_glm_1m_rope_theta_500k() {
    let cfg = glm4_9b_1m();
    assert!(
        (cfg.rope_theta - 500_000.0).abs() < 1e-3,
        "GLM-4-9B-1M uses 500K rope_theta for extended context"
    );
}

// ---------------------------------------------------------------------------
// QKV bias configuration
// ---------------------------------------------------------------------------

#[test]
fn test_glm4_has_qkv_bias() {
    let cfg = glm4_9b();
    assert!(
        cfg.add_qkv_bias,
        "GLM-4-9B has QKV bias (architectural requirement)"
    );
    assert!(
        !cfg.add_bias_linear,
        "GLM-4-9B does not use bias on other linears"
    );
}
