// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GLM-5 configuration construction, defaults, validation, and model-size
//! variant tests.
//!
//! Part of #4186. Covers:
//! - Default config matches GLM-4-9B HuggingFace spec
//! - `glm4_9b_chat()` preset differences from base
//! - `Glm5Config::new()` constructor field preservation
//! - Validation rejects invalid hidden_size, ffn_hidden_size, head_dim mismatches
//! - Config invariant: hidden_size == num_heads * kv_channels
//! - Derived accessor consistency (head_dim, num_kv_groups)
//! - Edge-case configs: MQA, MHA, large layer counts

use super::*;
use crate::test_utils::tiny_config;

// ---------------------------------------------------------------------------
// Default config matches GLM-4-9B HuggingFace specification
// ---------------------------------------------------------------------------

#[test]
fn test_default_config_validates_and_matches_hf_spec() {
    let cfg = Glm5Config::default();
    assert!(cfg.validate().is_ok(), "GLM-4-9B default must validate");
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.ffn_hidden_size, 13696);
    assert_eq!(cfg.num_layers, 40);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.multi_query_group_num, 2);
    assert_eq!(cfg.padded_vocab_size, 151552);
    assert_eq!(cfg.kv_channels, 128);
    assert!(cfg.rmsnorm);
    assert!(cfg.add_qkv_bias);
    assert!(!cfg.add_bias_linear);
}

#[test]
fn test_default_config_head_dim_equals_kv_channels() {
    let cfg = Glm5Config::default();
    assert_eq!(
        cfg.head_dim(),
        cfg.kv_channels,
        "head_dim() should return kv_channels"
    );
    assert_eq!(cfg.head_dim(), 128);
}

// ---------------------------------------------------------------------------
// glm4_9b_chat() preset differs from default in specific fields
// ---------------------------------------------------------------------------

#[test]
fn test_chat_config_differs_from_base_in_expected_fields() {
    let base = Glm5Config::default();
    let chat = Glm5Config::glm4_9b_chat();

    // Fields that should differ
    assert_ne!(
        base.layernorm_epsilon, chat.layernorm_epsilon,
        "chat uses different layernorm_epsilon"
    );
    assert_ne!(
        base.rope_theta, chat.rope_theta,
        "chat uses rope_ratio=500 → rope_theta=5M"
    );
    assert_ne!(
        base.seq_length, chat.seq_length,
        "chat supports longer sequences"
    );

    // Verify the actual chat values
    assert_eq!(chat.layernorm_epsilon, 1.5625e-7);
    assert_eq!(chat.rope_theta, 5_000_000.0);
    assert_eq!(chat.seq_length, 131_072);

    // Fields that should be the same
    assert_eq!(base.hidden_size, chat.hidden_size);
    assert_eq!(base.ffn_hidden_size, chat.ffn_hidden_size);
    assert_eq!(base.num_layers, chat.num_layers);
    assert_eq!(base.num_attention_heads, chat.num_attention_heads);
    assert_eq!(base.multi_query_group_num, chat.multi_query_group_num);
    assert_eq!(base.padded_vocab_size, chat.padded_vocab_size);
    assert_eq!(base.kv_channels, chat.kv_channels);
}

#[test]
fn test_chat_config_validates() {
    let chat = Glm5Config::glm4_9b_chat();
    assert!(chat.validate().is_ok(), "glm4_9b_chat must validate");
    assert_eq!(chat.num_kv_groups().unwrap(), 16);
}

// ---------------------------------------------------------------------------
// Config::new() constructor preserves all fields
// ---------------------------------------------------------------------------

#[test]
fn test_config_new_preserves_all_fields_roundtrip() {
    let cfg = Glm5Config::new(
        512,       // hidden_size
        2048,      // ffn_hidden_size
        8,         // num_layers
        8,         // num_attention_heads
        4,         // multi_query_group_num
        65536,     // padded_vocab_size
        64,        // kv_channels
        1e-6,      // layernorm_epsilon
        4096,      // seq_length
        false,     // rmsnorm
        false,     // add_qkv_bias
        true,      // add_bias_linear
        100_000.0, // rope_theta
    );
    assert_eq!(cfg.hidden_size, 512);
    assert_eq!(cfg.ffn_hidden_size, 2048);
    assert_eq!(cfg.num_layers, 8);
    assert_eq!(cfg.num_attention_heads, 8);
    assert_eq!(cfg.multi_query_group_num, 4);
    assert_eq!(cfg.padded_vocab_size, 65536);
    assert_eq!(cfg.kv_channels, 64);
    assert_eq!(cfg.layernorm_epsilon, 1e-6);
    assert_eq!(cfg.seq_length, 4096);
    assert!(!cfg.rmsnorm);
    assert!(!cfg.add_qkv_bias);
    assert!(cfg.add_bias_linear);
    assert_eq!(cfg.rope_theta, 100_000.0);
    assert!(cfg.validate().is_ok());
}

// ---------------------------------------------------------------------------
// Validation rejects invalid configurations
// ---------------------------------------------------------------------------

#[test]
fn test_validate_rejects_zero_hidden_size() {
    let mut cfg = tiny_config();
    cfg.hidden_size = 0;
    let err = cfg.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("hidden_size"),
        "error should mention hidden_size: {msg}"
    );
}

#[test]
fn test_validate_rejects_zero_ffn_hidden_size() {
    let mut cfg = tiny_config();
    cfg.ffn_hidden_size = 0;
    let err = cfg.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("ffn_hidden_size"),
        "error should mention ffn_hidden_size: {msg}"
    );
}

#[test]
fn test_validate_rejects_head_dim_not_multiple_of_4() {
    // HalfRotaryEmbedding requires kv_channels divisible by 4
    let mut cfg = tiny_config();
    cfg.kv_channels = 7;
    let err = cfg.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("kv_channels") && msg.contains("multiple of 4"),
        "error should explain the kv_channels constraint: {msg}"
    );
}

#[test]
fn test_validate_rejects_negative_layernorm_epsilon() {
    let mut cfg = tiny_config();
    cfg.layernorm_epsilon = -1e-5;
    let err = cfg.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("layernorm_epsilon") && msg.contains("positive"),
        "error should explain epsilon must be positive: {msg}"
    );
}

#[test]
fn test_validate_rejects_inf_layernorm_epsilon() {
    let mut cfg = tiny_config();
    cfg.layernorm_epsilon = f64::INFINITY;
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// Config invariant: hidden_size == num_heads * kv_channels for GLM
// ---------------------------------------------------------------------------

#[test]
fn test_hidden_size_equals_num_heads_times_kv_channels() {
    // This is the fundamental GLM architectural invariant
    for (hidden, heads, kv_ch) in [(256, 4, 64), (4096, 32, 128), (512, 8, 64), (128, 2, 64)] {
        assert_eq!(
            hidden,
            heads * kv_ch,
            "hidden_size={hidden} should equal num_heads={heads} * kv_channels={kv_ch}"
        );
    }
}

// ---------------------------------------------------------------------------
// Derived accessor consistency
// ---------------------------------------------------------------------------

#[test]
fn test_num_kv_groups_for_gqa_config() {
    // GLM-4-9B uses GQA with 32 Q heads and 2 KV heads
    let cfg = Glm5Config::default();
    let groups = cfg.num_kv_groups().unwrap();
    assert_eq!(groups, 16, "32 Q heads / 2 KV heads = 16 groups");
}

#[test]
fn test_num_kv_groups_for_mqa_config() {
    // MQA: single KV head shared across all Q heads
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 8;
    cfg.multi_query_group_num = 1;
    let groups = cfg.num_kv_groups().unwrap();
    assert_eq!(groups, 8, "MQA: 8 Q heads / 1 KV head = 8 groups");
}

#[test]
fn test_num_kv_groups_for_mha_config() {
    // MHA: every Q head has its own KV head
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 4;
    cfg.multi_query_group_num = 4;
    let groups = cfg.num_kv_groups().unwrap();
    assert_eq!(
        groups, 1,
        "MHA: 4 Q heads / 4 KV heads = 1 group (no repeat)"
    );
}

#[test]
fn test_num_kv_groups_rejects_indivisible_heads() {
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 7;
    cfg.multi_query_group_num = 3;
    let err = cfg.num_kv_groups().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("7") && msg.contains("3"),
        "error should mention both values: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Edge-case configs: extreme but valid configurations
// ---------------------------------------------------------------------------

#[test]
fn test_single_head_single_layer_config_validates() {
    let cfg = Glm5Config::new(
        64,      // hidden_size
        128,     // ffn_hidden_size
        1,       // num_layers
        1,       // num_attention_heads
        1,       // multi_query_group_num
        16,      // padded_vocab_size
        64,      // kv_channels
        1e-5,    // layernorm_epsilon
        16,      // seq_length
        true,    // rmsnorm
        false,   // add_qkv_bias
        false,   // add_bias_linear
        10000.0, // rope_theta
    );
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.head_dim(), 64);
    assert_eq!(cfg.num_kv_groups().unwrap(), 1);
}

#[test]
fn test_large_layer_count_config_validates() {
    let mut cfg = tiny_config();
    cfg.num_layers = 128;
    assert!(
        cfg.validate().is_ok(),
        "large layer count should be valid structurally"
    );
}

#[test]
fn test_very_large_rope_theta_validates() {
    // Some models use extremely large rope_theta for long context
    let mut cfg = tiny_config();
    cfg.rope_theta = 1e12;
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_very_small_layernorm_epsilon_validates() {
    let mut cfg = tiny_config();
    cfg.layernorm_epsilon = 1e-12;
    assert!(cfg.validate().is_ok());
}

// ---------------------------------------------------------------------------
// QKV fused dimension formula validation
// ---------------------------------------------------------------------------

#[test]
fn test_qkv_fused_dimension_formula_across_configs() {
    // Fused QKV weight row count: (nh + 2 * nkv) * hd
    let configs = vec![
        ("tiny", tiny_config()),
        ("9B", Glm5Config::default()),
        ("chat", Glm5Config::glm4_9b_chat()),
    ];
    for (name, cfg) in &configs {
        let nh = cfg.num_attention_heads;
        let nkv = cfg.multi_query_group_num;
        let hd = cfg.kv_channels;
        let fused = (nh + 2 * nkv) * hd;
        let q = nh * hd;
        let k = nkv * hd;
        let v = nkv * hd;
        assert_eq!(fused, q + k + v, "{name}: fused QKV = Q + K + V");
    }
}
