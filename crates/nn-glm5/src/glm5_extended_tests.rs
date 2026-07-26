// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended GLM-5 model component tests.
//!
//! Part of #4186. Covers configuration defaults, architectural constraints,
//! GQA invariants, MLP ratio relationships, special token capacity, and
//! max position embedding bounds across GLM-4/5 model variants.

use super::*;
use crate::test_utils::tiny_config;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::kv_cache::KvCache;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// Configuration defaults
// ---------------------------------------------------------------------------

#[test]
fn test_glm5_config_defaults() {
    let cfg = Glm5Config::default();
    assert!(cfg.validate().is_ok(), "default config must validate");
    // GLM-4-9B defaults from HuggingFace config.json
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.ffn_hidden_size, 13696);
    assert_eq!(cfg.num_layers, 40);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.multi_query_group_num, 2);
    assert_eq!(cfg.padded_vocab_size, 151552);
    assert_eq!(cfg.kv_channels, 128);
    assert_eq!(cfg.layernorm_epsilon, 1.5625e-5);
    assert_eq!(cfg.seq_length, 8192);
    assert!(cfg.rmsnorm);
    assert!(cfg.add_qkv_bias);
    assert!(!cfg.add_bias_linear);
    assert_eq!(cfg.rope_theta, 10_000.0);
}

#[test]
fn test_glm5_hidden_size() {
    let cfg = Glm5Config::default();
    // hidden_size must be reasonable for a 9B-class model
    assert!(
        cfg.hidden_size >= 1024,
        "hidden_size should be >= 1024 for a production model, got {}",
        cfg.hidden_size
    );
    assert!(
        cfg.hidden_size <= 16384,
        "hidden_size should be <= 16384, got {}",
        cfg.hidden_size
    );
    // Must equal num_heads * head_dim (fundamental GLM invariant)
    assert_eq!(
        cfg.hidden_size,
        cfg.num_attention_heads * cfg.kv_channels,
        "hidden_size must equal num_attention_heads * kv_channels"
    );

    // Chat variant preserves the same hidden_size
    let chat = Glm5Config::glm4_9b_chat();
    assert_eq!(chat.hidden_size, cfg.hidden_size);
}

#[test]
fn test_glm5_num_layers() {
    let cfg = Glm5Config::default();
    assert!(cfg.num_layers > 0, "num_layers must be positive");
    assert_eq!(cfg.num_layers, 40, "GLM-4-9B has 40 transformer layers");

    // Chat variant preserves the same layer count
    let chat = Glm5Config::glm4_9b_chat();
    assert_eq!(chat.num_layers, cfg.num_layers);

    // Validation rejects zero layers
    let mut bad = tiny_config();
    bad.num_layers = 0;
    assert!(bad.validate().is_err(), "zero layers must be rejected");
}

#[test]
fn test_glm5_head_dim() {
    let cfg = Glm5Config::default();
    // head_dim = hidden_size / num_heads = kv_channels
    assert_eq!(
        cfg.head_dim(),
        cfg.kv_channels,
        "head_dim() should return kv_channels"
    );
    assert_eq!(
        cfg.head_dim(),
        cfg.hidden_size / cfg.num_attention_heads,
        "head_dim should equal hidden_size / num_attention_heads"
    );
    assert_eq!(cfg.head_dim(), 128, "GLM-4-9B head_dim is 128");

    // Tiny config also satisfies the invariant
    let tiny = tiny_config();
    assert_eq!(
        tiny.head_dim(),
        tiny.hidden_size / tiny.num_attention_heads,
        "tiny config: head_dim = hidden_size / num_heads"
    );
}

#[test]
fn test_glm5_vocab_size() {
    let cfg = Glm5Config::default();
    assert!(
        cfg.padded_vocab_size > 30_000,
        "vocab_size should be > 30000 for a multilingual model, got {}",
        cfg.padded_vocab_size
    );
    assert_eq!(
        cfg.padded_vocab_size, 151552,
        "GLM-4-9B uses padded_vocab_size=151552"
    );

    // Chat variant preserves the same vocab
    let chat = Glm5Config::glm4_9b_chat();
    assert_eq!(chat.padded_vocab_size, cfg.padded_vocab_size);

    // Vocab is padded: 151552 is divisible by 64 for efficient GPU dispatch
    assert_eq!(
        cfg.padded_vocab_size % 64,
        0,
        "padded_vocab_size should be aligned to 64"
    );
}

// ---------------------------------------------------------------------------
// Architecture: RoPE
// ---------------------------------------------------------------------------

#[test]
fn test_glm5_rope_base() {
    let cfg = Glm5Config::default();
    // Base model uses standard theta=10000
    assert_eq!(cfg.rope_theta, 10_000.0, "base model uses theta=10000");
    assert!(cfg.rope_theta > 0.0, "rope_theta must be positive");
    assert!(cfg.rope_theta.is_finite(), "rope_theta must be finite");

    // Chat variant uses extended context via rope_ratio=500
    let chat = Glm5Config::glm4_9b_chat();
    assert_eq!(
        chat.rope_theta, 5_000_000.0,
        "chat uses rope_ratio=500 -> theta=5M"
    );
    assert!(
        chat.rope_theta > cfg.rope_theta,
        "chat theta should be larger than base for longer context"
    );

    // Validation rejects non-positive and non-finite theta
    let mut bad = tiny_config();
    bad.rope_theta = 0.0;
    assert!(bad.validate().is_err(), "zero rope_theta must be rejected");

    bad.rope_theta = -1.0;
    assert!(
        bad.validate().is_err(),
        "negative rope_theta must be rejected"
    );

    bad.rope_theta = f64::NAN;
    assert!(bad.validate().is_err(), "NaN rope_theta must be rejected");
}

#[test]
fn test_glm5_kv_heads() {
    // GQA: num_kv_heads (multi_query_group_num) <= num_attention_heads
    let cfg = Glm5Config::default();
    assert!(
        cfg.multi_query_group_num <= cfg.num_attention_heads,
        "GQA: kv_heads ({}) must be <= attention_heads ({})",
        cfg.multi_query_group_num,
        cfg.num_attention_heads
    );
    assert_eq!(
        cfg.multi_query_group_num, 2,
        "GLM-4-9B uses 2 KV head groups"
    );
    assert_eq!(cfg.num_attention_heads, 32, "GLM-4-9B uses 32 Q heads");

    // Divisibility: num_heads must be divisible by multi_query_group_num
    assert_eq!(
        cfg.num_attention_heads % cfg.multi_query_group_num,
        0,
        "num_heads must be divisible by kv_heads for GQA"
    );

    // GQA repeat factor
    let repeat = cfg.num_attention_heads / cfg.multi_query_group_num;
    assert_eq!(repeat, 16, "GLM-4-9B GQA repeat factor is 16");

    // Chat variant preserves same GQA structure
    let chat = Glm5Config::glm4_9b_chat();
    assert_eq!(chat.multi_query_group_num, cfg.multi_query_group_num);
    assert_eq!(chat.num_attention_heads, cfg.num_attention_heads);
}

#[test]
fn test_glm5_mlp_ratio() {
    // MLP intermediate size relationship to hidden_size
    let cfg = Glm5Config::default();
    let ratio = cfg.ffn_hidden_size as f64 / cfg.hidden_size as f64;
    // GLM-4-9B: 13696 / 4096 = 3.34375
    assert!(
        ratio > 2.0 && ratio < 6.0,
        "MLP expansion ratio should be in (2, 6), got {ratio}"
    );
    assert!(
        (ratio - 3.34375).abs() < 0.01,
        "GLM-4-9B MLP ratio should be ~3.34375, got {ratio}"
    );

    // SwiGLU fused output is 2x ffn_hidden_size
    let fused_output = cfg.ffn_hidden_size * 2;
    assert_eq!(
        fused_output, 27392,
        "fused gate+up output: 13696 * 2 = 27392"
    );
    assert_eq!(
        fused_output % 2,
        0,
        "fused dim must be even for gate/up split"
    );

    // Tiny config also has valid MLP ratio
    let tiny = tiny_config();
    let tiny_ratio = tiny.ffn_hidden_size as f64 / tiny.hidden_size as f64;
    assert!(
        tiny_ratio >= 1.0,
        "MLP intermediate should be >= hidden_size"
    );
}

// ---------------------------------------------------------------------------
// Special tokens and tokenizer capacity
// ---------------------------------------------------------------------------

#[test]
fn test_glm5_special_tokens() {
    // GLM-4/5 uses a large vocab (151552) that includes special tokens
    // (system, assistant, user, observation markers, tool call tokens, etc.).
    // The padded_vocab_size must accommodate all special tokens.
    let cfg = Glm5Config::default();

    // GLM-4-9B's tokenizer has ~151,000 tokens plus specials.
    // Padding to 151552 ensures GPU-aligned vocab size.
    assert!(
        cfg.padded_vocab_size >= 150_000,
        "vocab must be large enough for full GLM tokenizer + specials, got {}",
        cfg.padded_vocab_size
    );

    // The model can load and forward any token ID in [0, padded_vocab_size).
    // Verify with tiny config that boundary IDs work.
    let tiny = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, tiny.clone()).unwrap();

    // Token ID 0 (typically <pad> or <unk>)
    let logits_0 = model.forward(&[0], &[0]);
    assert!(logits_0.is_ok(), "token_id=0 should be valid");

    // Last valid token ID
    let last = tiny.padded_vocab_size - 1;
    let logits_last = model.forward(&[last], &[0]);
    assert!(
        logits_last.is_ok(),
        "token_id={last} (last valid) should be accepted"
    );
}

// ---------------------------------------------------------------------------
// Max position embeddings
// ---------------------------------------------------------------------------

#[test]
fn test_glm5_max_position_embeddings() {
    // Base model: 8192 context length
    let cfg = Glm5Config::default();
    assert_eq!(cfg.seq_length, 8192, "base GLM-4-9B: 8K context");
    assert!(
        cfg.seq_length >= 2048,
        "seq_length should be >= 2048 for a modern LLM"
    );

    // Chat variant: 128K context (rope_ratio=500 extension)
    let chat = Glm5Config::glm4_9b_chat();
    assert_eq!(chat.seq_length, 131_072, "chat variant: 128K context");
    assert!(
        chat.seq_length > cfg.seq_length,
        "chat variant should support longer sequences than base"
    );

    // Validation rejects zero seq_length
    let mut bad = tiny_config();
    bad.seq_length = 0;
    assert!(bad.validate().is_err(), "zero seq_length must be rejected");

    // RoPE can handle positions up to seq_length - 1
    let tiny = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, tiny.clone()).unwrap();
    let max_pos = tiny.seq_length - 1;
    let result = model.forward(&[0], &[max_pos]);
    assert!(
        result.is_ok(),
        "should handle position at seq_length-1 ({max_pos})"
    );
}

// ---------------------------------------------------------------------------
// Cross-variant consistency: base vs chat share architectural dimensions
// ---------------------------------------------------------------------------

#[test]
fn test_base_and_chat_share_architecture_differ_in_context() {
    let base = Glm5Config::default();
    let chat = Glm5Config::glm4_9b_chat();

    // Shared architectural parameters
    assert_eq!(base.hidden_size, chat.hidden_size);
    assert_eq!(base.ffn_hidden_size, chat.ffn_hidden_size);
    assert_eq!(base.num_layers, chat.num_layers);
    assert_eq!(base.num_attention_heads, chat.num_attention_heads);
    assert_eq!(base.multi_query_group_num, chat.multi_query_group_num);
    assert_eq!(base.padded_vocab_size, chat.padded_vocab_size);
    assert_eq!(base.kv_channels, chat.kv_channels);
    assert_eq!(base.rmsnorm, chat.rmsnorm);
    assert_eq!(base.add_qkv_bias, chat.add_qkv_bias);
    assert_eq!(base.add_bias_linear, chat.add_bias_linear);

    // Context-extension parameters differ
    assert_ne!(base.seq_length, chat.seq_length);
    assert_ne!(base.rope_theta, chat.rope_theta);
    assert_ne!(base.layernorm_epsilon, chat.layernorm_epsilon);

    // Both must validate
    assert!(base.validate().is_ok());
    assert!(chat.validate().is_ok());
}

// ---------------------------------------------------------------------------
// Error path: mismatched input_ids and positions length
// ---------------------------------------------------------------------------

#[test]
fn test_forward_rejects_mismatched_ids_and_positions() {
    let tiny = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, tiny).unwrap();

    let err = model.forward(&[0, 1, 2], &[0, 1]).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("input_ids len") && msg.contains("positions len"),
        "should report length mismatch: {msg}"
    );
}

// ---------------------------------------------------------------------------
// KV cache layer count mismatch
// ---------------------------------------------------------------------------

#[test]
fn test_forward_rejects_cache_layer_mismatch() {
    let tiny = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, tiny).unwrap();

    // Create cache with wrong layer count
    let mut wrong_cache = KvCache::new(1); // model has 2 layers
    let err = model
        .forward_cached(&[0], &[0], Some(&mut wrong_cache))
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("cache") && msg.contains("1") && msg.contains("2"),
        "should report cache/model layer mismatch: {msg}"
    );
}

// ---------------------------------------------------------------------------
// forward_from_embeddings: wrong hidden_size rejected
// ---------------------------------------------------------------------------

#[test]
fn test_forward_from_embeddings_rejects_wrong_hidden_size() {
    let tiny = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, tiny).unwrap();

    // hidden_size is 256 but we pass 128
    let bad_emb = DynTensor::ones(&[1, 3, 128], DType::F32, &Device::Cpu).unwrap();
    let err = model
        .forward_from_embeddings(&bad_emb, &[0, 1, 2], None)
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("hidden_size"),
        "should report hidden_size mismatch: {msg}"
    );
}
