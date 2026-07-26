// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Qwen3 config constructors, builder methods, GQA head arithmetic,
//! RoPE theta variants, error types, SwiGLU properties, and architectural
//! invariants. Covers gaps not addressed by the existing test files.

use super::*;
use crate::test_utils::tiny_config;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// -- Qwen3Config::new() constructor -------------------------------------------

#[test]
fn test_config_new_constructor_roundtrip() {
    // Verify that Qwen3Config::new() sets all fields correctly.
    let cfg = Qwen3Config::new(
        1024,      // hidden_size
        2048,      // intermediate_size
        12,        // num_hidden_layers
        8,         // num_attention_heads
        4,         // num_key_value_heads
        32000,     // vocab_size
        1e-5,      // rms_norm_eps
        500_000.0, // rope_theta
        4096,      // max_position_embeddings
        false,     // tie_word_embeddings
        None,      // rope_scaling
    );
    assert_eq!(cfg.hidden_size, 1024);
    assert_eq!(cfg.intermediate_size, 2048);
    assert_eq!(cfg.num_hidden_layers, 12);
    assert_eq!(cfg.num_attention_heads, 8);
    assert_eq!(cfg.num_key_value_heads, 4);
    assert_eq!(cfg.vocab_size, 32000);
    assert!((cfg.rms_norm_eps - 1e-5).abs() < 1e-12);
    assert!((cfg.rope_theta - 500_000.0).abs() < 1e-6);
    assert_eq!(cfg.max_position_embeddings, 4096);
    assert!(!cfg.tie_word_embeddings);
    assert!(cfg.rope_scaling.is_none());
    assert!(cfg.validate().is_ok());
}

// -- Builder methods ----------------------------------------------------------

#[test]
fn test_config_with_vocab_size_builder() {
    let cfg = tiny_config().with_vocab_size(50_000);
    assert_eq!(cfg.vocab_size, 50_000);
    // Other fields unchanged
    assert_eq!(cfg.hidden_size, 256);
    assert_eq!(cfg.num_attention_heads, 2);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_with_num_hidden_layers_builder() {
    let cfg = tiny_config().with_num_hidden_layers(8);
    assert_eq!(cfg.num_hidden_layers, 8);
    // Other fields unchanged
    assert_eq!(cfg.hidden_size, 256);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_builder_chaining() {
    // Builder methods should chain.
    let cfg = tiny_config().with_vocab_size(200).with_num_hidden_layers(4);
    assert_eq!(cfg.vocab_size, 200);
    assert_eq!(cfg.num_hidden_layers, 4);
    assert!(cfg.validate().is_ok());
}

// -- GQA head dimension calculations ------------------------------------------

#[test]
fn test_head_dim_is_always_128() {
    // Qwen3 head_dim is a constant 128 regardless of config.
    let cfg = tiny_config();
    assert_eq!(cfg.head_dim(), 128);

    let cfg2 = Qwen3Config::new(
        4096,
        11008,
        32,
        32,
        8,
        151_936,
        1e-6,
        1_000_000.0,
        32768,
        false,
        None,
    );
    assert_eq!(cfg2.head_dim(), 128);
}

#[test]
fn test_gqa_groups_mha_equals_1() {
    // MHA: num_attention_heads == num_key_value_heads -> groups = 1
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 4;
    assert_eq!(cfg.num_kv_groups().unwrap(), 1);
}

#[test]
fn test_gqa_groups_various_ratios() {
    // GQA with various head ratios.
    let cases: Vec<(usize, usize, usize)> = vec![
        (16, 4, 4),  // 16/4 = 4 groups
        (32, 8, 4),  // 32/8 = 4 groups
        (64, 4, 16), // 64/4 = 16 groups (Qwen3-235B)
        (32, 4, 8),  // 32/4 = 8 groups (Qwen3-30B)
        (12, 3, 4),  // 12/3 = 4 groups
        (8, 1, 8),   // 8/1 = 8 groups (MQA-like)
    ];
    for (nh, nkv, expected) in cases {
        let mut cfg = tiny_config();
        cfg.num_attention_heads = nh;
        cfg.num_key_value_heads = nkv;
        assert_eq!(
            cfg.num_kv_groups().unwrap(),
            expected,
            "num_kv_groups({nh}, {nkv}) should be {expected}"
        );
    }
}

#[test]
fn test_gqa_groups_non_divisible_error_message() {
    // Error message should include both head counts.
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 7;
    cfg.num_key_value_heads = 3;
    let err = cfg.num_kv_groups().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("7"),
        "should mention num_attention_heads=7: {msg}"
    );
    assert!(
        msg.contains("3"),
        "should mention num_key_value_heads=3: {msg}"
    );
}

// -- RoPE theta for different model sizes -------------------------------------

#[test]
fn test_rope_theta_standard_10k() {
    // Standard RoPE theta (10,000) used by smaller Qwen3 models.
    let cfg = tiny_config();
    assert!((cfg.rope_theta - 10_000.0).abs() < 1e-6);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_rope_theta_1m_large_model() {
    // Large Qwen3 models use rope_theta = 1,000,000.
    let mut cfg = tiny_config();
    cfg.rope_theta = 1_000_000.0;
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_rope_theta_nan_rejected() {
    let mut cfg = tiny_config();
    cfg.rope_theta = f64::NAN;
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("rope_theta"),
        "NaN rope_theta error should mention rope_theta: {msg}"
    );
}

#[test]
fn test_rope_theta_neg_infinity_rejected() {
    let mut cfg = tiny_config();
    cfg.rope_theta = f64::NEG_INFINITY;
    assert!(cfg.validate().is_err());
}

// -- Error type coverage ------------------------------------------------------

#[test]
fn test_error_display_invalid_config() {
    let err = Qwen3Error::InvalidConfig {
        reason: "test reason".into(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("invalid config"));
    assert!(msg.contains("test reason"));
}

#[test]
fn test_error_display_invalid_input() {
    let err = Qwen3Error::InvalidInput {
        reason: "mismatched lengths".into(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("invalid input"));
    assert!(msg.contains("mismatched lengths"));
}

#[test]
fn test_error_display_cache_mismatch() {
    let err = Qwen3Error::CacheMismatch {
        cache_layers: 5,
        model_layers: 2,
    };
    let msg = format!("{err}");
    assert!(msg.contains("5"), "should mention cache_layers: {msg}");
    assert!(msg.contains("2"), "should mention model_layers: {msg}");
}

#[test]
fn test_error_display_non_finite_output() {
    let err = Qwen3Error::NonFiniteOutput {
        stage: "test_stage",
        count: 42,
    };
    let msg = format!("{err}");
    assert!(msg.contains("test_stage"));
    assert!(msg.contains("42"));
}

#[test]
fn test_error_display_weight_load() {
    let err = Qwen3Error::WeightLoad {
        reason: "missing tensor".into(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("weight load"));
    assert!(msg.contains("missing tensor"));
}

// -- SwiGLU forward property --------------------------------------------------

#[test]
fn test_swiglu_output_shape_preserved() {
    // SwiGLU MLP should produce [batch, seq, hidden_size] from [batch, seq, hidden_size].
    // Verify by running a single-layer forward and checking output shape.
    let cfg = tiny_config().with_num_hidden_layers(1);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let logits = model.forward(&[0, 1, 2, 3], &[0, 1, 2, 3]).unwrap();
    assert_eq!(
        logits.dims(),
        &[1, 4, cfg.vocab_size],
        "output should be [1, seq_len, vocab_size]"
    );
}

// -- Config clone + debug -----------------------------------------------------

#[test]
fn test_config_clone_is_independent() {
    let cfg1 = tiny_config();
    let mut cfg2 = cfg1.clone();
    cfg2.hidden_size = 999;
    // Original unchanged
    assert_eq!(cfg1.hidden_size, 256);
    assert_eq!(cfg2.hidden_size, 999);
}

#[test]
fn test_config_debug_format_contains_fields() {
    let cfg = tiny_config();
    let debug = format!("{cfg:?}");
    assert!(debug.contains("hidden_size"));
    assert!(debug.contains("256"));
    assert!(debug.contains("num_attention_heads"));
    assert!(debug.contains("rope_theta"));
}

// -- Model accessor coverage --------------------------------------------------

#[test]
fn test_embed_tokens_accessor_weight_shape() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let embed = model.embed_tokens();
    let weight = embed.weight();
    assert_eq!(weight.dims(), &[cfg.vocab_size, cfg.hidden_size]);
}

#[test]
fn test_dtype_accessor_matches_vb() {
    let cfg = tiny_config();

    let vb_f32 = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model_f32 = Qwen3Model::load(&vb_f32, cfg.clone()).unwrap();
    assert_eq!(model_f32.dtype(), DType::F32);

    let vb_bf16 = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    let model_bf16 = Qwen3Model::load(&vb_bf16, cfg).unwrap();
    assert_eq!(model_bf16.dtype(), DType::BF16);
}

// -- Qwen3 production config shapes ------------------------------------------

#[test]
fn test_qwen3_0_6b_config_validates() {
    // Qwen3-0.6B: smallest Qwen3 variant.
    // Source: Kani proof validate_accepts_production_configs, arXiv:2505.09388
    let cfg = Qwen3Config::new(
        896,         // hidden_size
        4864,        // intermediate_size
        28,          // num_hidden_layers
        14,          // num_attention_heads
        2,           // num_key_value_heads
        151_936,     // vocab_size
        1e-6,        // rms_norm_eps
        1_000_000.0, // rope_theta
        40_960,      // max_position_embeddings
        true,        // tie_word_embeddings
        None,
    );
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.head_dim(), 128);
    assert_eq!(cfg.num_kv_groups().unwrap(), 7); // 14/2 = 7
}

#[test]
fn test_qwen3_8b_config_validates() {
    // Qwen3-8B: medium Qwen3 variant.
    let cfg = Qwen3Config::new(
        4096,        // hidden_size
        14336,       // intermediate_size
        36,          // num_hidden_layers
        32,          // num_attention_heads
        8,           // num_key_value_heads
        151_936,     // vocab_size
        1e-6,        // rms_norm_eps
        1_000_000.0, // rope_theta
        131072,      // max_position_embeddings (128K context)
        false,       // tie_word_embeddings
        None,
    );
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.head_dim(), 128);
    assert_eq!(cfg.num_kv_groups().unwrap(), 4);
    // hidden_size should be num_attention_heads * head_dim
    assert_eq!(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim());
}

// -- Qwen3MoeConfig::new() constructor ----------------------------------------

#[test]
fn test_moe_config_new_constructor() {
    let base = tiny_config();
    let moe_cfg = Qwen3MoeConfig::new(
        base,
        128,       // num_experts
        8,         // num_experts_per_tok
        true,      // shared_expert
        Some(768), // shared_expert_intermediate_size
    );
    assert_eq!(moe_cfg.num_experts, 128);
    assert_eq!(moe_cfg.num_experts_per_tok, 8);
    assert!(moe_cfg.shared_expert);
    assert_eq!(moe_cfg.shared_expert_intermediate_size, Some(768));
    assert_eq!(moe_cfg.shared_expert_ff_dim(), 768);
    assert!(moe_cfg.validate().is_ok());
}

// -- Multi-token output shape -------------------------------------------------

#[test]
fn test_forward_multi_token_output_shape() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let logits = model.forward(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4]).unwrap();
    assert_eq!(
        logits.dims(),
        &[1, 5, cfg.vocab_size],
        "output shape should be [1, num_tokens, vocab_size]"
    );
    let vals = logits.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 5 * cfg.vocab_size);
}

// -- hidden_size = num_heads * head_dim invariant -----------------------------

#[test]
fn test_hidden_size_head_dim_consistency() {
    // For Qwen3, hidden_size must equal num_attention_heads * head_dim.
    // This is a structural requirement for the q_proj weight shape.
    let cfg = tiny_config();
    assert_eq!(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim());
}

// -- Config validation boundary: eps just above zero --------------------------

#[test]
fn test_config_validation_very_small_positive_eps() {
    // Very small but positive eps should pass validation.
    let mut cfg = tiny_config();
    cfg.rms_norm_eps = 1e-30;
    assert!(cfg.validate().is_ok());
}

// -- Qwen3Error → TensorError conversion -------------------------------------

#[test]
fn test_error_to_tensor_error_conversion() {
    use nn_core::TensorError;
    let qwen_err = Qwen3Error::InvalidConfig {
        reason: "test conversion".into(),
    };
    let tensor_err: TensorError = qwen_err.into();
    let msg = format!("{tensor_err}");
    assert!(
        msg.contains("test conversion") || msg.contains("invalid config"),
        "TensorError should preserve original message: {msg}"
    );
}
