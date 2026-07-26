// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Config construction, validation, GQA arithmetic, vocab/embedding consistency,
//! and RoPE parameter tests for Qwen3 (#4186).

use crate::test_utils::tiny_config;
use crate::{Qwen3Config, Qwen3MoeConfig};

// ---------------------------------------------------------------------------
// Config construction for different model sizes
// ---------------------------------------------------------------------------

#[test]
fn test_config_qwen3_0_6b_dimensions() {
    // Qwen3-0.6B: hidden_size=896, 14 heads, 2 kv heads, head_dim=128
    // hidden_size != num_attention_heads * head_dim here (896 != 14*128=1792).
    // Qwen3 defines head_dim as a constant 128; projection weights are
    // [num_heads * head_dim, hidden_size], not [hidden_size, hidden_size].
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
    assert_eq!(cfg.head_dim(), 128);
    assert_eq!(cfg.num_kv_groups().unwrap(), 7);
    // q_proj shape would be [14*128, 896] = [1792, 896]
    assert_eq!(cfg.num_attention_heads * cfg.head_dim(), 1792);
}

#[test]
fn test_config_qwen3_1_7b_dimensions() {
    // Qwen3-1.7B approximate: hidden=2048, 16 heads, 4 kv heads
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
    assert_eq!(cfg.num_kv_groups().unwrap(), 4);
    assert_eq!(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim());
}

#[test]
fn test_config_qwen3_4b_dimensions() {
    // Qwen3-4B approximate: hidden=2560, 20 heads, 4 kv heads
    let cfg = Qwen3Config::new(
        2560,
        9216,
        36,
        20,
        4,
        151_936,
        1e-6,
        1_000_000.0,
        40_960,
        false,
        None,
    );
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.num_kv_groups().unwrap(), 5);
}

#[test]
fn test_config_qwen3_32b_dimensions() {
    // Qwen3-32B: hidden=5120, 40 heads, 8 kv heads
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
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.num_kv_groups().unwrap(), 5);
    assert_eq!(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim());
}

// ---------------------------------------------------------------------------
// Config validation: head_dim must work with hidden_size
// ---------------------------------------------------------------------------

#[test]
fn test_config_validate_accepts_valid_gqa_ratios() {
    // Every valid GQA ratio: num_attention_heads must be divisible by num_key_value_heads
    for (nh, nkv) in [(2, 1), (4, 2), (8, 1), (8, 4), (8, 8), (16, 2), (32, 4)] {
        let cfg = Qwen3Config::new(
            nh * 128,
            1024,
            2,
            nh,
            nkv,
            100,
            1e-6,
            10_000.0,
            64,
            true,
            None,
        );
        assert!(cfg.validate().is_ok(), "should accept nh={nh}, nkv={nkv}");
        assert_eq!(cfg.num_kv_groups().unwrap(), nh / nkv);
    }
}

#[test]
fn test_config_validate_rejects_non_divisible_heads() {
    // 5 attention heads, 3 kv heads: 5 % 3 != 0
    let cfg = Qwen3Config::new(640, 1024, 2, 5, 3, 100, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_rejects_kv_heads_greater_than_attention_heads() {
    // More kv heads than attention heads makes no sense for GQA
    let cfg = Qwen3Config::new(256, 512, 2, 2, 4, 100, 1e-6, 10_000.0, 64, true, None);
    // This should fail because 2 % 4 != 0
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// GQA group count edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_num_kv_groups_mha_identity() {
    // MHA: num_attention_heads == num_key_value_heads => 1 group (no repetition)
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 8;
    cfg.num_key_value_heads = 8;
    assert_eq!(cfg.num_kv_groups().unwrap(), 1);
}

#[test]
fn test_num_kv_groups_mqa_single_kv_head() {
    // MQA: 1 kv head shared across all attention heads
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 32;
    cfg.num_key_value_heads = 1;
    assert_eq!(cfg.num_kv_groups().unwrap(), 32);
}

#[test]
fn test_num_kv_groups_zero_kv_heads_is_error() {
    let mut cfg = tiny_config();
    cfg.num_key_value_heads = 0;
    let err = cfg.num_kv_groups();
    assert!(err.is_err());
    let msg = format!("{}", err.unwrap_err());
    assert!(
        msg.contains("num_key_value_heads"),
        "error should reference kv heads: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Vocab size and embedding dim consistency
// ---------------------------------------------------------------------------

#[test]
fn test_vocab_size_standard_qwen3() {
    // All Qwen3 models use vocab_size=151_936
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
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.vocab_size, 151_936);
}

#[test]
fn test_vocab_size_custom_valid() {
    // Custom vocab sizes should validate if > 0
    for vocab in [1, 100, 50_000, 200_000] {
        let cfg = tiny_config().with_vocab_size(vocab);
        assert!(cfg.validate().is_ok(), "vocab_size={vocab} should be valid");
    }
}

#[test]
fn test_vocab_size_zero_invalid() {
    let cfg = tiny_config().with_vocab_size(0);
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("vocab_size"),
        "error should mention vocab_size: {msg}"
    );
}

#[test]
fn test_embedding_weight_shape_is_vocab_by_hidden() {
    // The embedding weight should be [vocab_size, hidden_size]
    use nn_core::var_builder::VarBuilder;
    use nn_core::{DType, Device};

    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = crate::Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let embed_weight = model.embed_tokens().weight();
    assert_eq!(embed_weight.dims(), &[cfg.vocab_size, cfg.hidden_size]);
}

// ---------------------------------------------------------------------------
// RoPE parameters
// ---------------------------------------------------------------------------

#[test]
fn test_rope_theta_10k_standard() {
    let cfg = Qwen3Config::new(256, 512, 2, 2, 2, 100, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_ok());
    assert!((cfg.rope_theta - 10_000.0).abs() < f64::EPSILON);
}

#[test]
fn test_rope_theta_1m_for_long_context() {
    // Qwen3 uses base=1_000_000 for 128K+ context windows
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
    assert!(cfg.validate().is_ok());
    assert!((cfg.rope_theta - 1_000_000.0).abs() < f64::EPSILON);
}

#[test]
fn test_rope_theta_zero_rejected() {
    let cfg = Qwen3Config::new(256, 512, 2, 2, 2, 100, 1e-6, 0.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_rope_theta_negative_rejected() {
    let cfg = Qwen3Config::new(256, 512, 2, 2, 2, 100, 1e-6, -10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_rope_theta_infinity_rejected() {
    let cfg = Qwen3Config::new(256, 512, 2, 2, 2, 100, 1e-6, f64::INFINITY, 64, true, None);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_rope_theta_nan_rejected() {
    let cfg = Qwen3Config::new(256, 512, 2, 2, 2, 100, 1e-6, f64::NAN, 64, true, None);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_rope_max_position_embeddings_zero_rejected() {
    let cfg = Qwen3Config::new(256, 512, 2, 2, 2, 100, 1e-6, 10_000.0, 0, true, None);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_rope_with_yarn_scaling_validates() {
    use nn_core::layers::YarnScaling;
    let cfg = Qwen3Config::new(
        256,
        512,
        2,
        2,
        2,
        100,
        1e-6,
        1_000_000.0,
        131_072,
        true,
        Some(YarnScaling::new(4.0, 1.0, 32.0, 1.0, 64)),
    );
    assert!(cfg.validate().is_ok());
    assert!(cfg.rope_scaling.is_some());
}

// ---------------------------------------------------------------------------
// MoE config validation
// ---------------------------------------------------------------------------

#[test]
fn test_moe_config_validate_zero_experts_rejected() {
    let base = tiny_config();
    let moe = Qwen3MoeConfig::new(base, 0, 4, false, None);
    assert!(moe.validate().is_err());
}

#[test]
fn test_moe_config_validate_experts_per_tok_exceeds_total() {
    let base = tiny_config();
    let moe = Qwen3MoeConfig::new(base, 8, 10, false, None);
    assert!(moe.validate().is_err());
}

#[test]
fn test_moe_config_validate_zero_experts_per_tok() {
    let base = tiny_config();
    let moe = Qwen3MoeConfig::new(base, 8, 0, false, None);
    assert!(moe.validate().is_err());
}

#[test]
fn test_moe_config_validate_shared_expert_zero_intermediate_rejected() {
    let base = tiny_config();
    let moe = Qwen3MoeConfig::new(base, 8, 4, true, Some(0));
    assert!(moe.validate().is_err());
}

#[test]
fn test_moe_config_shared_expert_ff_dim_fallback() {
    let base = tiny_config(); // intermediate_size = 512
    let moe = Qwen3MoeConfig::new(base, 8, 4, true, None);
    assert_eq!(moe.shared_expert_ff_dim(), 512);
}

#[test]
fn test_moe_config_shared_expert_ff_dim_explicit() {
    let base = tiny_config();
    let moe = Qwen3MoeConfig::new(base, 8, 4, true, Some(768));
    assert_eq!(moe.shared_expert_ff_dim(), 768);
}
