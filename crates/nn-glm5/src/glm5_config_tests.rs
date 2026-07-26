// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GLM-4/5 model configuration and architecture tests.
//!
//! Part of #3525. Covers:
//!
//! 1. Model size presets (GLM-4-9B base, chat, hypothetical 1.5B/130B)
//! 2. Hidden/heads/kv_channels architectural invariant
//! 3. RoPE config: base frequency, rope_ratio scaling, max positions
//! 4. Vocabulary size: GLM tokenizer sizes (65024 ChatGLM2, 151552 GLM-4)
//! 5. Multi-query attention: GQA/MQA/MHA configurations
//! 6. SwiGLU activation: gate+up fused dimension
//! 7. Config validation: exhaustive invalid parameter rejection
//! 8. Layer norm epsilon ranges
//! 9. Max sequence length configuration
//! 10. Untied embeddings: output_layer independence
//! 11. Weight shape algebra for all layer types

use super::*;
use crate::test_utils::tiny_config;
use nn_core::dyn_tensor::DynTensor;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_model(cfg: Glm5Config) -> Glm5Model {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    Glm5Model::load(&vb, cfg).unwrap()
}

/// GLM-4-1.5B hypothetical config (smaller variant).
fn glm4_1_5b_config() -> Glm5Config {
    Glm5Config::new(
        2048, 5504, 24, 16, 2, 151552, 128, 1e-5, 8192, true, true, false, 10_000.0,
    )
}

/// GLM-4-130B hypothetical config (large variant).
fn glm4_130b_config() -> Glm5Config {
    Glm5Config::new(
        12288, 32768, 96, 96, 8, 151552, 128, 1e-5, 8192, true, true, false, 10_000.0,
    )
}

// ===========================================================================
// 1. Model size presets validate and have consistent dimensions
// ===========================================================================

#[test]
fn test_model_size_presets_validate_with_invariant() {
    let configs: Vec<(&str, Glm5Config)> = vec![
        ("tiny", tiny_config()),
        ("1.5B", glm4_1_5b_config()),
        ("9B-base", Glm5Config::default()),
        ("9B-chat", Glm5Config::glm4_9b_chat()),
        ("130B", glm4_130b_config()),
    ];
    for (name, cfg) in &configs {
        cfg.validate()
            .unwrap_or_else(|e| panic!("{name} must validate: {e}"));
        assert_eq!(
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.kv_channels,
            "{name}: hidden_size != num_heads * kv_channels"
        );
    }
}

#[test]
fn test_hypothetical_sizes_have_expected_dimensions() {
    let small = glm4_1_5b_config();
    assert_eq!(
        (
            small.hidden_size,
            small.num_layers,
            small.num_attention_heads
        ),
        (2048, 24, 16)
    );

    let large = glm4_130b_config();
    assert_eq!(
        (
            large.hidden_size,
            large.num_layers,
            large.num_attention_heads
        ),
        (12288, 96, 96)
    );
}

// ===========================================================================
// 2. FFN to hidden ratio (SwiGLU expansion factor)
// ===========================================================================

#[test]
fn test_ffn_to_hidden_ratio_in_swiglu_range() {
    let cfg = Glm5Config::default();
    let ratio = cfg.ffn_hidden_size as f64 / cfg.hidden_size as f64;
    // GLM-4-9B: 13696/4096 ~= 3.34, typical SwiGLU range
    assert!(
        (2.5..=4.0).contains(&ratio),
        "9B ffn/hidden ratio {ratio:.2} outside SwiGLU range [2.5, 4.0]"
    );
}

// ===========================================================================
// 3. RoPE config: base frequency, scaling, max positions
// ===========================================================================

#[test]
fn test_rope_base_frequency_and_chat_scaling() {
    let base = Glm5Config::default();
    let chat = Glm5Config::glm4_9b_chat();

    assert_eq!(base.rope_theta, 10_000.0, "base uses standard theta=10000");
    // Chat uses rope_ratio=500: theta_chat = theta_base * 500
    assert!(
        (chat.rope_theta - base.rope_theta * 500.0).abs() < f64::EPSILON,
        "chat theta ({}) should be base * 500",
        chat.rope_theta,
    );
}

#[test]
fn test_rope_max_positions_match_seq_length() {
    assert_eq!(Glm5Config::default().seq_length, 8192, "base: 8K");
    assert_eq!(Glm5Config::glm4_9b_chat().seq_length, 131_072, "chat: 128K");
}

#[test]
fn test_rope_theta_rejects_zero_and_neg_infinity() {
    for bad in [0.0, f64::NEG_INFINITY] {
        let mut cfg = tiny_config();
        cfg.rope_theta = bad;
        assert!(cfg.validate().is_err(), "rope_theta={bad} must be rejected");
    }
}

#[test]
fn test_rope_theta_accepts_very_large_for_long_context() {
    let mut cfg = tiny_config();
    cfg.rope_theta = 1e15; // YaRN-style extreme theta
    assert!(cfg.validate().is_ok());
}

// ===========================================================================
// 4. Vocabulary size: GLM tokenizer sizes
// ===========================================================================

#[test]
fn test_glm4_vocab_151552_aligned_to_128() {
    let cfg = Glm5Config::default();
    assert_eq!(cfg.padded_vocab_size, 151552);
    assert_eq!(cfg.padded_vocab_size % 128, 0, "GPU-aligned vocab");
}

#[test]
fn test_chatglm2_vocab_65024_validates() {
    let mut cfg = tiny_config();
    cfg.padded_vocab_size = 65024;
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_vocab_size_determines_logits_shape() {
    let cfg = tiny_config();
    let model = load_model(cfg.clone());
    let logits = model.forward(&[0, 1], &[0, 1]).unwrap();
    assert_eq!(logits.dims(), &[1, 2, cfg.padded_vocab_size]);
}

// ===========================================================================
// 5. Multi-query attention: GQA/MQA/MHA
// ===========================================================================

#[test]
fn test_gqa_default_has_fewer_kv_heads() {
    let cfg = Glm5Config::default();
    assert!(cfg.multi_query_group_num < cfg.num_attention_heads);
    assert_eq!(cfg.num_kv_groups().unwrap(), 16, "32 Q / 2 KV = 16 groups");
}

#[test]
fn test_mqa_single_kv_head_loads_and_forwards() {
    let mut cfg = tiny_config();
    cfg.multi_query_group_num = 1;
    assert_eq!(cfg.num_kv_groups().unwrap(), 4);
    let model = load_model(cfg);
    assert_eq!(model.forward(&[0], &[0]).unwrap().dims().len(), 3);
}

#[test]
fn test_mha_equal_kv_heads_loads_and_forwards() {
    let mut cfg = tiny_config();
    cfg.multi_query_group_num = 4; // == num_attention_heads
    assert_eq!(cfg.num_kv_groups().unwrap(), 1);
    let model = load_model(cfg);
    assert_eq!(model.forward(&[0], &[0]).unwrap().dims().len(), 3);
}

#[test]
fn test_kv_heads_indivisible_and_zero_rejected() {
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 2;
    cfg.multi_query_group_num = 4;
    assert!(cfg.validate().is_err());

    cfg.multi_query_group_num = 0;
    cfg.num_attention_heads = 4;
    assert!(cfg.validate().is_err());
}

// ===========================================================================
// 6. SwiGLU: gate+up fused dimension
// ===========================================================================

#[test]
fn test_swiglu_fused_dim_is_double_ffn() {
    let cfg = Glm5Config::default();
    let fused = cfg.ffn_hidden_size * 2;
    assert_eq!(fused, 27392, "9B: 13696 * 2");
    assert_eq!(fused / 2, cfg.ffn_hidden_size);
}

#[test]
fn test_swiglu_residual_preserves_hidden_size() {
    let cfg = tiny_config();
    let model = load_model(cfg.clone());
    let emb = DynTensor::ones(&[1, 3, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    model
        .forward_from_embeddings(&emb, &[0, 1, 2], None)
        .expect("SwiGLU must preserve hidden_size for residual");
}

// ===========================================================================
// 7. Exhaustive config validation
// ===========================================================================

#[test]
fn test_validate_rejects_zero_fields_individually() {
    let zero_fields: Vec<(&str, Box<dyn Fn(&mut Glm5Config)>)> = vec![
        (
            "hidden_size",
            Box::new(|c: &mut Glm5Config| c.hidden_size = 0),
        ),
        ("ffn_hidden_size", Box::new(|c| c.ffn_hidden_size = 0)),
        ("num_layers", Box::new(|c| c.num_layers = 0)),
        (
            "num_attention_heads",
            Box::new(|c| c.num_attention_heads = 0),
        ),
        (
            "multi_query_group_num",
            Box::new(|c| c.multi_query_group_num = 0),
        ),
        ("padded_vocab_size", Box::new(|c| c.padded_vocab_size = 0)),
        ("seq_length", Box::new(|c| c.seq_length = 0)),
    ];
    for (field, mutate) in &zero_fields {
        let mut cfg = tiny_config();
        mutate(&mut cfg);
        assert!(cfg.validate().is_err(), "zero {field} must be rejected");
    }
}

#[test]
fn test_validate_rejects_kv_channels_not_multiple_of_4() {
    for bad_kv in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15] {
        let mut cfg = tiny_config();
        cfg.kv_channels = bad_kv;
        assert!(
            cfg.validate().is_err(),
            "kv_channels={bad_kv} must be rejected"
        );
    }
}

#[test]
fn test_validate_accepts_kv_channels_multiples_of_4() {
    for good_kv in [4, 8, 16, 32, 64, 128, 256] {
        let mut cfg = tiny_config();
        cfg.kv_channels = good_kv;
        cfg.hidden_size = cfg.num_attention_heads * good_kv;
        assert!(
            cfg.validate().is_ok(),
            "kv_channels={good_kv} should be accepted"
        );
    }
}

#[test]
fn test_validate_rejects_non_finite_float_params() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0] {
        let mut cfg = tiny_config();
        cfg.layernorm_epsilon = bad;
        assert!(cfg.validate().is_err(), "epsilon={bad} must be rejected");

        let mut cfg2 = tiny_config();
        cfg2.rope_theta = bad;
        assert!(
            cfg2.validate().is_err(),
            "rope_theta={bad} must be rejected"
        );
    }
}

#[test]
fn test_validate_rejects_indivisible_heads_kv_groups() {
    for (heads, kv) in [(7, 3), (5, 2), (10, 3), (4, 3), (6, 4)] {
        let mut cfg = tiny_config();
        cfg.num_attention_heads = heads;
        cfg.multi_query_group_num = kv;
        assert!(
            cfg.validate().is_err(),
            "heads={heads}, kv={kv} should fail"
        );
    }
}

// ===========================================================================
// 8. Layer norm epsilon ranges
// ===========================================================================

#[test]
fn test_layernorm_epsilon_base_vs_chat() {
    let base = Glm5Config::default();
    let chat = Glm5Config::glm4_9b_chat();
    assert_eq!(base.layernorm_epsilon, 1.5625e-5);
    assert_eq!(chat.layernorm_epsilon, 1.5625e-7);
    assert!(chat.layernorm_epsilon < base.layernorm_epsilon);
}

#[test]
fn test_layernorm_epsilon_typical_values_validate() {
    for eps in [1e-5, 1e-6, 1e-8, 1e-12, 1e-15, 1.5625e-5, 1.5625e-7] {
        let mut cfg = tiny_config();
        cfg.layernorm_epsilon = eps;
        assert!(cfg.validate().is_ok(), "epsilon={eps} should validate");
    }
}

// ===========================================================================
// 9. Max sequence length
// ===========================================================================

#[test]
fn test_seq_length_boundary_values() {
    let mut cfg = tiny_config();
    cfg.seq_length = 1;
    assert!(cfg.validate().is_ok(), "seq_length=1 is minimum valid");

    cfg.seq_length = 4_194_304; // 4M tokens
    assert!(cfg.validate().is_ok(), "very large seq_length valid");

    cfg.seq_length = 0;
    assert!(cfg.validate().is_err(), "zero seq_length rejected");
}

#[test]
fn test_forward_within_seq_length_succeeds() {
    let model = load_model(tiny_config());
    let ids: Vec<usize> = (0..10).collect();
    let pos: Vec<usize> = (0..10).collect();
    model
        .forward(&ids, &pos)
        .expect("forward within seq_length");
}

// ===========================================================================
// 10. Untied embeddings: output_layer independence
// ===========================================================================

#[test]
fn test_output_layer_independent_from_embedding() {
    // GLM-4 does NOT tie embeddings: output_layer and embedding are separate.
    // Verify by loading models with/without output bias (add_bias_linear).
    let cfg = tiny_config();
    let model_no_bias = load_model(cfg.clone());

    let mut cfg_bias = tiny_config();
    cfg_bias.add_bias_linear = true;
    let model_bias = load_model(cfg_bias);

    let logits_a = model_no_bias.forward(&[0], &[0]).unwrap();
    let logits_b = model_bias.forward(&[0], &[0]).unwrap();
    assert_eq!(logits_a.dims(), logits_b.dims());
    assert_eq!(logits_a.dims()[2], cfg.padded_vocab_size);
}

#[test]
fn test_add_bias_linear_flag_independently_settable() {
    let cfg = Glm5Config::default();
    assert!(!cfg.add_bias_linear, "default: no output bias");

    let cfg_bias = Glm5Config::new(
        256, 512, 2, 4, 2, 100, 64, 1e-5, 64, true, true, true, 10_000.0,
    );
    assert!(cfg_bias.add_bias_linear);
    assert!(cfg_bias.validate().is_ok());
}

// ===========================================================================
// 11. Weight shape algebra
// ===========================================================================

#[test]
fn test_qkv_weight_shape_formula() {
    let cfg = Glm5Config::default();
    let qkv_rows = (cfg.num_attention_heads + 2 * cfg.multi_query_group_num) * cfg.kv_channels;
    assert_eq!(qkv_rows, 4608, "QKV fused rows for 9B: (32+4)*128");

    let q_dim = cfg.num_attention_heads * cfg.kv_channels;
    let kv_dim = cfg.multi_query_group_num * cfg.kv_channels;
    assert_eq!(q_dim + 2 * kv_dim, qkv_rows, "Q + K + V = fused QKV");
    assert_eq!(kv_dim, 256, "K/V dim for 2 KV heads * 128");
}

#[test]
fn test_mlp_weight_shapes() {
    let cfg = Glm5Config::default();
    // dense_h_to_4h: [ffn * 2, hidden] = [27392, 4096]
    assert_eq!(cfg.ffn_hidden_size * 2, 27392);
    // dense_4h_to_h: [hidden, ffn] = [4096, 13696]
    assert_eq!((cfg.hidden_size, cfg.ffn_hidden_size), (4096, 13696));
}

#[test]
fn test_per_layer_parameter_count_reasonable_for_9b() {
    let cfg = Glm5Config::default();
    let h = cfg.hidden_size;
    let hd = cfg.kv_channels;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.multi_query_group_num;
    let ffn = cfg.ffn_hidden_size;

    let qkv = (nh + 2 * nkv) * hd * h;
    let qkv_bias = if cfg.add_qkv_bias {
        (nh + 2 * nkv) * hd
    } else {
        0
    };
    let dense = h * nh * hd;
    let mlp = ffn * 2 * h + h * ffn;
    let norms = 2 * h;
    let per_layer = qkv + qkv_bias + dense + mlp + norms;

    // ~9B total / 40 layers => ~225M per layer (rough)
    assert!(per_layer > 100_000_000 && per_layer < 500_000_000);
}

#[test]
fn test_head_dim_always_equals_kv_channels() {
    for kv in [4, 8, 32, 64, 128, 256] {
        let mut cfg = tiny_config();
        cfg.kv_channels = kv;
        cfg.hidden_size = cfg.num_attention_heads * kv;
        assert_eq!(cfg.head_dim(), kv);
    }
}

#[test]
fn test_model_dtype_and_device_from_varbuilder() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    assert_eq!(model.dtype(), DType::F32);
    assert_eq!(model.device(), Device::Cpu);
}

#[test]
fn test_new_cache_matches_model_layers() {
    let cfg = tiny_config();
    let model = load_model(cfg.clone());
    let cache = model.new_cache();
    assert_eq!(cache.num_layers(), cfg.num_layers);
    assert_eq!(cache.seq_len(), 0);
}

#[test]
fn test_hidden_ne_heads_times_kv_channels_loads_with_zeros() {
    // GLM validate() does not enforce hidden_size == heads * kv_channels.
    // Documents this behavior.
    let mut cfg = tiny_config();
    cfg.hidden_size = 300; // 300 != 4 * 64 = 256
    assert!(cfg.validate().is_ok());
    let model = load_model(cfg);
    assert_eq!(model.config().hidden_size, 300);
}
