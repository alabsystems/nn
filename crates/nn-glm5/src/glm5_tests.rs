#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::test_utils::tiny_config;
use nn_core::layers::repeat_kv;
use nn_core::{DType, Device};

// -- Config validation --------------------------------------------------------

#[test]
fn test_config_validation_ok() {
    let cfg = tiny_config();
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.head_dim(), 64);
    assert_eq!(cfg.num_kv_groups().unwrap(), 2);
}

#[test]
fn test_config_validation_zero_heads() {
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validation_zero_kv_groups() {
    let mut cfg = tiny_config();
    cfg.multi_query_group_num = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_num_kv_groups_zero_divisor_returns_error() {
    let mut cfg = tiny_config();
    cfg.multi_query_group_num = 0;
    assert!(cfg.num_kv_groups().is_err());
}

#[test]
fn test_num_kv_groups_not_divisible_returns_error() {
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 5;
    cfg.multi_query_group_num = 2;
    assert!(cfg.num_kv_groups().is_err());
}

#[test]
fn test_config_validation_heads_not_divisible() {
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 5;
    cfg.multi_query_group_num = 2;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validation_zero_vocab_size() {
    let mut cfg = tiny_config();
    cfg.padded_vocab_size = 0;
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("padded_vocab_size"),
        "error should mention padded_vocab_size: {msg}"
    );
}

#[test]
fn test_config_validation_zero_kv_channels() {
    let mut cfg = tiny_config();
    cfg.kv_channels = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validation_kv_channels_not_multiple_of_4() {
    let mut cfg = tiny_config();
    cfg.kv_channels = 65; // not divisible by 4 → HalfRotaryEmbedding would fail
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validation_zero_seq_length() {
    let mut cfg = tiny_config();
    cfg.seq_length = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validation_nan_layernorm_epsilon() {
    let mut cfg = tiny_config();
    cfg.layernorm_epsilon = f64::NAN;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validation_zero_layernorm_epsilon() {
    let mut cfg = tiny_config();
    cfg.layernorm_epsilon = 0.0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validation_negative_rope_theta() {
    let mut cfg = tiny_config();
    cfg.rope_theta = -10_000.0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validation_inf_rope_theta() {
    let mut cfg = tiny_config();
    cfg.rope_theta = f64::INFINITY;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_gqa_groups() {
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 8;
    cfg.multi_query_group_num = 2;
    assert_eq!(cfg.num_kv_groups().unwrap(), 4);
}

// -- Causal mask ---------------------------------------------------------------

#[test]
fn test_causal_mask_shape() {
    let mask = causal_mask(4, DType::F32, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 4, 4]);
}

#[test]
fn test_causal_mask_values() {
    let mask = causal_mask(3, DType::F32, &Device::Cpu).unwrap();
    let data = mask.to_flat_vec::<f32>().unwrap();
    // Row 0: [0, -inf, -inf]
    assert_eq!(data[0], 0.0);
    assert!(data[1].is_infinite() && data[1] < 0.0);
    assert!(data[2].is_infinite() && data[2] < 0.0);
    // Row 1: [0, 0, -inf]
    assert_eq!(data[3], 0.0);
    assert_eq!(data[4], 0.0);
    assert!(data[5].is_infinite() && data[5] < 0.0);
    // Row 2: [0, 0, 0]
    assert_eq!(data[6], 0.0);
    assert_eq!(data[7], 0.0);
    assert_eq!(data[8], 0.0);
}

#[test]
fn test_causal_mask_single_token() {
    let mask = causal_mask(1, DType::F32, &Device::Cpu).unwrap();
    let data = mask.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, &[0.0]);
}

#[test]
fn test_causal_mask_total_less_than_new_returns_error() {
    let result = causal_mask_with_offset(5, 3, DType::F32, &Device::Cpu);
    assert!(result.is_err(), "total_tokens < new_tokens should error");
}

// -- repeat_kv ----------------------------------------------------------------

#[test]
fn test_repeat_kv_identity() {
    let x = DynTensor::ones(&[1, 2, 3, 64], DType::F32, &Device::Cpu).unwrap();
    let result = repeat_kv(&x, 1).unwrap();
    assert_eq!(result.dims(), &[1, 2, 3, 64]);
}

#[test]
fn test_repeat_kv_expand() {
    let data: Vec<f32> = (0..2 * 2 * 4).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data, &[1, 2, 2, 4], &Device::Cpu).unwrap();
    let result = repeat_kv(&x, 2).unwrap();
    assert_eq!(result.dims(), &[1, 4, 2, 4]);

    let flat = result.to_flat_vec::<f32>().unwrap();
    let head_size = 2 * 4;
    assert_eq!(&flat[0..head_size], &flat[head_size..2 * head_size]);
}

// Model forward pass and BF16 tests extracted to `glm5_tests_forward.rs`
// for 500-line compliance.
#[path = "glm5_tests_forward.rs"]
mod tests_forward;

// Additional tests: config, error Display, mask, model accessors (#3797).
#[path = "glm5_tests_extra.rs"]
mod tests_extra;

// Shape invariants, error conversions, config edge cases (#3797).
#[path = "glm5_tests_shapes.rs"]
mod tests_shapes;

// Config edge cases, error conversions, mask variants, forward coverage (#3811).
#[path = "glm5_tests_issue3811.rs"]
mod tests_issue3811;

// Architecture validation: production configs, parameter counts, shape propagation (#3942).
#[path = "glm5_tests_arch_validation.rs"]
mod arch_validation;

// Wave 33: glm4_9b_chat config, KV cache consistency, error chain, edge cases (#4274).
#[path = "glm5_tests_wave33.rs"]
mod wave33;

// Comprehensive architecture tests: half-RoPE, SwiGLU, QKV split, cache lifecycle,
// cached-vs-uncached consistency, embedding shapes, config invariants (#4353).
#[path = "glm5_tests_arch_comprehensive.rs"]
mod arch_comprehensive;

// Config construction, layer validation, error types, KV cache creation,
// accessors, input validation edge cases.
#[path = "glm5_tests_config_layer.rs"]
mod config_layer;

// -- Glm5Config::new() constructor tests (#1655) -----------------------------

#[test]
fn test_config_new_matches_struct_literal() {
    let from_new = Glm5Config::new(
        256, 512, 2, 4, 2, 100, 64, 1e-5, 64, true, true, false, 10_000.0,
    );
    let from_literal = tiny_config();
    assert_eq!(from_new.hidden_size, from_literal.hidden_size);
    assert_eq!(from_new.ffn_hidden_size, from_literal.ffn_hidden_size);
    assert_eq!(from_new.num_layers, from_literal.num_layers);
    assert_eq!(
        from_new.num_attention_heads,
        from_literal.num_attention_heads
    );
    assert_eq!(
        from_new.multi_query_group_num,
        from_literal.multi_query_group_num
    );
    assert_eq!(from_new.padded_vocab_size, from_literal.padded_vocab_size);
    assert_eq!(from_new.kv_channels, from_literal.kv_channels);
    assert_eq!(from_new.layernorm_epsilon, from_literal.layernorm_epsilon);
    assert_eq!(from_new.seq_length, from_literal.seq_length);
    assert_eq!(from_new.rmsnorm, from_literal.rmsnorm);
    assert_eq!(from_new.add_qkv_bias, from_literal.add_qkv_bias);
    assert_eq!(from_new.add_bias_linear, from_literal.add_bias_linear);
    assert_eq!(from_new.rope_theta, from_literal.rope_theta);
    assert!(from_new.validate().is_ok());
}

#[test]
fn test_config_new_validates_correctly() {
    // Construct a valid config via new(), validate should pass
    let cfg = Glm5Config::new(
        4096, 13696, 40, 32, 2, 151552, 128, 1.5625e-5, 8192, true, true, false, 10_000.0,
    );
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.head_dim(), 128);
    assert_eq!(cfg.num_kv_groups().unwrap(), 16);
}

// -- num_layers validation (Finding 6) ----------------------------------------

#[test]
fn test_config_validation_zero_num_layers() {
    let mut cfg = tiny_config();
    cfg.num_layers = 0;
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("num_layers"),
        "error should mention num_layers: {msg}"
    );
}

// -- GLM-4-9B config ----------------------------------------------------------

#[test]
fn test_glm4_9b_config() {
    // Validate the actual GLM-4-9B configuration from HuggingFace
    let cfg = Glm5Config {
        hidden_size: 4096,
        ffn_hidden_size: 13696,
        num_layers: 40,
        num_attention_heads: 32,
        multi_query_group_num: 2,
        padded_vocab_size: 151552,
        kv_channels: 128,
        layernorm_epsilon: 1.5625e-5,
        seq_length: 8192,
        rmsnorm: true,
        add_qkv_bias: true,
        add_bias_linear: false,
        rope_theta: 10_000.0,
    };
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.head_dim(), 128);
    assert_eq!(cfg.num_kv_groups().unwrap(), 16); // 32 heads / 2 kv groups
}
