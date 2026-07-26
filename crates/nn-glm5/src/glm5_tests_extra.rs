// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional tests for GLM-4/5 crate: config, error, mask, model accessors.
//!
//! Included from `glm5_tests.rs` via `#[path]`.
//! Issue: #3797

use super::*;
use crate::test_utils::tiny_config;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// -- Config default field values -----------------------------------------------

#[test]
fn test_config_default_matches_glm4_9b_fields() {
    let cfg = Glm5Config::default();
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.ffn_hidden_size, 13696);
    assert_eq!(cfg.num_layers, 40);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.multi_query_group_num, 2);
    assert_eq!(cfg.padded_vocab_size, 151552);
    assert_eq!(cfg.kv_channels, 128);
    assert_eq!(cfg.seq_length, 8192);
    assert!(cfg.rmsnorm);
    assert!(cfg.add_qkv_bias);
    assert!(!cfg.add_bias_linear);
    assert_eq!(cfg.rope_theta, 10_000.0);
    assert_eq!(cfg.layernorm_epsilon, 1.5625e-5);
}

// -- head_dim identity ---------------------------------------------------------

#[test]
fn test_config_head_dim_is_kv_channels() {
    let cfg = tiny_config();
    assert_eq!(cfg.head_dim(), cfg.kv_channels);

    // Also check with default (GLM-4-9B) config
    let cfg_default = Glm5Config::default();
    assert_eq!(cfg_default.head_dim(), cfg_default.kv_channels);
}

// -- Config clone fidelity -----------------------------------------------------

#[test]
fn test_config_clone_deep_equality() {
    let original = tiny_config();
    let cloned = original.clone();

    assert_eq!(original.hidden_size, cloned.hidden_size);
    assert_eq!(original.ffn_hidden_size, cloned.ffn_hidden_size);
    assert_eq!(original.num_layers, cloned.num_layers);
    assert_eq!(original.num_attention_heads, cloned.num_attention_heads);
    assert_eq!(original.multi_query_group_num, cloned.multi_query_group_num);
    assert_eq!(original.padded_vocab_size, cloned.padded_vocab_size);
    assert_eq!(original.kv_channels, cloned.kv_channels);
    assert_eq!(original.layernorm_epsilon, cloned.layernorm_epsilon);
    assert_eq!(original.seq_length, cloned.seq_length);
    assert_eq!(original.rmsnorm, cloned.rmsnorm);
    assert_eq!(original.add_qkv_bias, cloned.add_qkv_bias);
    assert_eq!(original.add_bias_linear, cloned.add_bias_linear);
    assert_eq!(original.rope_theta, cloned.rope_theta);

    // Clone must also validate identically
    assert!(cloned.validate().is_ok());
}

// -- Error Display impls -------------------------------------------------------

#[test]
fn test_error_display_invalid_config() {
    let err = Glm5Error::InvalidConfig {
        reason: "test reason".into(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("invalid config"), "got: {msg}");
    assert!(msg.contains("test reason"), "got: {msg}");
}

#[test]
fn test_error_display_invalid_input() {
    let err = Glm5Error::InvalidInput {
        reason: "lengths mismatch".into(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("invalid input"), "got: {msg}");
    assert!(msg.contains("lengths mismatch"), "got: {msg}");
}

#[test]
fn test_error_display_cache_mismatch() {
    let err = Glm5Error::CacheMismatch {
        cache_layers: 5,
        model_layers: 40,
    };
    let msg = format!("{err}");
    assert!(msg.contains("cache mismatch"), "got: {msg}");
    assert!(msg.contains("5"), "got: {msg}");
    assert!(msg.contains("40"), "got: {msg}");
}

#[test]
fn test_error_display_non_finite_output() {
    let err = Glm5Error::NonFiniteOutput {
        stage: "Glm5MLP",
        count: 42,
    };
    let msg = format!("{err}");
    assert!(msg.contains("non-finite"), "got: {msg}");
    assert!(msg.contains("Glm5MLP"), "got: {msg}");
    assert!(msg.contains("42"), "got: {msg}");
}

#[test]
fn test_error_display_weight_load() {
    let err = Glm5Error::WeightLoad {
        reason: "missing tensor foo.weight".into(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("weight load"), "got: {msg}");
    assert!(msg.contains("missing tensor foo.weight"), "got: {msg}");
}

// -- Error → TensorError roundtrip for Tensor variant -------------------------

#[test]
fn test_error_tensor_variant_roundtrip() {
    use nn_core::TensorError;

    // Create a TensorError, wrap in Glm5Error::Tensor, convert back
    let original = TensorError::dtype_mismatch(DType::F32, DType::BF16);
    let glm_err = Glm5Error::Tensor(original);

    // Converting Glm5Error::Tensor back to TensorError should unwrap
    let recovered: TensorError = glm_err.into();
    let msg = format!("{recovered}");
    // DType Display may use lowercase (f32/bf16) or uppercase (F32/BF16)
    let msg_lower = msg.to_lowercase();
    assert!(
        msg_lower.contains("f32") || msg_lower.contains("bf16"),
        "recovered error should preserve dtype info: {msg}"
    );
}

// -- Causal mask with offset shape --------------------------------------------

#[test]
fn test_causal_mask_with_offset_has_correct_shape() {
    // 3 new tokens with 5 total → mask is [1, 1, 3, 5]
    let mask = causal_mask_with_offset(3, 5, DType::F32, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 3, 5]);
}

#[test]
fn test_causal_mask_with_offset_values() {
    // 2 new tokens, 4 total (2 cached + 2 new)
    // Row 0 (token at position 2): can attend to positions 0,1,2 → [0,0,0,-inf]
    // Row 1 (token at position 3): can attend to all → [0,0,0,0]
    let mask = causal_mask_with_offset(2, 4, DType::F32, &Device::Cpu).unwrap();
    let data = mask.to_flat_vec::<f32>().unwrap();
    assert_eq!(data.len(), 2 * 4); // 2 rows x 4 cols

    // Row 0: [0, 0, 0, -inf]
    assert_eq!(data[0], 0.0);
    assert_eq!(data[1], 0.0);
    assert_eq!(data[2], 0.0);
    assert!(data[3].is_infinite() && data[3] < 0.0);

    // Row 1: [0, 0, 0, 0]
    assert_eq!(data[4], 0.0);
    assert_eq!(data[5], 0.0);
    assert_eq!(data[6], 0.0);
    assert_eq!(data[7], 0.0);
}

// -- Model accessor tests -----------------------------------------------------

#[test]
fn test_model_config_accessor() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    let model_cfg = model.config();
    assert_eq!(model_cfg.hidden_size, 256);
    assert_eq!(model_cfg.num_layers, 2);
    assert_eq!(model_cfg.num_attention_heads, 4);
}

#[test]
fn test_model_dtype_accessor_f32() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    assert_eq!(model.dtype(), DType::F32);
}

#[test]
fn test_model_dtype_accessor_bf16() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    assert_eq!(model.dtype(), DType::BF16);
}

// -- Config validation edge cases ---------------------------------------------

#[test]
fn test_config_validation_neg_inf_epsilon() {
    let mut cfg = tiny_config();
    cfg.layernorm_epsilon = f64::NEG_INFINITY;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validation_neg_inf_rope_theta() {
    let mut cfg = tiny_config();
    cfg.rope_theta = f64::NEG_INFINITY;
    assert!(cfg.validate().is_err());
}
