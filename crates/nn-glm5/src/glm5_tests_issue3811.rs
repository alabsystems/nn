// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional tests for GLM-4/5 crate: config edge cases, error conversions,
//! causal mask variants, model forward-path coverage.
//!
//! Included from `glm5_tests.rs` via `#[path]`.
//! Issue: #3811

use super::*;
use crate::test_utils::tiny_config;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device, TensorError};

// -- Config: positive infinity epsilon ----------------------------------------

#[test]
fn test_config_validation_positive_inf_epsilon() {
    let mut cfg = tiny_config();
    cfg.layernorm_epsilon = f64::INFINITY;
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("layernorm_epsilon"),
        "error should mention layernorm_epsilon: {msg}"
    );
}

// -- Config: NaN rope_theta ---------------------------------------------------

#[test]
fn test_config_validation_nan_rope_theta() {
    let mut cfg = tiny_config();
    cfg.rope_theta = f64::NAN;
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("rope_theta"),
        "error should mention rope_theta: {msg}"
    );
}

// -- Config: zero hidden_size alone -------------------------------------------

#[test]
fn test_config_validation_zero_hidden_size() {
    let mut cfg = tiny_config();
    cfg.hidden_size = 0;
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("hidden_size"),
        "error should mention hidden_size: {msg}"
    );
}

// -- Config: zero ffn_hidden_size alone ---------------------------------------

#[test]
fn test_config_validation_zero_ffn_hidden_size() {
    let mut cfg = tiny_config();
    cfg.ffn_hidden_size = 0;
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("ffn_hidden_size"),
        "error should mention ffn_hidden_size: {msg}"
    );
}

// -- Config: kv_channels boundary values not multiple of 4 --------------------

#[test]
fn test_config_kv_channels_1_rejected() {
    let mut cfg = tiny_config();
    cfg.kv_channels = 1;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_kv_channels_2_rejected() {
    let mut cfg = tiny_config();
    cfg.kv_channels = 2;
    assert!(cfg.validate().is_err());
}

// -- Config: large valid kv_channels ------------------------------------------

#[test]
fn test_config_kv_channels_256_valid() {
    let mut cfg = tiny_config();
    cfg.kv_channels = 256;
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.head_dim(), 256);
}

// -- Error conversion: InvalidConfig → TensorError ----------------------------

#[test]
fn test_error_conversion_invalid_config_to_tensor_error() {
    let err = Glm5Error::InvalidConfig {
        reason: "heads must be > 0".into(),
    };
    let te: TensorError = err.into();
    let msg = format!("{te}");
    assert!(
        msg.contains("heads must be > 0"),
        "TensorError should preserve InvalidConfig reason: {msg}"
    );
}

// -- Error conversion: InvalidInput → TensorError -----------------------------

#[test]
fn test_error_conversion_invalid_input_to_tensor_error() {
    let err = Glm5Error::InvalidInput {
        reason: "input_ids len (3) != positions len (2)".into(),
    };
    let te: TensorError = err.into();
    let msg = format!("{te}");
    assert!(
        msg.contains("input_ids len"),
        "TensorError should preserve InvalidInput reason: {msg}"
    );
}

// -- Error conversion: CacheMismatch → TensorError ----------------------------

#[test]
fn test_error_conversion_cache_mismatch_to_tensor_error() {
    let err = Glm5Error::CacheMismatch {
        cache_layers: 10,
        model_layers: 40,
    };
    let te: TensorError = err.into();
    let msg = format!("{te}");
    assert!(
        msg.contains("10") && msg.contains("40"),
        "TensorError should preserve CacheMismatch layer counts: {msg}"
    );
}

// -- Error Debug impl for all variants ----------------------------------------

#[test]
fn test_error_debug_all_variants() {
    // Verify Debug is implemented and produces non-empty output for every variant
    let variants: Vec<Glm5Error> = vec![
        Glm5Error::InvalidConfig {
            reason: "test".into(),
        },
        Glm5Error::InvalidInput {
            reason: "test".into(),
        },
        Glm5Error::CacheMismatch {
            cache_layers: 1,
            model_layers: 2,
        },
        Glm5Error::NonFiniteOutput {
            stage: "test",
            count: 1,
        },
        Glm5Error::WeightLoad {
            reason: "test".into(),
        },
    ];
    for variant in &variants {
        let dbg = format!("{variant:?}");
        assert!(!dbg.is_empty(), "Debug output must be non-empty");
        // Debug output should contain the variant name
        let display = format!("{variant}");
        assert!(!display.is_empty(), "Display output must be non-empty");
    }
}

// -- Causal mask: size 2 detailed values --------------------------------------

#[test]
fn test_causal_mask_size_2_values() {
    let mask = causal_mask(2, DType::F32, &Device::Cpu).unwrap();
    let data = mask.to_flat_vec::<f32>().unwrap();
    assert_eq!(data.len(), 4);
    // Row 0: [0, -inf]
    assert_eq!(data[0], 0.0);
    assert!(data[1].is_infinite() && data[1] < 0.0);
    // Row 1: [0, 0]
    assert_eq!(data[2], 0.0);
    assert_eq!(data[3], 0.0);
}

// -- Causal mask with offset: new == total (no cached tokens) -----------------

#[test]
fn test_causal_mask_with_offset_no_cache() {
    // 3 new tokens, 3 total → same as causal_mask(3)
    let mask_offset = causal_mask_with_offset(3, 3, DType::F32, &Device::Cpu).unwrap();
    let mask_plain = causal_mask(3, DType::F32, &Device::Cpu).unwrap();
    assert_eq!(mask_offset.dims(), mask_plain.dims());
    let data_offset = mask_offset.to_flat_vec::<f32>().unwrap();
    let data_plain = mask_plain.to_flat_vec::<f32>().unwrap();
    // Compare element-wise (both contain -inf, so use bitwise comparison)
    assert_eq!(data_offset.len(), data_plain.len());
    for (a, b) in data_offset.iter().zip(data_plain.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "mask values should be identical");
    }
}

// -- Model: forward_cached with None cache same as forward --------------------

#[test]
fn test_model_forward_cached_none_same_as_forward() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    let ids = &[0, 1, 2];
    let pos = &[0, 1, 2];
    let logits_forward = model.forward(ids, pos).unwrap();
    let logits_cached = model.forward_cached(ids, pos, None).unwrap();

    assert_eq!(logits_forward.dims(), logits_cached.dims());
    let v1 = logits_forward.to_flat_vec::<f32>().unwrap();
    let v2 = logits_cached.to_flat_vec::<f32>().unwrap();
    for (a, b) in v1.iter().zip(v2.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "forward() and forward_cached(None) must produce identical results"
        );
    }
}

// -- Model: deterministic forward (same input → same output) ------------------

#[test]
fn test_model_forward_deterministic() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    let ids = &[5, 10, 15];
    let pos = &[0, 1, 2];
    let out1 = model
        .forward(ids, pos)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let out2 = model
        .forward(ids, pos)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(out1, out2, "forward must be deterministic");
}

// -- Config: validate after new() with invalid params -------------------------

#[test]
fn test_config_new_invalid_then_validate_fails() {
    // zero num_layers via new()
    let cfg = Glm5Config::new(
        256, 512, 0, /* num_layers=0 */ 4, 2, 100, 64, 1e-5, 64, true, true, false, 10_000.0,
    );
    assert!(cfg.validate().is_err());
}

// -- Config: num_kv_groups with various GQA ratios ----------------------------

#[test]
fn test_config_num_kv_groups_various_ratios() {
    // 8 heads / 4 kv groups = 2
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 8;
    cfg.multi_query_group_num = 4;
    assert_eq!(cfg.num_kv_groups().unwrap(), 2);

    // 16 heads / 8 kv groups = 2
    cfg.num_attention_heads = 16;
    cfg.multi_query_group_num = 8;
    assert_eq!(cfg.num_kv_groups().unwrap(), 2);

    // 32 heads / 1 kv group = 32 (extreme MQA)
    cfg.num_attention_heads = 32;
    cfg.multi_query_group_num = 1;
    assert_eq!(cfg.num_kv_groups().unwrap(), 32);
}

// -- Model: forward_from_embeddings with 2D input (wrong rank) ----------------

#[test]
fn test_forward_from_embeddings_wrong_rank() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    // 2D tensor instead of 3D: dims3() should fail
    let embeddings = DynTensor::zeros(&[3, 256], DType::F32, &Device::Cpu).unwrap();
    let err = model.forward_from_embeddings(&embeddings, &[0, 1, 2], None);
    assert!(err.is_err(), "2D input should be rejected (needs 3D)");
}

// -- Model: cache seq_len starts at zero --------------------------------------

#[test]
fn test_model_new_cache_starts_empty() {
    let cfg = tiny_config();
    let num_layers = cfg.num_layers;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    let cache = model.new_cache();
    assert_eq!(cache.seq_len(), 0, "fresh cache must have seq_len 0");
    assert_eq!(cache.num_layers(), num_layers);
}

// -- Causal mask: BF16 dtype --------------------------------------------------

#[test]
fn test_causal_mask_bf16_shape() {
    let mask = causal_mask(3, DType::BF16, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 3, 3]);
}

// -- Config: kv_channels = 8 valid --------------------------------------------

#[test]
fn test_config_kv_channels_8_valid() {
    let mut cfg = tiny_config();
    cfg.kv_channels = 8;
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.head_dim(), 8);
}
