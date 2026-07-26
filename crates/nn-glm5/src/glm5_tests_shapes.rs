// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional tests for GLM-4/5: shape invariants, error conversions, config
//! edge cases, forward path coverage.
//!
//! Included from `glm5_tests.rs` via `#[path]`.
//! Issue: #3797

use super::*;
use crate::test_utils::tiny_config;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// -- Config: MHA mode (num_heads == multi_query_group_num) --------------------

#[test]
fn test_config_mha_mode_validates() {
    // MHA: every query head has its own KV head (no sharing)
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 4;
    cfg.multi_query_group_num = 4;
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.num_kv_groups().unwrap(), 1);
}

#[test]
fn test_config_mha_mode_model_loads() {
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 4;
    cfg.multi_query_group_num = 4;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg);
    assert!(model.is_ok(), "MHA mode should load: {:?}", model.err());
}

// -- Config: single KV head (MQA) mode ----------------------------------------

#[test]
fn test_config_mqa_mode_validates() {
    // MQA: single KV head shared across all query heads
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 4;
    cfg.multi_query_group_num = 1;
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.num_kv_groups().unwrap(), 4);
}

// -- Config: default num_kv_groups correctness --------------------------------

#[test]
fn test_default_config_num_kv_groups() {
    let cfg = Glm5Config::default();
    // GLM-4-9B: 32 heads / 2 kv groups = 16
    let groups = cfg.num_kv_groups().unwrap();
    assert_eq!(groups, 16);
}

// -- Config Debug impl --------------------------------------------------------

#[test]
fn test_config_debug_contains_field_names() {
    let cfg = tiny_config();
    let dbg = format!("{cfg:?}");
    assert!(
        dbg.contains("hidden_size"),
        "Debug must show hidden_size: {dbg}"
    );
    assert!(
        dbg.contains("ffn_hidden_size"),
        "Debug must show ffn_hidden_size: {dbg}"
    );
    assert!(
        dbg.contains("num_layers"),
        "Debug must show num_layers: {dbg}"
    );
    assert!(
        dbg.contains("num_attention_heads"),
        "Debug must show num_attention_heads: {dbg}"
    );
    assert!(
        dbg.contains("multi_query_group_num"),
        "Debug must show multi_query_group_num: {dbg}"
    );
    assert!(
        dbg.contains("rope_theta"),
        "Debug must show rope_theta: {dbg}"
    );
}

// -- Error conversion: all non-Tensor variants → TensorError ------------------

#[test]
fn test_error_conversion_weight_load_to_tensor_error() {
    use nn_core::TensorError;

    let err = Glm5Error::WeightLoad {
        reason: "missing query_key_value.weight".into(),
    };
    let te: TensorError = err.into();
    let msg = format!("{te}");
    assert!(
        msg.contains("missing query_key_value.weight"),
        "TensorError should preserve reason: {msg}"
    );
}

#[test]
fn test_error_conversion_non_finite_output_to_tensor_error() {
    use nn_core::TensorError;

    let err = Glm5Error::NonFiniteOutput {
        stage: "Glm5MLP",
        count: 7,
    };
    let te: TensorError = err.into();
    let msg = format!("{te}");
    assert!(
        msg.contains("Glm5MLP"),
        "TensorError should preserve stage: {msg}"
    );
    assert!(
        msg.contains("7"),
        "TensorError should preserve count: {msg}"
    );
}

// -- Forward: embedding seq_len mismatch --------------------------------------

#[test]
fn test_forward_from_embeddings_seq_len_mismatch() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    // 3 positions but embedding has seq_len=2
    let embeddings = DynTensor::zeros(&[1, 2, 256], DType::F32, &Device::Cpu).unwrap();
    let err = model.forward_from_embeddings(&embeddings, &[0, 1, 2], None);
    assert!(err.is_err(), "seq_len mismatch should error");
}

// -- Forward: causal mask with offset (single token) --------------------------

#[test]
fn test_causal_mask_with_offset_single_new_token() {
    // 1 new token, 3 total → mask is [1, 1, 1, 3]
    // Single new token can attend to all 3 positions
    let mask = causal_mask_with_offset(1, 3, DType::F32, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 1, 3]);
    let data = mask.to_flat_vec::<f32>().unwrap();
    // All positions should be attendable (0.0, not -inf)
    assert_eq!(data, &[0.0, 0.0, 0.0]);
}

// -- Forward: model with all biases enabled + forward -------------------------

#[test]
fn test_model_all_biases_forward() {
    let mut cfg = tiny_config();
    cfg.add_qkv_bias = true;
    cfg.add_bias_linear = true;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    let logits = model.forward(&[0, 1], &[0, 1]).unwrap();
    assert_eq!(logits.dims(), &[1, 2, 100]);
    let vals = logits.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "all biased outputs must be finite"
    );
}

// -- Forward: multi-step cached decode (3+ steps) -----------------------------

#[test]
fn test_model_cached_forward_three_steps() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    let mut cache = model.new_cache();

    // Prompt: 3 tokens
    let logits0 = model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(logits0.dims(), &[1, 3, 100]);
    assert_eq!(cache.seq_len(), 3);

    // Decode step 1
    let logits1 = model.forward_cached(&[3], &[3], Some(&mut cache)).unwrap();
    assert_eq!(logits1.dims(), &[1, 1, 100]);
    assert_eq!(cache.seq_len(), 4);

    // Decode step 2
    let logits2 = model.forward_cached(&[4], &[4], Some(&mut cache)).unwrap();
    assert_eq!(logits2.dims(), &[1, 1, 100]);
    assert_eq!(cache.seq_len(), 5);

    // Decode step 3
    let logits3 = model.forward_cached(&[5], &[5], Some(&mut cache)).unwrap();
    assert_eq!(logits3.dims(), &[1, 1, 100]);
    assert_eq!(cache.seq_len(), 6);
}

// -- Config: boundary kv_channels values --------------------------------------

#[test]
fn test_config_kv_channels_minimum_valid() {
    // Minimum valid kv_channels: 4 (first positive multiple of 4)
    let mut cfg = tiny_config();
    cfg.kv_channels = 4;
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.head_dim(), 4);
}

#[test]
fn test_config_kv_channels_3_rejected() {
    let mut cfg = tiny_config();
    cfg.kv_channels = 3;
    assert!(cfg.validate().is_err());
}

// -- Config: negative epsilon -------------------------------------------------

#[test]
fn test_config_negative_epsilon_rejected() {
    let mut cfg = tiny_config();
    cfg.layernorm_epsilon = -0.001;
    assert!(cfg.validate().is_err());
}

// -- Error: Glm5Error is Send + Sync -----------------------------------------

#[test]
fn test_glm5_error_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    // Glm5Error must be Send + Sync for use in async/threaded contexts
    assert_send_sync::<Glm5Error>();
}

// -- Config new: no-qkv-bias mode preserved -----------------------------------

#[test]
fn test_config_new_no_qkv_bias_preserved() {
    let cfg = Glm5Config::new(
        256, 512, 2, 4, 2, 100, 64, 1e-5, 64, true,  // rmsnorm
        false, // add_qkv_bias = false
        false, // add_bias_linear = false
        10_000.0,
    );
    assert!(
        !cfg.add_qkv_bias,
        "no-qkv-bias must be preserved through new()"
    );
    assert!(cfg.validate().is_ok());
}
