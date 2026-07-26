// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for GLM-5 configuration validation, layer construction, error types,
//! KV cache creation, config accessors, and input validation edge cases.
//!
//! Included from `glm5_tests.rs` via `#[path]`.

use super::*;
use crate::test_utils::tiny_config;
use nn_core::layers::kv_cache::KvCache;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device, TensorError};

// -- Config construction with valid parameters --------------------------------

#[test]
fn test_config_new_all_fields_roundtrip() {
    let cfg = Glm5Config::new(
        512,      // hidden_size
        2048,     // ffn_hidden_size
        6,        // num_layers
        8,        // num_attention_heads
        4,        // multi_query_group_num
        32000,    // padded_vocab_size
        64,       // kv_channels
        1e-6,     // layernorm_epsilon
        2048,     // seq_length
        true,     // rmsnorm
        false,    // add_qkv_bias
        true,     // add_bias_linear
        50_000.0, // rope_theta
    );
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.hidden_size, 512);
    assert_eq!(cfg.ffn_hidden_size, 2048);
    assert_eq!(cfg.num_layers, 6);
    assert_eq!(cfg.num_attention_heads, 8);
    assert_eq!(cfg.multi_query_group_num, 4);
    assert_eq!(cfg.padded_vocab_size, 32000);
    assert_eq!(cfg.kv_channels, 64);
    assert_eq!(cfg.layernorm_epsilon, 1e-6);
    assert_eq!(cfg.seq_length, 2048);
    assert!(cfg.rmsnorm);
    assert!(!cfg.add_qkv_bias);
    assert!(cfg.add_bias_linear);
    assert_eq!(cfg.rope_theta, 50_000.0);
}

#[test]
fn test_config_new_bias_linear_true_preserved() {
    let cfg = Glm5Config::new(
        256, 512, 2, 4, 2, 100, 64, 1e-5, 64, true, true, true, 10_000.0,
    );
    assert!(
        cfg.add_bias_linear,
        "add_bias_linear=true must be preserved"
    );
    assert!(cfg.validate().is_ok());
}

// -- Config validation: num_layers > 0 ----------------------------------------

#[test]
fn test_config_num_layers_exactly_one_validates() {
    let mut cfg = tiny_config();
    cfg.num_layers = 1;
    assert!(
        cfg.validate().is_ok(),
        "num_layers=1 should be the minimum valid value"
    );
}

#[test]
fn test_config_num_layers_zero_error_message_content() {
    let mut cfg = tiny_config();
    cfg.num_layers = 0;
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("num_layers") && msg.contains("0"),
        "error should mention num_layers and > 0: {msg}"
    );
}

// -- Config validation: hidden_size > 0 ---------------------------------------

#[test]
fn test_config_hidden_size_exactly_one_validates() {
    // hidden_size=1 is unusual but structurally valid
    let cfg = Glm5Config::new(1, 4, 1, 1, 1, 10, 4, 1e-5, 4, true, false, false, 10_000.0);
    assert!(cfg.validate().is_ok(), "hidden_size=1 should validate");
}

#[test]
fn test_config_hidden_size_zero_error_message_content() {
    let mut cfg = tiny_config();
    cfg.hidden_size = 0;
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("hidden_size"),
        "error should mention hidden_size: {msg}"
    );
}

// -- Config validation: head count consistency --------------------------------

#[test]
fn test_config_heads_not_divisible_by_kv_groups_error_message() {
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 7;
    cfg.multi_query_group_num = 3;
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("divisible"),
        "error should mention divisibility: {msg}"
    );
    assert!(msg.contains("7"), "error should mention head count: {msg}");
    assert!(
        msg.contains("3"),
        "error should mention kv group count: {msg}"
    );
}

#[test]
fn test_config_heads_equal_to_kv_groups_mha_validates() {
    // MHA: every query head has its own KV head
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 8;
    cfg.multi_query_group_num = 8;
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.num_kv_groups().unwrap(), 1, "MHA has group ratio 1");
}

#[test]
fn test_config_single_kv_group_mqa_validates() {
    // MQA: all query heads share a single KV head
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 16;
    cfg.multi_query_group_num = 1;
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.num_kv_groups().unwrap(), 16);
}

#[test]
fn test_config_heads_prime_number_kv_groups_fail() {
    // 13 is prime, so only divisible by 1 and 13
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 13;
    cfg.multi_query_group_num = 3;
    assert!(
        cfg.validate().is_err(),
        "13 heads with 3 kv groups should fail"
    );
}

// -- Error types: InvalidInput ------------------------------------------------

#[test]
fn test_error_invalid_input_display_format() {
    let err = Glm5Error::InvalidInput {
        reason: "input_ids len (5) != positions len (3)".into(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("invalid input"), "Display prefix: {msg}");
    assert!(msg.contains("5"), "reason preserved: {msg}");
    assert!(msg.contains("3"), "reason preserved: {msg}");
}

#[test]
fn test_error_invalid_input_to_tensor_error_preserves_message() {
    let err = Glm5Error::InvalidInput {
        reason: "test: embedding dim mismatch".into(),
    };
    let te: TensorError = err.into();
    let msg = format!("{te}");
    assert!(
        msg.contains("embedding dim mismatch"),
        "TensorError should preserve InvalidInput reason: {msg}"
    );
}

// -- Error types: CacheMismatch -----------------------------------------------

#[test]
fn test_error_cache_mismatch_fields_accessible() {
    let err = Glm5Error::CacheMismatch {
        cache_layers: 3,
        model_layers: 10,
    };
    let msg = format!("{err}");
    assert!(msg.contains("3"), "should contain cache_layers: {msg}");
    assert!(msg.contains("10"), "should contain model_layers: {msg}");
    assert!(msg.contains("cache mismatch"), "should have prefix: {msg}");
}

#[test]
fn test_error_cache_mismatch_to_tensor_error() {
    let err = Glm5Error::CacheMismatch {
        cache_layers: 7,
        model_layers: 2,
    };
    let te: TensorError = err.into();
    let msg = format!("{te}");
    assert!(
        msg.contains("7") && msg.contains("2"),
        "TensorError preserves cache/model layer counts: {msg}"
    );
}

// -- KV cache creation --------------------------------------------------------

#[test]
fn test_new_cache_correct_layer_count_tiny() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg.clone()).unwrap();
    let cache = model.new_cache();
    assert_eq!(cache.num_layers(), cfg.num_layers);
    assert_eq!(cache.num_layers(), 2);
}

#[test]
fn test_new_cache_correct_layer_count_single_layer() {
    let mut cfg = tiny_config();
    cfg.num_layers = 1;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    let cache = model.new_cache();
    assert_eq!(cache.num_layers(), 1);
    assert_eq!(cache.seq_len(), 0);
}

#[test]
fn test_new_cache_correct_layer_count_many_layers() {
    let mut cfg = tiny_config();
    cfg.num_layers = 12;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    let cache = model.new_cache();
    assert_eq!(cache.num_layers(), 12);
    assert!(cache.is_empty());
}

#[test]
fn test_new_cache_empty_initially() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    let cache = model.new_cache();
    assert_eq!(cache.seq_len(), 0);
    assert!(cache.is_empty());
}

// -- Config accessors: config(), dtype() --------------------------------------

#[test]
fn test_config_accessor_returns_same_config() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    let mc = model.config();
    // Verify all fields match what was passed in
    assert_eq!(mc.hidden_size, 256);
    assert_eq!(mc.ffn_hidden_size, 512);
    assert_eq!(mc.num_layers, 2);
    assert_eq!(mc.num_attention_heads, 4);
    assert_eq!(mc.multi_query_group_num, 2);
    assert_eq!(mc.padded_vocab_size, 100);
    assert_eq!(mc.kv_channels, 64);
    assert_eq!(mc.layernorm_epsilon, 1e-5);
    assert_eq!(mc.seq_length, 64);
    assert!(mc.rmsnorm);
    assert!(mc.add_qkv_bias);
    assert!(!mc.add_bias_linear);
    assert_eq!(mc.rope_theta, 10_000.0);
}

#[test]
fn test_dtype_accessor_f32() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    assert_eq!(model.dtype(), DType::F32);
}

#[test]
fn test_dtype_accessor_bf16() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    assert_eq!(model.dtype(), DType::BF16);
}

#[test]
fn test_dtype_accessor_f16() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F16, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    assert_eq!(model.dtype(), DType::F16);
}

// -- Forward pass input validation: mismatched lengths ------------------------

#[test]
fn test_forward_mismatched_lengths_error_message() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    let err = model.forward(&[0, 1, 2], &[0, 1]).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("3") && msg.contains("2"),
        "error should mention both lengths: {msg}"
    );
}

#[test]
fn test_forward_cached_mismatched_lengths() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    let err = model.forward_cached(&[0], &[0, 1], None);
    assert!(err.is_err(), "1 input_id vs 2 positions should error");
}

#[test]
fn test_forward_positions_longer_than_ids_error() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    let err = model.forward(&[0], &[0, 1, 2, 3]).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("1") && msg.contains("4"),
        "error should mention 1 and 4: {msg}"
    );
}

// -- Cache mismatch detection -------------------------------------------------

#[test]
fn test_cache_mismatch_too_many_layers() {
    let cfg = tiny_config(); // 2 layers
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    let mut cache = KvCache::new(10); // wrong: 10 layers

    let err = model.forward_cached(&[0], &[0], Some(&mut cache));
    assert!(
        err.is_err(),
        "10-layer cache with 2-layer model should fail"
    );
}

#[test]
fn test_cache_mismatch_too_few_layers() {
    let cfg = tiny_config(); // 2 layers
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    let mut cache = KvCache::new(1); // wrong: 1 layer

    let err = model.forward_cached(&[0], &[0], Some(&mut cache));
    assert!(err.is_err(), "1-layer cache with 2-layer model should fail");
}

#[test]
fn test_cache_mismatch_zero_layers() {
    let cfg = tiny_config(); // 2 layers
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    let mut cache = KvCache::new(0); // wrong: 0 layers

    let err = model.forward_cached(&[0], &[0], Some(&mut cache));
    assert!(err.is_err(), "0-layer cache with 2-layer model should fail");
}

#[test]
fn test_correct_cache_succeeds() {
    let cfg = tiny_config(); // 2 layers
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    let mut cache = model.new_cache(); // correct: 2 layers

    let result = model.forward_cached(&[0], &[0], Some(&mut cache));
    assert!(
        result.is_ok(),
        "correct cache should work: {:?}",
        result.err()
    );
}

// -- Embedding input validation -----------------------------------------------

#[test]
fn test_forward_from_embeddings_wrong_hidden_size_error_message() {
    let cfg = tiny_config(); // hidden_size=256
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    // hidden_size=128 instead of 256
    let embeddings = DynTensor::zeros(&[1, 2, 128], DType::F32, &Device::Cpu).unwrap();
    let err = model
        .forward_from_embeddings(&embeddings, &[0, 1], None)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("128") && msg.contains("256"),
        "error should mention both hidden sizes: {msg}"
    );
}

#[test]
fn test_forward_from_embeddings_wrong_seq_len_error_message() {
    let cfg = tiny_config(); // hidden_size=256
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    // seq_len=3 in tensor but 2 positions
    let embeddings = DynTensor::zeros(&[1, 3, 256], DType::F32, &Device::Cpu).unwrap();
    let err = model
        .forward_from_embeddings(&embeddings, &[0, 1], None)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("3") && msg.contains("2"),
        "error should mention seq_len mismatch: {msg}"
    );
}

#[test]
fn test_forward_from_embeddings_larger_hidden_size_rejected() {
    let cfg = tiny_config(); // hidden_size=256
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    // hidden_size=512 instead of 256
    let embeddings = DynTensor::zeros(&[1, 2, 512], DType::F32, &Device::Cpu).unwrap();
    let err = model.forward_from_embeddings(&embeddings, &[0, 1], None);
    assert!(err.is_err(), "larger hidden_size should be rejected");
}

#[test]
fn test_forward_from_embeddings_with_cache_wrong_layers() {
    let cfg = tiny_config(); // 2 layers
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    let embeddings = DynTensor::zeros(&[1, 2, 256], DType::F32, &Device::Cpu).unwrap();
    let mut cache = KvCache::new(5); // wrong layer count

    let err = model.forward_from_embeddings(&embeddings, &[0, 1], Some(&mut cache));
    assert!(
        err.is_err(),
        "forward_from_embeddings with wrong cache layers should fail"
    );
}

// -- Default config values match GLM-4 standard -------------------------------

#[test]
fn test_default_config_is_glm4_9b_standard() {
    let cfg = Glm5Config::default();
    // GLM-4-9B HuggingFace config.json values
    assert_eq!(cfg.hidden_size, 4096, "GLM-4-9B hidden_size");
    assert_eq!(cfg.ffn_hidden_size, 13696, "GLM-4-9B ffn_hidden_size");
    assert_eq!(cfg.num_layers, 40, "GLM-4-9B num_layers");
    assert_eq!(cfg.num_attention_heads, 32, "GLM-4-9B num_attention_heads");
    assert_eq!(
        cfg.multi_query_group_num, 2,
        "GLM-4-9B multi_query_group_num"
    );
    assert_eq!(cfg.padded_vocab_size, 151552, "GLM-4-9B padded_vocab_size");
    assert_eq!(cfg.kv_channels, 128, "GLM-4-9B kv_channels");
    assert_eq!(
        cfg.layernorm_epsilon, 1.5625e-5,
        "GLM-4-9B layernorm_epsilon"
    );
    assert_eq!(cfg.seq_length, 8192, "GLM-4-9B seq_length");
    assert!(cfg.rmsnorm, "GLM-4-9B uses RMSNorm");
    assert!(cfg.add_qkv_bias, "GLM-4-9B has QKV bias");
    assert!(!cfg.add_bias_linear, "GLM-4-9B has no linear bias");
    assert_eq!(cfg.rope_theta, 10_000.0, "GLM-4-9B rope_theta");
    // Validate the standard config
    assert!(cfg.validate().is_ok(), "GLM-4-9B default must validate");
}

#[test]
fn test_default_config_head_dim_consistency() {
    let cfg = Glm5Config::default();
    // GLM-4-9B: hidden_size (4096) = num_attention_heads (32) * kv_channels (128)
    assert_eq!(
        cfg.hidden_size,
        cfg.num_attention_heads * cfg.kv_channels,
        "hidden_size must equal num_attention_heads * kv_channels"
    );
}

#[test]
fn test_default_config_gqa_ratio() {
    let cfg = Glm5Config::default();
    // GLM-4-9B: 32 heads / 2 kv groups = 16 GQA groups
    let groups = cfg.num_kv_groups().unwrap();
    assert_eq!(groups, 16, "GLM-4-9B has 16 GQA groups");
}

// -- Config validation order: first failure wins ------------------------------

#[test]
fn test_config_validation_catches_zero_heads_before_hidden_size() {
    // Both heads and hidden are zero; validation should catch heads first
    // (validate() checks heads/kv_groups before hidden_size)
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 0;
    cfg.hidden_size = 0;
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    // The first check is heads/multi_query_group_num
    assert!(
        msg.contains("attention heads") || msg.contains("multi_query_group_num"),
        "should catch head validation first: {msg}"
    );
}

// -- Model load with invalid config rejects early ----------------------------

#[test]
fn test_model_load_rejects_invalid_config() {
    let mut cfg = tiny_config();
    cfg.num_layers = 0;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let err = Glm5Model::load(&vb, cfg);
    assert!(
        err.is_err(),
        "load() should reject invalid config (num_layers=0)"
    );
}

#[test]
fn test_model_load_rejects_zero_heads() {
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 0;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let err = Glm5Model::load(&vb, cfg);
    assert!(err.is_err(), "load() should reject config with zero heads");
}

// -- Flat KVCache (kv_cache.rs) construction ---------------------------------

#[test]
fn test_flat_kv_cache_new_is_empty() {
    let cache = kv_cache::KVCache::new(4, 2, 8, 32);
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.num_layers(), 4);
    assert_eq!(cache.num_heads(), 2);
    assert_eq!(cache.head_dim(), 8);
    assert_eq!(cache.max_seq_len(), 32);
}

#[test]
fn test_flat_kv_cache_clone_independence() {
    let mut cache = kv_cache::KVCache::new(2, 2, 4, 16);
    let token_size = 2 * 4;
    let key: Vec<f32> = (0..token_size).map(|i| i as f32).collect();
    let val: Vec<f32> = (0..token_size).map(|i| (i + 100) as f32).collect();

    cache.append(0, &key, &val).unwrap();
    cache.append(1, &key, &val).unwrap();
    assert_eq!(cache.len(), 1);

    let cache2 = cache.clone();
    assert_eq!(cache2.len(), 1);

    cache.clear();
    assert_eq!(cache.len(), 0);
    assert_eq!(cache2.len(), 1, "clone should be independent");
}

// -- Error variant exhaustiveness: all variants round-trip through Display ----

#[test]
fn test_all_error_variants_have_nonempty_display() {
    let errors: Vec<Glm5Error> = vec![
        Glm5Error::InvalidConfig { reason: "x".into() },
        Glm5Error::InvalidInput { reason: "y".into() },
        Glm5Error::CacheMismatch {
            cache_layers: 1,
            model_layers: 2,
        },
        Glm5Error::NonFiniteOutput {
            stage: "test",
            count: 1,
        },
        Glm5Error::WeightLoad { reason: "z".into() },
    ];
    for e in &errors {
        let display = format!("{e}");
        assert!(!display.is_empty(), "Display must be non-empty for {e:?}");
        let debug = format!("{e:?}");
        assert!(!debug.is_empty(), "Debug must be non-empty");
    }
}

// -- Config: small layer counts work with model construction -----------------

#[test]
fn test_model_with_single_layer_forward() {
    let mut cfg = tiny_config();
    cfg.num_layers = 1;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    let logits = model.forward(&[0, 1], &[0, 1]).unwrap();
    assert_eq!(logits.dims(), &[1, 2, 100]);
    let vals = logits.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|v| v.is_finite()));
}

#[test]
fn test_model_with_single_layer_cache() {
    let mut cfg = tiny_config();
    cfg.num_layers = 1;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();
    let mut cache = model.new_cache();
    assert_eq!(cache.num_layers(), 1);

    model.forward_cached(&[0], &[0], Some(&mut cache)).unwrap();
    assert_eq!(cache.seq_len(), 1);
}
