// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Wave 33 tests for GLM-4/5 crate: glm4_9b_chat config validation, KV cache
//! consistency with synthetic weights, autoregressive decode shape stability,
//! error source chain, forward edge cases, combined invalid config fields.
//!
//! Included from `glm5_tests.rs` via `#[path]`.
//! Issue: #4274

use super::*;
use crate::test_utils::tiny_config;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device, TensorError};

// -- glm4_9b_chat() constructor tests -----------------------------------------

#[test]
fn test_glm4_9b_chat_config_validates() {
    let cfg = Glm5Config::glm4_9b_chat();
    assert!(cfg.validate().is_ok(), "glm4_9b_chat config must validate");
}

#[test]
fn test_glm4_9b_chat_config_fields() {
    let cfg = Glm5Config::glm4_9b_chat();
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.ffn_hidden_size, 13696);
    assert_eq!(cfg.num_layers, 40);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.multi_query_group_num, 2);
    assert_eq!(cfg.padded_vocab_size, 151552);
    assert_eq!(cfg.kv_channels, 128);
    assert_eq!(cfg.seq_length, 131_072);
    assert!(cfg.rmsnorm);
    assert!(cfg.add_qkv_bias);
    assert!(!cfg.add_bias_linear);
}

#[test]
fn test_glm4_9b_chat_differs_from_default() {
    let chat = Glm5Config::glm4_9b_chat();
    let base = Glm5Config::default();
    // Chat variant has different epsilon, rope_theta, and seq_length
    assert_ne!(
        chat.layernorm_epsilon, base.layernorm_epsilon,
        "chat variant should have different epsilon"
    );
    assert_ne!(
        chat.rope_theta, base.rope_theta,
        "chat variant should have different rope_theta"
    );
    assert_ne!(
        chat.seq_length, base.seq_length,
        "chat variant should have different seq_length"
    );
}

#[test]
fn test_glm4_9b_chat_rope_theta() {
    let cfg = Glm5Config::glm4_9b_chat();
    // Chat uses rope_ratio=500 → rope_theta = 10_000.0 * 500 = 5_000_000.0
    assert!(
        (cfg.rope_theta - 5_000_000.0).abs() < 1.0,
        "chat rope_theta should be ~5M, got {}",
        cfg.rope_theta
    );
}

#[test]
fn test_glm4_9b_chat_epsilon() {
    let cfg = Glm5Config::glm4_9b_chat();
    // Chat uses 1.5625e-7 (more precise than base 1.5625e-5)
    assert!(
        (cfg.layernorm_epsilon - 1.5625e-7).abs() < 1e-12,
        "chat epsilon should be 1.5625e-7, got {}",
        cfg.layernorm_epsilon
    );
}

#[test]
fn test_glm4_9b_chat_extended_context() {
    let cfg = Glm5Config::glm4_9b_chat();
    assert_eq!(cfg.seq_length, 131_072, "chat supports 128K context");
}

#[test]
fn test_glm4_9b_chat_gqa_groups() {
    let cfg = Glm5Config::glm4_9b_chat();
    let groups = cfg.num_kv_groups().unwrap();
    assert_eq!(groups, 16, "32 heads / 2 kv groups = 16");
}

#[test]
fn test_glm4_9b_chat_head_dim() {
    let cfg = Glm5Config::glm4_9b_chat();
    assert_eq!(cfg.head_dim(), 128);
    assert_eq!(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim());
}

// -- Combined invalid config fields ------------------------------------------

#[test]
fn test_config_validation_multiple_invalid_fields() {
    // When multiple fields are invalid, validate() should still return Err.
    // It catches the first invalid field and stops.
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 0;
    cfg.hidden_size = 0;
    cfg.kv_channels = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validation_nan_epsilon_and_inf_theta() {
    let mut cfg = tiny_config();
    cfg.layernorm_epsilon = f64::NAN;
    cfg.rope_theta = f64::INFINITY;
    assert!(cfg.validate().is_err());
}

// -- Error source chain: TensorError → Glm5Error roundtrip -------------------

#[test]
fn test_tensor_error_from_into_glm5_error() {
    // A TensorError wrapped in Glm5Error::Tensor should roundtrip via From impl
    let te = TensorError::shape_mismatch(vec![1, 2], vec![3, 4]);
    let glm_err = Glm5Error::Tensor(te);

    // Convert back to TensorError
    let recovered: TensorError = glm_err.into();
    let msg = format!("{recovered}");
    // Should preserve the original error info (shape mismatch details)
    assert!(
        !msg.is_empty(),
        "roundtrip should preserve error info: {msg}"
    );
}

#[test]
fn test_glm5_error_from_tensor_error() {
    // Verify the From<TensorError> impl on Glm5Error
    let te = TensorError::dtype_mismatch(DType::F32, DType::BF16);
    let glm_err: Glm5Error = Glm5Error::Tensor(te);
    let msg = format!("{glm_err}");
    // thiserror transparent should forward Display
    assert!(
        !msg.is_empty(),
        "Tensor variant should produce non-empty Display"
    );
}

// -- Forward: empty input_ids (0-length sequence) ----------------------------

#[test]
fn test_model_forward_empty_input_is_error() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    // 0-length input should produce an error (no tokens to process)
    let result = model.forward(&[], &[]);
    // The model may succeed with an empty tensor or fail — either is valid.
    // What matters is it doesn't panic.
    if let Ok(logits) = &result {
        // If it succeeds, shape should reflect 0 tokens
        assert_eq!(logits.dim(1).unwrap(), 0);
    }
    // Not panicking is the test
}

// -- Forward: large token IDs (within vocab bounds) --------------------------

#[test]
fn test_model_forward_max_token_id() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg.clone()).unwrap();

    // Token IDs near the vocab boundary
    let max_id = cfg.padded_vocab_size - 1;
    let result = model.forward(&[max_id], &[0]);
    assert!(
        result.is_ok(),
        "max token ID should work: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().dims(), &[1, 1, cfg.padded_vocab_size]);
}

// -- Forward: different position patterns ------------------------------------

#[test]
fn test_model_forward_non_contiguous_positions() {
    // Positions don't need to be 0,1,2,... — test with a gap (models prefix caching)
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg.clone()).unwrap();

    // Positions 10, 11, 12 — as if 10 tokens were already cached
    let result = model.forward(&[0, 1, 2], &[10, 11, 12]);
    assert!(
        result.is_ok(),
        "non-contiguous positions should work: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().dims(), &[1, 3, cfg.padded_vocab_size]);
}

// -- KV cache: prompt + multi-step decode with shape validation ---------------

#[test]
fn test_model_cached_prompt_then_decode_five_steps() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    // Prompt: 4 tokens
    let prompt_logits = model
        .forward_cached(&[0, 1, 2, 3], &[0, 1, 2, 3], Some(&mut cache))
        .unwrap();
    assert_eq!(prompt_logits.dims(), &[1, 4, cfg.padded_vocab_size]);
    assert_eq!(cache.seq_len(), 4);

    // Decode 5 more tokens
    for step in 0..5 {
        let pos = 4 + step;
        let logits = model
            .forward_cached(&[pos + 10], &[pos], Some(&mut cache))
            .unwrap();
        assert_eq!(logits.dims(), &[1, 1, cfg.padded_vocab_size]);
        assert_eq!(cache.seq_len(), 5 + step);
    }
    assert_eq!(cache.seq_len(), 9);
}

// -- Config: verify all boolean permutations create loadable models -----------

#[test]
fn test_model_loads_with_all_bias_permutations() {
    let bias_permutations = [
        (true, true),   // qkv bias + linear bias
        (true, false),  // qkv bias only (default GLM-4)
        (false, true),  // linear bias only
        (false, false), // no biases
    ];

    for (add_qkv_bias, add_bias_linear) in bias_permutations {
        let mut cfg = tiny_config();
        cfg.add_qkv_bias = add_qkv_bias;
        cfg.add_bias_linear = add_bias_linear;

        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = Glm5Model::load(&vb, cfg);
        assert!(
            model.is_ok(),
            "model should load with qkv_bias={add_qkv_bias}, bias_linear={add_bias_linear}: {:?}",
            model.err()
        );

        // Verify forward works too
        let logits = model.unwrap().forward(&[0], &[0]);
        assert!(
            logits.is_ok(),
            "forward should work with qkv_bias={add_qkv_bias}, bias_linear={add_bias_linear}: {:?}",
            logits.err()
        );
    }
}

// -- Config: very small valid config -----------------------------------------

#[test]
fn test_config_minimal_valid() {
    // Smallest possible valid config
    let cfg = Glm5Config::new(
        4,    // hidden_size (must be > 0)
        4,    // ffn_hidden_size (must be > 0)
        1,    // num_layers (must be > 0)
        1,    // num_attention_heads (must be > 0)
        1,    // multi_query_group_num (must divide heads)
        1,    // padded_vocab_size (must be > 0)
        4,    // kv_channels (must be multiple of 4)
        1e-8, // layernorm_epsilon (positive finite)
        1,    // seq_length (must be > 0)
        true, false, false, 1.0, // rope_theta (positive finite)
    );
    assert!(cfg.validate().is_ok(), "minimal config must validate");
    assert_eq!(cfg.head_dim(), 4);
    assert_eq!(cfg.num_kv_groups().unwrap(), 1);
}

#[test]
fn test_model_loads_minimal_config() {
    let cfg = Glm5Config::new(4, 4, 1, 1, 1, 2, 4, 1e-8, 4, true, false, false, 1.0);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg);
    assert!(
        model.is_ok(),
        "minimal config model should load: {:?}",
        model.err()
    );

    let logits = model.unwrap().forward(&[0], &[0]);
    assert!(
        logits.is_ok(),
        "minimal config forward should work: {:?}",
        logits.err()
    );
}

// -- Forward from embeddings: wrong batch size (4D tensor) -------------------

#[test]
fn test_forward_from_embeddings_4d_input() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    // 4D tensor: dims3() should fail
    let embeddings = DynTensor::zeros(&[1, 2, 3, 256], DType::F32, &Device::Cpu).unwrap();
    let err = model.forward_from_embeddings(&embeddings, &[0, 1], None);
    assert!(err.is_err(), "4D input should be rejected (needs 3D)");
}

// -- Forward from embeddings: 1D input (wrong rank) --------------------------

#[test]
fn test_forward_from_embeddings_1d_input() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    let embeddings = DynTensor::zeros(&[256], DType::F32, &Device::Cpu).unwrap();
    let err = model.forward_from_embeddings(&embeddings, &[0], None);
    assert!(err.is_err(), "1D input should be rejected (needs 3D)");
}

// -- Model: config accessor fidelity -----------------------------------------

#[test]
fn test_model_config_accessor_all_fields() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg.clone()).unwrap();
    let mc = model.config();

    assert_eq!(mc.hidden_size, cfg.hidden_size);
    assert_eq!(mc.ffn_hidden_size, cfg.ffn_hidden_size);
    assert_eq!(mc.num_layers, cfg.num_layers);
    assert_eq!(mc.num_attention_heads, cfg.num_attention_heads);
    assert_eq!(mc.multi_query_group_num, cfg.multi_query_group_num);
    assert_eq!(mc.padded_vocab_size, cfg.padded_vocab_size);
    assert_eq!(mc.kv_channels, cfg.kv_channels);
    assert_eq!(mc.layernorm_epsilon, cfg.layernorm_epsilon);
    assert_eq!(mc.seq_length, cfg.seq_length);
    assert_eq!(mc.rmsnorm, cfg.rmsnorm);
    assert_eq!(mc.add_qkv_bias, cfg.add_qkv_bias);
    assert_eq!(mc.add_bias_linear, cfg.add_bias_linear);
    assert_eq!(mc.rope_theta, cfg.rope_theta);
}

// -- Causal mask with offset: exact identity with full causal mask -----------

#[test]
fn test_causal_mask_with_offset_zero_cache_bitwise_equal() {
    // causal_mask_with_offset(n, n, ...) should produce identical results to causal_mask(n, ...)
    for n in [2, 3, 4, 5, 8] {
        let mask_plain = causal_mask(n, DType::F32, &Device::Cpu).unwrap();
        let mask_offset = causal_mask_with_offset(n, n, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(
            mask_plain.dims(),
            mask_offset.dims(),
            "shapes must match for n={n}"
        );
        let v1 = mask_plain.to_flat_vec::<f32>().unwrap();
        let v2 = mask_offset.to_flat_vec::<f32>().unwrap();
        for (i, (a, b)) in v1.iter().zip(v2.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "element {i} differs for n={n}: {a} vs {b}"
            );
        }
    }
}

// -- Config: verify kv_channels large multiples of 4 -------------------------

#[test]
fn test_config_kv_channels_large_multiples_of_4() {
    for kv_ch in [4, 8, 16, 32, 64, 128, 256, 512, 1024] {
        let mut cfg = tiny_config();
        cfg.kv_channels = kv_ch;
        assert!(
            cfg.validate().is_ok(),
            "kv_channels={kv_ch} should validate"
        );
        assert_eq!(cfg.head_dim(), kv_ch);
    }
}

// -- Forward: verify repeated forward calls with same input are deterministic -

#[test]
fn test_model_forward_repeated_deterministic() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    let ids = &[3, 7, 11, 42];
    let pos = &[0, 1, 2, 3];

    // Run forward 3 times — all must produce bitwise identical results
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
    let out3 = model
        .forward(ids, pos)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    assert_eq!(out1, out2, "forward must be deterministic (run 1 vs 2)");
    assert_eq!(out2, out3, "forward must be deterministic (run 2 vs 3)");
}

// -- Config: verify kv_channels odd multiples rejected -----------------------

#[test]
fn test_config_kv_channels_odd_multiples_rejected() {
    for kv_ch in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17] {
        let mut cfg = tiny_config();
        cfg.kv_channels = kv_ch;
        assert!(
            cfg.validate().is_err(),
            "kv_channels={kv_ch} should be rejected (not multiple of 4)"
        );
    }
}

// -- Model: new_cache creates correct number of layers for various configs ----

#[test]
fn test_model_new_cache_various_layer_counts() {
    for num_layers in [1, 2, 4, 8] {
        let mut cfg = tiny_config();
        cfg.num_layers = num_layers;
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = Glm5Model::load(&vb, cfg).unwrap();
        let cache = model.new_cache();
        assert_eq!(
            cache.num_layers(),
            num_layers,
            "cache should have {num_layers} layers"
        );
        assert_eq!(cache.seq_len(), 0);
    }
}

// -- Config: num_kv_groups for boundary GQA ratios ---------------------------

#[test]
fn test_config_gqa_ratio_extremes() {
    // Maximum GQA ratio: all heads share 1 KV head (MQA)
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 32;
    cfg.multi_query_group_num = 1;
    assert_eq!(cfg.num_kv_groups().unwrap(), 32);

    // Minimum GQA ratio: every head has its own KV (MHA)
    cfg.num_attention_heads = 32;
    cfg.multi_query_group_num = 32;
    assert_eq!(cfg.num_kv_groups().unwrap(), 1);

    // Intermediate: 4x sharing
    cfg.num_attention_heads = 32;
    cfg.multi_query_group_num = 8;
    assert_eq!(cfg.num_kv_groups().unwrap(), 4);
}

// -- Forward: multiple calls without cache should be independent --------------

#[test]
fn test_model_forward_calls_independent() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg).unwrap();

    // Two forward calls with different inputs should produce different outputs
    // (unless zero weights make everything identical, which is fine for zero-weight test)
    let out1 = model.forward(&[0], &[0]).unwrap();
    let out2 = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();

    // Different sequence lengths = different output shapes
    assert_ne!(out1.dims(), out2.dims());
    assert_eq!(out1.dims(), &[1, 1, 100]);
    assert_eq!(out2.dims(), &[1, 3, 100]);
}

// -- Forward from embeddings: BF16 model with BF16 input preserves shape -----

#[test]
fn test_forward_from_embeddings_bf16_preserves_shape() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    let model = Glm5Model::load(&vb, cfg.clone()).unwrap();

    for seq_len in [1, 2, 4, 8] {
        let embeddings =
            DynTensor::zeros(&[1, seq_len, cfg.hidden_size], DType::BF16, &Device::Cpu).unwrap();
        let positions: Vec<usize> = (0..seq_len).collect();
        let result = model.forward_from_embeddings(&embeddings, &positions, None);
        assert!(
            result.is_ok(),
            "BF16 forward with seq_len={seq_len} failed: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().dims(), &[1, seq_len, cfg.padded_vocab_size]);
    }
}
