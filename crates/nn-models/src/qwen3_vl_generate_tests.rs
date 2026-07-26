// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Qwen3-VL KV cache integration and autoregressive generation.

use super::*;
use crate::qwen3_vl::{Qwen3VL, Qwen3VLConfig};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::KvCache;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

/// Helper: build a tiny Qwen3-VL model from zero weights for testing.
fn build_tiny_model() -> (Qwen3VL, Qwen3VLConfig) {
    let cfg = Qwen3VLConfig {
        hidden_size: 64,
        num_heads: 4,
        num_kv_heads: 2,
        intermediate_size: 128,
        num_layers: 2,
        vocab_size: 32,
        vision_hidden: 32,
        vision_heads: 4,
        vision_layers: 2,
        vision_patch_size: 14,
        vision_temporal_patch: 2,
        rms_norm_eps: 1e-6,
        num_experts: 0,
        active_experts: 0,
    };
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3VL::load(&vb, cfg.clone()).unwrap();
    (model, cfg)
}

// ---------------------------------------------------------------------------
// GenerationConfig tests
// ---------------------------------------------------------------------------

#[test]
fn test_generation_config_defaults() {
    let cfg = Qwen3VLGenerationConfig::default();
    assert_eq!(cfg.max_new_tokens, 128);
    assert!((cfg.temperature - 0.0).abs() < f64::EPSILON);
    assert!(cfg.top_p.is_none());
    assert!(cfg.eos_token_id.is_none());
    cfg.validate().unwrap();
}

#[test]
fn test_generation_config_builder() {
    let cfg = Qwen3VLGenerationConfig::new(64)
        .with_temperature(0.7)
        .with_top_p(0.9)
        .with_eos_token_id(2);
    assert_eq!(cfg.max_new_tokens, 64);
    assert!((cfg.temperature - 0.7).abs() < 1e-9);
    assert_eq!(cfg.top_p, Some(0.9));
    assert_eq!(cfg.eos_token_id, Some(2));
    cfg.validate().unwrap();
}

#[test]
fn test_generation_config_rejects_negative_temperature() {
    let cfg = Qwen3VLGenerationConfig::new(10).with_temperature(-1.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_generation_config_rejects_nan_temperature() {
    let cfg = Qwen3VLGenerationConfig {
        temperature: f64::NAN,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_generation_config_rejects_invalid_top_p() {
    // top_p = 0.0 is out of range (must be > 0)
    let cfg = Qwen3VLGenerationConfig::new(10).with_top_p(0.0);
    assert!(cfg.validate().is_err());

    // top_p > 1.0 is out of range
    let cfg = Qwen3VLGenerationConfig::new(10).with_top_p(1.5);
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// KV cache shape and forward_cached tests
// ---------------------------------------------------------------------------

#[test]
fn test_create_cache_layer_count() {
    let (model, cfg) = build_tiny_model();
    let cache = model.create_cache();
    assert_eq!(cache.num_layers(), cfg.num_layers);
    assert!(cache.is_empty());
    assert_eq!(cache.seq_len(), 0);
}

#[test]
fn test_cache_shape_per_layer() {
    let (model, cfg) = build_tiny_model();
    let mut cache = model.create_cache();

    // Run prefill with 5 text tokens
    let prompt: Vec<usize> = (0..5).collect();
    let logits = model.forward_cached(None, &prompt, &mut cache).unwrap();

    // Logits shape: [1, 1, vocab_size] (last position only)
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);

    // Cache should have seq_len == 5 (the 5 prefilled positions)
    assert_eq!(cache.seq_len(), 5);

    // Each layer's cache should have key/value tensors
    for i in 0..cfg.num_layers {
        let layer = cache.layer(i).unwrap();
        assert_eq!(layer.seq_len(), 5);
        let key = layer.key().unwrap();
        assert!(key.is_some(), "layer {i} should have cached keys");
        let key = key.unwrap();
        // Key shape: [1, num_kv_heads, 5, head_dim]
        assert_eq!(key.dim(0).unwrap(), 1); // batch
        assert_eq!(key.dim(1).unwrap(), cfg.num_kv_heads); // KV heads
        assert_eq!(key.dim(2).unwrap(), 5); // seq_len
        assert_eq!(key.dim(3).unwrap(), cfg.head_dim()); // head_dim
    }
}

#[test]
fn test_forward_cached_text_only() {
    let (model, cfg) = build_tiny_model();
    let mut cache = model.create_cache();

    let prompt: Vec<usize> = (0..8).collect();
    let logits = model.forward_cached(None, &prompt, &mut cache).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    assert_eq!(cache.seq_len(), 8);
}

#[test]
fn test_forward_cached_with_vision() {
    let (model, cfg) = build_tiny_model();
    let mut cache = model.create_cache();

    let vis = DynTensor::zeros(&[1, 4, cfg.vision_hidden], DType::F32, &Device::Cpu).unwrap();
    let prompt: Vec<usize> = (0..3).collect();
    let logits = model
        .forward_cached(Some(&vis), &prompt, &mut cache)
        .unwrap();

    // Logits: [1, 1, vocab_size]
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    // Cache: 4 vision tokens + 3 text tokens = 7
    assert_eq!(cache.seq_len(), 7);
}

#[test]
fn test_forward_cached_decode_step() {
    let (model, cfg) = build_tiny_model();
    let mut cache = model.create_cache();

    // Prefill
    let prompt: Vec<usize> = (0..5).collect();
    model.forward_cached(None, &prompt, &mut cache).unwrap();
    assert_eq!(cache.seq_len(), 5);

    // Decode step: single token
    let logits = model.forward_cached(None, &[10], &mut cache).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    assert_eq!(cache.seq_len(), 6);

    // Another decode step
    let logits = model.forward_cached(None, &[15], &mut cache).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    assert_eq!(cache.seq_len(), 7);
}

#[test]
fn test_cache_mismatch_error() {
    let (model, _cfg) = build_tiny_model();
    // Create cache with wrong number of layers
    let mut cache = KvCache::new(99);
    let result = model.forward_cached(None, &[1, 2, 3], &mut cache);
    assert!(result.is_err(), "mismatched cache layers should error");
}

// ---------------------------------------------------------------------------
// Generation tests
// ---------------------------------------------------------------------------

#[test]
fn test_generate_greedy_shape() {
    let (model, _cfg) = build_tiny_model();
    let mut generator = Qwen3VLGenerator::new(&model);
    let config = Qwen3VLGenerationConfig::new(10);

    let prompt: Vec<usize> = (0..5).collect();
    let output = generator.generate(&prompt, None, &config).unwrap();

    // Should produce up to max_new_tokens tokens
    assert!(!output.token_ids.is_empty());
    assert!(output.token_ids.len() <= 10);
    // All tokens should be valid vocab indices
    for &tok in &output.token_ids {
        assert!(tok < 32, "token {tok} exceeds vocab_size 32");
    }
}

#[test]
fn test_generate_with_temperature() {
    let (model, _cfg) = build_tiny_model();
    let mut generator = Qwen3VLGenerator::new(&model);
    let config = Qwen3VLGenerationConfig::new(5).with_temperature(1.0);

    let prompt: Vec<usize> = (0..3).collect();
    let output = generator.generate(&prompt, None, &config).unwrap();
    assert!(!output.token_ids.is_empty());
    assert!(output.token_ids.len() <= 5);
}

#[test]
fn test_max_tokens_limit() {
    let (model, _cfg) = build_tiny_model();
    let mut generator = Qwen3VLGenerator::new(&model);
    let config = Qwen3VLGenerationConfig::new(3);

    let prompt: Vec<usize> = (0..5).collect();
    let output = generator.generate(&prompt, None, &config).unwrap();
    assert!(output.token_ids.len() <= 3);
}

#[test]
fn test_generate_zero_max_tokens() {
    let (model, _cfg) = build_tiny_model();
    let mut generator = Qwen3VLGenerator::new(&model);
    let config = Qwen3VLGenerationConfig::new(0);

    let prompt: Vec<usize> = (0..5).collect();
    let output = generator.generate(&prompt, None, &config).unwrap();
    assert!(output.token_ids.is_empty());
    assert!(!output.finished);
}

#[test]
fn test_generate_empty_prompt_error() {
    let (model, _cfg) = build_tiny_model();
    let mut generator = Qwen3VLGenerator::new(&model);
    let config = Qwen3VLGenerationConfig::new(10);

    let result = generator.generate(&[], None, &config);
    assert!(result.is_err(), "empty prompt should error");
}

#[test]
fn test_generate_text_only() {
    let (model, _cfg) = build_tiny_model();
    let mut generator = Qwen3VLGenerator::new(&model);
    let config = Qwen3VLGenerationConfig::new(5);

    let prompt: Vec<usize> = vec![1, 2, 3, 4];
    let output = generator.generate(&prompt, None, &config).unwrap();
    assert!(!output.token_ids.is_empty());
}

#[test]
fn test_generate_with_vision() {
    let (model, cfg) = build_tiny_model();
    let mut generator = Qwen3VLGenerator::new(&model);
    let config = Qwen3VLGenerationConfig::new(5);

    let vis = DynTensor::zeros(&[1, 4, cfg.vision_hidden], DType::F32, &Device::Cpu).unwrap();
    let prompt: Vec<usize> = vec![1, 2, 3];
    let output = generator.generate(&prompt, Some(&vis), &config).unwrap();
    assert!(!output.token_ids.is_empty());
}

#[test]
fn test_generate_with_eos() {
    let (model, _cfg) = build_tiny_model();
    let mut generator = Qwen3VLGenerator::new(&model);
    // With zero weights, all logits are equal, argmax returns index 0.
    // Set eos=0 so generation stops after first token.
    let config = Qwen3VLGenerationConfig::new(100).with_eos_token_id(0);

    let prompt: Vec<usize> = vec![1, 2, 3];
    let output = generator.generate(&prompt, None, &config).unwrap();
    assert!(output.finished);
    assert_eq!(output.token_ids.len(), 1);
    assert_eq!(output.token_ids[0], 0);
}

#[test]
fn test_generator_reset() {
    let (model, _cfg) = build_tiny_model();
    let mut generator = Qwen3VLGenerator::new(&model);
    let config = Qwen3VLGenerationConfig::new(3);

    let prompt: Vec<usize> = vec![1, 2, 3];
    generator.generate(&prompt, None, &config).unwrap();
    assert!(generator.cached_seq_len() > 0);

    generator.reset();
    assert_eq!(generator.cached_seq_len(), 0);
    assert!(generator.cache().is_empty());
}

#[test]
fn test_prefill_decode_equivalence() {
    let (model, _cfg) = build_tiny_model();

    // Full forward (non-cached): get logits for last position
    let prompt: Vec<usize> = (0..6).collect();
    let full_logits = model.forward(None, &prompt).unwrap();
    // full_logits: [1, 6, vocab_size] — extract last position
    let full_last = full_logits.narrow(1, 5, 1).unwrap();
    let full_last_vec = full_last.to_flat_vec::<f32>().unwrap();

    // Cached prefill: should produce equivalent last-position logits
    let mut cache = model.create_cache();
    let cached_logits = model.forward_cached(None, &prompt, &mut cache).unwrap();
    // cached_logits: [1, 1, vocab_size]
    let cached_vec = cached_logits.to_flat_vec::<f32>().unwrap();

    assert_eq!(full_last_vec.len(), cached_vec.len());
    for (i, (a, b)) in full_last_vec.iter().zip(cached_vec.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-4,
            "logit mismatch at index {i}: full={a}, cached={b}, diff={diff}"
        );
    }
}

#[test]
fn test_generate_with_top_p() {
    let (model, _cfg) = build_tiny_model();
    let mut generator = Qwen3VLGenerator::new(&model);
    let config = Qwen3VLGenerationConfig::new(5)
        .with_temperature(1.0)
        .with_top_p(0.9);

    let prompt: Vec<usize> = vec![1, 2, 3];
    let output = generator.generate(&prompt, None, &config).unwrap();
    assert!(!output.token_ids.is_empty());
    assert!(output.token_ids.len() <= 5);
}
