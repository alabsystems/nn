#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use crate::dyn_tensor::DynTensor;
use crate::layers::kv_cache::KvCache;
use crate::{DType, Device, TensorError};

use super::{generate, GenerationConfig, GenerationOutput};

/// Mock model: returns logits where token `(step % vocab)` has highest logit.
/// This makes the generation deterministic: step 0 → token 0, step 1 → token 1, etc.
fn deterministic_model(input: &DynTensor, _cache: &mut KvCache) -> crate::Result<DynTensor> {
    // Use the last input token value as a "step counter".
    // ids_to_tensor creates U32 tensors, so convert to f32 first.
    let input_f32 = input.to_dtype(DType::F32)?;
    let flat = input_f32.to_flat_vec::<f32>()?;
    let last_val = flat[flat.len() - 1];
    let next_token = (last_val as usize + 1) % 5;

    // Return logits [1, vocab_size=5] with next_token having highest logit
    let mut logits = vec![0.0f32; 5];
    logits[next_token] = 10.0;
    DynTensor::from_vec(logits, &[1, 5], &Device::Cpu)
}

/// Model that always returns token 2 (the EOS token).
fn eos_model(_input: &DynTensor, _cache: &mut KvCache) -> crate::Result<DynTensor> {
    let mut logits = vec![0.0f32; 5];
    logits[2] = 10.0; // token 2 always wins
    DynTensor::from_vec(logits, &[1, 5], &Device::Cpu)
}

/// Model that returns 3D logits [1, seq_len, vocab] to test 3D handling.
fn model_3d_logits(input: &DynTensor, _cache: &mut KvCache) -> crate::Result<DynTensor> {
    let seq_len = input.dim(1)?;
    // Return [1, seq_len, 5] where last position has token 3 highest
    let mut data = vec![0.0f32; seq_len * 5];
    // Set last position, token 3 to highest
    data[(seq_len - 1) * 5 + 3] = 10.0;
    DynTensor::from_vec(data, &[1, seq_len, 5], &Device::Cpu)
}

// -- GenerationConfig tests ---------------------------------------------------

#[test]
fn test_generation_config_default() {
    let config = GenerationConfig::default();
    assert_eq!(config.max_new_tokens, 128);
    assert_eq!(config.temperature, 0.0);
    assert!(config.top_k.is_none());
    assert!(config.eos_token_id.is_none());
    assert!(config.seed.is_none());
}

// -- generate() tests ---------------------------------------------------------

#[test]
fn test_generate_empty_prompt_returns_error() {
    let config = GenerationConfig::default();
    let mut cache = KvCache::new(1);
    let result = generate(deterministic_model, &[], &mut cache, &config, &Device::Cpu);
    assert!(result.is_err());
}

#[test]
fn test_generate_zero_tokens() {
    let config = GenerationConfig {
        max_new_tokens: 0,
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = generate(deterministic_model, &[0], &mut cache, &config, &Device::Cpu).unwrap();
    assert!(result.token_ids.is_empty());
    assert!(!result.finished);
}

#[test]
fn test_generate_greedy_deterministic() {
    let config = GenerationConfig {
        max_new_tokens: 3,
        temperature: 0.0,
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = generate(deterministic_model, &[0], &mut cache, &config, &Device::Cpu).unwrap();
    assert_eq!(result.token_ids.len(), 3);
    assert!(!result.finished);
}

#[test]
fn test_generate_stops_at_eos() {
    let config = GenerationConfig {
        max_new_tokens: 100,
        eos_token_id: Some(2),
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = generate(eos_model, &[0], &mut cache, &config, &Device::Cpu).unwrap();
    // First token should be 2 (EOS), so generation stops immediately
    assert_eq!(result.token_ids.len(), 1);
    assert_eq!(result.token_ids[0], 2);
    assert!(result.finished);
}

#[test]
fn test_generate_3d_logits() {
    let config = GenerationConfig {
        max_new_tokens: 2,
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = generate(model_3d_logits, &[0, 1], &mut cache, &config, &Device::Cpu).unwrap();
    assert_eq!(result.token_ids.len(), 2);
    // Both tokens should be 3 (highest logit in last position)
    assert_eq!(result.token_ids[0], 3);
    assert_eq!(result.token_ids[1], 3);
}

#[test]
fn test_generate_model_error_propagates() {
    fn failing_model(_input: &DynTensor, _cache: &mut KvCache) -> crate::Result<DynTensor> {
        Err(TensorError::InvalidShape("mock model error".into()))
    }

    let config = GenerationConfig {
        max_new_tokens: 5,
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = generate(failing_model, &[0], &mut cache, &config, &Device::Cpu);
    assert!(result.is_err());
}

#[test]
fn test_generate_multi_token_prompt() {
    let config = GenerationConfig {
        max_new_tokens: 1,
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    // Prompt with multiple tokens
    let result = generate(
        deterministic_model,
        &[10, 20, 30],
        &mut cache,
        &config,
        &Device::Cpu,
    )
    .unwrap();
    assert_eq!(result.token_ids.len(), 1);
}

#[test]
fn test_generate_with_topk_no_rand() {
    // Without rand feature, top_k + temperature > 0 uses argmax fallback
    let config = GenerationConfig {
        max_new_tokens: 2,
        temperature: 1.0,
        top_k: Some(3),
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = generate(deterministic_model, &[0], &mut cache, &config, &Device::Cpu).unwrap();
    assert_eq!(result.token_ids.len(), 2);
}

#[test]
fn test_generation_output_debug() {
    // GenerationOutput derives Debug
    let output = GenerationOutput {
        token_ids: vec![1, 2, 3],
        finished: true,
    };
    let dbg = format!("{output:?}");
    assert!(dbg.contains("token_ids"));
    assert!(dbg.contains("finished"));
}

// -- top_k_indices boundary tests (algorithm_audit) ----------------------------

#[test]
fn test_top_k_indices_k0_returns_empty() {
    // Regression: k=0 caused usize underflow (k-1 = usize::MAX) panic
    // before the early-return guard was added.
    let indices = super::top_k_indices(&[3.0, 1.0, 2.0], 0);
    assert!(indices.is_empty());
}

#[test]
fn test_top_k_indices_k_equals_len() {
    // k == values.len(): should return all indices, sorted descending.
    let indices = super::top_k_indices(&[1.0, 3.0, 2.0], 3);
    assert_eq!(indices.len(), 3);
    assert_eq!(indices[0], 1); // value 3.0
    assert_eq!(indices[1], 2); // value 2.0
    assert_eq!(indices[2], 0); // value 1.0
}

#[test]
fn test_top_k_indices_k_greater_than_len() {
    // k > values.len(): should return all indices (capped by available).
    let indices = super::top_k_indices(&[1.0, 3.0], 5);
    assert_eq!(indices.len(), 2);
    assert_eq!(indices[0], 1); // value 3.0
    assert_eq!(indices[1], 0); // value 1.0
}

#[test]
fn test_top_k_indices_nan_handling() {
    // NaN in values: should not panic, NaN sorted to consistent position.
    let values = [1.0, f32::NAN, 3.0, 2.0];
    let indices = super::top_k_indices(&values, 2);
    assert_eq!(indices.len(), 2);
    // All returned indices must be valid.
    for &idx in &indices {
        assert!(idx < values.len());
    }
}

#[test]
fn test_top_k_indices_all_equal() {
    // All values equal: top-k should return k indices (any k valid).
    let indices = super::top_k_indices(&[5.0, 5.0, 5.0, 5.0], 2);
    assert_eq!(indices.len(), 2);
    for &idx in &indices {
        assert!(idx < 4);
    }
}

#[test]
fn test_top_k_indices_single_element() {
    let indices = super::top_k_indices(&[42.0], 1);
    assert_eq!(indices, vec![0]);
}

#[test]
fn test_argmax_with_neg_infinity() {
    // argmax on all-neg-inf should return index 0 (or any valid index).
    let idx = super::argmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY]);
    assert!(idx < 2);
}

#[test]
fn test_argmax_single_element() {
    assert_eq!(super::argmax(&[1.0]), 0);
}

#[test]
fn test_argmax_nan_stability() {
    // NaN comparisons yield Equal, so argmax should not panic.
    let idx = super::argmax(&[f32::NAN, 1.0, f32::NAN]);
    assert!(idx < 3);
}

// -- top_p_filter tests -------------------------------------------------------

#[test]
fn test_top_p_filter_concentrates_mass() {
    // With a skewed distribution [0.7, 0.2, 0.1], top_p=0.8 should keep
    // only the first two tokens (cumsum 0.7 + 0.2 = 0.9 >= 0.8).
    let probs = vec![(0, 0.7), (1, 0.2), (2, 0.1)];
    let filtered = super::top_p_filter(probs, 0.8);
    // Should keep indices 0 and 1 (already sorted descending by prob).
    assert_eq!(filtered.len(), 2);
    assert_eq!(filtered[0].0, 0);
    assert_eq!(filtered[1].0, 1);
    // Renormalized: 0.7/0.9 ≈ 0.778, 0.2/0.9 ≈ 0.222
    let sum: f32 = filtered.iter().map(|&(_, p)| p).sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "filtered probs should sum to ~1.0"
    );
}

#[test]
fn test_top_p_filter_1_0_keeps_all() {
    // top_p = 1.0 should keep all tokens.
    let probs = vec![(0, 0.5), (1, 0.3), (2, 0.2)];
    let filtered = super::top_p_filter(probs, 1.0);
    assert_eq!(filtered.len(), 3);
}

#[test]
fn test_top_p_filter_very_small_keeps_at_least_one() {
    // top_p near 0 should keep at least the top token.
    let probs = vec![(0, 0.3), (1, 0.3), (2, 0.4)];
    let filtered = super::top_p_filter(probs, 0.01);
    assert!(!filtered.is_empty(), "must keep at least one token");
    // The highest-prob token (0.4 at index 2) should be the one kept.
    assert_eq!(filtered[0].0, 2);
}

#[test]
fn test_top_p_filter_empty_input() {
    let probs: Vec<(usize, f32)> = vec![];
    let filtered = super::top_p_filter(probs, 0.9);
    assert!(filtered.is_empty());
}

#[test]
fn test_top_p_filter_single_token() {
    let probs = vec![(42, 1.0)];
    let filtered = super::top_p_filter(probs, 0.5);
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].0, 42);
    assert!((filtered[0].1 - 1.0).abs() < 1e-6);
}

#[test]
fn test_generate_with_top_p_no_rand() {
    // Without rand feature, top_p + temperature > 0 uses argmax fallback.
    // The highest-logit token should still be chosen.
    let config = GenerationConfig {
        max_new_tokens: 2,
        temperature: 1.0,
        top_p: Some(0.9),
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = generate(deterministic_model, &[0], &mut cache, &config, &Device::Cpu).unwrap();
    assert_eq!(result.token_ids.len(), 2);
}

#[test]
fn test_generate_top_p_and_top_k_compose() {
    // Both top_k and top_p set: top_k filters first, then top_p filters further.
    let config = GenerationConfig {
        max_new_tokens: 2,
        temperature: 1.0,
        top_k: Some(3),
        top_p: Some(0.5),
        ..Default::default()
    };
    let mut cache = KvCache::new(1);
    let result = generate(deterministic_model, &[0], &mut cache, &config, &Device::Cpu).unwrap();
    assert_eq!(result.token_ids.len(), 2);
}

#[test]
fn test_generation_config_default_has_no_top_p() {
    let config = GenerationConfig::default();
    assert!(config.top_p.is_none());
}

// Validate tests extracted to autoregressive_validate_tests.rs for 500-line compliance.
#[path = "autoregressive_validate_tests.rs"]
mod validate_tests;
