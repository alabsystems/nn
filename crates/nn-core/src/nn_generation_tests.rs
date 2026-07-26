// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for generation utilities: BeamSearchConfig, GenerationConfig,
//! KvCache, KvCacheLayer, and CTC decoding.
//!
//! Part of #4186.

use crate::dyn_tensor::DynTensor;
use crate::layers::generation::{
    ctc_greedy_decode, BeamSearchConfig, CtcConfig, GenerationConfig, KvCache, KvCacheLayer,
};
use crate::test_prng::rand_f32_vec;
use crate::Device;

/// Helper: create a DynTensor with deterministic random data.
fn rand_tensor(seed: u64, dims: &[usize]) -> DynTensor {
    let numel: usize = dims.iter().product();
    let data = rand_f32_vec(seed, numel, -1.0, 1.0);
    DynTensor::from_vec(data, dims, &Device::Cpu).unwrap()
}

#[test]
fn test_beam_search_config_default() {
    let config = BeamSearchConfig::default();
    assert_eq!(config.beam_width, 4);
    assert_eq!(config.max_new_tokens, 128);
    assert!((config.length_penalty - 1.0).abs() < 1e-10);
    assert!(!config.early_stopping);
    assert!(config.eos_token_id.is_none());
    config.validate().unwrap();
}

#[test]
fn test_generation_config_default() {
    let config = GenerationConfig::default();
    assert_eq!(config.max_new_tokens, 128);
    assert!((config.temperature - 0.0).abs() < 1e-10);
    assert!(config.top_k.is_none());
    assert!(config.top_p.is_none());
    assert!(config.eos_token_id.is_none());
    assert!(config.seed.is_none());
}

#[test]
fn test_beam_search_single_beam() {
    // beam_size=1 config should validate successfully — equivalent to greedy
    let config = BeamSearchConfig::new(1);
    assert_eq!(config.beam_width, 1);
    config.validate().unwrap();
}

#[test]
fn test_beam_search_config_zero_width_rejected() {
    let config = BeamSearchConfig::new(0);
    assert!(config.validate().is_err());
}

#[test]
fn test_beam_search_config_nan_penalty_rejected() {
    let config = BeamSearchConfig::default().with_length_penalty(f64::NAN);
    assert!(config.validate().is_err());
}

#[test]
fn test_kv_cache_creation() {
    let num_layers = 12;
    let cache = KvCache::new(num_layers);
    assert_eq!(cache.num_layers(), num_layers);

    // Each layer should start empty (seq_len = 0)
    for i in 0..num_layers {
        let layer = cache.layer(i).unwrap();
        assert_eq!(layer.seq_len(), 0);
    }
}

#[test]
fn test_kv_cache_layer_append() {
    let mut layer = KvCacheLayer::empty();
    assert_eq!(layer.seq_len(), 0);

    // Create K/V tensors: [batch=1, num_kv_heads=2, seq=3, head_dim=4]
    let batch = 1;
    let heads = 2;
    let seq = 3;
    let head_dim = 4;

    let new_k = rand_tensor(700, &[batch, heads, seq, head_dim]);
    let new_v = rand_tensor(701, &[batch, heads, seq, head_dim]);

    let (full_k, full_v) = layer.append(&new_k, &new_v).unwrap();
    assert_eq!(layer.seq_len(), seq);
    assert_eq!(full_k.dims(), &[batch, heads, seq, head_dim]);
    assert_eq!(full_v.dims(), &[batch, heads, seq, head_dim]);

    // Append more tokens: [batch=1, heads=2, seq=2, head_dim=4]
    let more_k = rand_tensor(702, &[batch, heads, 2, head_dim]);
    let more_v = rand_tensor(703, &[batch, heads, 2, head_dim]);

    // Drop old views before appending to avoid COW copy
    drop(full_k);
    drop(full_v);

    let (full_k2, full_v2) = layer.append(&more_k, &more_v).unwrap();
    assert_eq!(layer.seq_len(), seq + 2);
    assert_eq!(full_k2.dims(), &[batch, heads, seq + 2, head_dim]);
    assert_eq!(full_v2.dims(), &[batch, heads, seq + 2, head_dim]);
}

#[test]
fn test_ctc_greedy_decode_basic() {
    // Simple logits: 3 time steps, vocab=4, blank=0
    // Step 0: token 1 is highest
    // Step 1: token 2 is highest
    // Step 2: token 1 is highest
    // Expected output: [1, 2, 1] (no repeats to collapse, no blanks)
    let logits_data = vec![
        -1.0f32, 5.0, -1.0, -1.0, // step 0: argmax=1
        -1.0, -1.0, 5.0, -1.0, // step 1: argmax=2
        -1.0, 5.0, -1.0, -1.0, // step 2: argmax=1
    ];
    let logits = DynTensor::from_vec(logits_data, &[3, 4], &Device::Cpu).unwrap();
    let config = CtcConfig::new(0);
    let decoded = ctc_greedy_decode(&logits, &config).unwrap();
    assert_eq!(decoded, vec![1, 2, 1]);
}

#[test]
fn test_ctc_greedy_decode_blank_removal() {
    // 5 time steps, vocab=4, blank=0
    // Step 0: token 1
    // Step 1: token 1 (duplicate)
    // Step 2: blank (0)
    // Step 3: blank (0)
    // Step 4: token 2
    // After collapse: [1, 0, 2]
    // After blank removal: [1, 2]
    let logits_data = vec![
        -1.0f32, 5.0, -1.0, -1.0, // step 0: argmax=1
        -1.0, 5.0, -1.0, -1.0, // step 1: argmax=1 (repeat)
        5.0, -1.0, -1.0, -1.0, // step 2: argmax=0 (blank)
        5.0, -1.0, -1.0, -1.0, // step 3: argmax=0 (blank, repeat)
        -1.0, -1.0, 5.0, -1.0, // step 4: argmax=2
    ];
    let logits = DynTensor::from_vec(logits_data, &[5, 4], &Device::Cpu).unwrap();
    let config = CtcConfig::new(0);
    let decoded = ctc_greedy_decode(&logits, &config).unwrap();
    assert_eq!(decoded, vec![1, 2]);
}
