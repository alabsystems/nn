#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for autoregressive generation and KV cache integration with attention.

use crate::dyn_tensor::DynTensor;
use crate::layers::autoregressive::{generate, GenerationConfig};
use crate::layers::kv_cache::{KvCache, KvCacheLayer};
use crate::layers::{Linear, Module};
use crate::{DType, Device};

// ---------------------------------------------------------------------------
// Autoregressive generation tests
// ---------------------------------------------------------------------------

/// A minimal "model" that returns logits where token ID = position index.
/// This tests the generation loop mechanics without real transformer weights.
fn dummy_model_fn(input: &DynTensor, cache: &mut KvCache) -> crate::Result<DynTensor> {
    let seq_len = input.dim(1)?;
    let cache_len = cache.seq_len();

    // Fake KV cache: append dummy tensors to layer 0
    let k = DynTensor::ones(&[1, 1, seq_len, 4], DType::F32, &Device::Cpu)?;
    let v = DynTensor::ones(&[1, 1, seq_len, 4], DType::F32, &Device::Cpu)?;
    cache.layer_mut(0)?.append(&k, &v)?;

    // Return logits: vocab size 10, argmax at position (cache_len % 10)
    let vocab_size = 10;
    let mut logits_data = vec![0.0f32; vocab_size];
    let predicted_token = (cache_len + seq_len) % vocab_size;
    logits_data[predicted_token] = 10.0;
    DynTensor::from_vec(logits_data, &[1, vocab_size], &Device::Cpu)
}

#[test]
fn test_generate_greedy() {
    let mut cache = KvCache::new(1);
    let config = GenerationConfig {
        max_new_tokens: 5,
        temperature: 0.0,
        top_k: None,
        top_p: None,
        eos_token_id: None,
        seed: None,
    };

    let output = generate(dummy_model_fn, &[0], &mut cache, &config, &Device::Cpu).unwrap();
    assert_eq!(output.token_ids.len(), 5);
    assert!(!output.finished);
}

#[test]
fn test_generate_stops_at_eos() {
    let mut cache = KvCache::new(1);

    let config = GenerationConfig {
        max_new_tokens: 100,
        temperature: 0.0,
        top_k: None,
        top_p: None,
        eos_token_id: Some(3),
        seed: None,
    };

    let output = generate(dummy_model_fn, &[0], &mut cache, &config, &Device::Cpu).unwrap();
    assert!(output.finished);
    assert!(*output.token_ids.last().unwrap() == 3);
}

#[test]
fn test_generate_empty_prompt_rejected() {
    let mut cache = KvCache::new(1);
    let config = GenerationConfig::default();
    let result = generate(dummy_model_fn, &[], &mut cache, &config, &Device::Cpu);
    assert!(result.is_err());
}

#[test]
fn test_generate_zero_max_tokens() {
    let mut cache = KvCache::new(1);
    let config = GenerationConfig {
        max_new_tokens: 0,
        ..Default::default()
    };
    let output = generate(dummy_model_fn, &[0], &mut cache, &config, &Device::Cpu).unwrap();
    assert!(output.token_ids.is_empty());
    assert!(!output.finished);
}

#[test]
fn test_generate_with_top_k() {
    let mut cache = KvCache::new(1);
    let config = GenerationConfig {
        max_new_tokens: 3,
        temperature: 1.0,
        top_k: Some(3),
        top_p: None,
        eos_token_id: None,
        seed: None,
    };

    let output = generate(dummy_model_fn, &[0], &mut cache, &config, &Device::Cpu).unwrap();
    assert_eq!(output.token_ids.len(), 3);
}

// ---------------------------------------------------------------------------
// Integration test: dummy attention model with KV cache
// ---------------------------------------------------------------------------

/// A toy single-head attention layer using DynTensor ops + KV cache.
struct ToyAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    head_dim: usize,
}

impl ToyAttention {
    fn new(dim: usize) -> crate::Result<Self> {
        let wq = Linear::new(
            DynTensor::ones(&[dim, dim], DType::F32, &Device::Cpu)?,
            None,
        )?;
        let wk = Linear::new(
            DynTensor::ones(&[dim, dim], DType::F32, &Device::Cpu)?,
            None,
        )?;
        let wv = Linear::new(
            DynTensor::ones(&[dim, dim], DType::F32, &Device::Cpu)?,
            None,
        )?;
        let wo = Linear::new(
            DynTensor::ones(&[dim, dim], DType::F32, &Device::Cpu)?,
            None,
        )?;
        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            head_dim: dim,
        })
    }

    /// Forward with KV cache.
    ///
    /// x: [batch, seq_len, dim]
    /// Returns: [batch, seq_len, dim]
    fn forward(&self, x: &DynTensor, cache: &mut KvCacheLayer) -> crate::Result<DynTensor> {
        let (batch, seq_len, dim) = x.dims3()?;

        let x_flat = x.reshape([batch * seq_len, dim])?;

        let q = self.wq.forward(&x_flat)?.reshape([batch, seq_len, dim])?;
        let k = self.wk.forward(&x_flat)?.reshape([batch, seq_len, dim])?;
        let v = self.wv.forward(&x_flat)?.reshape([batch, seq_len, dim])?;

        let k_4d = k.reshape([batch, 1, seq_len, self.head_dim])?;
        let v_4d = v.reshape([batch, 1, seq_len, self.head_dim])?;

        let (full_k, full_v) = cache.append(&k_4d, &v_4d)?;
        let kv_len = full_k.dim(2)?;

        let q_3d = q.reshape([batch, seq_len, self.head_dim])?;
        let k_3d = full_k.reshape([batch, kv_len, self.head_dim])?;
        let v_3d = full_v.reshape([batch, kv_len, self.head_dim])?;

        let k_t = k_3d.transpose(1, 2)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_scores = q_3d.matmul(&k_t)?.mul_scalar(scale)?;
        let attn_weights = crate::layers::softmax(&attn_scores, 2)?;
        let attn_out = attn_weights.matmul(&v_3d)?;

        let out_flat = attn_out.reshape([batch * seq_len, dim])?;
        #[allow(clippy::tuple_array_conversions)]
        let out = self.wo.forward(&out_flat)?.reshape([batch, seq_len, dim])?;
        Ok(out)
    }
}

#[test]
fn test_toy_attention_with_kv_cache() {
    let dim = 8;
    let attn = ToyAttention::new(dim).unwrap();
    let mut cache = KvCacheLayer::empty();

    let x = DynTensor::ones(&[1, 4, dim], DType::F32, &Device::Cpu).unwrap();
    let out1 = attn.forward(&x, &mut cache).unwrap();
    assert_eq!(out1.dims(), &[1, 4, dim]);
    assert_eq!(cache.seq_len(), 4);

    let x2 = DynTensor::ones(&[1, 1, dim], DType::F32, &Device::Cpu).unwrap();
    let out2 = attn.forward(&x2, &mut cache).unwrap();
    assert_eq!(out2.dims(), &[1, 1, dim]);
    assert_eq!(cache.seq_len(), 5);

    let x3 = DynTensor::ones(&[1, 1, dim], DType::F32, &Device::Cpu).unwrap();
    let out3 = attn.forward(&x3, &mut cache).unwrap();
    assert_eq!(out3.dims(), &[1, 1, dim]);
    assert_eq!(cache.seq_len(), 6);

    let data = out3.to_flat_vec::<f32>().unwrap();
    for v in &data {
        assert!(v.is_finite(), "non-finite output: {v}");
    }
}

#[test]
fn test_toy_attention_cache_reset_and_rerun() {
    let dim = 4;
    let attn = ToyAttention::new(dim).unwrap();
    let mut cache = KvCacheLayer::empty();

    let x = DynTensor::ones(&[1, 3, dim], DType::F32, &Device::Cpu).unwrap();
    attn.forward(&x, &mut cache).unwrap();
    assert_eq!(cache.seq_len(), 3);

    cache.reset();
    assert_eq!(cache.seq_len(), 0);

    let out = attn.forward(&x, &mut cache).unwrap();
    assert_eq!(out.dims(), &[1, 3, dim]);
    assert_eq!(cache.seq_len(), 3);
}

// ---------------------------------------------------------------------------
// Categorical sampling tests (require `rand` feature)
// ---------------------------------------------------------------------------

/// A model that returns competitive logits across multiple tokens,
/// suitable for testing that categorical sampling produces diversity.
#[cfg(feature = "rand")]
fn competitive_model_fn(input: &DynTensor, cache: &mut KvCache) -> crate::Result<DynTensor> {
    let seq_len = input.dim(1)?;
    let k = DynTensor::ones(&[1, 1, seq_len, 4], DType::F32, &Device::Cpu)?;
    let v = DynTensor::ones(&[1, 1, seq_len, 4], DType::F32, &Device::Cpu)?;
    cache.layer_mut(0)?.append(&k, &v)?;

    // Logits: tokens 0-4 have similar values (1.0-2.0), rest are low.
    // With temperature=1.0, sampling should pick from tokens 0-4 with
    // varying probability, demonstrating real sampling behavior.
    let logits_data = vec![2.0, 1.8, 1.6, 1.4, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0f32];
    DynTensor::from_vec(logits_data, &[1, 10], &Device::Cpu)
}

#[cfg(feature = "rand")]
#[test]
fn test_seeded_sampling_reproducible() {
    // Same seed should produce identical sequences
    let config = GenerationConfig {
        max_new_tokens: 10,
        temperature: 1.0,
        top_k: Some(5),
        top_p: None,
        eos_token_id: None,
        seed: Some(42),
    };

    let mut cache1 = KvCache::new(1);
    let out1 = generate(
        competitive_model_fn,
        &[0],
        &mut cache1,
        &config,
        &Device::Cpu,
    )
    .unwrap();

    let mut cache2 = KvCache::new(1);
    let out2 = generate(
        competitive_model_fn,
        &[0],
        &mut cache2,
        &config,
        &Device::Cpu,
    )
    .unwrap();

    assert_eq!(
        out1.token_ids, out2.token_ids,
        "same seed must produce identical sequences"
    );
}

#[cfg(feature = "rand")]
#[test]
fn test_different_seeds_produce_different_output() {
    // Different seeds should (very likely) produce different sequences.
    // With 5 competitive candidates over 20 tokens, collision probability is negligible.
    let config_a = GenerationConfig {
        max_new_tokens: 20,
        temperature: 1.0,
        top_k: Some(5),
        top_p: None,
        eos_token_id: None,
        seed: Some(42),
    };
    let config_b = GenerationConfig {
        max_new_tokens: 20,
        temperature: 1.0,
        top_k: Some(5),
        top_p: None,
        eos_token_id: None,
        seed: Some(999),
    };

    let mut cache_a = KvCache::new(1);
    let out_a = generate(
        competitive_model_fn,
        &[0],
        &mut cache_a,
        &config_a,
        &Device::Cpu,
    )
    .unwrap();

    let mut cache_b = KvCache::new(1);
    let out_b = generate(
        competitive_model_fn,
        &[0],
        &mut cache_b,
        &config_b,
        &Device::Cpu,
    )
    .unwrap();

    assert_ne!(
        out_a.token_ids, out_b.token_ids,
        "different seeds should produce different sequences"
    );
}

#[cfg(feature = "rand")]
#[test]
fn test_temperature_zero_ignores_seed() {
    // temperature=0 should give argmax regardless of seed
    let config = GenerationConfig {
        max_new_tokens: 5,
        temperature: 0.0,
        top_k: None,
        top_p: None,
        eos_token_id: None,
        seed: Some(42),
    };

    let mut cache = KvCache::new(1);
    let out = generate(
        competitive_model_fn,
        &[0],
        &mut cache,
        &config,
        &Device::Cpu,
    )
    .unwrap();

    // With competitive_model_fn, token 0 always has the highest logit (2.0),
    // so greedy decoding should always pick token 0
    for &tok in &out.token_ids {
        assert_eq!(tok, 0, "temperature=0 should always pick argmax (token 0)");
    }
}

#[cfg(feature = "rand")]
#[test]
fn test_sampling_produces_diversity() {
    // With high temperature and a seed, we should see multiple different tokens.
    let config = GenerationConfig {
        max_new_tokens: 50,
        temperature: 2.0,
        top_k: Some(5),
        top_p: None,
        eos_token_id: None,
        seed: Some(123),
    };

    let mut cache = KvCache::new(1);
    let out = generate(
        competitive_model_fn,
        &[0],
        &mut cache,
        &config,
        &Device::Cpu,
    )
    .unwrap();

    let unique_tokens: std::collections::HashSet<usize> = out.token_ids.iter().copied().collect();
    assert!(
        unique_tokens.len() > 1,
        "sampling with temperature=2.0 should produce multiple distinct tokens, got {unique_tokens:?}"
    );
}
