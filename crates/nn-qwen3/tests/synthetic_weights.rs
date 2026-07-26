#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Qwen3Model with synthetic (non-zero) weights.
//!
//! Constructs a TensorMapBackend with all required Qwen3 weight keys using
//! deterministic pseudo-random values, loads the model via VarBuilder, and
//! runs forward passes. Validates the full weight-loading pipeline end-to-end
//! without requiring real model weights.
//!
//! Mirrors the pattern from `nn-whisper/tests/safetensors_load.rs`.
//! Uses `TensorMapBackend` (backend-agnostic) rather than `SafeTensorsBackend`
//! (Metal-specific) since nn-qwen3 is hardware-agnostic.

use std::collections::HashMap;
use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::var_builder::{TensorMapBackend, VarBuilder};
use nn_core::DType;
use nn_qwen3::test_utils::tiny_config;
use nn_qwen3::{Qwen3Config, Qwen3Model};

// ---------------------------------------------------------------------------
// Synthetic weight builders
// ---------------------------------------------------------------------------

/// Deterministic xorshift64 pseudo-random f32 generator.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 as f64 / u64::MAX as f64) * 0.02 - 0.01) as f32
    }

    fn tensor(&mut self, shape: &[usize]) -> DynTensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| self.next_f32()).collect();
        DynTensor::from_vec(data, shape, &cpu()).unwrap()
    }
}

/// Insert a tensor of ones (used for norm weights).
fn ones(map: &mut HashMap<String, DynTensor>, name: String, shape: &[usize]) {
    map.insert(name, DynTensor::ones(shape, DType::F32, &cpu()).unwrap());
}

/// Build self-attention projection weights for one layer.
fn push_self_attn(
    map: &mut HashMap<String, DynTensor>,
    prefix: &str,
    cfg: &Qwen3Config,
    rng: &mut Rng,
) {
    let h = cfg.hidden_size;
    let hd = cfg.head_dim();
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;

    map.insert(format!("{prefix}.q_proj.weight"), rng.tensor(&[nh * hd, h]));
    map.insert(
        format!("{prefix}.k_proj.weight"),
        rng.tensor(&[nkv * hd, h]),
    );
    map.insert(
        format!("{prefix}.v_proj.weight"),
        rng.tensor(&[nkv * hd, h]),
    );
    map.insert(format!("{prefix}.o_proj.weight"), rng.tensor(&[h, nh * hd]));
    // QK-Norm
    ones(map, format!("{prefix}.q_norm.weight"), &[hd]);
    ones(map, format!("{prefix}.k_norm.weight"), &[hd]);
}

/// Build MLP weights for one layer (SwiGLU: gate_proj, up_proj, down_proj).
fn push_mlp(map: &mut HashMap<String, DynTensor>, prefix: &str, cfg: &Qwen3Config, rng: &mut Rng) {
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    map.insert(format!("{prefix}.gate_proj.weight"), rng.tensor(&[i, h]));
    map.insert(format!("{prefix}.up_proj.weight"), rng.tensor(&[i, h]));
    map.insert(format!("{prefix}.down_proj.weight"), rng.tensor(&[h, i]));
}

/// Build all weights for a Qwen3 model as a TensorMapBackend.
fn build_synthetic_vb(config: &Qwen3Config) -> VarBuilder {
    let mut rng = Rng::new(42);
    let h = config.hidden_size;
    let mut map = HashMap::new();

    // Embedding
    map.insert(
        "model.embed_tokens.weight".into(),
        rng.tensor(&[config.vocab_size, h]),
    );

    // Decoder layers
    for i in 0..config.num_hidden_layers {
        let bp = format!("model.layers.{i}");
        // Layer norms (RMSNorm: weight only, no bias)
        ones(&mut map, format!("{bp}.input_layernorm.weight"), &[h]);
        ones(
            &mut map,
            format!("{bp}.post_attention_layernorm.weight"),
            &[h],
        );
        // Self-attention
        push_self_attn(&mut map, &format!("{bp}.self_attn"), config, &mut rng);
        // MLP
        push_mlp(&mut map, &format!("{bp}.mlp"), config, &mut rng);
    }

    // Final RMSNorm
    ones(&mut map, "model.norm.weight".into(), &[h]);

    // lm_head (only when untied)
    if !config.tie_word_embeddings {
        map.insert("lm_head.weight".into(), rng.tensor(&[config.vocab_size, h]));
    }

    VarBuilder::from_backend(Arc::new(TensorMapBackend::new(map)), DType::F32, cpu())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_load_synthetic_weights_full_model() {
    let config = tiny_config();
    let vb = build_synthetic_vb(&config);
    let result = Qwen3Model::load(&vb, config);
    assert!(
        result.is_ok(),
        "model load should succeed: {:?}",
        result.err()
    );
}

#[test]
fn test_forward_with_synthetic_weights() {
    let config = tiny_config();
    let vb = build_synthetic_vb(&config);
    let model = Qwen3Model::load(&vb, config.clone()).unwrap();

    let logits = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();

    assert_eq!(logits.rank(), 3);
    assert_eq!(logits.dim(0).unwrap(), 1);
    assert_eq!(logits.dim(1).unwrap(), 3);
    assert_eq!(logits.dim(2).unwrap(), config.vocab_size);

    let flat = logits.to_flat_vec::<f32>().unwrap();
    let non_finite = flat.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(non_finite, 0, "output should have no NaN/Inf");
}

#[test]
fn test_forward_single_token() {
    let config = tiny_config();
    let vb = build_synthetic_vb(&config);
    let model = Qwen3Model::load(&vb, config.clone()).unwrap();

    let logits = model.forward(&[42], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, config.vocab_size]);
}

#[test]
fn test_forward_cached_matches_uncached() {
    let config = tiny_config();
    let vb = build_synthetic_vb(&config);
    let model = Qwen3Model::load(&vb, config).unwrap();

    // Uncached forward
    let logits_plain = model.forward(&[10], &[0]).unwrap();
    // Cached forward with None cache should match
    let logits_cached = model.forward_cached(&[10], &[0], None).unwrap();

    let plain = logits_plain.to_flat_vec::<f32>().unwrap();
    let cached = logits_cached.to_flat_vec::<f32>().unwrap();
    assert_eq!(plain.len(), cached.len());
    for (a, b) in plain.iter().zip(cached.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "uncached and cached(None) should match: {a} vs {b}"
        );
    }
}

#[test]
fn test_autoregressive_decode_with_cache() {
    let config = tiny_config();
    let vb = build_synthetic_vb(&config);
    let model = Qwen3Model::load(&vb, config.clone()).unwrap();
    let mut cache = model.new_cache();

    // Step 0: first token
    let logits0 = model.forward_cached(&[0], &[0], Some(&mut cache)).unwrap();
    assert_eq!(logits0.dims(), &[1, 1, config.vocab_size]);

    // Step 1: second token
    let logits1 = model.forward_cached(&[1], &[1], Some(&mut cache)).unwrap();
    assert_eq!(logits1.dims(), &[1, 1, config.vocab_size]);

    // Step 2: third token
    let logits2 = model.forward_cached(&[2], &[2], Some(&mut cache)).unwrap();
    assert_eq!(logits2.dims(), &[1, 1, config.vocab_size]);

    // All outputs should be finite
    for (i, logits) in [&logits0, &logits1, &logits2].iter().enumerate() {
        let flat = logits.to_flat_vec::<f32>().unwrap();
        let nf = flat.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(nf, 0, "step {i} logits should be finite");
    }
}

#[test]
fn test_untied_embeddings_forward() {
    let mut config = tiny_config();
    config.tie_word_embeddings = false;
    let vb = build_synthetic_vb(&config);
    let model = Qwen3Model::load(&vb, config.clone()).unwrap();

    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, config.vocab_size]);
}

#[test]
fn test_missing_weight_key_returns_error() {
    let config = tiny_config();
    // Create a VarBuilder without the embed_tokens weight
    let mut rng = Rng::new(42);
    let h = config.hidden_size;
    let mut map = HashMap::new();

    // Skip embedding — this should cause a load error
    let _ = rng.tensor(&[config.vocab_size, h]); // consume RNG state

    // Decoder layers only
    for i in 0..config.num_hidden_layers {
        let bp = format!("model.layers.{i}");
        ones(&mut map, format!("{bp}.input_layernorm.weight"), &[h]);
        ones(
            &mut map,
            format!("{bp}.post_attention_layernorm.weight"),
            &[h],
        );
        push_self_attn(&mut map, &format!("{bp}.self_attn"), &config, &mut rng);
        push_mlp(&mut map, &format!("{bp}.mlp"), &config, &mut rng);
    }
    ones(&mut map, "model.norm.weight".into(), &[h]);

    let vb_missing =
        VarBuilder::from_backend(Arc::new(TensorMapBackend::new(map)), DType::F32, cpu());

    let result = Qwen3Model::load(&vb_missing, config);
    let err = match result {
        Ok(_) => panic!("should fail with missing weight key"),
        Err(e) => e,
    };
    let err_msg = format!("{err}");
    assert!(
        err_msg.contains("embed_tokens")
            || err_msg.contains("not found")
            || err_msg.contains("TensorNotFound"),
        "error should reference the missing key, got: {err_msg}"
    );
}
