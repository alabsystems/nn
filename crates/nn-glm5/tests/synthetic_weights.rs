#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Glm5Model with synthetic (non-zero) weights.
//!
//! Constructs a TensorMapBackend with all required GLM-5 weight keys using
//! deterministic pseudo-random values, loads the model via VarBuilder, and
//! runs forward passes. Validates the full weight-loading pipeline end-to-end
//! without requiring real model weights.
//!
//! Mirrors the pattern from `nn-qwen3/tests/synthetic_weights.rs`.
//! Uses `TensorMapBackend` (backend-agnostic) rather than `SafeTensorsBackend`
//! (Metal-specific) since nn-glm5 is hardware-agnostic.

use std::collections::HashMap;
use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::var_builder::{TensorMapBackend, VarBuilder};
use nn_core::DType;
use nn_glm5::test_utils::tiny_config;
use nn_glm5::{Glm5Config, Glm5Model};

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

/// Build self-attention weights for one GLM layer (fused QKV projection).
fn push_self_attn(
    map: &mut HashMap<String, DynTensor>,
    prefix: &str,
    cfg: &Glm5Config,
    rng: &mut Rng,
) {
    let h = cfg.hidden_size;
    let hd = cfg.head_dim();
    let nh = cfg.num_attention_heads;
    let nkv = cfg.multi_query_group_num;

    // Fused QKV: output size = (nh + 2 * nkv) * hd
    let qkv_size = (nh + 2 * nkv) * hd;
    map.insert(
        format!("{prefix}.query_key_value.weight"),
        rng.tensor(&[qkv_size, h]),
    );
    if cfg.add_qkv_bias {
        map.insert(
            format!("{prefix}.query_key_value.bias"),
            rng.tensor(&[qkv_size]),
        );
    }

    // Dense (output projection)
    map.insert(format!("{prefix}.dense.weight"), rng.tensor(&[h, nh * hd]));
    if cfg.add_bias_linear {
        map.insert(format!("{prefix}.dense.bias"), rng.tensor(&[h]));
    }
}

/// Build MLP weights for one GLM layer (SwiGLU: fused gate+up, down).
fn push_mlp(map: &mut HashMap<String, DynTensor>, prefix: &str, cfg: &Glm5Config, rng: &mut Rng) {
    let h = cfg.hidden_size;
    let ffn = cfg.ffn_hidden_size;

    // dense_h_to_4h: [ffn * 2, h] (gate+up fused)
    map.insert(
        format!("{prefix}.dense_h_to_4h.weight"),
        rng.tensor(&[ffn * 2, h]),
    );
    if cfg.add_bias_linear {
        map.insert(
            format!("{prefix}.dense_h_to_4h.bias"),
            rng.tensor(&[ffn * 2]),
        );
    }

    // dense_4h_to_h: [h, ffn]
    map.insert(
        format!("{prefix}.dense_4h_to_h.weight"),
        rng.tensor(&[h, ffn]),
    );
    if cfg.add_bias_linear {
        map.insert(format!("{prefix}.dense_4h_to_h.bias"), rng.tensor(&[h]));
    }
}

/// Build all weights for a GLM-5 model as a TensorMapBackend.
fn build_synthetic_vb(config: &Glm5Config) -> VarBuilder {
    let mut rng = Rng::new(42);
    let h = config.hidden_size;
    let mut map = HashMap::new();

    // Embedding: transformer.embedding.word_embeddings.weight
    map.insert(
        "transformer.embedding.word_embeddings.weight".into(),
        rng.tensor(&[config.padded_vocab_size, h]),
    );

    // Decoder layers: transformer.encoder.layers.{i}.*
    for i in 0..config.num_layers {
        let bp = format!("transformer.encoder.layers.{i}");

        // Layer norms (RMSNorm: weight only)
        ones(&mut map, format!("{bp}.input_layernorm.weight"), &[h]);
        ones(
            &mut map,
            format!("{bp}.post_attention_layernorm.weight"),
            &[h],
        );

        // Self-attention (fused QKV)
        push_self_attn(&mut map, &format!("{bp}.self_attention"), config, &mut rng);

        // MLP (SwiGLU)
        push_mlp(&mut map, &format!("{bp}.mlp"), config, &mut rng);
    }

    // Final layer norm: transformer.encoder.final_layernorm.weight
    ones(
        &mut map,
        "transformer.encoder.final_layernorm.weight".into(),
        &[h],
    );

    // Output layer: transformer.output_layer.weight
    map.insert(
        "transformer.output_layer.weight".into(),
        rng.tensor(&[config.padded_vocab_size, h]),
    );

    // Output layer bias (only when add_bias_linear)
    if config.add_bias_linear {
        map.insert(
            "transformer.output_layer.bias".into(),
            rng.tensor(&[config.padded_vocab_size]),
        );
    }

    VarBuilder::from_backend(Arc::new(TensorMapBackend::new(map)), DType::F32, cpu())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_glm5_config_creation() {
    let cfg = Glm5Config::new(
        64,       // hidden_size
        128,      // ffn_hidden_size
        2,        // num_layers
        4,        // num_attention_heads
        2,        // multi_query_group_num
        100,      // padded_vocab_size
        16,       // kv_channels
        1e-5,     // layernorm_epsilon
        64,       // seq_length
        true,     // rmsnorm
        true,     // add_qkv_bias
        false,    // add_bias_linear
        10_000.0, // rope_theta
    );
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.head_dim(), 16);
    assert_eq!(cfg.num_kv_groups().unwrap(), 2); // 4 heads / 2 kv groups
}

#[test]
fn test_glm5_forward_shape() {
    let config = tiny_config();
    let vb = build_synthetic_vb(&config);
    let model = Glm5Model::load(&vb, config.clone()).unwrap();

    let seq_len = 8;
    let input_ids: Vec<usize> = (0..seq_len).collect();
    let positions: Vec<usize> = (0..seq_len).collect();

    let logits = model.forward(&input_ids, &positions).unwrap();

    assert_eq!(logits.rank(), 3);
    assert_eq!(logits.dim(0).unwrap(), 1);
    assert_eq!(logits.dim(1).unwrap(), seq_len);
    assert_eq!(logits.dim(2).unwrap(), config.padded_vocab_size);

    let flat = logits.to_flat_vec::<f32>().unwrap();
    let non_finite = flat.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(non_finite, 0, "output should have no NaN/Inf");
}

#[test]
fn test_glm5_forward_cached() {
    let config = tiny_config();
    let vb = build_synthetic_vb(&config);
    let model = Glm5Model::load(&vb, config).unwrap();

    // Uncached forward
    let logits_plain = model.forward(&[10], &[0]).unwrap();
    // Cached forward with None cache should produce identical output
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
fn test_glm5_forward_from_embeddings() {
    let config = tiny_config();
    let vb = build_synthetic_vb(&config);
    let model = Glm5Model::load(&vb, config.clone()).unwrap();

    let seq_len = 4;
    // Create synthetic embeddings: [1, seq_len, hidden_size]
    let mut rng = Rng::new(99);
    let hidden_states = rng.tensor(&[1, seq_len, config.hidden_size]);
    let positions: Vec<usize> = (0..seq_len).collect();

    let logits = model
        .forward_from_embeddings(&hidden_states, &positions, None)
        .unwrap();

    assert_eq!(logits.dims(), &[1, seq_len, config.padded_vocab_size]);

    let flat = logits.to_flat_vec::<f32>().unwrap();
    let non_finite = flat.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite, 0,
        "embedding forward should produce finite output"
    );
}

#[test]
fn test_glm5_bf16_forward() {
    let config = tiny_config();
    // Build weights as BF16 by loading F32 then converting via BF16 VarBuilder
    let mut rng = Rng::new(42);
    let h = config.hidden_size;
    let mut map = HashMap::new();

    // Build the same weight map but wrap in BF16 VarBuilder
    map.insert(
        "transformer.embedding.word_embeddings.weight".into(),
        rng.tensor(&[config.padded_vocab_size, h]),
    );
    for i in 0..config.num_layers {
        let bp = format!("transformer.encoder.layers.{i}");
        ones(&mut map, format!("{bp}.input_layernorm.weight"), &[h]);
        ones(
            &mut map,
            format!("{bp}.post_attention_layernorm.weight"),
            &[h],
        );
        push_self_attn(&mut map, &format!("{bp}.self_attention"), &config, &mut rng);
        push_mlp(&mut map, &format!("{bp}.mlp"), &config, &mut rng);
    }
    ones(
        &mut map,
        "transformer.encoder.final_layernorm.weight".into(),
        &[h],
    );
    map.insert(
        "transformer.output_layer.weight".into(),
        rng.tensor(&[config.padded_vocab_size, h]),
    );

    let vb = VarBuilder::from_backend(Arc::new(TensorMapBackend::new(map)), DType::BF16, cpu());
    let model = Glm5Model::load(&vb, config.clone()).unwrap();
    assert_eq!(model.dtype(), DType::BF16);

    let logits = model.forward(&[0, 1, 2, 3], &[0, 1, 2, 3]).unwrap();
    assert_eq!(logits.dims(), &[1, 4, config.padded_vocab_size]);

    let flat = logits.to_flat_vec::<f32>().unwrap();
    let non_finite = flat.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(non_finite, 0, "BF16 forward should produce finite output");
}

#[test]
fn test_glm5_single_layer() {
    // Verify that a model with exactly 1 layer loads and runs.
    let mut config = tiny_config();
    config.num_layers = 1;
    let vb = build_synthetic_vb(&config);
    let model = Glm5Model::load(&vb, config.clone()).unwrap();

    let logits = model.forward(&[0, 1], &[0, 1]).unwrap();
    assert_eq!(logits.dims(), &[1, 2, config.padded_vocab_size]);

    let flat = logits.to_flat_vec::<f32>().unwrap();
    let non_finite = flat.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(non_finite, 0, "single-layer forward should be finite");
}

#[test]
fn test_glm5_attention_mask() {
    let config = tiny_config();
    let vb = build_synthetic_vb(&config);
    let model = Glm5Model::load(&vb, config.clone()).unwrap();

    // Multi-token forward (triggers causal mask path: seq_len > 1)
    let seq_len = 6;
    let input_ids: Vec<usize> = (0..seq_len).collect();
    let positions: Vec<usize> = (0..seq_len).collect();
    let logits_multi = model.forward(&input_ids, &positions).unwrap();
    assert_eq!(logits_multi.dims(), &[1, seq_len, config.padded_vocab_size]);

    // Single-token forward (no mask path: seq_len == 1)
    let logits_single = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits_single.dims(), &[1, 1, config.padded_vocab_size]);

    // Both should be finite
    for logits in [&logits_multi, &logits_single] {
        let flat = logits.to_flat_vec::<f32>().unwrap();
        let non_finite = flat.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(non_finite, 0, "output should be finite");
    }
}

#[test]
fn test_glm5_kv_cache() {
    let config = tiny_config();
    let vb = build_synthetic_vb(&config);
    let model = Glm5Model::load(&vb, config.clone()).unwrap();
    let mut cache = model.new_cache();

    // Step 0: first token
    let logits0 = model.forward_cached(&[0], &[0], Some(&mut cache)).unwrap();
    assert_eq!(logits0.dims(), &[1, 1, config.padded_vocab_size]);

    // Step 1: second token (cache should have grown)
    let logits1 = model.forward_cached(&[1], &[1], Some(&mut cache)).unwrap();
    assert_eq!(logits1.dims(), &[1, 1, config.padded_vocab_size]);

    // Step 2: third token
    let logits2 = model.forward_cached(&[2], &[2], Some(&mut cache)).unwrap();
    assert_eq!(logits2.dims(), &[1, 1, config.padded_vocab_size]);

    // Step 3: fourth token
    let logits3 = model.forward_cached(&[3], &[3], Some(&mut cache)).unwrap();
    assert_eq!(logits3.dims(), &[1, 1, config.padded_vocab_size]);

    // All outputs should be finite
    for (i, logits) in [&logits0, &logits1, &logits2, &logits3].iter().enumerate() {
        let flat = logits.to_flat_vec::<f32>().unwrap();
        let nf = flat.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(nf, 0, "step {i} logits should be finite");
    }

    // Logits at each step should differ (cache accumulation changes attention)
    let v0 = logits0.to_flat_vec::<f32>().unwrap();
    let v1 = logits1.to_flat_vec::<f32>().unwrap();
    let max_diff: f32 = v0
        .iter()
        .zip(v1.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-8,
        "step 0 and step 1 logits should differ (cache accumulation), max_diff={max_diff}"
    );
}

#[test]
fn test_glm5_weight_loading_error() {
    let config = tiny_config();
    // Create a VarBuilder missing the embedding weight
    let mut rng = Rng::new(42);
    let h = config.hidden_size;
    let mut map = HashMap::new();

    // Skip embedding — consume RNG state to keep determinism
    let _ = rng.tensor(&[config.padded_vocab_size, h]);

    // Decoder layers only (no embedding)
    for i in 0..config.num_layers {
        let bp = format!("transformer.encoder.layers.{i}");
        ones(&mut map, format!("{bp}.input_layernorm.weight"), &[h]);
        ones(
            &mut map,
            format!("{bp}.post_attention_layernorm.weight"),
            &[h],
        );
        push_self_attn(&mut map, &format!("{bp}.self_attention"), &config, &mut rng);
        push_mlp(&mut map, &format!("{bp}.mlp"), &config, &mut rng);
    }
    ones(
        &mut map,
        "transformer.encoder.final_layernorm.weight".into(),
        &[h],
    );
    map.insert(
        "transformer.output_layer.weight".into(),
        rng.tensor(&[config.padded_vocab_size, h]),
    );

    let vb_missing =
        VarBuilder::from_backend(Arc::new(TensorMapBackend::new(map)), DType::F32, cpu());

    let result = Glm5Model::load(&vb_missing, config);
    let err = match result {
        Ok(_) => panic!("should fail with missing weight key"),
        Err(e) => e,
    };
    let err_msg = format!("{err}");
    assert!(
        err_msg.contains("word_embeddings")
            || err_msg.contains("not found")
            || err_msg.contains("TensorNotFound"),
        "error should reference the missing key, got: {err_msg}"
    );
}

#[test]
fn test_glm5_config_validation() {
    // Zero attention heads
    let bad_heads = Glm5Config::new(
        64, 128, 2, 0, 2, 100, 16, 1e-5, 64, true, true, false, 10_000.0,
    );
    assert!(bad_heads.validate().is_err());

    // Zero multi_query_group_num
    let bad_kv = Glm5Config::new(
        64, 128, 2, 4, 0, 100, 16, 1e-5, 64, true, true, false, 10_000.0,
    );
    assert!(bad_kv.validate().is_err());

    // num_attention_heads not divisible by multi_query_group_num
    let bad_divisibility = Glm5Config::new(
        64, 128, 2, 4, 3, 100, 16, 1e-5, 64, true, true, false, 10_000.0,
    );
    assert!(bad_divisibility.validate().is_err());

    // Zero hidden_size
    let bad_hidden = Glm5Config::new(
        0, 128, 2, 4, 2, 100, 16, 1e-5, 64, true, true, false, 10_000.0,
    );
    assert!(bad_hidden.validate().is_err());

    // Zero ffn_hidden_size
    let bad_ffn = Glm5Config::new(
        64, 0, 2, 4, 2, 100, 16, 1e-5, 64, true, true, false, 10_000.0,
    );
    assert!(bad_ffn.validate().is_err());

    // kv_channels not multiple of 4
    let bad_kv_channels = Glm5Config::new(
        64, 128, 2, 4, 2, 100, 5, 1e-5, 64, true, true, false, 10_000.0,
    );
    assert!(bad_kv_channels.validate().is_err());

    // NaN layernorm_epsilon
    let bad_eps = Glm5Config::new(
        64,
        128,
        2,
        4,
        2,
        100,
        16,
        f64::NAN,
        64,
        true,
        true,
        false,
        10_000.0,
    );
    assert!(bad_eps.validate().is_err());

    // Negative rope_theta
    let bad_theta = Glm5Config::new(64, 128, 2, 4, 2, 100, 16, 1e-5, 64, true, true, false, -1.0);
    assert!(bad_theta.validate().is_err());

    // Zero seq_length
    let bad_seq = Glm5Config::new(
        64, 128, 2, 4, 2, 100, 16, 1e-5, 0, true, true, false, 10_000.0,
    );
    assert!(bad_seq.validate().is_err());

    // Zero padded_vocab_size
    let bad_vocab = Glm5Config::new(
        64, 128, 2, 4, 2, 0, 16, 1e-5, 64, true, true, false, 10_000.0,
    );
    assert!(bad_vocab.validate().is_err());

    // Zero num_layers
    let bad_layers = Glm5Config::new(
        64, 128, 0, 4, 2, 100, 16, 1e-5, 64, true, true, false, 10_000.0,
    );
    assert!(bad_layers.validate().is_err());
}
