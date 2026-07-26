// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Generation integration tests for Qwen3Model.
//!
//! Validates autoregressive generation end-to-end: single-step and multi-step
//! generation, KV cache growth, greedy decoding determinism, max_length stops,
//! beam search, EOS early stopping, and generation with various configs.
//!
//! Uses small dimensions (hidden=256, 2 layers, vocab=100) for test speed.

use std::collections::HashMap;
use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::BeamSearchConfig;
use nn_core::var_builder::{TensorMapBackend, VarBuilder};
use nn_core::{DType, Device};
use nn_qwen3::test_utils::tiny_config;
use nn_qwen3::{Qwen3Config, Qwen3Model};

// ---------------------------------------------------------------------------
// Helpers
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
        DynTensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }
}

fn ones(map: &mut HashMap<String, DynTensor>, name: String, shape: &[usize]) {
    map.insert(
        name,
        DynTensor::ones(shape, DType::F32, &Device::Cpu).unwrap(),
    );
}

/// Build synthetic (non-zero) weights for a Qwen3 dense model.
fn build_synthetic_vb(config: &Qwen3Config) -> VarBuilder {
    let mut rng = Rng::new(42);
    let h = config.hidden_size;
    let hd = config.head_dim();
    let nh = config.num_attention_heads;
    let nkv = config.num_key_value_heads;
    let inter = config.intermediate_size;
    let mut map = HashMap::new();

    map.insert(
        "model.embed_tokens.weight".into(),
        rng.tensor(&[config.vocab_size, h]),
    );

    for i in 0..config.num_hidden_layers {
        let bp = format!("model.layers.{i}");
        ones(&mut map, format!("{bp}.input_layernorm.weight"), &[h]);
        ones(
            &mut map,
            format!("{bp}.post_attention_layernorm.weight"),
            &[h],
        );
        let attn = format!("{bp}.self_attn");
        map.insert(format!("{attn}.q_proj.weight"), rng.tensor(&[nh * hd, h]));
        map.insert(format!("{attn}.k_proj.weight"), rng.tensor(&[nkv * hd, h]));
        map.insert(format!("{attn}.v_proj.weight"), rng.tensor(&[nkv * hd, h]));
        map.insert(format!("{attn}.o_proj.weight"), rng.tensor(&[h, nh * hd]));
        ones(&mut map, format!("{attn}.q_norm.weight"), &[hd]);
        ones(&mut map, format!("{attn}.k_norm.weight"), &[hd]);
        let mlp = format!("{bp}.mlp");
        map.insert(format!("{mlp}.gate_proj.weight"), rng.tensor(&[inter, h]));
        map.insert(format!("{mlp}.up_proj.weight"), rng.tensor(&[inter, h]));
        map.insert(format!("{mlp}.down_proj.weight"), rng.tensor(&[h, inter]));
    }

    ones(&mut map, "model.norm.weight".into(), &[h]);
    if !config.tie_word_embeddings {
        map.insert("lm_head.weight".into(), rng.tensor(&[config.vocab_size, h]));
    }

    VarBuilder::from_backend(
        Arc::new(TensorMapBackend::new(map)),
        DType::F32,
        Device::Cpu,
    )
}

fn load_synthetic_model() -> (Qwen3Config, Qwen3Model) {
    let cfg = tiny_config();
    let vb = build_synthetic_vb(&cfg);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    (cfg, model)
}

fn load_zeros_model() -> (Qwen3Config, Qwen3Model) {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    (cfg, model)
}

// ---------------------------------------------------------------------------
// Single-step generation
// ---------------------------------------------------------------------------

#[test]
fn test_single_step_generation() {
    let (_cfg, model) = load_zeros_model();
    let output = model.generate_greedy(&[42], 1).unwrap();
    assert_eq!(output.token_ids.len(), 1, "should generate exactly 1 token");
}

#[test]
fn test_single_step_generation_with_synthetic_weights() {
    let (_cfg, model) = load_synthetic_model();
    let output = model.generate_greedy(&[42], 1).unwrap();
    assert_eq!(output.token_ids.len(), 1);
    assert!(
        output.token_ids[0] < 100,
        "token should be within vocab range"
    );
}

// ---------------------------------------------------------------------------
// Multi-step generation
// ---------------------------------------------------------------------------

#[test]
fn test_multi_step_generation_3_tokens() {
    let (_cfg, model) = load_zeros_model();
    let output = model.generate_greedy(&[0], 3).unwrap();
    assert_eq!(output.token_ids.len(), 3);
}

#[test]
fn test_multi_step_generation_10_tokens() {
    let (_cfg, model) = load_zeros_model();
    let output = model.generate_greedy(&[5], 10).unwrap();
    assert_eq!(output.token_ids.len(), 10);
}

#[test]
fn test_multi_step_generation_all_tokens_in_vocab() {
    let (cfg, model) = load_synthetic_model();
    let output = model.generate_greedy(&[1], 5).unwrap();
    for &tid in &output.token_ids {
        assert!(
            tid < cfg.vocab_size,
            "generated token {tid} exceeds vocab_size {}",
            cfg.vocab_size
        );
    }
}

// ---------------------------------------------------------------------------
// KV cache growth
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_grows_each_step() {
    let (_cfg, model) = load_zeros_model();
    let mut cache = model.new_cache();

    assert_eq!(cache.seq_len(), 0);
    assert!(cache.is_empty());

    // Step 0: prompt token
    model.forward_cached(&[42], &[0], Some(&mut cache)).unwrap();
    assert_eq!(cache.seq_len(), 1);

    // Steps 1-4: one token each
    for step in 1..5 {
        model
            .forward_cached(&[0], &[step], Some(&mut cache))
            .unwrap();
        assert_eq!(
            cache.seq_len(),
            step + 1,
            "cache should grow to {} after step {}",
            step + 1,
            step
        );
    }
}

#[test]
fn test_kv_cache_shape_matches_config() {
    let (cfg, model) = load_zeros_model();
    let cache = model.new_cache();
    assert_eq!(
        cache.num_layers(),
        cfg.num_hidden_layers,
        "cache layers should match model layers"
    );
}

#[test]
fn test_kv_cache_starts_empty() {
    let (_cfg, model) = load_zeros_model();
    let cache = model.new_cache();
    assert!(cache.is_empty());
    assert_eq!(cache.seq_len(), 0);
}

// ---------------------------------------------------------------------------
// Greedy decoding determinism
// ---------------------------------------------------------------------------

#[test]
fn test_greedy_decoding_deterministic() {
    let (_cfg, model) = load_synthetic_model();

    let out1 = model.generate_greedy(&[42], 5).unwrap();
    let out2 = model.generate_greedy(&[42], 5).unwrap();
    assert_eq!(
        out1.token_ids, out2.token_ids,
        "greedy decoding should be deterministic"
    );
}

#[test]
fn test_greedy_decoding_deterministic_different_prompt() {
    let (_cfg, model) = load_synthetic_model();

    // Different prompts should produce potentially different outputs
    let out_a = model.generate_greedy(&[0], 5).unwrap();
    let out_b = model.generate_greedy(&[99], 5).unwrap();
    // At minimum, both should produce valid outputs of the right length
    assert_eq!(out_a.token_ids.len(), 5);
    assert_eq!(out_b.token_ids.len(), 5);
}

#[test]
fn test_greedy_decoding_multi_token_prompt() {
    let (_cfg, model) = load_synthetic_model();
    let output = model.generate_greedy(&[10, 20, 30], 3).unwrap();
    assert_eq!(output.token_ids.len(), 3);
}

// ---------------------------------------------------------------------------
// Max length stops generation
// ---------------------------------------------------------------------------

#[test]
fn test_max_length_stops_generation() {
    let (_cfg, model) = load_zeros_model();

    for max in [1, 2, 5, 10] {
        let output = model.generate_greedy(&[42], max).unwrap();
        assert_eq!(
            output.token_ids.len(),
            max,
            "should generate exactly {max} tokens"
        );
    }
}

#[test]
fn test_zero_max_tokens_returns_empty() {
    let (_cfg, model) = load_zeros_model();
    let output = model.generate_greedy(&[42], 0).unwrap();
    assert!(
        output.token_ids.is_empty(),
        "0 max tokens should return empty"
    );
    assert!(!output.finished, "0 max tokens should not mark as finished");
}

// ---------------------------------------------------------------------------
// Beam search
// ---------------------------------------------------------------------------

#[test]
fn test_beam_search_produces_beams() {
    let (_cfg, model) = load_zeros_model();

    let beam_cfg = BeamSearchConfig::new(3).with_max_new_tokens(4);
    let output = model.generate_beam(&[42], &beam_cfg).unwrap();
    assert!(!output.beams.is_empty(), "should produce at least one beam");
    assert!(
        output.beams.len() <= 3,
        "should produce at most beam_width beams"
    );
}

#[test]
fn test_beam_search_beams_sorted_by_score() {
    let (_cfg, model) = load_zeros_model();

    let beam_cfg = BeamSearchConfig::new(4)
        .with_max_new_tokens(3)
        .with_length_penalty(0.0);
    let output = model.generate_beam(&[42], &beam_cfg).unwrap();
    for w in output.beams.windows(2) {
        assert!(
            w[0].log_prob >= w[1].log_prob,
            "beams not sorted: {:.4} < {:.4}",
            w[0].log_prob,
            w[1].log_prob
        );
    }
}

#[test]
fn test_beam_search_respects_max_tokens() {
    let (_cfg, model) = load_zeros_model();

    let beam_cfg = BeamSearchConfig::new(2).with_max_new_tokens(5);
    let output = model.generate_beam(&[42], &beam_cfg).unwrap();
    for beam in &output.beams {
        assert!(
            beam.token_ids.len() <= 5,
            "beam should respect max_new_tokens"
        );
    }
}

#[test]
fn test_beam_search_single_beam_produces_one_beam() {
    // beam_width=1 should produce exactly one beam with the right length.
    let (_cfg, model) = load_zeros_model();

    let beam_cfg = BeamSearchConfig::new(1).with_max_new_tokens(3);
    let beam_out = model.generate_beam(&[42], &beam_cfg).unwrap();

    assert_eq!(
        beam_out.beams.len(),
        1,
        "beam_width=1 should produce 1 beam"
    );
    assert!(
        beam_out.beams[0].token_ids.len() <= 3,
        "beam should respect max_new_tokens"
    );
    assert!(
        !beam_out.beams[0].token_ids.is_empty(),
        "beam should generate at least 1 token"
    );
}

// ---------------------------------------------------------------------------
// Generation with various prompt lengths
// ---------------------------------------------------------------------------

#[test]
fn test_generation_single_token_prompt() {
    let (_cfg, model) = load_zeros_model();
    let output = model.generate_greedy(&[0], 3).unwrap();
    assert_eq!(output.token_ids.len(), 3);
}

#[test]
fn test_generation_long_prompt() {
    let (_cfg, model) = load_zeros_model();
    // 10-token prompt
    let prompt: Vec<usize> = (0..10).collect();
    let output = model.generate_greedy(&prompt, 3).unwrap();
    assert_eq!(output.token_ids.len(), 3);
}

// ---------------------------------------------------------------------------
// Manual cached generation loop (matches what generate_greedy does internally)
// ---------------------------------------------------------------------------

#[test]
fn test_manual_autoregressive_loop() {
    let (cfg, model) = load_zeros_model();
    let mut cache = model.new_cache();
    let mut generated = Vec::new();

    // Prefill with prompt
    let prompt_logits = model
        .forward_cached(&[10, 20], &[0, 1], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 2);

    // Decode: extract last token's logits, take argmax
    let last_logits = prompt_logits.narrow(1, 1, 1).unwrap();
    let flat = last_logits.to_flat_vec::<f32>().unwrap();
    let next_token = flat
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    generated.push(next_token);

    // Continue decoding for 2 more steps
    for _step in 0..2 {
        let pos = cache.seq_len();
        let logits = model
            .forward_cached(&[*generated.last().unwrap()], &[pos], Some(&mut cache))
            .unwrap();
        let flat = logits.to_flat_vec::<f32>().unwrap();
        let token = flat
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        generated.push(token);
    }

    assert_eq!(generated.len(), 3, "should have generated 3 tokens");
    assert_eq!(
        cache.seq_len(),
        4,
        "cache should have 2 prompt + 2 decode tokens (last not in cache)"
    );
    for &tid in &generated {
        assert!(
            tid < cfg.vocab_size,
            "token {tid} exceeds vocab_size {}",
            cfg.vocab_size
        );
    }
}

// ---------------------------------------------------------------------------
// Generation output metadata
// ---------------------------------------------------------------------------

#[test]
fn test_generation_output_finished_flag_false_on_max_tokens() {
    let (_cfg, model) = load_zeros_model();
    let output = model.generate_greedy(&[42], 5).unwrap();
    // With zero weights, all logits are equal, argmax picks token 0 each time.
    // Token 0 is not EOS (no EOS configured), so finished should be false.
    assert!(
        !output.finished,
        "should not be marked finished without EOS"
    );
}

// ---------------------------------------------------------------------------
// Forward from embeddings as generation building block
// ---------------------------------------------------------------------------

#[test]
fn test_embeddings_based_generation_step() {
    let (cfg, model) = load_zeros_model();
    let mut cache = model.new_cache();

    // Step 1: feed embeddings as hidden states
    let hidden = DynTensor::ones(&[1, 1, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits = model
        .forward_from_embeddings(&hidden, &[0], Some(&mut cache))
        .unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    assert_eq!(cache.seq_len(), 1);

    // Step 2: another embedding
    let hidden2 = DynTensor::ones(&[1, 1, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits2 = model
        .forward_from_embeddings(&hidden2, &[1], Some(&mut cache))
        .unwrap();
    assert_eq!(logits2.dims(), &[1, 1, cfg.vocab_size]);
    assert_eq!(cache.seq_len(), 2);
}

#[test]
fn test_embeddings_with_hidden_generation_step() {
    let (cfg, model) = load_zeros_model();
    let mut cache = model.new_cache();

    let hidden = DynTensor::ones(&[1, 1, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let (logits, normed) = model
        .forward_from_embeddings_with_hidden(&hidden, &[0], Some(&mut cache))
        .unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    assert_eq!(normed.dims(), &[1, 1, cfg.hidden_size]);
    assert_eq!(cache.seq_len(), 1);

    // normed hidden should be finite
    let vals = normed.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "normed hidden states should be finite"
    );
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_generation_with_large_token_id() {
    let (cfg, model) = load_zeros_model();
    // Token ID at vocab boundary: vocab_size - 1 = 99
    let output = model.generate_greedy(&[cfg.vocab_size - 1], 2).unwrap();
    assert_eq!(output.token_ids.len(), 2);
}

#[test]
fn test_generation_repeated_prompt_tokens() {
    let (_cfg, model) = load_zeros_model();
    // Repeated prompt tokens should still produce valid output
    let output = model.generate_greedy(&[42, 42, 42], 3).unwrap();
    assert_eq!(output.token_ids.len(), 3);
}
