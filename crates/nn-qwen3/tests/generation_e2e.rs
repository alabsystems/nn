// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end generation tests for Qwen3 with real weights (Qwen3-0.6B).
//!
//! These tests validate the full text generation pipeline — not just forward
//! pass correctness (covered by `real_weights.rs`), but autoregressive
//! generation behavior: single/multi-token generation, KV cache consistency,
//! temperature effects on logit distributions, top-k candidate restriction,
//! greedy determinism, logit shapes, and attention mask behavior.
//!
//! Gated behind `QWEN3_WEIGHTS` env var.
//!
//! ```bash
//! export QWEN3_WEIGHTS=./nn/weights/qwen3-0.6b.safetensors
//! cargo test -p nn-qwen3 --test generation_e2e -- --nocapture
//! ```

use nn_core::layers::kv_cache::KvCache;
use nn_core::layers::{generate, GenerationConfig};
use nn_qwen3::{Qwen3Config, Qwen3Model};

// ---------------------------------------------------------------------------
// Config for Qwen3-0.6B
// ---------------------------------------------------------------------------

fn qwen3_0_6b_config() -> Qwen3Config {
    Qwen3Config::new(
        1024,        // hidden_size
        3072,        // intermediate_size
        28,          // num_hidden_layers
        16,          // num_attention_heads
        8,           // num_key_value_heads
        151_936,     // vocab_size
        1e-6,        // rms_norm_eps
        1_000_000.0, // rope_theta
        40_960,      // max_position_embeddings
        true,        // tie_word_embeddings
        None,        // rope_scaling
    )
}

const VOCAB_SIZE: usize = 151_936;

/// "Hello world" token IDs from Qwen3 tokenizer.
const PROMPT_HELLO: &[usize] = &[9707, 1879];

/// A longer prompt using distinct valid IDs for Qwen3's vocab.
const PROMPT_LONGER: &[usize] = &[785, 3743, 13806, 38654, 34399, 916];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn weights_path() -> Option<String> {
    std::env::var("QWEN3_WEIGHTS").ok()
}

macro_rules! require_weights {
    () => {
        match weights_path() {
            Some(p) => p,
            None => {
                eprintln!("QWEN3_WEIGHTS not set, skipping generation e2e test");
                return;
            }
        }
    };
}

fn load_model(path: &str) -> Qwen3Model {
    let config = qwen3_0_6b_config();
    Qwen3Model::load_safetensors(path, config).expect("Failed to load Qwen3-0.6B")
}

/// Adapter closure for `nn_core::layers::generate` — bridges
/// the generic `(DynTensor, &mut KvCache) -> Result<DynTensor>` interface
/// to `Qwen3Model::forward_cached`.
fn make_model_fn(
    model: &Qwen3Model,
) -> impl Fn(
    &nn_core::dyn_tensor::DynTensor,
    &mut KvCache,
) -> nn_core::Result<nn_core::dyn_tensor::DynTensor>
       + '_ {
    move |input, cache| {
        let u32_data = input.to_flat_vec::<u32>()?;
        let ids: Vec<usize> = u32_data.iter().map(|&v| v as usize).collect();
        let offset = cache.seq_len();
        let positions: Vec<usize> = (0..ids.len()).map(|i| offset + i).collect();
        model.forward_cached(&ids, &positions, Some(cache))
    }
}

/// Extract argmax token from logits tensor of shape [1, seq_len, vocab_size].
/// Returns the argmax of the last position.
fn argmax_last(logits: &nn_core::dyn_tensor::DynTensor) -> usize {
    let seq_len = logits.dim(1).unwrap();
    let last_logits = logits
        .narrow(1, seq_len - 1, 1)
        .unwrap()
        .squeeze(1)
        .unwrap();
    let flat = last_logits.to_flat_vec::<f32>().unwrap();
    flat.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

/// Compute softmax probabilities from a flat f32 slice.
fn softmax_probs(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ---------------------------------------------------------------------------
// Test: single token generation
// ---------------------------------------------------------------------------

#[test]
fn test_single_token_generation() {
    let wpath = require_weights!();
    let model = load_model(&wpath);

    let output = model.generate_greedy(PROMPT_HELLO, 1).unwrap();

    assert_eq!(output.token_ids.len(), 1, "should generate exactly 1 token");
    assert!(
        output.token_ids[0] < VOCAB_SIZE,
        "generated token {} exceeds vocab size {}",
        output.token_ids[0],
        VOCAB_SIZE
    );

    // The generated token should match the argmax from a direct forward pass
    // at the last prompt position.
    let logits = model.forward(PROMPT_HELLO, &[0, 1]).unwrap();
    let expected = argmax_last(&logits);
    assert_eq!(
        output.token_ids[0], expected,
        "generate_greedy single token should match direct forward argmax"
    );

    eprintln!(
        "Single token generation: token={} (matches forward argmax)",
        output.token_ids[0]
    );
}

// ---------------------------------------------------------------------------
// Test: multi-token generation (10 tokens)
// ---------------------------------------------------------------------------

#[test]
fn test_multi_token_generation() {
    let wpath = require_weights!();
    let model = load_model(&wpath);

    let num_tokens = 10;
    let output = model.generate_greedy(PROMPT_HELLO, num_tokens).unwrap();

    assert_eq!(
        output.token_ids.len(),
        num_tokens,
        "should generate exactly {num_tokens} tokens"
    );

    // All tokens must be within vocab range
    for (i, &tid) in output.token_ids.iter().enumerate() {
        assert!(tid < VOCAB_SIZE, "token {i} ({tid}) exceeds vocab size");
    }

    // Verify the output is not marked as finished (no EOS configured)
    assert!(
        !output.finished,
        "should not be finished without EOS token configured"
    );

    eprintln!(
        "Multi-token generation ({num_tokens}): {:?}",
        &output.token_ids
    );
}

// ---------------------------------------------------------------------------
// Test: KV cache consistency -- cached step-by-step vs. no-cache full sequence
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_consistency() {
    let wpath = require_weights!();
    let model = load_model(&wpath);

    // Generate 5 tokens with greedy, then replay the full sequence
    // (prompt + generated tokens) in a single forward pass without cache.
    // The logits at each position should match within tolerance.

    let num_gen = 5;
    let output = model.generate_greedy(PROMPT_HELLO, num_gen).unwrap();
    assert_eq!(output.token_ids.len(), num_gen);

    // Build full sequence: prompt + generated
    let mut full_seq: Vec<usize> = PROMPT_HELLO.to_vec();
    full_seq.extend_from_slice(&output.token_ids);
    let positions: Vec<usize> = (0..full_seq.len()).collect();

    // Forward pass on the full sequence without cache
    let full_logits = model.forward(&full_seq, &positions).unwrap();
    assert_eq!(
        full_logits.dim(1).unwrap(),
        full_seq.len(),
        "full forward seq_len"
    );

    // Replay with cache step by step
    let mut cache = model.new_cache();

    // Prefill prompt
    let prompt_positions: Vec<usize> = (0..PROMPT_HELLO.len()).collect();
    let prefill_logits = model
        .forward_cached(PROMPT_HELLO, &prompt_positions, Some(&mut cache))
        .unwrap();

    // Compare prefill last-position logits vs full-sequence at same position
    let prefill_last = prefill_logits
        .narrow(1, PROMPT_HELLO.len() - 1, 1)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let full_at_prompt_end = full_logits
        .narrow(1, PROMPT_HELLO.len() - 1, 1)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let mut max_err: f32 = 0.0;
    for (a, b) in prefill_last.iter().zip(full_at_prompt_end.iter()) {
        let err = (a - b).abs();
        if err > max_err {
            max_err = err;
        }
    }
    assert!(
        max_err < 1e-4,
        "prefill last-position logits vs full-sequence mismatch: max_err={max_err:.6e}"
    );
    eprintln!("Prefill vs full-sequence last position: max_err={max_err:.6e}");

    // Decode each generated token and compare against full-sequence logits
    for (step, &gen_token) in output.token_ids.iter().enumerate() {
        let pos = PROMPT_HELLO.len() + step;
        let step_logits = model
            .forward_cached(&[gen_token], &[pos], Some(&mut cache))
            .unwrap();
        let step_flat = step_logits.to_flat_vec::<f32>().unwrap();

        let full_at_pos = full_logits
            .narrow(1, pos, 1)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap();

        let mut step_max_err: f32 = 0.0;
        for (a, b) in step_flat.iter().zip(full_at_pos.iter()) {
            let err = (a - b).abs();
            if err > step_max_err {
                step_max_err = err;
            }
        }

        assert!(
            step_max_err < 1e-4,
            "step {step} (pos {pos}) cached vs full logits: max_err={step_max_err:.6e}"
        );
        eprintln!("Step {step} (pos {pos}): cached vs full max_err={step_max_err:.6e}");
    }
}

// ---------------------------------------------------------------------------
// Test: temperature affects output distribution
// ---------------------------------------------------------------------------

#[test]
fn test_temperature_sampling() {
    let wpath = require_weights!();
    let model = load_model(&wpath);

    // Get raw logits for the prompt
    let logits = model.forward(PROMPT_HELLO, &[0, 1]).unwrap();
    let last_logits = logits
        .narrow(1, 1, 1)
        .unwrap()
        .squeeze(1)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Compute softmax at different temperatures
    let probs_t1: Vec<f32> = softmax_probs(&last_logits);

    // Temperature 0.5 (sharper)
    let scaled_05: Vec<f32> = last_logits.iter().map(|&x| x / 0.5).collect();
    let probs_t05 = softmax_probs(&scaled_05);

    // Temperature 2.0 (flatter)
    let scaled_20: Vec<f32> = last_logits.iter().map(|&x| x / 2.0).collect();
    let probs_t20 = softmax_probs(&scaled_20);

    // Find the max probability under each temperature
    let max_p_t1 = probs_t1.iter().copied().fold(0.0f32, f32::max);
    let max_p_t05 = probs_t05.iter().copied().fold(0.0f32, f32::max);
    let max_p_t20 = probs_t20.iter().copied().fold(0.0f32, f32::max);

    // Lower temperature -> sharper distribution -> higher max probability
    assert!(
        max_p_t05 > max_p_t1,
        "T=0.5 should be sharper: max_p_t05={max_p_t05:.6} vs max_p_t1={max_p_t1:.6}"
    );
    assert!(
        max_p_t1 > max_p_t20,
        "T=2.0 should be flatter: max_p_t1={max_p_t1:.6} vs max_p_t20={max_p_t20:.6}"
    );

    // Compute entropy at each temperature
    let entropy = |probs: &[f32]| -> f64 {
        probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -f64::from(p) * f64::from(p).ln())
            .sum::<f64>()
    };

    let h_t05 = entropy(&probs_t05);
    let h_t1 = entropy(&probs_t1);
    let h_t20 = entropy(&probs_t20);

    // Higher temperature -> higher entropy
    assert!(
        h_t20 > h_t1,
        "T=2.0 entropy ({h_t20:.4}) should exceed T=1.0 ({h_t1:.4})"
    );
    assert!(
        h_t1 > h_t05,
        "T=1.0 entropy ({h_t1:.4}) should exceed T=0.5 ({h_t05:.4})"
    );

    eprintln!("Temperature effects verified:");
    eprintln!("  T=0.5: max_p={max_p_t05:.6}, entropy={h_t05:.4}");
    eprintln!("  T=1.0: max_p={max_p_t1:.6}, entropy={h_t1:.4}");
    eprintln!("  T=2.0: max_p={max_p_t20:.6}, entropy={h_t20:.4}");

    // The argmax token should be the same regardless of temperature
    let argmax_t1 = probs_t1
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let argmax_t05 = probs_t05
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let argmax_t20 = probs_t20
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    assert_eq!(
        argmax_t1, argmax_t05,
        "argmax preserved across temperatures"
    );
    assert_eq!(
        argmax_t1, argmax_t20,
        "argmax preserved across temperatures"
    );
}

// ---------------------------------------------------------------------------
// Test: top-k constrains candidates
// ---------------------------------------------------------------------------

#[test]
fn test_top_k_sampling() {
    let wpath = require_weights!();
    let model = load_model(&wpath);

    // Get raw logits
    let logits = model.forward(PROMPT_HELLO, &[0, 1]).unwrap();
    let last_logits = logits
        .narrow(1, 1, 1)
        .unwrap()
        .squeeze(1)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Sort logits descending
    let mut indexed: Vec<(usize, f32)> = last_logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!(
        "Top-1 token: id={}, logit={:.4}",
        indexed[0].0, indexed[0].1
    );

    // Verify subset relationships
    let top_5: Vec<usize> = indexed.iter().take(5).map(|(i, _)| *i).collect();
    let top_10: Vec<usize> = indexed.iter().take(10).map(|(i, _)| *i).collect();
    let top_50: Vec<usize> = indexed.iter().take(50).map(|(i, _)| *i).collect();

    for &t in &top_5 {
        assert!(top_10.contains(&t), "top-5 token {t} should be in top-10");
        assert!(top_50.contains(&t), "top-5 token {t} should be in top-50");
    }

    // Verify logits are non-increasing after sort
    for w in indexed.windows(2).take(50) {
        assert!(
            w[0].1 >= w[1].1,
            "logits non-increasing: {} vs {}",
            w[0].1,
            w[1].1
        );
    }

    // Compute probability mass in top-k for various k values
    let probs = softmax_probs(&last_logits);
    for k in [1, 5, 10, 50, 100] {
        let mass: f32 = indexed.iter().take(k).map(|(i, _)| probs[*i]).sum();
        eprintln!("Top-{k} probability mass: {mass:.6}");
        if k >= 50 {
            assert!(
                mass > 0.5,
                "top-{k} should capture >50% probability mass, got {mass:.4}"
            );
        }
    }

    // Verify the logit spread is non-trivial
    let logit_range = indexed[0].1 - indexed.last().unwrap().1;
    assert!(
        logit_range > 1.0,
        "logit range should be significant, got {logit_range:.4}"
    );
    eprintln!(
        "Logit range: {logit_range:.4} (top={:.4}, bottom={:.4})",
        indexed[0].1,
        indexed.last().unwrap().1
    );
}

// ---------------------------------------------------------------------------
// Test: greedy determinism -- same input produces same output
// ---------------------------------------------------------------------------

#[test]
fn test_greedy_determinism() {
    let wpath = require_weights!();
    let model = load_model(&wpath);

    let num_tokens = 10;
    let out1 = model.generate_greedy(PROMPT_HELLO, num_tokens).unwrap();
    let out2 = model.generate_greedy(PROMPT_HELLO, num_tokens).unwrap();

    assert_eq!(
        out1.token_ids, out2.token_ids,
        "greedy decoding must be deterministic: run1={:?} vs run2={:?}",
        &out1.token_ids, &out2.token_ids
    );

    // Also test with a different prompt
    let out3 = model.generate_greedy(PROMPT_LONGER, num_tokens).unwrap();
    let out4 = model.generate_greedy(PROMPT_LONGER, num_tokens).unwrap();

    assert_eq!(
        out3.token_ids, out4.token_ids,
        "greedy decoding must be deterministic for longer prompt"
    );

    // Different prompts should (very likely) produce different output
    if out1.token_ids != out3.token_ids {
        eprintln!("Different prompts produce different outputs (expected)");
    } else {
        eprintln!("WARNING: different prompts produced identical output (unlikely but possible)");
    }

    eprintln!("Greedy determinism verified: {:?}", &out1.token_ids);
}

// ---------------------------------------------------------------------------
// Test: logit shapes match vocab size
// ---------------------------------------------------------------------------

#[test]
fn test_logit_shapes() {
    let wpath = require_weights!();
    let model = load_model(&wpath);

    // Single token input
    let logits_1 = model.forward(&[9707], &[0]).unwrap();
    assert_eq!(
        logits_1.dims(),
        &[1, 1, VOCAB_SIZE],
        "single token logit shape"
    );

    // Two token input
    let logits_2 = model.forward(PROMPT_HELLO, &[0, 1]).unwrap();
    assert_eq!(
        logits_2.dims(),
        &[1, 2, VOCAB_SIZE],
        "two token logit shape"
    );

    // Longer input
    let positions: Vec<usize> = (0..PROMPT_LONGER.len()).collect();
    let logits_6 = model.forward(PROMPT_LONGER, &positions).unwrap();
    assert_eq!(
        logits_6.dims(),
        &[1, PROMPT_LONGER.len(), VOCAB_SIZE],
        "six token logit shape"
    );

    // With KV cache: logits should always be [1, input_len, vocab_size]
    let mut cache = model.new_cache();

    let cached_logits_1 = model
        .forward_cached(&[9707], &[0], Some(&mut cache))
        .unwrap();
    assert_eq!(
        cached_logits_1.dims(),
        &[1, 1, VOCAB_SIZE],
        "cached single token logit shape"
    );

    let cached_logits_2 = model
        .forward_cached(&[1879], &[1], Some(&mut cache))
        .unwrap();
    assert_eq!(
        cached_logits_2.dims(),
        &[1, 1, VOCAB_SIZE],
        "cached step-2 logit shape"
    );

    // All logits should be finite
    for (name, logits) in [
        ("single", &logits_1),
        ("two", &logits_2),
        ("six", &logits_6),
        ("cached_1", &cached_logits_1),
        ("cached_2", &cached_logits_2),
    ] {
        let flat = logits.to_flat_vec::<f32>().unwrap();
        let non_finite = flat.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(non_finite, 0, "{name} logits should all be finite");
    }

    eprintln!("All logit shapes verified: [1, seq_len, {VOCAB_SIZE}]");
}

// ---------------------------------------------------------------------------
// Test: attention mask affects output (causal masking)
// ---------------------------------------------------------------------------

#[test]
fn test_attention_mask() {
    let wpath = require_weights!();
    let model = load_model(&wpath);

    // With a causal mask, logits at position 0 should be identical
    // regardless of future tokens (position 0 cannot attend to positions > 0).

    // Forward full sequence [A, B, C]
    let tokens = &[9707, 1879, 220];
    let positions = &[0, 1, 2];
    let full_logits = model.forward(tokens, positions).unwrap();

    // Extract logits at position 0 from full forward
    let full_pos0 = full_logits
        .narrow(1, 0, 1)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Forward only position 0 (single token)
    let single_logits = model.forward(&[9707], &[0]).unwrap();
    let single_pos0 = single_logits.to_flat_vec::<f32>().unwrap();

    // Position 0 logits should be identical
    let mut max_err: f32 = 0.0;
    for (a, b) in full_pos0.iter().zip(single_pos0.iter()) {
        let err = (a - b).abs();
        if err > max_err {
            max_err = err;
        }
    }

    assert!(
        max_err < 1e-5,
        "position 0 logits should be identical with/without future tokens: max_err={max_err:.6e}"
    );
    eprintln!("Causal mask: position 0 logits identical (max_err={max_err:.6e})");

    // Verify that position 1 logits ARE different when different tokens precede
    let alt_full_logits = model.forward(&[220, 1879, 9707], &[0, 1, 2]).unwrap();
    let alt_pos1 = alt_full_logits
        .narrow(1, 1, 1)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let orig_pos1 = full_logits
        .narrow(1, 1, 1)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let mut diff: f32 = 0.0;
    for (a, b) in orig_pos1.iter().zip(alt_pos1.iter()) {
        diff += (a - b).abs();
    }

    assert!(
        diff > 1.0,
        "changing preceding tokens should affect position 1 logits: total_diff={diff:.4}"
    );
    eprintln!("Attention context: different prefix -> different logits (diff={diff:.4})");
}

// ---------------------------------------------------------------------------
// Test: generation with custom GenerationConfig (via direct generate call)
// ---------------------------------------------------------------------------

#[test]
fn test_generation_with_custom_config() {
    let wpath = require_weights!();
    let model = load_model(&wpath);

    // Use the generate function directly with a custom config
    let device = model.device();
    let mut cache = model.new_cache();
    let config = GenerationConfig::new(5).with_temperature(0.0);

    let output = generate(
        make_model_fn(&model),
        PROMPT_HELLO,
        &mut cache,
        &config,
        &device,
    )
    .unwrap();

    assert_eq!(output.token_ids.len(), 5);
    for &tid in &output.token_ids {
        assert!(tid < VOCAB_SIZE, "token {tid} exceeds vocab size");
    }

    // Should match generate_greedy output
    let greedy_output = model.generate_greedy(PROMPT_HELLO, 5).unwrap();
    assert_eq!(
        output.token_ids, greedy_output.token_ids,
        "generate with temperature=0 should match generate_greedy"
    );

    eprintln!("Custom config generation: {:?}", &output.token_ids);
}

// ---------------------------------------------------------------------------
// Test: multi-token prompt prefill + single-step decode
// ---------------------------------------------------------------------------

#[test]
fn test_prefill_then_decode() {
    let wpath = require_weights!();
    let model = load_model(&wpath);

    let mut cache = model.new_cache();
    let positions: Vec<usize> = (0..PROMPT_LONGER.len()).collect();
    let prefill_logits = model
        .forward_cached(PROMPT_LONGER, &positions, Some(&mut cache))
        .unwrap();

    assert_eq!(cache.seq_len(), PROMPT_LONGER.len());
    assert_eq!(prefill_logits.dims(), &[1, PROMPT_LONGER.len(), VOCAB_SIZE]);

    // Decode one token
    let next_token = argmax_last(&prefill_logits);
    assert!(next_token < VOCAB_SIZE);

    let decode_pos = PROMPT_LONGER.len();
    let decode_logits = model
        .forward_cached(&[next_token], &[decode_pos], Some(&mut cache))
        .unwrap();

    assert_eq!(decode_logits.dims(), &[1, 1, VOCAB_SIZE]);
    assert_eq!(cache.seq_len(), PROMPT_LONGER.len() + 1);

    let flat = decode_logits.to_flat_vec::<f32>().unwrap();
    let non_finite = flat.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(non_finite, 0, "decode step logits should be finite");

    eprintln!(
        "Prefill ({} tokens) + decode: next_token={next_token}, cache_seq_len={}",
        PROMPT_LONGER.len(),
        cache.seq_len()
    );
}

// ---------------------------------------------------------------------------
// Test: generation coherence (not random noise)
// ---------------------------------------------------------------------------

#[test]
fn test_generation_coherence() {
    let wpath = require_weights!();
    let model = load_model(&wpath);

    // Generate 20 tokens from "Hello world"
    let output = model.generate_greedy(PROMPT_HELLO, 20).unwrap();
    assert_eq!(output.token_ids.len(), 20);

    // All tokens are valid
    for &tid in &output.token_ids {
        assert!(tid < VOCAB_SIZE, "token {tid} exceeds vocab size");
    }

    // With a real 0.6B model, we expect variety. Allow up to 15 repeats of a
    // single token (very conservative).
    let max_repeat = output
        .token_ids
        .iter()
        .map(|t| output.token_ids.iter().filter(|&x| x == t).count())
        .max()
        .unwrap_or(0);
    assert!(
        max_repeat <= 15,
        "degenerate repetition: one token appears {max_repeat}/20 times"
    );

    let unique: std::collections::HashSet<usize> = output.token_ids.iter().copied().collect();
    eprintln!(
        "Generated 20 tokens, {} unique: {:?}",
        unique.len(),
        &output.token_ids
    );
}
