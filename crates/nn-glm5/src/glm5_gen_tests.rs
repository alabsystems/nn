// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Glm5Model generate_greedy() and generate_beam() convenience wrappers.
//!
//! Coverage: greedy generation with various prompt lengths, beam search with
//! different beam widths, KV cache consistency, cache reuse across generation
//! steps, position encoding auto-increment, edge cases (empty prompt,
//! max_new_tokens=0), output shape verification.

use super::*;
use crate::test_utils::tiny_config;
use nn_core::layers::BeamSearchConfig;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// Helper: build a tiny model with zero weights on CPU.
// ---------------------------------------------------------------------------

fn load_tiny_model() -> Glm5Model {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    Glm5Model::load(&vb, cfg).unwrap()
}

// ---------------------------------------------------------------------------
// Greedy generation — basic
// ---------------------------------------------------------------------------

#[test]
fn test_generate_greedy_produces_tokens() {
    let model = load_tiny_model();
    // Zero weights -> all logits equal -> argmax picks token 0 each step
    let output = model.generate_greedy(&[42], 3).unwrap();
    assert_eq!(output.token_ids.len(), 3, "should generate 3 tokens");
}

#[test]
fn test_generate_greedy_respects_max_tokens() {
    let model = load_tiny_model();

    let out1 = model.generate_greedy(&[42], 1).unwrap();
    assert_eq!(out1.token_ids.len(), 1);

    let out5 = model.generate_greedy(&[42], 5).unwrap();
    assert_eq!(out5.token_ids.len(), 5);
}

// ---------------------------------------------------------------------------
// Greedy generation — various prompt lengths (1, 5, 20 tokens)
// ---------------------------------------------------------------------------

#[test]
fn test_greedy_prompt_length_1() {
    let model = load_tiny_model();
    let output = model.generate_greedy(&[10], 4).unwrap();
    assert_eq!(output.token_ids.len(), 4);
    // All generated tokens should be valid vocab indices (< padded_vocab_size=100)
    for &tok in &output.token_ids {
        assert!(tok < 100, "token {tok} out of vocab range");
    }
}

#[test]
fn test_greedy_prompt_length_5() {
    let model = load_tiny_model();
    let prompt: Vec<usize> = (0..5).collect();
    let output = model.generate_greedy(&prompt, 3).unwrap();
    assert_eq!(output.token_ids.len(), 3);
    for &tok in &output.token_ids {
        assert!(tok < 100, "token {tok} out of vocab range");
    }
}

#[test]
fn test_greedy_prompt_length_20() {
    let model = load_tiny_model();
    let prompt: Vec<usize> = (0..20).collect();
    let output = model.generate_greedy(&prompt, 2).unwrap();
    assert_eq!(output.token_ids.len(), 2);
    for &tok in &output.token_ids {
        assert!(tok < 100, "token {tok} out of vocab range");
    }
}

#[test]
fn test_greedy_deterministic_across_calls() {
    // With zero weights and temperature=0 (default), greedy generation
    // should produce identical output across two independent calls.
    let model = load_tiny_model();
    let out_a = model.generate_greedy(&[5, 10, 15], 4).unwrap();
    let out_b = model.generate_greedy(&[5, 10, 15], 4).unwrap();
    assert_eq!(
        out_a.token_ids, out_b.token_ids,
        "greedy generation should be deterministic"
    );
}

// ---------------------------------------------------------------------------
// Beam search — beam_width=2 and beam_width=4
// ---------------------------------------------------------------------------

#[test]
fn test_generate_beam_produces_beams() {
    let model = load_tiny_model();
    let mut beam_cfg = BeamSearchConfig::default();
    beam_cfg.beam_width = 2;
    beam_cfg.max_new_tokens = 3;

    let output = model.generate_beam(&[42], &beam_cfg).unwrap();
    assert!(!output.beams.is_empty(), "should produce at least one beam");
    assert!(
        output.beams.len() <= 2,
        "should produce at most beam_width beams"
    );
    for beam in &output.beams {
        assert!(
            !beam.token_ids.is_empty(),
            "beam should have generated tokens"
        );
        assert!(
            beam.token_ids.len() <= 3,
            "beam should respect max_new_tokens"
        );
    }
}

#[test]
fn test_generate_beam_width_2() {
    let model = load_tiny_model();
    let mut beam_cfg = BeamSearchConfig::default();
    beam_cfg.beam_width = 2;
    beam_cfg.max_new_tokens = 4;
    beam_cfg.length_penalty = 0.0;

    let output = model.generate_beam(&[1, 2, 3], &beam_cfg).unwrap();
    assert!(
        output.beams.len() <= 2,
        "beam_width=2 should produce at most 2 beams, got {}",
        output.beams.len()
    );
    // Each beam should have at most max_new_tokens tokens
    for (i, beam) in output.beams.iter().enumerate() {
        assert!(
            beam.token_ids.len() <= 4,
            "beam {i} has {} tokens, expected <= 4",
            beam.token_ids.len()
        );
    }
}

#[test]
fn test_generate_beam_width_4() {
    let model = load_tiny_model();
    let mut beam_cfg = BeamSearchConfig::default();
    beam_cfg.beam_width = 4;
    beam_cfg.max_new_tokens = 3;
    beam_cfg.length_penalty = 0.0;

    let output = model.generate_beam(&[42], &beam_cfg).unwrap();
    assert!(
        output.beams.len() <= 4,
        "beam_width=4 should produce at most 4 beams, got {}",
        output.beams.len()
    );
    // All beams should have generated at least one token
    for beam in &output.beams {
        assert!(
            !beam.token_ids.is_empty(),
            "each beam should have at least one token"
        );
    }
}

#[test]
fn test_generate_beam_sorted_by_score() {
    let model = load_tiny_model();
    let mut beam_cfg = BeamSearchConfig::default();
    beam_cfg.beam_width = 4;
    beam_cfg.max_new_tokens = 2;
    beam_cfg.length_penalty = 0.0;

    let output = model.generate_beam(&[42], &beam_cfg).unwrap();
    // Beams should be sorted by log_prob (descending)
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
fn test_generate_beam_token_ids_in_vocab_range() {
    let model = load_tiny_model();
    let vocab_size = model.config().padded_vocab_size;
    let mut beam_cfg = BeamSearchConfig::default();
    beam_cfg.beam_width = 3;
    beam_cfg.max_new_tokens = 5;

    let output = model.generate_beam(&[10, 20], &beam_cfg).unwrap();
    for beam in &output.beams {
        for &tok in &beam.token_ids {
            assert!(
                tok < vocab_size,
                "token {tok} exceeds vocab_size {vocab_size}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// KV cache consistency: cached vs uncached produce same logits
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_consistency_single_step() {
    // Forward with cache (single step prefill) should produce the same logits
    // as forward without cache for the same input.
    let model = load_tiny_model();
    let input_ids = &[0, 1, 2];
    let positions = &[0, 1, 2];

    // Uncached forward
    let logits_uncached = model.forward(input_ids, positions).unwrap();

    // Cached forward (fresh cache)
    let mut cache = model.new_cache();
    let logits_cached = model
        .forward_cached(input_ids, positions, Some(&mut cache))
        .unwrap();

    let uncached_flat = logits_uncached.to_flat_vec::<f32>().unwrap();
    let cached_flat = logits_cached.to_flat_vec::<f32>().unwrap();

    assert_eq!(
        uncached_flat.len(),
        cached_flat.len(),
        "logit vectors should have the same length"
    );
    for (i, (a, b)) in uncached_flat.iter().zip(cached_flat.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "logit mismatch at index {i}: uncached={a}, cached={b}"
        );
    }
}

#[test]
fn test_kv_cache_consistency_incremental_vs_full() {
    // Process tokens [0, 1, 2] incrementally (one at a time with cache) and
    // compare the final-position logits with a full uncached forward pass.
    let model = load_tiny_model();

    // Full pass without cache
    let logits_full = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
    let full_flat = logits_full.to_flat_vec::<f32>().unwrap();
    let vocab = model.config().padded_vocab_size;
    // Extract last position logits (position 2)
    let full_last = &full_flat[2 * vocab..3 * vocab];

    // Incremental pass with cache
    let mut cache = model.new_cache();
    let _ = model.forward_cached(&[0], &[0], Some(&mut cache)).unwrap();
    let _ = model.forward_cached(&[1], &[1], Some(&mut cache)).unwrap();
    let logits_incr = model.forward_cached(&[2], &[2], Some(&mut cache)).unwrap();
    let incr_flat = logits_incr.to_flat_vec::<f32>().unwrap();

    assert_eq!(incr_flat.len(), vocab);
    for (i, (a, b)) in full_last.iter().zip(incr_flat.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "logit mismatch at vocab index {i}: full={a}, incremental={b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Cache reuse across multiple generation steps
// ---------------------------------------------------------------------------

#[test]
fn test_cache_reuse_multi_step() {
    let model = load_tiny_model();
    let mut cache = model.new_cache();

    // Step 1: prefill with 3 tokens
    let _ = model
        .forward_cached(&[5, 10, 15], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(
        cache.seq_len(),
        3,
        "cache should have 3 tokens after prefill"
    );

    // Step 2: decode one token
    let _ = model.forward_cached(&[20], &[3], Some(&mut cache)).unwrap();
    assert_eq!(
        cache.seq_len(),
        4,
        "cache should have 4 tokens after step 2"
    );

    // Step 3: decode another token
    let _ = model.forward_cached(&[25], &[4], Some(&mut cache)).unwrap();
    assert_eq!(
        cache.seq_len(),
        5,
        "cache should have 5 tokens after step 3"
    );

    // Step 4: decode two tokens at once
    let logits = model
        .forward_cached(&[30, 35], &[5, 6], Some(&mut cache))
        .unwrap();
    assert_eq!(
        cache.seq_len(),
        7,
        "cache should have 7 tokens after step 4"
    );
    assert_eq!(
        logits.dims(),
        &[1, 2, 100],
        "logits shape for 2-token decode"
    );
}

#[test]
fn test_cache_seq_len_grows_monotonically() {
    let model = load_tiny_model();
    let mut cache = model.new_cache();

    let mut prev_len = 0;
    for step in 0..6 {
        let token_id = step % model.config().padded_vocab_size;
        let _ = model
            .forward_cached(&[token_id], &[step], Some(&mut cache))
            .unwrap();
        let new_len = cache.seq_len();
        assert!(
            new_len > prev_len,
            "cache length should grow: step={step}, prev={prev_len}, new={new_len}"
        );
        prev_len = new_len;
    }
    assert_eq!(prev_len, 6);
}

// ---------------------------------------------------------------------------
// Position encoding correctness (positions auto-increment in adapter)
// ---------------------------------------------------------------------------

#[test]
fn test_model_fn_adapter_position_calculation() {
    let model = load_tiny_model();
    let mut cache = model.new_cache();

    // First call: cache is empty, so offset = 0, positions = [0, 1]
    let input = DynTensor::from_vec_u32(vec![42, 7], &[2], &Device::Cpu).unwrap();
    let logits = model.model_fn_adapter(&input, &mut cache).unwrap();
    assert_eq!(cache.seq_len(), 2);
    assert_eq!(logits.dims()[1], 2);

    // Second call: cache has 2 tokens, so offset = 2, positions = [2]
    let input2 = DynTensor::from_vec_u32(vec![0], &[1], &Device::Cpu).unwrap();
    let logits2 = model.model_fn_adapter(&input2, &mut cache).unwrap();
    assert_eq!(cache.seq_len(), 3);
    assert_eq!(logits2.dims()[1], 1);
}

#[test]
fn test_adapter_positions_auto_increment_through_multiple_steps() {
    // Verify that the model_fn_adapter correctly computes positions from cache
    // seq_len across many sequential single-token calls.
    let model = load_tiny_model();
    let mut cache = model.new_cache();

    for step in 0..8 {
        let input = DynTensor::from_vec_u32(vec![step as u32], &[1], &Device::Cpu).unwrap();
        let logits = model.model_fn_adapter(&input, &mut cache).unwrap();

        // After each step, cache should reflect the number of tokens processed
        assert_eq!(
            cache.seq_len(),
            step + 1,
            "cache seq_len should be {} after step {step}",
            step + 1
        );
        // Logits should always be [1, 1, vocab_size] for single-token input
        assert_eq!(
            logits.dims(),
            &[1, 1, 100],
            "single-token logits shape wrong at step {step}"
        );
    }
}

#[test]
fn test_adapter_multi_token_prefix_then_single() {
    // Prefill with a multi-token prompt, then decode one-by-one.
    // Positions should be [0..n] for prefill, then [n], [n+1], ... for decode.
    let model = load_tiny_model();
    let mut cache = model.new_cache();

    // Prefill 4 tokens
    let prefix = DynTensor::from_vec_u32(vec![1, 2, 3, 4], &[4], &Device::Cpu).unwrap();
    let prefill_logits = model.model_fn_adapter(&prefix, &mut cache).unwrap();
    assert_eq!(cache.seq_len(), 4);
    assert_eq!(prefill_logits.dims()[1], 4);

    // Decode 3 more tokens
    for i in 0..3 {
        let tok = DynTensor::from_vec_u32(vec![10 + i], &[1], &Device::Cpu).unwrap();
        let logits = model.model_fn_adapter(&tok, &mut cache).unwrap();
        assert_eq!(cache.seq_len(), 5 + i as usize);
        assert_eq!(logits.dims()[1], 1);
    }
    assert_eq!(cache.seq_len(), 7);
}

// ---------------------------------------------------------------------------
// Edge case: empty prompt (should error gracefully)
// ---------------------------------------------------------------------------

#[test]
fn test_generate_greedy_empty_prompt_errors() {
    let model = load_tiny_model();
    let result = model.generate_greedy(&[], 5);
    assert!(
        result.is_err(),
        "empty prompt should produce an error, not succeed"
    );
}

#[test]
fn test_generate_beam_empty_prompt_errors() {
    let model = load_tiny_model();
    let mut beam_cfg = BeamSearchConfig::default();
    beam_cfg.beam_width = 2;
    beam_cfg.max_new_tokens = 3;

    let result = model.generate_beam(&[], &beam_cfg);
    assert!(
        result.is_err(),
        "beam search with empty prompt should produce an error"
    );
}

// ---------------------------------------------------------------------------
// Edge case: max_new_tokens=0 (should return just prompt / empty tokens)
// ---------------------------------------------------------------------------

#[test]
fn test_generate_greedy_zero_new_tokens() {
    let model = load_tiny_model();
    let output = model.generate_greedy(&[42, 7], 0).unwrap();
    assert!(
        output.token_ids.is_empty(),
        "max_new_tokens=0 should produce no generated tokens, got {:?}",
        output.token_ids
    );
}

#[test]
fn test_generate_beam_zero_new_tokens() {
    let model = load_tiny_model();
    let mut beam_cfg = BeamSearchConfig::default();
    beam_cfg.beam_width = 2;
    beam_cfg.max_new_tokens = 0;

    let output = model.generate_beam(&[42], &beam_cfg).unwrap();
    // With max_new_tokens=0, each beam should have no generated tokens
    for (i, beam) in output.beams.iter().enumerate() {
        assert!(
            beam.token_ids.is_empty(),
            "beam {i} should have 0 tokens with max_new_tokens=0, got {}",
            beam.token_ids.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Output shape verification: logits should be [1, seq_len, padded_vocab_size]
// ---------------------------------------------------------------------------

#[test]
fn test_forward_logits_shape_single_token() {
    let model = load_tiny_model();
    let logits = model.forward(&[42], &[0]).unwrap();
    assert_eq!(
        logits.dims(),
        &[1, 1, 100],
        "single-token logits: [1, 1, padded_vocab_size]"
    );
}

#[test]
fn test_forward_logits_shape_multi_token() {
    let model = load_tiny_model();
    let logits = model.forward(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4]).unwrap();
    assert_eq!(
        logits.dims(),
        &[1, 5, 100],
        "5-token logits: [1, 5, padded_vocab_size]"
    );
}

#[test]
fn test_forward_logits_shape_various_lengths() {
    let model = load_tiny_model();
    let vocab = model.config().padded_vocab_size;

    for seq_len in [1, 2, 3, 7, 10, 16] {
        let ids: Vec<usize> = (0..seq_len).collect();
        let positions: Vec<usize> = (0..seq_len).collect();
        let logits = model.forward(&ids, &positions).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, seq_len, vocab],
            "logits shape mismatch for seq_len={seq_len}"
        );
    }
}

#[test]
fn test_cached_forward_logits_shape() {
    // Verify logits shape through cached decode steps.
    let model = load_tiny_model();
    let vocab = model.config().padded_vocab_size;
    let mut cache = model.new_cache();

    // Prefill 3 tokens
    let logits = model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(logits.dims(), &[1, 3, vocab]);

    // Decode 1 token
    let logits = model.forward_cached(&[3], &[3], Some(&mut cache)).unwrap();
    assert_eq!(logits.dims(), &[1, 1, vocab]);

    // Decode 2 tokens at once
    let logits = model
        .forward_cached(&[4, 5], &[4, 5], Some(&mut cache))
        .unwrap();
    assert_eq!(logits.dims(), &[1, 2, vocab]);
}

#[test]
fn test_logits_are_finite() {
    // All logits should be finite (no NaN, no Inf) with zero weights.
    let model = load_tiny_model();
    let logits = model.forward(&[0, 1, 2, 3], &[0, 1, 2, 3]).unwrap();
    let flat = logits.to_flat_vec::<f32>().unwrap();
    for (i, &v) in flat.iter().enumerate() {
        assert!(v.is_finite(), "logit at index {i} is not finite: {v}");
    }
}

// ---------------------------------------------------------------------------
// Device accessor
// ---------------------------------------------------------------------------

#[test]
fn test_device_accessor() {
    let model = load_tiny_model();
    assert!(matches!(model.device(), Device::Cpu));
}

// ---------------------------------------------------------------------------
// Config accessor
// ---------------------------------------------------------------------------

#[test]
fn test_config_accessor_matches() {
    let model = load_tiny_model();
    let cfg = model.config();
    assert_eq!(cfg.hidden_size, 256);
    assert_eq!(cfg.num_layers, 2);
    assert_eq!(cfg.padded_vocab_size, 100);
    assert_eq!(cfg.num_attention_heads, 4);
    assert_eq!(cfg.multi_query_group_num, 2);
}

// ---------------------------------------------------------------------------
// Greedy generation with EOS token
// ---------------------------------------------------------------------------

#[test]
fn test_greedy_generation_all_tokens_deterministic() {
    // With zero weights, all logits are equal so argmax breaks the tie
    // deterministically. The `argmax` helper uses `max_by(total_cmp)`, which
    // returns the LAST maximal element on ties, so it selects the highest index
    // (padded_vocab_size - 1), not 0. Verify this holds across all decode steps.
    let model = load_tiny_model();
    let expected = model.config().padded_vocab_size - 1;
    let out = model.generate_greedy(&[50], 5).unwrap();
    // With zero weights, all logits are identical -> argmax picks the last index
    for &tok in &out.token_ids {
        assert_eq!(
            tok, expected,
            "zero-weight greedy should always produce the last vocab token (argmax tie-break)"
        );
    }
}
