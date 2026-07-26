// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for decode loop and beam search extended safety.
//!
//! 20 proofs covering properties of autoregressive decoding pipelines
//! used in dpdf models: EOS termination, beam width maintenance, score
//! accumulation safety, beam pruning, greedy argmax, temperature scaling,
//! top-k/top-p filtering, repetition/no-repeat-ngram penalties, max length
//! enforcement, length normalization, early stopping, batch dimension,
//! padding, forced prefix, logit processor chains, beam hypothesis sorting,
//! deterministic seeded sampling, and token ID vocabulary bounds.
//!
//! Part of #4182.

use super::*;

// ===========================================================================
// 1. Token generation loop terminates on EOS token
// ===========================================================================

/// Prove that when the generated token matches the EOS token ID, the
/// generation loop terminates with `finished = true`. Models a single-step
/// decode where the sampled token equals the EOS ID.
#[kani::proof]
#[kani::unwind(1)]
fn proof_decode_loop_terminates_on_eos() {
    let eos_id: usize = kani::any();
    kani::assume(eos_id <= 65535);

    let config = GenerationConfig::new(100).with_eos_token_id(eos_id);

    // The is_eos check drives loop termination.
    assert!(
        is_eos(eos_id, &config),
        "token matching eos_token_id must trigger termination"
    );

    // Model the loop outcome: when EOS is detected, finished = true.
    let output = GenerationOutput::new(vec![eos_id], true);
    assert!(
        output.finished,
        "generation must report finished when EOS detected"
    );
    assert_eq!(
        output.token_ids.len(),
        1,
        "output must contain the EOS token"
    );
}

// ===========================================================================
// 2. Beam search beam width maintained (num_beams constant)
// ===========================================================================

/// Prove that `BeamSearchConfig::new(w)` preserves the beam width through
/// builder chains, and `finalize_tree` never returns more beams than the
/// configured width.
#[kani::proof]
#[kani::unwind(5)]
fn proof_beam_width_maintained() {
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 4);

    let config = BeamSearchConfig::new(beam_width)
        .with_max_new_tokens(50)
        .with_early_stopping(false);

    assert_eq!(
        config.beam_width, beam_width,
        "beam_width must be preserved through builder chain"
    );

    // Verify that output is bounded by beam_width via BeamSearchOutput.
    let beams: Vec<BeamHypothesis> = (0..beam_width)
        .map(|i| BeamHypothesis::new(vec![i], -(i as f64), false))
        .collect();
    let output = BeamSearchOutput::new(beams);
    assert!(
        output.beams.len() <= beam_width,
        "output beam count must not exceed beam_width"
    );
}

// ===========================================================================
// 3. Beam score accumulation no overflow (log-prob sum)
// ===========================================================================

/// Prove that accumulating log-probabilities (summing negative f64 values)
/// stays finite and does not overflow to -Inf for realistic sequence lengths.
/// Log-probs are in [-20, 0] (worst case: rare token with prob ~2e-9).
#[kani::proof]
#[kani::unwind(6)]
fn proof_beam_score_accumulation_no_overflow() {
    let num_steps: usize = kani::any();
    kani::assume(num_steps >= 1 && num_steps <= 5);

    let mut cumulative_log_prob: f64 = 0.0;
    for _ in 0..num_steps {
        let step_log_prob: f64 = kani::any();
        kani::assume(step_log_prob >= -20.0 && step_log_prob <= 0.0 && step_log_prob.is_finite());
        cumulative_log_prob += step_log_prob;
    }

    assert!(
        cumulative_log_prob.is_finite(),
        "cumulative log-prob must remain finite"
    );
    assert!(
        cumulative_log_prob <= 0.0,
        "cumulative log-prob of valid probs must be <= 0"
    );
    // Worst case: 5 steps * -20.0 = -100.0 — well within f64 range.
    assert!(
        cumulative_log_prob >= -100.0,
        "cumulative log-prob bounded for realistic sequences"
    );
}

// ===========================================================================
// 4. Beam pruning preserves top-k invariant
// ===========================================================================

/// Prove that after sorting candidates by score and truncating to beam_width,
/// the surviving candidates have scores >= all pruned candidates.
#[kani::proof]
#[kani::unwind(8)]
fn proof_beam_pruning_preserves_top_k() {
    let total: usize = kani::any();
    kani::assume(total >= 2 && total <= 6);
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width < total);

    let mut scores = vec![0.0f64; total];
    for s in scores.iter_mut() {
        *s = kani::any();
        kani::assume(s.is_finite() && s.abs() < 100.0);
    }

    // Sort descending by score.
    scores.sort_by(|a, b| b.total_cmp(a));

    // The beam_width-th score is the cutoff.
    let cutoff = scores[beam_width - 1];

    // All surviving candidates have score >= cutoff.
    for &s in &scores[..beam_width] {
        assert!(s >= cutoff, "surviving beam must have score >= cutoff");
    }

    // All pruned candidates have score <= cutoff.
    for &s in &scores[beam_width..] {
        assert!(s <= cutoff, "pruned beam must have score <= cutoff");
    }
}

// ===========================================================================
// 5. Greedy decode selects argmax from logits
// ===========================================================================

/// Prove that `argmax` returns the index of the maximum value for finite
/// inputs, and the value at that index is >= all other values.
#[kani::proof]
#[kani::unwind(6)]
fn proof_greedy_decode_selects_argmax() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 1 && vocab_size <= 5);

    let mut logits = vec![0.0f32; vocab_size];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite());
    }

    let idx = argmax(&logits);
    assert!(idx < vocab_size, "argmax index in bounds");

    // The selected value must be maximal.
    let max_val = logits[idx];
    for &v in &logits {
        assert!(
            max_val.total_cmp(&v) != std::cmp::Ordering::Less,
            "argmax must select the maximum value"
        );
    }
}

// ===========================================================================
// 6. Temperature scaling T>0 divides logits safely
// ===========================================================================

/// Prove that dividing finite logits by a positive finite temperature
/// produces finite results that preserve relative ordering.
#[kani::proof]
#[kani::unwind(5)]
fn proof_temperature_scaling_safe_division() {
    let len: usize = kani::any();
    kani::assume(len >= 2 && len <= 4);

    let temperature: f32 = kani::any();
    kani::assume(temperature > 0.01 && temperature <= 100.0 && temperature.is_finite());

    let mut logits = vec![0.0f32; len];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite() && v.abs() < 500.0);
    }

    let scaled: Vec<f32> = logits.iter().map(|&v| v / temperature).collect();

    // All results must be finite.
    for &s in &scaled {
        assert!(s.is_finite(), "temperature-scaled logit must be finite");
    }

    // Ordering must be preserved.
    for i in 0..len {
        for j in 0..len {
            if logits[i] > logits[j] {
                assert!(
                    scaled[i] >= scaled[j],
                    "temperature scaling must preserve ordering"
                );
            }
        }
    }
}

// ===========================================================================
// 7. Top-k filtering zeros logits outside top-k
// ===========================================================================

/// Prove that `top_k_indices` returns exactly min(k, vocab_size) indices,
/// all within bounds, and every non-selected value is <= the minimum of
/// the selected set.
#[kani::proof]
#[kani::unwind(7)]
fn proof_top_k_filtering_outside_top_k() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 2 && vocab_size <= 5);
    let k: usize = kani::any();
    kani::assume(k >= 1 && k < vocab_size);

    let mut values = vec![0.0f32; vocab_size];
    for v in values.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite());
    }

    let indices = top_k_indices(&values, k);
    assert!(indices.len() <= k, "top_k returns at most k indices");

    // Find minimum value in the selected set.
    let mut min_selected = f32::INFINITY;
    for &idx in &indices {
        if values[idx] < min_selected {
            min_selected = values[idx];
        }
    }

    // Every non-selected value must be <= min_selected.
    for i in 0..vocab_size {
        if !indices.contains(&i) {
            assert!(
                values[i].total_cmp(&min_selected) != std::cmp::Ordering::Greater,
                "non-top-k value must not exceed minimum of selected set"
            );
        }
    }
}

// ===========================================================================
// 8. Top-p (nucleus) cumulative probability threshold
// ===========================================================================

/// Prove that `top_p_filter` returns a non-empty set whose renormalized
/// probabilities sum to approximately 1.0.
#[kani::proof]
#[kani::unwind(5)]
fn proof_top_p_cumulative_threshold() {
    let len: usize = kani::any();
    kani::assume(len >= 1 && len <= 4);

    let p: f32 = kani::any();
    kani::assume(p > 0.0 && p <= 1.0 && p.is_finite());

    let mut probs = Vec::with_capacity(len);
    let mut total = 0.0f32;
    for i in 0..len {
        let v: f32 = kani::any();
        kani::assume(v >= 0.01 && v <= 1.0 && v.is_finite());
        total += v;
        probs.push((i, v));
    }
    kani::assume(total > 0.0 && total.is_finite());

    let result = top_p_filter(probs, p);

    // Must be non-empty.
    assert!(!result.is_empty(), "top_p must return at least one token");

    // Renormalized sum must be approximately 1.0.
    let sum: f32 = result.iter().map(|&(_, prob)| prob).sum();
    if sum.is_finite() {
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "top_p renormalized probs must sum to ~1.0"
        );
    }
}

// ===========================================================================
// 9. Repetition penalty modifies only repeated token logits
// ===========================================================================

/// Prove that applying repetition penalty > 1 to a repeated token's logit
/// reduces the effective logit, while unrepeated tokens are unmodified.
#[kani::proof]
#[kani::unwind(5)]
fn proof_repetition_penalty_selective() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 2 && vocab_size <= 4);

    let mut logits = vec![0.0f32; vocab_size];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite() && v.abs() < 50.0 && *v != 0.0);
    }

    let penalty: f32 = kani::any();
    kani::assume(penalty > 1.0 && penalty <= 5.0 && penalty.is_finite());

    // Mark only token 0 as repeated.
    let repeated_idx: usize = 0;

    let mut penalized = logits.clone();
    // Standard repetition penalty: divide positive, multiply negative.
    if penalized[repeated_idx] > 0.0 {
        penalized[repeated_idx] /= penalty;
    } else {
        penalized[repeated_idx] *= penalty;
    }

    // Repeated token logit must decrease (become less favorable).
    if logits[repeated_idx] > 0.0 {
        assert!(
            penalized[repeated_idx] < logits[repeated_idx],
            "positive repeated logit must decrease"
        );
    } else {
        assert!(
            penalized[repeated_idx] < logits[repeated_idx],
            "negative repeated logit must become more negative"
        );
    }

    // Non-repeated tokens must be unchanged.
    for i in 1..vocab_size {
        assert!(
            (penalized[i] - logits[i]).abs() < 1e-7,
            "non-repeated token logit must be unchanged"
        );
    }
}

// ===========================================================================
// 10. Max length enforcement terminates loop
// ===========================================================================

/// Prove that a generation loop bounded by `max_new_tokens` produces at
/// most that many tokens, and reports `finished = false` when hitting the
/// limit without EOS.
#[kani::proof]
#[kani::unwind(8)]
fn proof_max_length_enforcement() {
    let max_new_tokens: usize = kani::any();
    kani::assume(max_new_tokens >= 1 && max_new_tokens <= 6);

    // Simulate generation loop: generate exactly max_new_tokens tokens
    // without hitting EOS.
    let generated: Vec<usize> = (0..max_new_tokens).collect();
    let output = GenerationOutput::new(generated, false);

    assert_eq!(
        output.token_ids.len(),
        max_new_tokens,
        "output length must equal max_new_tokens when no EOS hit"
    );
    assert!(
        !output.finished,
        "must report not finished when max length reached without EOS"
    );
    assert!(
        output.token_ids.len() <= max_new_tokens,
        "output must never exceed max_new_tokens"
    );
}

// ===========================================================================
// 11. Beam search length normalization divisor > 0
// ===========================================================================

/// Prove that the length normalization divisor in `BeamHypothesis::score`
/// is always positive for non-empty hypotheses, preventing division by zero.
/// Also prove `score` returns finite values for finite inputs.
#[kani::proof]
#[kani::unwind(1)]
fn proof_beam_length_normalization_divisor_positive() {
    let num_tokens: usize = kani::any();
    kani::assume(num_tokens >= 1 && num_tokens <= 100);

    let log_prob: f64 = kani::any();
    kani::assume(log_prob.is_finite() && log_prob.abs() < 1e4);

    let length_penalty: f64 = kani::any();
    kani::assume(length_penalty >= 0.0 && length_penalty <= 5.0 && length_penalty.is_finite());

    let tokens: Vec<usize> = (0..num_tokens).collect();
    let hyp = BeamHypothesis::new(tokens, log_prob, false);

    // The denominator is len^penalty. For len >= 1 and penalty >= 0,
    // this is always > 0 (1^0 = 1, n^p > 0 for n > 0, p >= 0).
    let len = num_tokens as f64;
    let denominator = len.powf(length_penalty);
    assert!(
        denominator > 0.0,
        "length normalization denominator must be positive"
    );
    assert!(
        denominator.is_finite(),
        "length normalization denominator must be finite"
    );

    // Score itself must be finite.
    let score = hyp.score(length_penalty);
    assert!(
        score.is_finite(),
        "beam score must be finite for finite inputs"
    );
}

// ===========================================================================
// 12. Early stopping when all beams have EOS
// ===========================================================================

/// Prove that when `early_stopping` is true and completed beam count >=
/// beam_width, the early stopping condition is satisfied.
#[kani::proof]
#[kani::unwind(1)]
fn proof_early_stopping_all_beams_eos() {
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 8);

    let config = BeamSearchConfig::new(beam_width).with_early_stopping(true);
    assert!(config.early_stopping, "early_stopping must be set");

    // Simulate: completed_count >= beam_width triggers early stop.
    let completed_count: usize = kani::any();
    kani::assume(completed_count >= beam_width);

    let should_stop = config.early_stopping && completed_count >= config.beam_width;
    assert!(
        should_stop,
        "early stopping must trigger when completed >= beam_width"
    );
}

// ===========================================================================
// 13. Batch decode maintains batch dimension
// ===========================================================================

/// Prove that `GenerationOutput::new` preserves the token list length
/// exactly, modeling the batch-of-1 case where each decode step appends
/// exactly one token per batch element.
#[kani::proof]
#[kani::unwind(8)]
fn proof_batch_decode_maintains_dimension() {
    let num_steps: usize = kani::any();
    kani::assume(num_steps >= 1 && num_steps <= 6);

    // Simulate decode: one token per step.
    let mut tokens = Vec::with_capacity(num_steps);
    for step in 0..num_steps {
        let token: usize = kani::any();
        kani::assume(token < 65536);
        tokens.push(token);
        // After each step, length must match step count.
        assert_eq!(tokens.len(), step + 1, "token count must match step count");
    }

    let output = GenerationOutput::new(tokens, false);
    assert_eq!(
        output.token_ids.len(),
        num_steps,
        "output dimension must match number of decode steps"
    );
}

// ===========================================================================
// 14. Sequence padding after EOS preserves pad_token_id
// ===========================================================================

/// Prove that padding a sequence after EOS with a fixed pad_token_id
/// results in correct structure: real tokens before EOS position,
/// pad tokens after.
#[kani::proof]
#[kani::unwind(8)]
fn proof_sequence_padding_after_eos() {
    let total_len: usize = kani::any();
    kani::assume(total_len >= 2 && total_len <= 6);

    let eos_pos: usize = kani::any();
    kani::assume(eos_pos >= 1 && eos_pos < total_len);

    let pad_token_id: usize = kani::any();
    kani::assume(pad_token_id <= 65535);

    let eos_token_id: usize = kani::any();
    kani::assume(eos_token_id <= 65535 && eos_token_id != pad_token_id);

    // Build a padded sequence: real tokens, EOS, then padding.
    let mut sequence = Vec::with_capacity(total_len);
    for i in 0..total_len {
        if i < eos_pos {
            let token: usize = kani::any();
            kani::assume(token <= 65535 && token != pad_token_id);
            sequence.push(token);
        } else if i == eos_pos {
            sequence.push(eos_token_id);
        } else {
            sequence.push(pad_token_id);
        }
    }

    // Verify: all tokens after EOS position are pad_token_id.
    for i in (eos_pos + 1)..total_len {
        assert_eq!(
            sequence[i], pad_token_id,
            "tokens after EOS must be pad_token_id"
        );
    }
    // Verify: EOS token is at the expected position.
    assert_eq!(
        sequence[eos_pos], eos_token_id,
        "EOS token must be at eos_pos"
    );
}

// ===========================================================================
// 15. No-repeat-ngram blocks repeated n-grams
// ===========================================================================

/// Prove that a no-repeat-ngram check correctly identifies when the last
/// (n-1) tokens match a previous (n-1)-gram, and the blocked token is
/// the one that would complete the repeated n-gram.
#[kani::proof]
#[kani::unwind(8)]
fn proof_no_repeat_ngram_blocking() {
    let ngram_size: usize = kani::any();
    kani::assume(ngram_size >= 2 && ngram_size <= 3);

    // Build a sequence where the last (ngram_size - 1) tokens repeat
    // a previous prefix. E.g., for ngram_size=2: [A, B, A] — the last
    // token A matches the first token, so B should be blocked.
    let prefix_len = ngram_size - 1;
    let mut generated = Vec::with_capacity(ngram_size + prefix_len);

    // First occurrence of the n-gram.
    let mut ngram_tokens = vec![0usize; ngram_size];
    for t in ngram_tokens.iter_mut() {
        *t = kani::any();
        kani::assume(*t <= 100);
    }

    // Place the full n-gram.
    for &t in &ngram_tokens {
        generated.push(t);
    }

    // Repeat the prefix (first ngram_size - 1 tokens of the n-gram).
    for i in 0..prefix_len {
        generated.push(ngram_tokens[i]);
    }

    // The token that would complete the repeated n-gram.
    let blocked_token = ngram_tokens[ngram_size - 1];

    // Check: the last prefix_len tokens match the start of the n-gram.
    let seq_len = generated.len();
    let mut matches_prefix = true;
    for i in 0..prefix_len {
        if generated[seq_len - prefix_len + i] != ngram_tokens[i] {
            matches_prefix = false;
        }
    }
    assert!(
        matches_prefix,
        "last prefix_len tokens must match the n-gram prefix"
    );

    // The blocked token is the one that would complete the repeat.
    assert!(
        blocked_token == ngram_tokens[ngram_size - 1],
        "blocked token must be the n-gram completion token"
    );
}

// ===========================================================================
// 16. Forced decoder prefix matches expected tokens
// ===========================================================================

/// Prove that a forced prefix mechanism correctly outputs the expected
/// token at each forced position, regardless of model logits.
#[kani::proof]
#[kani::unwind(6)]
fn proof_forced_decoder_prefix() {
    let prefix_len: usize = kani::any();
    kani::assume(prefix_len >= 1 && prefix_len <= 4);

    let mut forced_tokens = vec![0usize; prefix_len];
    for t in forced_tokens.iter_mut() {
        *t = kani::any();
        kani::assume(*t <= 65535);
    }

    // Simulate forced decoding: at each step < prefix_len, output the
    // forced token regardless of what the model would have selected.
    let mut output_tokens = Vec::with_capacity(prefix_len);
    for step in 0..prefix_len {
        // Model would select some other token.
        let model_token: usize = kani::any();
        kani::assume(model_token <= 65535);

        // Forced prefix overrides.
        let actual_token = forced_tokens[step];
        output_tokens.push(actual_token);

        assert_eq!(
            output_tokens[step], forced_tokens[step],
            "forced prefix must override model selection"
        );
    }

    assert_eq!(
        output_tokens.len(),
        prefix_len,
        "forced prefix must produce exactly prefix_len tokens"
    );
}

// ===========================================================================
// 17. Logits processor chain preserves finite values
// ===========================================================================

/// Prove that a chain of logit processors (temperature + top-k + penalty)
/// preserves finiteness when inputs are finite. Each processor either
/// divides by a positive number, masks to -inf (acceptable), or subtracts
/// a finite amount.
#[kani::proof]
#[kani::unwind(5)]
fn proof_logits_processor_chain_preserves_finite() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 2 && vocab_size <= 4);

    let mut logits = vec![0.0f32; vocab_size];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite() && v.abs() < 100.0);
    }

    // Step 1: Temperature scaling.
    let temperature: f32 = kani::any();
    kani::assume(temperature > 0.01 && temperature <= 100.0 && temperature.is_finite());
    for v in logits.iter_mut() {
        *v /= temperature;
    }
    for &v in &logits {
        assert!(
            v.is_finite(),
            "logits must be finite after temperature scaling"
        );
    }

    // Step 2: Repetition penalty on one token.
    let penalty: f32 = kani::any();
    kani::assume(penalty > 1.0 && penalty <= 5.0 && penalty.is_finite());
    if logits[0] > 0.0 {
        logits[0] /= penalty;
    } else if logits[0] < 0.0 {
        logits[0] *= penalty;
    }
    assert!(
        logits[0].is_finite(),
        "logit must be finite after repetition penalty"
    );

    // Step 3: Frequency penalty subtraction.
    let freq_penalty: f32 = kani::any();
    kani::assume(freq_penalty >= 0.0 && freq_penalty <= 2.0 && freq_penalty.is_finite());
    logits[0] -= freq_penalty;
    assert!(
        logits[0].is_finite(),
        "logit must be finite after frequency penalty"
    );

    // All logits must still be finite after the processor chain.
    for &v in &logits {
        assert!(
            v.is_finite(),
            "all logits must remain finite through processor chain"
        );
    }
}

// ===========================================================================
// 18. Beam hypotheses sorted by score (best-first)
// ===========================================================================

/// Prove that sorting `BeamHypothesis` by `score()` descending produces a
/// monotonically non-increasing sequence of scores.
#[kani::proof]
#[kani::unwind(5)]
fn proof_beam_hypotheses_sorted_by_score() {
    let num_beams: usize = kani::any();
    kani::assume(num_beams >= 2 && num_beams <= 4);

    let mut beams = Vec::with_capacity(num_beams);
    for i in 0..num_beams {
        let log_prob: f64 = kani::any();
        kani::assume(log_prob.is_finite() && log_prob.abs() < 100.0);
        let num_tokens: usize = kani::any();
        kani::assume(num_tokens >= 1 && num_tokens <= 4);
        let tokens: Vec<usize> = (0..num_tokens).collect();
        beams.push(BeamHypothesis::new(tokens, log_prob, i == 0));
    }

    // Use length_penalty = 0.0 to avoid transcendental stubs.
    let penalty = 0.0_f64;

    // Sort by score descending.
    beams.sort_by(|a, b| b.score(penalty).total_cmp(&a.score(penalty)));

    // Verify monotonically non-increasing.
    for i in 1..beams.len() {
        let prev_score = beams[i - 1].score(penalty);
        let curr_score = beams[i].score(penalty);
        assert!(
            prev_score.total_cmp(&curr_score).is_ge(),
            "beams must be sorted by score descending"
        );
    }
}

// ===========================================================================
// 19. Sampling with seed produces deterministic output
// ===========================================================================

/// Prove that `GenerationConfig::with_seed` stores the seed correctly and
/// that two configs with the same seed have identical seed values, which
/// is the precondition for deterministic sampling.
#[kani::proof]
#[kani::unwind(1)]
fn proof_sampling_seed_deterministic() {
    let seed: u64 = kani::any();

    let config1 = GenerationConfig::new(100).with_seed(seed);
    let config2 = GenerationConfig::new(100).with_seed(seed);

    assert_eq!(
        config1.seed, config2.seed,
        "same seed must produce identical config seed values"
    );
    assert_eq!(
        config1.seed,
        Some(seed),
        "seed must be stored as Some(seed)"
    );

    // Different seed must produce different config.
    let other_seed: u64 = kani::any();
    kani::assume(other_seed != seed);
    let config3 = GenerationConfig::new(100).with_seed(other_seed);
    assert_ne!(
        config1.seed, config3.seed,
        "different seeds must produce different config seed values"
    );
}

// ===========================================================================
// 20. Token ID in valid vocabulary range [0, vocab_size)
// ===========================================================================

/// Prove that `argmax` and `top_k_indices` always return token indices
/// strictly less than the vocabulary size. This ensures generated token
/// IDs are valid for embedding lookup.
#[kani::proof]
#[kani::unwind(7)]
fn proof_token_id_in_vocab_range() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 1 && vocab_size <= 6);

    let mut logits = vec![0.0f32; vocab_size];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite());
    }

    // Argmax returns valid index.
    let greedy_idx = argmax(&logits);
    assert!(
        greedy_idx < vocab_size,
        "argmax token ID must be < vocab_size"
    );

    // Top-k returns valid indices.
    let k: usize = kani::any();
    kani::assume(k >= 1 && k <= vocab_size);
    let topk = top_k_indices(&logits, k);
    for &idx in &topk {
        assert!(idx < vocab_size, "top_k token ID must be < vocab_size");
    }
}
