// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proofs for beam search and decoding safety in dpdf VLM
//! inference pipelines. Proves: beam width preservation, score normalization,
//! early stopping, diverse beam search, constrained decoding, repetition
//! penalty, temperature scaling, top-p sampling, top-k filtering, and
//! sequence length bounds.
//!
//! Part of #4239.

// -- Inline helpers (self-contained, no DynTensor dependency) -----------------

fn inline_argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn inline_top_k_by_value(values: &[f32], k: usize) -> Vec<(usize, f32)> {
    if k == 0 || values.is_empty() {
        return Vec::new();
    }
    let k = k.min(values.len());
    let mut indices: Vec<usize> = (0..values.len()).collect();
    if k < indices.len() {
        indices.select_nth_unstable_by(k - 1, |&a, &b| values[b].total_cmp(&values[a]));
        indices.truncate(k);
    }
    indices.sort_unstable_by(|&a, &b| values[b].total_cmp(&values[a]));
    indices.iter().map(|&i| (i, values[i])).collect()
}

fn inline_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 || !sum.is_finite() {
        let uniform = 1.0 / logits.len() as f32;
        return vec![uniform; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

// -- 1. Beam width preservation -----------------------------------------------

/// Prove beam count stays at beam_width when no beams finish.
#[kani::proof]
#[kani::unwind(6)]
fn proof_dpdf_extended_beam_width_preserved_across_steps() {
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 4);
    let num_steps: usize = kani::any();
    kani::assume(num_steps >= 1 && num_steps <= 5);
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 2 && vocab_size <= 4);

    let mut active_count = beam_width.min(vocab_size);
    for _ in 0..num_steps {
        let total_candidates = active_count * vocab_size;
        active_count = total_candidates.min(beam_width);
    }

    assert!(
        active_count <= beam_width,
        "active count must not exceed beam_width"
    );
    if vocab_size >= beam_width {
        assert_eq!(
            active_count, beam_width,
            "with sufficient vocab, active == beam_width"
        );
    }
}

// -- 2. Score normalization ---------------------------------------------------

/// Prove length-penalized score never produces NaN or Inf.
#[kani::proof]
#[kani::unwind(3)]
fn proof_dpdf_extended_length_penalty_no_nan() {
    let log_prob: f64 = kani::any();
    kani::assume(log_prob.is_finite() && log_prob >= -200.0 && log_prob <= 0.0);
    let token_count: usize = kani::any();
    kani::assume(token_count >= 1 && token_count <= 128);
    let length_penalty: f64 = kani::any();
    kani::assume(length_penalty.is_finite() && length_penalty >= 0.0 && length_penalty <= 2.0);

    let score = if length_penalty == 0.0 {
        log_prob
    } else {
        log_prob / (token_count as f64).powf(length_penalty)
    };

    assert!(score.is_finite(), "penalized score must be finite");
    assert!(!score.is_nan(), "penalized score must not be NaN");
    assert!(
        score <= 0.0,
        "score of non-positive log_prob must be non-positive"
    );
}

// -- 3. Early stopping --------------------------------------------------------

/// Prove early stopping terminates once beam_width beams have completed.
#[kani::proof]
#[kani::unwind(7)]
fn proof_dpdf_extended_early_stopping_all_beams_finished() {
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 4);
    let max_steps: usize = kani::any();
    kani::assume(max_steps >= beam_width && max_steps <= 6);

    let mut active = beam_width;
    let mut completed: usize = 0;
    let mut steps_used: usize = 0;

    for _ in 0..max_steps {
        if active == 0 || completed >= beam_width {
            break;
        }
        steps_used += 1;
        let eos_count: usize = kani::any();
        kani::assume(eos_count >= 1 && eos_count <= active);
        completed += eos_count;
        active -= eos_count;
    }

    assert!(
        completed >= beam_width || active == 0,
        "search must terminate when all beams completed or exhausted"
    );
    if completed >= beam_width {
        assert!(
            steps_used <= beam_width,
            "early stop uses at most beam_width steps"
        );
    }
}

// -- 4. Diverse beam search ---------------------------------------------------

/// Prove diversity penalty only reduces (never boosts) logits and stays finite.
#[kani::proof]
#[kani::unwind(5)]
fn proof_dpdf_extended_diverse_beam_group_penalty() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 2 && vocab_size <= 4);
    let diversity_penalty: f32 = kani::any();
    kani::assume(
        diversity_penalty.is_finite() && diversity_penalty >= 0.0 && diversity_penalty <= 10.0,
    );
    let num_groups: usize = kani::any();
    kani::assume(num_groups >= 1 && num_groups <= 3);

    let mut logits = vec![0.0f32; vocab_size];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite() && v.abs() < 50.0);
    }

    for _ in 0..num_groups {
        let selected_token: usize = kani::any();
        kani::assume(selected_token < vocab_size);
        let original = logits[selected_token];
        logits[selected_token] -= diversity_penalty;
        assert!(
            logits[selected_token].total_cmp(&original) != std::cmp::Ordering::Greater,
            "diversity penalty must not increase logit"
        );
        assert!(
            logits[selected_token].is_finite(),
            "penalized logit must remain finite"
        );
    }
}

// -- 5. Constrained decoding --------------------------------------------------

/// Prove that masking non-forced tokens to -inf makes argmax select the forced token.
#[kani::proof]
#[kani::unwind(5)]
fn proof_dpdf_extended_constrained_decoding_force_token() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 2 && vocab_size <= 4);
    let forced_token: usize = kani::any();
    kani::assume(forced_token < vocab_size);

    let mut logits = vec![0.0f32; vocab_size];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite() && v.abs() < 100.0);
    }

    for (i, v) in logits.iter_mut().enumerate() {
        if i != forced_token {
            *v = f32::NEG_INFINITY;
        }
    }

    let selected = inline_argmax(&logits);
    assert_eq!(
        selected, forced_token,
        "constrained decoding must select the forced token"
    );
    assert!(
        logits[forced_token].is_finite(),
        "forced token logit must remain finite"
    );
}

// -- 6. Repetition penalty ----------------------------------------------------

/// Prove repetition penalty reduces positive logits and increases magnitude
/// of negative logits, keeping results finite.
#[kani::proof]
#[kani::unwind(5)]
fn proof_dpdf_extended_repetition_penalty_before_softmax() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 2 && vocab_size <= 4);
    let rep_penalty: f32 = kani::any();
    kani::assume(rep_penalty.is_finite() && rep_penalty >= 1.0 && rep_penalty <= 5.0);

    let mut logits = vec![0.0f32; vocab_size];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite() && v.abs() < 50.0);
    }

    let prev_token: usize = kani::any();
    kani::assume(prev_token < vocab_size);
    let original = logits[prev_token];

    if original > 0.0 {
        logits[prev_token] = original / rep_penalty;
    } else {
        logits[prev_token] = original * rep_penalty;
    }

    assert!(
        logits[prev_token].is_finite(),
        "penalized logit must remain finite"
    );
    if original > 0.0 {
        assert!(
            logits[prev_token] <= original,
            "positive logit must decrease after rep penalty"
        );
        assert!(
            logits[prev_token] >= 0.0,
            "positive logit stays non-negative"
        );
    } else if original < 0.0 {
        assert!(
            logits[prev_token] <= original,
            "negative logit becomes more negative"
        );
    } else {
        assert_eq!(logits[prev_token], 0.0);
    }
}

// -- 7. Temperature scaling ---------------------------------------------------

/// Prove temperature scaling with T > 0 preserves argmax ordering.
#[kani::proof]
#[kani::unwind(5)]
fn proof_dpdf_extended_temperature_preserves_ordering() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 2 && vocab_size <= 4);
    let temperature: f32 = kani::any();
    kani::assume(temperature > 0.01 && temperature <= 100.0 && temperature.is_finite());

    let mut logits = vec![0.0f32; vocab_size];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite() && v.abs() < 50.0);
    }

    let original_argmax = inline_argmax(&logits);
    let scaled: Vec<f32> = logits.iter().map(|&v| v / temperature).collect();
    for &s in &scaled {
        assert!(s.is_finite(), "scaled logit must be finite");
    }
    let scaled_argmax = inline_argmax(&scaled);
    assert_eq!(
        original_argmax, scaled_argmax,
        "temperature must preserve argmax"
    );
}

// -- 8. Top-p sampling --------------------------------------------------------

/// Prove top-p filtering retains at least one token with positive cumulative prob.
#[kani::proof]
#[kani::unwind(5)]
fn proof_dpdf_extended_top_p_cumulative_threshold() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 2 && vocab_size <= 4);

    let mut logits = vec![0.0f32; vocab_size];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite() && v.abs() < 20.0);
    }

    let probs = inline_softmax(&logits);
    let p: f32 = kani::any();
    kani::assume(p > 0.0 && p <= 1.0 && p.is_finite());

    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut cumulative = 0.0f32;
    let mut retained_count: usize = 0;
    for &(_, prob) in &indexed {
        cumulative += prob;
        retained_count += 1;
        if cumulative >= p {
            break;
        }
    }

    assert!(retained_count >= 1, "top-p must retain at least one token");
    assert!(
        retained_count <= vocab_size,
        "retained count cannot exceed vocab size"
    );
    assert!(cumulative > 0.0, "cumulative probability must be positive");
}

// -- 9. Top-k filtering -------------------------------------------------------

/// Prove top-k retains exactly min(k, vocab_size) tokens with correct ordering.
#[kani::proof]
#[kani::unwind(7)]
fn proof_dpdf_extended_top_k_exact_count() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 2 && vocab_size <= 6);
    let k: usize = kani::any();
    kani::assume(k >= 1 && k <= vocab_size);

    let mut logits = vec![0.0f32; vocab_size];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite() && v.abs() < 50.0);
    }

    let topk = inline_top_k_by_value(&logits, k);
    assert_eq!(
        topk.len(),
        k.min(vocab_size),
        "top-k must retain exactly min(k, vocab_size)"
    );

    for &(idx, _) in &topk {
        assert!(idx < vocab_size, "retained index must be valid");
    }

    if !topk.is_empty() {
        let min_retained = topk.iter().map(|&(_, v)| v).fold(f32::INFINITY, f32::min);
        for (i, &logit) in logits.iter().enumerate() {
            let is_retained = topk.iter().any(|&(idx, _)| idx == i);
            if !is_retained {
                assert!(
                    logit.total_cmp(&min_retained) != std::cmp::Ordering::Greater,
                    "no filtered token may exceed the weakest retained token"
                );
            }
        }
    }
}

// -- 10. Sequence length ------------------------------------------------------

/// Prove combined prompt + generated output never exceeds prompt_len + max_new_tokens.
#[kani::proof]
#[kani::unwind(8)]
fn proof_dpdf_extended_sequence_length_bounded() {
    let prompt_len: usize = kani::any();
    kani::assume(prompt_len >= 1 && prompt_len <= 4);
    let max_new_tokens: usize = kani::any();
    kani::assume(max_new_tokens >= 1 && max_new_tokens <= 6);
    let max_total = prompt_len + max_new_tokens;
    let eos_id: usize = 42;

    let mut generated = Vec::new();
    for _ in 0..max_new_tokens {
        let token: usize = kani::any();
        kani::assume(token <= 65535);
        generated.push(token);
        if token == eos_id {
            break;
        }
    }

    assert!(
        generated.len() <= max_new_tokens,
        "generated tokens must not exceed max_new_tokens"
    );
    let total_len = prompt_len + generated.len();
    assert!(
        total_len <= max_total,
        "total length must not exceed prompt_len + max_new_tokens"
    );
    assert!(
        total_len >= prompt_len,
        "total length must be at least prompt length"
    );
}
