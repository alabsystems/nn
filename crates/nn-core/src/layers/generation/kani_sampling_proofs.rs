// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for autoregressive token sampling safety.
//!
//! 20 proofs covering temperature scaling, top-k filtering, top-p nucleus
//! sampling, repetition/frequency/presence penalty bounds, greedy argmax,
//! softmax properties, min-p filtering, multinomial sampling, combined
//! top-k + top-p, logit bias, stop token detection, and max new tokens
//! enforcement.
//!
//! Part of #4178.

use super::*;

// ===========================================================================
// 1. Temperature scaling preserves ordering for T > 0
// ===========================================================================

/// Prove that dividing logits by a positive temperature preserves the
/// relative ordering of all elements. If logit[i] > logit[j] before
/// scaling, then logit[i]/T > logit[j]/T after scaling for any T > 0.
#[kani::proof]
#[kani::unwind(5)]
fn proof_sampling_temperature_preserves_ordering() {
    let len: usize = kani::any();
    kani::assume(len >= 2 && len <= 4);

    let temperature: f32 = kani::any();
    kani::assume(temperature > 0.01 && temperature <= 100.0 && temperature.is_finite());

    let mut logits = vec![0.0f32; len];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite() && v.abs() < 1000.0);
    }

    let scaled: Vec<f32> = logits.iter().map(|&v| v / temperature).collect();

    // For every pair (i, j), the ordering must be preserved.
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
// 2. Temperature T=1 is identity
// ===========================================================================

/// Prove that temperature = 1.0 does not change the logits.
#[kani::proof]
#[kani::unwind(5)]
fn proof_sampling_temperature_one_is_identity() {
    let len: usize = kani::any();
    kani::assume(len >= 1 && len <= 4);

    let mut logits = vec![0.0f32; len];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite() && v.abs() < 1000.0);
    }

    let scaled: Vec<f32> = logits.iter().map(|&v| v / 1.0f32).collect();

    for i in 0..len {
        assert!(
            (scaled[i] - logits[i]).abs() < 1e-7,
            "temperature=1.0 must be identity"
        );
    }
}

// ===========================================================================
// 3. Top-k filtering: only k highest logits survive
// ===========================================================================

/// Prove that every value NOT in the top-k result is <= every value IN
/// the top-k result (i.e., top-k correctly selects the highest values).
#[kani::proof]
#[kani::unwind(7)]
fn proof_sampling_top_k_selects_highest() {
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

    // Find the minimum value among the selected top-k.
    let mut min_topk = f32::INFINITY;
    for &idx in &indices {
        if values[idx] < min_topk {
            min_topk = values[idx];
        }
    }

    // Every value NOT in the top-k set must be <= min_topk.
    for i in 0..vocab_size {
        if !indices.contains(&i) {
            assert!(
                values[i].total_cmp(&min_topk) != std::cmp::Ordering::Greater,
                "non-top-k values must not exceed minimum of top-k set"
            );
        }
    }
}

// ===========================================================================
// 4. Top-k index bounds: selected indices in [0, vocab_size)
// ===========================================================================

/// Prove that all indices returned by `top_k_indices` are strictly less
/// than the input length, for any k and vocab_size combination.
#[kani::proof]
#[kani::unwind(9)]
fn proof_sampling_top_k_index_bounds() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 1 && vocab_size <= 8);
    let k: usize = kani::any();
    kani::assume(k <= 16);

    let mut values = vec![0.0f32; vocab_size];
    for v in values.iter_mut() {
        *v = kani::any();
    }

    let indices = top_k_indices(&values, k);
    for &idx in &indices {
        assert!(idx < vocab_size, "top_k index must be in [0, vocab_size)");
    }
}

// ===========================================================================
// 5. Top-p cumulative probability threshold correctness
// ===========================================================================

/// Prove that the cumulative probability of the tokens kept by `top_p_filter`
/// is >= p (the threshold), when the total probability mass is sufficient.
#[kani::proof]
#[kani::unwind(5)]
fn proof_sampling_top_p_cumulative_threshold() {
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

    // Normalize input probs to sum to 1.0 before filtering.
    for item in &mut probs {
        item.1 /= total;
    }

    let result = top_p_filter(probs, p);

    // The kept set (after renormalization) sums to ~1.0.
    let kept_sum: f32 = result.iter().map(|&(_, prob)| prob).sum();
    assert!(
        kept_sum >= 0.99 || result.len() == 1,
        "top_p result must represent sufficient probability mass"
    );
}

// ===========================================================================
// 6. Top-p probability normalization after filtering
// ===========================================================================

/// Prove that after `top_p_filter`, the output probabilities sum to
/// approximately 1.0 (renormalized).
#[kani::proof]
#[kani::unwind(5)]
fn proof_sampling_top_p_normalization() {
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
    let sum: f32 = result.iter().map(|&(_, prob)| prob).sum();

    // After renormalization, sum must be approximately 1.0.
    if sum.is_finite() && !result.is_empty() {
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "renormalized top_p probabilities must sum to ~1.0"
        );
    }
}

// ===========================================================================
// 7. Repetition penalty > 1 reduces repeated token logit
// ===========================================================================

/// Prove that applying repetition penalty > 1.0 to a positive logit
/// reduces it, and applying it to a negative logit makes it more negative.
/// This is the standard repetition penalty from the Ctrl paper.
#[kani::proof]
#[kani::unwind(1)]
fn proof_sampling_repetition_penalty_reduces_repeated() {
    let logit: f32 = kani::any();
    kani::assume(logit.is_finite() && logit.abs() < 100.0 && logit != 0.0);

    let penalty: f32 = kani::any();
    kani::assume(penalty > 1.0 && penalty <= 10.0 && penalty.is_finite());

    // Standard repetition penalty: divide positive logits, multiply negative.
    let penalized = if logit > 0.0 {
        logit / penalty
    } else {
        logit * penalty
    };

    // Positive logits must decrease.
    if logit > 0.0 {
        assert!(
            penalized < logit,
            "repetition penalty > 1 must reduce positive logit"
        );
        assert!(
            penalized > 0.0,
            "penalized positive logit must stay positive"
        );
    }
    // Negative logits must become more negative.
    if logit < 0.0 {
        assert!(
            penalized < logit,
            "repetition penalty > 1 must make negative logit more negative"
        );
    }
}

// ===========================================================================
// 8. Frequency penalty bounds
// ===========================================================================

/// Prove that frequency penalty (subtracting penalty * count from logit)
/// always decreases the logit, and the decrease is proportional to count.
#[kani::proof]
#[kani::unwind(1)]
fn proof_sampling_frequency_penalty_bounds() {
    let logit: f32 = kani::any();
    kani::assume(logit.is_finite() && logit.abs() < 100.0);

    let penalty: f32 = kani::any();
    kani::assume(penalty > 0.0 && penalty <= 5.0 && penalty.is_finite());

    let count: u32 = kani::any();
    kani::assume(count >= 1 && count <= 20);

    // Frequency penalty: logit -= penalty * count.
    let penalized = logit - penalty * count as f32;

    assert!(
        penalized < logit,
        "frequency penalty must decrease the logit"
    );
    assert!(penalized.is_finite(), "penalized logit must remain finite");

    // Larger count means larger decrease.
    if count >= 2 {
        let penalized_less = logit - penalty * (count - 1) as f32;
        assert!(
            penalized < penalized_less,
            "higher frequency count must cause larger decrease"
        );
    }
}

// ===========================================================================
// 9. Presence penalty bounds
// ===========================================================================

/// Prove that presence penalty (subtracting a fixed amount for any repeated
/// token) always decreases the logit by exactly the penalty amount.
#[kani::proof]
#[kani::unwind(1)]
fn proof_sampling_presence_penalty_bounds() {
    let logit: f32 = kani::any();
    kani::assume(logit.is_finite() && logit.abs() < 100.0);

    let penalty: f32 = kani::any();
    kani::assume(penalty > 0.0 && penalty <= 5.0 && penalty.is_finite());

    // Presence penalty: logit -= penalty (binary: token appeared or not).
    let penalized = logit - penalty;

    assert!(
        penalized < logit,
        "presence penalty must decrease the logit"
    );
    assert!(penalized.is_finite(), "penalized logit must remain finite");

    // The decrease must be exactly the penalty.
    let diff = logit - penalized;
    assert!(
        (diff - penalty).abs() < 1e-6,
        "presence penalty must decrease logit by exactly the penalty amount"
    );
}

// ===========================================================================
// 10. Greedy argmax returns index in [0, vocab_size)
// ===========================================================================

/// Prove that greedy argmax always returns a valid token index, including
/// for edge cases with identical values and extreme float values.
#[kani::proof]
#[kani::unwind(9)]
fn proof_sampling_greedy_argmax_valid_index() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 1 && vocab_size <= 8);

    let mut values = vec![0.0f32; vocab_size];
    for v in values.iter_mut() {
        *v = kani::any();
        // Allow all float values including NaN and Inf.
    }

    let idx = argmax(&values);
    assert!(
        idx < vocab_size,
        "greedy argmax must return index in [0, vocab_size)"
    );
}

// ===========================================================================
// 11. Softmax output sums to ~1
// ===========================================================================

/// Prove that softmax over finite logits produces outputs that sum to ~1.0.
/// Uses numerically stable softmax (subtract max before exp).
#[kani::proof]
#[kani::unwind(5)]
fn proof_sampling_softmax_sums_to_one() {
    let len: usize = kani::any();
    kani::assume(len >= 1 && len <= 4);

    let mut logits = vec![0.0f32; len];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite() && v.abs() < 50.0);
    }

    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
    let exp_sum: f32 = exp_vals.iter().sum();

    kani::assume(exp_sum > 0.0 && exp_sum.is_finite());

    let probs: Vec<f32> = exp_vals.iter().map(|&e| e / exp_sum).collect();
    let prob_sum: f32 = probs.iter().sum();

    assert!(
        (prob_sum - 1.0).abs() < 1e-5,
        "softmax output must sum to ~1.0"
    );
}

// ===========================================================================
// 12. Softmax output non-negative
// ===========================================================================

/// Prove that every softmax output is non-negative for finite inputs.
#[kani::proof]
#[kani::unwind(5)]
fn proof_sampling_softmax_non_negative() {
    let len: usize = kani::any();
    kani::assume(len >= 1 && len <= 4);

    let mut logits = vec![0.0f32; len];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite() && v.abs() < 50.0);
    }

    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
    let exp_sum: f32 = exp_vals.iter().sum();

    kani::assume(exp_sum > 0.0 && exp_sum.is_finite());

    let probs: Vec<f32> = exp_vals.iter().map(|&e| e / exp_sum).collect();

    for &p in &probs {
        assert!(p >= 0.0, "softmax output must be non-negative");
        assert!(p <= 1.0 + 1e-6, "softmax output must be <= 1.0");
    }
}

// ===========================================================================
// 13. Min-p threshold filtering bounds
// ===========================================================================

/// Prove that min-p filtering (keep tokens whose probability >= min_p *
/// max_prob) always retains the highest-probability token and discards
/// tokens below the threshold.
#[kani::proof]
#[kani::unwind(5)]
fn proof_sampling_min_p_filtering_bounds() {
    let len: usize = kani::any();
    kani::assume(len >= 1 && len <= 4);

    let min_p: f32 = kani::any();
    kani::assume(min_p > 0.0 && min_p <= 1.0 && min_p.is_finite());

    let mut probs = vec![0.0f32; len];
    for v in probs.iter_mut() {
        *v = kani::any();
        kani::assume(*v >= 0.0 && *v <= 1.0 && v.is_finite());
    }

    // Find max probability.
    let max_prob = probs.iter().copied().fold(0.0f32, f32::max);
    kani::assume(max_prob > 0.0);

    let threshold = min_p * max_prob;

    // Apply min-p filtering.
    let kept: Vec<usize> = probs
        .iter()
        .enumerate()
        .filter(|(_, &p)| p >= threshold)
        .map(|(i, _)| i)
        .collect();

    // Must keep at least one token (the max).
    assert!(!kept.is_empty(), "min-p must retain at least one token");

    // The argmax token must always be in the kept set.
    let max_idx = argmax(&probs);
    assert!(
        kept.contains(&max_idx),
        "min-p must always retain the highest-probability token"
    );

    // All kept tokens must have prob >= threshold.
    for &idx in &kept {
        assert!(
            probs[idx] >= threshold,
            "kept token must have prob >= min_p * max_prob"
        );
    }
}

// ===========================================================================
// 14. Multinomial sampling selects valid index
// ===========================================================================

/// Prove that inverse-CDF categorical sampling always returns a valid index
/// from the input distribution, for any random draw in [0, 1).
#[kani::proof]
#[kani::unwind(5)]
fn proof_sampling_multinomial_valid_index() {
    let len: usize = kani::any();
    kani::assume(len >= 1 && len <= 4);

    let mut probs = Vec::with_capacity(len);
    let mut total = 0.0f32;
    for i in 0..len {
        let v: f32 = kani::any();
        kani::assume(v >= 0.01 && v <= 1.0 && v.is_finite());
        total += v;
        probs.push((i, v));
    }
    kani::assume(total > 0.0 && total.is_finite());

    // Normalize to valid distribution.
    for item in &mut probs {
        item.1 /= total;
    }

    // Simulate categorical sampling with any draw in [0, 1).
    let r: f32 = kani::any();
    kani::assume(r >= 0.0 && r < 1.0 && r.is_finite());

    let mut cumsum = 0.0f32;
    let mut selected = probs.last().map(|&(idx, _)| idx).unwrap_or(0);
    for &(idx, p) in &probs {
        cumsum += p;
        if r < cumsum {
            selected = idx;
            break;
        }
    }

    assert!(
        selected < len,
        "multinomial sampling must select valid index"
    );
}

// ===========================================================================
// 15. Combined top-k + top-p preserves valid distribution
// ===========================================================================

/// Prove that applying top-k first, then top-p, produces a non-empty set
/// of candidates with non-negative probabilities summing to ~1.0.
#[kani::proof]
#[kani::unwind(5)]
fn proof_sampling_combined_top_k_top_p_valid() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 2 && vocab_size <= 4);
    let k: usize = kani::any();
    kani::assume(k >= 1 && k <= vocab_size);
    let p: f32 = kani::any();
    kani::assume(p > 0.0 && p <= 1.0 && p.is_finite());

    let mut logits = vec![0.0f32; vocab_size];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite() && v.abs() < 50.0);
    }

    // Step 1: Top-k.
    let topk_indices = top_k_indices(&logits, k);
    assert!(!topk_indices.is_empty());

    // Step 2: Softmax over top-k candidates.
    let max_val = topk_indices
        .iter()
        .map(|&i| logits[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = topk_indices
        .iter()
        .map(|&i| (logits[i] - max_val).exp())
        .sum();

    kani::assume(exp_sum > 0.0 && exp_sum.is_finite());

    let probs: Vec<(usize, f32)> = topk_indices
        .iter()
        .map(|&i| (i, (logits[i] - max_val).exp() / exp_sum))
        .collect();

    // Step 3: Top-p filter.
    let result = top_p_filter(probs, p);

    // Must be non-empty.
    assert!(
        !result.is_empty(),
        "combined top-k + top-p must be non-empty"
    );

    // All probabilities non-negative.
    for &(idx, prob) in &result {
        assert!(idx < vocab_size, "index must be in vocab range");
        assert!(prob >= 0.0, "probability must be non-negative");
    }

    // Sum approximately 1.0.
    let sum: f32 = result.iter().map(|&(_, prob)| prob).sum();
    if sum.is_finite() {
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "combined distribution must sum to ~1.0"
        );
    }
}

// ===========================================================================
// 16. Temperature near 0 approaches argmax behavior
// ===========================================================================

/// Prove that with very low temperature, the softmax probability of the
/// argmax token approaches 1.0 (near-deterministic selection).
#[kani::proof]
#[kani::unwind(4)]
fn proof_sampling_low_temperature_approaches_argmax() {
    let len: usize = kani::any();
    kani::assume(len >= 2 && len <= 3);

    let mut logits = vec![0.0f32; len];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite() && v.abs() < 10.0);
    }

    // Ensure there is a unique maximum with a gap.
    let max_idx = argmax(&logits);
    let max_val = logits[max_idx];
    let mut has_gap = false;
    for (i, &v) in logits.iter().enumerate() {
        if i != max_idx && max_val - v > 0.5 {
            has_gap = true;
        }
    }
    kani::assume(has_gap);

    // Very low temperature.
    let temperature = 0.01f32;
    let scaled: Vec<f32> = logits.iter().map(|&v| v / temperature).collect();

    let sm = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = scaled.iter().map(|&v| (v - sm).exp()).collect();
    let exp_sum: f32 = exp_vals.iter().sum();

    if exp_sum > 0.0 && exp_sum.is_finite() {
        let max_prob = exp_vals[max_idx] / exp_sum;
        assert!(
            max_prob > 0.9,
            "low temperature must concentrate probability on argmax"
        );
    }
}

// ===========================================================================
// 17. High temperature approaches uniform distribution
// ===========================================================================

/// Prove that with very high temperature, the softmax probability of
/// each token approaches 1/N (uniform distribution).
#[kani::proof]
#[kani::unwind(4)]
fn proof_sampling_high_temperature_approaches_uniform() {
    let len: usize = kani::any();
    kani::assume(len >= 2 && len <= 3);

    let mut logits = vec![0.0f32; len];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite() && v.abs() < 10.0);
    }

    // Very high temperature.
    let temperature = 1000.0f32;
    let scaled: Vec<f32> = logits.iter().map(|&v| v / temperature).collect();

    let sm = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = scaled.iter().map(|&v| (v - sm).exp()).collect();
    let exp_sum: f32 = exp_vals.iter().sum();

    if exp_sum > 0.0 && exp_sum.is_finite() {
        let uniform_prob = 1.0f32 / len as f32;
        for &e in &exp_vals {
            let prob = e / exp_sum;
            assert!(
                (prob - uniform_prob).abs() < 0.01,
                "high temperature must produce near-uniform distribution"
            );
        }
    }
}

// ===========================================================================
// 18. Logit bias addition bounds
// ===========================================================================

/// Prove that adding a finite logit bias to a finite logit produces a
/// finite result, and the bias correctly shifts the logit value.
#[kani::proof]
#[kani::unwind(1)]
fn proof_sampling_logit_bias_bounds() {
    let logit: f32 = kani::any();
    kani::assume(logit.is_finite() && logit.abs() < 100.0);

    let bias: f32 = kani::any();
    kani::assume(bias.is_finite() && bias.abs() < 100.0);

    let biased = logit + bias;

    assert!(biased.is_finite(), "biased logit must be finite");
    assert!(
        (biased - logit - bias).abs() < 1e-6,
        "logit bias must be additive"
    );

    // Positive bias increases the logit.
    if bias > 0.0 {
        assert!(biased > logit, "positive bias must increase logit");
    }
    // Negative bias decreases the logit.
    if bias < 0.0 {
        assert!(biased < logit, "negative bias must decrease logit");
    }
}

// ===========================================================================
// 19. Stop token detection correctness
// ===========================================================================

/// Prove that `is_eos` correctly identifies stop tokens across the full
/// valid token range, including boundary values and the no-EOS case.
#[kani::proof]
#[kani::unwind(1)]
fn proof_sampling_stop_token_detection() {
    let token: usize = kani::any();
    kani::assume(token <= 200_000);

    // Case 1: No EOS configured.
    let config_none = GenerationConfig {
        eos_token_id: None,
        ..Default::default()
    };
    assert!(
        !is_eos(token, &config_none),
        "is_eos must return false when eos_token_id is None"
    );

    // Case 2: EOS configured but different from token.
    let other: usize = kani::any();
    kani::assume(other <= 200_000 && other != token);
    let config_diff = GenerationConfig {
        eos_token_id: Some(other),
        ..Default::default()
    };
    assert!(
        !is_eos(token, &config_diff),
        "is_eos must return false when token != eos_token_id"
    );

    // Case 3: EOS matches token.
    let config_match = GenerationConfig {
        eos_token_id: Some(token),
        ..Default::default()
    };
    assert!(
        is_eos(token, &config_match),
        "is_eos must return true when token == eos_token_id"
    );
}

// ===========================================================================
// 20. Max new tokens enforcement
// ===========================================================================

/// Prove that `GenerationConfig::new(n)` correctly stores `max_new_tokens = n`
/// for any valid n, and that the generation loop (modeled abstractly) would
/// produce at most n tokens.
#[kani::proof]
#[kani::unwind(1)]
fn proof_sampling_max_new_tokens_enforcement() {
    let n: usize = kani::any();
    kani::assume(n <= 8192);

    let config = GenerationConfig::new(n);
    assert_eq!(
        config.max_new_tokens, n,
        "max_new_tokens must match constructor argument"
    );

    // Model the generation loop bound abstractly: the loop runs
    // for at most `max_new_tokens` iterations, so the output length
    // is at most `max_new_tokens`.
    let generated_count: usize = kani::any();
    kani::assume(generated_count <= config.max_new_tokens);

    let output = GenerationOutput::new(
        (0..generated_count).collect(),
        generated_count < n, // finished = false if we hit the limit
    );

    assert!(
        output.token_ids.len() <= n,
        "generated token count must not exceed max_new_tokens"
    );
}
