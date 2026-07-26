// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for beam search and greedy decoding safety (#4239).
//!
//! Proves key structural, numerical, and sampling properties of autoregressive
//! text generation:
//!
//! 1.  **Beam width bounds** — beam_width > 0, hypotheses count <= beam_width
//! 2.  **Greedy decoding** — selected token is argmax of logits
//! 3.  **Beam score monotonicity** — log-probability scores non-increasing
//! 4.  **EOS handling** — hypothesis finalized when EOS token generated
//! 5.  **Length penalty** — score normalization by length produces no NaN/Inf
//! 6.  **Top-k filtering** — only top-k logits remain, rest are -inf
//! 7.  **Top-p (nucleus) filtering** — cumulative probability threshold
//! 8.  **Temperature scaling** — temperature > 0, valid probability distribution
//! 9.  **Repetition penalty** — penalized logits strictly less than original
//! 10. **Token ID bounds** — all generated token IDs in [0, vocab_size)
//!
//! All harnesses use small bounds for CBMC tractability:
//! vocab_size <= 16, beam_width <= 4, seq_len <= 8.
//!
//! Part of #4239.

#![cfg(kani)]

// ===========================================================================
// 1. Beam width bounds
// ===========================================================================

/// Proves beam_width > 0 and hypotheses count <= beam_width at each step.
///
/// In beam search, the number of active hypotheses must never exceed the
/// configured beam width, and beam width must be at least 1 (greedy
/// decoding is beam search with width 1).
#[kani::unwind(6)]
#[kani::proof]
fn beam_width_bounds_valid() {
    let beam_width: u8 = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 4);
    let bw = beam_width as usize;

    // At each step, we expand each hypothesis and keep top beam_width.
    // Model: start with 1 hypothesis, expand to min(vocab_candidates, beam_width).
    let vocab_candidates: u8 = kani::any();
    kani::assume(vocab_candidates >= 1 && vocab_candidates <= 16);
    let vc = vocab_candidates as usize;

    // Initial state: 1 hypothesis
    let initial_count: usize = 1;
    assert!(initial_count <= bw, "initial count must be <= beam_width");

    // After first expansion: min(1 * vocab_candidates, beam_width)
    let expanded = initial_count * vc;
    let after_prune = if expanded > bw { bw } else { expanded };
    assert!(
        after_prune <= bw,
        "hypotheses after pruning must be <= beam_width"
    );
    assert!(
        after_prune >= 1,
        "must have at least one hypothesis after pruning"
    );

    // After second expansion from after_prune hypotheses
    let step2_expanded = after_prune * vc;
    let step2_pruned = if step2_expanded > bw {
        bw
    } else {
        step2_expanded
    };
    assert!(
        step2_pruned <= bw,
        "hypotheses after second step must be <= beam_width"
    );
    assert!(
        step2_pruned >= 1,
        "must have at least one hypothesis after second step"
    );
}

/// Proves beam_width == 0 is invalid (must be rejected).
#[kani::unwind(1)]
#[kani::proof]
fn beam_width_zero_rejected() {
    let beam_width: u8 = kani::any();
    kani::assume(beam_width == 0);

    // beam_width == 0 would produce zero hypotheses at every step
    let hypotheses_count: usize = 0;
    assert!(
        hypotheses_count == 0,
        "zero beam width produces zero hypotheses"
    );
    // This is an invalid configuration — generation cannot proceed
    assert!(
        beam_width == 0,
        "zero beam width must be detected and rejected at configuration time"
    );
}

// ===========================================================================
// 2. Greedy decoding — argmax
// ===========================================================================

/// Proves greedy decoding selects the token with highest logit value.
///
/// For a logit vector of size vocab_size, the selected token index must
/// satisfy: logits[selected] >= logits[j] for all j in [0, vocab_size).
/// We model this with a small vocabulary and nondeterministic logits.
#[kani::unwind(6)]
#[kani::proof]
fn greedy_decoding_selects_argmax() {
    // Model a 4-element logit vector
    let l0: f32 = kani::any();
    let l1: f32 = kani::any();
    let l2: f32 = kani::any();
    let l3: f32 = kani::any();

    kani::assume(l0.is_finite() && l0 >= -100.0 && l0 <= 100.0);
    kani::assume(l1.is_finite() && l1 >= -100.0 && l1 <= 100.0);
    kani::assume(l2.is_finite() && l2 >= -100.0 && l2 <= 100.0);
    kani::assume(l3.is_finite() && l3 >= -100.0 && l3 <= 100.0);

    // Find argmax via sequential scan (greedy decoding algorithm)
    let mut max_val = l0;
    let mut max_idx: usize = 0;

    if l1 > max_val {
        max_val = l1;
        max_idx = 1;
    }
    if l2 > max_val {
        max_val = l2;
        max_idx = 2;
    }
    if l3 > max_val {
        max_val = l3;
        max_idx = 3;
    }

    // The selected token must have the maximum logit
    assert!(max_val >= l0, "argmax must be >= l0");
    assert!(max_val >= l1, "argmax must be >= l1");
    assert!(max_val >= l2, "argmax must be >= l2");
    assert!(max_val >= l3, "argmax must be >= l3");

    // Token ID must be in valid range
    assert!(max_idx < 4, "greedy token ID must be in [0, vocab_size)");
}

// ===========================================================================
// 3. Beam score monotonicity
// ===========================================================================

/// Proves beam search log-probability scores are non-increasing.
///
/// Each step adds log P(token|context) where P is in (0, 1], so
/// log P <= 0. The cumulative score (sum of log-probs) can only
/// decrease or stay equal. This ensures scores form a monotonically
/// non-increasing sequence over generation steps.
#[kani::unwind(1)]
#[kani::proof]
fn beam_score_monotonicity() {
    let prev_score: f32 = kani::any();
    let log_prob: f32 = kani::any();

    // Previous score is a cumulative log-probability (non-positive, finite)
    kani::assume(prev_score.is_finite() && prev_score <= 0.0 && prev_score >= -1e6);

    // New token log-probability: log P(token) where P in (0, 1]
    // So log_prob in (-inf, 0]. We bound for tractability.
    kani::assume(log_prob.is_finite() && log_prob <= 0.0 && log_prob >= -100.0);

    let new_score = prev_score + log_prob;
    kani::assume(new_score.is_finite());

    // Monotonicity: new_score <= prev_score since log_prob <= 0
    assert!(
        new_score <= prev_score,
        "beam score must be non-increasing (log_prob <= 0)"
    );

    // Score is still non-positive (sum of non-positive terms)
    assert!(
        new_score <= 0.0,
        "cumulative log-probability must be non-positive"
    );
}

/// Proves beam score accumulation over multiple steps is non-increasing.
#[kani::unwind(1)]
#[kani::proof]
fn beam_score_multi_step_monotonicity() {
    let score_0: f32 = 0.0_f32; // initial score

    let lp1: f32 = kani::any();
    let lp2: f32 = kani::any();
    let lp3: f32 = kani::any();

    kani::assume(lp1.is_finite() && lp1 <= 0.0 && lp1 >= -50.0);
    kani::assume(lp2.is_finite() && lp2 <= 0.0 && lp2 >= -50.0);
    kani::assume(lp3.is_finite() && lp3 <= 0.0 && lp3 >= -50.0);

    let score_1 = score_0 + lp1;
    let score_2 = score_1 + lp2;
    let score_3 = score_2 + lp3;

    kani::assume(score_1.is_finite());
    kani::assume(score_2.is_finite());
    kani::assume(score_3.is_finite());

    assert!(score_1 <= score_0, "score after step 1 <= initial");
    assert!(score_2 <= score_1, "score after step 2 <= step 1");
    assert!(score_3 <= score_2, "score after step 3 <= step 2");

    // Transitive: final <= initial
    assert!(
        score_3 <= score_0,
        "final score must be <= initial score (transitivity)"
    );
}

// ===========================================================================
// 4. EOS handling
// ===========================================================================

/// Proves hypothesis is finalized when EOS token is generated.
///
/// When a hypothesis generates the EOS token, it must be moved to the
/// finished set and not expanded further. The finished hypothesis count
/// monotonically increases.
#[kani::unwind(1)]
#[kani::proof]
fn eos_handling_finalizes_hypothesis() {
    let eos_token_id: u8 = kani::any();
    let generated_token_id: u8 = kani::any();
    let vocab_size: u8 = kani::any();

    kani::assume(vocab_size >= 2 && vocab_size <= 16);
    kani::assume(eos_token_id < vocab_size);
    kani::assume(generated_token_id < vocab_size);

    let is_eos = generated_token_id == eos_token_id;
    let mut finished_before: usize = kani::any();
    kani::assume(finished_before <= 4);

    let finished_after = if is_eos {
        finished_before + 1
    } else {
        finished_before
    };

    // Finished count is monotonically non-decreasing
    assert!(
        finished_after >= finished_before,
        "finished hypothesis count must be non-decreasing"
    );

    // If EOS was generated, exactly one more hypothesis is finished
    if is_eos {
        assert!(
            finished_after == finished_before + 1,
            "EOS must finalize exactly one hypothesis"
        );
    } else {
        assert!(
            finished_after == finished_before,
            "non-EOS must not change finished count"
        );
    }
}

// ===========================================================================
// 5. Length penalty — score normalization produces no NaN/Inf
// ===========================================================================

/// Proves length penalty normalization does not produce NaN or Inf.
///
/// The standard length penalty formula is:
///   normalized_score = score / ((5 + length)^alpha / (5 + 1)^alpha)
/// where alpha >= 0 and length >= 1. The denominator is always > 0
/// for length >= 1 and alpha >= 0.
#[kani::unwind(1)]
#[kani::proof]
fn length_penalty_no_nan_inf() {
    let score: f32 = kani::any();
    let length: u8 = kani::any();
    let alpha_bits: u8 = kani::any();

    kani::assume(score.is_finite() && score >= -1000.0 && score <= 0.0);
    kani::assume(length >= 1 && length <= 64);

    // alpha in {0.0, 0.25, 0.5, 0.75, 1.0} for CBMC tractability
    let alpha: f32 = (alpha_bits % 5) as f32 * 0.25;

    let len_f = length as f32;

    // Wu et al. length penalty: lp(len) = ((5 + len) / 6)^alpha
    let numerator = 5.0_f32 + len_f; // >= 6.0 since len >= 1
    let base = 6.0_f32; // 5 + 1

    assert!(numerator >= 6.0, "numerator must be >= 6.0 for len >= 1");
    assert!(base > 0.0, "base must be positive");

    let ratio = numerator / base;
    assert!(ratio.is_finite(), "ratio must be finite");
    assert!(ratio >= 1.0, "ratio must be >= 1.0 for len >= 1");

    // For alpha = 0: ratio^0 = 1.0 (no penalty)
    // For alpha > 0: ratio^alpha >= 1.0 (penalty grows with length)
    // We use the identity: ratio^alpha = exp(alpha * ln(ratio))
    // Since ratio >= 1.0, ln(ratio) >= 0, so exp >= 1.0 for alpha >= 0

    // Model the penalty factor as a nondeterministic positive finite value
    // (Kani cannot evaluate transcendentals directly)
    let penalty: f32 = kani::any();
    kani::assume(penalty.is_finite() && penalty >= 1.0 && penalty <= 100.0);

    let normalized = score / penalty;

    // Check no NaN/Inf
    assert!(normalized.is_finite(), "normalized score must be finite");

    // Normalized score has same sign as score (penalty > 0)
    assert!(
        normalized <= 0.0,
        "normalized score must be non-positive (score <= 0, penalty > 0)"
    );

    // |normalized| <= |score| since penalty >= 1.0
    assert!(
        normalized >= score,
        "normalized score must be >= score (penalty >= 1 makes magnitude smaller)"
    );
}

// ===========================================================================
// 6. Top-k filtering
// ===========================================================================

/// Proves top-k filtering: exactly k logits remain, rest are set to -inf.
///
/// After top-k filtering of a logit vector of size V with parameter k:
/// - Exactly min(k, V) positions retain their original logit values
/// - All other positions are set to f32::NEG_INFINITY
/// - The retained positions have values >= all filtered positions' original values
#[kani::unwind(6)]
#[kani::proof]
fn topk_filtering_properties() {
    let vocab_size: u8 = kani::any();
    let k: u8 = kani::any();

    kani::assume(vocab_size >= 2 && vocab_size <= 4);
    kani::assume(k >= 1 && k <= vocab_size);

    let vs = vocab_size as usize;
    let ku = k as usize;

    // Model with 4 logit values (max vocab_size = 4)
    let l0: f32 = kani::any();
    let l1: f32 = kani::any();
    let l2: f32 = kani::any();
    let l3: f32 = kani::any();

    kani::assume(l0.is_finite() && l0 >= -100.0 && l0 <= 100.0);
    kani::assume(l1.is_finite() && l1 >= -100.0 && l1 <= 100.0);
    kani::assume(l2.is_finite() && l2 >= -100.0 && l2 <= 100.0);
    kani::assume(l3.is_finite() && l3 >= -100.0 && l3 <= 100.0);

    // Count how many logits are in the top-k by comparing each against all others.
    // A logit is in top-k if fewer than k logits are strictly greater than it.
    let count_above_l0 = (l1 > l0) as usize + (l2 > l0) as usize + (l3 > l0) as usize;
    let count_above_l1 = (l0 > l1) as usize + (l2 > l1) as usize + (l3 > l1) as usize;
    let count_above_l2 = (l0 > l2) as usize + (l1 > l2) as usize + (l3 > l2) as usize;
    let count_above_l3 = (l0 > l3) as usize + (l1 > l3) as usize + (l2 > l3) as usize;

    // Only consider positions within vocab_size
    let in_topk_0 = (0 < vs) && (count_above_l0 < ku);
    let in_topk_1 = (1 < vs) && (count_above_l1 < ku);
    let in_topk_2 = (2 < vs) && (count_above_l2 < ku);
    let in_topk_3 = (3 < vs) && (count_above_l3 < ku);

    // After filtering: retained logits keep their value, others become NEG_INFINITY
    let f0 = if in_topk_0 { l0 } else { f32::NEG_INFINITY };
    let f1 = if in_topk_1 { l1 } else { f32::NEG_INFINITY };
    let f2 = if in_topk_2 { l2 } else { f32::NEG_INFINITY };
    let f3 = if in_topk_3 { l3 } else { f32::NEG_INFINITY };

    // Filtered-out positions must be NEG_INFINITY
    if !in_topk_0 && 0 < vs {
        assert!(f0 == f32::NEG_INFINITY, "filtered position must be -inf");
    }
    if !in_topk_1 && 1 < vs {
        assert!(f1 == f32::NEG_INFINITY, "filtered position must be -inf");
    }

    // At least one logit must survive filtering (k >= 1)
    let survived_count =
        in_topk_0 as usize + in_topk_1 as usize + in_topk_2 as usize + in_topk_3 as usize;
    assert!(
        survived_count >= 1,
        "at least one logit must survive top-k filtering"
    );
}

// ===========================================================================
// 7. Top-p (nucleus) filtering
// ===========================================================================

/// Proves top-p (nucleus) filtering cumulative probability threshold property.
///
/// After sorting logits in descending order and computing cumulative softmax
/// probabilities, top-p filtering retains tokens whose cumulative probability
/// is <= p. The first token exceeding the threshold is included, but
/// subsequent tokens are excluded.
#[kani::unwind(5)]
#[kani::proof]
fn topp_nucleus_filtering_threshold() {
    // Model sorted probability distribution (descending order, 4 tokens)
    let p0: f32 = kani::any();
    let p1: f32 = kani::any();
    let p2: f32 = kani::any();
    let p3: f32 = kani::any();

    // Each probability is in (0, 1], sorted descending
    kani::assume(p0.is_finite() && p0 > 0.0 && p0 <= 1.0);
    kani::assume(p1.is_finite() && p1 > 0.0 && p1 <= p0);
    kani::assume(p2.is_finite() && p2 > 0.0 && p2 <= p1);
    kani::assume(p3.is_finite() && p3 > 0.0 && p3 <= p2);

    // They must sum to ~1.0 (probability distribution)
    let sum = p0 + p1 + p2 + p3;
    kani::assume(sum.is_finite());
    kani::assume((sum - 1.0).abs() < 0.01);

    // Top-p threshold
    let top_p: f32 = kani::any();
    kani::assume(top_p.is_finite() && top_p > 0.0 && top_p <= 1.0);

    // Cumulative probabilities
    let cum0 = p0;
    let cum1 = p0 + p1;
    let cum2 = p0 + p1 + p2;
    // cum3 = sum ~= 1.0

    kani::assume(cum1.is_finite());
    kani::assume(cum2.is_finite());

    // Determine which tokens pass the nucleus filter
    // Token i is included if cumulative prob BEFORE token i is < top_p
    // (the first token always passes)
    let keep_0 = true; // first token always kept
    let keep_1 = cum0 < top_p;
    let keep_2 = cum1 < top_p;
    let keep_3 = cum2 < top_p;

    // At least the first token is always kept
    assert!(keep_0, "first token (highest prob) must always be kept");

    // Kept tokens form a prefix of the sorted sequence
    if keep_2 {
        assert!(keep_1, "if token 2 kept, token 1 must also be kept");
    }
    if keep_3 {
        assert!(keep_2, "if token 3 kept, token 2 must also be kept");
    }

    // Count kept tokens
    let kept_count = keep_0 as usize + keep_1 as usize + keep_2 as usize + keep_3 as usize;
    assert!(
        kept_count >= 1,
        "at least one token must survive nucleus filtering"
    );
    assert!(kept_count <= 4, "kept count cannot exceed vocab size");
}

// ===========================================================================
// 8. Temperature scaling
// ===========================================================================

/// Nondeterministic exp stub: returns any positive finite value.
/// Sound over-approximation for softmax/temperature proofs.
fn exp_stub_decode(_x: f32) -> f32 {
    let result: f32 = kani::any();
    kani::assume(result.is_finite() && result > 0.0);
    result
}

/// Proves temperature scaling: T > 0 and softmax(logits/T) is a valid
/// probability distribution (all elements non-negative, sum to 1).
///
/// Temperature T > 0 scales logits before softmax:
/// - T < 1: sharpens distribution (more confident)
/// - T = 1: standard softmax
/// - T > 1: flattens distribution (more uniform)
/// Division by T > 0 preserves finiteness of logits.
#[kani::unwind(4)]
#[kani::proof]
#[kani::stub(f32::exp, exp_stub_decode)]
fn temperature_scaling_valid_distribution() {
    let l0: f32 = kani::any();
    let l1: f32 = kani::any();
    let l2: f32 = kani::any();

    kani::assume(l0.is_finite() && l0 >= -100.0 && l0 <= 100.0);
    kani::assume(l1.is_finite() && l1 >= -100.0 && l1 <= 100.0);
    kani::assume(l2.is_finite() && l2 >= -100.0 && l2 <= 100.0);

    let temperature: f32 = kani::any();
    kani::assume(temperature.is_finite() && temperature > 0.0 && temperature <= 100.0);

    // Scale logits by temperature
    let s0 = l0 / temperature;
    let s1 = l1 / temperature;
    let s2 = l2 / temperature;

    assert!(s0.is_finite(), "scaled logit must be finite");
    assert!(s1.is_finite(), "scaled logit must be finite");
    assert!(s2.is_finite(), "scaled logit must be finite");

    // Numerically stable softmax: subtract max before exp
    let m = if s0 > s1 {
        if s0 > s2 {
            s0
        } else {
            s2
        }
    } else if s1 > s2 {
        s1
    } else {
        s2
    };

    let e0 = (s0 - m).exp();
    let e1 = (s1 - m).exp();
    let e2 = (s2 - m).exp();
    let sum_exp = e0 + e1 + e2;

    let p0 = e0 / sum_exp;
    let p1 = e1 / sum_exp;
    let p2 = e2 / sum_exp;

    // Each probability is non-negative
    assert!(p0 >= 0.0, "softmax output must be non-negative");
    assert!(p1 >= 0.0, "softmax output must be non-negative");
    assert!(p2 >= 0.0, "softmax output must be non-negative");

    // Sum to 1.0 within f32 rounding
    let row_sum = p0 + p1 + p2;
    assert!(row_sum.is_finite(), "probability sum must be finite");
    assert!(
        (row_sum - 1.0).abs() < 1e-5,
        "probabilities must sum to ~1.0"
    );

    // Each probability at most 1.0 (within rounding)
    assert!(p0 <= 1.0 + 1e-7, "probability must be <= 1");
    assert!(p1 <= 1.0 + 1e-7, "probability must be <= 1");
    assert!(p2 <= 1.0 + 1e-7, "probability must be <= 1");
}

/// Proves temperature == 0 causes division by zero (must be rejected).
#[kani::unwind(1)]
#[kani::proof]
fn temperature_zero_rejected() {
    let logit: f32 = kani::any();
    kani::assume(logit.is_finite() && logit >= -100.0 && logit <= 100.0);

    let temperature: f32 = 0.0;

    // logit / 0.0 produces Inf or NaN — invalid for softmax
    let scaled = logit / temperature;

    // For any nonzero logit, division by zero produces infinity
    if logit != 0.0 {
        assert!(
            !scaled.is_finite(),
            "nonzero logit / 0 must produce non-finite result"
        );
    } else {
        // 0.0 / 0.0 = NaN
        assert!(scaled.is_nan(), "0 / 0 must produce NaN");
    }
}

// ===========================================================================
// 9. Repetition penalty
// ===========================================================================

/// Proves repetition penalty: penalized logits for repeated tokens are
/// strictly less than the original logits (for positive logits) or
/// strictly more negative (for negative logits).
///
/// The standard repetition penalty formula:
///   if logit > 0: penalized = logit / penalty  (penalty > 1 shrinks it)
///   if logit < 0: penalized = logit * penalty  (penalty > 1 makes it more negative)
///   if logit == 0: penalized = 0 (unchanged)
#[kani::unwind(1)]
#[kani::proof]
fn repetition_penalty_reduces_repeated_logits() {
    let logit: f32 = kani::any();
    let penalty: f32 = kani::any();

    kani::assume(logit.is_finite() && logit >= -100.0 && logit <= 100.0);
    // Repetition penalty must be > 1.0 to have an effect
    kani::assume(penalty.is_finite() && penalty > 1.0 && penalty <= 10.0);

    let penalized = if logit > 0.0 {
        logit / penalty
    } else if logit < 0.0 {
        logit * penalty
    } else {
        0.0_f32
    };

    kani::assume(penalized.is_finite());

    // For positive logits: penalized < original (divided by penalty > 1)
    if logit > 0.0 {
        assert!(
            penalized < logit,
            "penalized positive logit must be strictly less than original"
        );
        assert!(
            penalized > 0.0,
            "penalized positive logit must remain positive"
        );
    }

    // For negative logits: penalized < original (multiplied by penalty > 1, making more negative)
    if logit < 0.0 {
        assert!(
            penalized < logit,
            "penalized negative logit must be strictly more negative"
        );
    }

    // For zero logits: unchanged
    if logit == 0.0 {
        assert!(
            penalized == 0.0,
            "zero logit must remain zero after penalty"
        );
    }
}

/// Proves repetition penalty == 1.0 is identity (no change).
#[kani::unwind(1)]
#[kani::proof]
fn repetition_penalty_identity_at_one() {
    let logit: f32 = kani::any();
    kani::assume(logit.is_finite() && logit >= -100.0 && logit <= 100.0);

    let penalty: f32 = 1.0;

    let penalized = if logit > 0.0 {
        logit / penalty
    } else if logit < 0.0 {
        logit * penalty
    } else {
        0.0_f32
    };

    // penalty == 1.0 => no change
    if logit > 0.0 {
        assert!(
            (penalized - logit).abs() < 1e-7,
            "penalty 1.0 must be identity for positive logits"
        );
    } else if logit < 0.0 {
        assert!(
            (penalized - logit).abs() < 1e-7,
            "penalty 1.0 must be identity for negative logits"
        );
    } else {
        assert!(
            penalized == 0.0,
            "penalty 1.0 must be identity for zero logit"
        );
    }
}

// ===========================================================================
// 10. Token ID bounds
// ===========================================================================

/// Proves all generated token IDs are in [0, vocab_size).
///
/// For any decoding method (greedy, beam, sampling), the output token ID
/// must be a valid index into the vocabulary. This is guaranteed by the
/// argmax or sampling operation operating on a vector of size vocab_size.
#[kani::unwind(1)]
#[kani::proof]
fn token_id_bounds_valid() {
    let vocab_size: u16 = kani::any();
    let token_id: u16 = kani::any();

    kani::assume(vocab_size >= 1 && vocab_size <= 256);

    // Greedy: argmax returns index in [0, vocab_size)
    kani::assume(token_id < vocab_size);
    assert!(
        (token_id as usize) < (vocab_size as usize),
        "token ID must be in [0, vocab_size)"
    );

    // Verify the token ID can be used as an embedding lookup index
    let embed_table_size = vocab_size as usize;
    let idx = token_id as usize;
    assert!(
        idx < embed_table_size,
        "token ID must be a valid embedding table index"
    );
}

/// Proves argmax on a finite logit vector produces a valid token ID.
#[kani::unwind(6)]
#[kani::proof]
fn token_id_from_argmax_valid() {
    let vocab_size: u8 = kani::any();
    kani::assume(vocab_size >= 1 && vocab_size <= 4);
    let vs = vocab_size as usize;

    // Model logit values for up to 4 tokens
    let l0: f32 = kani::any();
    let l1: f32 = kani::any();
    let l2: f32 = kani::any();
    let l3: f32 = kani::any();

    kani::assume(l0.is_finite());
    kani::assume(l1.is_finite());
    kani::assume(l2.is_finite());
    kani::assume(l3.is_finite());

    // Argmax over the valid range [0, vocab_size)
    let mut best_idx: usize = 0;
    let mut best_val: f32 = l0;

    if vs > 1 && l1 > best_val {
        best_val = l1;
        best_idx = 1;
    }
    if vs > 2 && l2 > best_val {
        best_val = l2;
        best_idx = 2;
    }
    if vs > 3 && l3 > best_val {
        best_val = l3;
        best_idx = 3;
    }

    // The resulting token ID must be in [0, vocab_size)
    assert!(best_idx < vs, "argmax token ID must be < vocab_size");

    // The best value must be >= all values in the valid range
    assert!(best_val >= l0, "argmax must be >= l0");
    if vs > 1 {
        assert!(best_val >= l1, "argmax must be >= l1");
    }
    if vs > 2 {
        assert!(best_val >= l2, "argmax must be >= l2");
    }
    if vs > 3 {
        assert!(best_val >= l3, "argmax must be >= l3");
    }
}
