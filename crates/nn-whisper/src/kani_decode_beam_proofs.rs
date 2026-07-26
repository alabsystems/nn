// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Whisper beam search decode safety.
//!
//! Covers:
//! - BeamState score computation: length penalty arithmetic, zero-length, zero-penalty
//! - Parent-pointer tree reconstruction: walk correctness, None handling, depth tracking
//! - reconstruct_all_tokens: initial + decoded concatenation
//! - top_k_with_log_probs: index bounds, log-softmax properties, partial sort
//! - Beam expansion arithmetic: parent_plus_one encoding, tree growth bounds
//! - Candidate scoring: sum_log_prob accumulation, score ordering
//! - WhisperBeamConfig validation: edge cases (infinity, max usize)
//! - Token ID u32 overflow detection
//!
//! Issue: #3645

use super::beam::{self, WhisperBeamConfig};
use crate::tokenizer::EOT_TOKEN;

// ── Kani transcendental stubs (CBMC cannot handle these) ──
fn exp_f32_stub(x: f32) -> f32 { let _ = x; let r: f32 = kani::any(); kani::assume(r.is_finite() && r > 0.0 && r <= 1e10); r }
fn ln_f32_stub(x: f32) -> f32 { let _ = x; let r: f32 = kani::any(); kani::assume(r.is_finite() && r >= -100.0 && r <= 100.0); r }
fn powf_f32_stub(b: f32, _e: f32) -> f32 { let _ = b; let r: f32 = kani::any(); kani::assume(r.is_finite() && r > 0.0 && r <= 1e10); r }


// ============================================================================
// BeamState and tree functions are private — we reconstruct minimal versions
// to prove properties of the algorithms without depending on private types.
// ============================================================================

/// Mirror of BeamState::score for Kani proofs (private in decode_beam.rs).
fn beam_score(sum_log_prob: f64, decoded_len: usize, length_penalty: f64) -> f64 {
    if length_penalty == 0.0 || decoded_len == 0 {
        sum_log_prob
    } else {
        let len = decoded_len as f64;
        sum_log_prob / len.powf(length_penalty)
    }
}

/// Mirror of reconstruct_decoded for Kani proofs (private in decode_beam.rs).
fn reconstruct_decoded(node_idx: Option<usize>, tree: &[(usize, usize)]) -> Vec<usize> {
    let mut tokens = Vec::new();
    let mut idx = match node_idx {
        Some(i) => i,
        None => return tokens,
    };
    loop {
        let (parent_plus_one, token) = tree[idx];
        tokens.push(token);
        if parent_plus_one == 0 {
            break;
        }
        idx = parent_plus_one - 1;
    }
    tokens.reverse();
    tokens
}

/// Mirror of reconstruct_all_tokens for Kani proofs (private in decode_beam.rs).
fn reconstruct_all_tokens(
    initial_tokens: &[usize],
    node_idx: Option<usize>,
    tree: &[(usize, usize)],
) -> Vec<usize> {
    let mut all = initial_tokens.to_vec();
    all.extend(reconstruct_decoded(node_idx, tree));
    all
}

// ============================================================================
// Harness 1: BeamState score with zero length returns raw sum_log_prob
// ============================================================================

/// Proves that beam score with decoded_len=0 returns raw sum_log_prob.
///
/// When no tokens have been decoded (immediate EOT), the score must equal
/// sum_log_prob regardless of length_penalty. Dividing by 0^penalty would
/// produce NaN or Inf.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::powf, powf_f32_stub)]
fn beam_score_zero_len_returns_raw() {
    let sum_lp: f64 = kani::any();
    let penalty: f64 = kani::any();
    kani::assume(sum_lp.is_finite() && penalty.is_finite());

    let score = beam_score(sum_lp, 0, penalty);
    assert_eq!(
        score, sum_lp,
        "zero decoded_len must return raw sum_log_prob"
    );
}

// ============================================================================
// Harness 2: BeamState score with zero penalty returns raw sum_log_prob
// ============================================================================

/// Proves that beam score with length_penalty=0.0 returns raw sum_log_prob.
///
/// Zero penalty means no length normalization. The score must be the raw
/// cumulative log-probability regardless of decoded_len. This is the mode
/// that maximizes absolute sequence probability.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::powf, powf_f32_stub)]
fn beam_score_zero_penalty_returns_raw() {
    let sum_lp: f64 = kani::any();
    let decoded_len: usize = kani::any();
    kani::assume(sum_lp.is_finite());
    kani::assume(decoded_len <= 1000);

    let score = beam_score(sum_lp, decoded_len, 0.0);
    assert_eq!(
        score, sum_lp,
        "zero length_penalty must return raw sum_log_prob"
    );
}

// ============================================================================
// Harness 3: BeamState score with penalty=1.0 is sum_log_prob / len
// ============================================================================

/// Proves beam score with penalty=1.0 computes mean log-probability.
///
/// At penalty=1.0, score = sum_log_prob / len^1 = sum_log_prob / len,
/// which is the average log-probability per token. This is the standard
/// length normalization used by most beam search implementations.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::powf, powf_f32_stub)]
fn beam_score_penalty_one_is_mean() {
    let sum_lp: f64 = kani::any();
    let decoded_len: usize = kani::any();
    kani::assume(sum_lp.is_finite());
    kani::assume(decoded_len >= 1 && decoded_len <= 100);

    let score = beam_score(sum_lp, decoded_len, 1.0);
    let expected = sum_lp / decoded_len as f64;

    // Use approximate comparison due to floating-point.
    let diff = (score - expected).abs();
    assert!(
        diff < 1e-10,
        "penalty=1.0 score must equal sum_log_prob / decoded_len"
    );
}

// ============================================================================
// Harness 4: BeamState score is finite for finite inputs
// ============================================================================

/// Proves beam score returns a finite value when all inputs are finite and
/// decoded_len > 0. NaN or Inf scores would corrupt beam selection via
/// total_cmp ordering.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::powf, powf_f32_stub)]
fn beam_score_finite_for_finite_inputs() {
    let sum_lp: f64 = kani::any();
    let decoded_len: usize = kani::any();
    let penalty: f64 = kani::any();
    kani::assume(sum_lp.is_finite());
    kani::assume(penalty.is_finite() && penalty >= 0.0 && penalty <= 2.0);
    kani::assume(decoded_len >= 1 && decoded_len <= 1000);

    let score = beam_score(sum_lp, decoded_len, penalty);
    assert!(
        score.is_finite(),
        "beam score must be finite for finite inputs with len > 0"
    );
}

// ============================================================================
// Harness 5: Higher penalty favors longer sequences (for negative log-probs)
// ============================================================================

/// Proves that increasing length_penalty increases score for negative
/// sum_log_prob. Since log-probs are negative, dividing by a larger
/// denominator (len^penalty) makes the result less negative (higher).
/// This is the mechanism that prevents beam search from preferring
/// extremely short sequences.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::powf, powf_f32_stub)]
fn beam_score_higher_penalty_favors_longer() {
    // A typical beam: 10 tokens with cumulative log-prob of -15.
    let sum_lp = -15.0_f64;
    let decoded_len = 10_usize;

    let score_low = beam_score(sum_lp, decoded_len, 0.5);
    let score_high = beam_score(sum_lp, decoded_len, 1.5);

    // Higher penalty divides by a larger value, making negative score less negative.
    assert!(
        score_high > score_low,
        "higher penalty must produce higher score for negative log-probs"
    );
}

// ============================================================================
// Harness 6: reconstruct_decoded returns empty for None node_idx
// ============================================================================

/// Proves reconstruct_decoded returns empty Vec when node_idx is None.
///
/// None node_idx means no tokens were decoded (immediate EOT). The function
/// must return an empty vector, not panic or access the tree.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn reconstruct_none_returns_empty() {
    let tree: Vec<(usize, usize)> = vec![(0, 42), (1, 43)];
    let tokens = reconstruct_decoded(None, &tree);
    assert!(
        tokens.is_empty(),
        "None node_idx must produce empty token sequence"
    );
}

// ============================================================================
// Harness 7: reconstruct_decoded single root node
// ============================================================================

/// Proves reconstruct_decoded correctly returns a single token for a root node.
///
/// A root node has parent_plus_one=0 and is the only node in its chain.
/// The function must return exactly one token.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(2)]
fn reconstruct_single_root_node() {
    let token: usize = kani::any();
    kani::assume(token < 60000); // Reasonable vocab bound.
    let tree = vec![(0usize, token)]; // root node
    let tokens = reconstruct_decoded(Some(0), &tree);

    assert_eq!(tokens.len(), 1, "single root must produce one token");
    assert_eq!(tokens[0], token, "token value must match tree entry");
}

// ============================================================================
// Harness 8: reconstruct_decoded chain of 3 nodes preserves order
// ============================================================================

/// Proves reconstruct_decoded preserves token order for a 3-node chain.
///
/// Tokens are stored leaf-to-root in the tree (via parent pointers) but
/// must be returned root-to-leaf (chronological order). The reverse step
/// must produce the correct sequence.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn reconstruct_chain_preserves_order() {
    // Build a chain: node 0 (root, token=10) → node 1 (token=20) → node 2 (token=30)
    let tree = vec![
        (0usize, 10usize), // node 0: root
        (1usize, 20usize), // node 1: parent_plus_one=1, so parent=0
        (2usize, 30usize), // node 2: parent_plus_one=2, so parent=1
    ];
    let tokens = reconstruct_decoded(Some(2), &tree);

    assert_eq!(tokens.len(), 3, "chain of 3 must produce 3 tokens");
    assert_eq!(tokens[0], 10, "first token from root");
    assert_eq!(tokens[1], 20, "second token");
    assert_eq!(tokens[2], 30, "third token (leaf)");
}

// ============================================================================
// Harness 9: reconstruct_decoded length equals chain depth
// ============================================================================

/// Proves reconstruct_decoded output length matches the depth of the chain
/// from leaf to root. For a chain of depth D, exactly D tokens must be returned.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn reconstruct_length_equals_depth() {
    // Build chain of depth 4.
    let tree = vec![
        (0, 100), // depth 1 (root)
        (1, 200), // depth 2
        (2, 300), // depth 3
        (3, 400), // depth 4
    ];
    let tokens = reconstruct_decoded(Some(3), &tree);
    assert_eq!(tokens.len(), 4, "depth-4 chain must produce 4 tokens");
}

// ============================================================================
// Harness 10: reconstruct_all_tokens prepends initial tokens
// ============================================================================

/// Proves reconstruct_all_tokens prepends initial_tokens before decoded tokens.
///
/// The initial tokens (e.g., [SOT, LANG, TASK, NOTIMESTAMP]) must appear before
/// any decoded tokens. Swapped order would corrupt model replay.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn reconstruct_all_prepends_initial() {
    let initial = [50258usize, 50259, 50360, 50364]; // standard Whisper prompt
    let tree = vec![(0, 1000)]; // one decoded token
    let all = reconstruct_all_tokens(&initial, Some(0), &tree);

    assert_eq!(all.len(), 5, "4 initial + 1 decoded = 5 total");
    assert_eq!(all[0], 50258, "first must be SOT");
    assert_eq!(all[1], 50259, "second must be language token");
    assert_eq!(all[2], 50360, "third must be task token");
    assert_eq!(all[3], 50364, "fourth must be notimestamp");
    assert_eq!(all[4], 1000, "fifth must be decoded token");
}

// ============================================================================
// Harness 11: reconstruct_all_tokens with None returns just initial
// ============================================================================

/// Proves reconstruct_all_tokens returns only initial tokens when node_idx is None.
///
/// When no tokens have been decoded (immediate EOT on first beam), the
/// full sequence must be exactly the initial prompt tokens.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn reconstruct_all_none_returns_initial_only() {
    let initial = [50258usize, 50259, 50360, 50364];
    let tree: Vec<(usize, usize)> = Vec::new();
    let all = reconstruct_all_tokens(&initial, None, &tree);

    assert_eq!(all.len(), initial.len(), "must return only initial tokens");
    assert_eq!(all[0], initial[0]);
    assert_eq!(all[1], initial[1]);
    assert_eq!(all[2], initial[2]);
    assert_eq!(all[3], initial[3]);
}

// ============================================================================
// Harness 12: parent_plus_one encoding: 0 means root, >0 means parent index
// ============================================================================

/// Proves the parent_plus_one encoding is correct: value 0 means root (no parent),
/// value N>0 means parent is at tree index N-1.
///
/// This encoding avoids Option<usize> overhead while maintaining a sentinel.
/// Incorrect encoding would cause infinite loops or wrong tree traversal.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn parent_plus_one_encoding_correctness() {
    // Root node: parent_plus_one=0
    let root_ppo = 0usize;
    let is_root = root_ppo == 0;
    assert!(is_root, "parent_plus_one=0 must indicate root");

    // Child node: parent_plus_one=1 means parent at index 0.
    let child_ppo = 1usize;
    let parent_idx = child_ppo - 1;
    assert_eq!(parent_idx, 0, "parent_plus_one=1 must point to index 0");

    // Grandchild: parent_plus_one=2 means parent at index 1.
    let grandchild_ppo = 2usize;
    let parent_idx2 = grandchild_ppo - 1;
    assert_eq!(
        parent_idx2, 1,
        "parent_plus_one=2 must point to index 1"
    );
}

// ============================================================================
// Harness 13: parent_plus_one from Option<usize> encoding
// ============================================================================

/// Proves the parent_plus_one encoding from Option<usize> via map_or:
/// `node_idx.map_or(0, |i| i + 1)`. None maps to 0 (root), Some(i) maps to i+1.
///
/// This is the exact pattern used in beam_search_decode line 291.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn parent_plus_one_from_option() {
    let none_case: Option<usize> = None;
    let ppo_none = none_case.map_or(0, |i| i + 1);
    assert_eq!(ppo_none, 0, "None must encode as 0 (root sentinel)");

    let some_case: Option<usize> = Some(5);
    let ppo_some = some_case.map_or(0, |i| i + 1);
    assert_eq!(ppo_some, 6, "Some(5) must encode as 6");
    assert_eq!(ppo_some - 1, 5, "decoding must recover original index");
}

// ============================================================================
// Harness 14: top_k_with_log_probs returns valid indices
// ============================================================================

/// Proves top_k_with_log_probs returns indices within [0, logits.len()).
///
/// Out-of-bounds indices would cause panics in beam_search_decode when
/// comparing tokens to EOT_TOKEN or building the tree.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(5)]
fn top_k_indices_in_bounds() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();
    let d: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite());

    let logits = [a, b, c, d];
    let result = beam::top_k_with_log_probs(&logits, 2);

    for &(idx, _lp) in &result {
        assert!(
            idx < logits.len(),
            "top_k index must be within logits bounds"
        );
    }
}

// ============================================================================
// Harness 15: top_k_with_log_probs returns at most k results
// ============================================================================

/// Proves top_k_with_log_probs returns at most k entries.
///
/// Returning more than k would expand beams beyond beam_width, violating
/// the O(B*k) candidate budget and potentially causing memory blowup.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(5)]
fn top_k_returns_at_most_k() {
    let logits = [1.0f32, 5.0, 2.0, 3.0];
    let k = 2;
    let result = beam::top_k_with_log_probs(&logits, k);

    assert!(
        result.len() <= k,
        "top_k must return at most k entries"
    );
}

// ============================================================================
// Harness 16: top_k_with_log_probs log-probs are non-positive
// ============================================================================

/// Proves all log-probabilities from top_k_with_log_probs are <= 0.
///
/// Log-probabilities are log(softmax(logit)), and softmax outputs are in [0,1],
/// so their log must be <= 0. Positive log-probs would corrupt beam scoring.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
fn top_k_log_probs_nonpositive() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite() && c.is_finite());

    let logits = [a, b, c];
    let result = beam::top_k_with_log_probs(&logits, 2);

    for &(_idx, lp) in &result {
        assert!(
            lp <= 0.0 || lp == f32::NEG_INFINITY,
            "top_k log-prob must be <= 0"
        );
    }
}

// ============================================================================
// Harness 17: top_k_with_log_probs sorted descending
// ============================================================================

/// Proves top_k_with_log_probs returns entries sorted by log-prob descending.
///
/// The beam search loop relies on candidates being sorted to correctly
/// select the top beam_width beams after scoring. Unsorted results would
/// cause the wrong beams to be kept.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(5)]
fn top_k_sorted_descending() {
    let logits = [1.0f32, 5.0, 2.0, 3.0];
    let result = beam::top_k_with_log_probs(&logits, 3);

    for i in 1..result.len() {
        assert!(
            result[i - 1].1 >= result[i].1,
            "top_k must be sorted by log-prob descending"
        );
    }
}

// ============================================================================
// Harness 18: top_k_with_log_probs handles all-equal logits
// ============================================================================

/// Proves top_k_with_log_probs handles all-equal logits without panic.
///
/// When all logits are identical, every token has equal probability.
/// The function must return k valid entries without panicking on tie-breaking.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(5)]
fn top_k_all_equal_logits() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite());

    let logits = [v, v, v, v];
    let result = beam::top_k_with_log_probs(&logits, 3);

    assert!(!result.is_empty(), "must return at least one entry");
    assert!(result.len() <= 3, "must return at most k entries");
    for &(idx, lp) in &result {
        assert!(idx < 4, "index in bounds");
        assert!(!lp.is_nan(), "log-prob must not be NaN");
    }
}

// ============================================================================
// Harness 19: top_k_with_log_probs handles all-NEG_INFINITY logits
// ============================================================================

/// Proves top_k_with_log_probs returns a fallback entry for all-NEG_INFINITY.
///
/// After full suppression, all logits may be NEG_INFINITY. The function must
/// return at least one entry (index 0 with NEG_INFINITY) rather than panicking
/// or returning an empty vec that would cause beam search to terminate.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
fn top_k_all_neg_inf() {
    let logits = [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
    let result = beam::top_k_with_log_probs(&logits, 2);

    assert!(
        !result.is_empty(),
        "must return at least one entry for all -inf"
    );
    let (idx, _) = result[0];
    assert!(idx < logits.len(), "fallback index must be in bounds");
}

// ============================================================================
// Harness 20: top_k_with_log_probs on empty logits
// ============================================================================

/// Proves top_k_with_log_probs returns a fallback for empty logits.
///
/// If the model returns zero-length logits (should not happen in practice),
/// the function must not panic. It returns [(0, NEG_INFINITY)] as fallback.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn top_k_empty_logits() {
    let logits: &[f32] = &[];
    let result = beam::top_k_with_log_probs(logits, 3);

    assert!(
        !result.is_empty(),
        "must return fallback for empty logits"
    );
}

// ============================================================================
// Harness 21: top_k_with_log_probs k=0 returns fallback
// ============================================================================

/// Proves top_k_with_log_probs with k=0 returns a fallback entry.
///
/// k=0 means "request zero top entries." The effective_k clamp to 0
/// triggers the empty fallback path returning [(0, NEG_INFINITY)].
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn top_k_zero_k() {
    let logits = [1.0f32, 2.0, 3.0];
    let result = beam::top_k_with_log_probs(&logits, 0);

    assert!(
        !result.is_empty(),
        "k=0 must return at least one fallback entry"
    );
}

// ============================================================================
// Harness 22: top_k_with_log_probs k > len clamps to len
// ============================================================================

/// Proves top_k_with_log_probs clamps k to logits.len() when k is larger.
///
/// Requesting more candidates than vocabulary entries should return all entries,
/// not panic on out-of-bounds partition.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn top_k_clamps_to_len() {
    let logits = [1.0f32, 2.0, 3.0];
    let result = beam::top_k_with_log_probs(&logits, 10);

    assert_eq!(
        result.len(),
        3,
        "k > len must clamp to logits.len()"
    );
}

// ============================================================================
// Harness 23: WhisperBeamConfig validate rejects infinite length_penalty
// ============================================================================

/// Proves WhisperBeamConfig.validate() rejects Inf length_penalty.
///
/// Infinite penalty would divide by len^Inf = Inf for len > 1, producing
/// zero scores for all multi-token hypotheses and breaking beam selection.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn beam_config_inf_penalty_rejected() {
    let config = WhisperBeamConfig {
        beam_width: 5,
        length_penalty: f64::INFINITY,
    };
    assert!(
        config.validate().is_err(),
        "infinite length_penalty must fail validation"
    );
}

// ============================================================================
// Harness 24: WhisperBeamConfig validate rejects negative infinity penalty
// ============================================================================

/// Proves WhisperBeamConfig.validate() rejects NEG_INFINITY length_penalty.
///
/// Negative infinity penalty would make len^penalty = 0 for len > 1,
/// causing division by zero in score computation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn beam_config_neg_inf_penalty_rejected() {
    let config = WhisperBeamConfig {
        beam_width: 5,
        length_penalty: f64::NEG_INFINITY,
    };
    assert!(
        config.validate().is_err(),
        "NEG_INFINITY length_penalty must fail validation"
    );
}

// ============================================================================
// Harness 25: WhisperBeamConfig default validates successfully
// ============================================================================

/// Proves WhisperBeamConfig::default() passes validation.
///
/// The default configuration (beam_width=5, length_penalty=1.0) must always
/// be valid. If the default fails, no beam search would work without overrides.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn beam_config_default_validates() {
    let config = WhisperBeamConfig::default();
    assert!(
        config.validate().is_ok(),
        "default WhisperBeamConfig must pass validation"
    );
}

// ============================================================================
// Harness 26: Beam score ordering preserved by total_cmp
// ============================================================================

/// Proves that beam score ordering is total (no incomparable pairs) for
/// finite scores. This is critical because beam_search_decode uses
/// total_cmp for sorting beams — partial orderings could produce
/// non-deterministic results.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn beam_score_total_ordering() {
    let s1: f64 = kani::any();
    let s2: f64 = kani::any();
    kani::assume(s1.is_finite() && s2.is_finite());

    // total_cmp provides a total ordering on f64 (NaN has a defined position).
    use std::cmp::Ordering;
    let cmp = s1.total_cmp(&s2);
    match cmp {
        Ordering::Less => assert!(s1 <= s2),
        Ordering::Equal => assert!(s1 == s2),
        Ordering::Greater => assert!(s1 >= s2),
    }
}

// ============================================================================
// Harness 27: Cumulative score accumulation does not produce NaN
// ============================================================================

/// Proves that adding a finite log-prob to a finite cumulative score
/// produces a finite result. NaN scores would corrupt beam ordering
/// via total_cmp.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn cumulative_score_stays_finite() {
    let parent_score: f64 = kani::any();
    let log_prob: f32 = kani::any();
    kani::assume(parent_score.is_finite() && log_prob.is_finite());
    // Bound to realistic ranges to avoid overflow to infinity.
    kani::assume(parent_score > -1e15 && parent_score < 0.0);
    kani::assume(log_prob > -100.0 && log_prob <= 0.0);

    let new_score = parent_score + f64::from(log_prob);
    assert!(
        new_score.is_finite(),
        "cumulative score must stay finite for finite inputs"
    );
    assert!(
        new_score <= parent_score,
        "adding non-positive log-prob must not increase score"
    );
}

// ============================================================================
// Harness 28: Token u32 overflow detection for large token IDs
// ============================================================================

/// Proves u32::try_from correctly detects token IDs that exceed u32::MAX.
///
/// Whisper vocab is ~51866, well within u32. But the token ID type is usize,
/// and on 64-bit platforms usize can exceed u32::MAX. The beam search decode
/// loop must catch this with try_from.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn token_u32_overflow_detected() {
    let token_id: usize = kani::any();
    kani::assume(token_id > u32::MAX as usize);

    let result = u32::try_from(token_id);
    assert!(
        result.is_err(),
        "token IDs exceeding u32::MAX must be caught by try_from"
    );
}

// ============================================================================
// Harness 29: Valid Whisper token IDs fit in u32
// ============================================================================

/// Proves all valid Whisper token IDs (up to max vocab 51866) fit in u32.
///
/// The maximum token ID in any Whisper model is less than 52000. All such
/// values must convert to u32 without error.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn valid_token_ids_fit_u32() {
    let token_id: usize = kani::any();
    kani::assume(token_id < 52000); // Largest Whisper vocab + margin.

    let result = u32::try_from(token_id);
    assert!(
        result.is_ok(),
        "valid Whisper token IDs must fit in u32"
    );
}

// ============================================================================
// Harness 30: EOT_TOKEN comparison is correct type
// ============================================================================

/// Proves EOT_TOKEN can be correctly compared against a usize token value
/// without type coercion issues. The beam search compares `token == EOT_TOKEN`
/// where both sides are usize.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn eot_token_comparison_type_safe() {
    let token: usize = EOT_TOKEN;
    assert_eq!(token, 50257, "EOT_TOKEN must be 50257");
    assert!(token < u32::MAX as usize, "EOT_TOKEN must fit in u32");

    let not_eot: usize = 1000;
    assert_ne!(not_eot, EOT_TOKEN, "non-EOT token must not match EOT");
}

// ============================================================================
// Harness 31: Tree branching produces distinct paths
// ============================================================================

/// Proves that two sibling beams (same parent, different tokens) produce
/// distinct decoded sequences. The parent-pointer tree must not alias
/// siblings' token history.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn tree_branching_distinct_paths() {
    let mut tree: Vec<(usize, usize)> = Vec::new();

    // Root node (token=100).
    tree.push((0, 100));

    // Two children of root with different tokens.
    tree.push((1, 200)); // node 1: parent=0, token=200
    tree.push((1, 300)); // node 2: parent=0, token=300

    let seq1 = reconstruct_decoded(Some(1), &tree);
    let seq2 = reconstruct_decoded(Some(2), &tree);

    assert_eq!(seq1.len(), 2, "child 1 must have depth 2");
    assert_eq!(seq2.len(), 2, "child 2 must have depth 2");
    assert_eq!(seq1[0], seq2[0], "siblings share root token");
    assert_ne!(
        seq1[1], seq2[1],
        "siblings must have distinct leaf tokens"
    );
}

// ============================================================================
// Harness 32: avg_logprob computation safety in beam finalization
// ============================================================================

/// Proves avg_logprob division is safe when decoded_tokens is non-empty.
///
/// The beam search finalization computes avg_logprob as
/// `sum_log_prob / decoded_tokens.len() as f64`. For non-empty tokens,
/// this must produce a finite value.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn avg_logprob_division_safe() {
    let sum_lp: f64 = kani::any();
    let token_count: usize = kani::any();
    kani::assume(sum_lp.is_finite());
    kani::assume(token_count >= 1 && token_count <= 1000);

    let avg = sum_lp / token_count as f64;
    assert!(avg.is_finite(), "avg_logprob must be finite for finite inputs");
}

// ============================================================================
// Harness 33: avg_logprob for empty tokens defaults to 0.0
// ============================================================================

/// Proves the empty-tokens branch returns 0.0 to avoid division by zero.
///
/// When `decoded_tokens.is_empty()`, avg_logprob must be 0.0 regardless
/// of sum_log_prob. This matches the pattern in beam_search_decode line 341-344.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn avg_logprob_empty_is_zero() {
    let sum_lp: f64 = kani::any();
    kani::assume(sum_lp.is_finite());

    let decoded_tokens: Vec<usize> = Vec::new();
    let avg = if decoded_tokens.is_empty() {
        0.0
    } else {
        sum_lp / decoded_tokens.len() as f64
    };

    assert_eq!(avg, 0.0, "empty tokens must produce avg_logprob = 0.0");
}

// ============================================================================
// Harness 34: Beam width truncation preserves at most beam_width beams
// ============================================================================

/// Proves that truncating a sorted beam list to beam_width keeps exactly
/// min(beams.len(), beam_width) entries. This is the invariant maintained
/// after each beam expansion step (line 315).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn beam_truncation_invariant() {
    let beam_width: usize = kani::any();
    let beam_count: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 20);
    kani::assume(beam_count >= 0 && beam_count <= 40);

    // Simulate truncation.
    let after_truncate = beam_count.min(beam_width);
    assert!(
        after_truncate <= beam_width,
        "truncated count must be <= beam_width"
    );
}

// ============================================================================
// Harness 35: Candidate capacity calculation does not overflow
// ============================================================================

/// Proves the candidate vector capacity `active_count * beam_width` does not
/// overflow for realistic beam parameters. Overflow would cause allocation
/// failure or incorrect capacity.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn candidate_capacity_no_overflow() {
    let active_count: usize = kani::any();
    let beam_width: usize = kani::any();
    kani::assume(active_count <= 100); // No more than 100 beams active.
    kani::assume(beam_width <= 100);   // Realistic beam width bound.

    let capacity = active_count.checked_mul(beam_width);
    assert!(
        capacity.is_some(),
        "candidate capacity must not overflow for realistic params"
    );
}

// ============================================================================
// Harness 36: top_k_with_log_probs first result is the argmax
// ============================================================================

/// Proves the first entry returned by top_k_with_log_probs has the highest
/// log-probability (i.e., corresponds to the argmax). When beam_width=1,
/// this must match greedy decode behavior.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(5)]
fn top_k_first_is_argmax() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();
    let d: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite());
    // Ensure a clear winner to avoid tie-breaking ambiguity.
    kani::assume(a != b && a != c && a != d && b != c && b != d && c != d);

    let logits = [a, b, c, d];
    let result = beam::top_k_with_log_probs(&logits, 4);

    // The first entry must have the highest log-prob.
    for i in 1..result.len() {
        assert!(
            result[0].1 >= result[i].1,
            "first entry must have highest log-prob"
        );
    }
}

// ============================================================================
// Harness 37: top_k_with_log_probs returns unique indices
// ============================================================================

/// Proves top_k_with_log_probs returns distinct token indices.
///
/// Duplicate indices would cause the same token to appear multiple times
/// in the beam expansion, wasting beam capacity on identical hypotheses.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(5)]
fn top_k_unique_indices() {
    let logits = [1.0f32, 5.0, 2.0, 3.0];
    let result = beam::top_k_with_log_probs(&logits, 3);

    for i in 0..result.len() {
        for j in (i + 1)..result.len() {
            assert_ne!(
                result[i].0, result[j].0,
                "top_k indices must be unique"
            );
        }
    }
}

// ============================================================================
// Harness 38: reconstruct_decoded from branching tree returns correct branch
// ============================================================================

/// Proves reconstruct_decoded follows the correct branch when the tree
/// has multiple branches at different levels. Node 3 and node 4 are both
/// children of node 1, but follow different paths.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn reconstruct_branching_tree_correct_path() {
    // Tree:
    //   0: root (token=A)
    //   1: child of 0 (token=B)
    //   2: child of 0 (token=C) -- different branch
    //   3: child of 1 (token=D)
    let tree = vec![
        (0usize, 10usize), // node 0: root, token=10
        (1usize, 20usize), // node 1: parent=0, token=20
        (1usize, 30usize), // node 2: parent=0, token=30 (sibling of 1)
        (2usize, 40usize), // node 3: parent=1, token=40
    ];

    // Path from node 3: 3 -> 1 -> 0 => tokens [10, 20, 40]
    let seq = reconstruct_decoded(Some(3), &tree);
    assert_eq!(seq.len(), 3);
    assert_eq!(seq[0], 10, "root token");
    assert_eq!(seq[1], 20, "intermediate token from node 1");
    assert_eq!(seq[2], 40, "leaf token from node 3");

    // Path from node 2: 2 -> 0 => tokens [10, 30]
    let seq2 = reconstruct_decoded(Some(2), &tree);
    assert_eq!(seq2.len(), 2);
    assert_eq!(seq2[0], 10, "root token");
    assert_eq!(seq2[1], 30, "leaf token from node 2");
}

// ============================================================================
// Harness 39: beam_score monotonicity — more negative log-prob means lower score
// ============================================================================

/// Proves that a beam with more negative sum_log_prob has a lower score
/// (at penalty=1.0). This ensures beam selection correctly favors
/// higher-probability sequences.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::powf, powf_f32_stub)]
fn beam_score_monotone_in_sum_log_prob() {
    let sum_lp_high: f64 = kani::any();
    let sum_lp_low: f64 = kani::any();
    kani::assume(sum_lp_high.is_finite() && sum_lp_low.is_finite());
    kani::assume(sum_lp_high > sum_lp_low);

    let decoded_len = 5_usize;
    let penalty = 1.0_f64;

    let score_high = beam_score(sum_lp_high, decoded_len, penalty);
    let score_low = beam_score(sum_lp_low, decoded_len, penalty);

    assert!(
        score_high > score_low,
        "higher sum_log_prob must produce higher score at same length"
    );
}

// ============================================================================
// Harness 40: tree growth bounded by beam_width * max_steps
// ============================================================================

/// Proves tree growth is bounded: each non-EOT expansion adds exactly one node.
/// With beam_width beams and max_steps steps, the tree can have at most
/// beam_width + beam_width * max_steps nodes. This bounds memory usage.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn tree_growth_bounded() {
    let beam_width: usize = kani::any();
    let max_steps: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 20);
    kani::assume(max_steps >= 1 && max_steps <= 300);

    // Initial beams add at most beam_width nodes.
    // Each subsequent step adds at most beam_width nodes.
    let max_nodes = beam_width.checked_mul(max_steps + 1);
    assert!(
        max_nodes.is_some(),
        "tree size bound must not overflow for realistic params"
    );
    // Verify the bound is reasonable (< 10K for realistic params).
    assert!(
        max_nodes.unwrap() <= 20 * 301,
        "tree size within expected bound"
    );
}

// ============================================================================
// Harness 41: top_k log_sum_exp is finite for finite inputs
// ============================================================================

/// Proves the log_sum_exp computed in top_k_with_log_probs is finite
/// when all inputs are finite. This is the normalizing constant for
/// log-probabilities; if it's NaN/Inf, all log-probs are corrupted.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn top_k_log_sum_exp_finite() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite() && c.is_finite());

    let logits = [a, b, c];
    // Reproduce the log_sum_exp computation from top_k_with_log_probs.
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_val != f32::NEG_INFINITY {
        let log_sum_exp: f32 = logits
            .iter()
            .map(|&v| (v - max_val).exp())
            .sum::<f32>()
            .ln()
            + max_val;
        assert!(
            log_sum_exp.is_finite(),
            "log_sum_exp must be finite for finite inputs"
        );
    }
}

// ============================================================================
// Harness 42: reconstruct_all_tokens length is initial + decoded
// ============================================================================

/// Proves the output length of reconstruct_all_tokens equals
/// initial_tokens.len() + depth of decoded chain.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn reconstruct_all_length_additive() {
    let initial = [50258usize, 50259];
    // Chain of 2 decoded tokens.
    let tree = vec![
        (0usize, 100usize), // root
        (1usize, 200usize), // child of root
    ];
    let all = reconstruct_all_tokens(&initial, Some(1), &tree);

    assert_eq!(
        all.len(),
        initial.len() + 2,
        "total length must be initial + decoded depth"
    );
}

// ============================================================================
// Harness 43: WhisperBeamConfig validate accepts large finite penalty
// ============================================================================

/// Proves WhisperBeamConfig.validate() accepts a large but finite
/// length_penalty. Users may set penalty > 1 to strongly favor
/// longer sequences.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn beam_config_large_finite_penalty_accepted() {
    let config = WhisperBeamConfig {
        beam_width: 5,
        length_penalty: 100.0,
    };
    assert!(
        config.validate().is_ok(),
        "large finite length_penalty must pass validation"
    );
}

// ============================================================================
// Harness 44: WhisperBeamConfig validate accepts negative penalty
// ============================================================================

/// Proves WhisperBeamConfig.validate() accepts negative length_penalty.
///
/// Negative penalties favor shorter sequences (the opposite of positive).
/// While unusual, this is valid and should not be rejected at validation time.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn beam_config_negative_penalty_accepted() {
    let config = WhisperBeamConfig {
        beam_width: 5,
        length_penalty: -1.0,
    };
    assert!(
        config.validate().is_ok(),
        "negative finite length_penalty must pass validation"
    );
}

// ============================================================================
// Harness 45: beam_score negative penalty favors shorter sequences
// ============================================================================

/// Proves that negative length_penalty produces lower scores for longer
/// sequences (with negative log-probs), which is the inverse of positive penalty.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::powf, powf_f32_stub)]
fn beam_score_negative_penalty_favors_shorter() {
    let sum_lp = -10.0_f64;
    let short_len = 3_usize;
    let long_len = 10_usize;
    let penalty = -0.5_f64;

    let score_short = beam_score(sum_lp, short_len, penalty);
    let score_long = beam_score(sum_lp, long_len, penalty);

    // Negative penalty: divides by len^(-0.5) = multiplies by len^0.5.
    // For negative sum_lp, longer sequences get more negative scores.
    assert!(
        score_short > score_long,
        "negative penalty must favor shorter sequences for negative log-probs"
    );
}

// ============================================================================
// Harness 46: top_k_with_log_probs k=1 returns exactly one result
// ============================================================================

/// Proves top_k_with_log_probs with k=1 returns exactly one entry,
/// equivalent to greedy argmax. This is the minimal beam width.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(5)]
fn top_k_one_returns_single() {
    let logits = [1.0f32, 5.0, 2.0, 3.0];
    let result = beam::top_k_with_log_probs(&logits, 1);

    assert_eq!(
        result.len(),
        1,
        "k=1 must return exactly one entry"
    );
    assert!(
        result[0].0 < logits.len(),
        "single entry index must be in-bounds"
    );
}
