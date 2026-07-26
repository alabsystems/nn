// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for beam search decoding safety.
//!
//! Complements the inline `#[cfg(kani)]` proofs in `beam_search.rs` and
//! `beam_search_helpers.rs` with additional safety properties:
//! - Builder chain preserves config validity
//! - Default config validity
//! - BeamHypothesis::new constructor consistency
//! - Top-k returns no duplicate indices
//! - Finalize tree: output beams have token_ids.len() >= 1
//! - Finalize tree: finished flag correctness
//! - Finalize tree: output sorted by normalized score
//! - EOS detection enables early stopping
//! - Log-softmax NaN input sanitization
//! - Log-softmax all-NEG_INFINITY guard
//! - Reconstruct tokens path depth bound
//! - Score length penalty normalization effect
//! - BeamSearchOutput::new preserves beam count

use super::*;

fn powf_f64_stub(_b: f64, _e: f64) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
    r
}

fn exp_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
    if x <= 0.0 {
        kani::assume(r <= 1.0);
    }
    if x > 0.0 {
        kani::assume(r > 1.0);
    }
    r
}

fn ln_f32_stub(_x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -100.0 && r <= 100.0);
    r
}

// ---------------------------------------------------------------------------
// 1. BeamSearchConfig::new preserves beam_width > 0 and validates
// ---------------------------------------------------------------------------

/// Prove that `BeamSearchConfig::new(w)` with `w >= 1` always produces a
/// config where `validate()` succeeds (beam_width > 0 and length_penalty
/// finite).
#[kani::unwind(1)]
#[kani::proof]
fn proof_config_new_preserves_valid_width() {
    let width: usize = kani::any();
    kani::assume(width >= 1 && width <= 64);
    let config = BeamSearchConfig::new(width);
    assert_eq!(config.beam_width, width, "new() must set beam_width");
    assert!(
        config.validate().is_ok(),
        "config from new(w >= 1) must validate"
    );
}

// ---------------------------------------------------------------------------
// 2. Builder chain preserves validity
// ---------------------------------------------------------------------------

/// Prove that chaining all builder methods on a valid config preserves
/// validation, provided the length_penalty argument is finite.
#[kani::unwind(1)]
#[kani::proof]
fn proof_builder_chain_preserves_validity() {
    let width: usize = kani::any();
    kani::assume(width >= 1 && width <= 32);
    let max_tokens: usize = kani::any();
    kani::assume(max_tokens <= 512);
    let penalty: f64 = kani::any();
    kani::assume(penalty.is_finite() && penalty >= 0.0 && penalty <= 10.0);
    let eos: usize = kani::any();
    kani::assume(eos <= 65535);

    let config = BeamSearchConfig::new(width)
        .with_max_new_tokens(max_tokens)
        .with_length_penalty(penalty)
        .with_early_stopping(true)
        .with_eos_token_id(eos);

    assert!(
        config.validate().is_ok(),
        "builder chain with valid args must validate"
    );
    assert_eq!(config.beam_width, width);
    assert_eq!(config.max_new_tokens, max_tokens);
    assert_eq!(config.eos_token_id, Some(eos));
    assert!(config.early_stopping);
}

// ---------------------------------------------------------------------------
// 3. Default config is valid
// ---------------------------------------------------------------------------

/// Prove that `BeamSearchConfig::default()` always passes validation.
#[kani::unwind(1)]
#[kani::proof]
fn proof_default_config_is_valid() {
    let config = BeamSearchConfig::default();
    assert!(config.validate().is_ok(), "default config must be valid");
    assert!(config.beam_width > 0, "default beam_width must be > 0");
    assert!(
        config.length_penalty.is_finite(),
        "default length_penalty must be finite"
    );
}

// ---------------------------------------------------------------------------
// 4. BeamHypothesis::new constructor consistency
// ---------------------------------------------------------------------------

/// Prove that `BeamHypothesis::new` stores its arguments faithfully.
#[kani::unwind(5)]
#[kani::proof]
fn proof_hypothesis_new_stores_args() {
    let num_tokens: usize = kani::any();
    kani::assume(num_tokens <= 4);
    let log_prob: f64 = kani::any();
    kani::assume(log_prob.is_finite() && log_prob.abs() < 1e6);
    let finished: bool = kani::any();

    let ids: Vec<usize> = (0..num_tokens).collect();
    let hyp = BeamHypothesis::new(ids.clone(), log_prob, finished);

    assert_eq!(hyp.token_ids.len(), num_tokens);
    assert!((hyp.log_prob - log_prob).abs() < 1e-15);
    assert_eq!(hyp.finished, finished);
}

// ---------------------------------------------------------------------------
// 5. Top-k returns no duplicate indices
// ---------------------------------------------------------------------------

/// Prove that indices returned by `top_k_by_value` are all distinct (no
/// duplicates). This is critical: duplicate beam indices would cause the
/// decoder to track the same hypothesis twice.
#[kani::unwind(7)]
#[kani::proof]
fn proof_top_k_no_duplicate_indices() {
    let len: usize = kani::any();
    kani::assume(len >= 2 && len <= 6);
    let k: usize = kani::any();
    kani::assume(k >= 1 && k <= len);

    let mut values = vec![0.0f32; len];
    for v in values.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite());
    }

    let result = top_k_by_value(&values, k);
    // Check uniqueness: for every pair (i, j) with i != j, indices differ.
    for i in 0..result.len() {
        for j in (i + 1)..result.len() {
            assert_ne!(
                result[i].0, result[j].0,
                "top_k must return distinct indices"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Finalize tree: all output beams have token_ids.len() >= 1
// ---------------------------------------------------------------------------

/// Prove that every beam in the output of `finalize_tree` has at least one
/// token (output length >= 1), given that all tree nodes exist and have valid
/// parent pointers.
#[kani::unwind(5)]
#[kani::proof]
#[kani::stub(f64::powf, powf_f64_stub)]
fn proof_finalize_tree_output_nonempty_tokens() {
    let num_nodes: usize = kani::any();
    kani::assume(num_nodes >= 1 && num_nodes <= 4);

    let mut tree: Vec<(Option<usize>, usize)> = Vec::with_capacity(num_nodes);
    tree.push((None, 0)); // root
    for i in 1..num_nodes {
        let parent: usize = kani::any();
        kani::assume(parent < i);
        let token: usize = kani::any();
        kani::assume(token < 10);
        tree.push((Some(parent), token));
    }

    // One completed beam pointing to a valid node.
    let node_idx: usize = kani::any();
    kani::assume(node_idx < num_nodes);
    let log_prob: f64 = kani::any();
    kani::assume(log_prob.is_finite() && log_prob.abs() < 100.0);
    let completed = vec![(node_idx, log_prob, 1_usize)];
    let active: Vec<(usize, f64)> = Vec::new();

    let config = BeamSearchConfig {
        beam_width: 4,
        length_penalty: 1.0,
        ..Default::default()
    };

    let output = finalize_tree(completed, &active, &tree, &config);
    for beam in &output.beams {
        assert!(
            !beam.token_ids.is_empty(),
            "every output beam must have >= 1 token"
        );
    }
}

// ---------------------------------------------------------------------------
// 7. Finalize tree: completed beams marked finished, active beams not
// ---------------------------------------------------------------------------

/// Prove that `finalize_tree` correctly marks completed beams as `finished`
/// and active beams as not `finished`.
#[kani::unwind(5)]
#[kani::proof]
#[kani::stub(f64::powf, powf_f64_stub)]
fn proof_finalize_tree_finished_flag_correctness() {
    let num_nodes: usize = kani::any();
    kani::assume(num_nodes >= 2 && num_nodes <= 4);

    let mut tree: Vec<(Option<usize>, usize)> = Vec::with_capacity(num_nodes);
    tree.push((None, 0));
    for i in 1..num_nodes {
        let parent: usize = kani::any();
        kani::assume(parent < i);
        tree.push((Some(parent), i));
    }

    // One completed + one active beam.
    let c_idx: usize = kani::any();
    kani::assume(c_idx < num_nodes);
    let c_lp: f64 = kani::any();
    kani::assume(c_lp.is_finite() && c_lp.abs() < 50.0);
    let completed = vec![(c_idx, c_lp, 1_usize)];

    let a_idx: usize = kani::any();
    kani::assume(a_idx < num_nodes);
    let a_lp: f64 = kani::any();
    kani::assume(a_lp.is_finite() && a_lp.abs() < 50.0);
    let active = vec![(a_idx, a_lp)];

    let config = BeamSearchConfig {
        beam_width: 4,
        length_penalty: 1.0,
        ..Default::default()
    };

    let output = finalize_tree(completed, &active, &tree, &config);
    // Count finished vs not-finished.
    let finished_count = output.beams.iter().filter(|b| b.finished).count();
    let active_count = output.beams.iter().filter(|b| !b.finished).count();
    // We provided 1 completed + 1 active, total <= beam_width=4, so both appear.
    assert!(
        finished_count >= 1,
        "at least one completed beam must be marked finished"
    );
    assert!(
        active_count >= 1,
        "at least one active beam must be marked not-finished"
    );
}

// ---------------------------------------------------------------------------
// 8. Finalize tree: output beams sorted by normalized score descending
// ---------------------------------------------------------------------------

/// Prove that `finalize_tree` output beams are sorted in descending order
/// of length-normalized score.
#[kani::unwind(5)]
#[kani::proof]
#[kani::stub(f64::powf, powf_f64_stub)]
fn proof_finalize_tree_sorted_by_score() {
    let num_nodes: usize = kani::any();
    kani::assume(num_nodes >= 2 && num_nodes <= 3);

    let mut tree: Vec<(Option<usize>, usize)> = Vec::with_capacity(num_nodes);
    tree.push((None, 0));
    for i in 1..num_nodes {
        tree.push((Some(i - 1), i));
    }

    // Two completed beams with different log_probs.
    let lp1: f64 = kani::any();
    kani::assume(lp1.is_finite() && lp1.abs() < 50.0);
    let lp2: f64 = kani::any();
    kani::assume(lp2.is_finite() && lp2.abs() < 50.0);

    let idx1: usize = kani::any();
    kani::assume(idx1 < num_nodes);
    let idx2: usize = kani::any();
    kani::assume(idx2 < num_nodes);

    let completed = vec![(idx1, lp1, 1_usize), (idx2, lp2, 1_usize)];
    let active: Vec<(usize, f64)> = Vec::new();

    let penalty: f64 = kani::any();
    kani::assume(penalty.is_finite() && penalty >= 0.0 && penalty <= 5.0);
    let config = BeamSearchConfig {
        beam_width: 4,
        length_penalty: penalty,
        ..Default::default()
    };

    let output = finalize_tree(completed, &active, &tree, &config);
    // Verify descending sort order by score.
    // score() uses the same length_penalty, so we check output ordering.
    for i in 1..output.beams.len() {
        let s_prev = if penalty == 0.0 || output.beams[i - 1].token_ids.is_empty() {
            output.beams[i - 1].log_prob
        } else {
            output.beams[i - 1].log_prob
                / (output.beams[i - 1].token_ids.len() as f64).powf(penalty)
        };
        let s_curr = if penalty == 0.0 || output.beams[i].token_ids.is_empty() {
            output.beams[i].log_prob
        } else {
            output.beams[i].log_prob / (output.beams[i].token_ids.len() as f64).powf(penalty)
        };
        assert!(
            s_prev.total_cmp(&s_curr).is_ge(),
            "output beams must be sorted by score descending"
        );
    }
}

// ---------------------------------------------------------------------------
// 9. EOS detection: early stopping recognizes finished beams
// ---------------------------------------------------------------------------

/// Prove that when `eos_token_id` is set and matches a token, `is_eos`
/// returns true; combined with `early_stopping`, this terminates decoding.
/// Also prove non-matching tokens do not trigger EOS.
#[kani::unwind(1)]
#[kani::proof]
fn proof_eos_detection_enables_early_stopping() {
    let eos_id: usize = kani::any();
    kani::assume(eos_id <= 65535);

    let config = BeamSearchConfig::new(4)
        .with_eos_token_id(eos_id)
        .with_early_stopping(true);

    // Token matching EOS must trigger detection.
    assert!(
        is_eos(eos_id, &config),
        "is_eos must return true for matching EOS token"
    );
    assert!(config.early_stopping, "early_stopping must be set");

    // Token not matching EOS must not trigger.
    let other_token: usize = kani::any();
    kani::assume(other_token != eos_id && other_token <= 65535);
    assert!(
        !is_eos(other_token, &config),
        "is_eos must return false for non-EOS token"
    );
}

// ---------------------------------------------------------------------------
// 10. Log-softmax sanitizes NaN inputs to NEG_INFINITY
// ---------------------------------------------------------------------------

/// Prove that `log_softmax` with a mix of NaN and finite inputs does not
/// produce NaN in the output. NaN inputs are sanitized to NEG_INFINITY,
/// meaning outputs are valid log-probabilities or NEG_INFINITY.
#[kani::unwind(5)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn proof_log_softmax_nan_sanitized() {
    let len: usize = kani::any();
    kani::assume(len >= 2 && len <= 4);

    let mut logits = vec![0.0f32; len];
    // First element is finite, rest are NaN.
    logits[0] = kani::any();
    kani::assume(logits[0].is_finite());
    for v in logits.iter_mut().skip(1) {
        *v = f32::NAN;
    }

    let result = log_softmax(&logits);
    assert_eq!(result.len(), len, "output length must match input");
    // No NaN in output (NaN inputs become NEG_INFINITY).
    for &v in &result {
        assert!(
            !v.is_nan(),
            "log_softmax must not produce NaN after sanitization"
        );
    }
}

// ---------------------------------------------------------------------------
// 11. Log-softmax: all-NEG_INFINITY inputs produce all-NEG_INFINITY output
// ---------------------------------------------------------------------------

/// Prove that when all logits are NEG_INFINITY, log_softmax returns
/// NEG_INFINITY for every element (not NaN from inf-inf).
#[kani::unwind(5)]
#[kani::proof]
fn proof_log_softmax_all_neginf_guard() {
    let len: usize = kani::any();
    kani::assume(len >= 1 && len <= 4);

    let logits = vec![f32::NEG_INFINITY; len];
    let result = log_softmax(&logits);
    assert_eq!(result.len(), len);
    for &v in &result {
        assert_eq!(
            v,
            f32::NEG_INFINITY,
            "all-NEG_INF input must produce all-NEG_INF output"
        );
    }
}

// ---------------------------------------------------------------------------
// 12. Reconstruct tokens: path length equals node depth
// ---------------------------------------------------------------------------

/// Prove that `reconstruct_tokens` from a tree returns a path whose length
/// equals the node's depth (distance from root + 1). The root itself has
/// depth 1.
#[kani::unwind(6)]
#[kani::proof]
fn proof_reconstruct_tokens_path_depth_bound() {
    let len: usize = kani::any();
    kani::assume(len >= 1 && len <= 5);

    let mut tree: Vec<(Option<usize>, usize)> = Vec::with_capacity(len);
    let mut depths: Vec<usize> = Vec::with_capacity(len);
    tree.push((None, 0));
    depths.push(1); // root depth = 1

    for i in 1..len {
        let parent: usize = kani::any();
        kani::assume(parent < i);
        let token: usize = kani::any();
        kani::assume(token <= 100);
        tree.push((Some(parent), token));
        depths.push(depths[parent] + 1);
    }

    let node_idx: usize = kani::any();
    kani::assume(node_idx < len);

    let tokens = reconstruct_tokens(node_idx, &tree);
    assert_eq!(
        tokens.len(),
        depths[node_idx],
        "reconstructed path length must equal node depth"
    );
}

// ---------------------------------------------------------------------------
// 13. Score: length normalization effect
// ---------------------------------------------------------------------------

/// Prove that for the same negative `log_prob`, a longer hypothesis has a
/// higher (less negative) normalized score when `length_penalty > 0`.
/// This confirms the length penalty normalizes for sequence length,
/// preventing short sequences from dominating just because they accumulate
/// fewer negative log-prob terms.
#[kani::unwind(1)]
#[kani::proof]
fn proof_score_length_normalization_effect() {
    let log_prob: f64 = kani::any();
    kani::assume(log_prob.is_finite() && log_prob < -0.01 && log_prob > -1e4);

    // With penalty > 0, score = log_prob / len^penalty.
    // For negative log_prob: dividing by larger denominator makes it less negative.
    // short: score = log_prob / 1 = log_prob
    // long:  score = log_prob / 3 (with penalty=1), which is less negative.
    let short_hyp = BeamHypothesis::new(vec![1], log_prob, false);
    let long_hyp = BeamHypothesis::new(vec![1, 2, 3], log_prob, false);

    // Use penalty = 1.0 (deterministic, no stub needed).
    let short_score = short_hyp.score(1.0);
    let long_score = long_hyp.score(1.0);

    // long_score = log_prob / 3.0, short_score = log_prob / 1.0 = log_prob.
    // Since log_prob < 0: log_prob < log_prob / 3 (less negative).
    assert!(
        short_score <= long_score,
        "length normalization makes short score <= long score for same negative log_prob"
    );
}

// ---------------------------------------------------------------------------
// 14. BeamSearchOutput::new preserves beam count
// ---------------------------------------------------------------------------

/// Prove that `BeamSearchOutput::new` preserves the beam count and order.
#[kani::unwind(5)]
#[kani::proof]
fn proof_beam_search_output_new_preserves_count() {
    let num_beams: usize = kani::any();
    kani::assume(num_beams <= 4);

    let beams: Vec<BeamHypothesis> = (0..num_beams)
        .map(|i| BeamHypothesis::new(vec![i], -(i as f64), false))
        .collect();
    let output = BeamSearchOutput::new(beams);
    assert_eq!(
        output.beams.len(),
        num_beams,
        "BeamSearchOutput::new must preserve beam count"
    );
}
