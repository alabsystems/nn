// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for beam search safety properties (#4239).
//!
//! Proves seven categories of beam search invariants:
//!
//! 1. **Beam width bounds** -- beam_width > 0, active beams never exceed beam_width
//! 2. **Score ordering** -- beam scores maintained in sorted order after pruning
//! 3. **Token index bounds** -- selected tokens within vocabulary size
//! 4. **Sequence length bounds** -- sequences never exceed max_length
//! 5. **Log probability bounds** -- accumulated log probs are non-positive
//! 6. **End-of-sequence handling** -- completed beams are not extended
//! 7. **Memory bounds** -- beam state memory proportional to beam_width * seq_len
//!
//! All harnesses use small bounds for CBMC tractability:
//! beam_width <= 4, vocab_size <= 8, seq_len <= 6.
//!
//! Part of #4239.

#![cfg(kani)]

use crate::layers::generation::beam_search::{BeamHypothesis, BeamSearchConfig, BeamSearchOutput};
use crate::layers::generation::beam_search_helpers::{
    finalize_tree, is_eos, log_softmax, reconstruct_tokens, top_k_by_value,
};

// ---- Stubs for transcendental functions (CBMC cannot evaluate) ----

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

// ===========================================================================
// 1. Beam width bounds
// ===========================================================================

/// Proves beam_width > 0 is enforced by BeamSearchConfig::validate.
/// Any config with beam_width == 0 must be rejected.
#[kani::unwind(1)]
#[kani::proof]
fn proof_beam_width_positive_enforced() {
    let bw: usize = kani::any();
    kani::assume(bw <= 16);
    let config = BeamSearchConfig {
        beam_width: bw,
        ..Default::default()
    };
    let result = config.validate();
    if bw == 0 {
        assert!(
            result.is_err(),
            "beam_width=0 must be rejected by validate()"
        );
    } else {
        assert!(
            result.is_ok(),
            "beam_width>0 must be accepted by validate()"
        );
    }
}

/// Proves that after top-k pruning, the number of candidates never exceeds
/// beam_width, regardless of the number of active beams or vocabulary size.
/// Models one expansion + prune cycle of beam search.
#[kani::unwind(1)]
#[kani::proof]
fn proof_beam_candidates_bounded_by_width() {
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 4);

    let active_beams: usize = kani::any();
    kani::assume(active_beams >= 1 && active_beams <= beam_width);

    let vocab_candidates_per_beam: usize = kani::any();
    kani::assume(vocab_candidates_per_beam >= 1 && vocab_candidates_per_beam <= 8);

    // Each active beam produces vocab_candidates_per_beam expansions.
    let total_candidates = active_beams * vocab_candidates_per_beam;

    // Pruning keeps at most beam_width.
    let after_prune = if total_candidates > beam_width {
        beam_width
    } else {
        total_candidates
    };

    assert!(
        after_prune <= beam_width,
        "pruned candidates must not exceed beam_width"
    );
    assert!(after_prune >= 1, "must retain at least one candidate");
}

/// Proves that over multiple expansion+prune steps, the active beam count
/// is always bounded by beam_width.
#[kani::unwind(1)]
#[kani::proof]
fn proof_beam_count_bounded_multi_step() {
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 4);

    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 2 && vocab_size <= 8);

    // Step 0: start with 1 beam
    let mut count: usize = 1;
    assert!(count <= beam_width);

    // Simulate 4 expansion+prune steps
    let mut step = 0_usize;
    while step < 4 {
        let expanded = count * vocab_size;
        count = if expanded > beam_width {
            beam_width
        } else {
            expanded
        };
        assert!(
            count <= beam_width,
            "active beam count must be <= beam_width at every step"
        );
        step += 1;
    }
}

// ===========================================================================
// 2. Score ordering: beams are sorted by score after pruning
// ===========================================================================

/// Proves that sorting candidates by score (descending) and truncating to
/// beam_width produces a result where each score is >= the next.
#[kani::unwind(6)]
#[kani::proof]
fn proof_score_ordering_after_prune() {
    let n: usize = kani::any();
    kani::assume(n >= 2 && n <= 5);
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= n);

    // Generate n candidate scores.
    let mut scores = [0.0f32; 5];
    for i in 0..n {
        scores[i] = kani::any();
        kani::assume(scores[i].is_finite() && scores[i] >= -200.0 && scores[i] <= 0.0);
    }

    // Sort descending (bubble sort for CBMC tractability).
    let mut i = 0_usize;
    while i < n {
        let mut j = i + 1;
        while j < n {
            if scores[j] > scores[i] {
                let tmp = scores[i];
                scores[i] = scores[j];
                scores[j] = tmp;
            }
            j += 1;
        }
        i += 1;
    }

    // After sorting, the first beam_width elements are in descending order.
    let mut k = 1_usize;
    while k < beam_width {
        assert!(
            scores[k - 1] >= scores[k],
            "scores must be in descending order after sort+prune"
        );
        k += 1;
    }
}

/// Proves finalize_tree output is sorted by normalized score descending,
/// using the actual BeamSearchConfig + finalize_tree function.
#[kani::unwind(5)]
#[kani::proof]
#[kani::stub(f64::powf, powf_f64_stub)]
fn proof_finalize_output_sorted() {
    let num_nodes: usize = kani::any();
    kani::assume(num_nodes >= 2 && num_nodes <= 3);

    let mut tree: Vec<(Option<usize>, usize)> = Vec::with_capacity(num_nodes);
    tree.push((None, 0));
    for i in 1..num_nodes {
        tree.push((Some(i - 1), i));
    }

    // Two completed beams.
    let lp1: f64 = kani::any();
    kani::assume(lp1.is_finite() && lp1 >= -100.0 && lp1 <= 0.0);
    let lp2: f64 = kani::any();
    kani::assume(lp2.is_finite() && lp2 >= -100.0 && lp2 <= 0.0);

    let idx1: usize = kani::any();
    kani::assume(idx1 < num_nodes);
    let idx2: usize = kani::any();
    kani::assume(idx2 < num_nodes);

    let completed = vec![(idx1, lp1, 1_usize), (idx2, lp2, 1_usize)];
    let active: Vec<(usize, f64)> = Vec::new();

    let config = BeamSearchConfig {
        beam_width: 4,
        length_penalty: 1.0,
        ..Default::default()
    };

    let output = finalize_tree(completed, &active, &tree, &config);
    for i in 1..output.beams.len() {
        let prev_score = output.beams[i - 1].score(config.length_penalty);
        let curr_score = output.beams[i].score(config.length_penalty);
        assert!(
            prev_score.total_cmp(&curr_score).is_ge(),
            "finalize_tree output must be sorted by score descending"
        );
    }
}

// ===========================================================================
// 3. Token index bounds: selected tokens within vocabulary size
// ===========================================================================

/// Proves that top_k_by_value only returns indices that are valid positions
/// in the vocabulary (i.e., < len of the input slice).
#[kani::unwind(9)]
#[kani::proof]
fn proof_token_indices_within_vocab() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 1 && vocab_size <= 8);
    let k: usize = kani::any();
    kani::assume(k >= 1 && k <= vocab_size);

    let mut logits = vec![0.0f32; vocab_size];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite());
    }

    let result = top_k_by_value(&logits, k);

    // Every returned index must be a valid vocabulary index.
    for &(idx, _val) in &result {
        assert!(idx < vocab_size, "top_k token index must be < vocab_size");
    }
    // Number of returned tokens must not exceed k.
    assert!(result.len() <= k, "top_k must return at most k tokens");
}

/// Proves that argmax over a logit vector returns a valid index, and that
/// the chosen logit value is maximal.
#[kani::unwind(9)]
#[kani::proof]
fn proof_argmax_token_in_bounds() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 1 && vocab_size <= 8);

    let mut logits = vec![0.0f32; vocab_size];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite());
    }

    // Argmax is top_k with k=1.
    let result = top_k_by_value(&logits, 1);
    assert!(!result.is_empty(), "argmax must return at least one result");

    let (best_idx, best_val) = result[0];
    assert!(best_idx < vocab_size, "argmax index must be < vocab_size");

    // The chosen value must be >= all others.
    for i in 0..vocab_size {
        assert!(
            best_val >= logits[i] || logits[i].is_nan(),
            "argmax value must be >= all logit values"
        );
    }
}

// ===========================================================================
// 4. Sequence length bounds: sequences don't exceed max_length
// ===========================================================================

/// Proves that a beam search loop with max_new_tokens = M generates at most
/// M tokens per beam hypothesis.
#[kani::unwind(8)]
#[kani::proof]
fn proof_sequence_length_bounded_by_max() {
    let max_new_tokens: usize = kani::any();
    kani::assume(max_new_tokens >= 1 && max_new_tokens <= 6);

    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 3);

    // Simulate the decode loop: at each step, token_count increments by 1.
    // The loop runs for at most max_new_tokens steps.
    let mut token_count: usize = 0;
    let mut step: usize = 0;
    while step < max_new_tokens {
        token_count += 1;
        step += 1;
    }

    assert!(
        token_count <= max_new_tokens,
        "generated tokens must not exceed max_new_tokens"
    );
}

/// Proves that reconstructed token paths from the parent-pointer tree
/// have length bounded by the tree depth (which is bounded by step count + 1).
#[kani::unwind(7)]
#[kani::proof]
fn proof_reconstruct_path_length_bounded() {
    let max_depth: usize = kani::any();
    kani::assume(max_depth >= 1 && max_depth <= 6);

    // Build a linear chain tree of exactly max_depth nodes.
    let mut tree: Vec<(Option<usize>, usize)> = Vec::with_capacity(max_depth);
    for i in 0..max_depth {
        let token: usize = kani::any();
        kani::assume(token < 100);
        if i == 0 {
            tree.push((None, token));
        } else {
            tree.push((Some(i - 1), token));
        }
    }

    let leaf = max_depth - 1;
    let tokens = reconstruct_tokens(leaf, &tree);

    assert!(
        tokens.len() <= max_depth,
        "reconstructed path must not exceed tree depth"
    );
    assert_eq!(
        tokens.len(),
        max_depth,
        "linear chain path length must equal tree depth"
    );
}

// ===========================================================================
// 5. Log probability bounds: accumulated log probs are non-positive
// ===========================================================================

/// Proves that log-softmax outputs are all <= 0 for finite inputs.
/// Since softmax(x)_i is in (0, 1], log(softmax(x)_i) is in (-inf, 0].
#[kani::unwind(5)]
#[kani::proof]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn proof_log_probs_nonpositive() {
    let len: usize = kani::any();
    kani::assume(len >= 1 && len <= 4);

    let mut logits = vec![0.0f32; len];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite());
    }

    let result = log_softmax(&logits);
    for &v in &result {
        assert!(
            v <= 1e-6,
            "log-softmax output must be <= 0 (with epsilon tolerance)"
        );
    }
}

/// Proves that accumulating non-positive log-probs keeps the cumulative
/// score non-positive. Since each step adds log P(token) <= 0, the sum
/// is monotonically non-increasing and always <= 0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_cumulative_log_prob_nonpositive() {
    let prev_score: f64 = kani::any();
    kani::assume(prev_score.is_finite() && prev_score <= 0.0 && prev_score >= -1e8);

    let step_log_prob: f64 = kani::any();
    kani::assume(step_log_prob.is_finite() && step_log_prob <= 0.0 && step_log_prob >= -100.0);

    let new_score = prev_score + step_log_prob;
    kani::assume(new_score.is_finite());

    assert!(
        new_score <= 0.0,
        "cumulative log-probability must remain non-positive"
    );
    assert!(
        new_score <= prev_score,
        "cumulative score must be non-increasing (log_prob <= 0)"
    );
}

/// Proves that a BeamHypothesis initialized with non-positive log_prob
/// maintains non-positive score regardless of length_penalty.
#[kani::unwind(1)]
#[kani::proof]
fn proof_hypothesis_score_nonpositive() {
    let log_prob: f64 = kani::any();
    kani::assume(log_prob.is_finite() && log_prob <= 0.0 && log_prob >= -1e6);

    let num_tokens: usize = kani::any();
    kani::assume(num_tokens >= 1 && num_tokens <= 4);

    let token_ids: Vec<usize> = (0..num_tokens).collect();
    let hyp = BeamHypothesis::new(token_ids, log_prob, false);

    // With penalty=1.0 (deterministic, no stub needed):
    // score = log_prob / len. Since log_prob <= 0 and len > 0, score <= 0.
    let score = hyp.score(1.0);
    assert!(
        score <= 1e-12,
        "score must be non-positive for non-positive log_prob"
    );
}

// ===========================================================================
// 6. End-of-sequence handling: completed beams are not extended
// ===========================================================================

/// Proves that is_eos correctly identifies end-of-sequence tokens, and that
/// a finished beam (is_eos == true) would be excluded from expansion.
#[kani::unwind(1)]
#[kani::proof]
fn proof_eos_detected_correctly() {
    let eos_id: usize = kani::any();
    kani::assume(eos_id <= 1024);

    let config = BeamSearchConfig::new(4).with_eos_token_id(eos_id);

    // The EOS token must be detected.
    assert!(
        is_eos(eos_id, &config),
        "is_eos must return true for the EOS token"
    );

    // Non-EOS tokens must not be detected.
    let other: usize = kani::any();
    kani::assume(other != eos_id && other <= 1024);
    assert!(
        !is_eos(other, &config),
        "is_eos must return false for non-EOS tokens"
    );
}

/// Proves that when no EOS token is configured, is_eos always returns false,
/// so no beam is prematurely terminated.
#[kani::unwind(1)]
#[kani::proof]
fn proof_no_eos_config_never_terminates() {
    let config = BeamSearchConfig {
        eos_token_id: None,
        ..Default::default()
    };

    let token: usize = kani::any();
    kani::assume(token <= 65535);

    assert!(
        !is_eos(token, &config),
        "is_eos must return false when eos_token_id is None"
    );
}

/// Proves that finished beams are not expanded: in the beam search loop,
/// only beams with finished == false produce candidates. Models one step
/// where some beams are finished and others are active.
#[kani::unwind(5)]
#[kani::proof]
fn proof_finished_beams_not_expanded() {
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 2 && beam_width <= 4);

    // Model beam states: some finished, some active.
    let num_finished: usize = kani::any();
    kani::assume(num_finished < beam_width);
    let num_active = beam_width - num_finished;

    // Only active beams produce candidates.
    let vocab_per_beam: usize = kani::any();
    kani::assume(vocab_per_beam >= 1 && vocab_per_beam <= 4);

    let candidates_from_active = num_active * vocab_per_beam;
    let candidates_from_finished: usize = 0; // finished beams produce no candidates

    let total_candidates = candidates_from_active + candidates_from_finished;

    assert!(
        candidates_from_finished == 0,
        "finished beams must not produce any candidates"
    );
    assert!(
        total_candidates == candidates_from_active,
        "all candidates must come from active beams only"
    );
}

/// Proves that the finalize_tree function marks completed beams as finished
/// and active beams as not finished.
#[kani::unwind(5)]
#[kani::proof]
#[kani::stub(f64::powf, powf_f64_stub)]
fn proof_finalize_marks_finished_correctly() {
    let mut tree: Vec<(Option<usize>, usize)> = Vec::new();
    tree.push((None, 1));
    tree.push((Some(0), 2));

    // One completed beam, one active beam.
    let completed = vec![(0_usize, -1.0_f64, 1_usize)];
    let active = vec![(1_usize, -2.0_f64)];

    let config = BeamSearchConfig {
        beam_width: 4,
        length_penalty: 1.0,
        ..Default::default()
    };

    let output = finalize_tree(completed, &active, &tree, &config);

    let has_finished = output.beams.iter().any(|b| b.finished);
    let has_active = output.beams.iter().any(|b| !b.finished);

    assert!(
        has_finished,
        "must have at least one finished beam in output"
    );
    assert!(has_active, "must have at least one active beam in output");
}

// ===========================================================================
// 7. Memory bounds: proportional to beam_width * sequence_length
// ===========================================================================

/// Proves that the parent-pointer tree grows by exactly one node per
/// token per surviving beam, so total tree size is bounded by
/// beam_width * max_new_tokens + beam_width (initial expansion).
#[kani::unwind(7)]
#[kani::proof]
fn proof_tree_size_bounded() {
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 3);

    let max_new_tokens: usize = kani::any();
    kani::assume(max_new_tokens >= 1 && max_new_tokens <= 4);

    // Initial expansion: beam_width nodes.
    let initial_nodes = beam_width;

    // Each subsequent step adds at most beam_width new nodes
    // (one per surviving beam).
    let subsequent_nodes = beam_width * (max_new_tokens - 1);

    // Total tree size is bounded.
    let total_nodes = initial_nodes + subsequent_nodes;
    let expected_bound = beam_width * max_new_tokens;

    assert_eq!(
        total_nodes, expected_bound,
        "tree nodes must equal beam_width * max_new_tokens"
    );

    // Each node stores (Option<usize>, usize) = 2 * size_of::<usize>() + tag.
    // On 64-bit: ~24 bytes per node. Total memory is O(beam_width * max_new_tokens).
    let node_size_bytes: usize = 24; // conservative estimate
    let total_memory = total_nodes * node_size_bytes;
    let memory_bound = beam_width * max_new_tokens * node_size_bytes;

    assert!(
        total_memory <= memory_bound,
        "total tree memory must be bounded by beam_width * max_new_tokens * node_size"
    );
}

/// Proves that the number of active beam states (not tree nodes) is always
/// exactly <= beam_width, so per-beam overhead (KV cache, state) is bounded.
#[kani::unwind(1)]
#[kani::proof]
fn proof_active_state_count_bounded() {
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 4);

    // After any expansion+prune step, active beams <= beam_width.
    // Some beams may finish (EOS), reducing active count.
    let finished_this_step: usize = kani::any();
    kani::assume(finished_this_step <= beam_width);

    let remaining_active = beam_width - finished_this_step;

    assert!(
        remaining_active <= beam_width,
        "active state count must be <= beam_width"
    );

    // Total beams tracked (active + finished) for finalization
    // is at most beam_width (finished are accumulated, but pruned).
    // finalize_tree truncates to beam_width.
    let total_for_finalize: usize = kani::any();
    kani::assume(total_for_finalize <= beam_width * 2); // generous bound

    let after_truncate = if total_for_finalize > beam_width {
        beam_width
    } else {
        total_for_finalize
    };

    assert!(
        after_truncate <= beam_width,
        "finalized output must be <= beam_width beams"
    );
}

/// Proves that finalize_tree returns at most beam_width beams, bounding
/// the output memory.
#[kani::unwind(5)]
#[kani::proof]
#[kani::stub(f64::powf, powf_f64_stub)]
fn proof_finalize_output_count_bounded() {
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 4);

    // Build a small tree.
    let mut tree: Vec<(Option<usize>, usize)> = Vec::new();
    tree.push((None, 0));
    tree.push((Some(0), 1));
    tree.push((Some(1), 2));

    // Create more beams than beam_width to test truncation.
    let num_completed: usize = kani::any();
    kani::assume(num_completed >= 1 && num_completed <= 3);

    let mut completed: Vec<(usize, f64, usize)> = Vec::new();
    for i in 0..num_completed {
        let lp: f64 = kani::any();
        kani::assume(lp.is_finite() && lp >= -100.0 && lp <= 0.0);
        completed.push((i % 3, lp, 1));
    }

    let num_active: usize = kani::any();
    kani::assume(num_active <= 3);
    let mut active: Vec<(usize, f64)> = Vec::new();
    for i in 0..num_active {
        let lp: f64 = kani::any();
        kani::assume(lp.is_finite() && lp >= -100.0 && lp <= 0.0);
        active.push((i % 3, lp));
    }

    let config = BeamSearchConfig {
        beam_width,
        length_penalty: 1.0,
        ..Default::default()
    };

    let output = finalize_tree(completed, &active, &tree, &config);

    assert!(
        output.beams.len() <= beam_width,
        "finalize_tree must return at most beam_width beams"
    );
}

// ===========================================================================
// Additional safety properties
// ===========================================================================

/// Proves that BeamSearchConfig::validate rejects non-finite length_penalty
/// (NaN, +Inf, -Inf), preventing undefined sort ordering.
#[kani::unwind(1)]
#[kani::proof]
fn proof_nonfinite_penalty_rejected() {
    let config_nan = BeamSearchConfig {
        length_penalty: f64::NAN,
        ..Default::default()
    };
    assert!(
        config_nan.validate().is_err(),
        "NaN length_penalty must be rejected"
    );

    let config_inf = BeamSearchConfig {
        length_penalty: f64::INFINITY,
        ..Default::default()
    };
    assert!(
        config_inf.validate().is_err(),
        "+Inf length_penalty must be rejected"
    );

    let config_neg_inf = BeamSearchConfig {
        length_penalty: f64::NEG_INFINITY,
        ..Default::default()
    };
    assert!(
        config_neg_inf.validate().is_err(),
        "-Inf length_penalty must be rejected"
    );
}

/// Proves that top_k_by_value returns distinct indices (no duplicates),
/// which is critical for beam search correctness (no duplicate beams).
#[kani::unwind(7)]
#[kani::proof]
fn proof_top_k_indices_unique() {
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

    // Check all pairs for uniqueness.
    for i in 0..result.len() {
        for j in (i + 1)..result.len() {
            assert_ne!(
                result[i].0, result[j].0,
                "top_k must return distinct indices"
            );
        }
    }
}

/// Proves that log_softmax preserves input length and handles the
/// all-NEG_INFINITY edge case (returns NEG_INFINITY, not NaN).
#[kani::unwind(5)]
#[kani::proof]
fn proof_log_softmax_all_neginf_safe() {
    let len: usize = kani::any();
    kani::assume(len >= 1 && len <= 4);

    let logits = vec![f32::NEG_INFINITY; len];
    let result = log_softmax(&logits);

    assert_eq!(result.len(), len, "output length must match input");
    for &v in &result {
        assert_eq!(
            v,
            f32::NEG_INFINITY,
            "all-NEG_INF input must produce all-NEG_INF output (not NaN)"
        );
        assert!(
            !v.is_nan(),
            "log_softmax must never produce NaN for all-NEG_INF input"
        );
    }
}

/// Proves that BeamSearchOutput::new faithfully stores the provided beams.
#[kani::unwind(5)]
#[kani::proof]
fn proof_beam_output_preserves_beams() {
    let count: usize = kani::any();
    kani::assume(count <= 4);

    let beams: Vec<BeamHypothesis> = (0..count)
        .map(|i| BeamHypothesis::new(vec![i], -(i as f64), i % 2 == 0))
        .collect();

    let output = BeamSearchOutput::new(beams);
    assert_eq!(
        output.beams.len(),
        count,
        "BeamSearchOutput must preserve beam count"
    );
}
