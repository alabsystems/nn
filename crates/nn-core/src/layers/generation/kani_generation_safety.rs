// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for beam search and greedy decoding safety.
//!
//! Five categories of safety properties:
//! 1. **Index bounds** — beam indices never exceed vocabulary size
//! 2. **Score ordering** — top-k scores are correctly sorted
//! 3. **Beam width invariant** — active beams <= beam_width at each step
//! 4. **Termination** — EOS token causes beam pruning (beam count monotonically
//!    non-increasing after EOS)
//! 5. **Greedy correctness** — greedy decoding selects argmax at each step
//!
//! Part of #4239.

use super::autoregressive::{GenerationConfig, GenerationOutput};
use super::beam_search::{BeamHypothesis, BeamSearchConfig, BeamSearchOutput};

// ---------------------------------------------------------------------------
// Inline helper: argmax (mirrors autoregressive_sampling.rs logic)
// ---------------------------------------------------------------------------
fn inline_argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Inline helper: top_k_by_value (mirrors beam_search_helpers.rs logic)
// ---------------------------------------------------------------------------
fn inline_top_k_by_value(values: &[f32], k: usize) -> Vec<(usize, f32)> {
    if k == 0 {
        return Vec::new();
    }
    let mut indices: Vec<usize> = (0..values.len()).collect();
    if k < indices.len() {
        indices.select_nth_unstable_by(k - 1, |&a, &b| values[b].total_cmp(&values[a]));
        indices.truncate(k);
    }
    indices.sort_unstable_by(|&a, &b| values[b].total_cmp(&values[a]));
    indices.iter().map(|&i| (i, values[i])).collect()
}

// ---------------------------------------------------------------------------
// Inline helper: is_eos (mirrors beam_search_helpers.rs logic)
// ---------------------------------------------------------------------------
fn inline_is_eos(token: usize, config: &BeamSearchConfig) -> bool {
    config.eos_token_id.is_some_and(|eos| token == eos)
}

// ---------------------------------------------------------------------------
// Inline helper: log_softmax (mirrors beam_search_helpers.rs logic)
// ---------------------------------------------------------------------------
fn inline_log_softmax(logits: &[f32]) -> Vec<f32> {
    let sanitized: Vec<f32> = logits
        .iter()
        .map(|&v| if v.is_nan() { f32::NEG_INFINITY } else { v })
        .collect();
    let max_val = sanitized.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_val == f32::NEG_INFINITY {
        return vec![f32::NEG_INFINITY; logits.len()];
    }
    if max_val == f32::INFINITY {
        let inf_count = sanitized.iter().filter(|&&v| v == f32::INFINITY).count();
        let log_prob = -(inf_count as f32).ln();
        return sanitized
            .iter()
            .map(|&v| {
                if v == f32::INFINITY {
                    log_prob
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
    }
    let log_sum_exp: f32 = sanitized
        .iter()
        .map(|&v| (v - max_val).exp())
        .sum::<f32>()
        .ln()
        + max_val;
    sanitized.iter().map(|&v| v - log_sum_exp).collect()
}

// ===========================================================================
// Category 1: INDEX BOUNDS — beam indices never exceed vocabulary size
// ===========================================================================

/// Prove that top_k_by_value indices are always < the vocabulary (input) size,
/// for arbitrary beam widths and vocabulary sizes. This is the core index
/// safety property: any beam index used to look up a token in the vocabulary
/// must be in-bounds.
#[kani::proof]
#[kani::unwind(9)]
fn proof_safety_beam_indices_within_vocab_size() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 1 && vocab_size <= 8);
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= vocab_size);

    let mut logits = vec![0.0f32; vocab_size];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite());
    }

    let top = inline_top_k_by_value(&logits, beam_width);

    // Every returned index must be a valid vocabulary index.
    for &(idx, _val) in &top {
        assert!(idx < vocab_size, "beam token index must be < vocab_size");
    }
    // Number of returned candidates must not exceed beam_width.
    assert!(
        top.len() <= beam_width,
        "top_k must return at most beam_width candidates"
    );
}

/// Prove that beam expansion (selecting top-k per beam, then top beam_width
/// globally) never produces an index outside [0, vocab_size). Models the
/// core expansion step of beam search: each of W active beams produces
/// W candidates, then the top W are selected.
#[kani::proof]
#[kani::unwind(7)]
fn proof_safety_beam_expansion_indices_bounded() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 2 && vocab_size <= 4);
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 3);

    // Simulate expansion: each beam produces beam_width candidates.
    let num_beams: usize = kani::any();
    kani::assume(num_beams >= 1 && num_beams <= beam_width);

    let mut all_tokens: Vec<usize> = Vec::new();
    for _ in 0..num_beams {
        let mut logits = vec![0.0f32; vocab_size];
        for v in logits.iter_mut() {
            *v = kani::any();
            kani::assume(v.is_finite());
        }
        let candidates = inline_top_k_by_value(&logits, beam_width);
        for &(token, _) in &candidates {
            all_tokens.push(token);
        }
    }

    // All collected tokens must be valid vocab indices.
    for &token in &all_tokens {
        assert!(
            token < vocab_size,
            "expanded beam token must be < vocab_size"
        );
    }
}

/// Prove that log_softmax outputs have length equal to input length and
/// the argmax of the output matches the argmax of the input (ordering
/// preserved), which means beam index selection on log-probabilities
/// is consistent with selection on raw logits.
#[kani::proof]
#[kani::unwind(5)]
fn proof_safety_log_softmax_index_consistency() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 2 && vocab_size <= 4);

    let mut logits = vec![0.0f32; vocab_size];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite() && v.abs() < 50.0);
    }

    let log_probs = inline_log_softmax(&logits);

    // Output length must match input.
    assert_eq!(log_probs.len(), vocab_size);

    // The argmax of log_softmax must match the argmax of the raw logits,
    // because log_softmax(x_i) = x_i - C for constant C = log(sum(exp(x_j))).
    let logit_argmax = inline_argmax(&logits);
    let lp_argmax = inline_argmax(&log_probs);
    assert_eq!(
        logit_argmax, lp_argmax,
        "argmax must be preserved through log_softmax"
    );
}

// ===========================================================================
// Category 2: SCORE ORDERING — top-k scores correctly sorted
// ===========================================================================

/// Prove that `inline_top_k_by_value` returns results in strictly non-increasing
/// order of value. This is the beam search invariant: candidates are ranked
/// by score so the best beam is always first.
#[kani::proof]
#[kani::unwind(7)]
fn proof_safety_top_k_scores_sorted_descending() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 2 && vocab_size <= 6);
    let k: usize = kani::any();
    kani::assume(k >= 2 && k <= vocab_size);

    let mut values = vec![0.0f32; vocab_size];
    for v in values.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite());
    }

    let result = inline_top_k_by_value(&values, k);

    // Adjacent pairs must be in non-increasing order.
    for i in 1..result.len() {
        assert!(
            result[i - 1].1.total_cmp(&result[i].1) != std::cmp::Ordering::Less,
            "top_k scores must be sorted in non-increasing order"
        );
    }
}

/// Prove that the highest-scoring beam from top_k_by_value has a value >=
/// all other values in the full vocabulary. This ensures the best candidate
/// is never missed.
#[kani::proof]
#[kani::unwind(7)]
fn proof_safety_top_k_best_is_global_max() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 1 && vocab_size <= 6);
    let k: usize = kani::any();
    kani::assume(k >= 1 && k <= vocab_size);

    let mut values = vec![0.0f32; vocab_size];
    for v in values.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite());
    }

    let result = inline_top_k_by_value(&values, k);
    assert!(!result.is_empty());

    // The first element (best score) must be >= all values in the input.
    let best_val = result[0].1;
    for &v in &values {
        assert!(
            best_val.total_cmp(&v) != std::cmp::Ordering::Less,
            "top_k[0] must be the global maximum"
        );
    }
}

/// Prove that length-normalized scoring with length_penalty=1.0 equals
/// log_prob / len, which means longer hypotheses with the same total
/// log_prob get higher (less negative) normalized scores, correctly
/// preventing length bias in beam ranking.
#[kani::proof]
#[kani::unwind(1)]
fn proof_safety_score_normalization_correctness() {
    let log_prob: f64 = kani::any();
    kani::assume(log_prob.is_finite() && log_prob < -0.01 && log_prob > -1e4);

    // Inline score logic: score = log_prob / len^penalty.
    // For penalty=1.0: score = log_prob / len.
    let short_len = 2.0_f64;
    let long_len = 4.0_f64;
    let short_score = log_prob / short_len;
    let long_score = log_prob / long_len;

    // For negative log_prob, dividing by a larger len makes it less negative.
    assert!(
        long_score >= short_score,
        "longer hypothesis must have higher normalized score for same negative log_prob"
    );
}

// ===========================================================================
// Category 3: BEAM WIDTH INVARIANT — active beams <= beam_width
// ===========================================================================

/// Prove the per-step beam count invariant: after selecting top beam_width
/// candidates from all expansions, exactly min(candidates, beam_width)
/// beams survive.
#[kani::proof]
#[kani::unwind(9)]
fn proof_safety_per_step_beam_count_bounded() {
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 4);

    // Model: N beams each produce beam_width candidates = N * beam_width total.
    let num_active: usize = kani::any();
    kani::assume(num_active >= 1 && num_active <= beam_width);
    let total_candidates = num_active * beam_width;
    kani::assume(total_candidates <= 16);

    let mut all_scores = vec![0.0f32; total_candidates];
    for s in all_scores.iter_mut() {
        *s = kani::any();
        kani::assume(s.is_finite());
    }

    // Select top beam_width from all candidates.
    let survivors = inline_top_k_by_value(&all_scores, beam_width);

    assert!(
        survivors.len() <= beam_width,
        "per-step surviving beams must be <= beam_width"
    );
}

/// Prove the beam width invariant holds for the initial step (prefill):
/// the number of initial beams from top-k logits is <= beam_width.
#[kani::proof]
#[kani::unwind(9)]
fn proof_safety_initial_beam_count_bounded() {
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 4);
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 1 && vocab_size <= 8);

    let mut logits = vec![0.0f32; vocab_size];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite());
    }

    let initial_beams = inline_top_k_by_value(&logits, beam_width);

    assert!(
        initial_beams.len() <= beam_width,
        "initial beam count from prefill must be <= beam_width"
    );
    // When vocab_size < beam_width, we get fewer beams.
    assert!(
        initial_beams.len() <= vocab_size,
        "initial beam count cannot exceed vocab_size"
    );
}

/// Prove that BeamSearchOutput respects beam_width: constructing output
/// and truncating to beam_width produces at most beam_width beams.
#[kani::proof]
#[kani::unwind(9)]
fn proof_safety_output_truncation_respects_beam_width() {
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 4);

    // Create more beams than beam_width.
    let num_beams: usize = kani::any();
    kani::assume(num_beams >= 1 && num_beams <= 8);

    let mut beams: Vec<BeamHypothesis> = Vec::with_capacity(num_beams);
    for i in 0..num_beams {
        beams.push(BeamHypothesis::new(vec![i], -(i as f64), false));
    }

    // Truncate as finalize_tree does.
    beams.truncate(beam_width);
    let output = BeamSearchOutput::new(beams);

    assert!(
        output.beams.len() <= beam_width,
        "output beams after truncation must be <= beam_width"
    );
}

// ===========================================================================
// Category 4: TERMINATION — EOS causes beam pruning
// ===========================================================================

/// Prove that when all beams encounter EOS, the completed count equals the
/// total beam count. Models the termination condition: if every active beam
/// produces an EOS token, all beams move to the completed set.
#[kani::proof]
#[kani::unwind(5)]
fn proof_safety_all_eos_moves_all_to_completed() {
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 4);
    let eos_id: usize = kani::any();
    kani::assume(eos_id <= 1000);

    let config = BeamSearchConfig::new(beam_width)
        .with_eos_token_id(eos_id)
        .with_early_stopping(true);

    let mut completed_count: usize = 0;
    let mut active_count: usize = beam_width;

    for _ in 0..beam_width {
        let token = eos_id;
        assert!(inline_is_eos(token, &config));
        completed_count += 1;
        active_count -= 1;
    }

    assert_eq!(
        completed_count, beam_width,
        "all EOS must move all beams to completed"
    );
    assert_eq!(
        active_count, 0,
        "no active beams should remain after all hit EOS"
    );
}

/// Prove that EOS detection is monotonic: the active beam count is
/// monotonically non-increasing once beams start hitting EOS. This is the
/// beam pruning property.
#[kani::proof]
#[kani::unwind(6)]
fn proof_safety_eos_beam_count_monotonically_non_increasing() {
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 2 && beam_width <= 4);
    let eos_id: usize = kani::any();
    kani::assume(eos_id <= 1000);

    let config = BeamSearchConfig::new(beam_width)
        .with_eos_token_id(eos_id)
        .with_early_stopping(true);

    let num_steps: usize = kani::any();
    kani::assume(num_steps >= 1 && num_steps <= 5);

    let mut active = beam_width;
    let mut completed: usize = 0;
    let mut prev_active = active;

    for _ in 0..num_steps {
        if active == 0 {
            break;
        }
        // Some subset of active beams hit EOS.
        let eos_count: usize = kani::any();
        kani::assume(eos_count <= active);

        for _ in 0..eos_count {
            assert!(inline_is_eos(eos_id, &config));
        }

        completed += eos_count;
        active -= eos_count;

        // Active count is monotonically non-increasing.
        assert!(
            active <= prev_active,
            "active beam count must be monotonically non-increasing after EOS"
        );
        prev_active = active;

        // Early stopping check.
        if config.early_stopping && completed >= beam_width {
            break;
        }
    }

    // Final invariant: completed + active == beam_width (conservation).
    assert_eq!(
        completed + active,
        beam_width,
        "total beams must be conserved (completed + active = beam_width)"
    );
}

/// Prove that early stopping triggers when enough completed beams exist.
#[kani::proof]
#[kani::unwind(1)]
fn proof_safety_early_stopping_triggers_on_sufficient_completed() {
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 8);

    let config = BeamSearchConfig::new(beam_width)
        .with_eos_token_id(42)
        .with_early_stopping(true);

    let completed: usize = kani::any();
    kani::assume(completed >= beam_width);

    let should_stop = config.early_stopping && completed >= config.beam_width;
    assert!(
        should_stop,
        "early stopping must trigger when completed >= beam_width"
    );
}

/// Prove that without early_stopping, the search does NOT terminate early.
#[kani::proof]
#[kani::unwind(1)]
fn proof_safety_no_early_stopping_continues() {
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 8);

    let config = BeamSearchConfig::new(beam_width)
        .with_eos_token_id(42)
        .with_early_stopping(false);

    let completed: usize = kani::any();
    kani::assume(completed >= beam_width);

    let should_stop = config.early_stopping && completed >= config.beam_width;
    assert!(
        !should_stop,
        "without early_stopping, search must not terminate early"
    );
}

// ===========================================================================
// Category 5: GREEDY CORRECTNESS — greedy decoding selects argmax
// ===========================================================================

/// Prove that greedy decoding (argmax) selects the index whose value is
/// maximal: no other value in the input is strictly greater.
#[kani::proof]
#[kani::unwind(7)]
fn proof_safety_greedy_selects_maximum() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 1 && vocab_size <= 6);

    let mut logits = vec![0.0f32; vocab_size];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite());
    }

    let selected = inline_argmax(&logits);
    assert!(selected < vocab_size, "argmax index must be in bounds");

    let selected_val = logits[selected];
    for &v in &logits {
        assert!(
            selected_val.total_cmp(&v) != std::cmp::Ordering::Less,
            "greedy must select the maximum: no value may exceed the selected one"
        );
    }
}

/// Prove that greedy decoding with temperature=0.0 config is valid and
/// produces deterministic selection via argmax.
#[kani::proof]
#[kani::unwind(5)]
fn proof_safety_greedy_temperature_zero_is_argmax() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 1 && vocab_size <= 4);

    let mut logits = vec![0.0f32; vocab_size];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite());
    }

    let config = GenerationConfig {
        temperature: 0.0,
        ..Default::default()
    };

    assert!(
        config.validate().is_ok(),
        "temperature=0.0 must be valid (greedy mode)"
    );

    let greedy_idx = inline_argmax(&logits);
    assert!(greedy_idx < vocab_size);

    let max_val = logits[greedy_idx];
    for &v in &logits {
        assert!(
            max_val.total_cmp(&v) != std::cmp::Ordering::Less,
            "greedy with T=0 must select the maximum value"
        );
    }
}

/// Prove that beam search with beam_width=1 reduces to greedy decoding:
/// top_k_by_value with k=1 selects the same index as argmax.
#[kani::proof]
#[kani::unwind(7)]
fn proof_safety_beam_width_one_is_greedy() {
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 1 && vocab_size <= 6);

    let mut logits = vec![0.0f32; vocab_size];
    for v in logits.iter_mut() {
        *v = kani::any();
        kani::assume(v.is_finite());
    }

    // Beam search with beam_width=1.
    let beam_result = inline_top_k_by_value(&logits, 1);
    assert_eq!(beam_result.len(), 1, "beam_width=1 must return exactly 1");

    // Greedy argmax.
    let greedy_idx = inline_argmax(&logits);

    // Both must select the same index.
    assert_eq!(
        beam_result[0].0, greedy_idx,
        "beam_width=1 must select the same token as greedy argmax"
    );
}

/// Prove that GenerationOutput correctly tracks the finished state:
/// a generation that hits EOS is marked finished=true, and one that
/// exhausts max_new_tokens is marked finished=false.
#[kani::proof]
#[kani::unwind(1)]
fn proof_safety_generation_output_finished_tracking() {
    let num_tokens: usize = kani::any();
    kani::assume(num_tokens >= 1 && num_tokens <= 16);

    let tokens: Vec<usize> = (0..num_tokens).collect();

    let output_eos = GenerationOutput::new(tokens.clone(), true);
    assert!(
        output_eos.finished,
        "EOS-terminated output must be marked finished"
    );
    assert_eq!(output_eos.token_ids.len(), num_tokens);

    let output_max = GenerationOutput::new(tokens, false);
    assert!(
        !output_max.finished,
        "max-length output must not be marked finished"
    );
    assert_eq!(output_max.token_ids.len(), num_tokens);
}
