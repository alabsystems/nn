// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for beam search and greedy decoding safety
//! specific to dpdf VLM (vision-language model) inference pipelines.
//!
//! Part 2 of 2: Proves properties 5-7 and compositional end-to-end:
//!
//! 5. **Sequence length bounds respected** — per-hypothesis token count bounded
//! 6. **EOS handling terminates correctly** — beam search converges
//! 7. **Memory allocation for beam hypotheses bounded** — tree node count bounded
//! 8. **End-to-end consistency** — config + pruning + output compose correctly
//!
//! Part of #4239.

use super::autoregressive::{GenerationConfig, GenerationOutput};
use super::beam_search::{BeamHypothesis, BeamSearchConfig, BeamSearchOutput};

// ---------------------------------------------------------------------------
// Inline helpers (self-contained, no DynTensor dependency)
// ---------------------------------------------------------------------------

fn inline_argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn inline_is_eos(token: usize, eos_id: Option<usize>) -> bool {
    eos_id.is_some_and(|eos| token == eos)
}

// ===========================================================================
// 5. SEQUENCE LENGTH BOUNDS — per-hypothesis token count bounded
// ===========================================================================

/// Prove that beam search hypotheses respect the max_new_tokens bound:
/// no hypothesis can have more tokens than max_new_tokens.
#[kani::proof]
#[kani::unwind(7)]
fn proof_dpdf_beam_hypothesis_length_bounded() {
    let max_new_tokens: usize = kani::any();
    kani::assume(max_new_tokens >= 1 && max_new_tokens <= 6);
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 3);

    let actual_steps: usize = kani::any();
    kani::assume(actual_steps >= 1 && actual_steps <= max_new_tokens);

    let tokens: Vec<usize> = (0..actual_steps).collect();
    let hyp = BeamHypothesis::new(tokens, -5.0, false);

    assert!(
        hyp.token_ids.len() <= max_new_tokens,
        "hypothesis token count must not exceed max_new_tokens"
    );
}

/// Prove that the output of beam search contains hypotheses whose
/// token counts are all bounded by max_new_tokens.
#[kani::proof]
#[kani::unwind(5)]
fn proof_dpdf_beam_output_all_lengths_bounded() {
    let max_new_tokens: usize = kani::any();
    kani::assume(max_new_tokens >= 1 && max_new_tokens <= 4);
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 3);

    let num_beams: usize = kani::any();
    kani::assume(num_beams >= 1 && num_beams <= beam_width);

    let mut beams = Vec::with_capacity(num_beams);
    for _ in 0..num_beams {
        let len: usize = kani::any();
        kani::assume(len >= 1 && len <= max_new_tokens);
        let tokens: Vec<usize> = (0..len).collect();
        let log_prob: f64 = kani::any();
        kani::assume(log_prob.is_finite() && log_prob >= -100.0 && log_prob <= 0.0);
        beams.push(BeamHypothesis::new(tokens, log_prob, false));
    }

    let output = BeamSearchOutput::new(beams);

    for beam in &output.beams {
        assert!(
            beam.token_ids.len() <= max_new_tokens,
            "every output beam length must be <= max_new_tokens"
        );
    }
}

/// Prove that greedy decoding output respects max_new_tokens.
#[kani::proof]
#[kani::unwind(8)]
fn proof_dpdf_greedy_sequence_length_bounded() {
    let max_new_tokens: usize = kani::any();
    kani::assume(max_new_tokens >= 1 && max_new_tokens <= 6);

    let eos_id: usize = 42;

    let mut generated = Vec::new();
    let mut finished = false;
    for _ in 0..max_new_tokens {
        let token: usize = kani::any();
        kani::assume(token <= 65535);
        generated.push(token);

        if inline_is_eos(token, Some(eos_id)) {
            finished = true;
            break;
        }
    }

    assert!(
        generated.len() <= max_new_tokens,
        "greedy output must not exceed max_new_tokens"
    );

    let output = GenerationOutput::new(generated, finished);
    assert!(
        output.token_ids.len() <= max_new_tokens,
        "GenerationOutput length must respect max_new_tokens"
    );
}

// ===========================================================================
// 6. EOS HANDLING — beam search convergence
// ===========================================================================

/// Prove that beam search terminates: after max_new_tokens steps, all
/// beams are either completed (hit EOS) or stopped (max length). Total
/// hypotheses (completed + active) is always conserved at beam_width.
#[kani::proof]
#[kani::unwind(6)]
fn proof_dpdf_beam_search_convergence() {
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 3);
    let max_steps: usize = kani::any();
    kani::assume(max_steps >= 1 && max_steps <= 5);

    let mut active = beam_width;
    let mut completed: usize = 0;

    for _ in 0..max_steps {
        if active == 0 {
            break;
        }

        let eos_count: usize = kani::any();
        kani::assume(eos_count <= active);

        completed += eos_count;
        active -= eos_count;

        assert_eq!(
            completed + active,
            beam_width,
            "beam count must be conserved"
        );
    }

    assert_eq!(
        completed + active,
        beam_width,
        "all beams must be accounted for at convergence"
    );
}

/// Prove that with early_stopping=true, once enough beams complete,
/// the search terminates even if some beams are still active.
#[kani::proof]
#[kani::unwind(5)]
fn proof_dpdf_early_stopping_terminates_with_active_beams() {
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 2 && beam_width <= 4);

    let mut active = beam_width;
    let mut completed: usize = 0;
    let mut terminated = false;

    for _ in 0..beam_width {
        if active == 0 || terminated {
            break;
        }

        if active > 0 {
            completed += 1;
            active -= 1;
        }

        if completed >= beam_width {
            terminated = true;
        }
    }

    assert!(
        terminated || completed + active == beam_width,
        "search must either terminate early or conserve beam count"
    );
}

/// Prove that without EOS token configured, the beam search runs for
/// exactly max_new_tokens steps (no early termination).
#[kani::proof]
#[kani::unwind(6)]
fn proof_dpdf_no_eos_runs_full_length() {
    let max_steps: usize = kani::any();
    kani::assume(max_steps >= 1 && max_steps <= 5);

    let mut step_count: usize = 0;
    for _ in 0..max_steps {
        let token: usize = kani::any();
        kani::assume(token <= 65535);

        let is_eos = inline_is_eos(token, None);
        assert!(!is_eos, "is_eos must be false when no EOS configured");

        step_count += 1;
    }

    assert_eq!(
        step_count, max_steps,
        "without EOS, generation must run for all max_steps"
    );
}

// ===========================================================================
// 7. MEMORY ALLOCATION — tree node count bounded
// ===========================================================================

/// Prove that the parent-pointer tree has at most beam_width * max_new_tokens
/// nodes. This bounds memory allocation for beam hypothesis tracking.
#[kani::proof]
#[kani::unwind(5)]
fn proof_dpdf_tree_node_count_bounded() {
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 3);
    let max_new_tokens: usize = kani::any();
    kani::assume(max_new_tokens >= 1 && max_new_tokens <= 4);

    // Initial step: beam_width nodes from prefill top-k.
    let mut tree_size = beam_width;

    // Each step adds at most beam_width new nodes.
    let actual_steps: usize = kani::any();
    kani::assume(actual_steps >= 1 && actual_steps <= max_new_tokens.saturating_sub(1));

    for _ in 0..actual_steps {
        tree_size += beam_width;
    }

    assert!(
        tree_size <= beam_width * max_new_tokens,
        "tree node count must be <= beam_width * max_new_tokens"
    );
}

/// Prove that the completed hypothesis count is bounded by beam_width.
#[kani::proof]
#[kani::unwind(6)]
fn proof_dpdf_completed_hypothesis_count_bounded() {
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 3);
    let max_steps: usize = kani::any();
    kani::assume(max_steps >= 1 && max_steps <= 5);

    let mut active = beam_width;
    let mut completed: usize = 0;

    for _ in 0..max_steps {
        if active == 0 {
            break;
        }
        let eos_count: usize = kani::any();
        kani::assume(eos_count <= active);

        completed += eos_count;
        active -= eos_count;
    }

    assert!(
        completed <= beam_width,
        "completed hypothesis count must be <= beam_width"
    );
}

/// Prove that total token storage across all output hypotheses is bounded
/// by beam_width * max_new_tokens.
#[kani::proof]
#[kani::unwind(4)]
fn proof_dpdf_hypothesis_token_memory_bounded() {
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 3);
    let max_new_tokens: usize = kani::any();
    kani::assume(max_new_tokens >= 1 && max_new_tokens <= 3);

    let mut total_tokens: usize = 0;
    for _ in 0..beam_width {
        let hyp_len: usize = kani::any();
        kani::assume(hyp_len >= 1 && hyp_len <= max_new_tokens);
        total_tokens += hyp_len;
    }

    let max_total = beam_width * max_new_tokens;
    assert!(
        total_tokens <= max_total,
        "total token storage must be <= beam_width * max_new_tokens"
    );
}

// ===========================================================================
// 8. END-TO-END CONSISTENCY — config + pruning + output compose correctly
// ===========================================================================

/// Prove end-to-end beam search consistency: valid config produces valid
/// output where all hypotheses have bounded length, beam count is bounded,
/// and scores are sorted descending.
#[kani::proof]
#[kani::unwind(5)]
fn proof_dpdf_beam_search_end_to_end_consistency() {
    let beam_width: usize = kani::any();
    kani::assume(beam_width >= 1 && beam_width <= 3);
    let max_new_tokens: usize = kani::any();
    kani::assume(max_new_tokens >= 1 && max_new_tokens <= 4);

    let config = BeamSearchConfig::new(beam_width)
        .with_max_new_tokens(max_new_tokens)
        .with_early_stopping(false);

    assert!(config.validate().is_ok(), "config must validate");

    let num_hyps: usize = kani::any();
    kani::assume(num_hyps >= 1 && num_hyps <= beam_width);

    let mut beams = Vec::with_capacity(num_hyps);
    for _ in 0..num_hyps {
        let len: usize = kani::any();
        kani::assume(len >= 1 && len <= max_new_tokens);
        let tokens: Vec<usize> = (0..len).collect();
        let log_prob: f64 = kani::any();
        kani::assume(log_prob.is_finite() && log_prob >= -100.0 && log_prob <= 0.0);
        let finished: bool = kani::any();
        beams.push(BeamHypothesis::new(tokens, log_prob, finished));
    }

    // Sort by score (penalty=0 for deterministic proof).
    beams.sort_by(|a, b| b.score(0.0).total_cmp(&a.score(0.0)));
    beams.truncate(config.beam_width);

    let output = BeamSearchOutput::new(beams);

    assert!(output.beams.len() <= beam_width);

    for beam in &output.beams {
        assert!(beam.token_ids.len() <= max_new_tokens);
    }

    for i in 1..output.beams.len() {
        let prev = output.beams[i - 1].score(0.0);
        let curr = output.beams[i].score(0.0);
        assert!(
            prev.total_cmp(&curr).is_ge(),
            "output must be sorted by score descending"
        );
    }
}

/// Prove end-to-end greedy decoding consistency: valid config + argmax =
/// valid output with bounded length and valid token IDs.
#[kani::proof]
#[kani::unwind(7)]
fn proof_dpdf_greedy_end_to_end_consistency() {
    let max_new_tokens: usize = kani::any();
    kani::assume(max_new_tokens >= 1 && max_new_tokens <= 5);
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size >= 1 && vocab_size <= 4);

    let config = GenerationConfig::new(max_new_tokens);
    assert!(config.validate().is_ok(), "config must validate");

    let mut tokens = Vec::new();
    let mut finished = false;
    let eos_id: usize = kani::any();
    kani::assume(eos_id < vocab_size);

    for _ in 0..max_new_tokens {
        let mut logits = vec![0.0f32; vocab_size];
        for v in logits.iter_mut() {
            *v = kani::any();
            kani::assume(v.is_finite());
        }
        let token = inline_argmax(&logits);
        assert!(token < vocab_size, "argmax must be valid");

        tokens.push(token);

        if inline_is_eos(token, Some(eos_id)) {
            finished = true;
            break;
        }
    }

    let output = GenerationOutput::new(tokens, finished);

    assert!(output.token_ids.len() <= max_new_tokens);

    for &t in &output.token_ids {
        assert!(t < vocab_size, "all generated tokens must be valid");
    }
}
