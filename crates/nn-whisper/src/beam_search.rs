// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Full-featured beam search decoding for Whisper.
//!
//! Extends the basic beam search in `decode_beam.rs` with:
//! - No-repeat n-gram filtering
//! - Temperature-scaled sampling within beams
//! - Blank token suppression at sequence start
//! - Explicit Whisper special token configuration
//! - Rich output with all hypotheses and normalized scores
//!
//! Uses the same parent-pointer tree approach as `decode_beam.rs` for O(1)
//! beam expansion. Full token sequences are reconstructed only at finalization.

use crate::decode::{apply_suppression_inplace, check_logit_finiteness};
use crate::tokenizer::{EOT_TOKEN, NO_TIMESTAMPS_TOKEN, SOT_TOKEN};
use crate::{WhisperError, WhisperModel};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{Result, TensorError, D};

/// Configuration for full-featured Whisper beam search.
///
/// Provides more control than [`crate::WhisperBeamConfig`] (from `decode_beam`),
/// including n-gram blocking, temperature, and blank suppression.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct WhisperBeamSearchConfig {
    /// Number of beams to maintain. Must be > 0. Default: 5.
    pub beam_width: usize,
    /// Maximum tokens to generate (excluding initial prompt). Default: 448.
    pub max_tokens: usize,
    /// Length penalty exponent: `score / len^penalty`. Default: 1.0.
    pub length_penalty: f32,
    /// Block repeated n-grams of this size. 0 = disabled. Default: 0.
    pub no_repeat_ngram_size: usize,
    /// Temperature for log-prob scaling. 0.0 = greedy within beam. Default: 0.0.
    pub temperature: f32,
    /// Suppress blank token (space, empty) at the start of generation. Default: true.
    pub suppress_blank: bool,
    /// Start-of-transcript token ID. Default: [`SOT_TOKEN`].
    pub sot_token: usize,
    /// End-of-transcript token ID. Default: [`EOT_TOKEN`].
    pub eot_token: usize,
    /// No-timestamps token ID. Default: [`NO_TIMESTAMPS_TOKEN`].
    pub no_timestamps_token: usize,
    /// Additional token IDs to suppress at every step.
    pub suppress_tokens: Vec<usize>,
}

impl Default for WhisperBeamSearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 5,
            max_tokens: 448,
            length_penalty: 1.0,
            no_repeat_ngram_size: 0,
            temperature: 0.0,
            suppress_blank: true,
            sot_token: SOT_TOKEN,
            eot_token: EOT_TOKEN,
            no_timestamps_token: NO_TIMESTAMPS_TOKEN,
            suppress_tokens: Vec::new(),
        }
    }
}

impl WhisperBeamSearchConfig {
    /// Validate configuration fields.
    pub fn validate(&self) -> Result<()> {
        if self.beam_width == 0 {
            return Err(WhisperError::ZeroConfigField {
                field: "beam_width",
            }
            .into());
        }
        if self.max_tokens == 0 {
            return Err(WhisperError::ZeroConfigField {
                field: "max_tokens",
            }
            .into());
        }
        if !self.length_penalty.is_finite() {
            return Err(WhisperError::NonFiniteConfigField {
                field: "length_penalty",
                value: f64::from(self.length_penalty),
            }
            .into());
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(WhisperError::NonFiniteConfigField {
                field: "temperature",
                value: f64::from(self.temperature),
            }
            .into());
        }
        Ok(())
    }
}

/// A single beam hypothesis with score information.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BeamHypothesis {
    /// Generated token IDs (excluding initial prompt tokens).
    pub tokens: Vec<usize>,
    /// Cumulative log-probability sum.
    pub score: f32,
    /// Length-normalized score (with length penalty applied).
    pub normalized_score: f32,
}

impl BeamHypothesis {
    /// Create a new hypothesis.
    #[must_use]
    pub fn new(tokens: Vec<usize>, score: f32, normalized_score: f32) -> Self {
        Self {
            tokens,
            score,
            normalized_score,
        }
    }
}

/// Output from beam search decoding.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct WhisperBeamOutput {
    /// All hypotheses ranked by normalized score (best first).
    pub hypotheses: Vec<BeamHypothesis>,
    /// The best hypothesis (same as `hypotheses[0]`).
    pub best: BeamHypothesis,
}

/// Internal beam state using parent-pointer tree.
#[derive(Debug, Clone)]
struct BeamState {
    /// Index into the parent-pointer tree for the last decoded token.
    /// `None` if no decoded tokens yet.
    node_idx: Option<usize>,
    /// Number of decoded tokens (excluding initial prompt).
    decoded_len: usize,
    /// Cumulative log-probability.
    sum_log_prob: f64,
    /// Whether EOT was reached.
    finished: bool,
}

impl BeamState {
    fn score(&self, length_penalty: f32) -> f64 {
        if length_penalty == 0.0 || self.decoded_len == 0 {
            self.sum_log_prob
        } else {
            let len = self.decoded_len as f64;
            self.sum_log_prob / len.powf(f64::from(length_penalty))
        }
    }
}

/// Reconstruct decoded tokens by walking the parent-pointer tree.
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

/// Reconstruct full token sequence (initial_tokens ++ decoded_tokens).
fn reconstruct_all_tokens(
    initial_tokens: &[usize],
    node_idx: Option<usize>,
    tree: &[(usize, usize)],
) -> Vec<usize> {
    let mut all = initial_tokens.to_vec();
    all.extend(reconstruct_decoded(node_idx, tree));
    all
}

/// Check if a token sequence ending with `candidate` would create a repeated n-gram.
///
/// Walks the parent-pointer tree to extract the last `n` tokens (including
/// `candidate`) and checks if that n-gram appeared earlier in the sequence.
///
/// Returns `true` if adding `candidate` would create a repeated n-gram.
fn would_repeat_ngram(
    node_idx: Option<usize>,
    candidate: usize,
    n: usize,
    tree: &[(usize, usize)],
    initial_tokens: &[usize],
) -> bool {
    if n == 0 {
        return false;
    }
    // Reconstruct the full token sequence for this beam including the candidate.
    let mut tokens = reconstruct_all_tokens(initial_tokens, node_idx, tree);
    tokens.push(candidate);

    let total = tokens.len();
    if total < n {
        return false;
    }

    // The candidate n-gram is the last `n` tokens.
    let candidate_ngram = &tokens[total - n..];

    // Check all earlier positions for this n-gram.
    // We only need to check up to `total - n` (exclusive) to avoid
    // comparing the candidate against itself.
    for start in 0..total - n {
        if tokens[start..start + n] == *candidate_ngram {
            return true;
        }
    }
    false
}

/// Compute log-softmax with optional temperature scaling.
///
/// Returns `(token_index, log_probability)` pairs for the top `k` tokens.
fn top_k_log_probs(logits: &[f32], k: usize, temperature: f32) -> Vec<(usize, f32)> {
    if logits.is_empty() {
        return vec![(0, f32::NEG_INFINITY)];
    }

    // Apply temperature scaling if > 0.
    let scaled: Vec<f32> = if temperature > 1e-8 {
        logits.iter().map(|&v| v / temperature).collect()
    } else {
        logits.to_vec()
    };

    // Compute log-softmax.
    let max_val = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_val == f32::NEG_INFINITY {
        return vec![(0, f32::NEG_INFINITY)];
    }
    let log_sum_exp: f32 = scaled
        .iter()
        .map(|&v| (v - max_val).exp())
        .sum::<f32>()
        .ln()
        + max_val;

    // Partial sort for top-k.
    let mut indices: Vec<usize> = (0..scaled.len()).collect();
    let effective_k = k.min(indices.len());
    if effective_k == 0 {
        return vec![(0, f32::NEG_INFINITY)];
    }

    let cmp = |&a: &usize, &b: &usize| -> std::cmp::Ordering {
        scaled[b].total_cmp(&scaled[a]).then(b.cmp(&a))
    };
    if effective_k < indices.len() {
        indices.select_nth_unstable_by(effective_k - 1, cmp);
        indices.truncate(effective_k);
    }
    indices.sort_unstable_by(cmp);
    indices
        .iter()
        .map(|&i| (i, scaled[i] - log_sum_exp))
        .collect()
}

/// Blank token ID — the space token in Whisper's GPT-2 vocabulary.
///
/// In GPT-2 byte-level BPE, the space character maps to token 220.
const BLANK_TOKEN: usize = 220;

/// Run full-featured beam search decode on Whisper.
///
/// Takes encoder output (audio features) and expands `beam_width` hypotheses
/// at each decode step. Supports:
/// - Length-normalized scoring
/// - No-repeat n-gram filtering
/// - Temperature-scaled log-probs
/// - Blank suppression at generation start
/// - Whisper-specific token suppression
///
/// Returns all hypotheses ranked by normalized score, plus the best one.
pub fn beam_search(
    model: &mut WhisperModel,
    encoder_output: &DynTensor,
    initial_tokens: &[usize],
    config: &WhisperBeamSearchConfig,
) -> Result<WhisperBeamOutput> {
    config.validate()?;
    let device = encoder_output.device();
    let beam_width = config.beam_width;

    // First step: feed initial tokens to get starting logits.
    model.reset_kv_cache();
    let initial_u32: Vec<u32> = initial_tokens
        .iter()
        .map(|&t| {
            u32::try_from(t)
                .map_err(|_| TensorError::from(WhisperError::TokenIdOverflow { token_id: t }))
        })
        .collect::<Result<_>>()?;
    let seq_len = initial_u32.len();
    let tokens_tensor = DynTensor::from_vec_u32(initial_u32, &[1, seq_len], &device)?;
    let logits = model.decode(&tokens_tensor, encoder_output, true, 0)?;
    check_logit_finiteness(&logits, 0)?;

    let vocab_size = logits.dim(D::Minus1)?;
    let last_logits = extract_last_step_logits(&logits, vocab_size)?;

    // Apply suppression.
    let mut suppressed = last_logits;
    apply_suppression_inplace(&mut suppressed, &config.suppress_tokens);
    if config.suppress_blank {
        suppress_blank_tokens(&mut suppressed, config.eot_token);
    }

    let top = top_k_log_probs(&suppressed, beam_width, config.temperature);

    // Parent-pointer tree: (parent_plus_one, token). 0 = root.
    let mut tree: Vec<(usize, usize)> = Vec::new();

    let mut beams: Vec<BeamState> = top
        .into_iter()
        .map(|(token, log_prob)| {
            if token == config.eot_token {
                BeamState {
                    node_idx: None,
                    decoded_len: 0,
                    sum_log_prob: f64::from(log_prob),
                    finished: true,
                }
            } else {
                let idx = tree.len();
                tree.push((0, token));
                BeamState {
                    node_idx: Some(idx),
                    decoded_len: 1,
                    sum_log_prob: f64::from(log_prob),
                    finished: false,
                }
            }
        })
        .collect();

    // Autoregressive beam search loop.
    for step in 1..config.max_tokens {
        let active_count = beams.iter().filter(|b| !b.finished).count();
        if active_count == 0 {
            break;
        }

        let mut all_candidates: Vec<(usize, usize, f32, bool)> =
            Vec::with_capacity(active_count * beam_width);

        for (beam_idx, beam) in beams.iter().enumerate() {
            if beam.finished {
                continue;
            }

            // Replay full sequence through the model (KV cache reset per beam).
            model.reset_kv_cache();
            let all_tokens = reconstruct_all_tokens(initial_tokens, beam.node_idx, &tree);
            let all_u32: Vec<u32> = all_tokens
                .iter()
                .map(|&t| {
                    u32::try_from(t).map_err(|_| {
                        TensorError::from(WhisperError::TokenIdOverflow { token_id: t })
                    })
                })
                .collect::<Result<_>>()?;
            let slen = all_u32.len();
            let t = DynTensor::from_vec_u32(all_u32, &[1, slen], &device)?;
            let logits = model.decode(&t, encoder_output, true, 0)?;
            check_logit_finiteness(&logits, step)?;

            let vocab_size = logits.dim(D::Minus1)?;
            let mut last = extract_last_step_logits(&logits, vocab_size)?;
            apply_suppression_inplace(&mut last, &config.suppress_tokens);

            // Apply no-repeat n-gram filter before top-k selection.
            if config.no_repeat_ngram_size > 0 {
                apply_ngram_blocking(
                    &mut last,
                    beam.node_idx,
                    config.no_repeat_ngram_size,
                    &tree,
                    initial_tokens,
                );
            }

            let top = top_k_log_probs(&last, beam_width, config.temperature);
            for (token, lp) in top {
                all_candidates.push((beam_idx, token, lp, token == config.eot_token));
            }
        }

        // Score candidates by cumulative log-prob.
        let mut scored: Vec<(f64, usize, usize, f32, bool)> = all_candidates
            .iter()
            .map(|&(bi, tok, lp, is_eot)| {
                let parent_score = beams[bi].sum_log_prob;
                let new_score = parent_score + f64::from(lp);
                (new_score, bi, tok, lp, is_eot)
            })
            .collect();

        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.truncate(beam_width);

        // Build new beam set.
        let mut new_beams: Vec<BeamState> = Vec::with_capacity(beam_width * 2);
        for &(score, parent_idx, token, _lp, is_eot) in &scored {
            let parent = &beams[parent_idx];
            if is_eot {
                new_beams.push(BeamState {
                    node_idx: parent.node_idx,
                    decoded_len: parent.decoded_len,
                    sum_log_prob: score,
                    finished: true,
                });
            } else {
                let idx = tree.len();
                let parent_plus_one = parent.node_idx.map_or(0, |i| i + 1);
                tree.push((parent_plus_one, token));
                new_beams.push(BeamState {
                    node_idx: Some(idx),
                    decoded_len: parent.decoded_len + 1,
                    sum_log_prob: score,
                    finished: false,
                });
            }
        }

        // Carry forward finished beams from prior steps.
        for beam in &beams {
            if beam.finished {
                new_beams.push(beam.clone());
            }
        }

        // Keep the best beam_width beams by length-normalized score.
        new_beams.sort_by(|a, b| {
            b.score(config.length_penalty)
                .total_cmp(&a.score(config.length_penalty))
        });
        new_beams.truncate(beam_width);

        beams = new_beams;

        if beams.iter().all(|b| b.finished) {
            break;
        }
    }

    // Build output hypotheses sorted by normalized score.
    let mut hypotheses: Vec<BeamHypothesis> = beams
        .iter()
        .map(|b| {
            let tokens = reconstruct_decoded(b.node_idx, &tree);
            let score = b.sum_log_prob as f32;
            let normalized_score = b.score(config.length_penalty) as f32;
            BeamHypothesis {
                tokens,
                score,
                normalized_score,
            }
        })
        .collect();

    hypotheses.sort_by(|a, b| b.normalized_score.total_cmp(&a.normalized_score));

    let best = hypotheses.first().cloned().ok_or_else(|| {
        TensorError::from(WhisperError::EmptyDecodeResult {
            reason: "beam search produced no hypotheses",
        })
    })?;

    Ok(WhisperBeamOutput { hypotheses, best })
}

/// Extract the last time step's logits as a mutable Vec.
fn extract_last_step_logits(logits: &DynTensor, vocab_size: usize) -> Result<Vec<f32>> {
    let logits_view = logits.to_f32_array()?;
    let logits_contiguous = logits_view.as_standard_layout();
    let flat = logits_contiguous.as_slice().ok_or_else(|| {
        TensorError::InvalidShape("logits not contiguous after as_standard_layout".into())
    })?;
    let offset = flat.len().checked_sub(vocab_size).ok_or_else(|| {
        TensorError::from(WhisperError::LogitTooSmall {
            logit_len: flat.len(),
            vocab_size,
        })
    })?;
    Ok(flat[offset..].to_vec())
}

/// Suppress blank-like tokens (space token and EOT) at generation start.
fn suppress_blank_tokens(logits: &mut [f32], eot_token: usize) {
    if BLANK_TOKEN < logits.len() {
        logits[BLANK_TOKEN] = f32::NEG_INFINITY;
    }
    if eot_token < logits.len() {
        logits[eot_token] = f32::NEG_INFINITY;
    }
}

/// Block tokens that would create a repeated n-gram by setting their logits
/// to negative infinity.
fn apply_ngram_blocking(
    logits: &mut [f32],
    node_idx: Option<usize>,
    n: usize,
    tree: &[(usize, usize)],
    initial_tokens: &[usize],
) {
    for token_id in 0..logits.len() {
        if would_repeat_ngram(node_idx, token_id, n, tree, initial_tokens) {
            logits[token_id] = f32::NEG_INFINITY;
        }
    }
}

/// Compute length-normalized score.
#[must_use]
pub fn normalize_score(score: f32, length: usize, length_penalty: f32) -> f32 {
    if length_penalty == 0.0 || length == 0 {
        score
    } else {
        let len = length as f32;
        score / len.powf(length_penalty)
    }
}

#[cfg(test)]
#[path = "beam_search_tests.rs"]
mod tests;
