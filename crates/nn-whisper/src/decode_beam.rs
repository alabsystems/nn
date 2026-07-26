// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Beam search decode adapter for Whisper.
//!
//! Wraps `WhisperModel::decode()` with a beam search strategy that maintains
//! `beam_width` hypotheses in parallel. Since the model manages its own
//! internal KV cache (not the nn-core `KvCache`), each beam step resets
//! the cache and replays the full token sequence (inherently O(B × S²)).
//!
//! Token history uses a parent-pointer tree (matching `nn_core::layers::beam_search`)
//! so beam expansion is O(1) per candidate instead of O(S) Vec clones. Full
//! token sequences are reconstructed only when needed: once per active beam
//! per step (for model replay) and once at finalization.

use crate::decode::{
    apply_suppression_inplace, check_logit_finiteness, compression_ratio, DecodeConfig,
    DecodingResult, EOT_TOKEN,
};
use crate::{WhisperError, WhisperModel};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{Result, TensorError, D};

/// Configuration for Whisper beam search.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct WhisperBeamConfig {
    /// Number of beams to maintain. Must be > 0.
    pub beam_width: usize,
    /// Length penalty exponent: score = log_prob / length^penalty.
    /// 0.0 = no penalty, 1.0 = full normalization.
    pub length_penalty: f64,
}

impl Default for WhisperBeamConfig {
    fn default() -> Self {
        Self {
            beam_width: 5,
            length_penalty: 1.0,
        }
    }
}

impl WhisperBeamConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<()> {
        if self.beam_width == 0 {
            return Err(WhisperError::ZeroConfigField {
                field: "beam_width",
            }
            .into());
        }
        if !self.length_penalty.is_finite() {
            return Err(WhisperError::NonFiniteConfigField {
                field: "length_penalty",
                value: self.length_penalty,
            }
            .into());
        }
        Ok(())
    }
}

/// Lightweight beam state using parent-pointer tree for token history.
///
/// Instead of cloning `Vec<usize>` at every expansion (O(S) per candidate),
/// each beam stores only an index into a shared `tree` and reconstructs
/// token sequences on demand. Tree nodes are `(parent_plus_one, token)` where
/// `parent_plus_one == 0` indicates a root node (no parent).
#[derive(Debug, Clone)]
struct BeamState {
    /// Index into the parent-pointer tree for the last decoded token.
    /// `None` if no decoded tokens yet (EOT on first expansion).
    node_idx: Option<usize>,
    /// Number of decoded tokens (excluding initial prompt).
    decoded_len: usize,
    /// Cumulative log-probability of decoded tokens.
    sum_log_prob: f64,
    /// Whether EOT was reached.
    finished: bool,
}

impl BeamState {
    fn score(&self, length_penalty: f64) -> f64 {
        if length_penalty == 0.0 || self.decoded_len == 0 {
            self.sum_log_prob
        } else {
            let len = self.decoded_len as f64;
            self.sum_log_prob / len.powf(length_penalty)
        }
    }
}

/// Reconstruct decoded token sequence by walking the parent-pointer tree
/// from leaf to root, then reversing. Returns empty Vec if `node_idx` is None.
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

/// Reconstruct the full token sequence (initial_tokens ++ decoded_tokens)
/// needed for model replay (Whisper resets KV cache each step).
fn reconstruct_all_tokens(
    initial_tokens: &[usize],
    node_idx: Option<usize>,
    tree: &[(usize, usize)],
) -> Vec<usize> {
    let mut all = initial_tokens.to_vec();
    all.extend(reconstruct_decoded(node_idx, tree));
    all
}

/// Decode with beam search.
///
/// Runs `beam_width` hypotheses in parallel, expanding each at every step
/// by the top candidates and keeping the best beams by score.
///
/// Returns the best hypothesis as a `DecodingResult` for API compatibility
/// with `greedy_decode()` and `temperature_fallback_decode()`.
pub fn beam_search_decode(
    model: &mut WhisperModel,
    encoder_output: &DynTensor,
    config: &DecodeConfig,
    beam_config: &WhisperBeamConfig,
) -> Result<DecodingResult> {
    beam_config.validate()?;
    config.validate()?;
    let device = encoder_output.device();
    let beam_width = beam_config.beam_width;

    // First step: feed initial tokens, get logits for all beams' starting point.
    model.reset_kv_cache();
    let initial_u32: Vec<u32> = config
        .initial_tokens
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
    let last_logits = &flat[offset..];

    // Compute no-speech probability from initial step (same as greedy).
    let no_speech_prob = super::compute_no_speech_prob(last_logits);

    // Apply suppression and get top-k tokens for initial beams.
    let mut suppressed = last_logits.to_vec();
    apply_suppression_inplace(&mut suppressed, &config.suppress_tokens);
    let top = top_k_with_log_probs(&suppressed, beam_width);

    // Parent-pointer tree: each entry is (parent_plus_one, token).
    // parent_plus_one == 0 means root (no parent). Using +1 offset avoids
    // Option<usize> overhead while maintaining the sentinel value.
    let mut tree: Vec<(usize, usize)> = Vec::new();

    let mut beams: Vec<BeamState> = top
        .into_iter()
        .map(|(token, log_prob)| {
            if token == EOT_TOKEN {
                BeamState {
                    node_idx: None,
                    decoded_len: 0,
                    sum_log_prob: f64::from(log_prob),
                    finished: true,
                }
            } else {
                let idx = tree.len();
                tree.push((0, token)); // root node, no parent
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
    for step in 1..config.max_length {
        let active_count = beams.iter().filter(|b| !b.finished).count();
        if active_count == 0 {
            break;
        }

        let mut all_candidates: Vec<(usize, usize, f32, bool)> =
            Vec::with_capacity(active_count * beam_width); // (beam_idx, token, log_prob, is_eot)

        for (beam_idx, beam) in beams.iter().enumerate() {
            if beam.finished {
                continue;
            }

            // Replay this beam's token sequence through the model.
            // Reconstruction happens once per active beam per step.
            model.reset_kv_cache();
            let all_tokens = reconstruct_all_tokens(&config.initial_tokens, beam.node_idx, &tree);
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
            let logits_view = logits.to_f32_array()?;
            let logits_contiguous = logits_view.as_standard_layout();
            let flat = logits_contiguous.as_slice().ok_or_else(|| {
                TensorError::InvalidShape("logits not contiguous after as_standard_layout".into())
            })?;
            let off = flat.len().checked_sub(vocab_size).ok_or_else(|| {
                TensorError::from(WhisperError::LogitTooSmall {
                    logit_len: flat.len(),
                    vocab_size,
                })
            })?;
            let mut last = flat[off..].to_vec();
            apply_suppression_inplace(&mut last, &config.suppress_tokens);

            let top = top_k_with_log_probs(&last, beam_width);
            for (token, lp) in top {
                all_candidates.push((beam_idx, token, lp, token == EOT_TOKEN));
            }
        }

        // Score candidates: parent beam's score + new log prob.
        let mut scored: Vec<(f64, usize, usize, f32, bool)> = all_candidates
            .iter()
            .map(|&(bi, tok, lp, is_eot)| {
                let parent_score = beams[bi].sum_log_prob;
                let new_score = parent_score + f64::from(lp);
                (new_score, bi, tok, lp, is_eot)
            })
            .collect();

        // Sort by score descending.
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.truncate(beam_width);

        // Build new beam set — O(1) per candidate via tree push.
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
                // Parent pointer uses +1 offset; 0 = no parent (root).
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

        // Carry forward finished beams from prior steps so the globally
        // best hypothesis is never dropped (#1636).
        for beam in &beams {
            if beam.finished {
                new_beams.push(beam.clone());
            }
        }

        // Keep the best beam_width beams by length-normalized score.
        new_beams.sort_by(|a, b| {
            b.score(beam_config.length_penalty)
                .total_cmp(&a.score(beam_config.length_penalty))
        });
        new_beams.truncate(beam_width);

        beams = new_beams;

        // All beams finished → stop.
        if beams.iter().all(|b| b.finished) {
            break;
        }
    }

    // Select best beam by length-normalized score.
    let best = beams
        .iter()
        .max_by(|a, b| {
            a.score(beam_config.length_penalty)
                .total_cmp(&b.score(beam_config.length_penalty))
        })
        .ok_or_else(|| {
            TensorError::from(WhisperError::EmptyDecodeResult {
                reason: "beam search produced no beams",
            })
        })?;

    // Reconstruct decoded tokens only once, at finalization.
    let decoded_tokens = reconstruct_decoded(best.node_idx, &tree);
    let cr = compression_ratio(&decoded_tokens);
    let avg_logprob = if decoded_tokens.is_empty() {
        0.0
    } else {
        best.sum_log_prob / decoded_tokens.len() as f64
    };

    Ok(DecodingResult {
        tokens: decoded_tokens,
        avg_logprob,
        compression_ratio: cr,
        reached_eot: best.finished,
        temperature: 0.0,
        no_speech_prob,
    })
}

/// Get top-k tokens by log-probability from a logit slice.
///
/// Uses indices-only partial sort O(V + k log k) instead of full O(V log V) sort.
/// For V=51,865 and k=5, this avoids sorting ~51,860 irrelevant entries.
// pub(super) for Kani proof harnesses in sibling kani_decode_beam_proofs module (#3645).
pub(super) fn top_k_with_log_probs(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    // Compute log-softmax.
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_val == f32::NEG_INFINITY {
        return vec![(0, f32::NEG_INFINITY)];
    }
    let log_sum_exp: f32 = logits
        .iter()
        .map(|&v| (v - max_val).exp())
        .sum::<f32>()
        .ln()
        + max_val;

    // Partial sort: partition indices around the k-th largest log-prob,
    // then sort only the top k. Avoids full O(V log V) sort of all vocab entries.
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    let effective_k = k.min(indices.len());
    if effective_k == 0 {
        return vec![(0, f32::NEG_INFINITY)];
    }
    // Comparator: descending by logit value, ties broken by descending index
    // (higher index wins, matching argmax_f32 behavior for beam_width=1).
    let cmp = |&a: &usize, &b: &usize| -> std::cmp::Ordering {
        logits[b].total_cmp(&logits[a]).then(b.cmp(&a))
    };
    if effective_k < indices.len() {
        indices.select_nth_unstable_by(effective_k - 1, cmp);
        indices.truncate(effective_k);
    }
    // Sort the top k by log-prob descending, ties by index descending.
    indices.sort_unstable_by(cmp);
    indices
        .iter()
        .map(|&i| (i, logits[i] - log_sum_exp))
        .collect()
}

// No-speech probability computation: delegates to parent module's
// `compute_no_speech_prob` via `super::compute_no_speech_prob`.
