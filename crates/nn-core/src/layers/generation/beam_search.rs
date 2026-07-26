// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Beam search decoding for autoregressive generation.
//!
//! Maintains `beam_width` hypotheses in parallel, expanding each by the
//! top candidates at each step and keeping the best beams by cumulative
//! log-probability. Supports length normalization and early stopping.

use super::kv_cache::KvCache;
use crate::dyn_tensor::DynTensor;
use crate::{Device, Result, TensorError};

/// Configuration for beam search decoding.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BeamSearchConfig {
    /// Number of beams (hypotheses) to maintain.
    pub beam_width: usize,
    /// Maximum number of new tokens to generate per beam.
    pub max_new_tokens: usize,
    /// Length penalty exponent: score = log_prob / length^length_penalty.
    /// 0.0 = no penalty, 1.0 = full normalization, >1.0 = favor longer.
    pub length_penalty: f64,
    /// If true, stop as soon as `beam_width` complete hypotheses exist.
    pub early_stopping: bool,
    /// Token ID that signals end of a beam.
    pub eos_token_id: Option<usize>,
}

impl BeamSearchConfig {
    /// Create config with the given beam width. Other fields use defaults.
    #[must_use]
    pub fn new(beam_width: usize) -> Self {
        Self {
            beam_width,
            ..Default::default()
        }
    }

    /// Set maximum new tokens to generate per beam.
    #[must_use]
    pub fn with_max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.max_new_tokens = max_new_tokens;
        self
    }

    /// Set length penalty exponent.
    #[must_use]
    pub fn with_length_penalty(mut self, length_penalty: f64) -> Self {
        self.length_penalty = length_penalty;
        self
    }

    /// Set early stopping behavior.
    #[must_use]
    pub fn with_early_stopping(mut self, early_stopping: bool) -> Self {
        self.early_stopping = early_stopping;
        self
    }

    /// Set end-of-sequence token ID.
    #[must_use]
    pub fn with_eos_token_id(mut self, eos_token_id: usize) -> Self {
        self.eos_token_id = Some(eos_token_id);
        self
    }

    /// Validate configuration parameters.
    ///
    /// Rejects `beam_width == 0` and non-finite `length_penalty` (NaN or Inf),
    /// which would cause nondeterministic beam ordering.
    pub fn validate(&self) -> Result<()> {
        if self.beam_width == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "BeamSearchConfig: beam_width must be > 0",
            });
        }
        if !self.length_penalty.is_finite() {
            return Err(TensorError::ValueOutOfRange {
                description: "BeamSearchConfig: length_penalty must be finite",
            });
        }
        Ok(())
    }
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            max_new_tokens: 128,
            length_penalty: 1.0,
            early_stopping: false,
            eos_token_id: None,
        }
    }
}

/// A single beam hypothesis with its cumulative log-probability.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BeamHypothesis {
    /// Generated token IDs (not including the prompt).
    pub token_ids: Vec<usize>,
    /// Cumulative log-probability (sum of log-softmax values).
    pub log_prob: f64,
    /// Whether this beam has hit the EOS token.
    pub finished: bool,
}

impl BeamHypothesis {
    /// Create a beam hypothesis.
    pub fn new(token_ids: Vec<usize>, log_prob: f64, finished: bool) -> Self {
        Self {
            token_ids,
            log_prob,
            finished,
        }
    }

    /// Length-normalized score.
    fn score(&self, length_penalty: f64) -> f64 {
        if length_penalty == 0.0 || self.token_ids.is_empty() {
            self.log_prob
        } else {
            let len = self.token_ids.len() as f64;
            self.log_prob / len.powf(length_penalty)
        }
    }
}

/// Output from beam search decoding.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BeamSearchOutput {
    /// Top beams ranked by length-normalized score (best first).
    pub beams: Vec<BeamHypothesis>,
}

impl BeamSearchOutput {
    /// Create a beam search output.
    pub fn new(beams: Vec<BeamHypothesis>) -> Self {
        Self { beams }
    }
}

/// Run beam search decoding with a model forward function.
///
/// `model_fn` takes `(input_tensor, &mut KvCache)` and returns logits shaped
/// `[1, vocab_size]` or `[1, seq_len, vocab_size]`.
///
/// `prompt_ids` are the initial token IDs to prefill the cache with.
///
/// Returns up to `beam_width` hypotheses ranked by score.
///
/// **Note:** Beam search creates independent KV cache copies for each beam.
/// The provided `cache` is used for the initial prefill; each beam then
/// maintains its own cache state.
pub fn beam_search<F>(
    model_fn: F,
    prompt_ids: &[usize],
    cache: &mut KvCache,
    config: &BeamSearchConfig,
    device: &Device,
) -> Result<BeamSearchOutput>
where
    F: Fn(&DynTensor, &mut KvCache) -> Result<DynTensor>,
{
    if prompt_ids.is_empty() {
        return Err(TensorError::InvalidShape(
            "beam_search: prompt_ids must not be empty".into(),
        ));
    }
    config.validate()?;
    if config.max_new_tokens == 0 {
        return Ok(BeamSearchOutput {
            beams: vec![BeamHypothesis {
                token_ids: Vec::new(),
                log_prob: 0.0,
                finished: false,
            }],
        });
    }

    // Prefill: process the entire prompt
    let prompt_tensor = ids_to_tensor(prompt_ids, device)?;
    let logits = model_fn(&prompt_tensor, cache)?;
    let vocab_logits = extract_last_vocab_logits(&logits)?;
    let log_probs = log_softmax(&vocab_logits);

    // Parent-pointer tree: instead of materializing full token_ids at each step
    // (O(W * S²) total copies), store only (parent_node_idx, token) per beam per
    // step and reconstruct at the end (O(W * S) total). Each node in `tree` is
    // (parent_index_or_none, token).
    let mut tree: Vec<(Option<usize>, usize)> = Vec::new();

    // Initialize beams from top-k tokens of prefill logits.
    let top_tokens = top_k_by_value(&log_probs, config.beam_width);
    let mut beam_caches: Vec<KvCache> = (0..top_tokens.len()).map(|_| cache.clone()).collect();

    // Lightweight beam state: tracks tree node index instead of full token_ids.
    struct BeamState {
        node_idx: usize,
        last_token: usize,
        log_prob: f64,
        token_count: usize,
        finished: bool,
    }

    let mut active: Vec<BeamState> = top_tokens
        .into_iter()
        .map(|(token, log_p)| {
            let node_idx = tree.len();
            tree.push((None, token));
            BeamState {
                node_idx,
                last_token: token,
                log_prob: f64::from(log_p),
                token_count: 1,
                finished: is_eos(token, config),
            }
        })
        .collect();

    // Completed beams stored as (node_idx, log_prob, token_count).
    let mut completed: Vec<(usize, f64, usize)> = Vec::new();

    // Collect any already-finished beams from prefill
    let mut next_active = Vec::new();
    let mut next_caches = Vec::new();
    for (i, beam) in active.into_iter().enumerate() {
        if beam.finished {
            completed.push((beam.node_idx, beam.log_prob, beam.token_count));
        } else {
            next_active.push(beam);
            next_caches.push(std::mem::replace(&mut beam_caches[i], KvCache::new(0)));
        }
    }
    active = next_active;
    beam_caches = next_caches;

    // Check early stopping after prefill
    if config.early_stopping && completed.len() >= config.beam_width {
        let active_info: Vec<_> = active.iter().map(|b| (b.node_idx, b.log_prob)).collect();
        return Ok(finalize_tree(completed, &active_info, &tree, config));
    }

    // Decode step by step
    for _ in 1..config.max_new_tokens {
        if active.is_empty() {
            break;
        }

        // Run model forward for each active beam and collect lightweight candidates.
        struct Candidate {
            parent_beam_idx: usize,
            parent_node_idx: usize,
            token: usize,
            log_prob: f64,
            token_count: usize,
            finished: bool,
        }
        impl Candidate {
            fn score(&self, length_penalty: f64) -> f64 {
                if length_penalty == 0.0 || self.token_count == 0 {
                    self.log_prob
                } else {
                    self.log_prob / (self.token_count as f64).powf(length_penalty)
                }
            }
        }

        let mut all_candidates: Vec<Candidate> = Vec::new();

        for (beam_idx, beam) in active.iter().enumerate() {
            let input = ids_to_tensor(&[beam.last_token], device)?;
            let logits = model_fn(&input, &mut beam_caches[beam_idx])?;
            let vocab_logits = extract_last_vocab_logits(&logits)?;
            let log_probs = log_softmax(&vocab_logits);

            let expansions = top_k_by_value(&log_probs, config.beam_width);
            for (token, log_p) in expansions {
                all_candidates.push(Candidate {
                    parent_beam_idx: beam_idx,
                    parent_node_idx: beam.node_idx,
                    token,
                    log_prob: beam.log_prob + f64::from(log_p),
                    token_count: beam.token_count + 1,
                    finished: is_eos(token, config),
                });
            }
        }

        // Sort candidates by score (descending) and keep top beam_width.
        all_candidates.sort_by(|a, b| {
            b.score(config.length_penalty)
                .total_cmp(&a.score(config.length_penalty))
        });
        all_candidates.truncate(config.beam_width);

        // Append new nodes to tree and create beam states (O(1) per beam, no token copy).
        let surviving: Vec<(BeamState, usize)> = all_candidates
            .into_iter()
            .map(|c| {
                let node_idx = tree.len();
                tree.push((Some(c.parent_node_idx), c.token));
                (
                    BeamState {
                        node_idx,
                        last_token: c.token,
                        log_prob: c.log_prob,
                        token_count: c.token_count,
                        finished: c.finished,
                    },
                    c.parent_beam_idx,
                )
            })
            .collect();

        // Reorder KV caches based on which parent beams survived.
        // Only clone when two surviving beams share the same parent.
        let parent_indices: Vec<usize> = surviving.iter().map(|(_, idx)| *idx).collect();
        let mut new_caches: Vec<KvCache> = Vec::with_capacity(parent_indices.len());
        let mut moved: Vec<bool> = vec![false; beam_caches.len()];
        // Track where each parent was first placed in new_caches.
        let mut first_placement: Vec<Option<usize>> = vec![None; beam_caches.len()];
        for &parent_idx in &parent_indices {
            if !moved[parent_idx] {
                moved[parent_idx] = true;
                first_placement[parent_idx] = Some(new_caches.len());
                new_caches.push(std::mem::replace(
                    &mut beam_caches[parent_idx],
                    KvCache::new(0),
                ));
            } else {
                let src_pos =
                    first_placement[parent_idx].ok_or(TensorError::DimensionOutOfRange {
                        dim: parent_idx,
                        rank: first_placement.len(),
                    })?;
                new_caches.push(new_caches[src_pos].clone());
            }
        }
        beam_caches = new_caches;

        // Separate finished and active beams
        let mut next_active = Vec::new();
        let mut next_caches = Vec::new();
        for (i, (beam, _)) in surviving.into_iter().enumerate() {
            if beam.finished {
                completed.push((beam.node_idx, beam.log_prob, beam.token_count));
            } else {
                next_active.push(beam);
                next_caches.push(std::mem::replace(&mut beam_caches[i], KvCache::new(0)));
            }
        }
        active = next_active;
        beam_caches = next_caches;

        // Early stopping: we have enough completed beams
        if config.early_stopping && completed.len() >= config.beam_width {
            break;
        }
    }

    let active_info: Vec<_> = active.iter().map(|b| (b.node_idx, b.log_prob)).collect();
    Ok(finalize_tree(completed, &active_info, &tree, config))
}

#[path = "beam_search_helpers.rs"]
mod helpers;
use helpers::{
    extract_last_vocab_logits, finalize_tree, ids_to_tensor, is_eos, log_softmax, top_k_by_value,
};

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    fn powf_f64_stub(_b: f64, _e: f64) -> f64 {
        let r: f64 = kani::any();
        kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
        r
    }

    /// Prove `BeamHypothesis::score` never panics for any finite inputs.
    /// Covers: empty token_ids, zero length penalty, positive/negative log_prob.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::stub(f64::powf, powf_f64_stub)]
    fn proof_beam_hypothesis_score_no_panic() {
        let log_prob: f64 = kani::any();
        kani::assume(log_prob.is_finite() && log_prob.abs() < 1e6);
        let length_penalty: f64 = kani::any();
        kani::assume(length_penalty.is_finite() && length_penalty >= 0.0 && length_penalty <= 10.0);
        let num_tokens: usize = kani::any();
        kani::assume(num_tokens <= 4);

        let token_ids: Vec<usize> = (0..num_tokens).collect();
        let hyp = BeamHypothesis {
            token_ids,
            log_prob,
            finished: false,
        };
        let score = hyp.score(length_penalty);
        // Score must be finite (no NaN/Inf) for finite inputs.
        assert!(score.is_finite(), "score must be finite for finite inputs");
    }

    /// Prove `BeamHypothesis::score` with length_penalty=0 returns raw log_prob.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::stub(f64::powf, powf_f64_stub)]
    fn proof_beam_hypothesis_score_zero_penalty() {
        let log_prob: f64 = kani::any();
        kani::assume(log_prob.is_finite() && log_prob.abs() < 1e6);
        let num_tokens: usize = kani::any();
        kani::assume(num_tokens >= 1 && num_tokens <= 8);

        let token_ids: Vec<usize> = (0..num_tokens).collect();
        let hyp = BeamHypothesis {
            token_ids,
            log_prob,
            finished: false,
        };
        let score = hyp.score(0.0);
        assert!(
            (score - log_prob).abs() < 1e-12,
            "score with penalty=0 must equal raw log_prob"
        );
    }

    /// Prove `BeamSearchConfig::validate` rejects beam_width=0.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_beam_config_validate_rejects_zero_width() {
        let config = BeamSearchConfig {
            beam_width: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err(), "beam_width=0 must be rejected");
    }

    /// Prove `BeamSearchConfig::validate` rejects NaN/Inf length_penalty.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_beam_config_validate_rejects_nonfinite_penalty() {
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
            "Inf length_penalty must be rejected"
        );
    }

    /// Prove `BeamSearchConfig::validate` accepts valid configurations.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_beam_config_validate_accepts_valid() {
        let beam_width: usize = kani::any();
        kani::assume(beam_width >= 1 && beam_width <= 16);
        let length_penalty: f64 = kani::any();
        kani::assume(length_penalty.is_finite() && length_penalty >= 0.0);
        let config = BeamSearchConfig {
            beam_width,
            length_penalty,
            ..Default::default()
        };
        assert!(
            config.validate().is_ok(),
            "valid config must pass validation"
        );
    }
}

#[cfg(kani)]
#[path = "kani_beam_search_proofs.rs"]
mod kani_beam_search_proofs;

#[cfg(test)]
#[path = "beam_search_tests.rs"]
mod tests;
