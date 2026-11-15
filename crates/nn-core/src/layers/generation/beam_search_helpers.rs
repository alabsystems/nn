// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Private helper functions for beam search decoding.
//!
//! Contains tree traversal, tensor conversion, log-softmax, top-k selection,
//! and hypothesis finalization. Extracted from `beam_search.rs` for 500-line
//! compliance.

use super::{BeamHypothesis, BeamSearchConfig, BeamSearchOutput};
use crate::dyn_tensor::DynTensor;
use crate::{Device, Result, TensorError};

/// Reconstruct token_ids from the parent-pointer tree by walking from leaf to root.
pub(super) fn reconstruct_tokens(node_idx: usize, tree: &[(Option<usize>, usize)]) -> Vec<usize> {
    let mut tokens = Vec::new();
    let mut idx = node_idx;
    loop {
        let (parent, token) = tree[idx];
        tokens.push(token);
        match parent {
            Some(p) => idx = p,
            None => break,
        }
    }
    tokens.reverse();
    tokens
}

/// Finalize beam search: reconstruct token histories from tree, merge, sort.
/// `active` contains `(node_idx, log_prob)` for incomplete beams.
pub(super) fn finalize_tree(
    completed: Vec<(usize, f64, usize)>,
    active: &[(usize, f64)],
    tree: &[(Option<usize>, usize)],
    config: &BeamSearchConfig,
) -> BeamSearchOutput {
    let mut all: Vec<BeamHypothesis> = completed
        .into_iter()
        .map(|(node_idx, log_prob, _)| {
            let token_ids = reconstruct_tokens(node_idx, tree);
            BeamHypothesis {
                token_ids,
                log_prob,
                finished: true,
            }
        })
        .collect();

    // Add remaining active beams as incomplete hypotheses.
    for &(node_idx, log_prob) in active {
        let token_ids = reconstruct_tokens(node_idx, tree);
        all.push(BeamHypothesis {
            token_ids,
            log_prob,
            finished: false,
        });
    }

    // Sort by length-normalized score (best first)
    all.sort_by(|a, b| {
        b.score(config.length_penalty)
            .total_cmp(&a.score(config.length_penalty))
    });

    // Keep at most beam_width results
    all.truncate(config.beam_width);

    BeamSearchOutput { beams: all }
}

/// Convert token IDs to a 2D DynTensor `[1, seq_len]` with U32 dtype.
///
/// Uses `from_vec_u32` instead of f32 to avoid precision loss for IDs > 2^24.
/// `Embedding::forward()` handles U32 inputs natively.
pub(super) fn ids_to_tensor(ids: &[usize], device: &Device) -> Result<DynTensor> {
    let data: Vec<u32> = ids
        .iter()
        .map(|&id| {
            u32::try_from(id).map_err(|_| TensorError::ValueOutOfRange {
                description: "token id exceeds u32::MAX",
            })
        })
        .collect::<Result<Vec<_>>>()?;
    DynTensor::from_vec_u32(data, &[1, ids.len()], device)
}

/// Check if a token matches the EOS token ID.
pub(super) fn is_eos(token: usize, config: &BeamSearchConfig) -> bool {
    config.eos_token_id.is_some_and(|eos| token == eos)
}

/// Extract vocabulary logits from the last position of a 2D or 3D logits tensor.
pub(super) fn extract_last_vocab_logits(logits: &DynTensor) -> Result<Vec<f32>> {
    let logits_2d = if logits.rank() == 3 {
        let seq_len = logits.dim(1)?;
        logits
            .narrow(1, seq_len - 1, 1)?
            .reshape([logits.dim(0)?, logits.dim(2)?])?
    } else if logits.rank() == 2 {
        logits.clone()
    } else {
        return Err(TensorError::RankMismatch {
            expected: 3,
            actual: logits.rank(),
        });
    };

    let batch_logits = logits_2d.narrow(0, 0, 1)?;
    let cpu_logits = batch_logits.to_device(&Device::Cpu)?;
    let arr = cpu_logits.to_f32_array()?;
    let vocab_logits: Vec<f32> = arr.iter().copied().collect();
    if vocab_logits.is_empty() {
        return Err(TensorError::InvalidShape(
            "beam_search: empty vocabulary".into(),
        ));
    }
    Ok(vocab_logits)
}

/// Compute log-softmax of a slice (numerically stable).
///
/// When all logits are `NEG_INFINITY`, returns `NEG_INFINITY` for every element
/// instead of NaN. This matches the CPU softmax NaN guard (#1310).
pub(super) fn log_softmax(logits: &[f32]) -> Vec<f32> {
    // Sanitize NaN logits to NEG_INFINITY before computation.
    // A single NaN in the input would otherwise poison the entire sum
    // (NaN - max).exp() = NaN, sum(... + NaN + ...) = NaN, making ALL
    // outputs NaN. Treating NaN as -inf (impossible token) is the safe default.
    let sanitized: Vec<f32> = logits
        .iter()
        .map(|&v| if v.is_nan() { f32::NEG_INFINITY } else { v })
        .collect();
    let max_val = sanitized.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // Guard: if max_val is NEG_INFINITY (all inputs are -inf or NaN),
    // the subtraction `v - max_val` produces NaN (inf - inf).
    // Return -inf for all elements.
    if max_val == f32::NEG_INFINITY {
        return vec![f32::NEG_INFINITY; logits.len()];
    }
    // Guard: +inf in input → uniform probability over +inf positions, 0 for others.
    // IEEE 754: inf - inf = NaN, so the max-subtract trick fails.
    // Mathematically correct: exp(+inf) dominates, so +inf positions share
    // probability 1/count, giving log(1/count) = -ln(count). Others get -inf.
    // Matches DynTensor log_softmax guard (softmax.rs:141).
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

/// Return indices and values of top-k elements sorted by value descending.
/// Equal values prefer the larger token index, matching greedy sampling.
///
/// Uses indices-only partial sort (8 bytes/element) instead of
/// `(usize, f32)` pairs (16 bytes/element) for the sort phase,
/// then reconstructs pairs only for the k results.
pub(super) fn top_k_by_value(values: &[f32], k: usize) -> Vec<(usize, f32)> {
    if k == 0 {
        return Vec::new();
    }
    let mut indices: Vec<usize> = (0..values.len()).collect();
    let compare = |&a: &usize, &b: &usize| values[b].total_cmp(&values[a]).then_with(|| b.cmp(&a));
    if k < indices.len() {
        indices.select_nth_unstable_by(k - 1, compare);
        indices.truncate(k);
    }
    indices.sort_unstable_by(compare);
    indices.iter().map(|&i| (i, values[i])).collect()
}
