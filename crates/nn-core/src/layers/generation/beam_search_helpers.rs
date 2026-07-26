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
///
/// Uses indices-only partial sort (8 bytes/element) instead of
/// `(usize, f32)` pairs (16 bytes/element) for the sort phase,
/// then reconstructs pairs only for the k results.
pub(super) fn top_k_by_value(values: &[f32], k: usize) -> Vec<(usize, f32)> {
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

#[cfg(kani)]
mod kani_proofs {
    use super::*;

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

    fn powf_f64_stub(_b: f64, _e: f64) -> f64 {
        let r: f64 = kani::any();
        kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
        r
    }

    /// Prove `reconstruct_tokens` never panics on a well-formed parent-pointer
    /// tree (all parent indices < tree length, root has parent=None).
    /// Bounded to trees of up to 6 nodes.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(7)]
    fn proof_reconstruct_tokens_no_panic() {
        let len: usize = kani::any();
        kani::assume(len >= 1 && len <= 6);

        let mut tree: Vec<(Option<usize>, usize)> = Vec::with_capacity(len);
        for i in 0..len {
            let token: usize = kani::any();
            kani::assume(token <= 1024);
            if i == 0 {
                // Root node has no parent.
                tree.push((None, token));
            } else {
                let has_parent: bool = kani::any();
                if has_parent {
                    let parent: usize = kani::any();
                    // Parent must point to an earlier node (DAG invariant
                    // ensures termination — no cycles).
                    kani::assume(parent < i);
                    tree.push((Some(parent), token));
                } else {
                    tree.push((None, token));
                }
            }
        }

        let node_idx: usize = kani::any();
        kani::assume(node_idx < len);

        let result = reconstruct_tokens(node_idx, &tree);
        // Output must be non-empty (at least the starting node's token).
        assert!(!result.is_empty(), "must return at least one token");
        // Output length bounded by tree size.
        assert!(
            result.len() <= len,
            "cannot return more tokens than tree nodes"
        );
    }

    /// Prove `reconstruct_tokens` returns exactly the tokens on the root-to-leaf
    /// path for a linear chain tree (each node's parent is the previous node).
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_reconstruct_tokens_linear_chain() {
        let len: usize = kani::any();
        kani::assume(len >= 1 && len <= 4);

        let mut tree: Vec<(Option<usize>, usize)> = Vec::with_capacity(len);
        for i in 0..len {
            let token: usize = kani::any();
            kani::assume(token <= 256);
            if i == 0 {
                tree.push((None, token));
            } else {
                tree.push((Some(i - 1), token));
            }
        }

        // Reconstruct from the last node (leaf).
        let result = reconstruct_tokens(len - 1, &tree);
        // A linear chain from root to leaf includes all nodes.
        assert_eq!(result.len(), len, "linear chain must include all nodes");
        // First token in result must be root's token.
        assert_eq!(result[0], tree[0].1, "first token must be root's token");
        // Last token in result must be leaf's token.
        assert_eq!(
            result[len - 1],
            tree[len - 1].1,
            "last token must be leaf's token"
        );
    }

    /// Prove `top_k_by_value` returns valid indices and at most k elements.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(7)]
    fn proof_top_k_by_value_valid() {
        let len: usize = kani::any();
        kani::assume(len >= 1 && len <= 6);
        let k: usize = kani::any();
        kani::assume(k <= len);

        let mut values = vec![0.0f32; len];
        for v in values.iter_mut() {
            *v = kani::any();
        }

        let result = top_k_by_value(&values, k);
        assert!(result.len() <= k, "returned more than k elements");
        for &(idx, val) in &result {
            assert!(idx < len, "index out of bounds");
            // Value must match the source array at the returned index.
            assert!(val.to_bits() == values[idx].to_bits(), "value mismatch");
        }
    }

    /// Prove `top_k_by_value` with k=0 returns empty.
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_top_k_zero_returns_empty() {
        let values = vec![1.0f32, 2.0, 3.0];
        let result = top_k_by_value(&values, 0);
        assert!(result.is_empty());
    }

    /// Prove `is_eos` is consistent with config (mirrors autoregressive proof).
    #[kani::unwind(1)]
    #[kani::proof]
    fn proof_is_eos_consistency() {
        let token: usize = kani::any();
        kani::assume(token <= 1024);

        let config_none = BeamSearchConfig {
            eos_token_id: None,
            ..Default::default()
        };
        assert!(!is_eos(token, &config_none), "None eos should never match");

        let eos_id: usize = kani::any();
        kani::assume(eos_id <= 1024);
        let config_some = BeamSearchConfig {
            eos_token_id: Some(eos_id),
            ..Default::default()
        };
        assert_eq!(
            is_eos(token, &config_some),
            token == eos_id,
            "is_eos must match iff token == eos_id"
        );
    }

    /// Prove `log_softmax` always returns a vector of the same length as input
    /// and never panics for any f32 values (up to 6 elements).
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(7)]
    #[kani::stub(f32::exp, exp_f32_stub)]
    #[kani::stub(f32::ln, ln_f32_stub)]
    fn proof_log_softmax_length_preserved() {
        let len: usize = kani::any();
        kani::assume(len >= 1 && len <= 6);

        let mut logits = vec![0.0f32; len];
        for v in logits.iter_mut() {
            *v = kani::any();
        }

        let result = log_softmax(&logits);
        assert_eq!(result.len(), len, "output length must match input length");
    }

    /// Prove `log_softmax` outputs are all <= 0 for finite non-NaN inputs.
    /// log(softmax(x)) = log(exp(x_i) / sum(exp(x_j))) <= 0 because the
    /// fraction is in (0, 1].
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    #[kani::stub(f32::exp, exp_f32_stub)]
    #[kani::stub(f32::ln, ln_f32_stub)]
    fn proof_log_softmax_outputs_nonpositive() {
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
                "log_softmax outputs must be <= 0 (with epsilon tolerance)"
            );
        }
    }

    /// Prove `finalize_tree` returns at most `beam_width` beams.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    #[kani::stub(f64::powf, powf_f64_stub)]
    fn proof_finalize_tree_bounded_output() {
        let beam_width: usize = kani::any();
        kani::assume(beam_width >= 1 && beam_width <= 4);

        // Build a small tree: root + up to 4 children.
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

        // Create a mix of completed and active beams.
        let num_completed: usize = kani::any();
        kani::assume(num_completed <= num_nodes);
        let mut completed: Vec<(usize, f64, usize)> = Vec::new();
        for _ in 0..num_completed {
            let node_idx: usize = kani::any();
            kani::assume(node_idx < num_nodes);
            let log_prob: f64 = kani::any();
            kani::assume(log_prob.is_finite() && log_prob.abs() < 100.0);
            completed.push((node_idx, log_prob, 1));
        }

        let num_active: usize = kani::any();
        kani::assume(num_active <= num_nodes);
        let mut active: Vec<(usize, f64)> = Vec::new();
        for _ in 0..num_active {
            let node_idx: usize = kani::any();
            kani::assume(node_idx < num_nodes);
            let log_prob: f64 = kani::any();
            kani::assume(log_prob.is_finite() && log_prob.abs() < 100.0);
            active.push((node_idx, log_prob));
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

    /// Prove `top_k_by_value` returns results sorted by value descending.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn proof_top_k_by_value_sorted_descending() {
        let len: usize = kani::any();
        kani::assume(len >= 2 && len <= 4);
        let k: usize = kani::any();
        kani::assume(k >= 2 && k <= len);

        let mut values = vec![0.0f32; len];
        for v in values.iter_mut() {
            *v = kani::any();
            kani::assume(v.is_finite());
        }

        let result = top_k_by_value(&values, k);
        // Verify descending sort order.
        for i in 1..result.len() {
            assert!(
                result[i - 1].1 >= result[i].1 || result[i - 1].1.is_nan() || result[i].1.is_nan(),
                "top_k results must be sorted descending by value"
            );
        }
    }
}
