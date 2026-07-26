// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Sampling helper functions for autoregressive generation.
//!
//! Extracted from `autoregressive.rs` to keep the parent under 500 lines.
//! Contains `argmax`, `top_k_indices`, and `top_p_filter`.

/// Argmax over a slice of f32.
///
/// Uses `total_cmp` for deterministic NaN ordering: NaN sorts after everything,
/// so if all values are NaN, index 0 is returned (via `unwrap_or(0)`).
pub(super) fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Return indices of top-k values (sorted descending by value).
///
/// Uses partial sort O(D + k log k) instead of full sort O(D log D).
/// Allocates `Vec<usize>` (8 bytes/element) instead of `Vec<(usize, f32)>`
/// (16 bytes/element), halving memory for large vocabularies (e.g., Qwen3 151K).
pub(super) fn top_k_indices(values: &[f32], k: usize) -> Vec<usize> {
    if k == 0 {
        return Vec::new();
    }
    let mut indices: Vec<usize> = (0..values.len()).collect();
    if k < indices.len() {
        indices.select_nth_unstable_by(k - 1, |&a, &b| values[b].total_cmp(&values[a]));
        indices[..k].sort_unstable_by(|&a, &b| values[b].total_cmp(&values[a]));
        indices.truncate(k);
    } else {
        indices.sort_unstable_by(|&a, &b| values[b].total_cmp(&values[a]));
    }
    indices
}

/// Filter probabilities by nucleus (top-p): keep the smallest set of tokens
/// whose cumulative probability exceeds `p`, sorted by descending probability.
/// Returns renormalized (index, probability) pairs.
#[cfg(any(feature = "rand", test, kani))]
pub(super) fn top_p_filter(mut probs: Vec<(usize, f32)>, p: f32) -> Vec<(usize, f32)> {
    if probs.is_empty() {
        return probs;
    }

    // Sort by probability descending.
    probs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    // Find the cutoff index: the smallest set whose cumulative sum >= p.
    let mut cumsum = 0.0_f32;
    let mut cutoff = probs.len();
    for (i, &(_, prob)) in probs.iter().enumerate() {
        cumsum += prob;
        if cumsum >= p {
            cutoff = i + 1;
            break;
        }
    }

    // Always keep at least one token.
    cutoff = cutoff.max(1);
    probs.truncate(cutoff);

    // Renormalize.
    let total: f32 = probs.iter().map(|&(_, prob)| prob).sum();
    if total > 0.0 && total.is_finite() {
        for item in &mut probs {
            item.1 /= total;
        }
    }

    probs
}
