// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Sampling helpers for the decode loop.
//!
//! Extracted from `decode_loop.rs` to keep the parent under 500 lines.
//! Contains `sample_from_logits`, `extract_vocab_logits`, `argmax`, and the
//! `rand`-gated distribution sampling pipeline.

use crate::dyn_tensor::DynTensor;
use crate::layers::generation::autoregressive::GenerationConfig;
use crate::{Device, Result, TensorError};

#[cfg(feature = "rand")]
use rand::rngs::StdRng;

/// Sample a token from logits using the generation config.
///
/// Handles both 2D `[B, V]` and 3D `[B, T, V]` logits (takes last position).
/// Delegates to argmax when temperature is 0 or no RNG is available.
pub(super) fn sample_from_logits(
    logits: &DynTensor,
    config: &GenerationConfig,
    #[cfg(feature = "rand")] rng: Option<&mut StdRng>,
) -> Result<usize> {
    let vocab_logits = extract_vocab_logits(logits)?;

    #[cfg(feature = "rand")]
    {
        if let Some(rng) = rng {
            if config.temperature != 0.0 {
                return sample_from_distribution(&vocab_logits, config, rng);
            }
        }
    }

    // Greedy: argmax.
    let _ = config; // suppress unused warning in no-rand path
    Ok(argmax(&vocab_logits))
}

/// Extract vocabulary logits as flat Vec<f32> from 2D or 3D tensor.
fn extract_vocab_logits(logits: &DynTensor) -> Result<Vec<f32>> {
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
            "sample_from_logits: empty vocabulary".into(),
        ));
    }

    Ok(vocab_logits)
}

/// Argmax over a slice of f32.
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Categorical sampling from temperature-scaled, top-k/top-p filtered logits.
#[cfg(feature = "rand")]
fn sample_from_distribution(
    vocab_logits: &[f32],
    config: &GenerationConfig,
    rng: &mut StdRng,
) -> Result<usize> {
    use rand::RngExt;

    let scaled: Vec<f32> = vocab_logits
        .iter()
        .map(|&v| v / config.temperature as f32)
        .collect();

    let candidates = if let Some(k) = config.top_k {
        top_k_indices(&scaled, k)
    } else {
        (0..scaled.len()).collect()
    };

    if candidates.is_empty() {
        return Ok(argmax(vocab_logits));
    }

    let max_val = candidates
        .iter()
        .map(|&i| scaled[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = candidates
        .iter()
        .map(|&i| (scaled[i] - max_val).exp())
        .sum();

    if !exp_sum.is_finite() || exp_sum == 0.0 {
        return Ok(argmax(vocab_logits));
    }

    let mut probs: Vec<(usize, f32)> = candidates
        .iter()
        .map(|&i| (i, (scaled[i] - max_val).exp() / exp_sum))
        .collect();

    if let Some(p) = config.top_p {
        if p < 1.0 {
            probs = top_p_filter(probs, p as f32);
        }
    }

    // Inverse CDF sampling.
    let r: f32 = rng.random();
    let mut cumsum = 0.0_f32;
    for &(idx, p) in &probs {
        cumsum += p;
        if r < cumsum {
            return Ok(idx);
        }
    }
    Ok(probs.last().map(|&(idx, _)| idx).unwrap_or(0))
}

/// Return indices of top-k values (sorted descending).
#[cfg(feature = "rand")]
fn top_k_indices(values: &[f32], k: usize) -> Vec<usize> {
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

/// Top-p (nucleus) filtering.
#[cfg(feature = "rand")]
fn top_p_filter(mut probs: Vec<(usize, f32)>, p: f32) -> Vec<(usize, f32)> {
    if probs.is_empty() {
        return probs;
    }
    probs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    let mut cumsum = 0.0_f32;
    let mut cutoff = probs.len();
    for (i, &(_, prob)) in probs.iter().enumerate() {
        cumsum += prob;
        if cumsum >= p {
            cutoff = i + 1;
            break;
        }
    }
    cutoff = cutoff.max(1);
    probs.truncate(cutoff);

    let total: f32 = probs.iter().map(|&(_, prob)| prob).sum();
    if total > 0.0 && total.is_finite() {
        for item in &mut probs {
            item.1 /= total;
        }
    }
    probs
}
