// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Token sampling pipeline for autoregressive generation.
//!
//! Extracted from `autoregressive.rs` to keep the parent under 400 lines.
//! Contains `sample_token`, `extract_vocab_logits`, distribution sampling,
//! and the no-rand fallback path.

use super::sampling::{argmax, top_k_indices};
use super::GenerationConfig;
use crate::dyn_tensor::DynTensor;
use crate::{Device, Result, TensorError};

#[cfg(feature = "rand")]
use super::sampling::top_p_filter;
#[cfg(feature = "rand")]
use rand::rngs::StdRng;
#[cfg(feature = "rand")]
use rand::RngExt;

/// Sample a token from logits using the generation config.
///
/// Expects logits shaped `[batch, vocab_size]` or `[batch, seq_len, vocab_size]`
/// (uses last position if 3D).
///
/// When `rand` feature is enabled and `rng` is `Some`, uses categorical sampling
/// from the softmax distribution. Otherwise falls back to argmax.
#[cfg(feature = "rand")]
pub(super) fn sample_token(
    logits: &DynTensor,
    config: &GenerationConfig,
    rng: Option<&mut StdRng>,
) -> Result<usize> {
    let vocab_logits = extract_vocab_logits(logits)?;

    match rng {
        Some(rng) if config.temperature != 0.0 => {
            sample_from_distribution(&vocab_logits, config, rng)
        }
        _ => Ok(argmax(&vocab_logits)),
    }
}

/// Sample a token from logits using the generation config (no-rand fallback).
#[cfg(not(feature = "rand"))]
pub(super) fn sample_token(logits: &DynTensor, config: &GenerationConfig) -> Result<usize> {
    let vocab_logits = extract_vocab_logits(logits)?;

    if config.temperature == 0.0 {
        return Ok(argmax(&vocab_logits));
    }

    // Without rand feature, temperature > 0 still falls back to argmax over
    // temperature-scaled top-k candidates for backward compatibility.
    sample_argmax_from_candidates(&vocab_logits, config)
}

/// Extract vocabulary logits from a 2D or 3D logits tensor.
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
            "sample_token: empty vocabulary".into(),
        ));
    }

    Ok(vocab_logits)
}

/// Compute softmax probabilities over (optionally top-k filtered) candidates,
/// then sample from the categorical distribution using the provided RNG.
#[cfg(feature = "rand")]
fn sample_from_distribution(
    vocab_logits: &[f32],
    config: &GenerationConfig,
    rng: &mut StdRng,
) -> Result<usize> {
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

    // Apply top-p (nucleus) filtering: keep smallest set whose cumulative
    // probability exceeds the threshold, then renormalize.
    if let Some(p) = config.top_p {
        if p < 1.0 {
            probs = top_p_filter(probs, p as f32);
        }
    }

    Ok(categorical_sample(&probs, rng))
}

/// Categorical sampling via inverse CDF: draw from the probability distribution.
#[cfg(feature = "rand")]
fn categorical_sample(probs: &[(usize, f32)], rng: &mut StdRng) -> usize {
    let r: f32 = rng.random();
    let mut cumsum = 0.0_f32;
    for &(idx, p) in probs {
        cumsum += p;
        if r < cumsum {
            return idx;
        }
    }
    // Fallback to last candidate (rounding error)
    probs.last().map(|&(idx, _)| idx).unwrap_or(0)
}

/// Argmax over top-k/top-p candidates (backward-compatible no-rand path).
#[cfg(not(feature = "rand"))]
fn sample_argmax_from_candidates(vocab_logits: &[f32], config: &GenerationConfig) -> Result<usize> {
    let scaled: Vec<f32> = vocab_logits
        .iter()
        .map(|&v| v / config.temperature as f32)
        .collect();

    let candidates = if let Some(k) = config.top_k {
        top_k_indices(&scaled, k)
    } else if config.top_p.is_some() {
        // No top-k but top-p set: start with all candidates.
        (0..scaled.len()).collect()
    } else {
        return Ok(argmax(&scaled));
    };

    if candidates.is_empty() {
        return Ok(argmax(vocab_logits));
    }

    // Top-p in no-rand mode doesn't change argmax result (the highest-probability
    // token is always included in the nucleus set), but we apply it for consistency.

    Ok(candidates
        .into_iter()
        .max_by(|&a, &b| scaled[a].total_cmp(&scaled[b]))
        .unwrap_or(0))
}
