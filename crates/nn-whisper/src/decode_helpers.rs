// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Decode helper functions: sampling, suppression, finiteness checks.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::check_output_finite;
use nn_core::Result;
use rand::rngs::StdRng;
use rand::RngExt;

/// Check logit tensor for NaN/Inf values (GPU-native, no CPU round-trip).
///
/// Uses `check_output_finite` which delegates to `GpuBackend::count_non_finite`
/// on Metal tensors, avoiding `to_flat_vec::<f32>()`. Called before token
/// suppression, so no legitimate -Inf should be present.
pub(crate) fn check_logit_finiteness(logits: &DynTensor, step: usize) -> Result<()> {
    check_output_finite(logits, &format!("decode_logits_step_{step}"))
}

/// Apply token suppression in-place by setting suppressed logit positions to negative infinity.
///
/// `last_logits` is the mutable slice for the last time step's logits (length = vocab_size).
pub(crate) fn apply_suppression_inplace(last_logits: &mut [f32], suppress_tokens: &[usize]) {
    for &tok in suppress_tokens {
        if tok < last_logits.len() {
            last_logits[tok] = f32::NEG_INFINITY;
        }
    }
}

/// Sample token from the last-step logits with temperature.
///
/// `last_logits` is a slice of vocab-sized logits for the last time step.
/// At temperature 0.0 (or very small), uses argmax (greedy).
/// At positive temperature with an RNG, samples from the categorical distribution.
/// At positive temperature without an RNG, falls back to argmax.
pub(crate) fn sample_token(
    last_logits: &[f32],
    temperature: f64,
    rng: Option<&mut StdRng>,
) -> (usize, f32) {
    if temperature < 1e-8 {
        let idx = argmax_f32(last_logits);
        let log_prob = compute_log_prob(last_logits, idx);
        return (idx, log_prob);
    }

    // Temperature-scaled softmax in a single allocation.
    // Fuses scale → exp → normalize into one pass over a single buffer.
    // Guard: f64→f32 cast saturates to INFINITY for temperature > f32::MAX (~3.4e38).
    // Fall back to greedy when the cast produces a non-finite f32.
    let temp_f32 = temperature as f32;
    if !temp_f32.is_finite() || temp_f32 == 0.0 {
        let idx = argmax_f32(last_logits);
        let log_prob = compute_log_prob(last_logits, idx);
        return (idx, log_prob);
    }
    let mut probs: Vec<f32> = last_logits.iter().map(|&v| v / temp_f32).collect();

    let max_val = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in probs.iter_mut() {
        *v = (*v - max_val).exp();
    }
    let sum: f32 = probs.iter().sum();

    if !sum.is_finite() || sum == 0.0 {
        let idx = argmax_f32(last_logits);
        let log_prob = compute_log_prob(last_logits, idx);
        return (idx, log_prob);
    }

    for v in probs.iter_mut() {
        *v /= sum;
    }

    let idx = match rng {
        Some(rng) => categorical_sample(&probs, rng),
        None => argmax_f32(last_logits),
    };

    let log_prob = compute_log_prob(last_logits, idx);
    (idx, log_prob)
}

/// Argmax over a slice of f32 values.
///
/// Uses `f32::total_cmp` for deterministic ordering. With `total_cmp`, NaN sorts
/// above all other values, so NaN inputs are caught by the upstream
/// `check_logit_finiteness` guard rather than silently producing arbitrary results
/// from `partial_cmp(...).unwrap_or(Equal)`.
pub(crate) fn argmax_f32(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Sample from a categorical distribution using inverse CDF.
fn categorical_sample(probs: &[f32], rng: &mut StdRng) -> usize {
    let r: f32 = rng.random();
    let mut cumsum = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

/// Compute log-probability at a given index using log-softmax.
///
/// Returns `NEG_INFINITY` if `logits` is empty or `idx` is out of bounds.
pub(crate) fn compute_log_prob(logits: &[f32], idx: usize) -> f32 {
    if idx >= logits.len() {
        return f32::NEG_INFINITY;
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_val == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    let log_sum_exp: f32 = logits
        .iter()
        .map(|&v| (v - max_val).exp())
        .sum::<f32>()
        .ln()
        + max_val;
    logits[idx] - log_sum_exp
}
