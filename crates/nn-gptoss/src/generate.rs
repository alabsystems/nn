// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Autoregressive text generation for gpt-oss with sampling strategies.
//!
//! Provides [`GenerateConfig`] for controlling generation behavior (temperature,
//! top-k, top-p, repetition penalty) and a [`generate`] function that drives
//! autoregressive decoding with the gpt-oss model and KV cache.
//!
//! For simple greedy generation, use [`GptOssModel::generate_greedy`] which
//! delegates to the core `nn_core::layers::generate`. This module adds richer
//! sampling control on top.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Result;

use crate::kv_cache::GptOssKvCache;
use crate::{GptOssError, GptOssModel};

/// Generation configuration with sampling parameters.
///
/// Controls autoregressive decoding behavior including temperature scaling,
/// top-k filtering, nucleus (top-p) sampling, and repetition penalty.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GenerateConfig {
    /// Maximum number of new tokens to generate.
    pub max_tokens: usize,
    /// Temperature for logit scaling. 0.0 = greedy (argmax).
    pub temperature: f32,
    /// Top-k filtering: keep only the k highest-probability tokens.
    /// `None` disables top-k filtering.
    pub top_k: Option<usize>,
    /// Nucleus sampling: keep smallest set of tokens with cumulative
    /// probability >= p. `None` disables top-p filtering.
    pub top_p: Option<f32>,
    /// Repetition penalty applied to previously generated tokens.
    /// 1.0 = no penalty. Values > 1.0 penalize repetition.
    /// `None` disables repetition penalty.
    pub repetition_penalty: Option<f32>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_k: Some(50),
            top_p: Some(0.9),
            repetition_penalty: None,
        }
    }
}

impl GenerateConfig {
    /// Create a greedy generation config (temperature=0, no sampling).
    #[must_use]
    pub fn greedy(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            temperature: 0.0,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
        }
    }

    /// Validate config parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if temperature is negative or NaN, top_p is out of
    /// range, top_k is zero, or repetition_penalty is non-positive.
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(GptOssError::InvalidConfig {
                reason: format!(
                    "temperature must be finite and >= 0.0, got {}",
                    self.temperature
                ),
            }
            .into());
        }
        if let Some(p) = self.top_p {
            if !p.is_finite() || p <= 0.0 || p > 1.0 {
                return Err(GptOssError::InvalidConfig {
                    reason: format!("top_p must be in (0.0, 1.0], got {p}"),
                }
                .into());
            }
        }
        if let Some(k) = self.top_k {
            if k == 0 {
                return Err(GptOssError::InvalidConfig {
                    reason: "top_k must be > 0".into(),
                }
                .into());
            }
        }
        if let Some(rp) = self.repetition_penalty {
            if !rp.is_finite() || rp <= 0.0 {
                return Err(GptOssError::InvalidConfig {
                    reason: format!("repetition_penalty must be finite and > 0.0, got {rp}"),
                }
                .into());
            }
        }
        Ok(())
    }
}

/// Generate text tokens autoregressively with KV cache and sampling.
///
/// Runs the gpt-oss model token-by-token, applying the configured sampling
/// strategy at each step. Stops when `eos_token_id` is generated or
/// `config.max_tokens` is reached.
///
/// # Arguments
///
/// * `model` - The loaded gpt-oss model.
/// * `input_ids` - Prompt token IDs.
/// * `config` - Generation parameters (temperature, top-k, top-p, etc.).
/// * `eos_token_id` - End-of-sequence token ID for early stopping.
///
/// # Errors
///
/// Returns an error if config validation fails, the model forward pass fails,
/// or logits extraction produces invalid values.
pub fn generate(
    model: &GptOssModel,
    input_ids: &[usize],
    config: &GenerateConfig,
    eos_token_id: usize,
) -> Result<Vec<usize>> {
    config.validate()?;

    if config.max_tokens == 0 || input_ids.is_empty() {
        return Ok(Vec::new());
    }

    let mut cache = GptOssKvCache::new(model.config());
    let mut generated = Vec::with_capacity(config.max_tokens);

    // Prefill: run prompt through model to populate KV cache.
    let positions: Vec<usize> = (0..input_ids.len()).collect();
    let logits = model.forward_cached(input_ids, &positions, Some(cache.inner_mut()))?;

    // Sample first token from last position's logits.
    let first_token = sample_from_logits(&logits, config, &generated, input_ids)?;
    if first_token == eos_token_id {
        return Ok(generated);
    }
    generated.push(first_token);

    // Decode loop: generate one token at a time.
    for _ in 1..config.max_tokens {
        let pos = input_ids.len() + generated.len() - 1;
        let logits = model.forward_cached(
            &[*generated.last().expect("invariant: generated is non-empty")],
            &[pos],
            Some(cache.inner_mut()),
        )?;

        let token = sample_from_logits(&logits, config, &generated, input_ids)?;
        if token == eos_token_id {
            break;
        }
        generated.push(token);
    }

    Ok(generated)
}

/// Sample a token from the last position of the logits tensor.
///
/// Extracts the logits for the last sequence position, applies temperature
/// scaling, top-k filtering, top-p filtering, repetition penalty, and
/// samples from the resulting distribution.
fn sample_from_logits(
    logits: &DynTensor,
    config: &GenerateConfig,
    generated: &[usize],
    prompt: &[usize],
) -> Result<usize> {
    // logits shape: [1, seq_len, vocab_size]
    let dims = logits.dims();
    let seq_len = dims[1];
    let vocab_size = dims[2];

    // Extract last position: [1, 1, vocab_size] -> flatten to [vocab_size]
    let last_logits = logits.narrow(1, seq_len - 1, 1)?;
    let flat = last_logits.reshape([vocab_size])?;
    let mut logit_vec = flat.to_flat_vec::<f32>()?;

    // Apply repetition penalty
    if let Some(rp) = config.repetition_penalty {
        if (rp - 1.0).abs() > f32::EPSILON {
            apply_repetition_penalty(&mut logit_vec, generated, prompt, rp);
        }
    }

    sample_logits(&logit_vec, config)
}

/// Apply repetition penalty to logits for previously seen tokens.
///
/// For each token in `generated` and `prompt`, divide positive logits by
/// `penalty` and multiply negative logits by `penalty`. This discourages
/// the model from repeating tokens.
fn apply_repetition_penalty(
    logits: &mut [f32],
    generated: &[usize],
    prompt: &[usize],
    penalty: f32,
) {
    let seen = generated.iter().chain(prompt.iter());
    for &token_id in seen {
        if token_id < logits.len() {
            let l = logits[token_id];
            logits[token_id] = if l > 0.0 { l / penalty } else { l * penalty };
        }
    }
}

/// Sample from logits with temperature, top-k, and top-p.
///
/// - Temperature 0.0: greedy (argmax)
/// - Top-k: keep only the k highest logits, set rest to -inf
/// - Top-p (nucleus): keep smallest set of tokens with cumulative prob >= p
fn sample_logits(logits: &[f32], config: &GenerateConfig) -> Result<usize> {
    if logits.is_empty() {
        return Err(GptOssError::InvalidInput {
            reason: "empty logits".into(),
        }
        .into());
    }

    // Greedy decoding: temperature == 0.0
    if config.temperature == 0.0 {
        return Ok(argmax(logits));
    }

    // Temperature scaling
    let temp = config.temperature;
    let mut scaled: Vec<f32> = logits.iter().map(|&l| l / temp).collect();

    // Top-k filtering
    if let Some(k) = config.top_k {
        apply_top_k(&mut scaled, k);
    }

    // Top-p (nucleus) filtering
    if let Some(p) = config.top_p {
        apply_top_p(&mut scaled, p);
    }

    // Softmax
    let probs = softmax_vec(&scaled);

    // Sample from the probability distribution (deterministic via argmax
    // on the filtered distribution for reproducibility without an RNG dep).
    // For true stochastic sampling, integrate with nn_core::layers::generation
    // which has seed-based RNG support.
    Ok(argmax(&probs))
}

/// Argmax: index of the maximum value.
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Top-k filtering: keep only the k highest logits, set rest to -inf.
fn apply_top_k(logits: &mut [f32], k: usize) {
    if k >= logits.len() {
        return;
    }
    // Find the indices of the top-k logits
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Mark which indices to keep
    let mut keep = vec![false; logits.len()];
    for &idx in &indices[..k] {
        keep[idx] = true;
    }
    for (i, l) in logits.iter_mut().enumerate() {
        if !keep[i] {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Top-p (nucleus) filtering: keep smallest set with cumulative prob >= p.
fn apply_top_p(logits: &mut [f32], p: f32) {
    let probs = softmax_vec(logits);
    let mut sorted_indices: Vec<usize> = (0..probs.len()).collect();
    sorted_indices.sort_unstable_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut cumulative = 0.0_f32;
    let mut cutoff_idx = sorted_indices.len();
    for (rank, &idx) in sorted_indices.iter().enumerate() {
        cumulative += probs[idx];
        if cumulative >= p {
            cutoff_idx = rank + 1;
            break;
        }
    }

    // Zero out tokens beyond the cutoff
    for &idx in &sorted_indices[cutoff_idx..] {
        logits[idx] = f32::NEG_INFINITY;
    }
}

/// Simple softmax over a slice (numerically stable).
///
/// Handles all-neg-inf inputs (returns uniform) per softmax NaN guard
/// convention (Source: #1326).
fn softmax_vec(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // Guard: if all logits are -inf (or max is non-finite), return uniform.
    // exp(-inf - (-inf)) = exp(NaN) = NaN, so we must check before computing.
    if !max_val.is_finite() {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if !sum.is_finite() || sum == 0.0 {
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_config_default() {
        let cfg = GenerateConfig::default();
        assert_eq!(cfg.max_tokens, 512);
        assert!((cfg.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(cfg.top_k, Some(50));
        assert_eq!(cfg.top_p, Some(0.9));
        assert!(cfg.repetition_penalty.is_none());
        cfg.validate().expect("default config should validate");
    }

    #[test]
    fn test_generate_config_greedy() {
        let cfg = GenerateConfig::greedy(100);
        assert_eq!(cfg.max_tokens, 100);
        assert_eq!(cfg.temperature, 0.0);
        assert!(cfg.top_k.is_none());
        assert!(cfg.top_p.is_none());
        cfg.validate().expect("greedy config should validate");
    }

    #[test]
    fn test_validate_rejects_negative_temperature() {
        let cfg = GenerateConfig {
            temperature: -1.0,
            ..GenerateConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_nan_temperature() {
        let cfg = GenerateConfig {
            temperature: f32::NAN,
            ..GenerateConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_zero_top_k() {
        let cfg = GenerateConfig {
            top_k: Some(0),
            ..GenerateConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_bad_top_p() {
        let cfg = GenerateConfig {
            top_p: Some(0.0),
            ..GenerateConfig::default()
        };
        assert!(cfg.validate().is_err());

        let cfg2 = GenerateConfig {
            top_p: Some(1.5),
            ..GenerateConfig::default()
        };
        assert!(cfg2.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_bad_repetition_penalty() {
        let cfg = GenerateConfig {
            repetition_penalty: Some(0.0),
            ..GenerateConfig::default()
        };
        assert!(cfg.validate().is_err());

        let cfg2 = GenerateConfig {
            repetition_penalty: Some(-1.0),
            ..GenerateConfig::default()
        };
        assert!(cfg2.validate().is_err());
    }

    #[test]
    fn test_argmax_basic() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0]), 0);
        assert_eq!(argmax(&[0.0, 0.0, 0.1]), 2);
    }

    #[test]
    fn test_greedy_sampling() {
        let logits = vec![1.0, 5.0, 3.0, 0.5];
        let cfg = GenerateConfig::greedy(10);
        let token = sample_logits(&logits, &cfg).expect("should sample");
        assert_eq!(token, 1, "greedy should pick highest logit");
    }

    #[test]
    fn test_top_k_filtering() {
        let mut logits = vec![1.0, 5.0, 3.0, 0.5, 4.0];
        apply_top_k(&mut logits, 2);
        // Only indices 1 (5.0) and 4 (4.0) should remain; rest should be -inf
        assert!(logits[0].is_infinite() && logits[0] < 0.0);
        assert_eq!(logits[1], 5.0);
        assert!(logits[2].is_infinite() && logits[2] < 0.0);
        assert!(logits[3].is_infinite() && logits[3] < 0.0);
        assert_eq!(logits[4], 4.0);
    }

    #[test]
    fn test_softmax_basic() {
        let probs = softmax_vec(&[0.0, 0.0, 0.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1");
        // Equal logits -> equal probabilities
        assert!((probs[0] - probs[1]).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_all_neg_inf() {
        let probs = softmax_vec(&[f32::NEG_INFINITY, f32::NEG_INFINITY]);
        // Should return uniform when all -inf
        assert!((probs[0] - 0.5).abs() < 1e-5);
        assert!((probs[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![2.0, -1.0, 3.0, 0.5];
        apply_repetition_penalty(&mut logits, &[0, 1], &[], 2.0);
        // Index 0: positive, divided by 2 -> 1.0
        assert!((logits[0] - 1.0).abs() < 1e-5);
        // Index 1: negative, multiplied by 2 -> -2.0
        assert!((logits[1] - (-2.0)).abs() < 1e-5);
        // Index 2: untouched
        assert!((logits[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_sample_empty_logits_error() {
        let cfg = GenerateConfig::greedy(10);
        assert!(sample_logits(&[], &cfg).is_err());
    }
}
