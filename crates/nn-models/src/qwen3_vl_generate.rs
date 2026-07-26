// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! KV-cache-based autoregressive generation for Qwen3-VL.
//!
//! Provides [`Qwen3VLGenerationConfig`] and [`Qwen3VLGenerator`] for
//! efficient prefill + decode generation with KV caching.
//!
//! # Usage
//!
//! ```ignore
//! // NOTE: ignore — requires loaded model weights
//! let gen_cfg = Qwen3VLGenerationConfig::new(128);
//! let mut generator = Qwen3VLGenerator::new(&model);
//! let output = generator.generate(&[1, 50, 100], None, &gen_cfg)?;
//! println!("generated {} tokens", output.token_ids.len());
//! ```

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::KvCache;
use nn_core::{Result, TensorError};

use super::Qwen3VL;

/// Configuration for Qwen3-VL autoregressive generation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Qwen3VLGenerationConfig {
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Temperature for sampling (0.0 = greedy/argmax).
    pub temperature: f64,
    /// Nucleus sampling threshold. `None` disables top-p filtering.
    pub top_p: Option<f64>,
    /// End-of-sequence token ID. Generation stops when this token is produced.
    pub eos_token_id: Option<usize>,
}

impl Default for Qwen3VLGenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            temperature: 0.0,
            top_p: None,
            eos_token_id: None,
        }
    }
}

impl Qwen3VLGenerationConfig {
    /// Create config with a maximum new token count.
    #[must_use]
    pub fn new(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            ..Default::default()
        }
    }

    /// Set sampling temperature.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set nucleus (top-p) sampling threshold.
    #[must_use]
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set end-of-sequence token ID.
    #[must_use]
    pub fn with_eos_token_id(mut self, eos_token_id: usize) -> Self {
        self.eos_token_id = Some(eos_token_id);
        self
    }

    /// Validate config parameters.
    pub fn validate(&self) -> Result<()> {
        if self.temperature < 0.0 || !self.temperature.is_finite() {
            return Err(TensorError::ValueOutOfRange {
                description: "Qwen3VLGenerationConfig: temperature must be finite and >= 0.0",
            });
        }
        if let Some(p) = self.top_p {
            if p <= 0.0 || p > 1.0 || !p.is_finite() {
                return Err(TensorError::ValueOutOfRange {
                    description: "Qwen3VLGenerationConfig: top_p must be in (0.0, 1.0]",
                });
            }
        }
        Ok(())
    }
}

/// Output from Qwen3-VL generation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Qwen3VLGenerationOutput {
    /// Generated token IDs (not including the prompt).
    pub token_ids: Vec<usize>,
    /// Whether generation stopped due to EOS token (vs max_new_tokens).
    pub finished: bool,
}

impl Qwen3VLGenerationOutput {
    /// Create a generation output.
    pub(crate) fn new(token_ids: Vec<usize>, finished: bool) -> Self {
        Self {
            token_ids,
            finished,
        }
    }
}

/// Stateful autoregressive generator for Qwen3-VL with KV cache.
///
/// Holds the KV cache and provides `generate()` for efficient decoding.
/// Call `reset()` between independent generation runs.
pub struct Qwen3VLGenerator<'m> {
    model: &'m Qwen3VL,
    cache: KvCache,
}

impl<'m> Qwen3VLGenerator<'m> {
    /// Create a new generator for the given model.
    pub fn new(model: &'m Qwen3VL) -> Self {
        let cache = model.create_cache();
        Self { model, cache }
    }

    /// Reset the KV cache for a new generation run.
    pub fn reset(&mut self) {
        self.cache.reset();
    }

    /// Current cached sequence length.
    #[must_use]
    pub fn cached_seq_len(&self) -> usize {
        self.cache.seq_len()
    }

    /// Access the KV cache.
    #[must_use]
    pub fn cache(&self) -> &KvCache {
        &self.cache
    }

    /// Run autoregressive generation.
    ///
    /// - `prompt_ids`: initial token IDs to prefill the cache.
    /// - `vision_features`: optional pre-encoded vision tokens `[B, N, vision_hidden]`.
    /// - `config`: generation parameters.
    ///
    /// Returns generated token IDs (not including the prompt).
    pub fn generate(
        &mut self,
        prompt_ids: &[usize],
        vision_features: Option<&DynTensor>,
        config: &Qwen3VLGenerationConfig,
    ) -> Result<Qwen3VLGenerationOutput> {
        config.validate()?;
        if prompt_ids.is_empty() {
            return Err(TensorError::InvalidShape(
                "Qwen3VL generate: prompt_ids must not be empty".into(),
            ));
        }
        if config.max_new_tokens == 0 {
            return Ok(Qwen3VLGenerationOutput::new(Vec::new(), false));
        }

        // Prefill: process full prompt (+ optional vision) through the model
        let logits = self
            .model
            .forward_cached(vision_features, prompt_ids, &mut self.cache)?;

        // Sample first token from prefill logits
        let first_token = sample_greedy_or_temperature(&logits, config)?;
        let mut generated = vec![first_token];

        if is_eos(first_token, config) {
            return Ok(Qwen3VLGenerationOutput::new(generated, true));
        }

        // Decode: generate one token at a time using cached K/V
        for _ in 1..config.max_new_tokens {
            let last_token = *generated.last().ok_or_else(|| {
                TensorError::InvalidShape("Qwen3VL generate: empty generated tokens".into())
            })?;

            let logits = self
                .model
                .forward_cached(None, &[last_token], &mut self.cache)?;

            let token = sample_greedy_or_temperature(&logits, config)?;
            generated.push(token);

            if is_eos(token, config) {
                return Ok(Qwen3VLGenerationOutput::new(generated, true));
            }
        }

        Ok(Qwen3VLGenerationOutput::new(generated, false))
    }
}

/// Sample a token from logits using greedy or temperature-scaled sampling.
///
/// `logits` shape: `[B, 1, vocab_size]` or `[B, S, vocab_size]` (uses last position).
fn sample_greedy_or_temperature(
    logits: &DynTensor,
    config: &Qwen3VLGenerationConfig,
) -> Result<usize> {
    // Extract last-position logits: [vocab_size]
    let rank = logits.rank();
    let last_logits = if rank == 3 {
        let seq_len = logits.dim(1)?;
        logits
            .narrow(1, seq_len.saturating_sub(1), 1)?
            .squeeze(1)?
            .squeeze(0)?
    } else if rank == 2 {
        logits.squeeze(0)?
    } else {
        return Err(TensorError::RankMismatch {
            expected: 3,
            actual: rank,
        });
    };

    let values = last_logits.to_flat_vec::<f32>()?;
    if values.is_empty() {
        return Err(TensorError::InvalidShape(
            "Qwen3VL generate: empty logits".into(),
        ));
    }

    if config.temperature <= 0.0 {
        // Greedy: argmax
        Ok(argmax(&values))
    } else {
        // Temperature-scaled softmax → argmax (deterministic; stochastic requires rand)
        let inv_temp = 1.0 / config.temperature as f32;
        let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scaled: Vec<f32> = values
            .iter()
            .map(|&v| ((v - max_val) * inv_temp).exp())
            .collect();
        let sum: f32 = scaled.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            // Fallback to greedy if temperature scaling produces degenerate distribution
            return Ok(argmax(&values));
        }
        // With top_p filtering
        if let Some(top_p) = config.top_p {
            let top_p = top_p as f32;
            let probs: Vec<f32> = scaled.iter().map(|&v| v / sum).collect();
            let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut cumulative = 0.0_f32;
            for &(idx, prob) in &indexed {
                cumulative += prob;
                if cumulative >= top_p {
                    return Ok(idx);
                }
            }
            // Fallback: return highest-prob token
            Ok(indexed.first().map_or(0, |&(idx, _)| idx))
        } else {
            // Argmax of scaled (equivalent to argmax of raw logits)
            Ok(argmax(&values))
        }
    }
}

/// Return the index of the maximum value (ties broken by first occurrence).
fn argmax(values: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

/// Check whether a token matches the configured EOS token.
fn is_eos(token: usize, config: &Qwen3VLGenerationConfig) -> bool {
    config.eos_token_id.is_some_and(|eos| token == eos)
}

#[cfg(test)]
#[path = "qwen3_vl_generate_tests.rs"]
mod tests;
