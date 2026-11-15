// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Autoregressive generation loop for transformer-based language models.
//!
//! Provides [`GenerationConfig`] and the [`generate`] function that drives
//! token-by-token decoding using a KV cache.
//!
//! # Example
//!
//! ```ignore
//! // NOTE: ignore — requires model with specific forward(input, cache) signature
//! let config = GenerationConfig::new(100)
//!     .with_eos_token_id(2);
//! let output = generate(
//!     |input, cache| model.forward(input, cache),
//!     &[1, 50, 100],  // prompt token ids
//!     &mut KvCache::new(12),
//!     &config,
//!     &Device::Cpu,
//! )?;
//! ```

#[path = "autoregressive_sampling.rs"]
mod sampling;

#[path = "autoregressive_token_sampler.rs"]
mod token_sampler;
use token_sampler::sample_token;

use super::kv_cache::KvCacheBackend;
use crate::dyn_tensor::DynTensor;
use crate::{Device, Result, TensorError};

#[cfg(feature = "rand")]
use rand::rngs::StdRng;
#[cfg(feature = "rand")]
use rand::SeedableRng;

// Tests use sampling helpers directly.
#[cfg(test)]
use sampling::{argmax, top_k_indices, top_p_filter};

/// Configuration for autoregressive text generation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Temperature for sampling (1.0 = no change, <1.0 = sharper, >1.0 = flatter).
    /// Set to 0.0 for greedy decoding (argmax).
    pub temperature: f64,
    /// If set, only sample from the top-k most likely tokens.
    pub top_k: Option<usize>,
    /// If set, only sample from the smallest set of tokens whose cumulative
    /// probability exceeds this threshold (nucleus sampling). Must be in (0, 1].
    /// `top_p = 1.0` disables filtering. Composes with `top_k`: top-k is applied
    /// first, then top-p filters the remaining candidates.
    pub top_p: Option<f64>,
    /// Token ID that signals end of generation.
    pub eos_token_id: Option<usize>,
    /// Optional RNG seed for reproducible categorical sampling.
    /// When `Some(n)` and temperature > 0, tokens are sampled from the probability
    /// distribution rather than using argmax. When `None`, argmax is used regardless
    /// of temperature (backward-compatible). Requires the `rand` feature.
    pub seed: Option<u64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            temperature: 0.0,
            top_k: None,
            top_p: None,
            eos_token_id: None,
            seed: None,
        }
    }
}

impl GenerationConfig {
    /// Create config with a maximum new token count. Other fields use defaults.
    #[must_use]
    pub fn new(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            ..Default::default()
        }
    }

    /// Set sampling temperature (0.0 = greedy, 1.0 = standard).
    #[must_use]
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top-k filtering.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set nucleus (top-p) filtering.
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

    /// Set RNG seed for reproducible sampling.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Validate generation config parameters.
    ///
    /// Rejects NaN, Inf, and negative temperature. Rejects invalid top_p.
    /// Temperature of 0.0 is valid (greedy decoding).
    pub fn validate(&self) -> Result<()> {
        if self.temperature < 0.0 || !self.temperature.is_finite() {
            return Err(TensorError::ValueOutOfRange {
                description: "GenerationConfig: temperature must be finite and >= 0.0",
            });
        }
        if let Some(p) = self.top_p {
            if p <= 0.0 || p > 1.0 || !p.is_finite() {
                return Err(TensorError::ValueOutOfRange {
                    description: "GenerationConfig: top_p must be in (0.0, 1.0]",
                });
            }
        }
        Ok(())
    }
}

/// Output from autoregressive generation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GenerationOutput {
    /// Generated token IDs (not including the prompt).
    pub token_ids: Vec<usize>,
    /// Whether generation stopped due to EOS token (vs max_new_tokens).
    pub finished: bool,
}

impl GenerationOutput {
    /// Create a generation output.
    pub fn new(token_ids: Vec<usize>, finished: bool) -> Self {
        Self {
            token_ids,
            finished,
        }
    }
}

/// Run autoregressive generation with a model forward function.
///
/// `model_fn` takes `(input_tensor, &mut C)` and returns logits shaped
/// `[batch, vocab_size]` (only the last token position's logits).
///
/// `prompt_ids` are the initial token IDs to prefill the cache with.
///
/// `cache` can be any [`KvCacheBackend`] — [`KvCache`] provides O(1)
/// amortized appends via doubling buffers.
///
/// `device` controls where token tensors are created (should match the model's device).
///
/// Returns the generated token IDs (not including the prompt).
pub fn generate<C, F>(
    model_fn: F,
    prompt_ids: &[usize],
    cache: &mut C,
    config: &GenerationConfig,
    device: &Device,
) -> Result<GenerationOutput>
where
    C: KvCacheBackend,
    F: Fn(&DynTensor, &mut C) -> Result<DynTensor>,
{
    config.validate()?;
    if prompt_ids.is_empty() {
        return Err(TensorError::InvalidShape(
            "generate: prompt_ids must not be empty".into(),
        ));
    }
    if config.max_new_tokens == 0 {
        return Ok(GenerationOutput {
            token_ids: Vec::new(),
            finished: false,
        });
    }

    #[cfg(feature = "rand")]
    let mut rng = config.seed.map(StdRng::seed_from_u64);

    // Prefill: process the entire prompt through the model
    let prompt_tensor = ids_to_tensor(prompt_ids, device)?;
    let logits = model_fn(&prompt_tensor, cache)?;

    // Sample first token from prefill logits
    #[cfg(feature = "rand")]
    let first_token = sample_token(&logits, config, rng.as_mut())?;
    #[cfg(not(feature = "rand"))]
    let first_token = sample_token(&logits, config)?;

    let mut generated = vec![first_token];
    let mut last_token = first_token;

    if is_eos(first_token, config) {
        return Ok(GenerationOutput {
            token_ids: generated,
            finished: true,
        });
    }

    // Decode: generate one token at a time
    for _ in 1..config.max_new_tokens {
        let input = ids_to_tensor(&[last_token], device)?;
        let logits = model_fn(&input, cache)?;

        #[cfg(feature = "rand")]
        let token = sample_token(&logits, config, rng.as_mut())?;
        #[cfg(not(feature = "rand"))]
        let token = sample_token(&logits, config)?;

        generated.push(token);
        last_token = token;

        if is_eos(token, config) {
            return Ok(GenerationOutput {
                token_ids: generated,
                finished: true,
            });
        }
    }

    Ok(GenerationOutput {
        token_ids: generated,
        finished: false,
    })
}

/// Convert token IDs to a 2D DynTensor `[1, seq_len]` with U32 dtype.
///
/// Uses `from_vec_u32` instead of f32 to avoid precision loss for IDs > 2^24.
/// `Embedding::forward()` handles U32 inputs natively.
fn ids_to_tensor(ids: &[usize], device: &Device) -> Result<DynTensor> {
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
fn is_eos(token: usize, config: &GenerationConfig) -> bool {
    config.eos_token_id.is_some_and(|eos| token == eos)
}

#[cfg(test)]
#[path = "autoregressive_tests.rs"]
mod tests;
