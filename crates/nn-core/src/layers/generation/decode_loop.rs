// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-resident KV cache decode loop for compiled transformers.
//!
//! Provides [`DecodeContext`] (per-model KV cache state) and three entry
//! points for autoregressive generation:
//!
//! - [`prefill()`] — full sequence forward, populating the KV cache
//! - [`decode_step()`] — single-token decode with cache update
//! - [`decode_generate()`] — complete loop: prefill + N decode steps
//!
//! # Key insight
//!
//! During decode, Q is `[B, 1, D]` (single new token), but K/V grow each
//! step via the cache. The cache stores accumulated K/V tensors shaped
//! `[B, num_kv_heads, seq_len, head_dim]`, growing along dim=2 per step.
//!
//! # Cache backend
//!
//! [`DecodeContext`] is generic over [`KvCacheBackend`], so it works with
//! both [`KvCache`] (doubling buffers) and [`PreallocKvCache`] (fixed-capacity
//! GPU-resident buffers for compiled models).
//!
//! # Example
//!
//! ```ignore
//! // NOTE: ignore — requires model with specific forward(input, cache) signature
//! let mut ctx = DecodeContext::new(PreallocKvCache::new(12, 2048)?, 2048);
//! let output = decode_generate(
//!     |input, cache| model.forward(input, cache),
//!     &[1, 50, 100],  // prompt token ids
//!     &mut ctx,
//!     &GenerationConfig::new(100).with_eos_token_id(2),
//!     &Device::Cpu,
//! )?;
//! ```

use super::autoregressive::GenerationConfig;
use super::kv_cache::KvCacheBackend;
use crate::dyn_tensor::DynTensor;
use crate::{Device, Result, TensorError};

#[cfg(feature = "rand")]
use rand::rngs::StdRng;
#[cfg(feature = "rand")]
use rand::SeedableRng;

use super::autoregressive::GenerationOutput;

// Sampling helpers extracted to keep this file under 500 lines.
#[path = "decode_loop_sampling.rs"]
mod sampling;
use sampling::sample_from_logits;

// ---------------------------------------------------------------------------
// DecodeContext
// ---------------------------------------------------------------------------

/// KV cache context for a compiled transformer decode loop.
///
/// Wraps a [`KvCacheBackend`] with bookkeeping for the current sequence
/// position. The cache backend determines GPU residency strategy:
///
/// - [`PreallocKvCache`](super::PreallocKvCache) — fixed-capacity, no
///   mid-inference reallocation, ideal for compiled models.
/// - [`KvCache`](super::KvCache) — doubling buffers, flexible capacity.
///
/// [`DecodeContext`] is `Clone` when the underlying cache is `Clone`.
#[derive(Debug)]
pub struct DecodeContext<C: KvCacheBackend> {
    cache: C,
    /// Maximum total sequence length (prompt + generated tokens).
    max_seq_len: usize,
    /// Number of tokens generated so far (excluding prompt).
    generated_count: usize,
}

impl<C: KvCacheBackend> DecodeContext<C> {
    /// Create a new decode context with the given cache backend.
    ///
    /// `max_seq_len` is the maximum total sequence length the model supports
    /// (prompt + generated tokens combined). Set this to the model's context
    /// window size (e.g., 2048 for GPT-2, 131072 for Qwen3 YaRN).
    #[must_use]
    pub fn new(cache: C, max_seq_len: usize) -> Self {
        Self {
            cache,
            max_seq_len,
            generated_count: 0,
        }
    }

    /// Current total sequence length in the cache (prompt + generated).
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.cache.seq_len()
    }

    /// Number of tokens generated so far (excluding the prompt).
    #[must_use]
    pub fn generated_count(&self) -> usize {
        self.generated_count
    }

    /// Maximum total sequence length.
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Remaining capacity in sequence positions.
    #[must_use]
    pub fn remaining_capacity(&self) -> usize {
        self.max_seq_len.saturating_sub(self.cache.seq_len())
    }

    /// Whether the context window is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.cache.seq_len() >= self.max_seq_len
    }

    /// Mutable access to the underlying cache backend.
    ///
    /// Use this to pass the cache to model forward functions that take
    /// `&mut C` directly.
    pub fn cache_mut(&mut self) -> &mut C {
        &mut self.cache
    }

    /// Immutable access to the underlying cache backend.
    pub fn cache(&self) -> &C {
        &self.cache
    }

    /// Number of layers in the cache.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.cache.num_layers()
    }

    /// Reset the context: clear cache and generation counter.
    pub fn reset(&mut self) {
        self.cache.reset();
        self.generated_count = 0;
    }

    /// Clear cached entries but preserve buffer allocations for reuse.
    ///
    /// Resets the generation counter. Ideal for batch inference where
    /// successive inputs reuse the same buffer.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.generated_count = 0;
    }
}

// ---------------------------------------------------------------------------
// Prefill
// ---------------------------------------------------------------------------

/// Run the prefill phase: process the full prompt through the model and
/// populate the KV cache.
///
/// `model_fn` takes `(input_tensor, &mut C)` where:
/// - `input_tensor` is shaped `[1, prompt_len]` (token IDs as U32)
/// - `C` is the KV cache backend
///
/// Returns the logits from the last prompt position (for sampling the
/// first generated token).
///
/// # Errors
///
/// - Empty prompt
/// - Prompt exceeds `ctx.max_seq_len`
/// - Model forward fails
pub fn prefill<C, F>(
    model_fn: &F,
    prompt_ids: &[usize],
    ctx: &mut DecodeContext<C>,
    device: &Device,
) -> Result<DynTensor>
where
    C: KvCacheBackend,
    F: Fn(&DynTensor, &mut C) -> Result<DynTensor>,
{
    if prompt_ids.is_empty() {
        return Err(TensorError::InvalidShape(
            "prefill: prompt_ids must not be empty".into(),
        ));
    }
    if prompt_ids.len() > ctx.max_seq_len {
        return Err(TensorError::ValueOutOfRange {
            description: "prefill: prompt exceeds max_seq_len",
        });
    }

    // Reset before prefill to ensure clean state.
    ctx.reset();

    let prompt_tensor = ids_to_tensor(prompt_ids, device)?;
    let logits = model_fn(&prompt_tensor, &mut ctx.cache)?;
    Ok(logits)
}

// ---------------------------------------------------------------------------
// Decode step
// ---------------------------------------------------------------------------

/// Run a single decode step: feed one token, update KV cache, return logits.
///
/// `model_fn` takes `(input_tensor, &mut C)` where:
/// - `input_tensor` is shaped `[1, 1]` (single token ID as U32)
/// - The model uses the existing cache for K/V context and appends new K/V
///
/// Returns logits shaped `[1, vocab_size]` or `[1, 1, vocab_size]`.
///
/// # Errors
///
/// - Context window full (`ctx.is_full()`)
/// - Model forward fails
pub fn decode_step<C, F>(
    model_fn: &F,
    token_id: usize,
    ctx: &mut DecodeContext<C>,
    device: &Device,
) -> Result<DynTensor>
where
    C: KvCacheBackend,
    F: Fn(&DynTensor, &mut C) -> Result<DynTensor>,
{
    if ctx.is_full() {
        return Err(TensorError::ValueOutOfRange {
            description: "decode_step: context window is full",
        });
    }

    let input = ids_to_tensor(&[token_id], device)?;
    let logits = model_fn(&input, &mut ctx.cache)?;
    ctx.generated_count += 1;
    Ok(logits)
}

// ---------------------------------------------------------------------------
// Generate (prefill + decode loop)
// ---------------------------------------------------------------------------

/// Complete generation loop: prefill the prompt, then decode up to
/// `config.max_new_tokens` tokens.
///
/// This is the compiled-model counterpart to [`generate()`](super::generate),
/// using [`DecodeContext`] for explicit cache lifecycle management.
///
/// `model_fn` takes `(input_tensor, &mut C)` and returns logits. During
/// prefill, `input_tensor` is `[1, prompt_len]`. During decode steps,
/// `input_tensor` is `[1, 1]`.
///
/// # Sampling
///
/// Uses the same sampling pipeline as [`generate()`](super::generate):
/// temperature scaling, top-k, top-p (nucleus), with argmax fallback.
/// Enable the `rand` feature for categorical sampling.
///
/// # Errors
///
/// - Empty prompt
/// - Prompt exceeds `ctx.max_seq_len`
/// - `config` validation fails
/// - Model forward fails
pub fn decode_generate<C, F>(
    model_fn: F,
    prompt_ids: &[usize],
    ctx: &mut DecodeContext<C>,
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
            "decode_generate: prompt_ids must not be empty".into(),
        ));
    }
    if config.max_new_tokens == 0 {
        return Ok(GenerationOutput::new(Vec::new(), false));
    }

    #[cfg(feature = "rand")]
    let mut rng = config.seed.map(StdRng::seed_from_u64);

    // Prefill phase.
    let logits = prefill(&model_fn, prompt_ids, ctx, device)?;

    // Sample first token from prefill logits.
    let first_token = sample_from_logits(
        &logits,
        config,
        #[cfg(feature = "rand")]
        rng.as_mut(),
    )?;

    let mut generated = vec![first_token];
    let mut last_token = first_token;

    if is_eos(first_token, config) {
        return Ok(GenerationOutput::new(generated, true));
    }

    // Decode loop.
    for _ in 1..config.max_new_tokens {
        if ctx.is_full() {
            break;
        }

        let logits = decode_step(&model_fn, last_token, ctx, device)?;

        let token = sample_from_logits(
            &logits,
            config,
            #[cfg(feature = "rand")]
            rng.as_mut(),
        )?;

        generated.push(token);
        last_token = token;

        if is_eos(token, config) {
            return Ok(GenerationOutput::new(generated, true));
        }
    }

    Ok(GenerationOutput::new(generated, false))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert token IDs to a 2D DynTensor `[1, seq_len]` with U32 dtype.
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

#[cfg(kani)]
#[path = "kani_decode_loop.rs"]
mod kani_decode_loop;

#[cfg(test)]
#[path = "decode_loop_tests.rs"]
mod tests;
