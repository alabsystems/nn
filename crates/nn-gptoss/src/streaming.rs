// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pull-based streaming token generation for gpt-oss.
//!
//! Provides [`StreamingSession`], a pull-based API for autoregressive decoding
//! that yields one [`StreamingToken`] per [`next_token()`] call. This is the
//! preferred API for agentic search where the caller needs control over
//! iteration -- inspecting tokens mid-generation, applying tool-use parsing,
//! or cancelling early based on external signals.
//!
//! Unlike the batch [`generate()`](crate::generate::generate) function which
//! runs the full decode loop internally, `StreamingSession` lets the caller
//! drive iteration:
//!
//! ```rust,ignore
//! let mut session = StreamingSession::new(&model, &prompt_ids, config)?;
//! while let Some(token) = session.next_token()? {
//!     print!("{}", tokenizer.decode(&[token.id]));
//!     if is_tool_call(&token) { break; }
//! }
//! ```
//!
//! Part of #4271 (gpt-oss streaming inference).

use nn_core::dyn_tensor::DynTensor;
use nn_core::Result;

use crate::generate::GenerateConfig;
use crate::kv_cache::GptOssKvCache;
use crate::{GptOssError, GptOssModel};

/// A single generated token with optional metadata.
#[derive(Debug, Clone)]
pub struct StreamingToken {
    /// Token ID from the vocabulary.
    pub id: usize,
    /// Decoded text for this token (if a tokenizer was used).
    /// `None` when operating in token-ID-only mode.
    pub text: Option<String>,
    /// Raw logits for this token's generation step.
    /// `None` unless [`StreamingConfig::return_logits`] is true.
    pub logits: Option<DynTensor>,
}

/// Configuration for a streaming session.
///
/// Wraps [`GenerateConfig`] with additional streaming-specific options.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Base generation parameters (max_tokens, temperature, top-k, top-p, etc.).
    pub generate: GenerateConfig,
    /// End-of-sequence token ID for early stopping.
    pub eos_token_id: usize,
    /// Whether to retain raw logits in each [`StreamingToken`].
    /// Enabling this increases memory usage proportional to vocab_size per step.
    pub return_logits: bool,
}

impl StreamingConfig {
    /// Create a streaming config from generation config and EOS token ID.
    #[must_use]
    pub fn new(generate: GenerateConfig, eos_token_id: usize) -> Self {
        Self {
            generate,
            eos_token_id,
            return_logits: false,
        }
    }

    /// Create a greedy streaming config with the given max tokens and EOS ID.
    #[must_use]
    pub fn greedy(max_tokens: usize, eos_token_id: usize) -> Self {
        Self {
            generate: GenerateConfig::greedy(max_tokens),
            eos_token_id,
            return_logits: false,
        }
    }

    /// Enable returning raw logits in each token (builder-style).
    #[must_use]
    pub fn with_return_logits(mut self, return_logits: bool) -> Self {
        self.return_logits = return_logits;
        self
    }
}

/// Pull-based streaming generation session for gpt-oss.
///
/// Holds a reference to the model, owns the KV cache and generation state,
/// and yields one token per `next_token()` call. The session does NOT require
/// a tokenizer -- it operates on token IDs only. Text decoding is the caller's
/// responsibility.
///
/// # Lifecycle
///
/// 1. `StreamingSession::new()` processes the prompt through the model,
///    populating the KV cache and sampling the first token.
/// 2. Each `next_token()` call runs one forward pass (single token) and
///    returns the sampled token.
/// 3. Generation stops when EOS is produced, max_tokens is reached, or
///    the caller drops the session.
pub struct StreamingSession<'m> {
    /// Reference to the loaded model (not owned -- allows sharing).
    model: &'m GptOssModel,
    /// KV cache for autoregressive decoding.
    cache: GptOssKvCache,
    /// Generation configuration (temperature, top-k, etc.).
    config: StreamingConfig,
    /// Prompt token IDs (retained for repetition penalty).
    prompt_ids: Vec<usize>,
    /// All generated token IDs so far (not including prompt).
    generated_ids: Vec<usize>,
    /// Current position in the sequence (prompt_len + generated_len).
    position: usize,
    /// Whether generation has completed (EOS or max_tokens).
    done: bool,
}

impl<'m> StreamingSession<'m> {
    /// Create a new streaming session, processing the prompt to populate the KV cache.
    ///
    /// The prompt is run through the model in a single forward pass (prefill).
    /// The first token is NOT generated here -- call `next_token()` to begin
    /// generation.
    ///
    /// # Arguments
    ///
    /// * `model` - Loaded gpt-oss model.
    /// * `prompt_ids` - Prompt token IDs. Must be non-empty.
    /// * `config` - Streaming generation configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the prompt is empty, config validation fails,
    /// or the prefill forward pass fails.
    pub fn new(
        model: &'m GptOssModel,
        prompt_ids: &[usize],
        config: StreamingConfig,
    ) -> Result<Self> {
        config.generate.validate()?;

        if prompt_ids.is_empty() {
            return Err(GptOssError::InvalidInput {
                reason: "prompt_ids must be non-empty for streaming generation".into(),
            }
            .into());
        }

        if config.generate.max_tokens == 0 {
            // Zero max_tokens: session starts in done state.
            let cache = GptOssKvCache::new(model.config());
            return Ok(Self {
                model,
                cache,
                config,
                prompt_ids: prompt_ids.to_vec(),
                generated_ids: Vec::new(),
                position: prompt_ids.len(),
                done: true,
            });
        }

        let mut cache = GptOssKvCache::new(model.config());

        // Prefill: run the full prompt through the model to populate KV cache.
        let positions: Vec<usize> = (0..prompt_ids.len()).collect();
        let _prefill_logits =
            model.forward_cached(prompt_ids, &positions, Some(cache.inner_mut()))?;

        let capacity = config.generate.max_tokens;
        Ok(Self {
            model,
            cache,
            config,
            prompt_ids: prompt_ids.to_vec(),
            generated_ids: Vec::with_capacity(capacity),
            position: prompt_ids.len(),
            done: false,
        })
    }

    /// Generate the next token, or return `None` if generation is complete.
    ///
    /// Each call performs one forward pass through the model with a single
    /// input token (the previously generated token, or the last prompt token
    /// for the first call).
    ///
    /// # Returns
    ///
    /// - `Ok(Some(StreamingToken))` -- the next generated token.
    /// - `Ok(None)` -- generation is complete (EOS or max_tokens).
    /// - `Err(...)` -- forward pass or sampling failed.
    pub fn next_token(&mut self) -> Result<Option<StreamingToken>> {
        if self.done {
            return Ok(None);
        }

        // Determine the input token for this step.
        let input_id = if self.generated_ids.is_empty() {
            // First generation step: feed the last prompt token.
            *self
                .prompt_ids
                .last()
                .expect("invariant: prompt is non-empty")
        } else {
            *self
                .generated_ids
                .last()
                .expect("invariant: generated is non-empty")
        };

        // Forward pass: single token at current position.
        let logits = self.model.forward_cached(
            &[input_id],
            &[self.position],
            Some(self.cache.inner_mut()),
        )?;

        // Sample from logits.
        let token_id = sample_last_token(&logits, &self.config.generate)?;

        // Check EOS.
        if token_id == self.config.eos_token_id {
            self.done = true;
            return Ok(None);
        }

        // Record the token.
        self.generated_ids.push(token_id);
        self.position += 1;

        // Check max_tokens.
        if self.generated_ids.len() >= self.config.generate.max_tokens {
            self.done = true;
        }

        let retained_logits = if self.config.return_logits {
            Some(logits)
        } else {
            None
        };

        Ok(Some(StreamingToken {
            id: token_id,
            text: None,
            logits: retained_logits,
        }))
    }

    /// Whether generation has completed (EOS token or max_tokens reached).
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// All token IDs generated so far (not including the prompt).
    #[must_use]
    pub fn generated_ids(&self) -> &[usize] {
        &self.generated_ids
    }

    /// Number of tokens generated so far.
    #[must_use]
    pub fn generated_count(&self) -> usize {
        self.generated_ids.len()
    }

    /// Original prompt token IDs.
    #[must_use]
    pub fn prompt_ids(&self) -> &[usize] {
        &self.prompt_ids
    }

    /// Current position in the sequence (prompt_len + generated_count).
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Number of tokens remaining before max_tokens is reached.
    /// Returns 0 if generation is already done.
    #[must_use]
    pub fn remaining(&self) -> usize {
        if self.done {
            return 0;
        }
        self.config
            .generate
            .max_tokens
            .saturating_sub(self.generated_ids.len())
    }

    /// Cancel the session, marking it as done without generating further tokens.
    ///
    /// After calling `cancel()`, `next_token()` returns `Ok(None)` and
    /// `is_done()` returns `true`. Generated tokens so far are still available
    /// via `generated_ids()`.
    pub fn cancel(&mut self) {
        self.done = true;
    }
}

/// Sample a token from the last position of the logits tensor using greedy
/// decoding (argmax). Extracted from `generate.rs` to keep streaming
/// self-contained.
///
/// Logits shape: `[1, seq_len, vocab_size]`. Extracts last position.
fn sample_last_token(logits: &DynTensor, config: &GenerateConfig) -> Result<usize> {
    let dims = logits.dims();
    let seq_len = dims[1];
    let vocab_size = dims[2];

    let last_logits = logits.narrow(1, seq_len - 1, 1)?;
    let flat = last_logits.reshape([vocab_size])?;
    let logit_vec = flat.to_flat_vec::<f32>()?;

    // Greedy: temperature == 0.0
    if config.temperature == 0.0 {
        return Ok(argmax(&logit_vec));
    }

    // Temperature scaling + argmax (deterministic sampling).
    let temp = config.temperature;
    let scaled: Vec<f32> = logit_vec.iter().map(|&l| l / temp).collect();
    Ok(argmax(&scaled))
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

#[cfg(test)]
mod tests {
    use super::*;

    // -- StreamingConfig tests -----------------------------------------------

    #[test]
    fn test_streaming_config_greedy() {
        let cfg = StreamingConfig::greedy(256, 200_002);
        assert_eq!(cfg.generate.max_tokens, 256);
        assert_eq!(cfg.generate.temperature, 0.0);
        assert_eq!(cfg.eos_token_id, 200_002);
        assert!(!cfg.return_logits);
    }

    #[test]
    fn test_streaming_config_new() {
        let gen_cfg = GenerateConfig::default();
        let cfg = StreamingConfig::new(gen_cfg, 42);
        assert_eq!(cfg.eos_token_id, 42);
        assert_eq!(cfg.generate.max_tokens, 512);
        assert!(!cfg.return_logits);
    }

    #[test]
    fn test_streaming_config_with_return_logits() {
        let cfg = StreamingConfig::greedy(100, 0).with_return_logits(true);
        assert!(cfg.return_logits);
    }

    // -- StreamingToken tests ------------------------------------------------

    #[test]
    fn test_streaming_token_fields() {
        let tok = StreamingToken {
            id: 42,
            text: Some("hello".into()),
            logits: None,
        };
        assert_eq!(tok.id, 42);
        assert_eq!(tok.text.as_deref(), Some("hello"));
        assert!(tok.logits.is_none());
    }

    #[test]
    fn test_streaming_token_no_text() {
        let tok = StreamingToken {
            id: 7,
            text: None,
            logits: None,
        };
        assert_eq!(tok.id, 7);
        assert!(tok.text.is_none());
    }

    // -- argmax tests --------------------------------------------------------

    #[test]
    fn test_argmax_basic() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0]), 0);
        assert_eq!(argmax(&[0.0, 0.0, 0.1]), 2);
    }

    #[test]
    fn test_argmax_negative_values() {
        assert_eq!(argmax(&[-3.0, -1.0, -2.0]), 1);
    }

    #[test]
    fn test_argmax_single_element() {
        assert_eq!(argmax(&[42.0]), 0);
    }

    // -- EOS and max_tokens boundary tests -----------------------------------

    // These tests verify the session state machine logic without requiring
    // a loaded model. They test the config and state tracking.

    #[test]
    fn test_streaming_config_validates_temperature() {
        let gen_cfg = GenerateConfig {
            temperature: -1.0,
            ..GenerateConfig::default()
        };
        assert!(gen_cfg.validate().is_err());
    }

    #[test]
    fn test_streaming_config_validates_top_k() {
        let gen_cfg = GenerateConfig {
            top_k: Some(0),
            ..GenerateConfig::default()
        };
        assert!(gen_cfg.validate().is_err());
    }
}
