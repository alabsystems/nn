// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Qwen3 model configuration.
//!
//! Extracted from `lib.rs` (#1667) to keep files under 400 lines.

use nn_core::layers::YarnScaling;
use nn_core::Result;

use crate::Qwen3Error;

/// Qwen3 model configuration.
///
/// Constants: `head_dim = 128`, `vocab_size = 151_936`, `rms_norm_eps = 1e-6`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    /// Optional YaRN scaling for extended context (e.g. Qwen3 131K tokens).
    pub rope_scaling: Option<YarnScaling>,
}

impl Qwen3Config {
    /// Create a new Qwen3 configuration.
    ///
    /// Required because `#[non_exhaustive]` prevents struct literal construction
    /// outside this crate. All fields are specified explicitly.
    ///
    /// Use [`with_vocab_size()`](Self::with_vocab_size) and
    /// [`with_num_hidden_layers()`](Self::with_num_hidden_layers) for
    /// builder-style overrides after construction.
    ///
    /// Call [`validate()`](Self::validate) to check invariants before use.
    #[must_use]
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        vocab_size: usize,
        rms_norm_eps: f64,
        rope_theta: f64,
        max_position_embeddings: usize,
        tie_word_embeddings: bool,
        rope_scaling: Option<YarnScaling>,
    ) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            vocab_size,
            rms_norm_eps,
            rope_theta,
            max_position_embeddings,
            tie_word_embeddings,
            rope_scaling,
        }
    }

    /// Head dimension (constant across all Qwen3 variants).
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        128
    }

    /// Number of GQA groups (heads / kv_heads).
    ///
    /// Returns an error if `num_key_value_heads` is zero or does not evenly
    /// divide `num_attention_heads`. Safe to call on unvalidated configs.
    pub fn num_kv_groups(&self) -> Result<usize> {
        if self.num_key_value_heads == 0 {
            return Err(Qwen3Error::InvalidConfig {
                reason: "num_key_value_heads must be > 0 for GQA group count".into(),
            }
            .into());
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(Qwen3Error::InvalidConfig {
                reason: format!(
                    "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                    self.num_attention_heads, self.num_key_value_heads
                ),
            }
            .into());
        }
        Ok(self.num_attention_heads / self.num_key_value_heads)
    }

    /// Validate configuration invariants.
    pub fn validate(&self) -> Result<()> {
        if self.num_attention_heads == 0 || self.num_key_value_heads == 0 {
            return Err(Qwen3Error::InvalidConfig {
                reason: "attention heads must be > 0".into(),
            }
            .into());
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(Qwen3Error::InvalidConfig {
                reason: format!(
                    "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                    self.num_attention_heads, self.num_key_value_heads
                ),
            }
            .into());
        }
        if self.hidden_size == 0 || self.intermediate_size == 0 {
            return Err(Qwen3Error::InvalidConfig {
                reason: "hidden_size and intermediate_size must be > 0".into(),
            }
            .into());
        }
        if self.vocab_size == 0 {
            return Err(Qwen3Error::InvalidConfig {
                reason: "vocab_size must be > 0".into(),
            }
            .into());
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(Qwen3Error::InvalidConfig {
                reason: format!(
                    "rms_norm_eps must be positive and finite, got {}",
                    self.rms_norm_eps
                ),
            }
            .into());
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(Qwen3Error::InvalidConfig {
                reason: format!(
                    "rope_theta must be positive and finite, got {}",
                    self.rope_theta
                ),
            }
            .into());
        }
        if self.max_position_embeddings == 0 {
            return Err(Qwen3Error::InvalidConfig {
                reason: "max_position_embeddings must be > 0".into(),
            }
            .into());
        }
        Ok(())
    }

    /// Set `vocab_size` (builder-style).
    #[must_use]
    pub fn with_vocab_size(mut self, vocab_size: usize) -> Self {
        self.vocab_size = vocab_size;
        self
    }

    /// Set `num_hidden_layers` (builder-style).
    #[must_use]
    pub fn with_num_hidden_layers(mut self, num_hidden_layers: usize) -> Self {
        self.num_hidden_layers = num_hidden_layers;
        self
    }
}
