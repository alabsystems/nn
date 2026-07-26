// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GLM-4/5 model configuration.

use crate::Glm5Error;
use nn_core::Result;

/// GLM-4/5 model configuration.
///
/// Maps to HuggingFace `config.json` fields. Defaults match GLM-4-9B.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Glm5Config {
    pub hidden_size: usize,
    pub ffn_hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    /// Number of KV head groups for multi-query attention.
    /// HF field: `multi_query_group_num`.
    pub multi_query_group_num: usize,
    pub padded_vocab_size: usize,
    /// Per-head channel dimension. HF field: `kv_channels`.
    pub kv_channels: usize,
    pub layernorm_epsilon: f64,
    pub seq_length: usize,
    /// Whether to use RMSNorm (true) or LayerNorm (false).
    /// Note: GLM-4/5 always uses RMSNorm; this field exists for HF config
    /// compatibility but is not checked — the model unconditionally uses RmsNorm.
    pub rmsnorm: bool,
    /// Whether QKV projection includes bias. HF field: `add_qkv_bias`.
    pub add_qkv_bias: bool,
    /// Whether other linear layers have bias. HF field: `add_bias_linear`.
    pub add_bias_linear: bool,
    /// RoPE base frequency (derived from `rope_ratio` if present).
    pub rope_theta: f64,
}

impl Default for Glm5Config {
    /// GLM-4-9B defaults from HuggingFace `config.json`.
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            ffn_hidden_size: 13696,
            num_layers: 40,
            num_attention_heads: 32,
            multi_query_group_num: 2,
            padded_vocab_size: 151552,
            kv_channels: 128,
            layernorm_epsilon: 1.5625e-5,
            seq_length: 8192,
            rmsnorm: true,
            add_qkv_bias: true,
            add_bias_linear: false,
            rope_theta: 10_000.0,
        }
    }
}

impl Glm5Config {
    /// Create a new GLM-4/5 configuration.
    ///
    /// Required because `#[non_exhaustive]` prevents struct literal construction
    /// outside this crate. All fields are specified explicitly.
    ///
    /// Call [`validate()`](Self::validate) to check invariants before use.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_size: usize,
        ffn_hidden_size: usize,
        num_layers: usize,
        num_attention_heads: usize,
        multi_query_group_num: usize,
        padded_vocab_size: usize,
        kv_channels: usize,
        layernorm_epsilon: f64,
        seq_length: usize,
        rmsnorm: bool,
        add_qkv_bias: bool,
        add_bias_linear: bool,
        rope_theta: f64,
    ) -> Self {
        Self {
            hidden_size,
            ffn_hidden_size,
            num_layers,
            num_attention_heads,
            multi_query_group_num,
            padded_vocab_size,
            kv_channels,
            layernorm_epsilon,
            seq_length,
            rmsnorm,
            add_qkv_bias,
            add_bias_linear,
            rope_theta,
        }
    }

    /// GLM-4-9B-chat configuration matching `THUDM/glm-4-9b-chat`.
    ///
    /// Verified against the HuggingFace `config.json` (2026-03-28).
    /// Key differences from the base GLM-4-9B ([`Default::default()`]):
    /// - `layernorm_epsilon`: 1.5625e-7 (base uses 1.5625e-5)
    /// - `rope_ratio`: 500 -> `rope_theta`: 5_000_000.0 (base uses 10_000.0)
    /// - `seq_length`: 131_072 (base uses 8_192)
    #[must_use]
    pub fn glm4_9b_chat() -> Self {
        Self {
            hidden_size: 4096,
            ffn_hidden_size: 13696,
            num_layers: 40,
            num_attention_heads: 32,
            multi_query_group_num: 2,
            padded_vocab_size: 151552,
            kv_channels: 128,
            layernorm_epsilon: 1.5625e-7,
            seq_length: 131_072,
            rmsnorm: true,
            add_qkv_bias: true,
            add_bias_linear: false,
            rope_theta: 5_000_000.0,
        }
    }

    /// Head dimension (same as `kv_channels`).
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.kv_channels
    }

    /// Number of GQA groups (heads / multi_query_group_num).
    ///
    /// Returns `Err` if `multi_query_group_num` is zero or does not divide
    /// `num_attention_heads`, matching the guard pattern in [`validate()`].
    pub fn num_kv_groups(&self) -> Result<usize> {
        if self.multi_query_group_num == 0 {
            return Err(Glm5Error::InvalidConfig {
                reason: "multi_query_group_num must be > 0".into(),
            }
            .into());
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.multi_query_group_num)
        {
            return Err(Glm5Error::InvalidConfig {
                reason: format!(
                    "num_attention_heads ({}) not divisible by multi_query_group_num ({})",
                    self.num_attention_heads, self.multi_query_group_num
                ),
            }
            .into());
        }
        Ok(self.num_attention_heads / self.multi_query_group_num)
    }

    /// Validate configuration invariants.
    pub fn validate(&self) -> Result<()> {
        if self.num_attention_heads == 0 || self.multi_query_group_num == 0 {
            return Err(Glm5Error::InvalidConfig {
                reason: "attention heads and multi_query_group_num must be > 0".into(),
            }
            .into());
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.multi_query_group_num)
        {
            return Err(Glm5Error::InvalidConfig {
                reason: format!(
                    "num_attention_heads ({}) must be divisible by multi_query_group_num ({})",
                    self.num_attention_heads, self.multi_query_group_num
                ),
            }
            .into());
        }
        if self.hidden_size == 0 || self.ffn_hidden_size == 0 {
            return Err(Glm5Error::InvalidConfig {
                reason: "hidden_size and ffn_hidden_size must be > 0".into(),
            }
            .into());
        }
        if self.kv_channels == 0 || !self.kv_channels.is_multiple_of(4) {
            return Err(Glm5Error::InvalidConfig {
                reason: format!(
                    "kv_channels must be a positive multiple of 4 (HalfRotaryEmbedding requirement), got {}",
                    self.kv_channels
                ),
            }
            .into());
        }
        if self.seq_length == 0 {
            return Err(Glm5Error::InvalidConfig {
                reason: "seq_length must be > 0".into(),
            }
            .into());
        }
        if self.padded_vocab_size == 0 {
            return Err(Glm5Error::InvalidConfig {
                reason: "padded_vocab_size must be > 0".into(),
            }
            .into());
        }
        if self.num_layers == 0 {
            return Err(Glm5Error::InvalidConfig {
                reason: "num_layers must be > 0".into(),
            }
            .into());
        }
        if !self.layernorm_epsilon.is_finite() || self.layernorm_epsilon <= 0.0 {
            return Err(Glm5Error::InvalidConfig {
                reason: format!(
                    "layernorm_epsilon must be positive and finite, got {}",
                    self.layernorm_epsilon
                ),
            }
            .into());
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(Glm5Error::InvalidConfig {
                reason: format!(
                    "rope_theta must be positive and finite, got {}",
                    self.rope_theta
                ),
            }
            .into());
        }
        Ok(())
    }
}
