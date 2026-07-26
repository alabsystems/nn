// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! gpt-oss-20b (Chroma Context-1) model configuration.

use nn_core::layers::YarnScaling;
use nn_core::Result;

use crate::GptOssError;

/// Attention layer type: alternating sliding window and full causal attention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LayerType {
    /// Sliding window attention with limited context (e.g. 128 tokens).
    SlidingAttention,
    /// Full causal attention over the entire sequence.
    FullAttention,
}

/// gpt-oss-20b (Chroma Context-1) model configuration.
///
/// 20B-parameter MoE decoder-only transformer with GQA, alternating
/// sliding/full attention, YaRN RoPE scaling, and SwiGLU experts with
/// clamped gate activation.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct GptOssConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    /// YaRN RoPE scaling configuration.
    pub rope_scaling: Option<YarnScaling>,
    /// Attention bias on Q/K/V/O projections.
    pub attention_bias: bool,
    /// Total number of experts per MoE layer.
    pub num_local_experts: usize,
    /// Number of experts active per token (top-k routing).
    pub experts_per_token: usize,
    /// SwiGLU gate clamping limit: `silu(gate).clamp(-limit, limit)`.
    pub swiglu_limit: f64,
    /// Per-layer attention type pattern (length == `num_hidden_layers`).
    pub layer_types: Vec<LayerType>,
    /// Sliding window size for `SlidingAttention` layers.
    pub sliding_window: usize,
    /// End-of-sequence token ID.
    pub eos_token_id: usize,
}

impl GptOssConfig {
    /// Create a new gpt-oss configuration with all fields specified.
    ///
    /// Required because `#[non_exhaustive]` prevents struct literal construction
    /// outside this crate.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        vocab_size: usize,
        rms_norm_eps: f64,
        rope_theta: f64,
        max_position_embeddings: usize,
        tie_word_embeddings: bool,
        rope_scaling: Option<YarnScaling>,
        attention_bias: bool,
        num_local_experts: usize,
        experts_per_token: usize,
        swiglu_limit: f64,
        layer_types: Vec<LayerType>,
        sliding_window: usize,
        eos_token_id: usize,
    ) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            vocab_size,
            rms_norm_eps,
            rope_theta,
            max_position_embeddings,
            tie_word_embeddings,
            rope_scaling,
            attention_bias,
            num_local_experts,
            experts_per_token,
            swiglu_limit,
            layer_types,
            sliding_window,
            eos_token_id,
        }
    }

    /// Preset configuration for gpt-oss-20b (Chroma Context-1).
    ///
    /// Derived from the actual Context-1 `config.json` and weight shapes:
    /// - 64 Q heads x 64 head_dim = 4096 (Q proj dimension > hidden_size=2880)
    /// - 8 KV heads x 64 head_dim = 512
    /// - GQA groups: 64 / 8 = 8 (cleanly divisible)
    /// - Fused expert weights: `[32, 2880, 5760]` gate_up_proj per layer
    /// - Router has bias: `[32]`
    /// - Per-layer attention sinks: `[64]` (= head_dim)
    /// - `rope_parameters` contains `rope_theta: 150000`
    #[must_use]
    pub fn gptoss_20b() -> Self {
        let layer_types: Vec<LayerType> = (0..24)
            .map(|i| {
                if i % 2 == 0 {
                    LayerType::SlidingAttention
                } else {
                    LayerType::FullAttention
                }
            })
            .collect();

        Self::new(
            2880,      // hidden_size
            2880,      // intermediate_size
            24,        // num_hidden_layers
            64,        // num_attention_heads (actual weights: Q proj [4096, 2880])
            8,         // num_key_value_heads
            64,        // head_dim
            201_088,   // vocab_size
            1e-5,      // rms_norm_eps
            150_000.0, // rope_theta (inside rope_parameters in config.json)
            131_072,   // max_position_embeddings
            false,     // tie_word_embeddings
            Some(YarnScaling::new(
                32.0, // factor
                1.0,  // attention_factor
                32.0, // beta_fast
                1.0,  // beta_slow
                4096, // original_max_position_embeddings
            )),
            true, // attention_bias
            32,   // num_local_experts
            4,    // experts_per_token
            7.0,  // swiglu_limit
            layer_types,
            128,     // sliding_window
            200_002, // eos_token_id
        )
    }

    /// GQA repeat factor: `num_attention_heads / num_key_value_heads`.
    ///
    /// For gpt-oss with 64 Q heads / 8 KV heads, this returns 8 (cleanly
    /// divisible). Standard GQA repeat_kv without narrowing.
    pub fn kv_repeat_factor(&self) -> Result<usize> {
        if self.num_key_value_heads == 0 {
            return Err(GptOssError::InvalidConfig {
                reason: "num_key_value_heads must be > 0".into(),
            }
            .into());
        }
        Ok(self.num_attention_heads / self.num_key_value_heads)
    }

    /// Attention internal dimension: `num_attention_heads * head_dim`.
    ///
    /// For gpt-oss this is 64 * 64 = 4096, which is LARGER than `hidden_size`
    /// (2880). The O projection maps back from 4096 to 2880.
    #[must_use]
    pub fn attn_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// KV projection dimension: `num_key_value_heads * head_dim`.
    #[must_use]
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Validate configuration invariants.
    pub fn validate(&self) -> Result<()> {
        if self.num_attention_heads == 0 || self.num_key_value_heads == 0 {
            return Err(GptOssError::InvalidConfig {
                reason: "attention heads must be > 0".into(),
            }
            .into());
        }
        if self.num_attention_heads < self.num_key_value_heads {
            return Err(GptOssError::InvalidConfig {
                reason: format!(
                    "num_attention_heads ({}) must be >= num_key_value_heads ({})",
                    self.num_attention_heads, self.num_key_value_heads
                ),
            }
            .into());
        }
        // GQA: num_attention_heads must be evenly divisible by num_key_value_heads.
        // For gpt-oss: 64 / 8 = 8 groups.
        if !self.num_attention_heads.is_multiple_of(self.num_key_value_heads) {
            return Err(GptOssError::InvalidConfig {
                reason: format!(
                    "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                    self.num_attention_heads, self.num_key_value_heads
                ),
            }
            .into());
        }
        if self.hidden_size == 0 || self.intermediate_size == 0 {
            return Err(GptOssError::InvalidConfig {
                reason: "hidden_size and intermediate_size must be > 0".into(),
            }
            .into());
        }
        if self.head_dim == 0 {
            return Err(GptOssError::InvalidConfig {
                reason: "head_dim must be > 0".into(),
            }
            .into());
        }
        if self.vocab_size == 0 {
            return Err(GptOssError::InvalidConfig {
                reason: "vocab_size must be > 0".into(),
            }
            .into());
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(GptOssError::InvalidConfig {
                reason: format!(
                    "rms_norm_eps must be positive and finite, got {}",
                    self.rms_norm_eps
                ),
            }
            .into());
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(GptOssError::InvalidConfig {
                reason: format!(
                    "rope_theta must be positive and finite, got {}",
                    self.rope_theta
                ),
            }
            .into());
        }
        if self.max_position_embeddings == 0 {
            return Err(GptOssError::InvalidConfig {
                reason: "max_position_embeddings must be > 0".into(),
            }
            .into());
        }
        if self.num_local_experts == 0 {
            return Err(GptOssError::InvalidConfig {
                reason: "num_local_experts must be > 0".into(),
            }
            .into());
        }
        if self.experts_per_token == 0 || self.experts_per_token > self.num_local_experts {
            return Err(GptOssError::InvalidConfig {
                reason: format!(
                    "experts_per_token ({}) must be in [1, {}]",
                    self.experts_per_token, self.num_local_experts
                ),
            }
            .into());
        }
        if !self.swiglu_limit.is_finite() || self.swiglu_limit <= 0.0 {
            return Err(GptOssError::InvalidConfig {
                reason: format!(
                    "swiglu_limit must be positive and finite, got {}",
                    self.swiglu_limit
                ),
            }
            .into());
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(GptOssError::InvalidConfig {
                reason: format!(
                    "layer_types length ({}) must match num_hidden_layers ({})",
                    self.layer_types.len(),
                    self.num_hidden_layers
                ),
            }
            .into());
        }
        if self.sliding_window == 0 {
            return Err(GptOssError::InvalidConfig {
                reason: "sliding_window must be > 0".into(),
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

    /// Set `num_hidden_layers` and regenerate layer_types (builder-style).
    #[must_use]
    pub fn with_num_hidden_layers(mut self, num_hidden_layers: usize) -> Self {
        self.num_hidden_layers = num_hidden_layers;
        self.layer_types = (0..num_hidden_layers)
            .map(|i| {
                if i % 2 == 0 {
                    LayerType::SlidingAttention
                } else {
                    LayerType::FullAttention
                }
            })
            .collect();
        self
    }

    /// Set `num_local_experts` (builder-style).
    #[must_use]
    pub fn with_num_local_experts(mut self, n: usize) -> Self {
        self.num_local_experts = n;
        self
    }

    /// Set `experts_per_token` (builder-style).
    #[must_use]
    pub fn with_experts_per_token(mut self, k: usize) -> Self {
        self.experts_per_token = k;
        self
    }
}

#[cfg(kani)]
#[path = "kani_config_proofs.rs"]
mod kani_config_proofs;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gptoss_20b_preset_validates() {
        let cfg = GptOssConfig::gptoss_20b();
        cfg.validate().expect("20b preset should validate");
        assert_eq!(cfg.hidden_size, 2880);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.num_attention_heads, 64);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.vocab_size, 201_088);
        assert_eq!(cfg.num_local_experts, 32);
        assert_eq!(cfg.experts_per_token, 4);
        assert_eq!(cfg.layer_types.len(), 24);
        assert_eq!(cfg.layer_types[0], LayerType::SlidingAttention);
        assert_eq!(cfg.layer_types[1], LayerType::FullAttention);
        assert_eq!(cfg.sliding_window, 128);
        assert_eq!(cfg.eos_token_id, 200_002);
        assert!(cfg.attention_bias);
        assert!(!cfg.tie_word_embeddings);
    }

    #[test]
    fn test_kv_repeat_factor_clean_gqa() {
        let cfg = GptOssConfig::gptoss_20b();
        // 64 Q heads / 8 KV heads = 8 (cleanly divisible, no narrowing needed).
        let factor = cfg.kv_repeat_factor().expect("should succeed");
        assert_eq!(factor, 8);
    }

    #[test]
    fn test_attn_dim_larger_than_hidden() {
        let cfg = GptOssConfig::gptoss_20b();
        // attn_dim = 64 * 64 = 4096 > hidden_size = 2880
        assert_eq!(cfg.attn_dim(), 4096);
        assert_eq!(cfg.kv_dim(), 512);
        assert!(cfg.attn_dim() > cfg.hidden_size);
    }

    #[test]
    fn test_gqa_groups_evenly_divisible() {
        let cfg = GptOssConfig::gptoss_20b();
        assert_eq!(cfg.num_attention_heads % cfg.num_key_value_heads, 0);
        assert_eq!(cfg.num_attention_heads / cfg.num_key_value_heads, 8);
    }

    #[test]
    fn test_validation_zero_heads() {
        let mut cfg = GptOssConfig::gptoss_20b();
        cfg.num_attention_heads = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validation_zero_experts() {
        let mut cfg = GptOssConfig::gptoss_20b();
        cfg.num_local_experts = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_builder_setters() {
        let cfg = GptOssConfig::gptoss_20b()
            .with_vocab_size(100)
            .with_num_hidden_layers(4)
            .with_num_local_experts(8)
            .with_experts_per_token(2);
        assert_eq!(cfg.vocab_size, 100);
        assert_eq!(cfg.num_hidden_layers, 4);
        assert_eq!(cfg.layer_types.len(), 4);
        assert_eq!(cfg.num_local_experts, 8);
        assert_eq!(cfg.experts_per_token, 2);
    }
}
