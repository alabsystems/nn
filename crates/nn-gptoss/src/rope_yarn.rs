// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! YaRN RoPE wrapper for gpt-oss.
//!
//! Wraps the core [`RotaryEmbedding`] with YaRN scaling (arXiv:2309.00071)
//! configured from [`GptOssConfig`]. For gpt-oss-20b:
//! - factor=32, beta_fast=32, beta_slow=1
//! - Extends context from original_max=4096 to max_position_embeddings=131072
//!
//! This module provides a convenience wrapper that constructs the correct
//! `RotaryEmbedding` from the model config and delegates `apply()` to the
//! core implementation.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::RotaryEmbedding;
use nn_core::{Device, Result};

use crate::config::GptOssConfig;

/// YaRN-aware rotary position embedding for gpt-oss.
///
/// Constructs from [`GptOssConfig`], using `RotaryEmbedding::new_yarn()` when
/// the config has `rope_scaling`, falling back to standard RoPE otherwise.
pub(crate) struct YarnRotaryEmbedding {
    inner: RotaryEmbedding,
}

impl YarnRotaryEmbedding {
    /// Create from gpt-oss config.
    ///
    /// Uses YaRN scaling when `cfg.rope_scaling` is `Some`, falls back to
    /// standard RoPE when `None`.
    pub(crate) fn new(cfg: &GptOssConfig, device: &Device) -> Result<Self> {
        let inner = match &cfg.rope_scaling {
            Some(yarn) => RotaryEmbedding::new_yarn(
                cfg.head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
                yarn,
                device,
            )?,
            None => RotaryEmbedding::new(
                cfg.head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
                device,
            )?,
        };
        Ok(Self { inner })
    }

    /// Apply YaRN RoPE to Q and K tensors using half-split convention.
    ///
    /// Delegates to `RotaryEmbedding::apply_pair_half_split()` which matches
    /// HuggingFace's `rotate_half` convention used by gpt-oss.
    ///
    /// # Arguments
    /// - `q`: Query tensor `[batch, heads, seq, head_dim]`
    /// - `k`: Key tensor `[batch, kv_heads, seq, head_dim]`
    /// - `positions`: Position indices for each token in the sequence
    pub(crate) fn apply(
        &self,
        q: &DynTensor,
        k: &DynTensor,
        positions: &[usize],
    ) -> Result<(DynTensor, DynTensor)> {
        self.inner.apply_pair_half_split(q, k, positions)
    }

    /// Reference to the underlying `RotaryEmbedding`.
    #[must_use]
    pub(crate) fn inner(&self) -> &RotaryEmbedding {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yarn_construction() {
        let cfg = GptOssConfig::gptoss_20b();
        let rope = YarnRotaryEmbedding::new(&cfg, &Device::Cpu);
        assert!(rope.is_ok(), "YaRN RoPE construction should succeed");
    }

    #[test]
    fn test_yarn_apply_shape() -> Result<()> {
        let cfg = GptOssConfig::gptoss_20b();
        let rope = YarnRotaryEmbedding::new(&cfg, &Device::Cpu)?;

        let batch = 1;
        let seq = 4;
        let heads = cfg.num_attention_heads; // 64
        let kv_heads = cfg.num_key_value_heads; // 8
        let hd = cfg.head_dim; // 64

        let q = DynTensor::ones(&[batch, heads, seq, hd], nn_core::DType::F32, &Device::Cpu)?;
        let k = DynTensor::ones(
            &[batch, kv_heads, seq, hd],
            nn_core::DType::F32,
            &Device::Cpu,
        )?;
        let positions: Vec<usize> = (0..seq).collect();

        let (q_out, k_out) = rope.apply(&q, &k, &positions)?;
        assert_eq!(q_out.dims(), &[batch, heads, seq, hd]);
        assert_eq!(k_out.dims(), &[batch, kv_heads, seq, hd]);
        Ok(())
    }

    #[test]
    fn test_yarn_no_scaling_fallback() {
        let cfg = GptOssConfig::gptoss_20b();
        // Create config without YaRN scaling
        let cfg_no_yarn = GptOssConfig::new(
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.vocab_size,
            cfg.rms_norm_eps,
            cfg.rope_theta,
            cfg.max_position_embeddings,
            cfg.tie_word_embeddings,
            None, // No YaRN scaling
            cfg.attention_bias,
            cfg.num_local_experts,
            cfg.experts_per_token,
            cfg.swiglu_limit,
            cfg.layer_types.clone(),
            cfg.sliding_window,
            cfg.eos_token_id,
        );
        let rope = YarnRotaryEmbedding::new(&cfg_no_yarn, &Device::Cpu);
        assert!(rope.is_ok(), "Fallback to standard RoPE should succeed");
    }

    #[test]
    fn test_yarn_position_offset() -> Result<()> {
        let cfg = GptOssConfig::gptoss_20b();
        let rope = YarnRotaryEmbedding::new(&cfg, &Device::Cpu)?;

        let hd = cfg.head_dim;
        let q = DynTensor::ones(&[1, 1, 2, hd], nn_core::DType::F32, &Device::Cpu)?;
        let k = DynTensor::ones(&[1, 1, 2, hd], nn_core::DType::F32, &Device::Cpu)?;

        // Positions [0, 1] vs [100, 101] should produce different embeddings
        let (q0, _) = rope.apply(&q, &k, &[0, 1])?;
        let (q1, _) = rope.apply(&q, &k, &[100, 101])?;

        let diff = q0.sub(&q1)?.abs()?;
        let max_diff = diff.max_all()?.to_scalar::<f32>()?;
        assert!(
            max_diff > 1e-6,
            "Different positions should produce different RoPE embeddings"
        );
        Ok(())
    }
}
