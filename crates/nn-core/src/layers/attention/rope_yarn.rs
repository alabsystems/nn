// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! YaRN (Yet Another RoPE extension) scaling for extended context lengths.
//!
//! Implements NTK-aware frequency interpolation per arXiv:2309.00071.
//! Used by Qwen3 for extending context from 40,960 to 131,072 tokens.

use super::RotaryEmbedding;
use crate::dyn_tensor::DynTensor;
use crate::{Device, Result, TensorError};

/// YaRN (Yet Another RoPE extension) scaling configuration.
///
/// Enables extended context lengths by applying NTK-aware frequency
/// interpolation per arXiv:2309.00071. Low-frequency dimensions are
/// linearly interpolated (divided by `factor`), high-frequency dimensions
/// are kept unchanged, and a smooth ramp blends between the two regimes.
///
/// Default values match Qwen3's `rope_scaling` config.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct YarnScaling {
    /// Context extension factor (e.g. 4.0 for 4× context).
    pub factor: f64,
    /// Attention temperature scaling factor (typically 1/ln(factor)).
    pub attention_factor: f64,
    /// High-frequency cutoff wavelength parameter (default: 32.0).
    pub beta_fast: f64,
    /// Low-frequency cutoff wavelength parameter (default: 1.0).
    pub beta_slow: f64,
    /// Original maximum position embeddings before scaling.
    pub original_max_position_embeddings: usize,
}

impl YarnScaling {
    /// Create a YaRN scaling configuration.
    #[must_use]
    pub fn new(
        factor: f64,
        attention_factor: f64,
        beta_fast: f64,
        beta_slow: f64,
        original_max_position_embeddings: usize,
    ) -> Self {
        Self {
            factor,
            attention_factor,
            beta_fast,
            beta_slow,
            original_max_position_embeddings,
        }
    }
}

impl RotaryEmbedding {
    /// Create a RotaryEmbedding with YaRN extended context scaling.
    ///
    /// Applies NTK-aware frequency interpolation per arXiv:2309.00071:
    /// - High-frequency dimensions (short wavelength): no scaling
    /// - Low-frequency dimensions (long wavelength): linear interpolation
    /// - Middle dimensions: smooth blend via linear ramp
    ///
    /// Used by Qwen3 for extending context from 40,960 to 131,072 tokens.
    pub fn new_yarn(
        head_dim: usize,
        max_seq_len: usize,
        base: f64,
        yarn: &YarnScaling,
        device: &Device,
    ) -> Result<Self> {
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(TensorError::ValueOutOfRange {
                description: "RotaryEmbedding: head_dim must be a positive even number",
            });
        }
        if max_seq_len == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "RotaryEmbedding: max_seq_len must be > 0",
            });
        }
        if !base.is_finite() || base <= 0.0 {
            return Err(TensorError::ValueOutOfRange {
                description: "RotaryEmbedding: base must be positive finite",
            });
        }
        if yarn.factor <= 0.0 || !yarn.factor.is_finite() {
            return Err(TensorError::ValueOutOfRange {
                description: "YaRN: factor must be positive finite",
            });
        }

        let half_dim = head_dim / 2;
        let dim = head_dim as f64;

        // Wavelength boundaries for the ramp function.
        // λ_low = 2π × original_ctx / beta_fast
        // λ_high = 2π × original_ctx / beta_slow
        let orig_ctx = yarn.original_max_position_embeddings as f64;
        let low_freq_wavelen = orig_ctx / yarn.beta_fast;
        let high_freq_wavelen = orig_ctx / yarn.beta_slow;
        let wavelen_range = (high_freq_wavelen - low_freq_wavelen).max(1e-12);

        // Compute YaRN-scaled inv_freq.
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| {
                let exponent = (2 * i) as f64 / dim;
                let freq = 1.0 / base.powf(exponent);
                let wavelen = 2.0 * std::f64::consts::PI / freq;

                // Ramp: 0.0 for high-freq (short wavelength), 1.0 for low-freq
                let ramp = ((wavelen - low_freq_wavelen) / wavelen_range).clamp(0.0, 1.0);
                // Blend: high_freq keeps original, low_freq divides by factor
                let scaled_freq = (1.0 - ramp) * freq + ramp * (freq / yarn.factor);
                scaled_freq as f32
            })
            .collect();

        let cache_len =
            max_seq_len
                .checked_mul(half_dim)
                .ok_or(TensorError::DimensionOverflow {
                    dims: vec![max_seq_len, half_dim],
                })?;
        let mut cos_data = Vec::with_capacity(cache_len);
        let mut sin_data = Vec::with_capacity(cache_len);

        // Attention scaling per YaRN (temperature adjustment).
        let attn_scale = if yarn.attention_factor.is_finite() && yarn.attention_factor > 0.0 {
            yarn.attention_factor as f32
        } else {
            1.0
        };

        for pos in 0..max_seq_len {
            for &freq in &inv_freq {
                let angle = (pos as f64 * f64::from(freq)) as f32;
                cos_data.push(angle.cos() * attn_scale);
                sin_data.push(angle.sin() * attn_scale);
            }
        }

        let cos_cache = DynTensor::from_vec(cos_data, &[max_seq_len, half_dim], &Device::Cpu)?
            .to_device(device)?;
        let sin_cache = DynTensor::from_vec(sin_data, &[max_seq_len, half_dim], &Device::Cpu)?
            .to_device(device)?;

        Ok(Self {
            cos_cache,
            sin_cache,
            head_dim,
            max_seq_len,
        })
    }
}
