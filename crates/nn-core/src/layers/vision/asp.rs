// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Attentive Statistics Pooling for speaker verification.
//!
//! Computes attention-weighted mean and standard deviation over the temporal
//! dimension, producing a fixed-length representation from variable-length input.
//!
//! Output is `[mean, std]` concatenated: `[B, 2*C]`.
//!
//! Citation: Okabe et al. 2018, "Attentive Statistics Pooling for Deep Speaker
//! Embedding", Interspeech.
//!
//! Used by ECAPA-TDNN speaker verification.

use crate::dyn_tensor::DynTensor;
use crate::layers::{check_output_finite, Linear, Module};
use crate::var_builder::VarBuilder;
use crate::{Result, TensorError};

/// Attentive Statistics Pooling.
///
/// Input `[B, C, T]` → output `[B, 2*C]`.
///
/// Learns attention weights over the temporal dimension, then computes
/// weighted mean and standard deviation. The concatenation of mean and std
/// produces a fixed-length speaker embedding regardless of input duration.
#[derive(Debug, Clone)]
pub struct AttentiveStatisticsPooling {
    attention: Linear,
    channels: usize,
}

impl AttentiveStatisticsPooling {
    /// Create from pre-built components.
    pub fn new(attention: Linear, channels: usize) -> Result<Self> {
        if channels == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "AttentiveStatisticsPooling: channels must be > 0",
            });
        }
        Ok(Self {
            attention,
            channels,
        })
    }

    /// Load from VarBuilder.
    ///
    /// - `channels`: number of input channels (C)
    pub fn load(vb: impl AsRef<VarBuilder>, channels: usize) -> Result<Self> {
        let vb = vb.as_ref();
        let attention = Linear::load(vb.pp("attention"), channels, 1)?;
        Self::new(attention, channels)
    }

    /// Number of input channels.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }
}

impl Module for AttentiveStatisticsPooling {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        if x.rank() != 3 {
            return Err(TensorError::RankMismatch {
                expected: 3,
                actual: x.rank(),
            });
        }
        // x: [B, C, T]
        let x_t = x.transpose(1, 2)?; // [B, T, C]
        let attn = self.attention.forward(&x_t)?; // [B, T, 1]
        let attn = attn.softmax(1)?; // softmax over T
        let attn = attn.transpose(1, 2)?; // [B, 1, T]

        // Weighted mean: [B, C, 1]
        let mean = x.broadcast_mul(&attn)?; // [B, C, T]
        let mean = mean.sum_keepdim(2)?; // [B, C, 1]

        // Weighted variance: E[x^2] - E[x]^2
        let x_sq = x.mul(x)?; // [B, C, T]
        let mean_sq = x_sq.broadcast_mul(&attn)?; // [B, C, T]
        let mean_sq = mean_sq.sum_keepdim(2)?; // [B, C, 1]
        let var = mean_sq.sub(&mean.mul(&mean)?)?; // [B, C, 1]

        // Clamp variance to avoid sqrt of negative (numerical stability).
        let eps = DynTensor::full(&[], 1e-7, x.dtype(), &x.device())?;
        let var = var.maximum(&eps)?;
        let std = var.sqrt()?; // [B, C, 1]

        // Concatenate mean and std.
        let mean = mean.squeeze(2)?; // [B, C]
        let std = std.squeeze(2)?; // [B, C]
        let output = DynTensor::cat(&[&mean, &std], 1)?; // [B, 2*C]

        check_output_finite(&output, "AttentiveStatisticsPooling")?;
        Ok(output)
    }
}

#[cfg(test)]
#[path = "asp_tests.rs"]
mod tests;
