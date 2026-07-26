// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv1d with weight normalization.
//!
//! Extracted from `conv.rs` for 500-line compliance (#1280 Direction 3).

use super::{Conv1d, Conv1dConfig};
use crate::dyn_tensor::DynTensor;
use crate::error::{Result, TensorError};
use crate::layers::Module;
use crate::var_builder::VarBuilder;

/// Conv1d with weight normalization applied at construction time.
///
/// Weight normalization decomposes the weight as `w = g * v / ||v||` where:
/// - `v` is the raw weight tensor (the `weight_v` parameter)
/// - `g` is a per-output-channel gain scalar (the `weight_g` parameter)
///
/// The normalized weight is computed once at construction, not per-forward.
/// Used in Kokoro TTS ISTFTNet decoder.
#[derive(Debug, Clone)]
pub struct WeightNormConv1d {
    inner: Conv1d,
}

impl WeightNormConv1d {
    /// Create from weight_v `[out, in/groups, k]`, weight_g `[out, 1, 1]`,
    /// and optional bias `[out]`.
    pub fn new(
        weight_v: DynTensor,
        weight_g: DynTensor,
        bias: Option<DynTensor>,
        config: Conv1dConfig,
    ) -> Result<Self> {
        let normalized = Self::normalize_weight(&weight_v, &weight_g)?;
        Ok(Self {
            inner: Conv1d::new(normalized, bias, config)?,
        })
    }

    /// Load from a [`VarBuilder`] using PyTorch weight normalization names.
    ///
    /// Loads `weight_v` `[out, in/groups, k]`, `weight_g` `[out, 1, 1]`,
    /// and optional `bias` `[out]`.
    ///
    /// - `in_channels`: Number of input channels.
    /// - `out_channels`: Number of output channels.
    /// - `kernel_size`: Convolution kernel size.
    /// - `config`: Conv1d configuration (stride, padding, dilation, groups).
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        config: Conv1dConfig,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let groups = config.groups;
        if groups == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "groups",
                value: 0,
                reason: "must be > 0",
            });
        }
        if !in_channels.is_multiple_of(groups) {
            return Err(TensorError::ValueOutOfRange {
                description: "WeightNormConv1d: in_channels not divisible by groups",
            });
        }
        let weight_v = vb.get(
            &[out_channels, in_channels / groups, kernel_size],
            "weight_v",
        )?;
        let weight_g = vb.get(&[out_channels, 1, 1], "weight_g")?;
        let bias = if vb.contains_tensor("bias") {
            Some(vb.get(&[out_channels], "bias")?)
        } else {
            None
        };
        Self::new(weight_v, weight_g, bias, config)
    }

    fn normalize_weight(v: &DynTensor, g: &DynTensor) -> Result<DynTensor> {
        // v: [out, in_ch, k], g: [out, 1, 1]
        // ||v|| per output channel: reduce over dims 1,2, keepdim
        let v_sq = v.sqr()?;
        let v_norm_sq = v_sq.sum_keepdim(2)?.sum_keepdim(1)?; // [out, 1, 1]
        let v_norm = v_norm_sq.sqrt()?;
        // w = g * v / ||v||
        let g_over_norm = g.broadcast_div(&v_norm)?;
        v.broadcast_mul(&g_over_norm)
    }
}

impl Module for WeightNormConv1d {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        self.inner.forward(x)
    }
}
