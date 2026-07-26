// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Res2Net multi-scale feature extraction block.
//!
//! Splits input channels into `scale` groups. Each group (except the first)
//! is processed by a Conv1d+BatchNorm+ReLU and added to the previous group's
//! output, creating a hierarchical residual-like structure.
//!
//! Citation: Gao et al. 2019, "Res2Net: A New Multi-scale Backbone Architecture",
//! IEEE TPAMI.
//!
//! Used by ECAPA-TDNN speaker verification.

use crate::dyn_tensor::DynTensor;
use crate::layers::{BatchNorm, BatchNormConfig, Conv1d, Conv1dConfig, Module};
use crate::var_builder::VarBuilder;
use crate::{Result, TensorError};

/// Res2Net multi-scale feature extraction block.
///
/// Input `[B, C, T]` is split into `scale` chunks of `[B, C/scale, T]`.
/// Each chunk (except the first) passes through Conv1d+BN+ReLU and receives
/// the previous chunk's output as an additive skip connection.
#[derive(Debug, Clone)]
pub struct Res2NetBlock {
    convs: Vec<(Conv1d, BatchNorm)>,
    scale: usize,
    width: usize,
}

impl Res2NetBlock {
    /// Create from pre-built convolution and batch-norm pairs.
    ///
    /// - `convs`: one `(Conv1d, BatchNorm)` pair per scale group (except the first pass-through group)
    /// - `scale`: total number of multi-scale groups
    /// - `width`: channel width of each group (`channels / scale`)
    pub fn new(convs: Vec<(Conv1d, BatchNorm)>, scale: usize, width: usize) -> Result<Self> {
        if scale == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Res2NetBlock: scale must be > 0",
            });
        }
        if width == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Res2NetBlock: width must be > 0",
            });
        }
        if scale > 1 && convs.len() != scale - 1 {
            return Err(TensorError::InvalidShape(format!(
                "Res2NetBlock: expected {} conv pairs for scale={}, got {}",
                scale - 1,
                scale,
                convs.len()
            )));
        }
        Ok(Self {
            convs,
            scale,
            width,
        })
    }

    /// Load from VarBuilder.
    ///
    /// - `channels`: total input/output channels (must be divisible by `scale`)
    /// - `kernel_size`: convolution kernel width
    /// - `dilation`: dilation factor for dilated convolution
    /// - `scale`: number of multi-scale groups (typically 8)
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        channels: usize,
        kernel_size: usize,
        dilation: usize,
        scale: usize,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        if scale == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "Res2NetBlock: scale must be > 0",
            });
        }
        if channels == 0 || !channels.is_multiple_of(scale) {
            return Err(TensorError::ValueOutOfRange {
                description: "Res2NetBlock: channels must be > 0 and divisible by scale",
            });
        }
        let width = channels / scale;
        let padding = (kernel_size - 1) * dilation / 2;
        let config = Conv1dConfig::default()
            .with_dilation(dilation)
            .with_padding(padding);
        let mut convs = Vec::with_capacity(scale - 1);
        for i in 0..scale - 1 {
            let conv =
                crate::layers::conv1d(width, width, kernel_size, config, vb.pp(format!("conv{i}")))?;
            let bn = BatchNorm::load(vb.pp(format!("bn{i}")), width, BatchNormConfig::default())?;
            convs.push((conv, bn));
        }
        Ok(Self {
            convs,
            scale,
            width,
        })
    }

    /// Number of multi-scale groups.
    #[must_use]
    pub fn scale(&self) -> usize {
        self.scale
    }

    /// Width of each sub-group (channels / scale).
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }
}

impl Module for Res2NetBlock {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        if x.rank() != 3 {
            return Err(TensorError::RankMismatch {
                expected: 3,
                actual: x.rank(),
            });
        }
        // Split into `scale` chunks along channel dim.
        let chunks = x.chunk(self.scale, 1)?;
        let mut outputs = vec![chunks[0].clone()];
        let mut prev = chunks[0].clone();
        for (i, (conv, bn)) in self.convs.iter().enumerate() {
            let input = prev.add(&chunks[i + 1])?;
            let out = conv.forward(&input)?;
            let out = bn.forward(&out)?;
            let out = out.relu()?;
            outputs.push(out.clone());
            prev = out;
        }
        DynTensor::cat(&outputs, 1)
    }
}

#[cfg(test)]
#[path = "res2net_tests.rs"]
mod tests;
