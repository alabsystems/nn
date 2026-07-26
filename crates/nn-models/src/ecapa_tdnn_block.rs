// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SE-Res2 block for ECAPA-TDNN speaker verification.
//!
//! Composed block: Conv1d → BN → ReLU → Res2Net → Conv1d → BN → SE1d + skip.
//!
//! Citation: Desplanques et al. 2020, "ECAPA-TDNN: Emphasized Channel Attention,
//! Propagation and Aggregation in TDNN Based Speaker Verification", Interspeech.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{
    BatchNorm, BatchNormConfig, Conv1d, Conv1dConfig, Module, Res2NetBlock, SqueezeExcitation1d,
};
use nn_core::var_builder::VarBuilder;
use nn_core::{Result, TensorError};

/// SE-Res2 block: Conv1d + Res2Net + SE1d + skip connection.
///
/// Pre-conv expands/contracts channels, Res2Net provides multi-scale
/// feature extraction, SE provides channel attention, and a residual
/// skip connection preserves the input signal.
#[derive(Debug, Clone)]
pub struct SERes2Block {
    pre_conv: Conv1d,
    pre_bn: BatchNorm,
    res2net: Res2NetBlock,
    se: SqueezeExcitation1d,
    post_conv: Conv1d,
    post_bn: BatchNorm,
    shortcut: Option<(Conv1d, BatchNorm)>,
}

impl SERes2Block {
    /// Load from VarBuilder.
    ///
    /// - `in_channels`: input channel count
    /// - `out_channels`: output channel count (also used for internal processing)
    /// - `kernel_size`: convolution kernel width
    /// - `dilation`: dilation factor for the Res2Net convolutions
    /// - `scale`: number of multi-scale groups in Res2Net (typically 8)
    /// - `se_reduction`: SE bottleneck reduction factor (typically 128)
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        scale: usize,
        se_reduction: usize,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        if in_channels == 0 {
            return Err(TensorError::ZeroLengthDimension {
                axis: 0,
                operation: "SERes2Block::load (in_channels)",
            });
        }
        if out_channels == 0 {
            return Err(TensorError::ZeroLengthDimension {
                axis: 0,
                operation: "SERes2Block::load (out_channels)",
            });
        }

        // Pre-convolution: in_channels → out_channels, kernel=1.
        let pre_conv = nn_core::layers::conv1d(
            in_channels,
            out_channels,
            1,
            Conv1dConfig::default(),
            vb.pp("pre_conv"),
        )?;
        let pre_bn = BatchNorm::load(vb.pp("pre_bn"), out_channels, BatchNormConfig::default())?;

        // Res2Net multi-scale block.
        let res2net =
            Res2NetBlock::load(vb.pp("res2net"), out_channels, kernel_size, dilation, scale)?;

        // Squeeze-and-Excitation 1D.
        let se = SqueezeExcitation1d::load(vb.pp("se"), out_channels, se_reduction)?;

        // Post-convolution: out_channels → out_channels, kernel=1.
        let post_conv = nn_core::layers::conv1d(
            out_channels,
            out_channels,
            1,
            Conv1dConfig::default(),
            vb.pp("post_conv"),
        )?;
        let post_bn = BatchNorm::load(vb.pp("post_bn"), out_channels, BatchNormConfig::default())?;

        // Shortcut projection if channel counts differ.
        let shortcut = if in_channels != out_channels {
            let sc_conv = nn_core::layers::conv1d(
                in_channels,
                out_channels,
                1,
                Conv1dConfig::default(),
                vb.pp("shortcut_conv"),
            )?;
            let sc_bn = BatchNorm::load(
                vb.pp("shortcut_bn"),
                out_channels,
                BatchNormConfig::default(),
            )?;
            Some((sc_conv, sc_bn))
        } else {
            None
        };

        Ok(Self {
            pre_conv,
            pre_bn,
            res2net,
            se,
            post_conv,
            post_bn,
            shortcut,
        })
    }
}

impl Module for SERes2Block {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        if x.rank() != 3 {
            return Err(TensorError::RankMismatch {
                expected: 3,
                actual: x.rank(),
            });
        }

        // Main path: pre_conv → BN → ReLU → Res2Net → SE → post_conv → BN.
        let out = self.pre_conv.forward(x)?;
        let out = self.pre_bn.forward(&out)?;
        let out = out.relu()?;
        let out = self.res2net.forward(&out)?;
        let out = self.se.forward(&out)?;
        let out = self.post_conv.forward(&out)?;
        let out = self.post_bn.forward(&out)?;

        // Skip connection.
        let skip = match &self.shortcut {
            Some((conv, bn)) => {
                let s = conv.forward(x)?;
                bn.forward(&s)?
            }
            None => x.clone(),
        };

        let output = out.add(&skip)?;
        output.relu()
    }
}

#[cfg(test)]
#[path = "ecapa_tdnn_block_tests.rs"]
mod tests;
