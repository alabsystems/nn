// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv2d + BatchNorm + Activation fused building block.
//!
//! Standard pattern in YOLO architectures where nearly every convolution
//! is followed by BatchNorm and SiLU activation. [`ConvBnAct`] bundles
//! these into a single Module for cleaner model construction.

use crate::dyn_tensor::DynTensor;
use crate::error::Result;
use crate::layers::{Activation, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Module};
use crate::var_builder::VarBuilder;

/// Fused Conv2d → BatchNorm → Activation block.
///
/// Standard building block in YOLO, ResNet, and most CNN detection models.
/// When `act` is `None`, the block is Conv2d → BatchNorm only (used for
/// the final projection in some architectures).
///
/// # Weight names
///
/// Loads from VarBuilder with PyTorch-standard names:
/// - `"conv.weight"` — convolution kernel
/// - `"bn.weight"`, `"bn.bias"`, `"bn.running_mean"`, `"bn.running_var"` — batch norm
#[derive(Clone, Debug)]
pub struct ConvBnAct {
    conv: Conv2d,
    bn: BatchNorm,
    act: Option<Activation>,
}

impl ConvBnAct {
    /// Create from pre-loaded components.
    pub fn new(conv: Conv2d, bn: BatchNorm, act: Option<Activation>) -> Self {
        Self { conv, bn, act }
    }

    /// Load from a VarBuilder using PyTorch YOLO naming conventions.
    ///
    /// - `in_c`: input channels
    /// - `out_c`: output channels
    /// - `kernel_size`: convolution kernel size (square)
    /// - `stride`: convolution stride
    /// - `act`: activation function (typically `Some(Activation::Silu)`)
    ///
    /// Padding is auto-computed as `kernel_size / 2` (same-padding convention).
    /// Groups defaults to 1.
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        stride: usize,
        act: Option<Activation>,
    ) -> Result<Self> {
        Self::load_grouped(vb.as_ref(), in_c, out_c, kernel_size, stride, 1, act)
    }

    /// Load with explicit groups parameter (for depthwise convolutions).
    pub fn load_grouped(
        vb: impl AsRef<VarBuilder>,
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        act: Option<Activation>,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let padding = kernel_size / 2;
        let conv_cfg = Conv2dConfig::new(padding, stride, 1).with_groups(groups);

        let conv_vb = vb.pp("conv");
        // Conv2d::load auto-detects bias; YOLO CBN blocks have no conv bias.
        let conv = Conv2d::load(&conv_vb, in_c, out_c, kernel_size, conv_cfg)?;

        let bn_vb = vb.pp("bn");
        let bn_cfg = BatchNormConfig {
            eps: 1e-3, // YOLO default
            ..BatchNormConfig::default()
        };
        let bn = BatchNorm::load(&bn_vb, out_c, bn_cfg)?;

        Ok(Self { conv, bn, act })
    }

    /// Access the underlying Conv2d layer.
    #[must_use]
    pub fn conv(&self) -> &Conv2d {
        &self.conv
    }

    /// Access the underlying BatchNorm layer.
    #[must_use]
    pub fn bn(&self) -> &BatchNorm {
        &self.bn
    }
}

impl Module for ConvBnAct {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let y = self.conv.forward(x)?;
        let y = self.bn.forward(&y)?;
        match self.act {
            Some(act) => act.forward(&y),
            None => Ok(y),
        }
    }
}

#[cfg(test)]
#[path = "conv_bn_tests.rs"]
mod tests;
