// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Mobile Inverted Bottleneck Convolution (MBConv) — EfficientNet building block.
//!
//! Sandler et al., 2018 — "MobileNetV2: Inverted Residuals and Linear Bottlenecks".
//! Tan & Le, 2019 — "EfficientNet: Rethinking Model Scaling for CNNs".
//!
//! Architecture: `expand(1×1) → depthwise(k×k) → SE → project(1×1) + residual`
//! Expansion ratio controls the hidden dimension; SE ratio controls the attention bottleneck.

use crate::dyn_tensor::DynTensor;
use crate::layers::{Activation, BatchNorm, Conv2d, Conv2dConfig, Module, SqueezeExcitation};
use crate::var_builder::VarBuilder;
use crate::{Result, TensorError};

/// MBConv configuration.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct MBConvConfig {
    /// Expansion ratio: hidden_dim = in_channels * expand_ratio.
    pub expand_ratio: usize,
    /// Kernel size for the depthwise convolution (typically 3 or 5).
    pub kernel_size: usize,
    /// Stride for the depthwise convolution.
    pub stride: usize,
    /// SE reduction ratio: se_dim = max(1, in_channels / se_ratio).
    pub se_ratio: usize,
}

impl MBConvConfig {
    /// Create config with expansion ratio and kernel size.
    ///
    /// Stride and SE ratio use defaults (1 and 4 respectively).
    #[must_use]
    pub fn new(expand_ratio: usize, kernel_size: usize) -> Self {
        Self {
            expand_ratio,
            kernel_size,
            ..Default::default()
        }
    }

    /// Set stride for the depthwise convolution.
    #[must_use]
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Set SE reduction ratio.
    #[must_use]
    pub fn with_se_ratio(mut self, se_ratio: usize) -> Self {
        self.se_ratio = se_ratio;
        self
    }
}

impl Default for MBConvConfig {
    fn default() -> Self {
        Self {
            expand_ratio: 1,
            kernel_size: 3,
            stride: 1,
            se_ratio: 4,
        }
    }
}

/// Mobile Inverted Bottleneck Convolution (MBConv) block.
///
/// Given input `[B, C_in, H, W]`:
/// 1. **Expand:** `Conv2d(C_in, C_hidden, 1×1) + BN + SiLU` (skipped if expand_ratio=1)
/// 2. **Depthwise:** `Conv2d(C_hidden, C_hidden, k×k, groups=C_hidden) + BN + SiLU`
/// 3. **SE:** Squeeze-and-Excitation attention on expanded channels
/// 4. **Project:** `Conv2d(C_hidden, C_out, 1×1) + BN` (no activation — linear bottleneck)
/// 5. **Residual:** Add input if `stride=1` and `C_in == C_out`
#[derive(Debug, Clone)]
pub struct MBConv {
    expand: Option<(Conv2d, BatchNorm)>,
    depthwise: Conv2d,
    dw_bn: BatchNorm,
    se: SqueezeExcitation,
    project: Conv2d,
    proj_bn: BatchNorm,
    use_residual: bool,
}

impl MBConv {
    /// Create a new MBConv block.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        expand: Option<(Conv2d, BatchNorm)>,
        depthwise: Conv2d,
        dw_bn: BatchNorm,
        se: SqueezeExcitation,
        project: Conv2d,
        proj_bn: BatchNorm,
        use_residual: bool,
    ) -> Self {
        Self {
            expand,
            depthwise,
            dw_bn,
            se,
            project,
            proj_bn,
            use_residual,
        }
    }

    /// Load from VarBuilder with EfficientNet-style weight names.
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        in_channels: usize,
        out_channels: usize,
        config: MBConvConfig,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        if in_channels == 0 || out_channels == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "MBConv: channels must be > 0",
            });
        }
        if config.kernel_size == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "kernel_size",
                value: 0,
                reason: "must be > 0",
            });
        }
        if config.stride == 0 {
            return Err(TensorError::ConvParameterInvalid {
                param: "stride",
                value: 0,
                reason: "must be > 0",
            });
        }
        if config.expand_ratio == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "MBConv: expand_ratio must be > 0",
            });
        }
        if config.se_ratio == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "MBConv: se_ratio must be > 0",
            });
        }
        let hidden = in_channels * config.expand_ratio;
        // Expand phase (skip if expand_ratio == 1).
        let expand = if config.expand_ratio > 1 {
            let conv = Conv2d::load(
                vb.pp("expand_conv"),
                in_channels,
                hidden,
                1,
                Conv2dConfig::default(),
            )?;
            let bn = BatchNorm::load(
                vb.pp("expand_bn"),
                hidden,
                crate::layers::BatchNormConfig::default(),
            )?;
            Some((conv, bn))
        } else {
            None
        };
        // Depthwise conv.
        let dw_padding = (config.kernel_size - 1) / 2;
        let dw_config = Conv2dConfig {
            padding: dw_padding,
            stride: config.stride,
            groups: hidden,
            ..Conv2dConfig::default()
        };
        let depthwise = Conv2d::load(
            vb.pp("depthwise_conv"),
            hidden,
            hidden,
            config.kernel_size,
            dw_config,
        )?;
        let dw_bn = BatchNorm::load(
            vb.pp("depthwise_bn"),
            hidden,
            crate::layers::BatchNormConfig::default(),
        )?;
        // SE block.
        let se_dim = (in_channels / config.se_ratio).max(1);
        let se = SqueezeExcitation::load(vb.pp("se"), hidden, se_dim)?;
        // Project phase.
        let project = Conv2d::load(
            vb.pp("project_conv"),
            hidden,
            out_channels,
            1,
            Conv2dConfig::default(),
        )?;
        let proj_bn = BatchNorm::load(
            vb.pp("project_bn"),
            out_channels,
            crate::layers::BatchNormConfig::default(),
        )?;
        let use_residual = config.stride == 1 && in_channels == out_channels;
        Ok(Self::new(
            expand,
            depthwise,
            dw_bn,
            se,
            project,
            proj_bn,
            use_residual,
        ))
    }

    /// Whether this block uses a residual connection.
    #[must_use]
    pub fn use_residual(&self) -> bool {
        self.use_residual
    }
}

impl Module for MBConv {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        if x.rank() != 4 {
            return Err(TensorError::RankMismatch {
                expected: 4,
                actual: x.rank(),
            });
        }
        // Expand.
        let mut y = if let Some((ref conv, ref bn)) = self.expand {
            let y = conv.forward(x)?;
            let y = bn.forward(&y)?;
            Activation::Silu.forward(&y)?
        } else {
            x.clone()
        };
        // Depthwise.
        y = self.depthwise.forward(&y)?;
        y = self.dw_bn.forward(&y)?;
        y = Activation::Silu.forward(&y)?;
        // SE.
        y = self.se.forward(&y)?;
        // Project (linear bottleneck — no activation).
        y = self.project.forward(&y)?;
        y = self.proj_bn.forward(&y)?;
        // Residual.
        if self.use_residual {
            y = y.add(x)?;
        }
        Ok(y)
    }
}

#[cfg(test)]
#[path = "mbconv_tests.rs"]
mod tests;
