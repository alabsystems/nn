// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Squeeze-and-Excitation (SE) block — channel attention mechanism.
//!
//! Hu et al., 2018 — "Squeeze-and-Excitation Networks".
//! Used by EfficientNet, MobileNetV3, and other modern vision architectures.
//!
//! Architecture: `AdaptiveAvgPool2d(1,1) → Linear → ReLU → Linear → Sigmoid → scale`
//! Learns per-channel attention weights from global spatial information.

use crate::dyn_tensor::DynTensor;
use crate::layers::{check_output_finite, Activation, AdaptiveAvgPool2d, Linear, Module};
use crate::var_builder::VarBuilder;
use crate::{Result, TensorError};

/// Squeeze-and-Excitation (SE) block for channel attention.
///
/// Given input `[B, C, H, W]`, computes per-channel attention weights:
/// 1. **Squeeze:** Global average pool → `[B, C, 1, 1]`
/// 2. **Excitation:** `Linear(C, C/r) → ReLU → Linear(C/r, C) → Sigmoid`
/// 3. **Scale:** Element-wise multiply input by attention weights
///
/// The reduction ratio `r` controls the bottleneck dimension.
#[derive(Debug, Clone)]
pub struct SqueezeExcitation {
    pool: AdaptiveAvgPool2d,
    fc1: Linear,
    fc2: Linear,
    channels: usize,
}

impl SqueezeExcitation {
    /// Create a new SE block.
    ///
    /// - `channels`: number of input channels (C)
    /// - `reduced`: bottleneck dimension (typically `C / reduction_ratio`)
    /// - `fc1`: Linear(C → reduced)
    /// - `fc2`: Linear(reduced → C)
    pub fn new(fc1: Linear, fc2: Linear, channels: usize) -> Result<Self> {
        if channels == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "SqueezeExcitation: channels must be > 0",
            });
        }
        let pool = AdaptiveAvgPool2d::new(1, 1)?;
        Ok(Self {
            pool,
            fc1,
            fc2,
            channels,
        })
    }

    /// Load from VarBuilder with weight names `fc1.weight`, `fc1.bias`, `fc2.weight`, `fc2.bias`.
    pub fn load(vb: impl AsRef<VarBuilder>, channels: usize, reduced: usize) -> Result<Self> {
        let vb = vb.as_ref();
        let fc1 = Linear::load(vb.pp("fc1"), channels, reduced)?;
        let fc2 = Linear::load(vb.pp("fc2"), reduced, channels)?;
        Self::new(fc1, fc2, channels)
    }

    /// Number of input/output channels.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }
}

impl Module for SqueezeExcitation {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        if x.rank() != 4 {
            return Err(TensorError::RankMismatch {
                expected: 4,
                actual: x.rank(),
            });
        }
        // Squeeze: [B, C, H, W] → [B, C, 1, 1]
        let squeezed = self.pool.forward(x)?;
        // Flatten for linear layers: [B, C, 1, 1] → [B, C]
        let flat = squeezed.reshape([squeezed.dim(0)?, self.channels])?;
        // Excitation: Linear → ReLU → Linear → Sigmoid
        let y = self.fc1.forward(&flat)?;
        let y = Activation::Relu.forward(&y)?;
        let y = self.fc2.forward(&y)?;
        let y = Activation::Sigmoid.forward(&y)?;
        // Reshape back to [B, C, 1, 1] for broadcasting
        let scale = y.reshape([y.dim(0)?, self.channels, 1, 1])?;
        // Scale: element-wise multiply
        let output = x.broadcast_mul(&scale)?;
        // Tier 1 finiteness check (#1209): sigmoid uses exp, can amplify NaN.
        check_output_finite(&output, "SqueezeExcitation")?;
        Ok(output)
    }
}

#[cfg(test)]
#[path = "se_block_tests.rs"]
mod tests;
