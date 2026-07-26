// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! 1D Squeeze-and-Excitation block — channel attention for temporal features.
//!
//! Adapted from the 2D SE block (se_block.rs) for rank-3 `[B, C, T]` inputs.
//! Used by ECAPA-TDNN speaker verification.
//!
//! Architecture: `mean_keepdim(2) → Linear → ReLU → Linear → Sigmoid → scale`
//!
//! Citation: Hu et al. 2018, "Squeeze-and-Excitation Networks", CVPR.

use crate::dyn_tensor::DynTensor;
use crate::layers::{check_output_finite, Activation, Linear, Module};
use crate::var_builder::VarBuilder;
use crate::{Result, TensorError};

/// 1D Squeeze-and-Excitation block for channel attention on temporal features.
///
/// Given input `[B, C, T]`, computes per-channel attention weights:
/// 1. **Squeeze:** Global average pool over T → `[B, C, 1]` → `[B, C]`
/// 2. **Excitation:** `Linear(C, C/r) → ReLU → Linear(C/r, C) → Sigmoid`
/// 3. **Scale:** Element-wise multiply input by attention weights `[B, C, 1]`
#[derive(Debug, Clone)]
pub struct SqueezeExcitation1d {
    fc1: Linear,
    fc2: Linear,
    channels: usize,
}

impl SqueezeExcitation1d {
    /// Create a new 1D SE block.
    pub fn new(fc1: Linear, fc2: Linear, channels: usize) -> Result<Self> {
        if channels == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "SqueezeExcitation1d: channels must be > 0",
            });
        }
        Ok(Self { fc1, fc2, channels })
    }

    /// Load from VarBuilder with weight names `fc1.*`, `fc2.*`.
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

impl Module for SqueezeExcitation1d {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        if x.rank() != 3 {
            return Err(TensorError::RankMismatch {
                expected: 3,
                actual: x.rank(),
            });
        }
        // Squeeze: [B, C, T] → [B, C, 1] → [B, C]
        let squeezed = x.mean_keepdim(2)?;
        let squeezed = squeezed.squeeze(2)?;
        // Excitation: Linear → ReLU → Linear → Sigmoid
        let y = self.fc1.forward(&squeezed)?;
        let y = Activation::Relu.forward(&y)?;
        let y = self.fc2.forward(&y)?;
        let y = Activation::Sigmoid.forward(&y)?;
        // Reshape back to [B, C, 1] for broadcasting
        let scale = y.unsqueeze(2)?;
        // Scale: element-wise multiply
        let output = x.broadcast_mul(&scale)?;
        // Tier 1 finiteness check: sigmoid uses exp, can amplify NaN.
        check_output_finite(&output, "SqueezeExcitation1d")?;
        Ok(output)
    }
}

#[cfg(test)]
#[path = "se_block_1d_tests.rs"]
mod tests;
