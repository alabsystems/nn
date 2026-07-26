// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Spatial Pyramid Pooling — Fast (SPPF).
//!
//! YOLOv5/v8 SPPF: three sequential max-pools (kernel=5, stride=1, padding=2)
//! applied in series, concatenated with the original input along the channel
//! dimension, and projected back. Captures multi-scale spatial context with
//! minimal computational cost.
//!
//! ```text
//! input
//!   ├─ Conv1x1 (reduce channels)
//!   │    ├─ y1 (pass-through)
//!   │    ├─ y2 = MaxPool5(y1)
//!   │    ├─ y3 = MaxPool5(y2)
//!   │    └─ y4 = MaxPool5(y3)
//!   │    └─ Cat([y1, y2, y3, y4], dim=1)
//!   └─ Conv1x1 (restore channels)
//! ```

use crate::dyn_tensor::DynTensor;
use crate::error::Result;
use crate::layers::{Activation, Module};
use crate::var_builder::VarBuilder;

use super::ConvBnAct;

/// Spatial Pyramid Pooling — Fast (SPPF).
///
/// Input/output shape: `[B, C, H, W]` (channels unchanged).
///
/// # Weight names
///
/// Expects VarBuilder scoped to the SPPF module:
/// - `"cv1.conv.weight"`, `"cv1.bn.*"` — input 1×1 conv
/// - `"cv2.conv.weight"`, `"cv2.bn.*"` — output 1×1 conv
#[derive(Clone, Debug)]
pub struct Sppf {
    cv1: ConvBnAct,
    cv2: ConvBnAct,
    pool_kernel: usize,
}

impl Sppf {
    /// Create from pre-loaded components.
    ///
    /// - `cv1`: input 1×1 convolution (reduces channels by 2)
    /// - `cv2`: output 1×1 convolution (restores channels from 4× hidden)
    /// - `pool_kernel`: max pool kernel size (default 5)
    pub fn new(cv1: ConvBnAct, cv2: ConvBnAct, pool_kernel: usize) -> Self {
        Self {
            cv1,
            cv2,
            pool_kernel,
        }
    }

    /// Load from a VarBuilder.
    ///
    /// - `channels`: input/output channel count
    /// - `pool_kernel`: max pool kernel size (typically 5)
    pub fn load(vb: impl AsRef<VarBuilder>, channels: usize, pool_kernel: usize) -> Result<Self> {
        let vb = vb.as_ref();
        let hidden = channels / 2;
        let cv1 = ConvBnAct::load(vb.pp("cv1"), channels, hidden, 1, 1, Some(Activation::Silu))?;
        // Output conv takes 4 * hidden channels (concat of 4 branches) -> channels
        let cv2 = ConvBnAct::load(
            vb.pp("cv2"),
            hidden * 4,
            channels,
            1,
            1,
            Some(Activation::Silu),
        )?;
        Ok(Self {
            cv1,
            cv2,
            pool_kernel,
        })
    }
}

impl Module for Sppf {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let x = self.cv1.forward(x)?;
        let pad = self.pool_kernel / 2;
        let y1 = x.max_pool2d(self.pool_kernel, 1, pad)?;
        let y2 = y1.max_pool2d(self.pool_kernel, 1, pad)?;
        let y3 = y2.max_pool2d(self.pool_kernel, 1, pad)?;
        let cat = DynTensor::cat(&[&x, &y1, &y2, &y3], 1)?;
        self.cv2.forward(&cat)
    }
}

#[cfg(test)]
#[path = "sppf_tests.rs"]
mod tests;
