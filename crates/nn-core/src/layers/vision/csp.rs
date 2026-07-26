// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-Stage Partial (CSP) bottleneck blocks — C2f module from YOLOv8.
//!
//! C2f splits the input channels, processes one half through a series of
//! bottleneck blocks, and concatenates all intermediate outputs with the
//! other half before a final 1×1 projection.
//!
//! ```text
//! input
//!   ├─ Conv1x1 -> split into [chunk0, chunk1]
//!   │    chunk1 -> Bottleneck[0] -> out0
//!   │              out0 -> Bottleneck[1] -> out1
//!   │              ...
//!   │    Cat([chunk0, chunk1, out0, out1, ...])
//!   └─ Conv1x1 (project concatenated channels -> out_channels)
//! ```

use crate::dyn_tensor::DynTensor;
use crate::error::Result;
use crate::layers::{Activation, Module};
use crate::var_builder::VarBuilder;

use super::ConvBnAct;

/// Single bottleneck block: two 3×3 convolutions with optional shortcut.
///
/// ```text
/// x -> Conv3x3 -> Conv3x3 -> (+x if shortcut) -> out
/// ```
#[derive(Clone, Debug)]
pub struct Bottleneck {
    cv1: ConvBnAct,
    cv2: ConvBnAct,
    shortcut: bool,
}

impl Bottleneck {
    /// Create from pre-loaded components.
    pub fn new(cv1: ConvBnAct, cv2: ConvBnAct, shortcut: bool) -> Self {
        Self { cv1, cv2, shortcut }
    }

    /// Load from a VarBuilder.
    ///
    /// - `channels`: input and output channel count (must be equal for shortcut)
    /// - `shortcut`: whether to add the residual connection
    pub fn load(vb: impl AsRef<VarBuilder>, channels: usize, shortcut: bool) -> Result<Self> {
        let vb = vb.as_ref();
        let cv1 = ConvBnAct::load(
            vb.pp("cv1"),
            channels,
            channels,
            3,
            1,
            Some(Activation::Silu),
        )?;
        let cv2 = ConvBnAct::load(
            vb.pp("cv2"),
            channels,
            channels,
            3,
            1,
            Some(Activation::Silu),
        )?;
        Ok(Self { cv1, cv2, shortcut })
    }
}

impl Module for Bottleneck {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let y = self.cv1.forward(x)?;
        let y = self.cv2.forward(&y)?;
        if self.shortcut {
            y.broadcast_add(x)
        } else {
            Ok(y)
        }
    }
}

/// C2f module — CSP bottleneck with two convolutions (YOLOv8).
///
/// Input: `[B, in_c, H, W]`
/// Output: `[B, out_c, H, W]`
///
/// # Weight names
///
/// Expects VarBuilder scoped to the C2f module:
/// - `"cv1.conv.weight"`, `"cv1.bn.*"` — input 1×1 conv
/// - `"cv2.conv.weight"`, `"cv2.bn.*"` — output 1×1 conv
/// - `"m.0.cv1.*"`, `"m.0.cv2.*"` — bottleneck 0
/// - `"m.1.cv1.*"`, `"m.1.cv2.*"` — bottleneck 1
/// - etc.
#[derive(Clone, Debug)]
pub struct C2f {
    cv1: ConvBnAct,
    cv2: ConvBnAct,
    bottlenecks: Vec<Bottleneck>,
}

impl C2f {
    /// Create from pre-loaded components.
    pub fn new(cv1: ConvBnAct, cv2: ConvBnAct, bottlenecks: Vec<Bottleneck>) -> Self {
        Self {
            cv1,
            cv2,
            bottlenecks,
        }
    }

    /// Load from a VarBuilder.
    ///
    /// - `in_c`: input channels
    /// - `out_c`: output channels
    /// - `n_bottlenecks`: number of bottleneck blocks
    /// - `shortcut`: whether bottlenecks use residual connections
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        in_c: usize,
        out_c: usize,
        n_bottlenecks: usize,
        shortcut: bool,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let hidden = out_c / 2;
        // cv1 projects in_c -> 2 * hidden (will be split into two chunks)
        let cv1 = ConvBnAct::load(vb.pp("cv1"), in_c, 2 * hidden, 1, 1, Some(Activation::Silu))?;
        // cv2 projects (2 + n_bottlenecks) * hidden -> out_c
        let cat_channels = (2 + n_bottlenecks) * hidden;
        let cv2 = ConvBnAct::load(
            vb.pp("cv2"),
            cat_channels,
            out_c,
            1,
            1,
            Some(Activation::Silu),
        )?;

        let mut bottlenecks = Vec::with_capacity(n_bottlenecks);
        for i in 0..n_bottlenecks {
            let b = Bottleneck::load(vb.pp(format!("m.{i}")), hidden, shortcut)?;
            bottlenecks.push(b);
        }

        Ok(Self {
            cv1,
            cv2,
            bottlenecks,
        })
    }
}

impl Module for C2f {
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let x = self.cv1.forward(x)?;
        // Split along channel dim into two equal halves
        let chunks = x.chunk(2, 1)?;
        let mut outputs = vec![chunks[0].clone(), chunks[1].clone()];
        let mut y = chunks[1].clone();
        for bottleneck in &self.bottlenecks {
            y = bottleneck.forward(&y)?;
            outputs.push(y.clone());
        }
        let refs: Vec<&DynTensor> = outputs.iter().collect();
        let cat = DynTensor::cat(&refs, 1)?;
        self.cv2.forward(&cat)
    }
}

#[cfg(test)]
#[path = "csp_tests.rs"]
mod tests;
