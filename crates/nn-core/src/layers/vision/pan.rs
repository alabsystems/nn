// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Path Aggregation Network (PAN) neck for multi-scale feature fusion.
//!
//! Standard FPN + PAN architecture used in YOLOv5/v8 detection models.
//! Takes multi-scale backbone features (typically 3 scales: P3/P4/P5) and
//! fuses them top-down then bottom-up to produce detection-ready features.
//!
//! ```text
//! Backbone:  P3(stride=8)    P4(stride=16)    P5(stride=32)
//!                                 │                 │
//! Top-down:                       │        Upsample(P5)
//!                                 ├─── Cat ───┤
//!                                C2f → N4      │
//!                    Upsample(N4)               │
//!                  ├─── Cat ──┤                 │
//!                 C2f → N3                      │
//!                                               │
//! Bottom-up:  Conv(s=2)(N3)                     │
//!                  ├─── Cat → N4' ──┤           │
//!                                C2f → N4'      │
//!                            Conv(s=2)(N4')     │
//!                                  ├─── Cat ───┤
//!                                 C2f → N5'
//!
//! Output:    N3              N4'              N5'
//! ```

use crate::dyn_tensor::DynTensor;
use crate::error::Result;
use crate::layers::{Activation, Module};
use crate::var_builder::VarBuilder;

use super::{C2f, ConvBnAct, Upsample2d, UpsampleMode};

/// PAN neck — Path Aggregation Network for multi-scale feature fusion.
///
/// Takes 3 multi-scale feature maps from the backbone and produces 3 fused
/// feature maps suitable for detection heads.
///
/// Input: 3 tensors `[B, C_i, H_i, W_i]` at decreasing spatial resolution.
/// Output: 3 tensors at the same spatial resolutions with fused features.
///
/// # Weight names
///
/// Expects VarBuilder scoped to the PAN neck:
/// - `"up1.cv1.*"`, `"up1.cv2.*"`, `"up1.m.*"` — top-down C2f (P5+P4 → N4)
/// - `"up2.cv1.*"`, `"up2.cv2.*"`, `"up2.m.*"` — top-down C2f (N4+P3 → N3)
/// - `"down1_conv.*"` — bottom-up stride-2 conv (N3 → downsample)
/// - `"down1.cv1.*"`, `"down1.cv2.*"`, `"down1.m.*"` — bottom-up C2f
/// - `"down2_conv.*"` — bottom-up stride-2 conv (N4' → downsample)
/// - `"down2.cv1.*"`, `"down2.cv2.*"`, `"down2.m.*"` — bottom-up C2f
#[derive(Clone, Debug)]
pub struct PanNeck {
    // Top-down path
    up1_c2f: C2f,
    up2_c2f: C2f,
    upsample: Upsample2d,
    // Bottom-up path
    down1_conv: ConvBnAct,
    down1_c2f: C2f,
    down2_conv: ConvBnAct,
    down2_c2f: C2f,
}

impl PanNeck {
    /// Create from pre-loaded components.
    ///
    /// - `up1_c2f`, `up2_c2f`: top-down fusion C2f blocks
    /// - `upsample`: 2× nearest-neighbor upsampler
    /// - `down1_conv`, `down2_conv`: stride-2 convolutions for bottom-up
    /// - `down1_c2f`, `down2_c2f`: bottom-up fusion C2f blocks
    pub fn new(
        up1_c2f: C2f,
        up2_c2f: C2f,
        upsample: Upsample2d,
        down1_conv: ConvBnAct,
        down1_c2f: C2f,
        down2_conv: ConvBnAct,
        down2_c2f: C2f,
    ) -> Self {
        Self {
            up1_c2f,
            up2_c2f,
            upsample,
            down1_conv,
            down1_c2f,
            down2_conv,
            down2_c2f,
        }
    }

    /// Load a PAN neck from a VarBuilder.
    ///
    /// - `channels`: channel counts for the 3 input scales `[p3_c, p4_c, p5_c]`
    /// - `n_bottlenecks`: number of bottleneck blocks per C2f module
    ///
    /// The top-down path halves the deeper channels at each level; the bottom-up
    /// path restores them. Output channel counts match `channels`.
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        channels: [usize; 3],
        n_bottlenecks: usize,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let [c3, c4, c5] = channels;

        let upsample = Upsample2d::new(2.0, 2.0, UpsampleMode::Nearest)?;

        // Top-down: P5 upsampled + P4 → C2f → N4 (c4 channels)
        let up1_c2f = C2f::load(vb.pp("up1"), c5 + c4, c4, n_bottlenecks, false)?;

        // Top-down: N4 upsampled + P3 → C2f → N3 (c3 channels)
        let up2_c2f = C2f::load(vb.pp("up2"), c4 + c3, c3, n_bottlenecks, false)?;

        // Bottom-up: N3 → stride-2 conv → cat with N4 → C2f → N4'
        let down1_conv =
            ConvBnAct::load(vb.pp("down1_conv"), c3, c3, 3, 2, Some(Activation::Silu))?;
        let down1_c2f = C2f::load(vb.pp("down1"), c3 + c4, c4, n_bottlenecks, false)?;

        // Bottom-up: N4' → stride-2 conv → cat with P5 → C2f → N5'
        let down2_conv =
            ConvBnAct::load(vb.pp("down2_conv"), c4, c4, 3, 2, Some(Activation::Silu))?;
        let down2_c2f = C2f::load(vb.pp("down2"), c4 + c5, c5, n_bottlenecks, false)?;

        Ok(Self {
            up1_c2f,
            up2_c2f,
            upsample,
            down1_conv,
            down1_c2f,
            down2_conv,
            down2_c2f,
        })
    }

    /// Forward pass: fuse 3 multi-scale feature maps.
    ///
    /// - `p3`: `[B, C3, H3, W3]` — stride-8 features (largest spatial)
    /// - `p4`: `[B, C4, H4, W4]` — stride-16 features
    /// - `p5`: `[B, C5, H5, W5]` — stride-32 features (smallest spatial)
    ///
    /// Returns `(n3, n4, n5)` — fused features at the same 3 spatial scales.
    pub fn forward_multi(
        &self,
        p3: &DynTensor,
        p4: &DynTensor,
        p5: &DynTensor,
    ) -> Result<(DynTensor, DynTensor, DynTensor)> {
        // --- Top-down (FPN) ---
        // P5 upsampled + P4 → N4
        let p5_up = self.upsample.forward(p5)?;
        let cat1 = DynTensor::cat(&[&p5_up, p4], 1)?;
        let n4 = self.up1_c2f.forward(&cat1)?;

        // N4 upsampled + P3 → N3
        let n4_up = self.upsample.forward(&n4)?;
        let cat2 = DynTensor::cat(&[&n4_up, p3], 1)?;
        let n3 = self.up2_c2f.forward(&cat2)?;

        // --- Bottom-up (PAN) ---
        // N3 downsampled + N4 → N4'
        let n3_down = self.down1_conv.forward(&n3)?;
        let cat3 = DynTensor::cat(&[&n3_down, &n4], 1)?;
        let n4_prime = self.down1_c2f.forward(&cat3)?;

        // N4' downsampled + P5 → N5'
        let n4_down = self.down2_conv.forward(&n4_prime)?;
        let cat4 = DynTensor::cat(&[&n4_down, p5], 1)?;
        let n5_prime = self.down2_c2f.forward(&cat4)?;

        Ok((n3, n4_prime, n5_prime))
    }
}

#[cfg(test)]
#[path = "pan_tests.rs"]
mod tests;
