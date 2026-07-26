// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ResNet-18 backbone for detection models (Table Transformer, DETR).
//!
//! Implements the standard ResNet-18 architecture from:
//! *Deep Residual Learning for Image Recognition* (He et al., 2015).
//!
//! - [`BasicBlock`]: Two-convolution residual block with skip connection.
//! - [`ResNet18`]: Full ResNet-18 backbone with multi-scale feature extraction.
//!
//! # Feature maps
//!
//! [`ResNet18::forward_features`] returns `[C2, C3, C4, C5]` multi-scale feature
//! maps suitable for Feature Pyramid Networks (FPN) and DETR-style detectors.
//!
//! # Shape propagation (224x224 input)
//!
//! | Stage     | Output shape            |
//! |-----------|-------------------------|
//! | conv1+bn1 | `[B, 64, 112, 112]`    |
//! | maxpool   | `[B, 64, 56, 56]`      |
//! | layer1    | `[B, 64, 56, 56]` (C2) |
//! | layer2    | `[B, 128, 28, 28]` (C3)|
//! | layer3    | `[B, 256, 14, 14]` (C4)|
//! | layer4    | `[B, 512, 7, 7]` (C5)  |

use crate::dyn_tensor::DynTensor;
use crate::error::Result;
use crate::layers::{BatchNorm2d, BatchNormConfig, Conv2d, Conv2dConfig, Linear, Module};
use crate::var_builder::VarBuilder;

// ---------------------------------------------------------------------------
// BasicBlock
// ---------------------------------------------------------------------------

/// Two-convolution residual block (ResNet-18/34 building block).
///
/// ```text
/// x ─┬─ conv1(3x3) → bn1 → relu → conv2(3x3) → bn2 ─┬─ relu → out
///    │                                                  │
///    └──────────── downsample (optional 1x1 conv) ──────┘
/// ```
///
/// When `in_channels != out_channels` or `stride != 1`, a 1x1 convolution
/// downsample path matches dimensions for the skip addition.
#[derive(Clone, Debug)]
pub struct BasicBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    downsample: Option<(Conv2d, BatchNorm2d)>,
}

impl BasicBlock {
    /// Create from pre-loaded layers.
    pub fn new(
        conv1: Conv2d,
        bn1: BatchNorm2d,
        conv2: Conv2d,
        bn2: BatchNorm2d,
        downsample: Option<(Conv2d, BatchNorm2d)>,
    ) -> Self {
        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            downsample,
        }
    }

    /// Load a BasicBlock from a [`VarBuilder`].
    ///
    /// - `in_c`: input channels
    /// - `out_c`: output channels
    /// - `stride`: stride for the first convolution (1 or 2)
    ///
    /// Weight names follow PyTorch's `torchvision.models.resnet` naming:
    /// `"conv1.weight"`, `"bn1.*"`, `"conv2.weight"`, `"bn2.*"`,
    /// `"downsample.0.weight"`, `"downsample.1.*"`.
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        in_c: usize,
        out_c: usize,
        stride: usize,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let bn_cfg = BatchNormConfig::default();

        // conv1: 3x3, stride as specified, padding 1
        let conv1_cfg = Conv2dConfig::new(1, stride, 1);
        let conv1 = Conv2d::load(vb.pp("conv1"), in_c, out_c, 3, conv1_cfg)?;
        let bn1 = BatchNorm2d::load(vb.pp("bn1"), out_c, bn_cfg)?;

        // conv2: 3x3, stride 1, padding 1
        let conv2_cfg = Conv2dConfig::new(1, 1, 1);
        let conv2 = Conv2d::load(vb.pp("conv2"), out_c, out_c, 3, conv2_cfg)?;
        let bn2 = BatchNorm2d::load(vb.pp("bn2"), out_c, bn_cfg)?;

        // Downsample when dimensions change
        let downsample = if stride != 1 || in_c != out_c {
            let ds_vb = vb.pp("downsample");
            let ds_conv_cfg = Conv2dConfig::new(0, stride, 1);
            let ds_conv = Conv2d::load(ds_vb.pp("0"), in_c, out_c, 1, ds_conv_cfg)?;
            let ds_bn = BatchNorm2d::load(ds_vb.pp("1"), out_c, bn_cfg)?;
            Some((ds_conv, ds_bn))
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            downsample,
        })
    }

    /// Forward pass with residual addition.
    pub fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let identity = match &self.downsample {
            Some((conv, bn)) => bn.forward(&conv.forward(x)?)?,
            None => x.clone(),
        };

        let out = self.conv1.forward(x)?;
        let out = self.bn1.forward(&out)?;
        let out = out.relu()?;
        let out = self.conv2.forward(&out)?;
        let out = self.bn2.forward(&out)?;

        let out = out.broadcast_add(&identity)?;
        out.relu()
    }
}

// ---------------------------------------------------------------------------
// ResNet18
// ---------------------------------------------------------------------------

/// ResNet-18 backbone.
///
/// Standard architecture: conv1(7x7) → bn1 → relu → maxpool → 4 layer groups
/// of 2 [`BasicBlock`]s each, producing feature maps at strides 4, 8, 16, 32.
///
/// # Classification vs. feature extraction
///
/// - [`forward`](ResNet18::forward): Global average pool → flatten → FC.
///   Returns `[B, num_classes]`.
/// - [`forward_features`](ResNet18::forward_features): Returns multi-scale
///   feature maps `[C2, C3, C4, C5]` for FPN / DETR decoders.
#[derive(Clone, Debug)]
pub struct ResNet18 {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    layer1: [BasicBlock; 2],
    layer2: [BasicBlock; 2],
    layer3: [BasicBlock; 2],
    layer4: [BasicBlock; 2],
    fc: Option<Linear>,
}

impl ResNet18 {
    /// Load from a [`VarBuilder`].
    ///
    /// - `num_classes`: number of output classes. Pass `Some(1000)` for
    ///   ImageNet classification, or `None` for feature-extraction-only
    ///   (no FC layer loaded).
    ///
    /// Weight names follow PyTorch `torchvision.models.resnet18` naming.
    pub fn load(vb: impl AsRef<VarBuilder>, num_classes: Option<usize>) -> Result<Self> {
        let vb = vb.as_ref();
        let bn_cfg = BatchNormConfig::default();

        // Stem: 7x7 conv, stride 2, padding 3
        let conv1_cfg = Conv2dConfig::new(3, 2, 1);
        let conv1 = Conv2d::load(vb.pp("conv1"), 3, 64, 7, conv1_cfg)?;
        let bn1 = BatchNorm2d::load(vb.pp("bn1"), 64, bn_cfg)?;

        // Layer groups
        let l1_vb = vb.pp("layer1");
        let layer1 = [
            BasicBlock::load(l1_vb.pp("0"), 64, 64, 1)?,
            BasicBlock::load(l1_vb.pp("1"), 64, 64, 1)?,
        ];

        let l2_vb = vb.pp("layer2");
        let layer2 = [
            BasicBlock::load(l2_vb.pp("0"), 64, 128, 2)?,
            BasicBlock::load(l2_vb.pp("1"), 128, 128, 1)?,
        ];

        let l3_vb = vb.pp("layer3");
        let layer3 = [
            BasicBlock::load(l3_vb.pp("0"), 128, 256, 2)?,
            BasicBlock::load(l3_vb.pp("1"), 256, 256, 1)?,
        ];

        let l4_vb = vb.pp("layer4");
        let layer4 = [
            BasicBlock::load(l4_vb.pp("0"), 256, 512, 2)?,
            BasicBlock::load(l4_vb.pp("1"), 512, 512, 1)?,
        ];

        let fc = match num_classes {
            Some(n) => Some(Linear::load(vb.pp("fc"), 512, n)?),
            None => None,
        };

        Ok(Self {
            conv1,
            bn1,
            layer1,
            layer2,
            layer3,
            layer4,
            fc,
        })
    }

    /// Stem forward: conv1 → bn1 → relu → maxpool.
    ///
    /// Input: `[B, 3, H, W]` → Output: `[B, 64, H/4, W/4]`.
    fn forward_stem(&self, x: &DynTensor) -> Result<DynTensor> {
        let x = self.conv1.forward(x)?;
        let x = self.bn1.forward(&x)?;
        let x = x.relu()?;
        // MaxPool2d(kernel_size=3, stride=2, padding=1)
        x.max_pool2d(3, 2, 1)
    }

    /// Forward a layer group (2 BasicBlocks).
    fn forward_layer(blocks: &[BasicBlock; 2], x: &DynTensor) -> Result<DynTensor> {
        let x = blocks[0].forward(x)?;
        blocks[1].forward(&x)
    }

    /// Classification forward: stem → layers → avgpool → flatten → fc.
    ///
    /// Returns `[B, num_classes]`. Requires `num_classes` to have been set
    /// at load time.
    pub fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let x = self.forward_stem(x)?;
        let x = Self::forward_layer(&self.layer1, &x)?;
        let x = Self::forward_layer(&self.layer2, &x)?;
        let x = Self::forward_layer(&self.layer3, &x)?;
        let x = Self::forward_layer(&self.layer4, &x)?;

        // Global average pooling → [B, 512, 1, 1] → [B, 512]
        let x = x.adaptive_avg_pool2d(1, 1)?;
        let x = x.flatten(1, 3)?;

        match &self.fc {
            Some(fc) => fc.forward(&x),
            None => Ok(x),
        }
    }

    /// Multi-scale feature extraction.
    ///
    /// Returns `[C2, C3, C4, C5]`:
    /// - C2: layer1 output, stride 4, channels 64
    /// - C3: layer2 output, stride 8, channels 128
    /// - C4: layer3 output, stride 16, channels 256
    /// - C5: layer4 output, stride 32, channels 512
    pub fn forward_features(&self, x: &DynTensor) -> Result<Vec<DynTensor>> {
        let x = self.forward_stem(x)?;
        let c2 = Self::forward_layer(&self.layer1, &x)?;
        let c3 = Self::forward_layer(&self.layer2, &c2)?;
        let c4 = Self::forward_layer(&self.layer3, &c3)?;
        let c5 = Self::forward_layer(&self.layer4, &c4)?;
        Ok(vec![c2, c3, c4, c5])
    }

    /// Access the stem conv layer.
    #[must_use]
    pub fn conv1(&self) -> &Conv2d {
        &self.conv1
    }

    /// Access the stem batch norm layer.
    #[must_use]
    pub fn bn1(&self) -> &BatchNorm2d {
        &self.bn1
    }
}

#[cfg(test)]
#[path = "resnet_tests.rs"]
mod resnet_tests;
