// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HuggingFace-compatible ResNet-18 backbone for RT-DETR detection models.
//!
//! HuggingFace RT-DETR uses a different stem architecture than torchvision:
//! three sequential 3x3 convolutions instead of a single 7x7 convolution.
//! Both produce `[B, 64, H/4, W/4]` after the stem + maxpool.
//!
//! # Stem differences
//!
//! | Component | Torchvision (`ResNet18`) | HuggingFace (`ResNet18Hf`) |
//! |-----------|------------------------|---------------------------|
//! | conv0     | 7x7, s=2, 3->64       | 3x3, s=2, 3->32          |
//! | conv1     | (none)                 | 3x3, s=1, 32->32         |
//! | conv2     | (none)                 | 3x3, s=1, 32->64         |
//! | maxpool   | 3x3, s=2, p=1         | 3x3, s=2, p=1            |
//!
//! The residual block structure (BasicBlock) is identical.
//!
//! # Weight naming (internal nn convention)
//!
//! ```text
//! stem.0.conv.weight, stem.0.bn.weight, stem.0.bn.bias, ...
//! stem.1.conv.weight, stem.1.bn.weight, ...
//! stem.2.conv.weight, stem.2.bn.weight, ...
//! layer1.0.conv1.weight, layer1.0.bn1.weight, ...
//! layer2.0.downsample.0.weight, layer2.0.downsample.1.weight, ...
//! ```
//!
//! The weight key remapper in `convert_dpdf.rs` maps HuggingFace naming
//! to these internal keys.

use crate::dyn_tensor::DynTensor;
use crate::error::Result;
use crate::layers::{BatchNorm2d, BatchNormConfig, Conv2d, Conv2dConfig, Linear, Module};
use crate::var_builder::VarBuilder;

use super::resnet::BasicBlock;

// ---------------------------------------------------------------------------
// HF Stem: three 3x3 convolutions
// ---------------------------------------------------------------------------

/// HuggingFace-style ResNet stem with three sequential 3x3 convolutions.
///
/// ```text
/// [B, 3, H, W] -> conv0(3x3,s=2) -> bn0 -> relu
///              -> conv1(3x3,s=1) -> bn1 -> relu
///              -> conv2(3x3,s=1) -> bn2 -> relu
///              -> maxpool(3x3,s=2,p=1)
///              -> [B, 64, H/4, W/4]
/// ```
#[derive(Clone, Debug)]
pub(crate) struct HfStem {
    conv0: Conv2d,
    bn0: BatchNorm2d,
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
}

impl HfStem {
    /// Load from VarBuilder with keys like `stem.0.conv.weight`, `stem.0.bn.*`.
    fn load(vb: &VarBuilder) -> Result<Self> {
        let bn_cfg = BatchNormConfig::default();

        let s0 = vb.pp("0");
        let conv0 = Conv2d::load(s0.pp("conv"), 3, 32, 3, Conv2dConfig::new(1, 2, 1))?;
        let bn0 = BatchNorm2d::load(s0.pp("bn"), 32, bn_cfg)?;

        let s1 = vb.pp("1");
        let conv1 = Conv2d::load(s1.pp("conv"), 32, 32, 3, Conv2dConfig::new(1, 1, 1))?;
        let bn1 = BatchNorm2d::load(s1.pp("bn"), 32, bn_cfg)?;

        let s2 = vb.pp("2");
        let conv2 = Conv2d::load(s2.pp("conv"), 32, 64, 3, Conv2dConfig::new(1, 1, 1))?;
        let bn2 = BatchNorm2d::load(s2.pp("bn"), 64, bn_cfg)?;

        Ok(Self {
            conv0,
            bn0,
            conv1,
            bn1,
            conv2,
            bn2,
        })
    }

    /// Forward: three conv+bn+relu stages followed by maxpool.
    ///
    /// Input: `[B, 3, H, W]` -> Output: `[B, 64, H/4, W/4]`
    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let x = self.conv0.forward(x)?;
        let x = self.bn0.forward(&x)?;
        let x = x.relu()?;

        let x = self.conv1.forward(&x)?;
        let x = self.bn1.forward(&x)?;
        let x = x.relu()?;

        let x = self.conv2.forward(&x)?;
        let x = self.bn2.forward(&x)?;
        let x = x.relu()?;

        // MaxPool2d(kernel_size=3, stride=2, padding=1)
        x.max_pool2d(3, 2, 1)
    }
}

// ---------------------------------------------------------------------------
// ResNet18Hf
// ---------------------------------------------------------------------------

/// HuggingFace-compatible ResNet-18 backbone.
///
/// Uses a 3-stage 3x3 stem (HF convention) instead of the single 7x7
/// conv stem used by torchvision. The residual blocks are identical.
///
/// # Feature extraction
///
/// [`forward_features`](ResNet18Hf::forward_features) returns multi-scale
/// feature maps `[C2, C3, C4, C5]` for FPN / DETR decoders, matching the
/// interface of [`super::ResNet18`].
#[derive(Clone, Debug)]
pub struct ResNet18Hf {
    stem: HfStem,
    layer1: [BasicBlock; 2],
    layer2: [BasicBlock; 2],
    layer3: [BasicBlock; 2],
    layer4: [BasicBlock; 2],
    fc: Option<Linear>,
}

impl ResNet18Hf {
    /// Load from a [`VarBuilder`].
    ///
    /// Weight keys expected:
    /// - `stem.{0,1,2}.conv.weight`, `stem.{0,1,2}.bn.*`
    /// - `layer{1-4}.{0,1}.conv{1,2}.weight`, `layer{1-4}.{0,1}.bn{1,2}.*`
    /// - `layer{2-4}.0.downsample.{0,1}.*`
    /// - `fc.weight`, `fc.bias` (only if `num_classes` is Some)
    pub fn load(vb: impl AsRef<VarBuilder>, num_classes: Option<usize>) -> Result<Self> {
        let vb = vb.as_ref();

        let stem = HfStem::load(&vb.pp("stem"))?;

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
            stem,
            layer1,
            layer2,
            layer3,
            layer4,
            fc,
        })
    }

    /// Forward a layer group (2 BasicBlocks).
    fn forward_layer(blocks: &[BasicBlock; 2], x: &DynTensor) -> Result<DynTensor> {
        let x = blocks[0].forward(x)?;
        blocks[1].forward(&x)
    }

    /// Classification forward: stem -> layers -> avgpool -> flatten -> fc.
    ///
    /// Returns `[B, num_classes]`. Requires `num_classes` to have been set
    /// at load time.
    pub fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let x = self.stem.forward(x)?;
        let x = Self::forward_layer(&self.layer1, &x)?;
        let x = Self::forward_layer(&self.layer2, &x)?;
        let x = Self::forward_layer(&self.layer3, &x)?;
        let x = Self::forward_layer(&self.layer4, &x)?;

        // Global average pooling -> [B, 512, 1, 1] -> [B, 512]
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
        let x = self.stem.forward(x)?;
        let c2 = Self::forward_layer(&self.layer1, &x)?;
        let c3 = Self::forward_layer(&self.layer2, &c2)?;
        let c4 = Self::forward_layer(&self.layer3, &c3)?;
        let c5 = Self::forward_layer(&self.layer4, &c4)?;
        Ok(vec![c2, c3, c4, c5])
    }
}

#[cfg(test)]
#[path = "resnet_hf_tests.rs"]
mod resnet_hf_tests;
