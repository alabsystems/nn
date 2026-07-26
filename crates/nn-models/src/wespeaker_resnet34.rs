// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! WeSpeaker ResNet34 speaker embedding model.
//!
//! Produces 256-dimensional speaker embeddings from 80-bin fbank features.
//! Architecture: Conv2d stem → ResNet34 body → TSTP pooling → Linear head.
//!
//! Reference: pyannote/wespeaker-voxceleb-resnet34-LM (6.6M params).
//!
//! Input: `[B, 1, T, 80]` fbank features (T = number of frames).
//! Output: `[B, 256]` speaker embedding.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Linear, Module};
use nn_core::var_builder::VarBuilder;
use nn_core::{Result, TensorError};

/// Number of mel/fbank frequency bins.
const FEAT_DIM: usize = 80;
/// Output embedding dimension.
const EMBED_DIM: usize = 256;
/// Base channel multiplier (m_channels in WeSpeaker).
const M: usize = 32;

/// Layer configurations: (num_blocks, out_channels, stride).
const LAYER_CONFIGS: [(usize, usize, usize); 4] = [
    (3, M, 1),     // layer1: 3x BasicBlock(32→32, stride=1)
    (4, M * 2, 2), // layer2: 4x BasicBlock(32→64, stride=2 first block)
    (6, M * 4, 2), // layer3: 6x BasicBlock(64→128, stride=2 first block)
    (3, M * 8, 2), // layer4: 3x BasicBlock(128→256, stride=2 first block)
];

// ---------------------------------------------------------------------------
// BasicBlock
// ---------------------------------------------------------------------------

/// ResNet BasicBlock: two 3×3 convolutions with optional downsampling shortcut.
///
/// ```text
/// x → Conv2d(3×3) → BN → ReLU → Conv2d(3×3) → BN → + shortcut → ReLU
/// ```
///
/// When `stride > 1` or `in_channels != out_channels`, the shortcut uses
/// `Conv2d(1×1, stride) → BN` to match dimensions.
#[derive(Debug, Clone)]
struct BasicBlock {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    downsample: Option<(Conv2d, BatchNorm)>,
}

impl BasicBlock {
    fn load(
        vb: impl AsRef<VarBuilder>,
        in_channels: usize,
        out_channels: usize,
        stride: usize,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let bn_cfg = BatchNormConfig::default();

        // Conv1: 3×3, stride, padding=1.
        let conv1_cfg = Conv2dConfig::new(1, stride, 1);
        let conv1 = Conv2d::load(vb.pp("conv1"), in_channels, out_channels, 3, conv1_cfg)?;
        let bn1 = BatchNorm::load(vb.pp("bn1"), out_channels, bn_cfg)?;

        // Conv2: 3×3, stride=1, padding=1.
        let conv2_cfg = Conv2dConfig::new(1, 1, 1);
        let conv2 = Conv2d::load(vb.pp("conv2"), out_channels, out_channels, 3, conv2_cfg)?;
        let bn2 = BatchNorm::load(vb.pp("bn2"), out_channels, bn_cfg)?;

        // Downsample shortcut when dimensions change.
        let downsample = if stride != 1 || in_channels != out_channels {
            let ds_conv = Conv2d::load(
                vb.pp("downsample.0"),
                in_channels,
                out_channels,
                1,
                Conv2dConfig::new(0, stride, 1),
            )?;
            let ds_bn = BatchNorm::load(vb.pp("downsample.1"), out_channels, bn_cfg)?;
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

    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
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
// ResNet34 layer (sequence of BasicBlocks)
// ---------------------------------------------------------------------------

/// A ResNet layer: sequence of `BasicBlock`s.
///
/// The first block may downsample (stride > 1); remaining blocks use stride 1.
#[derive(Debug, Clone)]
struct ResNetLayer {
    blocks: Vec<BasicBlock>,
}

impl ResNetLayer {
    fn load(
        vb: impl AsRef<VarBuilder>,
        in_channels: usize,
        out_channels: usize,
        num_blocks: usize,
        stride: usize,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let mut blocks = Vec::with_capacity(num_blocks);

        // First block: may downsample.
        blocks.push(BasicBlock::load(
            vb.pp("0"),
            in_channels,
            out_channels,
            stride,
        )?);

        // Remaining blocks: no downsampling.
        for i in 1..num_blocks {
            blocks.push(BasicBlock::load(
                vb.pp(i.to_string()),
                out_channels,
                out_channels,
                1,
            )?);
        }

        Ok(Self { blocks })
    }

    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let mut out = self.blocks[0].forward(x)?;
        for block in &self.blocks[1..] {
            out = block.forward(&out)?;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// TSTP: Temporal Statistics Pooling
// ---------------------------------------------------------------------------

/// Temporal Statistics Pooling (TSTP).
///
/// Input: `[B, C, T, F]` where T = time frames, F = frequency bins.
/// Permutes to `[B, C, F, T]`, reshapes to `[B, C*F, T]`, computes
/// mean and unbiased standard deviation over the time dimension.
/// Output: `[B, 2*C*F]`.
fn tstp_pool(x: &DynTensor) -> Result<DynTensor> {
    let dims = x.dims();
    if dims.len() != 4 {
        return Err(TensorError::RankMismatch {
            expected: 4,
            actual: dims.len(),
        });
    }
    let (b, c, t, f) = (dims[0], dims[1], dims[2], dims[3]);

    // Permute [B, C, T, F] → [B, C, F, T] then reshape to [B, C*F, T].
    let x = x.permute([0, 1, 3, 2])?;
    let x = x.reshape([b, c * f, t])?;

    // Mean over time dim (dim=2), keep rank for broadcast.
    let mean = x.mean_keepdim(2)?;

    // Unbiased variance: var = sum((x - mean)^2) / (T - 1).
    let diff = x.broadcast_sub(&mean)?;
    let sq = diff.sqr()?;
    let var = sq.sum_keepdim(2)?;
    let n = if t > 1 { (t - 1) as f64 } else { 1.0 };
    let var = var.mul_scalar(1.0 / n)?;
    let std = var.sqrt()?;

    // Squeeze time dim: [B, C*F, 1] → [B, C*F].
    let mean = mean.squeeze(2)?;
    let std = std.squeeze(2)?;

    // Concatenate mean and std: [B, 2*C*F].
    DynTensor::cat(&[&mean, &std], 1)
}

// ---------------------------------------------------------------------------
// WeSpeaker ResNet34
// ---------------------------------------------------------------------------

/// WeSpeaker ResNet34 speaker embedding model.
///
/// Input: `[B, 1, T, 80]` fbank features (1 channel, T frames, 80 freq bins).
/// Output: `[B, 256]` speaker embedding.
///
/// # Architecture
///
/// ```text
/// fbank [B, 1, T, 80]
///   → Conv2d(1, 32, 3×3, pad=1) + BN + ReLU
///   → layer1: 3× BasicBlock(32, 32, s=1)
///   → layer2: 4× BasicBlock(32, 64, s=2)
///   → layer3: 6× BasicBlock(64, 128, s=2)
///   → layer4: 3× BasicBlock(128, 256, s=2)
///   → TSTP pool → [B, 5120]
///   → Linear(5120, 256) → [B, 256]
/// ```
#[derive(Debug, Clone)]
pub struct WeSpeakerResNet34 {
    conv1: Conv2d,
    bn1: BatchNorm,
    layer1: ResNetLayer,
    layer2: ResNetLayer,
    layer3: ResNetLayer,
    layer4: ResNetLayer,
    fc: Linear,
}

impl WeSpeakerResNet34 {
    /// Load from a VarBuilder (safetensors weight mapping).
    ///
    /// Expected weight prefix layout:
    /// - `conv1.{weight,bias}`, `bn1.{weight,bias,running_mean,running_var}`
    /// - `layer1.0.{conv1,bn1,conv2,bn2}...`, `layer1.1...`, etc.
    /// - `layer2.0.{..., downsample.0, downsample.1}`, etc.
    /// - `fc.{weight,bias}`
    pub fn load(vb: impl AsRef<VarBuilder>) -> Result<Self> {
        let vb = vb.as_ref();
        let bn_cfg = BatchNormConfig::default();

        // Stem: Conv2d(1, 32, 3×3, stride=1, padding=1) + BN.
        let conv1 = Conv2d::load(vb.pp("conv1"), 1, M, 3, Conv2dConfig::new(1, 1, 1))?;
        let bn1 = BatchNorm::load(vb.pp("bn1"), M, bn_cfg)?;

        // ResNet layers.
        let mut in_ch = M;
        let layer1 = {
            let (n, out_ch, s) = LAYER_CONFIGS[0];
            let l = ResNetLayer::load(vb.pp("layer1"), in_ch, out_ch, n, s)?;
            in_ch = out_ch;
            l
        };
        let layer2 = {
            let (n, out_ch, s) = LAYER_CONFIGS[1];
            let l = ResNetLayer::load(vb.pp("layer2"), in_ch, out_ch, n, s)?;
            in_ch = out_ch;
            l
        };
        let layer3 = {
            let (n, out_ch, s) = LAYER_CONFIGS[2];
            let l = ResNetLayer::load(vb.pp("layer3"), in_ch, out_ch, n, s)?;
            in_ch = out_ch;
            l
        };
        let layer4 = {
            let (n, out_ch, s) = LAYER_CONFIGS[3];
            let l = ResNetLayer::load(vb.pp("layer4"), in_ch, out_ch, n, s)?;
            in_ch = out_ch;
            l
        };

        // After layer4, spatial dims: freq = 80 / 8 = 10 (3 stride-2 layers).
        // TSTP output: 2 * (in_ch * 10) = 2 * 2560 = 5120.
        let pool_dim = in_ch * (FEAT_DIM / 8) * 2;
        let fc = Linear::load(vb.pp("fc"), pool_dim, EMBED_DIM)?;

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

    /// Embedding dimension (256).
    #[must_use]
    pub fn embed_dim(&self) -> usize {
        EMBED_DIM
    }

    /// Compute speaker embedding from fbank features.
    ///
    /// Input: `[B, 1, T, 80]` fbank features.
    /// Output: `[B, 256]` speaker embedding.
    pub fn forward(&self, fbank: &DynTensor) -> Result<DynTensor> {
        if fbank.rank() != 4 {
            return Err(TensorError::RankMismatch {
                expected: 4,
                actual: fbank.rank(),
            });
        }
        if fbank.dims()[1] != 1 {
            return Err(TensorError::shape_mismatch(
                vec![0, 1, 0, FEAT_DIM],
                fbank.dims().to_vec(),
            ));
        }
        if fbank.dims()[3] != FEAT_DIM {
            return Err(TensorError::shape_mismatch(
                vec![0, 1, 0, FEAT_DIM],
                fbank.dims().to_vec(),
            ));
        }

        // Stem.
        let x = self.conv1.forward(fbank)?;
        let x = self.bn1.forward(&x)?;
        let x = x.relu()?;

        // ResNet body.
        let x = self.layer1.forward(&x)?;
        let x = self.layer2.forward(&x)?;
        let x = self.layer3.forward(&x)?;
        let x = self.layer4.forward(&x)?;

        // TSTP pooling → [B, 5120].
        let x = tstp_pool(&x)?;

        // Embedding head.
        self.fc.forward(&x)
    }
}

#[cfg(test)]
#[path = "wespeaker_resnet34_tests.rs"]
mod tests;
