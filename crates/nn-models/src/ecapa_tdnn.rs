// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ECAPA-TDNN speaker verification encoder.
//!
//! Produces 192-dimensional speaker embeddings from 80-bin mel spectrograms.
//! Architecture: Conv1d → 3 SE-Res2Blocks → Cat → Conv1d → ASP → BN + Linear.
//!
//! Standard architecture from Desplanques et al. 2020, ~6.2M params:
//! - Input: `[B, 80, T]` mel spectrogram
//! - Output: `[B, 192]` speaker embedding (L2-normalized)
//!
//! Citation: Desplanques et al. 2020, "ECAPA-TDNN: Emphasized Channel Attention,
//! Propagation and Aggregation in TDNN Based Speaker Verification", Interspeech.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{
    AttentiveStatisticsPooling, BatchNorm, BatchNormConfig, Conv1dConfig, Linear, Module,
};
use nn_core::var_builder::VarBuilder;
use nn_core::{Result, TensorError};

use crate::ecapa_tdnn_block::SERes2Block;

/// ECAPA-TDNN-512 architecture constants.
const MEL_CHANNELS: usize = 80;
const HIDDEN_CHANNELS: usize = 512;
const EMBED_DIM: usize = 192;
const RES2NET_SCALE: usize = 8;
const SE_REDUCTION: usize = 128;
const DILATIONS: [usize; 3] = [2, 3, 4];

/// ECAPA-TDNN speaker verification encoder.
///
/// Produces 192-dimensional speaker embeddings from 80-bin mel spectrograms.
///
/// # Architecture
///
/// ```text
/// mel [B, 80, T]
///   → Conv1d(80, 512, k=5) + ReLU + BN
///   → SE-Res2Block(512, 512, k=3, d=2) → skip_1
///   → SE-Res2Block(512, 512, k=3, d=3) → skip_2
///   → SE-Res2Block(512, 512, k=3, d=4) → skip_3
///   → Cat(skip_1, skip_2, skip_3) → [B, 1536, T]
///   → Conv1d(1536, 1536, k=1) + ReLU
///   → AttentiveStatisticsPooling → [B, 3072]
///   → BN + Linear(3072, 192) → embedding [B, 192]
/// ```
#[derive(Debug, Clone)]
pub struct EcapaTdnn {
    initial_conv: nn_core::layers::Conv1d,
    initial_bn: BatchNorm,
    blocks: Vec<SERes2Block>,
    cat_conv: nn_core::layers::Conv1d,
    asp: AttentiveStatisticsPooling,
    final_bn: BatchNorm,
    final_linear: Linear,
}

impl EcapaTdnn {
    /// Load from VarBuilder.
    pub fn load(vb: impl AsRef<VarBuilder>) -> Result<Self> {
        let vb = vb.as_ref();
        // Initial Conv1d(80, 512, k=5) with padding=2 to preserve temporal dim.
        let initial_conv = nn_core::layers::conv1d(
            MEL_CHANNELS,
            HIDDEN_CHANNELS,
            5,
            Conv1dConfig::default().with_padding(2),
            vb.pp("initial_conv"),
        )?;
        let initial_bn = BatchNorm::load(
            vb.pp("initial_bn"),
            HIDDEN_CHANNELS,
            BatchNormConfig::default(),
        )?;

        // 3 SE-Res2Blocks with dilations [2, 3, 4].
        let mut blocks = Vec::with_capacity(DILATIONS.len());
        for (i, &dilation) in DILATIONS.iter().enumerate() {
            let block = SERes2Block::load(
                vb.pp(format!("blocks.{i}")),
                HIDDEN_CHANNELS,
                HIDDEN_CHANNELS,
                3, // kernel_size
                dilation,
                RES2NET_SCALE,
                SE_REDUCTION,
            )?;
            blocks.push(block);
        }

        // Concatenation conv: 3 * 512 = 1536 → 1536.
        let cat_channels = HIDDEN_CHANNELS * DILATIONS.len();
        let cat_conv = nn_core::layers::conv1d(
            cat_channels,
            cat_channels,
            1,
            Conv1dConfig::default(),
            vb.pp("cat_conv"),
        )?;

        // Attentive Statistics Pooling.
        let asp = AttentiveStatisticsPooling::load(vb.pp("asp"), cat_channels)?;

        // Final BN + Linear(2 * 1536, 192).
        let final_bn = BatchNorm::load(
            vb.pp("final_bn"),
            cat_channels * 2,
            BatchNormConfig::default(),
        )?;
        let final_linear = Linear::load(vb.pp("final_linear"), cat_channels * 2, EMBED_DIM)?;

        Ok(Self {
            initial_conv,
            initial_bn,
            blocks,
            cat_conv,
            asp,
            final_bn,
            final_linear,
        })
    }

    /// Embedding dimension (192).
    #[must_use]
    pub fn embed_dim(&self) -> usize {
        EMBED_DIM
    }

    /// Compute speaker embedding from mel spectrogram.
    ///
    /// Input: `[B, 80, T]` mel features.
    /// Output: `[B, 192]` L2-normalized speaker embedding.
    pub fn forward(&self, mel: &DynTensor) -> Result<DynTensor> {
        if mel.rank() != 3 {
            return Err(TensorError::RankMismatch {
                expected: 3,
                actual: mel.rank(),
            });
        }
        if mel.dims()[1] != MEL_CHANNELS {
            return Err(TensorError::shape_mismatch(
                vec![0, MEL_CHANNELS, 0],
                mel.dims().to_vec(),
            ));
        }

        // Initial conv + ReLU + BN.
        let x = self.initial_conv.forward(mel)?;
        let x = x.relu()?;
        let x = self.initial_bn.forward(&x)?;

        // 3 SE-Res2Blocks, collecting skip connections.
        let mut skips = Vec::with_capacity(self.blocks.len());
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x)?;
            skips.push(x.clone());
        }

        // Concatenate skip connections along channel dim.
        let skip_refs: Vec<&DynTensor> = skips.iter().collect();
        let x = DynTensor::cat(&skip_refs, 1)?;

        // Cat conv + ReLU.
        let x = self.cat_conv.forward(&x)?;
        let x = x.relu()?;

        // ASP → [B, 2*C].
        let x = self.asp.forward(&x)?;

        // Final BN + Linear → [B, 192].
        let x = self.final_bn.forward(&x)?;
        let x = self.final_linear.forward(&x)?;

        // L2-normalize the embedding.
        let norm = x.sqr()?.sum_keepdim(1)?.sqrt()?;
        let device = x.device();
        let eps = DynTensor::full(&[], 1e-12, x.dtype(), &device)?;
        let norm = norm.maximum(&eps)?;
        x.broadcast_div(&norm)
    }
}

#[cfg(test)]
#[path = "ecapa_tdnn_tests.rs"]
mod tests;
