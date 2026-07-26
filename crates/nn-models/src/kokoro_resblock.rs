// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Residual block for Kokoro ISTFTNet decoder.
//!
//! AdaIN → Snake → Conv1d → AdaIN → Snake → Conv1d with residual skip.
//! Extracted from `kokoro_decoder.rs` for 450-line compliance.

use crate::kokoro_error::KokoroError;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{AdaIn, Conv1d, Conv1dConfig, InstanceNormPrecision, Linear, Module};
use nn_core::var_builder::VarBuilder;
use nn_core::Result;

/// Single residual block: AdaIN -> Snake -> Conv1d -> AdaIN -> Snake -> Conv1d with residual.
///
/// Each block applies two snake-activated convolutions with a residual skip connection.
/// Matches Kokoro's `AdaINResBlock1` from `decoder.rs`.
pub struct ResBlock {
    convs: Vec<(Conv1d, Conv1d)>,
    adains: Vec<(AdaIn, AdaIn)>,
    alpha1: Vec<DynTensor>,
    alpha2: Vec<DynTensor>,
}

impl ResBlock {
    /// Load a residual block with `num_layers` dilated conv pairs.
    ///
    /// Each layer: AdaIN1 -> Snake(alpha1) -> Conv1d(dilation) -> AdaIN2 -> Snake(alpha2) -> Conv1d(1).
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
        style_dim: usize,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let mut convs = Vec::with_capacity(dilations.len());
        let mut adains = Vec::with_capacity(dilations.len());
        let mut alpha1 = Vec::with_capacity(dilations.len());
        let mut alpha2 = Vec::with_capacity(dilations.len());

        for (i, &dilation) in dilations.iter().enumerate() {
            let padding = (kernel_size - 1) * dilation / 2;
            // First conv: dilated
            let w1 = vb.get(
                &[channels, channels, kernel_size],
                &format!("convs1.{i}.weight"),
            )?;
            let b1 = vb.get(&[channels], &format!("convs1.{i}.bias"))?;
            let conv1 = Conv1d::new(
                w1,
                Some(b1),
                Conv1dConfig::default()
                    .with_padding(padding)
                    .with_dilation(dilation),
            )?;
            // Second conv: no dilation
            let padding2 = (kernel_size - 1) / 2;
            let w2 = vb.get(
                &[channels, channels, kernel_size],
                &format!("convs2.{i}.weight"),
            )?;
            let b2 = vb.get(&[channels], &format!("convs2.{i}.bias"))?;
            let conv2 = Conv1d::new(w2, Some(b2), Conv1dConfig::default().with_padding(padding2))?;
            convs.push((conv1, conv2));

            // AdaIN layers (project style -> per-channel affine)
            // MatchPyTorchCpu: F32 accumulation matches PyTorch ATen CPU to prevent
            // compound drift over 48 Generator AdaINs amplified by exp(). (#2691)
            let adain1_w = vb.get(&[2 * channels, style_dim], &format!("adain1.{i}.fc.weight"))?;
            let adain1_b = vb.get(&[2 * channels], &format!("adain1.{i}.fc.bias"))?;
            let adain1 = AdaIn::new_with_precision(
                Linear::new(adain1_w, Some(adain1_b))?,
                1e-5,
                InstanceNormPrecision::MatchPyTorchCpu,
            )?;

            let adain2_w = vb.get(&[2 * channels, style_dim], &format!("adain2.{i}.fc.weight"))?;
            let adain2_b = vb.get(&[2 * channels], &format!("adain2.{i}.fc.bias"))?;
            let adain2 = AdaIn::new_with_precision(
                Linear::new(adain2_w, Some(adain2_b))?,
                1e-5,
                InstanceNormPrecision::MatchPyTorchCpu,
            )?;
            adains.push((adain1, adain2));

            // Snake alpha parameters (learnable, per-channel)
            let a1 = vb.get(&[1, channels, 1], &format!("alpha1.{i}"))?;
            let a2 = vb.get(&[1, channels, 1], &format!("alpha2.{i}"))?;
            alpha1.push(a1);
            alpha2.push(a2);
        }

        Ok(Self {
            convs,
            adains,
            alpha1,
            alpha2,
        })
    }

    /// Number of dilation layers in this block.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.convs.len()
    }

    /// Access conv layer pairs: `(conv1, conv2)` per dilation layer.
    ///
    /// Used by LoRA wrapping to extract frozen weights for adapter construction.
    #[must_use]
    pub fn conv_pairs(&self) -> &[(Conv1d, Conv1d)] {
        &self.convs
    }

    /// Forward: x + sum of residual layers.
    ///
    /// `x`: `[B, C, T]`, `style`: `[B, style_dim]`.
    ///
    /// Each dilation layer traces individual ops so that `AdaIn::forward_snake`
    /// compiles as `NativeOp::AdainSnake` (1 Metal dispatch). The former
    /// `FusedAdainResBlock` path produced ~45 Metal dispatches per layer
    /// due to `expand_norm_ops` decomposition. See #2590.
    pub fn forward(
        &self,
        x: &DynTensor,
        style: &DynTensor,
    ) -> std::result::Result<DynTensor, KokoroError> {
        let mut out = x.clone();
        for (i, ((conv1, conv2), (adain1, adain2))) in
            self.convs.iter().zip(self.adains.iter()).enumerate()
        {
            let alpha1 = &self.alpha1[i];
            let alpha2 = &self.alpha2[i];

            let x_in = out;

            // AdaIN1+Snake1 -> Conv1
            let h = adain1.forward_snake(&x_in, style, alpha1)?;
            let h = conv1.forward(&h)?;

            // AdaIN2+Snake2 -> Conv2
            let h = adain2.forward_snake(&h, style, alpha2)?;
            let h = conv2.forward(&h)?;

            out = x_in.add(&h)?;
        }
        Ok(out)
    }
}
