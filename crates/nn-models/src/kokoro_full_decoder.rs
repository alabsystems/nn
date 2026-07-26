// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro FullDecoder: Stage 1 feature preprocessing + Stage 2 ISTFTNet Generator.
//!
//! The Kokoro Decoder has two stages:
//!   **Stage 1** (this module): AdaIN residual blocks process encoder features
//!     with F0/energy conditioning before the waveform generator.
//!   **Stage 2** (`kokoro_decoder.rs`): ISTFTNet Generator produces audio.
//!
//! Architecture (from hexgrad/Kokoro-82M istftnet.py Decoder):
//!   1. `F0_conv`, `N_conv`: stride-2 Conv1d downsample F0/N from 2T → T
//!   2. `encode`: Stage1ResBlk(514→1024) on cat([asr, F0, N])
//!   3. `asr_res`: Conv1d(512→64, k=1) compressed skip connection
//!   4. `decode[0..2]`: 3× Stage1ResBlk(1090→1024) with [x, asr_res, F0, N] skip
//!   5. `decode[3]`: Stage1ResBlk(1090→512, upsample=2×)
//!   6. Generator([B, 512, 2T], style, har_source) → magnitude + phase
//!
//! Reference: dvoice `crates/dvoice-tts/src/kokoro/stage1.rs`.
//! Part of #2498.

use crate::kokoro_decoder::Generator;
use crate::kokoro_error::{check_tensor_finite, KokoroError};
use crate::kokoro_tts::KokoroConfig;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{
    AdaIn, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, InstanceNormPrecision,
    Module,
};
use nn_core::var_builder::VarBuilder;
use nn_core::Result;

// ---------------------------------------------------------------------------
// Nearest-neighbor 2× upsample
// ---------------------------------------------------------------------------

/// Repeat each timestep twice: `[B, C, T]` → `[B, C, 2T]`.
///
/// Matches PyTorch `nn.Upsample(scale_factor=2, mode='nearest')`.
fn nearest_upsample_2x(x: &DynTensor) -> Result<DynTensor> {
    let (batch, channels, t) = (x.dim(0)?, x.dim(1)?, x.dim(2)?);
    x.unsqueeze(3)?
        .expand([batch, channels, t, 2])?
        .reshape([batch, channels, t * 2])
}

// ---------------------------------------------------------------------------
// Stage1ResBlk: AdaIN residual block for the Decoder's encode/decode blocks
// ---------------------------------------------------------------------------

/// Stage 1 residual block with adaptive instance normalization.
///
/// Two conv layers with AdaIN + LeakyReLU(0.2), optional ConvTranspose1d upsample.
/// Output normalized by `1/sqrt(2)`: `(residual + shortcut) / sqrt(2)`.
///
/// Different from Generator's `ResBlock` (Snake activation, multi-dilation)
/// and from ProsodyPredictor's blocks (no `sqrt(2)` normalization).
///
/// Weight key paths: `conv1`, `conv2`, `conv1x1`, `norm1`, `norm2`, `pool` (if upsample).
pub struct Stage1ResBlk {
    conv1: Conv1d,
    conv2: Conv1d,
    conv1x1: Option<Conv1d>,
    norm1: AdaIn,
    norm2: AdaIn,
    pool: Option<ConvTranspose1d>,
    upsample: bool,
}

impl Stage1ResBlk {
    /// Load a Stage1ResBlk from weights.
    ///
    /// - `dim_in`: input channels.
    /// - `dim_out`: output channels.
    /// - `style_dim`: style embedding dimension (128 for Kokoro).
    /// - `upsample`: if true, adds ConvTranspose1d(k=3, s=2, p=1, groups=dim_in) for 2× upsample.
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        dim_in: usize,
        dim_out: usize,
        style_dim: usize,
        upsample: bool,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        // Conv layers: kernel_size=3, padding=1, dilation=1
        let conv_cfg = Conv1dConfig::default().with_padding(1);

        let conv1_w = vb.get(&[dim_out, dim_in, 3], "conv1.weight")?;
        let conv1_b = vb.get(&[dim_out], "conv1.bias")?;
        let conv1 = Conv1d::new(conv1_w, Some(conv1_b), conv_cfg)?;

        let conv2_w = vb.get(&[dim_out, dim_out, 3], "conv2.weight")?;
        let conv2_b = vb.get(&[dim_out], "conv2.bias")?;
        let conv2 = Conv1d::new(conv2_w, Some(conv2_b), conv_cfg)?;

        // AdaIN normalization layers
        // MatchPyTorchCpu: F32 accumulation matches PyTorch ATen CPU to prevent
        // compound drift over 10 Stage1 AdaINs feeding into Generator. (#2691)
        let norm1 = AdaIn::load_with_precision(
            vb.pp("norm1"),
            style_dim,
            dim_in,
            1e-5,
            InstanceNormPrecision::MatchPyTorchCpu,
        )?;
        let norm2 = AdaIn::load_with_precision(
            vb.pp("norm2"),
            style_dim,
            dim_out,
            1e-5,
            InstanceNormPrecision::MatchPyTorchCpu,
        )?;

        // 1×1 conv for channel projection when dim_in != dim_out.
        // Bias is optional — PyTorch v1.0 Kokoro uses bias=False for conv1x1.
        let conv1x1 = if dim_in != dim_out {
            let w = vb.get(&[dim_out, dim_in, 1], "conv1x1.weight")?;
            let b = if vb.contains_tensor("conv1x1.bias") {
                Some(vb.get(&[dim_out], "conv1x1.bias")?)
            } else {
                None
            };
            Some(Conv1d::new(w, b, Conv1dConfig::default())?)
        } else {
            None
        };

        // Upsample via depthwise ConvTranspose1d: groups=dim_in, k=3, s=2, p=1, op=1
        // output_padding=1 ensures exact 2× output to match nearest_upsample shortcut.
        let pool = if upsample {
            let pool_w = vb.get(&[dim_in, 1, 3], "pool.weight")?;
            let pool_b = vb.get(&[dim_in], "pool.bias")?;
            let pool_cfg = ConvTranspose1dConfig::new(1, 2, 1)
                .with_groups(dim_in)
                .with_output_padding(1);
            Some(ConvTranspose1d::new(pool_w, Some(pool_b), pool_cfg)?)
        } else {
            None
        };

        Ok(Self {
            conv1,
            conv2,
            conv1x1,
            norm1,
            norm2,
            pool,
            upsample,
        })
    }

    /// Forward: `x [B, C_in, T]`, `style [B, style_dim]` → `[B, C_out, T]` (or `2T` if upsample).
    pub fn forward(
        &self,
        x: &DynTensor,
        style: &DynTensor,
    ) -> std::result::Result<DynTensor, KokoroError> {
        let residual = self.residual_path(x, style)?;
        let shortcut = self.shortcut_path(x)?;

        // (residual + shortcut) / sqrt(2)
        let sum = residual.add(&shortcut)?;
        let out = sum.mul_scalar(1.0 / std::f64::consts::SQRT_2)?;
        Ok(out)
    }

    /// Residual path: norm1 → LeakyReLU(0.2) → [pool] → conv1 → norm2 → LeakyReLU(0.2) → conv2.
    fn residual_path(
        &self,
        x: &DynTensor,
        style: &DynTensor,
    ) -> std::result::Result<DynTensor, KokoroError> {
        let mut out = self.norm1.forward_leaky_relu(x, style, 0.2)?;
        if let Some(pool) = &self.pool {
            out = pool.forward(&out)?;
        }
        out = self.conv1.forward(&out)?;
        out = self.norm2.forward_leaky_relu(&out, style, 0.2)?;
        out = self.conv2.forward(&out)?;
        Ok(out)
    }

    /// Access conv1 layer (for LoRA wrapping / weight extraction).
    #[must_use]
    pub fn conv1(&self) -> &Conv1d {
        &self.conv1
    }

    /// Access conv2 layer (for LoRA wrapping / weight extraction).
    #[must_use]
    pub fn conv2(&self) -> &Conv1d {
        &self.conv2
    }

    /// Shortcut path: [nearest_upsample_2x] → [conv1x1].
    fn shortcut_path(&self, x: &DynTensor) -> std::result::Result<DynTensor, KokoroError> {
        let mut out = if self.upsample {
            nearest_upsample_2x(x)?
        } else {
            x.clone()
        };
        if let Some(conv1x1) = &self.conv1x1 {
            out = conv1x1.forward(&out)?;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// FullDecoder: Stage 1 preprocessing + ISTFTNet Generator
// ---------------------------------------------------------------------------

/// Complete Kokoro Decoder: Stage 1 feature preprocessing + Stage 2 waveform generation.
///
/// Stage 1 processes encoder features (asr) with F0/energy conditioning through
/// AdaIN residual blocks with skip connections, then feeds the result to the
/// ISTFTNet Generator for audio synthesis.
///
/// Input:
/// - `asr`: `[B, 512, T]` — length-regulated TextEncoder output.
/// - `f0_curve`: `[B, 1, 2T]` — F0 contour at 2× time resolution.
/// - `n_curve`: `[B, 1, 2T]` — energy at 2× time resolution.
/// - `style`: `[B, 128]` — decoder style vector.
///
/// Output: `(magnitude [B, n_bins, T_out], phase [B, n_bins, T_out])`.
pub struct FullDecoder {
    /// F0 downsampling: Conv1d(1, 1, k=3, s=2, p=1).
    f0_conv: Conv1d,
    /// Energy downsampling: Conv1d(1, 1, k=3, s=2, p=1).
    n_conv: Conv1d,
    /// Compressed skip: Conv1d(512, 64, k=1).
    asr_res: Conv1d,
    /// Encode block: Stage1ResBlk(514→1024).
    encode: Stage1ResBlk,
    /// Decode blocks: 3× (1090→1024) + 1× (1090→512, upsample=2×).
    decode: Vec<Stage1ResBlk>,
    /// Stage 2: ISTFTNet Generator.
    generator: Generator,
}

impl FullDecoder {
    /// Load the FullDecoder from weights.
    ///
    /// Weight prefix: `decoder.` in safetensors. The Generator loads from
    /// `decoder.generator.` via the passed VarBuilder (which should already
    /// be scoped to `decoder`).
    pub fn load(vb: impl AsRef<VarBuilder>, config: &KokoroConfig) -> Result<Self> {
        let vb = vb.as_ref();
        let d_en = config.d_en;
        let style_dim = config.style_dim;
        let asr_res_ch = (d_en / 8).max(1); // 64 for d_en=512
        let hidden = 2 * d_en; // 1024 for d_en=512
        let encode_in = d_en + 2; // asr + F0 + N
        let decode_in = hidden + asr_res_ch + 2; // encoded + asr_res + F0 + N

        // F0/N downsampling: Conv1d(1, 1, k=3, s=2, p=1)
        let f0n_cfg = Conv1dConfig::default().with_padding(1).with_stride(2);
        let f0_w = vb.get(&[1, 1, 3], "F0_conv.weight")?;
        let f0_b = vb.get(&[1], "F0_conv.bias")?;
        let f0_conv = Conv1d::new(f0_w, Some(f0_b), f0n_cfg)?;

        let n_w = vb.get(&[1, 1, 3], "N_conv.weight")?;
        let n_b = vb.get(&[1], "N_conv.bias")?;
        let n_conv = Conv1d::new(n_w, Some(n_b), f0n_cfg)?;

        // Compressed skip: Conv1d(d_en, asr_res_ch, k=1)
        let asr_w = vb.get(&[asr_res_ch, d_en, 1], "asr_res.weight")?;
        let asr_b = vb.get(&[asr_res_ch], "asr_res.bias")?;
        let asr_res = Conv1d::new(asr_w, Some(asr_b), Conv1dConfig::default())?;

        // Encode: Stage1ResBlk(d_en+2 → 2*d_en)
        let encode = Stage1ResBlk::load(vb.pp("encode"), encode_in, hidden, style_dim, false)?;

        // Decode: 3× (decode_in→hidden) + 1× (decode_in→d_en, upsample=2×)
        let mut decode = Vec::with_capacity(4);
        for i in 0..3 {
            decode.push(Stage1ResBlk::load(
                vb.pp(format!("decode.{i}")),
                decode_in,
                hidden,
                style_dim,
                false,
            )?);
        }
        decode.push(Stage1ResBlk::load(
            vb.pp("decode.3"),
            decode_in,
            d_en,
            style_dim,
            true,
        )?);

        // Stage 2: ISTFTNet Generator
        let generator = Generator::load(vb.pp("generator"), config)?;

        Ok(Self {
            f0_conv,
            n_conv,
            asr_res,
            encode,
            decode,
            generator,
        })
    }

    /// Access the inner Generator (diagnostic parity testing).
    #[must_use]
    pub fn generator(&self) -> &Generator {
        &self.generator
    }

    /// Access the F0 downsampling conv (diagnostic parity testing).
    #[must_use]
    pub fn f0_conv(&self) -> &Conv1d {
        &self.f0_conv
    }

    /// Access the energy downsampling conv (diagnostic parity testing).
    #[must_use]
    pub fn n_conv(&self) -> &Conv1d {
        &self.n_conv
    }

    /// Access the encode block (diagnostic parity testing).
    #[must_use]
    pub fn encode_block(&self) -> &Stage1ResBlk {
        &self.encode
    }

    /// Access the compressed skip conv (diagnostic parity testing).
    #[must_use]
    pub fn asr_res_conv(&self) -> &Conv1d {
        &self.asr_res
    }

    /// Access the decode blocks (diagnostic parity testing).
    #[must_use]
    pub fn decode_blocks(&self) -> &[Stage1ResBlk] {
        &self.decode
    }

    /// Decode features + F0 + energy → (magnitude, phase).
    ///
    /// `asr`: `[B, 512, T]` — length-regulated TextEncoder features.
    /// `f0_curve`: `[B, 1, 2T]` — F0 at 2× resolution.
    /// `n_curve`: `[B, 1, 2T]` — energy at 2× resolution.
    /// `style`: `[B, 128]` — decoder style vector.
    /// `har_source`: `[B, 2*n_bins, T_full]` — harmonic source for Generator.
    ///
    /// Returns `(magnitude [B, n_bins, T_out], phase [B, n_bins, T_out])`.
    pub fn forward(
        &self,
        asr: &DynTensor,
        f0_curve: &DynTensor,
        n_curve: &DynTensor,
        style: &DynTensor,
        har_source: &DynTensor,
    ) -> std::result::Result<(DynTensor, DynTensor), KokoroError> {
        // Downsample F0/N from 2T → T to match asr time resolution.
        let f0 = self.f0_conv.forward(f0_curve)?; // [B, 1, T]
        let n = self.n_conv.forward(n_curve)?; // [B, 1, T]

        // Encode: cat([asr, F0, N]) → [B, 514, T] → Stage1ResBlk → [B, 1024, T]
        let encode_input = DynTensor::cat(&[asr, &f0, &n], 1)?; // [B, 514, T]
        let mut x = self.encode.forward(&encode_input, style)?;
        check_tensor_finite(&x, "stage1_encode")?;

        // Compressed skip connection: asr [B, 512, T] → [B, 64, T]
        let asr_res = self.asr_res.forward(asr)?; // [B, 64, T]

        // Decode blocks 0-2: inject skip [x, asr_res, F0, N] → [B, 1090, T]
        for block in &self.decode[..3] {
            let skip_input = DynTensor::cat(&[&x, &asr_res, &f0, &n], 1)?; // [B, 1090, T]
            x = block.forward(&skip_input, style)?;
            check_tensor_finite(&x, "stage1_decode_loop")?;
        }

        // Decode block 3: inject skip + upsample 2×
        let skip_input = DynTensor::cat(&[&x, &asr_res, &f0, &n], 1)?; // [B, 1090, T]
        let x = self.decode[3].forward(&skip_input, style)?; // [B, 512, 2T]
        check_tensor_finite(&x, "stage1_decode_3")?;

        // Stage 2: Generator produces magnitude + phase
        self.generator.forward(&x, style, har_source)
    }
}

#[cfg(test)]
#[path = "kokoro_full_decoder_tests.rs"]
mod tests;
