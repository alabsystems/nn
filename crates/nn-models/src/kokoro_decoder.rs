// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro TTS Generator (ISTFTNet): converts aligned features + F0 source to magnitude + phase.
//!
//! Architecture: conv_pre -> upsampling blocks (LeakyReLU -> ConvTranspose1d -> noise injection
//! -> ResBlocks averaged) -> LeakyReLU -> conv_post -> split into exp(mag) and sin(phase).
//!
//! See `designs/archive/2026-03-16-kokoro-architecture-correction.md` Direction 5.

use crate::kokoro_error::{
    check_tensor_finite, validate_generator_config, KokoroError, LOG_MAG_CLAMP_MAX,
};
use crate::kokoro_tts::KokoroConfig;
use nn_core::dyn_tensor::trace::is_tracing;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module};
use nn_core::var_builder::VarBuilder;
use nn_core::Result;

pub use crate::kokoro_resblock::ResBlock;

/// Kokoro Generator (ISTFTNet): upsampling decoder producing magnitude + phase for iSTFT.
///
/// Architecture:
/// - conv_pre: Conv1d input projection
/// - Per stage: LeakyReLU(0.1) -> ConvTranspose1d -> noise injection -> ResBlocks (averaged)
/// - LeakyReLU(0.01) -> conv_post -> split into magnitude (exp) and phase (sin)
pub struct Generator {
    input_conv: Conv1d,
    ups: Vec<ConvTranspose1d>,
    res_blocks: Vec<ResBlock>,
    noise_convs: Vec<Conv1d>,
    noise_res: Vec<ResBlock>,
    output_conv: Conv1d,
    num_ups: usize,
}

/// Per-stage components loaded by [`Generator::load_upsample_stage`].
struct UpsampleStage {
    up: ConvTranspose1d,
    noise_conv: Conv1d,
    noise_rb: ResBlock,
    res_blocks: Vec<ResBlock>,
}

impl Generator {
    /// Load one upsample stage: ConvTranspose1d + noise Conv1d + noise ResBlock + ResBlocks.
    fn load_upsample_stage(
        vb: impl AsRef<VarBuilder>,
        config: &KokoroConfig,
        i: usize,
        ch: usize,
        n_bins: usize,
    ) -> Result<UpsampleStage> {
        let vb = vb.as_ref();
        let next_ch = ch / 2;
        let stride = config.upsample_rates[i];
        let k = config.upsample_kernel_sizes[i];
        let padding = (k - stride) / 2;

        let up_w = vb.get(&[ch, next_ch, k], &format!("ups.{i}.weight"))?;
        let up_b = vb.get(&[next_ch], &format!("ups.{i}.bias"))?;
        let up = ConvTranspose1d::new(
            up_w,
            Some(up_b),
            ConvTranspose1dConfig::default()
                .with_padding(padding)
                .with_stride(stride),
        )?;

        let cumulative_stride: usize = config.upsample_rates[i + 1..].iter().product();
        let is_last = i == config.upsample_rates.len() - 1;
        let noise_kernel = if is_last { 1 } else { cumulative_stride * 2 };
        let noise_stride = cumulative_stride.max(1);
        // PyTorch: padding=(stride_f0+1)//2 — ceiling division to match ONNX export.
        let noise_padding = if is_last {
            0
        } else {
            cumulative_stride.div_ceil(2)
        };

        let nc_w = vb.get(
            &[next_ch, 2 * n_bins, noise_kernel],
            &format!("noise_convs.{i}.weight"),
        )?;
        let nc_b = vb.get(&[next_ch], &format!("noise_convs.{i}.bias"))?;
        let noise_conv = Conv1d::new(
            nc_w,
            Some(nc_b),
            Conv1dConfig::default()
                .with_stride(noise_stride)
                .with_padding(noise_padding),
        )?;

        // PyTorch reference (istftnet.py): noise_res uses kernel_size=7 for non-last
        // stages, kernel_size=11 for the last stage. NOT from resblock_kernel_sizes config.
        // Dilations are also hardcoded [1,3,5] in PyTorch (not from config).
        let noise_kernel_size = if is_last { 11 } else { 7 };
        let noise_dilations: &[usize] = &[1, 3, 5];
        let noise_rb = ResBlock::load(
            vb.pp(format!("noise_res.{i}")),
            next_ch,
            noise_kernel_size,
            noise_dilations,
            config.style_dim,
        )?;

        let num_rk = config.resblock_kernel_sizes.len();
        let mut res_blocks = Vec::with_capacity(num_rk);
        for (j, rk) in config.resblock_kernel_sizes.iter().enumerate() {
            let rb = ResBlock::load(
                vb.pp(format!("resblocks.{}", i * num_rk + j)),
                next_ch,
                *rk,
                &config.resblock_dilations[j],
                config.style_dim,
            )?;
            res_blocks.push(rb);
        }

        Ok(UpsampleStage {
            up,
            noise_conv,
            noise_rb,
            res_blocks,
        })
    }

    /// Load Generator from VarBuilder and [`KokoroConfig`].
    ///
    /// All decoder hyperparameters (upsample rates, ResBlock config, channels,
    /// style dim, n_fft) are read from the config struct.
    pub fn load(vb: impl AsRef<VarBuilder>, config: &KokoroConfig) -> Result<Self> {
        let vb = vb.as_ref();
        // Validate config vector lengths before indexing to prevent panics.
        validate_generator_config(config)?;

        let initial_channels = config.gen_initial_channels;
        let n_fft = config.n_fft;
        let num_ups = config.upsample_rates.len();
        let n_bins = n_fft / 2 + 1;

        let in_w = vb.get(&[initial_channels, initial_channels, 7], "conv_pre.weight")?;
        let in_b = vb.get(&[initial_channels], "conv_pre.bias")?;
        let input_conv = Conv1d::new(in_w, Some(in_b), Conv1dConfig::default().with_padding(3))?;

        let mut ups = Vec::with_capacity(num_ups);
        let mut res_blocks = Vec::new();
        let mut noise_convs = Vec::with_capacity(num_ups);
        let mut noise_res = Vec::with_capacity(num_ups);
        let mut ch = initial_channels;

        for i in 0..num_ups {
            let stage = Self::load_upsample_stage(vb, config, i, ch, n_bins)?;
            ups.push(stage.up);
            noise_convs.push(stage.noise_conv);
            noise_res.push(stage.noise_rb);
            res_blocks.extend(stage.res_blocks);
            ch /= 2;
        }

        let out_w = vb.get(&[2 * n_bins, ch, 7], "conv_post.weight")?;
        let out_b = vb.get(&[2 * n_bins], "conv_post.bias")?;
        let output_conv = Conv1d::new(out_w, Some(out_b), Conv1dConfig::default().with_padding(3))?;

        Ok(Self {
            input_conv,
            ups,
            res_blocks,
            noise_convs,
            noise_res,
            output_conv,
            num_ups,
        })
    }

    /// Number of upsampling stages.
    #[must_use]
    pub fn num_stages(&self) -> usize {
        self.num_ups
    }

    /// Number of ResBlocks per upsampling stage.
    fn num_resblocks_per_stage(&self) -> usize {
        if self.num_ups == 0 {
            return 0;
        }
        self.res_blocks.len() / self.num_ups
    }

    /// Access ResBlocks (for LoRA wrapping / weight extraction).
    #[must_use]
    pub fn res_blocks(&self) -> &[ResBlock] {
        &self.res_blocks
    }

    /// Access the input conv layer (for LoRA wrapping / weight extraction).
    #[must_use]
    pub fn input_conv(&self) -> &Conv1d {
        &self.input_conv
    }

    /// Access the output conv layer (for LoRA wrapping / weight extraction).
    #[must_use]
    pub fn output_conv(&self) -> &Conv1d {
        &self.output_conv
    }

    /// Sub-block 0: conv_pre only.
    ///
    /// `x`: `[B, channels, T]` -> `[B, channels, T]`.
    /// Single Conv1d projection — IBP tractable.
    pub fn forward_conv_pre(&self, x: &DynTensor) -> std::result::Result<DynTensor, KokoroError> {
        Ok(self.input_conv.forward(&x.contiguous()?)?)
    }

    /// Sub-block i+1: one upsample stage (LeakyReLU -> ConvTranspose1d -> noise -> ResBlocks).
    ///
    /// `h`: `[B, ch, T_in]`, `style`: `[B, style_dim]`, `har_source`: `[B, 2*n_bins, T_full]`.
    /// Returns `[B, ch/2, T_out]` where `T_out = T_in * upsample_rate`.
    pub fn forward_upsample_stage(
        &self,
        stage: usize,
        h: &DynTensor,
        style: &DynTensor,
        har_source: &DynTensor,
    ) -> std::result::Result<DynTensor, KokoroError> {
        let rb_per_stage = self.num_resblocks_per_stage();
        let mut h = h.leaky_relu(0.1)?;
        h = self.ups[stage].forward(&h)?;

        if stage == self.num_ups - 1 {
            h = h.reflection_pad1d(1, 0)?;
        }

        // Noise injection: run noise_res on FULL conv output BEFORE trimming.
        // InstanceNorm inside noise_res computes stats over the full temporal extent.
        // Trimming first changes the normalization statistics (dvoice processes
        // the full signal through noise_res, then truncates at addition).
        let noise = self.noise_convs[stage].forward(har_source)?;
        let noise_out = self.noise_res[stage].forward(&noise, style)?;
        let t_h = h.dim(2)?;
        let t_noise = noise_out.dim(2)?;
        let noise_trimmed = if t_noise > t_h {
            noise_out.narrow(2, 0, t_h)?
        } else if t_noise < t_h {
            noise_out.pad1d(0, t_h - t_noise)?
        } else {
            noise_out
        };
        h = h.add(&noise_trimmed)?;

        let rb_start = stage * rb_per_stage;
        let rb_end = rb_start + rb_per_stage;
        if rb_per_stage > 0 {
            let mut sum = self.res_blocks[rb_start].forward(&h, style)?;
            for rb in &self.res_blocks[rb_start + 1..rb_end] {
                sum = sum.add(&rb.forward(&h, style)?)?;
            }
            h = sum.mul_scalar(1.0 / rb_per_stage as f64)?;
        }
        Ok(h)
    }

    /// Sub-block N+1a: output conv_post only (LeakyReLU -> conv_post).
    ///
    /// Returns `(log_magnitude, phase_raw)` each `[B, n_bins, T]` — the raw
    /// conv_post output BEFORE clamp/exp/sin. Used to verify junction contracts
    /// J3_MAGNITUDE and J3B_PHASE against pre-activation values (#2597).
    pub fn forward_output_conv_post(
        &self,
        h: &DynTensor,
    ) -> std::result::Result<(DynTensor, DynTensor), KokoroError> {
        let h = h.leaky_relu(0.01)?;
        let out = self.output_conv.forward(&h)?;
        let n_out = out.dim(1)?;
        let n_bins = n_out / 2;
        let log_mag = out.narrow(1, 0, n_bins)?;
        let phase_raw = out.narrow(1, n_bins, n_bins)?;
        Ok((log_mag, phase_raw))
    }

    /// Sub-block N+1: output stage (LeakyReLU -> conv_post -> clamp -> exp/sin).
    ///
    /// `h`: `[B, ch_final, T]` -> `(magnitude, phase)` each `[B, n_bins, T]`.
    /// Clamp + exp bounds magnitude; sin bounds phase to [-1, 1].
    pub fn forward_output_stage(
        &self,
        h: &DynTensor,
    ) -> std::result::Result<(DynTensor, DynTensor), KokoroError> {
        let h = h.leaky_relu(0.01)?;
        let out = self.output_conv.forward(&h)?;
        let n_out = out.dim(1)?;
        let n_bins = n_out / 2;
        let log_mag = out.narrow(1, 0, n_bins)?;
        let phase_raw = out.narrow(1, n_bins, n_bins)?;
        // Use DynTensor ops when on GPU OR when tracing is active. The ndarray
        // mapv path is invisible to the trace system — resulting tensors have
        // trace_node_id: None, causing the compiled model to miss clamp/exp/sin
        // and output raw log_mag instead of exp(log_mag). (#2683)
        let (magnitude, phase) = if log_mag.device().is_gpu() || is_tracing() {
            let log_mag_clamped = log_mag.clamp(-LOG_MAG_CLAMP_MAX, LOG_MAG_CLAMP_MAX)?;
            (log_mag_clamped.exp()?, phase_raw.sin()?)
        } else {
            // CPU F64 for clamp/exp/sin: prevents InstanceNorm drift amplification
            // through exp(). Tensor is tiny ([1, n_bins, T]), negligible cost. (#2689)
            let mag_f32 = log_mag.to_f32_array()?;
            let mag = DynTensor::from_cpu_f32(mag_f32.mapv(|v| {
                let v64 = f64::from(v).clamp(-LOG_MAG_CLAMP_MAX, LOG_MAG_CLAMP_MAX);
                v64.exp() as f32
            }))?;
            let phase_f32 = phase_raw.to_f32_array()?;
            let ph = DynTensor::from_cpu_f32(phase_f32.mapv(|v| f64::from(v).sin() as f32))?;
            (mag, ph)
        };
        check_tensor_finite(&magnitude, "decoder_magnitude")?;
        check_tensor_finite(&phase, "decoder_phase")?;
        Ok((magnitude, phase))
    }

    /// Forward: features + style + harmonic source -> (magnitude, phase).
    ///
    /// Composed from sub-block methods: `forward_conv_pre` -> per-stage
    /// `forward_upsample_stage` -> `forward_output_stage`. Sub-block methods
    /// enable segmented verification (#2597).
    ///
    /// `x`: `[B, channels, T]`, `style`: `[B, style_dim]`, `har_source`: `[B, 2*n_bins, T_full]`.
    /// Returns `(magnitude, phase)` each `[B, n_bins, T_out]`.
    pub fn forward(
        &self,
        x: &DynTensor,
        style: &DynTensor,
        har_source: &DynTensor,
    ) -> std::result::Result<(DynTensor, DynTensor), KokoroError> {
        let mut h = self.forward_conv_pre(x)?;
        check_tensor_finite(&h, "generator_conv_pre")?;
        for i in 0..self.num_ups {
            h = self.forward_upsample_stage(i, &h, style, har_source)?;
            check_tensor_finite(
                &h,
                match i {
                    0 => "generator_upsample_0",
                    1 => "generator_upsample_1",
                    _ => "generator_upsample_n",
                },
            )?;
        }
        self.forward_output_stage(&h)
    }
}

#[cfg(test)]
#[path = "kokoro_decoder_tests.rs"]
mod tests;

#[cfg(kani)]
#[path = "kokoro_decoder_kani_tests.rs"]
mod kani_proofs;
