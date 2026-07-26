// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro TTS F0/energy prediction: AdainResBlk1d residual blocks + prediction heads.
//!
//! The F0 predictor takes aligned features from the duration predictor, processes
//! through a shared BiLSTM, then splits into parallel F0 and energy heads (each
//! 3 AdainResBlk1d blocks + Conv1d(k=1) projection). Output is at 2x phoneme resolution
//! (one block upsamples by 2).
//!
//! See `designs/archive/2026-03-16-kokoro-architecture-correction.md` and dvoice `prosody.rs` / `prosody_blocks.rs`.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{
    AdaIn, BiLstm, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig,
    InstanceNormPrecision, Linear, Module,
};
use nn_core::var_builder::VarBuilder;
use nn_core::Result;

use crate::kokoro_error::KokoroError;

// -- AdainResBlk1d ------------------------------------------------------------

/// Residual block with AdaIN normalization and optional 2x upsampling.
///
/// Architecture:
/// - Residual path: AdaIN(x, style) → LeakyReLU(0.2) → [ConvTranspose1d 2x if upsample]
///   → Conv1d(k=3, pad=1) → AdaIN → LeakyReLU(0.2) → Conv1d(k=3, pad=1)
/// - Shortcut path: [upsample_nearest_2x if upsample] → [Conv1d(k=1) if dim change]
/// - Output: (residual + shortcut) / sqrt(2)
///
/// Used in Kokoro TTS F0 and energy prediction heads.
pub struct AdainResBlk1d {
    n1: AdaIn,
    n2: AdaIn,
    c1: Conv1d,
    c2: Conv1d,
    skip_conv: Option<Conv1d>,
    pool: Option<ConvTranspose1d>,
    upsample: bool,
}

impl AdainResBlk1d {
    /// Load from VarBuilder.
    ///
    /// `dim_in`, `dim_out`: input/output channel dimensions.
    /// `style_dim`: style embedding dimension for AdaIN.
    /// `upsample`: if true, applies 2x upsampling (ConvTranspose1d on residual, nearest on shortcut).
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        dim_in: usize,
        dim_out: usize,
        style_dim: usize,
        upsample: bool,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        // AdaIN layers: project style to 2*dim affine params
        let n1_w = vb.get(&[2 * dim_in, style_dim], "n1.fc.weight")?;
        let n1_b = vb.get(&[2 * dim_in], "n1.fc.bias")?;
        // MatchPyTorchCpu: F32 accumulation matches PyTorch ATen CPU to prevent
        // compound drift over 12 F0/energy AdaINs (6 per head). (#2691)
        let n1 = AdaIn::new_with_precision(
            Linear::new(n1_w, Some(n1_b))?,
            1e-5,
            InstanceNormPrecision::MatchPyTorchCpu,
        )?;

        let n2_w = vb.get(&[2 * dim_out, style_dim], "n2.fc.weight")?;
        let n2_b = vb.get(&[2 * dim_out], "n2.fc.bias")?;
        let n2 = AdaIn::new_with_precision(
            Linear::new(n2_w, Some(n2_b))?,
            1e-5,
            InstanceNormPrecision::MatchPyTorchCpu,
        )?;

        // Conv1d layers (kernel=3, padding=1)
        let c1_w = vb.get(&[dim_out, dim_in, 3], "c1.weight")?;
        let c1_b = vb.get(&[dim_out], "c1.bias")?;
        let c1 = Conv1d::new(c1_w, Some(c1_b), Conv1dConfig::default().with_padding(1))?;

        let c2_w = vb.get(&[dim_out, dim_out, 3], "c2.weight")?;
        let c2_b = vb.get(&[dim_out], "c2.bias")?;
        let c2 = Conv1d::new(c2_w, Some(c2_b), Conv1dConfig::default().with_padding(1))?;

        // Skip connection: 1x1 conv if dim changes (Python uses bias=False)
        let skip_conv = if dim_in != dim_out {
            let sw = vb.get(&[dim_out, dim_in, 1], "skip.weight")?;
            let sb = if vb.contains_tensor("skip.bias") {
                Some(vb.get(&[dim_out], "skip.bias")?)
            } else {
                None
            };
            Some(Conv1d::new(sw, sb, Conv1dConfig::default())?)
        } else {
            None
        };

        // Upsample pool: depthwise ConvTranspose1d (groups=dim_in, stride=2, kernel=3, pad=1)
        let pool = if upsample {
            let pw = vb.get(&[dim_in, 1, 3], "pool.weight")?;
            let pb = vb.get(&[dim_in], "pool.bias")?;
            Some(ConvTranspose1d::new(
                pw,
                Some(pb),
                ConvTranspose1dConfig::default()
                    .with_stride(2)
                    .with_padding(1)
                    .with_output_padding(1)
                    .with_groups(dim_in),
            )?)
        } else {
            None
        };

        Ok(Self {
            n1,
            n2,
            c1,
            c2,
            skip_conv,
            pool,
            upsample,
        })
    }

    /// Forward: residual block with optional 2x upsampling.
    ///
    /// `x`: `[B, C_in, T]`, `style`: `[B, style_dim]`.
    /// Returns `[B, C_out, T]` (or `[B, C_out, 2T]` if upsample).
    ///
    /// Always uses the decomposed path so that `AdaIn::forward_leaky_relu`
    /// traces as `TraceOp::AdainLeakyRelu` → NativeOp (1 Metal dispatch).
    /// The former `FusedAdainResBlock` path produced ~45 Metal dispatches
    /// per block due to `expand_norm_ops` decomposition. See #2590.
    pub fn forward(
        &self,
        x: &DynTensor,
        style: &DynTensor,
    ) -> std::result::Result<DynTensor, KokoroError> {
        Ok(self.forward_impl(x, style)?)
    }

    /// Forward implementation: each sub-op traces individually, allowing
    /// AdainLeakyRelu to compile as a NativeOp (single Metal dispatch).
    fn forward_impl(&self, x: &DynTensor, style: &DynTensor) -> Result<DynTensor> {
        let mut h = self.n1.forward_leaky_relu(x, style, 0.2)?;
        if let Some(ref pool) = self.pool {
            h = pool.forward(&h)?;
        }
        h = self.c1.forward(&h)?;
        h = self.n2.forward_leaky_relu(&h, style, 0.2)?;
        h = self.c2.forward(&h)?;

        // Shortcut path
        let mut shortcut = x.clone();
        if self.upsample {
            shortcut = shortcut.upsample_nearest_1d(2)?;
        }
        if let Some(ref skip) = self.skip_conv {
            shortcut = skip.forward(&shortcut)?;
        }

        // (residual + shortcut) / sqrt(2)
        let sum = h.add(&shortcut)?;
        let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
        sum.mul_scalar(inv_sqrt2)
    }
}

// -- F0/Energy Predictor ------------------------------------------------------

/// F0 and energy prediction head for Kokoro TTS.
///
/// Takes aligned features from the duration predictor and produces F0 (fundamental
/// frequency) and energy (noise) signals at 2x phoneme resolution.
///
/// Architecture per head:
/// - shared_bilstm: BiLSTM on aligned features → [B, 2*hidden, T_mel]
/// - 3 AdainResBlk1d blocks (block 1 upsamples 2x)
/// - Conv1d(k=1) projection → [B, 1, 2*T_mel]  (#3512: eliminates 4 transpose dispatches)
pub struct F0EnergyPredictor {
    shared_bilstm: BiLstm,
    f0_blocks: Vec<AdainResBlk1d>,
    f0_proj: Conv1d,
    energy_blocks: Vec<AdainResBlk1d>,
    energy_proj: Conv1d,
}

impl F0EnergyPredictor {
    /// Load from VarBuilder under the predictor prefix.
    ///
    /// `d_model`: feature dimension from duration predictor (e.g., 512).
    /// `style_dim`: style embedding dimension (e.g., 128).
    /// `bilstm_hidden`: BiLSTM hidden size per direction (e.g., 256, output = 512).
    ///
    /// The shared BiLSTM input dimension is `d_model + style_dim` (640 for Kokoro defaults)
    /// because the DurationEncoder output already includes style (see kokoro_prosody.rs).
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        d_model: usize,
        style_dim: usize,
        bilstm_hidden: usize,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let bilstm_out = 2 * bilstm_hidden;
        let bilstm_input_dim = d_model + style_dim;

        // Shared BiLSTM: input is d_model + style_dim (style included from DurationEncoder).
        // Uses BiLstm::load which supports both PyTorch-native (weight_ih_l0) and
        // decomposed (forward.weight_ih.weight) naming conventions. Part of #2691.
        let shared_bilstm = BiLstm::load(vb.pp("shared"), bilstm_input_dim, bilstm_hidden)?;

        // F0 blocks: 3 AdainResBlk1d — block 0: 512→512, block 1: 512→256 (upsample), block 2: 256→256
        let f0_vb = vb.pp("F0");
        let f0_b0 = AdainResBlk1d::load(f0_vb.pp("0"), bilstm_out, bilstm_out, style_dim, false)?;
        let f0_b1 = AdainResBlk1d::load(f0_vb.pp("1"), bilstm_out, bilstm_hidden, style_dim, true)?;
        let f0_b2 = AdainResBlk1d::load(
            f0_vb.pp("2"),
            bilstm_hidden,
            bilstm_hidden,
            style_dim,
            false,
        )?;

        // F0 projection: Conv1d(k=1) — equivalent to Linear but operates on [B, C, T]
        // directly, eliminating transpose→Linear→transpose. (#3512)
        // Safetensors weight is [1, hidden] (Linear shape); reshape to [1, hidden, 1] for Conv1d.
        let f0_proj_w = vb.get(&[1, bilstm_hidden], "F0_proj.weight")?;
        let f0_proj_w = f0_proj_w.reshape([1, bilstm_hidden, 1])?;
        let f0_proj_b = vb.get(&[1], "F0_proj.bias")?;
        let f0_proj = Conv1d::new(f0_proj_w, Some(f0_proj_b), Conv1dConfig::default())?;

        // Energy (N) blocks: same architecture as F0
        let n_vb = vb.pp("N");
        let n_b0 = AdainResBlk1d::load(n_vb.pp("0"), bilstm_out, bilstm_out, style_dim, false)?;
        let n_b1 = AdainResBlk1d::load(n_vb.pp("1"), bilstm_out, bilstm_hidden, style_dim, true)?;
        let n_b2 =
            AdainResBlk1d::load(n_vb.pp("2"), bilstm_hidden, bilstm_hidden, style_dim, false)?;

        // Energy projection: Conv1d(k=1) — same optimization as F0. (#3512)
        let n_proj_w = vb.get(&[1, bilstm_hidden], "N_proj.weight")?;
        let n_proj_w = n_proj_w.reshape([1, bilstm_hidden, 1])?;
        let n_proj_b = vb.get(&[1], "N_proj.bias")?;
        let energy_proj = Conv1d::new(n_proj_w, Some(n_proj_b), Conv1dConfig::default())?;

        Ok(Self {
            shared_bilstm,
            f0_blocks: vec![f0_b0, f0_b1, f0_b2],
            f0_proj,
            energy_blocks: vec![n_b0, n_b1, n_b2],
            energy_proj,
        })
    }

    /// Predict F0 and energy from aligned features.
    ///
    /// `aligned`: `[B, d_model+style_dim, T_mel]` — aligned features from length_regulate
    ///   (style already included from DurationEncoder output).
    /// `style`: `[B, style_dim]` — prosody style embedding (used by AdainResBlk1d blocks).
    ///
    /// Returns `(f0 [B, 1, 2*T_mel], energy [B, 1, 2*T_mel])`.
    pub fn forward(
        &self,
        aligned: &DynTensor,
        style: &DynTensor,
    ) -> std::result::Result<(DynTensor, DynTensor), KokoroError> {
        // aligned already includes style (d_model+style_dim=640) from DurationEncoder.
        // Python: F0Ntrain(self, x, s): x, _ = self.shared(x.transpose(-1, -2))
        // where x is `en` = [B, 640, T_mel], passed directly without additional style cat.
        // [B, D+S, T] → [T, B, D+S] for BiLSTM (single permute, not double-transpose)
        let cat_t = aligned.permute([2, 0, 1])?;
        let (bilstm_out, _, _) = self.shared_bilstm.forward_seq(&cat_t, None, None)?;
        // [T, B, 2*H] → [B, 2*H, T] (single permute, not double-transpose)
        let shared = bilstm_out.permute([1, 2, 0])?;

        // F0 head: 3 AdainResBlk1d blocks → project
        let mut f0 = shared.clone();
        for block in &self.f0_blocks {
            f0 = block.forward(&f0, style)?;
        }
        // Conv1d(k=1) on [B, H, 2T] → [B, 1, 2T] directly, no transposes. (#3512)
        let f0_out = self.f0_proj.forward(&f0)?;

        // Energy head: same architecture
        let mut energy = shared;
        for block in &self.energy_blocks {
            energy = block.forward(&energy, style)?;
        }
        // Conv1d(k=1) on [B, H, 2T] → [B, 1, 2T] directly, no transposes. (#3512)
        let energy_out = self.energy_proj.forward(&energy)?;

        Ok((f0_out, energy_out))
    }
}

#[cfg(test)]
#[path = "kokoro_f0_tests.rs"]
mod tests;
