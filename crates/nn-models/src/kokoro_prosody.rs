// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro TTS ProsodyPredictor: style-conditioned duration encoding.
//!
//! DurationEncoder (3× BiLSTM + AdaLayerNorm blocks) followed by a final
//! duration BiLSTM predicts duration logits and produces aligned features.
//! Matches dvoice v0.19 reference architecture.
//!
//! See `designs/archive/2026-03-16-kokoro-architecture-correction.md` Phase B2.

use nn_core::dyn_tensor::trace::{KokoroFusedOp, TraceOp};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{BiLstm, Linear, Module};
use nn_core::var_builder::VarBuilder;
use nn_core::Result;

use crate::kokoro_error::KokoroError;

// -- AdaLayerNorm -------------------------------------------------------------

/// Adaptive Layer Normalization: style-conditioned affine after LayerNorm.
///
/// `y = (1 + gamma(style)) * LayerNorm(x) + beta(style)`
pub struct AdaLayerNorm {
    norm: nn_core::layers::LayerNorm,
    style_proj: Linear,
    channels: usize,
    eps: f64,
}

impl AdaLayerNorm {
    /// Load from VarBuilder.
    ///
    /// Expects weights: `fc.weight`, `fc.bias`. Optionally `norm.weight`, `norm.bias`.
    /// Python Kokoro AdaLayerNorm uses parameter-free F.layer_norm (no learned
    /// norm params), so when `norm.weight`/`norm.bias` are absent, defaults to
    /// unit weight (ones) and zero bias.
    pub fn load(vb: impl AsRef<VarBuilder>, channels: usize, style_dim: usize) -> Result<Self> {
        let vb = vb.as_ref();
        let eps = 1e-5;
        let dev = vb.device();
        let norm = if vb.contains_tensor("norm.weight") {
            let w = vb.get(&[channels], "norm.weight")?;
            let b = vb.get(&[channels], "norm.bias")?;
            nn_core::layers::LayerNorm::new(w, b, eps)?
        } else {
            // Python uses parameter-free F.layer_norm: weight=1, bias=0.
            let w = DynTensor::ones(&[channels], nn_core::DType::F32, dev)?;
            let b = DynTensor::zeros(&[channels], nn_core::DType::F32, dev)?;
            nn_core::layers::LayerNorm::new(w, b, eps)?
        };
        let style_proj = {
            let w = vb.get(&[2 * channels, style_dim], "fc.weight")?;
            let b = vb.get(&[2 * channels], "fc.bias")?;
            Linear::new(w, Some(b))?
        };
        Ok(Self {
            norm,
            style_proj,
            channels,
            eps,
        })
    }

    /// Forward: `(1 + gamma(style)) * LayerNorm(x) + beta(style)`.
    ///
    /// `x`: `[B, T, channels]`, `style`: `[B, style_dim]`.
    pub fn forward(
        &self,
        x: &DynTensor,
        style: &DynTensor,
    ) -> std::result::Result<DynTensor, KokoroError> {
        let projected = self.style_proj.forward(style)?;
        let gamma = projected.narrow(1, 0, self.channels)?.unsqueeze(1)?;
        let beta = projected
            .narrow(1, self.channels, self.channels)?
            .unsqueeze(1)?;

        let eps = self.eps;
        let norm_weight_ref = self.norm.weight().to_weight_ref()?;
        let norm_bias_ref = self.norm.bias().to_weight_ref()?;
        Ok(nn_core::dyn_tensor::trace::traced_forward(
            &[x, &gamma, &beta],
            || {
                Ok(TraceOp::KokoroFused(KokoroFusedOp::AdaLayerNorm {
                    norm_weight: norm_weight_ref.clone(),
                    norm_bias: norm_bias_ref.clone(),
                    eps,
                }))
            },
            || {
                let normed = self.norm.forward(x)?;
                let ones = DynTensor::full(gamma.dims(), 1.0, nn_core::DType::F32, &x.device())?;
                let scale = ones.broadcast_add(&gamma)?;
                normed.broadcast_mul(&scale)?.broadcast_add(&beta)
            },
        )?)
    }
}

// -- DurationEncoder ----------------------------------------------------------

/// Duration encoder: 3× [cat(style) → BiLSTM → AdaLayerNorm → cat(style)] blocks.
///
/// Input: text features `[B, T, d_model]` + style `[B, style_dim]`.
/// Output: encoded `[B, T, d_model + style_dim]` (style concatenated after final block).
///
/// Python reference re-concatenates style after every AdaLayerNorm, including the
/// last one. The returned features include style so that downstream consumers
/// (duration BiLSTM, F0 predictor) receive `d_model + style_dim` directly.
///
/// Weight prefix: `duration.lstms.{i}.*`, `duration.norms.{i}.*`.
struct DurationEncoder {
    bilstms: Vec<BiLstm>,
    ada_norms: Vec<AdaLayerNorm>,
    duration_proj: Linear,
}

impl DurationEncoder {
    fn load(
        vb: impl AsRef<VarBuilder>,
        d_model: usize,
        style_dim: usize,
        n_layers: usize,
        max_dur: usize,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let hidden = d_model / 2;
        let bilstm_input = d_model + style_dim;
        let mut bilstms = Vec::with_capacity(n_layers);
        let mut ada_norms = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let lstm_vb = vb.pp(format!("duration.lstms.{i}"));
            let bilstm = BiLstm::load(&lstm_vb, bilstm_input, hidden)?;
            bilstms.push(bilstm);
            let norm_vb = vb.pp(format!("duration.norms.{i}"));
            let ada_norm = AdaLayerNorm::load(&norm_vb, d_model, style_dim)?;
            ada_norms.push(ada_norm);
        }
        let proj_vb = vb.pp("duration");
        let duration_proj = {
            let w = proj_vb.get(&[max_dur, d_model], "duration_proj.weight")?;
            let b = proj_vb.get(&[max_dur], "duration_proj.bias")?;
            Linear::new(w, Some(b))?
        };
        Ok(Self {
            bilstms,
            ada_norms,
            duration_proj,
        })
    }

    /// Forward: encode text features with style conditioning.
    ///
    /// `x`: `[B, T, d_model]`, `style`: `[B, style_dim]`.
    /// Returns encoded `[B, T, d_model + style_dim]`.
    ///
    /// Matches Python DurationEncoder: after each AdaLayerNorm (including the last),
    /// style is re-concatenated. The returned tensor includes style features so
    /// downstream consumers receive `d_model + style_dim` dimensions directly.
    fn forward(
        &self,
        x: &DynTensor,
        style: &DynTensor,
        style_dim: usize,
    ) -> std::result::Result<DynTensor, KokoroError> {
        let mut h = x.contiguous()?;
        for (bilstm, ada_norm) in self.bilstms.iter().zip(self.ada_norms.iter()) {
            let batch = h.dim(0)?;
            let seq_len = h.dim(1)?;
            // cat(x, style_expanded) -> [B, T, d_model+style_dim]
            let style_exp = style.unsqueeze(1)?.expand([batch, seq_len, style_dim])?;
            let cat_input = DynTensor::cat(&[&h, &style_exp], 2)?;
            // BiLSTM batch-first: [B, T, input] → [B, T, 2*hidden=d_model]
            let (bilstm_out, _, _) = bilstm.forward_seq_batch_first(&cat_input, None, None)?;
            h = ada_norm.forward(&bilstm_out, style)?;
        }
        // Re-concatenate style after the last AdaLayerNorm (matches Python reference).
        // h: [B, T, d_model] → cat(h, style_exp) → [B, T, d_model + style_dim]
        let batch = h.dim(0)?;
        let seq_len = h.dim(1)?;
        let style_exp = style.unsqueeze(1)?.expand([batch, seq_len, style_dim])?;
        let out = DynTensor::cat(&[&h, &style_exp], 2)?;
        Ok(out)
    }

    /// Project encoded features to duration logits.
    fn project(&self, x: &DynTensor) -> std::result::Result<DynTensor, KokoroError> {
        Ok(self.duration_proj.forward(x)?)
    }
}

// -- ProsodyPredictor ---------------------------------------------------------

/// Prosody predictor: predicts duration logits and produces aligned features.
///
/// Input: text features `[B, d_en, T]` + style embedding `[B, style_dim]`.
/// Output: `(duration_logits [B, T, max_dur], encoded_features [B, d_model+style_dim, T])`.
///
/// Architecture: DurationEncoder (3× BiLSTM+AdaLayerNorm blocks) followed by
/// a final duration BiLSTM and linear projection to duration bins.
pub struct ProsodyPredictor {
    duration_encoder: DurationEncoder,
    duration_bilstm: BiLstm,
    style_dim: usize,
}

impl ProsodyPredictor {
    /// Load from VarBuilder.
    ///
    /// `d_model`: feature dimension (e.g., 512).
    /// `style_dim`: style embedding dimension (e.g., 128).
    /// `n_layers`: number of DurationEncoder blocks (e.g., 3).
    /// `max_dur`: maximum duration bins (e.g., 50).
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        d_model: usize,
        style_dim: usize,
        n_layers: usize,
        max_dur: usize,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let duration_encoder = DurationEncoder::load(vb, d_model, style_dim, n_layers, max_dur)?;
        let hidden = d_model / 2;
        let bilstm_input = d_model + style_dim;
        let lstm_vb = vb.pp("lstm");
        let duration_bilstm = BiLstm::load(&lstm_vb, bilstm_input, hidden)?;
        Ok(Self {
            duration_encoder,
            duration_bilstm,
            style_dim,
        })
    }

    /// Forward: predict durations and produce aligned features.
    ///
    /// `x`: `[B, d_model, T]`, `style`: `[B, style_dim]`.
    /// Returns `(dur_logits [B, T, max_dur], features [B, d_model+style_dim, T])`.
    ///
    /// DurationEncoder returns `[B, T, d_model+style_dim]` (style already included).
    /// The duration BiLSTM takes this directly (input_size = d_model+style_dim).
    /// Features include style so F0EnergyPredictor receives full-width input.
    pub fn forward(
        &self,
        x: &DynTensor,
        style: &DynTensor,
    ) -> std::result::Result<(DynTensor, DynTensor), KokoroError> {
        // [B, d_model, T] -> [B, T, d_model]
        let x_t = x.transpose(1, 2)?;
        // DurationEncoder: 3× [cat(style) → BiLSTM → AdaLayerNorm → cat(style)]
        // Returns [B, T, d_model+style_dim] with style already included.
        let encoded = self.duration_encoder.forward(&x_t, style, self.style_dim)?;
        // Duration BiLSTM: encoded already includes style (d_model+style_dim=640).
        // Python: self.lstm(d) where d is text_encoder output [B, T, 640].
        let (dur_out, _, _) = self
            .duration_bilstm
            .forward_seq_batch_first(&encoded, None, None)?;
        let dur_logits = self.duration_encoder.project(&dur_out)?;
        // Return encoded features as [B, d_model+style_dim, T]
        let features = encoded.transpose(1, 2)?;
        Ok((dur_logits, features))
    }
}
