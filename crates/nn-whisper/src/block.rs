// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Residual attention block for Whisper encoder/decoder.
//!
//! Each block contains:
//! - LayerNorm + MultiHeadAttention + residual (self-attention)
//! - [Decoder only] LayerNorm + MultiHeadAttention + residual (cross-attention)
//! - LayerNorm + FFN (Linear→GELU→Linear) + residual

use crate::attention::MultiHeadAttention;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{LayerNorm, Linear, Module};
use nn_core::{Result, VarBuilder};

/// Residual attention block.
///
/// Encoder blocks have `cross_attn = None`.
/// Decoder blocks have both self-attention and cross-attention.
pub struct ResidualAttentionBlock {
    /// Self-attention layer norm.
    self_attn_ln: LayerNorm,
    /// Self-attention.
    self_attn: MultiHeadAttention,
    /// Cross-attention layer norm (decoder only).
    cross_attn_ln: Option<LayerNorm>,
    /// Cross-attention (decoder only).
    cross_attn: Option<MultiHeadAttention>,
    /// FFN layer norm.
    final_ln: LayerNorm,
    /// FFN first linear: d_model -> ffn_dim.
    fc1: Linear,
    /// FFN second linear: ffn_dim -> d_model.
    fc2: Linear,
}

impl ResidualAttentionBlock {
    /// Load an encoder block (self-attention only).
    pub fn load_encoder(
        vb: impl AsRef<VarBuilder>,
        n_heads: usize,
        d_model: usize,
        ffn_dim: usize,
    ) -> Result<Self> {
        Self::load_inner(vb.as_ref(), n_heads, d_model, ffn_dim, false)
    }

    /// Load a decoder block (self-attention + cross-attention).
    pub fn load_decoder(
        vb: impl AsRef<VarBuilder>,
        n_heads: usize,
        d_model: usize,
        ffn_dim: usize,
    ) -> Result<Self> {
        Self::load_inner(vb.as_ref(), n_heads, d_model, ffn_dim, true)
    }

    fn load_inner(
        vb: impl AsRef<VarBuilder>,
        n_heads: usize,
        d_model: usize,
        ffn_dim: usize,
        has_cross_attn: bool,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let self_attn_ln = load_layer_norm(vb, "self_attn_layer_norm", d_model)?;
        let self_attn = MultiHeadAttention::load(vb.pp("self_attn"), n_heads, d_model)?;

        let (cross_attn_ln, cross_attn) = if has_cross_attn {
            let ln = load_layer_norm(vb, "encoder_attn_layer_norm", d_model)?;
            let attn = MultiHeadAttention::load(vb.pp("encoder_attn"), n_heads, d_model)?;
            (Some(ln), Some(attn))
        } else {
            (None, None)
        };

        let final_ln = load_layer_norm(vb, "final_layer_norm", d_model)?;

        let fc1_w = vb.get(&[ffn_dim, d_model], "fc1.weight")?;
        let fc1_b = vb.get(&[ffn_dim], "fc1.bias")?;
        let fc1 = Linear::new(fc1_w, Some(fc1_b))?;

        let fc2_w = vb.get(&[d_model, ffn_dim], "fc2.weight")?;
        let fc2_b = vb.get(&[d_model], "fc2.bias")?;
        let fc2 = Linear::new(fc2_w, Some(fc2_b))?;

        Ok(Self {
            self_attn_ln,
            self_attn,
            cross_attn_ln,
            cross_attn,
            final_ln,
            fc1,
            fc2,
        })
    }

    /// Forward pass for encoder blocks (self-attention only).
    pub fn forward_encoder(&mut self, x: &DynTensor) -> Result<DynTensor> {
        // Self-attention + residual.
        let residual = x.clone();
        let h = self.self_attn_ln.forward(x)?;
        let h = self.self_attn.forward(&h, None, None, false)?;
        let x = residual.add(&h)?;

        // FFN + residual.
        let residual = x.clone();
        let h = self.final_ln.forward(&x)?;
        let h = self.fc1.forward(&h)?;
        let h = h.gelu_erf()?;
        let h = self.fc2.forward(&h)?;
        residual.add(&h)
    }

    /// Cache-free encoder forward for tracing.
    ///
    /// Equivalent to `forward_encoder` but uses `forward_self_attn_no_cache`
    /// to avoid KV cache ops that the trace compiler cannot handle.
    pub fn forward_encoder_no_cache(&self, x: &DynTensor) -> Result<DynTensor> {
        let residual = x.clone();
        let h = self.self_attn_ln.forward(x)?;
        let h = self.self_attn.forward_self_attn_no_cache(&h)?;
        let x = residual.add(&h)?;

        let residual = x.clone();
        let h = self.final_ln.forward(&x)?;
        let h = self.fc1.forward(&h)?;
        let h = h.gelu_erf()?;
        let h = self.fc2.forward(&h)?;
        residual.add(&h)
    }

    /// Forward pass for decoder blocks (self-attention + cross-attention).
    pub fn forward_decoder(
        &mut self,
        x: &DynTensor,
        encoder_output: &DynTensor,
        mask: &DynTensor,
        flush_kv_cache: bool,
    ) -> Result<DynTensor> {
        // Self-attention + residual (with causal mask).
        let residual = x.clone();
        let h = self.self_attn_ln.forward(x)?;
        let h = self
            .self_attn
            .forward(&h, None, Some(mask), flush_kv_cache)?;
        let x = residual.add(&h)?;

        // Cross-attention + residual (with KV cache).
        let x = if let (Some(ref ln), Some(ref mut attn)) =
            (&self.cross_attn_ln, &mut self.cross_attn)
        {
            let residual = x.clone();
            let h = ln.forward(&x)?;
            let h = attn.forward(&h, Some(encoder_output), None, flush_kv_cache)?;
            residual.add(&h)?
        } else {
            x
        };

        // FFN + residual.
        let residual = x.clone();
        let h = self.final_ln.forward(&x)?;
        let h = self.fc1.forward(&h)?;
        let h = h.gelu_erf()?;
        let h = self.fc2.forward(&h)?;
        residual.add(&h)
    }

    /// Cache-free decoder forward for tracing.
    ///
    /// Equivalent to `forward_decoder` but avoids all KV cache ops. Uses
    /// `forward_self_attn_no_cache_masked` for causal self-attention and
    /// `forward_cross_attn_no_cache` for encoder cross-attention.
    pub fn forward_decoder_no_cache(
        &self,
        x: &DynTensor,
        encoder_output: &DynTensor,
        mask: &DynTensor,
    ) -> Result<DynTensor> {
        // Self-attention + residual (with causal mask, no cache).
        let residual = x.clone();
        let h = self.self_attn_ln.forward(x)?;
        let h = self
            .self_attn
            .forward_self_attn_no_cache_masked(&h, Some(mask))?;
        let x = residual.add(&h)?;

        // Cross-attention + residual (no cache).
        let x = if let (Some(ref ln), Some(ref attn)) = (&self.cross_attn_ln, &self.cross_attn) {
            let residual = x.clone();
            let h = ln.forward(&x)?;
            let h = attn.forward_cross_attn_no_cache(&h, encoder_output)?;
            residual.add(&h)?
        } else {
            x
        };

        // FFN + residual.
        let residual = x.clone();
        let h = self.final_ln.forward(&x)?;
        let h = self.fc1.forward(&h)?;
        let h = h.gelu_erf()?;
        let h = self.fc2.forward(&h)?;
        residual.add(&h)
    }

    /// Current self-attention KV cache sequence length (0 if empty).
    pub fn self_cache_len(&self) -> usize {
        self.self_attn.self_cache_len()
    }

    /// Reset all KV caches (self-attention and cross-attention).
    pub fn reset_cache(&mut self) {
        self.self_attn.reset_cache();
        if let Some(ref mut attn) = self.cross_attn {
            attn.reset_cache();
        }
    }
}

/// Load a LayerNorm from VarBuilder at a given key prefix.
fn load_layer_norm(vb: impl AsRef<VarBuilder>, name: &str, d_model: usize) -> Result<LayerNorm> {
    let vb = vb.as_ref();
    let w = vb.get(&[d_model], &format!("{name}.weight"))?;
    let b = vb.get(&[d_model], &format!("{name}.bias"))?;
    LayerNorm::new(w, b, 1e-5)
}
