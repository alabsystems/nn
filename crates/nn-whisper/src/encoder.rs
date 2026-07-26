// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Whisper audio encoder.
//!
//! Conv1d stem (mel→d_model) + sinusoidal positional embedding +
//! N residual attention blocks + final LayerNorm.

use crate::block::ResidualAttentionBlock;
use crate::config::WhisperConfig;
use crate::positional::sinusoidal_embedding;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{
    check_output_finite, with_nan_check_policy, Conv1d, Conv1dConfig, LayerNorm, Module,
    NanCheckPolicy,
};
use nn_core::{Result, VarBuilder};

/// Whisper audio encoder.
pub struct AudioEncoder {
    /// Conv1d(num_mel_bins → d_model, k=3, s=1, pad=1) + GELU.
    conv1: Conv1d,
    /// Conv1d(d_model → d_model, k=3, s=2, pad=1) + GELU (stride-2 downsample).
    conv2: Conv1d,
    /// Sinusoidal positional embedding: `[max_source_positions, d_model]`.
    positional_embedding: DynTensor,
    /// Encoder transformer blocks.
    blocks: Vec<ResidualAttentionBlock>,
    /// Final layer norm.
    ln_post: LayerNorm,
}

impl AudioEncoder {
    /// Load encoder weights from VarBuilder.
    ///
    /// VarBuilder should be scoped to `model.encoder` prefix.
    pub fn load(vb: impl AsRef<VarBuilder>, config: &WhisperConfig) -> Result<Self> {
        let vb = vb.as_ref();
        let d = config.d_model;

        // Conv1d stem.
        let c1_w = vb.get(&[d, config.num_mel_bins, 3], "conv1.weight")?;
        let c1_b = vb.get(&[d], "conv1.bias")?;
        let conv1 = Conv1d::new(c1_w, Some(c1_b), Conv1dConfig::default().with_padding(1))?;

        let c2_w = vb.get(&[d, d, 3], "conv2.weight")?;
        let c2_b = vb.get(&[d], "conv2.bias")?;
        let conv2 = Conv1d::new(
            c2_w,
            Some(c2_b),
            Conv1dConfig::default().with_padding(1).with_stride(2),
        )?;

        // Sinusoidal positional embedding [max_source_positions, d_model].
        // Use VarBuilder's dtype so embedding matches model weight dtype
        // (e.g., BF16) for GPU binary ops (#1710).
        let positional_embedding =
            sinusoidal_embedding(config.max_source_positions, d, vb.dtype(), vb.device())?;

        // Transformer blocks.
        let mut blocks = Vec::with_capacity(config.encoder_layers);
        for i in 0..config.encoder_layers {
            let block_vb = vb.pp(format!("layers.{i}"));
            let block = ResidualAttentionBlock::load_encoder(
                &block_vb,
                config.encoder_attention_heads,
                d,
                config.encoder_ffn_dim,
            )?;
            blocks.push(block);
        }

        // Final layer norm.
        let ln_w = vb.get(&[d], "layer_norm.weight")?;
        let ln_b = vb.get(&[d], "layer_norm.bias")?;
        let ln_post = LayerNorm::new(ln_w, ln_b, 1e-5)?;

        Ok(Self {
            conv1,
            conv2,
            positional_embedding,
            blocks,
            ln_post,
        })
    }

    /// Encode mel spectrogram to audio features.
    ///
    /// Input: `[batch, num_mel_bins, n_frames]` (e.g., `[1, 128, 3000]`).
    /// Output: `[batch, seq_len, d_model]` (e.g., `[1, 1500, 1280]`).
    pub fn forward(&mut self, mel: &DynTensor) -> Result<DynTensor> {
        // Reset encoder self-attention KV caches before each full encode.
        // The encoder processes the full mel in one shot — stale caches from
        // prior encode() calls would corrupt attention (R1-873 finding).
        self.reset_cache();

        // Conv1d expects [B, C, L] — mel is already in this format.
        let x = self.conv1.forward(mel)?;
        let x = x.gelu_erf()?;
        let x = self.conv2.forward(&x)?;
        let x = x.gelu_erf()?;

        // Transpose from [B, D, T] to [B, T, D] for transformer blocks.
        let x = x.transpose(1, 2)?;

        // Add sinusoidal positional embedding [T, D] broadcast over batch.
        let seq_len = x.dim(1)?;
        let pos_emb = self.positional_embedding.narrow(0, 0, seq_len)?;
        let pos_emb = pos_emb.unsqueeze(0)?; // [1, T, D]
        let mut x = x.broadcast_add(&pos_emb)?;

        // Transformer blocks — skip per-block NaN checks to avoid N flush+readback
        // cycles that cause Metal GPU timeout. Final output check below provides
        // defense-in-depth.
        x = with_nan_check_policy(NanCheckPolicy::Skip, || -> Result<DynTensor> {
            let mut h = x;
            for block in &mut self.blocks {
                h = block.forward_encoder(&h)?;
            }
            Ok(h)
        })?;

        // Final layer norm.
        let output = self.ln_post.forward(&x)?;
        // Defense-in-depth: validate final output OUTSIDE the skip scope
        check_output_finite(&output, "AudioEncoder")?;
        Ok(output)
    }

    /// Cache-free encoder forward for tracing.
    ///
    /// Equivalent to `forward` but avoids KV cache `slice_set` ops that the
    /// trace compiler cannot handle. Produces a clean computation graph for
    /// `trace_graph()` and `CompiledModel`. Takes `&self` (not `&mut self`).
    pub fn forward_no_cache(&self, mel: &DynTensor) -> Result<DynTensor> {
        let x = self.conv1.forward(mel)?;
        let x = x.gelu_erf()?;
        let x = self.conv2.forward(&x)?;
        let x = x.gelu_erf()?;

        let x = x.transpose(1, 2)?;

        let seq_len = x.dim(1)?;
        let pos_emb = self.positional_embedding.narrow(0, 0, seq_len)?;
        let pos_emb = pos_emb.unsqueeze(0)?;
        let mut x = x.broadcast_add(&pos_emb)?;

        // Skip per-block NaN checks — same rationale as forward().
        x = with_nan_check_policy(NanCheckPolicy::Skip, || -> Result<DynTensor> {
            let mut h = x;
            for block in &self.blocks {
                h = block.forward_encoder_no_cache(&h)?;
            }
            Ok(h)
        })?;

        let output = self.ln_post.forward(&x)?;
        // Defense-in-depth: validate final output OUTSIDE the skip scope
        check_output_finite(&output, "AudioEncoder")?;
        Ok(output)
    }

    /// Reset all encoder self-attention KV caches.
    pub fn reset_cache(&mut self) {
        for block in &mut self.blocks {
            block.reset_cache();
        }
    }
}
