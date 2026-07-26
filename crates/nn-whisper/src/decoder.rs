// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Whisper text decoder.
//!
//! Token embedding + learned positional embedding +
//! N residual attention blocks (self + cross) + LayerNorm + tied output projection.

use crate::block::ResidualAttentionBlock;
use crate::config::WhisperConfig;
use crate::positional::causal_mask;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{
    check_output_finite, with_nan_check_policy, Embedding, LayerNorm, Module, NanCheckPolicy,
};
use nn_core::Result;
use nn_core::TensorError;
use nn_core::VarBuilder;

/// Whisper text decoder.
pub struct TextDecoder {
    /// Token embedding: `[vocab_size, d_model]`.
    token_embedding: Embedding,
    /// Learned positional embedding: `[max_target_positions, d_model]`.
    positional_embedding: DynTensor,
    /// Decoder transformer blocks.
    blocks: Vec<ResidualAttentionBlock>,
    /// Final layer norm.
    ln: LayerNorm,
    /// Pre-computed causal mask: `[max_target_positions, max_target_positions]`.
    mask: DynTensor,
}

impl TextDecoder {
    /// Load decoder weights from VarBuilder.
    ///
    /// VarBuilder should be scoped to `model.decoder` prefix.
    pub fn load(vb: impl AsRef<VarBuilder>, config: &WhisperConfig) -> Result<Self> {
        let vb = vb.as_ref();
        let d = config.d_model;

        // Token embedding.
        let embed_w = vb.get(&[config.vocab_size, d], "embed_tokens.weight")?;
        let token_embedding = Embedding::new(embed_w)?;

        // Learned positional embedding.
        let positional_embedding =
            vb.get(&[config.max_target_positions, d], "embed_positions.weight")?;

        // Transformer blocks.
        let mut blocks = Vec::with_capacity(config.decoder_layers);
        for i in 0..config.decoder_layers {
            let block_vb = vb.pp(format!("layers.{i}"));
            let block = ResidualAttentionBlock::load_decoder(
                &block_vb,
                config.decoder_attention_heads,
                d,
                config.decoder_ffn_dim,
            )?;
            blocks.push(block);
        }

        // Final layer norm.
        let ln_w = vb.get(&[d], "layer_norm.weight")?;
        let ln_b = vb.get(&[d], "layer_norm.bias")?;
        let ln = LayerNorm::new(ln_w, ln_b, 1e-5)?;

        // Pre-compute causal mask. Use VarBuilder's dtype so mask matches
        // attention weight dtype (e.g., BF16) for GPU binary ops (#1710).
        let mask = causal_mask(config.max_target_positions, vb.dtype(), vb.device())?;

        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            mask,
        })
    }

    /// Decode one step: token IDs + encoder output -> logits.
    ///
    /// - `tokens`: `[batch, seq_len]` U32 token IDs
    /// - `encoder_output`: `[batch, audio_len, d_model]`
    /// - `flush_kv_cache`: true on first step to populate cross-attention cache
    /// - `position_offset`: token position offset for positional embedding lookup
    ///   (0 for initial prompt, then cumulative token count)
    ///
    /// Returns: `[batch, seq_len, vocab_size]` logits.
    pub fn forward(
        &mut self,
        tokens: &DynTensor,
        encoder_output: &DynTensor,
        flush_kv_cache: bool,
        position_offset: usize,
    ) -> Result<DynTensor> {
        let (_batch, seq_len) = tokens.dims2()?;

        // Token embedding: [B, T] -> [B, T, D].
        let x = self.token_embedding.forward(tokens)?;

        // Slice learned positional embedding for current positions.
        let pos_emb = self
            .positional_embedding
            .narrow(0, position_offset, seq_len)?;
        let pos_emb = pos_emb.unsqueeze(0)?; // [1, T, D]
        let mut x = x.broadcast_add(&pos_emb)?;

        // Slice causal mask to [seq_len, total_kv_len].
        // With self-attention KV cache, total KV length = cached tokens + new tokens.
        let total_kv_len = position_offset.checked_add(seq_len).ok_or_else(|| {
            TensorError::from(crate::WhisperError::PositionOverflow {
                offset: position_offset,
                seq_len,
            })
        })?;
        let mask = self.mask.narrow(0, position_offset, seq_len)?;
        let mask = mask.narrow(1, 0, total_kv_len)?;

        // Decoder blocks — skip per-block NaN checks to avoid N flush+readback
        // cycles that cause Metal GPU timeout. Final output check below provides
        // defense-in-depth.
        x = with_nan_check_policy(NanCheckPolicy::Skip, || -> Result<DynTensor> {
            let mut h = x;
            for block in &mut self.blocks {
                h = block.forward_decoder(&h, encoder_output, &mask, flush_kv_cache)?;
            }
            Ok(h)
        })?;

        // Final layer norm.
        let x = self.ln.forward(&x)?;

        // Tied output projection: logits = x @ embed_weight^T.
        let embed_weight = self.token_embedding.weight();
        let embed_weight_t = embed_weight.transpose(0, 1)?; // [D, vocab]
        let logits = x.matmul(&embed_weight_t)?;
        // Defense-in-depth: validate final output OUTSIDE the skip scope
        check_output_finite(&logits, "TextDecoder")?;
        Ok(logits)
    }

    /// Cache-free decoder forward for tracing.
    ///
    /// Processes the full token sequence at once (teacher-forcing) without
    /// any KV cache ops. Causal masking is applied via the pre-computed mask.
    /// Produces a clean computation graph for `trace_graph()` and `CompiledModel`.
    ///
    /// - `tokens`: `[batch, seq_len]` U32 token IDs
    /// - `encoder_output`: `[batch, audio_len, d_model]`
    ///
    /// Returns: `[batch, seq_len, vocab_size]` logits.
    pub fn forward_no_cache(
        &self,
        tokens: &DynTensor,
        encoder_output: &DynTensor,
    ) -> Result<DynTensor> {
        let (_batch, seq_len) = tokens.dims2()?;

        // Token embedding: [B, T] -> [B, T, D].
        let x = self.token_embedding.forward(tokens)?;

        // Slice learned positional embedding for positions [0, seq_len).
        let pos_emb = self.positional_embedding.narrow(0, 0, seq_len)?;
        let pos_emb = pos_emb.unsqueeze(0)?; // [1, T, D]
        let mut x = x.broadcast_add(&pos_emb)?;

        // Causal mask: [seq_len, seq_len] (no offset, full sequence).
        let mask = self.mask.narrow(0, 0, seq_len)?;
        let mask = mask.narrow(1, 0, seq_len)?;

        // Decoder blocks (cache-free).
        for block in &self.blocks {
            x = block.forward_decoder_no_cache(&x, encoder_output, &mask)?;
        }

        // Final layer norm.
        let x = self.ln.forward(&x)?;

        // Tied output projection: logits = x @ embed_weight^T.
        let embed_weight = self.token_embedding.weight();
        let embed_weight_t = embed_weight.transpose(0, 1)?; // [D, vocab]
        let logits = x.matmul(&embed_weight_t)?;
        check_output_finite(&logits, "TextDecoder")?;
        Ok(logits)
    }

    /// Reset all KV caches (call between utterances).
    pub fn reset_cache(&mut self) {
        for block in &mut self.blocks {
            block.reset_cache();
        }
    }
}

#[cfg(test)]
#[path = "decoder_tests.rs"]
mod decoder_tests;
