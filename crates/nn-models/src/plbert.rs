// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PlBert (ALBERT) text encoder for Kokoro TTS.
//!
//! ALBERT (A Lite BERT) uses factorized embeddings and cross-layer weight sharing.
//! A single transformer layer is reused for all `num_hidden_layers` iterations.
//!
//! Architecture:
//! - Factorized embeddings: 128-dim → 768-dim via linear projection
//! - Single shared transformer layer (12 heads, 768 hidden, 2048 intermediate)
//! - Applied `num_hidden_layers` times (default: 12)
//!
//! See `designs/archive/2026-03-16-kokoro-architecture-correction.md`.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Activation, Embedding, LayerNorm, Linear, Module};
use nn_core::var_builder::VarBuilder;
use nn_core::{Result, TensorError};

/// Configuration for PlBert (ALBERT encoder).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PlbertConfig {
    /// Vocabulary size (default: 178 for Kokoro).
    pub vocab_size: usize,
    /// Factorized embedding dimension (default: 128).
    pub embedding_dim: usize,
    /// Hidden size after projection (default: 768).
    pub hidden_size: usize,
    /// Number of attention heads (default: 12).
    pub num_attention_heads: usize,
    /// FFN intermediate size (default: 2048).
    pub intermediate_size: usize,
    /// Maximum position embeddings (default: 512).
    pub max_position_embeddings: usize,
    /// Number of shared layer iterations (default: 12).
    pub num_hidden_layers: usize,
    /// Layer norm epsilon (default: 1e-12).
    pub layer_norm_eps: f64,
}

impl Default for PlbertConfig {
    fn default() -> Self {
        Self {
            vocab_size: 178,
            embedding_dim: 128,
            hidden_size: 768,
            num_attention_heads: 12,
            intermediate_size: 2048,
            max_position_embeddings: 512,
            num_hidden_layers: 12,
            layer_norm_eps: 1e-12,
        }
    }
}

/// Multi-head self-attention for ALBERT.
///
/// Uses ALBERT-style weight names: `query`, `key`, `value`, `dense`.
struct AlbertAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dense: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl AlbertAttention {
    fn load(vb: impl AsRef<VarBuilder>, hidden_size: usize, num_heads: usize) -> Result<Self> {
        let vb = vb.as_ref();
        let head_dim = hidden_size / num_heads;
        let query = {
            let w = vb.get(&[hidden_size, hidden_size], "query.weight")?;
            let b = vb.get(&[hidden_size], "query.bias")?;
            Linear::new(w, Some(b))?
        };
        let key = {
            let w = vb.get(&[hidden_size, hidden_size], "key.weight")?;
            let b = vb.get(&[hidden_size], "key.bias")?;
            Linear::new(w, Some(b))?
        };
        let value = {
            let w = vb.get(&[hidden_size, hidden_size], "value.weight")?;
            let b = vb.get(&[hidden_size], "value.bias")?;
            Linear::new(w, Some(b))?
        };
        let dense = {
            let w = vb.get(&[hidden_size, hidden_size], "dense.weight")?;
            let b = vb.get(&[hidden_size], "dense.bias")?;
            Linear::new(w, Some(b))?
        };
        Ok(Self {
            query,
            key,
            value,
            dense,
            num_heads,
            head_dim,
        })
    }

    /// Forward: `hidden` is `[B, T, hidden_size]`.
    fn forward(&self, hidden: &DynTensor) -> Result<DynTensor> {
        let dims = hidden.dims();
        let batch = dims[0];
        let seq_len = dims[1];

        // Project Q, K, V: [B, T, H] → [B, T, H]
        let q = self.query.forward(hidden)?;
        let k = self.key.forward(hidden)?;
        let v = self.value.forward(hidden)?;

        // Reshape to [B, T, num_heads, head_dim] → [B, num_heads, T, head_dim]
        let q = q
            .reshape([batch, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = k
            .reshape([batch, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = v
            .reshape([batch, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        // Scaled dot-product attention via SDPA (FlashAttention on GPU).
        // Replaces manual Q@K^T + softmax + attn@V, saving 4 GPU dispatches
        // per attention block (K^T transpose, 2 matmuls, softmax → 1 SDPA).
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_output = nn_core::layers::attention::sdpa(&q, &k, &v, None, scale)?;

        // [B, num_heads, T, head_dim] → [B, T, hidden_size]
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape([
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ])?;

        // Output projection
        self.dense.forward(&attn_output)
    }
}

/// ALBERT feed-forward network: up-project → GELU → down-project.
struct AlbertFfn {
    up: Linear,
    down: Linear,
}

impl AlbertFfn {
    fn load(
        vb: impl AsRef<VarBuilder>,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let up = {
            let w = vb.get(&[intermediate_size, hidden_size], "ffn.weight")?;
            let b = vb.get(&[intermediate_size], "ffn.bias")?;
            Linear::new(w, Some(b))?
        };
        let down = {
            let w = vb.get(&[hidden_size, intermediate_size], "ffn_output.weight")?;
            let b = vb.get(&[hidden_size], "ffn_output.bias")?;
            Linear::new(w, Some(b))?
        };
        Ok(Self { up, down })
    }

    fn forward(&self, hidden: &DynTensor) -> Result<DynTensor> {
        let h = self.up.forward(hidden)?;
        let h = Activation::Gelu.forward(&h)?;
        self.down.forward(&h)
    }
}

/// Single shared ALBERT transformer layer.
///
/// Applied `num_hidden_layers` times with the same weights (cross-layer sharing).
struct AlbertLayer {
    attention: AlbertAttention,
    ffn: AlbertFfn,
    ln_post_attn: LayerNorm,
    ln_post_ffn: LayerNorm,
}

impl AlbertLayer {
    fn load(vb: impl AsRef<VarBuilder>, config: &PlbertConfig) -> Result<Self> {
        let vb = vb.as_ref();
        let attention = AlbertAttention::load(
            vb.pp("attention"),
            config.hidden_size,
            config.num_attention_heads,
        )?;
        let ffn = AlbertFfn::load(vb, config.hidden_size, config.intermediate_size)?;
        let ln_post_attn = {
            let w = vb.get(&[config.hidden_size], "attention.LayerNorm.weight")?;
            let b = vb.get(&[config.hidden_size], "attention.LayerNorm.bias")?;
            LayerNorm::new(w, b, config.layer_norm_eps)?
        };
        let ln_post_ffn = {
            let w = vb.get(&[config.hidden_size], "full_layer_layer_norm.weight")?;
            let b = vb.get(&[config.hidden_size], "full_layer_layer_norm.bias")?;
            LayerNorm::new(w, b, config.layer_norm_eps)?
        };
        Ok(Self {
            attention,
            ffn,
            ln_post_attn,
            ln_post_ffn,
        })
    }

    /// Forward: attention + residual + LN, then FFN + residual + LN.
    fn forward(&self, hidden: &DynTensor) -> Result<DynTensor> {
        let attn_out = self.attention.forward(hidden)?;
        let h = hidden.add(&attn_out)?;
        let h = self.ln_post_attn.forward(&h)?;
        let ffn_out = self.ffn.forward(&h)?;
        let h = h.add(&ffn_out)?;
        self.ln_post_ffn.forward(&h)
    }
}

/// PlBert: ALBERT encoder for Kokoro TTS text understanding.
///
/// Produces contextual embeddings `[B, T, 768]` from token IDs `[B, T]`.
///
/// Key ALBERT feature: all transformer layers share the **same** weights.
/// Only one `AlbertLayer` is instantiated and reused `num_hidden_layers` times.
pub struct PlBert {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    embedding_layer_norm: LayerNorm,
    embedding_projection: Linear,
    shared_layer: AlbertLayer,
    num_hidden_layers: usize,
}

impl PlBert {
    fn validate_seq_len(&self, seq_len: usize) -> Result<()> {
        let max_position_embeddings = self.position_embeddings.weight().dims()[0];
        if seq_len > max_position_embeddings {
            return Err(TensorError::Unsupported(format!(
                "PlBert seq_len {seq_len} exceeds max_position_embeddings {max_position_embeddings}"
            )));
        }
        Ok(())
    }

    /// Load PlBert weights from a VarBuilder.
    pub fn load(vb: impl AsRef<VarBuilder>, config: &PlbertConfig) -> Result<Self> {
        let vb = vb.as_ref();
        let emb_vb = vb.pp("embeddings");
        let word_embeddings = {
            let w = emb_vb.get(
                &[config.vocab_size, config.embedding_dim],
                "word_embeddings.weight",
            )?;
            Embedding::new(w)?
        };
        let position_embeddings = {
            let w = emb_vb.get(
                &[config.max_position_embeddings, config.embedding_dim],
                "position_embeddings.weight",
            )?;
            Embedding::new(w)?
        };
        let token_type_embeddings = {
            let w = emb_vb.get(&[2, config.embedding_dim], "token_type_embeddings.weight")?;
            Embedding::new(w)?
        };
        let embedding_layer_norm = {
            let w = emb_vb.get(&[config.embedding_dim], "LayerNorm.weight")?;
            let b = emb_vb.get(&[config.embedding_dim], "LayerNorm.bias")?;
            LayerNorm::new(w, b, config.layer_norm_eps)?
        };
        let enc_vb = vb.pp("encoder");
        let embedding_projection = {
            let w = enc_vb.get(
                &[config.hidden_size, config.embedding_dim],
                "embedding_hidden_mapping_in.weight",
            )?;
            let b = enc_vb.get(&[config.hidden_size], "embedding_hidden_mapping_in.bias")?;
            Linear::new(w, Some(b))?
        };
        let shared_layer =
            AlbertLayer::load(enc_vb.pp("albert_layer_groups.0.albert_layers.0"), config)?;
        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embedding_layer_norm,
            embedding_projection,
            shared_layer,
            num_hidden_layers: config.num_hidden_layers,
        })
    }

    /// Forward: token IDs → contextual embeddings `[B, T, hidden_size]`.
    ///
    /// `input_ids`: `[B, T]` token indices (u32 DynTensor).
    pub fn forward(&self, input_ids: &DynTensor) -> Result<DynTensor> {
        let dims = input_ids.dims();
        if dims.len() != 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: dims.len(),
            });
        }
        let seq_len = dims[1];
        self.validate_seq_len(seq_len)?;
        let seq_len_u32 = u32::try_from(seq_len).map_err(|_| TensorError::ValueOutOfRange {
            description: "PlBert seq_len exceeds u32::MAX",
        })?;

        // Word embeddings: [B, T] → [B, T, emb_dim]
        let word_emb = self.word_embeddings.forward(input_ids)?;

        // Position embeddings: [T] → [T, emb_dim] → [1, T, emb_dim]
        let input_device = input_ids.device();
        let position_ids = DynTensor::arange_u32(0, seq_len_u32, &input_device)?;
        let pos_emb = self
            .position_embeddings
            .forward(&position_ids)?
            .unsqueeze(0)?;

        // Token type embeddings: all zeros [T] → [T, emb_dim] → [1, T, emb_dim]
        let token_type_ids = DynTensor::zeros(&[seq_len], nn_core::DType::U32, &input_device)?;
        let type_emb = self
            .token_type_embeddings
            .forward(&token_type_ids)?
            .unsqueeze(0)?;

        // Sum embeddings + LayerNorm: all [B, T, emb_dim] via broadcast
        let emb = word_emb.broadcast_add(&pos_emb)?.broadcast_add(&type_emb)?;
        let emb = self.embedding_layer_norm.forward(&emb)?;

        // Factorized projection: emb_dim → hidden_size
        let mut hidden = self.embedding_projection.forward(&emb)?;

        // Apply shared transformer layer num_hidden_layers times
        for _ in 0..self.num_hidden_layers {
            hidden = self.shared_layer.forward(&hidden)?;
        }

        Ok(hidden)
    }

    /// Forward from pre-combined embeddings (for compiled segment tracing).
    ///
    /// Runs LayerNorm → projection → `num_hidden_layers` × shared AlbertLayer.
    /// This separates the constant-per-seq_len embedding computation (position
    /// and token-type) from the dynamic word embedding, allowing the trace
    /// system to compile the transformer layers while position/token-type
    /// embeddings are pre-computed outside the trace scope.
    ///
    /// `combined_emb`: `[B, T, embedding_dim]` — sum of word + position + type
    /// embeddings (before LayerNorm).
    ///
    /// Part of #2744, #2218.
    pub fn forward_core(&self, combined_emb: &DynTensor) -> Result<DynTensor> {
        if let Some(&seq_len) = combined_emb.dims().get(1) {
            self.validate_seq_len(seq_len)?;
        }
        let emb = self.embedding_layer_norm.forward(combined_emb)?;
        let mut hidden = self.embedding_projection.forward(&emb)?;
        for _ in 0..self.num_hidden_layers {
            hidden = self.shared_layer.forward(&hidden)?;
        }
        Ok(hidden)
    }

    /// Access the word embedding table.
    #[must_use]
    pub fn word_embeddings(&self) -> &Embedding {
        &self.word_embeddings
    }

    /// Access the position embedding table.
    #[must_use]
    pub fn position_embeddings(&self) -> &Embedding {
        &self.position_embeddings
    }

    /// Access the token-type embedding table.
    #[must_use]
    pub fn token_type_embeddings(&self) -> &Embedding {
        &self.token_type_embeddings
    }

    /// Number of shared transformer layer iterations.
    #[must_use]
    pub fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    /// Hidden size of the encoder output.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        // Inferred from embedding projection output dim
        self.embedding_projection.weight().dims()[0]
    }

    /// Current vocabulary size (number of rows in `word_embeddings`).
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.word_embeddings.weight().dims()[0]
    }

    /// Expand the word embedding table to accommodate `new_vocab_size` tokens.
    ///
    /// New rows are initialized as the mean of all existing embedding rows
    /// (similar to HuggingFace `model.resize_token_embeddings()`), which works
    /// better than zero-init for zero-shot use of new tokens.
    ///
    /// No-op if `new_vocab_size <= current vocab_size`.
    ///
    /// # Errors
    /// Returns `TensorError` if the underlying tensor operations fail.
    pub fn expand_vocab(&mut self, new_vocab_size: usize) -> Result<()> {
        let current = self.vocab_size();
        if new_vocab_size <= current {
            return Ok(());
        }
        let weight = self.word_embeddings.weight();
        let embed_dim = weight.dims()[1];
        let device = weight.device();

        // Compute mean of existing embeddings: [vocab, dim] → mean(dim=0) → [dim]
        let mean_row = weight.mean(0)?; // [embed_dim]
                                        // Expand to [n_new, embed_dim]
        let n_new = new_vocab_size - current;
        let mean_expanded = mean_row.unsqueeze(0)?.expand([n_new, embed_dim])?;
        // Concatenate: [vocab, dim] + [n_new, dim] → [new_vocab, dim]
        let new_weight = DynTensor::cat(&[weight, &mean_expanded], 0)?;
        self.word_embeddings = Embedding::new(new_weight.to_device(&device)?)?;
        Ok(())
    }
}

#[cfg(kani)]
#[path = "kani_plbert.rs"]
mod kani_proofs;

#[cfg(test)]
#[path = "plbert_tests.rs"]
mod tests;
