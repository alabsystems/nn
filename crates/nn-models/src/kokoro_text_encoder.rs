// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! [`TextEncoder`] — embedding + conv layers + bidirectional LSTM for Kokoro-82M.
//!
//! Architecture: Embedding → 3×[Conv1d(k=5,p=2) → LayerNorm → LeakyReLU(0.2)]
//! → BiLSTM(d_en, d_en/2) → Linear projection.
//!
//! Takes raw token IDs `[B, T]` and produces encoded features `[B, d_en, T]`.
//!
//! See `designs/archive/2026-03-16-kokoro-architecture-correction.md`, correction #5.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{BiLstm, Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear, Module};
use nn_core::var_builder::VarBuilder;
use nn_core::{Result, TensorError};

use crate::kokoro_error::KokoroError;

/// Text encoder: embedding + conv layers + bidirectional LSTM + projection.
///
/// Input: token IDs `[B, T]`.
/// Output: `[B, d_en, T]` encoded text features.
///
/// Matches PyTorch Kokoro `TextEncoder` (hexgrad/kokoro `modules.py`):
/// - `Embedding(vocab_size, d_en)` (no scaling — PyTorch original has no sqrt(d_en))
/// - 3× `Conv1d(d_en, d_en, k=5, p=2)` + `LayerNorm(d_en)` + `LeakyReLU(0.2)`
/// - `BiLSTM(d_en, d_en/2)` → output `d_en`
/// - `Linear(d_en, d_en)` projection
pub struct TextEncoder {
    embedding: Embedding,
    convs: Vec<Conv1d>,
    norms: Vec<LayerNorm>,
    bilstm: BiLstm,
    lstm_proj: Linear,
}

impl TextEncoder {
    /// Load from VarBuilder under the `text_encoder` prefix.
    ///
    /// `vocab_size`: token vocabulary size (178 for Kokoro).
    /// `d_en`: encoder dimension (512 for Kokoro).
    pub fn load(vb: impl AsRef<VarBuilder>, vocab_size: usize, d_en: usize) -> Result<Self> {
        let vb = vb.as_ref();
        if !d_en.is_multiple_of(2) {
            return Err(TensorError::ValueOutOfRange {
                description: "TextEncoder d_en must be even for BiLSTM (hidden = d_en/2)",
            });
        }

        // Embedding(vocab_size, d_en)
        let embed_w = vb.get(&[vocab_size, d_en], "embedding.weight")?;
        let embedding = Embedding::new(embed_w)?;

        // 3× Conv1d(d_en, d_en, k=5, p=2) + LayerNorm(d_en)
        let num_convs = 3;
        let mut convs = Vec::with_capacity(num_convs);
        let mut norms = Vec::with_capacity(num_convs);
        for i in 0..num_convs {
            let cw = vb.get(&[d_en, d_en, 5], &format!("convs.{i}.weight"))?;
            let cb = vb.get(&[d_en], &format!("convs.{i}.bias"))?;
            convs.push(Conv1d::new(
                cw,
                Some(cb),
                Conv1dConfig::default().with_padding(2),
            )?);
            norms.push(LayerNorm::load(vb.pp(format!("norms.{i}")), d_en, 1e-5)?);
        }

        // BiLSTM(d_en, hidden=d_en/2) → output 2*hidden = d_en
        // Uses BiLstm::load() which supports both PyTorch-native and
        // dvoice decomposed LSTM weight naming (#2741).
        let hidden = d_en / 2;
        let bilstm = BiLstm::load(vb.pp("lstm"), d_en, hidden)?;

        // Linear(d_en, d_en) projection
        let w = vb.get(&[d_en, d_en], "lstm.linear.weight")?;
        let b = vb.get(&[d_en], "lstm.linear.bias")?;
        let lstm_proj = Linear::new(w, Some(b))?;

        Ok(Self {
            embedding,
            convs,
            norms,
            bilstm,
            lstm_proj,
        })
    }

    /// Embed token IDs to channel-first float features.
    ///
    /// `tokens`: `[B, T]` — integer token IDs.
    /// Returns: `[B, d_en, T]` — embedded, transposed to channel-first layout.
    ///
    /// Note: No scaling applied. PyTorch Kokoro TextEncoder does NOT scale
    /// embeddings by sqrt(d_en). Part of #2691.
    pub fn embed_to_channels_first(
        &self,
        tokens: &DynTensor,
    ) -> std::result::Result<DynTensor, KokoroError> {
        let x = self.embedding.forward(tokens)?;
        Ok(x.transpose(1, 2)?)
    }

    /// Post-embedding forward: Conv + LayerNorm + BiLSTM + projection.
    ///
    /// `h`: `[B, d_en, T]` — channel-first embedded features.
    /// Returns: `[B, d_en, T]` — encoded text features.
    pub fn forward_post_embedding(
        &self,
        h: &DynTensor,
    ) -> std::result::Result<DynTensor, KokoroError> {
        let mut h = h.contiguous()?;
        for (conv, norm) in self.convs.iter().zip(self.norms.iter()) {
            let conv_out = conv.forward(&h)?;
            let conv_t = conv_out.transpose(1, 2)?;
            let normed = norm.forward(&conv_t)?;
            h = normed.transpose(1, 2)?.leaky_relu(0.2)?;
        }
        let h_t = h.permute([2, 0, 1])?;
        let (lstm_out, _, _) = self.bilstm.forward_seq(&h_t, None, None)?;
        let lstm_bt = lstm_out.transpose(0, 1)?;
        let projected = self.lstm_proj.forward(&lstm_bt)?;
        Ok(projected.transpose(1, 2)?)
    }

    /// Forward pass: token IDs → encoded features.
    ///
    /// `tokens`: `[B, T]` — integer token IDs.
    /// Returns: `[B, d_en, T]` encoded text features.
    pub fn forward(&self, tokens: &DynTensor) -> std::result::Result<DynTensor, KokoroError> {
        let h = self.embed_to_channels_first(tokens)?;
        self.forward_post_embedding(&h)
    }
}
