// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! UniTable model builder for dpdf table extraction.
//!
//! Architecture: linear patch projection + transformer encoder/decoder.
//! Reference: Peng et al. 2024, "UniTable".

use crate::table_transformer::TransformerEncoderLayer;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Activation, Embedding, LayerNorm, Linear, Module, MultiHeadAttention};
use nn_core::var_builder::VarBuilder;
use nn_core::{Device, Result, TensorError};

const IMAGE_CHANNELS: usize = 3;
const LAYER_NORM_EPS: f64 = 1e-5;

/// UniTable configuration.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct UniTableConfig {
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub max_seq_len: usize,
    pub vocab_size: usize,
    pub patch_size: usize,
    pub image_size: usize,
}

impl UniTableConfig {
    /// Default UniTable preset from Peng et al. 2024.
    #[must_use]
    pub fn preset() -> Self {
        Self {
            hidden_dim: 768,
            num_layers: 6,
            num_heads: 12,
            max_seq_len: 1024,
            vocab_size: 200,
            patch_size: 16,
            image_size: 448,
        }
    }

    /// Validate configuration consistency.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_dim == 0 || self.num_layers == 0 || self.num_heads == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "UniTableConfig: hidden_dim, num_layers, and num_heads must be > 0",
            });
        }
        if !self.hidden_dim.is_multiple_of(self.num_heads) {
            return Err(TensorError::ValueOutOfRange {
                description: "UniTableConfig: hidden_dim must be divisible by num_heads",
            });
        }
        if self.patch_size == 0 || self.image_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "UniTableConfig: patch_size and image_size must be > 0",
            });
        }
        if !self.image_size.is_multiple_of(self.patch_size) {
            return Err(TensorError::ValueOutOfRange {
                description: "UniTableConfig: image_size must be divisible by patch_size",
            });
        }
        if self.image_seq_len() > self.max_seq_len {
            return Err(TensorError::ValueOutOfRange {
                description: "UniTableConfig: max_seq_len must cover the image patch sequence",
            });
        }
        Ok(())
    }

    #[must_use]
    fn image_seq_len(&self) -> usize {
        let grid = self.image_size / self.patch_size;
        grid * grid
    }

    #[must_use]
    fn patch_dim(&self) -> usize {
        IMAGE_CHANNELS * self.patch_size * self.patch_size
    }

    fn ffn_dim(&self) -> Result<usize> {
        self.hidden_dim
            .checked_mul(4)
            .ok_or_else(|| TensorError::InvalidShape("UniTableConfig: FFN dim overflow".into()))
    }
}

/// Single decoder layer: self-attention + cross-attention + FFN.
#[derive(Clone, Debug)]
struct UniTableDecoderLayer {
    self_attn: MultiHeadAttention,
    cross_attn: MultiHeadAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

impl UniTableDecoderLayer {
    fn load(
        vb: impl AsRef<VarBuilder>,
        hidden_dim: usize,
        num_heads: usize,
        ffn_dim: usize,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        Ok(Self {
            self_attn: MultiHeadAttention::load(
                vb.pp("self_attn"),
                hidden_dim,
                num_heads,
                num_heads,
                true,
            )?,
            cross_attn: MultiHeadAttention::load(
                vb.pp("cross_attn"),
                hidden_dim,
                num_heads,
                num_heads,
                true,
            )?,
            norm1: LayerNorm::load(vb.pp("norm1"), hidden_dim, LAYER_NORM_EPS)?,
            norm2: LayerNorm::load(vb.pp("norm2"), hidden_dim, LAYER_NORM_EPS)?,
            norm3: LayerNorm::load(vb.pp("norm3"), hidden_dim, LAYER_NORM_EPS)?,
            linear1: Linear::load(vb.pp("linear1"), hidden_dim, ffn_dim)?,
            linear2: Linear::load(vb.pp("linear2"), ffn_dim, hidden_dim)?,
        })
    }

    fn forward_layer(&self, x: &DynTensor, memory: &DynTensor) -> Result<DynTensor> {
        let residual = x.clone();
        let h = self.norm1.forward(x)?;
        let h = self.self_attn.forward(&h, None, None, None, 0)?;
        let x = residual.broadcast_add(&h)?;

        let residual = x.clone();
        let h = self.norm2.forward(&x)?;
        let h = self.cross_attn.forward(&h, Some(memory), None, None, 0)?;
        let x = residual.broadcast_add(&h)?;

        let residual = x.clone();
        let h = self.norm3.forward(&x)?;
        let h = self.linear1.forward(&h)?;
        let h = Activation::Relu.forward(&h)?;
        let h = self.linear2.forward(&h)?;
        residual.broadcast_add(&h)
    }
}

/// UniTable decoder output.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct UniTableOutput {
    /// Token logits `[B, T, vocab_size]`.
    pub logits: DynTensor,
}

/// UniTable table extraction model.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct UniTable {
    patch_projection: Linear,
    token_embeddings: Embedding,
    position_embeddings: Embedding,
    encoder_layers: Vec<TransformerEncoderLayer>,
    encoder_norm: LayerNorm,
    decoder_layers: Vec<UniTableDecoderLayer>,
    decoder_norm: LayerNorm,
    vocab_head: Linear,
    config: UniTableConfig,
}

impl UniTable {
    /// Load UniTable weights from a VarBuilder.
    pub fn load(vb: impl AsRef<VarBuilder>, config: &UniTableConfig) -> Result<Self> {
        config.validate()?;
        let vb = vb.as_ref();
        let ffn_dim = config.ffn_dim()?;

        let patch_projection = Linear::load(
            vb.pp("patch_projection"),
            config.patch_dim(),
            config.hidden_dim,
        )?;
        let token_embeddings = Embedding::load(
            vb.pp("token_embeddings"),
            config.vocab_size,
            config.hidden_dim,
        )?;
        let position_embeddings = Embedding::load(
            vb.pp("position_embeddings"),
            config.max_seq_len,
            config.hidden_dim,
        )?;

        let mut encoder_layers = Vec::with_capacity(config.num_layers);
        let mut decoder_layers = Vec::with_capacity(config.num_layers);
        for idx in 0..config.num_layers {
            encoder_layers.push(TransformerEncoderLayer::load(
                vb.pp(format!("encoder.layers.{idx}")),
                config.hidden_dim,
                config.num_heads,
                ffn_dim,
            )?);
            decoder_layers.push(UniTableDecoderLayer::load(
                vb.pp(format!("decoder.layers.{idx}")),
                config.hidden_dim,
                config.num_heads,
                ffn_dim,
            )?);
        }

        Ok(Self {
            patch_projection,
            token_embeddings,
            position_embeddings,
            encoder_layers,
            encoder_norm: LayerNorm::load(
                vb.pp("encoder.norm"),
                config.hidden_dim,
                LAYER_NORM_EPS,
            )?,
            decoder_layers,
            decoder_norm: LayerNorm::load(
                vb.pp("decoder.norm"),
                config.hidden_dim,
                LAYER_NORM_EPS,
            )?,
            vocab_head: Linear::load(vb.pp("vocab_head"), config.hidden_dim, config.vocab_size)?,
            config: config.clone(),
        })
    }

    /// Forward pass.
    ///
    /// - `image`: `[B, 3, image_size, image_size]`
    /// - `decoder_input_ids`: `[B, T]` token ids
    pub fn forward(
        &self,
        image: &DynTensor,
        decoder_input_ids: &DynTensor,
    ) -> Result<UniTableOutput> {
        let memory = self.encode_image(image)?;
        let (batch, seq_len) = decoder_input_ids.dims2()?;
        if batch != memory.dim(0)? {
            return Err(TensorError::InvalidShape(format!(
                "UniTable: image batch {} != decoder batch {batch}",
                memory.dim(0)?
            )));
        }
        if seq_len > self.config.max_seq_len {
            return Err(TensorError::Unsupported(format!(
                "UniTable: decoder seq_len {seq_len} exceeds max_seq_len {}",
                self.config.max_seq_len
            )));
        }

        let hidden = self.token_embeddings.forward(decoder_input_ids)?;
        let pos = self
            .position_embeddings
            .forward(&position_ids(seq_len, &hidden.device())?)?
            .unsqueeze(0)?;
        let mut hidden = hidden.broadcast_add(&pos)?;

        for layer in &self.decoder_layers {
            hidden = layer.forward_layer(&hidden, &memory)?;
        }
        let hidden = self.decoder_norm.forward(&hidden)?;
        Ok(UniTableOutput {
            logits: self.vocab_head.forward(&hidden)?,
        })
    }

    /// Access the model configuration.
    #[must_use]
    pub fn config(&self) -> &UniTableConfig {
        &self.config
    }

    fn encode_image(&self, image: &DynTensor) -> Result<DynTensor> {
        let patches = patchify_image(image, &self.config)?;
        let seq_len = patches.dim(1)?;
        let hidden = self.patch_projection.forward(&patches)?;
        let pos = self
            .position_embeddings
            .forward(&position_ids(seq_len, &hidden.device())?)?
            .unsqueeze(0)?;
        let mut hidden = hidden.broadcast_add(&pos)?;
        for layer in &self.encoder_layers {
            hidden = layer.forward_layer(&hidden)?;
        }
        self.encoder_norm.forward(&hidden)
    }
}

fn patchify_image(image: &DynTensor, config: &UniTableConfig) -> Result<DynTensor> {
    let (batch, channels, height, width) = image.dims4()?;
    if channels != IMAGE_CHANNELS {
        return Err(TensorError::InvalidShape(format!(
            "UniTable: expected 3 image channels, got {channels}"
        )));
    }
    if height != config.image_size || width != config.image_size {
        return Err(TensorError::Unsupported(format!(
            "UniTable: expected image size {}x{}, got {height}x{width}",
            config.image_size, config.image_size
        )));
    }

    let patch_size = config.patch_size;
    let grid_h = height / patch_size;
    let grid_w = width / patch_size;
    let seq_len = grid_h
        .checked_mul(grid_w)
        .ok_or_else(|| TensorError::InvalidShape("UniTable: patch grid overflow".into()))?;

    let patches = image
        .unfold(2, patch_size, patch_size)?
        .unfold(3, patch_size, patch_size)?
        .transpose(1, 2)?
        .transpose(2, 3)?
        .contiguous()?;

    patches.reshape([batch, seq_len, config.patch_dim()])
}

fn position_ids(seq_len: usize, device: &Device) -> Result<DynTensor> {
    let seq_len_u32 = u32::try_from(seq_len).map_err(|_| TensorError::ValueOutOfRange {
        description: "position_ids: seq_len exceeds u32::MAX",
    })?;
    DynTensor::arange_u32(0, seq_len_u32, device)
}

#[cfg(test)]
#[path = "unitable_tests.rs"]
mod tests;
