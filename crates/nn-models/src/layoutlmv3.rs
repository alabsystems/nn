// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LayoutLMv3 multi-modal form model builder.

use crate::table_transformer::TransformerEncoderLayer;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Embedding, LayerNorm, Linear, Module};
use nn_core::var_builder::VarBuilder;
use nn_core::{Device, Result, TensorError};

const IMAGE_CHANNELS: usize = 3;
const LAYER_NORM_EPS: f64 = 1e-5;

/// LayoutLMv3 configuration.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct LayoutLMv3Config {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub vocab_size: usize,
    pub max_pos: usize,
    pub max_2d_pos: usize,
    pub patch_size: usize,
    pub image_size: usize,
    pub intermediate_size: usize,
    pub num_labels: usize,
}

impl LayoutLMv3Config {
    /// Default LayoutLMv3 base preset.
    #[must_use]
    pub fn preset(num_labels: usize) -> Self {
        Self {
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            vocab_size: 50_265,
            max_pos: 514,
            max_2d_pos: 1024,
            patch_size: 16,
            image_size: 224,
            intermediate_size: 3072,
            num_labels,
        }
    }

    /// Validate configuration consistency.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.num_layers == 0
            || self.num_heads == 0
            || self.num_labels == 0
        {
            return Err(TensorError::ValueOutOfRange {
                description:
                    "LayoutLMv3Config: hidden_size, num_layers, num_heads, and num_labels must be > 0",
            });
        }
        if !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(TensorError::ValueOutOfRange {
                description: "LayoutLMv3Config: hidden_size must be divisible by num_heads",
            });
        }
        if self.patch_size == 0 || self.image_size == 0 || self.max_pos == 0 || self.max_2d_pos == 0
        {
            return Err(TensorError::ValueOutOfRange {
                description:
                    "LayoutLMv3Config: patch_size, image_size, max_pos, and max_2d_pos must be > 0",
            });
        }
        if !self.image_size.is_multiple_of(self.patch_size) {
            return Err(TensorError::ValueOutOfRange {
                description: "LayoutLMv3Config: image_size must be divisible by patch_size",
            });
        }
        Ok(())
    }

    #[must_use]
    fn patch_dim(&self) -> usize {
        IMAGE_CHANNELS * self.patch_size * self.patch_size
    }

    #[must_use]
    fn visual_seq_len(&self) -> usize {
        let grid = self.image_size / self.patch_size;
        grid * grid
    }
}

/// LayoutLMv3 output.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct LayoutLMv3Output {
    /// Classification logits `[B, num_labels]`.
    pub logits: DynTensor,
    /// Final multimodal hidden state `[B, text_len + visual_len, hidden_size]`.
    pub last_hidden_state: DynTensor,
}

/// LayoutLMv3 form model.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct LayoutLMv3 {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    text_layer_norm: LayerNorm,
    visual_projection: Linear,
    visual_position_embeddings: Embedding,
    x_position_embeddings: Embedding,
    y_position_embeddings: Embedding,
    h_position_embeddings: Embedding,
    w_position_embeddings: Embedding,
    encoder_layers: Vec<TransformerEncoderLayer>,
    encoder_norm: LayerNorm,
    classifier: Linear,
    config: LayoutLMv3Config,
}

impl LayoutLMv3 {
    /// Load LayoutLMv3 weights from a VarBuilder.
    pub fn load(vb: impl AsRef<VarBuilder>, config: &LayoutLMv3Config) -> Result<Self> {
        config.validate()?;
        let vb = vb.as_ref();

        let mut encoder_layers = Vec::with_capacity(config.num_layers);
        for idx in 0..config.num_layers {
            encoder_layers.push(TransformerEncoderLayer::load(
                vb.pp(format!("encoder.layers.{idx}")),
                config.hidden_size,
                config.num_heads,
                config.intermediate_size,
            )?);
        }

        Ok(Self {
            word_embeddings: Embedding::load(
                vb.pp("text_embeddings.word_embeddings"),
                config.vocab_size,
                config.hidden_size,
            )?,
            position_embeddings: Embedding::load(
                vb.pp("text_embeddings.position_embeddings"),
                config.max_pos,
                config.hidden_size,
            )?,
            text_layer_norm: LayerNorm::load(
                vb.pp("text_embeddings.layer_norm"),
                config.hidden_size,
                LAYER_NORM_EPS,
            )?,
            visual_projection: Linear::load(
                vb.pp("visual_projection"),
                config.patch_dim(),
                config.hidden_size,
            )?,
            visual_position_embeddings: Embedding::load(
                vb.pp("visual_position_embeddings"),
                config.visual_seq_len(),
                config.hidden_size,
            )?,
            x_position_embeddings: Embedding::load(
                vb.pp("spatial.x_position_embeddings"),
                config.max_2d_pos,
                config.hidden_size,
            )?,
            y_position_embeddings: Embedding::load(
                vb.pp("spatial.y_position_embeddings"),
                config.max_2d_pos,
                config.hidden_size,
            )?,
            h_position_embeddings: Embedding::load(
                vb.pp("spatial.h_position_embeddings"),
                config.max_2d_pos,
                config.hidden_size,
            )?,
            w_position_embeddings: Embedding::load(
                vb.pp("spatial.w_position_embeddings"),
                config.max_2d_pos,
                config.hidden_size,
            )?,
            encoder_layers,
            encoder_norm: LayerNorm::load(
                vb.pp("encoder.norm"),
                config.hidden_size,
                LAYER_NORM_EPS,
            )?,
            classifier: Linear::load(vb.pp("classifier"), config.hidden_size, config.num_labels)?,
            config: config.clone(),
        })
    }

    /// Forward pass.
    ///
    /// - `input_ids`: `[B, T]` token ids
    /// - `bbox`: `[B, T, 4]` with `(x, y, h, w)` integer coordinates
    /// - `image`: `[B, 3, image_size, image_size]`
    pub fn forward(
        &self,
        input_ids: &DynTensor,
        bbox: &DynTensor,
        image: &DynTensor,
    ) -> Result<LayoutLMv3Output> {
        let (batch, seq_len) = input_ids.dims2()?;
        if seq_len > self.config.max_pos {
            return Err(TensorError::Unsupported(format!(
                "LayoutLMv3: seq_len {seq_len} exceeds max_pos {}",
                self.config.max_pos
            )));
        }
        let bbox_dims = bbox.dims();
        if bbox_dims.len() != 3
            || bbox_dims[0] != batch
            || bbox_dims[1] != seq_len
            || bbox_dims[2] != 4
        {
            return Err(TensorError::InvalidShape(format!(
                "LayoutLMv3: expected bbox shape [{batch}, {seq_len}, 4], got {bbox_dims:?}"
            )));
        }

        let mut hidden = self.embed_text(input_ids, bbox)?;
        let visual = self.embed_image(image)?;
        if visual.dim(0)? != batch {
            return Err(TensorError::InvalidShape(format!(
                "LayoutLMv3: image batch {} != text batch {batch}",
                visual.dim(0)?
            )));
        }
        hidden = DynTensor::cat(&[&hidden, &visual], 1)?;

        for layer in &self.encoder_layers {
            hidden = layer.forward_layer(&hidden)?;
        }
        let hidden = self.encoder_norm.forward(&hidden)?;
        let pooled = hidden.narrow(1, 0, 1)?.squeeze(1)?;

        Ok(LayoutLMv3Output {
            logits: self.classifier.forward(&pooled)?,
            last_hidden_state: hidden,
        })
    }

    /// Access the model configuration.
    #[must_use]
    pub fn config(&self) -> &LayoutLMv3Config {
        &self.config
    }

    fn embed_text(&self, input_ids: &DynTensor, bbox: &DynTensor) -> Result<DynTensor> {
        let seq_len = input_ids.dim(1)?;
        let word = self.word_embeddings.forward(input_ids)?;
        let pos = self
            .position_embeddings
            .forward(&position_ids(seq_len, &word.device())?)?
            .unsqueeze(0)?;
        let spatial = self.embed_spatial(bbox)?;
        self.text_layer_norm
            .forward(&word.broadcast_add(&pos)?.broadcast_add(&spatial)?)
    }

    fn embed_spatial(&self, bbox: &DynTensor) -> Result<DynTensor> {
        let x = bbox.narrow(2, 0, 1)?.squeeze(2)?;
        let y = bbox.narrow(2, 1, 1)?.squeeze(2)?;
        let h = bbox.narrow(2, 2, 1)?.squeeze(2)?;
        let w = bbox.narrow(2, 3, 1)?.squeeze(2)?;

        let x = self.x_position_embeddings.forward(&x)?;
        let y = self.y_position_embeddings.forward(&y)?;
        let h = self.h_position_embeddings.forward(&h)?;
        let w = self.w_position_embeddings.forward(&w)?;
        x.broadcast_add(&y)?.broadcast_add(&h)?.broadcast_add(&w)
    }

    fn embed_image(&self, image: &DynTensor) -> Result<DynTensor> {
        let patches = patchify_image(image, &self.config)?;
        let seq_len = patches.dim(1)?;
        let visual = self.visual_projection.forward(&patches)?;
        let pos = self
            .visual_position_embeddings
            .forward(&position_ids(seq_len, &visual.device())?)?
            .unsqueeze(0)?;
        visual.broadcast_add(&pos)
    }
}

fn patchify_image(image: &DynTensor, config: &LayoutLMv3Config) -> Result<DynTensor> {
    let (batch, channels, height, width) = image.dims4()?;
    if channels != IMAGE_CHANNELS {
        return Err(TensorError::InvalidShape(format!(
            "LayoutLMv3: expected 3 image channels, got {channels}"
        )));
    }
    if height != config.image_size || width != config.image_size {
        return Err(TensorError::Unsupported(format!(
            "LayoutLMv3: expected image size {}x{}, got {height}x{width}",
            config.image_size, config.image_size
        )));
    }

    let patch_size = config.patch_size;
    let grid_h = height / patch_size;
    let grid_w = width / patch_size;
    let seq_len = grid_h
        .checked_mul(grid_w)
        .ok_or_else(|| TensorError::InvalidShape("LayoutLMv3: patch grid overflow".into()))?;

    image
        .unfold(2, patch_size, patch_size)?
        .unfold(3, patch_size, patch_size)?
        .transpose(1, 2)?
        .transpose(2, 3)?
        .contiguous()?
        .reshape([batch, seq_len, config.patch_dim()])
}

fn position_ids(seq_len: usize, device: &Device) -> Result<DynTensor> {
    let seq_len_u32 = u32::try_from(seq_len).map_err(|_| TensorError::ValueOutOfRange {
        description: "position_ids: seq_len exceeds u32::MAX",
    })?;
    DynTensor::arange_u32(0, seq_len_u32, device)
}

#[cfg(test)]
#[path = "layoutlmv3_tests.rs"]
mod tests;
