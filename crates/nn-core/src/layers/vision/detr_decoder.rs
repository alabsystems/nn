// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! DETR-style transformer decoder for object detection.
//!
//! Standard DETR decoder architecture: learned object queries attend to encoder
//! features via cross-attention to produce detection predictions. Used by
//! Table Transformer (DETR variant for table structure recognition).
//!
//! ```text
//! object_queries  [B, N_q, D]     encoder_features  [B, H*W, D]
//!       │                                  │
//!       ├── self-attention (queries ↔ queries)
//!       ├── cross-attention (queries → encoder features)
//!       ├── FFN (Linear → ReLU → Linear)
//!       │   (repeated N_layers times)
//!       ▼
//!   decoded  [B, N_q, D]
//!       │
//!       ├── class_head → [B, N_q, num_classes + 1]
//!       └── bbox_head  → [B, N_q, 4]
//! ```

use crate::dyn_tensor::DynTensor;
use crate::error::Result;
use crate::layers::{Activation, LayerNorm, Linear, Module, MultiHeadAttention};
use crate::var_builder::VarBuilder;

/// Single DETR decoder layer: self-attn + cross-attn + FFN.
///
/// Each layer has:
/// - Layer-norm + self-attention (queries attend to each other)
/// - Layer-norm + cross-attention (queries attend to encoder features)
/// - Layer-norm + feed-forward network
///
/// Uses pre-norm (layer norm before attention) matching common DETR variants.
#[derive(Clone, Debug)]
pub struct DetrDecoderLayer {
    self_attn: MultiHeadAttention,
    cross_attn: MultiHeadAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    ffn_linear1: Linear,
    ffn_linear2: Linear,
}

impl DetrDecoderLayer {
    /// Create from pre-loaded components.
    pub fn new(
        self_attn: MultiHeadAttention,
        cross_attn: MultiHeadAttention,
        norm1: LayerNorm,
        norm2: LayerNorm,
        norm3: LayerNorm,
        ffn_linear1: Linear,
        ffn_linear2: Linear,
    ) -> Self {
        Self {
            self_attn,
            cross_attn,
            norm1,
            norm2,
            norm3,
            ffn_linear1,
            ffn_linear2,
        }
    }

    /// Load from a VarBuilder.
    ///
    /// - `dim`: model dimension
    /// - `num_heads`: number of attention heads
    /// - `ffn_dim`: feed-forward hidden dimension (typically 4 * dim)
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        dim: usize,
        num_heads: usize,
        ffn_dim: usize,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let self_attn =
            MultiHeadAttention::load(vb.pp("self_attn"), dim, num_heads, num_heads, true)?;
        let cross_attn =
            MultiHeadAttention::load(vb.pp("cross_attn"), dim, num_heads, num_heads, true)?;
        let norm1 = LayerNorm::load(vb.pp("norm1"), dim, 1e-5)?;
        let norm2 = LayerNorm::load(vb.pp("norm2"), dim, 1e-5)?;
        let norm3 = LayerNorm::load(vb.pp("norm3"), dim, 1e-5)?;
        let ffn_linear1 = Linear::load(vb.pp("linear1"), dim, ffn_dim)?;
        let ffn_linear2 = Linear::load(vb.pp("linear2"), ffn_dim, dim)?;
        Ok(Self {
            self_attn,
            cross_attn,
            norm1,
            norm2,
            norm3,
            ffn_linear1,
            ffn_linear2,
        })
    }

    /// Forward pass.
    ///
    /// - `tgt`: object query features `[B, N_q, D]`
    /// - `memory`: encoder output features `[B, H*W, D]`
    /// - `pos_embed`: optional positional embedding added to queries
    pub fn forward_layer(
        &self,
        tgt: &DynTensor,
        memory: &DynTensor,
        pos_embed: Option<&DynTensor>,
    ) -> Result<DynTensor> {
        // Self-attention: pre-norm on content, then add position to Q/K input.
        // NOTE: MHA API uses kv_input for both K and V, so V also receives
        // position embedding here. Standard DETR has V=tgt (position-free),
        // but separating K from V requires MHA API changes. This is a known
        // simplification that works well in practice.
        let residual = tgt;
        let x = self.norm1.forward(tgt)?;
        let x = match pos_embed {
            Some(pe) => {
                let q_with_pos = x.broadcast_add(pe)?;
                self.self_attn.forward(&q_with_pos, None, None, None, 0)?
            }
            None => self.self_attn.forward(&x, None, None, None, 0)?,
        };
        let x = x.broadcast_add(residual)?;

        // Cross-attention: queries attend to encoder memory.
        // Add query position to Q for cross-attention (standard DETR convention).
        let residual = x.clone();
        let x = self.norm2.forward(&x)?;
        let x = match pos_embed {
            Some(pe) => {
                let q_with_pos = x.broadcast_add(pe)?;
                self.cross_attn
                    .forward(&q_with_pos, Some(memory), None, None, 0)?
            }
            None => self.cross_attn.forward(&x, Some(memory), None, None, 0)?,
        };
        let x = x.broadcast_add(&residual)?;

        // FFN
        let residual = x.clone();
        let x = self.norm3.forward(&x)?;
        let x = self.ffn_linear1.forward(&x)?;
        let x = Activation::Relu.forward(&x)?;
        let x = self.ffn_linear2.forward(&x)?;
        x.broadcast_add(&residual)
    }
}

/// DETR transformer decoder with object queries.
///
/// Composes N [`DetrDecoderLayer`]s with learned object query embeddings.
/// Output can be fed to classification and bounding box prediction heads.
///
/// # Weight names
///
/// - `"query_embed.weight"` — learned object queries `[num_queries, dim]`
/// - `"layers.{i}.self_attn.*"` — self-attention per layer
/// - `"layers.{i}.cross_attn.*"` — cross-attention per layer
/// - `"layers.{i}.norm1.*"`, `"layers.{i}.norm2.*"`, `"layers.{i}.norm3.*"`
/// - `"layers.{i}.linear1.*"`, `"layers.{i}.linear2.*"` — FFN per layer
/// - `"class_head.weight"`, `"class_head.bias"` — classification projection
/// - `"bbox_head.weight"`, `"bbox_head.bias"` — bounding box projection
#[derive(Clone, Debug)]
pub struct DetrDecoder {
    query_embed: DynTensor,
    layers: Vec<DetrDecoderLayer>,
    final_norm: LayerNorm,
    class_head: Linear,
    bbox_head: Linear,
    num_queries: usize,
}

/// DETR decoder output.
#[derive(Debug)]
pub struct DetrOutput {
    /// Classification logits: `[B, num_queries, num_classes + 1]`.
    /// Last class is "no object".
    pub class_logits: DynTensor,
    /// Bounding box predictions: `[B, num_queries, 4]`.
    /// Format: (cx, cy, w, h) normalized to [0, 1].
    pub bbox_preds: DynTensor,
}

impl DetrDecoder {
    /// Create from pre-loaded components.
    pub fn new(
        query_embed: DynTensor,
        layers: Vec<DetrDecoderLayer>,
        final_norm: LayerNorm,
        class_head: Linear,
        bbox_head: Linear,
    ) -> Result<Self> {
        let num_queries = query_embed.dim(0)?;
        Ok(Self {
            query_embed,
            layers,
            final_norm,
            class_head,
            bbox_head,
            num_queries,
        })
    }

    /// Load from a VarBuilder.
    ///
    /// - `dim`: model dimension
    /// - `num_heads`: attention heads per layer
    /// - `ffn_dim`: FFN hidden dimension
    /// - `num_layers`: number of decoder layers
    /// - `num_queries`: number of object queries (e.g. 100 for DETR, 125 for Table Transformer)
    /// - `num_classes`: number of object classes (output includes +1 for "no object")
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        dim: usize,
        num_heads: usize,
        ffn_dim: usize,
        num_layers: usize,
        num_queries: usize,
        num_classes: usize,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let query_embed = vb.pp("query_embed").get(&[num_queries, dim], "weight")?;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer =
                DetrDecoderLayer::load(vb.pp(format!("layers.{i}")), dim, num_heads, ffn_dim)?;
            layers.push(layer);
        }

        let final_norm = LayerNorm::load(vb.pp("final_norm"), dim, 1e-5)?;
        // +1 for "no object" class
        let class_head = Linear::load(vb.pp("class_head"), dim, num_classes + 1)?;
        let bbox_head = Linear::load(vb.pp("bbox_head"), dim, 4)?;

        Self::new(query_embed, layers, final_norm, class_head, bbox_head)
    }

    /// Number of object queries.
    #[must_use]
    pub fn num_queries(&self) -> usize {
        self.num_queries
    }

    /// Forward pass.
    ///
    /// - `memory`: encoder output features `[B, H*W, D]`
    /// - `pos_embed`: optional positional encoding for the queries
    ///
    /// Returns [`DetrOutput`] with classification logits and bbox predictions.
    pub fn forward_decode(
        &self,
        memory: &DynTensor,
        pos_embed: Option<&DynTensor>,
    ) -> Result<DetrOutput> {
        let b = memory.dim(0)?;
        // Expand learned queries to batch: [num_queries, D] -> [B, num_queries, D]
        let queries = self.query_embed.unsqueeze(0)?;
        let mut tgt = queries.expand([b, self.num_queries, queries.dim(2)?])?;

        for layer in &self.layers {
            tgt = layer.forward_layer(&tgt, memory, pos_embed)?;
        }

        let tgt = self.final_norm.forward(&tgt)?;
        let class_logits = self.class_head.forward(&tgt)?;
        let bbox_preds = self.bbox_head.forward(&tgt)?;
        let bbox_preds = bbox_preds.sigmoid()?;

        Ok(DetrOutput {
            class_logits,
            bbox_preds,
        })
    }
}

#[cfg(test)]
#[path = "detr_decoder_tests.rs"]
mod tests;
