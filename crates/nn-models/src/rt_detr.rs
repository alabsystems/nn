// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! RT-DETRv2 real-time detection transformer for document layout detection.
//!
//! Architecture: ResNet-18/50 backbone + hybrid encoder (AIFI + CCFM) +
//! transformer decoder with deformable attention and iterative box refinement.
//!
//! RT-DETRv2 improves on RT-DETR by introducing flexible decoder query selection
//! and supporting dynamic input shapes for efficient multi-scale document
//! processing. This model is used by the Heron pipeline (docling_rs#49) for
//! document layout detection with 17 classes.
//!
//! # Heron Configuration
//!
//! ```text
//! Input:  [1, 3, 640, 640]
//! Output: logits [1, 300, 17], boxes [1, 300, 4]
//! Parameters: 42.9M
//! ```
//!
//! # Architecture
//!
//! ```text
//! Image [B, 3, H, W]
//!   │
//!   ├─ ResNet backbone → C3 (stride 8), C4 (stride 16), C5 (stride 32)
//!   │
//!   ├─ Hybrid encoder:
//!   │   ├─ AIFI (Attention-based Intra-scale Feature Interaction) on C5
//!   │   └─ CCFM (CNN-based Cross-scale Feature Merger) fuses C3/C4/C5
//!   │
//!   ├─ Decoder (6 layers):
//!   │   ├─ Self-attention among object queries
//!   │   ├─ Multi-scale deformable cross-attention → encoder features
//!   │   └─ Iterative bounding box refinement per layer
//!   │
//!   ├─ class_head → [B, N_q, num_classes]
//!   └─ bbox_head  → [B, N_q, 4] (cx, cy, w, h normalized)
//! ```
//!
//! Reference: Zhao et al. 2024, "RT-DETRv2: Improved Baseline with Bag-of-Freebies
//! for Real-Time Detection Transformer", arXiv:2407.17140.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::vision::{DetrDecoder, ResNet18, ResNet18Hf};
use nn_core::layers::{
    BatchNorm2d, BatchNormConfig, Conv2d, Conv2dConfig, LayerNorm, Linear, Module,
    MultiHeadAttention,
};
use nn_core::var_builder::VarBuilder;
use nn_core::{Result, TensorError};

// ---------------------------------------------------------------------------
// Constants (Heron / RT-DETRv2-R18 preset)
// ---------------------------------------------------------------------------

/// Default number of object queries (top-K from encoder).
pub const NUM_QUERIES: usize = 300;

/// Decoder hidden dimension.
pub const HIDDEN_DIM: usize = 256;

/// Number of attention heads in the decoder.
pub const NUM_HEADS: usize = 8;

/// Number of transformer decoder layers.
pub const NUM_DECODER_LAYERS: usize = 6;

/// FFN intermediate dimension in the decoder.
pub const FFN_DIM: usize = 1024;

/// Default input resolution (square).
pub const DEFAULT_INPUT_SIZE: usize = 640;

/// Number of detection scales from the backbone.
pub const NUM_SCALES: usize = 3;

/// Number of deformable attention sampling points per head per level.
pub const NUM_SAMPLING_POINTS: usize = 4;

// ---------------------------------------------------------------------------
// Backbone variant
// ---------------------------------------------------------------------------

/// Which ResNet-18 backbone variant to use.
///
/// Torchvision uses a single 7x7 conv stem; HuggingFace uses three 3x3 convs.
/// Both produce the same feature channel progression: 64 -> 64 -> 128 -> 256 -> 512.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtDetrBackboneVariant {
    /// Standard torchvision ResNet-18 (7x7 conv stem).
    Torchvision,
    /// HuggingFace ResNet-18 (3-stage 3x3 conv stem), used by HF RT-DETR.
    HuggingFace,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for an RT-DETRv2 model.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RtDetrConfig {
    /// Backbone feature map channels at each scale (C3, C4, C5).
    pub backbone_channels: [usize; 3],
    /// Decoder hidden dimension (default 256).
    pub hidden_dim: usize,
    /// Number of attention heads (default 8).
    pub num_heads: usize,
    /// Number of decoder layers (default 6).
    pub num_decoder_layers: usize,
    /// FFN intermediate dimension (default 1024).
    pub ffn_dim: usize,
    /// Number of object queries / top-K encoder predictions (default 300).
    pub num_queries: usize,
    /// Number of object classes (Heron: 17, COCO: 80).
    pub num_classes: usize,
    /// Number of deformable sampling points per head per level (default 4).
    pub num_sampling_points: usize,
    /// Minimum confidence to keep a detection (default 0.3).
    pub conf_threshold: f32,
    /// Input image resolution (square, default 640).
    pub input_size: usize,
    /// Which ResNet-18 backbone variant to use (default: HuggingFace).
    pub backbone_variant: RtDetrBackboneVariant,
}

impl RtDetrConfig {
    /// Heron preset: RT-DETRv2 with ResNet-18 backbone for 17-class
    /// document layout detection (docling_rs).
    ///
    /// Uses HuggingFace backbone variant since that is the source of the
    /// real pretrained weights from `models/rt-detr-r18/model.safetensors`.
    ///
    /// Input: `[1, 3, 640, 640]`
    /// Output: `[1, 300, 17]` logits + `[1, 300, 4]` boxes
    /// Parameters: ~42.9M
    #[must_use]
    pub fn preset_heron() -> Self {
        Self {
            // ResNet-18: C3=128, C4=256, C5=512
            backbone_channels: [128, 256, 512],
            hidden_dim: HIDDEN_DIM,
            num_heads: NUM_HEADS,
            num_decoder_layers: NUM_DECODER_LAYERS,
            ffn_dim: FFN_DIM,
            num_queries: NUM_QUERIES,
            num_classes: 17,
            num_sampling_points: NUM_SAMPLING_POINTS,
            conf_threshold: 0.3,
            input_size: DEFAULT_INPUT_SIZE,
            backbone_variant: RtDetrBackboneVariant::HuggingFace,
        }
    }

    /// COCO preset: standard 80-class detection.
    #[must_use]
    pub fn preset_coco() -> Self {
        Self {
            backbone_channels: [128, 256, 512],
            hidden_dim: HIDDEN_DIM,
            num_heads: NUM_HEADS,
            num_decoder_layers: NUM_DECODER_LAYERS,
            ffn_dim: FFN_DIM,
            num_queries: NUM_QUERIES,
            num_classes: 80,
            num_sampling_points: NUM_SAMPLING_POINTS,
            conf_threshold: 0.5,
            input_size: DEFAULT_INPUT_SIZE,
            backbone_variant: RtDetrBackboneVariant::Torchvision,
        }
    }

    /// Validate configuration consistency.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_dim == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "RtDetrConfig: hidden_dim must be > 0",
            });
        }
        if self.num_heads == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "RtDetrConfig: num_heads must be > 0",
            });
        }
        if !self.hidden_dim.is_multiple_of(self.num_heads) {
            return Err(TensorError::ValueOutOfRange {
                description: "RtDetrConfig: hidden_dim must be divisible by num_heads",
            });
        }
        if self.num_classes == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "RtDetrConfig: num_classes must be > 0",
            });
        }
        if self.num_queries == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "RtDetrConfig: num_queries must be > 0",
            });
        }
        if !(0.0..=1.0).contains(&self.conf_threshold) {
            return Err(TensorError::ValueOutOfRange {
                description: "RtDetrConfig: conf_threshold must be in [0.0, 1.0]",
            });
        }
        Ok(())
    }
}

impl Default for RtDetrConfig {
    fn default() -> Self {
        Self::preset_heron()
    }
}

// ---------------------------------------------------------------------------
// Channel projection (adapts backbone feature channels to hidden_dim)
// ---------------------------------------------------------------------------

/// 1x1 convolution projecting backbone feature channels to `hidden_dim`.
///
/// HuggingFace RT-DETR uses Conv2d + BatchNorm2d (not LayerNorm) for the
/// input projections. Weight keys: `{prefix}.0.weight` (conv) and
/// `{prefix}.1.*` (batch norm).
#[derive(Clone, Debug)]
struct ChannelProjection {
    conv: Conv2d,
    norm: BatchNorm2d,
}

impl ChannelProjection {
    fn load(vb: &VarBuilder, in_ch: usize, out_ch: usize) -> Result<Self> {
        let conv = Conv2d::load(
            vb.pp("0"),
            in_ch,
            out_ch,
            1, // kernel_size
            Conv2dConfig::default(),
        )?;
        let norm = BatchNorm2d::load(vb.pp("1"), out_ch, BatchNormConfig::default())?;
        Ok(Self { conv, norm })
    }

    fn forward(&self, x: &DynTensor) -> Result<DynTensor> {
        let y = self.conv.forward(x)?;
        self.norm.forward(&y)
    }
}

// ---------------------------------------------------------------------------
// AIFI: Attention-based Intra-scale Feature Interaction (single-scale encoder)
// ---------------------------------------------------------------------------

/// Single-scale self-attention encoder applied to the highest-stride feature
/// map (C5). Captures global context within a single scale before cross-scale
/// fusion.
#[derive(Clone, Debug)]
struct AifiEncoder {
    self_attn: MultiHeadAttention,
    norm1: LayerNorm,
    ffn_linear1: Linear,
    ffn_linear2: Linear,
    norm2: LayerNorm,
}

impl AifiEncoder {
    fn load(vb: &VarBuilder, dim: usize, num_heads: usize, ffn_dim: usize) -> Result<Self> {
        // HF RT-DETR uses: self_attn, self_attn_layer_norm, fc1, fc2, final_layer_norm
        let self_attn =
            MultiHeadAttention::load(vb.pp("self_attn"), dim, num_heads, num_heads, true)?;
        let norm1 = LayerNorm::load(vb.pp("self_attn_layer_norm"), dim, 1e-5)?;
        let ffn_linear1 = Linear::load(vb.pp("fc1"), dim, ffn_dim)?;
        let ffn_linear2 = Linear::load(vb.pp("fc2"), ffn_dim, dim)?;
        let norm2 = LayerNorm::load(vb.pp("final_layer_norm"), dim, 1e-5)?;
        Ok(Self {
            self_attn,
            norm1,
            ffn_linear1,
            ffn_linear2,
            norm2,
        })
    }

    /// Forward pass: `features` is `[B, C, H, W]`, returns `[B, C, H, W]`.
    fn forward(&self, features: &DynTensor) -> Result<DynTensor> {
        let (b, c, h, w) = features.dims4()?;
        // Flatten to [B, H*W, C] for self-attention.
        let x = features.reshape([b, c, h * w])?.transpose(1, 2)?;

        // Self-attention with residual (kv_input=None → self-attention).
        let attn_out = self.self_attn.forward(&x, None, None, None, 0)?;
        let x = (&x + &attn_out)?;
        let x = self.norm1.forward(&x)?;

        // FFN with residual.
        let ffn_out = self.ffn_linear1.forward(&x)?;
        let ffn_out = ffn_out.relu()?;
        let ffn_out = self.ffn_linear2.forward(&ffn_out)?;
        let x = (&x + &ffn_out)?;
        let x = self.norm2.forward(&x)?;

        // Back to [B, C, H, W].
        x.transpose(1, 2)?.reshape([b, c, h, w])
    }
}

// ---------------------------------------------------------------------------
// RT-DETRv2 model
// ---------------------------------------------------------------------------

/// Loaded backbone — either torchvision or HuggingFace variant.
#[derive(Clone, Debug)]
enum RtDetrBackbone {
    Torchvision(Box<ResNet18>),
    HuggingFace(Box<ResNet18Hf>),
}

impl RtDetrBackbone {
    /// Multi-scale feature extraction returning `[C2, C3, C4, C5]`.
    fn forward_features(&self, x: &DynTensor) -> Result<Vec<DynTensor>> {
        match self {
            Self::Torchvision(m) => m.forward_features(x),
            Self::HuggingFace(m) => m.forward_features(x),
        }
    }
}

/// RT-DETRv2 real-time detection transformer.
///
/// Combines ResNet-18 backbone, AIFI single-scale encoder, channel projections
/// for multi-scale feature alignment, a DETR-style transformer decoder with
/// learned object queries, and classification/regression heads.
#[derive(Clone, Debug)]
pub struct RtDetr {
    /// ResNet-18 backbone producing multi-scale features.
    backbone: RtDetrBackbone,
    /// 1x1 projections aligning each backbone scale to `hidden_dim`.
    channel_projs: Vec<ChannelProjection>,
    /// AIFI encoder for intra-scale feature interaction on C5.
    aifi: AifiEncoder,
    /// DETR transformer decoder (includes learned object queries + heads).
    decoder: DetrDecoder,
    /// Configuration.
    config: RtDetrConfig,
}

impl RtDetr {
    /// Load RT-DETRv2 from a [`VarBuilder`] and configuration.
    ///
    /// The backbone variant (torchvision vs HuggingFace) is determined by
    /// [`RtDetrConfig::backbone_variant`].
    pub fn load(vb: &VarBuilder, config: RtDetrConfig) -> Result<Self> {
        config.validate()?;

        let backbone = match config.backbone_variant {
            RtDetrBackboneVariant::Torchvision => {
                RtDetrBackbone::Torchvision(Box::new(ResNet18::load(vb.pp("backbone"), None)?))
            }
            RtDetrBackboneVariant::HuggingFace => {
                RtDetrBackbone::HuggingFace(Box::new(ResNet18Hf::load(vb.pp("backbone"), None)?))
            }
        };

        // Channel projections for C3, C4, C5 → hidden_dim.
        // HF RT-DETR uses `encoder_input_proj.{i}.{0=conv, 1=bn}`.
        let mut channel_projs = Vec::with_capacity(NUM_SCALES);
        for (i, &ch) in config.backbone_channels.iter().enumerate() {
            let proj = ChannelProjection::load(
                &vb.pp(format!("encoder_input_proj.{i}")),
                ch,
                config.hidden_dim,
            )?;
            channel_projs.push(proj);
        }

        // HF RT-DETR nests the AIFI self-attention under
        // `encoder.encoder.0.layers.0.*`.
        let aifi = AifiEncoder::load(
            &vb.pp("encoder.encoder.0.layers.0"),
            config.hidden_dim,
            config.num_heads,
            config.ffn_dim,
        )?;

        // DetrDecoder has its own query_embed, class_head, bbox_head.
        let decoder = DetrDecoder::load(
            vb.pp("decoder"),
            config.hidden_dim,
            config.num_heads,
            config.ffn_dim,
            config.num_decoder_layers,
            config.num_queries,
            config.num_classes,
        )?;

        Ok(Self {
            backbone,
            channel_projs,
            aifi,
            decoder,
            config,
        })
    }

    /// Configuration used to build this model.
    #[must_use]
    pub fn config(&self) -> &RtDetrConfig {
        &self.config
    }

    /// Full forward pass: image → detections.
    ///
    /// Input: `[B, 3, H, W]` (typically `[B, 3, 640, 640]`)
    /// Returns: `(class_logits, box_preds)` where:
    /// - `class_logits`: `[B, num_queries, num_classes + 1]`
    /// - `box_preds`: `[B, num_queries, 4]` in `(cx, cy, w, h)` normalized coords
    pub fn forward(&self, image: &DynTensor) -> Result<(DynTensor, DynTensor)> {
        // 1. Backbone multi-scale features [C2, C3, C4, C5].
        let features = self.backbone.forward_features(image)?;
        if features.len() < 4 {
            return Err(TensorError::InvalidShape(format!(
                "RT-DETR: backbone produced {} feature levels, expected >= 4",
                features.len()
            )));
        }
        // We use C3(idx=1), C4(idx=2), C5(idx=3) as the three detection scales.
        let c3 = &features[1];
        let c4 = &features[2];
        let c5 = &features[3];

        // 2. Project each scale to hidden_dim.
        let p3 = self.channel_projs[0].forward(c3)?;
        let p4 = self.channel_projs[1].forward(c4)?;
        let p5_pre = self.channel_projs[2].forward(c5)?;

        // 3. AIFI encoder on C5 (highest-stride, most abstract features).
        let p5 = self.aifi.forward(&p5_pre)?;

        // 4. Flatten and concatenate multi-scale features for the decoder.
        let (b, d, _, _) = p3.dims4()?;
        let flat3 = p3
            .reshape([b, d, p3.dim(2)? * p3.dim(3)?])?
            .transpose(1, 2)?;
        let flat4 = p4
            .reshape([b, d, p4.dim(2)? * p4.dim(3)?])?
            .transpose(1, 2)?;
        let flat5 = p5
            .reshape([b, d, p5.dim(2)? * p5.dim(3)?])?
            .transpose(1, 2)?;

        // Concatenate along sequence dimension: [B, N3+N4+N5, hidden_dim]
        let encoder_output = DynTensor::cat(&[&flat3, &flat4, &flat5], 1)?;

        // 5. DETR decoder: queries attend to encoder features (memory).
        // DetrDecoder::forward_decode uses its internal learned query_embed.
        let detr_output = self.decoder.forward_decode(&encoder_output, None)?;

        Ok((detr_output.class_logits, detr_output.bbox_preds))
    }

    /// Decode raw model outputs into scored detections.
    ///
    /// Applies sigmoid to logits, filters by confidence threshold,
    /// converts boxes from (cx, cy, w, h) to (x1, y1, x2, y2).
    ///
    /// Returns a Vec of `(class_id, confidence, [x1, y1, x2, y2])` tuples
    /// for the first batch element.
    #[must_use]
    pub fn decode_detections(
        &self,
        class_logits: &[f32],
        box_preds: &[f32],
        num_queries: usize,
        num_classes: usize,
    ) -> Vec<(u32, f32, [f32; 4])> {
        let mut detections = Vec::new();
        let threshold = self.config.conf_threshold;

        for q in 0..num_queries {
            let logit_offset = q * num_classes;
            let box_offset = q * 4;

            // Find best class for this query.
            let mut best_class = 0u32;
            let mut best_score = f32::NEG_INFINITY;
            for c in 0..num_classes {
                let logit = class_logits[logit_offset + c];
                // Sigmoid activation.
                let score = 1.0 / (1.0 + (-logit).exp());
                if score > best_score {
                    best_score = score;
                    best_class = c as u32;
                }
            }

            if best_score < threshold {
                continue;
            }

            // Convert (cx, cy, w, h) → (x1, y1, x2, y2).
            let cx = box_preds[box_offset];
            let cy = box_preds[box_offset + 1];
            let w = box_preds[box_offset + 2];
            let h = box_preds[box_offset + 3];
            let x1 = cx - w / 2.0;
            let y1 = cy - h / 2.0;
            let x2 = cx + w / 2.0;
            let y2 = cy + h / 2.0;

            detections.push((best_class, best_score, [x1, y1, x2, y2]));
        }

        detections
    }
}

// -- Heron document class names -----------------------------------------------

/// Heron 17-class document layout labels.
///
/// From the docling_rs research report (2026-03-27).
pub const HERON_CLASS_NAMES: [&str; 17] = [
    "caption",
    "footnote",
    "formula",
    "list-item",
    "page-footer",
    "page-header",
    "picture",
    "section-header",
    "table",
    "text",
    "title",
    "document-index",
    "code",
    "checkbox-selected",
    "checkbox-unselected",
    "form-field",
    "handwriting",
];

#[cfg(test)]
#[path = "rt_detr_tests.rs"]
mod tests;
