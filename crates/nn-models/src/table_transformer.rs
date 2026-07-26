// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Table Transformer (DETR) model builder for dpdf table detection.
//!
//! Architecture: ResNet-18 backbone + DETR encoder/decoder (28.8M params).
//! Two presets: detection (2 classes) and structure recognition (6 classes).
//!
//! Reference: Smock et al. 2022, "PubTables-1M", CVPR 2022.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::vision::DetrDecoder;
use nn_core::layers::{
    Activation, BatchNorm2d, BatchNormConfig, Conv2d, Conv2dConfig, LayerNorm, Linear, Module,
    MultiHeadAttention,
};
use nn_core::var_builder::VarBuilder;
use nn_core::{Device, Result, TensorError};

/// DETR hidden dimension.
pub const HIDDEN_DIM: usize = 256;
/// Number of attention heads.
pub const NUM_HEADS: usize = 8;
/// Number of transformer encoder layers.
pub const NUM_ENCODER_LAYERS: usize = 6;
/// Number of transformer decoder layers.
pub const NUM_DECODER_LAYERS: usize = 6;
/// Number of learned object queries.
pub const NUM_QUERIES: usize = 125;
/// FFN intermediate dimension.
pub const FFN_DIM: usize = 2048;
/// ResNet-18 final feature map channels.
const BACKBONE_OUT_CHANNELS: usize = 512;

/// Table Transformer configuration.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TableTransformerConfig {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_encoder_layers: usize,
    pub num_decoder_layers: usize,
    pub num_queries: usize,
    pub num_classes: usize,
    pub ffn_dim: usize,
}

impl TableTransformerConfig {
    /// Preset for table detection (2 classes: table + no-object).
    #[must_use]
    pub fn preset_detection() -> Self {
        Self {
            hidden_dim: HIDDEN_DIM,
            num_heads: NUM_HEADS,
            num_encoder_layers: NUM_ENCODER_LAYERS,
            num_decoder_layers: NUM_DECODER_LAYERS,
            num_queries: NUM_QUERIES,
            num_classes: 2,
            ffn_dim: FFN_DIM,
        }
    }

    /// Preset for table structure recognition (6 classes).
    ///
    /// Classes: table, row, column, spanning-cell, projected-row-header, no-object.
    #[must_use]
    pub fn preset_structure() -> Self {
        Self {
            hidden_dim: HIDDEN_DIM,
            num_heads: NUM_HEADS,
            num_encoder_layers: NUM_ENCODER_LAYERS,
            num_decoder_layers: NUM_DECODER_LAYERS,
            num_queries: NUM_QUERIES,
            num_classes: 6,
            ffn_dim: FFN_DIM,
        }
    }

    /// Validate configuration consistency.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_dim == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "TableTransformerConfig: hidden_dim must be > 0",
            });
        }
        if !self.hidden_dim.is_multiple_of(self.num_heads) {
            return Err(TensorError::ValueOutOfRange {
                description: "TableTransformerConfig: hidden_dim must be divisible by num_heads",
            });
        }
        if self.num_queries == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "TableTransformerConfig: num_queries must be > 0",
            });
        }
        Ok(())
    }
}

/// ResNet-18 basic residual block: two 3x3 convs with BN + optional downsample.
#[derive(Clone, Debug)]
pub struct BasicBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    downsample: Option<(Conv2d, BatchNorm2d)>,
}

impl BasicBlock {
    /// Load from a VarBuilder scoped to `layer{n}.{block_idx}`.
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        in_channels: usize,
        out_channels: usize,
        stride: usize,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let bn_cfg = BatchNormConfig::default();

        let conv1_cfg = Conv2dConfig::new(1, stride, 1);
        let conv1 = Conv2d::load(vb.pp("conv1"), in_channels, out_channels, 3, conv1_cfg)?;
        let bn1 = BatchNorm2d::load(vb.pp("bn1"), out_channels, bn_cfg)?;

        let conv2_cfg = Conv2dConfig::new(1, 1, 1);
        let conv2 = Conv2d::load(vb.pp("conv2"), out_channels, out_channels, 3, conv2_cfg)?;
        let bn2 = BatchNorm2d::load(vb.pp("bn2"), out_channels, bn_cfg)?;

        let downsample = if stride != 1 || in_channels != out_channels {
            let ds_conv_cfg = Conv2dConfig::new(0, stride, 1);
            let ds_conv = Conv2d::load(
                vb.pp("downsample.0"),
                in_channels,
                out_channels,
                1,
                ds_conv_cfg,
            )?;
            let ds_bn = BatchNorm2d::load(vb.pp("downsample.1"), out_channels, bn_cfg)?;
            Some((ds_conv, ds_bn))
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            downsample,
        })
    }

    /// Forward pass with residual connection.
    pub fn forward_block(&self, x: &DynTensor) -> Result<DynTensor> {
        let identity = match &self.downsample {
            Some((conv, bn)) => bn.forward(&conv.forward(x)?)?,
            None => x.clone(),
        };
        let out = self.conv1.forward(x)?;
        let out = self.bn1.forward(&out)?;
        let out = Activation::Relu.forward(&out)?;
        let out = self.conv2.forward(&out)?;
        let out = self.bn2.forward(&out)?;
        let out = out.broadcast_add(&identity)?;
        Activation::Relu.forward(&out)
    }
}

/// ResNet-18 backbone: `[B, 3, H, W]` -> `[B, 512, H/32, W/32]`.
#[derive(Clone, Debug)]
pub struct ResNet18Backbone {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    layer1: Vec<BasicBlock>,
    layer2: Vec<BasicBlock>,
    layer3: Vec<BasicBlock>,
    layer4: Vec<BasicBlock>,
}

impl ResNet18Backbone {
    /// Load from a VarBuilder scoped to `backbone`.
    pub fn load(vb: impl AsRef<VarBuilder>) -> Result<Self> {
        let vb = vb.as_ref();

        // Initial conv: 7x7, stride 2, padding 3
        let conv1_cfg = Conv2dConfig::new(3, 2, 1);
        let conv1 = Conv2d::load(vb.pp("conv1"), 3, 64, 7, conv1_cfg)?;
        let bn1 = BatchNorm2d::load(vb.pp("bn1"), 64, BatchNormConfig::default())?;

        let layer1 = Self::load_layer(vb, "layer1", 64, 64, 1)?;
        let layer2 = Self::load_layer(vb, "layer2", 64, 128, 2)?;
        let layer3 = Self::load_layer(vb, "layer3", 128, 256, 2)?;
        let layer4 = Self::load_layer(vb, "layer4", 256, 512, 2)?;

        Ok(Self {
            conv1,
            bn1,
            layer1,
            layer2,
            layer3,
            layer4,
        })
    }

    fn load_layer(
        vb: &VarBuilder,
        name: &str,
        in_channels: usize,
        out_channels: usize,
        stride: usize,
    ) -> Result<Vec<BasicBlock>> {
        let layer_vb = vb.pp(name);
        let block0 = BasicBlock::load(layer_vb.pp("0"), in_channels, out_channels, stride)?;
        let block1 = BasicBlock::load(layer_vb.pp("1"), out_channels, out_channels, 1)?;
        Ok(vec![block0, block1])
    }

    /// Forward: `[B, 3, H, W]` -> `[B, 512, H/32, W/32]`.
    pub fn forward_backbone(&self, x: &DynTensor) -> Result<DynTensor> {
        let x = self.conv1.forward(x)?;
        let x = self.bn1.forward(&x)?;
        let x = Activation::Relu.forward(&x)?;
        let x = x.max_pool2d(3, 2, 1)?;

        let mut x = x;
        for block in &self.layer1 {
            x = block.forward_block(&x)?;
        }
        for block in &self.layer2 {
            x = block.forward_block(&x)?;
        }
        for block in &self.layer3 {
            x = block.forward_block(&x)?;
        }
        for block in &self.layer4 {
            x = block.forward_block(&x)?;
        }
        Ok(x)
    }
}

/// Single DETR encoder layer: self-attention + FFN with pre-norm.
#[derive(Clone, Debug)]
pub struct TransformerEncoderLayer {
    self_attn: MultiHeadAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    ffn_linear1: Linear,
    ffn_linear2: Linear,
}

impl TransformerEncoderLayer {
    /// Load from a VarBuilder.
    pub fn load(
        vb: impl AsRef<VarBuilder>,
        dim: usize,
        num_heads: usize,
        ffn_dim: usize,
    ) -> Result<Self> {
        let vb = vb.as_ref();
        let self_attn =
            MultiHeadAttention::load(vb.pp("self_attn"), dim, num_heads, num_heads, true)?;
        let norm1 = LayerNorm::load(vb.pp("norm1"), dim, 1e-5)?;
        let norm2 = LayerNorm::load(vb.pp("norm2"), dim, 1e-5)?;
        let ffn_linear1 = Linear::load(vb.pp("linear1"), dim, ffn_dim)?;
        let ffn_linear2 = Linear::load(vb.pp("linear2"), ffn_dim, dim)?;
        Ok(Self {
            self_attn,
            norm1,
            norm2,
            ffn_linear1,
            ffn_linear2,
        })
    }

    /// Forward: self-attention + FFN with residual connections.
    pub fn forward_layer(&self, x: &DynTensor) -> Result<DynTensor> {
        // Self-attention block
        let residual = x;
        let h = self.norm1.forward(x)?;
        let h = self.self_attn.forward(&h, None, None, None, 0)?;
        let x = h.broadcast_add(residual)?;

        // FFN block
        let residual = x.clone();
        let h = self.norm2.forward(&x)?;
        let h = self.ffn_linear1.forward(&h)?;
        let h = Activation::Relu.forward(&h)?;
        let h = self.ffn_linear2.forward(&h)?;
        h.broadcast_add(&residual)
    }
}

/// Table Transformer output: class logits + bounding box predictions.
#[derive(Debug)]
pub struct TableTransformerOutput {
    /// Classification logits: `[B, num_queries, num_classes + 1]`.
    /// Last class is "no object".
    pub logits: DynTensor,
    /// Bounding box predictions: `[B, num_queries, 4]`.
    /// Format: (cx, cy, w, h) normalized to [0, 1] via sigmoid.
    pub boxes: DynTensor,
}

/// Table Transformer: ResNet-18 backbone + DETR encoder/decoder.
#[derive(Clone, Debug)]
pub struct TableTransformer {
    backbone: ResNet18Backbone,
    input_proj: Conv2d,
    encoder_layers: Vec<TransformerEncoderLayer>,
    encoder_norm: LayerNorm,
    decoder: DetrDecoder,
    config: TableTransformerConfig,
}

impl TableTransformer {
    /// Load from a VarBuilder with the given configuration.
    pub fn load(vb: impl AsRef<VarBuilder>, config: &TableTransformerConfig) -> Result<Self> {
        config.validate()?;
        let vb = vb.as_ref();

        let backbone = ResNet18Backbone::load(vb.pp("backbone"))?;

        let proj_cfg = Conv2dConfig::new(0, 1, 1);
        let input_proj = Conv2d::load(
            vb.pp("input_proj"),
            BACKBONE_OUT_CHANNELS,
            config.hidden_dim,
            1,
            proj_cfg,
        )?;

        let mut encoder_layers = Vec::with_capacity(config.num_encoder_layers);
        for i in 0..config.num_encoder_layers {
            let layer = TransformerEncoderLayer::load(
                vb.pp(format!("encoder.layers.{i}")),
                config.hidden_dim,
                config.num_heads,
                config.ffn_dim,
            )?;
            encoder_layers.push(layer);
        }
        let encoder_norm = LayerNorm::load(vb.pp("encoder.norm"), config.hidden_dim, 1e-5)?;

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
            input_proj,
            encoder_layers,
            encoder_norm,
            decoder,
            config: config.clone(),
        })
    }

    /// Forward pass.
    ///
    /// Input: `[B, 3, H, W]` image tensor (H, W should be divisible by 32).
    /// Output: [`TableTransformerOutput`] with logits and boxes.
    pub fn forward(&self, image: &DynTensor) -> Result<TableTransformerOutput> {
        let rank = image.rank();
        if rank != 4 {
            return Err(TensorError::RankMismatch {
                expected: 4,
                actual: rank,
            });
        }

        // 1. Backbone: [B, 3, H, W] -> [B, 512, H/32, W/32]
        let features = self.backbone.forward_backbone(image)?;

        // 2. Input projection: [B, 512, H/32, W/32] -> [B, 256, H/32, W/32]
        let proj = self.input_proj.forward(&features)?;

        // 3. Flatten spatial dims: [B, 256, h, w] -> [B, h*w, 256]
        let b = proj.dim(0)?;
        let c = proj.dim(1)?;
        let h = proj.dim(2)?;
        let w = proj.dim(3)?;
        let seq_len = h * w;
        let flat = proj.reshape([b, c, seq_len])?;
        let flat = flat.transpose(1, 2)?; // [B, seq_len, 256]

        // 4. Add 2D sinusoidal positional encoding
        let pos_embed = sinusoidal_2d_pos_encoding(h, w, c, &flat.device())?;
        let encoded = flat.broadcast_add(&pos_embed.unsqueeze(0)?)?;

        // 5. Transformer encoder
        let mut x = encoded;
        for layer in &self.encoder_layers {
            x = layer.forward_layer(&x)?;
        }
        let memory = self.encoder_norm.forward(&x)?;

        // 6. DETR decoder
        let detr_out = self.decoder.forward_decode(&memory, None)?;

        Ok(TableTransformerOutput {
            logits: detr_out.class_logits,
            boxes: detr_out.bbox_preds,
        })
    }

    /// Access the model configuration.
    #[must_use]
    pub fn config(&self) -> &TableTransformerConfig {
        &self.config
    }
}

/// 2D sinusoidal positional encoding: `[h*w, dim]` with row/column sin/cos.
pub(crate) fn sinusoidal_2d_pos_encoding(
    h: usize,
    w: usize,
    dim: usize,
    device: &Device,
) -> Result<DynTensor> {
    let half_dim = dim / 2;
    let seq_len = h * w;
    let mut data = vec![0.0f32; seq_len * dim];

    for row in 0..h {
        for col in 0..w {
            let pos_idx = row * w + col;
            for i in 0..half_dim {
                let denom = 10000.0_f64.powf(2.0 * (i as f64) / half_dim as f64);

                // Row encoding in first half
                let row_angle = row as f64 / denom;
                data[pos_idx * dim + i] = if i % 2 == 0 {
                    row_angle.sin() as f32
                } else {
                    row_angle.cos() as f32
                };

                // Column encoding in second half
                let col_angle = col as f64 / denom;
                data[pos_idx * dim + half_dim + i] = if i % 2 == 0 {
                    col_angle.sin() as f32
                } else {
                    col_angle.cos() as f32
                };
            }
        }
    }

    DynTensor::from_vec(data, &[seq_len, dim], device)
}

/// Detection class names.
pub const DETECTION_CLASSES: [&str; 2] = ["table", "no-object"];

/// Structure recognition class names.
pub const STRUCTURE_CLASSES: [&str; 6] = [
    "table",
    "row",
    "column",
    "spanning-cell",
    "projected-row-header",
    "no-object",
];

#[cfg(test)]
#[path = "table_transformer_tests.rs"]
mod tests;
