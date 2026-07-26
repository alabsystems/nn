// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Configuration for Table Transformer / DETR-based table structure recognition.
//!
//! Extends [`super::table_transformer::TableTransformerConfig`] with backbone,
//! decoder, and post-processing parameters needed for end-to-end table
//! detection and structure extraction in the dpdf pipeline.
//!
//! # Architecture Overview
//!
//! - **Backbone**: ResNet-18/50 CNN feature extractor producing multi-scale
//!   feature maps. Configurable depth and pretrained initialization.
//! - **Encoder**: Transformer encoder with positional encoding over flattened
//!   feature maps.
//! - **Decoder**: Transformer decoder with learned object queries for
//!   set-prediction of table elements (rows, columns, cells, spans).
//! - **Post-processing**: Hungarian matching + NMS + confidence filtering.
//!
//! Reference: Smock et al. 2022, "PubTables-1M", CVPR 2022.

use nn_core::{Result, TensorError};

// ---------------------------------------------------------------------------
// Backbone config
// ---------------------------------------------------------------------------

/// Backbone architecture variant for the Table Transformer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackboneVariant {
    /// ResNet-18 (11.7M params, faster).
    ResNet18,
    /// ResNet-50 (25.6M params, more accurate).
    ResNet50,
}

/// Configuration for the CNN backbone feature extractor.
#[derive(Debug, Clone)]
pub struct TableBackboneConfig {
    /// Backbone architecture variant.
    pub variant: BackboneVariant,
    /// Number of input image channels (default 3 for RGB).
    pub input_channels: usize,
    /// Whether to freeze backbone weights during fine-tuning (default true).
    pub freeze_backbone: bool,
    /// Output feature map channels from the backbone's final stage.
    pub output_channels: usize,
    /// Whether to use pretrained ImageNet weights (default true).
    pub pretrained: bool,
    /// Dilation rates for the last backbone stage (default `[1, 1]`).
    /// Using `[1, 2]` enables dilated C5 for higher-resolution features.
    pub dilation_rates: [usize; 2],
}

impl Default for TableBackboneConfig {
    fn default() -> Self {
        Self {
            variant: BackboneVariant::ResNet18,
            input_channels: 3,
            freeze_backbone: true,
            output_channels: 512,
            pretrained: true,
            dilation_rates: [1, 1],
        }
    }
}

impl TableBackboneConfig {
    /// Config for ResNet-18 backbone.
    #[must_use]
    pub fn resnet18() -> Self {
        Self {
            variant: BackboneVariant::ResNet18,
            output_channels: 512,
            ..Default::default()
        }
    }

    /// Config for ResNet-50 backbone.
    #[must_use]
    pub fn resnet50() -> Self {
        Self {
            variant: BackboneVariant::ResNet50,
            output_channels: 2048,
            ..Default::default()
        }
    }

    /// Validate the backbone configuration.
    pub fn validate(&self) -> Result<()> {
        if self.input_channels == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "TableBackboneConfig: input_channels must be > 0",
            });
        }
        if self.output_channels == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "TableBackboneConfig: output_channels must be > 0",
            });
        }
        for (i, &rate) in self.dilation_rates.iter().enumerate() {
            if rate == 0 {
                return Err(TensorError::ValueOutOfRange {
                    description: match i {
                        0 => "TableBackboneConfig: dilation_rates[0] must be > 0",
                        _ => "TableBackboneConfig: dilation_rates[1] must be > 0",
                    },
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Decoder config
// ---------------------------------------------------------------------------

/// Configuration for the DETR transformer decoder head.
#[derive(Debug, Clone)]
pub struct TableDecoderConfig {
    /// Hidden dimension of the transformer (default 256).
    pub hidden_dim: usize,
    /// Number of attention heads (default 8).
    pub num_heads: usize,
    /// Number of decoder layers (default 6).
    pub num_layers: usize,
    /// FFN intermediate dimension (default 2048).
    pub ffn_dim: usize,
    /// Number of learned object queries (default 125).
    pub num_queries: usize,
    /// Dropout rate in transformer layers (default 0.1).
    pub dropout: f32,
    /// Whether to use auxiliary decoding loss at each layer (default true).
    pub aux_loss: bool,
}

impl Default for TableDecoderConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 256,
            num_heads: 8,
            num_layers: 6,
            ffn_dim: 2048,
            num_queries: 125,
            dropout: 0.1,
            aux_loss: true,
        }
    }
}

impl TableDecoderConfig {
    /// Validate the decoder configuration.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_dim == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "TableDecoderConfig: hidden_dim must be > 0",
            });
        }
        if self.num_heads == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "TableDecoderConfig: num_heads must be > 0",
            });
        }
        if !self.hidden_dim.is_multiple_of(self.num_heads) {
            return Err(TensorError::ValueOutOfRange {
                description: "TableDecoderConfig: hidden_dim must be divisible by num_heads",
            });
        }
        if self.num_layers == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "TableDecoderConfig: num_layers must be > 0",
            });
        }
        if self.num_queries == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "TableDecoderConfig: num_queries must be > 0",
            });
        }
        if !(0.0..=1.0).contains(&self.dropout) || !self.dropout.is_finite() {
            return Err(TensorError::ValueOutOfRange {
                description: "TableDecoderConfig: dropout must be in [0, 1]",
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Post-processing config
// ---------------------------------------------------------------------------

/// Post-processing thresholds for table detection / structure outputs.
#[derive(Debug, Clone)]
pub struct TablePostProcessConfig {
    /// Confidence threshold for table detection (default 0.5).
    pub table_confidence: f32,
    /// Confidence threshold for row/column detection (default 0.5).
    pub structure_confidence: f32,
    /// IoU threshold for NMS across table detections (default 0.5).
    pub nms_iou_threshold: f32,
    /// Maximum number of detections to keep after NMS (default 100).
    pub max_detections: usize,
    /// Whether to apply class-aware NMS (default true).
    pub class_aware_nms: bool,
}

impl Default for TablePostProcessConfig {
    fn default() -> Self {
        Self {
            table_confidence: 0.5,
            structure_confidence: 0.5,
            nms_iou_threshold: 0.5,
            max_detections: 100,
            class_aware_nms: true,
        }
    }
}

impl TablePostProcessConfig {
    /// Validate the post-processing configuration.
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.table_confidence) || !self.table_confidence.is_finite() {
            return Err(TensorError::ValueOutOfRange {
                description: "TablePostProcessConfig: table_confidence must be in [0, 1]",
            });
        }
        if !(0.0..=1.0).contains(&self.structure_confidence)
            || !self.structure_confidence.is_finite()
        {
            return Err(TensorError::ValueOutOfRange {
                description: "TablePostProcessConfig: structure_confidence must be in [0, 1]",
            });
        }
        if !(0.0..=1.0).contains(&self.nms_iou_threshold) || !self.nms_iou_threshold.is_finite() {
            return Err(TensorError::ValueOutOfRange {
                description: "TablePostProcessConfig: nms_iou_threshold must be in [0, 1]",
            });
        }
        if self.max_detections == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "TablePostProcessConfig: max_detections must be > 0",
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Composite model config
// ---------------------------------------------------------------------------

/// Complete configuration for a Table Transformer model.
///
/// Combines backbone, decoder, and post-processing settings into a single
/// validated configuration struct.
#[derive(Debug, Clone)]
pub struct TableStructureModelConfig {
    /// Backbone feature extractor config.
    pub backbone: TableBackboneConfig,
    /// Transformer decoder head config.
    pub decoder: TableDecoderConfig,
    /// Post-processing thresholds.
    pub postprocess: TablePostProcessConfig,
    /// Number of output classes (including no-object).
    pub num_classes: usize,
    /// Input image size (square, default 800).
    pub input_size: usize,
}

impl TableStructureModelConfig {
    /// Preset for table detection (2 classes: table + no-object).
    #[must_use]
    pub fn preset_detection() -> Self {
        Self {
            backbone: TableBackboneConfig::resnet18(),
            decoder: TableDecoderConfig::default(),
            postprocess: TablePostProcessConfig::default(),
            num_classes: 2,
            input_size: 800,
        }
    }

    /// Preset for table structure recognition (6 classes).
    ///
    /// Classes: table, row, column, spanning-cell, projected-row-header, no-object.
    #[must_use]
    pub fn preset_structure() -> Self {
        Self {
            backbone: TableBackboneConfig::resnet18(),
            decoder: TableDecoderConfig::default(),
            postprocess: TablePostProcessConfig {
                table_confidence: 0.5,
                structure_confidence: 0.3,
                ..Default::default()
            },
            num_classes: 6,
            input_size: 800,
        }
    }

    /// Validate all sub-configurations.
    pub fn validate(&self) -> Result<()> {
        self.backbone.validate()?;
        self.decoder.validate()?;
        self.postprocess.validate()?;
        if self.num_classes == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "TableStructureModelConfig: num_classes must be > 0",
            });
        }
        if self.input_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "TableStructureModelConfig: input_size must be > 0",
            });
        }
        Ok(())
    }

    /// Compute the number of feature map positions after backbone + projection.
    ///
    /// For a ResNet backbone with stride 32, the feature map spatial size is
    /// `input_size / 32`, so the total sequence length is `(input_size / 32)^2`.
    #[must_use]
    pub fn feature_sequence_length(&self) -> usize {
        let spatial = self.input_size / 32;
        spatial * spatial
    }
}

#[cfg(test)]
#[path = "table_structure_model_config_tests.rs"]
mod tests;
