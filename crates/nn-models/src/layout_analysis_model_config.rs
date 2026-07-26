// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Configuration for document layout analysis models.
//!
//! Defines inference-time configuration for DocLayout-YOLO style multi-scale
//! detection models used in the dpdf pipeline. This extends the existing
//! [`super::doclayout_yolo::DocLayoutYoloConfig`] with additional parameters
//! for multi-resolution processing, anchor-free detection heads, and
//! document-specific augmentation policies.
//!
//! # Multi-Scale Detection
//!
//! Document layout detection benefits from multi-scale processing because
//! page elements range from small footnotes to full-width tables. The
//! model processes feature maps at 3 detection scales (stride 8, 16, 32)
//! through a PAN (Path Aggregation Network) neck:
//!
//! ```text
//!   Backbone C3(stride 8) ──┐
//!   Backbone C4(stride 16) ─┤─→ PAN Neck ─→ DetectHead ─→ Detections
//!   Backbone C5(stride 32) ─┘
//! ```
//!
//! Reference: Zhao et al. 2024, "DocLayout-YOLO: Enhancing Document Layout
//! Analysis through Diverse Synthetic Data and Global-to-Local Adaptive
//! Perception", arXiv:2410.12628.

use nn_core::{Result, TensorError};

// ---------------------------------------------------------------------------
// Multi-scale detection config
// ---------------------------------------------------------------------------

/// Configuration for multi-scale detection head parameters.
#[derive(Debug, Clone)]
pub struct MultiScaleDetectionConfig {
    /// Detection strides for each output scale (default `[8, 16, 32]`).
    pub strides: Vec<usize>,
    /// Channel width per detection scale head (default 256).
    pub head_channels: usize,
    /// DFL (Distribution Focal Loss) regression bin count (default 16).
    pub reg_max: usize,
    /// Whether to share classification weights across scales (default false).
    pub share_cls_weights: bool,
    /// Whether to share regression weights across scales (default false).
    pub share_reg_weights: bool,
}

impl Default for MultiScaleDetectionConfig {
    fn default() -> Self {
        Self {
            strides: vec![8, 16, 32],
            head_channels: 256,
            reg_max: 16,
            share_cls_weights: false,
            share_reg_weights: false,
        }
    }
}

impl MultiScaleDetectionConfig {
    /// Validate the multi-scale detection configuration.
    pub fn validate(&self) -> Result<()> {
        if self.strides.is_empty() {
            return Err(TensorError::ValueOutOfRange {
                description: "MultiScaleDetectionConfig: strides must not be empty",
            });
        }
        for (i, &stride) in self.strides.iter().enumerate() {
            if stride == 0 {
                return Err(TensorError::ValueOutOfRange {
                    description: match i {
                        0 => "MultiScaleDetectionConfig: strides[0] must be > 0",
                        1 => "MultiScaleDetectionConfig: strides[1] must be > 0",
                        _ => "MultiScaleDetectionConfig: all strides must be > 0",
                    },
                });
            }
            if !stride.is_power_of_two() {
                return Err(TensorError::ValueOutOfRange {
                    description: "MultiScaleDetectionConfig: strides must be powers of 2",
                });
            }
        }
        // Strides should be strictly increasing.
        for pair in self.strides.windows(2) {
            if pair[0] >= pair[1] {
                return Err(TensorError::ValueOutOfRange {
                    description: "MultiScaleDetectionConfig: strides must be strictly increasing",
                });
            }
        }
        if self.head_channels == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "MultiScaleDetectionConfig: head_channels must be > 0",
            });
        }
        if self.reg_max == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "MultiScaleDetectionConfig: reg_max must be > 0",
            });
        }
        Ok(())
    }

    /// Compute the total number of anchor points across all scales.
    ///
    /// For a square input of `input_size`, each scale produces
    /// `(input_size / stride)^2` anchor points.
    #[must_use]
    pub fn total_anchors(&self, input_size: usize) -> usize {
        self.strides
            .iter()
            .map(|&s| {
                let grid = input_size / s;
                grid * grid
            })
            .sum()
    }
}

// ---------------------------------------------------------------------------
// PAN neck config
// ---------------------------------------------------------------------------

/// Configuration for the PAN (Path Aggregation Network) neck.
#[derive(Debug, Clone)]
pub struct PanNeckConfig {
    /// Channel widths for each backbone feature level (e.g., `[64, 128, 256]`).
    pub backbone_channels: Vec<usize>,
    /// Output channel width after PAN fusion (default 256).
    pub output_channels: usize,
    /// Number of CSP (Cross-Stage Partial) bottleneck blocks per level (default 3).
    pub csp_depth: usize,
    /// Whether to use depthwise separable convolutions (default false).
    /// Reduces parameters at the cost of some accuracy.
    pub depthwise: bool,
}

impl Default for PanNeckConfig {
    fn default() -> Self {
        Self {
            backbone_channels: vec![64, 128, 256],
            output_channels: 256,
            csp_depth: 3,
            depthwise: false,
        }
    }
}

impl PanNeckConfig {
    /// Validate the PAN neck configuration.
    pub fn validate(&self) -> Result<()> {
        if self.backbone_channels.is_empty() {
            return Err(TensorError::ValueOutOfRange {
                description: "PanNeckConfig: backbone_channels must not be empty",
            });
        }
        for &ch in &self.backbone_channels {
            if ch == 0 {
                return Err(TensorError::ValueOutOfRange {
                    description: "PanNeckConfig: all backbone_channels must be > 0",
                });
            }
        }
        if self.output_channels == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "PanNeckConfig: output_channels must be > 0",
            });
        }
        if self.csp_depth == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "PanNeckConfig: csp_depth must be > 0",
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Document-specific preprocessing config
// ---------------------------------------------------------------------------

/// Document-specific input preprocessing settings.
#[derive(Debug, Clone)]
pub struct DocumentPreprocessConfig {
    /// Input resolution (square, default 800).
    pub input_size: usize,
    /// Whether to preserve aspect ratio with letterboxing (default true).
    pub letterbox: bool,
    /// Letterbox fill color (RGB, default `[114, 114, 114]`).
    pub fill_color: [u8; 3],
    /// Channel-wise mean for normalization (default ImageNet: `[0.485, 0.456, 0.406]`).
    pub mean: [f32; 3],
    /// Channel-wise std for normalization (default ImageNet: `[0.229, 0.224, 0.225]`).
    pub std: [f32; 3],
    /// Maximum image dimension before downscaling (default 2048).
    /// Pages larger than this are resized before model input.
    pub max_dimension: usize,
}

impl Default for DocumentPreprocessConfig {
    fn default() -> Self {
        Self {
            input_size: 800,
            letterbox: true,
            fill_color: [114, 114, 114],
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            max_dimension: 2048,
        }
    }
}

impl DocumentPreprocessConfig {
    /// Validate preprocessing parameters.
    pub fn validate(&self) -> Result<()> {
        if self.input_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "DocumentPreprocessConfig: input_size must be > 0",
            });
        }
        for (i, &s) in self.std.iter().enumerate() {
            if s <= 0.0 || !s.is_finite() {
                return Err(TensorError::ValueOutOfRange {
                    description: match i {
                        0 => "DocumentPreprocessConfig: std[0] must be > 0 and finite",
                        1 => "DocumentPreprocessConfig: std[1] must be > 0 and finite",
                        _ => "DocumentPreprocessConfig: std[2] must be > 0 and finite",
                    },
                });
            }
        }
        for (i, &m) in self.mean.iter().enumerate() {
            if !m.is_finite() {
                return Err(TensorError::ValueOutOfRange {
                    description: match i {
                        0 => "DocumentPreprocessConfig: mean[0] must be finite",
                        1 => "DocumentPreprocessConfig: mean[1] must be finite",
                        _ => "DocumentPreprocessConfig: mean[2] must be finite",
                    },
                });
            }
        }
        if self.max_dimension == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "DocumentPreprocessConfig: max_dimension must be > 0",
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Composite layout analysis config
// ---------------------------------------------------------------------------

/// Complete configuration for a document layout analysis model.
///
/// Combines backbone, PAN neck, multi-scale detection head, and
/// preprocessing settings.
#[derive(Debug, Clone)]
pub struct LayoutAnalysisModelConfig {
    /// Number of input channels (default 3 for RGB).
    pub input_channels: usize,
    /// Backbone channel widths per stage (e.g., `[16, 32, 64, 128, 256]`).
    pub backbone_channels: Vec<usize>,
    /// PAN neck configuration.
    pub neck: PanNeckConfig,
    /// Multi-scale detection head configuration.
    pub detection: MultiScaleDetectionConfig,
    /// Document preprocessing configuration.
    pub preprocess: DocumentPreprocessConfig,
    /// Number of layout classes (default 10).
    pub num_classes: usize,
    /// Class names for human-readable output.
    pub class_names: Vec<String>,
}

impl LayoutAnalysisModelConfig {
    /// DocLayout-YOLO nano preset (10 classes, YOLOv8-nano backbone).
    #[must_use]
    pub fn preset_doclayout_yolo_nano() -> Self {
        Self {
            input_channels: 3,
            backbone_channels: vec![16, 32, 64, 128, 256],
            neck: PanNeckConfig {
                backbone_channels: vec![64, 128, 256],
                output_channels: 256,
                csp_depth: 1,
                depthwise: false,
            },
            detection: MultiScaleDetectionConfig::default(),
            preprocess: DocumentPreprocessConfig::default(),
            num_classes: 10,
            class_names: vec![
                "caption".into(),
                "footnote".into(),
                "formula".into(),
                "list-item".into(),
                "page-footer".into(),
                "page-header".into(),
                "picture".into(),
                "section-header".into(),
                "table".into(),
                "text".into(),
            ],
        }
    }

    /// DocLayout-YOLO small preset (higher accuracy, more parameters).
    #[must_use]
    pub fn preset_doclayout_yolo_small() -> Self {
        Self {
            input_channels: 3,
            backbone_channels: vec![32, 64, 128, 256, 512],
            neck: PanNeckConfig {
                backbone_channels: vec![128, 256, 512],
                output_channels: 256,
                csp_depth: 2,
                depthwise: false,
            },
            detection: MultiScaleDetectionConfig::default(),
            preprocess: DocumentPreprocessConfig {
                input_size: 1024,
                ..Default::default()
            },
            num_classes: 10,
            class_names: vec![
                "caption".into(),
                "footnote".into(),
                "formula".into(),
                "list-item".into(),
                "page-footer".into(),
                "page-header".into(),
                "picture".into(),
                "section-header".into(),
                "table".into(),
                "text".into(),
            ],
        }
    }

    /// Validate all sub-configurations.
    pub fn validate(&self) -> Result<()> {
        if self.input_channels == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "LayoutAnalysisModelConfig: input_channels must be > 0",
            });
        }
        if self.backbone_channels.is_empty() {
            return Err(TensorError::ValueOutOfRange {
                description: "LayoutAnalysisModelConfig: backbone_channels must not be empty",
            });
        }
        for &ch in &self.backbone_channels {
            if ch == 0 {
                return Err(TensorError::ValueOutOfRange {
                    description: "LayoutAnalysisModelConfig: all backbone_channels must be > 0",
                });
            }
        }
        if self.num_classes == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "LayoutAnalysisModelConfig: num_classes must be > 0",
            });
        }
        if self.class_names.len() != self.num_classes {
            return Err(TensorError::ValueOutOfRange {
                description: "LayoutAnalysisModelConfig: class_names length must equal num_classes",
            });
        }
        self.neck.validate()?;
        self.detection.validate()?;
        self.preprocess.validate()?;
        Ok(())
    }

    /// Compute total anchor points across all detection scales.
    #[must_use]
    pub fn total_anchors(&self) -> usize {
        self.detection.total_anchors(self.preprocess.input_size)
    }

    /// Compute the DFL output dimension per anchor: `4 * reg_max`.
    #[must_use]
    pub fn dfl_output_dim(&self) -> usize {
        4 * self.detection.reg_max
    }
}

#[cfg(test)]
#[path = "layout_analysis_model_config_tests.rs"]
mod tests;
