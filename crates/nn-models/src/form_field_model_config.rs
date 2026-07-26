// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Configuration for form key-value detection models.
//!
//! Defines the inference-time configuration for LayoutLMv3-based form
//! understanding models used in the dpdf pipeline. The model has two
//! logical heads:
//!
//! - **Field detection head**: token-level BIO sequence labeling to
//!   identify question, answer, and header spans.
//! - **Value extraction head**: optional linking head that predicts
//!   key-value associations directly, bypassing spatial heuristics.
//!
//! # Architecture
//!
//! The field detection head is a linear classifier on top of the
//! LayoutLMv3 hidden states (`hidden_size -> num_entity_labels`).
//! The optional value extraction head adds a biaffine attention layer
//! that scores all (key, value) span pairs for linking.
//!
//! Reference: Huang et al. 2022, "LayoutLMv3: Pre-training for Document
//! AI with Unified Text and Image Masking", ACM MM 2022.

use nn_core::{Result, TensorError};

// ---------------------------------------------------------------------------
// Field detection head config
// ---------------------------------------------------------------------------

/// Configuration for the token-level BIO entity labeling head.
#[derive(Debug, Clone)]
pub struct FieldDetectionHeadConfig {
    /// Hidden size from the backbone encoder (default 768).
    pub hidden_size: usize,
    /// Number of BIO entity labels (default 7: O, B-Q, I-Q, B-A, I-A, B-H, I-H).
    pub num_labels: usize,
    /// Dropout applied before the classification head (default 0.1).
    pub classifier_dropout: f32,
    /// Whether to use CRF layer on top of the linear classifier (default false).
    /// CRF improves label consistency (e.g., I-Q never follows B-A) but adds
    /// inference-time Viterbi decoding cost.
    pub use_crf: bool,
}

impl Default for FieldDetectionHeadConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            num_labels: 7,
            classifier_dropout: 0.1,
            use_crf: false,
        }
    }
}

impl FieldDetectionHeadConfig {
    /// Preset for FUNSD entity labeling (7 BIO labels).
    #[must_use]
    pub fn preset_funsd() -> Self {
        Self::default()
    }

    /// Preset for CORD receipt key-value extraction (30 BIO labels).
    #[must_use]
    pub fn preset_cord() -> Self {
        Self {
            num_labels: 30,
            ..Default::default()
        }
    }

    /// Validate the field detection head configuration.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "FieldDetectionHeadConfig: hidden_size must be > 0",
            });
        }
        if self.num_labels == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "FieldDetectionHeadConfig: num_labels must be > 0",
            });
        }
        if !(0.0..=1.0).contains(&self.classifier_dropout) || !self.classifier_dropout.is_finite() {
            return Err(TensorError::ValueOutOfRange {
                description: "FieldDetectionHeadConfig: classifier_dropout must be in [0, 1]",
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Value extraction head config
// ---------------------------------------------------------------------------

/// Configuration for the biaffine key-value linking head.
///
/// This head scores (key_span, value_span) pairs using biaffine attention:
/// `score(k, v) = k^T W v + b`, where `k` and `v` are span representations
/// derived from the backbone hidden states.
#[derive(Debug, Clone)]
pub struct ValueExtractionHeadConfig {
    /// Hidden size of the span representations (default 768).
    pub hidden_size: usize,
    /// Reduced dimension for biaffine attention (default 128).
    /// The span vectors are projected from `hidden_size` to `biaffine_dim`
    /// before the biaffine product.
    pub biaffine_dim: usize,
    /// Maximum number of key-value links per page (default 64).
    pub max_links: usize,
    /// Minimum score threshold for accepting a link (default 0.5).
    pub link_threshold: f32,
    /// Whether to use span width embedding (default true).
    pub use_width_embedding: bool,
    /// Maximum span width for width embedding (default 8 tokens).
    pub max_span_width: usize,
}

impl Default for ValueExtractionHeadConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            biaffine_dim: 128,
            max_links: 64,
            link_threshold: 0.5,
            use_width_embedding: true,
            max_span_width: 8,
        }
    }
}

impl ValueExtractionHeadConfig {
    /// Validate the value extraction head configuration.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "ValueExtractionHeadConfig: hidden_size must be > 0",
            });
        }
        if self.biaffine_dim == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "ValueExtractionHeadConfig: biaffine_dim must be > 0",
            });
        }
        if self.biaffine_dim > self.hidden_size {
            return Err(TensorError::ValueOutOfRange {
                description: "ValueExtractionHeadConfig: biaffine_dim must be <= hidden_size",
            });
        }
        if self.max_links == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "ValueExtractionHeadConfig: max_links must be > 0",
            });
        }
        if !(0.0..=1.0).contains(&self.link_threshold) || !self.link_threshold.is_finite() {
            return Err(TensorError::ValueOutOfRange {
                description: "ValueExtractionHeadConfig: link_threshold must be in [0, 1]",
            });
        }
        if self.use_width_embedding && self.max_span_width == 0 {
            return Err(TensorError::ValueOutOfRange {
                description:
                    "ValueExtractionHeadConfig: max_span_width must be > 0 when width embedding is enabled",
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Composite model config
// ---------------------------------------------------------------------------

/// Complete configuration for a form key-value detection model.
///
/// Combines the LayoutLMv3 backbone config with the field detection head
/// and optional value extraction head.
#[derive(Debug, Clone)]
pub struct FormFieldModelConfig {
    /// LayoutLMv3 backbone hidden size.
    pub hidden_size: usize,
    /// Number of transformer layers in the backbone (default 12).
    pub num_layers: usize,
    /// Number of attention heads (default 12).
    pub num_heads: usize,
    /// Vocabulary size (default 50265).
    pub vocab_size: usize,
    /// Maximum 2D position coordinate (default 1024).
    pub max_2d_pos: usize,
    /// Input image size for visual features (default 224).
    pub image_size: usize,
    /// Patch size for visual tokenization (default 16).
    pub patch_size: usize,
    /// Field detection (BIO labeling) head.
    pub field_head: FieldDetectionHeadConfig,
    /// Optional key-value linking head.
    pub value_head: Option<ValueExtractionHeadConfig>,
}

impl FormFieldModelConfig {
    /// Preset for FUNSD form understanding (BIO labeling only).
    #[must_use]
    pub fn preset_funsd() -> Self {
        Self {
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            vocab_size: 50_265,
            max_2d_pos: 1024,
            image_size: 224,
            patch_size: 16,
            field_head: FieldDetectionHeadConfig::preset_funsd(),
            value_head: None,
        }
    }

    /// Preset for FUNSD with biaffine linking head.
    #[must_use]
    pub fn preset_funsd_with_linking() -> Self {
        let mut config = Self::preset_funsd();
        config.value_head = Some(ValueExtractionHeadConfig::default());
        config
    }

    /// Preset for CORD receipt extraction.
    #[must_use]
    pub fn preset_cord() -> Self {
        Self {
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            vocab_size: 50_265,
            max_2d_pos: 1024,
            image_size: 224,
            patch_size: 16,
            field_head: FieldDetectionHeadConfig::preset_cord(),
            value_head: None,
        }
    }

    /// Validate all sub-configurations.
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "FormFieldModelConfig: hidden_size must be > 0",
            });
        }
        if self.num_heads == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "FormFieldModelConfig: num_heads must be > 0",
            });
        }
        if !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(TensorError::ValueOutOfRange {
                description: "FormFieldModelConfig: hidden_size must be divisible by num_heads",
            });
        }
        if self.num_layers == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "FormFieldModelConfig: num_layers must be > 0",
            });
        }
        if self.patch_size == 0 || self.image_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "FormFieldModelConfig: patch_size and image_size must be > 0",
            });
        }
        if !self.image_size.is_multiple_of(self.patch_size) {
            return Err(TensorError::ValueOutOfRange {
                description: "FormFieldModelConfig: image_size must be divisible by patch_size",
            });
        }
        self.field_head.validate()?;
        if let Some(ref vh) = self.value_head {
            if vh.hidden_size != self.hidden_size {
                return Err(TensorError::ValueOutOfRange {
                    description:
                        "FormFieldModelConfig: value_head hidden_size must match backbone hidden_size",
                });
            }
            vh.validate()?;
        }
        Ok(())
    }

    /// Compute the visual sequence length (number of image patch tokens).
    #[must_use]
    pub fn visual_seq_len(&self) -> usize {
        let grid = self.image_size / self.patch_size;
        grid * grid
    }
}

#[cfg(test)]
#[path = "form_field_model_config_tests.rs"]
mod tests;
