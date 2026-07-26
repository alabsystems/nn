// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! dpdf model registry and dispatch routing.
//!
//! Maintains a registry of document-processing models with their metadata,
//! preprocessing configurations, and parameter counts. Supports lookup by
//! name and filtering by [`ModelType`].
//!
//! # Usage
//!
//! ```rust
//! use nn_models::dpdf_registry::{DpdfModelRegistry, ModelType};
//!
//! let registry = DpdfModelRegistry::default_pipeline();
//! assert_eq!(registry.list_by_type(ModelType::OCR).len(), 2);
//!
//! let entry = registry.get("granite_docling").unwrap();
//! assert_eq!(entry.model_type, ModelType::VLM);
//! ```

use std::collections::HashMap;

use crate::dpdf_image_preprocess::DpdfPreprocessConfig;

// ---------------------------------------------------------------------------
// Model type classification
// ---------------------------------------------------------------------------

/// Classification of a dpdf model by its role in the document pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelType {
    /// Full-page layout detection (bounding boxes + class labels).
    LayoutDetection,
    /// Optical character recognition (text extraction from image crops).
    OCR,
    /// Table structure recognition (rows, columns, cells).
    TableStructure,
    /// Vision-language model (multimodal understanding / generation).
    VLM,
}

impl ModelType {
    /// Human-readable label for this model type.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::LayoutDetection => "Layout Detection",
            Self::OCR => "OCR",
            Self::TableStructure => "Table Structure",
            Self::VLM => "VLM",
        }
    }
}

// ---------------------------------------------------------------------------
// Model entry
// ---------------------------------------------------------------------------

/// Metadata for a single model in the dpdf registry.
#[derive(Debug, Clone)]
pub struct ModelEntry {
    /// Unique identifier (e.g. `"granite_docling"`).
    pub name: String,
    /// Pipeline role of this model.
    pub model_type: ModelType,
    /// Short human-readable description.
    pub description: String,
    /// Image preprocessing configuration for this model's input contract.
    pub preprocess_config: DpdfPreprocessConfig,
    /// Approximate trainable parameter count.
    pub parameter_count: usize,
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Registry of dpdf document-processing models.
///
/// Wraps a name-keyed map of [`ModelEntry`] values. Use
/// [`default_pipeline`](Self::default_pipeline) to get a pre-populated
/// registry with all 8 standard dpdf models.
#[derive(Debug, Clone)]
pub struct DpdfModelRegistry {
    entries: HashMap<String, ModelEntry>,
}

impl DpdfModelRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Register a model entry. Overwrites any existing entry with the same name.
    pub fn register(&mut self, entry: ModelEntry) {
        self.entries.insert(entry.name.clone(), entry);
    }

    /// Look up a model by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&ModelEntry> {
        self.entries.get(name)
    }

    /// Return all models matching the given type.
    #[must_use]
    pub fn list_by_type(&self, model_type: ModelType) -> Vec<&ModelEntry> {
        self.entries
            .values()
            .filter(|e| e.model_type == model_type)
            .collect()
    }

    /// Iterator over all registered model entries.
    pub fn models(&self) -> impl Iterator<Item = &ModelEntry> {
        self.entries.values()
    }

    /// Number of registered models.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Create a registry pre-populated with all 8 standard dpdf models.
    ///
    /// | Name | Type | Parameters |
    /// |------|------|-----------|
    /// | `granite_docling` | VLM | 258M |
    /// | `doclayout_yolo` | LayoutDetection | 16M |
    /// | `glm_ocr` | OCR | 900M |
    /// | `table_transformer` | TableStructure | 28.8M |
    /// | `qwen3_vl` | VLM | 8B |
    /// | `paddle_ocr` | VLM | ~1.3B |
    /// | `firered_ocr` | OCR | 2B |
    /// | `rt_detr_heron` | LayoutDetection | 42.9M |
    #[must_use]
    pub fn default_pipeline() -> Self {
        let mut registry = Self::new();

        registry.register(ModelEntry {
            name: "granite_docling".into(),
            model_type: ModelType::VLM,
            description: "Granite-Docling-258M: SigLIP2 vision encoder + Granite-165M decoder"
                .into(),
            preprocess_config: DpdfPreprocessConfig::for_granite_docling(),
            parameter_count: 258_000_000,
        });

        registry.register(ModelEntry {
            name: "doclayout_yolo".into(),
            model_type: ModelType::LayoutDetection,
            description: "DocLayout-YOLO: 10-class document layout detection".into(),
            preprocess_config: DpdfPreprocessConfig::for_doclayout_yolo(),
            parameter_count: 16_000_000,
        });

        registry.register(ModelEntry {
            name: "glm_ocr".into(),
            model_type: ModelType::OCR,
            description: "GLM-OCR 0.9B: multi-token prediction document OCR".into(),
            preprocess_config: DpdfPreprocessConfig::for_glm_ocr(),
            parameter_count: 900_000_000,
        });

        registry.register(ModelEntry {
            name: "table_transformer".into(),
            model_type: ModelType::TableStructure,
            description: "Table Transformer (DETR): table detection and structure recognition"
                .into(),
            preprocess_config: DpdfPreprocessConfig::for_table_transformer(),
            parameter_count: 28_800_000,
        });

        registry.register(ModelEntry {
            name: "qwen3_vl".into(),
            model_type: ModelType::VLM,
            description: "Qwen3-VL-8B: multimodal vision-language model".into(),
            preprocess_config: DpdfPreprocessConfig::for_qwen3_vl(),
            parameter_count: 8_000_000_000,
        });

        registry.register(ModelEntry {
            name: "paddle_ocr".into(),
            model_type: ModelType::VLM,
            description: "PaddleOCR-VL-1.5: SigLIP ViT + ERNIE-4.5 GQA decoder".into(),
            preprocess_config: DpdfPreprocessConfig::for_paddle_ocr_detect(),
            parameter_count: 1_300_000_000,
        });

        registry.register(ModelEntry {
            name: "firered_ocr".into(),
            model_type: ModelType::OCR,
            description: "FireRed-OCR (Qwen3-VL-2B): high-accuracy document OCR".into(),
            // FireRed-OCR is based on Qwen3-VL-2B; uses the Qwen3-VL
            // dynamic-resolution preprocessing pipeline.
            preprocess_config: DpdfPreprocessConfig::for_qwen3_vl(),
            parameter_count: 2_000_000_000,
        });

        registry.register(ModelEntry {
            name: "rt_detr_heron".into(),
            model_type: ModelType::LayoutDetection,
            description: "RT-DETRv2 (Heron): 17-class document layout detection transformer".into(),
            preprocess_config: DpdfPreprocessConfig::for_rt_detr(),
            parameter_count: 42_900_000,
        });

        registry
    }
}

impl Default for DpdfModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "dpdf_registry_tests.rs"]
mod tests;
