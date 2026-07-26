// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! DynTensor-based model forward passes for the dpdf pipeline.
//!
//! Provides [`DpdfModelWeights`] for holding loaded (or synthetic) model
//! weights, and actual forward-pass implementations for layout detection,
//! OCR, and table structure recognition within [`DpdfPipeline`].

use nn_core::dyn_tensor::DynTensor;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device, Result, TensorError};

use crate::doclayout_yolo::{DocLayoutYolo, DocLayoutYoloConfig};
use crate::dpdf_pipeline::{DocumentRegion, DpdfPipeline, PageOutput, PipelineConfig};
use crate::firered_ocr::FireRedOcr;
use crate::glm_ocr::{GlmOcr, GlmOcrConfig};
use crate::granite_docling::{GraniteDocling, GraniteDoclingConfig};
use crate::paddle_ocr::PaddleOcrVl;
use crate::rt_detr::RtDetr;
use crate::table_transformer::{TableTransformer, TableTransformerConfig};

// ---------------------------------------------------------------------------
// Weight structs
// ---------------------------------------------------------------------------

/// Loaded model weights for the dpdf pipeline's models.
///
/// Each field is `Option` because the pipeline can run with a subset of
/// models (e.g., layout-only without OCR or table structure).
#[derive(Clone)]
pub struct DpdfModelWeights {
    /// DocLayout-YOLO layout detection model.
    pub layout_model: Option<DocLayoutYolo>,
    /// GLM-OCR text recognition model (0.9B, MTP-capable).
    ///
    /// Note: GLM-OCR is available but dpdf's current pipeline uses
    /// PaddleOCR-VL (`paddle_ocr_model`) for Tier 1B OCR instead.
    pub ocr_model: Option<GlmOcr>,
    /// Granite-Docling-258M OCR model (SigLIP2 + Granite-165M decoder).
    pub granite_docling_model: Option<GraniteDocling>,
    /// Table Transformer (DETR) structure recognition model.
    pub table_model: Option<TableTransformer>,
    /// PaddleOCR-VL-1.5 detection + recognition model (Tier 1B OCR).
    pub paddle_ocr_model: Option<PaddleOcrVl>,
    /// FireRed-OCR (Qwen3-VL-2B fine-tuned) document OCR model (WIP).
    pub firered_ocr_model: Option<FireRedOcr>,
    /// RT-DETRv2 (Heron) layout detection model (docling_rs integration).
    pub rt_detr_model: Option<RtDetr>,
}

impl std::fmt::Debug for DpdfModelWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DpdfModelWeights")
            .field("has_layout", &self.layout_model.is_some())
            .field("has_ocr", &self.ocr_model.is_some())
            .field("has_granite_docling", &self.granite_docling_model.is_some())
            .field("has_table", &self.table_model.is_some())
            .field("has_paddle_ocr", &self.paddle_ocr_model.is_some())
            .field("has_firered_ocr", &self.firered_ocr_model.is_some())
            .field("has_rt_detr", &self.rt_detr_model.is_some())
            .finish()
    }
}

impl DpdfModelWeights {
    /// Create empty weights (no models loaded).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            layout_model: None,
            ocr_model: None,
            granite_docling_model: None,
            table_model: None,
            paddle_ocr_model: None,
            firered_ocr_model: None,
            rt_detr_model: None,
        }
    }

    /// Create synthetic (zero) weights for all models.
    ///
    /// Loads layout (DocLayout-YOLO), OCR (GLM-OCR), and table structure
    /// (Table Transformer) with zero-filled tensors on CPU. Useful for
    /// shape-validation tests without real trained weights.
    pub fn synthetic() -> Result<Self> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let layout_cfg = DocLayoutYoloConfig::default();
        let layout_model = DocLayoutYolo::load(&vb, layout_cfg)?;

        let ocr_cfg = GlmOcrConfig::preset_900m();
        let ocr_model = GlmOcr::load(&vb, ocr_cfg)?;

        let table_cfg = TableTransformerConfig::preset_structure();
        let table_model = TableTransformer::load(&vb, &table_cfg)?;

        Ok(Self {
            layout_model: Some(layout_model),
            ocr_model: Some(ocr_model),
            granite_docling_model: None,
            table_model: Some(table_model),
            paddle_ocr_model: None,
            firered_ocr_model: None,
            rt_detr_model: None,
        })
    }

    /// Create synthetic weights with Granite-Docling as the OCR backend.
    ///
    /// Loads layout (DocLayout-YOLO) and Granite-Docling-258M with
    /// zero-filled tensors. GLM-OCR and Table Transformer are not loaded.
    pub fn synthetic_with_granite_docling() -> Result<Self> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let layout_cfg = DocLayoutYoloConfig::default();
        let layout_model = DocLayoutYolo::load(&vb, layout_cfg)?;

        let gd_cfg = GraniteDoclingConfig::default_258m();
        let granite_docling_model = GraniteDocling::load(&vb, gd_cfg)?;

        Ok(Self {
            layout_model: Some(layout_model),
            ocr_model: None,
            granite_docling_model: Some(granite_docling_model),
            table_model: None,
            paddle_ocr_model: None,
            firered_ocr_model: None,
            rt_detr_model: None,
        })
    }

    /// Create synthetic weights for layout detection only.
    pub fn synthetic_layout_only() -> Result<Self> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let layout_cfg = DocLayoutYoloConfig::default();
        let layout_model = DocLayoutYolo::load(&vb, layout_cfg)?;
        Ok(Self {
            layout_model: Some(layout_model),
            ocr_model: None,
            granite_docling_model: None,
            table_model: None,
            paddle_ocr_model: None,
            firered_ocr_model: None,
            rt_detr_model: None,
        })
    }

    /// Create synthetic weights for table structure recognition only.
    pub fn synthetic_table_only() -> Result<Self> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let table_cfg = TableTransformerConfig::preset_structure();
        let table_model = TableTransformer::load(&vb, &table_cfg)?;
        Ok(Self {
            layout_model: None,
            ocr_model: None,
            granite_docling_model: None,
            table_model: Some(table_model),
            paddle_ocr_model: None,
            firered_ocr_model: None,
            rt_detr_model: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Pipeline with weights
// ---------------------------------------------------------------------------

/// Extended pipeline that holds model weights and runs actual forward passes.
#[derive(Debug, Clone)]
pub struct DpdfInferencePipeline {
    /// Orchestration and post-processing config.
    pub(crate) pipeline: DpdfPipeline,
    /// Model weights.
    pub(crate) weights: DpdfModelWeights,
}

impl DpdfInferencePipeline {
    /// Create an inference pipeline from config and weights.
    #[must_use]
    pub fn new(config: PipelineConfig, weights: DpdfModelWeights) -> Self {
        Self {
            pipeline: DpdfPipeline::new(config),
            weights,
        }
    }

    /// Create an inference pipeline with synthetic (zero) weights.
    pub fn with_synthetic_weights(config: PipelineConfig) -> Result<Self> {
        let weights = DpdfModelWeights::synthetic()?;
        Ok(Self::new(config, weights))
    }

    /// Access the orchestration pipeline.
    #[must_use]
    pub fn pipeline(&self) -> &DpdfPipeline {
        &self.pipeline
    }

    /// Access the model weights.
    #[must_use]
    pub fn weights(&self) -> &DpdfModelWeights {
        &self.weights
    }

    /// Run end-to-end page inference: layout detection -> structured page.
    ///
    /// Input: `[1, 3, H, W]` page image tensor.
    /// Returns: [`PageOutput`] with detected, classified regions and
    /// reading order. Post-processing (confidence filter, merge, dedup)
    /// is applied automatically.
    pub fn process_page(
        &self,
        image: &DynTensor,
        width: usize,
        height: usize,
    ) -> Result<PageOutput> {
        let regions = self.run_layout_detection(image)?;
        Ok(self.pipeline.build_page(regions, width, height))
    }

    /// Run layout detection on an image tensor.
    ///
    /// Input: `[B, 3, H, W]` image tensor (typically `[1, 3, 800, 800]`).
    /// Returns classified [`DocumentRegion`] objects with bounding boxes.
    ///
    /// Falls back to empty detections if no layout model is loaded.
    pub fn run_layout_detection(&self, image: &DynTensor) -> Result<Vec<DocumentRegion>> {
        let model = match &self.weights.layout_model {
            Some(m) => m,
            None => return Ok(Vec::new()),
        };

        // Validate input shape.
        let dims = image.dims();
        if dims.len() != 4 || dims[1] != 3 {
            return Err(TensorError::shape_mismatch(vec![0, 3, 0, 0], dims.to_vec()));
        }

        // Run backbone forward pass to validate shapes propagate.
        let (p3, p4, p5) = model.forward_backbone(image)?;

        // Validate intermediate feature map shapes.
        let h = dims[2];
        let w = dims[3];
        validate_feature_map_shape(&p3, h / 8, w / 8, "P3")?;
        validate_feature_map_shape(&p4, h / 16, w / 16, "P4")?;
        validate_feature_map_shape(&p5, h / 32, w / 32, "P5")?;

        // Run neck + head for full detection.
        let detections = model.forward(image)?;

        // Convert Detection objects to DocumentRegion.
        let regions: Vec<DocumentRegion> = detections
            .iter()
            .map(|det| {
                DpdfPipeline::classify_detection(
                    det.class_id as usize,
                    [det.x1, det.y1, det.x2, det.y2],
                    det.confidence,
                )
            })
            .collect();
        Ok(regions)
    }

    /// Run table structure recognition on an image tensor.
    ///
    /// Input: `[B, 3, H, W]` image tensor.
    /// Returns raw logits and box predictions.
    ///
    /// Falls back to `None` if no table model is loaded.
    pub fn run_table_structure(&self, image: &DynTensor) -> Result<Option<(DynTensor, DynTensor)>> {
        let model = match &self.weights.table_model {
            Some(m) => m,
            None => return Ok(None),
        };

        let output = model.forward(image)?;

        // Validate output shapes.
        let batch = image.dim(0)?;
        let num_queries = model.config().num_queries;
        let num_classes = model.config().num_classes;

        let logit_dims = output.logits.dims();
        if logit_dims != [batch, num_queries, num_classes + 1] {
            return Err(TensorError::shape_mismatch(
                vec![batch, num_queries, num_classes + 1],
                logit_dims.to_vec(),
            ));
        }

        let box_dims = output.boxes.dims();
        if box_dims != [batch, num_queries, 4] {
            return Err(TensorError::shape_mismatch(
                vec![batch, num_queries, 4],
                box_dims.to_vec(),
            ));
        }

        Ok(Some((output.logits, output.boxes)))
    }

    /// Run GLM-OCR on an image tensor with prompt token IDs.
    ///
    /// Input: `[B, 3, H, W]` image, prompt token IDs.
    /// Returns logits tensor `[B, S, vocab_size]`.
    ///
    /// Falls back to `None` if no OCR model is loaded.
    pub fn run_ocr(&self, image: &DynTensor, prompt_ids: &[usize]) -> Result<Option<DynTensor>> {
        let model = match &self.weights.ocr_model {
            Some(m) => m,
            None => return Ok(None),
        };

        let output = model.forward(image, prompt_ids)?;

        // Validate output shape: [B, S, vocab_size].
        let logit_dims = output.logits.dims();
        if logit_dims.len() != 3 {
            return Err(TensorError::RankMismatch {
                expected: 3,
                actual: logit_dims.len(),
            });
        }
        if logit_dims[2] != model.config().vocab_size {
            return Err(TensorError::shape_mismatch(
                vec![logit_dims[0], logit_dims[1], model.config().vocab_size],
                logit_dims.to_vec(),
            ));
        }

        Ok(Some(output.logits))
    }

    /// Run Granite-Docling OCR on an image tensor with prompt token IDs.
    ///
    /// Input: `[B, 3, 512, 512]` image, prompt token IDs.
    /// Returns logits tensor `[B, num_patches + text_len, vocab_size]`.
    ///
    /// Falls back to `None` if no Granite-Docling model is loaded.
    pub fn run_granite_docling_ocr(
        &self,
        image: &DynTensor,
        prompt_ids: &[usize],
    ) -> Result<Option<DynTensor>> {
        let model = match &self.weights.granite_docling_model {
            Some(m) => m,
            None => return Ok(None),
        };

        let logits = model.forward(image, prompt_ids)?;

        // Validate output shape: [B, S, vocab_size].
        let logit_dims = logits.dims();
        if logit_dims.len() != 3 {
            return Err(TensorError::RankMismatch {
                expected: 3,
                actual: logit_dims.len(),
            });
        }
        if logit_dims[2] != model.config().vocab_size {
            return Err(TensorError::shape_mismatch(
                vec![logit_dims[0], logit_dims[1], model.config().vocab_size],
                logit_dims.to_vec(),
            ));
        }

        Ok(Some(logits))
    }

    /// Run PaddleOCR-VL vision encoding on an image tensor.
    ///
    /// Input: `[B, 3, H, W]` image tensor (H, W divisible by 28).
    /// Returns vision embeddings `[B, merged_tokens, 1024]`.
    ///
    /// Falls back to `None` if no PaddleOCR-VL model is loaded.
    pub fn run_paddle_ocr(&self, image: &DynTensor) -> Result<Option<DynTensor>> {
        let model = match &self.weights.paddle_ocr_model {
            Some(m) => m,
            None => return Ok(None),
        };

        let vision_out = model.vision_encode(image)?;

        // Validate output shape: [B, N, 1024].
        let out_dims = vision_out.dims();
        if out_dims.len() != 3 {
            return Err(TensorError::RankMismatch {
                expected: 3,
                actual: out_dims.len(),
            });
        }
        if out_dims[2] != model.config().vision.merge_output_size {
            return Err(TensorError::shape_mismatch(
                vec![
                    out_dims[0],
                    out_dims[1],
                    model.config().vision.merge_output_size,
                ],
                out_dims.to_vec(),
            ));
        }

        Ok(Some(vision_out))
    }

    /// Run FireRed-OCR on vision features and prompt token IDs.
    ///
    /// Input: `[B, 3, H, W]` image tensor (or `None` for text-only),
    ///        prompt token IDs.
    /// Returns logits tensor `[B, S, vocab_size]`.
    ///
    /// Falls back to `None` if no FireRed-OCR model is loaded.
    pub fn run_firered_ocr(
        &self,
        image: &DynTensor,
        prompt_ids: &[usize],
    ) -> Result<Option<DynTensor>> {
        let model = match &self.weights.firered_ocr_model {
            Some(m) => m,
            None => return Ok(None),
        };

        let output = model.forward(Some(image), prompt_ids)?;

        // Validate output shape: [B, S, vocab_size].
        let logit_dims = output.logits.dims();
        if logit_dims.len() != 3 {
            return Err(TensorError::RankMismatch {
                expected: 3,
                actual: logit_dims.len(),
            });
        }
        if logit_dims[2] != model.config().vocab_size() {
            return Err(TensorError::shape_mismatch(
                vec![logit_dims[0], logit_dims[1], model.config().vocab_size()],
                logit_dims.to_vec(),
            ));
        }

        Ok(Some(output.logits))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Validate that a feature map has the expected spatial dimensions.
pub(crate) fn validate_feature_map_shape(
    tensor: &DynTensor,
    expected_h: usize,
    expected_w: usize,
    label: &str,
) -> Result<()> {
    let dims = tensor.dims();
    if dims.len() != 4 {
        return Err(TensorError::RankMismatch {
            expected: 4,
            actual: dims.len(),
        });
    }
    if dims[2] != expected_h || dims[3] != expected_w {
        return Err(TensorError::InvalidShape(format!(
            "{label} feature map: expected spatial [{expected_h}, {expected_w}], \
             got [{}, {}]",
            dims[2], dims[3]
        )));
    }
    Ok(())
}

#[cfg(test)]
#[path = "dpdf_pipeline_forward_tests.rs"]
mod tests;
