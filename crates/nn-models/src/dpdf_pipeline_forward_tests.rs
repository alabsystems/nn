// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for DynTensor-based dpdf pipeline forward passes with synthetic weights.

use super::*;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// DpdfModelWeights construction
// ---------------------------------------------------------------------------

#[test]
fn test_model_weights_empty() {
    let w = DpdfModelWeights::empty();
    assert!(w.layout_model.is_none());
    assert!(w.ocr_model.is_none());
    assert!(w.granite_docling_model.is_none());
    assert!(w.table_model.is_none());
    assert!(w.paddle_ocr_model.is_none());
    assert!(w.firered_ocr_model.is_none());
}

#[test]
fn test_model_weights_synthetic_layout_only() {
    let w =
        DpdfModelWeights::synthetic_layout_only().expect("synthetic layout weights should load");
    assert!(w.layout_model.is_some());
    assert!(w.ocr_model.is_none());
    assert!(w.granite_docling_model.is_none());
    assert!(w.table_model.is_none());
    assert!(w.paddle_ocr_model.is_none());
    assert!(w.firered_ocr_model.is_none());
}

#[test]
fn test_model_weights_synthetic_table_only() {
    let w = DpdfModelWeights::synthetic_table_only().expect("synthetic table weights should load");
    assert!(w.layout_model.is_none());
    assert!(w.ocr_model.is_none());
    assert!(w.granite_docling_model.is_none());
    assert!(w.table_model.is_some());
    assert!(w.paddle_ocr_model.is_none());
    assert!(w.firered_ocr_model.is_none());
}

#[test]
fn test_model_weights_synthetic_with_granite_docling() {
    let w = DpdfModelWeights::synthetic_with_granite_docling()
        .expect("synthetic granite docling weights should load");
    assert!(w.layout_model.is_some());
    assert!(w.ocr_model.is_none());
    assert!(w.granite_docling_model.is_some());
    assert!(w.table_model.is_none());
    assert!(w.paddle_ocr_model.is_none());
    assert!(w.firered_ocr_model.is_none());
}

#[test]
fn test_model_weights_debug_format() {
    let w = DpdfModelWeights::empty();
    let debug = format!("{w:?}");
    assert!(debug.contains("has_layout: false"));
    assert!(debug.contains("has_ocr: false"));
    assert!(debug.contains("has_granite_docling: false"));
    assert!(debug.contains("has_table: false"));
    assert!(debug.contains("has_paddle_ocr: false"));
    assert!(debug.contains("has_firered_ocr: false"));
}

// ---------------------------------------------------------------------------
// DpdfInferencePipeline construction
// ---------------------------------------------------------------------------

#[test]
fn test_inference_pipeline_with_empty_weights() {
    let w = DpdfModelWeights::empty();
    let pipeline = DpdfInferencePipeline::new(PipelineConfig::default(), w);
    assert!(pipeline.weights().layout_model.is_none());
    assert!(pipeline.weights().ocr_model.is_none());
    assert!(pipeline.weights().granite_docling_model.is_none());
    assert!(pipeline.weights().table_model.is_none());
    assert!(pipeline.weights().paddle_ocr_model.is_none());
    assert!(pipeline.weights().firered_ocr_model.is_none());
}

#[test]
fn test_inference_pipeline_accessors() {
    let w = DpdfModelWeights::empty();
    let cfg = PipelineConfig::default();
    let pipeline = DpdfInferencePipeline::new(cfg, w);
    assert!((pipeline.pipeline().config().layout_conf_threshold - 0.25).abs() < f32::EPSILON);
}

// ---------------------------------------------------------------------------
// Layout detection forward pass
// ---------------------------------------------------------------------------

#[test]
fn test_layout_detection_no_model_returns_empty() {
    let w = DpdfModelWeights::empty();
    let pipeline = DpdfInferencePipeline::new(PipelineConfig::default(), w);
    let image = DynTensor::zeros(&[1, 3, 800, 800], DType::F32, &Device::Cpu).unwrap();
    let regions = pipeline.run_layout_detection(&image).unwrap();
    assert!(
        regions.is_empty(),
        "no layout model should return empty regions"
    );
}

#[test]
fn test_layout_detection_invalid_rank_error() {
    let w = DpdfModelWeights::synthetic_layout_only().unwrap();
    let pipeline = DpdfInferencePipeline::new(PipelineConfig::default(), w);
    // Rank 3 input should fail.
    let image = DynTensor::zeros(&[3, 800, 800], DType::F32, &Device::Cpu).unwrap();
    assert!(pipeline.run_layout_detection(&image).is_err());
}

#[test]
fn test_layout_detection_invalid_channels_error() {
    let w = DpdfModelWeights::synthetic_layout_only().unwrap();
    let pipeline = DpdfInferencePipeline::new(PipelineConfig::default(), w);
    // 1 channel instead of 3.
    let image = DynTensor::zeros(&[1, 1, 800, 800], DType::F32, &Device::Cpu).unwrap();
    assert!(pipeline.run_layout_detection(&image).is_err());
}

#[test]
fn test_layout_detection_forward_pass_shape_validation() {
    // DocLayout-YOLO backbone with zero weights produces zero feature maps.
    // The forward pass validates shapes at P3/P4/P5 levels.
    let w = DpdfModelWeights::synthetic_layout_only().unwrap();
    let pipeline = DpdfInferencePipeline::new(PipelineConfig::default(), w);
    let image = DynTensor::zeros(&[1, 3, 800, 800], DType::F32, &Device::Cpu).unwrap();

    // With zero weights the detection head produces no confident detections
    // (all scores near zero), so we get 0 regions. The important thing is
    // the forward pass completes without shape errors.
    let result = pipeline.run_layout_detection(&image);
    assert!(
        result.is_ok(),
        "forward pass should succeed: {:?}",
        result.err()
    );
}

#[test]
fn test_layout_detection_backbone_shapes_at_smaller_input() {
    // Validate backbone shape propagation at 320x320 (non-default).
    let w = DpdfModelWeights::synthetic_layout_only().unwrap();
    let model = w.layout_model.as_ref().unwrap();
    let image = DynTensor::zeros(&[1, 3, 320, 320], DType::F32, &Device::Cpu).unwrap();

    let (p3, p4, p5) = model
        .forward_backbone(&image)
        .expect("backbone at 320x320 should succeed");

    // P3: 320/8=40, P4: 320/16=20, P5: 320/32=10
    assert_eq!(p3.dims(), &[1, 64, 40, 40]);
    assert_eq!(p4.dims(), &[1, 128, 20, 20]);
    assert_eq!(p5.dims(), &[1, 256, 10, 10]);
}

// ---------------------------------------------------------------------------
// Table structure forward pass
// ---------------------------------------------------------------------------

#[test]
fn test_table_structure_no_model_returns_none() {
    let w = DpdfModelWeights::empty();
    let pipeline = DpdfInferencePipeline::new(PipelineConfig::default(), w);
    let image = DynTensor::zeros(&[1, 3, 640, 640], DType::F32, &Device::Cpu).unwrap();
    let result = pipeline.run_table_structure(&image).unwrap();
    assert!(result.is_none());
}

#[test]
fn test_table_structure_forward_pass_output_shapes() {
    let w = DpdfModelWeights::synthetic_table_only().unwrap();
    let pipeline = DpdfInferencePipeline::new(PipelineConfig::default(), w);
    // Table Transformer expects [B, 3, H, W] divisible by 32.
    let image = DynTensor::zeros(&[1, 3, 640, 640], DType::F32, &Device::Cpu).unwrap();

    let result = pipeline.run_table_structure(&image);
    assert!(
        result.is_ok(),
        "table forward should succeed: {:?}",
        result.err()
    );

    let (logits, boxes) = result.unwrap().expect("should return Some");
    // TableTransformerConfig::preset_structure(): 125 queries, 6 classes (+1 no-object = 7).
    assert_eq!(logits.dims(), &[1, 125, 7]);
    assert_eq!(boxes.dims(), &[1, 125, 4]);
}

// ---------------------------------------------------------------------------
// GLM-OCR forward pass
// ---------------------------------------------------------------------------

#[test]
fn test_ocr_no_model_returns_none() {
    let w = DpdfModelWeights::empty();
    let pipeline = DpdfInferencePipeline::new(PipelineConfig::default(), w);
    let image = DynTensor::zeros(&[1, 3, 384, 384], DType::F32, &Device::Cpu).unwrap();
    let result = pipeline.run_ocr(&image, &[1, 2, 3]).unwrap();
    assert!(result.is_none());
}

#[test]
fn test_glm_ocr_forward_pass_output_shape() {
    // GLM-OCR 0.9B: [1, 3, 384, 384] + 8 tokens -> [1, 576+8, 65024]
    let w = DpdfModelWeights::synthetic().unwrap();
    let pipeline = DpdfInferencePipeline::new(PipelineConfig::default(), w);

    let image = DynTensor::zeros(&[1, 3, 384, 384], DType::F32, &Device::Cpu).unwrap();
    let prompt_ids: Vec<usize> = (0..8).collect();

    let logits = pipeline
        .run_ocr(&image, &prompt_ids)
        .expect("GLM-OCR forward should succeed")
        .expect("should return Some logits");

    // 576 vision patches + 8 text tokens = 584
    assert_eq!(logits.dims(), &[1, 584, 65024]);
}

// ---------------------------------------------------------------------------
// Granite-Docling OCR forward pass
// ---------------------------------------------------------------------------

#[test]
fn test_granite_docling_no_model_returns_none() {
    let w = DpdfModelWeights::empty();
    let pipeline = DpdfInferencePipeline::new(PipelineConfig::default(), w);
    let image = DynTensor::zeros(&[1, 3, 512, 512], DType::F32, &Device::Cpu).unwrap();
    let result = pipeline
        .run_granite_docling_ocr(&image, &[0, 1, 2])
        .unwrap();
    assert!(result.is_none());
}

#[test]
fn test_granite_docling_ocr_forward_pass_output_shape() {
    // Granite-Docling-258M: [1, 3, 512, 512] + 5 tokens -> [1, 1024+5, 49152]
    let w = DpdfModelWeights::synthetic_with_granite_docling().unwrap();
    let pipeline = DpdfInferencePipeline::new(PipelineConfig::default(), w);

    let image = DynTensor::zeros(&[1, 3, 512, 512], DType::F32, &Device::Cpu).unwrap();
    let prompt_ids: Vec<usize> = (0..5).collect();

    let logits = pipeline
        .run_granite_docling_ocr(&image, &prompt_ids)
        .expect("Granite-Docling forward should succeed")
        .expect("should return Some logits");

    // 1024 vision patches + 5 text tokens = 1029
    assert_eq!(logits.dims(), &[1, 1029, 49152]);
}

// ---------------------------------------------------------------------------
// PaddleOCR-VL forward pass
// ---------------------------------------------------------------------------

#[test]
fn test_paddle_ocr_no_model_returns_none() {
    let w = DpdfModelWeights::empty();
    let pipeline = DpdfInferencePipeline::new(PipelineConfig::default(), w);
    let image = DynTensor::zeros(&[1, 3, 392, 392], DType::F32, &Device::Cpu).unwrap();
    let result = pipeline.run_paddle_ocr(&image).unwrap();
    assert!(result.is_none());
}

// ---------------------------------------------------------------------------
// FireRed-OCR forward pass
// ---------------------------------------------------------------------------

#[test]
fn test_firered_ocr_no_model_returns_none() {
    let w = DpdfModelWeights::empty();
    let pipeline = DpdfInferencePipeline::new(PipelineConfig::default(), w);
    let image = DynTensor::zeros(&[1, 3, 448, 448], DType::F32, &Device::Cpu).unwrap();
    let result = pipeline.run_firered_ocr(&image, &[0, 1, 2]).unwrap();
    assert!(result.is_none());
}

// ---------------------------------------------------------------------------
// process_page integration
// ---------------------------------------------------------------------------

#[test]
fn test_process_page_produces_page_output() {
    let w = DpdfModelWeights::synthetic_layout_only().unwrap();
    let pipeline = DpdfInferencePipeline::new(PipelineConfig::default(), w);
    let image = DynTensor::zeros(&[1, 3, 800, 800], DType::F32, &Device::Cpu).unwrap();

    let page = pipeline
        .process_page(&image, 800, 800)
        .expect("process_page should succeed");

    assert_eq!(page.width, 800);
    assert_eq!(page.height, 800);
    // Reading order indices should all be valid.
    for &idx in &page.reading_order {
        assert!(
            idx < page.regions.len(),
            "reading order index out of bounds"
        );
    }
}

#[test]
fn test_process_page_no_model_returns_empty_page() {
    let w = DpdfModelWeights::empty();
    let pipeline = DpdfInferencePipeline::new(PipelineConfig::default(), w);
    let image = DynTensor::zeros(&[1, 3, 800, 800], DType::F32, &Device::Cpu).unwrap();

    let page = pipeline
        .process_page(&image, 612, 792)
        .expect("process_page should succeed with no models");

    assert_eq!(page.width, 612);
    assert_eq!(page.height, 792);
    assert!(page.regions.is_empty());
    assert!(page.reading_order.is_empty());
}

// ---------------------------------------------------------------------------
// Feature map shape validation
// ---------------------------------------------------------------------------

#[test]
fn test_validate_feature_map_shape_ok() {
    let t = DynTensor::zeros(&[1, 64, 100, 100], DType::F32, &Device::Cpu).unwrap();
    assert!(validate_feature_map_shape(&t, 100, 100, "test").is_ok());
}

#[test]
fn test_validate_feature_map_shape_mismatch() {
    let t = DynTensor::zeros(&[1, 64, 100, 100], DType::F32, &Device::Cpu).unwrap();
    assert!(validate_feature_map_shape(&t, 50, 50, "test").is_err());
}

#[test]
fn test_validate_feature_map_shape_wrong_rank() {
    let t = DynTensor::zeros(&[64, 100, 100], DType::F32, &Device::Cpu).unwrap();
    assert!(validate_feature_map_shape(&t, 100, 100, "test").is_err());
}
