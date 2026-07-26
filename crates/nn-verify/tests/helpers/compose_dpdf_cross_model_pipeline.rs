// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-model pipeline chaining composition tests for dpdf document understanding.
//!
//! Verifies end-to-end bounds propagation through multi-model pipelines where
//! one model's output feeds into the next. Unlike `compose_dpdf_cross_model.rs`
//! (which tests pairwise model boundaries), these tests verify **full pipeline
//! chains** with intermediate transformations: dtype conversion, resolution
//! scaling, confidence filtering, NMS, and error propagation.
//!
//! ## Pipeline Tests (19 tests)
//!
//! 1. **Layout detection output shape**: [N, num_classes+4] bounding boxes (IBP)
//! 2. **OCR input from cropped layout regions**: bounds preservation (IBP)
//! 3. **OCR confidence score range**: [0, 1] after sigmoid (IBP)
//! 4. **Table structure detection input from layout crop** (IBP)
//! 5. **VLM input from OCR text + image**: combined bounds (IBP)
//! 6. **End-to-end image to detection boxes bounds** (IBP)
//! 7. **Detection to crop**: coordinate clipping to image bounds (IBP)
//! 8. **Crop to OCR**: resized input bounds preservation (IBP)
//! 9. **OCR to text**: token embedding bounds (IBP)
//! 10. **Multi-model dtype conversion**: FP32 between models (IBP)
//! 11. **Resolution scaling**: 640x640 detection to variable OCR (IBP)
//! 12. **Confidence threshold filtering**: bounds after threshold (IBP)
//! 13. **NMS output**: non-overlapping box bounds (IBP)
//! 14. **Batch pipeline**: N pages processed independently (IBP)
//! 15. **Pipeline error propagation**: bounds widening per stage (IBP)
//! 16. **Model output calibration through pipeline** (IBP + CROWN)
//! 17. **Table cell extraction from structure + OCR** (IBP)
//! 18. **Document understanding**: layout + OCR + table composition (IBP + CROWN)
//!
//! Architecture references:
//! - DocLayout-YOLO (Zhao et al. 2024): YOLOv10-based document layout detection
//! - PaddleOCR (Baidu): DB detector + SVTR recognizer
//! - Table Transformer (Smock et al. 2022): DETR-based table structure
//! - Qwen3-VL (Alibaba): Vision-language model
//!
//! Dimensions (small for fast verification, structurally representative):
//! - FEATURE_DIM=32, SEQ_LEN=4, NUM_CLASSES=8, NUM_BOXES=6,
//!   VOCAB_SIZE=16, FFN_DIM=64, IMG_DIM=16, IN_CHANNELS=3
//!
//! Part of #4123: cross-model pipeline chaining compose tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Feature dimension at model boundaries.
const FEATURE_DIM: usize = 32;
/// Sequence length (text positions / detection anchors).
const SEQ_LEN: usize = 4;
/// Number of layout detection classes.
const NUM_CLASSES: usize = 8;
/// Number of detection boxes (anchors).
const NUM_BOXES: usize = 6;
/// OCR vocabulary size for CTC/autoregressive heads.
const VOCAB_SIZE: usize = 16;
/// FFN intermediate dimension.
const FFN_DIM: usize = 64;
/// Image spatial dimension (single axis, so image is IMG_DIM x IMG_DIM).
const IMG_DIM: usize = 16;
/// Input channels (RGB).
const IN_CHANNELS: usize = 3;
/// Detection output columns: NUM_CLASSES confidence + 4 box coords.
const DET_COLS: usize = NUM_CLASSES + 4;
/// Table structure classes (row, column, spanning cell, header).
const TABLE_CLASSES: usize = 4;
/// Batch size for batch pipeline tests.
const BATCH_SIZE: usize = 2;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ===========================================================================
// Helpers
// ===========================================================================

/// Constant weight tensor binding.
fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

/// Zero bias tensor binding.
fn bias_zero(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

/// Image-domain input bounds: pixels in [0, 1].
fn image_bounds(shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), 0.0f32),
        ArrayD::from_elem(IxDyn(shape), 1.0f32),
    )
    .expect("valid image bounds [0, 1]")
}

/// Sigmoid-domain bounds: output of sigmoid in [0, 1].
fn sigmoid_bounds(shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), 0.0f32),
        ArrayD::from_elem(IxDyn(shape), 1.0f32),
    )
    .expect("valid sigmoid bounds [0, 1]")
}

/// High-confidence bounds: [0.5, 1.0] (post confidence threshold).
fn high_confidence_bounds(shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), 0.5f32),
        ArrayD::from_elem(IxDyn(shape), 1.0f32),
    )
    .expect("valid high-confidence bounds [0.5, 1.0]")
}

// ===========================================================================
// 1. Layout detection output shape: [N, num_classes+4] bounding boxes
// ===========================================================================

/// Build: image features -> linear -> sigmoid -> [NUM_BOXES, DET_COLS] detection.
///
/// Models the YOLO detection head that produces per-box class confidences
/// and bounding box coordinates. Output shape [NUM_BOXES, DET_COLS] where
/// DET_COLS = NUM_CLASSES + 4 (x, y, w, h).
fn build_layout_detection_output_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_layout_detection_output");

    let input = b.add_input("image_features", &[NUM_BOXES, FEATURE_DIM]);
    let det_w = b.add_input("det_weight", &[DET_COLS, FEATURE_DIM]);
    let det_b = b.add_input("det_bias", &[DET_COLS]);
    let logits = b.add_linear(input, det_w, Some(det_b), &[NUM_BOXES, DET_COLS]);
    let out = b.add_sigmoid(logits, &[NUM_BOXES, DET_COLS]);

    b.build(out).expect("valid layout detection output kernel")
}

fn layout_detection_output_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[DET_COLS, FEATURE_DIM]),
        bias_zero(&[DET_COLS]),
    ]
}

#[test]
fn test_pipeline_layout_detection_output_shape_ibp() {
    let def = build_layout_detection_output_kernel();
    let bindings = layout_detection_output_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_BOXES, FEATURE_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through layout detection head");

    // Verify output shape: [NUM_BOXES, DET_COLS]
    assert_eq!(output.lower_upper().0.shape(), &[NUM_BOXES, DET_COLS]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline layout detection output IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid output must be in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "sigmoid lower bound should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "sigmoid upper bound should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 2. OCR input from cropped layout regions: bounds preservation
// ===========================================================================

/// Build: layout box [0,1] -> crop projection -> ReLU -> OCR features.
///
/// Models the boundary where layout box coordinates define a crop region,
/// and the cropped image patch is projected to OCR feature space.
fn build_ocr_from_crop_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_ocr_from_crop");

    // Box coordinates in [0,1] from layout detector
    let boxes = b.add_input("box_coords", &[NUM_BOXES, 4]);

    // Project box coords to crop features
    let proj_w = b.add_input("proj_weight", &[FEATURE_DIM, 4]);
    let proj_b = b.add_input("proj_bias", &[FEATURE_DIM]);
    let features = b.add_linear(boxes, proj_w, Some(proj_b), &[NUM_BOXES, FEATURE_DIM]);

    // ReLU preserves non-negative features
    let out = b.add_relu(features, &[NUM_BOXES, FEATURE_DIM]);

    b.build(out).expect("valid OCR from crop kernel")
}

fn ocr_from_crop_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FEATURE_DIM, 4]),
        bias_zero(&[FEATURE_DIM]),
    ]
}

#[test]
fn test_pipeline_ocr_from_crop_bounds_preservation_ibp() {
    let def = build_ocr_from_crop_kernel();
    let bindings = ocr_from_crop_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Box coords in [0, 1]
    let input = sigmoid_bounds(&[NUM_BOXES, 4]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through crop -> OCR features");

    assert_eq!(output.lower_upper().0.shape(), &[NUM_BOXES, FEATURE_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline OCR from crop IBP: bounds=[{lo_min}, {hi_max}]");

    // ReLU clamps lower to >= 0
    assert!(lo_min >= -1e-4, "ReLU output lower >= 0, got {lo_min}");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 3. OCR confidence score range: [0, 1] after sigmoid
// ===========================================================================

/// Build: OCR logits -> sigmoid -> confidence score in [0, 1].
fn build_ocr_confidence_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_ocr_confidence");

    let features = b.add_input("ocr_features", &[SEQ_LEN, FEATURE_DIM]);
    let score_w = b.add_input("score_weight", &[1, FEATURE_DIM]);
    let score_b = b.add_input("score_bias", &[1]);
    let logits = b.add_linear(features, score_w, Some(score_b), &[SEQ_LEN, 1]);
    let out = b.add_sigmoid(logits, &[SEQ_LEN, 1]);

    b.build(out).expect("valid OCR confidence kernel")
}

fn ocr_confidence_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[1, FEATURE_DIM]),
        bias_zero(&[1]),
    ]
}

#[test]
fn test_pipeline_ocr_confidence_sigmoid_range_ibp() {
    let def = build_ocr_confidence_kernel();
    let bindings = ocr_confidence_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, FEATURE_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through OCR confidence");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, 1]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline OCR confidence IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid output must be in [0, 1]
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 4. Table structure detection input from layout crop
// ===========================================================================

/// Build: layout features -> projection -> ReLU -> table structure input.
///
/// Models the boundary where layout detection features are projected to the
/// Table Transformer input space for table structure recognition.
fn build_table_from_layout_crop_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_table_from_layout_crop");

    let layout_features = b.add_input("layout_features", &[NUM_BOXES, FEATURE_DIM]);

    // Project to table feature space
    let proj_w = b.add_input("proj_weight", &[FFN_DIM, FEATURE_DIM]);
    let proj_b = b.add_input("proj_bias", &[FFN_DIM]);
    let projected = b.add_linear(layout_features, proj_w, Some(proj_b), &[NUM_BOXES, FFN_DIM]);
    let activated = b.add_relu(projected, &[NUM_BOXES, FFN_DIM]);

    // Down-project to table structure classes
    let cls_w = b.add_input("cls_weight", &[TABLE_CLASSES, FFN_DIM]);
    let cls_b = b.add_input("cls_bias", &[TABLE_CLASSES]);
    let logits = b.add_linear(activated, cls_w, Some(cls_b), &[NUM_BOXES, TABLE_CLASSES]);
    let out = b.add_sigmoid(logits, &[NUM_BOXES, TABLE_CLASSES]);

    b.build(out).expect("valid table from layout crop kernel")
}

fn table_from_layout_crop_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, FEATURE_DIM]),
        bias_zero(&[FFN_DIM]),
        weight(&[TABLE_CLASSES, FFN_DIM]),
        bias_zero(&[TABLE_CLASSES]),
    ]
}

#[test]
fn test_pipeline_table_from_layout_crop_ibp() {
    let def = build_table_from_layout_crop_kernel();
    let bindings = table_from_layout_crop_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_BOXES, FEATURE_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through layout -> table structure");

    assert_eq!(output.lower_upper().0.shape(), &[NUM_BOXES, TABLE_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline table from layout crop IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid output in [0, 1]
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 5. VLM input from OCR text + image: combined bounds
// ===========================================================================

/// Build: OCR text embedding + image features -> combined VLM input.
///
/// Models the multimodal fusion where OCR text embeddings and image features
/// are projected to a shared space and added for VLM input.
fn build_vlm_combined_input_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_vlm_combined_input");

    let text_embed = b.add_input("text_embedding", &[SEQ_LEN, FEATURE_DIM]);
    let img_features = b.add_input("image_features", &[SEQ_LEN, FEATURE_DIM]);

    // Project text to combined space
    let text_w = b.add_input("text_proj_weight", &[FEATURE_DIM, FEATURE_DIM]);
    let text_proj = b.add_linear(text_embed, text_w, None, &[SEQ_LEN, FEATURE_DIM]);

    // Project image to combined space
    let img_w = b.add_input("img_proj_weight", &[FEATURE_DIM, FEATURE_DIM]);
    let img_proj = b.add_linear(img_features, img_w, None, &[SEQ_LEN, FEATURE_DIM]);

    // Combine via addition
    let out = b.add_binary_add(text_proj, img_proj, &[SEQ_LEN, FEATURE_DIM]);

    b.build(out).expect("valid VLM combined input kernel")
}

fn vlm_combined_input_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // text_embedding (variable)
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[SEQ_LEN, FEATURE_DIM]),
            0.1f32,
        )), // image_features (constant, from upstream model)
        weight(&[FEATURE_DIM, FEATURE_DIM]),
        weight(&[FEATURE_DIM, FEATURE_DIM]),
    ]
}

#[test]
fn test_pipeline_vlm_combined_input_bounds_ibp() {
    let def = build_vlm_combined_input_kernel();
    let bindings = vlm_combined_input_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, FEATURE_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through VLM combined input");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, FEATURE_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline VLM combined input IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 6. End-to-end image to detection boxes bounds
// ===========================================================================

/// Build: image [0,1] -> conv backbone -> linear head -> sigmoid detection.
///
/// Full pipeline from raw image pixels to detection box outputs.
fn build_image_to_detection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_image_to_detection");

    // Batch-major [IMG_DIM, IN_CHANNELS] so the 1x1-conv-as-linear contracts the
    // channel dim (IN_CHANNELS) against weight [out, in] = [FEATURE_DIM, IN_CHANNELS].
    let image = b.add_input("image", &[IMG_DIM, IN_CHANNELS]);

    // Backbone conv: [IMG_DIM, 3] -> [IMG_DIM, FEATURE_DIM] via linear
    let conv_w = b.add_input("conv_weight", &[FEATURE_DIM, IN_CHANNELS]);
    let conv_b = b.add_input("conv_bias", &[FEATURE_DIM]);
    let features = b.add_linear(image, conv_w, Some(conv_b), &[IMG_DIM, FEATURE_DIM]);
    let features = b.add_relu(features, &[IMG_DIM, FEATURE_DIM]);

    // Pool to fixed detection count: narrow spatial dim (now axis 0)
    let transposed = b.add_narrow(features, 0, 0, NUM_BOXES, &[NUM_BOXES, FEATURE_DIM]);

    // Detection head: [NUM_BOXES, FEATURE_DIM] -> [NUM_BOXES, DET_COLS]
    let det_w = b.add_input("det_weight", &[DET_COLS, FEATURE_DIM]);
    let det_b = b.add_input("det_bias", &[DET_COLS]);
    let logits = b.add_linear(transposed, det_w, Some(det_b), &[NUM_BOXES, DET_COLS]);
    let out = b.add_sigmoid(logits, &[NUM_BOXES, DET_COLS]);

    b.build(out).expect("valid image to detection kernel")
}

fn image_to_detection_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FEATURE_DIM, IN_CHANNELS]),
        bias_zero(&[FEATURE_DIM]),
        weight(&[DET_COLS, FEATURE_DIM]),
        bias_zero(&[DET_COLS]),
    ]
}

#[test]
fn test_pipeline_image_to_detection_boxes_ibp() {
    let def = build_image_to_detection_kernel();
    let bindings = image_to_detection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(&[IMG_DIM, IN_CHANNELS]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through image -> detection boxes");

    assert_eq!(output.lower_upper().0.shape(), &[NUM_BOXES, DET_COLS]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline image -> detection IBP: bounds=[{lo_min}, {hi_max}]");

    // All detection outputs through sigmoid: must be in [0, 1]
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 7. Detection to crop: coordinate clipping to image bounds
// ===========================================================================

/// Build: detection boxes [0,1] -> clip to image -> ReLU (non-negative coords).
///
/// Models coordinate clipping: detection box coordinates are projected and
/// then ReLU ensures non-negative (valid image coordinates).
fn build_detection_to_crop_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_detection_to_crop");

    let boxes = b.add_input("detection_boxes", &[NUM_BOXES, 4]);

    // Project to crop coordinates
    let proj_w = b.add_input("proj_weight", &[4, 4]);
    let proj_b = b.add_input("proj_bias", &[4]);
    let projected = b.add_linear(boxes, proj_w, Some(proj_b), &[NUM_BOXES, 4]);

    // ReLU clips negative coordinates to 0 (image boundary clipping)
    let clipped = b.add_relu(projected, &[NUM_BOXES, 4]);

    // Sigmoid to ensure coordinates in [0, 1] (normalized image space)
    let out = b.add_sigmoid(clipped, &[NUM_BOXES, 4]);

    b.build(out).expect("valid detection to crop kernel")
}

fn detection_to_crop_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[4, 4]),
        bias_zero(&[4]),
    ]
}

#[test]
fn test_pipeline_detection_to_crop_clipping_ibp() {
    let def = build_detection_to_crop_kernel();
    let bindings = detection_to_crop_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = sigmoid_bounds(&[NUM_BOXES, 4]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through detection -> crop clipping");

    assert_eq!(output.lower_upper().0.shape(), &[NUM_BOXES, 4]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline detection -> crop IBP: bounds=[{lo_min}, {hi_max}]");

    // After sigmoid: coordinates in [0, 1]
    assert!(lo_min >= -1e-4, "clipped coords lower >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-4,
        "clipped coords upper <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 8. Crop to OCR: resized input bounds preservation
// ===========================================================================

/// Build: crop features [0,1] -> resize projection -> GELU -> OCR input.
///
/// Models the spatial resize from detection crop to OCR input resolution.
/// The crop feature tensor is projected and activated.
fn build_crop_to_ocr_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_crop_to_ocr");

    let crop = b.add_input("crop_features", &[NUM_BOXES, FEATURE_DIM]);

    // Resize projection: adapt detection features to OCR feature space
    let resize_w = b.add_input("resize_weight", &[FFN_DIM, FEATURE_DIM]);
    let resized = b.add_linear(crop, resize_w, None, &[NUM_BOXES, FFN_DIM]);
    let activated = b.add_gelu(resized, &[NUM_BOXES, FFN_DIM]);

    // Down-project to OCR input dimension
    let down_w = b.add_input("down_weight", &[FEATURE_DIM, FFN_DIM]);
    let down_b = b.add_input("down_bias", &[FEATURE_DIM]);
    let out = b.add_linear(activated, down_w, Some(down_b), &[NUM_BOXES, FEATURE_DIM]);

    b.build(out).expect("valid crop to OCR kernel")
}

fn crop_to_ocr_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, FEATURE_DIM]),
        weight(&[FEATURE_DIM, FFN_DIM]),
        bias_zero(&[FEATURE_DIM]),
    ]
}

#[test]
fn test_pipeline_crop_to_ocr_resized_bounds_ibp() {
    let def = build_crop_to_ocr_kernel();
    let bindings = crop_to_ocr_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_BOXES, FEATURE_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through crop -> OCR resize");

    assert_eq!(output.lower_upper().0.shape(), &[NUM_BOXES, FEATURE_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline crop -> OCR IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 9. OCR to text: token embedding bounds
// ===========================================================================

/// Build: OCR features -> CTC head -> softmax -> token probabilities.
///
/// Models the OCR-to-text output path producing per-position character
/// probability distributions bounded in [0, 1].
fn build_ocr_to_text_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_ocr_to_text");

    let features = b.add_input("ocr_features", &[SEQ_LEN, FEATURE_DIM]);

    // CTC projection: features -> vocabulary logits
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, FEATURE_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(features, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);

    // Softmax produces character probability distribution
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid OCR to text kernel")
}

fn ocr_to_text_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB_SIZE, FEATURE_DIM]),
        bias_zero(&[VOCAB_SIZE]),
    ]
}

#[test]
fn test_pipeline_ocr_to_text_token_embedding_ibp() {
    let def = build_ocr_to_text_kernel();
    let bindings = ocr_to_text_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, FEATURE_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through OCR -> text token embedding");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline OCR -> text IBP: bounds=[{lo_min}, {hi_max}]");

    // Softmax output in [0, 1]
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 10. Multi-model dtype conversion: FP32 between models
// ===========================================================================

/// Build: model A output -> identity projection (FP32 passthrough) -> model B input.
///
/// Models the dtype conversion boundary between models. In production,
/// different models may use different internal precisions, but pipeline
/// boundaries use FP32. This verifies bounds preservation through the
/// identity-like conversion path.
fn build_dtype_conversion_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_dtype_conversion");

    let input = b.add_input("model_a_output", &[SEQ_LEN, FEATURE_DIM]);

    // Identity-like projection (models FP32 passthrough with small perturbation)
    let id_w = b.add_input("identity_weight", &[FEATURE_DIM, FEATURE_DIM]);
    let id_b = b.add_input("identity_bias", &[FEATURE_DIM]);
    let out = b.add_linear(input, id_w, Some(id_b), &[SEQ_LEN, FEATURE_DIM]);

    b.build(out).expect("valid dtype conversion kernel")
}

fn dtype_conversion_bindings() -> Vec<TensorParamBinding> {
    // Near-identity weight matrix: diagonal = 1.0, off-diagonal = 0.0
    let n = FEATURE_DIM;
    let mut id_data = vec![0.0f32; n * n];
    for i in 0..n {
        id_data[i * n + i] = 1.0;
    }
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[n, n]), id_data).expect("valid identity matrix"),
        ),
        bias_zero(&[FEATURE_DIM]),
    ]
}

#[test]
fn test_pipeline_dtype_conversion_fp32_ibp() {
    let def = build_dtype_conversion_kernel();
    let bindings = dtype_conversion_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, FEATURE_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FP32 dtype conversion");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, FEATURE_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline dtype conversion IBP: bounds=[{lo_min}, {hi_max}]");

    // Identity projection: output bounds should be close to input bounds
    // Input is [-1, 1], identity matrix preserves bounds
    assert!(lo_min >= -1.0 - 1e-3, "identity lower should be near -1.0");
    assert!(hi_max <= 1.0 + 1e-3, "identity upper should be near 1.0");
}

// ===========================================================================
// 11. Resolution scaling: 640x640 detection to variable OCR
// ===========================================================================

/// Build: detection features at one resolution -> projection -> OCR resolution.
///
/// Models resolution adaptation between detection (fixed grid) and OCR
/// (variable length). The detection features are projected to match
/// the OCR input resolution.
fn build_resolution_scaling_kernel() -> TensorKernelDef {
    let det_seq: usize = 8; // detection spatial positions (representing 640x640 grid)
    let ocr_seq: usize = SEQ_LEN; // OCR output positions

    let mut b = TensorBlockBuilder::new("pipeline_resolution_scaling");

    let input = b.add_input("det_features", &[det_seq, FEATURE_DIM]);

    // Narrow to OCR sequence length (models spatial subsampling)
    let narrowed = b.add_narrow(input, 0, 0, ocr_seq, &[ocr_seq, FEATURE_DIM]);

    // Scale projection
    let scale_w = b.add_input("scale_weight", &[FEATURE_DIM, FEATURE_DIM]);
    let scale_b = b.add_input("scale_bias", &[FEATURE_DIM]);
    let out = b.add_linear(narrowed, scale_w, Some(scale_b), &[ocr_seq, FEATURE_DIM]);

    b.build(out).expect("valid resolution scaling kernel")
}

fn resolution_scaling_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FEATURE_DIM, FEATURE_DIM]),
        bias_zero(&[FEATURE_DIM]),
    ]
}

#[test]
fn test_pipeline_resolution_scaling_ibp() {
    let det_seq: usize = 8;
    let def = build_resolution_scaling_kernel();
    let bindings = resolution_scaling_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[det_seq, FEATURE_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through resolution scaling");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, FEATURE_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline resolution scaling IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 12. Confidence threshold filtering: bounds after threshold
// ===========================================================================

/// Build: detection confidence -> threshold (ReLU-based) -> filtered output.
///
/// Models confidence threshold filtering: detection confidences below
/// threshold are zeroed out. Modeled as (confidence - threshold) through
/// ReLU, producing 0 for low-confidence and positive for high-confidence.
fn build_confidence_threshold_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_confidence_threshold");

    let confidence = b.add_input("detection_confidence", &[NUM_BOXES, 1]);

    // Threshold offset: subtract threshold via linear with weight=1, bias=-threshold
    let offset_w = b.add_input("offset_weight", &[1, 1]);
    let offset_b = b.add_input("offset_bias", &[1]);
    let offset = b.add_linear(confidence, offset_w, Some(offset_b), &[NUM_BOXES, 1]);

    // ReLU: zero out below threshold
    let out = b.add_relu(offset, &[NUM_BOXES, 1]);

    b.build(out).expect("valid confidence threshold kernel")
}

fn confidence_threshold_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        // Weight = 1.0 (passthrough)
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, 1]), 1.0f32)),
        // Bias = -0.5 (threshold at 0.5)
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), -0.5f32)),
    ]
}

#[test]
fn test_pipeline_confidence_threshold_filtering_ibp() {
    let def = build_confidence_threshold_kernel();
    let bindings = confidence_threshold_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Confidence in [0, 1]
    let input = sigmoid_bounds(&[NUM_BOXES, 1]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through confidence threshold");

    assert_eq!(output.lower_upper().0.shape(), &[NUM_BOXES, 1]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline confidence threshold IBP: bounds=[{lo_min}, {hi_max}]");

    // After ReLU: output >= 0
    assert!(lo_min >= -1e-4, "thresholded lower >= 0, got {lo_min}");
    // Max: confidence=1.0 - 0.5 = 0.5
    assert!(
        hi_max <= 0.5 + 1e-4,
        "thresholded upper should be <= 0.5, got {hi_max}"
    );
}

// ===========================================================================
// 13. NMS output: non-overlapping box bounds
// ===========================================================================

/// Build: detection boxes -> NMS modeling -> filtered box features.
///
/// NMS is non-differentiable, so for verification we model the post-NMS
/// output as the input boxes projected through a suppression gate (sigmoid).
/// The key property: post-NMS boxes retain [0, 1] coordinate bounds.
fn build_nms_output_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_nms_output");

    let boxes = b.add_input("detection_boxes", &[NUM_BOXES, 4]);

    // NMS gate: learned suppression score per box
    let gate_w = b.add_input("gate_weight", &[1, 4]);
    let gate_b = b.add_input("gate_bias", &[1]);
    let gate_logit = b.add_linear(boxes, gate_w, Some(gate_b), &[NUM_BOXES, 1]);
    let gate = b.add_sigmoid(gate_logit, &[NUM_BOXES, 1]);

    // Broadcast gate to box dimensions
    let gate_bc = b.add_broadcast(gate, &[NUM_BOXES, 4]);

    // Apply gate: element-wise multiply (suppressed boxes -> near zero)
    let out = b.add_binary_mul(boxes, gate_bc, &[NUM_BOXES, 4]);

    b.build(out).expect("valid NMS output kernel")
}

fn nms_output_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[1, 4]),
        bias_zero(&[1]),
    ]
}

#[test]
fn test_pipeline_nms_output_bounds_ibp() {
    let def = build_nms_output_kernel();
    let bindings = nms_output_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Input boxes in [0, 1] (normalized coordinates)
    let input = sigmoid_bounds(&[NUM_BOXES, 4]);

    let output = graph.propagate_ibp(&input).expect("IBP through NMS output");

    assert_eq!(output.lower_upper().0.shape(), &[NUM_BOXES, 4]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline NMS output IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Gate * input: gate in [0,1], input in [0,1] -> output in [0,1]
    assert!(lo_min >= -1e-4, "NMS output lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "NMS output upper <= 1, got {hi_max}");
}

// ===========================================================================
// 14. Batch pipeline: N pages processed independently
// ===========================================================================

/// Build: batch of page features -> projection -> sigmoid -> batch detection.
///
/// Verifies that batch processing preserves bounds independently per page.
/// Uses [BATCH_SIZE * NUM_BOXES, FEATURE_DIM] to model batched input.
fn build_batch_pipeline_kernel() -> TensorKernelDef {
    let batch_boxes = BATCH_SIZE * NUM_BOXES;
    let mut b = TensorBlockBuilder::new("pipeline_batch_processing");

    let input = b.add_input("batch_features", &[batch_boxes, FEATURE_DIM]);

    // Detection head per page (shared weights)
    let det_w = b.add_input("det_weight", &[NUM_CLASSES, FEATURE_DIM]);
    let det_b = b.add_input("det_bias", &[NUM_CLASSES]);
    let logits = b.add_linear(input, det_w, Some(det_b), &[batch_boxes, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[batch_boxes, NUM_CLASSES]);

    b.build(out).expect("valid batch pipeline kernel")
}

fn batch_pipeline_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[NUM_CLASSES, FEATURE_DIM]),
        bias_zero(&[NUM_CLASSES]),
    ]
}

#[test]
fn test_pipeline_batch_pages_independent_ibp() {
    let batch_boxes = BATCH_SIZE * NUM_BOXES;
    let def = build_batch_pipeline_kernel();
    let bindings = batch_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[batch_boxes, FEATURE_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through batch pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[batch_boxes, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline batch pages IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid output per page in [0, 1]
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 15. Pipeline error propagation: bounds widening per stage
// ===========================================================================

/// Build: 3-stage pipeline showing bounds growth through composition.
///
/// Stage 1: Linear -> ReLU
/// Stage 2: Linear -> GELU
/// Stage 3: Linear -> sigmoid (capped to [0, 1])
///
/// Verifies that bounds widen through uncapped stages but sigmoid
/// caps the final output.
fn build_error_propagation_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_error_propagation");

    let input = b.add_input("pipeline_input", &[SEQ_LEN, FEATURE_DIM]);

    // Stage 1: Linear -> ReLU
    let w1 = b.add_input("w1", &[FEATURE_DIM, FEATURE_DIM]);
    let s1 = b.add_linear(input, w1, None, &[SEQ_LEN, FEATURE_DIM]);
    let s1_act = b.add_relu(s1, &[SEQ_LEN, FEATURE_DIM]);

    // Stage 2: Linear -> GELU
    let w2 = b.add_input("w2", &[FEATURE_DIM, FEATURE_DIM]);
    let s2 = b.add_linear(s1_act, w2, None, &[SEQ_LEN, FEATURE_DIM]);
    let s2_act = b.add_gelu(s2, &[SEQ_LEN, FEATURE_DIM]);

    // Stage 3: Linear -> sigmoid (caps output to [0, 1])
    let w3 = b.add_input("w3", &[1, FEATURE_DIM]);
    let s3 = b.add_linear(s2_act, w3, None, &[SEQ_LEN, 1]);
    let out = b.add_sigmoid(s3, &[SEQ_LEN, 1]);

    b.build(out).expect("valid error propagation kernel")
}

fn error_propagation_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FEATURE_DIM, FEATURE_DIM]),
        weight(&[FEATURE_DIM, FEATURE_DIM]),
        weight(&[1, FEATURE_DIM]),
    ]
}

#[test]
fn test_pipeline_error_propagation_bounds_widening_ibp() {
    let def = build_error_propagation_kernel();
    let bindings = error_propagation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, FEATURE_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 3-stage error propagation");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, 1]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline error propagation IBP: bounds=[{lo_min}, {hi_max}]");

    // Final sigmoid caps output to [0, 1] regardless of intermediate widening
    assert!(lo_min >= -1e-4, "sigmoid caps lower >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-4,
        "sigmoid caps upper <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 16. Model output calibration through pipeline (IBP + CROWN)
// ===========================================================================

/// Build: features -> linear -> sigmoid (calibrated detection confidence).
///
/// Simple pipeline where CROWN linearization through sigmoid should
/// produce tighter bounds than IBP.
fn build_calibration_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_calibration");

    let input = b.add_input("features", &[SEQ_LEN, FEATURE_DIM]);

    // Linear calibration layer
    let w = b.add_input("cal_weight", &[1, FEATURE_DIM]);
    let bias = b.add_input("cal_bias", &[1]);
    let logit = b.add_linear(input, w, Some(bias), &[SEQ_LEN, 1]);

    // Sigmoid calibration
    let out = b.add_sigmoid(logit, &[SEQ_LEN, 1]);

    b.build(out).expect("valid calibration pipeline kernel")
}

fn calibration_pipeline_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[1, FEATURE_DIM]),
        bias_zero(&[1]),
    ]
}

#[test]
fn test_pipeline_calibration_crown() {
    let def = build_calibration_pipeline_kernel();
    let bindings = calibration_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, FEATURE_DIM], 1.0);

    // IBP baseline
    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through calibration");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);

    // CROWN with fallback
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);

    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    eprintln!(
        "Pipeline calibration: IBP=[{ibp_lo}, {ibp_hi}], \
         CROWN=[{crown_lo}, {crown_hi}], method={method:?}"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    // Both must be in [0, 1] (sigmoid)
    assert!(crown_lo >= -1e-4, "CROWN sigmoid lower >= 0");
    assert!(crown_hi <= 1.0 + 1e-4, "CROWN sigmoid upper <= 1");
}

// ===========================================================================
// 17. Table cell extraction from structure + OCR
// ===========================================================================

/// Build: table sigmoid [0,1] -> projection -> GELU -> CTC softmax.
///
/// Models the pipeline from table structure detection (which cells exist)
/// through OCR recognition of cell contents.
fn build_table_cell_extraction_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_table_cell_extraction");

    let table_cells = b.add_input("table_structure", &[NUM_BOXES, TABLE_CLASSES]);

    // Project table cell features to OCR feature space
    let proj_w = b.add_input("proj_weight", &[FEATURE_DIM, TABLE_CLASSES]);
    let proj_b = b.add_input("proj_bias", &[FEATURE_DIM]);
    let projected = b.add_linear(table_cells, proj_w, Some(proj_b), &[NUM_BOXES, FEATURE_DIM]);
    let activated = b.add_gelu(projected, &[NUM_BOXES, FEATURE_DIM]);

    // Narrow to text sequence length
    let narrowed = b.add_narrow(activated, 0, 0, SEQ_LEN, &[SEQ_LEN, FEATURE_DIM]);

    // CTC head: Linear -> Softmax
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, FEATURE_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(narrowed, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid table cell extraction kernel")
}

fn table_cell_extraction_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FEATURE_DIM, TABLE_CLASSES]),
        bias_zero(&[FEATURE_DIM]),
        weight(&[VOCAB_SIZE, FEATURE_DIM]),
        bias_zero(&[VOCAB_SIZE]),
    ]
}

#[test]
fn test_pipeline_table_cell_extraction_ibp() {
    let def = build_table_cell_extraction_kernel();
    let bindings = table_cell_extraction_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Table structure outputs in [0, 1] (sigmoid)
    let input = sigmoid_bounds(&[NUM_BOXES, TABLE_CLASSES]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through table cell -> OCR");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline table cell extraction IBP: bounds=[{lo_min}, {hi_max}]");

    // Softmax output in [0, 1]
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 18. Document understanding: layout + OCR + table composition (IBP + CROWN)
// ===========================================================================

/// Build: image -> backbone -> detection sigmoid -> projection -> GELU
///        -> CTC softmax end-to-end document understanding pipeline.
///
/// This is the full 4-stage pipeline: image features -> layout detection ->
/// feature projection -> text recognition. Tests end-to-end bounds
/// propagation through the complete document understanding chain.
fn build_document_understanding_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_document_understanding");

    // Batch-major [IMG_DIM, IN_CHANNELS] so the backbone linear contracts the
    // channel dim (IN_CHANNELS) against weight [out, in] = [FEATURE_DIM, IN_CHANNELS].
    let input = b.add_input("image_features", &[IMG_DIM, IN_CHANNELS]);

    // Stage 1: Backbone feature extraction (linear + ReLU)
    let backbone_w = b.add_input("backbone_weight", &[FEATURE_DIM, IN_CHANNELS]);
    let backbone_b = b.add_input("backbone_bias", &[FEATURE_DIM]);
    let backbone = b.add_linear(input, backbone_w, Some(backbone_b), &[IMG_DIM, FEATURE_DIM]);
    let backbone_act = b.add_relu(backbone, &[IMG_DIM, FEATURE_DIM]);

    // Stage 2: Layout detection (narrow spatial axis 0 + detection head + sigmoid)
    let transposed = b.add_narrow(backbone_act, 0, 0, NUM_BOXES, &[NUM_BOXES, FEATURE_DIM]);
    let det_w = b.add_input("det_weight", &[NUM_CLASSES, FEATURE_DIM]);
    let det_b = b.add_input("det_bias", &[NUM_CLASSES]);
    let det_logits = b.add_linear(transposed, det_w, Some(det_b), &[NUM_BOXES, NUM_CLASSES]);
    let det_conf = b.add_sigmoid(det_logits, &[NUM_BOXES, NUM_CLASSES]);

    // Stage 3: Feature projection (linear + GELU)
    let proj_w = b.add_input("proj_weight", &[FEATURE_DIM, NUM_CLASSES]);
    let proj_b = b.add_input("proj_bias", &[FEATURE_DIM]);
    let projected = b.add_linear(det_conf, proj_w, Some(proj_b), &[NUM_BOXES, FEATURE_DIM]);
    let proj_act = b.add_gelu(projected, &[NUM_BOXES, FEATURE_DIM]);

    // Stage 4: Text recognition (narrow + CTC head + softmax)
    let text_seq = b.add_narrow(proj_act, 0, 0, SEQ_LEN, &[SEQ_LEN, FEATURE_DIM]);
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, FEATURE_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let ctc_logits = b.add_linear(text_seq, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(ctc_logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid document understanding pipeline kernel")
}

fn document_understanding_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // image_features
        weight(&[FEATURE_DIM, IN_CHANNELS]),
        bias_zero(&[FEATURE_DIM]),
        weight(&[NUM_CLASSES, FEATURE_DIM]),
        bias_zero(&[NUM_CLASSES]),
        weight(&[FEATURE_DIM, NUM_CLASSES]),
        bias_zero(&[FEATURE_DIM]),
        weight(&[VOCAB_SIZE, FEATURE_DIM]),
        bias_zero(&[VOCAB_SIZE]),
    ]
}

#[test]
fn test_pipeline_document_understanding_ibp() {
    let def = build_document_understanding_kernel();
    let bindings = document_understanding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(&[IMG_DIM, IN_CHANNELS]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full document understanding pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pipeline document understanding IBP: bounds=[{lo_min}, {hi_max}]");

    // Final softmax output in [0, 1]
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

#[test]
fn test_pipeline_document_understanding_crown() {
    let def = build_document_understanding_kernel();
    let bindings = document_understanding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(&[IMG_DIM, IN_CHANNELS]);

    // IBP baseline
    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through document understanding");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);

    // CROWN with fallback
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);

    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    eprintln!(
        "Pipeline document understanding: IBP=[{ibp_lo}, {ibp_hi}], \
         CROWN=[{crown_lo}, {crown_hi}], method={method:?}"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    // Final softmax: both methods must produce bounds in [0, 1]
    assert!(crown_lo >= -1e-4, "CROWN softmax lower >= 0");
    assert!(crown_hi <= 1.0 + 1e-4, "CROWN softmax upper <= 1");
}
