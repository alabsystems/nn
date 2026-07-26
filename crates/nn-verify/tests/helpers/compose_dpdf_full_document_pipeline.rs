// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for full document pipeline (layout + table + OCR) bounds.
//!
//! Verifies IBP and CROWN bound propagation through the three-model document
//! processing pipeline: DocLayout-YOLO detection, Table Transformer cell
//! detection, and PaddleOCR text recognition composed end-to-end.
//!
//! ## Tests (15 tests)
//!
//! 1.  **DocLayout-YOLO detection -> region extraction bounds** (IBP)
//! 2.  **Region feature extraction for table detection** (IBP)
//! 3.  **Table Transformer cell detection bounds** (IBP + CROWN)
//! 4.  **Cell content -> OCR input preprocessing** (IBP)
//! 5.  **PaddleOCR recognition on cell content** (IBP + CROWN)
//! 6.  **Layout-to-table pipeline composition** (IBP)
//! 7.  **Table-to-OCR pipeline composition** (IBP + CROWN)
//! 8.  **Full three-model pipeline bounds** (IBP)
//! 9.  **Document page preprocessing bounds** (IBP)
//! 10. **ROI extraction bounds** (IBP)
//! 11. **Text line grouping from layout** (IBP + CROWN)
//! 12. **Table cell content extraction** (IBP)
//! 13. **Cross-model dimension compatibility** (IBP)
//! 14. **Confidence score aggregation** (IBP)
//! 15. **Multi-page batch processing bounds** (IBP)
//!
//! Architecture references:
//! - DocLayout-YOLO: YOLO-based document layout detection (boxes + classes)
//! - Table Transformer (DETR-based): Table cell structure detection
//! - PaddleOCR (SVTR): CTC-based text recognition
//! - Pipeline: image -> layout detection -> region crop -> table structure ->
//!   cell crop -> OCR recognition -> text output
//!
//! Dimensions (small for fast verification, structurally representative):
//! - IMG_SIZE=4, IN_CHANNELS=3, HIDDEN_DIM=4, NUM_BOXES=4
//! - NUM_CELLS=4, SEQ_LEN=4, VOCAB_SIZE=6, BASE_CH=4
//!
//! Part of #4199: Compose tests for full document pipeline.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const IMG_SIZE: usize = 4;
const IN_CHANNELS: usize = 3;
const HIDDEN_DIM: usize = 4;
const NUM_BOXES: usize = 4;
const NUM_CELLS: usize = 4;
const SEQ_LEN: usize = 4;
const VOCAB_SIZE: usize = 6;
const BASE_CH: usize = 4;
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Constant weight tensor binding.
fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

/// Zero bias tensor binding.
fn bias_zero(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

/// Ones tensor binding (for LayerNorm weight).
fn ones(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 1.0f32))
}

/// Scalar epsilon binding.
fn eps_binding() -> TensorParamBinding {
    TensorParamBinding::ConstantScalar(1e-5)
}

/// Image-domain input bounds: pixels in [0, 1].
fn image_bounds(channels: usize, h: usize, w: usize) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[channels, h, w]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[channels, h, w]), 1.0f32),
    )
    .expect("valid image bounds [0, 1]")
}

/// Sequence-domain input bounds: embeddings in [-range, +range].
fn seq_bounds(seq_len: usize, dim: usize, range: f32) -> BoundedTensor {
    uniform_bounds(&[seq_len, dim], range)
}

// ===========================================================================
// 1. DocLayout-YOLO detection -> region extraction bounds (IBP)
// ===========================================================================

#[test]
fn test_full_doc_layout_detection_region_extraction_ibp() {
    // DocLayout-YOLO backbone: Conv2d -> ReLU -> Linear -> sigmoid (box + class scores)
    // Image [C, H, W] -> Conv -> flatten -> Linear -> sigmoid -> [NUM_BOXES, 5]
    // 5 = 4 (box coords) + 1 (confidence)
    let conv_out_h = IMG_SIZE / 2;
    let conv_out_w = IMG_SIZE / 2;
    let flat_dim = BASE_CH * conv_out_h * conv_out_w;
    let out_dim = NUM_BOXES * 5;

    let mut b = TensorBlockBuilder::new("doc_layout_detect");
    let image = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let conv_w = b.add_input("conv_w", &[BASE_CH, IN_CHANNELS, 2, 2]);
    let conv_b = b.add_input("conv_b", &[BASE_CH]);
    let conv_out = b.add_conv2d(
        image,
        conv_w,
        Some(conv_b),
        2,
        2,
        0,
        0,
        &[BASE_CH, conv_out_h, conv_out_w],
    );
    let act = b.add_relu(conv_out, &[BASE_CH, conv_out_h, conv_out_w]);
    let flat = b.add_reshape(act, &[1, flat_dim]);

    let fc_w = b.add_input("fc_w", &[out_dim, flat_dim]);
    let fc_b = b.add_input("fc_b", &[out_dim]);
    let logits = b.add_linear(flat, fc_w, Some(fc_b), &[1, out_dim]);
    let out = b.add_sigmoid(logits, &[1, out_dim]);
    let def = b.build(out).expect("valid layout detect kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[BASE_CH, IN_CHANNELS, 2, 2]),
        bias_zero(&[BASE_CH]),
        weight(&[out_dim, flat_dim]),
        bias_zero(&[out_dim]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(IN_CHANNELS, IMG_SIZE, IMG_SIZE);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("layout detect IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Sigmoid output must be in [0, 1]
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 2. Region feature extraction for table detection (IBP)
// ===========================================================================

#[test]
fn test_full_doc_region_feature_extraction_ibp() {
    // Extracted region features: Linear(HIDDEN_DIM -> HIDDEN_DIM) + GELU
    // Represents feature extraction from cropped layout regions for table detection.
    let mut b = TensorBlockBuilder::new("doc_region_features");
    let input = b.add_input("region_features", &[NUM_BOXES, HIDDEN_DIM]);
    let fc_w = b.add_input("fc_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let fc_b = b.add_input("fc_b", &[HIDDEN_DIM]);
    let proj = b.add_linear(input, fc_w, Some(fc_b), &[NUM_BOXES, HIDDEN_DIM]);
    let out = b.add_gelu(proj, &[NUM_BOXES, HIDDEN_DIM]);
    let def = b.build(out).expect("valid region feature kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(NUM_BOXES, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[NUM_BOXES, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("region feature extraction IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 3. Table Transformer cell detection bounds (IBP + CROWN)
// ===========================================================================

#[test]
fn test_full_doc_table_cell_detection_ibp_crown() {
    // Table Transformer cell detection head:
    // Linear(HIDDEN_DIM -> HIDDEN_DIM) -> GELU -> Linear(HIDDEN_DIM -> NUM_CELLS*4) -> sigmoid
    // Outputs normalized cell bounding box coordinates.
    let cell_coords = NUM_CELLS * 4;

    let mut b = TensorBlockBuilder::new("doc_table_cell_detect");
    let input = b.add_input("table_features", &[1, HIDDEN_DIM]);
    let fc1_w = b.add_input("fc1_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let fc1_b = b.add_input("fc1_b", &[HIDDEN_DIM]);
    let h = b.add_linear(input, fc1_w, Some(fc1_b), &[1, HIDDEN_DIM]);
    let act = b.add_gelu(h, &[1, HIDDEN_DIM]);
    let fc2_w = b.add_input("fc2_w", &[cell_coords, HIDDEN_DIM]);
    let fc2_b = b.add_input("fc2_b", &[cell_coords]);
    let logits = b.add_linear(act, fc2_w, Some(fc2_b), &[1, cell_coords]);
    let out = b.add_sigmoid(logits, &[1, cell_coords]);
    let def = b.build(out).expect("valid table cell detect kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        weight(&[cell_coords, HIDDEN_DIM]),
        bias_zero(&[cell_coords]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(1, HIDDEN_DIM, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("table cell detect IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("table cell detect CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 4. Cell content -> OCR input preprocessing (IBP)
// ===========================================================================

#[test]
fn test_full_doc_cell_to_ocr_preprocessing_ibp() {
    // Cell content preprocessing: crop region -> resize -> normalize
    // Modeled as: Linear(HIDDEN_DIM -> HIDDEN_DIM) -> LayerNorm
    // Represents the feature transformation from table cell ROI to OCR input.
    let mut b = TensorBlockBuilder::new("doc_cell_preprocess");
    let input = b.add_input("cell_crop", &[NUM_CELLS, HIDDEN_DIM]);
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj = b.add_linear(input, proj_w, None, &[NUM_CELLS, HIDDEN_DIM]);

    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let out = b.add_layer_norm(proj, eps, 1, ln_w, ln_b, &[NUM_CELLS, HIDDEN_DIM]);
    let def = b.build(out).expect("valid cell preprocess kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        ones(&[HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        eps_binding(),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(NUM_CELLS, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[NUM_CELLS, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("cell-to-OCR preprocess IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 5. PaddleOCR recognition on cell content (IBP + CROWN)
// ===========================================================================

#[test]
fn test_full_doc_paddle_ocr_recognition_ibp_crown() {
    // PaddleOCR SVTR recognition: Linear -> GELU -> Linear -> softmax (CTC output)
    // Input: [SEQ_LEN, HIDDEN_DIM], Output: [SEQ_LEN, VOCAB_SIZE] probabilities.
    let mut b = TensorBlockBuilder::new("doc_ocr_recognize");
    let input = b.add_input("cell_features", &[SEQ_LEN, HIDDEN_DIM]);
    let fc1_w = b.add_input("fc1_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let fc1_b = b.add_input("fc1_b", &[HIDDEN_DIM]);
    let h = b.add_linear(input, fc1_w, Some(fc1_b), &[SEQ_LEN, HIDDEN_DIM]);
    let act = b.add_gelu(h, &[SEQ_LEN, HIDDEN_DIM]);
    let fc2_w = b.add_input("fc2_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let fc2_b = b.add_input("fc2_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(act, fc2_w, Some(fc2_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid OCR recognize kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        bias_zero(&[VOCAB_SIZE]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("OCR recognition IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "softmax upper <= 1, got {hi_max}");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("OCR recognition CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 6. Layout-to-table pipeline composition (IBP)
// ===========================================================================

#[test]
fn test_full_doc_layout_to_table_pipeline_ibp() {
    // Compose: layout detection features -> projection -> table cell detection head
    // Linear(HIDDEN_DIM -> HIDDEN_DIM) -> GELU -> Linear(HIDDEN_DIM -> NUM_CELLS*4) -> sigmoid
    let cell_coords = NUM_CELLS * 4;

    let mut b = TensorBlockBuilder::new("doc_layout_to_table");
    let input = b.add_input("layout_features", &[1, HIDDEN_DIM]);

    // Layout projection stage
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_b", &[HIDDEN_DIM]);
    let proj = b.add_linear(input, proj_w, Some(proj_b), &[1, HIDDEN_DIM]);
    let proj_act = b.add_gelu(proj, &[1, HIDDEN_DIM]);

    // Table cell detection stage
    let cell_w = b.add_input("cell_w", &[cell_coords, HIDDEN_DIM]);
    let cell_b = b.add_input("cell_b", &[cell_coords]);
    let cell_logits = b.add_linear(proj_act, cell_w, Some(cell_b), &[1, cell_coords]);
    let out = b.add_sigmoid(cell_logits, &[1, cell_coords]);
    let def = b.build(out).expect("valid layout-to-table kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        weight(&[cell_coords, HIDDEN_DIM]),
        bias_zero(&[cell_coords]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(1, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[1, cell_coords]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("layout-to-table IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 7. Table-to-OCR pipeline composition (IBP + CROWN)
// ===========================================================================

#[test]
fn test_full_doc_table_to_ocr_pipeline_ibp_crown() {
    // Compose: table cell features -> LayerNorm -> Linear -> GELU -> CTC softmax
    // Models the flow from detected table cell regions through OCR recognition.
    let mut b = TensorBlockBuilder::new("doc_table_to_ocr");
    let input = b.add_input("cell_features", &[SEQ_LEN, HIDDEN_DIM]);

    // LayerNorm normalization stage
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &[SEQ_LEN, HIDDEN_DIM]);

    // OCR recognition stage
    let fc_w = b.add_input("fc_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let h = b.add_linear(normed, fc_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let act = b.add_gelu(h, &[SEQ_LEN, HIDDEN_DIM]);
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(act, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid table-to-OCR kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        ones(&[HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        eps_binding(),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        bias_zero(&[VOCAB_SIZE]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("table-to-OCR IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "softmax upper <= 1, got {hi_max}");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("table-to-OCR CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 8. Full three-model pipeline bounds (IBP)
// ===========================================================================

#[test]
fn test_full_doc_three_model_pipeline_ibp() {
    // Full pipeline: detection -> table structure -> OCR
    // Linear(HIDDEN_DIM) -> GELU -> Linear(HIDDEN_DIM) -> LayerNorm ->
    // Linear(HIDDEN_DIM) -> GELU -> Linear(VOCAB_SIZE) -> softmax
    let mut b = TensorBlockBuilder::new("doc_full_pipeline");
    let input = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);

    // Stage 1: Detection feature projection
    let det_w = b.add_input("det_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let det_b = b.add_input("det_b", &[HIDDEN_DIM]);
    let det = b.add_linear(input, det_w, Some(det_b), &[SEQ_LEN, HIDDEN_DIM]);
    let det_act = b.add_gelu(det, &[SEQ_LEN, HIDDEN_DIM]);

    // Stage 2: Table structure normalization
    let tbl_w = b.add_input("tbl_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let tbl = b.add_linear(det_act, tbl_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let normed = b.add_layer_norm(tbl, eps, 1, ln_w, ln_b, &[SEQ_LEN, HIDDEN_DIM]);

    // Stage 3: OCR recognition
    let ocr_w = b.add_input("ocr_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let ocr = b.add_linear(normed, ocr_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let ocr_act = b.add_gelu(ocr, &[SEQ_LEN, HIDDEN_DIM]);
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(ocr_act, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid full pipeline kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        ones(&[HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        eps_binding(),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        bias_zero(&[VOCAB_SIZE]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("full 3-model pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 9. Document page preprocessing bounds (IBP)
// ===========================================================================

#[test]
fn test_full_doc_page_preprocessing_ibp() {
    // Document page preprocessing: Conv2d -> ReLU -> flatten -> Linear
    // Models the initial feature extraction from a raw document page image.
    let conv_out_h = IMG_SIZE / 2;
    let conv_out_w = IMG_SIZE / 2;
    let flat_dim = BASE_CH * conv_out_h * conv_out_w;

    let mut b = TensorBlockBuilder::new("doc_page_preprocess");
    let image = b.add_input("page", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let conv_w = b.add_input("conv_w", &[BASE_CH, IN_CHANNELS, 2, 2]);
    let conv_b = b.add_input("conv_b", &[BASE_CH]);
    let conv_out = b.add_conv2d(
        image,
        conv_w,
        Some(conv_b),
        2,
        2,
        0,
        0,
        &[BASE_CH, conv_out_h, conv_out_w],
    );
    let act = b.add_relu(conv_out, &[BASE_CH, conv_out_h, conv_out_w]);
    let flat = b.add_reshape(act, &[1, flat_dim]);
    let fc_w = b.add_input("fc_w", &[HIDDEN_DIM, flat_dim]);
    let fc_b = b.add_input("fc_b", &[HIDDEN_DIM]);
    let out = b.add_linear(flat, fc_w, Some(fc_b), &[1, HIDDEN_DIM]);
    let def = b.build(out).expect("valid page preprocess kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[BASE_CH, IN_CHANNELS, 2, 2]),
        bias_zero(&[BASE_CH]),
        weight(&[HIDDEN_DIM, flat_dim]),
        bias_zero(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(IN_CHANNELS, IMG_SIZE, IMG_SIZE);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[1, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("page preprocess IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 10. ROI extraction bounds (IBP)
// ===========================================================================

#[test]
fn test_full_doc_roi_extraction_ibp() {
    // ROI (Region of Interest) extraction: feature pooling from detected regions.
    // Models spatial pooling of features within bounding box coordinates.
    // Input features [NUM_BOXES, HIDDEN_DIM] -> Linear -> ReLU -> Linear -> [NUM_BOXES, HIDDEN_DIM]
    let mut b = TensorBlockBuilder::new("doc_roi_extract");
    let input = b.add_input("region_features", &[NUM_BOXES, HIDDEN_DIM]);
    let fc1_w = b.add_input("fc1_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let fc1_b = b.add_input("fc1_b", &[HIDDEN_DIM]);
    let h = b.add_linear(input, fc1_w, Some(fc1_b), &[NUM_BOXES, HIDDEN_DIM]);
    let act = b.add_relu(h, &[NUM_BOXES, HIDDEN_DIM]);
    let fc2_w = b.add_input("fc2_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let fc2_b = b.add_input("fc2_b", &[HIDDEN_DIM]);
    let out = b.add_linear(act, fc2_w, Some(fc2_b), &[NUM_BOXES, HIDDEN_DIM]);
    let def = b.build(out).expect("valid ROI extraction kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(NUM_BOXES, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[NUM_BOXES, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ROI extraction IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 11. Text line grouping from layout (IBP + CROWN)
// ===========================================================================

#[test]
fn test_full_doc_text_line_grouping_ibp_crown() {
    // Text line grouping: group detected text regions by spatial proximity.
    // Models a two-layer MLP that scores pairwise region affinity.
    // Input: [SEQ_LEN, HIDDEN_DIM] -> Linear -> GELU -> Linear -> sigmoid
    let mut b = TensorBlockBuilder::new("doc_text_line_group");
    let input = b.add_input("region_pairs", &[SEQ_LEN, HIDDEN_DIM]);
    let fc1_w = b.add_input("fc1_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let fc1_b = b.add_input("fc1_b", &[HIDDEN_DIM]);
    let h = b.add_linear(input, fc1_w, Some(fc1_b), &[SEQ_LEN, HIDDEN_DIM]);
    let act = b.add_gelu(h, &[SEQ_LEN, HIDDEN_DIM]);
    let fc2_w = b.add_input("fc2_w", &[1, HIDDEN_DIM]);
    let fc2_b = b.add_input("fc2_b", &[1]);
    let logits = b.add_linear(act, fc2_w, Some(fc2_b), &[SEQ_LEN, 1]);
    let out = b.add_sigmoid(logits, &[SEQ_LEN, 1]);
    let def = b.build(out).expect("valid text line grouping kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        weight(&[1, HIDDEN_DIM]),
        bias_zero(&[1]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("text line grouping IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &inp);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("text line grouping CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 12. Table cell content extraction (IBP)
// ===========================================================================

#[test]
fn test_full_doc_table_cell_content_extraction_ibp() {
    // Table cell content extraction: cell coordinates -> feature crop -> projection
    // Models ROI pooling + linear projection for each detected table cell.
    // Input: [NUM_CELLS, HIDDEN_DIM] -> Linear -> LayerNorm -> [NUM_CELLS, HIDDEN_DIM]
    let mut b = TensorBlockBuilder::new("doc_cell_content");
    let input = b.add_input("cell_rois", &[NUM_CELLS, HIDDEN_DIM]);
    let fc_w = b.add_input("fc_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let fc_b = b.add_input("fc_b", &[HIDDEN_DIM]);
    let proj = b.add_linear(input, fc_w, Some(fc_b), &[NUM_CELLS, HIDDEN_DIM]);

    let ln_w = b.add_input("ln_w", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_b", &[HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let out = b.add_layer_norm(proj, eps, 1, ln_w, ln_b, &[NUM_CELLS, HIDDEN_DIM]);
    let def = b.build(out).expect("valid cell content kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        ones(&[HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
        eps_binding(),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let inp = seq_bounds(NUM_CELLS, HIDDEN_DIM, 1.0);

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[NUM_CELLS, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("cell content extraction IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 13. Cross-model dimension compatibility (IBP)
// ===========================================================================

#[test]
fn test_full_doc_cross_model_dimension_compatibility_ibp() {
    // Verify that pipeline stages with matching dimensions produce compatible bounds.
    // Stage A output [SEQ_LEN, HIDDEN_DIM] feeds Stage B input [SEQ_LEN, HIDDEN_DIM].
    // Same Linear -> GELU block run on two different input ranges should give
    // nested output bounds when inputs are nested.

    let mut b = TensorBlockBuilder::new("doc_cross_model_compat");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let fc_w = b.add_input("fc_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let fc_b = b.add_input("fc_b", &[HIDDEN_DIM]);
    let h = b.add_linear(input, fc_w, Some(fc_b), &[SEQ_LEN, HIDDEN_DIM]);
    let out = b.add_gelu(h, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid cross-model kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        bias_zero(&[HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Narrow input (layout output bounds)
    let narrow_inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 0.5);
    let narrow_out = graph.propagate_ibp(&narrow_inp).expect("IBP narrow");
    assert_bounds_valid(&narrow_out);

    // Wide input (larger perturbation)
    let wide_inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 2.0);
    let wide_out = graph.propagate_ibp(&wide_inp).expect("IBP wide");
    assert_bounds_valid(&wide_out);

    let (narrow_lo, narrow_hi) = bounds_min_max(&narrow_out);
    let (wide_lo, wide_hi) = bounds_min_max(&wide_out);
    let narrow_width = narrow_hi - narrow_lo;
    let wide_width = wide_hi - wide_lo;

    eprintln!("cross-model compat: narrow=[{narrow_lo:.6}, {narrow_hi:.6}] w={narrow_width:.4}");
    eprintln!("cross-model compat: wide=[{wide_lo:.6}, {wide_hi:.6}] w={wide_width:.4}");

    // Wider input should produce wider (or comparable) output bounds
    assert!(
        wide_width >= narrow_width * 0.9,
        "wider input should produce at least comparable output width: {wide_width} < {narrow_width} * 0.9"
    );
}

// ===========================================================================
// 14. Confidence score aggregation (IBP)
// ===========================================================================

#[test]
fn test_full_doc_confidence_score_aggregation_ibp() {
    // Confidence score aggregation: combine detection + table + OCR scores.
    // Models a learned weighted combination of per-stage confidence scores.
    // Input: [NUM_BOXES, 3] (3 confidence scores) -> Linear -> sigmoid -> [NUM_BOXES, 1]
    let num_scores = 3;

    let mut b = TensorBlockBuilder::new("doc_confidence_agg");
    let input = b.add_input("scores", &[NUM_BOXES, num_scores]);
    let fc_w = b.add_input("fc_w", &[1, num_scores]);
    let fc_b = b.add_input("fc_b", &[1]);
    let logits = b.add_linear(input, fc_w, Some(fc_b), &[NUM_BOXES, 1]);
    let out = b.add_sigmoid(logits, &[NUM_BOXES, 1]);
    let def = b.build(out).expect("valid confidence aggregation kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[1, num_scores]),
        bias_zero(&[1]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Confidence scores in [0, 1]
    let inp = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[NUM_BOXES, num_scores]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[NUM_BOXES, num_scores]), 1.0f32),
    )
    .expect("valid confidence bounds");

    let output = graph.propagate_ibp(&inp).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[NUM_BOXES, 1]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("confidence aggregation IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Sigmoid output must be in [0, 1]
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 15. Multi-page batch processing bounds (IBP)
// ===========================================================================

#[test]
fn test_full_doc_multi_page_batch_processing_ibp() {
    // Multi-page batch: verify bounds are consistent across different "pages"
    // modeled as different input ranges through the same pipeline.
    // Pipeline: Linear -> GELU -> Linear -> softmax
    let mut b = TensorBlockBuilder::new("doc_multi_page_batch");
    let input = b.add_input("page_features", &[SEQ_LEN, HIDDEN_DIM]);
    let fc1_w = b.add_input("fc1_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let h = b.add_linear(input, fc1_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let act = b.add_gelu(h, &[SEQ_LEN, HIDDEN_DIM]);
    let fc2_w = b.add_input("fc2_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let fc2_b = b.add_input("fc2_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(act, fc2_w, Some(fc2_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);
    let def = b.build(out).expect("valid multi-page kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        bias_zero(&[VOCAB_SIZE]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Page 1: tight input (clean scan)
    let page1_inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 0.5);
    let page1_out = graph.propagate_ibp(&page1_inp).expect("IBP page 1");
    assert_bounds_valid(&page1_out);

    // Page 2: medium input (standard quality)
    let page2_inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 1.0);
    let page2_out = graph.propagate_ibp(&page2_inp).expect("IBP page 2");
    assert_bounds_valid(&page2_out);

    // Page 3: wide input (noisy/degraded scan)
    let page3_inp = seq_bounds(SEQ_LEN, HIDDEN_DIM, 2.0);
    let page3_out = graph.propagate_ibp(&page3_inp).expect("IBP page 3");
    assert_bounds_valid(&page3_out);

    // All pages should produce softmax output in [0, 1]
    for (label, out) in [
        ("page1", &page1_out),
        ("page2", &page2_out),
        ("page3", &page3_out),
    ] {
        let (lo, hi) = bounds_min_max(out);
        eprintln!("multi-page {label} IBP: bounds=[{lo:.6}, {hi:.6}]");
        assert!(lo >= -1e-5, "{label}: softmax lower >= 0, got {lo}");
        assert!(hi <= 1.0 + 1e-5, "{label}: softmax upper <= 1, got {hi}");
    }

    // Monotonicity: tighter input -> tighter output
    let width1 = {
        let (lo, hi) = bounds_min_max(&page1_out);
        hi - lo
    };
    let width3 = {
        let (lo, hi) = bounds_min_max(&page3_out);
        hi - lo
    };
    eprintln!("multi-page widths: page1={width1:.4}, page3={width3:.4}");
    assert!(
        width3 >= width1 * 0.9,
        "wider input should produce at least comparable output width"
    );
}
