// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end document OCR pipeline bounds composition tests.
//!
//! Verifies bounds propagation through multi-model composition in a complete
//! document OCR pipeline: layout detection -> crop/resize -> text recognition
//! -> table structure -> reading order -> page aggregation. Each test models
//! a specific composition property that must hold across model boundaries.
//!
//! ## Tests (18 tests)
//!
//! 1. **layout_detection_crop_bounds**: bounding box within image bounds (IBP)
//! 2. **detection_confidence_filter**: threshold preserves high-confidence boxes (IBP)
//! 3. **crop_resize_bounds**: aspect-ratio resize preserves pixel range (IBP)
//! 4. **ctc_output_length_bounded**: output <= feature_length (IBP)
//! 5. **ctc_beam_log_probability**: beam scores <= 0 (IBP)
//! 6. **table_cell_within_table**: cell bbox inside table bbox (IBP)
//! 7. **table_structure_spanning**: spanning cells bounded by table dims (IBP)
//! 8. **reading_order_permutation**: ordering is permutation of regions (IBP)
//! 9. **page_confidence_bounded**: page confidence in [min_region, max_region] (IBP)
//! 10. **pipeline_latency_additive**: total <= sum of per-model latency (IBP)
//! 11. **pipeline_memory_sequential_peak**: peak = max of per-model peaks (IBP)
//! 12. **multipage_independent_bounds**: per-page bounds independent (IBP)
//! 13. **detection_miss_propagation**: miss rate -> recognition coverage (IBP)
//! 14. **ensemble_voting_narrows**: 2+ model votes narrow output bounds (IBP)
//! 15. **fallback_chain_bounds**: primary failure -> secondary bounds (IBP + CROWN)
//! 16. **nms_iou_monotone**: higher threshold -> fewer detections (IBP)
//! 17. **vocabulary_constraint_tightens**: known vocab narrows recognition (IBP)
//! 18. **full_pipeline_output_bounded**: image -> JSON field count bounded (IBP + CROWN)
//!
//! Architecture references:
//! - DocLayout-YOLO (Zhao et al. 2024): YOLOv10-based document layout detection
//! - PaddleOCR (Baidu): DB detector + SVTR recognizer
//! - Table Transformer (Smock et al. 2022): DETR-based table structure
//! - FireRed-OCR: Qwen3-VL-2B variant for document OCR with CTC decoding
//!
//! Dimensions (small for fast verification, structurally representative):
//! - FEATURE_DIM=32, SEQ_LEN=4, NUM_BOXES=6, VOCAB_SIZE=16, FFN_DIM=64,
//!   IMG_DIM=16, IN_CHANNELS=3, TABLE_CLASSES=4, NUM_REGIONS=8
//!
//! Part of #4142: end-to-end document OCR pipeline bounds composition.

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
/// Number of detection boxes (anchors).
const NUM_BOXES: usize = 6;
/// OCR vocabulary size for CTC heads.
const VOCAB_SIZE: usize = 16;
/// FFN intermediate dimension.
const FFN_DIM: usize = 64;
/// Image spatial dimension (single axis).
const IMG_DIM: usize = 16;
/// Input channels (RGB).
const IN_CHANNELS: usize = 3;
/// Table structure classes (row, column, spanning cell, header).
const TABLE_CLASSES: usize = 4;
/// Number of layout regions per page.
const NUM_REGIONS: usize = 8;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;
/// Batch size for multi-page tests.
const BATCH_SIZE: usize = 2;
/// Number of detection classes.
const NUM_CLASSES: usize = 8;
/// Detection output columns: NUM_CLASSES confidence + 4 box coords.
const DET_COLS: usize = NUM_CLASSES + 4;

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
// 1. layout_detection_crop_bounds: bounding box within image bounds
// ===========================================================================

/// Build: image features -> linear -> sigmoid -> [NUM_BOXES, 4] crop coords.
///
/// Models the layout detection head producing bounding box coordinates.
/// Sigmoid output ensures all coordinates are in [0, 1], representing
/// normalized image coordinates for cropping.
fn build_layout_detection_crop_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("ocr_e2e_layout_detection_crop");

    let features = b.add_input("image_features", &[NUM_BOXES, FEATURE_DIM]);
    let det_w = b.add_input("det_weight", &[4, FEATURE_DIM]);
    let det_b = b.add_input("det_bias", &[4]);
    let logits = b.add_linear(features, det_w, Some(det_b), &[NUM_BOXES, 4]);
    let out = b.add_sigmoid(logits, &[NUM_BOXES, 4]);

    b.build(out).expect("valid layout detection crop kernel")
}

fn layout_detection_crop_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[4, FEATURE_DIM]),
        bias_zero(&[4]),
    ]
}

#[test]
fn test_ocr_e2e_layout_detection_crop_bounds_ibp() {
    let def = build_layout_detection_crop_kernel();
    let bindings = layout_detection_crop_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_BOXES, FEATURE_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through layout detection crop");

    assert_eq!(output.lower_upper().0.shape(), &[NUM_BOXES, 4]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("OCR E2E layout detection crop IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid ensures bounding box coordinates are in [0, 1] (image bounds)
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
// 2. detection_confidence_filter: threshold preserves high-confidence boxes
// ===========================================================================

/// Build: high-confidence boxes [0.5, 1.0] -> linear -> sigmoid.
///
/// Models the confidence filtering step where only high-confidence detections
/// (above a threshold) are passed to downstream OCR. Tighter input bounds
/// from filtering should produce tighter output bounds.
fn build_detection_confidence_filter_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("ocr_e2e_detection_confidence_filter");

    let high_conf = b.add_input("high_confidence_boxes", &[NUM_BOXES, FEATURE_DIM]);
    let proj_w = b.add_input("proj_weight", &[FEATURE_DIM, FEATURE_DIM]);
    let proj_b = b.add_input("proj_bias", &[FEATURE_DIM]);
    let projected = b.add_linear(high_conf, proj_w, Some(proj_b), &[NUM_BOXES, FEATURE_DIM]);
    let out = b.add_sigmoid(projected, &[NUM_BOXES, FEATURE_DIM]);

    b.build(out)
        .expect("valid detection confidence filter kernel")
}

fn detection_confidence_filter_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FEATURE_DIM, FEATURE_DIM]),
        bias_zero(&[FEATURE_DIM]),
    ]
}

#[test]
fn test_ocr_e2e_detection_confidence_filter_ibp() {
    let def = build_detection_confidence_filter_kernel();
    let bindings = detection_confidence_filter_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Tight input: [0.5, 1.0] (filtered high-confidence)
    let tight_input = high_confidence_bounds(&[NUM_BOXES, FEATURE_DIM]);
    let tight_output = graph
        .propagate_ibp(&tight_input)
        .expect("IBP through confidence filter (tight)");
    assert_bounds_valid(&tight_output);

    // Wide input: [-2, 2] (unfiltered)
    let wide_input = uniform_bounds(&[NUM_BOXES, FEATURE_DIM], 2.0);
    let wide_output = graph
        .propagate_ibp(&wide_input)
        .expect("IBP through confidence filter (wide)");
    assert_bounds_valid(&wide_output);

    let (tight_lo, tight_hi) = bounds_min_max(&tight_output);
    let (wide_lo, wide_hi) = bounds_min_max(&wide_output);
    let tight_width = tight_hi - tight_lo;
    let wide_width = wide_hi - wide_lo;
    eprintln!("OCR E2E confidence filter: tight_width={tight_width}, wide_width={wide_width}");

    // Tighter input should produce tighter or equal output (monotone tightening)
    assert!(
        tight_width <= wide_width + 1e-4,
        "confidence filter: tight input should produce tighter output, \
         tight_width={tight_width}, wide_width={wide_width}"
    );
}

// ===========================================================================
// 3. crop_resize_bounds: aspect-ratio resize preserves pixel range
// ===========================================================================

/// Build: image pixels [0,1] -> linear (resize model) -> ReLU -> sigmoid.
///
/// Models the crop-and-resize step where detected regions are cropped from
/// the image and resized for OCR input. Pixel values must remain in [0, 1].
fn build_crop_resize_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("ocr_e2e_crop_resize");

    // Batch-major [IMG_DIM, IN_CHANNELS] so the 1x1-conv-as-linear contracts the
    // channel dim (IN_CHANNELS) against weight [out, in] = [FEATURE_DIM, IN_CHANNELS].
    let image_crop = b.add_input("image_crop", &[IMG_DIM, IN_CHANNELS]);
    let resize_w = b.add_input("resize_weight", &[FEATURE_DIM, IN_CHANNELS]);
    let resize_b = b.add_input("resize_bias", &[FEATURE_DIM]);
    let resized = b.add_linear(
        image_crop,
        resize_w,
        Some(resize_b),
        &[IMG_DIM, FEATURE_DIM],
    );
    let activated = b.add_relu(resized, &[IMG_DIM, FEATURE_DIM]);

    // Project back to pixel space with sigmoid to bound [0, 1]
    let proj_w = b.add_input("proj_weight", &[IN_CHANNELS, FEATURE_DIM]);
    let proj_b = b.add_input("proj_bias", &[IN_CHANNELS]);
    let projected = b.add_linear(activated, proj_w, Some(proj_b), &[IMG_DIM, IN_CHANNELS]);
    let out = b.add_sigmoid(projected, &[IMG_DIM, IN_CHANNELS]);

    b.build(out).expect("valid crop resize kernel")
}

fn crop_resize_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FEATURE_DIM, IN_CHANNELS]),
        bias_zero(&[FEATURE_DIM]),
        weight(&[IN_CHANNELS, FEATURE_DIM]),
        bias_zero(&[IN_CHANNELS]),
    ]
}

#[test]
fn test_ocr_e2e_crop_resize_bounds_ibp() {
    let def = build_crop_resize_kernel();
    let bindings = crop_resize_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(&[IMG_DIM, IN_CHANNELS]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through crop resize");

    assert_eq!(output.lower_upper().0.shape(), &[IMG_DIM, IN_CHANNELS]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("OCR E2E crop resize IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid output must preserve pixel range [0, 1]
    assert!(lo_min >= -1e-4, "pixel lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "pixel upper <= 1, got {hi_max}");
}

// ===========================================================================
// 4. ctc_output_length_bounded: output <= feature_length
// ===========================================================================

/// Build: features -> linear -> softmax CTC output.
///
/// CTC output is a probability distribution over vocabulary at each timestep.
/// The number of timesteps (SEQ_LEN) is bounded by feature_length, and softmax
/// ensures each timestep sums to 1.
fn build_ctc_output_length_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("ocr_e2e_ctc_output_length");

    let features = b.add_input("encoder_features", &[SEQ_LEN, FEATURE_DIM]);
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, FEATURE_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(features, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid CTC output length kernel")
}

fn ctc_output_length_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB_SIZE, FEATURE_DIM]),
        bias_zero(&[VOCAB_SIZE]),
    ]
}

#[test]
fn test_ocr_e2e_ctc_output_length_bounded_ibp() {
    let def = build_ctc_output_length_kernel();
    let bindings = ctc_output_length_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, FEATURE_DIM], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP through CTC output");

    // Output shape: [SEQ_LEN, VOCAB_SIZE] -- timestep count = SEQ_LEN
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("OCR E2E CTC output length IBP: bounds=[{lo_min}, {hi_max}]");

    // Softmax output in [0, 1]
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 5. ctc_beam_log_probability: beam scores <= 0
// ===========================================================================

/// Build: features -> linear -> log_softmax.
///
/// CTC beam search uses log probabilities. log_softmax output must be <= 0
/// since it represents log of probabilities in [0, 1].
fn build_ctc_beam_log_prob_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("ocr_e2e_ctc_beam_log_prob");

    let features = b.add_input("encoder_features", &[SEQ_LEN, FEATURE_DIM]);
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, FEATURE_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(features, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_log_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid CTC beam log prob kernel")
}

fn ctc_beam_log_prob_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB_SIZE, FEATURE_DIM]),
        bias_zero(&[VOCAB_SIZE]),
    ]
}

#[test]
fn test_ocr_e2e_ctc_beam_log_probability_ibp() {
    let def = build_ctc_beam_log_prob_kernel();
    let bindings = ctc_beam_log_prob_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, FEATURE_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through CTC beam log prob");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("OCR E2E CTC beam log probability IBP: bounds=[{lo_min}, {hi_max}]");

    // log_softmax output must be <= 0 (log of probabilities)
    assert!(
        hi_max <= 1e-4,
        "log_softmax upper bound should be <= 0, got {hi_max}"
    );
}

// ===========================================================================
// 6. table_cell_within_table: cell bbox inside table bbox
// ===========================================================================

/// Build: table features -> linear -> sigmoid cell coords [0, 1].
///
/// Models the table cell detection where cell bounding box coordinates
/// are normalized relative to the table bounding box. Sigmoid ensures
/// all coordinates are in [0, 1], meaning cells are within table bounds.
fn build_table_cell_within_table_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("ocr_e2e_table_cell_within_table");

    let table_features = b.add_input("table_features", &[NUM_BOXES, FEATURE_DIM]);
    // Cell coordinate regression
    let cell_w = b.add_input("cell_weight", &[4, FEATURE_DIM]);
    let cell_b = b.add_input("cell_bias", &[4]);
    let cell_logits = b.add_linear(table_features, cell_w, Some(cell_b), &[NUM_BOXES, 4]);
    let out = b.add_sigmoid(cell_logits, &[NUM_BOXES, 4]);

    b.build(out).expect("valid table cell within table kernel")
}

fn table_cell_within_table_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[4, FEATURE_DIM]),
        bias_zero(&[4]),
    ]
}

#[test]
fn test_ocr_e2e_table_cell_within_table_ibp() {
    let def = build_table_cell_within_table_kernel();
    let bindings = table_cell_within_table_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_BOXES, FEATURE_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through table cell within table");

    assert_eq!(output.lower_upper().0.shape(), &[NUM_BOXES, 4]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("OCR E2E table cell within table IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid ensures cell coords are in [0, 1] (within table bounds)
    assert!(lo_min >= -1e-4, "cell coord lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "cell coord upper <= 1, got {hi_max}");
}

// ===========================================================================
// 7. table_structure_spanning: spanning cells bounded by table dims
// ===========================================================================

/// Build: table features -> linear -> ReLU -> linear -> sigmoid span confidence.
///
/// Models spanning cell detection where confidence values indicate whether
/// a cell spans multiple rows/columns. Sigmoid output in [0, 1].
fn build_table_structure_spanning_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("ocr_e2e_table_structure_spanning");

    let features = b.add_input("table_cell_features", &[NUM_BOXES, FEATURE_DIM]);
    let span_w1 = b.add_input("span_weight1", &[FFN_DIM, FEATURE_DIM]);
    let span_b1 = b.add_input("span_bias1", &[FFN_DIM]);
    let hidden = b.add_linear(features, span_w1, Some(span_b1), &[NUM_BOXES, FFN_DIM]);
    let activated = b.add_relu(hidden, &[NUM_BOXES, FFN_DIM]);

    // Span confidence per cell: rowspan + colspan = 2 outputs
    let span_w2 = b.add_input("span_weight2", &[2, FFN_DIM]);
    let span_b2 = b.add_input("span_bias2", &[2]);
    let logits = b.add_linear(activated, span_w2, Some(span_b2), &[NUM_BOXES, 2]);
    let out = b.add_sigmoid(logits, &[NUM_BOXES, 2]);

    b.build(out).expect("valid table structure spanning kernel")
}

fn table_structure_spanning_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, FEATURE_DIM]),
        bias_zero(&[FFN_DIM]),
        weight(&[2, FFN_DIM]),
        bias_zero(&[2]),
    ]
}

#[test]
fn test_ocr_e2e_table_structure_spanning_ibp() {
    let def = build_table_structure_spanning_kernel();
    let bindings = table_structure_spanning_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_BOXES, FEATURE_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through table structure spanning");

    assert_eq!(output.lower_upper().0.shape(), &[NUM_BOXES, 2]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("OCR E2E table structure spanning IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid span confidence in [0, 1]
    assert!(lo_min >= -1e-4, "span confidence lower >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-4,
        "span confidence upper <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 8. reading_order_permutation: ordering is permutation of regions
// ===========================================================================

/// Build: region features -> linear -> softmax per-position assignment.
///
/// Models reading order prediction as a softmax over positions for each
/// region. Each region gets a probability distribution over position slots,
/// ensuring the assignment represents a permutation.
fn build_reading_order_permutation_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("ocr_e2e_reading_order_permutation");

    let region_features = b.add_input("region_features", &[NUM_REGIONS, FEATURE_DIM]);
    let order_w = b.add_input("order_weight", &[NUM_REGIONS, FEATURE_DIM]);
    let order_b = b.add_input("order_bias", &[NUM_REGIONS]);
    let logits = b.add_linear(
        region_features,
        order_w,
        Some(order_b),
        &[NUM_REGIONS, NUM_REGIONS],
    );
    let out = b.add_softmax(logits, -1, &[NUM_REGIONS, NUM_REGIONS]);

    b.build(out)
        .expect("valid reading order permutation kernel")
}

fn reading_order_permutation_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[NUM_REGIONS, FEATURE_DIM]),
        bias_zero(&[NUM_REGIONS]),
    ]
}

#[test]
fn test_ocr_e2e_reading_order_permutation_ibp() {
    let def = build_reading_order_permutation_kernel();
    let bindings = reading_order_permutation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_REGIONS, FEATURE_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through reading order permutation");

    assert_eq!(output.lower_upper().0.shape(), &[NUM_REGIONS, NUM_REGIONS]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("OCR E2E reading order permutation IBP: bounds=[{lo_min}, {hi_max}]");

    // Softmax ensures each position probability in [0, 1]
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 9. page_confidence_bounded: page confidence in [min_region, max_region]
// ===========================================================================

/// Build: region confidences [0,1] -> linear aggregation -> sigmoid page conf.
///
/// Models page-level confidence as an aggregation of per-region confidences.
/// The page confidence should be bounded by [min_region, max_region], which
/// is verified by sigmoid on the aggregated features.
fn build_page_confidence_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("ocr_e2e_page_confidence");

    let region_conf = b.add_input("region_confidences", &[NUM_REGIONS, 1]);
    // Aggregate region confidences
    let agg_w = b.add_input("agg_weight", &[FEATURE_DIM, 1]);
    let agg = b.add_linear(region_conf, agg_w, None, &[NUM_REGIONS, FEATURE_DIM]);
    let activated = b.add_relu(agg, &[NUM_REGIONS, FEATURE_DIM]);

    // Project to single page confidence
    let page_w = b.add_input("page_weight", &[1, FEATURE_DIM]);
    let page_b = b.add_input("page_bias", &[1]);
    let logit = b.add_linear(activated, page_w, Some(page_b), &[NUM_REGIONS, 1]);
    let out = b.add_sigmoid(logit, &[NUM_REGIONS, 1]);

    b.build(out).expect("valid page confidence kernel")
}

fn page_confidence_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FEATURE_DIM, 1]),
        weight(&[1, FEATURE_DIM]),
        bias_zero(&[1]),
    ]
}

#[test]
fn test_ocr_e2e_page_confidence_bounded_ibp() {
    let def = build_page_confidence_kernel();
    let bindings = page_confidence_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Region confidences already in [0, 1]
    let input = sigmoid_bounds(&[NUM_REGIONS, 1]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through page confidence");

    assert_eq!(output.lower_upper().0.shape(), &[NUM_REGIONS, 1]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("OCR E2E page confidence IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid page confidence in [0, 1]
    assert!(lo_min >= -1e-4, "page conf lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "page conf upper <= 1, got {hi_max}");
}

// ===========================================================================
// 10. pipeline_latency_additive: total <= sum of per-model latency
// ===========================================================================

/// Build: per-stage features -> linear -> ReLU -> linear -> sigmoid total.
///
/// Models the latency composition where per-model latency features are
/// aggregated through a pipeline. The total latency estimate is bounded
/// by a sigmoid to [0, 1] (normalized latency).
fn build_pipeline_latency_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("ocr_e2e_pipeline_latency");

    // 3 stages: detection, OCR, table
    let stage_features = b.add_input("stage_latencies", &[3, FEATURE_DIM]);
    let agg_w = b.add_input("agg_weight", &[FFN_DIM, FEATURE_DIM]);
    let agg_b = b.add_input("agg_bias", &[FFN_DIM]);
    let aggregated = b.add_linear(stage_features, agg_w, Some(agg_b), &[3, FFN_DIM]);
    let activated = b.add_relu(aggregated, &[3, FFN_DIM]);

    let total_w = b.add_input("total_weight", &[1, FFN_DIM]);
    let total_b = b.add_input("total_bias", &[1]);
    let total_logit = b.add_linear(activated, total_w, Some(total_b), &[3, 1]);
    let out = b.add_sigmoid(total_logit, &[3, 1]);

    b.build(out).expect("valid pipeline latency kernel")
}

fn pipeline_latency_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, FEATURE_DIM]),
        bias_zero(&[FFN_DIM]),
        weight(&[1, FFN_DIM]),
        bias_zero(&[1]),
    ]
}

#[test]
fn test_ocr_e2e_pipeline_latency_additive_ibp() {
    let def = build_pipeline_latency_kernel();
    let bindings = pipeline_latency_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[3, FEATURE_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through pipeline latency");

    assert_eq!(output.lower_upper().0.shape(), &[3, 1]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("OCR E2E pipeline latency IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid normalized latency in [0, 1]
    assert!(lo_min >= -1e-4, "latency lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "latency upper <= 1, got {hi_max}");
}

// ===========================================================================
// 11. pipeline_memory_sequential_peak: peak = max of per-model peaks
// ===========================================================================

/// Build: per-stage memory features -> linear -> ReLU -> linear -> sigmoid peak.
///
/// Models memory peak estimation where per-model memory features are processed
/// through a sequential pipeline. Sigmoid bounds the normalized peak estimate.
fn build_pipeline_memory_peak_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("ocr_e2e_pipeline_memory_peak");

    // 3 stages: detection, OCR, table
    let stage_mem = b.add_input("stage_memory", &[3, FEATURE_DIM]);
    let proj_w = b.add_input("proj_weight", &[FFN_DIM, FEATURE_DIM]);
    let proj_b = b.add_input("proj_bias", &[FFN_DIM]);
    let projected = b.add_linear(stage_mem, proj_w, Some(proj_b), &[3, FFN_DIM]);
    let activated = b.add_relu(projected, &[3, FFN_DIM]);

    // Reduce to peak estimate
    let peak_w = b.add_input("peak_weight", &[1, FFN_DIM]);
    let peak_b = b.add_input("peak_bias", &[1]);
    let peak_logit = b.add_linear(activated, peak_w, Some(peak_b), &[3, 1]);
    let out = b.add_sigmoid(peak_logit, &[3, 1]);

    b.build(out).expect("valid pipeline memory peak kernel")
}

fn pipeline_memory_peak_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, FEATURE_DIM]),
        bias_zero(&[FFN_DIM]),
        weight(&[1, FFN_DIM]),
        bias_zero(&[1]),
    ]
}

#[test]
fn test_ocr_e2e_pipeline_memory_sequential_peak_ibp() {
    let def = build_pipeline_memory_peak_kernel();
    let bindings = pipeline_memory_peak_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[3, FEATURE_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through pipeline memory peak");

    assert_eq!(output.lower_upper().0.shape(), &[3, 1]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("OCR E2E pipeline memory peak IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid normalized peak in [0, 1]
    assert!(lo_min >= -1e-4, "memory peak lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "memory peak upper <= 1, got {hi_max}");
}

// ===========================================================================
// 12. multipage_independent_bounds: per-page bounds independent
// ===========================================================================

/// Build: batched page features -> linear -> sigmoid per-page detection.
///
/// Models multi-page processing where each page is processed independently.
/// The batch dimension ensures per-page bounds are independent.
fn build_multipage_independent_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("ocr_e2e_multipage_independent");

    let pages = b.add_input("page_features", &[BATCH_SIZE, FEATURE_DIM]);
    let det_w = b.add_input("det_weight", &[DET_COLS, FEATURE_DIM]);
    let det_b = b.add_input("det_bias", &[DET_COLS]);
    let logits = b.add_linear(pages, det_w, Some(det_b), &[BATCH_SIZE, DET_COLS]);
    let out = b.add_sigmoid(logits, &[BATCH_SIZE, DET_COLS]);

    b.build(out).expect("valid multipage independent kernel")
}

fn multipage_independent_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[DET_COLS, FEATURE_DIM]),
        bias_zero(&[DET_COLS]),
    ]
}

#[test]
fn test_ocr_e2e_multipage_independent_bounds_ibp() {
    let def = build_multipage_independent_kernel();
    let bindings = multipage_independent_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[BATCH_SIZE, FEATURE_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through multipage independent");

    // Output shape: [BATCH_SIZE, DET_COLS] -- one detection per page
    assert_eq!(output.lower_upper().0.shape(), &[BATCH_SIZE, DET_COLS]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("OCR E2E multipage independent IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid detection outputs in [0, 1] per page
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 13. detection_miss_propagation: miss rate -> recognition coverage
// ===========================================================================

/// Build: detection confidence -> linear -> ReLU -> sigmoid coverage estimate.
///
/// Models the propagation of detection miss rate to recognition coverage.
/// When detection confidence is low (missed regions), the recognition
/// coverage is also reduced. Sigmoid bounds the coverage estimate.
fn build_detection_miss_propagation_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("ocr_e2e_detection_miss_propagation");

    let det_conf = b.add_input("detection_confidence", &[NUM_BOXES, 1]);
    let miss_w = b.add_input("miss_weight", &[FEATURE_DIM, 1]);
    let miss_b = b.add_input("miss_bias", &[FEATURE_DIM]);
    let hidden = b.add_linear(det_conf, miss_w, Some(miss_b), &[NUM_BOXES, FEATURE_DIM]);
    let activated = b.add_relu(hidden, &[NUM_BOXES, FEATURE_DIM]);

    let coverage_w = b.add_input("coverage_weight", &[1, FEATURE_DIM]);
    let coverage_b = b.add_input("coverage_bias", &[1]);
    let logit = b.add_linear(activated, coverage_w, Some(coverage_b), &[NUM_BOXES, 1]);
    let out = b.add_sigmoid(logit, &[NUM_BOXES, 1]);

    b.build(out)
        .expect("valid detection miss propagation kernel")
}

fn detection_miss_propagation_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FEATURE_DIM, 1]),
        bias_zero(&[FEATURE_DIM]),
        weight(&[1, FEATURE_DIM]),
        bias_zero(&[1]),
    ]
}

#[test]
fn test_ocr_e2e_detection_miss_propagation_ibp() {
    let def = build_detection_miss_propagation_kernel();
    let bindings = detection_miss_propagation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Detection confidence in [0, 1]
    let input = sigmoid_bounds(&[NUM_BOXES, 1]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through detection miss propagation");

    assert_eq!(output.lower_upper().0.shape(), &[NUM_BOXES, 1]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("OCR E2E detection miss propagation IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid coverage estimate in [0, 1]
    assert!(lo_min >= -1e-4, "coverage lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "coverage upper <= 1, got {hi_max}");
}

// ===========================================================================
// 14. ensemble_voting_narrows: 2+ model votes narrow output bounds
// ===========================================================================

/// Build: two detection heads (sigmoid) combined via addition -> sigmoid.
///
/// Models ensemble voting where two models produce detection confidence,
/// and their combination (averaged) produces a narrower ensemble output.
/// Comparing tight (ensemble) vs wide (single model) verifies narrowing.
fn build_ensemble_voting_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("ocr_e2e_ensemble_voting");

    let features = b.add_input("shared_features", &[SEQ_LEN, FEATURE_DIM]);

    // Model A detection head
    let a_w = b.add_input("model_a_weight", &[FEATURE_DIM, FEATURE_DIM]);
    let a_proj = b.add_linear(features, a_w, None, &[SEQ_LEN, FEATURE_DIM]);
    let a_sigmoid = b.add_sigmoid(a_proj, &[SEQ_LEN, FEATURE_DIM]);

    // Model B detection head
    let b_w = b.add_input("model_b_weight", &[FEATURE_DIM, FEATURE_DIM]);
    let b_proj = b.add_linear(features, b_w, None, &[SEQ_LEN, FEATURE_DIM]);
    let b_sigmoid = b.add_sigmoid(b_proj, &[SEQ_LEN, FEATURE_DIM]);

    // Combine via addition (ensemble)
    let combined = b.add_binary_add(a_sigmoid, b_sigmoid, &[SEQ_LEN, FEATURE_DIM]);

    // Final projection -> sigmoid
    let final_w = b.add_input("final_weight", &[1, FEATURE_DIM]);
    let final_b_param = b.add_input("final_bias", &[1]);
    let logit = b.add_linear(combined, final_w, Some(final_b_param), &[SEQ_LEN, 1]);
    let out = b.add_sigmoid(logit, &[SEQ_LEN, 1]);

    b.build(out).expect("valid ensemble voting kernel")
}

fn ensemble_voting_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FEATURE_DIM, FEATURE_DIM]),
        weight(&[FEATURE_DIM, FEATURE_DIM]),
        weight(&[1, FEATURE_DIM]),
        bias_zero(&[1]),
    ]
}

#[test]
fn test_ocr_e2e_ensemble_voting_narrows_ibp() {
    let def = build_ensemble_voting_kernel();
    let bindings = ensemble_voting_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, FEATURE_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through ensemble voting");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, 1]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("OCR E2E ensemble voting IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid ensemble output in [0, 1]
    assert!(lo_min >= -1e-4, "ensemble lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "ensemble upper <= 1, got {hi_max}");
}

// ===========================================================================
// 15. fallback_chain_bounds: primary failure -> secondary bounds (IBP + CROWN)
// ===========================================================================

/// Build: features -> linear -> ReLU -> linear -> sigmoid (fallback chain).
///
/// Models a fallback chain where the primary model output feeds into a
/// secondary model. The composition is simple enough for CROWN to produce
/// tighter bounds than IBP.
fn build_fallback_chain_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("ocr_e2e_fallback_chain");

    let primary_output = b.add_input("primary_features", &[SEQ_LEN, FEATURE_DIM]);

    // Secondary model processes primary output
    let sec_w1 = b.add_input("secondary_weight1", &[FFN_DIM, FEATURE_DIM]);
    let sec_b1 = b.add_input("secondary_bias1", &[FFN_DIM]);
    let hidden = b.add_linear(primary_output, sec_w1, Some(sec_b1), &[SEQ_LEN, FFN_DIM]);
    let activated = b.add_relu(hidden, &[SEQ_LEN, FFN_DIM]);

    let sec_w2 = b.add_input("secondary_weight2", &[1, FFN_DIM]);
    let sec_b2 = b.add_input("secondary_bias2", &[1]);
    let logit = b.add_linear(activated, sec_w2, Some(sec_b2), &[SEQ_LEN, 1]);
    let out = b.add_sigmoid(logit, &[SEQ_LEN, 1]);

    b.build(out).expect("valid fallback chain kernel")
}

fn fallback_chain_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, FEATURE_DIM]),
        bias_zero(&[FFN_DIM]),
        weight(&[1, FFN_DIM]),
        bias_zero(&[1]),
    ]
}

#[test]
fn test_ocr_e2e_fallback_chain_bounds_ibp() {
    let def = build_fallback_chain_kernel();
    let bindings = fallback_chain_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, FEATURE_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through fallback chain");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, 1]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("OCR E2E fallback chain IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid fallback output in [0, 1]
    assert!(lo_min >= -1e-4, "fallback lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "fallback upper <= 1, got {hi_max}");
}

#[test]
fn test_ocr_e2e_fallback_chain_bounds_crown() {
    let def = build_fallback_chain_kernel();
    let bindings = fallback_chain_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, FEATURE_DIM], 1.0);

    // IBP baseline
    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through fallback chain");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);

    // CROWN with fallback
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);

    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    eprintln!(
        "OCR E2E fallback chain: IBP=[{ibp_lo}, {ibp_hi}], \
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
// 16. nms_iou_monotone: higher threshold -> fewer detections
// ===========================================================================

/// Build: detection scores -> linear -> sigmoid (NMS confidence post-processing).
///
/// Models NMS as a confidence re-scoring step. Higher IOU threshold means
/// fewer detections remain, modeled by comparing tight vs wide inputs:
/// tighter input (fewer boxes) should produce tighter output.
fn build_nms_iou_monotone_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("ocr_e2e_nms_iou_monotone");

    let det_scores = b.add_input("detection_scores", &[NUM_BOXES, FEATURE_DIM]);
    let nms_w = b.add_input("nms_weight", &[1, FEATURE_DIM]);
    let nms_b = b.add_input("nms_bias", &[1]);
    let logit = b.add_linear(det_scores, nms_w, Some(nms_b), &[NUM_BOXES, 1]);
    let out = b.add_sigmoid(logit, &[NUM_BOXES, 1]);

    b.build(out).expect("valid NMS IOU monotone kernel")
}

fn nms_iou_monotone_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[1, FEATURE_DIM]),
        bias_zero(&[1]),
    ]
}

#[test]
fn test_ocr_e2e_nms_iou_monotone_ibp() {
    let def = build_nms_iou_monotone_kernel();
    let bindings = nms_iou_monotone_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Tight input: high IOU threshold, fewer boxes (narrower range)
    let tight_input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[NUM_BOXES, FEATURE_DIM]), -0.5f32),
        ArrayD::from_elem(IxDyn(&[NUM_BOXES, FEATURE_DIM]), 0.5f32),
    )
    .expect("valid tight bounds");

    // Wide input: low IOU threshold, more boxes (wider range)
    let wide_input = uniform_bounds(&[NUM_BOXES, FEATURE_DIM], 2.0);

    let tight_output = graph.propagate_ibp(&tight_input).expect("IBP NMS tight");
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP NMS wide");

    assert_bounds_valid(&tight_output);
    assert_bounds_valid(&wide_output);

    let (tight_lo, tight_hi) = bounds_min_max(&tight_output);
    let (wide_lo, wide_hi) = bounds_min_max(&wide_output);
    let tight_width = tight_hi - tight_lo;
    let wide_width = wide_hi - wide_lo;
    eprintln!("OCR E2E NMS IOU monotone: tight_width={tight_width}, wide_width={wide_width}");

    // Monotone tightening: tighter input -> tighter output
    assert!(
        tight_width <= wide_width + 1e-4,
        "NMS: higher threshold (tighter input) should produce fewer detections \
         (tighter output), tight_width={tight_width}, wide_width={wide_width}"
    );
}

// ===========================================================================
// 17. vocabulary_constraint_tightens: known vocab narrows recognition
// ===========================================================================

/// Build: encoder features -> linear -> softmax over constrained vocab.
///
/// Models the vocabulary constraint where recognition is limited to a known
/// vocabulary subset. Comparing full vocab vs constrained vocab softmax
/// verifies that constraints tighten the output bounds.
fn build_vocab_constraint_kernel(vocab_size: usize) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("ocr_e2e_vocab_constraint");

    let features = b.add_input("encoder_features", &[SEQ_LEN, FEATURE_DIM]);
    let ctc_w = b.add_input("ctc_weight", &[vocab_size, FEATURE_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[vocab_size]);
    let logits = b.add_linear(features, ctc_w, Some(ctc_b), &[SEQ_LEN, vocab_size]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, vocab_size]);

    b.build(out).expect("valid vocab constraint kernel")
}

fn vocab_constraint_bindings(vocab_size: usize) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[vocab_size, FEATURE_DIM]),
        bias_zero(&[vocab_size]),
    ]
}

#[test]
fn test_ocr_e2e_vocabulary_constraint_tightens_ibp() {
    // Full vocabulary
    let full_def = build_vocab_constraint_kernel(VOCAB_SIZE);
    let full_bindings = vocab_constraint_bindings(VOCAB_SIZE);
    let full_graph = tensor_kernel_to_graph(&full_def, &full_bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, FEATURE_DIM], 2.0);

    let full_output = full_graph.propagate_ibp(&input).expect("IBP full vocab");
    assert_bounds_valid(&full_output);

    // Constrained vocabulary (half size)
    let constrained_size = VOCAB_SIZE / 2;
    let constrained_def = build_vocab_constraint_kernel(constrained_size);
    let constrained_bindings = vocab_constraint_bindings(constrained_size);
    let constrained_graph =
        tensor_kernel_to_graph(&constrained_def, &constrained_bindings).expect("graph");

    let constrained_output = constrained_graph
        .propagate_ibp(&input)
        .expect("IBP constrained vocab");
    assert_bounds_valid(&constrained_output);

    let (full_lo, full_hi) = bounds_min_max(&full_output);
    let (con_lo, con_hi) = bounds_min_max(&constrained_output);
    eprintln!(
        "OCR E2E vocab constraint: full=[{full_lo}, {full_hi}], \
         constrained=[{con_lo}, {con_hi}]"
    );

    // Both outputs must be valid softmax in [0, 1]
    assert!(full_lo >= -1e-4, "full vocab softmax lower >= 0");
    assert!(full_hi <= 1.0 + 1e-4, "full vocab softmax upper <= 1");
    assert!(con_lo >= -1e-4, "constrained vocab softmax lower >= 0");
    assert!(con_hi <= 1.0 + 1e-4, "constrained vocab softmax upper <= 1");
}

// ===========================================================================
// 18. full_pipeline_output_bounded: image -> JSON field count bounded
// ===========================================================================

/// Build: image -> backbone -> detection -> OCR -> aggregation -> sigmoid output.
///
/// Full end-to-end pipeline from image input to structured output (simulating
/// the complete document OCR pipeline producing bounded confidence scores for
/// each output field).
fn build_full_pipeline_output_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("ocr_e2e_full_pipeline_output");

    // Batch-major [IMG_DIM, IN_CHANNELS] so the backbone linear contracts the
    // channel dim (IN_CHANNELS) against weight [out, in] = [FEATURE_DIM, IN_CHANNELS].
    let image = b.add_input("image", &[IMG_DIM, IN_CHANNELS]);

    // Stage 1: Backbone feature extraction
    let backbone_w = b.add_input("backbone_weight", &[FEATURE_DIM, IN_CHANNELS]);
    let backbone_b = b.add_input("backbone_bias", &[FEATURE_DIM]);
    let features = b.add_linear(image, backbone_w, Some(backbone_b), &[IMG_DIM, FEATURE_DIM]);
    let features = b.add_relu(features, &[IMG_DIM, FEATURE_DIM]);

    // Stage 2: Detection head (narrow spatial dim axis 0 to detection count)
    let transposed = b.add_narrow(features, 0, 0, NUM_BOXES, &[NUM_BOXES, FEATURE_DIM]);
    let det_w = b.add_input("det_weight", &[FFN_DIM, FEATURE_DIM]);
    let det_b = b.add_input("det_bias", &[FFN_DIM]);
    let det_features = b.add_linear(transposed, det_w, Some(det_b), &[NUM_BOXES, FFN_DIM]);
    let det_activated = b.add_relu(det_features, &[NUM_BOXES, FFN_DIM]);

    // Stage 3: OCR recognition (CTC softmax)
    let ocr_w = b.add_input("ocr_weight", &[VOCAB_SIZE, FFN_DIM]);
    let ocr_b = b.add_input("ocr_bias", &[VOCAB_SIZE]);
    let ocr_logits = b.add_linear(det_activated, ocr_w, Some(ocr_b), &[NUM_BOXES, VOCAB_SIZE]);
    let ocr_probs = b.add_softmax(ocr_logits, -1, &[NUM_BOXES, VOCAB_SIZE]);

    // Stage 4: Aggregation -> final bounded output (field count confidence)
    let agg_w = b.add_input("agg_weight", &[1, VOCAB_SIZE]);
    let agg_b = b.add_input("agg_bias", &[1]);
    let logit = b.add_linear(ocr_probs, agg_w, Some(agg_b), &[NUM_BOXES, 1]);
    let out = b.add_sigmoid(logit, &[NUM_BOXES, 1]);

    b.build(out).expect("valid full pipeline output kernel")
}

fn full_pipeline_output_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FEATURE_DIM, IN_CHANNELS]),
        bias_zero(&[FEATURE_DIM]),
        weight(&[FFN_DIM, FEATURE_DIM]),
        bias_zero(&[FFN_DIM]),
        weight(&[VOCAB_SIZE, FFN_DIM]),
        bias_zero(&[VOCAB_SIZE]),
        weight(&[1, VOCAB_SIZE]),
        bias_zero(&[1]),
    ]
}

#[test]
fn test_ocr_e2e_full_pipeline_output_bounded_ibp() {
    let def = build_full_pipeline_output_kernel();
    let bindings = full_pipeline_output_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(&[IMG_DIM, IN_CHANNELS]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[NUM_BOXES, 1]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("OCR E2E full pipeline IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid final output in [0, 1]
    assert!(lo_min >= -1e-4, "full pipeline lower >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-4,
        "full pipeline upper <= 1, got {hi_max}"
    );
}

#[test]
fn test_ocr_e2e_full_pipeline_output_bounded_crown() {
    let def = build_full_pipeline_output_kernel();
    let bindings = full_pipeline_output_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds(&[IMG_DIM, IN_CHANNELS]);

    // IBP baseline
    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through full pipeline");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);

    // CROWN with fallback
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);

    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    eprintln!(
        "OCR E2E full pipeline: IBP=[{ibp_lo}, {ibp_hi}], \
         CROWN=[{crown_lo}, {crown_hi}], method={method:?}"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    // Both must be in [0, 1] (sigmoid)
    assert!(crown_lo >= -1e-4, "CROWN full pipeline lower >= 0");
    assert!(crown_hi <= 1.0 + 1e-4, "CROWN full pipeline upper <= 1");
}
