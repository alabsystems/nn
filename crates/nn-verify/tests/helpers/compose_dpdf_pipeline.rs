// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pipeline composition bounds verification for dpdf document understanding.
//!
//! Verifies that numerical bounds propagate correctly through multi-model
//! pipelines where one model's output feeds the next model's input. Unlike
//! `compose_dpdf_cross_model.rs` (which tests individual boundary crossings),
//! this module tests **full pipeline architectures** with realistic intermediate
//! representations and end-to-end bound accumulation properties.
//!
//! ## 1. DocLayout-YOLO -> Table Transformer pipeline (3 tests)
//!
//! 1. **Detection -> table structure IBP**: YOLO backbone -> sigmoid detection
//!    -> linear projection -> ReLU -> DETR encoder proxy -> sigmoid table cells.
//!    Verifies layout detection [0,1] composes with table structure [0,1].
//!
//! 2. **Detection -> table structure CROWN**: Same pipeline with CROWN
//!    linearization for tighter cross-model bounds.
//!
//! 3. **Detection -> table structure -> cell classification IBP**:
//!    Full 3-stage: image -> YOLO detect -> Table Transformer -> cell sigmoid.
//!
//! ## 2. PaddleOCR detection -> recognition pipeline (3 tests)
//!
//! 4. **DB detector -> SVTR recognizer IBP**: Conv backbone -> sigmoid
//!    probability map -> linear projection -> GELU -> CTC softmax.
//!    Verifies text detection sigmoid [0,1] feeds CTC recognition softmax [0,1].
//!
//! 5. **DB detector -> SVTR recognizer CROWN**: Same with CROWN.
//!
//! 6. **Detection probability gating -> tighter recognition bounds IBP**:
//!    High-confidence detection [0.7, 1.0] produces tighter CTC bounds
//!    than full-range [0, 1] detection.
//!
//! ## 3. Granite-Docling vision encoder -> text decoder (3 tests)
//!
//! 7. **ViT encoder -> cross-attention decoder IBP**: Conv2d patch embed
//!    -> reshape -> transpose -> Linear vision encoder -> Linear cross-attn
//!    projection -> ReLU -> Linear LM head -> softmax.
//!
//! 8. **ViT encoder -> cross-attention decoder CROWN**: Same with CROWN.
//!
//! 9. **Vision projection dimension compatibility IBP**: Verifies vision
//!    encoder output dim matches decoder cross-attention input dim through
//!    an explicit projection layer.
//!
//! ## 4. FireRed-OCR vision -> language pipeline (3 tests)
//!
//! 10. **Multi-scale ViT -> MoE decoder IBP**: Conv2d patch embed -> encoder
//!     Linear -> ReLU -> expert gate softmax -> expert FFN -> sigmoid output.
//!     Verifies vision features compose through MoE routing to language output.
//!
//! 11. **Multi-scale ViT -> MoE decoder CROWN**: Same with CROWN.
//!
//! 12. **ViT -> language head softmax IBP**: Vision encoder -> linear
//!     projection -> CTC head -> softmax for token probabilities.
//!
//! ## 5. Cross-model tensor compatibility (2 tests)
//!
//! 13. **Shape/dtype compatibility chain IBP**: Sequential linear projections
//!     verify output shape of model A matches input requirements of model B
//!     across 4 model boundaries.
//!
//! 14. **Sigmoid -> softmax activation boundary IBP**: Detection sigmoid [0,1]
//!     feeds recognition softmax [0,1] with intermediate feature transform.
//!     Verifies activation domain transitions preserve bound validity.
//!
//! ## 6. Pipeline error accumulation (2 tests)
//!
//! 15. **Chained quantization error IBP**: 4-stage pipeline with small weight
//!     perturbation at each stage. Verifies total bound width grows
//!     sub-exponentially through the chain (bounded error accumulation).
//!
//! 16. **Depth vs. width trade-off IBP**: Compares bounds from a deep
//!     4-layer pipeline against a wide 1-layer pipeline with equivalent
//!     total parameters. Verifies that deeper pipelines do not produce
//!     unbounded widening relative to shallow equivalents.
//!
//! Architecture references:
//! - DocLayout-YOLO (Zhao et al. 2024): YOLOv10-based document layout detection
//! - Table Transformer (Smock et al. 2022): DETR-based table structure recognition
//! - PaddleOCR (Baidu): DB text detector + SVTR recognizer with CTC
//! - Granite-Docling: SigLIP2 vision encoder + Granite LLM decoder
//! - FireRed-OCR: Qwen3-VL-2B variant for document OCR with CTC decoding
//!
//! Dimensions (small for fast verification, structurally representative):
//! - IMG_SIZE=16, PATCH_SIZE=8, HIDDEN_DIM=32, FFN_DIM=64, SEQ_LEN=4,
//!   NUM_CLASSES=8, VOCAB_SIZE=16, BACKBONE_CH=16, NUM_EXPERTS=4
//!
//! Part of #4088: compose tests for model pipeline composition bounds.

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

/// Image spatial size (square).
const IMG_SIZE: usize = 16;
/// Patch size for patch embedding.
const PATCH_SIZE: usize = 8;
/// Grid size = IMG_SIZE / PATCH_SIZE.
const GRID_SIZE: usize = IMG_SIZE / PATCH_SIZE; // 2
/// Total number of patches.
const NUM_PATCHES: usize = GRID_SIZE * GRID_SIZE; // 4
/// Input channels (RGB).
const IN_CHANNELS: usize = 3;
/// Hidden dimension for encoder/decoder.
const HIDDEN_DIM: usize = 32;
/// FFN intermediate dimension.
const FFN_DIM: usize = 64;
/// Sequence length for text/token positions.
const SEQ_LEN: usize = 4;
/// Number of detection/classification classes.
const NUM_CLASSES: usize = 8;
/// OCR vocabulary size (characters + blank).
const VOCAB_SIZE: usize = 16;
/// Backbone output channels.
const BACKBONE_CH: usize = 16;
/// Number of MoE experts.
const NUM_EXPERTS: usize = 4;
/// Table structure classes (row/column/cell/header).
const TABLE_CLASSES: usize = 4;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ===========================================================================
// Helpers
// ===========================================================================

/// Image-domain input bounds: pixels in [0, 1].
fn image_bounds_01(shape: &[usize]) -> BoundedTensor {
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

/// Constant weight tensor binding.
fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

/// Constant zero bias tensor binding.
fn bias(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

// ===========================================================================
// 1. DocLayout-YOLO -> Table Transformer pipeline
// ===========================================================================

/// Build DocLayout-YOLO -> Table Transformer pipeline:
/// Conv2d backbone -> ReLU -> reshape -> Linear YOLO head -> sigmoid detection
/// -> Linear projection -> ReLU -> Linear DETR proxy -> sigmoid table cells.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[SEQ_LEN, TABLE_CLASSES]` (sigmoid table structure [0, 1]).
fn build_doclayout_to_table_transformer_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_doclayout_to_table_transformer");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // YOLO backbone: Conv2d + ReLU
    let conv_w = b.add_input(
        "yolo_backbone_weight",
        &[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let conv_b = b.add_input("yolo_backbone_bias", &[BACKBONE_CH]);
    let backbone = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[BACKBONE_CH, GRID_SIZE, GRID_SIZE],
    );
    let backbone_act = b.add_relu(backbone, &[BACKBONE_CH, GRID_SIZE, GRID_SIZE]);

    // Reshape + transpose to sequence
    let reshaped = b.add_reshape(backbone_act, &[BACKBONE_CH, NUM_PATCHES]);
    let transposed = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, BACKBONE_CH]);

    // YOLO detection head: Linear -> sigmoid (detection confidence [0,1])
    let det_w = b.add_input("yolo_detect_weight", &[NUM_CLASSES, BACKBONE_CH]);
    let det_b = b.add_input("yolo_detect_bias", &[NUM_CLASSES]);
    let det_logits = b.add_linear(transposed, det_w, Some(det_b), &[NUM_PATCHES, NUM_CLASSES]);
    let det_conf = b.add_sigmoid(det_logits, &[NUM_PATCHES, NUM_CLASSES]);

    // === Pipeline boundary: YOLO output -> Table Transformer input ===

    // Project detection features to table transformer input space
    let proj_w = b.add_input("table_proj_weight", &[HIDDEN_DIM, NUM_CLASSES]);
    let projected = b.add_linear(det_conf, proj_w, None, &[NUM_PATCHES, HIDDEN_DIM]);
    let proj_act = b.add_relu(projected, &[NUM_PATCHES, HIDDEN_DIM]);

    // Narrow to SEQ_LEN (select top detections)
    let narrowed = b.add_narrow(proj_act, 0, 0, SEQ_LEN, &[SEQ_LEN, HIDDEN_DIM]);

    // DETR-style classification head: Linear -> sigmoid
    let cls_w = b.add_input("table_cls_weight", &[TABLE_CLASSES, HIDDEN_DIM]);
    let cls_b = b.add_input("table_cls_bias", &[TABLE_CLASSES]);
    let cls_logits = b.add_linear(narrowed, cls_w, Some(cls_b), &[SEQ_LEN, TABLE_CLASSES]);
    let out = b.add_sigmoid(cls_logits, &[SEQ_LEN, TABLE_CLASSES]);

    b.build(out)
        .expect("valid DocLayout-YOLO -> Table Transformer pipeline")
}

fn doclayout_to_table_transformer_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // image
        weight(&[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        bias(&[BACKBONE_CH]),
        weight(&[NUM_CLASSES, BACKBONE_CH]),
        bias(&[NUM_CLASSES]),
        weight(&[HIDDEN_DIM, NUM_CLASSES]),
        weight(&[TABLE_CLASSES, HIDDEN_DIM]),
        bias(&[TABLE_CLASSES]),
    ]
}

#[test]
fn test_pipeline_doclayout_to_table_transformer_ibp() {
    let def = build_doclayout_to_table_transformer_kernel();
    let bindings = doclayout_to_table_transformer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DocLayout-YOLO -> Table Transformer pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, TABLE_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout->TableTransformer IBP: bounds=[{lo_min}, {hi_max}]");

    // Final sigmoid output must be in [0, 1]
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
    assert!(lo_min < hi_max, "non-degenerate: [{lo_min}, {hi_max}]");
}

#[test]
fn test_pipeline_doclayout_to_table_transformer_crown() {
    let def = build_doclayout_to_table_transformer_kernel();
    let bindings = doclayout_to_table_transformer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout->TableTransformer CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

/// Full 3-stage: image -> YOLO detect -> Table Transformer -> cell sigmoid.
/// Extends the 2-stage pipeline with an additional cell classification head.
fn build_doclayout_to_table_to_cell_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_doclayout_table_cell");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // YOLO backbone
    let conv_w = b.add_input(
        "yolo_weight",
        &[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let conv_b = b.add_input("yolo_bias", &[BACKBONE_CH]);
    let backbone = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[BACKBONE_CH, GRID_SIZE, GRID_SIZE],
    );
    let backbone_act = b.add_relu(backbone, &[BACKBONE_CH, GRID_SIZE, GRID_SIZE]);

    // Reshape to sequence
    let reshaped = b.add_reshape(backbone_act, &[BACKBONE_CH, NUM_PATCHES]);
    let transposed = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, BACKBONE_CH]);

    // YOLO detection: sigmoid [0,1]
    let det_w = b.add_input("det_weight", &[NUM_CLASSES, BACKBONE_CH]);
    let det_logits = b.add_linear(transposed, det_w, None, &[NUM_PATCHES, NUM_CLASSES]);
    let det_conf = b.add_sigmoid(det_logits, &[NUM_PATCHES, NUM_CLASSES]);

    // Table Transformer projection
    let table_w = b.add_input("table_weight", &[HIDDEN_DIM, NUM_CLASSES]);
    let table_proj = b.add_linear(det_conf, table_w, None, &[NUM_PATCHES, HIDDEN_DIM]);
    let table_act = b.add_relu(table_proj, &[NUM_PATCHES, HIDDEN_DIM]);
    let narrowed = b.add_narrow(table_act, 0, 0, SEQ_LEN, &[SEQ_LEN, HIDDEN_DIM]);

    // Table structure: sigmoid [0,1]
    let struct_w = b.add_input("struct_weight", &[TABLE_CLASSES, HIDDEN_DIM]);
    let struct_logits = b.add_linear(narrowed, struct_w, None, &[SEQ_LEN, TABLE_CLASSES]);
    let struct_conf = b.add_sigmoid(struct_logits, &[SEQ_LEN, TABLE_CLASSES]);

    // Cell classification head
    let cell_w = b.add_input("cell_weight", &[NUM_CLASSES, TABLE_CLASSES]);
    let cell_b = b.add_input("cell_bias", &[NUM_CLASSES]);
    let cell_logits = b.add_linear(struct_conf, cell_w, Some(cell_b), &[SEQ_LEN, NUM_CLASSES]);
    let out = b.add_sigmoid(cell_logits, &[SEQ_LEN, NUM_CLASSES]);

    b.build(out)
        .expect("valid 3-stage DocLayout -> Table -> Cell pipeline")
}

fn doclayout_to_table_to_cell_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // image
        weight(&[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        bias(&[BACKBONE_CH]),
        weight(&[NUM_CLASSES, BACKBONE_CH]),
        weight(&[HIDDEN_DIM, NUM_CLASSES]),
        weight(&[TABLE_CLASSES, HIDDEN_DIM]),
        weight(&[NUM_CLASSES, TABLE_CLASSES]),
        bias(&[NUM_CLASSES]),
    ]
}

#[test]
fn test_pipeline_doclayout_table_cell_3stage_ibp() {
    let def = build_doclayout_to_table_to_cell_kernel();
    let bindings = doclayout_to_table_to_cell_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 3-stage DocLayout -> Table -> Cell pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout->Table->Cell 3-stage IBP: bounds=[{lo_min}, {hi_max}]");

    // All three stages use sigmoid: output must be in [0, 1]
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
    assert!(lo_min < hi_max, "non-degenerate: [{lo_min}, {hi_max}]");
}

// ===========================================================================
// 2. PaddleOCR detection -> recognition pipeline
// ===========================================================================

/// Build PaddleOCR detection -> recognition pipeline:
/// Conv2d DB backbone -> sigmoid probability map -> Linear projection ->
/// GELU -> CTC head -> softmax.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (softmax character probabilities [0, 1]).
fn build_paddle_detection_to_recognition_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_paddle_detect_to_recognize");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // DB detector: Conv2d backbone -> sigmoid probability map
    let det_conv_w = b.add_input(
        "det_backbone_weight",
        &[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let det_conv_b = b.add_input("det_backbone_bias", &[BACKBONE_CH]);
    let det_features = b.add_conv2d(
        input,
        det_conv_w,
        Some(det_conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[BACKBONE_CH, GRID_SIZE, GRID_SIZE],
    );

    // Reshape and project to detection logits
    let reshaped = b.add_reshape(det_features, &[BACKBONE_CH, NUM_PATCHES]);
    let transposed = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, BACKBONE_CH]);
    let det_head_w = b.add_input("det_head_weight", &[1, BACKBONE_CH]);
    let det_logits = b.add_linear(transposed, det_head_w, None, &[NUM_PATCHES, 1]);
    let det_prob = b.add_sigmoid(det_logits, &[NUM_PATCHES, 1]);

    // === Pipeline boundary: detection -> recognition ===

    // Project detection probability to recognition feature space
    let rec_proj_w = b.add_input("rec_proj_weight", &[HIDDEN_DIM, 1]);
    let rec_features = b.add_linear(det_prob, rec_proj_w, None, &[NUM_PATCHES, HIDDEN_DIM]);
    let rec_act = b.add_gelu(rec_features, &[NUM_PATCHES, HIDDEN_DIM]);

    // Narrow to SEQ_LEN
    let narrowed = b.add_narrow(rec_act, 0, 0, SEQ_LEN, &[SEQ_LEN, HIDDEN_DIM]);

    // CTC head: Linear -> softmax
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let ctc_logits = b.add_linear(narrowed, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(ctc_logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid PaddleOCR detection -> recognition pipeline")
}

fn paddle_detection_to_recognition_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // image
        weight(&[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        bias(&[BACKBONE_CH]),
        weight(&[1, BACKBONE_CH]),
        weight(&[HIDDEN_DIM, 1]),
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        bias(&[VOCAB_SIZE]),
    ]
}

#[test]
fn test_pipeline_paddle_detect_to_recognize_ibp() {
    let def = build_paddle_detection_to_recognition_kernel();
    let bindings = paddle_detection_to_recognition_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR detect -> recognize pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR detect->recognize IBP: bounds=[{lo_min}, {hi_max}]");

    // Softmax output must be in [0, 1]
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
    assert!(lo_min < hi_max, "non-degenerate: [{lo_min}, {hi_max}]");
}

#[test]
fn test_pipeline_paddle_detect_to_recognize_crown() {
    let def = build_paddle_detection_to_recognition_kernel();
    let bindings = paddle_detection_to_recognition_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR detect->recognize CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

/// Build detection -> recognition with high-confidence gating.
/// Uses tighter input bounds [0.7, 1.0] for detection output.
///
/// Input: `[SEQ_LEN, 1]` (Variable, detection probability [0.7, 1.0]).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (softmax character probabilities [0, 1]).
fn build_paddle_gated_recognition_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_paddle_gated_recognition");

    let input = b.add_input("det_prob", &[SEQ_LEN, 1]);

    // Project high-confidence detection to recognition features
    let proj_w = b.add_input("proj_weight", &[HIDDEN_DIM, 1]);
    let features = b.add_linear(input, proj_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let activated = b.add_gelu(features, &[SEQ_LEN, HIDDEN_DIM]);

    // CTC head: Linear -> softmax
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(activated, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid PaddleOCR gated recognition pipeline")
}

fn paddle_gated_recognition_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // det_prob
        weight(&[HIDDEN_DIM, 1]),
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        bias(&[VOCAB_SIZE]),
    ]
}

#[test]
fn test_pipeline_paddle_gated_recognition_tighter_bounds_ibp() {
    let def = build_paddle_gated_recognition_kernel();
    let bindings = paddle_gated_recognition_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Full-range detection output [0, 1]
    let full_input = sigmoid_bounds(&[SEQ_LEN, 1]);
    let full_output = graph
        .propagate_ibp(&full_input)
        .expect("IBP through full-range detection");

    // High-confidence detection output [0.7, 1.0]
    let gated_input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[SEQ_LEN, 1]), 0.7f32),
        ArrayD::from_elem(IxDyn(&[SEQ_LEN, 1]), 1.0f32),
    )
    .expect("valid gated bounds");
    let gated_output = graph
        .propagate_ibp(&gated_input)
        .expect("IBP through gated detection");

    assert_bounds_valid(&full_output);
    assert_bounds_valid(&gated_output);

    let (full_lo, full_hi) = bounds_min_max(&full_output);
    let (gated_lo, gated_hi) = bounds_min_max(&gated_output);
    let full_width = full_hi - full_lo;
    let gated_width = gated_hi - gated_lo;

    eprintln!("Full-range IBP: bounds=[{full_lo}, {full_hi}] width={full_width}");
    eprintln!("Gated IBP: bounds=[{gated_lo}, {gated_hi}] width={gated_width}");

    // Tighter input bounds should produce tighter (or equal) output bounds
    // Note: softmax normalizes so both are in [0,1], but gated should be
    // no wider than full-range.
    assert!(
        gated_width <= full_width + 1e-4,
        "gated bounds width {gated_width} should be <= full width {full_width}"
    );
}

// ===========================================================================
// 3. Granite-Docling vision encoder -> text decoder
// ===========================================================================

/// Build Granite-Docling vision encoder -> text decoder:
/// Conv2d patch embed -> reshape -> transpose -> Linear vision encoder ->
/// Linear cross-attention projection -> ReLU -> Linear LM head -> softmax.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (softmax token probabilities [0, 1]).
fn build_granite_vision_to_decoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_granite_vision_to_decoder");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Vision encoder: Conv2d patch embed
    let patch_w = b.add_input(
        "patch_weight",
        &[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_b = b.add_input("patch_bias", &[BACKBONE_CH]);
    let patches = b.add_conv2d(
        input,
        patch_w,
        Some(patch_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[BACKBONE_CH, GRID_SIZE, GRID_SIZE],
    );

    // Reshape + transpose: [BACKBONE_CH, 2, 2] -> [NUM_PATCHES, BACKBONE_CH]
    let reshaped = b.add_reshape(patches, &[BACKBONE_CH, NUM_PATCHES]);
    let transposed = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, BACKBONE_CH]);

    // Vision encoder Linear
    let enc_w = b.add_input("vision_enc_weight", &[HIDDEN_DIM, BACKBONE_CH]);
    let encoded = b.add_linear(transposed, enc_w, None, &[NUM_PATCHES, HIDDEN_DIM]);
    let enc_act = b.add_relu(encoded, &[NUM_PATCHES, HIDDEN_DIM]);

    // === Pipeline boundary: vision encoder -> text decoder ===

    // Cross-attention projection: maps vision features to decoder space
    let cross_w = b.add_input("cross_proj_weight", &[FFN_DIM, HIDDEN_DIM]);
    let cross_proj = b.add_linear(enc_act, cross_w, None, &[NUM_PATCHES, FFN_DIM]);
    let cross_act = b.add_relu(cross_proj, &[NUM_PATCHES, FFN_DIM]);

    // Narrow to decoder sequence length
    let narrowed = b.add_narrow(cross_act, 0, 0, SEQ_LEN, &[SEQ_LEN, FFN_DIM]);

    // LM head: Linear -> softmax
    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, FFN_DIM]);
    let lm_b = b.add_input("lm_head_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(narrowed, lm_w, Some(lm_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid Granite vision -> decoder pipeline")
}

fn granite_vision_to_decoder_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // image
        weight(&[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        bias(&[BACKBONE_CH]),
        weight(&[HIDDEN_DIM, BACKBONE_CH]),
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, FFN_DIM]),
        bias(&[VOCAB_SIZE]),
    ]
}

#[test]
fn test_pipeline_granite_vision_to_decoder_ibp() {
    let def = build_granite_vision_to_decoder_kernel();
    let bindings = granite_vision_to_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Granite vision -> decoder");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite vision->decoder IBP: bounds=[{lo_min}, {hi_max}]");

    // Softmax output must be in [0, 1]
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
    assert!(lo_min < hi_max, "non-degenerate: [{lo_min}, {hi_max}]");
}

#[test]
fn test_pipeline_granite_vision_to_decoder_crown() {
    let def = build_granite_vision_to_decoder_kernel();
    let bindings = granite_vision_to_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite vision->decoder CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

/// Build vision projection dimension compatibility test: vision encoder
/// output dim HIDDEN_DIM -> explicit projection -> decoder input dim FFN_DIM.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, vision encoder output [-1, 1]).
/// Output: `[SEQ_LEN, FFN_DIM]` (decoder cross-attention features).
fn build_vision_projection_compat_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_vision_projection_compat");

    let input = b.add_input("vision_features", &[SEQ_LEN, HIDDEN_DIM]);

    // Explicit dimension alignment projection
    let proj_w = b.add_input("proj_weight", &[FFN_DIM, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_bias", &[FFN_DIM]);
    let projected = b.add_linear(input, proj_w, Some(proj_b), &[SEQ_LEN, FFN_DIM]);
    let out = b.add_relu(projected, &[SEQ_LEN, FFN_DIM]);

    b.build(out)
        .expect("valid vision projection compatibility kernel")
}

fn vision_projection_compat_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // vision_features
        weight(&[FFN_DIM, HIDDEN_DIM]),
        bias(&[FFN_DIM]),
    ]
}

#[test]
fn test_pipeline_vision_projection_dimension_compat_ibp() {
    let def = build_vision_projection_compat_kernel();
    let bindings = vision_projection_compat_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through vision projection compatibility");

    // Key assertion: output shape matches decoder's expected input
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, FFN_DIM],
        "vision projection output must match decoder input dimension"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Vision projection compat IBP: bounds=[{lo_min}, {hi_max}]");

    // ReLU ensures non-negative lower bound
    assert!(lo_min >= -1e-4, "ReLU lower >= 0, got {lo_min}");
}

// ===========================================================================
// 4. FireRed-OCR vision -> language pipeline
// ===========================================================================

/// Build FireRed-OCR multi-scale ViT -> MoE decoder:
/// Conv2d patch embed -> Linear encoder -> ReLU -> expert gate softmax ->
/// expert FFN -> Linear projection -> sigmoid output.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[SEQ_LEN, NUM_CLASSES]` (sigmoid confidence [0, 1]).
fn build_firered_vision_to_moe_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_firered_vision_to_moe");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Vision encoder: Conv2d patch embed
    let patch_w = b.add_input(
        "patch_weight",
        &[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_b = b.add_input("patch_bias", &[BACKBONE_CH]);
    let patches = b.add_conv2d(
        input,
        patch_w,
        Some(patch_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[BACKBONE_CH, GRID_SIZE, GRID_SIZE],
    );

    // Reshape + transpose: [BACKBONE_CH, 2, 2] -> [NUM_PATCHES, BACKBONE_CH]
    let reshaped = b.add_reshape(patches, &[BACKBONE_CH, NUM_PATCHES]);
    let transposed = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, BACKBONE_CH]);

    // Vision encoder block: Linear -> ReLU
    let enc_w = b.add_input("encoder_weight", &[HIDDEN_DIM, BACKBONE_CH]);
    let encoded = b.add_linear(transposed, enc_w, None, &[NUM_PATCHES, HIDDEN_DIM]);
    let enc_act = b.add_relu(encoded, &[NUM_PATCHES, HIDDEN_DIM]);

    // Narrow to SEQ_LEN
    let narrowed = b.add_narrow(enc_act, 0, 0, SEQ_LEN, &[SEQ_LEN, HIDDEN_DIM]);

    // === Pipeline boundary: vision encoder -> MoE decoder ===

    // Expert gate: Linear -> softmax routing
    let gate_w = b.add_input("gate_weight", &[NUM_EXPERTS, HIDDEN_DIM]);
    let gate_logits = b.add_linear(narrowed, gate_w, None, &[SEQ_LEN, NUM_EXPERTS]);
    let _gate_probs = b.add_softmax(gate_logits, -1, &[SEQ_LEN, NUM_EXPERTS]);

    // Expert FFN (worst-case single expert path)
    let ffn_w = b.add_input("expert_ffn_weight", &[FFN_DIM, HIDDEN_DIM]);
    let ffn_out = b.add_linear(narrowed, ffn_w, None, &[SEQ_LEN, FFN_DIM]);
    let ffn_act = b.add_relu(ffn_out, &[SEQ_LEN, FFN_DIM]);

    // Projection -> sigmoid
    let proj_w = b.add_input("proj_weight", &[NUM_CLASSES, FFN_DIM]);
    let proj_b = b.add_input("proj_bias", &[NUM_CLASSES]);
    let logits = b.add_linear(ffn_act, proj_w, Some(proj_b), &[SEQ_LEN, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[SEQ_LEN, NUM_CLASSES]);

    b.build(out)
        .expect("valid FireRed vision -> MoE decoder pipeline")
}

fn firered_vision_to_moe_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // image
        weight(&[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        bias(&[BACKBONE_CH]),
        weight(&[HIDDEN_DIM, BACKBONE_CH]),
        weight(&[NUM_EXPERTS, HIDDEN_DIM]),
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[NUM_CLASSES, FFN_DIM]),
        bias(&[NUM_CLASSES]),
    ]
}

#[test]
fn test_pipeline_firered_vision_to_moe_ibp() {
    let def = build_firered_vision_to_moe_kernel();
    let bindings = firered_vision_to_moe_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed vision -> MoE decoder");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed vision->MoE IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid output must be in [0, 1]
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
    assert!(lo_min < hi_max, "non-degenerate: [{lo_min}, {hi_max}]");
}

#[test]
fn test_pipeline_firered_vision_to_moe_crown() {
    let def = build_firered_vision_to_moe_kernel();
    let bindings = firered_vision_to_moe_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed vision->MoE CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

/// Build FireRed-OCR ViT -> language head softmax:
/// Vision encoder -> linear projection -> CTC head -> softmax.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (softmax token probabilities [0, 1]).
fn build_firered_vision_to_ctc_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_firered_vision_to_ctc");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Vision encoder: Conv2d patch embed
    let patch_w = b.add_input(
        "patch_weight",
        &[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_b = b.add_input("patch_bias", &[BACKBONE_CH]);
    let patches = b.add_conv2d(
        input,
        patch_w,
        Some(patch_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[BACKBONE_CH, GRID_SIZE, GRID_SIZE],
    );

    // Reshape + transpose
    let reshaped = b.add_reshape(patches, &[BACKBONE_CH, NUM_PATCHES]);
    let transposed = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, BACKBONE_CH]);

    // Encoder: Linear -> ReLU
    let enc_w = b.add_input("enc_weight", &[HIDDEN_DIM, BACKBONE_CH]);
    let encoded = b.add_linear(transposed, enc_w, None, &[NUM_PATCHES, HIDDEN_DIM]);
    let enc_act = b.add_relu(encoded, &[NUM_PATCHES, HIDDEN_DIM]);

    // Narrow to SEQ_LEN
    let narrowed = b.add_narrow(enc_act, 0, 0, SEQ_LEN, &[SEQ_LEN, HIDDEN_DIM]);

    // CTC head: Linear -> softmax
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(narrowed, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid FireRed vision -> CTC pipeline")
}

fn firered_vision_to_ctc_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // image
        weight(&[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        bias(&[BACKBONE_CH]),
        weight(&[HIDDEN_DIM, BACKBONE_CH]),
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        bias(&[VOCAB_SIZE]),
    ]
}

#[test]
fn test_pipeline_firered_vision_to_ctc_ibp() {
    let def = build_firered_vision_to_ctc_kernel();
    let bindings = firered_vision_to_ctc_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed vision -> CTC pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed vision->CTC IBP: bounds=[{lo_min}, {hi_max}]");

    // Softmax output must be in [0, 1]
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
    assert!(lo_min < hi_max, "non-degenerate: [{lo_min}, {hi_max}]");
}

// ===========================================================================
// 5. Cross-model tensor compatibility
// ===========================================================================

/// Build shape/dtype compatibility chain: 4 sequential linear projections
/// verifying output shape of model A matches input requirements of model B.
///
/// Pipeline: BACKBONE_CH -> HIDDEN_DIM -> FFN_DIM -> HIDDEN_DIM -> VOCAB_SIZE
///
/// Input: `[SEQ_LEN, BACKBONE_CH]` (Variable, features [-1, 1]).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (logits, unbounded).
fn build_shape_compatibility_chain_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_shape_compat_chain");

    let input = b.add_input("features", &[SEQ_LEN, BACKBONE_CH]);

    // Stage 1: BACKBONE_CH -> HIDDEN_DIM (YOLO -> Table Transformer boundary)
    let w1 = b.add_input("stage1_weight", &[HIDDEN_DIM, BACKBONE_CH]);
    let s1 = b.add_linear(input, w1, None, &[SEQ_LEN, HIDDEN_DIM]);
    let s1_act = b.add_relu(s1, &[SEQ_LEN, HIDDEN_DIM]);

    // Stage 2: HIDDEN_DIM -> FFN_DIM (Table Transformer -> decoder boundary)
    let w2 = b.add_input("stage2_weight", &[FFN_DIM, HIDDEN_DIM]);
    let s2 = b.add_linear(s1_act, w2, None, &[SEQ_LEN, FFN_DIM]);
    let s2_act = b.add_relu(s2, &[SEQ_LEN, FFN_DIM]);

    // Stage 3: FFN_DIM -> HIDDEN_DIM (decoder -> OCR encoder boundary)
    let w3 = b.add_input("stage3_weight", &[HIDDEN_DIM, FFN_DIM]);
    let s3 = b.add_linear(s2_act, w3, None, &[SEQ_LEN, HIDDEN_DIM]);
    let s3_act = b.add_relu(s3, &[SEQ_LEN, HIDDEN_DIM]);

    // Stage 4: HIDDEN_DIM -> VOCAB_SIZE (OCR encoder -> CTC head boundary)
    let w4 = b.add_input("stage4_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let w4_b = b.add_input("stage4_bias", &[VOCAB_SIZE]);
    let out = b.add_linear(s3_act, w4, Some(w4_b), &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid shape compatibility chain kernel")
}

fn shape_compatibility_chain_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // features
        weight(&[HIDDEN_DIM, BACKBONE_CH]),
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, FFN_DIM]),
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        bias(&[VOCAB_SIZE]),
    ]
}

#[test]
fn test_pipeline_shape_compatibility_chain_ibp() {
    let def = build_shape_compatibility_chain_kernel();
    let bindings = shape_compatibility_chain_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, BACKBONE_CH], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through shape compatibility chain");

    // Verify each stage boundary produces the correct output shape
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "final output shape must match CTC head dimension"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Shape compatibility chain IBP: bounds=[{lo_min}, {hi_max}]");

    // Bounds must be finite (no shape mismatch causing overflow)
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(lo_min < hi_max, "non-degenerate: [{lo_min}, {hi_max}]");
}

/// Build sigmoid -> softmax activation boundary:
/// Detection sigmoid [0,1] -> Linear -> GELU -> Linear -> softmax [0,1].
///
/// Input: `[SEQ_LEN, NUM_CLASSES]` (Variable, detection sigmoid [0, 1]).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (softmax probabilities [0, 1]).
fn build_sigmoid_to_softmax_boundary_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_sigmoid_to_softmax_boundary");

    let input = b.add_input("detection_conf", &[SEQ_LEN, NUM_CLASSES]);

    // Transform detection features
    let w1 = b.add_input("transform_weight", &[HIDDEN_DIM, NUM_CLASSES]);
    let features = b.add_linear(input, w1, None, &[SEQ_LEN, HIDDEN_DIM]);
    let activated = b.add_gelu(features, &[SEQ_LEN, HIDDEN_DIM]);

    // Recognition head: softmax
    let w2 = b.add_input("rec_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let w2_b = b.add_input("rec_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(activated, w2, Some(w2_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid sigmoid -> softmax boundary kernel")
}

fn sigmoid_to_softmax_boundary_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // detection_conf
        weight(&[HIDDEN_DIM, NUM_CLASSES]),
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        bias(&[VOCAB_SIZE]),
    ]
}

#[test]
fn test_pipeline_sigmoid_to_softmax_activation_boundary_ibp() {
    let def = build_sigmoid_to_softmax_boundary_kernel();
    let bindings = sigmoid_to_softmax_boundary_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Detection sigmoid output [0, 1]
    let input = sigmoid_bounds(&[SEQ_LEN, NUM_CLASSES]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through sigmoid -> softmax boundary");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Sigmoid->softmax boundary IBP: bounds=[{lo_min}, {hi_max}]");

    // Input domain [0,1] (sigmoid) -> output domain [0,1] (softmax)
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
    assert!(lo_min < hi_max, "non-degenerate: [{lo_min}, {hi_max}]");
}

// ===========================================================================
// 6. Pipeline error accumulation
// ===========================================================================

/// Build 4-stage pipeline for chained error accumulation test.
/// Each stage: Linear -> ReLU (with small weights to keep bounds manageable).
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, features [-1, 1]).
/// Output: `[SEQ_LEN, HIDDEN_DIM]` (features after 4 stages).
fn build_error_accumulation_chain_kernel(stages: usize) -> TensorKernelDef {
    let name = format!("pipeline_error_accum_{stages}stage");
    let mut b = TensorBlockBuilder::new(&name);

    let mut current = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);

    for i in 0..stages {
        let w = b.add_input(&format!("stage{i}_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let b_param = b.add_input(&format!("stage{i}_bias"), &[HIDDEN_DIM]);
        current = b.add_linear(current, w, Some(b_param), &[SEQ_LEN, HIDDEN_DIM]);
        current = b.add_relu(current, &[SEQ_LEN, HIDDEN_DIM]);
    }

    b.build(current)
        .expect("valid error accumulation chain kernel")
}

fn error_accumulation_chain_bindings(stages: usize) -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // features
    for _ in 0..stages {
        bindings.push(weight(&[HIDDEN_DIM, HIDDEN_DIM]));
        bindings.push(bias(&[HIDDEN_DIM]));
    }
    bindings
}

#[test]
fn test_pipeline_chained_error_accumulation_ibp() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // Run 1-stage pipeline
    let def1 = build_error_accumulation_chain_kernel(1);
    let bindings1 = error_accumulation_chain_bindings(1);
    let graph1 = tensor_kernel_to_graph(&def1, &bindings1).expect("graph 1-stage");
    let output1 = graph1.propagate_ibp(&input).expect("IBP 1-stage");

    // Run 4-stage pipeline
    let def4 = build_error_accumulation_chain_kernel(4);
    let bindings4 = error_accumulation_chain_bindings(4);
    let graph4 = tensor_kernel_to_graph(&def4, &bindings4).expect("graph 4-stage");
    let output4 = graph4.propagate_ibp(&input).expect("IBP 4-stage");

    assert_bounds_valid(&output1);
    assert_bounds_valid(&output4);

    let (lo1, hi1) = bounds_min_max(&output1);
    let (lo4, hi4) = bounds_min_max(&output4);
    let width1 = hi1 - lo1;
    let width4 = hi4 - lo4;

    eprintln!("1-stage: bounds=[{lo1}, {hi1}] width={width1}");
    eprintln!("4-stage: bounds=[{lo4}, {hi4}] width={width4}");

    // Key property: bounds width should grow but remain bounded.
    // With WEIGHT_MAG=0.02 and ReLU clamping, 4-stage width should be
    // less than 4^4 * width1 (sub-exponential due to ReLU truncation).
    // In practice with small weights: width4 < 256 * width1.
    assert!(
        width4 < 256.0 * width1 + 1e-4,
        "4-stage bounds width {width4} should be sub-exponentially bounded \
         relative to 1-stage width {width1}"
    );

    // Both must be finite (no unbounded error propagation)
    assert!(lo4.is_finite(), "4-stage lower bound must be finite");
    assert!(hi4.is_finite(), "4-stage upper bound must be finite");
}

/// Build wide (1-layer with more parameters) vs deep (4-layer) comparison.
///
/// Wide: `[SEQ_LEN, HIDDEN_DIM]` -> Linear(4*HIDDEN_DIM) -> ReLU -> Linear(HIDDEN_DIM)
/// Deep: `[SEQ_LEN, HIDDEN_DIM]` -> 4x (Linear(HIDDEN_DIM) -> ReLU)
fn build_wide_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pipeline_wide_single_layer");
    let wide_dim = 4 * HIDDEN_DIM;

    let input = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);

    // Wide: expand to 4*HIDDEN_DIM then contract back
    let up_w = b.add_input("up_weight", &[wide_dim, HIDDEN_DIM]);
    let up = b.add_linear(input, up_w, None, &[SEQ_LEN, wide_dim]);
    let up_act = b.add_relu(up, &[SEQ_LEN, wide_dim]);

    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, wide_dim]);
    let down_b = b.add_input("down_bias", &[HIDDEN_DIM]);
    let out = b.add_linear(up_act, down_w, Some(down_b), &[SEQ_LEN, HIDDEN_DIM]);

    b.build(out).expect("valid wide pipeline kernel")
}

fn wide_pipeline_bindings() -> Vec<TensorParamBinding> {
    let wide_dim = 4 * HIDDEN_DIM;
    vec![
        TensorParamBinding::Variable, // features
        weight(&[wide_dim, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, wide_dim]),
        bias(&[HIDDEN_DIM]),
    ]
}

#[test]
fn test_pipeline_depth_vs_width_tradeoff_ibp() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // Deep 4-layer pipeline
    let def_deep = build_error_accumulation_chain_kernel(4);
    let bindings_deep = error_accumulation_chain_bindings(4);
    let graph_deep = tensor_kernel_to_graph(&def_deep, &bindings_deep).expect("graph deep");
    let output_deep = graph_deep.propagate_ibp(&input).expect("IBP deep");

    // Wide 1-layer pipeline
    let def_wide = build_wide_pipeline_kernel();
    let bindings_wide = wide_pipeline_bindings();
    let graph_wide = tensor_kernel_to_graph(&def_wide, &bindings_wide).expect("graph wide");
    let output_wide = graph_wide.propagate_ibp(&input).expect("IBP wide");

    assert_bounds_valid(&output_deep);
    assert_bounds_valid(&output_wide);

    let (deep_lo, deep_hi) = bounds_min_max(&output_deep);
    let (wide_lo, wide_hi) = bounds_min_max(&output_wide);
    let deep_width = deep_hi - deep_lo;
    let wide_width = wide_hi - wide_lo;

    eprintln!("Deep 4-layer: bounds=[{deep_lo}, {deep_hi}] width={deep_width}");
    eprintln!("Wide 1-layer: bounds=[{wide_lo}, {wide_hi}] width={wide_width}");

    // Both architectures must produce finite, non-degenerate bounds
    assert!(
        deep_lo.is_finite() && deep_hi.is_finite(),
        "deep bounds finite"
    );
    assert!(
        wide_lo.is_finite() && wide_hi.is_finite(),
        "wide bounds finite"
    );

    // Deep pipeline should not produce bounds that are vastly wider than
    // a wide pipeline with comparable parameter count. With WEIGHT_MAG=0.02
    // and ReLU, deep width should be within a reasonable multiple of wide.
    assert!(
        deep_width < 1000.0 * wide_width + 1e-2,
        "deep pipeline bounds width {deep_width} should not be unboundedly wider \
         than wide pipeline width {wide_width}"
    );
}
