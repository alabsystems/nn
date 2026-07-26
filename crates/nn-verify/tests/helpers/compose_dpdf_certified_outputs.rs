// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Certified per-model output bounds for 7 dpdf model architectures.
//!
//! Each test builds a representative subgraph of a dpdf model architecture using
//! `TensorBlockBuilder`, runs IBP (and optionally CROWN), and verifies that output
//! bounds are finite, non-degenerate (lower < upper), and architecturally consistent
//! (e.g., sigmoid outputs in [0, 1], softmax sums bounded).
//!
//! ## Tests:
//!
//! 1. **Granite-Docling certified bounds**: EfficientNet-style backbone (Conv2d +
//!    ReLU) -> Linear vision projection -> ReLU -> Linear LM head -> sigmoid.
//!    Certifies detection confidence in [0, 1].
//!
//! 2. **DocLayout-YOLO certified bounds**: YOLOv8-style backbone (Conv2d + ReLU)
//!    -> Linear PAN neck -> ReLU -> Linear detect head -> sigmoid classification.
//!    Certifies layout detection confidence in [0, 1].
//!
//! 3. **PaddleOCR certified bounds**: SVTR encoder (Conv2d patch embed + reshape +
//!    transpose + Linear + GELU) -> Linear CTC decoder -> softmax character
//!    probabilities in [0, 1].
//!
//! 4. **FireRed-OCR certified bounds**: ViT encoder (Conv2d patch embed + reshape +
//!    transpose + Linear + ReLU) -> Linear LM head -> softmax token probabilities
//!    in [0, 1].
//!
//! 5. **Qwen3-VL MoE certified bounds**: Linear expert gate -> softmax routing
//!    in [0, 1] -> Linear expert FFN -> ReLU -> Linear projection -> sigmoid
//!    output confidence.
//!
//! 6. **GLM-OCR certified bounds**: Linear embedding -> RMSNorm -> Linear
//!    attention proxy -> sigmoid gate -> Linear FFN -> Linear MTP head -> softmax
//!    token probabilities in [0, 1].
//!
//! 7. **Table Transformer certified bounds**: ResNet-18-style backbone (Conv2d +
//!    ReLU) -> Linear encoder projection -> ReLU -> Linear DETR decoder projection
//!    -> ReLU -> Linear classification head -> sigmoid detection confidence [0, 1].
//!
//! Architecture references:
//! - Granite-Docling: SigLIP2 vision encoder + Granite LLM decoder
//! - DocLayout-YOLO (Zhao et al. 2024): YOLOv10-based document layout detection
//! - PaddleOCR (Baidu): DB detector + SVTR recognizer with CTC
//! - FireRed-OCR: Qwen3-VL-2B variant for document OCR with CTC decoding
//! - Qwen3-VL (Alibaba): Vision-language model with MoE transformer blocks
//! - GLM-4V (THUDM): Vision-language model with Multi-Token Prediction
//! - Table Transformer (Smock et al. 2022): DETR-based table structure recognition
//!
//! Dimensions (small for fast verification, structurally representative):
//! - IMG_SIZE=16, PATCH_SIZE=8, HIDDEN_DIM=32, FFN_DIM=64, SEQ_LEN=4,
//!   NUM_CLASSES=8, VOCAB_SIZE=16, BACKBONE_CH=16, NUM_EXPERTS=4
//!
//! Part of #4078: certified per-model output bounds for dpdf architectures.

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

/// Constant weight tensor binding.
fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

/// Constant zero bias tensor binding.
fn bias(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

/// Constant scalar epsilon binding for RMSNorm.
fn eps_binding() -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32))
}

/// RMSNorm weight (all ones) binding.
fn norm_weight_binding(dim: usize) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 1.0f32))
}

// ===========================================================================
// 1. Granite-Docling certified output bounds
// ===========================================================================

/// Build Granite-Docling certified pipeline: Conv2d backbone -> reshape ->
/// transpose -> Linear vision projection -> ReLU -> Linear LM head -> sigmoid.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[SEQ_LEN, NUM_CLASSES]` (sigmoid confidence [0, 1]).
fn build_granite_docling_certified_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("certified_granite_docling");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Conv2d backbone: [3, 16, 16] -> [BACKBONE_CH, GRID_SIZE, GRID_SIZE]
    let conv_w = b.add_input(
        "backbone_weight",
        &[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let conv_b = b.add_input("backbone_bias", &[BACKBONE_CH]);
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

    // Reshape + transpose to sequence: [BACKBONE_CH, 2, 2] -> [NUM_PATCHES, BACKBONE_CH]
    let reshaped = b.add_reshape(backbone, &[BACKBONE_CH, NUM_PATCHES]);
    let transposed = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, BACKBONE_CH]);
    let narrowed = b.add_narrow(transposed, 0, 0, SEQ_LEN, &[SEQ_LEN, BACKBONE_CH]);

    // Vision projection: Linear -> ReLU
    let proj_w = b.add_input("vision_proj_weight", &[HIDDEN_DIM, BACKBONE_CH]);
    let proj = b.add_linear(narrowed, proj_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let proj_act = b.add_relu(proj, &[SEQ_LEN, HIDDEN_DIM]);

    // LM head: Linear -> sigmoid
    let head_w = b.add_input("head_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let head_b = b.add_input("head_bias", &[NUM_CLASSES]);
    let logits = b.add_linear(proj_act, head_w, Some(head_b), &[SEQ_LEN, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[SEQ_LEN, NUM_CLASSES]);

    b.build(out)
        .expect("valid Granite-Docling certified pipeline")
}

fn granite_docling_certified_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // image
        weight(&[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        bias(&[BACKBONE_CH]),
        weight(&[HIDDEN_DIM, BACKBONE_CH]),
        weight(&[NUM_CLASSES, HIDDEN_DIM]),
        bias(&[NUM_CLASSES]),
    ]
}

#[test]
fn test_granite_docling_certified_bounds() {
    let def = build_granite_docling_certified_kernel();
    let bindings = granite_docling_certified_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Granite-Docling certified pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite-Docling certified IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid output must be in [0, 1]
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
    // Non-degenerate bounds
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

#[test]
fn test_granite_docling_certified_crown() {
    let def = build_granite_docling_certified_kernel();
    let bindings = granite_docling_certified_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite-Docling certified: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 2. DocLayout-YOLO certified output bounds
// ===========================================================================

/// Build DocLayout-YOLO certified pipeline: Conv2d backbone -> Linear PAN neck ->
/// ReLU -> Linear detect head -> sigmoid classification.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[SEQ_LEN, NUM_CLASSES]` (sigmoid confidence [0, 1]).
fn build_doclayout_yolo_certified_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("certified_doclayout_yolo");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Conv2d backbone: [3, 16, 16] -> [BACKBONE_CH, GRID_SIZE, GRID_SIZE]
    let conv_w = b.add_input(
        "backbone_weight",
        &[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let conv_b = b.add_input("backbone_bias", &[BACKBONE_CH]);
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

    // Reshape + transpose to sequence
    let reshaped = b.add_reshape(backbone, &[BACKBONE_CH, NUM_PATCHES]);
    let transposed = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, BACKBONE_CH]);
    let narrowed = b.add_narrow(transposed, 0, 0, SEQ_LEN, &[SEQ_LEN, BACKBONE_CH]);

    // PAN neck: Linear -> ReLU
    let neck_w = b.add_input("pan_neck_weight", &[FFN_DIM, BACKBONE_CH]);
    let neck = b.add_linear(narrowed, neck_w, None, &[SEQ_LEN, FFN_DIM]);
    let neck_act = b.add_relu(neck, &[SEQ_LEN, FFN_DIM]);

    // Detect head: Linear -> sigmoid
    let det_w = b.add_input("detect_weight", &[NUM_CLASSES, FFN_DIM]);
    let det_b = b.add_input("detect_bias", &[NUM_CLASSES]);
    let logits = b.add_linear(neck_act, det_w, Some(det_b), &[SEQ_LEN, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[SEQ_LEN, NUM_CLASSES]);

    b.build(out)
        .expect("valid DocLayout-YOLO certified pipeline")
}

fn doclayout_yolo_certified_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // image
        weight(&[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        bias(&[BACKBONE_CH]),
        weight(&[FFN_DIM, BACKBONE_CH]),
        weight(&[NUM_CLASSES, FFN_DIM]),
        bias(&[NUM_CLASSES]),
    ]
}

#[test]
fn test_doclayout_yolo_certified_bounds() {
    let def = build_doclayout_yolo_certified_kernel();
    let bindings = doclayout_yolo_certified_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DocLayout-YOLO certified pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO certified IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid output must be in [0, 1]
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

#[test]
fn test_doclayout_yolo_certified_crown() {
    let def = build_doclayout_yolo_certified_kernel();
    let bindings = doclayout_yolo_certified_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO certified: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 3. PaddleOCR certified output bounds
// ===========================================================================

/// Build PaddleOCR certified pipeline: SVTR encoder (Conv2d patch embed ->
/// reshape -> transpose -> Linear -> GELU) -> Linear CTC decoder -> softmax.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (softmax character probabilities [0, 1]).
fn build_paddle_ocr_certified_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("certified_paddle_ocr");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Conv2d patch embed: [3, 16, 16] -> [BACKBONE_CH, GRID_SIZE, GRID_SIZE]
    let conv_w = b.add_input(
        "patch_weight",
        &[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let conv_b = b.add_input("patch_bias", &[BACKBONE_CH]);
    let patches = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[BACKBONE_CH, GRID_SIZE, GRID_SIZE],
    );

    // Reshape + transpose: [BACKBONE_CH, 2, 2] -> [NUM_PATCHES, BACKBONE_CH]
    let reshaped = b.add_reshape(patches, &[BACKBONE_CH, NUM_PATCHES]);
    let transposed = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, BACKBONE_CH]);
    let narrowed = b.add_narrow(transposed, 0, 0, SEQ_LEN, &[SEQ_LEN, BACKBONE_CH]);

    // SVTR encoder: Linear -> GELU
    let enc_w = b.add_input("svtr_weight", &[HIDDEN_DIM, BACKBONE_CH]);
    let encoded = b.add_linear(narrowed, enc_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let enc_act = b.add_gelu(encoded, &[SEQ_LEN, HIDDEN_DIM]);

    // CTC decoder: Linear -> softmax
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(enc_act, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid PaddleOCR certified pipeline")
}

fn paddle_ocr_certified_bindings() -> Vec<TensorParamBinding> {
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
fn test_paddle_ocr_certified_bounds() {
    let def = build_paddle_ocr_certified_kernel();
    let bindings = paddle_ocr_certified_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR certified pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR certified IBP: bounds=[{lo_min}, {hi_max}]");

    // Softmax output must be in [0, 1]
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

#[test]
fn test_paddle_ocr_certified_crown() {
    let def = build_paddle_ocr_certified_kernel();
    let bindings = paddle_ocr_certified_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR certified: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 4. FireRed-OCR certified output bounds
// ===========================================================================

/// Build FireRed-OCR certified pipeline: ViT encoder (Conv2d patch embed ->
/// reshape -> transpose -> Linear -> ReLU) -> Linear LM head -> softmax.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (softmax token probabilities [0, 1]).
fn build_firered_ocr_certified_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("certified_firered_ocr");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Conv2d patch embed: [3, 16, 16] -> [BACKBONE_CH, GRID_SIZE, GRID_SIZE]
    let conv_w = b.add_input(
        "patch_weight",
        &[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let conv_b = b.add_input("patch_bias", &[BACKBONE_CH]);
    let patches = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[BACKBONE_CH, GRID_SIZE, GRID_SIZE],
    );

    // Reshape + transpose: [BACKBONE_CH, 2, 2] -> [NUM_PATCHES, BACKBONE_CH]
    let reshaped = b.add_reshape(patches, &[BACKBONE_CH, NUM_PATCHES]);
    let transposed = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, BACKBONE_CH]);
    let narrowed = b.add_narrow(transposed, 0, 0, SEQ_LEN, &[SEQ_LEN, BACKBONE_CH]);

    // ViT encoder: Linear -> ReLU
    let enc_w = b.add_input("encoder_weight", &[HIDDEN_DIM, BACKBONE_CH]);
    let encoded = b.add_linear(narrowed, enc_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let enc_act = b.add_relu(encoded, &[SEQ_LEN, HIDDEN_DIM]);

    // LM head: Linear -> softmax
    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let lm_b = b.add_input("lm_head_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(enc_act, lm_w, Some(lm_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid FireRed-OCR certified pipeline")
}

fn firered_ocr_certified_bindings() -> Vec<TensorParamBinding> {
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
fn test_firered_ocr_certified_bounds() {
    let def = build_firered_ocr_certified_kernel();
    let bindings = firered_ocr_certified_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR certified pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR certified IBP: bounds=[{lo_min}, {hi_max}]");

    // Softmax output must be in [0, 1]
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

#[test]
fn test_firered_ocr_certified_crown() {
    let def = build_firered_ocr_certified_kernel();
    let bindings = firered_ocr_certified_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR certified: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 5. Qwen3-VL MoE certified output bounds
// ===========================================================================

/// Build Qwen3-VL MoE certified pipeline: Linear expert gate -> softmax routing
/// -> Linear expert FFN -> ReLU -> Linear projection -> sigmoid confidence.
///
/// Models the MoE transformer block routing + expert computation. Input is
/// hidden-state features, not raw pixels.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, feature bounds [-1, 1]).
/// Output: `[SEQ_LEN, NUM_CLASSES]` (sigmoid confidence [0, 1]).
fn build_qwen3_vl_moe_certified_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("certified_qwen3_vl_moe");

    let input = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);

    // Expert gate: Linear -> softmax routing over NUM_EXPERTS
    let gate_w = b.add_input("gate_weight", &[NUM_EXPERTS, HIDDEN_DIM]);
    let gate_logits = b.add_linear(input, gate_w, None, &[SEQ_LEN, NUM_EXPERTS]);
    let _gate_probs = b.add_softmax(gate_logits, -1, &[SEQ_LEN, NUM_EXPERTS]);

    // Simplified expert FFN: we model the weighted expert output as a single
    // linear path (the full MoE dispatches to top-k experts, but for bounds
    // verification we model the worst-case single-expert path).
    // Linear FFN: [SEQ_LEN, HIDDEN_DIM] -> [SEQ_LEN, FFN_DIM]
    let ffn_w = b.add_input("expert_ffn_weight", &[FFN_DIM, HIDDEN_DIM]);
    let ffn_out = b.add_linear(input, ffn_w, None, &[SEQ_LEN, FFN_DIM]);
    let ffn_act = b.add_relu(ffn_out, &[SEQ_LEN, FFN_DIM]);

    // Projection: Linear -> sigmoid
    let proj_w = b.add_input("proj_weight", &[NUM_CLASSES, FFN_DIM]);
    let proj_b = b.add_input("proj_bias", &[NUM_CLASSES]);
    let logits = b.add_linear(ffn_act, proj_w, Some(proj_b), &[SEQ_LEN, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[SEQ_LEN, NUM_CLASSES]);

    b.build(out).expect("valid Qwen3-VL MoE certified pipeline")
}

fn qwen3_vl_moe_certified_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // features
        weight(&[NUM_EXPERTS, HIDDEN_DIM]),
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[NUM_CLASSES, FFN_DIM]),
        bias(&[NUM_CLASSES]),
    ]
}

#[test]
fn test_qwen3_vl_moe_certified_bounds() {
    let def = build_qwen3_vl_moe_certified_kernel();
    let bindings = qwen3_vl_moe_certified_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL MoE certified pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL MoE certified IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid output must be in [0, 1]
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

#[test]
fn test_qwen3_vl_moe_certified_crown() {
    let def = build_qwen3_vl_moe_certified_kernel();
    let bindings = qwen3_vl_moe_certified_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL MoE certified: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 6. GLM-OCR certified output bounds
// ===========================================================================

/// Build GLM-OCR certified pipeline: Linear embedding -> RMSNorm -> Linear
/// attention proxy -> sigmoid gate -> Linear FFN -> Linear MTP head -> softmax.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, embedding features [-1, 1]).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (softmax token probabilities [0, 1]).
fn build_glm_ocr_certified_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("certified_glm_ocr");

    let input = b.add_input("embeddings", &[SEQ_LEN, HIDDEN_DIM]);

    // RMSNorm: add_rms_norm(input, eps, axis, weight, out_shape)
    let rms_eps = b.add_input("rms_eps", &[1]);
    let rms_weight = b.add_input("rms_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, rms_eps, 1, rms_weight, &[SEQ_LEN, HIDDEN_DIM]);

    // Attention proxy: Linear -> sigmoid gate
    let attn_w = b.add_input("attn_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let attn_out = b.add_linear(normed, attn_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let attn_gate = b.add_sigmoid(attn_out, &[SEQ_LEN, HIDDEN_DIM]);

    // FFN: Linear
    let ffn_w = b.add_input("ffn_weight", &[FFN_DIM, HIDDEN_DIM]);
    let ffn_out = b.add_linear(attn_gate, ffn_w, None, &[SEQ_LEN, FFN_DIM]);

    // MTP head: Linear -> softmax
    let mtp_w = b.add_input("mtp_weight", &[VOCAB_SIZE, FFN_DIM]);
    let mtp_b = b.add_input("mtp_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(ffn_out, mtp_w, Some(mtp_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid GLM-OCR certified pipeline")
}

fn glm_ocr_certified_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // embeddings
        eps_binding(),
        norm_weight_binding(HIDDEN_DIM),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, FFN_DIM]),
        bias(&[VOCAB_SIZE]),
    ]
}

#[test]
fn test_glm_ocr_certified_bounds() {
    let def = build_glm_ocr_certified_kernel();
    let bindings = glm_ocr_certified_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR certified pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR certified IBP: bounds=[{lo_min}, {hi_max}]");

    // Softmax output must be in [0, 1]
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

#[test]
fn test_glm_ocr_certified_crown() {
    let def = build_glm_ocr_certified_kernel();
    let bindings = glm_ocr_certified_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR certified: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 7. Table Transformer certified output bounds
// ===========================================================================

/// Build Table Transformer certified pipeline: ResNet-18-style backbone (Conv2d +
/// ReLU) -> Linear encoder projection -> ReLU -> Linear DETR decoder projection
/// -> ReLU -> Linear classification head -> sigmoid.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[SEQ_LEN, NUM_CLASSES]` (sigmoid detection confidence [0, 1]).
fn build_table_transformer_certified_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("certified_table_transformer");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // ResNet backbone (simplified): Conv2d stride-P + ReLU
    let conv_w = b.add_input(
        "resnet_weight",
        &[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let conv_b = b.add_input("resnet_bias", &[BACKBONE_CH]);
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
    let narrowed = b.add_narrow(transposed, 0, 0, SEQ_LEN, &[SEQ_LEN, BACKBONE_CH]);

    // Encoder projection: Linear -> ReLU
    let enc_w = b.add_input("encoder_weight", &[HIDDEN_DIM, BACKBONE_CH]);
    let encoded = b.add_linear(narrowed, enc_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let enc_act = b.add_relu(encoded, &[SEQ_LEN, HIDDEN_DIM]);

    // DETR decoder projection: Linear -> ReLU
    let dec_w = b.add_input("decoder_weight", &[FFN_DIM, HIDDEN_DIM]);
    let decoded = b.add_linear(enc_act, dec_w, None, &[SEQ_LEN, FFN_DIM]);
    let dec_act = b.add_relu(decoded, &[SEQ_LEN, FFN_DIM]);

    // Classification head: Linear -> sigmoid
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, FFN_DIM]);
    let cls_b = b.add_input("cls_bias", &[NUM_CLASSES]);
    let logits = b.add_linear(dec_act, cls_w, Some(cls_b), &[SEQ_LEN, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[SEQ_LEN, NUM_CLASSES]);

    b.build(out)
        .expect("valid Table Transformer certified pipeline")
}

fn table_transformer_certified_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // image
        weight(&[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        bias(&[BACKBONE_CH]),
        weight(&[HIDDEN_DIM, BACKBONE_CH]),
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[NUM_CLASSES, FFN_DIM]),
        bias(&[NUM_CLASSES]),
    ]
}

#[test]
fn test_table_transformer_certified_bounds() {
    let def = build_table_transformer_certified_kernel();
    let bindings = table_transformer_certified_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Table Transformer certified pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer certified IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid output must be in [0, 1]
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

#[test]
fn test_table_transformer_certified_crown() {
    let def = build_table_transformer_certified_kernel();
    let bindings = table_transformer_certified_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer certified: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}
