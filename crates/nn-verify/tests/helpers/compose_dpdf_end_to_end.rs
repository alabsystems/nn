// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end model pipeline compose tests for dpdf document understanding.
//!
//! These tests verify NY IBP/CROWN bounds propagation through complete
//! model pipelines — from raw image input to final output (class probabilities,
//! bounding box coordinates, character probabilities). While per-block tests
//! (granite_docling, doclayout_yolo, etc.) verify sub-graphs in isolation,
//! these tests verify the compositional property: that bounds compose correctly
//! when blocks are chained into full inference pipelines.
//!
//! ## Tests:
//!
//! 1. **DocLayout-YOLO full pipeline IBP**: Conv2d backbone -> Linear neck ->
//!    ReLU -> Linear detect -> sigmoid classification. End-to-end from image
//!    pixels [0,1] to class confidence [0,1].
//!
//! 2. **Table Transformer full pipeline IBP**: Conv2d ResNet backbone -> Linear
//!    encoder -> ReLU -> Linear decoder -> sigmoid heads. End-to-end from image
//!    to table detection + structure confidence.
//!
//! 3. **PaddleOCR detection pipeline IBP**: Conv2d -> Linear -> ReLU ->
//!    sigmoid probability map. Text detection from image pixels.
//!
//! 4. **PaddleOCR recognition pipeline IBP**: Conv2d patch embed -> reshape ->
//!    transpose -> Linear SVTR -> GELU -> Linear CTC -> softmax character
//!    probabilities [0,1].
//!
//! 5. **GLM-OCR full pipeline IBP**: Linear embedding -> RMSNorm -> Linear
//!    attention proj -> sigmoid -> Linear FFN -> RMSNorm -> Linear MTP head
//!    -> softmax token probabilities.
//!
//! 6. **Granite-Docling full pipeline IBP**: Conv2d patch embed -> reshape ->
//!    transpose -> Linear ViT proj -> ReLU -> Linear vision-to-LM projection
//!    -> Linear decoder FFN -> sigmoid output.
//!
//! 7. **Qwen3-VL full pipeline IBP**: Conv2d patch embed -> reshape -> transpose
//!    -> Linear window attention -> sigmoid gate -> Linear projection -> Linear
//!    decoder FFN -> softmax output.
//!
//! 8. **FireRed-OCR full pipeline IBP**: Conv2d patch embed -> reshape ->
//!    transpose -> Linear encoder -> ReLU -> Linear CTC head -> softmax
//!    character probabilities [0,1].
//!
//! 9. **Detection + recognition cascade IBP**: Conv2d backbone -> sigmoid
//!    detection -> Linear projection -> GELU -> Linear CTC -> softmax
//!    recognition. Two-stage from image to characters.
//!
//! 10. **VLM + layout detection pipeline IBP**: Linear VLM features -> GELU
//!     -> Linear projection -> sigmoid layout detection. VLM guiding layout.
//!
//! 11. **Multi-model ensemble (detection union) IBP**: Two parallel detection
//!     heads (sigmoid) combined via add for ensemble bounds.
//!
//! 12. **Pipeline with quantized decoder IBP**: Linear dequant (scale * code)
//!     -> Linear projection -> sigmoid output. INT4-aware pipeline.
//!
//! 13. **Pipeline monotone tightening IBP**: Full detection pipeline with
//!     two epsilon levels proving tighter input -> tighter output.
//!
//! 14. **Detection + recognition cascade CROWN**: Same as test 9 with CROWN
//!     linearization for tighter cross-stage bounds.
//!
//! 15. **Granite-Docling full pipeline CROWN**: Same as test 6 with CROWN
//!     linearization through the full VLM pipeline.
//!
//! Architecture references:
//! - DocLayout-YOLO (Zhao et al. 2024): YOLOv10-based document layout detection
//! - Table Transformer (Smock et al. 2022): DETR-based table structure
//! - PaddleOCR (Baidu): DB detector + SVTR recognizer
//! - GLM-4V (THUDM): Vision-language model with GLM-4 decoder
//! - Granite-Docling: SigLIP2 vision encoder + Granite LLM decoder
//! - Qwen3-VL (Alibaba): Vision-language model with 3D patch embedding
//! - FireRed-OCR: Qwen3-VL-2B variant for document OCR with CTC decoding
//!
//! Dimensions (small for fast verification, structurally representative):
//! - IMG_SIZE=16, PATCH_SIZE=8, HIDDEN_DIM=32, FFN_DIM=64, SEQ_LEN=4,
//!   NUM_CLASSES=8, VOCAB_SIZE=16, BACKBONE_CH=16, IN_CHANNELS=3,
//!   NUM_PATCHES=4
//!
//! Part of #3987: end-to-end model pipeline compose tests.

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
/// Sequence length for text positions.
const SEQ_LEN: usize = 4;
/// Number of detection classes.
const NUM_CLASSES: usize = 8;
/// OCR vocabulary size (characters + blank).
const VOCAB_SIZE: usize = 16;
/// Backbone output channels.
const BACKBONE_CH: usize = 16;
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
// 1. DocLayout-YOLO full pipeline IBP
// ===========================================================================

/// Build DocLayout-YOLO full pipeline: image -> backbone -> neck -> detect -> sigmoid.
///
/// Conv2d backbone extracts spatial features, linear neck projects to detection
/// space, and sigmoid produces class confidence in [0, 1].
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[SEQ_LEN, NUM_CLASSES]` (detection sigmoid [0, 1]).
fn build_doclayout_yolo_full_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("e2e_doclayout_yolo_full");

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

    // Reshape: [BACKBONE_CH, 2, 2] -> [BACKBONE_CH, NUM_PATCHES]
    let reshaped = b.add_reshape(backbone, &[BACKBONE_CH, NUM_PATCHES]);

    // Transpose: [BACKBONE_CH, NUM_PATCHES] -> [NUM_PATCHES, BACKBONE_CH]
    let transposed = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, BACKBONE_CH]);

    // Narrow to SEQ_LEN positions
    let narrowed = b.add_narrow(transposed, 0, 0, SEQ_LEN, &[SEQ_LEN, BACKBONE_CH]);

    // Neck: Linear [SEQ_LEN, BACKBONE_CH] -> [SEQ_LEN, FFN_DIM]
    let neck_w = b.add_input("neck_weight", &[FFN_DIM, BACKBONE_CH]);
    let neck = b.add_linear(narrowed, neck_w, None, &[SEQ_LEN, FFN_DIM]);
    let neck_act = b.add_relu(neck, &[SEQ_LEN, FFN_DIM]);

    // Detect head: Linear -> sigmoid
    let det_w = b.add_input("detect_weight", &[NUM_CLASSES, FFN_DIM]);
    let det_b = b.add_input("detect_bias", &[NUM_CLASSES]);
    let logits = b.add_linear(neck_act, det_w, Some(det_b), &[SEQ_LEN, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[SEQ_LEN, NUM_CLASSES]);

    b.build(out).expect("valid DocLayout-YOLO full pipeline")
}

fn doclayout_yolo_full_bindings() -> Vec<TensorParamBinding> {
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
fn test_e2e_doclayout_yolo_full_ibp() {
    let def = build_doclayout_yolo_full_kernel();
    let bindings = doclayout_yolo_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DocLayout-YOLO full pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DocLayout-YOLO full pipeline IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Sigmoid output bounded in (0, 1)
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 2. Table Transformer full pipeline IBP
// ===========================================================================

/// Build Table Transformer full pipeline: image -> ResNet backbone -> encoder
/// projection -> ReLU -> decoder projection -> sigmoid heads.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[SEQ_LEN, NUM_CLASSES]` (detection sigmoid [0, 1]).
fn build_table_transformer_full_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("e2e_table_transformer_full");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // ResNet backbone (simplified): Conv2d stride-P
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

    // Reshape + transpose to sequence
    let reshaped = b.add_reshape(backbone, &[BACKBONE_CH, NUM_PATCHES]);
    let transposed = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, BACKBONE_CH]);
    let narrowed = b.add_narrow(transposed, 0, 0, SEQ_LEN, &[SEQ_LEN, BACKBONE_CH]);

    // Encoder projection: Linear -> ReLU
    let enc_w = b.add_input("encoder_weight", &[HIDDEN_DIM, BACKBONE_CH]);
    let encoded = b.add_linear(narrowed, enc_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let enc_act = b.add_relu(encoded, &[SEQ_LEN, HIDDEN_DIM]);

    // Decoder projection: Linear -> ReLU
    let dec_w = b.add_input("decoder_weight", &[FFN_DIM, HIDDEN_DIM]);
    let decoded = b.add_linear(enc_act, dec_w, None, &[SEQ_LEN, FFN_DIM]);
    let dec_act = b.add_relu(decoded, &[SEQ_LEN, FFN_DIM]);

    // Classification head: Linear -> sigmoid
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, FFN_DIM]);
    let cls_b = b.add_input("cls_bias", &[NUM_CLASSES]);
    let logits = b.add_linear(dec_act, cls_w, Some(cls_b), &[SEQ_LEN, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[SEQ_LEN, NUM_CLASSES]);

    b.build(out).expect("valid Table Transformer full pipeline")
}

fn table_transformer_full_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        bias(&[BACKBONE_CH]),
        weight(&[HIDDEN_DIM, BACKBONE_CH]),
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[NUM_CLASSES, FFN_DIM]),
        bias(&[NUM_CLASSES]),
    ]
}

#[test]
fn test_e2e_table_transformer_full_ibp() {
    let def = build_table_transformer_full_kernel();
    let bindings = table_transformer_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Table Transformer full pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table Transformer full pipeline IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite() && hi_max.is_finite());
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 3. PaddleOCR detection pipeline IBP
// ===========================================================================

/// Build PaddleOCR detection: Conv2d backbone -> Linear -> ReLU -> sigmoid map.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[SEQ_LEN, 1]` (sigmoid probability map [0, 1]).
fn build_paddle_ocr_detection_full_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("e2e_paddle_ocr_detection");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Conv-BN backbone (simplified as Conv2d)
    let conv_w = b.add_input(
        "conv_weight",
        &[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let conv_b = b.add_input("conv_bias", &[BACKBONE_CH]);
    let features = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[BACKBONE_CH, GRID_SIZE, GRID_SIZE],
    );

    // Reshape to sequence
    let reshaped = b.add_reshape(features, &[BACKBONE_CH, NUM_PATCHES]);
    let transposed = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, BACKBONE_CH]);
    let narrowed = b.add_narrow(transposed, 0, 0, SEQ_LEN, &[SEQ_LEN, BACKBONE_CH]);

    // Projection -> ReLU -> sigmoid probability map
    let proj_w = b.add_input("proj_weight", &[HIDDEN_DIM, BACKBONE_CH]);
    let projected = b.add_linear(narrowed, proj_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let activated = b.add_relu(projected, &[SEQ_LEN, HIDDEN_DIM]);

    let sig_w = b.add_input("sigmoid_weight", &[1, HIDDEN_DIM]);
    let sig_b = b.add_input("sigmoid_bias", &[1]);
    let logits = b.add_linear(activated, sig_w, Some(sig_b), &[SEQ_LEN, 1]);
    let out = b.add_sigmoid(logits, &[SEQ_LEN, 1]);

    b.build(out)
        .expect("valid PaddleOCR detection full pipeline")
}

fn paddle_ocr_detection_full_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        bias(&[BACKBONE_CH]),
        weight(&[HIDDEN_DIM, BACKBONE_CH]),
        weight(&[1, HIDDEN_DIM]),
        bias(&[1]),
    ]
}

#[test]
fn test_e2e_paddle_ocr_detection_ibp() {
    let def = build_paddle_ocr_detection_full_kernel();
    let bindings = paddle_ocr_detection_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR detection pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, 1]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR detection pipeline IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite() && hi_max.is_finite());
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 4. PaddleOCR recognition pipeline IBP
// ===========================================================================

/// Build PaddleOCR recognition: patch embed -> SVTR projection -> GELU ->
/// CTC head -> softmax character probabilities.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (CTC softmax [0, 1]).
fn build_paddle_ocr_recognition_full_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("e2e_paddle_ocr_recognition");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Patch embedding: Conv2d -> reshape -> transpose
    let patch_w = b.add_input(
        "patch_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_b = b.add_input("patch_bias", &[HIDDEN_DIM]);
    let conv_out = b.add_conv2d(
        input,
        patch_w,
        Some(patch_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);
    let patches = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, HIDDEN_DIM]);
    let narrowed = b.add_narrow(patches, 0, 0, SEQ_LEN, &[SEQ_LEN, HIDDEN_DIM]);

    // SVTR encoder (simplified): Linear -> GELU
    let svtr_w = b.add_input("svtr_weight", &[FFN_DIM, HIDDEN_DIM]);
    let svtr_out = b.add_linear(narrowed, svtr_w, None, &[SEQ_LEN, FFN_DIM]);
    let svtr_act = b.add_gelu(svtr_out, &[SEQ_LEN, FFN_DIM]);

    // CTC head: Linear -> Softmax
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, FFN_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(svtr_act, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid PaddleOCR recognition full pipeline")
}

fn paddle_ocr_recognition_full_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        bias(&[HIDDEN_DIM]),
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, FFN_DIM]),
        bias(&[VOCAB_SIZE]),
    ]
}

#[test]
fn test_e2e_paddle_ocr_recognition_ibp() {
    let def = build_paddle_ocr_recognition_full_kernel();
    let bindings = paddle_ocr_recognition_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through PaddleOCR recognition pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("PaddleOCR recognition pipeline IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite() && hi_max.is_finite());
    // Softmax output bounded in [0, 1]
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 5. GLM-OCR full pipeline IBP
// ===========================================================================

/// Build GLM-OCR full pipeline: embedding -> RMSNorm -> attention proj ->
/// SiLU gate -> FFN -> RMSNorm -> MTP head -> softmax.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, token embeddings [-1, 1]).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (softmax token probabilities [0, 1]).
fn build_glm_ocr_full_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("e2e_glm_ocr_full");

    let input = b.add_input("embeddings", &[SEQ_LEN, HIDDEN_DIM]);

    // RMSNorm
    let eps1 = b.add_input("eps1", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps1, 1, norm1_w, &[SEQ_LEN, HIDDEN_DIM]);

    // Attention projection (simplified as Linear -> sigmoid gate)
    let attn_w = b.add_input("attn_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let attn_out = b.add_linear(normed, attn_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let gate = b.add_sigmoid(attn_out, &[SEQ_LEN, HIDDEN_DIM]);

    // Residual + gated output
    let gated = b.add_binary_mul(attn_out, gate, &[SEQ_LEN, HIDDEN_DIM]);
    let residual1 = b.add_binary_add(input, gated, &[SEQ_LEN, HIDDEN_DIM]);

    // FFN: Linear -> ReLU -> Linear
    let ffn1_w = b.add_input("ffn1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let ffn1 = b.add_linear(residual1, ffn1_w, None, &[SEQ_LEN, FFN_DIM]);
    let ffn1_act = b.add_relu(ffn1, &[SEQ_LEN, FFN_DIM]);
    let ffn2_w = b.add_input("ffn2_weight", &[HIDDEN_DIM, FFN_DIM]);
    let ffn2 = b.add_linear(ffn1_act, ffn2_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let residual2 = b.add_binary_add(residual1, ffn2, &[SEQ_LEN, HIDDEN_DIM]);

    // Final RMSNorm
    let eps2 = b.add_input("eps2", &[1]);
    let norm2_w = b.add_input("norm2_weight", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(residual2, eps2, 1, norm2_w, &[SEQ_LEN, HIDDEN_DIM]);

    // MTP head: Linear -> softmax
    let head_w = b.add_input("head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let head_b = b.add_input("head_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(normed2, head_w, Some(head_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid GLM-OCR full pipeline")
}

fn glm_ocr_full_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,      // embeddings
        eps_binding(),                     // eps1
        norm_weight_binding(HIDDEN_DIM),   // norm1_weight
        weight(&[HIDDEN_DIM, HIDDEN_DIM]), // attn_weight
        weight(&[FFN_DIM, HIDDEN_DIM]),    // ffn1_weight
        weight(&[HIDDEN_DIM, FFN_DIM]),    // ffn2_weight
        eps_binding(),                     // eps2
        norm_weight_binding(HIDDEN_DIM),   // norm2_weight
        weight(&[VOCAB_SIZE, HIDDEN_DIM]), // head_weight
        bias(&[VOCAB_SIZE]),               // head_bias
    ]
}

#[test]
fn test_e2e_glm_ocr_full_ibp() {
    let def = build_glm_ocr_full_kernel();
    let bindings = glm_ocr_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through GLM-OCR full pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GLM-OCR full pipeline IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite() && hi_max.is_finite());
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 6. Granite-Docling full pipeline IBP
// ===========================================================================

/// Build Granite-Docling full pipeline: patch embed -> ViT projection ->
/// ReLU -> vision-to-LM projection -> decoder FFN -> sigmoid output.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[SEQ_LEN, NUM_CLASSES]` (sigmoid [0, 1]).
fn build_granite_docling_full_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("e2e_granite_docling_full");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Patch embedding: Conv2d -> reshape -> transpose
    let patch_w = b.add_input(
        "patch_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_b = b.add_input("patch_bias", &[HIDDEN_DIM]);
    let conv_out = b.add_conv2d(
        input,
        patch_w,
        Some(patch_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);
    let patches = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, HIDDEN_DIM]);
    let narrowed = b.add_narrow(patches, 0, 0, SEQ_LEN, &[SEQ_LEN, HIDDEN_DIM]);

    // ViT projection: Linear -> ReLU
    let vit_w = b.add_input("vit_weight", &[FFN_DIM, HIDDEN_DIM]);
    let vit_out = b.add_linear(narrowed, vit_w, None, &[SEQ_LEN, FFN_DIM]);
    let vit_act = b.add_relu(vit_out, &[SEQ_LEN, FFN_DIM]);

    // Vision-to-LM projection: Linear
    let proj_w = b.add_input("projection_weight", &[HIDDEN_DIM, FFN_DIM]);
    let projected = b.add_linear(vit_act, proj_w, None, &[SEQ_LEN, HIDDEN_DIM]);

    // Decoder FFN: Linear -> ReLU -> Linear -> sigmoid
    let dec1_w = b.add_input("decoder_ffn1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let dec1 = b.add_linear(projected, dec1_w, None, &[SEQ_LEN, FFN_DIM]);
    let dec1_act = b.add_relu(dec1, &[SEQ_LEN, FFN_DIM]);

    let dec2_w = b.add_input("decoder_ffn2_weight", &[NUM_CLASSES, FFN_DIM]);
    let dec2_b = b.add_input("decoder_ffn2_bias", &[NUM_CLASSES]);
    let logits = b.add_linear(dec1_act, dec2_w, Some(dec2_b), &[SEQ_LEN, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[SEQ_LEN, NUM_CLASSES]);

    b.build(out).expect("valid Granite-Docling full pipeline")
}

fn granite_docling_full_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        bias(&[HIDDEN_DIM]),
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, FFN_DIM]),
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[NUM_CLASSES, FFN_DIM]),
        bias(&[NUM_CLASSES]),
    ]
}

#[test]
fn test_e2e_granite_docling_full_ibp() {
    let def = build_granite_docling_full_kernel();
    let bindings = granite_docling_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Granite-Docling full pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite-Docling full pipeline IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite() && hi_max.is_finite());
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 7. Qwen3-VL full pipeline IBP
// ===========================================================================

/// Build Qwen3-VL full pipeline: patch embed -> window attention (simplified) ->
/// sigmoid gate -> projection -> decoder FFN -> softmax.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (softmax [0, 1]).
fn build_qwen3_vl_full_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("e2e_qwen3_vl_full");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Patch embedding: Conv2d -> reshape -> transpose
    let patch_w = b.add_input(
        "patch_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_b = b.add_input("patch_bias", &[HIDDEN_DIM]);
    let conv_out = b.add_conv2d(
        input,
        patch_w,
        Some(patch_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);
    let patches = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, HIDDEN_DIM]);
    let narrowed = b.add_narrow(patches, 0, 0, SEQ_LEN, &[SEQ_LEN, HIDDEN_DIM]);

    // Window attention (simplified as Linear -> sigmoid gate)
    let attn_w = b.add_input("window_attn_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let attn_out = b.add_linear(narrowed, attn_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let gate = b.add_sigmoid(attn_out, &[SEQ_LEN, HIDDEN_DIM]);
    let gated = b.add_binary_mul(attn_out, gate, &[SEQ_LEN, HIDDEN_DIM]);

    // Vision-to-LM projection
    let proj_w = b.add_input("projection_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let projected = b.add_linear(gated, proj_w, None, &[SEQ_LEN, HIDDEN_DIM]);

    // Decoder FFN: Linear -> ReLU -> Linear -> softmax
    let ffn1_w = b.add_input("ffn1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let ffn1 = b.add_linear(projected, ffn1_w, None, &[SEQ_LEN, FFN_DIM]);
    let ffn1_act = b.add_relu(ffn1, &[SEQ_LEN, FFN_DIM]);

    let ffn2_w = b.add_input("ffn2_weight", &[VOCAB_SIZE, FFN_DIM]);
    let ffn2_b = b.add_input("ffn2_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(ffn1_act, ffn2_w, Some(ffn2_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid Qwen3-VL full pipeline")
}

fn qwen3_vl_full_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        bias(&[HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, FFN_DIM]),
        bias(&[VOCAB_SIZE]),
    ]
}

#[test]
fn test_e2e_qwen3_vl_full_ibp() {
    let def = build_qwen3_vl_full_kernel();
    let bindings = qwen3_vl_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL full pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL full pipeline IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite() && hi_max.is_finite());
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 8. FireRed-OCR full pipeline IBP
// ===========================================================================

/// Build FireRed-OCR full pipeline: patch embed -> encoder (Linear + ReLU) ->
/// CTC head -> softmax character probabilities.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (CTC softmax [0, 1]).
fn build_firered_ocr_full_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("e2e_firered_ocr_full");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Patch embedding: Conv2d -> reshape -> transpose
    let patch_w = b.add_input(
        "patch_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_b = b.add_input("patch_bias", &[HIDDEN_DIM]);
    let conv_out = b.add_conv2d(
        input,
        patch_w,
        Some(patch_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);
    let patches = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, HIDDEN_DIM]);
    let narrowed = b.add_narrow(patches, 0, 0, SEQ_LEN, &[SEQ_LEN, HIDDEN_DIM]);

    // Encoder block (simplified): Linear -> ReLU + residual
    let enc_w = b.add_input("encoder_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let enc_out = b.add_linear(narrowed, enc_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let enc_act = b.add_relu(enc_out, &[SEQ_LEN, HIDDEN_DIM]);
    let residual = b.add_binary_add(narrowed, enc_act, &[SEQ_LEN, HIDDEN_DIM]);

    // CTC head: Linear -> Softmax
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(residual, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid FireRed-OCR full pipeline")
}

fn firered_ocr_full_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        bias(&[HIDDEN_DIM]),
        weight(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, HIDDEN_DIM]),
        bias(&[VOCAB_SIZE]),
    ]
}

#[test]
fn test_e2e_firered_ocr_full_ibp() {
    let def = build_firered_ocr_full_kernel();
    let bindings = firered_ocr_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR full pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR full pipeline IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite() && hi_max.is_finite());
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 9. Detection + recognition cascade IBP
// ===========================================================================

/// Build detection + recognition cascade: Conv backbone -> sigmoid detection ->
/// projection -> GELU -> CTC head -> softmax recognition.
///
/// Two-stage pipeline: image -> detection confidence -> character probabilities.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (CTC softmax [0, 1]).
fn build_detection_recognition_cascade_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("e2e_detection_recognition_cascade");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Detection backbone: Conv2d -> reshape -> transpose
    let conv_w = b.add_input(
        "det_conv_weight",
        &[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let conv_b = b.add_input("det_conv_bias", &[BACKBONE_CH]);
    let features = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[BACKBONE_CH, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(features, &[BACKBONE_CH, NUM_PATCHES]);
    let transposed = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, BACKBONE_CH]);
    let narrowed = b.add_narrow(transposed, 0, 0, SEQ_LEN, &[SEQ_LEN, BACKBONE_CH]);

    // Detection head: Linear -> sigmoid [0, 1]
    let det_w = b.add_input("det_weight", &[HIDDEN_DIM, BACKBONE_CH]);
    let det_logits = b.add_linear(narrowed, det_w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let det_out = b.add_sigmoid(det_logits, &[SEQ_LEN, HIDDEN_DIM]);

    // Recognition: projection -> GELU -> CTC -> softmax
    let rec_w = b.add_input("rec_weight", &[FFN_DIM, HIDDEN_DIM]);
    let rec_out = b.add_linear(det_out, rec_w, None, &[SEQ_LEN, FFN_DIM]);
    let rec_act = b.add_gelu(rec_out, &[SEQ_LEN, FFN_DIM]);

    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, FFN_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(rec_act, ctc_w, Some(ctc_b), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid detection + recognition cascade")
}

fn detection_recognition_cascade_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        bias(&[BACKBONE_CH]),
        weight(&[HIDDEN_DIM, BACKBONE_CH]),
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[VOCAB_SIZE, FFN_DIM]),
        bias(&[VOCAB_SIZE]),
    ]
}

#[test]
fn test_e2e_detection_recognition_cascade_ibp() {
    let def = build_detection_recognition_cascade_kernel();
    let bindings = detection_recognition_cascade_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through detection + recognition cascade");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Detection + recognition cascade IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite() && hi_max.is_finite());
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 10. VLM + layout detection pipeline IBP
// ===========================================================================

/// Build VLM + layout detection: VLM features -> GELU -> projection -> sigmoid.
///
/// Models a VLM (e.g., Qwen3-VL or Granite-Docling) guiding layout detection.
/// VLM decoder features are projected and GELU-activated, then a detection head
/// produces sigmoid layout confidence.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, VLM decoder features [-1, 1]).
/// Output: `[SEQ_LEN, NUM_CLASSES]` (sigmoid layout confidence [0, 1]).
fn build_vlm_layout_detection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("e2e_vlm_layout_detection");

    let input = b.add_input("vlm_features", &[SEQ_LEN, HIDDEN_DIM]);

    // VLM feature projection: Linear -> GELU
    let proj_w = b.add_input("proj_weight", &[FFN_DIM, HIDDEN_DIM]);
    let projected = b.add_linear(input, proj_w, None, &[SEQ_LEN, FFN_DIM]);
    let activated = b.add_gelu(projected, &[SEQ_LEN, FFN_DIM]);

    // Layout detection head: Linear -> sigmoid
    let det_w = b.add_input("det_weight", &[NUM_CLASSES, FFN_DIM]);
    let det_b = b.add_input("det_bias", &[NUM_CLASSES]);
    let logits = b.add_linear(activated, det_w, Some(det_b), &[SEQ_LEN, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[SEQ_LEN, NUM_CLASSES]);

    b.build(out).expect("valid VLM + layout detection pipeline")
}

fn vlm_layout_detection_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[NUM_CLASSES, FFN_DIM]),
        bias(&[NUM_CLASSES]),
    ]
}

#[test]
fn test_e2e_vlm_layout_detection_ibp() {
    let def = build_vlm_layout_detection_kernel();
    let bindings = vlm_layout_detection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through VLM + layout detection pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("VLM + layout detection pipeline IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite() && hi_max.is_finite());
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 11. Multi-model ensemble (detection union) IBP
// ===========================================================================

/// Build multi-model ensemble: two parallel detection heads with additive fusion.
///
/// Two independent Linear -> sigmoid detection branches are combined via
/// element-wise addition. The ensemble bounds are wider than individual heads,
/// verifying that compositional bounds remain valid.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, shared features [-1, 1]).
/// Output: `[SEQ_LEN, NUM_CLASSES]` (ensemble detection scores).
fn build_multi_model_ensemble_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("e2e_multi_model_ensemble");

    let input = b.add_input("features", &[SEQ_LEN, HIDDEN_DIM]);

    // Head A: Linear -> sigmoid
    let head_a_w = b.add_input("head_a_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let head_a_b = b.add_input("head_a_bias", &[NUM_CLASSES]);
    let logits_a = b.add_linear(input, head_a_w, Some(head_a_b), &[SEQ_LEN, NUM_CLASSES]);
    let det_a = b.add_sigmoid(logits_a, &[SEQ_LEN, NUM_CLASSES]);

    // Head B: Linear -> sigmoid
    let head_b_w = b.add_input("head_b_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let head_b_b = b.add_input("head_b_bias", &[NUM_CLASSES]);
    let logits_b = b.add_linear(input, head_b_w, Some(head_b_b), &[SEQ_LEN, NUM_CLASSES]);
    let det_b = b.add_sigmoid(logits_b, &[SEQ_LEN, NUM_CLASSES]);

    // Ensemble fusion: element-wise add
    let out = b.add_binary_add(det_a, det_b, &[SEQ_LEN, NUM_CLASSES]);

    b.build(out).expect("valid multi-model ensemble pipeline")
}

fn multi_model_ensemble_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight(&[NUM_CLASSES, HIDDEN_DIM]),
        bias(&[NUM_CLASSES]),
        weight(&[NUM_CLASSES, HIDDEN_DIM]),
        bias(&[NUM_CLASSES]),
    ]
}

#[test]
fn test_e2e_multi_model_ensemble_ibp() {
    let def = build_multi_model_ensemble_kernel();
    let bindings = multi_model_ensemble_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through multi-model ensemble");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Multi-model ensemble IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite() && hi_max.is_finite());
    // Sum of two sigmoids: lower >= 0 (each sigmoid >= 0)
    assert!(lo_min >= -1e-4, "ensemble lower >= 0, got {lo_min}");
    // Sum of two sigmoids: upper <= 2 (each sigmoid <= 1)
    assert!(hi_max <= 2.0 + 1e-4, "ensemble upper <= 2, got {hi_max}");
}

// ===========================================================================
// 12. Pipeline with quantized decoder IBP
// ===========================================================================

/// Build quantized decoder pipeline: dequantize (scale * code) -> Linear -> sigmoid.
///
/// Models INT4 dequantization followed by a decoder projection. The
/// dequantization is modeled as element-wise multiply (scale * discrete code),
/// which introduces bounded quantization noise.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, quantized codes [-8, 7]).
/// Output: `[SEQ_LEN, NUM_CLASSES]` (sigmoid [0, 1]).
fn build_quantized_decoder_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("e2e_quantized_decoder_pipeline");

    let input = b.add_input("quant_codes", &[SEQ_LEN, HIDDEN_DIM]);

    // Dequantization: scale * code (modeled as Linear with scale matrix)
    let scale_w = b.add_input("scale_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let dequantized = b.add_linear(input, scale_w, None, &[SEQ_LEN, HIDDEN_DIM]);

    // Projection: Linear -> ReLU
    let proj_w = b.add_input("proj_weight", &[FFN_DIM, HIDDEN_DIM]);
    let projected = b.add_linear(dequantized, proj_w, None, &[SEQ_LEN, FFN_DIM]);
    let activated = b.add_relu(projected, &[SEQ_LEN, FFN_DIM]);

    // Output head: Linear -> sigmoid
    let out_w = b.add_input("out_weight", &[NUM_CLASSES, FFN_DIM]);
    let out_b = b.add_input("out_bias", &[NUM_CLASSES]);
    let logits = b.add_linear(activated, out_w, Some(out_b), &[SEQ_LEN, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[SEQ_LEN, NUM_CLASSES]);

    b.build(out).expect("valid quantized decoder pipeline")
}

fn quantized_decoder_pipeline_bindings() -> Vec<TensorParamBinding> {
    // Scale matrix with small magnitude (typical INT4 scale factors)
    let scale = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), 0.01f32);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(scale),
        weight(&[FFN_DIM, HIDDEN_DIM]),
        weight(&[NUM_CLASSES, FFN_DIM]),
        bias(&[NUM_CLASSES]),
    ]
}

#[test]
fn test_e2e_quantized_decoder_pipeline_ibp() {
    let def = build_quantized_decoder_pipeline_kernel();
    let bindings = quantized_decoder_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // INT4 codes: values in [-8, 7]
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), -8.0f32),
        ArrayD::from_elem(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), 7.0f32),
    )
    .expect("valid INT4 code bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through quantized decoder pipeline");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Quantized decoder pipeline IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite() && hi_max.is_finite());
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 13. Pipeline monotone tightening IBP
// ===========================================================================

/// Verify monotone tightening: tighter input bounds produce tighter output bounds
/// through the full DocLayout-YOLO detection pipeline.
///
/// Two runs:
/// - Wide: image [0, 1] (full pixel range)
/// - Tight: image [0.3, 0.7] (restricted pixel range)
///
/// The tight-input run must produce equal or tighter output bounds.
#[test]
fn test_e2e_pipeline_monotone_tightening_ibp() {
    let def = build_doclayout_yolo_full_kernel();
    let bindings = doclayout_yolo_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Wide input: pixels [0, 1]
    let wide_input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let wide_output = graph.propagate_ibp(&wide_input).expect("wide IBP");

    // Tight input: pixels [0.3, 0.7]
    let tight_input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 0.3f32),
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 0.7f32),
    )
    .expect("valid tight bounds");
    let tight_output = graph.propagate_ibp(&tight_input).expect("tight IBP");

    assert_bounds_valid(&wide_output);
    assert_bounds_valid(&tight_output);

    let (wide_lo, wide_hi) = wide_output.lower_upper();
    let (tight_lo, tight_hi) = tight_output.lower_upper();

    // Tight bounds should produce equal or tighter output
    let eps = 1e-4;
    let tighter_or_equal = tight_lo
        .iter()
        .zip(wide_lo.iter())
        .all(|(&tl, &wl)| tl >= wl - eps)
        && tight_hi
            .iter()
            .zip(wide_hi.iter())
            .all(|(&th, &wh)| th <= wh + eps);

    let wide_width: f32 = wide_hi
        .iter()
        .zip(wide_lo.iter())
        .map(|(&h, &l)| h - l)
        .sum();
    let tight_width: f32 = tight_hi
        .iter()
        .zip(tight_lo.iter())
        .map(|(&h, &l)| h - l)
        .sum();

    eprintln!(
        "Monotone tightening: wide_width={wide_width:.6}, tight_width={tight_width:.6}, \
         tighter_or_equal={tighter_or_equal}"
    );

    assert!(
        tighter_or_equal,
        "tighter input must produce equal or tighter output"
    );
    assert!(
        tight_width <= wide_width + eps,
        "tight total width ({tight_width:.6}) must be <= wide total width ({wide_width:.6})"
    );
}

// ===========================================================================
// 14. Detection + recognition cascade CROWN
// ===========================================================================

/// CROWN bounds through detection + recognition cascade for tighter cross-stage
/// bounds. Same pipeline as test 9 with CROWN linearization.
#[test]
fn test_e2e_detection_recognition_cascade_crown() {
    let def = build_detection_recognition_cascade_kernel();
    let bindings = detection_recognition_cascade_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Detection + recognition cascade CROWN (method={method:?}): bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("  fallback reason: {reason}");
    }

    assert!(lo_min.is_finite() && hi_max.is_finite());
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 15. Granite-Docling full pipeline CROWN
// ===========================================================================

/// CROWN bounds through the full Granite-Docling pipeline for tighter
/// end-to-end VLM bounds. Same pipeline as test 6 with CROWN linearization.
#[test]
fn test_e2e_granite_docling_full_crown() {
    let def = build_granite_docling_full_kernel();
    let bindings = granite_docling_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Granite-Docling full pipeline CROWN (method={method:?}): bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("  fallback reason: {reason}");
    }

    assert!(lo_min.is_finite() && hi_max.is_finite());
    assert!(lo_min >= -1e-4, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "sigmoid upper <= 1, got {hi_max}");
}
