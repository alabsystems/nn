// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-model standalone bounds verification for the dpdf 7-model ensemble.
//!
//! Each of the 7 models in the document processing ensemble gets an individual
//! bounds test verifying its subgraph produces correctly bounded output.
//! Complements `compose_dpdf_7model_ensemble.rs` (pipeline-level composition).
//!
//! ## Tests (7 tests)
//!
//! 1. DocLayout-YOLO: conv -> relu -> sigmoid cls + sigmoid bbox (IBP)
//! 2. Table Transformer: encoder FFN + residual -> sigmoid head (IBP + CROWN)
//! 3. Granite-Docling: vision proj -> LM FFN -> softmax (IBP)
//! 4. PaddleOCR-VL: SVTR feature extractor -> softmax char dist (IBP)
//! 5. FireRed-OCR: CTC decoder -> softmax (IBP + CROWN)
//! 6. GLM-OCR: SwiGLU-style FFN + residual -> LM head softmax (IBP)
//! 7. Qwen3-VL: vision MLP (GELU) -> decoder FFN -> sigmoid (IBP)
//!
//! Part of #4243: dpdf 7-model ensemble compose verification tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

const HIDDEN: usize = 8;
const SEQ: usize = 4;
const NUM_CLASSES: usize = 6;
const VOCAB: usize = 8;
const FFN_DIM: usize = HIDDEN * 2;
const PATCH_DIM: usize = 4;
const WEIGHT_MAG: f32 = 0.02;

fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

fn bias_zero(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

// ===========================================================================
// 1. DocLayout-YOLO standalone (IBP)
// ===========================================================================

/// DocLayout-YOLO: conv feature extraction -> ReLU -> sigmoid classification
/// + sigmoid bbox regression. Verifies detection head outputs in [0, 1].
#[test]
fn test_7model_ext_doclayout_yolo_standalone_ibp() {
    let mut b = TensorBlockBuilder::new("doclayout_standalone");
    let input = b.add_input("image_features", &[SEQ, HIDDEN]);

    // Conv-like projection (Linear approximation) -> ReLU
    let conv_w = b.add_input("conv_w", &[FFN_DIM, HIDDEN]);
    let conv_out = b.add_linear(input, conv_w, None, &[SEQ, FFN_DIM]);
    let conv_act = b.add_relu(conv_out, &[SEQ, FFN_DIM]);

    // Classification head -> sigmoid
    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, FFN_DIM]);
    let cls_b = b.add_input("cls_b", &[NUM_CLASSES]);
    let cls_logits = b.add_linear(conv_act, cls_w, Some(cls_b), &[SEQ, NUM_CLASSES]);
    let cls_out = b.add_sigmoid(cls_logits, &[SEQ, NUM_CLASSES]);

    // Bbox head -> sigmoid (normalized coords)
    let box_w = b.add_input("box_w", &[4, FFN_DIM]);
    let box_b = b.add_input("box_b", &[4]);
    let box_logits = b.add_linear(conv_act, box_w, Some(box_b), &[SEQ, 4]);
    let box_out = b.add_sigmoid(box_logits, &[SEQ, 4]);

    // Merge cls + box via projection to unified output
    let merge_cls_w = b.add_input("merge_cls_w", &[HIDDEN, NUM_CLASSES]);
    let merge_cls = b.add_linear(cls_out, merge_cls_w, None, &[SEQ, HIDDEN]);
    let merge_box_w = b.add_input("merge_box_w", &[HIDDEN, 4]);
    let merge_box = b.add_linear(box_out, merge_box_w, None, &[SEQ, HIDDEN]);
    let merged = b.add_binary_add(merge_cls, merge_box, &[SEQ, HIDDEN]);
    let out = b.add_sigmoid(merged, &[SEQ, HIDDEN]);
    let def = b.build(out).expect("valid doclayout kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN]),
        weight(&[NUM_CLASSES, FFN_DIM]),
        bias_zero(&[NUM_CLASSES]),
        weight(&[4, FFN_DIM]),
        bias_zero(&[4]),
        weight(&[HIDDEN, NUM_CLASSES]),
        weight(&[HIDDEN, 4]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("doclayout standalone IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lo >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid hi <= 1, got {hi_max}");
}

// ===========================================================================
// 2. Table Transformer standalone (IBP + CROWN)
// ===========================================================================

/// Table Transformer: encoder FFN (Linear -> ReLU -> Linear) with residual,
/// then DETR-style sigmoid head for table cell classification.
#[test]
fn test_7model_ext_table_transformer_standalone_ibp_crown() {
    let mut b = TensorBlockBuilder::new("table_transformer_standalone");
    let input = b.add_input("encoder_features", &[SEQ, HIDDEN]);

    // Encoder FFN with residual
    let ffn_w1 = b.add_input("ffn_w1", &[FFN_DIM, HIDDEN]);
    let ffn_h = b.add_linear(input, ffn_w1, None, &[SEQ, FFN_DIM]);
    let ffn_act = b.add_relu(ffn_h, &[SEQ, FFN_DIM]);
    let ffn_w2 = b.add_input("ffn_w2", &[HIDDEN, FFN_DIM]);
    let ffn_out = b.add_linear(ffn_act, ffn_w2, None, &[SEQ, HIDDEN]);
    let residual = b.add_binary_add(input, ffn_out, &[SEQ, HIDDEN]);

    // Cell classification head -> sigmoid
    let cell_w = b.add_input("cell_w", &[NUM_CLASSES, HIDDEN]);
    let cell_logits = b.add_linear(residual, cell_w, None, &[SEQ, NUM_CLASSES]);
    let out = b.add_sigmoid(cell_logits, &[SEQ, NUM_CLASSES]);
    let def = b.build(out).expect("valid table transformer kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN]),
        weight(&[HIDDEN, FFN_DIM]),
        weight(&[NUM_CLASSES, HIDDEN]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("table transformer standalone IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5 && hi_max <= 1.0 + 1e-5);

    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("table transformer standalone CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 3. Granite-Docling standalone (IBP)
// ===========================================================================

/// Granite-Docling: vision projection (Linear) -> LM decoder FFN
/// (Linear -> ReLU -> Linear) -> softmax vocabulary distribution.
#[test]
fn test_7model_ext_granite_docling_standalone_ibp() {
    let mut b = TensorBlockBuilder::new("granite_docling_standalone");
    let input = b.add_input("vision_features", &[SEQ, PATCH_DIM]);

    let proj_w = b.add_input("proj_w", &[HIDDEN, PATCH_DIM]);
    let proj_b = b.add_input("proj_b", &[HIDDEN]);
    let projected = b.add_linear(input, proj_w, Some(proj_b), &[SEQ, HIDDEN]);

    let ffn_w1 = b.add_input("ffn_w1", &[FFN_DIM, HIDDEN]);
    let ffn_h = b.add_linear(projected, ffn_w1, None, &[SEQ, FFN_DIM]);
    let ffn_act = b.add_relu(ffn_h, &[SEQ, FFN_DIM]);
    let ffn_w2 = b.add_input("ffn_w2", &[HIDDEN, FFN_DIM]);
    let ffn_out = b.add_linear(ffn_act, ffn_w2, None, &[SEQ, HIDDEN]);

    let lm_w = b.add_input("lm_w", &[VOCAB, HIDDEN]);
    let lm_logits = b.add_linear(ffn_out, lm_w, None, &[SEQ, VOCAB]);
    let out = b.add_softmax(lm_logits, -1, &[SEQ, VOCAB]);
    let def = b.build(out).expect("valid granite docling kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN, PATCH_DIM]),
        bias_zero(&[HIDDEN]),
        weight(&[FFN_DIM, HIDDEN]),
        weight(&[HIDDEN, FFN_DIM]),
        weight(&[VOCAB, HIDDEN]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, PATCH_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("granite docling standalone IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lo >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "softmax hi <= 1, got {hi_max}");
}

// ===========================================================================
// 4. PaddleOCR-VL standalone (IBP)
// ===========================================================================

/// PaddleOCR-VL: DB text detector features -> SVTR recognizer head.
/// Linear -> ReLU -> Linear -> softmax character distribution.
#[test]
fn test_7model_ext_paddleocr_vl_standalone_ibp() {
    let mut b = TensorBlockBuilder::new("paddleocr_standalone");
    let input = b.add_input("text_det_features", &[SEQ, HIDDEN]);

    let svtr_w1 = b.add_input("svtr_w1", &[FFN_DIM, HIDDEN]);
    let svtr_b1 = b.add_input("svtr_b1", &[FFN_DIM]);
    let svtr_h = b.add_linear(input, svtr_w1, Some(svtr_b1), &[SEQ, FFN_DIM]);
    let svtr_act = b.add_relu(svtr_h, &[SEQ, FFN_DIM]);

    let char_w = b.add_input("char_w", &[VOCAB, FFN_DIM]);
    let char_b = b.add_input("char_b", &[VOCAB]);
    let char_logits = b.add_linear(svtr_act, char_w, Some(char_b), &[SEQ, VOCAB]);
    let out = b.add_softmax(char_logits, -1, &[SEQ, VOCAB]);
    let def = b.build(out).expect("valid paddleocr kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN]),
        bias_zero(&[FFN_DIM]),
        weight(&[VOCAB, FFN_DIM]),
        bias_zero(&[VOCAB]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("paddleocr standalone IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5 && hi_max <= 1.0 + 1e-5);
}

// ===========================================================================
// 5. FireRed-OCR standalone (IBP + CROWN)
// ===========================================================================

/// FireRed-OCR: CTC-style decoder. Linear -> ReLU -> Linear -> softmax.
/// Qwen3-VL-2B variant specialized for document OCR with CTC decoding.
#[test]
fn test_7model_ext_firered_ocr_standalone_ibp_crown() {
    let mut b = TensorBlockBuilder::new("firered_standalone");
    let input = b.add_input("encoder_features", &[SEQ, HIDDEN]);

    let ctc_w1 = b.add_input("ctc_w1", &[FFN_DIM, HIDDEN]);
    let ctc_h = b.add_linear(input, ctc_w1, None, &[SEQ, FFN_DIM]);
    let ctc_act = b.add_relu(ctc_h, &[SEQ, FFN_DIM]);
    let ctc_w2 = b.add_input("ctc_w2", &[VOCAB, FFN_DIM]);
    let ctc_b2 = b.add_input("ctc_b2", &[VOCAB]);
    let ctc_logits = b.add_linear(ctc_act, ctc_w2, Some(ctc_b2), &[SEQ, VOCAB]);
    let out = b.add_softmax(ctc_logits, -1, &[SEQ, VOCAB]);
    let def = b.build(out).expect("valid firered kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN]),
        weight(&[VOCAB, FFN_DIM]),
        bias_zero(&[VOCAB]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("firered standalone IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5 && hi_max <= 1.0 + 1e-5);

    let (method, crown_out, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_out);
    eprintln!("firered standalone CROWN ({method:?}): [{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 6. GLM-OCR standalone (IBP)
// ===========================================================================

/// GLM-OCR: FFN with SwiGLU-style gating (approximated as Linear -> sigmoid
/// gate -> mul -> Linear) then LM head -> softmax. GLM-4V style.
#[test]
fn test_7model_ext_glm_ocr_standalone_ibp() {
    let mut b = TensorBlockBuilder::new("glm_ocr_standalone");
    let input = b.add_input("text_features", &[SEQ, HIDDEN]);

    // SwiGLU-style FFN: gate path + value path
    let gate_w = b.add_input("gate_w", &[FFN_DIM, HIDDEN]);
    let gate_h = b.add_linear(input, gate_w, None, &[SEQ, FFN_DIM]);
    let gate_act = b.add_sigmoid(gate_h, &[SEQ, FFN_DIM]);

    let up_w = b.add_input("up_w", &[FFN_DIM, HIDDEN]);
    let up_h = b.add_linear(input, up_w, None, &[SEQ, FFN_DIM]);
    let gated = b.add_binary_mul(gate_act, up_h, &[SEQ, FFN_DIM]);

    let down_w = b.add_input("down_w", &[HIDDEN, FFN_DIM]);
    let down_out = b.add_linear(gated, down_w, None, &[SEQ, HIDDEN]);

    // Residual + LM head
    let residual = b.add_binary_add(input, down_out, &[SEQ, HIDDEN]);
    let lm_w = b.add_input("lm_w", &[VOCAB, HIDDEN]);
    let lm_logits = b.add_linear(residual, lm_w, None, &[SEQ, VOCAB]);
    let out = b.add_softmax(lm_logits, -1, &[SEQ, VOCAB]);
    let def = b.build(out).expect("valid glm ocr kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN]),
        weight(&[FFN_DIM, HIDDEN]),
        weight(&[HIDDEN, FFN_DIM]),
        weight(&[VOCAB, HIDDEN]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("glm ocr standalone IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5 && hi_max <= 1.0 + 1e-5);
}

// ===========================================================================
// 7. Qwen3-VL standalone (IBP)
// ===========================================================================

/// Qwen3-VL: vision MLP projection (GELU) -> decoder FFN -> sigmoid.
/// The vision encoder projects patch features via 2-layer MLP into the
/// decoder's embedding space.
#[test]
fn test_7model_ext_qwen3_vl_standalone_ibp() {
    let mut b = TensorBlockBuilder::new("qwen3_vl_standalone");
    let input = b.add_input("patch_features", &[SEQ, PATCH_DIM]);

    // 2-layer MLP vision projection with GELU
    let mlp_w1 = b.add_input("mlp_w1", &[HIDDEN, PATCH_DIM]);
    let mlp_b1 = b.add_input("mlp_b1", &[HIDDEN]);
    let mlp_h = b.add_linear(input, mlp_w1, Some(mlp_b1), &[SEQ, HIDDEN]);
    let mlp_act = b.add_gelu(mlp_h, &[SEQ, HIDDEN]);
    let mlp_w2 = b.add_input("mlp_w2", &[HIDDEN, HIDDEN]);
    let mlp_out = b.add_linear(mlp_act, mlp_w2, None, &[SEQ, HIDDEN]);

    // Decoder FFN
    let ffn_w1 = b.add_input("ffn_w1", &[FFN_DIM, HIDDEN]);
    let ffn_h = b.add_linear(mlp_out, ffn_w1, None, &[SEQ, FFN_DIM]);
    let ffn_act = b.add_relu(ffn_h, &[SEQ, FFN_DIM]);
    let ffn_w2 = b.add_input("ffn_w2", &[HIDDEN, FFN_DIM]);
    let ffn_out = b.add_linear(ffn_act, ffn_w2, None, &[SEQ, HIDDEN]);

    // Output confidence
    let out_w = b.add_input("out_w", &[NUM_CLASSES, HIDDEN]);
    let out_logits = b.add_linear(ffn_out, out_w, None, &[SEQ, NUM_CLASSES]);
    let out = b.add_sigmoid(out_logits, &[SEQ, NUM_CLASSES]);
    let def = b.build(out).expect("valid qwen3 vl kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN, PATCH_DIM]),
        bias_zero(&[HIDDEN]),
        weight(&[HIDDEN, HIDDEN]),
        weight(&[FFN_DIM, HIDDEN]),
        weight(&[HIDDEN, FFN_DIM]),
        weight(&[NUM_CLASSES, HIDDEN]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, PATCH_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("qwen3 vl standalone IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5 && hi_max <= 1.0 + 1e-5);
}
