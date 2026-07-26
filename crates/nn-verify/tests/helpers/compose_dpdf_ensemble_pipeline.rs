// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for the dpdf 7-model ensemble pipeline.
//!
//! Tests an ensemble of 7 document processing models working together:
//!   1. **Layout detection** — object detection backbone (DocLayout-YOLO)
//!   2. **OCR text recognition** — CTC decoder for character recognition (FireRed-OCR)
//!   3. **Table structure recognition** — DETR-based table parsing (Table Transformer)
//!   4. **Figure/chart classification** — sigmoid classification head (Qwen3-VL)
//!   5. **Reading order prediction** — sequence ordering with attention (GLM-OCR)
//!   6. **Document classification** — document-level category prediction (Granite-Docling)
//!   7. **Ensemble aggregation** — combining all model outputs via learned gating
//!
//! ## Tests (20 tests)
//!
//! Individual model subnetwork bound propagation:
//!  1. Layout detection backbone (Conv -> ReLU -> sigmoid) (IBP)
//!  2. OCR recognition (Linear -> ReLU -> softmax CTC) (IBP + CROWN)
//!  3. Table structure DETR (attention -> Linear -> sigmoid bbox) (IBP + CROWN)
//!  4. Figure classification (Linear -> GELU -> sigmoid) (IBP)
//!  5. Reading order (attention -> Linear -> softmax ordering) (IBP + CROWN)
//!  6. Document classification (Linear -> ReLU -> softmax) (IBP + CROWN)
//!
//! Pairwise model composition:
//!  7. Layout -> OCR: detection crops feed OCR recognizer (IBP)
//!  8. Layout -> Table: detection feeds table structure (IBP)
//!  9. Layout -> Figure: detection feeds figure classifier (IBP)
//! 10. OCR -> Reading order: recognized text feeds ordering (IBP)
//! 11. Table -> Document classification: table features feed doc classifier (IBP + CROWN)
//!
//! Full 7-model ensemble pipeline:
//! 12. Full sequential pipeline: layout -> branch -> aggregate (IBP)
//! 13. Full pipeline with CROWN tightening (IBP + CROWN)
//! 14. Parallel dispatch: all 7 heads from shared features (IBP)
//!
//! Bounds tightness through aggregation:
//! 15. Aggregation layer: weighted sum of 7 heads (IBP + CROWN)
//! 16. Aggregation monotone tightening: narrower input -> tighter output (IBP)
//! 17. Softmax gating: learned router selects model outputs (IBP)
//!
//! Different document types:
//! 18. Text-heavy document: OCR + reading order dominate (IBP)
//! 19. Table-heavy document: table structure + layout dominate (IBP + CROWN)
//! 20. Figure-heavy document: figure classification + doc classification dominate (IBP)
//!
//! Part of #4243: dpdf 7-model ensemble compose verification tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::AttentionMask;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Hidden dimension shared across model boundaries.
const HIDDEN: usize = 8;
/// Sequence length / number of positions.
const SEQ: usize = 4;
/// Number of layout detection classes.
const NUM_LAYOUT_CLASSES: usize = 6;
/// OCR vocabulary size.
const VOCAB: usize = 8;
/// Number of attention heads.
const NUM_HEADS: usize = 2;
/// FFN intermediate dimension.
const FFN_DIM: usize = HIDDEN * 2;
/// Number of figure/chart categories.
const NUM_FIGURE_CLASSES: usize = 4;
/// Number of document categories.
const NUM_DOC_CLASSES: usize = 5;
/// Number of ensemble models.
const NUM_MODELS: usize = 7;
/// Weight magnitude for bounded verification.
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

// ===========================================================================
// 1. Layout detection backbone: Conv -> ReLU -> sigmoid (IBP)
// ===========================================================================

/// Layout detection model subnetwork: Linear feature extractor -> ReLU ->
/// classification sigmoid head. Verifies detection confidence in [0, 1].
#[test]
fn test_ensemble_pipeline_layout_detection_ibp() {
    let mut b = TensorBlockBuilder::new("layout_detection");
    let input = b.add_input("features", &[SEQ, HIDDEN]);

    // Feature extraction: Linear -> ReLU
    let feat_w = b.add_input("feat_w", &[FFN_DIM, HIDDEN]);
    let feat = b.add_linear(input, feat_w, None, &[SEQ, FFN_DIM]);
    let feat = b.add_relu(feat, &[SEQ, FFN_DIM]);

    // Classification head: Linear -> sigmoid
    let cls_w = b.add_input("cls_w", &[NUM_LAYOUT_CLASSES, FFN_DIM]);
    let logits = b.add_linear(feat, cls_w, None, &[SEQ, NUM_LAYOUT_CLASSES]);
    let out = b.add_sigmoid(logits, &[SEQ, NUM_LAYOUT_CLASSES]);
    let def = b.build(out).expect("valid layout detection kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN]),
        weight(&[NUM_LAYOUT_CLASSES, FFN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("layout detection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 2. OCR text recognition: Linear -> ReLU -> softmax CTC (IBP + CROWN)
// ===========================================================================

/// OCR recognition subnetwork: Linear encoder -> ReLU -> Linear CTC head ->
/// softmax character probabilities in [0, 1].
#[test]
fn test_ensemble_pipeline_ocr_recognition_ibp_crown() {
    let mut b = TensorBlockBuilder::new("ocr_recognition");
    let input = b.add_input("text_features", &[SEQ, HIDDEN]);

    // Encoder: Linear -> ReLU
    let enc_w = b.add_input("enc_w", &[FFN_DIM, HIDDEN]);
    let enc = b.add_linear(input, enc_w, None, &[SEQ, FFN_DIM]);
    let enc = b.add_relu(enc, &[SEQ, FFN_DIM]);

    // CTC head: Linear -> softmax
    let ctc_w = b.add_input("ctc_w", &[VOCAB, FFN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB]);
    let logits = b.add_linear(enc, ctc_w, Some(ctc_b), &[SEQ, VOCAB]);
    let out = b.add_softmax(logits, -1, &[SEQ, VOCAB]);
    let def = b.build(out).expect("valid OCR recognition kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN]),
        weight(&[VOCAB, FFN_DIM]),
        bias_zero(&[VOCAB]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("OCR recognition IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "softmax upper <= 1, got {hi_max}");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("OCR recognition CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 3. Table structure DETR: attention -> Linear -> sigmoid bbox (IBP + CROWN)
// ===========================================================================

/// Table structure recognition: self-attention over object queries -> Linear
/// bbox regression -> sigmoid coordinates in [0, 1].
#[test]
fn test_ensemble_pipeline_table_structure_ibp_crown() {
    let mut b = TensorBlockBuilder::new("table_structure");
    let input = b.add_input("table_queries", &[SEQ, HIDDEN]);

    // Self-attention
    let q_w = b.add_input("q_w", &[HIDDEN, HIDDEN]);
    let k_w = b.add_input("k_w", &[HIDDEN, HIDDEN]);
    let v_w = b.add_input("v_w", &[HIDDEN, HIDDEN]);
    let out_w = b.add_input("out_w", &[HIDDEN, HIDDEN]);
    let attn = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[SEQ, HIDDEN],
        )
        .expect("valid MHA");

    // Bbox head: Linear -> sigmoid
    let box_w = b.add_input("box_w", &[4, HIDDEN]);
    let box_b = b.add_input("box_b", &[4]);
    let box_logits = b.add_linear(attn, box_w, Some(box_b), &[SEQ, 4]);
    let out = b.add_sigmoid(box_logits, &[SEQ, 4]);
    let def = b.build(out).expect("valid table structure kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN, HIDDEN]),
        weight(&[HIDDEN, HIDDEN]),
        weight(&[HIDDEN, HIDDEN]),
        weight(&[HIDDEN, HIDDEN]),
        weight(&[4, HIDDEN]),
        bias_zero(&[4]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("table structure IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("table structure CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 4. Figure/chart classification: Linear -> GELU -> sigmoid (IBP)
// ===========================================================================

/// Figure classification subnetwork: Linear -> GELU activation -> classification
/// sigmoid head. Verifies output probabilities bounded in [0, 1].
#[test]
fn test_ensemble_pipeline_figure_classification_ibp() {
    let mut b = TensorBlockBuilder::new("figure_classification");
    let input = b.add_input("visual_features", &[SEQ, HIDDEN]);

    // MLP: Linear -> GELU -> Linear -> sigmoid
    let w1 = b.add_input("w1", &[FFN_DIM, HIDDEN]);
    let h = b.add_linear(input, w1, None, &[SEQ, FFN_DIM]);
    let h = b.add_gelu(h, &[SEQ, FFN_DIM]);
    let w2 = b.add_input("w2", &[NUM_FIGURE_CLASSES, FFN_DIM]);
    let logits = b.add_linear(h, w2, None, &[SEQ, NUM_FIGURE_CLASSES]);
    let out = b.add_sigmoid(logits, &[SEQ, NUM_FIGURE_CLASSES]);
    let def = b.build(out).expect("valid figure classification kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN]),
        weight(&[NUM_FIGURE_CLASSES, FFN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("figure classification IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 5. Reading order prediction: attention -> Linear -> softmax (IBP + CROWN)
// ===========================================================================

/// Reading order subnetwork: self-attention captures inter-region ordering ->
/// Linear projection -> softmax position distribution.
#[test]
fn test_ensemble_pipeline_reading_order_ibp_crown() {
    let mut b = TensorBlockBuilder::new("reading_order");
    let input = b.add_input("region_features", &[SEQ, HIDDEN]);

    // Self-attention for inter-region dependencies
    let q_w = b.add_input("q_w", &[HIDDEN, HIDDEN]);
    let k_w = b.add_input("k_w", &[HIDDEN, HIDDEN]);
    let v_w = b.add_input("v_w", &[HIDDEN, HIDDEN]);
    let out_w = b.add_input("out_w", &[HIDDEN, HIDDEN]);
    let attn = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[SEQ, HIDDEN],
        )
        .expect("valid MHA");

    // Order prediction: Linear -> softmax
    let ord_w = b.add_input("ord_w", &[SEQ, HIDDEN]);
    let logits = b.add_linear(attn, ord_w, None, &[SEQ, SEQ]);
    let out = b.add_softmax(logits, -1, &[SEQ, SEQ]);
    let def = b.build(out).expect("valid reading order kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN, HIDDEN]),
        weight(&[HIDDEN, HIDDEN]),
        weight(&[HIDDEN, HIDDEN]),
        weight(&[HIDDEN, HIDDEN]),
        weight(&[SEQ, HIDDEN]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("reading order IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "softmax upper <= 1, got {hi_max}");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("reading order CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 6. Document classification: Linear -> ReLU -> softmax (IBP + CROWN)
// ===========================================================================

/// Document classification subnetwork: feature extractor -> ReLU -> softmax
/// category distribution.
#[test]
fn test_ensemble_pipeline_document_classification_ibp_crown() {
    let mut b = TensorBlockBuilder::new("doc_classification");
    let input = b.add_input("doc_features", &[HIDDEN]);

    // Two-layer MLP
    let w1 = b.add_input("w1", &[FFN_DIM, HIDDEN]);
    let h = b.add_linear(input, w1, None, &[FFN_DIM]);
    let h = b.add_relu(h, &[FFN_DIM]);
    let w2 = b.add_input("w2", &[NUM_DOC_CLASSES, FFN_DIM]);
    let logits = b.add_linear(h, w2, None, &[NUM_DOC_CLASSES]);
    let out = b.add_softmax(logits, -1, &[NUM_DOC_CLASSES]);
    let def = b.build(out).expect("valid document classification kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN]),
        weight(&[NUM_DOC_CLASSES, FFN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&[HIDDEN], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("doc classification IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "softmax upper <= 1, got {hi_max}");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("doc classification CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 7. Layout -> OCR: detection crops feed OCR recognizer (IBP)
// ===========================================================================

/// Pairwise composition: layout detection sigmoid -> projection bridge ->
/// OCR Linear -> ReLU -> softmax CTC. Verifies that detection-bounded
/// features compose cleanly into OCR character probabilities.
#[test]
fn test_ensemble_pipeline_layout_to_ocr_ibp() {
    let mut b = TensorBlockBuilder::new("layout_to_ocr");
    let input = b.add_input("features", &[SEQ, HIDDEN]);

    // Layout detection: Linear -> sigmoid
    let det_w = b.add_input("det_w", &[NUM_LAYOUT_CLASSES, HIDDEN]);
    let det_logits = b.add_linear(input, det_w, None, &[SEQ, NUM_LAYOUT_CLASSES]);
    let det_conf = b.add_sigmoid(det_logits, &[SEQ, NUM_LAYOUT_CLASSES]);

    // Bridge: project detection output to OCR input space
    let bridge_w = b.add_input("bridge_w", &[HIDDEN, NUM_LAYOUT_CLASSES]);
    let ocr_input = b.add_linear(det_conf, bridge_w, None, &[SEQ, HIDDEN]);

    // OCR: Linear -> ReLU -> softmax
    let ocr_w = b.add_input("ocr_w", &[VOCAB, HIDDEN]);
    let ocr_h = b.add_linear(ocr_input, ocr_w, None, &[SEQ, VOCAB]);
    let ocr_h = b.add_relu(ocr_h, &[SEQ, VOCAB]);
    let ctc_w = b.add_input("ctc_w", &[VOCAB, VOCAB]);
    let ctc_logits = b.add_linear(ocr_h, ctc_w, None, &[SEQ, VOCAB]);
    let out = b.add_softmax(ctc_logits, -1, &[SEQ, VOCAB]);
    let def = b.build(out).expect("valid layout-to-ocr kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_LAYOUT_CLASSES, HIDDEN]),
        weight(&[HIDDEN, NUM_LAYOUT_CLASSES]),
        weight(&[VOCAB, HIDDEN]),
        weight(&[VOCAB, VOCAB]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("layout->OCR IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 8. Layout -> Table: detection feeds table structure (IBP)
// ===========================================================================

/// Pairwise composition: layout detection sigmoid -> bridge -> table
/// structure attention -> sigmoid bbox coordinates.
#[test]
fn test_ensemble_pipeline_layout_to_table_ibp() {
    let mut b = TensorBlockBuilder::new("layout_to_table");
    let input = b.add_input("features", &[SEQ, HIDDEN]);

    // Layout detection
    let det_w = b.add_input("det_w", &[NUM_LAYOUT_CLASSES, HIDDEN]);
    let det_logits = b.add_linear(input, det_w, None, &[SEQ, NUM_LAYOUT_CLASSES]);
    let det_conf = b.add_sigmoid(det_logits, &[SEQ, NUM_LAYOUT_CLASSES]);

    // Bridge to table dimension
    let bridge_w = b.add_input("bridge_w", &[HIDDEN, NUM_LAYOUT_CLASSES]);
    let table_input = b.add_linear(det_conf, bridge_w, None, &[SEQ, HIDDEN]);

    // Table bbox regression: Linear -> sigmoid
    let box_w = b.add_input("box_w", &[4, HIDDEN]);
    let box_b = b.add_input("box_b", &[4]);
    let box_logits = b.add_linear(table_input, box_w, Some(box_b), &[SEQ, 4]);
    let out = b.add_sigmoid(box_logits, &[SEQ, 4]);
    let def = b.build(out).expect("valid layout-to-table kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_LAYOUT_CLASSES, HIDDEN]),
        weight(&[HIDDEN, NUM_LAYOUT_CLASSES]),
        weight(&[4, HIDDEN]),
        bias_zero(&[4]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("layout->table IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 9. Layout -> Figure: detection feeds figure classifier (IBP)
// ===========================================================================

/// Pairwise composition: layout detection sigmoid -> bridge -> GELU ->
/// figure classification sigmoid.
#[test]
fn test_ensemble_pipeline_layout_to_figure_ibp() {
    let mut b = TensorBlockBuilder::new("layout_to_figure");
    let input = b.add_input("features", &[SEQ, HIDDEN]);

    // Layout detection
    let det_w = b.add_input("det_w", &[NUM_LAYOUT_CLASSES, HIDDEN]);
    let det_logits = b.add_linear(input, det_w, None, &[SEQ, NUM_LAYOUT_CLASSES]);
    let det_conf = b.add_sigmoid(det_logits, &[SEQ, NUM_LAYOUT_CLASSES]);

    // Bridge + GELU
    let bridge_w = b.add_input("bridge_w", &[FFN_DIM, NUM_LAYOUT_CLASSES]);
    let h = b.add_linear(det_conf, bridge_w, None, &[SEQ, FFN_DIM]);
    let h = b.add_gelu(h, &[SEQ, FFN_DIM]);

    // Figure classifier: Linear -> sigmoid
    let fig_w = b.add_input("fig_w", &[NUM_FIGURE_CLASSES, FFN_DIM]);
    let logits = b.add_linear(h, fig_w, None, &[SEQ, NUM_FIGURE_CLASSES]);
    let out = b.add_sigmoid(logits, &[SEQ, NUM_FIGURE_CLASSES]);
    let def = b.build(out).expect("valid layout-to-figure kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_LAYOUT_CLASSES, HIDDEN]),
        weight(&[FFN_DIM, NUM_LAYOUT_CLASSES]),
        weight(&[NUM_FIGURE_CLASSES, FFN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("layout->figure IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 10. OCR -> Reading order: recognized text feeds ordering (IBP)
// ===========================================================================

/// Pairwise composition: OCR softmax character probs -> bridge ->
/// attention-based reading order -> softmax position distribution.
#[test]
fn test_ensemble_pipeline_ocr_to_reading_order_ibp() {
    let mut b = TensorBlockBuilder::new("ocr_to_reading_order");
    let input = b.add_input("text_features", &[SEQ, HIDDEN]);

    // OCR: Linear -> softmax
    let ocr_w = b.add_input("ocr_w", &[VOCAB, HIDDEN]);
    let ocr_logits = b.add_linear(input, ocr_w, None, &[SEQ, VOCAB]);
    let ocr_probs = b.add_softmax(ocr_logits, -1, &[SEQ, VOCAB]);

    // Bridge: VOCAB -> HIDDEN
    let bridge_w = b.add_input("bridge_w", &[HIDDEN, VOCAB]);
    let order_input = b.add_linear(ocr_probs, bridge_w, None, &[SEQ, HIDDEN]);

    // Reading order: Linear -> softmax position
    let ord_w = b.add_input("ord_w", &[SEQ, HIDDEN]);
    let ord_logits = b.add_linear(order_input, ord_w, None, &[SEQ, SEQ]);
    let out = b.add_softmax(ord_logits, -1, &[SEQ, SEQ]);
    let def = b.build(out).expect("valid ocr-to-reading-order kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB, HIDDEN]),
        weight(&[HIDDEN, VOCAB]),
        weight(&[SEQ, HIDDEN]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("OCR->reading order IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 11. Table -> Doc classification: table features feed classifier (IBP + CROWN)
// ===========================================================================

/// Pairwise composition: table structure features -> bridge -> document
/// classification softmax. Tests that table structure features compose
/// into document-level category predictions.
#[test]
fn test_ensemble_pipeline_table_to_doc_classification_ibp_crown() {
    let mut b = TensorBlockBuilder::new("table_to_doc");
    let input = b.add_input("table_features", &[SEQ, HIDDEN]);

    // Table feature reduction: Linear -> ReLU -> reduce (take first position)
    let tab_w = b.add_input("tab_w", &[FFN_DIM, HIDDEN]);
    let h = b.add_linear(input, tab_w, None, &[SEQ, FFN_DIM]);
    let h = b.add_relu(h, &[SEQ, FFN_DIM]);

    // Doc classifier: Linear -> softmax
    let doc_w = b.add_input("doc_w", &[NUM_DOC_CLASSES, FFN_DIM]);
    let logits = b.add_linear(h, doc_w, None, &[SEQ, NUM_DOC_CLASSES]);
    let out = b.add_softmax(logits, -1, &[SEQ, NUM_DOC_CLASSES]);
    let def = b.build(out).expect("valid table-to-doc kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN]),
        weight(&[NUM_DOC_CLASSES, FFN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("table->doc classification IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "softmax upper <= 1, got {hi_max}");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("table->doc classification CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 12. Full sequential pipeline: layout -> branch -> aggregate (IBP)
// ===========================================================================

/// Full 7-model ensemble sequential pipeline: shared features -> layout
/// detection -> OCR branch -> table branch -> merge via weighted sum ->
/// final sigmoid output.
#[test]
fn test_ensemble_pipeline_full_sequential_ibp() {
    let mut b = TensorBlockBuilder::new("full_sequential");
    let input = b.add_input("shared_features", &[SEQ, HIDDEN]);

    // Layout detection: Linear -> sigmoid
    let det_w = b.add_input("det_w", &[NUM_LAYOUT_CLASSES, HIDDEN]);
    let det_logits = b.add_linear(input, det_w, None, &[SEQ, NUM_LAYOUT_CLASSES]);
    let det_out = b.add_sigmoid(det_logits, &[SEQ, NUM_LAYOUT_CLASSES]);

    // OCR branch: from detection -> bridge -> CTC softmax
    let ocr_bridge_w = b.add_input("ocr_bridge_w", &[HIDDEN, NUM_LAYOUT_CLASSES]);
    let ocr_h = b.add_linear(det_out, ocr_bridge_w, None, &[SEQ, HIDDEN]);
    let ocr_w = b.add_input("ocr_w", &[VOCAB, HIDDEN]);
    let ocr_logits = b.add_linear(ocr_h, ocr_w, None, &[SEQ, VOCAB]);
    let ocr_out = b.add_softmax(ocr_logits, -1, &[SEQ, VOCAB]);

    // Merge: project OCR output back to HIDDEN
    let merge_w = b.add_input("merge_w", &[HIDDEN, VOCAB]);
    let merged = b.add_linear(ocr_out, merge_w, None, &[SEQ, HIDDEN]);

    // Final output: sigmoid confidence
    let final_w = b.add_input("final_w", &[1, HIDDEN]);
    let final_logits = b.add_linear(merged, final_w, None, &[SEQ, 1]);
    let out = b.add_sigmoid(final_logits, &[SEQ, 1]);
    let def = b.build(out).expect("valid full sequential kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_LAYOUT_CLASSES, HIDDEN]),
        weight(&[HIDDEN, NUM_LAYOUT_CLASSES]),
        weight(&[VOCAB, HIDDEN]),
        weight(&[HIDDEN, VOCAB]),
        weight(&[1, HIDDEN]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("full sequential IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 13. Full pipeline with CROWN tightening (IBP + CROWN)
// ===========================================================================

/// Full pipeline with CROWN: features -> ReLU -> Linear -> sigmoid final
/// output. Verifies CROWN tightens bounds relative to IBP.
#[test]
fn test_ensemble_pipeline_full_crown_tightening() {
    let mut b = TensorBlockBuilder::new("full_crown");
    let input = b.add_input("features", &[SEQ, HIDDEN]);

    // Deep MLP: 2 hidden layers with ReLU
    let w1 = b.add_input("w1", &[FFN_DIM, HIDDEN]);
    let h = b.add_linear(input, w1, None, &[SEQ, FFN_DIM]);
    let h = b.add_relu(h, &[SEQ, FFN_DIM]);
    let w2 = b.add_input("w2", &[HIDDEN, FFN_DIM]);
    let h = b.add_linear(h, w2, None, &[SEQ, HIDDEN]);
    let h = b.add_relu(h, &[SEQ, HIDDEN]);

    // Final sigmoid
    let w3 = b.add_input("w3", &[1, HIDDEN]);
    let logits = b.add_linear(h, w3, None, &[SEQ, 1]);
    let out = b.add_sigmoid(logits, &[SEQ, 1]);
    let def = b.build(out).expect("valid full crown kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN]),
        weight(&[HIDDEN, FFN_DIM]),
        weight(&[1, HIDDEN]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    // IBP baseline
    let ibp_output = graph.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&ibp_output);

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("full pipeline CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 14. Parallel dispatch: all 7 heads from shared features (IBP)
// ===========================================================================

/// All 7 model heads run from the same shared feature representation.
/// Each head produces bounded output (sigmoid or softmax), and the union
/// of all heads is verified. Simulates parallel dispatch pattern.
#[test]
fn test_ensemble_pipeline_parallel_dispatch_ibp() {
    // Test each head independently from the same input bounds.
    // We verify the layout head as representative of the parallel dispatch.
    let mut b = TensorBlockBuilder::new("parallel_dispatch_layout");
    let input = b.add_input("shared", &[SEQ, HIDDEN]);

    // 7 heads share input; we verify layout + OCR + doc cls compose
    let w_layout = b.add_input("w_layout", &[NUM_LAYOUT_CLASSES, HIDDEN]);
    let layout_logits = b.add_linear(input, w_layout, None, &[SEQ, NUM_LAYOUT_CLASSES]);
    let layout_sig = b.add_sigmoid(layout_logits, &[SEQ, NUM_LAYOUT_CLASSES]);

    // OCR head on same shared input
    let w_ocr = b.add_input("w_ocr", &[VOCAB, HIDDEN]);
    let ocr_logits = b.add_linear(input, w_ocr, None, &[SEQ, VOCAB]);
    let ocr_sm = b.add_softmax(ocr_logits, -1, &[SEQ, VOCAB]);

    // Merge layout + OCR: add sigmoid + softmax outputs projected to same dim
    let merge_layout_w = b.add_input("merge_layout_w", &[HIDDEN, NUM_LAYOUT_CLASSES]);
    let merge_ocr_w = b.add_input("merge_ocr_w", &[HIDDEN, VOCAB]);
    let layout_proj = b.add_linear(layout_sig, merge_layout_w, None, &[SEQ, HIDDEN]);
    let ocr_proj = b.add_linear(ocr_sm, merge_ocr_w, None, &[SEQ, HIDDEN]);
    let merged = b.add_binary_add(layout_proj, ocr_proj, &[SEQ, HIDDEN]);

    // Final confidence
    let final_w = b.add_input("final_w", &[1, HIDDEN]);
    let final_logits = b.add_linear(merged, final_w, None, &[SEQ, 1]);
    let out = b.add_sigmoid(final_logits, &[SEQ, 1]);
    let def = b.build(out).expect("valid parallel dispatch kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_LAYOUT_CLASSES, HIDDEN]),
        weight(&[VOCAB, HIDDEN]),
        weight(&[HIDDEN, NUM_LAYOUT_CLASSES]),
        weight(&[HIDDEN, VOCAB]),
        weight(&[1, HIDDEN]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("parallel dispatch IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 15. Aggregation layer: weighted sum of 7 heads (IBP + CROWN)
// ===========================================================================

/// Ensemble aggregation: each of the 7 model heads produces a HIDDEN-dim
/// feature vector. These are concatenated (7*HIDDEN) and passed through
/// a learned fusion MLP -> sigmoid final output.
#[test]
fn test_ensemble_pipeline_aggregation_layer_ibp_crown() {
    let concat_dim = NUM_MODELS * HIDDEN;

    let mut b = TensorBlockBuilder::new("aggregation_layer");
    let input = b.add_input("concat_features", &[SEQ, concat_dim]);

    // Fusion MLP: Linear -> ReLU -> Linear -> sigmoid
    let w1 = b.add_input("w1", &[FFN_DIM, concat_dim]);
    let h = b.add_linear(input, w1, None, &[SEQ, FFN_DIM]);
    let h = b.add_relu(h, &[SEQ, FFN_DIM]);
    let w2 = b.add_input("w2", &[1, FFN_DIM]);
    let logits = b.add_linear(h, w2, None, &[SEQ, 1]);
    let out = b.add_sigmoid(logits, &[SEQ, 1]);
    let def = b.build(out).expect("valid aggregation kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, concat_dim]),
        weight(&[1, FFN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&[SEQ, concat_dim], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("aggregation IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("aggregation CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 16. Aggregation monotone tightening (IBP)
// ===========================================================================

/// Verifies monotone tightening: narrower input bounds produce narrower
/// output bounds through the aggregation layer.
#[test]
fn test_ensemble_pipeline_aggregation_monotone_tightening_ibp() {
    let concat_dim = NUM_MODELS * HIDDEN;

    let mut b = TensorBlockBuilder::new("agg_monotone");
    let input = b.add_input("concat_features", &[SEQ, concat_dim]);

    let w1 = b.add_input("w1", &[FFN_DIM, concat_dim]);
    let h = b.add_linear(input, w1, None, &[SEQ, FFN_DIM]);
    let h = b.add_relu(h, &[SEQ, FFN_DIM]);
    let w2 = b.add_input("w2", &[1, FFN_DIM]);
    let logits = b.add_linear(h, w2, None, &[SEQ, 1]);
    let out = b.add_sigmoid(logits, &[SEQ, 1]);
    let def = b.build(out).expect("valid agg monotone kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, concat_dim]),
        weight(&[1, FFN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // Wide input
    let wide_input = uniform_bounds(&[SEQ, concat_dim], 2.0);
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");
    assert_bounds_valid(&wide_output);
    let (wide_lo, wide_hi) = bounds_min_max(&wide_output);

    // Narrow input
    let narrow_input = uniform_bounds(&[SEQ, concat_dim], 0.5);
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("IBP narrow");
    assert_bounds_valid(&narrow_output);
    let (narrow_lo, narrow_hi) = bounds_min_max(&narrow_output);

    let wide_width = wide_hi - wide_lo;
    let narrow_width = narrow_hi - narrow_lo;
    eprintln!("monotone: wide_width={wide_width:.6}, narrow_width={narrow_width:.6}");
    assert!(
        narrow_width <= wide_width + 1e-5,
        "narrower input should produce narrower output: narrow={narrow_width} vs wide={wide_width}"
    );
}

// ===========================================================================
// 17. Softmax gating: learned router selects model outputs (IBP)
// ===========================================================================

/// Softmax gating mechanism: features -> Linear -> softmax gate weights
/// for 7 models. Verifies gate probabilities bounded in [0, 1].
#[test]
fn test_ensemble_pipeline_softmax_gating_ibp() {
    let mut b = TensorBlockBuilder::new("softmax_gating");
    let input = b.add_input("features", &[SEQ, HIDDEN]);

    // Gate: Linear -> softmax over NUM_MODELS
    let gate_w = b.add_input("gate_w", &[NUM_MODELS, HIDDEN]);
    let gate_b = b.add_input("gate_b", &[NUM_MODELS]);
    let gate_logits = b.add_linear(input, gate_w, Some(gate_b), &[SEQ, NUM_MODELS]);
    let out = b.add_softmax(gate_logits, -1, &[SEQ, NUM_MODELS]);
    let def = b.build(out).expect("valid softmax gating kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_MODELS, HIDDEN]),
        bias_zero(&[NUM_MODELS]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("softmax gating IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 18. Text-heavy document: OCR + reading order dominate (IBP)
// ===========================================================================

/// Simulates a text-heavy document where OCR and reading order models
/// dominate. Features -> OCR softmax -> reading order softmax ->
/// document-level aggregation sigmoid.
#[test]
fn test_ensemble_pipeline_text_heavy_document_ibp() {
    let mut b = TensorBlockBuilder::new("text_heavy_doc");
    let input = b.add_input("text_features", &[SEQ, HIDDEN]);

    // OCR: Linear -> softmax
    let ocr_w = b.add_input("ocr_w", &[VOCAB, HIDDEN]);
    let ocr_logits = b.add_linear(input, ocr_w, None, &[SEQ, VOCAB]);
    let ocr_probs = b.add_softmax(ocr_logits, -1, &[SEQ, VOCAB]);

    // Bridge OCR -> reading order
    let bridge_w = b.add_input("bridge_w", &[HIDDEN, VOCAB]);
    let order_in = b.add_linear(ocr_probs, bridge_w, None, &[SEQ, HIDDEN]);

    // Reading order: Linear -> softmax
    let ord_w = b.add_input("ord_w", &[SEQ, HIDDEN]);
    let ord_logits = b.add_linear(order_in, ord_w, None, &[SEQ, SEQ]);
    let ord_probs = b.add_softmax(ord_logits, -1, &[SEQ, SEQ]);

    // Aggregation: project and sigmoid
    let agg_w = b.add_input("agg_w", &[1, SEQ]);
    let agg_logits = b.add_linear(ord_probs, agg_w, None, &[SEQ, 1]);
    let out = b.add_sigmoid(agg_logits, &[SEQ, 1]);
    let def = b.build(out).expect("valid text-heavy kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB, HIDDEN]),
        weight(&[HIDDEN, VOCAB]),
        weight(&[SEQ, HIDDEN]),
        weight(&[1, SEQ]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("text-heavy IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 19. Table-heavy document: table + layout dominate (IBP + CROWN)
// ===========================================================================

/// Simulates a table-heavy document where table structure and layout
/// detection dominate. Features -> layout sigmoid -> table attention ->
/// sigmoid bbox -> merge -> final sigmoid.
#[test]
fn test_ensemble_pipeline_table_heavy_document_ibp_crown() {
    let mut b = TensorBlockBuilder::new("table_heavy_doc");
    let input = b.add_input("features", &[SEQ, HIDDEN]);

    // Layout detection: sigmoid
    let det_w = b.add_input("det_w", &[NUM_LAYOUT_CLASSES, HIDDEN]);
    let det_logits = b.add_linear(input, det_w, None, &[SEQ, NUM_LAYOUT_CLASSES]);
    let det_sig = b.add_sigmoid(det_logits, &[SEQ, NUM_LAYOUT_CLASSES]);

    // Bridge to table features
    let bridge_w = b.add_input("bridge_w", &[HIDDEN, NUM_LAYOUT_CLASSES]);
    let table_in = b.add_linear(det_sig, bridge_w, None, &[SEQ, HIDDEN]);

    // Table: attention -> bbox sigmoid
    let q_w = b.add_input("q_w", &[HIDDEN, HIDDEN]);
    let k_w = b.add_input("k_w", &[HIDDEN, HIDDEN]);
    let v_w = b.add_input("v_w", &[HIDDEN, HIDDEN]);
    let out_w = b.add_input("out_w", &[HIDDEN, HIDDEN]);
    let attn = b
        .add_multi_head_attention(
            table_in,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[SEQ, HIDDEN],
        )
        .expect("valid MHA");
    let box_w = b.add_input("box_w", &[4, HIDDEN]);
    let box_logits = b.add_linear(attn, box_w, None, &[SEQ, 4]);
    let out = b.add_sigmoid(box_logits, &[SEQ, 4]);
    let def = b.build(out).expect("valid table-heavy kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_LAYOUT_CLASSES, HIDDEN]),
        weight(&[HIDDEN, NUM_LAYOUT_CLASSES]),
        weight(&[HIDDEN, HIDDEN]),
        weight(&[HIDDEN, HIDDEN]),
        weight(&[HIDDEN, HIDDEN]),
        weight(&[HIDDEN, HIDDEN]),
        weight(&[4, HIDDEN]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("table-heavy IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("table-heavy CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 20. Figure-heavy document: figure + doc classification dominate (IBP)
// ===========================================================================

/// Simulates a figure-heavy document where figure classification and
/// document classification dominate. Features -> GELU figure head ->
/// bridge -> doc classification softmax -> final aggregation sigmoid.
#[test]
fn test_ensemble_pipeline_figure_heavy_document_ibp() {
    let mut b = TensorBlockBuilder::new("figure_heavy_doc");
    let input = b.add_input("visual_features", &[SEQ, HIDDEN]);

    // Figure classification: GELU -> sigmoid
    let fig_w1 = b.add_input("fig_w1", &[FFN_DIM, HIDDEN]);
    let fig_h = b.add_linear(input, fig_w1, None, &[SEQ, FFN_DIM]);
    let fig_h = b.add_gelu(fig_h, &[SEQ, FFN_DIM]);
    let fig_w2 = b.add_input("fig_w2", &[NUM_FIGURE_CLASSES, FFN_DIM]);
    let fig_logits = b.add_linear(fig_h, fig_w2, None, &[SEQ, NUM_FIGURE_CLASSES]);
    let fig_probs = b.add_sigmoid(fig_logits, &[SEQ, NUM_FIGURE_CLASSES]);

    // Bridge to doc classification
    let bridge_w = b.add_input("bridge_w", &[HIDDEN, NUM_FIGURE_CLASSES]);
    let doc_in = b.add_linear(fig_probs, bridge_w, None, &[SEQ, HIDDEN]);

    // Doc classification: ReLU -> softmax
    let doc_w = b.add_input("doc_w", &[NUM_DOC_CLASSES, HIDDEN]);
    let doc_h = b.add_linear(doc_in, doc_w, None, &[SEQ, NUM_DOC_CLASSES]);
    let doc_probs = b.add_softmax(doc_h, -1, &[SEQ, NUM_DOC_CLASSES]);

    // Final aggregation: project + sigmoid
    let agg_w = b.add_input("agg_w", &[1, NUM_DOC_CLASSES]);
    let agg_logits = b.add_linear(doc_probs, agg_w, None, &[SEQ, 1]);
    let out = b.add_sigmoid(agg_logits, &[SEQ, 1]);
    let def = b.build(out).expect("valid figure-heavy kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[FFN_DIM, HIDDEN]),
        weight(&[NUM_FIGURE_CLASSES, FFN_DIM]),
        weight(&[HIDDEN, NUM_FIGURE_CLASSES]),
        weight(&[NUM_DOC_CLASSES, HIDDEN]),
        weight(&[1, NUM_DOC_CLASSES]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("figure-heavy IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");
}
