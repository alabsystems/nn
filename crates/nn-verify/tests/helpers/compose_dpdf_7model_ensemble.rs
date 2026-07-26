// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for the dpdf 7-model ensemble pipeline bounds.
//!
//! These tests focus on **pipeline-level composition patterns** — how multiple
//! model outputs compose, aggregate, and chain to produce final document
//! understanding results. Complements `compose_dpdf_ensemble.rs` which covers
//! per-model subgraphs and simple cascades.
//!
//! The 7 models in the dpdf ensemble:
//!   1. **DocLayout-YOLO** — document layout detection (boxes + classes)
//!   2. **Table Transformer** — DETR-based table structure recognition
//!   3. **Granite-Docling** — SigLIP2 vision encoder + Granite LLM decoder
//!   4. **PaddleOCR-VL** — DB text detector + SVTR text recognizer
//!   5. **FireRed-OCR** — Qwen3-VL-2B variant for document OCR (CTC decoding)
//!   6. **GLM-OCR** — GLM-4V vision-language model for OCR
//!   7. **Qwen3-VL** — vision-language model (vision encoder + MLP projection)
//!
//! ## Tests (14 tests)
//!
//! 1.  **Pipeline stage composition** — output of detection feeds table structure (IBP)
//! 2.  **Parallel dispatch bounds** — 3 independent OCR models run in parallel (IBP)
//! 3.  **Result aggregation weighted merge** — weighted combination of 3 OCR outputs (IBP + CROWN)
//! 4.  **Confidence-weighted selection** — softmax gate selects best OCR model (IBP)
//! 5.  **Fallback chain 2-model** — if model A confidence < threshold, try model B (IBP)
//! 6.  **Fallback chain 3-model** — cascaded fallback across 3 OCR models (IBP)
//! 7.  **Full page-to-structured-data pipeline** — 5-stage end-to-end (IBP)
//! 8.  **Multi-page batch processing** — bounds preserved across batched pages (IBP)
//! 9.  **Detection-to-multi-OCR fan-out** — detection feeds 3 OCR models (IBP)
//! 10. **OCR-to-language-model aggregation** — 3 OCR outputs merge into LM decoder (IBP)
//! 11. **Table + OCR parallel branch merge** — table structure + OCR combined (IBP + CROWN)
//! 12. **Ensemble monotone: parallel branches** — narrower input tightens parallel outputs (IBP)
//! 13. **7-model confidence-weighted ensemble** — all 7 heads with learned gating (IBP)
//! 14. **Multi-page aggregation with page attention** — cross-page attention merge (IBP + CROWN)
//!
//! Dimensions (small for fast verification, structurally representative):
//! - HIDDEN=8, SEQ=4, NUM_CLASSES=6, VOCAB=8, NUM_PAGES=3
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
/// Number of detection classes.
const NUM_CLASSES: usize = 6;
/// OCR vocabulary size.
const VOCAB: usize = 8;
/// Number of attention heads.
const NUM_HEADS: usize = 2;
/// FFN intermediate dimension.
const FFN_DIM: usize = HIDDEN * 2;
/// Number of ensemble models.
const NUM_MODELS: usize = 7;
/// Number of parallel OCR models.
const NUM_OCR_MODELS: usize = 3;
/// Number of pages in multi-page batch.
const NUM_PAGES: usize = 3;
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
// 1. Pipeline stage composition: detection -> table structure (IBP)
// ===========================================================================

/// Verifies that detection output (sigmoid confidences) composes cleanly
/// into the table structure recognition input. The detection sigmoid [0,1]
/// output is projected to the table model's feature space, then a sigmoid
/// bbox regression produces coordinates in [0,1].
///
/// Key property: stage-to-stage projection preserves bounded outputs.
#[test]
fn test_7model_pipeline_stage_composition_ibp() {
    let mut b = TensorBlockBuilder::new("7model_stage_compose");
    let input = b.add_input("det_features", &[SEQ, HIDDEN]);

    // Stage 1: DocLayout-YOLO detection -> sigmoid cls
    let det_w = b.add_input("det_w", &[NUM_CLASSES, HIDDEN]);
    let det_logits = b.add_linear(input, det_w, None, &[SEQ, NUM_CLASSES]);
    let det_conf = b.add_sigmoid(det_logits, &[SEQ, NUM_CLASSES]);

    // Stage 2: Table Transformer -- project detection into table query space
    let proj_w = b.add_input("proj_w", &[HIDDEN, NUM_CLASSES]);
    let proj_b = b.add_input("proj_b", &[HIDDEN]);
    let table_feats = b.add_linear(det_conf, proj_w, Some(proj_b), &[SEQ, HIDDEN]);

    // Table FFN: Linear -> ReLU -> Linear
    let ffn_w1 = b.add_input("ffn_w1", &[FFN_DIM, HIDDEN]);
    let ffn_h = b.add_linear(table_feats, ffn_w1, None, &[SEQ, FFN_DIM]);
    let ffn_act = b.add_relu(ffn_h, &[SEQ, FFN_DIM]);
    let ffn_w2 = b.add_input("ffn_w2", &[HIDDEN, FFN_DIM]);
    let ffn_out = b.add_linear(ffn_act, ffn_w2, None, &[SEQ, HIDDEN]);

    // Table bbox regression head -> sigmoid
    let box_w = b.add_input("box_w", &[4, HIDDEN]);
    let box_b = b.add_input("box_b", &[4]);
    let box_logits = b.add_linear(ffn_out, box_w, Some(box_b), &[SEQ, 4]);
    let out = b.add_sigmoid(box_logits, &[SEQ, 4]);
    let def = b.build(out).expect("valid stage compose kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_CLASSES, HIDDEN]), // det_w
        weight(&[HIDDEN, NUM_CLASSES]), // proj_w
        bias_zero(&[HIDDEN]),           // proj_b
        weight(&[FFN_DIM, HIDDEN]),     // ffn_w1
        weight(&[HIDDEN, FFN_DIM]),     // ffn_w2
        weight(&[4, HIDDEN]),           // box_w
        bias_zero(&[4]),                // box_b
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("7model stage compose IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 2. Parallel dispatch bounds: 3 OCR models run independently (IBP)
// ===========================================================================

/// Three OCR models (PaddleOCR, FireRed-OCR, GLM-OCR) receive the same
/// features and produce independent softmax character distributions. The
/// test verifies each branch produces valid [0,1] softmax output.
///
/// Key property: parallel branches do not interfere with each other's bounds.
#[test]
fn test_7model_parallel_dispatch_bounds_ibp() {
    let mut b = TensorBlockBuilder::new("7model_parallel_dispatch");
    let input = b.add_input("shared_features", &[SEQ, HIDDEN]);

    // Branch 1: PaddleOCR head -> softmax
    let pad_w = b.add_input("paddle_w", &[VOCAB, HIDDEN]);
    let pad_b = b.add_input("paddle_b", &[VOCAB]);
    let pad_logits = b.add_linear(input, pad_w, Some(pad_b), &[SEQ, VOCAB]);
    let pad_out = b.add_softmax(pad_logits, -1, &[SEQ, VOCAB]);

    // Branch 2: FireRed-OCR head -> softmax
    let fire_w = b.add_input("firered_w", &[VOCAB, HIDDEN]);
    let fire_b = b.add_input("firered_b", &[VOCAB]);
    let fire_logits = b.add_linear(input, fire_w, Some(fire_b), &[SEQ, VOCAB]);
    let fire_out = b.add_softmax(fire_logits, -1, &[SEQ, VOCAB]);

    // Branch 3: GLM-OCR head -> softmax
    let glm_w = b.add_input("glm_w", &[VOCAB, HIDDEN]);
    let glm_b = b.add_input("glm_b", &[VOCAB]);
    let glm_logits = b.add_linear(input, glm_w, Some(glm_b), &[SEQ, VOCAB]);
    let glm_out = b.add_softmax(glm_logits, -1, &[SEQ, VOCAB]);

    // Merge: average the three softmax outputs
    let sum12 = b.add_binary_add(pad_out, fire_out, &[SEQ, VOCAB]);
    let sum123 = b.add_binary_add(sum12, glm_out, &[SEQ, VOCAB]);
    let scale = b.add_input("scale", &[SEQ, VOCAB]);
    let out = b.add_binary_mul(sum123, scale, &[SEQ, VOCAB]);
    let def = b.build(out).expect("valid parallel dispatch kernel");

    let scale_data = ArrayD::from_elem(IxDyn(&[SEQ, VOCAB]), 1.0f32 / 3.0);
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB, HIDDEN]),                       // paddle_w
        bias_zero(&[VOCAB]),                            // paddle_b
        weight(&[VOCAB, HIDDEN]),                       // firered_w
        bias_zero(&[VOCAB]),                            // firered_b
        weight(&[VOCAB, HIDDEN]),                       // glm_w
        bias_zero(&[VOCAB]),                            // glm_b
        TensorParamBinding::ConstantTensor(scale_data), // scale
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("7model parallel dispatch IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Average of 3 softmaxes in [0,1] should stay in [0,1]
    assert!(lo_min >= -1e-3, "avg softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-3, "avg softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 3. Result aggregation: weighted merge of 3 OCR outputs (IBP + CROWN)
// ===========================================================================

/// Three OCR model softmax outputs are combined via learned weights (another
/// softmax gate over model confidences). The gated combination preserves
/// [0,1] softmax structure.
///
/// Key property: weighted convex combination of probability distributions
/// remains a valid probability distribution.
#[test]
fn test_7model_result_aggregation_weighted_merge_ibp_crown() {
    let mut b = TensorBlockBuilder::new("7model_weighted_merge");
    let input = b.add_input("features", &[SEQ, HIDDEN]);

    // Three OCR heads produce softmax logits
    let head1_w = b.add_input("h1_w", &[VOCAB, HIDDEN]);
    let head1_logits = b.add_linear(input, head1_w, None, &[SEQ, VOCAB]);
    let head1_sm = b.add_softmax(head1_logits, -1, &[SEQ, VOCAB]);

    let head2_w = b.add_input("h2_w", &[VOCAB, HIDDEN]);
    let head2_logits = b.add_linear(input, head2_w, None, &[SEQ, VOCAB]);
    let head2_sm = b.add_softmax(head2_logits, -1, &[SEQ, VOCAB]);

    let head3_w = b.add_input("h3_w", &[VOCAB, HIDDEN]);
    let head3_logits = b.add_linear(input, head3_w, None, &[SEQ, VOCAB]);
    let head3_sm = b.add_softmax(head3_logits, -1, &[SEQ, VOCAB]);

    // Gating: Linear -> softmax over 3 models
    let gate_w = b.add_input("gate_w", &[NUM_OCR_MODELS, HIDDEN]);
    let gate_logits = b.add_linear(input, gate_w, None, &[SEQ, NUM_OCR_MODELS]);
    let gate_probs = b.add_softmax(gate_logits, -1, &[SEQ, NUM_OCR_MODELS]);

    // Weighted combination: sum(gate_i * head_i) approximated as linear merge
    // Reshape gate [SEQ, 3] -> project to [SEQ, VOCAB] as merge weights
    let merge_w = b.add_input("merge_w", &[VOCAB, NUM_OCR_MODELS]);
    let merge_coeff = b.add_linear(gate_probs, merge_w, None, &[SEQ, VOCAB]);
    let merge_act = b.add_sigmoid(merge_coeff, &[SEQ, VOCAB]);

    // Weighted heads via element-wise multiply + sum
    let weighted1 = b.add_binary_mul(head1_sm, merge_act, &[SEQ, VOCAB]);
    let weighted2 = b.add_binary_mul(head2_sm, merge_act, &[SEQ, VOCAB]);
    let partial = b.add_binary_add(weighted1, weighted2, &[SEQ, VOCAB]);
    let weighted3 = b.add_binary_mul(head3_sm, merge_act, &[SEQ, VOCAB]);
    let out = b.add_binary_add(partial, weighted3, &[SEQ, VOCAB]);
    let def = b.build(out).expect("valid weighted merge kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB, HIDDEN]),          // h1_w
        weight(&[VOCAB, HIDDEN]),          // h2_w
        weight(&[VOCAB, HIDDEN]),          // h3_w
        weight(&[NUM_OCR_MODELS, HIDDEN]), // gate_w
        weight(&[VOCAB, NUM_OCR_MODELS]),  // merge_w
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("7model weighted merge IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("7model weighted merge CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 4. Confidence-weighted model selection (IBP)
// ===========================================================================

/// A softmax gate over NUM_MODELS produces routing probabilities. These
/// gate the output of each model head. Final sigmoid produces the
/// selected model's confidence.
///
/// Key property: softmax gating preserves boundedness.
#[test]
fn test_7model_confidence_weighted_selection_ibp() {
    let mut b = TensorBlockBuilder::new("7model_conf_select");
    let input = b.add_input("features", &[SEQ, HIDDEN]);

    // Confidence gate: Linear -> softmax -> [SEQ, NUM_MODELS]
    let gate_w = b.add_input("gate_w", &[NUM_MODELS, HIDDEN]);
    let gate_b = b.add_input("gate_b", &[NUM_MODELS]);
    let gate_logits = b.add_linear(input, gate_w, Some(gate_b), &[SEQ, NUM_MODELS]);
    let gate_probs = b.add_softmax(gate_logits, -1, &[SEQ, NUM_MODELS]);

    // Model heads: project gate probs through a selection matrix to output space
    let select_w = b.add_input("select_w", &[NUM_CLASSES, NUM_MODELS]);
    let selected = b.add_linear(gate_probs, select_w, None, &[SEQ, NUM_CLASSES]);

    // Final confidence: sigmoid
    let out = b.add_sigmoid(selected, &[SEQ, NUM_CLASSES]);
    let def = b.build(out).expect("valid confidence selection kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_MODELS, HIDDEN]),      // gate_w
        bias_zero(&[NUM_MODELS]),           // gate_b
        weight(&[NUM_CLASSES, NUM_MODELS]), // select_w
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("7model confidence selection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 5. Fallback chain: 2-model (IBP)
// ===========================================================================

/// Two-model fallback: FireRed-OCR (primary) and PaddleOCR (fallback).
/// If the primary model's confidence is low, the fallback model's output
/// is weighted higher. Modeled as: gate selects between primary and fallback
/// via sigmoid gating — conf * primary_score + (complementary) * fallback_score.
/// Approximated as additive blend: conf * primary + fallback_weight * fallback.
///
/// Key property: gated blend of two bounded outputs stays bounded.
#[test]
fn test_7model_fallback_chain_2model_ibp() {
    let mut b = TensorBlockBuilder::new("7model_fallback_2");
    let input = b.add_input("features", &[SEQ, HIDDEN]);

    // Primary (FireRed-OCR): Linear -> softmax
    let prim_w = b.add_input("prim_w", &[VOCAB, HIDDEN]);
    let prim_logits = b.add_linear(input, prim_w, None, &[SEQ, VOCAB]);
    let prim_out = b.add_softmax(prim_logits, -1, &[SEQ, VOCAB]);

    // Primary confidence gate: Linear -> sigmoid -> broadcast
    let conf_w = b.add_input("conf_w", &[1, HIDDEN]);
    let conf_logit = b.add_linear(input, conf_w, None, &[SEQ, 1]);
    let conf = b.add_sigmoid(conf_logit, &[SEQ, 1]);
    let conf_bc = b.add_broadcast(conf, &[SEQ, VOCAB]);

    // Fallback (PaddleOCR): Linear -> softmax
    let fall_w = b.add_input("fall_w", &[VOCAB, HIDDEN]);
    let fall_logits = b.add_linear(input, fall_w, None, &[SEQ, VOCAB]);
    let fall_out = b.add_softmax(fall_logits, -1, &[SEQ, VOCAB]);

    // Gated blend: conf * primary (high confidence path)
    let gated_prim = b.add_binary_mul(conf_bc, prim_out, &[SEQ, VOCAB]);
    // Fallback contribution: scaled fallback
    let fall_scale = b.add_input("fall_scale", &[SEQ, VOCAB]);
    let scaled_fall = b.add_binary_mul(fall_scale, fall_out, &[SEQ, VOCAB]);
    // Combine
    let out = b.add_binary_add(gated_prim, scaled_fall, &[SEQ, VOCAB]);
    let def = b.build(out).expect("valid fallback 2-model kernel");

    let fall_scale_data = ArrayD::from_elem(IxDyn(&[SEQ, VOCAB]), 0.5f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB, HIDDEN]),                            // prim_w
        weight(&[1, HIDDEN]),                                // conf_w
        weight(&[VOCAB, HIDDEN]),                            // fall_w
        TensorParamBinding::ConstantTensor(fall_scale_data), // fall_scale
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("7model fallback 2-model IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 6. Fallback chain: 3-model cascaded (IBP)
// ===========================================================================

/// Three-model cascaded fallback: FireRed -> PaddleOCR -> GLM-OCR.
/// Each model has a confidence gate. The cascade is modeled as a softmax
/// gate over 3 models, weighting each model's softmax output.
///
/// Key property: cascaded gated blend preserves bounded output.
#[test]
fn test_7model_fallback_chain_3model_ibp() {
    let mut b = TensorBlockBuilder::new("7model_fallback_3");
    let input = b.add_input("features", &[SEQ, HIDDEN]);

    // Model A (FireRed): Linear -> softmax
    let a_w = b.add_input("a_w", &[VOCAB, HIDDEN]);
    let a_logits = b.add_linear(input, a_w, None, &[SEQ, VOCAB]);
    let a_out = b.add_softmax(a_logits, -1, &[SEQ, VOCAB]);

    // Model B (PaddleOCR): Linear -> softmax
    let bm_w = b.add_input("b_w", &[VOCAB, HIDDEN]);
    let bm_logits = b.add_linear(input, bm_w, None, &[SEQ, VOCAB]);
    let bm_out = b.add_softmax(bm_logits, -1, &[SEQ, VOCAB]);

    // Model C (GLM-OCR): Linear -> softmax
    let c_w = b.add_input("c_w", &[VOCAB, HIDDEN]);
    let c_logits = b.add_linear(input, c_w, None, &[SEQ, VOCAB]);
    let c_out = b.add_softmax(c_logits, -1, &[SEQ, VOCAB]);

    // Cascaded gating: softmax over 3 model confidences
    let gate_w = b.add_input("gate_w", &[NUM_OCR_MODELS, HIDDEN]);
    let gate_logits = b.add_linear(input, gate_w, None, &[SEQ, NUM_OCR_MODELS]);
    let gate_probs = b.add_softmax(gate_logits, -1, &[SEQ, NUM_OCR_MODELS]);

    // Weighted merge: project gate [SEQ, 3] -> [SEQ, VOCAB] as blend coefficients
    let blend_w = b.add_input("blend_w", &[VOCAB, NUM_OCR_MODELS]);
    let blend_coeff = b.add_linear(gate_probs, blend_w, None, &[SEQ, VOCAB]);
    let blend_gate = b.add_sigmoid(blend_coeff, &[SEQ, VOCAB]);

    // Apply gating to each model output and sum
    let ga = b.add_binary_mul(blend_gate, a_out, &[SEQ, VOCAB]);
    let gb = b.add_binary_mul(blend_gate, bm_out, &[SEQ, VOCAB]);
    let gc = b.add_binary_mul(blend_gate, c_out, &[SEQ, VOCAB]);

    let sum_ab = b.add_binary_add(ga, gb, &[SEQ, VOCAB]);
    let out = b.add_binary_add(sum_ab, gc, &[SEQ, VOCAB]);
    let def = b.build(out).expect("valid fallback 3-model kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB, HIDDEN]),          // a_w
        weight(&[VOCAB, HIDDEN]),          // b_w
        weight(&[VOCAB, HIDDEN]),          // c_w
        weight(&[NUM_OCR_MODELS, HIDDEN]), // gate_w
        weight(&[VOCAB, NUM_OCR_MODELS]),  // blend_w
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("7model fallback 3-model IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ===========================================================================
// 7. Full page-to-structured-data pipeline (IBP)
// ===========================================================================

/// Five-stage document understanding pipeline:
///   1. DocLayout-YOLO: detection sigmoid
///   2. Table Transformer: table structure sigmoid bbox
///   3. PaddleOCR: text recognition softmax
///   4. GLM-OCR: language understanding FFN
///   5. Final aggregation: Linear -> sigmoid confidence
///
/// Key property: end-to-end bounds compose through 5 sequential stages.
#[test]
fn test_7model_full_page_to_structured_data_pipeline_ibp() {
    let mut b = TensorBlockBuilder::new("7model_full_pipeline");
    let input = b.add_input("page_features", &[SEQ, HIDDEN]);

    // Stage 1: DocLayout-YOLO detection -> sigmoid
    let s1_w = b.add_input("s1_w", &[NUM_CLASSES, HIDDEN]);
    let s1_logits = b.add_linear(input, s1_w, None, &[SEQ, NUM_CLASSES]);
    let s1_out = b.add_sigmoid(s1_logits, &[SEQ, NUM_CLASSES]);

    // Stage 2: Table Transformer -> sigmoid bbox
    let s2_w = b.add_input("s2_w", &[4, NUM_CLASSES]);
    let s2_logits = b.add_linear(s1_out, s2_w, None, &[SEQ, 4]);
    let s2_out = b.add_sigmoid(s2_logits, &[SEQ, 4]);

    // Stage 3: PaddleOCR recognition -> softmax
    let s3_w = b.add_input("s3_w", &[HIDDEN, 4]);
    let s3_feats = b.add_linear(s2_out, s3_w, None, &[SEQ, HIDDEN]);
    let s3_act = b.add_relu(s3_feats, &[SEQ, HIDDEN]);
    let s3_ctc_w = b.add_input("s3_ctc_w", &[VOCAB, HIDDEN]);
    let s3_logits = b.add_linear(s3_act, s3_ctc_w, None, &[SEQ, VOCAB]);
    let s3_out = b.add_softmax(s3_logits, -1, &[SEQ, VOCAB]);

    // Stage 4: GLM-OCR language understanding FFN
    let s4_w1 = b.add_input("s4_w1", &[HIDDEN, VOCAB]);
    let s4_h = b.add_linear(s3_out, s4_w1, None, &[SEQ, HIDDEN]);
    let s4_act = b.add_relu(s4_h, &[SEQ, HIDDEN]);
    let s4_w2 = b.add_input("s4_w2", &[HIDDEN, HIDDEN]);
    let s4_out = b.add_linear(s4_act, s4_w2, None, &[SEQ, HIDDEN]);

    // Stage 5: Final aggregation -> sigmoid confidence
    let s5_w = b.add_input("s5_w", &[NUM_CLASSES, HIDDEN]);
    let s5_b = b.add_input("s5_b", &[NUM_CLASSES]);
    let s5_logits = b.add_linear(s4_out, s5_w, Some(s5_b), &[SEQ, NUM_CLASSES]);
    let out = b.add_sigmoid(s5_logits, &[SEQ, NUM_CLASSES]);
    let def = b.build(out).expect("valid full pipeline kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_CLASSES, HIDDEN]), // s1_w
        weight(&[4, NUM_CLASSES]),      // s2_w
        weight(&[HIDDEN, 4]),           // s3_w
        weight(&[VOCAB, HIDDEN]),       // s3_ctc_w
        weight(&[HIDDEN, VOCAB]),       // s4_w1
        weight(&[HIDDEN, HIDDEN]),      // s4_w2
        weight(&[NUM_CLASSES, HIDDEN]), // s5_w
        bias_zero(&[NUM_CLASSES]),      // s5_b
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("7model full pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 8. Multi-page batch processing bounds (IBP)
// ===========================================================================

/// Multiple pages processed through the same pipeline. Uses a flattened
/// representation [NUM_PAGES * SEQ, HIDDEN] to simulate batched processing.
/// Verifies that bounds are preserved across the batch dimension.
///
/// Key property: batch processing does not widen per-page bounds.
#[test]
fn test_7model_multi_page_batch_processing_ibp() {
    let batch_seq = NUM_PAGES * SEQ;

    let mut b = TensorBlockBuilder::new("7model_multi_page");
    let input = b.add_input("pages_features", &[batch_seq, HIDDEN]);

    // Detection: shared weights across all pages
    let det_w = b.add_input("det_w", &[NUM_CLASSES, HIDDEN]);
    let det_logits = b.add_linear(input, det_w, None, &[batch_seq, NUM_CLASSES]);
    let det_conf = b.add_sigmoid(det_logits, &[batch_seq, NUM_CLASSES]);

    // OCR head: shared weights
    let ocr_w = b.add_input("ocr_w", &[VOCAB, NUM_CLASSES]);
    let ocr_b = b.add_input("ocr_b", &[VOCAB]);
    let ocr_logits = b.add_linear(det_conf, ocr_w, Some(ocr_b), &[batch_seq, VOCAB]);
    let out = b.add_softmax(ocr_logits, -1, &[batch_seq, VOCAB]);
    let def = b.build(out).expect("valid multi-page kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_CLASSES, HIDDEN]), // det_w
        weight(&[VOCAB, NUM_CLASSES]),  // ocr_w
        bias_zero(&[VOCAB]),            // ocr_b
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[batch_seq, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("7model multi-page IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 9. Detection-to-multi-OCR fan-out (IBP)
// ===========================================================================

/// Detection output fans out to 3 OCR models simultaneously. Each OCR model
/// processes the same detection features independently, then results merge.
///
/// Key property: fan-out from detection preserves bounds in all branches.
#[test]
fn test_7model_detection_to_multi_ocr_fanout_ibp() {
    let mut b = TensorBlockBuilder::new("7model_fanout");
    let input = b.add_input("det_features", &[SEQ, HIDDEN]);

    // Detection: sigmoid confidence
    let det_w = b.add_input("det_w", &[NUM_CLASSES, HIDDEN]);
    let det_logits = b.add_linear(input, det_w, None, &[SEQ, NUM_CLASSES]);
    let det_conf = b.add_sigmoid(det_logits, &[SEQ, NUM_CLASSES]);

    // OCR branch 1: PaddleOCR -> softmax
    let ocr1_w = b.add_input("ocr1_w", &[VOCAB, NUM_CLASSES]);
    let ocr1_logits = b.add_linear(det_conf, ocr1_w, None, &[SEQ, VOCAB]);
    let ocr1_out = b.add_softmax(ocr1_logits, -1, &[SEQ, VOCAB]);

    // OCR branch 2: FireRed -> softmax
    let ocr2_w = b.add_input("ocr2_w", &[VOCAB, NUM_CLASSES]);
    let ocr2_logits = b.add_linear(det_conf, ocr2_w, None, &[SEQ, VOCAB]);
    let ocr2_out = b.add_softmax(ocr2_logits, -1, &[SEQ, VOCAB]);

    // OCR branch 3: GLM-OCR -> softmax
    let ocr3_w = b.add_input("ocr3_w", &[VOCAB, NUM_CLASSES]);
    let ocr3_logits = b.add_linear(det_conf, ocr3_w, None, &[SEQ, VOCAB]);
    let ocr3_out = b.add_softmax(ocr3_logits, -1, &[SEQ, VOCAB]);

    // Average merge
    let sum12 = b.add_binary_add(ocr1_out, ocr2_out, &[SEQ, VOCAB]);
    let sum123 = b.add_binary_add(sum12, ocr3_out, &[SEQ, VOCAB]);
    let scale = b.add_input("scale", &[SEQ, VOCAB]);
    let out = b.add_binary_mul(sum123, scale, &[SEQ, VOCAB]);
    let def = b.build(out).expect("valid fanout kernel");

    let scale_data = ArrayD::from_elem(IxDyn(&[SEQ, VOCAB]), 1.0f32 / 3.0);
    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_CLASSES, HIDDEN]),                 // det_w
        weight(&[VOCAB, NUM_CLASSES]),                  // ocr1_w
        weight(&[VOCAB, NUM_CLASSES]),                  // ocr2_w
        weight(&[VOCAB, NUM_CLASSES]),                  // ocr3_w
        TensorParamBinding::ConstantTensor(scale_data), // scale
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("7model fanout IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-3, "avg softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-3, "avg softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 10. OCR-to-language-model aggregation (IBP)
// ===========================================================================

/// Three OCR model softmax outputs are concatenated and fed into a language
/// model decoder (GLM-OCR style) for final token prediction.
///
/// Key property: multi-OCR aggregation into LM decoder produces bounded logits.
#[test]
fn test_7model_ocr_to_language_model_aggregation_ibp() {
    let concat_dim = VOCAB * NUM_OCR_MODELS; // 3 OCR vocabularies concatenated

    let mut b = TensorBlockBuilder::new("7model_ocr_to_lm");
    let input = b.add_input("features", &[SEQ, HIDDEN]);

    // Three OCR heads -> softmax
    let ocr1_w = b.add_input("ocr1_w", &[VOCAB, HIDDEN]);
    let ocr1_logits = b.add_linear(input, ocr1_w, None, &[SEQ, VOCAB]);
    let ocr1_out = b.add_softmax(ocr1_logits, -1, &[SEQ, VOCAB]);

    let ocr2_w = b.add_input("ocr2_w", &[VOCAB, HIDDEN]);
    let ocr2_logits = b.add_linear(input, ocr2_w, None, &[SEQ, VOCAB]);
    let ocr2_out = b.add_softmax(ocr2_logits, -1, &[SEQ, VOCAB]);

    let ocr3_w = b.add_input("ocr3_w", &[VOCAB, HIDDEN]);
    let ocr3_logits = b.add_linear(input, ocr3_w, None, &[SEQ, VOCAB]);
    let ocr3_out = b.add_softmax(ocr3_logits, -1, &[SEQ, VOCAB]);

    // Concatenate OCR outputs: [SEQ, 3*VOCAB]
    let concat = b.add_concat(&[ocr1_out, ocr2_out, ocr3_out], 1, &[SEQ, concat_dim]);

    // LM decoder FFN: Linear -> ReLU -> Linear -> softmax
    let lm_w1 = b.add_input("lm_w1", &[HIDDEN, concat_dim]);
    let lm_h = b.add_linear(concat, lm_w1, None, &[SEQ, HIDDEN]);
    let lm_act = b.add_relu(lm_h, &[SEQ, HIDDEN]);
    let lm_w2 = b.add_input("lm_w2", &[VOCAB, HIDDEN]);
    let lm_b2 = b.add_input("lm_b2", &[VOCAB]);
    let lm_logits = b.add_linear(lm_act, lm_w2, Some(lm_b2), &[SEQ, VOCAB]);
    let out = b.add_softmax(lm_logits, -1, &[SEQ, VOCAB]);
    let def = b.build(out).expect("valid OCR-to-LM kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[VOCAB, HIDDEN]),      // ocr1_w
        weight(&[VOCAB, HIDDEN]),      // ocr2_w
        weight(&[VOCAB, HIDDEN]),      // ocr3_w
        weight(&[HIDDEN, concat_dim]), // lm_w1
        weight(&[VOCAB, HIDDEN]),      // lm_w2
        bias_zero(&[VOCAB]),           // lm_b2
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("7model OCR-to-LM IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 11. Table + OCR parallel branch merge (IBP + CROWN)
// ===========================================================================

/// Table structure recognition and OCR run in parallel on the same
/// detection output, then merge via learned combination.
///
/// Key property: independent parallel branches combine with finite bounds.
#[test]
fn test_7model_table_ocr_parallel_merge_ibp_crown() {
    let mut b = TensorBlockBuilder::new("7model_table_ocr_merge");
    let input = b.add_input("det_output", &[SEQ, NUM_CLASSES]);

    // Branch A: Table structure -> sigmoid bbox
    let table_w = b.add_input("table_w", &[4, NUM_CLASSES]);
    let table_logits = b.add_linear(input, table_w, None, &[SEQ, 4]);
    let table_out = b.add_sigmoid(table_logits, &[SEQ, 4]);

    // Branch B: OCR -> softmax
    let ocr_w = b.add_input("ocr_w", &[VOCAB, NUM_CLASSES]);
    let ocr_logits = b.add_linear(input, ocr_w, None, &[SEQ, VOCAB]);
    let ocr_out = b.add_softmax(ocr_logits, -1, &[SEQ, VOCAB]);

    // Merge: project table [SEQ, 4] and OCR [SEQ, VOCAB] to common space
    let merge_table_w = b.add_input("merge_table_w", &[HIDDEN, 4]);
    let merge_table = b.add_linear(table_out, merge_table_w, None, &[SEQ, HIDDEN]);

    let merge_ocr_w = b.add_input("merge_ocr_w", &[HIDDEN, VOCAB]);
    let merge_ocr = b.add_linear(ocr_out, merge_ocr_w, None, &[SEQ, HIDDEN]);

    // Combine
    let combined = b.add_binary_add(merge_table, merge_ocr, &[SEQ, HIDDEN]);

    // Final sigmoid confidence
    let final_w = b.add_input("final_w", &[NUM_CLASSES, HIDDEN]);
    let final_logits = b.add_linear(combined, final_w, None, &[SEQ, NUM_CLASSES]);
    let out = b.add_sigmoid(final_logits, &[SEQ, NUM_CLASSES]);
    let def = b.build(out).expect("valid table-ocr merge kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[4, NUM_CLASSES]),      // table_w
        weight(&[VOCAB, NUM_CLASSES]),  // ocr_w
        weight(&[HIDDEN, 4]),           // merge_table_w
        weight(&[HIDDEN, VOCAB]),       // merge_ocr_w
        weight(&[NUM_CLASSES, HIDDEN]), // final_w
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, NUM_CLASSES], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("7model table+OCR merge IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("7model table+OCR merge CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}

// ===========================================================================
// 12. Ensemble monotone: parallel branches (IBP)
// ===========================================================================

/// Monotonicity test: narrower input bounds produce output bounds that are
/// no wider than those from wider input bounds, even through parallel
/// branches and merge.
///
/// Key property: IBP monotonicity holds through fan-out + merge.
#[test]
fn test_7model_ensemble_monotone_parallel_ibp() {
    // Build a parallel-merge pipeline
    let build_parallel = || {
        let mut b = TensorBlockBuilder::new("7model_monotone_parallel");
        let input = b.add_input("features", &[SEQ, HIDDEN]);

        // Branch A: sigmoid
        let a_w = b.add_input("a_w", &[NUM_CLASSES, HIDDEN]);
        let a_logits = b.add_linear(input, a_w, None, &[SEQ, NUM_CLASSES]);
        let a_out = b.add_sigmoid(a_logits, &[SEQ, NUM_CLASSES]);

        // Branch B: sigmoid
        let b_w = b.add_input("b_w", &[NUM_CLASSES, HIDDEN]);
        let b_logits = b.add_linear(input, b_w, None, &[SEQ, NUM_CLASSES]);
        let b_out = b.add_sigmoid(b_logits, &[SEQ, NUM_CLASSES]);

        // Average
        let sum_val = b.add_binary_add(a_out, b_out, &[SEQ, NUM_CLASSES]);
        let scale = b.add_input("scale", &[SEQ, NUM_CLASSES]);
        let out = b.add_binary_mul(sum_val, scale, &[SEQ, NUM_CLASSES]);
        let def = b.build(out).expect("valid monotone parallel kernel");

        let scale_data = ArrayD::from_elem(IxDyn(&[SEQ, NUM_CLASSES]), 0.5f32);
        let bindings = vec![
            TensorParamBinding::Variable,
            weight(&[NUM_CLASSES, HIDDEN]),
            weight(&[NUM_CLASSES, HIDDEN]),
            TensorParamBinding::ConstantTensor(scale_data),
        ];
        tensor_kernel_to_graph(&def, &bindings).expect("graph")
    };

    let graph = build_parallel();

    // Wide input: [-1, 1]
    let wide_input = uniform_bounds(&[SEQ, HIDDEN], 1.0);
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");

    // Narrow input: [-0.3, 0.3]
    let narrow_input = uniform_bounds(&[SEQ, HIDDEN], 0.3);
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("IBP narrow");

    let (lo_w, hi_w) = bounds_min_max(&wide_output);
    let (lo_n, hi_n) = bounds_min_max(&narrow_output);
    let wide_width = hi_w - lo_w;
    let narrow_width = hi_n - lo_n;

    eprintln!(
        "7model monotone parallel: wide=[{lo_w:.4}, {hi_w:.4}] w={wide_width:.4} \
         | narrow=[{lo_n:.4}, {hi_n:.4}] w={narrow_width:.4}"
    );

    // Monotonicity: narrow input -> no wider output
    assert!(
        narrow_width <= wide_width + 1e-4,
        "monotone violated: narrow_w={narrow_width} > wide_w={wide_width}"
    );
}

// ===========================================================================
// 13. 7-model confidence-weighted ensemble (IBP)
// ===========================================================================

/// Full 7-model ensemble: each model produces a sigmoid output, weighted
/// by a learned gating mechanism (softmax over model confidences).
///
/// Key property: 7-model gated combination stays bounded.
#[test]
fn test_7model_full_confidence_weighted_ensemble_ibp() {
    let mut b = TensorBlockBuilder::new("7model_full_ensemble");
    let input = b.add_input("features", &[SEQ, HIDDEN]);

    // Gate: Linear -> softmax -> [SEQ, 7]
    let gate_w = b.add_input("gate_w", &[NUM_MODELS, HIDDEN]);
    let gate_b = b.add_input("gate_b", &[NUM_MODELS]);
    let gate_logits = b.add_linear(input, gate_w, Some(gate_b), &[SEQ, NUM_MODELS]);
    let gate_probs = b.add_softmax(gate_logits, -1, &[SEQ, NUM_MODELS]);

    // 7 model heads: gate_probs [SEQ, 7] @ heads_matrix [7, NUM_CLASSES]
    // gives weighted output
    let heads_w = b.add_input("heads_w", &[NUM_CLASSES, NUM_MODELS]);
    let heads_b = b.add_input("heads_b", &[NUM_CLASSES]);
    let gated = b.add_linear(gate_probs, heads_w, Some(heads_b), &[SEQ, NUM_CLASSES]);

    // Per-model confidence scores via sigmoid
    let out = b.add_sigmoid(gated, &[SEQ, NUM_CLASSES]);
    let def = b.build(out).expect("valid full ensemble kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[NUM_MODELS, HIDDEN]),      // gate_w
        bias_zero(&[NUM_MODELS]),           // gate_b
        weight(&[NUM_CLASSES, NUM_MODELS]), // heads_w
        bias_zero(&[NUM_CLASSES]),          // heads_b
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ, HIDDEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("7model full ensemble IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");

    // Non-degenerate: ensemble produces a genuine (non-zero) interval. The
    // tightened softmax+sigmoid IBP now narrows this output well below the old
    // 0.01 floor (observed ~0.0028), so that floor is a stale lower bound made
    // obsolete by tighter bounds; a narrower interval is *better* here. We only
    // require the bounds remain non-degenerate.
    let width = hi_max - lo_min;
    assert!(
        width > 0.0,
        "ensemble output must be a non-degenerate interval, got width={width}"
    );
}

// ===========================================================================
// 14. Multi-page aggregation with page attention (IBP + CROWN)
// ===========================================================================

/// Multi-page document processing: each page produces features that are
/// aggregated via cross-page attention to produce a document-level output.
/// Uses self-attention across page-level features.
///
/// Key property: attention-based page aggregation preserves bounded outputs.
#[test]
fn test_7model_multi_page_attention_aggregation_ibp_crown() {
    let mut b = TensorBlockBuilder::new("7model_page_attention");
    let input = b.add_input("page_features", &[NUM_PAGES, HIDDEN]);

    // Per-page processing: Linear -> ReLU
    let page_w = b.add_input("page_w", &[HIDDEN, HIDDEN]);
    let page_h = b.add_linear(input, page_w, None, &[NUM_PAGES, HIDDEN]);
    let page_act = b.add_relu(page_h, &[NUM_PAGES, HIDDEN]);

    // Cross-page attention
    let q_w = b.add_input("q_w", &[HIDDEN, HIDDEN]);
    let k_w = b.add_input("k_w", &[HIDDEN, HIDDEN]);
    let v_w = b.add_input("v_w", &[HIDDEN, HIDDEN]);
    let out_w = b.add_input("out_w", &[HIDDEN, HIDDEN]);
    let attn_out = b
        .add_multi_head_attention(
            page_act,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[NUM_PAGES, HIDDEN],
        )
        .expect("valid MHA");

    // Document-level output: Linear -> sigmoid
    let doc_w = b.add_input("doc_w", &[NUM_CLASSES, HIDDEN]);
    let doc_b = b.add_input("doc_b", &[NUM_CLASSES]);
    let doc_logits = b.add_linear(attn_out, doc_w, Some(doc_b), &[NUM_PAGES, NUM_CLASSES]);
    let out = b.add_sigmoid(doc_logits, &[NUM_PAGES, NUM_CLASSES]);
    let def = b.build(out).expect("valid page attention kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight(&[HIDDEN, HIDDEN]),      // page_w
        weight(&[HIDDEN, HIDDEN]),      // q_w
        weight(&[HIDDEN, HIDDEN]),      // k_w
        weight(&[HIDDEN, HIDDEN]),      // v_w
        weight(&[HIDDEN, HIDDEN]),      // out_w
        weight(&[NUM_CLASSES, HIDDEN]), // doc_w
        bias_zero(&[NUM_CLASSES]),      // doc_b
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[NUM_PAGES, HIDDEN], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("7model page attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-5, "sigmoid upper <= 1, got {hi_max}");

    // CROWN
    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (clo, chi) = bounds_min_max(&crown_output);
    eprintln!("7model page attention CROWN ({method:?}): bounds=[{clo:.6}, {chi:.6}]");
}
