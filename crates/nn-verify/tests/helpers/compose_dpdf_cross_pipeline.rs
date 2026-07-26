// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-model verification compose tests for dpdf pipeline compositions.
//!
//! The dpdf document understanding pipeline chains multiple models:
//!   DocLayout-YOLO (detection) -> Table Transformer (table structure)
//!   DocLayout-YOLO (detection) -> FireRed-OCR / Granite-Docling (OCR)
//!   FireRed-OCR (CTC) -> token embedding -> decoder (language model)
//!
//! These tests verify that bounds compose correctly across model boundaries,
//! ensuring that outputs from one model stage satisfy the input assumptions
//! of the next stage. Each test constructs a multi-model graph that crosses
//! at least two architectural boundaries.
//!
//! ## Tests
//!
//! 1. **Detection -> OCR (IBP)**: DocLayout-YOLO sigmoid output (detection
//!    confidence in [0, 1]) feeds into FireRed-OCR feature projection followed
//!    by CTC softmax. Verifies character probabilities remain in [0, 1].
//!
//! 2. **Detection -> Table (IBP)**: DocLayout-YOLO bbox sigmoid coordinates
//!    feed into Table Transformer query projection and self-attention. Verifies
//!    attention output bounds propagate through cross-model boundary.
//!
//! 3. **OCR -> Language (IBP)**: FireRed-OCR CTC softmax probabilities feed
//!    into token embedding lookup (modeled as linear projection on probability
//!    simplex) followed by a decoder layer (LayerNorm -> FFN -> sigmoid).
//!    Verifies bounds from OCR probabilities to decoder output.
//!
//! 4. **Multi-head pipeline (IBP + CROWN)**: Simultaneous classification
//!    (sigmoid), box regression (sigmoid), and CTC recognition (softmax) heads
//!    sharing a common feature backbone. Verifies all 3 head outputs maintain
//!    respective [0, 1] bounds through a single composed graph.
//!
//! 5. **Tier routing (IBP)**: Input features branch to different model heads
//!    based on region type. Models the dpdf dispatch pattern where detection
//!    features are routed to either table structure or OCR recognition heads
//!    via softmax gating. Verifies each branch preserves output bounds.
//!
//! Dimensions (small for fast verification, structurally representative):
//! - HIDDEN=16, SEQ=4, NUM_ANCHORS=4, NUM_CLASSES=8, VOCAB_SIZE=12
//!
//! Part of #3870: NY compose tests for dpdf models.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Hidden dimension shared across model boundaries.
const HIDDEN: usize = 16;
/// Sequence length / number of query positions.
const SEQ: usize = 4;
/// Number of detection anchors.
const NUM_ANCHORS: usize = 4;
/// Number of detection classes (DocLayout-YOLO output).
const NUM_CLASSES: usize = 8;
/// OCR vocabulary size (FireRed-OCR CTC output).
const VOCAB_SIZE: usize = 12;
/// Number of attention heads for Table Transformer queries.
const NUM_HEADS: usize = 2;
/// Per-head dimension.
const HEAD_DIM: usize = HIDDEN / NUM_HEADS;
/// FFN intermediate dimension.
const FFN_DIM: usize = HIDDEN * 2;
/// Number of routing branches (table vs OCR).
const NUM_BRANCHES: usize = 2;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// Suppress unused constant warnings for constants used only in specific tests.
const _: () = {
    let _ = HEAD_DIM;
    let _ = FFN_DIM;
    let _ = NUM_BRANCHES;
};

/// Helper: weight tensor of given shape filled with WEIGHT_MAG.
fn w(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG)
}

/// Helper: zeros tensor.
fn zeros(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 0.0f32)
}

/// Helper: ones tensor (for normalization scale parameters).
fn ones(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 1.0f32)
}

// ===========================================================================
// 1. Detection -> OCR: YOLO sigmoid -> FireRed-OCR softmax
// ===========================================================================

/// Build a detection -> OCR cross-model pipeline.
///
/// Stage 1 (DocLayout-YOLO): Linear -> Sigmoid (detection confidence [0, 1]).
/// Stage 2 (FireRed-OCR feature bridge): Linear projection to OCR space.
/// Stage 3 (FireRed-OCR CTC head): Linear -> Softmax (character probs [0, 1]).
///
/// Input: `[NUM_ANCHORS, HIDDEN]` (detection backbone features).
/// Output: `[NUM_ANCHORS, VOCAB_SIZE]` (CTC character probabilities).
fn build_detection_to_ocr_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_cross_detection_to_ocr");

    let input = b.add_input("det_features", &[NUM_ANCHORS, HIDDEN]);

    // Stage 1: DocLayout-YOLO detection head (sigmoid confidence)
    let det_w = b.add_input("det_cls_weight", &[HIDDEN, HIDDEN]);
    let det_b = b.add_input("det_cls_bias", &[HIDDEN]);
    let det_logits = b.add_linear(input, det_w, Some(det_b), &[NUM_ANCHORS, HIDDEN]);
    let det_conf = b.add_sigmoid(det_logits, &[NUM_ANCHORS, HIDDEN]);

    // Stage 2: Feature bridge (detection -> OCR feature space)
    let bridge_w = b.add_input("bridge_weight", &[HIDDEN, HIDDEN]);
    let bridge_b = b.add_input("bridge_bias", &[HIDDEN]);
    let ocr_features = b.add_linear(det_conf, bridge_w, Some(bridge_b), &[NUM_ANCHORS, HIDDEN]);

    // Stage 3: FireRed-OCR CTC head (Linear -> Softmax)
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let ctc_logits = b.add_linear(ocr_features, ctc_w, Some(ctc_b), &[NUM_ANCHORS, VOCAB_SIZE]);
    let out = b.add_softmax(ctc_logits, -1, &[NUM_ANCHORS, VOCAB_SIZE]);

    b.build(out)
        .expect("valid detection -> OCR cross-pipeline kernel")
}

fn detection_to_ocr_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,                             // det_features
        TensorParamBinding::ConstantTensor(w(&[HIDDEN, HIDDEN])), // det_cls_weight
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN])),     // det_cls_bias
        TensorParamBinding::ConstantTensor(w(&[HIDDEN, HIDDEN])), // bridge_weight
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN])),     // bridge_bias
        TensorParamBinding::ConstantTensor(w(&[VOCAB_SIZE, HIDDEN])), // ctc_weight
        TensorParamBinding::ConstantTensor(zeros(&[VOCAB_SIZE])), // ctc_bias
    ]
}

/// Detection -> OCR IBP: sigmoid detection output feeds CTC softmax.
///
/// Verifies the key cross-model boundary: DocLayout-YOLO detection features
/// (bounded in [0, 1] by sigmoid) flow through a bridge projection to
/// FireRed-OCR, producing character probabilities bounded in [0, 1] via softmax.
#[test]
fn test_cross_detection_to_ocr_ibp() {
    let def = build_detection_to_ocr_kernel();
    let bindings = detection_to_ocr_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_ANCHORS, HIDDEN], 5.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through detection -> OCR pipeline");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_ANCHORS, VOCAB_SIZE],
        "detection -> OCR output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("cross detection -> OCR IBP: [{lo_min}, {hi_max}]");

    let eps = 1e-4;
    assert!(
        lo_min >= 0.0 - eps,
        "detection -> OCR: softmax lower >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "detection -> OCR: softmax upper <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 2. Detection -> Table: YOLO bbox -> Table Transformer queries + attention
// ===========================================================================

/// Build a detection -> table structure cross-model pipeline.
///
/// Stage 1 (DocLayout-YOLO): Linear -> Sigmoid (bbox coordinates [0, 1]).
/// Stage 2 (Table Transformer): Query projection + self-attention over
///   detected bounding box features.
///
/// Input: `[NUM_ANCHORS, HIDDEN]` (detection backbone features).
/// Output: `[NUM_ANCHORS, HIDDEN]` (table structure query representations).
fn build_detection_to_table_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_cross_detection_to_table");

    let input = b.add_input("det_features", &[NUM_ANCHORS, HIDDEN]);

    // Stage 1: DocLayout-YOLO box head (sigmoid normalized coords)
    let box_w = b.add_input("box_weight", &[HIDDEN, HIDDEN]);
    let box_b = b.add_input("box_bias", &[HIDDEN]);
    let box_logits = b.add_linear(input, box_w, Some(box_b), &[NUM_ANCHORS, HIDDEN]);
    let box_coords = b.add_sigmoid(box_logits, &[NUM_ANCHORS, HIDDEN]);

    // Stage 2: Table Transformer query projection
    let q_w = b.add_input("query_weight", &[HIDDEN, HIDDEN]);
    let q_b = b.add_input("query_bias", &[HIDDEN]);
    let queries = b.add_linear(box_coords, q_w, Some(q_b), &[NUM_ANCHORS, HIDDEN]);

    // K and V projections (from same detection features for self-attention)
    let k_w = b.add_input("key_weight", &[HIDDEN, HIDDEN]);
    let k_b = b.add_input("key_bias", &[HIDDEN]);
    let keys = b.add_linear(box_coords, k_w, Some(k_b), &[NUM_ANCHORS, HIDDEN]);

    let v_w = b.add_input("value_weight", &[HIDDEN, HIDDEN]);
    let v_b = b.add_input("value_bias", &[HIDDEN]);
    let values = b.add_linear(box_coords, v_w, Some(v_b), &[NUM_ANCHORS, HIDDEN]);

    // Self-attention over table queries
    let scale = 1.0 / (HIDDEN as f32).sqrt();
    let attn_out = b.add_attention(
        queries,
        keys,
        values,
        AttentionMask::Standard,
        Some(scale),
        &[NUM_ANCHORS, HIDDEN],
    );

    // Output projection
    let out_w = b.add_input("out_weight", &[HIDDEN, HIDDEN]);
    let out_b = b.add_input("out_bias", &[HIDDEN]);
    let out = b.add_linear(attn_out, out_w, Some(out_b), &[NUM_ANCHORS, HIDDEN]);

    b.build(out)
        .expect("valid detection -> table cross-pipeline kernel")
}

fn detection_to_table_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,                             // det_features
        TensorParamBinding::ConstantTensor(w(&[HIDDEN, HIDDEN])), // box_weight
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN])),     // box_bias
        TensorParamBinding::ConstantTensor(w(&[HIDDEN, HIDDEN])), // query_weight
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN])),     // query_bias
        TensorParamBinding::ConstantTensor(w(&[HIDDEN, HIDDEN])), // key_weight
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN])),     // key_bias
        TensorParamBinding::ConstantTensor(w(&[HIDDEN, HIDDEN])), // value_weight
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN])),     // value_bias
        TensorParamBinding::ConstantTensor(w(&[HIDDEN, HIDDEN])), // out_weight
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN])),     // out_bias
    ]
}

/// Detection -> Table IBP: YOLO bbox sigmoid feeds Table Transformer queries.
///
/// Verifies that bounded detection coordinates (sigmoid in [0, 1]) produce
/// finite, valid bounds when projected through Table Transformer query/key/value
/// projections and self-attention.
#[test]
fn test_cross_detection_to_table_ibp() {
    let def = build_detection_to_table_kernel();
    let bindings = detection_to_table_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_ANCHORS, HIDDEN], 5.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through detection -> table pipeline");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_ANCHORS, HIDDEN],
        "detection -> table output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("cross detection -> table IBP: [{lo_min}, {hi_max}]");

    // Attention output is not sigmoid-bounded, but must be finite and non-vacuous
    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "detection -> table: bounds must be finite"
    );
    assert!(
        hi_max - lo_min < 1e6,
        "detection -> table: bounds should not be vacuously wide, width={}",
        hi_max - lo_min
    );
}

// ===========================================================================
// 3. OCR -> Language: CTC softmax -> embedding -> decoder
// ===========================================================================

/// Build an OCR -> language model cross-model pipeline.
///
/// Stage 1 (FireRed-OCR): Linear -> Softmax (CTC character probs in [0, 1]).
/// Stage 2 (Embedding): Probability-weighted embedding lookup (Linear on
///   probability simplex -> embedding space).
/// Stage 3 (Decoder): LayerNorm -> Linear -> ReLU -> Linear -> Sigmoid.
///
/// Input: `[SEQ, HIDDEN]` (OCR encoder features).
/// Output: `[SEQ, NUM_CLASSES]` (language model output probs).
fn build_ocr_to_language_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_cross_ocr_to_language");

    let input = b.add_input("ocr_features", &[SEQ, HIDDEN]);

    // Stage 1: FireRed-OCR CTC head (Linear -> Softmax)
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let ctc_logits = b.add_linear(input, ctc_w, Some(ctc_b), &[SEQ, VOCAB_SIZE]);
    let ctc_probs = b.add_softmax(ctc_logits, -1, &[SEQ, VOCAB_SIZE]);

    // Stage 2: Token embedding (probability-weighted linear projection)
    // Models soft embedding: embed = probs @ embedding_matrix
    let embed_w = b.add_input("embed_weight", &[HIDDEN, VOCAB_SIZE]);
    let embed_b = b.add_input("embed_bias", &[HIDDEN]);
    let embeddings = b.add_linear(ctc_probs, embed_w, Some(embed_b), &[SEQ, HIDDEN]);

    // Stage 3: Decoder layer (LayerNorm -> FFN -> Sigmoid)
    let ln_w = b.add_input("ln_weight", &[HIDDEN]);
    let ln_b = b.add_input("ln_bias", &[HIDDEN]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let normed = b.add_layer_norm(embeddings, ln_eps, 1, ln_w, ln_b, &[SEQ, HIDDEN]);

    // FFN: Linear -> ReLU -> Linear
    let ffn1_w = b.add_input("ffn1_weight", &[FFN_DIM, HIDDEN]);
    let ffn1_b = b.add_input("ffn1_bias", &[FFN_DIM]);
    let ffn1 = b.add_linear(normed, ffn1_w, Some(ffn1_b), &[SEQ, FFN_DIM]);
    let ffn1_act = b.add_relu(ffn1, &[SEQ, FFN_DIM]);

    let ffn2_w = b.add_input("ffn2_weight", &[NUM_CLASSES, FFN_DIM]);
    let ffn2_b = b.add_input("ffn2_bias", &[NUM_CLASSES]);
    let ffn2 = b.add_linear(ffn1_act, ffn2_w, Some(ffn2_b), &[SEQ, NUM_CLASSES]);

    let out = b.add_sigmoid(ffn2, &[SEQ, NUM_CLASSES]);

    b.build(out)
        .expect("valid OCR -> language cross-pipeline kernel")
}

fn ocr_to_language_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // ocr_features
        TensorParamBinding::ConstantTensor(w(&[VOCAB_SIZE, HIDDEN])), // ctc_weight
        TensorParamBinding::ConstantTensor(zeros(&[VOCAB_SIZE])), // ctc_bias
        TensorParamBinding::ConstantTensor(w(&[HIDDEN, VOCAB_SIZE])), // embed_weight
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN])), // embed_bias
        TensorParamBinding::ConstantTensor(ones(&[HIDDEN])), // ln_weight
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN])), // ln_bias
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)), // ln_eps
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, HIDDEN])), // ffn1_weight
        TensorParamBinding::ConstantTensor(zeros(&[FFN_DIM])), // ffn1_bias
        TensorParamBinding::ConstantTensor(w(&[NUM_CLASSES, FFN_DIM])), // ffn2_weight
        TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])), // ffn2_bias
    ]
}

/// OCR -> Language IBP: CTC softmax -> token embedding -> decoder.
///
/// Verifies that character probabilities from FireRed-OCR CTC (in [0, 1])
/// propagate through a soft embedding lookup and a decoder layer, producing
/// bounded classification output via final sigmoid in [0, 1].
#[test]
fn test_cross_ocr_to_language_ibp() {
    let def = build_ocr_to_language_kernel();
    let bindings = ocr_to_language_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, HIDDEN], 5.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through OCR -> language pipeline");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ, NUM_CLASSES],
        "OCR -> language output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("cross OCR -> language IBP: [{lo_min}, {hi_max}]");

    // Final sigmoid output must be in [0, 1]
    let eps = 1e-4;
    assert!(
        lo_min >= 0.0 - eps,
        "OCR -> language: sigmoid lower >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "OCR -> language: sigmoid upper <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 4. Multi-head pipeline: cls sigmoid + box sigmoid + CTC softmax
// ===========================================================================

/// Build a multi-head pipeline with 3 simultaneous output heads.
///
/// Shared backbone features feed into:
///   Head 1: Classification sigmoid (cls probs in [0, 1])
///   Head 2: Box regression sigmoid (coords in [0, 1])
///   Head 3: CTC recognition softmax (char probs in [0, 1])
///
/// All heads share a common linear feature projection, then branch.
/// Combined output via concat: `[NUM_ANCHORS, NUM_CLASSES + 4 + VOCAB_SIZE]`.
///
/// Input: `[NUM_ANCHORS, HIDDEN]` (backbone features).
/// Output: `[NUM_ANCHORS, NUM_CLASSES + 4 + VOCAB_SIZE]`.
fn build_multi_head_pipeline_kernel() -> TensorKernelDef {
    let total_out = NUM_CLASSES + 4 + VOCAB_SIZE;
    let mut b = TensorBlockBuilder::new("dpdf_cross_multi_head_pipeline");

    let input = b.add_input("backbone_features", &[NUM_ANCHORS, HIDDEN]);

    // Shared backbone projection
    let proj_w = b.add_input("proj_weight", &[HIDDEN, HIDDEN]);
    let proj_b = b.add_input("proj_bias", &[HIDDEN]);
    let shared = b.add_linear(input, proj_w, Some(proj_b), &[NUM_ANCHORS, HIDDEN]);

    // Head 1: Classification (sigmoid)
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN]);
    let cls_b = b.add_input("cls_bias", &[NUM_CLASSES]);
    let cls_logits = b.add_linear(shared, cls_w, Some(cls_b), &[NUM_ANCHORS, NUM_CLASSES]);
    let cls_probs = b.add_sigmoid(cls_logits, &[NUM_ANCHORS, NUM_CLASSES]);

    // Head 2: Box regression (sigmoid)
    let box_w = b.add_input("box_weight", &[4, HIDDEN]);
    let box_b = b.add_input("box_bias", &[4]);
    let box_logits = b.add_linear(shared, box_w, Some(box_b), &[NUM_ANCHORS, 4]);
    let box_coords = b.add_sigmoid(box_logits, &[NUM_ANCHORS, 4]);

    // Head 3: CTC recognition (softmax)
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let ctc_logits = b.add_linear(shared, ctc_w, Some(ctc_b), &[NUM_ANCHORS, VOCAB_SIZE]);
    let ctc_probs = b.add_softmax(ctc_logits, -1, &[NUM_ANCHORS, VOCAB_SIZE]);

    // Concat all heads: [NUM_ANCHORS, NUM_CLASSES + 4 + VOCAB_SIZE]
    let out = b.add_concat(
        &[cls_probs, box_coords, ctc_probs],
        1,
        &[NUM_ANCHORS, total_out],
    );

    b.build(out).expect("valid multi-head pipeline kernel")
}

fn multi_head_pipeline_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // backbone_features
        TensorParamBinding::ConstantTensor(w(&[HIDDEN, HIDDEN])), // proj_weight
        TensorParamBinding::ConstantTensor(zeros(&[HIDDEN])), // proj_bias
        TensorParamBinding::ConstantTensor(w(&[NUM_CLASSES, HIDDEN])), // cls_weight
        TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])), // cls_bias
        TensorParamBinding::ConstantTensor(w(&[4, HIDDEN])), // box_weight
        TensorParamBinding::ConstantTensor(zeros(&[4])), // box_bias
        TensorParamBinding::ConstantTensor(w(&[VOCAB_SIZE, HIDDEN])), // ctc_weight
        TensorParamBinding::ConstantTensor(zeros(&[VOCAB_SIZE])), // ctc_bias
    ]
}

/// Multi-head pipeline IBP: cls + box + CTC heads all bounded in [0, 1].
///
/// Verifies that all 3 output heads (classification sigmoid, box regression
/// sigmoid, CTC recognition softmax) maintain [0, 1] bounds when composed
/// through a shared backbone feature projection.
#[test]
fn test_cross_multi_head_pipeline_ibp() {
    let total_out = NUM_CLASSES + 4 + VOCAB_SIZE;
    let def = build_multi_head_pipeline_kernel();
    let bindings = multi_head_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_ANCHORS, HIDDEN], 5.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through multi-head pipeline");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_ANCHORS, total_out],
        "multi-head pipeline output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("cross multi-head pipeline IBP: [{lo_min}, {hi_max}]");

    // All three heads use sigmoid or softmax, so entire output in [0, 1].
    let eps = 1e-4;
    assert!(
        lo_min >= 0.0 - eps,
        "multi-head: all outputs >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "multi-head: all outputs <= 1, got {hi_max}"
    );
}

/// Multi-head pipeline CROWN: tighter bounds through shared backbone + 3 heads.
///
/// CROWN linearization should produce tighter (or equal) bounds compared to IBP
/// for the multi-head pipeline. This tests the interaction of CROWN with the
/// branching topology (shared backbone -> 3 divergent heads -> concat).
#[test]
fn test_cross_multi_head_pipeline_crown() {
    let def = build_multi_head_pipeline_kernel();
    let bindings = multi_head_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_ANCHORS, HIDDEN], 5.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("cross multi-head pipeline CROWN: method={method:?}, [{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    let eps = 1e-4;
    assert!(
        lo_min >= 0.0 - eps,
        "multi-head CROWN: all outputs >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "multi-head CROWN: all outputs <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 5. Tier routing: input -> softmax gate -> branch to table/OCR heads
// ===========================================================================

/// Build a tier-routing pipeline that dispatches to different model heads.
///
/// Models the dpdf dispatch pattern: detection features are classified by
/// region type (e.g., table vs text), then routed to the appropriate
/// recognition head.
///
/// Stage 1 (Router): Linear -> Softmax gate (routing weights in [0, 1]).
/// Stage 2a (Table head): Linear -> Sigmoid (table structure confidence).
/// Stage 2b (OCR head): Linear -> Softmax (character probabilities).
/// Stage 3 (Merge): Weighted combination via element-wise multiply + add.
///
/// The merge uses the routing softmax output as attention-like weights:
///   out = gate[0] * table_out + gate[1] * ocr_out
/// Since gate sums to 1 and both heads are bounded in [0, 1], the *true* merged
/// output is bounded in [0, 1]. Plain IBP, however, cannot track the simplex
/// correlation gate[0] + gate[1] = 1, so its sound bound is the [0, 2] envelope
/// of the sum of two independent [0, 1] products (CROWN recovers the [0, 1]).
///
/// Input: `[SEQ, HIDDEN]` (detection features).
/// Output: `[SEQ, NUM_CLASSES]` (merged prediction).
fn build_tier_routing_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_cross_tier_routing");

    let input = b.add_input("features", &[SEQ, HIDDEN]);

    // Stage 1: Routing gate (softmax over NUM_BRANCHES=2)
    let gate_w = b.add_input("gate_weight", &[NUM_BRANCHES, HIDDEN]);
    let gate_b = b.add_input("gate_bias", &[NUM_BRANCHES]);
    let gate_logits = b.add_linear(input, gate_w, Some(gate_b), &[SEQ, NUM_BRANCHES]);
    let gate_probs = b.add_softmax(gate_logits, -1, &[SEQ, NUM_BRANCHES]);

    // Stage 2a: Table structure head (sigmoid)
    let table_w = b.add_input("table_weight", &[NUM_CLASSES, HIDDEN]);
    let table_b = b.add_input("table_bias", &[NUM_CLASSES]);
    let table_logits = b.add_linear(input, table_w, Some(table_b), &[SEQ, NUM_CLASSES]);
    let table_out = b.add_sigmoid(table_logits, &[SEQ, NUM_CLASSES]);

    // Stage 2b: OCR recognition head (softmax)
    let ocr_w = b.add_input("ocr_weight", &[NUM_CLASSES, HIDDEN]);
    let ocr_b = b.add_input("ocr_bias", &[NUM_CLASSES]);
    let ocr_logits = b.add_linear(input, ocr_w, Some(ocr_b), &[SEQ, NUM_CLASSES]);
    let ocr_out = b.add_softmax(ocr_logits, -1, &[SEQ, NUM_CLASSES]);

    // Stage 3: Weighted merge using gate probabilities.
    // gate_probs shape: [SEQ, 2]. We use narrow to extract each branch weight,
    // broadcast to [SEQ, NUM_CLASSES], then multiply with respective head output.

    // Extract gate_0 = gate_probs[:, 0:1] -> broadcast to [SEQ, NUM_CLASSES]
    let gate_0 = b.add_narrow(gate_probs, 1, 0, 1, &[SEQ, 1]);
    let gate_0_bc = b.add_broadcast(gate_0, &[SEQ, NUM_CLASSES]);
    let weighted_table = b.add_binary_mul(gate_0_bc, table_out, &[SEQ, NUM_CLASSES]);

    // Extract gate_1 = gate_probs[:, 1:2] -> broadcast to [SEQ, NUM_CLASSES]
    let gate_1 = b.add_narrow(gate_probs, 1, 1, 1, &[SEQ, 1]);
    let gate_1_bc = b.add_broadcast(gate_1, &[SEQ, NUM_CLASSES]);
    let weighted_ocr = b.add_binary_mul(gate_1_bc, ocr_out, &[SEQ, NUM_CLASSES]);

    // Merge: gate_0 * table_out + gate_1 * ocr_out
    let out = b.add_binary_add(weighted_table, weighted_ocr, &[SEQ, NUM_CLASSES]);

    b.build(out)
        .expect("valid tier routing cross-pipeline kernel")
}

fn tier_routing_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // features
        TensorParamBinding::ConstantTensor(w(&[NUM_BRANCHES, HIDDEN])), // gate_weight
        TensorParamBinding::ConstantTensor(zeros(&[NUM_BRANCHES])), // gate_bias
        TensorParamBinding::ConstantTensor(w(&[NUM_CLASSES, HIDDEN])), // table_weight
        TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])), // table_bias
        TensorParamBinding::ConstantTensor(w(&[NUM_CLASSES, HIDDEN])), // ocr_weight
        TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])), // ocr_bias
    ]
}

/// Tier routing IBP: softmax gate dispatches to table/OCR heads.
///
/// Verifies the softmax-gated merge of sigmoid (table) and softmax (OCR) heads:
///   out = gate[0] * table_out + gate[1] * ocr_out
/// Analytically this is a convex combination (gate sums to 1, each head in
/// [0, 1]) so the true output is in [0, 1]. Under plain IBP the simplex
/// correlation gate[0] + gate[1] = 1 is untracked, so the sound bound is the
/// [0, 2] envelope of the sum of two independent [0, 1] products.
#[test]
fn test_cross_tier_routing_ibp() {
    let def = build_tier_routing_kernel();
    let bindings = tier_routing_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, HIDDEN], 5.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through tier routing pipeline");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ, NUM_CLASSES],
        "tier routing output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("cross tier routing IBP: [{lo_min}, {hi_max}]");

    // The *analytic* merge gate_0*table + gate_1*ocr is a convex combination of
    // [0,1]-bounded values (gate_0 + gate_1 = 1 via softmax), so the true output
    // is in [0, 1]. But plain IBP cannot track the simplex correlation
    // gate_0 + gate_1 = 1: it bounds each product independently as
    // gate_i*head_i in [0,1] and sums them, giving a sound IBP envelope of
    // [0, 2] (the analytic 1.0 upper would require CROWN). The observed upper
    // ~1.55 is a sound over-approximation within that [0, 2] envelope.
    let eps = 1e-3;
    assert!(
        lo_min >= 0.0 - eps,
        "tier routing: merged output >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 2.0 + eps,
        "tier routing: IBP sum-of-products envelope must be <= 2 (simplex \
         correlation untracked by IBP; analytic bound is 1.0), got {hi_max}"
    );
}
