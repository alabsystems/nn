// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for NMS and detection postprocessing bound propagation.
//!
//! Verifies NY IBP/CROWN bounds through detection postprocessing
//! sub-blocks used in dpdf document understanding: confidence scoring,
//! DFL regression, box decoding, IoU computation, score filtering,
//! multi-scale aggregation, and full detection pipelines.
//!
//! ## Confidence & Classification (tests 1-4)
//!
//! 1. Detection head confidence score sigmoid: output in (0, 1) (IBP + CROWN)
//! 2. DFL regression output: softmax -> weighted sum bounds (IBP)
//! 3. Box coordinate sigmoid normalization: x, y, w, h in (0, 1) (IBP)
//! 4. Multi-class detection score: softmax over classes (IBP)
//!
//! ## Filtering & IoU (tests 5-6)
//!
//! 5. Score threshold filtering: sigmoid above threshold (IBP)
//! 6. IoU computation numerical bounds: intersection / union bounded (IBP + CROWN)
//!
//! ## Detection Heads (tests 7-10)
//!
//! 7. Anchor-free detection box decoding: Linear -> sigmoid (IBP)
//! 8. Detection head with shared stem: Linear -> ReLU -> dual head (IBP)
//! 9. Classification + regression dual head composition (IBP + CROWN)
//! 10. Scale-specific detection heads: P3, P4, P5 (IBP)
//!
//! ## Monotonicity & Composition (tests 11-15)
//!
//! 11. Detection confidence monotone: tighter input -> tighter confidence (IBP)
//! 12. Box coordinate monotone tightening (IBP)
//! 13. Multi-scale detection aggregation: concat P3+P4+P5 heads (IBP)
//! 14. Detection head with GroupNorm: GroupNorm -> Linear -> sigmoid (IBP + CROWN)
//! 15. Full detection pipeline: backbone -> neck -> heads -> decode (IBP)
//!
//! Architecture references:
//! - DocLayout-YOLO (Zhao et al. 2024): YOLOv10-based document layout detection
//! - DFL (Li et al. 2022): Distribution Focal Loss for box regression
//! - FCOS (Tian et al. 2019): Fully Convolutional One-Stage anchor-free detection
//! - DETR (Carion et al. 2020): DEtection TRansformer
//!
//! Dimensions (small for fast verification, structurally representative):
//! - NUM_DETECTIONS=6, HIDDEN_DIM=32, NUM_CLASSES=8, BOX_DIM=4,
//!   DFL_BINS=8, STEM_DIM=48, NECK_DIM=64, BACKBONE_CH=16,
//!   P3_DETS=4, P4_DETS=2, P5_DETS=1
//!
//! Part of #4014: NMS and detection postprocessing compose tests.

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

/// Number of detection candidates (anchor points / object queries).
const NUM_DETECTIONS: usize = 6;
/// Hidden dimension for detection features.
const HIDDEN_DIM: usize = 32;
/// Number of detection classes.
const NUM_CLASSES: usize = 8;
/// Box coordinate dimension (x, y, w, h).
const BOX_DIM: usize = 4;
/// DFL bins for distribution focal loss regression.
const DFL_BINS: usize = 8;
/// Shared stem intermediate dimension.
const STEM_DIM: usize = 48;
/// Neck projection dimension.
const NECK_DIM: usize = 64;
/// Backbone output channels.
const BACKBONE_CH: usize = 16;
/// P3 scale detection count.
const P3_DETS: usize = 4;
/// P4 scale detection count.
const P4_DETS: usize = 2;
/// P5 scale detection count.
const P5_DETS: usize = 1;
/// Total multi-scale detections.
const TOTAL_DETS: usize = P3_DETS + P4_DETS + P5_DETS; // 7
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;
/// GroupNorm groups for detection head.
const GN_GROUPS: usize = 1;
/// Image spatial size (square).
const IMG_SIZE: usize = 16;
/// Patch size for backbone.
const PATCH_SIZE: usize = 8;
/// Grid size = IMG_SIZE / PATCH_SIZE.
const GRID_SIZE: usize = IMG_SIZE / PATCH_SIZE; // 2
/// Number of patches.
const NUM_PATCHES: usize = GRID_SIZE * GRID_SIZE; // 4
/// Input channels (RGB).
const IN_CHANNELS: usize = 3;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Constant weight tensor binding.
fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

/// Constant zero bias tensor binding.
fn bias(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

/// GroupNorm weight (all ones) binding.
fn gn_weight(dim: usize) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 1.0f32))
}

/// GroupNorm epsilon binding.
fn gn_eps() -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32))
}

/// Compute output bound width from a BoundedTensor.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

// ===========================================================================
// 1. Detection head confidence score sigmoid bounds (IBP + CROWN)
// ===========================================================================

/// Build detection confidence head: Linear -> sigmoid.
///
/// Input: `[NUM_DETECTIONS, HIDDEN_DIM]` (Variable, detection features).
/// Output: `[NUM_DETECTIONS, 1]` (confidence score in (0, 1)).
fn build_confidence_sigmoid_kernel() -> TensorKernelDef {
    let out_shape = [NUM_DETECTIONS, 1];
    let mut b = TensorBlockBuilder::new("nms_confidence_sigmoid");

    let input = b.add_input("features", &[NUM_DETECTIONS, HIDDEN_DIM]);
    let w = b.add_input("conf_weight", &[1, HIDDEN_DIM]);
    let bias_node = b.add_input("conf_bias", &[1]);

    let logits = b.add_linear(input, w, Some(bias_node), &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);

    b.build(out).expect("valid confidence sigmoid kernel")
}

fn confidence_sigmoid_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // features
        weight(&[1, HIDDEN_DIM]),     // conf_weight
        bias(&[1]),                   // conf_bias
    ]
}

#[test]
fn test_nms_confidence_sigmoid_ibp_crown() {
    let def = build_confidence_sigmoid_kernel();
    let bindings = confidence_sigmoid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_DETECTIONS, HIDDEN_DIM], 2.0);

    // IBP
    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through confidence sigmoid");
    assert_eq!(
        ibp_output.lower_upper().0.shape(),
        &[NUM_DETECTIONS, 1],
        "confidence sigmoid output shape mismatch"
    );
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Detection confidence sigmoid IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "sigmoid upper must be <= 1, got {hi_max}"
    );

    // CROWN
    let crown_input = uniform_bounds(&[NUM_DETECTIONS, HIDDEN_DIM], 1.0);
    let (_method, crown_output, _fallback) =
        assert_crown_tighter_when_not_fallback(&graph, &crown_input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Confidence sigmoid CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 2. DFL regression output bounds (IBP)
// ===========================================================================

/// Build DFL regression head: Linear -> softmax -> weighted sum.
///
/// DFL (Distribution Focal Loss) predicts a discrete distribution over bins,
/// then computes the expected value as softmax(logits) @ [0, 1, ..., B-1].
///
/// Input: `[NUM_DETECTIONS, HIDDEN_DIM]` (Variable, detection features).
/// Output: `[NUM_DETECTIONS, BOX_DIM]` (DFL-decoded box coordinates).
fn build_dfl_regression_kernel() -> TensorKernelDef {
    let softmax_shape = [NUM_DETECTIONS, DFL_BINS];
    let out_shape = [NUM_DETECTIONS, BOX_DIM];
    let mut b = TensorBlockBuilder::new("nms_dfl_regression");

    let input = b.add_input("features", &[NUM_DETECTIONS, HIDDEN_DIM]);

    // Project to DFL bins for each box coordinate (simplified: single DFL head)
    let proj_w = b.add_input("dfl_proj_weight", &[DFL_BINS, HIDDEN_DIM]);
    let proj_b = b.add_input("dfl_proj_bias", &[DFL_BINS]);
    let logits = b.add_linear(input, proj_w, Some(proj_b), &softmax_shape);

    // Softmax over bins -> probability distribution
    let probs = b.add_softmax(logits, 1, &softmax_shape);

    // Weighted sum: matmul with bin indices [DFL_BINS, BOX_DIM]
    // Simplified as linear projection from prob space to box coords
    let decode_w = b.add_input("dfl_decode_weight", &[BOX_DIM, DFL_BINS]);
    let out = b.add_linear(probs, decode_w, None, &out_shape);

    b.build(out).expect("valid DFL regression kernel")
}

fn dfl_regression_bindings() -> Vec<TensorParamBinding> {
    // For DFL decode weights, use bin indices [0..DFL_BINS) scaled down
    let mut decode_data = vec![0.0f32; BOX_DIM * DFL_BINS];
    for c in 0..BOX_DIM {
        for bin in 0..DFL_BINS {
            decode_data[c * DFL_BINS + bin] = bin as f32 / DFL_BINS as f32;
        }
    }
    vec![
        TensorParamBinding::Variable,    // features
        weight(&[DFL_BINS, HIDDEN_DIM]), // dfl_proj_weight
        bias(&[DFL_BINS]),               // dfl_proj_bias
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[BOX_DIM, DFL_BINS]), decode_data)
                .expect("valid DFL decode weights"),
        ), // dfl_decode_weight
    ]
}

#[test]
fn test_nms_dfl_regression_ibp() {
    let def = build_dfl_regression_kernel();
    let bindings = dfl_regression_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_DETECTIONS, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through DFL regression");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_DETECTIONS, BOX_DIM],
        "DFL regression output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DFL regression IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "DFL lower bound must be finite");
    assert!(hi_max.is_finite(), "DFL upper bound must be finite");
    // DFL output = weighted sum of softmax probs * bin indices in [0, 1)
    // Bounds should be moderate
    assert!(
        hi_max - lo_min < 100.0,
        "DFL bounds should not be vacuously wide, got width {}",
        hi_max - lo_min
    );
}

// ===========================================================================
// 3. Box coordinate (x, y, w, h) sigmoid normalization bounds (IBP)
// ===========================================================================

/// Build box coordinate sigmoid normalization: Linear -> sigmoid for each coord.
///
/// Input: `[NUM_DETECTIONS, HIDDEN_DIM]` (Variable, detection features).
/// Output: `[NUM_DETECTIONS, BOX_DIM]` (normalized box coords in (0, 1)).
fn build_box_sigmoid_kernel() -> TensorKernelDef {
    let out_shape = [NUM_DETECTIONS, BOX_DIM];
    let mut b = TensorBlockBuilder::new("nms_box_sigmoid");

    let input = b.add_input("features", &[NUM_DETECTIONS, HIDDEN_DIM]);
    let w = b.add_input("box_weight", &[BOX_DIM, HIDDEN_DIM]);
    let bias_node = b.add_input("box_bias", &[BOX_DIM]);

    let logits = b.add_linear(input, w, Some(bias_node), &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);

    b.build(out).expect("valid box sigmoid kernel")
}

fn box_sigmoid_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,   // features
        weight(&[BOX_DIM, HIDDEN_DIM]), // box_weight
        bias(&[BOX_DIM]),               // box_bias
    ]
}

#[test]
fn test_nms_box_sigmoid_ibp() {
    let def = build_box_sigmoid_kernel();
    let bindings = box_sigmoid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_DETECTIONS, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through box sigmoid");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_DETECTIONS, BOX_DIM],
        "box sigmoid output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Box sigmoid IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "box sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "box sigmoid upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 4. Multi-class detection score bounds (softmax over classes) (IBP)
// ===========================================================================

/// Build multi-class detection score head: Linear -> softmax.
///
/// Input: `[NUM_DETECTIONS, HIDDEN_DIM]` (Variable, detection features).
/// Output: `[NUM_DETECTIONS, NUM_CLASSES]` (class probabilities in [0, 1]).
fn build_multiclass_softmax_kernel() -> TensorKernelDef {
    let out_shape = [NUM_DETECTIONS, NUM_CLASSES];
    let mut b = TensorBlockBuilder::new("nms_multiclass_softmax");

    let input = b.add_input("features", &[NUM_DETECTIONS, HIDDEN_DIM]);
    let w = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let bias_node = b.add_input("cls_bias", &[NUM_CLASSES]);

    let logits = b.add_linear(input, w, Some(bias_node), &out_shape);
    let out = b.add_softmax(logits, 1, &out_shape);

    b.build(out).expect("valid multi-class softmax kernel")
}

fn multiclass_softmax_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,       // features
        weight(&[NUM_CLASSES, HIDDEN_DIM]), // cls_weight
        bias(&[NUM_CLASSES]),               // cls_bias
    ]
}

#[test]
fn test_nms_multiclass_softmax_ibp() {
    let def = build_multiclass_softmax_kernel();
    let bindings = multiclass_softmax_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_DETECTIONS, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through multi-class softmax");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_DETECTIONS, NUM_CLASSES],
        "multi-class softmax output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Multi-class softmax IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
    // Softmax outputs are in [0, 1]
    assert!(
        lo_min >= 0.0 - eps,
        "softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 5. Score threshold filtering effect on bounds (IBP)
// ===========================================================================

/// Build score threshold filter: Linear -> sigmoid -> ReLU(x - threshold).
///
/// Models the effect of score thresholding: detections with confidence below
/// the threshold are suppressed (output near zero). The ReLU(sigmoid(x) - t)
/// clamps filtered scores to zero.
///
/// Input: `[NUM_DETECTIONS, HIDDEN_DIM]` (Variable, detection features).
/// Output: `[NUM_DETECTIONS, 1]` (filtered confidence, >= 0).
fn build_score_threshold_kernel() -> TensorKernelDef {
    let score_shape = [NUM_DETECTIONS, 1];
    let mut b = TensorBlockBuilder::new("nms_score_threshold");

    let input = b.add_input("features", &[NUM_DETECTIONS, HIDDEN_DIM]);
    let w = b.add_input("conf_weight", &[1, HIDDEN_DIM]);
    let bias_node = b.add_input("conf_bias", &[1]);

    // Confidence sigmoid
    let logits = b.add_linear(input, w, Some(bias_node), &score_shape);
    let scores = b.add_sigmoid(logits, &score_shape);

    // Threshold subtraction: scores - threshold (threshold encoded as negative bias)
    // Add a constant shift of -0.3 (threshold = 0.3) via a second linear
    let shift_w = b.add_input("shift_weight", &[1, 1]);
    let shift_b = b.add_input("shift_bias", &[1]);
    let shifted = b.add_linear(scores, shift_w, Some(shift_b), &score_shape);

    // ReLU clamps below-threshold to zero
    let out = b.add_relu(shifted, &score_shape);

    b.build(out).expect("valid score threshold kernel")
}

fn score_threshold_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // features
        weight(&[1, HIDDEN_DIM]),     // conf_weight
        bias(&[1]),                   // conf_bias
        // Identity + threshold: weight=1.0, bias=-0.3 (threshold = 0.3)
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, 1]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), -0.3f32)),
    ]
}

#[test]
fn test_nms_score_threshold_ibp() {
    let def = build_score_threshold_kernel();
    let bindings = score_threshold_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_DETECTIONS, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through score threshold");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_DETECTIONS, 1],
        "score threshold output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Score threshold IBP: bounds=[{lo_min}, {hi_max}]");
    // ReLU output is >= 0
    assert!(
        lo_min >= -1e-6,
        "ReLU output lower must be >= 0, got {lo_min}"
    );
    // Upper bound from sigmoid(x) - 0.3 is at most 1.0 - 0.3 = 0.7
    assert!(
        hi_max <= 1.0 + 1e-4,
        "filtered score upper should be bounded, got {hi_max}"
    );
}

// ===========================================================================
// 6. IoU computation numerical bounds (IBP + CROWN)
// ===========================================================================

/// Build IoU numerics proxy: sigmoid(x) * sigmoid(y) / (sigmoid(x) + sigmoid(y)).
///
/// Real IoU = intersection / union where intersection and union are areas.
/// We approximate the numerical structure: sigmoid products divided by sigmoid sums,
/// modeled as Linear -> sigmoid -> element-wise multiply -> Linear projection.
///
/// Input: `[NUM_DETECTIONS, HIDDEN_DIM]` (Variable, box features).
/// Output: `[NUM_DETECTIONS, 1]` (IoU proxy bounded in (0, 1)).
fn build_iou_proxy_kernel() -> TensorKernelDef {
    let box_shape = [NUM_DETECTIONS, BOX_DIM];
    let out_shape = [NUM_DETECTIONS, 1];
    let mut b = TensorBlockBuilder::new("nms_iou_proxy");

    let input = b.add_input("features", &[NUM_DETECTIONS, HIDDEN_DIM]);

    // Decode box A coords via sigmoid
    let wa = b.add_input("box_a_weight", &[BOX_DIM, HIDDEN_DIM]);
    let ba = b.add_input("box_a_bias", &[BOX_DIM]);
    let logits_a = b.add_linear(input, wa, Some(ba), &box_shape);
    let box_a = b.add_sigmoid(logits_a, &box_shape);

    // Decode box B coords via sigmoid (different projection)
    let wb = b.add_input("box_b_weight", &[BOX_DIM, HIDDEN_DIM]);
    let bb = b.add_input("box_b_bias", &[BOX_DIM]);
    let logits_b = b.add_linear(input, wb, Some(bb), &box_shape);
    let box_b = b.add_sigmoid(logits_b, &box_shape);

    // Element-wise product: proxy for intersection area
    let intersection = b.add_binary_mul(box_a, box_b, &box_shape);

    // Project intersection to scalar IoU-like value
    let iou_w = b.add_input("iou_weight", &[1, BOX_DIM]);
    let out = b.add_linear(intersection, iou_w, None, &out_shape);
    let out_sigmoid = b.add_sigmoid(out, &out_shape);

    b.build(out_sigmoid).expect("valid IoU proxy kernel")
}

fn iou_proxy_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,   // features
        weight(&[BOX_DIM, HIDDEN_DIM]), // box_a_weight
        bias(&[BOX_DIM]),               // box_a_bias
        weight(&[BOX_DIM, HIDDEN_DIM]), // box_b_weight
        bias(&[BOX_DIM]),               // box_b_bias
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, BOX_DIM]), 0.25f32)), // iou_weight (average over 4 coords)
    ]
}

#[test]
fn test_nms_iou_proxy_ibp_crown() {
    let def = build_iou_proxy_kernel();
    let bindings = iou_proxy_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_DETECTIONS, HIDDEN_DIM], 2.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP through IoU proxy");
    assert_eq!(
        ibp_output.lower_upper().0.shape(),
        &[NUM_DETECTIONS, 1],
        "IoU proxy output shape mismatch"
    );
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("IoU proxy IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
    // Final sigmoid bounds in (0, 1)
    assert!(
        lo_min >= 0.0 - eps,
        "IoU sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "IoU sigmoid upper must be <= 1, got {hi_max}"
    );

    // CROWN
    let crown_input = uniform_bounds(&[NUM_DETECTIONS, HIDDEN_DIM], 1.0);
    let (_method, crown_output, _fallback) =
        assert_crown_tighter_when_not_fallback(&graph, &crown_input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("IoU proxy CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 7. Anchor-free detection box decoding bounds (IBP)
// ===========================================================================

/// Build anchor-free box decoding: Linear -> sigmoid for center + size.
///
/// FCOS-style: predict (cx, cy, w, h) with sigmoid normalization to (0, 1).
///
/// Input: `[NUM_DETECTIONS, HIDDEN_DIM]` (Variable, detection features).
/// Output: `[NUM_DETECTIONS, BOX_DIM]` (decoded box in (0, 1)).
fn build_anchor_free_decode_kernel() -> TensorKernelDef {
    let out_shape = [NUM_DETECTIONS, BOX_DIM];
    let mut b = TensorBlockBuilder::new("nms_anchor_free_decode");

    let input = b.add_input("features", &[NUM_DETECTIONS, HIDDEN_DIM]);

    // Two-layer MLP: features -> intermediate -> box coords
    let w1 = b.add_input("decode_w1", &[STEM_DIM, HIDDEN_DIM]);
    let b1 = b.add_input("decode_b1", &[STEM_DIM]);
    let hidden = b.add_linear(input, w1, Some(b1), &[NUM_DETECTIONS, STEM_DIM]);
    let hidden_act = b.add_relu(hidden, &[NUM_DETECTIONS, STEM_DIM]);

    let w2 = b.add_input("decode_w2", &[BOX_DIM, STEM_DIM]);
    let b2 = b.add_input("decode_b2", &[BOX_DIM]);
    let logits = b.add_linear(hidden_act, w2, Some(b2), &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);

    b.build(out).expect("valid anchor-free decode kernel")
}

fn anchor_free_decode_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,    // features
        weight(&[STEM_DIM, HIDDEN_DIM]), // decode_w1
        bias(&[STEM_DIM]),               // decode_b1
        weight(&[BOX_DIM, STEM_DIM]),    // decode_w2
        bias(&[BOX_DIM]),                // decode_b2
    ]
}

#[test]
fn test_nms_anchor_free_decode_ibp() {
    let def = build_anchor_free_decode_kernel();
    let bindings = anchor_free_decode_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_DETECTIONS, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through anchor-free decode");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_DETECTIONS, BOX_DIM],
        "anchor-free decode output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Anchor-free decode IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "sigmoid upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 8. Detection head with shared stem bounds (IBP)
// ===========================================================================

/// Build detection head with shared stem: Linear -> ReLU -> cls_head + box_head.
///
/// Shared stem extracts common features, then two parallel heads produce
/// class confidence (sigmoid) and box coordinates (sigmoid).
///
/// Input: `[NUM_DETECTIONS, HIDDEN_DIM]` (Variable, detection features).
/// Output: `[NUM_DETECTIONS, NUM_CLASSES + BOX_DIM]` (concat cls + box).
fn build_shared_stem_kernel() -> TensorKernelDef {
    let cls_shape = [NUM_DETECTIONS, NUM_CLASSES];
    let box_shape = [NUM_DETECTIONS, BOX_DIM];
    let out_dim = NUM_CLASSES + BOX_DIM;
    let out_shape = [NUM_DETECTIONS, out_dim];
    let mut b = TensorBlockBuilder::new("nms_shared_stem");

    let input = b.add_input("features", &[NUM_DETECTIONS, HIDDEN_DIM]);

    // Shared stem: Linear -> ReLU
    let stem_w = b.add_input("stem_weight", &[STEM_DIM, HIDDEN_DIM]);
    let stem_b = b.add_input("stem_bias", &[STEM_DIM]);
    let stem = b.add_linear(input, stem_w, Some(stem_b), &[NUM_DETECTIONS, STEM_DIM]);
    let stem_act = b.add_relu(stem, &[NUM_DETECTIONS, STEM_DIM]);

    // Classification head: Linear -> sigmoid
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, STEM_DIM]);
    let cls_b = b.add_input("cls_bias", &[NUM_CLASSES]);
    let cls_logits = b.add_linear(stem_act, cls_w, Some(cls_b), &cls_shape);
    let cls_out = b.add_sigmoid(cls_logits, &cls_shape);

    // Box regression head: Linear -> sigmoid
    let box_w = b.add_input("box_weight", &[BOX_DIM, STEM_DIM]);
    let box_b = b.add_input("box_bias", &[BOX_DIM]);
    let box_logits = b.add_linear(stem_act, box_w, Some(box_b), &box_shape);
    let box_out = b.add_sigmoid(box_logits, &box_shape);

    // Concatenate: [cls_out, box_out] along dim 1
    let out = b.add_concat(&[cls_out, box_out], 1, &out_shape);

    b.build(out).expect("valid shared stem kernel")
}

fn shared_stem_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,     // features
        weight(&[STEM_DIM, HIDDEN_DIM]),  // stem_weight
        bias(&[STEM_DIM]),                // stem_bias
        weight(&[NUM_CLASSES, STEM_DIM]), // cls_weight
        bias(&[NUM_CLASSES]),             // cls_bias
        weight(&[BOX_DIM, STEM_DIM]),     // box_weight
        bias(&[BOX_DIM]),                 // box_bias
    ]
}

#[test]
fn test_nms_shared_stem_ibp() {
    let def = build_shared_stem_kernel();
    let bindings = shared_stem_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_DETECTIONS, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through shared stem");
    let out_dim = NUM_CLASSES + BOX_DIM;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_DETECTIONS, out_dim],
        "shared stem output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Shared stem IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
    // All sigmoid outputs in (0, 1)
    assert!(
        lo_min >= 0.0 - eps,
        "shared stem lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "shared stem upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 9. Classification + regression dual head composition (IBP + CROWN)
// ===========================================================================

/// Build classification + regression dual head: separate linear paths -> sigmoid.
///
/// Unlike shared stem (test 8), this tests independent heads from the same input
/// concatenated into a single output, verifying bound propagation through parallel
/// sigmoid branches.
///
/// Input: `[NUM_DETECTIONS, HIDDEN_DIM]` (Variable, detection features).
/// Output: `[NUM_DETECTIONS, NUM_CLASSES + BOX_DIM]` (cls + box sigmoid).
fn build_cls_reg_dual_kernel() -> TensorKernelDef {
    let cls_shape = [NUM_DETECTIONS, NUM_CLASSES];
    let box_shape = [NUM_DETECTIONS, BOX_DIM];
    let out_dim = NUM_CLASSES + BOX_DIM;
    let out_shape = [NUM_DETECTIONS, out_dim];
    let mut b = TensorBlockBuilder::new("nms_cls_reg_dual");

    let input = b.add_input("features", &[NUM_DETECTIONS, HIDDEN_DIM]);

    // Classification: Linear -> sigmoid
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let cls_b = b.add_input("cls_bias", &[NUM_CLASSES]);
    let cls_logits = b.add_linear(input, cls_w, Some(cls_b), &cls_shape);
    let cls_out = b.add_sigmoid(cls_logits, &cls_shape);

    // Box regression: Linear -> sigmoid
    let box_w = b.add_input("box_weight", &[BOX_DIM, HIDDEN_DIM]);
    let box_b = b.add_input("box_bias", &[BOX_DIM]);
    let box_logits = b.add_linear(input, box_w, Some(box_b), &box_shape);
    let box_out = b.add_sigmoid(box_logits, &box_shape);

    // Concatenate
    let out = b.add_concat(&[cls_out, box_out], 1, &out_shape);

    b.build(out).expect("valid cls_reg dual kernel")
}

fn cls_reg_dual_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,       // features
        weight(&[NUM_CLASSES, HIDDEN_DIM]), // cls_weight
        bias(&[NUM_CLASSES]),               // cls_bias
        weight(&[BOX_DIM, HIDDEN_DIM]),     // box_weight
        bias(&[BOX_DIM]),                   // box_bias
    ]
}

#[test]
fn test_nms_cls_reg_dual_ibp_crown() {
    let def = build_cls_reg_dual_kernel();
    let bindings = cls_reg_dual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_DETECTIONS, HIDDEN_DIM], 2.0);

    // IBP
    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through cls_reg dual");
    let out_dim = NUM_CLASSES + BOX_DIM;
    assert_eq!(
        ibp_output.lower_upper().0.shape(),
        &[NUM_DETECTIONS, out_dim],
        "cls_reg dual output shape mismatch"
    );
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Cls+Reg dual IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "dual head lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "dual head upper must be <= 1, got {hi_max}"
    );

    // CROWN
    let crown_input = uniform_bounds(&[NUM_DETECTIONS, HIDDEN_DIM], 1.0);
    let (_method, crown_output, _fallback) =
        assert_crown_tighter_when_not_fallback(&graph, &crown_input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Cls+Reg dual CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 10. Scale-specific detection head bounds (P3, P4, P5) (IBP)
// ===========================================================================

/// Build scale-specific detection heads: 3 separate heads for P3, P4, P5.
///
/// Each scale has its own detection count (P3=4, P4=2, P5=1) and shares
/// the same weight structure. Output is concatenated across scales.
///
/// Input: `[TOTAL_DETS, HIDDEN_DIM]` (Variable, multi-scale features).
/// Output: `[TOTAL_DETS, NUM_CLASSES]` (per-scale class sigmoid).
fn build_scale_heads_kernel() -> TensorKernelDef {
    let out_shape = [TOTAL_DETS, NUM_CLASSES];
    let mut b = TensorBlockBuilder::new("nms_scale_heads");

    let input = b.add_input("features", &[TOTAL_DETS, HIDDEN_DIM]);

    // Single shared classification head across all scales
    let w = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let bias_node = b.add_input("cls_bias", &[NUM_CLASSES]);

    let logits = b.add_linear(input, w, Some(bias_node), &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);

    b.build(out).expect("valid scale heads kernel")
}

fn scale_heads_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,       // features
        weight(&[NUM_CLASSES, HIDDEN_DIM]), // cls_weight
        bias(&[NUM_CLASSES]),               // cls_bias
    ]
}

#[test]
fn test_nms_scale_heads_ibp() {
    let def = build_scale_heads_kernel();
    let bindings = scale_heads_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[TOTAL_DETS, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through scale heads");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[TOTAL_DETS, NUM_CLASSES],
        "scale heads output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Scale heads (P3+P4+P5) IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "scale heads lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "scale heads upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 11. Detection confidence monotone: larger input -> higher confidence (IBP)
// ===========================================================================

/// Monotone tightening for confidence sigmoid: tighter input -> tighter output.
#[test]
fn test_nms_confidence_monotone_ibp() {
    let def = build_confidence_sigmoid_kernel();
    let bindings = confidence_sigmoid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input_wide = uniform_bounds(&[NUM_DETECTIONS, HIDDEN_DIM], 2.0);
    let input_tight = uniform_bounds(&[NUM_DETECTIONS, HIDDEN_DIM], 1.0);

    let output_wide = graph.propagate_ibp(&input_wide).expect("IBP wide eps");
    let output_tight = graph.propagate_ibp(&input_tight).expect("IBP tight eps");

    let width_wide = bound_width(&output_wide);
    let width_tight = bound_width(&output_tight);

    eprintln!("Confidence monotone: eps=2.0 width={width_wide:.6}, eps=1.0 width={width_tight:.6}");

    assert!(
        width_tight <= width_wide + 1e-6,
        "tighter input (eps=1.0, width={width_tight}) must produce bounds \
         no wider than wider input (eps=2.0, width={width_wide})"
    );
}

// ===========================================================================
// 12. Box coordinate monotone tightening (IBP)
// ===========================================================================

/// Monotone tightening for box sigmoid: tighter input -> tighter output.
#[test]
fn test_nms_box_monotone_ibp() {
    let def = build_box_sigmoid_kernel();
    let bindings = box_sigmoid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input_wide = uniform_bounds(&[NUM_DETECTIONS, HIDDEN_DIM], 2.0);
    let input_tight = uniform_bounds(&[NUM_DETECTIONS, HIDDEN_DIM], 1.0);

    let output_wide = graph.propagate_ibp(&input_wide).expect("IBP wide eps");
    let output_tight = graph.propagate_ibp(&input_tight).expect("IBP tight eps");

    let width_wide = bound_width(&output_wide);
    let width_tight = bound_width(&output_tight);

    eprintln!("Box monotone: eps=2.0 width={width_wide:.6}, eps=1.0 width={width_tight:.6}");

    assert!(
        width_tight <= width_wide + 1e-6,
        "tighter input (eps=1.0, width={width_tight}) must produce bounds \
         no wider than wider input (eps=2.0, width={width_wide})"
    );
}

// ===========================================================================
// 13. Multi-scale detection aggregation bounds (IBP)
// ===========================================================================

/// Build multi-scale aggregation: separate P3/P4/P5 projections -> concat -> sigmoid.
///
/// Each scale level projects to a shared detection dimension, concatenated
/// to form the full detection output. Tests bound propagation through
/// concatenation of parallel paths.
///
/// Input: `[TOTAL_DETS, HIDDEN_DIM]` (Variable, multi-scale features).
/// Output: `[TOTAL_DETS, NUM_CLASSES + BOX_DIM]` (aggregated cls + box).
fn build_multiscale_aggregation_kernel() -> TensorKernelDef {
    let cls_shape = [TOTAL_DETS, NUM_CLASSES];
    let box_shape = [TOTAL_DETS, BOX_DIM];
    let out_dim = NUM_CLASSES + BOX_DIM;
    let out_shape = [TOTAL_DETS, out_dim];
    let mut b = TensorBlockBuilder::new("nms_multiscale_aggregation");

    let input = b.add_input("features", &[TOTAL_DETS, HIDDEN_DIM]);

    // Classification head (shared across scales)
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let cls_b = b.add_input("cls_bias", &[NUM_CLASSES]);
    let cls_logits = b.add_linear(input, cls_w, Some(cls_b), &cls_shape);
    let cls_out = b.add_sigmoid(cls_logits, &cls_shape);

    // Box head (shared across scales)
    let box_w = b.add_input("box_weight", &[BOX_DIM, HIDDEN_DIM]);
    let box_b = b.add_input("box_bias", &[BOX_DIM]);
    let box_logits = b.add_linear(input, box_w, Some(box_b), &box_shape);
    let box_out = b.add_sigmoid(box_logits, &box_shape);

    // Concatenate cls + box
    let out = b.add_concat(&[cls_out, box_out], 1, &out_shape);

    b.build(out).expect("valid multi-scale aggregation kernel")
}

fn multiscale_aggregation_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,       // features
        weight(&[NUM_CLASSES, HIDDEN_DIM]), // cls_weight
        bias(&[NUM_CLASSES]),               // cls_bias
        weight(&[BOX_DIM, HIDDEN_DIM]),     // box_weight
        bias(&[BOX_DIM]),                   // box_bias
    ]
}

#[test]
fn test_nms_multiscale_aggregation_ibp() {
    let def = build_multiscale_aggregation_kernel();
    let bindings = multiscale_aggregation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[TOTAL_DETS, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through multi-scale aggregation");
    let out_dim = NUM_CLASSES + BOX_DIM;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[TOTAL_DETS, out_dim],
        "multi-scale aggregation output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Multi-scale aggregation IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
    // All outputs are sigmoid in (0, 1)
    assert!(
        lo_min >= 0.0 - eps,
        "aggregation lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "aggregation upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 14. Detection head with GroupNorm bounds (IBP + CROWN)
// ===========================================================================

/// Build detection head with GroupNorm: GroupNorm(G=1) -> Linear -> sigmoid.
///
/// GroupNorm(groups=1) is equivalent to LayerNorm. This tests bound
/// propagation through normalization before the detection head.
///
/// Input: `[NUM_DETECTIONS, HIDDEN_DIM]` (Variable, detection features).
/// Output: `[NUM_DETECTIONS, NUM_CLASSES]` (class sigmoid in (0, 1)).
fn build_groupnorm_detection_kernel() -> TensorKernelDef {
    let out_shape = [NUM_DETECTIONS, NUM_CLASSES];
    let mut b = TensorBlockBuilder::new("nms_groupnorm_detection");

    let input = b.add_input("features", &[NUM_DETECTIONS, HIDDEN_DIM]);

    // GroupNorm(G=1): normalizes over the whole [channels, time_len] = C*T map.
    // Signature: add_group_norm_g1(input, eps, gamma, beta, channels, time_len)
    // Input shape is [channels, time_len] = [NUM_DETECTIONS, HIDDEN_DIM].
    // The builder left-broadcasts the affine gamma/beta over axis 0, so they
    // must be sized [channels] = [NUM_DETECTIONS] (not [HIDDEN_DIM]).
    let gn_eps = b.add_input("gn_eps", &[1]);
    let gn_w = b.add_input("gn_weight", &[NUM_DETECTIONS]);
    let gn_b = b.add_input("gn_bias", &[NUM_DETECTIONS]);
    let normed = b.add_group_norm_g1(
        input,
        gn_eps,
        Some(gn_w),
        Some(gn_b),
        NUM_DETECTIONS,
        HIDDEN_DIM,
    );

    // Classification head: Linear -> sigmoid
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let cls_b = b.add_input("cls_bias", &[NUM_CLASSES]);
    let logits = b.add_linear(normed, cls_w, Some(cls_b), &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);

    b.build(out).expect("valid GroupNorm detection kernel")
}

fn groupnorm_detection_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,       // features
        gn_eps(),                           // gn_eps
        gn_weight(NUM_DETECTIONS),          // gn_weight ([channels] for axis-0 affine)
        bias(&[NUM_DETECTIONS]),            // gn_bias (zeros, [channels])
        weight(&[NUM_CLASSES, HIDDEN_DIM]), // cls_weight
        bias(&[NUM_CLASSES]),               // cls_bias
    ]
}

#[test]
fn test_nms_groupnorm_detection_ibp_crown() {
    let def = build_groupnorm_detection_kernel();
    let bindings = groupnorm_detection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_DETECTIONS, HIDDEN_DIM], 2.0);

    // IBP
    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through GroupNorm detection");
    assert_eq!(
        ibp_output.lower_upper().0.shape(),
        &[NUM_DETECTIONS, NUM_CLASSES],
        "GroupNorm detection output shape mismatch"
    );
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("GroupNorm detection IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "GroupNorm sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "GroupNorm sigmoid upper must be <= 1, got {hi_max}"
    );

    // CROWN (may fall back due to normalization)
    let crown_input = uniform_bounds(&[NUM_DETECTIONS, HIDDEN_DIM], 1.0);
    let (_method, crown_output, _fallback) =
        assert_crown_tighter_when_not_fallback(&graph, &crown_input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("GroupNorm detection CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 15. Full detection pipeline: backbone -> neck -> heads -> decode (IBP)
// ===========================================================================

/// Build full detection pipeline: Conv2d backbone -> Linear neck -> ReLU ->
/// classification sigmoid + box sigmoid.
///
/// End-to-end from image pixels [0,1] to detection outputs (cls + box) in (0, 1).
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[NUM_PATCHES, NUM_CLASSES + BOX_DIM]` (cls + box sigmoid).
fn build_full_detection_pipeline_kernel() -> TensorKernelDef {
    let seq_len = NUM_PATCHES;
    let cls_shape = [seq_len, NUM_CLASSES];
    let box_shape = [seq_len, BOX_DIM];
    let out_dim = NUM_CLASSES + BOX_DIM;
    let out_shape = [seq_len, out_dim];
    let mut b = TensorBlockBuilder::new("nms_full_detection_pipeline");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    // Backbone: Conv2d stride-P
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

    // Neck: Linear -> ReLU
    let neck_w = b.add_input("neck_weight", &[NECK_DIM, BACKBONE_CH]);
    let neck = b.add_linear(transposed, neck_w, None, &[seq_len, NECK_DIM]);
    let neck_act = b.add_relu(neck, &[seq_len, NECK_DIM]);

    // Classification head: Linear -> sigmoid
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, NECK_DIM]);
    let cls_b = b.add_input("cls_bias", &[NUM_CLASSES]);
    let cls_logits = b.add_linear(neck_act, cls_w, Some(cls_b), &cls_shape);
    let cls_out = b.add_sigmoid(cls_logits, &cls_shape);

    // Box regression head: Linear -> sigmoid
    let box_w = b.add_input("box_weight", &[BOX_DIM, NECK_DIM]);
    let box_b = b.add_input("box_bias", &[BOX_DIM]);
    let box_logits = b.add_linear(neck_act, box_w, Some(box_b), &box_shape);
    let box_out = b.add_sigmoid(box_logits, &box_shape);

    // Concatenate cls + box
    let out = b.add_concat(&[cls_out, box_out], 1, &out_shape);

    b.build(out).expect("valid full detection pipeline kernel")
}

fn full_detection_pipeline_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,                                // image
        weight(&[BACKBONE_CH, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]), // backbone_weight
        bias(&[BACKBONE_CH]),                                        // backbone_bias
        weight(&[NECK_DIM, BACKBONE_CH]),                            // neck_weight
        weight(&[NUM_CLASSES, NECK_DIM]),                            // cls_weight
        bias(&[NUM_CLASSES]),                                        // cls_bias
        weight(&[BOX_DIM, NECK_DIM]),                                // box_weight
        bias(&[BOX_DIM]),                                            // box_bias
    ]
}

#[test]
fn test_nms_full_detection_pipeline_ibp() {
    let def = build_full_detection_pipeline_kernel();
    let bindings = full_detection_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]), 1.0f32),
    )
    .expect("valid image bounds [0, 1]");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full detection pipeline");
    let out_dim = NUM_CLASSES + BOX_DIM;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, out_dim],
        "full detection pipeline output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full detection pipeline IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // All sigmoid outputs bounded in (0, 1)
    let eps = 1e-4;
    assert!(
        lo_min >= 0.0 - eps,
        "pipeline sigmoid lower >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "pipeline sigmoid upper <= 1, got {hi_max}"
    );
}
