// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for post-processing pipeline bounds: NMS,
//! box decoding, score thresholding, text line merging, and table cell
//! assignment.
//!
//! These tests verify NY IBP and CROWN bound propagation through
//! the post-processing stages that follow detection model inference:
//! box decoding (center-to-corner format conversion), confidence filtering
//! via sigmoid/softmax, non-maximum suppression score filtering, multi-class
//! NMS composition, text line merging criteria, OCR decode output ranges,
//! table cell grid assignment, and the full detection-to-merge pipeline.
//!
//! ## Tests (14 tests)
//!
//!  1. **Box decoding center-to-corner bounds** — (cx,cy,w,h) -> (x1,y1,x2,y2) (IBP)
//!  2. **Score thresholding via sigmoid** — sigmoid confidence filtering (IBP)
//!  3. **NMS IoU threshold score propagation** — sigmoid -> threshold -> ReLU (IBP)
//!  4. **Multi-class NMS pipeline** — per-class sigmoid + threshold + ReLU (IBP)
//!  5. **Softmax class probability bounds** — softmax output in [0,1] (IBP)
//!  6. **Text line horizontal merge criterion** — width-ratio linear proxy (IBP)
//!  7. **Text line vertical merge criterion** — height-overlap linear proxy (IBP)
//!  8. **OCR CTC decode output bounds** — Linear -> softmax character probs (IBP)
//!  9. **Table cell grid assignment** — sigmoid row/col assignment in [0,1] (IBP)
//! 10. **Full detect -> NMS -> decode pipeline** — end-to-end sigmoid+thresh (IBP)
//! 11. **IBP vs CROWN on score thresholding** — CROWN tighter on thresh (IBP+CROWN)
//! 12. **IBP vs CROWN on box decoding** — CROWN tighter on affine decode (IBP+CROWN)
//! 13. **DFL box regression bounds** — softmax -> weighted sum in [0, bins-1] (IBP)
//! 14. **Detection -> OCR -> merge pipeline** — 3-stage composition (IBP)
//!
//! Architecture references:
//! - DETR (Carion et al. 2020): DEtection TRansformer box decoding
//! - NMS: Standard non-maximum suppression with IoU thresholding
//! - DFL (Li et al. 2022): Distribution Focal Loss for box regression
//! - CTC (Graves et al. 2006): Connectionist Temporal Classification decoding
//!
//! Dimensions (small for fast verification, structurally representative):
//! - NUM_QUERIES=8, NUM_CLASSES=4, BOX_DIM=4
//! - VOCAB_SIZE=16, SEQ_LEN=6, HIDDEN_DIM=8
//! - GRID_ROWS=4, GRID_COLS=4, DFL_BINS=4
//!
//! Part of #4213: Compose tests for post-processing pipeline bounds.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const NUM_QUERIES: usize = 8;
const NUM_CLASSES: usize = 4;
const BOX_DIM: usize = 4; // (cx, cy, w, h) or (x1, y1, x2, y2)
const VOCAB_SIZE: usize = 16;
const SEQ_LEN: usize = 6;
const HIDDEN_DIM: usize = 8;
const GRID_ROWS: usize = 4;
const GRID_COLS: usize = 4;
const DFL_BINS: usize = 4;
const W_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn w(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), W_MAG)
}

fn zeros(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 0.0f32)
}

// ===========================================================================
// 1. Box decoding center-to-corner bounds (IBP)
// ===========================================================================

/// DETR box decoding: (cx, cy, w, h) -> (x1, y1, x2, y2) via affine transform.
///
/// Modeled as: sigmoid(pred) gives normalized (cx,cy,w,h) in [0,1],
/// then affine scaling to pixel coordinates. The affine transform is:
///   x1 = (cx - w/2) * img_w,  y1 = (cy - h/2) * img_h
///   x2 = (cx + w/2) * img_w,  y2 = (cy + h/2) * img_h
///
/// We model the sigmoid + linear scaling as: sigmoid -> mul(scale) + add(shift).
#[test]
fn test_postprocess_box_decode_center_to_corner_ibp() {
    let shape = [NUM_QUERIES, BOX_DIM];

    let mut b = TensorBlockBuilder::new("postproc_box_decode");
    let input = b.add_input("raw_box_pred", &shape);

    // Sigmoid normalizes predictions to [0, 1]
    let normed = b.add_sigmoid(input, &shape);

    // Scale by image dimensions (modeled as constant multiply)
    // For simplicity: scale all by a constant factor (e.g., 640)
    let scale = b.add_input("img_scale", &shape);
    let scaled = b.add_binary_mul(normed, scale, &shape);

    // Offset for center-to-corner conversion (add half-width shift)
    let offset = b.add_input("box_offset", &shape);
    let out = b.add_binary_add(scaled, offset, &shape);
    let def = b.build(out).expect("valid box decode kernel");

    let img_scale = 640.0f32;
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&shape), img_scale)),
        TensorParamBinding::ConstantTensor(zeros(&shape)),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&shape, 5.0);

    let output = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through box decode");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Postprocess box decode IBP: [{lo_min:.4}, {hi_max:.4}]");
    // sigmoid in [0,1] * 640 + 0 => output in [0, 640]
    assert!(lo_min >= -1e-2, "box decode lower >= 0, got {lo_min}");
    assert!(
        hi_max <= img_scale + 1e-2,
        "box decode upper <= {img_scale}, got {hi_max}"
    );
}

// ===========================================================================
// 2. Score thresholding via sigmoid (IBP)
// ===========================================================================

/// Sigmoid confidence: raw logits -> sigmoid -> scores in [0, 1].
#[test]
fn test_postprocess_score_threshold_sigmoid_ibp() {
    let shape = [NUM_QUERIES, NUM_CLASSES];

    let mut b = TensorBlockBuilder::new("postproc_score_sigmoid");
    let input = b.add_input("cls_logits", &shape);
    let out = b.add_sigmoid(input, &shape);
    let def = b.build(out).expect("valid sigmoid scoring kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&shape, 6.0);

    let output = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through sigmoid");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Postprocess sigmoid scoring IBP: [{lo_min:.6}, {hi_max:.6}]");
    let eps = 1e-5;
    assert!(lo_min >= 0.0 - eps, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + eps, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 3. NMS IoU threshold score propagation (IBP)
// ===========================================================================

/// NMS score filter: sigmoid(logits) - threshold, clamped by ReLU.
/// After ReLU, only scores above threshold survive as positive values.
#[test]
fn test_postprocess_nms_iou_threshold_score_ibp() {
    let shape = [NUM_QUERIES, NUM_CLASSES];

    let mut b = TensorBlockBuilder::new("postproc_nms_score_thresh");
    let input = b.add_input("cls_logits", &shape);
    let conf = b.add_sigmoid(input, &shape);

    // Subtract threshold (modeled as add with negative constant)
    let thresh = b.add_input("neg_threshold", &shape);
    let diff = b.add_binary_add(conf, thresh, &shape);
    let out = b.add_relu(diff, &shape);
    let def = b.build(out).expect("valid NMS score threshold kernel");

    // Threshold = 0.5 => subtract 0.5 from sigmoid outputs
    let thresh_val = -0.5f32;
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&shape), thresh_val)),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&shape, 5.0);

    let output = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through NMS thresh");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Postprocess NMS score threshold IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-5, "ReLU output lower >= 0, got {lo_min}");
    // Max is sigmoid(5) - 0.5 = ~0.493, but IBP may widen
    assert!(hi_max <= 1.01, "NMS upper <= 1, got {hi_max}");
}

// ===========================================================================
// 4. Multi-class NMS pipeline (IBP)
// ===========================================================================

/// Per-class NMS: separate sigmoid + threshold + ReLU per class channel,
/// then combine via addition (proxy for per-class filtering).
#[test]
fn test_postprocess_multiclass_nms_pipeline_ibp() {
    let flat = [NUM_QUERIES, NUM_CLASSES];

    let mut b = TensorBlockBuilder::new("postproc_multiclass_nms");
    let input = b.add_input("cls_logits", &flat);

    // Sigmoid confidence
    let conf = b.add_sigmoid(input, &flat);

    // Per-class threshold (different thresholds per class)
    let thresh = b.add_input("class_thresholds", &flat);
    let diff = b.add_binary_add(conf, thresh, &flat);

    // ReLU: zero out below-threshold
    let filtered = b.add_relu(diff, &flat);

    // Objectness gate: multiply by objectness score
    let obj_logits = b.add_input("obj_logits", &[NUM_QUERIES, 1]);
    let obj_score = b.add_sigmoid(obj_logits, &[NUM_QUERIES, 1]);

    // Broadcast objectness across classes
    let obj_broad = b.add_broadcast(obj_score, &flat);
    let out = b.add_binary_mul(filtered, obj_broad, &flat);
    let def = b.build(out).expect("valid multi-class NMS kernel");

    // Per-class thresholds: -0.3 for all
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&flat), -0.3f32)),
        TensorParamBinding::ConstantTensor(zeros(&[NUM_QUERIES, 1])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&flat, 5.0);

    let output = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP multi-class NMS");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &flat);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Postprocess multi-class NMS IBP: [{lo_min:.6}, {hi_max:.6}]");
    // ReLU * sigmoid => output >= 0
    assert!(lo_min >= -1e-4, "multi-class NMS lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.01, "multi-class NMS upper <= 1, got {hi_max}");
}

// ===========================================================================
// 5. Softmax class probability bounds (IBP)
// ===========================================================================

/// Softmax class assignment: raw logits -> softmax -> probabilities in [0,1].
#[test]
fn test_postprocess_softmax_class_probability_ibp() {
    let shape = [NUM_QUERIES, NUM_CLASSES];

    let mut b = TensorBlockBuilder::new("postproc_softmax_cls");
    let input = b.add_input("cls_logits", &shape);
    let out = b.add_softmax(input, -1, &shape);
    let def = b.build(out).expect("valid softmax cls kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&shape, 4.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP softmax cls");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Postprocess softmax class probs IBP: [{lo_min:.6}, {hi_max:.6}]");
    let eps = 1e-4;
    assert!(lo_min >= 0.0 - eps, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + eps, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 6. Text line horizontal merge criterion (IBP)
// ===========================================================================

/// Horizontal merge: boxes close in x-axis get merged. Modeled as:
/// Linear projection of box pair features -> sigmoid merge score.
/// Input: concatenated [x1_a, x2_a, w_a, x1_b, x2_b, w_b] per pair.
#[test]
fn test_postprocess_text_line_horizontal_merge_ibp() {
    let num_pairs = 6;
    let pair_feat_dim = 6; // x1_a, x2_a, w_a, x1_b, x2_b, w_b
    let shape = [num_pairs, pair_feat_dim];

    let mut b = TensorBlockBuilder::new("postproc_hmerge");
    let input = b.add_input("pair_features", &shape);

    // Linear projection to merge score
    let merge_w = b.add_input("merge_w", &[1, pair_feat_dim]);
    let merge_b = b.add_input("merge_b", &[1]);
    let proj = b.add_linear(input, merge_w, Some(merge_b), &[num_pairs, 1]);
    let out = b.add_sigmoid(proj, &[num_pairs, 1]);
    let def = b.build(out).expect("valid hmerge kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[1, pair_feat_dim])),
        TensorParamBinding::ConstantTensor(zeros(&[1])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    // Box coordinates normalized in [0, 1]
    let input_bounds = uniform_bounds(&shape, 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP hmerge");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[num_pairs, 1]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Postprocess text line hmerge IBP: [{lo_min:.6}, {hi_max:.6}]");
    let eps = 1e-5;
    assert!(lo_min >= 0.0 - eps, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + eps, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 7. Text line vertical merge criterion (IBP)
// ===========================================================================

/// Vertical merge: boxes close in y-axis with height overlap. Modeled as:
/// Linear projection of vertical pair features -> sigmoid merge score.
/// Input: [y1_a, y2_a, h_a, y1_b, y2_b, h_b] per pair.
#[test]
fn test_postprocess_text_line_vertical_merge_ibp() {
    let num_pairs = 6;
    let pair_feat_dim = 6; // y1_a, y2_a, h_a, y1_b, y2_b, h_b
    let shape = [num_pairs, pair_feat_dim];

    let mut b = TensorBlockBuilder::new("postproc_vmerge");
    let input = b.add_input("pair_features", &shape);

    // Linear projection to merge score
    let merge_w = b.add_input("merge_w", &[1, pair_feat_dim]);
    let merge_b = b.add_input("merge_b", &[1]);
    let proj = b.add_linear(input, merge_w, Some(merge_b), &[num_pairs, 1]);
    let out = b.add_sigmoid(proj, &[num_pairs, 1]);
    let def = b.build(out).expect("valid vmerge kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[1, pair_feat_dim])),
        TensorParamBinding::ConstantTensor(zeros(&[1])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&shape, 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP vmerge");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[num_pairs, 1]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Postprocess text line vmerge IBP: [{lo_min:.6}, {hi_max:.6}]");
    let eps = 1e-5;
    assert!(lo_min >= 0.0 - eps, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + eps, "sigmoid upper <= 1, got {hi_max}");
}

// ===========================================================================
// 8. OCR CTC decode output bounds (IBP)
// ===========================================================================

/// CTC decode: Linear(HIDDEN_DIM, VOCAB_SIZE) -> softmax -> character probs.
/// Verifies character probability distribution is in [0, 1].
#[test]
fn test_postprocess_ocr_ctc_decode_output_ibp() {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let out_shape = [SEQ_LEN, VOCAB_SIZE];

    let mut b = TensorBlockBuilder::new("postproc_ctc_decode");
    let input = b.add_input("encoder_hidden", &shape);
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);
    let logits = b.add_linear(input, ctc_w, Some(ctc_b), &out_shape);
    let out = b.add_softmax(logits, -1, &out_shape);
    let def = b.build(out).expect("valid CTC decode kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[VOCAB_SIZE, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[VOCAB_SIZE])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&shape, 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP CTC decode");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &out_shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Postprocess CTC decode IBP: [{lo_min:.6}, {hi_max:.6}]");
    let eps = 1e-4;
    assert!(lo_min >= 0.0 - eps, "CTC softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + eps, "CTC softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 9. Table cell grid assignment (IBP)
// ===========================================================================

/// Table cell assignment: box features -> Linear -> sigmoid row/col scores.
/// Each detection gets a [GRID_ROWS + GRID_COLS] sigmoid vector for
/// soft assignment to grid positions.
#[test]
fn test_postprocess_table_cell_grid_assignment_ibp() {
    let grid_dim = GRID_ROWS + GRID_COLS; // 8
    let feat_dim = BOX_DIM + NUM_CLASSES; // 8 (box coords + class features)
    let in_shape = [NUM_QUERIES, feat_dim];
    let out_shape = [NUM_QUERIES, grid_dim];

    let mut b = TensorBlockBuilder::new("postproc_table_grid_assign");
    let input = b.add_input("det_features", &in_shape);

    // Linear projection to grid assignment logits
    let grid_w = b.add_input("grid_w", &[grid_dim, feat_dim]);
    let grid_b = b.add_input("grid_b", &[grid_dim]);
    let logits = b.add_linear(input, grid_w, Some(grid_b), &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);
    let def = b.build(out).expect("valid grid assignment kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[grid_dim, feat_dim])),
        TensorParamBinding::ConstantTensor(zeros(&[grid_dim])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&in_shape, 1.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP grid assign");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &out_shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Postprocess table grid assign IBP: [{lo_min:.6}, {hi_max:.6}]");
    let eps = 1e-5;
    assert!(
        lo_min >= 0.0 - eps,
        "grid assign sigmoid lower >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "grid assign sigmoid upper <= 1, got {hi_max}"
    );

    // Per-query, per-grid-position check
    let (lo_arr, hi_arr) = output.lower_upper();
    for q in 0..NUM_QUERIES {
        for g in 0..grid_dim {
            let l = lo_arr[[q, g]];
            let h = hi_arr[[q, g]];
            assert!(
                l >= 0.0 - eps && h <= 1.0 + eps,
                "grid[{q},{g}] out of [0,1]: [{l}, {h}]"
            );
        }
    }
}

// ===========================================================================
// 10. Full detect -> NMS -> decode pipeline (IBP)
// ===========================================================================

/// End-to-end: raw detection features -> sigmoid cls + sigmoid box ->
/// threshold -> ReLU -> softmax class assignment.
#[test]
fn test_postprocess_full_detect_nms_decode_pipeline_ibp() {
    let feat_dim = HIDDEN_DIM;
    let in_shape = [NUM_QUERIES, feat_dim];
    let cls_shape = [NUM_QUERIES, NUM_CLASSES];
    let box_shape = [NUM_QUERIES, BOX_DIM];

    let mut b = TensorBlockBuilder::new("postproc_full_pipeline");
    let input = b.add_input("det_features", &in_shape);

    // Classification branch: Linear -> sigmoid
    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, feat_dim]);
    let cls_b = b.add_input("cls_b", &[NUM_CLASSES]);
    let cls_logits = b.add_linear(input, cls_w, Some(cls_b), &cls_shape);
    let cls_conf = b.add_sigmoid(cls_logits, &cls_shape);

    // NMS threshold filter: sigmoid - thresh -> ReLU
    let thresh = b.add_input("neg_threshold", &cls_shape);
    let diff = b.add_binary_add(cls_conf, thresh, &cls_shape);
    let filtered = b.add_relu(diff, &cls_shape);

    // Box branch: Linear -> sigmoid (normalized coordinates)
    let box_w = b.add_input("box_w", &[BOX_DIM, feat_dim]);
    let box_b = b.add_input("box_b", &[BOX_DIM]);
    let box_logits = b.add_linear(input, box_w, Some(box_b), &box_shape);
    let box_coords = b.add_sigmoid(box_logits, &box_shape);

    // Combine cls + box via concatenation -> final output
    let combined_shape = [NUM_QUERIES, NUM_CLASSES + BOX_DIM];
    let combined = b.add_concat(&[filtered, box_coords], 1, &combined_shape);

    // Final projection to produce detection output
    let out_dim = NUM_CLASSES + BOX_DIM;
    let final_w = b.add_input("final_w", &[out_dim, out_dim]);
    let out = b.add_linear(combined, final_w, None, &[NUM_QUERIES, out_dim]);
    let def = b.build(out).expect("valid full pipeline kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[NUM_CLASSES, feat_dim])),
        TensorParamBinding::ConstantTensor(zeros(&[NUM_CLASSES])),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&cls_shape), -0.5f32)),
        TensorParamBinding::ConstantTensor(w(&[BOX_DIM, feat_dim])),
        TensorParamBinding::ConstantTensor(zeros(&[BOX_DIM])),
        TensorParamBinding::ConstantTensor(w(&[out_dim, out_dim])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&in_shape, 2.0);

    let output = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP full pipeline");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, out_dim]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Postprocess full pipeline IBP: [{lo_min:.4}, {hi_max:.4}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
    let width = hi_max - lo_min;
    assert!(width < 1e6, "full pipeline bounds vacuously wide: {width}");
}

// ===========================================================================
// 11. IBP vs CROWN on score thresholding (IBP + CROWN)
// ===========================================================================

/// CROWN should produce tighter bounds on the sigmoid -> threshold -> ReLU
/// pipeline than IBP alone.
#[test]
fn test_postprocess_crown_score_thresholding() {
    let shape = [NUM_QUERIES, NUM_CLASSES];

    let mut b = TensorBlockBuilder::new("postproc_crown_score_thresh");
    let input = b.add_input("cls_logits", &shape);
    let conf = b.add_sigmoid(input, &shape);
    let thresh = b.add_input("neg_threshold", &shape);
    let diff = b.add_binary_add(conf, thresh, &shape);
    let out = b.add_relu(diff, &shape);
    let def = b.build(out).expect("valid CROWN score threshold kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&shape), -0.25f32)),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&shape, 3.0);

    // IBP pass
    let ibp_output = graph.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&ibp_output);

    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    eprintln!("Postprocess score thresh IBP: [{ibp_lo:.6}, {ibp_hi:.6}]");

    // CROWN pass
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    eprintln!("Postprocess score thresh CROWN: method={method:?}, [{crown_lo:.6}, {crown_hi:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 12. IBP vs CROWN on box decoding (IBP + CROWN)
// ===========================================================================

/// Box decoding (sigmoid + affine scale) is purely piecewise-linear after
/// sigmoid linearization, so CROWN should produce tight bounds.
#[test]
fn test_postprocess_crown_box_decoding() {
    let shape = [NUM_QUERIES, BOX_DIM];

    let mut b = TensorBlockBuilder::new("postproc_crown_box_decode");
    let input = b.add_input("raw_box_pred", &shape);
    let normed = b.add_sigmoid(input, &shape);
    let scale = b.add_input("img_scale", &shape);
    let out = b.add_binary_mul(normed, scale, &shape);
    let def = b.build(out).expect("valid CROWN box decode kernel");

    let img_scale = 640.0f32;
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&shape), img_scale)),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&shape, 3.0);

    // IBP pass
    let ibp_output = graph.propagate_ibp(&input_bounds).expect("IBP");
    assert_bounds_valid(&ibp_output);

    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    eprintln!("Postprocess box decode IBP: [{ibp_lo:.4}, {ibp_hi:.4}]");

    // CROWN pass
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    eprintln!("Postprocess box decode CROWN: method={method:?}, [{crown_lo:.4}, {crown_hi:.4}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    // Sigmoid * 640 => output in [0, 640]
    assert!(crown_lo >= -1e-2, "CROWN box lower >= 0, got {crown_lo}");
    assert!(
        crown_hi <= img_scale + 1e-2,
        "CROWN box upper <= {img_scale}, got {crown_hi}"
    );
}

// ===========================================================================
// 13. DFL box regression bounds (IBP)
// ===========================================================================

/// DFL (Distribution Focal Loss): softmax over bins -> weighted sum.
/// Output per box side is in [0, DFL_BINS - 1].
#[test]
fn test_postprocess_dfl_box_regression_ibp() {
    let num_sides = 4; // cx, cy, w, h
    let input_dim = DFL_BINS * num_sides;
    let flat_shape = [NUM_QUERIES, input_dim];

    let mut b = TensorBlockBuilder::new("postproc_dfl_box_reg");
    let input = b.add_input("box_logits", &flat_shape);

    // Reshape to [NUM_QUERIES * 4, DFL_BINS] for per-side softmax
    let reshape_shape = [NUM_QUERIES * num_sides, DFL_BINS];
    let reshaped = b.add_reshape(input, &reshape_shape);
    let softmax = b.add_softmax(reshaped, -1, &reshape_shape);

    // Weighted sum: [N*4, DFL_BINS] x [DFL_BINS, 1] -> [N*4, 1]
    let proj_w = b.add_input("dfl_proj", &[DFL_BINS, 1]);
    let proj_out = b.add_matmul(softmax, proj_w, false, None, &[NUM_QUERIES * num_sides, 1]);

    // Reshape to [NUM_QUERIES, 4]
    let out = b.add_reshape(proj_out, &[NUM_QUERIES, num_sides]);
    let def = b.build(out).expect("valid DFL box regression kernel");

    let bins_data: Vec<f32> = (0..DFL_BINS).map(|i| i as f32).collect();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[DFL_BINS, 1]), bins_data).expect("valid bins"),
        ),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&flat_shape, 3.0);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP DFL");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, num_sides]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Postprocess DFL regression IBP: [{lo_min:.6}, {hi_max:.6}]");
    let eps = 1e-4;
    let max_bin = (DFL_BINS - 1) as f32;
    assert!(lo_min >= 0.0 - eps, "DFL lower >= 0, got {lo_min}");
    assert!(
        hi_max <= max_bin + eps,
        "DFL upper <= {max_bin}, got {hi_max}"
    );
}

// ===========================================================================
// 14. Detection -> OCR -> merge pipeline (IBP)
// ===========================================================================

/// 3-stage composition: detection features -> sigmoid cls + box decode ->
/// OCR features via Linear bridge -> CTC softmax output.
/// Tests that bounds propagate soundly through the full post-processing chain.
#[test]
fn test_postprocess_detect_ocr_merge_pipeline_ibp() {
    let feat_dim = HIDDEN_DIM;
    let det_shape = [NUM_QUERIES, feat_dim];
    let cls_shape = [NUM_QUERIES, NUM_CLASSES];
    let ocr_seq_shape = [SEQ_LEN, HIDDEN_DIM];
    let ctc_shape = [SEQ_LEN, VOCAB_SIZE];

    let mut b = TensorBlockBuilder::new("postproc_detect_ocr_merge");
    let input = b.add_input("det_features", &det_shape);

    // Stage 1: Detection cls sigmoid
    let cls_w = b.add_input("cls_w", &[NUM_CLASSES, feat_dim]);
    let cls_logits = b.add_linear(input, cls_w, None, &cls_shape);
    let cls_conf = b.add_sigmoid(cls_logits, &cls_shape);

    // NMS threshold filter
    let thresh = b.add_input("neg_threshold", &cls_shape);
    let filtered = b.add_binary_add(cls_conf, thresh, &cls_shape);
    let nms_out = b.add_relu(filtered, &cls_shape);

    // Stage 2: Bridge detection -> OCR via linear projection.
    // MatMul requires both operands rank >= 2, so feed the flattened cls vector
    // as a [K, 1] column: bridge_w [M, K] @ bridge_in [K, 1] -> [M, 1], then
    // reshape the M = SEQ_LEN*HIDDEN_DIM result into the OCR sequence layout.
    let bridge_in = b.add_reshape(nms_out, &[NUM_QUERIES * NUM_CLASSES, 1]);
    let bridge_w = b.add_input(
        "bridge_w",
        &[SEQ_LEN * HIDDEN_DIM, NUM_QUERIES * NUM_CLASSES],
    );
    let bridge_out = b.add_matmul(bridge_w, bridge_in, false, None, &[SEQ_LEN * HIDDEN_DIM, 1]);
    let ocr_input = b.add_reshape(bridge_out, &ocr_seq_shape);

    // Stage 3: CTC head -> softmax
    let ctc_w = b.add_input("ctc_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b = b.add_input("ctc_b", &[VOCAB_SIZE]);
    let ctc_logits = b.add_linear(ocr_input, ctc_w, Some(ctc_b), &ctc_shape);
    let out = b.add_softmax(ctc_logits, -1, &ctc_shape);
    let def = b.build(out).expect("valid detect-ocr-merge kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[NUM_CLASSES, feat_dim])),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&cls_shape), -0.3f32)),
        TensorParamBinding::ConstantTensor(w(&[SEQ_LEN * HIDDEN_DIM, NUM_QUERIES * NUM_CLASSES])),
        TensorParamBinding::ConstantTensor(w(&[VOCAB_SIZE, HIDDEN_DIM])),
        TensorParamBinding::ConstantTensor(zeros(&[VOCAB_SIZE])),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&det_shape, 2.0);

    let output = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP detect-ocr-merge");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &ctc_shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Postprocess detect->OCR->merge IBP: [{lo_min:.6}, {hi_max:.6}]");
    // Final softmax output must be in [0, 1]
    let eps = 1e-4;
    assert!(
        lo_min >= 0.0 - eps,
        "pipeline softmax lower >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "pipeline softmax upper <= 1, got {hi_max}"
    );
}
