// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for dpdf table and form extraction models.
//!
//! Verifies IBP and CROWN bound propagation through table extraction,
//! form field detection, layout analysis, and coordinate normalization
//! sub-blocks used in dpdf document understanding.
//!
//! ## Table Cell Detection (tests 1-3)
//!
//! 1. Cell detector sigmoid: object detection confidence in (0, 1) (IBP + CROWN)
//! 2. Cell bounding box regression: Linear -> sigmoid for [x, y, w, h] (IBP)
//! 3. Cell detector dual-head: cls + bbox sigmoid combined (IBP + CROWN)
//!
//! ## Table Structure Recognition (tests 4-6)
//!
//! 4. Row classifier softmax: row-id assignment per cell (IBP)
//! 5. Column classifier softmax: column-id assignment per cell (IBP)
//! 6. Row-column joint head: row + column softmax concatenated (IBP + CROWN)
//!
//! ## Table Content OCR (tests 7-9)
//!
//! 7. Cell OCR feature extractor: Linear -> ReLU -> Linear (IBP)
//! 8. Cell OCR CTC head: Linear -> softmax char distribution (IBP + CROWN)
//! 9. Cell OCR pipeline: feature extractor -> CTC softmax end-to-end (IBP)
//!
//! ## Form Field Detection and Classification (tests 10-12)
//!
//! 10. Form field type classifier: softmax over field types (IBP + CROWN)
//! 11. Form field value extractor: Linear -> sigmoid confidence (IBP)
//! 12. Form key-value pair head: type + value + confidence combined (IBP + CROWN)
//!
//! ## Layout Analysis (tests 13-15)
//!
//! 13. Column count predictor: softmax over column counts (IBP)
//! 14. Reading order predictor: Linear -> sigmoid pairwise ordering (IBP + CROWN)
//! 15. Layout region classifier: softmax over region types (IBP + CROWN)
//!
//! ## Coordinate Normalization (tests 16-18)
//!
//! 16. Coordinate normalization: Linear -> sigmoid mapping to [0, 1] (IBP)
//! 17. Coordinate de-normalization: Linear scaling from [0, 1] (IBP)
//! 18. Normalized coord pipeline: detect -> normalize -> classify (IBP + CROWN)
//!
//! Architecture references:
//! - Table Transformer (Smock et al. 2022): DETR-based table detection/structure
//! - DETR (Carion et al. 2020): DEtection TRansformer
//! - PaddleOCR (Baidu): Production OCR with DB detector + SVTR recognizer
//! - LayoutLMv3 (Huang et al. 2022): Multi-modal document layout analysis
//!
//! Dimensions (small for fast verification, structurally representative):
//! - NUM_CELLS=8, HIDDEN_DIM=32, NUM_CLS=5, MAX_ROWS=6, MAX_COLS=6
//! - VOCAB_SIZE=64, NUM_FIELD_TYPES=6, MAX_COLUMNS=4, NUM_REGIONS=8
//!
//! Part of #4320: Compose tests for table/form extraction models.

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

/// Number of detected cell / region candidates.
const NUM_CELLS: usize = 8;
/// Hidden dimension from decoder/backbone output.
const HIDDEN_DIM: usize = 32;
/// Number of detection classes (table cell types).
const NUM_CLS: usize = 5;
/// Maximum number of rows for row classification.
const MAX_ROWS: usize = 6;
/// Maximum number of columns for column classification.
const MAX_COLS: usize = 6;
/// OCR vocabulary size for CTC decoding.
const VOCAB_SIZE: usize = 64;
/// Intermediate feature dimension for OCR feature extractor.
const OCR_FEAT_DIM: usize = 16;
/// Number of form field types (text, checkbox, radio, dropdown, signature, date).
const NUM_FIELD_TYPES: usize = 6;
/// Maximum number of document columns for layout analysis.
const MAX_COLUMNS: usize = 4;
/// Number of layout region types (text, table, figure, header, footer, list, caption, other).
const NUM_REGIONS: usize = 8;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute output bound width from a BoundedTensor.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

/// Assert sigmoid output is in [0, 1] (within numerical tolerance).
fn assert_sigmoid_range(bounds: &BoundedTensor, label: &str) {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "{label}: sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "{label}: sigmoid upper must be <= 1, got {hi_max}"
    );
}

/// Assert softmax output is in [0, 1] (within numerical tolerance).
fn assert_softmax_range(bounds: &BoundedTensor, label: &str) {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "{label}: softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "{label}: softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// Kernel builders
// ===========================================================================

/// Build a cell detector sigmoid head: Linear -> sigmoid.
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable, backbone features).
/// Output: `[NUM_CELLS, NUM_CLS]` (cell type probabilities in (0, 1)).
fn build_cell_detector_sigmoid() -> TensorKernelDef {
    let out_shape = [NUM_CELLS, NUM_CLS];
    let mut b = TensorBlockBuilder::new("table_extraction_cell_detector");

    let input = b.add_input("features", &[NUM_CELLS, HIDDEN_DIM]);
    let w = b.add_input("det_weight", &[NUM_CLS, HIDDEN_DIM]);
    let bias = b.add_input("det_bias", &[NUM_CLS]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);

    b.build(out).expect("valid cell detector sigmoid kernel")
}

fn cell_detector_sigmoid_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CLS]), 0.0f32)),
    ]
}

/// Build a cell bounding box regression head: Linear -> sigmoid.
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable).
/// Output: `[NUM_CELLS, 4]` (normalized bbox [x, y, w, h] in (0, 1)).
fn build_cell_bbox_regression() -> TensorKernelDef {
    let out_shape = [NUM_CELLS, 4];
    let mut b = TensorBlockBuilder::new("table_extraction_cell_bbox");

    let input = b.add_input("features", &[NUM_CELLS, HIDDEN_DIM]);
    let w = b.add_input("bbox_weight", &[4, HIDDEN_DIM]);
    let bias = b.add_input("bbox_bias", &[4]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);

    b.build(out).expect("valid cell bbox regression kernel")
}

fn cell_bbox_regression_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4, HIDDEN_DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4]), 0.0f32)),
    ]
}

/// Build a dual-head cell detector: cls sigmoid + bbox sigmoid concatenated.
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable).
/// Output: `[NUM_CELLS, NUM_CLS + 4]` (cls probs + bbox coords).
fn build_cell_detector_dual_head() -> TensorKernelDef {
    let cls_shape = [NUM_CELLS, NUM_CLS];
    let bbox_shape = [NUM_CELLS, 4];
    let concat_shape = [NUM_CELLS, NUM_CLS + 4];
    let mut b = TensorBlockBuilder::new("table_extraction_cell_dual_head");

    let input = b.add_input("features", &[NUM_CELLS, HIDDEN_DIM]);

    // Classification head
    let cls_w = b.add_input("cls_weight", &[NUM_CLS, HIDDEN_DIM]);
    let cls_b = b.add_input("cls_bias", &[NUM_CLS]);
    let cls_logits = b.add_linear(input, cls_w, Some(cls_b), &cls_shape);
    let cls_out = b.add_sigmoid(cls_logits, &cls_shape);

    // Bbox regression head
    let bbox_w = b.add_input("bbox_weight", &[4, HIDDEN_DIM]);
    let bbox_b = b.add_input("bbox_bias", &[4]);
    let bbox_logits = b.add_linear(input, bbox_w, Some(bbox_b), &bbox_shape);
    let bbox_out = b.add_sigmoid(bbox_logits, &bbox_shape);

    let out = b.add_concat(&[cls_out, bbox_out], 1, &concat_shape);

    b.build(out).expect("valid cell dual head kernel")
}

fn cell_detector_dual_head_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        // cls head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CLS]), 0.0f32)),
        // bbox head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4, HIDDEN_DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4]), 0.0f32)),
    ]
}

/// Build a row classifier: Linear -> softmax over row indices.
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable).
/// Output: `[NUM_CELLS, MAX_ROWS]` (row assignment probabilities).
fn build_row_classifier() -> TensorKernelDef {
    let out_shape = [NUM_CELLS, MAX_ROWS];
    let mut b = TensorBlockBuilder::new("table_extraction_row_classifier");

    let input = b.add_input("cell_features", &[NUM_CELLS, HIDDEN_DIM]);
    let w = b.add_input("row_weight", &[MAX_ROWS, HIDDEN_DIM]);
    let bias = b.add_input("row_bias", &[MAX_ROWS]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_softmax(logits, 1, &out_shape);

    b.build(out).expect("valid row classifier kernel")
}

fn row_classifier_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAX_ROWS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAX_ROWS]), 0.0f32)),
    ]
}

/// Build a column classifier: Linear -> softmax over column indices.
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable).
/// Output: `[NUM_CELLS, MAX_COLS]` (column assignment probabilities).
fn build_col_classifier() -> TensorKernelDef {
    let out_shape = [NUM_CELLS, MAX_COLS];
    let mut b = TensorBlockBuilder::new("table_extraction_col_classifier");

    let input = b.add_input("cell_features", &[NUM_CELLS, HIDDEN_DIM]);
    let w = b.add_input("col_weight", &[MAX_COLS, HIDDEN_DIM]);
    let bias = b.add_input("col_bias", &[MAX_COLS]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_softmax(logits, 1, &out_shape);

    b.build(out).expect("valid column classifier kernel")
}

fn col_classifier_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAX_COLS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAX_COLS]), 0.0f32)),
    ]
}

/// Build a row-column joint classification head: row softmax + col softmax concat.
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable).
/// Output: `[NUM_CELLS, MAX_ROWS + MAX_COLS]` (row + col assignment probabilities).
fn build_row_col_joint_head() -> TensorKernelDef {
    let row_shape = [NUM_CELLS, MAX_ROWS];
    let col_shape = [NUM_CELLS, MAX_COLS];
    let concat_shape = [NUM_CELLS, MAX_ROWS + MAX_COLS];
    let mut b = TensorBlockBuilder::new("table_extraction_row_col_joint");

    let input = b.add_input("cell_features", &[NUM_CELLS, HIDDEN_DIM]);

    // Row head
    let row_w = b.add_input("row_weight", &[MAX_ROWS, HIDDEN_DIM]);
    let row_b = b.add_input("row_bias", &[MAX_ROWS]);
    let row_logits = b.add_linear(input, row_w, Some(row_b), &row_shape);
    let row_out = b.add_softmax(row_logits, 1, &row_shape);

    // Column head
    let col_w = b.add_input("col_weight", &[MAX_COLS, HIDDEN_DIM]);
    let col_b = b.add_input("col_bias", &[MAX_COLS]);
    let col_logits = b.add_linear(input, col_w, Some(col_b), &col_shape);
    let col_out = b.add_softmax(col_logits, 1, &col_shape);

    let out = b.add_concat(&[row_out, col_out], 1, &concat_shape);

    b.build(out).expect("valid row-col joint head kernel")
}

fn row_col_joint_head_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        // row head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAX_ROWS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAX_ROWS]), 0.0f32)),
        // col head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAX_COLS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAX_COLS]), 0.0f32)),
    ]
}

/// Build a cell OCR feature extractor: Linear -> ReLU -> Linear.
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable, cell region features).
/// Output: `[NUM_CELLS, OCR_FEAT_DIM]` (refined OCR features).
fn build_ocr_feature_extractor() -> TensorKernelDef {
    let mid_shape = [NUM_CELLS, OCR_FEAT_DIM];
    let out_shape = [NUM_CELLS, OCR_FEAT_DIM];
    let mut b = TensorBlockBuilder::new("table_extraction_ocr_features");

    let input = b.add_input("cell_features", &[NUM_CELLS, HIDDEN_DIM]);

    // First linear
    let w1 = b.add_input("feat_w1", &[OCR_FEAT_DIM, HIDDEN_DIM]);
    let b1 = b.add_input("feat_b1", &[OCR_FEAT_DIM]);
    let h = b.add_linear(input, w1, Some(b1), &mid_shape);
    let h_act = b.add_relu(h, &mid_shape);

    // Second linear
    let w2 = b.add_input("feat_w2", &[OCR_FEAT_DIM, OCR_FEAT_DIM]);
    let b2 = b.add_input("feat_b2", &[OCR_FEAT_DIM]);
    let out = b.add_linear(h_act, w2, Some(b2), &out_shape);

    b.build(out).expect("valid OCR feature extractor kernel")
}

fn ocr_feature_extractor_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[OCR_FEAT_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[OCR_FEAT_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[OCR_FEAT_DIM, OCR_FEAT_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[OCR_FEAT_DIM]), 0.0f32)),
    ]
}

/// Build a cell OCR CTC head: Linear -> softmax (character probabilities).
///
/// Input: `[NUM_CELLS, OCR_FEAT_DIM]` (Variable, OCR features).
/// Output: `[NUM_CELLS, VOCAB_SIZE]` (character distribution in [0, 1]).
fn build_ocr_ctc_head() -> TensorKernelDef {
    let out_shape = [NUM_CELLS, VOCAB_SIZE];
    let mut b = TensorBlockBuilder::new("table_extraction_ocr_ctc");

    let input = b.add_input("ocr_features", &[NUM_CELLS, OCR_FEAT_DIM]);
    let w = b.add_input("ctc_weight", &[VOCAB_SIZE, OCR_FEAT_DIM]);
    let bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_softmax(logits, 1, &out_shape);

    b.build(out).expect("valid OCR CTC head kernel")
}

fn ocr_ctc_head_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, OCR_FEAT_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)),
    ]
}

/// Build OCR pipeline: Linear -> ReLU -> Linear -> Linear -> softmax.
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable).
/// Output: `[NUM_CELLS, VOCAB_SIZE]` (character distribution).
fn build_ocr_pipeline() -> TensorKernelDef {
    let mid_shape = [NUM_CELLS, OCR_FEAT_DIM];
    let out_shape = [NUM_CELLS, VOCAB_SIZE];
    let mut b = TensorBlockBuilder::new("table_extraction_ocr_pipeline");

    let input = b.add_input("cell_features", &[NUM_CELLS, HIDDEN_DIM]);

    // Feature extractor: Linear -> ReLU -> Linear
    let w1 = b.add_input("feat_w1", &[OCR_FEAT_DIM, HIDDEN_DIM]);
    let b1 = b.add_input("feat_b1", &[OCR_FEAT_DIM]);
    let h = b.add_linear(input, w1, Some(b1), &mid_shape);
    let h_act = b.add_relu(h, &mid_shape);

    let w2 = b.add_input("feat_w2", &[OCR_FEAT_DIM, OCR_FEAT_DIM]);
    let b2 = b.add_input("feat_b2", &[OCR_FEAT_DIM]);
    let feats = b.add_linear(h_act, w2, Some(b2), &mid_shape);

    // CTC head: Linear -> softmax
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, OCR_FEAT_DIM]);
    let ctc_b = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(feats, ctc_w, Some(ctc_b), &out_shape);
    let out = b.add_softmax(logits, 1, &out_shape);

    b.build(out).expect("valid OCR pipeline kernel")
}

fn ocr_pipeline_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        // Feature extractor layer 1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[OCR_FEAT_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[OCR_FEAT_DIM]), 0.0f32)),
        // Feature extractor layer 2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[OCR_FEAT_DIM, OCR_FEAT_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[OCR_FEAT_DIM]), 0.0f32)),
        // CTC head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOCAB_SIZE, OCR_FEAT_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32)),
    ]
}

/// Build a form field type classifier: Linear -> softmax over field types.
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable, detected field features).
/// Output: `[NUM_CELLS, NUM_FIELD_TYPES]` (field type probabilities).
fn build_form_field_type_classifier() -> TensorKernelDef {
    let out_shape = [NUM_CELLS, NUM_FIELD_TYPES];
    let mut b = TensorBlockBuilder::new("table_extraction_form_field_type");

    let input = b.add_input("field_features", &[NUM_CELLS, HIDDEN_DIM]);
    let w = b.add_input("type_weight", &[NUM_FIELD_TYPES, HIDDEN_DIM]);
    let bias = b.add_input("type_bias", &[NUM_FIELD_TYPES]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_softmax(logits, 1, &out_shape);

    b.build(out)
        .expect("valid form field type classifier kernel")
}

fn form_field_type_classifier_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_FIELD_TYPES, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_FIELD_TYPES]), 0.0f32)),
    ]
}

/// Build a form field value extractor: Linear -> sigmoid (confidence score).
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable).
/// Output: `[NUM_CELLS, 1]` (value extraction confidence in (0, 1)).
fn build_form_field_value_extractor() -> TensorKernelDef {
    let out_shape = [NUM_CELLS, 1];
    let mut b = TensorBlockBuilder::new("table_extraction_form_field_value");

    let input = b.add_input("field_features", &[NUM_CELLS, HIDDEN_DIM]);
    let w = b.add_input("value_weight", &[1, HIDDEN_DIM]);
    let bias = b.add_input("value_bias", &[1]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);

    b.build(out)
        .expect("valid form field value extractor kernel")
}

fn form_field_value_extractor_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, HIDDEN_DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 0.0f32)),
    ]
}

/// Build a form key-value pair head: type softmax + value sigmoid + confidence sigmoid.
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable).
/// Output: `[NUM_CELLS, NUM_FIELD_TYPES + 1 + 1]` (type probs + value conf + key conf).
fn build_form_kv_pair_head() -> TensorKernelDef {
    let type_shape = [NUM_CELLS, NUM_FIELD_TYPES];
    let val_shape = [NUM_CELLS, 1];
    let key_shape = [NUM_CELLS, 1];
    let concat_shape = [NUM_CELLS, NUM_FIELD_TYPES + 2];
    let mut b = TensorBlockBuilder::new("table_extraction_form_kv_pair");

    let input = b.add_input("field_features", &[NUM_CELLS, HIDDEN_DIM]);

    // Type classification (softmax)
    let type_w = b.add_input("type_weight", &[NUM_FIELD_TYPES, HIDDEN_DIM]);
    let type_b = b.add_input("type_bias", &[NUM_FIELD_TYPES]);
    let type_logits = b.add_linear(input, type_w, Some(type_b), &type_shape);
    let type_out = b.add_softmax(type_logits, 1, &type_shape);

    // Value confidence (sigmoid)
    let val_w = b.add_input("value_weight", &[1, HIDDEN_DIM]);
    let val_b = b.add_input("value_bias", &[1]);
    let val_logits = b.add_linear(input, val_w, Some(val_b), &val_shape);
    let val_out = b.add_sigmoid(val_logits, &val_shape);

    // Key confidence (sigmoid)
    let key_w = b.add_input("key_weight", &[1, HIDDEN_DIM]);
    let key_b = b.add_input("key_bias", &[1]);
    let key_logits = b.add_linear(input, key_w, Some(key_b), &key_shape);
    let key_out = b.add_sigmoid(key_logits, &key_shape);

    let out = b.add_concat(&[type_out, val_out, key_out], 1, &concat_shape);

    b.build(out).expect("valid form key-value pair head kernel")
}

fn form_kv_pair_head_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        // type head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_FIELD_TYPES, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_FIELD_TYPES]), 0.0f32)),
        // value confidence head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, HIDDEN_DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 0.0f32)),
        // key confidence head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, HIDDEN_DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 0.0f32)),
    ]
}

/// Build a column count predictor: Linear -> softmax over column counts.
///
/// Input: `[1, HIDDEN_DIM]` (Variable, pooled document features).
/// Output: `[1, MAX_COLUMNS]` (column count distribution).
fn build_column_count_predictor() -> TensorKernelDef {
    let out_shape = [1, MAX_COLUMNS];
    let mut b = TensorBlockBuilder::new("table_extraction_column_count");

    let input = b.add_input("doc_features", &[1, HIDDEN_DIM]);
    let w = b.add_input("col_count_weight", &[MAX_COLUMNS, HIDDEN_DIM]);
    let bias = b.add_input("col_count_bias", &[MAX_COLUMNS]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_softmax(logits, 1, &out_shape);

    b.build(out).expect("valid column count predictor kernel")
}

fn column_count_predictor_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAX_COLUMNS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAX_COLUMNS]), 0.0f32)),
    ]
}

/// Build a reading order predictor: Linear -> sigmoid (pairwise ordering score).
///
/// For each pair of regions, predicts probability that region i comes before region j.
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable, region features).
/// Output: `[NUM_CELLS, 1]` (pairwise ordering probability in (0, 1)).
fn build_reading_order_predictor() -> TensorKernelDef {
    let out_shape = [NUM_CELLS, 1];
    let mut b = TensorBlockBuilder::new("table_extraction_reading_order");

    let input = b.add_input("region_features", &[NUM_CELLS, HIDDEN_DIM]);
    let w = b.add_input("order_weight", &[1, HIDDEN_DIM]);
    let bias = b.add_input("order_bias", &[1]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);

    b.build(out).expect("valid reading order predictor kernel")
}

fn reading_order_predictor_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, HIDDEN_DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 0.0f32)),
    ]
}

/// Build a layout region classifier: Linear -> softmax over region types.
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable, detected region features).
/// Output: `[NUM_CELLS, NUM_REGIONS]` (region type probabilities).
fn build_layout_region_classifier() -> TensorKernelDef {
    let out_shape = [NUM_CELLS, NUM_REGIONS];
    let mut b = TensorBlockBuilder::new("table_extraction_layout_region");

    let input = b.add_input("region_features", &[NUM_CELLS, HIDDEN_DIM]);
    let w = b.add_input("region_weight", &[NUM_REGIONS, HIDDEN_DIM]);
    let bias = b.add_input("region_bias", &[NUM_REGIONS]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_softmax(logits, 1, &out_shape);

    b.build(out).expect("valid layout region classifier kernel")
}

fn layout_region_classifier_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_REGIONS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_REGIONS]), 0.0f32)),
    ]
}

/// Build coordinate normalization: Linear -> sigmoid mapping to [0, 1].
///
/// Input: `[NUM_CELLS, 4]` (Variable, raw bbox coordinates).
/// Output: `[NUM_CELLS, 4]` (normalized coordinates in (0, 1)).
fn build_coord_normalization() -> TensorKernelDef {
    let out_shape = [NUM_CELLS, 4];
    let mut b = TensorBlockBuilder::new("table_extraction_coord_norm");

    let input = b.add_input("raw_coords", &[NUM_CELLS, 4]);
    let w = b.add_input("norm_weight", &[4, 4]);
    let bias = b.add_input("norm_bias", &[4]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);

    b.build(out).expect("valid coordinate normalization kernel")
}

fn coord_normalization_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4, 4]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4]), 0.0f32)),
    ]
}

/// Build coordinate de-normalization: Linear scaling from [0, 1] to page coords.
///
/// Input: `[NUM_CELLS, 4]` (Variable, normalized coordinates).
/// Output: `[NUM_CELLS, 4]` (scaled page coordinates).
fn build_coord_denormalization() -> TensorKernelDef {
    let out_shape = [NUM_CELLS, 4];
    let mut b = TensorBlockBuilder::new("table_extraction_coord_denorm");

    let input = b.add_input("norm_coords", &[NUM_CELLS, 4]);
    let w = b.add_input("denorm_weight", &[4, 4]);
    let bias = b.add_input("denorm_bias", &[4]);

    let out = b.add_linear(input, w, Some(bias), &out_shape);

    b.build(out)
        .expect("valid coordinate de-normalization kernel")
}

fn coord_denormalization_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        // Diagonal scale matrix: page_w * x, page_h * y, etc.
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4, 4]), 0.01)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4]), 0.0f32)),
    ]
}

/// Build a normalized coordinate pipeline: detect -> normalize -> classify.
///
/// Features -> bbox sigmoid -> cls softmax, with coords in [0, 1].
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable).
/// Output: `[NUM_CELLS, NUM_CLS + 4]` (cls probs + normalized bbox).
fn build_normalized_coord_pipeline() -> TensorKernelDef {
    let cls_shape = [NUM_CELLS, NUM_CLS];
    let bbox_shape = [NUM_CELLS, 4];
    let concat_shape = [NUM_CELLS, NUM_CLS + 4];
    let mut b = TensorBlockBuilder::new("table_extraction_normalized_pipeline");

    let input = b.add_input("features", &[NUM_CELLS, HIDDEN_DIM]);

    // Classification head (softmax for multi-class)
    let cls_w = b.add_input("cls_weight", &[NUM_CLS, HIDDEN_DIM]);
    let cls_b = b.add_input("cls_bias", &[NUM_CLS]);
    let cls_logits = b.add_linear(input, cls_w, Some(cls_b), &cls_shape);
    let cls_out = b.add_softmax(cls_logits, 1, &cls_shape);

    // Bbox regression through coordinate normalization (sigmoid -> [0, 1])
    let bbox_w = b.add_input("bbox_weight", &[4, HIDDEN_DIM]);
    let bbox_b = b.add_input("bbox_bias", &[4]);
    let bbox_logits = b.add_linear(input, bbox_w, Some(bbox_b), &bbox_shape);
    let bbox_out = b.add_sigmoid(bbox_logits, &bbox_shape);

    let out = b.add_concat(&[cls_out, bbox_out], 1, &concat_shape);

    b.build(out)
        .expect("valid normalized coordinate pipeline kernel")
}

fn normalized_coord_pipeline_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        // cls head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_CLS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_CLS]), 0.0f32)),
        // bbox head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4, HIDDEN_DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4]), 0.0f32)),
    ]
}

// ===========================================================================
// Test 1: Cell detector sigmoid (IBP + CROWN)
// ===========================================================================

/// Cell detector sigmoid: IBP bounds must be in (0, 1). CROWN should tighten.
#[test]
fn test_table_cell_detector_sigmoid_ibp_crown() {
    let def = build_cell_detector_sigmoid();
    let bindings = cell_detector_sigmoid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    // IBP
    let ibp_out = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(ibp_out.lower_upper().0.shape(), &[NUM_CELLS, NUM_CLS]);
    assert_bounds_valid(&ibp_out);
    assert_sigmoid_range(&ibp_out, "cell detector IBP");

    let (lo, hi) = bounds_min_max(&ibp_out);
    eprintln!("Cell detector sigmoid IBP: bounds=[{lo}, {hi}]");

    // CROWN
    let crown_input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 1.0);
    let (_method, crown_out, _fb) = assert_crown_tighter_when_not_fallback(&graph, &crown_input);
    assert_sigmoid_range(&crown_out, "cell detector CROWN");

    let ibp_w = bound_width(&ibp_out);
    let crown_w = bound_width(&crown_out);
    eprintln!("Cell detector: IBP width={ibp_w:.6}, CROWN width={crown_w:.6}");
}

// ===========================================================================
// Test 2: Cell bounding box regression (IBP)
// ===========================================================================

/// Cell bbox regression: sigmoid produces normalized coords in [0, 1].
#[test]
fn test_table_cell_bbox_regression_ibp() {
    let def = build_cell_bbox_regression();
    let bindings = cell_bbox_regression_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[NUM_CELLS, 4]);
    assert_bounds_valid(&output);
    assert_sigmoid_range(&output, "cell bbox regression IBP");

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Cell bbox regression IBP: bounds=[{lo}, {hi}]");
}

// ===========================================================================
// Test 3: Cell detector dual-head (IBP + CROWN)
// ===========================================================================

/// Dual-head detector: cls + bbox both bounded by their respective activations.
#[test]
fn test_table_cell_detector_dual_head_ibp_crown() {
    let def = build_cell_detector_dual_head();
    let bindings = cell_detector_dual_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    // IBP
    let ibp_out = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(ibp_out.lower_upper().0.shape(), &[NUM_CELLS, NUM_CLS + 4]);
    assert_bounds_valid(&ibp_out);

    // All outputs are sigmoid -> should be in [0, 1]
    assert_sigmoid_range(&ibp_out, "dual head IBP");

    let (lo, hi) = bounds_min_max(&ibp_out);
    eprintln!("Dual head detector IBP: bounds=[{lo}, {hi}]");

    // CROWN
    let crown_input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 1.0);
    let (_method, crown_out, _fb) = assert_crown_tighter_when_not_fallback(&graph, &crown_input);
    assert_sigmoid_range(&crown_out, "dual head CROWN");

    let ibp_w = bound_width(&ibp_out);
    let crown_w = bound_width(&crown_out);
    eprintln!("Dual head: IBP width={ibp_w:.6}, CROWN width={crown_w:.6}");
}

// ===========================================================================
// Test 4: Row classifier softmax (IBP)
// ===========================================================================

/// Row classifier: softmax produces valid probability distribution in [0, 1].
#[test]
fn test_table_row_classifier_ibp() {
    let def = build_row_classifier();
    let bindings = row_classifier_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[NUM_CELLS, MAX_ROWS]);
    assert_bounds_valid(&output);
    assert_softmax_range(&output, "row classifier IBP");

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Row classifier IBP: bounds=[{lo}, {hi}]");
}

// ===========================================================================
// Test 5: Column classifier softmax (IBP)
// ===========================================================================

/// Column classifier: softmax produces valid probability distribution in [0, 1].
#[test]
fn test_table_col_classifier_ibp() {
    let def = build_col_classifier();
    let bindings = col_classifier_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[NUM_CELLS, MAX_COLS]);
    assert_bounds_valid(&output);
    assert_softmax_range(&output, "column classifier IBP");

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Column classifier IBP: bounds=[{lo}, {hi}]");
}

// ===========================================================================
// Test 6: Row-column joint head (IBP + CROWN)
// ===========================================================================

/// Row-column joint head: both softmax heads produce [0, 1] distributions.
#[test]
fn test_table_row_col_joint_head_ibp_crown() {
    let def = build_row_col_joint_head();
    let bindings = row_col_joint_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    // IBP
    let ibp_out = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(
        ibp_out.lower_upper().0.shape(),
        &[NUM_CELLS, MAX_ROWS + MAX_COLS]
    );
    assert_bounds_valid(&ibp_out);
    assert_softmax_range(&ibp_out, "row-col joint IBP");

    let (lo, hi) = bounds_min_max(&ibp_out);
    eprintln!("Row-col joint head IBP: bounds=[{lo}, {hi}]");

    // CROWN
    let crown_input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 1.0);
    let (_method, crown_out, _fb) = assert_crown_tighter_when_not_fallback(&graph, &crown_input);

    let ibp_w = bound_width(&ibp_out);
    let crown_w = bound_width(&crown_out);
    eprintln!("Row-col joint: IBP width={ibp_w:.6}, CROWN width={crown_w:.6}");
}

// ===========================================================================
// Test 7: OCR feature extractor (IBP)
// ===========================================================================

/// OCR feature extractor: Linear -> ReLU -> Linear produces finite bounds.
#[test]
fn test_table_ocr_feature_extractor_ibp() {
    let def = build_ocr_feature_extractor();
    let bindings = ocr_feature_extractor_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[NUM_CELLS, OCR_FEAT_DIM]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("OCR feature extractor IBP: bounds=[{lo}, {hi}]");
    // ReLU ensures lower >= 0 on first layer output, but second linear can go negative
    assert!(
        lo.is_finite() && hi.is_finite(),
        "OCR features must be finite"
    );
}

// ===========================================================================
// Test 8: OCR CTC head (IBP + CROWN)
// ===========================================================================

/// OCR CTC head: softmax produces character probabilities in [0, 1].
#[test]
fn test_table_ocr_ctc_head_ibp_crown() {
    let def = build_ocr_ctc_head();
    let bindings = ocr_ctc_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, OCR_FEAT_DIM], 1.0);

    // IBP
    let ibp_out = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(ibp_out.lower_upper().0.shape(), &[NUM_CELLS, VOCAB_SIZE]);
    assert_bounds_valid(&ibp_out);
    assert_softmax_range(&ibp_out, "OCR CTC IBP");

    let (lo, hi) = bounds_min_max(&ibp_out);
    eprintln!("OCR CTC head IBP: bounds=[{lo}, {hi}]");

    // CROWN
    let crown_input = uniform_bounds(&[NUM_CELLS, OCR_FEAT_DIM], 0.5);
    let (_method, crown_out, _fb) = assert_crown_tighter_when_not_fallback(&graph, &crown_input);
    assert_softmax_range(&crown_out, "OCR CTC CROWN");

    let ibp_w = bound_width(&ibp_out);
    let crown_w = bound_width(&crown_out);
    eprintln!("OCR CTC: IBP width={ibp_w:.6}, CROWN width={crown_w:.6}");
}

// ===========================================================================
// Test 9: OCR pipeline end-to-end (IBP)
// ===========================================================================

/// OCR pipeline: features -> Linear -> ReLU -> Linear -> Linear -> softmax.
#[test]
fn test_table_ocr_pipeline_ibp() {
    let def = build_ocr_pipeline();
    let bindings = ocr_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[NUM_CELLS, VOCAB_SIZE]);
    assert_bounds_valid(&output);
    assert_softmax_range(&output, "OCR pipeline IBP");

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("OCR pipeline end-to-end IBP: bounds=[{lo}, {hi}]");
}

// ===========================================================================
// Test 10: Form field type classifier (IBP + CROWN)
// ===========================================================================

/// Form field type classifier: softmax produces [0, 1] distribution.
#[test]
fn test_table_form_field_type_ibp_crown() {
    let def = build_form_field_type_classifier();
    let bindings = form_field_type_classifier_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    // IBP
    let ibp_out = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(
        ibp_out.lower_upper().0.shape(),
        &[NUM_CELLS, NUM_FIELD_TYPES]
    );
    assert_bounds_valid(&ibp_out);
    assert_softmax_range(&ibp_out, "form field type IBP");

    let (lo, hi) = bounds_min_max(&ibp_out);
    eprintln!("Form field type classifier IBP: bounds=[{lo}, {hi}]");

    // CROWN
    let crown_input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 1.0);
    let (_method, crown_out, _fb) = assert_crown_tighter_when_not_fallback(&graph, &crown_input);
    assert_softmax_range(&crown_out, "form field type CROWN");

    let ibp_w = bound_width(&ibp_out);
    let crown_w = bound_width(&crown_out);
    eprintln!("Form field type: IBP width={ibp_w:.6}, CROWN width={crown_w:.6}");
}

// ===========================================================================
// Test 11: Form field value extractor (IBP)
// ===========================================================================

/// Form field value extractor: sigmoid confidence in (0, 1).
#[test]
fn test_table_form_field_value_ibp() {
    let def = build_form_field_value_extractor();
    let bindings = form_field_value_extractor_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[NUM_CELLS, 1]);
    assert_bounds_valid(&output);
    assert_sigmoid_range(&output, "form field value IBP");

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Form field value extractor IBP: bounds=[{lo}, {hi}]");
}

// ===========================================================================
// Test 12: Form key-value pair head (IBP + CROWN)
// ===========================================================================

/// Form KV pair head: type softmax + value sigmoid + key sigmoid combined.
#[test]
fn test_table_form_kv_pair_head_ibp_crown() {
    let def = build_form_kv_pair_head();
    let bindings = form_kv_pair_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    // IBP
    let ibp_out = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(
        ibp_out.lower_upper().0.shape(),
        &[NUM_CELLS, NUM_FIELD_TYPES + 2]
    );
    assert_bounds_valid(&ibp_out);

    // All sub-heads are softmax or sigmoid -> all outputs in [0, 1]
    let (lo, hi) = bounds_min_max(&ibp_out);
    let eps = 1e-6;
    assert!(lo >= 0.0 - eps, "form KV pair lower must be >= 0, got {lo}");
    assert!(hi <= 1.0 + eps, "form KV pair upper must be <= 1, got {hi}");
    eprintln!("Form KV pair head IBP: bounds=[{lo}, {hi}]");

    // CROWN
    let crown_input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 1.0);
    let (_method, crown_out, _fb) = assert_crown_tighter_when_not_fallback(&graph, &crown_input);

    let ibp_w = bound_width(&ibp_out);
    let crown_w = bound_width(&crown_out);
    eprintln!("Form KV pair: IBP width={ibp_w:.6}, CROWN width={crown_w:.6}");
}

// ===========================================================================
// Test 13: Column count predictor (IBP)
// ===========================================================================

/// Column count predictor: softmax distribution over column counts.
#[test]
fn test_table_column_count_predictor_ibp() {
    let def = build_column_count_predictor();
    let bindings = column_count_predictor_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[1, HIDDEN_DIM], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[1, MAX_COLUMNS]);
    assert_bounds_valid(&output);
    assert_softmax_range(&output, "column count IBP");

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Column count predictor IBP: bounds=[{lo}, {hi}]");
}

// ===========================================================================
// Test 14: Reading order predictor (IBP + CROWN)
// ===========================================================================

/// Reading order predictor: sigmoid pairwise ordering in (0, 1).
#[test]
fn test_table_reading_order_predictor_ibp_crown() {
    let def = build_reading_order_predictor();
    let bindings = reading_order_predictor_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    // IBP
    let ibp_out = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(ibp_out.lower_upper().0.shape(), &[NUM_CELLS, 1]);
    assert_bounds_valid(&ibp_out);
    assert_sigmoid_range(&ibp_out, "reading order IBP");

    let (lo, hi) = bounds_min_max(&ibp_out);
    eprintln!("Reading order predictor IBP: bounds=[{lo}, {hi}]");

    // CROWN
    let crown_input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 1.0);
    let (_method, crown_out, _fb) = assert_crown_tighter_when_not_fallback(&graph, &crown_input);
    assert_sigmoid_range(&crown_out, "reading order CROWN");

    let ibp_w = bound_width(&ibp_out);
    let crown_w = bound_width(&crown_out);
    eprintln!("Reading order: IBP width={ibp_w:.6}, CROWN width={crown_w:.6}");
}

// ===========================================================================
// Test 15: Layout region classifier (IBP + CROWN)
// ===========================================================================

/// Layout region classifier: softmax over region types in [0, 1].
#[test]
fn test_table_layout_region_classifier_ibp_crown() {
    let def = build_layout_region_classifier();
    let bindings = layout_region_classifier_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    // IBP
    let ibp_out = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(ibp_out.lower_upper().0.shape(), &[NUM_CELLS, NUM_REGIONS]);
    assert_bounds_valid(&ibp_out);
    assert_softmax_range(&ibp_out, "layout region IBP");

    let (lo, hi) = bounds_min_max(&ibp_out);
    eprintln!("Layout region classifier IBP: bounds=[{lo}, {hi}]");

    // CROWN
    let crown_input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 1.0);
    let (_method, crown_out, _fb) = assert_crown_tighter_when_not_fallback(&graph, &crown_input);
    assert_softmax_range(&crown_out, "layout region CROWN");

    let ibp_w = bound_width(&ibp_out);
    let crown_w = bound_width(&crown_out);
    eprintln!("Layout region: IBP width={ibp_w:.6}, CROWN width={crown_w:.6}");
}

// ===========================================================================
// Test 16: Coordinate normalization (IBP)
// ===========================================================================

/// Coordinate normalization: sigmoid maps raw coords to [0, 1].
#[test]
fn test_table_coord_normalization_ibp() {
    let def = build_coord_normalization();
    let bindings = coord_normalization_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Raw coordinates: arbitrary range
    let input = uniform_bounds(&[NUM_CELLS, 4], 100.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[NUM_CELLS, 4]);
    assert_bounds_valid(&output);
    assert_sigmoid_range(&output, "coord normalization IBP");

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Coordinate normalization IBP: bounds=[{lo}, {hi}]");
}

// ===========================================================================
// Test 17: Coordinate de-normalization (IBP)
// ===========================================================================

/// Coordinate de-normalization: linear scaling preserves finite bounds.
#[test]
fn test_table_coord_denormalization_ibp() {
    let def = build_coord_denormalization();
    let bindings = coord_denormalization_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Normalized input: [0, 1] range (simulate sigmoid output)
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[NUM_CELLS, 4]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[NUM_CELLS, 4]), 1.0f32),
    )
    .expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(output.lower_upper().0.shape(), &[NUM_CELLS, 4]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Coordinate de-normalization IBP: bounds=[{lo}, {hi}]");
    // Linear scaling of [0, 1] input: output should be bounded
    assert!(lo.is_finite(), "de-normalized lower must be finite");
    assert!(hi.is_finite(), "de-normalized upper must be finite");
}

// ===========================================================================
// Test 18: Normalized coordinate pipeline (IBP + CROWN)
// ===========================================================================

/// End-to-end: features -> cls softmax + bbox sigmoid, all in [0, 1].
#[test]
fn test_table_normalized_coord_pipeline_ibp_crown() {
    let def = build_normalized_coord_pipeline();
    let bindings = normalized_coord_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    // IBP
    let ibp_out = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(ibp_out.lower_upper().0.shape(), &[NUM_CELLS, NUM_CLS + 4]);
    assert_bounds_valid(&ibp_out);

    // Both softmax and sigmoid produce [0, 1] outputs
    let (lo, hi) = bounds_min_max(&ibp_out);
    let eps = 1e-6;
    assert!(lo >= 0.0 - eps, "pipeline lower must be >= 0, got {lo}");
    assert!(hi <= 1.0 + eps, "pipeline upper must be <= 1, got {hi}");
    eprintln!("Normalized coord pipeline IBP: bounds=[{lo}, {hi}]");

    // CROWN
    let crown_input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 1.0);
    let (_method, crown_out, _fb) = assert_crown_tighter_when_not_fallback(&graph, &crown_input);

    let ibp_w = bound_width(&ibp_out);
    let crown_w = bound_width(&crown_out);
    eprintln!("Normalized coord pipeline: IBP width={ibp_w:.6}, CROWN width={crown_w:.6}");
}
