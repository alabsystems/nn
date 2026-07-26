// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for table structure recognition pipeline.
//!
//! Verifies IBP and CROWN bound propagation through the table structure
//! recognition sub-blocks used in dpdf document understanding: cell detection,
//! row/column separator parsing, spanning cell prediction, and composed
//! detection-to-structure pipelines.
//!
//! ## Cell Detection Heads (tests 1-4)
//!
//! 1. Cell classification sigmoid: output in (0, 1) (IBP + CROWN)
//! 2. Cell bbox regression sigmoid: normalized coordinates (IBP)
//! 3. Row separator detection: binary classification head (IBP)
//! 4. Column separator detection: binary classification head (IBP)
//!
//! ## Structure Parsing (tests 5-8)
//!
//! 5. Row count prediction: integer output bounded (IBP)
//! 6. Column count prediction: integer output bounded (IBP)
//! 7. Cell-to-row assignment: softmax probability (IBP)
//! 8. Cell-to-column assignment: softmax probability (IBP)
//!
//! ## Spanning Cells (tests 9-11)
//!
//! 9. Rowspan prediction: [1, max_rows] bounded (IBP)
//! 10. Colspan prediction: [1, max_cols] bounded (IBP)
//! 11. Span confidence: sigmoid gating for span detection (IBP + CROWN)
//!
//! ## Composition (tests 12-15)
//!
//! 12. Detection -> structure pipeline: end-to-end (IBP)
//! 13. Structure monotone tightening: smaller eps -> tighter cell bounds (IBP)
//! 14. Multi-head table: detection + structure + span combined (IBP + CROWN)
//! 15. Table -> HTML: confidence-weighted structure assembly (IBP)
//!
//! Architecture references:
//! - Table Transformer (Smock et al. 2022): DETR-based table structure recognition
//! - DETR (Carion et al. 2020): DEtection TRansformer, end-to-end object detection
//!
//! Dimensions (small for fast verification, structurally representative):
//! - NUM_CELLS=6, HIDDEN_DIM=32, MAX_ROWS=8, MAX_COLS=8, NUM_QUERIES=6
//!
//! Part of #3996: Compose tests for table structure recognition.

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

/// Number of detected cell candidates (DETR object queries).
const NUM_CELLS: usize = 6;
/// Hidden dimension from decoder output.
const HIDDEN_DIM: usize = 32;
/// Maximum number of rows in a table.
const MAX_ROWS: usize = 8;
/// Maximum number of columns in a table.
const MAX_COLS: usize = 8;
/// Number of structure classes (row, column, header, spanning, etc.).
const NUM_STRUCT_CLASSES: usize = 4;
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

// ===========================================================================
// Kernel builders
// ===========================================================================

/// Build a cell classification sigmoid head: Linear -> sigmoid.
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable, decoder output).
/// Output: `[NUM_CELLS, NUM_STRUCT_CLASSES]` (cell class probabilities in (0, 1)).
fn build_cell_cls_sigmoid_kernel() -> TensorKernelDef {
    let out_shape = [NUM_CELLS, NUM_STRUCT_CLASSES];
    let mut b = TensorBlockBuilder::new("table_structure_cell_cls_sigmoid");

    let input = b.add_input("decoder_output", &[NUM_CELLS, HIDDEN_DIM]);
    let w = b.add_input("cls_weight", &[NUM_STRUCT_CLASSES, HIDDEN_DIM]);
    let bias = b.add_input("cls_bias", &[NUM_STRUCT_CLASSES]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);

    b.build(out)
        .expect("valid cell classification sigmoid kernel")
}

/// Bindings for cell classification sigmoid head.
fn cell_cls_sigmoid_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // decoder_output
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_STRUCT_CLASSES, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // cls_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_STRUCT_CLASSES]), 0.0f32)), // cls_bias
    ]
}

/// Build a cell bbox regression sigmoid head: Linear -> sigmoid.
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable, decoder output).
/// Output: `[NUM_CELLS, 4]` (normalized cell bbox coordinates in (0, 1)).
fn build_cell_bbox_sigmoid_kernel() -> TensorKernelDef {
    let out_shape = [NUM_CELLS, 4];
    let mut b = TensorBlockBuilder::new("table_structure_cell_bbox_sigmoid");

    let input = b.add_input("decoder_output", &[NUM_CELLS, HIDDEN_DIM]);
    let w = b.add_input("bbox_weight", &[4, HIDDEN_DIM]);
    let bias = b.add_input("bbox_bias", &[4]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);

    b.build(out)
        .expect("valid cell bbox regression sigmoid kernel")
}

/// Bindings for cell bbox regression sigmoid head.
fn cell_bbox_sigmoid_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // decoder_output
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4, HIDDEN_DIM]), WEIGHT_MAG)), // bbox_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4]), 0.0f32)), // bbox_bias
    ]
}

/// Build a row separator detection head: Linear -> sigmoid (binary classification).
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable, decoder features).
/// Output: `[NUM_CELLS, 1]` (row separator probability in (0, 1)).
fn build_row_separator_kernel() -> TensorKernelDef {
    let out_shape = [NUM_CELLS, 1];
    let mut b = TensorBlockBuilder::new("table_structure_row_separator");

    let input = b.add_input("features", &[NUM_CELLS, HIDDEN_DIM]);
    let w = b.add_input("row_sep_weight", &[1, HIDDEN_DIM]);
    let bias = b.add_input("row_sep_bias", &[1]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);

    b.build(out).expect("valid row separator detection kernel")
}

/// Bindings for row separator detection.
fn row_separator_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, HIDDEN_DIM]), WEIGHT_MAG)), // row_sep_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 0.0f32)), // row_sep_bias
    ]
}

/// Build a column separator detection head: Linear -> sigmoid (binary classification).
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable, decoder features).
/// Output: `[NUM_CELLS, 1]` (column separator probability in (0, 1)).
fn build_col_separator_kernel() -> TensorKernelDef {
    let out_shape = [NUM_CELLS, 1];
    let mut b = TensorBlockBuilder::new("table_structure_col_separator");

    let input = b.add_input("features", &[NUM_CELLS, HIDDEN_DIM]);
    let w = b.add_input("col_sep_weight", &[1, HIDDEN_DIM]);
    let bias = b.add_input("col_sep_bias", &[1]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);

    b.build(out)
        .expect("valid column separator detection kernel")
}

/// Bindings for column separator detection.
fn col_separator_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, HIDDEN_DIM]), WEIGHT_MAG)), // col_sep_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 0.0f32)), // col_sep_bias
    ]
}

/// Build a row count prediction head: Linear -> sigmoid (bounded integer proxy).
///
/// Predicts row count as sigmoid(logit) * MAX_ROWS, bounding output in [0, MAX_ROWS].
/// We verify the sigmoid stage; the scaling is a post-processing multiply.
///
/// Input: `[1, HIDDEN_DIM]` (Variable, pooled table features).
/// Output: `[1, MAX_ROWS]` (softmax distribution over row count bins).
fn build_row_count_kernel() -> TensorKernelDef {
    let out_shape = [1, MAX_ROWS];
    let mut b = TensorBlockBuilder::new("table_structure_row_count");

    let input = b.add_input("table_features", &[1, HIDDEN_DIM]);
    let w = b.add_input("row_count_weight", &[MAX_ROWS, HIDDEN_DIM]);
    let bias = b.add_input("row_count_bias", &[MAX_ROWS]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_softmax(logits, 1, &out_shape);

    b.build(out).expect("valid row count prediction kernel")
}

/// Bindings for row count prediction.
fn row_count_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // table_features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAX_ROWS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // row_count_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAX_ROWS]), 0.0f32)), // row_count_bias
    ]
}

/// Build a column count prediction head: Linear -> softmax over column count bins.
///
/// Input: `[1, HIDDEN_DIM]` (Variable, pooled table features).
/// Output: `[1, MAX_COLS]` (softmax distribution over column count bins).
fn build_col_count_kernel() -> TensorKernelDef {
    let out_shape = [1, MAX_COLS];
    let mut b = TensorBlockBuilder::new("table_structure_col_count");

    let input = b.add_input("table_features", &[1, HIDDEN_DIM]);
    let w = b.add_input("col_count_weight", &[MAX_COLS, HIDDEN_DIM]);
    let bias = b.add_input("col_count_bias", &[MAX_COLS]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_softmax(logits, 1, &out_shape);

    b.build(out).expect("valid column count prediction kernel")
}

/// Bindings for column count prediction.
fn col_count_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // table_features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAX_COLS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // col_count_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAX_COLS]), 0.0f32)), // col_count_bias
    ]
}

/// Build a cell-to-row assignment head: Linear -> softmax (assignment probability).
///
/// Each cell is assigned to one of MAX_ROWS rows via softmax probability.
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable, decoder output per cell).
/// Output: `[NUM_CELLS, MAX_ROWS]` (row assignment probabilities in [0, 1]).
fn build_cell_to_row_kernel() -> TensorKernelDef {
    let out_shape = [NUM_CELLS, MAX_ROWS];
    let mut b = TensorBlockBuilder::new("table_structure_cell_to_row");

    let input = b.add_input("cell_features", &[NUM_CELLS, HIDDEN_DIM]);
    let w = b.add_input("row_assign_weight", &[MAX_ROWS, HIDDEN_DIM]);
    let bias = b.add_input("row_assign_bias", &[MAX_ROWS]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_softmax(logits, 1, &out_shape);

    b.build(out).expect("valid cell-to-row assignment kernel")
}

/// Bindings for cell-to-row assignment.
fn cell_to_row_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // cell_features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAX_ROWS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // row_assign_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAX_ROWS]), 0.0f32)), // row_assign_bias
    ]
}

/// Build a cell-to-column assignment head: Linear -> softmax.
///
/// Each cell is assigned to one of MAX_COLS columns via softmax probability.
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable, decoder output per cell).
/// Output: `[NUM_CELLS, MAX_COLS]` (column assignment probabilities in [0, 1]).
fn build_cell_to_col_kernel() -> TensorKernelDef {
    let out_shape = [NUM_CELLS, MAX_COLS];
    let mut b = TensorBlockBuilder::new("table_structure_cell_to_col");

    let input = b.add_input("cell_features", &[NUM_CELLS, HIDDEN_DIM]);
    let w = b.add_input("col_assign_weight", &[MAX_COLS, HIDDEN_DIM]);
    let bias = b.add_input("col_assign_bias", &[MAX_COLS]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_softmax(logits, 1, &out_shape);

    b.build(out)
        .expect("valid cell-to-column assignment kernel")
}

/// Bindings for cell-to-column assignment.
fn cell_to_col_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // cell_features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAX_COLS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // col_assign_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAX_COLS]), 0.0f32)), // col_assign_bias
    ]
}

/// Build a rowspan prediction head: Linear -> sigmoid (bounded in [0, 1]).
///
/// Sigmoid output is scaled by MAX_ROWS in post-processing to get span count.
/// We verify the sigmoid stage produces output in (0, 1).
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable, cell features).
/// Output: `[NUM_CELLS, 1]` (rowspan fraction in (0, 1)).
fn build_rowspan_kernel() -> TensorKernelDef {
    let out_shape = [NUM_CELLS, 1];
    let mut b = TensorBlockBuilder::new("table_structure_rowspan");

    let input = b.add_input("cell_features", &[NUM_CELLS, HIDDEN_DIM]);
    let w = b.add_input("rowspan_weight", &[1, HIDDEN_DIM]);
    let bias = b.add_input("rowspan_bias", &[1]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);

    b.build(out).expect("valid rowspan prediction kernel")
}

/// Bindings for rowspan prediction.
fn rowspan_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // cell_features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, HIDDEN_DIM]), WEIGHT_MAG)), // rowspan_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 0.0f32)), // rowspan_bias
    ]
}

/// Build a colspan prediction head: Linear -> sigmoid (bounded in [0, 1]).
///
/// Sigmoid output is scaled by MAX_COLS in post-processing to get span count.
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable, cell features).
/// Output: `[NUM_CELLS, 1]` (colspan fraction in (0, 1)).
fn build_colspan_kernel() -> TensorKernelDef {
    let out_shape = [NUM_CELLS, 1];
    let mut b = TensorBlockBuilder::new("table_structure_colspan");

    let input = b.add_input("cell_features", &[NUM_CELLS, HIDDEN_DIM]);
    let w = b.add_input("colspan_weight", &[1, HIDDEN_DIM]);
    let bias = b.add_input("colspan_bias", &[1]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);

    b.build(out).expect("valid colspan prediction kernel")
}

/// Bindings for colspan prediction.
fn colspan_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // cell_features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, HIDDEN_DIM]), WEIGHT_MAG)), // colspan_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 0.0f32)), // colspan_bias
    ]
}

/// Build a span confidence head: Linear -> sigmoid (gating for span detection).
///
/// This predicts whether a cell has any spanning at all (rowspan > 1 or colspan > 1).
/// Used as a gating signal before applying span predictions.
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable, cell features).
/// Output: `[NUM_CELLS, 1]` (span confidence in (0, 1)).
fn build_span_confidence_kernel() -> TensorKernelDef {
    let out_shape = [NUM_CELLS, 1];
    let mut b = TensorBlockBuilder::new("table_structure_span_confidence");

    let input = b.add_input("cell_features", &[NUM_CELLS, HIDDEN_DIM]);
    let w = b.add_input("span_conf_weight", &[1, HIDDEN_DIM]);
    let bias = b.add_input("span_conf_bias", &[1]);

    let logits = b.add_linear(input, w, Some(bias), &out_shape);
    let out = b.add_sigmoid(logits, &out_shape);

    b.build(out).expect("valid span confidence kernel")
}

/// Bindings for span confidence.
fn span_confidence_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // cell_features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, HIDDEN_DIM]), WEIGHT_MAG)), // span_conf_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 0.0f32)), // span_conf_bias
    ]
}

/// Build a detection -> structure pipeline kernel.
///
/// Cell detection (cls sigmoid) feeds into row/column assignment (softmax).
/// Shared decoder features -> cls head + row assignment + col assignment.
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable, decoder output).
/// Output: concatenated `[NUM_CELLS, NUM_STRUCT_CLASSES + MAX_ROWS + MAX_COLS]`.
fn build_detection_structure_pipeline_kernel() -> TensorKernelDef {
    let cls_shape = [NUM_CELLS, NUM_STRUCT_CLASSES];
    let row_shape = [NUM_CELLS, MAX_ROWS];
    let col_shape = [NUM_CELLS, MAX_COLS];
    let concat_shape = [NUM_CELLS, NUM_STRUCT_CLASSES + MAX_ROWS + MAX_COLS];
    let mut b = TensorBlockBuilder::new("table_structure_detection_pipeline");

    let input = b.add_input("decoder_output", &[NUM_CELLS, HIDDEN_DIM]);

    // Cell classification head (sigmoid)
    let cls_w = b.add_input("cls_weight", &[NUM_STRUCT_CLASSES, HIDDEN_DIM]);
    let cls_b = b.add_input("cls_bias", &[NUM_STRUCT_CLASSES]);
    let cls_logits = b.add_linear(input, cls_w, Some(cls_b), &cls_shape);
    let cls_out = b.add_sigmoid(cls_logits, &cls_shape);

    // Row assignment head (softmax)
    let row_w = b.add_input("row_assign_weight", &[MAX_ROWS, HIDDEN_DIM]);
    let row_b = b.add_input("row_assign_bias", &[MAX_ROWS]);
    let row_logits = b.add_linear(input, row_w, Some(row_b), &row_shape);
    let row_out = b.add_softmax(row_logits, 1, &row_shape);

    // Column assignment head (softmax)
    let col_w = b.add_input("col_assign_weight", &[MAX_COLS, HIDDEN_DIM]);
    let col_b = b.add_input("col_assign_bias", &[MAX_COLS]);
    let col_logits = b.add_linear(input, col_w, Some(col_b), &col_shape);
    let col_out = b.add_softmax(col_logits, 1, &col_shape);

    // Concatenate cls + row + col
    let out = b.add_concat(&[cls_out, row_out, col_out], 1, &concat_shape);

    b.build(out)
        .expect("valid detection-structure pipeline kernel")
}

/// Bindings for detection -> structure pipeline.
fn detection_structure_pipeline_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // decoder_output
        // cls head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_STRUCT_CLASSES, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_STRUCT_CLASSES]), 0.0f32)),
        // row assignment head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAX_ROWS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAX_ROWS]), 0.0f32)),
        // col assignment head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAX_COLS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAX_COLS]), 0.0f32)),
    ]
}

/// Build a multi-head table detection + structure + span kernel.
///
/// Five parallel heads from shared decoder features:
/// - Cell classification (sigmoid) -> [NUM_CELLS, NUM_STRUCT_CLASSES]
/// - Box regression (sigmoid) -> [NUM_CELLS, 4]
/// - Row assignment (softmax) -> [NUM_CELLS, MAX_ROWS]
/// - Column assignment (softmax) -> [NUM_CELLS, MAX_COLS]
/// - Span confidence (sigmoid) -> [NUM_CELLS, 1]
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable).
/// Output: concatenated `[NUM_CELLS, NUM_STRUCT_CLASSES + 4 + MAX_ROWS + MAX_COLS + 1]`.
fn build_multi_head_table_kernel() -> TensorKernelDef {
    let cls_shape = [NUM_CELLS, NUM_STRUCT_CLASSES];
    let box_shape = [NUM_CELLS, 4];
    let row_shape = [NUM_CELLS, MAX_ROWS];
    let col_shape = [NUM_CELLS, MAX_COLS];
    let span_shape = [NUM_CELLS, 1];
    let total_dim = NUM_STRUCT_CLASSES + 4 + MAX_ROWS + MAX_COLS + 1;
    let concat_shape = [NUM_CELLS, total_dim];
    let mut b = TensorBlockBuilder::new("table_structure_multi_head");

    let input = b.add_input("decoder_output", &[NUM_CELLS, HIDDEN_DIM]);

    // 1. Cell classification (sigmoid)
    let cls_w = b.add_input("cls_weight", &[NUM_STRUCT_CLASSES, HIDDEN_DIM]);
    let cls_b = b.add_input("cls_bias", &[NUM_STRUCT_CLASSES]);
    let cls_logits = b.add_linear(input, cls_w, Some(cls_b), &cls_shape);
    let cls_out = b.add_sigmoid(cls_logits, &cls_shape);

    // 2. Box regression (sigmoid)
    let box_w = b.add_input("box_weight", &[4, HIDDEN_DIM]);
    let box_b = b.add_input("box_bias", &[4]);
    let box_logits = b.add_linear(input, box_w, Some(box_b), &box_shape);
    let box_out = b.add_sigmoid(box_logits, &box_shape);

    // 3. Row assignment (softmax)
    let row_w = b.add_input("row_weight", &[MAX_ROWS, HIDDEN_DIM]);
    let row_b = b.add_input("row_bias", &[MAX_ROWS]);
    let row_logits = b.add_linear(input, row_w, Some(row_b), &row_shape);
    let row_out = b.add_softmax(row_logits, 1, &row_shape);

    // 4. Column assignment (softmax)
    let col_w = b.add_input("col_weight", &[MAX_COLS, HIDDEN_DIM]);
    let col_b = b.add_input("col_bias", &[MAX_COLS]);
    let col_logits = b.add_linear(input, col_w, Some(col_b), &col_shape);
    let col_out = b.add_softmax(col_logits, 1, &col_shape);

    // 5. Span confidence (sigmoid)
    let span_w = b.add_input("span_weight", &[1, HIDDEN_DIM]);
    let span_b = b.add_input("span_bias", &[1]);
    let span_logits = b.add_linear(input, span_w, Some(span_b), &span_shape);
    let span_out = b.add_sigmoid(span_logits, &span_shape);

    // Concatenate all heads
    let out = b.add_concat(
        &[cls_out, box_out, row_out, col_out, span_out],
        1,
        &concat_shape,
    );

    b.build(out).expect("valid multi-head table kernel")
}

/// Bindings for multi-head table detection + structure + span.
fn multi_head_table_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // decoder_output
        // cls head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_STRUCT_CLASSES, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[NUM_STRUCT_CLASSES]), 0.0f32)),
        // box head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4, HIDDEN_DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[4]), 0.0f32)),
        // row assignment head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAX_ROWS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAX_ROWS]), 0.0f32)),
        // col assignment head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAX_COLS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAX_COLS]), 0.0f32)),
        // span confidence head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, HIDDEN_DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 0.0f32)),
    ]
}

/// Build a table -> HTML confidence-weighted structure assembly kernel.
///
/// Combines cell classification with row/column assignment, weighted by
/// cell detection confidence. Simulates the assembly stage that produces
/// HTML table structure.
///
/// Architecture:
/// - Cell cls sigmoid -> confidence weights
/// - Row assignment softmax -> weighted by confidence
/// - Column assignment softmax -> weighted by confidence
/// - Output: confidence-weighted row + column assignments
///
/// Input: `[NUM_CELLS, HIDDEN_DIM]` (Variable, decoder output).
/// Output: `[NUM_CELLS, MAX_ROWS + MAX_COLS]` (confidence-weighted assignments).
fn build_table_html_assembly_kernel() -> TensorKernelDef {
    let row_shape = [NUM_CELLS, MAX_ROWS];
    let col_shape = [NUM_CELLS, MAX_COLS];
    let conf_shape = [NUM_CELLS, 1];
    let concat_shape = [NUM_CELLS, MAX_ROWS + MAX_COLS];
    let mut b = TensorBlockBuilder::new("table_structure_html_assembly");

    let input = b.add_input("decoder_output", &[NUM_CELLS, HIDDEN_DIM]);

    // Confidence head (sigmoid, single value per cell)
    let conf_w = b.add_input("conf_weight", &[1, HIDDEN_DIM]);
    let conf_b = b.add_input("conf_bias", &[1]);
    let conf_logits = b.add_linear(input, conf_w, Some(conf_b), &conf_shape);
    let conf = b.add_sigmoid(conf_logits, &conf_shape);

    // Row assignment (softmax)
    let row_w = b.add_input("row_weight", &[MAX_ROWS, HIDDEN_DIM]);
    let row_b = b.add_input("row_bias", &[MAX_ROWS]);
    let row_logits = b.add_linear(input, row_w, Some(row_b), &row_shape);
    let row_probs = b.add_softmax(row_logits, 1, &row_shape);

    // Broadcast confidence [NUM_CELLS, 1] -> [NUM_CELLS, MAX_ROWS]
    let conf_row = b.add_broadcast(conf, &row_shape);
    let weighted_row = b.add_binary_mul(conf_row, row_probs, &row_shape);

    // Column assignment (softmax)
    let col_w = b.add_input("col_weight", &[MAX_COLS, HIDDEN_DIM]);
    let col_b = b.add_input("col_bias", &[MAX_COLS]);
    let col_logits = b.add_linear(input, col_w, Some(col_b), &col_shape);
    let col_probs = b.add_softmax(col_logits, 1, &col_shape);

    // Broadcast confidence [NUM_CELLS, 1] -> [NUM_CELLS, MAX_COLS]
    let conf_col = b.add_broadcast(conf, &col_shape);
    let weighted_col = b.add_binary_mul(conf_col, col_probs, &col_shape);

    // Concatenate weighted row + col assignments
    let out = b.add_concat(&[weighted_row, weighted_col], 1, &concat_shape);

    b.build(out).expect("valid table HTML assembly kernel")
}

/// Bindings for table -> HTML assembly.
fn table_html_assembly_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable, // decoder_output
        // confidence head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, HIDDEN_DIM]), WEIGHT_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 0.0f32)),
        // row assignment head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAX_ROWS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAX_ROWS]), 0.0f32)),
        // col assignment head
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAX_COLS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[MAX_COLS]), 0.0f32)),
    ]
}

// ===========================================================================
// 1. Cell classification sigmoid: output in (0, 1) (IBP + CROWN)
// ===========================================================================

/// Cell classification sigmoid head: IBP bounds must be in (0, 1).
#[test]
fn test_cell_cls_sigmoid_ibp_crown() {
    let def = build_cell_cls_sigmoid_kernel();
    let bindings = cell_cls_sigmoid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    // IBP
    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through cell cls sigmoid");
    assert_eq!(
        ibp_output.lower_upper().0.shape(),
        &[NUM_CELLS, NUM_STRUCT_CLASSES],
        "cell cls sigmoid output shape mismatch"
    );
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Cell classification sigmoid IBP: bounds=[{lo_min}, {hi_max}]");
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
    let crown_input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 1.0);
    let (_method, crown_output, _fallback) =
        assert_crown_tighter_when_not_fallback(&graph, &crown_input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Cell cls sigmoid CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 2. Cell bbox regression sigmoid: normalized coordinates (IBP)
// ===========================================================================

/// Cell bbox regression sigmoid: all coordinates bounded in [0, 1].
#[test]
fn test_cell_bbox_sigmoid_ibp() {
    let def = build_cell_bbox_sigmoid_kernel();
    let bindings = cell_bbox_sigmoid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cell bbox sigmoid");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_CELLS, 4],
        "cell bbox output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cell bbox regression sigmoid IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "bbox sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "bbox sigmoid upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 3. Row separator detection: binary classification head (IBP)
// ===========================================================================

/// Row separator detection: sigmoid output in (0, 1).
#[test]
fn test_row_separator_ibp() {
    let def = build_row_separator_kernel();
    let bindings = row_separator_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through row separator");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_CELLS, 1],
        "row separator output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Row separator detection IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "row separator lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "row separator upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 4. Column separator detection: binary classification head (IBP)
// ===========================================================================

/// Column separator detection: sigmoid output in (0, 1).
#[test]
fn test_col_separator_ibp() {
    let def = build_col_separator_kernel();
    let bindings = col_separator_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through column separator");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_CELLS, 1],
        "column separator output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Column separator detection IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "column separator lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "column separator upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 5. Row count prediction: integer output bounded (IBP)
// ===========================================================================

/// Row count prediction: softmax produces valid probability distribution in [0, 1].
#[test]
fn test_row_count_prediction_ibp() {
    let def = build_row_count_kernel();
    let bindings = row_count_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[1, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through row count prediction");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[1, MAX_ROWS],
        "row count output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Row count prediction IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
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
// 6. Column count prediction: integer output bounded (IBP)
// ===========================================================================

/// Column count prediction: softmax produces valid probability distribution in [0, 1].
#[test]
fn test_col_count_prediction_ibp() {
    let def = build_col_count_kernel();
    let bindings = col_count_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[1, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through column count prediction");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[1, MAX_COLS],
        "column count output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Column count prediction IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
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
// 7. Cell-to-row assignment: softmax probability (IBP)
// ===========================================================================

/// Cell-to-row assignment: softmax per cell produces row probabilities in [0, 1].
#[test]
fn test_cell_to_row_assignment_ibp() {
    let def = build_cell_to_row_kernel();
    let bindings = cell_to_row_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cell-to-row assignment");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_CELLS, MAX_ROWS],
        "cell-to-row output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cell-to-row assignment IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
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
// 8. Cell-to-column assignment: softmax probability (IBP)
// ===========================================================================

/// Cell-to-column assignment: softmax per cell produces column probabilities in [0, 1].
#[test]
fn test_cell_to_col_assignment_ibp() {
    let def = build_cell_to_col_kernel();
    let bindings = cell_to_col_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cell-to-column assignment");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_CELLS, MAX_COLS],
        "cell-to-column output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cell-to-column assignment IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
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
// 9. Rowspan prediction: [1, max_rows] bounded (IBP)
// ===========================================================================

/// Rowspan prediction: sigmoid output in (0, 1), which post-processing
/// scales to [1, MAX_ROWS].
#[test]
fn test_rowspan_prediction_ibp() {
    let def = build_rowspan_kernel();
    let bindings = rowspan_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through rowspan prediction");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_CELLS, 1],
        "rowspan output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Rowspan prediction IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "rowspan sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "rowspan sigmoid upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 10. Colspan prediction: [1, max_cols] bounded (IBP)
// ===========================================================================

/// Colspan prediction: sigmoid output in (0, 1), which post-processing
/// scales to [1, MAX_COLS].
#[test]
fn test_colspan_prediction_ibp() {
    let def = build_colspan_kernel();
    let bindings = colspan_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through colspan prediction");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_CELLS, 1],
        "colspan output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Colspan prediction IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "colspan sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "colspan sigmoid upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 11. Span confidence: sigmoid gating for span detection (IBP + CROWN)
// ===========================================================================

/// Span confidence gating: sigmoid output in (0, 1) with CROWN tightening.
#[test]
fn test_span_confidence_ibp_crown() {
    let def = build_span_confidence_kernel();
    let bindings = span_confidence_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    // IBP
    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through span confidence");
    assert_eq!(
        ibp_output.lower_upper().0.shape(),
        &[NUM_CELLS, 1],
        "span confidence output shape mismatch"
    );
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Span confidence IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "span confidence lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "span confidence upper must be <= 1, got {hi_max}"
    );

    // CROWN
    let crown_input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 1.0);
    let (_method, crown_output, _fallback) =
        assert_crown_tighter_when_not_fallback(&graph, &crown_input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Span confidence CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 12. Detection -> structure pipeline: end-to-end (IBP)
// ===========================================================================

/// Detection -> structure pipeline: cls sigmoid + row softmax + col softmax,
/// all bounded in [0, 1].
#[test]
fn test_detection_structure_pipeline_ibp() {
    let def = build_detection_structure_pipeline_kernel();
    let bindings = detection_structure_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through detection-structure pipeline");

    let expected_dim = NUM_STRUCT_CLASSES + MAX_ROWS + MAX_COLS;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_CELLS, expected_dim],
        "detection-structure pipeline output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Detection -> structure pipeline IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
    // All heads (sigmoid + softmax) produce output in [0, 1]
    assert!(
        lo_min >= 0.0 - eps,
        "pipeline lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "pipeline upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 13. Structure monotone tightening: smaller eps -> tighter cell bounds (IBP)
// ===========================================================================

/// Monotone tightening: tighter input perturbation -> tighter output bounds.
///
/// Uses the cell classification sigmoid head as the test target.
/// Epsilon 1.0 should produce wider output bounds than epsilon 0.5.
#[test]
fn test_structure_monotone_tightening_ibp() {
    let def = build_cell_cls_sigmoid_kernel();
    let bindings = cell_cls_sigmoid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input_wide = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 1.0);
    let input_tight = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 0.5);

    let output_wide = graph.propagate_ibp(&input_wide).expect("IBP wide eps");
    let output_tight = graph.propagate_ibp(&input_tight).expect("IBP tight eps");

    let width_wide = bound_width(&output_wide);
    let width_tight = bound_width(&output_tight);

    eprintln!("Monotone tightening: eps=1.0 width={width_wide:.6}, eps=0.5 width={width_tight:.6}");

    assert!(
        width_tight <= width_wide + 1e-6,
        "tighter input (eps=0.5, width={width_tight}) must produce bounds \
         no wider than wider input (eps=1.0, width={width_wide})"
    );
}

// ===========================================================================
// 14. Multi-head table: detection + structure + span combined (IBP + CROWN)
// ===========================================================================

/// Multi-head table pipeline: 5 parallel heads from shared features.
/// All sigmoid outputs in (0, 1), all softmax outputs in [0, 1].
#[test]
fn test_multi_head_table_ibp_crown() {
    let def = build_multi_head_table_kernel();
    let bindings = multi_head_table_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    // IBP
    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through multi-head table");

    let total_dim = NUM_STRUCT_CLASSES + 4 + MAX_ROWS + MAX_COLS + 1;
    assert_eq!(
        ibp_output.lower_upper().0.shape(),
        &[NUM_CELLS, total_dim],
        "multi-head table output shape mismatch"
    );
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Multi-head table IBP: bounds=[{lo_min}, {hi_max}]");
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "all heads lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "all heads upper must be <= 1, got {hi_max}"
    );

    // CROWN
    let crown_input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 1.0);
    let (_method, crown_output, _fallback) =
        assert_crown_tighter_when_not_fallback(&graph, &crown_input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Multi-head table CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 15. Table -> HTML: confidence-weighted structure assembly (IBP)
// ===========================================================================

/// Table -> HTML assembly: confidence-weighted row/column assignments.
///
/// Since confidence is sigmoid in [0, 1] and assignments are softmax in [0, 1],
/// the product (confidence * assignment) must also be in [0, 1].
#[test]
fn test_table_html_assembly_ibp() {
    let def = build_table_html_assembly_kernel();
    let bindings = table_html_assembly_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through table HTML assembly");

    let expected_dim = MAX_ROWS + MAX_COLS;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_CELLS, expected_dim],
        "table HTML assembly output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table -> HTML assembly IBP: bounds=[{lo_min}, {hi_max}]");

    // Products of values in [0, 1] are themselves in [0, 1]
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "confidence-weighted lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "confidence-weighted upper must be <= 1, got {hi_max}"
    );
}
