// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for document reading order and layout spatial reasoning.
//!
//! Verifies IBP and CROWN bound propagation through the spatial reasoning
//! and reading order components used in document understanding models:
//!
//! ## Spatial Position Encoding (tests 1-3)
//!
//! 1. Box coordinate normalization (x, y, w, h in [0, 1]) IBP
//! 2. Pairwise box relationship features (overlap, distance) IBP
//! 3. Reading order classifier MLP bounds (IBP + CROWN)
//!
//! ## Layout Structure (tests 4-6)
//!
//! 4. Column detection spatial features bounds IBP
//! 5. Table cell adjacency feature bounds IBP
//! 6. Multi-column layout spatial reasoning bounds IBP
//!
//! ## Aggregation & Attention (tests 7-9)
//!
//! 7. Page-level aggregation bounds IBP
//! 8. Spatial self-attention for layout understanding IBP
//! 9. Box coordinate normalization (0-1 range) bounds IBP
//!
//! ## Classification & Output (tests 10-12)
//!
//! 10. Layout classification head (text, table, figure, etc.) IBP + CROWN
//! 11. Reading order pairwise comparison: sigmoid output bounded IBP
//! 12. Spatial distance features: L1/L2 distance bounded IBP
//!
//! ## Composition & Pipeline (tests 13-15)
//!
//! 13. Layout region merging features bounds IBP
//! 14. Hierarchical layout structure bounds (page -> column -> paragraph) IBP
//! 15. Full layout pipeline: detection -> spatial features -> ordering IBP + CROWN
//!
//! Dimensions (small for fast verification, structurally representative):
//! - NUM_BOXES=4, FEAT_DIM=16, NUM_CLASSES=6, NUM_PAIRS=6
//!
//! Part of #4015: Compose tests for document reading order and layout.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, ReduceOp, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Number of bounding boxes on a page.
const NUM_BOXES: usize = 4;
/// Feature dimension for spatial representations.
const FEAT_DIM: usize = 16;
/// Number of layout classes (text, table, figure, header, list, other).
const NUM_CLASSES: usize = 6;
/// Number of pairwise box combinations: C(NUM_BOXES, 2) = 6.
const NUM_PAIRS: usize = NUM_BOXES * (NUM_BOXES - 1) / 2;
/// Coordinate feature dimension (x, y, w, h) = 4.
const COORD_DIM: usize = 4;
/// MLP hidden dimension for reading order classifier.
const MLP_HIDDEN: usize = 32;
/// Number of attention heads for spatial self-attention.
const NUM_HEADS: usize = 4;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;
/// Number of columns for multi-column layout.
const NUM_COLUMNS: usize = 2;
/// Number of table cells for adjacency tests.
const NUM_CELLS: usize = 4;
/// Spatial feature dimension for pairwise relationships.
const SPATIAL_FEAT_DIM: usize = 8;
/// Number of hierarchical levels (page, column, paragraph).
const NUM_LEVELS: usize = 3;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute output bound width from a BoundedTensor.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

/// Create box coordinate bounds in [0, 1] range (normalized page coordinates).
fn box_coord_bounds(num_boxes: usize) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[num_boxes, COORD_DIM]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[num_boxes, COORD_DIM]), 1.0f32),
    )
    .expect("valid box coordinate bounds [0, 1]")
}

// ===========================================================================
// 1. Box coordinate normalization (x, y, w, h in [0, 1]) IBP
// ===========================================================================

/// Spatial position encoding: normalize box coordinates and project to features.
/// Input: [NUM_BOXES, COORD_DIM] (x, y, w, h in [0, 1])
/// Output: [NUM_BOXES, FEAT_DIM]
fn build_spatial_position_encoding_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_ro_spatial_pos_enc");
    let coords = b.add_input("box_coords", &[NUM_BOXES, COORD_DIM]);
    let proj_w = b.add_input("coord_proj_weight", &[FEAT_DIM, COORD_DIM]);
    let proj_b = b.add_input("coord_proj_bias", &[FEAT_DIM]);

    // Linear projection from coordinate space to feature space
    let features = b.add_linear(coords, proj_w, Some(proj_b), &[NUM_BOXES, FEAT_DIM]);
    // ReLU activation for non-negative spatial features
    let out = b.add_relu(features, &[NUM_BOXES, FEAT_DIM]);

    b.build(out)
        .expect("valid spatial position encoding kernel")
}

fn spatial_position_encoding_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[FEAT_DIM, COORD_DIM]), WEIGHT_MAG);
    let proj_b = ArrayD::from_elem(IxDyn(&[FEAT_DIM]), 0.0f32);
    vec![
        TensorParamBinding::Variable,               // box_coords
        TensorParamBinding::ConstantTensor(proj_w), // coord_proj_weight
        TensorParamBinding::ConstantTensor(proj_b), // coord_proj_bias
    ]
}

#[test]
fn test_spatial_position_encoding_ibp() {
    let def = build_spatial_position_encoding_kernel();
    let bindings = spatial_position_encoding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = box_coord_bounds(NUM_BOXES);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_BOXES, FEAT_DIM],
        "spatial position encoding output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Spatial position encoding IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // ReLU output: lower bound >= 0
    assert!(
        lo_min >= -1e-6,
        "ReLU output lower should be >= 0, got {lo_min}"
    );
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 2. Pairwise box relationship features (overlap, distance) IBP
// ===========================================================================

/// Pairwise spatial features between box pairs.
/// Models overlap/distance by projecting concatenated coordinate pairs.
/// Input: [NUM_PAIRS, 2 * COORD_DIM] (concatenated box pair coordinates)
/// Output: [NUM_PAIRS, SPATIAL_FEAT_DIM]
fn build_pairwise_box_features_kernel() -> TensorKernelDef {
    let pair_input_dim = 2 * COORD_DIM; // concatenated pair features
    let mut b = TensorBlockBuilder::new("dpdf_ro_pairwise_box_feat");
    let pair_coords = b.add_input("pair_coords", &[NUM_PAIRS, pair_input_dim]);
    let proj_w = b.add_input("pair_proj_weight", &[SPATIAL_FEAT_DIM, pair_input_dim]);
    let proj_b = b.add_input("pair_proj_bias", &[SPATIAL_FEAT_DIM]);

    // Project concatenated pair coordinates to spatial features
    let features = b.add_linear(
        pair_coords,
        proj_w,
        Some(proj_b),
        &[NUM_PAIRS, SPATIAL_FEAT_DIM],
    );
    let out = b.add_relu(features, &[NUM_PAIRS, SPATIAL_FEAT_DIM]);

    b.build(out).expect("valid pairwise box features kernel")
}

#[test]
fn test_pairwise_box_features_ibp() {
    let pair_input_dim = 2 * COORD_DIM;
    let def = build_pairwise_box_features_kernel();
    let proj_w = ArrayD::from_elem(IxDyn(&[SPATIAL_FEAT_DIM, pair_input_dim]), WEIGHT_MAG);
    let proj_b = ArrayD::from_elem(IxDyn(&[SPATIAL_FEAT_DIM]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,               // pair_coords
        TensorParamBinding::ConstantTensor(proj_w), // pair_proj_weight
        TensorParamBinding::ConstantTensor(proj_b), // pair_proj_bias
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Pairwise coordinates: each pair is [box_i coords, box_j coords], all in [0, 1]
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[NUM_PAIRS, pair_input_dim]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[NUM_PAIRS, pair_input_dim]), 1.0f32),
    )
    .expect("valid pairwise input bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pairwise box features IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-6, "ReLU output lower >= 0, got {lo_min}");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 3. Reading order classifier MLP bounds (IBP + CROWN)
// ===========================================================================

/// MLP classifier that predicts reading order from pairwise spatial features.
/// Input: [NUM_PAIRS, SPATIAL_FEAT_DIM]
/// Output: [NUM_PAIRS, 1] (sigmoid probability: box_i before box_j)
fn build_reading_order_mlp_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_ro_classifier_mlp");
    let input = b.add_input("pair_features", &[NUM_PAIRS, SPATIAL_FEAT_DIM]);
    let w1 = b.add_input("mlp_w1", &[MLP_HIDDEN, SPATIAL_FEAT_DIM]);
    let b1 = b.add_input("mlp_b1", &[MLP_HIDDEN]);
    let w2 = b.add_input("mlp_w2", &[1, MLP_HIDDEN]);
    let b2 = b.add_input("mlp_b2", &[1]);

    // Hidden layer
    let hidden = b.add_linear(input, w1, Some(b1), &[NUM_PAIRS, MLP_HIDDEN]);
    let activated = b.add_relu(hidden, &[NUM_PAIRS, MLP_HIDDEN]);
    // Output layer -> sigmoid
    let logits = b.add_linear(activated, w2, Some(b2), &[NUM_PAIRS, 1]);
    let out = b.add_sigmoid(logits, &[NUM_PAIRS, 1]);

    b.build(out).expect("valid reading order MLP kernel")
}

fn reading_order_mlp_bindings() -> Vec<TensorParamBinding> {
    let w1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN, SPATIAL_FEAT_DIM]), WEIGHT_MAG);
    let b1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN]), 0.0f32);
    let w2 = ArrayD::from_elem(IxDyn(&[1, MLP_HIDDEN]), WEIGHT_MAG);
    let b2 = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);
    vec![
        TensorParamBinding::Variable,           // pair_features
        TensorParamBinding::ConstantTensor(w1), // mlp_w1
        TensorParamBinding::ConstantTensor(b1), // mlp_b1
        TensorParamBinding::ConstantTensor(w2), // mlp_w2
        TensorParamBinding::ConstantTensor(b2), // mlp_b2
    ]
}

#[test]
fn test_reading_order_mlp_ibp() {
    let def = build_reading_order_mlp_kernel();
    let bindings = reading_order_mlp_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_PAIRS, SPATIAL_FEAT_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Reading order MLP IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Sigmoid output must be in (0, 1)
    assert!(lo_min >= -1e-6, "sigmoid output lower >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-6,
        "sigmoid output upper <= 1.0, got {hi_max}"
    );
}

#[test]
fn test_reading_order_mlp_crown() {
    let def = build_reading_order_mlp_kernel();
    let bindings = reading_order_mlp_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_PAIRS, SPATIAL_FEAT_DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Reading order MLP CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 4. Column detection spatial features bounds IBP
// ===========================================================================

/// Column detection: project box features to column assignment logits.
/// Input: [NUM_BOXES, FEAT_DIM] (per-box spatial features)
/// Output: [NUM_BOXES, NUM_COLUMNS] (softmax column assignment probabilities)
fn build_column_detection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_ro_column_detect");
    let input = b.add_input("box_features", &[NUM_BOXES, FEAT_DIM]);
    let w = b.add_input("col_proj_weight", &[NUM_COLUMNS, FEAT_DIM]);
    let bias = b.add_input("col_proj_bias", &[NUM_COLUMNS]);

    let logits = b.add_linear(input, w, Some(bias), &[NUM_BOXES, NUM_COLUMNS]);
    let out = b.add_softmax(logits, 1, &[NUM_BOXES, NUM_COLUMNS]);

    b.build(out).expect("valid column detection kernel")
}

#[test]
fn test_column_detection_ibp() {
    let def = build_column_detection_kernel();
    let w = ArrayD::from_elem(IxDyn(&[NUM_COLUMNS, FEAT_DIM]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[NUM_COLUMNS]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,             // box_features
        TensorParamBinding::ConstantTensor(w),    // col_proj_weight
        TensorParamBinding::ConstantTensor(bias), // col_proj_bias
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_BOXES, FEAT_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Column detection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Softmax output: all values in [0, 1]
    assert!(lo_min >= -1e-6, "softmax output lower >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-6,
        "softmax output upper <= 1.0, got {hi_max}"
    );
}

// ===========================================================================
// 5. Table cell adjacency feature bounds IBP
// ===========================================================================

/// Table cell adjacency: project cell pair features to adjacency score.
/// Input: [NUM_CELLS, 2 * FEAT_DIM] (concatenated cell feature pairs)
/// Output: [NUM_CELLS, 1] (sigmoid adjacency probability)
fn build_table_cell_adjacency_kernel() -> TensorKernelDef {
    let pair_dim = 2 * FEAT_DIM;
    let mut b = TensorBlockBuilder::new("dpdf_ro_table_cell_adj");
    let input = b.add_input("cell_pair_features", &[NUM_CELLS, pair_dim]);
    let w1 = b.add_input("adj_w1", &[FEAT_DIM, pair_dim]);
    let b1 = b.add_input("adj_b1", &[FEAT_DIM]);
    let w2 = b.add_input("adj_w2", &[1, FEAT_DIM]);
    let b2 = b.add_input("adj_b2", &[1]);

    let hidden = b.add_linear(input, w1, Some(b1), &[NUM_CELLS, FEAT_DIM]);
    let activated = b.add_relu(hidden, &[NUM_CELLS, FEAT_DIM]);
    let logits = b.add_linear(activated, w2, Some(b2), &[NUM_CELLS, 1]);
    let out = b.add_sigmoid(logits, &[NUM_CELLS, 1]);

    b.build(out).expect("valid table cell adjacency kernel")
}

#[test]
fn test_table_cell_adjacency_ibp() {
    let pair_dim = 2 * FEAT_DIM;
    let def = build_table_cell_adjacency_kernel();
    let w1 = ArrayD::from_elem(IxDyn(&[FEAT_DIM, pair_dim]), WEIGHT_MAG);
    let b1 = ArrayD::from_elem(IxDyn(&[FEAT_DIM]), 0.0f32);
    let w2 = ArrayD::from_elem(IxDyn(&[1, FEAT_DIM]), WEIGHT_MAG);
    let b2 = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,           // cell_pair_features
        TensorParamBinding::ConstantTensor(w1), // adj_w1
        TensorParamBinding::ConstantTensor(b1), // adj_b1
        TensorParamBinding::ConstantTensor(w2), // adj_w2
        TensorParamBinding::ConstantTensor(b2), // adj_b2
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, pair_dim], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table cell adjacency IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Sigmoid output in (0, 1)
    assert!(lo_min >= -1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1.0, got {hi_max}");
}

// ===========================================================================
// 6. Multi-column layout spatial reasoning bounds IBP
// ===========================================================================

/// Multi-column layout reasoning: Linear -> ReLU -> Linear -> softmax.
/// Assigns boxes to columns based on spatial features.
/// Input: [NUM_BOXES, FEAT_DIM] (spatial features per box)
/// Output: [NUM_BOXES, NUM_COLUMNS] (column assignment probabilities)
fn build_multi_column_layout_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_ro_multi_col_layout");
    let input = b.add_input("spatial_features", &[NUM_BOXES, FEAT_DIM]);
    let w1 = b.add_input("col_w1", &[MLP_HIDDEN, FEAT_DIM]);
    let b1 = b.add_input("col_b1", &[MLP_HIDDEN]);
    let w2 = b.add_input("col_w2", &[NUM_COLUMNS, MLP_HIDDEN]);
    let b2 = b.add_input("col_b2", &[NUM_COLUMNS]);

    let hidden = b.add_linear(input, w1, Some(b1), &[NUM_BOXES, MLP_HIDDEN]);
    let activated = b.add_relu(hidden, &[NUM_BOXES, MLP_HIDDEN]);
    let logits = b.add_linear(activated, w2, Some(b2), &[NUM_BOXES, NUM_COLUMNS]);
    let out = b.add_softmax(logits, 1, &[NUM_BOXES, NUM_COLUMNS]);

    b.build(out).expect("valid multi-column layout kernel")
}

#[test]
fn test_multi_column_layout_ibp() {
    let def = build_multi_column_layout_kernel();
    let w1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN, FEAT_DIM]), WEIGHT_MAG);
    let b1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN]), 0.0f32);
    let w2 = ArrayD::from_elem(IxDyn(&[NUM_COLUMNS, MLP_HIDDEN]), WEIGHT_MAG);
    let b2 = ArrayD::from_elem(IxDyn(&[NUM_COLUMNS]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,           // spatial_features
        TensorParamBinding::ConstantTensor(w1), // col_w1
        TensorParamBinding::ConstantTensor(b1), // col_b1
        TensorParamBinding::ConstantTensor(w2), // col_w2
        TensorParamBinding::ConstantTensor(b2), // col_b2
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_BOXES, FEAT_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Multi-column layout IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Softmax output in [0, 1]
    assert!(lo_min >= -1e-6, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "softmax upper <= 1.0, got {hi_max}");
}

// ===========================================================================
// 7. Page-level aggregation bounds IBP
// ===========================================================================

/// Page-level aggregation: project per-box features and reduce (mean) to page.
/// Input: [NUM_BOXES, FEAT_DIM] (per-box features)
/// Output: [FEAT_DIM] (page-level summary via mean pooling + linear)
fn build_page_aggregation_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_ro_page_agg");
    let input = b.add_input("box_features", &[NUM_BOXES, FEAT_DIM]);

    // Mean pooling over boxes (reduce axis=0)
    let pooled = b.add_reduce(input, ReduceOp::Mean, 0, false, &[FEAT_DIM]);

    b.build(pooled).expect("valid page aggregation kernel")
}

#[test]
fn test_page_aggregation_ibp() {
    let def = build_page_aggregation_kernel();
    let bindings = vec![
        TensorParamBinding::Variable, // box_features
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_BOXES, FEAT_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Page aggregation IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Mean of [-1, 1] values should stay in [-1, 1]
    assert!(
        lo_min >= -1.0 - 1e-6,
        "page agg lower >= -1.0, got {lo_min}"
    );
    assert!(hi_max <= 1.0 + 1e-6, "page agg upper <= 1.0, got {hi_max}");
}

// ===========================================================================
// 8. Spatial self-attention for layout understanding IBP
// ===========================================================================

/// Spatial self-attention: boxes attend to each other based on spatial features.
/// Input: [NUM_BOXES, FEAT_DIM] (spatial features)
/// Output: [NUM_BOXES, FEAT_DIM] (attention-refined features)
fn build_spatial_self_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_ro_spatial_self_attn");
    let input = b.add_input("box_features", &[NUM_BOXES, FEAT_DIM]);
    let q_w = b.add_input("q_weight", &[FEAT_DIM, FEAT_DIM]);
    let k_w = b.add_input("k_weight", &[FEAT_DIM, FEAT_DIM]);
    let v_w = b.add_input("v_weight", &[FEAT_DIM, FEAT_DIM]);
    let out_w = b.add_input("out_weight", &[FEAT_DIM, FEAT_DIM]);

    let attn_out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[NUM_BOXES, FEAT_DIM],
        )
        .expect("valid MHA");

    // Residual connection
    let out = b.add_binary_add(input, attn_out, &[NUM_BOXES, FEAT_DIM]);

    b.build(out).expect("valid spatial self-attention kernel")
}

fn spatial_self_attention_bindings() -> Vec<TensorParamBinding> {
    let q_w = ArrayD::from_elem(IxDyn(&[FEAT_DIM, FEAT_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[FEAT_DIM, FEAT_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[FEAT_DIM, FEAT_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[FEAT_DIM, FEAT_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,              // box_features
        TensorParamBinding::ConstantTensor(q_w),   // q_weight
        TensorParamBinding::ConstantTensor(k_w),   // k_weight
        TensorParamBinding::ConstantTensor(v_w),   // v_weight
        TensorParamBinding::ConstantTensor(out_w), // out_weight
    ]
}

#[test]
fn test_spatial_self_attention_ibp() {
    let def = build_spatial_self_attention_kernel();
    let bindings = spatial_self_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_BOXES, FEAT_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_BOXES, FEAT_DIM],
        "spatial self-attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Spatial self-attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 9. Box coordinate normalization (0-1 range) bounds IBP
// ===========================================================================

/// Box coordinate normalization: project raw coordinates and sigmoid to [0, 1].
/// Input: [NUM_BOXES, COORD_DIM] (raw coordinates, variable range)
/// Output: [NUM_BOXES, COORD_DIM] (normalized coordinates in (0, 1))
fn build_box_coord_normalization_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_ro_box_coord_norm");
    let input = b.add_input("raw_coords", &[NUM_BOXES, COORD_DIM]);
    let w = b.add_input("norm_weight", &[COORD_DIM, COORD_DIM]);
    let bias = b.add_input("norm_bias", &[COORD_DIM]);

    let projected = b.add_linear(input, w, Some(bias), &[NUM_BOXES, COORD_DIM]);
    let out = b.add_sigmoid(projected, &[NUM_BOXES, COORD_DIM]);

    b.build(out).expect("valid box coord normalization kernel")
}

#[test]
fn test_box_coord_normalization_ibp() {
    let def = build_box_coord_normalization_kernel();
    let w = ArrayD::from_elem(IxDyn(&[COORD_DIM, COORD_DIM]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[COORD_DIM]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,             // raw_coords
        TensorParamBinding::ConstantTensor(w),    // norm_weight
        TensorParamBinding::ConstantTensor(bias), // norm_bias
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Raw coordinates can be in a wider range before normalization
    let input = uniform_bounds(&[NUM_BOXES, COORD_DIM], 5.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Box coord normalization IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Sigmoid output must be in (0, 1)
    assert!(lo_min >= -1e-6, "sigmoid output lower >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-6,
        "sigmoid output upper <= 1.0, got {hi_max}"
    );
}

// ===========================================================================
// 10. Layout classification head (text, table, figure, etc.) IBP + CROWN
// ===========================================================================

/// Layout classification: Linear -> ReLU -> Linear -> sigmoid multi-label.
/// Input: [NUM_BOXES, FEAT_DIM] (per-box features)
/// Output: [NUM_BOXES, NUM_CLASSES] (per-class probability in (0, 1))
fn build_layout_classification_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_ro_layout_cls");
    let input = b.add_input("box_features", &[NUM_BOXES, FEAT_DIM]);
    let w1 = b.add_input("cls_w1", &[MLP_HIDDEN, FEAT_DIM]);
    let b1 = b.add_input("cls_b1", &[MLP_HIDDEN]);
    let w2 = b.add_input("cls_w2", &[NUM_CLASSES, MLP_HIDDEN]);
    let b2 = b.add_input("cls_b2", &[NUM_CLASSES]);

    let hidden = b.add_linear(input, w1, Some(b1), &[NUM_BOXES, MLP_HIDDEN]);
    let activated = b.add_relu(hidden, &[NUM_BOXES, MLP_HIDDEN]);
    let logits = b.add_linear(activated, w2, Some(b2), &[NUM_BOXES, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[NUM_BOXES, NUM_CLASSES]);

    b.build(out).expect("valid layout classification kernel")
}

fn layout_classification_bindings() -> Vec<TensorParamBinding> {
    let w1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN, FEAT_DIM]), WEIGHT_MAG);
    let b1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN]), 0.0f32);
    let w2 = ArrayD::from_elem(IxDyn(&[NUM_CLASSES, MLP_HIDDEN]), WEIGHT_MAG);
    let b2 = ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32);
    vec![
        TensorParamBinding::Variable,           // box_features
        TensorParamBinding::ConstantTensor(w1), // cls_w1
        TensorParamBinding::ConstantTensor(b1), // cls_b1
        TensorParamBinding::ConstantTensor(w2), // cls_w2
        TensorParamBinding::ConstantTensor(b2), // cls_b2
    ]
}

#[test]
fn test_layout_classification_ibp() {
    let def = build_layout_classification_kernel();
    let bindings = layout_classification_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_BOXES, FEAT_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Layout classification IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Sigmoid multi-label output in (0, 1)
    assert!(lo_min >= -1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1.0, got {hi_max}");
}

#[test]
fn test_layout_classification_crown() {
    let def = build_layout_classification_kernel();
    let bindings = layout_classification_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_BOXES, FEAT_DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!(
        "Layout classification CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}"
    );
}

// ===========================================================================
// 11. Reading order pairwise comparison: sigmoid output bounded IBP
// ===========================================================================

/// Simple pairwise comparison: single linear layer + sigmoid.
/// Input: [NUM_PAIRS, 2 * COORD_DIM] (concatenated box coordinate pairs)
/// Output: [NUM_PAIRS, 1] (probability box_i before box_j)
fn build_pairwise_comparison_kernel() -> TensorKernelDef {
    let pair_dim = 2 * COORD_DIM;
    let mut b = TensorBlockBuilder::new("dpdf_ro_pairwise_cmp");
    let input = b.add_input("pair_coords", &[NUM_PAIRS, pair_dim]);
    let w = b.add_input("cmp_weight", &[1, pair_dim]);
    let bias = b.add_input("cmp_bias", &[1]);

    let logits = b.add_linear(input, w, Some(bias), &[NUM_PAIRS, 1]);
    let out = b.add_sigmoid(logits, &[NUM_PAIRS, 1]);

    b.build(out).expect("valid pairwise comparison kernel")
}

#[test]
fn test_pairwise_comparison_sigmoid_bounded_ibp() {
    let pair_dim = 2 * COORD_DIM;
    let def = build_pairwise_comparison_kernel();
    let w = ArrayD::from_elem(IxDyn(&[1, pair_dim]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,             // pair_coords
        TensorParamBinding::ConstantTensor(w),    // cmp_weight
        TensorParamBinding::ConstantTensor(bias), // cmp_bias
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[NUM_PAIRS, pair_dim]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[NUM_PAIRS, pair_dim]), 1.0f32),
    )
    .expect("valid pair input bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pairwise comparison sigmoid IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1.0, got {hi_max}");
}

// ===========================================================================
// 12. Spatial distance features: L1/L2 distance bounded IBP
// ===========================================================================

/// Spatial distance features: project coordinate differences to distance features.
/// Models L1/L2 distance as Linear(|coord_i - coord_j|) approximation.
/// Input: [NUM_PAIRS, COORD_DIM] (absolute coordinate differences)
/// Output: [NUM_PAIRS, SPATIAL_FEAT_DIM] (distance features)
fn build_spatial_distance_features_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_ro_spatial_dist_feat");
    let input = b.add_input("coord_diffs", &[NUM_PAIRS, COORD_DIM]);
    let w = b.add_input("dist_proj_weight", &[SPATIAL_FEAT_DIM, COORD_DIM]);
    let bias = b.add_input("dist_proj_bias", &[SPATIAL_FEAT_DIM]);

    let projected = b.add_linear(input, w, Some(bias), &[NUM_PAIRS, SPATIAL_FEAT_DIM]);
    let out = b.add_relu(projected, &[NUM_PAIRS, SPATIAL_FEAT_DIM]);

    b.build(out)
        .expect("valid spatial distance features kernel")
}

#[test]
fn test_spatial_distance_features_ibp() {
    let def = build_spatial_distance_features_kernel();
    let w = ArrayD::from_elem(IxDyn(&[SPATIAL_FEAT_DIM, COORD_DIM]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[SPATIAL_FEAT_DIM]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,             // coord_diffs
        TensorParamBinding::ConstantTensor(w),    // dist_proj_weight
        TensorParamBinding::ConstantTensor(bias), // dist_proj_bias
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Absolute coordinate differences in [0, 1] (page-normalized)
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[NUM_PAIRS, COORD_DIM]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[NUM_PAIRS, COORD_DIM]), 1.0f32),
    )
    .expect("valid distance input bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Spatial distance features IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // ReLU output: lower >= 0
    assert!(lo_min >= -1e-6, "ReLU output lower >= 0, got {lo_min}");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 13. Layout region merging features bounds IBP
// ===========================================================================

/// Layout region merging: merge features from adjacent regions.
/// Models region merging as feature concatenation -> Linear -> ReLU -> Linear -> sigmoid.
/// Input: [NUM_BOXES, 2 * FEAT_DIM] (concatenated adjacent region features)
/// Output: [NUM_BOXES, 1] (merge probability)
fn build_region_merging_kernel() -> TensorKernelDef {
    let merge_dim = 2 * FEAT_DIM;
    let mut b = TensorBlockBuilder::new("dpdf_ro_region_merge");
    let input = b.add_input("region_pair_features", &[NUM_BOXES, merge_dim]);
    let w1 = b.add_input("merge_w1", &[FEAT_DIM, merge_dim]);
    let b1 = b.add_input("merge_b1", &[FEAT_DIM]);
    let w2 = b.add_input("merge_w2", &[1, FEAT_DIM]);
    let b2 = b.add_input("merge_b2", &[1]);

    let hidden = b.add_linear(input, w1, Some(b1), &[NUM_BOXES, FEAT_DIM]);
    let activated = b.add_relu(hidden, &[NUM_BOXES, FEAT_DIM]);
    let logits = b.add_linear(activated, w2, Some(b2), &[NUM_BOXES, 1]);
    let out = b.add_sigmoid(logits, &[NUM_BOXES, 1]);

    b.build(out).expect("valid region merging kernel")
}

#[test]
fn test_region_merging_ibp() {
    let merge_dim = 2 * FEAT_DIM;
    let def = build_region_merging_kernel();
    let w1 = ArrayD::from_elem(IxDyn(&[FEAT_DIM, merge_dim]), WEIGHT_MAG);
    let b1 = ArrayD::from_elem(IxDyn(&[FEAT_DIM]), 0.0f32);
    let w2 = ArrayD::from_elem(IxDyn(&[1, FEAT_DIM]), WEIGHT_MAG);
    let b2 = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,           // region_pair_features
        TensorParamBinding::ConstantTensor(w1), // merge_w1
        TensorParamBinding::ConstantTensor(b1), // merge_b1
        TensorParamBinding::ConstantTensor(w2), // merge_w2
        TensorParamBinding::ConstantTensor(b2), // merge_b2
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_BOXES, merge_dim], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Region merging IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Sigmoid output in (0, 1)
    assert!(lo_min >= -1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1.0, got {hi_max}");
}

// ===========================================================================
// 14. Hierarchical layout structure bounds (page -> column -> paragraph) IBP
// ===========================================================================

/// Hierarchical layout: classify boxes at multiple hierarchy levels.
/// Input: [NUM_BOXES, FEAT_DIM] (per-box features)
/// Output: [NUM_BOXES, NUM_LEVELS] (softmax probability per hierarchy level)
fn build_hierarchical_layout_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_ro_hierarchical_layout");
    let input = b.add_input("box_features", &[NUM_BOXES, FEAT_DIM]);
    let w1 = b.add_input("hier_w1", &[MLP_HIDDEN, FEAT_DIM]);
    let b1 = b.add_input("hier_b1", &[MLP_HIDDEN]);
    let w2 = b.add_input("hier_w2", &[NUM_LEVELS, MLP_HIDDEN]);
    let b2 = b.add_input("hier_b2", &[NUM_LEVELS]);

    let hidden = b.add_linear(input, w1, Some(b1), &[NUM_BOXES, MLP_HIDDEN]);
    let activated = b.add_relu(hidden, &[NUM_BOXES, MLP_HIDDEN]);
    let logits = b.add_linear(activated, w2, Some(b2), &[NUM_BOXES, NUM_LEVELS]);
    let out = b.add_softmax(logits, 1, &[NUM_BOXES, NUM_LEVELS]);

    b.build(out).expect("valid hierarchical layout kernel")
}

#[test]
fn test_hierarchical_layout_ibp() {
    let def = build_hierarchical_layout_kernel();
    let w1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN, FEAT_DIM]), WEIGHT_MAG);
    let b1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN]), 0.0f32);
    let w2 = ArrayD::from_elem(IxDyn(&[NUM_LEVELS, MLP_HIDDEN]), WEIGHT_MAG);
    let b2 = ArrayD::from_elem(IxDyn(&[NUM_LEVELS]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,           // box_features
        TensorParamBinding::ConstantTensor(w1), // hier_w1
        TensorParamBinding::ConstantTensor(b1), // hier_b1
        TensorParamBinding::ConstantTensor(w2), // hier_w2
        TensorParamBinding::ConstantTensor(b2), // hier_b2
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_BOXES, FEAT_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Hierarchical layout IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Softmax output in [0, 1]
    assert!(lo_min >= -1e-6, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "softmax upper <= 1.0, got {hi_max}");
}

// ===========================================================================
// 15. Full layout pipeline: detection -> spatial features -> ordering IBP + CROWN
// ===========================================================================

/// Full layout pipeline end-to-end:
/// 1. Spatial position encoding (coords -> features)
/// 2. Spatial self-attention (feature refinement)
/// 3. Layout classification head (sigmoid multi-label output)
///
/// Input: [NUM_BOXES, COORD_DIM] (normalized box coordinates)
/// Output: [NUM_BOXES, NUM_CLASSES] (layout class probabilities in (0, 1))
fn build_full_layout_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_ro_full_layout_pipeline");

    // Stage 1: Coordinate projection
    let coords = b.add_input("box_coords", &[NUM_BOXES, COORD_DIM]);
    let coord_proj_w = b.add_input("coord_proj_weight", &[FEAT_DIM, COORD_DIM]);
    let coord_proj_b = b.add_input("coord_proj_bias", &[FEAT_DIM]);

    let features = b.add_linear(
        coords,
        coord_proj_w,
        Some(coord_proj_b),
        &[NUM_BOXES, FEAT_DIM],
    );
    let features = b.add_relu(features, &[NUM_BOXES, FEAT_DIM]);

    // Stage 2: Spatial self-attention
    let q_w = b.add_input("q_weight", &[FEAT_DIM, FEAT_DIM]);
    let k_w = b.add_input("k_weight", &[FEAT_DIM, FEAT_DIM]);
    let v_w = b.add_input("v_weight", &[FEAT_DIM, FEAT_DIM]);
    let out_w = b.add_input("out_weight", &[FEAT_DIM, FEAT_DIM]);

    let attn_out = b
        .add_multi_head_attention(
            features,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[NUM_BOXES, FEAT_DIM],
        )
        .expect("valid MHA");
    let refined = b.add_binary_add(features, attn_out, &[NUM_BOXES, FEAT_DIM]);

    // Stage 3: Classification head
    let cls_w1 = b.add_input("cls_w1", &[MLP_HIDDEN, FEAT_DIM]);
    let cls_b1 = b.add_input("cls_b1", &[MLP_HIDDEN]);
    let cls_w2 = b.add_input("cls_w2", &[NUM_CLASSES, MLP_HIDDEN]);
    let cls_b2 = b.add_input("cls_b2", &[NUM_CLASSES]);

    let hidden = b.add_linear(refined, cls_w1, Some(cls_b1), &[NUM_BOXES, MLP_HIDDEN]);
    let activated = b.add_relu(hidden, &[NUM_BOXES, MLP_HIDDEN]);
    let logits = b.add_linear(activated, cls_w2, Some(cls_b2), &[NUM_BOXES, NUM_CLASSES]);
    let out = b.add_sigmoid(logits, &[NUM_BOXES, NUM_CLASSES]);

    b.build(out).expect("valid full layout pipeline kernel")
}

fn full_layout_pipeline_bindings() -> Vec<TensorParamBinding> {
    let coord_proj_w = ArrayD::from_elem(IxDyn(&[FEAT_DIM, COORD_DIM]), WEIGHT_MAG);
    let coord_proj_b = ArrayD::from_elem(IxDyn(&[FEAT_DIM]), 0.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[FEAT_DIM, FEAT_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[FEAT_DIM, FEAT_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[FEAT_DIM, FEAT_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[FEAT_DIM, FEAT_DIM]), WEIGHT_MAG);
    let cls_w1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN, FEAT_DIM]), WEIGHT_MAG);
    let cls_b1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN]), 0.0f32);
    let cls_w2 = ArrayD::from_elem(IxDyn(&[NUM_CLASSES, MLP_HIDDEN]), WEIGHT_MAG);
    let cls_b2 = ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32);
    vec![
        TensorParamBinding::Variable,                     // box_coords
        TensorParamBinding::ConstantTensor(coord_proj_w), // coord_proj_weight
        TensorParamBinding::ConstantTensor(coord_proj_b), // coord_proj_bias
        TensorParamBinding::ConstantTensor(q_w),          // q_weight
        TensorParamBinding::ConstantTensor(k_w),          // k_weight
        TensorParamBinding::ConstantTensor(v_w),          // v_weight
        TensorParamBinding::ConstantTensor(out_w),        // out_weight
        TensorParamBinding::ConstantTensor(cls_w1),       // cls_w1
        TensorParamBinding::ConstantTensor(cls_b1),       // cls_b1
        TensorParamBinding::ConstantTensor(cls_w2),       // cls_w2
        TensorParamBinding::ConstantTensor(cls_b2),       // cls_b2
    ]
}

#[test]
fn test_full_layout_pipeline_ibp() {
    let def = build_full_layout_pipeline_kernel();
    let bindings = full_layout_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = box_coord_bounds(NUM_BOXES);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_BOXES, NUM_CLASSES],
        "full pipeline output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full layout pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Sigmoid output in (0, 1)
    assert!(lo_min >= -1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1.0, got {hi_max}");
}

#[test]
fn test_full_layout_pipeline_crown() {
    let def = build_full_layout_pipeline_kernel();
    let bindings = full_layout_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Narrower input bounds for CROWN tractability
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[NUM_BOXES, COORD_DIM]), 0.2f32),
        ArrayD::from_elem(IxDyn(&[NUM_BOXES, COORD_DIM]), 0.8f32),
    )
    .expect("valid narrowed input bounds");

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Full layout pipeline CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}
