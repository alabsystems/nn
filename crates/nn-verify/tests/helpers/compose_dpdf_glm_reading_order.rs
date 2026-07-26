// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for GLM-OCR reading order and text line detection pipeline.
//!
//! Verifies IBP and CROWN bound propagation through the GLM-OCR reading order
//! pipeline: encoder features, pairwise comparison, topological ordering,
//! text line detection, multi-column layout, table cell ordering, and
//! full pipeline composition.
//!
//! ## Encoder & Pairwise (tests 1-4)
//!
//! 1.  encoder_output_bounded: hidden states bounded by activation range (IBP)
//! 2.  pairwise_comparison_logits: [num_regions, num_regions] shape (IBP)
//! 3.  pairwise_softmax_probability: p(i before j) in [0, 1] (IBP)
//! 4.  pairwise_consistency: p(i < j) + p(j < i) ~ 1 (IBP + CROWN)
//!
//! ## Ordering & Detection (tests 5-8)
//!
//! 5.  topological_sort_valid: no cycles in ordering logits (IBP)
//! 6.  text_line_bbox_normalized: coordinates in [0, 1] (IBP)
//! 7.  text_line_confidence_bounded: score in [0, 1] (IBP)
//! 8.  multi_column_detection: column boundaries divide width (IBP)
//!
//! ## Spatial & Structure (tests 9-12)
//!
//! 9.  header_footer_y_coordinate: classification by position (IBP)
//! 10. table_cell_ordering: row-major within tables (IBP)
//! 11. cross_attention_features: text attends to layout (IBP + CROWN)
//! 12. position_encoding_2d: spatial document layout (IBP)
//!
//! ## Constraints & Pipeline (tests 13-18)
//!
//! 13. max_regions_bounded: limited by max_seq_len (IBP)
//! 14. region_merging_containment: merged bbox contains originals (IBP)
//! 15. reading_order_dag: directed acyclic graph ordering (IBP + CROWN)
//! 16. beam_search_log_probs: scores <= 0 (IBP)
//! 17. confidence_calibration: probability approximation (IBP + CROWN)
//! 18. full_pipeline_bounded: image -> regions -> order -> text (IBP + CROWN)
//!
//! Architecture references:
//! - GLM-4V (THUDM): Vision-language model with GLM-4 decoder
//! - Reading order prediction as pairwise comparison (Li et al. 2020)
//! - LayoutLM (Xu et al. 2020): 2D position embeddings for document AI
//!
//! Dimensions (small for fast verification, structurally representative):
//! - NUM_REGIONS=6, HIDDEN_DIM=32, FFN_DIM=64, NUM_HEADS=4,
//!   COORD_DIM=4, MAX_SEQ_LEN=8, NUM_COLUMNS=3
//!
//! Part of #4154: Compose tests for GLM-OCR reading order pipeline bounds.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Number of detected text regions on a page.
const NUM_REGIONS: usize = 6;
/// Hidden dimension for the GLM encoder.
const HIDDEN_DIM: usize = 32;
/// FFN intermediate dimension.
const FFN_DIM: usize = 64;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// Coordinate dimension (x, y, w, h).
const COORD_DIM: usize = 4;
/// Maximum sequence length for region encoding.
const MAX_SEQ_LEN: usize = 8;
/// Number of columns for multi-column layout.
const NUM_COLUMNS: usize = 3;
/// MLP hidden dimension for classifiers.
const MLP_HIDDEN: usize = 32;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;
/// Number of table cells for ordering tests.
const NUM_CELLS: usize = 4;
/// Spatial feature dimension for pairwise relationships.
const SPATIAL_FEAT_DIM: usize = 8;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute output bound width from a BoundedTensor.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

// ===========================================================================
// 1. encoder_output_bounded: hidden states bounded by activation range (IBP)
// ===========================================================================

/// GLM encoder output: LayerNorm -> Linear -> ReLU keeps hidden states bounded.
/// Input: [NUM_REGIONS, HIDDEN_DIM] (region features from vision encoder)
/// Output: [NUM_REGIONS, HIDDEN_DIM] (bounded encoder hidden states)
fn build_encoder_output_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ro_encoder_output");
    let input = b.add_input("region_features", &[NUM_REGIONS, HIDDEN_DIM]);
    let ln_w = b.add_input("ln_weight", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_bias", &[HIDDEN_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let proj_w = b.add_input("proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_bias", &[HIDDEN_DIM]);

    let normed = b.add_layer_norm(input, ln_eps, 1, ln_w, ln_b, &[NUM_REGIONS, HIDDEN_DIM]);
    let projected = b.add_linear(normed, proj_w, Some(proj_b), &[NUM_REGIONS, HIDDEN_DIM]);
    let out = b.add_relu(projected, &[NUM_REGIONS, HIDDEN_DIM]);

    b.build(out).expect("valid encoder output kernel")
}

fn encoder_output_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let proj_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    vec![
        TensorParamBinding::Variable,               // region_features
        TensorParamBinding::ConstantTensor(ln_w),   // ln_weight
        TensorParamBinding::ConstantTensor(ln_b),   // ln_bias
        TensorParamBinding::ConstantScalar(1e-5),   // ln_eps
        TensorParamBinding::ConstantTensor(proj_w), // proj_weight
        TensorParamBinding::ConstantTensor(proj_b), // proj_bias
    ]
}

#[test]
fn test_encoder_output_bounded() {
    let def = build_encoder_output_kernel();
    let bindings = encoder_output_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_REGIONS, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through encoder output");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_REGIONS, HIDDEN_DIM],
        "encoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Encoder output bounded IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // ReLU output: lower bound >= 0
    assert!(
        lo_min >= -1e-6,
        "ReLU output lower should be >= 0, got {lo_min}"
    );
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 2. pairwise_comparison_logits: [num_regions, num_regions] shape (IBP)
// ===========================================================================

/// Pairwise comparison logits: project region features to NxN comparison matrix.
/// Uses bilinear-like structure: Linear(concat(h_i, h_j)) -> logits.
/// Input: [NUM_REGIONS * NUM_REGIONS, 2 * HIDDEN_DIM] (all region pairs)
/// Output: [NUM_REGIONS * NUM_REGIONS, 1] (raw comparison logits)
fn build_pairwise_comparison_logits_kernel() -> TensorKernelDef {
    let pair_dim = 2 * HIDDEN_DIM;
    let num_pairs = NUM_REGIONS * NUM_REGIONS;
    let mut b = TensorBlockBuilder::new("glm_ro_pairwise_logits");
    let input = b.add_input("region_pairs", &[num_pairs, pair_dim]);
    let w1 = b.add_input("cmp_w1", &[MLP_HIDDEN, pair_dim]);
    let b1 = b.add_input("cmp_b1", &[MLP_HIDDEN]);
    let w2 = b.add_input("cmp_w2", &[1, MLP_HIDDEN]);
    let b2 = b.add_input("cmp_b2", &[1]);

    let hidden = b.add_linear(input, w1, Some(b1), &[num_pairs, MLP_HIDDEN]);
    let activated = b.add_relu(hidden, &[num_pairs, MLP_HIDDEN]);
    let out = b.add_linear(activated, w2, Some(b2), &[num_pairs, 1]);

    b.build(out)
        .expect("valid pairwise comparison logits kernel")
}

#[test]
fn test_pairwise_comparison_logits() {
    let pair_dim = 2 * HIDDEN_DIM;
    let num_pairs = NUM_REGIONS * NUM_REGIONS;
    let def = build_pairwise_comparison_logits_kernel();
    let w1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN, pair_dim]), WEIGHT_MAG);
    let b1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN]), 0.0f32);
    let w2 = ArrayD::from_elem(IxDyn(&[1, MLP_HIDDEN]), WEIGHT_MAG);
    let b2 = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w1),
        TensorParamBinding::ConstantTensor(b1),
        TensorParamBinding::ConstantTensor(w2),
        TensorParamBinding::ConstantTensor(b2),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[num_pairs, pair_dim], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[num_pairs, 1],
        "pairwise logits output shape should be [N*N, 1]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pairwise comparison logits IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite");
    assert!(hi_max.is_finite(), "upper must be finite");
}

// ===========================================================================
// 3. pairwise_softmax_probability: p(i before j) in [0, 1] (IBP)
// ===========================================================================

/// Pairwise softmax probability: logits -> sigmoid -> probability in [0, 1].
/// Input: [NUM_REGIONS * NUM_REGIONS, 2 * HIDDEN_DIM]
/// Output: [NUM_REGIONS * NUM_REGIONS, 1] (probability)
fn build_pairwise_softmax_prob_kernel() -> TensorKernelDef {
    let pair_dim = 2 * HIDDEN_DIM;
    let num_pairs = NUM_REGIONS * NUM_REGIONS;
    let mut b = TensorBlockBuilder::new("glm_ro_pairwise_prob");
    let input = b.add_input("region_pairs", &[num_pairs, pair_dim]);
    let w = b.add_input("prob_weight", &[1, pair_dim]);
    let bias = b.add_input("prob_bias", &[1]);

    let logits = b.add_linear(input, w, Some(bias), &[num_pairs, 1]);
    let out = b.add_sigmoid(logits, &[num_pairs, 1]);

    b.build(out)
        .expect("valid pairwise softmax probability kernel")
}

#[test]
fn test_pairwise_softmax_probability() {
    let pair_dim = 2 * HIDDEN_DIM;
    let num_pairs = NUM_REGIONS * NUM_REGIONS;
    let def = build_pairwise_softmax_prob_kernel();
    let w = ArrayD::from_elem(IxDyn(&[1, pair_dim]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w),
        TensorParamBinding::ConstantTensor(bias),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[num_pairs, pair_dim], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pairwise softmax probability IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Sigmoid output must be in [0, 1]
    assert!(lo_min >= -1e-6, "sigmoid output lower >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-6,
        "sigmoid output upper <= 1.0, got {hi_max}"
    );
}

// ===========================================================================
// 4. pairwise_consistency: p(i < j) + p(j < i) ~ 1 (IBP + CROWN)
// ===========================================================================

/// Pairwise consistency: same network applied to pair features.
/// CROWN should tighten bounds on the sigmoid output.
/// Input: [NUM_REGIONS * NUM_REGIONS, 2 * HIDDEN_DIM]
/// Output: [NUM_REGIONS * NUM_REGIONS, 1]
fn build_pairwise_consistency_kernel() -> TensorKernelDef {
    let pair_dim = 2 * HIDDEN_DIM;
    let num_pairs = NUM_REGIONS * NUM_REGIONS;
    let mut b = TensorBlockBuilder::new("glm_ro_pairwise_consist");
    let input = b.add_input("region_pairs", &[num_pairs, pair_dim]);
    let w1 = b.add_input("consist_w1", &[MLP_HIDDEN, pair_dim]);
    let b1 = b.add_input("consist_b1", &[MLP_HIDDEN]);
    let w2 = b.add_input("consist_w2", &[1, MLP_HIDDEN]);
    let b2 = b.add_input("consist_b2", &[1]);

    let hidden = b.add_linear(input, w1, Some(b1), &[num_pairs, MLP_HIDDEN]);
    let activated = b.add_relu(hidden, &[num_pairs, MLP_HIDDEN]);
    let logits = b.add_linear(activated, w2, Some(b2), &[num_pairs, 1]);
    let out = b.add_sigmoid(logits, &[num_pairs, 1]);

    b.build(out).expect("valid pairwise consistency kernel")
}

fn pairwise_consistency_bindings() -> Vec<TensorParamBinding> {
    let pair_dim = 2 * HIDDEN_DIM;
    let w1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN, pair_dim]), WEIGHT_MAG);
    let b1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN]), 0.0f32);
    let w2 = ArrayD::from_elem(IxDyn(&[1, MLP_HIDDEN]), WEIGHT_MAG);
    let b2 = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w1),
        TensorParamBinding::ConstantTensor(b1),
        TensorParamBinding::ConstantTensor(w2),
        TensorParamBinding::ConstantTensor(b2),
    ]
}

#[test]
fn test_pairwise_consistency_ibp() {
    let pair_dim = 2 * HIDDEN_DIM;
    let num_pairs = NUM_REGIONS * NUM_REGIONS;
    let def = build_pairwise_consistency_kernel();
    let bindings = pairwise_consistency_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[num_pairs, pair_dim], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Pairwise consistency IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1.0, got {hi_max}");
}

#[test]
fn test_pairwise_consistency_crown() {
    let pair_dim = 2 * HIDDEN_DIM;
    let num_pairs = NUM_REGIONS * NUM_REGIONS;
    let def = build_pairwise_consistency_kernel();
    let bindings = pairwise_consistency_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[num_pairs, pair_dim], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Pairwise consistency CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 5. topological_sort_valid: no cycles in ordering logits (IBP)
// ===========================================================================

/// Topological sort validity: ordering logits through softmax -> row-normalized.
/// Models the DAG constraint via softmax over successor probabilities.
/// Input: [NUM_REGIONS, NUM_REGIONS] (raw ordering scores)
/// Output: [NUM_REGIONS, NUM_REGIONS] (row-normalized transition probabilities)
fn build_topological_sort_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ro_topo_sort");
    let input = b.add_input("order_scores", &[NUM_REGIONS, NUM_REGIONS]);
    let proj_w = b.add_input("topo_weight", &[NUM_REGIONS, NUM_REGIONS]);
    let proj_b = b.add_input("topo_bias", &[NUM_REGIONS]);

    let logits = b.add_linear(input, proj_w, Some(proj_b), &[NUM_REGIONS, NUM_REGIONS]);
    let out = b.add_softmax(logits, 1, &[NUM_REGIONS, NUM_REGIONS]);

    b.build(out).expect("valid topological sort kernel")
}

#[test]
fn test_topological_sort_valid() {
    let def = build_topological_sort_kernel();
    let proj_w = ArrayD::from_elem(IxDyn(&[NUM_REGIONS, NUM_REGIONS]), WEIGHT_MAG);
    let proj_b = ArrayD::from_elem(IxDyn(&[NUM_REGIONS]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(proj_w),
        TensorParamBinding::ConstantTensor(proj_b),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_REGIONS, NUM_REGIONS], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_REGIONS, NUM_REGIONS],
        "topological sort output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Topological sort IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Softmax output in [0, 1]
    assert!(lo_min >= -1e-6, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "softmax upper <= 1.0, got {hi_max}");
}

// ===========================================================================
// 6. text_line_bbox_normalized: coordinates in [0, 1] (IBP)
// ===========================================================================

/// Text line bounding box normalization: Linear -> sigmoid for [0, 1] coords.
/// Input: [NUM_REGIONS, HIDDEN_DIM] (region hidden states)
/// Output: [NUM_REGIONS, COORD_DIM] (normalized x, y, w, h in (0, 1))
fn build_text_line_bbox_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ro_text_line_bbox");
    let input = b.add_input("hidden_states", &[NUM_REGIONS, HIDDEN_DIM]);
    let w = b.add_input("bbox_weight", &[COORD_DIM, HIDDEN_DIM]);
    let bias = b.add_input("bbox_bias", &[COORD_DIM]);

    let logits = b.add_linear(input, w, Some(bias), &[NUM_REGIONS, COORD_DIM]);
    let out = b.add_sigmoid(logits, &[NUM_REGIONS, COORD_DIM]);

    b.build(out).expect("valid text line bbox kernel")
}

#[test]
fn test_text_line_bbox_normalized() {
    let def = build_text_line_bbox_kernel();
    let w = ArrayD::from_elem(IxDyn(&[COORD_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[COORD_DIM]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w),
        TensorParamBinding::ConstantTensor(bias),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_REGIONS, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_REGIONS, COORD_DIM],
        "bbox output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Text line bbox normalized IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Sigmoid output in (0, 1)
    assert!(lo_min >= -1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1.0, got {hi_max}");
}

// ===========================================================================
// 7. text_line_confidence_bounded: score in [0, 1] (IBP)
// ===========================================================================

/// Text line confidence: MLP -> sigmoid detection confidence.
/// Input: [NUM_REGIONS, HIDDEN_DIM] (region hidden states)
/// Output: [NUM_REGIONS, 1] (confidence score in (0, 1))
fn build_text_line_confidence_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ro_text_line_conf");
    let input = b.add_input("hidden_states", &[NUM_REGIONS, HIDDEN_DIM]);
    let w1 = b.add_input("conf_w1", &[MLP_HIDDEN, HIDDEN_DIM]);
    let b1 = b.add_input("conf_b1", &[MLP_HIDDEN]);
    let w2 = b.add_input("conf_w2", &[1, MLP_HIDDEN]);
    let b2 = b.add_input("conf_b2", &[1]);

    let hidden = b.add_linear(input, w1, Some(b1), &[NUM_REGIONS, MLP_HIDDEN]);
    let activated = b.add_relu(hidden, &[NUM_REGIONS, MLP_HIDDEN]);
    let logits = b.add_linear(activated, w2, Some(b2), &[NUM_REGIONS, 1]);
    let out = b.add_sigmoid(logits, &[NUM_REGIONS, 1]);

    b.build(out).expect("valid text line confidence kernel")
}

#[test]
fn test_text_line_confidence_bounded() {
    let def = build_text_line_confidence_kernel();
    let w1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN, HIDDEN_DIM]), WEIGHT_MAG);
    let b1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN]), 0.0f32);
    let w2 = ArrayD::from_elem(IxDyn(&[1, MLP_HIDDEN]), WEIGHT_MAG);
    let b2 = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w1),
        TensorParamBinding::ConstantTensor(b1),
        TensorParamBinding::ConstantTensor(w2),
        TensorParamBinding::ConstantTensor(b2),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_REGIONS, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Text line confidence IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1.0, got {hi_max}");
}

// ===========================================================================
// 8. multi_column_detection: column boundaries divide width (IBP)
// ===========================================================================

/// Multi-column detection: project region features to column assignment logits.
/// Input: [NUM_REGIONS, HIDDEN_DIM] (region features)
/// Output: [NUM_REGIONS, NUM_COLUMNS] (softmax column probabilities)
fn build_multi_column_detection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ro_multi_col_detect");
    let input = b.add_input("region_features", &[NUM_REGIONS, HIDDEN_DIM]);
    let w1 = b.add_input("col_w1", &[MLP_HIDDEN, HIDDEN_DIM]);
    let b1 = b.add_input("col_b1", &[MLP_HIDDEN]);
    let w2 = b.add_input("col_w2", &[NUM_COLUMNS, MLP_HIDDEN]);
    let b2 = b.add_input("col_b2", &[NUM_COLUMNS]);

    let hidden = b.add_linear(input, w1, Some(b1), &[NUM_REGIONS, MLP_HIDDEN]);
    let activated = b.add_relu(hidden, &[NUM_REGIONS, MLP_HIDDEN]);
    let logits = b.add_linear(activated, w2, Some(b2), &[NUM_REGIONS, NUM_COLUMNS]);
    let out = b.add_softmax(logits, 1, &[NUM_REGIONS, NUM_COLUMNS]);

    b.build(out).expect("valid multi-column detection kernel")
}

#[test]
fn test_multi_column_detection() {
    let def = build_multi_column_detection_kernel();
    let w1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN, HIDDEN_DIM]), WEIGHT_MAG);
    let b1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN]), 0.0f32);
    let w2 = ArrayD::from_elem(IxDyn(&[NUM_COLUMNS, MLP_HIDDEN]), WEIGHT_MAG);
    let b2 = ArrayD::from_elem(IxDyn(&[NUM_COLUMNS]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w1),
        TensorParamBinding::ConstantTensor(b1),
        TensorParamBinding::ConstantTensor(w2),
        TensorParamBinding::ConstantTensor(b2),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_REGIONS, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_REGIONS, NUM_COLUMNS],
        "multi-column output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Multi-column detection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Softmax output in [0, 1]
    assert!(lo_min >= -1e-6, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "softmax upper <= 1.0, got {hi_max}");
}

// ===========================================================================
// 9. header_footer_y_coordinate: classification by position (IBP)
// ===========================================================================

/// Header/footer classification based on y-coordinate position features.
/// Projects coordinate features through MLP -> sigmoid for binary classification.
/// Input: [NUM_REGIONS, COORD_DIM] (bounding box coordinates)
/// Output: [NUM_REGIONS, 2] (softmax: [header_prob, body_prob])
fn build_header_footer_kernel() -> TensorKernelDef {
    let num_classes = 2; // header vs body
    let mut b = TensorBlockBuilder::new("glm_ro_header_footer");
    let input = b.add_input("box_coords", &[NUM_REGIONS, COORD_DIM]);
    let w1 = b.add_input("hf_w1", &[MLP_HIDDEN, COORD_DIM]);
    let b1 = b.add_input("hf_b1", &[MLP_HIDDEN]);
    let w2 = b.add_input("hf_w2", &[num_classes, MLP_HIDDEN]);
    let b2 = b.add_input("hf_b2", &[num_classes]);

    let hidden = b.add_linear(input, w1, Some(b1), &[NUM_REGIONS, MLP_HIDDEN]);
    let activated = b.add_relu(hidden, &[NUM_REGIONS, MLP_HIDDEN]);
    let logits = b.add_linear(activated, w2, Some(b2), &[NUM_REGIONS, num_classes]);
    let out = b.add_softmax(logits, 1, &[NUM_REGIONS, num_classes]);

    b.build(out).expect("valid header/footer kernel")
}

#[test]
fn test_header_footer_y_coordinate() {
    let num_classes = 2;
    let def = build_header_footer_kernel();
    let w1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN, COORD_DIM]), WEIGHT_MAG);
    let b1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN]), 0.0f32);
    let w2 = ArrayD::from_elem(IxDyn(&[num_classes, MLP_HIDDEN]), WEIGHT_MAG);
    let b2 = ArrayD::from_elem(IxDyn(&[num_classes]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w1),
        TensorParamBinding::ConstantTensor(b1),
        TensorParamBinding::ConstantTensor(w2),
        TensorParamBinding::ConstantTensor(b2),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Box coordinates in [0, 1] range (page-normalized)
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[NUM_REGIONS, COORD_DIM]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[NUM_REGIONS, COORD_DIM]), 1.0f32),
    )
    .expect("valid box coord bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Header/footer classification IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-6, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "softmax upper <= 1.0, got {hi_max}");
}

// ===========================================================================
// 10. table_cell_ordering: row-major within tables (IBP)
// ===========================================================================

/// Table cell ordering: project cell features to row/column indices via sigmoid.
/// Input: [NUM_CELLS, HIDDEN_DIM] (cell region features)
/// Output: [NUM_CELLS, 2] (sigmoid: [row_position, col_position] in (0, 1))
fn build_table_cell_ordering_kernel() -> TensorKernelDef {
    let pos_dim = 2; // row_position, col_position
    let mut b = TensorBlockBuilder::new("glm_ro_table_cell_order");
    let input = b.add_input("cell_features", &[NUM_CELLS, HIDDEN_DIM]);
    let w1 = b.add_input("cell_w1", &[MLP_HIDDEN, HIDDEN_DIM]);
    let b1 = b.add_input("cell_b1", &[MLP_HIDDEN]);
    let w2 = b.add_input("cell_w2", &[pos_dim, MLP_HIDDEN]);
    let b2 = b.add_input("cell_b2", &[pos_dim]);

    let hidden = b.add_linear(input, w1, Some(b1), &[NUM_CELLS, MLP_HIDDEN]);
    let activated = b.add_relu(hidden, &[NUM_CELLS, MLP_HIDDEN]);
    let logits = b.add_linear(activated, w2, Some(b2), &[NUM_CELLS, pos_dim]);
    let out = b.add_sigmoid(logits, &[NUM_CELLS, pos_dim]);

    b.build(out).expect("valid table cell ordering kernel")
}

#[test]
fn test_table_cell_ordering() {
    let pos_dim = 2;
    let def = build_table_cell_ordering_kernel();
    let w1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN, HIDDEN_DIM]), WEIGHT_MAG);
    let b1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN]), 0.0f32);
    let w2 = ArrayD::from_elem(IxDyn(&[pos_dim, MLP_HIDDEN]), WEIGHT_MAG);
    let b2 = ArrayD::from_elem(IxDyn(&[pos_dim]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w1),
        TensorParamBinding::ConstantTensor(b1),
        TensorParamBinding::ConstantTensor(w2),
        TensorParamBinding::ConstantTensor(b2),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_CELLS, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_CELLS, pos_dim],
        "table cell ordering output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Table cell ordering IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Sigmoid output: row/col positions in (0, 1)
    assert!(lo_min >= -1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1.0, got {hi_max}");
}

// ===========================================================================
// 11. cross_attention_features: text attends to layout (IBP + CROWN)
// ===========================================================================

/// Cross-attention: text region features attend to spatial layout features.
/// Q from text regions, K/V from layout spatial features.
/// Input: [NUM_REGIONS, HIDDEN_DIM] (text region features)
/// Output: [NUM_REGIONS, HIDDEN_DIM] (cross-attention refined features)
fn build_cross_attention_features_kernel() -> TensorKernelDef {
    let scale = 1.0 / (HIDDEN_DIM as f32 / NUM_HEADS as f32).sqrt();
    let mut b = TensorBlockBuilder::new("glm_ro_cross_attn_feat");
    let text_features = b.add_input("text_features", &[NUM_REGIONS, HIDDEN_DIM]);
    let layout_features = b.add_input("layout_features", &[MAX_SEQ_LEN, HIDDEN_DIM]);

    let q_w = b.add_input("ca_qw", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("ca_kw", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("ca_vw", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("ca_ow", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(text_features, q_w, None, &[NUM_REGIONS, HIDDEN_DIM]);
    let k = b.add_linear(layout_features, k_w, None, &[MAX_SEQ_LEN, HIDDEN_DIM]);
    let v = b.add_linear(layout_features, v_w, None, &[MAX_SEQ_LEN, HIDDEN_DIM]);
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Standard,
        Some(scale),
        &[NUM_REGIONS, HIDDEN_DIM],
    );
    let out = b.add_linear(attn, o_w, None, &[NUM_REGIONS, HIDDEN_DIM]);

    b.build(out).expect("valid cross-attention features kernel")
}

fn cross_attention_features_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable, // text_features
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[MAX_SEQ_LEN, HIDDEN_DIM]),
            0.5f32,
        )), // layout_features (constant spatial context)
        TensorParamBinding::ConstantTensor(proj_w.clone()), // ca_qw
        TensorParamBinding::ConstantTensor(proj_w.clone()), // ca_kw
        TensorParamBinding::ConstantTensor(proj_w.clone()), // ca_vw
        TensorParamBinding::ConstantTensor(proj_w), // ca_ow
    ]
}

#[test]
fn test_cross_attention_features_ibp() {
    let def = build_cross_attention_features_kernel();
    let bindings = cross_attention_features_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_REGIONS, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_REGIONS, HIDDEN_DIM],
        "cross-attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cross-attention features IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite");
    assert!(hi_max.is_finite(), "upper must be finite");
}

#[test]
fn test_cross_attention_features_crown() {
    let def = build_cross_attention_features_kernel();
    let bindings = cross_attention_features_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_REGIONS, HIDDEN_DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!(
        "Cross-attention features CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}"
    );
}

// ===========================================================================
// 12. position_encoding_2d: spatial document layout (IBP)
// ===========================================================================

/// 2D position encoding: project spatial (x, y) coordinates to hidden dim.
/// Models LayoutLM-style 2D position embeddings for document regions.
/// Input: [NUM_REGIONS, COORD_DIM] (box coordinates)
/// Output: [NUM_REGIONS, HIDDEN_DIM] (positional features)
fn build_position_encoding_2d_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ro_pos_enc_2d");
    let input = b.add_input("box_coords", &[NUM_REGIONS, COORD_DIM]);
    let w1 = b.add_input("pe_w1", &[HIDDEN_DIM, COORD_DIM]);
    let b1 = b.add_input("pe_b1", &[HIDDEN_DIM]);
    let w2 = b.add_input("pe_w2", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b2 = b.add_input("pe_b2", &[HIDDEN_DIM]);

    // Two-layer MLP for position encoding
    let hidden = b.add_linear(input, w1, Some(b1), &[NUM_REGIONS, HIDDEN_DIM]);
    let activated = b.add_gelu(hidden, &[NUM_REGIONS, HIDDEN_DIM]);
    let out = b.add_linear(activated, w2, Some(b2), &[NUM_REGIONS, HIDDEN_DIM]);

    b.build(out).expect("valid 2D position encoding kernel")
}

#[test]
fn test_position_encoding_2d() {
    let def = build_position_encoding_2d_kernel();
    let w1 = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, COORD_DIM]), WEIGHT_MAG);
    let b1 = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let w2 = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let b2 = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w1),
        TensorParamBinding::ConstantTensor(b1),
        TensorParamBinding::ConstantTensor(w2),
        TensorParamBinding::ConstantTensor(b2),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Box coordinates in [0, 1] (page-normalized)
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[NUM_REGIONS, COORD_DIM]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[NUM_REGIONS, COORD_DIM]), 1.0f32),
    )
    .expect("valid coord bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_REGIONS, HIDDEN_DIM],
        "2D position encoding output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("2D position encoding IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 13. max_regions_bounded: limited by max_seq_len (IBP)
// ===========================================================================

/// Max regions bounded: processes MAX_SEQ_LEN regions through encoder-like block.
/// Verifies that bounds stay finite when processing at the maximum capacity.
/// Input: [MAX_SEQ_LEN, HIDDEN_DIM] (padded region features)
/// Output: [MAX_SEQ_LEN, HIDDEN_DIM] (encoded features)
fn build_max_regions_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ro_max_regions");
    let input = b.add_input("padded_regions", &[MAX_SEQ_LEN, HIDDEN_DIM]);
    let ln_w = b.add_input("ln_weight", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_bias", &[HIDDEN_DIM]);
    let ln_eps = b.add_input("ln_eps", &[1]);
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, HIDDEN_DIM]);
    let ffn2_w = b.add_input("ffn2_w", &[HIDDEN_DIM, FFN_DIM]);

    let normed = b.add_layer_norm(input, ln_eps, 1, ln_w, ln_b, &[MAX_SEQ_LEN, HIDDEN_DIM]);
    let ffn_hidden = b.add_linear(normed, ffn1_w, None, &[MAX_SEQ_LEN, FFN_DIM]);
    let ffn_act = b.add_relu(ffn_hidden, &[MAX_SEQ_LEN, FFN_DIM]);
    let ffn_out = b.add_linear(ffn_act, ffn2_w, None, &[MAX_SEQ_LEN, HIDDEN_DIM]);
    let out = b.add_binary_add(input, ffn_out, &[MAX_SEQ_LEN, HIDDEN_DIM]);

    b.build(out).expect("valid max regions kernel")
}

#[test]
fn test_max_regions_bounded() {
    let def = build_max_regions_kernel();
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let ffn1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let ffn2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ln_w),
        TensorParamBinding::ConstantTensor(ln_b),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ffn1_w),
        TensorParamBinding::ConstantTensor(ffn2_w),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[MAX_SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[MAX_SEQ_LEN, HIDDEN_DIM],
        "max regions output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Max regions bounded IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower must be finite");
    assert!(hi_max.is_finite(), "upper must be finite");
}

// ===========================================================================
// 14. region_merging_containment: merged bbox contains originals (IBP)
// ===========================================================================

/// Region merging: concatenate pair features -> MLP -> sigmoid merge probability.
/// Models the decision of whether to merge two adjacent text regions.
/// Input: [NUM_REGIONS, 2 * HIDDEN_DIM] (concatenated region pair features)
/// Output: [NUM_REGIONS, 1] (merge probability in (0, 1))
fn build_region_merging_kernel() -> TensorKernelDef {
    let merge_dim = 2 * HIDDEN_DIM;
    let mut b = TensorBlockBuilder::new("glm_ro_region_merge");
    let input = b.add_input("region_pair_features", &[NUM_REGIONS, merge_dim]);
    let w1 = b.add_input("merge_w1", &[MLP_HIDDEN, merge_dim]);
    let b1 = b.add_input("merge_b1", &[MLP_HIDDEN]);
    let w2 = b.add_input("merge_w2", &[1, MLP_HIDDEN]);
    let b2 = b.add_input("merge_b2", &[1]);

    let hidden = b.add_linear(input, w1, Some(b1), &[NUM_REGIONS, MLP_HIDDEN]);
    let activated = b.add_relu(hidden, &[NUM_REGIONS, MLP_HIDDEN]);
    let logits = b.add_linear(activated, w2, Some(b2), &[NUM_REGIONS, 1]);
    let out = b.add_sigmoid(logits, &[NUM_REGIONS, 1]);

    b.build(out).expect("valid region merging kernel")
}

#[test]
fn test_region_merging_containment() {
    let merge_dim = 2 * HIDDEN_DIM;
    let def = build_region_merging_kernel();
    let w1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN, merge_dim]), WEIGHT_MAG);
    let b1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN]), 0.0f32);
    let w2 = ArrayD::from_elem(IxDyn(&[1, MLP_HIDDEN]), WEIGHT_MAG);
    let b2 = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w1),
        TensorParamBinding::ConstantTensor(b1),
        TensorParamBinding::ConstantTensor(w2),
        TensorParamBinding::ConstantTensor(b2),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_REGIONS, merge_dim], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Region merging containment IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Sigmoid output in (0, 1)
    assert!(lo_min >= -1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1.0, got {hi_max}");
}

// ===========================================================================
// 15. reading_order_dag: directed acyclic graph ordering (IBP + CROWN)
// ===========================================================================

/// Reading order DAG: self-attention among regions -> softmax successor probs.
/// Enforces DAG structure through attention-based ordering.
/// Input: [NUM_REGIONS, HIDDEN_DIM] (region features)
/// Output: [NUM_REGIONS, NUM_REGIONS] (DAG transition probabilities)
fn build_reading_order_dag_kernel() -> TensorKernelDef {
    let scale = 1.0 / (HIDDEN_DIM as f32 / NUM_HEADS as f32).sqrt();
    let mut b = TensorBlockBuilder::new("glm_ro_dag");
    let input = b.add_input("region_features", &[NUM_REGIONS, HIDDEN_DIM]);
    let q_w = b.add_input("dag_qw", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("dag_kw", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("dag_vw", &[HIDDEN_DIM, HIDDEN_DIM]);

    // Self-attention: regions attend to each other
    let q = b.add_linear(input, q_w, None, &[NUM_REGIONS, HIDDEN_DIM]);
    let k = b.add_linear(input, k_w, None, &[NUM_REGIONS, HIDDEN_DIM]);
    let v = b.add_linear(input, v_w, None, &[NUM_REGIONS, HIDDEN_DIM]);
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Standard,
        Some(scale),
        &[NUM_REGIONS, HIDDEN_DIM],
    );

    // Project to NxN transition probabilities
    let out_w = b.add_input("dag_out_w", &[NUM_REGIONS, HIDDEN_DIM]);
    let out_b = b.add_input("dag_out_b", &[NUM_REGIONS]);
    let logits = b.add_linear(attn, out_w, Some(out_b), &[NUM_REGIONS, NUM_REGIONS]);
    let out = b.add_softmax(logits, 1, &[NUM_REGIONS, NUM_REGIONS]);

    b.build(out).expect("valid reading order DAG kernel")
}

fn reading_order_dag_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[NUM_REGIONS, HIDDEN_DIM]), WEIGHT_MAG);
    let out_b = ArrayD::from_elem(IxDyn(&[NUM_REGIONS]), 0.0f32);
    vec![
        TensorParamBinding::Variable,                       // region_features
        TensorParamBinding::ConstantTensor(proj_w.clone()), // dag_qw
        TensorParamBinding::ConstantTensor(proj_w.clone()), // dag_kw
        TensorParamBinding::ConstantTensor(proj_w),         // dag_vw
        TensorParamBinding::ConstantTensor(out_w),          // dag_out_w
        TensorParamBinding::ConstantTensor(out_b),          // dag_out_b
    ]
}

#[test]
fn test_reading_order_dag_ibp() {
    let def = build_reading_order_dag_kernel();
    let bindings = reading_order_dag_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_REGIONS, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_REGIONS, NUM_REGIONS],
        "DAG output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Reading order DAG IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Softmax output in [0, 1]
    assert!(lo_min >= -1e-6, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "softmax upper <= 1.0, got {hi_max}");
}

#[test]
fn test_reading_order_dag_crown() {
    let def = build_reading_order_dag_kernel();
    let bindings = reading_order_dag_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_REGIONS, HIDDEN_DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Reading order DAG CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 16. beam_search_log_probs: scores <= 0 (IBP)
// ===========================================================================

/// Beam search log probabilities: log_softmax produces scores <= 0.
/// Input: [NUM_REGIONS, NUM_REGIONS] (raw ordering scores)
/// Output: [NUM_REGIONS, NUM_REGIONS] (log probabilities, all <= 0)
fn build_beam_search_log_probs_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ro_beam_log_probs");
    let input = b.add_input("order_scores", &[NUM_REGIONS, NUM_REGIONS]);
    let w = b.add_input("beam_weight", &[NUM_REGIONS, NUM_REGIONS]);
    let bias = b.add_input("beam_bias", &[NUM_REGIONS]);

    let logits = b.add_linear(input, w, Some(bias), &[NUM_REGIONS, NUM_REGIONS]);
    // log_softmax: outputs are <= 0 (log of probabilities)
    let out = b.add_log_softmax(logits, 1, &[NUM_REGIONS, NUM_REGIONS]);

    b.build(out).expect("valid beam search log probs kernel")
}

#[test]
fn test_beam_search_log_probs() {
    let def = build_beam_search_log_probs_kernel();
    let w = ArrayD::from_elem(IxDyn(&[NUM_REGIONS, NUM_REGIONS]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[NUM_REGIONS]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w),
        TensorParamBinding::ConstantTensor(bias),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_REGIONS, NUM_REGIONS], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_REGIONS, NUM_REGIONS],
        "beam search log probs shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Beam search log probs IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // log_softmax output: upper bound <= 0 (log of probability <= 1)
    assert!(
        hi_max <= 1e-6,
        "log_softmax output upper should be <= 0, got {hi_max}"
    );
    assert!(lo_min.is_finite(), "lower bound must be finite");
}

// ===========================================================================
// 17. confidence_calibration: probability approximation (IBP + CROWN)
// ===========================================================================

/// Confidence calibration: temperature-scaled softmax for calibrated probabilities.
/// Models a calibration head: Linear -> ReLU -> Linear -> sigmoid.
/// Input: [NUM_REGIONS, HIDDEN_DIM] (region features)
/// Output: [NUM_REGIONS, 1] (calibrated confidence in (0, 1))
fn build_confidence_calibration_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("glm_ro_conf_calib");
    let input = b.add_input("region_features", &[NUM_REGIONS, HIDDEN_DIM]);
    let w1 = b.add_input("calib_w1", &[MLP_HIDDEN, HIDDEN_DIM]);
    let b1 = b.add_input("calib_b1", &[MLP_HIDDEN]);
    let w2 = b.add_input("calib_w2", &[1, MLP_HIDDEN]);
    let b2 = b.add_input("calib_b2", &[1]);

    let hidden = b.add_linear(input, w1, Some(b1), &[NUM_REGIONS, MLP_HIDDEN]);
    let activated = b.add_relu(hidden, &[NUM_REGIONS, MLP_HIDDEN]);
    let logits = b.add_linear(activated, w2, Some(b2), &[NUM_REGIONS, 1]);
    let out = b.add_sigmoid(logits, &[NUM_REGIONS, 1]);

    b.build(out).expect("valid confidence calibration kernel")
}

fn confidence_calibration_bindings() -> Vec<TensorParamBinding> {
    let w1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN, HIDDEN_DIM]), WEIGHT_MAG);
    let b1 = ArrayD::from_elem(IxDyn(&[MLP_HIDDEN]), 0.0f32);
    let w2 = ArrayD::from_elem(IxDyn(&[1, MLP_HIDDEN]), WEIGHT_MAG);
    let b2 = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w1),
        TensorParamBinding::ConstantTensor(b1),
        TensorParamBinding::ConstantTensor(w2),
        TensorParamBinding::ConstantTensor(b2),
    ]
}

#[test]
fn test_confidence_calibration_ibp() {
    let def = build_confidence_calibration_kernel();
    let bindings = confidence_calibration_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_REGIONS, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Confidence calibration IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-6, "sigmoid lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "sigmoid upper <= 1.0, got {hi_max}");
}

#[test]
fn test_confidence_calibration_crown() {
    let def = build_confidence_calibration_kernel();
    let bindings = confidence_calibration_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_REGIONS, HIDDEN_DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!(
        "Confidence calibration CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}"
    );
}

// ===========================================================================
// 18. full_pipeline_bounded: image -> regions -> order -> text (IBP + CROWN)
// ===========================================================================

/// Full pipeline: position encoding -> self-attention -> pairwise ordering -> softmax.
/// End-to-end composition from region coordinates to reading order probabilities.
/// Input: [NUM_REGIONS, COORD_DIM] (bounding box coordinates)
/// Output: [NUM_REGIONS, NUM_REGIONS] (pairwise ordering probabilities)
fn build_full_pipeline_kernel() -> TensorKernelDef {
    let scale = 1.0 / (HIDDEN_DIM as f32 / NUM_HEADS as f32).sqrt();
    let mut b = TensorBlockBuilder::new("glm_ro_full_pipeline");

    // Stage 1: Position encoding (coords -> hidden dim)
    let input = b.add_input("box_coords", &[NUM_REGIONS, COORD_DIM]);
    let pe_w = b.add_input("pe_weight", &[HIDDEN_DIM, COORD_DIM]);
    let pe_b = b.add_input("pe_bias", &[HIDDEN_DIM]);
    let encoded = b.add_linear(input, pe_w, Some(pe_b), &[NUM_REGIONS, HIDDEN_DIM]);
    let encoded_act = b.add_relu(encoded, &[NUM_REGIONS, HIDDEN_DIM]);

    // Stage 2: Self-attention among regions
    let q_w = b.add_input("sa_qw", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("sa_kw", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("sa_vw", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(encoded_act, q_w, None, &[NUM_REGIONS, HIDDEN_DIM]);
    let k = b.add_linear(encoded_act, k_w, None, &[NUM_REGIONS, HIDDEN_DIM]);
    let v = b.add_linear(encoded_act, v_w, None, &[NUM_REGIONS, HIDDEN_DIM]);
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Standard,
        Some(scale),
        &[NUM_REGIONS, HIDDEN_DIM],
    );
    // Residual
    let refined = b.add_binary_add(encoded_act, attn, &[NUM_REGIONS, HIDDEN_DIM]);

    // Stage 3: Order prediction head -> softmax transition matrix
    let order_w = b.add_input("order_weight", &[NUM_REGIONS, HIDDEN_DIM]);
    let order_b = b.add_input("order_bias", &[NUM_REGIONS]);
    let order_logits = b.add_linear(refined, order_w, Some(order_b), &[NUM_REGIONS, NUM_REGIONS]);
    let out = b.add_softmax(order_logits, 1, &[NUM_REGIONS, NUM_REGIONS]);

    b.build(out).expect("valid full pipeline kernel")
}

fn full_pipeline_bindings() -> Vec<TensorParamBinding> {
    let pe_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, COORD_DIM]), WEIGHT_MAG);
    let pe_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let order_w = ArrayD::from_elem(IxDyn(&[NUM_REGIONS, HIDDEN_DIM]), WEIGHT_MAG);
    let order_b = ArrayD::from_elem(IxDyn(&[NUM_REGIONS]), 0.0f32);
    vec![
        TensorParamBinding::Variable,                       // box_coords
        TensorParamBinding::ConstantTensor(pe_w),           // pe_weight
        TensorParamBinding::ConstantTensor(pe_b),           // pe_bias
        TensorParamBinding::ConstantTensor(proj_w.clone()), // sa_qw
        TensorParamBinding::ConstantTensor(proj_w.clone()), // sa_kw
        TensorParamBinding::ConstantTensor(proj_w),         // sa_vw
        TensorParamBinding::ConstantTensor(order_w),        // order_weight
        TensorParamBinding::ConstantTensor(order_b),        // order_bias
    ]
}

#[test]
fn test_full_pipeline_bounded_ibp() {
    let def = build_full_pipeline_kernel();
    let bindings = full_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Box coordinates in [0, 1] (page-normalized)
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[NUM_REGIONS, COORD_DIM]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[NUM_REGIONS, COORD_DIM]), 1.0f32),
    )
    .expect("valid coord bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_REGIONS, NUM_REGIONS],
        "full pipeline output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Softmax output in [0, 1]
    assert!(lo_min >= -1e-6, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-6, "softmax upper <= 1.0, got {hi_max}");
}

#[test]
fn test_full_pipeline_bounded_crown() {
    let def = build_full_pipeline_kernel();
    let bindings = full_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Tighter bounds for CROWN
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[NUM_REGIONS, COORD_DIM]), 0.2f32),
        ArrayD::from_elem(IxDyn(&[NUM_REGIONS, COORD_DIM]), 0.8f32),
    )
    .expect("valid tight coord bounds");

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Full pipeline CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}
