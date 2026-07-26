// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: DETR object detection head NY composition.
//!
//! Verifies bounds propagation through the DETR detection head that converts
//! decoder output (object queries) into class logits and bounding box
//! predictions.
//!
//! Architecture (Carion et al. 2020):
//!   - **Class head:** Linear(D, num_classes + 1) -- single linear projection
//!     from decoder output to class logits including "no object" class.
//!   - **Bbox head:** 3-layer MLP: Linear(D, D) -> ReLU -> Linear(D, D)
//!     -> ReLU -> Linear(D, 4) -> Sigmoid -- predicts normalized (cx, cy, w, h).
//!   - Both heads operate independently on each object query.
//!
//! The bbox head sigmoid is critical for verification: it clamps output
//! to [0, 1], which should produce tighter output bounds than the class head.
//!
//! Part of #3556: DETR object detection compose verification tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ===========================================================================
// Dimensions
// ===========================================================================

/// Number of object queries (decoder output slots).
const NUM_QUERIES: usize = 10;
/// Embedding dimension (decoder output dim).
const EMBED_DIM: usize = 64;
/// Number of classes (COCO: 91 classes + 1 "no object" = 92).
/// Using small value for fast verification.
const NUM_CLASSES: usize = 12;
/// Bounding box output dimension: (cx, cy, w, h).
const BBOX_DIM: usize = 4;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ===========================================================================
// Builder helpers
// ===========================================================================

/// Build the DETR class prediction head: Linear(D, num_classes).
///
/// Input: `[NUM_QUERIES, EMBED_DIM]` (Variable -- decoder output).
/// Output: `[NUM_QUERIES, NUM_CLASSES]` (class logits).
///
/// Single linear projection. No softmax -- loss function handles that.
fn build_class_head_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("detr_class_head");

    let input = b.add_input("decoder_output", &[NUM_QUERIES, EMBED_DIM]);
    let weight = b.add_input("class_weight", &[NUM_CLASSES, EMBED_DIM]);
    let bias = b.add_input("class_bias", &[NUM_CLASSES]);

    let out = b.add_linear(input, weight, Some(bias), &[NUM_QUERIES, NUM_CLASSES]);

    b.build(out).expect("valid class head kernel")
}

/// Bindings for the class head.
fn class_head_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[NUM_CLASSES, EMBED_DIM]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // decoder_output [Q, D]
        TensorParamBinding::ConstantTensor(w),    // class_weight [C, D]
        TensorParamBinding::ConstantTensor(bias), // class_bias [C]
    ]
}

/// Build the DETR bbox prediction head: 3-layer MLP with sigmoid output.
///
/// Input: `[NUM_QUERIES, EMBED_DIM]` (Variable -- decoder output).
/// Output: `[NUM_QUERIES, BBOX_DIM]` (normalized bbox coordinates in [0, 1]).
///
/// Architecture: Linear(D, D) -> ReLU -> Linear(D, D) -> ReLU -> Linear(D, 4) -> Sigmoid
///
/// The final sigmoid ensures outputs are in [0, 1], representing normalized
/// bounding box coordinates (center_x, center_y, width, height).
fn build_bbox_head_kernel() -> TensorKernelDef {
    let d = EMBED_DIM;
    let mut b = TensorBlockBuilder::new("detr_bbox_head");

    let input = b.add_input("decoder_output", &[NUM_QUERIES, d]);
    let w1 = b.add_input("bbox_w1", &[d, d]);
    let w2 = b.add_input("bbox_w2", &[d, d]);
    let w3 = b.add_input("bbox_w3", &[BBOX_DIM, d]);

    let shape_d = [NUM_QUERIES, d];
    let shape_bbox = [NUM_QUERIES, BBOX_DIM];

    // Layer 1: Linear -> ReLU
    let h1 = b.add_linear(input, w1, None, &shape_d);
    let h1_act = b.add_relu(h1, &shape_d);

    // Layer 2: Linear -> ReLU
    let h2 = b.add_linear(h1_act, w2, None, &shape_d);
    let h2_act = b.add_relu(h2, &shape_d);

    // Layer 3: Linear -> Sigmoid
    let h3 = b.add_linear(h2_act, w3, None, &shape_bbox);
    let out = b.add_sigmoid(h3, &shape_bbox);

    b.build(out).expect("valid bbox head kernel")
}

/// Bindings for the bbox head.
fn bbox_head_bindings() -> Vec<TensorParamBinding> {
    let d = EMBED_DIM;
    let w1 = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let w2 = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let w3 = ArrayD::from_elem(IxDyn(&[BBOX_DIM, d]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,           // decoder_output [Q, D]
        TensorParamBinding::ConstantTensor(w1), // bbox_w1 [D, D]
        TensorParamBinding::ConstantTensor(w2), // bbox_w2 [D, D]
        TensorParamBinding::ConstantTensor(w3), // bbox_w3 [4, D]
    ]
}

/// Build the combined detection head: class logits + bbox regression.
///
/// Input: `[NUM_QUERIES, EMBED_DIM]` (Variable -- decoder output).
/// Outputs are separate, but we model them as a concatenated output for
/// single-graph verification:
///   class_logits [Q, C] ++ bbox_coords [Q, 4] -> cat -> [Q, C + 4]
///
/// This tests the full detection head as a single verification graph.
fn build_combined_detection_head_kernel() -> TensorKernelDef {
    let d = EMBED_DIM;
    let mut b = TensorBlockBuilder::new("detr_combined_head");

    let input = b.add_input("decoder_output", &[NUM_QUERIES, d]);

    // Class head weights
    let cls_w = b.add_input("class_weight", &[NUM_CLASSES, d]);
    let cls_b = b.add_input("class_bias", &[NUM_CLASSES]);

    // Bbox head weights (3-layer MLP)
    let bbox_w1 = b.add_input("bbox_w1", &[d, d]);
    let bbox_w2 = b.add_input("bbox_w2", &[d, d]);
    let bbox_w3 = b.add_input("bbox_w3", &[BBOX_DIM, d]);

    // Class head: Linear
    let cls_out = b.add_linear(input, cls_w, Some(cls_b), &[NUM_QUERIES, NUM_CLASSES]);

    // Bbox head: Linear -> ReLU -> Linear -> ReLU -> Linear -> Sigmoid
    let shape_d = [NUM_QUERIES, d];
    let shape_bbox = [NUM_QUERIES, BBOX_DIM];

    let h1 = b.add_linear(input, bbox_w1, None, &shape_d);
    let h1_act = b.add_relu(h1, &shape_d);
    let h2 = b.add_linear(h1_act, bbox_w2, None, &shape_d);
    let h2_act = b.add_relu(h2, &shape_d);
    let h3 = b.add_linear(h2_act, bbox_w3, None, &shape_bbox);
    let bbox_out = b.add_sigmoid(h3, &shape_bbox);

    // Concatenate class logits and bbox predictions along last dimension
    let out_dim = NUM_CLASSES + BBOX_DIM;
    let out = b.add_concat(
        &[cls_out, bbox_out],
        1, // concat along dim=1
        &[NUM_QUERIES, out_dim],
    );

    b.build(out).expect("valid combined detection head kernel")
}

/// Bindings for the combined detection head.
fn combined_detection_head_bindings() -> Vec<TensorParamBinding> {
    let d = EMBED_DIM;
    let cls_w = ArrayD::from_elem(IxDyn(&[NUM_CLASSES, d]), WEIGHT_MAG);
    let cls_b = ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32);
    let bbox_w1 = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let bbox_w2 = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let bbox_w3 = ArrayD::from_elem(IxDyn(&[BBOX_DIM, d]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                // decoder_output [Q, D]
        TensorParamBinding::ConstantTensor(cls_w),   // class_weight [C, D]
        TensorParamBinding::ConstantTensor(cls_b),   // class_bias [C]
        TensorParamBinding::ConstantTensor(bbox_w1), // bbox_w1 [D, D]
        TensorParamBinding::ConstantTensor(bbox_w2), // bbox_w2 [D, D]
        TensorParamBinding::ConstantTensor(bbox_w3), // bbox_w3 [4, D]
    ]
}

// ===========================================================================
// Tests: Class prediction head
// ===========================================================================

/// Class head TensorKernelDef validates.
#[test]
fn test_detr_class_head_def_validates() {
    let def = build_class_head_kernel();
    def.validate().expect("class head kernel should validate");
}

/// Class head graph builds.
#[test]
fn test_detr_class_head_graph_builds() {
    let def = build_class_head_kernel();
    let bindings = class_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("class head graph should translate");

    // Single linear layer = at least 1 node
    assert!(
        graph.num_nodes() >= 1,
        "class head graph should have >= 1 node, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through class head.
///
/// Single linear projection: output bounds scale with weight * input range.
/// With 0.02 weights, [-1, 1] input, D=64: each output ~= sum(0.02 * x_i)
/// over D values, so max output ~= 0.02 * 64 * 1 = 1.28 per element.
#[test]
fn test_detr_class_head_ibp_propagates() {
    let def = build_class_head_kernel();
    let bindings = class_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, EMBED_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through class head");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, NUM_CLASSES],
        "class head output shape must be [NUM_QUERIES, NUM_CLASSES]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR class head IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");

    // With D=64, weight=0.02, input in [-1, 1]:
    // worst case each output = sum(|w_i| * 1.0) = 64 * 0.02 = 1.28
    assert!(
        hi_max < 5.0,
        "class head upper bound should be < 5.0 with small weights, got {hi_max}"
    );
}

/// CROWN propagation through class head (pure linear -- should succeed).
#[test]
fn test_detr_class_head_crown_propagation() {
    let def = build_class_head_kernel();
    let bindings = class_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, NUM_CLASSES],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR class head: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record class head.
#[test]
fn test_detr_class_head_verify_and_record() {
    let def = build_class_head_kernel();
    let bindings = class_head_bindings();
    let input = uniform_bounds(&[NUM_QUERIES, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "detr_class_head");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_QUERIES, NUM_CLASSES]);
}

// ===========================================================================
// Tests: Bbox prediction head (3-layer MLP + sigmoid)
// ===========================================================================

/// Bbox head TensorKernelDef validates.
#[test]
fn test_detr_bbox_head_def_validates() {
    let def = build_bbox_head_kernel();
    def.validate().expect("bbox head kernel should validate");
}

/// Bbox head graph builds with sufficient depth for 3-layer MLP.
#[test]
fn test_detr_bbox_head_graph_builds() {
    let def = build_bbox_head_kernel();
    let bindings = bbox_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("bbox head graph should translate");

    // 3 Linear + 2 ReLU + 1 Sigmoid = at least 6 nodes
    assert!(
        graph.num_nodes() >= 6,
        "bbox head graph should have >= 6 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through bbox head.
///
/// The final sigmoid clamps output to [0, 1]. This is the key verification
/// property: regardless of how wide the bounds get through the MLP layers,
/// the sigmoid output must be in [0, 1].
#[test]
fn test_detr_bbox_head_ibp_propagates() {
    let def = build_bbox_head_kernel();
    let bindings = bbox_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, EMBED_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through bbox head");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, BBOX_DIM],
        "bbox head output shape must be [NUM_QUERIES, 4]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR bbox head IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid output must be in [0, 1]. IBP through sigmoid should give
    // bounds within [0, 1] (possibly wider than the true range but still
    // within the sigmoid's codomain).
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "bbox sigmoid lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "bbox sigmoid upper bound must be <= 1, got {hi_max}"
    );
}

/// CROWN propagation through bbox head.
///
/// The 3-layer MLP with ReLU should be CROWN-friendly (piecewise linear
/// activations). Sigmoid at the end requires CROWN linearization.
#[test]
fn test_detr_bbox_head_crown_propagation() {
    let def = build_bbox_head_kernel();
    let bindings = bbox_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, BBOX_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR bbox head: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    // Even under CROWN, sigmoid bounds must be in [0, 1].
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "bbox sigmoid lower bound must be >= 0 under CROWN, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "bbox sigmoid upper bound must be <= 1 under CROWN, got {hi_max}"
    );
}

/// Verify and record bbox head.
#[test]
fn test_detr_bbox_head_verify_and_record() {
    let def = build_bbox_head_kernel();
    let bindings = bbox_head_bindings();
    let input = uniform_bounds(&[NUM_QUERIES, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "detr_bbox_head");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_QUERIES, BBOX_DIM]);
}

// ===========================================================================
// Tests: Combined detection head (class + bbox)
// ===========================================================================

/// Combined detection head TensorKernelDef validates.
#[test]
fn test_detr_combined_head_def_validates() {
    let def = build_combined_detection_head_kernel();
    def.validate()
        .expect("combined detection head kernel should validate");
}

/// Combined detection head graph builds with both branches.
#[test]
fn test_detr_combined_head_graph_builds() {
    let def = build_combined_detection_head_kernel();
    let bindings = combined_detection_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("combined detection head graph should translate");

    // Class branch (1 linear) + Bbox branch (3 linear + 2 relu + sigmoid) + concat
    assert!(
        graph.num_nodes() >= 8,
        "combined detection head should have >= 8 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through combined detection head.
///
/// Output is [NUM_QUERIES, NUM_CLASSES + BBOX_DIM] where:
/// - First NUM_CLASSES dims are class logits (unbounded linear output)
/// - Last BBOX_DIM dims are bbox coordinates (sigmoid-bounded [0, 1])
#[test]
fn test_detr_combined_head_ibp_propagates() {
    let def = build_combined_detection_head_kernel();
    let bindings = combined_detection_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through combined detection head");

    let out_dim = NUM_CLASSES + BBOX_DIM;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, out_dim],
        "combined head output shape must be [NUM_QUERIES, {out_dim}]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR combined head IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN propagation through combined detection head.
#[test]
fn test_detr_combined_head_crown_propagation() {
    let def = build_combined_detection_head_kernel();
    let bindings = combined_detection_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let out_dim = NUM_CLASSES + BBOX_DIM;
    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, out_dim],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR combined head: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record combined detection head.
#[test]
fn test_detr_combined_head_verify_and_record() {
    let def = build_combined_detection_head_kernel();
    let bindings = combined_detection_head_bindings();
    let input = uniform_bounds(&[NUM_QUERIES, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "detr_combined_detection_head");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let out_dim = NUM_CLASSES + BBOX_DIM;
    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_QUERIES, out_dim]);
}
