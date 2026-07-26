// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: DETR learned positional encoding NY composition.
//!
//! Verifies bounds propagation through DETR's learned positional encoding,
//! which differs from the sinusoidal encoding used in the original Transformer.
//!
//! Architecture (Carion et al. 2020):
//!   DETR uses **learned** positional encodings for both:
//!   1. Spatial positional encoding for encoder features (row + column embeddings)
//!   2. Object query embeddings for decoder input
//!
//!   The positional encoding is added element-wise to the input:
//!     x_pos = x + pos_embed
//!
//!   where `pos_embed` is a learned parameter (constant during inference).
//!   Since pos_embed is constant, its contribution to output bounds is
//!   a fixed additive shift. This test verifies that NY correctly
//!   handles the add-constant pattern through subsequent layers.
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

/// Number of spatial positions (flattened H/P * W/P feature map).
const NUM_POSITIONS: usize = 16;
/// Embedding dimension.
const EMBED_DIM: usize = 64;
/// Number of object queries (decoder side).
const NUM_QUERIES: usize = 10;
/// Weight magnitude for linear layers.
const WEIGHT_MAG: f32 = 0.02;

// ===========================================================================
// Builder helpers
// ===========================================================================

/// Build a learned spatial positional encoding kernel for encoder features.
///
/// Models: x + pos_embed -> LayerNorm -> Linear
///
/// Input: `[NUM_POSITIONS, EMBED_DIM]` (Variable -- flattened CNN features).
/// pos_embed: `[NUM_POSITIONS, EMBED_DIM]` (Constant -- learned encoding).
/// Output: `[NUM_POSITIONS, EMBED_DIM]`.
///
/// The LayerNorm + Linear after addition exercises NY's handling
/// of constant additive shifts propagating through normalization.
fn build_spatial_positional_encoding_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("detr_spatial_pos_enc");

    let input = b.add_input("features", &[NUM_POSITIONS, EMBED_DIM]);
    let pos_embed = b.add_input("pos_embed", &[NUM_POSITIONS, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[EMBED_DIM]);
    let ln_b = b.add_input("ln_bias", &[EMBED_DIM]);
    let proj_w = b.add_input("proj_weight", &[EMBED_DIM, EMBED_DIM]);

    let shape = [NUM_POSITIONS, EMBED_DIM];

    // Add learned positional encoding
    let x_pos = b.add_binary_add(input, pos_embed, &shape);

    // LayerNorm after position addition
    let normed = b.add_layer_norm(x_pos, eps, 1, ln_w, ln_b, &shape);

    // Linear projection (e.g., first layer of attention Q/K/V)
    let out = b.add_linear(normed, proj_w, None, &shape);

    b.build(out)
        .expect("valid spatial positional encoding kernel")
}

/// Bindings for spatial positional encoding.
fn spatial_pos_encoding_bindings() -> Vec<TensorParamBinding> {
    let d = EMBED_DIM;
    // Learned positional encoding with small magnitude (typical init)
    let pos_embed = ArrayD::from_elem(IxDyn(&[NUM_POSITIONS, d]), 0.01f32);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                  // features [S, D]
        TensorParamBinding::ConstantTensor(pos_embed), // pos_embed [S, D]
        TensorParamBinding::ConstantScalar(1e-5),      // eps
        TensorParamBinding::ConstantTensor(ln_w),      // ln_weight [D]
        TensorParamBinding::ConstantTensor(ln_b),      // ln_bias [D]
        TensorParamBinding::ConstantTensor(w_proj),    // proj_weight [D, D]
    ]
}

/// Build a learned object query embedding kernel for decoder input.
///
/// Models: query_embed -> LayerNorm -> Linear
///
/// Input: `[NUM_QUERIES, EMBED_DIM]` (Variable -- learned object queries).
/// Output: `[NUM_QUERIES, EMBED_DIM]`.
///
/// In DETR, object queries are the decoder's initial input. They are learned
/// embeddings that the decoder transforms into detection predictions. This
/// tests the simple LN -> Linear pipeline that object queries pass through
/// before self-attention.
fn build_object_query_encoding_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("detr_object_query_enc");

    let input = b.add_input("object_queries", &[NUM_QUERIES, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[EMBED_DIM]);
    let ln_b = b.add_input("ln_bias", &[EMBED_DIM]);
    let proj_w = b.add_input("proj_weight", &[EMBED_DIM, EMBED_DIM]);

    let shape = [NUM_QUERIES, EMBED_DIM];

    // LayerNorm on object queries
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &shape);

    // Linear projection
    let out = b.add_linear(normed, proj_w, None, &shape);

    b.build(out).expect("valid object query encoding kernel")
}

/// Bindings for object query encoding.
fn object_query_encoding_bindings() -> Vec<TensorParamBinding> {
    let d = EMBED_DIM;
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,               // object_queries [Q, D]
        TensorParamBinding::ConstantScalar(1e-5),   // eps
        TensorParamBinding::ConstantTensor(ln_w),   // ln_weight [D]
        TensorParamBinding::ConstantTensor(ln_b),   // ln_bias [D]
        TensorParamBinding::ConstantTensor(w_proj), // proj_weight [D, D]
    ]
}

/// Build a combined encoder positional encoding: features + row_embed + col_embed.
///
/// DETR uses 2D positional encoding by adding separate row and column
/// learned embeddings. This models the flattened version:
///   x_pos = features + pos_embed_concat
///
/// Input: `[NUM_POSITIONS, EMBED_DIM]` (Variable).
/// Output: `[NUM_POSITIONS, EMBED_DIM]` after addition of constant encoding.
///
/// This is the simplest test: just the add-constant pattern.
fn build_pos_add_only_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("detr_pos_add_only");

    let input = b.add_input("features", &[NUM_POSITIONS, EMBED_DIM]);
    let pos_embed = b.add_input("pos_embed", &[NUM_POSITIONS, EMBED_DIM]);

    let shape = [NUM_POSITIONS, EMBED_DIM];

    // Add positional encoding (constant shift)
    let out = b.add_binary_add(input, pos_embed, &shape);

    b.build(out).expect("valid pos add-only kernel")
}

/// Bindings for pos_add_only kernel.
fn pos_add_only_bindings() -> Vec<TensorParamBinding> {
    let pos_embed = ArrayD::from_elem(IxDyn(&[NUM_POSITIONS, EMBED_DIM]), 0.01f32);
    vec![
        TensorParamBinding::Variable,                  // features [S, D]
        TensorParamBinding::ConstantTensor(pos_embed), // pos_embed [S, D]
    ]
}

// ===========================================================================
// Tests: Spatial positional encoding (features + pos_embed -> LN -> Linear)
// ===========================================================================

/// Spatial positional encoding TensorKernelDef validates.
#[test]
fn test_detr_spatial_pos_enc_def_validates() {
    let def = build_spatial_positional_encoding_kernel();
    def.validate()
        .expect("spatial positional encoding kernel should validate");
}

/// Spatial positional encoding graph builds.
#[test]
fn test_detr_spatial_pos_enc_graph_builds() {
    let def = build_spatial_positional_encoding_kernel();
    let bindings = spatial_pos_encoding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("spatial pos encoding graph should translate");

    // BinaryAdd + LayerNorm + Linear = at least 3 nodes
    assert!(
        graph.num_nodes() >= 3,
        "spatial pos encoding graph should have >= 3 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through spatial positional encoding.
///
/// Constant positional encoding adds a fixed shift. IBP should handle
/// this as: [x_lo + c, x_hi + c] -> LayerNorm -> Linear.
#[test]
fn test_detr_spatial_pos_enc_ibp_propagates() {
    let def = build_spatial_positional_encoding_kernel();
    let bindings = spatial_pos_encoding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_POSITIONS, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through spatial positional encoding");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_POSITIONS, EMBED_DIM],
        "output shape must be [NUM_POSITIONS, EMBED_DIM]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR spatial pos encoding IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN propagation through spatial positional encoding.
#[test]
fn test_detr_spatial_pos_enc_crown_propagation() {
    let def = build_spatial_positional_encoding_kernel();
    let bindings = spatial_pos_encoding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_POSITIONS, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[NUM_POSITIONS, EMBED_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR spatial pos encoding: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record spatial positional encoding.
#[test]
fn test_detr_spatial_pos_enc_verify_and_record() {
    let def = build_spatial_positional_encoding_kernel();
    let bindings = spatial_pos_encoding_bindings();
    let input = uniform_bounds(&[NUM_POSITIONS, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "detr_spatial_pos_encoding");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_POSITIONS, EMBED_DIM]);
}

// ===========================================================================
// Tests: Object query encoding (queries -> LN -> Linear)
// ===========================================================================

/// Object query encoding TensorKernelDef validates.
#[test]
fn test_detr_object_query_enc_def_validates() {
    let def = build_object_query_encoding_kernel();
    def.validate()
        .expect("object query encoding kernel should validate");
}

/// IBP bounds propagate through object query encoding.
#[test]
fn test_detr_object_query_enc_ibp_propagates() {
    let def = build_object_query_encoding_kernel();
    let bindings = object_query_encoding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through object query encoding");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_QUERIES, EMBED_DIM],
        "output shape must be [NUM_QUERIES, EMBED_DIM]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR object query encoding IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN propagation through object query encoding.
#[test]
fn test_detr_object_query_enc_crown_propagation() {
    let def = build_object_query_encoding_kernel();
    let bindings = object_query_encoding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_QUERIES, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[NUM_QUERIES, EMBED_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR object query encoding: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record object query encoding.
#[test]
fn test_detr_object_query_enc_verify_and_record() {
    let def = build_object_query_encoding_kernel();
    let bindings = object_query_encoding_bindings();
    let input = uniform_bounds(&[NUM_QUERIES, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "detr_object_query_encoding");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_QUERIES, EMBED_DIM]);
}

// ===========================================================================
// Tests: Pure position-add (constant shift only)
// ===========================================================================

/// Pos-add-only TensorKernelDef validates.
#[test]
fn test_detr_pos_add_only_def_validates() {
    let def = build_pos_add_only_kernel();
    def.validate().expect("pos add-only kernel should validate");
}

/// IBP bounds propagate through pure positional add.
///
/// Adding a constant should shift bounds by exactly the constant value.
/// With input [-1, 1] and pos_embed = 0.01, output should be [-0.99, 1.01].
#[test]
fn test_detr_pos_add_only_ibp_propagates() {
    let def = build_pos_add_only_kernel();
    let bindings = pos_add_only_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_POSITIONS, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through pos add-only");

    assert_eq!(output.lower_upper().0.shape(), &[NUM_POSITIONS, EMBED_DIM],);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR pos add-only IBP: bounds=[{lo_min}, {hi_max}]");

    // Constant add shifts bounds exactly: [-1 + 0.01, 1 + 0.01] = [-0.99, 1.01]
    let eps = 1e-5;
    assert!(
        (lo_min - (-0.99)).abs() < eps,
        "lower should be -0.99, got {lo_min}"
    );
    assert!(
        (hi_max - 1.01).abs() < eps,
        "upper should be 1.01, got {hi_max}"
    );
}

/// CROWN through pos-add-only should match IBP exactly (linear operation).
#[test]
fn test_detr_pos_add_only_crown_propagation() {
    let def = build_pos_add_only_kernel();
    let bindings = pos_add_only_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_POSITIONS, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[NUM_POSITIONS, EMBED_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("DETR pos add-only: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}
