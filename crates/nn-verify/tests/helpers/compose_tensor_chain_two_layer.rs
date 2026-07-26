// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: two-layer Demucs encoder block composition.
//!
//! Validates that a two-layer Conv1d + Snake + InstanceNorm chain translates
//! through `tensor_kernel_to_graph` and produces a single NY
//! `GraphNetwork` where IBP and CROWN bounds propagate end-to-end.
//!
//! Single-block composition tests are in `compose_tensor_chain.rs`.

use super::common::{assert_bounds_valid, assert_crown_tighter_when_not_fallback, uniform_bounds};
use nn_dsl::adain::build_snake_scalar_kernel;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Two-layer Demucs encoder block builder
// ---------------------------------------------------------------------------

/// Build a two-layer Demucs encoder: Block1 → Block2, using TensorBlockBuilder.
///
/// Block 1: Conv1d(in_ch→mid_ch, k, stride, pad) → Snake → InstanceNorm
/// Block 2: Conv1d(mid_ch→out_ch, k, stride, pad) → Snake → InstanceNorm
///
/// Both blocks share alpha and eps parameters (typical in Demucs).
fn build_two_layer_demucs(
    in_ch: usize,
    mid_ch: usize,
    out_ch: usize,
    in_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> (nn_dsl::tensor_ir::TensorKernelDef, Vec<usize>, Vec<usize>) {
    let mid_len = (in_len + 2 * padding - kernel_size) / stride + 1;
    let out_len = (mid_len + 2 * padding - kernel_size) / stride + 1;

    let snake1 = build_snake_scalar_kernel().expect("snake kernel 1");
    let snake2 = build_snake_scalar_kernel().expect("snake kernel 2");
    let mid_shape = [mid_ch, mid_len];
    let out_shape = [out_ch, out_len];

    let mut b = TensorBlockBuilder::new("demucs_2layer");

    // Inputs
    let data = b.add_input("data", &[in_ch, in_len]);
    let w1 = b.add_input("weight1", &[mid_ch, in_ch, kernel_size]);
    let w2 = b.add_input("weight2", &[out_ch, mid_ch, kernel_size]);
    let alpha = b.add_input("alpha", &[1]);
    let eps = b.add_input("eps", &[1]);

    // Block 1: Conv1d → Snake → InstanceNorm
    let conv1 = b.add_conv1d(data, w1, None, stride, padding, &mid_shape);
    let alpha_bc1 = b.add_broadcast(alpha, &mid_shape);
    let act1 = b.add_elementwise(snake1, &[conv1, alpha_bc1], &mid_shape);
    let norm1 = b.add_instance_norm(act1, eps, 1, None, None, &mid_shape);

    // Block 2: Conv1d → Snake → InstanceNorm (reads norm1 output)
    let conv2 = b.add_conv1d(norm1, w2, None, stride, padding, &out_shape);
    let alpha_bc2 = b.add_broadcast(alpha, &out_shape);
    let act2 = b.add_elementwise(snake2, &[conv2, alpha_bc2], &out_shape);
    let norm2 = b.add_instance_norm(act2, eps, 1, None, None, &out_shape);

    let def = b.build(norm2).expect("valid graph");
    (def, mid_shape.to_vec(), out_shape.to_vec())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Two-layer Demucs encoder block builds and translates to a NY graph.
#[test]
fn test_two_layer_demucs_graph_builds() {
    let (def, _, out_shape) = build_two_layer_demucs(1, 4, 8, 16, 3, 1, 1);
    assert_eq!(def.nodes.last().unwrap().shape, out_shape);

    let w1 = ArrayD::from_elem(IxDyn(&[4, 1, 3]), 0.1f32);
    let w2 = ArrayD::from_elem(IxDyn(&[8, 4, 3]), 0.05f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w1),
        TensorParamBinding::ConstantTensor(w2),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("two-layer graph");
    assert!(graph.num_nodes() >= 6, "two-layer graph needs >= 6 nodes");
}

/// IBP bounds propagate through both layers of the two-layer Demucs encoder.
#[test]
fn test_two_layer_demucs_ibp_propagates() {
    let (def, _, out_shape) = build_two_layer_demucs(1, 4, 8, 16, 3, 1, 1);

    let w1 = ArrayD::from_elem(IxDyn(&[4, 1, 3]), 0.1f32);
    let w2 = ArrayD::from_elem(IxDyn(&[8, 4, 3]), 0.05f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w1),
        TensorParamBinding::ConstantTensor(w2),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = uniform_bounds(&[1, 16], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through two-layer Demucs");

    assert_eq!(output.lower_upper().0.shape(), out_shape.as_slice());
    assert_bounds_valid(&output);
}

/// CROWN propagation through two-layer Demucs encoder.
///
/// When CROWN succeeds (no IBP fallback), asserts bounds are tighter than IBP.
#[test]
fn test_two_layer_demucs_crown_propagates() {
    let (def, _, out_shape) = build_two_layer_demucs(1, 4, 8, 16, 3, 1, 1);

    let w1 = ArrayD::from_elem(IxDyn(&[4, 1, 3]), 0.1f32);
    let w2 = ArrayD::from_elem(IxDyn(&[8, 4, 3]), 0.05f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w1),
        TensorParamBinding::ConstantTensor(w2),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = uniform_bounds(&[1, 16], 1.0);

    let (_method, output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), out_shape.as_slice());
}

// ---------------------------------------------------------------------------
// Affine InstanceNorm composition
// ---------------------------------------------------------------------------

/// Build a single-block Demucs encoder with affine InstanceNorm (learnable gamma/beta).
///
/// Matches the dvoice Demucs pattern where InstanceNorm has per-channel scale/shift.
/// Input bindings: data(Variable), weight(ConstantTensor), alpha(ConstantScalar),
///   eps(ConstantScalar), gamma(ConstantTensor[out_ch]), beta(ConstantTensor[out_ch]).
fn build_affine_norm_block(
    in_ch: usize,
    out_ch: usize,
    in_len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> (nn_dsl::tensor_ir::TensorKernelDef, Vec<usize>) {
    let out_len = (in_len + 2 * padding - kernel_size) / stride + 1;
    let out_shape = [out_ch, out_len];

    let snake = build_snake_scalar_kernel().expect("snake kernel");
    let mut b = TensorBlockBuilder::new("demucs_affine_enc");

    let data = b.add_input("data", &[in_ch, in_len]);
    let weight = b.add_input("weight", &[out_ch, in_ch, kernel_size]);
    let alpha = b.add_input("alpha", &[1]);
    let eps = b.add_input("eps", &[1]);
    let gamma = b.add_input("gamma", &[out_ch]);
    let beta = b.add_input("beta", &[out_ch]);

    let conv = b.add_conv1d(data, weight, None, stride, padding, &out_shape);
    let alpha_bc = b.add_broadcast(alpha, &out_shape);
    let act = b.add_elementwise(snake, &[conv, alpha_bc], &out_shape);
    let norm = b.add_instance_norm(act, eps, 1, Some(gamma), Some(beta), &out_shape);

    let def = b.build(norm).expect("valid graph");
    (def, out_shape.to_vec())
}

/// Affine InstanceNorm composition: graph builds and translates.
#[test]
fn test_affine_norm_block_graph_builds() {
    let (def, out_shape) = build_affine_norm_block(1, 4, 16, 3, 1, 1);
    assert_eq!(def.nodes.last().unwrap().shape, out_shape);

    let weight = ArrayD::from_elem(IxDyn(&[4, 1, 3]), 0.1f32);
    let gamma = ArrayD::from_elem(IxDyn(&[4]), 1.0f32);
    let beta = ArrayD::from_elem(IxDyn(&[4]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
        TensorParamBinding::ConstantScalar(1.0),  // alpha
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(gamma),
        TensorParamBinding::ConstantTensor(beta),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("affine norm graph");
    assert!(
        graph.num_nodes() >= 3,
        "affine norm needs >= 3 NY nodes"
    );
}

/// Affine InstanceNorm: IBP bounds propagate with non-trivial gamma/beta.
#[test]
fn test_affine_norm_block_ibp_nontrivial_params() {
    let (def, out_shape) = build_affine_norm_block(1, 4, 16, 3, 1, 1);

    let weight = ArrayD::from_elem(IxDyn(&[4, 1, 3]), 0.1f32);
    // Non-trivial affine: gamma=2.0 (scale up), beta=0.5 (shift up).
    // This matches a realistic learnable parameter configuration.
    let gamma = ArrayD::from_elem(IxDyn(&[4]), 2.0f32);
    let beta = ArrayD::from_elem(IxDyn(&[4]), 0.5f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(gamma),
        TensorParamBinding::ConstantTensor(beta),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = uniform_bounds(&[1, 16], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through affine norm block");

    assert_eq!(output.lower_upper().0.shape(), out_shape.as_slice());
    assert_bounds_valid(&output);
}

/// Affine InstanceNorm: CROWN propagation with non-trivial gamma/beta.
///
/// When CROWN succeeds (no IBP fallback), asserts bounds are tighter than IBP.
#[test]
fn test_affine_norm_block_crown_propagates() {
    let (def, out_shape) = build_affine_norm_block(1, 4, 8, 3, 1, 1);

    let weight = ArrayD::from_elem(IxDyn(&[4, 1, 3]), 0.1f32);
    let gamma = ArrayD::from_elem(IxDyn(&[4]), 2.0f32);
    let beta = ArrayD::from_elem(IxDyn(&[4]), 0.5f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(gamma),
        TensorParamBinding::ConstantTensor(beta),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = uniform_bounds(&[1, 8], 1.0);

    let (_method, output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), out_shape.as_slice());
    assert_bounds_valid(&output);
}

// ---------------------------------------------------------------------------
// Dvoice-scale tests
// ---------------------------------------------------------------------------

/// Dvoice-scale two-layer Demucs: Conv1d(1→48, k=8, s=4) → Conv1d(48→96, k=8, s=4).
#[test]
fn test_two_layer_demucs_dvoice_scale() {
    let (def, _, _) = build_two_layer_demucs(1, 48, 96, 256, 8, 4, 2);

    let w1 = ArrayD::from_elem(IxDyn(&[48, 1, 8]), 0.01f32);
    let w2 = ArrayD::from_elem(IxDyn(&[96, 48, 8]), 0.005f32);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w1),
        TensorParamBinding::ConstantTensor(w2),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantScalar(1e-5),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("dvoice 2-layer graph");

    let input = uniform_bounds(&[1, 256], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through dvoice 2-layer");

    // Block 1: (256 + 4 - 8)/4 + 1 = 64; Block 2: (64 + 4 - 8)/4 + 1 = 16
    assert_eq!(output.lower_upper().0.shape(), &[96, 16]);
    assert_bounds_valid(&output);
}
