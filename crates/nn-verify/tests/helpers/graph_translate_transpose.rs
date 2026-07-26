// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `TensorOpKind::Transpose` → NY `TransposeLayer`.
//!
//! Tests:
//! - Graph builds from 2D transpose
//! - Graph builds from 3D transpose (attention-style)
//! - IBP bounds propagate correctly through transpose
//! - CROWN backward propagation succeeds
//! - Constant input passes through unchanged
//! - Validation rejects invalid permutations
//!
//! Part of #809.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{
    propagate_with_crown_fallback, tensor_kernel_to_graph, BoundedTensor, TensorParamBinding,
};
use ndarray::{ArrayD, IxDyn};

/// Build a Transpose kernel using TensorBlockBuilder.
fn transpose_kernel(in_shape: &[usize], axes: &[usize]) -> TensorKernelDef {
    let out_shape: Vec<usize> = axes.iter().map(|&a| in_shape[a]).collect();
    let mut b = TensorBlockBuilder::new("transpose_test");
    let x = b.add_input("x", in_shape);
    let out = b.add_transpose(x, axes, &out_shape);
    b.build(out).expect("valid transpose graph")
}

/// 2D transpose [M, N] → [N, M] builds a valid graph.
#[test]
fn test_transpose_2d_builds_graph() {
    let def = transpose_kernel(&[3, 4], &[1, 0]);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable])
        .expect("2D transpose graph should build");
    assert!(
        graph.num_nodes() >= 1,
        "graph should have at least the transpose node"
    );
}

/// 3D transpose [T, H, D] → [H, T, D] (attention-style head permutation).
#[test]
fn test_transpose_3d_attention_builds_graph() {
    let def = transpose_kernel(&[8, 4, 16], &[1, 0, 2]);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable])
        .expect("3D attention transpose graph should build");
    assert!(
        graph.num_nodes() >= 1,
        "graph should have at least the transpose node"
    );
}

/// IBP bounds propagate correctly through 2D transpose.
///
/// TransposeLayer permutes bounds arrays using the same axes permutation.
/// If input bounds are [-1, 3] uniformly on shape [3, 4], the transposed
/// output [4, 3] should also have bounds [-1, 3].
#[test]
fn test_transpose_2d_ibp_uniform_bounds() {
    let m = 3;
    let n = 4;
    let def = transpose_kernel(&[m, n], &[1, 0]);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable]).expect("graph");

    let lower = ArrayD::from_elem(IxDyn(&[m, n]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[m, n]), 3.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP through transpose");

    let out_lower = output.lower();
    let out_upper = output.upper();

    // Output should have n*m = 12 elements.
    assert_eq!(
        out_lower.len(),
        n * m,
        "output should have n*m = {} elements",
        n * m
    );

    for &v in out_lower.iter() {
        assert!((v - (-1.0)).abs() < 1e-5, "expected lower ~-1.0, got {v}");
    }
    for &v in out_upper.iter() {
        assert!((v - 3.0).abs() < 1e-5, "expected upper ~3.0, got {v}");
    }
}

/// IBP bounds propagate through 3D transpose with non-uniform bounds.
///
/// Uses different bounds per element to verify the permutation is applied
/// correctly (not just preserved for uniform bounds). Checks that specific
/// elements moved to the expected permuted positions.
#[test]
fn test_transpose_3d_ibp_nonuniform_bounds() {
    // Shape [2, 3, 4] → transpose [1, 0, 2] → [3, 2, 4]
    // axes=[1,0,2] swaps dim0 and dim1, keeps dim2 unchanged.
    let def = transpose_kernel(&[2, 3, 4], &[1, 0, 2]);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable]).expect("graph");

    // Use bounds that vary per element so we can verify actual permutation.
    let in_shape = [2, 3, 4];
    let total = 24;
    let lower_data: Vec<f32> = (0..total).map(|i| -(i as f32) - 1.0).collect();
    let upper_data: Vec<f32> = (0..total).map(|i| (i as f32) + 1.0).collect();
    let lower = ArrayD::from_shape_vec(IxDyn(&in_shape), lower_data).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&in_shape), upper_data).unwrap();
    let input = BoundedTensor::new(lower.clone(), upper.clone()).expect("input bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 3D transpose");

    // Output shape: [3, 2, 4]
    assert_eq!(output.lower().shape(), &[3, 2, 4]);
    assert_eq!(output.upper().shape(), &[3, 2, 4]);

    // Verify actual permutation: input[i,j,k] should map to output[j,i,k].
    // Spot-check specific elements to confirm data was reordered, not just copied.
    let out_lo = output.lower();
    let out_hi = output.upper();

    // input[0,1,2] → output[1,0,2]: lower = -(0*12 + 1*4 + 2) - 1 = -7.0
    assert!(
        (out_lo[[1, 0, 2]] - lower[[0, 1, 2]]).abs() < 1e-5,
        "permutation check: output[1,0,2] lower should be input[0,1,2] lower = {}, got {}",
        lower[[0, 1, 2]],
        out_lo[[1, 0, 2]]
    );
    assert!(
        (out_hi[[1, 0, 2]] - upper[[0, 1, 2]]).abs() < 1e-5,
        "permutation check: output[1,0,2] upper should be input[0,1,2] upper = {}, got {}",
        upper[[0, 1, 2]],
        out_hi[[1, 0, 2]]
    );

    // input[1,2,3] → output[2,1,3]: lower = -(1*12 + 2*4 + 3) - 1 = -24.0
    assert!(
        (out_lo[[2, 1, 3]] - lower[[1, 2, 3]]).abs() < 1e-5,
        "permutation check: output[2,1,3] lower should be input[1,2,3] lower = {}, got {}",
        lower[[1, 2, 3]],
        out_lo[[2, 1, 3]]
    );

    // All bounds should be valid: lower <= upper everywhere.
    for (lo, up) in out_lo.iter().zip(out_hi.iter()) {
        assert!(lo <= up, "bound violation: lower {lo} > upper {up}");
    }
}

/// CROWN backward propagation succeeds through transpose with non-uniform bounds.
///
/// Uses non-uniform bounds so CROWN vs IBP comparison is meaningful.
/// With uniform bounds on a linear op, CROWN and IBP produce identical results,
/// making tightness assertions trivially true.
#[test]
fn test_transpose_crown_propagation() {
    let def = transpose_kernel(&[3, 4], &[1, 0]);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable]).expect("graph");

    // Non-uniform bounds: each element has different range width.
    let total = 12;
    let lower_data: Vec<f32> = (0..total).map(|i| -(i as f32) - 1.0).collect();
    let upper_data: Vec<f32> = (0..total).map(|i| (i as f32) + 1.0).collect();
    let lower = ArrayD::from_shape_vec(IxDyn(&[3, 4]), lower_data).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3, 4]), upper_data).unwrap();
    let input = BoundedTensor::new(lower, upper).expect("input bounds");

    // CROWN should succeed for linear ops like transpose.
    let (_method, crown_bounds, _diag) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN through transpose");
    let ibp_output = graph.propagate_ibp(&input).expect("IBP through transpose");

    // CROWN bounds should be at least as tight as IBP.
    for (&crown_lo, &ibp_lo) in crown_bounds.lower().iter().zip(ibp_output.lower().iter()) {
        assert!(
            crown_lo >= ibp_lo - 1e-5,
            "CROWN lower {crown_lo} should be >= IBP lower {ibp_lo}"
        );
    }
    for (&crown_up, &ibp_up) in crown_bounds.upper().iter().zip(ibp_output.upper().iter()) {
        assert!(
            crown_up <= ibp_up + 1e-5,
            "CROWN upper {crown_up} should be <= IBP upper {ibp_up}"
        );
    }
}

/// Constant input passes through transpose unchanged.
#[test]
fn test_transpose_constant_passthrough() {
    let def = transpose_kernel(&[3, 4], &[1, 0]);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::ConstantScalar(5.0)])
        .expect("constant transpose should succeed");

    // Constant-fold output is AddConstant(5.0).
    let lower = ArrayD::from_elem(IxDyn(&[4, 3]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4, 3]), 0.0f32);
    let input = BoundedTensor::new(lower, upper).expect("input bounds");
    let output = graph.propagate_ibp(&input).expect("IBP propagation");

    for &v in output.lower().iter() {
        assert!(
            (v - 5.0).abs() < 1e-5,
            "expected constant fold result 5.0, got {v}"
        );
    }
}

/// Validation rejects axes with wrong length.
#[test]
fn test_transpose_validation_rejects_wrong_axes_length() {
    let in_shape = vec![3, 4];
    let mut b = TensorBlockBuilder::new("bad_transpose");
    let x = b.add_input("x", &in_shape);
    let out = b.add_transpose(x, &[2, 0, 1], &[4, 3, 0]); // axes len 3 != rank 2
    let def = b.build(out);
    // build validates, so it should fail.
    assert!(
        def.is_err(),
        "Transpose with axes length != rank should fail validation"
    );
}

/// Validation rejects axes with duplicate entries.
#[test]
fn test_transpose_validation_rejects_duplicate_axis() {
    let in_shape = vec![3, 4, 5];
    let mut b = TensorBlockBuilder::new("dup_transpose");
    let x = b.add_input("x", &in_shape);
    let out = b.add_transpose(x, &[0, 0, 2], &[3, 3, 5]); // axis 0 appears twice
    let def = b.build(out);
    assert!(
        def.is_err(),
        "Transpose with duplicate axis should fail validation"
    );
}

/// Validation rejects axes with out-of-bounds index.
#[test]
fn test_transpose_validation_rejects_out_of_bounds_axis() {
    let in_shape = vec![3, 4];
    let mut b = TensorBlockBuilder::new("oob_transpose");
    let x = b.add_input("x", &in_shape);
    let out = b.add_transpose(x, &[0, 5], &[3, 4]); // axis 5 >= rank 2
    let def = b.build(out);
    assert!(
        def.is_err(),
        "Transpose with axis >= rank should fail validation"
    );
}

/// Identity permutation [0, 1, 2] preserves bounds exactly.
#[test]
fn test_transpose_identity_preserves_bounds() {
    let def = transpose_kernel(&[2, 3, 4], &[0, 1, 2]);
    let graph = tensor_kernel_to_graph(&def, &[TensorParamBinding::Variable]).expect("graph");

    let lower_data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let upper_data: Vec<f32> = (0..24).map(|i| i as f32 + 1.0).collect();
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), lower_data).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), upper_data).unwrap();
    let input = BoundedTensor::new(lower.clone(), upper.clone()).expect("input bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through identity transpose");

    // Identity transpose should preserve bounds exactly.
    for (orig, out) in lower.iter().zip(output.lower().iter()) {
        assert!(
            (orig - out).abs() < 1e-5,
            "identity transpose should preserve lower bounds: {orig} vs {out}"
        );
    }
    for (orig, out) in upper.iter().zip(output.upper().iter()) {
        assert!(
            (orig - out).abs() < 1e-5,
            "identity transpose should preserve upper bounds: {orig} vs {out}"
        );
    }
}
