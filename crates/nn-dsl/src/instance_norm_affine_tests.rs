// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for affine InstanceNorm K2 (#302). Extracted from
//! instance_norm_tests.rs for the 500-line limit (#356).

use super::*;
use crate::tensor_ir::tensor_ir_pretty_print;

// --- Affine InstanceNorm tests (#302) ---

#[test]
fn test_instance_norm_k2_affine_validates() {
    let k2 = build_instance_norm_affine(2, 4, 16).expect("build must succeed");
    k2.validate()
        .expect("K2 affine InstanceNorm IR must validate");
}

#[test]
fn test_instance_norm_k2_affine_node_count() {
    let k2 = build_instance_norm_affine(2, 4, 16).expect("build must succeed");
    assert_eq!(
        k2.nodes.len(),
        5,
        "native affine: 4 inputs (x, eps, gamma, beta) + 1 InstanceNorm1d"
    );
}

#[test]
fn test_instance_norm_k2_affine_output_shape() {
    let k2 = build_instance_norm_affine(2, 4, 16).expect("build must succeed");
    let output_shape = &k2.nodes[k2.output.index()].shape;
    assert_eq!(output_shape, &[2, 4, 16]);
}

#[test]
fn test_instance_norm_k2_affine_gamma_beta_shapes() {
    let k2 = build_instance_norm_affine(2, 4, 16).expect("build must succeed");
    // gamma at index 2, beta at index 3 — both shape [C]
    assert_eq!(k2.nodes[2].shape, vec![4], "gamma shape must be [C]");
    assert_eq!(k2.nodes[3].shape, vec![4], "beta shape must be [C]");
}

#[test]
fn test_instance_norm_k2_affine_zero_dim_returns_err() {
    let result = build_instance_norm_affine(1, 0, 16);
    assert!(result.is_err(), "zero dimension must return Err");
}

#[test]
fn test_instance_norm_k2_affine_pretty_print() {
    let k2 = build_instance_norm_affine(1, 2, 4).expect("build must succeed");
    let ir = tensor_ir_pretty_print(&k2);
    assert!(
        ir.contains("instance_norm_1d(%0, eps=%1, axis=2, gamma=%2, beta=%3)"),
        "pretty print must show gamma/beta, got:\n{ir}"
    );
}

#[test]
fn test_instance_norm_k2_decomposed_affine_validates() {
    let k2 =
        build_instance_norm_decomposed_affine(2, 4, 16).expect("decomposed build must succeed");
    k2.validate()
        .expect("K2 decomposed affine InstanceNorm IR must validate");
}

#[test]
fn test_instance_norm_k2_decomposed_affine_node_count() {
    let k2 =
        build_instance_norm_decomposed_affine(2, 4, 16).expect("decomposed build must succeed");
    assert_eq!(
        k2.nodes.len(),
        20,
        "decomposed affine: 4 inputs + 2 reductions + 5 broadcasts + 2 reshapes + 7 elementwise"
    );
}

#[test]
fn test_instance_norm_k2_decomposed_affine_has_reshape_nodes() {
    use crate::tensor_ir::TensorOpKind;
    let k2 = build_instance_norm_decomposed_affine(1, 3, 8).expect("decomposed build must succeed");
    let reshape_count = k2
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, TensorOpKind::Reshape { .. }))
        .count();
    assert_eq!(
        reshape_count, 2,
        "need 2 reshapes: gamma [C]→[1,C,1] and beta [C]→[1,C,1]"
    );
}

#[test]
fn test_instance_norm_affine_ref_known_values() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let gamma = [2.0];
    let beta = [10.0];
    let eps = 1e-5;
    let out = instance_norm_affine_ref(&x, &gamma, &beta, 1, 1, 4, eps).expect("ref must succeed");

    // Non-affine normalized values
    let mean = 2.5;
    let var = 1.25;
    let inv_std = 1.0 / (var + eps).sqrt();
    let expected: Vec<f32> = x
        .iter()
        .map(|v| 2.0 * (v - mean) * inv_std + 10.0)
        .collect();

    for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "mismatch at index {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
fn test_instance_norm_affine_ref_identity_matches_non_affine() {
    // gamma=1, beta=0 should match the non-affine reference
    let x: Vec<f32> = (0..24).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let gamma = [1.0; 3];
    let beta = [0.0; 3];
    let eps = 1e-5;

    let affine_out =
        instance_norm_affine_ref(&x, &gamma, &beta, 1, 3, 8, eps).expect("affine ref must succeed");
    let non_affine_out = instance_norm_ref(&x, 1, 3, 8, eps).expect("non-affine ref must succeed");

    for (i, (&a, &b)) in affine_out.iter().zip(non_affine_out.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "affine(gamma=1,beta=0) must match non-affine at index {i}: {a} vs {b}"
        );
    }
}

#[test]
fn test_instance_norm_affine_ref_constant_input_returns_beta() {
    // Constant input: (x - mean) = 0, so output = gamma * 0 + beta = beta
    let x = vec![5.0f32; 2 * 3 * 8];
    let gamma = [2.0, 3.0, 4.0];
    let beta = [10.0, 20.0, 30.0];
    let eps = 1e-5;

    let out = instance_norm_affine_ref(&x, &gamma, &beta, 2, 3, 8, eps).expect("ref must succeed");

    for bi in 0..2 {
        for (ci, &expected_beta) in beta.iter().enumerate() {
            for ti in 0..8 {
                let idx = (bi * 3 + ci) * 8 + ti;
                assert!(
                    (out[idx] - expected_beta).abs() < 1e-3,
                    "constant input should give beta[{ci}]={expected_beta}, got {} at [{bi},{ci},{ti}]",
                    out[idx]
                );
            }
        }
    }
}

#[test]
fn test_instance_norm_affine_ref_gamma_length_mismatch_returns_err() {
    let x = vec![1.0f32; 8];
    let gamma = [1.0, 2.0]; // wrong: 2 != c=1
    let beta = [0.0];
    let result = instance_norm_affine_ref(&x, &gamma, &beta, 1, 1, 8, 1e-5);
    assert!(result.is_err(), "gamma length mismatch must return Err");
}

#[test]
fn test_instance_norm_affine_ref_beta_length_mismatch_returns_err() {
    let x = vec![1.0f32; 8];
    let gamma = [1.0];
    let beta = [0.0, 1.0]; // wrong: 2 != c=1
    let result = instance_norm_affine_ref(&x, &gamma, &beta, 1, 1, 8, 1e-5);
    assert!(result.is_err(), "beta length mismatch must return Err");
}

#[test]
fn test_instance_norm_affine_scalar_known_values() {
    // x=3, mean=2, var=1, eps=0 (eps=1e-5 for safety), gamma=2, beta=10
    // normalized = (3 - 2) / sqrt(1 + 1e-5) ≈ 1.0
    // output = 2 * 1.0 + 10 = 12.0
    let y = instance_norm_affine_scalar(3.0, 2.0, 1.0, 1e-5, 2.0, 10.0)
        .expect("must succeed for valid inputs");
    assert!((y - 12.0).abs() < 0.01, "expected ~12.0, got {y}");
}

#[test]
fn test_instance_norm_affine_scalar_nan_input_returns_err() {
    let result = instance_norm_affine_scalar(f32::NAN, 0.0, 1.0, 1e-5, 1.0, 0.0);
    assert!(result.is_err(), "NaN input must return Err");
}

#[test]
fn test_instance_norm_affine_scalar_kernel_builds() {
    let kernel = build_instance_norm_affine_scalar_kernel().expect("scalar kernel must build");
    assert_eq!(
        kernel.params.len(),
        6,
        "6 params: x, mean, var_val, eps, gamma, beta"
    );
    assert_eq!(kernel.name, "instance_norm_affine_scalar");
}

#[test]
fn test_instance_norm_affine_ref_nan_x_rejected() {
    let x = &[1.0, f32::NAN, 3.0, 4.0];
    let gamma = &[1.0, 1.0];
    let beta = &[0.0, 0.0];
    let err = instance_norm_affine_ref(x, gamma, beta, 1, 2, 2, 1e-5).unwrap_err();
    assert!(
        matches!(
            err,
            KernelError::NonFiniteSliceElement {
                name: "x",
                index: 1,
                ..
            }
        ),
        "NaN at x[1] should be caught, got: {err}"
    );
}

#[test]
fn test_instance_norm_affine_ref_nan_gamma_rejected() {
    let x = &[1.0, 2.0, 3.0, 4.0];
    let gamma = &[f32::NAN, 1.0];
    let beta = &[0.0, 0.0];
    let err = instance_norm_affine_ref(x, gamma, beta, 1, 2, 2, 1e-5).unwrap_err();
    assert!(
        matches!(
            err,
            KernelError::NonFiniteSliceElement {
                name: "gamma",
                index: 0,
                ..
            }
        ),
        "NaN at gamma[0] should be caught, got: {err}"
    );
}
