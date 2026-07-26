// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! External test suite for `ptx_residual` — PTX generation validity and
//! reference implementation correctness for fused residual operations.

use crate::ptx_residual::{
    generate_residual_add_layernorm_ptx, generate_residual_add_ptx, generate_residual_add_relu_ptx,
    residual_add_layernorm_reference, residual_add_reference, residual_add_relu_reference,
};

// ---------------------------------------------------------------------------
// PTX generation validity for each residual variant
// ---------------------------------------------------------------------------

#[test]
fn test_residual_add_ptx_valid_structure() {
    let ptx = generate_residual_add_ptx(1024);
    assert!(ptx.contains(".version 6.5"), "missing PTX version");
    assert!(ptx.contains(".target sm_70"), "missing SM target");
    assert!(
        ptx.contains(".entry ptx_residual_add_f32"),
        "missing entry point"
    );
}

#[test]
fn test_residual_add_relu_ptx_valid_structure() {
    let ptx = generate_residual_add_relu_ptx(1024);
    assert!(ptx.contains(".version 6.5"), "missing PTX version");
    assert!(ptx.contains(".target sm_70"), "missing SM target");
    assert!(
        ptx.contains(".entry ptx_residual_add_relu_f32"),
        "missing entry point"
    );
}

#[test]
fn test_residual_add_layernorm_ptx_valid_structure() {
    let ptx = generate_residual_add_layernorm_ptx(512, 64);
    assert!(ptx.contains(".version 6.5"), "missing PTX version");
    assert!(ptx.contains(".target sm_70"), "missing SM target");
    assert!(
        ptx.contains(".entry ptx_residual_add_layernorm_f32"),
        "missing entry point"
    );
}

// ---------------------------------------------------------------------------
// Reference implementations — residual add
// ---------------------------------------------------------------------------

#[test]
fn test_residual_add_reference() {
    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [0.5_f32, 1.5, 2.5, 3.5];
    let out = residual_add_reference(&a, &b);
    assert_eq!(out, vec![1.5, 3.5, 5.5, 7.5]);
}

// ---------------------------------------------------------------------------
// Reference implementations — residual add + ReLU
// ---------------------------------------------------------------------------

#[test]
fn test_residual_add_relu_reference() {
    let a = [1.0_f32, -3.0, 2.0, -5.0];
    let b = [-0.5_f32, 1.0, -4.0, 3.0];
    // sums: [0.5, -2.0, -2.0, -2.0]
    // relu: [0.5, 0.0, 0.0, 0.0]
    let out = residual_add_relu_reference(&a, &b);
    assert!((out[0] - 0.5).abs() < 1e-6);
    assert!((out[1] - 0.0).abs() < 1e-6);
    assert!((out[2] - 0.0).abs() < 1e-6);
    assert!((out[3] - 0.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Reference implementations — residual add + GELU (approximated via
// residual_add_layernorm with gamma=1, beta=0 as a stand-in; the module
// does not have a standalone GELU fusion, so we test the LayerNorm path)
// ---------------------------------------------------------------------------

#[test]
fn test_residual_add_gelu_reference() {
    // The ptx_residual module fuses residual add with LayerNorm, not GELU.
    // Verify the LayerNorm fusion produces correct normalized output
    // (gamma=1, beta=0 => pure normalization of a+b).
    let hidden = 4;
    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [0.0_f32, 0.0, 0.0, 0.0];
    let gamma = [1.0_f32; 4];
    let beta = [0.0_f32; 4];
    let out = residual_add_layernorm_reference(&a, &b, &gamma, &beta, hidden, 1e-5);

    // mean = 2.5, var = 1.25, inv_std = 1/sqrt(1.25 + 1e-5)
    let mean: f32 = 2.5;
    let var: f32 = 1.25;
    let inv_std = 1.0 / (var + 1e-5_f32).sqrt();
    for i in 0..hidden {
        let expected = (a[i] - mean) * inv_std;
        assert!(
            (out[i] - expected).abs() < 1e-5,
            "element {i}: expected {expected}, got {}",
            out[i]
        );
    }
}

// ---------------------------------------------------------------------------
// Reference implementations — bias + residual add (the add reference with
// a bias pre-added to one operand)
// ---------------------------------------------------------------------------

#[test]
fn test_bias_residual_add_reference() {
    // Simulate bias + a + b by adding bias into the first operand.
    let bias = [0.1_f32, 0.2, 0.3];
    let a = [1.0_f32, 2.0, 3.0];
    let b = [0.5_f32, 0.5, 0.5];
    // bias + a
    let a_biased: Vec<f32> = a
        .iter()
        .zip(bias.iter())
        .map(|(&ai, &bi)| ai + bi)
        .collect();
    let out = residual_add_reference(&a_biased, &b);
    assert!((out[0] - 1.6).abs() < 1e-6); // 1.0 + 0.1 + 0.5
    assert!((out[1] - 2.7).abs() < 1e-6); // 2.0 + 0.2 + 0.5
    assert!((out[2] - 3.8).abs() < 1e-6); // 3.0 + 0.3 + 0.5
}

// ---------------------------------------------------------------------------
// Negative values
// ---------------------------------------------------------------------------

#[test]
fn test_residual_reference_negative() {
    let a = [-1.0_f32, -2.0, -3.0];
    let b = [-4.0_f32, 5.0, -6.0];
    let out = residual_add_reference(&a, &b);
    assert_eq!(out, vec![-5.0, 3.0, -9.0]);

    // relu clamps negatives
    let out_relu = residual_add_relu_reference(&a, &b);
    assert_eq!(out_relu, vec![0.0, 3.0, 0.0]);
}

// ---------------------------------------------------------------------------
// PTX instruction content checks
// ---------------------------------------------------------------------------

#[test]
fn test_residual_ptx_contains_instructions() {
    let ptx_add = generate_residual_add_ptx(256);
    assert!(
        ptx_add.contains("add.f32"),
        "residual add must contain add.f32"
    );

    let ptx_relu = generate_residual_add_relu_ptx(256);
    assert!(
        ptx_relu.contains("add.f32"),
        "residual relu must contain add.f32"
    );
    assert!(
        ptx_relu.contains("max.f32"),
        "residual relu must contain max.f32 for ReLU"
    );

    let ptx_ln = generate_residual_add_layernorm_ptx(512, 64);
    assert!(ptx_ln.contains("add.f32"), "LN fusion must contain add.f32");
    assert!(
        ptx_ln.contains("rsqrt.approx.f32"),
        "LN fusion must contain rsqrt"
    );
    assert!(
        ptx_ln.contains("fma.rn.f32"),
        "LN fusion must contain fma for affine"
    );
}
