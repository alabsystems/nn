// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: TensorOpKind::BatchNorm → NY BatchNormLayer translation.
//!
//! Part of #1101: Verifies W4's BatchNorm NY translation (7e4a08fe)
//! with IBP and CROWN propagation tests following the GroupNorm/LayerNorm patterns.

use super::common;

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{
    propagate_with_crown_fallback, tensor_kernel_to_graph, PropMethod, TensorParamBinding,
};
use ndarray::{ArrayD, IxDyn};

/// Build a BatchNorm kernel for [C, T] input using frozen running statistics.
fn batch_norm_kernel_2d(channels: usize, time_len: usize) -> nn_dsl::TensorKernelDef {
    let mut b = TensorBlockBuilder::new("batch_norm_2d_test");
    let x = b.add_input("x", &[channels, time_len]);
    let mean = b.add_input("running_mean", &[channels]);
    let var = b.add_input("running_var", &[channels]);
    let weight = b.add_input("weight", &[channels]);
    let bias = b.add_input("bias", &[channels]);
    let eps = b.add_input("eps", &[1]);

    let out = b.add_batch_norm(x, mean, var, weight, bias, eps, &[channels, time_len]);
    b.build(out).expect("valid graph")
}

/// Build a BatchNorm kernel for [B, C, T] input (3D with batch dimension).
fn batch_norm_kernel_3d(
    batch: usize,
    channels: usize,
    time_len: usize,
) -> nn_dsl::TensorKernelDef {
    let mut b = TensorBlockBuilder::new("batch_norm_3d_test");
    let x = b.add_input("x", &[batch, channels, time_len]);
    let mean = b.add_input("running_mean", &[channels]);
    let var = b.add_input("running_var", &[channels]);
    let weight = b.add_input("weight", &[channels]);
    let bias = b.add_input("bias", &[channels]);
    let eps = b.add_input("eps", &[1]);

    let out = b.add_batch_norm(
        x,
        mean,
        var,
        weight,
        bias,
        eps,
        &[batch, channels, time_len],
    );
    b.build(out).expect("valid graph")
}

// -- Builder and shape tests ---------------------------------------------------

#[test]
fn test_batch_norm_2d_validates() {
    let _def = batch_norm_kernel_2d(4, 8);
}

#[test]
fn test_batch_norm_3d_validates() {
    let _def = batch_norm_kernel_3d(2, 4, 8);
}

#[test]
fn test_batch_norm_2d_output_shape() {
    let def = batch_norm_kernel_2d(4, 8);
    let output = &def.nodes[def.output.index()];
    assert_eq!(
        output.shape,
        vec![4, 8],
        "2D BatchNorm output shape should match input [C, T]"
    );
}

#[test]
fn test_batch_norm_3d_output_shape() {
    let def = batch_norm_kernel_3d(2, 4, 8);
    let output = &def.nodes[def.output.index()];
    assert_eq!(
        output.shape,
        vec![2, 4, 8],
        "3D BatchNorm output shape should match input [B, C, T]"
    );
}

#[test]
fn test_batch_norm_2d_node_count() {
    let def = batch_norm_kernel_2d(4, 8);
    // 6 inputs (x, mean, var, weight, bias, eps) + 1 BatchNorm node = 7
    assert_eq!(
        def.nodes.len(),
        7,
        "BatchNorm kernel should have 6 inputs + 1 op = 7 nodes"
    );
}

// -- Graph build tests ---------------------------------------------------------

#[test]
fn test_batch_norm_2d_builds_gamma_crown_graph() {
    let channels = 3;
    let time_len = 4;
    let def = batch_norm_kernel_2d(channels, time_len);

    let mean = ArrayD::from_elem(IxDyn(&[channels]), 0.0f32);
    let var = ArrayD::from_elem(IxDyn(&[channels]), 1.0f32);
    let gamma = ArrayD::from_elem(IxDyn(&[channels]), 1.0f32);
    let beta = ArrayD::from_elem(IxDyn(&[channels]), 0.0f32);

    let bindings = [
        TensorParamBinding::Variable,              // x
        TensorParamBinding::ConstantTensor(mean),  // running_mean
        TensorParamBinding::ConstantTensor(var),   // running_var
        TensorParamBinding::ConstantTensor(gamma), // weight (gamma)
        TensorParamBinding::ConstantTensor(beta),  // bias (beta)
        TensorParamBinding::ConstantScalar(1e-5),  // eps
    ];
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("BatchNorm should build NY graph");

    // BatchNormLayer is a single NY layer. Node count depends on
    // NY's internal representation — at minimum 1 (the layer itself).
    assert!(
        graph.num_nodes() >= 1,
        "graph should have at least the BatchNormLayer, got {}",
        graph.num_nodes()
    );
}

// -- IBP propagation tests -----------------------------------------------------

#[test]
fn test_batch_norm_2d_ibp_identity() {
    // When mean=0, var=1, gamma=1, beta=0, eps≈0 → BatchNorm ≈ identity
    let channels = 2;
    let time_len = 4;
    let def = batch_norm_kernel_2d(channels, time_len);

    let mean = ArrayD::from_elem(IxDyn(&[channels]), 0.0f32);
    let var = ArrayD::from_elem(IxDyn(&[channels]), 1.0f32);
    let gamma = ArrayD::from_elem(IxDyn(&[channels]), 1.0f32);
    let beta = ArrayD::from_elem(IxDyn(&[channels]), 0.0f32);

    let bindings = [
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(mean),
        TensorParamBinding::ConstantTensor(var),
        TensorParamBinding::ConstantTensor(gamma),
        TensorParamBinding::ConstantTensor(beta),
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("BatchNorm identity graph should build");

    let input = common::uniform_bounds(&[channels, time_len], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    common::assert_bounds_valid(&output);

    // Identity-like BatchNorm: output bounds should be close to input bounds.
    // scale = gamma / sqrt(var + eps) = 1 / sqrt(1 + 1e-5) ≈ 0.999995
    // bias_internal = beta - mean * scale = 0 - 0 = 0
    // Output ≈ input * 0.999995 ≈ [-1, 1]
    let (lo, hi) = output.lower_upper();
    for (&l, &h) in lo.iter().zip(hi.iter()) {
        assert!(l > -1.1, "identity lower bound {l} should be close to -1.0");
        assert!(h < 1.1, "identity upper bound {h} should be close to 1.0");
    }
}

#[test]
fn test_batch_norm_2d_ibp_non_trivial() {
    // Non-trivial parameters: mean=2, var=4, gamma=0.5, beta=1.0
    // scale = 0.5 / sqrt(4 + 1e-5) ≈ 0.25
    // bias_internal = 1.0 - 2.0 * 0.25 = 0.5
    // Output = 0.25 * x + 0.5
    // For x ∈ [-3, 3]: output ∈ [0.25*(-3) + 0.5, 0.25*3 + 0.5] = [-0.25, 1.25]
    let channels = 2;
    let time_len = 4;
    let def = batch_norm_kernel_2d(channels, time_len);

    let mean = ArrayD::from_elem(IxDyn(&[channels]), 2.0f32);
    let var = ArrayD::from_elem(IxDyn(&[channels]), 4.0f32);
    let gamma = ArrayD::from_elem(IxDyn(&[channels]), 0.5f32);
    let beta = ArrayD::from_elem(IxDyn(&[channels]), 1.0f32);

    let bindings = [
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(mean),
        TensorParamBinding::ConstantTensor(var),
        TensorParamBinding::ConstantTensor(gamma),
        TensorParamBinding::ConstantTensor(beta),
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("BatchNorm graph should build");

    let input = common::uniform_bounds(&[channels, time_len], 3.0);
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    common::assert_bounds_valid(&output);

    // Bounds should contain the exact interval [-0.25, 1.25]
    let (lo, hi) = output.lower_upper();
    for (&l, &h) in lo.iter().zip(hi.iter()) {
        assert!(l <= -0.24, "lower bound {l} must cover exact lower -0.25");
        assert!(h >= 1.24, "upper bound {h} must cover exact upper 1.25");
    }
}

#[test]
fn test_batch_norm_3d_ibp_finite() {
    let batch = 2;
    let channels = 3;
    let time_len = 8;
    let def = batch_norm_kernel_3d(batch, channels, time_len);

    let mean = ArrayD::from_shape_vec(IxDyn(&[channels]), vec![0.5, -0.3, 1.2]).unwrap();
    let var = ArrayD::from_shape_vec(IxDyn(&[channels]), vec![2.0, 0.5, 3.0]).unwrap();
    let gamma = ArrayD::from_shape_vec(IxDyn(&[channels]), vec![1.0, 0.8, 1.5]).unwrap();
    let beta = ArrayD::from_shape_vec(IxDyn(&[channels]), vec![0.0, 0.1, -0.2]).unwrap();

    let bindings = [
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(mean),
        TensorParamBinding::ConstantTensor(var),
        TensorParamBinding::ConstantTensor(gamma),
        TensorParamBinding::ConstantTensor(beta),
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("3D BatchNorm graph should build");

    let input = common::uniform_bounds(&[batch, channels, time_len], 5.0);
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    common::assert_bounds_valid(&output);
}

// -- CROWN propagation tests ---------------------------------------------------

#[test]
fn test_batch_norm_2d_crown_succeeds() {
    // BatchNormLayer is a linear layer (affine transform), so CROWN should succeed.
    let channels = 2;
    let time_len = 4;
    let def = batch_norm_kernel_2d(channels, time_len);

    let mean = ArrayD::from_elem(IxDyn(&[channels]), 0.5f32);
    let var = ArrayD::from_elem(IxDyn(&[channels]), 2.0f32);
    let gamma = ArrayD::from_elem(IxDyn(&[channels]), 1.0f32);
    let beta = ArrayD::from_elem(IxDyn(&[channels]), 0.0f32);

    let bindings = [
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(mean),
        TensorParamBinding::ConstantTensor(var),
        TensorParamBinding::ConstantTensor(gamma),
        TensorParamBinding::ConstantTensor(beta),
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("BatchNorm graph should build");

    let input = common::uniform_bounds(&[channels, time_len], 1.0);
    let (method, crown_output, _) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN propagation");

    assert_eq!(
        method,
        PropMethod::Crown,
        "BatchNormLayer is linear — CROWN should succeed"
    );
    common::assert_bounds_valid(&crown_output);
}

#[test]
fn test_batch_norm_2d_crown_tighter_than_ibp() {
    // For a single linear layer, CROWN and IBP should produce identical bounds.
    // But verify the invariant holds: CROWN >= IBP lower, CROWN <= IBP upper.
    let channels = 2;
    let time_len = 4;
    let def = batch_norm_kernel_2d(channels, time_len);

    let mean = ArrayD::from_shape_vec(IxDyn(&[channels]), vec![1.0, -0.5]).unwrap();
    let var = ArrayD::from_shape_vec(IxDyn(&[channels]), vec![3.0, 1.0]).unwrap();
    let gamma = ArrayD::from_shape_vec(IxDyn(&[channels]), vec![2.0, 0.5]).unwrap();
    let beta = ArrayD::from_shape_vec(IxDyn(&[channels]), vec![0.1, -0.3]).unwrap();

    let bindings = [
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(mean),
        TensorParamBinding::ConstantTensor(var),
        TensorParamBinding::ConstantTensor(gamma),
        TensorParamBinding::ConstantTensor(beta),
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("BatchNorm graph should build");

    let input = common::uniform_bounds(&[channels, time_len], 2.0);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    let (method, crown_output, _) = propagate_with_crown_fallback(&graph, &input).expect("CROWN");

    assert_eq!(method, PropMethod::Crown);
    common::assert_crown_tighter_than_ibp(&crown_output, &ibp_output);
}

// -- Verify-and-record pipeline test -------------------------------------------

#[test]
fn test_batch_norm_verify_and_record() {
    let channels = 3;
    let time_len = 4;
    let def = batch_norm_kernel_2d(channels, time_len);

    let mean = ArrayD::from_elem(IxDyn(&[channels]), 0.0f32);
    let var = ArrayD::from_elem(IxDyn(&[channels]), 1.0f32);
    let gamma = ArrayD::from_elem(IxDyn(&[channels]), 1.0f32);
    let beta = ArrayD::from_elem(IxDyn(&[channels]), 0.0f32);

    let bindings = [
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(mean),
        TensorParamBinding::ConstantTensor(var),
        TensorParamBinding::ConstantTensor(gamma),
        TensorParamBinding::ConstantTensor(beta),
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let input = common::uniform_bounds(&[channels, time_len], 1.0);
    let _result = common::verify_and_assert(&def, &bindings, &input, "batch_norm_2d");
}

// -- Constant-fold test --------------------------------------------------------

#[test]
fn test_batch_norm_constant_input_at_mean_folds() {
    // When constant input equals mean, (x - mean) = 0, so BatchNorm output = beta.
    // The constant-fold path produces TensorNodeValue::Constant(beta[0]).
    // The graph wrapper then applies AddConstant(beta) to NETWORK_INPUT (#477),
    // so the graph output = NETWORK_INPUT + beta, not a true constant.
    let channels = 2;
    let time_len = 4;
    let def = batch_norm_kernel_2d(channels, time_len);

    let const_val = 2.0f32;
    let mean = ArrayD::from_elem(IxDyn(&[channels]), const_val); // mean matches input
    let var = ArrayD::from_elem(IxDyn(&[channels]), 1.0f32);
    let gamma = ArrayD::from_elem(IxDyn(&[channels]), 1.0f32);
    let beta = ArrayD::from_elem(IxDyn(&[channels]), 0.5f32);

    let bindings = [
        TensorParamBinding::ConstantScalar(const_val), // constant input == mean
        TensorParamBinding::ConstantTensor(mean),
        TensorParamBinding::ConstantTensor(var),
        TensorParamBinding::ConstantTensor(gamma),
        TensorParamBinding::ConstantTensor(beta),
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    // The translation should succeed — when const == mean, output = beta.
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("constant-fold path should succeed");

    // The graph wraps the constant as AddConstant(beta) on NETWORK_INPUT (#477).
    // With input bounds [0, 0], output = 0 + beta = beta = 0.5.
    let input = common::uniform_bounds(&[channels, time_len], 0.0);
    let output = graph.propagate_ibp(&input).expect("IBP on constant-fold");
    let (lo, hi) = output.lower_upper();
    for (&l, &h) in lo.iter().zip(hi.iter()) {
        assert!(
            (l - 0.5).abs() < 1e-4,
            "at-mean constant-fold lower {l} should be ~beta=0.5"
        );
        assert!(
            (h - 0.5).abs() < 1e-4,
            "at-mean constant-fold upper {h} should be ~beta=0.5"
        );
    }
}

#[test]
fn test_batch_norm_constant_input_off_mean_folds_correctly() {
    // When const != mean, output = gamma * (const - mean) / sqrt(var + eps) + beta.
    // const=3.0, mean=0.0, var=1.0, gamma=1.0, beta=0.5, eps=1e-5
    // output = 1.0 * (3.0 - 0.0) / sqrt(1.0 + 1e-5) + 0.5 ≈ 3.0 + 0.5 = 3.5
    let channels = 2;
    let time_len = 4;
    let def = batch_norm_kernel_2d(channels, time_len);

    let mean = ArrayD::from_elem(IxDyn(&[channels]), 0.0f32);
    let var = ArrayD::from_elem(IxDyn(&[channels]), 1.0f32);
    let gamma = ArrayD::from_elem(IxDyn(&[channels]), 1.0f32);
    let beta = ArrayD::from_elem(IxDyn(&[channels]), 0.5f32);

    let bindings = [
        TensorParamBinding::ConstantScalar(3.0), // const != mean
        TensorParamBinding::ConstantTensor(mean),
        TensorParamBinding::ConstantTensor(var),
        TensorParamBinding::ConstantTensor(gamma),
        TensorParamBinding::ConstantTensor(beta),
        TensorParamBinding::ConstantScalar(1e-5),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("off-mean constant-fold should succeed (uniform params)");

    // Graph wraps constant as AddConstant(3.5) on NETWORK_INPUT.
    // With input [0, 0], output = 0 + 3.5 = 3.5.
    let input = common::uniform_bounds(&[channels, time_len], 0.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();

    let expected = 1.0 * (3.0 - 0.0) / (1.0 + 1e-5_f32).sqrt() + 0.5;
    for (&l, &h) in lo.iter().zip(hi.iter()) {
        assert!(
            (l - expected).abs() < 1e-3,
            "off-mean lower {l} should be ~{expected}"
        );
        assert!(
            (h - expected).abs() < 1e-3,
            "off-mean upper {h} should be ~{expected}"
        );
    }
}
