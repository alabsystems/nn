// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Linear tensor IR → NY translation.
//!
//! Tests cover:
//! - Basic Linear graph construction (no bias)
//! - Linear with bias
//! - IBP bound propagation through Linear
//! - Mixed-sign weights (W+/W- splitting)
//! - dvoice-representative dimensions
//! - Error cases (constant input, variable weight)

use nn_dsl::linear::build_linear;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Build a `TensorParamBinding::ConstantTensor` from a shape and constant value.
fn constant_weight(shape: &[usize], value: f32) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), value))
}

// ---------------------------------------------------------------------------
// Graph construction tests
// ---------------------------------------------------------------------------

#[test]
fn test_linear_no_bias_graph_builds() {
    // Linear: in=4, out=2, no bias
    let def = build_linear("linear_basic", 4, 2, false).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,  // data
        constant_weight(&[2, 4], 0.1), // weight [out, in]
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("basic linear graph should build");
    assert!(graph.num_nodes() >= 1, "graph should have at least 1 node");
}

#[test]
fn test_linear_with_bias_graph_builds() {
    // Linear: in=4, out=2, with bias
    let def = build_linear("linear_bias", 4, 2, true).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,  // data
        constant_weight(&[2, 4], 0.1), // weight [out, in]
        constant_weight(&[2], 0.0),    // bias [out]
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("linear+bias graph should build");
    assert!(graph.num_nodes() >= 1);
}

// ---------------------------------------------------------------------------
// Error case tests
// ---------------------------------------------------------------------------

#[test]
fn test_linear_param_count_mismatch_errors() {
    let def = build_linear("linear_test", 4, 2, false).expect("build");
    // Provide wrong number of bindings (3 instead of 2)
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[2, 4], 0.1),
        constant_weight(&[2], 0.0), // extra
    ];
    let result = tensor_kernel_to_graph(&def, &bindings);
    assert!(result.is_err(), "should error on param count mismatch");
}

#[test]
fn test_linear_constant_input_errors() {
    let def = build_linear("linear_test", 4, 2, false).expect("build");
    let bindings = vec![
        TensorParamBinding::ConstantScalar(1.0), // wrong: data should be Variable
        constant_weight(&[2, 4], 0.1),
    ];
    let result = tensor_kernel_to_graph(&def, &bindings);
    assert!(result.is_err(), "should error when data is ConstantScalar");
}

#[test]
fn test_linear_variable_weight_errors() {
    let def = build_linear("linear_test", 4, 2, false).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable, // wrong: weight should be ConstantTensor
    ];
    let result = tensor_kernel_to_graph(&def, &bindings);
    assert!(result.is_err(), "should error when weight is Variable");
}

// ---------------------------------------------------------------------------
// Numerical IBP bounds tests
// ---------------------------------------------------------------------------

/// Linear IBP: identity weight, no bias. Output bounds = input bounds.
#[test]
fn test_linear_ibp_identity_weight() {
    // in=2, out=2, weight=identity matrix, no bias
    let def = build_linear("linear_identity", 2, 2, false).expect("build");
    let weight = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("identity linear graph");

    let lower = ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -2.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through identity Linear");
    let (lo, hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[2]);
    // Identity: output bounds should match input bounds exactly.
    assert!((lo[0] - (-1.0)).abs() < 0.01, "lo[0]={}", lo[0]);
    assert!((lo[1] - (-2.0)).abs() < 0.01, "lo[1]={}", lo[1]);
    assert!((hi[0] - 1.0).abs() < 0.01, "hi[0]={}", hi[0]);
    assert!((hi[1] - 2.0).abs() < 0.01, "hi[1]={}", hi[1]);
}

/// Linear IBP with bias: bias shifts bounds by constant offset.
#[test]
fn test_linear_ibp_with_bias() {
    // in=2, out=2, weight=identity, bias=[10, -5]
    let def = build_linear("linear_bias_ibp", 2, 2, true).expect("build");
    let weight = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let bias = ArrayD::from_shape_vec(IxDyn(&[2]), vec![10.0, -5.0]).unwrap();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
        TensorParamBinding::ConstantTensor(bias),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("biased linear graph");

    let lower = ArrayD::from_elem(IxDyn(&[2]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through biased Linear");
    let (lo, hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[2]);
    // output[0] = x[0] + 10 => [10, 11]
    assert!((lo[0] - 10.0).abs() < 0.01, "lo[0]={}", lo[0]);
    assert!((hi[0] - 11.0).abs() < 0.01, "hi[0]={}", hi[0]);
    // output[1] = x[1] - 5 => [-5, -4]
    assert!((lo[1] - (-5.0)).abs() < 0.01, "lo[1]={}", lo[1]);
    assert!((hi[1] - (-4.0)).abs() < 0.01, "hi[1]={}", hi[1]);
}

/// Linear IBP with mixed-sign weights: W+/W- splitting.
///
/// Weight = [[1, -1], [-1, 1]], input in [-1, 1]:
///   lower[0] = 1*(-1) + (-1)*1 = -2
///   upper[0] = 1*1 + (-1)*(-1) = 2
#[test]
fn test_linear_ibp_mixed_sign_weights() {
    let def = build_linear("linear_mixed", 2, 2, false).expect("build");
    let weight = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, -1.0, -1.0, 1.0]).unwrap();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("mixed-weight linear graph");

    let lower = ArrayD::from_elem(IxDyn(&[2]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through mixed Linear");
    let (lo, hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[2]);
    // Both outputs: lower=-2, upper=2 (symmetric weight pattern).
    for i in 0..2 {
        assert!((lo[i] - (-2.0)).abs() < 0.05, "lo[{i}]={}", lo[i]);
        assert!((hi[i] - 2.0).abs() < 0.05, "hi[{i}]={}", hi[i]);
        let width = hi[i] - lo[i];
        assert!((width - 4.0).abs() < 0.1, "width[{i}]={width}");
    }
}

/// Linear IBP soundness: concrete forward pass falls within bounds.
#[test]
fn test_linear_ibp_soundness_concrete_forward() {
    // in=3, out=2, weight=[[0.3,-0.2,0.5],[0.1,0.4,-0.3]], no bias
    let def = build_linear("linear_sound", 3, 2, false).expect("build");
    let weight =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.3, -0.2, 0.5, 0.1, 0.4, -0.3]).unwrap();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("soundness linear graph");

    let lower = ArrayD::from_elem(IxDyn(&[3]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph.propagate_ibp(&input).expect("IBP soundness");
    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[2]);

    // Concrete forward: x = [0.5, -0.3, 0.8]
    // out[0] = 0.3*0.5 + (-0.2)*(-0.3) + 0.5*0.8 = 0.15 + 0.06 + 0.40 = 0.61
    // out[1] = 0.1*0.5 + 0.4*(-0.3) + (-0.3)*0.8 = 0.05 - 0.12 - 0.24 = -0.31
    let fwd_0 = 0.61f32;
    let fwd_1 = -0.31f32;

    // Soundness: forward pass must lie within IBP bounds.
    assert!(lo[0] <= fwd_0 + 0.01, "lo[0]={} <= fwd[0]={fwd_0}", lo[0]);
    assert!(hi[0] >= fwd_0 - 0.01, "hi[0]={} >= fwd[0]={fwd_0}", hi[0]);
    assert!(lo[1] <= fwd_1 + 0.01, "lo[1]={} <= fwd[1]={fwd_1}", lo[1]);
    assert!(hi[1] >= fwd_1 - 0.01, "hi[1]={} >= fwd[1]={fwd_1}", hi[1]);

    // Lower <= upper (sound bounds).
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite());
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

/// Linear IBP with dvoice-representative dimensions.
///
/// Qwen3 MLP: Linear(768, 3072) — typical FFN intermediate projection.
/// Uses small uniform weights to keep bounds tractable.
#[test]
fn test_linear_ibp_dvoice_dimensions() {
    let in_features = 768;
    let out_features = 3072;
    let def = build_linear("linear_qwen_mlp", in_features, out_features, true).expect("build");

    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[out_features, in_features], 0.001),
        constant_weight(&[out_features], 0.0),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("dvoice-dimension linear graph should build");

    let lower = ArrayD::from_elem(IxDyn(&[in_features]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[in_features]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through large Linear");
    let (lo, hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[out_features]);

    // All weights positive (0.001), input in [-1,1]:
    //   lower = 0.001 * (-768) + 0 = -0.768
    //   upper = 0.001 * 768 + 0 = 0.768
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite());
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
    let min_lo = lo.iter().copied().reduce(f32::min).unwrap();
    let max_hi = hi.iter().copied().reduce(f32::max).unwrap();
    assert!(
        min_lo >= -1.0,
        "lower >= -1.0 for small weights, got {min_lo}"
    );
    assert!(
        max_hi <= 1.0,
        "upper <= 1.0 for small weights, got {max_hi}"
    );
}

/// Linear with all-positive weights: bounds are tight.
#[test]
fn test_linear_ibp_positive_weights_tight() {
    // in=3, out=1, weight=[[0.3, 0.5, 0.2]], bias=[1.0]
    // Input in [0, 1]:
    //   lower = 0.3*0 + 0.5*0 + 0.2*0 + 1.0 = 1.0
    //   upper = 0.3*1 + 0.5*1 + 0.2*1 + 1.0 = 2.0
    let def = build_linear("linear_positive", 3, 1, true).expect("build");
    let weight = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![0.3, 0.5, 0.2]).unwrap();
    let bias = ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weight),
        TensorParamBinding::ConstantTensor(bias),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("positive-weight linear graph");

    let lower = ArrayD::from_elem(IxDyn(&[3]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through positive Linear");
    let (lo, hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[1]);
    assert!(
        (lo[0] - 1.0).abs() < 0.01,
        "lower should be ~1.0, got {}",
        lo[0]
    );
    assert!(
        (hi[0] - 2.0).abs() < 0.01,
        "upper should be ~2.0, got {}",
        hi[0]
    );
}
