// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for ConvTranspose1d tensor IR → NY translation.
//!
//! Tests cover:
//! - Basic ConvTranspose1d graph construction (no bias)
//! - ConvTranspose1d with bias
//! - ConvTranspose1d with stride (dvoice Demucs decoder upsampling pattern)
//! - Pointwise (1x1) transposed convolution
//! - IBP bound propagation through ConvTranspose1d
//! - CROWN backward bounds through ConvTranspose1d

use nn_dsl::conv_transpose_1d::build_conv_transpose_1d;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Build a `TensorParamBinding::ConstantTensor` from a shape and constant value.
fn constant_weight(shape: &[usize], value: f32) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), value))
}

// ---------------------------------------------------------------------------
// Graph construction tests
// ---------------------------------------------------------------------------

#[test]
fn test_conv_transpose_1d_basic_graph_builds() {
    // ConvTranspose1d: [4, 8] @ [4, 2, 3] -> [2, 10], stride=1, padding=0, no bias
    // out_len = (8 - 1) * 1 + 3 - 0 = 10
    let def =
        build_conv_transpose_1d("ct1d_basic", 4, 2, 3, 8, 1, 0, 1, 1, false, 0).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,     // data
        constant_weight(&[4, 2, 3], 0.1), // weight [in_ch, out_ch, kernel]
    ];
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("basic ConvTranspose1d graph should build");
    assert!(graph.num_nodes() >= 1, "graph should have at least 1 node");
}

#[test]
fn test_conv_transpose_1d_with_bias_graph_builds() {
    // ConvTranspose1d: [4, 8] @ [4, 2, 3] + bias[2] -> [2, 10]
    let def = build_conv_transpose_1d("ct1d_bias", 4, 2, 3, 8, 1, 0, 1, 1, true, 0).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,     // data
        constant_weight(&[4, 2, 3], 0.1), // weight
        constant_weight(&[2], 0.0),       // bias
    ];
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("ConvTranspose1d with bias should build");
    assert!(graph.num_nodes() >= 1);
}

#[test]
fn test_conv_transpose_1d_stride_dvoice_decoder_pattern() {
    // dvoice Demucs decoder upsampling: ConvTranspose1d(48, 1, 8, stride=4)
    // With input length 16 and padding 2:
    // out_len = (16 - 1) * 4 + 8 - 4 = 60 + 4 = 64
    let def =
        build_conv_transpose_1d("ct1d_demucs", 48, 1, 8, 16, 4, 2, 1, 1, false, 0).expect("build");
    assert_eq!(def.nodes.last().unwrap().shape, vec![1, 64]);

    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[48, 1, 8], 0.01), // weight [in_ch, out_ch, kernel]
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("dvoice decoder ConvTranspose1d should build");
    assert!(graph.num_nodes() >= 1);
}

#[test]
fn test_conv_transpose_1d_pointwise_1x1() {
    // 1x1 (pointwise) transposed convolution: kernel_size=1, stride=1, padding=0
    // out_len = (16 - 1) * 1 + 1 - 0 = 16
    let def =
        build_conv_transpose_1d("ct1d_1x1", 96, 48, 1, 16, 1, 0, 1, 1, true, 0).expect("build");
    assert_eq!(def.nodes.last().unwrap().shape, vec![48, 16]);

    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[96, 48, 1], 0.02), // weight
        constant_weight(&[48], 0.0),         // bias
    ];
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("pointwise ConvTranspose1d should build");
    assert!(graph.num_nodes() >= 1);
}

// ---------------------------------------------------------------------------
// Error handling tests
// ---------------------------------------------------------------------------

#[test]
fn test_conv_transpose_1d_param_count_mismatch_errors() {
    let def = build_conv_transpose_1d("ct1d_err", 4, 2, 3, 8, 1, 0, 1, 1, false, 0).expect("build");
    // Provide wrong number of bindings (3 instead of 2)
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[4, 2, 3], 0.1),
        constant_weight(&[2], 0.0), // extra
    ];
    let result = tensor_kernel_to_graph(&def, &bindings);
    assert!(result.is_err(), "should error on param count mismatch");
}

#[test]
fn test_conv_transpose_1d_non_variable_input_errors() {
    let def = build_conv_transpose_1d("ct1d_err", 4, 2, 3, 8, 1, 0, 1, 1, false, 0).expect("build");
    // Pass a constant scalar for data (should be Variable)
    let bindings = vec![
        TensorParamBinding::ConstantScalar(1.0), // wrong
        constant_weight(&[4, 2, 3], 0.1),
    ];
    let result = tensor_kernel_to_graph(&def, &bindings);
    assert!(result.is_err(), "should error when data is not Variable");
}

#[test]
fn test_conv_transpose_1d_non_constant_weight_errors() {
    let def = build_conv_transpose_1d("ct1d_err", 4, 2, 3, 8, 1, 0, 1, 1, false, 0).expect("build");
    // Pass Variable for weight (should be ConstantTensor)
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable, // wrong
    ];
    let result = tensor_kernel_to_graph(&def, &bindings);
    assert!(result.is_err(), "should error when weight is Variable");
}

// ---------------------------------------------------------------------------
// Numerical IBP bounds tests
// ---------------------------------------------------------------------------

/// ConvTranspose1d IBP propagation: input bounds [-1, 1] through a
/// ConvTranspose1d with known weights should produce finite, bounded output.
#[test]
fn test_conv_transpose_1d_ibp_bounds_basic() {
    use nn_verify::BoundedTensor;

    // in_ch=2, out_ch=3, kernel=2, in_len=4, stride=1, pad=0
    // out_len = (4 - 1) * 1 + 2 - 0 = 5
    let def = build_conv_transpose_1d("ct1d_ibp", 2, 3, 2, 4, 1, 0, 1, 1, false, 0).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[2, 3, 2], 0.5), // weight [in_ch, out_ch, kernel]
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("ConvTranspose1d graph");

    let lower = ArrayD::from_elem(IxDyn(&[2, 4]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 4]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through ConvTranspose1d");
    let (lo, hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[3, 5], "output shape [3, 5]");

    // Bounds must be finite and sound (lower <= upper).
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite(), "bounds must be finite");
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

/// ConvTranspose1d IBP with bias: verify bias shifts bounds correctly.
#[test]
fn test_conv_transpose_1d_ibp_bounds_with_bias() {
    use nn_verify::BoundedTensor;

    // in_ch=1, out_ch=2, kernel=3, in_len=5, stride=1, pad=0, bias
    // out_len = (5 - 1) * 1 + 3 - 0 = 7
    let def =
        build_conv_transpose_1d("ct1d_bias_ibp", 1, 2, 3, 5, 1, 0, 1, 1, true, 0).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[1, 2, 3], 1.0), // weight
        constant_weight(&[2], 10.0),      // bias
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("ConvTranspose1d+bias graph");

    let lower = ArrayD::from_elem(IxDyn(&[1, 5]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 5]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through ConvTranspose1d+bias");
    let (lo, hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[2, 7], "output shape [2, 7]");

    // With bias 10.0, lower bounds should be >= 10.0 (no negative contributions).
    assert!(
        lo.iter().all(|&v| v >= 9.9),
        "lower >= ~10 with bias, got min={:.4}",
        lo.iter().copied().reduce(f32::min).unwrap()
    );
    // All bounds should be finite and sound.
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite());
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

/// ConvTranspose1d IBP with stride (dvoice Demucs decoder upsampling).
#[test]
fn test_conv_transpose_1d_stride_ibp_dvoice() {
    use nn_verify::BoundedTensor;

    // Demucs decoder: ConvTranspose1d(48, 1, kernel=8, stride=4, padding=2)
    // Smaller in_length=8 for test speed.
    // out_len = (8 - 1) * 4 + 8 - 4 = 32
    let def = build_conv_transpose_1d("ct1d_demucs_ibp", 48, 1, 8, 8, 4, 2, 1, 1, false, 0)
        .expect("build");
    assert_eq!(def.nodes.last().unwrap().shape, vec![1, 32]);

    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[48, 1, 8], 0.01), // weight
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("dvoice ConvTranspose1d graph");

    let lower = ArrayD::from_elem(IxDyn(&[48, 8]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[48, 8]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through strided ConvTranspose1d");
    let (lo, hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[1, 32], "output shape must be [1, 32]");

    // Bounds must be finite and sound.
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite());
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

/// ConvTranspose1d IBP with pointwise (1x1): exact numerical bounds.
#[test]
fn test_conv_transpose_1d_pointwise_1x1_ibp_bounds() {
    use nn_verify::BoundedTensor;

    let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 1, 1]), vec![0.3, 0.7]).unwrap();

    // in_ch=2, out_ch=1, kernel=1, in_len=4, stride=1, pad=0, bias
    // out_len = (4 - 1) * 1 + 1 - 0 = 4
    let def =
        build_conv_transpose_1d("ct1d_1x1_ibp", 2, 1, 1, 4, 1, 0, 1, 1, true, 0).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(kernel),
        constant_weight(&[1], 0.5), // bias
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("1x1 ConvTranspose1d graph");

    let lower = ArrayD::from_elem(IxDyn(&[2, 4]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 4]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 1x1 ConvTranspose1d");
    let (lo, hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[1, 4]);

    // Bounds must be finite and sound.
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite());
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

// CROWN backward bounds + soundness guard tests extracted to
// graph_translate_conv_transpose_1d_crown.rs (#1567).
