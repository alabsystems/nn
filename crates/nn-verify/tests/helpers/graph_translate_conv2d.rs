// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Conv2d tensor IR → NY translation.
//!
//! Tests cover:
//! - Basic Conv2d graph construction (no bias)
//! - Conv2d with bias
//! - Conv2d with stride + padding (Demucs spectral decoder pattern)
//! - Pointwise (1×1) convolution
//! - Asymmetric kernel (1×3)
//! - Error cases (param mismatch, wrong binding types)
//! - IBP bound propagation through Conv2d

use nn_dsl::conv2d::{build_conv2d, build_conv2d_full};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Build a `TensorParamBinding::ConstantTensor` from a shape and constant value.
fn constant_weight(shape: &[usize], value: f32) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), value))
}

// ===========================================================================
// Graph construction tests
// ===========================================================================

#[test]
fn test_conv2d_basic_graph_builds() {
    // Conv2d: [4, 8, 8] @ [2, 4, 3, 3] -> [2, 6, 6], stride=1, pad=0, no bias
    let def = build_conv2d("conv2d_basic", 4, 2, 3, 3, 8, 8, 1, 1, 0, 0, false).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,        // data [4, 8, 8]
        constant_weight(&[2, 4, 3, 3], 0.1), // weight
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("basic conv2d graph should build");
    assert!(graph.num_nodes() >= 1, "graph should have at least 1 node");
}

#[test]
fn test_conv2d_with_bias_graph_builds() {
    // Conv2d: [4, 8, 8] @ [2, 4, 3, 3] + bias[2] -> [2, 6, 6]
    let def = build_conv2d("conv2d_bias", 4, 2, 3, 3, 8, 8, 1, 1, 0, 0, true).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[2, 4, 3, 3], 0.1),
        constant_weight(&[2], 0.0), // bias
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("conv2d with bias should build");
    assert!(graph.num_nodes() >= 1);
}

#[test]
fn test_conv2d_stride_padding_demucs_pattern() {
    // Demucs spectral decoder: 3×3 kernel, stride=1, pad=1 (same padding)
    // in: [48, 16, 16] -> out: [96, 16, 16]
    let def =
        build_conv2d("conv2d_demucs", 48, 96, 3, 3, 16, 16, 1, 1, 1, 1, false).expect("build");
    assert_eq!(def.nodes.last().unwrap().shape, vec![96, 16, 16]);

    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[96, 48, 3, 3], 0.01),
    ];
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("demucs spectral conv2d should build");
    assert!(graph.num_nodes() >= 1);
}

#[test]
fn test_conv2d_pointwise_1x1() {
    // 1×1 (pointwise) convolution: kernel=1×1, stride=1, pad=0
    let def = build_conv2d("conv2d_1x1", 48, 96, 1, 1, 16, 16, 1, 1, 0, 0, true).expect("build");
    assert_eq!(def.nodes.last().unwrap().shape, vec![96, 16, 16]);

    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[96, 48, 1, 1], 0.02),
        constant_weight(&[96], 0.0),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("pointwise conv2d should build");
    assert!(graph.num_nodes() >= 1);
}

#[test]
fn test_conv2d_asymmetric_kernel_1x3() {
    // 1×3 kernel (width-only convolution)
    let def = build_conv2d("conv2d_1x3", 4, 8, 1, 3, 10, 10, 1, 1, 0, 0, false).expect("build");
    // out_h = (10 - 1)/1 + 1 = 10, out_w = (10 - 3)/1 + 1 = 8
    assert_eq!(def.nodes.last().unwrap().shape, vec![8, 10, 8]);

    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[8, 4, 1, 3], 0.1),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("asymmetric conv2d should build");
    assert!(graph.num_nodes() >= 1);
}

#[test]
fn test_conv2d_dilation() {
    // dilation=2, kernel=3×3: effective kernel = 5×5
    // out_h = (8 - 5)/1 + 1 = 4, out_w = (8 - 5)/1 + 1 = 4
    let def = build_conv2d_full(
        "conv2d_dilated",
        4,
        2,
        3,
        3,
        8,
        8,
        1,
        1,
        0,
        0,
        2,
        2,
        1,
        false,
    )
    .expect("build");
    assert_eq!(def.nodes.last().unwrap().shape, vec![2, 4, 4]);

    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[2, 4, 3, 3], 0.1),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("dilated conv2d should build");
    assert!(graph.num_nodes() >= 1);
}

// ===========================================================================
// Error case tests
// ===========================================================================

#[test]
fn test_conv2d_param_count_mismatch_errors() {
    let def = build_conv2d("conv2d_test", 4, 2, 3, 3, 8, 8, 1, 1, 0, 0, false).expect("build");
    // Provide wrong number of bindings (3 instead of 2)
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[2, 4, 3, 3], 0.1),
        constant_weight(&[2], 0.0), // extra
    ];
    let result = tensor_kernel_to_graph(&def, &bindings);
    assert!(result.is_err(), "should error on param count mismatch");
}

#[test]
fn test_conv2d_non_variable_input_errors() {
    let def = build_conv2d("conv2d_test", 4, 2, 3, 3, 8, 8, 1, 1, 0, 0, false).expect("build");
    let bindings = vec![
        TensorParamBinding::ConstantScalar(1.0), // wrong: data should be Variable
        constant_weight(&[2, 4, 3, 3], 0.1),
    ];
    let result = tensor_kernel_to_graph(&def, &bindings);
    assert!(result.is_err(), "should error when data is not Variable");
}

#[test]
fn test_conv2d_non_constant_weight_errors() {
    let def = build_conv2d("conv2d_test", 4, 2, 3, 3, 8, 8, 1, 1, 0, 0, false).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable, // wrong: weight should be ConstantTensor
    ];
    let result = tensor_kernel_to_graph(&def, &bindings);
    assert!(result.is_err(), "should error when weight is Variable");
}

// ===========================================================================
// IBP bound propagation tests
// ===========================================================================

/// Conv2d IBP propagation: input bounds [-1, 1] through Conv2d with known
/// weights should produce finite, bounded output.
#[test]
fn test_conv2d_ibp_bounds_basic() {
    // in_ch=2, out_ch=3, kernel=2×2, in=4×4, stride=1, pad=0 -> out=3×3
    let def = build_conv2d("conv2d_ibp", 2, 3, 2, 2, 4, 4, 1, 1, 0, 0, false).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[3, 2, 2, 2], 0.5), // 8 weights per output, all 0.5
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("conv2d graph");

    let lower = ArrayD::from_elem(IxDyn(&[2, 4, 4]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 4, 4]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph.propagate_ibp(&input).expect("IBP through Conv2d");
    let (lo, hi) = output.lower_upper();

    // Each output = sum of 2*2*2=8 products with weight=0.5, input in [-1,1].
    // Exact: lower=-4.0, upper=4.0. IBP may be looser.
    assert!(
        lo.iter().all(|&v| v.is_finite() && v >= -5.0),
        "lower finite and >= -5.0, got min={:.4}",
        lo.iter().copied().reduce(f32::min).unwrap()
    );
    assert!(
        hi.iter().all(|&v| v.is_finite() && v <= 5.0),
        "upper finite and <= 5.0, got max={:.4}",
        hi.iter().copied().reduce(f32::max).unwrap()
    );
    assert_eq!(lo.shape(), &[3, 3, 3], "output shape [3, 3, 3]");
}

/// Conv2d IBP with bias: verify bias shifts bounds correctly.
#[test]
fn test_conv2d_ibp_bounds_with_bias() {
    // in_ch=1, out_ch=2, kernel=3×3, in=5×5, stride=1, pad=0, bias -> out=3×3
    let def = build_conv2d("conv2d_bias_ibp", 1, 2, 3, 3, 5, 5, 1, 1, 0, 0, true).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[2, 1, 3, 3], 1.0), // weight: all 1.0
        constant_weight(&[2], 10.0),         // bias: 10.0
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("conv2d+bias graph");

    let lower = ArrayD::from_elem(IxDyn(&[1, 5, 5]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 5, 5]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Conv2d+bias");
    let (lo, hi) = output.lower_upper();

    // output = sum(1*1*3*3=9 weights * input) + bias = [0..9] + 10.
    // lower = 0 + 10 = 10, upper = 9 + 10 = 19.
    assert!(
        lo.iter().all(|&v| v >= 9.9),
        "lower >= ~10, got min={:.4}",
        lo.iter().copied().reduce(f32::min).unwrap()
    );
    assert!(
        hi.iter().all(|&v| v <= 19.1),
        "upper <= ~19, got max={:.4}",
        hi.iter().copied().reduce(f32::max).unwrap()
    );
    assert_eq!(lo.shape(), &[2, 3, 3], "output shape [2, 3, 3]");
}

/// Conv2d IBP soundness check: verify bounds contain the [0, 0] extreme.
#[test]
fn test_conv2d_ibp_soundness_bounds_contain_zero() {
    // If input is uniformly [0, 0], the output should be within proved bounds.
    let def = build_conv2d("conv2d_sound", 2, 2, 3, 3, 6, 6, 1, 1, 0, 0, false).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[2, 2, 3, 3], 0.3),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("conv2d graph");

    // Proved bounds for input in [-1, 1].
    let lower = ArrayD::from_elem(IxDyn(&[2, 6, 6]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 6, 6]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");
    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, hi) = output.lower_upper();

    // Zero input should produce zero output (no bias). Zero must be within bounds.
    for v in lo.iter() {
        assert!(*v <= 0.001, "lower bound should be <= 0 (no bias), got {v}");
    }
    for v in hi.iter() {
        assert!(
            *v >= -0.001,
            "upper bound should be >= 0 (no bias), got {v}"
        );
    }
}

/// Conv2d IBP with stride=2: downsampling pattern.
#[test]
fn test_conv2d_ibp_stride_padding() {
    // Downsampling: stride=2, kernel=3, pad=1 -> out = (8+2-3)/2+1 = 4
    let def = build_conv2d("conv2d_ds_ibp", 2, 4, 3, 3, 8, 8, 2, 2, 1, 1, false).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[4, 2, 3, 3], 0.1),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("conv2d graph");

    let lower = ArrayD::from_elem(IxDyn(&[2, 8, 8]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 8, 8]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Conv2d stride=2");
    let (lo, hi) = output.lower_upper();

    assert!(lo.iter().all(|v| v.is_finite()), "lower must be finite");
    assert!(hi.iter().all(|v| v.is_finite()), "upper must be finite");
    assert_eq!(lo.shape(), &[4, 4, 4], "output shape [4, 4, 4]");
    // Soundness: lo <= hi everywhere.
    for (l, h) in lo.iter().zip(hi.iter()) {
        assert!(l <= h, "lo ({l}) must be <= hi ({h})");
    }
}

/// Conv2d IBP with 1×1 pointwise: exact bounds check.
#[test]
fn test_conv2d_ibp_pointwise_exact() {
    // 1×1 conv with weight 0.5: output = 0.5 * input per channel.
    // 1 input channel, 1 output channel. output[h,w] = 0.5 * input[h,w].
    // Input in [-1, 1] -> output in [-0.5, 0.5].
    let def = build_conv2d("conv2d_1x1_ibp", 1, 1, 1, 1, 4, 4, 1, 1, 0, 0, false).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[1, 1, 1, 1], 0.5),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("conv2d graph");

    let lower = ArrayD::from_elem(IxDyn(&[1, 4, 4]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph.propagate_ibp(&input).expect("IBP through 1×1 Conv2d");
    let (lo, hi) = output.lower_upper();

    // Exact IBP for a single linear layer: output = w * input.
    // w=0.5, input in [-1, 1] → output in [-0.5, 0.5].
    let tol = 1e-4;
    for v in lo.iter() {
        assert!((*v - (-0.5)).abs() < tol, "expected lower ≈ -0.5, got {v}");
    }
    for v in hi.iter() {
        assert!((*v - 0.5).abs() < tol, "expected upper ≈ 0.5, got {v}");
    }
}
