// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Conv1d tensor IR → NY translation.
//!
//! Tests cover:
//! - Basic Conv1d graph construction (no bias)
//! - Conv1d with bias
//! - Conv1d with stride + padding (dvoice downsampling pattern)
//! - Pointwise (1x1) convolution
//! - IBP bound propagation through Conv1d

use nn_dsl::conv1d::build_conv1d;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Build a `TensorParamBinding::ConstantTensor` from a shape and constant value.
fn constant_weight(shape: &[usize], value: f32) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), value))
}

#[test]
fn test_conv1d_basic_graph_builds() {
    // Conv1d: [4, 8] @ [2, 4, 3] -> [2, 6], stride=1, padding=0, no bias
    let def = build_conv1d("conv1d_basic", 4, 2, 3, 8, 1, 0, false).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,     // data
        constant_weight(&[2, 4, 3], 0.1), // weight
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("basic conv1d graph should build");
    // Should have one Conv1d node
    assert!(graph.num_nodes() >= 1, "graph should have at least 1 node");
}

#[test]
fn test_conv1d_with_bias_graph_builds() {
    // Conv1d: [4, 8] @ [2, 4, 3] + bias[2] -> [2, 6]
    let def = build_conv1d("conv1d_bias", 4, 2, 3, 8, 1, 0, true).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,     // data
        constant_weight(&[2, 4, 3], 0.1), // weight
        constant_weight(&[2], 0.0),       // bias
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("conv1d with bias should build");
    assert!(graph.num_nodes() >= 1);
}

#[test]
fn test_conv1d_stride_padding_dvoice_pattern() {
    // dvoice Demucs encoder opening layer: conv1d(1, 48, 8, stride=4)
    // With input length 64 and padding 2:
    // out_len = (64 + 4 - 8) / 4 + 1 = 60/4 + 1 = 16
    let def = build_conv1d("conv1d_demucs", 1, 48, 8, 64, 4, 2, false).expect("build");
    assert_eq!(def.nodes.last().unwrap().shape, vec![48, 16]);

    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[48, 1, 8], 0.01),
    ];
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("dvoice pattern conv1d should build");
    assert!(graph.num_nodes() >= 1);
}

#[test]
fn test_conv1d_pointwise_1x1() {
    // 1x1 (pointwise) convolution: kernel_size=1, stride=1, padding=0
    let def = build_conv1d("conv1d_1x1", 48, 96, 1, 16, 1, 0, true).expect("build");
    // out_len = (16 + 0 - 1) / 1 + 1 = 16
    assert_eq!(def.nodes.last().unwrap().shape, vec![96, 16]);

    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[96, 48, 1], 0.02),
        constant_weight(&[96], 0.0),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("pointwise conv1d should build");
    assert!(graph.num_nodes() >= 1);
}

#[test]
fn test_conv1d_param_count_mismatch_errors() {
    let def = build_conv1d("conv1d_test", 4, 2, 3, 8, 1, 0, false).expect("build");
    // Provide wrong number of bindings (3 instead of 2)
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[2, 4, 3], 0.1),
        constant_weight(&[2], 0.0), // extra
    ];
    let result = tensor_kernel_to_graph(&def, &bindings);
    assert!(result.is_err(), "should error on param count mismatch");
}

#[test]
fn test_conv1d_non_variable_input_errors() {
    let def = build_conv1d("conv1d_test", 4, 2, 3, 8, 1, 0, false).expect("build");
    // Pass a constant tensor for data (should be Variable)
    let bindings = vec![
        TensorParamBinding::ConstantScalar(1.0), // wrong: data should be Variable
        constant_weight(&[2, 4, 3], 0.1),
    ];
    let result = tensor_kernel_to_graph(&def, &bindings);
    assert!(result.is_err(), "should error when data is not Variable");
}

#[test]
fn test_conv1d_non_constant_weight_errors() {
    let def = build_conv1d("conv1d_test", 4, 2, 3, 8, 1, 0, false).expect("build");
    // Pass Variable for weight (should be ConstantTensor)
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable, // wrong: weight should be ConstantTensor
    ];
    let result = tensor_kernel_to_graph(&def, &bindings);
    assert!(result.is_err(), "should error when weight is Variable");
}

// ---------------------------------------------------------------------------
// Numerical IBP bounds tests: verify bounds propagate correctly through
// Conv1d, not just that graph nodes are created. (#579)
// ---------------------------------------------------------------------------

/// Conv1d IBP propagation: input bounds [-1, 1] through a Conv1d with known
/// weights should produce finite, bounded output.
#[test]
fn test_conv1d_ibp_bounds_basic() {
    use nn_verify::BoundedTensor;

    // in_ch=2, out_ch=3, kernel=2, in_len=4, stride=1, pad=0 -> out_len=3
    let def = build_conv1d("conv1d_ibp", 2, 3, 2, 4, 1, 0, false).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[3, 2, 2], 0.5),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("conv1d graph");

    let lower = ArrayD::from_elem(IxDyn(&[2, 4]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 4]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph.propagate_ibp(&input).expect("IBP through Conv1d");
    let (lo, hi) = output.lower_upper();

    // Each output = sum of 2*2=4 products with weight=0.5, input in [-1,1].
    // Exact: lower=-2.0, upper=2.0. IBP may be looser.
    assert!(
        lo.iter().all(|&v| v.is_finite() && v >= -3.0),
        "lower finite and >= -3.0, got min={:.4}",
        lo.iter().copied().reduce(f32::min).unwrap()
    );
    assert!(
        hi.iter().all(|&v| v.is_finite() && v <= 3.0),
        "upper finite and <= 3.0, got max={:.4}",
        hi.iter().copied().reduce(f32::max).unwrap()
    );
    assert_eq!(lo.shape(), &[3, 3], "output shape [3, 3]");
}

/// Conv1d IBP with bias: verify bias shifts bounds correctly.
#[test]
fn test_conv1d_ibp_bounds_with_bias() {
    use nn_verify::BoundedTensor;

    // in_ch=1, out_ch=2, kernel=3, in_len=5, stride=1, pad=0, bias -> out_len=3
    let def = build_conv1d("conv1d_bias_ibp", 1, 2, 3, 5, 1, 0, true).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[2, 1, 3], 1.0),
        constant_weight(&[2], 10.0),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("conv1d+bias graph");

    let lower = ArrayD::from_elem(IxDyn(&[1, 5]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 5]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Conv1d+bias");
    let (lo, hi) = output.lower_upper();

    // output = sum(3 weights * input) + bias = [0..3] + 10.
    assert!(
        lo.iter().all(|&v| v >= 9.9),
        "lower >= ~10, got min={:.4}",
        lo.iter().copied().reduce(f32::min).unwrap()
    );
    assert!(
        hi.iter().all(|&v| v <= 13.1),
        "upper <= ~13, got max={:.4}",
        hi.iter().copied().reduce(f32::max).unwrap()
    );
    assert_eq!(lo.shape(), &[2, 3], "output shape [2, 3]");
}

/// Conv1d IBP soundness: verify a concrete forward pass falls within bounds.
///
/// Uses mixed-sign weights and manually computed forward values.
#[test]
fn test_conv1d_ibp_soundness_concrete_forward() {
    use nn_verify::BoundedTensor;

    // in_ch=2, out_ch=1, kernel=3, stride=1, pad=0. Output: [1, 2].
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 3]), vec![0.3, -0.2, 0.1, 0.4, 0.5, -0.1]).unwrap();

    let def = build_conv1d("conv1d_sound", 2, 1, 3, 4, 1, 0, false).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(kernel),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let lower = ArrayD::from_elem(IxDyn(&[2, 4]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 4]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph.propagate_ibp(&input).expect("IBP soundness");
    let (lo, hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[1, 2]);

    // Bounds must be finite and sound (lower <= upper).
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite());
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }

    // Manual forward for x=[[0.5,-0.3,0.8,-0.1],[0.2,0.7,-0.5,0.9]]:
    // pos 0: ic0=0.3*0.5+(-0.2)*(-0.3)+0.1*0.8=0.29, ic1=0.4*0.2+0.5*0.7+(-0.1)*(-0.5)=0.48
    //   total=0.77
    // pos 1: ic0=0.3*(-0.3)+(-0.2)*0.8+0.1*(-0.1)=-0.26, ic1=0.4*0.7+0.5*(-0.5)+(-0.1)*0.9=-0.06
    //   total=-0.32
    let fwd_0 = 0.77f32;
    let fwd_1 = -0.32f32;

    // Soundness: forward pass must lie within IBP bounds.
    assert!(lo[[0, 0]] <= fwd_0 + 0.01, "lo <= fwd[0]");
    assert!(hi[[0, 0]] >= fwd_0 - 0.01, "hi >= fwd[0]");
    assert!(lo[[0, 1]] <= fwd_1 + 0.01, "lo <= fwd[1]");
    assert!(hi[[0, 1]] >= fwd_1 - 0.01, "hi >= fwd[1]");
}

/// Conv1d IBP with stride + padding (dvoice Demucs encoder pattern).
///
/// Verifies that IBP bounds propagate correctly through a strided Conv1d
/// with padding, matching the dvoice Demucs encoder opening layer pattern.
#[test]
fn test_conv1d_stride_padding_ibp_dvoice() {
    use nn_verify::BoundedTensor;

    // Demucs encoder: conv1d(1, 48, kernel=8, stride=4, padding=2)
    // Smaller in_length=32 for test speed. out_len = (32 + 4 - 8) / 4 + 1 = 8.
    let def = build_conv1d("conv1d_demucs_ibp", 1, 48, 8, 32, 4, 2, false).expect("build");
    assert_eq!(def.nodes.last().unwrap().shape, vec![48, 8]);

    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[48, 1, 8], 0.01),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("dvoice conv1d graph");

    let lower = ArrayD::from_elem(IxDyn(&[1, 32]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 32]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through strided Conv1d");
    let (lo, hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[48, 8], "output shape must be [48, 8]");

    // All weights positive (0.01), W+ = W, W- = 0.
    // Bounds must be finite and sound (lower <= upper).
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite());
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
    // With all-positive weights of 0.01 and input in [-1,1], bounds are tight.
    assert!(
        lo.iter().all(|&v| v >= -0.1),
        "lower >= -0.1 for small positive weights"
    );
    assert!(
        hi.iter().all(|&v| v <= 0.1),
        "upper <= 0.1 for small positive weights"
    );
}

/// Conv1d IBP with pointwise (1x1) convolution: exact numerical bounds.
///
/// A 1x1 convolution is a linear map per spatial position. With known
/// weights [0.3, 0.7] and bias 0.5, input in [0,1] gives exact bounds.
#[test]
fn test_conv1d_pointwise_1x1_ibp_bounds() {
    use nn_verify::BoundedTensor;

    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 2, 1]), vec![0.3, 0.7]).unwrap();

    // in_ch=2, out_ch=1, kernel=1, in_len=4, stride=1, pad=0, bias
    let def = build_conv1d("conv1d_1x1_ibp", 2, 1, 1, 4, 1, 0, true).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(kernel),
        constant_weight(&[1], 0.5),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("1x1 conv1d graph");

    let lower = ArrayD::from_elem(IxDyn(&[2, 4]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 4]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph.propagate_ibp(&input).expect("IBP through 1x1 Conv1d");
    let (lo, hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[1, 4]);

    // All weights positive, input in [0,1]:
    //   lower = 0.3*0 + 0.7*0 + 0.5 = 0.5
    //   upper = 0.3*1 + 0.7*1 + 0.5 = 1.5
    for &v in lo.iter() {
        assert!((v - 0.5).abs() < 0.05, "lower should be ~0.5, got {v}");
    }
    for &v in hi.iter() {
        assert!((v - 1.5).abs() < 0.05, "upper should be ~1.5, got {v}");
    }
}

/// Conv1d IBP with mixed-sign weights: W+/W- splitting produces wider bounds.
///
/// Weight = [+1.0, -1.0], input in [-1,1]. IBP uses W+ and W- splitting:
///   lower_y = W+ * l + W- * u = 1*(-1) + (-1)*1 = -2
///   upper_y = W+ * u + W- * l = 1*1 + (-1)*(-1) = 2
/// Verifies the characteristic 4-wide interval from mixed-sign weights.
#[test]
fn test_conv1d_mixed_weights_ibp_wider_bounds() {
    use nn_verify::BoundedTensor;

    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![1.0, -1.0]).unwrap();

    // in_ch=1, out_ch=1, kernel=2, in_len=4, stride=1, pad=0 -> out_len=3
    let def = build_conv1d("conv1d_mixed", 1, 1, 2, 4, 1, 0, false).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(kernel),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("mixed-weight conv1d graph");

    let lower = ArrayD::from_elem(IxDyn(&[1, 4]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 4]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through mixed Conv1d");
    let (lo, hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[1, 3]);

    // IBP bounds: lower=-2, upper=2 for each output position.
    for &v in lo.iter() {
        assert!((v - (-2.0)).abs() < 0.05, "lower should be ~-2.0, got {v}");
    }
    for &v in hi.iter() {
        assert!((v - 2.0).abs() < 0.05, "upper should be ~2.0, got {v}");
    }
    // Interval width should be ~4.0 (characteristic of mixed-sign weights).
    for (l, u) in lo.iter().zip(hi.iter()) {
        let width = u - l;
        assert!(
            (width - 4.0).abs() < 0.1,
            "interval width should be ~4.0, got {width}"
        );
    }
}

// CROWN backward bounds tests extracted to graph_translate_conv1d_crown.rs (#579).
