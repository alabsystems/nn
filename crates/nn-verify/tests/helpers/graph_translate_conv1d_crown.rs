// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CROWN backward bounds tests for Conv1d tensor IR → NY translation.
//!
//! Extracted from `graph_translate_conv1d.rs` (file size limit).
//!
//! Tests verify Conv1d supports CROWN propagation through the nn-verify
//! pipeline. Conv1dLayer::with_input_length() sets the input_length required
//! for CROWN backward. (#579)

use nn_dsl::conv1d::build_conv1d;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Build a `TensorParamBinding::ConstantTensor` from a shape and constant value.
fn constant_weight(shape: &[usize], value: f32) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), value))
}

/// CROWN backward succeeds on a basic Conv1d graph.
///
/// For a pure linear layer (Conv1d has no nonlinearity), CROWN and IBP
/// produce identical bounds — the A-matrix is exactly the convolution weight
/// matrix and the bias contribution matches the constant bias. This test
/// verifies CROWN succeeds and produces sound, finite bounds.
#[test]
fn test_conv1d_crown_basic() {
    use nn_verify::{propagate_with_crown_fallback, BoundedTensor, PropMethod};

    // in_ch=2, out_ch=3, kernel=2, in_len=4, stride=1, pad=0 -> out_len=3
    let def = build_conv1d("conv1d_crown", 2, 3, 2, 4, 1, 0, false).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[3, 2, 2], 0.5),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("conv1d graph");

    let lower = ArrayD::from_elem(IxDyn(&[2, 4]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 4]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let (method, output, fallback_reason) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN through Conv1d");

    assert_eq!(
        method,
        PropMethod::Crown,
        "CROWN should succeed for Conv1d (linear layer)"
    );
    assert!(fallback_reason.is_none(), "no fallback when CROWN succeeds");

    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[3, 3], "output shape [3, 3]");

    // Bounds must be finite and sound (lower <= upper).
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite(), "bounds must be finite");
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

/// CROWN and IBP produce identical bounds for pure linear Conv1d.
///
/// Conv1d is a linear layer (no activation function). For linear layers,
/// CROWN backward produces exact bounds (the A-matrix captures the full
/// linear relationship). IBP also computes exact bounds via W+/W- splitting
/// for all-positive weights. Therefore CROWN == IBP within floating-point
/// tolerance.
#[test]
fn test_conv1d_crown_vs_ibp_identical_for_linear() {
    use nn_verify::{propagate_with_crown_fallback, BoundedTensor, PropMethod};

    // in_ch=2, out_ch=1, kernel=3, in_len=6, stride=1, pad=0 -> out_len=4
    // Use mixed-sign weights so IBP is not trivially tight.
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 3]), vec![0.3, -0.2, 0.1, 0.4, 0.5, -0.1]).unwrap();

    let def = build_conv1d("conv1d_crown_vs_ibp", 2, 1, 3, 6, 1, 0, false).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(kernel),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("conv1d graph");

    let lower = ArrayD::from_elem(IxDyn(&[2, 6]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 6]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    let (ibp_lo, ibp_hi) = ibp_output.lower_upper();

    // CROWN
    let (method, crown_output, _) = propagate_with_crown_fallback(&graph, &input).expect("CROWN");
    assert_eq!(method, PropMethod::Crown);
    let (crown_lo, crown_hi) = crown_output.lower_upper();

    assert_eq!(ibp_lo.shape(), crown_lo.shape(), "same output shape");

    // For a pure linear layer, CROWN and IBP should be identical.
    let tol = 1e-4;
    for ((&cl, &il), (&cu, &iu)) in crown_lo
        .iter()
        .zip(ibp_lo.iter())
        .zip(crown_hi.iter().zip(ibp_hi.iter()))
    {
        assert!(
            (cl - il).abs() < tol,
            "CROWN lower {cl} should match IBP lower {il}"
        );
        assert!(
            (cu - iu).abs() < tol,
            "CROWN upper {cu} should match IBP upper {iu}"
        );
    }
}

/// CROWN backward with bias: verify bias contribution propagates correctly.
///
/// With weight=1.0 (all positive) and bias=10.0, input in [0,1]:
///   lower = sum(weights * 0) + bias = 10.0
///   upper = sum(weights * 1) + bias = 13.0 (3 weights of 1.0 + bias 10.0)
/// CROWN should match IBP exactly for this linear layer.
#[test]
fn test_conv1d_crown_with_bias() {
    use nn_verify::{propagate_with_crown_fallback, BoundedTensor, PropMethod};

    // in_ch=1, out_ch=2, kernel=3, in_len=5, stride=1, pad=0, bias -> out_len=3
    let def = build_conv1d("conv1d_crown_bias", 1, 2, 3, 5, 1, 0, true).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[2, 1, 3], 1.0),
        constant_weight(&[2], 10.0),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("conv1d+bias graph");

    let lower = ArrayD::from_elem(IxDyn(&[1, 5]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 5]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    // CROWN
    let (method, crown_output, fallback_reason) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN+bias");
    assert_eq!(method, PropMethod::Crown);
    assert!(fallback_reason.is_none());

    let (lo, hi) = crown_output.lower_upper();
    assert_eq!(lo.shape(), &[2, 3]);

    // lower = 0 + 10 = 10, upper = 3 + 10 = 13 (for each output position)
    for &v in lo.iter() {
        assert!(
            (v - 10.0).abs() < 0.1,
            "CROWN lower should be ~10.0, got {v}"
        );
    }
    for &v in hi.iter() {
        assert!(
            (v - 13.0).abs() < 0.1,
            "CROWN upper should be ~13.0, got {v}"
        );
    }
}

/// CROWN backward soundness with concrete forward: concrete values must
/// lie within CROWN bounds.
///
/// Uses the same mixed-sign weights as the IBP soundness test, verifying
/// that CROWN bounds also contain the concrete forward pass result.
#[test]
fn test_conv1d_crown_soundness_concrete_forward() {
    use nn_verify::{propagate_with_crown_fallback, BoundedTensor, PropMethod};

    // in_ch=2, out_ch=1, kernel=3, stride=1, pad=0. Output: [1, 2].
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 3]), vec![0.3, -0.2, 0.1, 0.4, 0.5, -0.1]).unwrap();

    let def = build_conv1d("conv1d_crown_sound", 2, 1, 3, 4, 1, 0, false).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(kernel),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let lower = ArrayD::from_elem(IxDyn(&[2, 4]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 4]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let (method, crown_output, _) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN soundness");
    assert_eq!(method, PropMethod::Crown);

    let (lo, hi) = crown_output.lower_upper();
    assert_eq!(lo.shape(), &[1, 2]);

    // Bounds must be finite and sound (lower <= upper).
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite());
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }

    // Manual forward for x=[[0.5,-0.3,0.8,-0.1],[0.2,0.7,-0.5,0.9]]:
    //   pos 0: 0.77, pos 1: -0.32  (same as IBP soundness test)
    let fwd_0 = 0.77f32;
    let fwd_1 = -0.32f32;

    assert!(
        lo[[0, 0]] <= fwd_0 + 0.01,
        "CROWN lo <= fwd[0], got lo={}, fwd={}",
        lo[[0, 0]],
        fwd_0
    );
    assert!(
        hi[[0, 0]] >= fwd_0 - 0.01,
        "CROWN hi >= fwd[0], got hi={}, fwd={}",
        hi[[0, 0]],
        fwd_0
    );
    assert!(
        lo[[0, 1]] <= fwd_1 + 0.01,
        "CROWN lo <= fwd[1], got lo={}, fwd={}",
        lo[[0, 1]],
        fwd_1
    );
    assert!(
        hi[[0, 1]] >= fwd_1 - 0.01,
        "CROWN hi >= fwd[1], got hi={}, fwd={}",
        hi[[0, 1]],
        fwd_1
    );
}
