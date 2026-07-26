// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CROWN backward bounds tests for Conv2d tensor IR → NY translation.
//!
//! Conv2d uses `Conv2dLayer::with_input_shape()` which sets the input_shape
//! required for CROWN backward (conv2d_transpose for A-matrix propagation).
//!
//! Tests verify CROWN succeeds, produces finite/sound bounds, and matches IBP
//! for pure linear layers (Conv2d without activation).
//!
//! Pattern follows `graph_translate_conv1d_crown.rs`.
//! Part of #779, Re: R1-365 finding (0 CROWN tests for Conv2d).

use nn_dsl::conv2d::{build_conv2d, build_conv2d_full};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Build a `TensorParamBinding::ConstantTensor` from shape and constant value.
fn constant_weight(shape: &[usize], value: f32) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), value))
}

/// CROWN backward succeeds on a basic Conv2d graph (no bias).
///
/// For a pure linear layer (Conv2d without activation), CROWN and IBP produce
/// identical bounds — the A-matrix captures the full linear relationship via
/// conv2d_transpose. This test verifies CROWN succeeds and produces sound,
/// finite bounds.
#[test]
fn test_conv2d_crown_basic() {
    use nn_verify::{propagate_with_crown_fallback, BoundedTensor, PropMethod};

    // in_ch=2, out_ch=3, kernel=3×3, in=6×6, stride=1, pad=0 -> out=4×4
    let def = build_conv2d("conv2d_crown", 2, 3, 3, 3, 6, 6, 1, 1, 0, 0, false).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[3, 2, 3, 3], 0.1),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("conv2d graph");

    let lower = ArrayD::from_elem(IxDyn(&[2, 6, 6]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 6, 6]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let (method, output, fallback_reason) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN through Conv2d");

    assert_eq!(
        method,
        PropMethod::Crown,
        "CROWN should succeed for Conv2d (linear layer)"
    );
    assert!(fallback_reason.is_none(), "no fallback when CROWN succeeds");

    let (lo, hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[3, 4, 4],
        "output shape [out_ch=3, out_h=4, out_w=4]"
    );

    // Bounds must be finite and sound (lower <= upper).
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite(), "bounds must be finite");
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

/// CROWN and IBP produce identical bounds for pure linear Conv2d.
///
/// Conv2d is a linear layer (no activation). For linear layers, CROWN backward
/// produces exact bounds via conv2d_transpose. IBP computes bounds via W+/W-
/// splitting. Both are exact for linear maps, so they must match within
/// floating-point tolerance.
#[test]
fn test_conv2d_crown_vs_ibp_identical_for_linear() {
    use nn_verify::{propagate_with_crown_fallback, BoundedTensor, PropMethod};

    // in_ch=2, out_ch=1, kernel=3×3, in=5×5, stride=1, pad=0 -> out=3×3
    // Mixed-sign weights so IBP is not trivially tight.
    let kernel_data: Vec<f32> = vec![
        0.3, -0.2, 0.1, 0.4, 0.5, -0.1, 0.2, -0.3, 0.4, // in_ch=0
        -0.1, 0.2, 0.3, -0.4, 0.1, 0.5, -0.2, 0.3, -0.1, // in_ch=1
    ];
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 2, 3, 3]), kernel_data).unwrap();

    let def =
        build_conv2d("conv2d_crown_vs_ibp", 2, 1, 3, 3, 5, 5, 1, 1, 0, 0, false).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(kernel),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("conv2d graph");

    let lower = ArrayD::from_elem(IxDyn(&[2, 5, 5]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 5, 5]), 1.0f32);
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
/// With weight=1.0 (all positive) and bias=5.0, input in [0,1]:
///   Each output element sums 2*3*3=18 weight*input products + bias.
///   lower = 0 + 5.0 = 5.0
///   upper = 18*1.0 + 5.0 = 23.0
#[test]
fn test_conv2d_crown_with_bias() {
    use nn_verify::{propagate_with_crown_fallback, BoundedTensor, PropMethod};

    // in_ch=2, out_ch=1, kernel=3×3, in=5×5, stride=1, pad=0, bias -> out=3×3
    let def = build_conv2d("conv2d_crown_bias", 2, 1, 3, 3, 5, 5, 1, 1, 0, 0, true).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[1, 2, 3, 3], 1.0),
        constant_weight(&[1], 5.0),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("conv2d+bias graph");

    let lower = ArrayD::from_elem(IxDyn(&[2, 5, 5]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 5, 5]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    // CROWN
    let (method, crown_output, fallback_reason) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN+bias");
    assert_eq!(method, PropMethod::Crown);
    assert!(fallback_reason.is_none());

    let (lo, hi) = crown_output.lower_upper();
    assert_eq!(lo.shape(), &[1, 3, 3]);

    // lower = 0 + 5 = 5, upper = 18 + 5 = 23
    for &v in lo.iter() {
        assert!((v - 5.0).abs() < 0.5, "CROWN lower should be ~5.0, got {v}");
    }
    for &v in hi.iter() {
        assert!(
            (v - 23.0).abs() < 0.5,
            "CROWN upper should be ~23.0, got {v}"
        );
    }
}

/// CROWN backward with same-padding (Demucs spectral decoder pattern).
///
/// 3×3 kernel, stride=1, pad=1 preserves spatial dimensions.
/// Verifies CROWN succeeds for the exact Conv2d configuration used by
/// the Demucs spectral decoder.
#[test]
fn test_conv2d_crown_demucs_same_padding() {
    use nn_verify::{propagate_with_crown_fallback, BoundedTensor, PropMethod};

    // in_ch=4, out_ch=8, kernel=3×3, in=8×8, stride=1, pad=1 -> out=8×8
    let def =
        build_conv2d("conv2d_crown_demucs", 4, 8, 3, 3, 8, 8, 1, 1, 1, 1, false).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[8, 4, 3, 3], 0.05),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("conv2d demucs graph");

    let lower = ArrayD::from_elem(IxDyn(&[4, 8, 8]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4, 8, 8]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let (method, output, fallback_reason) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN Demucs pattern");

    assert_eq!(
        method,
        PropMethod::Crown,
        "CROWN should succeed for Demucs Conv2d"
    );
    assert!(fallback_reason.is_none());

    let (lo, hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[8, 8, 8],
        "same-padding output [out_ch=8, 8, 8]"
    );

    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite(), "bounds must be finite");
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

/// CROWN backward with dilation: dilated 3×3 kernel on 10×10 input.
///
/// Dilation expands the effective kernel to 5×5. Verifies CROWN works
/// through the expanded kernel via `expand_dilated_kernel_2d`.
#[test]
fn test_conv2d_crown_dilated() {
    use nn_verify::{propagate_with_crown_fallback, BoundedTensor, PropMethod};

    // in_ch=2, out_ch=2, kernel=3×3, in=10×10, stride=1, pad=0, dilation=2
    // effective kernel: 5×5. out=6×6.
    let def = build_conv2d_full(
        "conv2d_crown_dilated",
        2,
        2,
        3,
        3,
        10,
        10,
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
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[2, 2, 3, 3], 0.2),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("dilated conv2d graph");

    let lower = ArrayD::from_elem(IxDyn(&[2, 10, 10]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 10, 10]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let (method, output, fallback_reason) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN dilated Conv2d");

    assert_eq!(
        method,
        PropMethod::Crown,
        "CROWN should succeed for dilated Conv2d"
    );
    assert!(fallback_reason.is_none());

    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[2, 6, 6], "dilated output [2, 6, 6]");

    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite(), "bounds must be finite");
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}
