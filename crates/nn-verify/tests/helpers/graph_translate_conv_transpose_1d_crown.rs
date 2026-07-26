// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CROWN backward bounds tests for ConvTranspose1d (#1567).
//!
//! Extracted from graph_translate_conv_transpose_1d.rs for file-size compliance.
//! Tests cover CROWN propagation and soundness guard (dilation/groups rejection).

use nn_dsl::conv_transpose_1d::build_conv_transpose_1d;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Build a `TensorParamBinding::ConstantTensor` from a shape and constant value.
fn constant_weight(shape: &[usize], value: f32) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), value))
}

// ---------------------------------------------------------------------------
// CROWN backward bounds tests
// ---------------------------------------------------------------------------

/// CROWN backward succeeds on a basic ConvTranspose1d graph.
///
/// ConvTranspose1d is a linear layer (no nonlinearity). CROWN should succeed
/// and produce sound, finite bounds.
#[test]
fn test_conv_transpose_1d_crown_basic() {
    use nn_verify::{propagate_with_crown_fallback, BoundedTensor, PropMethod};

    // in_ch=2, out_ch=3, kernel=2, in_len=4, stride=1, pad=0 -> out_len=5
    let def =
        build_conv_transpose_1d("ct1d_crown", 2, 3, 2, 4, 1, 0, 1, 1, false, 0).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[2, 3, 2], 0.5), // weight
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("ConvTranspose1d graph");

    let lower = ArrayD::from_elem(IxDyn(&[2, 4]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 4]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let (method, output, fallback_reason) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN through ConvTranspose1d");

    assert_eq!(
        method,
        PropMethod::Crown,
        "CROWN should succeed for ConvTranspose1d (linear layer)"
    );
    assert!(fallback_reason.is_none(), "no fallback when CROWN succeeds");

    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[3, 5], "output shape [3, 5]");

    // Bounds must be finite and sound.
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite(), "bounds must be finite");
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

/// CROWN and IBP produce identical bounds for pure linear ConvTranspose1d.
#[test]
fn test_conv_transpose_1d_crown_vs_ibp_identical_for_linear() {
    use nn_verify::{propagate_with_crown_fallback, BoundedTensor, PropMethod};

    // in_ch=2, out_ch=1, kernel=3, in_len=6, stride=1, pad=0
    // out_len = (6 - 1) * 1 + 3 - 0 = 8
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![0.3, -0.2, 0.1, 0.4, 0.5, -0.1]).unwrap();

    let def = build_conv_transpose_1d("ct1d_crown_vs_ibp", 2, 1, 3, 6, 1, 0, 1, 1, false, 0)
        .expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(kernel),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("ConvTranspose1d graph");

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
#[test]
fn test_conv_transpose_1d_crown_with_bias() {
    use nn_verify::{propagate_with_crown_fallback, BoundedTensor, PropMethod};

    // in_ch=1, out_ch=2, kernel=3, in_len=5, stride=1, pad=0, bias
    // out_len = (5 - 1) * 1 + 3 - 0 = 7
    let def =
        build_conv_transpose_1d("ct1d_crown_bias", 1, 2, 3, 5, 1, 0, 1, 1, true, 0).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[1, 2, 3], 1.0), // weight
        constant_weight(&[2], 10.0),      // bias
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("ConvTranspose1d+bias graph");

    let lower = ArrayD::from_elem(IxDyn(&[1, 5]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 5]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let (method, crown_output, fallback_reason) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN+bias");
    assert_eq!(method, PropMethod::Crown);
    assert!(fallback_reason.is_none());

    let (lo, hi) = crown_output.lower_upper();
    assert_eq!(lo.shape(), &[2, 7]);

    // Lower bounds should include bias (>= 10.0 for non-negative weights).
    assert!(
        lo.iter().all(|&v| v >= 9.9),
        "CROWN lower should be >= ~10 with bias, got min={:.4}",
        lo.iter().copied().reduce(f32::min).unwrap()
    );

    // Bounds must be finite and sound.
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite());
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

// ---------------------------------------------------------------------------
// Dilation and groups support (#2989): NY ConvTranspose1dLayer
// supports both via with_input_length_full(). nn-verify passes them through.
// ---------------------------------------------------------------------------

#[test]
fn test_conv_transpose_1d_dilation_produces_valid_bounds() {
    use nn_verify::{propagate_with_crown_fallback, BoundedTensor, PropMethod};

    // in_ch=4, out_ch=2, kernel=3, in_len=8, stride=1, pad=0, dilation=2, groups=1
    let def = build_conv_transpose_1d("ct1d_dil", 4, 2, 3, 8, 1, 0, 2, 1, false, 0).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[4, 2, 3], 0.1),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("ConvTranspose1d dilation=2 graph");

    let lower = ArrayD::from_elem(IxDyn(&[4, 8]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4, 8]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let (method, output, _) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN with dilation=2");
    assert_eq!(method, PropMethod::Crown);

    let (lo, hi) = output.lower_upper();
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite(), "bounds must be finite");
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

#[test]
fn test_conv_transpose_1d_groups_produces_valid_bounds() {
    use nn_verify::{propagate_with_crown_fallback, BoundedTensor, PropMethod};

    // in_ch=4, out_ch_per_group=1, kernel=3, in_len=8, stride=1, pad=0, dilation=1, groups=2
    // Total out_ch = out_ch_per_group * groups = 1 * 2 = 2
    let def = build_conv_transpose_1d("ct1d_grp", 4, 2, 3, 8, 1, 0, 1, 2, false, 0).expect("build");
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[4, 1, 3], 0.1), // [in=4, out_per_group=1, kernel=3]
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("ConvTranspose1d groups=2 graph");

    let lower = ArrayD::from_elem(IxDyn(&[4, 8]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4, 8]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let (method, output, _) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN with groups=2");
    assert_eq!(method, PropMethod::Crown);

    let (lo, hi) = output.lower_upper();
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite(), "bounds must be finite");
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

/// Depthwise ConvTranspose1d matching Kokoro F0EnergyPredictor's upsample pool.
///
/// groups=dim_in (depthwise), stride=2, pad=1, kernel=3. Part of #2989.
#[test]
fn test_conv_transpose_1d_depthwise_kokoro_f0_pattern() {
    use nn_verify::{propagate_with_crown_fallback, BoundedTensor, PropMethod};

    let dim_in = 8; // miniaturized (production: 256)
                    // Depthwise: in_ch=dim_in, out_ch_per_group=1, groups=dim_in
                    // Weight: [in_ch, 1, kernel=3]
    let def = build_conv_transpose_1d(
        "ct1d_dw", dim_in, dim_in, // in_ch, out_ch (= 1 * groups)
        3,      // kernel
        16,     // in_len
        2,      // stride
        1,      // padding
        1,      // dilation
        dim_in, // groups = depthwise
        true,   // bias
        0,      // output_padding
    )
    .expect("build depthwise conv transpose");

    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[dim_in, 1, 3], 0.25), // [in_ch, out/groups, k]
        constant_weight(&[dim_in], 0.1),        // bias
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("depthwise ConvTranspose1d graph (Kokoro F0 pattern)");

    let lower = ArrayD::from_elem(IxDyn(&[dim_in, 16]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[dim_in, 16]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let (method, output, _) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN depthwise ConvTranspose1d");
    assert_eq!(
        method,
        PropMethod::Crown,
        "depthwise ConvTranspose1d is linear — CROWN should succeed"
    );

    let (lo, hi) = output.lower_upper();
    // Output: [dim_in, out_len] where out_len = (16-1)*2 - 2*1 + 3 = 31
    assert_eq!(lo.shape()[0], dim_in, "output channels = dim_in");
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite(), "bounds must be finite");
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

// ---------------------------------------------------------------------------
// output_padding decomposition (#2558): ConvTranspose1d(output_padding=P) is
// decomposed into ConvTranspose1d(output_padding=0) + LinearLayer zero-pad.
// ---------------------------------------------------------------------------

/// ConvTranspose1d with output_padding=1 produces correct output shape and
/// finite CROWN bounds. The decomposition appends P zero-padded elements.
#[test]
fn test_conv_transpose_1d_output_padding_crown() {
    use nn_verify::{propagate_with_crown_fallback, BoundedTensor, PropMethod};

    // in_ch=2, out_ch=1, kernel=3, in_len=4, stride=2, pad=0, dilation=1, groups=1, output_padding=1
    // T_mid = (4-1)*2 - 0 + 1*(3-1) + 1 = 6+2+1 = 9
    // T_out = 9 + 1 = 10
    let def = build_conv_transpose_1d("ct1d_op", 2, 1, 3, 4, 2, 0, 1, 1, false, 1)
        .expect("build with output_padding=1");
    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[2, 1, 3], 0.5), // weight
    ];
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("ConvTranspose1d output_padding graph");

    let lower = ArrayD::from_elem(IxDyn(&[2, 4]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 4]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let (method, output, fallback_reason) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN with output_padding");

    assert_eq!(
        method,
        PropMethod::Crown,
        "CROWN should succeed (ConvTranspose1d + Linear are both linear)"
    );
    assert!(fallback_reason.is_none(), "no fallback expected");

    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[1, 10], "output shape should be [1, T_out=10]");

    // All bounds must be finite and sound.
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite(), "bounds must be finite");
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }

    // The last element (from output_padding) should have exact zero bounds
    // since it comes from the zero-pad rows of the LinearLayer.
    let last_lo = lo.as_slice().unwrap()[9];
    let last_hi = hi.as_slice().unwrap()[9];
    assert!(
        last_lo.abs() < 1e-6 && last_hi.abs() < 1e-6,
        "output_padding element should have zero bounds, got [{last_lo}, {last_hi}]"
    );
}

/// IBP through ConvTranspose1d with output_padding produces identical results
/// to CROWN (both layers are linear).
#[test]
fn test_conv_transpose_1d_output_padding_ibp_vs_crown() {
    use nn_verify::{propagate_with_crown_fallback, BoundedTensor, PropMethod};

    let def =
        build_conv_transpose_1d("ct1d_op_cmp", 1, 1, 2, 3, 2, 0, 1, 1, true, 1).expect("build");
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![0.3, -0.2]).unwrap();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(kernel),
        constant_weight(&[1], 1.0), // bias
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let lower = ArrayD::from_elem(IxDyn(&[1, 3]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 3]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    let (ibp_lo, ibp_hi) = ibp_output.lower_upper();

    let (method, crown_output, _) = propagate_with_crown_fallback(&graph, &input).expect("CROWN");
    assert_eq!(method, PropMethod::Crown);
    let (crown_lo, crown_hi) = crown_output.lower_upper();

    assert_eq!(ibp_lo.shape(), crown_lo.shape(), "same output shape");

    let tol = 1e-4;
    for ((&cl, &il), (&cu, &iu)) in crown_lo
        .iter()
        .zip(ibp_lo.iter())
        .zip(crown_hi.iter().zip(ibp_hi.iter()))
    {
        assert!((cl - il).abs() < tol, "CROWN lower {cl} vs IBP lower {il}");
        assert!((cu - iu).abs() < tol, "CROWN upper {cu} vs IBP upper {iu}");
    }
}
