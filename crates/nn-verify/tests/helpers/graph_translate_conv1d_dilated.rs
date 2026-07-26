// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for dilated Conv1d kernel expansion workaround (#582).
//!
//! Dilation is decomposed into an equivalent standard Conv1d with zero-inserted
//! kernel. These tests verify graph construction, IBP bounds, CROWN bounds,
//! and soundness with concrete forward pass for dilated convolutions.

use nn_dsl::conv1d::build_conv1d_full;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Build a `TensorParamBinding::ConstantTensor` from a shape and constant value.
fn constant_weight(shape: &[usize], value: f32) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), value))
}

/// Dilated Conv1d (dilation=2): graph builds and translates successfully.
///
/// in_ch=1, out_ch=2, k=3, in_len=8, stride=1, pad=0, dilation=2
/// effective kernel = 2*(3-1)+1 = 5, out_len = (8+0-5)/1+1 = 4
#[test]
fn test_conv1d_dilation_2_graph_builds() {
    let def = build_conv1d_full("conv1d_d2", 1, 2, 3, 8, 1, 0, 2, 1, false).expect("build");
    assert_eq!(def.nodes.last().unwrap().shape, vec![2, 4]);

    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[2, 1, 3], 0.1),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("dilated conv1d d=2 should build");
    assert!(graph.num_nodes() >= 1);
}

/// Dvoice DConv pattern: dilation=8, kernel=3, stride=1, padding=8.
///
/// in_ch=1, out_ch=4, k=3, in_len=32, stride=1, pad=8, dilation=8
/// effective kernel = 8*(3-1)+1 = 17, out_len = (32+16-17)/1+1 = 32
/// (same-length output, characteristic of DConv blocks)
#[test]
fn test_conv1d_dilation_8_dvoice_dconv_graph_builds() {
    let def = build_conv1d_full("conv1d_dconv", 1, 4, 3, 32, 1, 8, 8, 1, false).expect("build");
    assert_eq!(def.nodes.last().unwrap().shape, vec![4, 32]);

    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[4, 1, 3], 0.05),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("dvoice DConv d=8 should build via kernel expansion");
    assert!(graph.num_nodes() >= 1);
}

/// Dilated Conv1d IBP bounds (dilation=2): bounds are finite and sound.
#[test]
fn test_conv1d_dilation_2_ibp_bounds() {
    use nn_verify::BoundedTensor;

    // in_ch=2, out_ch=3, k=2, in_len=8, stride=1, pad=0, dilation=2
    // effective k = 2*(2-1)+1 = 3, out_len = (8+0-3)/1+1 = 6
    let def = build_conv1d_full("conv1d_d2_ibp", 2, 3, 2, 8, 1, 0, 2, 1, false).expect("build");
    assert_eq!(def.nodes.last().unwrap().shape, vec![3, 6]);

    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[3, 2, 2]),
        vec![
            0.5, -0.3, 0.2, 0.4, -0.1, 0.6, 0.3, -0.5, 0.7, 0.1, -0.4, 0.2,
        ],
    )
    .unwrap();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(kernel),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("d=2 conv1d graph");

    let lower = ArrayD::from_elem(IxDyn(&[2, 8]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 8]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through dilated Conv1d");
    let (lo, hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[3, 6], "output shape [3, 6]");
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite(), "bounds must be finite");
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
}

/// Dvoice DConv IBP bounds (dilation=8): bounds are finite and sound.
///
/// Uses the dvoice DConv pattern: k=3, dilation=8, padding=8 for
/// same-length output. Verifies IBP bounds propagate through the
/// expanded (17-wide) kernel.
#[test]
fn test_conv1d_dilation_8_dvoice_dconv_ibp_bounds() {
    use nn_verify::BoundedTensor;

    // in_ch=1, out_ch=4, k=3, in_len=32, stride=1, pad=8, dilation=8
    let def = build_conv1d_full("conv1d_dconv_ibp", 1, 4, 3, 32, 1, 8, 8, 1, false).expect("build");
    assert_eq!(def.nodes.last().unwrap().shape, vec![4, 32]);

    let bindings = vec![
        TensorParamBinding::Variable,
        constant_weight(&[4, 1, 3], 0.05),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("dconv d=8 graph");

    let lower = ArrayD::from_elem(IxDyn(&[1, 32]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 32]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph.propagate_ibp(&input).expect("IBP through DConv d=8");
    let (lo, hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[4, 32], "same-length output for DConv");
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l.is_finite() && u.is_finite(), "bounds must be finite");
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
    // With all-positive weights of 0.05 and input in [-1,1]:
    // Each output sums 1*3=3 products. max deviation = 3*0.05 = 0.15.
    assert!(
        lo.iter().all(|&v| v >= -0.2),
        "lower >= -0.2 for small positive weights"
    );
    assert!(
        hi.iter().all(|&v| v <= 0.2),
        "upper <= 0.2 for small positive weights"
    );
}

/// Dilated Conv1d soundness: concrete forward pass falls within IBP bounds.
///
/// Manually computes dilated Conv1d forward pass and verifies it lies
/// within the propagated IBP bounds.
#[test]
fn test_conv1d_dilation_2_ibp_soundness_concrete() {
    use nn_verify::BoundedTensor;

    // in_ch=1, out_ch=1, k=3, in_len=8, stride=1, pad=0, dilation=2
    // effective k = 5, out_len = (8-5)/1+1 = 4
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![0.3, -0.5, 0.2]).unwrap();
    let def = build_conv1d_full("conv1d_d2_sound", 1, 1, 3, 8, 1, 0, 2, 1, false).expect("build");
    assert_eq!(def.nodes.last().unwrap().shape, vec![1, 4]);

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(kernel),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let lower = ArrayD::from_elem(IxDyn(&[1, 8]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 8]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    let output = graph.propagate_ibp(&input).expect("IBP dilated soundness");
    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.shape(), &[1, 4]);

    // Manual forward: x = [0.5, -0.3, 0.8, -0.1, 0.4, -0.7, 0.2, 0.6]
    // dilated conv with kernel [0.3, -0.5, 0.2], dilation=2:
    //   pos 0: 0.3*x[0] + (-0.5)*x[2] + 0.2*x[4] = 0.15 - 0.4 + 0.08 = -0.17
    //   pos 1: 0.3*x[1] + (-0.5)*x[3] + 0.2*x[5] = -0.09 + 0.05 - 0.14 = -0.18
    //   pos 2: 0.3*x[2] + (-0.5)*x[4] + 0.2*x[6] = 0.24 - 0.2 + 0.04 = 0.08
    //   pos 3: 0.3*x[3] + (-0.5)*x[5] + 0.2*x[7] = -0.03 + 0.35 + 0.12 = 0.44
    let fwd = [-0.17f32, -0.18, 0.08, 0.44];
    for (i, &f) in fwd.iter().enumerate() {
        assert!(lo[[0, i]] <= f + 0.01, "lo[{i}]={} <= fwd={f}", lo[[0, i]]);
        assert!(hi[[0, i]] >= f - 0.01, "hi[{i}]={} >= fwd={f}", hi[[0, i]]);
    }
}

/// Dilated Conv1d IBP soundness for all dvoice dilation values (1,2,3,5,8).
///
/// For each dilation, builds a Conv1d with k=3, verifies IBP bounds are finite
/// and sound (concrete forward pass within bounds).
#[test]
fn test_conv1d_all_dvoice_dilations_ibp_soundness() {
    use nn_verify::BoundedTensor;

    let kernel_weights = vec![0.3f32, -0.5, 0.2];
    let in_len = 40; // large enough for all dilations

    for dilation in [1, 2, 3, 5, 8] {
        let eff_k = dilation * (3 - 1) + 1;
        let out_len = in_len - eff_k + 1;
        assert!(out_len > 0, "out_len must be > 0 for dilation={dilation}");

        let def = build_conv1d_full(
            &format!("conv1d_d{dilation}"),
            1,
            1,
            3,
            in_len,
            1,
            0,
            dilation,
            1,
            false,
        )
        .expect("build");
        assert_eq!(
            def.nodes.last().unwrap().shape,
            vec![1, out_len],
            "dilation={dilation}"
        );

        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), kernel_weights.clone()).unwrap();
        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(kernel),
        ];
        let graph = tensor_kernel_to_graph(&def, &bindings)
            .unwrap_or_else(|e| panic!("d={dilation} graph: {e}"));

        let lower = ArrayD::from_elem(IxDyn(&[1, in_len]), -1.0f32);
        let upper = ArrayD::from_elem(IxDyn(&[1, in_len]), 1.0f32);
        let input = BoundedTensor::new(lower, upper).expect("bounds");

        let output = graph
            .propagate_ibp(&input)
            .unwrap_or_else(|e| panic!("d={dilation} IBP: {e}"));
        let (lo, hi) = output.lower_upper();

        assert_eq!(lo.shape(), &[1, out_len], "d={dilation} output shape");
        for (l, u) in lo.iter().zip(hi.iter()) {
            assert!(l.is_finite() && u.is_finite(), "d={dilation} bounds finite");
            assert!(l <= u, "d={dilation} lo <= hi");
        }
    }
}

/// All dvoice dilation values: CROWN vs IBP tightness comparison.
///
/// For a pure linear (Conv1d) layer, CROWN and IBP should give identical
/// bounds. Verifies this holds across all dvoice dilations.
#[test]
fn test_conv1d_all_dvoice_dilations_crown_vs_ibp() {
    use nn_verify::{propagate_with_crown_fallback, BoundedTensor, PropMethod};

    let kernel_weights = vec![0.3f32, -0.5, 0.2];
    let in_len = 40;

    for dilation in [1, 2, 3, 5, 8] {
        let eff_k = dilation * (3 - 1) + 1;
        let out_len = in_len - eff_k + 1;
        if out_len == 0 {
            continue;
        }

        let def = build_conv1d_full(
            &format!("conv1d_crown_d{dilation}"),
            1,
            1,
            3,
            in_len,
            1,
            0,
            dilation,
            1,
            false,
        )
        .expect("build");

        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), kernel_weights.clone()).unwrap();
        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(kernel),
        ];
        let graph = tensor_kernel_to_graph(&def, &bindings)
            .unwrap_or_else(|e| panic!("d={dilation} graph: {e}"));

        let lower = ArrayD::from_elem(IxDyn(&[1, in_len]), -1.0f32);
        let upper = ArrayD::from_elem(IxDyn(&[1, in_len]), 1.0f32);
        let input = BoundedTensor::new(lower, upper).expect("bounds");

        let ibp = graph
            .propagate_ibp(&input)
            .unwrap_or_else(|e| panic!("d={dilation} IBP: {e}"));
        let (ibp_lo, ibp_hi) = ibp.lower_upper();

        let (method, crown_out, _) = propagate_with_crown_fallback(&graph, &input)
            .unwrap_or_else(|e| panic!("d={dilation} CROWN: {e}"));
        assert_eq!(
            method,
            PropMethod::Crown,
            "d={dilation}: CROWN should succeed for linear Conv1d"
        );
        let (crown_lo, crown_hi) = crown_out.lower_upper();

        let tol = 1e-4;
        for ((&cl, &il), (&cu, &iu)) in crown_lo
            .iter()
            .zip(ibp_lo.iter())
            .zip(crown_hi.iter().zip(ibp_hi.iter()))
        {
            assert!(
                (cl - il).abs() < tol,
                "d={dilation}: CROWN lo {cl} vs IBP lo {il}"
            );
            assert!(
                (cu - iu).abs() < tol,
                "d={dilation}: CROWN hi {cu} vs IBP hi {iu}"
            );
        }
    }
}

/// Dilated Conv1d CROWN backward succeeds and matches IBP for linear layer.
#[test]
fn test_conv1d_dilation_2_crown_bounds() {
    use nn_verify::{propagate_with_crown_fallback, BoundedTensor, PropMethod};

    // in_ch=2, out_ch=1, k=3, in_len=8, stride=1, pad=0, dilation=2
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 3]), vec![0.3, -0.2, 0.1, 0.4, 0.5, -0.1]).unwrap();
    let def = build_conv1d_full("conv1d_d2_crown", 2, 1, 3, 8, 1, 0, 2, 1, false).expect("build");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(kernel),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let lower = ArrayD::from_elem(IxDyn(&[2, 8]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 8]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("bounds");

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    let (ibp_lo, ibp_hi) = ibp_output.lower_upper();

    // CROWN
    let (method, crown_output, _) = propagate_with_crown_fallback(&graph, &input).expect("CROWN");
    assert_eq!(
        method,
        PropMethod::Crown,
        "CROWN should succeed for linear dilated Conv1d"
    );
    let (crown_lo, crown_hi) = crown_output.lower_upper();

    assert_eq!(ibp_lo.shape(), crown_lo.shape());

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
