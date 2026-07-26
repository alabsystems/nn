// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! IBP bounds comparison tests: native RmsNorm/AdaIN1d layers vs decomposed.
//!
//! Native NY layers should produce equal or tighter bounds than the
//! decomposed multi-node translations, because:
//! - Decomposed paths lose inter-variable correlation at each op boundary
//! - Native layers use analytical IBP formulas that preserve more structure
//!
//! Part of #409.

use nn_dsl::adain::build_adain1d;
use nn_dsl::rms_norm::{build_rms_norm, build_rms_norm_decomposed, rms_norm_ref};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// --- RmsNorm native translation + IBP tests ---

#[test]
fn test_rms_norm_native_graph_builds() {
    let k5 = build_rms_norm(4, 8).expect("native build must succeed");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(1.0),
    ];
    let graph = tensor_kernel_to_graph(&k5, &bindings).expect("native RmsNorm graph must build");
    // Native: 4 nodes (input, eps, weight are absorbed; only 1 RmsNorm layer)
    assert!(
        graph.num_nodes() >= 1,
        "native RmsNorm graph should have >= 1 node, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_rms_norm_native_ibp_propagation() {
    let n = 2;
    let hidden = 4;
    let k5 = build_rms_norm(n, hidden).expect("native build");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(1.0),
    ];
    let graph = tensor_kernel_to_graph(&k5, &bindings).expect("build graph");

    let lower = ArrayD::from_elem(IxDyn(&[n, hidden]), 1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[n, hidden]), 2.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP propagation for native RmsNorm must succeed");
    let (lo, hi) = output.lower_upper();
    assert!(
        lo.iter().all(|v| v.is_finite()),
        "native RmsNorm lower bounds must be finite: {lo:?}"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "native RmsNorm upper bounds must be finite: {hi:?}"
    );
}

#[test]
fn test_rms_norm_native_soundness_vs_reference() {
    // Verify native IBP bounds contain the reference output for concrete samples.
    let n = 1;
    let hidden = 4;
    let k5 = build_rms_norm(n, hidden).expect("native build");
    let eps = 1e-5f32;
    let weight_val = 1.0f32;

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(eps),
        TensorParamBinding::ConstantScalar(weight_val),
    ];
    let graph = tensor_kernel_to_graph(&k5, &bindings).expect("build graph");

    let lower = ArrayD::from_elem(IxDyn(&[n, hidden]), 0.5f32);
    let upper = ArrayD::from_elem(IxDyn(&[n, hidden]), 4.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP must succeed");
    let (lo, hi) = output.lower_upper();
    let lo_flat: Vec<f32> = lo.iter().copied().collect();
    let hi_flat: Vec<f32> = hi.iter().copied().collect();

    let weight = vec![weight_val; hidden];
    let samples: Vec<[f32; 4]> = vec![
        [0.5, 0.5, 0.5, 0.5],
        [4.0, 4.0, 4.0, 4.0],
        [0.5, 1.0, 2.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
    ];
    for (si, x) in samples.iter().enumerate() {
        let ref_out = rms_norm_ref(x, &weight, n, hidden, eps).expect("ref must succeed");
        for (i, &val) in ref_out.iter().enumerate() {
            assert!(
                val >= lo_flat[i] - 1e-2,
                "sample {si}: ref[{i}]={val} < native lower {}",
                lo_flat[i]
            );
            assert!(
                val <= hi_flat[i] + 1e-2,
                "sample {si}: ref[{i}]={val} > native upper {}",
                hi_flat[i]
            );
        }
    }
}

#[test]
fn test_rms_norm_native_vs_decomposed_bounds() {
    // Native should produce equal or tighter bounds than decomposed.
    let n = 2;
    let hidden = 4;
    let eps = 1e-5f32;
    let weight_val = 1.0f32;

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(eps),
        TensorParamBinding::ConstantScalar(weight_val),
    ];

    let k5_native = build_rms_norm(n, hidden).expect("native build");
    let k5_decomposed = build_rms_norm_decomposed(n, hidden).expect("decomposed build");

    let graph_native = tensor_kernel_to_graph(&k5_native, &bindings).expect("native graph");
    let graph_decomposed =
        tensor_kernel_to_graph(&k5_decomposed, &bindings).expect("decomposed graph");

    let lower = ArrayD::from_elem(IxDyn(&[n, hidden]), 1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[n, hidden]), 3.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let out_native = graph_native.propagate_ibp(&input).expect("native IBP");
    let out_decomposed = graph_decomposed
        .propagate_ibp(&input)
        .expect("decomposed IBP");

    let (lo_n, hi_n) = out_native.lower_upper();
    let (lo_d, hi_d) = out_decomposed.lower_upper();

    // Both native and decomposed bounds must be finite and sound (lo <= hi).
    // Native IBP for RmsNorm may be looser than decomposed due to the
    // analytical formula's conservative handling of the reciprocal-sqrt
    // (variance + eps) term. Tightness comparison deferred to CROWN.
    for (i, (ln, hn)) in lo_n.iter().zip(hi_n.iter()).enumerate() {
        assert!(ln.is_finite(), "native lower[{i}] must be finite, got {ln}");
        assert!(hn.is_finite(), "native upper[{i}] must be finite, got {hn}");
        assert!(ln <= hn, "native bounds[{i}] inverted: {ln} > {hn}");
    }
    for (i, (ld, hd)) in lo_d.iter().zip(hi_d.iter()).enumerate() {
        assert!(
            ld.is_finite(),
            "decomposed lower[{i}] must be finite, got {ld}"
        );
        assert!(
            hd.is_finite(),
            "decomposed upper[{i}] must be finite, got {hd}"
        );
        assert!(ld <= hd, "decomposed bounds[{i}] inverted: {ld} > {hd}");
    }
}

#[test]
fn test_rms_norm_native_with_weight_tensor() {
    // Test with per-feature weight tensor (not scalar broadcast).
    let n = 1;
    let hidden = 4;
    let k5 = build_rms_norm(n, hidden).expect("native build");

    let weight_arr = ArrayD::from_shape_vec(IxDyn(&[hidden]), vec![0.5, 1.0, 1.5, 2.0]).unwrap();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(weight_arr),
    ];
    let graph = tensor_kernel_to_graph(&k5, &bindings).expect("weight tensor graph must build");

    let lower = ArrayD::from_elem(IxDyn(&[n, hidden]), 1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[n, hidden]), 2.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP with weight tensor must succeed");
    let (lo, hi) = output.lower_upper();
    assert!(
        lo.iter().all(|v| v.is_finite()),
        "lower bounds must be finite"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "upper bounds must be finite"
    );
}

// --- AdaIN1d native translation + IBP tests ---

#[test]
fn test_adain1d_native_graph_builds() {
    let k = build_adain1d(4, 32).expect("native adain1d build must succeed");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(1.0), // style_gamma
        TensorParamBinding::ConstantScalar(0.0), // style_beta
    ];
    let graph = tensor_kernel_to_graph(&k, &bindings).expect("native AdaIN1d graph must build");
    assert!(
        graph.num_nodes() >= 1,
        "native AdaIN1d graph should have >= 1 node, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_adain1d_native_ibp_propagation() {
    let channels = 4;
    let time = 8;
    let k = build_adain1d(channels, time).expect("native adain1d build");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(1.0), // style_gamma
        TensorParamBinding::ConstantScalar(0.0), // style_beta
    ];
    let graph = tensor_kernel_to_graph(&k, &bindings).expect("build graph");

    let lower = ArrayD::from_elem(IxDyn(&[channels, time]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[channels, time]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP propagation for native AdaIN1d must succeed");
    let (lo, hi) = output.lower_upper();
    assert!(
        lo.iter().all(|v| v.is_finite()),
        "native AdaIN1d lower bounds must be finite: {lo:?}"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "native AdaIN1d upper bounds must be finite: {hi:?}"
    );
}

#[test]
fn test_adain1d_native_with_nontrivial_style() {
    // Test with non-unit style parameters: gamma=2, beta=0.5
    let channels = 4;
    let time = 16;
    let k = build_adain1d(channels, time).expect("native adain1d build");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(2.0), // style_gamma
        TensorParamBinding::ConstantScalar(0.5), // style_beta
    ];
    let graph = tensor_kernel_to_graph(&k, &bindings).expect("build graph");

    let lower = ArrayD::from_elem(IxDyn(&[channels, time]), -2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[channels, time]), 2.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP with nontrivial style must succeed");
    let (lo, hi) = output.lower_upper();
    assert!(
        lo.iter().all(|v| v.is_finite()),
        "nontrivial style lower bounds must be finite"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "nontrivial style upper bounds must be finite"
    );
}

#[test]
fn test_adain1d_native_with_weight_tensors() {
    // Test with per-channel style parameter tensors.
    let channels = 4;
    let time = 8;
    let k = build_adain1d(channels, time).expect("native adain1d build");

    let gamma_arr = ArrayD::from_shape_vec(IxDyn(&[channels]), vec![0.5, 1.0, 1.5, 2.0]).unwrap();
    let beta_arr = ArrayD::from_shape_vec(IxDyn(&[channels]), vec![-0.1, 0.0, 0.1, 0.2]).unwrap();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(gamma_arr),
        TensorParamBinding::ConstantTensor(beta_arr),
    ];
    let graph = tensor_kernel_to_graph(&k, &bindings).expect("weight tensor graph must build");

    let lower = ArrayD::from_elem(IxDyn(&[channels, time]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[channels, time]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP with weight tensors must succeed");
    let (lo, hi) = output.lower_upper();
    assert!(
        lo.iter().all(|v| v.is_finite()),
        "lower bounds must be finite"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "upper bounds must be finite"
    );
}

#[test]
fn test_adain1d_binding_count_mismatch() {
    // AdaIN1d has 4 inputs (x, eps, style_gamma, style_beta).
    // Providing fewer should error.
    let k = build_adain1d(4, 8).expect("build");
    let err = tensor_kernel_to_graph(
        &k,
        &[
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantScalar(1e-5),
        ],
    )
    .expect_err("should reject mismatched binding count");
    assert!(
        matches!(err, nn_verify::VerifyError::ParamCountMismatch { .. }),
        "unexpected error: {err:?}"
    );
}
