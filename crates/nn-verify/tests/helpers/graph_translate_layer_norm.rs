// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LayerNorm NY IBP verification tests (decomposed K7 + monolithic).
//! Part of #19 (K2-K8 kernel ports), #253 (end-to-end verification coverage).

use nn_dsl::layer_norm::{build_layer_norm_decomposed, layer_norm_ref};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// --- K7 LayerNorm NY translation ---

#[test]
fn test_layer_norm_k7_gamma_crown_translation() {
    let k7 = build_layer_norm_decomposed(4, 8).expect("build K7 must succeed");
    // x is Variable; eps, gamma, beta are constants.
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantScalar(0.0),
    ];
    let graph =
        tensor_kernel_to_graph(&k7, &bindings).expect("K7 LayerNorm NY graph must build");
    // The 18-node decomposed IR translates to: NETWORK_INPUT + reduce layers
    // + elementwise layers. Broadcasts are absorbed by consumers in NY.
    // Must be at least 9 (input + 2 reduces + 6 elementwise minimum).
    assert!(
        graph.num_nodes() >= 9,
        "K7 LayerNorm graph needs input + reduce + elementwise nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_layer_norm_k7_ibp_tight_bounds() {
    // Tight input range: x in [1, 2]. With gamma=1, beta=0, LayerNorm
    // normalizes to zero-mean unit-variance, so output should be in
    // approximately [-2, 2] for small hidden dimensions.
    let n = 2;
    let hidden = 4;
    let k7 = build_layer_norm_decomposed(n, hidden).expect("build K7");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantScalar(0.0),
    ];
    let graph = tensor_kernel_to_graph(&k7, &bindings).expect("build K7 graph");

    let lower = ArrayD::from_elem(IxDyn(&[n, hidden]), 1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[n, hidden]), 2.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP propagation for K7 LayerNorm must succeed");
    let (lo, hi) = output.lower_upper();

    // All bounds must be finite — IBP through ReduceMean + rsqrt can
    // produce infinity if the decomposition is wrong.
    assert!(
        lo.iter().all(|v| v.is_finite()),
        "K7 output lower bounds must be finite, got: {lo:?}"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "K7 output upper bounds must be finite, got: {hi:?}"
    );
}

#[test]
fn test_layer_norm_k7_ibp_wider_bounds() {
    // Wider input range: x in [-10, 10]. The decomposed form may suffer
    // from correlation loss under IBP (variance can underestimate), but
    // output must remain finite.
    let n = 2;
    let hidden = 8;
    let k7 = build_layer_norm_decomposed(n, hidden).expect("build K7");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantScalar(0.0),
    ];
    let graph = tensor_kernel_to_graph(&k7, &bindings).expect("build K7 graph");

    let lower = ArrayD::from_elem(IxDyn(&[n, hidden]), -10.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[n, hidden]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP for wider K7 LayerNorm input must succeed");
    let (lo, hi) = output.lower_upper();
    assert!(
        lo.iter().all(|v| v.is_finite()),
        "K7 output lower bounds must be finite for wide input, got: {lo:?}"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "K7 output upper bounds must be finite for wide input, got: {hi:?}"
    );
}

#[test]
fn test_layer_norm_k7_ibp_with_affine() {
    // Test with non-trivial gamma and beta constants.
    let n = 1;
    let hidden = 4;
    let k7 = build_layer_norm_decomposed(n, hidden).expect("build K7");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(2.0), // gamma = 2
        TensorParamBinding::ConstantScalar(0.5), // beta = 0.5
    ];
    let graph = tensor_kernel_to_graph(&k7, &bindings).expect("build K7 graph");

    let lower = ArrayD::from_elem(IxDyn(&[n, hidden]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[n, hidden]), 5.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP for affine K7 LayerNorm must succeed");
    let (lo, hi) = output.lower_upper();
    assert!(
        lo.iter().all(|v| v.is_finite()),
        "K7 affine output lower bounds must be finite, got: {lo:?}"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "K7 affine output upper bounds must be finite, got: {hi:?}"
    );
}

#[test]
fn test_layer_norm_k7_ibp_soundness_vs_reference() {
    // Verify soundness: sample a concrete input within bounds, compute
    // the reference output, and check that it falls within the IBP bounds.
    let n = 1;
    let hidden = 4;
    let k7 = build_layer_norm_decomposed(n, hidden).expect("build K7");

    let gamma_val = 1.0f32;
    let beta_val = 0.0f32;
    let eps_val = 1e-5f32;

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(eps_val),
        TensorParamBinding::ConstantScalar(gamma_val),
        TensorParamBinding::ConstantScalar(beta_val),
    ];
    let graph = tensor_kernel_to_graph(&k7, &bindings).expect("build K7 graph");

    // Input bounds: x in [0, 4]
    let lower = ArrayD::from_elem(IxDyn(&[n, hidden]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[n, hidden]), 4.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP propagation must succeed");
    let (lo, hi) = output.lower_upper();

    // Sample input: [1.0, 2.0, 3.0, 4.0] — within [0, 4]
    let x_sample = [1.0f32, 2.0, 3.0, 4.0];
    let gamma = vec![gamma_val; hidden];
    let beta = vec![beta_val; hidden];
    let ref_out =
        layer_norm_ref(&x_sample, &gamma, &beta, n, hidden, eps_val).expect("ref must succeed");

    // Check: each reference output value must fall within the IBP bounds.
    let lo_flat: Vec<f32> = lo.iter().copied().collect();
    let hi_flat: Vec<f32> = hi.iter().copied().collect();
    assert_eq!(
        ref_out.len(),
        lo_flat.len(),
        "reference output length {} != IBP bounds length {}",
        ref_out.len(),
        lo_flat.len()
    );
    for (i, &val) in ref_out.iter().enumerate() {
        assert!(
            val >= lo_flat[i] - 1e-3,
            "ref_out[{i}]={val} below IBP lower bound {}",
            lo_flat[i]
        );
        assert!(
            val <= hi_flat[i] + 1e-3,
            "ref_out[{i}]={val} above IBP upper bound {}",
            hi_flat[i]
        );
    }
}

#[test]
fn test_layer_norm_k7_ibp_soundness_multiple_samples() {
    // Verify soundness across multiple sample inputs to increase confidence
    // in the IBP bounds.
    let n = 1;
    let hidden = 4;
    let k7 = build_layer_norm_decomposed(n, hidden).expect("build K7");

    let gamma_val = 1.5f32;
    let beta_val = -0.3f32;
    let eps_val = 1e-5f32;

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(eps_val),
        TensorParamBinding::ConstantScalar(gamma_val),
        TensorParamBinding::ConstantScalar(beta_val),
    ];
    let graph = tensor_kernel_to_graph(&k7, &bindings).expect("build K7 graph");

    // Input bounds: x in [-5, 5]
    let lower = ArrayD::from_elem(IxDyn(&[n, hidden]), -5.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[n, hidden]), 5.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP propagation must succeed");
    let (lo, hi) = output.lower_upper();
    let lo_flat: Vec<f32> = lo.iter().copied().collect();
    let hi_flat: Vec<f32> = hi.iter().copied().collect();

    let gamma = vec![gamma_val; hidden];
    let beta = vec![beta_val; hidden];

    // Sample inputs that span the input range
    let samples: Vec<[f32; 4]> = vec![
        [-5.0, -5.0, -5.0, -5.0], // constant (all same)
        [5.0, 5.0, 5.0, 5.0],     // constant (all same)
        [-5.0, -2.0, 2.0, 5.0],   // linearly spaced
        [0.0, 0.0, 0.0, 0.0],     // zeros
        [-5.0, 5.0, -5.0, 5.0],   // alternating extremes
        [1.0, 2.0, 3.0, 4.0],     // ascending
        [-1.0, -0.5, 0.5, 1.0],   // centered
    ];

    for (si, x_sample) in samples.iter().enumerate() {
        let ref_out =
            layer_norm_ref(x_sample, &gamma, &beta, n, hidden, eps_val).expect("ref must succeed");
        assert_eq!(
            ref_out.len(),
            lo_flat.len(),
            "sample {si}: reference output length {} != IBP bounds length {}",
            ref_out.len(),
            lo_flat.len()
        );
        for (i, &val) in ref_out.iter().enumerate() {
            assert!(
                val >= lo_flat[i] - 1e-2,
                "sample {si}: ref_out[{i}]={val} below IBP lower bound {}",
                lo_flat[i]
            );
            assert!(
                val <= hi_flat[i] + 1e-2,
                "sample {si}: ref_out[{i}]={val} above IBP upper bound {}",
                hi_flat[i]
            );
        }
    }
}

#[test]
fn test_layer_norm_k7_binding_count_mismatch() {
    // K7 has 4 inputs (x, eps, gamma, beta). Providing fewer should error.
    let k7 = build_layer_norm_decomposed(4, 8).expect("build K7");
    let err = tensor_kernel_to_graph(
        &k7,
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
