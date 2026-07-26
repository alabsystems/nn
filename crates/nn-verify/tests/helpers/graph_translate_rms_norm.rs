// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! K5 RMSNorm NY IBP verification tests.
//!
//! Exercises the full tensor pipeline for the 12-node decomposed RMSNorm
//! kernel: Elementwise(square) → ReduceMean → Broadcast → Elementwise(add,
//! rsqrt, mul) → Broadcast(weight) → Elementwise(mul).
//!
//! Part of #19 (K2-K8 kernel ports).

use nn_dsl::rms_norm::{build_rms_norm_decomposed, rms_norm_ref};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// --- K5 RMSNorm NY translation ---

#[test]
fn test_rms_norm_k5_gamma_crown_translation() {
    let k5 = build_rms_norm_decomposed(4, 8).expect("build K5 must succeed");
    // x is Variable; eps and weight are constants.
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(1.0),
    ];
    let graph =
        tensor_kernel_to_graph(&k5, &bindings).expect("K5 RMSNorm NY graph must build");
    // 12-node decomposed IR → 7 NY nodes:
    // MulBinary(x²) + ReduceMean + AddConstant(+eps) + Sqrt + Reciprocal
    // + MulBinary(x*rsqrt) + MulConstant(*weight). Broadcasts absorbed.
    assert!(
        graph.num_nodes() >= 7,
        "K5 RMSNorm graph needs 7 nodes (square + reduce + add + sqrt + recip + 2 mul), got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_rms_norm_k5_ibp_tight_bounds() {
    // Tight positive input range: x in [1, 2]. With weight=1, RMSNorm
    // normalizes by 1/sqrt(mean(x²)+eps), so for constant input c,
    // output ≈ c/|c| = sign(c). For a range, bounds should be finite and
    // within a reasonable range.
    let n = 2;
    let hidden = 4;
    let k5 = build_rms_norm_decomposed(n, hidden).expect("build K5");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(1.0),
    ];
    let graph = tensor_kernel_to_graph(&k5, &bindings).expect("build K5 graph");

    let lower = ArrayD::from_elem(IxDyn(&[n, hidden]), 1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[n, hidden]), 2.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP propagation for K5 RMSNorm must succeed");
    let (lo, hi) = output.lower_upper();

    assert!(
        lo.iter().all(|v| v.is_finite()),
        "K5 output lower bounds must be finite, got: {lo:?}"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "K5 output upper bounds must be finite, got: {hi:?}"
    );
}

#[test]
fn test_rms_norm_k5_ibp_wider_bounds() {
    // Wider input range: x in [-10, 10]. The decomposed form uses
    // x² → ReduceMean → rsqrt, which under IBP loses correlation between
    // x and x². Bounds must remain finite despite this correlation loss.
    let n = 2;
    let hidden = 8;
    let k5 = build_rms_norm_decomposed(n, hidden).expect("build K5");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(1.0),
    ];
    let graph = tensor_kernel_to_graph(&k5, &bindings).expect("build K5 graph");

    let lower = ArrayD::from_elem(IxDyn(&[n, hidden]), -10.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[n, hidden]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP for wider K5 RMSNorm input must succeed");
    let (lo, hi) = output.lower_upper();
    assert!(
        lo.iter().all(|v| v.is_finite()),
        "K5 output lower bounds must be finite for wide input, got: {lo:?}"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "K5 output upper bounds must be finite for wide input, got: {hi:?}"
    );
}

#[test]
fn test_rms_norm_k5_ibp_with_weight() {
    // Test with non-trivial weight constant (weight=2.5).
    // RMSNorm output = x * rsqrt(mean(x²)+eps) * weight, so weight
    // scales the output linearly.
    let n = 1;
    let hidden = 4;
    let k5 = build_rms_norm_decomposed(n, hidden).expect("build K5");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(2.5), // weight = 2.5
    ];
    let graph = tensor_kernel_to_graph(&k5, &bindings).expect("build K5 graph");

    let lower = ArrayD::from_elem(IxDyn(&[n, hidden]), 0.5f32);
    let upper = ArrayD::from_elem(IxDyn(&[n, hidden]), 3.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP for weighted K5 RMSNorm must succeed");
    let (lo, hi) = output.lower_upper();
    assert!(
        lo.iter().all(|v| v.is_finite()),
        "K5 weighted output lower bounds must be finite, got: {lo:?}"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "K5 weighted output upper bounds must be finite, got: {hi:?}"
    );
}

#[test]
fn test_rms_norm_k5_ibp_soundness_vs_reference() {
    // Verify soundness: sample a concrete input within bounds, compute
    // the reference output, and check that it falls within the IBP bounds.
    let n = 1;
    let hidden = 4;
    let k5 = build_rms_norm_decomposed(n, hidden).expect("build K5");

    let weight_val = 1.0f32;
    let eps_val = 1e-5f32;

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(eps_val),
        TensorParamBinding::ConstantScalar(weight_val),
    ];
    let graph = tensor_kernel_to_graph(&k5, &bindings).expect("build K5 graph");

    // Input bounds: x in [0.5, 4]
    let lower = ArrayD::from_elem(IxDyn(&[n, hidden]), 0.5f32);
    let upper = ArrayD::from_elem(IxDyn(&[n, hidden]), 4.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP propagation must succeed");
    let (lo, hi) = output.lower_upper();

    // Sample input: [1.0, 2.0, 3.0, 4.0] — within [0.5, 4]
    let x_sample = [1.0f32, 2.0, 3.0, 4.0];
    let weight = vec![weight_val; hidden];
    let ref_out = rms_norm_ref(&x_sample, &weight, n, hidden, eps_val).expect("ref must succeed");

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
fn test_rms_norm_k5_ibp_soundness_multiple_samples() {
    // Verify soundness across multiple sample inputs to increase confidence
    // in the IBP bounds.
    let n = 1;
    let hidden = 4;
    let k5 = build_rms_norm_decomposed(n, hidden).expect("build K5");

    let weight_val = 1.5f32;
    let eps_val = 1e-5f32;

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(eps_val),
        TensorParamBinding::ConstantScalar(weight_val),
    ];
    let graph = tensor_kernel_to_graph(&k5, &bindings).expect("build K5 graph");

    // Input bounds: x in [0.1, 5] (positive range avoids sign-change complexity)
    let lower = ArrayD::from_elem(IxDyn(&[n, hidden]), 0.1f32);
    let upper = ArrayD::from_elem(IxDyn(&[n, hidden]), 5.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP propagation must succeed");
    let (lo, hi) = output.lower_upper();
    let lo_flat: Vec<f32> = lo.iter().copied().collect();
    let hi_flat: Vec<f32> = hi.iter().copied().collect();

    let weight = vec![weight_val; hidden];

    // Sample inputs that span the input range
    let samples: Vec<[f32; 4]> = vec![
        [0.1, 0.1, 0.1, 0.1], // constant low
        [5.0, 5.0, 5.0, 5.0], // constant high
        [0.1, 1.0, 3.0, 5.0], // ascending
        [5.0, 3.0, 1.0, 0.1], // descending
        [0.1, 5.0, 0.1, 5.0], // alternating
        [1.0, 2.0, 3.0, 4.0], // linear
        [2.5, 2.5, 2.5, 2.5], // midpoint constant
    ];

    for (si, x_sample) in samples.iter().enumerate() {
        let ref_out =
            rms_norm_ref(x_sample, &weight, n, hidden, eps_val).expect("ref must succeed");
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
fn test_rms_norm_k5_ibp_zero_crossing() {
    // Test input range crossing zero: x in [-3, 3]. RMSNorm is odd-symmetric
    // (sign-preserving), so for symmetric input range around zero, the output
    // should also be symmetric. IBP may over-approximate due to correlation
    // loss in x * rsqrt(mean(x²)).
    let n = 1;
    let hidden = 4;
    let k5 = build_rms_norm_decomposed(n, hidden).expect("build K5");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantScalar(1.0),
    ];
    let graph = tensor_kernel_to_graph(&k5, &bindings).expect("build K5 graph");

    let lower = ArrayD::from_elem(IxDyn(&[n, hidden]), -3.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[n, hidden]), 3.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP for zero-crossing K5 RMSNorm must succeed");
    let (lo, hi) = output.lower_upper();
    assert!(
        lo.iter().all(|v| v.is_finite()),
        "K5 zero-crossing output lower bounds must be finite, got: {lo:?}"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "K5 zero-crossing output upper bounds must be finite, got: {hi:?}"
    );

    // Soundness check: sample concrete inputs within [-3, 3] and verify
    // reference outputs fall within IBP bounds. Zero-crossing is the
    // highest-risk regime for RMSNorm IBP (correlation loss in x * rsqrt(mean(x²))).
    let lo_flat: Vec<f32> = lo.iter().copied().collect();
    let hi_flat: Vec<f32> = hi.iter().copied().collect();
    let weight = vec![1.0f32; hidden];
    let zero_cross_samples: Vec<[f32; 4]> = vec![
        [-3.0, -3.0, -3.0, -3.0], // all negative
        [3.0, 3.0, 3.0, 3.0],     // all positive
        [-3.0, 3.0, -3.0, 3.0],   // alternating extremes
        [-0.1, 0.1, -0.1, 0.1],   // near-zero alternating
        [0.0, 0.0, 0.01, -0.01],  // near-zero with tiny perturbation
        [-1.0, 0.0, 1.0, 2.0],    // crossing zero linearly
    ];
    for (si, x_sample) in zero_cross_samples.iter().enumerate() {
        let ref_out = rms_norm_ref(x_sample, &weight, n, hidden, 1e-5).expect("ref must succeed");
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
fn test_rms_norm_k5_binding_count_mismatch() {
    // K5 has 3 inputs (x, eps, weight). Providing fewer should error.
    let k5 = build_rms_norm_decomposed(4, 8).expect("build K5");
    let err = tensor_kernel_to_graph(
        &k5,
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
