// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! K8 SiLU-Mul NY IBP verification tests.
//!
//! Exercises the scalar KernelDef → NY GraphNetwork translation
//! for `silu_mul(x, up) = silu(x) * up`, verifying IBP bounds propagation
//! and soundness against the analytical bounds function and reference impl.
//!
//! Part of #19 (K2-K8 kernel ports).

use nn_dsl::silu_mul::{build_silu_mul_kernel, silu_mul_scalar, silu_mul_scalar_bounds};
use nn_verify::{kernel_to_graph, BoundedTensor};
use ndarray::{ArrayD, IxDyn};

// --- K8 SiLU-Mul NY translation ---

#[test]
fn test_silu_mul_gamma_crown_translation() {
    let kernel = build_silu_mul_kernel().expect("build K8 SiLU-Mul");
    // x is Variable (param 0), up is Constant (param 1)
    // With native SiLULayer (#338 AC3), up=1.0 produces 1 node (SiLU only),
    // up!=1.0 produces 2 nodes (SiLU + MulConstant).
    let graph_up1 =
        kernel_to_graph(&kernel, &[1.0]).expect("K8 SiLU-Mul NY graph must build");
    assert_eq!(
        graph_up1.num_nodes(),
        1,
        "silu_mul with up=1.0 should use native SiLULayer (1 node)"
    );

    let graph_up2 =
        kernel_to_graph(&kernel, &[2.0]).expect("K8 SiLU-Mul NY graph must build");
    assert_eq!(
        graph_up2.num_nodes(),
        2,
        "silu_mul with up=2.0 should use SiLULayer + MulConstant (2 nodes)"
    );
}

#[test]
fn test_silu_mul_ibp_positive_range() {
    // x in [1, 5] with up = 1. silu is monotonically increasing for x > -1.278,
    // so output should be in [silu(1), silu(5)] ≈ [0.731, 4.966].
    let kernel = build_silu_mul_kernel().expect("build K8");
    let graph = kernel_to_graph(&kernel, &[1.0]).expect("build graph");

    let lower = ArrayD::from_elem(IxDyn(&[1]), 1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 5.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP must succeed");
    let (lo, hi) = output.lower_upper();

    assert!(lo[[0]].is_finite(), "output lower must be finite");
    assert!(hi[[0]].is_finite(), "output upper must be finite");
    // silu(1) ≈ 0.731, silu(5) ≈ 4.966
    assert!(
        lo[[0]] <= 0.731 + 0.01,
        "IBP lower should be <= silu(1)≈0.731, got {}",
        lo[[0]]
    );
    assert!(
        hi[[0]] >= 4.966 - 0.01,
        "IBP upper should be >= silu(5)≈4.966, got {}",
        hi[[0]]
    );
}

#[test]
fn test_silu_mul_ibp_negative_range() {
    // x in [-5, -1] with up = 1. SiLU has a global minimum at x ≈ -1.278.
    // For x in [-5, -1], the output spans from silu(-5)≈-0.034 to silu(-1.278)≈-0.278,
    // with silu(-1) ≈ -0.269. IBP should produce finite bounds containing these.
    let kernel = build_silu_mul_kernel().expect("build K8");
    let graph = kernel_to_graph(&kernel, &[1.0]).expect("build graph");

    let lower = ArrayD::from_elem(IxDyn(&[1]), -5.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), -1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP must succeed");
    let (lo, hi) = output.lower_upper();

    assert!(lo[[0]].is_finite(), "output lower must be finite");
    assert!(hi[[0]].is_finite(), "output upper must be finite");
}

#[test]
fn test_silu_mul_ibp_wide_range() {
    // x in [-10, 10] with up = 2.0. Wide range crossing zero and the
    // SiLU global minimum. Bounds must be finite.
    let kernel = build_silu_mul_kernel().expect("build K8");
    let graph = kernel_to_graph(&kernel, &[2.0]).expect("build graph");

    let lower = ArrayD::from_elem(IxDyn(&[1]), -10.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP must succeed");
    let (lo, hi) = output.lower_upper();

    assert!(lo[[0]].is_finite(), "output lower must be finite");
    assert!(hi[[0]].is_finite(), "output upper must be finite");
}

#[test]
fn test_silu_mul_ibp_negative_up() {
    // x in [0, 5] with up = -3.0. Negative up flips the sign of the output.
    // silu(x) >= 0 for x >= 0, so output = silu(x)*(-3) <= 0.
    let kernel = build_silu_mul_kernel().expect("build K8");
    let graph = kernel_to_graph(&kernel, &[-3.0]).expect("build graph");

    let lower = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 5.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP must succeed");
    let (lo, hi) = output.lower_upper();

    assert!(lo[[0]].is_finite(), "output lower must be finite");
    assert!(hi[[0]].is_finite(), "output upper must be finite");
    // silu(x)*(-3) for x in [0,5]: output in [-14.9, 0]
    assert!(
        lo[[0]] <= 0.0,
        "with negative up, lower bound should be <= 0, got {}",
        lo[[0]]
    );
}

#[test]
fn test_silu_mul_ibp_soundness_vs_reference() {
    // Verify IBP soundness: sample concrete inputs within bounds and check
    // reference output falls within the NY IBP bounds.
    let kernel = build_silu_mul_kernel().expect("build K8");
    let up_val = 2.0f32;
    let graph = kernel_to_graph(&kernel, &[up_val]).expect("build graph");

    // x in [-3, 4]
    let lower = ArrayD::from_elem(IxDyn(&[1]), -3.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 4.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP must succeed");
    let (lo, hi) = output.lower_upper();

    // Sample inputs within [-3, 4]
    let samples: &[f32] = &[-3.0, -2.0, -1.278, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    for &x in samples {
        let val = silu_mul_scalar(x, up_val).unwrap();
        assert!(
            val >= lo[[0]] - 1e-3,
            "silu_mul({x}, {up_val}) = {val} below IBP lower bound {}",
            lo[[0]]
        );
        assert!(
            val <= hi[[0]] + 1e-3,
            "silu_mul({x}, {up_val}) = {val} above IBP upper bound {}",
            hi[[0]]
        );
    }
}

#[test]
fn test_silu_mul_ibp_vs_analytical_bounds() {
    // Compare NY IBP bounds against the analytical bounds function.
    // Analytical bounds should be tighter (they use exact SiLU evaluation),
    // but IBP bounds must still be sound (contain the analytical bounds).
    let kernel = build_silu_mul_kernel().expect("build K8");

    let test_cases: Vec<(f32, f32, f32)> = vec![
        (1.0, 3.0, 1.0),   // positive x, positive up
        (-5.0, 5.0, 2.0),  // symmetric x, positive up
        (0.0, 10.0, -1.0), // positive x, negative up
        (-3.0, -1.0, 5.0), // negative x spanning SiLU minimum
    ];

    for &(x_lo, x_hi, up_val) in &test_cases {
        let graph = kernel_to_graph(&kernel, &[up_val]).expect("build graph");

        let lower = ArrayD::from_elem(IxDyn(&[1]), x_lo);
        let upper = ArrayD::from_elem(IxDyn(&[1]), x_hi);
        let input = BoundedTensor::new(lower, upper).expect("valid bounds");

        let ibp_output = graph.propagate_ibp(&input).expect("IBP must succeed");
        let (ibp_lo, ibp_hi) = ibp_output.lower_upper();

        // Get analytical bounds for comparison (constant up_val → point interval)
        let (up_lo, up_hi) = (up_val, up_val);
        let (anal_lo, anal_hi) =
            silu_mul_scalar_bounds(x_lo, x_hi, up_lo, up_hi).expect("analytical bounds");

        // IBP bounds must contain the analytical bounds (possibly looser).
        assert!(
            ibp_lo[[0]] <= anal_lo + 1e-2,
            "x:[{x_lo},{x_hi}] up={up_val}: IBP lower {} should be <= analytical lower {anal_lo}",
            ibp_lo[[0]]
        );
        assert!(
            ibp_hi[[0]] >= anal_hi - 1e-2,
            "x:[{x_lo},{x_hi}] up={up_val}: IBP upper {} should be >= analytical upper {anal_hi}",
            ibp_hi[[0]]
        );
    }
}

#[test]
fn test_silu_mul_gamma_crown_param_mismatch() {
    // silu_mul has 2 params (x, up). kernel_to_graph expects
    // constant_params.len() == params.len() - 1 == 1. Providing wrong
    // count should error.
    let kernel = build_silu_mul_kernel().expect("build K8");
    let err = kernel_to_graph(&kernel, &[]).expect_err("should reject zero constant params");
    assert!(
        format!("{err:?}").contains("ParamCountMismatch"),
        "expected ParamCountMismatch, got: {err:?}"
    );
}
