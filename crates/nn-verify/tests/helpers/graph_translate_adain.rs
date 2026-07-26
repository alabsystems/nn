// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! K3 AdaIN and K4 AdaIN+Snake fused NY IBP verification tests.
//!
//! Exercises the scalar KernelDef -> NY GraphNetwork translation
//! for both `adain(x, mu, var_val, gamma, beta, eps)` (K3) and
//! `adain_snake(x, mu, var_val, gamma, beta, alpha, eps)` (K4 fused).
//!
//! Part of #19 (K2-K8 kernel ports).

use nn_dsl::adain::{
    adain_scalar, adain_snake_fused_scalar, build_adain_scalar_kernel,
    build_adain_snake_fused_kernel,
};
use nn_verify::{kernel_to_graph, BoundedTensor};
use ndarray::{ArrayD, IxDyn};

// ── K3 AdaIN NY translation ────────────────────────────────

#[test]
fn test_adain_k3_gamma_crown_translation() {
    let kernel = build_adain_scalar_kernel().expect("build K3 AdaIN");
    // x is Variable (param 0); mu, var_val, gamma, beta, eps are constants.
    let graph = kernel_to_graph(&kernel, &[0.0, 1.0, 1.0, 0.0, 1e-5])
        .expect("K3 AdaIN NY graph must build");
    // adain IR: Param(x), Sub(x, mu), AddConst(var+eps), Rsqrt, Mul, MulConst(gamma), AddConst(beta)
    // After constant folding, at least a few nodes.
    assert!(
        graph.num_nodes() >= 3,
        "K3 AdaIN graph needs nodes for sub + rsqrt + mul chain, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_adain_k3_ibp_identity_params() {
    // mu=0, var=1, gamma=1, beta=0, eps=1e-5: AdaIN reduces to identity (approximately).
    // y = 1 * (x - 0) * rsqrt(1 + 1e-5) + 0 ≈ x
    // For x in [1, 5], output should be very close to [1, 5].
    let kernel = build_adain_scalar_kernel().expect("build K3");
    let graph = kernel_to_graph(&kernel, &[0.0, 1.0, 1.0, 0.0, 1e-5]).expect("build graph");

    let lower = ArrayD::from_elem(IxDyn(&[1]), 1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 5.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP must succeed");
    let (lo, hi) = output.lower_upper();

    assert!(lo[[0]].is_finite(), "output lower must be finite");
    assert!(hi[[0]].is_finite(), "output upper must be finite");
    // With identity params, output ≈ x, so bounds should be close to [1, 5].
    assert!(
        lo[[0]] <= 1.1,
        "IBP lower should be <= ~1 for identity params, got {}",
        lo[[0]]
    );
    assert!(
        hi[[0]] >= 4.9,
        "IBP upper should be >= ~5 for identity params, got {}",
        hi[[0]]
    );
}

#[test]
fn test_adain_k3_ibp_with_scale_shift() {
    // gamma=2, beta=3: y = 2*(x - 0)*rsqrt(1+1e-5) + 3 ≈ 2x + 3
    // For x in [0, 5], output ≈ [3, 13].
    let kernel = build_adain_scalar_kernel().expect("build K3");
    let graph = kernel_to_graph(&kernel, &[0.0, 1.0, 2.0, 3.0, 1e-5]).expect("build graph");

    let lower = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 5.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP must succeed");
    let (lo, hi) = output.lower_upper();

    assert!(lo[[0]].is_finite(), "output lower must be finite");
    assert!(hi[[0]].is_finite(), "output upper must be finite");
    // 2*0 + 3 = 3, 2*5 + 3 = 13
    assert!(lo[[0]] <= 3.1, "IBP lower should be <= ~3, got {}", lo[[0]]);
    assert!(
        hi[[0]] >= 12.9,
        "IBP upper should be >= ~13, got {}",
        hi[[0]]
    );
}

#[test]
fn test_adain_k3_ibp_negative_range() {
    // x in [-5, -1], mu=0, var=1, gamma=1, beta=0: output ≈ x
    let kernel = build_adain_scalar_kernel().expect("build K3");
    let graph = kernel_to_graph(&kernel, &[0.0, 1.0, 1.0, 0.0, 1e-5]).expect("build graph");

    let lower = ArrayD::from_elem(IxDyn(&[1]), -5.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), -1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP must succeed");
    let (lo, hi) = output.lower_upper();

    assert!(lo[[0]].is_finite(), "output lower must be finite");
    assert!(hi[[0]].is_finite(), "output upper must be finite");
}

#[test]
fn test_adain_k3_ibp_wide_range() {
    // x in [-10, 10] with typical params: mu=1, var=4, gamma=0.5, beta=-1, eps=1e-5.
    // The rsqrt(4+eps) ≈ 0.5, so y ≈ 0.5 * (x-1) * 0.5 - 1 = 0.25*(x-1) - 1
    let kernel = build_adain_scalar_kernel().expect("build K3");
    let graph = kernel_to_graph(&kernel, &[1.0, 4.0, 0.5, -1.0, 1e-5]).expect("build graph");

    let lower = ArrayD::from_elem(IxDyn(&[1]), -10.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP must succeed");
    let (lo, hi) = output.lower_upper();

    assert!(lo[[0]].is_finite(), "output lower must be finite");
    assert!(hi[[0]].is_finite(), "output upper must be finite");
}

#[test]
fn test_adain_k3_ibp_small_variance() {
    // Small variance (var=0.01) amplifies the normalization. y = gamma*(x-mu)/sqrt(var+eps)
    // For var=0.01, rsqrt ≈ 10, so small x differences are magnified.
    let kernel = build_adain_scalar_kernel().expect("build K3");
    let graph = kernel_to_graph(&kernel, &[0.0, 0.01, 1.0, 0.0, 1e-5]).expect("build graph");

    let lower = ArrayD::from_elem(IxDyn(&[1]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP must succeed");
    let (lo, hi) = output.lower_upper();

    assert!(lo[[0]].is_finite(), "output lower must be finite");
    assert!(hi[[0]].is_finite(), "output upper must be finite");
    // rsqrt(0.01+1e-5) ≈ 10, so x=[-1,1] → output ≈ [-10, 10]
    assert!(
        hi[[0]] - lo[[0]] > 15.0,
        "small variance should amplify bounds, got width {}",
        hi[[0]] - lo[[0]]
    );
}

#[test]
fn test_adain_k3_ibp_soundness_vs_reference() {
    // Verify soundness: sample concrete inputs within bounds and check
    // reference output falls within the NY IBP bounds.
    let kernel = build_adain_scalar_kernel().expect("build K3");
    let mu = 1.0f32;
    let var_val = 2.0f32;
    let gamma = 1.5f32;
    let beta = -0.5f32;
    let eps = 1e-5f32;
    let graph = kernel_to_graph(&kernel, &[mu, var_val, gamma, beta, eps]).expect("build graph");

    // x in [-3, 4]
    let lower = ArrayD::from_elem(IxDyn(&[1]), -3.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 4.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP must succeed");
    let (lo, hi) = output.lower_upper();

    // Sample inputs within [-3, 4]
    let samples: &[f32] = &[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    for &x in samples {
        let val = adain_scalar(x, mu, var_val, gamma, beta, eps)
            .expect("reference should succeed for finite inputs");
        assert!(
            val >= lo[[0]] - 1e-3,
            "adain({x}, mu={mu}, var={var_val}, gamma={gamma}, beta={beta}) = {val} below IBP lower bound {}",
            lo[[0]]
        );
        assert!(
            val <= hi[[0]] + 1e-3,
            "adain({x}, mu={mu}, var={var_val}, gamma={gamma}, beta={beta}) = {val} above IBP upper bound {}",
            hi[[0]]
        );
    }
}

#[test]
fn test_adain_k3_ibp_soundness_multiple_configs() {
    // Verify soundness across multiple parameter configurations to increase
    // confidence in the IBP bounds.
    let kernel = build_adain_scalar_kernel().expect("build K3");

    // (mu, var_val, gamma, beta, eps, x_lo, x_hi)
    let configs: &[(f32, f32, f32, f32, f32, f32, f32)] = &[
        (0.0, 1.0, 1.0, 0.0, 1e-5, -5.0, 5.0),   // identity-like
        (2.0, 0.5, 3.0, -1.0, 1e-5, -3.0, 3.0),  // large gamma, negative beta
        (0.0, 10.0, 0.1, 5.0, 1e-5, 0.0, 10.0),  // large variance, small gamma
        (-1.0, 0.01, 2.0, 0.0, 1e-5, -2.0, 2.0), // small variance (amplifies)
    ];

    for &(mu, var_val, gamma, beta, eps, x_lo, x_hi) in configs {
        let graph =
            kernel_to_graph(&kernel, &[mu, var_val, gamma, beta, eps]).expect("build graph");

        let lower = ArrayD::from_elem(IxDyn(&[1]), x_lo);
        let upper = ArrayD::from_elem(IxDyn(&[1]), x_hi);
        let input = BoundedTensor::new(lower, upper).expect("valid bounds");

        let output = graph.propagate_ibp(&input).expect("IBP must succeed");
        let (lo, hi) = output.lower_upper();

        // Sample at endpoints and midpoint
        for &x in &[x_lo, x_hi, f32::midpoint(x_lo, x_hi)] {
            let val =
                adain_scalar(x, mu, var_val, gamma, beta, eps).expect("reference should succeed");
            assert!(
                val >= lo[[0]] - 1e-2,
                "config mu={mu} var={var_val} gamma={gamma} beta={beta}: adain({x}) = {val} below IBP lower {}",
                lo[[0]]
            );
            assert!(
                val <= hi[[0]] + 1e-2,
                "config mu={mu} var={var_val} gamma={gamma} beta={beta}: adain({x}) = {val} above IBP upper {}",
                hi[[0]]
            );
        }
    }
}

#[test]
fn test_adain_k3_gamma_crown_param_mismatch() {
    // adain has 6 params (x, mu, var_val, gamma, beta, eps). kernel_to_graph
    // expects constant_params.len() == 5. Providing wrong count should error.
    let kernel = build_adain_scalar_kernel().expect("build K3");
    let err = kernel_to_graph(&kernel, &[]).expect_err("should reject zero constant params");
    assert!(
        format!("{err:?}").contains("ParamCountMismatch"),
        "expected ParamCountMismatch, got: {err:?}"
    );
}

// ── K4 AdaIN+Snake fused NY translation ────────────────────

#[test]
fn test_adain_snake_k4_gamma_crown_translation() {
    let kernel = build_adain_snake_fused_kernel().expect("build K4 fused");
    // x is Variable (param 0); mu, var_val, gamma, beta, alpha, eps are constants.
    let graph = kernel_to_graph(&kernel, &[0.0, 1.0, 1.0, 0.0, 1.0, 1e-5])
        .expect("K4 fused NY graph must build");
    // Fused kernel has AdaIN + Snake: sub, rsqrt, mul chain, then sin, powi, recip, mul, add.
    // After constant folding, should have several nodes.
    assert!(
        graph.num_nodes() >= 5,
        "K4 fused graph needs nodes for AdaIN + Snake chain, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_adain_snake_k4_ibp_identity_adain() {
    // AdaIN with identity params (mu=0, var=1, gamma=1, beta=0) reduces to x,
    // then Snake: x + (1/alpha)*sin^2(alpha*x). With alpha=1:
    // snake(x) = x + sin^2(x), so for x in [1, 3], output in [1+sin^2(1), 3+sin^2(3)].
    let kernel = build_adain_snake_fused_kernel().expect("build K4");
    let graph = kernel_to_graph(&kernel, &[0.0, 1.0, 1.0, 0.0, 1.0, 1e-5]).expect("build graph");

    let lower = ArrayD::from_elem(IxDyn(&[1]), 1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 3.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP must succeed");
    let (lo, hi) = output.lower_upper();

    assert!(lo[[0]].is_finite(), "output lower must be finite");
    assert!(hi[[0]].is_finite(), "output upper must be finite");
    // snake(x) >= x for all x (sin^2 >= 0), so lower bound >= 1
    assert!(
        lo[[0]] >= 0.9,
        "fused output lower should be >= ~1, got {}",
        lo[[0]]
    );
}

#[test]
fn test_adain_snake_k4_ibp_negative_range() {
    // x in [-5, -1], identity AdaIN, alpha=1.
    let kernel = build_adain_snake_fused_kernel().expect("build K4");
    let graph = kernel_to_graph(&kernel, &[0.0, 1.0, 1.0, 0.0, 1.0, 1e-5]).expect("build graph");

    let lower = ArrayD::from_elem(IxDyn(&[1]), -5.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), -1.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP must succeed");
    let (lo, hi) = output.lower_upper();

    assert!(lo[[0]].is_finite(), "output lower must be finite");
    assert!(hi[[0]].is_finite(), "output upper must be finite");
}

#[test]
fn test_adain_snake_k4_ibp_wide_range() {
    // x in [-10, 10] with realistic params.
    let kernel = build_adain_snake_fused_kernel().expect("build K4");
    let graph = kernel_to_graph(&kernel, &[1.0, 4.0, 0.5, -1.0, 2.0, 1e-5]).expect("build graph");

    let lower = ArrayD::from_elem(IxDyn(&[1]), -10.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP must succeed");
    let (lo, hi) = output.lower_upper();

    assert!(lo[[0]].is_finite(), "output lower must be finite");
    assert!(hi[[0]].is_finite(), "output upper must be finite");
}

#[test]
fn test_adain_snake_k4_ibp_high_alpha() {
    // High alpha means higher-frequency snake oscillation, but the sin^2/alpha
    // amplitude decreases. With alpha=50, snake contribution is small.
    let kernel = build_adain_snake_fused_kernel().expect("build K4");
    let graph = kernel_to_graph(&kernel, &[0.0, 1.0, 1.0, 0.0, 50.0, 1e-5]).expect("build graph");

    let lower = ArrayD::from_elem(IxDyn(&[1]), -5.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 5.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP must succeed");
    let (lo, hi) = output.lower_upper();

    assert!(lo[[0]].is_finite(), "output lower must be finite");
    assert!(hi[[0]].is_finite(), "output upper must be finite");
}

#[test]
fn test_adain_snake_k4_ibp_soundness_vs_reference() {
    // Verify soundness: sample concrete inputs within bounds and check
    // reference output falls within the NY IBP bounds.
    let kernel = build_adain_snake_fused_kernel().expect("build K4");
    let mu = 0.5f32;
    let var_val = 2.0f32;
    let gamma = 1.0f32;
    let beta = 0.0f32;
    let alpha = 1.0f32;
    let eps = 1e-5f32;
    let graph =
        kernel_to_graph(&kernel, &[mu, var_val, gamma, beta, alpha, eps]).expect("build graph");

    let lower = ArrayD::from_elem(IxDyn(&[1]), -3.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 4.0f32);
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = graph.propagate_ibp(&input).expect("IBP must succeed");
    let (lo, hi) = output.lower_upper();

    let samples: &[f32] = &[-3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0];
    for &x in samples {
        let val = adain_snake_fused_scalar(x, mu, var_val, gamma, beta, alpha, eps)
            .expect("reference should succeed");
        assert!(
            val >= lo[[0]] - 1e-3,
            "fused({x}) = {val} below IBP lower bound {}",
            lo[[0]]
        );
        assert!(
            val <= hi[[0]] + 1e-3,
            "fused({x}) = {val} above IBP upper bound {}",
            hi[[0]]
        );
    }
}

#[test]
fn test_adain_snake_k4_ibp_soundness_multiple_configs() {
    // Verify soundness across multiple parameter configurations.
    let kernel = build_adain_snake_fused_kernel().expect("build K4");

    // (mu, var_val, gamma, beta, alpha, eps, x_lo, x_hi)
    let configs: &[(f32, f32, f32, f32, f32, f32, f32, f32)] = &[
        (0.0, 1.0, 1.0, 0.0, 1.0, 1e-5, -5.0, 5.0),  // baseline
        (2.0, 0.5, 2.0, -1.0, 0.5, 1e-5, -3.0, 3.0), // shifted mu, low alpha
        (0.0, 4.0, 0.5, 3.0, 10.0, 1e-5, 0.0, 8.0),  // large variance, high alpha
        (-1.0, 0.1, 1.0, 0.0, 5.0, 1e-5, -2.0, 2.0), // small variance
    ];

    for &(mu, var_val, gamma, beta, alpha, eps, x_lo, x_hi) in configs {
        let graph =
            kernel_to_graph(&kernel, &[mu, var_val, gamma, beta, alpha, eps]).expect("build graph");

        let lower = ArrayD::from_elem(IxDyn(&[1]), x_lo);
        let upper = ArrayD::from_elem(IxDyn(&[1]), x_hi);
        let input = BoundedTensor::new(lower, upper).expect("valid bounds");

        let output = graph.propagate_ibp(&input).expect("IBP must succeed");
        let (lo, hi) = output.lower_upper();

        for &x in &[x_lo, x_hi, f32::midpoint(x_lo, x_hi)] {
            let val = adain_snake_fused_scalar(x, mu, var_val, gamma, beta, alpha, eps)
                .expect("reference should succeed");
            assert!(
                val >= lo[[0]] - 1e-2,
                "config mu={mu} var={var_val}: fused({x}) = {val} below IBP lower {}",
                lo[[0]]
            );
            assert!(
                val <= hi[[0]] + 1e-2,
                "config mu={mu} var={var_val}: fused({x}) = {val} above IBP upper {}",
                hi[[0]]
            );
        }
    }
}

#[test]
fn test_adain_snake_k4_gamma_crown_param_mismatch() {
    // K4 fused has 7 params. kernel_to_graph expects 6 constant params.
    let kernel = build_adain_snake_fused_kernel().expect("build K4");
    let err = kernel_to_graph(&kernel, &[]).expect_err("should reject zero constant params");
    assert!(
        format!("{err:?}").contains("ParamCountMismatch"),
        "expected ParamCountMismatch, got: {err:?}"
    );
}
