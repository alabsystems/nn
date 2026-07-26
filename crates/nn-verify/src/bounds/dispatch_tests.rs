// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for BOUNDS_REGISTRY dispatch wiring in compute_output_bounds_heuristic.
//!
//! Extracted from `ay/prove_tests_bounds_dispatch.rs` (#859) to run without
//! `ay-smt` feature flag.

use super::compute_output_bounds_heuristic;
use crate::bounds::{
    adain_output_bounds, exp_output_bounds, gelu_output_bounds, instance_norm_output_bounds,
    leaky_relu_output_bounds, norm_affine_output_bounds, relu_output_bounds,
    rms_norm_scalar_output_bounds, rope_output_bounds, sigmoid_output_bounds,
    silu_mul_output_bounds, snake_output_bounds, softplus_output_bounds, tanh_output_bounds,
};
use nn_dsl::ir::{IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType};

/// Helper: build a minimal KernelDef with the given name and param count.
fn stub_kernel(name: &str, n_params: usize) -> KernelDef {
    let mut params = Vec::with_capacity(n_params);
    for i in 0..n_params {
        params.push(Param::new(format!("p{i}"), ScalarType::F32));
    }
    KernelDef::new(
        name,
        params,
        ScalarType::F32,
        vec![IRNode::new(NodeId::new(0), IRNodeKind::Param(0))],
        NodeId::new(0),
    )
}

const TOL: f64 = 1e-6;

// ======================== snake ========================

#[test]
fn test_dispatch_snake() {
    let kernel = stub_kernel("snake", 2);
    let alpha = 1.0_f32;
    let (lo, hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[alpha], -1.0, 1.0).unwrap();
    assert!(!is_heuristic, "snake should dispatch to analytical bounds");
    let (exp_lo, exp_hi) = snake_output_bounds(-1.0, 1.0, f64::from(alpha)).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
}

// ======================== silu_mul ========================

#[test]
fn test_dispatch_silu_mul() {
    let kernel = stub_kernel("silu_mul", 2);
    let up = 2.0_f32;
    let (lo, hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[up], -1.0, 1.0).unwrap();
    assert!(
        !is_heuristic,
        "silu_mul should dispatch to analytical bounds"
    );
    let (exp_lo, exp_hi) = silu_mul_output_bounds(f64::from(up), -1.0, 1.0).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
}

// ======================== rope_cos ========================

#[test]
fn test_dispatch_rope_cos() {
    let kernel = stub_kernel("rope_cos", 3);
    let x1 = 0.5_f32;
    let freq = 1.0_f32;
    let (lo, hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[x1, freq], -1.0, 1.0).unwrap();
    assert!(
        !is_heuristic,
        "rope_cos should dispatch to analytical bounds"
    );
    let (exp_lo, exp_hi) =
        rope_output_bounds(x1, freq, -1.0, 1.0, nn_dsl::rope_cos_scalar_bounds).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
}

// ======================== rope_sin ========================

#[test]
fn test_dispatch_rope_sin() {
    let kernel = stub_kernel("rope_sin", 3);
    let x1 = 0.5_f32;
    let freq = 1.0_f32;
    let (lo, hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[x1, freq], -1.0, 1.0).unwrap();
    assert!(
        !is_heuristic,
        "rope_sin should dispatch to analytical bounds"
    );
    let (exp_lo, exp_hi) =
        rope_output_bounds(x1, freq, -1.0, 1.0, nn_dsl::rope_sin_scalar_bounds).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
}

// ======================== rms_norm_scalar ========================

#[test]
fn test_dispatch_rms_norm_scalar() {
    let kernel = stub_kernel("rms_norm_scalar", 3);
    let rms_inv = 0.5_f32;
    let weight = 1.0_f32;
    let (lo, hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[rms_inv, weight], -1.0, 1.0).unwrap();
    assert!(
        !is_heuristic,
        "rms_norm_scalar should dispatch to analytical bounds"
    );
    let (exp_lo, exp_hi) = rms_norm_scalar_output_bounds(rms_inv, weight, -1.0, 1.0).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
}

// ======================== layer_norm_scalar ========================

#[test]
fn test_dispatch_layer_norm_scalar() {
    let kernel = stub_kernel("layer_norm_scalar", 6);
    let mean = 0.0_f32;
    let var = 1.0_f32;
    let eps = 1e-5_f32;
    let gamma = 1.0_f32;
    let beta = 0.0_f32;
    let (lo, hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[mean, var, eps, gamma, beta], -1.0, 1.0)
            .unwrap();
    assert!(
        !is_heuristic,
        "layer_norm_scalar should dispatch to analytical bounds"
    );
    let (exp_lo, exp_hi) =
        norm_affine_output_bounds(mean, var, eps, gamma, beta, -1.0, 1.0).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
}

// ======================== instance_norm_scalar ========================

#[test]
fn test_dispatch_instance_norm_scalar() {
    let kernel = stub_kernel("instance_norm_scalar", 4);
    let mean = 0.0_f32;
    let var = 1.0_f32;
    let eps = 1e-5_f32;
    let (lo, hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[mean, var, eps], -1.0, 1.0).unwrap();
    assert!(
        !is_heuristic,
        "instance_norm_scalar should dispatch to analytical bounds"
    );
    let (exp_lo, exp_hi) = instance_norm_output_bounds(mean, var, eps, -1.0, 1.0).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
}

// ======================== instance_norm_affine_scalar ========================

#[test]
fn test_dispatch_instance_norm_affine_scalar() {
    let kernel = stub_kernel("instance_norm_affine_scalar", 6);
    let mean = 0.0_f32;
    let var = 1.0_f32;
    let eps = 1e-5_f32;
    let gamma = 1.0_f32;
    let beta = 0.0_f32;
    let (lo, hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[mean, var, eps, gamma, beta], -1.0, 1.0)
            .unwrap();
    assert!(
        !is_heuristic,
        "instance_norm_affine_scalar should dispatch to analytical bounds"
    );
    let (exp_lo, exp_hi) =
        norm_affine_output_bounds(mean, var, eps, gamma, beta, -1.0, 1.0).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
}

// ======================== adain ========================

#[test]
fn test_dispatch_adain() {
    let kernel = stub_kernel("adain", 6);
    let mu = 0.0_f32;
    let var = 1.0_f32;
    let gamma = 1.0_f32;
    let beta = 0.0_f32;
    let eps = 1e-5_f32;
    let (lo, hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[mu, var, gamma, beta, eps], -1.0, 1.0).unwrap();
    assert!(!is_heuristic, "adain should dispatch to analytical bounds");
    let (exp_lo, exp_hi) = adain_output_bounds(mu, var, gamma, beta, eps, -1.0, 1.0).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
}

// ======================== adain_snake ========================

#[test]
fn test_dispatch_adain_snake() {
    let kernel = stub_kernel("adain_snake", 7);
    let mu = 0.0_f32;
    let var = 1.0_f32;
    let gamma = 1.0_f32;
    let beta = 0.0_f32;
    let alpha = 1.0_f32;
    let eps = 1e-5_f32;
    let (lo, hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[mu, var, gamma, beta, alpha, eps], -1.0, 1.0)
            .unwrap();
    assert!(
        !is_heuristic,
        "adain_snake should dispatch to analytical bounds"
    );
    let (adain_lo, adain_hi) = adain_output_bounds(mu, var, gamma, beta, eps, -1.0, 1.0).unwrap();
    let alpha_clamped = f64::from(alpha).max(f64::from(nn_dsl::snake::SNAKE_MIN_ALPHA));
    let (exp_lo, exp_hi) = snake_output_bounds(adain_lo, adain_hi, alpha_clamped).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
}

// ======================== gelu ========================

#[test]
fn test_dispatch_gelu() {
    let kernel = stub_kernel("gelu", 1);
    let (lo, hi, is_heuristic) = compute_output_bounds_heuristic(&kernel, &[], -5.0, 5.0).unwrap();
    assert!(!is_heuristic, "gelu should dispatch to analytical bounds");
    let (exp_lo, exp_hi) = gelu_output_bounds(-5.0, 5.0).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
}

#[test]
fn test_dispatch_gelu_spanning_minimum() {
    let kernel = stub_kernel("gelu", 1);
    let (lo, hi, is_heuristic) = compute_output_bounds_heuristic(&kernel, &[], -2.0, 2.0).unwrap();
    assert!(!is_heuristic, "gelu should dispatch to analytical bounds");
    assert!(lo < 0.0, "gelu lower bound should be negative for [-2, 2]");
    assert!(
        lo > -0.2,
        "gelu lower bound should be > -0.2 (minimum is ~-0.170)"
    );
    assert!(hi > 1.9, "gelu upper bound should be > 1.9 for [-2, 2]");
}

// ======================== sigmoid ========================

#[test]
fn test_dispatch_sigmoid() {
    let kernel = stub_kernel("sigmoid", 1);
    let (lo, hi, is_heuristic) = compute_output_bounds_heuristic(&kernel, &[], -5.0, 5.0).unwrap();
    assert!(
        !is_heuristic,
        "sigmoid should dispatch to analytical bounds"
    );
    let (exp_lo, exp_hi) = sigmoid_output_bounds(-5.0, 5.0).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
}

#[test]
fn test_dispatch_sigmoid_monotonic_bounds() {
    let kernel = stub_kernel("sigmoid", 1);
    let (lo, hi, is_heuristic) = compute_output_bounds_heuristic(&kernel, &[], -5.0, 5.0).unwrap();
    assert!(
        !is_heuristic,
        "sigmoid should dispatch to analytical bounds"
    );
    assert!(lo > 0.0, "sigmoid lower bound must be > 0");
    assert!(lo < 0.01, "sigmoid lower bound should be < 0.01 for x=-5");
    assert!(hi > 0.99, "sigmoid upper bound should be > 0.99 for x=5");
    assert!(hi < 1.0, "sigmoid upper bound must be < 1");
}

// ======================== Fallback (unregistered kernel) ========================

#[test]
fn test_dispatch_unknown_kernel_uses_heuristic() {
    let kernel = stub_kernel("unknown_kernel_xyz", 2);
    let (_lo, _hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[1.0], -5.0, 5.0).unwrap();
    assert!(
        is_heuristic,
        "unregistered kernel should fall through to ±1e6 heuristic"
    );
}

// ======================== Insufficient params → heuristic fallback ========================

#[test]
fn test_dispatch_snake_insufficient_params_uses_heuristic() {
    let kernel = stub_kernel("snake", 2);
    let (_lo, _hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[], -1.0, 1.0).unwrap();
    assert!(
        is_heuristic,
        "snake with 0 constant params should fall through to heuristic"
    );
}

#[test]
fn test_dispatch_norm_affine_insufficient_params_uses_heuristic() {
    let kernel = stub_kernel("layer_norm_scalar", 6);
    let (_lo, _hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[0.0, 1.0, 1e-5], -1.0, 1.0).unwrap();
    assert!(
        is_heuristic,
        "layer_norm_scalar with 3 constant params should fall through to heuristic"
    );
}

#[test]
fn test_dispatch_adain_snake_insufficient_params_uses_heuristic() {
    let kernel = stub_kernel("adain_snake", 7);
    let (_lo, _hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[0.0, 1.0, 1.0, 0.0], -1.0, 1.0).unwrap();
    assert!(
        is_heuristic,
        "adain_snake with 4 constant params should fall through to heuristic"
    );
}

// ======================== relu ========================

#[test]
fn test_dispatch_relu() {
    let kernel = stub_kernel("relu", 1);
    let (lo, hi, is_heuristic) = compute_output_bounds_heuristic(&kernel, &[], -5.0, 5.0).unwrap();
    assert!(!is_heuristic, "relu should dispatch to analytical bounds");
    let (exp_lo, exp_hi) = relu_output_bounds(-5.0, 5.0).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
    assert!((lo - 0.0).abs() < TOL, "relu lo should be 0");
    assert!((hi - 5.0).abs() < TOL, "relu hi should be 5");
}

// ======================== tanh_act ========================

#[test]
fn test_dispatch_tanh_act() {
    let kernel = stub_kernel("tanh_act", 1);
    let (lo, hi, is_heuristic) = compute_output_bounds_heuristic(&kernel, &[], -5.0, 5.0).unwrap();
    assert!(
        !is_heuristic,
        "tanh_act should dispatch to analytical bounds"
    );
    let (exp_lo, exp_hi) = tanh_output_bounds(-5.0, 5.0).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
    assert!(lo < -0.999 && lo > -1.0, "tanh lo ∈ (-1, -0.999)");
    assert!(hi > 0.999 && hi < 1.0, "tanh hi ∈ (0.999, 1)");
}

// ======================== leaky_relu ========================

#[test]
fn test_dispatch_leaky_relu() {
    let kernel = stub_kernel("leaky_relu", 2);
    let slope = 0.01_f32;
    let (lo, hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[slope], -5.0, 5.0).unwrap();
    assert!(
        !is_heuristic,
        "leaky_relu should dispatch to analytical bounds"
    );
    let (expected_lo, expected_hi) = leaky_relu_output_bounds(f64::from(slope), -5.0, 5.0).unwrap();
    assert!((lo - expected_lo).abs() < TOL, "lo: {lo} != {expected_lo}");
    assert!((hi - expected_hi).abs() < TOL, "hi: {hi} != {expected_hi}");
}

#[test]
fn test_dispatch_leaky_relu_mixed_range() {
    let kernel = stub_kernel("leaky_relu", 2);
    let slope = 0.1_f32;
    let (lo, hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[slope], -3.0, 7.0).unwrap();
    assert!(!is_heuristic);
    // leaky_relu(-3, 0.1) = -0.3, leaky_relu(7, 0.1) = 7
    assert!((lo - (-0.3)).abs() < TOL, "lo should be -0.3, got {lo}");
    assert!((hi - 7.0).abs() < TOL, "hi should be 7.0, got {hi}");
}

// ======================== exp ========================

#[test]
fn test_dispatch_exp() {
    let kernel = stub_kernel("exp", 1);
    let (lo, hi, is_heuristic) = compute_output_bounds_heuristic(&kernel, &[], -5.0, 5.0).unwrap();
    assert!(!is_heuristic, "exp should dispatch to analytical bounds");
    let (expected_lo, expected_hi) = exp_output_bounds(-5.0, 5.0).unwrap();
    assert!((lo - expected_lo).abs() < TOL, "lo: {lo} != {expected_lo}");
    assert!((hi - expected_hi).abs() < TOL, "hi: {hi} != {expected_hi}");
    assert!(lo > 0.0, "exp output must be positive");
}

#[test]
fn test_dispatch_exp_negative_range() {
    let kernel = stub_kernel("exp", 1);
    let (lo, hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[], -10.0, -1.0).unwrap();
    assert!(!is_heuristic);
    // exp(-10) ~ 4.54e-5, exp(-1) ~ 0.368
    assert!(lo > 0.0 && lo < 0.001, "exp(-10) near 0, got {lo}");
    assert!(hi > 0.36 && hi < 0.37, "exp(-1) ~ 0.368, got {hi}");
}

// ======================== softplus ========================

#[test]
fn test_dispatch_softplus() {
    let kernel = stub_kernel("softplus", 1);
    let (lo, hi, is_heuristic) = compute_output_bounds_heuristic(&kernel, &[], -5.0, 5.0).unwrap();
    assert!(
        !is_heuristic,
        "softplus should dispatch to analytical bounds"
    );
    let (expected_lo, expected_hi) = softplus_output_bounds(-5.0, 5.0).unwrap();
    assert!((lo - expected_lo).abs() < TOL, "lo: {lo} != {expected_lo}");
    assert!((hi - expected_hi).abs() < TOL, "hi: {hi} != {expected_hi}");
    assert!(lo > 0.0, "softplus output must be positive");
}

#[test]
fn test_dispatch_softplus_large_positive() {
    let kernel = stub_kernel("softplus", 1);
    let (lo, hi, is_heuristic) = compute_output_bounds_heuristic(&kernel, &[], 10.0, 20.0).unwrap();
    assert!(!is_heuristic);
    // softplus(x) ≈ x for large x
    assert!((lo - 10.0).abs() < 0.001, "softplus(10) ≈ 10, got {lo}");
    assert!((hi - 20.0).abs() < 0.001, "softplus(20) ≈ 20, got {hi}");
}

// ======================== adain_leaky_relu ========================

#[test]
fn test_dispatch_adain_leaky_relu() {
    let kernel = stub_kernel("adain_leaky_relu", 7);
    let mu = 0.0_f32;
    let var = 1.0_f32;
    let gamma = 1.0_f32;
    let beta = 0.0_f32;
    let slope = 0.01_f32;
    let eps = 1e-5_f32;
    // constant_params: [mu, var_val, gamma, beta, slope, eps]
    let (lo, hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[mu, var, gamma, beta, slope, eps], -1.0, 1.0)
            .unwrap();
    assert!(
        !is_heuristic,
        "adain_leaky_relu should dispatch to analytical bounds"
    );
    // Compose: adain → leaky_relu
    let (adain_lo, adain_hi) = adain_output_bounds(mu, var, gamma, beta, eps, -1.0, 1.0).unwrap();
    let (exp_lo, exp_hi) = leaky_relu_output_bounds(f64::from(slope), adain_lo, adain_hi).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
}

#[test]
fn test_dispatch_adain_leaky_relu_negative_output() {
    let kernel = stub_kernel("adain_leaky_relu", 7);
    // beta=-5 shifts output negative, so leaky_relu scaling matters
    let mu = 0.0_f32;
    let var = 1.0_f32;
    let gamma = 1.0_f32;
    let beta = -5.0_f32;
    let slope = 0.2_f32;
    let eps = 1e-5_f32;
    let (lo, _hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[mu, var, gamma, beta, slope, eps], -1.0, 1.0)
            .unwrap();
    assert!(!is_heuristic);
    // With beta=-5, adain output ∈ [-6, -4] approx, so leaky_relu scales by 0.2
    assert!(
        lo < 0.0,
        "with large negative beta, output should be negative"
    );
}

#[test]
fn test_dispatch_adain_leaky_relu_insufficient_params_uses_heuristic() {
    let kernel = stub_kernel("adain_leaky_relu", 7);
    let (_lo, _hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[0.0, 1.0, 1.0, 0.0, 0.01], -1.0, 1.0).unwrap();
    assert!(
        is_heuristic,
        "adain_leaky_relu with 5 constant params should fall through to heuristic"
    );
}

// ======================== ada_layer_norm ========================

#[test]
fn test_dispatch_ada_layer_norm() {
    let kernel = stub_kernel("ada_layer_norm", 8);
    let mean = 0.0_f32;
    let var = 1.0_f32;
    let eps = 1e-5_f32;
    let norm_weight = 1.0_f32;
    let norm_bias = 0.0_f32;
    let gamma = 0.5_f32;
    let beta = 0.1_f32;
    // constant_params: [mean, var_val, eps, norm_weight, norm_bias, gamma, beta]
    let (lo, hi, is_heuristic) = compute_output_bounds_heuristic(
        &kernel,
        &[mean, var, eps, norm_weight, norm_bias, gamma, beta],
        -1.0,
        1.0,
    )
    .unwrap();
    assert!(
        !is_heuristic,
        "ada_layer_norm should dispatch to analytical bounds"
    );
    // Compose: norm_affine → adaptive_affine
    let (norm_lo, norm_hi) =
        norm_affine_output_bounds(mean, var, eps, norm_weight, norm_bias, -1.0, 1.0).unwrap();
    let scale = 1.0 + f64::from(gamma);
    let beta_f64 = f64::from(beta);
    let a = scale * norm_lo + beta_f64;
    let b = scale * norm_hi + beta_f64;
    let (exp_lo, exp_hi) = if a <= b { (a, b) } else { (b, a) };
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
}

#[test]
fn test_dispatch_ada_layer_norm_negative_gamma() {
    let kernel = stub_kernel("ada_layer_norm", 8);
    // gamma=-1.5 makes scale=(1+(-1.5))=-0.5, flipping the interval
    let mean = 0.0_f32;
    let var = 1.0_f32;
    let eps = 1e-5_f32;
    let norm_weight = 1.0_f32;
    let norm_bias = 0.0_f32;
    let gamma = -1.5_f32;
    let beta = 0.0_f32;
    let (lo, hi, is_heuristic) = compute_output_bounds_heuristic(
        &kernel,
        &[mean, var, eps, norm_weight, norm_bias, gamma, beta],
        -2.0,
        2.0,
    )
    .unwrap();
    assert!(!is_heuristic);
    // scale = -0.5, so interval flips: positive norm output → negative ada output
    assert!(lo < 0.0, "negative gamma should flip sign");
    assert!(hi > 0.0, "symmetric input should give symmetric-ish output");
}

#[test]
fn test_dispatch_ada_layer_norm_insufficient_params_uses_heuristic() {
    let kernel = stub_kernel("ada_layer_norm", 8);
    let (_lo, _hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[0.0, 1.0, 1e-5, 1.0, 0.0, 0.5], -1.0, 1.0)
            .unwrap();
    assert!(
        is_heuristic,
        "ada_layer_norm with 6 constant params should fall through to heuristic"
    );
}
