// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for BOUNDS_REGISTRY dispatch wiring in compute_output_bounds_heuristic.
//!
//! Each test constructs a KernelDef matching a registry entry, calls
//! compute_output_bounds_heuristic directly with valid inputs, and asserts:
//! 1. `is_heuristic == false` (registry matched, not ±1e6 fallback)
//! 2. Returned (lo, hi) match the expected analytical bounds
//!
//! Filed as #470. Catches wiring errors (wrong function, wrong min_constant_params,
//! wrong coefficient) that indirect integration tests miss.

use super::prove_dispatch::compute_output_bounds_heuristic;
use nn_dsl::ir::{IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType};

/// Helper: build a minimal KernelDef with the given name and param count.
/// The IR is trivial (identity on first param) — the registry dispatches
/// solely on name and constant_params length, not on IR structure.
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
    let kernel = stub_kernel("snake", 2); // x + alpha
    let alpha = 1.0_f32;
    let (lo, hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[alpha], -1.0, 1.0).unwrap();
    assert!(!is_heuristic, "snake should dispatch to analytical bounds");
    // Cross-check against snake_uf::snake_output_bounds
    let (exp_lo, exp_hi) =
        crate::ay::snake_uf::snake_output_bounds(-1.0, 1.0, f64::from(alpha)).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
}

// ======================== silu_mul ========================

#[test]
fn test_dispatch_silu_mul() {
    let kernel = stub_kernel("silu_mul", 2); // x + up
    let up = 2.0_f32;
    let (lo, hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[up], -1.0, 1.0).unwrap();
    assert!(
        !is_heuristic,
        "silu_mul should dispatch to analytical bounds"
    );
    let (exp_lo, exp_hi) =
        super::prove_dispatch::silu_mul_output_bounds(f64::from(up), -1.0, 1.0).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
}

// ======================== rope_cos ========================

#[test]
fn test_dispatch_rope_cos() {
    let kernel = stub_kernel("rope_cos", 3); // x0 + x1 + freq
    let x1 = 0.5_f32;
    let freq = 1.0_f32;
    let (lo, hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[x1, freq], -1.0, 1.0).unwrap();
    assert!(
        !is_heuristic,
        "rope_cos should dispatch to analytical bounds"
    );
    let (exp_lo, exp_hi) = super::prove_dispatch::rope_output_bounds(
        x1,
        freq,
        -1.0,
        1.0,
        nn_dsl::rope_cos_scalar_bounds,
    )
    .unwrap();
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
    let (exp_lo, exp_hi) = super::prove_dispatch::rope_output_bounds(
        x1,
        freq,
        -1.0,
        1.0,
        nn_dsl::rope_sin_scalar_bounds,
    )
    .unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
}

// ======================== rms_norm_scalar ========================

#[test]
fn test_dispatch_rms_norm_scalar() {
    let kernel = stub_kernel("rms_norm_scalar", 3); // x + rms_inv + weight
    let rms_inv = 0.5_f32;
    let weight = 1.0_f32;
    let (lo, hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[rms_inv, weight], -1.0, 1.0).unwrap();
    assert!(
        !is_heuristic,
        "rms_norm_scalar should dispatch to analytical bounds"
    );
    let (exp_lo, exp_hi) =
        super::prove_dispatch::rms_norm_scalar_output_bounds(rms_inv, weight, -1.0, 1.0).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
}

// ======================== layer_norm_scalar ========================

#[test]
fn test_dispatch_layer_norm_scalar() {
    let kernel = stub_kernel("layer_norm_scalar", 6);
    // Kernel param order after x: (mean, var, eps, gamma, beta)
    // constant_params[i] = kernel param i+1 (#448)
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
        super::prove_dispatch::norm_affine_output_bounds(mean, var, eps, gamma, beta, -1.0, 1.0)
            .unwrap();
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
    let (exp_lo, exp_hi) =
        super::prove_dispatch::instance_norm_output_bounds(mean, var, eps, -1.0, 1.0).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
}

// ======================== instance_norm_affine_scalar ========================

#[test]
fn test_dispatch_instance_norm_affine_scalar() {
    let kernel = stub_kernel("instance_norm_affine_scalar", 6);
    // Kernel param order after x: (mean, var, eps, gamma, beta) — same as layer_norm
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
    // Uses same norm_affine_output_bounds as layer_norm
    let (exp_lo, exp_hi) =
        super::prove_dispatch::norm_affine_output_bounds(mean, var, eps, gamma, beta, -1.0, 1.0)
            .unwrap();
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
    let (exp_lo, exp_hi) =
        super::prove_dispatch::adain_output_bounds(mu, var, gamma, beta, eps, -1.0, 1.0).unwrap();
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
    // cp order: mu, var, gamma, beta, alpha, eps (6 constant params)
    let (lo, hi, is_heuristic) =
        compute_output_bounds_heuristic(&kernel, &[mu, var, gamma, beta, alpha, eps], -1.0, 1.0)
            .unwrap();
    assert!(
        !is_heuristic,
        "adain_snake should dispatch to analytical bounds"
    );
    // Cross-check: adain_output_bounds then snake_output_bounds
    let (adain_lo, adain_hi) =
        super::prove_dispatch::adain_output_bounds(mu, var, gamma, beta, eps, -1.0, 1.0).unwrap();
    let alpha_clamped = f64::from(alpha).max(f64::from(nn_dsl::snake::SNAKE_MIN_ALPHA));
    let (exp_lo, exp_hi) =
        crate::ay::snake_uf::snake_output_bounds(adain_lo, adain_hi, alpha_clamped).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
}

// ======================== gelu ========================

#[test]
fn test_dispatch_gelu() {
    let kernel = stub_kernel("gelu", 1); // x only (0 constant params)
                                         // GELU has 0 constant params — empty slice.
    let (lo, hi, is_heuristic) = compute_output_bounds_heuristic(&kernel, &[], -5.0, 5.0).unwrap();
    assert!(!is_heuristic, "gelu should dispatch to analytical bounds");
    let (exp_lo, exp_hi) = super::prove_dispatch::gelu_output_bounds(-5.0, 5.0).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
}

#[test]
fn test_dispatch_gelu_spanning_minimum() {
    // GELU has a global minimum at x ≈ -0.752. This range spans it.
    let kernel = stub_kernel("gelu", 1);
    let (lo, hi, is_heuristic) = compute_output_bounds_heuristic(&kernel, &[], -2.0, 2.0).unwrap();
    assert!(!is_heuristic, "gelu should dispatch to analytical bounds");
    // Lower bound should be near the GELU minimum value (~-0.170).
    assert!(lo < 0.0, "gelu lower bound should be negative for [-2, 2]");
    assert!(
        lo > -0.2,
        "gelu lower bound should be > -0.2 (minimum is ~-0.170)"
    );
    // Upper bound should be near gelu(2) ≈ 1.9545.
    assert!(hi > 1.9, "gelu upper bound should be > 1.9 for [-2, 2]");
}

// ======================== sigmoid ========================

#[test]
fn test_dispatch_sigmoid() {
    let kernel = stub_kernel("sigmoid", 1); // x only (0 constant params)
    let (lo, hi, is_heuristic) = compute_output_bounds_heuristic(&kernel, &[], -5.0, 5.0).unwrap();
    assert!(
        !is_heuristic,
        "sigmoid should dispatch to analytical bounds"
    );
    let (exp_lo, exp_hi) = super::prove_dispatch::sigmoid_output_bounds(-5.0, 5.0).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
}

#[test]
fn test_dispatch_sigmoid_monotonic_bounds() {
    // Sigmoid is monotonically increasing — no global minimum.
    // sigmoid(-5) ≈ 0.00669, sigmoid(5) ≈ 0.99331
    let kernel = stub_kernel("sigmoid", 1);
    let (lo, hi, is_heuristic) = compute_output_bounds_heuristic(&kernel, &[], -5.0, 5.0).unwrap();
    assert!(
        !is_heuristic,
        "sigmoid should dispatch to analytical bounds"
    );
    // Lower bound should be close to 0 (sigmoid(-5) ≈ 0.00669).
    assert!(lo > 0.0, "sigmoid lower bound must be > 0");
    assert!(lo < 0.01, "sigmoid lower bound should be < 0.01 for x=-5");
    // Upper bound should be close to 1 (sigmoid(5) ≈ 0.99331).
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
// When constant_params.len() < min_constant_params, the registry skips the
// entry and falls through to the ±1e6 heuristic. The defense-in-depth guard
// inside each bounds function (require_params) is a secondary safety net.

#[test]
fn test_dispatch_snake_insufficient_params_uses_heuristic() {
    let kernel = stub_kernel("snake", 2);
    // Registry requires 1 constant param; provide 0 → registry skip → heuristic.
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
    // Registry requires 5 constant params; provide 3 → registry skip → heuristic.
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
    // Registry requires 6 constant params; provide 4 → registry skip → heuristic.
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
    let kernel = stub_kernel("relu", 1); // x only (0 constant params)
    let (lo, hi, is_heuristic) = compute_output_bounds_heuristic(&kernel, &[], -5.0, 5.0).unwrap();
    assert!(!is_heuristic, "relu should dispatch to analytical bounds");
    let (exp_lo, exp_hi) = super::prove_dispatch::relu_output_bounds(-5.0, 5.0).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
    // Monotonicity: relu(-5)=0, relu(5)=5
    assert!((lo - 0.0).abs() < TOL, "relu lo should be 0");
    assert!((hi - 5.0).abs() < TOL, "relu hi should be 5");
}

// ======================== tanh_act ========================

#[test]
fn test_dispatch_tanh_act() {
    let kernel = stub_kernel("tanh_act", 1); // x only (0 constant params)
    let (lo, hi, is_heuristic) = compute_output_bounds_heuristic(&kernel, &[], -5.0, 5.0).unwrap();
    assert!(
        !is_heuristic,
        "tanh_act should dispatch to analytical bounds"
    );
    let (exp_lo, exp_hi) = super::prove_dispatch::tanh_output_bounds(-5.0, 5.0).unwrap();
    assert!((lo - exp_lo).abs() < TOL, "lo: {lo} != {exp_lo}");
    assert!((hi - exp_hi).abs() < TOL, "hi: {hi} != {exp_hi}");
    // Monotonicity: tanh(-5) ≈ -0.99991, tanh(5) ≈ 0.99991
    assert!(lo < -0.999 && lo > -1.0, "tanh lo ∈ (-1, -0.999)");
    assert!(hi > 0.999 && hi < 1.0, "tanh hi ∈ (0.999, 1)");
}
