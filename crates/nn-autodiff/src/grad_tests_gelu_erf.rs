// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Gradient correctness tests for gelu_erf (exact erf-based GELU).
//!
//! Verifies backward rule matches:
//! - Finite-difference numerical derivative
//! - PyTorch reference values for the exact GELU derivative
//! - Edge cases: x=0 (Phi(0)=0.5), large positive/negative inputs

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::backward;
use crate::tracked::TrackedTensor;
use crate::var::Var;

use super::test_helpers::{check_fd_grad, sum_f64, vec_var};

// ── Finite-difference: basic values ────────────────────────────────

#[test]
fn test_gelu_erf_fd_basic() {
    let data = vec![0.5, 1.0, -0.5, -1.0, 2.0];
    let x = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.gelu_erf().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&g, &data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[5], &cpu()).unwrap();
        sum_f64(&t.gelu_erf().unwrap())
    });
}

// ── Finite-difference: near zero ───────────────────────────────────

#[test]
fn test_gelu_erf_fd_near_zero() {
    // Near-zero values to test the safe division guard at x≈0.
    let data = vec![1e-6, -1e-6, 1e-4, -1e-4, 0.01];
    let x = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.gelu_erf().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&g, &data, 1e-4, |d| {
        let t = DynTensor::from_vec(d, &[5], &cpu()).unwrap();
        sum_f64(&t.gelu_erf().unwrap())
    });
}

// ── Finite-difference: large magnitude ─────────────────────────────

#[test]
fn test_gelu_erf_fd_large() {
    // Large values: gelu_erf(x) ≈ x for large positive, ≈ 0 for large negative.
    // d/dx ≈ 1 for large positive, ≈ 0 for large negative.
    let data = vec![3.0, 5.0, -3.0, -5.0];
    let x = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.gelu_erf().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&g, &data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        sum_f64(&t.gelu_erf().unwrap())
    });
}

// ── PyTorch reference values ───────────────────────────────────────

/// Exact GELU derivative: d/dx = Phi(x) + x * phi(x)
/// where Phi = standard normal CDF, phi = standard normal PDF.
///
/// PyTorch reference (torch 2.2):
/// ```python
/// import torch
/// x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], requires_grad=True)
/// y = torch.nn.functional.gelu(x, approximate='none')
/// y.sum().backward()
/// print(x.grad.tolist())
/// # [-0.0852, -0.0833, 0.1325, 0.5000, 0.8675, 1.0833, 1.0852]
/// ```
#[test]
fn test_gelu_erf_pytorch_reference() {
    let data = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    // Analytically computed: d/dx gelu_erf(x) = Phi(x) + x * phi(x).
    // Phi = standard normal CDF, phi = standard normal PDF.
    // x=-2: 0.02275 + (-2)*0.05399 = -0.08523
    // x=-1: 0.15866 + (-1)*0.24197 = -0.08332
    // x=-0.5: 0.30854 + (-0.5)*0.35207 = 0.13250
    // x=0: 0.50000 + 0*0.39894 = 0.50000
    // x=0.5: 0.69146 + 0.5*0.35207 = 0.86750
    // x=1: 0.84134 + 1*0.24197 = 1.08332
    // x=2: 0.97725 + 2*0.05399 = 1.08523
    let expected = [-0.08523, -0.08332, 0.13250, 0.50000, 0.86750, 1.08332, 1.08523];

    let x = vec_var(data);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.gelu_erf().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    for (i, (&got, &want)) in g.iter().zip(expected.iter()).enumerate() {
        let err = (f64::from(got) - want).abs();
        assert!(
            err < 1e-3,
            "gelu_erf grad[{i}]: got={got}, expected={want}, err={err}",
        );
    }
}

// ── Exact value at x=0: d/dx gelu_erf(0) = 0.5 ───────────────────

#[test]
fn test_gelu_erf_grad_at_zero() {
    // At x=0: Phi(0) = 0.5, phi(0) = 1/sqrt(2*pi) ≈ 0.3989.
    // d/dx = Phi(0) + 0 * phi(0) = 0.5 exactly.
    let x = Var::new(DynTensor::from_vec(vec![0.0_f32], &[1], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.gelu_erf().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    let err = (f64::from(g[0]) - 0.5).abs();
    assert!(
        err < 1e-5,
        "gelu_erf grad at x=0: got={}, expected=0.5, err={err}",
        g[0],
    );
}

// ── Asymptotic behavior ────────────────────────────────────────────

#[test]
fn test_gelu_erf_gradient_asymptotics() {
    // For large positive x: d/dx gelu_erf(x) → 1
    // For large negative x: d/dx gelu_erf(x) → 0
    // At x=0: d/dx = 0.5 exactly
    let data = vec![-10.0, -5.0, 0.0, 5.0, 10.0];
    let x = vec_var(data);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.gelu_erf().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    // x=-10: gradient should be near 0
    assert!(
        (f64::from(g[0])).abs() < 0.01,
        "grad at x=-10 should be ~0, got {}",
        g[0]
    );
    // x=-5: gradient should be near 0
    assert!(
        (f64::from(g[1])).abs() < 0.01,
        "grad at x=-5 should be ~0, got {}",
        g[1]
    );
    // x=0: gradient should be 0.5
    assert!(
        (f64::from(g[2]) - 0.5).abs() < 1e-5,
        "grad at x=0 should be 0.5, got {}",
        g[2]
    );
    // x=5: gradient should be near 1
    assert!(
        (f64::from(g[3]) - 1.0).abs() < 0.01,
        "grad at x=5 should be ~1, got {}",
        g[3]
    );
    // x=10: gradient should be near 1
    assert!(
        (f64::from(g[4]) - 1.0).abs() < 0.01,
        "grad at x=10 should be ~1, got {}",
        g[4]
    );
}
