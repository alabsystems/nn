#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared test helpers for gradient tests.
//!
//! Eliminates 12 duplicate copies of `scalar_var`, `vec_var`, `sum_f64`,
//! and `check_fd_grad` across `grad_tests*.rs` files.

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::var::Var;

/// Create a scalar Var with shape [1].
pub(super) fn scalar_var(val: f32) -> Var {
    Var::new(DynTensor::from_vec(vec![val], &[1], &cpu()).unwrap())
}

/// Create a 1-D Var from a Vec<f32>.
pub(super) fn vec_var(data: Vec<f32>) -> Var {
    let n = data.len();
    Var::new(DynTensor::from_vec(data, &[n], &cpu()).unwrap())
}

/// Sum all elements of a DynTensor as f64.
pub(super) fn sum_f64(t: &DynTensor) -> f64 {
    t.to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .sum()
}

/// Sum of squares of all elements of a DynTensor as f64.
///
/// Used for nonlinear FD loss functions per #1538: `sum(sqr(x))` produces
/// input-dependent gradients (2*x_i), unlike `sum(x)` which gives uniform 1.0.
pub(super) fn sum_sqr_f64(t: &DynTensor) -> f64 {
    t.to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum()
}

/// Compare analytical gradients against finite-difference numerical gradients.
///
/// Uses f64 arithmetic for the comparison to avoid f32 precision loss.
/// Default tolerance: 1e-2. Use [`check_fd_grad_tol`] for custom tolerance.
pub(super) fn check_fd_grad(
    analytical: &[f32],
    data: &[f32],
    eps: f32,
    fwd: impl Fn(Vec<f32>) -> f64,
) {
    check_fd_grad_tol(analytical, data, eps, 1e-2, &fwd);
}

/// Compare analytical gradients against finite-difference numerical gradients
/// with a custom tolerance.
///
/// All comparisons use f64 arithmetic to avoid f32 precision loss in the
/// subtraction `analytical - numerical`.
pub(super) fn check_fd_grad_tol(
    analytical: &[f32],
    data: &[f32],
    eps: f32,
    tol: f64,
    fwd: &dyn Fn(Vec<f32>) -> f64,
) {
    for i in 0..data.len() {
        let mut plus = data.to_vec();
        let mut minus = data.to_vec();
        plus[i] += eps;
        minus[i] -= eps;
        let numerical = (fwd(plus) - fwd(minus)) / (2.0 * f64::from(eps));
        let analytical_f64 = f64::from(analytical[i]);
        let err = (analytical_f64 - numerical).abs();
        assert!(
            err < tol,
            "grad[{i}]: analytical={analytical_f64}, numerical={numerical}, err={err}, tol={tol}",
        );
    }
}
