#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Finite-difference gradient tests for elementwise math ops.
//!
//! Covers Sin, Cos, Recip, Powf — all have non-trivial backward rules
//! in backward_rules.rs:210-229 but had zero FD numerical verification.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::grad::test_helpers::{check_fd_grad, sum_f64, vec_var};
use crate::tracked::TrackedTensor;

/// Generic FD test for a unary tracked op that takes &Arc<TrackedTensor> -> Arc<TrackedTensor>.
fn fd_unary_test(
    data: Vec<f32>,
    tracked_op: fn(&Arc<TrackedTensor>) -> Arc<TrackedTensor>,
    tensor_op: fn(&DynTensor) -> DynTensor,
) {
    let n = data.len();
    let var = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let y = tracked_op(&t);
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&grad, &data, 1e-3, |d| {
        sum_f64(&tensor_op(&DynTensor::from_vec(d, &[n], &cpu()).unwrap()))
    });
}

#[test]
fn test_backward_sin_finite_diff() {
    fd_unary_test(
        vec![-2.0, -1.0, 0.0, 0.5, 1.5, 3.0],
        |t| t.sin().unwrap(),
        |t| t.sin().unwrap(),
    );
}

#[test]
fn test_backward_cos_finite_diff() {
    fd_unary_test(
        vec![-2.0, -1.0, 0.0, 0.5, 1.5, 3.0],
        |t| t.cos().unwrap(),
        |t| t.cos().unwrap(),
    );
}

#[test]
fn test_backward_recip_finite_diff() {
    // Avoid values near zero where recip gradient is huge
    fd_unary_test(
        vec![0.5, 1.0, 2.0, 5.0, 10.0],
        |t| t.recip().unwrap(),
        |t| t.recip().unwrap(),
    );
}

#[test]
fn test_backward_recip_negative_finite_diff() {
    fd_unary_test(
        vec![-5.0, -2.0, -1.0, -0.5],
        |t| t.recip().unwrap(),
        |t| t.recip().unwrap(),
    );
}

#[test]
fn test_backward_powf_square_finite_diff() {
    // p=2.0: d/dx x^2 = 2x
    let data = vec![-2.0, -1.0, 0.5, 1.0, 3.0];
    let n = data.len();
    let var = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let y = t.powf(2.0).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&grad, &data, 1e-3, |d| {
        sum_f64(
            &DynTensor::from_vec(d, &[n], &cpu())
                .unwrap()
                .powf(2.0)
                .unwrap(),
        )
    });
}

#[test]
fn test_backward_powf_cube_finite_diff() {
    // p=3.0: d/dx x^3 = 3x^2
    let data = vec![-1.5, -0.5, 0.5, 1.0, 2.0];
    let n = data.len();
    let var = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let y = t.powf(3.0).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&grad, &data, 1e-3, |d| {
        sum_f64(
            &DynTensor::from_vec(d, &[n], &cpu())
                .unwrap()
                .powf(3.0)
                .unwrap(),
        )
    });
}

#[test]
fn test_backward_powf_sqrt_finite_diff() {
    // p=0.5: d/dx x^0.5 = 0.5 * x^(-0.5) — must use positive inputs
    let data = vec![0.25, 1.0, 4.0, 9.0];
    let n = data.len();
    let var = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let y = t.powf(0.5).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&grad, &data, 1e-3, |d| {
        sum_f64(
            &DynTensor::from_vec(d, &[n], &cpu())
                .unwrap()
                .powf(0.5)
                .unwrap(),
        )
    });
}

#[test]
fn test_backward_powf_zero_finite_diff() {
    // p=0.0: x^0 = 1 (constant), gradient should be zero everywhere.
    // Regression: the general formula p * x^(p-1) = 0 * x^(-1) produces NaN at x=0.
    let data = vec![-2.0, -1.0, 0.0, 1.0, 3.0];
    let n = data.len();
    let var = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let y = t.powf(0.0).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    #[allow(deprecated)]
    let grad = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    // All gradients must be exactly zero (constant function).
    for (i, &g) in grad.iter().enumerate() {
        assert!(
            g == 0.0,
            "powf(0.0) gradient at index {i} should be 0.0, got {g}"
        );
    }
    // Also verify via FD: perturbing any input should not change the output.
    check_fd_grad(&grad, &data, 1e-3, |d| {
        sum_f64(
            &DynTensor::from_vec(d, &[n], &cpu())
                .unwrap()
                .powf(0.0)
                .unwrap(),
        )
    });
}
