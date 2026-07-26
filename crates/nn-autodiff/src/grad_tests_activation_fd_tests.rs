#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Finite-difference tests for activation backward rules.
//!
//! Covers Tanh, Sigmoid, Exp, Log, Sqrt — these have non-trivial derivative
//! formulas but previously only had single-point analytical checks.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::grad::test_helpers::{check_fd_grad, sum_f64, vec_var};
use crate::tracked::TrackedTensor;

fn fd_activation_test(
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
fn test_backward_tanh_finite_diff() {
    fd_activation_test(
        vec![-1.5, -0.5, 0.0, 0.5, 1.5],
        |t| t.tanh().unwrap(),
        |t| t.tanh().unwrap(),
    );
}

#[test]
fn test_backward_sigmoid_finite_diff() {
    fd_activation_test(
        vec![-2.0, -0.5, 0.0, 0.5, 2.0],
        |t| t.sigmoid().unwrap(),
        |t| t.sigmoid().unwrap(),
    );
}

#[test]
fn test_backward_exp_finite_diff() {
    fd_activation_test(
        vec![-1.0, 0.0, 0.5, 1.0, 2.0],
        |t| t.exp().unwrap(),
        |t| t.exp().unwrap(),
    );
}

#[test]
fn test_backward_log_finite_diff() {
    fd_activation_test(
        vec![0.5, 1.0, 2.0, 5.0, 10.0],
        |t| t.log().unwrap(),
        |t| t.log().unwrap(),
    );
}

#[test]
fn test_backward_sqrt_finite_diff() {
    fd_activation_test(
        vec![0.25, 1.0, 4.0, 9.0, 16.0],
        |t| t.sqrt().unwrap(),
        |t| t.sqrt().unwrap(),
    );
}

#[test]
fn test_backward_relu_finite_diff() {
    // ReLU has a discontinuity at 0, so avoid values near 0 for FD stability.
    fd_activation_test(
        vec![-2.0, -0.5, 0.1, 0.5, 2.0],
        |t| t.relu().unwrap(),
        |t| t.relu().unwrap(),
    );
}

#[test]
fn test_backward_gelu_finite_diff() {
    fd_activation_test(
        vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        |t| t.gelu().unwrap(),
        |t| t.gelu().unwrap(),
    );
}

#[test]
fn test_backward_silu_finite_diff() {
    fd_activation_test(
        vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        |t| t.silu().unwrap(),
        |t| t.silu().unwrap(),
    );
}

#[test]
fn test_backward_sqr_finite_diff() {
    fd_activation_test(
        vec![-3.0, -1.0, 0.0, 1.0, 3.0],
        |t| t.sqr().unwrap(),
        |t| t.sqr().unwrap(),
    );
}

#[test]
fn test_backward_abs_finite_diff() {
    // Abs has discontinuity at 0, avoid values near 0.
    fd_activation_test(
        vec![-3.0, -1.0, -0.1, 0.1, 1.0, 3.0],
        |t| t.abs().unwrap(),
        |t| t.abs().unwrap(),
    );
}

#[test]
fn test_backward_hard_sigmoid_finite_diff() {
    // Avoid exact boundary points -3, 3 where derivative is discontinuous
    fd_activation_test(
        vec![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0],
        |t| t.hard_sigmoid().unwrap(),
        |t| t.hard_sigmoid().unwrap(),
    );
}

#[test]
fn test_backward_hard_swish_finite_diff() {
    // Avoid exact boundary points -3, 3 where derivative is discontinuous
    fd_activation_test(
        vec![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0],
        |t| t.hard_swish().unwrap(),
        |t| t.hard_swish().unwrap(),
    );
}

#[test]
fn test_backward_mish_finite_diff() {
    fd_activation_test(
        vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        |t| t.mish().unwrap(),
        |t| t.mish().unwrap(),
    );
}

#[test]
fn test_backward_selu_finite_diff() {
    // Avoid exact zero where derivative is discontinuous
    fd_activation_test(
        vec![-2.0, -1.0, -0.5, 0.1, 0.5, 1.0, 2.0],
        |t| t.selu().unwrap(),
        |t| t.selu().unwrap(),
    );
}

#[test]
fn test_backward_softplus_finite_diff() {
    fd_activation_test(
        vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        |t| t.softplus().unwrap(),
        |t| t.softplus().unwrap(),
    );
}

#[test]
fn test_backward_celu_finite_diff() {
    // CELU has a parameter alpha, so use manual FD pattern
    let data = vec![-2.0, -1.0, -0.5, 0.1, 0.5, 1.0, 2.0];
    let n = data.len();
    let var = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let y = t.celu(1.0).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&grad, &data, 1e-3, |d| {
        sum_f64(
            &DynTensor::from_vec(d, &[n], &cpu())
                .unwrap()
                .celu(1.0)
                .unwrap(),
        )
    });
}

#[test]
fn test_backward_celu_alpha2_finite_diff() {
    // Test with non-default alpha
    let data = vec![-2.0, -1.0, -0.5, 0.1, 0.5, 1.0, 2.0];
    let n = data.len();
    let var = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let y = t.celu(0.5).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&grad, &data, 1e-3, |d| {
        sum_f64(
            &DynTensor::from_vec(d, &[n], &cpu())
                .unwrap()
                .celu(0.5)
                .unwrap(),
        )
    });
}

#[test]
fn test_backward_hard_sigmoid_at_key_points() {
    // x = 0: derivative = 1/6
    let var = vec_var(vec![0.0]);
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let y = t.hard_sigmoid().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 1.0 / 6.0).abs() < 1e-5,
        "expected 1/6 at x=0, got {}",
        grad[0]
    );

    // x = -5: derivative = 0 (saturated)
    let var2 = vec_var(vec![-5.0]);
    let t2 = Arc::new(TrackedTensor::from_var(&var2).unwrap());
    let y2 = t2.hard_sigmoid().unwrap();
    let grads2 = backward(&y2).unwrap();
    let grad2 = grads2.get(&var2).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad2[0]).abs() < 1e-5,
        "expected 0 at x=-5, got {}",
        grad2[0]
    );

    // x = 5: derivative = 0 (saturated)
    let var3 = vec_var(vec![5.0]);
    let t3 = Arc::new(TrackedTensor::from_var(&var3).unwrap());
    let y3 = t3.hard_sigmoid().unwrap();
    let grads3 = backward(&y3).unwrap();
    let grad3 = grads3.get(&var3).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad3[0]).abs() < 1e-5,
        "expected 0 at x=5, got {}",
        grad3[0]
    );
}

#[test]
fn test_backward_softplus_known_values() {
    // softplus'(x) = sigmoid(x)
    // At x=0: sigmoid(0) = 0.5
    let var = vec_var(vec![0.0]);
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let y = t.softplus().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 0.5).abs() < 1e-5,
        "expected 0.5 at x=0, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_selu_known_values() {
    // At x=1: d/dx = lambda ~= 1.0507
    let var = vec_var(vec![1.0]);
    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let y = t.selu().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    let lambda = 1.050_701_f32;
    assert!(
        (grad[0] - lambda).abs() < 1e-4,
        "expected lambda={} at x=1, got {}",
        lambda,
        grad[0]
    );
}
