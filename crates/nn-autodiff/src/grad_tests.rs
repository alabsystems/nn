#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for backward pass and gradient computation.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::error::AutodiffError;
use crate::grad::backward;
use crate::grad::test_helpers::{scalar_var, vec_var};
use crate::tracked::TrackedTensor;
use crate::var::Var;

// -- Basic gradient tests -----------------------------------------------------

#[test]
fn test_backward_identity() {
    // y = x, dy/dx = 1
    let x = scalar_var(5.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let grads = backward(&t).unwrap();
    let grad = grads.get(&x).unwrap();
    assert_eq!(grad.to_flat_vec::<f32>().unwrap(), vec![1.0]);
}

#[test]
fn test_backward_add() {
    // y = a + b, dy/da = 1, dy/db = 1
    let a = scalar_var(3.0);
    let b = scalar_var(4.0);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.add(&tb).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    assert_eq!(
        grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap(),
        vec![1.0]
    );
    assert_eq!(
        grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap(),
        vec![1.0]
    );
}

#[test]
fn test_backward_sub() {
    // y = a - b, dy/da = 1, dy/db = -1
    let a = scalar_var(5.0);
    let b = scalar_var(3.0);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.sub(&tb).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    assert_eq!(
        grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap(),
        vec![1.0]
    );
    assert_eq!(
        grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap(),
        vec![-1.0]
    );
}

#[test]
fn test_backward_mul() {
    // y = a * b, dy/da = b = 4, dy/db = a = 3
    let a = scalar_var(3.0);
    let b = scalar_var(4.0);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.mul(&tb).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    assert_eq!(
        grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap(),
        vec![4.0]
    );
    assert_eq!(
        grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap(),
        vec![3.0]
    );
}

#[test]
fn test_backward_sqr() {
    // y = x^2, dy/dx = 2x = 6
    let x = scalar_var(3.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.sqr().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 6.0).abs() < 1e-5,
        "expected 6.0, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_neg() {
    // y = -x, dy/dx = -1
    let x = scalar_var(3.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.neg().unwrap();
    let grads = backward(&y).unwrap();
    assert_eq!(
        grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap(),
        vec![-1.0]
    );
}

#[test]
fn test_backward_mul_scalar() {
    // y = 3*x, dy/dx = 3
    let x = scalar_var(2.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.mul_scalar(3.0).unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!((grad[0] - 3.0).abs() < 1e-5);
}

#[test]
fn test_backward_add_scalar() {
    // y = x + 5, dy/dx = 1
    let x = scalar_var(2.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.add_scalar(5.0).unwrap();
    let grads = backward(&y).unwrap();
    assert_eq!(
        grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap(),
        vec![1.0]
    );
}

#[test]
fn test_backward_chain_rule() {
    // y = (x * 2)^2 = 4x^2, dy/dx = 8x = 24
    let x = scalar_var(3.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let doubled = t.mul_scalar(2.0).unwrap();
    let y = doubled.sqr().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 24.0).abs() < 1e-4,
        "expected 24.0, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_exp() {
    // y = exp(x), dy/dx = exp(x) = exp(1) ≈ 2.718
    let x = scalar_var(1.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.exp().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let expected = 1.0_f32.exp();
    assert!(
        (grad[0] - expected).abs() < 1e-5,
        "expected {expected}, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_tanh() {
    // y = tanh(x), dy/dx = 1 - tanh(x)^2
    let x = scalar_var(0.5);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.tanh().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let expected = 1.0 - 0.5_f32.tanh().powi(2);
    assert!(
        (grad[0] - expected).abs() < 1e-5,
        "expected {expected}, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_relu() {
    // y = sum(relu(x)), x = [-1, 2, -3, 4]
    // dy/dx = [0, 1, 0, 1]
    let x = vec_var(vec![-1.0, 2.0, -3.0, 4.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.relu().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(grad, vec![0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn test_backward_div() {
    // y = a / b, a=6, b=3 => y=2
    // dy/da = 1/b = 1/3
    // dy/db = -a/b^2 = -6/9 = -2/3
    let a = scalar_var(6.0);
    let b = scalar_var(3.0);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.div(&tb).unwrap();
    let grads = backward(&y).unwrap();
    let ga = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let gb = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (ga[0] - 1.0 / 3.0).abs() < 1e-5,
        "da: expected 0.333, got {}",
        ga[0]
    );
    assert!(
        (gb[0] - (-2.0 / 3.0)).abs() < 1e-5,
        "db: expected -0.667, got {}",
        gb[0]
    );
}

#[test]
fn test_backward_reshape() {
    // Reshape doesn't change values, grad flows through
    let x = vec_var(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let reshaped = t.reshape(&[2, 3]).unwrap();
    let loss = reshaped.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // All gradients should be 1.0 (sum reduces all elements)
    assert_eq!(grad, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_backward_non_scalar_loss_rejected() {
    let x = vec_var(vec![1.0, 2.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let err = backward(&t).unwrap_err();
    assert!(matches!(err, AutodiffError::NonScalarLoss { .. }));
}

#[test]
fn test_backward_nan_loss_rejected() {
    // NaN loss value must be rejected before running the backward pass.
    // Construct NaN directly — DynTensor::div guards 0/0 on CPU.
    let nan_tensor = DynTensor::from_vec(vec![f32::NAN], &[1], &cpu()).unwrap();
    let nan_loss = Arc::new(TrackedTensor::from_tensor(nan_tensor));
    let err = backward(&nan_loss).unwrap_err();
    assert!(
        matches!(err, AutodiffError::NonFiniteLoss),
        "expected NonFiniteLoss, got {err:?}"
    );
}

#[test]
fn test_backward_inf_loss_rejected() {
    // Inf loss (e.g., from log(0)) must be rejected.
    let x = scalar_var(0.0);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    // log(0) = -inf
    let log_zero = tx.log().unwrap();
    let err = backward(&log_zero).unwrap_err();
    assert!(
        matches!(err, AutodiffError::NonFiniteLoss),
        "expected NonFiniteLoss, got {err:?}"
    );
}

#[test]
fn test_backward_no_grad_for_constants() {
    // Constants should not accumulate gradients
    let x = scalar_var(3.0);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let c = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![2.0], &[1], &cpu()).unwrap(),
    ));
    let y = tx.mul(&c).unwrap(); // y = x * 2
    let grads = backward(&y).unwrap();
    // x has gradient = 2 (the constant value)
    assert_eq!(
        grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap(),
        vec![2.0]
    );
}

#[test]
fn test_backward_matmul() {
    // y = a @ b, a=[1,2], b=[[1],[2]] => y = [[5]]
    let a_var = Var::new(DynTensor::from_vec(vec![1.0, 2.0], &[1, 2], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(vec![1.0, 2.0], &[2, 1], &cpu()).unwrap());
    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = ta.matmul(&tb).unwrap();
    // y is [1,1] scalar
    let loss = y.reshape(&[1]).unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    // dy/da = grad @ b^T = [[1]] @ [[1,2]] = [[1,2]]
    let ga = grads.get(&a_var).unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(ga, vec![1.0, 2.0]);
}

#[test]
fn test_backward_sigmoid() {
    // y = sigmoid(x), dy/dx = sigmoid(x)*(1-sigmoid(x))
    let x = scalar_var(0.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.sigmoid().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // sigmoid(0) = 0.5, derivative = 0.5 * 0.5 = 0.25
    assert!(
        (grad[0] - 0.25).abs() < 1e-5,
        "expected 0.25, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_log() {
    // y = ln(x), dy/dx = 1/x
    let x = scalar_var(2.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.log().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 0.5).abs() < 1e-5,
        "expected 0.5, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_sqrt() {
    // y = sqrt(x), dy/dx = 1/(2*sqrt(x))
    let x = scalar_var(4.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.sqrt().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // 1/(2*sqrt(4)) = 1/4 = 0.25
    assert!(
        (grad[0] - 0.25).abs() < 1e-5,
        "expected 0.25, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_sqrt_at_zero() {
    // sqrt(0) = 0 (finite), but derivative 1/(2*sqrt(0)) = Inf.
    // Guard: gradient should be 0 at x=0, not Inf (#2002).
    let x = Var::new(DynTensor::from_vec(vec![0.0, 4.0, 9.0], &[3], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.sqrt().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // x=0 → grad=0 (subderivative convention)
    // x=4 → grad=1/(2*2)=0.25
    // x=9 → grad=1/(2*3)≈0.1667
    assert_eq!(grad[0], 0.0, "gradient at x=0 should be 0, not Inf");
    assert!(
        (grad[1] - 0.25).abs() < 1e-5,
        "expected 0.25, got {}",
        grad[1]
    );
    assert!(
        (grad[2] - 1.0 / 6.0).abs() < 1e-5,
        "expected ~0.1667, got {}",
        grad[2]
    );
}

#[test]
fn test_backward_abs_positive() {
    // y = |x|, x>0 => dy/dx = 1
    let x = scalar_var(3.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.abs().unwrap();
    let grads = backward(&y).unwrap();
    assert_eq!(
        grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap(),
        vec![1.0]
    );
}

#[test]
fn test_backward_abs_negative() {
    // y = |x|, x<0 => dy/dx = -1
    let x = scalar_var(-3.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.abs().unwrap();
    let grads = backward(&y).unwrap();
    assert_eq!(
        grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap(),
        vec![-1.0]
    );
}

#[test]
fn test_backward_unsqueeze_squeeze() {
    // y = sum(squeeze(unsqueeze(x, 1), 1)) should recover original gradient
    let x = vec_var(vec![1.0, 2.0, 3.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let expanded = t.unsqueeze(1).unwrap(); // [3] -> [3, 1]
    let squeezed = expanded.squeeze(1).unwrap(); // [3, 1] -> [3]
    let loss = squeezed.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(grad, vec![1.0, 1.0, 1.0]);
}

#[path = "grad_tests_advanced.rs"]
mod advanced;

#[path = "grad_tests_advanced_matmul.rs"]
mod advanced_matmul;

#[path = "grad_tests_composite.rs"]
mod composite;

#[path = "grad_tests_extended.rs"]
mod extended;

#[path = "grad_tests_conv1d.rs"]
mod conv1d;

#[path = "grad_tests_activation_fd_tests.rs"]
mod activation_fd;

#[path = "grad_tests_dropout.rs"]
mod dropout;

#[path = "grad_tests_norm_fd.rs"]
mod norm_fd;

#[path = "grad_tests_norm_fd_extra.rs"]
mod norm_fd_extra;

#[path = "grad_tests_layer_norm_fd.rs"]
mod layer_norm_fd;

#[path = "grad_tests_conv_transpose1d_fd.rs"]
mod conv_transpose1d_fd;

#[path = "grad_tests_conv_transpose1d_fd_extra.rs"]
mod conv_transpose1d_fd_extra;

#[path = "grad_tests_misc_fd.rs"]
mod misc_fd;

#[path = "grad_tests_shape_reduce_fd.rs"]
mod shape_reduce_fd;

#[path = "grad_tests_shape_fd.rs"]
mod shape_fd;

#[path = "grad_tests_sub_neg_sum_fd.rs"]
mod sub_neg_sum_fd;

#[path = "lstm_fd_tests.rs"]
mod lstm_fd;

#[path = "swiglu_fd_tests.rs"]
mod swiglu_fd;

#[path = "grad_tests_math_fd.rs"]
mod math_fd;

#[path = "grad_tests_conv1d_fd.rs"]
mod conv1d_fd;

#[path = "grad_tests_bf16_fd.rs"]
mod bf16_fd;

#[path = "grad_tests_norm_rank_validation.rs"]
mod norm_rank_validation;
