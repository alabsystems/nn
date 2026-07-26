#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Advanced backward pass tests: diamond patterns, reductions, activation chains,
//! broadcast, and multi-op composition.
//! Matmul backward tests extracted to `grad_tests_advanced_matmul.rs`.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::grad::test_helpers::{scalar_var, vec_var};
use crate::tracked::TrackedTensor;
use crate::var::Var;

// -- Diamond pattern and gradient accumulation tests --------------------------

#[test]
fn test_backward_diamond_mul_same_var() {
    // y = x * x (both inputs are the same node)
    // dy/dx = x (from left) + x (from right) = 2x = 10
    // This tests gradient accumulation when the same variable is used as both
    // inputs to Op::Mul, unlike Op::Sqr which has dedicated handling.
    let x = scalar_var(5.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.mul(&t).unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 10.0).abs() < 1e-5,
        "expected 10.0 (2*5), got {}",
        grad[0]
    );
}

#[test]
fn test_backward_diamond_add_mul() {
    // y = (x + x) * x = 2x * x = 2x^2
    // dy/dx = 4x = 12 (x=3)
    // This exercises gradient accumulation through three paths:
    //   mul.left: grad * x (from the x in mul right)
    //   mul.right: grad * (x+x) (the add result)
    //   add flows grad to x twice
    let x = scalar_var(3.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let sum = t.add(&t).unwrap(); // 2x
    let y = sum.mul(&t).unwrap(); // 2x * x = 2x^2
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 12.0).abs() < 1e-4,
        "expected 12.0 (4*3), got {}",
        grad[0]
    );
}

// -- Reduction backward tests ------------------------------------------------

#[test]
fn test_backward_mean_keepdim() {
    // y = mean(x, dim=0), x = [2, 4, 6, 8]
    // dy/dx_i = 1/n = 1/4 = 0.25 for each element
    let x = vec_var(vec![2.0, 4.0, 6.0, 8.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.mean_keepdim(0).unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    for (i, &g) in grad.iter().enumerate() {
        assert!((g - 0.25).abs() < 1e-5, "grad[{i}]: expected 0.25, got {g}");
    }
}

#[test]
fn test_backward_mean_keepdim_2d() {
    // x = [[1, 2], [3, 4]] (shape [2, 2])
    // y = mean(x, dim=1) => [[1.5], [3.5]] (shape [2, 1])
    // loss = sum(y) = 5.0
    // dy/dx[i,j] = 1/2 for all i,j (mean over dim=1 with 2 elements)
    let x_var = Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = t.mean_keepdim(1).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(grad, vec![0.5, 0.5, 0.5, 0.5]);
}

// -- Transpose backward test -------------------------------------------------

#[test]
fn test_backward_transpose() {
    // x = [[1, 2, 3], [4, 5, 6]] (shape [2, 3])
    // y = sum(transpose(x)) = sum of all elements
    // dy/dx = all 1s (transpose is a shape-only op)
    let x_var =
        Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let transposed = t.transpose(0, 1).unwrap(); // [3, 2]
    let loss = transposed.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(grad, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

// -- Sum backward with intermediate flow test --------------------------------

#[test]
fn test_backward_sum_keepdim_intermediate() {
    // x = [1, 2, 3] (shape [3])
    // y = sum(x) * 2 (scalar mul after reduction)
    // dy/dx = 2 for each element
    let x = vec_var(vec![1.0, 2.0, 3.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let s = t.sum_keepdim(0).unwrap();
    let y = s.mul_scalar(2.0).unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(grad, vec![2.0, 2.0, 2.0]);
}

// -- Abs vector backward test -----------------------------------------------

#[test]
fn test_backward_abs_mixed_vector() {
    // x = [-3, 2, -1, 4, 0]
    // y = sum(|x|)
    // dy/dx = [-1, 1, -1, 1, 0] (sign of each element; 0 at zero)
    let x = Var::new(DynTensor::from_vec(vec![-3.0, 2.0, -1.0, 4.0, 0.0], &[5], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.abs().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // At x=0, abs'(0) = 0 (PyTorch sign(0)=0 convention).
    // gt(0) and lt(0) are both false at zero, so gradient is 0.
    assert_eq!(grad, vec![-1.0, 1.0, -1.0, 1.0, 0.0]);
}

// -- Activation chain: log(sigmoid(x)) backward ----------------------------

#[test]
fn test_backward_log_sigmoid_chain() {
    // y = log(sigmoid(x)) = -log(1 + exp(-x)) (log-sigmoid, common in BCE)
    // dy/dx = 1 - sigmoid(x) = sigmoid(-x)
    // At x = 1.0: sigmoid(-1) = 1/(1+e) ≈ 0.2689
    let x = scalar_var(1.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let sig = t.sigmoid().unwrap();
    let y = sig.log().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let expected = 1.0 / (1.0 + 1.0_f32.exp()); // sigmoid(-1)
    assert!(
        (grad[0] - expected).abs() < 1e-5,
        "log(sigmoid(1)) grad: expected {expected:.6}, got {:.6}",
        grad[0]
    );
}

// -- Sum keepdim dim=1 backward test ----------------------------------------

#[test]
fn test_backward_sum_keepdim_dim1() {
    // x = [[1, 2, 3], [4, 5, 6]] (shape [2, 3])
    // y = sum_keepdim(x, dim=1) => [[6], [15]] (shape [2, 1])
    // loss = sum_keepdim(y, dim=0) => [[21]] (shape [1, 1])
    // dy/dx[i,j] = 1 for all i,j
    let x_var =
        Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let s = t.sum_keepdim(1).unwrap(); // [2, 1]
    let loss = s.sum_keepdim(0).unwrap(); // [1, 1]
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(grad, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

// -- Multi-op composition: (x^2 + exp(x)) backward -------------------------

#[test]
fn test_backward_sqr_plus_exp() {
    // y = x^2 + exp(x), dy/dx = 2x + exp(x)
    // At x = 2: 2*2 + exp(2) = 4 + 7.389 = 11.389
    let x = scalar_var(2.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let sq = t.sqr().unwrap();
    let ex = t.exp().unwrap();
    let y = sq.add(&ex).unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    let expected = 2.0 * 2.0 + 2.0_f32.exp();
    assert!(
        (grad[0] - expected).abs() < 1e-4,
        "x^2 + exp(x) grad at 2: expected {expected:.4}, got {:.4}",
        grad[0]
    );
}

#[test]
fn test_backward_broadcast_reduces_gradient() {
    // f(x) = sum(broadcast(x, [3, 2])) where x has shape [1, 2]
    // broadcast replicates x along dim 0: [[a, b], [a, b], [a, b]]
    // sum = 3a + 3b, so d/da = 3, d/db = 3
    let x = Var::new(DynTensor::from_vec(vec![2.0, 5.0], &[1, 2], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let broad = t.broadcast_as(&[3, 2]).unwrap();
    let y = broad.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(grad.len(), 2);
    assert!(
        (grad[0] - 3.0).abs() < 1e-5,
        "broadcast grad[0]: expected 3.0, got {}",
        grad[0]
    );
    assert!(
        (grad[1] - 3.0).abs() < 1e-5,
        "broadcast grad[1]: expected 3.0, got {}",
        grad[1]
    );
}

#[test]
fn test_backward_broadcast_scalar_to_vector() {
    // f(x) = sum(broadcast(x, [4])) where x is scalar shape [1]
    // broadcast replicates x 4 times: [x, x, x, x]
    // sum = 4x, d/dx = 4
    let x = scalar_var(3.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let broad = t.broadcast_as(&[4]).unwrap();
    let y = broad.sum_keepdim(0).unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(grad.len(), 1);
    assert!(
        (grad[0] - 4.0).abs() < 1e-5,
        "broadcast scalar grad: expected 4.0, got {}",
        grad[0]
    );
}
