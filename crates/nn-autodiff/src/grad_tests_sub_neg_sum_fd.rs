#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Finite-difference gradient tests for Sub, Neg, and SumKeepDim.
//!
//! These 3 ops had structural backward tests but lacked numerical FD
//! verification. Sub backward negates one operand's gradient, Neg backward
//! negates the full gradient, and SumKeepDim backward expands the gradient
//! back to the input shape.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::grad::test_helpers::{check_fd_grad, sum_f64};
use crate::tracked::TrackedTensor;
use crate::var::Var;

// ── Sub FD tests ─────────────────────────────────────────────────────

/// FD test for Sub backward — both operands.
/// loss = sum((a - b)^2) => grad_a = 2*(a-b), grad_b = -2*(a-b).
#[test]
fn test_backward_sub_fd() {
    let a_data = vec![0.5, -1.3, 2.1, -0.7];
    let b_data = vec![0.2, 0.8, -1.0, 0.4];
    let eps = 1e-3_f32;

    let a_var = Var::new(DynTensor::from_vec(a_data.clone(), &[4], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(b_data.clone(), &[4], &cpu()).unwrap());
    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let diff = ta.sub(&tb).unwrap();
    let sq = diff.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical_a = grads.get(&a_var).unwrap().to_flat_vec::<f32>().unwrap();
    let analytical_b = grads.get(&b_var).unwrap().to_flat_vec::<f32>().unwrap();

    // FD for input a
    check_fd_grad(&analytical_a, &a_data, eps, |d| {
        let a = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[4], &cpu()).unwrap();
        let diff = a.sub(&b).unwrap();
        sum_f64(&diff.sqr().unwrap())
    });

    // FD for input b
    check_fd_grad(&analytical_b, &b_data, eps, |d| {
        let a = DynTensor::from_vec(a_data.clone(), &[4], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        let diff = a.sub(&b).unwrap();
        sum_f64(&diff.sqr().unwrap())
    });
}

/// FD test for Sub backward — 2D inputs with non-trivial downstream.
/// loss = sum((a - b)^2) for [2, 3] tensors.
#[test]
fn test_backward_sub_2d_fd() {
    let a_data = vec![0.3, -1.2, 0.8, 2.1, -0.5, 0.7];
    let b_data = vec![-0.1, 0.4, 1.5, -0.9, 0.6, -0.2];
    let eps = 1e-3_f32;

    let a_var = Var::new(DynTensor::from_vec(a_data.clone(), &[2, 3], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(b_data.clone(), &[2, 3], &cpu()).unwrap());
    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let diff = ta.sub(&tb).unwrap();
    let sq = diff.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical_a = grads.get(&a_var).unwrap().to_flat_vec::<f32>().unwrap();
    let analytical_b = grads.get(&b_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical_a, &a_data, eps, |d| {
        let a = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[2, 3], &cpu()).unwrap();
        let diff = a.sub(&b).unwrap();
        sum_f64(&diff.sqr().unwrap())
    });

    check_fd_grad(&analytical_b, &b_data, eps, |d| {
        let a = DynTensor::from_vec(a_data.clone(), &[2, 3], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let diff = a.sub(&b).unwrap();
        sum_f64(&diff.sqr().unwrap())
    });
}

// ── Neg FD tests ─────────────────────────────────────────────────────

/// FD test for Neg backward — 1D input.
/// loss = sum((-x)^2) = sum(x^2) => gradient should equal 2*x (since d/dx(-x)^2 = 2x).
/// But the key is that backward of neg correctly negates the upstream gradient.
#[test]
fn test_backward_neg_fd() {
    let data = vec![0.3, -1.5, 2.0, -0.8];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[4], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.neg().unwrap();
    let sq = y.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        let y = x.neg().unwrap();
        sum_f64(&y.sqr().unwrap())
    });
}

/// FD test for Neg backward — chained with add.
/// loss = sum(((-x) + c)^2), where c is a constant. Tests that neg correctly
/// negates gradient through a non-trivial downstream.
#[test]
fn test_backward_neg_chained_fd() {
    let data = vec![0.5, -1.2, 0.3];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[3], &cpu()).unwrap());
    let c_data = DynTensor::from_vec(vec![1.0, 2.0, -0.5], &[3], &cpu()).unwrap();
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tc = Arc::new(TrackedTensor::from_tensor(c_data.clone()));
    let neg_x = tx.neg().unwrap();
    let y = neg_x.add(&tc).unwrap();
    let sq = y.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        let c = c_data.clone();
        let neg_x = x.neg().unwrap();
        let y = neg_x.add(&c).unwrap();
        sum_f64(&y.sqr().unwrap())
    });
}

// ── SumKeepDim FD tests ──────────────────────────────────────────────

/// FD test for SumKeepDim backward — dim=0 on [2, 3].
/// loss = sum(sum(x, dim=0)^2). Backward expands gradient to input shape.
#[test]
fn test_backward_sum_keepdim_dim0_fd() {
    let data = vec![0.5, -1.2, 0.3, 2.1, -0.8, 0.7];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[2, 3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let s = tx.sum_keepdim(0).unwrap();
    let sq = s.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let s = x.sum_keepdim(0).unwrap();
        sum_f64(&s.sqr().unwrap())
    });
}

/// FD test for SumKeepDim backward — dim=1 on [2, 3].
/// loss = sum(sum(x, dim=1)^2). Gradient is expanded along dim=1.
#[test]
fn test_backward_sum_keepdim_dim1_fd() {
    let data = vec![0.3, -0.5, 1.2, 0.8, 2.1, -1.0];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[2, 3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let s = tx.sum_keepdim(1).unwrap();
    let sq = s.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let s = x.sum_keepdim(1).unwrap();
        sum_f64(&s.sqr().unwrap())
    });
}

/// FD test for SumKeepDim backward — 1D input.
/// loss = sum(x, dim=0)^2 (scalar result). Gradient should be 2*sum(x) everywhere.
#[test]
fn test_backward_sum_keepdim_1d_fd() {
    let data = vec![1.0, -0.5, 2.3, -1.7];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[4], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let s = tx.sum_keepdim(0).unwrap();
    let sq = s.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        let s = x.sum_keepdim(0).unwrap();
        sum_f64(&s.sqr().unwrap())
    });
}
