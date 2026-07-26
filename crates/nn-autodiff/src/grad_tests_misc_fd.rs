#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Finite-difference gradient tests for Add, Mul, Cat, Permute, MulScalar,
//! and AddScalar.
//!
//! These ops had structural backward tests (expected-value assertions) but
//! lacked numerical FD verification. Adding FD tests closes the verified-training
//! completeness gap for these backward rules.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::grad::test_helpers::{check_fd_grad, sum_f64};
use crate::tracked::TrackedTensor;
use crate::var::Var;

// ── Add FD tests ───────────────────────────────────────────────────────

/// FD test for tensor-tensor Add backward — both inputs (1D).
/// loss = sum((a + b)^2). da = 2*(a+b), db = 2*(a+b).
#[test]
fn test_backward_add_fd() {
    let a_data = vec![0.3, -1.2, 0.8];
    let b_data = vec![2.1, -0.5, 0.7];
    let eps = 1e-3_f32;

    let a_var = Var::new(DynTensor::from_vec(a_data.clone(), &[3], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(b_data.clone(), &[3], &cpu()).unwrap());
    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = ta.add(&tb).unwrap();
    let sq = y.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical_a = grads.get(&a_var).unwrap().to_flat_vec::<f32>().unwrap();
    let analytical_b = grads.get(&b_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical_a, &a_data, eps, |d| {
        let a = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[3], &cpu()).unwrap();
        sum_f64(&a.add(&b).unwrap().sqr().unwrap())
    });

    check_fd_grad(&analytical_b, &b_data, eps, |d| {
        let a = DynTensor::from_vec(a_data.clone(), &[3], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        sum_f64(&a.add(&b).unwrap().sqr().unwrap())
    });
}

/// FD test for tensor-tensor Add backward — 2D inputs.
/// loss = sum((a + b)^2) for [2, 3] tensors.
#[test]
fn test_backward_add_2d_fd() {
    let a_data = vec![0.5, -0.3, 1.2, 0.8, -1.1, 0.4];
    let b_data = vec![0.1, 0.7, -0.4, 0.9, -0.6, 1.3];
    let eps = 1e-3_f32;

    let a_var = Var::new(DynTensor::from_vec(a_data.clone(), &[2, 3], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(b_data.clone(), &[2, 3], &cpu()).unwrap());
    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = ta.add(&tb).unwrap();
    let sq = y.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical_a = grads.get(&a_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical_a, &a_data, eps, |d| {
        let a = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[2, 3], &cpu()).unwrap();
        sum_f64(&a.add(&b).unwrap().sqr().unwrap())
    });
}

// ── Mul FD tests ───────────────────────────────────────────────────────

/// FD test for tensor-tensor Mul backward — both inputs (1D).
/// loss = sum((a * b)^2). da = 2*(a*b)*b, db = 2*(a*b)*a.
#[test]
fn test_backward_mul_fd() {
    let a_data = vec![0.3, -1.2, 0.8];
    let b_data = vec![2.1, -0.5, 0.7];
    let eps = 1e-3_f32;

    let a_var = Var::new(DynTensor::from_vec(a_data.clone(), &[3], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(b_data.clone(), &[3], &cpu()).unwrap());
    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = ta.mul(&tb).unwrap();
    let sq = y.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical_a = grads.get(&a_var).unwrap().to_flat_vec::<f32>().unwrap();
    let analytical_b = grads.get(&b_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical_a, &a_data, eps, |d| {
        let a = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[3], &cpu()).unwrap();
        sum_f64(&a.mul(&b).unwrap().sqr().unwrap())
    });

    check_fd_grad(&analytical_b, &b_data, eps, |d| {
        let a = DynTensor::from_vec(a_data.clone(), &[3], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        sum_f64(&a.mul(&b).unwrap().sqr().unwrap())
    });
}

/// FD test for tensor-tensor Mul backward — 2D inputs.
/// loss = sum((a * b)^2) for [2, 3] tensors.
#[test]
fn test_backward_mul_2d_fd() {
    let a_data = vec![0.5, -0.3, 1.2, 0.8, -1.1, 0.4];
    let b_data = vec![0.1, 0.7, -0.4, 0.9, -0.6, 1.3];
    let eps = 1e-3_f32;

    let a_var = Var::new(DynTensor::from_vec(a_data.clone(), &[2, 3], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(b_data.clone(), &[2, 3], &cpu()).unwrap());
    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = ta.mul(&tb).unwrap();
    let sq = y.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical_a = grads.get(&a_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical_a, &a_data, eps, |d| {
        let a = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[2, 3], &cpu()).unwrap();
        sum_f64(&a.mul(&b).unwrap().sqr().unwrap())
    });
}

// ── Cat FD tests ───────────────────────────────────────────────────────

/// FD test for Cat backward along dim=0 (1D inputs).
/// Perturbs each element of input `a` and verifies gradient via FD.
#[test]
fn test_backward_cat_dim0_fd() {
    let a_data = vec![0.3, -1.2];
    let b_data = vec![2.1, -0.5, 0.7];
    let eps = 1e-3_f32;

    let a_var = Var::new(DynTensor::from_vec(a_data.clone(), &[2], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(b_data.clone(), &[3], &cpu()).unwrap());
    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let catted = TrackedTensor::cat(&[&ta, &tb], 0).unwrap();
    let loss = catted.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical_a = grads.get(&a_var).unwrap().to_flat_vec::<f32>().unwrap();
    let analytical_b = grads.get(&b_var).unwrap().to_flat_vec::<f32>().unwrap();

    // FD for input a
    check_fd_grad(&analytical_a, &a_data, eps, |d| {
        let a = DynTensor::from_vec(d, &[2], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[3], &cpu()).unwrap();
        let c = DynTensor::cat(&[&a, &b], 0).unwrap();
        sum_f64(&c)
    });

    // FD for input b
    check_fd_grad(&analytical_b, &b_data, eps, |d| {
        let a = DynTensor::from_vec(a_data.clone(), &[2], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        let c = DynTensor::cat(&[&a, &b], 0).unwrap();
        sum_f64(&c)
    });
}

/// FD test for Cat backward along dim=1 (2D inputs).
/// Verifies gradient through a non-trivial downstream: element-wise square then sum.
#[test]
fn test_backward_cat_dim1_fd() {
    let a_data = vec![0.5, -0.3, 1.2, 0.8];
    let b_data = vec![0.1, -0.7, 0.4, 0.9, -1.1, 0.6];
    let eps = 1e-3_f32;

    let a_var = Var::new(DynTensor::from_vec(a_data.clone(), &[2, 2], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(b_data.clone(), &[2, 3], &cpu()).unwrap());
    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let catted = TrackedTensor::cat(&[&ta, &tb], 1).unwrap();
    // Use sqr to make gradients input-dependent (not trivially all-ones)
    let sq = catted.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical_a = grads.get(&a_var).unwrap().to_flat_vec::<f32>().unwrap();
    let analytical_b = grads.get(&b_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical_a, &a_data, eps, |d| {
        let a = DynTensor::from_vec(d, &[2, 2], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[2, 3], &cpu()).unwrap();
        let c = DynTensor::cat(&[&a, &b], 1).unwrap();
        sum_f64(&c.sqr().unwrap())
    });

    check_fd_grad(&analytical_b, &b_data, eps, |d| {
        let a = DynTensor::from_vec(a_data.clone(), &[2, 2], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let c = DynTensor::cat(&[&a, &b], 1).unwrap();
        sum_f64(&c.sqr().unwrap())
    });
}

// ── Permute FD tests ───────────────────────────────────────────────────

/// FD test for Permute backward (2D transpose).
/// Uses sqr downstream so gradients are input-dependent.
#[test]
fn test_backward_permute_2d_fd() {
    let data = vec![0.3, -1.2, 0.8, 2.1, -0.5, 0.7];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[2, 3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.permute(&[1, 0]).unwrap();
    let sq = y.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let p = x.permute([1, 0]).unwrap();
        sum_f64(&p.sqr().unwrap())
    });
}

/// FD test for Permute backward (3D: [2,3,4] -> [2,0,1] -> [4,2,3]).
/// Tests that inverse permutation correctly routes gradients.
#[test]
fn test_backward_permute_3d_fd() {
    let data: Vec<f32> = (1..=24).map(|i| i as f32 * 0.1).collect();
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[2, 3, 4], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.permute(&[2, 0, 1]).unwrap();
    // Apply sqr to make gradients position-dependent
    let sq = y.sqr().unwrap();
    let loss = sq
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[2, 3, 4], &cpu()).unwrap();
        let p = x.permute([2, 0, 1]).unwrap();
        sum_f64(&p.sqr().unwrap())
    });
}

// ── MulScalar FD tests ─────────────────────────────────────────────────

/// FD test for MulScalar backward with multi-element tensor.
/// loss = sum((x * scalar)^2) => gradient is input-dependent.
#[test]
fn test_backward_mul_scalar_fd() {
    let data = vec![0.5, -1.3, 2.1, -0.7];
    let scalar = 2.5_f64;
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[4], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.mul_scalar(scalar).unwrap();
    let sq = y.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        let y = x.mul_scalar(scalar).unwrap();
        sum_f64(&y.sqr().unwrap())
    });
}

/// FD test for MulScalar with negative scalar (sign flip).
#[test]
fn test_backward_mul_scalar_negative_fd() {
    let data = vec![1.0, -2.0, 0.5];
    let scalar = -3.0_f64;
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.mul_scalar(scalar).unwrap();
    let sq = y.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        let y = x.mul_scalar(scalar).unwrap();
        sum_f64(&y.sqr().unwrap())
    });
}

// ── AddScalar FD tests ─────────────────────────────────────────────────

/// FD test for AddScalar backward with multi-element tensor.
/// loss = sum((x + scalar)^2) => gradient depends on x.
#[test]
fn test_backward_add_scalar_fd() {
    let data = vec![0.3, -1.5, 2.0, -0.8];
    let scalar = 1.7_f64;
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[4], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.add_scalar(scalar).unwrap();
    let sq = y.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        let y = x.add_scalar(scalar).unwrap();
        sum_f64(&y.sqr().unwrap())
    });
}

/// FD test for AddScalar chained with mul_scalar.
/// loss = sum((x + a) * b)^2, tests composed scalar ops.
#[test]
fn test_backward_add_mul_scalar_chain_fd() {
    let data = vec![-0.5, 1.2, 0.3];
    let add_val = 0.5_f64;
    let mul_val = 2.0_f64;
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.add_scalar(add_val).unwrap().mul_scalar(mul_val).unwrap();
    let sq = y.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        let y = x.add_scalar(add_val).unwrap().mul_scalar(mul_val).unwrap();
        sum_f64(&y.sqr().unwrap())
    });
}
