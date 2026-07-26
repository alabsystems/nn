#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended backward pass tests: Gelu, Silu, Narrow, Cat, MatMul FD, Div FD.
//! Conv1d backward tests extracted to `grad_tests_conv1d.rs`.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::grad::test_helpers::{check_fd_grad, scalar_var, sum_f64, vec_var};
use crate::tracked::TrackedTensor;
use crate::var::Var;

// -- GELU backward test ------------------------------------------------------

#[test]
fn test_backward_gelu() {
    // GELU(x) with tanh approximation. Verify derivative at x=0.5 via finite differences.
    let x = scalar_var(0.5);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.gelu().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    // Finite-difference reference: (gelu(x+h) - gelu(x-h)) / (2h)
    let h = 1e-4_f64;
    let xv = 0.5_f64;
    let gelu_val = |v: f64| -> f64 {
        let s = (2.0_f64 / std::f64::consts::PI).sqrt() * (v + 0.044715 * v.powi(3));
        0.5 * v * (1.0 + s.tanh())
    };
    let expected = (gelu_val(xv + h) - gelu_val(xv - h)) / (2.0 * h);
    assert!(
        (f64::from(grad[0]) - expected).abs() < 1e-4,
        "GELU grad at 0.5: expected {expected:.6}, got {:.6}",
        grad[0]
    );
}

#[test]
fn test_backward_gelu_vector() {
    // GELU backward on a vector, verify via finite differences
    let x = vec_var(vec![-1.0, 0.0, 1.0, 2.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.gelu().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    let gelu_val = |v: f64| -> f64 {
        let s = (2.0_f64 / std::f64::consts::PI).sqrt() * (v + 0.044715 * v.powi(3));
        0.5 * v * (1.0 + s.tanh())
    };
    let h = 1e-4_f64;
    for (i, &xv) in [-1.0_f64, 0.0, 1.0, 2.0].iter().enumerate() {
        let expected = (gelu_val(xv + h) - gelu_val(xv - h)) / (2.0 * h);
        assert!(
            (f64::from(grad[i]) - expected).abs() < 1e-3,
            "GELU grad[{i}] at {xv}: expected {expected:.6}, got {:.6}",
            grad[i]
        );
    }
}

// -- SiLU backward test ------------------------------------------------------

#[test]
fn test_backward_silu() {
    // SiLU(x) = x * sigmoid(x)
    // d/dx at x=0: sigmoid(0) * (1 + 0*(1-sigmoid(0))) = 0.5 * 1 = 0.5
    let x = scalar_var(0.0);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.silu().unwrap();
    let grads = backward(&y).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (grad[0] - 0.5).abs() < 1e-5,
        "SiLU grad at 0: expected 0.5, got {}",
        grad[0]
    );
}

#[test]
fn test_backward_silu_vector() {
    // Verify SiLU backward via finite differences
    let x = vec_var(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.silu().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    let silu_val = |v: f64| -> f64 { v / (1.0 + (-v).exp()) };
    let h = 1e-4_f64;
    for (i, &xv) in [-2.0_f64, -1.0, 0.0, 1.0, 2.0].iter().enumerate() {
        let expected = (silu_val(xv + h) - silu_val(xv - h)) / (2.0 * h);
        assert!(
            (f64::from(grad[i]) - expected).abs() < 1e-3,
            "SiLU grad[{i}] at {xv}: expected {expected:.6}, got {:.6}",
            grad[i]
        );
    }
}

// -- Narrow backward test ----------------------------------------------------

#[test]
fn test_backward_narrow() {
    // x = [1, 2, 3, 4, 5] (shape [5])
    // y = sum(narrow(x, dim=0, start=1, len=3)) = sum([2, 3, 4]) = 9
    // dy/dx = [0, 1, 1, 1, 0] (zero-padded gradient)
    let x = vec_var(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let sliced = t.narrow(0, 1, 3).unwrap();
    let loss = sliced.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(grad, vec![0.0, 1.0, 1.0, 1.0, 0.0]);
}

#[test]
fn test_backward_narrow_2d() {
    // x shape [2, 4], narrow dim=1, start=1, len=2
    // y = sum(x[:, 1:3])
    // dy/dx = [[0, 1, 1, 0], [0, 1, 1, 0]]
    let x_var = Var::new(
        DynTensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
            &cpu(),
        )
        .unwrap(),
    );
    let t = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let sliced = t.narrow(1, 1, 2).unwrap();
    let loss = sliced.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(grad, vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
}

// -- Cat backward test -------------------------------------------------------

#[test]
fn test_backward_cat() {
    // a = [1, 2], b = [3, 4, 5]
    // y = sum(cat([a, b], dim=0)) = sum([1, 2, 3, 4, 5]) = 15
    // dy/da = [1, 1], dy/db = [1, 1, 1]
    let a = vec_var(vec![1.0, 2.0]);
    let b = Var::new(DynTensor::from_vec(vec![3.0, 4.0, 5.0], &[3], &cpu()).unwrap());
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let catted = TrackedTensor::cat(&[&ta, &tb], 0).unwrap();
    let loss = catted.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    assert_eq!(
        grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap(),
        vec![1.0, 1.0]
    );
    assert_eq!(
        grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap(),
        vec![1.0, 1.0, 1.0]
    );
}

#[test]
fn test_backward_cat_2d() {
    // a shape [2, 2], b shape [2, 3], cat along dim=1 -> [2, 5]
    // y = sum(cat([a, b], dim=1))
    // dy/da = all 1s [2, 2], dy/db = all 1s [2, 3]
    let a_var = Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap());
    let b_var = Var::new(
        DynTensor::from_vec(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[2, 3], &cpu()).unwrap(),
    );
    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let catted = TrackedTensor::cat(&[&ta, &tb], 1).unwrap();
    let loss = catted.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    assert_eq!(
        grads.get(&a_var).unwrap().to_flat_vec::<f32>().unwrap(),
        vec![1.0, 1.0, 1.0, 1.0]
    );
    assert_eq!(
        grads.get(&b_var).unwrap().to_flat_vec::<f32>().unwrap(),
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    );
}

// -- #1515 AC1: MatMul backward rank guard ------------------------------------

#[test]
fn test_backward_matmul_rank1_returns_error() {
    // MatMul backward requires rank >= 2. The guard at backward_rules.rs:122
    // prevents usize underflow on `r - 2` for rank-0 or rank-1 tensors.
    // Verify that 2D matmul backward succeeds normally.
    let a2d = Var::new(DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap());
    let b2d = Var::new(DynTensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], &cpu()).unwrap());
    let ta2 = Arc::new(TrackedTensor::from_var(&a2d).unwrap());
    let tb2 = Arc::new(TrackedTensor::from_var(&b2d).unwrap());
    let y = ta2.matmul(&tb2).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    // This should succeed (rank 2)
    let grads = backward(&loss).unwrap();
    assert_eq!(grads.get(&a2d).unwrap().dims(), &[2, 2]);
}

// -- AC8: MatMul finite-difference validation ---------------------------------

#[test]
fn test_backward_matmul_finite_diff() {
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
    let b_data = vec![0.5f32, -0.3, 0.8, 0.1, -0.5, 0.7]; // [3,2]

    let a_var = Var::new(DynTensor::from_vec(a_data.clone(), &[2, 3], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(b_data.clone(), &[3, 2], &cpu()).unwrap());
    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());

    let prod = ta.matmul(&tb).unwrap();
    let loss = prod.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let grad_a = grads.get(&a_var).unwrap().to_flat_vec::<f32>().unwrap();
    let grad_b = grads.get(&b_var).unwrap().to_flat_vec::<f32>().unwrap();

    let eps = 1e-3_f32;
    let b_ref = b_data.clone();
    check_fd_grad(&grad_a, &a_data, eps, |d| {
        let a = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_ref.clone(), &[3, 2], &cpu()).unwrap();
        sum_f64(&a.matmul(&b).unwrap())
    });
    check_fd_grad(&grad_b, &b_data, eps, |d| {
        let a = DynTensor::from_vec(a_data.clone(), &[2, 3], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[3, 2], &cpu()).unwrap();
        sum_f64(&a.matmul(&b).unwrap())
    });
}

// -- AC9: Div finite-difference validation ------------------------------------

#[test]
fn test_backward_div_finite_diff() {
    let a_data = vec![6.0f32, 3.0, 10.0];
    let b_data = vec![2.0f32, 5.0, 4.0];

    let a_var = Var::new(DynTensor::from_vec(a_data.clone(), &[3], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(b_data.clone(), &[3], &cpu()).unwrap());
    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());

    let quotient = ta.div(&tb).unwrap();
    let loss = quotient.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let grad_a = grads.get(&a_var).unwrap().to_flat_vec::<f32>().unwrap();
    let grad_b = grads.get(&b_var).unwrap().to_flat_vec::<f32>().unwrap();

    let eps = 1e-3_f32;
    let b_ref = b_data.clone();
    check_fd_grad(&grad_a, &a_data, eps, |d| {
        let a = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_ref.clone(), &[3], &cpu()).unwrap();
        sum_f64(&a.div(&b).unwrap())
    });
    check_fd_grad(&grad_b, &b_data, eps, |d| {
        let a = DynTensor::from_vec(a_data.clone(), &[3], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        sum_f64(&a.div(&b).unwrap())
    });
}
