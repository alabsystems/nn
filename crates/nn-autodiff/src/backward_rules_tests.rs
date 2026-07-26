// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Backward rules tests: analytical gradient checks and finite-difference
//! numerical gradient verification for core autodiff operations.
//!
//! Tests cover multi-element tensors (not just scalars) to exercise
//! broadcast reduction, shape handling, and element-wise correctness.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::tracked::TrackedTensor;
use crate::var::Var;

/// Helper: create a Var from a flat vec with given shape.
fn var_from(data: Vec<f32>, shape: &[usize]) -> Var {
    Var::new(DynTensor::from_vec(data, shape, &cpu()).unwrap())
}

/// Helper: extract flat gradient for a var from GradStore.
fn grad_vec(grads: &crate::grad::GradStore, var: &Var) -> Vec<f32> {
    grads.get(var).unwrap().to_flat_vec::<f32>().unwrap()
}

/// Central-difference numerical gradient: (f(x+eps) - f(x-eps)) / (2*eps).
fn numerical_grad(data: &[f32], eps: f32, fwd: impl Fn(Vec<f32>) -> f64) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    for i in 0..data.len() {
        let mut plus = data.to_vec();
        let mut minus = data.to_vec();
        plus[i] += eps;
        minus[i] -= eps;
        result.push((fwd(plus) - fwd(minus)) / (2.0 * f64::from(eps)));
    }
    result
}

/// Assert analytical and numerical gradients match within tolerance.
fn assert_grad_close(analytical: &[f32], numerical: &[f64], tol: f64, label: &str) {
    assert_eq!(
        analytical.len(),
        numerical.len(),
        "{label}: length mismatch"
    );
    for (i, (&a, &n)) in analytical.iter().zip(numerical.iter()).enumerate() {
        let err = (f64::from(a) - n).abs();
        assert!(
            err < tol,
            "{label}[{i}]: analytical={a}, numerical={n}, err={err}, tol={tol}"
        );
    }
}

// Add backward: d/dx (x + y) = 1 for both inputs

#[test]
fn test_add_backward_multi_element() {
    let x = var_from(vec![1.0, 2.0, 3.0], &[3]);
    let y = var_from(vec![4.0, 5.0, 6.0], &[3]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
    let sum = tx.add(&ty).unwrap();
    let loss = sum.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    assert_eq!(grad_vec(&grads, &x), vec![1.0, 1.0, 1.0]);
    assert_eq!(grad_vec(&grads, &y), vec![1.0, 1.0, 1.0]);
}

#[test]
fn test_add_backward_2d() {
    let x = var_from(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let y = var_from(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
    let sum = tx.add(&ty).unwrap();
    let loss = sum.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    assert_eq!(grad_vec(&grads, &x), vec![1.0, 1.0, 1.0, 1.0]);
    assert_eq!(grad_vec(&grads, &y), vec![1.0, 1.0, 1.0, 1.0]);
}

// Mul backward: d/dx (x * y) = y, d/dy (x * y) = x

#[test]
fn test_mul_backward_multi_element() {
    let x = var_from(vec![2.0, 3.0, 4.0], &[3]);
    let y = var_from(vec![5.0, 6.0, 7.0], &[3]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
    let prod = tx.mul(&ty).unwrap();
    let loss = prod.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    assert_eq!(grad_vec(&grads, &x), vec![5.0, 6.0, 7.0]);
    assert_eq!(grad_vec(&grads, &y), vec![2.0, 3.0, 4.0]);
}

#[test]
fn test_mul_backward_fd() {
    let x_data = vec![2.0_f32, -1.5, 0.7];
    let y_data = vec![3.0_f32, 0.5, -2.0];

    let x = var_from(x_data.clone(), &[3]);
    let y = var_from(y_data.clone(), &[3]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
    let prod = tx.mul(&ty).unwrap();
    let loss = prod.sqr().unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let xt = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        let yt = DynTensor::from_vec(y_data.clone(), &[3], &cpu()).unwrap();
        let p = xt.mul(&yt).unwrap();
        p.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
    assert_grad_close(&gx, &num, 1e-2, "mul_fd_x");
}

// Sub backward: d/dx (x - y) = 1, d/dy (x - y) = -1

#[test]
fn test_sub_backward_multi_element() {
    let x = var_from(vec![10.0, 20.0, 30.0, 40.0], &[4]);
    let y = var_from(vec![1.0, 2.0, 3.0, 4.0], &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
    let diff = tx.sub(&ty).unwrap();
    let loss = diff.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    assert_eq!(grad_vec(&grads, &x), vec![1.0, 1.0, 1.0, 1.0]);
    assert_eq!(grad_vec(&grads, &y), vec![-1.0, -1.0, -1.0, -1.0]);
}

// Div backward: d/dx (x / y) = 1/y, d/dy (x / y) = -x/y^2

#[test]
fn test_div_backward_multi_element() {
    let x = var_from(vec![6.0, 10.0], &[2]);
    let y = var_from(vec![3.0, 2.0], &[2]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let ty = Arc::new(TrackedTensor::from_var(&y).unwrap());
    let quot = tx.div(&ty).unwrap();
    let loss = quot.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let gy = grad_vec(&grads, &y);
    assert!((gx[0] - 1.0 / 3.0).abs() < 1e-5, "dx[0]={}", gx[0]);
    assert!((gx[1] - 0.5).abs() < 1e-5, "dx[1]={}", gx[1]);
    assert!((gy[0] - (-2.0 / 3.0)).abs() < 1e-5, "dy[0]={}", gy[0]);
    assert!((gy[1] - (-2.5)).abs() < 1e-5, "dy[1]={}", gy[1]);
}

// MatMul backward: gradient shapes match input shapes

#[test]
fn test_matmul_backward_shapes() {
    let a = var_from(vec![1.0; 6], &[2, 3]);
    let b = var_from(vec![1.0; 12], &[3, 4]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.matmul(&tb).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();

    let ga = grads.get(&a).unwrap();
    let gb = grads.get(&b).unwrap();
    assert_eq!(ga.dims(), &[2, 3], "grad_a shape");
    assert_eq!(gb.dims(), &[3, 4], "grad_b shape");
}

#[test]
fn test_matmul_backward_values_fd() {
    let a_data = vec![0.5_f32, -0.3, 1.2, 0.8, -0.6, 0.4];
    let b_full = vec![1.0_f32, 0.3, -0.5, 0.7, 0.2, -0.8];

    let a = var_from(a_data.clone(), &[2, 3]);
    let b = var_from(b_full.clone(), &[3, 2]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.matmul(&tb).unwrap();
    let loss = y
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let ga = grad_vec(&grads, &a);
    let num = numerical_grad(&a_data, 1e-3, |d| {
        let at = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let bt = DynTensor::from_vec(b_full.clone(), &[3, 2], &cpu()).unwrap();
        let y = at.matmul(&bt).unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
    assert_grad_close(&ga, &num, 1e-2, "matmul_fd_a");
}

// Exp backward: d/dx exp(x) = exp(x)

#[test]
fn test_exp_backward_multi_element() {
    let x = var_from(vec![0.0, 1.0, -1.0], &[3]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.exp().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    let e = std::f32::consts::E;
    assert!((g[0] - 1.0).abs() < 1e-5, "g[0]={}", g[0]);
    assert!((g[1] - e).abs() < 1e-5, "g[1]={}", g[1]);
    assert!((g[2] - 1.0 / e).abs() < 1e-5, "g[2]={}", g[2]);
}

#[test]
fn test_exp_backward_fd() {
    let x_data = vec![0.5_f32, -0.3, 1.2, -0.8];
    let x = var_from(x_data.clone(), &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.exp().unwrap();
    let loss = y.sqr().unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        let y = t.exp().unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
    assert_grad_close(&gx, &num, 1e-2, "exp_fd");
}

// Log backward: d/dx log(x) = 1/x

#[test]
fn test_log_backward_multi_element() {
    let x = var_from(vec![1.0, 2.0, 4.0], &[3]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.log().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    assert!((g[0] - 1.0).abs() < 1e-5, "g[0]={}", g[0]);
    assert!((g[1] - 0.5).abs() < 1e-5, "g[1]={}", g[1]);
    assert!((g[2] - 0.25).abs() < 1e-5, "g[2]={}", g[2]);
}

#[test]
fn test_log_backward_fd() {
    let x_data = vec![0.5_f32, 1.0, 2.0, 3.0];
    let x = var_from(x_data.clone(), &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.log().unwrap();
    let loss = y.sqr().unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        let y = t.log().unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
    assert_grad_close(&gx, &num, 1e-2, "log_fd");
}

// Tanh backward: d/dx tanh(x) = 1 - tanh(x)^2

#[test]
fn test_tanh_backward_multi_element() {
    let vals = vec![0.0_f32, 0.5, -0.5, 1.0, -1.0];
    let x = var_from(vals.clone(), &[5]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.tanh().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    for (i, &v) in vals.iter().enumerate() {
        let expected = 1.0 - v.tanh().powi(2);
        assert!(
            (g[i] - expected).abs() < 1e-5,
            "tanh grad[{i}]: expected={expected}, got={}",
            g[i]
        );
    }
}

#[test]
fn test_tanh_backward_fd() {
    let x_data = vec![-1.0_f32, -0.3, 0.0, 0.7, 1.5];
    let x = var_from(x_data.clone(), &[5]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.tanh().unwrap();
    let loss = y.sqr().unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[5], &cpu()).unwrap();
        let y = t.tanh().unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
    assert_grad_close(&gx, &num, 1e-2, "tanh_fd");
}

// ReLU backward: 0 for x < 0, 1 for x > 0

#[test]
fn test_relu_backward_multi_element() {
    let x = var_from(vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], &[6]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.relu().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    assert_eq!(g, vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_relu_backward_fd() {
    let x_data = vec![-1.5_f32, -0.5, 0.5, 1.5, 2.5];
    let x = var_from(x_data.clone(), &[5]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.relu().unwrap();
    let loss = y.sqr().unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[5], &cpu()).unwrap();
        let y = t.relu().unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
    assert_grad_close(&gx, &num, 1e-2, "relu_fd");
}

// Sum backward: gradient broadcasts back

#[test]
fn test_sum_backward_broadcasts() {
    let x = var_from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let summed = tx.sum_keepdim(1).unwrap();
    let loss = summed.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    assert_eq!(g, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_sum_backward_preserves_shape() {
    let x = var_from(vec![1.0; 12], &[3, 4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let summed = tx.sum_keepdim(1).unwrap();
    let loss = summed.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x).unwrap();
    assert_eq!(gx.dims(), &[3, 4], "gradient shape must match input shape");
}

// Mean backward: gradient is 1/N

#[test]
fn test_mean_backward_1d() {
    let x = var_from(vec![2.0, 4.0, 6.0, 8.0], &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let m = tx.mean_keepdim(0).unwrap();
    let grads = backward(&m).unwrap();

    let g = grad_vec(&grads, &x);
    for (i, &v) in g.iter().enumerate() {
        assert!(
            (v - 0.25).abs() < 1e-6,
            "mean grad[{i}] expected 0.25, got {v}"
        );
    }
}

#[test]
fn test_mean_backward_2d() {
    let x = var_from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let m = tx.mean_keepdim(1).unwrap();
    let loss = m.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    let expected = 1.0 / 3.0;
    for (i, &v) in g.iter().enumerate() {
        assert!(
            (v - expected).abs() < 1e-6,
            "mean grad[{i}] expected {expected}, got {v}"
        );
    }
}

// Chain rule: d/dx exp(2x) = 2*exp(2x)

#[test]
fn test_chain_rule_exp_2x() {
    let vals = vec![0.5_f32, 1.0, -0.5];
    let x = var_from(vals.clone(), &[3]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let doubled = tx.mul_scalar(2.0).unwrap();
    let y = doubled.exp().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    for (i, &v) in vals.iter().enumerate() {
        let expected = 2.0 * (2.0 * v).exp();
        assert!(
            (g[i] - expected).abs() < 1e-4,
            "chain_rule[{i}]: expected={expected}, got={}",
            g[i]
        );
    }
}

#[test]
fn test_chain_rule_exp_2x_fd() {
    let x_data = vec![0.5_f32, 1.0, -0.5];
    let x = var_from(x_data.clone(), &[3]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let doubled = tx.mul_scalar(2.0).unwrap();
    let y = doubled.exp().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-4, |d| {
        let t = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        let y = t.mul_scalar(2.0).unwrap().exp().unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
    assert_grad_close(&gx, &num, 1e-2, "chain_rule_exp2x_fd");
}

// Numerical gradient check for composite expressions

#[test]
fn test_numerical_grad_composite_expression() {
    // f(x) = sum(tanh(x^2 + 1))
    let x_data = vec![0.3_f32, -0.7, 1.2, -0.1];
    let x = var_from(x_data.clone(), &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let x_sq = tx.sqr().unwrap();
    let x_sq_plus_1 = x_sq.add_scalar(1.0).unwrap();
    let y = x_sq_plus_1.tanh().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-4, |d| {
        let t = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        let y = t.sqr().unwrap().add_scalar(1.0).unwrap().tanh().unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
    assert_grad_close(&gx, &num, 1e-2, "composite_tanh_sqr");
}

#[test]
fn test_numerical_grad_log_mul_exp() {
    // f(x) = sum(log(exp(x) * 2)) = sum(x + log(2)), d/dx = 1
    let x_data = vec![0.5_f32, 1.0, 2.0];
    let x = var_from(x_data.clone(), &[3]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let ex = tx.exp().unwrap();
    let scaled = ex.mul_scalar(2.0).unwrap();
    let y = scaled.log().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    for (i, &v) in gx.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-4,
            "log_mul_exp grad[{i}] expected 1.0, got {v}"
        );
    }

    let num = numerical_grad(&x_data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        let y = t.exp().unwrap().mul_scalar(2.0).unwrap().log().unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
    assert_grad_close(&gx, &num, 1e-2, "log_mul_exp_fd");
}

// Chain rule: d/dx (x^2 * sin(x)) = 2x*sin(x) + x^2*cos(x)

#[test]
fn test_chain_rule_sqr_mul_sin_fd() {
    let x_data = vec![0.5_f32, 1.0, -0.7, 2.0];
    let x = var_from(x_data.clone(), &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let x_sq = tx.sqr().unwrap();
    let sin_x = tx.sin().unwrap();
    let prod = x_sq.mul(&sin_x).unwrap();
    let loss = prod.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    for (i, &v) in x_data.iter().enumerate() {
        let expected = 2.0 * v * v.sin() + v * v * v.cos();
        assert!(
            (gx[i] - expected).abs() < 1e-4,
            "sqr_sin grad[{i}]: expected={expected}, got={}",
            gx[i]
        );
    }

    let num = numerical_grad(&x_data, 1e-4, |d| {
        let t = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        let sq = t.sqr().unwrap();
        let s = t.sin().unwrap();
        let p = sq.mul(&s).unwrap();
        p.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
    assert_grad_close(&gx, &num, 1e-2, "sqr_sin_fd");
}

// Sigmoid backward FD

#[test]
fn test_sigmoid_backward_multi_element_fd() {
    let x_data = vec![-2.0_f32, -1.0, 0.0, 1.0, 2.0];
    let x = var_from(x_data.clone(), &[5]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.sigmoid().unwrap();
    let loss = y.sqr().unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[5], &cpu()).unwrap();
        let y = t.sigmoid().unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
    assert_grad_close(&gx, &num, 1e-2, "sigmoid_fd");
}

// Neg backward: d/dx (-x) = -1

#[test]
fn test_neg_backward_multi_element() {
    let x = var_from(vec![1.0, -2.0, 3.0, -4.0], &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.neg().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    assert_eq!(grad_vec(&grads, &x), vec![-1.0, -1.0, -1.0, -1.0]);
}

// Sqr backward: d/dx x^2 = 2x

#[test]
fn test_sqr_backward_multi_element() {
    let vals = vec![1.0_f32, -2.0, 3.0, -0.5];
    let x = var_from(vals.clone(), &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.sqr().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    for (i, &v) in vals.iter().enumerate() {
        let expected = 2.0 * v;
        assert!(
            (g[i] - expected).abs() < 1e-5,
            "sqr grad[{i}]: expected={expected}, got={}",
            g[i]
        );
    }
}

// Fan-out: gradient accumulation

#[test]
fn test_gradient_accumulation_fan_out() {
    let x = var_from(vec![3.0, -1.0, 0.5], &[3]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.add(&tx).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    assert_eq!(grad_vec(&grads, &x), vec![2.0, 2.0, 2.0]);
}

#[test]
fn test_gradient_accumulation_mul_self() {
    let x = var_from(vec![2.0, 3.0], &[2]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.mul(&tx).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    assert!((g[0] - 4.0).abs() < 1e-5, "d/dx(x^2) at x=2 should be 4");
    assert!((g[1] - 6.0).abs() < 1e-5, "d/dx(x^2) at x=3 should be 6");
}

// Sqrt backward FD

#[test]
fn test_sqrt_backward_multi_element_fd() {
    let x_data = vec![1.0_f32, 4.0, 9.0, 16.0];
    let x = var_from(x_data.clone(), &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.sqrt().unwrap();
    let loss = y.sqr().unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        let y = t.sqrt().unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
    assert_grad_close(&gx, &num, 1e-2, "sqrt_fd");
}

// Abs backward: sign(x)

#[test]
fn test_abs_backward_multi_element() {
    let x = var_from(vec![-3.0, -1.0, 0.0, 1.0, 5.0], &[5]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.abs().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    assert_eq!(g, vec![-1.0, -1.0, 0.0, 1.0, 1.0]);
}

// Powf backward FD

#[test]
fn test_powf_backward_fd() {
    let x_data = vec![1.0_f32, 2.0, 0.5, 3.0];
    let p = 2.5_f64;

    let x = var_from(x_data.clone(), &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.powf(p).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    for (i, &v) in x_data.iter().enumerate() {
        let expected = 2.5 * f64::from(v).powf(1.5);
        assert!(
            (f64::from(gx[i]) - expected).abs() < 1e-3,
            "powf grad[{i}]: expected={expected}, got={}",
            gx[i]
        );
    }

    let num = numerical_grad(&x_data, 1e-4, |d| {
        let t = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        let y = t.powf(p).unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
    assert_grad_close(&gx, &num, 2e-2, "powf_fd");
}

// Sin/Cos backward FD

#[test]
fn test_sin_backward_fd() {
    let x_data = vec![0.0_f32, 0.5, 1.0, -1.0, std::f32::consts::PI];
    let x = var_from(x_data.clone(), &[5]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.sin().unwrap();
    let loss = y.sqr().unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-4, |d| {
        let t = DynTensor::from_vec(d, &[5], &cpu()).unwrap();
        let y = t.sin().unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
    assert_grad_close(&gx, &num, 1e-2, "sin_fd");
}

#[test]
fn test_cos_backward_fd() {
    let x_data = vec![0.0_f32, 0.5, 1.0, -1.0, std::f32::consts::PI];
    let x = var_from(x_data.clone(), &[5]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.cos().unwrap();
    let loss = y.sqr().unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-4, |d| {
        let t = DynTensor::from_vec(d, &[5], &cpu()).unwrap();
        let y = t.cos().unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
    assert_grad_close(&gx, &num, 1e-2, "cos_fd");
}

// Recip backward FD

#[test]
fn test_recip_backward_fd() {
    let x_data = vec![0.5_f32, 1.0, 2.0, -1.5];
    let x = var_from(x_data.clone(), &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.recip().unwrap();
    let loss = y.sqr().unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-4, |d| {
        let t = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        let y = t.recip().unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
    assert_grad_close(&gx, &num, 1e-2, "recip_fd");
}

// Multi-step chain: d/dx sigmoid(tanh(x))

#[test]
fn test_chain_sigmoid_tanh_fd() {
    let x_data = vec![-1.0_f32, 0.0, 0.5, 1.5];
    let x = var_from(x_data.clone(), &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.tanh().unwrap().sigmoid().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-4, |d| {
        let t = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        let y = t.tanh().unwrap().sigmoid().unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
    assert_grad_close(&gx, &num, 1e-2, "sigmoid_tanh_chain_fd");
}

// Reshape backward

#[test]
fn test_reshape_backward_preserves_gradient() {
    let x_data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = var_from(x_data.clone(), &[2, 3]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let flat = tx.reshape(&[6]).unwrap();
    let y = flat.sqr().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x).unwrap();
    assert_eq!(gx.dims(), &[2, 3], "gradient shape must match original");
    let g = gx.to_flat_vec::<f32>().unwrap();
    for (i, &v) in x_data.iter().enumerate() {
        assert!(
            (g[i] - 2.0 * v).abs() < 1e-5,
            "reshape grad[{i}]: expected {}, got {}",
            2.0 * v,
            g[i]
        );
    }
}

// Transpose backward FD

#[test]
fn test_transpose_backward_fd() {
    let x_data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = var_from(x_data.clone(), &[2, 3]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let xt = tx.transpose(0, 1).unwrap();
    let y = xt.sqr().unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x).unwrap();
    assert_eq!(gx.dims(), &[2, 3], "gradient shape must match original");

    let g = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let y = t.transpose(0, 1).unwrap().sqr().unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
    assert_grad_close(&g, &num, 1e-2, "transpose_fd");
}

// Clamp backward

#[test]
fn test_clamp_backward() {
    let x = var_from(vec![-2.0, -0.5, 0.0, 0.5, 2.0], &[5]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.clamp(-1.0, 1.0).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    assert_eq!(grad_vec(&grads, &x), vec![0.0, 1.0, 1.0, 1.0, 0.0]);
}

// Mixed: add + mul + sum multi-variable

#[test]
fn test_mixed_add_mul_sum_fd() {
    // f(a, b) = sum((a + b) * a) = sum(a^2 + a*b)
    // df/da = 2a + b, df/db = a
    let a_data = vec![1.0_f32, -2.0, 3.0];
    let b_data = vec![4.0_f32, -1.0, 2.0];

    let a = var_from(a_data.clone(), &[3]);
    let b = var_from(b_data.clone(), &[3]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let sum_ab = ta.add(&tb).unwrap();
    let prod = sum_ab.mul(&ta).unwrap();
    let loss = prod.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let ga = grad_vec(&grads, &a);
    let gb = grad_vec(&grads, &b);
    for i in 0..3 {
        let expected_a = 2.0 * a_data[i] + b_data[i];
        let expected_b = a_data[i];
        assert!(
            (ga[i] - expected_a).abs() < 1e-4,
            "da[{i}]: expected={expected_a}, got={}",
            ga[i]
        );
        assert!(
            (gb[i] - expected_b).abs() < 1e-4,
            "db[{i}]: expected={expected_b}, got={}",
            gb[i]
        );
    }
}

// Gelu backward FD

#[test]
fn test_gelu_backward_fd() {
    let x_data = vec![-1.5_f32, -0.5, 0.0, 0.5, 1.5];
    let x = var_from(x_data.clone(), &[5]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.gelu().unwrap();
    let loss = y.sqr().unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[5], &cpu()).unwrap();
        let y = t.gelu().unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
    assert_grad_close(&gx, &num, 1e-2, "gelu_fd");
}

// SiLU backward FD

#[test]
fn test_silu_backward_fd() {
    let x_data = vec![-2.0_f32, -0.5, 0.0, 1.0, 2.0];
    let x = var_from(x_data.clone(), &[5]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.silu().unwrap();
    let loss = y.sqr().unwrap().sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[5], &cpu()).unwrap();
        let y = t.silu().unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
    assert_grad_close(&gx, &num, 1e-2, "silu_fd");
}
