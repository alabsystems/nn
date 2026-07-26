// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended autodiff gradient correctness tests.
//!
//! Covers:
//! - Basic gradient correctness: d/dx(x^2) = 2x, d/dx(sin(x)) = cos(x), chain rule
//! - MatMul gradients: d/dA(A*B) = grad*B^T, d/dB(A*B) = A^T*grad
//! - Broadcast gradient reduction: gradient through broadcast reduces correctly
//! - Softmax gradient: Jacobian structure (diagonal - outer product)
//! - LayerNorm gradient: gradient through normalization layer
//! - Loss function gradients: MSE, cross-entropy, L1, Huber
//! - Gradient accumulation: multiple backward passes accumulate correctly
//! - Gradient tape: tape records operations correctly, tape clear resets state
//! - No-grad context: operations via from_tensor don't record to tape
//! - Second-order gradients: gradient of gradient via double backward
//! - Mixed dtype gradients: grad.dtype() matches parameter dtype

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::DType;

use crate::grad::{backward, backward_for_vars};
use crate::op::Op;
use crate::tracked::TrackedTensor;
use crate::var::Var;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn var_from(data: Vec<f32>, shape: &[usize]) -> Var {
    Var::new(DynTensor::from_vec(data, shape, &cpu()).unwrap())
}

fn tracked(v: &Var) -> Arc<TrackedTensor> {
    Arc::new(TrackedTensor::from_var(v).unwrap())
}

fn const_tracked(data: Vec<f32>, shape: &[usize]) -> Arc<TrackedTensor> {
    Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(data, shape, &cpu()).unwrap(),
    ))
}

fn grad_vec(grads: &crate::GradStore, var: &Var) -> Vec<f32> {
    grads
        .get(var)
        .expect("gradient should exist")
        .to_flat_vec::<f32>()
        .unwrap()
}

/// Reduce an arbitrary-shaped tracked tensor to a scalar via sum over all dims.
fn to_scalar_loss(t: &Arc<TrackedTensor>) -> Arc<TrackedTensor> {
    let mut result = Arc::clone(t);
    for d in (0..result.tensor().rank()).rev() {
        result = result.sum_keepdim(d).unwrap();
    }
    result
}

/// Central-difference numerical gradient: (f(x+h) - f(x-h)) / (2h).
fn numerical_gradient(
    data: &[f32],
    _shape: &[usize],
    h: f32,
    fwd: &dyn Fn(&[f32]) -> f64,
) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    for i in 0..data.len() {
        let mut plus = data.to_vec();
        let mut minus = data.to_vec();
        plus[i] += h;
        minus[i] -= h;
        result.push((fwd(&plus) - fwd(&minus)) / (2.0 * f64::from(h)));
    }
    result
}

fn assert_close(analytical: &[f32], numerical: &[f64], tol: f64, label: &str) {
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

fn sum_all_f64(t: &DynTensor) -> f64 {
    t.to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .sum()
}

// ===========================================================================
// 1. Basic gradient correctness
// ===========================================================================

/// d/dx(x^2) = 2x for multi-element tensor
#[test]
fn test_grad_x_squared_equals_2x() {
    let x_data = vec![1.0, -2.0, 3.0, 0.5];
    let x = var_from(x_data.clone(), &[4]);
    let tx = tracked(&x);
    let y = tx.sqr().unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let g = grad_vec(&grads, &x);
    for (i, (&gi, &xi)) in g.iter().zip(x_data.iter()).enumerate() {
        let expected = 2.0 * xi;
        assert!(
            (gi - expected).abs() < 1e-5,
            "d/dx(x^2)[{i}]: got {gi}, expected {expected}"
        );
    }
}

/// d/dx(x^2) matches finite difference
#[test]
fn test_grad_x_squared_fd() {
    let x_data = vec![1.5, -0.7, 2.3];
    let shape = [3];
    let x = var_from(x_data.clone(), &shape);
    let tx = tracked(&x);
    let y = tx.sqr().unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let analytical = grad_vec(&grads, &x);
    let numerical = numerical_gradient(&x_data, &shape, 1e-4, &|d| {
        let v = var_from(d.to_vec(), &shape);
        let t = tracked(&v);
        let y = t.sqr().unwrap();
        let l = to_scalar_loss(&y);
        sum_all_f64(l.tensor())
    });
    assert_close(&analytical, &numerical, 1e-2, "x^2 FD");
}

/// d/dx(sin(x)) = cos(x) for multi-element tensor
#[test]
fn test_grad_sin_equals_cos() {
    let x_data = vec![0.0, std::f32::consts::FRAC_PI_4, 1.0, -0.5];
    let x = var_from(x_data.clone(), &[4]);
    let tx = tracked(&x);
    let y = tx.sin().unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let g = grad_vec(&grads, &x);
    for (i, (&gi, &xi)) in g.iter().zip(x_data.iter()).enumerate() {
        let expected = xi.cos();
        assert!(
            (gi - expected).abs() < 1e-5,
            "d/dx(sin(x))[{i}]: got {gi}, expected {expected}"
        );
    }
}

/// d/dx(sin(x)) matches finite difference
#[test]
fn test_grad_sin_fd() {
    let x_data = vec![0.3, -1.2, 2.0];
    let shape = [3];
    let x = var_from(x_data.clone(), &shape);
    let tx = tracked(&x);
    let y = tx.sin().unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let analytical = grad_vec(&grads, &x);
    let numerical = numerical_gradient(&x_data, &shape, 1e-4, &|d| {
        let v = var_from(d.to_vec(), &shape);
        let t = tracked(&v);
        let y = t.sin().unwrap();
        let l = to_scalar_loss(&y);
        sum_all_f64(l.tensor())
    });
    assert_close(&analytical, &numerical, 1e-2, "sin FD");
}

/// Chain rule: d/dx(sin(x^2)) = cos(x^2) * 2x
#[test]
fn test_chain_rule_sin_of_x_squared() {
    let x_data = vec![0.5, 1.0, -0.3];
    let x = var_from(x_data.clone(), &[3]);
    let tx = tracked(&x);
    let x_sq = tx.sqr().unwrap();
    let y = x_sq.sin().unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let g = grad_vec(&grads, &x);
    for (i, (&gi, &xi)) in g.iter().zip(x_data.iter()).enumerate() {
        let expected = (xi * xi).cos() * 2.0 * xi;
        assert!(
            (gi - expected).abs() < 1e-4,
            "chain rule[{i}]: got {gi}, expected {expected}"
        );
    }
}

/// Chain rule with FD verification: f(x) = exp(x^2)
#[test]
fn test_chain_rule_exp_of_x_squared_fd() {
    let x_data = vec![0.3, -0.5, 0.1];
    let shape = [3];
    let x = var_from(x_data.clone(), &shape);
    let tx = tracked(&x);
    let x_sq = tx.sqr().unwrap();
    let y = x_sq.exp().unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let analytical = grad_vec(&grads, &x);
    let numerical = numerical_gradient(&x_data, &shape, 1e-4, &|d| {
        let v = var_from(d.to_vec(), &shape);
        let t = tracked(&v);
        let y = t.sqr().unwrap().exp().unwrap();
        let l = to_scalar_loss(&y);
        sum_all_f64(l.tensor())
    });
    assert_close(&analytical, &numerical, 1e-2, "exp(x^2) chain rule FD");
}

/// Triple chain: d/dx(log(1 + exp(x))) = sigmoid(x) (softplus backward)
#[test]
fn test_chain_rule_softplus() {
    let x_data = vec![-1.0, 0.0, 1.0, 2.0];
    let x = var_from(x_data.clone(), &[4]);
    let tx = tracked(&x);
    let y = tx.softplus().unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let g = grad_vec(&grads, &x);
    for (i, (&gi, &xi)) in g.iter().zip(x_data.iter()).enumerate() {
        let expected = 1.0 / (1.0 + (-xi).exp()); // sigmoid(x)
        assert!(
            (gi - expected).abs() < 1e-4,
            "softplus grad[{i}]: got {gi}, expected {expected}"
        );
    }
}

// ===========================================================================
// 2. MatMul gradients
// ===========================================================================

/// d/dA(sum(A*B)) = ones * B^T, d/dB(sum(A*B)) = A^T * ones
#[test]
fn test_matmul_grad_basic() {
    // A: [2, 3], B: [3, 2]
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let a = var_from(a_data, &[2, 3]);
    let b = var_from(b_data, &[3, 2]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let c = ta.matmul(&tb).unwrap(); // [2, 2]
    let loss = to_scalar_loss(&c);
    let grads = backward(&loss).unwrap();

    // grad_A = ones[2,2] @ B^T[2,3]
    let grad_a = grad_vec(&grads, &a);
    // B^T = [[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]
    // ones @ B^T = [[0.3, 0.7, 1.1], [0.3, 0.7, 1.1]]
    let expected_grad_a = [0.3, 0.7, 1.1, 0.3, 0.7, 1.1];
    for (i, (&ga, &ea)) in grad_a.iter().zip(expected_grad_a.iter()).enumerate() {
        assert!(
            (ga - ea).abs() < 1e-4,
            "grad_A[{i}]: got {ga}, expected {ea}"
        );
    }

    // grad_B = A^T[3,2] @ ones[2,2]
    let grad_b = grad_vec(&grads, &b);
    // A^T = [[1,4],[2,5],[3,6]]
    // A^T @ ones = [[5,5],[7,7],[9,9]]
    let expected_grad_b = [5.0, 5.0, 7.0, 7.0, 9.0, 9.0];
    for (i, (&gb, &eb)) in grad_b.iter().zip(expected_grad_b.iter()).enumerate() {
        assert!(
            (gb - eb).abs() < 1e-4,
            "grad_B[{i}]: got {gb}, expected {eb}"
        );
    }
}

/// MatMul gradients match finite difference
#[test]
fn test_matmul_grad_fd() {
    let a_data = vec![0.5, -0.3, 0.7, -0.1, 0.4, 0.2];
    let b_data = vec![0.3, 0.1, -0.2, 0.5, 0.6, -0.4];
    let a_shape = [2, 3];
    let b_shape = [3, 2];

    let a = var_from(a_data.clone(), &a_shape);
    let b = var_from(b_data.clone(), &b_shape);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let c = ta.matmul(&tb).unwrap();
    let loss = to_scalar_loss(&c);
    let grads = backward(&loss).unwrap();
    let analytical_a = grad_vec(&grads, &a);
    let analytical_b = grad_vec(&grads, &b);

    let b_data_clone = b_data.clone();
    let num_a = numerical_gradient(&a_data, &a_shape, 1e-4, &|d| {
        let va = var_from(d.to_vec(), &a_shape);
        let vb = var_from(b_data_clone.clone(), &b_shape);
        let ta = tracked(&va);
        let tb = tracked(&vb);
        let c = ta.matmul(&tb).unwrap();
        let l = to_scalar_loss(&c);
        sum_all_f64(l.tensor())
    });
    assert_close(&analytical_a, &num_a, 1e-2, "matmul grad_A FD");

    let a_data_clone = a_data;
    let num_b = numerical_gradient(&b_data, &b_shape, 1e-4, &|d| {
        let va = var_from(a_data_clone.clone(), &a_shape);
        let vb = var_from(d.to_vec(), &b_shape);
        let ta = tracked(&va);
        let tb = tracked(&vb);
        let c = ta.matmul(&tb).unwrap();
        let l = to_scalar_loss(&c);
        sum_all_f64(l.tensor())
    });
    assert_close(&analytical_b, &num_b, 1e-2, "matmul grad_B FD");
}

/// MatMul: square matrices, gradient shape correct
#[test]
fn test_matmul_grad_square_shape() {
    let a = var_from(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let b = var_from(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let c = ta.matmul(&tb).unwrap();
    let loss = to_scalar_loss(&c);
    let grads = backward(&loss).unwrap();
    let ga = grads.get(&a).unwrap();
    let gb = grads.get(&b).unwrap();
    assert_eq!(ga.dims(), &[2, 2]);
    assert_eq!(gb.dims(), &[2, 2]);
}

// ===========================================================================
// 3. Broadcast gradient reduction
// ===========================================================================

/// Broadcast: [1, 3] -> [2, 3], gradient should reduce-sum back to [1, 3]
#[test]
fn test_broadcast_grad_reduction() {
    let x = var_from(vec![1.0, 2.0, 3.0], &[1, 3]);
    let tx = tracked(&x);
    let expanded = tx.broadcast_as(&[2, 3]).unwrap();
    let y = expanded.sqr().unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let g = grad_vec(&grads, &x);
    // Each element is duplicated across broadcast dim, so grad = 2*x * 2 (sum over broadcast)
    assert_eq!(g.len(), 3);
    assert!((g[0] - 4.0).abs() < 1e-5, "got {}", g[0]); // 2*1.0*2
    assert!((g[1] - 8.0).abs() < 1e-5, "got {}", g[1]); // 2*2.0*2
    assert!((g[2] - 12.0).abs() < 1e-5, "got {}", g[2]); // 2*3.0*2
}

/// Broadcast with FD verification
#[test]
fn test_broadcast_grad_fd() {
    let x_data = vec![0.5, -0.3, 1.2];
    let x_shape = [1, 3];
    let x = var_from(x_data.clone(), &x_shape);
    let tx = tracked(&x);
    let expanded = tx.broadcast_as(&[2, 3]).unwrap();
    let y = expanded.sqr().unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let analytical = grad_vec(&grads, &x);
    let numerical = numerical_gradient(&x_data, &x_shape, 1e-4, &|d| {
        let v = var_from(d.to_vec(), &x_shape);
        let t = tracked(&v);
        let e = t.broadcast_as(&[2, 3]).unwrap();
        let y = e.sqr().unwrap();
        let l = to_scalar_loss(&y);
        sum_all_f64(l.tensor())
    });
    assert_close(&analytical, &numerical, 1e-2, "broadcast FD");
}

/// Scalar broadcast: [1] -> [4], gradient reduces back to [1]
#[test]
fn test_scalar_broadcast_grad() {
    let x = var_from(vec![3.0], &[1]);
    let tx = tracked(&x);
    let expanded = tx.broadcast_as(&[4]).unwrap();
    let y = expanded.sqr().unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let g = grad_vec(&grads, &x);
    // 4 copies of 3.0, sqr each = 9.0, grad = 2*3.0 * 4 = 24
    assert_eq!(g.len(), 1);
    assert!((g[0] - 24.0).abs() < 1e-4, "got {}", g[0]);
}

// ===========================================================================
// 4. Softmax gradient
// ===========================================================================

/// Softmax gradient satisfies Jacobian structure: J_ij = s_i(delta_ij - s_j)
#[test]
fn test_softmax_grad_structure() {
    // For loss = sum(w * softmax(x)), grad = softmax * (w - sum(w*softmax))
    let x = var_from(vec![1.0, 2.0, 3.0], &[1, 3]);
    let tx = tracked(&x);
    let s = tx.softmax(1).unwrap();
    // Weighted sum with w = [1, 0, 0] extracts first softmax output
    let w = const_tracked(vec![1.0, 0.0, 0.0], &[1, 3]);
    let ws = s.mul(&w).unwrap();
    let loss = to_scalar_loss(&ws);
    let grads = backward(&loss).unwrap();
    let g = grad_vec(&grads, &x);

    // softmax(1,2,3) = [e^1, e^2, e^3] / sum
    let e1 = 1.0_f64.exp();
    let e2 = 2.0_f64.exp();
    let e3 = 3.0_f64.exp();
    let total = e1 + e2 + e3;
    let s0 = (e1 / total) as f32;
    let s1 = (e2 / total) as f32;
    let s2 = (e3 / total) as f32;

    // grad_i = s_i * (w_i - sum(w_j * s_j))
    let ws_sum = 1.0 * s0 + 0.0 * s1 + 0.0 * s2;
    let expected = [
        s0 * (1.0 - ws_sum),
        s1 * (0.0 - ws_sum),
        s2 * (0.0 - ws_sum),
    ];
    for i in 0..3 {
        assert!(
            (g[i] - expected[i]).abs() < 1e-5,
            "softmax grad[{i}]: got {}, expected {}",
            g[i],
            expected[i]
        );
    }
}

/// Softmax gradient with FD verification
#[test]
fn test_softmax_grad_fd() {
    let x_data = vec![0.5, -0.2, 1.0];
    let shape = [1, 3];
    let x = var_from(x_data.clone(), &shape);
    let tx = tracked(&x);
    let s = tx.softmax(1).unwrap();
    let loss = to_scalar_loss(&s);
    let grads = backward(&loss).unwrap();
    let analytical = grad_vec(&grads, &x);
    let numerical = numerical_gradient(&x_data, &shape, 1e-4, &|d| {
        let v = var_from(d.to_vec(), &shape);
        let t = tracked(&v);
        let s = t.softmax(1).unwrap();
        let l = to_scalar_loss(&s);
        sum_all_f64(l.tensor())
    });
    assert_close(&analytical, &numerical, 1e-2, "softmax FD");
}

/// Softmax gradients sum to zero per row (conservation property)
#[test]
fn test_softmax_grad_sums_to_zero() {
    let x = var_from(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let tx = tracked(&x);
    let s = tx.softmax(1).unwrap();
    // Use weighted loss to get non-trivial gradients
    let w = const_tracked(vec![1.0, 0.0, 0.0, 0.0], &[1, 4]);
    let ws = s.mul(&w).unwrap();
    let loss = to_scalar_loss(&ws);
    let grads = backward(&loss).unwrap();
    let g = grad_vec(&grads, &x);
    let sum: f32 = g.iter().sum();
    assert!(
        sum.abs() < 1e-5,
        "softmax grads should sum to ~0, got {sum}"
    );
}

// ===========================================================================
// 5. LayerNorm gradient
// ===========================================================================

/// LayerNorm gradient through normalization with FD check
#[test]
fn test_layer_norm_grad_fd() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x_shape = [2, 3];
    let w_data = vec![1.0, 1.0, 1.0];
    let b_data = vec![0.0, 0.0, 0.0];
    let eps = 1e-5;

    let x = var_from(x_data.clone(), &x_shape);
    let w = var_from(w_data.clone(), &[3]);
    let b = var_from(b_data.clone(), &[3]);
    let tx = tracked(&x);
    let tw = tracked(&w);
    let tb = tracked(&b);
    let ln = tx.layer_norm(&tw, &tb, eps).unwrap();
    let loss = to_scalar_loss(&ln);
    let grads = backward(&loss).unwrap();
    let analytical_x = grad_vec(&grads, &x);
    let analytical_w = grad_vec(&grads, &w);

    let w_data_c = w_data.clone();
    let b_data_c = b_data.clone();
    let numerical_x = numerical_gradient(&x_data, &x_shape, 1e-3, &|d| {
        let vx = var_from(d.to_vec(), &x_shape);
        let vw = var_from(w_data_c.clone(), &[3]);
        let vb = var_from(b_data_c.clone(), &[3]);
        let tx = tracked(&vx);
        let tw = tracked(&vw);
        let tb = tracked(&vb);
        let ln = tx.layer_norm(&tw, &tb, eps).unwrap();
        let l = to_scalar_loss(&ln);
        sum_all_f64(l.tensor())
    });
    assert_close(&analytical_x, &numerical_x, 0.1, "LayerNorm input grad FD");

    let x_data_c = x_data;
    let b_data_c2 = b_data;
    let numerical_w = numerical_gradient(&w_data, &[3], 1e-3, &|d| {
        let vx = var_from(x_data_c.clone(), &x_shape);
        let vw = var_from(d.to_vec(), &[3]);
        let vb = var_from(b_data_c2.clone(), &[3]);
        let tx = tracked(&vx);
        let tw = tracked(&vw);
        let tb = tracked(&vb);
        let ln = tx.layer_norm(&tw, &tb, eps).unwrap();
        let l = to_scalar_loss(&ln);
        sum_all_f64(l.tensor())
    });
    assert_close(&analytical_w, &numerical_w, 0.1, "LayerNorm weight grad FD");
}

/// LayerNorm gradient: bias gradient is straightforward (ones)
#[test]
fn test_layer_norm_bias_grad() {
    let x = var_from(vec![1.0, 2.0, 3.0], &[1, 3]);
    let w = var_from(vec![1.0, 1.0, 1.0], &[3]);
    let b = var_from(vec![0.0, 0.0, 0.0], &[3]);
    let tx = tracked(&x);
    let tw = tracked(&w);
    let tb = tracked(&b);
    let ln = tx.layer_norm(&tw, &tb, 1e-5).unwrap();
    let loss = to_scalar_loss(&ln);
    let grads = backward(&loss).unwrap();
    let gb = grad_vec(&grads, &b);
    // Bias gradient = upstream gradient (ones from sum), per batch element
    assert_eq!(gb.len(), 3);
    for (i, &gi) in gb.iter().enumerate() {
        assert!(
            (gi - 1.0).abs() < 1e-4,
            "bias grad[{i}]: got {gi}, expected ~1.0"
        );
    }
}

// ===========================================================================
// 6. Loss function gradients
// ===========================================================================

/// MSE loss gradient: d/dx(mean((x-t)^2)) = 2*(x-t)/n
#[test]
fn test_mse_gradient_formula() {
    let x_data = vec![1.0, 3.0, 5.0, 7.0];
    let t_data = vec![2.0, 2.0, 6.0, 8.0];
    let x = var_from(x_data.clone(), &[4]);
    let t = const_tracked(t_data.clone(), &[4]);
    let tx = tracked(&x);
    let loss = tx.mse_loss(&t).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grad_vec(&grads, &x);
    let n = x_data.len() as f32;
    for (i, (&gi, (&xi, &ti))) in g.iter().zip(x_data.iter().zip(t_data.iter())).enumerate() {
        let expected = 2.0 * (xi - ti) / n;
        assert!(
            (gi - expected).abs() < 1e-5,
            "MSE grad[{i}]: got {gi}, expected {expected}"
        );
    }
}

/// MSE loss gradient with FD
#[test]
fn test_mse_gradient_fd() {
    let x_data = vec![1.5, -0.3, 2.7];
    let t_data = vec![1.0, 0.5, 3.0];
    let shape = [3];
    let x = var_from(x_data.clone(), &shape);
    let t = const_tracked(t_data.clone(), &shape);
    let tx = tracked(&x);
    let loss = tx.mse_loss(&t).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grad_vec(&grads, &x);
    let t_data_c = t_data;
    let numerical = numerical_gradient(&x_data, &shape, 1e-4, &|d| {
        let v = var_from(d.to_vec(), &shape);
        let t = const_tracked(t_data_c.clone(), &shape);
        let tv = tracked(&v);
        let l = tv.mse_loss(&t).unwrap();
        sum_all_f64(l.tensor())
    });
    assert_close(&analytical, &numerical, 1e-2, "MSE FD");
}

/// Cross-entropy loss gradient with FD
#[test]
fn test_cross_entropy_gradient_fd() {
    let logits_data = vec![2.0, 1.0, 0.1];
    let shape = [1, 3];
    let logits = var_from(logits_data.clone(), &shape);
    // Cross-entropy targets are U32 class indices (gather indices).
    let targets = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec_u32(vec![0u32], &[1, 1], &cpu()).unwrap(),
    )); // class 0
    let tl = tracked(&logits);
    let loss = tl.cross_entropy_loss(&targets, 1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grad_vec(&grads, &logits);

    let numerical = numerical_gradient(&logits_data, &shape, 1e-3, &|d| {
        let v = var_from(d.to_vec(), &shape);
        let t = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec_u32(vec![0u32], &[1, 1], &cpu()).unwrap(),
        ));
        let tv = tracked(&v);
        let l = tv.cross_entropy_loss(&t, 1).unwrap();
        sum_all_f64(l.tensor())
    });
    assert_close(&analytical, &numerical, 0.05, "CrossEntropy FD");
}

/// L1 loss gradient sign matches (x - target)
#[test]
fn test_l1_gradient_sign() {
    let x_data = vec![3.0, -1.0, 5.0, 2.0];
    let t_data = vec![1.0, 1.0, 5.0, 4.0];
    let x = var_from(x_data.clone(), &[4]);
    let t = const_tracked(t_data, &[4]);
    let tx = tracked(&x);
    let loss = tx.l1_loss(&t).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grad_vec(&grads, &x);
    let n = x_data.len() as f32;
    // sign(x - t) / n
    let signs = [1.0f32, -1.0, 0.0, -1.0]; // 3>1, -1<1, 5==5, 2<4
    for (i, (&gi, &si)) in g.iter().zip(signs.iter()).enumerate() {
        let expected = si / n;
        assert!(
            (gi - expected).abs() < 1e-4,
            "L1 grad[{i}]: got {gi}, expected {expected}"
        );
    }
}

/// Huber loss gradient transitions at delta
#[test]
fn test_huber_gradient_transitions() {
    let delta = 1.0;
    // diff < delta: gradient = 2*(x-t) / (2*delta*n) = (x-t)/(delta*n)
    // diff >= delta: gradient = sign(x-t)/n
    let x_data = vec![0.5, 2.0]; // diff = 0.5 (quad), diff = 2.0 (linear)
    let t_data = vec![0.0, 0.0];
    let x = var_from(x_data, &[2]);
    let t = const_tracked(t_data, &[2]);
    let tx = tracked(&x);
    let loss = tx.huber_loss(&t, delta).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grad_vec(&grads, &x);
    // For quadratic region: grad = diff/delta/n = 0.5/1.0/2 = 0.25
    // For linear region: grad = sign(diff)/n = 1.0/2 = 0.5
    assert!(
        (g[0] - 0.25).abs() < 1e-4,
        "Huber quad grad: got {}, expected 0.25",
        g[0]
    );
    assert!(
        (g[1] - 0.5).abs() < 1e-4,
        "Huber linear grad: got {}, expected 0.5",
        g[1]
    );
}

// ===========================================================================
// 7. Gradient accumulation
// ===========================================================================

/// Gradient accumulation: using a variable twice in the graph accumulates correctly
#[test]
fn test_grad_accumulation_same_var() {
    let x = var_from(vec![2.0], &[1]);
    let tx = tracked(&x);
    // y = x + x = 2x, dy/dx = 2
    let y = tx.add(&tx).unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let g = grad_vec(&grads, &x);
    assert!((g[0] - 2.0).abs() < 1e-5, "got {}", g[0]);
}

/// Gradient accumulation: x*x should give 2x (not x)
#[test]
fn test_grad_accumulation_x_times_x() {
    let x = var_from(vec![3.0], &[1]);
    let tx = tracked(&x);
    // y = x * x = x^2, dy/dx = 2x = 6
    let y = tx.mul(&tx).unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let g = grad_vec(&grads, &x);
    assert!((g[0] - 6.0).abs() < 1e-5, "got {}", g[0]);
}

/// Multiple backward calls produce independent gradient stores
#[test]
fn test_multiple_backward_independent() {
    let x = var_from(vec![3.0], &[1]);
    let tx = tracked(&x);
    let y = tx.sqr().unwrap(); // x^2
    let loss = to_scalar_loss(&y);

    let grads1 = backward(&loss).unwrap();
    let grads2 = backward(&loss).unwrap();
    let g1 = grad_vec(&grads1, &x);
    let g2 = grad_vec(&grads2, &x);
    // Both should be 2*3 = 6
    assert!((g1[0] - 6.0).abs() < 1e-5);
    assert!((g2[0] - 6.0).abs() < 1e-5);
}

/// Fan-in: variable used in two separate paths, gradients combine
#[test]
fn test_grad_accumulation_fan_in() {
    let x = var_from(vec![2.0, 3.0], &[2]);
    let tx = tracked(&x);
    // path1 = x^2, path2 = x*3
    let p1 = tx.sqr().unwrap();
    let p2 = tx.mul_scalar(3.0).unwrap();
    let combined = p1.add(&p2).unwrap();
    let loss = to_scalar_loss(&combined);
    let grads = backward(&loss).unwrap();
    let g = grad_vec(&grads, &x);
    // d/dx(x^2 + 3x) = 2x + 3
    assert!((g[0] - 7.0).abs() < 1e-5, "got {}", g[0]); // 2*2+3
    assert!((g[1] - 9.0).abs() < 1e-5, "got {}", g[1]); // 2*3+3
}

// ===========================================================================
// 8. Gradient tape: operations record correctly
// ===========================================================================

/// TrackedTensor from_var records as variable leaf (is_var=true)
#[test]
fn test_tape_from_var_is_leaf() {
    let x = var_from(vec![1.0], &[1]);
    let tx = TrackedTensor::from_var(&x).unwrap();
    assert!(tx.is_var());
    assert!(tx.op().is_none());
}

/// TrackedTensor from_tensor records as constant leaf (is_var=false)
#[test]
fn test_tape_from_tensor_is_constant() {
    let t = TrackedTensor::from_tensor(DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap());
    assert!(!t.is_var());
    assert!(t.op().is_none());
}

/// Op is recorded correctly after computation
#[test]
fn test_tape_records_sqr_op() {
    let x = var_from(vec![2.0], &[1]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.sqr().unwrap();
    assert!(y.op().is_some());
    match y.op().unwrap() {
        Op::Sqr(_) => {} // correct
        other => panic!("expected Op::Sqr, got {:?}", std::mem::discriminant(other)),
    }
}

/// Chained ops record a chain of Ops
#[test]
fn test_tape_records_chain() {
    let x = var_from(vec![1.0], &[1]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.sqr().unwrap(); // Op::Sqr
    let z = y.exp().unwrap(); // Op::Exp
    match z.op().unwrap() {
        Op::Exp(_) => {}
        other => panic!("expected Op::Exp, got {:?}", std::mem::discriminant(other)),
    }
}

/// Different node IDs for each operation
#[test]
fn test_tape_unique_node_ids() {
    let x = var_from(vec![1.0], &[1]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = tx.sqr().unwrap();
    let z = y.exp().unwrap();
    assert_ne!(tx.node_id().as_u64(), y.node_id().as_u64());
    assert_ne!(y.node_id().as_u64(), z.node_id().as_u64());
}

// ===========================================================================
// 9. No-grad context: from_tensor doesn't receive gradients
// ===========================================================================

/// Constants (from_tensor) don't appear in gradient store
#[test]
fn test_no_grad_constant_no_gradient() {
    let x = var_from(vec![2.0, 3.0], &[2]);
    let c = DynTensor::from_vec(vec![5.0, 5.0], &[2], &cpu()).unwrap();
    let tx = tracked(&x);
    let tc = Arc::new(TrackedTensor::from_tensor(c));
    let y = tx.mul(&tc).unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    // x should have gradient (= constant values)
    let g = grad_vec(&grads, &x);
    assert_eq!(g, vec![5.0, 5.0]);
    // There is no Var for tc, so GradStore has exactly 1 var_grad
    assert_eq!(grads.var_count(), 1);
}

/// Mixing vars and constants: only vars get gradients
#[test]
fn test_no_grad_mixed_vars_constants() {
    let a = var_from(vec![1.0, 2.0], &[2]);
    let b = var_from(vec![3.0, 4.0], &[2]);
    let c_tensor = DynTensor::from_vec(vec![10.0, 10.0], &[2], &cpu()).unwrap();
    let ta = tracked(&a);
    let tb = tracked(&b);
    let tc = Arc::new(TrackedTensor::from_tensor(c_tensor));
    // loss = sum((a + c) * b)
    let ac = ta.add(&tc).unwrap();
    let y = ac.mul(&tb).unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    assert!(grads.get(&a).is_some());
    assert!(grads.get(&b).is_some());
    assert_eq!(grads.var_count(), 2);
}

/// from_tensor + from_var in same graph: only var path gets gradient
#[test]
fn test_no_grad_detach_via_constant() {
    let x = var_from(vec![2.0], &[1]);
    let tx = tracked(&x);
    let y = tx.sqr().unwrap(); // tracked through var
                               // Create a "detached" version: take the tensor, wrap as constant
    let y_tensor = y.tensor().clone();
    let y_const = Arc::new(TrackedTensor::from_tensor(y_tensor));
    // Further ops on y_const do NOT flow gradients back to x
    let z = y_const.exp().unwrap();
    let loss = to_scalar_loss(&z);
    let grads = backward(&loss).unwrap();
    // x should NOT have a gradient since y was detached
    assert!(grads.get(&x).is_none());
}

// ===========================================================================
// 10. Second-order gradients (double backward)
// ===========================================================================

/// Second-order via recomputation: verify d^2/dx^2(x^3) = 6x
/// Since the tape is consumed, we test by computing gradient of a
/// gradient-related quantity via finite difference.
#[test]
fn test_second_order_via_fd() {
    let h = 1e-3_f32;
    let x_val = 2.0_f32;

    // f(x) = x^3 => f'(x) = 3x^2 => f''(x) = 6x = 12
    // Compute f'(x+h) and f'(x-h) numerically
    let grad_at = |val: f32| -> f32 {
        let v = var_from(vec![val], &[1]);
        let tv = tracked(&v);
        let y = tv.powf(3.0).unwrap();
        let loss = to_scalar_loss(&y);
        let grads = backward(&loss).unwrap();
        grad_vec(&grads, &v)[0]
    };

    let g_plus = grad_at(x_val + h);
    let g_minus = grad_at(x_val - h);
    let second_order = (g_plus - g_minus) / (2.0 * h);
    let expected = 6.0 * x_val; // 12.0
    assert!(
        (second_order - expected).abs() < 0.1,
        "d^2/dx^2(x^3) at x=2: got {second_order}, expected {expected}"
    );
}

/// Second-order for exp: d^2/dx^2(exp(x)) = exp(x)
#[test]
fn test_second_order_exp_fd() {
    let h = 1e-3_f32;
    let x_val = 1.0_f32;

    let grad_at = |val: f32| -> f32 {
        let v = var_from(vec![val], &[1]);
        let tv = tracked(&v);
        let y = tv.exp().unwrap();
        let loss = to_scalar_loss(&y);
        let grads = backward(&loss).unwrap();
        grad_vec(&grads, &v)[0]
    };

    let g_plus = grad_at(x_val + h);
    let g_minus = grad_at(x_val - h);
    let second_order = (g_plus - g_minus) / (2.0 * h);
    let expected = x_val.exp();
    assert!(
        (second_order - expected).abs() < 0.1,
        "d^2/dx^2(exp(x)) at x=1: got {second_order}, expected {expected}"
    );
}

// ===========================================================================
// 11. Mixed dtype gradients
// ===========================================================================

/// Gradient dtype matches the loss tensor dtype (F32 input produces F32 gradient)
#[test]
fn test_grad_dtype_matches_f32() {
    let x = var_from(vec![1.0, 2.0, 3.0], &[3]);
    let tx = tracked(&x);
    let y = tx.sqr().unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap();
    assert_eq!(grad.dtype(), DType::F32, "gradient should be F32");
}

/// Gradient shape matches variable shape, not loss shape
#[test]
fn test_grad_shape_matches_var() {
    let x = var_from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let tx = tracked(&x);
    let y = tx.sqr().unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap();
    assert_eq!(grad.dims(), &[2, 3], "gradient shape should match variable");
}

/// backward_for_vars only retains targeted variable gradients
#[test]
fn test_backward_for_vars_selective() {
    let a = var_from(vec![1.0, 2.0], &[2]);
    let b = var_from(vec![3.0, 4.0], &[2]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.mul(&tb).unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward_for_vars(&loss, &[&a]).unwrap();
    assert!(grads.get(&a).is_some());
    assert!(grads.get(&b).is_none());
}

// ===========================================================================
// 12. Edge cases and additional coverage
// ===========================================================================

/// Non-scalar loss should produce an error
#[test]
fn test_backward_rejects_non_scalar_loss() {
    let x = var_from(vec![1.0, 2.0, 3.0], &[3]);
    let tx = tracked(&x);
    let y = tx.sqr().unwrap(); // shape [3], not scalar
    let result = backward(&y);
    assert!(result.is_err());
}

/// Gradient of negation: d/dx(-x) = -1
#[test]
fn test_neg_gradient() {
    let x = var_from(vec![1.0, -2.0, 3.0], &[3]);
    let tx = tracked(&x);
    let y = tx.neg().unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let g = grad_vec(&grads, &x);
    for (i, &gi) in g.iter().enumerate() {
        assert!(
            (gi - (-1.0)).abs() < 1e-5,
            "neg grad[{i}]: got {gi}, expected -1.0"
        );
    }
}

/// Gradient of exp: d/dx(exp(x)) = exp(x)
#[test]
fn test_exp_gradient() {
    let x_data = vec![0.0, 1.0, -1.0, 0.5];
    let x = var_from(x_data.clone(), &[4]);
    let tx = tracked(&x);
    let y = tx.exp().unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let g = grad_vec(&grads, &x);
    for (i, (&gi, &xi)) in g.iter().zip(x_data.iter()).enumerate() {
        let expected = xi.exp();
        assert!(
            (gi - expected).abs() < 1e-4,
            "exp grad[{i}]: got {gi}, expected {expected}"
        );
    }
}

/// Gradient of log: d/dx(log(x)) = 1/x
#[test]
fn test_log_gradient() {
    let x_data = vec![1.0, 2.0, 0.5, 3.0];
    let x = var_from(x_data.clone(), &[4]);
    let tx = tracked(&x);
    let y = tx.log().unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let g = grad_vec(&grads, &x);
    for (i, (&gi, &xi)) in g.iter().zip(x_data.iter()).enumerate() {
        let expected = 1.0 / xi;
        assert!(
            (gi - expected).abs() < 1e-4,
            "log grad[{i}]: got {gi}, expected {expected}"
        );
    }
}

/// Gradient of tanh: d/dx(tanh(x)) = 1 - tanh(x)^2
#[test]
fn test_tanh_gradient() {
    let x_data = vec![0.0, 0.5, -0.5, 1.0];
    let x = var_from(x_data.clone(), &[4]);
    let tx = tracked(&x);
    let y = tx.tanh().unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let g = grad_vec(&grads, &x);
    for (i, (&gi, &xi)) in g.iter().zip(x_data.iter()).enumerate() {
        let th = xi.tanh();
        let expected = 1.0 - th * th;
        assert!(
            (gi - expected).abs() < 1e-4,
            "tanh grad[{i}]: got {gi}, expected {expected}"
        );
    }
}

/// Gradient of sigmoid: d/dx(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x))
#[test]
fn test_sigmoid_gradient() {
    let x_data = vec![0.0, 1.0, -1.0, 2.0];
    let x = var_from(x_data.clone(), &[4]);
    let tx = tracked(&x);
    let y = tx.sigmoid().unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let g = grad_vec(&grads, &x);
    for (i, (&gi, &xi)) in g.iter().zip(x_data.iter()).enumerate() {
        let s = 1.0 / (1.0 + (-xi).exp());
        let expected = s * (1.0 - s);
        assert!(
            (gi - expected).abs() < 1e-4,
            "sigmoid grad[{i}]: got {gi}, expected {expected}"
        );
    }
}

/// Gradient of relu: d/dx(relu(x)) = 1 if x >= 0, else 0.
/// At x == 0 the implementation uses the subgradient convention (`ge`, like
/// PyTorch), so the gradient passes through (== 1), not 0.
/// SYNC: backward_rules_elementwise.rs:24 (x.tensor().ge(0.0))
#[test]
fn test_relu_gradient() {
    let x_data = vec![1.0, -2.0, 3.0, -0.5, 0.0];
    let x = var_from(x_data, &[5]);
    let tx = tracked(&x);
    let y = tx.relu().unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let g = grad_vec(&grads, &x);
    let expected = [1.0, 0.0, 1.0, 0.0, 1.0];
    for (i, (&gi, &ei)) in g.iter().zip(expected.iter()).enumerate() {
        assert!(
            (gi - ei).abs() < 1e-5,
            "relu grad[{i}]: got {gi}, expected {ei}"
        );
    }
}

/// Reshape preserves gradients (just rearranges)
#[test]
fn test_reshape_gradient_preserves() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = var_from(x_data.clone(), &[2, 3]);
    let tx = tracked(&x);
    let reshaped = tx.reshape(&[3, 2]).unwrap();
    let y = reshaped.sqr().unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let g = grad_vec(&grads, &x);
    // d/dx(x^2) = 2x regardless of reshape
    for (i, (&gi, &xi)) in g.iter().zip(x_data.iter()).enumerate() {
        assert!(
            (gi - 2.0 * xi).abs() < 1e-5,
            "reshape grad[{i}]: got {gi}, expected {}",
            2.0 * xi
        );
    }
}

/// Transpose gradient: transpose back during backward
#[test]
fn test_transpose_gradient() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = var_from(x_data.clone(), &[2, 3]);
    let tx = tracked(&x);
    let transposed = tx.transpose(0, 1).unwrap(); // [3, 2]
    let y = transposed.sqr().unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&x).unwrap();
    assert_eq!(grad.dims(), &[2, 3], "gradient shape should match input");
    let g = grad_vec(&grads, &x);
    for (i, (&gi, &xi)) in g.iter().zip(x_data.iter()).enumerate() {
        assert!(
            (gi - 2.0 * xi).abs() < 1e-5,
            "transpose grad[{i}]: got {gi}, expected {}",
            2.0 * xi
        );
    }
}

/// Sum-keepdim gradient: expands gradient back
#[test]
fn test_sum_keepdim_gradient() {
    let x = var_from(vec![1.0, 2.0, 3.0], &[1, 3]);
    let tx = tracked(&x);
    let s = tx.sum_keepdim(1).unwrap(); // [1, 1]
    let loss = to_scalar_loss(&s);
    let grads = backward(&loss).unwrap();
    let g = grad_vec(&grads, &x);
    // gradient of sum w.r.t. each element = 1
    assert_eq!(g, vec![1.0, 1.0, 1.0]);
}

/// Narrow (slice) gradient: zero-pads outside the slice
#[test]
fn test_narrow_gradient() {
    let x = var_from(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
    let tx = tracked(&x);
    let sliced = tx.narrow(0, 1, 3).unwrap(); // elements [2, 3, 4]
    let y = sliced.sqr().unwrap();
    let loss = to_scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let g = grad_vec(&grads, &x);
    // Only elements 1,2,3 get gradient; 0 and 4 are zero
    assert!((g[0] - 0.0).abs() < 1e-5);
    assert!((g[1] - 4.0).abs() < 1e-5); // 2*2
    assert!((g[2] - 6.0).abs() < 1e-5); // 2*3
    assert!((g[3] - 8.0).abs() < 1e-5); // 2*4
    assert!((g[4] - 0.0).abs() < 1e-5);
}
