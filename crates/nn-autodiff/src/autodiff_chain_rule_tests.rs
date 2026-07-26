// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended chain rule, gradient accumulation, tape management, no-grad context,
//! and loss backward tests for the nn-autodiff crate.
//!
//! Covers:
//! - Chain rule through add, mul, sub, div
//! - Chain rule through matmul
//! - Chain rule through nested operations
//! - Gradient accumulation for shared variables
//! - No-grad context (detach)
//! - TrackedTensor creation and operations
//! - Backward pass execution
//! - Gradient tape management
//! - Loss computation backward (MSE, L1, Huber)
//! - Multi-output gradient computation

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::DType;

use crate::grad::{backward, backward_for_vars, GradStore};
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

fn const_tensor(data: Vec<f32>, shape: &[usize]) -> Arc<TrackedTensor> {
    Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(data, shape, &cpu()).unwrap(),
    ))
}

fn grad_vec(grads: &GradStore, v: &Var) -> Vec<f32> {
    grads.get(v).unwrap().to_flat_vec::<f32>().unwrap()
}

/// Make a scalar loss from an arbitrary-shaped tensor by sum-of-all-elements.
fn scalar_loss(t: &Arc<TrackedTensor>) -> Arc<TrackedTensor> {
    let mut result = Arc::clone(t);
    for d in (0..result.tensor().rank()).rev() {
        result = result.sum_keepdim(d).unwrap();
    }
    result
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, msg: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{msg}: length mismatch: got {}, expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{msg}[{i}]: got {a}, expected {e}, diff={}",
            (a - e).abs()
        );
    }
}

/// Central-difference numerical gradient: (f(x+eps) - f(x-eps)) / (2*eps).
fn numerical_grad(
    data: &[f32],
    shape: &[usize],
    eps: f32,
    fwd: impl Fn(DynTensor) -> f64,
) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    for i in 0..data.len() {
        let mut plus = data.to_vec();
        let mut minus = data.to_vec();
        plus[i] += eps;
        minus[i] -= eps;
        let tp = DynTensor::from_vec(plus, shape, &cpu()).unwrap();
        let tm = DynTensor::from_vec(minus, shape, &cpu()).unwrap();
        result.push((fwd(tp) - fwd(tm)) / (2.0 * f64::from(eps)));
    }
    result
}

fn tensor_sum_f64(t: &DynTensor) -> f64 {
    t.to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .sum()
}

// ===========================================================================
// 1. Chain rule through add
// ===========================================================================

#[test]
fn test_chain_rule_add_scalar_sum() {
    // loss = sum(a + b), d/da = 1, d/db = 1
    let a = var_from(vec![1.0, 2.0, 3.0, 4.0], &[4]);
    let b = var_from(vec![5.0, 6.0, 7.0, 8.0], &[4]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.add(&tb).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    assert_close(&grad_vec(&grads, &a), &[1.0; 4], 1e-6, "d/da add");
    assert_close(&grad_vec(&grads, &b), &[1.0; 4], 1e-6, "d/db add");
}

#[test]
fn test_chain_rule_add_then_sqr() {
    // f(a,b) = sum((a + b)^2), df/da = 2(a+b), df/db = 2(a+b)
    let a = var_from(vec![1.0, 2.0], &[2]);
    let b = var_from(vec![3.0, 4.0], &[2]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.add(&tb).unwrap().sqr().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    // 2*(1+3) = 8, 2*(2+4) = 12
    assert_close(&grad_vec(&grads, &a), &[8.0, 12.0], 1e-5, "d/da add_sqr");
    assert_close(&grad_vec(&grads, &b), &[8.0, 12.0], 1e-5, "d/db add_sqr");
}

#[test]
fn test_chain_rule_add_with_activation() {
    // f(a,b) = sum(relu(a + b))
    // d/da = 1 where a+b > 0, else 0
    let a = var_from(vec![-2.0, 1.0, -1.0, 3.0], &[4]);
    let b = var_from(vec![1.0, -3.0, 2.0, -1.0], &[4]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.add(&tb).unwrap().relu().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    // a+b = [-1, -2, 1, 2], relu' = [0, 0, 1, 1]
    assert_close(
        &grad_vec(&grads, &a),
        &[0.0, 0.0, 1.0, 1.0],
        1e-6,
        "d/da add_relu",
    );
    assert_close(
        &grad_vec(&grads, &b),
        &[0.0, 0.0, 1.0, 1.0],
        1e-6,
        "d/db add_relu",
    );
}

// ===========================================================================
// 2. Chain rule through mul
// ===========================================================================

#[test]
fn test_chain_rule_mul_then_sum() {
    // f(a,b) = sum(a * b), df/da = b, df/db = a
    let a = var_from(vec![2.0, 3.0, 4.0], &[3]);
    let b = var_from(vec![5.0, 6.0, 7.0], &[3]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.mul(&tb).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    assert_close(&grad_vec(&grads, &a), &[5.0, 6.0, 7.0], 1e-6, "d/da mul");
    assert_close(&grad_vec(&grads, &b), &[2.0, 3.0, 4.0], 1e-6, "d/db mul");
}

#[test]
fn test_chain_rule_mul_then_exp() {
    // f(a,b) = sum(exp(a * b)), df/da = b * exp(a*b)
    let a_data = vec![0.5, 1.0];
    let b_data = vec![1.0, 0.5];
    let a = var_from(a_data.clone(), &[2]);
    let b = var_from(b_data.clone(), &[2]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.mul(&tb).unwrap().exp().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let ga = grad_vec(&grads, &a);
    for i in 0..2 {
        let expected = b_data[i] * (a_data[i] * b_data[i]).exp();
        assert!(
            (ga[i] - expected).abs() < 1e-4,
            "mul_exp d/da[{i}]: expected={expected}, got={}",
            ga[i]
        );
    }
}

#[test]
fn test_chain_rule_mul_chained_three_vars() {
    // f(a,b,c) = sum(a * b * c), df/da = b*c, df/db = a*c, df/dc = a*b
    let a = var_from(vec![2.0, 3.0], &[2]);
    let b = var_from(vec![4.0, 5.0], &[2]);
    let c = var_from(vec![6.0, 7.0], &[2]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let tc = tracked(&c);
    let ab = ta.mul(&tb).unwrap();
    let abc = ab.mul(&tc).unwrap();
    let loss = scalar_loss(&abc);
    let grads = backward(&loss).unwrap();
    assert_close(&grad_vec(&grads, &a), &[24.0, 35.0], 1e-5, "d/da");
    assert_close(&grad_vec(&grads, &b), &[12.0, 21.0], 1e-5, "d/db");
    assert_close(&grad_vec(&grads, &c), &[8.0, 15.0], 1e-5, "d/dc");
}

// ===========================================================================
// 3. Chain rule through sub
// ===========================================================================

#[test]
fn test_chain_rule_sub_then_sqr() {
    // f(a,b) = sum((a - b)^2), df/da = 2(a-b), df/db = -2(a-b)
    let a = var_from(vec![5.0, 3.0], &[2]);
    let b = var_from(vec![2.0, 1.0], &[2]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.sub(&tb).unwrap().sqr().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    // a-b = [3, 2], 2*(a-b) = [6, 4]
    assert_close(&grad_vec(&grads, &a), &[6.0, 4.0], 1e-5, "d/da sub_sqr");
    assert_close(&grad_vec(&grads, &b), &[-6.0, -4.0], 1e-5, "d/db sub_sqr");
}

#[test]
fn test_chain_rule_sub_with_sigmoid() {
    // f(a,b) = sum(sigmoid(a - b))
    let a_data = vec![1.0, 0.0, -1.0];
    let b_data = vec![0.0, 1.0, -2.0];
    let a = var_from(a_data.clone(), &[3]);
    let b = var_from(b_data.clone(), &[3]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.sub(&tb).unwrap().sigmoid().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let ga = grad_vec(&grads, &a);
    let gb = grad_vec(&grads, &b);
    // d/da sigmoid(a-b) = sigmoid(a-b)*(1-sigmoid(a-b))
    for i in 0..3 {
        let diff = a_data[i] - b_data[i];
        let s = 1.0 / (1.0 + (-diff).exp());
        let expected = s * (1.0 - s);
        assert!(
            (ga[i] - expected).abs() < 1e-5,
            "sub_sigmoid d/da[{i}]: expected={expected}, got={}",
            ga[i]
        );
        assert!(
            (gb[i] + expected).abs() < 1e-5,
            "sub_sigmoid d/db[{i}]: expected={}, got={}",
            -expected,
            gb[i]
        );
    }
}

// ===========================================================================
// 4. Chain rule through div
// ===========================================================================

#[test]
fn test_chain_rule_div_then_sum() {
    // f(a,b) = sum(a / b), df/da = 1/b, df/db = -a/b^2
    let a = var_from(vec![6.0, 10.0], &[2]);
    let b = var_from(vec![3.0, 2.0], &[2]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.div(&tb).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let ga = grad_vec(&grads, &a);
    let gb = grad_vec(&grads, &b);
    assert!((ga[0] - 1.0 / 3.0).abs() < 1e-5);
    assert!((ga[1] - 0.5).abs() < 1e-5);
    assert!((gb[0] - (-6.0 / 9.0)).abs() < 1e-5);
    assert!((gb[1] - (-10.0 / 4.0)).abs() < 1e-5);
}

#[test]
fn test_chain_rule_div_then_sqr_fd() {
    // f(a,b) = sum((a/b)^2), verify with finite differences
    let a_data = vec![2.0_f32, 3.0];
    let b_data = vec![4.0_f32, 5.0];
    let a = var_from(a_data.clone(), &[2]);
    let b = var_from(b_data.clone(), &[2]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.div(&tb).unwrap().sqr().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let ga = grad_vec(&grads, &a);
    let num_a = numerical_grad(&a_data, &[2], 1e-3, |t| {
        let bt = DynTensor::from_vec(b_data.clone(), &[2], &cpu()).unwrap();
        let r = t.div(&bt).unwrap().sqr().unwrap();
        tensor_sum_f64(&r)
    });
    for i in 0..2 {
        assert!(
            (f64::from(ga[i]) - num_a[i]).abs() < 1e-2,
            "div_sqr_fd d/da[{i}]: analytical={}, numerical={}",
            ga[i],
            num_a[i]
        );
    }
}

#[test]
fn test_chain_rule_div_reciprocal_identity() {
    // a / b should give same gradient as a * (1/b) for variable a
    let a = var_from(vec![6.0, 10.0], &[2]);
    let b = var_from(vec![3.0, 5.0], &[2]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y_div = ta.div(&tb).unwrap();
    let loss_div = scalar_loss(&y_div);
    let grads_div = backward(&loss_div).unwrap();

    let a2 = var_from(vec![6.0, 10.0], &[2]);
    let b2 = var_from(vec![3.0, 5.0], &[2]);
    let ta2 = tracked(&a2);
    let tb2 = tracked(&b2);
    let y_recip = ta2.mul(&tb2.recip().unwrap()).unwrap();
    let loss_recip = scalar_loss(&y_recip);
    let grads_recip = backward(&loss_recip).unwrap();

    assert_close(
        &grad_vec(&grads_div, &a),
        &grad_vec(&grads_recip, &a2),
        1e-5,
        "div vs mul_recip: d/da",
    );
}

// ===========================================================================
// 5. Chain rule through matmul
// ===========================================================================

#[test]
fn test_chain_rule_matmul_grad_shapes() {
    // A [2,3] @ B [3,4] = C [2,4], dL/dA has shape [2,3], dL/dB has shape [3,4]
    let a = var_from(vec![1.0; 6], &[2, 3]);
    let b = var_from(vec![1.0; 12], &[3, 4]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.matmul(&tb).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    assert_eq!(grads.get(&a).unwrap().dims(), &[2, 3]);
    assert_eq!(grads.get(&b).unwrap().dims(), &[3, 4]);
}

#[test]
fn test_chain_rule_matmul_values() {
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]], C = A@B
    // loss = sum(C), dL/dA = B^T summed properly
    let a = var_from(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = var_from(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.matmul(&tb).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    // dL/dA = ones @ B^T = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
    let ga = grad_vec(&grads, &a);
    assert_close(&ga, &[11.0, 15.0, 11.0, 15.0], 1e-5, "matmul d/da");

    // dL/dB = A^T @ ones = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
    let gb = grad_vec(&grads, &b);
    assert_close(&gb, &[4.0, 4.0, 6.0, 6.0], 1e-5, "matmul d/db");
}

#[test]
fn test_chain_rule_matmul_then_relu_fd() {
    // f(A) = sum(relu(A @ B)), verify with finite differences
    let a_data = vec![0.5_f32, -0.3, 1.2, 0.8];
    let b_data = vec![1.0_f32, -1.0, 0.5, 0.5];
    let a = var_from(a_data.clone(), &[2, 2]);
    let b = var_from(b_data.clone(), &[2, 2]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.matmul(&tb).unwrap().relu().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let ga = grad_vec(&grads, &a);
    let num = numerical_grad(&a_data, &[2, 2], 1e-3, |t| {
        let bt = DynTensor::from_vec(b_data.clone(), &[2, 2], &cpu()).unwrap();
        let r = t.matmul(&bt).unwrap().relu().unwrap();
        tensor_sum_f64(&r)
    });
    for i in 0..4 {
        assert!(
            (f64::from(ga[i]) - num[i]).abs() < 1e-2,
            "matmul_relu_fd d/da[{i}]: analytical={}, numerical={}",
            ga[i],
            num[i]
        );
    }
}

#[test]
fn test_chain_rule_matmul_chain_two_layers() {
    // f(W1, W2) = sum(x @ W1 @ W2) where x is constant input
    let w1 = var_from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let w2 = var_from(vec![1.0, 0.5, 0.5, 1.0, 0.0, 1.5], &[3, 2]);
    let x = const_tensor(vec![1.0, 1.0], &[1, 2]);
    let h = x.matmul(&tracked(&w1)).unwrap();
    let y = h.matmul(&tracked(&w2)).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    // Both should have correct gradient shapes
    assert_eq!(grads.get(&w1).unwrap().dims(), &[2, 3]);
    assert_eq!(grads.get(&w2).unwrap().dims(), &[3, 2]);

    // Verify with finite differences for w1
    let w1_data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gw1 = grad_vec(&grads, &w1);
    let num = numerical_grad(&w1_data, &[2, 3], 1e-3, |t| {
        let xt = DynTensor::from_vec(vec![1.0, 1.0], &[1, 2], &cpu()).unwrap();
        let w2t = DynTensor::from_vec(vec![1.0, 0.5, 0.5, 1.0, 0.0, 1.5], &[3, 2], &cpu()).unwrap();
        let r = xt.matmul(&t).unwrap().matmul(&w2t).unwrap();
        tensor_sum_f64(&r)
    });
    for i in 0..6 {
        assert!(
            (f64::from(gw1[i]) - num[i]).abs() < 1e-2,
            "two_layer_matmul_fd d/dw1[{i}]: analytical={}, numerical={}",
            gw1[i],
            num[i]
        );
    }
}

// ===========================================================================
// 6. Chain rule through nested operations
// ===========================================================================

#[test]
fn test_nested_exp_of_sqr() {
    // f(x) = sum(exp(x^2)), df/dx = 2x * exp(x^2)
    let vals = vec![0.5_f32, 1.0, -0.5];
    let x = var_from(vals.clone(), &[3]);
    let tx = tracked(&x);
    let y = tx.sqr().unwrap().exp().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    for (i, &v) in vals.iter().enumerate() {
        let expected = 2.0 * v * (v * v).exp();
        assert!(
            (gx[i] - expected).abs() < 1e-4,
            "exp_sqr grad[{i}]: expected={expected}, got={}",
            gx[i]
        );
    }
}

#[test]
fn test_nested_log_of_sigmoid() {
    // f(x) = sum(log(sigmoid(x)))
    let x_data = vec![0.5_f32, 1.0, -1.0, 2.0];
    let x = var_from(x_data.clone(), &[4]);
    let tx = tracked(&x);
    let y = tx.sigmoid().unwrap().log().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    // d/dx log(sigmoid(x)) = 1 - sigmoid(x)
    for (i, &v) in x_data.iter().enumerate() {
        let s = 1.0 / (1.0 + (-v).exp());
        let expected = 1.0 - s;
        assert!(
            (gx[i] - expected).abs() < 1e-4,
            "log_sigmoid grad[{i}]: expected={expected}, got={}",
            gx[i]
        );
    }
}

#[test]
fn test_nested_tanh_of_mul() {
    // f(a,b) = sum(tanh(a*b)), verify with FD
    let a_data = vec![0.5_f32, -0.3, 1.0];
    let b_data = vec![1.0_f32, 2.0, -0.5];
    let a = var_from(a_data.clone(), &[3]);
    let b = var_from(b_data.clone(), &[3]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.mul(&tb).unwrap().tanh().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let ga = grad_vec(&grads, &a);
    let num = numerical_grad(&a_data, &[3], 1e-4, |t| {
        let bt = DynTensor::from_vec(b_data.clone(), &[3], &cpu()).unwrap();
        let r = t.mul(&bt).unwrap().tanh().unwrap();
        tensor_sum_f64(&r)
    });
    for i in 0..3 {
        assert!(
            (f64::from(ga[i]) - num[i]).abs() < 1e-2,
            "tanh_mul_fd d/da[{i}]: analytical={}, numerical={}",
            ga[i],
            num[i]
        );
    }
}

#[test]
fn test_nested_triple_composition() {
    // f(x) = sum(sigmoid(tanh(exp(x))))
    let x_data = vec![-0.5_f32, 0.0, 0.3];
    let x = var_from(x_data.clone(), &[3]);
    let tx = tracked(&x);
    let y = tx.exp().unwrap().tanh().unwrap().sigmoid().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, &[3], 1e-4, |t| {
        let r = t.exp().unwrap().tanh().unwrap().sigmoid().unwrap();
        tensor_sum_f64(&r)
    });
    for i in 0..3 {
        assert!(
            (f64::from(gx[i]) - num[i]).abs() < 1e-2,
            "triple_compose_fd[{i}]: analytical={}, numerical={}",
            gx[i],
            num[i]
        );
    }
}

#[test]
fn test_nested_add_mul_sub_div_chain() {
    // f(a,b) = sum(((a + b) * a - b) / (a + 1))
    let a_data = vec![2.0_f32, 3.0];
    let b_data = vec![1.0_f32, 0.5];
    let a = var_from(a_data.clone(), &[2]);
    let b = var_from(b_data.clone(), &[2]);
    let ta = tracked(&a);
    let tb = tracked(&b);

    let sum_ab = ta.add(&tb).unwrap();
    let prod = sum_ab.mul(&ta).unwrap();
    let minus_b = prod.sub(&tb).unwrap();
    let a_plus_1 = ta.add_scalar(1.0).unwrap();
    let y = minus_b.div(&a_plus_1).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let ga = grad_vec(&grads, &a);
    let num_a = numerical_grad(&a_data, &[2], 1e-3, |t| {
        let bt = DynTensor::from_vec(b_data.clone(), &[2], &cpu()).unwrap();
        let s = t.add(&bt).unwrap();
        let p = s.mul(&t).unwrap();
        let m = p.sub(&bt).unwrap();
        let ap1 = t.add_scalar(1.0).unwrap();
        let r = m.div(&ap1).unwrap();
        tensor_sum_f64(&r)
    });
    for i in 0..2 {
        assert!(
            (f64::from(ga[i]) - num_a[i]).abs() < 1e-2,
            "add_mul_sub_div d/da[{i}]: analytical={}, numerical={}",
            ga[i],
            num_a[i]
        );
    }
}

// ===========================================================================
// 7. Gradient accumulation for shared variables
// ===========================================================================

#[test]
fn test_grad_accum_var_used_twice_add() {
    // f(x) = sum(x + x) = sum(2x), df/dx = 2
    let x = var_from(vec![1.0, 2.0, 3.0], &[3]);
    let tx = tracked(&x);
    let y = tx.add(&tx).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    assert_close(&grad_vec(&grads, &x), &[2.0, 2.0, 2.0], 1e-6, "x+x grad");
}

#[test]
fn test_grad_accum_var_used_twice_mul() {
    // f(x) = sum(x * x) = sum(x^2), df/dx = 2x
    let vals = vec![2.0_f32, 3.0, -1.0];
    let x = var_from(vals.clone(), &[3]);
    let tx = tracked(&x);
    let y = tx.mul(&tx).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let expected: Vec<f32> = vals.iter().map(|v| 2.0 * v).collect();
    assert_close(&grad_vec(&grads, &x), &expected, 1e-5, "x*x grad");
}

#[test]
fn test_grad_accum_multiple_paths() {
    // f(x) = sum(x^2 + x) => df/dx = 2x + 1
    let vals = vec![1.0_f32, -2.0, 3.0];
    let x = var_from(vals.clone(), &[3]);
    let tx = tracked(&x);
    let y = tx.sqr().unwrap().add(&tx).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let expected: Vec<f32> = vals.iter().map(|v| 2.0 * v + 1.0).collect();
    assert_close(&grad_vec(&grads, &x), &expected, 1e-5, "x^2+x grad");
}

#[test]
fn test_grad_accum_three_paths() {
    // f(x) = sum(x + x^2 + x^3) via x + x*x + x*x*x
    // df/dx = 1 + 2x + 3x^2
    let vals = vec![1.0_f32, 2.0];
    let x = var_from(vals.clone(), &[2]);
    let tx = tracked(&x);
    let x_sq = tx.mul(&tx).unwrap();
    let x_cu = x_sq.mul(&tx).unwrap();
    let s = tx.add(&x_sq).unwrap().add(&x_cu).unwrap();
    let loss = scalar_loss(&s);
    let grads = backward(&loss).unwrap();
    let expected: Vec<f32> = vals.iter().map(|v| 1.0 + 2.0 * v + 3.0 * v * v).collect();
    assert_close(&grad_vec(&grads, &x), &expected, 1e-4, "x+x^2+x^3 grad");
}

#[test]
fn test_grad_accum_shared_in_different_branches() {
    // f(a,b) = sum(a*b + a^2), df/da = b + 2a, df/db = a
    let a = var_from(vec![2.0, 3.0], &[2]);
    let b = var_from(vec![5.0, 7.0], &[2]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let branch1 = ta.mul(&tb).unwrap();
    let branch2 = ta.sqr().unwrap();
    let y = branch1.add(&branch2).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    // df/da = b + 2a = [5+4, 7+6] = [9, 13]
    assert_close(&grad_vec(&grads, &a), &[9.0, 13.0], 1e-5, "d/da");
    assert_close(&grad_vec(&grads, &b), &[2.0, 3.0], 1e-5, "d/db");
}

// ===========================================================================
// 8. No-grad context (detach)
// ===========================================================================

#[test]
fn test_detach_stops_gradient_to_var() {
    // y = detach(x^2), loss = sum(y * w)
    // w gets gradients but x does not
    let x = var_from(vec![2.0, 3.0], &[2]);
    let w = var_from(vec![1.0, 1.0], &[2]);
    let tx = tracked(&x);
    let tw = tracked(&w);
    let x_sq = tx.sqr().unwrap();
    let detached = x_sq.detach();
    let y = detached.mul(&tw).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    // x should not receive gradients (detach breaks the graph)
    assert!(grads.get(&x).is_none(), "detach should block gradient to x");
    // w should receive detached x^2 values as gradient: [4, 9]
    assert_close(
        &grad_vec(&grads, &w),
        &[4.0, 9.0],
        1e-5,
        "d/dw through detach",
    );
}

#[test]
fn test_detach_preserves_tensor_values() {
    let x = var_from(vec![2.0, 3.0, 4.0], &[3]);
    let tx = tracked(&x);
    let y = tx.sqr().unwrap();
    let detached = y.detach();
    let original = y.tensor().to_flat_vec::<f32>().unwrap();
    let det_vals = detached.tensor().to_flat_vec::<f32>().unwrap();
    assert_eq!(original, det_vals, "detach should preserve values");
}

#[test]
fn test_detach_has_no_op() {
    let x = var_from(vec![1.0], &[1]);
    let tx = tracked(&x);
    let y = tx.sqr().unwrap();
    let detached = y.detach();
    assert!(detached.op().is_none(), "detached tensor should have no op");
    assert!(!detached.is_var(), "detached tensor should not be a var");
}

#[test]
fn test_from_tensor_constant_no_gradient() {
    // Constants created with from_tensor should not receive gradients
    let x = var_from(vec![2.0, 3.0], &[2]);
    let c = const_tensor(vec![10.0, 20.0], &[2]);
    let tx = tracked(&x);
    let y = tx.mul(&c).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    // x should get gradient = c = [10, 20]
    assert_close(
        &grad_vec(&grads, &x),
        &[10.0, 20.0],
        1e-5,
        "d/dx with const",
    );
}

// ===========================================================================
// 9. TrackedTensor creation and operations
// ===========================================================================

#[test]
fn test_tracked_tensor_from_var_basics() {
    let v = var_from(vec![1.0, 2.0, 3.0], &[3]);
    let t = TrackedTensor::from_var(&v).unwrap();
    assert!(t.is_var());
    assert_eq!(t.var_id(), Some(v.id()));
    assert!(t.op().is_none());
    assert_eq!(t.dims(), &[3]);
    assert_eq!(t.numel(), 3);
}

#[test]
fn test_tracked_tensor_from_tensor_basics() {
    let data = DynTensor::from_vec(vec![4.0, 5.0], &[2], &cpu()).unwrap();
    let t = TrackedTensor::from_tensor(data);
    assert!(!t.is_var());
    assert_eq!(t.var_id(), None);
    assert!(t.op().is_none());
    assert_eq!(t.dims(), &[2]);
}

#[test]
fn test_tracked_tensor_op_recording_add() {
    let a = var_from(vec![1.0], &[1]);
    let b = var_from(vec![2.0], &[1]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.add(&tb).unwrap();
    assert!(y.op().is_some());
    let op_str = format!("{:?}", y.op().unwrap());
    assert_eq!(op_str, "Add");
}

#[test]
fn test_tracked_tensor_into_tensor() {
    let v = var_from(vec![1.0, 2.0], &[2]);
    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let sq = t.sqr().unwrap();
    // Unwrap the TrackedTensor to get the underlying DynTensor
    let inner = Arc::try_unwrap(sq).unwrap().into_tensor().unwrap();
    assert_eq!(inner.dims(), &[2]);
    let vals = inner.to_flat_vec::<f32>().unwrap();
    assert_close(&vals, &[1.0, 4.0], 1e-5, "into_tensor values");
}

#[test]
fn test_tracked_tensor_unique_node_ids() {
    let a = var_from(vec![1.0], &[1]);
    let b = var_from(vec![2.0], &[1]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    assert_ne!(ta.node_id(), tb.node_id());
    let y = ta.add(&tb).unwrap();
    assert_ne!(y.node_id(), ta.node_id());
    assert_ne!(y.node_id(), tb.node_id());
}

// ===========================================================================
// 10. Backward pass execution
// ===========================================================================

#[test]
fn test_backward_requires_scalar_loss() {
    // Non-scalar tensors should fail backward
    let x = var_from(vec![1.0, 2.0, 3.0], &[3]);
    let tx = tracked(&x);
    let result = backward(&tx);
    assert!(result.is_err(), "backward should reject non-scalar loss");
}

#[test]
fn test_backward_ones_initialization() {
    // For scalar x, backward should set dL/dx = 1.0
    let x = var_from(vec![5.0], &[1]);
    let tx = tracked(&x);
    // identity: loss = x (scalar, numel=1)
    let grads = backward(&tx).unwrap();
    assert_close(&grad_vec(&grads, &x), &[1.0], 1e-6, "identity backward");
}

#[test]
fn test_backward_leaf_var_receives_gradient() {
    let x = var_from(vec![3.0], &[1]);
    let tx = tracked(&x);
    let y = tx.sqr().unwrap(); // y = x^2 = 9
    let grads = backward(&y).unwrap();
    // dy/dx = 2x = 6
    assert_close(&grad_vec(&grads, &x), &[6.0], 1e-5, "sqr backward");
}

#[test]
fn test_backward_non_finite_loss_rejected() {
    // NaN loss should be rejected
    let x = var_from(vec![f32::NAN], &[1]);
    let tx = tracked(&x);
    let result = backward(&tx);
    assert!(result.is_err(), "NaN loss should be rejected");
}

#[test]
fn test_backward_inf_loss_rejected() {
    let x = var_from(vec![f32::INFINITY], &[1]);
    let tx = tracked(&x);
    let result = backward(&tx);
    assert!(result.is_err(), "Inf loss should be rejected");
}

// ===========================================================================
// 11. Gradient tape management
// ===========================================================================

#[test]
fn test_grad_store_var_count() {
    let a = var_from(vec![1.0, 2.0], &[2]);
    let b = var_from(vec![3.0, 4.0], &[2]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.mul(&tb).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    assert_eq!(grads.var_count(), 2, "should have gradients for a and b");
}

#[test]
fn test_grad_store_iteration() {
    let a = var_from(vec![1.0], &[1]);
    let b = var_from(vec![2.0], &[1]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.add(&tb).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let mut count = 0;
    for (_var_id, grad) in grads.var_grads() {
        assert_eq!(grad.dims(), &[1]);
        count += 1;
    }
    assert_eq!(count, 2);
}

#[test]
fn test_backward_for_vars_filtering() {
    let a = var_from(vec![1.0, 2.0], &[2]);
    let b = var_from(vec![3.0, 4.0], &[2]);
    let c = var_from(vec![5.0, 6.0], &[2]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let tc = tracked(&c);
    let y = ta.mul(&tb).unwrap().add(&tc).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward_for_vars(&loss, &[&a]).unwrap();
    assert!(grads.get(&a).is_some(), "should have gradient for a");
    assert!(grads.get(&b).is_none(), "should NOT have gradient for b");
    assert!(grads.get(&c).is_none(), "should NOT have gradient for c");
}

#[test]
fn test_independent_backward_calls() {
    // Two separate forward-backward passes should be independent
    let x = var_from(vec![2.0], &[1]);
    let tx1 = tracked(&x);
    let y1 = tx1.sqr().unwrap();
    let grads1 = backward(&y1).unwrap();

    let tx2 = tracked(&x);
    let y2 = tx2.mul_scalar(3.0).unwrap();
    let grads2 = backward(&y2).unwrap();

    // d/dx x^2 at x=2 = 4
    assert_close(&grad_vec(&grads1, &x), &[4.0], 1e-5, "first backward");
    // d/dx 3x = 3
    assert_close(&grad_vec(&grads2, &x), &[3.0], 1e-5, "second backward");
}

#[test]
fn test_grad_store_retain_only() {
    let a = var_from(vec![1.0], &[1]);
    let b = var_from(vec![2.0], &[1]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.mul(&tb).unwrap();
    let loss = scalar_loss(&y);
    let mut grads = backward(&loss).unwrap();
    assert_eq!(grads.var_count(), 2);
    grads.retain_only(&[&a]);
    assert_eq!(grads.var_count(), 1);
    assert!(grads.get(&a).is_some());
    assert!(grads.get(&b).is_none());
}

// ===========================================================================
// 12. Loss computation backward
// ===========================================================================

#[test]
fn test_mse_loss_backward_values() {
    // MSE = mean((pred - target)^2)
    // d/d(pred) = 2*(pred - target) / N
    let pred = var_from(vec![1.0, 2.0, 3.0, 4.0], &[4]);
    let target_data = vec![1.5, 2.5, 2.5, 3.5];
    let target = const_tensor(target_data.clone(), &[4]);
    let tp = tracked(&pred);
    let loss = tp.mse_loss(&target).unwrap();
    let grads = backward(&loss).unwrap();

    let gp = grad_vec(&grads, &pred);
    let n = 4.0_f32;
    let pred_data = [1.0_f32, 2.0, 3.0, 4.0];
    for i in 0..4 {
        let expected = 2.0 * (pred_data[i] - target_data[i]) / n;
        assert!(
            (gp[i] - expected).abs() < 1e-5,
            "mse grad[{i}]: expected={expected}, got={}",
            gp[i]
        );
    }
}

#[test]
fn test_mse_loss_zero_when_equal() {
    let pred = var_from(vec![1.0, 2.0, 3.0], &[3]);
    let target = const_tensor(vec![1.0, 2.0, 3.0], &[3]);
    let tp = tracked(&pred);
    let loss = tp.mse_loss(&target).unwrap();
    let loss_val = loss.tensor().to_flat_vec::<f32>().unwrap();
    assert!(
        (loss_val[0]).abs() < 1e-7,
        "MSE of identical tensors should be 0"
    );
}

#[test]
fn test_l1_loss_backward_values() {
    // L1 = mean(|pred - target|)
    // d/d(pred) = sign(pred - target) / N
    let pred = var_from(vec![1.0, 3.0, 2.0, 5.0], &[4]);
    let target = const_tensor(vec![2.0, 1.0, 2.0, 3.0], &[4]);
    let tp = tracked(&pred);
    let loss = tp.l1_loss(&target).unwrap();
    let grads = backward(&loss).unwrap();

    let gp = grad_vec(&grads, &pred);
    // pred - target = [-1, 2, 0, 2], sign = [-1, 1, 0, 1], /4 = [-0.25, 0.25, 0, 0.25]
    assert_close(&gp, &[-0.25, 0.25, 0.0, 0.25], 1e-5, "l1 grad");
}

#[test]
fn test_huber_loss_backward_quadratic_region() {
    // For |diff| < delta, Huber is quadratic: loss = 0.5 * diff^2 / delta
    // d/d(pred) = diff / delta / N
    let pred = var_from(vec![1.1, 1.9], &[2]);
    let target = const_tensor(vec![1.0, 2.0], &[2]);
    let tp = tracked(&pred);
    let delta = 1.0;
    let loss = tp.huber_loss(&target, delta).unwrap();
    let grads = backward(&loss).unwrap();

    let gp = grad_vec(&grads, &pred);
    // diff = [0.1, -0.1], |diff| < 1.0, quadratic region
    // d = diff / delta / N = [0.1/1/2, -0.1/1/2] = [0.05, -0.05]
    assert_close(&gp, &[0.05, -0.05], 1e-4, "huber quadratic grad");
}

// ===========================================================================
// 13. Multi-output gradient computation
// ===========================================================================

#[test]
fn test_multi_var_simultaneous_gradients() {
    // f(a,b,c) = sum(a*b + b*c + c*a)
    // df/da = b + c, df/db = a + c, df/dc = b + a
    let a = var_from(vec![2.0, 3.0], &[2]);
    let b = var_from(vec![4.0, 5.0], &[2]);
    let c = var_from(vec![6.0, 7.0], &[2]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let tc = tracked(&c);
    let ab = ta.mul(&tb).unwrap();
    let bc = tb.mul(&tc).unwrap();
    let ca = tc.mul(&ta).unwrap();
    let y = ab.add(&bc).unwrap().add(&ca).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    // df/da = b + c = [10, 12]
    assert_close(&grad_vec(&grads, &a), &[10.0, 12.0], 1e-5, "d/da");
    // df/db = a + c = [8, 10]
    assert_close(&grad_vec(&grads, &b), &[8.0, 10.0], 1e-5, "d/db");
    // df/dc = b + a = [6, 8]
    assert_close(&grad_vec(&grads, &c), &[6.0, 8.0], 1e-5, "d/dc");
}

#[test]
fn test_backward_for_vars_selective_two() {
    let a = var_from(vec![1.0], &[1]);
    let b = var_from(vec![2.0], &[1]);
    let c = var_from(vec![3.0], &[1]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let tc = tracked(&c);
    let y = ta.mul(&tb).unwrap().add(&tc).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward_for_vars(&loss, &[&a, &c]).unwrap();
    assert!(grads.get(&a).is_some());
    assert!(grads.get(&b).is_none());
    assert!(grads.get(&c).is_some());
    assert_eq!(grads.var_count(), 2);
}

#[test]
fn test_gradient_shapes_match_vars() {
    // Verify all gradient shapes match their variable shapes
    let a = var_from(vec![1.0; 6], &[2, 3]);
    let b = var_from(vec![1.0; 12], &[3, 4]);
    let ta = tracked(&a);
    let tb = tracked(&b);
    let y = ta.matmul(&tb).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let ga = grads.get(&a).unwrap();
    let gb = grads.get(&b).unwrap();
    assert_eq!(ga.dims(), &[2, 3], "grad_a shape must match a shape");
    assert_eq!(gb.dims(), &[3, 4], "grad_b shape must match b shape");
}

// ===========================================================================
// 14. Additional chain rule edge cases
// ===========================================================================

#[test]
fn test_chain_rule_mul_scalar_then_add() {
    // f(x) = sum(2*x + 3), df/dx = 2
    let x = var_from(vec![1.0, 2.0, 3.0], &[3]);
    let tx = tracked(&x);
    let y = tx.mul_scalar(2.0).unwrap().add_scalar(3.0).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    assert_close(&grad_vec(&grads, &x), &[2.0, 2.0, 2.0], 1e-6, "2x+3 grad");
}

#[test]
fn test_chain_rule_neg_then_exp() {
    // f(x) = sum(exp(-x)), df/dx = -exp(-x)
    let vals = vec![0.0_f32, 1.0, -1.0];
    let x = var_from(vals.clone(), &[3]);
    let tx = tracked(&x);
    let y = tx.neg().unwrap().exp().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    for (i, &v) in vals.iter().enumerate() {
        let expected = -(-v).exp();
        assert!(
            (gx[i] - expected).abs() < 1e-5,
            "neg_exp grad[{i}]: expected={expected}, got={}",
            gx[i]
        );
    }
}

#[test]
fn test_chain_rule_reshape_preserves_gradient_flow() {
    // reshape should not affect gradient values, only reshape back
    let x = var_from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let tx = tracked(&x);
    let flat = tx.reshape(&[6]).unwrap();
    let y = flat.sqr().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x).unwrap();
    assert_eq!(gx.dims(), &[2, 3], "gradient should have original shape");
    let g = gx.to_flat_vec::<f32>().unwrap();
    // d/dx x^2 = 2x
    assert_close(
        &g,
        &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        1e-5,
        "reshape grad values",
    );
}

#[test]
fn test_chain_rule_transpose_then_matmul() {
    // f(A) = sum(A^T @ A) -- Gram matrix sum
    let a_data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = var_from(a_data.clone(), &[3, 2]);
    let ta = tracked(&a);
    let at = ta.transpose(0, 1).unwrap();
    let gram = at.matmul(&ta).unwrap();
    let loss = scalar_loss(&gram);
    let grads = backward(&loss).unwrap();

    let ga = grad_vec(&grads, &a);
    let num = numerical_grad(&a_data, &[3, 2], 1e-3, |t| {
        let tt = t.transpose(0, 1).unwrap();
        let g = tt.matmul(&t).unwrap();
        tensor_sum_f64(&g)
    });
    for i in 0..6 {
        assert!(
            (f64::from(ga[i]) - num[i]).abs() < 1e-2,
            "gram_fd d/da[{i}]: analytical={}, numerical={}",
            ga[i],
            num[i]
        );
    }
}

#[test]
fn test_chain_rule_sum_then_sqr_vs_sqr_then_sum() {
    // These are different functions:
    // f1(x) = (sum(x))^2, f2(x) = sum(x^2)
    let vals = vec![1.0_f32, 2.0, 3.0];
    let x1 = var_from(vals.clone(), &[3]);
    let tx1 = tracked(&x1);
    let s1 = tx1.sum_keepdim(0).unwrap().sqr().unwrap();
    let grads1 = backward(&s1).unwrap();
    // d/dx (sum(x))^2 = 2*sum(x) = 2*6 = 12
    assert_close(
        &grad_vec(&grads1, &x1),
        &[12.0, 12.0, 12.0],
        1e-5,
        "sum_then_sqr",
    );

    let x2 = var_from(vals, &[3]);
    let tx2 = tracked(&x2);
    let s2 = tx2.sqr().unwrap().sum_keepdim(0).unwrap();
    let grads2 = backward(&s2).unwrap();
    // d/dx sum(x^2) = 2x = [2, 4, 6]
    assert_close(
        &grad_vec(&grads2, &x2),
        &[2.0, 4.0, 6.0],
        1e-5,
        "sqr_then_sum",
    );
}

#[test]
fn test_chain_rule_mean_keepdim_then_sqr() {
    // f(x) = mean(x)^2, df/dx = 2*mean(x) / N
    let x = var_from(vec![2.0, 4.0, 6.0, 8.0], &[4]);
    let tx = tracked(&x);
    let m = tx.mean_keepdim(0).unwrap();
    let y = m.sqr().unwrap();
    let grads = backward(&y).unwrap();

    // mean = 5, d/dx mean(x)^2 = 2*mean / N = 10/4 = 2.5
    assert_close(
        &grad_vec(&grads, &x),
        &[2.5, 2.5, 2.5, 2.5],
        1e-5,
        "mean_sqr grad",
    );
}

#[test]
fn test_chain_rule_softmax_then_log() {
    // log(softmax(x)) = log_softmax(x), compare gradients
    let x_data = vec![1.0_f32, 2.0, 3.0];
    let x1 = var_from(x_data.clone(), &[1, 3]);
    let tx1 = tracked(&x1);
    let y1 = tx1.softmax(1).unwrap().log().unwrap();
    let loss1 = scalar_loss(&y1);
    let grads1 = backward(&loss1).unwrap();

    let x2 = var_from(x_data, &[1, 3]);
    let tx2 = tracked(&x2);
    let y2 = tx2.log_softmax(1).unwrap();
    let loss2 = scalar_loss(&y2);
    let grads2 = backward(&loss2).unwrap();

    // Both should produce very similar gradients
    let g1 = grad_vec(&grads1, &x1);
    let g2 = grad_vec(&grads2, &x2);
    for i in 0..3 {
        assert!(
            (g1[i] - g2[i]).abs() < 1e-4,
            "softmax_log vs log_softmax grad[{i}]: {} vs {}",
            g1[i],
            g2[i]
        );
    }
}

#[test]
fn test_chain_rule_powf_then_sum() {
    // f(x) = sum(x^3), df/dx = 3x^2
    let vals = vec![1.0_f32, 2.0, 3.0];
    let x = var_from(vals.clone(), &[3]);
    let tx = tracked(&x);
    let y = tx.powf(3.0).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();
    let expected: Vec<f32> = vals.iter().map(|v| 3.0 * v * v).collect();
    assert_close(&grad_vec(&grads, &x), &expected, 1e-4, "powf3 grad");
}

#[test]
fn test_chain_rule_sin_cos_product_rule() {
    // f(x) = sum(sin(x) * cos(x)) = 0.5 * sum(sin(2x))
    // df/dx = cos(x)^2 - sin(x)^2 = cos(2x)
    let vals = vec![0.3_f32, 0.7, -0.5, 1.2];
    let x = var_from(vals.clone(), &[4]);
    let tx = tracked(&x);
    let y = tx.sin().unwrap().mul(&tx.cos().unwrap()).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    for (i, &v) in vals.iter().enumerate() {
        let expected = (2.0 * v).cos();
        assert!(
            (gx[i] - expected).abs() < 1e-4,
            "sin_cos grad[{i}]: expected={expected}, got={}",
            gx[i]
        );
    }
}

#[test]
fn test_var_zeros_creates_zero_initialized() {
    let v = Var::zeros(&[3, 4], DType::F32, &cpu()).unwrap();
    let data = v.data().unwrap();
    let vals = data.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|&x| x == 0.0));
    assert_eq!(data.dims(), &[3, 4]);
}

#[test]
fn test_var_set_updates_data() {
    let v = Var::zeros(&[2], DType::F32, &cpu()).unwrap();
    let new_data = DynTensor::from_vec(vec![3.0, 4.0], &[2], &cpu()).unwrap();
    v.set(&new_data).unwrap();
    let data = v.data().unwrap();
    let vals = data.to_flat_vec::<f32>().unwrap();
    assert_close(&vals, &[3.0, 4.0], 1e-6, "var set");
}

#[test]
fn test_var_set_rejects_shape_mismatch() {
    let v = Var::zeros(&[2], DType::F32, &cpu()).unwrap();
    let wrong = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let result = v.set(&wrong);
    assert!(result.is_err(), "set with wrong shape should fail");
}
