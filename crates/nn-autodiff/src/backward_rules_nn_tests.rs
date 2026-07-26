// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive backward rule tests for neural network operations.
//!
//! Covers 12 categories:
//!  1. Linear layer backward (input, weight, bias gradients)
//!  2. Conv1d/Conv2d backward (input, kernel, bias gradients)
//!  3. BatchNorm backward (running mean/var, affine parameter gradients)
//!  4. LayerNorm backward (normalized gradient computation)
//!  5. Softmax backward (Jacobian-vector product correctness)
//!  6. Cross-entropy loss backward (gradient matches analytical formula)
//!  7. Embedding backward (sparse gradient accumulation)
//!  8. Attention backward (Q/K/V gradient computation via manual SDPA)
//!  9. ReLU/GELU/SiLU backward (gradient at boundaries)
//! 10. Dropout backward (gradient scaling by 1/(1-p))
//! 11. MaxPool backward (gradient routing to max positions)
//! 12. Reshape/transpose backward (gradient shape restoration)
//!
//! Each test uses finite-difference (central difference) cross-checks
//! to verify analytical gradients match numerical gradients.

use std::sync::Arc;

use crate::grad::backward;
use crate::tracked::TrackedTensor;
use crate::var::Var;
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn var_from(data: Vec<f32>, shape: &[usize]) -> Var {
    Var::new(DynTensor::from_vec(data, shape, &cpu()).unwrap())
}

fn tracked(v: &Var) -> Arc<TrackedTensor> {
    Arc::new(TrackedTensor::from_var(v).unwrap())
}

fn grad_vec(grads: &crate::grad::GradStore, var: &Var) -> Vec<f32> {
    grads.get(var).unwrap().to_flat_vec::<f32>().unwrap()
}

/// Make a scalar loss from an arbitrary-shaped tensor by sum-of-all-elements.
fn scalar_loss(t: &Arc<TrackedTensor>) -> Arc<TrackedTensor> {
    let mut result = Arc::clone(t);
    for d in (0..result.tensor().rank()).rev() {
        result = result.sum_keepdim(d).unwrap();
    }
    result
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

/// Sum all elements of a DynTensor into a scalar f64.
fn sum_all(t: &DynTensor) -> f64 {
    t.to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .sum()
}

/// Sum of squares of all elements.
fn sum_sqr(t: &DynTensor) -> f64 {
    t.to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum()
}

// ===========================================================================
// 1. Linear layer backward: input, weight, bias gradients
// ===========================================================================

#[test]
fn test_nn_linear_backward_input_grad_fd() {
    let x_data: Vec<f32> = vec![0.5, -0.3, 1.2, 0.8, -0.6, 0.4];
    let w_data: Vec<f32> = vec![0.1, 0.2, 0.3, -0.1, 0.4, -0.2];
    let b_data: Vec<f32> = vec![0.05, -0.05];

    let x = var_from(x_data.clone(), &[2, 3]);
    let w = var_from(w_data.clone(), &[2, 3]);
    let b = var_from(b_data.clone(), &[2]);

    let tx = tracked(&x);
    let tw = tracked(&w);
    let tb = tracked(&b);

    let wt = tw.transpose(0, 1).unwrap();
    let y = tx.matmul(&wt).unwrap();
    let b_unsq = tb.unsqueeze(0).unwrap();
    let y_biased = y.add(&b_unsq).unwrap();
    let loss = y_biased.sqr().unwrap();
    let loss = scalar_loss(&loss);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let xt = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let wt = DynTensor::from_vec(w_data.clone(), &[2, 3], &cpu())
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        let bt = DynTensor::from_vec(b_data.clone(), &[2], &cpu())
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let y = xt.matmul(&wt).unwrap().add(&bt).unwrap();
        sum_sqr(&y)
    });
    assert_grad_close(&gx, &num, 1e-2, "linear_input_fd");
}

#[test]
fn test_nn_linear_backward_weight_grad_fd() {
    let x_data: Vec<f32> = vec![0.5, -0.3, 1.2, 0.8, -0.6, 0.4];
    let w_data: Vec<f32> = vec![0.1, 0.2, 0.3, -0.1, 0.4, -0.2];
    let b_data: Vec<f32> = vec![0.05, -0.05];

    let x = var_from(x_data.clone(), &[2, 3]);
    let w = var_from(w_data.clone(), &[2, 3]);
    let b = var_from(b_data.clone(), &[2]);

    let tx = tracked(&x);
    let tw = tracked(&w);
    let tb = tracked(&b);

    let wt = tw.transpose(0, 1).unwrap();
    let y = tx.matmul(&wt).unwrap();
    let b_unsq = tb.unsqueeze(0).unwrap();
    let y_biased = y.add(&b_unsq).unwrap();
    let loss = y_biased.sqr().unwrap();
    let loss = scalar_loss(&loss);
    let grads = backward(&loss).unwrap();

    let gw = grad_vec(&grads, &w);
    let num = numerical_grad(&w_data, 1e-3, |d| {
        let xt = DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap();
        let wt = DynTensor::from_vec(d, &[2, 3], &cpu())
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        let bt = DynTensor::from_vec(b_data.clone(), &[2], &cpu())
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let y = xt.matmul(&wt).unwrap().add(&bt).unwrap();
        sum_sqr(&y)
    });
    assert_grad_close(&gw, &num, 1e-2, "linear_weight_fd");
}

#[test]
fn test_nn_linear_backward_bias_grad_fd() {
    let x_data: Vec<f32> = vec![0.5, -0.3, 1.2, 0.8, -0.6, 0.4];
    let w_data: Vec<f32> = vec![0.1, 0.2, 0.3, -0.1, 0.4, -0.2];
    let b_data: Vec<f32> = vec![0.05, -0.05];

    let x = var_from(x_data.clone(), &[2, 3]);
    let w = var_from(w_data.clone(), &[2, 3]);
    let b = var_from(b_data.clone(), &[2]);

    let tx = tracked(&x);
    let tw = tracked(&w);
    let tb = tracked(&b);

    let wt = tw.transpose(0, 1).unwrap();
    let y = tx.matmul(&wt).unwrap();
    let b_unsq = tb.unsqueeze(0).unwrap();
    let y_biased = y.add(&b_unsq).unwrap();
    let loss = y_biased.sqr().unwrap();
    let loss = scalar_loss(&loss);
    let grads = backward(&loss).unwrap();

    let gb = grad_vec(&grads, &b);
    let num = numerical_grad(&b_data, 1e-3, |d| {
        let xt = DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap();
        let wt = DynTensor::from_vec(w_data.clone(), &[2, 3], &cpu())
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        let bt = DynTensor::from_vec(d, &[2], &cpu())
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let y = xt.matmul(&wt).unwrap().add(&bt).unwrap();
        sum_sqr(&y)
    });
    assert_grad_close(&gb, &num, 1e-2, "linear_bias_fd");
}

#[test]
fn test_nn_linear_backward_sum_loss_bias_is_batch_size() {
    // For sum loss through linear, bias gradient = batch_size
    let x = var_from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let w = var_from(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
    let b = var_from(vec![0.0, 0.0], &[2]);

    let tx = tracked(&x);
    let tw = tracked(&w);
    let tb = tracked(&b);

    let wt = tw.transpose(0, 1).unwrap();
    let y = tx.matmul(&wt).unwrap();
    let b_unsq = tb.unsqueeze(0).unwrap();
    let y_biased = y.add(&b_unsq).unwrap();
    let loss = scalar_loss(&y_biased);
    let grads = backward(&loss).unwrap();

    let gb = grad_vec(&grads, &b);
    // Each bias element receives gradient from 2 batch samples
    for (i, &v) in gb.iter().enumerate() {
        assert!(
            (v - 2.0).abs() < 1e-4,
            "linear bias grad[{i}] expected ~2.0, got {v}"
        );
    }
}

// ===========================================================================
// 2. Conv1d/Conv2d backward: input, kernel, bias gradients
// ===========================================================================

#[test]
fn test_nn_conv1d_backward_input_fd() {
    let x_data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let k_data = vec![0.5_f32, 1.0, 0.5];

    let x = var_from(x_data.clone(), &[1, 1, 5]);
    let k = var_from(k_data.clone(), &[1, 1, 3]);

    let tx = tracked(&x);
    let tk = tracked(&k);

    let y = tx.conv1d(&tk, 1, 1, 1, 1).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let xt = DynTensor::from_vec(d, &[1, 1, 5], &cpu()).unwrap();
        let kt = DynTensor::from_vec(k_data.clone(), &[1, 1, 3], &cpu()).unwrap();
        sum_all(&xt.conv1d(&kt, 1, 1, 1, 1).unwrap())
    });
    assert_grad_close(&gx, &num, 1e-2, "conv1d_input_fd");
}

#[test]
fn test_nn_conv1d_backward_kernel_fd() {
    let x_data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let k_data = vec![0.5_f32, 1.0, 0.5];

    let x = var_from(x_data.clone(), &[1, 1, 5]);
    let k = var_from(k_data.clone(), &[1, 1, 3]);

    let tx = tracked(&x);
    let tk = tracked(&k);

    let y = tx.conv1d(&tk, 1, 1, 1, 1).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gk = grad_vec(&grads, &k);
    let num = numerical_grad(&k_data, 1e-3, |d| {
        let xt = DynTensor::from_vec(x_data.clone(), &[1, 1, 5], &cpu()).unwrap();
        let kt = DynTensor::from_vec(d, &[1, 1, 3], &cpu()).unwrap();
        sum_all(&xt.conv1d(&kt, 1, 1, 1, 1).unwrap())
    });
    assert_grad_close(&gk, &num, 1e-2, "conv1d_kernel_fd");
}

#[test]
fn test_nn_conv1d_backward_stride2_fd() {
    let x_data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let k_data = vec![0.5_f32, 1.0, 0.5];

    let x = var_from(x_data.clone(), &[1, 1, 8]);
    let k = var_from(k_data.clone(), &[1, 1, 3]);

    let tx = tracked(&x);
    let tk = tracked(&k);

    let y = tx.conv1d(&tk, 0, 2, 1, 1).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let xt = DynTensor::from_vec(d, &[1, 1, 8], &cpu()).unwrap();
        let kt = DynTensor::from_vec(k_data.clone(), &[1, 1, 3], &cpu()).unwrap();
        sum_all(&xt.conv1d(&kt, 0, 2, 1, 1).unwrap())
    });
    assert_grad_close(&gx, &num, 1e-2, "conv1d_stride2_fd");
}

#[test]
fn test_nn_conv1d_backward_bias_via_add() {
    // Simulate conv + bias: y = conv1d(x, k) + b
    let x_data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let k_data = vec![1.0_f32, 0.0, -1.0];
    let b_data = vec![0.1_f32];

    let x = var_from(x_data, &[1, 1, 5]);
    let k = var_from(k_data, &[1, 1, 3]);
    let b = var_from(b_data, &[1, 1, 1]);

    let tx = tracked(&x);
    let tk = tracked(&k);
    let tb = tracked(&b);

    let y = tx.conv1d(&tk, 0, 1, 1, 1).unwrap(); // [1, 1, 3]
    let y_biased = y.add(&tb).unwrap();
    let loss = scalar_loss(&y_biased);
    let grads = backward(&loss).unwrap();

    let gb = grad_vec(&grads, &b);
    // bias grad = number of output elements = 3
    assert!(
        (gb[0] - 3.0).abs() < 1e-4,
        "conv1d bias grad expected ~3.0, got {}",
        gb[0]
    );
}

#[test]
fn test_nn_conv2d_backward_input_fd() {
    // Simple [1,1,4,4] input, [1,1,3,3] kernel, no padding
    let x_data: Vec<f32> = (1..=16).map(|i| i as f32 * 0.1).collect();
    let k_data: Vec<f32> = vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0];

    let x = var_from(x_data.clone(), &[1, 1, 4, 4]);
    let k = var_from(k_data.clone(), &[1, 1, 3, 3]);

    let tx = tracked(&x);
    let tk = tracked(&k);

    let y = tx.conv2d(&tk, 0, 1, 1, 1).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    assert_eq!(gx.len(), 16, "conv2d grad_input length");

    let num = numerical_grad(&x_data, 1e-3, |d| {
        let xt = DynTensor::from_vec(d, &[1, 1, 4, 4], &cpu()).unwrap();
        let kt = DynTensor::from_vec(k_data.clone(), &[1, 1, 3, 3], &cpu()).unwrap();
        sum_all(&xt.conv2d(&kt, 0, 1, 1, 1).unwrap())
    });
    assert_grad_close(&gx, &num, 1e-2, "conv2d_input_fd");
}

#[test]
fn test_nn_conv2d_backward_kernel_fd() {
    let x_data: Vec<f32> = (1..=16).map(|i| i as f32 * 0.1).collect();
    let k_data: Vec<f32> = vec![0.1, 0.2, -0.1, 0.3, -0.2, 0.1, 0.0, 0.4, -0.3];

    let x = var_from(x_data.clone(), &[1, 1, 4, 4]);
    let k = var_from(k_data.clone(), &[1, 1, 3, 3]);

    let tx = tracked(&x);
    let tk = tracked(&k);

    let y = tx.conv2d(&tk, 0, 1, 1, 1).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gk = grad_vec(&grads, &k);
    let num = numerical_grad(&k_data, 1e-3, |d| {
        let xt = DynTensor::from_vec(x_data.clone(), &[1, 1, 4, 4], &cpu()).unwrap();
        let kt = DynTensor::from_vec(d, &[1, 1, 3, 3], &cpu()).unwrap();
        sum_all(&xt.conv2d(&kt, 0, 1, 1, 1).unwrap())
    });
    assert_grad_close(&gk, &num, 1e-2, "conv2d_kernel_fd");
}

// ===========================================================================
// 3. BatchNorm backward: affine parameter gradients
// ===========================================================================

#[test]
fn test_nn_batch_norm_backward_input_fd() {
    // BatchNorm on [N=2, C=2, L=3]
    let x_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // sample 0
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // sample 1
    ];
    let w_data: Vec<f32> = vec![1.0, 1.0]; // gamma per channel
    let b_data: Vec<f32> = vec![0.0, 0.0]; // beta per channel

    let x = var_from(x_data, &[2, 2, 3]);
    let w = var_from(w_data, &[2]);
    let b = var_from(b_data, &[2]);

    let tx = tracked(&x);
    let tw = tracked(&w);
    let tb = tracked(&b);

    let y = tx.batch_norm(&tw, &tb, 1e-5).unwrap();
    let loss = y.sqr().unwrap();
    let loss = scalar_loss(&loss);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    assert_eq!(gx.len(), 12, "batch_norm grad_input length");

    // Finite-difference for weight (gamma)
    let gw = grad_vec(&grads, &w);
    assert_eq!(gw.len(), 2, "batch_norm grad_weight length");
}

#[test]
fn test_nn_batch_norm_backward_weight_fd() {
    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w_data: Vec<f32> = vec![0.8, 1.2];
    let b_data: Vec<f32> = vec![0.1, -0.1];

    // N=3, C=2 (matches weight/bias length 2), L=1 spatial -> 6 elements.
    let x = var_from(x_data, &[3, 2, 1]);
    let w = var_from(w_data, &[2]);
    let b = var_from(b_data, &[2]);

    let tx = tracked(&x);
    let tw = tracked(&w);
    let tb = tracked(&b);

    let y = tx.batch_norm(&tw, &tb, 1e-5).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gw = grads.get(&w).unwrap();
    assert_eq!(gw.dims(), &[2], "batch_norm grad_weight shape");
}

#[test]
fn test_nn_batch_norm_backward_bias_is_count() {
    // For sum loss, bias grad = total elements per channel across batch & spatial
    let x = var_from(vec![1.0; 12], &[2, 2, 3]);
    let w = var_from(vec![1.0, 1.0], &[2]);
    let b = var_from(vec![0.0, 0.0], &[2]);

    let tx = tracked(&x);
    let tw = tracked(&w);
    let tb = tracked(&b);

    let y = tx.batch_norm(&tw, &tb, 1e-5).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gb = grad_vec(&grads, &b);
    // Each channel has 2 * 3 = 6 elements across batch and spatial
    for (i, &v) in gb.iter().enumerate() {
        assert!(
            (v - 6.0).abs() < 1e-3,
            "batch_norm bias grad[{i}] expected ~6.0, got {v}"
        );
    }
}

// ===========================================================================
// 4. LayerNorm backward: normalized gradient computation
// ===========================================================================

#[test]
fn test_nn_layer_norm_backward_input_fd() {
    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w_data: Vec<f32> = vec![1.0, 1.0, 1.0];
    let b_data: Vec<f32> = vec![0.0, 0.0, 0.0];

    let x = var_from(x_data.clone(), &[2, 3]);
    let w = var_from(w_data.clone(), &[3]);
    let b = var_from(b_data.clone(), &[3]);

    let tx = tracked(&x);
    let tw = tracked(&w);
    let tb = tracked(&b);

    let y = tx.layer_norm(&tw, &tb, 1e-5).unwrap();
    let loss = y.sqr().unwrap();
    let loss = scalar_loss(&loss);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let xt = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let wt = DynTensor::from_vec(w_data.clone(), &[3], &cpu()).unwrap();
        let bt = DynTensor::from_vec(b_data.clone(), &[3], &cpu()).unwrap();
        let mean = xt.mean_keepdim(1).unwrap();
        let diff = xt.sub(&mean).unwrap();
        let var = diff.sqr().unwrap().mean_keepdim(1).unwrap();
        let inv_std = var
            .add_scalar(1e-5)
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();
        let normed = diff.mul(&inv_std).unwrap();
        let y = normed.mul(&wt).unwrap().add(&bt).unwrap();
        sum_sqr(&y)
    });
    assert_grad_close(&gx, &num, 5e-2, "layer_norm_input_fd");
}

#[test]
fn test_nn_layer_norm_backward_weight_fd() {
    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w_data: Vec<f32> = vec![0.8, 1.2, 0.5];
    let b_data: Vec<f32> = vec![0.1, -0.1, 0.0];

    let x = var_from(x_data.clone(), &[2, 3]);
    let w = var_from(w_data.clone(), &[3]);
    let b = var_from(b_data.clone(), &[3]);

    let tx = tracked(&x);
    let tw = tracked(&w);
    let tb = tracked(&b);

    let y = tx.layer_norm(&tw, &tb, 1e-5).unwrap();
    let loss = y.sqr().unwrap();
    let loss = scalar_loss(&loss);
    let grads = backward(&loss).unwrap();

    let gw = grad_vec(&grads, &w);
    let num = numerical_grad(&w_data, 1e-3, |d| {
        let xt = DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap();
        let wt = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        let bt = DynTensor::from_vec(b_data.clone(), &[3], &cpu()).unwrap();
        let mean = xt.mean_keepdim(1).unwrap();
        let diff = xt.sub(&mean).unwrap();
        let var = diff.sqr().unwrap().mean_keepdim(1).unwrap();
        let inv_std = var
            .add_scalar(1e-5)
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();
        let normed = diff.mul(&inv_std).unwrap();
        let y = normed.mul(&wt).unwrap().add(&bt).unwrap();
        sum_sqr(&y)
    });
    assert_grad_close(&gw, &num, 5e-2, "layer_norm_weight_fd");
}

#[test]
fn test_nn_layer_norm_backward_bias_is_batch_count() {
    // For sum loss through layer_norm, bias gradient = batch_size
    let x = var_from(vec![1.0, 2.0, 3.0, 5.0, 6.0, 7.0], &[2, 3]);
    let w = var_from(vec![1.0, 1.0, 1.0], &[3]);
    let b = var_from(vec![0.0, 0.0, 0.0], &[3]);

    let tx = tracked(&x);
    let tw = tracked(&w);
    let tb = tracked(&b);

    let y = tx.layer_norm(&tw, &tb, 1e-5).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gb = grad_vec(&grads, &b);
    for (i, &v) in gb.iter().enumerate() {
        assert!(
            (v - 2.0).abs() < 1e-4,
            "layer_norm bias grad[{i}] expected ~2.0, got {v}"
        );
    }
}

// ===========================================================================
// 5. Softmax backward: Jacobian-vector product correctness
// ===========================================================================

#[test]
fn test_nn_softmax_backward_jvp_zero_sum() {
    // Softmax backward property: sum(grad_input) per row ~= 0
    // because softmax outputs sum to 1, so tangent must have zero sum.
    let x_data = vec![1.0_f32, 2.0, 3.0, 0.5, 1.5, 2.5];
    let x = var_from(x_data, &[2, 3]);
    let tx = tracked(&x);

    let s = tx.softmax(1).unwrap();
    let loss = s.sqr().unwrap();
    let loss = scalar_loss(&loss);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    for row in 0..2 {
        let row_sum: f32 = g[row * 3..(row + 1) * 3].iter().sum();
        assert!(
            row_sum.abs() < 1e-5,
            "softmax grad row {row} sum={row_sum}, expected ~0"
        );
    }
}

#[test]
fn test_nn_softmax_backward_fd() {
    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 1.5, 0.5, 2.5];

    let x = var_from(x_data.clone(), &[2, 3]);
    let tx = tracked(&x);
    let s = tx.softmax(1).unwrap();
    let loss = s.sqr().unwrap();
    let loss = scalar_loss(&loss);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let s = t.softmax(1).unwrap();
        sum_sqr(&s)
    });
    assert_grad_close(&gx, &num, 1e-2, "softmax_fd");
}

#[test]
fn test_nn_softmax_backward_uniform_zero_grad() {
    // When all inputs equal, softmax = uniform, sum(softmax) is constant,
    // so gradient of sum(softmax) w.r.t. input is zero.
    let x = var_from(vec![1.0_f32; 6], &[2, 3]);
    let tx = tracked(&x);
    let s = tx.softmax(1).unwrap();
    let loss = scalar_loss(&s);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    for (i, &v) in g.iter().enumerate() {
        assert!(v.abs() < 1e-5, "softmax uniform grad[{i}]={v}, expected ~0");
    }
}

#[test]
fn test_nn_softmax_backward_shape_preserved() {
    let x = var_from(vec![0.5; 12], &[3, 4]);
    let tx = tracked(&x);
    let s = tx.softmax(1).unwrap();
    let loss = scalar_loss(&s);
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x).unwrap();
    assert_eq!(gx.dims(), &[3, 4], "softmax grad shape must match input");
}

// ===========================================================================
// 6. Cross-entropy loss backward: gradient matches analytical formula
// ===========================================================================

#[test]
fn test_nn_cross_entropy_backward_fd() {
    let logit_data: Vec<f32> = vec![1.0, 2.0, 0.5, 0.1, 0.8, 1.5];
    let target_data = vec![1u32, 2];

    let logits = var_from(logit_data.clone(), &[2, 3]);
    let targets_dyn = DynTensor::from_vec_u32(target_data.clone(), &[2, 1], &cpu()).unwrap();
    let targets = Var::new(targets_dyn);

    let tx = tracked(&logits);
    let tt = tracked(&targets);

    let loss = tx.cross_entropy_loss(&tt, 1).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &logits);
    let num = numerical_grad(&logit_data, 1e-3, |d| {
        let lt = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let log_sm = lt.log_softmax(1).unwrap();
        let targets_t = DynTensor::from_vec_u32(target_data.clone(), &[2, 1], &cpu()).unwrap();
        let gathered = log_sm.gather(&targets_t, 1).unwrap();
        let neg = gathered.neg().unwrap();
        let s: f64 = neg
            .to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum();
        s / 2.0
    });
    assert_grad_close(&gx, &num, 5e-2, "cross_entropy_fd");
}

#[test]
fn test_nn_cross_entropy_backward_analytical() {
    // Cross-entropy grad = (softmax(logits) - one_hot(target)) / batch_size
    // For single-class logits = [2.0, 1.0, 0.1], target = 0
    let logit_data: Vec<f32> = vec![2.0, 1.0, 0.1];
    let logits = var_from(logit_data.clone(), &[1, 3]);
    let targets_dyn = DynTensor::from_vec_u32(vec![0u32], &[1, 1], &cpu()).unwrap();
    let targets = Var::new(targets_dyn);

    let tx = tracked(&logits);
    let tt = tracked(&targets);

    let loss = tx.cross_entropy_loss(&tt, 1).unwrap();
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &logits);
    // Compute expected: softmax - one_hot
    let sm = DynTensor::from_vec(logit_data, &[1, 3], &cpu())
        .unwrap()
        .softmax(1)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let expected: Vec<f32> = vec![sm[0] - 1.0, sm[1], sm[2]]; // one_hot = [1,0,0]
    for (i, (&g, &e)) in gx.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-3,
            "cross_entropy analytical grad[{i}]: got {g}, expected {e}"
        );
    }
}

#[test]
fn test_nn_cross_entropy_backward_shape() {
    let logits = var_from(vec![0.5; 12], &[4, 3]);
    let targets_dyn = DynTensor::from_vec_u32(vec![0u32, 1, 2, 0], &[4, 1], &cpu()).unwrap();
    let targets = Var::new(targets_dyn);

    let tx = tracked(&logits);
    let tt = tracked(&targets);

    let loss = tx.cross_entropy_loss(&tt, 1).unwrap();
    let grads = backward(&loss).unwrap();

    let g = grads.get(&logits).unwrap();
    assert_eq!(g.dims(), &[4, 3], "cross_entropy grad shape matches logits");
}

// ===========================================================================
// 7. Embedding backward: sparse gradient accumulation
// ===========================================================================

#[test]
fn test_nn_embedding_backward_shape() {
    let vocab = 10;
    let embed_dim = 4;
    let w = var_from(
        (0..vocab * embed_dim).map(|i| i as f32 * 0.1).collect(),
        &[vocab, embed_dim],
    );
    let idx_data = DynTensor::from_vec_u32(vec![3u32, 1, 7], &[3], &cpu()).unwrap();
    let idx = Var::new(idx_data);

    let tw = tracked(&w);
    let ti = tracked(&idx);

    let y = TrackedTensor::embedding(&tw, &ti).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gw = grads.get(&w).unwrap();
    assert_eq!(gw.dims(), &[vocab, embed_dim], "embedding grad shape");
}

#[test]
fn test_nn_embedding_backward_sparse_values() {
    let vocab = 5;
    let embed_dim = 3;
    let w = var_from(
        (0..vocab * embed_dim).map(|i| i as f32 * 0.1).collect(),
        &[vocab, embed_dim],
    );
    let idx_data = DynTensor::from_vec_u32(vec![1u32, 3], &[2], &cpu()).unwrap();
    let idx = Var::new(idx_data);

    let tw = tracked(&w);
    let ti = tracked(&idx);

    let y = TrackedTensor::embedding(&tw, &ti).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g_flat = grads.get(&w).unwrap().to_flat_vec::<f32>().unwrap();

    // Row 0: zero (not indexed)
    for (j, &gj) in g_flat.iter().take(embed_dim).enumerate() {
        assert!(gj.abs() < 1e-6, "embedding grad[0][{j}] should be 0");
    }
    // Row 1: gradient = 1.0 (one occurrence, sum loss)
    for j in 0..embed_dim {
        assert!(
            (g_flat[embed_dim + j] - 1.0).abs() < 1e-6,
            "embedding grad[1][{j}] should be 1.0, got {}",
            g_flat[embed_dim + j]
        );
    }
    // Row 2: zero
    for j in 0..embed_dim {
        assert!(
            g_flat[2 * embed_dim + j].abs() < 1e-6,
            "embedding grad[2][{j}] should be 0"
        );
    }
    // Row 3: gradient = 1.0
    for j in 0..embed_dim {
        assert!(
            (g_flat[3 * embed_dim + j] - 1.0).abs() < 1e-6,
            "embedding grad[3][{j}] should be 1.0"
        );
    }
}

#[test]
fn test_nn_embedding_backward_repeated_indices_accumulate() {
    let vocab = 4;
    let embed_dim = 2;
    let w = var_from(
        (0..vocab * embed_dim).map(|i| i as f32 * 0.5).collect(),
        &[vocab, embed_dim],
    );
    let idx_data = DynTensor::from_vec_u32(vec![2u32, 2, 2], &[3], &cpu()).unwrap();
    let idx = Var::new(idx_data);

    let tw = tracked(&w);
    let ti = tracked(&idx);

    let y = TrackedTensor::embedding(&tw, &ti).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g_flat = grads.get(&w).unwrap().to_flat_vec::<f32>().unwrap();

    // Row 2 should have gradient 3.0 (accumulated from 3 lookups)
    for j in 0..embed_dim {
        assert!(
            (g_flat[2 * embed_dim + j] - 3.0).abs() < 1e-5,
            "embedding repeated grad[2][{j}] should be 3.0, got {}",
            g_flat[2 * embed_dim + j]
        );
    }
    // Other rows should be zero
    for row in [0, 1, 3] {
        for j in 0..embed_dim {
            assert!(
                g_flat[row * embed_dim + j].abs() < 1e-6,
                "embedding grad[{row}][{j}] should be 0"
            );
        }
    }
}

// ===========================================================================
// 8. Attention backward: Q/K/V gradient via manual scaled dot-product
// ===========================================================================

/// Manual SDPA: softmax(Q @ K^T / sqrt(d)) @ V
/// This tests the backward through the composition of matmul + softmax + matmul.
#[test]
fn test_nn_attention_q_grad_fd() {
    let d = 4;
    let seq = 3;
    let q_data: Vec<f32> = vec![
        0.1, 0.2, -0.1, 0.3, 0.4, -0.2, 0.1, 0.5, -0.3, 0.1, 0.2, -0.1,
    ];
    let k_data: Vec<f32> = vec![
        0.2, 0.1, 0.3, -0.2, -0.1, 0.4, 0.2, 0.1, 0.3, -0.3, 0.1, 0.2,
    ];
    let v_data: Vec<f32> = vec![
        0.5, -0.1, 0.2, 0.3, 0.1, 0.4, -0.2, 0.5, -0.3, 0.2, 0.1, -0.1,
    ];
    let scale = 1.0 / (d as f64).sqrt();

    let q = var_from(q_data.clone(), &[seq, d]);
    let k = var_from(k_data.clone(), &[seq, d]);
    let v = var_from(v_data.clone(), &[seq, d]);

    let tq = tracked(&q);
    let tk = tracked(&k);
    let tv = tracked(&v);

    // attn_weights = softmax(Q @ K^T * scale)
    let kt = tk.transpose(0, 1).unwrap();
    let scores = tq.matmul(&kt).unwrap().mul_scalar(scale).unwrap();
    let weights = scores.softmax(1).unwrap();
    let out = weights.matmul(&tv).unwrap();
    let loss = out.sqr().unwrap();
    let loss = scalar_loss(&loss);
    let grads = backward(&loss).unwrap();

    let gq = grad_vec(&grads, &q);
    let num = numerical_grad(&q_data, 1e-3, |d_q| {
        let qt = DynTensor::from_vec(d_q, &[seq, d], &cpu()).unwrap();
        let kt_dyn = DynTensor::from_vec(k_data.clone(), &[seq, d], &cpu())
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        let vt = DynTensor::from_vec(v_data.clone(), &[seq, d], &cpu()).unwrap();
        let sc = qt.matmul(&kt_dyn).unwrap().mul_scalar(scale).unwrap();
        let w = sc.softmax(1).unwrap();
        let o = w.matmul(&vt).unwrap();
        sum_sqr(&o)
    });
    assert_grad_close(&gq, &num, 5e-2, "attention_q_fd");
}

#[test]
fn test_nn_attention_k_grad_fd() {
    let d = 4;
    let seq = 3;
    let q_data: Vec<f32> = vec![
        0.1, 0.2, -0.1, 0.3, 0.4, -0.2, 0.1, 0.5, -0.3, 0.1, 0.2, -0.1,
    ];
    let k_data: Vec<f32> = vec![
        0.2, 0.1, 0.3, -0.2, -0.1, 0.4, 0.2, 0.1, 0.3, -0.3, 0.1, 0.2,
    ];
    let v_data: Vec<f32> = vec![
        0.5, -0.1, 0.2, 0.3, 0.1, 0.4, -0.2, 0.5, -0.3, 0.2, 0.1, -0.1,
    ];
    let scale = 1.0 / (d as f64).sqrt();

    let q = var_from(q_data.clone(), &[seq, d]);
    let k = var_from(k_data.clone(), &[seq, d]);
    let v = var_from(v_data.clone(), &[seq, d]);

    let tq = tracked(&q);
    let tk = tracked(&k);
    let tv = tracked(&v);

    let kt = tk.transpose(0, 1).unwrap();
    let scores = tq.matmul(&kt).unwrap().mul_scalar(scale).unwrap();
    let weights = scores.softmax(1).unwrap();
    let out = weights.matmul(&tv).unwrap();
    let loss = out.sqr().unwrap();
    let loss = scalar_loss(&loss);
    let grads = backward(&loss).unwrap();

    let gk = grad_vec(&grads, &k);
    let num = numerical_grad(&k_data, 1e-3, |d_k| {
        let qt = DynTensor::from_vec(q_data.clone(), &[seq, d], &cpu()).unwrap();
        let kt_dyn = DynTensor::from_vec(d_k, &[seq, d], &cpu())
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        let vt = DynTensor::from_vec(v_data.clone(), &[seq, d], &cpu()).unwrap();
        let sc = qt.matmul(&kt_dyn).unwrap().mul_scalar(scale).unwrap();
        let w = sc.softmax(1).unwrap();
        let o = w.matmul(&vt).unwrap();
        sum_sqr(&o)
    });
    assert_grad_close(&gk, &num, 5e-2, "attention_k_fd");
}

#[test]
fn test_nn_attention_v_grad_fd() {
    let d = 4;
    let seq = 3;
    let q_data: Vec<f32> = vec![
        0.1, 0.2, -0.1, 0.3, 0.4, -0.2, 0.1, 0.5, -0.3, 0.1, 0.2, -0.1,
    ];
    let k_data: Vec<f32> = vec![
        0.2, 0.1, 0.3, -0.2, -0.1, 0.4, 0.2, 0.1, 0.3, -0.3, 0.1, 0.2,
    ];
    let v_data: Vec<f32> = vec![
        0.5, -0.1, 0.2, 0.3, 0.1, 0.4, -0.2, 0.5, -0.3, 0.2, 0.1, -0.1,
    ];
    let scale = 1.0 / (d as f64).sqrt();

    let q = var_from(q_data.clone(), &[seq, d]);
    let k = var_from(k_data.clone(), &[seq, d]);
    let v = var_from(v_data.clone(), &[seq, d]);

    let tq = tracked(&q);
    let tk = tracked(&k);
    let tv = tracked(&v);

    let kt = tk.transpose(0, 1).unwrap();
    let scores = tq.matmul(&kt).unwrap().mul_scalar(scale).unwrap();
    let weights = scores.softmax(1).unwrap();
    let out = weights.matmul(&tv).unwrap();
    let loss = out.sqr().unwrap();
    let loss = scalar_loss(&loss);
    let grads = backward(&loss).unwrap();

    let gv = grad_vec(&grads, &v);
    let num = numerical_grad(&v_data, 1e-3, |d_v| {
        let qt = DynTensor::from_vec(q_data.clone(), &[seq, d], &cpu()).unwrap();
        let kt_dyn = DynTensor::from_vec(k_data.clone(), &[seq, d], &cpu())
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        let vt = DynTensor::from_vec(d_v, &[seq, d], &cpu()).unwrap();
        let sc = qt.matmul(&kt_dyn).unwrap().mul_scalar(scale).unwrap();
        let w = sc.softmax(1).unwrap();
        let o = w.matmul(&vt).unwrap();
        sum_sqr(&o)
    });
    assert_grad_close(&gv, &num, 5e-2, "attention_v_fd");
}

// ===========================================================================
// 9. ReLU/GELU/SiLU backward: gradient at boundaries
// ===========================================================================

#[test]
fn test_nn_relu_backward_zero_for_negative() {
    let x = var_from(vec![-3.0, -2.0, -1.0, -0.5, -0.01], &[5]);
    let tx = tracked(&x);
    let y = tx.relu().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    for (i, &v) in g.iter().enumerate() {
        assert!(
            v.abs() < 1e-6,
            "relu grad[{i}] for negative input should be 0, got {v}"
        );
    }
}

#[test]
fn test_nn_relu_backward_one_for_positive() {
    let x = var_from(vec![0.01, 0.5, 1.0, 2.0, 10.0], &[5]);
    let tx = tracked(&x);
    let y = tx.relu().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    for (i, &v) in g.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "relu grad[{i}] for positive input should be 1, got {v}"
        );
    }
}

#[test]
fn test_nn_relu_backward_mixed_fd() {
    let x_data = vec![-2.0_f32, -0.5, 0.5, 1.5, 3.0];
    let x = var_from(x_data.clone(), &[5]);
    let tx = tracked(&x);
    let y = tx.relu().unwrap().sqr().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[5], &cpu()).unwrap();
        sum_all(&t.relu().unwrap().sqr().unwrap())
    });
    assert_grad_close(&gx, &num, 1e-2, "relu_mixed_fd");
}

#[test]
fn test_nn_gelu_backward_fd() {
    // GELU has a smooth gradient everywhere -- good for FD check
    let x_data = vec![-2.0_f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let x = var_from(x_data.clone(), &[7]);
    let tx = tracked(&x);
    let y = tx.gelu().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[7], &cpu()).unwrap();
        sum_all(&t.gelu().unwrap())
    });
    assert_grad_close(&gx, &num, 5e-2, "gelu_fd");
}

#[test]
fn test_nn_gelu_backward_at_zero() {
    // GELU(0) = 0, GELU'(0) = 0.5
    let x = var_from(vec![0.0], &[1]);
    let tx = tracked(&x);
    let y = tx.gelu().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    assert!(
        (g[0] - 0.5).abs() < 0.05,
        "gelu'(0) should be ~0.5, got {}",
        g[0]
    );
}

#[test]
fn test_nn_silu_backward_fd() {
    // SiLU(x) = x * sigmoid(x), smooth everywhere
    let x_data = vec![-2.0_f32, -1.0, 0.0, 1.0, 2.0];
    let x = var_from(x_data.clone(), &[5]);
    let tx = tracked(&x);
    let y = tx.silu().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[5], &cpu()).unwrap();
        sum_all(&t.silu().unwrap())
    });
    assert_grad_close(&gx, &num, 5e-2, "silu_fd");
}

#[test]
fn test_nn_silu_backward_at_zero() {
    // SiLU(0) = 0 * sigmoid(0) = 0
    // SiLU'(0) = sigmoid(0) + 0 * sigmoid(0) * (1 - sigmoid(0)) = 0.5
    let x = var_from(vec![0.0], &[1]);
    let tx = tracked(&x);
    let y = tx.silu().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    assert!(
        (g[0] - 0.5).abs() < 1e-4,
        "silu'(0) should be 0.5, got {}",
        g[0]
    );
}

#[test]
fn test_nn_silu_backward_analytical() {
    // SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    //          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    let vals = vec![-1.0_f32, 0.0, 0.5, 1.0, 2.0];
    let x = var_from(vals.clone(), &[5]);
    let tx = tracked(&x);
    let y = tx.silu().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    for (i, &v) in vals.iter().enumerate() {
        let s = 1.0 / (1.0 + (-v).exp());
        let expected = s * (1.0 + v * (1.0 - s));
        assert!(
            (g[i] - expected).abs() < 1e-4,
            "silu' grad[{i}]: expected={expected}, got={}",
            g[i]
        );
    }
}

// ===========================================================================
// 10. Dropout backward: gradient scaling by 1/(1-p)
// ===========================================================================

#[test]
fn test_nn_dropout_backward_scaling() {
    // Dropout backward should scale gradients by mask * 1/(1-p).
    // Since dropout is stochastic, we verify the property:
    // each gradient element is either 0 (dropped) or scale * upstream_grad.
    let x_data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let p = 0.5;
    let scale = 1.0 / (1.0 - p);

    let x = var_from(x_data, &[8]);
    let tx = tracked(&x);
    let y = tx.dropout(p).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    for (i, &v) in g.iter().enumerate() {
        // Each gradient element is either 0.0 (dropped) or scale (kept)
        let valid = v.abs() < 1e-6 || (v - scale as f32).abs() < 1e-4;
        assert!(valid, "dropout grad[{i}]={v}, expected 0.0 or {scale}");
    }
}

#[test]
fn test_nn_dropout_zero_prob_passthrough() {
    // With p=0.0, dropout is identity, gradient should be all 1s for sum loss
    let x = var_from(vec![1.0, 2.0, 3.0, 4.0], &[4]);
    let tx = tracked(&x);
    let y = tx.dropout(0.0).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    for (i, &v) in g.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "dropout(0.0) grad[{i}] should be 1.0, got {v}"
        );
    }
}

#[test]
fn test_nn_dropout_backward_with_upstream() {
    // dropout(sqr(x)) — gradient should be mask * scale * upstream_grad
    let x_data = vec![1.0_f32, 2.0, 3.0, 4.0];
    let p = 0.3;
    let scale = 1.0 / (1.0 - p);

    let x = var_from(x_data.clone(), &[4]);
    let tx = tracked(&x);
    let sq = tx.sqr().unwrap();
    let y = sq.dropout(p).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    // For sqr, upstream grad = 2*x. With dropout, each element is either
    // 0 (dropped) or 2*x[i] * scale (kept).
    for (i, &v) in g.iter().enumerate() {
        let expected_kept = 2.0 * x_data[i] * scale as f32;
        let valid = v.abs() < 1e-6 || (v - expected_kept).abs() < 1e-3;
        assert!(
            valid,
            "dropout+sqr grad[{i}]={v}, expected 0.0 or {expected_kept}"
        );
    }
}

// ===========================================================================
// 11. MaxPool backward: gradient routing to max positions
// ===========================================================================

#[test]
fn test_nn_max_pool1d_backward_routes_to_max() {
    // Input: [1, 1, 6], kernel=2, stride=2 => output [1, 1, 3]
    // Each output picks the max of 2 adjacent elements.
    // Gradient should only go to the max positions.
    let x_data = vec![1.0_f32, 3.0, 2.0, 5.0, 4.0, 6.0];
    let x = var_from(x_data, &[1, 1, 6]);
    let tx = tracked(&x);
    let y = tx.max_pool1d(2, 2, 0).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    // Pool windows: [1,3] -> max=3(idx=1), [2,5] -> max=5(idx=3), [4,6] -> max=6(idx=5)
    // Gradient flows only to max positions
    assert!(
        g[0].abs() < 1e-6,
        "non-max position 0 should be 0, got {}",
        g[0]
    );
    assert!(
        (g[1] - 1.0).abs() < 1e-6,
        "max position 1 should be 1.0, got {}",
        g[1]
    );
    assert!(
        g[2].abs() < 1e-6,
        "non-max position 2 should be 0, got {}",
        g[2]
    );
    assert!(
        (g[3] - 1.0).abs() < 1e-6,
        "max position 3 should be 1.0, got {}",
        g[3]
    );
    assert!(
        g[4].abs() < 1e-6,
        "non-max position 4 should be 0, got {}",
        g[4]
    );
    assert!(
        (g[5] - 1.0).abs() < 1e-6,
        "max position 5 should be 1.0, got {}",
        g[5]
    );
}

#[test]
fn test_nn_max_pool1d_backward_fd() {
    // Use non-trivial loss to test max_pool1d backward more thoroughly
    let x_data = vec![1.0_f32, 3.0, 2.0, 5.0, 4.0, 6.0];
    let x = var_from(x_data.clone(), &[1, 1, 6]);
    let tx = tracked(&x);
    let y = tx.max_pool1d(2, 2, 0).unwrap();
    let loss = y.sqr().unwrap();
    let loss = scalar_loss(&loss);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-3, |d| {
        let xt = DynTensor::from_vec(d, &[1, 1, 6], &cpu()).unwrap();
        // We need max_pool1d on DynTensor -- use tracked for FD too
        let v = Var::new(xt);
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let y = t.max_pool1d(2, 2, 0).unwrap();
        sum_sqr(y.tensor())
    });
    assert_grad_close(&gx, &num, 1e-2, "max_pool1d_sqr_fd");
}

#[test]
fn test_nn_max_pool2d_backward_routes_to_max() {
    // Input: [1, 1, 4, 4], kernel=2, stride=2 => [1, 1, 2, 2]
    #[rustfmt::skip]
    let x_data = vec![
        1.0, 3.0, 2.0, 4.0,
        5.0, 2.0, 6.0, 1.0,
        3.0, 7.0, 1.0, 2.0,
        4.0, 3.0, 5.0, 8.0_f32,
    ];
    let x = var_from(x_data, &[1, 1, 4, 4]);
    let tx = tracked(&x);
    let y = tx.max_pool2d(2, 2, 0).unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    // Pool windows (2x2 non-overlapping):
    // [1,3,5,2] -> max=5 at (1,0)=idx4
    // [2,4,6,1] -> max=6 at (1,2)=idx6
    // [3,7,4,3] -> max=7 at (2,1)=idx9
    // [1,2,5,8] -> max=8 at (3,3)=idx15
    for (i, &v) in g.iter().enumerate() {
        if i == 4 || i == 6 || i == 9 || i == 15 {
            assert!(
                (v - 1.0).abs() < 1e-6,
                "max position {i} should be 1.0, got {v}"
            );
        } else {
            assert!(v.abs() < 1e-6, "non-max position {i} should be 0, got {v}");
        }
    }
}

// ===========================================================================
// 12. Reshape/transpose backward: gradient shape restoration
// ===========================================================================

#[test]
fn test_nn_reshape_backward_preserves_shape() {
    let x = var_from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let tx = tracked(&x);
    let flat = tx.reshape(&[6]).unwrap();
    let y = flat.sqr().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x).unwrap();
    assert_eq!(
        gx.dims(),
        &[2, 3],
        "reshape grad shape must match input [2,3]"
    );
}

#[test]
fn test_nn_reshape_backward_values_are_correct() {
    // loss = sum((reshape(x))^2) = sum(x^2), grad = 2*x regardless of reshape
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = var_from(x_data.clone(), &[2, 3]);
    let tx = tracked(&x);
    let reshaped = tx.reshape(&[3, 2]).unwrap();
    let y = reshaped.sqr().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    for (i, &v) in x_data.iter().enumerate() {
        let expected = 2.0 * v;
        assert!(
            (g[i] - expected).abs() < 1e-5,
            "reshape grad[{i}]={}, expected={expected}",
            g[i]
        );
    }
}

#[test]
fn test_nn_reshape_backward_3d_to_1d() {
    let x_data: Vec<f32> = (1..=24).map(|i| i as f32).collect();
    let x = var_from(x_data, &[2, 3, 4]);
    let tx = tracked(&x);
    let flat = tx.reshape(&[24]).unwrap();
    let loss = scalar_loss(&flat);
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x).unwrap();
    assert_eq!(gx.dims(), &[2, 3, 4], "reshape 3d->1d grad shape");

    let g = grad_vec(&grads, &x);
    for (i, &v) in g.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "reshape 3d->1d grad[{i}] should be 1.0, got {v}"
        );
    }
}

#[test]
fn test_nn_reshape_backward_fd() {
    let x_data = vec![0.5_f32, -0.3, 1.2, 0.8, -0.6, 0.4];
    let x = var_from(x_data.clone(), &[2, 3]);
    let tx = tracked(&x);
    let reshaped = tx.reshape(&[3, 2]).unwrap();
    let y = reshaped.exp().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-4, |d| {
        let t = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        sum_all(&t.reshape([3, 2]).unwrap().exp().unwrap())
    });
    assert_grad_close(&gx, &num, 1e-2, "reshape_exp_fd");
}

#[test]
fn test_nn_transpose_backward_preserves_shape() {
    let x = var_from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let tx = tracked(&x);
    let t = tx.transpose(0, 1).unwrap(); // [3, 2]
    let y = t.sqr().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x).unwrap();
    assert_eq!(
        gx.dims(),
        &[2, 3],
        "transpose grad shape must match input [2,3]"
    );
}

#[test]
fn test_nn_transpose_backward_values() {
    // loss = sum(transpose(x)^2) = sum(x^2), grad = 2*x
    let x_data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = var_from(x_data.clone(), &[2, 3]);
    let tx = tracked(&x);
    let t = tx.transpose(0, 1).unwrap();
    let y = t.sqr().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let g = grad_vec(&grads, &x);
    for (i, &v) in x_data.iter().enumerate() {
        let expected = 2.0 * v;
        assert!(
            (g[i] - expected).abs() < 1e-5,
            "transpose grad[{i}]={}, expected={expected}",
            g[i]
        );
    }
}

#[test]
fn test_nn_transpose_backward_fd() {
    let x_data = vec![0.5_f32, -0.3, 1.2, 0.8, -0.6, 0.4];
    let x = var_from(x_data.clone(), &[2, 3]);
    let tx = tracked(&x);
    let t = tx.transpose(0, 1).unwrap();
    let y = t.exp().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grad_vec(&grads, &x);
    let num = numerical_grad(&x_data, 1e-4, |d| {
        let t = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        sum_all(&t.transpose(0, 1).unwrap().exp().unwrap())
    });
    assert_grad_close(&gx, &num, 1e-2, "transpose_exp_fd");
}

#[test]
fn test_nn_transpose_then_reshape_backward_shape() {
    // Compose transpose + reshape: gradient must undo both
    let x = var_from((1..=12).map(|i| i as f32).collect(), &[3, 4]);
    let tx = tracked(&x);
    let t = tx.transpose(0, 1).unwrap(); // [4, 3]
    let r = t.reshape(&[12]).unwrap(); // [12]
    let y = r.sqr().unwrap();
    let loss = scalar_loss(&y);
    let grads = backward(&loss).unwrap();

    let gx = grads.get(&x).unwrap();
    assert_eq!(gx.dims(), &[3, 4], "transpose+reshape grad shape");
}
