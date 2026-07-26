#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Finite-difference gradient-checking tests for key autodiff operations.
//!
//! Each test computes f(x+eps) and f(x-eps) for central-difference numerical
//! gradients, then compares against autodiff backward results. Covers:
//!
//! - MatMul (2D, batched 3D x 2D with broadcast reduction)
//! - Softmax (nonlinear sum-of-squares loss to expose Jacobian bugs)
//! - Div (both operands)
//! - Conv1d with padding (both input and kernel)
//! - Scaled dot-product attention pattern (Q @ K^T / sqrt(d) -> softmax -> @ V)
//! - LogSoftmax (with sum-of-squares loss)
//! - MulScalar and AddScalar

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::grad::test_helpers::{check_fd_grad, check_fd_grad_tol};
use crate::tracked::TrackedTensor;
use crate::var::Var;

// ── MatMul 2D finite-difference ──────────────────────────────────────

/// Reference forward for 2D matmul: sum(sqr(a @ b)) as scalar loss.
/// Using sum-of-squares produces input-dependent gradients (unlike sum).
fn matmul_2d_loss(a_data: Vec<f32>, b_data: &[f32]) -> f64 {
    let a = DynTensor::from_vec(a_data, &[2, 3], &cpu()).unwrap();
    let b = DynTensor::from_vec(b_data.to_vec(), &[3, 2], &cpu()).unwrap();
    let y = a.matmul(&b).unwrap();
    y.to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum()
}

#[test]
fn test_backward_matmul_2d_fd_grad_a() {
    let a_data: Vec<f32> = vec![0.5, -0.3, 1.2, 0.8, -0.6, 0.4];
    let b_data: Vec<f32> = vec![1.0, -0.5, 0.3, 0.7, -0.2, 1.1];

    let a_var = Var::new(DynTensor::from_vec(a_data.clone(), &[2, 3], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(b_data.clone(), &[3, 2], &cpu()).unwrap());

    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = ta.matmul(&tb).unwrap();
    // loss = sum(y^2) to get input-dependent gradients
    let y_sq = y.sqr().unwrap();
    let loss = y_sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();

    let analytical_a = grads.get(&a_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_a, &a_data, 1e-3, |d| matmul_2d_loss(d, &b_data));
}

#[test]
fn test_backward_matmul_2d_fd_grad_b() {
    let a_data: Vec<f32> = vec![0.5, -0.3, 1.2, 0.8, -0.6, 0.4];
    let b_data: Vec<f32> = vec![1.0, -0.5, 0.3, 0.7, -0.2, 1.1];

    let a_var = Var::new(DynTensor::from_vec(a_data.clone(), &[2, 3], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(b_data.clone(), &[3, 2], &cpu()).unwrap());

    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = ta.matmul(&tb).unwrap();
    let y_sq = y.sqr().unwrap();
    let loss = y_sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();

    let analytical_b = grads.get(&b_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_b, &b_data, 1e-3, |d| {
        let a = DynTensor::from_vec(a_data.clone(), &[2, 3], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[3, 2], &cpu()).unwrap();
        let y = a.matmul(&b).unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
}

// ── MatMul 3D x 2D (broadcast reduction) finite-difference ──────────

/// Reference forward for 3D x 2D matmul: sum(sqr(a @ b)).
/// a is [B=2, S=1, D=3], b is [D=3, H=2].
fn matmul_3d_2d_loss_a(a_data: Vec<f32>, b_data: &[f32]) -> f64 {
    let a = DynTensor::from_vec(a_data, &[2, 1, 3], &cpu()).unwrap();
    let b = DynTensor::from_vec(b_data.to_vec(), &[3, 2], &cpu()).unwrap();
    let y = a.matmul(&b).unwrap();
    y.to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum()
}

#[test]
fn test_backward_matmul_3d_x_2d_fd_grad_a() {
    let a_data: Vec<f32> = vec![1.0, 0.5, -0.3, 0.8, -0.6, 0.4];
    let b_data: Vec<f32> = vec![0.5, -0.2, 1.0, 0.3, -0.7, 0.9];

    let a_var = Var::new(DynTensor::from_vec(a_data.clone(), &[2, 1, 3], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(b_data.clone(), &[3, 2], &cpu()).unwrap());

    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = ta.matmul(&tb).unwrap();
    let y_sq = y.sqr().unwrap();
    let loss = y_sq
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let analytical = grads.get(&a_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &a_data, 1e-3, |d| {
        matmul_3d_2d_loss_a(d, &b_data)
    });
}

#[test]
fn test_backward_matmul_3d_x_2d_fd_grad_b() {
    let a_data: Vec<f32> = vec![1.0, 0.5, -0.3, 0.8, -0.6, 0.4];
    let b_data: Vec<f32> = vec![0.5, -0.2, 1.0, 0.3, -0.7, 0.9];

    let a_var = Var::new(DynTensor::from_vec(a_data.clone(), &[2, 1, 3], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(b_data.clone(), &[3, 2], &cpu()).unwrap());

    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = ta.matmul(&tb).unwrap();
    let y_sq = y.sqr().unwrap();
    let loss = y_sq
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();

    // grad_b must be reduced back to [3, 2] from broadcast
    let gb = grads.get(&b_var).unwrap();
    assert_eq!(
        gb.dims(),
        &[3, 2],
        "grad_b shape must match b shape after reduction"
    );

    let analytical = gb.to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &b_data, 1e-3, |d| {
        let a = DynTensor::from_vec(a_data.clone(), &[2, 1, 3], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[3, 2], &cpu()).unwrap();
        let y = a.matmul(&b).unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
}

// ── Softmax with nonlinear loss finite-difference ────────────────────

/// Softmax backward with sum-of-squares loss exposes Jacobian formula errors
/// that sum-loss (uniform gradient) would miss.
#[test]
fn test_backward_softmax_sum_sqr_loss_fd() {
    let x_data: Vec<f32> = vec![1.0, 2.0, 0.5, -1.0, 0.3, 1.5];

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let sm = tx.softmax(1).unwrap();
    // loss = sum(softmax(x)^2) -- nonlinear loss
    let loss_t = sm.sqr().unwrap();
    let loss = loss_t.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();

    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let s = t.softmax(1).unwrap();
        s.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
}

// ── Div backward finite-difference ───────────────────────────────────

#[test]
fn test_backward_div_fd_grad_a() {
    // a / b: grad_a = grad / b
    let a_data: Vec<f32> = vec![2.0, -1.5, 3.0, 0.5];
    let b_data: Vec<f32> = vec![1.5, 2.0, -0.8, 3.0]; // no zeros

    let a_var = Var::new(DynTensor::from_vec(a_data.clone(), &[2, 2], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(b_data.clone(), &[2, 2], &cpu()).unwrap());

    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = ta.div(&tb).unwrap();
    let loss = y
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let analytical = grads.get(&a_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &a_data, 1e-3, |d| {
        let a = DynTensor::from_vec(d, &[2, 2], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[2, 2], &cpu()).unwrap();
        let y = a.div(&b).unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
}

#[test]
fn test_backward_div_fd_grad_b() {
    // a / b: grad_b = -a * grad / b^2
    let a_data: Vec<f32> = vec![2.0, -1.5, 3.0, 0.5];
    let b_data: Vec<f32> = vec![1.5, 2.0, -0.8, 3.0];

    let a_var = Var::new(DynTensor::from_vec(a_data.clone(), &[2, 2], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(b_data.clone(), &[2, 2], &cpu()).unwrap());

    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = ta.div(&tb).unwrap();
    let loss = y
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let analytical = grads.get(&b_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &b_data, 1e-3, |d| {
        let a = DynTensor::from_vec(a_data.clone(), &[2, 2], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[2, 2], &cpu()).unwrap();
        let y = a.div(&b).unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
}

// ── Conv1d with padding finite-difference ────────────────────────────

#[test]
fn test_backward_conv1d_padding_fd_input() {
    // [B=1, C_in=1, L=6], kernel [C_out=1, C_in=1, K=3], padding=1, stride=1
    let x_data: Vec<f32> = vec![1.0, -0.5, 2.0, 0.3, -1.0, 0.7];
    let k_data: Vec<f32> = vec![0.5, -0.3, 0.8];

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 1, 6], &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(k_data.clone(), &[1, 1, 3], &cpu()).unwrap());

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx.conv1d(&tk, 1, 1, 1, 1).unwrap();
    let loss = y
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-3, |d| {
        let x = DynTensor::from_vec(d, &[1, 1, 6], &cpu()).unwrap();
        let k = DynTensor::from_vec(k_data.clone(), &[1, 1, 3], &cpu()).unwrap();
        let y = x.conv1d(&k, 1, 1, 1, 1).unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
}

#[test]
fn test_backward_conv1d_padding_fd_kernel() {
    let x_data: Vec<f32> = vec![1.0, -0.5, 2.0, 0.3, -1.0, 0.7];
    let k_data: Vec<f32> = vec![0.5, -0.3, 0.8];

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 1, 6], &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(k_data.clone(), &[1, 1, 3], &cpu()).unwrap());

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx.conv1d(&tk, 1, 1, 1, 1).unwrap();
    let loss = y
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let analytical = grads.get(&k_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &k_data, 1e-3, |d| {
        let x = DynTensor::from_vec(x_data.clone(), &[1, 1, 6], &cpu()).unwrap();
        let k = DynTensor::from_vec(d, &[1, 1, 3], &cpu()).unwrap();
        let y = x.conv1d(&k, 1, 1, 1, 1).unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
}

// ── Scaled dot-product attention pattern FD ──────────────────────────

/// Tests the core attention computation: softmax(Q @ K^T / sqrt(d)) @ V.
/// Verifies gradient correctness for Q, K, and V simultaneously.
/// This is the pattern used in every transformer (Whisper, Qwen3, Kokoro).
#[test]
fn test_backward_attention_pattern_fd_q() {
    // Q=[1,2,4], K=[1,3,4], V=[1,3,2] => out=[1,2,2]
    // attention: softmax(Q @ K^T / sqrt(4)) @ V
    let q_data: Vec<f32> = vec![0.5, -0.3, 1.0, 0.2, 0.8, 0.1, -0.5, 0.7];
    let k_data: Vec<f32> = vec![
        0.3, -0.1, 0.4, 0.6, -0.2, 0.5, 0.1, 0.3, 0.7, -0.4, 0.2, 0.8,
    ];
    let v_data: Vec<f32> = vec![1.0, 0.5, -0.3, 0.7, 0.4, -0.6];

    let q_var = Var::new(DynTensor::from_vec(q_data.clone(), &[1, 2, 4], &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(k_data.clone(), &[1, 3, 4], &cpu()).unwrap());
    let v_var = Var::new(DynTensor::from_vec(v_data.clone(), &[1, 3, 2], &cpu()).unwrap());

    let tq = Arc::new(TrackedTensor::from_var(&q_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let tv = Arc::new(TrackedTensor::from_var(&v_var).unwrap());

    let scale = 1.0 / (4.0_f64).sqrt();
    let kt = tk.transpose(1, 2).unwrap();
    let scores = tq.matmul(&kt).unwrap(); // [1,2,3]
    let scaled = scores.mul_scalar(scale).unwrap();
    let attn_weights = scaled.softmax(2).unwrap(); // [1,2,3]
    let out = attn_weights.matmul(&tv).unwrap(); // [1,2,2]
    let loss = out
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();

    // FD check for Q gradients
    let analytical_q = grads.get(&q_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad_tol(&analytical_q, &q_data, 1e-3, 1e-2, &|d| {
        let q = DynTensor::from_vec(d, &[1, 2, 4], &cpu()).unwrap();
        let k = DynTensor::from_vec(k_data.clone(), &[1, 3, 4], &cpu()).unwrap();
        let v = DynTensor::from_vec(v_data.clone(), &[1, 3, 2], &cpu()).unwrap();
        let scores = q.matmul(&k.transpose(1, 2).unwrap()).unwrap();
        let attn = scores.mul_scalar(scale).unwrap().softmax(2).unwrap();
        let out = attn.matmul(&v).unwrap();
        out.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
}

#[test]
fn test_backward_attention_pattern_fd_k() {
    let q_data: Vec<f32> = vec![0.5, -0.3, 1.0, 0.2, 0.8, 0.1, -0.5, 0.7];
    let k_data: Vec<f32> = vec![
        0.3, -0.1, 0.4, 0.6, -0.2, 0.5, 0.1, 0.3, 0.7, -0.4, 0.2, 0.8,
    ];
    let v_data: Vec<f32> = vec![1.0, 0.5, -0.3, 0.7, 0.4, -0.6];
    let scale = 1.0 / (4.0_f64).sqrt();

    let q_var = Var::new(DynTensor::from_vec(q_data.clone(), &[1, 2, 4], &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(k_data.clone(), &[1, 3, 4], &cpu()).unwrap());
    let v_var = Var::new(DynTensor::from_vec(v_data.clone(), &[1, 3, 2], &cpu()).unwrap());

    let tq = Arc::new(TrackedTensor::from_var(&q_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let tv = Arc::new(TrackedTensor::from_var(&v_var).unwrap());

    let kt = tk.transpose(1, 2).unwrap();
    let scores = tq.matmul(&kt).unwrap();
    let scaled = scores.mul_scalar(scale).unwrap();
    let attn_weights = scaled.softmax(2).unwrap();
    let out = attn_weights.matmul(&tv).unwrap();
    let loss = out
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();

    // FD check for K gradients
    let analytical_k = grads.get(&k_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad_tol(&analytical_k, &k_data, 1e-3, 1e-2, &|d| {
        let q = DynTensor::from_vec(q_data.clone(), &[1, 2, 4], &cpu()).unwrap();
        let k = DynTensor::from_vec(d, &[1, 3, 4], &cpu()).unwrap();
        let v = DynTensor::from_vec(v_data.clone(), &[1, 3, 2], &cpu()).unwrap();
        let scores = q.matmul(&k.transpose(1, 2).unwrap()).unwrap();
        let attn = scores.mul_scalar(scale).unwrap().softmax(2).unwrap();
        let out = attn.matmul(&v).unwrap();
        out.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
}

#[test]
fn test_backward_attention_pattern_fd_v() {
    let q_data: Vec<f32> = vec![0.5, -0.3, 1.0, 0.2, 0.8, 0.1, -0.5, 0.7];
    let k_data: Vec<f32> = vec![
        0.3, -0.1, 0.4, 0.6, -0.2, 0.5, 0.1, 0.3, 0.7, -0.4, 0.2, 0.8,
    ];
    let v_data: Vec<f32> = vec![1.0, 0.5, -0.3, 0.7, 0.4, -0.6];
    let scale = 1.0 / (4.0_f64).sqrt();

    let q_var = Var::new(DynTensor::from_vec(q_data.clone(), &[1, 2, 4], &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(k_data.clone(), &[1, 3, 4], &cpu()).unwrap());
    let v_var = Var::new(DynTensor::from_vec(v_data.clone(), &[1, 3, 2], &cpu()).unwrap());

    let tq = Arc::new(TrackedTensor::from_var(&q_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let tv = Arc::new(TrackedTensor::from_var(&v_var).unwrap());

    let kt = tk.transpose(1, 2).unwrap();
    let scores = tq.matmul(&kt).unwrap();
    let scaled = scores.mul_scalar(scale).unwrap();
    let attn_weights = scaled.softmax(2).unwrap();
    let out = attn_weights.matmul(&tv).unwrap();
    let loss = out
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();

    // FD check for V gradients
    let analytical_v = grads.get(&v_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad_tol(&analytical_v, &v_data, 1e-3, 1e-2, &|d| {
        let q = DynTensor::from_vec(q_data.clone(), &[1, 2, 4], &cpu()).unwrap();
        let k = DynTensor::from_vec(k_data.clone(), &[1, 3, 4], &cpu()).unwrap();
        let v = DynTensor::from_vec(d, &[1, 3, 2], &cpu()).unwrap();
        let scores = q.matmul(&k.transpose(1, 2).unwrap()).unwrap();
        let attn = scores.mul_scalar(scale).unwrap().softmax(2).unwrap();
        let out = attn.matmul(&v).unwrap();
        out.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
}

// ── LogSoftmax with sum-of-squares loss FD ───────────────────────────

#[test]
fn test_backward_log_softmax_sum_sqr_fd() {
    let x_data: Vec<f32> = vec![1.0, 2.0, -0.5, 0.3, -1.0, 1.5];

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let ls = tx.log_softmax(1).unwrap();
    // Using sqr loss for nonlinear gradients
    let loss = ls
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let ls = t.log_softmax(1).unwrap();
        ls.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
}

// ── MulScalar backward finite-difference ─────────────────────────────

#[test]
fn test_backward_mul_scalar_fd() {
    let x_data: Vec<f32> = vec![1.0, -0.5, 2.3, 0.7];
    let scalar = 2.5_f64;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 2], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.mul_scalar(scalar).unwrap();
    let loss = y
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[2, 2], &cpu()).unwrap();
        let y = t.mul_scalar(scalar).unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
}

// ── AddScalar backward finite-difference ─────────────────────────────

#[test]
fn test_backward_add_scalar_fd() {
    let x_data: Vec<f32> = vec![1.0, -0.5, 2.3, 0.7];
    let scalar = -1.3_f64;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 2], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.add_scalar(scalar).unwrap();
    let loss = y
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[2, 2], &cpu()).unwrap();
        let y = t.add_scalar(scalar).unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
}

// ── Conv1d multi-channel FD (both input and kernel) ──────────────────

#[test]
fn test_backward_conv1d_multichannel_fd_input() {
    // [B=1, C_in=2, L=4], kernel [C_out=2, C_in=2, K=2]
    let x_data: Vec<f32> = vec![1.0, -0.5, 2.0, 0.3, 0.7, -1.0, 0.4, 1.5];
    let k_data: Vec<f32> = vec![0.5, -0.3, 0.8, 0.2, -0.4, 0.6, 0.1, -0.7];

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 2, 4], &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(k_data.clone(), &[2, 2, 2], &cpu()).unwrap());

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx.conv1d(&tk, 0, 1, 1, 1).unwrap();
    let loss = y
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &x_data, 1e-3, |d| {
        let x = DynTensor::from_vec(d, &[1, 2, 4], &cpu()).unwrap();
        let k = DynTensor::from_vec(k_data.clone(), &[2, 2, 2], &cpu()).unwrap();
        let y = x.conv1d(&k, 0, 1, 1, 1).unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
}

#[test]
fn test_backward_conv1d_multichannel_fd_kernel() {
    let x_data: Vec<f32> = vec![1.0, -0.5, 2.0, 0.3, 0.7, -1.0, 0.4, 1.5];
    let k_data: Vec<f32> = vec![0.5, -0.3, 0.8, 0.2, -0.4, 0.6, 0.1, -0.7];

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 2, 4], &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(k_data.clone(), &[2, 2, 2], &cpu()).unwrap());

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx.conv1d(&tk, 0, 1, 1, 1).unwrap();
    let loss = y
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let analytical = grads.get(&k_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical, &k_data, 1e-3, |d| {
        let x = DynTensor::from_vec(x_data.clone(), &[1, 2, 4], &cpu()).unwrap();
        let k = DynTensor::from_vec(d, &[2, 2, 2], &cpu()).unwrap();
        let y = x.conv1d(&k, 0, 1, 1, 1).unwrap();
        y.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
}

// ── LayerNorm with sum-of-squares loss (exercises all 3 parameter grads) ─

#[test]
fn test_backward_layer_norm_sum_sqr_fd_input() {
    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gamma_data: Vec<f32> = vec![1.0, 0.5, 2.0];
    let beta_data: Vec<f32> = vec![0.1, -0.2, 0.3];
    let eps = 1e-5_f64;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap());
    let g_var = Var::new(DynTensor::from_vec(gamma_data.clone(), &[3], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(beta_data.clone(), &[3], &cpu()).unwrap());

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tg = Arc::new(TrackedTensor::from_var(&g_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = tx.layer_norm(&tg, &tb, eps).unwrap();
    // sum-of-squares loss for nonlinear gradients
    let loss = y
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap();
    let grads = backward(&loss).unwrap();

    // FD check for grad_x
    let analytical_x = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_x, &x_data, 1e-3, |d| {
        let x = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let g = DynTensor::from_vec(gamma_data.clone(), &[3], &cpu()).unwrap();
        let b = DynTensor::from_vec(beta_data.clone(), &[3], &cpu()).unwrap();
        // Manual layer_norm forward
        let mean = x.mean_keepdim(1).unwrap();
        let diff = x.sub(&mean).unwrap();
        let var = diff.sqr().unwrap().mean_keepdim(1).unwrap();
        let inv_std = var
            .add_scalar(eps)
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();
        let normed = diff.mul(&inv_std).unwrap();
        let out = normed.mul(&g).unwrap().add(&b).unwrap();
        out.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
}

#[test]
fn test_backward_layer_norm_sum_sqr_fd_gamma() {
    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gamma_data: Vec<f32> = vec![1.0, 0.5, 2.0];
    let beta_data: Vec<f32> = vec![0.1, -0.2, 0.3];
    let eps = 1e-5_f64;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap());
    let g_var = Var::new(DynTensor::from_vec(gamma_data.clone(), &[3], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(beta_data.clone(), &[3], &cpu()).unwrap());

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tg = Arc::new(TrackedTensor::from_var(&g_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = tx.layer_norm(&tg, &tb, eps).unwrap();
    let loss = y
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap();
    let grads = backward(&loss).unwrap();

    // FD check for grad_gamma
    let analytical_g = grads.get(&g_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_g, &gamma_data, 1e-3, |d| {
        let x = DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap();
        let g = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        let b = DynTensor::from_vec(beta_data.clone(), &[3], &cpu()).unwrap();
        let mean = x.mean_keepdim(1).unwrap();
        let diff = x.sub(&mean).unwrap();
        let var = diff.sqr().unwrap().mean_keepdim(1).unwrap();
        let inv_std = var
            .add_scalar(eps)
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();
        let normed = diff.mul(&inv_std).unwrap();
        let out = normed.mul(&g).unwrap().add(&b).unwrap();
        out.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
}

#[test]
fn test_backward_layer_norm_sum_sqr_fd_beta() {
    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gamma_data: Vec<f32> = vec![1.0, 0.5, 2.0];
    let beta_data: Vec<f32> = vec![0.1, -0.2, 0.3];
    let eps = 1e-5_f64;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap());
    let g_var = Var::new(DynTensor::from_vec(gamma_data.clone(), &[3], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(beta_data.clone(), &[3], &cpu()).unwrap());

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tg = Arc::new(TrackedTensor::from_var(&g_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = tx.layer_norm(&tg, &tb, eps).unwrap();
    let loss = y
        .sqr()
        .unwrap()
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap();
    let grads = backward(&loss).unwrap();

    // FD check for grad_beta
    let analytical_b = grads.get(&b_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_b, &beta_data, 1e-3, |d| {
        let x = DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap();
        let g = DynTensor::from_vec(gamma_data.clone(), &[3], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        let mean = x.mean_keepdim(1).unwrap();
        let diff = x.sub(&mean).unwrap();
        let var = diff.sqr().unwrap().mean_keepdim(1).unwrap();
        let inv_std = var
            .add_scalar(eps)
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();
        let normed = diff.mul(&inv_std).unwrap();
        let out = normed.mul(&g).unwrap().add(&b).unwrap();
        out.to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum()
    });
}
