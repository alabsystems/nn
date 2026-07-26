#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive numerical gradient checking tests for autodiff backward rules.
//!
//! Uses central finite-difference: `(f(x+h) - f(x-h)) / (2*h)` vs analytical
//! gradients from backward(). Covers elementwise, reduction, matrix, normalization,
//! activation, shape, and loss ops. All tests use sum-of-squares loss to ensure
//! input-dependent gradients (not trivially uniform).
//!
//! This is a consolidated test suite ensuring every backward rule category has
//! at least one rigorous FD check with non-trivial inputs.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::tracked::TrackedTensor;
use crate::var::Var;

// ── Helpers ─────────────────────────────────────────────────────────────

/// Sum of all elements as f64 (for FD precision).
fn sum_f64(t: &DynTensor) -> f64 {
    t.to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .sum()
}

/// Sum of squares of all elements as f64.
fn sum_sqr_f64(t: &DynTensor) -> f64 {
    t.to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum()
}

/// Central FD gradient check: compare analytical vs numerical.
fn check_fd(analytical: &[f32], data: &[f32], eps: f32, tol: f64, fwd: &dyn Fn(Vec<f32>) -> f64) {
    for i in 0..data.len() {
        let mut plus = data.to_vec();
        let mut minus = data.to_vec();
        plus[i] += eps;
        minus[i] -= eps;
        let numerical = (fwd(plus) - fwd(minus)) / (2.0 * f64::from(eps));
        let analytical_f64 = f64::from(analytical[i]);
        let err = (analytical_f64 - numerical).abs();
        assert!(
            err < tol,
            "grad[{i}]: analytical={analytical_f64}, numerical={numerical}, err={err}, tol={tol}",
        );
    }
}

/// Shorthand for check_fd with default tolerance 1e-2.
fn check_fd_default(analytical: &[f32], data: &[f32], eps: f32, fwd: impl Fn(Vec<f32>) -> f64) {
    check_fd(analytical, data, eps, 1e-2, &fwd);
}

/// Create a Var from data and shape.
fn make_var(data: Vec<f32>, shape: &[usize]) -> Var {
    Var::new(DynTensor::from_vec(data, shape, &cpu()).unwrap())
}

/// Run backward on a sqr+sum loss, return flat gradient for the given var.
fn grad_sqr_sum_loss(build_loss: impl FnOnce() -> (Var, Arc<TrackedTensor>)) -> (Var, Vec<f32>) {
    let (var, output) = build_loss();
    // Build sum-of-squares loss by squaring each element then reducing all dims
    let mut loss = output.sqr().unwrap();
    for d in 0..loss.tensor().rank() {
        loss = loss.sum_keepdim(d).unwrap();
    }
    let grads = backward(&loss).unwrap();
    let g = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();
    (var, g)
}

// ════════════════════════════════════════════════════════════════════════
// 1. ELEMENTWISE OPS
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_gc_add_both_operands() {
    let a_data = vec![0.7, -1.3, 2.1, -0.4];
    let b_data = vec![0.2, 0.8, -1.0, 0.5];

    let a = make_var(a_data.clone(), &[4]);
    let b = make_var(b_data.clone(), &[4]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.add(&tb).unwrap().sqr().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let ga = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let gb = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&ga, &a_data, 1e-3, |d| {
        let a = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[4], &cpu()).unwrap();
        sum_sqr_f64(&a.add(&b).unwrap())
    });
    check_fd_default(&gb, &b_data, 1e-3, |d| {
        let a = DynTensor::from_vec(a_data.clone(), &[4], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        sum_sqr_f64(&a.add(&b).unwrap())
    });
}

#[test]
fn test_gc_sub_both_operands() {
    let a_data = vec![1.5, -0.3, 0.8];
    let b_data = vec![0.5, 1.2, -0.4];

    let a = make_var(a_data.clone(), &[3]);
    let b = make_var(b_data.clone(), &[3]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.sub(&tb).unwrap().sqr().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let ga = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let gb = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&ga, &a_data, 1e-3, |d| {
        let a = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[3], &cpu()).unwrap();
        sum_sqr_f64(&a.sub(&b).unwrap())
    });
    check_fd_default(&gb, &b_data, 1e-3, |d| {
        let a = DynTensor::from_vec(a_data.clone(), &[3], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        sum_sqr_f64(&a.sub(&b).unwrap())
    });
}

#[test]
fn test_gc_mul_both_operands() {
    let a_data = vec![0.5, -1.2, 0.8, 2.3];
    let b_data = vec![1.1, -0.5, 0.3, -0.9];

    let a = make_var(a_data.clone(), &[2, 2]);
    let b = make_var(b_data.clone(), &[2, 2]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.mul(&tb).unwrap().sqr().unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let ga = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&ga, &a_data, 1e-3, |d| {
        let a = DynTensor::from_vec(d, &[2, 2], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[2, 2], &cpu()).unwrap();
        sum_sqr_f64(&a.mul(&b).unwrap())
    });
}

#[test]
fn test_gc_div_both_operands() {
    let a_data = vec![2.0, -1.5, 3.0, 0.5];
    let b_data = vec![1.5, 2.0, -0.8, 3.0];

    let a = make_var(a_data.clone(), &[4]);
    let b = make_var(b_data.clone(), &[4]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.div(&tb).unwrap().sqr().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let ga = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let gb = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&ga, &a_data, 1e-3, |d| {
        let a = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[4], &cpu()).unwrap();
        sum_sqr_f64(&a.div(&b).unwrap())
    });
    check_fd_default(&gb, &b_data, 1e-3, |d| {
        let a = DynTensor::from_vec(a_data.clone(), &[4], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        sum_sqr_f64(&a.div(&b).unwrap())
    });
}

#[test]
fn test_gc_exp() {
    let data = vec![-1.0, 0.0, 0.5, 1.0, 2.0];
    let (_, g) = grad_sqr_sum_loss(|| {
        let v = make_var(data.clone(), &[5]);
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let y = t.exp().unwrap();
        (v, y)
    });
    check_fd_default(&g, &data, 1e-3, |d| {
        sum_sqr_f64(&DynTensor::from_vec(d, &[5], &cpu()).unwrap().exp().unwrap())
    });
}

#[test]
fn test_gc_log() {
    let data = vec![0.5, 1.0, 2.0, 5.0];
    let (_, g) = grad_sqr_sum_loss(|| {
        let v = make_var(data.clone(), &[4]);
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let y = t.log().unwrap();
        (v, y)
    });
    check_fd_default(&g, &data, 1e-3, |d| {
        sum_sqr_f64(&DynTensor::from_vec(d, &[4], &cpu()).unwrap().log().unwrap())
    });
}

#[test]
fn test_gc_sqrt() {
    let data = vec![0.25, 1.0, 4.0, 9.0];
    let (_, g) = grad_sqr_sum_loss(|| {
        let v = make_var(data.clone(), &[4]);
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let y = t.sqrt().unwrap();
        (v, y)
    });
    check_fd_default(&g, &data, 1e-3, |d| {
        sum_sqr_f64(
            &DynTensor::from_vec(d, &[4], &cpu())
                .unwrap()
                .sqrt()
                .unwrap(),
        )
    });
}

#[test]
fn test_gc_sin() {
    let data = vec![-2.0, -0.5, 0.0, 0.5, 1.5];
    let (_, g) = grad_sqr_sum_loss(|| {
        let v = make_var(data.clone(), &[5]);
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let y = t.sin().unwrap();
        (v, y)
    });
    check_fd_default(&g, &data, 1e-3, |d| {
        sum_sqr_f64(&DynTensor::from_vec(d, &[5], &cpu()).unwrap().sin().unwrap())
    });
}

#[test]
fn test_gc_cos() {
    let data = vec![-2.0, -0.5, 0.0, 0.5, 1.5];
    let (_, g) = grad_sqr_sum_loss(|| {
        let v = make_var(data.clone(), &[5]);
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let y = t.cos().unwrap();
        (v, y)
    });
    check_fd_default(&g, &data, 1e-3, |d| {
        sum_sqr_f64(&DynTensor::from_vec(d, &[5], &cpu()).unwrap().cos().unwrap())
    });
}

#[test]
fn test_gc_abs() {
    // Avoid 0 where derivative is discontinuous
    let data = vec![-3.0, -1.0, -0.1, 0.1, 1.0, 3.0];
    let (_, g) = grad_sqr_sum_loss(|| {
        let v = make_var(data.clone(), &[6]);
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let y = t.abs().unwrap();
        (v, y)
    });
    check_fd_default(&g, &data, 1e-3, |d| {
        sum_sqr_f64(&DynTensor::from_vec(d, &[6], &cpu()).unwrap().abs().unwrap())
    });
}

#[test]
fn test_gc_neg() {
    let data = vec![0.3, -1.5, 2.0, -0.8];
    let (_, g) = grad_sqr_sum_loss(|| {
        let v = make_var(data.clone(), &[4]);
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let y = t.neg().unwrap();
        (v, y)
    });
    check_fd_default(&g, &data, 1e-3, |d| {
        sum_sqr_f64(&DynTensor::from_vec(d, &[4], &cpu()).unwrap().neg().unwrap())
    });
}

#[test]
fn test_gc_recip() {
    // Avoid values near 0
    let data = vec![0.5, 1.0, 2.0, -1.5, 3.0];
    let (_, g) = grad_sqr_sum_loss(|| {
        let v = make_var(data.clone(), &[5]);
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        let y = t.recip().unwrap();
        (v, y)
    });
    check_fd_default(&g, &data, 1e-3, |d| {
        sum_sqr_f64(
            &DynTensor::from_vec(d, &[5], &cpu())
                .unwrap()
                .recip()
                .unwrap(),
        )
    });
}

#[test]
fn test_gc_powf_fractional() {
    // p=1.5: d/dx x^1.5 = 1.5 * x^0.5 (positive inputs only)
    let data = vec![0.25, 1.0, 2.0, 4.0];
    let v = make_var(data.clone(), &[4]);
    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let y = t.powf(1.5).unwrap().sqr().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&v).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&g, &data, 1e-3, |d| {
        sum_sqr_f64(
            &DynTensor::from_vec(d, &[4], &cpu())
                .unwrap()
                .powf(1.5)
                .unwrap(),
        )
    });
}

// ════════════════════════════════════════════════════════════════════════
// 2. REDUCTION OPS
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_gc_sum_keepdim_dim0() {
    let data = vec![0.5, -1.2, 0.3, 2.1, -0.8, 0.7];
    let v = make_var(data.clone(), &[2, 3]);
    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let s = t.sum_keepdim(0).unwrap().sqr().unwrap();
    let loss = s.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&v).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&g, &data, 1e-3, |d| {
        let x = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        sum_sqr_f64(&x.sum_keepdim(0).unwrap())
    });
}

#[test]
fn test_gc_sum_keepdim_dim1() {
    let data = vec![0.3, -0.5, 1.2, 0.8, 2.1, -1.0];
    let v = make_var(data.clone(), &[2, 3]);
    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let s = t.sum_keepdim(1).unwrap().sqr().unwrap();
    let loss = s.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&v).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&g, &data, 1e-3, |d| {
        let x = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        sum_sqr_f64(&x.sum_keepdim(1).unwrap())
    });
}

#[test]
fn test_gc_mean_keepdim_dim0() {
    let data = vec![1.0, -0.5, 2.3, -1.7];
    let v = make_var(data.clone(), &[4]);
    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let m = t.mean_keepdim(0).unwrap().sqr().unwrap();
    let loss = m.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&v).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&g, &data, 1e-3, |d| {
        let x = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        sum_sqr_f64(&x.mean_keepdim(0).unwrap())
    });
}

#[test]
fn test_gc_mean_keepdim_dim1_2d() {
    let data = vec![0.5, -1.2, 0.3, 2.1, -0.8, 0.7];
    let v = make_var(data.clone(), &[2, 3]);
    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let m = t.mean_keepdim(1).unwrap().sqr().unwrap();
    let loss = m.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&v).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&g, &data, 1e-3, |d| {
        let x = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        sum_sqr_f64(&x.mean_keepdim(1).unwrap())
    });
}

#[test]
fn test_gc_sum_keepdim_3d() {
    // 3D: reduce along middle dim
    let data: Vec<f32> = (1..=24).map(|i| i as f32 * 0.1).collect();
    let v = make_var(data.clone(), &[2, 3, 4]);
    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let s = t.sum_keepdim(1).unwrap().sqr().unwrap();
    let loss = s
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&v).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&g, &data, 1e-3, |d| {
        let x = DynTensor::from_vec(d, &[2, 3, 4], &cpu()).unwrap();
        sum_sqr_f64(&x.sum_keepdim(1).unwrap())
    });
}

#[test]
fn test_gc_mean_keepdim_3d() {
    let data: Vec<f32> = (1..=12).map(|i| i as f32 * 0.1 - 0.3).collect();
    let v = make_var(data.clone(), &[2, 2, 3]);
    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let m = t.mean_keepdim(2).unwrap().sqr().unwrap();
    let loss = m
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&v).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&g, &data, 1e-3, |d| {
        let x = DynTensor::from_vec(d, &[2, 2, 3], &cpu()).unwrap();
        sum_sqr_f64(&x.mean_keepdim(2).unwrap())
    });
}

// ════════════════════════════════════════════════════════════════════════
// 3. MATRIX OPS
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_gc_matmul_2d_both() {
    let a_data = vec![0.5, -0.3, 1.2, 0.8, -0.6, 0.4];
    let b_data = vec![1.0, -0.5, 0.3, 0.7, -0.2, 1.1];

    let a = make_var(a_data.clone(), &[2, 3]);
    let b = make_var(b_data.clone(), &[3, 2]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.matmul(&tb).unwrap().sqr().unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let ga = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let gb = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&ga, &a_data, 1e-3, |d| {
        let a = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[3, 2], &cpu()).unwrap();
        sum_sqr_f64(&a.matmul(&b).unwrap())
    });
    check_fd_default(&gb, &b_data, 1e-3, |d| {
        let a = DynTensor::from_vec(a_data.clone(), &[2, 3], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[3, 2], &cpu()).unwrap();
        sum_sqr_f64(&a.matmul(&b).unwrap())
    });
}

#[test]
fn test_gc_matmul_3d_broadcast_reduction() {
    // 3D x 2D: b is broadcast across batch, grad_b must reduce
    let a_data: Vec<f32> = vec![1.0, 0.5, -0.3, 0.8, -0.6, 0.4];
    let b_data: Vec<f32> = vec![0.5, -0.2, 1.0, 0.3, -0.7, 0.9];

    let a = make_var(a_data.clone(), &[2, 1, 3]);
    let b = make_var(b_data.clone(), &[3, 2]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.matmul(&tb).unwrap().sqr().unwrap();
    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let gb = grads.get(&b).unwrap();
    assert_eq!(gb.dims(), &[3, 2], "grad_b must be reduced to [3, 2]");
    let gb_flat = gb.to_flat_vec::<f32>().unwrap();

    check_fd_default(&gb_flat, &b_data, 1e-3, |d| {
        let a = DynTensor::from_vec(a_data.clone(), &[2, 1, 3], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[3, 2], &cpu()).unwrap();
        sum_sqr_f64(&a.matmul(&b).unwrap())
    });
}

#[test]
fn test_gc_transpose_matmul_chain() {
    // Transpose + matmul chain: tests gradient flow through transpose backward
    let data: Vec<f32> = vec![0.5, -0.3, 1.2, 0.8, -0.6, 0.4];
    let v = make_var(data.clone(), &[3, 2]);
    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    // y = x^T @ x (a positive semi-definite pattern)
    let xt = t.transpose(0, 1).unwrap();
    let y = xt.matmul(&t).unwrap().sqr().unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&v).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&g, &data, 1e-3, |d| {
        let x = DynTensor::from_vec(d, &[3, 2], &cpu()).unwrap();
        let xt = x.transpose(0, 1).unwrap();
        sum_sqr_f64(&xt.matmul(&x).unwrap())
    });
}

// ════════════════════════════════════════════════════════════════════════
// 4. NORMALIZATION OPS
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_gc_layer_norm_input() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gamma_data = vec![1.0, 0.5, 2.0];
    let beta_data = vec![0.1, -0.2, 0.3];
    let eps = 1e-5_f64;

    let x = make_var(x_data.clone(), &[2, 3]);
    let g = make_var(gamma_data.clone(), &[3]);
    let b = make_var(beta_data.clone(), &[3]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let tg = Arc::new(TrackedTensor::from_var(&g).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = tx.layer_norm(&tg, &tb, eps).unwrap().sqr().unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&gx, &x_data, 1e-3, |d| {
        let x = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let g = DynTensor::from_vec(gamma_data.clone(), &[3], &cpu()).unwrap();
        let b = DynTensor::from_vec(beta_data.clone(), &[3], &cpu()).unwrap();
        let mean = x.mean_keepdim(1).unwrap();
        let diff = x.sub(&mean).unwrap();
        let var = diff.sqr().unwrap().mean_keepdim(1).unwrap();
        let inv = var
            .add_scalar(eps)
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();
        let normed = diff.mul(&inv).unwrap();
        sum_sqr_f64(&normed.mul(&g).unwrap().add(&b).unwrap())
    });
}

#[test]
fn test_gc_layer_norm_weight() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gamma_data = vec![1.0, 0.5, 2.0];
    let beta_data = vec![0.1, -0.2, 0.3];
    let eps = 1e-5_f64;

    let x = make_var(x_data.clone(), &[2, 3]);
    let g = make_var(gamma_data.clone(), &[3]);
    let b = make_var(beta_data.clone(), &[3]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let tg = Arc::new(TrackedTensor::from_var(&g).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = tx.layer_norm(&tg, &tb, eps).unwrap().sqr().unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let gg = grads.get(&g).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&gg, &gamma_data, 1e-3, |d| {
        let x = DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap();
        let g = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        let b = DynTensor::from_vec(beta_data.clone(), &[3], &cpu()).unwrap();
        let mean = x.mean_keepdim(1).unwrap();
        let diff = x.sub(&mean).unwrap();
        let var = diff.sqr().unwrap().mean_keepdim(1).unwrap();
        let inv = var
            .add_scalar(eps)
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();
        let normed = diff.mul(&inv).unwrap();
        sum_sqr_f64(&normed.mul(&g).unwrap().add(&b).unwrap())
    });
}

#[test]
fn test_gc_instance_norm_input() {
    let x_data: Vec<f32> = vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0];
    let gamma_data = vec![1.0, 2.0];
    let beta_data = vec![0.0, 0.5];
    let eps = 1e-5_f64;

    let x = make_var(x_data.clone(), &[1, 2, 4]);
    let g = make_var(gamma_data.clone(), &[2]);
    let b = make_var(beta_data.clone(), &[2]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let tg = Arc::new(TrackedTensor::from_var(&g).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = tx.instance_norm(&tg, &tb, eps).unwrap().sqr().unwrap();
    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&gx, &x_data, 1e-3, |d| {
        let x = DynTensor::from_vec(d, &[1, 2, 4], &cpu()).unwrap();
        let g = DynTensor::from_vec(gamma_data.clone(), &[2], &cpu()).unwrap();
        let b = DynTensor::from_vec(beta_data.clone(), &[2], &cpu()).unwrap();
        let mean = x.mean_keepdim(2).unwrap();
        let diff = x.sub(&mean).unwrap();
        let var = diff.sqr().unwrap().mean_keepdim(2).unwrap();
        let inv = var
            .add_scalar(eps)
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();
        let normed = diff.mul(&inv).unwrap();
        let g_bc = g.reshape([1, 2, 1]).unwrap();
        let b_bc = b.reshape([1, 2, 1]).unwrap();
        sum_sqr_f64(&normed.mul(&g_bc).unwrap().add(&b_bc).unwrap())
    });
}

#[test]
fn test_gc_batch_norm_input() {
    let x_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let gamma_data = vec![1.0, 0.5];
    let beta_data = vec![0.0, 0.1];
    let eps = 1e-5_f64;

    let x = make_var(x_data.clone(), &[2, 2, 3]);
    let g = make_var(gamma_data.clone(), &[2]);
    let b = make_var(beta_data.clone(), &[2]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let tg = Arc::new(TrackedTensor::from_var(&g).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = tx.batch_norm(&tg, &tb, eps).unwrap().sqr().unwrap();
    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&gx, &x_data, 1e-3, |d| {
        let x = DynTensor::from_vec(d, &[2, 2, 3], &cpu()).unwrap();
        let g = DynTensor::from_vec(gamma_data.clone(), &[2], &cpu()).unwrap();
        let b = DynTensor::from_vec(beta_data.clone(), &[2], &cpu()).unwrap();
        let mean = x.mean_keepdim(2).unwrap().mean_keepdim(0).unwrap();
        let diff = x.sub(&mean).unwrap();
        let var = diff
            .sqr()
            .unwrap()
            .mean_keepdim(2)
            .unwrap()
            .mean_keepdim(0)
            .unwrap();
        let inv = var
            .add_scalar(eps)
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();
        let normed = diff.mul(&inv).unwrap();
        let g_bc = g.reshape([1, 2, 1]).unwrap();
        let b_bc = b.reshape([1, 2, 1]).unwrap();
        sum_sqr_f64(&normed.mul(&g_bc).unwrap().add(&b_bc).unwrap())
    });
}

#[test]
fn test_gc_rms_norm_input() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gamma_data = vec![1.0, 0.5, 2.0];
    let eps = 1e-5_f64;

    let x = make_var(x_data.clone(), &[2, 3]);
    let g = make_var(gamma_data.clone(), &[3]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let tg = Arc::new(TrackedTensor::from_var(&g).unwrap());
    let y = tx.rms_norm(&tg, eps).unwrap().sqr().unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&gx, &x_data, 1e-3, |d| {
        let x = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let g = DynTensor::from_vec(gamma_data.clone(), &[3], &cpu()).unwrap();
        let rms_sq = x.sqr().unwrap().mean_keepdim(1).unwrap();
        let inv = rms_sq
            .add_scalar(eps)
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();
        let normed = x.mul(&inv).unwrap();
        sum_sqr_f64(&normed.mul(&g).unwrap())
    });
}

#[test]
fn test_gc_group_norm_input() {
    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let gamma_data = vec![1.0, 0.5, 2.0, 1.5];
    let beta_data = vec![0.0, 0.1, -0.1, 0.0];
    let num_groups = 2;
    let eps = 1e-5_f64;

    let x = make_var(x_data.clone(), &[1, 4, 2]);
    let g = make_var(gamma_data.clone(), &[4]);
    let b = make_var(beta_data.clone(), &[4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let tg = Arc::new(TrackedTensor::from_var(&g).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = tx
        .group_norm(&tg, &tb, num_groups, eps)
        .unwrap()
        .sqr()
        .unwrap();
    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let gx = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&gx, &x_data, 1e-3, |d| {
        let x = DynTensor::from_vec(d, &[1, 4, 2], &cpu()).unwrap();
        let g = DynTensor::from_vec(gamma_data.clone(), &[4], &cpu()).unwrap();
        let b = DynTensor::from_vec(beta_data.clone(), &[4], &cpu()).unwrap();
        let xr = x.reshape([1, 2, 2, 2]).unwrap();
        let mean = xr.mean_keepdim(3).unwrap().mean_keepdim(2).unwrap();
        let diff = xr.sub(&mean).unwrap();
        let var = diff
            .sqr()
            .unwrap()
            .mean_keepdim(3)
            .unwrap()
            .mean_keepdim(2)
            .unwrap();
        let inv = var
            .add_scalar(eps)
            .unwrap()
            .sqrt()
            .unwrap()
            .recip()
            .unwrap();
        let normed = diff.mul(&inv).unwrap().reshape([1, 4, 2]).unwrap();
        let g_bc = g.reshape([1, 4, 1]).unwrap();
        let b_bc = b.reshape([1, 4, 1]).unwrap();
        sum_sqr_f64(&normed.mul(&g_bc).unwrap().add(&b_bc).unwrap())
    });
}

// ════════════════════════════════════════════════════════════════════════
// 5. ACTIVATION OPS
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_gc_relu() {
    // Avoid 0 for FD stability
    let data = vec![-2.0, -0.5, 0.1, 0.5, 2.0];
    let (_, g) = grad_sqr_sum_loss(|| {
        let v = make_var(data.clone(), &[5]);
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        (v, t.relu().unwrap())
    });
    check_fd_default(&g, &data, 1e-3, |d| {
        sum_sqr_f64(
            &DynTensor::from_vec(d, &[5], &cpu())
                .unwrap()
                .relu()
                .unwrap(),
        )
    });
}

#[test]
fn test_gc_gelu() {
    let data = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let (_, g) = grad_sqr_sum_loss(|| {
        let v = make_var(data.clone(), &[7]);
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        (v, t.gelu().unwrap())
    });
    check_fd_default(&g, &data, 1e-3, |d| {
        sum_sqr_f64(
            &DynTensor::from_vec(d, &[7], &cpu())
                .unwrap()
                .gelu()
                .unwrap(),
        )
    });
}

#[test]
fn test_gc_silu() {
    let data = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let (_, g) = grad_sqr_sum_loss(|| {
        let v = make_var(data.clone(), &[7]);
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        (v, t.silu().unwrap())
    });
    check_fd_default(&g, &data, 1e-3, |d| {
        sum_sqr_f64(
            &DynTensor::from_vec(d, &[7], &cpu())
                .unwrap()
                .silu()
                .unwrap(),
        )
    });
}

#[test]
fn test_gc_softmax_jacobian() {
    // Softmax with sum-of-squares loss exercises the full Jacobian
    let data = vec![1.0, 2.0, 0.5, -1.0, 0.3, 1.5];
    let v = make_var(data.clone(), &[2, 3]);
    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let sm = t.softmax(1).unwrap().sqr().unwrap();
    let loss = sm.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&v).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&g, &data, 1e-3, |d| {
        let x = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        sum_sqr_f64(&x.softmax(1).unwrap())
    });
}

#[test]
fn test_gc_tanh() {
    let data = vec![-1.5, -0.5, 0.0, 0.5, 1.5];
    let (_, g) = grad_sqr_sum_loss(|| {
        let v = make_var(data.clone(), &[5]);
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        (v, t.tanh().unwrap())
    });
    check_fd_default(&g, &data, 1e-3, |d| {
        sum_sqr_f64(
            &DynTensor::from_vec(d, &[5], &cpu())
                .unwrap()
                .tanh()
                .unwrap(),
        )
    });
}

#[test]
fn test_gc_sigmoid() {
    let data = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
    let (_, g) = grad_sqr_sum_loss(|| {
        let v = make_var(data.clone(), &[5]);
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        (v, t.sigmoid().unwrap())
    });
    check_fd_default(&g, &data, 1e-3, |d| {
        sum_sqr_f64(
            &DynTensor::from_vec(d, &[5], &cpu())
                .unwrap()
                .sigmoid()
                .unwrap(),
        )
    });
}

#[test]
fn test_gc_elu_mixed() {
    let data = vec![-1.0, 0.5, -0.3, 1.5];
    let v = make_var(data.clone(), &[4]);
    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let y = t.elu(1.5).unwrap().sqr().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&v).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&g, &data, 1e-3, |d| {
        sum_sqr_f64(
            &DynTensor::from_vec(d, &[4], &cpu())
                .unwrap()
                .elu(1.5)
                .unwrap(),
        )
    });
}

// ════════════════════════════════════════════════════════════════════════
// 6. SHAPE OPS
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_gc_reshape_gradient_passthrough() {
    let data = vec![0.5, -1.2, 0.3, 2.1, -0.8, 0.7];
    let (_, g) = grad_sqr_sum_loss(|| {
        let v = make_var(data.clone(), &[2, 3]);
        let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
        (v, t.reshape(&[3, 2]).unwrap())
    });
    check_fd_default(&g, &data, 1e-3, |d| {
        sum_sqr_f64(
            &DynTensor::from_vec(d, &[2, 3], &cpu())
                .unwrap()
                .reshape([3, 2])
                .unwrap(),
        )
    });
}

#[test]
fn test_gc_squeeze_unsqueeze_roundtrip() {
    let data = vec![0.5, -1.2, 0.3, 2.1];
    let v = make_var(data.clone(), &[1, 4]);
    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    // squeeze(0) -> [4], then unsqueeze(1) -> [4, 1]
    let y = t.squeeze(0).unwrap().unsqueeze(1).unwrap().sqr().unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&v).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&g, &data, 1e-3, |d| {
        let x = DynTensor::from_vec(d, &[1, 4], &cpu()).unwrap();
        let y = x.squeeze(0).unwrap().unsqueeze(1).unwrap();
        sum_sqr_f64(&y)
    });
}

#[test]
fn test_gc_broadcast_gradient_reduction() {
    let data = vec![0.7];
    let v = make_var(data.clone(), &[1, 1]);
    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let y = t.broadcast_as(&[3, 4]).unwrap().sqr().unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&v).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&g, &data, 1e-3, |d| {
        let x = DynTensor::from_vec(d, &[1, 1], &cpu()).unwrap();
        sum_sqr_f64(&x.broadcast_as([3, 4]).unwrap())
    });
}

#[test]
fn test_gc_narrow_zero_pad_backward() {
    let data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let v = make_var(data.clone(), &[4, 2]);
    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let n = t.narrow(0, 1, 2).unwrap().sqr().unwrap();
    let loss = n.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&v).unwrap().to_flat_vec::<f32>().unwrap();

    // Rows 0 and 3 should have zero gradient
    assert!(g[0].abs() < 1e-6, "row 0 grad should be 0");
    assert!(g[1].abs() < 1e-6, "row 0 grad should be 0");
    assert!(g[6].abs() < 1e-6, "row 3 grad should be 0");
    assert!(g[7].abs() < 1e-6, "row 3 grad should be 0");

    check_fd_default(&g, &data, 1e-3, |d| {
        let x = DynTensor::from_vec(d, &[4, 2], &cpu()).unwrap();
        sum_sqr_f64(&x.narrow(0, 1, 2).unwrap())
    });
}

// ════════════════════════════════════════════════════════════════════════
// 7. LOSS FUNCTIONS
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_gc_mse_loss_input() {
    let input_data = vec![1.0, 2.5, -0.5, 3.0];
    let target_data = vec![1.5, 2.0, 0.0, 2.5];

    let input_var = make_var(input_data.clone(), &[4]);
    let target_t = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(target_data.clone(), &[4], &cpu()).unwrap(),
    ));
    let ti = Arc::new(TrackedTensor::from_var(&input_var).unwrap());
    let loss = ti.mse_loss(&target_t).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&input_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&g, &input_data, 1e-3, |d| {
        let inp = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        let tgt = DynTensor::from_vec(target_data.clone(), &[4], &cpu()).unwrap();
        let diff = inp.sub(&tgt).unwrap();
        let mse = diff.sqr().unwrap().mean_keepdim(0).unwrap();
        sum_f64(&mse)
    });
}

#[test]
fn test_gc_l1_loss_input() {
    // Avoid exact equality between input and target (discontinuity)
    let input_data = vec![1.0, 2.5, -0.5, 3.0];
    let target_data = vec![1.5, 2.0, 0.2, 2.5];

    let input_var = make_var(input_data.clone(), &[4]);
    let target_t = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(target_data.clone(), &[4], &cpu()).unwrap(),
    ));
    let ti = Arc::new(TrackedTensor::from_var(&input_var).unwrap());
    let loss = ti.l1_loss(&target_t).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&input_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&g, &input_data, 1e-3, |d| {
        let inp = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        let tgt = DynTensor::from_vec(target_data.clone(), &[4], &cpu()).unwrap();
        let diff = inp.sub(&tgt).unwrap().abs().unwrap();
        let l1 = diff.mean_keepdim(0).unwrap();
        sum_f64(&l1)
    });
}

#[test]
fn test_gc_huber_loss_input() {
    // Mix of quadratic region (|diff| < delta) and linear region
    let input_data = vec![1.0, 5.0, -0.5, 3.0];
    let target_data = vec![1.2, 2.0, 0.0, 2.5];
    let delta = 1.0_f64;

    let input_var = make_var(input_data.clone(), &[4]);
    let target_t = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(target_data.clone(), &[4], &cpu()).unwrap(),
    ));
    let ti = Arc::new(TrackedTensor::from_var(&input_var).unwrap());
    let loss = ti.huber_loss(&target_t, delta).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&input_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&g, &input_data, 1e-3, |d| {
        // Manual Huber forward: 0.5*(diff^2)/n if |diff|<delta, delta*(|diff|-0.5*delta)/n otherwise
        let n = d.len() as f64;
        let mut total = 0.0_f64;
        for (i, &x) in d.iter().enumerate() {
            let diff = (f64::from(x) - f64::from(target_data[i])).abs();
            if diff < delta {
                total += 0.5 * diff * diff / n;
            } else {
                total += delta * (diff - 0.5 * delta) / n;
            }
        }
        total
    });
}

// ════════════════════════════════════════════════════════════════════════
// 8. CHAINED / COMPOSITE PATTERNS
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_gc_linear_pattern() {
    // y = x @ W + b (the core Linear layer pattern)
    let x_data = vec![0.5, -0.3, 1.2, 0.8, -0.6, 0.4];
    let w_data = vec![0.3, -0.1, 0.5, 0.2, -0.4, 0.6, 0.1, -0.3, 0.7];
    let b_data = vec![0.1, -0.2, 0.3];

    let w = make_var(w_data.clone(), &[3, 3]);
    let b = make_var(b_data.clone(), &[3]);
    let x_t = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap(),
    ));
    let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = x_t.matmul(&tw).unwrap().add(&tb).unwrap().sqr().unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let gw = grads.get(&w).unwrap().to_flat_vec::<f32>().unwrap();
    let gb = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&gw, &w_data, 1e-3, |d| {
        let x = DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap();
        let w = DynTensor::from_vec(d, &[3, 3], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[3], &cpu()).unwrap();
        sum_sqr_f64(&x.matmul(&w).unwrap().add(&b).unwrap())
    });
    check_fd_default(&gb, &b_data, 1e-3, |d| {
        let x = DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap();
        let w = DynTensor::from_vec(w_data.clone(), &[3, 3], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        sum_sqr_f64(&x.matmul(&w).unwrap().add(&b).unwrap())
    });
}

#[test]
fn test_gc_residual_connection() {
    // y = relu(x + linear(x)) -- residual pattern
    let x_data = vec![0.5, -0.3, 1.2, 0.8];
    let w_data = vec![
        0.3, -0.1, 0.5, 0.2, -0.4, 0.6, 0.1, -0.3, 0.7, -0.2, 0.4, -0.5, 0.8, -0.1, 0.3, 0.6,
    ];
    let x = make_var(x_data.clone(), &[1, 4]);
    let w = make_var(w_data.clone(), &[4, 4]);
    let tx = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let tw = Arc::new(TrackedTensor::from_var(&w).unwrap());
    // linear(x) = x @ w
    let linear_out = tx.matmul(&tw).unwrap();
    // residual: x + linear(x)
    let res = tx.add(&linear_out).unwrap();
    // relu + sqr loss
    let y = res.relu().unwrap().sqr().unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let gw = grads.get(&w).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&gw, &w_data, 1e-3, |d| {
        let x = DynTensor::from_vec(x_data.clone(), &[1, 4], &cpu()).unwrap();
        let w = DynTensor::from_vec(d, &[4, 4], &cpu()).unwrap();
        let lin = x.matmul(&w).unwrap();
        let res = x.add(&lin).unwrap();
        sum_sqr_f64(&res.relu().unwrap())
    });
}

#[test]
fn test_gc_cat_along_dim1() {
    let a_data = vec![0.5, -0.3, 1.2, 0.8];
    let b_data = vec![0.1, -0.7, 0.4, 0.9, -1.1, 0.6];

    let a = make_var(a_data.clone(), &[2, 2]);
    let b = make_var(b_data.clone(), &[2, 3]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let c = TrackedTensor::cat(&[&ta, &tb], 1).unwrap().sqr().unwrap();
    let loss = c.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let ga = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let gb = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&ga, &a_data, 1e-3, |d| {
        let a = DynTensor::from_vec(d, &[2, 2], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[2, 3], &cpu()).unwrap();
        sum_sqr_f64(&DynTensor::cat(&[&a, &b], 1).unwrap())
    });
    check_fd_default(&gb, &b_data, 1e-3, |d| {
        let a = DynTensor::from_vec(a_data.clone(), &[2, 2], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        sum_sqr_f64(&DynTensor::cat(&[&a, &b], 1).unwrap())
    });
}

#[test]
fn test_gc_maximum_with_fd() {
    // Avoid equal values where subgradient is non-unique
    let a_data = vec![1.0, 3.0, 2.0, 0.5];
    let b_data = vec![2.0, 1.0, 4.0, -0.5];

    let a = make_var(a_data.clone(), &[4]);
    let b = make_var(b_data.clone(), &[4]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.maximum(&tb).unwrap().sqr().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let ga = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let gb = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&ga, &a_data, 1e-3, |d| {
        let a = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[4], &cpu()).unwrap();
        sum_sqr_f64(&a.maximum(&b).unwrap())
    });
    check_fd_default(&gb, &b_data, 1e-3, |d| {
        let a = DynTensor::from_vec(a_data.clone(), &[4], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        sum_sqr_f64(&a.maximum(&b).unwrap())
    });
}

#[test]
fn test_gc_minimum_with_fd() {
    let a_data = vec![1.0, 3.0, 2.0, 0.5];
    let b_data = vec![2.0, 1.0, 4.0, -0.5];

    let a = make_var(a_data.clone(), &[4]);
    let b = make_var(b_data.clone(), &[4]);
    let ta = Arc::new(TrackedTensor::from_var(&a).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b).unwrap());
    let y = ta.minimum(&tb).unwrap().sqr().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let ga = grads.get(&a).unwrap().to_flat_vec::<f32>().unwrap();
    let gb = grads.get(&b).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&ga, &a_data, 1e-3, |d| {
        let a = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        let b = DynTensor::from_vec(b_data.clone(), &[4], &cpu()).unwrap();
        sum_sqr_f64(&a.minimum(&b).unwrap())
    });
    check_fd_default(&gb, &b_data, 1e-3, |d| {
        let a = DynTensor::from_vec(a_data.clone(), &[4], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        sum_sqr_f64(&a.minimum(&b).unwrap())
    });
}

#[test]
fn test_gc_log_softmax_with_sqr_loss() {
    let data = vec![1.0, 2.0, -0.5, 0.3, -1.0, 1.5];
    let v = make_var(data.clone(), &[2, 3]);
    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let ls = t.log_softmax(1).unwrap().sqr().unwrap();
    let loss = ls.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&v).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&g, &data, 1e-3, |d| {
        let x = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        sum_sqr_f64(&x.log_softmax(1).unwrap())
    });
}

#[test]
fn test_gc_stack_diamond() {
    // Diamond graph: same variable used twice in stack
    let data = vec![1.0, 2.0, 3.0];
    let v = make_var(data.clone(), &[3]);
    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let y = TrackedTensor::stack(&[Arc::clone(&t), Arc::clone(&t)], 0)
        .unwrap()
        .sqr()
        .unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&v).unwrap().to_flat_vec::<f32>().unwrap();

    // Both copies contribute: grad = 4*x (2*x from each copy, accumulated)
    for (i, &val) in data.iter().enumerate() {
        let expected = 4.0 * val;
        assert!(
            (g[i] - expected).abs() < 1e-4,
            "diamond grad[{i}]: got={}, expected={expected}",
            g[i]
        );
    }

    check_fd(&g, &data, 1e-3, 1e-2, &|d| {
        let x = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        let s = DynTensor::stack(&[&x, &x], 0).unwrap();
        sum_sqr_f64(&s)
    });
}

#[test]
fn test_gc_mul_scalar_negative() {
    let data = vec![1.0, -2.0, 0.5];
    let scalar = -3.0_f64;
    let v = make_var(data.clone(), &[3]);
    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let y = t.mul_scalar(scalar).unwrap().sqr().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&v).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_default(&g, &data, 1e-3, |d| {
        sum_sqr_f64(
            &DynTensor::from_vec(d, &[3], &cpu())
                .unwrap()
                .mul_scalar(scalar)
                .unwrap(),
        )
    });
}

#[test]
fn test_gc_clamp_mixed_regions() {
    // Mix of below-min, interior, above-max
    let data = vec![-2.0, 0.5, 1.5, -0.3, 3.0];
    let lo = -1.0_f64;
    let hi = 1.0_f64;
    let v = make_var(data.clone(), &[5]);
    let t = Arc::new(TrackedTensor::from_var(&v).unwrap());
    let y = t.clamp(lo, hi).unwrap().sqr().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&v).unwrap().to_flat_vec::<f32>().unwrap();

    // Outside clamp: zero grad
    assert!(
        g[0].abs() < 1e-6,
        "below min: grad should be 0, got {}",
        g[0]
    );
    assert!(
        g[2].abs() < 1e-6,
        "above max: grad should be 0, got {}",
        g[2]
    );
    assert!(
        g[4].abs() < 1e-6,
        "above max: grad should be 0, got {}",
        g[4]
    );

    check_fd_default(&g, &data, 1e-3, |d| {
        sum_sqr_f64(
            &DynTensor::from_vec(d, &[5], &cpu())
                .unwrap()
                .clamp(lo, hi)
                .unwrap(),
        )
    });
}
