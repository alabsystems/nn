#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Gradient correctness tests for Sin, Cos, Recip, Powf, Clamp, Permute.
//!
//! Uses finite-difference verification to prove backward rules match
//! the numerical derivative of each operation.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::backward;
use crate::tracked::TrackedTensor;
use crate::var::Var;

use super::test_helpers::{check_fd_grad, sum_f64, vec_var};

// ── Sin ──────────────────────────────────────────────────────────────

#[test]
fn test_backward_sin() {
    let data = vec![0.5, 1.0, -0.3];
    let x = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.sin().unwrap();
    // scalar loss = sum(sin(x))
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&g, &data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        sum_f64(&t.sin().unwrap())
    });
}

// ── Cos ──────────────────────────────────────────────────────────────

#[test]
fn test_backward_cos() {
    let data = vec![0.5, 1.0, -0.3];
    let x = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.cos().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&g, &data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        sum_f64(&t.cos().unwrap())
    });
}

// ── Recip ────────────────────────────────────────────────────────────

#[test]
fn test_backward_recip() {
    let data = vec![2.0, 0.5, 4.0];
    let x = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.recip().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&g, &data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        sum_f64(&t.recip().unwrap())
    });
}

// ── Powf ─────────────────────────────────────────────────────────────

#[test]
fn test_backward_powf_square() {
    // powf(x, 2.0) should match sqr
    let data = vec![1.0, 2.0, 3.0];
    let x = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.powf(2.0).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&g, &data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        sum_f64(&t.powf(2.0).unwrap())
    });
}

#[test]
fn test_backward_powf_fractional() {
    // powf(x, 0.5) = sqrt(x)
    let data = vec![1.0, 4.0, 9.0];
    let x = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.powf(0.5).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&g, &data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        sum_f64(&t.powf(0.5).unwrap())
    });
}

#[test]
fn test_backward_powf_negative() {
    // powf(x, -1.0) = 1/x, d/dx = -1/x^2
    // Verify negative exponents produce correct gradients via FD.
    let data = vec![1.0, 2.0, 4.0];
    let x = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.powf(-1.0).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&g, &data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        sum_f64(&t.powf(-1.0).unwrap())
    });
}

#[test]
fn test_backward_powf_negative_fractional() {
    // powf(x, -0.5) = 1/sqrt(x), d/dx = -0.5 * x^(-1.5)
    // Edge case: fractional negative exponent.
    let data = vec![1.0, 4.0, 9.0];
    let x = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.powf(-0.5).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&g, &data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        sum_f64(&t.powf(-0.5).unwrap())
    });
}

// ── Clamp ────────────────────────────────────────────────────────────

#[test]
fn test_backward_clamp_pass_through() {
    // Values within [lo, hi] → gradient = 1
    let data = vec![0.5, 1.0, 1.5];
    let x = vec_var(data);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.clamp(0.0, 2.0).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // All within range, gradient should be ~1.0
    for (i, &gi) in g.iter().enumerate() {
        assert!(
            (gi - 1.0).abs() < 1e-5,
            "grad[{i}]={gi}, expected 1.0 (within clamp range)"
        );
    }
}

#[test]
fn test_backward_clamp_clamped() {
    // Values outside [lo, hi] → gradient = 0
    let data = vec![-1.0, 0.5, 3.0];
    let x = vec_var(data);
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.clamp(0.0, 2.0).unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // -1.0 < 0.0 → clamped, grad = 0
    assert!(g[0].abs() < 1e-5, "clamped below: grad[0]={}", g[0]);
    // 0.5 in range → grad = 1
    assert!((g[1] - 1.0).abs() < 1e-5, "in range: grad[1]={}", g[1]);
    // 3.0 > 2.0 → clamped, grad = 0
    assert!(g[2].abs() < 1e-5, "clamped above: grad[2]={}", g[2]);
}

// ── Permute ──────────────────────────────────────────────────────────

#[test]
fn test_backward_permute() {
    // 2D tensor [2,3] → permute [1,0] → [3,2], then sum → scalar loss
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = Var::new(DynTensor::from_vec(data, &[2, 3], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.permute(&[1, 0]).unwrap();
    assert_eq!(y.dims(), &[3, 2]);
    // Sum all → scalar
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // d(sum(permute(x)))/dx = 1 everywhere (permute is a view op)
    for (i, &gi) in g.iter().enumerate() {
        assert!(
            (gi - 1.0).abs() < 1e-5,
            "grad[{i}]={gi}, expected 1.0 for permute"
        );
    }
}

#[test]
fn test_backward_permute_3d() {
    // [2,3,4] → permute [2,0,1] → [4,2,3]
    let n = 24;
    let data: Vec<f32> = (1..=n).map(|i| i as f32).collect();
    let x = Var::new(DynTensor::from_vec(data, &[2, 3, 4], &cpu()).unwrap());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.permute(&[2, 0, 1]).unwrap();
    assert_eq!(y.dims(), &[4, 2, 3]);
    // weighted sum using mul_scalar then sum
    let y2 = y.mul_scalar(2.0).unwrap();
    let loss = y2
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();
    // d(2*sum(permute(x)))/dx = 2 everywhere
    for (i, &gi) in g.iter().enumerate() {
        assert!(
            (gi - 2.0).abs() < 1e-5,
            "grad[{i}]={gi}, expected 2.0 for scaled permute"
        );
    }
}

// ── Op Debug format tests ────────────────────────────────────────────

#[test]
fn test_op_debug_new_variants() {
    let x = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap(),
    ));
    use crate::op::Op;
    assert_eq!(format!("{:?}", Op::Sin(Arc::clone(&x))), "Sin");
    assert_eq!(format!("{:?}", Op::Cos(Arc::clone(&x))), "Cos");
    assert_eq!(format!("{:?}", Op::Recip(Arc::clone(&x))), "Recip");
    assert_eq!(format!("{:?}", Op::Powf(Arc::clone(&x), 3.0)), "Powf(3)");
    assert_eq!(
        format!("{:?}", Op::Clamp(Arc::clone(&x), -1.0, 1.0)),
        "Clamp(-1, 1)"
    );
    assert_eq!(
        format!("{:?}", Op::Permute(Arc::clone(&x), vec![1, 0])),
        "Permute([1, 0])"
    );
}

// ── Chained operations ───────────────────────────────────────────────

#[test]
fn test_backward_sin_cos_chain() {
    // f(x) = cos(sin(x)), d/dx = -sin(sin(x)) * cos(x)
    let data = vec![0.5, 1.0];
    let x = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.sin().unwrap().cos().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&g, &data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[2], &cpu()).unwrap();
        sum_f64(&t.sin().unwrap().cos().unwrap())
    });
}

#[test]
fn test_backward_recip_chain() {
    // f(x) = 1/exp(x), d/dx = -exp(x)/exp(x)^2 = -1/exp(x)
    let data = vec![0.5, 1.0, 2.0];
    let x = vec_var(data.clone());
    let t = Arc::new(TrackedTensor::from_var(&x).unwrap());
    let y = t.exp().unwrap().recip().unwrap();
    let loss = y.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let g = grads.get(&x).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&g, &data, 1e-3, |d| {
        let t = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        sum_f64(&t.exp().unwrap().recip().unwrap())
    });
}
