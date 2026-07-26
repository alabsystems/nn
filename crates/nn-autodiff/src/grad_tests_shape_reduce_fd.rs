#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Finite-difference gradient tests for reduce and clamp ops:
//! Clamp, Broadcast, MeanKeepDim.
//!
//! Shape ops (Narrow, Reshape, Transpose, Unsqueeze, Squeeze) are in
//! `grad_tests_shape_fd.rs`, split for 500-line compliance.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::grad::test_helpers::{check_fd_grad, sum_f64};
use crate::tracked::TrackedTensor;
use crate::var::Var;

// ── Clamp FD tests ──────────────────────────────────────────────────────

/// FD test for Clamp backward — interior values (where gradient should pass through).
/// loss = sum(clamp(x, -1, 1)^2), tested with values in the interior of [-1, 1].
#[test]
fn test_backward_clamp_interior_fd() {
    let data = vec![0.3, -0.5, 0.8, -0.2];
    let lo = -1.0_f64;
    let hi = 1.0_f64;
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[4], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.clamp(lo, hi).unwrap();
    let sq = y.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        let y = x.clamp(lo, hi).unwrap();
        sum_f64(&y.sqr().unwrap())
    });
}

/// FD test for Clamp backward — values outside clamp range (gradient should be zero).
/// Includes values both below min and above max.
#[test]
fn test_backward_clamp_boundary_fd() {
    // Mix of interior, below-min, and above-max values
    let data = vec![-2.0, 0.5, 1.5, -0.3, 3.0, -1.5];
    let lo = -1.0_f64;
    let hi = 1.0_f64;
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[6], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.clamp(lo, hi).unwrap();
    let sq = y.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    // Elements outside clamp range should have zero gradient
    assert!(
        analytical[0].abs() < 1e-6,
        "grad at x=-2.0 (below min) should be 0, got {}",
        analytical[0]
    );
    assert!(
        analytical[2].abs() < 1e-6,
        "grad at x=1.5 (above max) should be 0, got {}",
        analytical[2]
    );
    assert!(
        analytical[4].abs() < 1e-6,
        "grad at x=3.0 (above max) should be 0, got {}",
        analytical[4]
    );
    assert!(
        analytical[5].abs() < 1e-6,
        "grad at x=-1.5 (below min) should be 0, got {}",
        analytical[5]
    );

    // Interior elements should have non-zero gradient — verify with FD
    // Only check interior elements for FD (boundary discontinuity makes FD unreliable
    // right at the boundary)
    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[6], &cpu()).unwrap();
        let y = x.clamp(lo, hi).unwrap();
        sum_f64(&y.sqr().unwrap())
    });
}

/// Test for Clamp backward — values exactly at lo and hi boundaries.
/// With ge/le (PyTorch convention), gradient flows at the boundaries.
/// FD is unreliable at discontinuities, so we verify analytical values directly.
#[test]
fn test_backward_clamp_exact_boundary() {
    // x = [-1.0, 0.0, 1.0] with clamp(-1, 1): all three are at or inside bounds.
    let data = vec![-1.0, 0.0, 1.0];
    let lo = -1.0_f64;
    let hi = 1.0_f64;

    let x_var = Var::new(DynTensor::from_vec(data, &[3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.clamp(lo, hi).unwrap();
    let sq = y.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    // At x=lo=-1: clamp'=1 (ge/le), loss=x^2, so grad=2*x=-2.0
    // At x=0:     clamp'=1 (interior), loss=x^2, so grad=2*x=0.0
    // At x=hi=1:  clamp'=1 (ge/le), loss=x^2, so grad=2*x=2.0
    // Matches PyTorch: clamp'(x) = 1 when lo <= x <= hi.
    assert_eq!(analytical, vec![-2.0, 0.0, 2.0]);
}

// ── Broadcast FD tests ──────────────────────────────────────────────────

/// FD test for Broadcast backward — [1,1] to [2,3].
/// loss = sum(broadcast(x, [2,3])^2). Backward must reduce the gradient from
/// shape [2,3] back to [1,1].
#[test]
fn test_backward_broadcast_scalar_to_2d_fd() {
    let data = vec![0.7];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[1, 1], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.broadcast_as(&[2, 3]).unwrap();
    let sq = y.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[1, 1], &cpu()).unwrap();
        let y = x.broadcast_as([2, 3]).unwrap();
        sum_f64(&y.sqr().unwrap())
    });
}

/// FD test for Broadcast backward — 1D [3] to 2D [2,3].
/// Backward must reduce along axis 0 (the broadcast axis).
#[test]
fn test_backward_broadcast_1d_to_2d_fd() {
    let data = vec![0.5, -1.2, 0.3];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[1, 3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.broadcast_as(&[2, 3]).unwrap();
    let sq = y.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[1, 3], &cpu()).unwrap();
        let y = x.broadcast_as([2, 3]).unwrap();
        sum_f64(&y.sqr().unwrap())
    });
}

// ── MeanKeepDim FD tests ────────────────────────────────────────────────

/// FD test for MeanKeepDim backward — verify 1/N scaling of gradient.
/// loss = sum(mean(x, dim=1)^2) for x: [2, 3].
#[test]
fn test_backward_mean_keepdim_dim1_fd() {
    let data = vec![0.5, -1.2, 0.3, 2.1, -0.8, 0.7];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[2, 3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let m = tx.mean_keepdim(1).unwrap();
    let sq = m.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let m = x.mean_keepdim(1).unwrap();
        sum_f64(&m.sqr().unwrap())
    });
}

/// FD test for MeanKeepDim backward along dim=0 — 1D input.
/// loss = sum(mean(x, dim=0)^2) for x: [4].
#[test]
fn test_backward_mean_keepdim_dim0_fd() {
    let data = vec![1.0, -0.5, 2.3, -1.7];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[4], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let m = tx.mean_keepdim(0).unwrap();
    let sq = m.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        let m = x.mean_keepdim(0).unwrap();
        sum_f64(&m.sqr().unwrap())
    });
}
