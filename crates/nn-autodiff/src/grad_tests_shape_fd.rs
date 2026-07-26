#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Finite-difference gradient tests for shape ops:
//! Narrow, Reshape, Transpose, Unsqueeze, Squeeze.
//!
//! Split from `grad_tests_shape_reduce_fd.rs` for 500-line compliance.
//! These ops are trivially correct inverses verified for completeness.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::grad::test_helpers::{check_fd_grad, sum_f64};
use crate::tracked::TrackedTensor;
use crate::var::Var;

// ── Narrow FD tests ─────────────────────────────────────────────────────

/// FD test for Narrow backward — middle slice along dim=0.
/// loss = sum(narrow(x, dim=0, start=1, len=2)^2) for x: [4, 2].
/// Backward must zero-pad before and after the slice.
#[test]
fn test_backward_narrow_dim0_fd() {
    let data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[4, 2], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let n = tx.narrow(0, 1, 2).unwrap();
    let sq = n.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    // Elements outside [1..3) should have zero gradient
    assert!(
        analytical[0].abs() < 1e-6,
        "grad at row 0 col 0 should be 0, got {}",
        analytical[0]
    );
    assert!(
        analytical[1].abs() < 1e-6,
        "grad at row 0 col 1 should be 0, got {}",
        analytical[1]
    );
    assert!(
        analytical[6].abs() < 1e-6,
        "grad at row 3 col 0 should be 0, got {}",
        analytical[6]
    );
    assert!(
        analytical[7].abs() < 1e-6,
        "grad at row 3 col 1 should be 0, got {}",
        analytical[7]
    );

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[4, 2], &cpu()).unwrap();
        let n = x.narrow(0, 1, 2).unwrap();
        sum_f64(&n.sqr().unwrap())
    });
}

/// FD test for Narrow backward — slice along dim=1.
/// loss = sum(narrow(x, dim=1, start=0, len=2)^2) for x: [2, 4].
#[test]
fn test_backward_narrow_dim1_fd() {
    let data = vec![1.0, -0.5, 0.3, 2.1, -1.2, 0.8, -0.7, 0.4];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[2, 4], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let n = tx.narrow(1, 0, 2).unwrap();
    let sq = n.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    // Elements in columns 2,3 should have zero gradient
    assert!(
        analytical[2].abs() < 1e-6,
        "grad at row 0 col 2 should be 0, got {}",
        analytical[2]
    );
    assert!(
        analytical[3].abs() < 1e-6,
        "grad at row 0 col 3 should be 0, got {}",
        analytical[3]
    );

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[2, 4], &cpu()).unwrap();
        let n = x.narrow(1, 0, 2).unwrap();
        sum_f64(&n.sqr().unwrap())
    });
}

// ── Reshape FD tests ──────────────────────────────────────────────────

/// FD test for Reshape backward — [2, 3] to [3, 2].
/// loss = sum(reshape(x, [3,2])^2). Backward reshapes gradient back to [2,3].
#[test]
fn test_backward_reshape_fd() {
    let data = vec![0.5, -1.2, 0.3, 2.1, -0.8, 0.7];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[2, 3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.reshape(&[3, 2]).unwrap();
    let sq = y.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let y = x.reshape([3, 2]).unwrap();
        sum_f64(&y.sqr().unwrap())
    });
}

/// FD test for Reshape backward — flatten [2, 2, 2] to [8].
#[test]
fn test_backward_reshape_flatten_fd() {
    let data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[2, 2, 2], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.reshape(&[8]).unwrap();
    let sq = y.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[2, 2, 2], &cpu()).unwrap();
        let y = x.reshape([8]).unwrap();
        sum_f64(&y.sqr().unwrap())
    });
}

// ── Transpose FD tests ────────────────────────────────────────────────

/// FD test for Transpose backward — swap dims 0 and 1 of [2, 3].
/// loss = sum(transpose(x, 0, 1)^2). Backward transposes gradient back.
#[test]
fn test_backward_transpose_fd() {
    let data = vec![0.5, -1.2, 0.3, 2.1, -0.8, 0.7];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[2, 3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.transpose(0, 1).unwrap();
    let sq = y.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let y = x.transpose(0, 1).unwrap();
        sum_f64(&y.sqr().unwrap())
    });
}

/// FD test for Transpose backward — 3D tensor, swap dims 1 and 2.
#[test]
fn test_backward_transpose_3d_fd() {
    let data: Vec<f32> = (1..=24).map(|i| i as f32 * 0.1).collect();
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[2, 3, 4], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.transpose(1, 2).unwrap();
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
        let y = x.transpose(1, 2).unwrap();
        sum_f64(&y.sqr().unwrap())
    });
}

// ── Unsqueeze FD tests ────────────────────────────────────────────────

/// FD test for Unsqueeze backward — add dim at axis 0 to [3].
/// loss = sum(unsqueeze(x, 0)^2). Backward squeezes gradient back.
#[test]
fn test_backward_unsqueeze_fd() {
    let data = vec![0.5, -1.2, 0.3];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.unsqueeze(0).unwrap();
    let sq = y.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        let y = x.unsqueeze(0).unwrap();
        sum_f64(&y.sqr().unwrap())
    });
}

/// FD test for Unsqueeze backward — add dim at axis 1 to [2, 3].
#[test]
fn test_backward_unsqueeze_middle_fd() {
    let data = vec![0.5, -1.2, 0.3, 2.1, -0.8, 0.7];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[2, 3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.unsqueeze(1).unwrap();
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
        let x = DynTensor::from_vec(d, &[2, 3], &cpu()).unwrap();
        let y = x.unsqueeze(1).unwrap();
        sum_f64(&y.sqr().unwrap())
    });
}

// ── Squeeze FD tests ──────────────────────────────────────────────────

/// FD test for Squeeze backward — remove dim 0 from [1, 4].
/// loss = sum(squeeze(x, 0)^2). Backward unsqueezes gradient back.
#[test]
fn test_backward_squeeze_fd() {
    let data = vec![0.5, -1.2, 0.3, 2.1];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[1, 4], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.squeeze(0).unwrap();
    let sq = y.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[1, 4], &cpu()).unwrap();
        let y = x.squeeze(0).unwrap();
        sum_f64(&y.sqr().unwrap())
    });
}

/// FD test for Squeeze backward — remove middle dim from [2, 1, 3].
#[test]
fn test_backward_squeeze_middle_fd() {
    let data = vec![0.5, -1.2, 0.3, 2.1, -0.8, 0.7];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[2, 1, 3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let y = tx.squeeze(1).unwrap();
    let sq = y.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[2, 1, 3], &cpu()).unwrap();
        let y = x.squeeze(1).unwrap();
        sum_f64(&y.sqr().unwrap())
    });
}

// ── Unfold FD tests ─────────────────────────────────────────────────────

/// FD test for Unfold backward — 1D overlapping windows.
/// [8].unfold(0, 4, 2) -> [3, 4]. Overlapping positions (2-5) receive
/// gradient contributions from multiple windows.
/// loss = sum(unfold(x)^2) to ensure gradient depends on input values.
#[test]
fn test_backward_unfold_1d_overlapping_fd() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[8], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let u = tx.unfold(0, 4, 2).unwrap(); // [3, 4]
    let sq = u.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[8], &cpu()).unwrap();
        let u = x.unfold(0, 4, 2).unwrap();
        sum_f64(&u.sqr().unwrap())
    });
}

/// FD test for Unfold backward — non-overlapping windows (step == size).
/// [8].unfold(0, 4, 4) -> [2, 4]. Each input position contributes to
/// exactly one window.
#[test]
fn test_backward_unfold_1d_non_overlapping_fd() {
    let data: Vec<f32> = vec![0.5, -1.2, 0.3, 2.1, -0.8, 0.7, 1.5, -0.4];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[8], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let u = tx.unfold(0, 4, 4).unwrap(); // [2, 4]
    let sq = u.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[8], &cpu()).unwrap();
        let u = x.unfold(0, 4, 4).unwrap();
        sum_f64(&u.sqr().unwrap())
    });
}

/// FD test for Unfold backward — 3D with unfold on dim=0 (non-last dim).
/// [4, 2, 3].unfold(0, 2, 1) -> [3, 2, 3, 2]. This tests the critical case
/// where dim != last dim, requiring transpose in the backward rule to align
/// the trailing `size` dimension back to position `dim`.
#[test]
fn test_backward_unfold_3d_dim0_fd() {
    let data: Vec<f32> = (1..=24).map(|i| i as f32 * 0.1).collect();
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[4, 2, 3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let u = tx.unfold(0, 2, 1).unwrap(); // [3, 2, 3, 2]
    let sq = u.sqr().unwrap();
    let loss = sq
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap()
        .sum_keepdim(3)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &data, eps, |d| {
        let x = DynTensor::from_vec(d, &[4, 2, 3], &cpu()).unwrap();
        let u = x.unfold(0, 2, 1).unwrap();
        sum_f64(&u.sqr().unwrap())
    });
}

/// FD test for Unfold backward — 2D with unfold on dim=1.
/// [2, 6].unfold(1, 3, 2) -> [2, 2, 3]. Tests non-zero dim.
#[test]
fn test_backward_unfold_2d_dim1_fd() {
    let data: Vec<f32> = (1..=12).map(|i| i as f32 * 0.1).collect();
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(data.clone(), &[2, 6], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let u = tx.unfold(1, 3, 2).unwrap(); // [2, 2, 3]
    let sq = u.sqr().unwrap();
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
        let x = DynTensor::from_vec(d, &[2, 6], &cpu()).unwrap();
        let u = x.unfold(1, 3, 2).unwrap();
        sum_f64(&u.sqr().unwrap())
    });
}
