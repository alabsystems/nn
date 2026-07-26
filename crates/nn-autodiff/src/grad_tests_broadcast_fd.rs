#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Finite-difference backward tests for broadcast gradient reduction (#1453).
//!
//! Extracted from `grad_tests_broadcast.rs` to keep the parent under 500 lines.
//! These tests verify the gradient formula is correct for arbitrary inputs
//! by comparing against numerical differentiation via `check_fd_grad`.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::grad::test_helpers::{check_fd_grad, sum_f64};
use crate::tracked::TrackedTensor;
use crate::var::Var;

/// Finite-difference test for Add broadcast backward.
/// Perturbs each element of bias and verifies the reduced gradient matches.
#[test]
fn test_backward_add_broadcast_fd() {
    let x_data = vec![0.3, -1.2, 0.8, 2.1, -0.5, 0.7, 1.4, -0.9];
    let b_data = vec![0.1, -0.2, 0.3, 0.4];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(x_data, &[2, 4], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(b_data.clone(), &[1, 4], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = tx.add(&tb).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&b_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &b_data, eps, |d| {
        let x = DynTensor::from_vec(
            vec![0.3, -1.2, 0.8, 2.1, -0.5, 0.7, 1.4, -0.9],
            &[2, 4],
            &cpu(),
        )
        .unwrap();
        let b = DynTensor::from_vec(d, &[1, 4], &cpu()).unwrap();
        sum_f64(&x.add(&b).unwrap())
    });
}

/// Finite-difference test for Mul broadcast backward.
/// scale = [1, 3] multiplied with x = [2, 3]: grad(scale) should reduce correctly.
#[test]
fn test_backward_mul_broadcast_fd() {
    let x_data = vec![0.3, -1.2, 0.8, 2.1, -0.5, 0.7];
    let s_data = vec![1.5, -0.3, 0.8];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap());
    let s_var = Var::new(DynTensor::from_vec(s_data.clone(), &[1, 3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let ts = Arc::new(TrackedTensor::from_var(&s_var).unwrap());
    let y = tx.mul(&ts).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&s_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &s_data, eps, |d| {
        let x = DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap();
        let s = DynTensor::from_vec(d, &[1, 3], &cpu()).unwrap();
        sum_f64(&x.mul(&s).unwrap())
    });
}

/// Finite-difference test for Div broadcast backward.
/// divisor = [1, 2] with x = [2, 2]: grad(divisor) involves -x/d^2 summed over batch.
#[test]
fn test_backward_div_broadcast_fd() {
    let x_data = vec![2.5, 4.0, 6.0, 8.5];
    let d_data = vec![1.5, 2.5];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 2], &cpu()).unwrap());
    let d_var = Var::new(DynTensor::from_vec(d_data.clone(), &[1, 2], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let td = Arc::new(TrackedTensor::from_var(&d_var).unwrap());
    let y = tx.div(&td).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&d_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &d_data, eps, |d| {
        let x = DynTensor::from_vec(x_data.clone(), &[2, 2], &cpu()).unwrap();
        let dv = DynTensor::from_vec(d, &[1, 2], &cpu()).unwrap();
        sum_f64(&x.div(&dv).unwrap())
    });
}

/// Finite-difference test for Sub broadcast backward.
/// offset = [1, 3] subtracted from x = [2, 3]: grad(offset) should reduce correctly.
#[test]
fn test_backward_sub_broadcast_fd() {
    let x_data = vec![0.3, -1.2, 0.8, 2.1, -0.5, 0.7];
    let o_data = vec![0.1, -0.2, 0.3];
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap());
    let o_var = Var::new(DynTensor::from_vec(o_data.clone(), &[1, 3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let to = Arc::new(TrackedTensor::from_var(&o_var).unwrap());
    let y = tx.sub(&to).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let analytical = grads.get(&o_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &o_data, eps, |d| {
        let x = DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap();
        let o = DynTensor::from_vec(d, &[1, 3], &cpu()).unwrap();
        sum_f64(&x.sub(&o).unwrap())
    });
}

/// Finite-difference test for MatMul broadcast backward.
/// a = [2, 3, 4], b = [4, 2]: b is broadcast across batch dim 0.
/// grad(b) should be reduced from [2, 4, 2] to [4, 2].
#[test]
fn test_backward_matmul_broadcast_fd() {
    let a_data: Vec<f32> = (1..=24).map(|v| v as f32 * 0.1).collect();
    let b_data: Vec<f32> = (1..=8).map(|v| v as f32 * 0.15 - 0.3).collect();
    let eps = 1e-3_f32;

    let a_var = Var::new(DynTensor::from_vec(a_data.clone(), &[2, 3, 4], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(b_data.clone(), &[4, 2], &cpu()).unwrap());
    let ta = Arc::new(TrackedTensor::from_var(&a_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = ta.matmul(&tb).unwrap();
    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let analytical_b = grads.get(&b_var).unwrap().to_flat_vec::<f32>().unwrap();

    // Verify shape was correctly reduced
    assert_eq!(
        grads.get(&b_var).unwrap().dims(),
        &[4, 2],
        "grad_b shape should be [4, 2] after reduce_to_shape"
    );

    check_fd_grad(&analytical_b, &b_data, eps, |d| {
        let a = DynTensor::from_vec(a_data.clone(), &[2, 3, 4], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[4, 2], &cpu()).unwrap();
        sum_f64(&a.matmul(&b).unwrap())
    });
}

/// Finite-difference test for rank-different Add broadcast backward.
/// TrainableLinear creates bias with shape [N] (1D), not [1, N] (2D).
/// When x = [B, N] and bias = [N], the gradient must reduce from [B, N] to [N]
/// via `reduce_to_shape` collapsing 1 extra leading dimension.
/// This tests the exact pattern produced by `TrainableLinear::new()` after
/// commit 4130d5a3 changed bias shape from [1, out_features] to [out_features].
#[test]
fn test_backward_add_broadcast_1d_bias_fd() {
    let x_data = vec![0.3, -1.2, 0.8, 2.1, -0.5, 0.7, 1.4, -0.9];
    let b_data = vec![0.1, -0.2, 0.3, 0.4]; // [4] — true 1D, not [1, 4]
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 4], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(b_data.clone(), &[4], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = tx.add(&tb).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let gb = grads.get(&b_var).unwrap();

    // Gradient must be [4] (1D), not [1, 4] or [2, 4]
    assert_eq!(
        gb.dims(),
        &[4],
        "grad_bias shape mismatch: expected [4] (1D) but got {:?} — \
         reduce_to_shape fails for rank-different broadcast",
        gb.dims()
    );

    let analytical = gb.to_flat_vec::<f32>().unwrap();

    // Verify values: grad(bias[j]) = sum_i(1) = 2 for all j
    for (i, &g) in analytical.iter().enumerate() {
        assert!(
            (g - 2.0).abs() < 1e-5,
            "grad_bias[{i}]: expected 2.0, got {g}"
        );
    }

    // FD check against numerical gradient
    check_fd_grad(&analytical, &b_data, eps, |d| {
        let x = DynTensor::from_vec(x_data.clone(), &[2, 4], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        sum_f64(&x.add(&b).unwrap())
    });
}

/// Finite-difference test for rank-different Mul broadcast backward.
/// Tests [B, N] * [N] where scale is 1D — the TrainableLinear bias pattern
/// applied to element-wise multiplication.
#[test]
fn test_backward_mul_broadcast_1d_scale_fd() {
    let x_data = vec![0.3, -1.2, 0.8, 2.1, -0.5, 0.7];
    let s_data = vec![1.5, -0.3, 0.8]; // [3] — true 1D
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap());
    let s_var = Var::new(DynTensor::from_vec(s_data.clone(), &[3], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let ts = Arc::new(TrackedTensor::from_var(&s_var).unwrap());
    let y = tx.mul(&ts).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();
    let gs = grads.get(&s_var).unwrap();

    assert_eq!(
        gs.dims(),
        &[3],
        "grad_scale shape: expected [3] (1D), got {:?}",
        gs.dims()
    );

    let analytical = gs.to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical, &s_data, eps, |d| {
        let x = DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap();
        let s = DynTensor::from_vec(d, &[3], &cpu()).unwrap();
        sum_f64(&x.mul(&s).unwrap())
    });
}

/// Finite-difference test for 3D input with 1D bias: [B, S, N] + [N].
/// Covers the transformer case where Linear forward is applied to 3D tensors
/// and bias must reduce across 2 extra leading dimensions.
#[test]
fn test_backward_add_broadcast_3d_input_1d_bias_fd() {
    let x_data: Vec<f32> = (1..=24).map(|v| v as f32 * 0.1 - 0.5).collect();
    let b_data = vec![0.1, -0.2, 0.3, 0.4]; // [4] — true 1D
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 3, 4], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(b_data.clone(), &[4], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = tx.add(&tb).unwrap();
    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let gb = grads.get(&b_var).unwrap();

    // Gradient must be [4] after reducing 2 extra leading dims
    assert_eq!(
        gb.dims(),
        &[4],
        "grad_bias shape: expected [4] (1D), got {:?} — \
         reduce_to_shape fails for 3D→1D rank difference",
        gb.dims()
    );

    let analytical = gb.to_flat_vec::<f32>().unwrap();

    // Each bias element receives gradient 1.0 from all B*S=6 positions
    for (i, &g) in analytical.iter().enumerate() {
        assert!(
            (g - 6.0).abs() < 1e-4,
            "grad_bias[{i}]: expected 6.0, got {g}"
        );
    }

    check_fd_grad(&analytical, &b_data, eps, |d| {
        let x = DynTensor::from_vec(x_data.clone(), &[2, 3, 4], &cpu()).unwrap();
        let b = DynTensor::from_vec(d, &[4], &cpu()).unwrap();
        sum_f64(&x.add(&b).unwrap())
    });
}
