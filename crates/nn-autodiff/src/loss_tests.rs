#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for MSE, L1, and Huber loss ops including finite-difference gradient verification.

use crate::{backward, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use std::sync::Arc;

// -- Forward value tests --

#[test]
fn test_mse_loss_value() {
    let input = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let target = DynTensor::from_vec(vec![1.5, 2.5, 3.5], &[3], &cpu()).unwrap();
    let i = Arc::new(TrackedTensor::from_tensor(input));
    let t = Arc::new(TrackedTensor::from_tensor(target));
    let loss = i.mse_loss(&t).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    // mean((1-1.5)^2 + (2-2.5)^2 + (3-3.5)^2) = mean(0.25+0.25+0.25) = 0.25
    assert!((val - 0.25).abs() < 1e-5, "MSE loss = {val}, expected 0.25");
}

#[test]
fn test_l1_loss_value() {
    let input = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let target = DynTensor::from_vec(vec![1.5, 2.5, 3.5], &[3], &cpu()).unwrap();
    let i = Arc::new(TrackedTensor::from_tensor(input));
    let t = Arc::new(TrackedTensor::from_tensor(target));
    let loss = i.l1_loss(&t).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    // mean(|1-1.5| + |2-2.5| + |3-3.5|) = mean(0.5+0.5+0.5) = 0.5
    assert!((val - 0.5).abs() < 1e-5, "L1 loss = {val}, expected 0.5");
}

#[test]
fn test_huber_loss_value_quadratic_region() {
    // All diffs |0.5| < delta=1.0, so fully in quadratic region
    let input = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let target = DynTensor::from_vec(vec![1.5, 2.5, 3.5], &[3], &cpu()).unwrap();
    let i = Arc::new(TrackedTensor::from_tensor(input));
    let t = Arc::new(TrackedTensor::from_tensor(target));
    let loss = i.huber_loss(&t, 1.0).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    // quadratic: 0.5 * x^2 / delta = 0.5 * 0.25 / 1.0 = 0.125 per elem
    // mean = 0.125
    assert!(
        (val - 0.125).abs() < 1e-5,
        "Huber loss = {val}, expected 0.125"
    );
}

#[test]
fn test_huber_loss_value_linear_region() {
    // diff = 2.0, |diff| > delta=0.5, so in linear region
    let input = DynTensor::from_vec(vec![3.0], &[1], &cpu()).unwrap();
    let target = DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap();
    let i = Arc::new(TrackedTensor::from_tensor(input));
    let t = Arc::new(TrackedTensor::from_tensor(target));
    let loss = i.huber_loss(&t, 0.5).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    // linear: |diff| - 0.5*delta = 2.0 - 0.25 = 1.75
    assert!(
        (val - 1.75).abs() < 1e-5,
        "Huber loss = {val}, expected 1.75"
    );
}

// -- Finite-difference gradient verification --

/// Finite-difference gradient check for a loss function.
fn finite_diff_check<F>(vals: &[f32], targets: &[f32], loss_fn: F, tol: f32)
where
    F: Fn(&Arc<TrackedTensor>, &Arc<TrackedTensor>) -> crate::error::Result<Arc<TrackedTensor>>,
{
    let eps = 1e-3_f64;
    let var = Var::new(DynTensor::from_vec(vals.to_vec(), &[vals.len()], &cpu()).unwrap());
    let target_t = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(targets.to_vec(), &[targets.len()], &cpu()).unwrap(),
    ));
    // Compute analytical gradient
    let x = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let loss = loss_fn(&x, &target_t).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads.get(&var).unwrap().to_flat_vec::<f32>().unwrap();

    // Finite difference for each element
    for i in 0..vals.len() {
        let mut vals_plus = vals.to_vec();
        vals_plus[i] += eps as f32;
        let v_plus = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vals_plus, &[vals.len()], &cpu()).unwrap(),
        ));
        let loss_plus = loss_fn(&v_plus, &target_t).unwrap();
        let l_plus = loss_plus.tensor().to_flat_vec::<f32>().unwrap()[0];

        let mut vals_minus = vals.to_vec();
        vals_minus[i] -= eps as f32;
        let v_minus = Arc::new(TrackedTensor::from_tensor(
            DynTensor::from_vec(vals_minus, &[vals.len()], &cpu()).unwrap(),
        ));
        let loss_minus = loss_fn(&v_minus, &target_t).unwrap();
        let l_minus = loss_minus.tensor().to_flat_vec::<f32>().unwrap()[0];

        let fd = (l_plus - l_minus) / (2.0 * eps as f32);
        let err = (grad[i] - fd).abs();
        assert!(
            err < tol,
            "FD check failed at i={i}: analytical={:.6}, fd={:.6}, err={:.6}",
            grad[i],
            fd,
            err
        );
    }
}

#[test]
fn test_mse_loss_backward_fd() {
    finite_diff_check(
        &[1.0, 2.0, 3.0, 4.0],
        &[1.5, 2.5, 3.5, 4.5],
        TrackedTensor::mse_loss,
        1e-3,
    );
}

#[test]
fn test_l1_loss_backward_fd() {
    // Avoid values near zero where L1 gradient is undefined
    finite_diff_check(
        &[1.0, 3.0, 5.0, 7.0],
        &[2.0, 4.0, 6.0, 8.0],
        TrackedTensor::l1_loss,
        1e-3,
    );
}

#[test]
fn test_huber_loss_backward_fd_quadratic() {
    // All diffs small (quadratic region, delta=2.0)
    finite_diff_check(
        &[1.0, 2.0, 3.0],
        &[1.5, 2.5, 3.5],
        |x, t| x.huber_loss(t, 2.0),
        1e-3,
    );
}

#[test]
fn test_huber_loss_backward_fd_linear() {
    // Large diffs (linear region, delta=0.1)
    finite_diff_check(&[5.0, 10.0], &[1.0, 2.0], |x, t| x.huber_loss(t, 0.1), 1e-3);
}

#[test]
fn test_mse_loss_zero_diff() {
    let input = DynTensor::from_vec(vec![2.0, 3.0], &[2], &cpu()).unwrap();
    let target = DynTensor::from_vec(vec![2.0, 3.0], &[2], &cpu()).unwrap();
    let i = Arc::new(TrackedTensor::from_tensor(input));
    let t = Arc::new(TrackedTensor::from_tensor(target));
    let loss = i.mse_loss(&t).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    assert!(
        val.abs() < 1e-7,
        "MSE of identical inputs should be ~0, got {val}"
    );
}

#[test]
fn test_mse_loss_2d() {
    let input = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let target = DynTensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[2, 2], &cpu()).unwrap();
    let i = Arc::new(TrackedTensor::from_tensor(input));
    let t = Arc::new(TrackedTensor::from_tensor(target));
    let loss = i.mse_loss(&t).unwrap();
    let val = loss.tensor().to_flat_vec::<f32>().unwrap()[0];
    // mean((0 + 1 + 4 + 9) / 4) = 3.5
    assert!((val - 3.5).abs() < 1e-5, "MSE 2D = {val}, expected 3.5");
}
