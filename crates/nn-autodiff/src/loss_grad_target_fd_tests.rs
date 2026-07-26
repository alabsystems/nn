#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Finite-difference gradient tests for loss function grad_target paths.
//!
//! The backward rules for MSE, L1, and Huber loss compute gradients for BOTH
//! input and target tensors. Existing FD tests (`loss_tests.rs`) only verify
//! grad_input. This file closes the gap by perturbing the target tensor and
//! comparing the analytical gradient against finite-difference numerical
//! gradient.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use super::backward;
use super::test_helpers::{check_fd_grad, sum_f64};
use crate::tracked::TrackedTensor;
use crate::Var;

/// Finite-difference gradient check for loss grad_target.
///
/// Perturbs the target tensor element-by-element and compares
/// the analytical gradient (from backward) against numerical FD.
fn fd_loss_grad_target<F>(input_vals: &[f32], target_vals: &[f32], loss_fn: F)
where
    F: Fn(&Arc<TrackedTensor>, &Arc<TrackedTensor>) -> crate::error::Result<Arc<TrackedTensor>>,
{
    let n = target_vals.len();

    // Make target a Var so backward accumulates gradient for it
    let target_var = Var::new(DynTensor::from_vec(target_vals.to_vec(), &[n], &cpu()).unwrap());
    let input_t = Arc::new(TrackedTensor::from_tensor(
        DynTensor::from_vec(input_vals.to_vec(), &[n], &cpu()).unwrap(),
    ));
    let target_t = Arc::new(TrackedTensor::from_var(&target_var).unwrap());

    // Compute analytical gradient via backward
    let loss = loss_fn(&input_t, &target_t).unwrap();
    let grads = backward(&loss).unwrap();
    let grad = grads
        .get(&target_var)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // Finite-difference: perturb target, compute loss, compare
    check_fd_grad(&grad, target_vals, 1e-3, |perturbed_target| {
        let inp = DynTensor::from_vec(input_vals.to_vec(), &[n], &cpu()).unwrap();
        let tgt = DynTensor::from_vec(perturbed_target, &[n], &cpu()).unwrap();
        let i = Arc::new(TrackedTensor::from_tensor(inp));
        let t = Arc::new(TrackedTensor::from_tensor(tgt));
        let l = loss_fn(&i, &t).unwrap();
        sum_f64(l.tensor())
    });
}

#[test]
fn test_mse_grad_target_fd() {
    fd_loss_grad_target(
        &[1.0, 2.0, 3.0, 4.0],
        &[1.5, 2.5, 3.5, 4.5],
        TrackedTensor::mse_loss,
    );
}

#[test]
fn test_mse_grad_target_fd_large_diff() {
    fd_loss_grad_target(
        &[10.0, -5.0, 0.0],
        &[1.0, 3.0, -2.0],
        TrackedTensor::mse_loss,
    );
}

#[test]
fn test_l1_grad_target_fd() {
    // Avoid diffs near zero where L1 gradient is undefined
    fd_loss_grad_target(
        &[1.0, 3.0, 5.0, 7.0],
        &[2.0, 4.0, 6.0, 8.0],
        TrackedTensor::l1_loss,
    );
}

#[test]
fn test_l1_grad_target_fd_negative() {
    // Target is below input — sign flips
    fd_loss_grad_target(&[5.0, 10.0, 15.0], &[1.0, 2.0, 3.0], TrackedTensor::l1_loss);
}

#[test]
fn test_huber_grad_target_fd_quadratic() {
    // Small diffs (quadratic region, delta=2.0)
    fd_loss_grad_target(&[1.0, 2.0, 3.0], &[1.5, 2.5, 3.5], |x, t| {
        x.huber_loss(t, 2.0)
    });
}

#[test]
fn test_huber_grad_target_fd_linear() {
    // Large diffs (linear region, delta=0.1)
    fd_loss_grad_target(&[5.0, 10.0], &[1.0, 2.0], |x, t| x.huber_loss(t, 0.1));
}

#[test]
fn test_huber_grad_target_fd_mixed() {
    // Mixed: some elements in quadratic, some in linear region (delta=1.0)
    // diffs: [0.3, 2.0, -0.5, -3.0] — elements 0,2 quadratic, 1,3 linear
    fd_loss_grad_target(&[1.0, 4.0, 2.0, 7.0], &[0.7, 2.0, 2.5, 10.0], |x, t| {
        x.huber_loss(t, 1.0)
    });
}
