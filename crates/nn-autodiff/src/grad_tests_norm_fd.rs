#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Finite-difference gradient tests for normalization ops.
//!
//! Covers backward rules for:
//! - RmsNorm (backward_rms_norm in backward_rules_special.rs)
//! - GroupNorm (backward_group_norm in backward_rules_special.rs)
//!
//! BatchNorm and InstanceNorm FD tests extracted to `grad_tests_norm_fd_extra.rs`.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::grad::test_helpers::check_fd_grad;
use crate::tracked::TrackedTensor;
use crate::var::Var;

// ── RmsNorm finite-difference test ──────────────────────────────────

/// Reference forward for RmsNorm: x / rms(x) * weight, summed to scalar. Returns f64 for FD precision.
fn rms_norm_scalar_loss(x_data: Vec<f32>, gamma: &[f32], eps: f64) -> f64 {
    let t = DynTensor::from_vec(x_data, &[2, 3], &cpu()).unwrap();
    let g = DynTensor::from_vec(gamma.to_vec(), &[3], &cpu()).unwrap();
    let rms_sq = t.sqr().unwrap().mean_keepdim(1).unwrap();
    let inv_rms = rms_sq
        .add_scalar(eps)
        .unwrap()
        .sqrt()
        .unwrap()
        .recip()
        .unwrap();
    let normed = t.mul(&inv_rms).unwrap();
    normed
        .mul(&g)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .sum()
}

#[test]
fn test_backward_rms_norm_fd() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gamma_data = vec![1.0, 0.5, 2.0];
    let eps = 1e-5_f64;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap());
    let g_var = Var::new(DynTensor::from_vec(gamma_data.clone(), &[3], &cpu()).unwrap());

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tg = Arc::new(TrackedTensor::from_var(&g_var).unwrap());
    let y = tx.rms_norm(&tg, eps).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();

    // Check shapes
    assert_eq!(grads.get(&x_var).unwrap().dims(), &[2, 3]);
    assert_eq!(grads.get(&g_var).unwrap().dims(), &[3]);

    // Finite-difference check for grad_x (tighter tol for norm ops)
    let analytical_x = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    let fd_eps = 1e-3_f32;
    check_fd_grad(&analytical_x, &x_data, fd_eps, |d| {
        rms_norm_scalar_loss(d, &gamma_data, eps)
    });

    // Finite-difference check for grad_weight
    let analytical_g = grads.get(&g_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_g, &gamma_data, fd_eps, |g| {
        rms_norm_scalar_loss(x_data.clone(), &g, eps)
    });
}

// ── GroupNorm finite-difference test ────────────────────────────────

/// Reference forward for GroupNorm: normalize within groups, apply affine. Returns f64 for FD precision.
fn group_norm_scalar_loss(
    x_data: Vec<f32>,
    gamma: &[f32],
    beta: &[f32],
    num_groups: usize,
    eps: f64,
) -> f64 {
    // Input shape [N=1, C=4, T=2], 2 groups of 2 channels each
    let n = 1;
    let c = gamma.len();
    let t = x_data.len() / (n * c);
    let x = DynTensor::from_vec(x_data, &[n, c, t], &cpu()).unwrap();
    let g = DynTensor::from_vec(gamma.to_vec(), &[c], &cpu()).unwrap();
    let b = DynTensor::from_vec(beta.to_vec(), &[c], &cpu()).unwrap();

    let channels_per_group = c / num_groups;
    let xr = x.reshape([n, num_groups, channels_per_group, t]).unwrap();
    // Mean and var over dims 2,3 (channels_per_group, T)
    let mean = xr.mean_keepdim(3).unwrap().mean_keepdim(2).unwrap();
    let diff = xr.sub(&mean).unwrap();
    let var = diff
        .sqr()
        .unwrap()
        .mean_keepdim(3)
        .unwrap()
        .mean_keepdim(2)
        .unwrap();
    let inv_std = var
        .add_scalar(eps)
        .unwrap()
        .sqrt()
        .unwrap()
        .recip()
        .unwrap();
    let normed = diff.mul(&inv_std).unwrap();
    let normed_flat = normed.reshape([n, c, t]).unwrap();
    let g_bc = g.reshape([1, c, 1]).unwrap();
    let b_bc = b.reshape([1, c, 1]).unwrap();
    normed_flat
        .mul(&g_bc)
        .unwrap()
        .add(&b_bc)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .sum()
}

#[test]
fn test_backward_group_norm_fd() {
    // [N=1, C=4, T=2], 2 groups
    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let gamma_data = vec![1.0, 0.5, 2.0, 1.5];
    let beta_data = vec![0.0, 0.1, -0.1, 0.0];
    let num_groups = 2;
    let eps = 1e-5_f64;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 4, 2], &cpu()).unwrap());
    let g_var = Var::new(DynTensor::from_vec(gamma_data.clone(), &[4], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(beta_data.clone(), &[4], &cpu()).unwrap());

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tg = Arc::new(TrackedTensor::from_var(&g_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = tx.group_norm(&tg, &tb, num_groups, eps).unwrap();
    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();

    // Finite-difference for grad_x
    let analytical_x = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    let fd_eps = 1e-3_f32;
    check_fd_grad(&analytical_x, &x_data, fd_eps, |d| {
        group_norm_scalar_loss(d, &gamma_data, &beta_data, num_groups, eps)
    });

    // Finite-difference for grad_weight
    let analytical_g = grads.get(&g_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_g, &gamma_data, fd_eps, |g| {
        group_norm_scalar_loss(x_data.clone(), &g, &beta_data, num_groups, eps)
    });
}

// -- reshape_for_channel_broadcast edge cases (#1633) --

#[test]
fn test_reshape_for_channel_broadcast_rank_0_returns_err() {
    let t = DynTensor::from_vec(vec![3.0f32], &[1], &cpu()).unwrap();
    let result = crate::backward_rules::reshape_for_channel_broadcast(&t, 0);
    assert!(result.is_err(), "target_rank=0 should return Err");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("target_rank must be >= 2"),
        "error message: {msg}"
    );
}

#[test]
fn test_reshape_for_channel_broadcast_rank_1_returns_err() {
    let t = DynTensor::from_vec(vec![5.0f32], &[1], &cpu()).unwrap();
    let result = crate::backward_rules::reshape_for_channel_broadcast(&t, 1);
    assert!(result.is_err(), "target_rank=1 should return Err");
}

#[test]
fn test_reshape_for_channel_broadcast_rank_2_ok() {
    let t = DynTensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3], &cpu()).unwrap();
    let result = crate::backward_rules::reshape_for_channel_broadcast(&t, 2).unwrap();
    assert_eq!(result.dims(), &[1, 3], "rank 2: shape should be [1, 3]");
}
