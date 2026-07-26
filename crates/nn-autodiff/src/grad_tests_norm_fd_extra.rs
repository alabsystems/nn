#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Finite-difference gradient tests for BatchNorm and InstanceNorm backward rules.
//! Extracted from `grad_tests_norm_fd.rs` for 500-line compliance.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::grad::test_helpers::check_fd_grad;
use crate::tracked::TrackedTensor;
use crate::var::Var;

// ── BatchNorm finite-difference test ────────────────────────────────

/// Reference forward for BatchNorm: normalize over batch+spatial, apply affine. Returns f64 for FD precision.
fn batch_norm_scalar_loss(x_data: Vec<f32>, gamma: &[f32], beta: &[f32], eps: f64) -> f64 {
    // Input shape [N=2, C=2, T=3]
    let c = gamma.len();
    let n = 2;
    let t = x_data.len() / (n * c);
    let x = DynTensor::from_vec(x_data, &[n, c, t], &cpu()).unwrap();
    let g = DynTensor::from_vec(gamma.to_vec(), &[c], &cpu()).unwrap();
    let b = DynTensor::from_vec(beta.to_vec(), &[c], &cpu()).unwrap();

    // Mean over dim 0 (batch) and dim 2 (spatial)
    let mean = x.mean_keepdim(2).unwrap().mean_keepdim(0).unwrap();
    let diff = x.sub(&mean).unwrap();
    let var = diff
        .sqr()
        .unwrap()
        .mean_keepdim(2)
        .unwrap()
        .mean_keepdim(0)
        .unwrap();
    let inv_std = var
        .add_scalar(eps)
        .unwrap()
        .sqrt()
        .unwrap()
        .recip()
        .unwrap();
    let normed = diff.mul(&inv_std).unwrap();
    let g_bc = g.reshape([1, c, 1]).unwrap();
    let b_bc = b.reshape([1, c, 1]).unwrap();
    normed
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
fn test_backward_batch_norm_fd() {
    // [N=2, C=2, T=3]
    let x_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // batch 1
    ];
    let gamma_data = vec![1.0, 0.5];
    let beta_data = vec![0.0, 0.1];
    let eps = 1e-5_f64;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 2, 3], &cpu()).unwrap());
    let g_var = Var::new(DynTensor::from_vec(gamma_data.clone(), &[2], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(beta_data.clone(), &[2], &cpu()).unwrap());

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tg = Arc::new(TrackedTensor::from_var(&g_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = tx.batch_norm(&tg, &tb, eps).unwrap();
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
        batch_norm_scalar_loss(d, &gamma_data, &beta_data, eps)
    });

    // grad_bias should sum to N*T = 2*3 = 6 per channel (loss = sum of all)
    let analytical_b = grads.get(&b_var).unwrap().to_flat_vec::<f32>().unwrap();
    for (i, &gb) in analytical_b.iter().enumerate() {
        assert!(
            (gb - 6.0).abs() < 1e-5,
            "batch_norm grad_bias[{i}]: expected 6.0, got {gb}"
        );
    }
}

// ── InstanceNorm finite-difference test ─────────────────────────────

/// Reference forward for InstanceNorm: normalize per (N,C) over spatial. Returns f64 for FD precision.
fn instance_norm_scalar_loss(x_data: Vec<f32>, gamma: &[f32], beta: &[f32], eps: f64) -> f64 {
    // Input shape [N=1, C=2, T=4]
    let c = gamma.len();
    let n = 1;
    let t = x_data.len() / (n * c);
    let x = DynTensor::from_vec(x_data, &[n, c, t], &cpu()).unwrap();
    let g = DynTensor::from_vec(gamma.to_vec(), &[c], &cpu()).unwrap();
    let b = DynTensor::from_vec(beta.to_vec(), &[c], &cpu()).unwrap();

    // Mean over spatial dim (2)
    let mean = x.mean_keepdim(2).unwrap();
    let diff = x.sub(&mean).unwrap();
    let var = diff.sqr().unwrap().mean_keepdim(2).unwrap();
    let inv_std = var
        .add_scalar(eps)
        .unwrap()
        .sqrt()
        .unwrap()
        .recip()
        .unwrap();
    let normed = diff.mul(&inv_std).unwrap();
    let g_bc = g.reshape([1, c, 1]).unwrap();
    let b_bc = b.reshape([1, c, 1]).unwrap();
    normed
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
fn test_backward_instance_norm_fd() {
    // [N=1, C=2, T=4]
    let x_data: Vec<f32> = vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0];
    let gamma_data = vec![1.0, 2.0];
    let beta_data = vec![0.0, 0.5];
    let eps = 1e-5_f64;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 2, 4], &cpu()).unwrap());
    let g_var = Var::new(DynTensor::from_vec(gamma_data.clone(), &[2], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(beta_data.clone(), &[2], &cpu()).unwrap());

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tg = Arc::new(TrackedTensor::from_var(&g_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = tx.instance_norm(&tg, &tb, eps).unwrap();
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
        instance_norm_scalar_loss(d, &gamma_data, &beta_data, eps)
    });

    // Finite-difference for grad_weight
    let analytical_g = grads.get(&g_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_g, &gamma_data, fd_eps, |g| {
        instance_norm_scalar_loss(x_data.clone(), &g, &beta_data, eps)
    });
}
