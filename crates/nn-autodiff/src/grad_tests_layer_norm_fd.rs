#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Finite-difference gradient test for LayerNorm.
//!
//! LayerNorm was the only normalization op missing a proper FD gradient test.
//! RmsNorm, GroupNorm, BatchNorm, InstanceNorm all have FD tests in
//! `grad_tests_norm_fd.rs`. The existing test in `grad_tests_composite.rs`
//! uses trivial gamma=[1,1,1] beta=[0,0,0] which doesn't exercise grad_weight
//! meaningfully.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::grad::test_helpers::check_fd_grad;
use crate::tracked::TrackedTensor;
use crate::var::Var;

/// Reference forward for LayerNorm: normalize over last dim, apply affine. Returns f64 for FD precision.
fn layer_norm_scalar_loss(x_data: Vec<f32>, gamma: &[f32], beta: &[f32], eps: f64) -> f64 {
    let n = 2;
    let d = gamma.len();
    let x = DynTensor::from_vec(x_data, &[n, d], &cpu()).unwrap();
    let g = DynTensor::from_vec(gamma.to_vec(), &[d], &cpu()).unwrap();
    let b = DynTensor::from_vec(beta.to_vec(), &[d], &cpu()).unwrap();

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
    normed
        .mul(&g)
        .unwrap()
        .add(&b)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .sum()
}

#[test]
fn test_backward_layer_norm_fd() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gamma_data = vec![1.0, 0.5, 2.0];
    let beta_data = vec![0.1, -0.2, 0.3];
    let eps = 1e-5_f64;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[2, 3], &cpu()).unwrap());
    let g_var = Var::new(DynTensor::from_vec(gamma_data.clone(), &[3], &cpu()).unwrap());
    let b_var = Var::new(DynTensor::from_vec(beta_data.clone(), &[3], &cpu()).unwrap());

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tg = Arc::new(TrackedTensor::from_var(&g_var).unwrap());
    let tb = Arc::new(TrackedTensor::from_var(&b_var).unwrap());
    let y = tx.layer_norm(&tg, &tb, eps).unwrap();
    let loss = y.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
    let grads = backward(&loss).unwrap();

    // Check shapes
    assert_eq!(grads.get(&x_var).unwrap().dims(), &[2, 3]);
    assert_eq!(grads.get(&g_var).unwrap().dims(), &[3]);
    assert_eq!(grads.get(&b_var).unwrap().dims(), &[3]);

    // Finite-difference check for grad_x
    let analytical_x = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    let fd_eps = 1e-3_f32;
    check_fd_grad(&analytical_x, &x_data, fd_eps, |d| {
        layer_norm_scalar_loss(d, &gamma_data, &beta_data, eps)
    });

    // Finite-difference check for grad_weight
    let analytical_g = grads.get(&g_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_g, &gamma_data, fd_eps, |g| {
        layer_norm_scalar_loss(x_data.clone(), &g, &beta_data, eps)
    });

    // grad_bias should sum to N=2 per element (loss = sum of all)
    let analytical_b = grads.get(&b_var).unwrap().to_flat_vec::<f32>().unwrap();
    for (i, &gb) in analytical_b.iter().enumerate() {
        assert!(
            (gb - 2.0).abs() < 1e-5,
            "layer_norm grad_bias[{i}]: expected 2.0, got {gb}"
        );
    }
}
