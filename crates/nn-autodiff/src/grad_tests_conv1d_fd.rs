#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv1d backward finite-difference tests for groups and dilation coverage.
//!
//! Complements the basic conv1d backward tests in `grad_tests_extended.rs`
//! which only cover groups=1 and dilation=1.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::grad::test_helpers::{check_fd_grad, sum_f64};
use crate::tracked::TrackedTensor;
use crate::var::Var;

// -- Conv1d backward with groups=2 -------------------------------------------

/// Conv1d backward with groups=2: in_channels=4, out_channels=4, kernel_size=3.
/// Group 0: in_ch [0,1] → out_ch [0,1], Group 1: in_ch [2,3] → out_ch [2,3].
/// Finite-difference validation for both grad_input and grad_kernel.
#[test]
fn test_backward_conv1d_groups2_finite_diff() {
    let in_ch = 4;
    let in_len = 6;
    let out_ch = 4;
    let k_size = 3;
    let groups = 2;
    let k_in_ch = in_ch / groups; // 2

    let x_data: Vec<f32> = (0..in_ch * in_len).map(|v| v as f32 * 0.1 - 0.5).collect();
    let k_data: Vec<f32> = (0..out_ch * k_in_ch * k_size)
        .map(|v| v as f32 * 0.07 - 0.3)
        .collect();
    let eps = 1e-3_f32;

    let x_shape = [1, in_ch, in_len];
    let k_shape = [out_ch, k_in_ch, k_size];

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &x_shape, &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(k_data.clone(), &k_shape, &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx.conv1d(&tk, 0, 1, 1, groups).unwrap();

    // output shape: [1, 4, 4] (out_len = 6 - 3 + 1 = 4)
    assert_eq!(y.tensor().dims(), &[1, out_ch, 4]);

    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let analytical_gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    let analytical_gk = grads.get(&k_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical_gx, &x_data, eps, |d| {
        let t = DynTensor::from_vec(d, &x_shape, &cpu()).unwrap();
        let k = DynTensor::from_vec(k_data.clone(), &k_shape, &cpu()).unwrap();
        sum_f64(&t.conv1d(&k, 0, 1, 1, groups).unwrap())
    });
    check_fd_grad(&analytical_gk, &k_data, eps, |d| {
        let t = DynTensor::from_vec(x_data.clone(), &x_shape, &cpu()).unwrap();
        let k = DynTensor::from_vec(d, &k_shape, &cpu()).unwrap();
        sum_f64(&t.conv1d(&k, 0, 1, 1, groups).unwrap())
    });
}

// -- Conv1d backward with dilation=2 -----------------------------------------

/// Conv1d backward with dilation=2: in_channels=2, out_channels=2, kernel_size=3.
/// Effective kernel size = 1 + (3-1)*2 = 5, so output_len = 8 - 5 + 1 = 4.
/// Finite-difference validation for both grad_input and grad_kernel.
#[test]
fn test_backward_conv1d_dilation2_finite_diff() {
    let in_ch = 2;
    let in_len = 8;
    let out_ch = 2;
    let k_size = 3;
    let dilation = 2;

    let x_data: Vec<f32> = (0..in_ch * in_len).map(|v| v as f32 * 0.12 - 0.4).collect();
    let k_data: Vec<f32> = (0..out_ch * in_ch * k_size)
        .map(|v| v as f32 * 0.08 - 0.25)
        .collect();
    let eps = 1e-3_f32;

    let x_shape = [1, in_ch, in_len];
    let k_shape = [out_ch, in_ch, k_size];

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &x_shape, &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(k_data.clone(), &k_shape, &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx.conv1d(&tk, 0, 1, dilation, 1).unwrap();

    // effective_k = 1 + (3-1)*2 = 5, out_len = 8 - 5 + 1 = 4
    assert_eq!(y.tensor().dims(), &[1, out_ch, 4]);

    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let analytical_gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    let analytical_gk = grads.get(&k_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical_gx, &x_data, eps, |d| {
        let t = DynTensor::from_vec(d, &x_shape, &cpu()).unwrap();
        let k = DynTensor::from_vec(k_data.clone(), &k_shape, &cpu()).unwrap();
        sum_f64(&t.conv1d(&k, 0, 1, dilation, 1).unwrap())
    });
    check_fd_grad(&analytical_gk, &k_data, eps, |d| {
        let t = DynTensor::from_vec(x_data.clone(), &x_shape, &cpu()).unwrap();
        let k = DynTensor::from_vec(d, &k_shape, &cpu()).unwrap();
        sum_f64(&t.conv1d(&k, 0, 1, dilation, 1).unwrap())
    });
}

// -- Conv1d backward with groups=2 + dilation=2 combined ---------------------

/// Conv1d backward with groups=2 AND dilation=2: in_channels=4, out_channels=4.
/// This exercises the combined groups+dilation code path in both
/// `conv1d_kernel_grad` and `conv_transpose1d` backward.
#[test]
fn test_backward_conv1d_groups2_dilation2_finite_diff() {
    let in_ch = 4;
    let in_len = 10;
    let out_ch = 4;
    let k_size = 3;
    let groups = 2;
    let dilation = 2;
    let k_in_ch = in_ch / groups; // 2

    let x_data: Vec<f32> = (0..in_ch * in_len).map(|v| v as f32 * 0.06 - 0.8).collect();
    let k_data: Vec<f32> = (0..out_ch * k_in_ch * k_size)
        .map(|v| v as f32 * 0.09 - 0.35)
        .collect();
    let eps = 1e-3_f32;

    let x_shape = [1, in_ch, in_len];
    let k_shape = [out_ch, k_in_ch, k_size];

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &x_shape, &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(k_data.clone(), &k_shape, &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx.conv1d(&tk, 0, 1, dilation, groups).unwrap();

    // effective_k = 1 + (3-1)*2 = 5, out_len = 10 - 5 + 1 = 6
    assert_eq!(y.tensor().dims(), &[1, out_ch, 6]);

    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let analytical_gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    let analytical_gk = grads.get(&k_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical_gx, &x_data, eps, |d| {
        let t = DynTensor::from_vec(d, &x_shape, &cpu()).unwrap();
        let k = DynTensor::from_vec(k_data.clone(), &k_shape, &cpu()).unwrap();
        sum_f64(&t.conv1d(&k, 0, 1, dilation, groups).unwrap())
    });
    check_fd_grad(&analytical_gk, &k_data, eps, |d| {
        let t = DynTensor::from_vec(x_data.clone(), &x_shape, &cpu()).unwrap();
        let k = DynTensor::from_vec(d, &k_shape, &cpu()).unwrap();
        sum_f64(&t.conv1d(&k, 0, 1, dilation, groups).unwrap())
    });
}
