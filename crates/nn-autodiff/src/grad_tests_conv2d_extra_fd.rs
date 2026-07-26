#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional Conv2d backward tests: groups and non-square spatial dimensions.
//!
//! Extracted from grad_tests_conv2d_fd.rs to keep that file under 500 lines.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::grad::test_helpers::check_fd_grad;
use crate::tracked::TrackedTensor;
use crate::var::Var;

/// Conv2d backward with non-square input and stride=2.
///
/// input: [1, 1, 5, 6], kernel: [1, 1, 2, 2], stride=2, padding=0
/// output: [1, 1, 2, 3] (asymmetric spatial dims)
///
/// This exercises `conv2d_input_grad_asymmetric` because:
///   output_padding_h = (5 - 2) % 2 = 1
///   output_padding_w = (6 - 2) % 2 = 0
/// So op_h != op_w, triggering the max(op_h, op_w) + narrow path.
#[test]
fn test_backward_conv2d_nonsquare_stride2_finite_diff() {
    let x_data: Vec<f32> = (1..=30).map(|v| v as f32 * 0.1).collect(); // [1,1,5,6]
    let k_data: Vec<f32> = vec![0.5, -0.3, 0.8, 0.1]; // [1,1,2,2]
    let eps = 1e-4_f32;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 1, 5, 6], &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(k_data.clone(), &[1, 1, 2, 2], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx.conv2d(&tk, 0, 2, 1, 1).unwrap(); // padding=0, stride=2, dilation=1, groups=1

    // out_h = (5-2)/2+1 = 2, out_w = (6-2)/2+1 = 3
    assert_eq!(y.tensor().dims(), &[1, 1, 2, 3]);

    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap()
        .sum_keepdim(3)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let analytical_gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    let analytical_gk = grads.get(&k_var).unwrap().to_flat_vec::<f32>().unwrap();

    // Verify grad_input shape matches input (the key check for asymmetric path)
    assert_eq!(grads.get(&x_var).unwrap().dims(), &[1, 1, 5, 6]);

    check_fd_grad(&analytical_gx, &x_data, eps, |d| {
        let t = DynTensor::from_vec(d, &[1, 1, 5, 6], &cpu()).unwrap();
        let k = DynTensor::from_vec(k_data.clone(), &[1, 1, 2, 2], &cpu()).unwrap();
        t.conv2d(&k, 0, 2, 1, 1)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
    check_fd_grad(&analytical_gk, &k_data, eps, |d| {
        let t = DynTensor::from_vec(x_data.clone(), &[1, 1, 5, 6], &cpu()).unwrap();
        let k = DynTensor::from_vec(d, &[1, 1, 2, 2], &cpu()).unwrap();
        t.conv2d(&k, 0, 2, 1, 1)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
}

/// Conv2d backward with groups > 1 (depthwise-style convolution).
///
/// input: [1, 4, 3, 3], kernel: [4, 1, 2, 2], groups=4 (depthwise: each channel convolved independently)
/// Finite-difference validation for both grad_input and grad_kernel.
///
/// This exercises the groups loop in `conv2d_kernel_grad` and `conv_transpose2d`
/// backward paths which are untested at groups=1.
#[test]
fn test_backward_conv2d_groups_finite_diff() {
    // 4-channel depthwise conv: groups=4, in_ch=4, out_ch=4, kernel=[4,1,2,2]
    let x_data: Vec<f32> = (1..=36).map(|v| v as f32 * 0.1).collect(); // [1,4,3,3]
    let k_data: Vec<f32> = vec![
        0.5, -0.3, 0.2, 0.7, // ch 0
        -0.1, 0.4, 0.6, -0.2, // ch 1
        0.3, 0.1, -0.5, 0.8, // ch 2
        -0.4, 0.9, 0.2, -0.6, // ch 3
    ]; // [4,1,2,2]
    let eps = 1e-4_f32;
    let groups = 4;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 4, 3, 3], &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(k_data.clone(), &[4, 1, 2, 2], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx.conv2d(&tk, 0, 1, 1, groups).unwrap();

    // output shape: [1, 4, 2, 2]
    assert_eq!(y.tensor().dims(), &[1, 4, 2, 2]);

    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap()
        .sum_keepdim(3)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let analytical_gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    let analytical_gk = grads.get(&k_var).unwrap().to_flat_vec::<f32>().unwrap();

    check_fd_grad(&analytical_gx, &x_data, eps, |d| {
        let t = DynTensor::from_vec(d, &[1, 4, 3, 3], &cpu()).unwrap();
        let k = DynTensor::from_vec(k_data.clone(), &[4, 1, 2, 2], &cpu()).unwrap();
        t.conv2d(&k, 0, 1, 1, groups)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
    check_fd_grad(&analytical_gk, &k_data, eps, |d| {
        let t = DynTensor::from_vec(x_data.clone(), &[1, 4, 3, 3], &cpu()).unwrap();
        let k = DynTensor::from_vec(d, &[4, 1, 2, 2], &cpu()).unwrap();
        t.conv2d(&k, 0, 1, 1, groups)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
}

/// Conv2d backward with groups=2 (non-depthwise grouped convolution).
///
/// input: [1, 4, 3, 3], kernel: [4, 2, 2, 2], groups=2
/// Group 0: in_ch [0,1] → out_ch [0,1], Group 1: in_ch [2,3] → out_ch [2,3]
/// This is a distinct code path from depthwise (groups=in_ch) because
/// k_in_ch=2 (not 1) per group.
#[test]
fn test_backward_conv2d_groups2_finite_diff() {
    let x_data: Vec<f32> = (1..=36).map(|v| v as f32 * 0.1).collect(); // [1,4,3,3]
                                                                       // Kernel [4, 2, 2, 2]: 4 out channels, each sees 2 in channels, 2x2 spatial
    let k_data: Vec<f32> = (1..=32).map(|v| v as f32 * 0.05 - 0.4).collect(); // [4,2,2,2]
    let groups = 2;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 4, 3, 3], &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(k_data.clone(), &[4, 2, 2, 2], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx.conv2d(&tk, 0, 1, 1, groups).unwrap();

    assert_eq!(y.tensor().dims(), &[1, 4, 2, 2]);

    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap()
        .sum_keepdim(3)
        .unwrap();
    let grads = backward(&loss).unwrap();
    let analytical_gx = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    let analytical_gk = grads.get(&k_var).unwrap().to_flat_vec::<f32>().unwrap();

    let eps = 1e-3_f32;
    check_fd_grad(&analytical_gx, &x_data, eps, |d| {
        let t = DynTensor::from_vec(d, &[1, 4, 3, 3], &cpu()).unwrap();
        let k = DynTensor::from_vec(k_data.clone(), &[4, 2, 2, 2], &cpu()).unwrap();
        t.conv2d(&k, 0, 1, 1, groups)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
    check_fd_grad(&analytical_gk, &k_data, eps, |d| {
        let t = DynTensor::from_vec(x_data.clone(), &[1, 4, 3, 3], &cpu()).unwrap();
        let k = DynTensor::from_vec(d, &[4, 2, 2, 2], &cpu()).unwrap();
        t.conv2d(&k, 0, 1, 1, groups)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
}
