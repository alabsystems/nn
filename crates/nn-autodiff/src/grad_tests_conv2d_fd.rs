#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv2d backward tests: analytical verification and finite-difference validation.
//!
//! Extracted from grad_tests_composite.rs to keep that file under 500 lines.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::grad::test_helpers::check_fd_grad;
use crate::tracked::TrackedTensor;
use crate::var::Var;

// -- Conv2d backward tests ---------------------------------------------------

/// Simple conv2d backward: 1x1 input channel, 1x1 output channel, 2x2 kernel.
/// input: [1,1,3,3], kernel: [1,1,2,2], padding=0, stride=1
/// output: [1,1,2,2]
/// loss = sum(output)
#[test]
fn test_backward_conv2d_simple() {
    // input: 3x3 = [1..9]
    let x_data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
    // kernel: 2x2 = [1,1,1,1]
    let k_data = vec![1.0f32; 4];
    let x_var = Var::new(DynTensor::from_vec(x_data, &[1, 1, 3, 3], &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(k_data, &[1, 1, 2, 2], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx.conv2d(&tk, 0, 1, 1, 1).unwrap();

    // output shape should be [1,1,2,2]
    assert_eq!(y.tensor().dims(), &[1, 1, 2, 2]);

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

    // grad_kernel: each element = sum of the 2x2 patch it covers in input
    //   gk[0,0] = 1+2+4+5=12, gk[0,1] = 2+3+5+6=16
    //   gk[1,0] = 4+5+7+8=24, gk[1,1] = 5+6+8+9=28
    let gk = grads.get(&k_var).unwrap().to_flat_vec::<f32>().unwrap();
    assert!(
        (gk[0] - 12.0).abs() < 1e-4
            && (gk[1] - 16.0).abs() < 1e-4
            && (gk[2] - 24.0).abs() < 1e-4
            && (gk[3] - 28.0).abs() < 1e-4,
        "grad_kernel: expected [12,16,24,28], got {gk:?}"
    );

    // grad_input shape should match input
    let gx = grads.get(&x_var).unwrap();
    assert_eq!(gx.dims(), &[1, 1, 3, 3]);
    let gx_vals = gx.to_flat_vec::<f32>().unwrap();
    // With all-ones kernel and all-ones grad_output:
    // Each input pixel receives grad from the output positions it contributes to.
    // Corner pixels participate in 1 output, edge in 2, center in 4.
    let expected_gx = [1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0];
    for (i, (&got, &exp)) in gx_vals.iter().zip(expected_gx.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "grad_input[{i}]: expected {exp}, got {got}"
        );
    }
}

/// Finite-difference validation for conv2d backward.
#[test]
fn test_backward_conv2d_finite_diff() {
    let x_data: Vec<f32> = vec![1.0, 0.5, -0.3, 0.8, -1.2, 0.7, 0.2, -0.5, 1.1];
    let k_data: Vec<f32> = vec![0.5, -0.3, 0.8, 0.1];
    let eps = 1e-4_f32;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 1, 3, 3], &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(k_data.clone(), &[1, 1, 2, 2], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx.conv2d(&tk, 0, 1, 1, 1).unwrap();
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

    // Finite-diff for grad_input
    check_fd_grad(&analytical_gx, &x_data, eps, |d| {
        let t = DynTensor::from_vec(d, &[1, 1, 3, 3], &cpu()).unwrap();
        let k = DynTensor::from_vec(k_data.clone(), &[1, 1, 2, 2], &cpu()).unwrap();
        t.conv2d(&k, 0, 1, 1, 1)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });

    // Finite-diff for grad_kernel
    check_fd_grad(&analytical_gk, &k_data, eps, |d| {
        let t = DynTensor::from_vec(x_data.clone(), &[1, 1, 3, 3], &cpu()).unwrap();
        let k = DynTensor::from_vec(d, &[1, 1, 2, 2], &cpu()).unwrap();
        t.conv2d(&k, 0, 1, 1, 1)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
}

/// Conv2d backward with stride=2 and padding=1.
#[test]
fn test_backward_conv2d_stride2_padding1() {
    let x_data: Vec<f32> = (1..=16).map(|v| v as f32).collect(); // [1,1,4,4]
    let k_data = vec![1.0, 0.5, -0.5, 1.0]; // [1,1,2,2]
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 1, 4, 4], &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(k_data.clone(), &[1, 1, 2, 2], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx.conv2d(&tk, 1, 2, 1, 1).unwrap();
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

    // Finite-diff check for grad_input (wider tol for stride+padding)
    check_fd_grad(&analytical_gx, &x_data, eps, |d| {
        let t = DynTensor::from_vec(d, &[1, 1, 4, 4], &cpu()).unwrap();
        let k = DynTensor::from_vec(k_data.clone(), &[1, 1, 2, 2], &cpu()).unwrap();
        t.conv2d(&k, 1, 2, 1, 1)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
}

/// Conv2d backward with dilation=2.
///
/// input: [1, 1, 5, 5], kernel: [1, 1, 2, 2], dilation=2, stride=1, padding=0
/// Effective kernel covers 3x3 spatial extent → output: [1, 1, 3, 3]
/// Finite-difference validation for both grad_input and grad_kernel.
#[test]
fn test_backward_conv2d_dilation2_finite_diff() {
    let x_data: Vec<f32> = (1..=25).map(|v| v as f32 * 0.1).collect(); // [1,1,5,5]
    let k_data: Vec<f32> = vec![0.5, -0.3, 0.8, 0.1]; // [1,1,2,2]
    let eps = 1e-4_f32;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 1, 5, 5], &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(k_data.clone(), &[1, 1, 2, 2], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx.conv2d(&tk, 0, 1, 2, 1).unwrap(); // padding=0, stride=1, dilation=2, groups=1

    // Effective kernel size = 1 + (2-1)*2 = 3, output_size = (5-3)/1+1 = 3
    assert_eq!(y.tensor().dims(), &[1, 1, 3, 3]);

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
        let t = DynTensor::from_vec(d, &[1, 1, 5, 5], &cpu()).unwrap();
        let k = DynTensor::from_vec(k_data.clone(), &[1, 1, 2, 2], &cpu()).unwrap();
        t.conv2d(&k, 0, 1, 2, 1)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
    check_fd_grad(&analytical_gk, &k_data, eps, |d| {
        let t = DynTensor::from_vec(x_data.clone(), &[1, 1, 5, 5], &cpu()).unwrap();
        let k = DynTensor::from_vec(d, &[1, 1, 2, 2], &cpu()).unwrap();
        t.conv2d(&k, 0, 1, 2, 1)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
}

/// Conv2d backward with stride=2 and dilation=2 combined.
///
/// input: [1, 1, 7, 7], kernel: [1, 1, 2, 2], stride=2, dilation=2, padding=1
/// Exercises both stride and dilation paths simultaneously in conv2d_kernel_grad.
#[test]
fn test_backward_conv2d_stride2_dilation2_finite_diff() {
    let x_data: Vec<f32> = (1..=49).map(|v| v as f32 * 0.05).collect(); // [1,1,7,7]
    let k_data: Vec<f32> = vec![0.3, -0.7, 0.4, 0.6]; // [1,1,2,2]
    let eps = 1e-3_f32;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 1, 7, 7], &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(k_data.clone(), &[1, 1, 2, 2], &cpu()).unwrap());
    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx.conv2d(&tk, 1, 2, 2, 1).unwrap(); // padding=1, stride=2, dilation=2, groups=1

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
        let t = DynTensor::from_vec(d, &[1, 1, 7, 7], &cpu()).unwrap();
        let k = DynTensor::from_vec(k_data.clone(), &[1, 1, 2, 2], &cpu()).unwrap();
        t.conv2d(&k, 1, 2, 2, 1)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
    check_fd_grad(&analytical_gk, &k_data, eps, |d| {
        let t = DynTensor::from_vec(x_data.clone(), &[1, 1, 7, 7], &cpu()).unwrap();
        let k = DynTensor::from_vec(d, &[1, 1, 2, 2], &cpu()).unwrap();
        t.conv2d(&k, 1, 2, 2, 1)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .sum()
    });
}

// Groups + non-square tests extracted to grad_tests_conv2d_extra_fd.rs
