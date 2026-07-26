#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Finite-difference gradient test for ConvTranspose1d backward rule.
//!
//! Covers backward_conv_transpose1d in backward_rules_conv_transpose.rs.
//! Dilation and groups FD tests extracted to `grad_tests_conv_transpose1d_fd_extra.rs`.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::grad::test_helpers::check_fd_grad;
use crate::tracked::TrackedTensor;
use crate::var::Var;

/// Reference forward for ConvTranspose1d (basic, no dilation/groups). Returns f64 for FD precision.
fn conv_transpose1d_scalar_loss(
    x_data: Vec<f32>,
    kernel: &[f32],
    kernel_shape: &[usize],
    padding: usize,
    stride: usize,
) -> f64 {
    let x = DynTensor::from_vec(x_data, &[1, 1, 3], &cpu()).unwrap();
    let k = DynTensor::from_vec(kernel.to_vec(), kernel_shape, &cpu()).unwrap();
    x.conv_transpose1d(&k, padding, 0, stride, 1, 1)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .sum()
}

/// Reference forward for ConvTranspose1d with output_padding. Returns f64 for FD precision.
fn conv_transpose1d_scalar_loss_full(
    x_data: Vec<f32>,
    input_shape: &[usize],
    kernel: &[f32],
    kernel_shape: &[usize],
    padding: usize,
    output_padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> f64 {
    let x = DynTensor::from_vec(x_data, input_shape, &cpu()).unwrap();
    let k = DynTensor::from_vec(kernel.to_vec(), kernel_shape, &cpu()).unwrap();
    x.conv_transpose1d(&k, padding, output_padding, stride, dilation, groups)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .sum()
}

#[test]
fn test_backward_conv_transpose1d_fd() {
    // Input: [B=1, C_in=1, L=3], Kernel: [C_in=1, C_out=1, K=2]
    // stride=1, padding=0 → output length = (3-1)*1 - 0 + 2 = 4
    let x_data = vec![1.0, 2.0, 3.0];
    let kernel_data = vec![1.0, 0.5];
    let kernel_shape = [1, 1, 2]; // [C_in, C_out, K]
    let padding = 0;
    let stride = 1;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &[1, 1, 3], &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(kernel_data.clone(), &kernel_shape, &cpu()).unwrap());

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx.conv_transpose1d(&tk, padding, stride, 1, 1, 0).unwrap();
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
        conv_transpose1d_scalar_loss(d, &kernel_data, &kernel_shape, padding, stride)
    });

    // Finite-difference for grad_kernel
    let analytical_k = grads.get(&k_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_k, &kernel_data, fd_eps, |k| {
        conv_transpose1d_scalar_loss(x_data.clone(), &k, &kernel_shape, padding, stride)
    });
}

/// AC3: Baseline regression guard — stride=2, output_padding=0.
#[test]
fn test_backward_conv_transpose1d_stride2_no_output_padding_fd() {
    // Input: [B=1, C_in=1, L=4], Kernel: [C_in=1, C_out=1, K=3]
    // stride=2, padding=1, output_padding=0
    // out_len = (4-1)*2 - 2*1 + 1*(3-1) + 0 + 1 = 6 - 2 + 2 + 1 = 7
    let x_data = vec![1.0, 2.0, 3.0, 0.5];
    let input_shape = [1, 1, 4];
    let kernel_data = vec![0.3, 0.7, -0.2];
    let kernel_shape = [1, 1, 3];
    let padding = 1;
    let output_padding = 0;
    let stride = 2;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &input_shape, &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(kernel_data.clone(), &kernel_shape, &cpu()).unwrap());

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx
        .conv_transpose1d(&tk, padding, stride, 1, 1, output_padding)
        .unwrap();
    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let analytical_x = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    let fd_eps = 1e-3_f32;
    check_fd_grad(&analytical_x, &x_data, fd_eps, |d| {
        conv_transpose1d_scalar_loss_full(
            d,
            &input_shape,
            &kernel_data,
            &kernel_shape,
            padding,
            output_padding,
            stride,
            1,
            1,
        )
    });

    let analytical_k = grads.get(&k_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_k, &kernel_data, fd_eps, |k| {
        conv_transpose1d_scalar_loss_full(
            x_data.clone(),
            &input_shape,
            &k,
            &kernel_shape,
            padding,
            output_padding,
            stride,
            1,
            1,
        )
    });
}

/// AC2: output_padding=1 with stride=2 — the bug scenario.
/// Without the fix, grad_input would have wrong length or wrong values.
#[test]
fn test_backward_conv_transpose1d_output_padding1_stride2_fd() {
    // Input: [B=1, C_in=1, L=4], Kernel: [C_in=1, C_out=1, K=3]
    // stride=2, padding=1, output_padding=1
    // out_len = (4-1)*2 - 2*1 + 1*(3-1) + 1 + 1 = 6 - 2 + 2 + 1 + 1 = 8
    let x_data = vec![1.0, 2.0, 3.0, 0.5];
    let input_shape = [1, 1, 4];
    let kernel_data = vec![0.3, 0.7, -0.2];
    let kernel_shape = [1, 1, 3];
    let padding = 1;
    let output_padding = 1;
    let stride = 2;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &input_shape, &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(kernel_data.clone(), &kernel_shape, &cpu()).unwrap());

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx
        .conv_transpose1d(&tk, padding, stride, 1, 1, output_padding)
        .unwrap();
    let loss = y
        .sum_keepdim(0)
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sum_keepdim(2)
        .unwrap();
    let grads = backward(&loss).unwrap();

    let analytical_x = grads.get(&x_var).unwrap().to_flat_vec::<f32>().unwrap();
    assert_eq!(
        analytical_x.len(),
        x_data.len(),
        "grad_input length must match input length"
    );

    let fd_eps = 1e-3_f32;
    check_fd_grad(&analytical_x, &x_data, fd_eps, |d| {
        conv_transpose1d_scalar_loss_full(
            d,
            &input_shape,
            &kernel_data,
            &kernel_shape,
            padding,
            output_padding,
            stride,
            1,
            1,
        )
    });

    let analytical_k = grads.get(&k_var).unwrap().to_flat_vec::<f32>().unwrap();
    check_fd_grad(&analytical_k, &kernel_data, fd_eps, |k| {
        conv_transpose1d_scalar_loss_full(
            x_data.clone(),
            &input_shape,
            &k,
            &kernel_shape,
            padding,
            output_padding,
            stride,
            1,
            1,
        )
    });
}
