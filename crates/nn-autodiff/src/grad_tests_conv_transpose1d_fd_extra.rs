#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ConvTranspose1d backward FD tests for dilation and groups.
//! Extracted from `grad_tests_conv_transpose1d_fd.rs` for 500-line compliance.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;

use crate::grad::backward;
use crate::grad::test_helpers::check_fd_grad;
use crate::tracked::TrackedTensor;
use crate::var::Var;

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

/// Dilation=2 exercises the dilated kernel backward path in backward_conv_transpose1d.
#[test]
fn test_backward_conv_transpose1d_dilation2_fd() {
    // Input: [B=1, C_in=1, L=4], Kernel: [C_in=1, C_out=1, K=3]
    // stride=1, padding=1, dilation=2, groups=1, output_padding=0
    // out_len = (4-1)*1 - 2*1 + 2*(3-1) + 0 + 1 = 3 - 2 + 4 + 1 = 6
    let x_data = vec![0.5, -0.3, 1.2, 0.7];
    let input_shape = [1, 1, 4];
    let kernel_data = vec![0.4, -0.6, 0.8];
    let kernel_shape = [1, 1, 3];
    let padding = 1;
    let output_padding = 0;
    let stride = 1;
    let dilation = 2;
    let groups = 1;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &input_shape, &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(kernel_data.clone(), &kernel_shape, &cpu()).unwrap());

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx
        .conv_transpose1d(&tk, padding, stride, dilation, groups, output_padding)
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
            dilation,
            groups,
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
            dilation,
            groups,
        )
    });
}

/// Groups=2 exercises the grouped convolution backward path.
/// Kernel shape for ConvTranspose1d with groups: [C_in, C_out/groups, K].
#[test]
fn test_backward_conv_transpose1d_groups2_fd() {
    // Input: [B=1, C_in=2, L=3], Kernel: [C_in=2, C_out_per_group=1, K=2]
    // groups=2 → C_out = C_out_per_group * groups = 2
    // stride=1, padding=0, dilation=1, output_padding=0
    // out_len = (3-1)*1 - 0 + 1*(2-1) + 0 + 1 = 2 + 1 + 1 = 4
    let x_data = vec![1.0, 2.0, 3.0, 0.5, -0.5, 1.5]; // [1, 2, 3]
    let input_shape = [1, 2, 3];
    // Kernel: [2, 1, 2] — 2 input channels, 1 output per group, kernel size 2
    let kernel_data = vec![0.3, 0.7, -0.4, 0.6];
    let kernel_shape = [2, 1, 2];
    let padding = 0;
    let output_padding = 0;
    let stride = 1;
    let dilation = 1;
    let groups = 2;

    let x_var = Var::new(DynTensor::from_vec(x_data.clone(), &input_shape, &cpu()).unwrap());
    let k_var = Var::new(DynTensor::from_vec(kernel_data.clone(), &kernel_shape, &cpu()).unwrap());

    let tx = Arc::new(TrackedTensor::from_var(&x_var).unwrap());
    let tk = Arc::new(TrackedTensor::from_var(&k_var).unwrap());
    let y = tx
        .conv_transpose1d(&tk, padding, stride, dilation, groups, output_padding)
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
            dilation,
            groups,
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
            dilation,
            groups,
        )
    });
}
