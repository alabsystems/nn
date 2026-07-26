// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Convolution parity tests: CPU vs Metal.
//!
//! Tests Conv1d, Conv2d, and ConvTranspose1d on both backends.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{
    Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module,
};
use nn_core::test_prng::rand_f32_vec;
use nn_core::Device;

const TOL: f32 = 1e-4;

fn init() {
    gpu_init();
}

// -- Conv1d (stride=1, no padding) -----------------------------------------

#[test]
fn test_parity_conv1d() {
    init();
    let in_ch = 4;
    let out_ch = 8;
    let kernel = 3;
    let length = 32;
    let batch = 1;

    let x_data = rand_f32_vec(100, batch * in_ch * length, -1.0, 1.0);
    let w_data = rand_f32_vec(101, out_ch * in_ch * kernel, -0.5, 0.5);
    let b_data = rand_f32_vec(102, out_ch, -0.1, 0.1);

    let config = Conv1dConfig::new(0, 1, 1);

    // CPU
    let w_cpu = DynTensor::new(&w_data, &[out_ch, in_ch, kernel], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[out_ch], &Device::Cpu).unwrap();
    let x_cpu = DynTensor::new(&x_data, &[batch, in_ch, length], &Device::Cpu).unwrap();
    let conv_cpu = Conv1d::new(w_cpu, Some(b_cpu), config).unwrap();
    let y_cpu = conv_cpu.forward(&x_cpu).unwrap();

    // GPU
    let w_gpu = DynTensor::new(&w_data, &[out_ch, in_ch, kernel], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[out_ch], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[batch, in_ch, length], &Device::metal()).unwrap();
    let conv_gpu = Conv1d::new(w_gpu, Some(b_gpu), config).unwrap();
    let y_gpu = conv_gpu.forward(&x_gpu).unwrap();

    assert_eq!(y_gpu.dims(), y_cpu.dims());
    assert_gpu_cpu_close(&y_gpu, &y_cpu, TOL, "conv1d");
}

// -- Conv1d with padding and stride ----------------------------------------

#[test]
fn test_parity_conv1d_padded() {
    init();
    let in_ch = 4;
    let out_ch = 8;
    let kernel = 5;
    let length = 64;
    let batch = 2;
    let padding = 2;
    let stride = 2;

    let x_data = rand_f32_vec(103, batch * in_ch * length, -1.0, 1.0);
    let w_data = rand_f32_vec(104, out_ch * in_ch * kernel, -0.5, 0.5);

    let config = Conv1dConfig::new(padding, stride, 1);

    let w_cpu = DynTensor::new(&w_data, &[out_ch, in_ch, kernel], &Device::Cpu).unwrap();
    let x_cpu = DynTensor::new(&x_data, &[batch, in_ch, length], &Device::Cpu).unwrap();
    let conv_cpu = Conv1d::new(w_cpu, None, config).unwrap();
    let y_cpu = conv_cpu.forward(&x_cpu).unwrap();

    let w_gpu = DynTensor::new(&w_data, &[out_ch, in_ch, kernel], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[batch, in_ch, length], &Device::metal()).unwrap();
    let conv_gpu = Conv1d::new(w_gpu, None, config).unwrap();
    let y_gpu = conv_gpu.forward(&x_gpu).unwrap();

    assert_eq!(y_gpu.dims(), y_cpu.dims());
    assert_gpu_cpu_close(&y_gpu, &y_cpu, TOL, "conv1d_padded");
}

// -- Conv2d ----------------------------------------------------------------

#[test]
fn test_parity_conv2d() {
    init();
    let in_ch = 3;
    let out_ch = 8;
    let kh = 3;
    let kw = 3;
    let h = 16;
    let w = 16;
    let batch = 1;

    let x_data = rand_f32_vec(105, batch * in_ch * h * w, -1.0, 1.0);
    let w_data = rand_f32_vec(106, out_ch * in_ch * kh * kw, -0.5, 0.5);
    let b_data = rand_f32_vec(107, out_ch, -0.1, 0.1);

    let config = Conv2dConfig::new(1, 1, 1);

    // CPU
    let wt_cpu = DynTensor::new(&w_data, &[out_ch, in_ch, kh, kw], &Device::Cpu).unwrap();
    let bt_cpu = DynTensor::new(&b_data, &[out_ch], &Device::Cpu).unwrap();
    let x_cpu = DynTensor::new(&x_data, &[batch, in_ch, h, w], &Device::Cpu).unwrap();
    let conv_cpu = Conv2d::new(wt_cpu, Some(bt_cpu), config).unwrap();
    let y_cpu = conv_cpu.forward(&x_cpu).unwrap();

    // GPU
    let wt_gpu = DynTensor::new(&w_data, &[out_ch, in_ch, kh, kw], &Device::metal()).unwrap();
    let bt_gpu = DynTensor::new(&b_data, &[out_ch], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[batch, in_ch, h, w], &Device::metal()).unwrap();
    let conv_gpu = Conv2d::new(wt_gpu, Some(bt_gpu), config).unwrap();
    let y_gpu = conv_gpu.forward(&x_gpu).unwrap();

    assert_eq!(y_gpu.dims(), y_cpu.dims());
    assert_gpu_cpu_close(&y_gpu, &y_cpu, 5e-4, "conv2d");
}

// -- ConvTranspose1d -------------------------------------------------------

#[test]
fn test_parity_conv_transpose1d() {
    init();
    let in_ch = 8;
    let out_ch = 4;
    let kernel = 4;
    let length = 16;
    let batch = 1;
    let stride = 2;

    let x_data = rand_f32_vec(108, batch * in_ch * length, -1.0, 1.0);
    // ConvTranspose1d weight: [in_channels, out_channels, kernel_size]
    let w_data = rand_f32_vec(109, in_ch * out_ch * kernel, -0.5, 0.5);
    let b_data = rand_f32_vec(110, out_ch, -0.1, 0.1);

    let config = ConvTranspose1dConfig::new(0, stride, 1);

    // CPU
    let w_cpu = DynTensor::new(&w_data, &[in_ch, out_ch, kernel], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[out_ch], &Device::Cpu).unwrap();
    let x_cpu = DynTensor::new(&x_data, &[batch, in_ch, length], &Device::Cpu).unwrap();
    let ct_cpu = ConvTranspose1d::new(w_cpu, Some(b_cpu), config).unwrap();
    let y_cpu = ct_cpu.forward(&x_cpu).unwrap();

    // GPU
    let w_gpu = DynTensor::new(&w_data, &[in_ch, out_ch, kernel], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[out_ch], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[batch, in_ch, length], &Device::metal()).unwrap();
    let ct_gpu = ConvTranspose1d::new(w_gpu, Some(b_gpu), config).unwrap();
    let y_gpu = ct_gpu.forward(&x_gpu).unwrap();

    assert_eq!(y_gpu.dims(), y_cpu.dims());
    assert_gpu_cpu_close(&y_gpu, &y_cpu, TOL, "conv_transpose1d");
}
