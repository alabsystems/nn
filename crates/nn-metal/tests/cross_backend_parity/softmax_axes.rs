// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended softmax parity tests: CPU vs Metal.
//!
//! Tests softmax and log_softmax along different axes and with
//! various tensor ranks (2D, 3D, 4D).

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_prng::rand_f32_vec;
use nn_core::Device;

const TOL: f32 = 1e-4;

fn init() {
    gpu_init();
}

// -- Softmax 3D over last axis ---------------------------------------------

#[test]
fn test_parity_softmax_3d_last() {
    init();
    let data = rand_f32_vec(1100, 2 * 8 * 16, -3.0, 3.0);

    let cpu = DynTensor::new(&data, &[2, 8, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[2, 8, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.softmax(2).unwrap();
    let gpu_out = gpu.softmax(2).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[2, 8, 16]);
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "softmax_3d_last");
}

// -- Softmax 3D over middle axis -------------------------------------------

#[test]
fn test_parity_softmax_3d_middle() {
    init();
    let data = rand_f32_vec(1101, 2 * 8 * 16, -3.0, 3.0);

    let cpu = DynTensor::new(&data, &[2, 8, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[2, 8, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.softmax(1).unwrap();
    let gpu_out = gpu.softmax(1).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[2, 8, 16]);
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "softmax_3d_middle");
}

// -- Softmax 2D over first axis --------------------------------------------

#[test]
fn test_parity_softmax_2d_first() {
    init();
    let data = rand_f32_vec(1102, 10 * 12, -3.0, 3.0);

    let cpu = DynTensor::new(&data, &[10, 12], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[10, 12], &Device::metal()).unwrap();

    let cpu_out = cpu.softmax(0).unwrap();
    let gpu_out = gpu.softmax(0).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[10, 12]);
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "softmax_2d_first");
}

// -- Log softmax 3D over last axis -----------------------------------------

#[test]
fn test_parity_log_softmax_3d() {
    init();
    let data = rand_f32_vec(1103, 2 * 8 * 16, -3.0, 3.0);

    let cpu = DynTensor::new(&data, &[2, 8, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[2, 8, 16], &Device::metal()).unwrap();

    let cpu_out = cpu.log_softmax(2).unwrap();
    let gpu_out = gpu.log_softmax(2).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[2, 8, 16]);
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "log_softmax_3d");
}

// -- Softmax large last dim ------------------------------------------------

#[test]
fn test_parity_softmax_large_vocab() {
    init();
    // Simulates logits over large vocabulary
    let data = rand_f32_vec(1104, 4 * 512, -5.0, 5.0);

    let cpu = DynTensor::new(&data, &[4, 512], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[4, 512], &Device::metal()).unwrap();

    let cpu_out = cpu.softmax(1).unwrap();
    let gpu_out = gpu.softmax(1).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[4, 512]);
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "softmax_large_vocab");
}
