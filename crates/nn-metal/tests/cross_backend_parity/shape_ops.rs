// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shape operation parity tests: CPU vs Metal.
//!
//! Tests permute, narrow (slice), pad, reshape, squeeze/unsqueeze,
//! and repeat/tile on both backends.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_prng::rand_f32_vec;
use nn_core::Device;

const TOL: f32 = 1e-6;

fn init() {
    gpu_init();
}

// -- Permute (multi-axis) --------------------------------------------------

#[test]
fn test_parity_permute() {
    init();
    let data = rand_f32_vec(800, 2 * 3 * 4 * 5, -2.0, 2.0);

    let cpu = DynTensor::new(&data, &[2, 3, 4, 5], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[2, 3, 4, 5], &Device::metal()).unwrap();

    // Reverse all axes
    let cpu_out = cpu.permute([3, 2, 1, 0]).unwrap();
    let gpu_out = gpu.permute([3, 2, 1, 0]).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[5, 4, 3, 2]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "permute");
}

// -- Transpose 3D (non-trivial axes) --------------------------------------

#[test]
fn test_parity_transpose_3d() {
    init();
    let data = rand_f32_vec(801, 2 * 6 * 8, -2.0, 2.0);

    let cpu = DynTensor::new(&data, &[2, 6, 8], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[2, 6, 8], &Device::metal()).unwrap();

    // Swap dim 0 and 2
    let cpu_out = cpu.transpose(0, 2).unwrap();
    let gpu_out = gpu.transpose(0, 2).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[8, 6, 2]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "transpose_3d");
}

// -- Narrow (slice) --------------------------------------------------------

#[test]
fn test_parity_narrow() {
    init();
    let data = rand_f32_vec(802, 4 * 16, -3.0, 3.0);

    let cpu = DynTensor::new(&data, &[4, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[4, 16], &Device::metal()).unwrap();

    // Narrow dim 1: take columns 4..12
    let cpu_out = cpu.narrow(1, 4, 8).unwrap();
    let gpu_out = gpu.narrow(1, 4, 8).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[4, 8]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "narrow");
}

// -- Narrow 3D (first axis) -----------------------------------------------

#[test]
fn test_parity_narrow_3d() {
    init();
    let data = rand_f32_vec(803, 4 * 8 * 16, -3.0, 3.0);

    let cpu = DynTensor::new(&data, &[4, 8, 16], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[4, 8, 16], &Device::metal()).unwrap();

    // Narrow dim 0: take batches 1..3
    let cpu_out = cpu.narrow(0, 1, 2).unwrap();
    let gpu_out = gpu.narrow(0, 1, 2).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[2, 8, 16]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "narrow_3d");
}

// -- Pad -------------------------------------------------------------------

#[test]
fn test_parity_pad() {
    init();
    let data = rand_f32_vec(804, 3 * 8, -2.0, 2.0);

    let cpu = DynTensor::new(&data, &[3, 8], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[3, 8], &Device::metal()).unwrap();

    // Pad last dim: 2 left, 3 right (padding format: [left_last, right_last])
    let cpu_out = cpu.pad(&[2, 3], 0.0).unwrap();
    let gpu_out = gpu.pad(&[2, 3], 0.0).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[3, 13]); // 8 + 2 + 3
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "pad");
}

// -- Reshape ---------------------------------------------------------------

#[test]
fn test_parity_reshape() {
    init();
    let data = rand_f32_vec(805, 2 * 3 * 4, -2.0, 2.0);

    let cpu = DynTensor::new(&data, &[2, 3, 4], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[2, 3, 4], &Device::metal()).unwrap();

    let cpu_out = cpu.reshape([6, 4]).unwrap();
    let gpu_out = gpu.reshape([6, 4]).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[6, 4]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "reshape");
}

// -- Repeat ----------------------------------------------------------------

#[test]
fn test_parity_repeat() {
    init();
    let data = rand_f32_vec(806, 3 * 4, -2.0, 2.0);

    let cpu = DynTensor::new(&data, &[3, 4], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[3, 4], &Device::metal()).unwrap();

    // Repeat: 2x along dim 0, 3x along dim 1
    let cpu_out = cpu.repeat([2, 3]).unwrap();
    let gpu_out = gpu.repeat([2, 3]).unwrap();

    assert_eq!(gpu_out.device(), Device::metal());
    assert_eq!(gpu_out.dims(), &[6, 12]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "repeat");
}

// -- Unsqueeze/squeeze round-trip -----------------------------------------

#[test]
fn test_parity_unsqueeze_squeeze() {
    init();
    let data = rand_f32_vec(807, 3 * 4, -2.0, 2.0);

    let cpu = DynTensor::new(&data, &[3, 4], &Device::Cpu).unwrap();
    let gpu = DynTensor::new(&data, &[3, 4], &Device::metal()).unwrap();

    // Unsqueeze dim 1: [3, 4] -> [3, 1, 4]
    let cpu_unsq = cpu.unsqueeze(1).unwrap();
    let gpu_unsq = gpu.unsqueeze(1).unwrap();

    assert_eq!(gpu_unsq.dims(), &[3, 1, 4]);
    assert_eq!(gpu_unsq.dims(), cpu_unsq.dims());
    assert_gpu_cpu_close(&gpu_unsq, &cpu_unsq, TOL, "unsqueeze");

    // Squeeze back: [3, 1, 4] -> [3, 4]
    let cpu_sq = cpu_unsq.squeeze(1).unwrap();
    let gpu_sq = gpu_unsq.squeeze(1).unwrap();

    assert_eq!(gpu_sq.dims(), &[3, 4]);
    assert_eq!(gpu_sq.dims(), cpu_sq.dims());
    assert_gpu_cpu_close(&gpu_sq, &cpu_sq, TOL, "squeeze");
}

// -- Cat 3D along middle dim -----------------------------------------------

#[test]
fn test_parity_cat_3d_middle() {
    init();
    let a_data = rand_f32_vec(808, 2 * 4 * 8, -1.0, 1.0);
    let b_data = rand_f32_vec(809, 2 * 6 * 8, -1.0, 1.0);

    let a_cpu = DynTensor::new(&a_data, &[2, 4, 8], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[2, 6, 8], &Device::Cpu).unwrap();
    let cpu_out = DynTensor::cat(&[&a_cpu, &b_cpu], 1).unwrap();

    let a_gpu = DynTensor::new(&a_data, &[2, 4, 8], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[2, 6, 8], &Device::metal()).unwrap();
    let gpu_out = DynTensor::cat(&[&a_gpu, &b_gpu], 1).unwrap();

    assert_eq!(gpu_out.dims(), &[2, 10, 8]);
    assert_eq!(gpu_out.dims(), cpu_out.dims());
    assert_gpu_cpu_close(&gpu_out, &cpu_out, TOL, "cat_3d_middle");
}
