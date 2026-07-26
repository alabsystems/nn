// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Normalization layer parity tests: CPU vs Metal.
//!
//! Tests LayerNorm, BatchNorm, GroupNorm, and RmsNorm on both backends.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{BatchNorm, GroupNorm, LayerNorm, Module, RmsNorm};
use nn_core::test_prng::rand_f32_vec;
use nn_core::Device;

const TOL: f32 = 1e-4;

fn init() {
    gpu_init();
}

// -- LayerNorm -------------------------------------------------------------

#[test]
fn test_parity_layernorm() {
    init();
    let hidden = 64;
    let batch = 4;

    let x_data = rand_f32_vec(80, batch * hidden, -2.0, 2.0);
    let w_data = rand_f32_vec(81, hidden, 0.5, 1.5);
    let b_data = rand_f32_vec(82, hidden, -0.1, 0.1);

    // CPU
    let w_cpu = DynTensor::new(&w_data, &[hidden], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[hidden], &Device::Cpu).unwrap();
    let x_cpu = DynTensor::new(&x_data, &[batch, hidden], &Device::Cpu).unwrap();
    let ln_cpu = LayerNorm::new(w_cpu, b_cpu, 1e-5).unwrap();
    let y_cpu = ln_cpu.forward(&x_cpu).unwrap();

    // GPU
    let w_gpu = DynTensor::new(&w_data, &[hidden], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[hidden], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[batch, hidden], &Device::metal()).unwrap();
    let ln_gpu = LayerNorm::new(w_gpu, b_gpu, 1e-5).unwrap();
    let y_gpu = ln_gpu.forward(&x_gpu).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dims(), &[batch, hidden]);
    assert_gpu_cpu_close(&y_gpu, &y_cpu, TOL, "layernorm");
}

// -- LayerNorm 3D ----------------------------------------------------------

#[test]
fn test_parity_layernorm_3d() {
    init();
    let hidden = 32;
    let seq = 8;
    let batch = 2;

    let x_data = rand_f32_vec(83, batch * seq * hidden, -2.0, 2.0);
    let w_data = rand_f32_vec(84, hidden, 0.5, 1.5);
    let b_data = rand_f32_vec(85, hidden, -0.1, 0.1);

    let w_cpu = DynTensor::new(&w_data, &[hidden], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[hidden], &Device::Cpu).unwrap();
    let x_cpu = DynTensor::new(&x_data, &[batch, seq, hidden], &Device::Cpu).unwrap();
    let ln_cpu = LayerNorm::new(w_cpu, b_cpu, 1e-5).unwrap();
    let y_cpu = ln_cpu.forward(&x_cpu).unwrap();

    let w_gpu = DynTensor::new(&w_data, &[hidden], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[hidden], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[batch, seq, hidden], &Device::metal()).unwrap();
    let ln_gpu = LayerNorm::new(w_gpu, b_gpu, 1e-5).unwrap();
    let y_gpu = ln_gpu.forward(&x_gpu).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dims(), &[batch, seq, hidden]);
    assert_gpu_cpu_close(&y_gpu, &y_cpu, TOL, "layernorm_3d");
}

// -- BatchNorm (inference mode) -------------------------------------------

#[test]
fn test_parity_batchnorm() {
    init();
    let channels = 16;
    let batch = 2;
    let length = 32;

    let x_data = rand_f32_vec(86, batch * channels * length, -2.0, 2.0);
    // Running stats
    let mean_data = rand_f32_vec(87, channels, -0.5, 0.5);
    let var_data = rand_f32_vec(88, channels, 0.5, 2.0);
    let w_data = rand_f32_vec(89, channels, 0.5, 1.5);
    let b_data = rand_f32_vec(90, channels, -0.1, 0.1);

    // CPU
    let mean_cpu = DynTensor::new(&mean_data, &[channels], &Device::Cpu).unwrap();
    let var_cpu = DynTensor::new(&var_data, &[channels], &Device::Cpu).unwrap();
    let w_cpu = DynTensor::new(&w_data, &[channels], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[channels], &Device::Cpu).unwrap();
    let x_cpu = DynTensor::new(&x_data, &[batch, channels, length], &Device::Cpu).unwrap();
    let bn_cpu = BatchNorm::new(mean_cpu, var_cpu, Some(w_cpu), Some(b_cpu), 1e-5).unwrap();
    let y_cpu = bn_cpu.forward(&x_cpu).unwrap();

    // GPU
    let mean_gpu = DynTensor::new(&mean_data, &[channels], &Device::metal()).unwrap();
    let var_gpu = DynTensor::new(&var_data, &[channels], &Device::metal()).unwrap();
    let w_gpu = DynTensor::new(&w_data, &[channels], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[channels], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[batch, channels, length], &Device::metal()).unwrap();
    let bn_gpu = BatchNorm::new(mean_gpu, var_gpu, Some(w_gpu), Some(b_gpu), 1e-5).unwrap();
    let y_gpu = bn_gpu.forward(&x_gpu).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dims(), y_cpu.dims());
    assert_gpu_cpu_close(&y_gpu, &y_cpu, TOL, "batchnorm");
}

// -- GroupNorm -------------------------------------------------------------

#[test]
fn test_parity_groupnorm() {
    init();
    let channels = 16;
    let groups = 4;
    let batch = 2;
    let length = 32;

    let x_data = rand_f32_vec(91, batch * channels * length, -2.0, 2.0);
    let w_data = rand_f32_vec(92, channels, 0.5, 1.5);
    let b_data = rand_f32_vec(93, channels, -0.1, 0.1);

    // CPU
    let w_cpu = DynTensor::new(&w_data, &[channels], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&b_data, &[channels], &Device::Cpu).unwrap();
    let x_cpu = DynTensor::new(&x_data, &[batch, channels, length], &Device::Cpu).unwrap();
    let gn_cpu = GroupNorm::new(groups, channels, w_cpu, b_cpu, 1e-5).unwrap();
    let y_cpu = gn_cpu.forward(&x_cpu).unwrap();

    // GPU
    let w_gpu = DynTensor::new(&w_data, &[channels], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&b_data, &[channels], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[batch, channels, length], &Device::metal()).unwrap();
    let gn_gpu = GroupNorm::new(groups, channels, w_gpu, b_gpu, 1e-5).unwrap();
    let y_gpu = gn_gpu.forward(&x_gpu).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dims(), y_cpu.dims());
    assert_gpu_cpu_close(&y_gpu, &y_cpu, TOL, "groupnorm");
}

// -- RmsNorm ---------------------------------------------------------------

#[test]
fn test_parity_rmsnorm() {
    init();
    let hidden = 64;
    let batch = 4;

    let x_data = rand_f32_vec(94, batch * hidden, -2.0, 2.0);
    let w_data = rand_f32_vec(95, hidden, 0.5, 1.5);

    // CPU
    let w_cpu = DynTensor::new(&w_data, &[hidden], &Device::Cpu).unwrap();
    let x_cpu = DynTensor::new(&x_data, &[batch, hidden], &Device::Cpu).unwrap();
    let rn_cpu = RmsNorm::new(w_cpu, 1e-5).unwrap();
    let y_cpu = rn_cpu.forward(&x_cpu).unwrap();

    // GPU
    let w_gpu = DynTensor::new(&w_data, &[hidden], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[batch, hidden], &Device::metal()).unwrap();
    let rn_gpu = RmsNorm::new(w_gpu, 1e-5).unwrap();
    let y_gpu = rn_gpu.forward(&x_gpu).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dims(), &[batch, hidden]);
    assert_gpu_cpu_close(&y_gpu, &y_cpu, TOL, "rmsnorm");
}

// -- RmsNorm 3D ------------------------------------------------------------

#[test]
fn test_parity_rmsnorm_3d() {
    init();
    let hidden = 32;
    let seq = 8;
    let batch = 2;

    let x_data = rand_f32_vec(96, batch * seq * hidden, -2.0, 2.0);
    let w_data = rand_f32_vec(97, hidden, 0.5, 1.5);

    let w_cpu = DynTensor::new(&w_data, &[hidden], &Device::Cpu).unwrap();
    let x_cpu = DynTensor::new(&x_data, &[batch, seq, hidden], &Device::Cpu).unwrap();
    let rn_cpu = RmsNorm::new(w_cpu, 1e-5).unwrap();
    let y_cpu = rn_cpu.forward(&x_cpu).unwrap();

    let w_gpu = DynTensor::new(&w_data, &[hidden], &Device::metal()).unwrap();
    let x_gpu = DynTensor::new(&x_data, &[batch, seq, hidden], &Device::metal()).unwrap();
    let rn_gpu = RmsNorm::new(w_gpu, 1e-5).unwrap();
    let y_gpu = rn_gpu.forward(&x_gpu).unwrap();

    assert_eq!(y_gpu.device(), Device::metal());
    assert_eq!(y_gpu.dims(), &[batch, seq, hidden]);
    assert_gpu_cpu_close(&y_gpu, &y_cpu, TOL, "rmsnorm_3d");
}
