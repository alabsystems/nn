#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU softmax BF16/F16 auto-upcast and +inf log_softmax tests.
//!
//! Extracted from `dyn_tensor_metal_shape_ops_softmax_tests.rs` to keep
//! both files under 500 lines.

use nn_core::dyn_tensor::{softmax_last_dim, DynTensor};
use nn_core::Device;

use crate::test_common::{assert_close, init};

// -- BF16/F16 auto-upcast softmax tests (#1813) ------------------------------
// Verify BF16/F16 GPU tensors auto-upcast to F32 for softmax computation,
// then convert back. Output dtype must match input dtype.

/// BF16 GPU softmax: output is BF16, values match F32 reference within BF16 tolerance.
#[test]
fn test_gpu_softmax_bf16_auto_upcast() {
    init();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0];
    let f32_cpu = DynTensor::new(&data, &[2, 3], &Device::Cpu).unwrap();
    let f32_ref = softmax_last_dim(&f32_cpu).unwrap();
    let f32_ref_vals = f32_ref.to_flat_vec::<f32>().unwrap();

    let bf16_cpu = f32_cpu.to_dtype(nn_core::DType::BF16).unwrap();
    let bf16_gpu = bf16_cpu.to_device(&Device::metal()).unwrap();
    let bf16_result = softmax_last_dim(&bf16_gpu).unwrap();

    assert_eq!(
        bf16_result.dtype(),
        nn_core::DType::BF16,
        "softmax output should preserve BF16 dtype"
    );
    assert_eq!(bf16_result.device(), Device::metal());

    let result_vals = bf16_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(nn_core::DType::F32)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    // BF16 has ~7-bit mantissa: tolerance ~1e-2
    assert_close(&result_vals, &f32_ref_vals, 1e-2, "softmax_bf16_gpu");
}

/// BF16 GPU log_softmax: output is BF16, values match F32 reference.
#[test]
fn test_gpu_log_softmax_bf16_auto_upcast() {
    init();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0];
    let f32_cpu = DynTensor::new(&data, &[2, 3], &Device::Cpu).unwrap();
    let f32_ref = f32_cpu
        .log_softmax(nn_core::dyn_tensor::D::Minus1)
        .unwrap();
    let f32_ref_vals = f32_ref.to_flat_vec::<f32>().unwrap();

    let bf16_cpu = f32_cpu.to_dtype(nn_core::DType::BF16).unwrap();
    let bf16_gpu = bf16_cpu.to_device(&Device::metal()).unwrap();
    let bf16_result = bf16_gpu
        .log_softmax(nn_core::dyn_tensor::D::Minus1)
        .unwrap();

    assert_eq!(
        bf16_result.dtype(),
        nn_core::DType::BF16,
        "log_softmax output should preserve BF16 dtype"
    );

    let result_vals = bf16_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(nn_core::DType::F32)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&result_vals, &f32_ref_vals, 1e-2, "log_softmax_bf16_gpu");
}

/// F16 GPU softmax: output is F16, values match F32 reference within F16 tolerance.
#[test]
fn test_gpu_softmax_f16_auto_upcast() {
    init();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0];
    let f32_cpu = DynTensor::new(&data, &[2, 3], &Device::Cpu).unwrap();
    let f32_ref = softmax_last_dim(&f32_cpu).unwrap();
    let f32_ref_vals = f32_ref.to_flat_vec::<f32>().unwrap();

    let f16_cpu = f32_cpu.to_dtype(nn_core::DType::F16).unwrap();
    let f16_gpu = f16_cpu.to_device(&Device::metal()).unwrap();
    let f16_result = softmax_last_dim(&f16_gpu).unwrap();

    assert_eq!(
        f16_result.dtype(),
        nn_core::DType::F16,
        "softmax output should preserve F16 dtype"
    );
    assert_eq!(f16_result.device(), Device::metal());

    let result_vals = f16_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(nn_core::DType::F32)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    // F16 has ~10-bit mantissa: tolerance ~1e-3
    assert_close(&result_vals, &f32_ref_vals, 1e-3, "softmax_f16_gpu");
}

/// F16 GPU log_softmax: output is F16, values match F32 reference within F16 tolerance.
#[test]
fn test_gpu_log_softmax_f16_auto_upcast() {
    init();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0];
    let f32_cpu = DynTensor::new(&data, &[2, 3], &Device::Cpu).unwrap();
    let f32_ref = f32_cpu
        .log_softmax(nn_core::dyn_tensor::D::Minus1)
        .unwrap();
    let f32_ref_vals = f32_ref.to_flat_vec::<f32>().unwrap();

    let f16_cpu = f32_cpu.to_dtype(nn_core::DType::F16).unwrap();
    let f16_gpu = f16_cpu.to_device(&Device::metal()).unwrap();
    let f16_result = f16_gpu
        .log_softmax(nn_core::dyn_tensor::D::Minus1)
        .unwrap();

    assert_eq!(
        f16_result.dtype(),
        nn_core::DType::F16,
        "log_softmax output should preserve F16 dtype"
    );
    assert_eq!(f16_result.device(), Device::metal());

    let result_vals = f16_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(nn_core::DType::F32)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    // F16 has ~10-bit mantissa: tolerance ~1e-3
    assert_close(&result_vals, &f32_ref_vals, 1e-3, "log_softmax_f16_gpu");
}

/// GPU log_softmax with +inf input: log(1/count) at +inf positions, -inf elsewhere.
#[test]
fn test_gpu_log_softmax_single_pos_inf() {
    init();
    let data = vec![f32::INFINITY, 1.0];
    let cpu = DynTensor::new(&data, &[1, 2], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let cpu_result = cpu.log_softmax(nn_core::dyn_tensor::D::Minus1).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    // CPU expected: [0.0, -inf] (log(1) = 0, log(0) = -inf)
    assert!(
        cpu_vals[0].abs() < 1e-6,
        "CPU log_softmax +inf should be 0.0, got {}",
        cpu_vals[0]
    );
    assert!(
        cpu_vals[1] == f32::NEG_INFINITY,
        "CPU log_softmax non-inf should be -inf, got {}",
        cpu_vals[1]
    );

    let gpu_result = gpu.log_softmax(nn_core::dyn_tensor::D::Minus1).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        gpu_vals[0].abs() < 1e-6,
        "GPU log_softmax +inf should be 0.0, got {}",
        gpu_vals[0]
    );
    assert!(
        gpu_vals[1] == f32::NEG_INFINITY,
        "GPU log_softmax non-inf should be -inf, got {}",
        gpu_vals[1]
    );
}
