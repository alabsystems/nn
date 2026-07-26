#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU division semantics tests (#1180, #1147).
//!
//! GPU division follows IEEE 754: x/0 = Inf, 0/0 = NaN. No per-op finiteness
//! check on GPU (removed for performance, #1147). CPU catches div-by-zero via
//! check_div_result_finite(). Model-level NaN guards (#941, #958) catch
//! non-finite values at forward-pass stage boundaries.
//!
//! Split from nn_gpu_forward_ops.rs for 500-line limit.

use super::test_utils::{assert_gpu_cpu_close, gpu_init};
use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

const TOL: f32 = 1e-4;

fn init() {
    gpu_init();
}

fn assert_close(gpu_result: &DynTensor, cpu_result: &DynTensor, label: &str) {
    assert_gpu_cpu_close(gpu_result, cpu_result, TOL, label);
}

#[test]
fn test_gpu_div_by_zero_produces_ieee754() {
    // GPU div by zero follows IEEE 754: x/0 = Inf (not error).
    init();
    let a = DynTensor::new(&[5.0f32], &[1], &Device::metal()).unwrap();
    let b = DynTensor::new(&[0.0f32], &[1], &Device::metal()).unwrap();
    let result = a.div(&b).unwrap();
    let vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        vals[0].is_infinite() && vals[0] > 0.0,
        "x/0 should be +Inf, got {}",
        vals[0]
    );
}

#[test]
fn test_gpu_zero_div_zero_produces_nan() {
    // GPU 0/0 follows IEEE 754: 0/0 = NaN (not error).
    init();
    let a = DynTensor::new(&[0.0f32], &[1], &Device::metal()).unwrap();
    let b = DynTensor::new(&[0.0f32], &[1], &Device::metal()).unwrap();
    let result = a.div(&b).unwrap();
    let vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(vals[0].is_nan(), "0/0 should be NaN, got {}", vals[0]);
}

#[test]
fn test_gpu_div_normal_succeeds() {
    // Normal GPU division should still work.
    init();
    let a_data = vec![6.0f32, 8.0, 10.0];
    let a_cpu = DynTensor::new(&a_data, &[3], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&[2.0f32, 4.0, 5.0], &[3], &Device::Cpu).unwrap();
    let c_cpu = a_cpu.broadcast_div(&b_cpu).unwrap();

    let a_gpu = DynTensor::new(&a_data, &[3], &Device::metal()).unwrap();
    let b_gpu = DynTensor::new(&[2.0f32, 4.0, 5.0], &[3], &Device::metal()).unwrap();
    let c_gpu = a_gpu.broadcast_div(&b_gpu).unwrap();

    assert_eq!(c_gpu.device(), Device::metal());
    assert_close(&c_gpu, &c_cpu, "gpu_div_normal");
}

#[test]
fn test_cpu_div_by_zero_returns_error() {
    // CPU div by zero returns error (check_div_result_finite guard).
    init();
    let a = DynTensor::new(&[1.0f32, 2.0], &[2], &Device::Cpu).unwrap();
    let b = DynTensor::new(&[0.0f32, 0.0], &[2], &Device::Cpu).unwrap();
    let result = a.broadcast_div(&b);
    assert!(result.is_err(), "CPU div by zero should error");
}

#[test]
fn test_gpu_broadcast_div_by_zero_produces_inf() {
    // GPU broadcast_div by zero produces Inf (IEEE 754), not error.
    init();
    let a = DynTensor::new(&[1.0f32, 2.0, 3.0], &[3], &Device::metal()).unwrap();
    let b = DynTensor::new(&[0.0f32, 1.0, 0.0], &[3], &Device::metal()).unwrap();
    let result = a.broadcast_div(&b).unwrap();
    let vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(vals[0].is_infinite(), "1/0 should be Inf, got {}", vals[0]);
    assert!(
        (vals[1] - 2.0).abs() < 1e-6,
        "2/1 should be 2.0, got {}",
        vals[1]
    );
    assert!(vals[2].is_infinite(), "3/0 should be Inf, got {}", vals[2]);
}
