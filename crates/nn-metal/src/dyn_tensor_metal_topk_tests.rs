#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU topk parity tests: verify GPU kernel matches CPU results.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::{assert_close, init};

// -- Basic topk parity tests ---

#[test]
fn test_gpu_topk_k1() {
    init();
    let data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.5).collect();
    let cpu = DynTensor::new(&data, &[1, 16], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let (cpu_vals, cpu_idxs) = cpu.topk(1, 1).unwrap();
    let (gpu_vals, gpu_idxs) = gpu.topk(1, 1).unwrap();

    assert_eq!(gpu_vals.dims(), cpu_vals.dims());
    assert_eq!(gpu_idxs.dims(), cpu_idxs.dims());
    assert_eq!(gpu_vals.device(), Device::metal());

    let gv = gpu_vals
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cv = cpu_vals.to_flat_vec::<f32>().unwrap();
    assert_close(&gv, &cv, 1e-5, "topk_k1_vals");

    let gi = gpu_idxs
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<u32>()
        .unwrap();
    let ci = cpu_idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(gi, ci, "topk_k1_idxs mismatch");
}

#[test]
fn test_gpu_topk_k2() {
    init();
    let data: Vec<f32> = vec![5.0, 3.0, 8.0, 1.0, 7.0, 2.0, 9.0, 4.0];
    let cpu = DynTensor::new(&data, &[1, 8], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let (cpu_vals, cpu_idxs) = cpu.topk(1, 2).unwrap();
    let (gpu_vals, gpu_idxs) = gpu.topk(1, 2).unwrap();

    assert_eq!(gpu_vals.dims(), &[1, 2]);

    let gv = gpu_vals
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cv = cpu_vals.to_flat_vec::<f32>().unwrap();
    assert_close(&gv, &cv, 1e-5, "topk_k2_vals");

    let gi = gpu_idxs
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<u32>()
        .unwrap();
    let ci = cpu_idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(gi, ci, "topk_k2_idxs mismatch");
}

// -- Batched topk ---

#[test]
fn test_gpu_topk_batch() {
    init();
    // [2, 8] input, top-3 along last dim
    let data: Vec<f32> = vec![
        5.0, 3.0, 8.0, 1.0, 7.0, 2.0, 9.0, 4.0, // batch 0
        1.0, 6.0, 2.0, 10.0, 3.0, 5.0, 0.0, 8.0, // batch 1
    ];
    let cpu = DynTensor::new(&data, &[2, 8], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let (cpu_vals, cpu_idxs) = cpu.topk(1, 3).unwrap();
    let (gpu_vals, gpu_idxs) = gpu.topk(1, 3).unwrap();

    assert_eq!(gpu_vals.dims(), &[2, 3]);
    assert_eq!(gpu_idxs.dims(), &[2, 3]);

    let gv = gpu_vals
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cv = cpu_vals.to_flat_vec::<f32>().unwrap();
    assert_close(&gv, &cv, 1e-5, "topk_batch_vals");

    let gi = gpu_idxs
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<u32>()
        .unwrap();
    let ci = cpu_idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(gi, ci, "topk_batch_idxs mismatch");
}

// -- Larger dimension (simulates autoregressive decode) ---

#[test]
fn test_gpu_topk_large_vocab() {
    init();
    // [1, 1024] (smaller vocab for test speed), top-10
    let data: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.73).sin()).collect();
    let cpu = DynTensor::new(&data, &[1, 1024], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let (cpu_vals, cpu_idxs) = cpu.topk(1, 10).unwrap();
    let (gpu_vals, gpu_idxs) = gpu.topk(1, 10).unwrap();

    assert_eq!(gpu_vals.dims(), &[1, 10]);

    let gv = gpu_vals
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cv = cpu_vals.to_flat_vec::<f32>().unwrap();
    assert_close(&gv, &cv, 1e-5, "topk_large_vocab_vals");

    let gi = gpu_idxs
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<u32>()
        .unwrap();
    let ci = cpu_idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(gi, ci, "topk_large_vocab_idxs mismatch");
}

// -- Top-k along dim 0 (non-last dim) ---

#[test]
fn test_gpu_topk_dim0() {
    init();
    // [6, 3] input, top-2 along dim 0
    let data: Vec<f32> = (0..18).map(|i| (i as f32 * 1.1).sin()).collect();
    let cpu = DynTensor::new(&data, &[6, 3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let (cpu_vals, cpu_idxs) = cpu.topk(0, 2).unwrap();
    let (gpu_vals, gpu_idxs) = gpu.topk(0, 2).unwrap();

    assert_eq!(gpu_vals.dims(), &[2, 3]);
    assert_eq!(cpu_vals.dims(), &[2, 3]);

    let gv = gpu_vals
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cv = cpu_vals.to_flat_vec::<f32>().unwrap();
    assert_close(&gv, &cv, 1e-5, "topk_dim0_vals");

    let gi = gpu_idxs
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<u32>()
        .unwrap();
    let ci = cpu_idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(gi, ci, "topk_dim0_idxs mismatch");
}

// -- NaN rejection ---

#[test]
fn test_gpu_topk_nan_rejection() {
    init();
    let data: Vec<f32> = vec![1.0, f32::NAN, 3.0, 2.0];
    let gpu = DynTensor::new(&data, &[1, 4], &Device::Cpu)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();

    let result = gpu.topk(1, 2);
    assert!(result.is_err(), "topk should reject NaN input");
    let err = result.unwrap_err().to_string();
    assert!(err.contains("NaN"), "error should mention NaN: {err}");
}

// -- Inf values should not trigger false NaN rejection ---

#[test]
fn test_gpu_topk_inf_no_false_nan_rejection() {
    init();
    // +Inf + (-Inf) = NaN under IEEE 754, but neither value is NaN.
    // The GPU NaN check must not reject this input.
    let data: Vec<f32> = vec![1.0, f32::INFINITY, 3.0, f32::NEG_INFINITY, 2.0, 5.0];
    let cpu = DynTensor::new(&data, &[1, 6], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let (cpu_vals, cpu_idxs) = cpu.topk(1, 2).unwrap();
    let (gpu_vals, gpu_idxs) = gpu.topk(1, 2).unwrap();

    let gv = gpu_vals
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cv = cpu_vals.to_flat_vec::<f32>().unwrap();
    // Top-2 should be [+Inf, 5.0] matching CPU behavior.
    assert_eq!(gv, cv, "topk with ±Inf should match CPU");

    let gi = gpu_idxs
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<u32>()
        .unwrap();
    let ci = cpu_idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(gi, ci, "topk with ±Inf indices should match CPU");
}

// -- k > 64 falls back to CPU ---

#[test]
fn test_gpu_topk_k_over_64_cpu_fallback() {
    init();
    // Create a tensor where k=65 exceeds GPU register limit, verifying CPU fallback.
    let data: Vec<f32> = (0..128).map(|i| i as f32).collect();
    let gpu = DynTensor::new(&data, &[1, 128], &Device::Cpu)
        .unwrap()
        .to_device(&Device::metal())
        .unwrap();

    // k=65 should still work (CPU fallback), returning correct results.
    let (vals, idxs) = gpu.topk(1, 65).unwrap();
    assert_eq!(vals.dims(), &[1, 65]);
    assert_eq!(idxs.dims(), &[1, 65]);

    // Top value should be 127.0 (the maximum).
    let v = vals
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        (v[0] - 127.0).abs() < 1e-5,
        "first value should be 127.0, got {}",
        v[0]
    );
}

// -- 3D tensor topk ---

#[test]
fn test_gpu_topk_3d() {
    init();
    // [2, 3, 4] input, top-2 along last dim
    let data: Vec<f32> = (0..24).map(|i| (i as f32 * 0.91).cos()).collect();
    let cpu = DynTensor::new(&data, &[2, 3, 4], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let (cpu_vals, cpu_idxs) = cpu.topk(2, 2).unwrap();
    let (gpu_vals, gpu_idxs) = gpu.topk(2, 2).unwrap();

    assert_eq!(gpu_vals.dims(), &[2, 3, 2]);

    let gv = gpu_vals
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cv = cpu_vals.to_flat_vec::<f32>().unwrap();
    assert_close(&gv, &cv, 1e-5, "topk_3d_vals");

    let gi = gpu_idxs
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<u32>()
        .unwrap();
    let ci = cpu_idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(gi, ci, "topk_3d_idxs mismatch");
}
