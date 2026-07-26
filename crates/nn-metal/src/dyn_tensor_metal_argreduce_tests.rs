#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU argmax/argmin tests.
//!
//! Extracted from `dyn_tensor_metal_data_ops_tests.rs` for file-size compliance.
//! Validates GPU-native argreduce kernels against CPU reference.

use nn_core::dyn_tensor::DynTensor;
use nn_core::dyn_tensor::GpuSelectionOps;
use nn_core::Device;

use crate::dyn_tensor_metal::MetalDynBackend;
use crate::test_common::init;

/// Helper: assert GPU U32 tensor values match expected.
fn assert_gpu_u32_vals(t: &DynTensor, expected: &[u32], label: &str) {
    let cpu = t.to_device(&Device::Cpu).unwrap();
    let flat = cpu.to_vec1::<u32>().unwrap();
    assert_eq!(flat.len(), expected.len(), "{label}: length mismatch");
    for (i, (g, e)) in flat.iter().zip(expected).enumerate() {
        assert_eq!(g, e, "{label} [{i}]: gpu={g}, expected={e}");
    }
}

#[test]
fn test_gpu_argmax_dim1() {
    init();
    // [[1, 3, 2], [5, 4, 6]] → argmax(dim=1) = [1, 2]
    let t = DynTensor::new(&[1.0, 3.0, 2.0, 5.0, 4.0, 6.0], &[2, 3], &Device::metal()).unwrap();
    let result = t.argmax(1).unwrap();
    assert_eq!(result.device(), Device::metal(), "argmax must stay on GPU");
    assert_eq!(result.dims(), &[2]);
    assert_eq!(result.dtype(), nn_core::DType::U32);
    assert_gpu_u32_vals(&result, &[1, 2], "argmax_dim1");
}

#[test]
fn test_gpu_argmin_dim1() {
    init();
    // [[1, 3, 2], [5, 4, 6]] → argmin(dim=1) = [0, 1]
    let t = DynTensor::new(&[1.0, 3.0, 2.0, 5.0, 4.0, 6.0], &[2, 3], &Device::metal()).unwrap();
    let result = t.argmin(1).unwrap();
    assert_eq!(result.device(), Device::metal(), "argmin must stay on GPU");
    assert_eq!(result.dims(), &[2]);
    assert_eq!(result.dtype(), nn_core::DType::U32);
    assert_gpu_u32_vals(&result, &[0, 1], "argmin_dim1");
}

#[test]
fn test_gpu_argmax_dim0() {
    init();
    // [[1, 5], [3, 4], [2, 6]] shape [3,2] → argmax(dim=0) = [1, 2]
    let t = DynTensor::new(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[3, 2], &Device::metal()).unwrap();
    let result = t.argmax(0).unwrap();
    assert_eq!(
        result.device(),
        Device::metal(),
        "argmax dim0 must stay on GPU"
    );
    assert_eq!(result.dims(), &[2]);
    assert_gpu_u32_vals(&result, &[1, 2], "argmax_dim0");
}

#[test]
fn test_gpu_argmax_rank1() {
    init();
    // [3.0, 1.0, 4.0, 1.0, 5.0] → argmax(0) = 4 (index of 5.0)
    let t = DynTensor::new(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5], &Device::metal()).unwrap();
    let result = t.argmax(0).unwrap();
    assert_eq!(
        result.device(),
        Device::metal(),
        "argmax rank1 must stay on GPU"
    );
    assert_eq!(
        result.dims(),
        &[] as &[usize],
        "scalar output for rank-1 argmax"
    );
    let cpu = result.to_device(&Device::Cpu).unwrap();
    let val = cpu.to_scalar::<u32>().unwrap();
    assert_eq!(val, 4, "argmax rank1");
}

#[test]
fn test_gpu_argmax_cpu_parity() {
    init();
    // Verify GPU and CPU produce identical results for a 3D tensor.
    let data: Vec<f32> = (0..24).map(|i| (i as f32 * 7.0 + 3.0) % 11.0).collect();
    let gpu_t = DynTensor::new(&data, &[2, 3, 4], &Device::metal()).unwrap();
    let cpu_t = DynTensor::new(&data, &[2, 3, 4], &Device::Cpu).unwrap();

    for dim in 0..3 {
        let gpu_result = gpu_t.argmax(dim).unwrap();
        let cpu_result = cpu_t.argmax(dim).unwrap();
        let n = gpu_result.numel();
        let gpu_vals = gpu_result.reshape([n]).unwrap().to_vec1::<u32>().unwrap();
        let cpu_vals = cpu_result.reshape([n]).unwrap().to_vec1::<u32>().unwrap();
        assert_eq!(gpu_vals, cpu_vals, "argmax dim={dim} CPU/GPU mismatch");
    }
}

#[test]
fn test_gpu_argmin_cpu_parity() {
    init();
    let data: Vec<f32> = (0..24).map(|i| (i as f32 * 7.0 + 3.0) % 11.0).collect();
    let gpu_t = DynTensor::new(&data, &[2, 3, 4], &Device::metal()).unwrap();
    let cpu_t = DynTensor::new(&data, &[2, 3, 4], &Device::Cpu).unwrap();

    for dim in 0..3 {
        let gpu_result = gpu_t.argmin(dim).unwrap();
        let cpu_result = cpu_t.argmin(dim).unwrap();
        let n = gpu_result.numel();
        let gpu_vals = gpu_result.reshape([n]).unwrap().to_vec1::<u32>().unwrap();
        let cpu_vals = cpu_result.reshape([n]).unwrap().to_vec1::<u32>().unwrap();
        assert_eq!(gpu_vals, cpu_vals, "argmin dim={dim} CPU/GPU mismatch");
    }
}

#[test]
fn test_backend_argmax_cpu_tensor_returns_none() {
    let cpu = DynTensor::new(&[1.0, 3.0, 2.0, 4.0], &[2, 2], &Device::Cpu).unwrap();
    assert!(
        <MetalDynBackend as GpuSelectionOps>::argmax(&MetalDynBackend, &cpu, 1).is_none(),
        "Metal backend must decline CPU tensors so DynTensor can use CPU fallback",
    );
}

#[test]
fn test_backend_argmin_cpu_tensor_returns_none() {
    let cpu = DynTensor::new(&[1.0, 3.0, 2.0, 4.0], &[2, 2], &Device::Cpu).unwrap();
    assert!(
        <MetalDynBackend as GpuSelectionOps>::argmin(&MetalDynBackend, &cpu, 1).is_none(),
        "Metal backend must decline CPU tensors so DynTensor can use CPU fallback",
    );
}
