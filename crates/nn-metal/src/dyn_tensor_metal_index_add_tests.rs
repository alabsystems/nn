#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU/CPU parity tests for `index_add`.
//!
//! `index_add` accumulates values from `src` into `self` along `dim` using a
//! 1-D U32 index tensor.  Each test runs the operation on both CPU and GPU and
//! compares results within tolerance.
//!
//! Issue: #1949

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::{assert_gpu_vals, init};

// =============================================================================
// index_add tests
// =============================================================================

#[test]
fn test_gpu_index_add_basic_1d() {
    init();
    // base: [0, 0, 0, 0, 0] shape [5]
    // src:  [10, 20, 30] shape [3]
    // index: [1, 3, 1] (1-D, length = src.dims()[dim=0] = 3)
    // => [0, 10+30, 0, 20, 0] = [0, 40, 0, 20, 0]
    let base = DynTensor::zeros(&[5], nn_core::DType::F32, &Device::metal()).unwrap();
    let src = DynTensor::new(&[10.0, 20.0, 30.0], &[3], &Device::metal()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![1, 3, 1], &[3], &Device::Cpu).unwrap();

    let cpu_base = base.to_device(&Device::Cpu).unwrap();
    let cpu_src = src.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_base.index_add(0, &ids, &cpu_src).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = base.index_add(0, &ids, &src).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-5, "index_add_basic_1d");
}

#[test]
fn test_gpu_index_add_2d_dim0() {
    init();
    // base: [[0,0],[0,0],[0,0]] shape [3,2]
    // src:  [[1,2],[3,4]] shape [2,2]
    // index: [2, 0] (1-D, length = src.dims()[dim=0] = 2)
    // => row 0 += src[1] = [3,4]; row 2 += src[0] = [1,2]
    // => [[3,4],[0,0],[1,2]]
    let base = DynTensor::zeros(&[3, 2], nn_core::DType::F32, &Device::metal()).unwrap();
    let src = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::metal()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![2, 0], &[2], &Device::Cpu).unwrap();

    let cpu_base = base.to_device(&Device::Cpu).unwrap();
    let cpu_src = src.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_base.index_add(0, &ids, &cpu_src).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = base.index_add(0, &ids, &src).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-5, "index_add_2d_dim0");
}

#[test]
fn test_gpu_index_add_2d_dim1() {
    init();
    // base: [[0,0,0],[0,0,0]] shape [2,3]
    // src:  [[10,20],[30,40]] shape [2,2]
    // index: [0, 2] (1-D, length = src.dims()[dim=1] = 2)
    // => col 0 += src[..][0]; col 2 += src[..][1]
    // => [[10,0,20],[30,0,40]]
    let base = DynTensor::zeros(&[2, 3], nn_core::DType::F32, &Device::metal()).unwrap();
    let src = DynTensor::new(&[10.0, 20.0, 30.0, 40.0], &[2, 2], &Device::metal()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 2], &[2], &Device::Cpu).unwrap();

    let cpu_base = base.to_device(&Device::Cpu).unwrap();
    let cpu_src = src.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_base.index_add(1, &ids, &cpu_src).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = base.index_add(1, &ids, &src).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-5, "index_add_2d_dim1");
}

#[test]
fn test_gpu_index_add_3d() {
    init();
    // base: shape [2,3,2] all zeros
    // src:  shape [2,2,2], values 1..8
    // index: [0, 2] (dim=1, length = src.dims()[1] = 2)
    let base = DynTensor::zeros(&[2, 3, 2], nn_core::DType::F32, &Device::metal()).unwrap();
    let src_vals: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let src = DynTensor::new(&src_vals, &[2, 2, 2], &Device::metal()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 2], &[2], &Device::Cpu).unwrap();

    let cpu_base = base.to_device(&Device::Cpu).unwrap();
    let cpu_src = src.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_base.index_add(1, &ids, &cpu_src).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = base.index_add(1, &ids, &src).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-5, "index_add_3d");
}

#[test]
fn test_gpu_index_add_accumulate() {
    init();
    // Test that duplicate indices accumulate correctly.
    // base: [100, 200, 300] shape [3]
    // src:  [1, 2, 3, 4] shape [4]
    // index: [0, 0, 0, 2] — index 0 gets 1+2+3=6 added, index 2 gets 4
    // => [106, 200, 304]
    let base = DynTensor::new(&[100.0, 200.0, 300.0], &[3], &Device::metal()).unwrap();
    let src = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[4], &Device::metal()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 0, 0, 2], &[4], &Device::Cpu).unwrap();

    let cpu_base = base.to_device(&Device::Cpu).unwrap();
    let cpu_src = src.to_device(&Device::Cpu).unwrap();
    let cpu_result = cpu_base.index_add(0, &ids, &cpu_src).unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();

    let gpu_result = base.index_add(0, &ids, &src).unwrap();
    assert_gpu_vals(&gpu_result, &cpu_vals, 1e-5, "index_add_accumulate");
}

#[test]
fn test_gpu_index_add_preserves_device() {
    init();
    let base = DynTensor::zeros(&[4], nn_core::DType::F32, &Device::metal()).unwrap();
    let src = DynTensor::new(&[1.0, 2.0], &[2], &Device::metal()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 3], &[2], &Device::Cpu).unwrap();

    let result = base.index_add(0, &ids, &src).unwrap();
    assert!(
        result.device().is_gpu(),
        "index_add must preserve GPU device"
    );
}

/// GPU index_add with an OOB index must return an error, not silently skip.
#[test]
fn test_gpu_index_add_oob_returns_error() {
    init();
    let base = DynTensor::zeros(&[3], nn_core::DType::F32, &Device::metal()).unwrap();
    let src = DynTensor::new(&[1.0], &[1], &Device::metal()).unwrap();
    // Index 5 is out of bounds for dim size 3.
    let ids = DynTensor::from_vec_u32(vec![5], &[1], &Device::Cpu).unwrap();

    let result = base.index_add(0, &ids, &src);
    assert!(result.is_err(), "OOB index_add must return Err");
}
