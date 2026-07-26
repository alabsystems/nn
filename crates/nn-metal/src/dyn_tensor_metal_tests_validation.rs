#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Validation tests for Metal DynTensor GPU backend.
//!
//! Extracted from `dyn_tensor_metal_tests.rs` (#1341).
//! Covers BF16 dtype acceptance (#1659), non-float rejection, and GPU
//! finiteness validation (#1320).

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

use crate::test_common::init;

// -- BF16 dtype acceptance tests (#1659, #1646 D7/D8) -------------------------
//
// dispatch_def accepts F32/BF16/F16 — Metal buffers store f16 (2 bytes/element)
// for bf16/f16 dtypes (#1646 D7). MSL kernels use `half` buffer types with
// `float` accumulators. These tests verify bf16-tagged GPU tensors pass through
// the full cpu_to_gpu → dispatch_def → gpu_to_cpu round-trip.

/// Helper: create a GPU DynTensor with BF16 dtype via cpu_to_gpu (f16 Metal buffer).
///
/// Uses the production cpu_to_gpu() path which creates 2-byte f16 Metal buffers
/// for bf16 tensors (#1646 D7). MSL kernels read `half*` and accumulate in `float`.
fn make_bf16_gpu_tensor(shape: &[usize]) -> DynTensor {
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = vec![0.0f32; numel];
    let cpu = DynTensor::new(&data, shape, &Device::Cpu).unwrap();
    let bf16_cpu = cpu.to_dtype(DType::BF16).unwrap();
    bf16_cpu.to_device(&Device::metal()).unwrap()
}

/// Helper: create a GPU DynTensor with BF16 dtype and specific f32 values.
fn make_bf16_gpu_tensor_with_values(data: &[f32], shape: &[usize]) -> DynTensor {
    let cpu = DynTensor::new(data, shape, &Device::Cpu).unwrap();
    let bf16_cpu = cpu.to_dtype(DType::BF16).unwrap();
    bf16_cpu.to_device(&Device::metal()).unwrap()
}

#[test]
fn test_bf16_binary_accepted() {
    init();
    let a = make_bf16_gpu_tensor(&[4]);
    let b = make_bf16_gpu_tensor(&[4]);
    let result = a
        .add(&b)
        .expect("bf16 binary op should succeed via dispatch_def");
    assert_eq!(result.dtype(), DType::BF16, "output preserves bf16 dtype");
    assert_eq!(result.dims(), &[4]);
}

#[test]
fn test_bf16_unary_accepted() {
    init();
    let a = make_bf16_gpu_tensor(&[4]);
    let result = a
        .relu()
        .expect("bf16 unary op should succeed via dispatch_def");
    assert_eq!(result.dtype(), DType::BF16);
}

#[test]
fn test_bf16_reduce_accepted() {
    init();
    let a = make_bf16_gpu_tensor(&[2, 3]);
    let result = a
        .sum_keepdim(1)
        .expect("bf16 reduce should succeed via dispatch_def");
    assert_eq!(result.dtype(), DType::BF16);
    assert_eq!(result.dims(), &[2, 1]);
}

#[test]
fn test_bf16_matmul_accepted() {
    init();
    let a = make_bf16_gpu_tensor(&[2, 3]);
    let b = make_bf16_gpu_tensor(&[3, 2]);
    let result = a
        .matmul(&b)
        .expect("bf16 matmul should succeed via dispatch_def");
    assert_eq!(result.dtype(), DType::BF16);
    assert_eq!(result.dims(), &[2, 2]);
}

#[test]
fn test_bf16_max_reduce_accepted() {
    init();
    let a = make_bf16_gpu_tensor(&[2, 3]);
    let result = a
        .max_keepdim(1)
        .expect("bf16 max reduce should succeed via dispatch_def");
    assert_eq!(result.dtype(), DType::BF16);
    assert_eq!(result.dims(), &[2, 1]);
}

// -- BF16 round-trip value verification (#1646 D7/D8) -------------------------
//
// These tests verify that bf16 tensors produce correct numerical results
// through the full round-trip: CPU bf16 → f16 Metal buffer → GPU compute
// (half in, float accumulators, half out) → f16 Metal buffer → CPU readback.

#[test]
fn test_bf16_add_round_trip_values() {
    init();
    let a = make_bf16_gpu_tensor_with_values(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let b = make_bf16_gpu_tensor_with_values(&[0.5, 1.5, 2.5, 3.5], &[4]);
    let result = a.add(&b).expect("bf16 add should succeed");
    assert_eq!(result.dtype(), DType::BF16);
    let cpu = result.to_device(&Device::Cpu).unwrap();
    let vals = cpu.to_flat_vec::<f32>().unwrap();
    let expected: &[f32] = &[1.5, 3.5, 5.5, 7.5];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 0.05,
            "bf16 add[{i}]: got {got}, expected {exp}"
        );
    }
}

#[test]
fn test_bf16_matmul_round_trip_values() {
    init();
    let a = make_bf16_gpu_tensor_with_values(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = make_bf16_gpu_tensor_with_values(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let result = a.matmul(&b).expect("bf16 matmul should succeed");
    assert_eq!(result.dtype(), DType::BF16);
    assert_eq!(result.dims(), &[2, 2]);
    let cpu = result.to_device(&Device::Cpu).unwrap();
    let vals = cpu.to_flat_vec::<f32>().unwrap();
    // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    let expected: &[f32] = &[19.0, 22.0, 43.0, 50.0];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 0.5,
            "bf16 matmul[{i}]: got {got}, expected {exp}"
        );
    }
}

#[test]
fn test_bf16_softmax_round_trip_values() {
    init();
    let a = make_bf16_gpu_tensor_with_values(&[1.0, 2.0, 3.0], &[1, 3]);
    let result = a.softmax(1).expect("bf16 softmax should succeed");
    assert_eq!(result.dtype(), DType::BF16);
    let cpu = result.to_device(&Device::Cpu).unwrap();
    let vals = cpu.to_flat_vec::<f32>().unwrap();
    // softmax([1,2,3]) ≈ [0.0900, 0.2447, 0.6652]
    assert!(vals[0] > 0.05 && vals[0] < 0.15, "softmax[0]={}", vals[0]);
    assert!(vals[1] > 0.15 && vals[1] < 0.35, "softmax[1]={}", vals[1]);
    assert!(vals[2] > 0.55 && vals[2] < 0.75, "softmax[2]={}", vals[2]);
    let sum: f32 = vals.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.05,
        "softmax sum should be ~1.0, got {sum}"
    );
}

#[test]
fn test_bf16_relu_round_trip_values() {
    init();
    let a = make_bf16_gpu_tensor_with_values(&[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], &[2, 3]);
    let result = a.relu().expect("bf16 relu should succeed");
    assert_eq!(result.dtype(), DType::BF16);
    assert_eq!(result.dims(), &[2, 3]);
    let cpu = result.to_device(&Device::Cpu).unwrap();
    let vals = cpu.to_flat_vec::<f32>().unwrap();
    // relu([-2,-1,0,1,2,3]) = [0,0,0,1,2,3]
    let expected: &[f32] = &[0.0, 0.0, 0.0, 1.0, 2.0, 3.0];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 0.05,
            "bf16 relu[{i}]: got {got}, expected {exp}"
        );
    }
}

#[test]
fn test_bf16_count_non_finite_f16_buffer() {
    init();
    let cpu = DynTensor::new(&[1.0f32, 2.0, 3.0, 4.0], &[4], &Device::Cpu).unwrap();
    let bf16_cpu = cpu.to_dtype(DType::BF16).unwrap();
    let gpu = bf16_cpu.to_device(&Device::metal()).unwrap();
    let count = super::MetalDynBackend::gpu_count_non_finite(&gpu).unwrap();
    assert_eq!(count, 0, "all-finite bf16 tensor should have 0 non-finite");
}

// -- Non-float dtype rejection ------------------------------------------------

#[test]
fn test_u32_dispatch_def_rejected() {
    init();
    let cpu = DynTensor::from_vec_u32(vec![1, 2, 3, 4], &[4], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();
    let err = gpu.relu();
    assert!(
        err.is_err(),
        "u32 unary op should be rejected by dispatch_def"
    );
}

// -- GPU count_non_finite (#1320) ---------------------------------------------

#[test]
fn test_gpu_count_non_finite_all_valid() {
    init();
    let cpu = DynTensor::new(&[1.0f32, 2.0, 3.0, 4.0], &[4], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();
    let count = super::MetalDynBackend::gpu_count_non_finite(&gpu).unwrap();
    assert_eq!(count, 0);
}

#[test]
fn test_gpu_count_non_finite_with_nan() {
    init();
    let cpu = DynTensor::new(&[1.0f32, f32::NAN, 3.0, f32::NAN], &[4], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();
    let count = super::MetalDynBackend::gpu_count_non_finite(&gpu).unwrap();
    assert_eq!(count, 2);
}

#[test]
fn test_gpu_count_non_finite_with_inf() {
    init();
    let cpu = DynTensor::new(&[f32::INFINITY, 2.0, f32::NEG_INFINITY], &[3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();
    let count = super::MetalDynBackend::gpu_count_non_finite(&gpu).unwrap();
    assert_eq!(count, 2);
}

#[test]
fn test_gpu_count_non_finite_mixed() {
    init();
    let cpu = DynTensor::new(
        &[1.0, f32::NAN, f32::INFINITY, 4.0, f32::NEG_INFINITY],
        &[5],
        &Device::Cpu,
    )
    .unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();
    let count = super::MetalDynBackend::gpu_count_non_finite(&gpu).unwrap();
    assert_eq!(count, 3);
}

#[test]
fn test_gpu_count_non_finite_u32_returns_zero() {
    init();
    let cpu = DynTensor::from_vec_u32(vec![1, 2, 3], &[3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();
    let count = super::MetalDynBackend::gpu_count_non_finite(&gpu).unwrap();
    assert_eq!(count, 0, "integer dtypes are always finite");
}

// -- check_output_finite GPU integration (#1320) ------------------------------

#[test]
fn test_check_output_finite_gpu_passes_valid() {
    init();
    let cpu = DynTensor::new(&[1.0f32, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();
    nn_core::layers::check_output_finite(&gpu, "test_gpu").unwrap();
}

#[test]
fn test_check_output_finite_gpu_detects_nan() {
    init();
    let cpu = DynTensor::new(&[1.0f32, f32::NAN, 3.0], &[3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();
    let err = nn_core::layers::check_output_finite(&gpu, "gpu_layer").unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("gpu_layer"),
        "error should name the layer: {msg}"
    );
}

#[test]
fn test_check_output_finite_gpu_detects_inf() {
    init();
    let cpu = DynTensor::new(&[f32::INFINITY, f32::NEG_INFINITY], &[2], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();
    let err = nn_core::layers::check_output_finite(&gpu, "gpu_inf").unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("gpu_inf"),
        "error should name the layer: {msg}"
    );
    assert!(msg.contains("2"), "error should report count=2: {msg}");
}

// -- Mixed-dtype rejection tests (#1708) ----------------------------------------
//
// dispatch_def uses a single dtype for ALL input buffers. Mixing BF16 (2-byte)
// and F32 (4-byte) buffers would silently reinterpret one with the wrong byte
// width. validate_same_float_dtype catches this at the GPU dispatch entry point.

/// Helper: create a GPU DynTensor with F32 dtype.
fn make_f32_gpu_tensor(shape: &[usize]) -> DynTensor {
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = (0..numel).map(|i| i as f32 * 0.1).collect();
    let cpu = DynTensor::new(&data, shape, &Device::Cpu).unwrap();
    cpu.to_device(&Device::metal()).unwrap()
}

#[test]
fn test_mixed_dtype_binary_bf16_f32_rejected() {
    init();
    let bf16 = make_bf16_gpu_tensor(&[4]);
    let f32_t = make_f32_gpu_tensor(&[4]);
    let err = bf16.add(&f32_t).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("mismatch") || msg.contains("Mismatch"),
        "should report dtype mismatch: {msg}"
    );
}

#[test]
fn test_mixed_dtype_matmul_bf16_f32_rejected() {
    init();
    let bf16 = make_bf16_gpu_tensor(&[2, 3]);
    let f32_t = make_f32_gpu_tensor(&[3, 2]);
    let err = bf16.matmul(&f32_t).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("mismatch") || msg.contains("Mismatch"),
        "should report dtype mismatch: {msg}"
    );
}

#[test]
fn test_same_dtype_bf16_binary_still_works() {
    init();
    let a = make_bf16_gpu_tensor(&[4]);
    let b = make_bf16_gpu_tensor(&[4]);
    let result = a.add(&b).expect("same-dtype bf16 binary should succeed");
    assert_eq!(result.dtype(), DType::BF16);
}

// -- BF16 fallback tests (extracted to dyn_tensor_metal_tests_validation_bf16_fallback.rs) --
#[path = "dyn_tensor_metal_tests_validation_bf16_fallback.rs"]
mod bf16_fallback;
