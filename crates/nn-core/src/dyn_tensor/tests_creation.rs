// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive DynTensor creation, conversion, and factory tests.
//!
//! Covers factory methods (zeros, ones, full, arange, rand), typed constructors
//! (from_cpu_*, from_vec_*), dtype conversion roundtrips, device operations,
//! and property queries. Complements `creation_shape_tests.rs` with deeper
//! coverage of multi-dtype paths, edge cases, and storage internals.

use crate::dyn_tensor::test_helpers::{cpu, t1d, t2d, tnd};
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device};

// ============================================================================
// 1. Factory methods — multi-dtype coverage
// ============================================================================

#[test]
fn test_zeros_u8_values() {
    let t = DynTensor::zeros(&[2, 3], DType::U8, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::U8);
    assert_eq!(t.dims(), &[2, 3]);
    let data = t.to_flat_vec::<u8>().unwrap();
    assert!(data.iter().all(|&v| v == 0));
}

#[test]
fn test_zeros_3d_f32() {
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 3, 4]);
    assert_eq!(t.numel(), 24);
    let data = t.to_flat_vec::<f32>().unwrap();
    assert!(data.iter().all(|&v| v == 0.0));
}

#[test]
fn test_zeros_4d_f32() {
    let t = DynTensor::zeros(&[1, 2, 3, 4], DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims(), &[1, 2, 3, 4]);
    assert_eq!(t.numel(), 24);
}

#[test]
fn test_zeros_f16_values() {
    let t = DynTensor::zeros(&[4], DType::F16, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::F16);
    let converted = t.to_dtype(DType::F32).unwrap();
    let data = converted.to_flat_vec::<f32>().unwrap();
    assert!(data.iter().all(|&v| v == 0.0));
}

#[test]
fn test_zeros_bf16_values() {
    let t = DynTensor::zeros(&[4], DType::BF16, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::BF16);
    let converted = t.to_dtype(DType::F32).unwrap();
    let data = converted.to_flat_vec::<f32>().unwrap();
    assert!(data.iter().all(|&v| v == 0.0));
}

#[test]
fn test_zeros_i32_unsupported() {
    let r = DynTensor::zeros(&[2], DType::I32, &cpu());
    assert!(r.is_err(), "I32 zeros should fail — not supported");
}

#[test]
fn test_zeros_bool_unsupported() {
    let r = DynTensor::zeros(&[2], DType::Bool, &cpu());
    assert!(r.is_err(), "Bool zeros should fail — not supported");
}

#[test]
fn test_ones_i64_values() {
    let t = DynTensor::ones(&[3], DType::I64, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::I64);
    let data = t.to_flat_vec::<i64>().unwrap();
    assert!(data.iter().all(|&v| v == 1));
}

#[test]
fn test_ones_u32_values() {
    let t = DynTensor::ones(&[2, 2], DType::U32, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::U32);
    let data = t.to_flat_vec::<u32>().unwrap();
    assert!(data.iter().all(|&v| v == 1));
}

#[test]
fn test_ones_f16_values() {
    let t = DynTensor::ones(&[3], DType::F16, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::F16);
    let converted = t.to_dtype(DType::F32).unwrap();
    let data = converted.to_flat_vec::<f32>().unwrap();
    assert!(data.iter().all(|&v| (v - 1.0).abs() < 1e-3));
}

#[test]
fn test_ones_bf16_values() {
    let t = DynTensor::ones(&[3], DType::BF16, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::BF16);
    let converted = t.to_dtype(DType::F32).unwrap();
    let data = converted.to_flat_vec::<f32>().unwrap();
    assert!(data.iter().all(|&v| (v - 1.0).abs() < 1e-2));
}

#[test]
fn test_ones_3d() {
    let t = DynTensor::ones(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    assert_eq!(t.numel(), 24);
    let data = t.to_flat_vec::<f32>().unwrap();
    assert!(data.iter().all(|&v| v == 1.0));
}

#[test]
fn test_full_f32_negative() {
    let t = DynTensor::full(&[2, 3], -3.14, DType::F32, &cpu()).unwrap();
    let data = t.to_flat_vec::<f32>().unwrap();
    assert!(data.iter().all(|&v| (v - (-3.14f32)).abs() < 1e-5));
}

#[test]
fn test_full_u8() {
    let t = DynTensor::full(&[5], 42.0, DType::U8, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::U8);
    let data = t.to_flat_vec::<u8>().unwrap();
    assert!(data.iter().all(|&v| v == 42));
}

#[test]
fn test_full_u8_rejects_negative() {
    let r = DynTensor::full(&[2], -1.0, DType::U8, &cpu());
    assert!(r.is_err());
}

#[test]
fn test_full_u8_rejects_overflow() {
    let r = DynTensor::full(&[2], 256.0, DType::U8, &cpu());
    assert!(r.is_err());
}

#[test]
fn test_full_i64_negative() {
    let t = DynTensor::full(&[3], -100.0, DType::I64, &cpu()).unwrap();
    let data = t.to_flat_vec::<i64>().unwrap();
    assert!(data.iter().all(|&v| v == -100));
}

#[test]
fn test_full_f16() {
    let t = DynTensor::full(&[4], 2.5, DType::F16, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::F16);
    let converted = t.to_dtype(DType::F32).unwrap();
    let data = converted.to_flat_vec::<f32>().unwrap();
    assert!(data.iter().all(|&v| (v - 2.5).abs() < 0.01));
}

#[test]
fn test_full_bf16() {
    let t = DynTensor::full(&[4], 2.5, DType::BF16, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::BF16);
    let converted = t.to_dtype(DType::F32).unwrap();
    let data = converted.to_flat_vec::<f32>().unwrap();
    assert!(data.iter().all(|&v| (v - 2.5).abs() < 0.05));
}

#[test]
fn test_full_f64_demotes_to_f32() {
    let t = DynTensor::full(&[3], 1.5, DType::F64, &cpu()).unwrap();
    // F64 demoted to F32 internally
    assert_eq!(t.dtype(), DType::F32);
    let data = t.to_flat_vec::<f32>().unwrap();
    assert!(data.iter().all(|&v| (v - 1.5).abs() < 1e-6));
}

#[test]
fn test_full_i32_unsupported() {
    let r = DynTensor::full(&[2], 1.0, DType::I32, &cpu());
    assert!(r.is_err());
}

#[test]
fn test_full_bool_unsupported() {
    let r = DynTensor::full(&[2], 1.0, DType::Bool, &cpu());
    assert!(r.is_err());
}

#[test]
fn test_full_scalar_shape() {
    let t = DynTensor::full(&[], 7.0, DType::F32, &cpu()).unwrap();
    assert_eq!(t.rank(), 0);
    assert_eq!(t.numel(), 1);
    let val = t.to_scalar::<f32>().unwrap();
    assert!((val - 7.0).abs() < 1e-6);
}

#[test]
fn test_arange_step_large() {
    let t = DynTensor::arange_step(0.0, 10.0, 2.5, &cpu()).unwrap();
    let data = t.to_vec1::<f32>().unwrap();
    assert_eq!(data.len(), 4); // 0.0, 2.5, 5.0, 7.5
    assert!((data[0] - 0.0).abs() < 1e-6);
    assert!((data[1] - 2.5).abs() < 1e-6);
    assert!((data[2] - 5.0).abs() < 1e-6);
    assert!((data[3] - 7.5).abs() < 1e-6);
}

#[test]
fn test_arange_step_negative_direction() {
    let t = DynTensor::arange_step(5.0, 0.0, -1.0, &cpu()).unwrap();
    let data = t.to_vec1::<f32>().unwrap();
    assert_eq!(data.len(), 5); // 5, 4, 3, 2, 1
    assert!((data[0] - 5.0).abs() < 1e-6);
    assert!((data[4] - 1.0).abs() < 1e-6);
}

#[test]
fn test_arange_step_inf_errors() {
    let r = DynTensor::arange_step(0.0, f64::INFINITY, 1.0, &cpu());
    assert!(r.is_err());
}

// ============================================================================
// 2. Typed constructors — from_cpu_* and from_vec_*
// ============================================================================

#[test]
fn test_from_cpu_f32_basic() {
    let arr = ndarray::ArrayD::from_shape_vec(
        ndarray::IxDyn(&[2, 3]),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let t = DynTensor::from_cpu_f32(arr).unwrap();
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(t.numel(), 6);
}

#[test]
fn test_from_cpu_f16_basic() {
    let data: Vec<half::f16> = vec![1.0f32, 2.0, 3.0]
        .into_iter()
        .map(half::f16::from_f32)
        .collect();
    let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), data).unwrap();
    let t = DynTensor::from_cpu_f16(arr).unwrap();
    assert_eq!(t.dims(), &[3]);
    assert_eq!(t.dtype(), DType::F16);
}

#[test]
fn test_from_cpu_bf16_basic() {
    let data: Vec<half::bf16> = vec![1.0f32, 2.0, 3.0]
        .into_iter()
        .map(half::bf16::from_f32)
        .collect();
    let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), data).unwrap();
    let t = DynTensor::from_cpu_bf16(arr).unwrap();
    assert_eq!(t.dims(), &[3]);
    assert_eq!(t.dtype(), DType::BF16);
}

#[test]
fn test_from_cpu_u32_basic() {
    let arr =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[4]), vec![10u32, 20, 30, 40]).unwrap();
    let t = DynTensor::from_cpu_u32(arr).unwrap();
    assert_eq!(t.dtype(), DType::U32);
    assert_eq!(t.dims(), &[4]);
    let data = t.to_flat_vec::<u32>().unwrap();
    assert_eq!(data, vec![10, 20, 30, 40]);
}

#[test]
fn test_from_cpu_u8_basic() {
    let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![1u8, 2, 255]).unwrap();
    let t = DynTensor::from_cpu_u8(arr).unwrap();
    assert_eq!(t.dtype(), DType::U8);
    let data = t.to_flat_vec::<u8>().unwrap();
    assert_eq!(data, vec![1, 2, 255]);
}

#[test]
fn test_from_cpu_i64_basic() {
    let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![-100i64, 0, 100]).unwrap();
    let t = DynTensor::from_cpu_i64(arr).unwrap();
    assert_eq!(t.dtype(), DType::I64);
    let data = t.to_flat_vec::<i64>().unwrap();
    assert_eq!(data, vec![-100, 0, 100]);
}

#[test]
fn test_from_vec_f16_roundtrip() {
    let original = [1.0f32, -2.5, 3.75, 0.0];
    let f16_data: Vec<half::f16> = original.iter().map(|&v| half::f16::from_f32(v)).collect();
    let t = DynTensor::from_vec_f16(f16_data, &[2, 2], &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 2]);
    assert_eq!(t.dtype(), DType::F16);
    let back = t.to_dtype(DType::F32).unwrap();
    let data = back.to_flat_vec::<f32>().unwrap();
    for (orig, got) in original.iter().zip(data.iter()) {
        assert!((orig - got).abs() < 0.01, "expected ~{orig}, got {got}");
    }
}

#[test]
fn test_from_vec_bf16_roundtrip() {
    let original = [1.0f32, -2.5, 3.75, 0.0];
    let bf16_data: Vec<half::bf16> = original.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let t = DynTensor::from_vec_bf16(bf16_data, &[4], &cpu()).unwrap();
    assert_eq!(t.dims(), &[4]);
    assert_eq!(t.dtype(), DType::BF16);
    let back = t.to_dtype(DType::F32).unwrap();
    let data = back.to_flat_vec::<f32>().unwrap();
    for (orig, got) in original.iter().zip(data.iter()) {
        assert!((orig - got).abs() < 0.05, "expected ~{orig}, got {got}");
    }
}

#[test]
fn test_from_vec_i64() {
    let t = DynTensor::from_vec_i64(vec![10, 20, 30], &[3], &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::I64);
    let data = t.to_flat_vec::<i64>().unwrap();
    assert_eq!(data, vec![10, 20, 30]);
}

#[test]
fn test_from_vec_u8() {
    let t = DynTensor::from_vec_u8(vec![0, 128, 255], &[3], &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::U8);
    let data = t.to_flat_vec::<u8>().unwrap();
    assert_eq!(data, vec![0, 128, 255]);
}

#[test]
fn test_from_vec_u32() {
    let t = DynTensor::from_vec_u32(vec![100, 200], &[2], &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::U32);
    let data = t.to_flat_vec::<u32>().unwrap();
    assert_eq!(data, vec![100, 200]);
}

#[test]
fn test_from_vec_f16_length_mismatch() {
    let data: Vec<half::f16> = vec![half::f16::from_f32(1.0), half::f16::from_f32(2.0)];
    let r = DynTensor::from_vec_f16(data, &[3], &cpu());
    assert!(r.is_err(), "length mismatch should error");
}

#[test]
fn test_from_vec_bf16_length_mismatch() {
    let data: Vec<half::bf16> = vec![half::bf16::from_f32(1.0)];
    let r = DynTensor::from_vec_bf16(data, &[2, 2], &cpu());
    assert!(r.is_err(), "length mismatch should error");
}

#[test]
fn test_from_f32_result_f32_target() {
    let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0f32, 2.0, 3.0, 4.0])
        .unwrap();
    let t = DynTensor::from_f32_result(arr, DType::F32).unwrap();
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(t.dims(), &[2, 2]);
}

#[test]
fn test_from_f32_result_f16_target() {
    let arr =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.0f32, 2.0, 3.0]).unwrap();
    let t = DynTensor::from_f32_result(arr, DType::F16).unwrap();
    assert_eq!(t.dtype(), DType::F16);
}

#[test]
fn test_from_f32_result_bf16_target() {
    let arr =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.0f32, 2.0, 3.0]).unwrap();
    let t = DynTensor::from_f32_result(arr, DType::BF16).unwrap();
    assert_eq!(t.dtype(), DType::BF16);
}

#[test]
fn test_from_f32_result_i32_target_errors() {
    let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![1.0f32, 2.0]).unwrap();
    let r = DynTensor::from_f32_result(arr, DType::I32);
    assert!(r.is_err(), "integer target from f32 result should error");
}

// ============================================================================
// 3. Typed arange variants
// ============================================================================

#[test]
fn test_arange_u32_basic() {
    let t = DynTensor::arange_u32(0, 5, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::U32);
    assert_eq!(t.dims(), &[5]);
    let data = t.to_flat_vec::<u32>().unwrap();
    assert_eq!(data, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_arange_u32_offset() {
    let t = DynTensor::arange_u32(3, 7, &cpu()).unwrap();
    let data = t.to_flat_vec::<u32>().unwrap();
    assert_eq!(data, vec![3, 4, 5, 6]);
}

#[test]
fn test_arange_u32_empty() {
    let t = DynTensor::arange_u32(5, 5, &cpu()).unwrap();
    assert_eq!(t.dims(), &[0]);
    assert_eq!(t.numel(), 0);
}

#[test]
fn test_arange_i64_basic() {
    let t = DynTensor::arange_i64(-2, 3, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::I64);
    let data = t.to_flat_vec::<i64>().unwrap();
    assert_eq!(data, vec![-2, -1, 0, 1, 2]);
}

#[test]
fn test_arange_i64_empty() {
    let t = DynTensor::arange_i64(3, 3, &cpu()).unwrap();
    assert_eq!(t.dims(), &[0]);
}

// ============================================================================
// 4. Dtype conversion roundtrips
// ============================================================================

#[test]
fn test_to_dtype_f32_to_bf16_to_f32_roundtrip() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], &cpu()).unwrap();
    let bf16 = t.to_dtype(DType::BF16).unwrap();
    assert_eq!(bf16.dtype(), DType::BF16);
    assert_eq!(bf16.dims(), &[4]);
    let back = bf16.to_dtype(DType::F32).unwrap();
    assert_eq!(back.dtype(), DType::F32);
    let data = back.to_flat_vec::<f32>().unwrap();
    for (i, &v) in data.iter().enumerate() {
        let expected = (i + 1) as f32;
        assert!(
            (v - expected).abs() < 0.05,
            "F32→BF16→F32 roundtrip: expected ~{expected}, got {v}"
        );
    }
}

#[test]
fn test_to_dtype_f32_to_f16_to_f32_roundtrip() {
    let t = DynTensor::from_vec(vec![0.5, 1.5, -2.0, 3.25], &[4], &cpu()).unwrap();
    let f16 = t.to_dtype(DType::F16).unwrap();
    assert_eq!(f16.dtype(), DType::F16);
    let back = f16.to_dtype(DType::F32).unwrap();
    assert_eq!(back.dtype(), DType::F32);
    let data = back.to_flat_vec::<f32>().unwrap();
    let original = [0.5, 1.5, -2.0, 3.25];
    for (orig, got) in original.iter().zip(data.iter()) {
        assert!(
            (orig - got).abs() < 0.01,
            "F32→F16→F32 roundtrip: expected ~{orig}, got {got}"
        );
    }
}

#[test]
fn test_to_dtype_bf16_to_f16() {
    let t = DynTensor::full(&[3], 1.5, DType::BF16, &cpu()).unwrap();
    let f16 = t.to_dtype(DType::F16).unwrap();
    assert_eq!(f16.dtype(), DType::F16);
    assert_eq!(f16.dims(), &[3]);
}

#[test]
fn test_to_dtype_f16_to_bf16() {
    let t = DynTensor::full(&[3], 2.0, DType::F16, &cpu()).unwrap();
    let bf16 = t.to_dtype(DType::BF16).unwrap();
    assert_eq!(bf16.dtype(), DType::BF16);
    assert_eq!(bf16.dims(), &[3]);
}

#[test]
fn test_to_dtype_same_dtype_is_clone() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let same = t.to_dtype(DType::F32).unwrap();
    assert_eq!(same.dtype(), DType::F32);
    let data = same.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_to_dtype_preserves_shape() {
    let t = DynTensor::from_vec(vec![1.0; 24], &[2, 3, 4], &cpu()).unwrap();
    let f16 = t.to_dtype(DType::F16).unwrap();
    assert_eq!(f16.dims(), &[2, 3, 4]);
    let bf16 = t.to_dtype(DType::BF16).unwrap();
    assert_eq!(bf16.dims(), &[2, 3, 4]);
}

#[test]
fn test_to_dtype_f32_to_u32() {
    let t = DynTensor::from_vec(vec![0.0, 1.0, 2.0, 3.0], &[4], &cpu()).unwrap();
    let u32t = t.to_dtype(DType::U32).unwrap();
    assert_eq!(u32t.dtype(), DType::U32);
    let data = u32t.to_flat_vec::<u32>().unwrap();
    assert_eq!(data, vec![0, 1, 2, 3]);
}

#[test]
fn test_to_dtype_f32_to_i64() {
    let t = DynTensor::from_vec(vec![-1.0, 0.0, 1.0, 100.0], &[4], &cpu()).unwrap();
    let i64t = t.to_dtype(DType::I64).unwrap();
    assert_eq!(i64t.dtype(), DType::I64);
    let data = i64t.to_flat_vec::<i64>().unwrap();
    assert_eq!(data, vec![-1, 0, 1, 100]);
}

#[test]
fn test_to_dtype_f32_to_u8() {
    let t = DynTensor::from_vec(vec![0.0, 128.0, 255.0], &[3], &cpu()).unwrap();
    let u8t = t.to_dtype(DType::U8).unwrap();
    assert_eq!(u8t.dtype(), DType::U8);
    let data = u8t.to_flat_vec::<u8>().unwrap();
    assert_eq!(data, vec![0, 128, 255]);
}

// ============================================================================
// 5. Device operations
// ============================================================================

#[test]
fn test_to_device_cpu_identity() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let same = t.to_device(&Device::Cpu).unwrap();
    assert_eq!(same.device(), Device::Cpu);
    assert_eq!(same.dims(), &[3]);
    let data = same.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_device_returns_cpu() {
    let t = DynTensor::zeros(&[2, 3], DType::F32, &cpu()).unwrap();
    assert_eq!(t.device(), Device::Cpu);
    assert!(t.device().is_cpu());
}

#[test]
fn test_is_contiguous_fresh_tensor() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    assert!(t.is_contiguous());
}

#[test]
fn test_is_contiguous_after_transpose() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    // Transpose makes data non-contiguous in many frameworks, but DynTensor
    // may produce contiguous copies. Test the is_contiguous method returns a bool.
    let transposed = t.transpose(0, 1).unwrap();
    // Whether contiguous or not, contiguous() should succeed
    let c = transposed.contiguous().unwrap();
    assert!(c.is_contiguous());
    assert_eq!(c.dims(), &[3, 2]);
}

#[test]
fn test_contiguous_preserves_values() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let narrowed = t.narrow(1, 0, 2).unwrap();
    let c = narrowed.contiguous().unwrap();
    assert!(c.is_contiguous());
    assert_eq!(c.dims(), &[2, 2]);
    let data = c.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![1.0, 2.0, 4.0, 5.0]);
}

#[test]
fn test_contiguous_already_contiguous_noop() {
    let t = DynTensor::zeros(&[3, 4], DType::F32, &cpu()).unwrap();
    assert!(t.is_contiguous());
    let c = t.contiguous().unwrap();
    assert!(c.is_contiguous());
    assert_eq!(c.dims(), &[3, 4]);
}

// ============================================================================
// 6. Property queries
// ============================================================================

#[test]
fn test_shape_1d() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    assert_eq!(t.dims(), &[3]);
    assert_eq!(t.rank(), 1);
    assert_eq!(t.numel(), 3);
}

#[test]
fn test_shape_2d() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.rank(), 2);
    assert_eq!(t.numel(), 6);
}

#[test]
fn test_shape_3d() {
    let t = tnd(&[0.0; 24], &[2, 3, 4]);
    assert_eq!(t.dims(), &[2, 3, 4]);
    assert_eq!(t.rank(), 3);
    assert_eq!(t.numel(), 24);
}

#[test]
fn test_shape_4d() {
    let t = tnd(&[0.0; 120], &[2, 3, 4, 5]);
    assert_eq!(t.dims(), &[2, 3, 4, 5]);
    assert_eq!(t.rank(), 4);
    assert_eq!(t.numel(), 120);
}

#[test]
fn test_dims1_accessor() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    assert_eq!(t.dims1().unwrap(), 3);
}

#[test]
fn test_dims2_accessor() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    assert_eq!(t.dims2().unwrap(), (2, 3));
}

#[test]
fn test_dims3_accessor() {
    let t = tnd(&[0.0; 24], &[2, 3, 4]);
    assert_eq!(t.dims3().unwrap(), (2, 3, 4));
}

#[test]
fn test_dims4_accessor() {
    let t = tnd(&[0.0; 120], &[2, 3, 4, 5]);
    assert_eq!(t.dims4().unwrap(), (2, 3, 4, 5));
}

#[test]
fn test_dims5_accessor() {
    let t = tnd(&[0.0; 720], &[2, 3, 4, 5, 6]);
    assert_eq!(t.dims5().unwrap(), (2, 3, 4, 5, 6));
}

#[test]
fn test_dims_accessor_wrong_rank_errors() {
    let t = t1d(&[1.0]);
    assert!(t.dims2().is_err());
    assert!(t.dims3().is_err());
    assert!(t.dims4().is_err());
    assert!(t.dims5().is_err());
}

#[test]
fn test_numel_empty() {
    let t = DynTensor::from_vec(vec![], &[0], &cpu()).unwrap();
    assert_eq!(t.numel(), 0);
}

#[test]
fn test_numel_scalar() {
    let t = DynTensor::full(&[], 1.0, DType::F32, &cpu()).unwrap();
    assert_eq!(t.numel(), 1);
    assert_eq!(t.rank(), 0);
}

#[test]
fn test_checked_numel_success() {
    let t = DynTensor::zeros(&[10, 20], DType::F32, &cpu()).unwrap();
    assert_eq!(t.checked_numel().unwrap(), 200);
}

#[test]
fn test_dtype_f32() {
    let t = DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::F32);
}

#[test]
fn test_dtype_u32() {
    let t = DynTensor::zeros(&[2], DType::U32, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::U32);
}

#[test]
fn test_dtype_u8() {
    let t = DynTensor::zeros(&[2], DType::U8, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::U8);
}

#[test]
fn test_dtype_i64() {
    let t = DynTensor::zeros(&[2], DType::I64, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::I64);
}

#[test]
fn test_dtype_f16() {
    let t = DynTensor::zeros(&[2], DType::F16, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::F16);
}

#[test]
fn test_dtype_bf16() {
    let t = DynTensor::zeros(&[2], DType::BF16, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::BF16);
}

#[test]
fn test_dim_method_positive_index() {
    let t = tnd(&[0.0; 24], &[2, 3, 4]);
    assert_eq!(t.dim(0).unwrap(), 2);
    assert_eq!(t.dim(1).unwrap(), 3);
    assert_eq!(t.dim(2).unwrap(), 4);
}

#[test]
fn test_dim_method_negative_index() {
    use crate::dyn_tensor::D;
    let t = tnd(&[0.0; 24], &[2, 3, 4]);
    assert_eq!(t.dim(D::Minus1).unwrap(), 4);
    assert_eq!(t.dim(D::Minus2).unwrap(), 3);
}

#[test]
fn test_dim_method_out_of_range() {
    let t = t1d(&[1.0, 2.0]);
    assert!(t.dim(1).is_err());
}

// ============================================================================
// 7. Scalar and edge-case construction
// ============================================================================

#[test]
fn test_scalar_from_full() {
    let t = DynTensor::full(&[], 42.0, DType::F32, &cpu()).unwrap();
    assert_eq!(t.rank(), 0);
    assert_eq!(t.numel(), 1);
    let val = t.to_scalar::<f32>().unwrap();
    assert!((val - 42.0).abs() < 1e-6);
}

#[test]
fn test_zeros_like_preserves_properties() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let z = t.zeros_like().unwrap();
    assert_eq!(z.dims(), t.dims());
    assert_eq!(z.dtype(), t.dtype());
    assert_eq!(z.device(), t.device());
    let data = z.to_flat_vec::<f32>().unwrap();
    assert!(data.iter().all(|&v| v == 0.0));
}

#[test]
fn test_ones_like_preserves_properties() {
    let t = DynTensor::zeros(&[3, 4], DType::F32, &cpu()).unwrap();
    let o = t.ones_like().unwrap();
    assert_eq!(o.dims(), t.dims());
    assert_eq!(o.dtype(), t.dtype());
    let data = o.to_flat_vec::<f32>().unwrap();
    assert!(data.iter().all(|&v| v == 1.0));
}

#[test]
fn test_full_like_preserves_properties() {
    let t = DynTensor::zeros(&[2, 3], DType::F32, &cpu()).unwrap();
    let f = t.full_like(7.5).unwrap();
    assert_eq!(f.dims(), t.dims());
    assert_eq!(f.dtype(), t.dtype());
    let data = f.to_flat_vec::<f32>().unwrap();
    assert!(data.iter().all(|&v| (v - 7.5).abs() < 1e-5));
}

// ============================================================================
// 8. Multi-dimensional from_vec construction
// ============================================================================

#[test]
fn test_from_vec_3d() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = DynTensor::from_vec(data.clone(), &[2, 3, 4], &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 3, 4]);
    assert_eq!(t.numel(), 24);
    let got = t.to_flat_vec::<f32>().unwrap();
    assert_eq!(got, data);
}

#[test]
fn test_from_vec_4d() {
    let data: Vec<f32> = (0..120).map(|i| i as f32).collect();
    let t = DynTensor::from_vec(data, &[2, 3, 4, 5], &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 3, 4, 5]);
    assert_eq!(t.numel(), 120);
}

#[test]
fn test_from_slice_3d() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = DynTensor::from_slice(&data, &[2, 3, 4], &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 3, 4]);
}

#[test]
fn test_new_with_tuple_shape() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 3]);
}

#[test]
fn test_new_with_vec_shape() {
    let shape = vec![2, 3];
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape, &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 3]);
}

// ============================================================================
// 9. Debug representation
// ============================================================================

#[test]
fn test_debug_repr_contains_dims_and_dtype() {
    let t = DynTensor::zeros(&[2, 3], DType::F32, &cpu()).unwrap();
    let debug = format!("{t:?}");
    assert!(debug.contains("dims"), "debug should contain dims");
    assert!(debug.contains("F32"), "debug should contain dtype");
    assert!(debug.contains("Cpu"), "debug should contain device");
}

// ============================================================================
// 10. to_f32_array accessor
// ============================================================================

#[test]
fn test_to_f32_array_from_f32() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let arr = t.to_f32_array().unwrap();
    assert_eq!(arr.shape(), &[3]);
    assert_eq!(arr.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_to_f32_array_from_f16() {
    let data: Vec<half::f16> = vec![1.0, 2.0, 3.0]
        .into_iter()
        .map(half::f16::from_f32)
        .collect();
    let t = DynTensor::from_vec_f16(data, &[3], &cpu()).unwrap();
    let arr = t.to_f32_array().unwrap();
    assert_eq!(arr.shape(), &[3]);
    for (i, &v) in arr.iter().enumerate() {
        let expected = (i + 1) as f32;
        assert!((v - expected).abs() < 0.01);
    }
}

#[test]
fn test_to_f32_array_from_bf16() {
    let data: Vec<half::bf16> = vec![1.0, 2.0, 3.0]
        .into_iter()
        .map(half::bf16::from_f32)
        .collect();
    let t = DynTensor::from_vec_bf16(data, &[3], &cpu()).unwrap();
    let arr = t.to_f32_array().unwrap();
    assert_eq!(arr.shape(), &[3]);
    for (i, &v) in arr.iter().enumerate() {
        let expected = (i + 1) as f32;
        assert!((v - expected).abs() < 0.05);
    }
}

#[test]
fn test_as_cpu_f32_returns_view() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let view = t.as_cpu_f32().unwrap();
    assert_eq!(view.shape(), &[3]);
    assert_eq!(view.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_as_cpu_f16_returns_view() {
    let data: Vec<half::f16> = vec![1.0, 2.0]
        .into_iter()
        .map(half::f16::from_f32)
        .collect();
    let t = DynTensor::from_vec_f16(data, &[2], &cpu()).unwrap();
    let view = t.as_cpu_f16().unwrap();
    assert_eq!(view.shape(), &[2]);
}

#[test]
fn test_as_cpu_bf16_returns_view() {
    let data: Vec<half::bf16> = vec![1.0, 2.0]
        .into_iter()
        .map(half::bf16::from_f32)
        .collect();
    let t = DynTensor::from_vec_bf16(data, &[2], &cpu()).unwrap();
    let view = t.as_cpu_bf16().unwrap();
    assert_eq!(view.shape(), &[2]);
}

// ============================================================================
// 11. Detach and clone semantics
// ============================================================================

#[test]
fn test_detach_returns_clone() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let d = t.detach();
    assert_eq!(d.dims(), t.dims());
    assert_eq!(d.dtype(), t.dtype());
    let data = d.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_get_method() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let row0 = t.get(0).unwrap();
    assert_eq!(row0.dims(), &[3]);
    let data = row0.to_vec1::<f32>().unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0]);

    let row1 = t.get(1).unwrap();
    let data1 = row1.to_vec1::<f32>().unwrap();
    assert_eq!(data1, vec![4.0, 5.0, 6.0]);
}

#[test]
fn test_reshape_as() {
    let a = DynTensor::from_vec(vec![1.0; 12], &[3, 4], &cpu()).unwrap();
    let b = DynTensor::zeros(&[4, 3], DType::F32, &cpu()).unwrap();
    let reshaped = a.reshape_as(&b).unwrap();
    assert_eq!(reshaped.dims(), &[4, 3]);
}

#[test]
fn test_expand_as() {
    let a = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let b = DynTensor::zeros(&[4, 3], DType::F32, &cpu()).unwrap();
    let expanded = a.expand_as(&b).unwrap();
    assert_eq!(expanded.dims(), &[4, 3]);
}
