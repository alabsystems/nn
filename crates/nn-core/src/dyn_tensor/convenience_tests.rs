#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for candle convenience API (D3 + D1 + D2 + D4).

use crate::dyn_tensor::indexing::IndexOp;
use crate::dyn_tensor::test_helpers::cpu;
use crate::{DType, DynTensor, D};

// =============================================================================
// D3: Convenience methods (.t(), .get(), u32 extraction, vec2/vec3)
// =============================================================================

#[test]
fn test_t_transposes_last_two_dims() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let result = t.t().unwrap();
    assert_eq!(result.dims(), &[3, 2]);
    let data = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_t_3d_transposes_last_two() {
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let t = DynTensor::new(&data, &[2, 2, 3], &cpu()).unwrap();
    let result = t.t().unwrap();
    assert_eq!(result.dims(), &[2, 3, 2]);
}

#[test]
fn test_t_rank_1_error() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    assert!(t.t().is_err());
}

#[test]
fn test_get_selects_dim_0() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &cpu()).unwrap();
    let row1 = t.get(1).unwrap();
    assert_eq!(row1.dims(), &[2]);
    assert_eq!(row1.to_vec1::<f32>().unwrap(), vec![3.0, 4.0]);
}

#[test]
fn test_get_out_of_range() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &cpu()).unwrap();
    assert!(t.get(3).is_err());
}

#[test]
fn test_to_vec1_u32() {
    let t = DynTensor::from_vec_u32(vec![10, 20, 30], &[3], &cpu()).unwrap();
    assert_eq!(t.to_vec1::<u32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn test_to_vec1_u32_rank_error() {
    let t = DynTensor::from_vec_u32(vec![1, 2, 3, 4], &[2, 2], &cpu()).unwrap();
    assert!(t.to_vec1::<u32>().is_err());
}

#[test]
fn test_to_scalar_u32() {
    let t = DynTensor::from_vec_u32(vec![42], &[1], &cpu()).unwrap();
    assert_eq!(t.to_scalar::<u32>().unwrap(), 42);
}

#[test]
fn test_to_scalar_u32_multiple_elements_error() {
    let t = DynTensor::from_vec_u32(vec![1, 2], &[2], &cpu()).unwrap();
    assert!(t.to_scalar::<u32>().is_err());
}

#[test]
fn test_to_vec2_f32() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let v: Vec<Vec<f32>> = t.to_vec2().unwrap();
    assert_eq!(v, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
}

#[test]
fn test_to_vec2_f32_rank_error() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], &cpu()).unwrap();
    assert!(t.to_vec2::<f32>().is_err());
}

#[test]
fn test_to_vec3_f32() {
    let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let t = DynTensor::new(&data, &[2, 2, 2], &cpu()).unwrap();
    let v: Vec<Vec<Vec<f32>>> = t.to_vec3().unwrap();
    assert_eq!(
        v,
        vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ]
    );
}

#[test]
fn test_to_vec3_f32_rank_error() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    assert!(t.to_vec3::<f32>().is_err());
}

// =============================================================================
// D1: D enum + Dim trait
// =============================================================================

#[test]
fn test_d_minus1_sum() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let result = t.sum(D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2]);
    let data = result.to_vec1::<f32>().unwrap();
    assert!((data[0] - 6.0).abs() < 1e-6);
    assert!((data[1] - 15.0).abs() < 1e-6);
}

#[test]
fn test_d_minus2_sum() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let result = t.sum(D::Minus2).unwrap();
    assert_eq!(result.dims(), &[3]);
    let data = result.to_vec1::<f32>().unwrap();
    assert!((data[0] - 5.0).abs() < 1e-6);
    assert!((data[1] - 7.0).abs() < 1e-6);
    assert!((data[2] - 9.0).abs() < 1e-6);
}

#[test]
fn test_d_minus1_argmax() {
    let t = DynTensor::new(&[1.0, 3.0, 2.0, 6.0, 4.0, 5.0], &[2, 3], &cpu()).unwrap();
    let result = t.argmax(D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2]);
    assert_eq!(result.dtype(), DType::U32);
    let data = result.to_vec1::<u32>().unwrap();
    assert_eq!(data[0], 1);
    assert_eq!(data[1], 0);
}

#[test]
fn test_d_minus1_on_rank0_error() {
    use crate::dyn_tensor::Dim;
    assert!(D::Minus1.to_index(0).is_err());
}

#[test]
fn test_d_minus2_on_rank1_error() {
    use crate::dyn_tensor::Dim;
    assert!(D::Minus2.to_index(1).is_err());
}

#[test]
fn test_usize_dim_out_of_range() {
    use crate::dyn_tensor::Dim;
    assert!(5usize.to_index(3).is_err());
}

#[test]
fn test_d_minus1_softmax() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let result = t.softmax(D::Minus1).unwrap();
    assert_eq!(result.dims(), &[1, 3]);
    let data = result.to_flat_vec::<f32>().unwrap();
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_d_minus1_narrow() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4], &cpu()).unwrap();
    // D::Minus1 on rank-2 tensor resolves to dim 1
    let result = t.narrow(1, 1, 2).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    let data = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![2.0, 3.0, 6.0, 7.0]);
}

#[test]
fn test_d_minus1_flatten() {
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let t = DynTensor::new(&data, &[2, 3, 4], &cpu()).unwrap();
    let result = t.flatten(D::Minus2, D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2, 12]);
}

// =============================================================================
// D2: IndexOp (.i())
// =============================================================================

#[test]
fn test_i_select_single() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &cpu()).unwrap();
    let row = t.i(0usize).unwrap();
    assert_eq!(row.dims(), &[2]);
    assert_eq!(row.to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn test_i_range() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2], &cpu()).unwrap();
    let sub = t.i(1..3).unwrap();
    assert_eq!(sub.dims(), &[2, 2]);
    assert_eq!(sub.to_flat_vec::<f32>().unwrap(), vec![3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_i_get_chain_for_scalar() {
    // Equivalent to t.i((0, 1)) — chained get() calls
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let scalar = t.get(0).unwrap().get(1).unwrap();
    assert_eq!(scalar.dims(), &[] as &[usize]);
    assert_eq!(scalar.to_scalar::<f32>().unwrap(), 2.0);
}

#[test]
fn test_i_get_chain_3d() {
    // Equivalent to t.i((0, 1, 2)) — chained get() calls on 3D tensor
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let t = DynTensor::new(&data, &[2, 3, 4], &cpu()).unwrap();
    let scalar = t.get(0).unwrap().get(1).unwrap().get(2).unwrap();
    assert_eq!(scalar.to_scalar::<f32>().unwrap(), 6.0);
}

#[test]
fn test_i_get_then_narrow() {
    // Equivalent to t.i((0, 1..3)) — get row, then narrow
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let t = DynTensor::new(&data, &[3, 4], &cpu()).unwrap();
    let sub = t.get(0).unwrap().narrow(0, 1, 2).unwrap();
    assert_eq!(sub.dims(), &[2]);
    assert_eq!(sub.to_vec1::<f32>().unwrap(), vec![2.0, 3.0]);
}

#[test]
fn test_i_narrow_column() {
    // Equivalent to t.i((.., 0)) — select column via narrow + squeeze
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let col = t.narrow(1, 0, 1).unwrap().squeeze(1).unwrap();
    assert_eq!(col.dims(), &[2]);
    assert_eq!(col.to_vec1::<f32>().unwrap(), vec![1.0, 4.0]);
}

// =============================================================================
// D4: to_dtype conversions
// =============================================================================

#[test]
fn test_to_dtype_same() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let same = t.to_dtype(DType::F32).unwrap();
    assert_eq!(same.dtype(), DType::F32);
    assert_eq!(same.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_to_dtype_u32_to_f32() {
    let t = DynTensor::from_vec_u32(vec![10, 20, 30], &[3], &cpu()).unwrap();
    let f = t.to_dtype(DType::F32).unwrap();
    assert_eq!(f.dtype(), DType::F32);
    assert_eq!(f.to_vec1::<f32>().unwrap(), vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_to_dtype_f32_to_u32() {
    let t = DynTensor::new(&[1.5, 2.7, 3.0], &[3], &cpu()).unwrap();
    let u = t.to_dtype(DType::U32).unwrap();
    assert_eq!(u.dtype(), DType::U32);
    assert_eq!(u.to_vec1::<u32>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn test_to_dtype_f32_to_f16_lossy() {
    let t = DynTensor::new(&[1.0001, 256.5], &[2], &cpu()).unwrap();
    let f16 = t.to_dtype(DType::F16).unwrap();
    // to_dtype(F16) creates native f16 storage.
    assert_eq!(f16.dtype(), DType::F16);
    let data = f16.to_flat_vec::<f32>().unwrap();
    assert!((data[0] - 1.0).abs() < 0.002);
    assert!((data[1] - 256.5).abs() < 0.01);
}

#[test]
fn test_to_dtype_f32_to_bf16_lossy() {
    let t = DynTensor::new(&[1.234], &[1], &cpu()).unwrap();
    let bf16 = t.to_dtype(DType::BF16).unwrap();
    let data = bf16.to_vec1::<f32>().unwrap();
    assert!((data[0] - 1.234).abs() < 0.02);
}

#[test]
fn test_to_dtype_bf16_to_f32() {
    let t = DynTensor::new(&[1.5, 2.0, 3.25], &[3], &cpu()).unwrap();
    let bf16 = t.to_dtype(DType::BF16).unwrap();
    // to_dtype(BF16) creates native bf16 storage.
    assert_eq!(bf16.dtype(), DType::BF16);
    let f32_back = bf16.to_dtype(DType::F32).unwrap();
    assert_eq!(f32_back.dtype(), DType::F32);
    let data = f32_back.to_vec1::<f32>().unwrap();
    assert!((data[0] - 1.5).abs() < 0.02);
    assert!((data[1] - 2.0).abs() < 0.02);
    assert!((data[2] - 3.25).abs() < 0.02);
}

#[test]
fn test_to_dtype_f16_to_f32() {
    let t = DynTensor::new(&[0.5, 1.0, 4.0], &[3], &cpu()).unwrap();
    let f16 = t.to_dtype(DType::F16).unwrap();
    // to_dtype(F16) creates native f16 storage.
    assert_eq!(f16.dtype(), DType::F16);
    let f32_back = f16.to_dtype(DType::F32).unwrap();
    assert_eq!(f32_back.dtype(), DType::F32);
    let data = f32_back.to_vec1::<f32>().unwrap();
    assert!((data[0] - 0.5).abs() < 0.002);
    assert!((data[1] - 1.0).abs() < 0.002);
    assert!((data[2] - 4.0).abs() < 0.002);
}

#[test]
fn test_to_dtype_unsupported() {
    // I32 and Bool have no storage paths — conversion should fail.
    let t = DynTensor::new(&[1.0, 2.0], &[2], &cpu()).unwrap();
    assert!(t.to_dtype(DType::I32).is_err());
    assert!(t.to_dtype(DType::Bool).is_err());
}

#[test]
fn test_to_dtype_f32_to_i64() {
    let t = DynTensor::new(&[1.0, 2.0, -3.0], &[3], &cpu()).unwrap();
    let i64_t = t.to_dtype(DType::I64).unwrap();
    assert_eq!(i64_t.dtype(), DType::I64);
    assert_eq!(i64_t.dims(), &[3]);
    // Convert back to f32 to verify values
    let back = i64_t.to_dtype(DType::F32).unwrap();
    let data = back.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![1.0, 2.0, -3.0]);
}

#[test]
fn test_to_dtype_i64_to_u32() {
    let t = DynTensor::from_vec_i64(vec![0, 5, 100], &[3], &cpu()).unwrap();
    let u32_t = t.to_dtype(DType::U32).unwrap();
    assert_eq!(u32_t.dtype(), DType::U32);
}

#[test]
fn test_to_dtype_i64_to_u32_negative_rejects() {
    let t = DynTensor::from_vec_i64(vec![-1, 5], &[2], &cpu()).unwrap();
    assert!(t.to_dtype(DType::U32).is_err());
}

// =============================================================================
// is_contiguous / detach (candle API compat)
// =============================================================================

#[test]
fn test_is_contiguous_fresh_tensor() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    assert!(t.is_contiguous());
}

#[test]
fn test_is_contiguous_after_transpose_eager_copy() {
    // nn transpose() eagerly copies to standard layout (unlike candle/PyTorch views).
    // Transposed output is always contiguous because `.as_standard_layout()` is called.
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let transposed = t.transpose(0, 1).unwrap();
    assert_eq!(transposed.dims(), &[3, 2]);
    // Verify data is correctly transposed (not just shape)
    let data = transposed.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert!(transposed.is_contiguous());
}

#[test]
fn test_is_contiguous_after_contiguous() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let c = t.contiguous().unwrap();
    assert!(c.is_contiguous());
}

#[test]
fn test_is_contiguous_u32_tensor() {
    let t = DynTensor::from_vec_u32(vec![1, 2, 3, 4], &[4], &cpu()).unwrap();
    assert!(t.is_contiguous());
}

#[test]
fn test_is_contiguous_i64_tensor() {
    let t = DynTensor::from_vec_i64(vec![10, 20, 30], &[3], &cpu()).unwrap();
    assert!(t.is_contiguous());
}

#[test]
fn test_is_contiguous_bf16_tensor() {
    // BF16 tensors use FloatStorage::BF16 internally.
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let bf16 = t.to_dtype(DType::BF16).unwrap();
    assert!(bf16.is_contiguous());
}

#[test]
fn test_is_contiguous_f16_tensor() {
    // F16 tensors use FloatStorage::F16 internally.
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let f16 = t.to_dtype(DType::F16).unwrap();
    assert!(f16.is_contiguous());
}

#[test]
fn test_detach_returns_same_data() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let d = t.detach();
    assert_eq!(d.dims(), t.dims());
    assert_eq!(d.dtype(), t.dtype());
    assert_eq!(d.to_vec1::<f32>().unwrap(), t.to_vec1::<f32>().unwrap());
}

#[test]
fn test_detach_preserves_shape_and_dtype() {
    let t = DynTensor::from_vec_u32(vec![1, 2, 3, 4, 5, 6], &[2, 3], &cpu()).unwrap();
    let d = t.detach();
    assert_eq!(d.dims(), &[2, 3]);
    assert_eq!(d.dtype(), DType::U32);
}

// D::Minus1/Minus2 negative-dim tests extracted to convenience_tests_dim.rs
