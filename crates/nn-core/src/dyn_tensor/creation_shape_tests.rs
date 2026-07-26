// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for DynTensor creation constructors and shape manipulation operations.
//!
//! Covers: zeros, ones, full, arange, arange_step, from_vec, from_slice,
//! reshape, transpose, permute, squeeze, unsqueeze, narrow, expand, flatten,
//! contiguous, chunk, split, flip, repeat, and dtype creation/conversion.

use crate::dyn_tensor::test_helpers::{cpu, t1d, t2d};
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device};

// ============================================================================
// Creation: zeros
// ============================================================================

#[test]
fn test_zeros_f32_2d() {
    let t = DynTensor::zeros(&[3, 4], DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims(), &[3, 4]);
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(t.numel(), 12);
    let data = t.to_vec1::<f32>().unwrap_or_else(|_| {
        // Not 1D, flatten first
        t.reshape([12]).unwrap().to_vec1::<f32>().unwrap()
    });
    assert!(data.iter().all(|&v| v == 0.0));
}

#[test]
fn test_zeros_1d() {
    let t = DynTensor::zeros(&[5], DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims(), &[5]);
    assert_eq!(t.rank(), 1);
    let data = t.to_vec1::<f32>().unwrap();
    assert_eq!(data, vec![0.0; 5]);
}

#[test]
fn test_zeros_scalar_shape() {
    // Empty dims = rank-0 scalar tensor
    let t = DynTensor::zeros(&[], DType::F32, &cpu()).unwrap();
    assert_eq!(t.rank(), 0);
    assert_eq!(t.numel(), 1);
    assert_eq!(t.dims(), &[] as &[usize]);
}

#[test]
fn test_zeros_u32() {
    let t = DynTensor::zeros(&[2, 3], DType::U32, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::U32);
    assert_eq!(t.dims(), &[2, 3]);
}

#[test]
fn test_zeros_i64() {
    let t = DynTensor::zeros(&[4], DType::I64, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::I64);
    assert_eq!(t.dims(), &[4]);
}

#[test]
fn test_zeros_bf16() {
    let t = DynTensor::zeros(&[2, 2], DType::BF16, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::BF16);
    assert_eq!(t.dims(), &[2, 2]);
}

#[test]
fn test_zeros_f16() {
    let t = DynTensor::zeros(&[3], DType::F16, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::F16);
}

#[test]
fn test_zeros_with_zero_dim() {
    let t = DynTensor::zeros(&[2, 0, 3], DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 0, 3]);
    assert_eq!(t.numel(), 0);
}

// ============================================================================
// Creation: ones
// ============================================================================

#[test]
fn test_ones_f32_2d() {
    let t = DynTensor::ones(&[2, 3], DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 3]);
    let flat = t.reshape([6]).unwrap().to_vec1::<f32>().unwrap();
    assert!(flat.iter().all(|&v| v == 1.0));
}

#[test]
fn test_ones_u8() {
    let t = DynTensor::ones(&[4], DType::U8, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::U8);
    assert_eq!(t.dims(), &[4]);
}

#[test]
fn test_ones_scalar() {
    let t = DynTensor::ones(&[], DType::F32, &cpu()).unwrap();
    assert_eq!(t.rank(), 0);
    assert_eq!(t.numel(), 1);
}

// ============================================================================
// Creation: full
// ============================================================================

#[test]
fn test_full_f32() {
    let t = DynTensor::full(&[3, 2], 4.5, DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims(), &[3, 2]);
    let flat = t.reshape([6]).unwrap().to_vec1::<f32>().unwrap();
    assert!(flat.iter().all(|&v| (v - 4.5).abs() < 1e-6));
}

#[test]
fn test_full_u32() {
    let t = DynTensor::full(&[2], 42.0, DType::U32, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::U32);
}

#[test]
fn test_full_i64() {
    let t = DynTensor::full(&[3], -7.0, DType::I64, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::I64);
    assert_eq!(t.dims(), &[3]);
}

#[test]
fn test_full_negative_value() {
    let t = DynTensor::full(&[2, 2], -3.14, DType::F32, &cpu()).unwrap();
    let flat = t.reshape([4]).unwrap().to_vec1::<f32>().unwrap();
    assert!(flat.iter().all(|&v| (v - (-3.14f32)).abs() < 1e-5));
}

#[test]
fn test_full_bf16() {
    let t = DynTensor::full(&[2], 1.5, DType::BF16, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::BF16);
}

#[test]
fn test_full_f16() {
    let t = DynTensor::full(&[3], 2.0, DType::F16, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::F16);
}

#[test]
fn test_full_u32_rejects_fractional() {
    let result = DynTensor::full(&[2], 3.5, DType::U32, &cpu());
    assert!(result.is_err(), "fractional value should fail for U32");
}

#[test]
fn test_full_u32_rejects_negative() {
    let result = DynTensor::full(&[2], -1.0, DType::U32, &cpu());
    assert!(result.is_err(), "negative value should fail for U32");
}

// ============================================================================
// Creation: from_vec / from_slice / new
// ============================================================================

#[test]
fn test_from_vec_1d() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    assert_eq!(t.dims(), &[3]);
    assert_eq!(t.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_from_vec_2d() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 2]);
    let rows = t.to_vec2::<f32>().unwrap();
    assert_eq!(rows, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
}

#[test]
fn test_from_vec_length_mismatch() {
    let result = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[2, 2], &cpu());
    assert!(result.is_err(), "length mismatch should fail");
}

#[test]
fn test_from_slice_equivalence() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = DynTensor::new(&data, &[2, 3], &cpu()).unwrap();
    let b = DynTensor::from_slice(&data, &[2, 3], &cpu()).unwrap();
    assert_eq!(a.dims(), b.dims());
    assert_eq!(
        a.reshape([6]).unwrap().to_vec1::<f32>().unwrap(),
        b.reshape([6]).unwrap().to_vec1::<f32>().unwrap()
    );
}

#[test]
fn test_new_empty_tensor() {
    let t = DynTensor::from_vec(vec![], &[0], &cpu()).unwrap();
    assert_eq!(t.dims(), &[0]);
    assert_eq!(t.numel(), 0);
}

// ============================================================================
// Creation: zeros_like / ones_like / full_like
// ============================================================================

#[test]
fn test_zeros_like() {
    let original = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let z = original.zeros_like().unwrap();
    assert_eq!(z.dims(), original.dims());
    assert_eq!(z.dtype(), original.dtype());
    let flat = z.reshape([4]).unwrap().to_vec1::<f32>().unwrap();
    assert!(flat.iter().all(|&v| v == 0.0));
}

#[test]
fn test_ones_like() {
    let original = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let o = original.ones_like().unwrap();
    assert_eq!(o.dims(), original.dims());
    let flat = o.reshape([4]).unwrap().to_vec1::<f32>().unwrap();
    assert!(flat.iter().all(|&v| v == 1.0));
}

#[test]
fn test_full_like() {
    let original = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let f = original.full_like(7.0).unwrap();
    assert_eq!(f.dims(), original.dims());
    let flat = f.reshape([4]).unwrap().to_vec1::<f32>().unwrap();
    assert!(flat.iter().all(|&v| (v - 7.0).abs() < 1e-6));
}

// ============================================================================
// Creation: arange
// ============================================================================

#[test]
fn test_arange_basic() {
    let t = DynTensor::arange(0.0, 5.0, &cpu()).unwrap();
    assert_eq!(t.dims(), &[5]);
    let data = t.to_vec1::<f32>().unwrap();
    assert_eq!(data, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_arange_nonzero_start() {
    let t = DynTensor::arange(2.0, 6.0, &cpu()).unwrap();
    assert_eq!(t.dims(), &[4]);
    let data = t.to_vec1::<f32>().unwrap();
    assert_eq!(data, vec![2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_arange_empty_range() {
    let t = DynTensor::arange(5.0, 5.0, &cpu()).unwrap();
    assert_eq!(t.dims(), &[0]);
    assert_eq!(t.numel(), 0);
}

#[test]
fn test_arange_step_fractional() {
    let t = DynTensor::arange_step(0.0, 1.0, 0.5, &cpu()).unwrap();
    assert_eq!(t.dims(), &[2]);
    let data = t.to_vec1::<f32>().unwrap();
    assert!((data[0] - 0.0).abs() < 1e-6);
    assert!((data[1] - 0.5).abs() < 1e-6);
}

#[test]
fn test_arange_step_negative() {
    // Negative step: should produce empty since end > start
    let t = DynTensor::arange_step(0.0, 5.0, -1.0, &cpu()).unwrap();
    assert_eq!(t.dims(), &[0]);
}

#[test]
fn test_arange_step_zero_errors() {
    let result = DynTensor::arange_step(0.0, 5.0, 0.0, &cpu());
    assert!(result.is_err(), "zero step should fail");
}

#[test]
fn test_arange_nan_errors() {
    let result = DynTensor::arange(f64::NAN, 5.0, &cpu());
    assert!(result.is_err(), "NaN start should fail");
}

// ============================================================================
// Shape queries: rank, dims, numel, is_contiguous
// ============================================================================

#[test]
fn test_shape_queries_basic() {
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    assert_eq!(t.rank(), 3);
    assert_eq!(t.dims(), &[2, 3, 4]);
    assert_eq!(t.numel(), 24);
    assert!(t.is_contiguous());
}

#[test]
fn test_dims_accessors() {
    let t1 = DynTensor::zeros(&[5], DType::F32, &cpu()).unwrap();
    assert_eq!(t1.dims1().unwrap(), 5);

    let t2 = DynTensor::zeros(&[3, 4], DType::F32, &cpu()).unwrap();
    assert_eq!(t2.dims2().unwrap(), (3, 4));

    let t3 = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    assert_eq!(t3.dims3().unwrap(), (2, 3, 4));

    let t4 = DynTensor::zeros(&[1, 2, 3, 4], DType::F32, &cpu()).unwrap();
    assert_eq!(t4.dims4().unwrap(), (1, 2, 3, 4));

    let t5 = DynTensor::zeros(&[1, 2, 3, 4, 5], DType::F32, &cpu()).unwrap();
    assert_eq!(t5.dims5().unwrap(), (1, 2, 3, 4, 5));
}

#[test]
fn test_dims_accessor_rank_mismatch() {
    let t = DynTensor::zeros(&[2, 3], DType::F32, &cpu()).unwrap();
    assert!(t.dims1().is_err());
    assert!(t.dims3().is_err());
}

#[test]
fn test_dim_single() {
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    assert_eq!(t.dim(0usize).unwrap(), 2);
    assert_eq!(t.dim(1usize).unwrap(), 3);
    assert_eq!(t.dim(2usize).unwrap(), 4);
}

#[test]
fn test_dim_negative_indexing() {
    use crate::dyn_tensor::D;
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    assert_eq!(t.dim(D::Minus1).unwrap(), 4);
    assert_eq!(t.dim(D::Minus2).unwrap(), 3);
}

#[test]
fn test_device_is_cpu() {
    let t = DynTensor::zeros(&[2], DType::F32, &cpu()).unwrap();
    assert_eq!(t.device(), Device::Cpu);
}

#[test]
fn test_checked_numel() {
    let t = DynTensor::zeros(&[100, 200], DType::F32, &cpu()).unwrap();
    assert_eq!(t.checked_numel().unwrap(), 20000);
}

// ============================================================================
// Reshape
// ============================================================================

#[test]
fn test_reshape_basic() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let r = t.reshape([3, 2]).unwrap();
    assert_eq!(r.dims(), &[3, 2]);
    // Data preserved in row-major order
    let flat_orig = t.reshape([6]).unwrap().to_vec1::<f32>().unwrap();
    let flat_new = r.reshape([6]).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(flat_orig, flat_new);
}

#[test]
fn test_reshape_to_1d() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let r = t.reshape([4]).unwrap();
    assert_eq!(r.dims(), &[4]);
    assert_eq!(r.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_reshape_to_higher_rank() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let r = t.reshape([1, 2, 3]).unwrap();
    assert_eq!(r.dims(), &[1, 2, 3]);
    assert_eq!(r.rank(), 3);
}

#[test]
fn test_reshape_numel_mismatch_fails() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let result = t.reshape([2, 2]);
    assert!(result.is_err(), "reshape with different numel should fail");
}

#[test]
fn test_reshape_scalar_to_1d() {
    let t = DynTensor::full(&[], 5.0, DType::F32, &cpu()).unwrap();
    let r = t.reshape([1]).unwrap();
    assert_eq!(r.dims(), &[1]);
    assert_eq!(r.to_vec1::<f32>().unwrap(), vec![5.0]);
}

#[test]
fn test_reshape_1d_to_scalar() {
    let t = t1d(&[5.0]);
    let r = t.reshape([]).unwrap();
    assert_eq!(r.rank(), 0);
    assert_eq!(r.numel(), 1);
}

// ============================================================================
// Flatten
// ============================================================================

#[test]
fn test_flatten_all_dims() {
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    let f = t.flatten(0usize, 2usize).unwrap();
    assert_eq!(f.dims(), &[24]);
}

#[test]
fn test_flatten_partial() {
    let t = DynTensor::zeros(&[2, 3, 4, 5], DType::F32, &cpu()).unwrap();
    let f = t.flatten(1usize, 2usize).unwrap();
    assert_eq!(f.dims(), &[2, 12, 5]);
}

#[test]
fn test_flatten_single_dim_noop() {
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    let f = t.flatten(1usize, 1usize).unwrap();
    assert_eq!(f.dims(), &[2, 3, 4]);
}

#[test]
fn test_flatten_invalid_range() {
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    let result = t.flatten(2usize, 1usize);
    assert!(result.is_err(), "start_dim > end_dim should fail");
}

// ============================================================================
// Transpose
// ============================================================================

#[test]
fn test_transpose_2d() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let tr = t.transpose(0usize, 1usize).unwrap();
    assert_eq!(tr.dims(), &[3, 2]);
    let rows = tr.to_vec2::<f32>().unwrap();
    assert_eq!(rows[0], vec![1.0, 4.0]);
    assert_eq!(rows[1], vec![2.0, 5.0]);
    assert_eq!(rows[2], vec![3.0, 6.0]);
}

#[test]
fn test_transpose_same_dim_noop() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let tr = t.transpose(0usize, 0usize).unwrap();
    assert_eq!(tr.dims(), t.dims());
}

#[test]
fn test_transpose_double_is_identity() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = DynTensor::from_vec(data, &[2, 3], &cpu()).unwrap();
    let double_t = t
        .transpose(0usize, 1usize)
        .unwrap()
        .transpose(0usize, 1usize)
        .unwrap();
    assert_eq!(double_t.dims(), t.dims());
    let orig = t.reshape([6]).unwrap().to_vec1::<f32>().unwrap();
    let result = double_t.reshape([6]).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(orig, result);
}

#[test]
fn test_transpose_3d() {
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    let tr = t.transpose(0usize, 2usize).unwrap();
    assert_eq!(tr.dims(), &[4, 3, 2]);
}

#[test]
fn test_t_shorthand() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let tr = t.t().unwrap();
    assert_eq!(tr.dims(), &[3, 2]);
}

#[test]
fn test_t_requires_rank2_or_more() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let result = t.t();
    assert!(result.is_err(), "t() on 1D should fail");
}

// ============================================================================
// Permute
// ============================================================================

#[test]
fn test_permute_3d() {
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    let p = t.permute([2, 0, 1]).unwrap();
    assert_eq!(p.dims(), &[4, 2, 3]);
}

#[test]
fn test_permute_identity() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let p = t.permute([0, 1]).unwrap();
    assert_eq!(p.dims(), t.dims());
    let orig = t.reshape([6]).unwrap().to_vec1::<f32>().unwrap();
    let result = p.reshape([6]).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(orig, result);
}

#[test]
fn test_permute_inverse() {
    // Apply permute [2, 0, 1] then its inverse [1, 2, 0] should be identity.
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = DynTensor::from_vec(data, &[2, 3, 4], &cpu()).unwrap();
    let p = t.permute([2, 0, 1]).unwrap();
    assert_eq!(p.dims(), &[4, 2, 3]);
    let back = p.permute([1, 2, 0]).unwrap();
    assert_eq!(back.dims(), &[2, 3, 4]);
    let orig_flat = t.reshape([24]).unwrap().to_vec1::<f32>().unwrap();
    let back_flat = back.reshape([24]).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(orig_flat, back_flat);
}

#[test]
fn test_permute_wrong_rank_fails() {
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    let result = t.permute([0, 1]);
    assert!(result.is_err(), "wrong number of axes should fail");
}

#[test]
fn test_permute_duplicate_axis_fails() {
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    let result = t.permute([0, 1, 1]);
    assert!(result.is_err(), "duplicate axis should fail");
}

#[test]
fn test_permute_out_of_range_axis_fails() {
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    let result = t.permute([0, 1, 5]);
    assert!(result.is_err(), "out of range axis should fail");
}

// ============================================================================
// Squeeze / Unsqueeze
// ============================================================================

#[test]
fn test_unsqueeze_at_0() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let u = t.unsqueeze(0usize).unwrap();
    assert_eq!(u.dims(), &[1, 3]);
}

#[test]
fn test_unsqueeze_at_end() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let u = t.unsqueeze(1usize).unwrap();
    assert_eq!(u.dims(), &[3, 1]);
}

#[test]
fn test_unsqueeze_middle() {
    let t = DynTensor::zeros(&[2, 3], DType::F32, &cpu()).unwrap();
    let u = t.unsqueeze(1usize).unwrap();
    assert_eq!(u.dims(), &[2, 1, 3]);
}

#[test]
fn test_squeeze_basic() {
    let t = DynTensor::zeros(&[1, 3, 1, 4], DType::F32, &cpu()).unwrap();
    let s = t.squeeze(0usize).unwrap();
    assert_eq!(s.dims(), &[3, 1, 4]);
}

#[test]
fn test_squeeze_non_unit_fails() {
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    let result = t.squeeze(1usize);
    assert!(result.is_err(), "squeeze on dim with size != 1 should fail");
}

#[test]
fn test_unsqueeze_then_squeeze_identity() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let expanded = t.unsqueeze(0usize).unwrap();
    assert_eq!(expanded.dims(), &[1, 3]);
    let back = expanded.squeeze(0usize).unwrap();
    assert_eq!(back.dims(), &[3]);
    assert_eq!(back.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_squeeze_scalar_squeeze() {
    // Unsqueeze a rank-0 tensor then squeeze back.
    let t = DynTensor::full(&[], 5.0, DType::F32, &cpu()).unwrap();
    assert_eq!(t.rank(), 0);
    let u = t.unsqueeze(0usize).unwrap();
    assert_eq!(u.dims(), &[1]);
    let s = u.squeeze(0usize).unwrap();
    assert_eq!(s.rank(), 0);
}

// ============================================================================
// Narrow
// ============================================================================

#[test]
fn test_narrow_basic() {
    let t = t1d(&[10.0, 20.0, 30.0, 40.0, 50.0]);
    let n = t.narrow(0usize, 1, 3).unwrap();
    assert_eq!(n.dims(), &[3]);
    let data = n.to_vec1::<f32>().unwrap();
    assert_eq!(data, vec![20.0, 30.0, 40.0]);
}

#[test]
fn test_narrow_2d() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    // Narrow along dim 1: columns 0..2
    let n = t.narrow(1usize, 0, 2).unwrap();
    assert_eq!(n.dims(), &[2, 2]);
    let rows = n.to_vec2::<f32>().unwrap();
    assert_eq!(rows[0], vec![1.0, 2.0]);
    assert_eq!(rows[1], vec![4.0, 5.0]);
}

#[test]
fn test_narrow_full_range_is_clone() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let n = t.narrow(0usize, 0, 3).unwrap();
    assert_eq!(n.dims(), t.dims());
    assert_eq!(n.to_vec1::<f32>().unwrap(), t.to_vec1::<f32>().unwrap());
}

#[test]
fn test_narrow_out_of_bounds_fails() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let result = t.narrow(0usize, 2, 3);
    assert!(result.is_err(), "narrow beyond dim size should fail");
}

#[test]
fn test_narrow_zero_length() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let n = t.narrow(0usize, 1, 0).unwrap();
    assert_eq!(n.dims(), &[0]);
    assert_eq!(n.numel(), 0);
}

// ============================================================================
// Expand / broadcast
// ============================================================================

#[test]
fn test_expand_1_to_n() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let e = t.expand([4, 3]).unwrap();
    assert_eq!(e.dims(), &[4, 3]);
    let rows = e.to_vec2::<f32>().unwrap();
    for row in &rows {
        assert_eq!(row, &[1.0, 2.0, 3.0]);
    }
}

#[test]
fn test_expand_multiple_dims() {
    let t = DynTensor::from_vec(vec![5.0], &[1, 1], &cpu()).unwrap();
    let e = t.expand([3, 4]).unwrap();
    assert_eq!(e.dims(), &[3, 4]);
    assert_eq!(e.numel(), 12);
}

#[test]
fn test_expand_noop_same_shape() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let e = t.expand([2, 2]).unwrap();
    assert_eq!(e.dims(), t.dims());
}

#[test]
fn test_expand_non_unit_dim_fails() {
    let t = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    let result = t.expand([3]);
    assert!(
        result.is_err(),
        "expand non-unit dim to different size should fail"
    );
}

#[test]
fn test_expand_rank_mismatch_fails() {
    let t = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    let result = t.expand([2, 3]);
    assert!(result.is_err(), "expand with different rank should fail");
}

// ============================================================================
// Contiguous
// ============================================================================

#[test]
fn test_contiguous_already_contiguous() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    assert!(t.is_contiguous());
    let c = t.contiguous().unwrap();
    assert_eq!(c.dims(), t.dims());
    assert!(c.is_contiguous());
}

#[test]
fn test_contiguous_preserves_data() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let c = t.contiguous().unwrap();
    let orig = t.reshape([6]).unwrap().to_vec1::<f32>().unwrap();
    let result = c.reshape([6]).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(orig, result);
}

// ============================================================================
// Chunk / Split
// ============================================================================

#[test]
fn test_chunk_even() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let chunks = t.chunk(3, 0usize).unwrap();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);
    assert_eq!(chunks[1].to_vec1::<f32>().unwrap(), vec![3.0, 4.0]);
    assert_eq!(chunks[2].to_vec1::<f32>().unwrap(), vec![5.0, 6.0]);
}

#[test]
fn test_chunk_uneven() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let chunks = t.chunk(2, 0usize).unwrap();
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(chunks[1].to_vec1::<f32>().unwrap(), vec![4.0, 5.0]);
}

#[test]
fn test_chunk_zero_fails() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let result = t.chunk(0, 0usize);
    assert!(result.is_err(), "zero chunks should fail");
}

#[test]
fn test_split_exact() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let parts = t.split([2, 3], 0usize).unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);
    assert_eq!(parts[1].to_vec1::<f32>().unwrap(), vec![3.0, 4.0, 5.0]);
}

#[test]
fn test_split_wrong_sum_fails() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let result = t.split([1, 1], 0usize);
    assert!(result.is_err(), "split sizes sum != dim size should fail");
}

#[test]
fn test_split_uniform() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let parts = t.split_uniform(2, 0usize).unwrap();
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].dims(), &[2]);
    assert_eq!(parts[1].dims(), &[2]);
    assert_eq!(parts[2].dims(), &[1]); // remainder
}

// ============================================================================
// Flip
// ============================================================================

#[test]
fn test_flip_1d() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0]);
    let f = t.flip(0usize).unwrap();
    assert_eq!(f.to_vec1::<f32>().unwrap(), vec![4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn test_flip_dim1_2d() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let f = t.flip(1usize).unwrap();
    let rows = f.to_vec2::<f32>().unwrap();
    assert_eq!(rows[0], vec![2.0, 1.0]);
    assert_eq!(rows[1], vec![4.0, 3.0]);
}

#[test]
fn test_flip_single_element_noop() {
    let t = t1d(&[42.0]);
    let f = t.flip(0usize).unwrap();
    assert_eq!(f.to_vec1::<f32>().unwrap(), vec![42.0]);
}

// ============================================================================
// Repeat / Tile
// ============================================================================

#[test]
fn test_repeat_1d() {
    let t = t1d(&[1.0, 2.0]);
    let r = t.repeat([3]).unwrap();
    assert_eq!(r.dims(), &[6]);
    assert_eq!(
        r.to_vec1::<f32>().unwrap(),
        vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    );
}

#[test]
fn test_repeat_2d() {
    let t = DynTensor::from_vec(vec![1.0, 2.0], &[1, 2], &cpu()).unwrap();
    let r = t.repeat([3, 2]).unwrap();
    assert_eq!(r.dims(), &[3, 4]);
}

#[test]
fn test_repeat_all_ones_noop() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let r = t.repeat([1, 1]).unwrap();
    assert_eq!(r.dims(), t.dims());
}

#[test]
fn test_repeat_wrong_rank_fails() {
    let t = t1d(&[1.0, 2.0]);
    let result = t.repeat([2, 3]);
    assert!(result.is_err(), "repeat with wrong rank should fail");
}

// ============================================================================
// DType creation and conversion
// ============================================================================

#[test]
fn test_dtype_f32_default() {
    let t = DynTensor::from_vec(vec![1.0], &[1], &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::F32);
}

#[test]
fn test_to_dtype_f32_to_bf16() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let bf = t.to_dtype(DType::BF16).unwrap();
    assert_eq!(bf.dtype(), DType::BF16);
    assert_eq!(bf.dims(), &[3]);
    // Convert back and check approximate values
    let f32_back = bf.to_dtype(DType::F32).unwrap();
    let data = f32_back.to_vec1::<f32>().unwrap();
    for (i, &v) in data.iter().enumerate() {
        assert!(
            (v - (i as f32 + 1.0)).abs() < 0.1,
            "BF16 roundtrip should preserve approximate value"
        );
    }
}

#[test]
fn test_to_dtype_f32_to_f16() {
    let t = DynTensor::from_vec(vec![0.5, 1.5], &[2], &cpu()).unwrap();
    let f16 = t.to_dtype(DType::F16).unwrap();
    assert_eq!(f16.dtype(), DType::F16);
    assert_eq!(f16.dims(), &[2]);
}

#[test]
fn test_to_dtype_same_is_noop() {
    let t = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    let same = t.to_dtype(DType::F32).unwrap();
    assert_eq!(same.dtype(), DType::F32);
    assert_eq!(same.to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn test_zeros_bf16_creation_and_conversion() {
    let bf = DynTensor::zeros(&[4], DType::BF16, &cpu()).unwrap();
    assert_eq!(bf.dtype(), DType::BF16);
    let f32_t = bf.to_dtype(DType::F32).unwrap();
    assert_eq!(f32_t.dtype(), DType::F32);
    let data = f32_t.to_vec1::<f32>().unwrap();
    assert!(data.iter().all(|&v| v == 0.0));
}

#[test]
fn test_ones_f16_creation_and_conversion() {
    let f16 = DynTensor::ones(&[3], DType::F16, &cpu()).unwrap();
    assert_eq!(f16.dtype(), DType::F16);
    let f32_t = f16.to_dtype(DType::F32).unwrap();
    let data = f32_t.to_vec1::<f32>().unwrap();
    for &v in &data {
        assert!((v - 1.0).abs() < 1e-3);
    }
}

// ============================================================================
// Edge cases: scalar tensors (rank 0)
// ============================================================================

#[test]
fn test_scalar_rank0_properties() {
    let t = DynTensor::full(&[], 3.14, DType::F32, &cpu()).unwrap();
    assert_eq!(t.rank(), 0);
    assert_eq!(t.dims(), &[] as &[usize]);
    assert_eq!(t.numel(), 1);
    assert!(t.is_contiguous());
}

#[test]
fn test_scalar_reshape_roundtrip() {
    let scalar = DynTensor::full(&[], 2.0, DType::F32, &cpu()).unwrap();
    let vec1 = scalar.reshape([1]).unwrap();
    assert_eq!(vec1.dims(), &[1]);
    let back = vec1.reshape([]).unwrap();
    assert_eq!(back.rank(), 0);
}

// ============================================================================
// Edge cases: empty tensors (zero-size dims)
// ============================================================================

#[test]
fn test_empty_tensor_reshape() {
    let t = DynTensor::zeros(&[0, 5], DType::F32, &cpu()).unwrap();
    assert_eq!(t.numel(), 0);
    let r = t.reshape([5, 0]).unwrap();
    assert_eq!(r.dims(), &[5, 0]);
    assert_eq!(r.numel(), 0);
}

#[test]
fn test_empty_tensor_narrow() {
    let t = DynTensor::zeros(&[0], DType::F32, &cpu()).unwrap();
    let n = t.narrow(0usize, 0, 0).unwrap();
    assert_eq!(n.dims(), &[0]);
}

// ============================================================================
// Edge cases: large dimensions
// ============================================================================

#[test]
fn test_large_dims_metadata() {
    // Only test shape metadata, don't allocate huge tensor
    let t = DynTensor::zeros(&[1, 1, 0], DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims(), &[1, 1, 0]);
    assert_eq!(t.numel(), 0);
}

// ============================================================================
// Tuple shape syntax
// ============================================================================

#[test]
fn test_tuple_shape_2d() {
    let t = DynTensor::zeros((3, 4), DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims(), &[3, 4]);
}

#[test]
fn test_tuple_shape_3d() {
    let t = DynTensor::zeros((2, 3, 4), DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 3, 4]);
}

#[test]
fn test_tuple_shape_4d() {
    let t = DynTensor::zeros((1, 2, 3, 4), DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims(), &[1, 2, 3, 4]);
}

// ============================================================================
// Random constructors (training feature only)
// ============================================================================

#[cfg(feature = "training")]
mod random_creation {
    use super::*;

    #[test]
    fn test_rand_shape_and_range() {
        let t = DynTensor::rand(0.0, 1.0, &[100], &cpu()).unwrap();
        assert_eq!(t.dims(), &[100]);
        assert_eq!(t.dtype(), DType::F32);
        let data = t.to_vec1::<f32>().unwrap();
        for &v in &data {
            assert!((0.0..=1.0).contains(&v), "rand value {v} out of [0, 1]");
        }
    }

    #[test]
    fn test_rand_custom_range() {
        let t = DynTensor::rand(-5.0, 5.0, &[50], &cpu()).unwrap();
        let data = t.to_vec1::<f32>().unwrap();
        for &v in &data {
            assert!((-5.0..=5.0).contains(&v), "rand value {v} out of [-5, 5]");
        }
    }

    #[test]
    fn test_randn_shape() {
        let t = DynTensor::randn(0.0, 1.0, &[3, 4], &cpu()).unwrap();
        assert_eq!(t.dims(), &[3, 4]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.numel(), 12);
    }

    #[test]
    fn test_rand_like() {
        let original = DynTensor::zeros(&[2, 3], DType::F32, &cpu()).unwrap();
        let r = original.rand_like().unwrap();
        assert_eq!(r.dims(), &[2, 3]);
        assert_eq!(r.dtype(), DType::F32);
    }

    #[test]
    fn test_randn_like() {
        let original = DynTensor::zeros(&[4, 5], DType::F32, &cpu()).unwrap();
        let r = original.randn_like().unwrap();
        assert_eq!(r.dims(), &[4, 5]);
        assert_eq!(r.dtype(), DType::F32);
    }
}
