// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended DynTensor operation tests covering arithmetic, comparison, reduction,
//! shape, indexing, math, matmul, concatenation, dtype conversion, device ops,
//! creation, and contiguity.

use crate::dyn_tensor::test_helpers::{assert_close, cpu, t1d, t2d, tnd};
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device};

// =============================================================================
// Arithmetic ops: add, sub, mul, div with broadcasting
// =============================================================================

#[test]
fn test_add_same_shape() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[4.0, 5.0, 6.0]);
    let r = a.add(&b).unwrap();
    assert_eq!(r.dims(), &[3]);
    assert_close(&r.to_flat_vec::<f32>().unwrap(), &[5.0, 7.0, 9.0], 1e-6);
}

#[test]
fn test_add_broadcast_row_col() {
    // [3,1] + [1,4] = [3,4]
    let a = tnd(&[1.0, 2.0, 3.0], &[3, 1]);
    let b = tnd(&[10.0, 20.0, 30.0, 40.0], &[1, 4]);
    let r = a.add(&b).unwrap();
    assert_eq!(r.dims(), &[3, 4]);
    let expected = vec![
        11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0,
    ];
    assert_close(&r.to_flat_vec::<f32>().unwrap(), &expected, 1e-6);
}

#[test]
fn test_add_broadcast_scalar_to_matrix() {
    // [1] + [2,3] = [2,3]
    let scalar = tnd(&[100.0], &[1]);
    let matrix = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let r = scalar.add(&matrix).unwrap();
    assert_eq!(r.dims(), &[2, 3]);
    assert_close(
        &r.to_flat_vec::<f32>().unwrap(),
        &[101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
        1e-6,
    );
}

#[test]
fn test_sub_same_shape() {
    let a = t1d(&[10.0, 20.0, 30.0]);
    let b = t1d(&[1.0, 2.0, 3.0]);
    let r = a.sub(&b).unwrap();
    assert_close(&r.to_flat_vec::<f32>().unwrap(), &[9.0, 18.0, 27.0], 1e-6);
}

#[test]
fn test_sub_broadcast() {
    // [2,3] - [3] (broadcast along dim 0)
    let a = t2d(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], 2, 3);
    let b = t1d(&[1.0, 2.0, 3.0]);
    let r = a.sub(&b).unwrap();
    assert_eq!(r.dims(), &[2, 3]);
    assert_close(
        &r.to_flat_vec::<f32>().unwrap(),
        &[9.0, 18.0, 27.0, 39.0, 48.0, 57.0],
        1e-6,
    );
}

#[test]
fn test_mul_same_shape() {
    let a = t1d(&[2.0, 3.0, 4.0]);
    let b = t1d(&[5.0, 6.0, 7.0]);
    let r = a.mul(&b).unwrap();
    assert_close(&r.to_flat_vec::<f32>().unwrap(), &[10.0, 18.0, 28.0], 1e-6);
}

#[test]
fn test_mul_broadcast() {
    // [2,3] * [1,3] = [2,3]
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = tnd(&[10.0, 100.0, 1000.0], &[1, 3]);
    let r = a.mul(&b).unwrap();
    assert_eq!(r.dims(), &[2, 3]);
    assert_close(
        &r.to_flat_vec::<f32>().unwrap(),
        &[10.0, 200.0, 3000.0, 40.0, 500.0, 6000.0],
        1e-6,
    );
}

#[test]
fn test_mul_scalar() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0]);
    let r = t.mul_scalar(2.5).unwrap();
    assert_close(
        &r.to_flat_vec::<f32>().unwrap(),
        &[2.5, 5.0, 7.5, 10.0],
        1e-6,
    );
}

#[test]
fn test_div_same_shape() {
    let a = t1d(&[10.0, 20.0, 30.0]);
    let b = t1d(&[2.0, 5.0, 10.0]);
    let r = a.div(&b).unwrap();
    assert_close(&r.to_flat_vec::<f32>().unwrap(), &[5.0, 4.0, 3.0], 1e-6);
}

#[test]
fn test_div_broadcast() {
    // [2,2] / [1,2] = [2,2]
    let a = t2d(&[10.0, 20.0, 30.0, 40.0], 2, 2);
    let b = tnd(&[2.0, 5.0], &[1, 2]);
    let r = a.div(&b).unwrap();
    assert_eq!(r.dims(), &[2, 2]);
    assert_close(
        &r.to_flat_vec::<f32>().unwrap(),
        &[5.0, 4.0, 15.0, 8.0],
        1e-6,
    );
}

#[test]
fn test_neg_and_abs() {
    let t = t1d(&[1.0, -2.0, 3.0, -4.0]);
    let neg = t.neg().unwrap();
    assert_close(
        &neg.to_flat_vec::<f32>().unwrap(),
        &[-1.0, 2.0, -3.0, 4.0],
        1e-6,
    );
    let abs = t.abs().unwrap();
    assert_close(
        &abs.to_flat_vec::<f32>().unwrap(),
        &[1.0, 2.0, 3.0, 4.0],
        1e-6,
    );
}

// =============================================================================
// Comparison ops: eq, ne, lt, gt, le, ge (scalar and tensor)
// =============================================================================

#[test]
fn test_eq_scalar() {
    let t = t1d(&[1.0, 2.0, 3.0, 2.0]);
    let r = t.eq(2.0).unwrap();
    assert_eq!(r.dtype(), DType::U8);
    assert_eq!(r.to_flat_vec::<u8>().unwrap(), vec![0, 1, 0, 1]);
}

#[test]
fn test_ne_scalar() {
    let t = t1d(&[1.0, 2.0, 3.0, 2.0]);
    let r = t.ne(2.0).unwrap();
    assert_eq!(r.dtype(), DType::U8);
    assert_eq!(r.to_flat_vec::<u8>().unwrap(), vec![1, 0, 1, 0]);
}

#[test]
fn test_lt_scalar() {
    let t = t1d(&[1.0, 3.0, 5.0, 7.0]);
    let r = t.lt(4.0).unwrap();
    assert_eq!(r.dtype(), DType::U8);
    assert_eq!(r.to_flat_vec::<u8>().unwrap(), vec![1, 1, 0, 0]);
}

#[test]
fn test_gt_scalar() {
    let t = t1d(&[1.0, 3.0, 5.0, 7.0]);
    let r = t.gt(4.0).unwrap();
    assert_eq!(r.dtype(), DType::U8);
    assert_eq!(r.to_flat_vec::<u8>().unwrap(), vec![0, 0, 1, 1]);
}

#[test]
fn test_le_scalar() {
    let t = t1d(&[1.0, 4.0, 5.0, 7.0]);
    let r = t.le(4.0).unwrap();
    assert_eq!(r.dtype(), DType::U8);
    assert_eq!(r.to_flat_vec::<u8>().unwrap(), vec![1, 1, 0, 0]);
}

#[test]
fn test_ge_scalar() {
    let t = t1d(&[1.0, 4.0, 5.0, 7.0]);
    let r = t.ge(4.0).unwrap();
    assert_eq!(r.dtype(), DType::U8);
    assert_eq!(r.to_flat_vec::<u8>().unwrap(), vec![0, 1, 1, 1]);
}

#[test]
fn test_eq_tensor() {
    let a = t1d(&[1.0, 2.0, 3.0, 4.0]);
    let b = t1d(&[1.0, 0.0, 3.0, 5.0]);
    let r = a.eq_tensor(&b).unwrap();
    assert_eq!(r.dtype(), DType::U8);
    assert_eq!(r.to_flat_vec::<u8>().unwrap(), vec![1, 0, 1, 0]);
}

#[test]
fn test_ne_tensor() {
    let a = t1d(&[1.0, 2.0, 3.0, 4.0]);
    let b = t1d(&[1.0, 0.0, 3.0, 5.0]);
    let r = a.ne_tensor(&b).unwrap();
    assert_eq!(r.dtype(), DType::U8);
    assert_eq!(r.to_flat_vec::<u8>().unwrap(), vec![0, 1, 0, 1]);
}

#[test]
fn test_lt_tensor() {
    let a = t1d(&[1.0, 5.0, 3.0, 7.0]);
    let b = t1d(&[2.0, 4.0, 3.0, 8.0]);
    let r = a.lt_tensor(&b).unwrap();
    assert_eq!(r.dtype(), DType::U8);
    assert_eq!(r.to_flat_vec::<u8>().unwrap(), vec![1, 0, 0, 1]);
}

#[test]
fn test_gt_tensor() {
    let a = t1d(&[1.0, 5.0, 3.0, 7.0]);
    let b = t1d(&[2.0, 4.0, 3.0, 8.0]);
    let r = a.gt_tensor(&b).unwrap();
    assert_eq!(r.dtype(), DType::U8);
    assert_eq!(r.to_flat_vec::<u8>().unwrap(), vec![0, 1, 0, 0]);
}

#[test]
fn test_le_tensor() {
    let a = t1d(&[1.0, 5.0, 3.0, 7.0]);
    let b = t1d(&[2.0, 4.0, 3.0, 8.0]);
    let r = a.le_tensor(&b).unwrap();
    assert_eq!(r.dtype(), DType::U8);
    assert_eq!(r.to_flat_vec::<u8>().unwrap(), vec![1, 0, 1, 1]);
}

#[test]
fn test_ge_tensor() {
    let a = t1d(&[1.0, 5.0, 3.0, 7.0]);
    let b = t1d(&[2.0, 4.0, 3.0, 8.0]);
    let r = a.ge_tensor(&b).unwrap();
    assert_eq!(r.dtype(), DType::U8);
    assert_eq!(r.to_flat_vec::<u8>().unwrap(), vec![0, 1, 1, 0]);
}

#[test]
fn test_where_cond() {
    let t = t1d(&[1.0, 3.0, 5.0, 7.0]);
    let mask = t.gt(4.0).unwrap();
    let on_true = t1d(&[10.0, 20.0, 30.0, 40.0]);
    let on_false = t1d(&[-1.0, -2.0, -3.0, -4.0]);
    let r = mask.where_cond(&on_true, &on_false).unwrap();
    assert_close(
        &r.to_flat_vec::<f32>().unwrap(),
        &[-1.0, -2.0, 30.0, 40.0],
        1e-6,
    );
}

// =============================================================================
// Reduction ops: sum, mean, max, min, argmax, argmin with dim parameter
// =============================================================================

#[test]
fn test_sum_all() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let r = t.sum_all().unwrap();
    let val = r.to_scalar::<f32>().unwrap();
    assert!((val - 21.0).abs() < 1e-5);
}

#[test]
fn test_sum_dim0() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let r = t.sum(0).unwrap();
    assert_eq!(r.dims(), &[3]);
    assert_close(&r.to_flat_vec::<f32>().unwrap(), &[5.0, 7.0, 9.0], 1e-6);
}

#[test]
fn test_sum_dim1() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let r = t.sum(1).unwrap();
    assert_eq!(r.dims(), &[2]);
    assert_close(&r.to_flat_vec::<f32>().unwrap(), &[6.0, 15.0], 1e-6);
}

#[test]
fn test_sum_3d_along_middle_dim() {
    // [2,3,2] sum along dim 1 -> [2,2]
    let data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 2]);
    let r = t.sum(1).unwrap();
    assert_eq!(r.dims(), &[2, 2]);
    // Batch 0: col0 = 1+3+5=9, col1 = 2+4+6=12
    // Batch 1: col0 = 7+9+11=27, col1 = 8+10+12=30
    assert_close(
        &r.to_flat_vec::<f32>().unwrap(),
        &[9.0, 12.0, 27.0, 30.0],
        1e-5,
    );
}

#[test]
fn test_mean_dim1() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let r = t.mean(1).unwrap();
    assert_eq!(r.dims(), &[2]);
    assert_close(&r.to_flat_vec::<f32>().unwrap(), &[2.0, 5.0], 1e-6);
}

#[test]
fn test_mean_dim0() {
    let t = t2d(&[2.0, 4.0, 6.0, 8.0], 2, 2);
    let r = t.mean(0).unwrap();
    assert_eq!(r.dims(), &[2]);
    assert_close(&r.to_flat_vec::<f32>().unwrap(), &[4.0, 6.0], 1e-6);
}

#[test]
fn test_max_dim1() {
    let t = t2d(&[1.0, 3.0, 2.0, 6.0, 4.0, 5.0], 2, 3);
    let r = t.max(1).unwrap();
    assert_eq!(r.dims(), &[2]);
    assert_close(&r.to_flat_vec::<f32>().unwrap(), &[3.0, 6.0], 1e-6);
}

#[test]
fn test_max_dim0() {
    let t = t2d(&[1.0, 5.0, 3.0, 2.0], 2, 2);
    let r = t.max(0).unwrap();
    assert_eq!(r.dims(), &[2]);
    assert_close(&r.to_flat_vec::<f32>().unwrap(), &[3.0, 5.0], 1e-6);
}

#[test]
fn test_min_dim1() {
    let t = t2d(&[1.0, 3.0, 2.0, 6.0, 4.0, 5.0], 2, 3);
    let r = t.min(1).unwrap();
    assert_eq!(r.dims(), &[2]);
    assert_close(&r.to_flat_vec::<f32>().unwrap(), &[1.0, 4.0], 1e-6);
}

#[test]
fn test_min_dim0() {
    let t = t2d(&[3.0, 5.0, 1.0, 2.0], 2, 2);
    let r = t.min(0).unwrap();
    assert_eq!(r.dims(), &[2]);
    assert_close(&r.to_flat_vec::<f32>().unwrap(), &[1.0, 2.0], 1e-6);
}

#[test]
fn test_argmax_dim1() {
    let t = t2d(&[1.0, 3.0, 2.0, 6.0, 4.0, 5.0], 2, 3);
    let r = t.argmax(1).unwrap();
    assert_eq!(r.dims(), &[2]);
    assert_eq!(r.dtype(), DType::U32);
    assert_eq!(r.to_flat_vec::<u32>().unwrap(), vec![1, 0]);
}

#[test]
fn test_argmax_dim0() {
    let t = t2d(&[1.0, 5.0, 3.0, 2.0], 2, 2);
    let r = t.argmax(0).unwrap();
    assert_eq!(r.dims(), &[2]);
    assert_eq!(r.dtype(), DType::U32);
    assert_eq!(r.to_flat_vec::<u32>().unwrap(), vec![1, 0]);
}

#[test]
fn test_argmin_dim1() {
    let t = t2d(&[3.0, 1.0, 2.0, 5.0, 6.0, 4.0], 2, 3);
    let r = t.argmin(1).unwrap();
    assert_eq!(r.dims(), &[2]);
    assert_eq!(r.dtype(), DType::U32);
    assert_eq!(r.to_flat_vec::<u32>().unwrap(), vec![1, 2]);
}

#[test]
fn test_argmin_dim0() {
    let t = t2d(&[5.0, 1.0, 3.0, 4.0], 2, 2);
    let r = t.argmin(0).unwrap();
    assert_eq!(r.dims(), &[2]);
    assert_eq!(r.dtype(), DType::U32);
    assert_eq!(r.to_flat_vec::<u32>().unwrap(), vec![1, 0]);
}

// =============================================================================
// Shape ops: reshape, transpose, permute, squeeze, unsqueeze, expand
// =============================================================================

#[test]
fn test_reshape_2d_to_3d() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let r = t.reshape([3, 2]).unwrap();
    assert_eq!(r.dims(), &[3, 2]);
    assert_eq!(
        r.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn test_reshape_flatten() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 4]);
    let r = t.reshape([24]).unwrap();
    assert_eq!(r.dims(), &[24]);
    assert_eq!(r.numel(), 24);
}

#[test]
fn test_reshape_numel_mismatch_fails() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0]);
    assert!(t.reshape([3, 2]).is_err());
}

#[test]
fn test_transpose_2d() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let r = t.transpose(0, 1).unwrap();
    assert_eq!(r.dims(), &[3, 2]);
    assert_eq!(
        r.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

#[test]
fn test_transpose_3d() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 4]);
    let r = t.transpose(0, 2).unwrap();
    assert_eq!(r.dims(), &[4, 3, 2]);
}

#[test]
fn test_permute_3d() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = tnd(&data, &[2, 3, 4]);
    let r = t.permute([2, 0, 1]).unwrap();
    assert_eq!(r.dims(), &[4, 2, 3]);
}

#[test]
fn test_permute_identity() {
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let t = t2d(&data, 2, 3);
    let r = t.permute([0, 1]).unwrap();
    assert_eq!(r.dims(), &[2, 3]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_permute_duplicate_axes_fails() {
    let t = DynTensor::zeros(&[2, 3], DType::F32, &cpu()).unwrap();
    assert!(t.permute([0, 0]).is_err());
}

#[test]
fn test_squeeze_removes_unit_dim() {
    let t = tnd(&[1.0, 2.0, 3.0], &[1, 3]);
    let r = t.squeeze(0).unwrap();
    assert_eq!(r.dims(), &[3]);
}

#[test]
fn test_squeeze_non_unit_fails() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    assert!(t.squeeze(0).is_err());
}

#[test]
fn test_unsqueeze_dim0() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let r = t.unsqueeze(0).unwrap();
    assert_eq!(r.dims(), &[1, 3]);
}

#[test]
fn test_unsqueeze_dim1() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let r = t.unsqueeze(1).unwrap();
    assert_eq!(r.dims(), &[3, 1]);
}

#[test]
fn test_squeeze_unsqueeze_roundtrip() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let expanded = t.unsqueeze(1).unwrap();
    assert_eq!(expanded.dims(), &[2, 1, 3]);
    let back = expanded.squeeze(1).unwrap();
    assert_eq!(back.dims(), &[2, 3]);
    assert_eq!(
        back.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn test_expand_broadcast() {
    let t = tnd(&[1.0, 2.0, 3.0], &[1, 3]);
    let r = t.expand([4, 3]).unwrap();
    assert_eq!(r.dims(), &[4, 3]);
    let flat = r.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat.len(), 12);
    // Each row should be [1, 2, 3]
    for row in flat.chunks(3) {
        assert_close(row, &[1.0, 2.0, 3.0], 1e-6);
    }
}

#[test]
fn test_expand_3d() {
    let t = tnd(&[1.0, 2.0], &[1, 1, 2]);
    let r = t.expand([3, 4, 2]).unwrap();
    assert_eq!(r.dims(), &[3, 4, 2]);
    assert_eq!(r.numel(), 24);
}

#[test]
fn test_flatten_all() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let flat = t.flatten_all().unwrap();
    assert_eq!(flat.dims(), &[6]);
    assert_eq!(
        flat.to_vec1::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

// =============================================================================
// Indexing ops: narrow, index_select, gather, scatter
// =============================================================================

#[test]
fn test_narrow_1d() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let r = t.narrow(0, 1, 3).unwrap();
    assert_eq!(r.dims(), &[3]);
    assert_eq!(r.to_vec1::<f32>().unwrap(), vec![2.0, 3.0, 4.0]);
}

#[test]
fn test_narrow_2d_dim0() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    let r = t.narrow(0, 0, 2).unwrap();
    assert_eq!(r.dims(), &[2, 2]);
    assert_close(
        &r.to_flat_vec::<f32>().unwrap(),
        &[1.0, 2.0, 3.0, 4.0],
        1e-6,
    );
}

#[test]
fn test_narrow_2d_dim1() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let r = t.narrow(1, 1, 2).unwrap();
    assert_eq!(r.dims(), &[2, 2]);
    assert_close(
        &r.to_flat_vec::<f32>().unwrap(),
        &[2.0, 3.0, 5.0, 6.0],
        1e-6,
    );
}

#[test]
fn test_narrow_out_of_bounds_fails() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    assert!(t.narrow(0, 2, 3).is_err());
}

#[test]
fn test_index_select_1d() {
    let t = t1d(&[10.0, 20.0, 30.0, 40.0, 50.0]);
    let ids = DynTensor::from_cpu_u32(
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![0u32, 2, 4]).unwrap(),
    )
    .unwrap();
    let r = t.index_select(&ids, 0).unwrap();
    assert_eq!(r.dims(), &[3]);
    assert_close(&r.to_flat_vec::<f32>().unwrap(), &[10.0, 30.0, 50.0], 1e-6);
}

#[test]
fn test_index_select_2d_dim0() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    let ids = DynTensor::from_cpu_u32(
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![0u32, 2]).unwrap(),
    )
    .unwrap();
    let r = t.index_select(&ids, 0).unwrap();
    assert_eq!(r.dims(), &[2, 2]);
    assert_close(
        &r.to_flat_vec::<f32>().unwrap(),
        &[1.0, 2.0, 5.0, 6.0],
        1e-6,
    );
}

#[test]
fn test_gather_2d() {
    // gather along dim 1: select per-row column indices
    let t = t2d(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], 2, 3);
    let ids = DynTensor::from_cpu_u32(
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![0u32, 2, 1, 0]).unwrap(),
    )
    .unwrap();
    let r = t.gather(&ids, 1).unwrap();
    assert_eq!(r.dims(), &[2, 2]);
    assert_close(
        &r.to_flat_vec::<f32>().unwrap(),
        &[10.0, 30.0, 50.0, 40.0],
        1e-6,
    );
}

#[test]
fn test_get_selects_along_dim0() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
    let r = t.get(1).unwrap();
    assert_eq!(r.dims(), &[2]);
    assert_close(&r.to_flat_vec::<f32>().unwrap(), &[3.0, 4.0], 1e-6);
}

// =============================================================================
// Math ops: exp, log, sqrt, abs, clamp, pow
// =============================================================================

#[test]
fn test_exp() {
    let t = t1d(&[0.0, 1.0, 2.0]);
    let r = t.exp().unwrap();
    let flat = r.to_flat_vec::<f32>().unwrap();
    assert_close(
        &flat,
        &[
            1.0,
            std::f32::consts::E,
            std::f32::consts::E * std::f32::consts::E,
        ],
        1e-4,
    );
}

#[test]
fn test_log() {
    let t = t1d(&[
        1.0,
        std::f32::consts::E,
        std::f32::consts::E * std::f32::consts::E,
    ]);
    let r = t.log().unwrap();
    let flat = r.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[0.0, 1.0, 2.0], 1e-4);
}

#[test]
fn test_sqrt() {
    let t = t1d(&[0.0, 1.0, 4.0, 9.0, 16.0]);
    let r = t.sqrt().unwrap();
    assert_close(
        &r.to_flat_vec::<f32>().unwrap(),
        &[0.0, 1.0, 2.0, 3.0, 4.0],
        1e-5,
    );
}

#[test]
fn test_abs_negative_values() {
    let t = t1d(&[-5.0, -0.0, 0.0, 3.14]);
    let r = t.abs().unwrap();
    assert_close(
        &r.to_flat_vec::<f32>().unwrap(),
        &[5.0, 0.0, 0.0, 3.14],
        1e-6,
    );
}

#[test]
fn test_clamp() {
    let t = t1d(&[-5.0, -1.0, 0.0, 1.0, 5.0, 10.0]);
    let r = t.clamp(-2.0, 3.0).unwrap();
    assert_close(
        &r.to_flat_vec::<f32>().unwrap(),
        &[-2.0, -1.0, 0.0, 1.0, 3.0, 3.0],
        1e-6,
    );
}

#[test]
fn test_clamp_only_upper() {
    let t = t1d(&[1.0, 10.0, 100.0]);
    let r = t.clamp(f64::NEG_INFINITY, 50.0).unwrap();
    assert_close(&r.to_flat_vec::<f32>().unwrap(), &[1.0, 10.0, 50.0], 1e-6);
}

#[test]
fn test_powf() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0]);
    let r = t.powf(2.0).unwrap();
    assert_close(
        &r.to_flat_vec::<f32>().unwrap(),
        &[1.0, 4.0, 9.0, 16.0],
        1e-5,
    );
}

#[test]
fn test_powf_fractional() {
    let t = t1d(&[4.0, 9.0, 16.0]);
    let r = t.powf(0.5).unwrap();
    assert_close(&r.to_flat_vec::<f32>().unwrap(), &[2.0, 3.0, 4.0], 1e-5);
}

#[test]
fn test_sqr() {
    let t = t1d(&[-3.0, -1.0, 0.0, 2.0, 5.0]);
    let r = t.sqr().unwrap();
    assert_close(
        &r.to_flat_vec::<f32>().unwrap(),
        &[9.0, 1.0, 0.0, 4.0, 25.0],
        1e-5,
    );
}

#[test]
fn test_recip() {
    let t = t1d(&[1.0, 2.0, 4.0, 5.0]);
    let r = t.recip().unwrap();
    assert_close(
        &r.to_flat_vec::<f32>().unwrap(),
        &[1.0, 0.5, 0.25, 0.2],
        1e-5,
    );
}

#[test]
fn test_sin_cos() {
    let t = t1d(&[0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI]);
    let s = t.sin().unwrap();
    let c = t.cos().unwrap();
    assert_close(&s.to_flat_vec::<f32>().unwrap(), &[0.0, 1.0, 0.0], 1e-5);
    assert_close(&c.to_flat_vec::<f32>().unwrap(), &[1.0, 0.0, -1.0], 1e-5);
}

#[test]
fn test_relu() {
    let t = t1d(&[-3.0, -1.0, 0.0, 1.0, 3.0]);
    let r = t.relu().unwrap();
    assert_close(
        &r.to_flat_vec::<f32>().unwrap(),
        &[0.0, 0.0, 0.0, 1.0, 3.0],
        1e-6,
    );
}

#[test]
fn test_sigmoid() {
    let t = t1d(&[0.0]);
    let r = t.sigmoid().unwrap();
    assert_close(&r.to_flat_vec::<f32>().unwrap(), &[0.5], 1e-5);
}

#[test]
fn test_tanh() {
    let t = t1d(&[0.0]);
    let r = t.tanh().unwrap();
    assert_close(&r.to_flat_vec::<f32>().unwrap(), &[0.0], 1e-5);
}

// =============================================================================
// Matmul: 2D, batched, broadcast
// =============================================================================

#[test]
fn test_matmul_2d() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = t2d(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
    let r = a.matmul(&b).unwrap();
    assert_eq!(r.dims(), &[2, 2]);
    assert_close(
        &r.to_flat_vec::<f32>().unwrap(),
        &[58.0, 64.0, 139.0, 154.0],
        1e-4,
    );
}

#[test]
fn test_matmul_identity() {
    // A @ I = A
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let eye = t2d(&[1.0, 0.0, 0.0, 1.0], 2, 2);
    let r = a.matmul(&eye).unwrap();
    assert_eq!(r.dims(), &[2, 2]);
    assert_close(
        &r.to_flat_vec::<f32>().unwrap(),
        &[1.0, 2.0, 3.0, 4.0],
        1e-5,
    );
}

#[test]
fn test_matmul_batched() {
    // [2,2,3] @ [2,3,2] = [2,2,2]
    let a_data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (1..=12).map(|i| i as f32 * 0.1).collect();
    let a = tnd(&a_data, &[2, 2, 3]);
    let b = tnd(&b_data, &[2, 3, 2]);
    let r = a.matmul(&b).unwrap();
    assert_eq!(r.dims(), &[2, 2, 2]);
    let flat = r.to_flat_vec::<f32>().unwrap();
    // Batch 0: [[1,2,3],[4,5,6]] @ [[0.1,0.2],[0.3,0.4],[0.5,0.6]]
    assert_close(&flat[0..2], &[2.2, 2.8], 1e-4);
    assert_close(&flat[2..4], &[4.9, 6.4], 1e-4);
}

#[test]
fn test_matmul_vec_vec_dot() {
    // 1D x 1D is treated as dot product -> scalar or [1,1]
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[4.0, 5.0, 6.0]);
    // matmul with 1D may need unsqueeze depending on implementation
    let a2d = a.unsqueeze(0).unwrap(); // [1,3]
    let b2d = b.unsqueeze(1).unwrap(); // [3,1]
    let r = a2d.matmul(&b2d).unwrap();
    assert_eq!(r.dims(), &[1, 1]);
    let val = r.to_flat_vec::<f32>().unwrap()[0];
    assert!((val - 32.0).abs() < 1e-4); // 1*4 + 2*5 + 3*6 = 32
}

// =============================================================================
// Concatenation: cat along different dims
// =============================================================================

#[test]
fn test_cat_1d() {
    let a = t1d(&[1.0, 2.0]);
    let b = t1d(&[3.0, 4.0, 5.0]);
    let r = DynTensor::cat(&[&a, &b], 0).unwrap();
    assert_eq!(r.dims(), &[5]);
    assert_eq!(r.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_cat_2d_dim0() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = t2d(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 2, 3);
    let r = DynTensor::cat(&[&a, &b], 0).unwrap();
    assert_eq!(r.dims(), &[4, 3]);
}

#[test]
fn test_cat_2d_dim1() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = t2d(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 2, 3);
    let r = DynTensor::cat(&[&a, &b], 1).unwrap();
    assert_eq!(r.dims(), &[2, 5]);
    let flat = r.to_flat_vec::<f32>().unwrap();
    assert_close(&flat[0..5], &[1.0, 2.0, 5.0, 6.0, 7.0], 1e-6);
    assert_close(&flat[5..10], &[3.0, 4.0, 8.0, 9.0, 10.0], 1e-6);
}

#[test]
fn test_cat_three_tensors() {
    let a = t1d(&[1.0]);
    let b = t1d(&[2.0, 3.0]);
    let c = t1d(&[4.0, 5.0, 6.0]);
    let r = DynTensor::cat(&[&a, &b, &c], 0).unwrap();
    assert_eq!(r.dims(), &[6]);
    assert_eq!(
        r.to_vec1::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn test_stack_dim0() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[4.0, 5.0, 6.0]);
    let r = DynTensor::stack(&[&a, &b], 0).unwrap();
    assert_eq!(r.dims(), &[2, 3]);
    assert_eq!(
        r.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn test_stack_dim1() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[4.0, 5.0, 6.0]);
    let r = DynTensor::stack(&[&a, &b], 1).unwrap();
    assert_eq!(r.dims(), &[3, 2]);
    assert_eq!(
        r.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

#[test]
fn test_chunk_even() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let chunks = t.chunk(3, 0).unwrap();
    assert_eq!(chunks.len(), 3);
    for c in &chunks {
        assert_eq!(c.dims(), &[2]);
    }
    assert_eq!(chunks[0].to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);
    assert_eq!(chunks[1].to_vec1::<f32>().unwrap(), vec![3.0, 4.0]);
    assert_eq!(chunks[2].to_vec1::<f32>().unwrap(), vec![5.0, 6.0]);
}

#[test]
fn test_chunk_uneven() {
    let t = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let chunks = t.chunk(2, 0).unwrap();
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].dims(), &[3]);
    assert_eq!(chunks[1].dims(), &[2]);
}

#[test]
fn test_split() {
    let t = t2d(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        2,
        6,
    );
    let splits = t.split([2, 4], 1).unwrap();
    assert_eq!(splits.len(), 2);
    assert_eq!(splits[0].dims(), &[2, 2]);
    assert_eq!(splits[1].dims(), &[2, 4]);
}

// =============================================================================
// Dtype conversion: to_dtype for F32/BF16/F16
// =============================================================================

#[test]
fn test_to_dtype_f32_to_bf16_roundtrip() {
    let t = t1d(&[1.0, 2.0, 3.0, 0.5]);
    let bf16 = t.to_dtype(DType::BF16).unwrap();
    assert_eq!(bf16.dtype(), DType::BF16);
    assert_eq!(bf16.dims(), &[4]);
    let back = bf16.to_dtype(DType::F32).unwrap();
    assert_eq!(back.dtype(), DType::F32);
    assert_close(
        &back.to_flat_vec::<f32>().unwrap(),
        &[1.0, 2.0, 3.0, 0.5],
        0.02, // BF16 has limited precision
    );
}

#[test]
fn test_to_dtype_f32_to_f16_roundtrip() {
    let t = t1d(&[1.0, 2.0, 3.0, 0.5]);
    let f16 = t.to_dtype(DType::F16).unwrap();
    assert_eq!(f16.dtype(), DType::F16);
    assert_eq!(f16.dims(), &[4]);
    let back = f16.to_dtype(DType::F32).unwrap();
    assert_eq!(back.dtype(), DType::F32);
    assert_close(
        &back.to_flat_vec::<f32>().unwrap(),
        &[1.0, 2.0, 3.0, 0.5],
        1e-3,
    );
}

#[test]
fn test_to_dtype_same_noop() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let same = t.to_dtype(DType::F32).unwrap();
    assert_eq!(same.dtype(), DType::F32);
    assert_eq!(
        same.to_flat_vec::<f32>().unwrap(),
        t.to_flat_vec::<f32>().unwrap()
    );
}

#[test]
fn test_to_dtype_bf16_to_f16() {
    let t = t1d(&[1.0, 2.0]);
    let bf16 = t.to_dtype(DType::BF16).unwrap();
    let f16 = bf16.to_dtype(DType::F16).unwrap();
    assert_eq!(f16.dtype(), DType::F16);
    let back = f16.to_dtype(DType::F32).unwrap();
    assert_close(&back.to_flat_vec::<f32>().unwrap(), &[1.0, 2.0], 0.02);
}

// =============================================================================
// Device ops: to_device (CPU)
// =============================================================================

#[test]
fn test_to_device_cpu_noop() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let on_cpu = t.to_device(&Device::Cpu).unwrap();
    assert_eq!(on_cpu.device(), Device::Cpu);
    assert_eq!(on_cpu.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_to_device_preserves_shape_and_dtype() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let r = t.to_device(&Device::Cpu).unwrap();
    assert_eq!(r.dims(), &[2, 2]);
    assert_eq!(r.dtype(), DType::F32);
}

// =============================================================================
// Creation ops: zeros, ones, full, arange, zeros_like, ones_like
// =============================================================================

#[test]
fn test_zeros_shape_and_values() {
    let t = DynTensor::zeros(&[3, 4], DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims(), &[3, 4]);
    assert!(t.to_flat_vec::<f32>().unwrap().iter().all(|&v| v == 0.0));
}

#[test]
fn test_ones_shape_and_values() {
    let t = DynTensor::ones(&[2, 3], DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims(), &[2, 3]);
    assert!(t.to_flat_vec::<f32>().unwrap().iter().all(|&v| v == 1.0));
}

#[test]
fn test_full_arbitrary_value() {
    let t = DynTensor::full(&[2, 2], 3.14, DType::F32, &cpu()).unwrap();
    let flat = t.to_flat_vec::<f32>().unwrap();
    for &v in &flat {
        assert!((v - 3.14_f32).abs() < 1e-5);
    }
}

#[test]
fn test_full_u32() {
    let t = DynTensor::full(&[3], 42.0, DType::U32, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::U32);
    assert_eq!(t.to_flat_vec::<u32>().unwrap(), vec![42, 42, 42]);
}

#[test]
fn test_full_negative_u32_fails() {
    assert!(DynTensor::full(&[1], -1.0, DType::U32, &cpu()).is_err());
}

#[test]
fn test_arange_basic() {
    let t = DynTensor::arange(0.0, 5.0, &cpu()).unwrap();
    assert_eq!(t.dims(), &[5]);
    assert_eq!(t.to_vec1::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_arange_nonzero_start() {
    let t = DynTensor::arange(3.0, 7.0, &cpu()).unwrap();
    assert_eq!(t.dims(), &[4]);
    assert_eq!(t.to_vec1::<f32>().unwrap(), vec![3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_arange_step() {
    let t = DynTensor::arange_step(0.0, 10.0, 2.0, &cpu()).unwrap();
    assert_eq!(t.dims(), &[5]);
    assert_eq!(t.to_vec1::<f32>().unwrap(), vec![0.0, 2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_arange_empty() {
    let t = DynTensor::arange(5.0, 5.0, &cpu()).unwrap();
    assert_eq!(t.dims(), &[0]);
    assert_eq!(t.numel(), 0);
}

#[test]
fn test_zeros_like() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let z = t.zeros_like().unwrap();
    assert_eq!(z.dims(), &[2, 2]);
    assert_eq!(z.dtype(), DType::F32);
    assert!(z.to_flat_vec::<f32>().unwrap().iter().all(|&v| v == 0.0));
}

#[test]
fn test_ones_like() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let o = t.ones_like().unwrap();
    assert_eq!(o.dims(), &[2, 2]);
    assert_eq!(o.dtype(), DType::F32);
    assert!(o.to_flat_vec::<f32>().unwrap().iter().all(|&v| v == 1.0));
}

#[test]
fn test_full_like() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let fl = t.full_like(7.0).unwrap();
    assert_eq!(fl.dims(), &[3]);
    assert_close(&fl.to_flat_vec::<f32>().unwrap(), &[7.0, 7.0, 7.0], 1e-6);
}

// =============================================================================
// Contiguity: is_contiguous, contiguous()
// =============================================================================

#[test]
fn test_is_contiguous_fresh_tensor() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    assert!(t.is_contiguous());
}

#[test]
fn test_contiguous_preserves_data() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let c = t.contiguous().unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert!(c.is_contiguous());
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn test_contiguous_after_transpose() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let tr = t.transpose(0, 1).unwrap();
    // After transpose, make contiguous and verify data
    let c = tr.contiguous().unwrap();
    assert_eq!(c.dims(), &[3, 2]);
    assert!(c.is_contiguous());
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

#[test]
fn test_detach_is_clone() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let d = t.detach();
    assert_eq!(d.dims(), t.dims());
    assert_eq!(
        d.to_flat_vec::<f32>().unwrap(),
        t.to_flat_vec::<f32>().unwrap()
    );
}

// =============================================================================
// Shape query helpers
// =============================================================================

#[test]
fn test_dims1() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    assert_eq!(t.dims1().unwrap(), 3);
}

#[test]
fn test_dims2() {
    let t = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    assert_eq!(t.dims2().unwrap(), (2, 3));
}

#[test]
fn test_dims3() {
    let t = tnd(&(0..24).map(|i| i as f32).collect::<Vec<_>>(), &[2, 3, 4]);
    assert_eq!(t.dims3().unwrap(), (2, 3, 4));
}

#[test]
fn test_dims4() {
    let t = DynTensor::zeros(&[2, 3, 4, 5], DType::F32, &cpu()).unwrap();
    assert_eq!(t.dims4().unwrap(), (2, 3, 4, 5));
}

#[test]
fn test_rank_and_numel() {
    let t = tnd(&(0..24).map(|i| i as f32).collect::<Vec<_>>(), &[2, 3, 4]);
    assert_eq!(t.rank(), 3);
    assert_eq!(t.numel(), 24);
}

#[test]
fn test_to_scalar() {
    let t = DynTensor::full(&[], 42.0, DType::F32, &cpu()).unwrap();
    assert_eq!(t.to_scalar::<f32>().unwrap(), 42.0);
}

#[test]
fn test_to_scalar_non_scalar_fails() {
    let t = t1d(&[1.0, 2.0]);
    assert!(t.to_scalar::<f32>().is_err());
}

// =============================================================================
// Operator overloads
// =============================================================================

#[test]
fn test_add_operator() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[4.0, 5.0, 6.0]);
    let r = (&a + &b).unwrap();
    assert_close(&r.to_flat_vec::<f32>().unwrap(), &[5.0, 7.0, 9.0], 1e-6);
}

#[test]
fn test_sub_operator() {
    let a = t1d(&[10.0, 20.0, 30.0]);
    let b = t1d(&[1.0, 2.0, 3.0]);
    let r = (&a - &b).unwrap();
    assert_close(&r.to_flat_vec::<f32>().unwrap(), &[9.0, 18.0, 27.0], 1e-6);
}

#[test]
fn test_mul_operator() {
    let a = t1d(&[2.0, 3.0, 4.0]);
    let b = t1d(&[5.0, 6.0, 7.0]);
    let r = (&a * &b).unwrap();
    assert_close(&r.to_flat_vec::<f32>().unwrap(), &[10.0, 18.0, 28.0], 1e-6);
}

#[test]
fn test_div_operator() {
    let a = t1d(&[10.0, 20.0, 30.0]);
    let b = t1d(&[2.0, 5.0, 10.0]);
    let r = (&a / &b).unwrap();
    assert_close(&r.to_flat_vec::<f32>().unwrap(), &[5.0, 4.0, 3.0], 1e-6);
}
