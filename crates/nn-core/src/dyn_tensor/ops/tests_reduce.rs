// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for DynTensor reduction operations:
//! sum, mean, max, min, argmax, argmin, sum_all, mean_all,
//! var, keepdim variants, negative-dim indexing, edge cases,
//! and broadcasting after reduction.

use crate::dyn_tensor::test_helpers::{approx_eq, cpu, t1d, t2d, tnd};
use crate::DType;
use crate::DynTensor;

// =============================================================================
// sum along each axis — 2D
// =============================================================================

#[test]
fn test_sum_2d_axis0() {
    // [[1, 2, 3], [4, 5, 6]] -> sum(axis=0) -> [5, 7, 9]
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let s = a.sum(0).unwrap();
    assert_eq!(s.dims(), &[3]);
    assert_eq!(s.to_vec1::<f32>().unwrap(), vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_sum_2d_axis1() {
    // [[1, 2, 3], [4, 5, 6]] -> sum(axis=1) -> [6, 15]
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let s = a.sum(1).unwrap();
    assert_eq!(s.dims(), &[2]);
    assert_eq!(s.to_vec1::<f32>().unwrap(), vec![6.0, 15.0]);
}

// =============================================================================
// sum along each axis — 3D
// =============================================================================

#[test]
fn test_sum_3d_axis0() {
    // shape [2, 2, 3]:
    // [[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]]
    // sum(axis=0) -> [[8,10,12],[14,16,18]]
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let a = tnd(&data, &[2, 2, 3]);
    let s = a.sum(0).unwrap();
    assert_eq!(s.dims(), &[2, 3]);
    assert_eq!(
        s.to_flat_vec::<f32>().unwrap(),
        vec![8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
    );
}

#[test]
fn test_sum_3d_axis1() {
    // shape [2, 2, 3]:
    // sum(axis=1) -> [[5,7,9],[17,19,21]]
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let a = tnd(&data, &[2, 2, 3]);
    let s = a.sum(1).unwrap();
    assert_eq!(s.dims(), &[2, 3]);
    assert_eq!(
        s.to_flat_vec::<f32>().unwrap(),
        vec![5.0, 7.0, 9.0, 17.0, 19.0, 21.0]
    );
}

#[test]
fn test_sum_3d_axis2() {
    // shape [2, 2, 3]:
    // sum(axis=2) -> [[6,15],[24,33]]
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let a = tnd(&data, &[2, 2, 3]);
    let s = a.sum(2).unwrap();
    assert_eq!(s.dims(), &[2, 2]);
    assert_eq!(s.to_flat_vec::<f32>().unwrap(), vec![6.0, 15.0, 24.0, 33.0]);
}

// =============================================================================
// sum keepdim true vs false
// =============================================================================

#[test]
fn test_sum_keepdim_true_2d() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let s = a.sum_keepdim(1).unwrap();
    assert_eq!(s.dims(), &[2, 1]); // dim 1 collapsed to size 1
    assert_eq!(s.to_flat_vec::<f32>().unwrap(), vec![6.0, 15.0]);
}

#[test]
fn test_sum_keepdim_false_2d() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let s = a.sum(1).unwrap(); // non-keepdim
    assert_eq!(s.dims(), &[2]); // dim 1 removed entirely
    assert_eq!(s.to_vec1::<f32>().unwrap(), vec![6.0, 15.0]);
}

#[test]
fn test_sum_keepdim_true_axis0() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let s = a.sum_keepdim(0).unwrap();
    assert_eq!(s.dims(), &[1, 3]);
    assert_eq!(s.to_flat_vec::<f32>().unwrap(), vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_sum_keepdim_3d() {
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let a = tnd(&data, &[2, 2, 3]);
    let s = a.sum_keepdim(1).unwrap();
    assert_eq!(s.dims(), &[2, 1, 3]);
    assert_eq!(
        s.to_flat_vec::<f32>().unwrap(),
        vec![5.0, 7.0, 9.0, 17.0, 19.0, 21.0]
    );
}

// =============================================================================
// mean along each axis
// =============================================================================

#[test]
fn test_mean_2d_axis0() {
    // [[1,2,3],[4,5,6]] -> mean(axis=0) -> [2.5, 3.5, 4.5]
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let m = a.mean(0).unwrap();
    assert_eq!(m.dims(), &[3]);
    let vals = m.to_vec1::<f32>().unwrap();
    assert!(approx_eq(vals[0], 2.5, 1e-6));
    assert!(approx_eq(vals[1], 3.5, 1e-6));
    assert!(approx_eq(vals[2], 4.5, 1e-6));
}

#[test]
fn test_mean_2d_axis1() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let m = a.mean(1).unwrap();
    assert_eq!(m.dims(), &[2]);
    let vals = m.to_vec1::<f32>().unwrap();
    assert!(approx_eq(vals[0], 2.0, 1e-6));
    assert!(approx_eq(vals[1], 5.0, 1e-6));
}

#[test]
fn test_mean_3d_axis2() {
    // shape [2,2,3], mean along last axis
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let a = tnd(&data, &[2, 2, 3]);
    let m = a.mean(2).unwrap();
    assert_eq!(m.dims(), &[2, 2]);
    let vals = m.to_flat_vec::<f32>().unwrap();
    assert!(approx_eq(vals[0], 2.0, 1e-6)); // mean(1,2,3)
    assert!(approx_eq(vals[1], 5.0, 1e-6)); // mean(4,5,6)
    assert!(approx_eq(vals[2], 8.0, 1e-6)); // mean(7,8,9)
    assert!(approx_eq(vals[3], 11.0, 1e-6)); // mean(10,11,12)
}

#[test]
fn test_mean_keepdim_axis0() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let m = a.mean_keepdim(0).unwrap();
    assert_eq!(m.dims(), &[1, 3]);
    let vals = m.to_flat_vec::<f32>().unwrap();
    assert!(approx_eq(vals[0], 2.5, 1e-6));
    assert!(approx_eq(vals[1], 3.5, 1e-6));
    assert!(approx_eq(vals[2], 4.5, 1e-6));
}

// =============================================================================
// max along each axis
// =============================================================================

#[test]
fn test_max_2d_axis0() {
    // [[1,5,3],[4,2,6]] -> max(axis=0) -> [4, 5, 6]
    let a = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let m = a.max(0).unwrap();
    assert_eq!(m.dims(), &[3]);
    assert_eq!(m.to_vec1::<f32>().unwrap(), vec![4.0, 5.0, 6.0]);
}

#[test]
fn test_max_2d_axis1() {
    // [[1,5,3],[4,2,6]] -> max(axis=1) -> [5, 6]
    let a = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let m = a.max(1).unwrap();
    assert_eq!(m.dims(), &[2]);
    assert_eq!(m.to_vec1::<f32>().unwrap(), vec![5.0, 6.0]);
}

#[test]
fn test_max_3d_axis0() {
    // shape [2,2,2]: [[[1,4],[3,2]], [[5,0],[1,6]]]
    let a = tnd(&[1.0, 4.0, 3.0, 2.0, 5.0, 0.0, 1.0, 6.0], &[2, 2, 2]);
    let m = a.max(0).unwrap();
    assert_eq!(m.dims(), &[2, 2]);
    assert_eq!(m.to_flat_vec::<f32>().unwrap(), vec![5.0, 4.0, 3.0, 6.0]);
}

// =============================================================================
// min along each axis
// =============================================================================

#[test]
fn test_min_2d_axis0() {
    let a = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let m = a.min(0).unwrap();
    assert_eq!(m.dims(), &[3]);
    assert_eq!(m.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_min_2d_axis1() {
    let a = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let m = a.min(1).unwrap();
    assert_eq!(m.dims(), &[2]);
    assert_eq!(m.to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn test_min_3d_axis2() {
    // shape [2,2,3]: min along last axis
    let data: Vec<f32> = vec![
        3.0, 1.0, 2.0, 6.0, 4.0, 5.0, 9.0, 7.0, 8.0, 12.0, 10.0, 11.0,
    ];
    let a = tnd(&data, &[2, 2, 3]);
    let m = a.min(2).unwrap();
    assert_eq!(m.dims(), &[2, 2]);
    assert_eq!(m.to_flat_vec::<f32>().unwrap(), vec![1.0, 4.0, 7.0, 10.0]);
}

// =============================================================================
// argmax — returns correct indices
// =============================================================================

#[test]
fn test_argmax_1d() {
    let a = t1d(&[1.0, 5.0, 3.0, 2.0]);
    let idx = a.argmax(0).unwrap();
    assert_eq!(idx.dims(), &[] as &[usize]); // scalar
    assert_eq!(idx.to_scalar::<u32>().unwrap(), 1);
}

#[test]
fn test_argmax_2d_axis0() {
    // [[1, 5, 3], [4, 2, 6]] -> argmax(axis=0) -> [1, 0, 1]
    let a = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let idx = a.argmax(0).unwrap();
    assert_eq!(idx.dims(), &[3]);
    assert_eq!(idx.to_flat_vec::<u32>().unwrap(), vec![1, 0, 1]);
}

#[test]
fn test_argmax_2d_axis1() {
    // [[1, 5, 3], [4, 2, 6]] -> argmax(axis=1) -> [1, 2]
    let a = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let idx = a.argmax(1).unwrap();
    assert_eq!(idx.dims(), &[2]);
    assert_eq!(idx.to_flat_vec::<u32>().unwrap(), vec![1, 2]);
}

#[test]
fn test_argmax_keepdim() {
    let a = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let idx = a.argmax_keepdim(1).unwrap();
    assert_eq!(idx.dims(), &[2, 1]); // keepdim preserves axis as size 1
    assert_eq!(idx.to_flat_vec::<u32>().unwrap(), vec![1, 2]);
}

// =============================================================================
// argmin — returns correct indices
// =============================================================================

#[test]
fn test_argmin_1d() {
    let a = t1d(&[3.0, 1.0, 5.0, 2.0]);
    let idx = a.argmin(0).unwrap();
    assert_eq!(idx.dims(), &[] as &[usize]);
    assert_eq!(idx.to_scalar::<u32>().unwrap(), 1);
}

#[test]
fn test_argmin_2d_axis0() {
    // [[1, 5, 3], [4, 2, 6]] -> argmin(axis=0) -> [0, 1, 0]
    let a = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let idx = a.argmin(0).unwrap();
    assert_eq!(idx.dims(), &[3]);
    assert_eq!(idx.to_flat_vec::<u32>().unwrap(), vec![0, 1, 0]);
}

#[test]
fn test_argmin_2d_axis1() {
    // [[1, 5, 3], [4, 2, 6]] -> argmin(axis=1) -> [0, 1]
    let a = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let idx = a.argmin(1).unwrap();
    assert_eq!(idx.dims(), &[2]);
    assert_eq!(idx.to_flat_vec::<u32>().unwrap(), vec![0, 1]);
}

#[test]
fn test_argmin_keepdim() {
    let a = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let idx = a.argmin_keepdim(1).unwrap();
    assert_eq!(idx.dims(), &[2, 1]);
    assert_eq!(idx.to_flat_vec::<u32>().unwrap(), vec![0, 1]);
}

// =============================================================================
// sum_all — total reduction to scalar
// =============================================================================

#[test]
fn test_sum_all_1d() {
    let a = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let s = a.sum_all().unwrap();
    assert_eq!(s.dims(), &[] as &[usize]);
    assert_eq!(s.to_scalar::<f32>().unwrap(), 15.0);
}

#[test]
fn test_sum_all_3d() {
    // 1+2+...+24 = 300
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let a = tnd(&data, &[2, 3, 4]);
    let s = a.sum_all().unwrap();
    assert_eq!(s.to_scalar::<f32>().unwrap(), 300.0);
}

// =============================================================================
// mean_all — total reduction to scalar
// =============================================================================

#[test]
fn test_mean_all_2d() {
    // [[1,2],[3,4]] -> mean = 2.5
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let m = a.mean_all().unwrap();
    assert_eq!(m.dims(), &[] as &[usize]);
    assert!(approx_eq(m.to_scalar::<f32>().unwrap(), 2.5, 1e-6));
}

#[test]
fn test_mean_all_3d() {
    // 1..=24, mean = 12.5
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let a = tnd(&data, &[2, 3, 4]);
    let m = a.mean_all().unwrap();
    assert!(approx_eq(m.to_scalar::<f32>().unwrap(), 12.5, 1e-6));
}

// =============================================================================
// max_all / min_all — total reduction to scalar
// =============================================================================

#[test]
fn test_max_all_3d() {
    let data: Vec<f32> = vec![1.0, 99.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let a = tnd(&data, &[2, 2, 2]);
    assert_eq!(a.max_all().unwrap().to_scalar::<f32>().unwrap(), 99.0);
}

#[test]
fn test_min_all_3d() {
    let data: Vec<f32> = vec![10.0, 5.0, 3.0, -7.0, 8.0, 2.0, 1.0, 4.0];
    let a = tnd(&data, &[2, 2, 2]);
    assert_eq!(a.min_all().unwrap().to_scalar::<f32>().unwrap(), -7.0);
}

// =============================================================================
// Edge case: reduce along axis=0 vs axis=-1 (negative dim via i32)
// =============================================================================

#[test]
fn test_sum_negative_dim_last() {
    // -1 means last axis
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let s = a.sum(-1i32).unwrap();
    assert_eq!(s.dims(), &[2]);
    assert_eq!(s.to_vec1::<f32>().unwrap(), vec![6.0, 15.0]);
}

#[test]
fn test_mean_negative_dim_first() {
    // -2 on a 2D tensor means axis=0
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let m = a.mean(-2i32).unwrap();
    assert_eq!(m.dims(), &[3]);
    let vals = m.to_vec1::<f32>().unwrap();
    assert!(approx_eq(vals[0], 2.5, 1e-6));
    assert!(approx_eq(vals[1], 3.5, 1e-6));
    assert!(approx_eq(vals[2], 4.5, 1e-6));
}

#[test]
fn test_max_negative_dim() {
    let a = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let m = a.max(-1i32).unwrap();
    assert_eq!(m.dims(), &[2]);
    assert_eq!(m.to_vec1::<f32>().unwrap(), vec![5.0, 6.0]);
}

#[test]
fn test_argmax_negative_dim() {
    let a = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let idx = a.argmax(-1i32).unwrap();
    assert_eq!(idx.dims(), &[2]);
    assert_eq!(idx.to_flat_vec::<u32>().unwrap(), vec![1, 2]);
}

#[test]
fn test_argmin_negative_dim() {
    let a = t2d(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], 2, 3);
    let idx = a.argmin(-1i32).unwrap();
    assert_eq!(idx.dims(), &[2]);
    assert_eq!(idx.to_flat_vec::<u32>().unwrap(), vec![0, 1]);
}

// =============================================================================
// Edge case: single element tensor
// =============================================================================

#[test]
fn test_sum_single_element() {
    let a = t1d(&[42.0]);
    let s = a.sum(0).unwrap();
    assert_eq!(s.dims(), &[] as &[usize]);
    assert_eq!(s.to_scalar::<f32>().unwrap(), 42.0);
}

#[test]
fn test_mean_single_element() {
    let a = t1d(&[42.0]);
    let m = a.mean(0).unwrap();
    assert_eq!(m.dims(), &[] as &[usize]);
    assert!(approx_eq(m.to_scalar::<f32>().unwrap(), 42.0, 1e-6));
}

#[test]
fn test_max_single_element() {
    let a = t1d(&[42.0]);
    let m = a.max(0).unwrap();
    assert_eq!(m.to_scalar::<f32>().unwrap(), 42.0);
}

#[test]
fn test_min_single_element() {
    let a = t1d(&[-3.0]);
    let m = a.min(0).unwrap();
    assert_eq!(m.to_scalar::<f32>().unwrap(), -3.0);
}

#[test]
fn test_argmax_single_element() {
    let a = t1d(&[7.0]);
    let idx = a.argmax(0).unwrap();
    assert_eq!(idx.to_scalar::<u32>().unwrap(), 0);
}

#[test]
fn test_argmin_single_element() {
    let a = t1d(&[7.0]);
    let idx = a.argmin(0).unwrap();
    assert_eq!(idx.to_scalar::<u32>().unwrap(), 0);
}

#[test]
fn test_sum_all_single_element() {
    let a = DynTensor::full(&[1, 1, 1], 5.0, DType::F32, &cpu()).unwrap();
    let s = a.sum_all().unwrap();
    assert_eq!(s.to_scalar::<f32>().unwrap(), 5.0);
}

#[test]
fn test_mean_all_single_element() {
    let a = DynTensor::full(&[1, 1], 99.0, DType::F32, &cpu()).unwrap();
    let m = a.mean_all().unwrap();
    assert!(approx_eq(m.to_scalar::<f32>().unwrap(), 99.0, 1e-6));
}

// =============================================================================
// Edge case: all same values — argmax/argmin returns first index (index 0)
// =============================================================================

#[test]
fn test_argmax_all_same_returns_first_index() {
    let a = t1d(&[5.0, 5.0, 5.0, 5.0]);
    let idx = a.argmax(0).unwrap();
    // When all values are equal, first occurrence (index 0) should be returned
    assert_eq!(idx.to_scalar::<u32>().unwrap(), 0);
}

#[test]
fn test_argmin_all_same_returns_first_index() {
    let a = t1d(&[3.0, 3.0, 3.0]);
    let idx = a.argmin(0).unwrap();
    assert_eq!(idx.to_scalar::<u32>().unwrap(), 0);
}

#[test]
fn test_argmax_2d_all_same_per_row() {
    // Each row has identical values, argmax along axis=1 should return 0 for each row
    let a = t2d(&[7.0, 7.0, 7.0, 2.0, 2.0, 2.0], 2, 3);
    let idx = a.argmax(1).unwrap();
    assert_eq!(idx.to_flat_vec::<u32>().unwrap(), vec![0, 0]);
}

// =============================================================================
// Edge case: negative values
// =============================================================================

#[test]
fn test_sum_negative_values() {
    let a = t1d(&[-1.0, -2.0, -3.0]);
    let s = a.sum(0).unwrap();
    assert_eq!(s.to_scalar::<f32>().unwrap(), -6.0);
}

#[test]
fn test_mean_negative_values() {
    let a = t1d(&[-4.0, -2.0, -6.0]);
    let m = a.mean(0).unwrap();
    assert!(approx_eq(m.to_scalar::<f32>().unwrap(), -4.0, 1e-6));
}

#[test]
fn test_max_all_negative() {
    // max of all-negative values
    let a = t1d(&[-10.0, -5.0, -20.0, -1.0]);
    let m = a.max(0).unwrap();
    assert_eq!(m.to_scalar::<f32>().unwrap(), -1.0);
}

#[test]
fn test_min_all_negative() {
    let a = t1d(&[-10.0, -5.0, -20.0, -1.0]);
    let m = a.min(0).unwrap();
    assert_eq!(m.to_scalar::<f32>().unwrap(), -20.0);
}

#[test]
fn test_argmax_negative_values() {
    // argmax of [-10, -5, -20, -1] should be index 3 (-1 is largest)
    let a = t1d(&[-10.0, -5.0, -20.0, -1.0]);
    let idx = a.argmax(0).unwrap();
    assert_eq!(idx.to_scalar::<u32>().unwrap(), 3);
}

#[test]
fn test_argmin_negative_values() {
    // argmin of [-10, -5, -20, -1] should be index 2 (-20 is smallest)
    let a = t1d(&[-10.0, -5.0, -20.0, -1.0]);
    let idx = a.argmin(0).unwrap();
    assert_eq!(idx.to_scalar::<u32>().unwrap(), 2);
}

#[test]
fn test_sum_all_negative() {
    let a = t2d(&[-1.0, -2.0, -3.0, -4.0], 2, 2);
    assert_eq!(a.sum_all().unwrap().to_scalar::<f32>().unwrap(), -10.0);
}

#[test]
fn test_mean_all_negative() {
    let a = t2d(&[-2.0, -4.0, -6.0, -8.0], 2, 2);
    assert!(approx_eq(
        a.mean_all().unwrap().to_scalar::<f32>().unwrap(),
        -5.0,
        1e-6
    ));
}

// =============================================================================
// Broadcasting after reduction
// =============================================================================

#[test]
fn test_broadcast_add_after_sum_keepdim() {
    // Reduce then broadcast back: common pattern in normalization
    // x - mean(x, keepdim=True)
    let x = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let mean = x.mean_keepdim(1).unwrap(); // [2, 1]
    assert_eq!(mean.dims(), &[2, 1]);
    let centered = x.broadcast_sub(&mean).unwrap(); // [2, 3]
    assert_eq!(centered.dims(), &[2, 3]);
    let vals = centered.to_flat_vec::<f32>().unwrap();
    // Row 0: [1-2, 2-2, 3-2] = [-1, 0, 1]
    // Row 1: [4-5, 5-5, 6-5] = [-1, 0, 1]
    assert!(approx_eq(vals[0], -1.0, 1e-6));
    assert!(approx_eq(vals[1], 0.0, 1e-6));
    assert!(approx_eq(vals[2], 1.0, 1e-6));
    assert!(approx_eq(vals[3], -1.0, 1e-6));
    assert!(approx_eq(vals[4], 0.0, 1e-6));
    assert!(approx_eq(vals[5], 1.0, 1e-6));
}

#[test]
fn test_broadcast_div_after_max_keepdim() {
    // x / max(x, keepdim=True) — normalize to [0, 1] range per row
    let x = t2d(&[2.0, 4.0, 8.0, 3.0, 6.0, 9.0], 2, 3);
    let m = x.max_keepdim(1).unwrap(); // [2, 1]: [8, 9]
    let normed = x.broadcast_div(&m).unwrap();
    assert_eq!(normed.dims(), &[2, 3]);
    let vals = normed.to_flat_vec::<f32>().unwrap();
    assert!(approx_eq(vals[0], 0.25, 1e-6)); // 2/8
    assert!(approx_eq(vals[1], 0.5, 1e-6)); // 4/8
    assert!(approx_eq(vals[2], 1.0, 1e-6)); // 8/8
    assert!(approx_eq(vals[3], 1.0 / 3.0, 1e-5)); // 3/9
    assert!(approx_eq(vals[4], 2.0 / 3.0, 1e-5)); // 6/9
    assert!(approx_eq(vals[5], 1.0, 1e-6)); // 9/9
}

#[test]
fn test_broadcast_after_sum_keepdim_3d() {
    // 3D: sum along middle axis with keepdim, then broadcast-add back
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let x = tnd(&data, &[2, 2, 3]);
    let s = x.sum_keepdim(1).unwrap(); // [2, 1, 3]
    assert_eq!(s.dims(), &[2, 1, 3]);
    let result = x.broadcast_add(&s).unwrap(); // [2, 2, 3]
    assert_eq!(result.dims(), &[2, 2, 3]);
    // First batch: x + sum_keepdim
    // sum along axis=1: [[5,7,9]] (from [1+4, 2+5, 3+6])
    // [1+5, 2+7, 3+9, 4+5, 5+7, 6+9] = [6, 9, 12, 9, 12, 15]
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals[0], 6.0);
    assert_eq!(vals[1], 9.0);
    assert_eq!(vals[2], 12.0);
    assert_eq!(vals[3], 9.0);
    assert_eq!(vals[4], 12.0);
    assert_eq!(vals[5], 15.0);
}

// =============================================================================
// var_keepdim
// =============================================================================

#[test]
fn test_var_keepdim_2d() {
    // var([1,2,3,4,5]) = mean((x - 3)^2) = mean([4,1,0,1,4]) = 2.0
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0], 1, 5);
    let v = a.var_keepdim(1).unwrap();
    assert_eq!(v.dims(), &[1, 1]);
    assert!(approx_eq(v.to_scalar::<f32>().unwrap(), 2.0, 1e-4));
}

#[test]
fn test_var_2d_axis0() {
    // [[1,2],[5,6]] -> mean along axis=0 = [3,4]
    // var = mean([(1-3)^2,(5-3)^2], [(2-4)^2,(6-4)^2]) = mean([4,4], [4,4]) = [4, 4]
    let a = t2d(&[1.0, 2.0, 5.0, 6.0], 2, 2);
    let v = a.var(0).unwrap();
    assert_eq!(v.dims(), &[2]);
    let vals = v.to_vec1::<f32>().unwrap();
    assert!(approx_eq(vals[0], 4.0, 1e-4));
    assert!(approx_eq(vals[1], 4.0, 1e-4));
}

// =============================================================================
// Out-of-range dim errors
// =============================================================================

#[test]
fn test_sum_out_of_range_dim_errors() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    assert!(a.sum(2).is_err());
}

#[test]
fn test_argmax_out_of_range_dim_errors() {
    let a = t1d(&[1.0, 2.0]);
    assert!(a.argmax(1).is_err());
}

#[test]
fn test_argmax_nan_input_errors() {
    let a = DynTensor::from_vec(vec![1.0, f32::NAN, 3.0], &[3], &cpu()).unwrap();
    assert!(a.argmax(0).is_err());
}

#[test]
fn test_argmin_nan_input_errors() {
    let a = DynTensor::from_vec(vec![f32::NAN, 2.0, 3.0], &[3], &cpu()).unwrap();
    assert!(a.argmin(0).is_err());
}

// =============================================================================
// Mixed signs in 2D reduction
// =============================================================================

#[test]
fn test_sum_mixed_signs_2d_axis0() {
    // [[-1, 2], [3, -4]] -> sum(axis=0) -> [2, -2]
    let a = t2d(&[-1.0, 2.0, 3.0, -4.0], 2, 2);
    let s = a.sum(0).unwrap();
    assert_eq!(s.to_vec1::<f32>().unwrap(), vec![2.0, -2.0]);
}

#[test]
fn test_mean_mixed_signs_2d_axis1() {
    // [[-2, 4], [6, -8]] -> mean(axis=1) -> [1, -1]
    let a = t2d(&[-2.0, 4.0, 6.0, -8.0], 2, 2);
    let m = a.mean(1).unwrap();
    let vals = m.to_vec1::<f32>().unwrap();
    assert!(approx_eq(vals[0], 1.0, 1e-6));
    assert!(approx_eq(vals[1], -1.0, 1e-6));
}
