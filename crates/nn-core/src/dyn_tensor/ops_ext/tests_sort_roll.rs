// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `sort` and `roll` DynTensor operations.

use crate::dyn_tensor::test_helpers::{cpu, tnd};
use crate::dyn_tensor::DynTensor;
use crate::DType;

// -- sort tests ---------------------------------------------------------------

#[test]
fn test_sort_1d_ascending() {
    let x = tnd(&[3.0, 1.0, 4.0, 1.5, 2.0], &[5]);
    let (vals, idxs) = x.sort(0, false).unwrap();
    assert_eq!(vals.dims(), &[5]);
    assert_eq!(idxs.dims(), &[5]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![1.0, 1.5, 2.0, 3.0, 4.0]);
    let i: Vec<u32> = idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(i, vec![1, 3, 4, 0, 2]);
}

#[test]
fn test_sort_1d_descending() {
    let x = tnd(&[3.0, 1.0, 4.0, 1.5, 2.0], &[5]);
    let (vals, idxs) = x.sort(0, true).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![4.0, 3.0, 2.0, 1.5, 1.0]);
    let i: Vec<u32> = idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(i, vec![2, 0, 4, 3, 1]);
}

#[test]
fn test_sort_2d_along_last_dim() {
    let x = tnd(&[3.0, 1.0, 2.0, 6.0, 4.0, 5.0], &[2, 3]);
    let (vals, idxs) = x.sort(1, false).unwrap();
    assert_eq!(vals.dims(), &[2, 3]);
    let v = vals.to_flat_vec::<f32>().unwrap();
    // Row 0: [3,1,2] sorted -> [1,2,3]
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 2.0);
    assert_eq!(v[2], 3.0);
    // Row 1: [6,4,5] sorted -> [4,5,6]
    assert_eq!(v[3], 4.0);
    assert_eq!(v[4], 5.0);
    assert_eq!(v[5], 6.0);
    let i: Vec<u32> = idxs.to_flat_vec::<u32>().unwrap();
    assert_eq!(i, vec![1, 2, 0, 1, 2, 0]);
}

#[test]
fn test_sort_2d_along_first_dim() {
    let x = tnd(&[3.0, 1.0, 2.0, 6.0, 4.0, 5.0], &[2, 3]);
    let (vals, _idxs) = x.sort(0, false).unwrap();
    let v = vals.to_flat_vec::<f32>().unwrap();
    // Col 0: [3,6] -> [3,6], Col 1: [1,4] -> [1,4], Col 2: [2,5] -> [2,5]
    assert_eq!(v, vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0]);
}

#[test]
fn test_sort_single_element() {
    let x = tnd(&[42.0], &[1]);
    let (vals, idxs) = x.sort(0, false).unwrap();
    assert_eq!(vals.to_flat_vec::<f32>().unwrap(), vec![42.0]);
    assert_eq!(idxs.to_flat_vec::<u32>().unwrap(), vec![0]);
}

#[test]
fn test_sort_indices_dtype_u32() {
    let x = tnd(&[2.0, 1.0, 3.0], &[3]);
    let (_vals, idxs) = x.sort(0, false).unwrap();
    assert_eq!(idxs.dtype(), DType::U32);
}

#[test]
fn test_sort_nan_rejected() {
    let x = tnd(&[1.0, f32::NAN, 3.0], &[3]);
    assert!(x.sort(0, false).is_err());
}

#[test]
fn test_sort_rank0_rejected() {
    let x = DynTensor::full(&[], 1.0, DType::F32, &cpu()).unwrap();
    assert!(x.sort(0, false).is_err());
}

// -- roll tests ---------------------------------------------------------------

#[test]
fn test_roll_1d_positive() {
    let x = tnd(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let result = x.roll(&[1], &[0]).unwrap();
    assert_eq!(result.dims(), &[4]);
    let v = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![4.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_roll_1d_negative() {
    let x = tnd(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let result = x.roll(&[-1], &[0]).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![2.0, 3.0, 4.0, 1.0]);
}

#[test]
fn test_roll_2d_along_rows() {
    let x = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let result = x.roll(&[1], &[0]).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    // Rows shifted: row1 becomes row0
    assert_eq!(v, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_roll_2d_along_cols() {
    let x = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let result = x.roll(&[1], &[1]).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    // Cols shifted: last col wraps to first
    assert_eq!(v, vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0]);
}

#[test]
fn test_roll_multiple_dims() {
    let x = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let result = x.roll(&[1, 1], &[0, 1]).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    // First roll dim=0 by 1: [[4,5,6],[1,2,3]]
    // Then roll dim=1 by 1: [[6,4,5],[3,1,2]]
    assert_eq!(v, vec![6.0, 4.0, 5.0, 3.0, 1.0, 2.0]);
}

#[test]
fn test_roll_zero_shift() {
    let x = tnd(&[1.0, 2.0, 3.0], &[3]);
    let result = x.roll(&[0], &[0]).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_roll_full_cycle() {
    let x = tnd(&[1.0, 2.0, 3.0], &[3]);
    let result = x.roll(&[3], &[0]).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_roll_preserves_dtype() {
    let x = tnd(&[1.0, 2.0, 3.0], &[3]);
    let result = x.roll(&[1], &[0]).unwrap();
    assert_eq!(result.dtype(), DType::F32);
}

#[test]
fn test_roll_shift_dims_length_mismatch() {
    let x = tnd(&[1.0, 2.0, 3.0], &[3]);
    assert!(x.roll(&[1, 2], &[0]).is_err());
}

#[test]
fn test_roll_dim_out_of_range() {
    let x = tnd(&[1.0, 2.0, 3.0], &[3]);
    assert!(x.roll(&[1], &[1]).is_err());
}

#[test]
fn test_roll_single_element() {
    let x = tnd(&[42.0], &[1]);
    let result = x.roll(&[5], &[0]).unwrap();
    assert_eq!(result.to_flat_vec::<f32>().unwrap(), vec![42.0]);
}
