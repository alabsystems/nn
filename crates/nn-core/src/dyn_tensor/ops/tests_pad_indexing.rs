// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for tuple-based padding, slice_assign, and flip_dims operations.

use crate::dyn_tensor::test_helpers::{cpu, t1d, t2d, tnd};
use crate::DynTensor;

// -- pad_with_value -----------------------------------------------------------

#[test]
fn test_pad_with_value_1d_zeros() {
    let x = t1d(&[1.0, 2.0, 3.0]);
    let y = x.pad_with_value(&[(1, 2)], 0.0).unwrap();
    assert_eq!(y.dims(), &[6]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
}

#[test]
fn test_pad_with_value_1d_nonzero() {
    let x = t1d(&[1.0, 2.0]);
    let y = x.pad_with_value(&[(2, 1)], 9.0).unwrap();
    assert_eq!(y.dims(), &[5]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![9.0, 9.0, 1.0, 2.0, 9.0]);
}

#[test]
fn test_pad_with_value_2d() {
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let y = x.pad_with_value(&[(1, 0), (0, 1)], 5.0).unwrap();
    assert_eq!(y.dims(), &[3, 3]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    #[rustfmt::skip]
    let expected = vec![
        5.0, 5.0, 5.0,
        1.0, 2.0, 5.0,
        3.0, 4.0, 5.0,
    ];
    assert_eq!(vals, expected);
}

#[test]
fn test_pad_with_value_3d() {
    // [1, 2, 3] tensor, pad only last dim
    let x = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
    let y = x.pad_with_value(&[(0, 0), (0, 0), (1, 1)], 0.0).unwrap();
    assert_eq!(y.dims(), &[1, 2, 5]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0]);
}

#[test]
fn test_pad_with_value_noop() {
    let x = t1d(&[1.0, 2.0]);
    let y = x.pad_with_value(&[(0, 0)], 0.0).unwrap();
    assert_eq!(y.dims(), x.dims());
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0]);
}

#[test]
fn test_pad_with_value_wrong_pads_len() {
    let x = t1d(&[1.0, 2.0]); // rank 1
    let result = x.pad_with_value(&[(1, 1), (2, 2)], 0.0); // 2 pairs for rank 1
    assert!(result.is_err());
}

// -- pad_zeros_nd -------------------------------------------------------------

#[test]
fn test_pad_zeros_nd_1d() {
    let x = t1d(&[10.0, 20.0]);
    let y = x.pad_zeros_nd(&[(1, 1)]).unwrap();
    assert_eq!(y.dims(), &[4]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 10.0, 20.0, 0.0]);
}

#[test]
fn test_pad_zeros_nd_2d_both_dims() {
    let x = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let y = x.pad_zeros_nd(&[(1, 1), (1, 1)]).unwrap();
    assert_eq!(y.dims(), &[4, 4]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    #[rustfmt::skip]
    let expected = vec![
        0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 2.0, 0.0,
        0.0, 3.0, 4.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(vals, expected);
}

#[test]
fn test_pad_zeros_nd_preserves_dtype() {
    let x = t1d(&[1.0, 2.0]);
    let y = x.pad_zeros_nd(&[(1, 1)]).unwrap();
    assert_eq!(y.dtype(), x.dtype());
}

// -- pad_reflect --------------------------------------------------------------

#[test]
fn test_pad_reflect_1d() {
    // [a, b, c, d, e] with (2, 1) → [c, b, a, b, c, d, e, d]
    let x = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = x.pad_reflect(&[(2, 1)]).unwrap();
    assert_eq!(y.dims(), &[8]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0]);
}

#[test]
fn test_pad_reflect_1d_left_only() {
    let x = t1d(&[10.0, 20.0, 30.0]);
    let y = x.pad_reflect(&[(2, 0)]).unwrap();
    assert_eq!(y.dims(), &[5]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![30.0, 20.0, 10.0, 20.0, 30.0]);
}

#[test]
fn test_pad_reflect_1d_right_only() {
    let x = t1d(&[10.0, 20.0, 30.0]);
    let y = x.pad_reflect(&[(0, 2)]).unwrap();
    assert_eq!(y.dims(), &[5]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![10.0, 20.0, 30.0, 20.0, 10.0]);
}

#[test]
fn test_pad_reflect_2d() {
    // [[1, 2, 3], [4, 5, 6]] shape [2, 3]
    // pad dim0 by (1, 0), dim1 by (1, 1)
    let x = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let y = x.pad_reflect(&[(1, 0), (1, 1)]).unwrap();
    // dim0: reflect before → row[1] = [4,5,6], so [4,5,6, 1,2,3, 4,5,6] → shape [3, 3]
    // dim1: reflect before → col[1], after → col[N-2]
    assert_eq!(y.dims(), &[3, 5]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Row 0 (reflected row 1): [5, 4, 5, 6, 5]
    // Row 1 (original row 0):  [2, 1, 2, 3, 2]
    // Row 2 (original row 1):  [5, 4, 5, 6, 5]
    let expected = vec![
        5.0, 4.0, 5.0, 6.0, 5.0, 2.0, 1.0, 2.0, 3.0, 2.0, 5.0, 4.0, 5.0, 6.0, 5.0,
    ];
    assert_eq!(vals, expected);
}

#[test]
fn test_pad_reflect_noop() {
    let x = t1d(&[1.0, 2.0, 3.0]);
    let y = x.pad_reflect(&[(0, 0)]).unwrap();
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_pad_reflect_exceeds_dim_size_error() {
    let x = t1d(&[1.0, 2.0, 3.0]); // size 3
                                   // Padding 3 >= dim size 3
    let result = x.pad_reflect(&[(3, 0)]);
    assert!(result.is_err());
}

#[test]
fn test_pad_reflect_wrong_pads_len_error() {
    let x = t1d(&[1.0, 2.0]); // rank 1
    let result = x.pad_reflect(&[(1, 1), (1, 1)]); // 2 pairs for rank 1
    assert!(result.is_err());
}

// -- slice_assign -------------------------------------------------------------

#[test]
fn test_slice_assign_1d() {
    let x = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let src = t1d(&[90.0, 91.0]);
    let y = x.slice_assign(0, 1, &src).unwrap();
    assert_eq!(y.dims(), &[5]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 90.0, 91.0, 4.0, 5.0]);
}

#[test]
fn test_slice_assign_2d() {
    let x = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    // Replace row 1 with [10, 20, 30]
    let src = DynTensor::from_vec(vec![10.0, 20.0, 30.0], &[1, 3], &cpu()).unwrap();
    let y = x.slice_assign(0, 1, &src).unwrap();
    assert_eq!(y.dims(), &[2, 3]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
}

#[test]
fn test_slice_assign_does_not_modify_original() {
    let x = t1d(&[1.0, 2.0, 3.0]);
    let src = t1d(&[99.0]);
    let _y = x.slice_assign(0, 1, &src).unwrap();
    // Original unchanged
    let orig_vals = x.to_flat_vec::<f32>().unwrap();
    assert_eq!(orig_vals, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_slice_assign_out_of_bounds_error() {
    let x = t1d(&[1.0, 2.0, 3.0]);
    let src = t1d(&[99.0, 98.0]);
    let result = x.slice_assign(0, 2, &src); // 2 + 2 = 4 > 3
    assert!(result.is_err());
}

#[test]
fn test_slice_assign_dim_out_of_range_error() {
    let x = t1d(&[1.0, 2.0]);
    let src = t1d(&[9.0]);
    let result = x.slice_assign(1, 0, &src); // dim 1 invalid for rank 1
    assert!(result.is_err());
}

#[test]
fn test_slice_assign_rank_mismatch_error() {
    let x = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let src = t1d(&[9.0, 8.0]); // rank 1 vs rank 2
    let result = x.slice_assign(0, 0, &src);
    assert!(result.is_err());
}

#[test]
fn test_slice_assign_shape_mismatch_error() {
    let x = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    // src shape [1, 2] -- cols don't match (need 3)
    let src = DynTensor::from_vec(vec![9.0, 8.0], &[1, 2], &cpu()).unwrap();
    let result = x.slice_assign(0, 0, &src);
    assert!(result.is_err());
}

// -- flip_dims ----------------------------------------------------------------

#[test]
fn test_flip_dims_single() {
    let x = t1d(&[1.0, 2.0, 3.0]);
    let y = x.flip_dims(&[0]).unwrap();
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![3.0, 2.0, 1.0]);
}

#[test]
fn test_flip_dims_two_dims() {
    // [[1, 2, 3], [4, 5, 6]] shape [2, 3]
    let x = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let y = x.flip_dims(&[0, 1]).unwrap();
    assert_eq!(y.dims(), &[2, 3]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // flip dim 0: [[4,5,6],[1,2,3]], flip dim 1: [[6,5,4],[3,2,1]]
    assert_eq!(vals, vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn test_flip_dims_empty() {
    let x = t1d(&[1.0, 2.0, 3.0]);
    let y = x.flip_dims(&[]).unwrap();
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_flip_dims_preserves_shape() {
    let x = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let y = x.flip_dims(&[0, 2]).unwrap();
    assert_eq!(y.dims(), &[2, 2, 2]);
}

#[test]
fn test_flip_dims_dim_out_of_range_error() {
    let x = t1d(&[1.0, 2.0]);
    let result = x.flip_dims(&[0, 1]); // dim 1 invalid for rank 1
    assert!(result.is_err());
}

#[test]
fn test_flip_dims_same_as_sequential_flip() {
    // Verify that flip_dims([0, 1]) == flip(0).flip(1)
    let x = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let y1 = x.flip_dims(&[0, 1]).unwrap();
    let y2 = x.flip(0).unwrap().flip(1).unwrap();
    assert_eq!(
        y1.to_flat_vec::<f32>().unwrap(),
        y2.to_flat_vec::<f32>().unwrap()
    );
}
