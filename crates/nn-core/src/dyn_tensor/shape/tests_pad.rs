// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for `DynTensor::pad()`: multi-dim, 3D, error cases.
//! Basic pad tests are in `tests_tile_pad.rs`.

use crate::dyn_tensor::test_helpers::{cpu, t1d, tnd};
use crate::DynTensor;

// -- Asymmetric padding -------------------------------------------------------

#[test]
fn test_pad_1d_asymmetric() {
    let x = t1d(&[10.0, 20.0]);
    let y = x.pad(&[1, 3], 0.0).unwrap();
    assert_eq!(y.dims(), &[6]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 10.0, 20.0, 0.0, 0.0, 0.0]);
}

// -- 2D both-dim padding ------------------------------------------------------

#[test]
fn test_pad_2d_both_dims() {
    // Pad both dims: [left_last, right_last, left_2nd_last, right_2nd_last]
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let y = x.pad(&[1, 1, 1, 1], 0.0).unwrap();
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

// -- 3D padding (typical for conv1d: [B, C, T]) -------------------------------

#[test]
fn test_pad_3d_time_dim() {
    // [B, C, T] = [1, 1, 3], pad time dimension with 2 on each side
    let x = tnd(&[1.0, 2.0, 3.0], &[1, 1, 3]);
    let y = x.pad(&[2, 2], 0.0).unwrap();
    assert_eq!(y.dims(), &[1, 1, 7]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
}

#[test]
fn test_pad_3d_two_dims() {
    // Pad both channel and time: [left_T, right_T, left_C, right_C]
    let x = tnd(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let y = x.pad(&[1, 0, 0, 1], 9.0).unwrap();
    // Shape: [1, 3, 3] (channel dim: 2+0+1=3, time dim: 2+1+0=3)
    assert_eq!(y.dims(), &[1, 3, 3]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Row 0 (original ch0): [9.0, 1.0, 2.0]
    // Row 1 (original ch1): [9.0, 3.0, 4.0]
    // Row 2 (pad ch):       [9.0, 9.0, 9.0]
    assert_eq!(vals, vec![9.0, 1.0, 2.0, 9.0, 3.0, 4.0, 9.0, 9.0, 9.0]);
}

// -- Error cases --------------------------------------------------------------

#[test]
fn test_pad_exceeds_rank_error() {
    let x = t1d(&[1.0]); // rank 1
                         // 4 padding values = 2 dim pairs, but rank is 1
    let result = x.pad(&[1, 1, 1, 1], 0.0);
    assert!(result.is_err(), "padding more dims than rank should fail");
}

#[test]
fn test_pad_preserves_dtype() {
    let x = t1d(&[1.0, 2.0]);
    let y = x.pad(&[1, 1], 0.0).unwrap();
    assert_eq!(y.dtype(), x.dtype());
}
