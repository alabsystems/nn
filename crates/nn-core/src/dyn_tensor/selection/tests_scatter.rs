// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `scatter` DynTensor operation.

use crate::dyn_tensor::test_helpers::{cpu, tnd};
use crate::dyn_tensor::DynTensor;
use crate::DType;

#[test]
fn test_scatter_1d_basic() {
    // Scatter src values into dst at index positions.
    let dst = tnd(&[0.0, 0.0, 0.0, 0.0, 0.0], &[5]);
    let src = tnd(&[10.0, 20.0, 30.0], &[3]);
    let index = DynTensor::from_vec_u32(vec![1, 3, 0], &[3], &cpu()).unwrap();
    let result = dst.scatter(0, &index, &src).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![30.0, 10.0, 0.0, 20.0, 0.0]);
}

#[test]
fn test_scatter_2d_dim0() {
    // 3x2 dst, scatter 2x2 src into rows specified by index along dim=0
    let dst = tnd(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[3, 2]);
    let src = tnd(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
    let index = DynTensor::from_vec_u32(vec![2, 0, 1, 1], &[2, 2], &cpu()).unwrap();
    let result = dst.scatter(0, &index, &src).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    // dst[2][0] = 10.0, dst[0][1] = 20.0, dst[1][0] = 30.0, dst[1][1] = 40.0
    assert_eq!(v, vec![0.0, 20.0, 30.0, 40.0, 10.0, 0.0]);
}

#[test]
fn test_scatter_2d_dim1() {
    // 2x3 dst, scatter 2x2 src along dim=1
    let dst = tnd(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[2, 3]);
    let src = tnd(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
    let index = DynTensor::from_vec_u32(vec![2, 0, 1, 2], &[2, 2], &cpu()).unwrap();
    let result = dst.scatter(1, &index, &src).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    // Row 0: dst[0][2]=10, dst[0][0]=20 => [20, 0, 10]
    // Row 1: dst[1][1]=30, dst[1][2]=40 => [0, 30, 40]
    assert_eq!(v, vec![20.0, 0.0, 10.0, 0.0, 30.0, 40.0]);
}

#[test]
fn test_scatter_overwrites_not_adds() {
    // Verify scatter overwrites rather than accumulating.
    let dst = tnd(&[100.0, 100.0, 100.0], &[3]);
    let src = tnd(&[1.0], &[1]);
    let index = DynTensor::from_vec_u32(vec![1], &[1], &cpu()).unwrap();
    let result = dst.scatter(0, &index, &src).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    // dst[1] = 1.0 (overwrite, not 101.0)
    assert_eq!(v, vec![100.0, 1.0, 100.0]);
}

#[test]
fn test_scatter_preserves_dtype() {
    let dst = tnd(&[0.0, 0.0, 0.0], &[3]);
    let src = tnd(&[1.0], &[1]);
    let index = DynTensor::from_vec_u32(vec![0], &[1], &cpu()).unwrap();
    let result = dst.scatter(0, &index, &src).unwrap();
    assert_eq!(result.dtype(), DType::F32);
}

#[test]
fn test_scatter_index_oob() {
    let dst = tnd(&[0.0, 0.0, 0.0], &[3]);
    let src = tnd(&[1.0], &[1]);
    let index = DynTensor::from_vec_u32(vec![5], &[1], &cpu()).unwrap();
    assert!(dst.scatter(0, &index, &src).is_err());
}

#[test]
fn test_scatter_index_src_shape_mismatch() {
    let dst = tnd(&[0.0, 0.0, 0.0], &[3]);
    let src = tnd(&[1.0, 2.0], &[2]);
    let index = DynTensor::from_vec_u32(vec![0], &[1], &cpu()).unwrap();
    assert!(dst.scatter(0, &index, &src).is_err());
}

#[test]
fn test_scatter_duplicate_indices() {
    // When multiple src elements target the same index, last write wins.
    let dst = tnd(&[0.0, 0.0, 0.0], &[3]);
    let src = tnd(&[10.0, 20.0], &[2]);
    let index = DynTensor::from_vec_u32(vec![1, 1], &[2], &cpu()).unwrap();
    let result = dst.scatter(0, &index, &src).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    // Both write to index 1: last write (20.0) wins
    assert_eq!(v[1], 20.0);
}
