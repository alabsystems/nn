// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `index_put` operation.

use crate::dyn_tensor::DynTensor;
use crate::DType;

#[test]
fn test_index_put_1d_basic() {
    // dst: [0, 0, 0, 0, 0]
    let dst = DynTensor::zeros(&[5], DType::F32, &crate::Device::Cpu).unwrap();
    // index: [1, 3] — write to positions 1 and 3
    let index = DynTensor::from_vec_u32(vec![1, 3], &[2], &crate::Device::Cpu).unwrap();
    // src: [10, 30]
    let src = DynTensor::from_vec(vec![10.0, 30.0], &[2], &crate::Device::Cpu).unwrap();

    let result = dst.index_put(0, &index, &src).unwrap();
    let vals = result.to_vec1::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 10.0, 0.0, 30.0, 0.0]);
}

#[test]
fn test_index_put_2d_dim0() {
    // dst: [[1, 2], [3, 4], [5, 6]]
    let dst = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[3, 2],
        &crate::Device::Cpu,
    )
    .unwrap();
    // index: [0, 2] — overwrite rows 0 and 2
    let index = DynTensor::from_vec_u32(vec![0, 2], &[2], &crate::Device::Cpu).unwrap();
    // src: [[10, 20], [50, 60]]
    let src =
        DynTensor::from_vec(vec![10.0, 20.0, 50.0, 60.0], &[2, 2], &crate::Device::Cpu).unwrap();

    let result = dst.index_put(0, &index, &src).unwrap();
    let vals = result.to_flat_vec::<f32>().unwrap();
    // row 0 replaced by [10, 20], row 1 unchanged [3, 4], row 2 replaced by [50, 60]
    assert_eq!(vals, vec![10.0, 20.0, 3.0, 4.0, 50.0, 60.0]);
}

#[test]
fn test_index_put_2d_dim1() {
    // dst: [[1, 2, 3], [4, 5, 6]]
    let dst = DynTensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        &crate::Device::Cpu,
    )
    .unwrap();
    // index: [0, 2] — overwrite columns 0 and 2
    let index = DynTensor::from_vec_u32(vec![0, 2], &[2], &crate::Device::Cpu).unwrap();
    // src: [[10, 30], [40, 60]]
    let src =
        DynTensor::from_vec(vec![10.0, 30.0, 40.0, 60.0], &[2, 2], &crate::Device::Cpu).unwrap();

    let result = dst.index_put(1, &index, &src).unwrap();
    let vals = result.to_flat_vec::<f32>().unwrap();
    // col 0 → [10, 40], col 1 unchanged [2, 5], col 2 → [30, 60]
    assert_eq!(vals, vec![10.0, 2.0, 30.0, 40.0, 5.0, 60.0]);
}

#[test]
fn test_index_put_duplicate_indices_last_write_wins() {
    // dst: [0, 0, 0]
    let dst = DynTensor::zeros(&[3], DType::F32, &crate::Device::Cpu).unwrap();
    // index: [1, 1] — both write to position 1
    let index = DynTensor::from_vec_u32(vec![1, 1], &[2], &crate::Device::Cpu).unwrap();
    // src: [10, 20] — second value (20) should win
    let src = DynTensor::from_vec(vec![10.0, 20.0], &[2], &crate::Device::Cpu).unwrap();

    let result = dst.index_put(0, &index, &src).unwrap();
    let vals = result.to_vec1::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 20.0, 0.0]);
}

#[test]
fn test_index_put_out_of_bounds() {
    let dst = DynTensor::zeros(&[3], DType::F32, &crate::Device::Cpu).unwrap();
    let index = DynTensor::from_vec_u32(vec![5], &[1], &crate::Device::Cpu).unwrap();
    let src = DynTensor::from_vec(vec![10.0], &[1], &crate::Device::Cpu).unwrap();

    let result = dst.index_put(0, &index, &src);
    assert!(result.is_err());
}

#[test]
fn test_index_put_rank_mismatch() {
    let dst = DynTensor::zeros(&[3, 2], DType::F32, &crate::Device::Cpu).unwrap();
    let index = DynTensor::from_vec_u32(vec![0], &[1], &crate::Device::Cpu).unwrap();
    // src has wrong rank (1-D instead of 2-D)
    let src = DynTensor::from_vec(vec![10.0], &[1], &crate::Device::Cpu).unwrap();

    let result = dst.index_put(0, &index, &src);
    assert!(result.is_err());
}

#[test]
fn test_index_put_i64_index() {
    let dst = DynTensor::zeros(&[5], DType::F32, &crate::Device::Cpu).unwrap();
    let index = DynTensor::from_vec_i64(vec![2, 4], &[2], &crate::Device::Cpu).unwrap();
    let src = DynTensor::from_vec(vec![20.0, 40.0], &[2], &crate::Device::Cpu).unwrap();

    let result = dst.index_put(0, &index, &src).unwrap();
    let vals = result.to_vec1::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 0.0, 20.0, 0.0, 40.0]);
}
