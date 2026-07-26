// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `tile` and `pad` operations.

use crate::dyn_tensor::DynTensor;

// --- tile tests ---

#[test]
fn test_tile_1d() {
    let x = DynTensor::from_vec(vec![1.0, 2.0], &[2], &crate::Device::Cpu).unwrap();
    let result = x.tile([3]).unwrap();
    assert_eq!(result.dims(), &[6]);
    let vals = result.to_vec1::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
}

#[test]
fn test_tile_2d() {
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &crate::Device::Cpu).unwrap();
    let result = x.tile([1, 2]).unwrap();
    assert_eq!(result.dims(), &[2, 4]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
}

#[test]
fn test_tile_is_repeat_alias() {
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &crate::Device::Cpu).unwrap();
    let tile_result = x.tile([2]).unwrap().to_vec1::<f32>().unwrap();
    let repeat_result = x.repeat([2]).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(tile_result, repeat_result);
}

// --- pad tests ---

#[test]
fn test_pad_1d_symmetric() {
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &crate::Device::Cpu).unwrap();
    // pad left=1, right=2
    let result = x.pad(&[1, 2], 0.0).unwrap();
    assert_eq!(result.dims(), &[6]);
    let vals = result.to_vec1::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
}

#[test]
fn test_pad_2d_last_dim_only() {
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], &crate::Device::Cpu).unwrap();
    // pad last dim: left=1, right=0
    let result = x.pad(&[1, 0], 0.0).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 1.0, 2.0, 0.0, 3.0, 4.0]);
}

#[test]
fn test_pad_custom_value() {
    let x = DynTensor::from_vec(vec![1.0, 2.0], &[2], &crate::Device::Cpu).unwrap();
    let result = x.pad(&[1, 1], -1.0).unwrap();
    let vals = result.to_vec1::<f32>().unwrap();
    assert_eq!(vals, vec![-1.0, 1.0, 2.0, -1.0]);
}

#[test]
fn test_pad_no_op() {
    let x = DynTensor::from_vec(vec![1.0, 2.0], &[2], &crate::Device::Cpu).unwrap();
    let result = x.pad(&[0, 0], 0.0).unwrap();
    let vals = result.to_vec1::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0]);
}

#[test]
fn test_pad_odd_length_errors() {
    let x = DynTensor::from_vec(vec![1.0], &[1], &crate::Device::Cpu).unwrap();
    let result = x.pad(&[1, 2, 3], 0.0);
    assert!(result.is_err());
}
