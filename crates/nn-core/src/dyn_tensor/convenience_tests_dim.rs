#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Negative-dimension (D::Minus1/Minus2) tests for shape, selection,
//! reduction, and softmax ops.
//!
//! Extracted from `convenience_tests.rs` for file-size compliance.

use crate::dyn_tensor::test_helpers::cpu;
use crate::{DType, DynTensor, D};

// =============================================================================
// D::Minus1/Minus2 tests for shape ops (W1-276 impl Dim batch conversion)
// =============================================================================

#[test]
fn test_d_minus1_cat() {
    // cat along D::Minus1 on rank-2 tensors => cat along dim 1
    let a = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let b = DynTensor::new(&[5.0, 6.0], &[2, 1], &cpu()).unwrap();
    let result = DynTensor::cat(&[&a, &b], D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
    let data = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
}

#[test]
fn test_d_minus2_cat() {
    // cat along D::Minus2 on rank-2 tensors => cat along dim 0
    let a = DynTensor::new(&[1.0, 2.0], &[1, 2], &cpu()).unwrap();
    let b = DynTensor::new(&[3.0, 4.0], &[1, 2], &cpu()).unwrap();
    let result = DynTensor::cat(&[&a, &b], D::Minus2).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
}

#[test]
fn test_d_minus1_stack() {
    // stack along D::Minus1 on rank-1 tensors => stack along dim 0 (rank+1=2, Minus1=>1)
    let a = DynTensor::new(&[1.0, 2.0], &[2], &cpu()).unwrap();
    let b = DynTensor::new(&[3.0, 4.0], &[2], &cpu()).unwrap();
    let result = DynTensor::stack(&[&a, &b], D::Minus1).unwrap();
    // D::Minus1 on new_rank=2 => dim 1
    assert_eq!(result.dims(), &[2, 2]);
    let data = result.to_flat_vec::<f32>().unwrap();
    // stack along dim 1: [[1,3],[2,4]]
    assert_eq!(data, vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn test_d_minus1_unsqueeze() {
    // unsqueeze(D::Minus1) on rank-2 [2,3] => rank+1=3, Minus1=>2 => [2,3,1]
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let result = t.unsqueeze(D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2, 3, 1]);
}

#[test]
fn test_d_minus2_unsqueeze() {
    // unsqueeze(D::Minus2) on rank-2 [2,3] => rank+1=3, Minus2=>1 => [2,1,3]
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let result = t.unsqueeze(D::Minus2).unwrap();
    assert_eq!(result.dims(), &[2, 1, 3]);
}

#[test]
fn test_d_minus1_squeeze() {
    // squeeze(D::Minus1) on [2,3,1] => removes dim 2 => [2,3]
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1], &cpu()).unwrap();
    let result = t.squeeze(D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
}

#[test]
fn test_d_minus2_squeeze() {
    // squeeze(D::Minus2) on [2,1,3] => removes dim 1 => [2,3]
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3], &cpu()).unwrap();
    let result = t.squeeze(D::Minus2).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
}

#[test]
fn test_d_minus1_transpose() {
    // transpose(D::Minus2, D::Minus1) on [2,3] => transpose(0, 1) => [3,2]
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let result = t.transpose(D::Minus2, D::Minus1).unwrap();
    assert_eq!(result.dims(), &[3, 2]);
    let data = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_d_minus1_transpose_3d() {
    // transpose(0, D::Minus1) on [2,3,4] => transpose(0, 2) => [4,3,2]
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    let result = t.transpose(0, D::Minus1).unwrap();
    assert_eq!(result.dims(), &[4, 3, 2]);
}

#[test]
fn test_d_minus1_chunk() {
    // chunk(2, D::Minus1) on [2,4] => chunk along dim 1 => two [2,2] tensors
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4], &cpu()).unwrap();
    let chunks = t.chunk(2, D::Minus1).unwrap();
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].dims(), &[2, 2]);
    assert_eq!(chunks[1].dims(), &[2, 2]);
    assert_eq!(
        chunks[0].to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 5.0, 6.0]
    );
    assert_eq!(
        chunks[1].to_flat_vec::<f32>().unwrap(),
        vec![3.0, 4.0, 7.0, 8.0]
    );
}

#[test]
fn test_d_minus1_flip() {
    // flip(D::Minus1) on [2,3] => flip along dim 1
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let result = t.flip(D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
    let data = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
}

// =============================================================================
// D::Minus1 tests for selection ops (W1-276 impl Dim batch conversion)
// =============================================================================

#[test]
fn test_d_minus1_index_select() {
    // index_select with D::Minus1 on [2,3] => select along dim 1
    let t = DynTensor::new(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3], &cpu()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![2, 0], &[2], &cpu()).unwrap();
    let result = t.index_select(&ids, D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    let data = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![30.0, 10.0, 60.0, 40.0]);
}

#[test]
fn test_d_minus1_gather() {
    // gather with D::Minus1 on [2,3] => gather along dim 1
    let t = DynTensor::new(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3], &cpu()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![1, 0, 2, 1], &[2, 2], &cpu()).unwrap();
    let result = t.gather(&ids, D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    let data = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![20.0, 10.0, 60.0, 50.0]);
}

#[test]
fn test_d_minus1_scatter_add() {
    // scatter_add with D::Minus1 on [2,3] => scatter along dim 1
    let base = DynTensor::zeros(&[2, 3], DType::F32, &cpu()).unwrap();
    let index = DynTensor::from_vec_u32(vec![1, 0, 2, 1], &[2, 2], &cpu()).unwrap();
    let src = DynTensor::new(&[10.0, 20.0, 30.0, 40.0], &[2, 2], &cpu()).unwrap();
    let result = base.scatter_add(D::Minus1, &index, &src).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
    let data = result.to_flat_vec::<f32>().unwrap();
    // row 0: [0,0,0] + 20@0 + 10@1 => [20, 10, 0]
    // row 1: [0,0,0] + 40@1 + 30@2 => [0, 40, 30]
    assert_eq!(data, vec![20.0, 10.0, 0.0, 0.0, 40.0, 30.0]);
}

// =============================================================================
// D::Minus1 tests for topk + pad_with_zeros (W1-276 impl Dim batch conversion)
// =============================================================================

#[test]
fn test_d_minus1_topk() {
    // topk with D::Minus1 on [2,4] => topk along dim 1, k=2
    let t = DynTensor::new(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], &[2, 4], &cpu()).unwrap();
    let (values, indices) = t.topk(D::Minus1, 2).unwrap();
    assert_eq!(values.dims(), &[2, 2]);
    assert_eq!(indices.dims(), &[2, 2]);
    let vals = values.to_flat_vec::<f32>().unwrap();
    // row 0 top-2: 4.0, 3.0; row 1 top-2: 9.0, 6.0
    assert_eq!(vals[0], 4.0);
    assert_eq!(vals[1], 3.0);
    assert_eq!(vals[2], 9.0);
    assert_eq!(vals[3], 6.0);
}

#[test]
fn test_d_minus1_pad_with_zeros() {
    // pad_with_zeros with D::Minus1 on [2,3] => pad along dim 1
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let result = t.pad_with_zeros(D::Minus1, 1, 2).unwrap();
    assert_eq!(result.dims(), &[2, 6]);
    let data = result.to_flat_vec::<f32>().unwrap();
    // row 0: [0, 1, 2, 3, 0, 0]; row 1: [0, 4, 5, 6, 0, 0]
    assert_eq!(
        data,
        vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0]
    );
}

// =============================================================================
// i32 negative indexing tests (#2471 Finding 1: impl Dim for i32)
// =============================================================================

#[test]
fn test_i32_positive_dim_narrow() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let result = t.narrow(1_i32, 0, 2).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
}

#[test]
fn test_i32_minus1_narrow() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    // -1 on rank 2 = dim 1
    let result = t.narrow(-1_i32, 0, 2).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
}

#[test]
fn test_i32_minus2_narrow() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    // -2 on rank 2 = dim 0
    let result = t.narrow(-2_i32, 0, 1).unwrap();
    assert_eq!(result.dims(), &[1, 3]);
}

#[test]
fn test_i32_minus3_on_4d() {
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let t = DynTensor::new(&data, &[2, 3, 2, 2], &cpu()).unwrap();
    // -3 on rank 4 = dim 1
    let result = t.narrow(-3_i32, 0, 1).unwrap();
    assert_eq!(result.dims(), &[2, 1, 2, 2]);
}

#[test]
fn test_i32_negative_out_of_range() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    // -2 on rank 1 tensor should fail
    let err = t.narrow(-2_i32, 0, 1).unwrap_err();
    assert!(
        format!("{err:?}").contains("DimensionOutOfRange"),
        "expected DimensionOutOfRange, got: {err:?}"
    );
}

#[test]
fn test_i32_positive_out_of_range() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let err = t.narrow(1_i32, 0, 1).unwrap_err();
    assert!(
        format!("{err:?}").contains("DimensionOutOfRange"),
        "expected DimensionOutOfRange, got: {err:?}"
    );
}

#[test]
fn test_i32_zero_dim() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let result = t.narrow(0_i32, 0, 1).unwrap();
    assert_eq!(result.dims(), &[1, 3]);
}

// Reduction, softmax, cumsum, argmin, flatten tests extracted to
// convenience_tests_dim_reduction.rs
