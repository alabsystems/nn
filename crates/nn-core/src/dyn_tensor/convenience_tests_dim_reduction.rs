#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Negative-dimension reduction, softmax, and argmin tests
//! (extracted from convenience_tests_dim.rs).
//!
//! Tests D::Minus1/Minus2 for sum, mean, max, min, var, softmax,
//! log_softmax, cumsum, argmin, and flatten.

use crate::dyn_tensor::test_helpers::cpu;
use crate::{DType, DynTensor, D};

// =============================================================================
// D::Minus1/Minus2 tests for reduction ops
// =============================================================================

#[test]
fn test_d_minus1_sum() {
    // sum(D::Minus1) on [2,3] => sum along dim 1 => [2]
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let result = t.sum(D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2]);
    let data = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![6.0, 15.0]);
}

#[test]
fn test_d_minus2_sum() {
    // sum(D::Minus2) on [2,3] => sum along dim 0 => [3]
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let result = t.sum(D::Minus2).unwrap();
    assert_eq!(result.dims(), &[3]);
    let data = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_d_minus1_sum_keepdim() {
    // sum_keepdim(D::Minus1) on [2,3] => sum along dim 1 keepdim => [2,1]
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let result = t.sum_keepdim(D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2, 1]);
    let data = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![6.0, 15.0]);
}

#[test]
fn test_d_minus1_mean() {
    // mean(D::Minus1) on [2,3] => mean along dim 1 => [2]
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let result = t.mean(D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2]);
    let data = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![2.0, 5.0]);
}

#[test]
fn test_d_minus2_mean() {
    // mean(D::Minus2) on [2,3] => mean along dim 0 => [3]
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let result = t.mean(D::Minus2).unwrap();
    assert_eq!(result.dims(), &[3]);
    let data = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![2.5, 3.5, 4.5]);
}

#[test]
fn test_d_minus1_mean_keepdim() {
    // mean_keepdim(D::Minus1) on [2,3] => [2,1]
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let result = t.mean_keepdim(D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2, 1]);
    let data = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![2.0, 5.0]);
}

#[test]
fn test_d_minus1_max() {
    // max(D::Minus1) on [2,3] => max along dim 1 => [2]
    let t = DynTensor::new(&[1.0, 3.0, 2.0, 6.0, 4.0, 5.0], &[2, 3], &cpu()).unwrap();
    let result = t.max(D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2]);
    let data = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![3.0, 6.0]);
}

#[test]
fn test_d_minus1_min() {
    // min(D::Minus1) on [2,3] => min along dim 1 => [2]
    let t = DynTensor::new(&[1.0, 3.0, 2.0, 6.0, 4.0, 5.0], &[2, 3], &cpu()).unwrap();
    let result = t.min(D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2]);
    let data = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![1.0, 4.0]);
}

#[test]
fn test_d_minus1_max_keepdim() {
    // max_keepdim(D::Minus1) on [2,3] => [2,1]
    let t = DynTensor::new(&[1.0, 3.0, 2.0, 6.0, 4.0, 5.0], &[2, 3], &cpu()).unwrap();
    let result = t.max_keepdim(D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2, 1]);
    let data = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![3.0, 6.0]);
}

#[test]
fn test_d_minus1_min_keepdim() {
    // min_keepdim(D::Minus1) on [2,3] => [2,1]
    let t = DynTensor::new(&[1.0, 3.0, 2.0, 6.0, 4.0, 5.0], &[2, 3], &cpu()).unwrap();
    let result = t.min_keepdim(D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2, 1]);
    let data = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![1.0, 4.0]);
}

#[test]
fn test_d_minus1_var_keepdim() {
    // var_keepdim(D::Minus1) on [2,3] => population variance along dim 1
    // row 0: [1,2,3], mean=2, var = ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = 2/3
    // row 1: [4,5,6], mean=5, var = ((4-5)^2 + (5-5)^2 + (6-5)^2)/3 = 2/3
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let result = t.var_keepdim(D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2, 1]);
    let data = result.to_flat_vec::<f32>().unwrap();
    let expected = 2.0_f32 / 3.0;
    assert!((data[0] - expected).abs() < 1e-6);
    assert!((data[1] - expected).abs() < 1e-6);
}

// =============================================================================
// D::Minus1/Minus2 tests for 3D reductions
// =============================================================================

#[test]
fn test_d_minus1_mean_3d() {
    // mean(D::Minus1) on [2,2,3] => mean along dim 2 => [2,2]
    let t = DynTensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[2, 2, 3],
        &cpu(),
    )
    .unwrap();
    let result = t.mean(D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    let data = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![2.0, 5.0, 8.0, 11.0]);
}

#[test]
fn test_d_minus2_max_3d() {
    // max(D::Minus2) on [2,3,2] => max along dim 1 => [2,2]
    let t = DynTensor::new(
        &[
            1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0, 11.0, 12.0, 9.0, 10.0,
        ],
        &[2, 3, 2],
        &cpu(),
    )
    .unwrap();
    let result = t.max(D::Minus2).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    let data = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![5.0, 6.0, 11.0, 12.0]);
}

// =============================================================================
// D::Minus1 tests for softmax, log_softmax, and cumsum
// =============================================================================

#[test]
fn test_d_minus1_softmax() {
    // softmax(D::Minus1) on [2,3] => softmax along dim 1
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3], &cpu()).unwrap();
    let result = t.softmax(D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
    let data = result.to_flat_vec::<f32>().unwrap();
    // Each row sums to 1.0
    assert!((data[0] + data[1] + data[2] - 1.0).abs() < 1e-6);
    assert!((data[3] + data[4] + data[5] - 1.0).abs() < 1e-6);
    // Values are monotonically increasing per row
    assert!(data[0] < data[1]);
    assert!(data[1] < data[2]);
}

#[test]
fn test_d_minus1_log_softmax() {
    // log_softmax(D::Minus1) on [2,3] => log_softmax along dim 1
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3], &cpu()).unwrap();
    let result = t.log_softmax(D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
    let data = result.to_flat_vec::<f32>().unwrap();
    // All log-softmax values should be negative (log of probability < 1)
    for &v in &data {
        assert!(v < 0.0);
    }
    // exp(log_softmax) should sum to 1.0
    let row0_sum: f32 = data[..3].iter().map(|x| x.exp()).sum();
    assert!((row0_sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_d_minus1_cumsum() {
    // cumsum(D::Minus1) on [2,4] => cumsum along dim 1
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4], &cpu()).unwrap();
    let result = t.cumsum(D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2, 4]);
    let data = result.to_flat_vec::<f32>().unwrap();
    // row 0: [1, 1+2, 1+2+3, 1+2+3+4] = [1, 3, 6, 10]
    // row 1: [5, 5+6, 5+6+7, 5+6+7+8] = [5, 11, 18, 26]
    assert_eq!(data, vec![1.0, 3.0, 6.0, 10.0, 5.0, 11.0, 18.0, 26.0]);
}

#[test]
fn test_d_minus2_cumsum() {
    // cumsum(D::Minus2) on [2,3] => cumsum along dim 0
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let result = t.cumsum(D::Minus2).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
    let data = result.to_flat_vec::<f32>().unwrap();
    // col-wise cumsum: [1,2,3] then [1+4, 2+5, 3+6] = [5, 7, 9]
    assert_eq!(data, vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);
}

// =============================================================================
// D::Minus1 tests for argmin and flatten
// =============================================================================

#[test]
fn test_d_minus1_argmin() {
    // argmin(D::Minus1) on [2,3] => argmin along dim 1 => [2] U32
    let t = DynTensor::new(&[3.0, 1.0, 2.0, 6.0, 4.0, 5.0], &[2, 3], &cpu()).unwrap();
    let result = t.argmin(D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2]);
    assert_eq!(result.dtype(), DType::U32);
    let data = result.to_vec1::<u32>().unwrap();
    // row 0 min at index 1 (value 1.0), row 1 min at index 1 (value 4.0)
    assert_eq!(data, vec![1, 1]);
}

#[test]
fn test_d_minus1_flatten() {
    // flatten(D::Minus2, D::Minus1) on [2,3,4] => flatten dims 1..2 => [2,12]
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    let result = t.flatten(D::Minus2, D::Minus1).unwrap();
    assert_eq!(result.dims(), &[2, 12]);
}
