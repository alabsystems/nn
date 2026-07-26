#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for selection operations on non-f32 dtypes (U32, I64, U8).
//! Verifies fix for #1264: index_select, gather, where_cond, expand.

use crate::dyn_tensor::test_helpers::cpu;
use crate::dyn_tensor::DynTensor;
use crate::DType;

// -- index_select -------------------------------------------------------------

#[test]
fn test_index_select_u32() {
    let data = DynTensor::from_vec_u32(vec![10, 20, 30, 40, 50], &[5], &cpu()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 2, 4], &[3], &cpu()).unwrap();
    let result = data.index_select(&ids, 0).unwrap();
    assert_eq!(result.dims(), &[3]);
    assert_eq!(result.dtype(), DType::U32);
    let vals = result.as_cpu_u32().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[10, 30, 50]);
}

#[test]
fn test_index_select_i64() {
    let data = DynTensor::from_vec_i64(vec![100, 200, 300, 400], &[4], &cpu()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![1, 3], &[2], &cpu()).unwrap();
    let result = data.index_select(&ids, 0).unwrap();
    assert_eq!(result.dims(), &[2]);
    assert_eq!(result.dtype(), DType::I64);
    let vals = result.as_cpu_i64().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[200, 400]);
}

#[test]
fn test_index_select_u32_2d() {
    let data = DynTensor::from_vec_u32(vec![1, 2, 3, 4, 5, 6], &[2, 3], &cpu()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 2], &[2], &cpu()).unwrap();
    let result = data.index_select(&ids, 1).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    assert_eq!(result.dtype(), DType::U32);
    let vals = result.as_cpu_u32().unwrap();
    // Original [[1,2,3],[4,5,6]], select cols 0,2 → [[1,3],[4,6]]
    assert_eq!(vals.as_slice().unwrap(), &[1, 3, 4, 6]);
}

// -- gather -------------------------------------------------------------------

#[test]
fn test_gather_u32() {
    // 1-D gather: self=[10,20,30], ids=[2,0,1], dim=0 → [30,10,20]
    let data = DynTensor::from_vec_u32(vec![10, 20, 30], &[3], &cpu()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![2, 0, 1], &[3], &cpu()).unwrap();
    let result = data.gather(&ids, 0).unwrap();
    assert_eq!(result.dims(), &[3]);
    assert_eq!(result.dtype(), DType::U32);
    let vals = result.as_cpu_u32().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[30, 10, 20]);
}

#[test]
fn test_gather_i64() {
    let data = DynTensor::from_vec_i64(vec![100, 200, 300], &[3], &cpu()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![1, 1, 0], &[3], &cpu()).unwrap();
    let result = data.gather(&ids, 0).unwrap();
    assert_eq!(result.dims(), &[3]);
    assert_eq!(result.dtype(), DType::I64);
    let vals = result.as_cpu_i64().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[200, 200, 100]);
}

// -- where_cond ---------------------------------------------------------------

#[test]
fn test_where_cond_u32() {
    let mask = DynTensor::from_vec_u8(vec![1, 0, 1, 0], &[4], &cpu()).unwrap();
    let on_true = DynTensor::from_vec_u32(vec![10, 20, 30, 40], &[4], &cpu()).unwrap();
    let on_false = DynTensor::from_vec_u32(vec![1, 2, 3, 4], &[4], &cpu()).unwrap();
    let result = mask.where_cond(&on_true, &on_false).unwrap();
    assert_eq!(result.dims(), &[4]);
    assert_eq!(result.dtype(), DType::U32);
    let vals = result.as_cpu_u32().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[10, 2, 30, 4]);
}

#[test]
fn test_where_cond_i64() {
    let mask = DynTensor::from_vec_u8(vec![0, 1], &[2], &cpu()).unwrap();
    let on_true = DynTensor::from_vec_i64(vec![100, 200], &[2], &cpu()).unwrap();
    let on_false = DynTensor::from_vec_i64(vec![999, 888], &[2], &cpu()).unwrap();
    let result = mask.where_cond(&on_true, &on_false).unwrap();
    assert_eq!(result.dims(), &[2]);
    assert_eq!(result.dtype(), DType::I64);
    let vals = result.as_cpu_i64().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[999, 200]);
}

#[test]
fn test_where_cond_u8() {
    let mask = DynTensor::from_vec_u8(vec![1, 0, 1, 0, 1], &[5], &cpu()).unwrap();
    let on_true = DynTensor::from_vec_u8(vec![10, 20, 30, 40, 50], &[5], &cpu()).unwrap();
    let on_false = DynTensor::from_vec_u8(vec![1, 2, 3, 4, 5], &[5], &cpu()).unwrap();
    let result = mask.where_cond(&on_true, &on_false).unwrap();
    assert_eq!(result.dims(), &[5]);
    assert_eq!(result.dtype(), DType::U8);
    let vals = result.as_cpu_u8().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[10, 2, 30, 4, 50]);
}

#[test]
fn test_where_cond_dtype_mismatch() {
    let mask = DynTensor::from_vec_u8(vec![1, 0], &[2], &cpu()).unwrap();
    let on_true = DynTensor::from_vec_u32(vec![1, 2], &[2], &cpu()).unwrap();
    let on_false = DynTensor::from_vec(vec![3.0f32, 4.0], &[2], &cpu()).unwrap();
    let result = mask.where_cond(&on_true, &on_false);
    assert!(result.is_err(), "where_cond should reject dtype mismatch");
}

// -- expand -------------------------------------------------------------------

#[test]
fn test_expand_u32() {
    let data = DynTensor::from_vec_u32(vec![42], &[1, 1], &cpu()).unwrap();
    let result = data.expand([3, 4]).unwrap();
    assert_eq!(result.dims(), &[3, 4]);
    assert_eq!(result.dtype(), DType::U32);
    let vals = result.as_cpu_u32().unwrap();
    assert!(vals.iter().all(|&v| v == 42));
    assert_eq!(vals.len(), 12);
}

#[test]
fn test_expand_i64() {
    let data = DynTensor::from_vec_i64(vec![7, 8, 9], &[1, 3], &cpu()).unwrap();
    let result = data.expand([2, 3]).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
    assert_eq!(result.dtype(), DType::I64);
    let vals = result.as_cpu_i64().unwrap();
    // [[7,8,9]] expanded to [[7,8,9],[7,8,9]]
    assert_eq!(vals.as_slice().unwrap(), &[7, 8, 9, 7, 8, 9]);
}

// -- f32 regressions ----------------------------------------------------------

#[test]
fn test_index_select_f32_regression() {
    let data = DynTensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &cpu()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![4, 2, 0], &[3], &cpu()).unwrap();
    let result = data.index_select(&ids, 0).unwrap();
    assert_eq!(result.dims(), &[3]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![5.0, 3.0, 1.0]);
}

#[test]
fn test_expand_f32_regression() {
    let data = DynTensor::from_vec(vec![1.0f32, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let result = data.expand([4, 3]).unwrap();
    assert_eq!(result.dims(), &[4, 3]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(
        vals,
        vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );
}
