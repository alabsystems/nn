// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comparison operator tests for non-f32 dtypes (I64, U32, U8).
//! Extracted from tests.rs for 500-line limit compliance.
//! Verifies fix for #1219: compare_scalar() dispatches by dtype natively.

use crate::dyn_tensor::test_helpers::cpu;
use crate::DynTensor;

// -- I64 comparison tests (#1207 gap from M1 directive) ----------------------

#[test]
fn test_i64_ge_basic() {
    let t = DynTensor::from_vec_i64(vec![1, 5, 10, 20], &[4], &cpu()).unwrap();
    let mask = t.ge(10.0).unwrap();
    assert_eq!(
        mask.as_cpu_u8().unwrap().as_slice().unwrap(),
        vec![0, 0, 1, 1]
    );
}

#[test]
fn test_i64_gt_basic() {
    let t = DynTensor::from_vec_i64(vec![1, 10, 20], &[3], &cpu()).unwrap();
    let mask = t.gt(10.0).unwrap();
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), vec![0, 0, 1]);
}

#[test]
fn test_i64_lt_basic() {
    let t = DynTensor::from_vec_i64(vec![1, 10, 20], &[3], &cpu()).unwrap();
    let mask = t.lt(10.0).unwrap();
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), vec![1, 0, 0]);
}

#[test]
fn test_i64_le_basic() {
    let t = DynTensor::from_vec_i64(vec![1, 10, 20], &[3], &cpu()).unwrap();
    let mask = t.le(10.0).unwrap();
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), vec![1, 1, 0]);
}

#[test]
fn test_i64_eq_basic() {
    let t = DynTensor::from_vec_i64(vec![5, 10, 15], &[3], &cpu()).unwrap();
    let mask = t.eq(10.0).unwrap();
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), vec![0, 1, 0]);
}

#[test]
fn test_i64_ne_basic() {
    let t = DynTensor::from_vec_i64(vec![5, 10, 15], &[3], &cpu()).unwrap();
    let mask = t.ne(10.0).unwrap();
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), vec![1, 0, 1]);
}

// -- U32 comparison tests ----------------------------------------------------

#[test]
fn test_u32_ge_basic() {
    let t = DynTensor::from_vec_u32(vec![1, 5, 10, 20], &[4], &cpu()).unwrap();
    let mask = t.ge(10.0).unwrap();
    assert_eq!(
        mask.as_cpu_u8().unwrap().as_slice().unwrap(),
        vec![0, 0, 1, 1]
    );
}

#[test]
fn test_u32_eq_basic() {
    let t = DynTensor::from_vec_u32(vec![5, 10, 15], &[3], &cpu()).unwrap();
    let mask = t.eq(10.0).unwrap();
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), vec![0, 1, 0]);
}

// -- U8 comparison tests -----------------------------------------------------

#[test]
fn test_u8_ge_basic() {
    let t = DynTensor::from_vec_u8(vec![1, 5, 10, 20], &[4], &cpu()).unwrap();
    let mask = t.ge(10.0).unwrap();
    assert_eq!(
        mask.as_cpu_u8().unwrap().as_slice().unwrap(),
        vec![0, 0, 1, 1]
    );
}

#[test]
fn test_u8_eq_basic() {
    let t = DynTensor::from_vec_u8(vec![0, 128, 255], &[3], &cpu()).unwrap();
    let mask = t.eq(128.0).unwrap();
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), vec![0, 1, 0]);
}
