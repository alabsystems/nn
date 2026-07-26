#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tensor-vs-tensor comparison ops, algorithm boundary condition tests,
//! and integer comparison regression tests (#1383).
//!
//! Extracted from `tests_extended.rs` to keep it under 500 lines (#1402).

use crate::dyn_tensor::test_helpers::{cpu, t1d};
use crate::{DType, DynTensor};

// -- Tensor-vs-tensor comparison ops ------------------------------------------

#[test]
fn test_broadcast_gt_same_shape() {
    let a = t1d(&[1.0, 5.0, 3.0]);
    let b = t1d(&[2.0, 4.0, 3.0]);
    let mask = a.broadcast_gt(&b).unwrap();
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), vec![0, 1, 0]);
}

#[test]
fn test_broadcast_eq_same_shape() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[1.0, 9.0, 3.0]);
    let mask = a.broadcast_eq(&b).unwrap();
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), vec![1, 0, 1]);
}

#[test]
fn test_broadcast_le_broadcast_shapes() {
    // [2,1] vs [1,3] → [2,3]
    let a = DynTensor::from_vec(vec![2.0, 4.0], &[2, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![1.0, 3.0, 5.0], &[1, 3], &cpu()).unwrap();
    let mask = a.broadcast_le(&b).unwrap();
    assert_eq!(mask.dims(), &[2, 3]);
    // 2<=1=0, 2<=3=1, 2<=5=1, 4<=1=0, 4<=3=0, 4<=5=1
    assert_eq!(
        mask.as_cpu_u8().unwrap().as_slice().unwrap(),
        vec![0, 1, 1, 0, 0, 1]
    );
}

#[test]
fn test_broadcast_ne_same_shape() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[1.0, 9.0, 3.0]);
    let mask = a.broadcast_ne(&b).unwrap();
    // 1!=1=0, 2!=9=1, 3!=3=0
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), vec![0, 1, 0]);
}

#[test]
fn test_broadcast_ne_with_broadcast() {
    // [2,1] vs [1,3] → [2,3]
    let a = DynTensor::from_vec(vec![1.0, 2.0], &[2, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let mask = a.broadcast_ne(&b).unwrap();
    assert_eq!(mask.dims(), &[2, 3]);
    // 1!=1=0, 1!=2=1, 1!=3=1, 2!=1=1, 2!=2=0, 2!=3=1
    assert_eq!(
        mask.as_cpu_u8().unwrap().as_slice().unwrap(),
        vec![0, 1, 1, 1, 0, 1]
    );
}

#[test]
fn test_broadcast_ne_incompatible_shapes_errors() {
    let a = t1d(&[1.0, 2.0]);
    let b = t1d(&[1.0, 2.0, 3.0]);
    assert!(a.broadcast_ne(&b).is_err());
}

#[test]
fn test_broadcast_ge_scalar_broadcast() {
    // [3] vs [] (scalar) → [3]
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = DynTensor::full(&[], 2.0, DType::F32, &cpu()).unwrap();
    let mask = a.broadcast_ge(&b).unwrap();
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), vec![0, 1, 1]);
}

#[test]
fn test_broadcast_lt_same_shape() {
    let a = t1d(&[1.0, 5.0, 3.0]);
    let b = t1d(&[2.0, 4.0, 3.0]);
    let mask = a.broadcast_lt(&b).unwrap();
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), vec![1, 0, 0]);
}

// -- Algorithm boundary condition tests (P1 algorithm_audit) ------------------

#[test]
fn test_gather_non_dim_bounds_error() {
    // gather with ids larger than self on non-gather dimension should error,
    // not panic with ndarray out-of-bounds.
    // self: [2, 3], ids: [4, 3] — ids dim 0 (4) > self dim 0 (2)
    let src = DynTensor::from_vec(vec![1.0; 6], &[2, 3], &cpu()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0; 12], &[4, 3], &cpu()).unwrap();
    let result = src.gather(&ids, 1);
    assert!(
        result.is_err(),
        "gather should error when ids exceeds self on non-gather dim"
    );
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("non-gather dim"),
        "error should mention non-gather dim: {msg}"
    );
}

#[test]
fn test_gather_valid_non_dim_same_size() {
    // ids same size as self on non-gather dims — should succeed.
    let src =
        DynTensor::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3], &cpu()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![2, 0, 1, 1, 2, 0], &[2, 3], &cpu()).unwrap();
    let result = src.gather(&ids, 1).unwrap();
    let v = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![30.0, 10.0, 20.0, 50.0, 60.0, 40.0]);
}

#[test]
fn test_scatter_add_non_dim_bounds_error() {
    // scatter_add with src larger than self on non-scatter dimension should error,
    // not panic with ndarray out-of-bounds.
    // self: [2, 3], src: [2, 5], index: [2, 5] — src dim 1 (5) > self dim 1 (3) for non-scatter dim
    let base = DynTensor::from_vec(vec![0.0; 6], &[2, 3], &cpu()).unwrap();
    let idx = DynTensor::from_vec_u32(vec![0; 10], &[2, 5], &cpu()).unwrap();
    let src = DynTensor::from_vec(vec![1.0; 10], &[2, 5], &cpu()).unwrap();
    let result = base.scatter_add(0, &idx, &src);
    assert!(
        result.is_err(),
        "scatter_add should error when src exceeds self on non-scatter dim"
    );
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("non-scatter dim"),
        "error should mention non-scatter dim: {msg}"
    );
}

// -- Regression tests for #1383 (compare_scalar integer-vs-fractional) --------

#[test]
fn test_i64_ge_fractional_excludes_truncated_value() {
    // Regression: `ge(1.5)` was truncated to `ge(1)`, incorrectly including x=1.
    let t = DynTensor::from_vec_i64(vec![0, 1, 2, 3], &[4], &cpu()).unwrap();
    let mask = t.ge(1.5).unwrap();
    // x=0 < 1.5 → 0, x=1 < 1.5 → 0, x=2 >= 1.5 → 1, x=3 >= 1.5 → 1
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), &[0, 0, 1, 1]);
}

#[test]
fn test_i64_lt_fractional_includes_truncated_value() {
    // Regression: `lt(1.5)` was truncated to `lt(1)`, excluding x=1.
    let t = DynTensor::from_vec_i64(vec![0, 1, 2, 3], &[4], &cpu()).unwrap();
    let mask = t.lt(1.5).unwrap();
    // x=0 < 1.5 → 1, x=1 < 1.5 → 1, x=2 < 1.5 → 0, x=3 < 1.5 → 0
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), &[1, 1, 0, 0]);
}

#[test]
fn test_i64_eq_fractional_returns_all_false() {
    // No integer equals 1.5.
    let t = DynTensor::from_vec_i64(vec![1, 2], &[2], &cpu()).unwrap();
    let mask = t.eq(1.5).unwrap();
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), &[0, 0]);
}

#[test]
fn test_u32_ge_fractional() {
    let t = DynTensor::from_vec_u32(vec![0, 1, 2, 3], &[4], &cpu()).unwrap();
    let mask = t.ge(1.5).unwrap();
    assert_eq!(mask.as_cpu_u8().unwrap().as_slice().unwrap(), &[0, 0, 1, 1]);
}

#[test]
fn test_i64_compare_nan_returns_error() {
    let t = DynTensor::from_vec_i64(vec![1, 2], &[2], &cpu()).unwrap();
    assert!(t.ge(f64::NAN).is_err());
}

#[test]
fn test_u32_compare_nan_returns_error() {
    let t = DynTensor::from_vec_u32(vec![1, 2], &[2], &cpu()).unwrap();
    assert!(t.ge(f64::NAN).is_err());
}

#[test]
fn test_u32_compare_inf_returns_error() {
    let t = DynTensor::from_vec_u32(vec![1, 2], &[2], &cpu()).unwrap();
    assert!(t.ge(f64::INFINITY).is_err());
}
