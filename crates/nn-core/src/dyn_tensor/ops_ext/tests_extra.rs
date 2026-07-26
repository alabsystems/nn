#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for DynTensor ops: cumsum, repeat_interleave, argmax/argmin
//! NaN boundary tests, argmax_keepdim, powf basic tests, and non-f32 dtype tests.
//!
//! Extracted from `dyn_tensor_ops_ext_tests.rs` to keep files under 500 lines.
//! Topk and powf boundary tests are in `tests_topk_boundary.rs`.

use crate::dyn_tensor::test_helpers::{cpu, tnd};
use crate::dyn_tensor::DynTensor;

// -- cumsum -------------------------------------------------------------------

#[test]
fn test_cumsum_1d() {
    let x = tnd(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let c = x.cumsum(0).unwrap();
    assert_eq!(c.dims(), &[4]);
    assert_eq!(c.to_flat_vec::<f32>().unwrap(), vec![1.0, 3.0, 6.0, 10.0]);
}

#[test]
fn test_cumsum_2d_dim0() {
    let x = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let c = x.cumsum(0).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]
    );
}

#[test]
fn test_cumsum_2d_dim1() {
    let x = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let c = x.cumsum(1).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(
        c.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]
    );
}

#[test]
fn test_cumsum_single_element() {
    let x = tnd(&[42.0], &[1]);
    let c = x.cumsum(0).unwrap();
    assert_eq!(c.to_flat_vec::<f32>().unwrap(), vec![42.0]);
}

#[test]
fn test_cumsum_invalid_dim() {
    let x = tnd(&[1.0, 2.0, 3.0], &[3]);
    assert!(x.cumsum(1).is_err());
}

// -- repeat_interleave --------------------------------------------------------

#[test]
fn test_repeat_interleave_basic() {
    let x = tnd(&[10.0, 20.0, 30.0], &[3]);
    let r = tnd(&[2.0, 1.0, 3.0], &[3]);
    let y = x.repeat_interleave(0, &r).unwrap();
    assert_eq!(y.dims(), &[6]);
    assert_eq!(
        y.to_flat_vec::<f32>().unwrap(),
        vec![10.0, 10.0, 20.0, 30.0, 30.0, 30.0]
    );
}

#[test]
fn test_repeat_interleave_2d_dim0() {
    let x = tnd(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let r = tnd(&[2.0, 1.0], &[2]);
    let y = x.repeat_interleave(0, &r).unwrap();
    assert_eq!(y.dims(), &[3, 2]);
    assert_eq!(
        y.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn test_repeat_interleave_2d_dim1() {
    let x = tnd(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let r = tnd(&[3.0, 1.0], &[2]);
    let y = x.repeat_interleave(1, &r).unwrap();
    assert_eq!(y.dims(), &[2, 4]);
    assert_eq!(
        y.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0]
    );
}

#[test]
fn test_repeat_interleave_uniform() {
    let x = tnd(&[1.0, 2.0, 3.0], &[3]);
    let r = tnd(&[2.0, 2.0, 2.0], &[3]);
    let y = x.repeat_interleave(0, &r).unwrap();
    assert_eq!(y.dims(), &[6]);
    assert_eq!(
        y.to_flat_vec::<f32>().unwrap(),
        vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    );
}

#[test]
fn test_repeat_interleave_all_zeros() {
    let x = tnd(&[1.0, 2.0, 3.0], &[3]);
    let r = tnd(&[0.0, 0.0, 0.0], &[3]);
    let y = x.repeat_interleave(0, &r).unwrap();
    assert_eq!(y.dims(), &[0]);
}

#[test]
fn test_repeat_interleave_length_mismatch() {
    let x = tnd(&[1.0, 2.0, 3.0], &[3]);
    let r = tnd(&[1.0, 1.0], &[2]);
    assert!(x.repeat_interleave(0, &r).is_err());
}

#[test]
fn test_repeat_interleave_negative_count() {
    let x = tnd(&[1.0, 2.0], &[2]);
    let r = tnd(&[1.0, -1.0], &[2]);
    assert!(x.repeat_interleave(0, &r).is_err());
}

#[test]
fn test_repeat_interleave_invalid_dim() {
    let x = tnd(&[1.0, 2.0, 3.0], &[3]);
    let r = tnd(&[1.0, 1.0, 1.0], &[3]);
    assert!(x.repeat_interleave(1, &r).is_err());
}

// -- Argmax/Argmin NaN boundary tests (#979 audit) ----------------------------

#[test]
fn test_argmax_nan_input_returns_error() {
    // NaN comparisons always return false, causing argmax to silently return
    // wrong index (0) instead of signaling invalid input.
    let x = tnd(&[1.0, f32::NAN, 3.0], &[3]);
    let result = x.argmax(0);
    assert!(result.is_err(), "argmax with NaN input should return Err");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("NaN"),
        "error should mention NaN, got: {err}"
    );
}

#[test]
fn test_argmin_nan_input_returns_error() {
    let x = tnd(&[1.0, f32::NAN, 3.0], &[3]);
    let result = x.argmin(0);
    assert!(result.is_err(), "argmin with NaN input should return Err");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("NaN"),
        "error should mention NaN, got: {err}"
    );
}

#[test]
fn test_argmax_empty_dim_returns_error() {
    // argmax on a dimension with size 0 has no valid index to return.
    let x = DynTensor::from_vec(vec![], &[0, 3], &cpu()).expect("empty tensor");
    let result = x.argmax(0);
    assert!(result.is_err(), "argmax on empty dim should return Err");
}

// -- argmax_keepdim / argmin_keepdim tests (#1226 follow-up) ------------------

#[test]
fn test_argmax_keepdim_basic() {
    let x = tnd(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
    let idx = x.argmax_keepdim(1).expect("argmax_keepdim");
    // Reduced dim kept as size 1: [2, 3] → [2, 1]
    assert_eq!(idx.dims(), &[2, 1]);
    assert_eq!(idx.dtype(), crate::DType::U32);
    let flat = idx.flatten_all().unwrap();
    assert_eq!(flat.to_vec1::<u32>().unwrap(), vec![1, 2]);
}

#[test]
fn test_argmin_keepdim_basic() {
    let x = tnd(&[3.0, 1.0, 5.0, 6.0, 4.0, 2.0], &[2, 3]);
    let idx = x.argmin_keepdim(1).expect("argmin_keepdim");
    assert_eq!(idx.dims(), &[2, 1]);
    assert_eq!(idx.dtype(), crate::DType::U32);
    let flat = idx.flatten_all().unwrap();
    assert_eq!(flat.to_vec1::<u32>().unwrap(), vec![1, 2]);
}

#[test]
fn test_argmax_keepdim_dim0() {
    let x = tnd(&[1.0, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
    let idx = x.argmax_keepdim(0).expect("argmax_keepdim dim0");
    // [2, 3] reduced along dim 0 → [1, 3]
    assert_eq!(idx.dims(), &[1, 3]);
    let flat = idx.flatten_all().unwrap();
    assert_eq!(flat.to_vec1::<u32>().unwrap(), vec![1, 0, 1]);
}

#[test]
fn test_argmax_keepdim_nan_returns_error() {
    let x = tnd(&[1.0, f32::NAN, 3.0], &[3]);
    let result = x.argmax_keepdim(0);
    assert!(result.is_err(), "argmax_keepdim with NaN should return Err");
}

// -- powf tests ---------------------------------------------------------------

#[test]
fn test_powf_square() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[4], &cpu()).unwrap();
    let y = x.powf(2.0).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 4.0, 9.0, 16.0]);
}

#[test]
fn test_powf_sqrt() {
    let x = DynTensor::new(&[1.0, 4.0, 9.0, 16.0], &[4], &cpu()).unwrap();
    let y = x.powf(0.5).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    for (a, b) in vals.iter().zip([1.0, 2.0, 3.0, 4.0].iter()) {
        assert!((a - b).abs() < 1e-6, "expected {b}, got {a}");
    }
}

#[test]
fn test_powf_identity() {
    let x = DynTensor::new(&[2.0, 3.0, 5.0], &[3], &cpu()).unwrap();
    let y = x.powf(1.0).unwrap();
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![2.0, 3.0, 5.0]);
}

#[test]
fn test_powf_2d() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();
    let y = x.powf(3.0).unwrap();
    assert_eq!(y.dims(), &[2, 2]);
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![1.0, 8.0, 27.0, 64.0]);
}

// -- cumsum non-f32 dtype tests (#1264) ---------------------------------------

#[test]
fn test_cumsum_u32() {
    let x = DynTensor::from_vec_u32(vec![1, 2, 3, 4], &[4], &cpu()).unwrap();
    let c = x.cumsum(0).unwrap();
    assert_eq!(c.dims(), &[4]);
    assert_eq!(c.dtype(), crate::DType::U32);
    assert_eq!(c.to_vec1::<u32>().unwrap(), vec![1, 3, 6, 10]);
}

#[test]
fn test_cumsum_i64() {
    let x = DynTensor::from_vec_i64(vec![10, -3, 5, -2], &[4], &cpu()).unwrap();
    let c = x.cumsum(0).unwrap();
    assert_eq!(c.dims(), &[4]);
    assert_eq!(c.dtype(), crate::DType::I64);
    assert_eq!(c.to_vec1::<i64>().unwrap(), vec![10, 7, 12, 10]);
}

#[test]
fn test_cumsum_u32_2d() {
    let x = DynTensor::from_vec_u32(vec![1, 2, 3, 4, 5, 6], &[2, 3], &cpu()).unwrap();
    let c = x.cumsum(1).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(c.dtype(), crate::DType::U32);
    // Row 0: [1, 1+2, 1+2+3] = [1, 3, 6]
    // Row 1: [4, 4+5, 4+5+6] = [4, 9, 15]
    let flat = c.flatten_all().unwrap();
    assert_eq!(flat.to_vec1::<u32>().unwrap(), vec![1, 3, 6, 4, 9, 15]);
}

// -- repeat_interleave non-f32 dtype tests (#1264) ----------------------------

#[test]
fn test_repeat_interleave_u32() {
    let x = DynTensor::from_vec_u32(vec![10, 20, 30], &[3], &cpu()).unwrap();
    let r = tnd(&[2.0, 1.0, 3.0], &[3]);
    let y = x.repeat_interleave(0, &r).unwrap();
    assert_eq!(y.dims(), &[6]);
    assert_eq!(y.dtype(), crate::DType::U32);
    assert_eq!(y.to_vec1::<u32>().unwrap(), vec![10, 10, 20, 30, 30, 30]);
}

#[test]
fn test_repeat_interleave_i64() {
    let x = DynTensor::from_vec_i64(vec![100, -200, 300], &[3], &cpu()).unwrap();
    let r = tnd(&[1.0, 2.0, 1.0], &[3]);
    let y = x.repeat_interleave(0, &r).unwrap();
    assert_eq!(y.dims(), &[4]);
    assert_eq!(y.dtype(), crate::DType::I64);
    assert_eq!(y.to_vec1::<i64>().unwrap(), vec![100, -200, -200, 300]);
}

#[test]
fn test_repeat_interleave_u32_2d() {
    let x = DynTensor::from_vec_u32(vec![1, 2, 3, 4], &[2, 2], &cpu()).unwrap();
    let r = tnd(&[2.0, 1.0], &[2]);
    let y = x.repeat_interleave(0, &r).unwrap();
    assert_eq!(y.dims(), &[3, 2]);
    assert_eq!(y.dtype(), crate::DType::U32);
    let flat = y.flatten_all().unwrap();
    assert_eq!(flat.to_vec1::<u32>().unwrap(), vec![1, 2, 1, 2, 3, 4]);
}

// -- masked_fill --------------------------------------------------------------

#[test]
fn test_masked_fill_basic() {
    let x = tnd(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let mask = DynTensor::from_vec_u8(vec![0, 1, 0, 1], &[4], &cpu()).unwrap();
    let y = x.masked_fill(&mask, -1.0).unwrap();
    assert_eq!(y.dims(), &[4]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, -1.0, 3.0, -1.0]);
}

#[test]
fn test_masked_fill_all_zeros_mask() {
    let x = tnd(&[10.0, 20.0, 30.0], &[3]);
    let mask = DynTensor::from_vec_u8(vec![0, 0, 0], &[3], &cpu()).unwrap();
    let y = x.masked_fill(&mask, 999.0).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_masked_fill_all_ones_mask() {
    let x = tnd(&[1.0, 2.0, 3.0], &[3]);
    let mask = DynTensor::from_vec_u8(vec![1, 1, 1], &[3], &cpu()).unwrap();
    let y = x.masked_fill(&mask, 0.0).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 0.0, 0.0]);
}

#[test]
fn test_masked_fill_2d() {
    let x = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let mask = DynTensor::from_vec_u8(vec![1, 0, 1, 0, 1, 0], &[2, 3], &cpu()).unwrap();
    let y = x.masked_fill(&mask, -9.0).unwrap();
    assert_eq!(y.dims(), &[2, 3]);
    let vals = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![-9.0, 2.0, -9.0, 4.0, -9.0, 6.0]);
}
