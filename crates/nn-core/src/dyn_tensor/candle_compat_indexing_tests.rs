#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! IndexOp `.i()` tests for candle-compatible DynTensor APIs.

use super::*;

// -- i() indexing tests ------------------------------------------------------

#[test]
fn test_i_select_single_index() {
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[2, 3], &cpu()).unwrap();
    // t.i(0) selects row 0, reducing rank: [2,3] -> [3]
    let r = t.i(0).unwrap();
    assert_eq!(r.dims(), &[3]);
    assert_eq!(r.to_vec1::<f32>().unwrap(), &[0.0, 1.0, 2.0]);
}

#[test]
fn test_i_select_second_row() {
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[2, 3], &cpu()).unwrap();
    let r = t.i(1).unwrap();
    assert_eq!(r.to_vec1::<f32>().unwrap(), &[3.0, 4.0, 5.0]);
}

#[test]
fn test_i_range_on_dim0() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[4, 3], &cpu()).unwrap();
    // t.i(1..3) narrows dim 0 to rows 1 and 2
    let r = t.i(1..3).unwrap();
    assert_eq!(r.dims(), &[2, 3]);
    let v = r.to_vec2::<f32>().unwrap();
    assert_eq!(v[0], &[3.0, 4.0, 5.0]);
    assert_eq!(v[1], &[6.0, 7.0, 8.0]);
}

#[test]
fn test_i_range_to_on_dim0() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[4, 3], &cpu()).unwrap();
    let r = t.i(..2).unwrap();
    assert_eq!(r.dims(), &[2, 3]);
}

#[test]
fn test_i_tuple_full_range_on_last_dim() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[3, 4], &cpu()).unwrap();
    // t.i((.., 1..3)) narrows last dim to [1, 3)
    let r = t.i((.., 1..3)).unwrap();
    assert_eq!(r.dims(), &[3, 2]);
    let v = r.to_vec2::<f32>().unwrap();
    assert_eq!(v[0], &[1.0, 2.0]);
    assert_eq!(v[1], &[5.0, 6.0]);
    assert_eq!(v[2], &[9.0, 10.0]);
}

#[test]
fn test_i_tuple_range_full() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[4, 3], &cpu()).unwrap();
    // t.i((1..3, ..)) narrows dim 0, keeps dim 1
    let r = t.i((1..3, ..)).unwrap();
    assert_eq!(r.dims(), &[2, 3]);
}

#[test]
fn test_i_out_of_bounds_error() {
    let t = DynTensor::new(&[1.0, 2.0], &[2], &cpu()).unwrap();
    assert!(t.i(5).is_err());
}

#[test]
fn test_i_range_out_of_bounds_error() {
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    assert!(t.i(1..5).is_err());
}

#[test]
fn test_i_select_with_full() {
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[2, 3], &cpu()).unwrap();
    // (0, ..) selects row 0 from a 2D tensor
    let r = t.i((0, ..)).unwrap();
    assert_eq!(r.dims(), &[3]);
    assert_eq!(r.to_vec1::<f32>().unwrap(), &[0.0, 1.0, 2.0]);
}

#[test]
fn test_i_range_from() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[4, 3], &cpu()).unwrap();
    // t.i(2..) narrows dim 0 from index 2 to end
    let r = t.i(2..).unwrap();
    assert_eq!(r.dims(), &[2, 3]);
    let v = r.to_vec2::<f32>().unwrap();
    assert_eq!(v[0], &[6.0, 7.0, 8.0]);
    assert_eq!(v[1], &[9.0, 10.0, 11.0]);
}

// -- 2-tuple IndexOp tests (AC7) ---------------------------------------------

#[test]
fn test_i_2tuple_select_select() {
    // t.i((0, 1)) — select row 0, then column 1 → scalar
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[2, 3], &cpu()).unwrap();
    let r = t.i((0usize, 1usize)).unwrap();
    assert_eq!(r.dims(), &[] as &[usize]);
    assert_eq!(r.to_scalar::<f32>().unwrap(), 1.0);
}

#[test]
fn test_i_2tuple_select_range() {
    // t.i((0, 1..3)) — select row 0, narrow cols to [1, 3)
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[3, 4], &cpu()).unwrap();
    let r = t.i((0usize, 1..3)).unwrap();
    assert_eq!(r.dims(), &[2]);
    assert_eq!(r.to_vec1::<f32>().unwrap(), &[1.0, 2.0]);
}

#[test]
fn test_i_2tuple_range_select() {
    // t.i((1..3, 0)) — narrow rows to [1, 3), select col 0
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[4, 3], &cpu()).unwrap();
    let r = t.i((1..3, 0usize)).unwrap();
    assert_eq!(r.dims(), &[2]);
    assert_eq!(r.to_vec1::<f32>().unwrap(), &[3.0, 6.0]);
}

// -- 3-tuple IndexOp tests (AC7) ---------------------------------------------

#[test]
fn test_i_3tuple_select_select_select() {
    // t.i((0, 1, 2)) — select batch 0, row 1, col 2 → scalar
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[2, 3, 4], &cpu()).unwrap();
    let r = t.i((0usize, 1usize, 2usize)).unwrap();
    assert_eq!(r.dims(), &[] as &[usize]);
    assert_eq!(r.to_scalar::<f32>().unwrap(), 6.0); // 0*12 + 1*4 + 2 = 6
}

#[test]
fn test_i_3tuple_select_full_range() {
    // t.i((0, .., 1..3)) — select batch 0, keep all rows, narrow cols
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[2, 3, 4], &cpu()).unwrap();
    let r = t.i((0usize, .., 1..3)).unwrap();
    assert_eq!(r.dims(), &[3, 2]);
    let v = r.to_vec2::<f32>().unwrap();
    assert_eq!(v[0], &[1.0, 2.0]);
    assert_eq!(v[1], &[5.0, 6.0]);
    assert_eq!(v[2], &[9.0, 10.0]);
}

#[test]
fn test_i_3tuple_range_full_select() {
    // t.i((1..2, .., 0)) — narrow batch, keep rows, select first col
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[2, 3, 4], &cpu()).unwrap();
    let r = t.i((1..2, .., 0usize)).unwrap();
    assert_eq!(r.dims(), &[1, 3]);
    let flat = r.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat, &[12.0, 16.0, 20.0]);
}

#[test]
fn test_i_3tuple_full_full_select() {
    // t.i((.., .., 0)) — keep batch, keep rows, select col 0
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[2, 3, 4], &cpu()).unwrap();
    let r = t.i((.., .., 0usize)).unwrap();
    assert_eq!(r.dims(), &[2, 3]);
    let v = r.to_vec2::<f32>().unwrap();
    assert_eq!(v[0], &[0.0, 4.0, 8.0]);
    assert_eq!(v[1], &[12.0, 16.0, 20.0]);
}

#[test]
fn test_i_3tuple_out_of_bounds() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[2, 3, 4], &cpu()).unwrap();
    assert!(t.i((5usize, 0usize, 0usize)).is_err());
}

// -- RangeInclusive / RangeToInclusive IndexOp tests --------------------------

#[test]
fn test_i_range_inclusive() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[4, 3], &cpu()).unwrap();
    // t.i(1..=2) narrows dim 0 to rows 1 and 2 (inclusive)
    let r = t.i(1..=2).unwrap();
    assert_eq!(r.dims(), &[2, 3]);
    let v = r.to_vec2::<f32>().unwrap();
    assert_eq!(v[0], &[3.0, 4.0, 5.0]);
    assert_eq!(v[1], &[6.0, 7.0, 8.0]);
}

#[test]
fn test_i_range_to_inclusive() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[4, 3], &cpu()).unwrap();
    // t.i(..=1) narrows dim 0 to rows 0 and 1 (inclusive)
    let r = t.i(..=1).unwrap();
    assert_eq!(r.dims(), &[2, 3]);
    let v = r.to_vec2::<f32>().unwrap();
    assert_eq!(v[0], &[0.0, 1.0, 2.0]);
    assert_eq!(v[1], &[3.0, 4.0, 5.0]);
}

#[test]
fn test_i_range_inclusive_single_element() {
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[6], &cpu()).unwrap();
    // t.i(2..=2) narrows to single element [2..3)
    let r = t.i(2..=2).unwrap();
    assert_eq!(r.dims(), &[1]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), &[2.0]);
}

#[test]
fn test_i_range_inclusive_in_tuple() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[2, 3, 4], &cpu()).unwrap();
    // t.i((0, .., 1..=2)) — select batch 0, keep rows, columns 1 through 2
    let r = t.i((0usize, .., 1..=2)).unwrap();
    assert_eq!(r.dims(), &[3, 2]);
    let v = r.to_vec2::<f32>().unwrap();
    assert_eq!(v[0], &[1.0, 2.0]);
    assert_eq!(v[1], &[5.0, 6.0]);
    assert_eq!(v[2], &[9.0, 10.0]);
}

#[test]
#[allow(clippy::reversed_empty_ranges)]
fn test_i_range_inclusive_inverted_is_empty() {
    // 5..=2 is an empty range — should produce a zero-length narrow and fail
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[6], &cpu()).unwrap();
    // An empty narrow (len=0) will fail at apply time since 5+0 <= 6 is valid
    // but narrow with len=0 produces an empty tensor which is fine.
    let r = t.i(5..=2usize);
    // Either Ok with empty dim or Err is acceptable — just must not panic.
    if let Ok(ref tensor) = r {
        assert_eq!(tensor.dims()[0], 0);
    }
}

#[test]
fn test_i_range_to_inclusive_zero() {
    // ..=0 should narrow to first element only
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[6], &cpu()).unwrap();
    let r = t.i(..=0usize).unwrap();
    assert_eq!(r.dims(), &[1]);
    assert_eq!(r.to_flat_vec::<f32>().unwrap(), &[0.0]);
}
