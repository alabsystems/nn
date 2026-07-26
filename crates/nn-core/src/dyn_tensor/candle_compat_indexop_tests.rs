#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Candle-compatible IndexOp tests for 4-tuple/5-tuple indexing,
//! Shape type, broadcast_as, broadcast_left, and Shape conversions.
//!
//! Extracted from `candle_compat_tests.rs` for file-size compliance (#1227).

use super::*;
use crate::dyn_tensor::test_helpers::cpu;
use crate::DType;

// -- 4-tuple and 5-tuple IndexOp tests ----------------------------------------

#[test]
fn test_i_4tuple_select_all() {
    let data: Vec<f32> = (0..120).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[2, 3, 4, 5], &cpu()).unwrap();
    // Select batch 0, row 1, col 2, elem 3 → scalar
    let r = t.i((0usize, 1usize, 2usize, 3usize)).unwrap();
    assert_eq!(r.dims(), &[] as &[usize]);
    // index = 0*60 + 1*20 + 2*5 + 3 = 33
    assert_eq!(r.to_scalar::<f32>().unwrap(), 33.0);
}

#[test]
fn test_i_4tuple_mixed() {
    let data: Vec<f32> = (0..120).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[2, 3, 4, 5], &cpu()).unwrap();
    // t.i((0, .., 1..3, ..)) — select batch 0, keep rows, narrow cols, keep elem
    let r = t.i((0usize, .., 1..3, ..)).unwrap();
    assert_eq!(r.dims(), &[3, 2, 5]);
}

#[test]
fn test_i_5tuple_all_select() {
    let data: Vec<f32> = (0..720).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[2, 3, 4, 5, 6], &cpu()).unwrap();
    // Select all dims → scalar
    let r = t.i((1usize, 2usize, 3usize, 4usize, 5usize)).unwrap();
    assert_eq!(r.dims(), &[] as &[usize]);
    // index = 1*360 + 2*120 + 3*30 + 4*6 + 5 = 360+240+90+24+5 = 719
    assert_eq!(r.to_scalar::<f32>().unwrap(), 719.0);
}

#[test]
fn test_i_5tuple_mixed_ranges() {
    let data: Vec<f32> = (0..720).map(|i| i as f32).collect();
    let t = DynTensor::new(&data, &[2, 3, 4, 5, 6], &cpu()).unwrap();
    // t.i((.., .., 1..3, .., 0)) — keep first 2 dims, narrow dim 2, keep dim 3, select dim 4
    let r = t.i((.., .., 1..3, .., 0usize)).unwrap();
    assert_eq!(r.dims(), &[2, 3, 2, 5]);
}

// -- Shape type tests (#1111) -------------------------------------------------

#[test]
fn test_shape_from_dims() {
    let s = Shape::from_dims(&[2, 3, 4]);
    assert_eq!(s.dims(), &[2, 3, 4]);
    assert_eq!(s.to_vec(), vec![2, 3, 4]);
    assert_eq!(s.elem_count(), 24);
    assert_eq!(s.rank(), 3);
}

#[test]
fn test_shape_empty_dims() {
    let s = Shape::from_dims(&[]);
    assert_eq!(s.dims(), &[] as &[usize]);
    assert_eq!(s.elem_count(), 1); // product of empty = 1
    assert_eq!(s.rank(), 0);
}

#[test]
fn test_shape_equality() {
    let a = Shape::from_dims(&[2, 3]);
    let b = Shape::from_dims(&[2, 3]);
    let c = Shape::from_dims(&[3, 2]);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_dyn_tensor_shape_method() {
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    let s = t.shape();
    assert_eq!(s.dims(), t.dims());
    assert_eq!(s.elem_count(), 24);
    assert_eq!(s.rank(), 3);
}

#[test]
fn test_dyn_tensor_elem_count() {
    let t = DynTensor::zeros(&[2, 3, 4], DType::F32, &cpu()).unwrap();
    assert_eq!(t.elem_count(), 24);
    assert_eq!(t.elem_count(), t.numel());
}

#[test]
fn test_broadcast_as_shape() {
    let t = DynTensor::new(&[1.0], &[1, 1], &cpu()).unwrap();
    let target = Shape::from_dims(&[3, 4]);
    let result = t.broadcast_as(&target).unwrap();
    assert_eq!(result.dims(), &[3, 4]);
    assert_eq!(result.elem_count(), 12);
}

#[test]
fn test_broadcast_as_candle_pattern() {
    // Candle pattern: Tensor::new(&[1e-12], device)?.broadcast_as(x.shape())?
    let eps = DynTensor::new(&[1e-12], &[1, 1], &cpu()).unwrap();
    let x = DynTensor::zeros(&[4, 8], DType::F32, &cpu()).unwrap();
    let result = eps.broadcast_as(x.shape()).unwrap();
    assert_eq!(result.dims(), &[4, 8]);
}

#[test]
fn test_broadcast_as_incompatible_shape() {
    let t = DynTensor::new(&[1.0, 2.0], &[2], &cpu()).unwrap();
    let target = Shape::from_dims(&[3]);
    assert!(t.broadcast_as(&target).is_err());
}

// ---------------------------------------------------------------------------
// broadcast_left tests
// ---------------------------------------------------------------------------

#[test]
fn test_broadcast_left_scalar_prepend() {
    // [T, C] -> broadcast_left(B) -> [B, T, C]
    let t = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let result = t.broadcast_left(4usize).unwrap();
    assert_eq!(result.dims(), &[4, 2, 3]);
    // Each batch slice should be identical to original
    let data = result.to_flat_vec::<f32>().unwrap();
    assert!((data[0] - 1.0).abs() < 1e-6); // batch 0
    assert!((data[6] - 1.0).abs() < 1e-6); // batch 1
}

#[test]
fn test_broadcast_left_tuple_prepend() {
    // [C] -> broadcast_left((B, T)) -> [B, T, C]
    let t = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let result = t.broadcast_left((2usize, 4usize)).unwrap();
    assert_eq!(result.dims(), &[2, 4, 3]);
    let data = result.to_flat_vec::<f32>().unwrap();
    // All positions should repeat [1, 2, 3]
    assert!((data[0] - 1.0).abs() < 1e-6);
    assert!((data[1] - 2.0).abs() < 1e-6);
    assert!((data[2] - 3.0).abs() < 1e-6);
    assert!((data[3] - 1.0).abs() < 1e-6); // next time step
}

#[test]
fn test_broadcast_left_1d_to_3d() {
    // Dvoice pattern: [channels].broadcast_left(batch)
    let t = DynTensor::new(&[10.0, 20.0], &[2], &cpu()).unwrap();
    let result = t.broadcast_left(3usize).unwrap();
    assert_eq!(result.dims(), &[3, 2]);
    let data = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, vec![10.0, 20.0, 10.0, 20.0, 10.0, 20.0]);
}

// ---------------------------------------------------------------------------
// Shape From conversions
// ---------------------------------------------------------------------------

#[test]
fn test_shape_from_usize() {
    let s: Shape = 5usize.into();
    assert_eq!(s.dims(), &[5]);
    assert_eq!(s.rank(), 1);
}

#[test]
fn test_shape_from_tuple_2d() {
    let s: Shape = (3usize, 4usize).into();
    assert_eq!(s.dims(), &[3, 4]);
    assert_eq!(s.rank(), 2);
}

#[test]
fn test_shape_from_tuple_3d() {
    let s: Shape = (2usize, 3usize, 4usize).into();
    assert_eq!(s.dims(), &[2, 3, 4]);
    assert_eq!(s.elem_count(), 24);
}

#[test]
fn test_shape_from_vec() {
    let s: Shape = vec![1, 2, 3, 4].into();
    assert_eq!(s.dims(), &[1, 2, 3, 4]);
    assert_eq!(s.rank(), 4);
}
