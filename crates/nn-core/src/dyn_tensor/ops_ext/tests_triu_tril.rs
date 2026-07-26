#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for triu (upper triangular) and tril (lower triangular) ops.
//!
//! Extracted from `tests_extra.rs` to keep files under 500 lines.

use crate::dyn_tensor::test_helpers::cpu;
use crate::dyn_tensor::DynTensor;

#[test]
fn test_triu_3x3_default_diagonal() {
    #[rustfmt::skip]
    let x = DynTensor::new(&[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ], &[3, 3], &cpu()).unwrap();
    let u = x.triu(0).unwrap();
    #[rustfmt::skip]
    assert_eq!(u.to_flat_vec::<f32>().unwrap(), vec![
        1.0, 2.0, 3.0,
        0.0, 5.0, 6.0,
        0.0, 0.0, 9.0,
    ]);
}

#[test]
fn test_tril_3x3_default_diagonal() {
    #[rustfmt::skip]
    let x = DynTensor::new(&[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ], &[3, 3], &cpu()).unwrap();
    let l = x.tril(0).unwrap();
    #[rustfmt::skip]
    assert_eq!(l.to_flat_vec::<f32>().unwrap(), vec![
        1.0, 0.0, 0.0,
        4.0, 5.0, 0.0,
        7.0, 8.0, 9.0,
    ]);
}

#[test]
fn test_triu_positive_diagonal() {
    #[rustfmt::skip]
    let x = DynTensor::new(&[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ], &[3, 3], &cpu()).unwrap();
    let u = x.triu(1).unwrap();
    #[rustfmt::skip]
    assert_eq!(u.to_flat_vec::<f32>().unwrap(), vec![
        0.0, 2.0, 3.0,
        0.0, 0.0, 6.0,
        0.0, 0.0, 0.0,
    ]);
}

#[test]
fn test_tril_negative_diagonal() {
    #[rustfmt::skip]
    let x = DynTensor::new(&[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ], &[3, 3], &cpu()).unwrap();
    let l = x.tril(-1).unwrap();
    #[rustfmt::skip]
    assert_eq!(l.to_flat_vec::<f32>().unwrap(), vec![
        0.0, 0.0, 0.0,
        4.0, 0.0, 0.0,
        7.0, 8.0, 0.0,
    ]);
}

#[test]
fn test_triu_batched_3d() {
    #[rustfmt::skip]
    let x = DynTensor::new(&[
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0,
    ], &[2, 2, 2], &cpu()).unwrap();
    let u = x.triu(0).unwrap();
    assert_eq!(u.dims(), &[2, 2, 2]);
    #[rustfmt::skip]
    assert_eq!(u.to_flat_vec::<f32>().unwrap(), vec![
        1.0, 2.0,
        0.0, 4.0,
        5.0, 6.0,
        0.0, 8.0,
    ]);
}

#[test]
fn test_triu_non_square() {
    #[rustfmt::skip]
    let x = DynTensor::new(&[
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
    ], &[3, 4], &cpu()).unwrap();
    let u = x.triu(0).unwrap();
    #[rustfmt::skip]
    assert_eq!(u.to_flat_vec::<f32>().unwrap(), vec![
        1.0, 2.0, 3.0, 4.0,
        0.0, 6.0, 7.0, 8.0,
        0.0, 0.0, 11.0, 12.0,
    ]);
}

#[test]
fn test_triu_rank1_error() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    assert!(x.triu(0).is_err());
}

#[test]
fn test_tril_rank1_error() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    assert!(x.tril(0).is_err());
}

#[test]
fn test_triu_tril_complement() {
    // triu(0) + tril(-1) should equal the original (diagonal counted once).
    #[rustfmt::skip]
    let x = DynTensor::new(&[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ], &[3, 3], &cpu()).unwrap();
    let u = x.triu(0).unwrap();
    let l = x.tril(-1).unwrap();
    let sum = u.broadcast_add(&l).unwrap();
    assert_eq!(
        sum.to_flat_vec::<f32>().unwrap(),
        x.to_flat_vec::<f32>().unwrap()
    );
}
