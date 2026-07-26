#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for pad1d and pad_with_zeros operations on DynTensor.
//!
//! Extracted from dyn_tensor_conv_tests.rs for 500-line limit.

use crate::dyn_tensor::test_helpers::cpu;
use crate::{DynTensor, TensorError};

// -- Pad1d tests ----------------------------------------------------------

#[test]
fn test_pad1d_basic() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let p = x.pad1d(1, 1).unwrap();
    assert_eq!(p.dims(), &[1, 1, 5]);
    let v = p.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![0.0, 1.0, 2.0, 3.0, 0.0]);
}

#[test]
fn test_pad1d_left_only() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let p = x.pad1d(2, 0).unwrap();
    assert_eq!(p.dims(), &[1, 1, 5]);
    let v = p.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![0.0, 0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_pad1d_1d_tensor() {
    let x = DynTensor::from_vec(vec![1.0, 2.0], &[2], &cpu()).unwrap();
    let p = x.pad1d(1, 2).unwrap();
    assert_eq!(p.dims(), &[5]);
    let v = p.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![0.0, 1.0, 2.0, 0.0, 0.0]);
}

// -- Pad1d non-contiguous input -------------------------------------------

#[test]
fn test_pad1d_non_contiguous_input() {
    // Original [2,3]: [[1,2,3],[4,5,6]]. After transpose: [[1,4],[2,5],[3,6]].
    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let x_t = x.transpose(0, 1).unwrap(); // [3, 2], non-contiguous
    assert_eq!(x_t.dims(), &[3, 2]);
    let result = x_t.pad1d(1, 1);
    assert!(
        result.is_ok(),
        "pad1d should handle non-contiguous input, got: {:?}",
        result.err()
    );
    let out = result.unwrap();
    assert_eq!(out.dims(), &[3, 4]); // padded last dim: 2 + 1 + 1 = 4
                                     // Verify values: each row is [0, original..., 0]
    let vals = out.to_flat_vec::<f32>().unwrap();
    assert_eq!(
        vals,
        vec![0.0, 1.0, 4.0, 0.0, 0.0, 2.0, 5.0, 0.0, 0.0, 3.0, 6.0, 0.0]
    );
}

// -- Pad1d rank-0 guard ---------------------------------------------------

#[test]
fn test_pad1d_rank0_returns_error() {
    let x = DynTensor::from_vec(vec![1.0], &[], &cpu()).unwrap();
    let err = x.pad1d(1, 1).unwrap_err();
    assert!(
        matches!(
            err,
            TensorError::RankMismatch {
                expected: 1,
                actual: 0
            }
        ),
        "expected RankMismatch for rank-0 pad1d, got: {err}"
    );
}

// -- pad_with_zeros tests (candle compat, dvoice#47) ----------------------

#[test]
fn test_pad_with_zeros_left_only() {
    // Causal left-pad: [B, C, T] → [B, C, left + T] — matches dvoice Qwen3 conv usage
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3], &cpu()).unwrap();
    let y = x.pad_with_zeros(2, 2, 0).unwrap();
    assert_eq!(y.dims(), &[1, 2, 5]);
    let v = y.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_pad_with_zeros_symmetric() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let y = x.pad_with_zeros(0, 1, 1).unwrap();
    assert_eq!(y.dims(), &[5]);
    assert_eq!(
        y.to_flat_vec::<f32>().unwrap(),
        vec![0.0, 1.0, 2.0, 3.0, 0.0]
    );
}

#[test]
fn test_pad_with_zeros_dim_out_of_range() {
    let x = DynTensor::new(&[1.0, 2.0], &[2], &cpu()).unwrap();
    assert!(x.pad_with_zeros(1, 1, 0).is_err());
}

#[test]
fn test_pad_with_zeros_noop() {
    let x = DynTensor::new(&[1.0, 2.0], &[2], &cpu()).unwrap();
    let y = x.pad_with_zeros(0, 0, 0).unwrap();
    assert_eq!(y.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0]);
}
