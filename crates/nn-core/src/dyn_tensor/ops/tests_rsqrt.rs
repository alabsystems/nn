// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `rsqrt` operation.

use crate::dyn_tensor::DynTensor;
use crate::DType;

#[test]
fn test_rsqrt_basic() {
    let x = DynTensor::from_vec(vec![1.0, 4.0, 9.0, 16.0], &[4], &crate::Device::Cpu).unwrap();
    let result = x.rsqrt().unwrap();
    let vals = result.to_vec1::<f32>().unwrap();
    assert!((vals[0] - 1.0).abs() < 1e-6);
    assert!((vals[1] - 0.5).abs() < 1e-6);
    assert!((vals[2] - 1.0 / 3.0).abs() < 1e-6);
    assert!((vals[3] - 0.25).abs() < 1e-6);
}

#[test]
fn test_rsqrt_preserves_shape() {
    let x = DynTensor::from_vec(
        vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0],
        &[2, 3],
        &crate::Device::Cpu,
    )
    .unwrap();
    let result = x.rsqrt().unwrap();
    assert_eq!(result.dims(), &[2, 3]);
    assert_eq!(result.dtype(), DType::F32);
}

#[test]
fn test_rsqrt_zero_input_errors() {
    // rsqrt(0) = 1/sqrt(0) = 1/0 = Inf, which should error on CPU
    let x = DynTensor::from_vec(vec![0.0], &[1], &crate::Device::Cpu).unwrap();
    let result = x.rsqrt();
    assert!(result.is_err());
}
