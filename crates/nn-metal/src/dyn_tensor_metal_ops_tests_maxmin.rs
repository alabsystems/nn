#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU maximum/minimum tests — BinaryOp::Maximum and BinaryOp::Minimum dispatch.
//!
//! Covers element-wise, NaN propagation, infinity edge cases, large values,
//! negative-negative, higher-rank, and scalar tensors.
//!
//! Extracted from `dyn_tensor_metal_ops_tests.rs` for 500-line compliance (#1361).

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::{assert_gpu_vals, init};

// -- BinaryOp::Maximum / BinaryOp::Minimum GPU parity tests -------------------
#[test]
fn test_gpu_maximum_elementwise() {
    init();
    let a = DynTensor::new(&[1.0, 5.0, 3.0, -2.0], &[2, 2], &Device::metal()).unwrap();
    let b = DynTensor::new(&[4.0, 2.0, 3.0, 0.0], &[2, 2], &Device::metal()).unwrap();
    let r = a.maximum(&b).unwrap();
    assert_eq!(r.device(), Device::metal(), "maximum must stay on GPU");
    assert_gpu_vals(&r, &[4.0, 5.0, 3.0, 0.0], 1e-6, "maximum");
}
#[test]
fn test_gpu_minimum_elementwise() {
    init();
    let a = DynTensor::new(&[1.0, 5.0, 3.0, -2.0], &[2, 2], &Device::metal()).unwrap();
    let b = DynTensor::new(&[4.0, 2.0, 3.0, 0.0], &[2, 2], &Device::metal()).unwrap();
    let r = a.minimum(&b).unwrap();
    assert_eq!(r.device(), Device::metal(), "minimum must stay on GPU");
    assert_gpu_vals(&r, &[1.0, 2.0, 3.0, -2.0], 1e-6, "minimum");
}

// -- maximum/minimum: NaN, large values, edge cases (#1321) ------------------

/// NaN in first operand: GPU Compare+Select returns b (NaN > x is false).
/// This matches CPU f32::max() which returns the non-NaN operand.
#[test]
fn test_gpu_maximum_nan_in_first() {
    init();
    let a = DynTensor::new(&[f32::NAN, 3.0], &[2], &Device::metal()).unwrap();
    let b = DynTensor::new(&[5.0, 1.0], &[2], &Device::metal()).unwrap();
    let r = a.maximum(&b).unwrap();
    let vals = r.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(vals[0], 5.0, "max(NaN, 5.0) should be 5.0");
    assert_eq!(vals[1], 3.0, "max(3.0, 1.0) should be 3.0");
}

/// NaN in second operand: GPU Compare+Select returns NaN (x > NaN is false → returns b=NaN).
/// NOTE: This diverges from CPU f32::max() which returns the non-NaN operand.
/// The GPU behavior is documented — NY weight splitting must sanitize NaN weights.
#[test]
fn test_gpu_maximum_nan_in_second() {
    init();
    let a = DynTensor::new(&[5.0, 3.0], &[2], &Device::metal()).unwrap();
    let b = DynTensor::new(&[f32::NAN, 1.0], &[2], &Device::metal()).unwrap();
    let r = a.maximum(&b).unwrap();
    let vals = r.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    // GPU: 5.0 > NaN is false → returns NaN (divergent from CPU's 5.0)
    assert!(
        vals[0].is_nan(),
        "GPU max(5.0, NaN) returns NaN (Compare+Select)"
    );
    assert_eq!(vals[1], 3.0, "max(3.0, 1.0) should be 3.0");
}

/// NaN in first operand for minimum: GPU returns b (NaN < x is false → returns b=x).
#[test]
fn test_gpu_minimum_nan_in_first() {
    init();
    let a = DynTensor::new(&[f32::NAN, 3.0], &[2], &Device::metal()).unwrap();
    let b = DynTensor::new(&[5.0, 1.0], &[2], &Device::metal()).unwrap();
    let r = a.minimum(&b).unwrap();
    let vals = r.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(
        vals[0], 5.0,
        "min(NaN, 5.0) should be 5.0 (NaN < x is false → returns b)"
    );
    assert_eq!(vals[1], 1.0, "min(3.0, 1.0) should be 1.0");
}

/// Large magnitude values near f32 boundary.
#[test]
fn test_gpu_maximum_large_values() {
    init();
    let a = DynTensor::new(&[f32::MAX, f32::MIN, -1e38, 0.0], &[4], &Device::metal()).unwrap();
    let b = DynTensor::new(&[0.0, 0.0, 1e38, f32::MIN], &[4], &Device::metal()).unwrap();
    let r = a.maximum(&b).unwrap();
    assert_gpu_vals(&r, &[f32::MAX, 0.0, 1e38, 0.0], 1e-6, "max_large");
}

#[test]
fn test_gpu_minimum_large_values() {
    init();
    let a = DynTensor::new(&[f32::MAX, f32::MIN, -1e38, 0.0], &[4], &Device::metal()).unwrap();
    let b = DynTensor::new(&[0.0, 0.0, 1e38, f32::MIN], &[4], &Device::metal()).unwrap();
    let r = a.minimum(&b).unwrap();
    assert_gpu_vals(&r, &[0.0, f32::MIN, -1e38, f32::MIN], 1e-6, "min_large");
}

/// Both inputs negative — common in NY W_neg = min(W, 0).
#[test]
fn test_gpu_maximum_negative_negative() {
    init();
    let a = DynTensor::new(&[-5.0, -1.0, -100.0, -0.01], &[2, 2], &Device::metal()).unwrap();
    let b = DynTensor::new(&[-3.0, -10.0, -50.0, -0.02], &[2, 2], &Device::metal()).unwrap();
    let r = a.maximum(&b).unwrap();
    assert_gpu_vals(&r, &[-3.0, -1.0, -50.0, -0.01], 1e-6, "max_neg_neg");
}

#[test]
fn test_gpu_minimum_negative_negative() {
    init();
    let a = DynTensor::new(&[-5.0, -1.0, -100.0, -0.01], &[2, 2], &Device::metal()).unwrap();
    let b = DynTensor::new(&[-3.0, -10.0, -50.0, -0.02], &[2, 2], &Device::metal()).unwrap();
    let r = a.minimum(&b).unwrap();
    assert_gpu_vals(&r, &[-5.0, -10.0, -100.0, -0.02], 1e-6, "min_neg_neg");
}

/// Higher-dimensional tensor: 3D [2,2,3].
#[test]
fn test_gpu_maximum_3d() {
    init();
    let a_data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..12).rev().map(|i| i as f32).collect();
    let a = DynTensor::from_vec(a_data, &[2, 2, 3], &Device::metal()).unwrap();
    let b = DynTensor::from_vec(b_data, &[2, 2, 3], &Device::metal()).unwrap();
    let r = a.maximum(&b).unwrap();
    assert_eq!(r.dims(), &[2, 2, 3]);
    // max(0,11)=11, max(1,10)=10, ..., max(5,6)=6, max(6,5)=6, ..., max(11,0)=11
    let expected: Vec<f32> = (0..12).map(|i| (i as f32).max(11.0 - i as f32)).collect();
    assert_gpu_vals(&r, &expected, 1e-6, "max_3d");
}

/// Single-element tensors.
#[test]
fn test_gpu_maximum_scalar() {
    init();
    let a = DynTensor::new(&[3.0], &[1], &Device::metal()).unwrap();
    let b = DynTensor::new(&[7.0], &[1], &Device::metal()).unwrap();
    let r = a.maximum(&b).unwrap();
    assert_gpu_vals(&r, &[7.0], 1e-6, "max_scalar");
}

#[test]
fn test_gpu_minimum_scalar() {
    init();
    let a = DynTensor::new(&[3.0], &[1], &Device::metal()).unwrap();
    let b = DynTensor::new(&[7.0], &[1], &Device::metal()).unwrap();
    let r = a.minimum(&b).unwrap();
    assert_gpu_vals(&r, &[3.0], 1e-6, "min_scalar");
}

/// Minimum with NaN in second operand: GPU Compare+Select returns first (non-NaN).
/// Mirrors test_gpu_maximum_nan_in_second — minimum uses CompareOpKind::Lt.
#[test]
fn test_gpu_minimum_nan_in_second() {
    init();
    let a = DynTensor::new(&[5.0, 3.0], &[2], &Device::metal()).unwrap();
    let b = DynTensor::new(&[f32::NAN, 1.0], &[2], &Device::metal()).unwrap();
    let r = a.minimum(&b).unwrap();
    let vals = r.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    // GPU: a[0] < NaN is false → select returns b = NaN (Compare+Select semantics).
    assert!(
        vals[0].is_nan(),
        "min(5.0, NaN) should be NaN on GPU: got {}",
        vals[0]
    );
    assert_eq!(vals[1], 1.0, "min(3.0, 1.0) should be 1.0");
}

/// Both operands NaN: GPU output is NaN for both max and min.
#[test]
fn test_gpu_maximum_both_nan() {
    init();
    let a = DynTensor::new(&[f32::NAN], &[1], &Device::metal()).unwrap();
    let b = DynTensor::new(&[f32::NAN], &[1], &Device::metal()).unwrap();
    let r = a.maximum(&b).unwrap();
    let vals = r.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    assert!(
        vals[0].is_nan(),
        "max(NaN, NaN) should be NaN: got {}",
        vals[0]
    );
}

#[test]
fn test_gpu_minimum_both_nan() {
    init();
    let a = DynTensor::new(&[f32::NAN], &[1], &Device::metal()).unwrap();
    let b = DynTensor::new(&[f32::NAN], &[1], &Device::metal()).unwrap();
    let r = a.minimum(&b).unwrap();
    let vals = r.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    assert!(
        vals[0].is_nan(),
        "min(NaN, NaN) should be NaN: got {}",
        vals[0]
    );
}

/// Infinity edge cases for max/min GPU dispatch.
/// Manual assertions because assert_gpu_vals computes (inf - inf) = NaN.
#[test]
fn test_gpu_maximum_infinity() {
    init();
    let a = DynTensor::new(
        &[f32::INFINITY, f32::NEG_INFINITY, 0.0],
        &[3],
        &Device::metal(),
    )
    .unwrap();
    let b = DynTensor::new(
        &[f32::NEG_INFINITY, 0.0, f32::INFINITY],
        &[3],
        &Device::metal(),
    )
    .unwrap();
    let r = a.maximum(&b).unwrap();
    let vals = r.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(vals[0], f32::INFINITY, "max(+inf, -inf) should be +inf");
    assert_eq!(vals[1], 0.0, "max(-inf, 0) should be 0");
    assert_eq!(vals[2], f32::INFINITY, "max(0, +inf) should be +inf");
}

#[test]
fn test_gpu_minimum_infinity() {
    init();
    let a = DynTensor::new(
        &[f32::INFINITY, f32::NEG_INFINITY, 0.0],
        &[3],
        &Device::metal(),
    )
    .unwrap();
    let b = DynTensor::new(
        &[f32::NEG_INFINITY, 0.0, f32::INFINITY],
        &[3],
        &Device::metal(),
    )
    .unwrap();
    let r = a.minimum(&b).unwrap();
    let vals = r.to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(vals[0], f32::NEG_INFINITY, "min(+inf, -inf) should be -inf");
    assert_eq!(vals[1], f32::NEG_INFINITY, "min(-inf, 0) should be -inf");
    assert_eq!(vals[2], 0.0, "min(0, +inf) should be 0");
}
