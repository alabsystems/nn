#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! `to_dtype` conversion tests including boundary conditions for F32↔I64, F32↔U32.
//!
//! Extracted from `tests_extended.rs` for file-size compliance (#1227).

use crate::dyn_tensor::test_helpers::cpu;
use crate::{DType, DynTensor};

// -- to_dtype Tests -----------------------------------------------------------

#[test]
fn test_to_dtype_same_dtype_passthrough() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let t2 = t.to_dtype(DType::F32).unwrap();
    assert_eq!(t2.dtype(), DType::F32);
    assert_eq!(t2.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_to_dtype_f32_to_bf16_lossy() {
    let t = DynTensor::new(&[1.0, 0.5, -2.75], &[3], &cpu()).unwrap();
    let t2 = t.to_dtype(DType::BF16).unwrap();
    // to_dtype(BF16) creates native bf16 storage.
    assert_eq!(t2.dtype(), DType::BF16);
    let v = t2.to_flat_vec::<f32>().unwrap();
    assert!((v[0] - 1.0).abs() < 0.01);
    assert!((v[1] - 0.5).abs() < 0.01);
    assert!((v[2] - (-2.75)).abs() < 0.05); // BF16 has limited precision
}

#[test]
fn test_to_dtype_f32_to_f16_lossy() {
    let t = DynTensor::new(&[1.0, 0.5, -2.75], &[3], &cpu()).unwrap();
    let t2 = t.to_dtype(DType::F16).unwrap();
    // to_dtype(F16) creates native f16 storage.
    assert_eq!(t2.dtype(), DType::F16);
    let v = t2.to_flat_vec::<f32>().unwrap();
    assert!((v[0] - 1.0).abs() < 0.001);
    assert!((v[1] - 0.5).abs() < 0.001);
    assert!((v[2] - (-2.75)).abs() < 0.01);
}

#[test]
fn test_to_dtype_bf16_to_f32() {
    let t = DynTensor::new(&[1.0, 2.0], &[2], &cpu()).unwrap();
    let bf = t.to_dtype(DType::BF16).unwrap();
    // bf has native bf16 storage; convert back to f32.
    assert_eq!(bf.dtype(), DType::BF16);
    let f32_back = bf.to_dtype(DType::F32).unwrap();
    assert_eq!(f32_back.dtype(), DType::F32);
    let v = f32_back.to_flat_vec::<f32>().unwrap();
    assert!((v[0] - 1.0).abs() < 0.01);
    assert!((v[1] - 2.0).abs() < 0.01);
}

#[test]
fn test_to_dtype_u32_to_f32_roundtrip() {
    let t = DynTensor::from_vec_u32(vec![1, 2, 3], &[3], &cpu()).unwrap();
    let f = t.to_dtype(DType::F32).unwrap();
    assert_eq!(f.dtype(), DType::F32);
    assert_eq!(f.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_to_dtype_f32_to_u32_truncates() {
    let t = DynTensor::new(&[1.5, 2.9], &[2], &cpu()).unwrap();
    let u = t.to_dtype(DType::U32).unwrap();
    assert_eq!(u.dtype(), DType::U32);
    let vals = u.as_cpu_u32().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[1, 2]);
}

#[test]
fn test_to_dtype_integer_to_bf16_via_f32() {
    // U32→BF16 goes through f32 intermediate (values must be ≤ 2^24 for U32→F32).
    let t = DynTensor::from_vec_u32(vec![1, 2, 3], &[3], &cpu()).unwrap();
    let bf = t.to_dtype(DType::BF16).unwrap();
    assert_eq!(bf.dtype(), DType::BF16);
    let v = bf.to_flat_vec::<f32>().unwrap();
    assert!((v[0] - 1.0).abs() < 0.01);
    assert!((v[1] - 2.0).abs() < 0.01);
    assert!((v[2] - 3.0).abs() < 0.01);
}

// -- Boundary condition regression tests (P1-140) -----------------------------

#[test]
fn test_to_dtype_f32_to_i64_large_positive_returns_error() {
    // 1e30 is finite but exceeds i64::MAX (~9.2e18). Without range check,
    // `1e30_f32 as i64` silently saturates to i64::MAX.
    let t = DynTensor::new(&[1e30], &[1], &cpu()).unwrap();
    let err = t.to_dtype(DType::I64);
    assert!(
        err.is_err(),
        "F32→I64 should reject values exceeding i64 range"
    );
}

#[test]
fn test_to_dtype_f32_to_i64_large_negative_returns_error() {
    let t = DynTensor::new(&[-1e30], &[1], &cpu()).unwrap();
    let err = t.to_dtype(DType::I64);
    assert!(
        err.is_err(),
        "F32→I64 should reject negative values exceeding i64 range"
    );
}

#[test]
fn test_to_dtype_f32_to_i64_within_range_succeeds() {
    // 1e9 is well within i64 range
    let t = DynTensor::new(&[1e9, -1e9, 0.0], &[3], &cpu()).unwrap();
    let r = t.to_dtype(DType::I64).unwrap();
    assert_eq!(r.dtype(), DType::I64);
}

#[test]
fn test_to_dtype_f32_to_u32_boundary_rejects_2_pow_32() {
    // 4_294_967_296.0 (2^32) is the f32 above u32::MAX.
    // The old guard `v > f64::from(u32::MAX) as f32` equals `v > 4294967296.0`
    // which lets this value through; the `as u32` then saturates.
    let t = DynTensor::new(&[4_294_967_296.0_f32], &[1], &cpu()).unwrap();
    let err = t.to_dtype(DType::U32);
    assert!(
        err.is_err(),
        "F32→U32 should reject 2^32 (not representable as u32)"
    );
}

#[test]
fn test_to_dtype_f32_to_u32_max_safe_accepted() {
    // 4_294_967_040.0 = f32::from_bits(0x4F7FFFFF) is the largest f32 that
    // safely converts to u32 (int value 4_294_967_040, below u32::MAX = 4_294_967_295).
    // The previous constant (4_294_966_784.0 = 0x4F7FFFFE) was one ULP too conservative.
    let max_safe = f32::from_bits(0x4F7F_FFFF); // 4_294_967_040.0
    let t = DynTensor::new(&[max_safe], &[1], &cpu()).unwrap();
    let r = t.to_dtype(DType::U32).unwrap();
    assert_eq!(r.to_vec1::<u32>().unwrap(), vec![4_294_967_040]);
}

#[test]
fn test_to_dtype_f32_to_u32_one_ulp_above_max_safe_rejected() {
    // f32::from_bits(0x4F800000) = 4_294_967_296.0 = 2^32, which exceeds u32::MAX.
    let one_above = f32::from_bits(0x4F80_0000);
    assert_eq!(one_above, 4_294_967_296.0);
    let t = DynTensor::new(&[one_above], &[1], &cpu()).unwrap();
    assert!(
        t.to_dtype(DType::U32).is_err(),
        "F32→U32 should reject 2^32"
    );
}

// -- I64 boundary-condition tests (P1-140) ------------------------------------

#[test]
fn test_to_dtype_f32_to_i64_nan_rejects() {
    let t = DynTensor::new(&[f32::NAN], &[1], &cpu()).unwrap();
    assert!(t.to_dtype(DType::I64).is_err(), "F32→I64 should reject NaN");
}

#[test]
fn test_to_dtype_f32_to_i64_inf_rejects() {
    let t = DynTensor::new(&[f32::INFINITY], &[1], &cpu()).unwrap();
    assert!(
        t.to_dtype(DType::I64).is_err(),
        "F32→I64 should reject +Inf"
    );
    let t_neg = DynTensor::new(&[f32::NEG_INFINITY], &[1], &cpu()).unwrap();
    assert!(
        t_neg.to_dtype(DType::I64).is_err(),
        "F32→I64 should reject -Inf"
    );
}

#[test]
fn test_to_dtype_f32_to_i64_exact_boundary_accepted() {
    // MAX_F32_FOR_I64 rounds to 9_223_371_487_098_961_920.0 (0x5EFFFFFF).
    // This is the largest f32 below 2^63 and must be accepted.
    let boundary = 9_223_371_487_098_961_920.0_f32;
    let t = DynTensor::new(&[boundary], &[1], &cpu()).unwrap();
    let r = t.to_dtype(DType::I64);
    assert!(
        r.is_ok(),
        "F32→I64 should accept the largest f32 below 2^63"
    );
}

#[test]
fn test_to_dtype_f32_to_i64_one_ulp_above_boundary_rejected() {
    // 2^63 = 9_223_372_036_854_775_808.0 (0x5F000000) exceeds i64::MAX.
    let above = f32::from_bits(0x5F000000);
    let t = DynTensor::new(&[above], &[1], &cpu()).unwrap();
    assert!(
        t.to_dtype(DType::I64).is_err(),
        "F32→I64 should reject 2^63"
    );
}

#[test]
fn test_to_dtype_f32_to_i64_min_boundary_accepted() {
    // i64::MIN = -2^63 is exactly representable as f32 and must be accepted.
    let min_val = -9_223_372_036_854_775_808.0_f32; // -2^63
    let t = DynTensor::new(&[min_val], &[1], &cpu()).unwrap();
    let r = t.to_dtype(DType::I64);
    assert!(r.is_ok(), "F32→I64 should accept -2^63 (exactly i64::MIN)");
}

#[test]
fn test_to_dtype_i64_to_u32_at_max() {
    // u32::MAX = 4_294_967_295 as i64 should convert successfully.
    let t = DynTensor::from_vec_i64(vec![i64::from(u32::MAX)], &[1], &cpu()).unwrap();
    let r = t.to_dtype(DType::U32).unwrap();
    assert_eq!(r.to_vec1::<u32>().unwrap(), vec![u32::MAX]);
}

#[test]
fn test_to_dtype_i64_to_u32_above_max_rejects() {
    // u32::MAX + 1 should be rejected.
    let t = DynTensor::from_vec_i64(vec![i64::from(u32::MAX) + 1], &[1], &cpu()).unwrap();
    assert!(
        t.to_dtype(DType::U32).is_err(),
        "I64→U32 should reject u32::MAX + 1"
    );
}

#[test]
fn test_to_dtype_i64_to_f32_large_value_rejected() {
    // Values with |v| > 2^24 cannot be exactly represented as f32.
    // The precision guard rejects these to prevent silent precision loss
    // in index/ID pipelines (#1577).
    let val: i64 = (1_i64 << 53) + 1;
    let t = DynTensor::from_vec_i64(vec![val], &[1], &cpu()).unwrap();
    assert!(
        t.to_dtype(DType::F32).is_err(),
        "I64→F32 should reject values exceeding f32 exact integer limit (2^24)"
    );
}

#[test]
fn test_to_dtype_i64_to_f32_within_exact_range_succeeds() {
    // Values within ±2^24 are exactly representable as f32.
    let t = DynTensor::from_vec_i64(vec![0, 1, -1, 16_777_216, -16_777_216], &[5], &cpu()).unwrap();
    let f = t.to_dtype(DType::F32).unwrap();
    let v = f.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![0.0, 1.0, -1.0, 16_777_216.0, -16_777_216.0]);
}

#[test]
fn test_to_dtype_i64_to_f32_one_above_limit_rejected() {
    // 2^24 + 1 = 16_777_217 cannot be exactly represented as f32.
    let t = DynTensor::from_vec_i64(vec![16_777_217], &[1], &cpu()).unwrap();
    assert!(
        t.to_dtype(DType::F32).is_err(),
        "I64→F32 should reject 2^24 + 1"
    );
}

#[test]
fn test_to_dtype_u32_to_f32_large_value_rejected() {
    // U32 values > 2^24 cannot be exactly represented as f32.
    let t = DynTensor::from_vec_u32(vec![16_777_217], &[1], &cpu()).unwrap();
    assert!(
        t.to_dtype(DType::F32).is_err(),
        "U32→F32 should reject 2^24 + 1"
    );
}

#[test]
fn test_to_dtype_u32_to_f32_at_limit_succeeds() {
    // 2^24 = 16_777_216 is exactly representable as f32.
    let t = DynTensor::from_vec_u32(vec![0, 1, 16_777_216], &[3], &cpu()).unwrap();
    let f = t.to_dtype(DType::F32).unwrap();
    let v = f.to_flat_vec::<f32>().unwrap();
    assert_eq!(v, vec![0.0, 1.0, 16_777_216.0]);
}
