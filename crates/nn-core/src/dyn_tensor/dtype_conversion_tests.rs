// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Dtype conversion edge-case tests for DynTensor::to_dtype().
//!
//! Covers precision loss, truncation, range limits, and identity conversions
//! across all supported dtype pairs:
//! - F32 <-> BF16 (precision loss from 23-bit to 7-bit mantissa)
//! - F32 <-> F16 (precision loss + narrower range: max ~65504)
//! - F32 -> U8/U32/I64 (truncation, range validation)
//! - U32 -> F32 (exact for small values, rejected for large)
//! - Identity conversion (same dtype returns clone)
//! - Large tensor conversion

use crate::dyn_tensor::test_helpers::{approx_eq, cpu};
use crate::{DType, DynTensor};

// ============================================================================
// F32 -> BF16 -> F32 round-trip (precision loss)
// ============================================================================

#[test]
fn test_f32_to_bf16_roundtrip_exact_values() {
    // Values exactly representable in bf16 (powers of 2, small integers)
    let t = DynTensor::from_vec(vec![0.0, 1.0, -1.0, 2.0, 0.5, 256.0], &[6], &cpu()).unwrap();
    let bf16 = t.to_dtype(DType::BF16).unwrap();
    assert_eq!(bf16.dtype(), DType::BF16);
    let back = bf16.to_dtype(DType::F32).unwrap();
    assert_eq!(back.dtype(), DType::F32);
    let vals = back.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 1.0, -1.0, 2.0, 0.5, 256.0]);
}

#[test]
fn test_f32_to_bf16_precision_loss() {
    // bf16 has only 7 bits of mantissa (vs 23 for f32).
    // 1.0009765625 (1 + 2^-10) is representable in f32 but rounds in bf16.
    let val = 1.000_976_6_f32;
    let t = DynTensor::from_vec(vec![val], &[1], &cpu()).unwrap();
    let bf16 = t.to_dtype(DType::BF16).unwrap();
    let back = bf16.to_dtype(DType::F32).unwrap();
    let result = back.to_flat_vec::<f32>().unwrap()[0];
    // bf16 rounds 1.0009765625 to 1.0 (7-bit mantissa cannot represent this)
    assert_ne!(result, val, "bf16 should lose precision for this value");
    // But it should be close
    assert!(approx_eq(result, val, 0.01));
}

#[test]
fn test_f32_to_bf16_preserves_sign() {
    let t = DynTensor::from_vec(vec![-3.5, 3.5], &[2], &cpu()).unwrap();
    let bf16 = t.to_dtype(DType::BF16).unwrap();
    let back = bf16.to_dtype(DType::F32).unwrap();
    let vals = back.to_flat_vec::<f32>().unwrap();
    assert!(vals[0] < 0.0, "negative sign preserved");
    assert!(vals[1] > 0.0, "positive sign preserved");
    assert!(approx_eq(vals[0], -3.5, 0.01));
    assert!(approx_eq(vals[1], 3.5, 0.01));
}

// ============================================================================
// F32 -> F16 -> F32 round-trip (precision loss + range)
// ============================================================================

#[test]
fn test_f32_to_f16_roundtrip_exact_values() {
    // Small integers and powers of 2 are exact in f16
    let t = DynTensor::from_vec(vec![0.0, 1.0, -1.0, 2.0, 0.5, 1024.0], &[6], &cpu()).unwrap();
    let f16 = t.to_dtype(DType::F16).unwrap();
    assert_eq!(f16.dtype(), DType::F16);
    let back = f16.to_dtype(DType::F32).unwrap();
    let vals = back.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 1.0, -1.0, 2.0, 0.5, 1024.0]);
}

#[test]
fn test_f32_to_f16_precision_loss() {
    // f16 has 10 bits of mantissa. Values needing more precision will round.
    let val = 1.000_488_3_f32; // 1 + 2^-11, needs 11 mantissa bits
    let t = DynTensor::from_vec(vec![val], &[1], &cpu()).unwrap();
    let f16 = t.to_dtype(DType::F16).unwrap();
    let back = f16.to_dtype(DType::F32).unwrap();
    let result = back.to_flat_vec::<f32>().unwrap()[0];
    // f16 can represent 1 + 2^-10 = 1.0009765625 but not 1 + 2^-11
    assert_ne!(result, val, "f16 should lose precision for this value");
    assert!(approx_eq(result, val, 0.001));
}

#[test]
fn test_f32_to_f16_range_clamp() {
    // f16 max is ~65504. Larger values become inf in f16.
    // Values within range should survive round-trip.
    let t = DynTensor::from_vec(vec![100.0, 1000.0, 60000.0], &[3], &cpu()).unwrap();
    let f16 = t.to_dtype(DType::F16).unwrap();
    let back = f16.to_dtype(DType::F32).unwrap();
    let vals = back.to_flat_vec::<f32>().unwrap();
    assert!(approx_eq(vals[0], 100.0, 0.1));
    assert!(approx_eq(vals[1], 1000.0, 1.0));
    assert!(approx_eq(vals[2], 60000.0, 32.0)); // f16 precision at 60000 is ~32
}

// ============================================================================
// F32 -> U8 (truncation, range)
// ============================================================================

#[test]
fn test_f32_to_u8_valid_range() {
    let t = DynTensor::from_vec(vec![0.0, 1.0, 127.0, 255.0], &[4], &cpu()).unwrap();
    let u8t = t.to_dtype(DType::U8).unwrap();
    assert_eq!(u8t.dtype(), DType::U8);
    let vals = u8t.to_flat_vec::<u8>().unwrap();
    assert_eq!(vals, vec![0, 1, 127, 255]);
}

#[test]
fn test_f32_to_u8_negative_fails() {
    let t = DynTensor::from_vec(vec![-1.0], &[1], &cpu()).unwrap();
    assert!(t.to_dtype(DType::U8).is_err());
}

#[test]
fn test_f32_to_u8_overflow_fails() {
    let t = DynTensor::from_vec(vec![256.0], &[1], &cpu()).unwrap();
    assert!(t.to_dtype(DType::U8).is_err());
}

// ============================================================================
// F32 -> U32 (truncation, range)
// ============================================================================

#[test]
fn test_f32_to_u32_valid_range() {
    let t = DynTensor::from_vec(vec![0.0, 1.0, 1000.0], &[3], &cpu()).unwrap();
    let u32t = t.to_dtype(DType::U32).unwrap();
    assert_eq!(u32t.dtype(), DType::U32);
    let vals = u32t.to_flat_vec::<u32>().unwrap();
    assert_eq!(vals, vec![0, 1, 1000]);
}

#[test]
fn test_f32_to_u32_negative_fails() {
    let t = DynTensor::from_vec(vec![-1.0], &[1], &cpu()).unwrap();
    assert!(t.to_dtype(DType::U32).is_err());
}

// ============================================================================
// F32 -> I64 (truncation)
// ============================================================================

#[test]
fn test_f32_to_i64_valid_range() {
    let t = DynTensor::from_vec(vec![-100.0, 0.0, 42.0, 1000000.0], &[4], &cpu()).unwrap();
    let i64t = t.to_dtype(DType::I64).unwrap();
    assert_eq!(i64t.dtype(), DType::I64);
    let vals = i64t.to_flat_vec::<i64>().unwrap();
    assert_eq!(vals, vec![-100, 0, 42, 1000000]);
}

#[test]
fn test_f32_to_i64_nan_fails() {
    let t = DynTensor::from_vec(vec![f32::NAN], &[1], &cpu()).unwrap();
    assert!(t.to_dtype(DType::I64).is_err());
}

#[test]
fn test_f32_to_i64_inf_fails() {
    let t = DynTensor::from_vec(vec![f32::INFINITY], &[1], &cpu()).unwrap();
    assert!(t.to_dtype(DType::I64).is_err());
}

// ============================================================================
// U32 -> F32 (exact for small values)
// ============================================================================

#[test]
fn test_u32_to_f32_small_values_exact() {
    // All integers up to 2^24 (16,777,216) are exactly representable in f32
    let t = DynTensor::from_vec_u32(vec![0, 1, 100, 16_777_216], &[4], &cpu()).unwrap();
    let f32t = t.to_dtype(DType::F32).unwrap();
    assert_eq!(f32t.dtype(), DType::F32);
    let vals = f32t.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 1.0, 100.0, 16_777_216.0]);
}

#[test]
fn test_u32_to_f32_large_value_rejected() {
    // Values > 2^24 cannot be exactly represented in f32
    let t = DynTensor::from_vec_u32(vec![16_777_217], &[1], &cpu()).unwrap();
    assert!(
        t.to_dtype(DType::F32).is_err(),
        "u32 > 2^24 should fail precision guard"
    );
}

// ============================================================================
// Identity conversion (same dtype)
// ============================================================================

#[test]
fn test_identity_conversion_f32() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &cpu()).unwrap();
    let same = t.to_dtype(DType::F32).unwrap();
    assert_eq!(same.dtype(), DType::F32);
    assert_eq!(same.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_identity_conversion_u32() {
    let t = DynTensor::from_vec_u32(vec![10, 20, 30], &[3], &cpu()).unwrap();
    let same = t.to_dtype(DType::U32).unwrap();
    assert_eq!(same.dtype(), DType::U32);
    assert_eq!(same.to_flat_vec::<u32>().unwrap(), vec![10, 20, 30]);
}

#[test]
fn test_identity_conversion_bf16() {
    let t = DynTensor::full(&[3], 1.0, DType::BF16, &cpu()).unwrap();
    let same = t.to_dtype(DType::BF16).unwrap();
    assert_eq!(same.dtype(), DType::BF16);
}

#[test]
fn test_identity_conversion_i64() {
    let t = DynTensor::from_vec_i64(vec![-5, 0, 5], &[3], &cpu()).unwrap();
    let same = t.to_dtype(DType::I64).unwrap();
    assert_eq!(same.dtype(), DType::I64);
    assert_eq!(same.to_flat_vec::<i64>().unwrap(), vec![-5, 0, 5]);
}

// ============================================================================
// Large tensor conversion (ensures no per-element overhead issues)
// ============================================================================

#[test]
fn test_large_tensor_f32_to_bf16() {
    let n = 10_000;
    let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let t = DynTensor::from_vec(data, &[n], &cpu()).unwrap();
    let bf16 = t.to_dtype(DType::BF16).unwrap();
    assert_eq!(bf16.dtype(), DType::BF16);
    assert_eq!(bf16.dims(), &[n]);
    // Round-trip back and check first/last elements are close
    let back = bf16.to_dtype(DType::F32).unwrap();
    let vals = back.to_flat_vec::<f32>().unwrap();
    assert!(approx_eq(vals[0], 0.0, 0.01));
    assert!(approx_eq(vals[n - 1], (n - 1) as f32 * 0.01, 0.5));
}

#[test]
fn test_large_tensor_f32_to_f16() {
    let n = 10_000;
    // Keep values in f16 range (max ~65504)
    let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
    let t = DynTensor::from_vec(data, &[n], &cpu()).unwrap();
    let f16 = t.to_dtype(DType::F16).unwrap();
    assert_eq!(f16.dtype(), DType::F16);
    assert_eq!(f16.dims(), &[n]);
    let back = f16.to_dtype(DType::F32).unwrap();
    let vals = back.to_flat_vec::<f32>().unwrap();
    assert!(approx_eq(vals[0], 0.0, 0.001));
    assert!(approx_eq(vals[n - 1], (n - 1) as f32 * 0.001, 0.01));
}

// ============================================================================
// Cross-type chains (bf16 -> i64 goes through f32 intermediate)
// ============================================================================

#[test]
fn test_bf16_to_i64_via_f32_intermediate() {
    let t = DynTensor::full(&[3], 42.0, DType::BF16, &cpu()).unwrap();
    let i64t = t.to_dtype(DType::I64).unwrap();
    assert_eq!(i64t.dtype(), DType::I64);
    let vals = i64t.to_flat_vec::<i64>().unwrap();
    assert_eq!(vals, vec![42, 42, 42]);
}

#[test]
fn test_i64_to_bf16_via_f32_intermediate() {
    let t = DynTensor::from_vec_i64(vec![1, 2, 3], &[3], &cpu()).unwrap();
    let bf16t = t.to_dtype(DType::BF16).unwrap();
    assert_eq!(bf16t.dtype(), DType::BF16);
    // Round-trip to verify values
    let back = bf16t.to_dtype(DType::F32).unwrap();
    let vals = back.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0]);
}

// ============================================================================
// Shape preservation across conversions
// ============================================================================

#[test]
fn test_to_dtype_preserves_shape_2d() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let bf16 = t.to_dtype(DType::BF16).unwrap();
    assert_eq!(bf16.dims(), &[2, 3]);
    let f16 = t.to_dtype(DType::F16).unwrap();
    assert_eq!(f16.dims(), &[2, 3]);
}

#[test]
fn test_to_dtype_preserves_shape_3d() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = DynTensor::from_vec(data, &[2, 3, 4], &cpu()).unwrap();
    let bf16 = t.to_dtype(DType::BF16).unwrap();
    assert_eq!(bf16.dims(), &[2, 3, 4]);
    assert_eq!(bf16.numel(), 24);
}
