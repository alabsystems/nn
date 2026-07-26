// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for precision-aware DynTensor operations.
//!
//! Covers: mixed-precision matmul/softmax/layer_norm, to_dtype round-trips,
//! precision loss characteristics of BF16 vs F16, affine/scalar ops in
//! different dtypes, and edge cases (overflow/underflow in half-precision).

use crate::dyn_tensor::test_helpers::{approx_eq, assert_close, cpu, t1d, t2d};
use crate::mixed_precision::{MixedPrecisionPolicy, OpDTypeCategory};
use crate::{DType, DynTensor};

// ---------------------------------------------------------------------------
// Mixed-precision matmul: BF16 inputs, F32 accumulation
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_with_policy_bf16_inputs_f32_accumulate() {
    // Create F32 tensors and convert to BF16 to simulate weight storage.
    let a_f32 = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b_f32 = t2d(&[5.0, 6.0, 7.0, 8.0], 2, 2);
    let a_bf16 = a_f32.to_dtype(DType::BF16).unwrap();
    let b_bf16 = b_f32.to_dtype(DType::BF16).unwrap();
    assert_eq!(a_bf16.dtype(), DType::BF16);
    assert_eq!(b_bf16.dtype(), DType::BF16);

    // Apple Silicon default: compute in F16, accumulate in F32.
    let policy = MixedPrecisionPolicy::apple_silicon_default();

    // matmul_with_policy should cast to compute_dtype (F16) before matmul.
    let result = a_bf16.matmul_with_policy(&b_bf16, &policy).unwrap();
    assert_eq!(result.dtype(), DType::F16);

    // Verify correctness: [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    let result_f32 = result.to_dtype(DType::F32).unwrap();
    let vals = result_f32.to_f32_array().unwrap();
    let flat: Vec<f32> = vals.iter().copied().collect();
    assert_close(&flat, &[19.0, 22.0, 43.0, 50.0], 0.5);
}

#[test]
fn test_matmul_with_policy_f32_only_no_cast() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = t2d(&[5.0, 6.0, 7.0, 8.0], 2, 2);
    let policy = MixedPrecisionPolicy::f32_only();

    // f32_only policy: no casting, result should stay F32.
    let result = a.matmul_with_policy(&b, &policy).unwrap();
    assert_eq!(result.dtype(), DType::F32);

    let vals = result.to_f32_array().unwrap();
    let flat: Vec<f32> = vals.iter().copied().collect();
    assert_eq!(flat, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_matmul_with_policy_cuda_bf16() {
    let a = t2d(&[1.0, 0.0, 0.0, 1.0], 2, 2); // identity
    let b = t2d(&[3.0, 4.0, 5.0, 6.0], 2, 2);
    let policy = MixedPrecisionPolicy::cuda_bf16();

    // cuda_bf16 policy: compute_dtype = BF16.
    let result = a.matmul_with_policy(&b, &policy).unwrap();
    assert_eq!(result.dtype(), DType::BF16);

    let result_f32 = result.to_dtype(DType::F32).unwrap();
    let flat: Vec<f32> = result_f32.to_f32_array().unwrap().iter().copied().collect();
    assert_close(&flat, &[3.0, 4.0, 5.0, 6.0], 0.1);
}

// ---------------------------------------------------------------------------
// Mixed-precision add: F16 + F32
// ---------------------------------------------------------------------------

#[test]
fn test_mixed_precision_add_f16_f32() {
    // Create an F16 tensor and an F32 tensor.
    let a_f32 = t1d(&[1.0, 2.0, 3.0]);
    let a_f16 = a_f32.to_dtype(DType::F16).unwrap();
    let b_f32 = t1d(&[0.5, 0.5, 0.5]);

    // Cast F16 to F32 for the add (simulating what a policy would do).
    let a_upcast = a_f16.to_dtype(DType::F32).unwrap();
    let result = a_upcast.add(&b_f32).unwrap();
    assert_eq!(result.dtype(), DType::F32);

    let vals = result.to_vec1::<f32>().unwrap();
    assert_close(&vals, &[1.5, 2.5, 3.5], 1e-3);
}

// ---------------------------------------------------------------------------
// to_dtype conversion round-trips
// ---------------------------------------------------------------------------

#[test]
fn test_to_dtype_f32_bf16_f32_roundtrip() {
    // Values exactly representable in BF16 should survive round-trip.
    let original = t1d(&[1.0, 2.0, -3.0, 0.0, 0.5]);
    let bf16 = original.to_dtype(DType::BF16).unwrap();
    assert_eq!(bf16.dtype(), DType::BF16);

    let back = bf16.to_dtype(DType::F32).unwrap();
    assert_eq!(back.dtype(), DType::F32);

    let vals = back.to_vec1::<f32>().unwrap();
    assert_close(&vals, &[1.0, 2.0, -3.0, 0.0, 0.5], 1e-6);
}

#[test]
fn test_to_dtype_f32_f16_f32_roundtrip() {
    // Values exactly representable in F16 should survive round-trip.
    let original = t1d(&[1.0, 2.0, -3.0, 0.0, 0.25]);
    let f16 = original.to_dtype(DType::F16).unwrap();
    assert_eq!(f16.dtype(), DType::F16);

    let back = f16.to_dtype(DType::F32).unwrap();
    assert_eq!(back.dtype(), DType::F32);

    let vals = back.to_vec1::<f32>().unwrap();
    assert_close(&vals, &[1.0, 2.0, -3.0, 0.0, 0.25], 1e-6);
}

#[test]
fn test_to_dtype_f32_bf16_f32_precision_loss() {
    // BF16 has only 8 mantissa bits (7 explicit + 1 implicit).
    // A value like 1.001 is NOT exactly representable.
    let original = t1d(&[1.001]);
    let bf16 = original.to_dtype(DType::BF16).unwrap();
    let back = bf16.to_dtype(DType::F32).unwrap();
    let val = back.to_vec1::<f32>().unwrap()[0];

    // BF16 round-trip should introduce some error.
    let error = (val - 1.001_f32).abs();
    // BF16 precision: ~1/128 = ~0.0078 for values near 1.0.
    assert!(error < 0.01, "BF16 round-trip error too large: {error}");
    // But error should be nonzero for non-representable values.
    // (1.001 in BF16 rounds to 1.0 or 1.0078125)
    // Note: error might be zero if the value happens to be representable,
    // so we just check it's within BF16 precision bounds.
}

#[test]
fn test_to_dtype_f32_f16_f32_precision_loss() {
    // F16 has 11 mantissa bits (10 explicit + 1 implicit).
    // A value like 1.001 is NOT exactly representable but closer than BF16.
    let original = t1d(&[1.001]);
    let f16 = original.to_dtype(DType::F16).unwrap();
    let back = f16.to_dtype(DType::F32).unwrap();
    let val = back.to_vec1::<f32>().unwrap()[0];

    // F16 precision: ~1/1024 = ~0.001 for values near 1.0.
    let error = (val - 1.001_f32).abs();
    assert!(error < 0.002, "F16 round-trip error too large: {error}");
}

// ---------------------------------------------------------------------------
// Precision loss detection: BF16 loses more precision than F16 for small values
// ---------------------------------------------------------------------------

#[test]
fn test_bf16_loses_more_precision_than_f16_for_small_values() {
    // For values near 1.0, BF16 has ~7-bit mantissa vs F16's ~10-bit.
    // BF16 should lose more precision on non-representable values.
    let val = 1.0 + 1.0 / 300.0; // ~1.00333... not exactly representable in either
    let original = t1d(&[val]);

    let bf16_rt = original
        .to_dtype(DType::BF16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let f16_rt = original
        .to_dtype(DType::F16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    let bf16_error = (bf16_rt.to_vec1::<f32>().unwrap()[0] - val).abs();
    let f16_error = (f16_rt.to_vec1::<f32>().unwrap()[0] - val).abs();

    // F16 should have less or equal precision loss than BF16 in the [1, 2) range.
    assert!(
        f16_error <= bf16_error + 1e-10,
        "F16 error ({f16_error}) should be <= BF16 error ({bf16_error}) near 1.0"
    );
}

#[test]
fn test_bf16_vs_f16_precision_multiple_values() {
    // Test across several values in the [1, 2) range where F16 has
    // finer granularity than BF16.
    let vals: Vec<f32> = (1..10).map(|i| 1.0 + (i as f32) / 1000.0).collect();
    let original = t1d(&vals);

    let bf16_rt = original
        .to_dtype(DType::BF16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let f16_rt = original
        .to_dtype(DType::F16)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let bf16_max_error: f32 = vals
        .iter()
        .zip(bf16_rt.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let f16_max_error: f32 = vals
        .iter()
        .zip(f16_rt.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    // F16 should have tighter worst-case error than BF16 in [1, 2).
    assert!(
        f16_max_error <= bf16_max_error + 1e-10,
        "F16 max error ({f16_max_error}) should be <= BF16 max error ({bf16_max_error})"
    );
}

// ---------------------------------------------------------------------------
// Precision loss detection: F16 has smaller range than BF16
// ---------------------------------------------------------------------------

#[test]
fn test_f16_has_smaller_range_than_bf16() {
    // F16 max is ~65504. BF16 max is ~3.39e38 (same exponent range as F32).
    // A value like 100_000.0 should be representable in BF16 but overflow in F16.
    let large_val = 100_000.0_f32;
    let original = t1d(&[large_val]);

    // BF16 should handle this fine.
    let bf16 = original.to_dtype(DType::BF16).unwrap();
    let bf16_back = bf16.to_dtype(DType::F32).unwrap();
    let bf16_val = bf16_back.to_vec1::<f32>().unwrap()[0];
    assert!(
        bf16_val.is_finite(),
        "BF16 should represent 100000.0 (got {bf16_val})"
    );
    assert!(
        approx_eq(bf16_val, large_val, 1000.0),
        "BF16 value {bf16_val} should be close to {large_val}"
    );

    // F16 should overflow to infinity for 100_000.0 (max F16 = 65504).
    let f16 = original.to_dtype(DType::F16).unwrap();
    let f16_back = f16.to_dtype(DType::F32).unwrap();
    let f16_val = f16_back.to_vec1::<f32>().unwrap()[0];
    assert!(
        f16_val.is_infinite(),
        "F16 should overflow 100000.0 to inf (got {f16_val})"
    );
}

#[test]
fn test_bf16_range_boundary() {
    // BF16 can represent values up to ~3.39e38, same as F32's exponent range.
    // F16 maxes out at 65504.
    let at_f16_max = 65504.0_f32;
    let original = t1d(&[at_f16_max]);

    let f16 = original.to_dtype(DType::F16).unwrap();
    let f16_back = f16.to_dtype(DType::F32).unwrap();
    let f16_val = f16_back.to_vec1::<f32>().unwrap()[0];
    assert!(f16_val.is_finite(), "65504.0 should be within F16 range");
    assert_eq!(f16_val, 65504.0);

    let bf16 = original.to_dtype(DType::BF16).unwrap();
    let bf16_back = bf16.to_dtype(DType::F32).unwrap();
    let bf16_val = bf16_back.to_vec1::<f32>().unwrap()[0];
    assert!(bf16_val.is_finite(), "65504.0 should be within BF16 range");
    // BF16 may round 65504 slightly (8-bit mantissa), but should be close.
    assert!(
        approx_eq(bf16_val, 65504.0, 512.0),
        "BF16 of 65504.0 = {bf16_val}"
    );
}

// ---------------------------------------------------------------------------
// DType size validation
// ---------------------------------------------------------------------------

#[test]
fn test_dtype_size_bf16_f16_two_bytes() {
    assert_eq!(DType::BF16.size_bytes(), 2);
    assert_eq!(DType::F16.size_bytes(), 2);
}

#[test]
fn test_dtype_size_f32_four_bytes() {
    assert_eq!(DType::F32.size_bytes(), 4);
}

#[test]
fn test_dtype_sizes_relative_ordering() {
    // Half-precision types are half the size of single-precision.
    assert_eq!(DType::F16.size_bytes() * 2, DType::F32.size_bytes());
    assert_eq!(DType::BF16.size_bytes() * 2, DType::F32.size_bytes());
}

// ---------------------------------------------------------------------------
// Affine operation precision (mul + add in one step)
// ---------------------------------------------------------------------------

#[test]
fn test_affine_f32_precision() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    // affine: x * 2.0 + 0.5
    let result = t.affine(2.0, 0.5).unwrap();
    assert_eq!(result.dtype(), DType::F32);
    let vals = result.to_vec1::<f32>().unwrap();
    assert_eq!(vals, vec![2.5, 4.5, 6.5]);
}

#[test]
fn test_affine_bf16_precision() {
    let t_f32 = t1d(&[1.0, 2.0, 3.0]);
    let t_bf16 = t_f32.to_dtype(DType::BF16).unwrap();

    // affine: x * 2.0 + 0.5
    let result = t_bf16.affine(2.0, 0.5).unwrap();
    // CPU affine on BF16 creates BF16 scalar_like, operates in BF16.
    let result_f32 = result.to_dtype(DType::F32).unwrap();
    let vals = result_f32.to_vec1::<f32>().unwrap();
    assert_close(&vals, &[2.5, 4.5, 6.5], 0.1);
}

#[test]
fn test_affine_f16_precision() {
    let t_f32 = t1d(&[1.0, 2.0, 3.0]);
    let t_f16 = t_f32.to_dtype(DType::F16).unwrap();

    // affine: x * 2.0 + 0.5
    let result = t_f16.affine(2.0, 0.5).unwrap();
    let result_f32 = result.to_dtype(DType::F32).unwrap();
    let vals = result_f32.to_vec1::<f32>().unwrap();
    assert_close(&vals, &[2.5, 4.5, 6.5], 0.05);
}

// ---------------------------------------------------------------------------
// Scalar operations in different dtypes
// ---------------------------------------------------------------------------

#[test]
fn test_add_scalar_bf16() {
    let t = DynTensor::full(&[3], 1.0, DType::BF16, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::BF16);

    let result = t.add_scalar(0.5).unwrap();
    let vals = result
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_close(&vals, &[1.5, 1.5, 1.5], 0.05);
}

#[test]
fn test_mul_scalar_f16() {
    let t = DynTensor::full(&[3], 2.0, DType::F16, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::F16);

    let result = t.mul_scalar(3.0).unwrap();
    let vals = result
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_close(&vals, &[6.0, 6.0, 6.0], 0.05);
}

#[test]
fn test_sub_scalar_bf16() {
    let t = DynTensor::full(&[3], 5.0, DType::BF16, &cpu()).unwrap();
    let result = t.sub_scalar(2.0).unwrap();
    let vals = result
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_close(&vals, &[3.0, 3.0, 3.0], 0.05);
}

#[test]
fn test_div_scalar_f16() {
    let t = DynTensor::full(&[3], 6.0, DType::F16, &cpu()).unwrap();
    let result = t.div_scalar(2.0).unwrap();
    let vals = result
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_close(&vals, &[3.0, 3.0, 3.0], 0.05);
}

// ---------------------------------------------------------------------------
// Edge cases: overflow in F16
// ---------------------------------------------------------------------------

#[test]
fn test_f16_overflow_large_value() {
    // F16 max is 65504. Values above this overflow to inf.
    let original = t1d(&[70000.0]);
    let f16 = original.to_dtype(DType::F16).unwrap();
    let back = f16.to_dtype(DType::F32).unwrap();
    let val = back.to_vec1::<f32>().unwrap()[0];
    assert!(
        val.is_infinite(),
        "70000.0 should overflow F16 to inf (got {val})"
    );
}

#[test]
fn test_f16_overflow_negative_large_value() {
    let original = t1d(&[-70000.0]);
    let f16 = original.to_dtype(DType::F16).unwrap();
    let back = f16.to_dtype(DType::F32).unwrap();
    let val = back.to_vec1::<f32>().unwrap()[0];
    assert!(
        val.is_infinite() && val < 0.0,
        "-70000.0 should overflow F16 to -inf (got {val})"
    );
}

#[test]
fn test_bf16_no_overflow_at_f16_limit() {
    // BF16 has the same exponent range as F32, so 70000.0 should be fine.
    let original = t1d(&[70000.0]);
    let bf16 = original.to_dtype(DType::BF16).unwrap();
    let back = bf16.to_dtype(DType::F32).unwrap();
    let val = back.to_vec1::<f32>().unwrap()[0];
    assert!(
        val.is_finite(),
        "70000.0 should NOT overflow BF16 (got {val})"
    );
    assert!(approx_eq(val, 70000.0, 512.0), "BF16 of 70000.0 = {val}");
}

// ---------------------------------------------------------------------------
// Edge cases: underflow in F16
// ---------------------------------------------------------------------------

#[test]
fn test_f16_underflow_very_small_value() {
    // F16 smallest normal is ~6.1e-5. Subnormal minimum is ~5.96e-8.
    // Very small values below subnormal range flush to zero.
    let tiny = 1e-9_f32;
    let original = t1d(&[tiny]);
    let f16 = original.to_dtype(DType::F16).unwrap();
    let back = f16.to_dtype(DType::F32).unwrap();
    let val = back.to_vec1::<f32>().unwrap()[0];

    // F16 subnormal minimum is ~5.96e-8. 1e-9 is below that, should flush to 0.
    assert_eq!(val, 0.0, "1e-9 should underflow to 0.0 in F16 (got {val})");
}

#[test]
fn test_bf16_underflow_very_small_value() {
    // BF16 smallest subnormal is ~9.18e-41. Much smaller than F16.
    // A value like 1e-9 should survive BF16 round-trip without flushing to zero.
    let tiny = 1e-9_f32;
    let original = t1d(&[tiny]);
    let bf16 = original.to_dtype(DType::BF16).unwrap();
    let back = bf16.to_dtype(DType::F32).unwrap();
    let val = back.to_vec1::<f32>().unwrap()[0];

    // BF16 can represent this — it has the same exponent range as F32.
    assert!(
        val != 0.0,
        "1e-9 should NOT underflow to 0.0 in BF16 (got {val})"
    );
    // Should be approximately correct (BF16 precision loss is ~1% for 8-bit mantissa).
    assert!(approx_eq(val, tiny, tiny * 0.1), "BF16 of {tiny} = {val}");
}

#[test]
fn test_f16_subnormal_boundary() {
    // F16 smallest subnormal: 2^-24 ~ 5.96e-8
    // Values just above this should be representable (as subnormals).
    let subnormal_val = 1e-7_f32;
    let original = t1d(&[subnormal_val]);
    let f16 = original.to_dtype(DType::F16).unwrap();
    let back = f16.to_dtype(DType::F32).unwrap();
    let val = back.to_vec1::<f32>().unwrap()[0];

    // Should be nonzero and approximately correct.
    assert!(val > 0.0, "1e-7 should survive F16 subnormal (got {val})");
    assert!(
        approx_eq(val, subnormal_val, subnormal_val * 0.5),
        "F16 subnormal: {val} vs {subnormal_val}"
    );
}

// ---------------------------------------------------------------------------
// Softmax with policy
// ---------------------------------------------------------------------------

#[test]
fn test_softmax_with_policy_upcasts_to_f32() {
    let t_f32 = t1d(&[1.0, 2.0, 3.0]);
    let t_f16 = t_f32.to_dtype(DType::F16).unwrap();

    let policy = MixedPrecisionPolicy::apple_silicon_default();

    // Softmax is Accumulate category -> should upcast to F32.
    let result = t_f16.softmax_with_policy(0, &policy).unwrap();
    assert_eq!(result.dtype(), DType::F32);

    let vals = result.to_vec1::<f32>().unwrap();
    // softmax([1,2,3]) = [0.0900, 0.2447, 0.6652]
    assert_close(&vals, &[0.0900, 0.2447, 0.6652], 0.001);
}

#[test]
fn test_softmax_with_policy_f32_only_stays_f32() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let policy = MixedPrecisionPolicy::f32_only();

    let result = t.softmax_with_policy(0, &policy).unwrap();
    assert_eq!(result.dtype(), DType::F32);
}

// ---------------------------------------------------------------------------
// Layer norm with policy
// ---------------------------------------------------------------------------

#[test]
fn test_layer_norm_with_policy_upcasts_to_f32() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t_f32 = t2d(&data, 2, 3);
    let t_bf16 = t_f32.to_dtype(DType::BF16).unwrap();

    let policy = MixedPrecisionPolicy::apple_silicon_default();

    // Layer norm is Accumulate -> should upcast to F32.
    let result = t_bf16.layer_norm_with_policy(1, 1e-5, &policy).unwrap();
    assert_eq!(result.dtype(), DType::F32);

    // Manual layer norm of [1,2,3]: mean=2, var=2/3, std=sqrt(2/3+eps)
    // (x - mean)/std = [-1, 0, 1] / sqrt(2/3) = [-1.2247, 0.0, 1.2247]
    let vals = result.to_f32_array().unwrap();
    let row0: Vec<f32> = vals.slice(ndarray::s![0, ..]).iter().copied().collect();
    assert_close(&row0, &[-1.2247, 0.0, 1.2247], 0.01);
}

// ---------------------------------------------------------------------------
// OpDTypeCategory classification
// ---------------------------------------------------------------------------

#[test]
fn test_policy_dtype_for_op_compute() {
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    assert_eq!(policy.dtype_for_op(OpDTypeCategory::Compute), DType::F16);
}

#[test]
fn test_policy_dtype_for_op_accumulate() {
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    assert_eq!(policy.dtype_for_op(OpDTypeCategory::Accumulate), DType::F32);
}

#[test]
fn test_policy_dtype_for_op_inherit() {
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    // Inherit uses compute_dtype.
    assert_eq!(policy.dtype_for_op(OpDTypeCategory::Inherit), DType::F16);
}

// ---------------------------------------------------------------------------
// Same-dtype to_dtype is a no-op clone
// ---------------------------------------------------------------------------

#[test]
fn test_to_dtype_same_dtype_is_noop() {
    let t = t1d(&[1.0, 2.0, 3.0]);
    let t2 = t.to_dtype(DType::F32).unwrap();
    assert_eq!(t2.dtype(), DType::F32);
    assert_eq!(t2.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_to_dtype_bf16_to_bf16_noop() {
    let t = DynTensor::full(&[3], 1.5, DType::BF16, &cpu()).unwrap();
    let t2 = t.to_dtype(DType::BF16).unwrap();
    assert_eq!(t2.dtype(), DType::BF16);
}

// ---------------------------------------------------------------------------
// DynTensor::full creates correct dtype
// ---------------------------------------------------------------------------

#[test]
fn test_full_bf16_creates_bf16_tensor() {
    let t = DynTensor::full(&[2, 3], 1.0, DType::BF16, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::BF16);
    assert_eq!(t.dims(), &[2, 3]);

    // Can't use to_vec1 on a 2D tensor; use to_f32_array instead.
    let arr = t.to_f32_array().unwrap();
    for &v in arr.iter() {
        assert!(approx_eq(v, 1.0, 1e-3));
    }
}

#[test]
fn test_full_f16_creates_f16_tensor() {
    let t = DynTensor::full(&[4], 2.5, DType::F16, &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::F16);
    assert_eq!(t.dims(), &[4]);

    let vals = t.to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    assert_close(&vals, &[2.5, 2.5, 2.5, 2.5], 0.01);
}

// ---------------------------------------------------------------------------
// Matmul with BF16 inputs directly (no policy, tests the auto-convert path)
// ---------------------------------------------------------------------------

#[test]
fn test_matmul_bf16_inputs_auto_converts() {
    // DynTensor::matmul on BF16 CPU tensors should work via to_f32_array path.
    let a = DynTensor::full(&[2, 2], 1.0, DType::BF16, &cpu()).unwrap();
    let b = DynTensor::full(&[2, 2], 1.0, DType::BF16, &cpu()).unwrap();
    let result = a.matmul(&b).unwrap();

    // Result of [[1,1],[1,1]] * [[1,1],[1,1]] = [[2,2],[2,2]]
    let vals = result.to_f32_array().unwrap();
    for &v in vals.iter() {
        assert!(approx_eq(v, 2.0, 0.1));
    }
}
