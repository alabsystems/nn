#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cat/stack and FloatStorage dtype tests (extracted from shape_dtype_tests.rs).
//!
//! Tests for cat, stack, and FloatStorage-related operations on non-f32 dtypes
//! (U32, I64, U8, BF16, F16).

use crate::dyn_tensor::test_helpers::cpu;
use crate::dyn_tensor::DynTensor;
use crate::DType;

// -- cat non-f32 dtypes (#1259) ------------------------------------------------

#[test]
fn test_cat_u32() {
    let a = DynTensor::from_vec_u32(vec![1, 2, 3], &[3], &cpu()).unwrap();
    let b = DynTensor::from_vec_u32(vec![4, 5], &[2], &cpu()).unwrap();
    let c = DynTensor::cat(&[&a, &b], 0).unwrap();
    assert_eq!(c.dims(), &[5]);
    assert_eq!(c.dtype(), DType::U32);
    let vals = c.as_cpu_u32().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[1, 2, 3, 4, 5]);
}

#[test]
fn test_cat_u32_2d() {
    let a = DynTensor::from_vec_u32(vec![1, 2, 3, 4], &[2, 2], &cpu()).unwrap();
    let b = DynTensor::from_vec_u32(vec![5, 6], &[1, 2], &cpu()).unwrap();
    let c = DynTensor::cat(&[&a, &b], 0).unwrap();
    assert_eq!(c.dims(), &[3, 2]);
    assert_eq!(c.dtype(), DType::U32);
    let vals = c.as_cpu_u32().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_cat_i64() {
    let a = DynTensor::from_vec_i64(vec![10, 20], &[2], &cpu()).unwrap();
    let b = DynTensor::from_vec_i64(vec![30, 40, 50], &[3], &cpu()).unwrap();
    let c = DynTensor::cat(&[&a, &b], 0).unwrap();
    assert_eq!(c.dims(), &[5]);
    assert_eq!(c.dtype(), DType::I64);
    let vals = c.as_cpu_i64().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[10, 20, 30, 40, 50]);
}

#[test]
fn test_cat_u8() {
    let a = DynTensor::from_vec_u8(vec![10, 20, 30], &[3], &cpu()).unwrap();
    let b = DynTensor::from_vec_u8(vec![40, 50], &[2], &cpu()).unwrap();
    let c = DynTensor::cat(&[&a, &b], 0).unwrap();
    assert_eq!(c.dims(), &[5]);
    assert_eq!(c.dtype(), DType::U8);
    let vals = c.as_cpu_u8().unwrap();
    assert_eq!(vals.as_slice().unwrap(), &[10, 20, 30, 40, 50]);
}

#[test]
fn test_cat_dtype_mismatch_error() {
    let a = DynTensor::from_vec_u32(vec![1, 2], &[2], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![3.0f32, 4.0], &[2], &cpu()).unwrap();
    let result = DynTensor::cat(&[&a, &b], 0);
    assert!(result.is_err(), "cat should reject dtype mismatch");
}

// -- stack non-f32 dtypes (delegates to cat) -----------------------------------

#[test]
fn test_stack_u32() {
    let a = DynTensor::from_vec_u32(vec![1, 2, 3], &[3], &cpu()).unwrap();
    let b = DynTensor::from_vec_u32(vec![4, 5, 6], &[3], &cpu()).unwrap();
    let s = DynTensor::stack(&[&a, &b], 0).unwrap();
    assert_eq!(s.dims(), &[2, 3]);
    assert_eq!(s.dtype(), DType::U32);
}

#[test]
fn test_stack_i64() {
    let a = DynTensor::from_vec_i64(vec![10, 20], &[2], &cpu()).unwrap();
    let b = DynTensor::from_vec_i64(vec![30, 40], &[2], &cpu()).unwrap();
    let s = DynTensor::stack(&[&a, &b], 0).unwrap();
    assert_eq!(s.dims(), &[2, 2]);
    assert_eq!(s.dtype(), DType::I64);
}

// -- cat f32 regression --------------------------------------------------------

#[test]
fn test_cat_f32_regression() {
    let a = DynTensor::from_vec(vec![1.0f32, 2.0], &[2], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![3.0f32, 4.0, 5.0], &[3], &cpu()).unwrap();
    let c = DynTensor::cat(&[&a, &b], 0).unwrap();
    assert_eq!(c.dims(), &[5]);
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

// -- bf16/f16 FloatStorage cat/slice_set (#1646 D3 — native BF16/F16) --------

/// BF16 cat uses native FloatStorage::BF16 path via `as_cpu_bf16`/`from_cpu_bf16`
/// in `cat_cpu`. Landed as part of #1646 D3.
#[test]
fn test_cat_bf16_native_storage() {
    let a = DynTensor::zeros(&[3], DType::BF16, &cpu()).unwrap();
    let b = DynTensor::full(&[2], 1.0, DType::BF16, &cpu()).unwrap();
    let c = DynTensor::cat(&[&a, &b], 0).unwrap();
    assert_eq!(c.dims(), &[5]);
    assert_eq!(c.dtype(), DType::BF16);
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 5);
}

/// F16 cat uses native FloatStorage::F16 path. Same #1646 D3 fix as BF16.
#[test]
fn test_cat_f16_native_storage() {
    let a = DynTensor::zeros(&[3], DType::F16, &cpu()).unwrap();
    let b = DynTensor::full(&[2], 1.0, DType::F16, &cpu()).unwrap();
    let c = DynTensor::cat(&[&a, &b], 0).unwrap();
    assert_eq!(c.dims(), &[5]);
    assert_eq!(c.dtype(), DType::F16);
}

/// BF16 slice_set dispatches to `slice_set_half` for native BF16 storage.
/// Landed as part of #1646 D3.
#[test]
fn test_slice_set_bf16_native_storage() {
    let t = DynTensor::zeros(&[5], DType::BF16, &cpu()).unwrap();
    let src = DynTensor::full(&[2], 1.0, DType::BF16, &cpu()).unwrap();
    let updated = t.slice_set(0, 1, &src).unwrap();
    let vals = updated.to_flat_vec::<f32>().unwrap();
    // After slice_set, elements [1] and [2] should be 1.0
    assert!((vals[1] - 1.0).abs() < 0.01);
    assert!((vals[2] - 1.0).abs() < 0.01);
}

/// F32 tensors created via zeros() use FloatStorage::F32 — this path
/// was broken by #1651 and fixed by W2-109. Regression guard.
#[test]
fn test_cat_f32_float_storage_regression() {
    let a = DynTensor::zeros(&[3], DType::F32, &cpu()).unwrap();
    let b = DynTensor::full(&[2], 1.0, DType::F32, &cpu()).unwrap();
    let c = DynTensor::cat(&[&a, &b], 0).unwrap();
    assert_eq!(c.dims(), &[5]);
    assert_eq!(c.dtype(), DType::F32);
    let vals = c.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 0.0, 0.0, 1.0, 1.0]);
}

/// F32 slice_set with FloatStorage::F32 — regression guard for #1651 fix.
#[test]
fn test_slice_set_f32_float_storage_regression() {
    let t = DynTensor::zeros(&[5], DType::F32, &cpu()).unwrap();
    let src = DynTensor::full(&[2], 7.0, DType::F32, &cpu()).unwrap();
    let t = t.slice_set(0, 1, &src).unwrap();
    let vals = t.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![0.0, 7.0, 7.0, 0.0, 0.0]);
}

// -- FloatStorage::full() overflow boundary (#1646 D1 gap) --------------------

/// f16 max representable value is ~65504. Values above this overflow to f16::INFINITY.
/// FloatStorage::full() now has an overflow guard for f16 (#1646 P1-225 finding 2).
#[test]
fn test_full_f16_overflow_returns_error() {
    // 100_000.0 > f16::MAX (~65504) — overflow guard rejects this.
    let result = DynTensor::full(&[1], 100_000.0, DType::F16, &cpu());
    assert!(result.is_err(), "f16 overflow should return Err");
}

/// bf16 shares f32's exponent range but has only 8 mantissa bits.
/// Values in bf16 range should round-trip correctly.
#[test]
fn test_full_bf16_round_trip_precision() {
    let t = DynTensor::full(&[1], 2.75, DType::BF16, &cpu()).unwrap();
    let val = t.to_scalar::<f32>().unwrap();
    // bf16 has ~2 decimal digits of precision. 2.75 is exactly representable.
    assert!(
        (val - 2.75).abs() < 0.02,
        "bf16 round-trip: expected ~2.75, got {val}"
    );
}

/// Narrow on FloatStorage::F32 is zero-copy — regression guard for #1651.
#[test]
fn test_narrow_f32_float_storage_regression() {
    let t = DynTensor::full(&[5], 42.0, DType::F32, &cpu()).unwrap();
    let sliced = t.narrow(0, 1, 3).unwrap();
    assert_eq!(sliced.dims(), &[3]);
    let vals = sliced.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![42.0, 42.0, 42.0]);
}

// -- BF16 CPU fallback regression tests (#1793) --------------------------------
//
// These tests prove that BF16 tensors on CPU work correctly for all core ops,
// ensuring that GPU→CPU fallback paths (when Metal returns None for BF16) do
// not crash with DTypeMismatch errors.

/// Assert each element of a BF16 tensor (converted to f32) matches expected values.
/// Uses 0.01 tolerance — appropriate for values exactly representable in BF16.
fn assert_bf16_vals(tensor: &DynTensor, expected: &[f32]) {
    let vals = tensor.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), expected.len(), "length mismatch");
    for (i, (&got, &want)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 0.01,
            "[{i}] expected {want}, got {got}"
        );
    }
}

/// AC1: BF16 tensors can be created and used on CPU without DTypeMismatch.
/// This covers the basic path: BF16 GPU tensor → to_device(CPU) → CPU ops.
#[test]
fn test_bf16_cpu_index_select() {
    // Create BF16 CPU tensor (simulates GPU→CPU fallback result).
    let data: Vec<half::bf16> = vec![10.0, 20.0, 30.0, 40.0, 50.0]
        .into_iter()
        .map(half::bf16::from_f32)
        .collect();
    let t = DynTensor::from_vec_bf16(data, &[5], &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::BF16);

    // index_select should work on BF16 via dispatch_cpu_typed! macro.
    let ids = DynTensor::from_vec_u32(vec![1, 3], &[2], &cpu()).unwrap();
    let selected = t.index_select(&ids, 0).unwrap();
    assert_eq!(selected.dims(), &[2]);
    assert_eq!(selected.dtype(), DType::BF16);

    // 20.0 and 40.0 are exactly representable in BF16.
    assert_bf16_vals(&selected, &[20.0, 40.0]);
}

/// AC3: Binary ops between BF16 tensors work on CPU fallback path.
/// Uses the promote-compute-demote pattern (to_f32_array → compute → from_f32_result).
/// All expected values are exactly representable in BF16.
#[test]
fn test_bf16_cpu_binary_ops() {
    let a_data: Vec<half::bf16> = vec![1.0, 2.0, 3.0]
        .into_iter()
        .map(half::bf16::from_f32)
        .collect();
    let b_data: Vec<half::bf16> = vec![4.0, 5.0, 6.0]
        .into_iter()
        .map(half::bf16::from_f32)
        .collect();
    let a = DynTensor::from_vec_bf16(a_data, &[3], &cpu()).unwrap();
    let b = DynTensor::from_vec_bf16(b_data, &[3], &cpu()).unwrap();

    // Addition: [1+4, 2+5, 3+6] = [5, 7, 9].
    let sum = (&a + &b).unwrap();
    assert_eq!(sum.dtype(), DType::BF16);
    assert_bf16_vals(&sum, &[5.0, 7.0, 9.0]);

    // Multiplication: [1*4, 2*5, 3*6] = [4, 10, 18].
    let prod = (&a * &b).unwrap();
    assert_eq!(prod.dtype(), DType::BF16);
    assert_bf16_vals(&prod, &[4.0, 10.0, 18.0]);

    // Subtraction: [4-1, 5-2, 6-3] = [3, 3, 3].
    let diff = (&b - &a).unwrap();
    assert_eq!(diff.dtype(), DType::BF16);
    assert_bf16_vals(&diff, &[3.0, 3.0, 3.0]);

    // Division: [4/1, 5/2, 6/3] = [4.0, 2.5, 2.0].
    let quot = (&b / &a).unwrap();
    assert_eq!(quot.dtype(), DType::BF16);
    assert_bf16_vals(&quot, &[4.0, 2.5, 2.0]);
}

/// AC4: F32 CPU ops are not regressed by BF16 support.
#[test]
fn test_f32_cpu_index_select_no_regression() {
    let t = DynTensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0, 50.0], &[5], &cpu()).unwrap();
    let ids = DynTensor::from_vec_u32(vec![0, 2, 4], &[3], &cpu()).unwrap();
    let selected = t.index_select(&ids, 0).unwrap();
    assert_eq!(selected.dims(), &[3]);
    assert_eq!(selected.dtype(), DType::F32);
    let vals = selected.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![10.0, 30.0, 50.0]);
}

/// BF16 narrow + transpose on CPU — shape ops use dispatch_cpu_typed! macro.
#[test]
fn test_bf16_cpu_shape_ops() {
    let data: Vec<half::bf16> = (0..12).map(|i| half::bf16::from_f32(i as f32)).collect();
    let t = DynTensor::from_vec_bf16(data, &[3, 4], &cpu()).unwrap();
    assert_eq!(t.dtype(), DType::BF16);

    // Narrow.
    let narrowed = t.narrow(0, 1, 2).unwrap();
    assert_eq!(narrowed.dims(), &[2, 4]);
    assert_eq!(narrowed.dtype(), DType::BF16);

    // Transpose.
    let transposed = narrowed.transpose(0, 1).unwrap();
    assert_eq!(transposed.dims(), &[4, 2]);
    assert_eq!(transposed.dtype(), DType::BF16);

    // Verify actual values survived narrow+transpose.
    // Input [3,4]: row 0=[0,1,2,3], row 1=[4,5,6,7], row 2=[8,9,10,11].
    // After narrow(0,1,2): rows 1-2 → [[4,5,6,7],[8,9,10,11]] shape [2,4].
    // After transpose(0,1): [[4,8],[5,9],[6,10],[7,11]] in row-major.
    assert_bf16_vals(&transposed, &[4.0, 8.0, 5.0, 9.0, 6.0, 10.0, 7.0, 11.0]);
}

/// BF16 math ops (exp, sqrt) on CPU — uses to_f32_array promote-compute-demote.
#[test]
fn test_bf16_cpu_math_ops() {
    let data: Vec<half::bf16> = vec![1.0, 4.0, 9.0]
        .into_iter()
        .map(half::bf16::from_f32)
        .collect();
    let t = DynTensor::from_vec_bf16(data, &[3], &cpu()).unwrap();

    // sqrt(1)=1.0, sqrt(4)=2.0, sqrt(9)=3.0 — all exactly representable in BF16.
    let result = t.sqrt().unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    assert_bf16_vals(&result, &[1.0, 2.0, 3.0]);
}
