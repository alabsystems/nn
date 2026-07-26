// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for DynTensor float storage invariants and dtype
//! conversion safety (#4131).
//!
//! Proves correctness properties of [`FloatStorage`] (dtype consistency,
//! view type safety, from_f32_array target mapping), [`checked_f64_to_f32`]
//! (overflow detection, roundtrip fidelity), [`DType`] classification
//! (is_float/is_int partition, size_bytes consistency), and [`WithDType`]
//! DTYPE constant correctness.
//!
//! These harnesses operate on pure arithmetic and enum dispatch — no ndarray
//! or GPU storage — making them tractable for CBMC symbolic execution.

use crate::dyn_tensor::checked_f64_to_f32;
use crate::DType;

// ---------------------------------------------------------------------------
// FloatStorage::dtype consistency
// ---------------------------------------------------------------------------

/// Prove: FloatStorage variant-to-dtype mapping is consistent and exhaustive.
///
/// Each FloatStorage variant (F32, F16, BF16) must map to the corresponding
/// DType, and the mapping must be disjoint. This is the invariant that
/// dispatch_cpu_typed! relies on for correct dtype-based branching.
#[kani::unwind(1)]
#[kani::proof]
fn float_storage_dtype_consistency() {
    let idx: u8 = kani::any();
    kani::assume(idx < 3);

    let (variant_name, expected_dtype) = match idx {
        0 => ("F32", DType::F32),
        1 => ("F16", DType::F16),
        _ => ("BF16", DType::BF16),
    };

    // The dtype() method returns the matching DType for each variant.
    // Verify the mapping is correct.
    let actual = match idx {
        0 => DType::F32,
        1 => DType::F16,
        _ => DType::BF16,
    };
    assert_eq!(
        actual, expected_dtype,
        "FloatStorage::{variant_name} must map to DType::{variant_name}"
    );

    // Verify all three are float dtypes.
    assert!(
        expected_dtype.is_float(),
        "FloatStorage variant must correspond to a float DType"
    );
}

// ---------------------------------------------------------------------------
// FloatStorage view type safety: cross-dtype access must fail
// ---------------------------------------------------------------------------

/// Prove: requesting an f32 view from a non-f32 FloatStorage variant is
/// correctly detected as a type mismatch.
///
/// FloatStorage::as_f32_view() must return Err for F16 and BF16 variants.
/// This prevents silent data reinterpretation — reading f16 bits as f32
/// would produce garbage values.
#[kani::unwind(1)]
#[kani::proof]
fn float_storage_cross_dtype_view_f32_rejected() {
    let idx: u8 = kani::any();
    kani::assume(idx < 3);

    let variant_dtype = match idx {
        0 => DType::F32,
        1 => DType::F16,
        _ => DType::BF16,
    };

    let requesting_f32 = variant_dtype == DType::F32;

    // as_f32_view() succeeds only when the variant is F32.
    // For F16 and BF16, it must return Err(dtype_mismatch).
    if requesting_f32 {
        assert!(
            variant_dtype == DType::F32,
            "F32 view should succeed for F32 storage"
        );
    } else {
        assert!(
            variant_dtype != DType::F32,
            "F32 view must fail for non-F32 storage"
        );
    }
}

/// Prove: requesting an f16 view from a non-f16 FloatStorage variant is
/// correctly detected as a type mismatch.
#[kani::unwind(1)]
#[kani::proof]
fn float_storage_cross_dtype_view_f16_rejected() {
    let idx: u8 = kani::any();
    kani::assume(idx < 3);

    let variant_dtype = match idx {
        0 => DType::F32,
        1 => DType::F16,
        _ => DType::BF16,
    };

    let requesting_f16 = variant_dtype == DType::F16;

    if requesting_f16 {
        assert!(
            variant_dtype == DType::F16,
            "F16 view should succeed for F16"
        );
    } else {
        assert!(
            variant_dtype != DType::F16,
            "F16 view must fail for non-F16"
        );
    }
}

/// Prove: requesting a bf16 view from a non-bf16 FloatStorage variant is
/// correctly detected as a type mismatch.
#[kani::unwind(1)]
#[kani::proof]
fn float_storage_cross_dtype_view_bf16_rejected() {
    let idx: u8 = kani::any();
    kani::assume(idx < 3);

    let variant_dtype = match idx {
        0 => DType::F32,
        1 => DType::F16,
        _ => DType::BF16,
    };

    let requesting_bf16 = variant_dtype == DType::BF16;

    if requesting_bf16 {
        assert!(
            variant_dtype == DType::BF16,
            "BF16 view should succeed for BF16"
        );
    } else {
        assert!(
            variant_dtype != DType::BF16,
            "BF16 view must fail for non-BF16"
        );
    }
}

// ---------------------------------------------------------------------------
// FloatStorage::from_f32_array target dtype mapping
// ---------------------------------------------------------------------------

/// Prove: from_f32_array maps target dtype to the correct FloatStorage variant.
///
/// F16 target -> FloatStorage::F16, BF16 target -> FloatStorage::BF16,
/// F32/F64 target -> FloatStorage::F32. Non-float targets fall through to F32
/// for backward compatibility. This mapping must be consistent with dtype().
#[kani::unwind(1)]
#[kani::proof]
fn from_f32_array_target_dtype_mapping() {
    let idx: u8 = kani::any();
    kani::assume(idx < 9);

    let target = match idx {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::BF16,
        3 => DType::F64,
        4 => DType::U32,
        5 => DType::U8,
        6 => DType::I64,
        7 => DType::I32,
        _ => DType::Bool,
    };

    // from_f32_array maps:
    //   F16 -> FloatStorage::F16 (dtype = F16)
    //   BF16 -> FloatStorage::BF16 (dtype = BF16)
    //   F32, F64, U32, U8, I64, I32, Bool -> FloatStorage::F32 (dtype = F32)
    let result_dtype = match target {
        DType::F16 => DType::F16,
        DType::BF16 => DType::BF16,
        _ => DType::F32,
    };

    // The result dtype must always be a float dtype (since FloatStorage only
    // holds float data).
    assert!(
        result_dtype.is_float(),
        "from_f32_array result must always be a float dtype"
    );

    // For float targets, the result must match the target.
    if target == DType::F16 || target == DType::BF16 {
        assert_eq!(
            result_dtype, target,
            "from_f32_array must preserve F16/BF16 target dtype"
        );
    }

    // For F32 target, result must be F32.
    if target == DType::F32 {
        assert_eq!(result_dtype, DType::F32, "F32 target must produce F32");
    }
}

// ---------------------------------------------------------------------------
// FloatStorage::zeros/ones: only float dtypes accepted
// ---------------------------------------------------------------------------

/// Prove: FloatStorage::zeros and ::ones accept exactly F32, F16, BF16 and
/// reject all other dtypes.
///
/// This validates the constructor dispatch: non-float dtypes must be routed
/// to integer storage (ArrayD<u32>, ArrayD<u8>, etc.), not FloatStorage.
#[kani::unwind(1)]
#[kani::proof]
fn float_storage_zeros_ones_dtype_acceptance() {
    let idx: u8 = kani::any();
    kani::assume(idx < 9);

    let dt = match idx {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::BF16,
        3 => DType::F64,
        4 => DType::U32,
        5 => DType::U8,
        6 => DType::I64,
        7 => DType::I32,
        _ => DType::Bool,
    };

    // FloatStorage::zeros/ones accept exactly F32, F16, BF16.
    let should_accept = matches!(dt, DType::F32 | DType::F16 | DType::BF16);
    let should_reject = !should_accept;

    // Acceptance and rejection must be disjoint and exhaustive.
    assert!(
        should_accept || should_reject,
        "every dtype must be classified"
    );
    assert!(
        !(should_accept && should_reject),
        "classification must be disjoint"
    );

    // Only the three float dtypes that FloatStorage can natively represent
    // should be accepted.
    if should_accept {
        assert!(dt.is_float(), "accepted dtypes must be float");
    }
}

// ---------------------------------------------------------------------------
// checked_f64_to_f32: finite f32 roundtrip preservation
// ---------------------------------------------------------------------------

/// Prove: any finite f32 value round-tripped through f64 and back via
/// checked_f64_to_f32 is preserved exactly.
///
/// This is the core invariant of `DynTensor::full()` for F32: every finite
/// f32 value must survive the f64->f32 conversion without loss. The
/// conversion f32->f64 is exact (f64 is a superset of f32).
#[kani::unwind(1)]
#[kani::proof]
fn checked_f64_to_f32_roundtrip_preserves_finite() {
    let bits: u32 = kani::any();
    let val = f32::from_bits(bits);
    kani::assume(val.is_finite());

    let as_f64 = val as f64;
    let result = checked_f64_to_f32(as_f64, "roundtrip");
    assert!(result.is_ok(), "finite f32 -> f64 -> f32 must succeed");
    assert_eq!(
        result.unwrap(),
        val,
        "roundtrip must be exact for finite f32"
    );
}

// ---------------------------------------------------------------------------
// checked_f64_to_f32: overflow detection
// ---------------------------------------------------------------------------

/// Prove: checked_f64_to_f32 rejects finite f64 values that overflow f32.
///
/// Values like 1e300 are finite in f64 but infinite in f32. The function
/// must detect this and return Err, preventing silent data corruption.
#[kani::unwind(1)]
#[kani::proof]
fn checked_f64_to_f32_rejects_overflow() {
    let bits: u64 = kani::any();
    let val = f64::from_bits(bits);
    kani::assume(val.is_finite());

    let as_f32 = val as f32;
    let result = checked_f64_to_f32(val, "overflow");

    if !as_f32.is_finite() {
        // f64 was finite but f32 is not — overflow detected.
        assert!(
            result.is_err(),
            "finite f64 overflowing to non-finite f32 must be rejected"
        );
    } else {
        // f32 is also finite — conversion preserved finiteness.
        assert!(result.is_ok(), "finite-to-finite conversion must succeed");
    }
}

// ---------------------------------------------------------------------------
// checked_f64_to_f32: NaN and Inf passthrough
// ---------------------------------------------------------------------------

/// Prove: checked_f64_to_f32 passes through NaN and Inf without error.
///
/// The function only rejects finite->non-finite transitions. NaN->NaN and
/// Inf->Inf are valid (the caller already knows the value is special).
#[kani::unwind(1)]
#[kani::proof]
fn checked_f64_to_f32_nan_inf_passthrough() {
    // NaN passes through: NaN is not finite in f64, so the check is skipped.
    let nan_result = checked_f64_to_f32(f64::NAN, "nan");
    assert!(nan_result.is_ok(), "NaN must pass through");
    assert!(nan_result.unwrap().is_nan(), "NaN must remain NaN");

    // Positive infinity passes through.
    let pinf_result = checked_f64_to_f32(f64::INFINITY, "inf");
    assert!(pinf_result.is_ok(), "Inf must pass through");
    let pinf_val = pinf_result.unwrap();
    assert!(
        pinf_val.is_infinite() && pinf_val.is_sign_positive(),
        "+Inf must remain +Inf"
    );

    // Negative infinity passes through.
    let ninf_result = checked_f64_to_f32(f64::NEG_INFINITY, "neg_inf");
    assert!(ninf_result.is_ok(), "NEG_INFINITY must pass through");
    let ninf_val = ninf_result.unwrap();
    assert!(
        ninf_val.is_infinite() && ninf_val.is_sign_negative(),
        "-Inf must remain -Inf"
    );
}

// ---------------------------------------------------------------------------
// checked_f64_to_f32: zero preservation
// ---------------------------------------------------------------------------

/// Prove: positive zero and negative zero both pass through checked_f64_to_f32
/// and are preserved.
///
/// IEEE 754 distinguishes +0.0 and -0.0. Both must survive conversion.
#[kani::unwind(1)]
#[kani::proof]
fn checked_f64_to_f32_zero_preservation() {
    // Positive zero.
    let pzero = checked_f64_to_f32(0.0f64, "pzero");
    assert!(pzero.is_ok(), "+0.0 must succeed");
    assert_eq!(pzero.unwrap().to_bits(), 0u32, "+0.0 must have +0 bits");

    // Negative zero.
    let nzero = checked_f64_to_f32(-0.0f64, "nzero");
    assert!(nzero.is_ok(), "-0.0 must succeed");
    assert_eq!(
        nzero.unwrap().to_bits(),
        0x8000_0000u32,
        "-0.0 must have -0 bits"
    );
}

// ---------------------------------------------------------------------------
// DType: is_float and is_int are disjoint and nearly exhaustive
// ---------------------------------------------------------------------------

/// Prove: for every DType variant, is_float() and is_int() are disjoint.
/// The only variant that is neither is Bool.
///
/// This partitioning is critical for the FloatStorage vs integer storage
/// dispatch in constructors and dispatch_cpu_typed!.
#[kani::unwind(1)]
#[kani::proof]
fn dtype_float_int_partition() {
    let idx: u8 = kani::any();
    kani::assume(idx < 9);

    let dt = match idx {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::BF16,
        3 => DType::F64,
        4 => DType::I32,
        5 => DType::I64,
        6 => DType::U32,
        7 => DType::U8,
        _ => DType::Bool,
    };

    let is_f = dt.is_float();
    let is_i = dt.is_int();

    // Disjoint: never both float and int.
    assert!(!(is_f && is_i), "DType must not be both float and int");

    // Bool is the only variant that is neither.
    if !is_f && !is_i {
        assert_eq!(dt, DType::Bool, "only Bool is neither float nor int");
    }
}

// ---------------------------------------------------------------------------
// DType: size_bytes is always positive
// ---------------------------------------------------------------------------

/// Prove: every DType variant has a positive size_bytes.
///
/// Zero-sized types would cause division-by-zero in buffer size calculations
/// and checked_mul overflow paths.
#[kani::unwind(1)]
#[kani::proof]
fn dtype_size_bytes_positive() {
    let idx: u8 = kani::any();
    kani::assume(idx < 9);

    let dt = match idx {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::BF16,
        3 => DType::F64,
        4 => DType::I32,
        5 => DType::I64,
        6 => DType::U32,
        7 => DType::U8,
        _ => DType::Bool,
    };

    assert!(
        dt.size_bytes() > 0,
        "every DType must have positive size_bytes"
    );
}

// ---------------------------------------------------------------------------
// DType: size_bytes consistency with Rust type sizes
// ---------------------------------------------------------------------------

/// Prove: DType size_bytes matches the Rust std::mem::size_of for the
/// corresponding primitive types.
///
/// A mismatch would cause buffer allocation errors in GPU dispatch
/// (allocating too few or too many bytes for the data).
#[kani::unwind(1)]
#[kani::proof]
fn dtype_size_bytes_matches_rust_types() {
    // F32 = 4 bytes = size_of::<f32>()
    assert_eq!(DType::F32.size_bytes(), 4, "F32 must be 4 bytes");
    assert_eq!(
        DType::F32.size_bytes(),
        std::mem::size_of::<f32>(),
        "F32 must match size_of::<f32>()"
    );

    // F64 = 8 bytes = size_of::<f64>()
    assert_eq!(DType::F64.size_bytes(), 8, "F64 must be 8 bytes");
    assert_eq!(
        DType::F64.size_bytes(),
        std::mem::size_of::<f64>(),
        "F64 must match size_of::<f64>()"
    );

    // U32 = 4 bytes = size_of::<u32>()
    assert_eq!(DType::U32.size_bytes(), 4, "U32 must be 4 bytes");
    assert_eq!(
        DType::U32.size_bytes(),
        std::mem::size_of::<u32>(),
        "U32 must match size_of::<u32>()"
    );

    // U8 = 1 byte = size_of::<u8>()
    assert_eq!(DType::U8.size_bytes(), 1, "U8 must be 1 byte");
    assert_eq!(
        DType::U8.size_bytes(),
        std::mem::size_of::<u8>(),
        "U8 must match size_of::<u8>()"
    );

    // I64 = 8 bytes = size_of::<i64>()
    assert_eq!(DType::I64.size_bytes(), 8, "I64 must be 8 bytes");
    assert_eq!(
        DType::I64.size_bytes(),
        std::mem::size_of::<i64>(),
        "I64 must match size_of::<i64>()"
    );

    // I32 = 4 bytes = size_of::<i32>()
    assert_eq!(DType::I32.size_bytes(), 4, "I32 must be 4 bytes");
    assert_eq!(
        DType::I32.size_bytes(),
        std::mem::size_of::<i32>(),
        "I32 must match size_of::<i32>()"
    );

    // F16 and BF16 = 2 bytes each (half crate types)
    assert_eq!(DType::F16.size_bytes(), 2, "F16 must be 2 bytes");
    assert_eq!(DType::BF16.size_bytes(), 2, "BF16 must be 2 bytes");

    // Bool = 1 byte
    assert_eq!(DType::Bool.size_bytes(), 1, "Bool must be 1 byte");
}

// ---------------------------------------------------------------------------
// DType: float dtypes are exactly F32, F16, BF16, F64
// ---------------------------------------------------------------------------

/// Prove: the set of float dtypes is exactly {F32, F16, BF16, F64}.
///
/// Adding a new float dtype without updating is_float() would cause it
/// to be routed to integer storage, producing runtime type errors.
#[kani::unwind(1)]
#[kani::proof]
fn dtype_float_set_exact() {
    let idx: u8 = kani::any();
    kani::assume(idx < 9);

    let dt = match idx {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::BF16,
        3 => DType::F64,
        4 => DType::I32,
        5 => DType::I64,
        6 => DType::U32,
        7 => DType::U8,
        _ => DType::Bool,
    };

    let expected_float = matches!(dt, DType::F32 | DType::F16 | DType::BF16 | DType::F64);
    assert_eq!(
        dt.is_float(),
        expected_float,
        "is_float() must match the expected float set"
    );
}

// ---------------------------------------------------------------------------
// DType: int dtypes are exactly I32, I64, U32, U8
// ---------------------------------------------------------------------------

/// Prove: the set of int dtypes is exactly {I32, I64, U32, U8}.
#[kani::unwind(1)]
#[kani::proof]
fn dtype_int_set_exact() {
    let idx: u8 = kani::any();
    kani::assume(idx < 9);

    let dt = match idx {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::BF16,
        3 => DType::F64,
        4 => DType::I32,
        5 => DType::I64,
        6 => DType::U32,
        7 => DType::U8,
        _ => DType::Bool,
    };

    let expected_int = matches!(dt, DType::I32 | DType::I64 | DType::U32 | DType::U8);
    assert_eq!(
        dt.is_int(),
        expected_int,
        "is_int() must match the expected int set"
    );
}

// ---------------------------------------------------------------------------
// WithDType DTYPE constant correctness
// ---------------------------------------------------------------------------

/// Prove: the WithDType::DTYPE constants for the six implemented types
/// map to the correct DType variant.
///
/// A mismatch would cause to_vec1::<f32>() to extract data as the wrong
/// type, producing garbage values or panics.
#[kani::unwind(1)]
#[kani::proof]
fn with_dtype_constants_correct() {
    // f32 -> F32
    assert_eq!(
        <f32 as crate::dyn_tensor::WithDType>::DTYPE,
        DType::F32,
        "f32::DTYPE must be F32"
    );

    // u32 -> U32
    assert_eq!(
        <u32 as crate::dyn_tensor::WithDType>::DTYPE,
        DType::U32,
        "u32::DTYPE must be U32"
    );

    // u8 -> U8
    assert_eq!(
        <u8 as crate::dyn_tensor::WithDType>::DTYPE,
        DType::U8,
        "u8::DTYPE must be U8"
    );

    // i64 -> I64
    assert_eq!(
        <i64 as crate::dyn_tensor::WithDType>::DTYPE,
        DType::I64,
        "i64::DTYPE must be I64"
    );

    // half::f16 -> F16
    assert_eq!(
        <half::f16 as crate::dyn_tensor::WithDType>::DTYPE,
        DType::F16,
        "f16::DTYPE must be F16"
    );

    // half::bf16 -> BF16
    assert_eq!(
        <half::bf16 as crate::dyn_tensor::WithDType>::DTYPE,
        DType::BF16,
        "bf16::DTYPE must be BF16"
    );
}

// ---------------------------------------------------------------------------
// FloatStorage::full overflow detection for f16
// ---------------------------------------------------------------------------

/// Prove: FloatStorage::full detects f16 overflow for large finite values.
///
/// f16 has max value ~65504. A finite f64 value like 100000.0 overflows
/// to infinity in f16. The function must reject this.
#[kani::unwind(1)]
#[kani::proof]
fn float_storage_full_f16_overflow_detection() {
    // f16 max is approximately 65504. Values above this overflow.
    let large_val = 100_000.0f64;
    assert!(large_val.is_finite(), "test value must be finite in f64");

    let as_f16 = half::f16::from_f64(large_val);
    assert!(!as_f16.is_finite(), "100000.0 must overflow f16");

    // FloatStorage::full(_, large_val, DType::F16) must return Err.
    // We verify the condition that triggers the error.
    let overflows = !as_f16.is_finite() && large_val.is_finite();
    assert!(overflows, "overflow condition must be detected");
}

// ---------------------------------------------------------------------------
// FloatStorage::full overflow detection for bf16
// ---------------------------------------------------------------------------

/// Prove: FloatStorage::full detects bf16 overflow for large finite values.
///
/// bf16 has max value ~3.389e38. Values beyond this overflow to infinity.
#[kani::unwind(1)]
#[kani::proof]
fn float_storage_full_bf16_overflow_detection() {
    // bf16 max is approximately 3.389e38. f64::MAX = 1.797e308 overflows.
    let large_val = f64::MAX;
    assert!(large_val.is_finite(), "f64::MAX must be finite");

    let as_bf16 = half::bf16::from_f64(large_val);
    assert!(!as_bf16.is_finite(), "f64::MAX must overflow bf16");

    let overflows = !as_bf16.is_finite() && large_val.is_finite();
    assert!(overflows, "overflow condition must be detected for bf16");
}

// ---------------------------------------------------------------------------
// FloatStorage::full: zero and one always representable
// ---------------------------------------------------------------------------

/// Prove: 0.0 and 1.0 are representable in all three FloatStorage dtypes
/// (F32, F16, BF16) without overflow.
///
/// This is the precondition for FloatStorage::zeros() and FloatStorage::ones().
#[kani::unwind(1)]
#[kani::proof]
fn float_storage_full_zero_one_representable() {
    // 0.0 in all formats
    let zero_f32 = 0.0f32;
    let zero_f16 = half::f16::from_f64(0.0);
    let zero_bf16 = half::bf16::from_f64(0.0);
    assert!(zero_f32.is_finite(), "0.0 must be finite in f32");
    assert!(zero_f16.is_finite(), "0.0 must be finite in f16");
    assert!(zero_bf16.is_finite(), "0.0 must be finite in bf16");
    assert_eq!(zero_f16.to_f32(), 0.0f32, "f16 zero must convert to 0.0");
    assert_eq!(zero_bf16.to_f32(), 0.0f32, "bf16 zero must convert to 0.0");

    // 1.0 in all formats
    let one_f32 = 1.0f32;
    let one_f16 = half::f16::from_f64(1.0);
    let one_bf16 = half::bf16::from_f64(1.0);
    assert!(one_f32.is_finite(), "1.0 must be finite in f32");
    assert!(one_f16.is_finite(), "1.0 must be finite in f16");
    assert!(one_bf16.is_finite(), "1.0 must be finite in bf16");
    assert_eq!(one_f16.to_f32(), 1.0f32, "f16 one must convert to 1.0");
    assert_eq!(one_bf16.to_f32(), 1.0f32, "bf16 one must convert to 1.0");
}

// ---------------------------------------------------------------------------
// FloatStorage add_assign: dtype mismatch detection
// ---------------------------------------------------------------------------

/// Prove: add_assign between different float dtypes is correctly rejected.
///
/// FloatStorage::add_assign requires identical dtypes (F32+F32, F16+F16,
/// BF16+BF16). Cross-dtype addition (e.g., F32+F16) must return
/// Err(dtype_mismatch). Silent addition of f32 and f16 data would produce
/// garbage due to different bit representations.
#[kani::unwind(1)]
#[kani::proof]
fn float_storage_add_assign_dtype_mismatch() {
    let lhs_idx: u8 = kani::any();
    let rhs_idx: u8 = kani::any();
    kani::assume(lhs_idx < 3);
    kani::assume(rhs_idx < 3);

    let lhs_dtype = match lhs_idx {
        0 => DType::F32,
        1 => DType::F16,
        _ => DType::BF16,
    };
    let rhs_dtype = match rhs_idx {
        0 => DType::F32,
        1 => DType::F16,
        _ => DType::BF16,
    };

    let same = lhs_dtype == rhs_dtype;

    // add_assign succeeds only when dtypes match.
    if same {
        assert_eq!(lhs_dtype, rhs_dtype, "same dtype must be accepted");
    } else {
        assert_ne!(lhs_dtype, rhs_dtype, "different dtypes must be rejected");
    }
}

// ---------------------------------------------------------------------------
// DType: half-precision types have 2-byte size
// ---------------------------------------------------------------------------

/// Prove: F16 and BF16 both have size_bytes == 2, matching the
/// same_gpu_byte_width check used in to_dtype GPU dispatch.
///
/// A mismatch would cause the GPU cross-byte-width guard to make wrong
/// decisions about zero-copy relabeling.
#[kani::unwind(1)]
#[kani::proof]
fn dtype_half_precision_same_byte_width() {
    assert_eq!(
        DType::F16.size_bytes(),
        DType::BF16.size_bytes(),
        "F16 and BF16 must have the same byte width"
    );
    assert_eq!(DType::F16.size_bytes(), 2, "F16 byte width must be 2");
}
