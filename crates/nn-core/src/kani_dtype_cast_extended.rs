// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for DType cast safety (#4316).
//!
//! Proves 8 properties of dtype conversion that the GPU dispatch, mixed
//! precision, and quantization pipelines depend on:
//!
//! 1. F32->F16 finite preservation (in-range values)
//! 2. F32->BF16 finite preservation
//! 3. F32->F16 overflow detection (|x| > 65504)
//! 4. Subnormal handling (f32 values below f16 min_normal)
//! 5. DType size_bytes consistency (F32=4, F16=2, BF16=2, U8=1, U32=4, I64=8)
//! 6. Cast chain equivalence (F32->F16->F32 within F16 epsilon)
//! 7. Cast preserves zero (0.0f32 through any float dtype)
//! 8. Cast preserves sign (non-zero finite f32 through f16/bf16)
//!
//! All harnesses operate on pure arithmetic and the `half` crate — no ndarray
//! or GPU storage — making them tractable for CBMC symbolic execution.

#![cfg(kani)]

use crate::DType;

// ---------------------------------------------------------------------------
// 1. F32->F16 finite preservation: in-range values stay finite
// ---------------------------------------------------------------------------

/// Prove: for any finite f32 value in [-65504, 65504] (the F16 representable
/// range), casting to f16 and back to f32 produces a finite result.
///
/// This guarantees the dtype conversion path in `to_dtype(F16)` does not
/// produce NaN or infinity from valid F16-range inputs. GPU kernels receiving
/// F16 data depend on this for correct activation computation.
#[kani::proof]
#[kani::unwind(4)]
fn proof_f32_to_f16_finite_preservation() {
    let val: f32 = kani::any();
    kani::assume!(val.is_finite());
    kani::assume!(val >= -65504.0);
    kani::assume!(val <= 65504.0);

    let as_f16 = half::f16::from_f32(val);
    let roundtrip = as_f16.to_f32();

    assert!(
        roundtrip.is_finite(),
        "F32->F16->F32 must preserve finiteness for in-range values"
    );
}

// ---------------------------------------------------------------------------
// 2. F32->BF16 finite preservation
// ---------------------------------------------------------------------------

/// Prove: for any finite f32 value within BF16 representable range, casting
/// to bf16 and back to f32 produces a finite result.
///
/// BF16 shares f32's exponent range but has only 7 mantissa bits. Finite f32
/// values within BF16 max (~3.39e38) must survive the round-trip as finite.
/// Mixed-precision training stores weights in BF16 and depends on this.
#[kani::proof]
#[kani::unwind(4)]
fn proof_f32_to_bf16_finite_preservation() {
    let val: f32 = kani::any();
    kani::assume!(val.is_finite());
    // BF16 max is approximately 3.3895e38 (same exponent range as f32,
    // but fewer mantissa bits). Use a conservative bound.
    kani::assume!(val.abs() <= 3.3e38);

    let as_bf16 = half::bf16::from_f32(val);
    let roundtrip = as_bf16.to_f32();

    assert!(
        roundtrip.is_finite(),
        "F32->BF16->F32 must preserve finiteness for in-range values"
    );
}

// ---------------------------------------------------------------------------
// 3. F32->F16 overflow detection: |x| > 65504 produces infinity
// ---------------------------------------------------------------------------

/// Prove: for f32 values strictly above the F16 rounding boundary (65520),
/// casting to f16 produces either infinity or the F16 max (65504).
///
/// The rounding boundary is 65520: values in (65504, 65520] may round to
/// 65504 under round-to-nearest-even, while values above 65520 overflow to
/// infinity. This harness verifies overflow detection for the to_dtype path.
#[kani::proof]
#[kani::unwind(4)]
fn proof_f32_to_f16_overflow_detection() {
    let val: f32 = kani::any();
    kani::assume!(val.is_finite());
    kani::assume!(val > 65520.0);
    // Keep bounded for CBMC tractability.
    kani::assume!(val <= 1.0e10);

    let as_f16 = half::f16::from_f32(val);
    let back = as_f16.to_f32();

    // Must overflow to infinity or saturate at f16 max.
    assert!(
        back.is_infinite() || back == 65504.0,
        "F32 values above 65520 must overflow to inf or clamp to f16 max"
    );
}

// ---------------------------------------------------------------------------
// 4. Subnormal handling: small f32 values -> f16 rounds to zero or subnormal
// ---------------------------------------------------------------------------

/// Prove: for f32 values below the F16 minimum normal (~6.1e-5), casting to
/// f16 produces either zero or a subnormal f16 (which is still finite and
/// has magnitude <= the original).
///
/// F16 min normal is 2^-14 = 6.103515625e-5. Values below this enter the
/// subnormal regime where precision degrades. The result must be finite and
/// not exceed the original magnitude.
#[kani::proof]
#[kani::unwind(4)]
fn proof_f16_subnormal_handling() {
    let val: f32 = kani::any();
    kani::assume!(val.is_finite());
    // Below f16 min normal but above zero.
    kani::assume!(val > 0.0);
    kani::assume!(val < 6.104e-5); // Just below f16 min normal

    let as_f16 = half::f16::from_f32(val);
    let back = as_f16.to_f32();

    // Result must be finite (no NaN/Inf from small values).
    assert!(
        back.is_finite(),
        "subnormal f16 conversion must produce finite result"
    );
    // Result is either zero (flushed) or a subnormal <= original.
    assert!(
        back >= 0.0,
        "positive input must not produce negative output"
    );
    // Magnitude cannot increase through narrowing conversion.
    assert!(
        back <= val,
        "f16 subnormal conversion must not increase magnitude"
    );
}

// ---------------------------------------------------------------------------
// 5. DType size_bytes consistency
// ---------------------------------------------------------------------------

/// Prove: DType byte sizes match their documented values.
///
/// GPU buffer allocation computes `numel * size_bytes()`. Incorrect sizes
/// cause buffer overrun (if too small) or waste (if too large). These values
/// are the ground truth for Metal buffer sizing, safetensors serialization,
/// and zero-copy dtype relabeling guards.
#[kani::proof]
#[kani::unwind(1)]
fn proof_dtype_size_bytes_consistency() {
    assert_eq!(DType::F32.size_bytes(), 4, "F32 must be 4 bytes");
    assert_eq!(DType::F16.size_bytes(), 2, "F16 must be 2 bytes");
    assert_eq!(DType::BF16.size_bytes(), 2, "BF16 must be 2 bytes");
    assert_eq!(DType::U8.size_bytes(), 1, "U8 must be 1 byte");
    assert_eq!(DType::U32.size_bytes(), 4, "U32 must be 4 bytes");
    assert_eq!(DType::I64.size_bytes(), 8, "I64 must be 8 bytes");

    // Additional consistency checks: half-precision types share width.
    assert_eq!(
        DType::F16.size_bytes(),
        DType::BF16.size_bytes(),
        "F16 and BF16 must have identical byte width"
    );
    // 32-bit types share width.
    assert_eq!(
        DType::F32.size_bytes(),
        DType::U32.size_bytes(),
        "F32 and U32 must have identical byte width"
    );
    assert_eq!(
        DType::F32.size_bytes(),
        DType::I32.size_bytes(),
        "F32 and I32 must have identical byte width"
    );
}

// ---------------------------------------------------------------------------
// 6. Cast chain equivalence: F32->F16->F32 within F16 epsilon
// ---------------------------------------------------------------------------

/// Prove: the F32->F16->F32 round-trip produces a value within F16 epsilon
/// of the original for normal-range values.
///
/// F16 has 10 explicit mantissa bits, giving a relative precision of 2^-10
/// (approximately 9.77e-4). The absolute error of the round-trip is bounded
/// by 2^-10 * |x|. This bound is critical for quantization safety: it
/// quantifies the worst-case information loss per F16 conversion step.
#[kani::proof]
#[kani::unwind(4)]
fn proof_f32_f16_f32_roundtrip_within_epsilon() {
    let val: f32 = kani::any();
    kani::assume!(val.is_finite());
    // Restrict to normal-range values where relative error bound applies.
    // F16 min normal is 2^-14 ~= 6.1e-5.
    kani::assume!(val.abs() >= 6.104e-5);
    kani::assume!(val.abs() <= 65504.0);

    let as_f16 = half::f16::from_f32(val);
    let roundtrip = as_f16.to_f32();

    assert!(
        roundtrip.is_finite(),
        "round-trip must produce finite result"
    );

    let abs_error = (roundtrip - val).abs();
    // F16 has 10 mantissa bits => relative precision 2^-10.
    // Use 2^-10 = 1/1024 as the relative error bound (conservative for
    // round-to-nearest-even which gives half-ULP = 2^-11).
    let error_bound = val.abs() * (1.0 / 1024.0);
    assert!(
        abs_error <= error_bound,
        "F16 round-trip error must be within 2^-10 * |x|"
    );
}

// ---------------------------------------------------------------------------
// 7. Cast preserves zero: 0.0f32 through any float dtype and back
// ---------------------------------------------------------------------------

/// Prove: 0.0f32 cast to f16 or bf16 and back is exactly 0.0.
///
/// IEEE 754 zero has a unique bit pattern (all exponent and mantissa bits
/// zero). Both f16 and bf16 share this encoding. Zero preservation is
/// critical for bias initialization, padding, and masking operations.
#[kani::proof]
#[kani::unwind(1)]
fn proof_cast_preserves_zero() {
    // F32 -> F16 -> F32: positive zero
    let f16_zero = half::f16::from_f32(0.0f32);
    let f16_back = f16_zero.to_f32();
    assert_eq!(
        f16_back.to_bits(),
        0.0f32.to_bits(),
        "+0.0 must survive F16 round-trip with exact bit pattern"
    );

    // F32 -> BF16 -> F32: positive zero
    let bf16_zero = half::bf16::from_f32(0.0f32);
    let bf16_back = bf16_zero.to_f32();
    assert_eq!(
        bf16_back.to_bits(),
        0.0f32.to_bits(),
        "+0.0 must survive BF16 round-trip with exact bit pattern"
    );

    // F32 -> F16 -> F32: negative zero
    let f16_neg_zero = half::f16::from_f32(-0.0f32);
    let f16_neg_back = f16_neg_zero.to_f32();
    assert_eq!(
        f16_neg_back.to_bits(),
        (-0.0f32).to_bits(),
        "-0.0 must survive F16 round-trip with exact bit pattern"
    );

    // F32 -> BF16 -> F32: negative zero
    let bf16_neg_zero = half::bf16::from_f32(-0.0f32);
    let bf16_neg_back = bf16_neg_zero.to_f32();
    assert_eq!(
        bf16_neg_back.to_bits(),
        (-0.0f32).to_bits(),
        "-0.0 must survive BF16 round-trip with exact bit pattern"
    );
}

// ---------------------------------------------------------------------------
// 8. Cast preserves sign: non-zero finite f32 through f16/bf16
// ---------------------------------------------------------------------------

/// Prove: for non-zero finite f32 values within the representable range,
/// the sign is preserved after casting to f16 or bf16 and back.
///
/// The sign bit is the MSB in all IEEE 754 formats (f32, f16, bf16).
/// Narrowing conversion preserves the sign bit. This is essential for
/// correct gradient computation in mixed-precision training.
#[kani::proof]
#[kani::unwind(4)]
fn proof_cast_preserves_sign() {
    let val: f32 = kani::any();
    kani::assume!(val.is_finite());
    kani::assume!(val != 0.0);
    // Stay above subnormal range to avoid flush-to-zero (which loses sign).
    kani::assume!(val.abs() >= 6.104e-5); // Above f16 min normal
    kani::assume!(val.abs() <= 65504.0); // Within f16 range

    let original_positive = val.is_sign_positive();

    // F16 path
    let as_f16 = half::f16::from_f32(val);
    let f16_back = as_f16.to_f32();
    if f16_back != 0.0 {
        assert_eq!(
            f16_back.is_sign_positive(),
            original_positive,
            "sign must be preserved through F16 conversion"
        );
    }

    // BF16 path
    let as_bf16 = half::bf16::from_f32(val);
    let bf16_back = as_bf16.to_f32();
    if bf16_back != 0.0 {
        assert_eq!(
            bf16_back.is_sign_positive(),
            original_positive,
            "sign must be preserved through BF16 conversion"
        );
    }
}
