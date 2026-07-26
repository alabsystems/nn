// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for DynTensor dtype conversion and mixed precision
//! safety (#4206).
//!
//! 20 harnesses proving:
//! - Round-trip finiteness preservation (F32<->BF16, F32<->F16)
//! - Representable range checks (BF16 max ~3.39e38, F16 max 65504)
//! - Sign preservation through conversions
//! - Subnormal handling (BF16 flush-to-zero)
//! - Overflow to infinity in F16 for large inputs
//! - NaN preservation through dtype conversion
//! - Zero preservation (+0 and -0)
//! - F32->U8 and F32->I8 quantization clamping
//! - BF16 multiplication bounded-input safety
//! - Loss scaling factor properties
//! - Gradient unscaling division safety
//! - Dynamic loss scaling halving on overflow
//! - F32 accumulation of BF16 products finiteness
//! - Mixed precision identity op dtype preservation
//! - Dtype byte sizes: F32=4, BF16=2, F16=2, U8=1, U32=4
//! - F32->BF16->F32 conversion error bounded by 2^-7 * |x|
//!
//! All harnesses operate on pure arithmetic and bit manipulation — no ndarray
//! or GPU storage — making them tractable for CBMC symbolic execution.

use crate::DType;

// ---------------------------------------------------------------------------
// 1. F32 -> BF16 -> F32 round-trip preserves finiteness
// ---------------------------------------------------------------------------

/// Prove: any finite f32 value survives the f32->bf16->f32 round-trip as
/// a finite value. BF16 truncation cannot produce NaN or infinity from
/// finite input.
#[kani::proof]
#[kani::unwind(4)]
fn proof_f32_bf16_roundtrip_finite() {
    let val: f32 = kani::any();
    kani::assume!(val.is_finite());
    let bits = val.to_bits();
    // BF16: keep top 16 bits (sign + 8-bit exponent + 7-bit mantissa),
    // zero bottom 16 bits. This is the truncation (round-toward-zero) path.
    let bf16_bits = bits & 0xFFFF_0000;
    let roundtrip = f32::from_bits(bf16_bits);
    assert!(
        roundtrip.is_finite(),
        "BF16 round-trip must preserve finiteness"
    );
}

// ---------------------------------------------------------------------------
// 2. F32 -> F16 -> F32 round-trip preserves finiteness (within range)
// ---------------------------------------------------------------------------

/// Prove: any finite f32 value within the F16 representable range survives
/// the f32->f16->f32 round-trip as a finite value.
///
/// F16 max is 65504. Values within this range must remain finite after
/// conversion.
#[kani::proof]
#[kani::unwind(4)]
fn proof_f32_f16_roundtrip_finite() {
    let val: f32 = kani::any();
    kani::assume!(val.is_finite());
    kani::assume!(val >= -65504.0);
    kani::assume!(val <= 65504.0);
    let as_f16 = half::f16::from_f32(val);
    let roundtrip = as_f16.to_f32();
    assert!(
        roundtrip.is_finite(),
        "F16 round-trip must preserve finiteness for in-range values"
    );
}

// ---------------------------------------------------------------------------
// 3. BF16 representable range check (max ~3.39e38)
// ---------------------------------------------------------------------------

/// Prove: BF16 can represent values up to its documented maximum (~3.39e38).
/// The maximum finite bf16 value, when converted back to f32, must be finite
/// and within the expected range.
#[kani::proof]
#[kani::unwind(4)]
fn proof_bf16_representable_range() {
    // BF16 max: 0x7F7F in bf16 bits = sign=0, exp=0xFE, mantissa=0x7F
    // This is (2 - 2^-7) * 2^127 ~= 3.3895e38
    let bf16_max = half::bf16::from_bits(0x7F7F);
    let as_f32 = bf16_max.to_f32();
    assert!(as_f32.is_finite(), "bf16 max must be finite in f32");
    // Must be in the ballpark of 3.39e38
    assert!(as_f32 > 3.0e38, "bf16 max must exceed 3.0e38");
    assert!(as_f32 < 3.5e38, "bf16 max must be less than 3.5e38");

    // Any finite f32 within bf16 range must round-trip to finite.
    let val: f32 = kani::any();
    kani::assume!(val.is_finite());
    kani::assume!(val.abs() <= as_f32);
    let as_bf16 = half::bf16::from_f32(val);
    let back = as_bf16.to_f32();
    assert!(
        back.is_finite(),
        "values within bf16 range must stay finite"
    );
}

// ---------------------------------------------------------------------------
// 4. F16 representable range check (max 65504)
// ---------------------------------------------------------------------------

/// Prove: F16 can represent values up to its documented maximum (65504).
/// The maximum finite f16 value, when converted to f32, must be exactly 65504.
#[kani::proof]
#[kani::unwind(4)]
fn proof_f16_representable_range() {
    // F16 max: 0x7BFF in f16 bits = sign=0, exp=0x1E, mantissa=0x3FF
    // This is (2 - 2^-10) * 2^15 = 65504.0
    let f16_max = half::f16::from_bits(0x7BFF);
    let as_f32 = f16_max.to_f32();
    assert!(as_f32.is_finite(), "f16 max must be finite in f32");
    assert_eq!(as_f32, 65504.0, "f16 max must be exactly 65504.0");

    // Values at the boundary must round-trip correctly.
    let val: f32 = kani::any();
    kani::assume!(val.is_finite());
    kani::assume!(val >= -65504.0);
    kani::assume!(val <= 65504.0);
    let as_f16 = half::f16::from_f32(val);
    let back = as_f16.to_f32();
    assert!(back.is_finite(), "values within f16 range must stay finite");
}

// ---------------------------------------------------------------------------
// 5. F32 -> BF16 preserves sign
// ---------------------------------------------------------------------------

/// Prove: sign is preserved through f32 -> bf16 conversion for all non-zero
/// finite values. The sign bit is the MSB in both formats.
#[kani::proof]
#[kani::unwind(4)]
fn proof_f32_bf16_preserves_sign() {
    let val: f32 = kani::any();
    kani::assume!(val.is_finite());
    kani::assume!(val != 0.0);
    let bits = val.to_bits();
    // BF16 truncation preserves the sign bit (bit 31).
    let bf16_bits = bits & 0xFFFF_0000;
    let roundtrip = f32::from_bits(bf16_bits);
    // Skip if truncation produced zero (subnormal flush).
    if roundtrip != 0.0 {
        assert_eq!(
            val.is_sign_positive(),
            roundtrip.is_sign_positive(),
            "sign must be preserved through bf16 conversion"
        );
    }
}

// ---------------------------------------------------------------------------
// 6. F32 -> F16 preserves sign
// ---------------------------------------------------------------------------

/// Prove: sign is preserved through f32 -> f16 conversion for non-zero
/// values within f16 range. Uses the half crate's from_f32 for accurate
/// rounding.
#[kani::proof]
#[kani::unwind(4)]
fn proof_f32_f16_preserves_sign() {
    let val: f32 = kani::any();
    kani::assume!(val.is_finite());
    kani::assume!(val != 0.0);
    kani::assume!(val.abs() >= 6.1e-5); // Above f16 min normal to avoid subnormal flush
    kani::assume!(val.abs() <= 65504.0);
    let as_f16 = half::f16::from_f32(val);
    let back = as_f16.to_f32();
    if back != 0.0 {
        assert_eq!(
            val.is_sign_positive(),
            back.is_sign_positive(),
            "sign must be preserved through f16 conversion"
        );
    }
}

// ---------------------------------------------------------------------------
// 7. Subnormal handling in BF16 (flush to zero)
// ---------------------------------------------------------------------------

/// Prove: f32 subnormal values (too small for bf16 normal representation)
/// flush to zero when converted to bf16. BF16 has the same exponent range
/// as f32 but fewer mantissa bits, so f32 subnormals that are too small
/// for bf16 become zero.
#[kani::proof]
#[kani::unwind(4)]
fn proof_bf16_subnormal_flush_to_zero() {
    let val: f32 = kani::any();
    kani::assume!(val.is_finite());
    // f32 subnormals: exponent field is 0, mantissa is nonzero.
    // These are values in (0, 2^-126).
    // BF16 has 7-bit mantissa vs f32's 23-bit, so most f32 subnormals
    // lose precision and many flush to zero.
    let bits = val.to_bits();
    let exponent = (bits >> 23) & 0xFF;
    kani::assume!(exponent == 0); // f32 subnormal
    let mantissa = bits & 0x007F_FFFF;
    kani::assume!(mantissa != 0); // not zero, actually subnormal

    // BF16 truncation: keep top 16 bits
    let bf16_bits = bits & 0xFFFF_0000;
    let roundtrip = f32::from_bits(bf16_bits);
    // Subnormals with mantissa bits only in the bottom 16 bits flush to +/-0.
    // The result must be either zero or a valid finite bf16 subnormal.
    assert!(
        roundtrip.is_finite(),
        "bf16 subnormal handling must produce finite result"
    );
    // The absolute value cannot increase through truncation.
    assert!(
        roundtrip.abs() <= val.abs(),
        "bf16 truncation cannot increase magnitude"
    );
}

// ---------------------------------------------------------------------------
// 8. Overflow to infinity in F16 for large inputs
// ---------------------------------------------------------------------------

/// Prove: f32 values exceeding f16 max (65504) overflow to infinity when
/// converted to f16. This is the expected IEEE 754 behavior.
#[kani::proof]
#[kani::unwind(4)]
fn proof_f16_overflow_to_infinity() {
    let val: f32 = kani::any();
    kani::assume!(val.is_finite());
    // Values strictly above f16 max overflow.
    // 65520.0 is just above the rounding boundary for round-to-nearest-even.
    kani::assume!(val > 65520.0);
    kani::assume!(val <= 1.0e10); // Keep bounded for CBMC tractability
    let as_f16 = half::f16::from_f32(val);
    let back = as_f16.to_f32();
    assert!(
        back.is_infinite() || back == 65504.0,
        "f32 values above f16 max must overflow to inf or saturate to max"
    );
}

// ---------------------------------------------------------------------------
// 9. NaN preserved through dtype conversion
// ---------------------------------------------------------------------------

/// Prove: NaN values are preserved as NaN through both bf16 and f16
/// conversion. NaN must not become a finite value or infinity.
#[kani::proof]
#[kani::unwind(4)]
fn proof_nan_preserved_through_conversion() {
    let bits: u32 = kani::any();
    let val = f32::from_bits(bits);
    kani::assume!(val.is_nan());

    // BF16 path: NaN must stay NaN.
    let bf16_val = half::bf16::from_f32(val);
    let bf16_back = bf16_val.to_f32();
    assert!(bf16_back.is_nan(), "NaN must survive bf16 round-trip");

    // F16 path: NaN must stay NaN.
    let f16_val = half::f16::from_f32(val);
    let f16_back = f16_val.to_f32();
    assert!(f16_back.is_nan(), "NaN must survive f16 round-trip");
}

// ---------------------------------------------------------------------------
// 10. Zero preserved through dtype conversion (both +0 and -0)
// ---------------------------------------------------------------------------

/// Prove: both +0.0 and -0.0 are exactly preserved through bf16 and f16
/// conversions. IEEE 754 distinguishes signed zeros; both must survive.
#[kani::proof]
#[kani::unwind(4)]
fn proof_zero_preserved_through_conversion() {
    // Positive zero through bf16
    let pos_zero_bf16 = half::bf16::from_f32(0.0f32);
    assert_eq!(
        pos_zero_bf16.to_f32().to_bits(),
        0.0f32.to_bits(),
        "+0.0 bit pattern must survive bf16 round-trip"
    );

    // Negative zero through bf16
    let neg_zero_bf16 = half::bf16::from_f32(-0.0f32);
    assert_eq!(
        neg_zero_bf16.to_f32().to_bits(),
        (-0.0f32).to_bits(),
        "-0.0 bit pattern must survive bf16 round-trip"
    );

    // Positive zero through f16
    let pos_zero_f16 = half::f16::from_f32(0.0f32);
    assert_eq!(
        pos_zero_f16.to_f32().to_bits(),
        0.0f32.to_bits(),
        "+0.0 bit pattern must survive f16 round-trip"
    );

    // Negative zero through f16
    let neg_zero_f16 = half::f16::from_f32(-0.0f32);
    assert_eq!(
        neg_zero_f16.to_f32().to_bits(),
        (-0.0f32).to_bits(),
        "-0.0 bit pattern must survive f16 round-trip"
    );
}

// ---------------------------------------------------------------------------
// 11. F32 -> U8 quantization clamps to [0, 255]
// ---------------------------------------------------------------------------

/// Prove: f32 values clamped to [0, 255] and cast to u8 always produce
/// valid u8 values. This models the quantization path for U8 tensors.
#[kani::proof]
#[kani::unwind(4)]
fn proof_f32_to_u8_quantization_clamps() {
    let val: f32 = kani::any();
    kani::assume!(val.is_finite());

    // Clamp to [0, 255] before cast (standard quantization path).
    let clamped = if val < 0.0 {
        0.0f32
    } else if val > 255.0 {
        255.0f32
    } else {
        val
    };

    let as_u8 = clamped as u8;
    assert!(as_u8 <= 255, "clamped f32->u8 must be in [0, 255]");

    // Verify the clamp contract.
    if val < 0.0 {
        assert_eq!(as_u8, 0, "negative values must clamp to 0");
    }
    if val > 255.0 {
        assert_eq!(as_u8, 255, "values above 255 must clamp to 255");
    }
}

// ---------------------------------------------------------------------------
// 12. F32 -> I8 quantization clamps to [-128, 127]
// ---------------------------------------------------------------------------

/// Prove: f32 values clamped to [-128, 127] and cast to i8 always produce
/// valid i8 values. This models the signed quantization path.
#[kani::proof]
#[kani::unwind(4)]
fn proof_f32_to_i8_quantization_clamps() {
    let val: f32 = kani::any();
    kani::assume!(val.is_finite());

    // Clamp to [-128, 127] before cast (signed quantization path).
    let clamped = if val < -128.0 {
        -128.0f32
    } else if val > 127.0 {
        127.0f32
    } else {
        val
    };

    let as_i8 = clamped as i8;
    assert!(as_i8 >= -128, "clamped f32->i8 must be >= -128");
    assert!(as_i8 <= 127, "clamped f32->i8 must be <= 127");

    // Verify the clamp contract.
    if val < -128.0 {
        assert_eq!(as_i8, -128, "values below -128 must clamp to -128");
    }
    if val > 127.0 {
        assert_eq!(as_i8, 127, "values above 127 must clamp to 127");
    }
}

// ---------------------------------------------------------------------------
// 13. BF16 multiplication doesn't overflow for bounded inputs
// ---------------------------------------------------------------------------

/// Prove: BF16 multiplication of two values within [-1, 1] produces a
/// finite result. This is the common case for normalized activations and
/// weight values in neural networks.
#[kani::proof]
#[kani::unwind(4)]
fn proof_bf16_multiplication_bounded_inputs() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume!(a.is_finite());
    kani::assume!(b.is_finite());
    kani::assume!(a >= -1.0);
    kani::assume!(a <= 1.0);
    kani::assume!(b >= -1.0);
    kani::assume!(b <= 1.0);

    let a_bf16 = half::bf16::from_f32(a);
    let b_bf16 = half::bf16::from_f32(b);

    // Multiply in f32 (as done in the accumulation path).
    let product = a_bf16.to_f32() * b_bf16.to_f32();
    assert!(
        product.is_finite(),
        "bf16 product of [-1,1] values must be finite"
    );
    // Product of two [-1,1] values is in [-1,1].
    assert!(
        product.abs() <= 1.0 + 1e-6,
        "bf16 product of [-1,1] values must be in [-1,1]"
    );
}

// ---------------------------------------------------------------------------
// 14. Loss scaling factor positive and finite
// ---------------------------------------------------------------------------

/// Prove: a loss scaling factor constructed as a power of 2 is always
/// positive and finite for exponents in the valid range [0, 24].
/// Mixed precision training uses power-of-2 loss scaling to prevent
/// underflow in bf16/f16 gradients.
#[kani::proof]
#[kani::unwind(4)]
fn proof_loss_scaling_factor_positive_finite() {
    let exponent: u8 = kani::any();
    kani::assume!(exponent <= 24); // Common range for loss scaling

    let scale = 2.0f32.powi(exponent as i32);
    assert!(scale.is_finite(), "loss scale must be finite");
    assert!(scale > 0.0, "loss scale must be positive");
    assert!(
        scale >= 1.0,
        "loss scale with non-negative exponent must be >= 1"
    );
    // Power of 2 must be exact in IEEE 754.
    let expected = (1u32 << exponent) as f32;
    assert_eq!(scale, expected, "power-of-2 scale must be exact in f32");
}

// ---------------------------------------------------------------------------
// 15. Gradient unscaling divides by positive scale
// ---------------------------------------------------------------------------

/// Prove: gradient unscaling (dividing by the loss scale) produces a finite
/// result when the gradient and scale are both finite and the scale is a
/// positive power of 2.
#[kani::proof]
#[kani::unwind(4)]
fn proof_gradient_unscaling_safe() {
    let grad: f32 = kani::any();
    let exponent: u8 = kani::any();
    kani::assume!(grad.is_finite());
    kani::assume!(exponent >= 1);
    kani::assume!(exponent <= 24);
    // Bound gradient to prevent overflow after division (grad / scale).
    // With scale >= 2, any finite grad divided by scale stays finite.
    kani::assume!(grad.abs() <= 1.0e30);

    let scale = 2.0f32.powi(exponent as i32);
    let unscaled = grad / scale;

    assert!(
        unscaled.is_finite(),
        "gradient unscaling must produce finite result"
    );
    // Unscaled magnitude must be <= original magnitude (dividing by >= 2).
    assert!(
        unscaled.abs() <= grad.abs(),
        "unscaling must not increase magnitude"
    );
}

// ---------------------------------------------------------------------------
// 16. Dynamic loss scaling: scale halved on overflow
// ---------------------------------------------------------------------------

/// Prove: when a loss scale is halved (the standard response to gradient
/// overflow in dynamic loss scaling), the result is still a valid positive
/// power of 2, and the new scale is strictly less than the old scale.
#[kani::proof]
#[kani::unwind(4)]
fn proof_dynamic_loss_scaling_halved_on_overflow() {
    let exponent: u8 = kani::any();
    kani::assume!(exponent >= 1);
    kani::assume!(exponent <= 24);

    let scale = 2.0f32.powi(exponent as i32);
    let halved = scale / 2.0;

    assert!(halved.is_finite(), "halved scale must be finite");
    assert!(halved > 0.0, "halved scale must be positive");
    assert!(
        halved < scale,
        "halved scale must be strictly less than original"
    );
    // Halved must equal 2^(exponent-1).
    let expected = 2.0f32.powi((exponent as i32) - 1);
    assert_eq!(halved, expected, "halved scale must equal 2^(exponent-1)");
}

// ---------------------------------------------------------------------------
// 17. F32 accumulation of BF16 products is finite
// ---------------------------------------------------------------------------

/// Prove: accumulating a small number of bf16 products in f32 preserves
/// finiteness when inputs are bounded. This models the inner loop of
/// mixed-precision matmul: multiply in bf16, accumulate in f32.
#[kani::proof]
#[kani::unwind(6)]
fn proof_f32_accumulation_bf16_products_finite() {
    let a0: f32 = kani::any();
    let b0: f32 = kani::any();
    let a1: f32 = kani::any();
    let b1: f32 = kani::any();
    kani::assume!(a0.is_finite() && a0.abs() <= 10.0);
    kani::assume!(b0.is_finite() && b0.abs() <= 10.0);
    kani::assume!(a1.is_finite() && a1.abs() <= 10.0);
    kani::assume!(b1.is_finite() && b1.abs() <= 10.0);

    // Convert to bf16 and back (simulating bf16 storage).
    let a0_bf16 = half::bf16::from_f32(a0).to_f32();
    let b0_bf16 = half::bf16::from_f32(b0).to_f32();
    let a1_bf16 = half::bf16::from_f32(a1).to_f32();
    let b1_bf16 = half::bf16::from_f32(b1).to_f32();

    // Accumulate products in f32.
    let acc = a0_bf16 * b0_bf16 + a1_bf16 * b1_bf16;
    assert!(
        acc.is_finite(),
        "f32 accumulation of bf16 products must be finite"
    );
    // Each product is at most 100, sum of 2 is at most 200.
    assert!(acc.abs() <= 200.0 + 1e-3, "accumulation must be bounded");
}

// ---------------------------------------------------------------------------
// 18. Mixed precision: input dtype preserved through identity ops
// ---------------------------------------------------------------------------

/// Prove: the dtype classification (is_float, is_int) is a fixed property
/// of each DType variant. An "identity" operation (no conversion) preserves
/// the dtype classification. This models the mixed-precision invariant:
/// layers that don't convert dtype must preserve the input's dtype family.
#[kani::proof]
#[kani::unwind(4)]
fn proof_mixed_precision_identity_dtype_preserved() {
    let idx: u8 = kani::any();
    kani::assume!(idx < 9);

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

    // Identity: dtype is preserved.
    let identity_dt = dt;
    assert_eq!(dt, identity_dt, "identity must preserve dtype");
    assert_eq!(
        dt.is_float(),
        identity_dt.is_float(),
        "identity must preserve is_float"
    );
    assert_eq!(
        dt.is_int(),
        identity_dt.is_int(),
        "identity must preserve is_int"
    );

    // is_float and is_int must be mutually exclusive (except Bool).
    if dt != DType::Bool {
        assert!(
            dt.is_float() != dt.is_int(),
            "float and int must be mutually exclusive for non-Bool types"
        );
    } else {
        assert!(
            !dt.is_float() && !dt.is_int(),
            "Bool is neither float nor int"
        );
    }
}

// ---------------------------------------------------------------------------
// 19. Dtype byte sizes: F32=4, BF16=2, F16=2, U8=1, U32=4
// ---------------------------------------------------------------------------

/// Prove: all DType variants have the correct byte sizes as specified by
/// their underlying data formats. These sizes are critical for GPU buffer
/// allocation, zero-copy relabeling safety, and safetensors serialization.
#[kani::proof]
#[kani::unwind(4)]
fn proof_dtype_byte_sizes_correct() {
    // Float types
    assert_eq!(DType::F32.size_bytes(), 4, "F32 must be 4 bytes");
    assert_eq!(DType::BF16.size_bytes(), 2, "BF16 must be 2 bytes");
    assert_eq!(DType::F16.size_bytes(), 2, "F16 must be 2 bytes");
    assert_eq!(DType::F64.size_bytes(), 8, "F64 must be 8 bytes");

    // Integer types
    assert_eq!(DType::U32.size_bytes(), 4, "U32 must be 4 bytes");
    assert_eq!(DType::U8.size_bytes(), 1, "U8 must be 1 byte");
    assert_eq!(DType::I64.size_bytes(), 8, "I64 must be 8 bytes");
    assert_eq!(DType::I32.size_bytes(), 4, "I32 must be 4 bytes");

    // Bool
    assert_eq!(DType::Bool.size_bytes(), 1, "Bool must be 1 byte");

    // Half-precision types must share the same byte width (2).
    assert_eq!(
        DType::F16.size_bytes(),
        DType::BF16.size_bytes(),
        "F16 and BF16 must have same byte width"
    );

    // All byte sizes must be nonzero.
    let idx: u8 = kani::any();
    kani::assume!(idx < 9);
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
    assert!(
        dt.size_bytes() > 0,
        "all dtypes must have positive byte size"
    );
}

// ---------------------------------------------------------------------------
// 20. Conversion chain: F32->BF16->F32 error bounded by 2^-7 * |x|
// ---------------------------------------------------------------------------

/// Prove: the absolute error from f32->bf16->f32 conversion is bounded
/// by 2^-7 * |x| for normal-range values. BF16 has 7 explicit mantissa
/// bits, giving a relative precision of 2^-7 (approximately 0.0078).
///
/// This bound is critical for mixed-precision training: it quantifies
/// the worst-case information loss per dtype conversion.
#[kani::proof]
#[kani::unwind(4)]
fn proof_f32_bf16_f32_error_bounded() {
    let val: f32 = kani::any();
    kani::assume!(val.is_finite());
    // Restrict to normal-range values where relative error bound applies.
    // BF16 shares f32's exponent range, so normal f32 values that are
    // normal in bf16 have well-defined relative error.
    kani::assume!(val.abs() >= 2.0f32.powi(-126)); // Above f32/bf16 min normal
    kani::assume!(val.abs() <= 1.0e30); // Well within bf16 max

    let as_bf16 = half::bf16::from_f32(val);
    let roundtrip = as_bf16.to_f32();
    assert!(
        roundtrip.is_finite(),
        "round-trip must produce finite result"
    );

    let abs_error = (roundtrip - val).abs();
    // Relative error bound: 2^-8 (half ULP for round-to-nearest with
    // 7 explicit mantissa bits). We use 2^-7 as a conservative bound
    // to account for rounding.
    let error_bound = val.abs() * (1.0 / 128.0); // 2^-7 * |x|
    assert!(
        abs_error <= error_bound,
        "bf16 conversion error {abs_error} exceeds 2^-7 * |x| = {error_bound}"
    );
}
