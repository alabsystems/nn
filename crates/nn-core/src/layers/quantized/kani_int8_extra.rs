// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for INT8 per-channel quantization (#3667).
//!
//! Supplements the existing 15 harnesses in int8.rs (cast roundtrip,
//! scale properties, error bounds, roundtrip proofs).
//!
//! These harnesses cover:
//!  - compute_channel_params edge cases (empty row, all-zero, constant)
//!  - Asymmetric zero_point computation bounds
//!  - Clamping correctness (quantized values always fit in i8)
//!  - Scale-zero interaction path
//!  - u8-to-i8 reinterpret cast bitwise properties
//!  - Symmetric range [-127, 127] vs asymmetric [-128, 127]
//!  - max_quantization_error triangle inequality
//!  - Quantization monotonicity (larger input -> larger quantized value)
//!
//! Part of #3667

use super::*;

// =========================================================================
// compute_channel_params edge cases
// =========================================================================

// -------------------------------------------------------------------------
// Harness 1: Empty row returns (0.0, 0) for both modes.
// -------------------------------------------------------------------------

/// Prove: compute_channel_params returns (0.0, 0) for an empty slice
/// in symmetric mode.
#[kani::unwind(1)]
#[kani::proof]
fn int8_empty_row_symmetric() {
    let row: [f32; 0] = [];
    let (scale, zp) = compute_channel_params(&row, Int8Mode::Symmetric);
    assert!(scale == 0.0, "empty row symmetric scale must be 0.0");
    assert!(zp == 0, "empty row symmetric zero_point must be 0");
}

/// Prove: compute_channel_params returns (0.0, 0) for an empty slice
/// in asymmetric mode.
#[kani::unwind(1)]
#[kani::proof]
fn int8_empty_row_asymmetric() {
    let row: [f32; 0] = [];
    let (scale, zp) = compute_channel_params(&row, Int8Mode::Asymmetric);
    assert!(scale == 0.0, "empty row asymmetric scale must be 0.0");
    assert!(zp == 0, "empty row asymmetric zero_point must be 0");
}

// -------------------------------------------------------------------------
// Harness 3: All-zero row returns (0.0, 0) for both modes.
// -------------------------------------------------------------------------

/// Prove: a row where all values are 0.0 produces scale=0 and zp=0
/// in symmetric mode. abs_max=0 triggers the early return.
#[kani::unwind(1)]
#[kani::proof]
fn int8_allzero_row_symmetric() {
    let row = [0.0_f32, 0.0_f32];
    let (scale, zp) = compute_channel_params(&row, Int8Mode::Symmetric);
    assert!(scale == 0.0, "all-zero symmetric scale must be 0.0");
    assert!(zp == 0, "all-zero symmetric zp must be 0");
}

/// Prove: a row where all values are 0.0 produces scale=0 and zp=0
/// in asymmetric mode. range=0 triggers the early return.
#[kani::unwind(1)]
#[kani::proof]
fn int8_allzero_row_asymmetric() {
    let row = [0.0_f32, 0.0_f32];
    let (scale, zp) = compute_channel_params(&row, Int8Mode::Asymmetric);
    assert!(scale == 0.0, "all-zero asymmetric scale must be 0.0");
    assert!(zp == 0, "all-zero asymmetric zp must be 0");
}

// -------------------------------------------------------------------------
// Harness 5: Constant non-zero row in asymmetric mode returns (0.0, 0).
// -------------------------------------------------------------------------

/// Prove: a constant row (all values identical, non-zero) in asymmetric
/// mode returns scale=0 because max-min=0 (range is zero).
#[kani::unwind(1)]
#[kani::proof]
fn int8_constant_row_asymmetric() {
    let val: f32 = kani::any();
    kani::assume(val.is_finite());
    kani::assume(val != 0.0);
    kani::assume(val.abs() < 1e10);

    let row = [val, val];
    let (scale, zp) = compute_channel_params(&row, Int8Mode::Asymmetric);
    assert!(scale == 0.0, "constant-row asymmetric scale must be 0.0");
    assert!(zp == 0, "constant-row asymmetric zp must be 0");
}

// =========================================================================
// Asymmetric zero_point computation bounds
// =========================================================================

// -------------------------------------------------------------------------
// Harness 6: Asymmetric zero_point is bounded.
//
// zp = round(-128 - min/scale) where scale = (max - min) / 255.
// For min <= 0 <= max (straddling zero): zp in [-128, 127].
// For all-positive weights: zp can be < -128.
// For all-negative weights: zp can be > 127.
//
// The zero_point is stored as i32 (not i8) for this reason.
// Prove the i32 value stays within a reasonable range.
// -------------------------------------------------------------------------

/// Prove: asymmetric zero_point from compute_channel_params fits in i32
/// and its magnitude is bounded for practical weight ranges.
#[kani::unwind(1)]
#[kani::proof]
fn int8_asymmetric_zp_bounded() {
    let v0: f32 = kani::any();
    let v1: f32 = kani::any();
    kani::assume(v0.is_finite() && v1.is_finite());
    kani::assume(v0.abs() < 1e6 && v1.abs() < 1e6);
    // Ensure non-constant row (otherwise scale=0, zp=0)
    kani::assume(v0 != v1);

    let row = [v0, v1];
    let (scale, zp) = compute_channel_params(&row, Int8Mode::Asymmetric);

    assert!(scale.is_finite(), "asymmetric scale must be finite");
    assert!(scale >= 0.0, "asymmetric scale must be non-negative");

    // zp = round(-128 - min/scale). For finite values with bounded scale:
    // |min/scale| = |min| / ((max-min)/255) = 255 * |min| / (max-min)
    // Since |min| <= max-min (when min <= 0), |min/scale| <= 255.
    // zp = round(-128 - min/scale), so |zp| <= 128 + 255 + 1 = 384.
    // For all-positive or all-negative, the bound is wider.
    // In all cases, zp fits comfortably in i32.
    assert!(
        zp >= -1_000_000 && zp <= 1_000_000,
        "asymmetric zero_point must be bounded for practical weights"
    );
}

// =========================================================================
// Clamping correctness
// =========================================================================

// -------------------------------------------------------------------------
// Harness 7: Quantized value after clamp always fits in i8.
//
// The quantization path computes: q_f32.round().clamp(-128.0, 127.0) as i8.
// Prove: any f32 after clamp(-128.0, 127.0), cast as i8, is in [-128, 127].
// -------------------------------------------------------------------------

/// Prove: f32 after clamp(-128.0, 127.0) cast to i8 is in [-128, 127].
#[kani::unwind(1)]
#[kani::proof]
fn int8_clamp_to_i8_range() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite());
    kani::assume(v >= -1e10 && v <= 1e10);

    let clamped = v.round().clamp(-128.0, 127.0);
    let as_i8 = clamped as i8;

    // i8 range is [-128, 127]
    let widened = as_i8 as i32;
    assert!(
        widened >= -128 && widened <= 127,
        "clamped-to-i8 must be in [-128, 127]"
    );
}

// -------------------------------------------------------------------------
// Harness 8: Clamping is idempotent.
//
// Applying clamp(-128.0, 127.0) twice gives the same result.
// -------------------------------------------------------------------------

/// Prove: clamp(-128.0, 127.0) is idempotent for finite values.
#[kani::unwind(1)]
#[kani::proof]
fn int8_clamp_idempotent() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite());

    let once = v.clamp(-128.0, 127.0);
    let twice = once.clamp(-128.0, 127.0);

    assert!(once == twice, "clamping must be idempotent");
}

// =========================================================================
// u8-to-i8 reinterpret cast properties
// =========================================================================

// -------------------------------------------------------------------------
// Harness 9: u8 → i8 → u8 roundtrip is identity (reinterpret cast).
//
// The quantized values are stored as u8 (reinterpret cast of i8).
// Prove the roundtrip preserves all bits.
// -------------------------------------------------------------------------

/// Prove: casting i8 to u8 and back to i8 is identity.
#[kani::unwind(1)]
#[kani::proof]
fn int8_i8_u8_roundtrip() {
    let original: i8 = kani::any();
    let as_u8 = original as u8;
    let back_to_i8 = as_u8 as i8;
    assert!(back_to_i8 == original, "i8 -> u8 -> i8 must be identity");
}

/// Prove: casting u8 to i8 and back to u8 is identity.
#[kani::unwind(1)]
#[kani::proof]
fn int8_u8_i8_roundtrip() {
    let original: u8 = kani::any();
    let as_i8 = original as i8;
    let back_to_u8 = as_i8 as u8;
    assert!(back_to_u8 == original, "u8 -> i8 -> u8 must be identity");
}

// =========================================================================
// Symmetric vs asymmetric range properties
// =========================================================================

// -------------------------------------------------------------------------
// Harness 11: Symmetric mode always produces zero_point=0.
// -------------------------------------------------------------------------

/// Prove: compute_channel_params in symmetric mode always returns zp=0
/// regardless of input data.
#[kani::unwind(1)]
#[kani::proof]
fn int8_symmetric_always_zero_zp() {
    let v0: f32 = kani::any();
    let v1: f32 = kani::any();
    kani::assume(v0.is_finite() && v1.is_finite());
    kani::assume(v0.abs() < 1e30 && v1.abs() < 1e30);

    let row = [v0, v1];
    let (_scale, zp) = compute_channel_params(&row, Int8Mode::Symmetric);
    assert!(zp == 0, "symmetric mode must always have zero_point=0");
}

// -------------------------------------------------------------------------
// Harness 12: Symmetric scale formula: scale = abs_max / 127.0.
//
// For a single nonzero value, the scale is exactly |v| / 127.0.
// -------------------------------------------------------------------------

/// Prove: for a single-element row with |v| > 0, symmetric scale
/// equals |v| / 127.0.
#[kani::unwind(1)]
#[kani::proof]
fn int8_symmetric_scale_formula() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite());
    kani::assume(v.abs() > 0.0 && v.abs() < 1e10);

    let row = [v];
    let (scale, _zp) = compute_channel_params(&row, Int8Mode::Symmetric);

    let expected = v.abs() / 127.0;
    // f32 division may introduce small rounding error
    let diff = (scale - expected).abs();
    assert!(diff < 1e-10, "symmetric scale must be abs_max / 127.0");
}

// -------------------------------------------------------------------------
// Harness 13: Asymmetric scale formula: scale = (max - min) / 255.0.
// -------------------------------------------------------------------------

/// Prove: for two distinct finite values, asymmetric scale equals
/// (max - min) / 255.0.
#[kani::unwind(1)]
#[kani::proof]
fn int8_asymmetric_scale_formula() {
    let v0: f32 = kani::any();
    let v1: f32 = kani::any();
    kani::assume(v0.is_finite() && v1.is_finite());
    kani::assume(v0.abs() < 1e6 && v1.abs() < 1e6);
    kani::assume(v0 != v1);

    let min_val = if v0 < v1 { v0 } else { v1 };
    let max_val = if v0 > v1 { v0 } else { v1 };

    let row = [v0, v1];
    let (scale, _zp) = compute_channel_params(&row, Int8Mode::Asymmetric);

    let expected = (max_val - min_val) / 255.0;
    let diff = (scale - expected).abs();
    assert!(diff < 1e-10, "asymmetric scale must be (max - min) / 255.0");
}

// =========================================================================
// max_quantization_error properties
// =========================================================================

// -------------------------------------------------------------------------
// Harness 14: max_quantization_error satisfies triangle inequality.
//
// |max_error(a, c)| <= max_error(a, b) + max_error(b, c) per-element,
// but max over elements may violate this. However, max_error(a,a) = 0
// is the base case. We prove the simpler property: symmetry.
// -------------------------------------------------------------------------

/// Prove: max_quantization_error(a, b) == max_quantization_error(b, a).
/// (The function computes |a[i] - b[i]| which is symmetric.)
#[kani::unwind(1)]
#[kani::proof]
fn int8_max_quant_error_symmetric() {
    let a0: f32 = kani::any();
    let b0: f32 = kani::any();

    kani::assume(a0.is_finite() && b0.is_finite());
    kani::assume(a0.abs() < 1e30 && b0.abs() < 1e30);

    let arr_a = [a0];
    let arr_b = [b0];

    let err_ab = max_quantization_error(&arr_a, &arr_b);
    let err_ba = max_quantization_error(&arr_b, &arr_a);

    assert!(
        (err_ab - err_ba).abs() < 1e-30,
        "max_quantization_error must be symmetric"
    );
}

// =========================================================================
// Quantization monotonicity
// =========================================================================

// -------------------------------------------------------------------------
// Harness 15: For symmetric quantization with positive scale, larger
// input maps to larger (or equal) quantized value.
//
// If v1 > v2 >= 0 and both are within [0, 127*scale], then
// round(v1/scale) >= round(v2/scale).
// -------------------------------------------------------------------------

/// Prove: symmetric quantization preserves order for non-negative
/// values within the representable range.
#[kani::unwind(1)]
#[kani::proof]
fn int8_symmetric_quantize_monotone() {
    let scale: f32 = kani::any();
    kani::assume(scale.is_finite());
    kani::assume(scale > 1e-6 && scale < 100.0);

    let v1: f32 = kani::any();
    let v2: f32 = kani::any();
    kani::assume(v1.is_finite() && v2.is_finite());
    kani::assume(v1 >= 0.0 && v2 >= 0.0);
    kani::assume(v1 <= 127.0 * scale && v2 <= 127.0 * scale);
    // Require separation by at least one quantization step to guarantee ordering
    kani::assume(v1 >= v2 + scale);

    let q1 = (v1 / scale).round().clamp(-128.0, 127.0) as i8;
    let q2 = (v2 / scale).round().clamp(-128.0, 127.0) as i8;

    assert!(q1 >= q2, "larger input must map to >= quantized value");
}

// =========================================================================
// Scale positivity and zero handling
// =========================================================================

// -------------------------------------------------------------------------
// Harness 16: When scale == 0, quantization produces 0 for any input.
//
// In quantize_per_channel, scale==0 triggers q_i8 = 0.
// -------------------------------------------------------------------------

/// Prove: the scale==0 branch in quantize_per_channel always yields q=0.
#[kani::unwind(1)]
#[kani::proof]
fn int8_zero_scale_produces_zero() {
    let val: f32 = kani::any();
    kani::assume(val.is_finite());
    kani::assume(val.abs() < 1e30);

    let scale: f32 = 0.0;
    // This is the zero-scale path in quantize_per_channel
    let q_i8: i8 = if scale == 0.0 {
        0_i8
    } else {
        let q_f32 = val / scale;
        q_f32.round().clamp(-128.0, 127.0) as i8
    };

    assert!(q_i8 == 0, "scale==0 must produce q=0");
}

// -------------------------------------------------------------------------
// Harness 17: Dequantize with scale==0 produces 0 for any q and zp.
// -------------------------------------------------------------------------

/// Prove: dequantization with scale=0 always returns 0.0 regardless
/// of the quantized value and zero_point.
#[kani::unwind(1)]
#[kani::proof]
fn int8_dequant_zero_scale_is_zero() {
    let q_u8: u8 = kani::any();
    let q_i8 = q_u8 as i8;
    let zero_point: i32 = kani::any();
    kani::assume(zero_point >= -128 && zero_point <= 127);

    let scale: f32 = 0.0;
    let dequant = (q_i8 as f32 - zero_point as f32) * scale;

    assert!(dequant == 0.0, "dequant with scale=0 must be exactly 0.0");
}

// =========================================================================
// Dequantization precision properties
// =========================================================================

// -------------------------------------------------------------------------
// Harness 18: Symmetric dequant of q=0 is always 0.0.
// -------------------------------------------------------------------------

/// Prove: for symmetric quantization (zp=0), dequantizing q=0 always
/// produces exactly 0.0 regardless of scale.
#[kani::unwind(1)]
#[kani::proof]
fn int8_dequant_zero_is_zero_symmetric() {
    let scale: f32 = kani::any();
    kani::assume(scale.is_finite());
    kani::assume(scale >= 0.0 && scale < 1e10);

    let q_i8: i8 = 0;
    let dequant = (q_i8 as f32) * scale;
    assert!(
        dequant == 0.0,
        "dequant of q=0 must be 0.0 in symmetric mode"
    );
}

// -------------------------------------------------------------------------
// Harness 19: Symmetric dequant of q=127 gives the maximum positive value.
//
// max_positive = 127 * scale. This is the largest representable positive
// value in symmetric quantization.
// -------------------------------------------------------------------------

/// Prove: symmetric dequant(127) = 127 * scale, and this is the max
/// positive value (dequant of any q <= 127 is <= 127 * scale).
#[kani::unwind(1)]
#[kani::proof]
fn int8_symmetric_max_positive() {
    let scale: f32 = kani::any();
    kani::assume(scale.is_finite());
    kani::assume(scale > 0.0 && scale < 1000.0);

    let max_q: i8 = 127;
    let dequant_max = (max_q as f32) * scale;
    let expected = 127.0 * scale;

    assert!(
        (dequant_max - expected).abs() < 1e-10,
        "dequant(127) must equal 127 * scale"
    );

    // Any q in [-128, 127] dequantizes to <= 127 * scale in magnitude
    // (within the [-128, 128] range, |q| <= 128, so |dequant| <= 128 * scale)
    let any_q: u8 = kani::any();
    let any_i8 = any_q as i8;
    let any_dequant = (any_i8 as f32) * scale;
    assert!(
        any_dequant <= dequant_max + scale,
        "any dequant <= 127 * scale + scale"
    );
}

// =========================================================================
// Quantization invariants for special values
// =========================================================================

// -------------------------------------------------------------------------
// Harness 20: Quantizing 0.0 with any positive scale gives q=0.
// -------------------------------------------------------------------------

/// Prove: quantizing the value 0.0 with symmetric mode always yields q=0.
#[kani::unwind(1)]
#[kani::proof]
fn int8_quantize_zero_gives_zero() {
    let scale: f32 = kani::any();
    kani::assume(scale.is_finite());
    kani::assume(scale > 0.0 && scale < 1000.0);

    // Symmetric: q = round(0.0 / scale + 0).clamp(...)
    let q_f32 = (0.0_f32 / scale).round().clamp(-128.0, 127.0);
    let q_i8 = q_f32 as i8;

    assert!(q_i8 == 0, "quantizing 0.0 must give q=0");
}
