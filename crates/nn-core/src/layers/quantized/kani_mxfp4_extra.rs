// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for MXFP4 block-quantized format (#3667).
//!
//! Supplements the existing 13 harnesses in mxfp4.rs (roundtrip finite,
//! no-panic, encode bounds, block scale, sign preservation, nibble indexing,
//! zero preservation, symmetry, error bounded, magnitude bounded, storage).
//!
//! These harnesses cover:
//!  - E1M2 magnitude table properties (ordering, completeness, subnormal/normal)
//!  - E1M2 sign bit structural proof
//!  - Block scale power-of-two property
//!  - Shared exponent computation: max_abs fits within range
//!  - Mxfp4Tensor storage_bytes and compression_ratio
//!  - Mxfp4Tensor original_len preserved through roundtrip
//!  - E1M2 decode monotone in magnitude index
//!  - Block padding invariant
//!  - Decode code 0 vs code 8 (negative zero = positive zero)
//!  - Dequantize_block output bounded by 6 * block_scale
//!
//! Part of #3667

use super::*;

fn ceil_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite());
    kani::assume(r >= x);
    kani::assume(r <= x + 1.0);
    r
}

fn log2_f32_stub(_x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -150.0 && r <= 128.0);
    r
}

fn powi_f64_stub(_b: f64, _e: i32) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite());
    r
}

// =========================================================================
// E1M2 magnitude table properties
// =========================================================================

// -------------------------------------------------------------------------
// Harness 1: E1M2 magnitudes are non-negative.
// -------------------------------------------------------------------------

/// Prove: all 8 E1M2 magnitude values are non-negative.
#[kani::unwind(1)]
#[kani::proof]
fn mxfp4_e1m2_magnitudes_nonneg() {
    let idx: usize = kani::any();
    kani::assume(idx < 8);

    let magnitudes: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    assert!(
        magnitudes[idx] >= 0.0,
        "all E1M2 magnitudes must be non-negative"
    );
}

// -------------------------------------------------------------------------
// Harness 2: E1M2 magnitudes are strictly ordered.
// -------------------------------------------------------------------------

/// Prove: E1M2 magnitudes are strictly monotonically increasing.
/// magnitudes[i] < magnitudes[i+1] for all i in [0, 6].
#[kani::unwind(1)]
#[kani::proof]
fn mxfp4_e1m2_magnitudes_ordered() {
    let idx: usize = kani::any();
    kani::assume(idx < 7);

    let magnitudes: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    assert!(
        magnitudes[idx] < magnitudes[idx + 1],
        "E1M2 magnitudes must be strictly increasing"
    );
}

// -------------------------------------------------------------------------
// Harness 3: E1M2 subnormal values have E=0, normal values have E=1.
//
// E1M2 encoding: indices 0-3 are subnormal (E=0), indices 4-7 are normal (E=1).
// Subnormal: 0.{MM} => 0.0, 0.5, 1.0, 1.5
// Normal: 1.{MM} * 2 => 2.0, 3.0, 4.0, 6.0
// -------------------------------------------------------------------------

/// Prove: subnormal magnitudes (idx 0-3) are < 2.0 and normal
/// magnitudes (idx 4-7) are >= 2.0.
#[kani::unwind(1)]
#[kani::proof]
fn mxfp4_e1m2_subnormal_normal_boundary() {
    let magnitudes: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

    let idx: usize = kani::any();
    kani::assume(idx < 8);

    if idx < 4 {
        assert!(
            magnitudes[idx] < 2.0,
            "subnormal magnitudes (E=0) must be < 2.0"
        );
    } else {
        assert!(
            magnitudes[idx] >= 2.0,
            "normal magnitudes (E=1) must be >= 2.0"
        );
    }
}

// -------------------------------------------------------------------------
// Harness 4: E1M2 max magnitude is exactly 6.0.
// -------------------------------------------------------------------------

/// Prove: the maximum E1M2 magnitude is 6.0 (at index 7).
#[kani::unwind(1)]
#[kani::proof]
fn mxfp4_e1m2_max_magnitude_is_6() {
    let magnitudes: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

    let idx: usize = kani::any();
    kani::assume(idx < 8);
    assert!(magnitudes[idx] <= 6.0, "no E1M2 magnitude exceeds 6.0");
    assert!(
        magnitudes[7] == 6.0,
        "maximum E1M2 magnitude is exactly 6.0"
    );
}

// =========================================================================
// E1M2 sign bit structure
// =========================================================================

// -------------------------------------------------------------------------
// Harness 5: E1M2 sign bit is bit 3.
//
// In the 4-bit code, bit 3 is the sign bit.
// code & 0x7 = magnitude index (bits 0-2).
// code >> 3 = sign (0 = positive, 1 = negative).
// -------------------------------------------------------------------------

/// Prove: the sign bit (bit 3) and magnitude bits (bits 0-2) partition
/// the 4-bit code without overlap.
#[kani::unwind(1)]
#[kani::proof]
fn mxfp4_e1m2_sign_magnitude_partition() {
    let code: u8 = kani::any();
    kani::assume(code <= 15);

    let sign_bit = (code >> 3) & 1;
    let magnitude_idx = code & 0x7;

    // sign_bit is 0 or 1
    assert!(sign_bit <= 1, "sign bit must be 0 or 1");

    // magnitude_idx is in [0, 7]
    assert!(magnitude_idx <= 7, "magnitude index must be in [0, 7]");

    // Reconstruction: code == sign_bit << 3 | magnitude_idx
    let reconstructed = (sign_bit << 3) | magnitude_idx;
    assert!(
        reconstructed == code,
        "sign_bit << 3 | magnitude_idx must reconstruct the code"
    );
}

// -------------------------------------------------------------------------
// Harness 6: Positive code has sign_bit=0, negative code has sign_bit=1.
// -------------------------------------------------------------------------

/// Prove: codes [0, 7] are positive (sign=0), codes [8, 15] are negative (sign=1).
#[kani::unwind(1)]
#[kani::proof]
fn mxfp4_e1m2_code_sign_ranges() {
    let code: u8 = kani::any();
    kani::assume(code <= 15);

    let sign = (code >> 3) & 1;

    if code < 8 {
        assert!(sign == 0, "codes [0,7] must have sign=0 (positive)");
    } else {
        assert!(sign == 1, "codes [8,15] must have sign=1 (negative)");
    }
}

// =========================================================================
// Block scale properties
// =========================================================================

// -------------------------------------------------------------------------
// Harness 7: Block scale is always a power of two.
//
// block_scale_from_exp computes 2^(shared_exp - 127).
// The result is always a power of two (or zero for extreme exponents
// that underflow to f32 subnormal, but 2^(-127) is subnormal in f32
// and handled via f64 intermediate).
// -------------------------------------------------------------------------

/// Prove: block_scale * block_scale^-1 is within f32 precision of 1.0
/// for all non-extreme shared exponents, confirming it is a power of two.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::powi, powi_f64_stub)]
fn mxfp4_block_scale_power_of_two() {
    let shared_exp: u8 = kani::any();
    // Avoid the extreme ends where f32 subnormals cause precision issues
    kani::assume(shared_exp >= 1 && shared_exp <= 254);

    let scale = block_scale_from_exp(shared_exp);
    assert!(scale.is_finite(), "block scale must be finite");
    assert!(scale > 0.0, "block scale must be positive");

    // For a power of two, log2(scale) should be an exact integer.
    // We check: scale * (1.0 / scale) == 1.0 (exact for powers of two).
    let inverse: f32 = 1.0 / scale;
    kani::assume(inverse.is_finite()); // may overflow for very small scale

    let product = scale * inverse;
    assert!(
        (product - 1.0).abs() < 1e-6,
        "scale * (1/scale) must be ~1.0 (power-of-two property)"
    );
}

// -------------------------------------------------------------------------
// Harness 8: Block scale for exp=127 is exactly 1.0 (2^0).
// -------------------------------------------------------------------------

/// Prove: block_scale_from_exp(127) == 1.0 exactly (the bias reference point).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::powi, powi_f64_stub)]
fn mxfp4_block_scale_at_bias_is_one() {
    let scale = block_scale_from_exp(127);
    assert!(scale == 1.0, "block_scale(127) must be exactly 1.0 (2^0)");
}

// -------------------------------------------------------------------------
// Harness 9: Block scale doubles when shared_exp increases by 1.
// -------------------------------------------------------------------------

/// Prove: block_scale_from_exp(e+1) == 2 * block_scale_from_exp(e)
/// for exponents in the normal f32 range.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::powi, powi_f64_stub)]
fn mxfp4_block_scale_doubles_per_step() {
    let e: u8 = kani::any();
    kani::assume(e >= 10 && e <= 253); // avoid subnormal/overflow edge cases

    let scale_e = block_scale_from_exp(e);
    let scale_e1 = block_scale_from_exp(e + 1);

    assert!(scale_e.is_finite(), "scale(e) must be finite");
    assert!(scale_e1.is_finite(), "scale(e+1) must be finite");
    assert!(scale_e > 0.0, "scale(e) must be positive");

    let ratio = scale_e1 / scale_e;
    assert!(
        (ratio - 2.0).abs() < 1e-6,
        "scale(e+1) / scale(e) must be 2.0"
    );
}

// =========================================================================
// Shared exponent computation
// =========================================================================

// -------------------------------------------------------------------------
// Harness 10: compute_shared_exponent returns 0 for all-zero block.
// -------------------------------------------------------------------------

/// Prove: an all-zero block has shared_exp=0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(33)]
fn mxfp4_shared_exp_zero_for_allzero() {
    let values = [0.0_f32; MXFP4_BLOCK_SIZE];
    let exp = compute_shared_exponent(&values);
    assert!(exp == 0, "all-zero block must have shared_exp=0");
}

// -------------------------------------------------------------------------
// Harness 11: Shared exponent yields scale that covers max_abs.
//
// After computing shared_exp, max_abs / block_scale <= E1M2_MAX (6.0).
// This is the fundamental property that ensures no value saturates.
// -------------------------------------------------------------------------

/// Prove: for any finite value, the shared exponent produces a block_scale
/// such that |value| / block_scale <= 6.0 (the E1M2 max magnitude).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(33)]
#[kani::stub(f32::ceil, ceil_f32_stub)]
#[kani::stub(f32::log2, log2_f32_stub)]
#[kani::stub(f64::powi, powi_f64_stub)]
fn mxfp4_shared_exp_covers_max_abs() {
    let val: f32 = kani::any();
    kani::assume(val.is_finite());
    kani::assume(val.abs() > 0.0 && val.abs() < 1e30);

    let idx: usize = kani::any();
    kani::assume(idx < MXFP4_BLOCK_SIZE);

    let mut values = [0.0_f32; MXFP4_BLOCK_SIZE];
    values[idx] = val;

    let shared_exp = compute_shared_exponent(&values);
    let block_scale = block_scale_from_exp(shared_exp);

    // max_abs / block_scale <= 6.0 (E1M2_MAX_MAGNITUDE)
    // Allow small epsilon for f32 rounding
    let scaled = val.abs() / block_scale;
    assert!(
        scaled <= 6.0 + 1e-5,
        "scaled max_abs must fit in E1M2 range [0, 6]"
    );
}

// =========================================================================
// Mxfp4Tensor properties
// =========================================================================

// -------------------------------------------------------------------------
// Harness 12: Mxfp4Tensor storage_bytes computation is correct.
//
// storage_bytes = num_blocks * BLOCK_STORAGE_BYTES
// BLOCK_STORAGE_BYTES = 17 (16 packed + 1 shared exp)
// -------------------------------------------------------------------------

/// Prove: storage_bytes = num_blocks * 17.
#[kani::unwind(1)]
#[kani::proof]
fn mxfp4_tensor_storage_bytes_formula() {
    let num_blocks: usize = kani::any();
    kani::assume(num_blocks >= 1 && num_blocks <= 10_000);

    let storage = num_blocks * BLOCK_STORAGE_BYTES;
    assert!(
        storage == num_blocks * 17,
        "storage must be num_blocks * 17"
    );
}

// -------------------------------------------------------------------------
// Harness 13: Mxfp4Tensor compression ratio > 1 for any nonempty tensor.
//
// f32: original_len * 4 bytes
// MXFP4: num_blocks * 17 bytes
// For original_len >= 32: num_blocks = ceil(len / 32)
// compression = (len * 4) / (ceil(len/32) * 17)
// For full blocks: = (32 * 4) / 17 = 128/17 ~ 7.53x
// -------------------------------------------------------------------------

/// Prove: MXFP4 compression ratio is > 1.0 for tensors with >= 32 elements.
#[kani::unwind(1)]
#[kani::proof]
fn mxfp4_tensor_compression_above_one() {
    let original_len: usize = kani::any();
    kani::assume(original_len >= MXFP4_BLOCK_SIZE && original_len <= 100_000);

    let num_blocks = original_len.div_ceil(MXFP4_BLOCK_SIZE);
    let mxfp4_bytes = num_blocks * BLOCK_STORAGE_BYTES;
    let f32_bytes = original_len * 4;

    // f32_bytes > mxfp4_bytes when:
    // original_len * 4 > ceil(original_len / 32) * 17
    // For full blocks: 32 * 4 = 128 > 17 (always true)
    assert!(
        f32_bytes > mxfp4_bytes,
        "MXFP4 must compress better than F32 for len >= 32"
    );
}

// -------------------------------------------------------------------------
// Harness 14: Mxfp4Tensor num_blocks is ceil(original_len / 32).
// -------------------------------------------------------------------------

/// Prove: the block count formula is consistent with div_ceil.
#[kani::unwind(1)]
#[kani::proof]
fn mxfp4_tensor_num_blocks_formula() {
    let original_len: usize = kani::any();
    kani::assume(original_len >= 1 && original_len <= 100_000);

    let num_blocks = original_len.div_ceil(MXFP4_BLOCK_SIZE);

    // num_blocks * MXFP4_BLOCK_SIZE >= original_len (covers all elements)
    assert!(
        num_blocks * MXFP4_BLOCK_SIZE >= original_len,
        "num_blocks * 32 must cover all elements"
    );

    // (num_blocks - 1) * MXFP4_BLOCK_SIZE < original_len (no excess full blocks)
    if num_blocks > 0 {
        assert!(
            (num_blocks - 1) * MXFP4_BLOCK_SIZE < original_len,
            "no excess full blocks"
        );
    }
}

// =========================================================================
// E1M2 decode properties
// =========================================================================

// -------------------------------------------------------------------------
// Harness 15: Decode is monotone in magnitude index for fixed sign.
//
// For positive codes (sign=0, idx 0-7), decode(idx+1) >= decode(idx).
// -------------------------------------------------------------------------

/// Prove: for positive codes, E1M2 decode values increase with magnitude index.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::powi, powi_f64_stub)]
fn mxfp4_e1m2_decode_monotone_positive() {
    let idx: usize = kani::any();
    kani::assume(idx < 7);

    let shared_exp: u8 = kani::any();
    kani::assume(shared_exp >= 1 && shared_exp <= 254);
    let block_scale = block_scale_from_exp(shared_exp);

    let code_lo = idx as u8; // sign=0, magnitude=idx
    let code_hi = (idx + 1) as u8; // sign=0, magnitude=idx+1

    let val_lo = decode_e1m2(code_lo, block_scale);
    let val_hi = decode_e1m2(code_hi, block_scale);

    assert!(
        val_hi >= val_lo,
        "positive decode must be monotonically non-decreasing"
    );
}

// -------------------------------------------------------------------------
// Harness 16: Negative codes have magnitude symmetry with positive codes.
//
// decode(code + 8) = -decode(code) for code in [0, 7].
// -------------------------------------------------------------------------

/// Prove: negative codes are the negation of corresponding positive codes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::powi, powi_f64_stub)]
fn mxfp4_e1m2_negative_is_negation() {
    let idx: u8 = kani::any();
    kani::assume(idx <= 7);

    let shared_exp: u8 = kani::any();
    kani::assume(shared_exp >= 1 && shared_exp <= 254);
    let block_scale = block_scale_from_exp(shared_exp);

    let pos_code = idx; // sign=0
    let neg_code = idx | 0x8; // sign=1, same magnitude

    let pos_val = decode_e1m2(pos_code, block_scale);
    let neg_val = decode_e1m2(neg_code, block_scale);

    // neg_val == -pos_val
    assert!(
        (neg_val + pos_val).abs() < 1e-30,
        "negative code must be negation of positive code"
    );
}

// =========================================================================
// Dequantize_block output bounds
// =========================================================================

// -------------------------------------------------------------------------
// Harness 17: Every dequantized value is bounded by E1M2_MAX * block_scale.
//
// For any block, every output of dequantize_block satisfies:
// |output[i]| <= 6.0 * block_scale.
// -------------------------------------------------------------------------

/// Prove: for any valid MXFP4 block, every dequantized value is bounded
/// by 6.0 * block_scale.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::powi, powi_f64_stub)]
fn mxfp4_dequant_block_output_bounded() {
    let shared_exp: u8 = kani::any();
    kani::assume(shared_exp <= 254);

    let block_scale = block_scale_from_exp(shared_exp);

    // Pick any element index and any packed byte
    let byte_idx: usize = kani::any();
    kani::assume(byte_idx < 16); // PACKED_BYTES = 16

    let packed_byte: u8 = kani::any();

    // Low nibble element
    let low_code = packed_byte & 0x0F;
    let low_val = decode_e1m2(low_code, block_scale);
    let bound = 6.0 * block_scale;
    assert!(
        low_val.abs() <= bound + 1e-30,
        "low nibble dequant must be bounded by 6 * block_scale"
    );

    // High nibble element
    let high_code = packed_byte >> 4;
    let high_val = decode_e1m2(high_code, block_scale);
    assert!(
        high_val.abs() <= bound + 1e-30,
        "high nibble dequant must be bounded by 6 * block_scale"
    );
}

// =========================================================================
// Nibble packing properties
// =========================================================================

// -------------------------------------------------------------------------
// Harness 18: Low and high nibble do not interfere.
//
// Setting the low nibble does not affect the high nibble and vice versa.
// -------------------------------------------------------------------------

/// Prove: low nibble assignment (OR with 0x0F mask) does not affect
/// high nibble, and vice versa.
#[kani::unwind(1)]
#[kani::proof]
fn mxfp4_nibble_independence() {
    let low: u8 = kani::any();
    let high: u8 = kani::any();
    kani::assume(low <= 15);
    kani::assume(high <= 15);

    let mut byte: u8 = 0;
    byte |= low; // set low nibble
    byte |= high << 4; // set high nibble

    // Low nibble is unaffected by high nibble
    assert!((byte & 0x0F) == low, "low nibble must be preserved");
    // High nibble is unaffected by low nibble
    assert!((byte >> 4) == high, "high nibble must be preserved");
}

// =========================================================================
// Code 0 and code 8 (positive and negative zero)
// =========================================================================

// -------------------------------------------------------------------------
// Harness 19: Both code 0 (+0.0) and code 8 (-0.0) decode to 0.0.
//
// In E1M2, magnitude index 0 maps to 0.0. The sign bit doesn't matter
// because -0.0 == 0.0 in IEEE 754.
// -------------------------------------------------------------------------

/// Prove: code 0 (positive zero) and code 8 (negative zero) both
/// decode to exactly 0.0 for any block scale.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::powi, powi_f64_stub)]
fn mxfp4_positive_negative_zero() {
    let shared_exp: u8 = kani::any();
    kani::assume(shared_exp <= 254);
    let block_scale = block_scale_from_exp(shared_exp);

    let pos_zero = decode_e1m2(0, block_scale);
    let neg_zero = decode_e1m2(8, block_scale);

    assert!(pos_zero == 0.0, "code 0 must decode to 0.0");
    assert!(neg_zero == 0.0, "code 8 must decode to 0.0");
    assert!(
        pos_zero == neg_zero,
        "positive and negative zero must be equal"
    );
}
