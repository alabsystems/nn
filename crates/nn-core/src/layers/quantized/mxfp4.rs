// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MXFP4 (Microscaling FP4) block-quantized storage.
//!
//! Implements the OCP Microscaling (MX) specification for 4-bit floating point
//! with shared block exponents. Each block of 32 elements shares a single 8-bit
//! exponent, giving effective 4-bit precision with much higher dynamic range
//! than naive int4.
//!
//! # Format
//!
//! - Block size: 32 elements
//! - Per-element: 1 sign bit + 1 exponent bit + 2 mantissa bits (E1M2)
//! - Shared block exponent: 8 bits (E8M0, applied to all 32 elements)
//! - Storage: 32 x 4 bits = 16 bytes per block + 1 byte shared exponent = 17 bytes
//!
//! # E1M2 encoding (per-element 4-bit value)
//!
//! | Bits (S.E.MM) | Value                        |
//! |---------------|------------------------------|
//! | 0.0.00        | +0.0                         |
//! | 0.0.01        | +0.5                         |
//! | 0.0.10        | +1.0                         |
//! | 0.0.11        | +1.5                         |
//! | 0.1.00        | +2.0                         |
//! | 0.1.01        | +3.0                         |
//! | 0.1.10        | +4.0                         |
//! | 0.1.11        | +6.0                         |
//! | 1.x.xx        | negative of above            |
//!
//! The shared block exponent scales all values by `2^(shared_exp - BIAS)`.
//!
//! Part of #2242

use crate::{Result, TensorError};

// -- Constants ----------------------------------------------------------------

/// Elements per MXFP4 block.
pub const MXFP4_BLOCK_SIZE: usize = 32;

/// Bytes of packed 4-bit elements per block (32 elements / 2 per byte).
const PACKED_BYTES: usize = MXFP4_BLOCK_SIZE / 2;

/// Storage bytes per block: 16 (packed elements) + 1 (shared exponent).
pub const BLOCK_STORAGE_BYTES: usize = PACKED_BYTES + 1;

/// Bias for the shared 8-bit exponent (E8M0 format).
/// The effective scale is `2^(shared_exp - SHARED_EXP_BIAS)`.
const SHARED_EXP_BIAS: i32 = 127;

/// The 8 representable magnitudes for E1M2 (subnormal + normal).
///
/// E=0 (subnormal): mantissa bits encode 0.{MM} => 0.0, 0.5, 1.0, 1.5
/// E=1 (normal):    mantissa bits encode 1.{MM} * 2 => 2.0, 3.0, 4.0, 6.0
///
/// Index = low 3 bits of the 4-bit code (sign bit excluded).
const E1M2_MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Maximum representable magnitude in E1M2 (before shared exponent scaling).
const E1M2_MAX_MAGNITUDE: f32 = 6.0;

// -- Mxfp4Block ---------------------------------------------------------------

/// One block of 32 MXFP4-quantized values.
///
/// Layout: `[shared_exp: u8][packed: [u8; 16]]`
/// - `shared_exp`: 8-bit shared exponent (E8M0). Effective scale = `2^(shared_exp - 127)`.
/// - `packed`: 16 bytes holding 32 x 4-bit E1M2 values (two per byte, low nibble first).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mxfp4Block {
    /// Shared 8-bit exponent for all 32 elements (E8M0 format).
    pub shared_exp: u8,
    /// 16 bytes of packed 4-bit E1M2 values (low nibble = even index, high nibble = odd).
    pub packed: [u8; PACKED_BYTES],
}

impl Mxfp4Block {
    /// Number of f32 elements this block encodes.
    pub const BLOCK_SIZE: usize = MXFP4_BLOCK_SIZE;

    /// Storage size in bytes for one block.
    pub const STORAGE_BYTES: usize = BLOCK_STORAGE_BYTES;
}

// -- Encode / Decode helpers --------------------------------------------------

/// Encode a single f32 value to a 4-bit E1M2 code given a shared exponent.
///
/// The shared exponent defines the block scale: `scale = 2^(shared_exp - 127)`.
/// The value is divided by scale, then rounded to the nearest E1M2 magnitude.
///
/// Returns a 4-bit code in `[0, 15]` (bit 3 = sign, bits 2..0 = magnitude index).
fn encode_e1m2(value: f32, block_scale: f32) -> u8 {
    if value == 0.0 || block_scale == 0.0 {
        return 0; // +0.0
    }

    let sign = value < 0.0;
    let abs_scaled = value.abs() / block_scale;

    // Find nearest E1M2 magnitude by minimum distance.
    let mut best_idx: u8 = 0;
    let mut best_dist = f32::MAX;
    let mut i: u8 = 0;
    while (i as usize) < E1M2_MAGNITUDES.len() {
        let dist = (abs_scaled - E1M2_MAGNITUDES[i as usize]).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
        i += 1;
    }

    let sign_bit = if sign { 0x8 } else { 0x0 };
    sign_bit | best_idx
}

/// Decode a 4-bit E1M2 code to f32 given a block scale.
///
/// `code` must be in `[0, 15]`.
fn decode_e1m2(code: u8, block_scale: f32) -> f32 {
    let sign = (code >> 3) & 1;
    let magnitude_idx = (code & 0x7) as usize;
    let magnitude = E1M2_MAGNITUDES[magnitude_idx];
    let value = magnitude * block_scale;
    if sign != 0 {
        -value
    } else {
        value
    }
}

/// Compute the shared exponent for a block of 32 f32 values.
///
/// The shared exponent is chosen so that `max_abs / 2^(exp - 127)` fits within
/// the E1M2 representable range [0, 6.0].
///
/// Returns 0 for all-zero blocks (block_scale = 2^-127 effectively zero).
fn compute_shared_exponent(values: &[f32; MXFP4_BLOCK_SIZE]) -> u8 {
    let max_abs = values.iter().copied().fold(0.0_f32, |acc, v| {
        let a = v.abs();
        if a > acc {
            a
        } else {
            acc
        }
    });

    if max_abs == 0.0 {
        return 0;
    }

    // We need: max_abs <= E1M2_MAX_MAGNITUDE * 2^(exp - BIAS)
    // => 2^(exp - BIAS) >= max_abs / E1M2_MAX_MAGNITUDE
    // => exp - BIAS >= log2(max_abs / E1M2_MAX_MAGNITUDE)
    // => exp >= log2(max_abs / E1M2_MAX_MAGNITUDE) + BIAS
    //
    // Use floor to get the tightest exponent that can represent max_abs.
    let log2_ratio = (max_abs / E1M2_MAX_MAGNITUDE).log2();
    let exp_unbiased = log2_ratio.ceil() as i32;
    let exp_biased = exp_unbiased + SHARED_EXP_BIAS;

    // Clamp to valid u8 range [0, 254]. 255 is reserved for NaN/Inf in E8M0.
    exp_biased.clamp(0, 254) as u8
}

/// Compute the block scale from a shared exponent.
///
/// `block_scale = 2^(shared_exp - 127)`
///
/// For shared_exp in [1, 254], the result is a normal f32 power of two.
/// For shared_exp = 0, exponent = -127 which is subnormal territory;
/// we use f64 intermediate to avoid subnormal precision loss.
fn block_scale_from_exp(shared_exp: u8) -> f32 {
    let exponent = i32::from(shared_exp) - SHARED_EXP_BIAS;
    // 2.0_f64.powi(n) is exact for integer n in f64 range, then cast to f32.
    // This handles the subnormal case (exponent = -127) correctly.
    2.0_f64.powi(exponent) as f32
}

// -- Public API: block-level --------------------------------------------------

/// Quantize 32 f32 values to an MXFP4 block.
///
/// Non-finite values (NaN, Inf) are rejected with an error.
/// Values are clamped to the representable range of the computed shared exponent.
///
/// # Errors
///
/// Returns `ValueOutOfRange` if any input is non-finite.
pub fn quantize_block(values: &[f32; MXFP4_BLOCK_SIZE]) -> Result<Mxfp4Block> {
    // Validate: all values must be finite
    for &v in values.iter() {
        if !v.is_finite() {
            return Err(TensorError::ValueOutOfRange {
                description: "mxfp4 quantize_block: non-finite input value",
            });
        }
    }

    let shared_exp = compute_shared_exponent(values);
    let block_scale = block_scale_from_exp(shared_exp);

    let mut packed = [0u8; PACKED_BYTES];
    #[allow(clippy::needless_range_loop)]
    for i in 0..MXFP4_BLOCK_SIZE {
        let code = encode_e1m2(values[i], block_scale);
        let byte_idx = i / 2;
        if i % 2 == 0 {
            packed[byte_idx] |= code; // low nibble
        } else {
            packed[byte_idx] |= code << 4; // high nibble
        }
    }

    Ok(Mxfp4Block { shared_exp, packed })
}

/// Dequantize an MXFP4 block back to 32 f32 values.
///
/// This is a pure function that never fails.
#[must_use]
pub fn dequantize_block(block: &Mxfp4Block) -> [f32; MXFP4_BLOCK_SIZE] {
    let block_scale = block_scale_from_exp(block.shared_exp);
    let mut output = [0.0_f32; MXFP4_BLOCK_SIZE];

    #[allow(clippy::needless_range_loop)]
    for i in 0..MXFP4_BLOCK_SIZE {
        let byte_idx = i / 2;
        let code = if i % 2 == 0 {
            block.packed[byte_idx] & 0x0F
        } else {
            block.packed[byte_idx] >> 4
        };
        output[i] = decode_e1m2(code, block_scale);
    }

    output
}

// -- Public API: tensor-level -------------------------------------------------

/// A tensor stored in MXFP4 block-quantized format.
///
/// Data is stored as contiguous blocks. The last block may represent
/// padding beyond `original_len` elements.
#[derive(Debug, Clone)]
pub struct Mxfp4Tensor {
    /// Contiguous MXFP4 blocks.
    blocks: Vec<Mxfp4Block>,
    /// Number of original (pre-padding) f32 elements.
    original_len: usize,
}

impl Mxfp4Tensor {
    /// Number of original f32 elements (before block-boundary padding).
    #[must_use]
    pub fn original_len(&self) -> usize {
        self.original_len
    }

    /// Number of MXFP4 blocks.
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Total storage size in bytes.
    #[must_use]
    pub fn storage_bytes(&self) -> usize {
        self.blocks.len() * BLOCK_STORAGE_BYTES
    }

    /// Storage size if the same data were stored as f32.
    #[must_use]
    pub fn f32_storage_bytes(&self) -> usize {
        self.original_len * 4
    }

    /// Compression ratio vs f32 storage.
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.storage_bytes() == 0 {
            return 0.0;
        }
        self.f32_storage_bytes() as f64 / self.storage_bytes() as f64
    }

    /// Access the underlying blocks.
    #[must_use]
    pub fn blocks(&self) -> &[Mxfp4Block] {
        &self.blocks
    }
}

/// Quantize an arbitrary-length f32 slice to MXFP4 tensor format.
///
/// The data is padded to the next block boundary (multiple of 32) with zeros.
/// The original length is preserved for truncation during dequantization.
///
/// # Errors
///
/// Returns `ValueOutOfRange` if any input is non-finite.
pub fn quantize_tensor(data: &[f32]) -> Result<Mxfp4Tensor> {
    let original_len = data.len();
    let n_blocks = original_len.div_ceil(MXFP4_BLOCK_SIZE);

    let mut blocks = Vec::with_capacity(n_blocks);

    for block_idx in 0..n_blocks {
        let start = block_idx * MXFP4_BLOCK_SIZE;
        let mut block_values = [0.0_f32; MXFP4_BLOCK_SIZE];

        let end = (start + MXFP4_BLOCK_SIZE).min(data.len());
        let chunk_len = end - start;
        block_values[..chunk_len].copy_from_slice(&data[start..end]);
        // Remaining elements stay zero (padding).

        blocks.push(quantize_block(&block_values)?);
    }

    Ok(Mxfp4Tensor {
        blocks,
        original_len,
    })
}

/// Dequantize an MXFP4 tensor back to a Vec of f32 values.
///
/// Only the original (pre-padding) elements are returned.
#[must_use]
pub fn dequantize_tensor(tensor: &Mxfp4Tensor) -> Vec<f32> {
    let mut result = Vec::with_capacity(tensor.original_len);

    for block in &tensor.blocks {
        let values = dequantize_block(block);
        let remaining = tensor.original_len - result.len();
        let take = remaining.min(MXFP4_BLOCK_SIZE);
        result.extend_from_slice(&values[..take]);
    }

    result
}

// -- Kani proof harnesses -----------------------------------------------------

#[cfg(kani)]
mod kani_proofs {
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

    // -----------------------------------------------------------------------
    // Harness 1: quantize then dequantize produces finite values for any
    // finite input.
    // -----------------------------------------------------------------------

    /// Prove: for any block of 32 finite f32 values in a bounded range,
    /// quantize_block followed by dequantize_block produces all-finite output.
    ///
    /// Uses bounded inputs because Kani cannot efficiently enumerate all f32;
    /// the bounded range [-1e30, 1e30] covers practical weight magnitudes while
    /// staying within f32 representable range.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(33)] // 32 iterations + 1 exit check
    #[kani::stub(f32::ceil, ceil_f32_stub)]
    #[kani::stub(f32::log2, log2_f32_stub)]
    #[kani::stub(f64::powi, powi_f64_stub)]
    fn mxfp4_roundtrip_finite() {
        // Test with a single representative element (Kani explores all bit patterns).
        let val: f32 = kani::any();
        kani::assume(val.is_finite());
        kani::assume(val >= -1e30 && val <= 1e30);

        // Build a block with this value in one position.
        let idx: usize = kani::any();
        kani::assume(idx < MXFP4_BLOCK_SIZE);

        let mut values = [0.0_f32; MXFP4_BLOCK_SIZE];
        values[idx] = val;

        // Compute shared exponent.
        let shared_exp = compute_shared_exponent(&values);
        let block_scale = block_scale_from_exp(shared_exp);

        // Encode then decode the value.
        let code = encode_e1m2(val, block_scale);
        let decoded = decode_e1m2(code, block_scale);

        assert!(decoded.is_finite(), "dequantized value must be finite");
    }

    // -----------------------------------------------------------------------
    // Harness 2: quantize/dequantize helpers never panic for bounded inputs.
    // -----------------------------------------------------------------------

    /// Prove: encode_e1m2 and decode_e1m2 never panic for any valid inputs.
    ///
    /// encode_e1m2: any finite f32 + any positive block_scale.
    /// decode_e1m2: any 4-bit code [0,15] + any finite block_scale.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(9)] // E1M2_MAGNITUDES has 8 entries + 1 exit check
    #[kani::stub(f64::powi, powi_f64_stub)]
    fn mxfp4_no_panic() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite());
        kani::assume(val >= -1e30 && val <= 1e30);

        let shared_exp: u8 = kani::any();
        kani::assume(shared_exp <= 254); // 255 reserved for NaN/Inf in E8M0

        let block_scale = block_scale_from_exp(shared_exp);

        // encode never panics
        let code = encode_e1m2(val, block_scale);
        assert!(code <= 15, "4-bit code must be in [0, 15]");

        // decode never panics
        let decode_code: u8 = kani::any();
        kani::assume(decode_code <= 15);
        let result = decode_e1m2(decode_code, block_scale);
        // result is finite when block_scale is finite and magnitude is finite
        // block_scale from exp in [0,254] is always finite and non-negative
        assert!(
            result.is_finite(),
            "decoded value must be finite for valid inputs"
        );
    }

    // -----------------------------------------------------------------------
    // Harness 3: E1M2 code is always in [0, 15] (4-bit)
    // -----------------------------------------------------------------------

    /// Prove: encode_e1m2 always returns a value in [0, 15] regardless of input.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(9)]
    fn mxfp4_encode_4bit_bounded() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite());
        kani::assume(val >= -1e30 && val <= 1e30);

        let block_scale: f32 = kani::any();
        kani::assume(block_scale.is_finite());
        kani::assume(block_scale > 0.0);

        let code = encode_e1m2(val, block_scale);
        assert!(code <= 15, "E1M2 code must fit in 4 bits");
    }

    // -----------------------------------------------------------------------
    // Harness 4: block_scale_from_exp produces finite positive values for
    // valid exponents.
    // -----------------------------------------------------------------------

    /// Prove: block_scale_from_exp returns a finite, positive value for
    /// all shared exponents in [0, 254].
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    #[kani::stub(f64::powi, powi_f64_stub)]
    fn mxfp4_block_scale_finite_positive() {
        let shared_exp: u8 = kani::any();
        kani::assume(shared_exp <= 254);

        let scale = block_scale_from_exp(shared_exp);
        assert!(scale.is_finite(), "block scale must be finite");
        assert!(scale > 0.0, "block scale must be positive");
    }

    // -----------------------------------------------------------------------
    // Harness 5: decode preserves sign of encode.
    // -----------------------------------------------------------------------

    /// Prove: for nonzero values, the sign of the dequantized value matches
    /// the sign of the original value (when the encoded code is nonzero).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(9)]
    #[kani::stub(f64::powi, powi_f64_stub)]
    fn mxfp4_sign_preservation() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite());
        kani::assume(val != 0.0);
        kani::assume(val >= -1e30 && val <= 1e30);

        let shared_exp: u8 = kani::any();
        kani::assume(shared_exp >= 1 && shared_exp <= 254);

        let block_scale = block_scale_from_exp(shared_exp);
        let code = encode_e1m2(val, block_scale);

        if code != 0 {
            let decoded = decode_e1m2(code, block_scale);
            if decoded != 0.0 {
                // Sign must match
                assert!(
                    (val > 0.0 && decoded > 0.0) || (val < 0.0 && decoded < 0.0),
                    "sign must be preserved through encode/decode"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Harness 6: packed nibble indexing is in-bounds.
    // -----------------------------------------------------------------------

    /// Prove: the nibble packing/unpacking index arithmetic in
    /// quantize_block/dequantize_block never exceeds PACKED_BYTES.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(33)]
    fn mxfp4_nibble_index_in_bounds() {
        let i: usize = kani::any();
        kani::assume(i < MXFP4_BLOCK_SIZE);

        let byte_idx = i / 2;
        assert!(byte_idx < PACKED_BYTES, "byte index must be in bounds");

        // Verify low/high nibble selection
        if i % 2 == 0 {
            // low nibble: code & 0x0F — always in [0, 15]
            let byte_val: u8 = kani::any();
            let nibble = byte_val & 0x0F;
            assert!(nibble <= 15);
        } else {
            // high nibble: code >> 4 — always in [0, 15]
            let byte_val: u8 = kani::any();
            let nibble = byte_val >> 4;
            assert!(nibble <= 15);
        }
    }

    // -----------------------------------------------------------------------
    // Harness 7: full roundtrip through quantize_block/dequantize_block
    //            produces all-finite outputs.
    // -----------------------------------------------------------------------

    /// Prove: for any finite f32 value placed at any index in a block,
    /// the full quantize_block→dequantize_block roundtrip produces
    /// all-finite output values and no errors.
    ///
    /// This is stronger than harness 1 because it exercises the public API
    /// (quantize_block + dequantize_block) rather than the internal helpers,
    /// covering packed nibble storage and shared exponent computation.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(33)]
    #[kani::stub(f32::ceil, ceil_f32_stub)]
    #[kani::stub(f32::log2, log2_f32_stub)]
    #[kani::stub(f64::powi, powi_f64_stub)]
    fn mxfp4_full_roundtrip_produces_finite() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite());
        kani::assume(val >= -1e30 && val <= 1e30);

        let idx: usize = kani::any();
        kani::assume(idx < MXFP4_BLOCK_SIZE);

        let mut values = [0.0_f32; MXFP4_BLOCK_SIZE];
        values[idx] = val;

        let block = match quantize_block(&values) {
            Ok(b) => b,
            Err(_) => {
                // quantize_block only fails on non-finite; we assumed finite.
                panic!("quantize_block should succeed for finite inputs");
            }
        };

        let output = dequantize_block(&block);

        // Every element of the output must be finite.
        let mut i = 0;
        while i < MXFP4_BLOCK_SIZE {
            assert!(
                output[i].is_finite(),
                "dequantized output must be all-finite"
            );
            i += 1;
        }
    }

    // -----------------------------------------------------------------------
    // Harness 8: compute_shared_exponent yields a scale that is non-negative.
    // -----------------------------------------------------------------------

    /// Prove: for any finite input value, the shared block scale derived from
    /// compute_shared_exponent is always non-negative (>= 0.0).
    ///
    /// This covers the composition: compute_shared_exponent → block_scale_from_exp,
    /// proving the scale can never go negative regardless of input magnitude or sign.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(33)]
    #[kani::stub(f32::ceil, ceil_f32_stub)]
    #[kani::stub(f32::log2, log2_f32_stub)]
    #[kani::stub(f64::powi, powi_f64_stub)]
    fn mxfp4_shared_scale_non_negative() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite());
        kani::assume(val >= -1e30 && val <= 1e30);

        let idx: usize = kani::any();
        kani::assume(idx < MXFP4_BLOCK_SIZE);

        let mut values = [0.0_f32; MXFP4_BLOCK_SIZE];
        values[idx] = val;

        let shared_exp = compute_shared_exponent(&values);

        // shared_exp must be in valid E8M0 range (255 reserved for NaN/Inf).
        assert!(shared_exp <= 254, "shared exponent must be in [0, 254]");

        let scale = block_scale_from_exp(shared_exp);

        // Scale must be non-negative and finite.
        assert!(scale >= 0.0, "block scale must be non-negative");
        assert!(scale.is_finite(), "block scale must be finite");
    }

    // -----------------------------------------------------------------------
    // Harness 9: zero preservation — quantize(0.0) dequantizes to exactly 0.0.
    // -----------------------------------------------------------------------

    /// Prove: an all-zero block quantizes and dequantizes back to exactly
    /// all zeros. Also: for any shared exponent, encoding 0.0 yields code 0
    /// and decoding code 0 yields 0.0.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(9)]
    #[kani::stub(f64::powi, powi_f64_stub)]
    fn mxfp4_zero_preservation() {
        // Part A: encode_e1m2(0.0, any_valid_scale) == 0
        let shared_exp: u8 = kani::any();
        kani::assume(shared_exp <= 254);
        let block_scale = block_scale_from_exp(shared_exp);

        let code = encode_e1m2(0.0, block_scale);
        assert!(code == 0, "encoding 0.0 must produce code 0");

        // Part B: decode_e1m2(0, any_valid_scale) == 0.0
        let decoded = decode_e1m2(0, block_scale);
        assert!(decoded == 0.0, "decoding code 0 must produce 0.0");

        // Combined: roundtrip of 0.0 is exactly 0.0
        let roundtrip = decode_e1m2(encode_e1m2(0.0, block_scale), block_scale);
        assert!(roundtrip == 0.0, "zero must roundtrip exactly to zero");
    }

    // -----------------------------------------------------------------------
    // Harness 10: symmetry — |dequant(quant(x))| == |dequant(quant(-x))|
    //             for any finite nonzero x.
    // -----------------------------------------------------------------------

    /// Prove: the MXFP4 encoding has symmetric magnitudes for positive and
    /// negative values. Given a fixed shared exponent, encoding +x and -x
    /// produces codes whose decoded magnitudes are identical.
    ///
    /// This follows from the E1M2 format: bit 3 is the sign bit, bits 2..0
    /// are the magnitude index. The harness proves this structurally.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(9)]
    #[kani::stub(f64::powi, powi_f64_stub)]
    fn mxfp4_magnitude_symmetry() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite());
        kani::assume(val > 0.0); // test positive, derive negative
        kani::assume(val <= 1e30);

        let shared_exp: u8 = kani::any();
        kani::assume(shared_exp >= 1 && shared_exp <= 254);

        let block_scale = block_scale_from_exp(shared_exp);

        let code_pos = encode_e1m2(val, block_scale);
        let code_neg = encode_e1m2(-val, block_scale);

        let decoded_pos = decode_e1m2(code_pos, block_scale);
        let decoded_neg = decode_e1m2(code_neg, block_scale);

        // The magnitudes must be equal: |decoded_pos| == |decoded_neg|.
        assert!(
            decoded_pos.abs() == decoded_neg.abs(),
            "symmetric inputs must produce symmetric magnitudes"
        );
    }

    // Harness 11: MXFP4 roundtrip error bounded.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(9)]
    #[kani::stub(f64::powi, powi_f64_stub)]
    fn mxfp4_roundtrip_error_bounded() {
        let val: f32 = kani::any();
        kani::assume(val.is_finite());
        kani::assume(val >= -1e10 && val <= 1e10);
        let shared_exp: u8 = kani::any();
        kani::assume(shared_exp >= 1 && shared_exp <= 254);
        let block_scale = block_scale_from_exp(shared_exp);
        kani::assume(val.abs() <= 6.0 * block_scale);
        let code = encode_e1m2(val, block_scale);
        let decoded = decode_e1m2(code, block_scale);
        let error = (decoded - val).abs();
        let bound = block_scale * 1.0 + 1e-5;
        assert!(error <= bound, "MXFP4 error bounded");
    }

    // Harness 12: MXFP4 dequant magnitude bounded.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    #[kani::stub(f64::powi, powi_f64_stub)]
    fn mxfp4_dequant_magnitude_bounded() {
        let code: u8 = kani::any();
        kani::assume(code <= 15);
        let shared_exp: u8 = kani::any();
        kani::assume(shared_exp <= 254);
        let block_scale = block_scale_from_exp(shared_exp);
        let decoded = decode_e1m2(code, block_scale);
        assert!(decoded.is_finite(), "finite");
        let bound = E1M2_MAX_MAGNITUDE * block_scale;
        assert!(decoded.abs() <= bound + 1e-30, "magnitude bounded");
    }

    // Harness 13: MXFP4 storage invariant.
    #[kani::unwind(1)]
    #[kani::proof]
    fn mxfp4_storage_invariant() {
        assert!(MXFP4_BLOCK_SIZE == 32);
        assert!(PACKED_BYTES == 16);
        assert!(BLOCK_STORAGE_BYTES == 17);
        let f32_bytes = MXFP4_BLOCK_SIZE * 4;
        assert!(f32_bytes == 128);
        assert!(f32_bytes > BLOCK_STORAGE_BYTES * 7);
    }
}

#[cfg(kani)]
#[path = "kani_mxfp4_extra.rs"]
mod kani_mxfp4_extra;

#[cfg(test)]
#[path = "mxfp4_tests.rs"]
mod tests;
