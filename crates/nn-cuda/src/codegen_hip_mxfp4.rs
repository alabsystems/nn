// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MXFP4 (OCP Microscaling FP4) format helpers for HIP codegen.
//!
//! MXFP4 uses the E2M1 element format (4-bit: 1 sign, 2 exponent, 1 mantissa)
//! with a shared E8M0 exponent per block of 32 elements. This module provides
//! HIP C++ code snippets for dequantization during GEMM kernel execution.
//!
//! # OCP MX Specification
//!
//! - Block size: 32 elements
//! - Shared exponent: E8M0 (8-bit unsigned, represents `2^(exp - 127)`)
//! - Element format: E2M1 (4-bit)
//! - Packing: 2 elements per byte (low nibble = even index, high nibble = odd)
//!
//! # E2M1 Value Table
//!
//! ```text
//! Bits  | Value
//! 0000  | +0.0
//! 0001  | +0.5
//! 0010  | +1.0
//! 0011  | +1.5
//! 0100  | +2.0
//! 0101  | +3.0
//! 0110  | +4.0
//! 0111  | +6.0
//! 1xxx  | negative of 0xxx
//! ```
//!
//! Part of #2543 (AMD x GPU MODE competition) and #2242 (MXFP4 dtype).

/// MXFP4 block size: number of elements sharing a single E8M0 scale.
pub const MXFP4_BLOCK_SIZE: usize = 32;

/// Bytes per MXFP4 block (32 elements × 4 bits / 8 bits per byte).
pub const MXFP4_BLOCK_BYTES: usize = MXFP4_BLOCK_SIZE / 2;

/// HIP C++ device-side lookup table for E2M1 → float dequantization.
///
/// Indexed by 4-bit E2M1 pattern. Returns the base float value before
/// applying the shared E8M0 scale factor.
pub const MXFP4_E2M1_LUT_HIP: &str = "\
__device__ __constant__ float MXFP4_E2M1_LUT[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};\n";

/// HIP C++ inline device function: unpack one E2M1 element from a packed byte.
///
/// Extracts the low nibble (even index) or high nibble (odd index) and
/// looks up the float value in the E2M1 LUT.
pub const MXFP4_UNPACK_FN_HIP: &str = "\
__device__ __forceinline__ float mxfp4_unpack(unsigned char packed_byte, unsigned int sub_idx) {
    // sub_idx 0 → low nibble, sub_idx 1 → high nibble
    unsigned int nibble = (sub_idx == 0u) ? (packed_byte & 0x0Fu) : (packed_byte >> 4u);
    return MXFP4_E2M1_LUT[nibble];
}\n";

/// HIP C++ inline device function: compute E8M0 scale factor.
///
/// E8M0 is an 8-bit exponent-only format: value = 2^(exp - 127).
/// Special case: exp == 0 maps to 0.0 (zero scale).
pub const MXFP4_SCALE_FN_HIP: &str = "\
__device__ __forceinline__ float mxfp4_scale(unsigned char e8m0_exp) {
    if (e8m0_exp == 0u) return 0.0f;
    // ldexpf(1.0f, exp - 127) = 2^(exp - 127), exact for all integer exponents.
    return ldexpf(1.0f, (int)e8m0_exp - 127);
}\n";

/// HIP C++ inline device function: dequantize one MXFP4 element.
///
/// Combines E2M1 unpack with E8M0 scale to produce the final float value.
pub const MXFP4_DEQUANT_FN_HIP: &str = "\
__device__ __forceinline__ float mxfp4_dequant(
    unsigned char packed_byte, unsigned int sub_idx, unsigned char e8m0_exp
) {
    return mxfp4_unpack(packed_byte, sub_idx) * mxfp4_scale(e8m0_exp);
}\n";

/// Emit all MXFP4 device helper functions as a single HIP C++ preamble.
///
/// Include this before any kernel that uses MXFP4 dequantization.
#[must_use]
pub fn mxfp4_preamble_hip() -> String {
    format!(
        "{MXFP4_E2M1_LUT_HIP}\n{MXFP4_UNPACK_FN_HIP}\n{MXFP4_SCALE_FN_HIP}\n{MXFP4_DEQUANT_FN_HIP}"
    )
}

/// Compute the number of MXFP4 packed bytes for a given element count.
///
/// Returns `Err` if `num_elements` is not a multiple of 2 (MXFP4 packs 2 per byte).
pub fn mxfp4_packed_bytes(num_elements: usize) -> Result<usize, crate::HipCodegenError> {
    if !num_elements.is_multiple_of(2) {
        return Err(crate::HipCodegenError::InvalidParameter(format!(
            "MXFP4 requires even element count, got {num_elements}"
        )));
    }
    Ok(num_elements / 2)
}

/// Compute the number of E8M0 scale values for a given element count.
///
/// Returns `Err` if `num_elements` is not a multiple of `MXFP4_BLOCK_SIZE` (32).
pub fn mxfp4_num_scales(num_elements: usize) -> Result<usize, crate::HipCodegenError> {
    if !num_elements.is_multiple_of(MXFP4_BLOCK_SIZE) {
        return Err(crate::HipCodegenError::InvalidParameter(format!(
            "MXFP4 requires element count divisible by {MXFP4_BLOCK_SIZE}, got {num_elements}"
        )));
    }
    Ok(num_elements / MXFP4_BLOCK_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mxfp4_preamble_contains_all_helpers() {
        let preamble = mxfp4_preamble_hip();
        assert!(preamble.contains("MXFP4_E2M1_LUT"));
        assert!(preamble.contains("mxfp4_unpack"));
        assert!(preamble.contains("mxfp4_scale"));
        assert!(preamble.contains("mxfp4_dequant"));
        assert!(preamble.contains("ldexpf"));
    }

    #[test]
    fn test_mxfp4_packed_bytes() {
        assert_eq!(mxfp4_packed_bytes(32).unwrap(), 16);
        assert_eq!(mxfp4_packed_bytes(64).unwrap(), 32);
        assert_eq!(mxfp4_packed_bytes(1024).unwrap(), 512);
        assert!(mxfp4_packed_bytes(33).is_err());
    }

    #[test]
    fn test_mxfp4_num_scales() {
        assert_eq!(mxfp4_num_scales(32).unwrap(), 1);
        assert_eq!(mxfp4_num_scales(1024).unwrap(), 32);
        assert!(mxfp4_num_scales(48).is_err());
    }

    #[test]
    fn test_mxfp4_constants() {
        assert_eq!(MXFP4_BLOCK_SIZE, 32);
        assert_eq!(MXFP4_BLOCK_BYTES, 16);
    }

    #[test]
    fn test_e2m1_lut_has_16_entries() {
        // LUT must have exactly 16 entries (4-bit index).
        let lut = MXFP4_E2M1_LUT_HIP;
        let count = lut.matches(',').count() + 1;
        // 16 entries separated by 15 commas, but the format has trailing items
        // in 2 rows of 8. Count the numeric literals.
        assert!(lut.contains("0.0f"));
        assert!(lut.contains("6.0f"));
        assert!(lut.contains("-6.0f"));
        assert_eq!(count, 16, "E2M1 LUT must have 16 entries");
    }
}
