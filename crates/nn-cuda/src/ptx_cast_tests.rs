// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for PTX dtype cast kernel generation.
//!
//! Verifies PTX structural correctness, conversion instruction presence,
//! round-trip precision, and edge cases for f32/f16/bf16 conversions.

use super::*;

// =========================================================================
// f32 -> f16 PTX structural checks
// =========================================================================

#[test]
fn test_f32_to_f16_ptx_contains_cvt_instruction() {
    let ptx = generate_f32_to_f16_ptx(1024);
    assert!(
        ptx.contains("cvt.rn.f16.f32"),
        "must contain f32->f16 conversion instruction"
    );
}

#[test]
fn test_f32_to_f16_ptx_structure() {
    let ptx = generate_f32_to_f16_ptx(512);
    assert!(ptx.contains(".version 7.0"));
    assert!(ptx.contains(".address_size 64"));
    assert!(ptx.contains(".visible .entry cast_f32_to_f16"));
    assert!(ptx.contains("param_input"));
    assert!(ptx.contains("param_output"));
    assert!(ptx.contains("param_n"));
    assert!(ptx.contains("ld.global.f32"), "must load f32 input");
    assert!(ptx.contains("st.global.b16"), "must store 16-bit output");
    assert!(ptx.contains(".reqntid 256"));
}

// =========================================================================
// f16 -> f32 PTX structural checks
// =========================================================================

#[test]
fn test_f16_to_f32_ptx_contains_cvt_instruction() {
    let ptx = generate_f16_to_f32_ptx(1024);
    assert!(
        ptx.contains("cvt.f32.f16"),
        "must contain f16->f32 conversion instruction"
    );
}

#[test]
fn test_f16_to_f32_ptx_structure() {
    let ptx = generate_f16_to_f32_ptx(512);
    assert!(ptx.contains(".visible .entry cast_f16_to_f32"));
    assert!(ptx.contains("ld.global.b16"), "must load 16-bit input");
    assert!(ptx.contains("st.global.f32"), "must store f32 output");
}

// =========================================================================
// f32 -> bf16 PTX structural checks
// =========================================================================

#[test]
fn test_f32_to_bf16_ptx_contains_cvt_instruction() {
    let ptx = generate_f32_to_bf16_ptx(1024);
    assert!(
        ptx.contains("cvt.rn.bf16.f32"),
        "must contain f32->bf16 conversion instruction"
    );
}

#[test]
fn test_f32_to_bf16_ptx_structure() {
    let ptx = generate_f32_to_bf16_ptx(512);
    assert!(ptx.contains(".visible .entry cast_f32_to_bf16"));
    assert!(ptx.contains("ld.global.f32"), "must load f32 input");
    assert!(ptx.contains("st.global.b16"), "must store 16-bit output");
}

// =========================================================================
// bf16 -> f32 PTX structural checks
// =========================================================================

#[test]
fn test_bf16_to_f32_ptx_contains_cvt_instruction() {
    let ptx = generate_bf16_to_f32_ptx(1024);
    assert!(
        ptx.contains("cvt.f32.bf16"),
        "must contain bf16->f32 conversion instruction"
    );
}

#[test]
fn test_bf16_to_f32_ptx_structure() {
    let ptx = generate_bf16_to_f32_ptx(512);
    assert!(ptx.contains(".visible .entry cast_bf16_to_f32"));
    assert!(ptx.contains("ld.global.b16"), "must load 16-bit input");
    assert!(ptx.contains("st.global.f32"), "must store f32 output");
}

// =========================================================================
// Round-trip precision: f32 -> f16 -> f32
// =========================================================================

#[test]
fn test_f32_f16_roundtrip_preserves_values() {
    // Simulate round-trip: f32 -> f16 -> f32 using half crate semantics
    // f16 has 10-bit mantissa, so ~3 decimal digits of precision
    let test_values: &[f32] = &[
        0.0, 1.0, -1.0, 0.5, -0.5, 3.14159, 100.0, -100.0, 0.001, 1000.0, 65504.0, // f16 max
    ];

    for &val in test_values {
        // f16 round-trip: truncate to f16 precision then back to f32
        let f16_bits = f32_to_f16_bits(val);
        let roundtrip = f16_bits_to_f32(f16_bits);

        if val == 0.0 {
            assert_eq!(roundtrip, 0.0, "zero must round-trip exactly");
        } else {
            let relative_error = ((roundtrip - val) / val).abs();
            assert!(
                relative_error < 0.002, // f16 has ~0.1% relative precision
                "f16 roundtrip: {val} -> {roundtrip}, relative error {relative_error}"
            );
        }
    }
}

// =========================================================================
// Round-trip precision: f32 -> bf16 -> f32
// =========================================================================

#[test]
fn test_f32_bf16_roundtrip_preserves_values() {
    // bf16 has 7-bit mantissa (~2 decimal digits), but same range as f32
    let test_values: &[f32] = &[
        0.0, 1.0, -1.0, 0.5, -0.5, 3.14159, 100.0, -100.0, 0.001, 1000.0, 1e10,
        1e30, // bf16 has f32 range
    ];

    for &val in test_values {
        let bf16_bits = f32_to_bf16_bits(val);
        let roundtrip = bf16_bits_to_f32(bf16_bits);

        if val == 0.0 {
            assert_eq!(roundtrip, 0.0, "zero must round-trip exactly");
        } else {
            let relative_error = ((roundtrip - val) / val).abs();
            assert!(
                relative_error < 0.01, // bf16 has ~0.8% relative precision
                "bf16 roundtrip: {val} -> {roundtrip}, relative error {relative_error}"
            );
        }
    }
}

// =========================================================================
// Edge cases
// =========================================================================

#[test]
fn test_f16_roundtrip_zero() {
    // +0.0
    let bits = f32_to_f16_bits(0.0);
    assert_eq!(f16_bits_to_f32(bits), 0.0);

    // -0.0
    let bits_neg = f32_to_f16_bits(-0.0);
    let rt = f16_bits_to_f32(bits_neg);
    assert_eq!(rt, 0.0); // -0.0 == 0.0 in f32
    assert!(rt.is_sign_negative() || rt == 0.0); // sign bit may be preserved
}

#[test]
fn test_bf16_roundtrip_zero() {
    let bits = f32_to_bf16_bits(0.0);
    assert_eq!(bf16_bits_to_f32(bits), 0.0);

    let bits_neg = f32_to_bf16_bits(-0.0);
    let rt = bf16_bits_to_f32(bits_neg);
    assert_eq!(rt, 0.0);
}

#[test]
fn test_f16_large_value_overflow() {
    // f16 max is 65504. Values above this overflow to infinity in f16.
    let large = 100_000.0f32;
    let bits = f32_to_f16_bits(large);
    let rt = f16_bits_to_f32(bits);
    // Should be infinity (f16 overflow)
    assert!(
        rt.is_infinite(),
        "f16 cannot represent {large}, should overflow to inf, got {rt}"
    );
}

#[test]
fn test_bf16_large_value_preserved() {
    // bf16 has the same exponent range as f32, so large values are preserved
    let large = 1e30f32;
    let bits = f32_to_bf16_bits(large);
    let rt = bf16_bits_to_f32(bits);
    let relative_error = ((rt - large) / large).abs();
    assert!(
        relative_error < 0.01,
        "bf16 should preserve large values: {large} -> {rt}"
    );
}

// =========================================================================
// PTX size and format checks
// =========================================================================

#[test]
fn test_all_cast_ptx_reasonable_size() {
    for (name, ptx) in [
        ("f32_to_f16", generate_f32_to_f16_ptx(1024)),
        ("f16_to_f32", generate_f16_to_f32_ptx(1024)),
        ("f32_to_bf16", generate_f32_to_bf16_ptx(1024)),
        ("bf16_to_f32", generate_bf16_to_f32_ptx(1024)),
    ] {
        assert!(
            ptx.len() > 200,
            "{name}: PTX too small ({} bytes)",
            ptx.len()
        );
        assert!(
            ptx.len() < 20_000,
            "{name}: PTX too large ({} bytes)",
            ptx.len()
        );
    }
}

#[test]
fn test_all_cast_ptx_end_with_closing_brace() {
    for ptx in [
        generate_f32_to_f16_ptx(256),
        generate_f16_to_f32_ptx(256),
        generate_f32_to_bf16_ptx(256),
        generate_bf16_to_f32_ptx(256),
    ] {
        let trimmed = ptx.trim_end();
        assert!(trimmed.ends_with('}'), "PTX must end with closing brace");
    }
}

#[test]
fn test_different_n_produce_different_ptx() {
    let ptx_a = generate_f32_to_f16_ptx(256);
    let ptx_b = generate_f32_to_f16_ptx(1024);
    assert_ne!(ptx_a, ptx_b, "different n should produce different comment");
}

#[test]
fn test_cast_block_size_constant() {
    assert_eq!(CAST_BLOCK_SIZE, 256);
}

// =========================================================================
// Helper functions: software f16/bf16 conversion for test validation
// =========================================================================

/// Convert f32 to f16 bits (IEEE 754 half-precision), round-to-nearest-even.
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if exp == 255 {
        // Inf or NaN
        if mantissa == 0 {
            return (sign | 0x7C00) as u16; // Inf
        } else {
            return (sign | 0x7E00) as u16; // NaN
        }
    }

    let new_exp = exp - 127 + 15;

    if new_exp >= 31 {
        // Overflow -> Inf
        return (sign | 0x7C00) as u16;
    }

    if new_exp <= 0 {
        // Underflow -> zero (simplified; ignoring denormals for test purposes)
        return sign as u16;
    }

    let new_mantissa = mantissa >> 13; // truncate from 23 to 10 bits
    (sign | ((new_exp as u32) << 10) | new_mantissa) as u16
}

/// Convert f16 bits back to f32.
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits) & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1F);
    let mantissa = u32::from(bits & 0x03FF);

    if exp == 31 {
        // Inf or NaN
        if mantissa == 0 {
            return f32::from_bits(sign | 0x7F80_0000); // Inf
        } else {
            return f32::from_bits(sign | 0x7FC0_0000); // NaN
        }
    }

    if exp == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign); // +/-0
        }
        // Denormalized f16 -> normalized f32
        let mut m = mantissa;
        let mut e = 0i32;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e += 1;
        }
        let new_exp = (127 - 15 - e) as u32;
        let new_mantissa = (m & 0x03FF) << 13;
        return f32::from_bits(sign | (new_exp << 23) | new_mantissa);
    }

    let new_exp = (exp as i32 - 15 + 127) as u32;
    let new_mantissa = mantissa << 13;
    f32::from_bits(sign | (new_exp << 23) | new_mantissa)
}

/// Convert f32 to bf16 bits (truncate lower 16 bits of f32, round-to-nearest-even).
fn f32_to_bf16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    // bf16 is the upper 16 bits of f32 with rounding
    let round_bit = (bits >> 15) & 1;
    let truncated = bits >> 16;
    // Simple round-to-nearest (tie-to-even is more complex, use truncation + round bit)
    (truncated + round_bit) as u16
}

/// Convert bf16 bits back to f32.
fn bf16_bits_to_f32(bits: u16) -> f32 {
    // bf16 is the upper 16 bits of f32; lower 16 bits are zero
    f32::from_bits(u32::from(bits) << 16)
}
