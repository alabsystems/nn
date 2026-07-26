// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized dtype casting between f32, f16, and bf16.
//!
//! f16 and bf16 are stored as `u16` (raw bit representation) on CPU.
//!
//! This module provides:
//! - `f32_to_f16` / `f16_to_f32` — IEEE 754 half-precision conversion
//! - `f32_to_bf16` / `bf16_to_f32` — bfloat16 conversion (truncate/shift)
//!
//! NEON (aarch64) and AVX2 (x86_64) paths use integer bit manipulation.
//! Scalar fallbacks are always available.

// ---------------------------------------------------------------------------
// Scalar fallback (always compiled, no cfg gate)
// ---------------------------------------------------------------------------

/// Scalar f32 to f16 conversion.
///
/// Uses IEEE 754 half-precision encoding: 1 sign, 5 exponent, 10 mantissa bits.
#[inline]
fn f32_to_f16_scalar_one(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if exponent == 255 {
        // Inf or NaN
        if mantissa != 0 {
            // NaN — preserve a non-zero mantissa bit
            return (sign | 0x7C00 | (mantissa >> 13).max(1)) as u16;
        }
        return (sign | 0x7C00) as u16;
    }

    let unbiased = exponent - 127;

    if unbiased > 15 {
        // Overflow → Inf
        return (sign | 0x7C00) as u16;
    }

    if unbiased < -24 {
        // Too small for even subnormal f16 → zero
        return sign as u16;
    }

    if unbiased < -14 {
        // Subnormal f16
        let shift = -1 - unbiased + 10;
        let subnormal = (0x0080_0000 | mantissa) >> shift;
        return (sign | subnormal) as u16;
    }

    // Normal f16
    let exp16 = ((unbiased + 15) as u32) << 10;
    let man16 = mantissa >> 13;
    // Round to nearest even
    let round_bit = (mantissa >> 12) & 1;
    let sticky = mantissa & 0xFFF;
    let rounded = man16
        + if round_bit != 0 && (sticky != 0 || (man16 & 1) != 0) {
            1
        } else {
            0
        };
    (sign | exp16 | rounded) as u16
}

/// Scalar f16 to f32 conversion.
#[inline]
fn f16_to_f32_scalar_one(bits: u16) -> f32 {
    let sign = (u32::from(bits) & 0x8000) << 16;
    let exponent = (u32::from(bits) >> 10) & 0x1F;
    let mantissa = u32::from(bits) & 0x03FF;

    if exponent == 0 {
        if mantissa == 0 {
            // Zero
            return f32::from_bits(sign);
        }
        // Subnormal f16 → normalize to f32
        let mut m = mantissa;
        let mut e: i32 = -14;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        let exp32 = ((e + 127) as u32) << 23;
        let man32 = m << 13;
        return f32::from_bits(sign | exp32 | man32);
    }

    if exponent == 31 {
        // Inf or NaN
        let man32 = mantissa << 13;
        return f32::from_bits(sign | 0x7F80_0000 | man32);
    }

    // Normal f16 → f32
    let exp32 = ((exponent as i32 - 15 + 127) as u32) << 23;
    let man32 = mantissa << 13;
    f32::from_bits(sign | exp32 | man32)
}

/// Scalar f32 to bf16: truncate lower 16 bits (round-to-nearest-even).
#[inline]
fn f32_to_bf16_scalar_one(val: f32) -> u16 {
    let bits = val.to_bits();
    // Round to nearest even: add rounding bias
    let round_bit = (bits >> 16) & 1;
    let lsb_any = if (bits & 0xFFFF) > 0x8000 { 1u32 } else { 0 };
    let bias = 0x7FFF + (round_bit | lsb_any);
    let rounded = bits.wrapping_add(bias);
    (rounded >> 16) as u16
}

/// Scalar bf16 to f32: shift left by 16 bits.
#[inline]
fn bf16_to_f32_scalar_one(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Convert f32 slice to f16 (stored as u16). Scalar fallback.
pub fn f32_to_f16_scalar(input: &[f32], output: &mut [u16]) {
    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal length"
    );
    for i in 0..input.len() {
        output[i] = f32_to_f16_scalar_one(input[i]);
    }
}

/// Convert f16 (stored as u16) to f32. Scalar fallback.
pub fn f16_to_f32_scalar(input: &[u16], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal length"
    );
    for i in 0..input.len() {
        output[i] = f16_to_f32_scalar_one(input[i]);
    }
}

/// Convert f32 slice to bf16 (stored as u16). Scalar fallback.
pub fn f32_to_bf16_scalar(input: &[f32], output: &mut [u16]) {
    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal length"
    );
    for i in 0..input.len() {
        output[i] = f32_to_bf16_scalar_one(input[i]);
    }
}

/// Convert bf16 (stored as u16) to f32. Scalar fallback.
pub fn bf16_to_f32_scalar(input: &[u16], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal length"
    );
    for i in 0..input.len() {
        output[i] = bf16_to_f32_scalar_one(input[i]);
    }
}

// ---------------------------------------------------------------------------
// NEON (aarch64) — 128-bit, 4x f32 / 4x u16 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
pub fn f32_to_f16_neon(input: &[f32], output: &mut [u16]) {
    // `vcvt_f16_f32` is still unstable on stable Rust (`stdarch_neon_f16`).
    // Keep the public NEON entrypoint but fall back to the stable scalar path.
    f32_to_f16_scalar(input, output);
}

#[cfg(not(target_arch = "aarch64"))]
pub fn f32_to_f16_neon(input: &[f32], output: &mut [u16]) {
    f32_to_f16_scalar(input, output);
}

#[cfg(target_arch = "aarch64")]
pub fn f16_to_f32_neon(input: &[u16], output: &mut [f32]) {
    // `vcvt_f32_f16` is still unstable on stable Rust (`stdarch_neon_f16`).
    // Keep the public NEON entrypoint but fall back to the stable scalar path.
    f16_to_f32_scalar(input, output);
}

#[cfg(not(target_arch = "aarch64"))]
pub fn f16_to_f32_neon(input: &[u16], output: &mut [f32]) {
    f16_to_f32_scalar(input, output);
}

#[cfg(target_arch = "aarch64")]
pub fn f32_to_bf16_neon(input: &[f32], output: &mut [u16]) {
    use std::arch::aarch64::*;

    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal length"
    );
    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: aarch64 NEON is always available. bf16 = f32 bits >> 16 with rounding.
    // We use integer NEON ops on the bit representation.
    unsafe {
        let bias_base = vdupq_n_u32(0x7FFF);
        for i in 0..chunks {
            let offset = i * 4;
            let v = vld1q_f32(input.as_ptr().add(offset));
            let bits = vreinterpretq_u32_f32(v);
            // Round-to-nearest-even: bias = 0x7FFF + ((bits >> 16) & 1)
            let lsb = vshrq_n_u32::<16>(bits);
            let lsb_masked = vandq_u32(lsb, vdupq_n_u32(1));
            let bias = vaddq_u32(bias_base, lsb_masked);
            let rounded = vaddq_u32(bits, bias);
            let shifted = vshrq_n_u32::<16>(rounded);
            // Narrow 4x u32 → 4x u16
            let narrow = vmovn_u32(shifted);
            vst1_u16(output.as_mut_ptr().add(offset), narrow);
        }
    }
    let tail = chunks * 4;
    for i in 0..remainder {
        output[tail + i] = f32_to_bf16_scalar_one(input[tail + i]);
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn f32_to_bf16_neon(input: &[f32], output: &mut [u16]) {
    f32_to_bf16_scalar(input, output);
}

#[cfg(target_arch = "aarch64")]
pub fn bf16_to_f32_neon(input: &[u16], output: &mut [f32]) {
    use std::arch::aarch64::*;

    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal length"
    );
    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: aarch64 NEON is always available. bf16→f32 is just shift left 16.
    unsafe {
        for i in 0..chunks {
            let offset = i * 4;
            let h = vld1_u16(input.as_ptr().add(offset));
            // Widen 4x u16 → 4x u32
            let wide = vmovl_u16(h);
            let shifted = vshlq_n_u32::<16>(wide);
            let result = vreinterpretq_f32_u32(shifted);
            vst1q_f32(output.as_mut_ptr().add(offset), result);
        }
    }
    let tail = chunks * 4;
    for i in 0..remainder {
        output[tail + i] = bf16_to_f32_scalar_one(input[tail + i]);
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn bf16_to_f32_neon(input: &[u16], output: &mut [f32]) {
    bf16_to_f32_scalar(input, output);
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64) — 256-bit, 8x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
pub fn f32_to_f16_avx2(input: &[f32], output: &mut [u16]) {
    if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx2") {
        // SAFETY: f16c + AVX2 detected above.
        unsafe { f32_to_f16_avx2_inner(input, output) };
    } else {
        f32_to_f16_scalar(input, output);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "f16c")]
unsafe fn f32_to_f16_avx2_inner(input: &[f32], output: &mut [u16]) {
    use std::arch::x86_64::*;

    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal length"
    );
    let n = input.len();
    let chunks = n / 8;
    let remainder = n % 8;

    for i in 0..chunks {
        let offset = i * 8;
        // SAFETY: offset + 8 <= n from loop bound.
        let v = _mm256_loadu_ps(input.as_ptr().add(offset));
        let h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128(output.as_mut_ptr().add(offset) as *mut __m128i, h);
    }
    let tail = chunks * 8;
    for i in 0..remainder {
        output[tail + i] = f32_to_f16_scalar_one(input[tail + i]);
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn f32_to_f16_avx2(input: &[f32], output: &mut [u16]) {
    f32_to_f16_scalar(input, output);
}

#[cfg(target_arch = "x86_64")]
pub fn f16_to_f32_avx2(input: &[u16], output: &mut [f32]) {
    if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx2") {
        // SAFETY: f16c + AVX2 detected above.
        unsafe { f16_to_f32_avx2_inner(input, output) };
    } else {
        f16_to_f32_scalar(input, output);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "f16c")]
unsafe fn f16_to_f32_avx2_inner(input: &[u16], output: &mut [f32]) {
    use std::arch::x86_64::*;

    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal length"
    );
    let n = input.len();
    let chunks = n / 8;
    let remainder = n % 8;

    for i in 0..chunks {
        let offset = i * 8;
        // SAFETY: offset + 8 <= n from loop bound.
        let h = _mm_loadu_si128(input.as_ptr().add(offset) as *const __m128i);
        let v = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), v);
    }
    let tail = chunks * 8;
    for i in 0..remainder {
        output[tail + i] = f16_to_f32_scalar_one(input[tail + i]);
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn f16_to_f32_avx2(input: &[u16], output: &mut [f32]) {
    f16_to_f32_scalar(input, output);
}

#[cfg(target_arch = "x86_64")]
pub fn f32_to_bf16_avx2(input: &[f32], output: &mut [u16]) {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 detected above.
        unsafe { f32_to_bf16_avx2_inner(input, output) };
    } else {
        f32_to_bf16_scalar(input, output);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn f32_to_bf16_avx2_inner(input: &[f32], output: &mut [u16]) {
    use std::arch::x86_64::*;

    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal length"
    );
    let n = input.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let bias_base = _mm256_set1_epi32(0x7FFF_i32);
    let one = _mm256_set1_epi32(1);
    for i in 0..chunks {
        let offset = i * 8;
        // SAFETY: offset + 8 <= n from loop bound.
        let v = _mm256_loadu_ps(input.as_ptr().add(offset));
        let bits = _mm256_castps_si256(v);
        // Round-to-nearest-even: bias = 0x7FFF + ((bits >> 16) & 1)
        let lsb = _mm256_srli_epi32::<16>(bits);
        let lsb_masked = _mm256_and_si256(lsb, one);
        let bias = _mm256_add_epi32(bias_base, lsb_masked);
        let rounded = _mm256_add_epi32(bits, bias);
        let shifted = _mm256_srli_epi32::<16>(rounded);
        // Pack 8x i32 → 8x i16 via two 128-bit halves
        let lo = _mm256_castsi256_si128(shifted);
        let hi = _mm256_extracti128_si256::<1>(shifted);
        let packed = _mm_packus_epi32(lo, hi);
        _mm_storeu_si128(output.as_mut_ptr().add(offset) as *mut __m128i, packed);
    }
    let tail = chunks * 8;
    for i in 0..remainder {
        output[tail + i] = f32_to_bf16_scalar_one(input[tail + i]);
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn f32_to_bf16_avx2(input: &[f32], output: &mut [u16]) {
    f32_to_bf16_scalar(input, output);
}

#[cfg(target_arch = "x86_64")]
pub fn bf16_to_f32_avx2(input: &[u16], output: &mut [f32]) {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 detected above.
        unsafe { bf16_to_f32_avx2_inner(input, output) };
    } else {
        bf16_to_f32_scalar(input, output);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bf16_to_f32_avx2_inner(input: &[u16], output: &mut [f32]) {
    use std::arch::x86_64::*;

    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal length"
    );
    let n = input.len();
    let chunks = n / 8;
    let remainder = n % 8;

    for i in 0..chunks {
        let offset = i * 8;
        // SAFETY: offset + 8 <= n from loop bound.
        // Load 8x u16, zero-extend to 8x u32 via pmovzx, shift left 16.
        let h = _mm_loadu_si128(input.as_ptr().add(offset) as *const __m128i);
        let wide = _mm256_cvtepu16_epi32(h);
        let shifted = _mm256_slli_epi32::<16>(wide);
        let result = _mm256_castsi256_ps(shifted);
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), result);
    }
    let tail = chunks * 8;
    for i in 0..remainder {
        output[tail + i] = bf16_to_f32_scalar_one(input[tail + i]);
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn bf16_to_f32_avx2(input: &[u16], output: &mut [f32]) {
    bf16_to_f32_scalar(input, output);
}

// ---------------------------------------------------------------------------
// Public dispatch: auto-selects best SIMD path
// ---------------------------------------------------------------------------

/// Convert f32 slice to f16 (stored as u16). Auto-dispatches to NEON/AVX2/scalar.
#[inline]
pub fn f32_to_f16(input: &[f32], output: &mut [u16]) {
    #[cfg(target_arch = "aarch64")]
    {
        f32_to_f16_neon(input, output);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx2") {
            f32_to_f16_avx2(input, output);
            return;
        }
    }

    #[allow(unreachable_code)]
    f32_to_f16_scalar(input, output);
}

/// Convert f16 (stored as u16) to f32. Auto-dispatches to NEON/AVX2/scalar.
#[inline]
pub fn f16_to_f32(input: &[u16], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        f16_to_f32_neon(input, output);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx2") {
            f16_to_f32_avx2(input, output);
            return;
        }
    }

    #[allow(unreachable_code)]
    f16_to_f32_scalar(input, output);
}

/// Convert f32 slice to bf16 (stored as u16). Auto-dispatches to NEON/AVX2/scalar.
#[inline]
pub fn f32_to_bf16(input: &[f32], output: &mut [u16]) {
    #[cfg(target_arch = "aarch64")]
    {
        f32_to_bf16_neon(input, output);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            f32_to_bf16_avx2(input, output);
            return;
        }
    }

    #[allow(unreachable_code)]
    f32_to_bf16_scalar(input, output);
}

/// Convert bf16 (stored as u16) to f32. Auto-dispatches to NEON/AVX2/scalar.
#[inline]
pub fn bf16_to_f32(input: &[u16], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        bf16_to_f32_neon(input, output);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            bf16_to_f32_avx2(input, output);
            return;
        }
    }

    #[allow(unreachable_code)]
    bf16_to_f32_scalar(input, output);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "simd_cast_tests.rs"]
mod simd_cast_tests;
