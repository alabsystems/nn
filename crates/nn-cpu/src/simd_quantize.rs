// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized quantization and dequantization operations.
//!
//! Provides per-tensor and per-channel int8 quantization with SIMD acceleration:
//! - NEON-optimized (aarch64)
//! - AVX2-optimized (x86_64)
//! - Pure scalar fallback
//!
//! Quantization formula: `q = clamp(round(x / scale) + zero_point, -128, 127)`
//! Dequantization formula: `x = (q - zero_point) * scale`

// ---------------------------------------------------------------------------
// Reference (scalar) implementations
// ---------------------------------------------------------------------------

/// Reference per-tensor f32 → i8 quantization (scalar, no SIMD).
///
/// `q[i] = clamp(round(input[i] / scale) + zero_point, -128, 127)`
pub fn quantize_f32_to_i8_reference(input: &[f32], output: &mut [i8], scale: f32, zero_point: i8) {
    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal length"
    );
    let zp = f32::from(zero_point);
    for i in 0..input.len() {
        let q = (input[i] / scale).round() + zp;
        output[i] = q.clamp(-128.0, 127.0) as i8;
    }
}

/// Reference per-tensor i8 → f32 dequantization (scalar, no SIMD).
///
/// `x[i] = (input[i] - zero_point) * scale`
pub fn dequantize_i8_to_f32_reference(
    input: &[i8],
    output: &mut [f32],
    scale: f32,
    zero_point: i8,
) {
    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal length"
    );
    let zp = f32::from(zero_point);
    for i in 0..input.len() {
        output[i] = (f32::from(input[i]) - zp) * scale;
    }
}

/// Reference per-channel f32 → i8 quantization (scalar, no SIMD).
///
/// Each channel has its own `scale` and `zero_point`. Layout is
/// `[channels, elements_per_channel]` (contiguous per channel).
pub fn quantize_per_channel_reference(
    input: &[f32],
    output: &mut [i8],
    scales: &[f32],
    zero_points: &[i8],
    channels: usize,
    elements_per_channel: usize,
) {
    assert_eq!(scales.len(), channels);
    assert_eq!(zero_points.len(), channels);
    assert_eq!(input.len(), channels * elements_per_channel);
    assert_eq!(output.len(), channels * elements_per_channel);

    for c in 0..channels {
        let off = c * elements_per_channel;
        let s = scales[c];
        let zp = f32::from(zero_points[c]);
        for i in 0..elements_per_channel {
            let q = (input[off + i] / s).round() + zp;
            output[off + i] = q.clamp(-128.0, 127.0) as i8;
        }
    }
}

// ---------------------------------------------------------------------------
// SIMD-accelerated per-tensor quantization
// ---------------------------------------------------------------------------

/// Per-tensor f32 → i8 quantization. Auto-dispatches to NEON/AVX2/scalar.
///
/// `q[i] = clamp(round(input[i] / scale) + zero_point, -128, 127)`
pub fn quantize_f32_to_i8(input: &[f32], output: &mut [i8], scale: f32, zero_point: i8) {
    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal length"
    );

    #[cfg(target_arch = "aarch64")]
    {
        quantize_f32_to_i8_neon(input, output, scale, zero_point);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            unsafe { quantize_f32_to_i8_avx2(input, output, scale, zero_point) };
            return;
        }
    }

    #[allow(unreachable_code)]
    quantize_f32_to_i8_reference(input, output, scale, zero_point);
}

/// Per-tensor i8 → f32 dequantization. Auto-dispatches to NEON/AVX2/scalar.
///
/// `x[i] = (input[i] - zero_point) * scale`
pub fn dequantize_i8_to_f32(input: &[i8], output: &mut [f32], scale: f32, zero_point: i8) {
    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal length"
    );

    #[cfg(target_arch = "aarch64")]
    {
        dequantize_i8_to_f32_neon(input, output, scale, zero_point);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            unsafe { dequantize_i8_to_f32_avx2(input, output, scale, zero_point) };
            return;
        }
    }

    #[allow(unreachable_code)]
    dequantize_i8_to_f32_reference(input, output, scale, zero_point);
}

/// Per-channel f32 → i8 quantization. Auto-dispatches to NEON/AVX2/scalar.
pub fn quantize_per_channel(
    input: &[f32],
    output: &mut [i8],
    scales: &[f32],
    zero_points: &[i8],
    channels: usize,
    elements_per_channel: usize,
) {
    assert_eq!(scales.len(), channels);
    assert_eq!(zero_points.len(), channels);
    assert_eq!(input.len(), channels * elements_per_channel);
    assert_eq!(output.len(), channels * elements_per_channel);

    for c in 0..channels {
        let off = c * elements_per_channel;
        quantize_f32_to_i8(
            &input[off..off + elements_per_channel],
            &mut output[off..off + elements_per_channel],
            scales[c],
            zero_points[c],
        );
    }
}

// ---------------------------------------------------------------------------
// NEON (aarch64) — 128-bit, 4x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
fn quantize_f32_to_i8_neon(input: &[f32], output: &mut [i8], scale: f32, zero_point: i8) {
    use std::arch::aarch64::*;

    let n = input.len();
    let inv_scale = 1.0 / scale;
    let zp = f32::from(zero_point);
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: aarch64 NEON is always available. Bounded loads/stores within slice.
    unsafe {
        let v_inv_scale = vdupq_n_f32(inv_scale);
        let v_zp = vdupq_n_f32(zp);
        let v_min = vdupq_n_f32(-128.0);
        let v_max = vdupq_n_f32(127.0);

        for i in 0..chunks {
            let offset = i * 4;
            let v = vld1q_f32(input.as_ptr().add(offset));
            // x / scale via multiply by inv_scale
            let scaled = vmulq_f32(v, v_inv_scale);
            // round to nearest
            let rounded = vrndnq_f32(scaled);
            // add zero_point
            let shifted = vaddq_f32(rounded, v_zp);
            // clamp to [-128, 127]
            let clamped = vminq_f32(vmaxq_f32(shifted, v_min), v_max);
            // Convert to i32 then narrow to i8
            let i32_vals = vcvtq_s32_f32(clamped);
            // Extract lanes and write i8 values
            output[offset] = vgetq_lane_s32::<0>(i32_vals) as i8;
            output[offset + 1] = vgetq_lane_s32::<1>(i32_vals) as i8;
            output[offset + 2] = vgetq_lane_s32::<2>(i32_vals) as i8;
            output[offset + 3] = vgetq_lane_s32::<3>(i32_vals) as i8;
        }
    }

    // Scalar tail
    let tail = chunks * 4;
    for i in 0..remainder {
        let q = (input[tail + i] * inv_scale).round() + zp;
        output[tail + i] = q.clamp(-128.0, 127.0) as i8;
    }
}

#[cfg(target_arch = "aarch64")]
fn dequantize_i8_to_f32_neon(input: &[i8], output: &mut [f32], scale: f32, zero_point: i8) {
    use std::arch::aarch64::*;

    let n = input.len();
    let zp = f32::from(zero_point);
    let chunks = n / 4;
    let remainder = n % 4;

    // SAFETY: aarch64 NEON is always available. Bounded loads/stores within slice.
    unsafe {
        let v_scale = vdupq_n_f32(scale);
        let v_zp = vdupq_n_f32(zp);

        for i in 0..chunks {
            let offset = i * 4;
            // Load 4 i8 values, widen to i32, convert to f32
            let i0 = i32::from(input[offset]);
            let i1 = i32::from(input[offset + 1]);
            let i2 = i32::from(input[offset + 2]);
            let i3 = i32::from(input[offset + 3]);
            let i32_vec = vcombine_s32(
                vcreate_s32(u64::from(i0 as u32) | (u64::from(i1 as u32) << 32)),
                vcreate_s32(u64::from(i2 as u32) | (u64::from(i3 as u32) << 32)),
            );
            let f32_vec = vcvtq_f32_s32(i32_vec);
            // (q - zero_point) * scale
            let shifted = vsubq_f32(f32_vec, v_zp);
            let result = vmulq_f32(shifted, v_scale);
            vst1q_f32(output.as_mut_ptr().add(offset), result);
        }
    }

    // Scalar tail
    let tail = chunks * 4;
    for i in 0..remainder {
        output[tail + i] = (f32::from(input[tail + i]) - zp) * scale;
    }
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64) — 256-bit, 8x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_f32_to_i8_avx2(input: &[f32], output: &mut [i8], scale: f32, zero_point: i8) {
    use std::arch::x86_64::*;

    let n = input.len();
    let inv_scale = 1.0 / scale;
    let zp = zero_point as f32;
    let chunks = n / 8;
    let remainder = n % 8;

    let v_inv_scale = _mm256_set1_ps(inv_scale);
    let v_zp = _mm256_set1_ps(zp);
    let v_min = _mm256_set1_ps(-128.0);
    let v_max = _mm256_set1_ps(127.0);

    for i in 0..chunks {
        let offset = i * 8;
        // SAFETY: offset + 8 <= n from loop bound. Unaligned loads.
        let v = _mm256_loadu_ps(input.as_ptr().add(offset));
        let scaled = _mm256_mul_ps(v, v_inv_scale);
        let rounded = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(scaled);
        let shifted = _mm256_add_ps(rounded, v_zp);
        let clamped = _mm256_min_ps(_mm256_max_ps(shifted, v_min), v_max);
        let i32_vals = _mm256_cvtps_epi32(clamped);

        // Extract i32 lanes to i8
        let mut buf = [0i32; 8];
        _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, i32_vals);
        for j in 0..8 {
            output[offset + j] = buf[j] as i8;
        }
    }

    // Scalar tail
    let tail = chunks * 8;
    for i in 0..remainder {
        let q = (input[tail + i] * inv_scale).round() + zp;
        output[tail + i] = q.clamp(-128.0, 127.0) as i8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequantize_i8_to_f32_avx2(input: &[i8], output: &mut [f32], scale: f32, zero_point: i8) {
    use std::arch::x86_64::*;

    let n = input.len();
    let zp = zero_point as f32;
    let chunks = n / 8;
    let remainder = n % 8;

    let v_scale = _mm256_set1_ps(scale);
    let v_zp = _mm256_set1_ps(zp);

    for i in 0..chunks {
        let offset = i * 8;
        // Load 8 i8 values, widen to i32, convert to f32
        let mut ibuf = [0i32; 8];
        for j in 0..8 {
            ibuf[j] = input[offset + j] as i32;
        }
        let i32_vec = _mm256_loadu_si256(ibuf.as_ptr() as *const __m256i);
        let f32_vec = _mm256_cvtepi32_ps(i32_vec);
        let shifted = _mm256_sub_ps(f32_vec, v_zp);
        let result = _mm256_mul_ps(shifted, v_scale);
        // SAFETY: offset + 8 <= n from loop bound. Unaligned stores.
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), result);
    }

    // Scalar tail
    let tail = chunks * 8;
    for i in 0..remainder {
        output[tail + i] = (input[tail + i] as f32 - zp) * scale;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "simd_quantize_tests.rs"]
mod simd_quantize_tests;
