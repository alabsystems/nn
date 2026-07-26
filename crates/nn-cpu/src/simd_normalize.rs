// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized normalization operations with explicit entry points.
//!
//! This module provides named entry points for each SIMD tier:
//! - `*_neon` — NEON-optimized (aarch64)
//! - `*_avx2` — AVX2-optimized (x86_64)
//! - `*_scalar` / `*_reference` — pure scalar fallback
//! - dispatch functions — auto-select best available
//!
//! Operations:
//! - `l2_normalize` — L2 normalization: x / ||x||_2
//! - `l1_normalize` — L1 normalization: x / ||x||_1
//! - `min_max_normalize` — min-max scaling: (x - min) / (max - min)
//!
//! All SIMD paths use the reduction primitives from `simd_reduce` for
//! computing norms, then vectorized division for the normalize step.

use crate::simd_reduce;

// ---------------------------------------------------------------------------
// Scalar reference implementations
// ---------------------------------------------------------------------------

/// L2 normalization (scalar): `output[i] = input[i] / ||input||_2`.
///
/// If the L2 norm is zero (all-zero input), output is all zeros.
/// `input` and `output` must have the same length.
pub fn l2_normalize_scalar(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal length"
    );
    let norm = simd_reduce::l2_norm_f32_scalar(input);
    if norm == 0.0 {
        for o in output.iter_mut() {
            *o = 0.0;
        }
        return;
    }
    let inv_norm = 1.0 / norm;
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = x * inv_norm;
    }
}

/// L2 normalization reference — allocates and returns result.
pub fn l2_normalize_reference(input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    l2_normalize_scalar(input, &mut output);
    output
}

/// L1 normalization (scalar): `output[i] = input[i] / ||input||_1`.
///
/// If the L1 norm is zero (all-zero input), output is all zeros.
/// `input` and `output` must have the same length.
pub fn l1_normalize_scalar(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal length"
    );
    let mut abs_sum = 0.0_f32;
    for &x in input {
        abs_sum += x.abs();
    }
    if abs_sum == 0.0 {
        for o in output.iter_mut() {
            *o = 0.0;
        }
        return;
    }
    let inv_norm = 1.0 / abs_sum;
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = x * inv_norm;
    }
}

/// L1 normalization reference — allocates and returns result.
pub fn l1_normalize_reference(input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    l1_normalize_scalar(input, &mut output);
    output
}

/// Min-max normalization (scalar): `output[i] = (input[i] - min) / (max - min)`.
///
/// If all values are the same (max == min), output is all zeros.
/// `input` and `output` must have the same length.
pub fn min_max_normalize_scalar(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal length"
    );
    if input.is_empty() {
        return;
    }
    let min_val = simd_reduce::min_f32_scalar(input);
    let max_val = simd_reduce::max_f32_scalar(input);
    let range = max_val - min_val;
    if range == 0.0 {
        for o in output.iter_mut() {
            *o = 0.0;
        }
        return;
    }
    let inv_range = 1.0 / range;
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = (x - min_val) * inv_range;
    }
}

/// Min-max normalization reference — allocates and returns result.
pub fn min_max_normalize_reference(input: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    min_max_normalize_scalar(input, &mut output);
    output
}

// ---------------------------------------------------------------------------
// NEON (aarch64) — 128-bit, 4x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
pub fn l2_normalize_neon(input: &[f32], output: &mut [f32]) {
    use std::arch::aarch64::*;

    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal length"
    );
    let norm = simd_reduce::l2_norm_f32_neon(input);
    if norm == 0.0 {
        for o in output.iter_mut() {
            *o = 0.0;
        }
        return;
    }
    let inv_norm = 1.0 / norm;
    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;
    let tail_start = chunks * 4;

    // SAFETY: aarch64 NEON is always available. Bounded loads/stores within slice.
    unsafe {
        let vinv = vdupq_n_f32(inv_norm);
        for i in 0..chunks {
            let offset = i * 4;
            let v = vld1q_f32(input.as_ptr().add(offset));
            let r = vmulq_f32(v, vinv);
            vst1q_f32(output.as_mut_ptr().add(offset), r);
        }
    }
    for i in 0..remainder {
        output[tail_start + i] = input[tail_start + i] * inv_norm;
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn l2_normalize_neon(input: &[f32], output: &mut [f32]) {
    l2_normalize_scalar(input, output);
}

#[cfg(target_arch = "aarch64")]
pub fn l1_normalize_neon(input: &[f32], output: &mut [f32]) {
    use std::arch::aarch64::*;

    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal length"
    );
    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;
    let tail_start = chunks * 4;

    // Compute L1 norm (sum of absolute values) using NEON.
    // SAFETY: aarch64 NEON is always available. Bounded loads within slice.
    let abs_sum = unsafe {
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(input.as_ptr().add(i * 4));
            acc = vaddq_f32(acc, vabsq_f32(v));
        }
        let pair = vpaddq_f32(acc, acc);
        let mut s = vgetq_lane_f32::<0>(vpaddq_f32(pair, pair));
        for i in 0..remainder {
            s += input[tail_start + i].abs();
        }
        s
    };

    if abs_sum == 0.0 {
        for o in output.iter_mut() {
            *o = 0.0;
        }
        return;
    }
    let inv_norm = 1.0 / abs_sum;

    // SAFETY: aarch64 NEON is always available. Bounded loads/stores within slice.
    unsafe {
        let vinv = vdupq_n_f32(inv_norm);
        for i in 0..chunks {
            let offset = i * 4;
            let v = vld1q_f32(input.as_ptr().add(offset));
            let r = vmulq_f32(v, vinv);
            vst1q_f32(output.as_mut_ptr().add(offset), r);
        }
    }
    for i in 0..remainder {
        output[tail_start + i] = input[tail_start + i] * inv_norm;
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn l1_normalize_neon(input: &[f32], output: &mut [f32]) {
    l1_normalize_scalar(input, output);
}

#[cfg(target_arch = "aarch64")]
pub fn min_max_normalize_neon(input: &[f32], output: &mut [f32]) {
    use std::arch::aarch64::*;

    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal length"
    );
    if input.is_empty() {
        return;
    }
    let min_val = simd_reduce::min_f32_neon(input);
    let max_val = simd_reduce::max_f32_neon(input);
    let range = max_val - min_val;
    if range == 0.0 {
        for o in output.iter_mut() {
            *o = 0.0;
        }
        return;
    }
    let inv_range = 1.0 / range;
    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;
    let tail_start = chunks * 4;

    // SAFETY: aarch64 NEON is always available. Bounded loads/stores within slice.
    unsafe {
        let vmin = vdupq_n_f32(min_val);
        let vinv = vdupq_n_f32(inv_range);
        for i in 0..chunks {
            let offset = i * 4;
            let v = vld1q_f32(input.as_ptr().add(offset));
            let shifted = vsubq_f32(v, vmin);
            let r = vmulq_f32(shifted, vinv);
            vst1q_f32(output.as_mut_ptr().add(offset), r);
        }
    }
    for i in 0..remainder {
        output[tail_start + i] = (input[tail_start + i] - min_val) * inv_range;
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn min_max_normalize_neon(input: &[f32], output: &mut [f32]) {
    min_max_normalize_scalar(input, output);
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64) — 256-bit, 8x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
pub fn l2_normalize_avx2(input: &[f32], output: &mut [f32]) {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 detected above.
        unsafe { l2_normalize_avx2_inner(input, output) };
    } else {
        l2_normalize_scalar(input, output);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_normalize_avx2_inner(input: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal length"
    );
    let norm = simd_reduce::l2_norm_f32_avx2(input);
    if norm == 0.0 {
        for o in output.iter_mut() {
            *o = 0.0;
        }
        return;
    }
    let inv_norm = 1.0 / norm;
    let n = input.len();
    let chunks = n / 8;
    let remainder = n % 8;
    let tail_start = chunks * 8;

    let vinv = _mm256_set1_ps(inv_norm);
    for i in 0..chunks {
        let offset = i * 8;
        let v = _mm256_loadu_ps(input.as_ptr().add(offset));
        let r = _mm256_mul_ps(v, vinv);
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), r);
    }
    for i in 0..remainder {
        output[tail_start + i] = input[tail_start + i] * inv_norm;
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn l2_normalize_avx2(input: &[f32], output: &mut [f32]) {
    l2_normalize_scalar(input, output);
}

#[cfg(target_arch = "x86_64")]
pub fn l1_normalize_avx2(input: &[f32], output: &mut [f32]) {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 detected above.
        unsafe { l1_normalize_avx2_inner(input, output) };
    } else {
        l1_normalize_scalar(input, output);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l1_normalize_avx2_inner(input: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal length"
    );
    let n = input.len();
    let chunks = n / 8;
    let remainder = n % 8;
    let tail_start = chunks * 8;

    // Compute L1 norm (sum of absolute values) using AVX2.
    // Sign bit mask: clear the top bit to compute abs.
    let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFFu32 as i32));
    let mut acc = _mm256_setzero_ps();
    for i in 0..chunks {
        let v = _mm256_loadu_ps(input.as_ptr().add(i * 8));
        let abs_v = _mm256_and_ps(v, sign_mask);
        acc = _mm256_add_ps(acc, abs_v);
    }
    // Horizontal sum of acc.
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut abs_sum = _mm_cvtss_f32(result);
    for i in 0..remainder {
        abs_sum += input[tail_start + i].abs();
    }

    if abs_sum == 0.0 {
        for o in output.iter_mut() {
            *o = 0.0;
        }
        return;
    }
    let inv_norm = 1.0 / abs_sum;

    let vinv = _mm256_set1_ps(inv_norm);
    for i in 0..chunks {
        let offset = i * 8;
        let v = _mm256_loadu_ps(input.as_ptr().add(offset));
        let r = _mm256_mul_ps(v, vinv);
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), r);
    }
    for i in 0..remainder {
        output[tail_start + i] = input[tail_start + i] * inv_norm;
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn l1_normalize_avx2(input: &[f32], output: &mut [f32]) {
    l1_normalize_scalar(input, output);
}

#[cfg(target_arch = "x86_64")]
pub fn min_max_normalize_avx2(input: &[f32], output: &mut [f32]) {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 detected above.
        unsafe { min_max_normalize_avx2_inner(input, output) };
    } else {
        min_max_normalize_scalar(input, output);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn min_max_normalize_avx2_inner(input: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    assert_eq!(
        input.len(),
        output.len(),
        "input and output must have equal length"
    );
    if input.is_empty() {
        return;
    }
    let min_val = simd_reduce::min_f32_avx2(input);
    let max_val = simd_reduce::max_f32_avx2(input);
    let range = max_val - min_val;
    if range == 0.0 {
        for o in output.iter_mut() {
            *o = 0.0;
        }
        return;
    }
    let inv_range = 1.0 / range;
    let n = input.len();
    let chunks = n / 8;
    let remainder = n % 8;
    let tail_start = chunks * 8;

    let vmin = _mm256_set1_ps(min_val);
    let vinv = _mm256_set1_ps(inv_range);
    for i in 0..chunks {
        let offset = i * 8;
        let v = _mm256_loadu_ps(input.as_ptr().add(offset));
        let shifted = _mm256_sub_ps(v, vmin);
        let r = _mm256_mul_ps(shifted, vinv);
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), r);
    }
    for i in 0..remainder {
        output[tail_start + i] = (input[tail_start + i] - min_val) * inv_range;
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn min_max_normalize_avx2(input: &[f32], output: &mut [f32]) {
    min_max_normalize_scalar(input, output);
}

// ---------------------------------------------------------------------------
// Public dispatch: auto-selects best SIMD path
// ---------------------------------------------------------------------------

/// L2 normalization: `output[i] = input[i] / ||input||_2`.
/// Auto-dispatches to NEON/AVX2/scalar.
///
/// If the L2 norm is zero, output is all zeros.
pub fn l2_normalize(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        l2_normalize_neon(input, output);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            l2_normalize_avx2(input, output);
            return;
        }
    }

    #[allow(unreachable_code)]
    l2_normalize_scalar(input, output);
}

/// L1 normalization: `output[i] = input[i] / ||input||_1`.
/// Auto-dispatches to NEON/AVX2/scalar.
///
/// If the L1 norm is zero, output is all zeros.
pub fn l1_normalize(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        l1_normalize_neon(input, output);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            l1_normalize_avx2(input, output);
            return;
        }
    }

    #[allow(unreachable_code)]
    l1_normalize_scalar(input, output);
}

/// Min-max normalization: `output[i] = (input[i] - min) / (max - min)`.
/// Auto-dispatches to NEON/AVX2/scalar.
///
/// If all values are the same, output is all zeros.
pub fn min_max_normalize(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        min_max_normalize_neon(input, output);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            min_max_normalize_avx2(input, output);
            return;
        }
    }

    #[allow(unreachable_code)]
    min_max_normalize_scalar(input, output);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "simd_normalize_tests.rs"]
mod simd_normalize_tests;
