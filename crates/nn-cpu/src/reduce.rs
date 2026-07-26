// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized reduction operations along the last axis.
//!
//! Provides sum, max, min, mean, argmax, and argmin reductions over
//! contiguous rows of `dim_size` elements. Each row is reduced independently.
//!
//! NEON (aarch64) and AVX2 (x86_64) paths with scalar fallback.

#[cfg(test)]
#[path = "reduce_tests.rs"]
mod reduce_tests;

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

/// Scalar sum reduction along the last axis.
///
/// `input` has shape `[..., dim_size]` flattened. `output` has length
/// `input.len() / dim_size`. Each output element is the sum of a
/// contiguous `dim_size`-element row.
pub fn sum_f32_scalar(input: &[f32], output: &mut [f32], dim_size: usize) {
    if dim_size == 0 {
        return;
    }
    assert_eq!(
        input.len() % dim_size,
        0,
        "input length must be a multiple of dim_size"
    );
    let rows = input.len() / dim_size;
    assert_eq!(
        output.len(),
        rows,
        "output length must equal number of rows"
    );

    for row in 0..rows {
        let start = row * dim_size;
        let mut acc = 0.0_f32;
        for i in 0..dim_size {
            acc += input[start + i];
        }
        output[row] = acc;
    }
}

/// Scalar max reduction along the last axis.
///
/// Returns `f32::NEG_INFINITY` for empty rows (dim_size == 0 is a no-op).
pub fn max_f32_scalar(input: &[f32], output: &mut [f32], dim_size: usize) {
    if dim_size == 0 {
        return;
    }
    assert_eq!(
        input.len() % dim_size,
        0,
        "input length must be a multiple of dim_size"
    );
    let rows = input.len() / dim_size;
    assert_eq!(
        output.len(),
        rows,
        "output length must equal number of rows"
    );

    for row in 0..rows {
        let start = row * dim_size;
        let mut acc = f32::NEG_INFINITY;
        for i in 0..dim_size {
            acc = acc.max(input[start + i]);
        }
        output[row] = acc;
    }
}

/// Scalar min reduction along the last axis.
///
/// Returns `f32::INFINITY` for empty rows (dim_size == 0 is a no-op).
pub fn min_f32_scalar(input: &[f32], output: &mut [f32], dim_size: usize) {
    if dim_size == 0 {
        return;
    }
    assert_eq!(
        input.len() % dim_size,
        0,
        "input length must be a multiple of dim_size"
    );
    let rows = input.len() / dim_size;
    assert_eq!(
        output.len(),
        rows,
        "output length must equal number of rows"
    );

    for row in 0..rows {
        let start = row * dim_size;
        let mut acc = f32::INFINITY;
        for i in 0..dim_size {
            acc = acc.min(input[start + i]);
        }
        output[row] = acc;
    }
}

/// Scalar mean reduction along the last axis (sum / count).
pub fn mean_f32_scalar(input: &[f32], output: &mut [f32], dim_size: usize) {
    if dim_size == 0 {
        return;
    }
    assert_eq!(
        input.len() % dim_size,
        0,
        "input length must be a multiple of dim_size"
    );
    let rows = input.len() / dim_size;
    assert_eq!(
        output.len(),
        rows,
        "output length must equal number of rows"
    );

    let inv_n = 1.0 / dim_size as f32;
    for row in 0..rows {
        let start = row * dim_size;
        let mut acc = 0.0_f32;
        for i in 0..dim_size {
            acc += input[start + i];
        }
        output[row] = acc * inv_n;
    }
}

/// Scalar argmax along the last axis. Returns index of max element per row.
///
/// Output elements are `u32` indices into the row. Ties broken by first occurrence.
/// For empty rows (dim_size == 0), this is a no-op.
pub fn argmax_f32_scalar(input: &[f32], output: &mut [u32], dim_size: usize) {
    if dim_size == 0 {
        return;
    }
    assert_eq!(
        input.len() % dim_size,
        0,
        "input length must be a multiple of dim_size"
    );
    let rows = input.len() / dim_size;
    assert_eq!(
        output.len(),
        rows,
        "output length must equal number of rows"
    );

    for row in 0..rows {
        let start = row * dim_size;
        let mut best_val = f32::NEG_INFINITY;
        let mut best_idx = 0_u32;
        for i in 0..dim_size {
            let v = input[start + i];
            if v > best_val {
                best_val = v;
                best_idx = i as u32;
            }
        }
        output[row] = best_idx;
    }
}

/// Scalar argmin along the last axis. Returns index of min element per row.
///
/// Output elements are `u32` indices into the row. Ties broken by first occurrence.
/// For empty rows (dim_size == 0), this is a no-op.
pub fn argmin_f32_scalar(input: &[f32], output: &mut [u32], dim_size: usize) {
    if dim_size == 0 {
        return;
    }
    assert_eq!(
        input.len() % dim_size,
        0,
        "input length must be a multiple of dim_size"
    );
    let rows = input.len() / dim_size;
    assert_eq!(
        output.len(),
        rows,
        "output length must equal number of rows"
    );

    for row in 0..rows {
        let start = row * dim_size;
        let mut best_val = f32::INFINITY;
        let mut best_idx = 0_u32;
        for i in 0..dim_size {
            let v = input[start + i];
            if v < best_val {
                best_val = v;
                best_idx = i as u32;
            }
        }
        output[row] = best_idx;
    }
}

// ---------------------------------------------------------------------------
// NEON (aarch64) — 128-bit, 4x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// Horizontal sum of a float32x4 vector.
    #[inline]
    unsafe fn hsum_f32x4(v: float32x4_t) -> f32 { unsafe {
        let pair = vpaddq_f32(v, v);
        vgetq_lane_f32::<0>(vpaddq_f32(pair, pair))
    }}

    /// Horizontal max of a float32x4 vector.
    #[inline]
    unsafe fn hmax_f32x4(v: float32x4_t) -> f32 { unsafe {
        let a = vgetq_lane_f32::<0>(v);
        let b = vgetq_lane_f32::<1>(v);
        let c = vgetq_lane_f32::<2>(v);
        let d = vgetq_lane_f32::<3>(v);
        a.max(b).max(c.max(d))
    }}

    /// Horizontal min of a float32x4 vector.
    #[inline]
    unsafe fn hmin_f32x4(v: float32x4_t) -> f32 { unsafe {
        let a = vgetq_lane_f32::<0>(v);
        let b = vgetq_lane_f32::<1>(v);
        let c = vgetq_lane_f32::<2>(v);
        let d = vgetq_lane_f32::<3>(v);
        a.min(b).min(c.min(d))
    }}

    /// NEON sum reduction for a single row.
    #[inline]
    unsafe fn sum_row(row: &[f32]) -> f32 { unsafe {
        let n = row.len();
        let chunks = n / 4;
        let remainder = n % 4;

        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(row.as_ptr().add(i * 4));
            acc = vaddq_f32(acc, v);
        }
        let mut result = hsum_f32x4(acc);
        let tail_start = chunks * 4;
        for i in 0..remainder {
            result += row[tail_start + i];
        }
        result
    }}

    /// NEON max reduction for a single row.
    #[inline]
    unsafe fn max_row(row: &[f32]) -> f32 { unsafe {
        let n = row.len();
        let chunks = n / 4;
        let remainder = n % 4;

        let mut acc = vdupq_n_f32(f32::NEG_INFINITY);
        for i in 0..chunks {
            let v = vld1q_f32(row.as_ptr().add(i * 4));
            acc = vmaxq_f32(acc, v);
        }
        let mut result = hmax_f32x4(acc);
        let tail_start = chunks * 4;
        for i in 0..remainder {
            result = result.max(row[tail_start + i]);
        }
        result
    }}

    /// NEON min reduction for a single row.
    #[inline]
    unsafe fn min_row(row: &[f32]) -> f32 { unsafe {
        let n = row.len();
        let chunks = n / 4;
        let remainder = n % 4;

        let mut acc = vdupq_n_f32(f32::INFINITY);
        for i in 0..chunks {
            let v = vld1q_f32(row.as_ptr().add(i * 4));
            acc = vminq_f32(acc, v);
        }
        let mut result = hmin_f32x4(acc);
        let tail_start = chunks * 4;
        for i in 0..remainder {
            result = result.min(row[tail_start + i]);
        }
        result
    }}

    /// NEON argmax for a single row. Returns index of first max element.
    #[inline]
    fn argmax_row(row: &[f32]) -> u32 {
        let mut best_val = f32::NEG_INFINITY;
        let mut best_idx = 0_u32;
        // argmax requires tracking indices — use scalar with NEON max for
        // candidate detection on large rows, but the index tracking is
        // inherently scalar. For simplicity and correctness, use scalar loop
        // (still benefits from CPU prefetch on contiguous access).
        for (i, &v) in row.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i as u32;
            }
        }
        best_idx
    }

    /// NEON argmin for a single row. Returns index of first min element.
    #[inline]
    fn argmin_row(row: &[f32]) -> u32 {
        let mut best_val = f32::INFINITY;
        let mut best_idx = 0_u32;
        for (i, &v) in row.iter().enumerate() {
            if v < best_val {
                best_val = v;
                best_idx = i as u32;
            }
        }
        best_idx
    }

    pub(super) fn sum_f32_neon(input: &[f32], output: &mut [f32], dim_size: usize) {
        let rows = input.len() / dim_size;
        for row in 0..rows {
            let start = row * dim_size;
            // SAFETY: NEON always available on aarch64. Bounded loads within slice.
            output[row] = unsafe { sum_row(&input[start..start + dim_size]) };
        }
    }

    pub(super) fn max_f32_neon(input: &[f32], output: &mut [f32], dim_size: usize) {
        let rows = input.len() / dim_size;
        for row in 0..rows {
            let start = row * dim_size;
            // SAFETY: NEON always available on aarch64. Bounded loads within slice.
            output[row] = unsafe { max_row(&input[start..start + dim_size]) };
        }
    }

    pub(super) fn min_f32_neon(input: &[f32], output: &mut [f32], dim_size: usize) {
        let rows = input.len() / dim_size;
        for row in 0..rows {
            let start = row * dim_size;
            // SAFETY: NEON always available on aarch64. Bounded loads within slice.
            output[row] = unsafe { min_row(&input[start..start + dim_size]) };
        }
    }

    pub(super) fn mean_f32_neon(input: &[f32], output: &mut [f32], dim_size: usize) {
        let rows = input.len() / dim_size;
        let inv_n = 1.0 / dim_size as f32;
        for row in 0..rows {
            let start = row * dim_size;
            // SAFETY: NEON always available on aarch64. Bounded loads within slice.
            let s = unsafe { sum_row(&input[start..start + dim_size]) };
            output[row] = s * inv_n;
        }
    }

    pub(super) fn argmax_f32_neon(input: &[f32], output: &mut [u32], dim_size: usize) {
        let rows = input.len() / dim_size;
        for row in 0..rows {
            let start = row * dim_size;
            output[row] = argmax_row(&input[start..start + dim_size]);
        }
    }

    pub(super) fn argmin_f32_neon(input: &[f32], output: &mut [u32], dim_size: usize) {
        let rows = input.len() / dim_size;
        for row in 0..rows {
            let start = row * dim_size;
            output[row] = argmin_row(&input[start..start + dim_size]);
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64) — 256-bit, 8x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// Horizontal sum of an __m256 register.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hsum_m256(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(result)
    }

    /// Horizontal max of an __m256 register.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hmax_m256(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let max128 = _mm_max_ps(lo, hi);
        let shuf = _mm_movehdup_ps(max128);
        let maxs = _mm_max_ps(max128, shuf);
        let shuf2 = _mm_movehl_ps(maxs, maxs);
        let result = _mm_max_ss(maxs, shuf2);
        _mm_cvtss_f32(result)
    }

    /// Horizontal min of an __m256 register.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hmin_m256(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let min128 = _mm_min_ps(lo, hi);
        let shuf = _mm_movehdup_ps(min128);
        let mins = _mm_min_ps(min128, shuf);
        let shuf2 = _mm_movehl_ps(mins, mins);
        let result = _mm_min_ss(mins, shuf2);
        _mm_cvtss_f32(result)
    }

    /// AVX2 sum reduction for a single row.
    #[target_feature(enable = "avx2")]
    unsafe fn sum_row(row: &[f32]) -> f32 {
        let n = row.len();
        let chunks = n / 8;
        let remainder = n % 8;

        let mut acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let v = _mm256_loadu_ps(row.as_ptr().add(i * 8));
            acc = _mm256_add_ps(acc, v);
        }
        let mut result = hsum_m256(acc);
        let tail_start = chunks * 8;
        for i in 0..remainder {
            result += row[tail_start + i];
        }
        result
    }

    /// AVX2 max reduction for a single row.
    #[target_feature(enable = "avx2")]
    unsafe fn max_row(row: &[f32]) -> f32 {
        let n = row.len();
        let chunks = n / 8;
        let remainder = n % 8;

        let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);
        for i in 0..chunks {
            let v = _mm256_loadu_ps(row.as_ptr().add(i * 8));
            acc = _mm256_max_ps(acc, v);
        }
        let mut result = hmax_m256(acc);
        let tail_start = chunks * 8;
        for i in 0..remainder {
            result = result.max(row[tail_start + i]);
        }
        result
    }

    /// AVX2 min reduction for a single row.
    #[target_feature(enable = "avx2")]
    unsafe fn min_row(row: &[f32]) -> f32 {
        let n = row.len();
        let chunks = n / 8;
        let remainder = n % 8;

        let mut acc = _mm256_set1_ps(f32::INFINITY);
        for i in 0..chunks {
            let v = _mm256_loadu_ps(row.as_ptr().add(i * 8));
            acc = _mm256_min_ps(acc, v);
        }
        let mut result = hmin_m256(acc);
        let tail_start = chunks * 8;
        for i in 0..remainder {
            result = result.min(row[tail_start + i]);
        }
        result
    }

    /// # Safety
    /// Caller must verify AVX2 is available.
    #[target_feature(enable = "avx2")]
    pub unsafe fn sum_f32_avx2(input: &[f32], output: &mut [f32], dim_size: usize) {
        let rows = input.len() / dim_size;
        for row in 0..rows {
            let start = row * dim_size;
            output[row] = sum_row(&input[start..start + dim_size]);
        }
    }

    /// # Safety
    /// Caller must verify AVX2 is available.
    #[target_feature(enable = "avx2")]
    pub unsafe fn max_f32_avx2(input: &[f32], output: &mut [f32], dim_size: usize) {
        let rows = input.len() / dim_size;
        for row in 0..rows {
            let start = row * dim_size;
            output[row] = max_row(&input[start..start + dim_size]);
        }
    }

    /// # Safety
    /// Caller must verify AVX2 is available.
    #[target_feature(enable = "avx2")]
    pub unsafe fn min_f32_avx2(input: &[f32], output: &mut [f32], dim_size: usize) {
        let rows = input.len() / dim_size;
        for row in 0..rows {
            let start = row * dim_size;
            output[row] = min_row(&input[start..start + dim_size]);
        }
    }

    /// # Safety
    /// Caller must verify AVX2 is available.
    #[target_feature(enable = "avx2")]
    pub unsafe fn mean_f32_avx2(input: &[f32], output: &mut [f32], dim_size: usize) {
        let rows = input.len() / dim_size;
        let inv_n = 1.0 / dim_size as f32;
        for row in 0..rows {
            let start = row * dim_size;
            let s = sum_row(&input[start..start + dim_size]);
            output[row] = s * inv_n;
        }
    }

    /// Argmax — index tracking is scalar (no SIMD benefit for index compare).
    ///
    /// # Safety
    /// Caller must verify AVX2 is available.
    #[target_feature(enable = "avx2")]
    pub unsafe fn argmax_f32_avx2(input: &[f32], output: &mut [u32], dim_size: usize) {
        let rows = input.len() / dim_size;
        for row in 0..rows {
            let start = row * dim_size;
            let row_data = &input[start..start + dim_size];
            let mut best_val = f32::NEG_INFINITY;
            let mut best_idx = 0_u32;
            for (i, &v) in row_data.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = i as u32;
                }
            }
            output[row] = best_idx;
        }
    }

    /// Argmin — index tracking is scalar.
    ///
    /// # Safety
    /// Caller must verify AVX2 is available.
    #[target_feature(enable = "avx2")]
    pub unsafe fn argmin_f32_avx2(input: &[f32], output: &mut [u32], dim_size: usize) {
        let rows = input.len() / dim_size;
        for row in 0..rows {
            let start = row * dim_size;
            let row_data = &input[start..start + dim_size];
            let mut best_val = f32::INFINITY;
            let mut best_idx = 0_u32;
            for (i, &v) in row_data.iter().enumerate() {
                if v < best_val {
                    best_val = v;
                    best_idx = i as u32;
                }
            }
            output[row] = best_idx;
        }
    }
}

// ---------------------------------------------------------------------------
// Public dispatch: auto-selects best SIMD path
// ---------------------------------------------------------------------------

/// Sum reduction along the last axis. Auto-dispatches to NEON/AVX2/scalar.
///
/// `input` has shape `[..., dim_size]` flattened. `output` has length
/// `input.len() / dim_size`.
pub fn sum_f32(input: &[f32], output: &mut [f32], dim_size: usize) {
    if dim_size == 0 {
        return;
    }
    assert_eq!(
        input.len() % dim_size,
        0,
        "input length must be a multiple of dim_size"
    );
    assert_eq!(
        output.len(),
        input.len() / dim_size,
        "output length must equal number of rows"
    );

    #[cfg(target_arch = "aarch64")]
    {
        neon::sum_f32_neon(input, output, dim_size);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            unsafe { avx2::sum_f32_avx2(input, output, dim_size) };
            return;
        }
    }

    #[allow(unreachable_code)]
    sum_f32_scalar(input, output, dim_size);
}

/// Max reduction along the last axis. Auto-dispatches to NEON/AVX2/scalar.
pub fn max_f32(input: &[f32], output: &mut [f32], dim_size: usize) {
    if dim_size == 0 {
        return;
    }
    assert_eq!(
        input.len() % dim_size,
        0,
        "input length must be a multiple of dim_size"
    );
    assert_eq!(
        output.len(),
        input.len() / dim_size,
        "output length must equal number of rows"
    );

    #[cfg(target_arch = "aarch64")]
    {
        neon::max_f32_neon(input, output, dim_size);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            unsafe { avx2::max_f32_avx2(input, output, dim_size) };
            return;
        }
    }

    #[allow(unreachable_code)]
    max_f32_scalar(input, output, dim_size);
}

/// Min reduction along the last axis. Auto-dispatches to NEON/AVX2/scalar.
pub fn min_f32(input: &[f32], output: &mut [f32], dim_size: usize) {
    if dim_size == 0 {
        return;
    }
    assert_eq!(
        input.len() % dim_size,
        0,
        "input length must be a multiple of dim_size"
    );
    assert_eq!(
        output.len(),
        input.len() / dim_size,
        "output length must equal number of rows"
    );

    #[cfg(target_arch = "aarch64")]
    {
        neon::min_f32_neon(input, output, dim_size);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            unsafe { avx2::min_f32_avx2(input, output, dim_size) };
            return;
        }
    }

    #[allow(unreachable_code)]
    min_f32_scalar(input, output, dim_size);
}

/// Mean reduction along the last axis (sum / count). Auto-dispatches to NEON/AVX2/scalar.
pub fn mean_f32(input: &[f32], output: &mut [f32], dim_size: usize) {
    if dim_size == 0 {
        return;
    }
    assert_eq!(
        input.len() % dim_size,
        0,
        "input length must be a multiple of dim_size"
    );
    assert_eq!(
        output.len(),
        input.len() / dim_size,
        "output length must equal number of rows"
    );

    #[cfg(target_arch = "aarch64")]
    {
        neon::mean_f32_neon(input, output, dim_size);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            unsafe { avx2::mean_f32_avx2(input, output, dim_size) };
            return;
        }
    }

    #[allow(unreachable_code)]
    mean_f32_scalar(input, output, dim_size);
}

/// Argmax along the last axis. Auto-dispatches to NEON/AVX2/scalar.
///
/// `output` contains `u32` indices of the max element in each row.
pub fn argmax_f32(input: &[f32], output: &mut [u32], dim_size: usize) {
    if dim_size == 0 {
        return;
    }
    assert_eq!(
        input.len() % dim_size,
        0,
        "input length must be a multiple of dim_size"
    );
    assert_eq!(
        output.len(),
        input.len() / dim_size,
        "output length must equal number of rows"
    );

    #[cfg(target_arch = "aarch64")]
    {
        neon::argmax_f32_neon(input, output, dim_size);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            unsafe { avx2::argmax_f32_avx2(input, output, dim_size) };
            return;
        }
    }

    #[allow(unreachable_code)]
    argmax_f32_scalar(input, output, dim_size);
}

/// Argmin along the last axis. Auto-dispatches to NEON/AVX2/scalar.
///
/// `output` contains `u32` indices of the min element in each row.
pub fn argmin_f32(input: &[f32], output: &mut [u32], dim_size: usize) {
    if dim_size == 0 {
        return;
    }
    assert_eq!(
        input.len() % dim_size,
        0,
        "input length must be a multiple of dim_size"
    );
    assert_eq!(
        output.len(),
        input.len() / dim_size,
        "output length must equal number of rows"
    );

    #[cfg(target_arch = "aarch64")]
    {
        neon::argmin_f32_neon(input, output, dim_size);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            unsafe { avx2::argmin_f32_avx2(input, output, dim_size) };
            return;
        }
    }

    #[allow(unreachable_code)]
    argmin_f32_scalar(input, output, dim_size);
}
