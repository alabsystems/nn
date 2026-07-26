// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized general matrix-vector multiplication (GEMV).
//!
//! Provides named entry points for each SIMD tier:
//! - `gemv_f32_neon` — NEON-optimized (aarch64), 4-wide FMA
//! - `gemv_f32_avx2` — AVX2-optimized (x86_64), 8-wide FMA
//! - `gemv_f32_scalar` — pure scalar fallback
//! - `gemv_f32` — auto-dispatch to best available
//!
//! Layout: matrix is row-major `[M, K]`, vec is `[K]`, out is `[M]`.
//! Each out[i] = sum_j matrix[i*k + j] * vec[j].
//!
//! Also provides `gemv_bias_f32` which adds a bias vector after the
//! matrix-vector product: out[i] = (matrix[i,:] dot vec) + bias[i].

// ---------------------------------------------------------------------------
// Scalar fallback (always compiled, no cfg gate)
// ---------------------------------------------------------------------------

/// Pure scalar GEMV: out[i] = dot(matrix[i,:], vec).
///
/// - `matrix`: row-major `[M, K]`
/// - `vec`: vector `[K]`
/// - `m`: number of rows
/// - `k`: number of columns (length of vec)
/// - `out`: output `[M]` (overwritten)
#[inline]
pub fn gemv_f32_scalar(matrix: &[f32], vec: &[f32], m: usize, k: usize, out: &mut [f32]) {
    assert_eq!(matrix.len(), m * k, "matrix must be [M, K]");
    assert_eq!(vec.len(), k, "vec must be [K]");
    assert_eq!(out.len(), m, "out must be [M]");

    if m == 0 || k == 0 {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    for i in 0..m {
        let row = &matrix[i * k..(i + 1) * k];
        let mut acc = 0.0_f32;
        for j in 0..k {
            acc += row[j] * vec[j];
        }
        out[i] = acc;
    }
}

/// Reference implementation for differential testing.
///
/// Returns a newly-allocated `[M]` output vector. Uses scalar path.
#[must_use]
pub fn gemv_reference(matrix: &[f32], vec: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m];
    gemv_f32_scalar(matrix, vec, m, k, &mut out);
    out
}

/// Scalar GEMV with bias: out[i] = dot(matrix[i,:], vec) + bias[i].
#[inline]
pub fn gemv_bias_f32_scalar(
    matrix: &[f32],
    vec: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    out: &mut [f32],
) {
    assert_eq!(matrix.len(), m * k, "matrix must be [M, K]");
    assert_eq!(vec.len(), k, "vec must be [K]");
    assert_eq!(bias.len(), m, "bias must be [M]");
    assert_eq!(out.len(), m, "out must be [M]");

    gemv_f32_scalar(matrix, vec, m, k, out);
    for i in 0..m {
        out[i] += bias[i];
    }
}

// ---------------------------------------------------------------------------
// NEON (aarch64) — 128-bit, 4x f32 lanes
// ---------------------------------------------------------------------------

/// NEON-optimized GEMV using `vfmaq_f32` for 4-wide FMA with
/// horizontal add via `vaddvq_f32`.
#[cfg(target_arch = "aarch64")]
pub fn gemv_f32_neon(matrix: &[f32], vec: &[f32], m: usize, k: usize, out: &mut [f32]) {
    use std::arch::aarch64::*;

    assert_eq!(matrix.len(), m * k, "matrix must be [M, K]");
    assert_eq!(vec.len(), k, "vec must be [K]");
    assert_eq!(out.len(), m, "out must be [M]");

    if m == 0 || k == 0 {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let k_chunks = k / 4;
    let k_rem = k % 4;

    for i in 0..m {
        let row = &matrix[i * k..(i + 1) * k];

        // SAFETY: NEON always available on aarch64. All pointer arithmetic
        // is bounded by k_chunks (which ensures offset + 4 <= k) and
        // k_rem (which ensures tail_start + r < k).
        unsafe {
            let mut acc = vdupq_n_f32(0.0);
            for c in 0..k_chunks {
                let offset = c * 4;
                let va = vld1q_f32(row.as_ptr().add(offset));
                let vv = vld1q_f32(vec.as_ptr().add(offset));
                acc = vfmaq_f32(acc, va, vv);
            }
            // Horizontal sum of 4 lanes.
            let mut result = vaddvq_f32(acc);

            // Scalar tail.
            let tail_start = k_chunks * 4;
            for r in 0..k_rem {
                result += row[tail_start + r] * vec[tail_start + r];
            }
            out[i] = result;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn gemv_f32_neon(matrix: &[f32], vec: &[f32], m: usize, k: usize, out: &mut [f32]) {
    // Fallback: delegate to scalar on non-aarch64 platforms.
    gemv_f32_scalar(matrix, vec, m, k, out);
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64) — 256-bit, 8x f32 lanes
// ---------------------------------------------------------------------------

/// AVX2-optimized GEMV using `_mm256_fmadd_ps` for 8-wide FMA.
#[cfg(target_arch = "x86_64")]
pub fn gemv_f32_avx2(matrix: &[f32], vec: &[f32], m: usize, k: usize, out: &mut [f32]) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: AVX2+FMA detected above.
        unsafe { gemv_f32_avx2_inner(matrix, vec, m, k, out) };
    } else {
        gemv_f32_scalar(matrix, vec, m, k, out);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn gemv_f32_avx2_inner(matrix: &[f32], vec: &[f32], m: usize, k: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;

    assert_eq!(matrix.len(), m * k, "matrix must be [M, K]");
    assert_eq!(vec.len(), k, "vec must be [K]");
    assert_eq!(out.len(), m, "out must be [M]");

    if m == 0 || k == 0 {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let k_chunks = k / 8;
    let k_rem = k % 8;

    for i in 0..m {
        let row = &matrix[i * k..(i + 1) * k];

        let mut acc = _mm256_setzero_ps();
        for c in 0..k_chunks {
            let offset = c * 8;
            // SAFETY: offset + 8 <= k (from k_chunks bound). Both
            // row and vec have length >= k.
            let va = _mm256_loadu_ps(row.as_ptr().add(offset));
            let vv = _mm256_loadu_ps(vec.as_ptr().add(offset));
            acc = _mm256_fmadd_ps(va, vv, acc);
        }

        // Horizontal sum of __m256 (8 lanes -> 1 scalar).
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result_sse = _mm_add_ss(sums, shuf2);
        let mut result = _mm_cvtss_f32(result_sse);

        // Scalar tail.
        let tail_start = k_chunks * 8;
        for r in 0..k_rem {
            result += row[tail_start + r] * vec[tail_start + r];
        }
        out[i] = result;
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn gemv_f32_avx2(matrix: &[f32], vec: &[f32], m: usize, k: usize, out: &mut [f32]) {
    // Fallback: delegate to scalar on non-x86_64 platforms.
    gemv_f32_scalar(matrix, vec, m, k, out);
}

// ---------------------------------------------------------------------------
// GEMV with bias: out = matrix * vec + bias
// ---------------------------------------------------------------------------

/// GEMV with bias add: out[i] = dot(matrix[i,:], vec) + bias[i].
///
/// Auto-dispatches to NEON/AVX2/scalar for the matrix-vector product,
/// then adds the bias vector.
///
/// - `matrix`: row-major `[M, K]`
/// - `vec`: vector `[K]`
/// - `bias`: vector `[M]`
/// - `m`: number of rows
/// - `k`: number of columns (length of vec)
/// - `out`: output `[M]` (overwritten)
pub fn gemv_bias_f32(
    matrix: &[f32],
    vec: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    out: &mut [f32],
) {
    assert_eq!(bias.len(), m, "bias must be [M]");
    assert_eq!(out.len(), m, "out must be [M]");

    // Compute matrix * vec into out.
    gemv_f32(matrix, vec, m, k, out);

    // Add bias.
    for i in 0..m {
        out[i] += bias[i];
    }
}

// ---------------------------------------------------------------------------
// Public dispatch: auto-selects best SIMD path
// ---------------------------------------------------------------------------

/// General GEMV: out = matrix * vec.
///
/// Auto-dispatches to NEON (aarch64), AVX2 (x86_64), or scalar fallback.
///
/// - `matrix`: row-major `[M, K]`
/// - `vec`: vector `[K]`
/// - `m`: number of rows
/// - `k`: number of columns
/// - `out`: output `[M]` (overwritten)
///
/// Each out[i] = sum_j matrix[i*k + j] * vec[j].
pub fn gemv_f32(matrix: &[f32], vec: &[f32], m: usize, k: usize, out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        gemv_f32_neon(matrix, vec, m, k, out);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            gemv_f32_avx2(matrix, vec, m, k, out);
            return;
        }
    }

    #[allow(unreachable_code)]
    gemv_f32_scalar(matrix, vec, m, k, out);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "simd_gemv_tests.rs"]
mod simd_gemv_tests;
