// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized matrix multiplication (SGEMM) with explicit entry points.
//!
//! Provides named entry points for each SIMD tier:
//! - `matmul_f32_neon` — NEON-optimized (aarch64), 4-wide FMA
//! - `matmul_f32_avx2` — AVX2-optimized (x86_64), 8-wide FMA
//! - `matmul_f32_scalar` — pure scalar fallback
//! - `matmul_f32` — auto-dispatch to best available
//!
//! Layout: A is row-major `[M, K]`, B is row-major `[K, N]`, out is `[M, N]`.
//! All functions write `out[i*n + j] = sum_p A[i*k + p] * B[p*n + j]`.
//!
//! The inner loop processes 4 rows at a time (NEON) or 4 rows at a time
//! (AVX2) for register reuse, with scalar tails for remainders.

// ---------------------------------------------------------------------------
// Scalar fallback (always compiled, no cfg gate)
// ---------------------------------------------------------------------------

/// Pure scalar matmul: C = A * B.
///
/// - `a`: row-major `[M, K]`
/// - `b`: row-major `[K, N]`
/// - `out`: row-major `[M, N]` (overwritten)
///
/// Uses a simple triple-loop with accumulator per output element.
#[inline]
pub fn matmul_f32_scalar(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    assert_eq!(a.len(), m * k, "a must be [M, K]");
    assert_eq!(b.len(), k * n, "b must be [K, N]");
    assert_eq!(out.len(), m * n, "out must be [M, N]");

    if m == 0 || k == 0 || n == 0 {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
}

/// Reference implementation for differential testing.
///
/// Returns a newly-allocated `[M, N]` output vector. Uses scalar path.
#[must_use]
pub fn matmul_reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    matmul_f32_scalar(a, b, m, k, n, &mut out);
    out
}

// ---------------------------------------------------------------------------
// NEON (aarch64) — 128-bit, 4x f32 lanes
// ---------------------------------------------------------------------------

/// NEON-optimized matmul processing 4 rows at a time.
///
/// Each group of 4 rows uses FMA to accumulate 4 output elements per
/// k-iteration, giving good register reuse.
#[cfg(target_arch = "aarch64")]
pub fn matmul_f32_neon(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    use std::arch::aarch64::*;

    assert_eq!(a.len(), m * k, "a must be [M, K]");
    assert_eq!(b.len(), k * n, "b must be [K, N]");
    assert_eq!(out.len(), m * n, "out must be [M, N]");

    if m == 0 || k == 0 || n == 0 {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let m_chunks = m / 4;
    let m_rem = m % 4;

    // Process 4 rows at a time.
    for mc in 0..m_chunks {
        let row_base = mc * 4;
        for j in 0..n {
            // SAFETY: NEON always available on aarch64. Accumulator lanes
            // are in-register; all slice accesses are bounds-checked via
            // the asserts above and loop bounds.
            unsafe {
                let mut acc = vdupq_n_f32(0.0);
                let k_chunks = k / 4;
                let k_rem = k % 4;

                for kc in 0..k_chunks {
                    let kk = kc * 4;
                    // Gather 4 B elements: B[kk..kk+4, j] (strided by n).
                    let b_vals = vld1q_f32(
                        [
                            b[kk * n + j],
                            b[(kk + 1) * n + j],
                            b[(kk + 2) * n + j],
                            b[(kk + 3) * n + j],
                        ]
                        .as_ptr(),
                    );

                    // For each of the 4 rows, accumulate a[row, kk..kk+4] dot b_vals.
                    // We use lane-wise FMA: each lane handles one row's contribution
                    // for 4 k-values at once.
                    let a0 = vld1q_f32(a.as_ptr().add(row_base * k + kk));
                    let a1 = vld1q_f32(a.as_ptr().add((row_base + 1) * k + kk));
                    let a2 = vld1q_f32(a.as_ptr().add((row_base + 2) * k + kk));
                    let a3 = vld1q_f32(a.as_ptr().add((row_base + 3) * k + kk));

                    // Dot product of a_row[kk..kk+4] with b_vals for each row.
                    // We accumulate into acc lanes: acc[0]+=dot(a0,b), etc.
                    // Using vmulq + vaddvq per row-chunk, accumulated into scalar.
                    let d0 = vmulq_f32(a0, b_vals);
                    let d1 = vmulq_f32(a1, b_vals);
                    let d2 = vmulq_f32(a2, b_vals);
                    let d3 = vmulq_f32(a3, b_vals);

                    // Horizontal sum each product into the accumulator lanes.
                    let sums = vld1q_f32(
                        [
                            vaddvq_f32(d0),
                            vaddvq_f32(d1),
                            vaddvq_f32(d2),
                            vaddvq_f32(d3),
                        ]
                        .as_ptr(),
                    );
                    acc = vaddq_f32(acc, sums);
                }

                // Scalar tail for remaining k elements.
                let k_tail = k_chunks * 4;
                let mut tail = [0.0f32; 4];
                vst1q_f32(tail.as_mut_ptr(), acc);
                for kr in 0..k_rem {
                    let kk = k_tail + kr;
                    let bv = b[kk * n + j];
                    tail[0] += a[(row_base) * k + kk] * bv;
                    tail[1] += a[(row_base + 1) * k + kk] * bv;
                    tail[2] += a[(row_base + 2) * k + kk] * bv;
                    tail[3] += a[(row_base + 3) * k + kk] * bv;
                }

                out[row_base * n + j] = tail[0];
                out[(row_base + 1) * n + j] = tail[1];
                out[(row_base + 2) * n + j] = tail[2];
                out[(row_base + 3) * n + j] = tail[3];
            }
        }
    }

    // Remainder rows (< 4).
    let row_tail = m_chunks * 4;
    for i in row_tail..row_tail + m_rem {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn matmul_f32_neon(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    // Fallback: delegate to scalar on non-aarch64 platforms.
    matmul_f32_scalar(a, b, m, k, n, out);
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64) — 256-bit, 8x f32 lanes
// ---------------------------------------------------------------------------

/// AVX2-optimized matmul processing 4 rows at a time with 8-wide FMA.
#[cfg(target_arch = "x86_64")]
pub fn matmul_f32_avx2(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: AVX2+FMA detected above.
        unsafe { matmul_f32_avx2_inner(a, b, m, k, n, out) };
    } else {
        matmul_f32_scalar(a, b, m, k, n, out);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn matmul_f32_avx2_inner(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;

    assert_eq!(a.len(), m * k, "a must be [M, K]");
    assert_eq!(b.len(), k * n, "b must be [K, N]");
    assert_eq!(out.len(), m * n, "out must be [M, N]");

    if m == 0 || k == 0 || n == 0 {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    // Process output columns in chunks of 8 using AVX2 for contiguous stores.
    let n_chunks = n / 8;
    let n_rem = n % 8;

    // Process 4 rows at a time for register reuse.
    let m_chunks = m / 4;
    let m_rem = m % 4;

    for mc in 0..m_chunks {
        let row_base = mc * 4;

        // SIMD columns: 8-wide output columns.
        for nc in 0..n_chunks {
            let col_base = nc * 8;

            let mut c0 = _mm256_setzero_ps();
            let mut c1 = _mm256_setzero_ps();
            let mut c2 = _mm256_setzero_ps();
            let mut c3 = _mm256_setzero_ps();

            for p in 0..k {
                // SAFETY: col_base + 8 <= n (from loop bound), p*n+col_base in bounds.
                let bv = _mm256_loadu_ps(b.as_ptr().add(p * n + col_base));

                let a0 = _mm256_set1_ps(*a.get_unchecked(row_base * k + p));
                c0 = _mm256_fmadd_ps(a0, bv, c0);

                let a1 = _mm256_set1_ps(*a.get_unchecked((row_base + 1) * k + p));
                c1 = _mm256_fmadd_ps(a1, bv, c1);

                let a2 = _mm256_set1_ps(*a.get_unchecked((row_base + 2) * k + p));
                c2 = _mm256_fmadd_ps(a2, bv, c2);

                let a3 = _mm256_set1_ps(*a.get_unchecked((row_base + 3) * k + p));
                c3 = _mm256_fmadd_ps(a3, bv, c3);
            }

            // SAFETY: row_base + 3 < m, col_base + 8 <= n.
            _mm256_storeu_ps(out.as_mut_ptr().add(row_base * n + col_base), c0);
            _mm256_storeu_ps(out.as_mut_ptr().add((row_base + 1) * n + col_base), c1);
            _mm256_storeu_ps(out.as_mut_ptr().add((row_base + 2) * n + col_base), c2);
            _mm256_storeu_ps(out.as_mut_ptr().add((row_base + 3) * n + col_base), c3);
        }

        // Remainder columns (< 8): scalar.
        let col_tail = n_chunks * 8;
        for j in col_tail..col_tail + n_rem {
            for r in 0..4_usize {
                let row = row_base + r;
                let mut acc = 0.0_f32;
                for p in 0..k {
                    acc += a[row * k + p] * b[p * n + j];
                }
                out[row * n + j] = acc;
            }
        }
    }

    // Remainder rows (< 4): scalar.
    let row_tail = m_chunks * 4;
    for i in row_tail..row_tail + m_rem {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn matmul_f32_avx2(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    // Fallback: delegate to scalar on non-x86_64 platforms.
    matmul_f32_scalar(a, b, m, k, n, out);
}

// ---------------------------------------------------------------------------
// Public dispatch: auto-selects best SIMD path
// ---------------------------------------------------------------------------

/// General SGEMM: out = A * B.
///
/// Auto-dispatches to NEON (aarch64), AVX2 (x86_64), or scalar fallback.
///
/// - `a`: row-major `[M, K]`
/// - `b`: row-major `[K, N]`
/// - `out`: row-major `[M, N]` (overwritten)
///
/// Uses FMA instructions on supported hardware for fused multiply-add
/// in the inner loop, processing 4 rows at a time for register reuse.
pub fn matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        matmul_f32_neon(a, b, m, k, n, out);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            matmul_f32_avx2(a, b, m, k, n, out);
            return;
        }
    }

    #[allow(unreachable_code)]
    matmul_f32_scalar(a, b, m, k, n, out);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "simd_matmul_tests.rs"]
mod simd_matmul_tests;
