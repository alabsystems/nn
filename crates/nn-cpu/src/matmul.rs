// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cache-friendly tiled matrix multiplication with SIMD micro-kernels.
//!
//! Implements [M, K] x [K, N] -> [M, N] with:
//! - L1-friendly tile sizes (TILE = 64 for 64 KB L1 on Apple M-series)
//! - NEON 4x4 micro-kernel using `vfmaq_laneq_f32` for the inner product
//! - AVX2 4x8 micro-kernel using `_mm256_fmadd_ps` with broadcasts
//! - Panel packing of B tiles into column-contiguous layout for SIMD loads
//! - Scalar fallback for remainder rows/columns and unsupported architectures

/// Tile size for cache-friendly blocking.
///
/// Chosen so that three tiles (A, B, C) of TILE x TILE x 4 bytes each fit
/// in L1 cache: 3 * 64 * 64 * 4 = 48 KB < 64 KB (Apple M-series L1).
const TILE: usize = 64;

/// NEON micro-kernel row count.
#[cfg(target_arch = "aarch64")]
const MR_NEON: usize = 4;

/// NEON micro-kernel column count.
#[cfg(target_arch = "aarch64")]
const NR_NEON: usize = 4;

/// AVX2 micro-kernel row count.
#[cfg(target_arch = "x86_64")]
const MR_AVX2: usize = 4;

/// AVX2 micro-kernel column count (8 f32 lanes per ymm register).
#[cfg(target_arch = "x86_64")]
const NR_AVX2: usize = 8;

/// Tiled matmul: `c[m][n] += a[m][k] * b[k][n]` over `k`.
///
/// `a` is row-major [M, K], `b` is row-major [K, N], `c` is row-major [M, N].
/// `c` must be pre-zeroed by the caller.
///
/// Uses tiling over (m, n, k) with SIMD micro-kernels within each tile.
pub fn matmul_tiled(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), m * k, "a must be [M, K]");
    assert_eq!(b.len(), k * n, "b must be [K, N]");
    assert_eq!(c.len(), m * n, "c must be [M, N]");

    // Degenerate cases.
    if m == 0 || k == 0 || n == 0 {
        return;
    }

    // Dispatch to platform-specific tiled implementation.
    #[cfg(target_arch = "aarch64")]
    {
        matmul_tiled_neon(a, b, c, m, k, n);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA detected above.
            unsafe { matmul_tiled_avx2(a, b, c, m, k, n) };
            return;
        }
    }

    #[allow(unreachable_code)]
    matmul_tiled_scalar(a, b, c, m, k, n);
}

/// Convenience wrapper: allocates output and runs tiled matmul.
///
/// `a` is [M, K], `b` is [K, N]. Returns [M, N].
#[must_use]
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    matmul_tiled(a, b, &mut c, m, k, n);
    c
}

// ---------------------------------------------------------------------------
// Scalar tiled implementation (fallback for all architectures)
// ---------------------------------------------------------------------------

/// Scalar tiled matmul: no SIMD, but still cache-friendly via tiling.
fn matmul_tiled_scalar(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    let mut kt = 0;
    while kt < k {
        let k_end = (kt + TILE).min(k);
        let mut mt = 0;
        while mt < m {
            let m_end = (mt + TILE).min(m);
            let mut nt = 0;
            while nt < n {
                let n_end = (nt + TILE).min(n);
                for i in mt..m_end {
                    for j in nt..n_end {
                        let mut acc = 0.0_f32;
                        for kk in kt..k_end {
                            acc += a[i * k + kk] * b[kk * n + j];
                        }
                        c[i * n + j] += acc;
                    }
                }
                nt += TILE;
            }
            mt += TILE;
        }
        kt += TILE;
    }
}

// ---------------------------------------------------------------------------
// NEON tiled implementation (aarch64)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
fn matmul_tiled_neon(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    // Scratch buffer for packing B tile columns into contiguous layout.
    let mut packed_b = vec![0.0f32; TILE * TILE];

    let mut kt = 0;
    while kt < k {
        let k_end = (kt + TILE).min(k);
        let k_len = k_end - kt;

        let mut nt = 0;
        while nt < n {
            let n_end = (nt + TILE).min(n);
            let n_len = n_end - nt;

            // Pack B[kt..k_end, nt..n_end] into column-contiguous panels.
            pack_b_neon(b, &mut packed_b, kt, k_end, nt, n_end, n);

            let mut mt = 0;
            while mt < m {
                let m_end = (mt + TILE).min(m);

                // Process MR_NEON rows at a time with the NEON micro-kernel.
                let mut i = mt;
                while i + MR_NEON <= m_end {
                    let mut j = 0;
                    while j + NR_NEON <= n_len {
                        // SAFETY: NEON is always available on aarch64.
                        unsafe {
                            microkernel_4x4_neon(
                                a,
                                c,
                                &packed_b,
                                i,
                                nt + j,
                                kt,
                                k_len,
                                k,
                                n,
                                j * k_len,
                            );
                        }
                        j += NR_NEON;
                    }
                    // Remainder columns (< NR_NEON).
                    let full_j = (n_len / NR_NEON) * NR_NEON;
                    for jj in (nt + full_j)..n_end {
                        for ii in i..i + MR_NEON {
                            let mut acc = 0.0_f32;
                            for kk in kt..k_end {
                                acc += a[ii * k + kk] * b[kk * n + jj];
                            }
                            c[ii * n + jj] += acc;
                        }
                    }
                    i += MR_NEON;
                }
                // Remainder rows (< MR_NEON).
                for ii in i..m_end {
                    for jj in nt..n_end {
                        let mut acc = 0.0_f32;
                        for kk in kt..k_end {
                            acc += a[ii * k + kk] * b[kk * n + jj];
                        }
                        c[ii * n + jj] += acc;
                    }
                }
                mt += TILE;
            }
            nt += TILE;
        }
        kt += TILE;
    }
}

/// Pack B[kt..k_end, nt..n_end] into panels of NR_NEON contiguous columns.
#[cfg(target_arch = "aarch64")]
fn pack_b_neon(
    b: &[f32],
    packed: &mut [f32],
    kt: usize,
    k_end: usize,
    nt: usize,
    n_end: usize,
    n_stride: usize,
) {
    let k_len = k_end - kt;
    let n_len = n_end - nt;
    let nr = NR_NEON;

    let mut j = 0;
    while j + nr <= n_len {
        let panel_offset = j * k_len;
        for kk in 0..k_len {
            let b_row = (kt + kk) * n_stride + nt + j;
            let p_row = panel_offset + kk * nr;
            packed[p_row] = b[b_row];
            packed[p_row + 1] = b[b_row + 1];
            packed[p_row + 2] = b[b_row + 2];
            packed[p_row + 3] = b[b_row + 3];
        }
        j += nr;
    }
}

/// NEON 4x4 micro-kernel: accumulates
/// C[i..i+4, col..col+4] += A[i..i+4, kt..kt+k_len] * packed_B.
///
/// Uses 4 float32x4 accumulators (one per output row, 4 output columns).
/// Inner loop uses `vfmaq_laneq_f32` to broadcast each A element across
/// 4 B columns.
///
/// # Safety
/// Requires aarch64 with NEON. All indices must be in bounds.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn microkernel_4x4_neon(
    a: &[f32],
    c: &mut [f32],
    packed_b: &[f32],
    row: usize,
    col: usize,
    kt: usize,
    k_len: usize,
    k_stride: usize,
    n_stride: usize,
    panel_offset: usize,
) { unsafe {
    use std::arch::aarch64::*;

    let mut c0 = vdupq_n_f32(0.0);
    let mut c1 = vdupq_n_f32(0.0);
    let mut c2 = vdupq_n_f32(0.0);
    let mut c3 = vdupq_n_f32(0.0);

    // Process 4 k-elements at a time for better ILP.
    let k_chunks = k_len / 4;
    let k_rem = k_len % 4;

    for kc in 0..k_chunks {
        let kk = kc * 4;

        let b0 = vld1q_f32(packed_b.as_ptr().add(panel_offset + kk * 4));
        let b1 = vld1q_f32(packed_b.as_ptr().add(panel_offset + (kk + 1) * 4));
        let b2 = vld1q_f32(packed_b.as_ptr().add(panel_offset + (kk + 2) * 4));
        let b3 = vld1q_f32(packed_b.as_ptr().add(panel_offset + (kk + 3) * 4));

        // Row 0
        let a0v = vld1q_f32(a.as_ptr().add(row * k_stride + kt + kk));
        c0 = vfmaq_laneq_f32::<0>(c0, b0, a0v);
        c0 = vfmaq_laneq_f32::<1>(c0, b1, a0v);
        c0 = vfmaq_laneq_f32::<2>(c0, b2, a0v);
        c0 = vfmaq_laneq_f32::<3>(c0, b3, a0v);

        // Row 1
        let a1v = vld1q_f32(a.as_ptr().add((row + 1) * k_stride + kt + kk));
        c1 = vfmaq_laneq_f32::<0>(c1, b0, a1v);
        c1 = vfmaq_laneq_f32::<1>(c1, b1, a1v);
        c1 = vfmaq_laneq_f32::<2>(c1, b2, a1v);
        c1 = vfmaq_laneq_f32::<3>(c1, b3, a1v);

        // Row 2
        let a2v = vld1q_f32(a.as_ptr().add((row + 2) * k_stride + kt + kk));
        c2 = vfmaq_laneq_f32::<0>(c2, b0, a2v);
        c2 = vfmaq_laneq_f32::<1>(c2, b1, a2v);
        c2 = vfmaq_laneq_f32::<2>(c2, b2, a2v);
        c2 = vfmaq_laneq_f32::<3>(c2, b3, a2v);

        // Row 3
        let a3v = vld1q_f32(a.as_ptr().add((row + 3) * k_stride + kt + kk));
        c3 = vfmaq_laneq_f32::<0>(c3, b0, a3v);
        c3 = vfmaq_laneq_f32::<1>(c3, b1, a3v);
        c3 = vfmaq_laneq_f32::<2>(c3, b2, a3v);
        c3 = vfmaq_laneq_f32::<3>(c3, b3, a3v);
    }

    // Remaining k elements (< 4).
    let k_tail_start = k_chunks * 4;
    for kr in 0..k_rem {
        let kk = k_tail_start + kr;
        let bv = vld1q_f32(packed_b.as_ptr().add(panel_offset + kk * 4));

        let a0 = *a.get_unchecked(row * k_stride + kt + kk);
        c0 = vfmaq_n_f32(c0, bv, a0);

        let a1 = *a.get_unchecked((row + 1) * k_stride + kt + kk);
        c1 = vfmaq_n_f32(c1, bv, a1);

        let a2 = *a.get_unchecked((row + 2) * k_stride + kt + kk);
        c2 = vfmaq_n_f32(c2, bv, a2);

        let a3 = *a.get_unchecked((row + 3) * k_stride + kt + kk);
        c3 = vfmaq_n_f32(c3, bv, a3);
    }

    // Store: load existing C, add accumulators, store back.
    let c0_ptr = c.as_mut_ptr().add(row * n_stride + col);
    let c1_ptr = c.as_mut_ptr().add((row + 1) * n_stride + col);
    let c2_ptr = c.as_mut_ptr().add((row + 2) * n_stride + col);
    let c3_ptr = c.as_mut_ptr().add((row + 3) * n_stride + col);

    vst1q_f32(c0_ptr, vaddq_f32(vld1q_f32(c0_ptr), c0));
    vst1q_f32(c1_ptr, vaddq_f32(vld1q_f32(c1_ptr), c1));
    vst1q_f32(c2_ptr, vaddq_f32(vld1q_f32(c2_ptr), c2));
    vst1q_f32(c3_ptr, vaddq_f32(vld1q_f32(c3_ptr), c3));
}}

// ---------------------------------------------------------------------------
// AVX2 tiled implementation (x86_64)
// ---------------------------------------------------------------------------

/// AVX2 tiled matmul with 4x8 micro-kernel.
///
/// # Safety
/// Caller must verify AVX2 and FMA are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn matmul_tiled_avx2(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    let mut packed_b = vec![0.0f32; TILE * TILE];

    let mut kt = 0;
    while kt < k {
        let k_end = (kt + TILE).min(k);
        let k_len = k_end - kt;

        let mut nt = 0;
        while nt < n {
            let n_end = (nt + TILE).min(n);
            let n_len = n_end - nt;

            pack_b_avx2(b, &mut packed_b, kt, k_end, nt, n_end, n);

            let mut mt = 0;
            while mt < m {
                let m_end = (mt + TILE).min(m);

                let mut i = mt;
                while i + MR_AVX2 <= m_end {
                    let mut j = 0;
                    while j + NR_AVX2 <= n_len {
                        microkernel_4x8_avx2(
                            a,
                            c,
                            &packed_b,
                            i,
                            nt + j,
                            kt,
                            k_len,
                            k,
                            n,
                            j * k_len,
                        );
                        j += NR_AVX2;
                    }
                    let full_j = (n_len / NR_AVX2) * NR_AVX2;
                    for jj in (nt + full_j)..n_end {
                        for ii in i..i + MR_AVX2 {
                            let mut acc = 0.0_f32;
                            for kk in kt..k_end {
                                acc += a[ii * k + kk] * b[kk * n + jj];
                            }
                            c[ii * n + jj] += acc;
                        }
                    }
                    i += MR_AVX2;
                }
                for ii in i..m_end {
                    for jj in nt..n_end {
                        let mut acc = 0.0_f32;
                        for kk in kt..k_end {
                            acc += a[ii * k + kk] * b[kk * n + jj];
                        }
                        c[ii * n + jj] += acc;
                    }
                }
                mt += TILE;
            }
            nt += TILE;
        }
        kt += TILE;
    }
}

/// Pack B[kt..k_end, nt..n_end] into panels of NR_AVX2=8 contiguous columns.
#[cfg(target_arch = "x86_64")]
fn pack_b_avx2(
    b: &[f32],
    packed: &mut [f32],
    kt: usize,
    k_end: usize,
    nt: usize,
    n_end: usize,
    n_stride: usize,
) {
    let k_len = k_end - kt;
    let n_len = n_end - nt;
    let nr = NR_AVX2;

    let mut j = 0;
    while j + nr <= n_len {
        let panel_offset = j * k_len;
        for kk in 0..k_len {
            let b_row = (kt + kk) * n_stride + nt + j;
            let p_row = panel_offset + kk * nr;
            packed[p_row..p_row + 8].copy_from_slice(&b[b_row..b_row + 8]);
        }
        j += nr;
    }
}

/// AVX2 4x8 micro-kernel: accumulates C[i..i+4, col..col+8].
///
/// Uses 4 __m256 accumulators (one per output row, 8 columns each).
/// Inner loop broadcasts each A element and FMAs with packed B row.
///
/// # Safety
/// Requires AVX2 and FMA. All indices must be in bounds.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn microkernel_4x8_avx2(
    a: &[f32],
    c: &mut [f32],
    packed_b: &[f32],
    row: usize,
    col: usize,
    kt: usize,
    k_len: usize,
    k_stride: usize,
    n_stride: usize,
    panel_offset: usize,
) {
    use std::arch::x86_64::*;

    let mut c0 = _mm256_setzero_ps();
    let mut c1 = _mm256_setzero_ps();
    let mut c2 = _mm256_setzero_ps();
    let mut c3 = _mm256_setzero_ps();

    for kk in 0..k_len {
        let bv = _mm256_loadu_ps(packed_b.as_ptr().add(panel_offset + kk * 8));

        let a0 = _mm256_set1_ps(*a.get_unchecked(row * k_stride + kt + kk));
        c0 = _mm256_fmadd_ps(a0, bv, c0);

        let a1 = _mm256_set1_ps(*a.get_unchecked((row + 1) * k_stride + kt + kk));
        c1 = _mm256_fmadd_ps(a1, bv, c1);

        let a2 = _mm256_set1_ps(*a.get_unchecked((row + 2) * k_stride + kt + kk));
        c2 = _mm256_fmadd_ps(a2, bv, c2);

        let a3 = _mm256_set1_ps(*a.get_unchecked((row + 3) * k_stride + kt + kk));
        c3 = _mm256_fmadd_ps(a3, bv, c3);
    }

    let c0_ptr = c.as_mut_ptr().add(row * n_stride + col);
    _mm256_storeu_ps(c0_ptr, _mm256_add_ps(_mm256_loadu_ps(c0_ptr), c0));

    let c1_ptr = c.as_mut_ptr().add((row + 1) * n_stride + col);
    _mm256_storeu_ps(c1_ptr, _mm256_add_ps(_mm256_loadu_ps(c1_ptr), c1));

    let c2_ptr = c.as_mut_ptr().add((row + 2) * n_stride + col);
    _mm256_storeu_ps(c2_ptr, _mm256_add_ps(_mm256_loadu_ps(c2_ptr), c2));

    let c3_ptr = c.as_mut_ptr().add((row + 3) * n_stride + col);
    _mm256_storeu_ps(c3_ptr, _mm256_add_ps(_mm256_loadu_ps(c3_ptr), c3));
}

// ---------------------------------------------------------------------------
// SIMD-optimized contiguous dot for transpose case
// ---------------------------------------------------------------------------

/// SIMD dot product of two contiguous f32 slices.
///
/// Uses the platform-specific reduction::dot() when the slice lengths match.
pub fn dot_contiguous(a: &[f32], b: &[f32]) -> f32 {
    crate::reduction::dot(a, b)
}

/// Matmul for pre-transposed B: A[M,K] * B^T[N,K] -> C[M,N].
///
/// When B is stored transposed, columns of B become contiguous rows,
/// enabling full SIMD dot products in the inner loop.
pub fn matmul_with_transposed_b(
    a: &[f32],
    b_t: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    assert_eq!(a.len(), m * k, "a must be [M, K]");
    assert_eq!(b_t.len(), n * k, "b_t must be [N, K] (transposed B)");
    assert_eq!(c.len(), m * n, "c must be [M, N]");

    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let b_row = &b_t[j * k..(j + 1) * k];
            c[i * n + j] = crate::reduction::dot(a_row, b_row);
        }
    }
}
