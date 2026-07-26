// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized 2D matrix transpose with explicit scalar/NEON/AVX2 entry points.
//!
//! This module provides:
//! - `transpose_2d` — auto-dispatch to best available (out-of-place)
//! - `transpose_reference` — pure scalar reference (returns new Vec)
//!
//! Uses block transpose for cache efficiency:
//! - NEON: 4x4 blocks with lane swizzling
//! - AVX2: 8x8 blocks with unpack + permute
//! - Scalar fallback for edges and non-SIMD platforms

// ---------------------------------------------------------------------------
// Scalar reference (returns new Vec)
// ---------------------------------------------------------------------------

/// Pure scalar transpose reference implementation returning a new Vec.
///
/// Transposes a `rows x cols` row-major matrix into a `cols x rows` row-major matrix.
pub fn transpose_reference(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(
        input.len(),
        rows * cols,
        "input length must equal rows * cols"
    );
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = input[r * cols + c];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Scalar fallback (out-of-place)
// ---------------------------------------------------------------------------

/// Scalar out-of-place 2D transpose.
fn transpose_scalar(input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        for c in 0..cols {
            output[c * rows + r] = input[r * cols + c];
        }
    }
}

// ---------------------------------------------------------------------------
// NEON (aarch64) — 4x4 block transpose
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
fn transpose_neon(input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    use std::arch::aarch64::*;

    let block_rows = rows / 4;
    let block_cols = cols / 4;

    // Process 4x4 blocks with NEON
    for br in 0..block_rows {
        for bc in 0..block_cols {
            let r0 = br * 4;
            let c0 = bc * 4;

            // SAFETY: aarch64 NEON always available. Pointer offsets within bounds
            // because br < block_rows = rows/4 and bc < block_cols = cols/4,
            // so r0+3 < rows and c0+3 < cols.
            unsafe {
                // Load 4 rows of 4 elements each
                let row0 = vld1q_f32(input.as_ptr().add(r0 * cols + c0));
                let row1 = vld1q_f32(input.as_ptr().add((r0 + 1) * cols + c0));
                let row2 = vld1q_f32(input.as_ptr().add((r0 + 2) * cols + c0));
                let row3 = vld1q_f32(input.as_ptr().add((r0 + 3) * cols + c0));

                // Transpose using zip (interleave) operations:
                // zip1/zip2 on pairs, then zip1/zip2 on results
                let t01_lo = vzip1q_f32(row0, row2); // [r0c0, r2c0, r0c1, r2c1]
                let t01_hi = vzip2q_f32(row0, row2); // [r0c2, r2c2, r0c3, r2c3]
                let t23_lo = vzip1q_f32(row1, row3); // [r1c0, r3c0, r1c1, r3c1]
                let t23_hi = vzip2q_f32(row1, row3); // [r1c2, r3c2, r1c3, r3c3]

                let col0 = vzip1q_f32(t01_lo, t23_lo); // [r0c0, r1c0, r2c0, r3c0]
                let col1 = vzip2q_f32(t01_lo, t23_lo); // [r0c1, r1c1, r2c1, r3c1]
                let col2 = vzip1q_f32(t01_hi, t23_hi); // [r0c2, r1c2, r2c2, r3c2]
                let col3 = vzip2q_f32(t01_hi, t23_hi); // [r0c3, r1c3, r2c3, r3c3]

                // Store transposed columns as rows in output
                vst1q_f32(output.as_mut_ptr().add(c0 * rows + r0), col0);
                vst1q_f32(output.as_mut_ptr().add((c0 + 1) * rows + r0), col1);
                vst1q_f32(output.as_mut_ptr().add((c0 + 2) * rows + r0), col2);
                vst1q_f32(output.as_mut_ptr().add((c0 + 3) * rows + r0), col3);
            }
        }
    }

    // Handle remaining rows (bottom edge)
    let row_tail = block_rows * 4;
    for r in row_tail..rows {
        for c in 0..cols {
            output[c * rows + r] = input[r * cols + c];
        }
    }

    // Handle remaining cols (right edge, but only for block rows to avoid double-counting)
    let col_tail = block_cols * 4;
    for r in 0..row_tail {
        for c in col_tail..cols {
            output[c * rows + r] = input[r * cols + c];
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64) — 8x8 block transpose
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose_avx2_inner(input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    use std::arch::x86_64::*;

    let block_rows = rows / 8;
    let block_cols = cols / 8;

    // Process 8x8 blocks
    for br in 0..block_rows {
        for bc in 0..block_cols {
            let r0 = br * 8;
            let c0 = bc * 8;

            // SAFETY: All pointer offsets bounded by block dimensions.
            // Load 8 rows of 8 floats each.
            let mut r = [_mm256_setzero_ps(); 8];
            for k in 0..8 {
                r[k] = _mm256_loadu_ps(input.as_ptr().add((r0 + k) * cols + c0));
            }

            // 8x8 transpose using unpack + permute:
            // Step 1: interleave pairs of rows
            let t0 = _mm256_unpacklo_ps(r[0], r[1]);
            let t1 = _mm256_unpackhi_ps(r[0], r[1]);
            let t2 = _mm256_unpacklo_ps(r[2], r[3]);
            let t3 = _mm256_unpackhi_ps(r[2], r[3]);
            let t4 = _mm256_unpacklo_ps(r[4], r[5]);
            let t5 = _mm256_unpackhi_ps(r[4], r[5]);
            let t6 = _mm256_unpacklo_ps(r[6], r[7]);
            let t7 = _mm256_unpackhi_ps(r[6], r[7]);

            // Step 2: shuffle 64-bit pairs
            let u0 = _mm256_castpd_ps(_mm256_unpacklo_pd(
                _mm256_castps_pd(t0),
                _mm256_castps_pd(t2),
            ));
            let u1 = _mm256_castpd_ps(_mm256_unpackhi_pd(
                _mm256_castps_pd(t0),
                _mm256_castps_pd(t2),
            ));
            let u2 = _mm256_castpd_ps(_mm256_unpacklo_pd(
                _mm256_castps_pd(t1),
                _mm256_castps_pd(t3),
            ));
            let u3 = _mm256_castpd_ps(_mm256_unpackhi_pd(
                _mm256_castps_pd(t1),
                _mm256_castps_pd(t3),
            ));
            let u4 = _mm256_castpd_ps(_mm256_unpacklo_pd(
                _mm256_castps_pd(t4),
                _mm256_castps_pd(t6),
            ));
            let u5 = _mm256_castpd_ps(_mm256_unpackhi_pd(
                _mm256_castps_pd(t4),
                _mm256_castps_pd(t6),
            ));
            let u6 = _mm256_castpd_ps(_mm256_unpacklo_pd(
                _mm256_castps_pd(t5),
                _mm256_castps_pd(t7),
            ));
            let u7 = _mm256_castpd_ps(_mm256_unpackhi_pd(
                _mm256_castps_pd(t5),
                _mm256_castps_pd(t7),
            ));

            // Step 3: permute 128-bit halves
            let out0 = _mm256_permute2f128_ps(u0, u4, 0x20);
            let out1 = _mm256_permute2f128_ps(u1, u5, 0x20);
            let out2 = _mm256_permute2f128_ps(u2, u6, 0x20);
            let out3 = _mm256_permute2f128_ps(u3, u7, 0x20);
            let out4 = _mm256_permute2f128_ps(u0, u4, 0x31);
            let out5 = _mm256_permute2f128_ps(u1, u5, 0x31);
            let out6 = _mm256_permute2f128_ps(u2, u6, 0x31);
            let out7 = _mm256_permute2f128_ps(u3, u7, 0x31);

            // Store transposed 8x8 block
            _mm256_storeu_ps(output.as_mut_ptr().add(c0 * rows + r0), out0);
            _mm256_storeu_ps(output.as_mut_ptr().add((c0 + 1) * rows + r0), out1);
            _mm256_storeu_ps(output.as_mut_ptr().add((c0 + 2) * rows + r0), out2);
            _mm256_storeu_ps(output.as_mut_ptr().add((c0 + 3) * rows + r0), out3);
            _mm256_storeu_ps(output.as_mut_ptr().add((c0 + 4) * rows + r0), out4);
            _mm256_storeu_ps(output.as_mut_ptr().add((c0 + 5) * rows + r0), out5);
            _mm256_storeu_ps(output.as_mut_ptr().add((c0 + 6) * rows + r0), out6);
            _mm256_storeu_ps(output.as_mut_ptr().add((c0 + 7) * rows + r0), out7);
        }
    }

    // Handle remaining rows (bottom edge)
    let row_tail = block_rows * 8;
    for r in row_tail..rows {
        for c in 0..cols {
            output[c * rows + r] = input[r * cols + c];
        }
    }

    // Handle remaining cols (right edge, only for block rows)
    let col_tail = block_cols * 8;
    for r in 0..row_tail {
        for c in col_tail..cols {
            output[c * rows + r] = input[r * cols + c];
        }
    }
}

// ---------------------------------------------------------------------------
// Public dispatch: auto-selects best SIMD path
// ---------------------------------------------------------------------------

/// Out-of-place 2D matrix transpose.
///
/// Transposes a `rows x cols` row-major matrix in `input` into a `cols x rows`
/// row-major matrix in `output`.
///
/// Uses block transpose with SIMD for cache efficiency:
/// - NEON: 4x4 blocks
/// - AVX2: 8x8 blocks
/// - Scalar fallback for edges and non-SIMD platforms
///
/// # Panics
/// Panics if `input.len() != rows * cols` or `output.len() != rows * cols`.
pub fn transpose_2d(input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    assert_eq!(
        input.len(),
        rows * cols,
        "input length must equal rows * cols"
    );
    assert_eq!(
        output.len(),
        rows * cols,
        "output length must equal rows * cols"
    );

    if rows == 0 || cols == 0 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        transpose_neon(input, output, rows, cols);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected above.
            unsafe {
                transpose_avx2_inner(input, output, rows, cols);
            }
            return;
        }
    }

    #[allow(unreachable_code)]
    transpose_scalar(input, output, rows, cols);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "simd_transpose_tests.rs"]
mod simd_transpose_tests;
