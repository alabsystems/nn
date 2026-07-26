// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized Rotary Position Embeddings (RoPE) with explicit scalar/NEON/AVX2 entry points.
//!
//! This module provides named entry points for each SIMD tier:
//! - `rope_apply` — auto-dispatch to best available (in-place)
//! - `rope_reference` — pure scalar reference (returns new Vec)
//!
//! RoPE applies a rotation to pairs of elements in the embedding dimension:
//!   x_new[i]        = x[i] * cos[i] - x[i + half] * sin[i]
//!   x_new[i + half] = x[i] * sin[i] + x[i + half] * cos[i]
//!
//! where `half = head_dim / 2`.

/// Chunk size for SIMD processing (8 f32 lanes for AVX2 compatibility).
pub const ROPE_CHUNK_SIZE: usize = 8;

// ---------------------------------------------------------------------------
// Scalar reference (returns new Vec)
// ---------------------------------------------------------------------------

/// Pure scalar RoPE reference implementation returning a new Vec.
///
/// Layout: `x` is `[seq_len, num_heads, head_dim]` flattened in row-major order.
/// `cos_cache` and `sin_cache` are `[seq_len, head_dim/2]` flattened.
///
/// For each (seq, head) pair and for each pair index `i` in `0..head_dim/2`:
///   out[..., i]        = x[..., i] * cos[seq, i] - x[..., i+half] * sin[seq, i]
///   out[..., i + half] = x[..., i] * sin[seq, i] + x[..., i+half] * cos[seq, i]
pub fn rope_reference(
    x: &[f32],
    cos_cache: &[f32],
    sin_cache: &[f32],
    head_dim: usize,
    seq_len: usize,
    num_heads: usize,
) -> Vec<f32> {
    assert!(
        head_dim > 0 && head_dim.is_multiple_of(2),
        "head_dim must be even and > 0"
    );
    let half = head_dim / 2;
    let total = seq_len * num_heads * head_dim;
    assert_eq!(
        x.len(),
        total,
        "x length must equal seq_len * num_heads * head_dim"
    );
    assert_eq!(
        cos_cache.len(),
        seq_len * half,
        "cos_cache length must equal seq_len * head_dim/2"
    );
    assert_eq!(
        sin_cache.len(),
        seq_len * half,
        "sin_cache length must equal seq_len * head_dim/2"
    );

    let mut out = vec![0.0f32; total];

    for s in 0..seq_len {
        let cache_offset = s * half;
        for h in 0..num_heads {
            let base = (s * num_heads + h) * head_dim;
            for i in 0..half {
                let cos_val = cos_cache[cache_offset + i];
                let sin_val = sin_cache[cache_offset + i];
                let x_lo = x[base + i];
                let x_hi = x[base + half + i];
                out[base + i] = x_lo * cos_val - x_hi * sin_val;
                out[base + half + i] = x_lo * sin_val + x_hi * cos_val;
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// In-place scalar fallback
// ---------------------------------------------------------------------------

/// Scalar RoPE applied in-place.
fn rope_apply_scalar(
    x: &mut [f32],
    cos_cache: &[f32],
    sin_cache: &[f32],
    head_dim: usize,
    seq_len: usize,
    num_heads: usize,
) {
    let half = head_dim / 2;

    for s in 0..seq_len {
        let cache_offset = s * half;
        for h in 0..num_heads {
            let base = (s * num_heads + h) * head_dim;
            for i in 0..half {
                let cos_val = cos_cache[cache_offset + i];
                let sin_val = sin_cache[cache_offset + i];
                let x_lo = x[base + i];
                let x_hi = x[base + half + i];
                x[base + i] = x_lo * cos_val - x_hi * sin_val;
                x[base + half + i] = x_lo * sin_val + x_hi * cos_val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NEON (aarch64) — 128-bit, 4x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
fn rope_apply_neon(
    x: &mut [f32],
    cos_cache: &[f32],
    sin_cache: &[f32],
    head_dim: usize,
    seq_len: usize,
    num_heads: usize,
) {
    use std::arch::aarch64::*;

    let half = head_dim / 2;
    let chunks = half / 4;
    let remainder = half % 4;

    for s in 0..seq_len {
        let cache_offset = s * half;
        for h in 0..num_heads {
            let base = (s * num_heads + h) * head_dim;

            // SAFETY: aarch64 NEON is always available. All pointer offsets are bounded
            // by base + half + i < total, verified by the assertion in rope_apply.
            unsafe {
                for c in 0..chunks {
                    let i = c * 4;
                    let vc = vld1q_f32(cos_cache.as_ptr().add(cache_offset + i));
                    let vs = vld1q_f32(sin_cache.as_ptr().add(cache_offset + i));
                    let vlo = vld1q_f32(x.as_ptr().add(base + i));
                    let vhi = vld1q_f32(x.as_ptr().add(base + half + i));

                    // x_new_lo = x_lo * cos - x_hi * sin
                    // vfmaq_f32(acc, a, b) = acc + a * b
                    // vnegq_f32 negates all lanes
                    let neg_sin = vnegq_f32(vs);
                    let new_lo = vfmaq_f32(vmulq_f32(vlo, vc), vhi, neg_sin);

                    // x_new_hi = x_lo * sin + x_hi * cos
                    let new_hi = vfmaq_f32(vmulq_f32(vhi, vc), vlo, vs);

                    vst1q_f32(x.as_mut_ptr().add(base + i), new_lo);
                    vst1q_f32(x.as_mut_ptr().add(base + half + i), new_hi);
                }
            }

            // Scalar tail
            let tail_start = chunks * 4;
            for r in 0..remainder {
                let i = tail_start + r;
                let cos_val = cos_cache[cache_offset + i];
                let sin_val = sin_cache[cache_offset + i];
                let x_lo = x[base + i];
                let x_hi = x[base + half + i];
                x[base + i] = x_lo * cos_val - x_hi * sin_val;
                x[base + half + i] = x_lo * sin_val + x_hi * cos_val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64) — 256-bit, 8x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn rope_apply_avx2_inner(
    x: &mut [f32],
    cos_cache: &[f32],
    sin_cache: &[f32],
    head_dim: usize,
    seq_len: usize,
    num_heads: usize,
) {
    use std::arch::x86_64::*;

    let half = head_dim / 2;
    let chunks = half / 8;
    let remainder = half % 8;

    for s in 0..seq_len {
        let cache_offset = s * half;
        for h in 0..num_heads {
            let base = (s * num_heads + h) * head_dim;

            for c in 0..chunks {
                let i = c * 8;
                // SAFETY: All pointer offsets bounded by slice length (asserted in rope_apply).
                let vc = _mm256_loadu_ps(cos_cache.as_ptr().add(cache_offset + i));
                let vs = _mm256_loadu_ps(sin_cache.as_ptr().add(cache_offset + i));
                let vlo = _mm256_loadu_ps(x.as_ptr().add(base + i));
                let vhi = _mm256_loadu_ps(x.as_ptr().add(base + half + i));

                // new_lo = x_lo * cos - x_hi * sin
                // fnmadd(a, b, c) = -(a*b) + c = c - a*b
                let lo_cos = _mm256_mul_ps(vlo, vc);
                let new_lo = _mm256_fnmadd_ps(vhi, vs, lo_cos);

                // new_hi = x_hi * cos + x_lo * sin
                let hi_cos = _mm256_mul_ps(vhi, vc);
                let new_hi = _mm256_fmadd_ps(vlo, vs, hi_cos);

                _mm256_storeu_ps(x.as_mut_ptr().add(base + i), new_lo);
                _mm256_storeu_ps(x.as_mut_ptr().add(base + half + i), new_hi);
            }

            // Scalar tail
            let tail_start = chunks * 8;
            for r in 0..remainder {
                let i = tail_start + r;
                let cos_val = cos_cache[cache_offset + i];
                let sin_val = sin_cache[cache_offset + i];
                let x_lo = x[base + i];
                let x_hi = x[base + half + i];
                x[base + i] = x_lo * cos_val - x_hi * sin_val;
                x[base + half + i] = x_lo * sin_val + x_hi * cos_val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public dispatch: applies RoPE in-place using best available SIMD
// ---------------------------------------------------------------------------

/// Applies Rotary Position Embeddings (RoPE) in-place.
///
/// Layout: `x` is `[seq_len, num_heads, head_dim]` flattened in row-major order.
/// `cos_cache` and `sin_cache` are `[seq_len, head_dim/2]` flattened.
///
/// # Panics
/// Panics if `head_dim` is 0 or odd, or if slice lengths are inconsistent.
pub fn rope_apply(
    x: &mut [f32],
    cos_cache: &[f32],
    sin_cache: &[f32],
    head_dim: usize,
    seq_len: usize,
    num_heads: usize,
) {
    assert!(
        head_dim > 0 && head_dim.is_multiple_of(2),
        "head_dim must be even and > 0"
    );
    let half = head_dim / 2;
    let total = seq_len * num_heads * head_dim;
    assert_eq!(
        x.len(),
        total,
        "x length must equal seq_len * num_heads * head_dim"
    );
    assert_eq!(
        cos_cache.len(),
        seq_len * half,
        "cos_cache length must equal seq_len * head_dim/2"
    );
    assert_eq!(
        sin_cache.len(),
        seq_len * half,
        "sin_cache length must equal seq_len * head_dim/2"
    );

    #[cfg(target_arch = "aarch64")]
    {
        rope_apply_neon(x, cos_cache, sin_cache, head_dim, seq_len, num_heads);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA detected above.
            unsafe {
                rope_apply_avx2_inner(x, cos_cache, sin_cache, head_dim, seq_len, num_heads);
            }
            return;
        }
    }

    #[allow(unreachable_code)]
    rope_apply_scalar(x, cos_cache, sin_cache, head_dim, seq_len, num_heads);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "simd_rope_tests.rs"]
mod simd_rope_tests;
