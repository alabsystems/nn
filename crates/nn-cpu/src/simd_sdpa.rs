// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized Scaled Dot-Product Attention (SDPA).
//!
//! Computes: `output = softmax(Q * K^T * scale) * V` per batch per head.
//!
//! Layout (all flattened row-major):
//! - Q: `[batch, num_heads, seq_len, head_dim]`
//! - K: `[batch, num_heads, seq_len, head_dim]`
//! - V: `[batch, num_heads, seq_len, head_dim]`
//! - output: `[batch, num_heads, seq_len, head_dim]`
//!
//! NEON (aarch64) and AVX2 (x86_64) paths with scalar fallback.

use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors for SDPA.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SdpaError {
    /// Zero dimension parameter.
    ZeroDim { param: &'static str },
    /// Input length mismatch.
    InvalidLength {
        name: &'static str,
        got: usize,
        expected: usize,
    },
}

impl fmt::Display for SdpaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDim { param } => write!(f, "{param} must be > 0"),
            Self::InvalidLength {
                name,
                got,
                expected,
            } => {
                write!(f, "{name} length {got}, expected {expected}")
            }
        }
    }
}

impl std::error::Error for SdpaError {}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    output: &mut [f32],
    batch: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<(), SdpaError> {
    if batch == 0 {
        return Err(SdpaError::ZeroDim { param: "batch" });
    }
    if num_heads == 0 {
        return Err(SdpaError::ZeroDim { param: "num_heads" });
    }
    if seq_len == 0 {
        return Err(SdpaError::ZeroDim { param: "seq_len" });
    }
    if head_dim == 0 {
        return Err(SdpaError::ZeroDim { param: "head_dim" });
    }
    let total = batch * num_heads * seq_len * head_dim;
    if query.len() != total {
        return Err(SdpaError::InvalidLength {
            name: "query",
            got: query.len(),
            expected: total,
        });
    }
    if key.len() != total {
        return Err(SdpaError::InvalidLength {
            name: "key",
            got: key.len(),
            expected: total,
        });
    }
    if value.len() != total {
        return Err(SdpaError::InvalidLength {
            name: "value",
            got: value.len(),
            expected: total,
        });
    }
    if output.len() != total {
        return Err(SdpaError::InvalidLength {
            name: "output",
            got: output.len(),
            expected: total,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Scalar reference
// ---------------------------------------------------------------------------

/// Pure scalar SDPA reference implementation.
///
/// `output = softmax(Q * K^T * scale) * V`
///
/// All tensors: `[batch, num_heads, seq_len, head_dim]` flattened.
pub fn sdpa_reference(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    output: &mut [f32],
    batch: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) -> Result<(), SdpaError> {
    validate(
        query, key, value, output, batch, num_heads, seq_len, head_dim,
    )?;
    sdpa_scalar_inner(
        query, key, value, output, batch, num_heads, seq_len, head_dim, scale,
    );
    Ok(())
}

fn sdpa_scalar_inner(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    output: &mut [f32],
    batch: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    let head_stride = seq_len * head_dim;
    let batch_stride = num_heads * head_stride;

    // Temporary buffer for attention scores: [seq_len, seq_len]
    let mut scores = vec![0.0f32; seq_len * seq_len];

    for b in 0..batch {
        for h in 0..num_heads {
            let base = b * batch_stride + h * head_stride;

            // Compute Q * K^T * scale => scores[i][j]
            for i in 0..seq_len {
                let q_row = base + i * head_dim;
                for j in 0..seq_len {
                    let k_row = base + j * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += query[q_row + d] * key[k_row + d];
                    }
                    scores[i * seq_len + j] = dot * scale;
                }
            }

            // Softmax each row of scores.
            for i in 0..seq_len {
                let row_start = i * seq_len;
                let row = &mut scores[row_start..row_start + seq_len];

                // Numerically stable: subtract max.
                let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max_val).exp();
                    sum += *v;
                }
                if sum > 0.0 {
                    for v in row.iter_mut() {
                        *v /= sum;
                    }
                }
            }

            // Multiply scores * V => output.
            for i in 0..seq_len {
                let out_row = base + i * head_dim;
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for j in 0..seq_len {
                        acc += scores[i * seq_len + j] * value[base + j * head_dim + d];
                    }
                    output[out_row + d] = acc;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SIMD-accelerated inner: dot products + softmax
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
fn sdpa_neon_inner(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    output: &mut [f32],
    batch: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    use std::arch::aarch64::*;

    let head_stride = seq_len * head_dim;
    let batch_stride = num_heads * head_stride;
    let mut scores = vec![0.0f32; seq_len * seq_len];

    for b in 0..batch {
        for h in 0..num_heads {
            let base = b * batch_stride + h * head_stride;

            // Q * K^T with NEON dot products.
            for i in 0..seq_len {
                let q_row = base + i * head_dim;
                for j in 0..seq_len {
                    let k_row = base + j * head_dim;
                    let chunks = head_dim / 4;
                    let tail_start = chunks * 4;

                    // SAFETY: NEON always available on aarch64.
                    // Bounded loads within query/key slices.
                    let mut dot = unsafe {
                        let mut acc = vdupq_n_f32(0.0);
                        for c in 0..chunks {
                            let off = c * 4;
                            let qv = vld1q_f32(query.as_ptr().add(q_row + off));
                            let kv = vld1q_f32(key.as_ptr().add(k_row + off));
                            acc = vfmaq_f32(acc, qv, kv);
                        }
                        vaddvq_f32(acc)
                    };

                    for d in tail_start..head_dim {
                        dot += query[q_row + d] * key[k_row + d];
                    }
                    scores[i * seq_len + j] = dot * scale;
                }
            }

            // Softmax rows (NEON-accelerated exp + reduce).
            for i in 0..seq_len {
                let row_start = i * seq_len;
                let row = &mut scores[row_start..row_start + seq_len];

                let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max_val).exp();
                    sum += *v;
                }
                if sum > 0.0 {
                    let inv_sum = 1.0 / sum;
                    let sm_chunks = seq_len / 4;
                    // SAFETY: NEON always available, bounded stores.
                    unsafe {
                        let inv_v = vdupq_n_f32(inv_sum);
                        for c in 0..sm_chunks {
                            let off = row_start + c * 4;
                            let ptr = scores.as_mut_ptr().add(off);
                            let v = vld1q_f32(ptr);
                            vst1q_f32(ptr, vmulq_f32(v, inv_v));
                        }
                    }
                    for idx in (sm_chunks * 4)..seq_len {
                        scores[row_start + idx] *= inv_sum;
                    }
                }
            }

            // Scores * V with NEON accumulation.
            for i in 0..seq_len {
                let out_row = base + i * head_dim;
                for d in 0..head_dim {
                    let chunks = seq_len / 4;
                    let tail_start = chunks * 4;
                    // SAFETY: NEON always available, bounded loads.
                    let mut acc = unsafe {
                        let mut vacc = vdupq_n_f32(0.0);
                        for c in 0..chunks {
                            let j_base = c * 4;
                            let sv = vld1q_f32(scores.as_ptr().add(i * seq_len + j_base));
                            let mut vv = [0.0f32; 4];
                            for lane in 0..4 {
                                vv[lane] = value[base + (j_base + lane) * head_dim + d];
                            }
                            let val_v = vld1q_f32(vv.as_ptr());
                            vacc = vfmaq_f32(vacc, sv, val_v);
                        }
                        vaddvq_f32(vacc)
                    };
                    for j in tail_start..seq_len {
                        acc += scores[i * seq_len + j] * value[base + j * head_dim + d];
                    }
                    output[out_row + d] = acc;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sdpa_avx2_inner(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    output: &mut [f32],
    batch: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    use std::arch::x86_64::*;

    let head_stride = seq_len * head_dim;
    let batch_stride = num_heads * head_stride;
    let mut scores = vec![0.0f32; seq_len * seq_len];

    for b in 0..batch {
        for h in 0..num_heads {
            let base = b * batch_stride + h * head_stride;

            // Q * K^T with AVX2 dot products.
            for i in 0..seq_len {
                let q_row = base + i * head_dim;
                for j in 0..seq_len {
                    let k_row = base + j * head_dim;
                    let chunks = head_dim / 8;
                    let tail_start = chunks * 8;

                    let mut acc = _mm256_setzero_ps();
                    for c in 0..chunks {
                        let off = c * 8;
                        let qv = _mm256_loadu_ps(query.as_ptr().add(q_row + off));
                        let kv = _mm256_loadu_ps(key.as_ptr().add(k_row + off));
                        acc = _mm256_fmadd_ps(qv, kv, acc);
                    }
                    // Horizontal sum of 8 lanes.
                    let hi = _mm256_extractf128_ps::<1>(acc);
                    let lo = _mm256_castps256_ps128(acc);
                    let sum4 = _mm_add_ps(lo, hi);
                    let shuf = _mm_movehdup_ps(sum4);
                    let sums = _mm_add_ps(sum4, shuf);
                    let shuf2 = _mm_movehl_ps(sums, sums);
                    let s = _mm_add_ss(sums, shuf2);
                    let mut dot = _mm_cvtss_f32(s);

                    for d in tail_start..head_dim {
                        dot += query[q_row + d] * key[k_row + d];
                    }
                    scores[i * seq_len + j] = dot * scale;
                }
            }

            // Softmax rows.
            for i in 0..seq_len {
                let row_start = i * seq_len;
                let row = &mut scores[row_start..row_start + seq_len];
                let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max_val).exp();
                    sum += *v;
                }
                if sum > 0.0 {
                    let inv_sum = 1.0 / sum;
                    let sm_chunks = seq_len / 8;
                    let inv_v = _mm256_set1_ps(inv_sum);
                    for c in 0..sm_chunks {
                        let off = row_start + c * 8;
                        let ptr = scores.as_mut_ptr().add(off);
                        _mm256_storeu_ps(ptr, _mm256_mul_ps(_mm256_loadu_ps(ptr), inv_v));
                    }
                    for idx in (sm_chunks * 8)..seq_len {
                        scores[row_start + idx] *= inv_sum;
                    }
                }
            }

            // Scores * V.
            for i in 0..seq_len {
                let out_row = base + i * head_dim;
                for d in 0..head_dim {
                    let chunks = seq_len / 8;
                    let tail_start = chunks * 8;
                    let mut vacc = _mm256_setzero_ps();
                    for c in 0..chunks {
                        let j_base = c * 8;
                        let sv = _mm256_loadu_ps(scores.as_ptr().add(i * seq_len + j_base));
                        let mut vv = [0.0f32; 8];
                        for lane in 0..8 {
                            vv[lane] = value[base + (j_base + lane) * head_dim + d];
                        }
                        let val_v = _mm256_loadu_ps(vv.as_ptr());
                        vacc = _mm256_fmadd_ps(sv, val_v, vacc);
                    }
                    // Horizontal sum.
                    let hi = _mm256_extractf128_ps::<1>(vacc);
                    let lo = _mm256_castps256_ps128(vacc);
                    let sum4 = _mm_add_ps(lo, hi);
                    let shuf = _mm_movehdup_ps(sum4);
                    let sums = _mm_add_ps(sum4, shuf);
                    let shuf2 = _mm_movehl_ps(sums, sums);
                    let s = _mm_add_ss(sums, shuf2);
                    let mut acc = _mm_cvtss_f32(s);

                    for j in tail_start..seq_len {
                        acc += scores[i * seq_len + j] * value[base + j * head_dim + d];
                    }
                    output[out_row + d] = acc;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public dispatch
// ---------------------------------------------------------------------------

/// SIMD-optimized Scaled Dot-Product Attention.
///
/// `output = softmax(Q * K^T * scale) * V`
///
/// All tensors: `[batch, num_heads, seq_len, head_dim]` flattened row-major.
///
/// Auto-dispatches to NEON (aarch64), AVX2 (x86_64), or scalar fallback.
pub fn sdpa(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    output: &mut [f32],
    batch: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) -> Result<(), SdpaError> {
    validate(
        query, key, value, output, batch, num_heads, seq_len, head_dim,
    )?;

    #[cfg(target_arch = "aarch64")]
    {
        sdpa_neon_inner(
            query, key, value, output, batch, num_heads, seq_len, head_dim, scale,
        );
        return Ok(());
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA detected above.
            unsafe {
                sdpa_avx2_inner(
                    query, key, value, output, batch, num_heads, seq_len, head_dim, scale,
                );
            }
            return Ok(());
        }
    }

    #[allow(unreachable_code)]
    {
        sdpa_scalar_inner(
            query, key, value, output, batch, num_heads, seq_len, head_dim, scale,
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "simd_sdpa_tests.rs"]
mod simd_sdpa_tests;
