// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized multi-head scaled dot-product attention.
//!
//! Computes multi-head attention: for each head h in [0, num_heads):
//!   Q_h = Q[:, h*head_dim..(h+1)*head_dim]   shape [seq_len, head_dim]
//!   K_h = K[:, h*head_dim..(h+1)*head_dim]   shape [kv_seq_len, head_dim]
//!   V_h = V[:, h*head_dim..(h+1)*head_dim]   shape [kv_seq_len, head_dim]
//!   output_h = softmax(Q_h * K_h^T / sqrt(head_dim)) * V_h
//!
//! Reuses `crate::reduction::dot` for SIMD dot products and
//! `crate::softmax::softmax` for numerically stable softmax.
//! NEON (aarch64) and AVX2 (x86_64) paths with scalar fallback.

/// Configuration for multi-head scaled dot-product attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Query sequence length.
    pub seq_len: usize,
    /// Key/value sequence length (may differ from seq_len for cross-attention).
    pub kv_seq_len: usize,
    /// If true, apply causal (triangular) mask: position i can only attend
    /// to positions j <= i.
    pub causal: bool,
}

impl AttentionConfig {
    /// Total model dimension (num_heads * head_dim).
    #[inline]
    pub fn model_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Scale factor: 1 / sqrt(head_dim).
    #[inline]
    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

/// SIMD-optimized multi-head scaled dot-product attention.
///
/// Q: `[seq_len, num_heads * head_dim]` row-major
/// K: `[kv_seq_len, num_heads * head_dim]` row-major
/// V: `[kv_seq_len, num_heads * head_dim]` row-major
/// output: `[seq_len, num_heads * head_dim]` row-major
///
/// Each head is processed independently. Within each head, the algorithm is:
///   1. Compute scores = Q_h * K_h^T * scale (SIMD dot products)
///   2. Apply causal mask if configured
///   3. Softmax over scores
///   4. Compute output_h = attn_weights * V_h (SIMD weighted sum)
pub fn scaled_dot_product_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    config: &AttentionConfig,
) {
    let model_dim = config.model_dim();
    assert_eq!(
        q.len(),
        config.seq_len * model_dim,
        "Q must be [seq_len, num_heads * head_dim]"
    );
    assert_eq!(
        k.len(),
        config.kv_seq_len * model_dim,
        "K must be [kv_seq_len, num_heads * head_dim]"
    );
    assert_eq!(
        v.len(),
        config.kv_seq_len * model_dim,
        "V must be [kv_seq_len, num_heads * head_dim]"
    );
    assert_eq!(
        output.len(),
        config.seq_len * model_dim,
        "output must be [seq_len, num_heads * head_dim]"
    );

    if config.seq_len == 0 || config.kv_seq_len == 0 || config.head_dim == 0 {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        sdpa_neon(q, k, v, output, config);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA detected above.
            unsafe { sdpa_avx2(q, k, v, output, config) };
            return;
        }
    }

    #[allow(unreachable_code)]
    sdpa_scalar(q, k, v, output, config);
}

/// CPU reference implementation for verification.
///
/// Pure scalar, no SIMD, no fast-exp approximation. Uses `f32::exp()`.
/// Intended as a ground-truth reference for differential testing.
pub fn attention_reference(q: &[f32], k: &[f32], v: &[f32], config: &AttentionConfig) -> Vec<f32> {
    let model_dim = config.model_dim();
    let mut output = vec![0.0f32; config.seq_len * model_dim];
    let scale = config.scale();

    for h in 0..config.num_heads {
        for i in 0..config.seq_len {
            // Step 1: Compute scaled dot-product scores for this query row.
            let mut scores = vec![0.0f32; config.kv_seq_len];
            for j in 0..config.kv_seq_len {
                let mut dot = 0.0f32;
                for d in 0..config.head_dim {
                    let q_idx = i * model_dim + h * config.head_dim + d;
                    let k_idx = j * model_dim + h * config.head_dim + d;
                    dot += q[q_idx] * k[k_idx];
                }
                scores[j] = dot * scale;
            }

            // Step 2: Apply causal mask.
            if config.causal {
                scores[(i + 1)..config.kv_seq_len].fill(f32::NEG_INFINITY);
            }

            // Step 3: Softmax (exact, using f32::exp).
            let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|&s| (s - max_val).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            let attn: Vec<f32> = exps.iter().map(|&e| e * inv_sum).collect();

            // Step 4: Weighted sum of V rows.
            for d in 0..config.head_dim {
                let mut acc = 0.0f32;
                for j in 0..config.kv_seq_len {
                    let v_idx = j * model_dim + h * config.head_dim + d;
                    acc += attn[j] * v[v_idx];
                }
                let out_idx = i * model_dim + h * config.head_dim + d;
                output[out_idx] = acc;
            }
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

/// Scalar multi-head attention (no SIMD, uses exact f32::exp).
fn sdpa_scalar(q: &[f32], k: &[f32], v: &[f32], output: &mut [f32], config: &AttentionConfig) {
    let model_dim = config.model_dim();
    let scale = config.scale();
    let mut scores = vec![0.0f32; config.kv_seq_len];
    let mut attn_weights = vec![0.0f32; config.kv_seq_len];

    for h in 0..config.num_heads {
        let head_off = h * config.head_dim;

        for i in 0..config.seq_len {
            // Step 1: Compute scaled scores.
            for j in 0..config.kv_seq_len {
                let mut dot = 0.0f32;
                for d in 0..config.head_dim {
                    dot += q[i * model_dim + head_off + d] * k[j * model_dim + head_off + d];
                }
                scores[j] = dot * scale;
            }

            // Step 2: Causal mask.
            if config.causal {
                scores[(i + 1)..config.kv_seq_len].fill(f32::NEG_INFINITY);
            }

            // Step 3: Softmax.
            softmax_row_scalar(&scores, &mut attn_weights);

            // Step 4: Weighted sum.
            for d in 0..config.head_dim {
                let mut acc = 0.0f32;
                for j in 0..config.kv_seq_len {
                    acc += attn_weights[j] * v[j * model_dim + head_off + d];
                }
                output[i * model_dim + head_off + d] = acc;
            }
        }
    }
}

/// Numerically stable softmax over a single row (scalar).
fn softmax_row_scalar(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    assert_eq!(n, output.len());

    // Pass 1: max
    let mut max_val = f32::NEG_INFINITY;
    for &x in input {
        if x > max_val {
            max_val = x;
        }
    }

    // Pass 2: exp(x - max)
    let mut sum = 0.0f32;
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        let e = (x - max_val).exp();
        *o = e;
        sum += e;
    }

    // Pass 3: normalize
    let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for o in output.iter_mut() {
        *o *= inv_sum;
    }
}

// ---------------------------------------------------------------------------
// NEON (aarch64) -- 128-bit, 4x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
fn sdpa_neon(q: &[f32], k: &[f32], v: &[f32], output: &mut [f32], config: &AttentionConfig) {
    use std::arch::aarch64::*;

    let model_dim = config.model_dim();
    let scale = config.scale();
    let mut scores = vec![0.0f32; config.kv_seq_len];
    let mut attn_weights = vec![0.0f32; config.kv_seq_len];

    for h in 0..config.num_heads {
        let head_off = h * config.head_dim;

        for i in 0..config.seq_len {
            let q_base = i * model_dim + head_off;

            // Step 1: SIMD dot products for Q[i,h] . K[j,h] * scale.
            for j in 0..config.kv_seq_len {
                let k_base = j * model_dim + head_off;
                let dot = crate::reduction::dot(
                    &q[q_base..q_base + config.head_dim],
                    &k[k_base..k_base + config.head_dim],
                );
                scores[j] = dot * scale;
            }

            // Step 2: Causal mask.
            if config.causal {
                scores[(i + 1)..config.kv_seq_len].fill(f32::NEG_INFINITY);
            }

            // Step 3: Softmax (SIMD).
            crate::softmax::softmax(
                &scores[..config.kv_seq_len],
                &mut attn_weights[..config.kv_seq_len],
                config.kv_seq_len,
            );

            // Step 4: NEON weighted sum of V rows.
            let out_base = i * model_dim + head_off;
            let out_row = &mut output[out_base..out_base + config.head_dim];
            for o in out_row.iter_mut() {
                *o = 0.0;
            }

            for j in 0..config.kv_seq_len {
                let w = attn_weights[j];
                if w == 0.0 {
                    continue;
                }
                let v_base = j * model_dim + head_off;
                let v_row = &v[v_base..v_base + config.head_dim];

                // SAFETY: NEON is always available on aarch64. Bounded loads/stores.
                unsafe {
                    let vw = vdupq_n_f32(w);
                    let chunks = config.head_dim / 4;
                    let remainder = config.head_dim % 4;

                    for c in 0..chunks {
                        let offset = c * 4;
                        let vo = vld1q_f32(out_row.as_ptr().add(offset));
                        let vv = vld1q_f32(v_row.as_ptr().add(offset));
                        let result = vfmaq_f32(vo, vv, vw);
                        vst1q_f32(out_row.as_mut_ptr().add(offset), result);
                    }

                    let tail_start = chunks * 4;
                    for r in 0..remainder {
                        out_row[tail_start + r] += w * v_row[tail_start + r];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2 (x86_64) -- 256-bit, 8x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sdpa_avx2(q: &[f32], k: &[f32], v: &[f32], output: &mut [f32], config: &AttentionConfig) {
    use std::arch::x86_64::*;

    let model_dim = config.model_dim();
    let scale = config.scale();
    let mut scores = vec![0.0f32; config.kv_seq_len];
    let mut attn_weights = vec![0.0f32; config.kv_seq_len];

    for h in 0..config.num_heads {
        let head_off = h * config.head_dim;

        for i in 0..config.seq_len {
            let q_base = i * model_dim + head_off;

            // Step 1: SIMD dot products for Q[i,h] . K[j,h] * scale.
            for j in 0..config.kv_seq_len {
                let k_base = j * model_dim + head_off;
                let dot = crate::reduction::dot(
                    &q[q_base..q_base + config.head_dim],
                    &k[k_base..k_base + config.head_dim],
                );
                scores[j] = dot * scale;
            }

            // Step 2: Causal mask.
            if config.causal {
                scores[(i + 1)..config.kv_seq_len].fill(f32::NEG_INFINITY);
            }

            // Step 3: Softmax (SIMD).
            crate::softmax::softmax(
                &scores[..config.kv_seq_len],
                &mut attn_weights[..config.kv_seq_len],
                config.kv_seq_len,
            );

            // Step 4: AVX2 weighted sum of V rows.
            let out_base = i * model_dim + head_off;
            let out_row = &mut output[out_base..out_base + config.head_dim];
            for o in out_row.iter_mut() {
                *o = 0.0;
            }

            for j in 0..config.kv_seq_len {
                let w = attn_weights[j];
                if w == 0.0 {
                    continue;
                }
                let v_base = j * model_dim + head_off;
                let v_row = &v[v_base..v_base + config.head_dim];

                let vw = _mm256_set1_ps(w);
                let chunks = config.head_dim / 8;
                let remainder = config.head_dim % 8;

                for c in 0..chunks {
                    let offset = c * 8;
                    let vo = _mm256_loadu_ps(out_row.as_ptr().add(offset));
                    let vv = _mm256_loadu_ps(v_row.as_ptr().add(offset));
                    let result = _mm256_fmadd_ps(vv, vw, vo);
                    _mm256_storeu_ps(out_row.as_mut_ptr().add(offset), result);
                }

                let tail_start = chunks * 8;
                for r in 0..remainder {
                    out_row[tail_start + r] += w * v_row[tail_start + r];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "simd_attention_tests.rs"]
mod simd_attention_tests;
