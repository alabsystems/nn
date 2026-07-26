// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SIMD-optimized scaled dot-product attention score computation.
//!
//! Computes single-head attention: softmax(Q * K^T * scale) * V
//!
//! Algorithm (per query row i):
//!   1. score[j] = dot(Q[i], K[j]) * scale    (SIMD dot product)
//!   2. if causal && j > i: score[j] = -inf    (causal mask)
//!   3. attn[j] = softmax(score[j])            (numerically stable softmax)
//!   4. output[i] = sum_j(attn[j] * V[j])      (SIMD weighted sum)
//!
//! Reuses `crate::reduction::dot` for SIMD dot products and
//! `crate::softmax::softmax` for numerically stable softmax.
//! NEON (aarch64) and AVX2 (x86_64) paths with scalar fallback.

/// Compute scaled dot-product attention score for one head.
///
/// Q: `[seq_len, head_dim]` — query (row-major)
/// K: `[kv_len, head_dim]` — key (row-major)
/// V: `[kv_len, head_dim]` — value (row-major)
/// Output: `[seq_len, head_dim]` — attention output (row-major)
/// scale: typically `1/sqrt(head_dim)`
/// causal: if true, mask positions where j > i with -inf
pub fn attention_score_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) {
    assert_eq!(q.len(), seq_len * head_dim, "q must be [seq_len, head_dim]");
    assert_eq!(k.len(), kv_len * head_dim, "k must be [kv_len, head_dim]");
    assert_eq!(v.len(), kv_len * head_dim, "v must be [kv_len, head_dim]");
    assert_eq!(
        output.len(),
        seq_len * head_dim,
        "output must be [seq_len, head_dim]"
    );

    if seq_len == 0 || kv_len == 0 || head_dim == 0 {
        return;
    }

    // Dispatch to platform-specific implementation.
    #[cfg(target_arch = "aarch64")]
    {
        attention_score_neon(q, k, v, output, seq_len, kv_len, head_dim, scale, causal);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA detected above.
            unsafe {
                attention_score_avx2(q, k, v, output, seq_len, kv_len, head_dim, scale, causal);
            }
            return;
        }
    }

    #[allow(unreachable_code)]
    attention_score_scalar(q, k, v, output, seq_len, kv_len, head_dim, scale, causal);
}

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

/// Scalar attention score computation.
pub fn attention_score_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) {
    let mut scores = vec![0.0f32; kv_len];
    let mut attn_weights = vec![0.0f32; kv_len];

    for i in 0..seq_len {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];

        // Step 1: Compute scaled dot-product scores.
        for j in 0..kv_len {
            let k_row = &k[j * head_dim..(j + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_row[d] * k_row[d];
            }
            scores[j] = dot * scale;
        }

        // Step 2: Apply causal mask.
        if causal {
            scores[(i + 1)..kv_len].fill(f32::NEG_INFINITY);
        }

        // Step 3: Softmax over scores.
        softmax_row_scalar(&scores, &mut attn_weights);

        // Step 4: Weighted sum of V rows.
        let out_row = &mut output[i * head_dim..(i + 1) * head_dim];
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for j in 0..kv_len {
                acc += attn_weights[j] * v[j * head_dim + d];
            }
            out_row[d] = acc;
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
// NEON (aarch64) — 128-bit, 4x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
fn attention_score_neon(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) {
    use std::arch::aarch64::*;

    let mut scores = vec![0.0f32; kv_len];
    let mut attn_weights = vec![0.0f32; kv_len];

    for i in 0..seq_len {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];

        // Step 1: SIMD dot products for Q[i] * K[j] * scale.
        for j in 0..kv_len {
            let k_row = &k[j * head_dim..(j + 1) * head_dim];
            let dot = crate::reduction::dot(q_row, k_row);
            scores[j] = dot * scale;
        }

        // Step 2: Apply causal mask.
        if causal {
            scores[(i + 1)..kv_len].fill(f32::NEG_INFINITY);
        }

        // Step 3: Softmax over scores (SIMD).
        crate::softmax::softmax(&scores, &mut attn_weights, kv_len);

        // Step 4: Weighted sum of V rows using NEON.
        let out_row = &mut output[i * head_dim..(i + 1) * head_dim];
        // Zero the output row.
        for o in out_row.iter_mut() {
            *o = 0.0;
        }

        for j in 0..kv_len {
            let w = attn_weights[j];
            if w == 0.0 {
                continue;
            }
            let v_row = &v[j * head_dim..(j + 1) * head_dim];

            // SAFETY: NEON is always available on aarch64. Bounded loads/stores.
            unsafe {
                let vw = vdupq_n_f32(w);
                let chunks = head_dim / 4;
                let remainder = head_dim % 4;

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

// ---------------------------------------------------------------------------
// AVX2 (x86_64) — 256-bit, 8x f32 lanes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn attention_score_avx2(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) {
    use std::arch::x86_64::*;

    let mut scores = vec![0.0f32; kv_len];
    let mut attn_weights = vec![0.0f32; kv_len];

    for i in 0..seq_len {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];

        // Step 1: SIMD dot products for Q[i] * K[j] * scale.
        for j in 0..kv_len {
            let k_row = &k[j * head_dim..(j + 1) * head_dim];
            let dot = crate::reduction::dot(q_row, k_row);
            scores[j] = dot * scale;
        }

        // Step 2: Apply causal mask.
        if causal {
            scores[(i + 1)..kv_len].fill(f32::NEG_INFINITY);
        }

        // Step 3: Softmax over scores (SIMD).
        crate::softmax::softmax(&scores, &mut attn_weights, kv_len);

        // Step 4: Weighted sum of V rows using AVX2.
        let out_row = &mut output[i * head_dim..(i + 1) * head_dim];
        // Zero the output row.
        for o in out_row.iter_mut() {
            *o = 0.0;
        }

        for j in 0..kv_len {
            let w = attn_weights[j];
            if w == 0.0 {
                continue;
            }
            let v_row = &v[j * head_dim..(j + 1) * head_dim];

            let vw = _mm256_set1_ps(w);
            let chunks = head_dim / 8;
            let remainder = head_dim % 8;

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive reference attention for verification.
    fn naive_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        kv_len: usize,
        head_dim: usize,
        scale: f32,
        causal: bool,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; seq_len * head_dim];

        for i in 0..seq_len {
            // Compute scores.
            let mut scores = vec![0.0f32; kv_len];
            for j in 0..kv_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                scores[j] = dot * scale;
            }

            // Causal mask.
            if causal {
                scores[(i + 1)..kv_len].fill(f32::NEG_INFINITY);
            }

            // Softmax.
            let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|&s| (s - max_val).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let attn: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

            // Weighted sum.
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..kv_len {
                    acc += attn[j] * v[j * head_dim + d];
                }
                output[i * head_dim + d] = acc;
            }
        }

        output
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (va - vb).abs();
            assert!(
                diff <= tol,
                "{label}[{i}]: {va} vs {vb} (diff={diff}, tol={tol})"
            );
        }
    }

    #[test]
    fn test_attention_score_identity_like() {
        // Q = K = V = identity-like: 2x2 head, 2 seq, 2 kv.
        // Q = [[1, 0], [0, 1]]
        // K = [[1, 0], [0, 1]]
        // V = [[1, 0], [0, 1]]
        // scale = 1.0, no causal mask
        //
        // scores = Q * K^T = [[1, 0], [0, 1]]
        // attn = softmax([[1, 0], [0, 1]])
        //   row 0: softmax([1, 0]) = [e/(e+1), 1/(e+1)] ~ [0.731, 0.269]
        //   row 1: softmax([0, 1]) = [1/(1+e), e/(1+e)] ~ [0.269, 0.731]
        // output = attn * V = attn (since V is identity)
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let mut output = vec![0.0f32; 4];

        attention_score_scalar(&q, &k, &v, &mut output, 2, 2, 2, 1.0, false);

        let expected = naive_attention(&q, &k, &v, 2, 2, 2, 1.0, false);
        assert_close(&output, &expected, 1e-6, "identity_like_scalar");
    }

    #[test]
    fn test_attention_score_dispatch_matches_scalar() {
        // Random-ish values, non-causal.
        let q = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let k = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2];
        let v = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let seq_len = 2;
        let kv_len = 3;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut scalar_out = vec![0.0f32; seq_len * head_dim];
        attention_score_scalar(
            &q,
            &k,
            &v,
            &mut scalar_out,
            seq_len,
            kv_len,
            head_dim,
            scale,
            false,
        );

        let mut dispatch_out = vec![0.0f32; seq_len * head_dim];
        attention_score_f32(
            &q,
            &k,
            &v,
            &mut dispatch_out,
            seq_len,
            kv_len,
            head_dim,
            scale,
            false,
        );

        // SIMD softmax uses fast_exp (Schraudolph) approximation with ~1-2% relative
        // error per element, which compounds through softmax normalization and the
        // weighted sum over V. Allow wider tolerance for this compound error.
        assert_close(
            &scalar_out,
            &dispatch_out,
            1e-1,
            "dispatch_vs_scalar_noncausal",
        );
    }

    #[test]
    fn test_attention_score_causal_mask() {
        // 3 seq, 3 kv, head_dim=2. Causal mask means row i only attends to j <= i.
        let q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let v = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5];

        let seq_len = 3;
        let kv_len = 3;
        let head_dim = 2;
        let scale = 1.0;

        let mut output = vec![0.0f32; seq_len * head_dim];
        attention_score_scalar(
            &q,
            &k,
            &v,
            &mut output,
            seq_len,
            kv_len,
            head_dim,
            scale,
            true,
        );

        // Row 0: only attends to position 0 (j=0). attn = [1, 0, 0].
        // output[0] = V[0] = [1.0, 0.0]
        assert!(
            (output[0] - 1.0).abs() < 1e-6,
            "causal row0 d0: {}",
            output[0]
        );
        assert!(
            (output[1] - 0.0).abs() < 1e-6,
            "causal row0 d1: {}",
            output[1]
        );

        // Row 1: attends to positions 0 and 1 (j<=1).
        // scores = [dot(Q1,K0)*1, dot(Q1,K1)*1] = [0, 1]
        // softmax([0, 1]) = [1/(1+e), e/(1+e)] ~ [0.269, 0.731]
        // output[1] = 0.269 * V[0] + 0.731 * V[1]
        let expected = naive_attention(&q, &k, &v, seq_len, kv_len, head_dim, scale, true);
        assert_close(&output, &expected, 1e-6, "causal_mask_scalar");
    }

    #[test]
    fn test_attention_score_causal_dispatch() {
        let q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let v = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5];

        let seq_len = 3;
        let kv_len = 3;
        let head_dim = 2;
        let scale = 1.0;

        let mut scalar_out = vec![0.0f32; seq_len * head_dim];
        attention_score_scalar(
            &q,
            &k,
            &v,
            &mut scalar_out,
            seq_len,
            kv_len,
            head_dim,
            scale,
            true,
        );

        let mut dispatch_out = vec![0.0f32; seq_len * head_dim];
        attention_score_f32(
            &q,
            &k,
            &v,
            &mut dispatch_out,
            seq_len,
            kv_len,
            head_dim,
            scale,
            true,
        );

        assert_close(
            &scalar_out,
            &dispatch_out,
            5e-2,
            "causal_dispatch_vs_scalar",
        );
    }

    #[test]
    fn test_attention_score_single_token() {
        // seq_len=1, kv_len=1: simplest case.
        // Output should just be V (softmax of single score is 1.0).
        let q = vec![1.0, 2.0, 3.0];
        let k = vec![0.5, 0.5, 0.5];
        let v = vec![10.0, 20.0, 30.0];
        let mut output = vec![0.0f32; 3];

        attention_score_f32(&q, &k, &v, &mut output, 1, 1, 3, 0.5, false);

        // With kv_len=1, softmax([score]) = [1.0], so output = V.
        assert_close(&output, &v, 5e-2, "single_token");
    }

    #[test]
    fn test_attention_score_uniform_weights() {
        // When all scores are equal, softmax gives uniform weights.
        // Output is the mean of V rows.
        let head_dim = 4;
        let kv_len = 3;
        // Q and K designed so all dots are equal.
        let q = vec![1.0, 0.0, 0.0, 0.0]; // 1 query
        let k = vec![
            1.0, 0.0, 0.0, 0.0, // K[0]: dot = 1
            1.0, 0.0, 0.0, 0.0, // K[1]: dot = 1
            1.0, 0.0, 0.0, 0.0, // K[2]: dot = 1
        ];
        let v = vec![
            3.0, 6.0, 9.0, 12.0, // V[0]
            0.0, 0.0, 0.0, 0.0, // V[1]
            0.0, 0.0, 0.0, 0.0, // V[2]
        ];
        let mut output = vec![0.0f32; head_dim];

        attention_score_scalar(&q, &k, &v, &mut output, 1, kv_len, head_dim, 1.0, false);

        // Uniform attn = [1/3, 1/3, 1/3], output = mean of V rows.
        let expected = vec![1.0, 2.0, 3.0, 4.0];
        assert_close(&output, &expected, 1e-6, "uniform_weights");
    }

    #[test]
    fn test_attention_score_degenerate_empty() {
        let mut output = vec![0.0f32; 0];
        // seq_len=0 should be a no-op.
        attention_score_f32(
            &[],
            &[1.0, 2.0],
            &[3.0, 4.0],
            &mut output,
            0,
            1,
            2,
            1.0,
            false,
        );
        // No panic = pass.
    }

    #[test]
    fn test_attention_score_scale_effect() {
        // With a large scale, the highest-scoring key should dominate.
        let head_dim = 2;
        let q = vec![1.0, 0.0]; // 1 query
        let k = vec![
            1.0, 0.0, // K[0]: dot = 1
            0.0, 1.0, // K[1]: dot = 0
        ];
        let v = vec![
            10.0, 10.0, // V[0]
            0.0, 0.0, // V[1]
        ];

        // Small scale: more uniform.
        let mut out_small = vec![0.0f32; 2];
        attention_score_scalar(&q, &k, &v, &mut out_small, 1, 2, head_dim, 0.1, false);

        // Large scale: sharper attention on V[0].
        let mut out_large = vec![0.0f32; 2];
        attention_score_scalar(&q, &k, &v, &mut out_large, 1, 2, head_dim, 10.0, false);

        // With large scale, output should be closer to V[0] = [10, 10].
        assert!(
            out_large[0] > out_small[0],
            "large scale output ({}) should be closer to 10 than small scale ({})",
            out_large[0],
            out_small[0],
        );
    }

    #[test]
    fn test_attention_score_against_naive_larger() {
        // Larger test: 4 seq, 6 kv, head_dim=8.
        let seq_len = 4;
        let kv_len = 6;
        let head_dim = 8;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Deterministic pseudo-random values.
        let q: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 7 + 3) % 19) as f32 * 0.1 - 0.9)
            .collect();
        let k: Vec<f32> = (0..kv_len * head_dim)
            .map(|i| ((i * 11 + 5) % 23) as f32 * 0.1 - 1.1)
            .collect();
        let v: Vec<f32> = (0..kv_len * head_dim)
            .map(|i| ((i * 13 + 7) % 17) as f32 * 0.2 - 1.5)
            .collect();

        let expected = naive_attention(&q, &k, &v, seq_len, kv_len, head_dim, scale, false);

        let mut output = vec![0.0f32; seq_len * head_dim];
        attention_score_f32(
            &q,
            &k,
            &v,
            &mut output,
            seq_len,
            kv_len,
            head_dim,
            scale,
            false,
        );

        assert_close(&output, &expected, 5e-2, "larger_noncausal");
    }

    #[test]
    fn test_attention_score_causal_against_naive_larger() {
        let seq_len = 4;
        let kv_len = 4;
        let head_dim = 8;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i * 7 + 3) % 19) as f32 * 0.1 - 0.9)
            .collect();
        let k: Vec<f32> = (0..kv_len * head_dim)
            .map(|i| ((i * 11 + 5) % 23) as f32 * 0.1 - 1.1)
            .collect();
        let v: Vec<f32> = (0..kv_len * head_dim)
            .map(|i| ((i * 13 + 7) % 17) as f32 * 0.2 - 1.5)
            .collect();

        let expected = naive_attention(&q, &k, &v, seq_len, kv_len, head_dim, scale, true);

        let mut output = vec![0.0f32; seq_len * head_dim];
        attention_score_f32(
            &q,
            &k,
            &v,
            &mut output,
            seq_len,
            kv_len,
            head_dim,
            scale,
            true,
        );

        assert_close(&output, &expected, 5e-2, "larger_causal");
    }

    #[test]
    fn test_attention_attn_weights_sum_to_one() {
        // Verify that internal softmax produces proper probability distributions.
        // We check indirectly: for uniform V, output should equal V[0].
        let head_dim = 3;
        let kv_len = 4;
        let val = 5.0f32;
        let q = vec![0.1, 0.2, 0.3];
        let k: Vec<f32> = (0..kv_len * head_dim).map(|i| (i as f32) * 0.1).collect();
        let v = vec![val; kv_len * head_dim]; // All V rows identical.
        let mut output = vec![0.0f32; head_dim];

        attention_score_scalar(&q, &k, &v, &mut output, 1, kv_len, head_dim, 1.0, false);

        // Since all V rows are identical [val, val, val], output must be [val, val, val]
        // regardless of attention weights (as long as they sum to 1).
        for (d, &od) in output.iter().enumerate() {
            assert!(
                (od - val).abs() < 1e-6,
                "uniform V output[{d}] = {od} expected {val}",
            );
        }
    }
}
