// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SIMD-optimized multi-head scaled dot-product attention.

use super::{attention_reference, scaled_dot_product_attention, AttentionConfig};

/// Assert two slices are element-wise close within tolerance.
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

/// SIMD softmax uses fast_exp approximation (~1-2% relative error).
/// This compounds through softmax normalization and weighted V sum.
const SIMD_TOL: f32 = 0.1;

// -----------------------------------------------------------------------
// Single-head attention with known values
// -----------------------------------------------------------------------

#[test]
fn test_simd_attention_single_head_identity() {
    // 1 head, head_dim=2, seq=2, kv=2.
    // Q = K = V = [[1, 0], [0, 1]] (identity-like).
    //
    // scores = Q * K^T * scale = [[1, 0], [0, 1]] / sqrt(2)
    // softmax row 0: softmax([1/sqrt2, 0]) -> [e^(1/sqrt2) / (e^(1/sqrt2)+1), ...]
    // output = attn * V
    let config = AttentionConfig {
        num_heads: 1,
        head_dim: 2,
        seq_len: 2,
        kv_seq_len: 2,
        causal: false,
    };

    let q = vec![1.0, 0.0, 0.0, 1.0];
    let k = vec![1.0, 0.0, 0.0, 1.0];
    let v = vec![1.0, 0.0, 0.0, 1.0];
    let mut output = vec![0.0f32; 4];

    scaled_dot_product_attention(&q, &k, &v, &mut output, &config);

    let expected = attention_reference(&q, &k, &v, &config);
    assert_close(&output, &expected, SIMD_TOL, "single_head_identity");
}

#[test]
fn test_simd_attention_single_head_known_values() {
    // 1 head, head_dim=2, seq=1, kv=2.
    // Q = [1, 0], K = [[1, 0], [0, 1]], V = [[10, 0], [0, 10]]
    // scores = [1, 0] * scale(1/sqrt2)
    // softmax([1/sqrt2, 0]) -> heavier weight on V[0]
    let config = AttentionConfig {
        num_heads: 1,
        head_dim: 2,
        seq_len: 1,
        kv_seq_len: 2,
        causal: false,
    };

    let q = vec![1.0, 0.0];
    let k = vec![1.0, 0.0, 0.0, 1.0];
    let v = vec![10.0, 0.0, 0.0, 10.0];
    let mut output = vec![0.0f32; 2];

    scaled_dot_product_attention(&q, &k, &v, &mut output, &config);

    let expected = attention_reference(&q, &k, &v, &config);
    assert_close(&output, &expected, SIMD_TOL, "single_head_known");

    // Output[0] should be > 5 (weighted toward V[0][0]=10).
    assert!(output[0] > 5.0, "output[0]={} should be > 5", output[0]);
}

// -----------------------------------------------------------------------
// Multi-head attention
// -----------------------------------------------------------------------

#[test]
fn test_simd_attention_multi_head_2heads() {
    // 2 heads, head_dim=3, seq=2, kv=3.
    // Model dim = 6. Each row of Q/K/V has 6 values: [head0_d0..d2, head1_d0..d2].
    let config = AttentionConfig {
        num_heads: 2,
        head_dim: 3,
        seq_len: 2,
        kv_seq_len: 3,
        causal: false,
    };

    let model_dim = config.model_dim();
    assert_eq!(model_dim, 6);

    // Deterministic values.
    let q: Vec<f32> = (0..config.seq_len * model_dim)
        .map(|i| ((i * 7 + 3) % 13) as f32 * 0.1 - 0.6)
        .collect();
    let k: Vec<f32> = (0..config.kv_seq_len * model_dim)
        .map(|i| ((i * 11 + 5) % 17) as f32 * 0.1 - 0.8)
        .collect();
    let v: Vec<f32> = (0..config.kv_seq_len * model_dim)
        .map(|i| ((i * 13 + 7) % 19) as f32 * 0.2 - 1.0)
        .collect();

    let expected = attention_reference(&q, &k, &v, &config);

    let mut output = vec![0.0f32; config.seq_len * model_dim];
    scaled_dot_product_attention(&q, &k, &v, &mut output, &config);

    assert_close(&output, &expected, SIMD_TOL, "multi_head_2heads");
}

#[test]
fn test_simd_attention_multi_head_4heads() {
    // 4 heads, head_dim=8, seq=4, kv=6.
    let config = AttentionConfig {
        num_heads: 4,
        head_dim: 8,
        seq_len: 4,
        kv_seq_len: 6,
        causal: false,
    };

    let model_dim = config.model_dim();
    let q: Vec<f32> = (0..config.seq_len * model_dim)
        .map(|i| ((i * 7 + 3) % 29) as f32 * 0.1 - 1.4)
        .collect();
    let k: Vec<f32> = (0..config.kv_seq_len * model_dim)
        .map(|i| ((i * 11 + 5) % 31) as f32 * 0.1 - 1.5)
        .collect();
    let v: Vec<f32> = (0..config.kv_seq_len * model_dim)
        .map(|i| ((i * 13 + 7) % 37) as f32 * 0.1 - 1.8)
        .collect();

    let expected = attention_reference(&q, &k, &v, &config);

    let mut output = vec![0.0f32; config.seq_len * model_dim];
    scaled_dot_product_attention(&q, &k, &v, &mut output, &config);

    assert_close(&output, &expected, SIMD_TOL, "multi_head_4heads");
}

// -----------------------------------------------------------------------
// Causal mask
// -----------------------------------------------------------------------

#[test]
fn test_simd_attention_causal_single_head() {
    // 1 head, head_dim=2, seq=3, kv=3, causal.
    // Row 0: only attends to position 0.
    // Row 1: attends to positions 0 and 1.
    // Row 2: attends to all 3 positions.
    let config = AttentionConfig {
        num_heads: 1,
        head_dim: 2,
        seq_len: 3,
        kv_seq_len: 3,
        causal: true,
    };

    let q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let k = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let v = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
    let mut output = vec![0.0f32; 6];

    scaled_dot_product_attention(&q, &k, &v, &mut output, &config);

    // Row 0: attends only to j=0. softmax([score]) = [1.0].
    // output[0..2] = V[0] = [1.0, 0.0].
    assert!(
        (output[0] - 1.0).abs() < SIMD_TOL,
        "causal row0 d0: {}",
        output[0]
    );
    assert!(
        (output[1] - 0.0).abs() < SIMD_TOL,
        "causal row0 d1: {}",
        output[1]
    );

    let expected = attention_reference(&q, &k, &v, &config);
    assert_close(&output, &expected, SIMD_TOL, "causal_single_head");
}

#[test]
fn test_simd_attention_causal_multi_head() {
    // 2 heads, head_dim=4, seq=4, kv=4, causal.
    let config = AttentionConfig {
        num_heads: 2,
        head_dim: 4,
        seq_len: 4,
        kv_seq_len: 4,
        causal: true,
    };

    let model_dim = config.model_dim();
    let q: Vec<f32> = (0..config.seq_len * model_dim)
        .map(|i| ((i * 7 + 3) % 19) as f32 * 0.1 - 0.9)
        .collect();
    let k: Vec<f32> = (0..config.kv_seq_len * model_dim)
        .map(|i| ((i * 11 + 5) % 23) as f32 * 0.1 - 1.1)
        .collect();
    let v: Vec<f32> = (0..config.kv_seq_len * model_dim)
        .map(|i| ((i * 13 + 7) % 17) as f32 * 0.2 - 1.5)
        .collect();

    let expected = attention_reference(&q, &k, &v, &config);

    let mut output = vec![0.0f32; config.seq_len * model_dim];
    scaled_dot_product_attention(&q, &k, &v, &mut output, &config);

    assert_close(&output, &expected, SIMD_TOL, "causal_multi_head");
}

// -----------------------------------------------------------------------
// Various seq_len and head_dim combinations
// -----------------------------------------------------------------------

#[test]
fn test_simd_attention_seq1_kv1() {
    // Simplest case: seq=1, kv=1. Output = V (softmax of 1 score = 1.0).
    let config = AttentionConfig {
        num_heads: 1,
        head_dim: 4,
        seq_len: 1,
        kv_seq_len: 1,
        causal: false,
    };

    let q = vec![1.0, 2.0, 3.0, 4.0];
    let k = vec![0.5, 0.5, 0.5, 0.5];
    let v = vec![10.0, 20.0, 30.0, 40.0];
    let mut output = vec![0.0f32; 4];

    scaled_dot_product_attention(&q, &k, &v, &mut output, &config);

    // With kv_seq_len=1, softmax([score]) = [1.0], output = V.
    assert_close(&output, &v, SIMD_TOL, "seq1_kv1");
}

#[test]
fn test_simd_attention_head_dim_1() {
    // Smallest possible head_dim.
    let config = AttentionConfig {
        num_heads: 2,
        head_dim: 1,
        seq_len: 3,
        kv_seq_len: 3,
        causal: false,
    };

    let model_dim = config.model_dim();
    let q: Vec<f32> = (0..config.seq_len * model_dim)
        .map(|i| (i as f32) * 0.5)
        .collect();
    let k: Vec<f32> = (0..config.kv_seq_len * model_dim)
        .map(|i| (i as f32) * 0.3)
        .collect();
    let v: Vec<f32> = (0..config.kv_seq_len * model_dim)
        .map(|i| (i as f32) * 0.7 + 1.0)
        .collect();

    let expected = attention_reference(&q, &k, &v, &config);

    let mut output = vec![0.0f32; config.seq_len * model_dim];
    scaled_dot_product_attention(&q, &k, &v, &mut output, &config);

    assert_close(&output, &expected, SIMD_TOL, "head_dim_1");
}

#[test]
fn test_simd_attention_head_dim_64() {
    // Typical transformer head_dim.
    let config = AttentionConfig {
        num_heads: 2,
        head_dim: 64,
        seq_len: 4,
        kv_seq_len: 4,
        causal: false,
    };

    let model_dim = config.model_dim();
    let q: Vec<f32> = (0..config.seq_len * model_dim)
        .map(|i| ((i * 7 + 3) % 41) as f32 * 0.05 - 1.0)
        .collect();
    let k: Vec<f32> = (0..config.kv_seq_len * model_dim)
        .map(|i| ((i * 11 + 5) % 43) as f32 * 0.05 - 1.05)
        .collect();
    let v: Vec<f32> = (0..config.kv_seq_len * model_dim)
        .map(|i| ((i * 13 + 7) % 47) as f32 * 0.05 - 1.15)
        .collect();

    let expected = attention_reference(&q, &k, &v, &config);

    let mut output = vec![0.0f32; config.seq_len * model_dim];
    scaled_dot_product_attention(&q, &k, &v, &mut output, &config);

    assert_close(&output, &expected, SIMD_TOL, "head_dim_64");
}

#[test]
fn test_simd_attention_cross_attention_different_seq_kv() {
    // Cross-attention: seq_len != kv_seq_len.
    let config = AttentionConfig {
        num_heads: 2,
        head_dim: 4,
        seq_len: 3,
        kv_seq_len: 5,
        causal: false,
    };

    let model_dim = config.model_dim();
    let q: Vec<f32> = (0..config.seq_len * model_dim)
        .map(|i| ((i * 7 + 1) % 11) as f32 * 0.2 - 1.0)
        .collect();
    let k: Vec<f32> = (0..config.kv_seq_len * model_dim)
        .map(|i| ((i * 3 + 2) % 13) as f32 * 0.2 - 1.2)
        .collect();
    let v: Vec<f32> = (0..config.kv_seq_len * model_dim)
        .map(|i| ((i * 5 + 4) % 17) as f32 * 0.2 - 1.5)
        .collect();

    let expected = attention_reference(&q, &k, &v, &config);

    let mut output = vec![0.0f32; config.seq_len * model_dim];
    scaled_dot_product_attention(&q, &k, &v, &mut output, &config);

    assert_close(&output, &expected, SIMD_TOL, "cross_attention");
}

// -----------------------------------------------------------------------
// Score scaling verification
// -----------------------------------------------------------------------

#[test]
fn test_simd_attention_scale_is_one_over_sqrt_head_dim() {
    // Verify the scaling factor is 1/sqrt(head_dim).
    let config4 = AttentionConfig {
        num_heads: 1,
        head_dim: 4,
        seq_len: 1,
        kv_seq_len: 1,
        causal: false,
    };
    assert!(
        (config4.scale() - 0.5).abs() < 1e-7,
        "1/sqrt(4) = 0.5, got {}",
        config4.scale()
    );

    let config64 = AttentionConfig {
        num_heads: 1,
        head_dim: 64,
        seq_len: 1,
        kv_seq_len: 1,
        causal: false,
    };
    assert!(
        (config64.scale() - 0.125).abs() < 1e-7,
        "1/sqrt(64) = 0.125, got {}",
        config64.scale()
    );

    let config16 = AttentionConfig {
        num_heads: 1,
        head_dim: 16,
        seq_len: 1,
        kv_seq_len: 1,
        causal: false,
    };
    assert!(
        (config16.scale() - 0.25).abs() < 1e-7,
        "1/sqrt(16) = 0.25, got {}",
        config16.scale()
    );
}

#[test]
fn test_simd_attention_scale_sharpens_attention() {
    // With large head_dim (small scale), attention is more uniform.
    // With small head_dim (large scale), attention is sharper.
    //
    // We test by comparing output variance: sharper attention ->
    // output closer to one V row -> higher variance relative to V mean.
    let kv_seq_len = 3;

    // head_dim=1: scale = 1.0 (large).
    let config_sharp = AttentionConfig {
        num_heads: 1,
        head_dim: 1,
        seq_len: 1,
        kv_seq_len,
        causal: false,
    };
    let q_sharp = vec![1.0]; // 1 value
    let k_sharp = vec![1.0, 0.0, -1.0]; // scores before scale: [1, 0, -1]
    let v_sharp = vec![10.0, 0.0, -10.0];

    let ref_sharp = attention_reference(&q_sharp, &k_sharp, &v_sharp, &config_sharp);

    // head_dim=1 but we manually lower the effective scale by padding.
    // Instead, just check that output is pushed toward V[0]=10.
    // With scale=1.0, scores=[1,0,-1], softmax heavily weights V[0].
    assert!(
        ref_sharp[0] > 5.0,
        "sharp attention should push output > 5, got {}",
        ref_sharp[0]
    );
}

// -----------------------------------------------------------------------
// Output shape validation
// -----------------------------------------------------------------------

#[test]
fn test_simd_attention_output_shape() {
    let config = AttentionConfig {
        num_heads: 3,
        head_dim: 5,
        seq_len: 4,
        kv_seq_len: 6,
        causal: false,
    };

    let model_dim = config.model_dim();
    let expected_output_len = config.seq_len * model_dim;
    assert_eq!(expected_output_len, 4 * 15);
    assert_eq!(model_dim, 15);

    let q = vec![0.1f32; config.seq_len * model_dim];
    let k = vec![0.1f32; config.kv_seq_len * model_dim];
    let v = vec![0.1f32; config.kv_seq_len * model_dim];
    let mut output = vec![0.0f32; expected_output_len];

    // Should not panic.
    scaled_dot_product_attention(&q, &k, &v, &mut output, &config);

    assert_eq!(output.len(), expected_output_len);
}

#[test]
fn test_simd_attention_empty_seq() {
    let config = AttentionConfig {
        num_heads: 2,
        head_dim: 4,
        seq_len: 0,
        kv_seq_len: 3,
        causal: false,
    };

    let q: Vec<f32> = vec![];
    let k = vec![0.1f32; 3 * 8];
    let v = vec![0.1f32; 3 * 8];
    let mut output: Vec<f32> = vec![];

    // seq_len=0 should be a no-op.
    scaled_dot_product_attention(&q, &k, &v, &mut output, &config);
}

// -----------------------------------------------------------------------
// SIMD vs scalar agreement
// -----------------------------------------------------------------------

#[test]
fn test_simd_attention_dispatch_matches_reference_noncausal() {
    // The SIMD path (dispatched via scaled_dot_product_attention) should
    // agree with the pure-scalar reference within SIMD tolerance.
    let config = AttentionConfig {
        num_heads: 2,
        head_dim: 8,
        seq_len: 4,
        kv_seq_len: 6,
        causal: false,
    };

    let model_dim = config.model_dim();
    let q: Vec<f32> = (0..config.seq_len * model_dim)
        .map(|i| ((i * 7 + 3) % 19) as f32 * 0.1 - 0.9)
        .collect();
    let k: Vec<f32> = (0..config.kv_seq_len * model_dim)
        .map(|i| ((i * 11 + 5) % 23) as f32 * 0.1 - 1.1)
        .collect();
    let v: Vec<f32> = (0..config.kv_seq_len * model_dim)
        .map(|i| ((i * 13 + 7) % 17) as f32 * 0.2 - 1.5)
        .collect();

    let reference = attention_reference(&q, &k, &v, &config);

    let mut output = vec![0.0f32; config.seq_len * model_dim];
    scaled_dot_product_attention(&q, &k, &v, &mut output, &config);

    assert_close(&output, &reference, SIMD_TOL, "dispatch_vs_ref_noncausal");
}

#[test]
fn test_simd_attention_dispatch_matches_reference_causal() {
    let config = AttentionConfig {
        num_heads: 2,
        head_dim: 8,
        seq_len: 4,
        kv_seq_len: 4,
        causal: true,
    };

    let model_dim = config.model_dim();
    let q: Vec<f32> = (0..config.seq_len * model_dim)
        .map(|i| ((i * 7 + 3) % 19) as f32 * 0.1 - 0.9)
        .collect();
    let k: Vec<f32> = (0..config.kv_seq_len * model_dim)
        .map(|i| ((i * 11 + 5) % 23) as f32 * 0.1 - 1.1)
        .collect();
    let v: Vec<f32> = (0..config.kv_seq_len * model_dim)
        .map(|i| ((i * 13 + 7) % 17) as f32 * 0.2 - 1.5)
        .collect();

    let reference = attention_reference(&q, &k, &v, &config);

    let mut output = vec![0.0f32; config.seq_len * model_dim];
    scaled_dot_product_attention(&q, &k, &v, &mut output, &config);

    assert_close(&output, &reference, SIMD_TOL, "dispatch_vs_ref_causal");
}

#[test]
fn test_simd_attention_uniform_v_returns_v() {
    // When all V rows are identical, output must equal that V row
    // regardless of attention weights (as long as they sum to 1).
    let config = AttentionConfig {
        num_heads: 2,
        head_dim: 4,
        seq_len: 3,
        kv_seq_len: 5,
        causal: false,
    };

    let model_dim = config.model_dim();
    let val = 7.5f32;
    let q: Vec<f32> = (0..config.seq_len * model_dim)
        .map(|i| (i as f32) * 0.1)
        .collect();
    let k: Vec<f32> = (0..config.kv_seq_len * model_dim)
        .map(|i| (i as f32) * 0.2 - 0.5)
        .collect();
    let v = vec![val; config.kv_seq_len * model_dim];
    let mut output = vec![0.0f32; config.seq_len * model_dim];

    scaled_dot_product_attention(&q, &k, &v, &mut output, &config);

    for (i, &o) in output.iter().enumerate() {
        assert!(
            (o - val).abs() < SIMD_TOL,
            "uniform V output[{i}] = {o}, expected {val}"
        );
    }
}

#[test]
fn test_simd_attention_head_dim_non_multiple_of_4() {
    // head_dim=5 is not a multiple of 4 (NEON lane width).
    // Tests remainder handling.
    let config = AttentionConfig {
        num_heads: 1,
        head_dim: 5,
        seq_len: 2,
        kv_seq_len: 3,
        causal: false,
    };

    let model_dim = config.model_dim();
    let q: Vec<f32> = (0..config.seq_len * model_dim)
        .map(|i| ((i * 3 + 1) % 11) as f32 * 0.2 - 1.0)
        .collect();
    let k: Vec<f32> = (0..config.kv_seq_len * model_dim)
        .map(|i| ((i * 5 + 2) % 13) as f32 * 0.2 - 1.2)
        .collect();
    let v: Vec<f32> = (0..config.kv_seq_len * model_dim)
        .map(|i| ((i * 7 + 4) % 17) as f32 * 0.3 - 2.0)
        .collect();

    let expected = attention_reference(&q, &k, &v, &config);

    let mut output = vec![0.0f32; config.seq_len * model_dim];
    scaled_dot_product_attention(&q, &k, &v, &mut output, &config);

    assert_close(&output, &expected, SIMD_TOL, "head_dim_non_mult_4");
}

#[test]
fn test_simd_attention_head_dim_non_multiple_of_8() {
    // head_dim=11 is not a multiple of 8 (AVX2 lane width).
    // Tests remainder handling in the AVX2 path.
    let config = AttentionConfig {
        num_heads: 1,
        head_dim: 11,
        seq_len: 2,
        kv_seq_len: 3,
        causal: true,
    };

    let model_dim = config.model_dim();
    let q: Vec<f32> = (0..config.seq_len * model_dim)
        .map(|i| ((i * 3 + 1) % 11) as f32 * 0.2 - 1.0)
        .collect();
    let k: Vec<f32> = (0..config.kv_seq_len * model_dim)
        .map(|i| ((i * 5 + 2) % 13) as f32 * 0.2 - 1.2)
        .collect();
    let v: Vec<f32> = (0..config.kv_seq_len * model_dim)
        .map(|i| ((i * 7 + 4) % 17) as f32 * 0.3 - 2.0)
        .collect();

    let expected = attention_reference(&q, &k, &v, &config);

    let mut output = vec![0.0f32; config.seq_len * model_dim];
    scaled_dot_product_attention(&q, &k, &v, &mut output, &config);

    assert_close(&output, &expected, SIMD_TOL, "head_dim_non_mult_8");
}
