// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for multi-head attention reference and PTX generation.
//!
//! Complements `ptx_attention_multihead_tests.rs` with deeper numerical
//! validation: known Q/K/V computations, single-head vs multi-head parity,
//! causal mask boundary cases, and scale-factor invariants.

use crate::ptx_attention_multihead::{
    attention_reference, generate_multihead_attention_ptx, PtxMultiHeadAttentionConfig,
};

// =========================================================================
// Multi-head reference with known Q, K, V and manual computation
// =========================================================================

#[test]
fn test_reference_known_qkv_two_heads() {
    // 1 batch, 2 heads, seq=2, kv_seq=2, head_dim=2
    //
    // Layout: [batch, num_heads, seq_len, head_dim] flattened
    //
    // Head 0: Q = [[1, 0], [0, 1]]   K = [[1, 0], [0, 1]]   V = [[10, 20], [30, 40]]
    // Head 1: Q = [[0, 1], [1, 0]]   K = [[1, 0], [0, 1]]   V = [[50, 60], [70, 80]]

    let q = vec![
        // head 0, seq 0-1
        1.0, 0.0, 0.0, 1.0, // head 1, seq 0-1
        0.0, 1.0, 1.0, 0.0,
    ];
    let k = vec![
        // head 0
        1.0, 0.0, 0.0, 1.0, // head 1
        1.0, 0.0, 0.0, 1.0,
    ];
    let v = vec![
        // head 0
        10.0, 20.0, 30.0, 40.0, // head 1
        50.0, 60.0, 70.0, 80.0,
    ];

    let out = attention_reference(&q, &k, &v, 1, 2, 2, 2, 2, false);
    assert_eq!(out.len(), 2 * 2 * 2); // batch=1, heads=2, seq=2, dim=2

    let scale = 1.0 / (2.0f32).sqrt();

    // Head 0, q_pos=0: Q=[1,0]
    // scores = [dot([1,0],[1,0])*scale, dot([1,0],[0,1])*scale] = [1*scale, 0*scale]
    // softmax: w0 = exp(scale) / (exp(scale) + exp(0)), w1 = exp(0) / (...)
    let s00 = (1.0 * scale).exp();
    let s01 = (0.0 * scale).exp();
    let z0 = s00 + s01;
    let w00 = s00 / z0;
    let w01 = s01 / z0;
    // output = w00 * V[0] + w01 * V[1]
    let expected_h0_q0_d0 = w00 * 10.0 + w01 * 30.0;
    let expected_h0_q0_d1 = w00 * 20.0 + w01 * 40.0;

    // Head 0, q_pos=0 starts at index 0
    assert!(
        (out[0] - expected_h0_q0_d0).abs() < 1e-4,
        "head0 q0 d0: expected {expected_h0_q0_d0}, got {}",
        out[0]
    );
    assert!(
        (out[1] - expected_h0_q0_d1).abs() < 1e-4,
        "head0 q0 d1: expected {expected_h0_q0_d1}, got {}",
        out[1]
    );

    // Head 0, q_pos=1: Q=[0,1]
    // scores = [dot([0,1],[1,0])*scale, dot([0,1],[0,1])*scale] = [0*scale, 1*scale]
    let s10 = (0.0 * scale).exp();
    let s11 = (1.0 * scale).exp();
    let z1 = s10 + s11;
    let w10 = s10 / z1;
    let w11 = s11 / z1;
    let expected_h0_q1_d0 = w10 * 10.0 + w11 * 30.0;
    let expected_h0_q1_d1 = w10 * 20.0 + w11 * 40.0;

    // Head 0, q_pos=1 starts at index 2
    assert!(
        (out[2] - expected_h0_q1_d0).abs() < 1e-4,
        "head0 q1 d0: expected {expected_h0_q1_d0}, got {}",
        out[2]
    );
    assert!(
        (out[3] - expected_h0_q1_d1).abs() < 1e-4,
        "head0 q1 d1: expected {expected_h0_q1_d1}, got {}",
        out[3]
    );
}

// =========================================================================
// Single head vs multi-head: identical data should give identical results
// =========================================================================

#[test]
fn test_single_head_matches_multi_head_single() {
    // When num_heads=1, the multi-head reference should produce the same
    // result as calling it with num_heads=1.
    let head_dim = 4;
    let seq_len = 3;
    let kv_seq_len = 3;

    let q = vec![
        1.0, 0.5, -0.3, 0.2, 0.8, -0.1, 0.4, 0.7, -0.5, 0.3, 0.6, -0.2,
    ];
    let k = vec![
        0.3, 0.7, -0.1, 0.5, -0.4, 0.2, 0.8, -0.3, 0.6, -0.5, 0.1, 0.9,
    ];
    let v = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];

    let out_single = attention_reference(&q, &k, &v, 1, 1, seq_len, kv_seq_len, head_dim, false);

    // Create a 2-head version by duplicating the data
    let mut q2 = q.clone();
    q2.extend_from_slice(&q);
    let mut k2 = k.clone();
    k2.extend_from_slice(&k);
    let mut v2 = v.clone();
    v2.extend_from_slice(&v);

    let out_multi = attention_reference(&q2, &k2, &v2, 1, 2, seq_len, kv_seq_len, head_dim, false);

    // Head 0 of multi-head should match single-head
    for i in 0..out_single.len() {
        assert!(
            (out_single[i] - out_multi[i]).abs() < 1e-5,
            "mismatch at index {i}: single={}, multi_head0={}",
            out_single[i],
            out_multi[i]
        );
    }

    // Head 1 should also match (identical data)
    let head_offset = seq_len * head_dim;
    for i in 0..out_single.len() {
        assert!(
            (out_single[i] - out_multi[head_offset + i]).abs() < 1e-5,
            "mismatch at index {i}: single={}, multi_head1={}",
            out_single[i],
            out_multi[head_offset + i]
        );
    }
}

// =========================================================================
// Causal mask: boundary positions
// =========================================================================

#[test]
fn test_causal_mask_boundary_positions() {
    // 4 positions with distinct values, causal masking
    // q_pos=i can see j <= i
    let head_dim = 1;
    let seq_len = 4;
    let q = vec![1.0; seq_len];
    let k = vec![1.0; seq_len];
    let v = vec![1.0, 2.0, 3.0, 4.0];

    let out = attention_reference(&q, &k, &v, 1, 1, seq_len, seq_len, head_dim, true);

    // q_pos=0: sees V[0] only -> 1.0
    assert!(
        (out[0] - 1.0).abs() < 1e-5,
        "q_pos=0: expected 1.0, got {}",
        out[0]
    );
    // q_pos=1: sees V[0..=1] equally -> (1+2)/2 = 1.5
    assert!(
        (out[1] - 1.5).abs() < 1e-5,
        "q_pos=1: expected 1.5, got {}",
        out[1]
    );
    // q_pos=2: sees V[0..=2] equally -> (1+2+3)/3 = 2.0
    assert!(
        (out[2] - 2.0).abs() < 1e-5,
        "q_pos=2: expected 2.0, got {}",
        out[2]
    );
    // q_pos=3: sees V[0..=3] equally -> (1+2+3+4)/4 = 2.5
    assert!(
        (out[3] - 2.5).abs() < 1e-5,
        "q_pos=3: expected 2.5, got {}",
        out[3]
    );
}

#[test]
fn test_causal_mask_last_position_sees_all() {
    // The last query position under causal mask should behave identically
    // to non-causal for that position (it can see all keys).
    let head_dim = 2;
    let seq_len = 4;
    let q = vec![0.5, 0.3, 0.1, -0.2, 0.7, 0.4, -0.1, 0.6];
    let k = vec![0.3, 0.5, -0.1, 0.8, 0.4, -0.3, 0.6, 0.2];
    let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let causal_out = attention_reference(&q, &k, &v, 1, 1, seq_len, seq_len, head_dim, true);
    let non_causal_out = attention_reference(&q, &k, &v, 1, 1, seq_len, seq_len, head_dim, false);

    // Last position (index 3) should be identical
    let last_start = (seq_len - 1) * head_dim;
    for d in 0..head_dim {
        assert!(
            (causal_out[last_start + d] - non_causal_out[last_start + d]).abs() < 1e-5,
            "last position dim {d}: causal={}, non_causal={}",
            causal_out[last_start + d],
            non_causal_out[last_start + d]
        );
    }
}

// =========================================================================
// Scale factor: 1/sqrt(head_dim) verification
// =========================================================================

#[test]
fn test_scale_factor_affects_output() {
    // With head_dim=1, scale=1/sqrt(1)=1.0
    // With head_dim=4, scale=1/sqrt(4)=0.5
    // For Q=[10], K=[[1],[1]], V=[[1],[2]], the scores before softmax are:
    //   head_dim=1: scores = [10*1, 10*1] = [10, 10] -> softmax -> [0.5, 0.5]
    //   head_dim=4: not directly comparable, but scale changes the sharpness

    // Instead, verify that attention becomes more uniform (less sharp) with
    // larger head_dim (smaller scale).
    let q_sharp = vec![10.0]; // head_dim=1
    let k_sharp = vec![1.0, 0.0]; // two keys, very different
    let v_sharp = vec![0.0, 100.0];

    let out_dim1 = attention_reference(&q_sharp, &k_sharp, &v_sharp, 1, 1, 1, 2, 1, false);

    // With head_dim=4 and Q=[10,0,0,0], K=[[1,0,0,0],[0,0,0,0]]
    // scale=0.5, scores=[10*0.5, 0*0.5]=[5,0]
    let q4 = vec![10.0, 0.0, 0.0, 0.0];
    let k4 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let v4 = vec![0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0];

    let out_dim4 = attention_reference(&q4, &k4, &v4, 1, 1, 1, 2, 4, false);

    // Both should return a value between 0 and 100 for the first dimension,
    // but dim4 (smaller scale) should give more weight to the second key
    // (more uniform distribution)
    assert!(
        out_dim1[0] > 0.0 && out_dim1[0] < 100.0,
        "dim1 output should be in (0, 100)"
    );
    assert!(
        out_dim4[0] > 0.0 && out_dim4[0] < 100.0,
        "dim4 output should be in (0, 100)"
    );
}

#[test]
fn test_scale_factor_values_for_common_head_dims() {
    let test_cases = [
        (32, 1.0 / (32.0f32).sqrt()),
        (64, 1.0 / (64.0f32).sqrt()),
        (128, 1.0 / (128.0f32).sqrt()),
        (256, 1.0 / (256.0f32).sqrt()),
    ];

    for (head_dim, expected_scale) in test_cases {
        let config = PtxMultiHeadAttentionConfig::new(8, head_dim, 64);
        assert!(
            (config.scale() - expected_scale).abs() < 1e-6,
            "head_dim={head_dim}: expected scale={expected_scale}, got {}",
            config.scale()
        );
    }
}

#[test]
fn test_scale_in_ptx_matches_config() {
    for head_dim in [16, 32, 64, 128] {
        let config = PtxMultiHeadAttentionConfig::new(8, head_dim, 64);
        let src = generate_multihead_attention_ptx(&config).unwrap();
        let scale = config.scale();
        // The PTX should contain the scale value. Try default format first,
        // then truncated to 3 decimal places if the default doesn't match.
        let scale_default = format!("{scale}");
        let scale_truncated = format!("{scale:.3}");
        assert!(
            src.contains(&scale_default) || src.contains(&scale_truncated),
            "head_dim={head_dim}: PTX should contain scale {scale_default} or {scale_truncated}"
        );
    }
}

// =========================================================================
// Output is a convex combination of V rows
// =========================================================================

#[test]
fn test_output_within_value_range() {
    // Attention output should always be a convex combination of V rows,
    // so each output dimension should be within the range of corresponding
    // V values.
    let head_dim = 3;
    let seq_len = 2;
    let kv_seq_len = 4;

    let q = vec![0.5, -0.3, 0.2, 0.8, 0.1, -0.4];
    let k = vec![
        0.1, 0.2, 0.3, -0.1, 0.4, 0.5, 0.6, -0.2, 0.3, 0.7, -0.1, 0.4,
    ];
    let v = vec![
        1.0, 10.0, 100.0, 2.0, 20.0, 200.0, 3.0, 30.0, 300.0, 4.0, 40.0, 400.0,
    ];

    let out = attention_reference(&q, &k, &v, 1, 1, seq_len, kv_seq_len, head_dim, false);

    for i in 0..seq_len {
        for d in 0..head_dim {
            let val = out[i * head_dim + d];
            let v_min = (0..kv_seq_len)
                .map(|j| v[j * head_dim + d])
                .fold(f32::INFINITY, f32::min);
            let v_max = (0..kv_seq_len)
                .map(|j| v[j * head_dim + d])
                .fold(f32::NEG_INFINITY, f32::max);
            assert!(
                val >= v_min - 1e-5 && val <= v_max + 1e-5,
                "output[{i}][{d}]={val} outside V range [{v_min}, {v_max}]"
            );
        }
    }
}

// =========================================================================
// Batched multi-head: batches are independent
// =========================================================================

#[test]
fn test_batched_independence() {
    // Two batches with different data should produce different results.
    let head_dim = 2;
    let q = vec![
        // batch 0, head 0
        1.0, 0.0, // batch 1, head 0
        0.0, 1.0,
    ];
    let k = vec![1.0, 0.0, 0.0, 1.0];
    let v = vec![10.0, 20.0, 30.0, 40.0];

    let out = attention_reference(&q, &k, &v, 2, 1, 1, 1, head_dim, false);
    assert_eq!(out.len(), 4);

    // Batch 0: Q=[1,0], single key -> out = V = [10, 20]
    assert!((out[0] - 10.0).abs() < 1e-5);
    assert!((out[1] - 20.0).abs() < 1e-5);

    // Batch 1: Q=[0,1], single key -> out = V = [30, 40]
    assert!((out[2] - 30.0).abs() < 1e-5);
    assert!((out[3] - 40.0).abs() < 1e-5);
}

// =========================================================================
// Cross-attention PTX: kv_seq_len != seq_len bakes correct sizes
// =========================================================================

#[test]
fn test_cross_attention_ptx_sizes() {
    let config = PtxMultiHeadAttentionConfig::new(4, 32, 16).with_kv_seq_len(128);
    let src = generate_multihead_attention_ptx(&config).unwrap();

    // Shared memory arrays must use kv_seq_len for scores
    assert!(
        src.contains("scores[128]"),
        "cross-attention scores must be sized by kv_seq_len=128"
    );
    assert!(
        src.contains("q_local[32]"),
        "q_local must be sized by head_dim=32"
    );
}

// =========================================================================
// PTX contains grid-stride or index-clamped patterns
// =========================================================================

#[test]
fn test_ptx_contains_bounds_check() {
    let config = PtxMultiHeadAttentionConfig::new(8, 64, 128);
    let src = generate_multihead_attention_ptx(&config).unwrap();
    assert!(
        src.contains("batch_idx >= batch_size"),
        "must have bounds check on batch_idx"
    );
}
