// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SIMD-optimized Rotary Position Embeddings (RoPE).

use super::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff < tol,
            "{label}[{i}]: {x} vs {y}, diff={diff} > tol={tol}"
        );
    }
}

// ---------------------------------------------------------------------------
// test_rope_reference_identity: cos=1, sin=0 => identity transform
// ---------------------------------------------------------------------------

#[test]
fn test_rope_reference_identity() {
    let head_dim = 8;
    let seq_len = 2;
    let num_heads = 2;
    let half = head_dim / 2;

    let x: Vec<f32> = (0..seq_len * num_heads * head_dim)
        .map(|i| (i as f32) * 0.1 + 1.0)
        .collect();
    let cos_cache = vec![1.0f32; seq_len * half];
    let sin_cache = vec![0.0f32; seq_len * half];

    let result = rope_reference(&x, &cos_cache, &sin_cache, head_dim, seq_len, num_heads);
    assert_close(&result, &x, 1e-6, "rope_identity");
}

// ---------------------------------------------------------------------------
// test_rope_reference_90deg: cos=0, sin=1 => 90 degree rotation
// ---------------------------------------------------------------------------

#[test]
fn test_rope_reference_90deg() {
    let head_dim = 4;
    let seq_len = 1;
    let num_heads = 1;
    let half = head_dim / 2;

    // x = [a, b, c, d] where lo=[a,b], hi=[c,d]
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let cos_cache = vec![0.0f32; seq_len * half]; // cos=0
    let sin_cache = vec![1.0f32; seq_len * half]; // sin=1

    let result = rope_reference(&x, &cos_cache, &sin_cache, head_dim, seq_len, num_heads);

    // new_lo[i] = x_lo[i]*0 - x_hi[i]*1 = -x_hi[i]
    // new_hi[i] = x_lo[i]*1 + x_hi[i]*0 = x_lo[i]
    let expected = vec![-3.0, -4.0, 1.0, 2.0];
    assert_close(&result, &expected, 1e-6, "rope_90deg");
}

// ---------------------------------------------------------------------------
// test_rope_apply_matches_reference: random values, SIMD matches reference
// ---------------------------------------------------------------------------

#[test]
fn test_rope_apply_matches_reference() {
    let head_dim = 64;
    let seq_len = 4;
    let num_heads = 8;
    let half = head_dim / 2;
    let total = seq_len * num_heads * head_dim;

    // Generate pseudo-random data using a simple LCG
    let mut seed: u64 = 42;
    let mut next_f32 = || -> f32 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };

    let x: Vec<f32> = (0..total).map(|_| next_f32()).collect();
    let cos_cache: Vec<f32> = (0..seq_len * half).map(|_| next_f32()).collect();
    let sin_cache: Vec<f32> = (0..seq_len * half).map(|_| next_f32()).collect();

    let reference = rope_reference(&x, &cos_cache, &sin_cache, head_dim, seq_len, num_heads);

    let mut x_inplace = x;
    rope_apply(
        &mut x_inplace,
        &cos_cache,
        &sin_cache,
        head_dim,
        seq_len,
        num_heads,
    );

    assert_close(&x_inplace, &reference, 1e-5, "rope_apply_vs_reference");
}

// ---------------------------------------------------------------------------
// test_rope_various_head_dims: head_dim = 32, 64, 128
// ---------------------------------------------------------------------------

#[test]
fn test_rope_various_head_dims() {
    let seq_len = 2;
    let num_heads = 4;

    for &head_dim in &[32, 64, 128] {
        let half = head_dim / 2;
        let total = seq_len * num_heads * head_dim;

        let mut seed: u64 = head_dim as u64;
        let mut next_f32 = || -> f32 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };

        let x: Vec<f32> = (0..total).map(|_| next_f32()).collect();
        let cos_cache: Vec<f32> = (0..seq_len * half).map(|_| next_f32()).collect();
        let sin_cache: Vec<f32> = (0..seq_len * half).map(|_| next_f32()).collect();

        let reference = rope_reference(&x, &cos_cache, &sin_cache, head_dim, seq_len, num_heads);

        let mut x_inplace = x.clone();
        rope_apply(
            &mut x_inplace,
            &cos_cache,
            &sin_cache,
            head_dim,
            seq_len,
            num_heads,
        );

        assert_close(
            &x_inplace,
            &reference,
            1e-5,
            &format!("rope_head_dim_{head_dim}"),
        );
    }
}

// ---------------------------------------------------------------------------
// Constant sanity check
// ---------------------------------------------------------------------------

#[test]
fn test_rope_chunk_size() {
    assert_eq!(
        ROPE_CHUNK_SIZE, 8,
        "ROPE_CHUNK_SIZE must be 8 for AVX2 compatibility"
    );
}
