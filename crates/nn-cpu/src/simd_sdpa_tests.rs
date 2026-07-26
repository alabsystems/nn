// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SIMD scaled dot-product attention.
//!
//! Strategy: compare `sdpa` (SIMD-dispatched) against `sdpa_reference`
//! (pure scalar) and an independent naive implementation.

use crate::simd_sdpa::{sdpa, sdpa_reference, SdpaError};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: length mismatch ({} vs {})",
        a.len(),
        b.len()
    );
    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (va - vb).abs();
        assert!(
            diff <= tol,
            "{label}[{i}]: {va} vs {vb} (diff={diff}, tol={tol})"
        );
    }
}

/// Independent naive SDPA for oracle testing.
fn naive_sdpa(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    batch: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let total = batch * num_heads * seq_len * head_dim;
    let mut output = vec![0.0f32; total];
    let head_stride = seq_len * head_dim;
    let batch_stride = num_heads * head_stride;

    for b in 0..batch {
        for h in 0..num_heads {
            let base = b * batch_stride + h * head_stride;

            // Compute attention scores.
            let mut scores = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += query[base + i * head_dim + d] * key[base + j * head_dim + d];
                    }
                    scores[i * seq_len + j] = dot * scale;
                }
            }

            // Row-wise softmax.
            for i in 0..seq_len {
                let row = &mut scores[i * seq_len..(i + 1) * seq_len];
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

            // Multiply by V.
            for i in 0..seq_len {
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for j in 0..seq_len {
                        acc += scores[i * seq_len + j] * value[base + j * head_dim + d];
                    }
                    output[base + i * head_dim + d] = acc;
                }
            }
        }
    }
    output
}

fn alloc(batch: usize, num_heads: usize, seq_len: usize, head_dim: usize) -> Vec<f32> {
    vec![0.0f32; batch * num_heads * seq_len * head_dim]
}

// ---------------------------------------------------------------------------
// Basic tests
// ---------------------------------------------------------------------------

#[test]
fn test_sdpa_single_head_small() {
    let (batch, nh, sl, hd) = (1, 1, 4, 8);
    let scale = 1.0 / (hd as f32).sqrt();

    let query: Vec<f32> = (0..batch * nh * sl * hd)
        .map(|i| (i as f32 * 0.1).sin())
        .collect();
    let key: Vec<f32> = (0..batch * nh * sl * hd)
        .map(|i| (i as f32 * 0.07).cos())
        .collect();
    let value: Vec<f32> = (0..batch * nh * sl * hd)
        .map(|i| (i as f32 * 0.03 + 1.0).sin())
        .collect();

    let mut out_simd = alloc(batch, nh, sl, hd);
    let mut out_ref = alloc(batch, nh, sl, hd);

    sdpa(
        &query,
        &key,
        &value,
        &mut out_simd,
        batch,
        nh,
        sl,
        hd,
        scale,
    )
    .unwrap();
    sdpa_reference(&query, &key, &value, &mut out_ref, batch, nh, sl, hd, scale).unwrap();

    let naive = naive_sdpa(&query, &key, &value, batch, nh, sl, hd, scale);
    assert_close(&out_simd, &out_ref, 1e-5, "small_simd_vs_ref");
    assert_close(&out_simd, &naive, 1e-5, "small_simd_vs_naive");
}

#[test]
fn test_sdpa_multi_head() {
    let (batch, nh, sl, hd) = (1, 4, 8, 16);
    let scale = 1.0 / (hd as f32).sqrt();

    let query: Vec<f32> = (0..batch * nh * sl * hd)
        .map(|i| ((i * 7 + 3) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let key: Vec<f32> = (0..batch * nh * sl * hd)
        .map(|i| ((i * 13 + 11) % 200) as f32 * 0.005 - 0.5)
        .collect();
    let value: Vec<f32> = (0..batch * nh * sl * hd)
        .map(|i| ((i * 3 + 7) % 150) as f32 * 0.007 - 0.5)
        .collect();

    let mut out_simd = alloc(batch, nh, sl, hd);
    let mut out_ref = alloc(batch, nh, sl, hd);

    sdpa(
        &query,
        &key,
        &value,
        &mut out_simd,
        batch,
        nh,
        sl,
        hd,
        scale,
    )
    .unwrap();
    sdpa_reference(&query, &key, &value, &mut out_ref, batch, nh, sl, hd, scale).unwrap();

    assert_close(&out_simd, &out_ref, 1e-4, "multi_head_simd_vs_ref");
}

#[test]
fn test_sdpa_batched() {
    let (batch, nh, sl, hd) = (2, 2, 4, 8);
    let scale = 1.0 / (hd as f32).sqrt();

    let query: Vec<f32> = (0..batch * nh * sl * hd)
        .map(|i| (i as f32 * 0.05).sin())
        .collect();
    let key: Vec<f32> = (0..batch * nh * sl * hd)
        .map(|i| (i as f32 * 0.03).cos())
        .collect();
    let value: Vec<f32> = (0..batch * nh * sl * hd)
        .map(|i| (i as f32 * 0.02 + 0.5).sin())
        .collect();

    let mut out_simd = alloc(batch, nh, sl, hd);
    let mut out_ref = alloc(batch, nh, sl, hd);

    sdpa(
        &query,
        &key,
        &value,
        &mut out_simd,
        batch,
        nh,
        sl,
        hd,
        scale,
    )
    .unwrap();
    sdpa_reference(&query, &key, &value, &mut out_ref, batch, nh, sl, hd, scale).unwrap();

    let naive = naive_sdpa(&query, &key, &value, batch, nh, sl, hd, scale);
    assert_close(&out_simd, &out_ref, 1e-4, "batched_simd_vs_ref");
    assert_close(&out_simd, &naive, 1e-4, "batched_simd_vs_naive");
}

#[test]
fn test_sdpa_uniform_query_gives_uniform_output() {
    // When Q rows are identical, all attention weights are equal (1/seq_len).
    // Output should be mean of V rows.
    let (batch, nh, sl, hd) = (1, 1, 4, 4);
    let scale = 1.0 / (hd as f32).sqrt();

    // Uniform query: all rows the same.
    let q_row = vec![1.0f32; hd];
    let query: Vec<f32> = q_row.iter().copied().cycle().take(sl * hd).collect();
    let key = query.clone();
    let value: Vec<f32> = (0..sl * hd).map(|i| (i + 1) as f32).collect();

    let mut out_simd = alloc(batch, nh, sl, hd);
    sdpa(
        &query,
        &key,
        &value,
        &mut out_simd,
        batch,
        nh,
        sl,
        hd,
        scale,
    )
    .unwrap();

    // All output rows should be approximately equal (mean of V).
    let row0 = &out_simd[0..hd];
    for i in 1..sl {
        let row_i = &out_simd[i * hd..(i + 1) * hd];
        assert_close(row0, row_i, 1e-5, &format!("uniform_row_0_vs_{i}"));
    }
}

#[test]
fn test_sdpa_deterministic() {
    let (batch, nh, sl, hd) = (1, 2, 4, 8);
    let scale = 1.0 / (hd as f32).sqrt();

    let query: Vec<f32> = (0..batch * nh * sl * hd)
        .map(|i| (i as f32) * 0.3)
        .collect();
    let key = query.clone();
    let value = query.clone();

    let mut out1 = alloc(batch, nh, sl, hd);
    let mut out2 = alloc(batch, nh, sl, hd);
    sdpa(&query, &key, &value, &mut out1, batch, nh, sl, hd, scale).unwrap();
    sdpa(&query, &key, &value, &mut out2, batch, nh, sl, hd, scale).unwrap();
    assert_close(&out1, &out2, 0.0, "deterministic");
}

#[test]
fn test_sdpa_output_sums_to_value_range() {
    // softmax weights sum to 1, so output should be a convex combination of V rows.
    // Each output element should be within [min(V_col), max(V_col)].
    let (batch, nh, sl, hd) = (1, 1, 3, 4);
    let scale = 1.0 / (hd as f32).sqrt();

    let query: Vec<f32> = (0..sl * hd).map(|i| (i as f32 * 0.5).sin()).collect();
    let key = query.clone();
    let value: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, // row 0
        5.0, 6.0, 7.0, 8.0, // row 1
        9.0, 10.0, 11.0, 12.0, // row 2
    ];

    let mut output = alloc(batch, nh, sl, hd);
    sdpa(&query, &key, &value, &mut output, batch, nh, sl, hd, scale).unwrap();

    for d in 0..hd {
        let col_min = (0..sl)
            .map(|i| value[i * hd + d])
            .fold(f32::INFINITY, f32::min);
        let col_max = (0..sl)
            .map(|i| value[i * hd + d])
            .fold(f32::NEG_INFINITY, f32::max);
        for i in 0..sl {
            let v = output[i * hd + d];
            assert!(
                v >= col_min - 1e-5 && v <= col_max + 1e-5,
                "output[{i}][{d}] = {v} outside [{col_min}, {col_max}]"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

#[test]
fn test_sdpa_error_zero_dim() {
    let mut output = vec![0.0f32; 4];
    let r = sdpa(
        &[1.0; 4],
        &[1.0; 4],
        &[1.0; 4],
        &mut output,
        0,
        1,
        1,
        4,
        1.0,
    );
    assert!(matches!(r, Err(SdpaError::ZeroDim { param: "batch" })));
}

#[test]
fn test_sdpa_error_wrong_query_len() {
    let mut output = vec![0.0f32; 8];
    let r = sdpa(
        &[1.0; 4],
        &[1.0; 8],
        &[1.0; 8],
        &mut output,
        1,
        1,
        2,
        4,
        1.0,
    );
    assert!(matches!(
        r,
        Err(SdpaError::InvalidLength { name: "query", .. })
    ));
}
