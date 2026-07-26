// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`super::demucs_transformer_helpers`].

use super::*;

// -- transpose_ct_to_tc -------------------------------------------------------

#[test]
fn test_transpose_ct_to_tc_identity_1x1() {
    let data = vec![42.0];
    let out = transpose_ct_to_tc(&data, 1, 1);
    assert_eq!(out, vec![42.0]);
}

#[test]
fn test_transpose_ct_to_tc_2x3() {
    // [C=2, T=3]: row-major [c0t0, c0t1, c0t2, c1t0, c1t1, c1t2]
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let out = transpose_ct_to_tc(&data, 2, 3);
    // [T=3, C=2]: [t0c0, t0c1, t1c0, t1c1, t2c0, t2c1]
    assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

// -- transpose_tc_to_ct -------------------------------------------------------

#[test]
fn test_transpose_tc_to_ct_2x3() {
    // [T=3, C=2]: [t0c0, t0c1, t1c0, t1c1, t2c0, t2c1]
    let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let out = transpose_tc_to_ct(&data, 3, 2);
    // [C=2, T=3]
    assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_transpose_roundtrip() {
    let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let channels = 4;
    let seq_len = 2;
    let tc = transpose_ct_to_tc(&original, channels, seq_len);
    let ct = transpose_tc_to_ct(&tc, seq_len, channels);
    assert_eq!(ct, original);
}

// -- add_sinusoidal_1d --------------------------------------------------------

#[test]
fn test_add_sinusoidal_1d_nonzero() {
    let dim = 8;
    let seq = 4;
    let mut data = vec![0.0f32; seq * dim];
    add_sinusoidal_1d(&mut data, seq, dim);
    // Position 0 should have cos(0)=1 in first half, sin(0)=0 in second half
    assert!(
        (data[0] - 1.0).abs() < 1e-6,
        "cos(0) should be 1.0, got {}",
        data[0]
    );
    assert!(
        data[dim / 2].abs() < 1e-6,
        "sin(0) should be 0.0, got {}",
        data[dim / 2]
    );
}

#[test]
fn test_add_sinusoidal_1d_position_varies() {
    let dim = 16;
    let seq = 3;
    let mut data = vec![0.0f32; seq * dim];
    add_sinusoidal_1d(&mut data, seq, dim);
    // Position 0 and position 1 should differ
    let pos0 = &data[0..dim];
    let pos1 = &data[dim..2 * dim];
    let diff: f32 = pos0
        .iter()
        .zip(pos1.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 0.01,
        "different positions should produce different embeddings"
    );
}

// -- build_sinusoidal_table ---------------------------------------------------

#[test]
fn test_build_sinusoidal_table_shape() {
    let seq = 8;
    let dim = 32;
    let table = build_sinusoidal_table(seq, dim);
    assert_eq!(table.len(), seq * dim);
}

#[test]
fn test_build_sinusoidal_table_matches_add() {
    let seq = 4;
    let dim = 16;
    let table = build_sinusoidal_table(seq, dim);
    let mut manual = vec![0.0f32; seq * dim];
    add_sinusoidal_1d(&mut manual, seq, dim);
    assert_eq!(table, manual);
}
