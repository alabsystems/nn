#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for ALiBi attention bias — extracted per 500-line limit.

use super::*;
use crate::dyn_tensor::test_helpers::cpu;

#[test]
fn test_alibi_slopes_12_heads() {
    // Emotion2vec reference values from the design doc
    let slopes = alibi_slopes(12).unwrap();
    assert_eq!(slopes.len(), 12);
    // Head 1: 2^(-8/12) ≈ 0.630
    assert!((slopes[0] - 0.630).abs() < 0.001, "slopes[0]={}", slopes[0]);
    // Head 2: 2^(-16/12) ≈ 0.397
    assert!((slopes[1] - 0.397).abs() < 0.001, "slopes[1]={}", slopes[1]);
    // Head 3: 2^(-24/12) = 2^(-2) = 0.25
    assert!((slopes[2] - 0.250).abs() < 0.001, "slopes[2]={}", slopes[2]);
    // Head 12: 2^(-96/12) = 2^(-8) ≈ 0.00391
    assert!(
        (slopes[11] - 0.003_906_25).abs() < 0.0001,
        "slopes[11]={}",
        slopes[11]
    );
}

#[test]
fn test_alibi_slopes_strictly_decreasing() {
    let slopes = alibi_slopes(8).unwrap();
    for i in 0..slopes.len() - 1 {
        assert!(
            slopes[i] > slopes[i + 1],
            "slopes not decreasing at {i}: {} <= {}",
            slopes[i],
            slopes[i + 1]
        );
    }
}

#[test]
fn test_alibi_slopes_all_positive_finite() {
    for n in 1..=32 {
        let slopes = alibi_slopes(n).unwrap();
        for (i, &s) in slopes.iter().enumerate() {
            assert!(s.is_finite(), "slopes[{i}] not finite for n={n}");
            assert!(s > 0.0, "slopes[{i}] not positive for n={n}");
        }
    }
}

#[test]
fn test_alibi_slopes_zero_heads_errors() {
    let result = alibi_slopes(0);
    assert!(result.is_err());
}

#[test]
fn test_alibi_bias_shape() {
    let bias = alibi_bias(4, 8, &cpu()).unwrap();
    assert_eq!(bias.dims(), &[1, 4, 8, 8]);
}

#[test]
fn test_alibi_bias_antisymmetric() {
    // bias[h][i][j] = -bias[h][j][i] (antisymmetric in query/key positions)
    let num_heads = 4;
    let seq_len = 6;
    let bias = alibi_bias(num_heads, seq_len, &cpu()).unwrap();
    let data = bias.to_flat_vec::<f32>().unwrap();

    for h in 0..num_heads {
        for i in 0..seq_len {
            for j in 0..seq_len {
                let idx_ij = h * seq_len * seq_len + i * seq_len + j;
                let idx_ji = h * seq_len * seq_len + j * seq_len + i;
                let val_ij = data[idx_ij];
                let val_ji = data[idx_ji];
                assert!(
                    (val_ij + val_ji).abs() < 1e-6,
                    "not antisymmetric at h={h} i={i} j={j}: {val_ij} + {val_ji} != 0"
                );
            }
        }
    }
}

#[test]
fn test_alibi_bias_diagonal_zero() {
    // bias[h][i][i] = slopes[h] * 0 = 0
    let bias = alibi_bias(4, 5, &cpu()).unwrap();
    let data = bias.to_flat_vec::<f32>().unwrap();
    for h in 0..4 {
        for i in 0..5 {
            let idx = h * 25 + i * 5 + i;
            assert!(
                data[idx].abs() < 1e-10,
                "diagonal not zero at h={h} i={i}: {}",
                data[idx]
            );
        }
    }
}

#[test]
fn test_alibi_bias_values() {
    // With 1 head: slope = 2^(-8) = 1/256
    // bias[0][0][j] = (1/256) * (j - 0) = j/256
    let bias = alibi_bias(1, 4, &cpu()).unwrap();
    let data = bias.to_flat_vec::<f32>().unwrap();
    let slope = 2f32.powf(-8.0);
    // Row 0: bias[0][0][j] = slope * j
    assert!((data[0] - slope * 0.0).abs() < 1e-7);
    assert!((data[1] - slope * 1.0).abs() < 1e-7);
    assert!((data[2] - slope * 2.0).abs() < 1e-7);
    assert!((data[3] - slope * 3.0).abs() < 1e-7);
    // Row 2: bias[0][2][j] = slope * (j - 2)
    assert!((data[8] - slope * -2.0).abs() < 1e-7); // j=0
    assert!((data[9] - (-slope)).abs() < 1e-7); // j=1
    assert!((data[10] - slope * 0.0).abs() < 1e-7); // j=2
    assert!((data[11] - slope * 1.0).abs() < 1e-7); // j=3
}

#[test]
fn test_alibi_bias_seq_len_1() {
    // Edge case: single token has zero bias
    let bias = alibi_bias(4, 1, &cpu()).unwrap();
    assert_eq!(bias.dims(), &[1, 4, 1, 1]);
    let data = bias.to_flat_vec::<f32>().unwrap();
    assert_eq!(data.len(), 4);
    for &v in &data {
        assert!(v.abs() < 1e-10, "single token bias should be zero");
    }
}

#[test]
fn test_alibi_bias_seq_len_0() {
    let bias = alibi_bias(4, 0, &cpu()).unwrap();
    assert_eq!(bias.dims(), &[1, 4, 0, 0]);
}

#[test]
fn test_alibi_bias_scaled_shape() {
    let scale = DynTensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[4], &cpu()).unwrap();
    let bias = alibi_bias_scaled(4, 6, &scale, &cpu()).unwrap();
    assert_eq!(bias.dims(), &[1, 4, 6, 6]);
}

#[test]
fn test_alibi_bias_scaled_unity_equals_unscaled() {
    let num_heads = 3;
    let seq_len = 5;
    let scale = DynTensor::from_vec(vec![1.0; num_heads], &[num_heads], &cpu()).unwrap();
    let unscaled = alibi_bias(num_heads, seq_len, &cpu()).unwrap();
    let scaled = alibi_bias_scaled(num_heads, seq_len, &scale, &cpu()).unwrap();
    let u_data = unscaled.to_flat_vec::<f32>().unwrap();
    let s_data = scaled.to_flat_vec::<f32>().unwrap();
    for (i, (&u, &s)) in u_data.iter().zip(s_data.iter()).enumerate() {
        assert!(
            (u - s).abs() < 1e-7,
            "mismatch at {i}: unscaled={u}, scaled={s}"
        );
    }
}

#[test]
fn test_alibi_bias_scaled_doubles() {
    // scale = [2.0, 2.0] should produce 2x the unscaled bias
    let num_heads = 2;
    let seq_len = 4;
    let scale = DynTensor::from_vec(vec![2.0; num_heads], &[num_heads], &cpu()).unwrap();
    let unscaled = alibi_bias(num_heads, seq_len, &cpu()).unwrap();
    let scaled = alibi_bias_scaled(num_heads, seq_len, &scale, &cpu()).unwrap();
    let u_data = unscaled.to_flat_vec::<f32>().unwrap();
    let s_data = scaled.to_flat_vec::<f32>().unwrap();
    for (i, (&u, &s)) in u_data.iter().zip(s_data.iter()).enumerate() {
        assert!(
            (2.0 * u - s).abs() < 1e-6,
            "mismatch at {i}: 2*unscaled={}, scaled={s}",
            2.0 * u
        );
    }
}

#[test]
fn test_alibi_bias_scaled_wrong_shape_errors() {
    let scale = DynTensor::from_vec(vec![1.0, 1.0], &[2], &cpu()).unwrap();
    let result = alibi_bias_scaled(4, 6, &scale, &cpu());
    assert!(result.is_err());
}

// -- Proof-coverage additions: mathematical invariant tests --

// ---------------------------------------------------------------------------
// Test: Linearity in distance — bias[h][i][j] - bias[h][i][k] = slope[h] * (j - k)
// The ALiBi bias is linear in relative position for each head.
// ---------------------------------------------------------------------------
#[test]
fn test_alibi_bias_linear_in_distance() {
    let num_heads = 4;
    let seq_len = 8;
    let slopes = alibi_slopes(num_heads).unwrap();
    let bias = alibi_bias(num_heads, seq_len, &cpu()).unwrap();
    let data = bias.to_flat_vec::<f32>().unwrap();

    for h in 0..num_heads {
        let slope = slopes[h];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let expected = slope * (j as f32 - i as f32);
                let actual = data[h * seq_len * seq_len + i * seq_len + j];
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "bias[{h}][{i}][{j}] = {actual}, expected slope*dist = {expected}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test: Large head count stability — slopes remain positive and finite
// for head counts used by real models (up to 128).
// ---------------------------------------------------------------------------
#[test]
fn test_alibi_slopes_large_head_count() {
    for num_heads in [32, 48, 64, 96, 128] {
        let slopes = alibi_slopes(num_heads).unwrap();
        assert_eq!(slopes.len(), num_heads);
        for (i, &s) in slopes.iter().enumerate() {
            assert!(
                s.is_finite() && s > 0.0,
                "slopes[{i}] = {s} for num_heads={num_heads} — must be positive finite"
            );
        }
        // Last slope should be the smallest (2^(-8))
        let last = slopes[num_heads - 1];
        let expected_last = 2f32.powf(-8.0);
        assert!(
            (last - expected_last).abs() < 1e-6,
            "last slope for H={num_heads}: got {last}, expected {expected_last}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test: Slope geometric progression ratio — consecutive slopes should have
// ratio 2^(-8/H) (constant geometric ratio).
// ---------------------------------------------------------------------------
#[test]
fn test_alibi_slopes_geometric_ratio() {
    for num_heads in [4, 8, 12, 16] {
        let slopes = alibi_slopes(num_heads).unwrap();
        let expected_ratio = 2f32.powf(-8.0 / num_heads as f32);
        for i in 0..slopes.len() - 1 {
            let ratio = slopes[i + 1] / slopes[i];
            assert!(
                (ratio - expected_ratio).abs() < 1e-5,
                "ratio slopes[{}]/slopes[{}] = {ratio}, expected {expected_ratio} for H={num_heads}",
                i + 1,
                i
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test: ALiBi bias causal property — positions before the query (j < i)
// get negative bias (penalty), positions after (j > i) get positive bias.
// This is correct for bidirectional attention; for causal masks, only j <= i
// positions are used, and the negative bias penalizes distant past tokens.
// ---------------------------------------------------------------------------
#[test]
fn test_alibi_bias_causal_sign_convention() {
    let num_heads = 2;
    let seq_len = 5;
    let bias = alibi_bias(num_heads, seq_len, &cpu()).unwrap();
    let data = bias.to_flat_vec::<f32>().unwrap();

    for h in 0..num_heads {
        for i in 0..seq_len {
            for j in 0..seq_len {
                let idx = h * seq_len * seq_len + i * seq_len + j;
                let val = data[idx];
                if j < i {
                    // Past positions: negative bias (penalty)
                    assert!(
                        val < 0.0,
                        "bias[{h}][{i}][{j}] = {val}, expected < 0 for j < i"
                    );
                } else if j > i {
                    // Future positions: positive bias
                    assert!(
                        val > 0.0,
                        "bias[{h}][{i}][{j}] = {val}, expected > 0 for j > i"
                    );
                }
                // j == i: zero (already tested by diagonal test)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test: Scaled bias with zero scale produces zero bias everywhere.
// ---------------------------------------------------------------------------
#[test]
fn test_alibi_bias_scaled_zero_scale() {
    let num_heads = 3;
    let seq_len = 4;
    let scale = DynTensor::from_vec(vec![0.0; num_heads], &[num_heads], &cpu()).unwrap();
    let bias = alibi_bias_scaled(num_heads, seq_len, &scale, &cpu()).unwrap();
    let data = bias.to_flat_vec::<f32>().unwrap();
    for (i, &v) in data.iter().enumerate() {
        assert!(v.abs() < 1e-10, "element {i}: expected 0.0, got {v}");
    }
}
