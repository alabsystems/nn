// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for attention module: causal masks, ALiBi, SDPA, repeat_kv.
//!
//! Part of #4186.

use crate::dyn_tensor::DynTensor;
use crate::layers::attention::{
    alibi_bias, alibi_slopes, causal_mask, causal_mask_with_offset, repeat_kv, sdpa_causal,
};
use crate::test_prng::rand_f32_vec;
use crate::{DType, Device};

/// Helper: create a DynTensor with deterministic random data.
fn rand_tensor(seed: u64, dims: &[usize]) -> DynTensor {
    let numel: usize = dims.iter().product();
    let data = rand_f32_vec(seed, numel, -1.0, 1.0);
    DynTensor::from_vec(data, dims, &Device::Cpu).unwrap()
}

#[test]
fn test_causal_mask_shape() {
    let seq_len = 8;
    let mask = causal_mask(seq_len, &Device::Cpu).unwrap();
    // causal_mask returns [1, 1, seq_len, seq_len]
    assert_eq!(mask.dims(), &[1, 1, seq_len, seq_len]);
}

#[test]
fn test_causal_mask_values() {
    let seq_len = 4;
    let mask = causal_mask(seq_len, &Device::Cpu).unwrap();
    let data = mask.to_f32_array().unwrap();
    // mask[0][0][i][j]: 0.0 if j <= i, -inf if j > i
    for i in 0..seq_len {
        for j in 0..seq_len {
            let val = data[[0, 0, i, j]];
            if j > i {
                assert!(
                    val == f32::NEG_INFINITY,
                    "Expected -inf at [{i}][{j}], got {val}"
                );
            } else {
                assert!(
                    (val - 0.0).abs() < 1e-6,
                    "Expected 0.0 at [{i}][{j}], got {val}"
                );
            }
        }
    }
}

#[test]
fn test_causal_mask_with_offset() {
    // Simulate cached decoding: 2 new tokens, 5 total tokens
    // offset = total - new = 3
    let new_tokens = 2;
    let total_tokens = 5;
    let mask = causal_mask_with_offset(new_tokens, total_tokens, DType::F32, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, new_tokens, total_tokens]);

    let data = mask.to_f32_array().unwrap();
    // Row 0: absolute position = 3, can attend to positions 0..=3
    // Row 1: absolute position = 4, can attend to positions 0..=4 (all)
    for col in 0..total_tokens {
        let val_row0 = data[[0, 0, 0, col]];
        if col > 3 {
            assert!(
                val_row0 == f32::NEG_INFINITY,
                "Row 0, col {col}: expected -inf, got {val_row0}"
            );
        } else {
            assert!(
                (val_row0 - 0.0).abs() < 1e-6,
                "Row 0, col {col}: expected 0.0, got {val_row0}"
            );
        }
    }
    // Row 1 at absolute position 4 can attend to all 5 positions
    for col in 0..total_tokens {
        let val_row1 = data[[0, 0, 1, col]];
        assert!(
            (val_row1 - 0.0).abs() < 1e-6,
            "Row 1, col {col}: expected 0.0, got {val_row1}"
        );
    }
}

#[test]
fn test_alibi_slopes_power_of_2_heads() {
    // 8 heads: slopes[h] = 2^(-8*(h+1)/8) = 2^(-(h+1))
    let slopes = alibi_slopes(8).unwrap();
    assert_eq!(slopes.len(), 8);
    for (h, &slope) in slopes.iter().enumerate() {
        let expected = 2f32.powf(-((h + 1) as f32));
        assert!(
            (slope - expected).abs() < 1e-6,
            "Head {h}: expected {expected}, got {slope}"
        );
    }
    // Verify geometric sequence: each slope is half the previous
    for h in 1..8 {
        let ratio = slopes[h] / slopes[h - 1];
        assert!(
            (ratio - 0.5).abs() < 1e-5,
            "Slopes ratio at head {h} should be 0.5, got {ratio}"
        );
    }
}

#[test]
fn test_alibi_slopes_non_power_of_2() {
    // 6 heads: slopes[h] = 2^(-8*(h+1)/6)
    let slopes = alibi_slopes(6).unwrap();
    assert_eq!(slopes.len(), 6);
    for (h, &slope) in slopes.iter().enumerate() {
        let expected = 2f32.powf(-8.0 * (h + 1) as f32 / 6.0);
        assert!(
            (slope - expected).abs() < 1e-6,
            "Head {h}: expected {expected}, got {slope}"
        );
    }
    // Slopes should be strictly decreasing
    for h in 1..6 {
        assert!(
            slopes[h] < slopes[h - 1],
            "Slopes should be strictly decreasing: slopes[{}]={} >= slopes[{}]={}",
            h,
            slopes[h],
            h - 1,
            slopes[h - 1]
        );
    }
}

#[test]
fn test_alibi_bias_shape() {
    let num_heads = 4;
    let seq_len = 16;
    let bias = alibi_bias(num_heads, seq_len, &Device::Cpu).unwrap();
    // alibi_bias returns [1, num_heads, seq_len, seq_len]
    assert_eq!(bias.dims(), &[1, num_heads, seq_len, seq_len]);
}

#[test]
fn test_sdpa_causal_output_shape() {
    let batch = 2;
    let num_heads = 4;
    let seq_len = 8;
    let head_dim = 16;
    let scale = 1.0 / (head_dim as f64).sqrt();

    let q = rand_tensor(42, &[batch, num_heads, seq_len, head_dim]);
    let k = rand_tensor(43, &[batch, num_heads, seq_len, head_dim]);
    let v = rand_tensor(44, &[batch, num_heads, seq_len, head_dim]);

    let output = sdpa_causal(&q, &k, &v, scale).unwrap();
    assert_eq!(output.dims(), &[batch, num_heads, seq_len, head_dim]);
}

#[test]
fn test_repeat_kv_no_repeat() {
    // repeat_kv with n=1 should return a clone (identity)
    let batch = 1;
    let num_kv_heads = 4;
    let seq_len = 8;
    let head_dim = 16;

    let x = rand_tensor(50, &[batch, num_kv_heads, seq_len, head_dim]);
    let result = repeat_kv(&x, 1).unwrap();
    assert_eq!(result.dims(), x.dims());
    // Values should be identical
    let x_data = x.to_f32_array().unwrap();
    let r_data = result.to_f32_array().unwrap();
    assert_eq!(x_data, r_data);
}

#[test]
fn test_repeat_kv_double() {
    // repeat_kv with n=2 should double the kv heads
    let batch = 1;
    let num_kv_heads = 2;
    let seq_len = 4;
    let head_dim = 8;

    let x = rand_tensor(60, &[batch, num_kv_heads, seq_len, head_dim]);
    let result = repeat_kv(&x, 2).unwrap();
    // Output shape: [batch, num_kv_heads * 2, seq_len, head_dim]
    assert_eq!(result.dims(), &[batch, num_kv_heads * 2, seq_len, head_dim]);
}
