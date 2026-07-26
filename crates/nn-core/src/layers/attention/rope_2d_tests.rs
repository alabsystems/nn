#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use crate::dyn_tensor::DynTensor;
use crate::{DType, Device};

use super::{sinusoidal_2d, RotaryEmbedding2d};

// -- RotaryEmbedding2d tests --------------------------------------------------

#[test]
fn test_rope_2d_output_shape() {
    let rope = RotaryEmbedding2d::new(8, 16, 10000.0, &Device::Cpu).unwrap();
    // 4 tokens in a 2×2 grid
    let x = DynTensor::ones(&[1, 4, 8], DType::F32, &Device::Cpu).unwrap();
    let h_pos = vec![0, 0, 1, 1];
    let w_pos = vec![0, 1, 0, 1];
    let y = rope.apply(&x, &h_pos, &w_pos).unwrap();
    assert_eq!(y.dims(), &[1, 4, 8]);
}

#[test]
fn test_rope_2d_preserves_norm() {
    // RoPE is a rotation — it preserves the L2 norm of the input.
    let rope = RotaryEmbedding2d::new(8, 16, 10000.0, &Device::Cpu).unwrap();
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = DynTensor::from_vec(data.clone(), &[1, 1, 8], &Device::Cpu).unwrap();
    let y = rope.apply(&x, &[3], &[7]).unwrap();

    let input_norm_sq: f32 = data.iter().map(|v| v * v).sum();
    let output_vals = y.to_flat_vec::<f32>().unwrap();
    let output_norm_sq: f32 = output_vals.iter().map(|v| v * v).sum();

    assert!(
        (input_norm_sq - output_norm_sq).abs() < 1e-4,
        "norm not preserved: input={input_norm_sq}, output={output_norm_sq}"
    );
}

#[test]
fn test_rope_2d_position_zero_identity() {
    // At position (0, 0), cos(0)=1, sin(0)=0, so rotation is identity.
    let rope = RotaryEmbedding2d::new(8, 16, 10000.0, &Device::Cpu).unwrap();
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = DynTensor::from_vec(data.clone(), &[1, 1, 8], &Device::Cpu).unwrap();
    let y = rope.apply(&x, &[0], &[0]).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    for (i, (&a, &b)) in vals.iter().zip(data.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "position (0,0) should be identity, dim {i}: {a} != {b}"
        );
    }
}

#[test]
fn test_rope_2d_different_positions_differ() {
    // Different spatial positions should produce different outputs.
    let rope = RotaryEmbedding2d::new(8, 16, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[1, 2, 8], DType::F32, &Device::Cpu).unwrap();
    let y = rope.apply(&x, &[0, 5], &[0, 5]).unwrap();
    let vals = y.to_flat_vec::<f32>().unwrap();
    // Token 0 at (0,0) and token 1 at (5,5) should differ.
    let t0 = &vals[..8];
    let t1 = &vals[8..16];
    let diff: f32 = t0.iter().zip(t1).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        diff > 0.01,
        "different positions should produce different outputs, diff={diff}"
    );
}

#[test]
fn test_rope_2d_batched() {
    // Batched input should work.
    let rope = RotaryEmbedding2d::new(8, 16, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[2, 3, 4, 8], DType::F32, &Device::Cpu).unwrap();
    let y = rope.apply(&x, &[0, 1, 2, 3], &[0, 1, 2, 3]).unwrap();
    assert_eq!(y.dims(), &[2, 3, 4, 8]);
}

#[test]
fn test_rope_2d_head_dim_not_multiple_of_4() {
    assert!(RotaryEmbedding2d::new(6, 16, 10000.0, &Device::Cpu).is_err());
    assert!(RotaryEmbedding2d::new(0, 16, 10000.0, &Device::Cpu).is_err());
    assert!(RotaryEmbedding2d::new(2, 16, 10000.0, &Device::Cpu).is_err());
}

#[test]
fn test_rope_2d_zero_max_position() {
    assert!(RotaryEmbedding2d::new(8, 0, 10000.0, &Device::Cpu).is_err());
}

#[test]
fn test_rope_2d_invalid_base() {
    assert!(RotaryEmbedding2d::new(8, 16, 0.0, &Device::Cpu).is_err());
    assert!(RotaryEmbedding2d::new(8, 16, -1.0, &Device::Cpu).is_err());
    assert!(RotaryEmbedding2d::new(8, 16, f64::NAN, &Device::Cpu).is_err());
    assert!(RotaryEmbedding2d::new(8, 16, f64::INFINITY, &Device::Cpu).is_err());
}

#[test]
fn test_rope_2d_position_out_of_range() {
    let rope = RotaryEmbedding2d::new(8, 4, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[1, 1, 8], DType::F32, &Device::Cpu).unwrap();
    // Position 4 is out of range for max_position=4.
    assert!(rope.apply(&x, &[4], &[0]).is_err());
    assert!(rope.apply(&x, &[0], &[4]).is_err());
}

#[test]
fn test_rope_2d_wrong_head_dim() {
    let rope = RotaryEmbedding2d::new(8, 16, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[1, 1, 12], DType::F32, &Device::Cpu).unwrap();
    assert!(rope.apply(&x, &[0], &[0]).is_err());
}

#[test]
fn test_rope_2d_positions_length_mismatch() {
    let rope = RotaryEmbedding2d::new(8, 16, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[1, 2, 8], DType::F32, &Device::Cpu).unwrap();
    // seq_len=2 but only 1 position given
    assert!(rope.apply(&x, &[0], &[0, 1]).is_err());
    assert!(rope.apply(&x, &[0, 1], &[0]).is_err());
}

#[test]
fn test_rope_2d_accessors() {
    let rope = RotaryEmbedding2d::new(16, 32, 10000.0, &Device::Cpu).unwrap();
    assert_eq!(rope.head_dim(), 16);
    assert_eq!(rope.max_position(), 32);
}

#[test]
fn test_rope_2d_rank_too_low() {
    let rope = RotaryEmbedding2d::new(8, 16, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[8], DType::F32, &Device::Cpu).unwrap();
    assert!(rope.apply(&x, &[0], &[0]).is_err());
}

// -- sinusoidal_2d tests ------------------------------------------------------

#[test]
fn test_sinusoidal_2d_output_shape() {
    let pe = sinusoidal_2d(4, 6, 16, 10000.0, &Device::Cpu).unwrap();
    assert_eq!(pe.dims(), &[24, 16]); // 4*6=24 tokens, dim=16
}

#[test]
fn test_sinusoidal_2d_values_bounded() {
    // sin/cos values are always in [-1, 1].
    let pe = sinusoidal_2d(3, 3, 8, 10000.0, &Device::Cpu).unwrap();
    let vals = pe.to_flat_vec::<f32>().unwrap();
    for &v in &vals {
        assert!(
            (-1.0 - 1e-6..=1.0 + 1e-6).contains(&v),
            "sinusoidal_2d value {v} out of [-1, 1]"
        );
    }
}

#[test]
fn test_sinusoidal_2d_origin_is_zero() {
    // At position (0, 0): sin(0)=0, cos(0)=1 for all frequencies.
    let pe = sinusoidal_2d(2, 2, 8, 10000.0, &Device::Cpu).unwrap();
    let vals = pe.to_flat_vec::<f32>().unwrap();
    // First row is position (0, 0).
    let origin = &vals[..8];
    // quarter_dim = 2: [sin_h(0), sin_h(0), cos_h(0), cos_h(0), sin_w(0), sin_w(0), cos_w(0), cos_w(0)]
    // sin(0) = 0, cos(0) = 1
    let quarter = 2;
    for &v in &origin[..quarter] {
        assert!(v.abs() < 1e-6, "sin_h should be 0 at origin, got {v}");
    }
    for &v in &origin[quarter..2 * quarter] {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "cos_h should be 1 at origin, got {v}"
        );
    }
    for &v in &origin[2 * quarter..3 * quarter] {
        assert!(v.abs() < 1e-6, "sin_w should be 0 at origin, got {v}");
    }
    for &v in &origin[3 * quarter..4 * quarter] {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "cos_w should be 1 at origin, got {v}"
        );
    }
}

#[test]
fn test_sinusoidal_2d_different_positions_differ() {
    let pe = sinusoidal_2d(3, 3, 8, 10000.0, &Device::Cpu).unwrap();
    let vals = pe.to_flat_vec::<f32>().unwrap();
    // Compare (0,0) = row 0 vs (1,1) = row 4
    let pos00 = &vals[0..8];
    let pos11 = &vals[4 * 8..5 * 8];
    let diff: f32 = pos00.iter().zip(pos11).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        diff > 0.01,
        "different positions should differ, diff={diff}"
    );
}

#[test]
fn test_sinusoidal_2d_same_row_different_col() {
    // Positions (0, 0) and (0, 1) should have same height encoding, different width.
    let pe = sinusoidal_2d(2, 3, 8, 10000.0, &Device::Cpu).unwrap();
    let vals = pe.to_flat_vec::<f32>().unwrap();
    let pos00 = &vals[0..8];
    let pos01 = &vals[8..16]; // (0, 1) is the next position in row-major order
    let quarter = 2;
    // Height components (first 2*quarter elements) should match.
    for i in 0..2 * quarter {
        assert!(
            (pos00[i] - pos01[i]).abs() < 1e-6,
            "same row should have same h encoding, dim {i}: {} vs {}",
            pos00[i],
            pos01[i]
        );
    }
    // Width components (last 2*quarter elements) should differ for (0,0) vs (0,1).
    let w_diff: f32 = (2 * quarter..4 * quarter)
        .map(|i| (pos00[i] - pos01[i]).abs())
        .sum();
    assert!(
        w_diff > 0.01,
        "different columns should have different w encoding, diff={w_diff}"
    );
}

#[test]
fn test_sinusoidal_2d_invalid_dim() {
    assert!(sinusoidal_2d(2, 2, 0, 10000.0, &Device::Cpu).is_err());
    assert!(sinusoidal_2d(2, 2, 3, 10000.0, &Device::Cpu).is_err());
    assert!(sinusoidal_2d(2, 2, 6, 10000.0, &Device::Cpu).is_err());
}

#[test]
fn test_sinusoidal_2d_zero_spatial() {
    assert!(sinusoidal_2d(0, 3, 8, 10000.0, &Device::Cpu).is_err());
    assert!(sinusoidal_2d(3, 0, 8, 10000.0, &Device::Cpu).is_err());
}

#[test]
fn test_sinusoidal_2d_invalid_temperature() {
    assert!(sinusoidal_2d(2, 2, 8, 0.0, &Device::Cpu).is_err());
    assert!(sinusoidal_2d(2, 2, 8, -1.0, &Device::Cpu).is_err());
    assert!(sinusoidal_2d(2, 2, 8, f64::NAN, &Device::Cpu).is_err());
    assert!(sinusoidal_2d(2, 2, 8, f64::INFINITY, &Device::Cpu).is_err());
}

#[test]
fn test_sinusoidal_2d_1x1_grid() {
    // Single-element grid should produce sin(0)/cos(0) for all frequencies.
    let pe = sinusoidal_2d(1, 1, 8, 10000.0, &Device::Cpu).unwrap();
    assert_eq!(pe.dims(), &[1, 8]);
    let vals = pe.to_flat_vec::<f32>().unwrap();
    // All sin terms = 0, all cos terms = 1
    let quarter = 2;
    for &v in &vals[..quarter] {
        assert!(v.abs() < 1e-6, "sin_h at origin: {v}");
    }
    for &v in &vals[quarter..2 * quarter] {
        assert!((v - 1.0).abs() < 1e-6, "cos_h at origin: {v}");
    }
}
