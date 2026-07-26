#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended RoPE tests — half-split free function and apply_pair/apply_at_positions.
//!
//! Extracted from `rope_tests.rs` for file-size compliance.

use super::RotaryEmbedding;
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device};

// -- Half-split rope free function tests (candle convention, half-dim cos/sin) --

#[test]
fn test_rope_free_fn_identity_at_zero_sin() {
    // cos=1, sin=0: rope(t, cos, sin) == t (identity rotation)
    // t: [1, 4], cos/sin: [1, 2] (half-dim)
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
    let cos = DynTensor::from_vec(vec![1.0, 1.0], &[1, 2], &Device::Cpu).unwrap();
    let sin = DynTensor::from_vec(vec![0.0, 0.0], &[1, 2], &Device::Cpu).unwrap();
    let y = super::rope(&t, &cos, &sin).unwrap();
    let y_flat = y.to_flat_vec::<f32>().unwrap();
    let t_flat = t.to_flat_vec::<f32>().unwrap();
    for (a, b) in y_flat.iter().zip(t_flat.iter()) {
        assert!((a - b).abs() < 1e-5, "expected {b}, got {a}");
    }
}

#[test]
fn test_rope_free_fn_90_degree_rotation() {
    // cos=0, sin=1: tests the half-split rotation with half-dim cos/sin
    // t = [x1=1, x1=2 | x2=3, x2=4]
    // y1 = x1*cos - x2*sin = [1*0 - 3*1, 2*0 - 4*1] = [-3, -4]
    // y2 = x1*sin + x2*cos = [1*1 + 3*0, 2*1 + 4*0] = [1, 2]
    // result = [-3, -4, 1, 2]
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
    let cos = DynTensor::from_vec(vec![0.0, 0.0], &[1, 2], &Device::Cpu).unwrap();
    let sin = DynTensor::from_vec(vec![1.0, 1.0], &[1, 2], &Device::Cpu).unwrap();
    let y = super::rope(&t, &cos, &sin).unwrap();
    let flat = y.to_flat_vec::<f32>().unwrap();
    assert!(
        (flat[0] - (-3.0)).abs() < 1e-5,
        "expected -3, got {}",
        flat[0]
    );
    assert!(
        (flat[1] - (-4.0)).abs() < 1e-5,
        "expected -4, got {}",
        flat[1]
    );
    assert!((flat[2] - 1.0).abs() < 1e-5, "expected 1, got {}", flat[2]);
    assert!((flat[3] - 2.0).abs() < 1e-5, "expected 2, got {}", flat[3]);
}

#[test]
fn test_rope_free_fn_odd_head_dim_rejected() {
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &Device::Cpu).unwrap();
    let cos = DynTensor::from_vec(vec![1.0], &[1, 1], &Device::Cpu).unwrap();
    let sin = DynTensor::from_vec(vec![0.0], &[1, 1], &Device::Cpu).unwrap();
    assert!(super::rope(&t, &cos, &sin).is_err());
}

#[test]
fn test_rope_free_fn_rank1_rejected() {
    let t = DynTensor::from_vec(vec![1.0, 2.0], &[2], &Device::Cpu).unwrap();
    let cos = DynTensor::from_vec(vec![1.0], &[1], &Device::Cpu).unwrap();
    let sin = DynTensor::from_vec(vec![0.0], &[1], &Device::Cpu).unwrap();
    assert!(super::rope(&t, &cos, &sin).is_err());
}

#[test]
fn test_rope_free_fn_full_dim_cos_sin_rejected() {
    // Full-dim cos/sin (head_dim=4, cos last dim=4) should be rejected
    let t = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
    let cos = DynTensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 4], &Device::Cpu).unwrap();
    let sin = DynTensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], &[1, 4], &Device::Cpu).unwrap();
    assert!(
        super::rope(&t, &cos, &sin).is_err(),
        "full-dim cos/sin should be rejected; candle uses half-dim"
    );
}

#[test]
fn test_rope_free_fn_norm_preservation() {
    // RoPE is a rotation — it preserves vector norms.
    // t: [1, 4], cos/sin: [1, 2] (half-dim)
    let t = DynTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[1, 4], &Device::Cpu).unwrap();
    let angle: f32 = 0.3;
    let cos_val = angle.cos();
    let sin_val = angle.sin();
    let cos = DynTensor::from_vec(vec![cos_val, cos_val], &[1, 2], &Device::Cpu).unwrap();
    let sin = DynTensor::from_vec(vec![sin_val, sin_val], &[1, 2], &Device::Cpu).unwrap();
    let y = super::rope(&t, &cos, &sin).unwrap();
    let t_flat = t.to_flat_vec::<f32>().unwrap();
    let y_flat = y.to_flat_vec::<f32>().unwrap();
    let norm_t: f32 = t_flat.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_y: f32 = y_flat.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm_t - norm_y).abs() < 1e-5,
        "norm changed: {norm_t} -> {norm_y}"
    );
}

// ---------------------------------------------------------------------------
// apply_pair / apply_at_positions tests
// ---------------------------------------------------------------------------

#[test]
fn test_rope_apply_pair_basic() {
    // apply_pair rotates both q and k at the given positions.
    let rope = RotaryEmbedding::new(4, 32, 10000.0, &Device::Cpu).unwrap();
    let q = DynTensor::ones(&[1, 1, 3, 4], DType::F32, &Device::Cpu).unwrap();
    let k = DynTensor::ones(&[1, 1, 3, 4], DType::F32, &Device::Cpu).unwrap();
    let (q_rot, k_rot) = rope.apply_pair(&q, &k, &[0, 1, 2]).unwrap();
    assert_eq!(q_rot.dims(), &[1, 1, 3, 4]);
    assert_eq!(k_rot.dims(), &[1, 1, 3, 4]);
    // Both outputs should be finite
    let q_flat = q_rot.to_flat_vec::<f32>().unwrap();
    let k_flat = k_rot.to_flat_vec::<f32>().unwrap();
    assert!(q_flat.iter().all(|v| v.is_finite()));
    assert!(k_flat.iter().all(|v| v.is_finite()));
}

#[test]
fn test_rope_apply_pair_matches_apply() {
    // apply_pair(positions=[5,6,7]) should match apply(offset=5) for seq_len=3.
    let rope = RotaryEmbedding::new(4, 32, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[1, 1, 3, 4],
        &Device::Cpu,
    )
    .unwrap();

    let via_apply = rope.apply(&x, 5).unwrap();
    let (via_pair, _) = rope.apply_pair(&x, &x, &[5, 6, 7]).unwrap();

    let a = via_apply.to_flat_vec::<f32>().unwrap();
    let b = via_pair.to_flat_vec::<f32>().unwrap();
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (av - bv).abs() < 1e-6,
            "apply vs apply_pair mismatch at [{i}]: {av} vs {bv}"
        );
    }
}

#[test]
fn test_rope_apply_pair_non_contiguous_positions() {
    // Non-contiguous positions (e.g., [0, 3, 7]) — common in KV cache decode.
    let rope = RotaryEmbedding::new(4, 32, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[1, 1, 3, 4], DType::F32, &Device::Cpu).unwrap();
    let (q_rot, _) = rope.apply_pair(&x, &x, &[0, 3, 7]).unwrap();
    assert_eq!(q_rot.dims(), &[1, 1, 3, 4]);
    let vals = q_rot.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|v| v.is_finite()));
    // Position 0 should match apply(offset=0, seq_len=1)
    let single = DynTensor::ones(&[1, 1, 1, 4], DType::F32, &Device::Cpu).unwrap();
    let pos0 = rope.apply(&single, 0).unwrap();
    let pos0_vals = pos0.to_flat_vec::<f32>().unwrap();
    for i in 0..4 {
        assert!(
            (vals[i] - pos0_vals[i]).abs() < 1e-6,
            "position 0 mismatch at [{i}]"
        );
    }
}

#[test]
fn test_rope_apply_pair_out_of_bounds_position() {
    let rope = RotaryEmbedding::new(4, 8, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[1, 1, 2, 4], DType::F32, &Device::Cpu).unwrap();
    // Position 8 >= max_seq_len 8 — should error.
    let result = rope.apply_pair(&x, &x, &[0, 8]);
    assert!(
        result.is_err(),
        "position >= max_seq_len should be rejected"
    );
}

#[test]
fn test_rope_apply_pair_position_length_mismatch() {
    let rope = RotaryEmbedding::new(4, 32, 10000.0, &Device::Cpu).unwrap();
    // seq_len=3 but only 2 positions provided.
    let x = DynTensor::ones(&[1, 1, 3, 4], DType::F32, &Device::Cpu).unwrap();
    let result = rope.apply_pair(&x, &x, &[0, 1]);
    assert!(
        result.is_err(),
        "positions length != seq_len should be rejected"
    );
}

#[test]
fn test_rope_apply_pair_norm_preservation() {
    // RoPE rotation should preserve the L2 norm of each position vector.
    let rope = RotaryEmbedding::new(8, 32, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1, 1, 1, 8],
        &Device::Cpu,
    )
    .unwrap();
    let (rotated, _) = rope.apply_pair(&x, &x, &[5]).unwrap();
    let x_flat = x.to_flat_vec::<f32>().unwrap();
    let r_flat = rotated.to_flat_vec::<f32>().unwrap();
    let norm_x: f32 = x_flat.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_r: f32 = r_flat.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm_x - norm_r).abs() < 1e-4,
        "norm changed: {norm_x} -> {norm_r}"
    );
}
