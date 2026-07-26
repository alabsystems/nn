#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`HalfRotaryEmbedding`].

use super::HalfRotaryEmbedding;
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device};

// ---------------------------------------------------------------------------
// Construction tests
// ---------------------------------------------------------------------------

#[test]
fn test_half_rope_new_basic() {
    let hrope = HalfRotaryEmbedding::new(128, 256, 1000000.0, &Device::Cpu).unwrap();
    assert_eq!(hrope.head_dim(), 128);
    assert_eq!(hrope.rope_dim(), 64);
    assert_eq!(hrope.max_seq_len(), 256);
}

#[test]
fn test_half_rope_rejects_non_multiple_of_4() {
    // head_dim=6: 6/2=3 which is odd — can't pair for RoPE
    assert!(HalfRotaryEmbedding::new(6, 128, 10000.0, &Device::Cpu).is_err());
    // head_dim=2: 2 % 4 != 0
    assert!(HalfRotaryEmbedding::new(2, 128, 10000.0, &Device::Cpu).is_err());
}

#[test]
fn test_half_rope_rejects_zero_head_dim() {
    assert!(HalfRotaryEmbedding::new(0, 128, 10000.0, &Device::Cpu).is_err());
}

#[test]
fn test_half_rope_minimum_head_dim() {
    // head_dim=4: 4/2=2 (even), valid
    let hrope = HalfRotaryEmbedding::new(4, 16, 10000.0, &Device::Cpu).unwrap();
    assert_eq!(hrope.head_dim(), 4);
    assert_eq!(hrope.rope_dim(), 2);
}

#[test]
fn test_half_rope_rejects_invalid_base() {
    assert!(HalfRotaryEmbedding::new(8, 128, 0.0, &Device::Cpu).is_err());
    assert!(HalfRotaryEmbedding::new(8, 128, -1.0, &Device::Cpu).is_err());
    assert!(HalfRotaryEmbedding::new(8, 128, f64::NAN, &Device::Cpu).is_err());
}

// ---------------------------------------------------------------------------
// Apply tests — correctness
// ---------------------------------------------------------------------------

#[test]
fn test_half_rope_identity_at_position_zero() {
    // At position 0, cos(0)=1, sin(0)=0 → rotated half is identity.
    // Pass-through half is always identity. So full output = input.
    let head_dim = 8;
    let hrope = HalfRotaryEmbedding::new(head_dim, 16, 10000.0, &Device::Cpu).unwrap();

    let data: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data.clone(), &[1, 1, head_dim], &Device::Cpu).unwrap();

    let y = hrope.apply(&x, 0).unwrap();
    let y_data = y.to_flat_vec::<f32>().unwrap();

    for (&xv, &yv) in data.iter().zip(y_data.iter()) {
        assert!(
            (xv - yv).abs() < 1e-5,
            "half-RoPE at pos 0 should be identity: x={xv}, y={yv}"
        );
    }
}

#[test]
fn test_half_rope_second_half_unchanged() {
    // The second half of head_dim should be unchanged at ANY position.
    let head_dim = 8;
    let hrope = HalfRotaryEmbedding::new(head_dim, 16, 10000.0, &Device::Cpu).unwrap();
    let rope_dim = head_dim / 2; // 4

    let data: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data.clone(), &[1, 1, head_dim], &Device::Cpu).unwrap();

    // Apply at a non-zero position where rotation is non-trivial
    let y = hrope.apply(&x, 3).unwrap();
    let y_data = y.to_flat_vec::<f32>().unwrap();

    // Second half (indices 4..8) should be unchanged
    for i in rope_dim..head_dim {
        assert!(
            (data[i] - y_data[i]).abs() < 1e-6,
            "second half should be unchanged: x[{i}]={}, y[{i}]={}",
            data[i],
            y_data[i]
        );
    }
}

#[test]
fn test_half_rope_first_half_rotated() {
    // At a non-zero position, the first half should differ from input
    // (unless the rotation angle happens to be a multiple of 2*pi).
    let head_dim = 8;
    let hrope = HalfRotaryEmbedding::new(head_dim, 16, 10000.0, &Device::Cpu).unwrap();
    let rope_dim = head_dim / 2;

    let data: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data.clone(), &[1, 1, head_dim], &Device::Cpu).unwrap();

    let y = hrope.apply(&x, 3).unwrap();
    let y_data = y.to_flat_vec::<f32>().unwrap();

    // At least one element in the first half should differ
    let any_changed = (0..rope_dim).any(|i| (data[i] - y_data[i]).abs() > 1e-5);
    assert!(
        any_changed,
        "first half should be rotated at non-zero position"
    );
}

#[test]
fn test_half_rope_norm_preservation_first_half() {
    // RoPE preserves L2 norm of each (even, odd) pair in the rotated half.
    let head_dim = 16;
    let seq_len = 4;
    let hrope = HalfRotaryEmbedding::new(head_dim, 32, 10000.0, &Device::Cpu).unwrap();
    let rope_dim = head_dim / 2;

    let data: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i as f32 * 0.7).sin() * 3.0)
        .collect();
    let x = DynTensor::from_vec(data.clone(), &[1, seq_len, head_dim], &Device::Cpu).unwrap();

    let y = hrope.apply(&x, 0).unwrap();
    let y_data = y.to_flat_vec::<f32>().unwrap();

    for pos in 0..seq_len {
        // Check norm of each pair in the rotated first half
        for pair in 0..(rope_dim / 2) {
            let idx = pos * head_dim + pair * 2;
            let x_norm = data[idx] * data[idx] + data[idx + 1] * data[idx + 1];
            let y_norm = y_data[idx] * y_data[idx] + y_data[idx + 1] * y_data[idx + 1];
            assert!(
                (x_norm - y_norm).abs() < 1e-4,
                "norm not preserved at pos={pos}, pair={pair}: x²={x_norm}, y²={y_norm}"
            );
        }
    }
}

#[test]
fn test_half_rope_matches_full_rope_on_first_half() {
    // The rotated portion should match what full RoPE would produce on that slice.
    use super::RotaryEmbedding;

    let head_dim = 8;
    let rope_dim = head_dim / 2;
    let base = 10000.0;

    let hrope = HalfRotaryEmbedding::new(head_dim, 16, base, &Device::Cpu).unwrap();
    let full_rope = RotaryEmbedding::new(rope_dim, 16, base, &Device::Cpu).unwrap();

    let data: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data.clone(), &[1, 1, head_dim], &Device::Cpu).unwrap();
    let x_first_half =
        DynTensor::from_vec(data[..rope_dim].to_vec(), &[1, 1, rope_dim], &Device::Cpu).unwrap();

    let y_half = hrope.apply(&x, 3).unwrap();
    let y_full_on_half = full_rope.apply(&x_first_half, 3).unwrap();

    let y_half_data = y_half.to_flat_vec::<f32>().unwrap();
    let y_full_data = y_full_on_half.to_flat_vec::<f32>().unwrap();

    // First `rope_dim` elements of half-RoPE output should match full RoPE on the first half
    for i in 0..rope_dim {
        assert!(
            (y_half_data[i] - y_full_data[i]).abs() < 1e-6,
            "half-RoPE first half should match full RoPE: idx={i}, half={}, full={}",
            y_half_data[i],
            y_full_data[i]
        );
    }
}

// ---------------------------------------------------------------------------
// Shape tests
// ---------------------------------------------------------------------------

#[test]
fn test_half_rope_apply_2d() {
    let hrope = HalfRotaryEmbedding::new(8, 16, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[3, 8], DType::F32, &Device::Cpu).unwrap();
    let y = hrope.apply(&x, 0).unwrap();
    assert_eq!(y.dims(), &[3, 8]);
}

#[test]
fn test_half_rope_apply_3d() {
    let hrope = HalfRotaryEmbedding::new(8, 16, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[2, 5, 8], DType::F32, &Device::Cpu).unwrap();
    let y = hrope.apply(&x, 0).unwrap();
    assert_eq!(y.dims(), &[2, 5, 8]);
}

#[test]
fn test_half_rope_apply_4d() {
    let hrope = HalfRotaryEmbedding::new(8, 16, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[2, 4, 5, 8], DType::F32, &Device::Cpu).unwrap();
    let y = hrope.apply(&x, 0).unwrap();
    assert_eq!(y.dims(), &[2, 4, 5, 8]);
}

#[test]
fn test_half_rope_output_is_finite() {
    let hrope = HalfRotaryEmbedding::new(16, 64, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::full(&[2, 4, 10, 16], 1.5, DType::F32, &Device::Cpu).unwrap();
    let y = hrope.apply(&x, 0).unwrap();
    let data = y.to_flat_vec::<f32>().unwrap();
    for &v in &data {
        assert!(v.is_finite(), "half-RoPE output should be finite, got {v}");
    }
}

// ---------------------------------------------------------------------------
// apply_pair tests
// ---------------------------------------------------------------------------

#[test]
fn test_half_rope_apply_pair() {
    let head_dim = 8;
    let hrope = HalfRotaryEmbedding::new(head_dim, 16, 10000.0, &Device::Cpu).unwrap();

    let q = DynTensor::ones(&[1, 2, 3, head_dim], DType::F32, &Device::Cpu).unwrap();
    let k = DynTensor::ones(&[1, 2, 3, head_dim], DType::F32, &Device::Cpu).unwrap();

    let (q_rot, k_rot) = hrope.apply_pair(&q, &k, &[0, 1, 2]).unwrap();
    assert_eq!(q_rot.dims(), &[1, 2, 3, head_dim]);
    assert_eq!(k_rot.dims(), &[1, 2, 3, head_dim]);
}

// ---------------------------------------------------------------------------
// Error tests
// ---------------------------------------------------------------------------

#[test]
fn test_half_rope_rejects_wrong_head_dim() {
    let hrope = HalfRotaryEmbedding::new(8, 16, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[1, 5, 12], DType::F32, &Device::Cpu).unwrap();
    assert!(hrope.apply(&x, 0).is_err());
}

#[test]
fn test_half_rope_rejects_rank_1() {
    let hrope = HalfRotaryEmbedding::new(8, 16, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[8], DType::F32, &Device::Cpu).unwrap();
    assert!(hrope.apply(&x, 0).is_err());
}

#[test]
fn test_half_rope_rejects_exceed_max_seq() {
    let hrope = HalfRotaryEmbedding::new(8, 4, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[1, 6, 8], DType::F32, &Device::Cpu).unwrap();
    assert!(hrope.apply(&x, 0).is_err());
}

#[test]
fn test_half_rope_offset_consistency() {
    // apply(x, offset=5) should match apply_pair positions=[5,6,7]
    let head_dim = 8;
    let hrope = HalfRotaryEmbedding::new(head_dim, 16, 10000.0, &Device::Cpu).unwrap();

    let data: Vec<f32> = (0..3 * head_dim).map(|i| (i as f32 * 0.3).cos()).collect();
    let x = DynTensor::from_vec(data, &[1, 1, 3, head_dim], &Device::Cpu).unwrap();

    let y_offset = hrope.apply(&x, 5).unwrap();
    let (y_pair, _) = hrope.apply_pair(&x, &x, &[5, 6, 7]).unwrap();

    let off_data = y_offset.to_flat_vec::<f32>().unwrap();
    let pair_data = y_pair.to_flat_vec::<f32>().unwrap();

    for (&a, &b) in off_data.iter().zip(pair_data.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "offset and positions should match: {a} vs {b}"
        );
    }
}
