#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`RotaryEmbedding`].

use super::RotaryEmbedding;
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device};

// ---------------------------------------------------------------------------
// Construction tests
// ---------------------------------------------------------------------------

#[test]
fn test_rope_new_basic() {
    let rope = RotaryEmbedding::new(64, 128, 10000.0, &Device::Cpu).unwrap();
    assert_eq!(rope.head_dim(), 64);
    assert_eq!(rope.max_seq_len(), 128);
}

#[test]
fn test_rope_rejects_odd_head_dim() {
    assert!(RotaryEmbedding::new(63, 128, 10000.0, &Device::Cpu).is_err());
}

#[test]
fn test_rope_rejects_zero_head_dim() {
    assert!(RotaryEmbedding::new(0, 128, 10000.0, &Device::Cpu).is_err());
}

#[test]
fn test_rope_rejects_zero_max_seq() {
    assert!(RotaryEmbedding::new(64, 0, 10000.0, &Device::Cpu).is_err());
}

#[test]
fn test_rope_rejects_invalid_base() {
    assert!(RotaryEmbedding::new(64, 128, 0.0, &Device::Cpu).is_err());
    assert!(RotaryEmbedding::new(64, 128, -1.0, &Device::Cpu).is_err());
    assert!(RotaryEmbedding::new(64, 128, f64::NAN, &Device::Cpu).is_err());
    assert!(RotaryEmbedding::new(64, 128, f64::INFINITY, &Device::Cpu).is_err());
}

// ---------------------------------------------------------------------------
// Apply tests — correctness
// ---------------------------------------------------------------------------

#[test]
fn test_rope_apply_identity_at_position_zero() {
    // At position 0, all frequencies are 0, so cos(0)=1, sin(0)=0.
    // RoPE at pos 0 should be close to identity.
    let head_dim = 4;
    let rope = RotaryEmbedding::new(head_dim, 16, 10000.0, &Device::Cpu).unwrap();

    let x = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, head_dim], &Device::Cpu).unwrap();

    let y = rope.apply(&x, 0).unwrap();
    assert_eq!(y.dims(), &[1, 1, head_dim]);

    let x_data = x.to_flat_vec::<f32>().unwrap();
    let y_data = y.to_flat_vec::<f32>().unwrap();

    // At position 0, cos(0*freq) = 1, sin(0*freq) = 0 → output = input
    for (&xv, &yv) in x_data.iter().zip(y_data.iter()) {
        assert!(
            (xv - yv).abs() < 1e-5,
            "position 0 should be identity: x={xv}, y={yv}"
        );
    }
}

#[test]
fn test_rope_apply_known_values() {
    // head_dim=2 with base=10000: one frequency pair.
    // inv_freq[0] = 1/(10000^(0/2)) = 1.0
    // At pos=1: angle = 1.0
    //   cos(1.0) ≈ 0.5403, sin(1.0) ≈ 0.8415
    //   y[0] = x[0]*cos(1) - x[1]*sin(1) = 1*0.5403 - 0*0.8415 = 0.5403
    //   y[1] = x[0]*sin(1) + x[1]*cos(1) = 1*0.8415 + 0*0.5403 = 0.8415
    let head_dim = 2;
    let rope = RotaryEmbedding::new(head_dim, 4, 10000.0, &Device::Cpu).unwrap();

    let x = DynTensor::from_vec(vec![1.0, 0.0], &[1, 1, head_dim], &Device::Cpu).unwrap();

    let y = rope.apply(&x, 1).unwrap(); // offset=1 → position 1
    let y_data = y.to_flat_vec::<f32>().unwrap();

    let cos_1 = 1.0_f32.cos();
    let sin_1 = 1.0_f32.sin();

    assert!(
        (y_data[0] - cos_1).abs() < 1e-5,
        "y[0] should be cos(1): got {}",
        y_data[0]
    );
    assert!(
        (y_data[1] - sin_1).abs() < 1e-5,
        "y[1] should be sin(1): got {}",
        y_data[1]
    );
}

#[test]
fn test_rope_norm_preservation() {
    // RoPE is a rotation — it should preserve the L2 norm of each (even, odd) pair.
    let head_dim = 8;
    let seq_len = 4;
    let rope = RotaryEmbedding::new(head_dim, 16, 10000.0, &Device::Cpu).unwrap();

    // Random-ish values
    let data: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i as f32 * 0.7).sin() * 3.0)
        .collect();
    let x = DynTensor::from_vec(data, &[1, seq_len, head_dim], &Device::Cpu).unwrap();

    let y = rope.apply(&x, 0).unwrap();

    let x_data = x.to_flat_vec::<f32>().unwrap();
    let y_data = y.to_flat_vec::<f32>().unwrap();

    // Check norm of each (even, odd) pair at each position
    for pos in 0..seq_len {
        for pair in 0..(head_dim / 2) {
            let idx = pos * head_dim + pair * 2;
            let x_norm = x_data[idx] * x_data[idx] + x_data[idx + 1] * x_data[idx + 1];
            let y_norm = y_data[idx] * y_data[idx] + y_data[idx + 1] * y_data[idx + 1];
            assert!(
                (x_norm - y_norm).abs() < 1e-4,
                "norm not preserved at pos={pos}, pair={pair}: x²={x_norm}, y²={y_norm}"
            );
        }
    }
}

#[test]
fn test_rope_apply_with_offset() {
    // Applying at offset=5 with seq_len=3 should use positions [5,6,7].
    // Verify this gives the same as applying at offset=0 with a longer seq
    // then taking positions [5,6,7].
    let head_dim = 4;
    let rope = RotaryEmbedding::new(head_dim, 16, 10000.0, &Device::Cpu).unwrap();

    let data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];

    // Approach 1: apply with offset=5
    let x = DynTensor::from_vec(data.clone(), &[1, 3, head_dim], &Device::Cpu).unwrap();
    let y_offset = rope.apply(&x, 5).unwrap();

    // Approach 2: create a longer input padded with zeros, apply at offset=0,
    // then extract positions [5..8]
    let mut padded_data = vec![0.0f32; 5 * head_dim];
    padded_data.extend_from_slice(&data);
    let x_padded = DynTensor::from_vec(padded_data, &[1, 8, head_dim], &Device::Cpu).unwrap();
    let y_full = rope.apply(&x_padded, 0).unwrap();
    let y_extracted = y_full.narrow(1, 5, 3).unwrap();

    let off_data = y_offset.to_flat_vec::<f32>().unwrap();
    let ext_data = y_extracted.to_flat_vec::<f32>().unwrap();

    for (&a, &b) in off_data.iter().zip(ext_data.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "offset apply should match extracted slice: {a} vs {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Shape tests
// ---------------------------------------------------------------------------

#[test]
fn test_rope_apply_2d_input() {
    // [seq_len, head_dim] — minimum valid input
    let rope = RotaryEmbedding::new(4, 8, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[3, 4], DType::F32, &Device::Cpu).unwrap();
    let y = rope.apply(&x, 0).unwrap();
    assert_eq!(y.dims(), &[3, 4]);
}

#[test]
fn test_rope_apply_3d_input() {
    // [batch, seq_len, head_dim]
    let rope = RotaryEmbedding::new(8, 16, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[2, 5, 8], DType::F32, &Device::Cpu).unwrap();
    let y = rope.apply(&x, 0).unwrap();
    assert_eq!(y.dims(), &[2, 5, 8]);
}

#[test]
fn test_rope_apply_4d_input() {
    // [batch, num_heads, seq_len, head_dim]
    let rope = RotaryEmbedding::new(8, 16, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[2, 4, 5, 8], DType::F32, &Device::Cpu).unwrap();
    let y = rope.apply(&x, 0).unwrap();
    assert_eq!(y.dims(), &[2, 4, 5, 8]);
}

#[test]
fn test_rope_output_is_finite() {
    let rope = RotaryEmbedding::new(16, 64, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::full(&[2, 4, 10, 16], 1.5, DType::F32, &Device::Cpu).unwrap();
    let y = rope.apply(&x, 0).unwrap();
    let data = y.to_flat_vec::<f32>().unwrap();
    for &v in &data {
        assert!(v.is_finite(), "RoPE output should be finite, got {v}");
    }
}

// ---------------------------------------------------------------------------
// Error tests
// ---------------------------------------------------------------------------

#[test]
fn test_rope_rejects_wrong_head_dim() {
    let rope = RotaryEmbedding::new(8, 16, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[1, 5, 6], DType::F32, &Device::Cpu).unwrap(); // head_dim=6 != 8
    assert!(rope.apply(&x, 0).is_err());
}

#[test]
fn test_rope_rejects_exceed_max_seq() {
    let rope = RotaryEmbedding::new(4, 8, 10000.0, &Device::Cpu).unwrap();
    // seq_len=10 > max_seq=8
    let x = DynTensor::ones(&[1, 10, 4], DType::F32, &Device::Cpu).unwrap();
    assert!(rope.apply(&x, 0).is_err());
}

#[test]
fn test_rope_rejects_offset_plus_seq_exceed_max() {
    let rope = RotaryEmbedding::new(4, 8, 10000.0, &Device::Cpu).unwrap();
    // offset=6 + seq_len=4 = 10 > max_seq=8
    let x = DynTensor::ones(&[1, 4, 4], DType::F32, &Device::Cpu).unwrap();
    assert!(rope.apply(&x, 6).is_err());
}

#[test]
fn test_rope_rejects_rank_1() {
    let rope = RotaryEmbedding::new(4, 8, 10000.0, &Device::Cpu).unwrap();
    let x = DynTensor::ones(&[4], DType::F32, &Device::Cpu).unwrap();
    assert!(rope.apply(&x, 0).is_err());
}

// ---------------------------------------------------------------------------
// Integration: RoPE + attention + KV cache
// ---------------------------------------------------------------------------

/// Helper: single-head attention step with RoPE and KV cache.
fn rope_attention_step(
    x: &DynTensor,
    wq: &crate::layers::Linear,
    wk: &crate::layers::Linear,
    wv: &crate::layers::Linear,
    rope: &RotaryEmbedding,
    cache: &mut crate::layers::KvCacheLayer,
    offset: usize,
    dim: usize,
) -> crate::Result<DynTensor> {
    use crate::layers::{softmax, Module};
    let (batch, seq) = (x.dim(0)?, x.dim(1)?);
    let x_flat = x.reshape([batch * seq, dim])?;
    let q = wq.forward(&x_flat)?.reshape([batch, seq, dim])?;
    let k = wk.forward(&x_flat)?.reshape([batch, seq, dim])?;
    let v = wv.forward(&x_flat)?.reshape([batch, seq, dim])?;

    let q_rot = rope.apply(&q, offset)?;
    let k_rot = rope.apply(&k, offset)?;

    let k_4d = k_rot.reshape([batch, 1, seq, dim])?;
    let v_4d = v.reshape([batch, 1, seq, dim])?;
    let (full_k, full_v) = cache.append(&k_4d, &v_4d)?;

    let kv_len = full_k.dim(2)?;
    let k_3d = full_k.reshape([batch, kv_len, dim])?;
    let v_3d = full_v.reshape([batch, kv_len, dim])?;
    let k_t = k_3d.transpose(1, 2)?;
    let scale = 1.0 / (dim as f64).sqrt();
    let scores = q_rot.matmul(&k_t)?.mul_scalar(scale)?;
    let weights = softmax(&scores, 2)?;
    weights.matmul(&v_3d)
}

#[test]
fn test_rope_attention_prefill() {
    use crate::layers::{KvCacheLayer, Linear};
    let dim = 8;
    let rope = RotaryEmbedding::new(dim, 32, 10000.0, &Device::Cpu).unwrap();
    let wq = Linear::new(
        DynTensor::ones(&[dim, dim], DType::F32, &Device::Cpu).unwrap(),
        None,
    )
    .unwrap();
    let wk = Linear::new(
        DynTensor::ones(&[dim, dim], DType::F32, &Device::Cpu).unwrap(),
        None,
    )
    .unwrap();
    let wv = Linear::new(
        DynTensor::ones(&[dim, dim], DType::F32, &Device::Cpu).unwrap(),
        None,
    )
    .unwrap();
    let mut cache = KvCacheLayer::empty();

    let x = DynTensor::ones(&[1, 4, dim], DType::F32, &Device::Cpu).unwrap();
    let out = rope_attention_step(&x, &wq, &wk, &wv, &rope, &mut cache, 0, dim).unwrap();
    assert_eq!(out.dims(), &[1, 4, dim]);
    assert_eq!(cache.seq_len(), 4);
}

#[test]
fn test_rope_attention_decode() {
    use crate::layers::{KvCacheLayer, Linear};
    let dim = 8;
    let rope = RotaryEmbedding::new(dim, 32, 10000.0, &Device::Cpu).unwrap();
    let wq = Linear::new(
        DynTensor::ones(&[dim, dim], DType::F32, &Device::Cpu).unwrap(),
        None,
    )
    .unwrap();
    let wk = Linear::new(
        DynTensor::ones(&[dim, dim], DType::F32, &Device::Cpu).unwrap(),
        None,
    )
    .unwrap();
    let wv = Linear::new(
        DynTensor::ones(&[dim, dim], DType::F32, &Device::Cpu).unwrap(),
        None,
    )
    .unwrap();
    let mut cache = KvCacheLayer::empty();

    // Prefill
    let x1 = DynTensor::ones(&[1, 4, dim], DType::F32, &Device::Cpu).unwrap();
    rope_attention_step(&x1, &wq, &wk, &wv, &rope, &mut cache, 0, dim).unwrap();

    // Decode: 1 token at offset=4
    let x2 = DynTensor::ones(&[1, 1, dim], DType::F32, &Device::Cpu).unwrap();
    let out2 = rope_attention_step(&x2, &wq, &wk, &wv, &rope, &mut cache, 4, dim).unwrap();
    assert_eq!(out2.dims(), &[1, 1, dim]);
    assert_eq!(cache.seq_len(), 5);

    let data = out2.to_flat_vec::<f32>().unwrap();
    for &v in &data {
        assert!(v.is_finite(), "output should be finite, got {v}");
    }
}

// Half-split rope free function tests and apply_pair/apply_at_positions tests
// are in rope_tests_extended.rs (extracted for file-size compliance).
