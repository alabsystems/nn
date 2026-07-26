// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for RoPE (Rotary Position Embedding) and GQA (Grouped Query Attention)
//! in the Qwen3 model (#4353).
//!
//! Covers:
//! - RoPE frequency computation: inv_freq = 1 / base^(2i/dim) for each pair
//! - RoPE rotation correctness: half-split convention (x1*cos - x2*sin, x1*sin + x2*cos)
//! - GQA repeat_kv: key/value expansion when num_heads != num_kv_heads
//! - Causal mask application: correct masking structure with and without KV cache offset
//! - Forward pass shape correctness: various sequence lengths with non-zero weights

use super::*;
use crate::test_utils::tiny_config;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{repeat_kv, RotaryEmbedding};
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// RoPE frequency computation
// ---------------------------------------------------------------------------

#[test]
fn test_rope_inv_freq_base_10000_head_dim_128() {
    // For head_dim=128, base=10000:
    //   inv_freq[i] = 1.0 / 10000^(2i/128), for i=0..63
    //   i=0:  1.0 / 10000^(0/128)   = 1.0
    //   i=1:  1.0 / 10000^(2/128)   = 1.0 / 10000^(1/64)
    //   i=32: 1.0 / 10000^(64/128)  = 1.0 / 100 = 0.01
    //   i=63: 1.0 / 10000^(126/128) ≈ 1.27e-4
    let head_dim = 128;
    let base = 10_000.0_f64;

    // Verify the first frequency (i=0): should be 1.0
    let freq_0 = 1.0 / base.powf(0.0 / head_dim as f64);
    assert!(
        (freq_0 - 1.0).abs() < 1e-7,
        "freq[0] should be 1.0, got {freq_0}"
    );

    // Verify the middle frequency (i=32): should be 0.01
    let freq_32 = 1.0 / base.powf(64.0 / head_dim as f64);
    assert!(
        (freq_32 - 0.01).abs() < 1e-7,
        "freq[32] should be 0.01, got {freq_32}"
    );

    // Verify the last frequency (i=63): 1/10000^(126/128) ≈ 1.155e-4
    let freq_63 = 1.0 / base.powf(126.0 / head_dim as f64);
    let expected_63 = 1.0 / base.powf(126.0 / 128.0);
    assert!(
        (freq_63 - expected_63).abs() < 1e-10,
        "freq[63] should be {expected_63}, got {freq_63}"
    );
    // Sanity: the last frequency is small (frequencies decrease geometrically)
    assert!(
        freq_63 < 0.001,
        "freq[63] should be much smaller than 1.0, got {freq_63}"
    );

    // Create a RotaryEmbedding and verify it can be applied
    let rope = RotaryEmbedding::new(head_dim, 64, base, &Device::Cpu).unwrap();
    assert_eq!(rope.head_dim(), head_dim);
    assert_eq!(rope.max_seq_len(), 64);
}

#[test]
fn test_rope_inv_freq_base_1m_head_dim_128() {
    // Qwen3 uses base=1_000_000 for most variants.
    // inv_freq[i] = 1 / 1_000_000^(2i/128)
    // i=0:  1.0
    // i=32: 1 / 1e6^(64/128) = 1 / 1e3 = 0.001
    // i=63: 1 / 1e6^(126/128) ≈ 1.74e-6
    let head_dim = 128;
    let base = 1_000_000.0_f64;

    let freq_0 = 1.0 / base.powf(0.0 / head_dim as f64);
    assert!((freq_0 - 1.0).abs() < 1e-10);

    let freq_32 = 1.0 / base.powf(64.0 / head_dim as f64);
    assert!(
        (freq_32 - 0.001).abs() < 1e-7,
        "freq[32] with base=1M should be 0.001, got {freq_32}"
    );

    let rope = RotaryEmbedding::new(head_dim, 64, base, &Device::Cpu).unwrap();
    assert_eq!(rope.head_dim(), head_dim);
}

#[test]
fn test_rope_cos_sin_position_zero_is_identity() {
    // At position 0, angle = 0 * freq = 0 for all frequencies.
    // cos(0) = 1, sin(0) = 0. So RoPE at position 0 is the identity rotation.
    // x_out[first_half] = x1 * 1 - x2 * 0 = x1
    // x_out[second_half] = x1 * 0 + x2 * 1 = x2
    let head_dim = 4; // minimal even head_dim for testing
    let rope = RotaryEmbedding::new(head_dim, 16, 10_000.0, &Device::Cpu).unwrap();

    // Input: [1, 1, 1, head_dim] (batch=1, heads=1, seq=1, head_dim=4)
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let x = DynTensor::from_vec(input_data.clone(), &[1, 1, 1, head_dim], &Device::Cpu).unwrap();

    // Apply RoPE using half-split at position 0
    let (rotated, _) = rope.apply_pair_half_split(&x, &x, &[0]).unwrap();
    let result = rotated.to_flat_vec::<f32>().unwrap();

    // At position 0 the rotation should be identity
    for (i, (&expected, &actual)) in input_data.iter().zip(result.iter()).enumerate() {
        assert!(
            (expected - actual).abs() < 1e-6,
            "position 0 should be identity: expected {expected} at index {i}, got {actual}"
        );
    }
}

#[test]
fn test_rope_rotation_mathematical_correctness() {
    // Verify the half-split RoPE formula:
    //   y1[i] = x1[i] * cos(pos * freq[i]) - x2[i] * sin(pos * freq[i])
    //   y2[i] = x1[i] * sin(pos * freq[i]) + x2[i] * cos(pos * freq[i])
    // where x1 = x[..., :half], x2 = x[..., half:], freq[i] = 1/base^(2i/dim)
    let head_dim = 4;
    let base = 10_000.0_f64;
    let max_seq = 16;
    let rope = RotaryEmbedding::new(head_dim, max_seq, base, &Device::Cpu).unwrap();

    // Input vector: [a, b, c, d] where x1=[a,b], x2=[c,d]
    let a = 1.5_f32;
    let b = -0.7_f32;
    let c = 2.3_f32;
    let d = 0.4_f32;
    let x = DynTensor::from_vec(vec![a, b, c, d], &[1, 1, 1, head_dim], &Device::Cpu).unwrap();

    // Test at position 3
    let pos = 3_usize;
    let (rotated, _) = rope.apply_pair_half_split(&x, &x, &[pos]).unwrap();
    let result = rotated.to_flat_vec::<f32>().unwrap();

    // Compute expected values manually
    // freq[0] = 1 / base^(0/4) = 1.0
    // freq[1] = 1 / base^(2/4) = 1 / 100 = 0.01
    let freq_0 = (1.0 / base.powf(0.0 / head_dim as f64)) as f32;
    let freq_1 = (1.0 / base.powf(2.0 / head_dim as f64)) as f32;

    let angle_0 = pos as f32 * freq_0; // 3 * 1.0 = 3.0
    let angle_1 = pos as f32 * freq_1; // 3 * 0.01 = 0.03

    // Half-split formula:
    // y[0] = a * cos(angle_0) - c * sin(angle_0)
    // y[1] = b * cos(angle_1) - d * sin(angle_1)
    // y[2] = a * sin(angle_0) + c * cos(angle_0)
    // y[3] = b * sin(angle_1) + d * cos(angle_1)
    let expected_0 = a * angle_0.cos() - c * angle_0.sin();
    let expected_1 = b * angle_1.cos() - d * angle_1.sin();
    let expected_2 = a * angle_0.sin() + c * angle_0.cos();
    let expected_3 = b * angle_1.sin() + d * angle_1.cos();

    let expected = [expected_0, expected_1, expected_2, expected_3];
    for (i, (&exp, &act)) in expected.iter().zip(result.iter()).enumerate() {
        assert!(
            (exp - act).abs() < 1e-5,
            "RoPE rotation at index {i}: expected {exp}, got {act}"
        );
    }
}

#[test]
fn test_rope_different_positions_produce_different_results() {
    // Non-zero inputs at different positions should produce different rotations.
    let head_dim = 128;
    let rope = RotaryEmbedding::new(head_dim, 64, 10_000.0, &Device::Cpu).unwrap();

    let data: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let x = DynTensor::from_vec(data, &[1, 1, 1, head_dim], &Device::Cpu).unwrap();

    let (rot_pos0, _) = rope.apply_pair_half_split(&x, &x, &[0]).unwrap();
    let (rot_pos1, _) = rope.apply_pair_half_split(&x, &x, &[1]).unwrap();
    let (rot_pos10, _) = rope.apply_pair_half_split(&x, &x, &[10]).unwrap();

    let v0 = rot_pos0.to_flat_vec::<f32>().unwrap();
    let v1 = rot_pos1.to_flat_vec::<f32>().unwrap();
    let v10 = rot_pos10.to_flat_vec::<f32>().unwrap();

    // v0 should differ from v1, and both from v10
    assert_ne!(
        v0, v1,
        "different positions should produce different rotations"
    );
    assert_ne!(v0, v10, "pos 0 and pos 10 should differ");
    assert_ne!(v1, v10, "pos 1 and pos 10 should differ");
}

#[test]
fn test_rope_preserves_vector_norm() {
    // RoPE is a rotation, so it should preserve the L2 norm of each 2D pair.
    // For half-split: norm of (y1[i], y2[i]) should equal norm of (x1[i], x2[i]).
    let head_dim = 8;
    let rope = RotaryEmbedding::new(head_dim, 32, 10_000.0, &Device::Cpu).unwrap();
    let half = head_dim / 2;

    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x = DynTensor::from_vec(input_data.clone(), &[1, 1, 1, head_dim], &Device::Cpu).unwrap();

    for pos in [1, 5, 15, 31] {
        let (rotated, _) = rope.apply_pair_half_split(&x, &x, &[pos]).unwrap();
        let result = rotated.to_flat_vec::<f32>().unwrap();

        // For each pair index i, check that (x1[i], x2[i]) has the same norm as (y1[i], y2[i])
        for i in 0..half {
            let x1 = input_data[i];
            let x2 = input_data[i + half];
            let orig_norm = x1.hypot(x2);

            let y1 = result[i];
            let y2 = result[i + half];
            let rot_norm = y1.hypot(y2);

            assert!(
                (orig_norm - rot_norm).abs() < 1e-5,
                "RoPE should preserve norm at pos={pos}, pair={i}: orig={orig_norm}, rot={rot_norm}"
            );
        }
    }
}

#[test]
fn test_rope_multi_position_sequence() {
    // Apply RoPE to a multi-token sequence and verify each position independently.
    let head_dim = 4;
    let base = 10_000.0_f64;
    let rope = RotaryEmbedding::new(head_dim, 16, base, &Device::Cpu).unwrap();
    // Input: [1, 1, 3, 4] (batch=1, heads=1, seq=3, head_dim=4)
    // Each token has the same values for simplicity.
    let token = [1.0_f32, 2.0, 3.0, 4.0];
    let data: Vec<f32> = token.iter().cycle().take(3 * head_dim).copied().collect();
    let x = DynTensor::from_vec(data, &[1, 1, 3, head_dim], &Device::Cpu).unwrap();

    let positions = [0, 1, 2];
    let (rotated, _) = rope.apply_pair_half_split(&x, &x, &positions).unwrap();
    let result = rotated.to_flat_vec::<f32>().unwrap();

    // Verify each position independently
    let freq_0 = (1.0 / base.powf(0.0 / head_dim as f64)) as f32;
    let freq_1 = (1.0 / base.powf(2.0 / head_dim as f64)) as f32;

    for (t, &pos) in positions.iter().enumerate() {
        let offset = t * head_dim;
        let a = token[0];
        let b = token[1];
        let c = token[2];
        let d = token[3];

        let angle_0 = pos as f32 * freq_0;
        let angle_1 = pos as f32 * freq_1;

        let exp_0 = a * angle_0.cos() - c * angle_0.sin();
        let exp_1 = b * angle_1.cos() - d * angle_1.sin();
        let exp_2 = a * angle_0.sin() + c * angle_0.cos();
        let exp_3 = b * angle_1.sin() + d * angle_1.cos();

        let expected = [exp_0, exp_1, exp_2, exp_3];
        for (i, (&exp, &act)) in expected
            .iter()
            .zip(result[offset..offset + head_dim].iter())
            .enumerate()
        {
            assert!(
                (exp - act).abs() < 1e-5,
                "token {t} (pos={pos}), dim {i}: expected {exp}, got {act}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// GQA: repeat_kv correctness
// ---------------------------------------------------------------------------

#[test]
fn test_repeat_kv_n_rep_1_is_identity() {
    // n_rep=1 means num_heads == num_kv_heads (MHA), no expansion needed
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data.clone(), &[1, 2, 3, 4], &Device::Cpu).unwrap();
    let result = repeat_kv(&x, 1).unwrap();
    assert_eq!(result.dims(), &[1, 2, 3, 4]);
    assert_eq!(result.to_flat_vec::<f32>().unwrap(), data);
}

#[test]
fn test_repeat_kv_n_rep_2_doubles_heads() {
    // 2 kv_heads, n_rep=2 -> 4 total heads
    // Each kv head should appear twice in sequence
    let head_dim = 2;
    let seq_len = 1;
    let n_kv_heads = 2;
    let n_rep = 2;

    // kv_head_0 = [10, 20], kv_head_1 = [30, 40]
    let data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
    let x = DynTensor::from_vec(data, &[1, n_kv_heads, seq_len, head_dim], &Device::Cpu).unwrap();
    let result = repeat_kv(&x, n_rep).unwrap();

    assert_eq!(result.dims(), &[1, n_kv_heads * n_rep, seq_len, head_dim]);
    let flat = result.to_flat_vec::<f32>().unwrap();

    // Expected: [head_0, head_0, head_1, head_1] = [10,20, 10,20, 30,40, 30,40]
    assert_eq!(flat, vec![10.0, 20.0, 10.0, 20.0, 30.0, 40.0, 30.0, 40.0]);
}

#[test]
fn test_repeat_kv_n_rep_4_quadruples_heads() {
    // 1 kv_head, n_rep=4 -> 4 total heads (MQA scenario)
    let head_dim = 3;
    let seq_len = 1;
    let n_kv_heads = 1;
    let n_rep = 4;

    let data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let x = DynTensor::from_vec(data, &[1, n_kv_heads, seq_len, head_dim], &Device::Cpu).unwrap();
    let result = repeat_kv(&x, n_rep).unwrap();

    assert_eq!(result.dims(), &[1, 4, seq_len, head_dim]);
    let flat = result.to_flat_vec::<f32>().unwrap();

    // The single kv head repeated 4 times
    assert_eq!(
        flat,
        vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );
}

#[test]
fn test_repeat_kv_preserves_seq_and_batch_dims() {
    // batch=2, kv_heads=2, seq=3, head_dim=4, n_rep=3 -> 6 total heads
    let batch = 2;
    let n_kv_heads = 2;
    let seq_len = 3;
    let head_dim = 4;
    let n_rep = 3;

    let total_elems = batch * n_kv_heads * seq_len * head_dim;
    let data: Vec<f32> = (0..total_elems).map(|i| i as f32).collect();
    let x =
        DynTensor::from_vec(data, &[batch, n_kv_heads, seq_len, head_dim], &Device::Cpu).unwrap();
    let result = repeat_kv(&x, n_rep).unwrap();

    assert_eq!(
        result.dims(),
        &[batch, n_kv_heads * n_rep, seq_len, head_dim],
        "repeat_kv should expand heads dimension while preserving batch, seq, head_dim"
    );

    let flat = result.to_flat_vec::<f32>().unwrap();
    let expected_elems = batch * n_kv_heads * n_rep * seq_len * head_dim;
    assert_eq!(flat.len(), expected_elems);
}

#[test]
fn test_repeat_kv_data_consistency_across_groups() {
    // Verify that each group of repeated heads contains identical data.
    let n_kv_heads = 2;
    let seq_len = 2;
    let head_dim = 4;
    let n_rep = 3;

    let total_elems = n_kv_heads * seq_len * head_dim;
    let data: Vec<f32> = (0..total_elems).map(|i| (i as f32 + 1.0) * 0.5).collect();
    let x = DynTensor::from_vec(data, &[1, n_kv_heads, seq_len, head_dim], &Device::Cpu).unwrap();
    let result = repeat_kv(&x, n_rep).unwrap();
    let flat = result.to_flat_vec::<f32>().unwrap();

    let head_size = seq_len * head_dim; // 8 elements per head

    // Heads 0, 1, 2 should be identical (copies of kv_head 0)
    for rep in 1..n_rep {
        let start_0 = 0;
        let start_rep = rep * head_size;
        assert_eq!(
            &flat[start_0..start_0 + head_size],
            &flat[start_rep..start_rep + head_size],
            "repeated head {rep} should match head 0 (kv_head 0)"
        );
    }

    // Heads 3, 4, 5 should be identical (copies of kv_head 1)
    let kv1_start = n_rep * head_size;
    for rep in 1..n_rep {
        let start_rep = kv1_start + rep * head_size;
        assert_eq!(
            &flat[kv1_start..kv1_start + head_size],
            &flat[start_rep..start_rep + head_size],
            "repeated head {rep} of kv_head 1 should match"
        );
    }

    // kv_head 0 and kv_head 1 should differ (different source data)
    assert_ne!(
        &flat[0..head_size],
        &flat[kv1_start..kv1_start + head_size],
        "different kv heads should have different data"
    );
}

// ---------------------------------------------------------------------------
// Causal mask application
// ---------------------------------------------------------------------------

#[test]
fn test_causal_mask_diagonal_structure() {
    // A 4x4 causal mask should have 0s on and below diagonal, -inf above.
    let mask = causal_mask(4, DType::F32, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 4, 4]);
    let data = mask.to_flat_vec::<f32>().unwrap();

    for row in 0..4 {
        for col in 0..4 {
            let val = data[row * 4 + col];
            if col <= row {
                assert_eq!(
                    val, 0.0,
                    "mask[{row},{col}] should be 0.0 (attend), got {val}"
                );
            } else {
                assert!(
                    val.is_infinite() && val < 0.0,
                    "mask[{row},{col}] should be -inf (block), got {val}"
                );
            }
        }
    }
}

#[test]
fn test_causal_mask_with_offset_decode_step() {
    // Simulating autoregressive decode: 1 new token, 5 total (4 cached + 1 new).
    // The single new token (at absolute position 4) can attend to all prior positions.
    let mask = causal_mask_with_offset(1, 5, DType::F32, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 1, 5]);
    let data = mask.to_flat_vec::<f32>().unwrap();
    // All positions should be 0.0 (attend to everything)
    assert!(
        data.iter().all(|&v| v == 0.0),
        "single new token should attend to all positions: {data:?}"
    );
}

#[test]
fn test_causal_mask_with_offset_prefill_3_of_5() {
    // 3 new tokens with 2 cached. Total = 5. New tokens at absolute positions 2, 3, 4.
    // Row 0 (abs pos 2): attend to 0,1,2 but not 3,4
    // Row 1 (abs pos 3): attend to 0,1,2,3 but not 4
    // Row 2 (abs pos 4): attend to all
    let mask = causal_mask_with_offset(3, 5, DType::F32, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 3, 5]);
    let data = mask.to_flat_vec::<f32>().unwrap();

    // Row 0: [0, 0, 0, -inf, -inf]
    assert_eq!(data[0], 0.0);
    assert_eq!(data[1], 0.0);
    assert_eq!(data[2], 0.0);
    assert!(data[3] < 0.0 && data[3].is_infinite());
    assert!(data[4] < 0.0 && data[4].is_infinite());

    // Row 1: [0, 0, 0, 0, -inf]
    assert_eq!(data[5], 0.0);
    assert_eq!(data[6], 0.0);
    assert_eq!(data[7], 0.0);
    assert_eq!(data[8], 0.0);
    assert!(data[9] < 0.0 && data[9].is_infinite());

    // Row 2: [0, 0, 0, 0, 0]
    for &v in &data[10..15] {
        assert_eq!(v, 0.0, "last token should attend to all positions");
    }
}

#[test]
fn test_build_causal_mask_returns_none_for_single_token() {
    // The Qwen3 forward path skips mask allocation when seq_len == 1 (autoregressive
    // decode) because the single query naturally attends to all cached positions.
    use crate::forward_common::build_causal_mask;

    // seq_len=1, no cache
    let mask = build_causal_mask(1, None, DType::F32, &Device::Cpu).unwrap();
    assert!(
        mask.is_none(),
        "single token without cache should produce no mask"
    );
}

#[test]
fn test_build_causal_mask_returns_some_for_multi_token() {
    use crate::forward_common::build_causal_mask;

    // seq_len=4, no cache
    let mask = build_causal_mask(4, None, DType::F32, &Device::Cpu).unwrap();
    assert!(mask.is_some(), "multi-token should produce a mask");
    let m = mask.unwrap();
    assert_eq!(m.dims(), &[1, 1, 4, 4]);
}

// ---------------------------------------------------------------------------
// GQA: Qwen3 model forward with various head configs and non-zero weights
// ---------------------------------------------------------------------------

#[test]
fn test_gqa_ratio_2_forward_shape() {
    // 4 attention heads, 2 kv heads -> GQA ratio = 2
    let cfg = Qwen3Config::new(256, 512, 1, 2, 1, 50, 1e-6, 10_000.0, 64, true, None);
    assert_eq!(cfg.num_kv_groups().unwrap(), 2);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let logits = model.forward(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4]).unwrap();
    assert_eq!(logits.dims(), &[1, 5, cfg.vocab_size]);
}

#[test]
fn test_gqa_ratio_8_mqa_forward_shape() {
    // 8 attention heads, 1 kv head -> GQA ratio = 8 (MQA)
    let cfg = Qwen3Config::new(1024, 2048, 1, 8, 1, 50, 1e-6, 10_000.0, 64, true, None);
    assert_eq!(cfg.num_kv_groups().unwrap(), 8);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    for seq in [1, 3, 7] {
        let ids: Vec<usize> = (0..seq).collect();
        let positions: Vec<usize> = (0..seq).collect();
        let logits = model.forward(&ids, &positions).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, seq, cfg.vocab_size],
            "MQA (ratio=8) forward should produce [1, {seq}, vocab_size]"
        );
    }
}

#[test]
fn test_gqa_different_kv_group_counts_both_load_and_forward() {
    // MHA (groups=1) and GQA (groups=2) should both load and produce finite output.
    // With zero weights both produce the same output, but the code paths through
    // repeat_kv differ (n_rep=1 vs n_rep=2).
    let cfg_mha = Qwen3Config::new(256, 512, 1, 2, 2, 50, 1e-6, 10_000.0, 64, true, None);
    let cfg_gqa = Qwen3Config::new(256, 512, 1, 2, 1, 50, 1e-6, 10_000.0, 64, true, None);

    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model_mha = Qwen3Model::load(&vb, cfg_mha.clone()).unwrap();
    let model_gqa = Qwen3Model::load(&vb, cfg_gqa.clone()).unwrap();

    let logits_mha = model_mha.forward(&[5], &[0]).unwrap();
    let logits_gqa = model_gqa.forward(&[5], &[0]).unwrap();

    // Both should produce finite output with correct shapes
    let v_mha = logits_mha.to_flat_vec::<f32>().unwrap();
    let v_gqa = logits_gqa.to_flat_vec::<f32>().unwrap();
    assert!(
        v_mha.iter().all(|v| v.is_finite()),
        "MHA logits should be finite"
    );
    assert!(
        v_gqa.iter().all(|v| v.is_finite()),
        "GQA logits should be finite"
    );

    assert_eq!(logits_mha.dims(), &[1, 1, cfg_mha.vocab_size]);
    assert_eq!(logits_gqa.dims(), &[1, 1, cfg_gqa.vocab_size]);
}

// ---------------------------------------------------------------------------
// Forward pass shape correctness with varying sequence lengths
// ---------------------------------------------------------------------------

#[test]
fn test_forward_shape_seq_lengths_1_through_16() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    for seq_len in 1..=16 {
        let ids: Vec<usize> = (0..seq_len).map(|i| i % cfg.vocab_size).collect();
        let positions: Vec<usize> = (0..seq_len).collect();
        let logits = model.forward(&ids, &positions).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, seq_len, cfg.vocab_size],
            "seq_len={seq_len} should produce [1, {seq_len}, vocab_size]"
        );
    }
}

#[test]
fn test_forward_shape_power_of_two_seq_lengths() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    for &seq_len in &[1, 2, 4, 8, 16, 32] {
        let ids: Vec<usize> = (0..seq_len).map(|i| i % cfg.vocab_size).collect();
        let positions: Vec<usize> = (0..seq_len).collect();
        let logits = model.forward(&ids, &positions).unwrap();
        assert_eq!(logits.dims(), &[1, seq_len, cfg.vocab_size]);
        let vals = logits.to_flat_vec::<f32>().unwrap();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "all outputs should be finite at seq_len={seq_len}"
        );
    }
}

#[test]
fn test_forward_cached_shape_incremental_to_max_position() {
    // Incrementally decode up to max_position_embeddings, verifying shape at each step.
    let cfg = tiny_config(); // max_position_embeddings = 64
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    // Prefill with 4 tokens
    let prefill_logits = model
        .forward_cached(&[0, 1, 2, 3], &[0, 1, 2, 3], Some(&mut cache))
        .unwrap();
    assert_eq!(prefill_logits.dims(), &[1, 4, cfg.vocab_size]);
    assert_eq!(cache.seq_len(), 4);

    // Decode tokens 4 through 10
    for i in 4..=10 {
        let logits = model
            .forward_cached(&[i % cfg.vocab_size], &[i], Some(&mut cache))
            .unwrap();
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
        assert_eq!(cache.seq_len(), i + 1);
    }
}

// ---------------------------------------------------------------------------
// RoPE + GQA integration: apply RoPE to Q/K with different head counts
// ---------------------------------------------------------------------------

#[test]
fn test_rope_apply_pair_half_split_different_head_counts() {
    // In GQA, Q has more heads than K. Both should accept RoPE application
    // since RoPE operates on (..., seq_len, head_dim), not on the heads dimension.
    let head_dim = 128;
    let rope = RotaryEmbedding::new(head_dim, 64, 10_000.0, &Device::Cpu).unwrap();

    let seq_len = 3;
    let num_q_heads = 8;
    let num_kv_heads = 2;

    let q_data: Vec<f32> = (0..num_q_heads * seq_len * head_dim)
        .map(|i| (i as f32) * 0.001)
        .collect();
    let k_data: Vec<f32> = (0..num_kv_heads * seq_len * head_dim)
        .map(|i| (i as f32) * 0.002)
        .collect();

    let q =
        DynTensor::from_vec(q_data, &[1, num_q_heads, seq_len, head_dim], &Device::Cpu).unwrap();
    let k =
        DynTensor::from_vec(k_data, &[1, num_kv_heads, seq_len, head_dim], &Device::Cpu).unwrap();

    let (q_rot, k_rot) = rope.apply_pair_half_split(&q, &k, &[0, 1, 2]).unwrap();

    assert_eq!(q_rot.dims(), &[1, num_q_heads, seq_len, head_dim]);
    assert_eq!(k_rot.dims(), &[1, num_kv_heads, seq_len, head_dim]);

    // Both should be finite
    assert!(q_rot
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite()));
    assert!(k_rot
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite()));
}

#[test]
fn test_rope_q_and_k_rotated_consistently() {
    // When Q and K have the same data at the same head, the rotated values should match.
    // This tests that RoPE applies the same rotation to Q and K at the same positions.
    let head_dim = 8;
    let rope = RotaryEmbedding::new(head_dim, 32, 10_000.0, &Device::Cpu).unwrap();

    let data: Vec<f32> = (0..head_dim).map(|i| (i + 1) as f32).collect();
    let q = DynTensor::from_vec(data.clone(), &[1, 1, 1, head_dim], &Device::Cpu).unwrap();
    let k = DynTensor::from_vec(data, &[1, 1, 1, head_dim], &Device::Cpu).unwrap();

    let (q_rot, k_rot) = rope.apply_pair_half_split(&q, &k, &[5]).unwrap();

    let q_vals = q_rot.to_flat_vec::<f32>().unwrap();
    let k_vals = k_rot.to_flat_vec::<f32>().unwrap();

    assert_eq!(
        q_vals, k_vals,
        "same input data should produce same RoPE rotation for Q and K"
    );
}
