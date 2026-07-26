#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! YaRN RoPE scaling tests (#1230).

use super::{RotaryEmbedding, YarnScaling};
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device};

/// Qwen3 production YaRN config: 40960 → 131072 tokens.
fn qwen3_yarn() -> YarnScaling {
    YarnScaling {
        factor: 4.0,
        attention_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        original_max_position_embeddings: 40_960,
    }
}

#[test]
fn test_yarn_basic_construction() {
    let yarn = qwen3_yarn();
    let rope = RotaryEmbedding::new_yarn(128, 131_072, 1_000_000.0, &yarn, &Device::Cpu);
    assert!(rope.is_ok());
    let rope = rope.unwrap();
    assert_eq!(rope.head_dim(), 128);
    assert_eq!(rope.max_seq_len(), 131_072);
}

#[test]
fn test_yarn_factor_1_matches_standard() {
    // factor=1.0 means no scaling — should produce identical results to standard RoPE.
    let head_dim = 8;
    let max_seq = 64;
    let base = 10_000.0;
    let identity_yarn = YarnScaling {
        factor: 1.0,
        attention_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        original_max_position_embeddings: 64,
    };
    let standard = RotaryEmbedding::new(head_dim, max_seq, base, &Device::Cpu).unwrap();
    let yarn =
        RotaryEmbedding::new_yarn(head_dim, max_seq, base, &identity_yarn, &Device::Cpu).unwrap();
    let x = DynTensor::from_vec(
        (0..head_dim).map(|i| i as f32 * 0.1).collect(),
        &[1, head_dim],
        &Device::Cpu,
    )
    .unwrap();
    let s_out = standard.apply(&x, 10).unwrap();
    let y_out = yarn.apply(&x, 10).unwrap();
    let s_flat = s_out.to_flat_vec::<f32>().unwrap();
    let y_flat = y_out.to_flat_vec::<f32>().unwrap();
    for (i, (&s, &y)) in s_flat.iter().zip(y_flat.iter()).enumerate() {
        assert!(
            (s - y).abs() < 1e-5,
            "factor=1 mismatch at {i}: standard={s}, yarn={y}"
        );
    }
}

#[test]
fn test_yarn_changes_frequencies_vs_standard() {
    // With factor > 1, YaRN should produce different frequencies than standard RoPE.
    let head_dim = 64;
    let max_seq = 256;
    let base = 1_000_000.0;
    let yarn_cfg = qwen3_yarn();
    let standard = RotaryEmbedding::new(head_dim, max_seq, base, &Device::Cpu).unwrap();
    let yarn = RotaryEmbedding::new_yarn(head_dim, max_seq, base, &yarn_cfg, &Device::Cpu).unwrap();
    let x = DynTensor::from_vec(
        (0..head_dim).map(|i| (i as f32 + 1.0) * 0.01).collect(),
        &[1, head_dim],
        &Device::Cpu,
    )
    .unwrap();
    // At high positions, scaled frequencies diverge more.
    let s_out = standard.apply(&x, 200).unwrap();
    let y_out = yarn.apply(&x, 200).unwrap();
    let s_flat = s_out.to_flat_vec::<f32>().unwrap();
    let y_flat = y_out.to_flat_vec::<f32>().unwrap();
    // At least some dimensions must differ (low-freq dims get scaled).
    let mut max_diff: f32 = 0.0;
    for (&s, &y) in s_flat.iter().zip(y_flat.iter()) {
        max_diff = max_diff.max((s - y).abs());
    }
    assert!(
        max_diff > 1e-6,
        "YaRN factor=4 should differ from standard at pos=200, but max_diff={max_diff}"
    );
}

#[test]
fn test_yarn_preserves_norm() {
    // RoPE is a rotation — it should preserve input norm, even with YaRN scaling.
    // Note: when attention_factor != 1, the norm changes by attention_factor.
    let head_dim = 16;
    let max_seq = 128;
    let yarn_cfg = YarnScaling {
        factor: 4.0,
        attention_factor: 1.0, // no attention scaling
        beta_fast: 32.0,
        beta_slow: 1.0,
        original_max_position_embeddings: 32,
    };
    let rope =
        RotaryEmbedding::new_yarn(head_dim, max_seq, 10_000.0, &yarn_cfg, &Device::Cpu).unwrap();
    let data: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let x = DynTensor::from_vec(data.clone(), &[1, head_dim], &Device::Cpu).unwrap();
    let y = rope.apply(&x, 50).unwrap();
    let norm_x: f32 = data.iter().map(|v| v * v).sum::<f32>().sqrt();
    let y_flat = y.to_flat_vec::<f32>().unwrap();
    let norm_y: f32 = y_flat.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm_x - norm_y).abs() < 1e-4,
        "YaRN should preserve norm: {norm_x} vs {norm_y}"
    );
}

#[test]
fn test_yarn_invalid_factor() {
    let bad_yarn = YarnScaling {
        factor: 0.0,
        attention_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        original_max_position_embeddings: 64,
    };
    let result = RotaryEmbedding::new_yarn(8, 64, 10_000.0, &bad_yarn, &Device::Cpu);
    assert!(result.is_err(), "factor=0 should be rejected");
}

#[test]
fn test_yarn_apply_pair_works() {
    let yarn_cfg = qwen3_yarn();
    let rope = RotaryEmbedding::new_yarn(8, 256, 1_000_000.0, &yarn_cfg, &Device::Cpu).unwrap();
    let q = DynTensor::ones(&[1, 2, 3, 8], DType::F32, &Device::Cpu).unwrap();
    let k = DynTensor::ones(&[1, 2, 3, 8], DType::F32, &Device::Cpu).unwrap();
    let (q_rot, k_rot) = rope.apply_pair(&q, &k, &[0, 1, 2]).unwrap();
    assert_eq!(q_rot.dims(), &[1, 2, 3, 8]);
    assert_eq!(k_rot.dims(), &[1, 2, 3, 8]);
}

#[test]
fn test_yarn_attention_factor_scales_output() {
    // attention_factor should scale cos/sin, affecting output magnitude.
    let head_dim = 8;
    let max_seq = 64;
    let base = 10_000.0;
    let yarn_1x = YarnScaling {
        factor: 2.0,
        attention_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        original_max_position_embeddings: 32,
    };
    let yarn_2x = YarnScaling {
        attention_factor: 2.0,
        ..yarn_1x
    };
    let rope_1x =
        RotaryEmbedding::new_yarn(head_dim, max_seq, base, &yarn_1x, &Device::Cpu).unwrap();
    let rope_2x =
        RotaryEmbedding::new_yarn(head_dim, max_seq, base, &yarn_2x, &Device::Cpu).unwrap();
    let x = DynTensor::from_vec(vec![1.0; head_dim], &[1, head_dim], &Device::Cpu).unwrap();
    let out_1x = rope_1x.apply(&x, 10).unwrap().to_flat_vec::<f32>().unwrap();
    let out_2x = rope_2x.apply(&x, 10).unwrap().to_flat_vec::<f32>().unwrap();
    // With attention_factor=2, the cos/sin are doubled, so outputs differ.
    let mut any_differ = false;
    for (&a, &b) in out_1x.iter().zip(out_2x.iter()) {
        if (a - b).abs() > 1e-5 {
            any_differ = true;
            break;
        }
    }
    assert!(
        any_differ,
        "attention_factor=2 should produce different output"
    );
}

#[test]
fn test_yarn_positions_beyond_40k() {
    // Verify YaRN generates correct output for positions > original context.
    let yarn_cfg = qwen3_yarn();
    let rope =
        RotaryEmbedding::new_yarn(128, 131_072, 1_000_000.0, &yarn_cfg, &Device::Cpu).unwrap();

    let x = DynTensor::from_vec(
        (0..128).map(|i| (i as f32 + 1.0) * 0.01).collect(),
        &[1, 128],
        &Device::Cpu,
    )
    .unwrap();
    let y = rope.apply(&x, 50_000).unwrap();
    assert_eq!(y.dims(), &[1, 128]);
    let y_flat = y.to_flat_vec::<f32>().unwrap();
    assert!(
        y_flat.iter().all(|v| v.is_finite()),
        "output at pos 50000 must be finite"
    );

    let y_high = rope.apply(&x, 130_000).unwrap();
    let y_high_flat = y_high.to_flat_vec::<f32>().unwrap();
    assert!(
        y_high_flat.iter().all(|v| v.is_finite()),
        "output at pos 130000 must be finite"
    );

    let max_diff: f32 = y_flat
        .iter()
        .zip(y_high_flat.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff > 1e-6,
        "pos 50000 vs 130000 should differ, max_diff={max_diff}"
    );
}

#[test]
fn test_yarn_frequency_interpolation_reference() {
    // Verify frequency interpolation matches YaRN reference formula.
    let head_dim = 8;
    let half_dim = head_dim / 2;
    let base: f64 = 10_000.0;
    let factor: f64 = 4.0;
    let beta_fast: f64 = 32.0;
    let beta_slow: f64 = 1.0;
    let orig_ctx: f64 = 64.0;
    let max_seq = 256;
    let pos = 100usize;

    let yarn_cfg = YarnScaling {
        factor,
        attention_factor: 1.0,
        beta_fast,
        beta_slow,
        original_max_position_embeddings: orig_ctx as usize,
    };

    let rope = RotaryEmbedding::new_yarn(head_dim, max_seq, base, &yarn_cfg, &Device::Cpu).unwrap();

    // Manually compute expected frequencies using reference formula.
    let low_freq_wavelen = orig_ctx / beta_fast;
    let high_freq_wavelen = orig_ctx / beta_slow;
    let wavelen_range = (high_freq_wavelen - low_freq_wavelen).max(1e-12);

    let mut expected_cos = Vec::with_capacity(half_dim);
    let mut expected_sin = Vec::with_capacity(half_dim);

    for i in 0..half_dim {
        let exponent = (2 * i) as f64 / head_dim as f64;
        let freq = 1.0 / base.powf(exponent);
        let wavelen = 2.0 * std::f64::consts::PI / freq;
        let ramp = ((wavelen - low_freq_wavelen) / wavelen_range).clamp(0.0, 1.0);
        let scaled_freq = (1.0 - ramp) * freq + ramp * (freq / factor);
        let angle = (pos as f64 * scaled_freq) as f32;
        expected_cos.push(angle.cos());
        expected_sin.push(angle.sin());
    }

    // Apply and compare against manual computation.
    let x_data: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.5).collect();
    let x = DynTensor::from_vec(x_data.clone(), &[1, head_dim], &Device::Cpu).unwrap();
    let y = rope.apply(&x, pos).unwrap();
    let y_flat = y.to_flat_vec::<f32>().unwrap();

    for i in 0..half_dim {
        let x_even = x_data[2 * i];
        let x_odd = x_data[2 * i + 1];
        let c = expected_cos[i];
        let s = expected_sin[i];
        let exp_even = x_even * c - x_odd * s;
        let exp_odd = x_even * s + x_odd * c;
        assert!(
            (y_flat[2 * i] - exp_even).abs() < 1e-4,
            "dim {}: even mismatch: got {} expected {exp_even}",
            2 * i,
            y_flat[2 * i]
        );
        assert!(
            (y_flat[2 * i + 1] - exp_odd).abs() < 1e-4,
            "dim {}: odd mismatch: got {} expected {exp_odd}",
            2 * i + 1,
            y_flat[2 * i + 1]
        );
    }
}
