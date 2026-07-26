// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`MlaLayer`] and [`MlaConfig`].

use super::{MlaConfig, MlaLayer};
use crate::dyn_tensor::test_helpers::make_linear_seeded as make_linear;
use crate::dyn_tensor::DynTensor;
use crate::layers::RmsNorm;
use crate::{DType, Device};

/// Default test config matching a small DeepSeek-V2 style MLA.
fn test_config() -> MlaConfig {
    MlaConfig {
        hidden_size: 64,
        num_heads: 4,
        kv_lora_rank: 16,
        q_lora_rank: None,
        rope_dim: 8,
        qk_nope_dim: 8,
        v_head_dim: 8,
        rms_norm_eps: 1e-6,
    }
}

/// Config with Q compression enabled.
fn test_config_with_q_compression() -> MlaConfig {
    MlaConfig {
        q_lora_rank: Some(12),
        ..test_config()
    }
}

/// Create a deterministic MLA layer from config (no VarBuilder, direct construction).
fn make_mla(cfg: MlaConfig) -> MlaLayer {
    let qk_head_dim = cfg.qk_head_dim(); // qk_nope_dim + rope_dim

    // Q path
    let (q_a_proj, q_a_norm) = if let Some(q_lr) = cfg.q_lora_rank {
        let q_a = make_linear(q_lr, cfg.hidden_size, 10.0);
        let w = DynTensor::ones(&[q_lr], DType::F32, &Device::Cpu).unwrap();
        let q_a_n = RmsNorm::new(w, cfg.rms_norm_eps).unwrap();
        (Some(q_a), Some(q_a_n))
    } else {
        (None, None)
    };

    let q_b_in = cfg.q_lora_rank.unwrap_or(cfg.hidden_size);
    let q_b_proj = make_linear(cfg.num_heads * qk_head_dim, q_b_in, 20.0);

    // KV path
    let kv_a_out = cfg.kv_lora_rank + cfg.rope_dim;
    let kv_a_proj = make_linear(kv_a_out, cfg.hidden_size, 30.0);
    let kv_norm_w = DynTensor::ones(&[cfg.kv_lora_rank], DType::F32, &Device::Cpu).unwrap();
    let kv_a_norm = RmsNorm::new(kv_norm_w, cfg.rms_norm_eps).unwrap();
    let kv_b_out = cfg.num_heads * (cfg.qk_nope_dim + cfg.v_head_dim);
    let kv_b_proj = make_linear(kv_b_out, cfg.kv_lora_rank, 40.0);

    // Output
    let out_proj = make_linear(cfg.hidden_size, cfg.num_heads * cfg.v_head_dim, 50.0);

    let scale = 1.0 / (qk_head_dim as f64).sqrt();

    MlaLayer {
        q_a_proj,
        q_a_norm,
        q_b_proj,
        kv_a_proj,
        kv_a_norm,
        kv_b_proj,
        out_proj,
        cfg,
        scale,
    }
}

/// Make a deterministic input tensor.
fn make_input(batch: usize, seq: usize, dim: usize, seed: f32) -> DynTensor {
    let n = batch * seq * dim;
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + seed) * 0.017).sin() * 0.5)
        .collect();
    DynTensor::from_vec(data, &[batch, seq, dim], &Device::Cpu).unwrap()
}

/// Make cos/sin tensors for RoPE.
fn make_rope_cos_sin(seq_len: usize, rope_dim: usize) -> (DynTensor, DynTensor) {
    let half_dim = rope_dim / 2;
    let base = 10000.0f64;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| (1.0 / base.powf((2 * i) as f64 / rope_dim as f64)) as f32)
        .collect();

    let n = seq_len * half_dim;
    let mut cos_data = Vec::with_capacity(n);
    let mut sin_data = Vec::with_capacity(n);
    for pos in 0..seq_len {
        for &freq in &inv_freq {
            let angle = pos as f32 * freq;
            cos_data.push(angle.cos());
            sin_data.push(angle.sin());
        }
    }
    let cos = DynTensor::from_vec(cos_data, &[seq_len, half_dim], &Device::Cpu).unwrap();
    let sin = DynTensor::from_vec(sin_data, &[seq_len, half_dim], &Device::Cpu).unwrap();
    (cos, sin)
}

// -- Config validation --------------------------------------------------------

#[test]
fn test_config_validate_valid() {
    test_config().validate().expect("valid config");
}

#[test]
fn test_config_validate_with_q_compression() {
    test_config_with_q_compression()
        .validate()
        .expect("valid config with Q compression");
}

#[test]
fn test_config_validate_zero_heads() {
    let mut cfg = test_config();
    cfg.num_heads = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_zero_hidden() {
    let mut cfg = test_config();
    cfg.hidden_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_zero_kv_lora_rank() {
    let mut cfg = test_config();
    cfg.kv_lora_rank = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_zero_rope_dim() {
    let mut cfg = test_config();
    cfg.rope_dim = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_odd_rope_dim() {
    let mut cfg = test_config();
    cfg.rope_dim = 7;
    let err = cfg.validate().unwrap_err();
    assert!(
        err.to_string().contains("even"),
        "should mention even: {}",
        err
    );
}

#[test]
fn test_config_validate_zero_qk_nope_dim() {
    let mut cfg = test_config();
    cfg.qk_nope_dim = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_zero_v_head_dim() {
    let mut cfg = test_config();
    cfg.v_head_dim = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_zero_q_lora_rank() {
    let mut cfg = test_config();
    cfg.q_lora_rank = Some(0);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_negative_eps() {
    let mut cfg = test_config();
    cfg.rms_norm_eps = -1.0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_nan_eps() {
    let mut cfg = test_config();
    cfg.rms_norm_eps = f64::NAN;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_qk_head_dim() {
    let cfg = test_config();
    assert_eq!(cfg.qk_head_dim(), cfg.qk_nope_dim + cfg.rope_dim);
    assert_eq!(cfg.qk_head_dim(), 16);
}

// -- Forward pass shape correctness -------------------------------------------

#[test]
fn test_forward_output_shape_basic() {
    let cfg = test_config();
    let mla = make_mla(cfg);
    let hidden = make_input(2, 5, 64, 0.0);
    let (cos, sin) = make_rope_cos_sin(5, cfg.rope_dim);
    let out = mla.forward(&hidden, &cos, &sin, None).unwrap();
    assert_eq!(out.dims(), &[2, 5, 64]);
}

#[test]
fn test_forward_output_shape_single_token() {
    let cfg = test_config();
    let mla = make_mla(cfg);
    let hidden = make_input(1, 1, 64, 0.0);
    let (cos, sin) = make_rope_cos_sin(1, cfg.rope_dim);
    let out = mla.forward(&hidden, &cos, &sin, None).unwrap();
    assert_eq!(out.dims(), &[1, 1, 64]);
}

#[test]
fn test_forward_output_shape_with_q_compression() {
    let cfg = test_config_with_q_compression();
    let mla = make_mla(cfg);
    let hidden = make_input(2, 4, 64, 1.0);
    let (cos, sin) = make_rope_cos_sin(4, cfg.rope_dim);
    let out = mla.forward(&hidden, &cos, &sin, None).unwrap();
    assert_eq!(out.dims(), &[2, 4, 64]);
}

#[test]
fn test_forward_batch_1_long_seq() {
    let cfg = test_config();
    let mla = make_mla(cfg);
    let hidden = make_input(1, 32, 64, 2.0);
    let (cos, sin) = make_rope_cos_sin(32, cfg.rope_dim);
    let out = mla.forward(&hidden, &cos, &sin, None).unwrap();
    assert_eq!(out.dims(), &[1, 32, 64]);
}

// -- Value verification (determinism) -----------------------------------------

#[test]
fn test_forward_deterministic() {
    let cfg = test_config();
    let mla = make_mla(cfg);
    let hidden = make_input(1, 3, 64, 0.0);
    let (cos, sin) = make_rope_cos_sin(3, cfg.rope_dim);
    let out1 = mla.forward(&hidden, &cos, &sin, None).unwrap();
    let out2 = mla.forward(&hidden, &cos, &sin, None).unwrap();
    let v1 = out1.to_flat_vec::<f32>().unwrap();
    let v2 = out2.to_flat_vec::<f32>().unwrap();
    assert_eq!(v1.len(), v2.len());
    for (a, b) in v1.iter().zip(v2.iter()) {
        assert!((a - b).abs() < 1e-6, "non-deterministic: {a} vs {b}");
    }
}

#[test]
fn test_forward_finite_output() {
    let cfg = test_config();
    let mla = make_mla(cfg);
    let hidden = make_input(2, 5, 64, 0.0);
    let (cos, sin) = make_rope_cos_sin(5, cfg.rope_dim);
    let out = mla.forward(&hidden, &cos, &sin, None).unwrap();
    let flat = out.to_flat_vec::<f32>().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "output has NaN/Inf values"
    );
}

#[test]
fn test_forward_with_q_compression_finite() {
    let cfg = test_config_with_q_compression();
    let mla = make_mla(cfg);
    let hidden = make_input(1, 4, 64, 3.0);
    let (cos, sin) = make_rope_cos_sin(4, cfg.rope_dim);
    let out = mla.forward(&hidden, &cos, &sin, None).unwrap();
    let flat = out.to_flat_vec::<f32>().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "output has NaN/Inf values"
    );
}

// -- With causal mask ---------------------------------------------------------

#[test]
fn test_forward_with_causal_mask() {
    let cfg = test_config();
    let mla = make_mla(cfg);
    let hidden = make_input(1, 4, 64, 0.0);
    let (cos, sin) = make_rope_cos_sin(4, cfg.rope_dim);
    let mask = crate::layers::attention::causal_mask(4, &Device::Cpu).unwrap();
    let out = mla.forward(&hidden, &cos, &sin, Some(&mask)).unwrap();
    assert_eq!(out.dims(), &[1, 4, 64]);
    let flat = out.to_flat_vec::<f32>().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()), "output has NaN/Inf");
}

// -- Accessor methods ---------------------------------------------------------

#[test]
fn test_accessors() {
    let cfg = test_config();
    let mla = make_mla(cfg);
    assert_eq!(mla.num_heads(), 4);
    assert_eq!(mla.kv_lora_rank(), 16);
    assert_eq!(mla.rope_dim(), 8);
    assert_eq!(mla.qk_nope_dim(), 8);
    assert_eq!(mla.v_head_dim(), 8);
    assert_eq!(mla.config().hidden_size, 64);
}

// -- Different input shapes change output -------------------------------------

#[test]
fn test_different_inputs_different_outputs() {
    let cfg = test_config();
    let mla = make_mla(cfg);
    let h1 = make_input(1, 3, 64, 0.0);
    let h2 = make_input(1, 3, 64, 100.0);
    let (cos, sin) = make_rope_cos_sin(3, cfg.rope_dim);
    let out1 = mla.forward(&h1, &cos, &sin, None).unwrap();
    let out2 = mla.forward(&h2, &cos, &sin, None).unwrap();
    let v1 = out1.to_flat_vec::<f32>().unwrap();
    let v2 = out2.to_flat_vec::<f32>().unwrap();
    // Different inputs should produce different outputs.
    let any_different = v1.iter().zip(v2.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(any_different, "different inputs should yield different outputs");
}
