#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::test_utils::tiny_config;
use nn_core::layers::repeat_kv;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

#[test]
fn test_config_validation_ok() {
    let cfg = tiny_config();
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.head_dim(), 128);
    assert_eq!(cfg.num_kv_groups().unwrap(), 1);
}

#[test]
fn test_config_validation_zero_heads() {
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validation_heads_not_divisible() {
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 3;
    cfg.num_key_value_heads = 2;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_gqa_groups() {
    let mut cfg = tiny_config();
    cfg.num_attention_heads = 8;
    cfg.num_key_value_heads = 2;
    assert_eq!(cfg.num_kv_groups().unwrap(), 4);
}

#[test]
fn test_causal_mask_shape() {
    let mask = causal_mask(4, DType::F32, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 4, 4]);
}

#[test]
fn test_causal_mask_values() {
    let mask = causal_mask(3, DType::F32, &Device::Cpu).unwrap();
    let data = mask.to_flat_vec::<f32>().unwrap();
    // Row 0: [0, -inf, -inf]
    assert_eq!(data[0], 0.0);
    assert!(data[1].is_infinite() && data[1] < 0.0);
    assert!(data[2].is_infinite() && data[2] < 0.0);
    // Row 1: [0, 0, -inf]
    assert_eq!(data[3], 0.0);
    assert_eq!(data[4], 0.0);
    assert!(data[5].is_infinite() && data[5] < 0.0);
    // Row 2: [0, 0, 0]
    assert_eq!(data[6], 0.0);
    assert_eq!(data[7], 0.0);
    assert_eq!(data[8], 0.0);
}

#[test]
fn test_causal_mask_single_token() {
    let mask = causal_mask(1, DType::F32, &Device::Cpu).unwrap();
    let data = mask.to_flat_vec::<f32>().unwrap();
    assert_eq!(data, &[0.0]);
}

#[test]
fn test_repeat_kv_identity() {
    // n_rep=1 should return the same tensor
    let x = DynTensor::ones(&[1, 2, 3, 128], DType::F32, &Device::Cpu).unwrap();
    let result = repeat_kv(&x, 1).unwrap();
    assert_eq!(result.dims(), &[1, 2, 3, 128]);
}

#[test]
fn test_repeat_kv_expand() {
    // 2 kv heads → 4 total heads (n_rep=2)
    let data: Vec<f32> = (0..2 * 2 * 4).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data, &[1, 2, 2, 4], &Device::Cpu).unwrap();
    let result = repeat_kv(&x, 2).unwrap();
    assert_eq!(result.dims(), &[1, 4, 2, 4]);

    // Check head 0 == head 1 (repeated from kv_head 0)
    let flat = result.to_flat_vec::<f32>().unwrap();
    let head_size = 2 * 4; // seq * head_dim
    assert_eq!(&flat[0..head_size], &flat[head_size..2 * head_size]);
}

#[test]
fn test_model_load_zeros() {
    // Load model with ZerosBackend — should succeed for structural test
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg);
    assert!(model.is_ok(), "model load failed: {:?}", model.err());
}

#[test]
fn test_model_forward_zeros() {
    // Forward with all-zero weights is degenerate but NOT non-finite.
    // RMSNorm uses `sqrt(mean(x^2) + eps)` — the eps (1e-6) prevents the
    // zero-input singularity. Result: 0/sqrt(eps) = 0, finite throughout.
    // Softmax on all-zero logits produces uniform 1/seq_len, also finite.
    // Zero weights * finite activations = zero outputs. All finite.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let input_ids = &[0, 1, 2];
    let positions = &[0, 1, 2];
    let result = model.forward(input_ids, positions);
    assert!(
        result.is_ok(),
        "zero-weight forward should succeed (eps prevents singularity): {:?}",
        result.err()
    );
    // All outputs should be finite (zeros or uniform softmax values).
    let output = result.unwrap();
    let vals = output.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "all output values should be finite with zero weights"
    );
}

#[test]
fn test_model_forward_single_token() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let logits = model.forward(&[42], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, 100]);
}

#[test]
fn test_model_forward_mismatched_lengths() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let err = model.forward(&[0, 1], &[0]);
    assert!(err.is_err());
}

#[test]
fn test_model_tied_embeddings() {
    // When tie_word_embeddings=true, lm_head shares embed_tokens weight
    let mut cfg = tiny_config();
    cfg.tie_word_embeddings = true;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    assert!(model.config().tie_word_embeddings);
}

#[test]
fn test_model_untied_embeddings() {
    let mut cfg = tiny_config();
    cfg.tie_word_embeddings = false;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg);
    assert!(model.is_ok());
}

#[test]
fn test_nan_input_rejected_by_finiteness_guard() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    // Create [1, 2, 256] embeddings with NaN injected
    let mut data = vec![0.0f32; 2 * 256];
    data[0] = f32::NAN;
    let nan_embeddings = DynTensor::from_vec(data, &[1, 2, 256], &Device::Cpu).unwrap();

    let err = model.forward_from_embeddings(&nan_embeddings, &[0, 1], None);
    assert!(
        err.is_err(),
        "NaN input should be rejected by finiteness guard"
    );
    let msg = format!("{}", err.unwrap_err());
    assert!(
        msg.contains("non-finite") || msg.contains("NonFinite") || msg.contains("NaN"),
        "error should mention non-finite: {msg}"
    );
}

// -- AC1-AC3, AC5: Validation guard tests (#1445) ----------------------------

#[test]
fn test_config_validation_zero_vocab_size() {
    let mut cfg = tiny_config();
    cfg.vocab_size = 0;
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("vocab_size"),
        "error should mention vocab_size: {msg}"
    );
}

#[test]
fn test_config_validation_zero_rms_norm_eps() {
    let mut cfg = tiny_config();
    cfg.rms_norm_eps = 0.0;
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("rms_norm_eps"),
        "error should mention rms_norm_eps: {msg}"
    );
}

#[test]
fn test_config_validation_negative_rms_norm_eps() {
    let mut cfg = tiny_config();
    cfg.rms_norm_eps = -1e-6;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validation_nan_rms_norm_eps() {
    let mut cfg = tiny_config();
    cfg.rms_norm_eps = f64::NAN;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validation_zero_rope_theta() {
    let mut cfg = tiny_config();
    cfg.rope_theta = 0.0;
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("rope_theta"),
        "error should mention rope_theta: {msg}"
    );
}

#[test]
fn test_config_validation_negative_rope_theta() {
    let mut cfg = tiny_config();
    cfg.rope_theta = -10_000.0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validation_inf_rope_theta() {
    let mut cfg = tiny_config();
    cfg.rope_theta = f64::INFINITY;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_causal_mask_total_less_than_new_returns_error() {
    let result = causal_mask_with_offset(5, 3, DType::F32, &Device::Cpu);
    assert!(result.is_err(), "total_tokens < new_tokens should error");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("total_tokens"),
        "error should mention total_tokens: {msg}"
    );
}

#[test]
fn test_config_validation_zero_max_position_embeddings() {
    let mut cfg = tiny_config();
    cfg.max_position_embeddings = 0;
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("max_position_embeddings"),
        "error should mention max_position_embeddings: {msg}"
    );
}

#[test]
fn test_num_kv_groups_zero_kv_heads_returns_error() {
    let mut cfg = tiny_config();
    cfg.num_key_value_heads = 0;
    assert!(
        cfg.num_kv_groups().is_err(),
        "num_kv_groups() should error on zero kv_heads"
    );
}

// KV cache, forward_from_embeddings, edge case, and YaRN tests extracted to
// qwen3_tests_cache.rs
#[path = "qwen3_tests_cache.rs"]
mod cache;

// generate_greedy() and generate_beam() convenience wrapper tests
#[path = "qwen3_tests_generation.rs"]
mod generation;

// Config constructors, builders, GQA arithmetic, RoPE theta, error types,
// SwiGLU properties, and architectural invariants
#[path = "qwen3_tests_config_and_arch.rs"]
mod config_and_arch;

// Architecture validation: production configs, parameter counts, shape propagation (#3942)
#[path = "qwen3_tests_arch_validation.rs"]
mod arch_validation;

// Expanded coverage: RoPE, SwiGLU, GQA shapes, cache stress, generation, MoE, multi-batch (#4296)
#[path = "qwen3_tests_expanded.rs"]
mod expanded;

// RoPE rotation correctness, GQA repeat_kv data verification, causal mask structure (#4353)
#[path = "qwen3_tests_rope_gqa.rs"]
mod rope_gqa;
