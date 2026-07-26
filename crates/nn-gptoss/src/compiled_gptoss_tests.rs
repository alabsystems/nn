// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for CompiledGptOss, GenerationOutput, and InferenceSession.
//!
//! Uses CPU device with tiny model configs to avoid requiring Metal GPU
//! or real 20B-parameter weights in CI.

use super::*;
use nn_core::var_builder::VarBuilder;

/// Build a tiny gpt-oss config suitable for CPU unit tests.
///
/// 2 layers, small dims, 2 experts with top-1 routing. Produces a model
/// with ~hundreds of KB of parameters instead of ~78 GB.
fn tiny_config() -> GptOssConfig {
    GptOssConfig::gptoss_20b()
        .with_vocab_size(64)
        .with_num_hidden_layers(2)
        .with_num_local_experts(2)
        .with_experts_per_token(1)
}

/// Load a tiny model with zero weights on CPU for unit testing.
fn tiny_model_cpu() -> (GptOssModel, GptOssConfig) {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = GptOssModel::load(&vb, cfg.clone()).expect("tiny model should load from zeros");
    (model, cfg)
}

/// Build a CompiledGptOss from a tiny zero-weight model on CPU.
fn tiny_compiled() -> CompiledGptOss {
    let (model, config) = tiny_model_cpu();
    CompiledGptOss::from_model(model, config)
}

// -- GenerationOutput tests ------------------------------------------------

#[test]
fn test_generation_output_new() {
    let out = GenerationOutput::new(vec![1, 2, 3], 10, 3, 100.0, 40.0, 60.0, 50.0);
    assert_eq!(out.tokens, vec![1, 2, 3]);
    assert_eq!(out.prompt_tokens, 10);
    assert_eq!(out.generated_tokens, 3);
    assert!((out.total_time_ms - 100.0).abs() < f64::EPSILON);
    assert!((out.prefill_time_ms - 40.0).abs() < f64::EPSILON);
    assert!((out.decode_time_ms - 60.0).abs() < f64::EPSILON);
    assert!((out.tokens_per_second - 50.0).abs() < f64::EPSILON);
}

#[test]
fn test_generation_output_empty() {
    let out = GenerationOutput::new(Vec::new(), 0, 0, 0.0, 0.0, 0.0, 0.0);
    assert!(out.tokens.is_empty());
    assert_eq!(out.generated_tokens, 0);
}

// -- CompiledGptOss construction tests -------------------------------------

#[test]
fn test_compiled_from_model_cpu() {
    let compiled = tiny_compiled();
    assert_eq!(*compiled.device(), Device::Cpu);
    assert_eq!(compiled.dtype(), DType::F32);
    assert_eq!(compiled.config().num_hidden_layers, 2);
    assert_eq!(compiled.config().vocab_size, 64);
}

#[test]
fn test_compiled_config_accessors() {
    let compiled = tiny_compiled();
    assert_eq!(compiled.config().num_local_experts, 2);
    assert_eq!(compiled.config().experts_per_token, 1);
    assert_eq!(compiled.config().hidden_size, 2880);
}

#[test]
fn test_compiled_model_accessor() {
    let compiled = tiny_compiled();
    assert_eq!(compiled.model().dtype(), DType::F32);
    assert_eq!(compiled.model().device(), Device::Cpu);
}

// -- Forward pass tests ----------------------------------------------------

#[test]
fn test_forward_shape_single_token() {
    let compiled = tiny_compiled();
    let logits = compiled.forward(&[0]).expect("forward should succeed");
    let dims = logits.dims();
    assert_eq!(dims.len(), 3);
    assert_eq!(dims[0], 1);
    assert_eq!(dims[1], 1);
    assert_eq!(dims[2], 64);
}

#[test]
fn test_forward_shape_multi_token() {
    let compiled = tiny_compiled();
    let logits = compiled
        .forward(&[0, 1, 2])
        .expect("forward should succeed");
    let dims = logits.dims();
    assert_eq!(dims, &[1, 3, 64]);
}

#[test]
fn test_forward_logits_finite() {
    let compiled = tiny_compiled();
    let logits = compiled.forward(&[0, 1]).expect("forward should succeed");
    let flat = logits.to_flat_vec::<f32>().expect("should flatten");
    for &v in &flat {
        assert!(v.is_finite(), "logit {v} is not finite");
    }
}

// -- Generation tests ------------------------------------------------------

#[test]
fn test_generate_empty_prompt_returns_empty() {
    let compiled = tiny_compiled();
    let cfg = GenerateConfig::greedy(10);
    let out = compiled.generate(&[], &cfg).expect("should succeed");
    assert!(out.tokens.is_empty());
    assert_eq!(out.prompt_tokens, 0);
    assert_eq!(out.generated_tokens, 0);
}

#[test]
fn test_generate_zero_max_tokens_returns_empty() {
    let compiled = tiny_compiled();
    let cfg = GenerateConfig::greedy(0);
    let out = compiled.generate(&[1, 2], &cfg).expect("should succeed");
    assert!(out.tokens.is_empty());
    assert_eq!(out.prompt_tokens, 2);
}

#[test]
fn test_generate_greedy_produces_tokens() {
    let compiled = tiny_compiled();
    let cfg = GenerateConfig::greedy(5);
    let out = compiled.generate(&[0], &cfg).expect("should succeed");
    assert!(out.generated_tokens <= 5);
    assert_eq!(out.prompt_tokens, 1);
    assert!(out.total_time_ms >= 0.0);
    assert!(out.prefill_time_ms >= 0.0);
    assert!(out.decode_time_ms >= 0.0);
}

#[test]
fn test_generate_timing_fields_nonnegative() {
    let compiled = tiny_compiled();
    let cfg = GenerateConfig::greedy(3);
    let out = compiled.generate(&[0, 1], &cfg).expect("should succeed");
    assert!(out.total_time_ms >= 0.0);
    assert!(out.prefill_time_ms >= 0.0);
    assert!(out.decode_time_ms >= 0.0);
    assert!(out.tokens_per_second >= 0.0);
}

#[test]
fn test_generate_sampled_produces_tokens() {
    let compiled = tiny_compiled();
    let gen_cfg = GenerateConfig::default();
    let samp_cfg = SamplingConfig::default();
    let out = compiled
        .generate_sampled(&[0], &gen_cfg, &samp_cfg)
        .expect("should succeed");
    assert!(out.generated_tokens <= gen_cfg.max_tokens);
    assert_eq!(out.prompt_tokens, 1);
}

// -- InferenceSession tests ------------------------------------------------

#[test]
fn test_session_initial_state() {
    let compiled = tiny_compiled();
    let session = compiled.new_session();
    assert_eq!(session.seq_len(), 0);
}

#[test]
fn test_session_step_advances_position() {
    let compiled = tiny_compiled();
    let mut session = compiled.new_session();
    assert_eq!(session.seq_len(), 0);

    let _logits = session.step(&[0, 1]).expect("step should succeed");
    assert_eq!(session.seq_len(), 2);

    let _logits = session.step(&[2]).expect("step should succeed");
    assert_eq!(session.seq_len(), 3);
}

#[test]
fn test_session_step_returns_correct_shape() {
    let compiled = tiny_compiled();
    let mut session = compiled.new_session();
    let logits = session.step(&[0, 1, 2]).expect("step should succeed");
    let dims = logits.dims();
    assert_eq!(dims, &[1, 3, 64]);
}

#[test]
fn test_session_reset_clears_state() {
    let compiled = tiny_compiled();
    let mut session = compiled.new_session();
    let _ = session.step(&[0, 1]).expect("step should succeed");
    assert_eq!(session.seq_len(), 2);

    session.reset();
    assert_eq!(session.seq_len(), 0);
}

#[test]
fn test_session_cache_accessible() {
    let compiled = tiny_compiled();
    let session = compiled.new_session();
    let cache = session.cache();
    assert_eq!(cache.num_layers(), 2);
}

#[test]
fn test_session_multi_turn() {
    let compiled = tiny_compiled();
    let mut session = compiled.new_session();

    // Turn 1: process prompt
    let logits1 = session.step(&[0, 1, 2]).expect("turn 1 should succeed");
    assert_eq!(session.seq_len(), 3);
    assert_eq!(logits1.dims(), &[1, 3, 64]);

    // Turn 2: process reply (single token)
    let logits2 = session.step(&[5]).expect("turn 2 should succeed");
    assert_eq!(session.seq_len(), 4);
    assert_eq!(logits2.dims(), &[1, 1, 64]);
}

// -- Device selection tests ------------------------------------------------

#[test]
fn test_default_device_and_dtype_valid() {
    let (device, dtype) = default_device_and_dtype();
    match device {
        Device::Cpu => assert_eq!(dtype, DType::F32),
        Device::Metal { .. } => assert_eq!(dtype, DType::BF16),
        _ => panic!("unexpected device: {device}"),
    }
}

#[test]
fn test_load_default_fails_without_env_var() {
    if std::env::var("CONTEXT1_WEIGHTS").is_err() {
        let result = CompiledGptOss::load_default();
        assert!(result.is_err(), "should fail without CONTEXT1_WEIGHTS");
    }
}
