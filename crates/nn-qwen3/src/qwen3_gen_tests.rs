// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Qwen3Model generate_greedy() and generate_beam() convenience wrappers.

use super::*;
use crate::test_utils::tiny_config;
use nn_core::layers::BeamSearchConfig;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

#[test]
fn test_generate_greedy_produces_tokens() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    // Zero weights -> all logits equal -> argmax picks token 0 each step
    let output = model.generate_greedy(&[42], 3).unwrap();
    assert_eq!(output.token_ids.len(), 3, "should generate 3 tokens");
}

#[test]
fn test_generate_greedy_respects_max_tokens() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let out1 = model.generate_greedy(&[42], 1).unwrap();
    assert_eq!(out1.token_ids.len(), 1);

    let out5 = model.generate_greedy(&[42], 5).unwrap();
    assert_eq!(out5.token_ids.len(), 5);
}

#[test]
fn test_generate_beam_produces_beams() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let mut beam_cfg = BeamSearchConfig::default();
    beam_cfg.beam_width = 2;
    beam_cfg.max_new_tokens = 3;

    let output = model.generate_beam(&[42], &beam_cfg).unwrap();
    assert!(!output.beams.is_empty(), "should produce at least one beam");
    assert!(
        output.beams.len() <= 2,
        "should produce at most beam_width beams"
    );
    // Each beam should have generated tokens
    for beam in &output.beams {
        assert!(
            !beam.token_ids.is_empty(),
            "beam should have generated tokens"
        );
        assert!(
            beam.token_ids.len() <= 3,
            "beam should respect max_new_tokens"
        );
    }
}

#[test]
fn test_generate_beam_sorted_by_score() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let mut beam_cfg = BeamSearchConfig::default();
    beam_cfg.beam_width = 4;
    beam_cfg.max_new_tokens = 2;
    beam_cfg.length_penalty = 0.0;

    let output = model.generate_beam(&[42], &beam_cfg).unwrap();
    // Beams should be sorted by log_prob (descending)
    for w in output.beams.windows(2) {
        assert!(
            w[0].log_prob >= w[1].log_prob,
            "beams not sorted: {:.4} < {:.4}",
            w[0].log_prob,
            w[1].log_prob
        );
    }
}

#[test]
fn test_model_fn_adapter_position_calculation() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let mut cache = model.new_cache();

    // First call: cache is empty, so offset = 0, positions = [0, 1]
    // Generation framework stores token IDs as U32 DynTensors via from_vec_u32.
    let input = DynTensor::from_vec_u32(vec![42, 7], &[2], &Device::Cpu).unwrap();
    let logits = model.model_fn_adapter(&input, &mut cache).unwrap();
    // After processing 2 tokens, cache seq_len should be 2
    assert_eq!(cache.seq_len(), 2);
    // Logits shape: [1, 2, vocab_size]
    assert_eq!(logits.dims()[1], 2);

    // Second call: cache has 2 tokens, so offset = 2, positions = [2]
    let input2 = DynTensor::from_vec_u32(vec![0], &[1], &Device::Cpu).unwrap();
    let logits2 = model.model_fn_adapter(&input2, &mut cache).unwrap();
    assert_eq!(cache.seq_len(), 3);
    assert_eq!(logits2.dims()[1], 1);
}

#[test]
fn test_device_accessor() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    assert!(matches!(model.device(), Device::Cpu));
}

#[test]
fn test_new_cache_layer_count() {
    let cfg = tiny_config();
    let num_layers = cfg.num_hidden_layers;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let cache = model.new_cache();
    assert_eq!(
        cache.num_layers(),
        num_layers,
        "cache layers should match model config"
    );
}

#[test]
fn test_generation_config_defaults() {
    use nn_core::layers::GenerationConfig;
    let cfg = GenerationConfig::default();
    // Greedy by default (temperature 0.0)
    assert!((cfg.temperature - 0.0).abs() < f64::EPSILON);
    assert_eq!(cfg.max_new_tokens, 128);
    assert!(cfg.eos_token_id.is_none());
}

#[test]
fn test_beam_search_config_construction() {
    let cfg = BeamSearchConfig::new(8);
    assert_eq!(cfg.beam_width, 8);
    assert_eq!(cfg.max_new_tokens, 128);
    assert!(!cfg.early_stopping);
}

#[test]
fn test_forward_shape_through_generate() {
    let cfg = tiny_config();
    let vocab_size = cfg.vocab_size;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    // Single token forward to verify logits shape
    let logits = model.forward(&[42], &[0]).unwrap();
    let dims = logits.dims();
    assert_eq!(dims.len(), 3, "logits should be 3D [batch, seq, vocab]");
    assert_eq!(dims[0], 1, "batch size should be 1");
    assert_eq!(dims[1], 1, "seq_len should be 1");
    assert_eq!(dims[2], vocab_size, "last dim should be vocab_size");
}
