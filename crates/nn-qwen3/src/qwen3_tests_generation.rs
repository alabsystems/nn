// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Qwen3 generate_greedy() and generate_beam() convenience wrappers.

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

    // Zero weights → all logits equal → argmax picks token 0 each step
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
fn test_device_accessor() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    assert!(matches!(model.device(), Device::Cpu));
}
