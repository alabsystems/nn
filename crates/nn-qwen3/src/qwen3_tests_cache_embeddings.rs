#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `forward_from_embeddings` and `forward_from_embeddings_with_hidden`.
//! Extracted from `qwen3_tests_cache.rs` to keep files under 500 lines.

use super::super::*;
use crate::test_utils::tiny_config;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// -- forward_from_embeddings tests -------------------------------------------

#[test]
fn test_forward_from_embeddings_shape() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    // Create hidden_states of shape [1, 3, hidden_size]
    let hidden = DynTensor::ones(&[1, 3, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let positions = &[0, 1, 2];
    let logits = model
        .forward_from_embeddings(&hidden, positions, None)
        .unwrap();
    assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
}

#[test]
fn test_forward_from_embeddings_single_token() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let hidden = DynTensor::ones(&[1, 1, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits = model.forward_from_embeddings(&hidden, &[0], None).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
}

#[test]
fn test_forward_from_embeddings_matches_forward_cached() {
    // With the same embedding, forward_from_embeddings should produce
    // identical output to forward_cached.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    // Single token with zero weights: embed_tokens produces zeros
    let logits_token = model.forward(&[42], &[0]).unwrap();

    // Manually embed: forward_ids returns zeros, unsqueeze to [1, 1, hidden]
    let hidden = model
        .embed_tokens()
        .forward_ids(&[42])
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let logits_embed = model.forward_from_embeddings(&hidden, &[0], None).unwrap();

    assert_eq!(logits_token.dims(), logits_embed.dims());
    assert_eq!(
        logits_token.to_flat_vec::<f32>().unwrap(),
        logits_embed.to_flat_vec::<f32>().unwrap(),
    );
}

#[test]
fn test_forward_from_embeddings_wrong_hidden_size() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    // Wrong hidden_size (64 instead of 256)
    let hidden = DynTensor::ones(&[1, 3, 64], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&hidden, &[0, 1, 2], None);
    assert!(result.is_err());
    let msg = format!("{:?}", result.unwrap_err());
    assert!(
        msg.contains("hidden_size"),
        "expected hidden_size error: {msg}"
    );
}

#[test]
fn test_forward_from_embeddings_seq_len_mismatch() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let hidden = DynTensor::ones(&[1, 3, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    // 2 positions for 3 tokens
    let result = model.forward_from_embeddings(&hidden, &[0, 1], None);
    assert!(result.is_err());
    let msg = format!("{:?}", result.unwrap_err());
    assert!(msg.contains("seq_len"), "expected seq_len error: {msg}");
}

#[test]
fn test_forward_from_embeddings_wrong_rank() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    // 2D tensor instead of 3D
    let hidden = DynTensor::ones(&[3, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&hidden, &[0, 1, 2], None);
    assert!(result.is_err(), "2D input should fail with rank error");
}

#[test]
fn test_forward_from_embeddings_with_cache() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    // Step 1: first token
    let h1 = DynTensor::ones(&[1, 1, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    model
        .forward_from_embeddings(&h1, &[0], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 1);

    // Step 2: second token (cache grows)
    let h2 = DynTensor::ones(&[1, 1, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits = model
        .forward_from_embeddings(&h2, &[1], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 2);
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
}

// -- forward_from_embeddings_with_hidden tests --------------------------------

#[test]
fn test_forward_with_hidden_shapes() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let hidden = DynTensor::ones(&[1, 3, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let (logits, normed) = model
        .forward_from_embeddings_with_hidden(&hidden, &[0, 1, 2], None)
        .unwrap();
    assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
    assert_eq!(normed.dims(), &[1, 3, cfg.hidden_size]);
}

#[test]
fn test_forward_with_hidden_logits_match_plain() {
    // Logits from with_hidden should be identical to forward_from_embeddings.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let hidden = DynTensor::ones(&[1, 2, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits_plain = model
        .forward_from_embeddings(&hidden, &[0, 1], None)
        .unwrap();
    let (logits_hidden, _normed) = model
        .forward_from_embeddings_with_hidden(&hidden, &[0, 1], None)
        .unwrap();
    assert_eq!(
        logits_plain.to_flat_vec::<f32>().unwrap(),
        logits_hidden.to_flat_vec::<f32>().unwrap(),
    );
}

#[test]
fn test_forward_with_hidden_normed_is_finite() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let hidden = DynTensor::ones(&[1, 1, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let (_logits, normed) = model
        .forward_from_embeddings_with_hidden(&hidden, &[0], None)
        .unwrap();
    let vals = normed.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "normed hidden states should be finite"
    );
}

#[test]
fn test_forward_with_hidden_wrong_hidden_size() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let hidden = DynTensor::ones(&[1, 3, 64], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings_with_hidden(&hidden, &[0, 1, 2], None);
    assert!(result.is_err());
    let msg = format!("{:?}", result.unwrap_err());
    assert!(
        msg.contains("hidden_size"),
        "expected hidden_size error: {msg}"
    );
}

#[test]
fn test_forward_with_hidden_seq_len_mismatch() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let hidden = DynTensor::ones(&[1, 3, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings_with_hidden(&hidden, &[0, 1], None);
    assert!(result.is_err());
    let msg = format!("{:?}", result.unwrap_err());
    assert!(msg.contains("seq_len"), "expected seq_len error: {msg}");
}

#[test]
fn test_forward_with_hidden_with_cache() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    // Step 1
    let h1 = DynTensor::ones(&[1, 1, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let (logits, normed) = model
        .forward_from_embeddings_with_hidden(&h1, &[0], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 1);
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    assert_eq!(normed.dims(), &[1, 1, cfg.hidden_size]);

    // Step 2 (cache grows)
    let h2 = DynTensor::ones(&[1, 1, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let (logits2, normed2) = model
        .forward_from_embeddings_with_hidden(&h2, &[1], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 2);
    assert_eq!(logits2.dims(), &[1, 1, cfg.vocab_size]);
    assert_eq!(normed2.dims(), &[1, 1, cfg.hidden_size]);
}

// -- BF16 dtype conversion tests (#1734) --------------------------------------

#[test]
fn test_forward_from_embeddings_bf16_model_f32_input() {
    // BF16 model with F32 embeddings — should auto-convert (#1734).
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    assert_eq!(model.dtype(), DType::BF16);

    let hidden = DynTensor::ones(&[1, 2, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&hidden, &[0, 1], None);
    assert!(
        result.is_ok(),
        "bf16 model with f32 embeddings should succeed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().dims(), &[1, 2, cfg.vocab_size]);
}

#[test]
fn test_forward_from_embeddings_bf16_model_bf16_input() {
    // BF16 model with BF16 embeddings — should work without conversion.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let hidden = DynTensor::zeros(&[1, 2, cfg.hidden_size], DType::BF16, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&hidden, &[0, 1], None);
    assert!(
        result.is_ok(),
        "bf16 model with bf16 embeddings should succeed: {:?}",
        result.err()
    );
}

#[test]
fn test_forward_from_embeddings_f32_model_unchanged() {
    // F32 model — embeddings are not converted (regression test for #1734).
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    assert_eq!(model.dtype(), DType::F32);

    let hidden = DynTensor::ones(&[1, 2, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&hidden, &[0, 1], None);
    assert!(
        result.is_ok(),
        "f32 model with f32 embeddings should still work: {:?}",
        result.err()
    );
}

#[test]
fn test_forward_with_hidden_bf16_model_f32_input() {
    // BF16 model with F32 embeddings via forward_from_embeddings_with_hidden (#1734).
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let hidden = DynTensor::ones(&[1, 2, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings_with_hidden(&hidden, &[0, 1], None);
    assert!(
        result.is_ok(),
        "bf16 model with_hidden f32 input should succeed: {:?}",
        result.err()
    );
    let (logits, normed) = result.unwrap();
    assert_eq!(logits.dims(), &[1, 2, cfg.vocab_size]);
    assert_eq!(normed.dims(), &[1, 2, cfg.hidden_size]);
}
