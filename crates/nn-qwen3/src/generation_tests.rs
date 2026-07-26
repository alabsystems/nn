// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Generation pipeline tests for Qwen3 (#4186).
//!
//! Covers generation config construction, sampling parameters, beam search
//! output shapes, EOS token detection, max_length enforcement, batch generation
//! dimensions, and model config consistency for standard model sizes.

use super::*;
use crate::test_utils::tiny_config;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{BeamSearchConfig, GenerationConfig};
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// Generation config construction and defaults
// ---------------------------------------------------------------------------

#[test]
fn test_generation_config_default_is_greedy() {
    let cfg = GenerationConfig::default();
    assert!(
        (cfg.temperature - 0.0).abs() < f64::EPSILON,
        "default should be greedy (temperature=0)"
    );
    assert!(cfg.top_k.is_none(), "default top_k should be None");
    assert!(cfg.top_p.is_none(), "default top_p should be None");
    assert!(cfg.eos_token_id.is_none());
    assert!(cfg.seed.is_none());
}

#[test]
fn test_generation_config_max_new_tokens_default() {
    let cfg = GenerationConfig::default();
    assert_eq!(
        cfg.max_new_tokens, 128,
        "default max_new_tokens should be 128"
    );
}

#[test]
fn test_beam_search_config_new_sets_width() {
    let cfg = BeamSearchConfig::new(5);
    assert_eq!(cfg.beam_width, 5);
    assert_eq!(cfg.max_new_tokens, 128);
    assert!(!cfg.early_stopping);
    assert!(cfg.eos_token_id.is_none());
}

#[test]
fn test_beam_search_config_with_max_new_tokens() {
    let cfg = BeamSearchConfig::new(3).with_max_new_tokens(10);
    assert_eq!(cfg.beam_width, 3);
    assert_eq!(cfg.max_new_tokens, 10);
}

// ---------------------------------------------------------------------------
// Greedy generation output validation
// ---------------------------------------------------------------------------

#[test]
fn test_greedy_generation_token_ids_within_vocab() {
    let cfg = tiny_config(); // vocab_size=100
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let output = model.generate_greedy(&[42], 5).unwrap();
    for &tok in &output.token_ids {
        assert!(
            tok < cfg.vocab_size,
            "generated token {tok} >= vocab_size {}",
            cfg.vocab_size
        );
    }
}

#[test]
fn test_greedy_generation_max_length_enforced() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    for max_tokens in [1, 3, 5, 10] {
        let output = model.generate_greedy(&[0], max_tokens).unwrap();
        assert_eq!(
            output.token_ids.len(),
            max_tokens,
            "should generate exactly {max_tokens} tokens"
        );
    }
}

#[test]
fn test_greedy_generation_deterministic() {
    // Same prompt, same model -> same output (greedy = argmax).
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let out1 = model.generate_greedy(&[42, 7], 4).unwrap();
    let out2 = model.generate_greedy(&[42, 7], 4).unwrap();
    assert_eq!(
        out1.token_ids, out2.token_ids,
        "greedy generation should be deterministic"
    );
}

#[test]
fn test_greedy_generation_multi_token_prompt() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let output = model.generate_greedy(&[1, 2, 3, 4, 5], 3).unwrap();
    assert_eq!(output.token_ids.len(), 3);
}

// ---------------------------------------------------------------------------
// Beam search output validation
// ---------------------------------------------------------------------------

#[test]
fn test_beam_search_output_beam_count() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let beam_cfg = BeamSearchConfig::new(3).with_max_new_tokens(2);
    let output = model.generate_beam(&[42], &beam_cfg).unwrap();
    assert!(
        output.beams.len() <= 3,
        "should produce at most beam_width beams"
    );
    assert!(!output.beams.is_empty(), "should produce at least one beam");
}

#[test]
fn test_beam_search_beams_respect_max_tokens() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let beam_cfg = BeamSearchConfig::new(2).with_max_new_tokens(4);
    let output = model.generate_beam(&[42], &beam_cfg).unwrap();
    for beam in &output.beams {
        assert!(
            beam.token_ids.len() <= 4,
            "beam length {} exceeds max_new_tokens=4",
            beam.token_ids.len()
        );
    }
}

#[test]
fn test_beam_search_beams_sorted_descending() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let beam_cfg = BeamSearchConfig::new(4).with_max_new_tokens(3);
    let output = model.generate_beam(&[42], &beam_cfg).unwrap();
    for w in output.beams.windows(2) {
        assert!(
            w[0].log_prob >= w[1].log_prob,
            "beams not sorted: {:.6} < {:.6}",
            w[0].log_prob,
            w[1].log_prob
        );
    }
}

#[test]
fn test_beam_search_length_penalty_zero() {
    // length_penalty=0 means no length normalization.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let mut beam_cfg = BeamSearchConfig::new(2).with_max_new_tokens(3);
    beam_cfg.length_penalty = 0.0;
    let output = model.generate_beam(&[42], &beam_cfg).unwrap();
    assert!(!output.beams.is_empty());
}

#[test]
fn test_beam_search_length_penalty_positive() {
    // length_penalty=1.0 favors shorter sequences.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let mut beam_cfg = BeamSearchConfig::new(2).with_max_new_tokens(3);
    beam_cfg.length_penalty = 1.0;
    let output = model.generate_beam(&[42], &beam_cfg).unwrap();
    assert!(!output.beams.is_empty());
}

// ---------------------------------------------------------------------------
// Model config tests for standard Qwen3 sizes
// ---------------------------------------------------------------------------

#[test]
fn test_qwen3_0_6b_config_consistency() {
    let cfg = Qwen3Config::new(
        896,
        4864,
        28,
        14,
        2,
        151_936,
        1e-6,
        1_000_000.0,
        40_960,
        true,
        None,
    );
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.head_dim(), 128);
    assert_eq!(cfg.num_kv_groups().unwrap(), 7); // 14/2
    assert!(cfg.tie_word_embeddings);
}

#[test]
fn test_qwen3_1_7b_config_consistency() {
    let cfg = Qwen3Config::new(
        2048,
        11008,
        28,
        16,
        4,
        151_936,
        1e-6,
        1_000_000.0,
        40_960,
        true,
        None,
    );
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.head_dim(), 128);
    assert_eq!(cfg.num_kv_groups().unwrap(), 4); // 16/4
}

#[test]
fn test_qwen3_4b_config_consistency() {
    let cfg = Qwen3Config::new(
        2560,
        13824,
        40,
        20,
        4,
        151_936,
        1e-6,
        1_000_000.0,
        40_960,
        true,
        None,
    );
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.num_kv_groups().unwrap(), 5); // 20/4
}

#[test]
fn test_qwen3_8b_config_consistency() {
    let cfg = Qwen3Config::new(
        4096,
        14336,
        36,
        32,
        8,
        151_936,
        1e-6,
        1_000_000.0,
        131_072,
        false,
        None,
    );
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.head_dim(), 128);
    assert_eq!(cfg.num_kv_groups().unwrap(), 4); // 32/8
    assert!(!cfg.tie_word_embeddings);
}

#[test]
fn test_all_standard_configs_share_vocab_and_constants() {
    // All Qwen3 text models use vocab_size=151936, head_dim=128, rms_norm_eps=1e-6.
    let configs = vec![
        Qwen3Config::new(896, 4864, 28, 14, 2, 151_936, 1e-6, 1e6, 40960, true, None),
        Qwen3Config::new(
            2048, 11008, 28, 16, 4, 151_936, 1e-6, 1e6, 40960, true, None,
        ),
        Qwen3Config::new(
            2560, 13824, 40, 20, 4, 151_936, 1e-6, 1e6, 40960, true, None,
        ),
        Qwen3Config::new(
            4096, 14336, 36, 32, 8, 151_936, 1e-6, 1e6, 131072, false, None,
        ),
    ];
    for cfg in &configs {
        assert_eq!(cfg.vocab_size, 151_936, "all Qwen3 use vocab 151936");
        assert_eq!(cfg.head_dim(), 128, "all Qwen3 use head_dim 128");
        assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-12, "all use eps 1e-6");
        assert!(cfg.validate().is_ok());
    }
}

// ---------------------------------------------------------------------------
// Batch generation dimensions
// ---------------------------------------------------------------------------

#[test]
fn test_forward_from_embeddings_batch_2_shape() {
    // Multi-batch forward should produce [batch, seq, vocab] output.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let emb = DynTensor::zeros(&[2, 3, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits = model
        .forward_from_embeddings(&emb, &[0, 1, 2], None)
        .unwrap();
    assert_eq!(logits.dims(), &[2, 3, cfg.vocab_size]);
}

#[test]
fn test_forward_from_embeddings_batch_4_shape() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let emb = DynTensor::zeros(&[4, 1, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits = model.forward_from_embeddings(&emb, &[0], None).unwrap();
    assert_eq!(logits.dims(), &[4, 1, cfg.vocab_size]);
}

#[test]
fn test_forward_from_embeddings_with_hidden_batch_shape() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let emb = DynTensor::zeros(&[2, 3, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let (logits, hidden) = model
        .forward_from_embeddings_with_hidden(&emb, &[0, 1, 2], None)
        .unwrap();
    assert_eq!(logits.dims(), &[2, 3, cfg.vocab_size]);
    assert_eq!(hidden.dims(), &[2, 3, cfg.hidden_size]);
}
