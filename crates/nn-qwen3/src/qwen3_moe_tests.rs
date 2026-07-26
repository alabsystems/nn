#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::Qwen3Config;
use nn_core::layers::kv_cache::KvCache;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

/// Minimal MoE config: 2 layers, 2 heads, 4 experts, top-2 routing.
fn tiny_moe_config() -> Qwen3MoeConfig {
    Qwen3MoeConfig {
        base: Qwen3Config {
            hidden_size: 256,
            intermediate_size: 512,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            vocab_size: 100,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            tie_word_embeddings: true,
            rope_scaling: None,
        },
        num_experts: 4,
        num_experts_per_tok: 2,
        shared_expert: false,
        shared_expert_intermediate_size: None,
    }
}

// -- Config validation --------------------------------------------------------

#[test]
fn test_moe_config_validation_ok() {
    let cfg = tiny_moe_config();
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_moe_config_zero_experts() {
    let mut cfg = tiny_moe_config();
    cfg.num_experts = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_moe_config_topk_exceeds_experts() {
    let mut cfg = tiny_moe_config();
    cfg.num_experts_per_tok = 5; // > 4 experts
    assert!(cfg.validate().is_err());
}

#[test]
fn test_moe_config_topk_zero() {
    let mut cfg = tiny_moe_config();
    cfg.num_experts_per_tok = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_moe_config_shared_expert_ff_dim_default() {
    let cfg = tiny_moe_config();
    assert_eq!(cfg.shared_expert_ff_dim(), 512);
}

#[test]
fn test_moe_config_shared_expert_ff_dim_override() {
    let mut cfg = tiny_moe_config();
    cfg.shared_expert_intermediate_size = Some(1024);
    assert_eq!(cfg.shared_expert_ff_dim(), 1024);
}

// -- Model load ---------------------------------------------------------------

#[test]
fn test_moe_model_load_zeros() {
    let cfg = tiny_moe_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg);
    assert!(model.is_ok(), "MoE model load failed: {:?}", model.err());
}

#[test]
fn test_moe_model_load_with_shared_expert() {
    let mut cfg = tiny_moe_config();
    cfg.shared_expert = true;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg);
    assert!(
        model.is_ok(),
        "MoE model with shared expert load failed: {:?}",
        model.err()
    );
}

#[test]
fn test_moe_model_load_untied_embeddings() {
    let mut cfg = tiny_moe_config();
    cfg.base.tie_word_embeddings = false;
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg);
    assert!(model.is_ok());
}

// -- Forward pass -------------------------------------------------------------

#[test]
fn test_moe_forward_single_token() {
    let cfg = tiny_moe_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();

    let logits = model.forward(&[42], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.base.vocab_size]);
}

#[test]
fn test_moe_forward_mismatched_lengths() {
    let cfg = tiny_moe_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg).unwrap();

    let result = model.forward(&[0, 1], &[0]);
    assert!(result.is_err());
}

// -- KV cache -----------------------------------------------------------------

#[test]
fn test_moe_new_cache_layer_count() {
    let cfg = tiny_moe_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg).unwrap();
    let cache = model.new_cache();
    assert_eq!(cache.num_layers(), 2);
    assert!(cache.is_empty());
}

#[test]
fn test_moe_forward_cached_none_matches_forward() {
    let cfg = tiny_moe_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg).unwrap();

    let logits_plain = model.forward(&[42], &[0]).unwrap();
    let logits_cached = model.forward_cached(&[42], &[0], None).unwrap();
    assert_eq!(logits_plain.dims(), logits_cached.dims());
    assert_eq!(
        logits_plain.to_flat_vec::<f32>().unwrap(),
        logits_cached.to_flat_vec::<f32>().unwrap()
    );
}

#[test]
fn test_moe_forward_cached_populates_cache() {
    let cfg = tiny_moe_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    let logits = model.forward_cached(&[42], &[0], Some(&mut cache)).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.base.vocab_size]);
    assert_eq!(cache.seq_len(), 1);
    assert!(!cache.is_empty());
}

#[test]
fn test_moe_forward_cached_incremental() {
    let cfg = tiny_moe_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg).unwrap();
    let mut cache = model.new_cache();

    model.forward_cached(&[10], &[0], Some(&mut cache)).unwrap();
    assert_eq!(cache.seq_len(), 1);

    model.forward_cached(&[20], &[1], Some(&mut cache)).unwrap();
    assert_eq!(cache.seq_len(), 2);

    let logits = model.forward_cached(&[30], &[2], Some(&mut cache)).unwrap();
    assert_eq!(cache.seq_len(), 3);
    assert_eq!(logits.dims(), &[1, 1, 100]);
}

#[test]
fn test_moe_forward_cached_wrong_layer_count() {
    let cfg = tiny_moe_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg).unwrap();

    let mut cache = KvCache::new(5);
    let result = model.forward_cached(&[0], &[0], Some(&mut cache));
    assert!(result.is_err());
}

// -- forward_from_embeddings --------------------------------------------------

#[test]
fn test_moe_forward_from_embeddings_shape() {
    let cfg = tiny_moe_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();

    let hidden = DynTensor::ones(&[1, 3, cfg.base.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits = model
        .forward_from_embeddings(&hidden, &[0, 1, 2], None)
        .unwrap();
    assert_eq!(logits.dims(), &[1, 3, cfg.base.vocab_size]);
}

#[test]
fn test_moe_forward_from_embeddings_wrong_hidden_size() {
    let cfg = tiny_moe_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg).unwrap();

    let hidden = DynTensor::ones(&[1, 3, 64], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&hidden, &[0, 1, 2], None);
    assert!(result.is_err());
}

#[test]
fn test_moe_forward_from_embeddings_seq_len_mismatch() {
    let cfg = tiny_moe_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();

    let hidden = DynTensor::ones(&[1, 3, cfg.base.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&hidden, &[0, 1], None);
    assert!(result.is_err());
}

// -- forward_from_embeddings_with_hidden --------------------------------------

#[test]
fn test_moe_forward_from_embeddings_with_hidden_shapes() {
    let cfg = tiny_moe_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();

    let hidden = DynTensor::ones(&[1, 3, cfg.base.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let (logits, normed) = model
        .forward_from_embeddings_with_hidden(&hidden, &[0, 1, 2], None)
        .unwrap();
    assert_eq!(logits.dims(), &[1, 3, cfg.base.vocab_size]);
    assert_eq!(normed.dims(), &[1, 3, cfg.base.hidden_size]);
}

#[test]
fn test_moe_forward_from_embeddings_with_hidden_logits_match() {
    let cfg = tiny_moe_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();

    let hidden = DynTensor::ones(&[1, 3, cfg.base.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits_plain = model
        .forward_from_embeddings(&hidden, &[0, 1, 2], None)
        .unwrap();
    let (logits_with, _normed) = model
        .forward_from_embeddings_with_hidden(&hidden, &[0, 1, 2], None)
        .unwrap();
    assert_eq!(
        logits_plain.to_flat_vec::<f32>().unwrap(),
        logits_with.to_flat_vec::<f32>().unwrap()
    );
}

// -- BF16 dtype conversion tests (#1734) --------------------------------------

#[test]
fn test_moe_forward_from_embeddings_bf16_model_f32_input() {
    // BF16 model with F32 embeddings — should auto-convert (#1734).
    let cfg = tiny_moe_config();
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();
    assert_eq!(model.dtype(), DType::BF16);

    let hidden = DynTensor::ones(&[1, 2, cfg.base.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&hidden, &[0, 1], None);
    assert!(
        result.is_ok(),
        "bf16 MoE model with f32 embeddings should succeed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().dims(), &[1, 2, cfg.base.vocab_size]);
}

#[test]
fn test_moe_forward_from_embeddings_bf16_model_bf16_input() {
    // BF16 model with BF16 embeddings — should work without conversion.
    let cfg = tiny_moe_config();
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();

    let hidden =
        DynTensor::zeros(&[1, 2, cfg.base.hidden_size], DType::BF16, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&hidden, &[0, 1], None);
    assert!(
        result.is_ok(),
        "bf16 MoE model with bf16 embeddings should succeed: {:?}",
        result.err()
    );
}

#[test]
fn test_moe_forward_from_embeddings_f32_model_unchanged() {
    // F32 model — embeddings are not converted (regression test for #1734).
    let cfg = tiny_moe_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();
    assert_eq!(model.dtype(), DType::F32);

    let hidden = DynTensor::ones(&[1, 2, cfg.base.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&hidden, &[0, 1], None);
    assert!(
        result.is_ok(),
        "f32 MoE model with f32 embeddings should still work: {:?}",
        result.err()
    );
}

#[test]
fn test_moe_forward_with_hidden_bf16_model_f32_input() {
    // BF16 model with F32 embeddings via forward_from_embeddings_with_hidden (#1734).
    let cfg = tiny_moe_config();
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();

    let hidden = DynTensor::ones(&[1, 2, cfg.base.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings_with_hidden(&hidden, &[0, 1], None);
    assert!(
        result.is_ok(),
        "bf16 MoE model with_hidden f32 input should succeed: {:?}",
        result.err()
    );
    let (logits, normed) = result.unwrap();
    assert_eq!(logits.dims(), &[1, 2, cfg.base.vocab_size]);
    assert_eq!(normed.dims(), &[1, 2, cfg.base.hidden_size]);
}

// -- Qwen3-30B-A3B and Qwen3-235B-A22B config shapes -------------------------

#[test]
fn test_qwen3_30b_a3b_config() {
    let cfg = Qwen3MoeConfig {
        base: Qwen3Config {
            hidden_size: 4096,
            intermediate_size: 2560,
            num_hidden_layers: 48,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            vocab_size: 151_936,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 40_960,
            tie_word_embeddings: false,
            rope_scaling: None,
        },
        num_experts: 128,
        num_experts_per_tok: 8,
        shared_expert: false,
        shared_expert_intermediate_size: None,
    };
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.base.num_kv_groups().unwrap(), 8);
}

#[test]
fn test_qwen3_235b_a22b_config() {
    let cfg = Qwen3MoeConfig {
        base: Qwen3Config {
            hidden_size: 4096,
            intermediate_size: 2560,
            num_hidden_layers: 94,
            num_attention_heads: 64,
            num_key_value_heads: 4,
            vocab_size: 151_936,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 40_960,
            tie_word_embeddings: false,
            rope_scaling: None,
        },
        num_experts: 128,
        num_experts_per_tok: 8,
        shared_expert: false,
        shared_expert_intermediate_size: None,
    };
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.base.num_kv_groups().unwrap(), 16);
}

// -- Config accessor ----------------------------------------------------------

#[test]
fn test_moe_config_accessor() {
    let cfg = tiny_moe_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();
    assert_eq!(model.config().num_experts, cfg.num_experts);
    assert_eq!(model.config().num_experts_per_tok, cfg.num_experts_per_tok);
}

// -- Shared expert intermediate size (#1367) ----------------------------------

#[test]
fn test_moe_load_shared_expert_different_ff_dim() {
    // shared_expert_intermediate_size != intermediate_size — should load with
    // a different-sized shared expert (weight shapes differ from regular experts).
    let mut cfg = tiny_moe_config();
    cfg.shared_expert = true;
    cfg.shared_expert_intermediate_size = Some(256); // != base.intermediate_size (512)
    assert!(cfg.validate().is_ok());

    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone());
    assert!(
        model.is_ok(),
        "load with different shared expert ff_dim failed: {:?}",
        model.err()
    );

    // Forward should also succeed.
    let model = model.unwrap();
    let logits = model.forward(&[42], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.base.vocab_size]);
}

#[test]
fn test_moe_validate_shared_expert_zero_intermediate_size() {
    let mut cfg = tiny_moe_config();
    cfg.shared_expert = true;
    cfg.shared_expert_intermediate_size = Some(0);
    let err = cfg.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("shared_expert_intermediate_size"),
        "error should mention shared_expert_intermediate_size: {msg}"
    );
}

#[test]
fn test_moe_shared_expert_ff_dim_none_uses_base() {
    // When shared_expert is true but shared_expert_intermediate_size is None,
    // should fall back to base.intermediate_size.
    let mut cfg = tiny_moe_config();
    cfg.shared_expert = true;
    cfg.shared_expert_intermediate_size = None;
    assert_eq!(cfg.shared_expert_ff_dim(), cfg.base.intermediate_size);

    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg);
    assert!(model.is_ok());
}
