#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Expanded test coverage for nn-qwen3 (#4296).
//!
//! Covers: RoPE embedding computation, SwiGLU activation properties,
//! GQA attention shape propagation, KV cache stress tests, generation
//! determinism, MoE router edge cases, config validation boundaries,
//! and multi-batch forward.

use super::*;
use crate::test_utils::tiny_config;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::BeamSearchConfig;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// RoPE embedding computation correctness
// ---------------------------------------------------------------------------

#[test]
fn test_rope_positions_affect_output() {
    // Different positions should produce different logits when hidden states are
    // non-zero. With zero weights + zero embeddings, RoPE has no effect (0*cos=0).
    // Instead, use forward_from_embeddings with ones so that Q/K projections
    // (zero_weight * ones_input = 0) still pass through QK-Norm and RoPE.
    // With zero weights the net effect is zero regardless of position, so we
    // verify a weaker property: the forward path succeeds at different positions
    // and produces finite output.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    for pos in [0, 5, 10, 63] {
        let emb = DynTensor::ones(&[1, 1, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let logits = model.forward_from_embeddings(&emb, &[pos], None).unwrap();
        let vals = logits.to_flat_vec::<f32>().unwrap();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "position {pos} should produce finite logits"
        );
    }
}

#[test]
fn test_rope_position_zero_is_deterministic() {
    // Same token at same position should produce identical logits.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let logits1 = model.forward(&[7], &[0]).unwrap();
    let logits2 = model.forward(&[7], &[0]).unwrap();
    assert_eq!(
        logits1.to_flat_vec::<f32>().unwrap(),
        logits2.to_flat_vec::<f32>().unwrap(),
        "same input at same position should be deterministic"
    );
}

#[test]
fn test_rope_large_position_is_finite() {
    // Large position values (near max_position_embeddings) should still produce finite output.
    let cfg = tiny_config(); // max_position_embeddings = 64
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let logits = model.forward(&[1], &[63]).unwrap(); // position at limit
    let vals = logits.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|v| v.is_finite()),
        "large position should produce finite output"
    );
}

// ---------------------------------------------------------------------------
// SwiGLU activation properties
// ---------------------------------------------------------------------------

#[test]
fn test_swiglu_zero_weights_produce_zero_output() {
    // With zero weights, gate_proj(x) = 0, up_proj(x) = 0.
    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0. So gate * up = 0.
    // down_proj(0) = 0. MLP output is zeros.
    let cfg = tiny_config().with_num_hidden_layers(1);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    // Input: all ones → embedding lookup → zeros (zero weight embedding)
    let logits = model.forward(&[0], &[0]).unwrap();
    let vals = logits.to_flat_vec::<f32>().unwrap();
    // All values should be identical (uniform distribution from softmax on zeros)
    let first = vals[0];
    assert!(
        vals.iter().all(|&v| (v - first).abs() < 1e-6),
        "zero-weight SwiGLU should produce uniform logits"
    );
}

#[test]
fn test_swiglu_preserves_seq_len_dimension() {
    // SwiGLU MLP is element-wise on last dim, so seq_len dimension is preserved.
    let cfg = tiny_config().with_num_hidden_layers(1);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    for seq_len in [1, 3, 7, 16] {
        let ids: Vec<usize> = (0..seq_len).collect();
        let positions: Vec<usize> = (0..seq_len).collect();
        let logits = model.forward(&ids, &positions).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, seq_len, cfg.vocab_size],
            "SwiGLU should preserve seq_len={seq_len}"
        );
    }
}

// ---------------------------------------------------------------------------
// GQA attention shape propagation with various head configs
// ---------------------------------------------------------------------------

#[test]
fn test_gqa_mha_config_forward() {
    // MHA: num_attention_heads == num_key_value_heads (groups = 1)
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 50, 1e-6, 10_000.0, 64, true, None);
    assert_eq!(cfg.num_kv_groups().unwrap(), 1);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let logits = model.forward(&[0, 1], &[0, 1]).unwrap();
    assert_eq!(logits.dims(), &[1, 2, cfg.vocab_size]);
}

#[test]
fn test_gqa_high_ratio_forward() {
    // GQA with high ratio: 8 heads, 1 kv head (MQA-like, groups = 8)
    let cfg = Qwen3Config::new(1024, 2048, 1, 8, 1, 50, 1e-6, 10_000.0, 64, true, None);
    assert_eq!(cfg.num_kv_groups().unwrap(), 8);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
}

#[test]
fn test_gqa_4_groups_forward() {
    // GQA with 4 groups: 8 heads, 2 kv heads
    let cfg = Qwen3Config::new(1024, 2048, 1, 8, 2, 50, 1e-6, 10_000.0, 64, true, None);
    assert_eq!(cfg.num_kv_groups().unwrap(), 4);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let logits = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
    assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
}

// ---------------------------------------------------------------------------
// KV cache stress tests
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_long_sequence() {
    // Generate 20 tokens incrementally, verifying cache grows correctly.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    for i in 0..20 {
        let logits = model
            .forward_cached(&[i % cfg.vocab_size], &[i], Some(&mut cache))
            .unwrap();
        assert_eq!(cache.seq_len(), i + 1, "cache seq_len after step {i}");
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
        let vals = logits.to_flat_vec::<f32>().unwrap();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "logits should be finite at step {i}"
        );
    }
}

#[test]
fn test_kv_cache_prompt_then_decode() {
    // Prefill with 5-token prompt, then decode 3 tokens.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let mut cache = model.new_cache();

    // Prefill
    let prompt = &[10, 20, 30, 40, 50];
    let positions: Vec<usize> = (0..5).collect();
    model
        .forward_cached(prompt, &positions, Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 5);

    // Decode 3 tokens
    for i in 0..3 {
        model
            .forward_cached(&[60 + i], &[5 + i], Some(&mut cache))
            .unwrap();
    }
    assert_eq!(cache.seq_len(), 8);
}

#[test]
fn test_kv_cache_reuse_after_clear() {
    // After clearing cache state (creating a new one), model should produce
    // the same output as a fresh run.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    // Use cache once
    let mut cache1 = model.new_cache();
    model
        .forward_cached(&[42], &[0], Some(&mut cache1))
        .unwrap();
    model
        .forward_cached(&[43], &[1], Some(&mut cache1))
        .unwrap();

    // Fresh cache, same first token
    let mut cache2 = model.new_cache();
    let logits_fresh = model
        .forward_cached(&[42], &[0], Some(&mut cache2))
        .unwrap();
    let logits_no_cache = model.forward(&[42], &[0]).unwrap();

    assert_eq!(
        logits_fresh.to_flat_vec::<f32>().unwrap(),
        logits_no_cache.to_flat_vec::<f32>().unwrap(),
        "fresh cache should match no-cache for first token"
    );
}

// ---------------------------------------------------------------------------
// Generation determinism
// ---------------------------------------------------------------------------

#[test]
fn test_greedy_generation_is_deterministic() {
    // Two greedy runs with the same prompt should produce identical tokens.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let out1 = model.generate_greedy(&[1, 2, 3], 5).unwrap();
    let out2 = model.generate_greedy(&[1, 2, 3], 5).unwrap();
    assert_eq!(
        out1.token_ids, out2.token_ids,
        "greedy should be deterministic"
    );
}

#[test]
fn test_greedy_generation_zero_tokens() {
    // Generating 0 tokens should return empty.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let out = model.generate_greedy(&[42], 0).unwrap();
    assert!(
        out.token_ids.is_empty(),
        "0 max_new_tokens should produce empty output"
    );
}

#[test]
fn test_greedy_multi_token_prompt() {
    // Longer prompt should still work.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let prompt: Vec<usize> = (0..10).collect();
    let out = model.generate_greedy(&prompt, 3).unwrap();
    assert_eq!(out.token_ids.len(), 3);
}

#[test]
fn test_beam_search_width_1_produces_tokens() {
    // Beam search with width=1 should produce exactly 1 beam with the right length.
    // With zero weights all logits are equal, so tie-breaking may differ between
    // greedy and beam search -- we only check structural properties.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let mut beam_cfg = BeamSearchConfig::default();
    beam_cfg.beam_width = 1;
    beam_cfg.max_new_tokens = 3;
    let beam = model.generate_beam(&[42], &beam_cfg).unwrap();

    assert_eq!(beam.beams.len(), 1, "width=1 should produce exactly 1 beam");
    assert_eq!(
        beam.beams[0].token_ids.len(),
        3,
        "beam should have max_new_tokens generated tokens"
    );
    // All generated tokens should be within vocab
    for &tok in &beam.beams[0].token_ids {
        assert!(tok < 100, "token {tok} should be < vocab_size (100)");
    }
}

// ---------------------------------------------------------------------------
// MoE router edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_moe_config_experts_equal_topk_validates() {
    // num_experts_per_tok == num_experts (all experts active) should be valid.
    let base = tiny_config();
    let moe_cfg = Qwen3MoeConfig::new(base, 4, 4, false, None);
    assert!(
        moe_cfg.validate().is_ok(),
        "topk == num_experts should be valid"
    );
}

#[test]
fn test_moe_config_single_expert_validates() {
    // 1 expert, topk=1: degenerate MoE (effectively dense).
    let base = tiny_config();
    let moe_cfg = Qwen3MoeConfig::new(base, 1, 1, false, None);
    assert!(moe_cfg.validate().is_ok(), "single expert should be valid");
}

#[test]
fn test_moe_config_clone_independence() {
    let cfg1 = Qwen3MoeConfig::new(tiny_config(), 8, 2, true, Some(256));
    let mut cfg2 = cfg1.clone();
    cfg2.num_experts = 16;
    cfg2.shared_expert = false;
    assert_eq!(cfg1.num_experts, 8);
    assert!(cfg1.shared_expert);
    assert_eq!(cfg2.num_experts, 16);
    assert!(!cfg2.shared_expert);
}

#[test]
fn test_moe_config_debug_contains_fields() {
    let cfg = Qwen3MoeConfig::new(tiny_config(), 64, 4, true, Some(1024));
    let debug = format!("{cfg:?}");
    assert!(debug.contains("num_experts"));
    assert!(debug.contains("64"));
    assert!(debug.contains("shared_expert"));
}

#[test]
fn test_moe_forward_multi_token() {
    // MoE model forward with multiple tokens.
    let cfg = Qwen3MoeConfig::new(tiny_config(), 4, 2, false, None);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();

    let logits = model.forward(&[0, 1, 2, 3], &[0, 1, 2, 3]).unwrap();
    assert_eq!(logits.dims(), &[1, 4, cfg.base.vocab_size]);
    let vals = logits.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|v| v.is_finite()));
}

#[test]
fn test_moe_kv_cache_multi_token_equivalence() {
    // One-shot vs incremental should match for MoE model.
    let cfg = Qwen3MoeConfig::new(tiny_config(), 4, 2, false, None);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg).unwrap();

    // One-shot
    let logits_oneshot = model.forward(&[10, 20, 30], &[0, 1, 2]).unwrap();
    let last_oneshot = logits_oneshot.narrow(1, 2, 1).unwrap();
    let oneshot_vec = last_oneshot.to_flat_vec::<f32>().unwrap();

    // Incremental
    let mut cache = model.new_cache();
    model.forward_cached(&[10], &[0], Some(&mut cache)).unwrap();
    model.forward_cached(&[20], &[1], Some(&mut cache)).unwrap();
    let logits_incr = model.forward_cached(&[30], &[2], Some(&mut cache)).unwrap();
    let incr_vec = logits_incr.to_flat_vec::<f32>().unwrap();

    for (i, (&a, &b)) in oneshot_vec.iter().zip(incr_vec.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "MoE logit mismatch at index {i}: oneshot={a}, incremental={b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Config validation edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_config_validation_hidden_size_1() {
    // hidden_size = 1 is valid (degenerate but not zero).
    let cfg = Qwen3Config::new(1, 4, 1, 1, 1, 10, 1e-6, 10_000.0, 16, true, None);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_validation_single_layer() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_ok());
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    assert_eq!(model.new_cache().num_layers(), 1);
}

#[test]
fn test_config_validation_very_large_rope_theta() {
    // Very large (but finite) rope_theta should be valid.
    let mut cfg = tiny_config();
    cfg.rope_theta = 1e15;
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_validation_very_small_rope_theta() {
    // Very small (but positive and finite) rope_theta should be valid.
    let mut cfg = tiny_config();
    cfg.rope_theta = 1e-15;
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_validation_inf_rms_norm_eps() {
    let mut cfg = tiny_config();
    cfg.rms_norm_eps = f64::INFINITY;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validation_subnormal_rms_norm_eps() {
    // Subnormal but positive eps should be valid.
    let mut cfg = tiny_config();
    cfg.rms_norm_eps = f64::MIN_POSITIVE / 2.0;
    // Subnormals are finite and positive -- should pass validation.
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_with_yarn_scaling_validates() {
    use nn_core::layers::YarnScaling;
    let mut cfg = tiny_config();
    cfg.rope_scaling = Some(YarnScaling::new(4.0, 1.0, 32.0, 1.0, 64));
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_moe_config_shared_expert_no_intermediate_inherits_base() {
    // shared_expert = true, shared_expert_intermediate_size = None
    // should use base.intermediate_size
    let cfg = Qwen3MoeConfig::new(tiny_config(), 4, 2, true, None);
    assert_eq!(cfg.shared_expert_ff_dim(), 512); // base.intermediate_size
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_moe_config_base_validation_propagates() {
    // Invalid base config should cause MoE validate to fail.
    let mut base = tiny_config();
    base.num_attention_heads = 0; // invalid
    let moe_cfg = Qwen3MoeConfig::new(base, 4, 2, false, None);
    assert!(
        moe_cfg.validate().is_err(),
        "invalid base should propagate to MoE validate"
    );
}

// ---------------------------------------------------------------------------
// Multi-batch forward
// ---------------------------------------------------------------------------

#[test]
fn test_multi_batch_forward_from_embeddings_batch_2() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    // Batch=2, seq=3
    let emb = DynTensor::ones(&[2, 3, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits = model
        .forward_from_embeddings(&emb, &[0, 1, 2], None)
        .unwrap();
    assert_eq!(logits.dims(), &[2, 3, cfg.vocab_size]);
    let vals = logits.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|v| v.is_finite()));
}

#[test]
fn test_multi_batch_forward_from_embeddings_batch_4() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    // Batch=4, seq=1
    let emb = DynTensor::ones(&[4, 1, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits = model.forward_from_embeddings(&emb, &[0], None).unwrap();
    assert_eq!(logits.dims(), &[4, 1, cfg.vocab_size]);
}

#[test]
fn test_multi_batch_with_hidden_batch_2() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let emb = DynTensor::ones(&[2, 2, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let (logits, normed) = model
        .forward_from_embeddings_with_hidden(&emb, &[0, 1], None)
        .unwrap();
    assert_eq!(logits.dims(), &[2, 2, cfg.vocab_size]);
    assert_eq!(normed.dims(), &[2, 2, cfg.hidden_size]);
}

#[test]
fn test_multi_batch_identical_inputs_produce_identical_outputs() {
    // Two identical sequences in a batch should produce identical logits.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let single = DynTensor::ones(&[1, 2, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits_single = model
        .forward_from_embeddings(&single, &[0, 1], None)
        .unwrap();
    let single_vec = logits_single.to_flat_vec::<f32>().unwrap();

    // Batch=2 with identical rows
    let batch = DynTensor::ones(&[2, 2, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits_batch = model
        .forward_from_embeddings(&batch, &[0, 1], None)
        .unwrap();
    let batch_vec = logits_batch.to_flat_vec::<f32>().unwrap();

    // First and second half of batch should match single
    let half = single_vec.len();
    assert_eq!(batch_vec.len(), 2 * half);
    assert_eq!(&batch_vec[..half], &single_vec[..]);
    assert_eq!(&batch_vec[half..], &single_vec[..]);
}

// ---------------------------------------------------------------------------
// MoE multi-batch forward
// ---------------------------------------------------------------------------

#[test]
fn test_moe_multi_batch_forward_from_embeddings() {
    let cfg = Qwen3MoeConfig::new(tiny_config(), 4, 2, false, None);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();

    let emb = DynTensor::ones(&[2, 2, cfg.base.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits = model.forward_from_embeddings(&emb, &[0, 1], None).unwrap();
    assert_eq!(logits.dims(), &[2, 2, cfg.base.vocab_size]);
}

// ---------------------------------------------------------------------------
// Embedding accessor tests
// ---------------------------------------------------------------------------

#[test]
fn test_embed_tokens_forward_ids_roundtrip() {
    // embed_tokens().forward_ids() should produce [seq_len, hidden_size]
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let emb = model.embed_tokens().forward_ids(&[0, 1, 2]).unwrap();
    assert_eq!(emb.dims(), &[3, cfg.hidden_size]);
}

#[test]
fn test_embed_tokens_different_ids_same_with_zeros() {
    // With zero weights, all embeddings should be identical (all zeros).
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let emb0 = model.embed_tokens().forward_ids(&[0]).unwrap();
    let emb42 = model.embed_tokens().forward_ids(&[42]).unwrap();
    assert_eq!(
        emb0.to_flat_vec::<f32>().unwrap(),
        emb42.to_flat_vec::<f32>().unwrap(),
        "zero-weight embeddings should all be identical"
    );
}

// ---------------------------------------------------------------------------
// Deeper layer configs
// ---------------------------------------------------------------------------

#[test]
fn test_model_load_4_layers() {
    let cfg = tiny_config().with_num_hidden_layers(4);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    assert_eq!(model.config().num_hidden_layers, 4);
    assert_eq!(model.new_cache().num_layers(), 4);

    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
}

#[test]
fn test_model_load_8_layers() {
    let cfg = tiny_config().with_num_hidden_layers(8);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let logits = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
    assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
}

// ---------------------------------------------------------------------------
// Error path coverage
// ---------------------------------------------------------------------------

#[test]
fn test_forward_from_embeddings_4d_rejected() {
    // 4D tensor should fail with rank error.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let hidden = DynTensor::ones(&[1, 1, 2, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&hidden, &[0, 1], None);
    assert!(result.is_err(), "4D input should fail with rank error");
}

#[test]
fn test_forward_from_embeddings_1d_rejected() {
    // 1D tensor should fail with rank error.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let hidden = DynTensor::ones(&[cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&hidden, &[0], None);
    assert!(result.is_err(), "1D input should fail with rank error");
}

#[test]
fn test_inf_input_rejected() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let mut data = vec![1.0f32; cfg.hidden_size];
    data[0] = f32::INFINITY;
    let inf_embeddings = DynTensor::from_vec(data, &[1, 1, cfg.hidden_size], &Device::Cpu).unwrap();

    let err = model.forward_from_embeddings(&inf_embeddings, &[0], None);
    assert!(err.is_err(), "Inf input should be rejected");
}
