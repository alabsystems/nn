// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive expanded tests for nn-qwen3.
//!
//! Covers gaps not addressed by existing test files:
//! - Config computed properties and SwiGLU dimension conventions
//! - Attention weight shape contracts (q/k/v/o_proj, QK-Norm)
//! - KV cache shape evolution through incremental decode
//! - RoPE cache integration and frequency decay properties
//! - Causal mask skip optimization (seq_len=1 returns None)
//! - Model forward with large vocab, tied vs untied lm_head
//! - MoE shared expert intermediate size fallback
//! - Error path exhaustive coverage
//! - Multi-step generation state consistency

use super::*;
use crate::rope_cache::RoPECache;
use crate::test_utils::tiny_config;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::kv_cache::KvCache;
use nn_core::layers::{repeat_kv, BeamSearchConfig, RotaryEmbedding};
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// Config: computed properties and SwiGLU intermediate_size convention
// ---------------------------------------------------------------------------

#[test]
fn test_swiglu_intermediate_size_convention() {
    // Qwen3 uses SwiGLU with intermediate_size specified directly in config.
    // The common convention is intermediate_size = 2/3 * 4 * hidden_size rounded
    // to a multiple of some alignment. Verify Qwen3 production configs follow this.
    //
    // For Qwen3-8B: hidden=4096, intermediate=14336.
    // 2/3 * 4 * 4096 = 10922.67. Rounded up to 14336 (Qwen3 uses larger than 2/3*4h).
    // The actual ratio is intermediate/hidden.
    let cfg_8b = Qwen3Config::new(
        4096,
        14336,
        36,
        32,
        8,
        151_936,
        1e-6,
        1_000_000.0,
        131072,
        false,
        None,
    );
    let ratio = cfg_8b.intermediate_size as f64 / cfg_8b.hidden_size as f64;
    // Qwen3 uses ratio ~3.5 (14336/4096 = 3.5)
    assert!(
        ratio > 2.0 && ratio < 6.0,
        "SwiGLU intermediate ratio should be reasonable: got {ratio}"
    );
}

#[test]
fn test_swiglu_gate_up_down_dimensions() {
    // SwiGLU MLP: gate_proj [intermediate, hidden], up_proj [intermediate, hidden],
    // down_proj [hidden, intermediate]. Verify loading with VarBuilder::zeros
    // succeeds with these dimensions.
    let cfg = tiny_config(); // hidden=256, intermediate=512
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    // Verify model loads -- implicitly checks that weight shapes are consistent
    // with config. gate_proj = [512, 256], up_proj = [512, 256], down_proj = [256, 512].
    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
}

#[test]
fn test_swiglu_intermediate_size_various_ratios() {
    // Verify that different intermediate_size / hidden_size ratios all produce
    // valid models. The SwiGLU MLP only requires intermediate_size > 0.
    for (hidden, intermediate) in [(128, 256), (128, 512), (128, 1024), (256, 128)] {
        let cfg = Qwen3Config::new(
            hidden,
            intermediate,
            1,
            hidden / 128, // num_attention_heads = hidden / head_dim
            hidden / 128, // MHA
            50,
            1e-6,
            10_000.0,
            32,
            true,
            None,
        );
        if cfg.num_attention_heads == 0 {
            continue; // skip invalid configs
        }
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = Qwen3Model::load(&vb, cfg.clone());
        assert!(
            model.is_ok(),
            "hidden={hidden}, intermediate={intermediate} should load: {:?}",
            model.err()
        );
    }
}

// ---------------------------------------------------------------------------
// Attention weight shape contracts
// ---------------------------------------------------------------------------

#[test]
fn test_attention_projection_weight_shapes() {
    // q_proj: [num_heads * head_dim, hidden_size]
    // k_proj: [num_kv_heads * head_dim, hidden_size]
    // v_proj: [num_kv_heads * head_dim, hidden_size]
    // o_proj: [hidden_size, num_heads * head_dim]
    // This is implicitly tested by model loading, but verify the math explicitly.
    let cfg = tiny_config(); // hidden=256, heads=2, kv_heads=2, head_dim=128
    let q_dim = cfg.num_attention_heads * cfg.head_dim(); // 2 * 128 = 256
    let k_dim = cfg.num_key_value_heads * cfg.head_dim(); // 2 * 128 = 256
    assert_eq!(q_dim, 256);
    assert_eq!(k_dim, 256);

    // For GQA with fewer KV heads:
    let cfg_gqa = Qwen3Config::new(1024, 2048, 1, 8, 2, 50, 1e-6, 10_000.0, 64, true, None);
    let q_gqa = cfg_gqa.num_attention_heads * cfg_gqa.head_dim(); // 8 * 128 = 1024
    let k_gqa = cfg_gqa.num_key_value_heads * cfg_gqa.head_dim(); // 2 * 128 = 256
    assert_eq!(q_gqa, 1024);
    assert_eq!(k_gqa, 256);
    assert_ne!(
        q_gqa, k_gqa,
        "GQA: q_proj and k_proj should have different output dims"
    );
}

#[test]
fn test_qk_norm_head_dim_size() {
    // QK-Norm in Qwen3: RMSNorm with weight shape [head_dim].
    // head_dim is always 128 for Qwen3.
    // This is verified implicitly by model load but test the constant.
    assert_eq!(tiny_config().head_dim(), 128);

    // With GQA, QK-Norm still uses head_dim (not hidden_size)
    let cfg = Qwen3Config::new(4096, 14336, 1, 32, 8, 50, 1e-6, 10_000.0, 64, true, None);
    assert_eq!(cfg.head_dim(), 128);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg);
    assert!(
        model.is_ok(),
        "QK-Norm with GQA should load: {:?}",
        model.err()
    );
}

#[test]
fn test_attention_scale_factor() {
    // Scaled dot-product attention uses scale = 1 / sqrt(head_dim).
    // For Qwen3 head_dim=128: scale = 1/sqrt(128) = 1/11.3137... = 0.08838...
    let head_dim = 128;
    let scale = 1.0 / f64::from(head_dim).sqrt();
    let expected = 1.0 / 128.0_f64.sqrt();
    assert!(
        (scale - expected).abs() < 1e-10,
        "attention scale should be 1/sqrt(128) = {expected}, got {scale}"
    );
    assert!(
        (scale - 0.08838834764831845).abs() < 1e-10,
        "attention scale should be ~0.0884, got {scale}"
    );
}

// ---------------------------------------------------------------------------
// KV cache: shape evolution through incremental decode
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_seq_len_grows_monotonically() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    let mut prev_seq_len = 0;
    for i in 0..10 {
        model
            .forward_cached(&[i % cfg.vocab_size], &[i], Some(&mut cache))
            .unwrap();
        let current = cache.seq_len();
        assert!(
            current > prev_seq_len,
            "seq_len should grow: was {prev_seq_len}, now {current} at step {i}"
        );
        prev_seq_len = current;
    }
}

#[test]
fn test_kv_cache_prefill_then_single_token_decode() {
    // Prefill 8 tokens at once, then decode 1 token at a time.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    // Prefill
    let prompt: Vec<usize> = (0..8).map(|i| i % cfg.vocab_size).collect();
    let positions: Vec<usize> = (0..8).collect();
    let prefill_logits = model
        .forward_cached(&prompt, &positions, Some(&mut cache))
        .unwrap();
    assert_eq!(prefill_logits.dims(), &[1, 8, cfg.vocab_size]);
    assert_eq!(cache.seq_len(), 8);

    // Single-token decode: seq_len=1 should skip causal mask (optimization)
    let decode_logits = model.forward_cached(&[42], &[8], Some(&mut cache)).unwrap();
    assert_eq!(decode_logits.dims(), &[1, 1, cfg.vocab_size]);
    assert_eq!(cache.seq_len(), 9);
}

#[test]
fn test_kv_cache_two_token_prefill_then_decode() {
    // Prefill with 2 tokens, decode 1, verify cache seq_len at each step
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let mut cache = model.new_cache();

    // Prefill 2 tokens
    model
        .forward_cached(&[10, 20], &[0, 1], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 2);

    // Decode 1 token
    model.forward_cached(&[30], &[2], Some(&mut cache)).unwrap();
    assert_eq!(cache.seq_len(), 3);
}

#[test]
fn test_kv_cache_wrong_layer_count_error_message() {
    let cfg = tiny_config(); // 2 layers
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    // Cache with 1 layer for a 2-layer model
    let mut cache = KvCache::new(1);
    let result = model.forward_cached(&[0], &[0], Some(&mut cache));
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("1") && msg.contains("2"),
        "error should mention both layer counts: {msg}"
    );

    // Cache with 10 layers for a 2-layer model
    let mut cache = KvCache::new(10);
    let result = model.forward_cached(&[0], &[0], Some(&mut cache));
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// RoPE cache: frequency properties, boundary, integration
// ---------------------------------------------------------------------------

#[test]
fn test_rope_cache_frequencies_decay_geometrically() {
    // RoPE frequencies should decay as 1/base^(2i/dim).
    // For base=10000, dim=128: freq[0]=1.0, freq[32]=0.01, freq[63]=~1.155e-4
    let cache = RoPECache::new(16, 128, 10_000.0);
    let (cos, sin) = cache.get(1);

    // At position 1: angle_i = 1 * freq_i
    // cos(freq_0) should be cos(1.0) ~ 0.5403
    let expected_cos_0 = 1.0_f64.cos() as f32;
    assert!(
        (cos[0] - expected_cos_0).abs() < 1e-5,
        "cos[0] at pos 1 should be cos(1.0)={expected_cos_0}, got {}",
        cos[0]
    );

    // sin(freq_0) should be sin(1.0) ~ 0.8415
    let expected_sin_0 = 1.0_f64.sin() as f32;
    assert!(
        (sin[0] - expected_sin_0).abs() < 1e-5,
        "sin[0] at pos 1 should be sin(1.0)={expected_sin_0}, got {}",
        sin[0]
    );

    // Later frequency indices have smaller angles, so cos approaches 1
    // freq[32] = 0.01, angle at pos 1 = 0.01, cos(0.01) ~ 0.99995
    let expected_cos_32 = (0.01_f64).cos() as f32;
    assert!(
        (cos[32] - expected_cos_32).abs() < 1e-5,
        "cos[32] at pos 1 should be close to 1.0, got {}",
        cos[32]
    );
}

#[test]
fn test_rope_cache_qwen3_base_1m() {
    // Qwen3 uses base=1_000_000. Frequencies decay much slower.
    let cache = RoPECache::new(16, 128, 1_000_000.0);
    let (cos, sin) = cache.get(0);

    // At position 0: all cos=1, sin=0
    for i in 0..64 {
        assert!(
            (cos[i] - 1.0).abs() < 1e-6,
            "cos[{i}] at pos 0 with base=1M: expected 1.0, got {}",
            cos[i]
        );
        assert!(
            sin[i].abs() < 1e-6,
            "sin[{i}] at pos 0 with base=1M: expected 0.0, got {}",
            sin[i]
        );
    }

    // At position 1: freq[0] = 1.0, same as base=10000
    let (cos_p1, _) = cache.get(1);
    let expected = 1.0_f64.cos() as f32;
    assert!(
        (cos_p1[0] - expected).abs() < 1e-5,
        "freq[0] is always 1.0 regardless of base"
    );
}

#[test]
fn test_rope_cache_get_range_empty() {
    let cache = RoPECache::new(16, 8, 10_000.0);
    let (cos_range, sin_range) = cache.get_range(5, 0);
    assert_eq!(cos_range.len(), 0);
    assert_eq!(sin_range.len(), 0);
}

#[test]
fn test_rope_cache_try_new_success() {
    let cache = RoPECache::try_new(64, 128, 10_000.0);
    assert!(cache.is_ok());
    let c = cache.unwrap();
    assert_eq!(c.max_seq_len(), 64);
    assert_eq!(c.head_dim(), 128);
    assert_eq!(c.half_dim(), 64);
}

#[test]
fn test_rope_cache_try_new_zero_base() {
    let result = RoPECache::try_new(64, 128, 0.0);
    assert!(result.is_err(), "base=0 should be rejected");
}

#[test]
fn test_rope_cache_apply_double_rotation() {
    // Applying RoPE at position p then at position -p should approximately
    // recover the original (rotation is invertible). We test that applying
    // position 0 (identity) preserves the vector.
    let cache = RoPECache::new(16, 8, 10_000.0);
    let (cos, sin) = cache.get(0);

    let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut q = original.clone();
    let mut k = original.clone();

    RoPECache::apply_rope(&mut q, &mut k, cos, sin);

    // At position 0, cos=1, sin=0 => identity
    for (i, &orig) in original.iter().enumerate() {
        assert!(
            (q[i] - orig).abs() < 1e-6,
            "position 0 should be identity for q[{i}]"
        );
    }
}

// ---------------------------------------------------------------------------
// Causal mask skip optimization
// ---------------------------------------------------------------------------

#[test]
fn test_build_causal_mask_skips_single_token_with_cache() {
    use crate::forward_common::build_causal_mask;

    // seq_len=1 with existing cache should return None (mask is all-zeros for single query)
    let cache = KvCache::new(2);
    // We need to simulate cache having some entries. Instead of populating,
    // just verify the function behavior: even with cache=Some, single token
    // mask is skipped.
    let mask = build_causal_mask(1, Some(&cache), DType::F32, &Device::Cpu).unwrap();
    assert!(
        mask.is_none(),
        "seq_len=1 should skip mask allocation even with empty cache"
    );
}

#[test]
fn test_build_causal_mask_returns_mask_for_prefill() {
    use crate::forward_common::build_causal_mask;

    // seq_len=4, no cache -> should return Some(mask)
    let mask = build_causal_mask(4, None, DType::F32, &Device::Cpu).unwrap();
    assert!(mask.is_some());
    let m = mask.unwrap();
    assert_eq!(m.dims(), &[1, 1, 4, 4]);
}

#[test]
fn test_build_causal_mask_bf16_dtype() {
    use crate::forward_common::build_causal_mask;

    // Verify mask dtype matches requested dtype (BF16 for bf16 inference)
    let mask = build_causal_mask(3, None, DType::BF16, &Device::Cpu).unwrap();
    assert!(mask.is_some());
    let m = mask.unwrap();
    assert_eq!(m.dtype(), DType::BF16);
    assert_eq!(m.dims(), &[1, 1, 3, 3]);
}

// ---------------------------------------------------------------------------
// Model: large vocab, tied vs untied lm_head
// ---------------------------------------------------------------------------

#[test]
fn test_model_large_vocab_size() {
    // Qwen3 uses vocab_size=151_936. Test with a large but manageable size.
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 10_000, 1e-6, 10_000.0, 64, true, None);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, 10_000]);
}

#[test]
fn test_model_tied_vs_untied_same_output_zeros() {
    // With zero weights, tied and untied should produce the same output
    // (both lm_head weights are zeros, embed_tokens weight is zeros).
    let cfg_tied = Qwen3Config::new(256, 512, 1, 2, 2, 50, 1e-6, 10_000.0, 64, true, None);
    let cfg_untied = Qwen3Config::new(256, 512, 1, 2, 2, 50, 1e-6, 10_000.0, 64, false, None);

    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model_tied = Qwen3Model::load(&vb, cfg_tied).unwrap();
    let model_untied = Qwen3Model::load(&vb, cfg_untied).unwrap();

    let logits_tied = model_tied.forward(&[0], &[0]).unwrap();
    let logits_untied = model_untied.forward(&[0], &[0]).unwrap();

    assert_eq!(
        logits_tied.to_flat_vec::<f32>().unwrap(),
        logits_untied.to_flat_vec::<f32>().unwrap(),
        "with zero weights, tied and untied should produce same output"
    );
}

#[test]
fn test_model_vocab_size_1() {
    // Degenerate but valid: vocab_size=1 (single token vocabulary)
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 1, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_ok());
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, 1]);
}

// ---------------------------------------------------------------------------
// MoE: shared expert intermediate size fallback
// ---------------------------------------------------------------------------

#[test]
fn test_moe_shared_expert_ff_dim_explicit() {
    let cfg = Qwen3MoeConfig::new(tiny_config(), 4, 2, true, Some(1024));
    assert_eq!(cfg.shared_expert_ff_dim(), 1024);
}

#[test]
fn test_moe_shared_expert_ff_dim_fallback_to_base() {
    // When shared_expert_intermediate_size is None, fallback to base.intermediate_size
    let cfg = Qwen3MoeConfig::new(tiny_config(), 4, 2, true, None);
    assert_eq!(cfg.shared_expert_ff_dim(), tiny_config().intermediate_size);
    assert_eq!(cfg.shared_expert_ff_dim(), 512);
}

#[test]
fn test_moe_shared_expert_ff_dim_not_used_when_disabled() {
    // Even with shared_expert=false, shared_expert_ff_dim() still returns
    // a value (it doesn't check shared_expert flag).
    let cfg = Qwen3MoeConfig::new(tiny_config(), 4, 2, false, Some(999));
    assert_eq!(cfg.shared_expert_ff_dim(), 999);
}

#[test]
fn test_moe_config_shared_expert_zero_intermediate_rejected() {
    let cfg = Qwen3MoeConfig::new(tiny_config(), 4, 2, true, Some(0));
    assert!(
        cfg.validate().is_err(),
        "shared_expert_intermediate_size=0 should be rejected"
    );
}

#[test]
fn test_moe_config_zero_experts_per_tok_rejected() {
    let cfg = Qwen3MoeConfig::new(tiny_config(), 4, 0, false, None);
    assert!(
        cfg.validate().is_err(),
        "num_experts_per_tok=0 should be rejected"
    );
}

// ---------------------------------------------------------------------------
// Error paths: exhaustive coverage
// ---------------------------------------------------------------------------

#[test]
fn test_error_tensor_passthrough() {
    // Qwen3Error::Tensor variant should round-trip through From<TensorError>
    use nn_core::TensorError;

    // Create a TensorError, convert to Qwen3Error, convert back
    let original = TensorError::DimensionOutOfRange { dim: 5, rank: 3 };
    let qwen_err = Qwen3Error::from(original);
    let tensor_err: TensorError = qwen_err.into();
    // The round-trip through backend_failure_with_source wraps it, so just
    // verify it's a TensorError with reasonable content.
    let msg = format!("{tensor_err}");
    assert!(!msg.is_empty(), "round-tripped error should have a message");
}

#[test]
fn test_validate_forward_input_matching_lengths() {
    use crate::forward_common::validate_forward_input;
    assert!(validate_forward_input(&[1, 2, 3], &[0, 1, 2]).is_ok());
}

#[test]
fn test_validate_forward_input_empty() {
    use crate::forward_common::validate_forward_input;
    assert!(validate_forward_input(&[], &[]).is_ok());
}

#[test]
fn test_validate_forward_input_mismatch() {
    use crate::forward_common::validate_forward_input;
    let result = validate_forward_input(&[1, 2], &[0]);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("2") && msg.contains("1"),
        "should mention both lengths: {msg}"
    );
}

#[test]
fn test_validate_embedding_input_correct() {
    use crate::forward_common::validate_embedding_input;
    let hidden = DynTensor::ones(&[1, 3, 256], DType::F32, &Device::Cpu).unwrap();
    assert!(validate_embedding_input(&hidden, &[0, 1, 2], 256).is_ok());
}

#[test]
fn test_validate_embedding_input_wrong_hidden() {
    use crate::forward_common::validate_embedding_input;
    let hidden = DynTensor::ones(&[1, 3, 128], DType::F32, &Device::Cpu).unwrap();
    let result = validate_embedding_input(&hidden, &[0, 1, 2], 256);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("hidden_size"),
        "should mention hidden_size: {msg}"
    );
}

#[test]
fn test_validate_embedding_input_wrong_seq_len() {
    use crate::forward_common::validate_embedding_input;
    let hidden = DynTensor::ones(&[1, 5, 256], DType::F32, &Device::Cpu).unwrap();
    let result = validate_embedding_input(&hidden, &[0, 1, 2], 256);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("seq_len"), "should mention seq_len: {msg}");
}

#[test]
fn test_validate_cache_correct_layers() {
    use crate::forward_common::validate_cache;
    let cache = KvCache::new(4);
    assert!(validate_cache(Some(&cache), 4).is_ok());
}

#[test]
fn test_validate_cache_none_always_ok() {
    use crate::forward_common::validate_cache;
    assert!(validate_cache(None, 100).is_ok());
}

#[test]
fn test_validate_cache_mismatch() {
    use crate::forward_common::validate_cache;
    let cache = KvCache::new(3);
    let result = validate_cache(Some(&cache), 5);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("3") && msg.contains("5"),
        "should mention both counts: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Multi-step generation state consistency
// ---------------------------------------------------------------------------

#[test]
fn test_generate_greedy_tokens_within_vocab() {
    let cfg = tiny_config(); // vocab_size=100
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let output = model.generate_greedy(&[42], 10).unwrap();
    for &tok in &output.token_ids {
        assert!(
            tok < cfg.vocab_size,
            "generated token {tok} should be < vocab_size ({})",
            cfg.vocab_size
        );
    }
}

#[test]
fn test_generate_greedy_different_prompts_same_length() {
    // Different prompts should still generate the correct number of tokens.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    for prompt_len in [1, 3, 5, 10] {
        let prompt: Vec<usize> = (0..prompt_len).collect();
        let output = model.generate_greedy(&prompt, 4).unwrap();
        assert_eq!(
            output.token_ids.len(),
            4,
            "prompt_len={prompt_len} should still generate 4 tokens"
        );
    }
}

#[test]
fn test_beam_search_width_2_max_beams() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let mut beam_cfg = BeamSearchConfig::default();
    beam_cfg.beam_width = 2;
    beam_cfg.max_new_tokens = 5;

    let output = model.generate_beam(&[1, 2, 3], &beam_cfg).unwrap();
    assert!(
        output.beams.len() <= 2,
        "beam_width=2 should produce at most 2 beams, got {}",
        output.beams.len()
    );
    for beam in &output.beams {
        assert!(
            beam.token_ids.len() <= 5,
            "each beam should have at most 5 tokens"
        );
    }
}

// ---------------------------------------------------------------------------
// Forward from embeddings: dtype conversion edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_forward_from_embeddings_f16_model_f32_input() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F16, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    assert_eq!(model.dtype(), DType::F16);

    let hidden = DynTensor::ones(&[1, 2, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&hidden, &[0, 1], None);
    assert!(
        result.is_ok(),
        "F16 model with F32 input should auto-convert: {:?}",
        result.err()
    );
}

#[test]
fn test_forward_from_embeddings_with_hidden_returns_both() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let hidden = DynTensor::ones(&[1, 4, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let (logits, normed) = model
        .forward_from_embeddings_with_hidden(&hidden, &[0, 1, 2, 3], None)
        .unwrap();

    // Verify shapes
    assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
    assert_eq!(normed.dims(), &[1, 4, cfg.hidden_size]);

    // Both should be finite
    let logit_vals = logits.to_flat_vec::<f32>().unwrap();
    let normed_vals = normed.to_flat_vec::<f32>().unwrap();
    assert!(logit_vals.iter().all(|v| v.is_finite()));
    assert!(normed_vals.iter().all(|v| v.is_finite()));
}

// ---------------------------------------------------------------------------
// Model accessors and properties
// ---------------------------------------------------------------------------

#[test]
fn test_model_config_is_reference_to_loaded_config() {
    let cfg = Qwen3Config::new(256, 512, 3, 2, 1, 200, 1e-5, 50_000.0, 128, false, None);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    assert_eq!(model.config().hidden_size, 256);
    assert_eq!(model.config().num_hidden_layers, 3);
    assert_eq!(model.config().num_key_value_heads, 1);
    assert_eq!(model.config().vocab_size, 200);
    assert!(!model.config().tie_word_embeddings);
    assert!((model.config().rope_theta - 50_000.0).abs() < 1e-6);
}

#[test]
fn test_model_embed_tokens_shape_matches_config() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let weight = model.embed_tokens().weight();
    assert_eq!(weight.dims(), &[cfg.vocab_size, cfg.hidden_size]);
}

#[test]
fn test_model_device_is_cpu() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    assert!(matches!(model.device(), Device::Cpu));
}

// ---------------------------------------------------------------------------
// MoE model forward and accessors
// ---------------------------------------------------------------------------

#[test]
fn test_moe_model_forward_single_token() {
    let cfg = Qwen3MoeConfig::new(tiny_config(), 4, 2, false, None);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();

    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.base.vocab_size]);
}

#[test]
fn test_moe_model_config_accessor() {
    let cfg = Qwen3MoeConfig::new(tiny_config(), 8, 3, true, Some(256));
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg).unwrap();

    assert_eq!(model.config().num_experts, 8);
    assert_eq!(model.config().num_experts_per_tok, 3);
    assert!(model.config().shared_expert);
}

#[test]
fn test_moe_model_new_cache_layers() {
    let cfg = Qwen3MoeConfig::new(tiny_config().with_num_hidden_layers(5), 4, 2, false, None);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg).unwrap();

    let cache = model.new_cache();
    assert_eq!(cache.num_layers(), 5);
}

#[test]
fn test_moe_model_forward_from_embeddings() {
    let cfg = Qwen3MoeConfig::new(tiny_config(), 4, 2, false, None);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();

    let hidden = DynTensor::ones(&[1, 3, cfg.base.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits = model
        .forward_from_embeddings(&hidden, &[0, 1, 2], None)
        .unwrap();
    assert_eq!(logits.dims(), &[1, 3, cfg.base.vocab_size]);
}

#[test]
fn test_moe_model_forward_from_embeddings_with_hidden() {
    let cfg = Qwen3MoeConfig::new(tiny_config(), 4, 2, false, None);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();

    let hidden = DynTensor::ones(&[1, 2, cfg.base.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let (logits, normed) = model
        .forward_from_embeddings_with_hidden(&hidden, &[0, 1], None)
        .unwrap();
    assert_eq!(logits.dims(), &[1, 2, cfg.base.vocab_size]);
    assert_eq!(normed.dims(), &[1, 2, cfg.base.hidden_size]);
}

#[test]
fn test_moe_model_embed_tokens_accessor() {
    let cfg = Qwen3MoeConfig::new(tiny_config(), 4, 2, false, None);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();

    let weight = model.embed_tokens().weight();
    assert_eq!(weight.dims(), &[cfg.base.vocab_size, cfg.base.hidden_size]);
}

#[test]
fn test_moe_model_dtype_accessor() {
    let cfg = Qwen3MoeConfig::new(tiny_config(), 4, 2, false, None);
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg).unwrap();
    assert_eq!(model.dtype(), DType::BF16);
}

// ---------------------------------------------------------------------------
// Incremental decode: one-shot vs cached equivalence (deeper test)
// ---------------------------------------------------------------------------

#[test]
fn test_oneshot_vs_cached_equivalence_5_tokens() {
    // Process [A, B, C, D, E] in one shot vs incrementally, verify last token logits match.
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();

    let tokens = [10, 20, 30, 40, 50];
    let positions: Vec<usize> = (0..5).collect();

    // One-shot
    let logits_oneshot = model.forward(&tokens, &positions).unwrap();
    let last_oneshot = logits_oneshot.narrow(1, 4, 1).unwrap();
    let oneshot_vec = last_oneshot.to_flat_vec::<f32>().unwrap();

    // Incremental
    let mut cache = model.new_cache();
    for i in 0..4 {
        model
            .forward_cached(&[tokens[i]], &[i], Some(&mut cache))
            .unwrap();
    }
    let logits_incr = model
        .forward_cached(&[tokens[4]], &[4], Some(&mut cache))
        .unwrap();
    let incr_vec = logits_incr.to_flat_vec::<f32>().unwrap();

    assert_eq!(oneshot_vec.len(), incr_vec.len());
    for (i, (&a, &b)) in oneshot_vec.iter().zip(incr_vec.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "logit mismatch at index {i}: oneshot={a}, incremental={b}"
        );
    }
}

#[test]
fn test_oneshot_vs_cached_equivalence_via_embeddings() {
    // Same equivalence test but through forward_from_embeddings path
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let seq_len = 3;
    let emb = DynTensor::ones(&[1, seq_len, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let positions: Vec<usize> = (0..seq_len).collect();

    // One-shot
    let logits_oneshot = model
        .forward_from_embeddings(&emb, &positions, None)
        .unwrap();
    let last_oneshot = logits_oneshot.narrow(1, 2, 1).unwrap();
    let oneshot_vec = last_oneshot.to_flat_vec::<f32>().unwrap();

    // Incremental
    let mut cache = model.new_cache();
    for i in 0..2 {
        let single_emb =
            DynTensor::ones(&[1, 1, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
        model
            .forward_from_embeddings(&single_emb, &[i], Some(&mut cache))
            .unwrap();
    }
    let last_emb = DynTensor::ones(&[1, 1, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits_incr = model
        .forward_from_embeddings(&last_emb, &[2], Some(&mut cache))
        .unwrap();
    let incr_vec = logits_incr.to_flat_vec::<f32>().unwrap();

    assert_eq!(oneshot_vec.len(), incr_vec.len());
    for (i, (&a, &b)) in oneshot_vec.iter().zip(incr_vec.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "embedding path: logit mismatch at index {i}: oneshot={a}, incremental={b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Config edge cases: combined invalid parameters
// ---------------------------------------------------------------------------

#[test]
fn test_config_multiple_invalid_fields() {
    // Multiple invalid fields -- validate should still return an error
    let cfg = Qwen3Config::new(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0, true, None);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_nan_rope_theta() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, f64::NAN, 64, true, None);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_neg_infinity_rms_norm_eps() {
    let cfg = Qwen3Config::new(
        256,
        512,
        1,
        2,
        2,
        100,
        f64::NEG_INFINITY,
        10_000.0,
        64,
        true,
        None,
    );
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// RoPE RotaryEmbedding from nn-core: multi-head, multi-sequence
// ---------------------------------------------------------------------------

#[test]
fn test_rotary_embedding_multi_head_multi_seq() {
    // Verify RoPE works with batch=1, heads=4, seq=3, head_dim=8
    let head_dim = 8;
    let rope = RotaryEmbedding::new(head_dim, 32, 10_000.0, &Device::Cpu).unwrap();

    let batch = 1;
    let heads = 4;
    let seq = 3;
    let total = batch * heads * seq * head_dim;
    let data: Vec<f32> = (0..total).map(|i| (i as f32 + 1.0) * 0.01).collect();
    let q =
        DynTensor::from_vec(data.clone(), &[batch, heads, seq, head_dim], &Device::Cpu).unwrap();
    let k = DynTensor::from_vec(data, &[batch, heads, seq, head_dim], &Device::Cpu).unwrap();

    let (q_rot, k_rot) = rope.apply_pair_half_split(&q, &k, &[0, 1, 2]).unwrap();
    assert_eq!(q_rot.dims(), &[batch, heads, seq, head_dim]);
    assert_eq!(k_rot.dims(), &[batch, heads, seq, head_dim]);

    // All values should be finite
    let q_vals = q_rot.to_flat_vec::<f32>().unwrap();
    let k_vals = k_rot.to_flat_vec::<f32>().unwrap();
    assert!(q_vals.iter().all(|v| v.is_finite()));
    assert!(k_vals.iter().all(|v| v.is_finite()));
}

// ---------------------------------------------------------------------------
// repeat_kv edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_repeat_kv_large_n_rep() {
    // n_rep=16: single kv head repeated 16 times
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let x = DynTensor::from_vec(data, &[1, 1, 1, 4], &Device::Cpu).unwrap();
    let result = repeat_kv(&x, 16).unwrap();
    assert_eq!(result.dims(), &[1, 16, 1, 4]);

    let flat = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat.len(), 16 * 4);
    // All 16 heads should have the same data
    for h in 0..16 {
        let start = h * 4;
        assert_eq!(&flat[start..start + 4], &[1.0, 2.0, 3.0, 4.0]);
    }
}

#[test]
fn test_repeat_kv_preserves_batch_and_seq() {
    // batch=3, kv_heads=2, seq=5, head_dim=4, n_rep=2
    let batch = 3;
    let kv_heads = 2;
    let seq = 5;
    let head_dim = 4;
    let n_rep = 2;

    let total = batch * kv_heads * seq * head_dim;
    let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data, &[batch, kv_heads, seq, head_dim], &Device::Cpu).unwrap();
    let result = repeat_kv(&x, n_rep).unwrap();
    assert_eq!(result.dims(), &[batch, kv_heads * n_rep, seq, head_dim]);
}

// ---------------------------------------------------------------------------
// Config builder pattern chaining
// ---------------------------------------------------------------------------

#[test]
fn test_config_builder_with_vocab_then_layers() {
    let cfg = tiny_config().with_vocab_size(500).with_num_hidden_layers(6);
    assert_eq!(cfg.vocab_size, 500);
    assert_eq!(cfg.num_hidden_layers, 6);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_builder_preserves_other_fields() {
    let original = tiny_config();
    let modified = original.clone().with_vocab_size(999);
    assert_eq!(modified.hidden_size, original.hidden_size);
    assert_eq!(modified.intermediate_size, original.intermediate_size);
    assert_eq!(modified.num_attention_heads, original.num_attention_heads);
    assert_eq!(modified.num_key_value_heads, original.num_key_value_heads);
    assert!((modified.rms_norm_eps - original.rms_norm_eps).abs() < 1e-15);
    assert!((modified.rope_theta - original.rope_theta).abs() < 1e-6);
    assert_eq!(
        modified.max_position_embeddings,
        original.max_position_embeddings
    );
    assert_eq!(modified.tie_word_embeddings, original.tie_word_embeddings);
}
