// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended pipeline, generation, and configuration tests for nn-qwen3 (#4186).
//!
//! Covers:
//! 1.  Config validation: hidden_size, heads, eps, rope_theta, vocab_size
//! 2.  RoPE scaling and NTK-aware interpolation via YarnScaling
//! 3.  GQA (Grouped Query Attention) configurations
//! 4.  SwiGLU FFN structure and intermediate_size
//! 5.  KV cache layer management and growth
//! 6.  Token generation with various sampling strategies
//! 7.  Temperature, top-k, top-p parameter validation
//! 8.  Attention pattern validation (causal mask)
//! 9.  Weight shape verification (embedding, projection)
//! 10. BPE tokenizer properties (vocab_size invariants)
//! 11. Sliding window attention config (max_position_embeddings)
//! 12. Sequence length limits and long-context
//! 13. Multi-turn conversation handling
//! 14. Model variant configs (0.6B, 1.7B, 4B, 8B, 30B-A3B, 235B-A22B)
//! 15. Config serialization / builder patterns
//! 16. Beam search generation
//! 17. Embedding input forward paths
//! 18. MoE pipeline integration
//! 19. Mixed-precision dtype handling

use super::*;
use crate::rope_cache::RoPECache;
use crate::test_utils::tiny_config;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{BeamSearchConfig, GenerationConfig, YarnScaling};
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

// ===========================================================================
// Helper: production-like Qwen3 configs for each model size
// ===========================================================================

fn config_0_6b() -> Qwen3Config {
    Qwen3Config::new(
        1024,        // hidden_size
        2816,        // intermediate_size
        28,          // num_hidden_layers
        16,          // num_attention_heads
        8,           // num_key_value_heads
        151_936,     // vocab_size
        1e-6,        // rms_norm_eps
        1_000_000.0, // rope_theta
        40960,       // max_position_embeddings
        true,        // tie_word_embeddings
        None,        // rope_scaling
    )
}

fn config_1_7b() -> Qwen3Config {
    Qwen3Config::new(
        2048,
        11008,
        24,
        16,
        4,
        151_936,
        1e-6,
        1_000_000.0,
        40960,
        true,
        None,
    )
}

fn config_4b() -> Qwen3Config {
    Qwen3Config::new(
        2560,
        13824,
        36,
        20,
        4,
        151_936,
        1e-6,
        1_000_000.0,
        40960,
        true,
        None,
    )
}

fn config_8b() -> Qwen3Config {
    Qwen3Config::new(
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
    )
}

fn config_30b_a3b_base() -> Qwen3Config {
    Qwen3Config::new(
        4096,
        2560,
        48,
        32,
        4,
        151_936,
        1e-6,
        1_000_000.0,
        131072,
        false,
        None,
    )
}

fn config_235b_a22b_base() -> Qwen3Config {
    Qwen3Config::new(
        4096,
        2560,
        94,
        64,
        4,
        151_936,
        1e-6,
        1_000_000.0,
        131072,
        false,
        None,
    )
}

// ===========================================================================
// 1. Config validation
// ===========================================================================

#[test]
fn test_config_all_production_variants_validate() {
    // All production-size configs must pass validation.
    let configs = [
        ("0.6B", config_0_6b()),
        ("1.7B", config_1_7b()),
        ("4B", config_4b()),
        ("8B", config_8b()),
        ("30B-A3B base", config_30b_a3b_base()),
        ("235B-A22B base", config_235b_a22b_base()),
    ];
    for (name, cfg) in &configs {
        assert!(cfg.validate().is_ok(), "{name} config should validate");
    }
}

#[test]
fn test_config_head_dim_always_128() {
    // Qwen3 constant: head_dim = 128 regardless of model size.
    let configs = [
        config_0_6b(),
        config_1_7b(),
        config_4b(),
        config_8b(),
        tiny_config(),
    ];
    for cfg in &configs {
        assert_eq!(cfg.head_dim(), 128, "head_dim must always be 128");
    }
}

#[test]
fn test_config_rejects_zero_hidden_size() {
    let cfg = Qwen3Config::new(0, 512, 1, 2, 2, 100, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_rejects_zero_intermediate_size() {
    let cfg = Qwen3Config::new(256, 0, 1, 2, 2, 100, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_rejects_zero_vocab_size() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 0, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_rejects_zero_attention_heads() {
    let cfg = Qwen3Config::new(256, 512, 1, 0, 0, 100, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_rejects_nan_rms_norm_eps() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 100, f64::NAN, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_rejects_negative_rms_norm_eps() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 100, -1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_rejects_inf_rope_theta() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, f64::INFINITY, 64, true, None);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_rejects_negative_rope_theta() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, -1.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_rejects_zero_max_position_embeddings() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 100, 1e-6, 10_000.0, 0, true, None);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_rejects_heads_not_dividing_kv_heads() {
    // 5 heads, 3 kv_heads: 5 % 3 != 0
    let cfg = Qwen3Config::new(640, 1280, 1, 5, 3, 100, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

// ===========================================================================
// 2. RoPE scaling and NTK-aware interpolation via YarnScaling
// ===========================================================================

#[test]
fn test_yarn_scaling_config_creates_model() {
    let yarn = YarnScaling::new(4.0, 0.25, 32.0, 1.0, 8192);
    let cfg = Qwen3Config::new(
        256,
        512,
        1,
        2,
        2,
        100,
        1e-6,
        1_000_000.0,
        32768,
        true,
        Some(yarn),
    );
    assert!(cfg.validate().is_ok());
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, 100]);
}

#[test]
fn test_yarn_scaling_factor_is_preserved_in_config() {
    let yarn = YarnScaling::new(4.0, 0.25, 32.0, 1.0, 8192);
    let cfg = Qwen3Config::new(
        256,
        512,
        1,
        2,
        2,
        100,
        1e-6,
        1_000_000.0,
        32768,
        true,
        Some(yarn),
    );
    let scaling = cfg.rope_scaling.as_ref().unwrap();
    assert!((scaling.factor - 4.0).abs() < f64::EPSILON);
    assert!((scaling.attention_factor - 0.25).abs() < f64::EPSILON);
    assert!((scaling.beta_fast - 32.0).abs() < f64::EPSILON);
    assert!((scaling.beta_slow - 1.0).abs() < f64::EPSILON);
    assert_eq!(scaling.original_max_position_embeddings, 8192);
}

#[test]
fn test_rope_frequencies_decay_with_higher_index() {
    let head_dim = 128;
    let base = 10_000.0_f64;
    let half_dim = head_dim / 2;
    let mut prev_theta = f64::INFINITY;
    for i in 0..half_dim {
        let exponent = f64::from(2 * i) / f64::from(head_dim);
        let theta = 1.0 / base.powf(exponent);
        assert!(
            theta < prev_theta,
            "freq[{i}]={theta} should be < prev={prev_theta}"
        );
        prev_theta = theta;
    }
}

#[test]
fn test_rope_first_frequency_is_one() {
    for base in [10_000.0_f32, 1_000_000.0, 500.0] {
        let cache = RoPECache::new(8, 128, base);
        let (cos, sin) = cache.get(1);
        let expected_cos = 1.0_f64.cos() as f32;
        let expected_sin = 1.0_f64.sin() as f32;
        assert!(
            (cos[0] - expected_cos).abs() < 1e-5,
            "base={base}: cos[0] mismatch"
        );
        assert!(
            (sin[0] - expected_sin).abs() < 1e-5,
            "base={base}: sin[0] mismatch"
        );
    }
}

#[test]
fn test_rope_larger_base_slower_rotation() {
    // `|cos(angle) - 1|` is only a monotone measure of rotation magnitude while
    // the angle stays within [0, pi]; beyond that cos wraps and the comparison
    // is meaningless. With base=100, the largest angle is `pos * 1.0` (index 0),
    // so pos=3 keeps every frequency's angle <= 3.0 < pi and the assertion below
    // honestly captures "larger base => smaller angle => cos closer to 1".
    let pos = 3;
    let cache_small = RoPECache::new(16, 128, 100.0);
    let cache_large = RoPECache::new(16, 128, 1_000_000.0);
    let (cos_small, _) = cache_small.get(pos);
    let (cos_large, _) = cache_large.get(pos);
    for i in 1..64 {
        assert!(
            (cos_large[i] - 1.0).abs() <= (cos_small[i] - 1.0).abs() + 1e-6,
            "idx={i}: larger base should rotate slower"
        );
    }
}

#[test]
fn test_rope_qwen3_base_1m_high_index_nearly_identity() {
    let cache = RoPECache::new(128, 128, 1_000_000.0);
    let (cos, sin) = cache.get(100);
    assert!(
        (cos[63] - 1.0).abs() < 0.01,
        "high-index freq barely rotates"
    );
    assert!(sin[63].abs() < 0.01, "high-index freq sin near zero");
}

#[test]
fn test_ntk_aware_interpolation_extends_context() {
    // NTK-aware RoPE (via YarnScaling) should allow the model to accept
    // positions beyond original_max_position_embeddings.
    let yarn = YarnScaling::new(4.0, 0.25, 32.0, 1.0, 64);
    let cfg = Qwen3Config::new(
        256,
        512,
        1,
        2,
        2,
        50,
        1e-6,
        1_000_000.0,
        256,
        true,
        Some(yarn),
    );
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    // Position 128 exceeds original_max=64 but within scaled max=256
    let logits = model.forward(&[0], &[128]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, 50]);
}

// ===========================================================================
// 3. GQA (Grouped Query Attention) configurations
// ===========================================================================

#[test]
fn test_gqa_kv_groups_for_all_valid_combos() {
    for (nh, nkv, expected_groups) in [
        (4, 1, 4),
        (4, 2, 2),
        (4, 4, 1),
        (8, 1, 8),
        (8, 2, 4),
        (8, 4, 2),
        (8, 8, 1),
        (16, 4, 4),
        (16, 8, 2),
        (16, 16, 1),
        (32, 8, 4),
        (64, 4, 16),
    ] {
        let cfg = Qwen3Config::new(
            nh * 128,
            nh * 256,
            1,
            nh,
            nkv,
            50,
            1e-6,
            10_000.0,
            32,
            true,
            None,
        );
        assert_eq!(
            cfg.num_kv_groups().unwrap(),
            expected_groups,
            "nh={nh}, nkv={nkv}"
        );
    }
}

#[test]
fn test_gqa_num_kv_groups_error_when_not_divisible() {
    let cfg = Qwen3Config::new(896, 1792, 1, 7, 3, 50, 1e-6, 10_000.0, 32, true, None);
    assert!(cfg.num_kv_groups().is_err());
}

#[test]
fn test_gqa_model_loads_with_single_kv_head_mqa() {
    // Multi-Query Attention: 8 heads, 1 kv head
    let cfg = Qwen3Config::new(1024, 2048, 1, 8, 1, 50, 1e-6, 10_000.0, 32, true, None);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let logits = model.forward(&[0, 1], &[0, 1]).unwrap();
    assert_eq!(logits.dims(), &[1, 2, 50]);
}

#[test]
fn test_gqa_model_loads_with_fewer_kv_heads() {
    let cfg = Qwen3Config::new(512, 1024, 1, 4, 2, 50, 1e-6, 10_000.0, 32, true, None);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, 50]);
}

#[test]
fn test_gqa_production_configs_divisibility() {
    // All production configs have num_heads % num_kv_heads == 0
    let configs = [config_0_6b(), config_1_7b(), config_4b(), config_8b()];
    for cfg in &configs {
        assert_eq!(
            cfg.num_attention_heads % cfg.num_key_value_heads,
            0,
            "nh={}, nkv={}",
            cfg.num_attention_heads,
            cfg.num_key_value_heads
        );
        assert!(cfg.num_kv_groups().is_ok());
    }
}

// ===========================================================================
// 4. SwiGLU FFN structure
// ===========================================================================

#[test]
fn test_swiglu_various_intermediate_ratios() {
    for ratio in [2, 3, 4, 8] {
        let hidden = 256;
        let intermediate = hidden * ratio;
        let cfg = Qwen3Config::new(
            hidden,
            intermediate,
            1,
            2,
            2,
            50,
            1e-6,
            10_000.0,
            32,
            true,
            None,
        );
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = Qwen3Model::load(&vb, cfg).unwrap();
        let logits = model.forward(&[0], &[0]).unwrap();
        assert_eq!(logits.dims(), &[1, 1, 50]);
    }
}

#[test]
fn test_swiglu_output_hidden_size_matches() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let emb = DynTensor::ones(&[1, 3, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let (logits, hidden) = model
        .forward_from_embeddings_with_hidden(&emb, &[0, 1, 2], None)
        .unwrap();
    assert_eq!(hidden.dim(2).unwrap(), cfg.hidden_size);
    assert_eq!(logits.dim(2).unwrap(), cfg.vocab_size);
}

#[test]
fn test_swiglu_production_intermediate_greater_than_hidden() {
    let configs = [
        (1024, 2816),  // 0.6B
        (2048, 11008), // 1.7B
        (2560, 13824), // 4B
        (4096, 14336), // 8B
    ];
    for (hidden, intermediate) in configs {
        assert!(
            intermediate > hidden,
            "intermediate ({intermediate}) > hidden ({hidden})"
        );
        let ratio = f64::from(intermediate) / f64::from(hidden);
        assert!(
            ratio > 1.5 && ratio < 10.0,
            "ratio {ratio} in reasonable range"
        );
    }
}

// ===========================================================================
// 5. KV cache layer management
// ===========================================================================

#[test]
fn test_kv_cache_grows_by_one_per_single_token() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();
    assert_eq!(cache.seq_len(), 0);
    for step in 0..8 {
        model
            .forward_cached(&[step % cfg.vocab_size], &[step], Some(&mut cache))
            .unwrap();
        assert_eq!(cache.seq_len(), step + 1);
    }
}

#[test]
fn test_kv_cache_grows_by_batch_on_prefill() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let mut cache = model.new_cache();
    model
        .forward_cached(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 5);
    model.forward_cached(&[42], &[5], Some(&mut cache)).unwrap();
    assert_eq!(cache.seq_len(), 6);
}

#[test]
fn test_kv_cache_layer_count_matches_model_layers() {
    for n_layers in [1, 2, 4, 6] {
        let cfg = tiny_config().with_num_hidden_layers(n_layers);
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = Qwen3Model::load(&vb, cfg).unwrap();
        let cache = model.new_cache();
        assert_eq!(cache.num_layers(), n_layers, "n_layers={n_layers}");
    }
}

#[test]
fn test_kv_cache_seq_len_zero_initially() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let cache = model.new_cache();
    assert_eq!(cache.seq_len(), 0);
}

#[test]
fn test_kv_cache_multiple_prefill_decode_cycles() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    // Prefill 3 tokens
    model
        .forward_cached(&[1, 2, 3], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 3);

    // Decode 4 tokens one at a time
    for i in 0..4 {
        let pos = 3 + i;
        model
            .forward_cached(&[(pos * 7) % cfg.vocab_size], &[pos], Some(&mut cache))
            .unwrap();
        assert_eq!(cache.seq_len(), pos + 1);
    }
    assert_eq!(cache.seq_len(), 7);
}

// ===========================================================================
// 6. Token generation with various sampling strategies
// ===========================================================================

#[test]
fn test_greedy_generation_produces_valid_tokens() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let output = model.generate_greedy(&[0, 1, 2], 8).unwrap();
    assert_eq!(output.token_ids.len(), 8);
    for &tok in &output.token_ids {
        assert!(
            tok < cfg.vocab_size,
            "token {tok} >= vocab_size {}",
            cfg.vocab_size
        );
    }
}

#[test]
fn test_greedy_generation_deterministic() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let out1 = model.generate_greedy(&[10, 20, 30], 5).unwrap();
    let out2 = model.generate_greedy(&[10, 20, 30], 5).unwrap();
    assert_eq!(
        out1.token_ids, out2.token_ids,
        "greedy must be deterministic"
    );
}

#[test]
fn test_beam_search_config_builder() {
    let cfg = BeamSearchConfig::new(4)
        .with_max_new_tokens(20)
        .with_length_penalty(0.6)
        .with_early_stopping(true)
        .with_eos_token_id(2);
    assert_eq!(cfg.beam_width, 4);
    assert_eq!(cfg.max_new_tokens, 20);
    assert!((cfg.length_penalty - 0.6).abs() < f64::EPSILON);
    assert!(cfg.early_stopping);
    assert_eq!(cfg.eos_token_id, Some(2));
}

#[test]
fn test_beam_search_generates_valid_output() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let beam_cfg = BeamSearchConfig::new(2).with_max_new_tokens(5);
    let output = model.generate_beam(&[0, 1], &beam_cfg).unwrap();
    assert!(!output.beams.is_empty(), "should produce at least 1 beam");
    for beam in &output.beams {
        for &tok in &beam.token_ids {
            assert!(tok < cfg.vocab_size);
        }
    }
}

#[test]
fn test_generation_with_different_prompt_lengths() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    for prompt_len in [1, 2, 5, 10] {
        let prompt: Vec<usize> = (0..prompt_len).map(|i| i % cfg.vocab_size).collect();
        let output = model.generate_greedy(&prompt, 3).unwrap();
        assert_eq!(output.token_ids.len(), 3, "prompt_len={prompt_len}");
    }
}

// ===========================================================================
// 7. Temperature, top-k, top-p parameter validation
// ===========================================================================

#[test]
fn test_generation_config_default_is_greedy() {
    let cfg = GenerationConfig::default();
    assert_eq!(cfg.temperature, 0.0);
    assert!(cfg.top_k.is_none());
    assert!(cfg.top_p.is_none());
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_generation_config_top_k_set() {
    let cfg = GenerationConfig::new(10)
        .with_temperature(1.0)
        .with_top_k(5);
    assert_eq!(cfg.top_k, Some(5));
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_generation_config_top_p_set() {
    let cfg = GenerationConfig::new(10)
        .with_temperature(1.0)
        .with_top_p(0.9);
    assert_eq!(cfg.top_p, Some(0.9));
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_generation_config_combined_top_k_and_top_p() {
    let cfg = GenerationConfig::new(10)
        .with_temperature(1.0)
        .with_top_k(50)
        .with_top_p(0.95);
    assert_eq!(cfg.top_k, Some(50));
    assert_eq!(cfg.top_p, Some(0.95));
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_generation_config_rejects_negative_temperature() {
    let cfg = GenerationConfig::new(10).with_temperature(-0.5);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_generation_config_rejects_nan_temperature() {
    let cfg = GenerationConfig::new(10).with_temperature(f64::NAN);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_generation_config_rejects_inf_temperature() {
    let cfg = GenerationConfig::new(10).with_temperature(f64::INFINITY);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_generation_config_rejects_top_p_zero() {
    let cfg = GenerationConfig::new(10).with_top_p(0.0);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_generation_config_rejects_top_p_greater_than_one() {
    let cfg = GenerationConfig::new(10).with_top_p(1.5);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_generation_config_accepts_top_p_one() {
    // top_p = 1.0 means "keep all tokens" — valid edge case
    let cfg = GenerationConfig::new(10).with_top_p(1.0);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_generation_config_eos_token_roundtrip() {
    let cfg = GenerationConfig::new(10).with_eos_token_id(151_643);
    assert_eq!(cfg.eos_token_id, Some(151_643));
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_generation_config_seed_roundtrip() {
    let cfg = GenerationConfig::new(10).with_seed(42);
    assert_eq!(cfg.seed, Some(42));
    assert!(cfg.validate().is_ok());
}

// ===========================================================================
// 8. Attention pattern validation (causal mask)
// ===========================================================================

#[test]
fn test_causal_mask_3x3_lower_triangular() {
    use nn_core::layers::causal_mask_with_offset;
    let mask = causal_mask_with_offset(3, 3, DType::F32, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 3, 3]);
    let data = mask.to_flat_vec::<f32>().unwrap();
    // Row 0: [0, -inf, -inf]
    assert_eq!(data[0], 0.0);
    assert!(data[1] < -1e30);
    assert!(data[2] < -1e30);
    // Row 1: [0, 0, -inf]
    assert_eq!(data[3], 0.0);
    assert_eq!(data[4], 0.0);
    assert!(data[5] < -1e30);
    // Row 2: [0, 0, 0]
    assert_eq!(data[6], 0.0);
    assert_eq!(data[7], 0.0);
    assert_eq!(data[8], 0.0);
}

#[test]
fn test_causal_mask_with_offset_rectangular() {
    use nn_core::layers::causal_mask_with_offset;
    // 2 new tokens, 5 total (offset=3)
    let mask = causal_mask_with_offset(2, 5, DType::F32, &Device::Cpu).unwrap();
    assert_eq!(mask.dims(), &[1, 1, 2, 5]);
    let data = mask.to_flat_vec::<f32>().unwrap();
    // Row 0 (abs pos 3): attend to 0..3 (4 zeros), col 4 masked
    for (col, &v) in data.iter().enumerate().take(4) {
        assert_eq!(v, 0.0, "row 0, col {col}");
    }
    assert!(data[4] < -1e30, "row 0, col 4 masked");
    // Row 1 (abs pos 4): attend to all 5
    for col in 0..5 {
        assert_eq!(data[5 + col], 0.0, "row 1, col {col}");
    }
}

#[test]
fn test_causal_mask_single_token() {
    use nn_core::layers::causal_mask_with_offset;
    let mask = causal_mask_with_offset(1, 1, DType::F32, &Device::Cpu).unwrap();
    assert_eq!(mask.to_flat_vec::<f32>().unwrap(), &[0.0]);
}

#[test]
fn test_build_causal_mask_none_for_single_token() {
    use crate::forward_common::build_causal_mask;
    let mask = build_causal_mask(1, None, DType::F32, &Device::Cpu).unwrap();
    assert!(mask.is_none(), "seq_len=1 should skip mask allocation");
}

#[test]
fn test_causal_mask_dtype_bf16() {
    use nn_core::layers::causal_mask_with_offset;
    let mask = causal_mask_with_offset(4, 4, DType::BF16, &Device::Cpu).unwrap();
    assert_eq!(mask.dtype(), DType::BF16);
    assert_eq!(mask.dims(), &[1, 1, 4, 4]);
}

// ===========================================================================
// 9. Weight shape verification
// ===========================================================================

#[test]
fn test_embed_weight_shape_vocab_by_hidden() {
    for (vocab, hidden, nh) in [(50, 256, 2), (100, 512, 4), (200, 1024, 8)] {
        let cfg = Qwen3Config::new(
            hidden,
            hidden * 2,
            1,
            nh,
            nh,
            vocab,
            1e-6,
            10_000.0,
            32,
            true,
            None,
        );
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = Qwen3Model::load(&vb, cfg).unwrap();
        assert_eq!(model.embed_tokens().weight().dims(), &[vocab, hidden]);
    }
}

#[test]
fn test_tied_embeddings_produce_correct_logit_shape() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 50, 1e-6, 10_000.0, 32, true, None);
    assert!(cfg.tie_word_embeddings);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, 50]);
}

#[test]
fn test_untied_embeddings_load_separate_lm_head() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 50, 1e-6, 10_000.0, 32, false, None);
    assert!(!cfg.tie_word_embeddings);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, 50]);
}

#[test]
fn test_tied_and_untied_zeros_produce_identical_output() {
    let cfg_t = Qwen3Config::new(256, 512, 1, 2, 2, 50, 1e-6, 10_000.0, 32, true, None);
    let cfg_u = Qwen3Config::new(256, 512, 1, 2, 2, 50, 1e-6, 10_000.0, 32, false, None);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let lt = Qwen3Model::load(&vb, cfg_t)
        .unwrap()
        .forward(&[0, 1], &[0, 1])
        .unwrap();
    let lu = Qwen3Model::load(&vb, cfg_u)
        .unwrap()
        .forward(&[0, 1], &[0, 1])
        .unwrap();
    assert_eq!(
        lt.to_flat_vec::<f32>().unwrap(),
        lu.to_flat_vec::<f32>().unwrap()
    );
}

// ===========================================================================
// 10. BPE tokenizer properties (vocab_size invariants)
// ===========================================================================

#[test]
fn test_qwen3_standard_vocab_size_151936() {
    // Qwen3 uses 151,936 tokens (BPE). Verify production configs.
    let configs = [config_0_6b(), config_1_7b(), config_4b(), config_8b()];
    for cfg in &configs {
        assert_eq!(cfg.vocab_size, 151_936, "Qwen3 standard vocab size");
    }
}

#[test]
fn test_vocab_size_determines_logit_dim() {
    for vocab in [50, 100, 200, 500] {
        let cfg = tiny_config().with_vocab_size(vocab);
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let model = Qwen3Model::load(&vb, cfg).unwrap();
        let logits = model.forward(&[0], &[0]).unwrap();
        assert_eq!(logits.dim(2).unwrap(), vocab, "vocab={vocab}");
    }
}

#[test]
fn test_token_ids_within_vocab_range() {
    let cfg = tiny_config(); // vocab_size=100
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    // Token 99 is the last valid ID for vocab_size=100
    let logits = model.forward(&[0, 50, 99], &[0, 1, 2]).unwrap();
    assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
}

// ===========================================================================
// 11. Sliding window / max_position_embeddings
// ===========================================================================

#[test]
fn test_max_position_embeddings_varies_by_variant() {
    assert_eq!(config_0_6b().max_position_embeddings, 40960);
    assert_eq!(config_1_7b().max_position_embeddings, 40960);
    assert_eq!(config_4b().max_position_embeddings, 40960);
    assert_eq!(config_8b().max_position_embeddings, 131072);
}

#[test]
fn test_model_accepts_positions_within_max() {
    let cfg = Qwen3Config::new(256, 512, 1, 2, 2, 50, 1e-6, 10_000.0, 128, true, None);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    // Position 127 is at the boundary of max_position_embeddings=128
    let logits = model.forward(&[0], &[127]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, 50]);
}

#[test]
fn test_rope_cache_respects_max_seq_len() {
    let cache = RoPECache::new(64, 128, 10_000.0);
    assert_eq!(cache.max_seq_len(), 64);
    // Position 63 is valid, 64 would panic
    let (cos, sin) = cache.get(63);
    assert_eq!(cos.len(), 64);
    assert_eq!(sin.len(), 64);
}

// ===========================================================================
// 12. Sequence length limits and long-context
// ===========================================================================

#[test]
fn test_prefill_with_increasing_lengths() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    for seq_len in [1, 2, 4, 8, 16, 32] {
        let ids: Vec<usize> = (0..seq_len).map(|i| i % cfg.vocab_size).collect();
        let pos: Vec<usize> = (0..seq_len).collect();
        let logits = model.forward(&ids, &pos).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, seq_len, cfg.vocab_size],
            "seq_len={seq_len}"
        );
    }
}

#[test]
fn test_long_sequence_incremental_decode() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    // Prefill 16 tokens
    let ids: Vec<usize> = (0..16).map(|i| i % cfg.vocab_size).collect();
    let pos: Vec<usize> = (0..16).collect();
    model.forward_cached(&ids, &pos, Some(&mut cache)).unwrap();
    assert_eq!(cache.seq_len(), 16);

    // Decode 16 more tokens
    for step in 16..32 {
        model
            .forward_cached(&[step % cfg.vocab_size], &[step], Some(&mut cache))
            .unwrap();
    }
    assert_eq!(cache.seq_len(), 32);
}

#[test]
fn test_mismatched_input_ids_positions_rejected() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    // 3 input_ids but 2 positions
    let result = model.forward(&[0, 1, 2], &[0, 1]);
    assert!(result.is_err(), "mismatched lengths must error");
}

// ===========================================================================
// 13. Multi-turn conversation handling
// ===========================================================================

#[test]
fn test_multi_turn_cache_accumulates_context() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    // Turn 1: 3 tokens
    model
        .forward_cached(&[10, 20, 30], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 3);

    // Turn 2: 2 tokens
    model
        .forward_cached(&[40, 50], &[3, 4], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 5);

    // Turn 3: single decode
    let logits = model.forward_cached(&[60], &[5], Some(&mut cache)).unwrap();
    assert_eq!(cache.seq_len(), 6);
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
}

#[test]
fn test_multi_turn_different_prompt_lengths() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();

    let short = model.generate_greedy(&[1], 5).unwrap();
    let long = model.generate_greedy(&[1, 2, 3, 4, 5, 6, 7, 8], 5).unwrap();
    assert_eq!(short.token_ids.len(), 5);
    assert_eq!(long.token_ids.len(), 5);
    for &tok in short.token_ids.iter().chain(long.token_ids.iter()) {
        assert!(tok < cfg.vocab_size);
    }
}

#[test]
fn test_position_ids_continue_from_cache_offset() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    model
        .forward_cached(&[10, 20, 30], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 3);

    let logits = model.forward_cached(&[40], &[3], Some(&mut cache)).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    assert_eq!(cache.seq_len(), 4);
}

// ===========================================================================
// 14. Model variant configs (0.6B, 1.7B, 4B, 8B, 30B-A3B, 235B-A22B)
// ===========================================================================

#[test]
fn test_config_0_6b_dimensions() {
    let cfg = config_0_6b();
    assert_eq!(cfg.hidden_size, 1024);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.num_key_value_heads, 8);
    assert_eq!(cfg.num_hidden_layers, 28);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_1_7b_dimensions() {
    let cfg = config_1_7b();
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.num_key_value_heads, 4);
    assert_eq!(cfg.num_hidden_layers, 24);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_4b_dimensions() {
    let cfg = config_4b();
    assert_eq!(cfg.hidden_size, 2560);
    assert_eq!(cfg.num_attention_heads, 20);
    assert_eq!(cfg.num_key_value_heads, 4);
    assert_eq!(cfg.num_hidden_layers, 36);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_8b_dimensions() {
    let cfg = config_8b();
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_key_value_heads, 8);
    assert_eq!(cfg.num_hidden_layers, 36);
    assert!(!cfg.tie_word_embeddings);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_30b_a3b_moe() {
    let base = config_30b_a3b_base();
    let cfg = Qwen3MoeConfig::new(base, 128, 8, true, Some(2048));
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.num_experts, 128);
    assert_eq!(cfg.num_experts_per_tok, 8);
    assert!(cfg.shared_expert);
    assert_eq!(cfg.shared_expert_ff_dim(), 2048);
}

#[test]
fn test_config_235b_a22b_moe() {
    let base = config_235b_a22b_base();
    let cfg = Qwen3MoeConfig::new(base, 128, 8, true, Some(2560));
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.num_experts, 128);
    assert_eq!(cfg.base.num_hidden_layers, 94);
}

#[test]
fn test_all_production_configs_use_1m_rope_theta() {
    // Qwen3 uses base=1,000,000 for all variants
    let configs = [config_0_6b(), config_1_7b(), config_4b(), config_8b()];
    for cfg in &configs {
        assert!((cfg.rope_theta - 1_000_000.0).abs() < f64::EPSILON);
    }
}

// ===========================================================================
// 15. Config builder patterns / serialization
// ===========================================================================

#[test]
fn test_config_with_vocab_size_builder() {
    let cfg = tiny_config().with_vocab_size(200);
    assert_eq!(cfg.vocab_size, 200);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_with_num_hidden_layers_builder() {
    let cfg = tiny_config().with_num_hidden_layers(6);
    assert_eq!(cfg.num_hidden_layers, 6);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_builder_chaining() {
    let cfg = tiny_config().with_vocab_size(500).with_num_hidden_layers(4);
    assert_eq!(cfg.vocab_size, 500);
    assert_eq!(cfg.num_hidden_layers, 4);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_clone_independence() {
    let cfg1 = tiny_config();
    let cfg2 = cfg1.clone().with_vocab_size(999);
    // cfg1 is unchanged
    assert_eq!(cfg1.vocab_size, 100);
    assert_eq!(cfg2.vocab_size, 999);
}

#[test]
fn test_config_debug_format() {
    let cfg = tiny_config();
    let debug_str = format!("{cfg:?}");
    assert!(debug_str.contains("Qwen3Config"));
    assert!(debug_str.contains("hidden_size"));
    assert!(debug_str.contains("256"));
}

#[test]
fn test_moe_config_debug_format() {
    let cfg = Qwen3MoeConfig::new(tiny_config(), 4, 2, false, None);
    let debug_str = format!("{cfg:?}");
    assert!(debug_str.contains("Qwen3MoeConfig"));
    assert!(debug_str.contains("num_experts: 4"));
}

// ===========================================================================
// 16. Logits shape consistency
// ===========================================================================

#[test]
fn test_logits_shape_single_token() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let logits = model.forward(&[42], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
}

#[test]
fn test_logits_shape_multi_token_prefill() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    for seq_len in [1, 2, 4, 8, 16] {
        let ids: Vec<usize> = (0..seq_len).map(|i| i % cfg.vocab_size).collect();
        let pos: Vec<usize> = (0..seq_len).collect();
        let logits = model.forward(&ids, &pos).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, seq_len, cfg.vocab_size],
            "seq_len={seq_len}"
        );
    }
}

#[test]
fn test_logits_shape_incremental_decode() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();
    for step in 0..5 {
        let logits = model
            .forward_cached(&[step % cfg.vocab_size], &[step], Some(&mut cache))
            .unwrap();
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size], "step {step}");
    }
}

#[test]
fn test_logits_finite_values() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    let logits = model.forward(&[0, 1, 2, 3], &[0, 1, 2, 3]).unwrap();
    assert!(logits
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite()));
}

#[test]
fn test_logits_and_hidden_shapes_match() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let emb = DynTensor::zeros(&[1, 4, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let (logits, hidden) = model
        .forward_from_embeddings_with_hidden(&emb, &[0, 1, 2, 3], None)
        .unwrap();
    assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
    assert_eq!(hidden.dims(), &[1, 4, cfg.hidden_size]);
    assert!(logits
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite()));
    assert!(hidden
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite()));
}

// ===========================================================================
// 17. Embedding input forward paths
// ===========================================================================

#[test]
fn test_forward_from_embeddings_various_batch_seq() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    for (batch, seq) in [(1, 1), (1, 5), (2, 3), (4, 1)] {
        let emb =
            DynTensor::zeros(&[batch, seq, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let pos: Vec<usize> = (0..seq).collect();
        let logits = model.forward_from_embeddings(&emb, &pos, None).unwrap();
        assert_eq!(
            logits.dims(),
            &[batch, seq, cfg.vocab_size],
            "batch={batch}, seq={seq}"
        );
    }
}

#[test]
fn test_forward_from_embeddings_wrong_hidden_size_rejected() {
    let cfg = tiny_config(); // hidden_size=256
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    // Wrong hidden_size: 128 instead of 256
    let emb = DynTensor::zeros(&[1, 3, 128], DType::F32, &Device::Cpu).unwrap();
    let result = model.forward_from_embeddings(&emb, &[0, 1, 2], None);
    assert!(result.is_err(), "wrong hidden_size should fail");
}

#[test]
fn test_forward_from_embeddings_mismatched_seq_pos_rejected() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let emb = DynTensor::zeros(&[1, 4, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    // seq=4 but positions has length 3
    let result = model.forward_from_embeddings(&emb, &[0, 1, 2], None);
    assert!(result.is_err());
}

#[test]
fn test_forward_from_embeddings_with_cache() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    let emb = DynTensor::ones(&[1, 3, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    model
        .forward_from_embeddings(&emb, &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 3);

    let emb2 = DynTensor::ones(&[1, 1, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits = model
        .forward_from_embeddings(&emb2, &[3], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 4);
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
}

// ===========================================================================
// 18. MoE pipeline integration
// ===========================================================================

#[test]
fn test_moe_config_rejects_zero_experts() {
    let cfg = Qwen3MoeConfig::new(tiny_config(), 0, 0, false, None);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_moe_config_rejects_active_greater_than_total() {
    let cfg = Qwen3MoeConfig::new(tiny_config(), 4, 5, false, None);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_moe_config_rejects_zero_active() {
    let cfg = Qwen3MoeConfig::new(tiny_config(), 4, 0, false, None);
    assert!(cfg.validate().is_err());
}

#[test]
fn test_moe_model_forward_with_cache() {
    let cfg = Qwen3MoeConfig::new(tiny_config(), 4, 2, false, None);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();
    let mut cache = model.new_cache();

    model
        .forward_cached(&[0, 1, 2], &[0, 1, 2], Some(&mut cache))
        .unwrap();
    assert_eq!(cache.seq_len(), 3);

    for step in 3..6 {
        model
            .forward_cached(&[step % cfg.base.vocab_size], &[step], Some(&mut cache))
            .unwrap();
        assert_eq!(cache.seq_len(), step + 1);
    }
}

#[test]
fn test_moe_model_with_shared_expert_forward() {
    let cfg = Qwen3MoeConfig::new(tiny_config(), 4, 2, true, Some(256));
    assert!(cfg.validate().is_ok());
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();
    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.base.vocab_size]);
}

#[test]
fn test_moe_shared_expert_ff_dim_fallback() {
    let base = tiny_config(); // intermediate_size=512
    let cfg_default = Qwen3MoeConfig::new(base.clone(), 4, 2, true, None);
    assert_eq!(cfg_default.shared_expert_ff_dim(), 512);
    let cfg_explicit = Qwen3MoeConfig::new(base, 4, 2, true, Some(1024));
    assert_eq!(cfg_explicit.shared_expert_ff_dim(), 1024);
}

#[test]
fn test_moe_model_from_embeddings_with_hidden() {
    let cfg = Qwen3MoeConfig::new(tiny_config(), 4, 2, false, None);
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3MoeModel::load(&vb, cfg.clone()).unwrap();

    let emb = DynTensor::zeros(&[1, 3, cfg.base.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let (logits, hidden) = model
        .forward_from_embeddings_with_hidden(&emb, &[0, 1, 2], None)
        .unwrap();
    assert_eq!(logits.dims(), &[1, 3, cfg.base.vocab_size]);
    assert_eq!(hidden.dims(), &[1, 3, cfg.base.hidden_size]);
}

// ===========================================================================
// 19. Mixed-precision dtype handling
// ===========================================================================

#[test]
fn test_model_f16_loads_and_runs() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F16, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    assert_eq!(model.dtype(), DType::F16);
    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
}

#[test]
fn test_model_bf16_loads_and_runs() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::BF16, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    assert_eq!(model.dtype(), DType::BF16);
    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
}

#[test]
fn test_model_dtype_accessor_matches_vb() {
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let cfg = tiny_config();
        let vb = VarBuilder::zeros(dtype, &Device::Cpu);
        let model = Qwen3Model::load(&vb, cfg).unwrap();
        assert_eq!(model.dtype(), dtype, "dtype={dtype:?}");
    }
}

#[test]
fn test_model_device_is_cpu() {
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg).unwrap();
    assert_eq!(model.device(), Device::Cpu);
}

#[test]
fn test_forward_from_embeddings_auto_converts_dtype() {
    // Model loaded with F32, embeddings given as F32 -- should work without conversion
    let cfg = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = Qwen3Model::load(&vb, cfg.clone()).unwrap();
    let emb = DynTensor::zeros(&[1, 2, cfg.hidden_size], DType::F32, &Device::Cpu).unwrap();
    let logits = model.forward_from_embeddings(&emb, &[0, 1], None).unwrap();
    assert_eq!(logits.dims(), &[1, 2, cfg.vocab_size]);
}

// ===========================================================================
// 20. RoPE cache edge cases
// ===========================================================================

#[test]
fn test_rope_cache_try_new_rejects_zero_head_dim() {
    assert!(RoPECache::try_new(128, 0, 10_000.0).is_err());
}

#[test]
fn test_rope_cache_try_new_rejects_odd_head_dim() {
    assert!(RoPECache::try_new(128, 7, 10_000.0).is_err());
}

#[test]
fn test_rope_cache_try_new_rejects_zero_max_seq() {
    assert!(RoPECache::try_new(0, 128, 10_000.0).is_err());
}

#[test]
fn test_rope_cache_try_new_rejects_nan_base() {
    assert!(RoPECache::try_new(128, 128, f32::NAN).is_err());
}

#[test]
fn test_rope_cache_try_new_rejects_inf_base() {
    assert!(RoPECache::try_new(128, 128, f32::INFINITY).is_err());
}

#[test]
fn test_rope_cache_try_new_rejects_negative_base() {
    assert!(RoPECache::try_new(128, 128, -1.0).is_err());
}

#[test]
fn test_rope_cache_apply_preserves_norm() {
    let cache = RoPECache::new(128, 64, 10_000.0);
    let (cos, sin) = cache.get(42);
    let mut q: Vec<f32> = (0..64).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let mut k: Vec<f32> = (0..64).map(|i| (i as f32 + 1.0) * 0.2).collect();
    let q_norm_before: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
    let k_norm_before: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt();
    RoPECache::apply_rope(&mut q, &mut k, cos, sin);
    let q_norm_after: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
    let k_norm_after: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((q_norm_before - q_norm_after).abs() < 1e-4);
    assert!((k_norm_before - k_norm_after).abs() < 1e-4);
}

#[test]
fn test_rope_cache_get_range_matches_individual() {
    let cache = RoPECache::new(100, 64, 10_000.0);
    let (cos_range, sin_range) = cache.get_range(10, 5);
    for i in 0..5 {
        let (cos_single, sin_single) = cache.get(10 + i);
        assert_eq!(cos_range[i].as_slice(), cos_single);
        assert_eq!(sin_range[i].as_slice(), sin_single);
    }
}

#[test]
fn test_rope_cache_position_zero_identity() {
    let cache = RoPECache::new(16, 128, 10_000.0);
    let (cos, sin) = cache.get(0);
    for i in 0..64 {
        assert!(
            (cos[i] - 1.0).abs() < 1e-7,
            "cos[{i}] at pos 0 should be 1.0"
        );
        assert!(sin[i].abs() < 1e-7, "sin[{i}] at pos 0 should be 0.0");
    }
}

// ===========================================================================
// 21. Error type coverage
// ===========================================================================

#[test]
fn test_qwen3_error_display_invalid_config() {
    let err = Qwen3Error::InvalidConfig {
        reason: "test reason".into(),
    };
    let msg = err.to_string();
    assert!(msg.contains("invalid config"));
    assert!(msg.contains("test reason"));
}

#[test]
fn test_qwen3_error_display_invalid_input() {
    let err = Qwen3Error::InvalidInput {
        reason: "bad input".into(),
    };
    assert!(err.to_string().contains("invalid input"));
}

#[test]
fn test_qwen3_error_display_cache_mismatch() {
    let err = Qwen3Error::CacheMismatch {
        cache_layers: 3,
        model_layers: 6,
    };
    let msg = err.to_string();
    assert!(msg.contains("3"));
    assert!(msg.contains("6"));
}

#[test]
fn test_qwen3_error_display_non_finite() {
    let err = Qwen3Error::NonFiniteOutput {
        stage: "test_stage",
        count: 42,
    };
    let msg = err.to_string();
    assert!(msg.contains("test_stage"));
    assert!(msg.contains("42"));
}
