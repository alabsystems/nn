// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for nn-gptoss.
//!
//! Tests core model functionality without requiring real weights.

use nn_core::DType;
use nn_gptoss::{
    estimate_kv_cache_memory, estimate_model_memory, estimate_mxfp4_memory,
    sampling::{
        apply_repetition_penalty, apply_temperature, apply_top_k, apply_top_p, sample_token,
        SamplingConfig,
    },
    AgentConfig, ContextManager, GenerateConfig, GptOssConfig, GptOssError, LayerType, SearchTool,
    StreamingConfig,
};

// ---------------------------------------------------------------------------
// Config tests
// ---------------------------------------------------------------------------

#[test]
fn test_config_preset_validates() {
    let cfg = GptOssConfig::gptoss_20b();
    cfg.validate().expect("preset should validate");
}

#[test]
fn test_config_dimensions() {
    let cfg = GptOssConfig::gptoss_20b();
    assert_eq!(cfg.hidden_size, 2880);
    assert_eq!(cfg.num_attention_heads, 64);
    assert_eq!(cfg.num_key_value_heads, 8);
    assert_eq!(cfg.head_dim, 64);
    assert_eq!(cfg.num_hidden_layers, 24);
    assert_eq!(cfg.num_local_experts, 32);
    assert_eq!(cfg.experts_per_token, 4);
}

#[test]
fn test_config_attn_dim() {
    let cfg = GptOssConfig::gptoss_20b();
    // attn_dim = num_heads * head_dim = 64 * 64 = 4096 > hidden=2880
    assert_eq!(cfg.num_attention_heads * cfg.head_dim, 4096);
}

#[test]
fn test_config_gqa_groups() {
    let cfg = GptOssConfig::gptoss_20b();
    // GQA: 64 Q heads / 8 KV heads = 8 groups
    assert_eq!(cfg.num_attention_heads % cfg.num_key_value_heads, 0);
    assert_eq!(cfg.num_attention_heads / cfg.num_key_value_heads, 8);
}

#[test]
fn test_config_layer_types() {
    let cfg = GptOssConfig::gptoss_20b();
    assert_eq!(cfg.layer_types.len(), 24);
    // Should alternate sliding/full
    for (i, lt) in cfg.layer_types.iter().enumerate() {
        let expected = if i % 2 == 0 {
            LayerType::SlidingAttention
        } else {
            LayerType::FullAttention
        };
        assert_eq!(*lt, expected, "layer {i} type mismatch");
    }
}

#[test]
fn test_config_rope_scaling() {
    let cfg = GptOssConfig::gptoss_20b();
    assert!(cfg.rope_scaling.is_some(), "should have YaRN scaling");
    assert_eq!(cfg.max_position_embeddings, 131072);
    assert_eq!(cfg.rope_theta, 150_000.0);
}

#[test]
fn test_config_swiglu_limit() {
    let cfg = GptOssConfig::gptoss_20b();
    assert_eq!(cfg.swiglu_limit, 7.0);
}

#[test]
fn test_config_sliding_window() {
    let cfg = GptOssConfig::gptoss_20b();
    assert_eq!(cfg.sliding_window, 128);
}

#[test]
fn test_config_vocab() {
    let cfg = GptOssConfig::gptoss_20b();
    assert_eq!(cfg.vocab_size, 201088);
    assert!(!cfg.tie_word_embeddings);
}

#[test]
fn test_config_attention_bias() {
    let cfg = GptOssConfig::gptoss_20b();
    assert!(cfg.attention_bias, "gpt-oss uses attention bias on Q/K/V/O");
}

// ---------------------------------------------------------------------------
// KV cache tests
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_new() {
    let cache = nn_gptoss::GptOssKvCache::new(&GptOssConfig::gptoss_20b());
    assert_eq!(cache.inner().num_layers(), 24);
}

#[test]
fn test_kv_cache_empty_seq_len() {
    let cache = nn_gptoss::GptOssKvCache::new(&GptOssConfig::gptoss_20b());
    assert_eq!(cache.inner().seq_len(), 0);
}

// ---------------------------------------------------------------------------
// MXFP4 tests
// ---------------------------------------------------------------------------

#[test]
fn test_mxfp4_quantize_dequantize_roundtrip() {
    use nn_gptoss::Mxfp4Tensor;

    // Small tensor for roundtrip test
    let data = vec![0.5f32, -0.25, 1.0, 0.0, -1.0, 0.75, -0.5, 0.125];
    let quantized = Mxfp4Tensor::quantize(&data, &[data.len()]);
    let dequantized = quantized.dequantize();

    // Roundtrip should preserve shape
    assert_eq!(dequantized.len(), data.len());

    // Values should be close (within quantization error)
    for (orig, deq) in data.iter().zip(dequantized.iter()) {
        let err = (orig - deq).abs();
        // MXFP4 has limited precision, but error should be bounded
        assert!(
            err < 2.0,
            "roundtrip error too large: orig={orig}, deq={deq}, err={err}"
        );
    }
}

#[test]
fn test_mxfp4_zeros() {
    use nn_gptoss::Mxfp4Tensor;

    let data = vec![0.0f32; 16];
    let quantized = Mxfp4Tensor::quantize(&data, &[data.len()]);
    let dequantized = quantized.dequantize();

    for val in &dequantized {
        assert_eq!(*val, 0.0, "zero should roundtrip exactly");
    }
}

// ---------------------------------------------------------------------------
// Error handling tests
// ---------------------------------------------------------------------------

#[test]
fn test_config_zero_heads_rejected() {
    let cfg = GptOssConfig::gptoss_20b();
    // Create invalid config with zero heads
    let invalid = GptOssConfig::new(
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_hidden_layers,
        0, // zero heads
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.vocab_size,
        cfg.rms_norm_eps,
        cfg.rope_theta,
        cfg.max_position_embeddings,
        cfg.tie_word_embeddings,
        cfg.rope_scaling,
        cfg.attention_bias,
        cfg.num_local_experts,
        cfg.experts_per_token,
        cfg.swiglu_limit,
        cfg.layer_types.clone(),
        cfg.sliding_window,
        cfg.eos_token_id,
    );
    assert!(invalid.validate().is_err());
}

#[test]
fn test_config_gqa_misalignment_rejected() {
    let cfg = GptOssConfig::gptoss_20b();
    // Create config where heads % kv_heads != 0
    let invalid = GptOssConfig::new(
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_hidden_layers,
        7, // not divisible by 8
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.vocab_size,
        cfg.rms_norm_eps,
        cfg.rope_theta,
        cfg.max_position_embeddings,
        cfg.tie_word_embeddings,
        cfg.rope_scaling,
        cfg.attention_bias,
        cfg.num_local_experts,
        cfg.experts_per_token,
        cfg.swiglu_limit,
        cfg.layer_types.clone(),
        cfg.sliding_window,
        cfg.eos_token_id,
    );
    assert!(invalid.validate().is_err());
}

// ---------------------------------------------------------------------------
// Tool parser integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_tool_parser_search_tool_variants() {
    // Verify that the public SearchTool enum covers all expected variants
    // and Debug formatting is non-empty.
    let search = SearchTool::SearchCorpus {
        query: "test query".into(),
    };
    let grep = SearchTool::GrepCorpus {
        pattern: "fn main".into(),
    };
    let read = SearchTool::ReadDocument {
        doc_id: "doc-1".into(),
    };
    let prune = SearchTool::PruneChunks {
        chunk_ids: vec!["c1".into(), "c2".into()],
    };

    assert!(!format!("{search:?}").is_empty());
    assert!(!format!("{grep:?}").is_empty());
    assert!(!format!("{read:?}").is_empty());
    assert!(!format!("{prune:?}").is_empty());
}

#[test]
fn test_agent_config_validation_defaults() {
    let cfg = AgentConfig::new();
    assert_eq!(cfg.token_budget, 32_768);
    assert_eq!(cfg.soft_threshold, 24_576);
    assert_eq!(cfg.max_turns, 128);
    assert!(
        cfg.soft_threshold < cfg.token_budget,
        "soft threshold must be below hard budget"
    );
}

#[test]
fn test_agent_config_builder_chain() {
    let cfg = AgentConfig::new()
        .with_token_budget(8192)
        .with_soft_threshold(6144)
        .with_max_turns(32);
    assert_eq!(cfg.token_budget, 8192);
    assert_eq!(cfg.soft_threshold, 6144);
    assert_eq!(cfg.max_turns, 32);
}

#[test]
fn test_context_manager_budget_enforcement_integration() {
    let cfg = AgentConfig::new().with_token_budget(100);
    let mut cm = ContextManager::new(cfg);
    cm.add_chunk("c1".into(), "data".into(), 50)
        .expect("should fit in budget");
    assert_eq!(cm.token_count(), 50);
    assert!(!cm.is_over_budget());

    // Add another chunk that exceeds budget
    let result = cm.add_chunk("c2".into(), "more data".into(), 60);
    assert!(result.is_err(), "should reject chunk exceeding budget");
    assert_eq!(cm.token_count(), 50);
    assert_eq!(cm.chunk_count(), 1);
}

#[test]
fn test_context_manager_prune_and_rebuild() {
    let cfg = AgentConfig::new().with_token_budget(200);
    let mut cm = ContextManager::new(cfg);
    cm.add_chunk("a".into(), "alpha".into(), 30).unwrap();
    cm.add_chunk("b".into(), "beta".into(), 40).unwrap();
    cm.add_chunk("c".into(), "gamma".into(), 50).unwrap();
    assert_eq!(cm.token_count(), 120);

    cm.prune(&["b".into()]);
    assert_eq!(cm.token_count(), 80);
    assert_eq!(cm.chunk_count(), 2);

    let ctx = cm.build_context();
    assert!(ctx.contains("[chunk:a]"));
    assert!(ctx.contains("alpha"));
    assert!(!ctx.contains("[chunk:b]"));
    assert!(ctx.contains("[chunk:c]"));
    assert!(ctx.contains("gamma"));
}

#[test]
fn test_context_manager_dedup_after_prune() {
    let cfg = AgentConfig::new().with_token_budget(200);
    let mut cm = ContextManager::new(cfg);
    cm.add_chunk("x".into(), "first".into(), 10).unwrap();
    cm.prune(&["x".into()]);
    assert_eq!(cm.chunk_count(), 0);
    assert_eq!(cm.token_count(), 0);

    // Re-adding same ID should be a no-op (seen_ids persists)
    cm.add_chunk("x".into(), "second".into(), 20).unwrap();
    assert_eq!(cm.chunk_count(), 0, "pruned IDs remain in seen set");
    assert_eq!(cm.token_count(), 0);
}

#[test]
fn test_context_manager_soft_threshold_advisory() {
    let cfg = AgentConfig::new()
        .with_token_budget(100)
        .with_soft_threshold(60);
    let mut cm = ContextManager::new(cfg);

    cm.add_chunk("c1".into(), "d".into(), 30).unwrap();
    assert!(!cm.is_over_soft_threshold());

    cm.add_chunk("c2".into(), "d".into(), 35).unwrap();
    assert!(cm.is_over_soft_threshold(), "65 > 60 soft threshold");
    assert!(!cm.is_over_budget(), "65 <= 100 hard budget");
}

#[test]
fn test_context_manager_chunks_iterator_order() {
    let cfg = AgentConfig::new().with_token_budget(500);
    let mut cm = ContextManager::new(cfg);
    cm.add_chunk("first".into(), "aaa".into(), 10).unwrap();
    cm.add_chunk("second".into(), "bbb".into(), 20).unwrap();
    cm.add_chunk("third".into(), "ccc".into(), 30).unwrap();

    let pairs: Vec<_> = cm.chunks().collect();
    assert_eq!(pairs.len(), 3);
    assert_eq!(pairs[0], ("first", "aaa"));
    assert_eq!(pairs[1], ("second", "bbb"));
    assert_eq!(pairs[2], ("third", "ccc"));
}

// ===========================================================================
// Sampling tests (5)
// ===========================================================================

#[test]
fn test_sampling_config_defaults_match_expected() {
    let cfg = SamplingConfig::default();
    assert!((cfg.temperature - 0.7).abs() < f32::EPSILON);
    assert_eq!(cfg.top_p, Some(0.9));
    assert_eq!(cfg.top_k, Some(50));
    assert!((cfg.repetition_penalty - 1.0).abs() < f32::EPSILON);
    assert!((cfg.frequency_penalty - 0.0).abs() < f32::EPSILON);

    // Greedy config
    let greedy = SamplingConfig::greedy();
    assert!(greedy.temperature < 1e-6);
    assert_eq!(greedy.top_k, Some(1));
    assert!(greedy.top_p.is_none());
}

#[test]
fn test_temperature_scaling_higher_temp_flattens() {
    // Higher temperature divides logits by a larger value, making the
    // distribution more uniform. After temperature scaling, the gap
    // between max and min logit should shrink.
    let logits_orig = vec![1.0f32, 5.0, 3.0, 0.5];

    let mut low_temp = logits_orig.clone();
    apply_temperature(&mut low_temp, 0.5);

    let mut high_temp = logits_orig;
    apply_temperature(&mut high_temp, 2.0);

    // Gap between max and min after low temperature (more peaked)
    let low_gap = low_temp.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - low_temp.iter().copied().fold(f32::INFINITY, f32::min);

    // Gap between max and min after high temperature (flatter)
    let high_gap = high_temp.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - high_temp.iter().copied().fold(f32::INFINITY, f32::min);

    assert!(
        low_gap > high_gap,
        "lower temperature should produce larger gaps: low_gap={low_gap}, high_gap={high_gap}"
    );
}

#[test]
fn test_top_k_only_k_logits_remain_finite() {
    let mut logits = vec![1.0, 5.0, 3.0, 0.5, 4.0, 2.0, 6.0, 0.1];
    let k = 3;
    let candidates = apply_top_k(&mut logits, k);

    // Candidates should have at most k entries
    assert!(candidates.len() <= k);

    // Non-top-k logits should be set to NEG_INFINITY
    let neg_inf_count = logits.iter().filter(|&&v| v == f32::NEG_INFINITY).count();
    assert_eq!(
        neg_inf_count,
        logits.len() - k,
        "exactly {} logits should be filtered to -inf",
        logits.len() - k
    );

    // The top-k candidates should include the highest values (5.0, 6.0, 4.0)
    let candidate_indices: Vec<usize> = candidates.iter().map(|&(i, _)| i).collect();
    assert!(
        candidate_indices.contains(&6),
        "index 6 (value 6.0) should be in top-3"
    );
    assert!(
        candidate_indices.contains(&1),
        "index 1 (value 5.0) should be in top-3"
    );
    assert!(
        candidate_indices.contains(&4),
        "index 4 (value 4.0) should be in top-3"
    );
}

#[test]
fn test_top_p_nucleus_sampling_returns_subset() {
    // With a strongly peaked distribution, top-p=0.5 should keep very few tokens
    let mut logits = vec![0.0, 10.0, 0.0, 0.0, 0.0]; // Peaked at index 1
    let candidates = apply_top_p(&mut logits, 0.5);

    assert!(!candidates.is_empty(), "must return at least one candidate");
    // The top candidate should be index 1 (logit=10.0)
    assert_eq!(
        candidates[0].0, 1,
        "highest logit should be first candidate"
    );

    // With top_p=1.0, all tokens should be included
    let mut logits2 = vec![1.0, 2.0, 3.0];
    let all = apply_top_p(&mut logits2, 1.0);
    assert_eq!(all.len(), 3, "top_p=1.0 should keep all tokens");
}

#[test]
fn test_repetition_penalty_reduces_repeated_scores() {
    let logits_orig = vec![3.9, 4.0, 3.8];
    let penalty = 10.0;

    // Apply repetition penalty to token 1 (the highest)
    let mut logits = logits_orig.clone();
    apply_repetition_penalty(&mut logits, penalty, &[1]);

    // Token 1 was positive (4.0), so after penalty: 4.0 / 10.0 = 0.4
    assert!(
        (logits[1] - 0.4).abs() < 1e-5,
        "penalized token 1 should be 0.4, got {}",
        logits[1]
    );

    // Token 0 should be untouched
    assert!(
        (logits[0] - 3.9).abs() < 1e-5,
        "unpenalized token 0 should stay at 3.9"
    );

    // Greedy sampling should now pick token 0 (3.9) instead of token 1 (0.4)
    let cfg = SamplingConfig::greedy().with_repetition_penalty(penalty);
    let token = sample_token(&logits_orig, &cfg, &[1], 0);
    assert_eq!(
        token, 0,
        "with penalty on token 1, greedy should pick token 0"
    );
}

// ===========================================================================
// Benchmark estimation tests (3)
// ===========================================================================

#[test]
fn test_model_memory_gptoss_20b_f32_over_30gb() {
    let cfg = GptOssConfig::gptoss_20b();
    let mem = estimate_model_memory(&cfg, DType::F32).expect("should not overflow");
    let gb = mem as f64 / (1024.0 * 1024.0 * 1024.0);
    assert!(
        gb > 30.0,
        "F32 model memory should be >30GB for 20B params, got {gb:.1}GB"
    );
    assert!(
        gb < 200.0,
        "F32 model memory should be <200GB, got {gb:.1}GB"
    );
}

#[test]
fn test_kv_cache_memory_batch1_seq4096() {
    let cfg = GptOssConfig::gptoss_20b();
    let mem = estimate_kv_cache_memory(&cfg, 4096).expect("should not overflow");

    // KV cache for 24 layers, 8 KV heads, head_dim=64, F32 (4 bytes)
    // Full layers: 2 * 512 * 4096 * 4 = 16MB per full-attn layer
    // Sliding layers (capped at 128): 2 * 512 * 128 * 4 = 512KB per sliding layer
    // 12 full + 12 sliding: ~192MB + ~6MB = ~198MB
    assert!(mem > 0);
    let mb = mem as f64 / (1024.0 * 1024.0);
    assert!(
        mb > 10.0,
        "KV cache at seq=4096 should be >10MB, got {mb:.1}MB"
    );
    assert!(
        mb < 1000.0,
        "KV cache at seq=4096 should be <1000MB, got {mb:.1}MB"
    );
}

#[test]
fn test_mxfp4_compression_ratio_vs_f32() {
    let cfg = GptOssConfig::gptoss_20b();
    let f32_mem = estimate_model_memory(&cfg, DType::F32).unwrap();
    let mxfp4_mem = estimate_mxfp4_memory(&cfg).unwrap();

    assert!(mxfp4_mem < f32_mem, "MXFP4 should use less memory than F32");

    let ratio = f32_mem as f64 / mxfp4_mem as f64;
    // F32 is 4 bytes/elem. MXFP4 is ~0.53 bytes/elem for experts.
    // Since experts dominate, expect 3-8x compression overall.
    assert!(
        ratio > 2.0 && ratio < 10.0,
        "F32/MXFP4 compression ratio {ratio:.2}x outside expected [2, 10] range"
    );
}

// ===========================================================================
// Streaming/Generate config tests (3)
// ===========================================================================

#[test]
fn test_streaming_config_defaults() {
    let gen_cfg = GenerateConfig::default();
    let cfg = StreamingConfig::new(gen_cfg, 200_002);
    assert_eq!(cfg.eos_token_id, 200_002);
    assert_eq!(cfg.generate.max_tokens, 512);
    assert!((cfg.generate.temperature - 0.7).abs() < f32::EPSILON);
    assert!(!cfg.return_logits);

    // Greedy streaming config
    let greedy = StreamingConfig::greedy(256, 42);
    assert_eq!(greedy.generate.max_tokens, 256);
    assert_eq!(greedy.generate.temperature, 0.0);
    assert_eq!(greedy.eos_token_id, 42);
    assert!(!greedy.return_logits);

    // With return_logits
    let with_logits = StreamingConfig::greedy(100, 0).with_return_logits(true);
    assert!(with_logits.return_logits);
}

#[test]
fn test_generate_config_defaults_and_custom() {
    // Default config
    let cfg = GenerateConfig::default();
    assert_eq!(cfg.max_tokens, 512);
    assert!((cfg.temperature - 0.7).abs() < f32::EPSILON);
    assert_eq!(cfg.top_k, Some(50));
    assert_eq!(cfg.top_p, Some(0.9));
    assert!(cfg.repetition_penalty.is_none());
    cfg.validate().expect("default should validate");

    // Greedy config
    let greedy = GenerateConfig::greedy(128);
    assert_eq!(greedy.max_tokens, 128);
    assert_eq!(greedy.temperature, 0.0);
    assert!(greedy.top_k.is_none());
    assert!(greedy.top_p.is_none());
    greedy.validate().expect("greedy should validate");
}

#[test]
fn test_generate_config_validation() {
    // Default config validates
    let default_cfg = GenerateConfig::default();
    default_cfg
        .validate()
        .expect("default config should validate");

    // Greedy config validates
    let greedy_cfg = GenerateConfig::greedy(256);
    greedy_cfg
        .validate()
        .expect("greedy config should validate");

    // Greedy with max_tokens=0 validates (edge case: zero tokens generates nothing)
    let zero_cfg = GenerateConfig::greedy(0);
    zero_cfg
        .validate()
        .expect("zero max_tokens should validate");

    // Verify default has expected sampling parameters
    assert_eq!(default_cfg.top_k, Some(50));
    assert_eq!(default_cfg.top_p, Some(0.9));
    assert!((default_cfg.temperature - 0.7).abs() < f32::EPSILON);
    assert!(default_cfg.repetition_penalty.is_none());

    // Verify greedy disables all sampling
    assert!(greedy_cfg.top_k.is_none());
    assert!(greedy_cfg.top_p.is_none());
    assert_eq!(greedy_cfg.temperature, 0.0);
    assert!(greedy_cfg.repetition_penalty.is_none());
}

// ===========================================================================
// Error handling tests (2)
// ===========================================================================

#[test]
fn test_error_display_messages_contain_info() {
    // GptOssError variants should produce informative Display messages
    let config_err = GptOssError::InvalidConfig {
        reason: "head_dim must be > 0".into(),
    };
    let msg = config_err.to_string();
    assert!(msg.contains("invalid config"), "msg: {msg}");
    assert!(
        msg.contains("head_dim"),
        "msg should mention the field: {msg}"
    );

    let input_err = GptOssError::InvalidInput {
        reason: "prompt_ids must be non-empty".into(),
    };
    let msg2 = input_err.to_string();
    assert!(msg2.contains("invalid input"), "msg: {msg2}");
    assert!(
        msg2.contains("prompt_ids"),
        "msg should mention prompt_ids: {msg2}"
    );

    let weight_err = GptOssError::WeightLoad {
        reason: "missing tensor: lm_head.weight".into(),
    };
    let msg3 = weight_err.to_string();
    assert!(msg3.contains("weight load"), "msg: {msg3}");

    let nonfinite_err = GptOssError::NonFiniteOutput {
        stage: "lm_head",
        count: 42,
    };
    let msg4 = nonfinite_err.to_string();
    assert!(msg4.contains("non-finite"), "msg: {msg4}");
    assert!(msg4.contains("42"), "msg should contain count: {msg4}");
    assert!(msg4.contains("lm_head"), "msg should contain stage: {msg4}");
}

#[test]
fn test_cache_mismatch_error_shows_layer_counts() {
    let err = GptOssError::CacheMismatch {
        cache_layers: 12,
        model_layers: 24,
    };
    let msg = err.to_string();
    assert!(msg.contains("12"), "msg should show cache_layers: {msg}");
    assert!(msg.contains("24"), "msg should show model_layers: {msg}");
    assert!(
        msg.contains("cache mismatch"),
        "msg should say cache mismatch: {msg}"
    );
}

// ===========================================================================
// Config edge case tests (2)
// ===========================================================================

#[test]
fn test_config_with_different_layer_counts_validates() {
    // Reduced config: 4 layers instead of 24
    let cfg = GptOssConfig::gptoss_20b()
        .with_num_hidden_layers(4)
        .with_num_local_experts(8)
        .with_experts_per_token(2);
    cfg.validate().expect("reduced config should validate");
    assert_eq!(cfg.num_hidden_layers, 4);
    assert_eq!(cfg.layer_types.len(), 4);
    assert_eq!(cfg.num_local_experts, 8);
    assert_eq!(cfg.experts_per_token, 2);

    // Layer types should still alternate
    assert_eq!(cfg.layer_types[0], LayerType::SlidingAttention);
    assert_eq!(cfg.layer_types[1], LayerType::FullAttention);
    assert_eq!(cfg.layer_types[2], LayerType::SlidingAttention);
    assert_eq!(cfg.layer_types[3], LayerType::FullAttention);

    // But mismatched layer_types length should fail
    let mut bad = GptOssConfig::gptoss_20b();
    bad.num_hidden_layers = 10; // does NOT update layer_types (still 24)
    assert!(
        bad.validate().is_err(),
        "layer_types.len() != num_hidden_layers should fail"
    );
}

#[test]
fn test_config_preset_architecture_parameters() {
    // Verify that the 20b preset matches expected architectural parameters
    let cfg = GptOssConfig::gptoss_20b();

    // attn_dim = 64 * 64 = 4096 > hidden_size = 2880 (unusual: Q dimension > hidden)
    assert_eq!(cfg.attn_dim(), 4096);
    assert!(cfg.attn_dim() > cfg.hidden_size);

    // KV dimension = 8 * 64 = 512
    assert_eq!(cfg.kv_dim(), 512);

    // GQA repeat factor = 64/8 = 8
    let repeat = cfg
        .kv_repeat_factor()
        .expect("should compute repeat factor");
    assert_eq!(repeat, 8);

    // Sliding window is 128 tokens
    assert_eq!(cfg.sliding_window, 128);

    // EOS token
    assert_eq!(cfg.eos_token_id, 200_002);

    // YaRN scaling with 131K context
    assert_eq!(cfg.max_position_embeddings, 131_072);
    assert!(cfg.rope_scaling.is_some());
}
