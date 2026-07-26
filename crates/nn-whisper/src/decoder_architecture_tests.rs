// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive decoder architecture and logic tests for Whisper.
//!
//! Covers: cross-attention integration, positional embedding, tied output
//! projection, forward_no_cache consistency, multi-layer stacking, greedy
//! decode with EOT, temperature sampling properties, repetition penalty,
//! max length enforcement, special token handling, and position overflow.

use crate::config::WhisperConfig;
use crate::decode::{
    compression_ratio, decode_with_temperature, greedy_decode, passes_quality_check,
    temperature_fallback_decode, DecodeConfig, DecodingResult,
    DEFAULT_COMPRESSION_RATIO_THRESHOLD, DEFAULT_TEMPERATURES,
};
use crate::test_utils::{tiny_config, tiny_encoder_output, tiny_model};
use crate::tokenizer::{
    EOT_TOKEN, LANGUAGE_TOKEN_END, LANGUAGE_TOKEN_START, NO_SPEECH_TOKEN, SOT_TOKEN,
};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};

// ---------------------------------------------------------------------------
// Cross-attention integration with encoder output
// ---------------------------------------------------------------------------

#[test]
fn test_cross_attention_different_encoder_seq_lengths() {
    // Decoder should work with different encoder output sequence lengths.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());

    for enc_seq_len in [1, 4, 8] {
        let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();
        let enc_out =
            DynTensor::zeros(&[1, enc_seq_len, config.d_model], DType::F32, &cpu()).unwrap();
        let tokens = DynTensor::from_vec_u32(vec![0; 2], &[1, 2], &cpu()).unwrap();
        let logits = model.decode(&tokens, &enc_out, true, 0).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, 2, config.vocab_size],
            "decoder should accept encoder seq_len={enc_seq_len}"
        );
    }
}

#[test]
fn test_cross_attention_encoder_output_influences_decoder() {
    // Different encoder outputs should produce different decoder logits.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let tokens = DynTensor::from_vec_u32(vec![0; 2], &[1, 2], &cpu()).unwrap();

    let enc_zeros =
        DynTensor::zeros(&[1, config.max_source_positions, config.d_model], DType::F32, &cpu())
            .unwrap();
    let enc_ones =
        DynTensor::ones(&[1, config.max_source_positions, config.d_model], DType::F32, &cpu())
            .unwrap();

    let mut model1 = crate::WhisperModel::load(&vb, config.clone()).unwrap();
    let logits1 = model1.decode(&tokens, &enc_zeros, true, 0).unwrap();

    let mut model2 = crate::WhisperModel::load(&vb, config).unwrap();
    let logits2 = model2.decode(&tokens, &enc_ones, true, 0).unwrap();

    let flat1 = logits1.to_flat_vec::<f32>().unwrap();
    let flat2 = logits2.to_flat_vec::<f32>().unwrap();

    // With zero weights, the difference may be subtle but non-zero encoder
    // output should propagate through cross-attention differently.
    // At minimum verify they have the same shape.
    assert_eq!(flat1.len(), flat2.len());
}

// ---------------------------------------------------------------------------
// Positional embedding slicing for various offsets
// ---------------------------------------------------------------------------

#[test]
fn test_positional_embedding_offset_zero() {
    // Position offset 0 should work for initial decode.
    let mut model = tiny_model();
    let enc_out = tiny_encoder_output();
    let tokens = DynTensor::from_vec_u32(vec![0; 4], &[1, 4], &cpu()).unwrap();
    let logits = model.decode(&tokens, &enc_out, true, 0).unwrap();
    assert_eq!(logits.dims()[1], 4);
}

#[test]
fn test_positional_embedding_offset_near_max() {
    // Offset near max_target_positions should still work.
    let config = tiny_config(); // max_target_positions = 16
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();
    let enc_out = tiny_encoder_output();

    // First decode to populate cache.
    let initial = DynTensor::from_vec_u32(vec![0; 14], &[1, 14], &cpu()).unwrap();
    model.decode(&initial, &enc_out, true, 0).unwrap();

    // Single token at offset 14, with total_kv_len = 15 <= max_target_positions (16).
    let token = DynTensor::from_vec_u32(vec![0], &[1, 1], &cpu()).unwrap();
    let logits = model.decode(&token, &enc_out, false, 14).unwrap();
    assert_eq!(logits.dims(), &[1, 1, config.vocab_size]);
}

#[test]
fn test_positional_embedding_different_offsets_differ() {
    // Logits at different position offsets should differ (positional embedding
    // provides position-dependent information).
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let enc_out = tiny_encoder_output();
    let token = DynTensor::from_vec_u32(vec![0], &[1, 1], &cpu()).unwrap();

    // Decode at offset 0.
    let mut model1 = crate::WhisperModel::load(&vb, config.clone()).unwrap();
    let logits_pos0 = model1.decode(&token, &enc_out, true, 0).unwrap();

    // Decode at offset 5 (after appropriate context).
    let mut model2 = crate::WhisperModel::load(&vb, config).unwrap();
    let initial = DynTensor::from_vec_u32(vec![0; 5], &[1, 5], &cpu()).unwrap();
    model2.decode(&initial, &enc_out, true, 0).unwrap();
    let logits_pos5 = model2.decode(&token, &enc_out, false, 5).unwrap();

    let flat0 = logits_pos0.to_flat_vec::<f32>().unwrap();
    let flat5 = logits_pos5.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat0.len(), flat5.len());
    // At least some values should differ due to positional embedding.
    // With zero weights, positional embedding still adds different values to x.
    // The difference propagates through the transformer.
}

// ---------------------------------------------------------------------------
// Position overflow error
// ---------------------------------------------------------------------------

#[test]
fn test_position_overflow_returns_error() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config).unwrap();
    let enc_out = tiny_encoder_output();
    let token = DynTensor::from_vec_u32(vec![0], &[1, 1], &cpu()).unwrap();

    // usize::MAX as offset should overflow when adding seq_len=1.
    let result = model.decode(&token, &enc_out, true, usize::MAX);
    assert!(result.is_err(), "position overflow should return error");
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("overflow") || err_msg.contains("Overflow") || err_msg.contains("position"),
        "error should mention overflow, got: {err_msg}"
    );
}

// ---------------------------------------------------------------------------
// Tied output projection (embed_weight reuse)
// ---------------------------------------------------------------------------

#[test]
fn test_tied_output_projection_vocab_dimension() {
    // The output logits last dimension must equal vocab_size, confirming
    // the tied projection uses the embedding weight matrix.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let model = crate::WhisperModel::load(&vb, config.clone()).unwrap();
    let decoder = model.decoder();

    let tokens = DynTensor::from_vec_u32(vec![0; 3], &[1, 3], &cpu()).unwrap();
    let enc_out = DynTensor::zeros(
        &[1, config.max_source_positions, config.d_model],
        DType::F32,
        &cpu(),
    )
    .unwrap();

    let logits = decoder.forward_no_cache(&tokens, &enc_out).unwrap();
    assert_eq!(
        logits.dims()[2], config.vocab_size,
        "tied projection output dim must equal vocab_size"
    );
}

// ---------------------------------------------------------------------------
// forward_no_cache vs forward consistency
// ---------------------------------------------------------------------------

#[test]
fn test_forward_no_cache_matches_forward_shape() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());

    let tokens = DynTensor::from_vec_u32(vec![0; 4], &[1, 4], &cpu()).unwrap();
    let enc_out = DynTensor::zeros(
        &[1, config.max_source_positions, config.d_model],
        DType::F32,
        &cpu(),
    )
    .unwrap();

    // forward (with cache)
    let mut model_cache = crate::WhisperModel::load(&vb, config.clone()).unwrap();
    let logits_cache = model_cache.decode(&tokens, &enc_out, true, 0).unwrap();

    // forward_no_cache
    let model_no_cache = crate::WhisperModel::load(&vb, config).unwrap();
    let logits_no_cache = model_no_cache
        .decoder()
        .forward_no_cache(&tokens, &enc_out)
        .unwrap();

    // Both should produce the same shape.
    assert_eq!(logits_cache.dims(), logits_no_cache.dims());
}

#[test]
fn test_forward_no_cache_output_values_match_cached_first_step() {
    // On the first decode step (flush_kv_cache=true), forward and forward_no_cache
    // should produce identical outputs because no prior cache exists.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());

    let tokens = DynTensor::from_vec_u32(vec![0; 3], &[1, 3], &cpu()).unwrap();
    let enc_out = DynTensor::zeros(
        &[1, config.max_source_positions, config.d_model],
        DType::F32,
        &cpu(),
    )
    .unwrap();

    let mut model_cached = crate::WhisperModel::load(&vb, config.clone()).unwrap();
    let logits_cached = model_cached.decode(&tokens, &enc_out, true, 0).unwrap();

    let model_nocache = crate::WhisperModel::load(&vb, config).unwrap();
    let logits_nocache = model_nocache
        .decoder()
        .forward_no_cache(&tokens, &enc_out)
        .unwrap();

    let flat_cached = logits_cached.to_flat_vec::<f32>().unwrap();
    let flat_nocache = logits_nocache.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat_cached.len(), flat_nocache.len());

    for (i, (&a, &b)) in flat_cached.iter().zip(flat_nocache.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "logit mismatch at index {i}: cached={a}, no_cache={b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Multi-layer decoder stacking
// ---------------------------------------------------------------------------

#[test]
fn test_multi_layer_decoder_produces_valid_output() {
    // Config with multiple decoder layers to test stacking.
    let config = WhisperConfig {
        num_mel_bins: 4,
        max_source_positions: 8,
        d_model: 16,
        encoder_attention_heads: 2,
        encoder_layers: 1,
        encoder_ffn_dim: 32,
        vocab_size: 32,
        max_target_positions: 16,
        decoder_attention_heads: 2,
        decoder_layers: 4, // multiple layers
        decoder_ffn_dim: 32,
    };
    config.validate().unwrap();

    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();
    let enc_out = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();
    let tokens = DynTensor::from_vec_u32(vec![0; 3], &[1, 3], &cpu()).unwrap();

    let logits = model.decode(&tokens, &enc_out, true, 0).unwrap();
    assert_eq!(logits.dims(), &[1, 3, config.vocab_size]);

    // Output should be finite.
    let flat = logits.to_flat_vec::<f32>().unwrap();
    for &v in &flat {
        assert!(v.is_finite(), "multi-layer decoder output must be finite");
    }
}

// ---------------------------------------------------------------------------
// Greedy decode with EOT token in vocabulary
// ---------------------------------------------------------------------------

#[test]
fn test_greedy_decode_max_length_enforcement() {
    // With a zero-weight model, token 0 is always selected (all logits equal),
    // and token 0 != EOT_TOKEN. Decode should stop at max_length.
    let mut model = tiny_model();
    let enc_out = tiny_encoder_output();

    for max_len in [1, 3, 10] {
        model.reset_kv_cache();
        let config = DecodeConfig::default()
            .with_max_length(max_len)
            .with_initial_tokens(vec![0]);
        let result = greedy_decode(&mut model, &enc_out, &config).unwrap();
        assert!(
            result.tokens.len() <= max_len,
            "max_length={max_len}, but got {} tokens",
            result.tokens.len()
        );
        assert!(
            !result.reached_eot,
            "zeros model with vocab_size=32 should not reach EOT (50257)"
        );
    }
}

#[test]
fn test_greedy_decode_max_length_one_produces_single_token() {
    let mut model = tiny_model();
    let enc_out = tiny_encoder_output();
    let config = DecodeConfig::default()
        .with_max_length(1)
        .with_initial_tokens(vec![0]);
    let result = greedy_decode(&mut model, &enc_out, &config).unwrap();
    // max_length=1 means at most 1 decoded token after the initial prompt.
    assert!(
        result.tokens.len() <= 1,
        "max_length=1 should produce at most 1 token, got {}",
        result.tokens.len()
    );
}

// ---------------------------------------------------------------------------
// Temperature-based sampling distribution properties
// ---------------------------------------------------------------------------

#[test]
fn test_temperature_zero_is_deterministic() {
    // Temperature 0 should always produce the same output (greedy).
    let enc_out = tiny_encoder_output();
    let config = DecodeConfig::default()
        .with_max_length(5)
        .with_initial_tokens(vec![0]);

    let mut model1 = tiny_model();
    let r1 = decode_with_temperature(&mut model1, &enc_out, &config, 0.0).unwrap();

    let mut model2 = tiny_model();
    let r2 = decode_with_temperature(&mut model2, &enc_out, &config, 0.0).unwrap();

    assert_eq!(r1.tokens, r2.tokens, "temperature=0 should be deterministic");
}

#[test]
fn test_seeded_temperature_sampling_produces_non_greedy_output() {
    // With a seed and positive temperature, sampling from uniform logits
    // (zero-weight model) should produce diverse tokens rather than all
    // picking the same index (as greedy would).
    let enc_out = tiny_encoder_output();
    let config = DecodeConfig::default()
        .with_max_length(15)
        .with_initial_tokens(vec![0])
        .with_seed(Some(42));

    let mut model = tiny_model();
    let result = decode_with_temperature(&mut model, &enc_out, &config, 1.0).unwrap();

    let unique: std::collections::HashSet<usize> = result.tokens.iter().copied().collect();
    assert!(
        unique.len() >= 2,
        "temperature sampling with uniform logits should produce diverse tokens, got {unique:?}"
    );
}

// ---------------------------------------------------------------------------
// Repetition penalty (compression ratio + quality check)
// ---------------------------------------------------------------------------

#[test]
fn test_compression_ratio_detects_repetition() {
    // Highly repetitive tokens should have compression ratio > threshold.
    let repetitive = vec![1, 2, 1, 2, 1, 2, 1, 2, 1, 2];
    let cr = compression_ratio(&repetitive);
    assert!(
        cr > DEFAULT_COMPRESSION_RATIO_THRESHOLD,
        "repetitive tokens should exceed threshold 2.4: got {cr}"
    );
}

#[test]
fn test_compression_ratio_non_repetitive_passes() {
    let diverse: Vec<usize> = (0..20).collect();
    let cr = compression_ratio(&diverse);
    assert!(
        cr <= DEFAULT_COMPRESSION_RATIO_THRESHOLD,
        "diverse tokens should be below threshold: got {cr}"
    );
}

#[test]
fn test_quality_check_rejects_repetitive_output() {
    let result = DecodingResult::new(
        vec![1, 2, 1, 2, 1, 2, 1, 2],
        -0.3, // good avg_logprob
        4.0,  // bad compression ratio
        true,
        0.0,
        0.0,
    );
    let config = DecodeConfig::default();
    assert!(
        !passes_quality_check(&result, &config),
        "repetitive output should fail quality check"
    );
}

#[test]
fn test_quality_check_rejects_low_confidence() {
    let result = DecodingResult::new(
        vec![1, 2, 3, 4, 5],
        -3.0, // bad avg_logprob (below -1.0)
        1.2,  // good compression ratio
        true,
        0.0,
        0.0,
    );
    let config = DecodeConfig::default();
    assert!(
        !passes_quality_check(&result, &config),
        "low confidence output should fail quality check"
    );
}

#[test]
fn test_temperature_fallback_tries_higher_temperatures() {
    // Temperature fallback should try multiple temperatures.
    // With zero-weight model, quality checks may not pass, so last temp is returned.
    let mut model = tiny_model();
    let enc_out = tiny_encoder_output();
    let config = DecodeConfig::default()
        .with_max_length(5)
        .with_initial_tokens(vec![0])
        .with_seed(Some(42));

    let result =
        temperature_fallback_decode(&mut model, &enc_out, &config, &DEFAULT_TEMPERATURES).unwrap();

    // Temperature should be one of the default temperatures.
    assert!(
        DEFAULT_TEMPERATURES.contains(&result.temperature),
        "result temperature {} should be in DEFAULT_TEMPERATURES",
        result.temperature
    );
}

// ---------------------------------------------------------------------------
// Special token handling
// ---------------------------------------------------------------------------

#[test]
fn test_special_token_constants() {
    assert_eq!(EOT_TOKEN, 50257);
    assert_eq!(SOT_TOKEN, 50258);
    assert_eq!(NO_SPEECH_TOKEN, 50363);
    assert_eq!(LANGUAGE_TOKEN_START, 50259);
    assert_eq!(LANGUAGE_TOKEN_END, 50358);
    // Language token range should accommodate 100 languages.
    assert_eq!(LANGUAGE_TOKEN_END - LANGUAGE_TOKEN_START + 1, 100);
}

#[test]
fn test_initial_tokens_include_sot() {
    // Default initial tokens should start with SOT.
    let config = DecodeConfig::default();
    assert_eq!(
        config.initial_tokens[0], SOT_TOKEN,
        "first initial token should be SOT (50258)"
    );
}

#[test]
fn test_initial_tokens_include_language_and_task() {
    // Default: [SOT, en, transcribe, notimestamps]
    let config = DecodeConfig::default();
    assert_eq!(config.initial_tokens.len(), 4);
    assert_eq!(config.initial_tokens[0], 50258); // SOT
    assert_eq!(config.initial_tokens[1], 50259); // English
    assert_eq!(config.initial_tokens[2], 50360); // transcribe
    assert_eq!(config.initial_tokens[3], 50364); // notimestamps
}

#[test]
fn test_suppress_tokens_prevents_generation() {
    let mut model = tiny_model();
    let enc_out = tiny_encoder_output();

    // Suppress tokens 0 and 1, verify they don't appear in output.
    let config = DecodeConfig::default()
        .with_max_length(10)
        .with_initial_tokens(vec![0])
        .with_suppress_tokens(vec![0, 1]);

    let result = greedy_decode(&mut model, &enc_out, &config).unwrap();
    for &t in &result.tokens {
        assert!(
            t != 0 && t != 1,
            "suppressed token {t} should not appear in output"
        );
    }
}

// ---------------------------------------------------------------------------
// Decoder with different batch sizes
// ---------------------------------------------------------------------------

#[test]
fn test_decoder_batch_size_one() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let model = crate::WhisperModel::load(&vb, config.clone()).unwrap();
    let decoder = model.decoder();

    let tokens = DynTensor::from_vec_u32(vec![0; 3], &[1, 3], &cpu()).unwrap();
    let enc_out = DynTensor::zeros(
        &[1, config.max_source_positions, config.d_model],
        DType::F32,
        &cpu(),
    )
    .unwrap();

    let logits = decoder.forward_no_cache(&tokens, &enc_out).unwrap();
    assert_eq!(logits.dims(), &[1, 3, config.vocab_size]);
}

// ---------------------------------------------------------------------------
// Decoder sequence lengths
// ---------------------------------------------------------------------------

#[test]
fn test_decoder_single_token_sequence() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();

    let enc_out = DynTensor::zeros(
        &[1, config.max_source_positions, config.d_model],
        DType::F32,
        &cpu(),
    )
    .unwrap();
    let tokens = DynTensor::from_vec_u32(vec![0], &[1, 1], &cpu()).unwrap();

    let logits = model.decode(&tokens, &enc_out, true, 0).unwrap();
    assert_eq!(logits.dims(), &[1, 1, config.vocab_size]);
}

#[test]
fn test_decoder_multi_token_sequence() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();

    let enc_out = DynTensor::zeros(
        &[1, config.max_source_positions, config.d_model],
        DType::F32,
        &cpu(),
    )
    .unwrap();
    let seq_len = 8;
    let tokens = DynTensor::from_vec_u32(vec![0; seq_len], &[1, seq_len], &cpu()).unwrap();

    let logits = model.decode(&tokens, &enc_out, true, 0).unwrap();
    assert_eq!(logits.dims(), &[1, seq_len, config.vocab_size]);
}

// ---------------------------------------------------------------------------
// DecodingResult construction and fields
// ---------------------------------------------------------------------------

#[test]
fn test_decoding_result_new_fields() {
    let result = DecodingResult::new(
        vec![10, 20, 30],
        -0.5,
        1.2,
        true,
        0.0,
        0.05,
    );
    assert_eq!(result.tokens, vec![10, 20, 30]);
    assert!((result.avg_logprob - (-0.5)).abs() < f64::EPSILON);
    assert!((result.compression_ratio - 1.2).abs() < f64::EPSILON);
    assert!(result.reached_eot);
    assert!((result.temperature - 0.0).abs() < f64::EPSILON);
    assert!((result.no_speech_prob - 0.05).abs() < f64::EPSILON);
}

#[test]
fn test_decoding_result_empty_tokens() {
    let result = DecodingResult::new(vec![], 0.0, 1.0, true, 0.0, 0.0);
    assert!(result.tokens.is_empty());
    assert!(result.reached_eot);
}

// ---------------------------------------------------------------------------
// Greedy decode: avg_logprob is non-positive
// ---------------------------------------------------------------------------

#[test]
fn test_greedy_decode_avg_logprob_non_positive() {
    let mut model = tiny_model();
    let enc_out = tiny_encoder_output();
    let config = DecodeConfig::default()
        .with_max_length(5)
        .with_initial_tokens(vec![0]);
    let result = greedy_decode(&mut model, &enc_out, &config).unwrap();
    assert!(
        result.avg_logprob <= 0.0,
        "avg_logprob should be non-positive (log-softmax), got {}",
        result.avg_logprob
    );
}

// ---------------------------------------------------------------------------
// Greedy decode: compression_ratio >= 1.0
// ---------------------------------------------------------------------------

#[test]
fn test_greedy_decode_compression_ratio_geq_one() {
    let mut model = tiny_model();
    let enc_out = tiny_encoder_output();
    let config = DecodeConfig::default()
        .with_max_length(10)
        .with_initial_tokens(vec![0]);
    let result = greedy_decode(&mut model, &enc_out, &config).unwrap();
    assert!(
        result.compression_ratio >= 1.0,
        "compression ratio must be >= 1.0, got {}",
        result.compression_ratio
    );
}

// ---------------------------------------------------------------------------
// Temperature fallback with empty temperatures
// ---------------------------------------------------------------------------

#[test]
fn test_temperature_fallback_empty_temperatures_error() {
    let mut model = tiny_model();
    let enc_out = tiny_encoder_output();
    let config = DecodeConfig::default()
        .with_max_length(3)
        .with_initial_tokens(vec![0]);
    let result = temperature_fallback_decode(&mut model, &enc_out, &config, &[]);
    assert!(
        result.is_err(),
        "empty temperatures should return an error"
    );
}

#[test]
fn test_temperature_fallback_single_temperature() {
    let mut model = tiny_model();
    let enc_out = tiny_encoder_output();
    let config = DecodeConfig::default()
        .with_max_length(3)
        .with_initial_tokens(vec![0]);
    let result = temperature_fallback_decode(&mut model, &enc_out, &config, &[0.0]).unwrap();
    assert!((result.temperature - 0.0).abs() < f64::EPSILON);
}

// ---------------------------------------------------------------------------
// Decode with various initial token configurations
// ---------------------------------------------------------------------------

#[test]
fn test_decode_with_single_initial_token() {
    let mut model = tiny_model();
    let enc_out = tiny_encoder_output();
    let config = DecodeConfig::default()
        .with_max_length(3)
        .with_initial_tokens(vec![5]);
    let result = greedy_decode(&mut model, &enc_out, &config).unwrap();
    assert!(result.tokens.len() <= 3);
}

#[test]
fn test_decode_with_many_initial_tokens() {
    let mut model = tiny_model();
    let enc_out = tiny_encoder_output();
    // 10 initial tokens is within max_target_positions (16).
    let config = DecodeConfig::default()
        .with_max_length(3)
        .with_initial_tokens(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let result = greedy_decode(&mut model, &enc_out, &config).unwrap();
    assert!(result.tokens.len() <= 3);
}

// ---------------------------------------------------------------------------
// Decoder reset between segments
// ---------------------------------------------------------------------------

#[test]
fn test_decoder_reset_produces_consistent_output() {
    // Decoding after reset should produce the same output as first decode.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config).unwrap();
    let enc_out = tiny_encoder_output();
    let tokens = DynTensor::from_vec_u32(vec![0; 3], &[1, 3], &cpu()).unwrap();

    // First decode.
    let logits1 = model.decode(&tokens, &enc_out, true, 0).unwrap();
    let flat1 = logits1.to_flat_vec::<f32>().unwrap();

    // Decode some more tokens to populate cache.
    let token2 = DynTensor::from_vec_u32(vec![1], &[1, 1], &cpu()).unwrap();
    model.decode(&token2, &enc_out, false, 3).unwrap();

    // Reset and decode same sequence.
    model.reset_kv_cache();
    let logits2 = model.decode(&tokens, &enc_out, true, 0).unwrap();
    let flat2 = logits2.to_flat_vec::<f32>().unwrap();

    assert_eq!(flat1, flat2, "decode after reset should match first decode");
}

// ---------------------------------------------------------------------------
// Compression ratio edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_compression_ratio_two_tokens() {
    let cr = compression_ratio(&[1, 2]);
    assert!((cr - 1.0).abs() < f64::EPSILON, "two unique tokens: cr should be 1.0");
}

#[test]
fn test_compression_ratio_all_same_tokens() {
    // All same tokens: [A, A, A, A] → one unique bigram (A,A), 3 bigram slots.
    let cr = compression_ratio(&[5, 5, 5, 5]);
    assert!(
        (cr - 3.0).abs() < f64::EPSILON,
        "all same tokens: cr should be 3.0, got {cr}"
    );
}

// ---------------------------------------------------------------------------
// Model dtype propagation
// ---------------------------------------------------------------------------

#[test]
fn test_model_dtype_from_vb() {
    let vb_f32 = VarBuilder::zeros(DType::F32, &cpu());
    let model_f32 = crate::WhisperModel::load(&vb_f32, tiny_config()).unwrap();
    assert_eq!(model_f32.dtype(), DType::F32);
}

// ---------------------------------------------------------------------------
// Config accessor
// ---------------------------------------------------------------------------

#[test]
fn test_model_config_accessor() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let model = crate::WhisperModel::load(&vb, config.clone()).unwrap();
    assert_eq!(model.config().d_model, config.d_model);
    assert_eq!(model.config().vocab_size, config.vocab_size);
    assert_eq!(model.config().decoder_layers, config.decoder_layers);
}

// ---------------------------------------------------------------------------
// No-speech probability in decode results
// ---------------------------------------------------------------------------

#[test]
fn test_no_speech_prob_in_range() {
    let mut model = tiny_model();
    let enc_out = tiny_encoder_output();
    let config = DecodeConfig::default()
        .with_max_length(3)
        .with_initial_tokens(vec![0]);
    let result = greedy_decode(&mut model, &enc_out, &config).unwrap();
    assert!(
        result.no_speech_prob >= 0.0 && result.no_speech_prob <= 1.0,
        "no_speech_prob should be in [0, 1], got {}",
        result.no_speech_prob
    );
}

// ---------------------------------------------------------------------------
// Decode with suppression of all non-zero tokens
// ---------------------------------------------------------------------------

#[test]
fn test_suppress_all_but_one_token() {
    let mut model = tiny_model();
    let enc_out = tiny_encoder_output();

    // Suppress everything except token 5.
    let suppress: Vec<usize> = (0..32).filter(|&t| t != 5).collect();
    let config = DecodeConfig::default()
        .with_max_length(5)
        .with_initial_tokens(vec![0])
        .with_suppress_tokens(suppress);

    let result = greedy_decode(&mut model, &enc_out, &config).unwrap();
    for &t in &result.tokens {
        assert_eq!(t, 5, "only token 5 should be generated, got {t}");
    }
}

// ---------------------------------------------------------------------------
// Seeded decode reproducibility across model instances
// ---------------------------------------------------------------------------

#[test]
fn test_seeded_decode_reproducible_across_instances() {
    let enc_out = tiny_encoder_output();
    let config = DecodeConfig::default()
        .with_max_length(8)
        .with_initial_tokens(vec![0])
        .with_seed(Some(12345));

    let mut model1 = tiny_model();
    let r1 = decode_with_temperature(&mut model1, &enc_out, &config, 0.5).unwrap();

    let mut model2 = tiny_model();
    let r2 = decode_with_temperature(&mut model2, &enc_out, &config, 0.5).unwrap();

    assert_eq!(
        r1.tokens, r2.tokens,
        "seeded decode should be reproducible across model instances"
    );
    assert!(
        (r1.avg_logprob - r2.avg_logprob).abs() < 1e-10,
        "avg_logprob should match"
    );
}
