// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Whisper pipeline tests (50+).
//!
//! Covers: model configuration scaling invariants, encoder/decoder architecture
//! constraints, weight key naming conventions, mel spectrogram pipeline,
//! audio preprocessing edge cases, tokenizer encode/decode paths, beam search
//! hypothesis scoring, quality metrics (WER/CER/MER/NED/SNR/PESQ), streaming
//! transcriber state machine, long-form config, compression ratio analysis,
//! DecodingResult semantics, error type conversion, and cross-config parameter
//! relationships.
//! Part of #4186.

use crate::audio::mel_filterbank;
use crate::audio_processing::{normalize_audio, pad_or_trim, preprocess_audio, resample, stereo_to_mono};
use crate::beam_search::{normalize_score, WhisperBeamSearchConfig, BeamHypothesis};
use crate::config::{
    WhisperConfig, CHUNK_LENGTH, HOP_LENGTH, NUM_MEL_BINS, N_FFT, N_FRAMES, N_SAMPLES, SAMPLE_RATE,
};
use crate::decode::{
    compression_ratio, passes_quality_check,
    DecodeConfig, DecodingResult, LongFormConfig, MAX_DECODE_LENGTH, DEFAULT_TEMPERATURES,
};
use crate::positional::{causal_mask, sinusoidal_embedding};
use crate::quality::{character_error_rate, match_error_rate, normalized_edit_distance, word_error_rate};
use crate::quality_audio::{audio_snr, pesq_approximation};
use crate::streaming::StreamingConfig;
use crate::test_utils::{tiny_config, tiny_model, tiny_encoder_output};
use crate::tokenizer::{
    WhisperTokenizer, DEFAULT_NO_SPEECH_THRESHOLD, EOT_TOKEN, LANGUAGE_TOKEN_END,
    LANGUAGE_TOKEN_START, NO_SPEECH_TOKEN, SOT_TOKEN,
};
use crate::WhisperError;
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, TensorError, VarBuilder};

// ============================================================================
// 1. Config scaling invariants across all standard sizes
// ============================================================================

#[test]
fn test_all_configs_ffn_dim_equals_4x_d_model() {
    let configs = [
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
        WhisperConfig::whisper_large_v2(),
        WhisperConfig::large_v3_turbo(),
    ];
    for c in &configs {
        assert_eq!(
            c.encoder_ffn_dim,
            c.d_model * 4,
            "encoder_ffn_dim should be 4x d_model for d_model={}",
            c.d_model
        );
        assert_eq!(
            c.decoder_ffn_dim,
            c.d_model * 4,
            "decoder_ffn_dim should be 4x d_model for d_model={}",
            c.d_model
        );
    }
}

#[test]
fn test_all_configs_head_dim_is_64() {
    // Whisper uses head_dim=64 across all sizes.
    let configs = [
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
        WhisperConfig::whisper_large_v2(),
        WhisperConfig::large_v3_turbo(),
    ];
    for c in &configs {
        assert_eq!(
            c.encoder_head_dim(),
            64,
            "encoder head_dim should be 64 for d_model={}",
            c.d_model
        );
        assert_eq!(
            c.decoder_head_dim(),
            64,
            "decoder head_dim should be 64 for d_model={}",
            c.d_model
        );
    }
}

#[test]
fn test_all_configs_validate_successfully() {
    let configs = [
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
        WhisperConfig::whisper_large_v2(),
        WhisperConfig::large_v3_turbo(),
    ];
    for c in &configs {
        c.validate().expect("all standard configs must validate");
    }
}

#[test]
fn test_config_d_model_scales_monotonically() {
    let dims = [
        WhisperConfig::whisper_tiny().d_model,
        WhisperConfig::whisper_base().d_model,
        WhisperConfig::whisper_small().d_model,
        WhisperConfig::whisper_medium().d_model,
        WhisperConfig::whisper_large_v2().d_model,
    ];
    for w in dims.windows(2) {
        assert!(
            w[0] < w[1],
            "d_model should increase: {} >= {}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn test_config_encoder_layers_scale_monotonically() {
    let layers = [
        WhisperConfig::whisper_tiny().encoder_layers,
        WhisperConfig::whisper_base().encoder_layers,
        WhisperConfig::whisper_small().encoder_layers,
        WhisperConfig::whisper_medium().encoder_layers,
        WhisperConfig::whisper_large_v2().encoder_layers,
    ];
    for w in layers.windows(2) {
        assert!(
            w[0] <= w[1],
            "encoder_layers should increase: {} > {}",
            w[0],
            w[1]
        );
    }
}

// ============================================================================
// 2. Config validation edge cases
// ============================================================================

#[test]
fn test_config_validation_rejects_zero_vocab_size() {
    let c = WhisperConfig::whisper_tiny().with_vocab_size(0);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_validation_rejects_zero_num_mel_bins() {
    let c = WhisperConfig::whisper_tiny().with_num_mel_bins(0);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_validation_rejects_zero_encoder_ffn_dim() {
    let c = WhisperConfig::whisper_tiny().with_encoder_ffn_dim(0);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_validation_rejects_zero_decoder_ffn_dim() {
    let c = WhisperConfig::whisper_tiny().with_decoder_ffn_dim(0);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_validation_rejects_zero_max_source_positions() {
    let c = WhisperConfig::whisper_tiny().with_max_source_positions(0);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_validation_rejects_zero_max_target_positions() {
    let c = WhisperConfig::whisper_tiny().with_max_target_positions(0);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_validation_rejects_indivisible_encoder_heads() {
    let c = WhisperConfig::whisper_tiny().with_encoder_attention_heads(5);
    // 384 is not divisible by 5
    assert!(c.validate().is_err());
}

#[test]
fn test_config_validation_rejects_indivisible_decoder_heads() {
    let c = WhisperConfig::whisper_tiny().with_decoder_attention_heads(5);
    assert!(c.validate().is_err());
}

// ============================================================================
// 3. Large-v3-turbo has distilled decoder (4 layers vs 32 encoder)
// ============================================================================

#[test]
fn test_large_v3_turbo_distilled_decoder() {
    let c = WhisperConfig::large_v3_turbo();
    assert_eq!(c.encoder_layers, 32, "turbo encoder has 32 layers");
    assert_eq!(c.decoder_layers, 4, "turbo decoder is distilled to 4 layers");
    assert!(c.encoder_layers > c.decoder_layers);
}

#[test]
fn test_large_v2_symmetric_layers() {
    let c = WhisperConfig::whisper_large_v2();
    assert_eq!(c.encoder_layers, c.decoder_layers);
    assert_eq!(c.encoder_layers, 32);
}

#[test]
fn test_small_models_use_80_mel_bins() {
    for c in &[
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
    ] {
        assert_eq!(c.num_mel_bins, 80, "models < large use 80 mel bins");
    }
}

#[test]
fn test_large_models_use_128_mel_bins() {
    for c in &[
        WhisperConfig::whisper_large_v2(),
        WhisperConfig::large_v3_turbo(),
    ] {
        assert_eq!(c.num_mel_bins, 128, "large models use 128 mel bins");
    }
}

// ============================================================================
// 4. Weight key naming conventions
// ============================================================================

#[test]
fn test_model_loads_with_encoder_decoder_prefix() {
    // WhisperModel::load expects model.encoder.* and model.decoder.* prefixes.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let model = crate::WhisperModel::load(&vb, config);
    assert!(model.is_ok(), "model should load from zeros VarBuilder");
}

#[test]
fn test_model_config_accessor() {
    let model = tiny_model();
    let config = model.config();
    assert_eq!(config.d_model, 16);
    assert_eq!(config.encoder_layers, 1);
}

#[test]
fn test_model_dtype_accessor() {
    let model = tiny_model();
    assert_eq!(model.dtype(), DType::F32);
}

// ============================================================================
// 5. Mel filterbank properties
// ============================================================================

#[test]
fn test_mel_filterbank_80_bins_shape() {
    let n_freqs = N_FFT / 2 + 1;
    let filters = mel_filterbank(80, N_FFT, SAMPLE_RATE);
    assert_eq!(filters.len(), 80 * n_freqs);
}

#[test]
fn test_mel_filterbank_128_bins_shape() {
    let n_freqs = N_FFT / 2 + 1;
    let filters = mel_filterbank(128, N_FFT, SAMPLE_RATE);
    assert_eq!(filters.len(), 128 * n_freqs);
}

#[test]
fn test_mel_filterbank_all_non_negative() {
    let filters = mel_filterbank(128, N_FFT, SAMPLE_RATE);
    for &v in &filters {
        assert!(v >= 0.0, "filterbank values must be non-negative, got {v}");
        assert!(v.is_finite(), "filterbank values must be finite, got {v}");
    }
}

#[test]
fn test_mel_filterbank_each_bin_has_nonzero_energy() {
    let n_freqs = N_FFT / 2 + 1;
    let n_mels = 80;
    let filters = mel_filterbank(n_mels, N_FFT, SAMPLE_RATE);
    for i in 0..n_mels {
        let row = &filters[i * n_freqs..(i + 1) * n_freqs];
        let sum: f32 = row.iter().sum();
        assert!(
            sum > 0.0,
            "mel bin {i} has zero total energy; filterbank is degenerate"
        );
    }
}

// ============================================================================
// 6. Audio preprocessing edge cases
// ============================================================================

#[test]
fn test_resample_48k_to_16k() {
    let source = vec![0.5f32; 48000]; // 1 second at 48 kHz
    let resampled = resample(&source, 48000, 16000).unwrap();
    // Output should be approximately 16000 samples (+/- 1 from rounding).
    assert!((resampled.len() as i64 - 16000).abs() <= 1);
}

#[test]
fn test_resample_22050_to_16000() {
    let source = vec![0.5f32; 22050]; // 1 second at 22.05 kHz
    let resampled = resample(&source, 22050, 16000).unwrap();
    assert!((resampled.len() as i64 - 16000).abs() <= 1);
}

#[test]
fn test_normalize_already_normalized() {
    let audio: Vec<f32> = vec![1.0, -1.0, 0.5, -0.5];
    let normalized = normalize_audio(&audio);
    // Peak is already 1.0 so output should be identical.
    for (a, b) in audio.iter().zip(normalized.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn test_normalize_amplifies_quiet_signal() {
    let audio: Vec<f32> = vec![0.1, -0.1, 0.05, -0.05];
    let normalized = normalize_audio(&audio);
    let peak = normalized.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!((peak - 1.0).abs() < 1e-6, "peak after normalization should be 1.0, got {peak}");
}

#[test]
fn test_pad_or_trim_very_short_audio() {
    let audio = vec![0.5f32; 100];
    let padded = pad_or_trim(&audio);
    assert_eq!(padded.len(), N_SAMPLES);
    // First 100 samples preserved.
    for &v in padded.iter().take(100) {
        assert!((v - 0.5).abs() < 1e-6);
    }
    // Rest is zero-padded.
    for &v in padded.iter().take(N_SAMPLES).skip(100) {
        assert!((v - 0.0).abs() < 1e-6);
    }
}

#[test]
fn test_preprocess_audio_stereo_48k() {
    // 1 second of stereo audio at 48 kHz (interleaved).
    let stereo = vec![0.3f32; 48000 * 2];
    let processed = preprocess_audio(&stereo, 2, 48000).unwrap();
    assert_eq!(processed.len(), N_SAMPLES, "output should be 30s padded");
}

#[test]
fn test_preprocess_audio_rejects_zero_channels() {
    let audio = vec![0.0f32; 1000];
    assert!(preprocess_audio(&audio, 0, 16000).is_err());
}

#[test]
fn test_preprocess_audio_rejects_three_channels() {
    let audio = vec![0.0f32; 3000];
    assert!(preprocess_audio(&audio, 3, 16000).is_err());
}

// ============================================================================
// 7. Tokenizer properties
// ============================================================================

#[test]
fn test_tokenizer_empty_vocab_has_zero_size() {
    let t = WhisperTokenizer::from_vocab_str("{}").unwrap();
    assert_eq!(t.vocab_size(), 0);
}

#[test]
fn test_tokenizer_is_special_boundary() {
    let t = WhisperTokenizer::from_vocab_str("{}").unwrap();
    // EOT_TOKEN = 50257 is the first special token.
    assert!(!t.is_special(50256));
    assert!(t.is_special(50257));
    assert!(t.is_special(50258));
    assert!(t.is_special(50365));
}

#[test]
fn test_tokenizer_is_timestamp_boundary() {
    let t = WhisperTokenizer::from_vocab_str("{}").unwrap();
    // TIMESTAMP_BEGIN = 50365.
    assert!(!t.is_timestamp(50364));
    assert!(t.is_timestamp(50365));
    assert!(t.is_timestamp(50366));
}

#[test]
fn test_tokenizer_timestamp_value_resolution() {
    let t = WhisperTokenizer::from_vocab_str("{}").unwrap();
    // First timestamp: 0.00s, second: 0.02s, etc.
    let v0 = t.timestamp_value(50365).unwrap();
    assert!((v0 - 0.0).abs() < 1e-10);
    let v1 = t.timestamp_value(50366).unwrap();
    assert!((v1 - 0.02).abs() < 1e-10);
    let v100 = t.timestamp_value(50465).unwrap();
    assert!((v100 - 2.0).abs() < 1e-10);
}

#[test]
fn test_tokenizer_timestamp_30s_is_token_51865() {
    // 30.00s = 1500 * 0.02 => token ID = 50365 + 1500 = 51865.
    let t = WhisperTokenizer::from_vocab_str("{}").unwrap();
    let v = t.timestamp_value(51865).unwrap();
    assert!((v - 30.0).abs() < 1e-10, "30s timestamp should be at ID 51865");
}

#[test]
fn test_tokenizer_timestamp_value_none_for_non_timestamp() {
    let t = WhisperTokenizer::from_vocab_str("{}").unwrap();
    assert!(t.timestamp_value(50364).is_none());
    assert!(t.timestamp_value(50257).is_none());
    assert!(t.timestamp_value(100).is_none());
}

#[test]
fn test_tokenizer_decode_with_timestamps_empty() {
    let t = WhisperTokenizer::from_vocab_str("{}").unwrap();
    let segments = t.decode_with_timestamps(&[]).unwrap();
    assert!(segments.is_empty());
}

#[test]
fn test_tokenizer_can_encode_without_merges_is_false() {
    let t = WhisperTokenizer::from_vocab_str("{}").unwrap();
    assert!(!t.can_encode(), "tokenizer without merges cannot encode");
}

// ============================================================================
// 8. Special token constants
// ============================================================================

#[test]
fn test_eot_token_value() {
    assert_eq!(EOT_TOKEN, 50257);
}

#[test]
fn test_sot_token_value() {
    assert_eq!(SOT_TOKEN, 50258);
}

#[test]
fn test_no_speech_token_value() {
    assert_eq!(NO_SPEECH_TOKEN, 50363);
}

#[test]
fn test_language_token_range() {
    assert_eq!(LANGUAGE_TOKEN_START, 50259);
    assert_eq!(LANGUAGE_TOKEN_END, 50358);
    let num_languages = LANGUAGE_TOKEN_END - LANGUAGE_TOKEN_START + 1;
    assert_eq!(num_languages, 100, "Whisper supports 100 languages");
}

#[test]
fn test_no_speech_threshold_default() {
    assert!((DEFAULT_NO_SPEECH_THRESHOLD - 0.6).abs() < 1e-10);
}

// ============================================================================
// 9. Beam search hypothesis scoring
// ============================================================================

#[test]
fn test_normalize_score_zero_penalty() {
    let score = normalize_score(-5.0, 10, 0.0);
    assert!((score - (-5.0)).abs() < 1e-6, "zero penalty should return raw score");
}

#[test]
fn test_normalize_score_unit_penalty() {
    let score = normalize_score(-10.0, 5, 1.0);
    // -10.0 / 5^1.0 = -2.0
    assert!((score - (-2.0)).abs() < 1e-6);
}

#[test]
fn test_normalize_score_fractional_penalty() {
    let score = normalize_score(-10.0, 4, 0.5);
    // -10.0 / 4^0.5 = -10.0 / 2.0 = -5.0
    assert!((score - (-5.0)).abs() < 1e-6);
}

#[test]
fn test_normalize_score_zero_length() {
    let score = normalize_score(-5.0, 0, 1.0);
    assert!((score - (-5.0)).abs() < 1e-6, "zero length should return raw score");
}

#[test]
fn test_beam_hypothesis_construction() {
    let h = BeamHypothesis::new(vec![1, 2, 3], -3.0, -1.0);
    assert_eq!(h.tokens, vec![1, 2, 3]);
    assert!((h.score - (-3.0)).abs() < 1e-6);
    assert!((h.normalized_score - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_beam_search_config_default() {
    let c = WhisperBeamSearchConfig::default();
    assert_eq!(c.beam_width, 5);
    assert_eq!(c.max_tokens, 448);
    assert!((c.length_penalty - 1.0).abs() < 1e-6);
    assert_eq!(c.no_repeat_ngram_size, 0);
    assert!((c.temperature - 0.0).abs() < 1e-6);
    assert!(c.suppress_blank);
    assert_eq!(c.sot_token, SOT_TOKEN);
    assert_eq!(c.eot_token, EOT_TOKEN);
}

#[test]
fn test_beam_search_config_validation_ok() {
    let c = WhisperBeamSearchConfig::default();
    c.validate().expect("default config should validate");
}

#[test]
fn test_beam_search_config_rejects_zero_max_tokens() {
    let c = WhisperBeamSearchConfig {
        max_tokens: 0,
        ..Default::default()
    };
    assert!(c.validate().is_err());
}

#[test]
fn test_beam_search_config_rejects_nan_length_penalty() {
    let c = WhisperBeamSearchConfig {
        length_penalty: f32::NAN,
        ..Default::default()
    };
    assert!(c.validate().is_err());
}

// ============================================================================
// 10. Compression ratio analysis
// ============================================================================

#[test]
fn test_compression_ratio_empty_tokens() {
    assert!((compression_ratio(&[]) - 1.0).abs() < 1e-10);
}

#[test]
fn test_compression_ratio_single_token() {
    assert!((compression_ratio(&[42]) - 1.0).abs() < 1e-10);
}

#[test]
fn test_compression_ratio_all_unique_bigrams() {
    // [1, 2, 3, 4] has 3 bigrams: (1,2), (2,3), (3,4) — all unique.
    // ratio = 3 / 3 = 1.0
    let cr = compression_ratio(&[1, 2, 3, 4]);
    assert!((cr - 1.0).abs() < 1e-10);
}

#[test]
fn test_compression_ratio_fully_repeated() {
    // [5, 5, 5, 5] has 3 bigram slots but only 1 unique bigram (5,5).
    // ratio = 3 / 1 = 3.0
    let cr = compression_ratio(&[5, 5, 5, 5]);
    assert!((cr - 3.0).abs() < 1e-10);
}

#[test]
fn test_compression_ratio_two_tokens() {
    // [1, 2] has 1 bigram slot, 1 unique bigram.
    let cr = compression_ratio(&[1, 2]);
    assert!((cr - 1.0).abs() < 1e-10);
}

// ============================================================================
// 11. Quality check logic
// ============================================================================

#[test]
fn test_passes_quality_check_normal_result() {
    let result = DecodingResult::new(vec![1, 2, 3], -0.5, 1.5, true, 0.0, 0.1);
    let config = DecodeConfig::default();
    assert!(passes_quality_check(&result, &config));
}

#[test]
fn test_quality_check_fails_high_compression() {
    let result = DecodingResult::new(vec![1, 1, 1], -0.5, 3.0, true, 0.0, 0.1);
    let config = DecodeConfig::default();
    // 3.0 > DEFAULT_COMPRESSION_RATIO_THRESHOLD (2.4)
    assert!(!passes_quality_check(&result, &config));
}

#[test]
fn test_quality_check_fails_low_logprob() {
    let result = DecodingResult::new(vec![1, 2, 3], -2.0, 1.0, true, 0.0, 0.1);
    let config = DecodeConfig::default();
    // -2.0 < DEFAULT_AVG_LOGPROB_THRESHOLD (-1.0)
    assert!(!passes_quality_check(&result, &config));
}

#[test]
fn test_quality_check_with_custom_thresholds() {
    let result = DecodingResult::new(vec![1, 2, 3], -3.0, 5.0, true, 0.0, 0.1);
    let config = DecodeConfig::default()
        .with_compression_ratio_threshold(10.0)
        .with_avg_logprob_threshold(-5.0);
    assert!(passes_quality_check(&result, &config));
}

// ============================================================================
// 12. DecodingResult construction and semantics
// ============================================================================

#[test]
fn test_decoding_result_fields() {
    let r = DecodingResult::new(vec![10, 20], -0.8, 1.2, true, 0.2, 0.05);
    assert_eq!(r.tokens, vec![10, 20]);
    assert!((r.avg_logprob - (-0.8)).abs() < 1e-10);
    assert!((r.compression_ratio - 1.2).abs() < 1e-10);
    assert!(r.reached_eot);
    assert!((r.temperature - 0.2).abs() < 1e-10);
    assert!((r.no_speech_prob - 0.05).abs() < 1e-10);
}

#[test]
fn test_decoding_result_no_speech_high_probability() {
    let r = DecodingResult::new(vec![], 0.0, 1.0, true, 0.0, 0.95);
    assert!(r.no_speech_prob > DEFAULT_NO_SPEECH_THRESHOLD);
}

#[test]
fn test_decoding_result_empty_tokens_with_eot() {
    let r = DecodingResult::new(vec![], 0.0, 1.0, true, 0.0, 0.0);
    assert!(r.reached_eot);
    assert!(r.tokens.is_empty());
}

// ============================================================================
// 13. Quality metrics: WER, CER, NED, MER
// ============================================================================

#[test]
fn test_wer_perfect_transcription() {
    assert!((word_error_rate("the cat sat on the mat", "the cat sat on the mat")).abs() < 1e-10);
}

#[test]
fn test_wer_completely_wrong() {
    let wer = word_error_rate("foo bar baz", "one two three");
    assert!((wer - 1.0).abs() < 1e-10);
}

#[test]
fn test_cer_identical_strings() {
    assert!((character_error_rate("hello world", "hello world")).abs() < 1e-10);
}

#[test]
fn test_cer_one_char_difference() {
    let cer = character_error_rate("hxllo", "hello");
    assert!((cer - 0.2).abs() < 1e-6);
}

#[test]
fn test_ned_identical_strings() {
    assert!((normalized_edit_distance("abc", "abc")).abs() < 1e-10);
}

#[test]
fn test_ned_completely_different() {
    let ned = normalized_edit_distance("xyz", "abc");
    assert!((ned - 1.0).abs() < 1e-6);
}

#[test]
fn test_mer_bounded_zero_to_one() {
    // MER is always in [0, 1] unlike WER.
    let mer = match_error_rate("a b c d e f g h", "x");
    assert!((0.0..=1.0 + 1e-6).contains(&mer));
}

#[test]
fn test_mer_perfect() {
    assert!((match_error_rate("the cat", "the cat")).abs() < 1e-10);
}

// ============================================================================
// 14. Audio quality: SNR and PESQ approximation
// ============================================================================

#[test]
fn test_snr_identical_signals_is_infinity() {
    let s = vec![1.0f32, -1.0, 0.5];
    assert_eq!(audio_snr(&s, &s), f32::INFINITY);
}

#[test]
fn test_snr_zero_signal_is_neg_infinity() {
    assert_eq!(
        audio_snr(&[0.0, 0.0, 0.0], &[0.1, -0.1, 0.1]),
        f32::NEG_INFINITY
    );
}

#[test]
fn test_pesq_identical_high_score() {
    let signal: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.01).sin()).collect();
    let score = pesq_approximation(&signal, &signal, 16000);
    assert!(score >= 4.0, "identical signals should score high, got {score}");
}

#[test]
fn test_pesq_empty_returns_one() {
    assert!((pesq_approximation(&[], &[], 16000) - 1.0).abs() < 1e-6);
}

#[test]
fn test_pesq_score_in_range() {
    let signal: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.02).sin()).collect();
    let degraded: Vec<f32> = signal.iter().map(|x| x * 0.5 + 0.1).collect();
    let score = pesq_approximation(&signal, &degraded, 16000);
    assert!(
        (1.0..=4.5).contains(&score),
        "PESQ score out of range: {score}"
    );
}

// ============================================================================
// 15. Positional encoding
// ============================================================================

#[test]
fn test_sinusoidal_embedding_correct_shape() {
    let emb = sinusoidal_embedding(1500, 1280, DType::F32, &cpu()).unwrap();
    assert_eq!(emb.dims(), &[1500, 1280]);
}

#[test]
fn test_sinusoidal_values_bounded() {
    let emb = sinusoidal_embedding(10, 16, DType::F32, &cpu()).unwrap();
    let flat = emb.to_flat_vec::<f32>().unwrap();
    for &v in &flat {
        assert!(v.is_finite());
        assert!((-1.0..=1.0).contains(&v), "sin/cos values must be in [-1,1], got {v}");
    }
}

#[test]
fn test_causal_mask_diagonal_is_zero() {
    let mask = causal_mask(8, DType::F32, &cpu()).unwrap();
    let flat = mask.to_flat_vec::<f32>().unwrap();
    for i in 0..8 {
        assert!((flat[i * 8 + i] - 0.0).abs() < 1e-10, "diagonal should be 0.0");
    }
}

#[test]
fn test_causal_mask_upper_triangle_neg_inf() {
    let n = 5;
    let mask = causal_mask(n, DType::F32, &cpu()).unwrap();
    let flat = mask.to_flat_vec::<f32>().unwrap();
    for i in 0..n {
        for j in (i + 1)..n {
            assert_eq!(
                flat[i * n + j],
                f32::NEG_INFINITY,
                "mask[{i}][{j}] should be -inf"
            );
        }
    }
}

// ============================================================================
// 16. Audio constants consistency
// ============================================================================

#[test]
fn test_audio_constants_consistency() {
    assert_eq!(SAMPLE_RATE, 16000);
    assert_eq!(N_FFT, 400);
    assert_eq!(HOP_LENGTH, 160);
    assert_eq!(CHUNK_LENGTH, 30);
    assert_eq!(N_SAMPLES, SAMPLE_RATE * CHUNK_LENGTH);
    assert_eq!(N_FRAMES, N_SAMPLES / HOP_LENGTH);
    assert_eq!(NUM_MEL_BINS, 128);
}

#[test]
fn test_n_samples_is_480000() {
    assert_eq!(N_SAMPLES, 480_000);
}

#[test]
fn test_n_frames_is_3000() {
    assert_eq!(N_FRAMES, 3000);
}

// ============================================================================
// 17. Decode config builder and validation
// ============================================================================

#[test]
fn test_decode_config_default_values() {
    let dc = DecodeConfig::default();
    assert_eq!(dc.max_length, MAX_DECODE_LENGTH);
    assert!((dc.compression_ratio_threshold - 2.4).abs() < 1e-10);
    assert!((dc.avg_logprob_threshold - (-1.0)).abs() < 1e-10);
    assert!(dc.suppress_tokens.is_empty());
    assert_eq!(dc.initial_tokens, vec![50258, 50259, 50360, 50364]);
    assert!(dc.seed.is_none());
}

#[test]
fn test_decode_config_builder_chain() {
    let dc = DecodeConfig::default()
        .with_max_length(100)
        .with_seed(Some(42))
        .with_suppress_tokens(vec![1, 2, 3])
        .with_initial_tokens(vec![50258, 50259]);
    assert_eq!(dc.max_length, 100);
    assert_eq!(dc.seed, Some(42));
    assert_eq!(dc.suppress_tokens, vec![1, 2, 3]);
    assert_eq!(dc.initial_tokens, vec![50258, 50259]);
}

#[test]
fn test_decode_config_rejects_exceeds_max_length() {
    let dc = DecodeConfig::default().with_max_length(MAX_DECODE_LENGTH + 1);
    assert!(dc.validate().is_err());
}

#[test]
fn test_decode_config_rejects_inf_avg_logprob_threshold() {
    let dc = DecodeConfig::default().with_avg_logprob_threshold(f64::INFINITY);
    assert!(dc.validate().is_err());
}

// ============================================================================
// 18. Streaming config
// ============================================================================

#[test]
fn test_streaming_config_default_values() {
    let sc = StreamingConfig::default();
    assert!((sc.no_speech_threshold - 0.6).abs() < 1e-10);
    assert_eq!(sc.temperatures.len(), 6);
    assert!((sc.temperatures[0] - 0.0).abs() < 1e-10);
}

#[test]
fn test_streaming_config_builder_chain() {
    let sc = StreamingConfig::default()
        .with_no_speech_threshold(0.9)
        .with_temperatures(vec![0.0, 0.3, 0.6]);
    assert!((sc.no_speech_threshold - 0.9).abs() < 1e-10);
    assert_eq!(sc.temperatures.len(), 3);
}

#[test]
fn test_streaming_config_with_decode_config() {
    let dc = DecodeConfig::default().with_max_length(50);
    let sc = StreamingConfig::default().with_decode_config(dc);
    assert_eq!(sc.decode_config.max_length, 50);
}

// ============================================================================
// 19. Long-form config
// ============================================================================

#[test]
fn test_long_form_config_default_values() {
    let lfc = LongFormConfig::default();
    assert!((lfc.no_speech_threshold - 0.6).abs() < 1e-10);
    assert_eq!(lfc.temperatures.len(), 6);
}

// ============================================================================
// 20. Error type conversions
// ============================================================================

#[test]
fn test_whisper_error_to_tensor_error_conversion() {
    let we = WhisperError::ZeroConfigField { field: "d_model" };
    let te: TensorError = we.into();
    let msg = te.to_string();
    assert!(msg.contains("d_model"), "converted error should preserve field name");
}

#[test]
fn test_whisper_error_display_zero_field() {
    let e = WhisperError::ZeroConfigField { field: "vocab_size" };
    assert!(e.to_string().contains("vocab_size"));
    assert!(e.to_string().contains("must be > 0"));
}

#[test]
fn test_whisper_error_display_not_divisible() {
    let e = WhisperError::ConfigNotDivisible {
        a_name: "d_model",
        a_val: 384,
        b_name: "encoder_attention_heads",
        b_val: 5,
    };
    let s = e.to_string();
    assert!(s.contains("384"));
    assert!(s.contains("5"));
}

#[test]
fn test_whisper_error_display_byte_alignment() {
    let e = WhisperError::ByteAlignment {
        tensor_name: "layer.0.weight".into(),
        byte_len: 7,
        alignment: 4,
    };
    assert!(e.to_string().contains("layer.0.weight"));
}

#[test]
fn test_whisper_error_display_token_overflow() {
    let e = WhisperError::TokenIdOverflow { token_id: usize::MAX };
    assert!(e.to_string().contains("u32"));
}

#[test]
fn test_whisper_error_display_batch_mismatch() {
    let e = WhisperError::BatchMismatch {
        encoder_batch: 2,
        decoder_batch: 4,
    };
    let s = e.to_string();
    assert!(s.contains("2"));
    assert!(s.contains("4"));
}

// ============================================================================
// 21. Model encoder/decoder interaction with tiny model
// ============================================================================

#[test]
fn test_tiny_model_encode_shape() {
    let mut model = tiny_model();
    let config = tiny_config();
    let mel = DynTensor::zeros(
        &[1, config.num_mel_bins, config.max_source_positions * 2],
        DType::F32,
        &cpu(),
    )
    .unwrap();
    let enc = model.encode(&mel).unwrap();
    // Encoder output: [batch=1, seq_len, d_model=16]
    assert_eq!(enc.dims()[0], 1);
    assert_eq!(enc.dims()[2], config.d_model);
}

#[test]
fn test_tiny_model_reset_kv_cache_no_panic() {
    let mut model = tiny_model();
    model.reset_kv_cache(); // Should not panic.
    model.reset_kv_cache(); // Double reset is fine.
}

#[test]
fn test_tiny_encoder_output_shape() {
    let enc = tiny_encoder_output();
    let config = tiny_config();
    assert_eq!(enc.dims(), &[1, 8, config.d_model]);
}

// ============================================================================
// 22. Default temperatures sequence
// ============================================================================

#[test]
fn test_default_temperatures_monotonically_increasing() {
    for w in DEFAULT_TEMPERATURES.windows(2) {
        assert!(w[0] < w[1], "temperatures should increase: {} >= {}", w[0], w[1]);
    }
}

#[test]
fn test_default_temperatures_start_at_zero() {
    assert!((DEFAULT_TEMPERATURES[0] - 0.0).abs() < 1e-10);
}

#[test]
fn test_default_temperatures_end_at_one() {
    assert!((DEFAULT_TEMPERATURES[DEFAULT_TEMPERATURES.len() - 1] - 1.0).abs() < 1e-10);
}

#[test]
fn test_default_temperatures_all_non_negative() {
    for &t in &DEFAULT_TEMPERATURES {
        assert!(t >= 0.0, "temperature must be non-negative, got {t}");
    }
}

// ============================================================================
// 23. Stereo-to-mono edge cases
// ============================================================================

#[test]
fn test_stereo_to_mono_empty_input() {
    let result = stereo_to_mono(&[]).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_stereo_to_mono_single_pair() {
    let result = stereo_to_mono(&[0.8, 0.2]).unwrap();
    assert_eq!(result.len(), 1);
    assert!((result[0] - 0.5).abs() < 1e-6);
}

#[test]
fn test_stereo_to_mono_preserves_length_halving() {
    let stereo = vec![0.0f32; 1000];
    let mono = stereo_to_mono(&stereo).unwrap();
    assert_eq!(mono.len(), 500);
}
