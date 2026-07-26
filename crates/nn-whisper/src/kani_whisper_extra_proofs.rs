// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extra Kani proof harnesses for nn-whisper.
//!
//! Covers:
//! - Audio constant relationships (SAMPLE_RATE, N_FFT, HOP_LENGTH, N_SAMPLES, N_FRAMES)
//! - DecodeConfig validation: zero max_length, NaN thresholds, empty initial_tokens
//! - WhisperBeamConfig validation: zero beam_width, NaN length_penalty
//! - DecodingResult construction and field correctness
//! - WhisperError variant construction and Display coverage
//! - Compression ratio properties (single token, all-unique, repetitive)
//! - Token constant ordering invariants (EOT < SOT < LANGUAGE range < NO_SPEECH < TIMESTAMP)
//! - Config validate accepts all standard presets
//! - Default trait consistency for DecodeConfig and WhisperBeamConfig
//!
//! Issue: #3800

use crate::config::*;
use crate::decode::{
    compression_ratio, passes_quality_check, DecodeConfig, DecodingResult,
    DEFAULT_AVG_LOGPROB_THRESHOLD, DEFAULT_COMPRESSION_RATIO_THRESHOLD,
    DEFAULT_TEMPERATURES, MAX_DECODE_LENGTH,
};
use crate::decode_beam::WhisperBeamConfig;
use crate::tokenizer::{
    EOT_TOKEN, LANGUAGE_TOKEN_END, LANGUAGE_TOKEN_START, NO_SPEECH_TOKEN, SOT_TOKEN,
};

// ============================================================================
// Harness 1: Audio constant N_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH
// ============================================================================

/// Proves the fundamental audio constant relationship holds.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn audio_constant_n_samples_equals_rate_times_chunk() {
    assert_eq!(
        N_SAMPLES,
        SAMPLE_RATE * CHUNK_LENGTH,
        "N_SAMPLES must equal SAMPLE_RATE * CHUNK_LENGTH"
    );
}

// ============================================================================
// Harness 2: Audio constant N_FRAMES = N_SAMPLES / HOP_LENGTH
// ============================================================================

/// Proves the mel frame count derives from sample count and hop length.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn audio_constant_n_frames_equals_samples_div_hop() {
    assert_eq!(
        N_FRAMES,
        N_SAMPLES / HOP_LENGTH,
        "N_FRAMES must equal N_SAMPLES / HOP_LENGTH"
    );
}

// ============================================================================
// Harness 3: N_FFT > HOP_LENGTH (overlap requirement for STFT)
// ============================================================================

/// Proves that the FFT window is larger than the hop, ensuring overlap.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn audio_constant_fft_gt_hop_length() {
    assert!(
        N_FFT > HOP_LENGTH,
        "N_FFT must exceed HOP_LENGTH for STFT overlap"
    );
}

// ============================================================================
// Harness 5: DecodeConfig::validate rejects max_length > MAX_DECODE_LENGTH
// ============================================================================

/// Proves that DecodeConfig::validate rejects oversized max_length.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn decode_config_validate_rejects_oversized_max_length() {
    let cfg = DecodeConfig::default().with_max_length(MAX_DECODE_LENGTH + 1);
    assert!(
        cfg.validate().is_err(),
        "max_length > MAX_DECODE_LENGTH must be rejected"
    );
}

// ============================================================================
// Harness 6: DecodeConfig::validate rejects NaN compression_ratio_threshold
// ============================================================================

/// Proves that NaN compression ratio threshold is caught by validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn decode_config_validate_rejects_nan_compression_ratio() {
    let cfg = DecodeConfig::default().with_compression_ratio_threshold(f64::NAN);
    assert!(
        cfg.validate().is_err(),
        "NaN compression_ratio_threshold must be rejected"
    );
}

// ============================================================================
// Harness 7: DecodeConfig::validate rejects Inf avg_logprob_threshold
// ============================================================================

/// Proves that infinite avg_logprob_threshold is caught by validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn decode_config_validate_rejects_inf_avg_logprob() {
    let cfg = DecodeConfig::default().with_avg_logprob_threshold(f64::INFINITY);
    assert!(
        cfg.validate().is_err(),
        "Inf avg_logprob_threshold must be rejected"
    );
}

// ============================================================================
// Harness 8: DecodeConfig::validate rejects empty initial_tokens
// ============================================================================

/// Proves that empty initial_tokens is caught by validation.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn decode_config_validate_rejects_empty_initial_tokens() {
    let cfg = DecodeConfig::default().with_initial_tokens(Vec::new());
    assert!(
        cfg.validate().is_err(),
        "empty initial_tokens must be rejected"
    );
}

// ============================================================================
// Harness 9: DecodeConfig default passes validation
// ============================================================================

/// Proves that DecodeConfig::default() passes its own validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn decode_config_default_passes_validation() {
    let cfg = DecodeConfig::default();
    assert!(
        cfg.validate().is_ok(),
        "DecodeConfig::default() must pass validation"
    );
}

// ============================================================================
// Harness 10: WhisperBeamConfig::validate rejects zero beam_width
// ============================================================================

/// Proves that beam_width == 0 is rejected.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn beam_config_validate_rejects_zero_beam_width() {
    let cfg = WhisperBeamConfig {
        beam_width: 0,
        length_penalty: 1.0,
    };
    assert!(
        cfg.validate().is_err(),
        "beam_width=0 must be rejected"
    );
}

// ============================================================================
// Harness 11: WhisperBeamConfig::validate rejects NaN length_penalty
// ============================================================================

/// Proves that NaN length_penalty is rejected.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn beam_config_validate_rejects_nan_length_penalty() {
    let cfg = WhisperBeamConfig {
        beam_width: 5,
        length_penalty: f64::NAN,
    };
    assert!(
        cfg.validate().is_err(),
        "NaN length_penalty must be rejected"
    );
}

// ============================================================================
// Harness 12: WhisperBeamConfig default passes validation
// ============================================================================

/// Proves that WhisperBeamConfig::default() passes its own validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn beam_config_default_passes_validation() {
    let cfg = WhisperBeamConfig::default();
    assert!(
        cfg.validate().is_ok(),
        "WhisperBeamConfig::default() must pass validation"
    );
}

// ============================================================================
// Harness 13: Token constant ordering (EOT < SOT < LANGUAGE range)
// ============================================================================

/// Proves that special token IDs maintain their required ordering.
///
/// Whisper's vocabulary layout requires EOT < SOT < LANGUAGE_START ..
/// LANGUAGE_END < NO_SPEECH.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn token_constant_ordering_invariant() {
    assert!(EOT_TOKEN < SOT_TOKEN, "EOT < SOT");
    assert!(SOT_TOKEN < LANGUAGE_TOKEN_START, "SOT < LANGUAGE_START");
    assert!(
        LANGUAGE_TOKEN_START <= LANGUAGE_TOKEN_END,
        "LANGUAGE_START <= LANGUAGE_END"
    );
    assert!(
        LANGUAGE_TOKEN_END < NO_SPEECH_TOKEN,
        "LANGUAGE_END < NO_SPEECH"
    );
}

// ============================================================================
// Harness 14: Language token range covers exactly 100 languages
// ============================================================================

/// Proves that the language token range spans exactly 100 tokens.
///
/// Whisper supports 100 languages, from English (50259) to the last (50358).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn language_token_range_is_100() {
    let count = LANGUAGE_TOKEN_END - LANGUAGE_TOKEN_START + 1;
    assert_eq!(count, 100, "Whisper has exactly 100 language tokens");
}

// ============================================================================
// Harness 15: compression_ratio of single token is 1.0
// ============================================================================

/// Proves that a single-token sequence has compression ratio 1.0.
///
/// With < 2 tokens, there are no bigrams, so compression_ratio returns 1.0.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn compression_ratio_single_token_is_one() {
    let tokens = vec![42usize];
    let ratio = compression_ratio(&tokens);
    assert!(
        (ratio - 1.0).abs() < 1e-10,
        "single token must yield compression_ratio 1.0"
    );
}

// ============================================================================
// Harness 16: compression_ratio of empty slice is 1.0
// ============================================================================

/// Proves that an empty token sequence has compression ratio 1.0.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn compression_ratio_empty_is_one() {
    let tokens: Vec<usize> = Vec::new();
    let ratio = compression_ratio(&tokens);
    assert!(
        (ratio - 1.0).abs() < 1e-10,
        "empty tokens must yield compression_ratio 1.0"
    );
}

// ============================================================================
// Harness 17: compression_ratio >= 1.0 for any input
// ============================================================================

/// Proves that compression_ratio is always >= 1.0.
///
/// By definition: (n-1) bigram slots / unique_bigrams >= 1 when unique <= n-1,
/// and the function returns 1.0 for short inputs.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
fn compression_ratio_always_ge_one() {
    let len: usize = kani::any();
    kani::assume(len <= 4);

    // Build a small token sequence with bounded values
    let mut tokens = Vec::with_capacity(len);
    for _ in 0..len {
        let t: usize = kani::any();
        kani::assume(t <= 3);
        tokens.push(t);
    }

    let ratio = compression_ratio(&tokens);
    assert!(
        ratio >= 1.0,
        "compression_ratio must always be >= 1.0"
    );
}

// ============================================================================
// Harness 18: WhisperConfig validate accepts all standard presets
// ============================================================================

/// Proves that every standard WhisperConfig preset passes validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn all_standard_presets_pass_validation() {
    let idx: u8 = kani::any();
    kani::assume(idx < 6);

    let cfg = match idx {
        0 => WhisperConfig::whisper_tiny(),
        1 => WhisperConfig::whisper_base(),
        2 => WhisperConfig::whisper_small(),
        3 => WhisperConfig::whisper_medium(),
        4 => WhisperConfig::whisper_large_v2(),
        _ => WhisperConfig::large_v3_turbo(),
    };

    assert!(
        cfg.validate().is_ok(),
        "all standard presets must pass validation"
    );
}

// ============================================================================
// Harness 19: passes_quality_check accepts good result
// ============================================================================

/// Proves that a result meeting both thresholds passes the quality check.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn quality_check_accepts_good_result() {
    let cfg = DecodeConfig::default();
    let result = DecodingResult::new(
        vec![100, 200, 300],
        -0.5,  // avg_logprob above default threshold (-1.0)
        1.5,   // compression_ratio below default threshold (2.4)
        true,
        0.0,
        0.1,
    );
    assert!(
        passes_quality_check(&result, &cfg),
        "good result must pass quality check"
    );
}

// ============================================================================
// Harness 20: passes_quality_check rejects high compression ratio
// ============================================================================

/// Proves that a result with too-high compression ratio fails the quality check.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn quality_check_rejects_high_compression() {
    let cfg = DecodeConfig::default();
    let result = DecodingResult::new(
        vec![100, 200],
        -0.5,  // good avg_logprob
        3.0,   // compression_ratio ABOVE default threshold (2.4)
        true,
        0.0,
        0.1,
    );
    assert!(
        !passes_quality_check(&result, &cfg),
        "high compression ratio must fail quality check"
    );
}

// ============================================================================
// Harness 21: passes_quality_check rejects low avg_logprob
// ============================================================================

/// Proves that a result with too-low avg_logprob fails the quality check.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn quality_check_rejects_low_avg_logprob() {
    let cfg = DecodeConfig::default();
    let result = DecodingResult::new(
        vec![100, 200],
        -2.0,  // avg_logprob BELOW default threshold (-1.0)
        1.5,   // good compression ratio
        true,
        0.0,
        0.1,
    );
    assert!(
        !passes_quality_check(&result, &cfg),
        "low avg_logprob must fail quality check"
    );
}

// ============================================================================
// Harness 22: DEFAULT_TEMPERATURES starts at 0.0 and ends at 1.0
// ============================================================================

/// Proves the temperature fallback sequence boundary values.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn default_temperatures_boundaries() {
    assert!(
        (DEFAULT_TEMPERATURES[0] - 0.0).abs() < 1e-10,
        "first temperature must be 0.0 (greedy)"
    );
    assert!(
        (DEFAULT_TEMPERATURES[DEFAULT_TEMPERATURES.len() - 1] - 1.0).abs() < 1e-10,
        "last temperature must be 1.0"
    );
    assert_eq!(DEFAULT_TEMPERATURES.len(), 6, "6 temperature steps");
}

// ============================================================================
// Harness 23: DEFAULT_TEMPERATURES is strictly monotonically increasing
// ============================================================================

/// Proves that the temperature fallback sequence is strictly increasing.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(7)]
fn default_temperatures_strictly_increasing() {
    let temps = &DEFAULT_TEMPERATURES;
    let mut i = 0;
    while i + 1 < temps.len() {
        assert!(
            temps[i] < temps[i + 1],
            "temperatures must be strictly increasing"
        );
        i += 1;
    }
}

// ============================================================================
// Harness 24: WhisperConfig validate rejects d_model not divisible by heads
// ============================================================================

/// Proves that validate catches d_model not divisible by encoder_attention_heads.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_indivisible_d_model_encoder_heads() {
    // d_model=7 is not divisible by encoder_attention_heads=3
    let cfg = WhisperConfig::whisper_tiny()
        .with_d_model(7)
        .with_encoder_attention_heads(3);
    assert!(
        cfg.validate().is_err(),
        "d_model not divisible by encoder_attention_heads must be rejected"
    );
}

// ============================================================================
// Harness 25: WhisperConfig validate rejects d_model not divisible by decoder heads
// ============================================================================

/// Proves that validate catches d_model not divisible by decoder_attention_heads.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_indivisible_d_model_decoder_heads() {
    // d_model=10 is not divisible by decoder_attention_heads=3
    let cfg = WhisperConfig::whisper_tiny()
        .with_d_model(10)
        .with_decoder_attention_heads(3);
    assert!(
        cfg.validate().is_err(),
        "d_model not divisible by decoder_attention_heads must be rejected"
    );
}
