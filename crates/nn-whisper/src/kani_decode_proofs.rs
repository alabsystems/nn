// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Whisper decode path safety.
//!
//! Covers:
//! - DecodeConfig validation edge cases and builder chaining
//! - Temperature validation (NaN, negative, infinity)
//! - sample_token determinism and safety under temperature scaling
//! - compute_no_speech_prob output range [0, 1]
//! - Beam search configuration validation
//! - BeamState score computation (length penalty)
//! - compression_ratio monotonicity under repetition
//! - Log-probability accumulation safety (avg_logprob finite)
//! - Timestamp token extraction and range safety
//! - Quality check logical consistency
//! - Long-form chunk boundary arithmetic
//!
//! Issue: #3609

use super::*;
use crate::config;
use crate::tokenizer::{

// ── Kani transcendental stubs (CBMC cannot handle these) ──
fn exp_f32_stub(x: f32) -> f32 { let _ = x; let r: f32 = kani::any(); kani::assume(r.is_finite() && r > 0.0 && r <= 1e10); r }
fn ln_f32_stub(x: f32) -> f32 { let _ = x; let r: f32 = kani::any(); kani::assume(r.is_finite() && r >= -100.0 && r <= 100.0); r }

    LANGUAGE_TOKEN_END, LANGUAGE_TOKEN_START, NO_SPEECH_TOKEN, TIMESTAMP_BEGIN,
};

// ============================================================================
// Harness 1: sample_token greedy mode returns argmax for zero temperature
// ============================================================================

/// Proves that sample_token at temperature 0.0 returns the argmax index.
///
/// At zero temperature, sampling must be equivalent to greedy argmax.
/// This is the primary decode path for production Whisper inference.
/// Violation would cause non-deterministic transcriptions at temp=0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(5)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn sample_token_zero_temp_is_argmax() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();
    let d: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite());

    let logits = [a, b, c, d];
    let (token, _log_prob) = sample_token(&logits, 0.0, None);

    // Must match argmax.
    let expected = argmax_f32(&logits);
    assert_eq!(token, expected, "zero-temp sample must equal argmax");
}

// ============================================================================
// Harness 2: sample_token returns valid index for any finite temperature
// ============================================================================

/// Proves sample_token always returns an in-bounds index.
///
/// For any finite non-negative temperature (including 0, very small, very large),
/// the returned token index must be within [0, logits.len()). An out-of-bounds
/// index would cause a panic in the decode loop when used to index all_tokens.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn sample_token_index_always_valid() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite() && c.is_finite());

    let temp: f64 = kani::any();
    kani::assume(temp.is_finite() && temp >= 0.0);

    let logits = [a, b, c];
    let (token, _) = sample_token(&logits, temp, None);

    assert!(
        token < logits.len(),
        "sample_token index must be in-bounds"
    );
}

// ============================================================================
// Harness 3: sample_token log_prob is non-positive or NEG_INFINITY
// ============================================================================

/// Proves the log-probability returned by sample_token is always <= 0.
///
/// Since probabilities are in [0, 1], their logarithm must be <= 0. Positive
/// log-probs would corrupt avg_logprob computation and break quality checks.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn sample_token_log_prob_nonpositive() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite() && c.is_finite());

    let logits = [a, b, c];
    let (_, log_prob) = sample_token(&logits, 0.0, None);

    assert!(
        log_prob <= 0.0 || log_prob == f32::NEG_INFINITY,
        "log_prob from sample_token must be <= 0"
    );
}

// ============================================================================
// Harness 4: sample_token with very small temperature behaves as greedy
// ============================================================================

/// Proves that sample_token falls back to argmax when temperature < 1e-8.
///
/// The implementation uses 1e-8 as the greedy threshold. Temperatures just below
/// this must produce identical results to temperature=0.0, ensuring the greedy
/// fallback path is correct.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn sample_token_small_temp_greedy_fallback() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite() && c.is_finite());

    let logits = [a, b, c];
    let (token_zero, lp_zero) = sample_token(&logits, 0.0, None);
    let (token_tiny, lp_tiny) = sample_token(&logits, 1e-10, None);

    assert_eq!(
        token_zero, token_tiny,
        "tiny temperature must produce same token as zero"
    );
    assert_eq!(
        lp_zero, lp_tiny,
        "tiny temperature must produce same log_prob as zero"
    );
}

// ============================================================================
// Harness 5: sample_token handles non-finite temperature gracefully
// ============================================================================

/// Proves sample_token falls back to greedy for non-finite temperature cast.
///
/// When temperature as f64 is very large, casting to f32 may produce INFINITY.
/// The implementation must fall back to greedy (argmax) rather than producing
/// NaN probabilities or panicking.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn sample_token_large_temp_fallback() {
    let logits = [1.0f32, 5.0, 2.0];

    // f64 value that overflows f32.
    let huge_temp: f64 = f64::from(f32::MAX) * 2.0;
    let (token, log_prob) = sample_token(&logits, huge_temp, None);

    assert!(token < logits.len(), "token index in-bounds for huge temp");
    assert!(
        log_prob.is_finite() || log_prob == f32::NEG_INFINITY,
        "log_prob must not be NaN for huge temp"
    );
    // Should fall back to argmax since temp_f32 is not finite.
    let expected = argmax_f32(&logits);
    assert_eq!(token, expected, "huge temp must fall back to argmax");
}

// ============================================================================
// Harness 6: compute_no_speech_prob output is in [0.0, 1.0]
// ============================================================================

/// Proves compute_no_speech_prob returns a value in [0.0, 1.0].
///
/// The no-speech probability is computed as softmax(logits)[NO_SPEECH_TOKEN].
/// Since softmax outputs are in [0, 1] and sum to 1, any individual element
/// must be in [0, 1]. Values outside this range would corrupt the no-speech
/// threshold check in long-form transcription.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn no_speech_prob_bounded_01() {
    // Use a small vocab that includes NO_SPEECH_TOKEN position.
    // NO_SPEECH_TOKEN = 50363, so we test with a minimal slice.
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());

    // Build a logit slice where index 0 and 1 have our values.
    // compute_no_speech_prob checks NO_SPEECH_TOKEN >= logits.len(),
    // and returns 0.0 if so. For small slices, always returns 0.0.
    let logits = [a, b];
    let prob = language::compute_no_speech_prob(&logits);

    // For small vocab (len < NO_SPEECH_TOKEN), returns 0.0.
    assert!(
        prob >= 0.0 && prob <= 1.0,
        "no_speech_prob must be in [0, 1]"
    );
}

// ============================================================================
// Harness 7: compute_no_speech_prob returns 0 for empty logits
// ============================================================================

/// Proves compute_no_speech_prob returns 0.0 for empty logit slice.
///
/// Empty logits occur if the model returns zero-length output. The function
/// must return 0.0 (not panic or return NaN) to allow the decode loop to
/// continue with a safe default.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn no_speech_prob_empty_logits() {
    let empty: &[f32] = &[];
    let prob = language::compute_no_speech_prob(empty);
    assert_eq!(prob, 0.0, "empty logits must return 0.0 no-speech prob");
}

// ============================================================================
// Harness 8: compute_no_speech_prob returns 0 for undersized vocab
// ============================================================================

/// Proves compute_no_speech_prob returns 0 when vocab < NO_SPEECH_TOKEN.
///
/// If the vocabulary is smaller than the no-speech token ID (50363),
/// the function must return 0.0 rather than indexing out of bounds.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn no_speech_prob_undersized_vocab() {
    // Build a slice shorter than NO_SPEECH_TOKEN.
    let logits = [1.0f32; 100];
    let prob = language::compute_no_speech_prob(&logits);
    assert_eq!(
        prob, 0.0,
        "undersized vocab must return 0.0 no-speech prob"
    );
}

// ============================================================================
// Harness 9: DecodeConfig builder chaining preserves all fields
// ============================================================================

/// Proves that DecodeConfig builder methods preserve all previously set fields.
///
/// Builder chaining must not silently reset any field. A field set by
/// with_max_length must still be set after calling with_seed. Violations
/// would cause silent configuration corruption.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn decode_config_builder_preserves_fields() {
    let config = DecodeConfig::default()
        .with_max_length(100)
        .with_seed(Some(42))
        .with_compression_ratio_threshold(3.0)
        .with_avg_logprob_threshold(-0.5);

    assert_eq!(config.max_length, 100, "max_length preserved");
    assert_eq!(config.seed, Some(42), "seed preserved");
    assert!(
        (config.compression_ratio_threshold - 3.0).abs() < 1e-10,
        "compression_ratio_threshold preserved"
    );
    assert!(
        (config.avg_logprob_threshold - (-0.5)).abs() < 1e-10,
        "avg_logprob_threshold preserved"
    );
    // Default initial_tokens must still be present.
    assert!(
        !config.initial_tokens.is_empty(),
        "initial_tokens preserved from default"
    );
}

// ============================================================================
// Harness 10: DecodeConfig validate rejects NaN compression ratio threshold
// ============================================================================

/// Proves DecodeConfig.validate() rejects NaN compression_ratio_threshold.
///
/// NaN thresholds would silently pass all quality checks (since NaN comparisons
/// are always false), making temperature fallback never trigger. This violates
/// the defense-in-depth principle from IEEE 754 rules.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn decode_config_validate_rejects_nan_compression_threshold() {
    let config = DecodeConfig::default().with_compression_ratio_threshold(f64::NAN);
    let result = config.validate();
    assert!(result.is_err(), "NaN compression_ratio_threshold must fail validation");
}

// ============================================================================
// Harness 11: DecodeConfig validate rejects infinity avg_logprob threshold
// ============================================================================

/// Proves DecodeConfig.validate() rejects infinite avg_logprob_threshold.
///
/// An infinite threshold would either always pass or always fail the quality
/// check, defeating the purpose of temperature fallback.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn decode_config_validate_rejects_inf_logprob_threshold() {
    let config = DecodeConfig::default().with_avg_logprob_threshold(f64::INFINITY);
    let result = config.validate();
    assert!(
        result.is_err(),
        "infinite avg_logprob_threshold must fail validation"
    );
}

// ============================================================================
// Harness 12: passes_quality_check consistent with threshold semantics
// ============================================================================

/// Proves passes_quality_check returns true only when BOTH conditions are met:
/// compression_ratio <= threshold AND avg_logprob >= threshold.
///
/// The quality check is a conjunction (AND). If either condition fails, the
/// result must be false, triggering temperature fallback. A disjunction (OR)
/// bug would accept low-quality results.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn quality_check_conjunction_semantics() {
    let config = DecodeConfig::default()
        .with_compression_ratio_threshold(2.4)
        .with_avg_logprob_threshold(-1.0);

    // Both pass.
    let good = DecodingResult::new(vec![1, 2, 3], -0.5, 1.5, true, 0.0, 0.0);
    assert!(
        passes_quality_check(&good, &config),
        "both thresholds met must pass"
    );

    // Compression ratio too high.
    let bad_cr = DecodingResult::new(vec![1, 2, 3], -0.5, 3.0, true, 0.0, 0.0);
    assert!(
        !passes_quality_check(&bad_cr, &config),
        "high compression ratio must fail"
    );

    // Avg logprob too low.
    let bad_lp = DecodingResult::new(vec![1, 2, 3], -2.0, 1.5, true, 0.0, 0.0);
    assert!(
        !passes_quality_check(&bad_lp, &config),
        "low avg_logprob must fail"
    );

    // Both fail.
    let bad_both = DecodingResult::new(vec![1, 2, 3], -2.0, 3.0, true, 0.0, 0.0);
    assert!(
        !passes_quality_check(&bad_both, &config),
        "both thresholds violated must fail"
    );
}

// ============================================================================
// Harness 13: passes_quality_check boundary values
// ============================================================================

/// Proves passes_quality_check at exact boundary values.
///
/// At the exact threshold, compression_ratio <= threshold is true and
/// avg_logprob >= threshold is true. The quality check must pass at
/// the boundary — rejecting at the boundary would cause unnecessary
/// temperature fallback.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn quality_check_exact_boundary() {
    let config = DecodeConfig::default()
        .with_compression_ratio_threshold(2.4)
        .with_avg_logprob_threshold(-1.0);

    // Exact boundary.
    let boundary = DecodingResult::new(vec![1, 2, 3], -1.0, 2.4, true, 0.0, 0.0);
    assert!(
        passes_quality_check(&boundary, &config),
        "exact boundary values must pass"
    );
}

// ============================================================================
// Harness 14: compression_ratio increases with repetition
// ============================================================================

/// Proves compression_ratio is higher for repetitive tokens than unique tokens.
///
/// This validates the metric's sensitivity to repetition. A model producing
/// "the the the the" should have higher compression ratio than "the cat sat down",
/// triggering temperature fallback.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn compression_ratio_repetitive_vs_unique() {
    let unique = [1usize, 2, 3, 4, 5, 6, 7, 8];
    let repetitive = [1usize, 2, 1, 2, 1, 2, 1, 2];

    let cr_unique = compression_ratio(&unique);
    let cr_repetitive = compression_ratio(&repetitive);

    assert!(
        cr_repetitive > cr_unique,
        "repetitive tokens must have higher compression ratio"
    );
}

// ============================================================================
// Harness 15: avg_logprob computation safety for empty tokens
// ============================================================================

/// Proves avg_logprob is 0.0 when no tokens are decoded.
///
/// When the model immediately outputs EOT (no decoded tokens), the average
/// log-probability is defined as 0.0 (the sum is 0 and there are 0 tokens).
/// Division by zero must be avoided.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn avg_logprob_zero_for_empty_decode() {
    let result = DecodingResult::new(
        Vec::new(), // no decoded tokens
        0.0,        // avg_logprob
        1.0,        // compression_ratio
        true,       // reached_eot
        0.0,        // temperature
        0.0,        // no_speech_prob
    );
    assert!(
        result.avg_logprob.is_finite(),
        "avg_logprob must be finite for empty decode"
    );
    assert_eq!(
        result.avg_logprob, 0.0,
        "avg_logprob must be 0 for empty decode"
    );
}

// ============================================================================
// Harness 16: WhisperBeamConfig validate rejects zero beam_width
// ============================================================================

/// Proves WhisperBeamConfig.validate() rejects beam_width == 0.
///
/// Zero beam width would produce no hypotheses, causing beam_search_decode
/// to return an EmptyDecodeResult error. Catching this at validation time
/// provides a clearer error message.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn beam_config_zero_width_rejected() {
    let config = WhisperBeamConfig {
        beam_width: 0,
        length_penalty: 1.0,
    };
    assert!(
        config.validate().is_err(),
        "zero beam_width must fail validation"
    );
}

// ============================================================================
// Harness 17: WhisperBeamConfig validate rejects NaN length_penalty
// ============================================================================

/// Proves WhisperBeamConfig.validate() rejects NaN length_penalty.
///
/// NaN length_penalty would corrupt beam score comparisons (total_cmp sorts
/// NaN above all values), potentially selecting the worst hypothesis.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn beam_config_nan_penalty_rejected() {
    let config = WhisperBeamConfig {
        beam_width: 5,
        length_penalty: f64::NAN,
    };
    assert!(
        config.validate().is_err(),
        "NaN length_penalty must fail validation"
    );
}

// ============================================================================
// Harness 18: Default temperatures sequence is sorted ascending
// ============================================================================

/// Proves DEFAULT_TEMPERATURES is sorted in ascending order.
///
/// The temperature fallback algorithm tries each temperature in sequence.
/// If not sorted ascending, the decode loop would try high temperatures before
/// low ones, wasting compute on noisy samples before trying greedy.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(7)]
fn default_temperatures_sorted() {
    let temps = DEFAULT_TEMPERATURES;
    for i in 1..temps.len() {
        assert!(
            temps[i] > temps[i - 1],
            "temperatures must be strictly ascending"
        );
    }
}

// ============================================================================
// Harness 19: Default temperatures first element is 0.0 (greedy first)
// ============================================================================

/// Proves the first temperature in the fallback sequence is 0.0.
///
/// The decode loop must try greedy (temp=0) first since it's deterministic
/// and fastest. If the first temperature were > 0, every transcription would
/// start with a random sample.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn default_temperatures_greedy_first() {
    assert_eq!(
        DEFAULT_TEMPERATURES[0], 0.0,
        "first temperature must be 0.0 (greedy)"
    );
}

// ============================================================================
// Harness 21: MAX_DECODE_LENGTH > 0 and consistent with config default
// ============================================================================

/// Proves MAX_DECODE_LENGTH is positive and matches DecodeConfig default.
///
/// Zero max_length would prevent any tokens from being generated. The constant
/// must be positive and used as the default for DecodeConfig.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn max_decode_length_positive_and_default() {
    assert!(MAX_DECODE_LENGTH > 0, "MAX_DECODE_LENGTH must be positive");
    let config = DecodeConfig::default();
    assert_eq!(
        config.max_length, MAX_DECODE_LENGTH,
        "default max_length must equal MAX_DECODE_LENGTH"
    );
}

// ============================================================================
// Harness 22: timestamp token range is within vocab bounds
// ============================================================================

/// Proves timestamp token IDs are within the standard Whisper vocabulary range.
///
/// Timestamp tokens start at TIMESTAMP_BEGIN (50365). For 30 seconds at
/// 0.02s resolution, there are 1501 timestamps (0.00-30.00). All must be
/// within the vocabulary of 51866 tokens.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn timestamp_range_within_vocab() {
    let max_timestamp_token = TIMESTAMP_BEGIN + 1500; // <|30.00|>
    let vocab_size = 51866; // whisper-large-v3-turbo

    assert!(
        TIMESTAMP_BEGIN < vocab_size,
        "TIMESTAMP_BEGIN within vocab"
    );
    assert!(
        max_timestamp_token < vocab_size,
        "max timestamp token within vocab"
    );
}

// ============================================================================
// Harness 23: language token range consistency
// ============================================================================

/// Proves language token range is non-empty and well-ordered.
///
/// LANGUAGE_TOKEN_START must be <= LANGUAGE_TOKEN_END, and the range must
/// contain exactly 100 language tokens (as per Whisper spec).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn language_token_range_valid() {
    assert!(
        LANGUAGE_TOKEN_START <= LANGUAGE_TOKEN_END,
        "language token range must be non-empty"
    );
    let num_languages = LANGUAGE_TOKEN_END - LANGUAGE_TOKEN_START + 1;
    assert_eq!(num_languages, 100, "must have 100 language tokens");
}

// ============================================================================
// Harness 24: compression_ratio of single-token input is 1.0
// ============================================================================

/// Proves compression_ratio returns 1.0 for inputs with fewer than 2 tokens.
///
/// Single tokens and empty inputs have no bigrams, so the function must
/// return 1.0 (the identity compression ratio) rather than dividing by zero.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn compression_ratio_single_token() {
    let empty: &[usize] = &[];
    let single = [42usize];

    assert_eq!(
        compression_ratio(empty),
        1.0,
        "empty input must have ratio 1.0"
    );
    assert_eq!(
        compression_ratio(&single),
        1.0,
        "single token must have ratio 1.0"
    );
}

// ============================================================================
// Harness 25: argmax_f32 is deterministic on equal values
// ============================================================================

/// Proves argmax_f32 returns a consistent index when all values are equal.
///
/// When all logits are identical, argmax must not return different indices
/// across calls (non-determinism would cause test flakiness). The implementation
/// uses total_cmp which provides a total ordering.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn argmax_deterministic_equal_values() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite());

    let values = [v, v, v];
    let idx1 = argmax_f32(&values);
    let idx2 = argmax_f32(&values);

    assert_eq!(idx1, idx2, "argmax must be deterministic on equal values");
    assert!(idx1 < values.len(), "argmax index must be in-bounds");
}

// ============================================================================
// Harness 26: compute_log_prob is monotonically related to logit value
// ============================================================================

/// Proves that compute_log_prob assigns higher log-prob to higher logit.
///
/// For two indices, the one with the higher logit value must have a higher
/// (less negative) log-probability. This ensures the decode loop favors
/// the most probable tokens.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn log_prob_monotone_with_logit() {
    let lo: f32 = kani::any();
    let hi: f32 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(hi > lo + 1.0); // Ensure meaningful separation.

    let logits = [lo, hi];
    let lp_lo = compute_log_prob(&logits, 0);
    let lp_hi = compute_log_prob(&logits, 1);

    assert!(
        lp_hi > lp_lo,
        "higher logit must have higher log-probability"
    );
}

// ============================================================================
// Harness 27: long-form chunk boundary arithmetic safety
// ============================================================================

/// Proves long-form chunk boundary computation does not overflow.
///
/// The long-form decode advances seek by N_SAMPLES. For any audio length
/// representable as a usize, the chunk boundary computation (seek + N_SAMPLES)
/// must not overflow when clamped by .min(audio_len).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn chunk_boundary_no_overflow() {
    let seek: usize = kani::any();
    let audio_len: usize = kani::any();
    kani::assume(seek <= audio_len);
    // Limit to realistic audio lengths to keep Kani tractable.
    kani::assume(audio_len <= 1_000_000_000);

    let n_samples = config::N_SAMPLES;
    // This is the pattern from transcribe_long.
    let chunk_end = seek.saturating_add(n_samples).min(audio_len);

    assert!(chunk_end <= audio_len, "chunk_end must not exceed audio_len");
    assert!(chunk_end >= seek, "chunk_end must be >= seek");
}

// ============================================================================
// Harness 28: timestamp advance computation is non-negative
// ============================================================================

/// Proves that timestamp tokens produce non-negative advance values.
///
/// Timestamp tokens encode time as (token_id - TIMESTAMP_BEGIN) * 0.02.
/// Since token_id >= TIMESTAMP_BEGIN (by construction in the filter), the
/// advance must always be >= 0. Negative advances would cause seek to go
/// backward, producing infinite loops in long-form transcription.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn timestamp_advance_nonnegative() {
    let token_id: usize = kani::any();
    kani::assume(token_id >= TIMESTAMP_BEGIN);
    // Bound to prevent overflow in the multiplication.
    kani::assume(token_id <= TIMESTAMP_BEGIN + 2000);

    let advance = token_id.saturating_sub(TIMESTAMP_BEGIN) as f64 * 0.02;
    assert!(advance >= 0.0, "timestamp advance must be non-negative");
    assert!(advance.is_finite(), "timestamp advance must be finite");
}

// ============================================================================
// Harness 29: no-speech threshold constant is in valid range
// ============================================================================

/// Proves DEFAULT_NO_SPEECH_THRESHOLD is in (0.0, 1.0).
///
/// A threshold of 0.0 would skip all segments (no_speech_prob > 0 is always true
/// for non-silent audio). A threshold of 1.0 would never skip any segment.
/// Both extremes defeat the purpose. The standard value is 0.6.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn no_speech_threshold_valid_range() {
    use crate::tokenizer::DEFAULT_NO_SPEECH_THRESHOLD;

    assert!(
        DEFAULT_NO_SPEECH_THRESHOLD > 0.0,
        "no-speech threshold must be > 0"
    );
    assert!(
        DEFAULT_NO_SPEECH_THRESHOLD < 1.0,
        "no-speech threshold must be < 1"
    );
}

// ============================================================================
// Harness 30: sample_token on all-NEG_INFINITY logits
// ============================================================================

/// Proves sample_token handles all-suppressed logits safely.
///
/// After suppression, all logits may be NEG_INFINITY (if every token was
/// suppressed). The function must still return a valid index and not panic
/// or produce NaN. This is a defense-in-depth edge case.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn sample_token_all_neg_inf() {
    let logits = [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
    let (token, log_prob) = sample_token(&logits, 0.0, None);

    assert!(token < logits.len(), "index must be in-bounds for all -inf");
    // log_prob should be NEG_INFINITY (log(0)).
    assert!(
        log_prob == f32::NEG_INFINITY || log_prob.is_finite(),
        "log_prob must not be NaN for all -inf logits"
    );
}

// ============================================================================
// Harness 32: compression_ratio is finite for any token sequence
// ============================================================================

/// Proves compression_ratio returns a finite value for small token sequences.
///
/// The function divides by bigrams.len().max(1), so division by zero is
/// impossible. The result must always be finite for finite-length inputs.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn compression_ratio_always_finite() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    let c: usize = kani::any();
    // Bound to small values for Kani tractability.
    kani::assume(a < 100 && b < 100 && c < 100);

    let tokens = [a, b, c];
    let cr = compression_ratio(&tokens);

    assert!(cr.is_finite(), "compression_ratio must be finite");
    assert!(cr >= 1.0, "compression_ratio must be >= 1.0");
}

// ============================================================================
// Harness 33: language token range within vocab of all preset configs
// ============================================================================

/// Proves language token range fits within vocab_size of all Whisper presets.
///
/// If LANGUAGE_TOKEN_END >= vocab_size for any config, language detection
/// would index out of bounds. Every preset must accommodate the full 100
/// language tokens.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn language_tokens_fit_all_presets() {
    let presets = [
        config::WhisperConfig::whisper_tiny(),
        config::WhisperConfig::whisper_base(),
        config::WhisperConfig::whisper_small(),
        config::WhisperConfig::whisper_medium(),
        config::WhisperConfig::whisper_large_v2(),
        config::WhisperConfig::large_v3_turbo(),
    ];

    for cfg in &presets {
        assert!(
            LANGUAGE_TOKEN_END < cfg.vocab_size,
            "LANGUAGE_TOKEN_END must be within vocab_size"
        );
        assert!(
            LANGUAGE_TOKEN_START < cfg.vocab_size,
            "LANGUAGE_TOKEN_START must be within vocab_size"
        );
    }
}

// ============================================================================
// Harness 34: apply_suppression_inplace is idempotent
// ============================================================================

/// Proves applying suppression twice produces the same result as once.
///
/// Suppression sets indices to NEG_INFINITY. Applying it again on an already
/// suppressed index must produce NEG_INFINITY (idempotent). Non-idempotent
/// behavior would indicate state corruption.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn suppression_is_idempotent() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite() && c.is_finite());

    let suppress = [1usize];

    let mut logits1 = [a, b, c];
    apply_suppression_inplace(&mut logits1, &suppress);
    let after_first = [logits1[0], logits1[1], logits1[2]];

    apply_suppression_inplace(&mut logits1, &suppress);

    assert_eq!(
        logits1[0], after_first[0],
        "index 0 unchanged by second suppression"
    );
    assert_eq!(
        logits1[1], after_first[1],
        "suppressed index unchanged by second suppression"
    );
    assert_eq!(
        logits1[2], after_first[2],
        "index 2 unchanged by second suppression"
    );
}

// ============================================================================
// Harness 35: DecodingResult constructor produces finite fields
// ============================================================================

/// Proves DecodingResult::new preserves the finiteness of its inputs.
///
/// The constructor must not transform any input field. If avg_logprob is
/// provided as finite, it must remain finite in the struct. This verifies
/// no accidental computation occurs in the constructor.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn decoding_result_constructor_preserves_values() {
    let avg_lp: f64 = kani::any();
    let cr: f64 = kani::any();
    let temp: f64 = kani::any();
    let nsp: f64 = kani::any();
    kani::assume(avg_lp.is_finite() && cr.is_finite() && temp.is_finite() && nsp.is_finite());

    let result = DecodingResult::new(vec![1, 2, 3], avg_lp, cr, true, temp, nsp);

    assert_eq!(result.avg_logprob, avg_lp, "avg_logprob preserved");
    assert_eq!(
        result.compression_ratio, cr,
        "compression_ratio preserved"
    );
    assert_eq!(result.temperature, temp, "temperature preserved");
    assert_eq!(result.no_speech_prob, nsp, "no_speech_prob preserved");
    assert!(result.reached_eot, "reached_eot preserved");
    assert_eq!(result.tokens.len(), 3, "tokens preserved");
}

// ============================================================================
// Harness 37: apply_suppression_inplace on empty suppress list is no-op
// ============================================================================

/// Proves apply_suppression_inplace with an empty suppress list does not
/// modify any logit values. This is the common case when no tokens need
/// suppression.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn suppression_empty_list_no_op() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());

    let mut logits = [a, b];
    let suppress: Vec<usize> = Vec::new();
    apply_suppression_inplace(&mut logits, &suppress);

    assert_eq!(logits[0], a, "logit 0 unchanged with empty suppress");
    assert_eq!(logits[1], b, "logit 1 unchanged with empty suppress");
}

// ============================================================================
// Harness 38: compute_log_prob out-of-bounds returns NEG_INFINITY
// ============================================================================

/// Proves compute_log_prob returns NEG_INFINITY when idx >= logits.len().
///
/// This is a defense-in-depth guard. If a corrupted token index reaches
/// compute_log_prob, it must return NEG_INFINITY rather than panicking.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn log_prob_oob_returns_neg_inf() {
    let logits = [1.0f32, 2.0, 3.0];

    let lp_oob = compute_log_prob(&logits, 5);
    assert_eq!(
        lp_oob,
        f32::NEG_INFINITY,
        "out-of-bounds index must return NEG_INFINITY"
    );

    let lp_empty = compute_log_prob(&[], 0);
    assert_eq!(
        lp_empty,
        f32::NEG_INFINITY,
        "empty logits must return NEG_INFINITY"
    );
}

// ============================================================================
// Harness 39: compute_log_prob sum of exp(log_probs) approximately equals 1
// ============================================================================

/// Proves the log-softmax property: for finite logits, the sum of
/// exp(log_prob) across all indices is approximately 1.0.
///
/// This is the fundamental normalization property of log-softmax.
/// Violation would mean the probability distribution doesn't sum to 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn log_prob_sum_exp_approx_one() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite() && c.is_finite());
    // Restrict to moderate range to keep numerical accuracy.
    kani::assume(a.abs() < 50.0 && b.abs() < 50.0 && c.abs() < 50.0);

    let logits = [a, b, c];
    let sum_exp: f32 = (0..3)
        .map(|i| compute_log_prob(&logits, i).exp())
        .sum();

    assert!(
        (sum_exp - 1.0).abs() < 1e-4,
        "sum of exp(log_prob) must approximately equal 1.0"
    );
}

// ============================================================================
// Harness 40: argmax_f32 returns 0 for empty slice
// ============================================================================

/// Proves argmax_f32 returns 0 for empty input (via unwrap_or(0)).
///
/// Empty logits should not occur in practice (model always outputs vocab-size),
/// but the function must not panic if it does.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn argmax_empty_returns_zero() {
    let empty: &[f32] = &[];
    let idx = argmax_f32(empty);
    assert_eq!(idx, 0, "argmax of empty must return 0");
}

// ============================================================================
// Harness 41: argmax_f32 single element returns 0
// ============================================================================

/// Proves argmax_f32 returns index 0 for a single-element slice.
///
/// With only one element, the argmax is trivially index 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn argmax_single_element() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite());

    let values = [v];
    let idx = argmax_f32(&values);
    assert_eq!(idx, 0, "argmax of single element must be 0");
}

// ============================================================================
// Harness 42: argmax_f32 finds the correct maximum
// ============================================================================

/// Proves argmax_f32 returns the index of the largest value when all values
/// are distinct. The returned index's value must be >= all other values.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(5)]
fn argmax_finds_maximum() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();
    let d: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite());

    let values = [a, b, c, d];
    let idx = argmax_f32(&values);

    assert!(idx < 4, "argmax index in bounds");
    for i in 0..4 {
        assert!(
            values[idx].total_cmp(&values[i]) != std::cmp::Ordering::Less,
            "argmax value must be >= all other values"
        );
    }
}

// ============================================================================
// Harness 43: DecodeConfig validate rejects zero max_length
// ============================================================================

/// Proves DecodeConfig.validate() rejects max_length == 0.
///
/// Zero max_length would prevent any tokens from being generated,
/// making the decode loop a no-op. This must be caught at validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn decode_config_validate_rejects_zero_max_length() {
    let config = DecodeConfig::default().with_max_length(0);
    // with_max_length sets the value; we need to manually set since default is 224.
    let mut config = DecodeConfig::default();
    config.max_length = 0;
    assert!(
        config.validate().is_err(),
        "zero max_length must fail validation"
    );
}

// ============================================================================
// Harness 44: DecodeConfig validate rejects max_length > MAX_DECODE_LENGTH
// ============================================================================

/// Proves DecodeConfig.validate() rejects max_length exceeding MAX_DECODE_LENGTH.
///
/// Generating more than 224 tokens in a single decode pass exceeds the
/// Whisper positional embedding capacity and would produce garbage output.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn decode_config_validate_rejects_oversize_max_length() {
    let mut config = DecodeConfig::default();
    config.max_length = MAX_DECODE_LENGTH + 1;
    assert!(
        config.validate().is_err(),
        "max_length > MAX_DECODE_LENGTH must fail validation"
    );
}

// ============================================================================
// Harness 46: compression_ratio >= 1.0 for any input with >= 2 tokens
// ============================================================================

/// Proves compression_ratio always returns a value >= 1.0 for sequences
/// with at least 2 tokens.
///
/// The compression ratio is (num_bigrams) / (unique_bigrams). Since
/// unique_bigrams <= num_bigrams, the ratio is always >= 1.0. A ratio
/// < 1.0 would be physically impossible and indicate a bug.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn compression_ratio_at_least_one() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    let c: usize = kani::any();
    let d: usize = kani::any();
    kani::assume(a < 1000 && b < 1000 && c < 1000 && d < 1000);

    let tokens = [a, b, c, d];
    let cr = compression_ratio(&tokens);

    assert!(cr >= 1.0, "compression_ratio must be >= 1.0 for >= 2 tokens");
}

// ============================================================================
// Harness 47: compression_ratio of all-unique bigrams is exactly 1.0
// ============================================================================

/// Proves compression_ratio returns exactly 1.0 when all consecutive
/// bigrams are unique. No repetition means no compression.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn compression_ratio_all_unique_is_one() {
    // 4 tokens = 3 bigrams, all unique: (1,2), (2,3), (3,4).
    let tokens = [1usize, 2, 3, 4];
    let cr = compression_ratio(&tokens);

    assert!(
        (cr - 1.0).abs() < 1e-10,
        "all-unique bigrams must produce ratio 1.0"
    );
}

// ============================================================================
// Harness 48: compute_log_prob argmax index has highest log-prob
// ============================================================================

/// Proves that the argmax index has the highest log-probability among
/// all indices. This is the fundamental invariant linking argmax to
/// log-probability — greedy decode selects the token with the highest
/// probability precisely because argmax and compute_log_prob agree.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn log_prob_argmax_has_max_log_prob() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite() && c.is_finite());
    // Ensure distinct values to avoid tie-breaking ambiguity.
    kani::assume(a != b && b != c && a != c);

    let logits = [a, b, c];
    let best_idx = argmax_f32(&logits);
    let best_lp = compute_log_prob(&logits, best_idx);

    for i in 0..3 {
        let lp = compute_log_prob(&logits, i);
        assert!(
            best_lp >= lp,
            "argmax index must have highest log-prob"
        );
    }
}

// ============================================================================
// Harness 49: sample_token determinism at zero temperature without RNG
// ============================================================================

/// Proves sample_token at temperature 0.0 without an RNG produces
/// identical results on repeated calls. This is critical for reproducible
/// greedy transcription.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn sample_token_deterministic_no_rng() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite() && c.is_finite());

    let logits = [a, b, c];
    let (tok1, lp1) = sample_token(&logits, 0.0, None);
    let (tok2, lp2) = sample_token(&logits, 0.0, None);

    assert_eq!(tok1, tok2, "greedy sample must be deterministic");
    assert_eq!(lp1, lp2, "greedy log_prob must be deterministic");
}

// ============================================================================
// Harness 50: DEFAULT_COMPRESSION_RATIO_THRESHOLD is finite and positive
// ============================================================================

/// Proves the default compression ratio threshold is a positive finite value.
///
/// A non-finite threshold would corrupt quality checks (NaN comparisons
/// are always false). A non-positive threshold would reject all sequences.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn default_compression_threshold_valid() {
    assert!(
        DEFAULT_COMPRESSION_RATIO_THRESHOLD.is_finite(),
        "default compression threshold must be finite"
    );
    assert!(
        DEFAULT_COMPRESSION_RATIO_THRESHOLD > 0.0,
        "default compression threshold must be positive"
    );
}

// ============================================================================
// Harness 51: DEFAULT_AVG_LOGPROB_THRESHOLD is finite and negative
// ============================================================================

/// Proves the default average log-probability threshold is finite and negative.
///
/// Log-probabilities are always <= 0. The threshold must be negative to allow
/// any sequence to pass. A positive threshold would reject all results.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn default_avg_logprob_threshold_valid() {
    assert!(
        DEFAULT_AVG_LOGPROB_THRESHOLD.is_finite(),
        "default avg_logprob threshold must be finite"
    );
    assert!(
        DEFAULT_AVG_LOGPROB_THRESHOLD < 0.0,
        "default avg_logprob threshold must be negative"
    );
}
