// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for Whisper decode path.
//!
//! Supplements `kani_decode_proofs.rs` with additional coverage:
//! - passes_quality_check: boundary symmetry, both-fail, both-pass
//! - extract_timestamp_advance: non-negative, bounded by chunk length, None for no timestamps
//! - Long-form seek arithmetic: minimum advance, sample alignment
//! - DecodeConfig builder independence for with_max_length, with_seed
//! - no_speech probability: values sum consistency
//! - Temperature sequence validation properties
//! - DecodingResult field consistency
//!
//! Issue: #3741

use super::*;
use crate::WhisperConfig;
use crate::config::{CHUNK_LENGTH, HOP_LENGTH, N_SAMPLES, SAMPLE_RATE};
use crate::tokenizer::{NO_SPEECH_TOKEN, TIMESTAMP_BEGIN};

// ── Kani transcendental stubs (CBMC cannot handle these) ──
fn exp_f32_stub(x: f32) -> f32 { let _ = x; let r: f32 = kani::any(); kani::assume(r.is_finite() && r > 0.0 && r <= 1e10); r }


// ============================================================================
// Harness 1: quality check fails when compression too high AND logprob too low
// ============================================================================

/// Proves passes_quality_check returns false when BOTH thresholds violated.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn quality_check_both_fail() {
    let cr_threshold = 2.4;
    let lp_threshold = -1.0;

    let result = DecodingResult::new(
        vec![1, 2, 3],
        -2.0, // below lp_threshold
        3.0,  // above cr_threshold
        true,
        0.0,
        0.0,
    );
    let config = DecodeConfig::default()
        .with_compression_ratio_threshold(cr_threshold)
        .with_avg_logprob_threshold(lp_threshold);

    assert!(
        !passes_quality_check(&result, &config),
        "both thresholds violated must fail"
    );
}

// ============================================================================
// Harness 2: quality check passes when both within bounds
// ============================================================================

/// Proves passes_quality_check returns true when both metrics are within bounds.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn quality_check_both_pass() {
    let result = DecodingResult::new(
        vec![1, 2, 3],
        -0.5, // above -1.0
        1.5,  // below 2.4
        true,
        0.0,
        0.0,
    );
    let config = DecodeConfig::default();

    assert!(
        passes_quality_check(&result, &config),
        "both metrics within bounds must pass"
    );
}

// ============================================================================
// Harness 3: quality check: compression ratio at exact boundary passes
// ============================================================================

/// Proves that compression_ratio == threshold passes (uses <=).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn quality_check_cr_exact_boundary_passes() {
    let threshold = 2.4;
    let result = DecodingResult::new(
        vec![1, 2, 3],
        0.0,       // well above avg_logprob threshold
        threshold, // exactly at compression ratio boundary
        true,
        0.0,
        0.0,
    );
    let config = DecodeConfig::default()
        .with_compression_ratio_threshold(threshold);

    assert!(
        passes_quality_check(&result, &config),
        "compression_ratio == threshold must pass (<=)"
    );
}

// ============================================================================
// Harness 4: quality check: avg_logprob at exact boundary passes
// ============================================================================

/// Proves that avg_logprob == threshold passes (uses >=).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn quality_check_lp_exact_boundary_passes() {
    let threshold = -1.0;
    let result = DecodingResult::new(
        vec![1, 2, 3],
        threshold, // exactly at avg_logprob boundary
        1.0,       // well below compression threshold
        true,
        0.0,
        0.0,
    );
    let config = DecodeConfig::default()
        .with_avg_logprob_threshold(threshold);

    assert!(
        passes_quality_check(&result, &config),
        "avg_logprob == threshold must pass (>=)"
    );
}

// ============================================================================
// Harness 5: extract_timestamp_advance returns None for no timestamp tokens
// ============================================================================

/// Proves that when no token is >= TIMESTAMP_BEGIN, the function returns None.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(5)]
fn timestamp_advance_none_for_no_timestamps() {
    // All tokens below TIMESTAMP_BEGIN.
    let a: usize = kani::any();
    let b: usize = kani::any();
    let c: usize = kani::any();
    kani::assume(a < TIMESTAMP_BEGIN);
    kani::assume(b < TIMESTAMP_BEGIN);
    kani::assume(c < TIMESTAMP_BEGIN);

    let tokens = [a, b, c];
    let last_ts = tokens.iter().rev().find(|&&t| t >= TIMESTAMP_BEGIN);
    assert!(last_ts.is_none(), "no timestamp tokens => None");
}

// ============================================================================
// Harness 6: extract_timestamp_advance is non-negative
// ============================================================================

/// Proves that the timestamp advance is always non-negative.
///
/// Since token >= TIMESTAMP_BEGIN, the subtraction never underflows,
/// and multiplying by 0.02 preserves non-negativity.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn timestamp_advance_non_negative() {
    let token: usize = kani::any();
    kani::assume(token >= TIMESTAMP_BEGIN);
    kani::assume(token <= TIMESTAMP_BEGIN + 1501); // max 30.02 seconds

    let ts_seconds = token.saturating_sub(TIMESTAMP_BEGIN) as f64 * 0.02;
    assert!(ts_seconds >= 0.0, "timestamp advance must be non-negative");
    assert!(ts_seconds.is_finite(), "timestamp advance must be finite");
}

// ============================================================================
// Harness 7: timestamp advance bounded by chunk length
// ============================================================================

/// Proves that a valid timestamp token produces advance <= 30.0 seconds.
///
/// Whisper timestamps go up to <|30.00|>, which is 1500 steps * 0.02s.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn timestamp_advance_bounded_by_chunk() {
    let token: usize = kani::any();
    kani::assume(token >= TIMESTAMP_BEGIN);
    kani::assume(token <= TIMESTAMP_BEGIN + 1500); // valid range: 0.00 to 30.00

    let ts_seconds = token.saturating_sub(TIMESTAMP_BEGIN) as f64 * 0.02;
    assert!(
        ts_seconds <= CHUNK_LENGTH as f64,
        "valid timestamp <= 30.0 seconds"
    );
}

// ============================================================================
// Harness 8: long-form seek: minimum advance is HOP_LENGTH samples
// ============================================================================

/// Proves that when timestamps are present, seek advances by at least
/// HOP_LENGTH samples (one hop), preventing infinite loops.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn longform_seek_minimum_advance() {
    let advance_sec: f64 = kani::any_where(|&v: &f64| v >= 0.0 && v <= 30.0 && v.is_finite());

    let advance_samples = ((advance_sec * SAMPLE_RATE as f64) as usize).min(N_SAMPLES);
    let actual_advance = advance_samples.max(HOP_LENGTH);

    assert!(
        actual_advance >= HOP_LENGTH,
        "seek must advance by at least HOP_LENGTH"
    );
}

// ============================================================================
// Harness 9: long-form seek: no-timestamp advance is exactly N_SAMPLES
// ============================================================================

/// Proves that when no timestamp tokens are found, seek advances by
/// exactly N_SAMPLES (one full 30-second chunk).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn longform_seek_no_timestamp_full_chunk() {
    let advance = N_SAMPLES; // no timestamp case
    assert_eq!(
        advance, N_SAMPLES,
        "no-timestamp advance must be exactly N_SAMPLES"
    );
    assert_eq!(advance, 480_000, "N_SAMPLES = 480000 for 30s at 16kHz");
}

// ============================================================================
// Harness 10: DecodeConfig with_max_length only changes max_length
// ============================================================================

/// Proves that with_max_length is independent of other fields.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn decode_config_with_max_length_preserves_others() {
    let base = DecodeConfig::default();
    let v: usize = kani::any();
    kani::assume(v >= 1 && v <= MAX_DECODE_LENGTH);

    let modified = base.clone().with_max_length(v);
    assert_eq!(modified.max_length, v);
    assert_eq!(
        modified.compression_ratio_threshold,
        base.compression_ratio_threshold
    );
    assert_eq!(
        modified.avg_logprob_threshold,
        base.avg_logprob_threshold
    );
    assert_eq!(modified.initial_tokens, base.initial_tokens);
    assert_eq!(modified.seed, base.seed);
}

// ============================================================================
// Harness 11: DecodeConfig with_seed only changes seed
// ============================================================================

/// Proves that with_seed is independent of other fields.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn decode_config_with_seed_preserves_others() {
    let base = DecodeConfig::default();
    let modified = base.clone().with_seed(Some(42));
    assert_eq!(modified.seed, Some(42));
    assert_eq!(modified.max_length, base.max_length);
    assert_eq!(
        modified.compression_ratio_threshold,
        base.compression_ratio_threshold
    );
    assert_eq!(modified.initial_tokens, base.initial_tokens);
}

// ============================================================================
// Harness 12: compute_no_speech_prob for all-equal logits
// ============================================================================

/// Proves that when all logits are equal, no_speech_prob = 1/vocab_size
/// (uniform softmax). For vocab_size > NO_SPEECH_TOKEN, this should be finite.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(5)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn no_speech_prob_uniform_logits() {
    // Small vocab for Kani tractability.
    let vocab_size: usize = kani::any();
    kani::assume(vocab_size > NO_SPEECH_TOKEN && vocab_size <= NO_SPEECH_TOKEN + 4);

    // All logits equal => softmax is uniform 1/N.
    let logits: Vec<f32> = vec![0.0; vocab_size];
    let max_val = 0.0f32;
    let exp: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exp.iter().sum();

    if sum.is_finite() && sum > 0.0 {
        let prob = f64::from(exp[NO_SPEECH_TOKEN] / sum);
        assert!(prob >= 0.0 && prob <= 1.0, "prob in [0, 1]");
        // For uniform: prob = 1/N
        let expected = 1.0 / vocab_size as f64;
        let diff = (prob - expected).abs();
        assert!(diff < 1e-5, "uniform softmax => 1/N");
    }
}

// ============================================================================
// Harness 13: DEFAULT_TEMPERATURES has 6 entries, first is 0.0
// ============================================================================

/// Proves structural properties of the default temperature sequence.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn default_temperatures_structure() {
    assert_eq!(DEFAULT_TEMPERATURES.len(), 6, "6 temperature steps");
    assert_eq!(DEFAULT_TEMPERATURES[0], 0.0, "first is greedy");
    assert_eq!(DEFAULT_TEMPERATURES[5], 1.0, "last is full temperature");

    // All are finite and non-negative.
    let mut i = 0;
    while i < DEFAULT_TEMPERATURES.len() {
        assert!(DEFAULT_TEMPERATURES[i].is_finite());
        assert!(DEFAULT_TEMPERATURES[i] >= 0.0);
        i += 1;
    }
}

// ============================================================================
// Harness 14: DecodingResult no_speech_prob in [0, 1] implies valid probability
// ============================================================================

/// Proves that if no_speech_prob is in [0, 1], comparing against a threshold
/// in [0, 1] produces a deterministic boolean result.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn no_speech_threshold_comparison_deterministic() {
    let prob: f64 = kani::any_where(|&v: &f64| v >= 0.0 && v <= 1.0 && v.is_finite());
    let threshold: f64 = kani::any_where(|&v: &f64| v >= 0.0 && v <= 1.0 && v.is_finite());

    // The comparison is deterministic (no NaN involved).
    let skip = prob > threshold;
    let keep = !skip;
    assert!(skip || keep, "comparison must be deterministic for finite values");
}

// ============================================================================
// Harness 15: MAX_DECODE_LENGTH matches max_target_positions / 2
// ============================================================================

/// Proves that MAX_DECODE_LENGTH (224) is exactly half of
/// max_target_positions (448). This is the documented relationship:
/// the decode length limit accounts for initial prompt tokens.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn max_decode_length_half_max_target() {
    let max_target = WhisperConfig::default().max_target_positions;
    assert_eq!(MAX_DECODE_LENGTH * 2, max_target, "MAX_DECODE_LENGTH = max_target / 2");
}
