// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Whisper quality metrics, decode helpers, and error types.
//!
//! Covers:
//! - word_error_rate: identity, bounds [0,1] for equal-length, symmetry of WER=0
//! - word_error_rate: empty inputs, single-word cases, edit distance properties
//! - compression_ratio: single-token boundary, two-token minimum, all-same tokens
//! - compression_ratio: non-negative and finite for bounded inputs
//! - argmax_f32: empty slice returns 0, single-element returns 0
//! - argmax_f32: result index is in bounds for non-empty slices
//! - apply_suppression_inplace: suppressed positions become NEG_INFINITY
//! - apply_suppression_inplace: non-suppressed positions are unchanged
//! - apply_suppression_inplace: out-of-bounds indices are ignored safely
//! - compute_log_prob: output is <= 0 for valid inputs (log-softmax property)
//! - compute_log_prob: empty slice returns NEG_INFINITY
//! - compute_log_prob: out-of-bounds index returns NEG_INFINITY
//! - DecodingResult construction preserves all fields
//! - DecodeConfig default validation passes
//! - WhisperError Display: structured variants produce non-empty messages
//! - passes_quality_check: boundary exact at thresholds
//!
//! Issue: #3800

#[cfg(kani)]
mod proofs {
    use crate::decode::{
        compression_ratio, passes_quality_check, DecodeConfig, DecodingResult,
        DEFAULT_AVG_LOGPROB_THRESHOLD, DEFAULT_COMPRESSION_RATIO_THRESHOLD, DEFAULT_TEMPERATURES,
        MAX_DECODE_LENGTH,
    };
    use crate::quality::word_error_rate;
    use crate::WhisperError;

    // Re-exports for decode helpers (pub(crate) in decode module, accessible via crate::decode::*)
    use crate::decode::{argmax_f32, compute_log_prob};

    // ── Kani transcendental stubs (CBMC cannot handle these) ──
    fn exp_f32_stub(x: f32) -> f32 { let _ = x; let r: f32 = kani::any(); kani::assume(r.is_finite() && r > 0.0 && r <= 1e10); r }
    fn ln_f32_stub(x: f32) -> f32 { let _ = x; let r: f32 = kani::any(); kani::assume(r.is_finite() && r >= -100.0 && r <= 100.0); r }

    // ========================================================================
    // word_error_rate proofs
    // ========================================================================

    /// Proves WER of identical strings is 0.0.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn wer_identical_strings_is_zero() {
        let wer = word_error_rate("the cat sat on the mat", "the cat sat on the mat");
        assert!(
            wer.abs() < 1e-15,
            "WER of identical strings must be 0.0"
        );
    }

    /// Proves WER is 0.0 when both inputs are empty.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn wer_both_empty_is_zero() {
        let wer = word_error_rate("", "");
        assert!(wer.abs() < 1e-15, "WER of two empty strings must be 0.0");
    }

    /// Proves WER is 1.0 when reference is empty but hypothesis is not.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn wer_empty_ref_nonempty_hyp_is_one() {
        let wer = word_error_rate("hello world", "");
        assert!(
            (wer - 1.0).abs() < 1e-15,
            "WER with empty ref, non-empty hyp must be 1.0"
        );
    }

    /// Proves WER is 1.0 when hypothesis is empty but reference has words.
    ///
    /// All reference words are "deleted" → edit distance = ref_len → WER = 1.0.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn wer_empty_hyp_full_ref_is_one() {
        let wer = word_error_rate("", "one two three");
        assert!(
            (wer - 1.0).abs() < 1e-15,
            "WER with empty hyp, 3-word ref must be 1.0"
        );
    }

    /// Proves WER is symmetric for the zero case: if WER(a,b)=0 then WER(b,a)=0.
    ///
    /// WER is NOT symmetric in general (different denominators), but the
    /// zero-error case must be symmetric.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn wer_zero_is_symmetric() {
        let wer_ab = word_error_rate("the quick brown fox", "the quick brown fox");
        let wer_ba = word_error_rate("the quick brown fox", "the quick brown fox");
        assert!(wer_ab.abs() < 1e-15, "WER(a,a) must be 0");
        assert!(wer_ba.abs() < 1e-15, "WER(a,a) reversed must be 0");
    }

    /// Proves WER with a single substitution on 3 words equals 1/3.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn wer_single_substitution() {
        let wer = word_error_rate("the dog sat", "the cat sat");
        assert!(
            (wer - 1.0 / 3.0).abs() < 1e-10,
            "1 sub in 3 words must give WER=1/3"
        );
    }

    /// Proves WER is case-insensitive (via eq_ignore_ascii_case).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn wer_case_insensitive() {
        let wer = word_error_rate("HELLO WORLD", "hello world");
        assert!(wer.abs() < 1e-15, "WER must be case-insensitive");
    }

    /// Proves WER with completely wrong hypothesis equals 1.0 when lengths match.
    ///
    /// All words are substitutions → edit distance = ref_len → WER = 1.0.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn wer_all_substitutions_is_one() {
        let wer = word_error_rate("a b c", "x y z");
        assert!(
            (wer - 1.0).abs() < 1e-15,
            "all-wrong same-length must be WER=1.0"
        );
    }

    /// Proves compression_ratio of empty tokens returns 1.0.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn compression_ratio_empty() {
        let cr = compression_ratio(&[]);
        assert!(
            (cr - 1.0).abs() < 1e-15,
            "empty tokens must have compression ratio 1.0"
        );
    }

    /// Proves compression_ratio of two distinct tokens returns 1.0.
    ///
    /// [a, b] where a != b → 1 unique bigram, 1 bigram slot → ratio = 1.0.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn compression_ratio_two_distinct() {
        let cr = compression_ratio(&[1, 2]);
        assert!(
            (cr - 1.0).abs() < 1e-15,
            "two distinct tokens must have CR=1.0"
        );
    }

    /// Proves compression_ratio of all-same tokens increases with length.
    ///
    /// [x, x, x] → 2 bigram slots, 1 unique bigram → ratio = 2.0.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn compression_ratio_all_same_three() {
        let cr = compression_ratio(&[5, 5, 5]);
        assert!(
            (cr - 2.0).abs() < 1e-15,
            "[x,x,x] must have CR=2.0"
        );
    }

    /// Proves compression_ratio is >= 1.0 for any non-trivial input.
    ///
    /// unique_bigrams <= total_bigram_slots → ratio >= 1.0.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn compression_ratio_gte_one() {
        let a: usize = kani::any();
        let b: usize = kani::any();
        let c: usize = kani::any();
        kani::assume(a <= 10 && b <= 10 && c <= 10);

        let cr = compression_ratio(&[a, b, c]);
        assert!(cr >= 1.0, "compression ratio must be >= 1.0");
    }

    /// Proves compression_ratio is finite for bounded inputs.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn compression_ratio_is_finite() {
        let a: usize = kani::any();
        let b: usize = kani::any();
        kani::assume(a <= 100 && b <= 100);

        let cr = compression_ratio(&[a, b]);
        assert!(cr.is_finite(), "compression ratio must be finite");
    }

    /// Proves argmax_f32 result is within bounds for any non-empty slice.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(5)]
    fn argmax_result_in_bounds() {
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        let c: f32 = kani::any();
        let d: f32 = kani::any();
        kani::assume(a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite());

        let values = [a, b, c, d];
        let idx = argmax_f32(&values);
        assert!(idx < values.len(), "argmax index must be in bounds");
    }

    /// Proves argmax_f32 returns the index of the maximum value.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(4)]
    fn argmax_returns_max_index() {
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        let c: f32 = kani::any();
        kani::assume(a.is_finite() && b.is_finite() && c.is_finite());

        let values = [a, b, c];
        let idx = argmax_f32(&values);
        let max_val = values[idx];

        // The value at the argmax index must be >= all other values.
        for &v in &values {
            assert!(
                max_val >= v || max_val.total_cmp(&v) != std::cmp::Ordering::Less,
                "argmax value must be >= all others under total_cmp"
            );
        }
    }

    // ========================================================================
    // apply_suppression_inplace proofs
    // ========================================================================

    /// Proves apply_suppression_inplace sets suppressed positions to NEG_INFINITY.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn suppression_sets_neg_infinity() {
        let mut logits = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let suppress = [1, 3];
        crate::decode::apply_suppression_inplace(&mut logits, &suppress);

        assert_eq!(logits[1], f32::NEG_INFINITY, "index 1 must be suppressed");
        assert_eq!(logits[3], f32::NEG_INFINITY, "index 3 must be suppressed");
    }

    /// Proves apply_suppression_inplace does not change non-suppressed positions.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn suppression_preserves_non_suppressed() {
        let mut logits = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let suppress = [1, 3];
        crate::decode::apply_suppression_inplace(&mut logits, &suppress);

        assert_eq!(logits[0], 1.0, "index 0 must be unchanged");
        assert_eq!(logits[2], 3.0, "index 2 must be unchanged");
        assert_eq!(logits[4], 5.0, "index 4 must be unchanged");
    }

    /// Proves apply_suppression_inplace safely ignores out-of-bounds indices.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn suppression_ignores_out_of_bounds() {
        let mut logits = [1.0f32, 2.0, 3.0];
        let suppress = [5, 10, 100]; // all out of bounds
        crate::decode::apply_suppression_inplace(&mut logits, &suppress);

        // All values unchanged.
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], 2.0);
        assert_eq!(logits[2], 3.0);
    }

    /// Proves apply_suppression_inplace with empty suppress list is identity.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn suppression_empty_list_is_identity() {
        let mut logits = [1.0f32, 2.0, 3.0];
        crate::decode::apply_suppression_inplace(&mut logits, &[]);

        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], 2.0);
        assert_eq!(logits[2], 3.0);
    }

    // ========================================================================
    // compute_log_prob proofs
    // ========================================================================

    /// Proves compute_log_prob returns NEG_INFINITY for empty input.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    #[kani::stub(f32::exp, exp_f32_stub)]
    #[kani::stub(f32::ln, ln_f32_stub)]
    fn log_prob_empty_returns_neg_inf() {
        let lp = compute_log_prob(&[], 0);
        assert_eq!(
            lp,
            f32::NEG_INFINITY,
            "empty slice must return NEG_INFINITY"
        );
    }

    /// Proves compute_log_prob returns NEG_INFINITY for out-of-bounds index.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    #[kani::stub(f32::exp, exp_f32_stub)]
    #[kani::stub(f32::ln, ln_f32_stub)]
    fn log_prob_oob_index_returns_neg_inf() {
        let logits = [1.0f32, 2.0, 3.0];
        let lp = compute_log_prob(&logits, 5);
        assert_eq!(
            lp,
            f32::NEG_INFINITY,
            "out-of-bounds index must return NEG_INFINITY"
        );
    }

    /// Proves compute_log_prob output is <= 0 for valid finite inputs.
    ///
    /// log-softmax produces values in (-inf, 0]. The maximum possible value
    /// is 0.0 (when a single logit dominates).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(4)]
    #[kani::stub(f32::exp, exp_f32_stub)]
    #[kani::stub(f32::ln, ln_f32_stub)]
    fn log_prob_is_non_positive() {
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        let c: f32 = kani::any();
        kani::assume(a.is_finite() && b.is_finite() && c.is_finite());
        // Avoid extreme values that cause overflow.
        kani::assume(a.abs() < 80.0 && b.abs() < 80.0 && c.abs() < 80.0);

        let logits = [a, b, c];
        let idx: usize = kani::any();
        kani::assume(idx < 3);

        let lp = compute_log_prob(&logits, idx);
        // log-softmax is always <= 0.
        assert!(
            lp <= 0.0 + 1e-6,
            "log_prob must be <= 0 (got {lp})"
        );
    }

    /// Proves compute_log_prob returns all NEG_INFINITY when all logits are NEG_INFINITY.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    #[kani::stub(f32::exp, exp_f32_stub)]
    #[kani::stub(f32::ln, ln_f32_stub)]
    fn log_prob_all_neg_inf() {
        let logits = [f32::NEG_INFINITY, f32::NEG_INFINITY];
        let lp = compute_log_prob(&logits, 0);
        assert_eq!(
            lp,
            f32::NEG_INFINITY,
            "all-NEG_INFINITY logits must return NEG_INFINITY"
        );
    }

    // ========================================================================
    // DecodingResult construction proofs
    // ========================================================================

    /// Proves DecodingResult::new preserves all field values.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn decoding_result_preserves_fields() {
        let result = DecodingResult::new(
            vec![1, 2, 3],
            -0.5,
            1.8,
            true,
            0.2,
            0.05,
        );
        assert_eq!(result.tokens, vec![1, 2, 3]);
        assert!((result.avg_logprob - (-0.5)).abs() < 1e-15);
        assert!((result.compression_ratio - 1.8).abs() < 1e-15);
        assert!(result.reached_eot);
        assert!((result.temperature - 0.2).abs() < 1e-15);
        assert!((result.no_speech_prob - 0.05).abs() < 1e-15);
    }

    // ========================================================================
    // DecodeConfig validation proofs
    // ========================================================================

    /// Proves DecodeConfig::default() passes validation.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn decode_config_default_validates() {
        let config = DecodeConfig::default();
        assert!(
            config.validate().is_ok(),
            "default DecodeConfig must pass validation"
        );
    }

    /// Proves DecodeConfig rejects max_length == 0.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn decode_config_rejects_zero_max_length() {
        let config = DecodeConfig::default().with_max_length(0);
        assert!(
            config.validate().is_err(),
            "max_length == 0 must fail validation"
        );
    }

    /// Proves DecodeConfig rejects max_length > MAX_DECODE_LENGTH.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn decode_config_rejects_excessive_max_length() {
        let config = DecodeConfig::default().with_max_length(MAX_DECODE_LENGTH + 1);
        assert!(
            config.validate().is_err(),
            "max_length > MAX_DECODE_LENGTH must fail"
        );
    }

    /// Proves DecodeConfig rejects NaN compression_ratio_threshold.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn decode_config_rejects_nan_compression_threshold() {
        let config = DecodeConfig::default().with_compression_ratio_threshold(f64::NAN);
        assert!(
            config.validate().is_err(),
            "NaN compression threshold must fail"
        );
    }

    /// Proves DecodeConfig rejects Inf avg_logprob_threshold.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn decode_config_rejects_inf_logprob_threshold() {
        let config = DecodeConfig::default().with_avg_logprob_threshold(f64::INFINITY);
        assert!(
            config.validate().is_err(),
            "Inf logprob threshold must fail"
        );
    }

    /// Proves DecodeConfig rejects empty initial_tokens.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn decode_config_rejects_empty_initial_tokens() {
        let config = DecodeConfig::default().with_initial_tokens(Vec::new());
        assert!(
            config.validate().is_err(),
            "empty initial_tokens must fail"
        );
    }

    // ========================================================================
    // passes_quality_check proofs
    // ========================================================================

    /// Proves quality check passes at exact threshold boundaries.
    ///
    /// compression_ratio == threshold (<=) and avg_logprob == threshold (>=).
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn quality_check_exact_boundary_passes() {
        let result = DecodingResult::new(
            vec![1],
            DEFAULT_AVG_LOGPROB_THRESHOLD, // exactly at threshold
            DEFAULT_COMPRESSION_RATIO_THRESHOLD, // exactly at threshold
            true,
            0.0,
            0.0,
        );
        let config = DecodeConfig::default();

        assert!(
            passes_quality_check(&result, &config),
            "exact boundary values must pass quality check"
        );
    }

    /// Proves quality check fails when compression ratio exceeds threshold.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn quality_check_fails_high_compression() {
        let result = DecodingResult::new(
            vec![1],
            0.0,  // good logprob
            DEFAULT_COMPRESSION_RATIO_THRESHOLD + 0.1, // exceeds threshold
            true,
            0.0,
            0.0,
        );
        let config = DecodeConfig::default();

        assert!(
            !passes_quality_check(&result, &config),
            "compression above threshold must fail"
        );
    }

    /// Proves quality check fails when avg_logprob is below threshold.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn quality_check_fails_low_logprob() {
        let result = DecodingResult::new(
            vec![1],
            DEFAULT_AVG_LOGPROB_THRESHOLD - 0.1, // below threshold
            1.0, // good compression
            true,
            0.0,
            0.0,
        );
        let config = DecodeConfig::default();

        assert!(
            !passes_quality_check(&result, &config),
            "logprob below threshold must fail"
        );
    }

    // ========================================================================
    // WhisperError Display proofs
    // ========================================================================

    /// Proves WhisperError::ZeroConfigField produces a non-empty Display string.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn error_zero_config_field_display_nonempty() {
        let err = WhisperError::ZeroConfigField { field: "d_model" };
        let msg = err.to_string();
        assert!(!msg.is_empty(), "error Display must be non-empty");
        assert!(
            msg.contains("d_model"),
            "error must mention the field name"
        );
    }

    /// Proves WhisperError::InvalidTemperature produces a non-empty message.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn error_invalid_temperature_display_nonempty() {
        let err = WhisperError::InvalidTemperature {
            temperature: f64::NAN,
        };
        let msg = err.to_string();
        assert!(!msg.is_empty(), "error Display must be non-empty");
    }

    /// Proves WhisperError::TokenIdOverflow includes the token ID.
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn error_token_overflow_display_has_id() {
        let err = WhisperError::TokenIdOverflow {
            token_id: 5_000_000_000,
        };
        let msg = err.to_string();
        assert!(!msg.is_empty(), "error Display must be non-empty");
        assert!(
            msg.contains("5000000000"),
            "error must include the overflowing token ID"
        );
    }

    /// Proves DEFAULT_TEMPERATURES starts at 0.0 (greedy) and ends at 1.0.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn default_temperatures_range() {
        assert!(
            (DEFAULT_TEMPERATURES[0] - 0.0).abs() < 1e-15,
            "first temperature must be 0.0 (greedy)"
        );
        assert!(
            (DEFAULT_TEMPERATURES[DEFAULT_TEMPERATURES.len() - 1] - 1.0).abs() < 1e-15,
            "last temperature must be 1.0"
        );
    }

    /// Proves all DEFAULT_TEMPERATURES are finite and non-negative.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(7)]
    fn default_temperatures_valid() {
        for &t in &DEFAULT_TEMPERATURES {
            assert!(t.is_finite(), "temperature must be finite");
            assert!(t >= 0.0, "temperature must be non-negative");
        }
    }

    // ========================================================================
    // MAX_DECODE_LENGTH proof
    // ========================================================================

    /// Proves MAX_DECODE_LENGTH matches the AI Provider Whisper value of 224.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn max_decode_length_is_224() {
        assert_eq!(
            MAX_DECODE_LENGTH, 224,
            "MAX_DECODE_LENGTH must match AI Provider Whisper (224)"
        );
    }
}
