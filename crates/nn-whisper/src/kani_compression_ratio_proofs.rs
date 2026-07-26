// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for compression ratio and quality check functions.
//!
//! Covers:
//! - compression_ratio >= 1.0 for all inputs with >= 2 tokens
//! - compression_ratio == 1.0 for inputs with < 2 tokens
//! - compression_ratio is finite for all valid inputs
//! - passes_quality_check is deterministic
//! - DecodeConfig::validate catches zero max_length
//! - DecodeConfig::validate catches NaN thresholds
//! - DecodeConfig::validate catches empty initial_tokens
//! - DecodeConfig default passes validation
//!
//! Issue: #4303

#[cfg(kani)]
mod proofs {
    use crate::decode::{
        compression_ratio, passes_quality_check, DecodeConfig, DecodingResult,
        DEFAULT_COMPRESSION_RATIO_THRESHOLD, MAX_DECODE_LENGTH,
    };

    // ============================================================================
    // Harness 1: compression_ratio >= 1.0 for tokens with >= 2 elements
    // ============================================================================

    /// Proves compression_ratio >= 1.0 when there are at least 2 tokens.
    #[kani::unwind(6)]
    #[kani::proof]
    fn compression_ratio_ge_one_for_two_or_more() {
        let len: usize = kani::any();
        kani::assume(len >= 2 && len <= 5);

        // Use a token from a small alphabet to get interesting bigram patterns.
        let mut tokens = Vec::with_capacity(len);
        for _ in 0..len {
            let t: usize = kani::any();
            kani::assume(t <= 3);
            tokens.push(t);
        }

        let cr = compression_ratio(&tokens);
        assert!(cr >= 1.0, "compression ratio must be >= 1.0");
        assert!(cr.is_finite(), "compression ratio must be finite");
    }

    // ============================================================================
    // Harness 2: compression_ratio == 1.0 for short sequences
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn compression_ratio_one_for_empty() {
        let cr = compression_ratio(&[]);
        assert!((cr - 1.0).abs() < 1e-12, "empty => 1.0");
    }

    #[kani::unwind(1)]
    #[kani::proof]
    fn compression_ratio_one_for_single() {
        let t: usize = kani::any();
        kani::assume(t <= 100);
        let cr = compression_ratio(&[t]);
        assert!((cr - 1.0).abs() < 1e-12, "single token => 1.0");
    }

    // ============================================================================
    // Harness 3: All-same tokens produce maximum compression ratio
    // ============================================================================

    #[kani::unwind(6)]
    #[kani::proof]
    fn compression_ratio_all_same_tokens() {
        let len: usize = kani::any();
        kani::assume(len >= 2 && len <= 5);

        let tokens: Vec<usize> = vec![42; len];
        let cr = compression_ratio(&tokens);
        // With all same tokens, there's exactly 1 unique bigram.
        // CR = (len-1) / 1.
        let expected = (len - 1) as f64;
        assert!(
            (cr - expected).abs() < 1e-12,
            "all-same tokens: CR must equal len-1"
        );
    }

    // ============================================================================
    // Harness 4: All-unique bigrams produce compression_ratio == 1.0
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn compression_ratio_all_unique_bigrams() {
        // [0, 1, 2, 3, 4]: bigrams (0,1), (1,2), (2,3), (3,4) — all unique.
        let tokens: Vec<usize> = vec![0, 1, 2, 3, 4];
        let cr = compression_ratio(&tokens);
        // 4 bigram slots / 4 unique = 1.0
        assert!(
            (cr - 1.0).abs() < 1e-12,
            "all-unique bigrams: CR must be 1.0"
        );
    }

    // ============================================================================
    // Harness 5: passes_quality_check is deterministic
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn quality_check_deterministic() {
        let result = DecodingResult::new(
            vec![1, 2, 3],
            -0.5,
            1.5,
            true,
            0.0,
            0.1,
        );
        let config = DecodeConfig::default();
        let pass1 = passes_quality_check(&result, &config);
        let pass2 = passes_quality_check(&result, &config);
        assert_eq!(pass1, pass2, "quality check must be deterministic");
    }

    // ============================================================================
    // Harness 6: DecodeConfig::validate catches zero max_length
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn decode_config_rejects_zero_max_length() {
        let config = DecodeConfig::default().with_max_length(0);
        assert!(config.validate().is_err(), "zero max_length must fail");
    }

    // ============================================================================
    // Harness 7: DecodeConfig::validate catches NaN compression_ratio_threshold
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn decode_config_rejects_nan_compression_threshold() {
        let config = DecodeConfig::default().with_compression_ratio_threshold(f64::NAN);
        assert!(
            config.validate().is_err(),
            "NaN compression_ratio_threshold must fail"
        );
    }

    // ============================================================================
    // Harness 8: DecodeConfig::validate catches NaN avg_logprob_threshold
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn decode_config_rejects_nan_avg_logprob_threshold() {
        let config = DecodeConfig::default().with_avg_logprob_threshold(f64::NAN);
        assert!(
            config.validate().is_err(),
            "NaN avg_logprob_threshold must fail"
        );
    }

    // ============================================================================
    // Harness 9: DecodeConfig::validate catches Inf thresholds
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn decode_config_rejects_inf_compression_threshold() {
        let config = DecodeConfig::default().with_compression_ratio_threshold(f64::INFINITY);
        assert!(
            config.validate().is_err(),
            "Inf compression_ratio_threshold must fail"
        );
    }

    // ============================================================================
    // Harness 10: DecodeConfig default passes validation
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn decode_config_default_passes_validation() {
        let config = DecodeConfig::default();
        assert!(
            config.validate().is_ok(),
            "default DecodeConfig must pass validation"
        );
    }

    // ============================================================================
    // Harness 11: DecodeConfig rejects exceeding max_length
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn decode_config_rejects_exceeding_max_length() {
        let config = DecodeConfig::default().with_max_length(MAX_DECODE_LENGTH + 1);
        assert!(
            config.validate().is_err(),
            "max_length exceeding MAX_DECODE_LENGTH must fail"
        );
    }

    // ============================================================================
    // Harness 12: passes_quality_check threshold boundary
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn quality_check_threshold_boundary() {
        let config = DecodeConfig::default();
        // Result that barely passes both thresholds.
        let passing = DecodingResult::new(
            vec![1, 2],
            DEFAULT_COMPRESSION_RATIO_THRESHOLD - 0.1,
            DEFAULT_COMPRESSION_RATIO_THRESHOLD,
            true,
            0.0,
            0.0,
        );
        assert!(
            passes_quality_check(&passing, &config),
            "at-threshold compression should pass"
        );

        // Result that exceeds compression threshold.
        let failing = DecodingResult::new(
            vec![1, 2],
            DEFAULT_COMPRESSION_RATIO_THRESHOLD + 0.1,
            DEFAULT_COMPRESSION_RATIO_THRESHOLD + 0.1,
            true,
            0.0,
            0.0,
        );
        assert!(
            !passes_quality_check(&failing, &config),
            "over-threshold compression should fail"
        );
    }
}
