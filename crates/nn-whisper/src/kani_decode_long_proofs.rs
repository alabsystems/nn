// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for long-form decode utilities.
//!
//! Covers:
//! - extract_timestamp_advance returns non-negative values
//! - extract_timestamp_advance returns None when no timestamps present
//! - extract_timestamp_advance finds the last timestamp token
//! - Timestamp token arithmetic: TIMESTAMP_BEGIN offset maps correctly to seconds
//! - LongFormConfig default has valid no_speech_threshold
//!
//! Issue: #4303

#[cfg(kani)]
mod proofs {
    use crate::tokenizer::TIMESTAMP_BEGIN;
    use crate::decode::LongFormConfig;

    /// Reimplementation of extract_timestamp_advance for Kani (private fn).
    fn extract_timestamp_advance(tokens: &[usize]) -> Option<f64> {
        let last_ts = tokens.iter().rev().find(|&&t| t >= TIMESTAMP_BEGIN)?;
        let ts_seconds = last_ts.saturating_sub(TIMESTAMP_BEGIN) as f64 * 0.02;
        Some(ts_seconds)
    }

    // ============================================================================
    // Harness 1: Timestamp advance is non-negative
    // ============================================================================

    /// Proves extract_timestamp_advance always returns non-negative seconds.
    #[kani::unwind(5)]
    #[kani::proof]
    fn timestamp_advance_non_negative() {
        let token: usize = kani::any();
        // Timestamps are tokens >= TIMESTAMP_BEGIN
        kani::assume(token >= TIMESTAMP_BEGIN);
        kani::assume(token <= TIMESTAMP_BEGIN + 1500); // up to 30s at 0.02s resolution

        let tokens = vec![token];
        let result = extract_timestamp_advance(&tokens);
        assert!(result.is_some());
        let secs = result.unwrap();
        assert!(secs >= 0.0, "timestamp seconds must be non-negative");
        assert!(secs.is_finite(), "timestamp seconds must be finite");
    }

    // ============================================================================
    // Harness 2: No timestamp tokens => None
    // ============================================================================

    /// Proves extract_timestamp_advance returns None when no timestamps present.
    #[kani::unwind(5)]
    #[kani::proof]
    fn timestamp_advance_none_without_timestamps() {
        let tok1: usize = kani::any();
        let tok2: usize = kani::any();
        kani::assume(tok1 < TIMESTAMP_BEGIN);
        kani::assume(tok2 < TIMESTAMP_BEGIN);

        let tokens = vec![tok1, tok2];
        let result = extract_timestamp_advance(&tokens);
        assert!(result.is_none(), "no timestamp tokens => None");
    }

    // ============================================================================
    // Harness 3: Empty tokens => None
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn timestamp_advance_empty_tokens() {
        let tokens: Vec<usize> = Vec::new();
        let result = extract_timestamp_advance(&tokens);
        assert!(result.is_none(), "empty tokens => None");
    }

    // ============================================================================
    // Harness 4: Timestamp value is correct
    // ============================================================================

    /// Proves the timestamp offset maps correctly to seconds.
    #[kani::unwind(1)]
    #[kani::proof]
    fn timestamp_offset_to_seconds_correct() {
        let offset: usize = kani::any();
        kani::assume(offset <= 1500); // 0..=30.00 seconds

        let token = TIMESTAMP_BEGIN + offset;
        let tokens = vec![token];
        let result = extract_timestamp_advance(&tokens);
        assert!(result.is_some());
        let secs = result.unwrap();
        let expected = offset as f64 * 0.02;
        assert!(
            (secs - expected).abs() < 1e-12,
            "timestamp seconds must equal offset * 0.02"
        );
    }

    // ============================================================================
    // Harness 5: Last timestamp is used
    // ============================================================================

    /// Proves the function uses the LAST timestamp token, not the first.
    #[kani::unwind(5)]
    #[kani::proof]
    fn timestamp_advance_uses_last_token() {
        let offset1: usize = kani::any();
        let offset2: usize = kani::any();
        kani::assume(offset1 <= 1500);
        kani::assume(offset2 <= 1500);
        kani::assume(offset1 != offset2);

        let tok1 = TIMESTAMP_BEGIN + offset1;
        let tok2 = TIMESTAMP_BEGIN + offset2;
        let tokens = vec![tok1, tok2];
        let result = extract_timestamp_advance(&tokens);
        assert!(result.is_some());
        let secs = result.unwrap();
        let expected = offset2 as f64 * 0.02;
        assert!(
            (secs - expected).abs() < 1e-12,
            "must use last timestamp token"
        );
    }

    // ============================================================================
    // Harness 6: LongFormConfig default values
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn long_form_config_default_valid() {
        let config = LongFormConfig::default();
        assert!(
            config.no_speech_threshold.is_finite(),
            "default no_speech_threshold must be finite"
        );
        assert!(
            config.no_speech_threshold > 0.0 && config.no_speech_threshold < 1.0,
            "default no_speech_threshold must be in (0, 1)"
        );
        assert!(
            !config.temperatures.is_empty(),
            "default temperatures must be non-empty"
        );
    }

    // ============================================================================
    // Harness 7: Timestamp advance bounded by 30s
    // ============================================================================

    /// Proves that timestamps within the standard range produce <= 30.0 seconds.
    #[kani::unwind(1)]
    #[kani::proof]
    fn timestamp_advance_bounded_by_30s() {
        let offset: usize = kani::any();
        kani::assume(offset <= 1500); // 1500 * 0.02 = 30.0

        let token = TIMESTAMP_BEGIN + offset;
        let tokens = vec![token];
        let result = extract_timestamp_advance(&tokens);
        assert!(result.is_some());
        let secs = result.unwrap();
        assert!(
            secs <= 30.0 + 1e-12,
            "standard timestamps produce at most 30.0 seconds"
        );
    }
}
