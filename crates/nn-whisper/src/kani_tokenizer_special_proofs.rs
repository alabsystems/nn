// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for tokenizer special token handling and timestamp logic.
//!
//! Covers:
//! - is_special for known special tokens
//! - is_special false for regular tokens
//! - is_timestamp for TIMESTAMP_BEGIN and above
//! - timestamp_value non-negative and finite
//! - timestamp_value None for non-timestamp tokens
//! - Language token range constants are consistent
//! - EOT < SOT < language range ordering
//!
//! Issue: #4303

#[cfg(kani)]
mod proofs {
    use crate::tokenizer::{
        WhisperTokenizer, EOT_TOKEN, LANGUAGE_TOKEN_END, LANGUAGE_TOKEN_START,
        NO_SPEECH_TOKEN, NO_TIMESTAMPS_TOKEN, SOT_TOKEN, TIMESTAMP_BEGIN,
    };

    /// Build a minimal tokenizer for testing (decode-only).
    fn minimal_tokenizer() -> WhisperTokenizer {
        // Need a vocab that covers special token IDs.
        // Build a minimal vocab JSON with entries up to TIMESTAMP_BEGIN + 10.
        let mut entries = Vec::new();
        for i in 0..=(TIMESTAMP_BEGIN + 10) {
            entries.push(format!(r#""tok{i}": {i}"#));
        }
        let json = format!("{{{}}}", entries.join(","));
        WhisperTokenizer::from_vocab_str(&json).expect("build minimal tokenizer")
    }

    // ============================================================================
    // Harness 1: Known special tokens are special
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn known_special_tokens_are_special() {
        let tok = minimal_tokenizer();
        assert!(tok.is_special(EOT_TOKEN), "EOT must be special");
        assert!(tok.is_special(SOT_TOKEN), "SOT must be special");
        assert!(
            tok.is_special(NO_SPEECH_TOKEN),
            "NO_SPEECH must be special"
        );
        assert!(
            tok.is_special(NO_TIMESTAMPS_TOKEN),
            "NO_TIMESTAMPS must be special"
        );
        assert!(
            tok.is_special(LANGUAGE_TOKEN_START),
            "language start must be special"
        );
        assert!(
            tok.is_special(LANGUAGE_TOKEN_END),
            "language end must be special"
        );
        assert!(
            tok.is_special(TIMESTAMP_BEGIN),
            "TIMESTAMP_BEGIN must be special"
        );
    }

    // ============================================================================
    // Harness 2: Regular tokens are not special
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn regular_tokens_not_special() {
        let tok = minimal_tokenizer();
        // Regular tokens have IDs below EOT_TOKEN (50257).
        assert!(!tok.is_special(0), "token 0 must not be special");
        assert!(!tok.is_special(100), "token 100 must not be special");
        assert!(
            !tok.is_special(50256),
            "token 50256 (just below EOT) must not be special"
        );
    }

    // ============================================================================
    // Harness 3: Timestamp tokens are correctly identified
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn timestamp_tokens_identified() {
        let tok = minimal_tokenizer();
        assert!(
            tok.is_timestamp(TIMESTAMP_BEGIN),
            "TIMESTAMP_BEGIN is a timestamp"
        );
        assert!(
            tok.is_timestamp(TIMESTAMP_BEGIN + 1),
            "TIMESTAMP_BEGIN + 1 is a timestamp"
        );
        assert!(
            !tok.is_timestamp(TIMESTAMP_BEGIN - 1),
            "TIMESTAMP_BEGIN - 1 is not a timestamp"
        );
        assert!(!tok.is_timestamp(0), "0 is not a timestamp");
    }

    // ============================================================================
    // Harness 4: timestamp_value non-negative and finite
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn timestamp_value_non_negative_finite() {
        let tok = minimal_tokenizer();
        let offset: usize = kani::any();
        kani::assume(offset <= 1500);

        let token_id = TIMESTAMP_BEGIN + offset;
        let val = tok.timestamp_value(token_id);
        assert!(val.is_some(), "timestamp token must have a value");
        let secs = val.unwrap();
        assert!(secs >= 0.0, "timestamp value must be non-negative");
        assert!(secs.is_finite(), "timestamp value must be finite");
    }

    // ============================================================================
    // Harness 5: timestamp_value None for non-timestamp tokens
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn timestamp_value_none_for_non_timestamp() {
        let tok = minimal_tokenizer();
        // Any token below TIMESTAMP_BEGIN should return None.
        let token_id: usize = kani::any();
        kani::assume(token_id < TIMESTAMP_BEGIN);

        let val = tok.timestamp_value(token_id);
        assert!(
            val.is_none(),
            "non-timestamp token must have no timestamp value"
        );
    }

    // ============================================================================
    // Harness 6: Token constant ordering
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn token_constant_ordering() {
        assert!(EOT_TOKEN < SOT_TOKEN, "EOT < SOT");
        assert!(SOT_TOKEN < LANGUAGE_TOKEN_START, "SOT < LANGUAGE_START");
        assert!(
            LANGUAGE_TOKEN_START <= LANGUAGE_TOKEN_END,
            "LANGUAGE_START <= LANGUAGE_END"
        );
        assert!(
            LANGUAGE_TOKEN_END < NO_TIMESTAMPS_TOKEN,
            "LANGUAGE_END < NO_TIMESTAMPS"
        );
        assert!(
            NO_TIMESTAMPS_TOKEN < TIMESTAMP_BEGIN,
            "NO_TIMESTAMPS < TIMESTAMP_BEGIN"
        );
    }

    // ============================================================================
    // Harness 7: Language token range has 100 tokens
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn language_token_range_size() {
        let count = LANGUAGE_TOKEN_END - LANGUAGE_TOKEN_START + 1;
        assert_eq!(count, 100, "Whisper has exactly 100 language tokens");
    }

    // ============================================================================
    // Harness 8: Timestamp resolution is 0.02s (50 per second)
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn timestamp_resolution_0_02s() {
        let tok = minimal_tokenizer();
        // Token at offset 50 should be exactly 1.0 second.
        let val = tok.timestamp_value(TIMESTAMP_BEGIN + 50).unwrap();
        assert!(
            (val - 1.0).abs() < 1e-12,
            "50 offsets * 0.02s = 1.0s"
        );
        // Token at offset 1500 should be exactly 30.0 seconds.
        let val30 = tok.timestamp_value(TIMESTAMP_BEGIN + 1500).unwrap();
        assert!(
            (val30 - 30.0).abs() < 1e-12,
            "1500 offsets * 0.02s = 30.0s"
        );
    }

    // ============================================================================
    // Harness 9: Vocab size is correct
    // ============================================================================

    #[kani::unwind(1)]
    #[kani::proof]
    fn minimal_tokenizer_vocab_size() {
        let tok = minimal_tokenizer();
        assert!(
            tok.vocab_size() > TIMESTAMP_BEGIN,
            "vocab must cover timestamp tokens"
        );
    }
}
