// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Whisper tokenizer bounds and special tokens.
//!
//! Covers:
//! - Regular token IDs must remain below `vocab_size`
//! - The special-token boundary is exact at `EOT_TOKEN`
//! - Special tokens are skipped during decode even when they exceed `vocab_size`
//! - Timestamp tokens start exactly at `TIMESTAMP_BEGIN`
//!
//! Issue: #3724

#[cfg(kani)]
mod proofs {
    use crate::tokenizer::{
        EOT_TOKEN, LANGUAGE_TOKEN_END, LANGUAGE_TOKEN_START, NO_SPEECH_TOKEN, NO_TIMESTAMPS_TOKEN,
        SOT_TOKEN, TIMESTAMP_BEGIN,
    };
    use crate::WhisperTokenizer;

    fn tiny_tokenizer() -> WhisperTokenizer {
        WhisperTokenizer::from_vocab_str(r#"{"A":0,"B":1," ":2}"#).expect("tiny vocab is valid")
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn regular_token_ids_must_fit_vocab() {
        let token_id: usize = kani::any();
        let tokenizer = tiny_tokenizer();

        kani::assume(token_id >= tokenizer.vocab_size());
        kani::assume(token_id < EOT_TOKEN);

        let result = tokenizer.decode(&[token_id]);
        assert!(
            result.is_err(),
            "non-special token IDs >= vocab_size must fail decode"
        );
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn special_token_boundary_is_exact() {
        let token_id: usize = kani::any();
        let tokenizer = tiny_tokenizer();

        kani::assume(token_id < EOT_TOKEN);

        assert!(
            !tokenizer.is_special(token_id),
            "IDs below EOT_TOKEN must remain regular tokens"
        );
        assert!(
            tokenizer.is_special(EOT_TOKEN),
            "EOT_TOKEN starts the special range"
        );
        assert!(tokenizer.is_special(SOT_TOKEN), "SOT_TOKEN must be special");
        assert!(
            tokenizer.is_special(LANGUAGE_TOKEN_START),
            "language tags must be special"
        );
        assert!(
            tokenizer.is_special(LANGUAGE_TOKEN_END),
            "the end of the language-token range must be special"
        );
        assert!(
            tokenizer.is_special(NO_SPEECH_TOKEN),
            "NO_SPEECH_TOKEN must be special"
        );
        assert!(
            tokenizer.is_special(NO_TIMESTAMPS_TOKEN),
            "NO_TIMESTAMPS_TOKEN must be special"
        );
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn decode_skips_special_tokens_even_above_vocab_size() {
        let which_special: u8 = kani::any();
        let tokenizer = tiny_tokenizer();

        kani::assume(which_special < 4);

        let special = match which_special {
            0 => SOT_TOKEN,
            1 => EOT_TOKEN,
            2 => NO_SPEECH_TOKEN,
            _ => TIMESTAMP_BEGIN,
        };

        let decoded = tokenizer
            .decode(&[special, 0, special])
            .expect("special tokens should be skipped before range checks");

        assert!(tokenizer.is_special(special));
        assert_eq!(
            decoded, "A",
            "only regular token payload should survive decode"
        );
    }

    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn timestamp_tokens_begin_at_exact_boundary() {
        let offset: usize = kani::any();
        let tokenizer = tiny_tokenizer();

        kani::assume(offset < 4);

        assert!(
            tokenizer.timestamp_value(TIMESTAMP_BEGIN - 1).is_none(),
            "the token immediately before TIMESTAMP_BEGIN is not a timestamp"
        );

        let timestamp = tokenizer
            .timestamp_value(TIMESTAMP_BEGIN + offset)
            .expect("TIMESTAMP_BEGIN and later tokens must decode to timestamps");
        let next_timestamp = tokenizer
            .timestamp_value(TIMESTAMP_BEGIN + offset + 1)
            .expect("next timestamp token must decode");

        assert!(timestamp >= 0.0, "timestamps must be non-negative");
        assert!(
            next_timestamp > timestamp,
            "timestamp token values must increase monotonically"
        );
    }
}
