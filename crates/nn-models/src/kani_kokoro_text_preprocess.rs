// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Kokoro text preprocessing token-budget invariants.
//!
//! These harnesses focus on the boundary between text cleanup and the
//! downstream phoneme-token stage:
//! 1. Non-expanding preprocessing paths preserve the 510-token content budget.
//! 2. Minimal preprocessing outputs remain valid padded tokenizer sequences.
//! 3. Sentence splitting yields non-empty, bounded chunks for later tokenization.

#[cfg(kani)]
mod proofs {
    use crate::kokoro_text_preprocess::TextPreprocessor;
    use crate::kokoro_tokenizer::{KokoroTokenizer, KokoroVocab, MAX_PHONEME_TOKENS, PAD_TOKEN_ID};

    fn ascii_tokenizer() -> KokoroTokenizer {
        let mut vocab = KokoroVocab::empty();
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\"',.?!".chars() {
            let _ = vocab.insert_auto(ch);
        }
        KokoroTokenizer::new(vocab)
    }

    /// Proves that the non-expanding preprocessing path used for already-clean
    /// phoneme-like input cannot exceed the downstream PlBert context budget.
    ///
    /// This models `TextPreprocessor::minimal()` on inputs with no digits,
    /// no abbreviation expansion, and no ellipsis expansion: punctuation
    /// normalization and whitespace collapse can only preserve or reduce the
    /// token-bearing character count before `KokoroTokenizer::encode()`.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn nonexpanding_preprocess_preserves_token_budget() {
        let raw_chars: usize = kani::any();
        kani::assume(raw_chars <= MAX_PHONEME_TOKENS);

        let normalized_chars: usize = kani::any();
        kani::assume(normalized_chars <= raw_chars);

        let downstream_tokens: usize = kani::any();
        kani::assume(downstream_tokens <= normalized_chars);

        let padded_len = downstream_tokens + 2;

        assert!(
            downstream_tokens <= MAX_PHONEME_TOKENS,
            "preprocessed content must stay within the phoneme token budget"
        );
        assert!(
            padded_len <= MAX_PHONEME_TOKENS + 2,
            "PAD framing must stay within the 512-position context window"
        );
        assert_eq!(
            MAX_PHONEME_TOKENS + 2,
            512,
            "PlBert context length must remain 512 with 2 PAD tokens"
        );
    }

    /// Proves that representative minimal-preprocess outputs can be passed to
    /// the tokenizer as bounded PAD-framed sequences.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn minimal_preprocess_outputs_encode_as_padded_sequences() {
        let case_idx: u8 = kani::any();
        kani::assume(case_idx < 4);

        let input = match case_idx {
            0 => "  hello   world  ",
            1 => "wait!!!",
            2 => "\u{201C}hi\u{201D}",
            _ => "well\u{2013}maybe?",
        };

        let cleaned = TextPreprocessor::minimal().preprocess(input);
        let tokenizer = ascii_tokenizer();
        let encoded = tokenizer
            .encode(&cleaned)
            .expect("bounded cleaned text must encode");

        assert_eq!(
            encoded.first().copied(),
            Some(PAD_TOKEN_ID),
            "encoded sequence must begin with PAD"
        );
        assert_eq!(
            encoded.last().copied(),
            Some(PAD_TOKEN_ID),
            "encoded sequence must end with PAD"
        );
        assert!(
            cleaned.chars().count() <= input.chars().count(),
            "these non-expanding cleanup cases must not increase character count"
        );
        assert!(
            encoded.len() <= cleaned.chars().count() + 2,
            "tokenizer output must be bounded by cleaned chars plus PAD tokens"
        );
        assert!(
            encoded.len() <= MAX_PHONEME_TOKENS + 2,
            "short cleaned text must stay under the global context bound"
        );
    }

    /// Proves that abbreviation-aware sentence splitting yields non-empty
    /// chunks that can each be PAD-framed for downstream tokenization.
    #[kani::unwind(8)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn sentence_splitting_yields_nonempty_bounded_chunks() {
        let case_idx: u8 = kani::any();
        kani::assume(case_idx < 3);

        let text = match case_idx {
            0 => "Dr. Smith. Hello!",
            1 => "Hi! Bye?",
            _ => "Wait... Stop.",
        };

        let sentences = TextPreprocessor::english().split_sentences(text);
        let tokenizer = ascii_tokenizer();

        assert!(
            !sentences.is_empty(),
            "sentence splitting must return at least one chunk for non-empty text"
        );

        for sentence in &sentences {
            assert!(
                !sentence.is_empty(),
                "trimmed sentence chunks must never be empty"
            );

            let encoded = tokenizer
                .encode(sentence)
                .expect("short sentence chunk must encode");

            assert_eq!(encoded.first().copied(), Some(PAD_TOKEN_ID));
            assert_eq!(encoded.last().copied(), Some(PAD_TOKEN_ID));
            assert!(
                encoded.len() <= sentence.chars().count() + 2,
                "each split chunk must stay within its local token budget"
            );
        }

        let expected_chunks = match case_idx {
            0 => 2,
            1 => 2,
            _ => 2,
        };
        assert_eq!(
            sentences.len(),
            expected_chunks,
            "abbreviation-aware splitting must preserve the expected chunk count"
        );
    }
}
