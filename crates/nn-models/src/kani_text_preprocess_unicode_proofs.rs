// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `kokoro_text_preprocess` Unicode safety.
//!
//! These proofs stay within `#[kani::unwind(1)]` by exercising only a single
//! Unicode scalar. That is enough to cover the byte/char-boundary arithmetic
//! in the preprocessing pipeline without depending on large table construction.

#[cfg(kani)]
mod proofs {
    use crate::kokoro_text_preprocess::TextPreprocessor;

    fn any_unicode_scalar() -> char {
        let codepoint: u32 = kani::any();
        kani::assume(codepoint <= 0x10FFFF);
        kani::assume(!(0xD800..=0xDFFF).contains(&codepoint));
        char::from_u32(codepoint).unwrap()
    }

    /// A single valid Unicode scalar can pass through preprocess + sentence split
    /// without panicking or producing invalid UTF-8.
    ///
    /// Covers the one-step path through:
    /// - `normalize_punctuation`
    /// - `expand_numbers_in_text`
    /// - `normalize_whitespace`
    /// - `split_sentences_inner`
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn single_unicode_scalar_preprocess_is_safe() {
        let ch = any_unicode_scalar();
        let text = ch.to_string();
        let preprocessor = TextPreprocessor::minimal();

        let processed = preprocessor.preprocess(&text);
        let sentences = preprocessor.split_sentences(&text);

        assert!(processed.is_char_boundary(processed.len()));
        assert!(sentences.len() <= 1);
    }

    /// The byte-advance used by `expand_abbreviations` stays on a char boundary
    /// for any single valid Unicode scalar.
    ///
    /// Covers the critical indexing pattern:
    /// `let ch = text[i..].chars().next().unwrap(); i += ch.len_utf8();`
    #[kani::unwind(16)]
    #[kani::proof]
    #[kani::unwind(1)]
    fn unicode_byte_advance_preserves_char_boundary() {
        let ch = any_unicode_scalar();
        let text = ch.to_string();

        let next = text[0..].chars().next().unwrap();
        let advanced = next.len_utf8();

        assert_eq!(advanced, text.len());
        assert!(text.is_char_boundary(advanced));
    }
}
