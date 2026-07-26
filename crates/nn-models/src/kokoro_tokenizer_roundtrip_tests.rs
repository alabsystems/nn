// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro tokenizer and phoneme vocabulary integration tests (#4186).
//!
//! Tests phoneme vocabulary mapping, text-to-phoneme token conversion,
//! token sequence padding/truncation, special tokens (PAD), encode/decode
//! roundtrip, IPA phoneme encoding, chunking behavior, and vocabulary
//! extension edge cases.

use crate::kokoro_tokenizer::{KokoroTokenizer, KokoroVocab, MAX_PHONEME_TOKENS, PAD_TOKEN_ID};

// =============================================================================
// 1. Phoneme vocabulary mapping
// =============================================================================

#[test]
fn test_vocab_default_covers_all_ipa_vowels() {
    // Verify key IPA vowels from the Kokoro vocabulary are present.
    let vocab = KokoroVocab::kokoro_default();
    let ipa_vowels = [
        ('\u{0251}', 69),  // open back unrounded (cot)
        ('\u{00E6}', 72),  // near-open front unrounded (cat)
        ('\u{0254}', 76),  // open-mid back rounded (thought)
        ('\u{0259}', 83),  // mid central (about)
        ('\u{025B}', 86),  // open-mid front unrounded (dress)
        ('\u{026A}', 102), // near-close near-front unrounded (kit)
        ('\u{028A}', 135), // near-close near-back rounded (foot)
        ('\u{028C}', 138), // open-mid back unrounded (strut)
    ];
    for (ch, expected_id) in ipa_vowels {
        assert_eq!(
            vocab.get(ch),
            Some(expected_id),
            "IPA vowel '{}' (U+{:04X}) should map to {}",
            ch,
            ch as u32,
            expected_id
        );
    }
}

#[test]
fn test_vocab_default_covers_tone_markers() {
    let vocab = KokoroVocab::kokoro_default();
    let tone_markers = [
        ('\u{2193}', 169), // down arrow
        ('\u{2192}', 171), // right arrow
        ('\u{2197}', 172), // northeast arrow
        ('\u{2198}', 173), // southeast arrow
    ];
    for (ch, expected_id) in tone_markers {
        assert_eq!(
            vocab.get(ch),
            Some(expected_id),
            "tone marker '{}' (U+{:04X}) should map to {}",
            ch,
            ch as u32,
            expected_id
        );
    }
}

#[test]
fn test_vocab_default_missing_chars_return_none() {
    let vocab = KokoroVocab::kokoro_default();
    // ASCII 'g' is NOT in the Kokoro vocab (they use IPA U+0261 instead).
    assert_eq!(vocab.get('g'), None, "'g' (ASCII) not in Kokoro vocab");
    // CJK and other scripts not in the vocab.
    assert_eq!(vocab.get('\u{4E00}'), None, "CJK char not in vocab");
    assert_eq!(vocab.get('\u{0410}'), None, "Cyrillic char not in vocab");
    assert_eq!(vocab.get('0'), None, "digit '0' not in vocab");
    // 'A' IS in the vocab as diphthong marker (ID 24) -- verify it is present.
    assert_eq!(
        vocab.get('A'),
        Some(24),
        "'A' is an uppercase diphthong marker"
    );
}

#[test]
fn test_vocab_decode_id_reverse_lookup() {
    let vocab = KokoroVocab::kokoro_default();
    // Verify roundtrip for a selection of entries.
    let test_pairs = [('a', 43), ('b', 44), ('.', 4), (' ', 16), ('\u{0283}', 131)];
    for (ch, id) in test_pairs {
        assert_eq!(vocab.get(ch), Some(id));
        assert_eq!(vocab.decode_id(id), Some(ch));
    }
}

// =============================================================================
// 2. Text-to-phoneme token conversion
// =============================================================================

#[test]
fn test_encode_ipa_word_hello() {
    // Encode IPA transcription of "hello": /hɛˈloʊ/
    let tok = KokoroTokenizer::kokoro_default();
    let ids = tok.encode("h\u{025B}\u{02C8}lo\u{028A}").unwrap();
    // Expected: [PAD, h(50), ɛ(86), ˈ(156), l(54), o(57), ʊ(135), PAD]
    assert_eq!(ids.len(), 8);
    assert_eq!(ids[0], PAD_TOKEN_ID);
    assert_eq!(ids[1], 50); // h
    assert_eq!(ids[2], 86); // ɛ
    assert_eq!(ids[3], 156); // ˈ
    assert_eq!(ids[4], 54); // l
    assert_eq!(ids[5], 57); // o
    assert_eq!(ids[6], 135); // ʊ
    assert_eq!(ids[7], PAD_TOKEN_ID);
}

#[test]
fn test_encode_punctuation_only() {
    let tok = KokoroTokenizer::kokoro_default();
    let ids = tok.encode(".,!?").unwrap();
    // [PAD, .(4), ,(3), !(5), ?(6), PAD]
    assert_eq!(ids, vec![0, 4, 3, 5, 6, 0]);
}

#[test]
fn test_encode_mixed_known_and_unknown_chars() {
    let tok = KokoroTokenizer::kokoro_default();
    // 'a' is in vocab (43), '1' is not, 'b' is in vocab (44).
    let ids = tok.encode("a1b").unwrap();
    // Unknown chars are dropped, so: [PAD, a(43), b(44), PAD]
    assert_eq!(ids, vec![0, 43, 44, 0]);
}

#[test]
fn test_encode_unicode_stress_markers() {
    let tok = KokoroTokenizer::kokoro_default();
    // Primary stress + secondary stress + length marker.
    let ids = tok.encode("\u{02C8}\u{02CC}\u{02D0}").unwrap();
    // [PAD, ˈ(156), ˌ(157), ː(158), PAD]
    assert_eq!(ids, vec![0, 156, 157, 158, 0]);
}

// =============================================================================
// 3. Token sequence padding/truncation
// =============================================================================

#[test]
fn test_encode_always_adds_start_and_end_padding() {
    let tok = KokoroTokenizer::kokoro_default();
    // Even a single character gets padding on both sides.
    let ids = tok.encode("a").unwrap();
    assert_eq!(ids.len(), 3); // PAD + a + PAD
    assert_eq!(ids[0], PAD_TOKEN_ID);
    assert_eq!(ids[2], PAD_TOKEN_ID);
}

#[test]
fn test_encode_empty_string_only_two_padding_tokens() {
    let tok = KokoroTokenizer::kokoro_default();
    let ids = tok.encode("").unwrap();
    assert_eq!(ids, vec![PAD_TOKEN_ID, PAD_TOKEN_ID]);
}

#[test]
fn test_encode_rejects_over_max_tokens() {
    let tok = KokoroTokenizer::kokoro_default();
    // 511 'a' characters -> 511 tokens, exceeds MAX_PHONEME_TOKENS (510).
    let long: String = std::iter::repeat_n('a', MAX_PHONEME_TOKENS + 1)
        .collect();
    let result = tok.encode(&long);
    assert!(result.is_err(), "511 tokens should exceed the 510 limit");
}

#[test]
fn test_encode_exactly_max_tokens_succeeds() {
    let tok = KokoroTokenizer::kokoro_default();
    // Exactly MAX_PHONEME_TOKENS = 510 'a' characters should succeed.
    let exact: String = std::iter::repeat_n('a', MAX_PHONEME_TOKENS).collect();
    let ids = tok.encode(&exact).unwrap();
    assert_eq!(ids.len(), MAX_PHONEME_TOKENS + 2); // tokens + 2 padding
}

// =============================================================================
// 4. Special tokens
// =============================================================================

#[test]
fn test_pad_token_id_is_zero() {
    assert_eq!(PAD_TOKEN_ID, 0, "PAD token must be ID 0");
}

#[test]
fn test_max_phoneme_tokens_is_510() {
    assert_eq!(
        MAX_PHONEME_TOKENS, 510,
        "max phoneme tokens = PlBert context (512) - 2 padding"
    );
}

// =============================================================================
// 5. Chunking behavior
// =============================================================================

#[test]
fn test_chunk_and_encode_respects_punctuation_split() {
    let tok = KokoroTokenizer::kokoro_default();
    // Build a string that overflows: 300 a's + period + 300 a's.
    let part1: String = std::iter::repeat_n('a', 300).collect();
    let part2: String = std::iter::repeat_n('a', 300).collect();
    let text = format!("{part1}.{part2}");

    let chunks = tok.chunk_and_encode(&text);
    assert!(chunks.len() >= 2, "should split at period");

    // First chunk should end with/include the period.
    let first_chunk_text = &chunks[0].0;
    assert!(
        first_chunk_text.ends_with('.'),
        "first chunk should end with period, got: ...{}",
        &first_chunk_text[first_chunk_text.len().saturating_sub(5)..]
    );
}

#[test]
fn test_chunk_and_encode_preserves_all_tokens() {
    // All chunks together should encode all the same tokens as individual chars.
    let tok = KokoroTokenizer::kokoro_default();
    let part1: String = std::iter::repeat_n('a', 300).collect();
    let part2: String = std::iter::repeat_n('b', 300).collect();
    let text = format!("{part1},{part2}");

    let chunks = tok.chunk_and_encode(&text);

    // Count total tokens across all chunks (excluding padding).
    let total_inner_tokens: usize = chunks
        .iter()
        .map(|(_, ids)| ids.len() - 2) // subtract start + end padding
        .sum();

    let expected_tokens = tok.count_tokens(&text);
    assert_eq!(
        total_inner_tokens, expected_tokens,
        "chunking should preserve all tokens"
    );
}

#[test]
fn test_chunk_and_encode_all_chunks_within_limit() {
    let tok = KokoroTokenizer::kokoro_default();
    // Very long string.
    let long: String = std::iter::repeat_n("abcde, ", 200).collect();
    let chunks = tok.chunk_and_encode(&long);

    for (text, ids) in &chunks {
        let inner_len = ids.len() - 2;
        assert!(
            inner_len <= MAX_PHONEME_TOKENS,
            "chunk '{}...' has {} tokens (limit {})",
            &text[..20.min(text.len())],
            inner_len,
            MAX_PHONEME_TOKENS
        );
    }
}

// =============================================================================
// 6. Vocabulary extension
// =============================================================================

#[test]
fn test_vocab_insert_auto_assigns_sequential_ids() {
    let mut vocab = KokoroVocab::empty();
    let id1 = vocab.insert_auto('x');
    let id2 = vocab.insert_auto('y');
    let id3 = vocab.insert_auto('z');
    assert_eq!(id1, 1); // first after padding(0)
    assert_eq!(id2, 2);
    assert_eq!(id3, 3);
}

#[test]
fn test_vocab_extend_json_adds_entries() {
    let mut vocab = KokoroVocab::empty();
    let added = vocab.extend_from_json(r#"{"x": 10, "y": 20}"#).unwrap();
    assert_eq!(added.len(), 2);
    assert_eq!(vocab.get('x'), Some(10));
    assert_eq!(vocab.get('y'), Some(20));
}

// =============================================================================
// 7. count_tokens consistency with encode
// =============================================================================

#[test]
fn test_count_tokens_matches_encode_length() {
    let tok = KokoroTokenizer::kokoro_default();
    let test_strings = [
        "hello world",
        "h\u{025B}\u{02C8}lo\u{028A}",
        ".,!?",
        "",
        "abc123",
    ];
    for s in &test_strings {
        let count = tok.count_tokens(s);
        let ids = tok.encode(s).unwrap();
        let encoded_inner_len = ids.len() - 2; // subtract padding
        assert_eq!(
            count, encoded_inner_len,
            "count_tokens vs encode mismatch for '{s}'"
        );
    }
}

// =============================================================================
// 8. Vocab validation
// =============================================================================

#[test]
fn test_vocab_validate_with_sufficient_embedding_size() {
    let vocab = KokoroVocab::kokoro_default();
    // Max ID is 177, so embedding_vocab_size >= 178 should pass.
    assert!(vocab.validate(178).is_ok());
    assert!(vocab.validate(256).is_ok());
    assert!(vocab.validate(1000).is_ok());
}

#[test]
fn test_vocab_validate_fails_when_id_exceeds_embedding_size() {
    let vocab = KokoroVocab::kokoro_default();
    // 177 is the max ID, so size=177 means 177 >= 177 -> fail.
    assert!(vocab.validate(177).is_err());
    assert!(vocab.validate(1).is_err());
}

#[test]
fn test_tokenizer_with_validated_vocab_rejects_small_embedding() {
    let vocab = KokoroVocab::kokoro_default();
    let result = KokoroTokenizer::with_validated_vocab(vocab, 100);
    assert!(
        result.is_err(),
        "should reject embedding size < max token ID"
    );
}
