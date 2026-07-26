// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`KokoroTokenizer`] and [`KokoroVocab`].

use super::*;

// -- KokoroVocab tests -------------------------------------------------------

#[test]
fn test_vocab_kokoro_default_has_178_tokens() {
    let vocab = KokoroVocab::kokoro_default();
    assert_eq!(
        vocab.n_tokens(),
        178,
        "Kokoro vocab should have 178 tokens (0-177)"
    );
}

#[test]
fn test_vocab_kokoro_default_punctuation() {
    let vocab = KokoroVocab::kokoro_default();
    assert_eq!(vocab.get(';'), Some(1));
    assert_eq!(vocab.get(':'), Some(2));
    assert_eq!(vocab.get(','), Some(3));
    assert_eq!(vocab.get('.'), Some(4));
    assert_eq!(vocab.get('!'), Some(5));
    assert_eq!(vocab.get('?'), Some(6));
    assert_eq!(vocab.get(' '), Some(16));
}

#[test]
fn test_vocab_kokoro_default_diphthongs() {
    let vocab = KokoroVocab::kokoro_default();
    assert_eq!(vocab.get('A'), Some(24), "A = eɪ diphthong");
    assert_eq!(vocab.get('I'), Some(25), "I = aɪ diphthong");
    assert_eq!(vocab.get('O'), Some(31), "O = oʊ diphthong (US)");
    assert_eq!(vocab.get('W'), Some(39), "W = aʊ diphthong");
    assert_eq!(vocab.get('Y'), Some(41), "Y = ɔɪ diphthong");
}

#[test]
fn test_vocab_kokoro_default_ipa_consonants() {
    let vocab = KokoroVocab::kokoro_default();
    assert_eq!(vocab.get('ʃ'), Some(131));
    assert_eq!(vocab.get('ʒ'), Some(147));
    assert_eq!(vocab.get('θ'), Some(119));
    assert_eq!(vocab.get('ð'), Some(81));
    assert_eq!(vocab.get('ŋ'), Some(112));
    assert_eq!(vocab.get('ɹ'), Some(123));
    assert_eq!(vocab.get('ʤ'), Some(82));
    assert_eq!(vocab.get('ʧ'), Some(133));
}

#[test]
fn test_vocab_kokoro_default_stress_markers() {
    let vocab = KokoroVocab::kokoro_default();
    assert_eq!(vocab.get('ˈ'), Some(156), "primary stress");
    assert_eq!(vocab.get('ˌ'), Some(157), "secondary stress");
    assert_eq!(vocab.get('ː'), Some(158), "length marker");
}

#[test]
fn test_vocab_round_trip() {
    let vocab = KokoroVocab::kokoro_default();
    for (ch, id) in vocab.iter() {
        assert_eq!(vocab.get(ch), Some(id));
        assert_eq!(vocab.decode_id(id), Some(ch));
    }
}

#[test]
fn test_vocab_insert_and_remove() {
    let mut vocab = KokoroVocab::empty();
    assert!(vocab.is_empty());
    vocab.insert('a', 1);
    vocab.insert('b', 2);
    assert_eq!(vocab.len(), 2);
    assert_eq!(vocab.get('a'), Some(1));
    assert_eq!(vocab.remove('a'), Some(1));
    assert_eq!(vocab.get('a'), None);
    assert_eq!(vocab.len(), 1);
}

#[test]
fn test_vocab_from_json_map() {
    let json = r#"{"vocab": {";": 1, "a": 43, "ˈ": 156}}"#;
    let parsed: serde_json::Value = serde_json::from_str(json).unwrap();
    let map = parsed["vocab"].as_object().unwrap();
    let vocab = KokoroVocab::from_json_map(map).expect("should parse");
    assert_eq!(vocab.get(';'), Some(1));
    assert_eq!(vocab.get('a'), Some(43));
    assert_eq!(vocab.get('\u{02C8}'), Some(156));
    assert_eq!(vocab.n_tokens(), 157);
}

#[test]
fn test_vocab_from_config_json() {
    let json = r#"{"istftnet": {}, "vocab": {".": 4, "!": 5}}"#;
    let vocab = KokoroVocab::from_config_json(json).expect("should parse");
    assert_eq!(vocab.get('.'), Some(4));
    assert_eq!(vocab.get('!'), Some(5));
}

#[test]
fn test_vocab_from_config_json_missing_vocab_returns_error() {
    let json = r#"{"istftnet": {}}"#;
    let result = KokoroVocab::from_config_json(json);
    assert!(result.is_err());
}

// -- KokoroTokenizer tests ---------------------------------------------------

#[test]
fn test_encode_hello_world() {
    let tok = KokoroTokenizer::kokoro_default();
    // "hɛˈloʊ" → h=50, ɛ=86, ˈ=156, l=54, o=57, ʊ=135
    // With padding: [0, 50, 86, 156, 54, 57, 135, 0]
    let ids = tok.encode("hɛˈloʊ").expect("should encode");
    assert_eq!(ids[0], PAD_TOKEN_ID, "starts with padding");
    assert_eq!(*ids.last().unwrap(), PAD_TOKEN_ID, "ends with padding");
    assert_eq!(ids[1], 50, "h=50");
    assert_eq!(ids[2], 86, "ɛ=86");
    assert_eq!(ids[3], 156, "ˈ=156");
    assert_eq!(ids.len(), 8); // 6 tokens + 2 padding
}

#[test]
fn test_encode_empty_string() {
    let tok = KokoroTokenizer::kokoro_default();
    let ids = tok.encode("").expect("should encode empty");
    assert_eq!(ids, vec![0, 0], "empty string gets just two padding tokens");
}

#[test]
fn test_encode_unknown_chars_dropped() {
    let tok = KokoroTokenizer::kokoro_default();
    // 'g' (ASCII) is not in vocab — 'ɡ' (U+0261) is
    let ids = tok.encode("g").expect("should encode");
    assert_eq!(ids, vec![0, 0], "'g' not in Kokoro vocab, dropped");
    let ids = tok.encode("ɡ").expect("should encode");
    assert_eq!(ids, vec![0, 92, 0], "ɡ (U+0261) = 92");
}

#[test]
fn test_encode_exceeds_limit_returns_error() {
    let tok = KokoroTokenizer::kokoro_default();
    // Create a string of 511 'a' characters (each maps to token 43)
    let long_phonemes: String = std::iter::repeat_n('a', 511).collect();
    let result = tok.encode(&long_phonemes);
    assert!(result.is_err(), "should reject >510 tokens");
}

#[test]
fn test_count_tokens() {
    let tok = KokoroTokenizer::kokoro_default();
    assert_eq!(tok.count_tokens("hɛˈloʊ"), 6);
    assert_eq!(tok.count_tokens(""), 0);
    // 'g' not in vocab, doesn't count
    assert_eq!(tok.count_tokens("gɡ"), 1);
}

#[test]
fn test_chunk_and_encode_short_text() {
    let tok = KokoroTokenizer::kokoro_default();
    let chunks = tok.chunk_and_encode("hɛˈloʊ");
    assert_eq!(chunks.len(), 1, "short text = single chunk");
    assert_eq!(chunks[0].0, "hɛˈloʊ");
}

#[test]
fn test_chunk_and_encode_empty() {
    let tok = KokoroTokenizer::kokoro_default();
    let chunks = tok.chunk_and_encode("");
    assert!(chunks.is_empty());
}

#[test]
fn test_chunk_and_encode_long_text_splits_at_punctuation() {
    let tok = KokoroTokenizer::kokoro_default();
    // Create a string that needs splitting: 300 'a' + '.' + 300 'a'
    let part1: String = std::iter::repeat_n('a', 300).collect();
    let part2: String = std::iter::repeat_n('a', 300).collect();
    let long_text = format!("{part1}.{part2}");
    let chunks = tok.chunk_and_encode(&long_text);
    assert!(
        chunks.len() >= 2,
        "should split at period, got {} chunks",
        chunks.len()
    );
    // Each chunk should fit within limit
    for (phonemes, ids) in &chunks {
        let token_count = ids.len() - 2; // subtract padding
        assert!(
            token_count <= MAX_PHONEME_TOKENS,
            "chunk '{}...' has {} tokens, max {}",
            &phonemes[..20.min(phonemes.len())],
            token_count,
            MAX_PHONEME_TOKENS
        );
    }
}

#[test]
fn test_chunk_and_encode_all_chunks_have_padding() {
    let tok = KokoroTokenizer::kokoro_default();
    let part1: String = std::iter::repeat_n('a', 300).collect();
    let part2: String = std::iter::repeat_n('a', 300).collect();
    let long_text = format!("{part1},{part2}");
    let chunks = tok.chunk_and_encode(&long_text);
    for (_, ids) in &chunks {
        assert_eq!(ids[0], PAD_TOKEN_ID, "chunk must start with padding");
        assert_eq!(
            *ids.last().unwrap(),
            PAD_TOKEN_ID,
            "chunk must end with padding"
        );
    }
}

// -- KokoroVocab::validate tests (#3460) ------------------------------------

#[test]
fn test_vocab_validate_default_passes_at_178() {
    let vocab = KokoroVocab::kokoro_default();
    assert!(vocab.validate(178).is_ok());
}

#[test]
fn test_vocab_validate_default_fails_at_177() {
    let vocab = KokoroVocab::kokoro_default();
    // Max ID in default vocab is 177, so embedding_vocab_size=177 should fail
    let result = vocab.validate(177);
    assert!(result.is_err(), "ID 177 >= size 177 should fail");
}

#[test]
fn test_vocab_validate_empty_passes() {
    let vocab = KokoroVocab::empty();
    assert!(vocab.validate(0).is_ok(), "empty vocab always valid");
}

#[test]
fn test_vocab_validate_oob_single_entry() {
    let mut vocab = KokoroVocab::empty();
    vocab.insert('x', 200);
    assert!(vocab.validate(200).is_err(), "ID 200 >= size 200");
    assert!(vocab.validate(201).is_ok(), "ID 200 < size 201");
}

// -- KokoroVocab::insert_auto tests (#3460) ---------------------------------

#[test]
fn test_vocab_insert_auto_sequential() {
    let mut vocab = KokoroVocab::empty();
    // n_tokens starts at 1 (pad=0)
    let id1 = vocab.insert_auto('a');
    assert_eq!(id1, 1);
    let id2 = vocab.insert_auto('b');
    assert_eq!(id2, 2);
    let id3 = vocab.insert_auto('c');
    assert_eq!(id3, 3);
    assert_eq!(vocab.n_tokens(), 4);
    assert_eq!(vocab.get('a'), Some(1));
    assert_eq!(vocab.get('b'), Some(2));
    assert_eq!(vocab.get('c'), Some(3));
}

#[test]
fn test_vocab_insert_auto_after_manual_insert() {
    let mut vocab = KokoroVocab::empty();
    vocab.insert('x', 50);
    // n_tokens = 51 after inserting ID 50
    let auto_id = vocab.insert_auto('y');
    assert_eq!(auto_id, 51, "should use next sequential ID after max");
}

// -- KokoroVocab::extend_from_json tests (#3460) ----------------------------

#[test]
fn test_vocab_extend_from_json_basic() {
    let mut vocab = KokoroVocab::empty();
    let added = vocab.extend_from_json(r#"{"a": 1, "b": 2}"#).unwrap();
    assert_eq!(added.len(), 2);
    assert_eq!(vocab.get('a'), Some(1));
    assert_eq!(vocab.get('b'), Some(2));
}

#[test]
fn test_vocab_extend_from_json_overwrites_existing() {
    let mut vocab = KokoroVocab::empty();
    vocab.insert('a', 1);
    let _ = vocab.extend_from_json(r#"{"a": 99}"#).unwrap();
    assert_eq!(vocab.get('a'), Some(99), "overwritten");
}

#[test]
fn test_vocab_extend_from_json_malformed_returns_error() {
    let mut vocab = KokoroVocab::empty();
    let result = vocab.extend_from_json("not json");
    assert!(result.is_err());
}

#[test]
fn test_vocab_extend_from_json_non_object_returns_error() {
    let mut vocab = KokoroVocab::empty();
    let result = vocab.extend_from_json("[1, 2, 3]");
    assert!(result.is_err());
}

#[test]
fn test_vocab_extend_from_json_skips_multichar_keys() {
    let mut vocab = KokoroVocab::empty();
    let added = vocab.extend_from_json(r#"{"ab": 1, "c": 2}"#).unwrap();
    assert_eq!(added.len(), 1, "multi-char key 'ab' skipped");
    assert_eq!(vocab.get('c'), Some(2));
}

// -- KokoroTokenizer::with_validated_vocab tests (#3460) --------------------

#[test]
fn test_tokenizer_with_validated_vocab_passes() {
    let vocab = KokoroVocab::kokoro_default();
    let tok = KokoroTokenizer::with_validated_vocab(vocab, 178);
    assert!(tok.is_ok());
}

#[test]
fn test_tokenizer_with_validated_vocab_fails_small_embedding() {
    let vocab = KokoroVocab::kokoro_default();
    let tok = KokoroTokenizer::with_validated_vocab(vocab, 100);
    assert!(tok.is_err(), "vocab has IDs > 100");
}
