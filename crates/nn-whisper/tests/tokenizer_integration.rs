// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Whisper tokenizer BPE encoding/decoding.
//!
//! Verifies the public API of `WhisperTokenizer` from an external crate
//! perspective: encode/decode round-trips, special token handling,
//! timestamp parsing, language tokens, empty inputs, and Unicode text.

use std::collections::HashMap;

use nn_whisper::{
    WhisperTokenizer, EOT_TOKEN, LANGUAGE_TOKEN_END, LANGUAGE_TOKEN_START, NO_SPEECH_TOKEN,
    SOT_TOKEN,
};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Build a realistic test vocabulary JSON with BPE-encoded tokens.
///
/// Includes individual ASCII character tokens, common merged tokens,
/// space-prefixed tokens (GPT-2 `Ġ` convention), and all special
/// Whisper tokens (SOT, EOT, language, task, timestamps).
fn realistic_vocab_json() -> String {
    let mut vocab: HashMap<&str, usize> = HashMap::new();

    // Individual ASCII byte-encoded characters (GPT-2 maps printable ASCII to itself)
    let chars: &[(&str, usize)] = &[
        ("H", 0),
        ("e", 1),
        ("l", 2),
        ("o", 3),
        ("w", 4),
        ("r", 5),
        ("d", 6),
        ("h", 7),
        ("a", 8),
        ("t", 9),
        ("i", 10),
        ("s", 11),
        ("n", 12),
        ("g", 13),
        ("p", 14),
        ("c", 15),
        ("m", 16),
        ("u", 17),
        ("b", 18),
        ("f", 19),
        ("k", 20),
        ("y", 21),
        ("v", 22),
        ("x", 23),
        ("z", 24),
        ("j", 25),
        ("q", 26),
        ("A", 27),
        ("B", 28),
        ("C", 29),
        ("D", 30),
        ("E", 31),
        ("F", 32),
        ("G", 33),
        ("I", 34),
        ("L", 35),
        ("M", 36),
        ("N", 37),
        ("O", 38),
        ("P", 39),
        ("R", 40),
        ("S", 41),
        ("T", 42),
        ("U", 43),
        ("W", 44),
        (".", 45),
        (",", 46),
        ("!", 47),
        ("?", 48),
        ("'", 49),
        ("-", 50),
        ("0", 51),
        ("1", 52),
        ("2", 53),
        ("3", 54),
        ("4", 55),
        ("5", 56),
        ("6", 57),
        ("7", 58),
        ("8", 59),
        ("9", 60),
    ];
    for &(k, v) in chars {
        vocab.insert(k, v);
    }

    // GPT-2 space encoding: byte 0x20 maps to U+0120 = 'Ġ'
    vocab.insert("\u{0120}", 100);

    // Merged tokens (results of BPE merges)
    let merged: &[(&str, usize)] = &[
        ("he", 200),
        ("ll", 201),
        ("lo", 202),
        ("\u{0120}w", 203),
        ("or", 204),
        ("ld", 205),
        ("hell", 206),
        ("hello", 207),
        ("orld", 208),
        ("world", 209),
        ("\u{0120}world", 210),
        ("th", 211),
        ("is", 212),
        ("\u{0120}is", 213),
        ("\u{0120}th", 214),
        ("\u{0120}the", 215),
        ("at", 216),
        ("\u{0120}a", 217),
        ("\u{0120}te", 218),
        ("st", 219),
        ("\u{0120}test", 220),
        ("en", 221),
        ("co", 222),
        ("de", 223),
        ("ing", 224),
        ("\u{0120}en", 225),
        ("\u{0120}de", 226),
        ("Hi", 227),
        ("this", 228),
    ];
    for &(k, v) in merged {
        vocab.insert(k, v);
    }

    // Special Whisper tokens
    let special: &[(&str, usize)] = &[
        ("<|endoftext|>", 50257),
        ("<|startoftranscript|>", 50258),
        // Language tokens (50259-50358, subset for testing)
        ("<|en|>", 50259),
        ("<|zh|>", 50260),
        ("<|de|>", 50261),
        ("<|es|>", 50262),
        ("<|ru|>", 50263),
        ("<|ko|>", 50264),
        ("<|fr|>", 50265),
        ("<|ja|>", 50266),
        ("<|pt|>", 50267),
        ("<|tr|>", 50268),
        ("<|pl|>", 50269),
        ("<|it|>", 50270),
        ("<|ar|>", 50271),
        ("<|nl|>", 50272),
        ("<|sv|>", 50273),
        ("<|hi|>", 50274),
        ("<|uk|>", 50275),
        // Task tokens
        ("<|translate|>", 50359),
        ("<|transcribe|>", 50360),
        // Control tokens
        ("<|startoflm|>", 50361),
        ("<|startofprev|>", 50362),
        ("<|nospeech|>", 50363),
        ("<|notimestamps|>", 50364),
        // Timestamp tokens (50365 + offset, at 0.02s resolution)
        ("<|0.00|>", 50365),
        ("<|0.02|>", 50366),
        ("<|0.04|>", 50367),
        ("<|1.00|>", 50415),
        ("<|2.00|>", 50465),
        ("<|5.00|>", 50615),
        ("<|10.00|>", 50865),
        ("<|15.00|>", 51115),
        ("<|30.00|>", 51865),
    ];
    for &(k, v) in special {
        vocab.insert(k, v);
    }

    serde_json::to_string(&vocab).unwrap()
}

/// BPE merge rules matching the realistic vocabulary above.
fn realistic_merges() -> &'static str {
    "\
#version: 0.2
h e
l l
l o
o r
l d
H i
t h
i s
a t
s t
e n
c o
d e
i n
\u{0120} w
\u{0120} a
\u{0120} t
he ll
or ld
hell o
\u{0120}w orld
th is
\u{0120} is
\u{0120}th e
\u{0120}a t
\u{0120}te st
in g
\u{0120} en
\u{0120} de
"
}

/// Build a tokenizer with both vocab and BPE merges for round-trip testing.
fn build_tokenizer() -> WhisperTokenizer {
    WhisperTokenizer::from_vocab_and_merges(&realistic_vocab_json(), realistic_merges()).unwrap()
}

/// Build a decode-only tokenizer (no BPE merges).
fn build_decode_only_tokenizer() -> WhisperTokenizer {
    WhisperTokenizer::from_vocab_str(&realistic_vocab_json()).unwrap()
}

// ---------------------------------------------------------------------------
// 1. Basic English text encoding/decoding round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_encode_decode_roundtrip_hello() {
    let tok = build_tokenizer();
    let original = "hello";
    let ids = tok.encode(original).unwrap();
    let decoded = tok.decode(&ids).unwrap();
    assert_eq!(decoded, original);
}

#[test]
fn test_encode_decode_roundtrip_hello_world() {
    let tok = build_tokenizer();
    let original = "hello world";
    let ids = tok.encode(original).unwrap();
    let decoded = tok.decode(&ids).unwrap();
    assert_eq!(decoded, original);
}

#[test]
fn test_encode_decode_roundtrip_this() {
    let tok = build_tokenizer();
    let original = "this";
    let ids = tok.encode(original).unwrap();
    let decoded = tok.decode(&ids).unwrap();
    assert_eq!(decoded, original);
}

#[test]
fn test_encode_produces_expected_bpe_tokens() {
    let tok = build_tokenizer();
    // "hello" should merge to a single token via BPE chain:
    // h,e,l,l,o -> he,l,l,o -> he,ll,o -> hell,o -> hello
    let ids = tok.encode("hello").unwrap();
    assert_eq!(ids, vec![207]); // "hello" = 207
}

#[test]
fn test_encode_space_prefixed_word() {
    let tok = build_tokenizer();
    // " world" -> pre-tokenize as [" world"]
    // byte-encode: space->Ġ, rest identity -> "Ġworld"
    // BPE: Ġ,w,o,r,l,d -> Ġw,o,r,l,d -> Ġw,or,l,d -> Ġw,or,ld -> Ġw,orld -> Ġworld
    let ids = tok.encode(" world").unwrap();
    assert_eq!(ids, vec![210]); // "Ġworld" = 210
}

#[test]
fn test_decode_concatenates_tokens() {
    let tok = build_tokenizer();
    // Decode "hello" + " world" (two tokens)
    let text = tok.decode(&[207, 210]).unwrap();
    assert_eq!(text, "hello world");
}

// ---------------------------------------------------------------------------
// 2. Special tokens (SOT, EOT, language tokens, timestamps)
// ---------------------------------------------------------------------------

#[test]
fn test_special_token_constants() {
    // Verify public constant values match Whisper spec
    assert_eq!(EOT_TOKEN, 50257);
    assert_eq!(SOT_TOKEN, 50258);
    assert_eq!(LANGUAGE_TOKEN_START, 50259);
    assert_eq!(LANGUAGE_TOKEN_END, 50358);
    assert_eq!(NO_SPEECH_TOKEN, 50363);
}

#[test]
fn test_is_special_for_all_special_tokens() {
    let tok = build_decode_only_tokenizer();

    // SOT and EOT
    assert!(tok.is_special(SOT_TOKEN));
    assert!(tok.is_special(EOT_TOKEN));

    // Language tokens
    assert!(tok.is_special(LANGUAGE_TOKEN_START)); // <|en|> = 50259
    assert!(tok.is_special(LANGUAGE_TOKEN_END)); // last language = 50358

    // Task tokens
    assert!(tok.is_special(50359)); // translate
    assert!(tok.is_special(50360)); // transcribe

    // Control tokens
    assert!(tok.is_special(50361)); // startoflm
    assert!(tok.is_special(50362)); // startofprev
    assert!(tok.is_special(NO_SPEECH_TOKEN)); // nospeech = 50363
    assert!(tok.is_special(50364)); // notimestamps

    // Timestamp tokens
    assert!(tok.is_special(50365)); // <|0.00|>
    assert!(tok.is_special(51865)); // <|30.00|>
}

#[test]
fn test_is_special_excludes_normal_tokens() {
    let tok = build_decode_only_tokenizer();
    assert!(!tok.is_special(0));
    assert!(!tok.is_special(100));
    assert!(!tok.is_special(50256)); // last normal token
}

#[test]
fn test_decode_skips_all_special_tokens() {
    let tok = build_decode_only_tokenizer();
    // Surround "hello" with every type of special token
    let ids = vec![
        SOT_TOKEN, // 50258 - start of transcript
        50259,     // <|en|> - language
        50360,     // <|transcribe|> - task
        50364,     // <|notimestamps|>
        207,       // "hello" - the only content token
        EOT_TOKEN, // 50257 - end of text
    ];
    let text = tok.decode(&ids).unwrap();
    assert_eq!(text, "hello");
}

#[test]
fn test_decode_with_timestamps_skips_control_tokens() {
    let tok = build_decode_only_tokenizer();
    // Full Whisper-style sequence: SOT, lang, task, notimestamps, timestamps, text, EOT
    let ids = vec![
        SOT_TOKEN, 50259, 50360, 50364, 50365, // <|0.00|>
        207,   // "hello"
        50415, // <|1.00|>
        EOT_TOKEN,
    ];
    let segments = tok.decode_with_timestamps(&ids).unwrap();
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].text, "hello");
    assert!((segments[0].start.unwrap() - 0.0).abs() < 1e-10);
    assert!((segments[0].end.unwrap() - 1.0).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// 3. Timestamp token parsing: <|0.00|> -> token ID -> back to 0.00
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_token_id_to_value_zero() {
    let tok = build_decode_only_tokenizer();
    // <|0.00|> = token 50365, value = 0.00
    let val = tok.timestamp_value(50365).unwrap();
    assert!((val - 0.0).abs() < 1e-10, "expected 0.00, got {val}");
}

#[test]
fn test_timestamp_token_id_to_value_one_step() {
    let tok = build_decode_only_tokenizer();
    // <|0.02|> = token 50366, value = 0.02 (one step at 0.02s resolution)
    let val = tok.timestamp_value(50366).unwrap();
    assert!((val - 0.02).abs() < 1e-10, "expected 0.02, got {val}");
}

#[test]
fn test_timestamp_token_id_to_value_one_second() {
    let tok = build_decode_only_tokenizer();
    // <|1.00|> = token 50415 = 50365 + 50, value = 50 * 0.02 = 1.00
    let val = tok.timestamp_value(50415).unwrap();
    assert!((val - 1.0).abs() < 1e-10, "expected 1.00, got {val}");
}

#[test]
fn test_timestamp_token_id_to_value_thirty_seconds() {
    let tok = build_decode_only_tokenizer();
    // <|30.00|> = token 51865 = 50365 + 1500, value = 1500 * 0.02 = 30.00
    let val = tok.timestamp_value(51865).unwrap();
    assert!((val - 30.0).abs() < 1e-10, "expected 30.00, got {val}");
}

#[test]
fn test_timestamp_roundtrip_id_value_id() {
    let tok = build_decode_only_tokenizer();
    // For a range of timestamp token IDs, verify value -> back-computed ID
    for offset in [0, 1, 50, 100, 250, 500, 1000, 1500] {
        let token_id = 50365 + offset;
        let expected_seconds = offset as f64 * 0.02;

        let value = tok.timestamp_value(token_id).unwrap();
        assert!(
            (value - expected_seconds).abs() < 1e-10,
            "token {token_id}: expected {expected_seconds}, got {value}"
        );

        // Compute back to token ID from value
        let recomputed_id = (value / 0.02).round() as usize + 50365;
        assert_eq!(
            recomputed_id, token_id,
            "roundtrip failed for offset {offset}"
        );
    }
}

#[test]
fn test_timestamp_non_timestamp_returns_none() {
    let tok = build_decode_only_tokenizer();
    // Normal tokens are not timestamps
    assert_eq!(tok.timestamp_value(0), None);
    assert_eq!(tok.timestamp_value(207), None);
    // EOT, SOT, and notimestamps are not timestamps
    assert_eq!(tok.timestamp_value(EOT_TOKEN), None);
    assert_eq!(tok.timestamp_value(SOT_TOKEN), None);
    assert_eq!(tok.timestamp_value(50364), None); // notimestamps
}

#[test]
fn test_is_timestamp_boundary() {
    let tok = build_decode_only_tokenizer();
    // 50364 = notimestamps (NOT a timestamp)
    assert!(!tok.is_timestamp(50364));
    // 50365 = first timestamp (<|0.00|>)
    assert!(tok.is_timestamp(50365));
    // Well beyond 30s
    assert!(tok.is_timestamp(60000));
}

#[test]
fn test_decode_with_timestamps_multi_segment() {
    let tok = build_decode_only_tokenizer();
    // Two timestamped segments: [0.00-1.00] "hello" and [1.00-2.00] " world"
    let ids = vec![
        50365, 207, 50415, // <|0.00|> hello <|1.00|>
        50415, 210, 50465, // <|1.00|> " world" <|2.00|>
    ];
    let segments = tok.decode_with_timestamps(&ids).unwrap();
    assert_eq!(segments.len(), 2);

    assert_eq!(segments[0].text, "hello");
    assert!((segments[0].start.unwrap() - 0.0).abs() < 1e-10);
    assert!((segments[0].end.unwrap() - 1.0).abs() < 1e-10);

    assert_eq!(segments[1].text, " world");
    assert!((segments[1].start.unwrap() - 1.0).abs() < 1e-10);
    assert!((segments[1].end.unwrap() - 2.0).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// 4. Language detection tokens
// ---------------------------------------------------------------------------

#[test]
fn test_language_token_english() {
    let tok = build_decode_only_tokenizer();
    assert_eq!(tok.language_token("en"), Some(50259));
}

#[test]
fn test_language_token_multilingual() {
    let tok = build_decode_only_tokenizer();
    assert_eq!(tok.language_token("zh"), Some(50260));
    assert_eq!(tok.language_token("de"), Some(50261));
    assert_eq!(tok.language_token("es"), Some(50262));
    assert_eq!(tok.language_token("ru"), Some(50263));
    assert_eq!(tok.language_token("ko"), Some(50264));
    assert_eq!(tok.language_token("fr"), Some(50265));
    assert_eq!(tok.language_token("ja"), Some(50266));
    assert_eq!(tok.language_token("pt"), Some(50267));
    assert_eq!(tok.language_token("tr"), Some(50268));
    assert_eq!(tok.language_token("it"), Some(50270));
    assert_eq!(tok.language_token("ar"), Some(50271));
}

#[test]
fn test_language_token_not_in_vocab_returns_none() {
    let tok = build_decode_only_tokenizer();
    // Language not in our test vocab
    assert_eq!(tok.language_token("xx"), None);
    assert_eq!(tok.language_token("zz"), None);
}

#[test]
fn test_language_token_range_constants() {
    // Whisper defines 100 language tokens: 50259 through 50358
    assert_eq!(LANGUAGE_TOKEN_END - LANGUAGE_TOKEN_START + 1, 100);
}

#[test]
fn test_language_tokens_are_special() {
    let tok = build_decode_only_tokenizer();
    // Every language token ID is in the special range
    for lang_id in LANGUAGE_TOKEN_START..=LANGUAGE_TOKEN_END {
        assert!(
            tok.is_special(lang_id),
            "language token {lang_id} should be special"
        );
    }
}

#[test]
fn test_language_tokens_are_not_timestamps() {
    let tok = build_decode_only_tokenizer();
    for lang_id in LANGUAGE_TOKEN_START..=LANGUAGE_TOKEN_END {
        assert!(
            !tok.is_timestamp(lang_id),
            "language token {lang_id} should not be a timestamp"
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Empty string encoding
// ---------------------------------------------------------------------------

#[test]
fn test_encode_empty_string() {
    let tok = build_tokenizer();
    let ids = tok.encode("").unwrap();
    assert!(
        ids.is_empty(),
        "encoding empty string should produce no tokens"
    );
}

#[test]
fn test_decode_empty_token_sequence() {
    let tok = build_decode_only_tokenizer();
    let text = tok.decode(&[]).unwrap();
    assert_eq!(text, "");
}

#[test]
fn test_encode_decode_roundtrip_empty() {
    let tok = build_tokenizer();
    let ids = tok.encode("").unwrap();
    let text = tok.decode(&ids).unwrap();
    assert_eq!(text, "");
}

#[test]
fn test_decode_with_timestamps_empty() {
    let tok = build_decode_only_tokenizer();
    let segments = tok.decode_with_timestamps(&[]).unwrap();
    assert!(segments.is_empty());
}

#[test]
fn test_decode_only_special_tokens_yields_empty() {
    let tok = build_decode_only_tokenizer();
    // Only special tokens, no content
    let text = tok.decode(&[SOT_TOKEN, 50259, 50360, EOT_TOKEN]).unwrap();
    assert_eq!(text, "");
}

// ---------------------------------------------------------------------------
// 6. Unicode handling (non-ASCII text)
// ---------------------------------------------------------------------------

#[test]
fn test_encode_decode_roundtrip_with_digits() {
    let tok = build_tokenizer();
    let original = "123";
    let ids = tok.encode(original).unwrap();
    let decoded = tok.decode(&ids).unwrap();
    assert_eq!(decoded, original);
}

#[test]
fn test_encode_decode_roundtrip_punctuation() {
    let tok = build_tokenizer();
    for punct in [".", ",", "!", "?", "'", "-"] {
        let ids = tok.encode(punct).unwrap();
        let decoded = tok.decode(&ids).unwrap();
        assert_eq!(decoded, punct, "roundtrip failed for punctuation '{punct}'");
    }
}

#[test]
fn test_encode_decode_roundtrip_mixed_ascii() {
    let tok = build_tokenizer();
    // Mix of letters, digits, punctuation
    let original = "hello, world! 123.";
    let ids = tok.encode(original).unwrap();
    let decoded = tok.decode(&ids).unwrap();
    assert_eq!(decoded, original);
}

// ---------------------------------------------------------------------------
// Additional edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_token_str_lookup() {
    let tok = build_decode_only_tokenizer();
    assert_eq!(tok.token_str(207), Some("hello"));
    assert_eq!(tok.token_str(210), Some("\u{0120}world"));
    assert_eq!(tok.token_str(EOT_TOKEN), Some("<|endoftext|>"));
    assert_eq!(tok.token_str(SOT_TOKEN), Some("<|startoftranscript|>"));
}

#[test]
fn test_token_id_lookup() {
    let tok = build_decode_only_tokenizer();
    assert_eq!(tok.token_id("hello"), Some(207));
    assert_eq!(tok.token_id("\u{0120}world"), Some(210));
    assert_eq!(tok.token_id("<|endoftext|>"), Some(EOT_TOKEN));
    assert_eq!(tok.token_id("nonexistent"), None);
}

#[test]
fn test_vocab_size_includes_special_tokens() {
    let tok = build_decode_only_tokenizer();
    // vocab_size is max(id) + 1; with timestamp token 51865, it's at least 51866
    assert!(
        tok.vocab_size() > 51865,
        "vocab_size should include all timestamp tokens"
    );
}

#[test]
fn test_can_encode_with_merges() {
    let tok = build_tokenizer();
    assert!(tok.can_encode());
}

#[test]
fn test_cannot_encode_without_merges() {
    let tok = build_decode_only_tokenizer();
    assert!(!tok.can_encode());
    let result = tok.encode("hello");
    assert!(result.is_err());
}

#[test]
fn test_out_of_range_token_decode_error() {
    // Build a tiny vocab to get a small vocab_size
    let vocab: HashMap<&str, usize> = [("a", 0), ("b", 1)].into_iter().collect();
    let json = serde_json::to_string(&vocab).unwrap();
    let tok = WhisperTokenizer::from_vocab_str(&json).unwrap();
    assert_eq!(tok.vocab_size(), 2);

    // Token ID 5 is < 50257 (not special) but >= vocab_size (2)
    let result = tok.decode(&[5]);
    assert!(result.is_err());
}

#[test]
fn test_decode_with_timestamps_text_before_first_timestamp() {
    let tok = build_decode_only_tokenizer();
    // Text tokens before any timestamp should produce a segment with no times
    let ids = vec![207, 50365, 210, 50415]; // hello <|0.00|> " world" <|1.00|>
    let segments = tok.decode_with_timestamps(&ids).unwrap();
    assert_eq!(segments.len(), 2);

    // First segment: text before timestamps, no start/end
    assert_eq!(segments[0].text, "hello");
    assert_eq!(segments[0].start, None);
    assert_eq!(segments[0].end, None);

    // Second segment: timestamped
    assert_eq!(segments[1].text, " world");
    assert!((segments[1].start.unwrap() - 0.0).abs() < 1e-10);
    assert!((segments[1].end.unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_decode_with_timestamps_trailing_text() {
    let tok = build_decode_only_tokenizer();
    // Text after a start timestamp with no closing timestamp
    let ids = vec![50365, 207]; // <|0.00|> hello (no end timestamp)
    let segments = tok.decode_with_timestamps(&ids).unwrap();
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].text, "hello");
    assert!((segments[0].start.unwrap() - 0.0).abs() < 1e-10);
    assert_eq!(segments[0].end, None);
}
