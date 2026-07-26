// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Whisper tokenizer.

use super::*;

/// Build a minimal test vocabulary JSON string.
///
/// Maps a few known GPT-2 byte-encoded tokens to IDs for testing.
fn test_vocab_json() -> String {
    // GPT-2 byte encoding: 'h' = 'h', 'e' = 'e', 'l' = 'l', 'o' = 'o'
    // ' ' (space, 0x20) maps to Unicode 'Ġ' (U+0120) in GPT-2
    // So "hello" is "hello" and " world" is "Ġworld"
    serde_json::json!({
        "hello": 0,
        "Ġworld": 1,
        "Ġ": 2,
        "the": 3,
        "Ġquick": 4,
        "Ġbrown": 5,
        "Ġfox": 6,
        "<|endoftext|>": 50257,
        "<|startoftranscript|>": 50258,
        "<|en|>": 50259,
        "<|fr|>": 50260,
        "<|transcribe|>": 50360,
        "<|notimestamps|>": 50364,
    })
    .to_string()
}

#[test]
fn test_load_from_vocab_str() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    assert!(tok.vocab_size() > 50364);
    assert_eq!(tok.token_id("hello"), Some(0));
    assert_eq!(tok.token_id("<|endoftext|>"), Some(50257));
}

#[test]
fn test_decode_simple_tokens() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // "hello" + " world" = "hello world"
    let text = tok.decode(&[0, 1]).unwrap();
    assert_eq!(text, "hello world");
}

#[test]
fn test_decode_single_token() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    let text = tok.decode(&[0]).unwrap();
    assert_eq!(text, "hello");
}

#[test]
fn test_decode_empty() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    let text = tok.decode(&[]).unwrap();
    assert_eq!(text, "");
}

#[test]
fn test_decode_skips_special_tokens() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Special tokens (>= 50257) should be silently skipped.
    let text = tok
        .decode(&[50258, 50259, 50360, 50364, 0, 1, 50257])
        .unwrap();
    assert_eq!(text, "hello world");
}

#[test]
fn test_decode_with_space_token() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Token 2 is "Ġ" which decodes to " " (space).
    // "Ġquick" already includes the leading space, so token 2 adds an extra space.
    let text = tok.decode(&[3, 2, 0]).unwrap();
    assert_eq!(text, "the hello");
    // Standalone space token between normal tokens:
    let text2 = tok.decode(&[0, 2, 0]).unwrap();
    assert_eq!(text2, "hello hello");
}

#[test]
fn test_out_of_range_non_special_token() {
    // Build a small vocab where non-special tokens can be out of range.
    let json = serde_json::json!({"hello": 0, "world": 1}).to_string();
    let tok = WhisperTokenizer::from_vocab_str(&json).unwrap();
    assert_eq!(tok.vocab_size(), 2);

    // Token ID 5 is non-special (< 50257) and out of vocab range (>= 2).
    let result = tok.decode(&[5]);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("out of vocabulary"),
        "expected out-of-range error: {err_msg}"
    );
}

#[test]
fn test_out_of_range_special_token_skipped() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Tokens >= 50257 are special and silently skipped, even if beyond vocab_size.
    let text = tok.decode(&[0, 999999, 1]).unwrap();
    assert_eq!(text, "hello world");
}

#[test]
fn test_is_special() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    assert!(!tok.is_special(0));
    assert!(!tok.is_special(50256));
    assert!(tok.is_special(50257)); // EOT
    assert!(tok.is_special(50258)); // SOT
    assert!(tok.is_special(50364)); // NO_TIMESTAMPS
    assert!(tok.is_special(50365)); // First timestamp
}

#[test]
fn test_is_timestamp() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    assert!(!tok.is_timestamp(0));
    assert!(!tok.is_timestamp(50257));
    assert!(!tok.is_timestamp(50364)); // NO_TIMESTAMPS, not a timestamp
    assert!(tok.is_timestamp(50365)); // First timestamp (<|0.00|>)
    assert!(tok.is_timestamp(50366)); // Second timestamp (<|0.02|>)
    assert!(tok.is_timestamp(51865)); // 30.00s = (51865 - 50365) * 0.02
}

#[test]
fn test_timestamp_value() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    assert_eq!(tok.timestamp_value(0), None);
    assert_eq!(tok.timestamp_value(50364), None); // NO_TIMESTAMPS, not a timestamp

    // 50365 = 0.00s (TIMESTAMP_BEGIN)
    let ts0 = tok.timestamp_value(50365).unwrap();
    assert!((ts0 - 0.0).abs() < 1e-10);

    // 50366 = 0.02s
    let ts1 = tok.timestamp_value(50366).unwrap();
    assert!((ts1 - 0.02).abs() < 1e-10);

    // 50415 = 1.00s (50365 + 50)
    let ts2 = tok.timestamp_value(50415).unwrap();
    assert!((ts2 - 1.0).abs() < 1e-10);

    // 51865 = 30.00s (50365 + 1500)
    let ts3 = tok.timestamp_value(51865).unwrap();
    assert!((ts3 - 30.0).abs() < 1e-10);
}

#[test]
fn test_language_token() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    assert_eq!(tok.language_token("en"), Some(50259));
    assert_eq!(tok.language_token("fr"), Some(50260));
    assert_eq!(tok.language_token("de"), None); // Not in test vocab.
}

#[test]
fn test_token_str_lookup() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    assert_eq!(tok.token_str(0), Some("hello"));
    assert_eq!(tok.token_str(1), Some("Ġworld"));
}

#[test]
fn test_decode_with_timestamps_no_timestamps() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    let segments = tok.decode_with_timestamps(&[0, 1]).unwrap();
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].text, "hello world");
    assert_eq!(segments[0].start, None);
    assert_eq!(segments[0].end, None);
}

#[test]
fn test_decode_with_timestamps_single_segment() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // <|0.00|> hello world <|2.00|>
    // 50365 = 0.00s, 50465 = (50465-50365)*0.02 = 2.00s
    let segments = tok.decode_with_timestamps(&[50365, 0, 1, 50465]).unwrap();
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].text, "hello world");
    assert!((segments[0].start.unwrap() - 0.0).abs() < 1e-10);
    assert!((segments[0].end.unwrap() - 2.0).abs() < 1e-10);
}

#[test]
fn test_decode_with_timestamps_multiple_segments() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // <|0.00|> hello <|1.00|> <|1.00|> world <|2.00|>
    // 50365 = 0.00s, 50415 = 1.00s, 50465 = 2.00s
    let segments = tok
        .decode_with_timestamps(&[50365, 0, 50415, 50415, 1, 50465])
        .unwrap();
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].text, "hello");
    assert!((segments[0].start.unwrap() - 0.0).abs() < 1e-10);
    assert!((segments[0].end.unwrap() - 1.0).abs() < 1e-10);
    assert_eq!(segments[1].text, " world");
    assert!((segments[1].start.unwrap() - 1.0).abs() < 1e-10);
    assert!((segments[1].end.unwrap() - 2.0).abs() < 1e-10);
}

#[test]
fn test_decode_with_timestamps_skips_eot() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    let segments = tok
        .decode_with_timestamps(&[50258, 50259, 50360, 50364, 50365, 0, 50415, 50257])
        .unwrap();
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].text, "hello");
}

#[test]
fn test_byte_encoder_roundtrip() {
    // Every byte value should have a mapping.
    let encoder = build_byte_encoder();
    assert_eq!(encoder.len(), 256);

    let decoder = build_byte_decoder();
    assert_eq!(decoder.len(), 256);

    // Roundtrip: byte → char → byte.
    for b in 0u8..=255u8 {
        let ch = encoder[&b];
        let b2 = decoder[&ch];
        assert_eq!(b, b2, "roundtrip failed for byte {b}");
    }
}

#[test]
fn test_byte_encoder_ascii_passthrough() {
    let encoder = build_byte_encoder();
    // Printable ASCII should map to themselves.
    for b in b'!'..=b'~' {
        assert_eq!(
            encoder[&b],
            char::from(b),
            "ASCII byte {b} should map to itself"
        );
    }
}

#[test]
fn test_byte_encoder_space_maps_to_special() {
    let encoder = build_byte_encoder();
    // Space (0x20) should NOT map to ' ' — it maps to a higher codepoint.
    let space_char = encoder[&0x20];
    assert_ne!(space_char, ' ');
    // In GPT-2, space maps to U+0120 ('Ġ').
    assert_eq!(space_char, 'Ġ');
}

#[test]
fn test_decode_multiple_words_space_handling() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // "the" + " quick" + " brown" + " fox"
    let text = tok.decode(&[3, 4, 5, 6]).unwrap();
    assert_eq!(text, "the quick brown fox");
}

#[test]
fn test_empty_vocab_json() {
    let tok = WhisperTokenizer::from_vocab_str("{}").unwrap();
    assert_eq!(tok.vocab_size(), 0);
    let text = tok.decode(&[]).unwrap();
    assert_eq!(text, "");
}

#[test]
fn test_invalid_json() {
    let result = WhisperTokenizer::from_vocab_str("not json");
    assert!(result.is_err());
}

/// Regression test for #1509: constants now match whisper-large-v3-turbo
/// (HuggingFace tokenizer.json). Verifies the corrected token IDs:
///   translate=50359, transcribe=50360, ..., notimestamps=50364, timestamps=50365+
#[test]
fn test_v3_turbo_token_ids_match_constants() {
    // Build a vocab with v3-turbo-style IDs matching the corrected constants.
    let vocab = serde_json::json!({
        "hello": 0,
        "Ġworld": 1,
        "<|endoftext|>": 50257,
        "<|startoftranscript|>": 50258,
        "<|en|>": 50259,
        "<|translate|>": 50359,
        "<|transcribe|>": 50360,
        "<|startoflm|>": 50361,
        "<|startofprev|>": 50362,
        "<|nospeech|>": 50363,
        "<|notimestamps|>": 50364,
        "<|0.00|>": 50365,
        "<|0.02|>": 50366,
    })
    .to_string();
    let tok = WhisperTokenizer::from_vocab_str(&vocab).unwrap();

    // Vocab lookup matches corrected constants.
    assert_eq!(tok.token_id("<|transcribe|>"), Some(50360));
    assert_eq!(tok.token_id("<|translate|>"), Some(50359));
    assert_eq!(tok.token_id("<|notimestamps|>"), Some(50364));
    assert_eq!(tok.token_id("<|nospeech|>"), Some(50363));

    // Constants match the v3-turbo vocab.
    assert_eq!(NO_TIMESTAMPS_TOKEN, 50364);
    assert_eq!(NO_SPEECH_TOKEN, 50363);

    // 50364 (notimestamps) is NOT a timestamp token.
    assert!(
        !tok.is_timestamp(50364),
        "50364 is notimestamps, not a timestamp"
    );
    // 50365 (<|0.00|>) IS the first timestamp token.
    assert!(
        tok.is_timestamp(50365),
        "50365 is the first timestamp in v3-turbo"
    );
    assert_eq!(tok.timestamp_value(50365), Some(0.0));
    assert_eq!(tok.timestamp_value(50366), Some(0.02));
}

// ---------------------------------------------------------------------------
// Special token ID constants
// ---------------------------------------------------------------------------

/// All special token constants must have consistent ordering:
/// EOT < SOT < LANGUAGE_TOKEN_START <= LANGUAGE_TOKEN_END < NO_SPEECH < NO_TIMESTAMPS < TIMESTAMP_BEGIN
#[test]
fn test_special_token_constant_ordering() {
    assert!(EOT_TOKEN < SOT_TOKEN);
    assert!(SOT_TOKEN < LANGUAGE_TOKEN_START);
    assert!(LANGUAGE_TOKEN_START <= LANGUAGE_TOKEN_END);
    assert!(LANGUAGE_TOKEN_END < NO_SPEECH_TOKEN);
    assert!(NO_SPEECH_TOKEN < NO_TIMESTAMPS_TOKEN);
    assert!(NO_TIMESTAMPS_TOKEN < TIMESTAMP_BEGIN);
}

#[test]
fn test_eot_token_value() {
    assert_eq!(EOT_TOKEN, 50257);
}

#[test]
fn test_sot_token_value() {
    assert_eq!(SOT_TOKEN, 50258);
}

#[test]
fn test_no_timestamps_token_value() {
    assert_eq!(NO_TIMESTAMPS_TOKEN, 50364);
}

#[test]
fn test_no_speech_token_value() {
    assert_eq!(NO_SPEECH_TOKEN, 50363);
}

#[test]
fn test_language_token_range_spans_100_languages() {
    // Whisper supports 100 languages: IDs 50259-50358 inclusive.
    let count = LANGUAGE_TOKEN_END - LANGUAGE_TOKEN_START + 1;
    assert_eq!(count, 100);
}

#[test]
fn test_timestamp_begin_follows_no_timestamps() {
    assert_eq!(TIMESTAMP_BEGIN, NO_TIMESTAMPS_TOKEN + 1);
}

#[test]
fn test_default_no_speech_threshold() {
    assert!((DEFAULT_NO_SPEECH_THRESHOLD - 0.6).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// Vocab size validation
// ---------------------------------------------------------------------------

#[test]
fn test_vocab_size_single_entry() {
    let json = serde_json::json!({"a": 0}).to_string();
    let tok = WhisperTokenizer::from_vocab_str(&json).unwrap();
    assert_eq!(tok.vocab_size(), 1);
}

#[test]
fn test_vocab_size_gap_in_ids() {
    // IDs 0 and 5 present, so vocab_size = max(5) + 1 = 6.
    let json = serde_json::json!({"a": 0, "b": 5}).to_string();
    let tok = WhisperTokenizer::from_vocab_str(&json).unwrap();
    assert_eq!(tok.vocab_size(), 6);
}

#[test]
fn test_vocab_size_with_special_tokens() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Highest ID in test_vocab_json is 50364 (notimestamps), so vocab_size = 50365.
    assert_eq!(tok.vocab_size(), 50365);
}

// ---------------------------------------------------------------------------
// token_id / token_str edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_token_id_missing_token() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    assert_eq!(tok.token_id("nonexistent"), None);
}

#[test]
fn test_token_str_out_of_range() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Way beyond vocab size.
    assert_eq!(tok.token_str(999999), None);
}

#[test]
fn test_token_str_at_boundary() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Exactly at vocab_size should be None.
    assert_eq!(tok.token_str(tok.vocab_size()), None);
    // One before should be a valid (possibly empty) string.
    assert!(tok.token_str(tok.vocab_size() - 1).is_some());
}

#[test]
fn test_token_id_and_str_roundtrip() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // "hello" → 0, and id 0 → "hello"
    let id = tok.token_id("hello").unwrap();
    let s = tok.token_str(id).unwrap();
    assert_eq!(s, "hello");
}

// ---------------------------------------------------------------------------
// is_special boundary tests
// ---------------------------------------------------------------------------

#[test]
fn test_is_special_boundary_just_below() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    assert!(!tok.is_special(EOT_TOKEN - 1));
}

#[test]
fn test_is_special_at_eot() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    assert!(tok.is_special(EOT_TOKEN));
}

#[test]
fn test_is_special_at_sot() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    assert!(tok.is_special(SOT_TOKEN));
}

#[test]
fn test_is_special_language_tokens() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // All language tokens (50259-50358) should be special.
    assert!(tok.is_special(LANGUAGE_TOKEN_START));
    assert!(tok.is_special(LANGUAGE_TOKEN_END));
    // Mid-range language token.
    assert!(tok.is_special(50300));
}

#[test]
fn test_is_special_at_max_usize() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    assert!(tok.is_special(usize::MAX));
}

// ---------------------------------------------------------------------------
// is_timestamp boundary tests
// ---------------------------------------------------------------------------

#[test]
fn test_is_timestamp_boundary_just_below() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // 50364 is NO_TIMESTAMPS, NOT a timestamp.
    assert!(!tok.is_timestamp(TIMESTAMP_BEGIN - 1));
}

#[test]
fn test_is_timestamp_at_begin() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    assert!(tok.is_timestamp(TIMESTAMP_BEGIN));
}

#[test]
fn test_is_timestamp_large_value() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Any value >= TIMESTAMP_BEGIN is a timestamp.
    assert!(tok.is_timestamp(100000));
}

// ---------------------------------------------------------------------------
// timestamp_value edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_value_below_begin_returns_none() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Every token below TIMESTAMP_BEGIN should return None.
    for id in [0, 100, 50257, 50258, 50363, 50364] {
        assert_eq!(
            tok.timestamp_value(id),
            None,
            "token {id} should not have a timestamp value"
        );
    }
}

#[test]
fn test_timestamp_value_fine_resolution() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // 0.02s resolution means each offset step = 0.02s.
    for offset in 0..=10 {
        let id = TIMESTAMP_BEGIN + offset;
        let expected = offset as f64 * 0.02;
        let actual = tok.timestamp_value(id).unwrap();
        assert!(
            (actual - expected).abs() < 1e-10,
            "offset {offset}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn test_timestamp_value_at_30_seconds() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // 30.00s = 1500 steps * 0.02s/step.
    let id = TIMESTAMP_BEGIN + 1500;
    let ts = tok.timestamp_value(id).unwrap();
    assert!((ts - 30.0).abs() < 1e-10);
}

#[test]
fn test_timestamp_value_beyond_30_seconds() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Timestamps beyond 30s are valid (no upper bound enforced).
    let id = TIMESTAMP_BEGIN + 2000;
    let ts = tok.timestamp_value(id).unwrap();
    assert!((ts - 40.0).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// Language token mapping
// ---------------------------------------------------------------------------

/// Build a vocab with all 100 language tokens to test the full range.
fn vocab_with_all_languages() -> String {
    let mut map = serde_json::Map::new();
    map.insert("hello".to_string(), serde_json::json!(0));
    // Add 100 language tokens: <|en|> at 50259, <|zh|>, <|de|>, ... up to 50358.
    let langs = [
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar",
        "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu",
        "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa",
        "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn",
        "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
        "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
        "mt", "sa", "lb", "nn", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw",
        "su", "yue",
    ];
    for (i, lang) in langs.iter().enumerate() {
        let key = format!("<|{lang}|>");
        map.insert(key, serde_json::json!(50259 + i));
    }
    // Add standard special tokens.
    map.insert("<|endoftext|>".to_string(), serde_json::json!(50257));
    map.insert(
        "<|startoftranscript|>".to_string(),
        serde_json::json!(50258),
    );
    map.insert("<|notimestamps|>".to_string(), serde_json::json!(50364));
    serde_json::Value::Object(map).to_string()
}

#[test]
fn test_language_token_all_100_languages() {
    let tok = WhisperTokenizer::from_vocab_str(&vocab_with_all_languages()).unwrap();
    assert_eq!(tok.language_token("en"), Some(50259));
    assert_eq!(tok.language_token("zh"), Some(50260));
    assert_eq!(tok.language_token("yue"), Some(50358));
}

#[test]
fn test_language_token_not_in_vocab() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Languages not in the test vocab return None.
    assert_eq!(tok.language_token("xx"), None);
    assert_eq!(tok.language_token(""), None);
    assert_eq!(tok.language_token("english"), None);
}

#[test]
fn test_language_token_boundary_ids() {
    let tok = WhisperTokenizer::from_vocab_str(&vocab_with_all_languages()).unwrap();
    // First language: en = LANGUAGE_TOKEN_START = 50259
    assert_eq!(tok.language_token("en"), Some(LANGUAGE_TOKEN_START));
    // Last language (100th): yue = LANGUAGE_TOKEN_END = 50358
    assert_eq!(tok.language_token("yue"), Some(LANGUAGE_TOKEN_END));
}

// ---------------------------------------------------------------------------
// Special token filtering in decode
// ---------------------------------------------------------------------------

#[test]
fn test_decode_filters_all_special_token_types() {
    let vocab = serde_json::json!({
        "hello": 0,
        "<|endoftext|>": 50257,
        "<|startoftranscript|>": 50258,
        "<|en|>": 50259,
        "<|translate|>": 50359,
        "<|transcribe|>": 50360,
        "<|startoflm|>": 50361,
        "<|startofprev|>": 50362,
        "<|nospeech|>": 50363,
        "<|notimestamps|>": 50364,
        "<|0.00|>": 50365,
        "<|0.02|>": 50366,
    })
    .to_string();
    let tok = WhisperTokenizer::from_vocab_str(&vocab).unwrap();

    // Every special token should be filtered; only token 0 ("hello") remains.
    let text = tok
        .decode(&[50257, 50258, 50259, 50359, 50360, 50361, 50362, 50363, 50364, 50365, 50366, 0])
        .unwrap();
    assert_eq!(text, "hello");
}

#[test]
fn test_decode_only_special_tokens_returns_empty() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    let text = tok.decode(&[50257, 50258, 50259, 50360, 50364]).unwrap();
    assert_eq!(text, "");
}

// ---------------------------------------------------------------------------
// decode_with_timestamps edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_decode_with_timestamps_empty_input() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    let segments = tok.decode_with_timestamps(&[]).unwrap();
    assert!(segments.is_empty());
}

#[test]
fn test_decode_with_timestamps_only_special_tokens() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Only SOT + language + transcribe + notimestamps + EOT, no text.
    let segments = tok
        .decode_with_timestamps(&[50258, 50259, 50360, 50364, 50257])
        .unwrap();
    assert!(segments.is_empty());
}

#[test]
fn test_decode_with_timestamps_trailing_text_no_end() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Start timestamp, text, but no end timestamp — should produce segment with start but no end.
    let segments = tok.decode_with_timestamps(&[50365, 0]).unwrap();
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].text, "hello");
    assert!((segments[0].start.unwrap() - 0.0).abs() < 1e-10);
    assert_eq!(segments[0].end, None);
}

#[test]
fn test_decode_with_timestamps_text_before_any_timestamp() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Text appears before any timestamp → segment with no start/end.
    // Then a timestamped segment follows.
    let segments = tok
        .decode_with_timestamps(&[0, 50365, 1, 50415])
        .unwrap();
    assert_eq!(segments.len(), 2);
    // First segment: text before timestamps.
    assert_eq!(segments[0].text, "hello");
    assert_eq!(segments[0].start, None);
    assert_eq!(segments[0].end, None);
    // Second segment: timestamped.
    assert_eq!(segments[1].text, " world");
    assert!((segments[1].start.unwrap() - 0.0).abs() < 1e-10);
    assert!((segments[1].end.unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_decode_with_timestamps_adjacent_timestamps_no_text() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Two adjacent timestamp pairs with no text between them — produces empty text segment.
    let segments = tok.decode_with_timestamps(&[50365, 50415]).unwrap();
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].text, "");
    assert!((segments[0].start.unwrap() - 0.0).abs() < 1e-10);
    assert!((segments[0].end.unwrap() - 1.0).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// Unicode and multilingual text handling
// ---------------------------------------------------------------------------

#[test]
fn test_decode_unicode_bytes() {
    // Build a vocab that encodes UTF-8 bytes for a unicode character.
    // The character 'e' with acute accent (U+00E9, "e\u{0301}" in NFD or "\u{00E9}" in NFC)
    // In UTF-8: 0xC3 0xA9.
    // GPT-2 byte encoder: 0xC3 maps to char 0xC3 (latin-1 range), 0xA9 maps to char 0xA9 (latin-1).
    let encoder = build_byte_encoder();
    let ch_c3 = encoder[&0xC3];
    let ch_a9 = encoder[&0xA9];
    let token_str: String = vec![ch_c3, ch_a9].into_iter().collect();

    let mut map = serde_json::Map::new();
    map.insert(token_str, serde_json::json!(0));
    let json = serde_json::Value::Object(map).to_string();

    let tok = WhisperTokenizer::from_vocab_str(&json).unwrap();
    let decoded = tok.decode(&[0]).unwrap();
    assert_eq!(decoded, "\u{00E9}"); // "e-acute"
}

#[test]
fn test_decode_chinese_character() {
    // Chinese character (U+4F60, "ni3" meaning "you") in UTF-8: 0xE4 0xBD 0xA0
    let encoder = build_byte_encoder();
    let ch_e4 = encoder[&0xE4];
    let ch_bd = encoder[&0xBD];
    let ch_a0 = encoder[&0xA0];
    let token_str: String = vec![ch_e4, ch_bd, ch_a0].into_iter().collect();

    let mut map = serde_json::Map::new();
    map.insert(token_str, serde_json::json!(0));
    let json = serde_json::Value::Object(map).to_string();

    let tok = WhisperTokenizer::from_vocab_str(&json).unwrap();
    let decoded = tok.decode(&[0]).unwrap();
    assert_eq!(decoded, "\u{4F60}");
}

#[test]
fn test_decode_emoji_bytes() {
    // Emoji (U+1F600, grinning face) in UTF-8: 0xF0 0x9F 0x98 0x80
    let encoder = build_byte_encoder();
    let chars: Vec<char> = [0xF0u8, 0x9F, 0x98, 0x80]
        .iter()
        .map(|b| encoder[b])
        .collect();
    let token_str: String = chars.into_iter().collect();

    let mut map = serde_json::Map::new();
    map.insert(token_str, serde_json::json!(0));
    let json = serde_json::Value::Object(map).to_string();

    let tok = WhisperTokenizer::from_vocab_str(&json).unwrap();
    let decoded = tok.decode(&[0]).unwrap();
    assert_eq!(decoded, "\u{1F600}");
}

// ---------------------------------------------------------------------------
// can_encode state
// ---------------------------------------------------------------------------

#[test]
fn test_can_encode_false_for_vocab_only() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    assert!(!tok.can_encode());
}

#[test]
fn test_can_encode_true_after_merges() {
    let vocab = serde_json::json!({"a": 0, "b": 1, "ab": 2}).to_string();
    let merges = "a b\n";
    let tok = WhisperTokenizer::from_vocab_and_merges(&vocab, merges).unwrap();
    assert!(tok.can_encode());
}

#[test]
fn test_can_encode_false_with_empty_merges_text() {
    let vocab = serde_json::json!({"a": 0}).to_string();
    let merges = "";
    let tok = WhisperTokenizer::from_vocab_and_merges(&vocab, merges).unwrap();
    assert!(!tok.can_encode());
}

// ---------------------------------------------------------------------------
// Encode/decode roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_encode_decode_roundtrip_with_space() {
    let vocab = serde_json::json!({
        "h": 0, "e": 1, "l": 2, "o": 3,
        "he": 4, "ll": 5, "hello": 6,
        "Ġ": 7, "w": 8, "r": 9, "d": 10,
        "Ġw": 11, "or": 12, "ld": 13, "Ġworld": 14, "world": 15,
    })
    .to_string();
    let merges = "#version: 0.2\nh e\nl l\nhe ll\nhell o\nĠ w\no r\nl d\nor ld\nĠw orld\n";
    let tok = WhisperTokenizer::from_vocab_and_merges(&vocab, merges).unwrap();

    let original = "hello world";
    let ids = tok.encode(original).unwrap();
    let decoded = tok.decode(&ids).unwrap();
    assert_eq!(decoded, original);
}

// ---------------------------------------------------------------------------
// Error conditions
// ---------------------------------------------------------------------------

#[test]
fn test_invalid_json_array() {
    // JSON array instead of object.
    let result = WhisperTokenizer::from_vocab_str("[1, 2, 3]");
    assert!(result.is_err());
}

#[test]
fn test_decode_token_at_exact_vocab_boundary() {
    let json = serde_json::json!({"a": 0, "b": 1, "c": 2}).to_string();
    let tok = WhisperTokenizer::from_vocab_str(&json).unwrap();
    assert_eq!(tok.vocab_size(), 3);
    // Token 2 is the last valid non-special token.
    assert!(tok.decode(&[2]).is_ok());
    // Token 3 is out of range and not special.
    assert!(tok.decode(&[3]).is_err());
}

// ---------------------------------------------------------------------------
// Byte encoder/decoder completeness
// ---------------------------------------------------------------------------

#[test]
fn test_byte_encoder_no_duplicate_chars() {
    let encoder = build_byte_encoder();
    let mut seen = std::collections::HashSet::new();
    for &ch in encoder.values() {
        assert!(seen.insert(ch), "duplicate char mapping: {ch}");
    }
}

#[test]
fn test_byte_decoder_covers_all_encoder_chars() {
    let encoder = build_byte_encoder();
    let decoder = build_byte_decoder();
    for &ch in encoder.values() {
        assert!(
            decoder.contains_key(&ch),
            "encoder char {ch} not in decoder"
        );
    }
}

#[test]
fn test_byte_encoder_control_chars_remapped() {
    let encoder = build_byte_encoder();
    // Control characters (0x00-0x20 minus space) should NOT map to themselves.
    for b in 0x00u8..0x20u8 {
        let ch = encoder[&b];
        assert_ne!(
            ch,
            char::from(b),
            "control byte {b:#04x} should be remapped"
        );
    }
    // DEL (0x7F) should also be remapped.
    let del_char = encoder[&0x7F];
    assert_ne!(del_char, '\x7F', "DEL should be remapped");
}

// ---------------------------------------------------------------------------
// DecodeConfig construction and initial tokens
// ---------------------------------------------------------------------------

#[test]
fn test_decode_config_default_initial_tokens() {
    let config = crate::DecodeConfig::default();
    // Default: SOT + en + transcribe + notimestamps
    assert_eq!(config.initial_tokens, vec![50258, 50259, 50360, 50364]);
}

#[test]
fn test_decode_config_default_suppress_tokens_empty() {
    let config = crate::DecodeConfig::default();
    assert!(config.suppress_tokens.is_empty());
}

#[test]
fn test_decode_config_with_suppress_tokens() {
    let config = crate::DecodeConfig::default().with_suppress_tokens(vec![0, 1, 50257]);
    assert_eq!(config.suppress_tokens, vec![0, 1, 50257]);
}

#[test]
fn test_decode_config_with_custom_initial_tokens_for_translate() {
    // For translation: SOT + source_lang + translate + notimestamps
    let config =
        crate::DecodeConfig::default().with_initial_tokens(vec![50258, 50260, 50359, 50364]);
    assert_eq!(config.initial_tokens, vec![50258, 50260, 50359, 50364]);
}

#[test]
fn test_decode_config_with_timestamps_initial_tokens() {
    // With timestamps: SOT + lang + transcribe (no notimestamps token)
    let config = crate::DecodeConfig::default().with_initial_tokens(vec![50258, 50259, 50360]);
    assert_eq!(config.initial_tokens, vec![50258, 50259, 50360]);
    // notimestamps is not in the list, so timestamps will be generated.
}

#[test]
fn test_decode_config_validation_rejects_empty_initial_tokens() {
    let config = crate::DecodeConfig::default().with_initial_tokens(vec![]);
    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn test_decode_config_validation_passes_default() {
    let config = crate::DecodeConfig::default();
    assert!(config.validate().is_ok());
}

// ---------------------------------------------------------------------------
// DecodedSegment structure
// ---------------------------------------------------------------------------

#[test]
fn test_decoded_segment_equality() {
    let a = DecodedSegment {
        text: "hello".to_string(),
        start: Some(0.0),
        end: Some(1.0),
    };
    let b = DecodedSegment {
        text: "hello".to_string(),
        start: Some(0.0),
        end: Some(1.0),
    };
    assert_eq!(a, b);
}

#[test]
fn test_decoded_segment_inequality_text() {
    let a = DecodedSegment {
        text: "hello".to_string(),
        start: Some(0.0),
        end: Some(1.0),
    };
    let b = DecodedSegment {
        text: "world".to_string(),
        start: Some(0.0),
        end: Some(1.0),
    };
    assert_ne!(a, b);
}

#[test]
fn test_decoded_segment_clone() {
    let a = DecodedSegment {
        text: "test".to_string(),
        start: Some(2.5),
        end: None,
    };
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn test_decoded_segment_debug_format() {
    let seg = DecodedSegment {
        text: "hi".to_string(),
        start: Some(0.0),
        end: Some(1.0),
    };
    let debug = format!("{seg:?}");
    assert!(debug.contains("DecodedSegment"));
    assert!(debug.contains("hi"));
}

// ---------------------------------------------------------------------------
// WhisperTokenizer Clone and Debug
// ---------------------------------------------------------------------------

#[test]
fn test_tokenizer_clone() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    let tok2 = tok.clone();
    assert_eq!(tok2.vocab_size(), tok.vocab_size());
    assert_eq!(tok2.token_id("hello"), tok.token_id("hello"));
}

#[test]
fn test_tokenizer_debug() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    let debug = format!("{tok:?}");
    assert!(debug.contains("WhisperTokenizer"));
}

// ---------------------------------------------------------------------------
// Decode with interleaved special and normal tokens
// ---------------------------------------------------------------------------

#[test]
fn test_decode_alternating_special_and_normal() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Alternating pattern: special, normal, special, normal, ...
    let text = tok.decode(&[50258, 0, 50259, 1, 50257]).unwrap();
    assert_eq!(text, "hello world");
}

#[test]
fn test_decode_repeated_same_token() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    let text = tok.decode(&[0, 0, 0]).unwrap();
    assert_eq!(text, "hellohellohello");
}

// ---------------------------------------------------------------------------
// Sparse vocab (IDs with gaps)
// ---------------------------------------------------------------------------

#[test]
fn test_sparse_vocab_decode() {
    // Vocab with IDs 0 and 100 — gap between them filled with empty strings.
    let json = serde_json::json!({"hi": 0, "Ġthere": 100}).to_string();
    let tok = WhisperTokenizer::from_vocab_str(&json).unwrap();
    assert_eq!(tok.vocab_size(), 101);
    let text = tok.decode(&[0, 100]).unwrap();
    assert_eq!(text, "hi there");
}

#[test]
fn test_sparse_vocab_gap_id_decodes_empty() {
    // Token ID 50 is in range but has no mapping (empty string in id_to_token).
    let json = serde_json::json!({"hi": 0, "lo": 100}).to_string();
    let tok = WhisperTokenizer::from_vocab_str(&json).unwrap();
    // ID 50 is in range [0, 101) but maps to empty string.
    let text = tok.decode(&[0, 50, 100]).unwrap();
    // The gap token decodes to empty (no chars), so result is "hilo".
    assert_eq!(text, "hilo");
}

// ---------------------------------------------------------------------------
// Timestamp token handling (additional edge cases)
// ---------------------------------------------------------------------------

#[test]
fn test_timestamp_value_at_token_zero_returns_none() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Token 0 is a regular token, not a timestamp.
    assert_eq!(tok.timestamp_value(0), None);
}

#[test]
fn test_timestamp_resolution_002_seconds() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Verify resolution by checking consecutive timestamps differ by 0.02.
    let t0 = tok.timestamp_value(TIMESTAMP_BEGIN).unwrap();
    let t1 = tok.timestamp_value(TIMESTAMP_BEGIN + 1).unwrap();
    let diff = t1 - t0;
    assert!(
        (diff - 0.02).abs() < 1e-10,
        "consecutive timestamps should differ by 0.02s, got {diff}"
    );
}

#[test]
fn test_timestamp_value_consistency_across_range() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Check several timestamp values for consistency.
    for offset in [0, 1, 50, 100, 500, 1000, 1500] {
        let id = TIMESTAMP_BEGIN + offset;
        let expected = offset as f64 * 0.02;
        let actual = tok.timestamp_value(id).unwrap();
        assert!(
            (actual - expected).abs() < 1e-10,
            "offset {offset}: expected {expected}, got {actual}"
        );
    }
}

// ---------------------------------------------------------------------------
// Segment decoding (complex multi-segment patterns)
// ---------------------------------------------------------------------------

#[test]
fn test_decode_with_timestamps_three_segments() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Three segments: <|0.00|> hello <|1.00|> <|1.00|> world <|2.00|> <|2.00|> the <|3.00|>
    let segments = tok
        .decode_with_timestamps(&[
            50365, 0,     // 0.00s start, "hello"
            50415, // 1.00s end
            50415, 1,     // 1.00s start, " world"
            50465, // 2.00s end
            50465, 3,     // 2.00s start, "the"
            50515, // 3.00s end
        ])
        .unwrap();
    assert_eq!(segments.len(), 3);
    assert_eq!(segments[0].text, "hello");
    assert_eq!(segments[1].text, " world");
    assert_eq!(segments[2].text, "the");
    assert!((segments[0].start.unwrap() - 0.0).abs() < 1e-10);
    assert!((segments[0].end.unwrap() - 1.0).abs() < 1e-10);
    assert!((segments[1].start.unwrap() - 1.0).abs() < 1e-10);
    assert!((segments[1].end.unwrap() - 2.0).abs() < 1e-10);
    assert!((segments[2].start.unwrap() - 2.0).abs() < 1e-10);
    assert!((segments[2].end.unwrap() - 3.0).abs() < 1e-10);
}

#[test]
fn test_decode_with_timestamps_mixed_special_and_text() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // SOT + lang + transcribe + timestamp + text + timestamp + EOT
    let segments = tok
        .decode_with_timestamps(&[50258, 50259, 50360, 50365, 0, 50415, 50257])
        .unwrap();
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].text, "hello");
    assert!((segments[0].start.unwrap() - 0.0).abs() < 1e-10);
    assert!((segments[0].end.unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_decode_with_timestamps_only_timestamps_no_text_multiple() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    // Pairs of timestamps with no text between: <|0.00|> <|1.00|> <|1.00|> <|2.00|>
    let segments = tok
        .decode_with_timestamps(&[50365, 50415, 50415, 50465])
        .unwrap();
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].text, "");
    assert_eq!(segments[1].text, "");
}

// ---------------------------------------------------------------------------
// Special token ID relationships
// ---------------------------------------------------------------------------

#[test]
fn test_language_token_start_is_english() {
    // First language token should be English.
    assert_eq!(LANGUAGE_TOKEN_START, 50259);
}

#[test]
fn test_sot_immediately_before_language_tokens() {
    // SOT (50258) + 1 = LANGUAGE_TOKEN_START (50259)
    assert_eq!(SOT_TOKEN + 1, LANGUAGE_TOKEN_START);
}

#[test]
fn test_all_language_tokens_are_special() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    for id in LANGUAGE_TOKEN_START..=LANGUAGE_TOKEN_END {
        assert!(
            tok.is_special(id),
            "language token {id} should be special"
        );
    }
}

#[test]
fn test_all_language_tokens_are_not_timestamps() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    for id in LANGUAGE_TOKEN_START..=LANGUAGE_TOKEN_END {
        assert!(
            !tok.is_timestamp(id),
            "language token {id} should not be a timestamp"
        );
    }
}

// ---------------------------------------------------------------------------
// Encode error conditions
// ---------------------------------------------------------------------------

#[test]
fn test_encode_without_merges_returns_error() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    let result = tok.encode("hello");
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Token string lookup for special tokens
// ---------------------------------------------------------------------------

#[test]
fn test_token_str_for_special_tokens() {
    let vocab = serde_json::json!({
        "hello": 0,
        "<|endoftext|>": 50257,
        "<|startoftranscript|>": 50258,
        "<|en|>": 50259,
    })
    .to_string();
    let tok = WhisperTokenizer::from_vocab_str(&vocab).unwrap();
    assert_eq!(tok.token_str(50257), Some("<|endoftext|>"));
    assert_eq!(tok.token_str(50258), Some("<|startoftranscript|>"));
    assert_eq!(tok.token_str(50259), Some("<|en|>"));
}

#[path = "tokenizer_bpe_tests.rs"]
mod bpe_tests;
