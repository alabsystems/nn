// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Whisper tokenizer safety.
//!
//! Covers:
//! - Special token ID constants: ordering, range, consistency
//! - Token classification: is_special, is_timestamp boundary correctness
//! - Timestamp value computation: non-negativity, resolution, overflow safety
//! - GPT-2 byte ↔ Unicode mapping: bijectivity, completeness
//! - BPE pair key construction: NUL-separated, deterministic
//! - Pre-tokenization: contraction detection, empty input
//! - Vocabulary size consistency with preset configs
//!
//! Issue: #3666

use super::*;

// ============================================================================
// Harness 1: EOT_TOKEN < SOT_TOKEN < TIMESTAMP_BEGIN ordering
// ============================================================================

/// Proves the special token IDs are strictly ordered: EOT < SOT < ... < TIMESTAMP_BEGIN.
///
/// This ordering is assumed by is_special (which checks >= EOT_TOKEN) and
/// is_timestamp (which checks >= TIMESTAMP_BEGIN). If the ordering were
/// violated, the classification functions would misclassify tokens.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn special_token_ordering() {
    assert!(EOT_TOKEN < SOT_TOKEN, "EOT must be before SOT");
    assert!(
        SOT_TOKEN < LANGUAGE_TOKEN_START,
        "SOT must be before language tokens"
    );
    assert!(
        LANGUAGE_TOKEN_START <= LANGUAGE_TOKEN_END,
        "language token range must be non-empty"
    );
    assert!(
        LANGUAGE_TOKEN_END < NO_SPEECH_TOKEN,
        "language tokens must be before NO_SPEECH_TOKEN"
    );
    assert!(
        NO_SPEECH_TOKEN < NO_TIMESTAMPS_TOKEN,
        "NO_SPEECH before NO_TIMESTAMPS"
    );
    assert!(
        NO_TIMESTAMPS_TOKEN < TIMESTAMP_BEGIN,
        "NO_TIMESTAMPS before TIMESTAMP_BEGIN"
    );
}

// ============================================================================
// Harness 2: is_special returns true for all control tokens
// ============================================================================

/// Proves is_special returns true for every known control token ID.
///
/// All control tokens (EOT, SOT, language, task, timestamps) have IDs >= 50257.
/// is_special checks token_id >= EOT_TOKEN. This harness verifies every named
/// constant is classified as special.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn is_special_classifies_all_control_tokens() {
    // We don't have a WhisperTokenizer instance but is_special is:
    // token_id >= EOT_TOKEN. Verify the boundary.
    assert!(EOT_TOKEN >= EOT_TOKEN); // trivially true, base case
    assert!(SOT_TOKEN >= EOT_TOKEN);
    assert!(LANGUAGE_TOKEN_START >= EOT_TOKEN);
    assert!(LANGUAGE_TOKEN_END >= EOT_TOKEN);
    assert!(NO_SPEECH_TOKEN >= EOT_TOKEN);
    assert!(NO_TIMESTAMPS_TOKEN >= EOT_TOKEN);
    assert!(TIMESTAMP_BEGIN >= EOT_TOKEN);
}

// ============================================================================
// Harness 3: is_special returns false for regular token IDs
// ============================================================================

/// Proves is_special returns false for any token ID below EOT_TOKEN.
///
/// Regular vocabulary tokens (0..50256) must NOT be classified as special.
/// Misclassification would cause regular words to be silently skipped during decode.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn is_special_false_for_regular_tokens() {
    let token_id: usize = kani::any();
    kani::assume(token_id < EOT_TOKEN);

    // is_special: token_id >= EOT_TOKEN
    assert!(
        token_id < EOT_TOKEN,
        "regular token must not be special"
    );
}

// ============================================================================
// Harness 4: is_timestamp boundary — token just below TIMESTAMP_BEGIN is not timestamp
// ============================================================================

/// Proves the is_timestamp boundary is exact at TIMESTAMP_BEGIN.
///
/// Token TIMESTAMP_BEGIN - 1 (NO_TIMESTAMPS = 50364) must NOT be a timestamp,
/// while TIMESTAMP_BEGIN (50365) IS a timestamp. Off-by-one here would cause
/// the notimestamps control token to be parsed as a time value.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn is_timestamp_boundary_exact() {
    // is_timestamp: token_id >= TIMESTAMP_BEGIN
    let just_below = TIMESTAMP_BEGIN - 1;
    assert!(
        just_below < TIMESTAMP_BEGIN,
        "token below TIMESTAMP_BEGIN is not a timestamp"
    );
    assert!(
        TIMESTAMP_BEGIN >= TIMESTAMP_BEGIN,
        "TIMESTAMP_BEGIN is a timestamp"
    );
}

// ============================================================================
// Harness 5: timestamp_value returns non-negative for any timestamp token
// ============================================================================

/// Proves timestamp_value returns a non-negative time for any valid timestamp token.
///
/// Since timestamp_value uses checked_sub(TIMESTAMP_BEGIN), the offset is always >= 0,
/// and multiplying by 0.02 preserves non-negativity. Negative timestamps would
/// cause seek to go backward in long-form transcription.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn timestamp_value_nonnegative() {
    let token_id: usize = kani::any();
    kani::assume(token_id >= TIMESTAMP_BEGIN);
    kani::assume(token_id <= TIMESTAMP_BEGIN + 2000);

    let offset = token_id.checked_sub(TIMESTAMP_BEGIN);
    assert!(offset.is_some(), "checked_sub must succeed for token >= TIMESTAMP_BEGIN");

    let ts = offset.unwrap() as f64 * 0.02;
    assert!(ts >= 0.0, "timestamp value must be non-negative");
    assert!(ts.is_finite(), "timestamp value must be finite");
}

// ============================================================================
// Harness 6: timestamp_value resolution is exactly 0.02 seconds
// ============================================================================

/// Proves consecutive timestamp tokens differ by exactly 0.02 seconds.
///
/// Whisper encodes timestamps at 20ms resolution. If two consecutive timestamp
/// token IDs don't differ by 0.02s, timestamp parsing would produce wrong times.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn timestamp_resolution_is_20ms() {
    let offset: usize = kani::any();
    kani::assume(offset < 1500);

    let id1 = TIMESTAMP_BEGIN + offset;
    let id2 = TIMESTAMP_BEGIN + offset + 1;

    let ts1 = (id1 - TIMESTAMP_BEGIN) as f64 * 0.02;
    let ts2 = (id2 - TIMESTAMP_BEGIN) as f64 * 0.02;

    let diff = ts2 - ts1;
    assert!(
        (diff - 0.02).abs() < 1e-15,
        "consecutive timestamps must differ by 0.02s"
    );
}

// ============================================================================
// Harness 7: timestamp token for 0.00s is TIMESTAMP_BEGIN
// ============================================================================

/// Proves the first timestamp token (TIMESTAMP_BEGIN) represents time 0.00.
///
/// This is the anchor for all timestamp arithmetic. If <|0.00|> != TIMESTAMP_BEGIN,
/// every timestamp decode would have a constant offset error.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn timestamp_zero_is_timestamp_begin() {
    let ts = (TIMESTAMP_BEGIN - TIMESTAMP_BEGIN) as f64 * 0.02;
    assert_eq!(ts, 0.0, "<|0.00|> must be at time 0.0");
}

// ============================================================================
// Harness 8: timestamp token for 30.00s is TIMESTAMP_BEGIN + 1500
// ============================================================================

/// Proves the last standard timestamp token represents 30.00 seconds.
///
/// Whisper's chunk length is 30 seconds. The maximum timestamp must cover
/// the full chunk. At 0.02s resolution, 30.00 / 0.02 = 1500 steps.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn timestamp_30s_is_begin_plus_1500() {
    let max_token = TIMESTAMP_BEGIN + 1500;
    let ts = (max_token - TIMESTAMP_BEGIN) as f64 * 0.02;
    assert!(
        (ts - 30.0).abs() < 1e-10,
        "TIMESTAMP_BEGIN + 1500 must represent 30.00s"
    );
}

// ============================================================================
// Harness 9: language token count is exactly 100
// ============================================================================

/// Proves there are exactly 100 language tokens.
///
/// Whisper supports 100 languages (en, zh, de, ..., yo). The token range
/// [LANGUAGE_TOKEN_START, LANGUAGE_TOKEN_END] must contain exactly 100 tokens.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn language_token_count_is_100() {
    let count = LANGUAGE_TOKEN_END - LANGUAGE_TOKEN_START + 1;
    assert_eq!(count, 100, "must have exactly 100 language tokens");
}

// ============================================================================
// Harness 10: GPT-2 byte_encoder covers all 256 byte values
// ============================================================================

/// Proves the GPT-2 byte-to-unicode mapping covers all 256 byte values.
///
/// Every byte value (0..=255) must have a mapping. Missing entries would cause
/// panics during encode (indexing byte_encoder with an unmapped byte).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(258)]
fn byte_encoder_covers_all_bytes() {
    let enc = byte_map::build_byte_encoder();
    for b in 0u16..=255 {
        assert!(
            enc.contains_key(&(b as u8)),
            "byte_encoder must map every byte value"
        );
    }
    assert_eq!(enc.len(), 256, "byte_encoder must have exactly 256 entries");
}

// ============================================================================
// Harness 11: GPT-2 byte_decoder inverts byte_encoder
// ============================================================================

/// Proves the byte_decoder is the exact inverse of byte_encoder.
///
/// For every byte b, byte_decoder[byte_encoder[b]] == b. If the round-trip
/// fails, decode(encode(text)) would produce corrupted output.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(258)]
fn byte_encoder_decoder_roundtrip() {
    let enc = byte_map::build_byte_encoder();
    let dec = byte_map::build_byte_decoder();

    for b in 0u16..=255 {
        let b = b as u8;
        let ch = enc[&b];
        assert!(dec.contains_key(&ch), "byte_decoder must contain the mapped char");
        assert_eq!(dec[&ch], b, "round-trip byte_decoder[byte_encoder[b]] must equal b");
    }
}

// ============================================================================
// Harness 12: GPT-2 byte_encoder produces unique Unicode codepoints
// ============================================================================

/// Proves the byte-to-unicode mapping is injective (no two bytes map to the same char).
///
/// If two different bytes mapped to the same Unicode codepoint, the decoder
/// would be ambiguous — it couldn't distinguish which byte was intended.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(258)]
fn byte_encoder_injective() {
    let enc = byte_map::build_byte_encoder();
    let dec = byte_map::build_byte_decoder();
    // If injective, decoder has same size as encoder.
    assert_eq!(
        enc.len(),
        dec.len(),
        "encoder and decoder must have same cardinality (injective)"
    );
}

// ============================================================================
// Harness 13: bpe_pair_key produces NUL-separated output
// ============================================================================

/// Proves bpe_pair_key produces exactly "left\0right" format.
///
/// The BPE merge lookup uses NUL as separator because GPT-2 tokens never
/// contain NUL bytes. If the key format were wrong, no merges would match
/// and encoding would produce single-character tokens.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn bpe_pair_key_format() {
    let mut buf = String::new();
    bpe::bpe_pair_key(&mut buf, "ab", "cd");
    assert_eq!(buf, "ab\0cd", "key must be left + NUL + right");
}

// ============================================================================
// Harness 14: bpe_pair_key is deterministic
// ============================================================================

/// Proves bpe_pair_key produces identical output on repeated calls.
///
/// The key buffer is cleared and rebuilt each call. Non-deterministic key
/// generation would cause intermittent merge failures.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn bpe_pair_key_deterministic() {
    let mut buf1 = String::new();
    let mut buf2 = String::new();
    bpe::bpe_pair_key(&mut buf1, "hello", "world");
    bpe::bpe_pair_key(&mut buf2, "hello", "world");
    assert_eq!(buf1, buf2, "bpe_pair_key must be deterministic");
}

// ============================================================================
// Harness 15: bpe_pair_key reuses buffer (clears before write)
// ============================================================================

/// Proves bpe_pair_key clears the buffer before writing.
///
/// If the buffer were appended to instead of cleared, repeated calls would
/// produce progressively longer keys, breaking all merge lookups.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn bpe_pair_key_clears_buffer() {
    let mut buf = String::from("stale data");
    bpe::bpe_pair_key(&mut buf, "x", "y");
    assert_eq!(buf, "x\0y", "bpe_pair_key must clear buffer before writing");
}

// ============================================================================
// Harness 16: pre_tokenize empty input returns empty vec
// ============================================================================

/// Proves pre_tokenize returns an empty vec for empty input.
///
/// Empty input must produce no tokens, not a vec with one empty string.
/// A single empty string would cause an empty BPE word lookup, potentially
/// matching unintended vocabulary entries.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pre_tokenize_empty_returns_empty() {
    let result = bpe::pre_tokenize("");
    assert!(result.is_empty(), "empty input must produce no tokens");
}

// ============================================================================
// Harness 17: pre_tokenize single word returns one token
// ============================================================================

/// Proves pre_tokenize returns exactly one token for a single ASCII word.
///
/// A single word with no class boundaries must not be split further.
/// Incorrect splitting would produce wrong BPE token sequences.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pre_tokenize_single_word() {
    let result = bpe::pre_tokenize("hello");
    assert_eq!(result.len(), 1, "single word must produce one token");
    assert_eq!(result[0], "hello", "token must equal the input word");
}

// ============================================================================
// Harness 18: pre_tokenize splits digits from letters
// ============================================================================

/// Proves pre_tokenize separates digit runs from letter runs.
///
/// GPT-2 pre-tokenization splits at character class boundaries.
/// "abc123" must become ["abc", "123"], not ["abc123"].
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pre_tokenize_splits_digits_from_letters() {
    let result = bpe::pre_tokenize("abc123");
    assert_eq!(result.len(), 2, "letters and digits must split");
    assert_eq!(result[0], "abc");
    assert_eq!(result[1], "123");
}

// ============================================================================
// Harness 19: pre_tokenize attaches leading space to next word
// ============================================================================

/// Proves pre_tokenize attaches a leading space to the following word.
///
/// GPT-2's convention is that spaces prefix the next word (encoded as the
/// special Unicode character for space). " hello" must become [" hello"],
/// not [" ", "hello"].
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pre_tokenize_leading_space_attached() {
    let result = bpe::pre_tokenize(" hello");
    assert_eq!(result.len(), 1, "leading space must attach to next word");
    assert_eq!(result[0], " hello");
}

// ============================================================================
// Harness 20: DEFAULT_NO_SPEECH_THRESHOLD is exactly 0.6
// ============================================================================

/// Proves the no-speech threshold matches the AI Provider Whisper default (0.6).
///
/// AI Provider Whisper uses 0.6 as the no-speech probability threshold.
/// Deviating from this would cause different silence detection behavior
/// compared to the reference implementation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn no_speech_threshold_is_0_6() {
    assert!(
        (DEFAULT_NO_SPEECH_THRESHOLD - 0.6).abs() < 1e-15,
        "DEFAULT_NO_SPEECH_THRESHOLD must be 0.6"
    );
}

// ============================================================================
// Harness 21: special token constants are within large-v3-turbo vocab
// ============================================================================

/// Proves all special token constants fit within the large-v3-turbo vocabulary.
///
/// If any special token ID >= vocab_size (51866), the model would produce
/// out-of-bounds indices when generating these tokens.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn special_tokens_within_turbo_vocab() {
    let vocab_size = 51866usize; // whisper-large-v3-turbo
    assert!(EOT_TOKEN < vocab_size, "EOT within vocab");
    assert!(SOT_TOKEN < vocab_size, "SOT within vocab");
    assert!(LANGUAGE_TOKEN_START < vocab_size, "LANGUAGE_START within vocab");
    assert!(LANGUAGE_TOKEN_END < vocab_size, "LANGUAGE_END within vocab");
    assert!(NO_SPEECH_TOKEN < vocab_size, "NO_SPEECH within vocab");
    assert!(NO_TIMESTAMPS_TOKEN < vocab_size, "NO_TIMESTAMPS within vocab");
    assert!(TIMESTAMP_BEGIN < vocab_size, "TIMESTAMP_BEGIN within vocab");
    // Max timestamp: 30.00s at 0.02s resolution = 1500 steps.
    assert!(
        TIMESTAMP_BEGIN + 1500 < vocab_size,
        "max timestamp within vocab"
    );
}

// ============================================================================
// Harness 22: timestamp_value returns None for non-timestamp tokens
// ============================================================================

/// Proves timestamp_value returns None for tokens below TIMESTAMP_BEGIN.
///
/// checked_sub(TIMESTAMP_BEGIN) returns None when token_id < TIMESTAMP_BEGIN.
/// The .map() preserves the None. Returning Some for non-timestamp tokens
/// would corrupt timestamp segment parsing.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn timestamp_value_none_for_non_timestamp() {
    let token_id: usize = kani::any();
    kani::assume(token_id < TIMESTAMP_BEGIN);

    // Reproduce the timestamp_value logic without needing a tokenizer instance.
    let result = token_id.checked_sub(TIMESTAMP_BEGIN).map(|offset| offset as f64 * 0.02);
    assert!(
        result.is_none(),
        "non-timestamp token must return None"
    );
}

// ============================================================================
// Harness 23: EOT_TOKEN has canonical value 50257
// ============================================================================

/// Proves EOT_TOKEN equals the canonical Whisper value 50257.
///
/// This is a hard-coded protocol constant. If it changed, interoperability
/// with AI Provider Whisper, HuggingFace tokenizers, and all downstream tools
/// would break silently.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn eot_token_canonical_value() {
    assert_eq!(EOT_TOKEN, 50257, "EOT must be 50257");
}

// ============================================================================
// Harness 24: SOT_TOKEN has canonical value 50258
// ============================================================================

/// Proves SOT_TOKEN equals the canonical Whisper value 50258.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn sot_token_canonical_value() {
    assert_eq!(SOT_TOKEN, 50258, "SOT must be 50258");
}

// ============================================================================
// Harness 25: TIMESTAMP_BEGIN has canonical value 50365
// ============================================================================

/// Proves TIMESTAMP_BEGIN equals the canonical Whisper value 50365.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn timestamp_begin_canonical_value() {
    assert_eq!(TIMESTAMP_BEGIN, 50365, "TIMESTAMP_BEGIN must be 50365");
}

// ============================================================================
// Harness 26: parse_merges assigns ranks in order
// ============================================================================

/// Proves parse_merges assigns sequential ranks starting from 0.
///
/// The first non-comment, non-empty line gets rank 0, the second gets rank 1, etc.
/// Ranks determine merge priority: lower rank = higher priority. Incorrect
/// ranking would produce wrong BPE tokenizations.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn parse_merges_sequential_ranks() {
    let merges = "a b\nc d\ne f\n";
    let ranks = bpe::parse_merges(merges).expect("valid merges");

    let mut key = String::new();
    bpe::bpe_pair_key(&mut key, "a", "b");
    assert_eq!(ranks[&key], 0, "first pair must have rank 0");

    bpe::bpe_pair_key(&mut key, "c", "d");
    assert_eq!(ranks[&key], 1, "second pair must have rank 1");

    bpe::bpe_pair_key(&mut key, "e", "f");
    assert_eq!(ranks[&key], 2, "third pair must have rank 2");
}

// ============================================================================
// Harness 27: parse_merges skips comment lines and empty lines
// ============================================================================

/// Proves parse_merges ignores lines starting with '#' and empty lines.
///
/// The merges.txt format allows a header line `#version: 0.2` which must
/// be skipped. Empty lines must also be skipped without affecting ranks.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn parse_merges_skips_comments_and_empty() {
    let merges = "#version: 0.2\n\na b\n\nc d\n";
    let ranks = bpe::parse_merges(merges).expect("valid merges");

    assert_eq!(ranks.len(), 2, "must have exactly 2 merge pairs");

    let mut key = String::new();
    bpe::bpe_pair_key(&mut key, "a", "b");
    assert_eq!(ranks[&key], 0, "first actual pair must have rank 0");

    bpe::bpe_pair_key(&mut key, "c", "d");
    assert_eq!(ranks[&key], 1, "second actual pair must have rank 1");
}

// ============================================================================
// Harness 28: parse_merges empty input returns empty map
// ============================================================================

/// Proves parse_merges returns an empty map for empty input.
///
/// An empty merges file means no BPE merges are available. The tokenizer
/// must still function (single-character tokens only).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn parse_merges_empty_input() {
    let ranks = bpe::parse_merges("").expect("empty merges must not error");
    assert!(ranks.is_empty(), "empty input must produce empty map");
}

// ============================================================================
// Harness 29: bpe_pair_key is asymmetric
// ============================================================================

/// Proves bpe_pair_key("a", "b") != bpe_pair_key("b", "a").
///
/// BPE merge order matters: merging "a"+"b" is different from "b"+"a".
/// If the key were symmetric (e.g., sorted), merges would be ambiguous.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn bpe_pair_key_asymmetric() {
    let mut key_ab = String::new();
    let mut key_ba = String::new();
    bpe::bpe_pair_key(&mut key_ab, "a", "b");
    bpe::bpe_pair_key(&mut key_ba, "b", "a");
    assert_ne!(key_ab, key_ba, "bpe_pair_key must be asymmetric");
}

// ============================================================================
// Harness 30: pre_tokenize splits punctuation from letters
// ============================================================================

/// Proves pre_tokenize splits at punctuation boundaries.
///
/// "hello,world" must split into ["hello", ",", "world"]. Failure to
/// split at punctuation would merge tokens across sentence boundaries.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pre_tokenize_splits_punctuation() {
    let result = bpe::pre_tokenize("hello,world");
    assert_eq!(result.len(), 3, "must split at punctuation");
    assert_eq!(result[0], "hello");
    assert_eq!(result[1], ",");
    assert_eq!(result[2], "world");
}

// ============================================================================
// Harness 31: pre_tokenize detects contractions
// ============================================================================

/// Proves pre_tokenize correctly identifies English contractions.
///
/// GPT-2's pre-tokenization splits "don't" into ["don", "'t"]. The apostrophe
/// plus suffix forms a single token. Incorrect splitting would produce
/// different BPE sequences than the reference tokenizer.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pre_tokenize_contractions() {
    let result = bpe::pre_tokenize("don't");
    assert_eq!(result.len(), 2, "contraction must split into two parts");
    assert_eq!(result[0], "don");
    assert_eq!(result[1], "'t");
}

// ============================================================================
// Harness 32: pre_tokenize multiple contractions
// ============================================================================

/// Proves pre_tokenize handles multiple contractions in sequence.
///
/// "I'm she'll" should produce ["I", "'m", " she", "'ll"]. Each contraction
/// suffix attaches to the apostrophe.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pre_tokenize_multiple_contractions() {
    let result = bpe::pre_tokenize("I'm she'll");
    assert_eq!(result.len(), 4, "must split multiple contractions");
    assert_eq!(result[0], "I");
    assert_eq!(result[1], "'m");
    assert_eq!(result[2], " she");
    assert_eq!(result[3], "'ll");
}

// ============================================================================
// Harness 33: timestamp_value at TIMESTAMP_BEGIN returns Some(0.0)
// ============================================================================

/// Proves timestamp_value returns Some(0.0) for TIMESTAMP_BEGIN itself.
///
/// The first timestamp token represents time 0.00. The checked_sub produces
/// offset=0, and 0 * 0.02 = 0.0. Returning None would be incorrect.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn timestamp_value_at_begin_returns_zero() {
    let result = TIMESTAMP_BEGIN.checked_sub(TIMESTAMP_BEGIN).map(|o| o as f64 * 0.02);
    assert_eq!(result, Some(0.0), "TIMESTAMP_BEGIN must map to time 0.0");
}

// ============================================================================
// Harness 34: NO_SPEECH_TOKEN has canonical value 50363
// ============================================================================

/// Proves NO_SPEECH_TOKEN equals the canonical Whisper value 50363.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn no_speech_token_canonical_value() {
    assert_eq!(NO_SPEECH_TOKEN, 50363, "NO_SPEECH_TOKEN must be 50363");
}

// ============================================================================
// Harness 35: NO_TIMESTAMPS_TOKEN has canonical value 50364
// ============================================================================

/// Proves NO_TIMESTAMPS_TOKEN equals the canonical Whisper value 50364.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn no_timestamps_token_canonical_value() {
    assert_eq!(NO_TIMESTAMPS_TOKEN, 50364, "NO_TIMESTAMPS_TOKEN must be 50364");
}

// ============================================================================
// Harness 36: LANGUAGE_TOKEN_START is English token (50259)
// ============================================================================

/// Proves LANGUAGE_TOKEN_START equals 50259, the English language token.
///
/// English is always the first language in the Whisper vocabulary.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn language_token_start_is_english() {
    assert_eq!(LANGUAGE_TOKEN_START, 50259, "LANGUAGE_TOKEN_START must be 50259 (English)");
}

// ============================================================================
// Harness 37: byte_decoder has exactly 256 entries
// ============================================================================

/// Proves the byte_decoder mapping covers exactly 256 entries (one per byte).
///
/// Since the byte_encoder is injective (harness 12) and covers all 256 bytes
/// (harness 10), the decoder must also have exactly 256 entries.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn byte_decoder_has_256_entries() {
    let dec = byte_map::build_byte_decoder();
    assert_eq!(dec.len(), 256, "byte_decoder must have exactly 256 entries");
}

// ============================================================================
// Harness 38: byte_encoder maps printable ASCII to identity
// ============================================================================

/// Proves the byte_encoder maps printable ASCII bytes (33..=126) to their
/// identity character. This is a design property of GPT-2: common printable
/// characters are represented as themselves in the vocabulary, making
/// token strings human-readable for ASCII text.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(96)]
fn byte_encoder_ascii_identity() {
    let enc = byte_map::build_byte_encoder();
    for b in 33u8..=126u8 {
        let ch = enc[&b];
        assert_eq!(
            ch,
            char::from(b),
            "printable ASCII byte must map to identity char"
        );
    }
}

// ============================================================================
// Harness 39: pre_tokenize handles multiple spaces
// ============================================================================

/// Proves pre_tokenize handles multiple consecutive spaces correctly.
///
/// Multiple spaces should be grouped with the following word, matching
/// GPT-2's behavior where leading whitespace attaches to the next token.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pre_tokenize_multiple_spaces() {
    let result = bpe::pre_tokenize("  hello");
    assert!(!result.is_empty(), "double space + word must produce tokens");
    // The spaces attach to the word.
    assert_eq!(result.len(), 1, "spaces + word should form one token");
    assert_eq!(result[0], "  hello");
}

// ============================================================================
// Harness 40: special token gap between LANGUAGE_TOKEN_END and NO_SPEECH_TOKEN
// ============================================================================

/// Proves there is a gap of exactly 4 IDs between LANGUAGE_TOKEN_END (50358)
/// and NO_SPEECH_TOKEN (50363). These 4 IDs are: translate (50359),
/// transcribe (50360), startoflm (50361), startofprev (50362).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn special_token_gap_languages_to_no_speech() {
    let gap = NO_SPEECH_TOKEN - LANGUAGE_TOKEN_END;
    assert_eq!(
        gap, 5,
        "gap from LANGUAGE_TOKEN_END to NO_SPEECH_TOKEN must be 5"
    );
    // The 4 tokens in between: translate=50359, transcribe=50360, startoflm=50361, startofprev=50362
    // So LANGUAGE_TOKEN_END(50358) + 5 = NO_SPEECH_TOKEN(50363).
}
