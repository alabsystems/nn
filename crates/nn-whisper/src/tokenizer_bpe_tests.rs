// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for BPE encoding in Whisper tokenizer.

use super::bpe::{bpe_pair_key, parse_merges, pre_tokenize};
use super::*;

/// Build a minimal vocab + merges for encode testing.
///
/// Vocab maps individual GPT-2 byte-encoded characters and some merged tokens.
/// Merges define how characters combine: "h e" → "he", "l l" → "ll", etc.
fn test_encode_setup() -> WhisperTokenizer {
    // We need GPT-2 byte-encoded tokens in the vocab.
    // For ASCII letters, the byte encoder maps them to themselves.
    // For space (0x20), GPT-2 maps to 'Ġ' (U+0120).
    let vocab = serde_json::json!({
        "h": 0,
        "e": 1,
        "l": 2,
        "o": 3,
        "he": 4,
        "ll": 5,
        "hello": 6,
        "Ġ": 7,       // space
        "w": 8,
        "r": 9,
        "d": 10,
        "Ġw": 11,
        "or": 12,
        "ld": 13,
        "Ġworld": 14,
        "world": 15,
        "1": 16,
        "2": 17,
        "3": 18,
        ",": 19,
        ".": 20,
    })
    .to_string();

    let merges = "\
#version: 0.2
h e
l l
he ll
hell o
Ġ w
o r
l d
or ld
Ġw orld
";

    WhisperTokenizer::from_vocab_and_merges(&vocab, merges).unwrap()
}

fn test_vocab_json() -> String {
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
fn test_can_encode_without_merges() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    assert!(!tok.can_encode());
}

#[test]
fn test_can_encode_with_merges() {
    let tok = test_encode_setup();
    assert!(tok.can_encode());
}

#[test]
fn test_encode_requires_merges() {
    let tok = WhisperTokenizer::from_vocab_str(&test_vocab_json()).unwrap();
    let result = tok.encode("hello");
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("BPE merges"), "expected merges error: {err}");
}

#[test]
fn test_encode_single_word() {
    let tok = test_encode_setup();
    // "hello" → pre-tokenize to ["hello"]
    // Byte-encode: "hello" → "hello" (ASCII maps to itself)
    // BPE: h,e,l,l,o → he,l,l,o → he,ll,o → hell,o → hello
    let ids = tok.encode("hello").unwrap();
    assert_eq!(ids, vec![6]); // "hello" = 6
}

#[test]
fn test_encode_word_with_space() {
    let tok = test_encode_setup();
    // " world" → pre-tokenize to [" world"]
    // Byte-encode: space→Ġ, w→w, o→o, r→r, l→l, d→d → "Ġworld"
    // BPE: Ġ,w,o,r,l,d → Ġw,o,r,l,d → Ġw,or,l,d → Ġw,or,ld → Ġw,orld → Ġworld
    let ids = tok.encode(" world").unwrap();
    assert_eq!(ids, vec![14]); // "Ġworld" = 14
}

#[test]
fn test_encode_decode_roundtrip() {
    let tok = test_encode_setup();
    let original = "hello";
    let ids = tok.encode(original).unwrap();
    let decoded = tok.decode(&ids).unwrap();
    assert_eq!(decoded, original);
}

#[test]
fn test_encode_empty_string() {
    let tok = test_encode_setup();
    let ids = tok.encode("").unwrap();
    assert!(ids.is_empty());
}

#[test]
fn test_encode_digits() {
    let tok = test_encode_setup();
    // "123" → pre-tokenize to ["123"]
    // Digits are single characters (no merges defined), so each stays individual.
    let ids = tok.encode("123").unwrap();
    assert_eq!(ids, vec![16, 17, 18]); // "1"=16, "2"=17, "3"=18
}

#[test]
fn test_encode_punctuation_split() {
    let tok = test_encode_setup();
    // "," and "." are single-character punctuation tokens.
    let ids = tok.encode(",").unwrap();
    assert_eq!(ids, vec![19]);
    let ids = tok.encode(".").unwrap();
    assert_eq!(ids, vec![20]);
}

#[test]
fn test_parse_merges_empty() {
    let ranks = parse_merges("").unwrap();
    assert!(ranks.is_empty());
}

#[test]
fn test_parse_merges_header_only() {
    let ranks = parse_merges("#version: 0.2\n").unwrap();
    assert!(ranks.is_empty());
}

#[test]
fn test_parse_merges_basic() {
    let text = "#version: 0.2\nh e\nl l\nhe ll\n";
    let ranks = parse_merges(text).unwrap();
    assert_eq!(ranks.len(), 3);
    let mut key = String::new();
    bpe_pair_key(&mut key, "h", "e");
    assert_eq!(ranks[&key], 0);
    bpe_pair_key(&mut key, "l", "l");
    assert_eq!(ranks[&key], 1);
    bpe_pair_key(&mut key, "he", "ll");
    assert_eq!(ranks[&key], 2);
}

#[test]
fn test_parse_merges_bad_line() {
    let result = parse_merges("hello");
    assert!(result.is_err());
}

#[test]
fn test_pre_tokenize_simple() {
    let words = pre_tokenize("hello world");
    assert_eq!(words, vec!["hello", " world"]);
}

#[test]
fn test_pre_tokenize_leading_space() {
    let words = pre_tokenize(" hello");
    assert_eq!(words, vec![" hello"]);
}

#[test]
fn test_pre_tokenize_digits_separate() {
    let words = pre_tokenize("abc123");
    assert_eq!(words, vec!["abc", "123"]);
}

#[test]
fn test_pre_tokenize_punctuation_separate() {
    let words = pre_tokenize("hello,world");
    assert_eq!(words, vec!["hello", ",", "world"]);
}

#[test]
fn test_pre_tokenize_contraction() {
    let words = pre_tokenize("don't");
    assert_eq!(words, vec!["don", "'t"]);
}

#[test]
fn test_pre_tokenize_empty() {
    let words = pre_tokenize("");
    assert!(words.is_empty());
}

#[test]
fn test_bpe_single_char() {
    let tok = test_encode_setup();
    let result = tok.bpe("h");
    assert_eq!(result, vec!["h"]);
}

#[test]
fn test_bpe_merge_pair() {
    let tok = test_encode_setup();
    // h,e → he (merge rank 0)
    let result = tok.bpe("he");
    assert_eq!(result, vec!["he"]);
}

#[test]
fn test_bpe_full_word() {
    let tok = test_encode_setup();
    // h,e,l,l,o → he,l,l,o → he,ll,o → hell,o → hello
    let result = tok.bpe("hello");
    assert_eq!(result, vec!["hello"]);
}

#[test]
fn test_bpe_no_merges_available() {
    let tok = test_encode_setup();
    // "xyz" has no merges defined, stays as individual chars.
    let result = tok.bpe("123");
    assert_eq!(result, vec!["1", "2", "3"]);
}

/// Performance proof: BPE merge loop is O(n log n) where n = initial character count.
///
/// The `bpe()` method uses a priority queue with a doubly-linked list:
///   - `BinaryHeap` (min-heap via reverse `Ord`) holds candidate merges by rank
///   - Parallel arrays (`text`, `prev`, `next`, `alive`) form a doubly-linked list
///   - Each merge: O(log n) heap pop, O(1) linked-list splice, O(log n) re-insert
///   - At most 2n heap insertions total (n initial + n re-inserts from merges)
///
/// Total work: O(n log n) heap operations + O(n) linked-list traversals.
///
/// This test verifies correctness and documents the O(n log n) scaling by
/// measuring timing ratio between small and large inputs.
#[test]
fn test_bpe_nlogn_scaling_documented() {
    // Build a tokenizer with many merge levels to exercise the full loop.
    // Chain: a,b → ab, ab,c → abc, abc,d → abcd, ..., up to 26 chars.
    let mut vocab_entries = Vec::new();
    let mut merges_lines = vec!["#version: 0.2".to_string()];

    // Individual characters a-z
    for (i, c) in ('a'..='z').enumerate() {
        vocab_entries.push(format!("\"{c}\": {i}"));
    }

    // Build merge chain: "a b" → "ab", "ab c" → "abc", etc.
    let mut current = "a".to_string();
    let mut next_id = 26;
    for c in 'b'..='z' {
        let right = c.to_string();
        merges_lines.push(format!("{current} {right}"));
        current = format!("{current}{right}");
        vocab_entries.push(format!("\"{current}\": {next_id}"));
        next_id += 1;
    }

    let vocab_json = format!("{{{}}}", vocab_entries.join(", "));
    let merges_text = merges_lines.join("\n");
    let tok = WhisperTokenizer::from_vocab_and_merges(&vocab_json, &merges_text).unwrap();

    // Small input: 5 characters — merges in 4 steps.
    let small = tok.bpe("abcde");
    assert_eq!(small, vec!["abcde"]);

    // Large input: 26 characters — merges in 25 steps (full chain).
    let large = tok.bpe("abcdefghijklmnopqrstuvwxyz");
    assert_eq!(large, vec!["abcdefghijklmnopqrstuvwxyz"]);

    // Timing comparison: 26-char vs 5-char.
    // With O(n log n), ratio should be ~(26·log26)/(5·log5) ≈ 6.5x.
    let iterations = 100;
    let start_small = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = tok.bpe("abcde");
    }
    let elapsed_small = start_small.elapsed();

    let start_large = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = tok.bpe("abcdefghijklmnopqrstuvwxyz");
    }
    let elapsed_large = start_large.elapsed();

    let ratio = elapsed_large.as_nanos() as f64 / elapsed_small.as_nanos().max(1) as f64;
    eprintln!(
        "BPE scaling: small(n=5)={:.1}ms/{iterations}iter, large(n=26)={:.1}ms/{iterations}iter, \
         ratio={ratio:.1}x (O(n log n) predicts ~{:.1}x)",
        elapsed_small.as_secs_f64() * 1000.0,
        elapsed_large.as_secs_f64() * 1000.0,
        (26.0 * 26.0_f64.ln()) / (5.0 * 5.0_f64.ln())
    );

    // Sanity: ratio should be > 1 (larger input takes longer).
    assert!(ratio > 1.0, "larger BPE input should take longer");
}

/// Verify BPE correctness with an input that has no applicable merges.
/// Every character stays separate — the loop runs n-1 scans but
/// exits without any merges, verifying the early-exit path.
#[test]
fn test_bpe_no_merges_full_scan() {
    let tok = test_encode_setup();
    // "123,." — all single-char tokens with no merge pairs defined.
    let result = tok.bpe("123,.");
    assert_eq!(result, vec!["1", "2", "3", ",", "."]);
}
