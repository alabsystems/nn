// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPT-2 BPE encoding helpers for Whisper tokenizer.
//!
//! Contains pre-tokenization (word splitting at character class boundaries)
//! and BPE merge parsing. Used by `WhisperTokenizer::encode()`.

use std::collections::HashMap;

use crate::WhisperError;
use nn_core::{Result, TensorError};

/// Build a NUL-separated key for BPE pair lookup: `"left\0right"`.
///
/// NUL is a safe separator because BPE tokens are valid UTF-8 substrings
/// (GPT-2 byte-encoded characters), which never contain NUL bytes.
#[inline]
pub(super) fn bpe_pair_key(buf: &mut String, left: &str, right: &str) {
    buf.clear();
    buf.push_str(left);
    buf.push('\0');
    buf.push_str(right);
}

/// Parse BPE merge rules from a merges.txt string.
///
/// Format: optional `#version: ...` header, then one merge pair per line
/// (space-separated). Line order determines priority rank (lower = higher priority).
///
/// Keys are stored as NUL-separated `"left\0right"` strings to enable
/// zero-allocation lookups in the BPE merge loop (see `bpe_pair_key`).
pub(super) fn parse_merges(merges_text: &str) -> Result<HashMap<String, usize>> {
    let mut ranks = HashMap::new();
    let mut key_buf = String::new();
    for (i, line) in merges_text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(2, ' ');
        let left = parts.next().ok_or_else(|| {
            TensorError::from(WhisperError::MergeParseError {
                line: i + 1,
                detail: "missing left token",
            })
        })?;
        let right = parts.next().ok_or_else(|| {
            TensorError::from(WhisperError::MergeParseError {
                line: i + 1,
                detail: "missing right token",
            })
        })?;
        bpe_pair_key(&mut key_buf, left, right);
        let rank = ranks.len();
        ranks.insert(key_buf.clone(), rank);
    }
    Ok(ranks)
}

/// GPT-2 pre-tokenization: split text into words at class boundaries.
///
/// Splits on transitions between letter, digit, space, and punctuation
/// character classes, with special handling for English contractions
/// ('s, 't, 're, 've, 'm, 'll, 'd). Leading spaces are attached to the
/// following word (matching GPT-2's `Ġ` convention).
pub(super) fn pre_tokenize(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let chars: Vec<char> = text.chars().collect();
    let mut words = Vec::new();
    let mut start = 0;

    while start < chars.len() {
        // Check for contractions starting with apostrophe.
        if chars[start] == '\'' && start > 0 {
            let suffix = contraction_suffix(&chars, start);
            if !suffix.is_empty() {
                words.push(suffix.clone());
                start += suffix.len();
                continue;
            }
        }

        // Accumulate characters of the same class.
        let cls = char_class(chars[start]);
        let mut end = start + 1;

        // Special case: a space followed by non-space characters forms one word
        // (the leading space attaches to the next word, matching GPT-2's Ġ prefix).
        if cls == CharClass::Space {
            // Consume the space(s).
            while end < chars.len() && char_class(chars[end]) == CharClass::Space {
                end += 1;
            }
            // If there are non-space characters following, attach them.
            if end < chars.len() {
                let next_cls = char_class(chars[end]);
                while end < chars.len() && char_class(chars[end]) == next_cls {
                    // Stop before contractions.
                    if chars[end] == '\'' && !contraction_suffix(&chars, end).is_empty() {
                        break;
                    }
                    end += 1;
                }
            }
        } else {
            while end < chars.len() && char_class(chars[end]) == cls {
                if chars[end] == '\'' && !contraction_suffix(&chars, end).is_empty() {
                    break;
                }
                end += 1;
            }
        }

        let word: String = chars[start..end].iter().collect();
        if !word.is_empty() {
            words.push(word);
        }
        start = end;
    }

    words
}

/// Character classification for pre-tokenization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CharClass {
    Letter,
    Digit,
    Space,
    Punct,
}

/// Classify a character for GPT-2 pre-tokenization boundaries.
fn char_class(ch: char) -> CharClass {
    if ch.is_alphabetic() {
        CharClass::Letter
    } else if ch.is_ascii_digit() {
        CharClass::Digit
    } else if ch.is_whitespace() {
        CharClass::Space
    } else {
        CharClass::Punct
    }
}

/// Match English contractions at position `i` in the character array.
///
/// Returns the contraction suffix if found (e.g., "'s", "'re"), or empty string.
fn contraction_suffix(chars: &[char], i: usize) -> String {
    if i >= chars.len() || chars[i] != '\'' {
        return String::new();
    }
    let remaining = chars.len() - i;
    // Check longest suffixes first.
    if remaining >= 3 {
        let two = [chars[i + 1], chars[i + 2]];
        match two {
            ['r', 'e'] | ['v', 'e'] | ['l', 'l'] => {
                return chars[i..i + 3].iter().collect();
            }
            _ => {}
        }
    }
    if remaining >= 2 {
        match chars[i + 1] {
            's' | 't' | 'm' | 'd' => {
                return chars[i..i + 2].iter().collect();
            }
            _ => {}
        }
    }
    String::new()
}
