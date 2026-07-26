// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Whisper tokenizer: GPT-2 byte-level BPE for decode (token IDs → text).
//!
//! Whisper uses a GPT-2-style byte-level BPE tokenizer with multilingual
//! extensions. This module handles the inference output path: converting
//! decoded token IDs back to human-readable text.
//!
//! # Vocabulary Format
//!
//! The vocabulary is stored as a JSON object mapping token strings to integer
//! IDs (e.g., `{"hello": 31373, ...}`). Token strings use GPT-2's byte-to-unicode
//! encoding where each byte value maps to a specific Unicode codepoint.
//!
//! # Special Tokens
//!
//! | Token | ID | Purpose |
//! |-------|-----|---------|
//! | `<\|endoftext\|>` | 50257 | End of text / End of transcript |
//! | `<\|startoftranscript\|>` | 50258 | Start of transcript |
//! | `<\|en\|>` | 50259 | Language: English |
//! | `<\|translate\|>` | 50359 | Task: translate |
//! | `<\|transcribe\|>` | 50360 | Task: transcribe |
//! | `<\|startoflm\|>` | 50361 | Start of language model prompt |
//! | `<\|startofprev\|>` | 50362 | Start of previous context |
//! | `<\|nospeech\|>` | 50363 | No speech detected |
//! | `<\|notimestamps\|>` | 50364 | No timestamps mode |
//! | `<\|0.00\|>` .. `<\|30.00\|>` | 50365.. | Timestamp tokens (0.02s resolution) |

use std::collections::HashMap;
use std::path::Path;

use crate::WhisperError;
use nn_core::{Result, TensorError};

/// First timestamp token ID in Whisper vocabulary.
///
/// In whisper-large-v3-turbo (HuggingFace), `<|0.00|>` = 50365.
/// There are 100 language tokens (50259-50358), then translate=50359,
/// transcribe=50360, ..., notimestamps=50364, timestamps=50365+.
pub(crate) const TIMESTAMP_BEGIN: usize = 50365;

/// End-of-text token ID.
pub const EOT_TOKEN: usize = 50257;

/// Start-of-transcript token ID.
pub const SOT_TOKEN: usize = 50258;

/// No-timestamps token ID.
pub const NO_TIMESTAMPS_TOKEN: usize = 50364;

/// No-speech token ID.
pub const NO_SPEECH_TOKEN: usize = 50363;

/// First language token ID (English = 50259).
pub const LANGUAGE_TOKEN_START: usize = 50259;

/// Last language token ID (inclusive, 100 languages: 50259-50358).
pub const LANGUAGE_TOKEN_END: usize = 50358;

/// Default no-speech probability threshold.
///
/// If `softmax(logits)[NO_SPEECH_TOKEN] > threshold`, the segment is considered
/// to contain no speech. AI Provider Whisper uses 0.6.
pub const DEFAULT_NO_SPEECH_THRESHOLD: f64 = 0.6;

/// Whisper tokenizer for converting between text and token IDs.
///
/// Supports vocabulary loading from JSON and GPT-2 byte-level BPE.
/// Decode (IDs → text) works with vocabulary only. Encode (text → IDs)
/// additionally requires BPE merge rules loaded via [`from_vocab_and_merges`].
#[derive(Debug, Clone)]
pub struct WhisperTokenizer {
    /// Reverse mapping: token ID → token string (GPT-2 byte-encoded).
    id_to_token: Vec<String>,
    /// Forward mapping: token string → token ID.
    token_to_id: HashMap<String, usize>,
    /// GPT-2 unicode-to-byte reverse mapping (for decode).
    byte_decoder: HashMap<char, u8>,
    /// GPT-2 byte-to-unicode forward mapping (for encode).
    byte_encoder: HashMap<u8, char>,
    /// BPE merge pair priorities: `"left\0right"` → rank. Lower rank = higher priority.
    /// Keys are NUL-separated to enable zero-allocation lookups via `bpe_pair_key`.
    /// Empty when only decoding is needed.
    bpe_ranks: HashMap<String, usize>,
    /// Total vocabulary size.
    vocab_size: usize,
}

/// A decoded text segment, optionally with timestamp information.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct DecodedSegment {
    /// The decoded text for this segment.
    pub text: String,
    /// Start time in seconds (from timestamp token), if present.
    pub start: Option<f64>,
    /// End time in seconds (from timestamp token), if present.
    pub end: Option<f64>,
}

impl WhisperTokenizer {
    /// Load tokenizer from a `vocab.json` file.
    ///
    /// The file should contain a JSON object mapping token strings to integer IDs.
    /// This is the format used by HuggingFace Whisper models.
    ///
    /// # Errors
    ///
    /// Returns error if the file cannot be read or parsed.
    pub fn from_vocab_json(path: impl AsRef<Path>) -> Result<Self> {
        let data = std::fs::read_to_string(path.as_ref()).map_err(TensorError::IoError)?;
        Self::from_vocab_str(&data)
    }

    /// Load tokenizer from a vocab JSON string (decode-only).
    ///
    /// Useful for testing or embedding vocabulary directly.
    /// For encoding (text → IDs), use [`from_vocab_and_merges`] instead.
    pub fn from_vocab_str(json: &str) -> Result<Self> {
        let token_to_id: HashMap<String, usize> = serde_json::from_str(json).map_err(|e| {
            TensorError::from(WhisperError::VocabParseError {
                detail: e.to_string(),
            })
        })?;

        Self::build(token_to_id, HashMap::new())
    }

    /// Load tokenizer from vocab JSON and BPE merges (decode + encode).
    ///
    /// `merges_text` is the content of `merges.txt` from HuggingFace Whisper
    /// models. First line is typically `#version: 0.2` (skipped). Each
    /// subsequent non-empty line is a space-separated pair of token strings
    /// with line order determining merge priority.
    pub fn from_vocab_and_merges(vocab_json: &str, merges_text: &str) -> Result<Self> {
        let token_to_id: HashMap<String, usize> =
            serde_json::from_str(vocab_json).map_err(|e| {
                TensorError::from(WhisperError::VocabParseError {
                    detail: e.to_string(),
                })
            })?;

        let bpe_ranks = parse_merges(merges_text)?;
        Self::build(token_to_id, bpe_ranks)
    }

    /// Load tokenizer from vocab.json and merges.txt file paths.
    pub fn from_files(vocab_path: impl AsRef<Path>, merges_path: impl AsRef<Path>) -> Result<Self> {
        let vocab_json =
            std::fs::read_to_string(vocab_path.as_ref()).map_err(TensorError::IoError)?;
        let merges_text =
            std::fs::read_to_string(merges_path.as_ref()).map_err(TensorError::IoError)?;
        Self::from_vocab_and_merges(&vocab_json, &merges_text)
    }

    /// Internal constructor shared by all loading paths.
    fn build(
        token_to_id: HashMap<String, usize>,
        bpe_ranks: HashMap<String, usize>,
    ) -> Result<Self> {
        let vocab_size = token_to_id
            .values()
            .copied()
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);

        // Build reverse mapping.
        let mut id_to_token = vec![String::new(); vocab_size];
        for (token, &id) in &token_to_id {
            if id < vocab_size {
                id_to_token[id] = token.clone();
            }
        }

        let byte_decoder = build_byte_decoder();
        let byte_encoder = build_byte_encoder();

        Ok(Self {
            id_to_token,
            token_to_id,
            byte_decoder,
            byte_encoder,
            bpe_ranks,
            vocab_size,
        })
    }

    /// Decode a sequence of token IDs to text.
    ///
    /// Special tokens (language, task, timestamps, EOT) are skipped.
    /// The remaining tokens are decoded using GPT-2 byte-level BPE mapping.
    pub fn decode(&self, token_ids: &[usize]) -> Result<String> {
        let mut bytes = Vec::new();

        for &id in token_ids {
            if self.is_special(id) {
                continue;
            }
            if id >= self.vocab_size {
                return Err(WhisperError::TokenOutOfRange {
                    id,
                    vocab_size: self.vocab_size,
                }
                .into());
            }

            let token_str = &self.id_to_token[id];
            for ch in token_str.chars() {
                if let Some(&byte) = self.byte_decoder.get(&ch) {
                    bytes.push(byte);
                }
                // Characters not in the byte decoder are silently skipped.
                // This handles any edge cases with special formatting chars.
            }
        }

        String::from_utf8(bytes).map_err(|e| {
            WhisperError::Utf8DecodeError {
                detail: e.to_string(),
            }
            .into()
        })
    }

    /// Decode token IDs into segments with timestamp boundaries.
    ///
    /// Splits the token sequence at timestamp token pairs, producing
    /// `DecodedSegment` values with start/end times and associated text.
    ///
    /// Tokens before any timestamp are collected into a segment with no times.
    pub fn decode_with_timestamps(&self, token_ids: &[usize]) -> Result<Vec<DecodedSegment>> {
        let mut segments: Vec<DecodedSegment> = Vec::new();
        let mut current_tokens: Vec<usize> = Vec::new();
        let mut current_start: Option<f64> = None;

        for &id in token_ids {
            if id == EOT_TOKEN
                || id == SOT_TOKEN
                || id == NO_TIMESTAMPS_TOKEN
                || id == NO_SPEECH_TOKEN
            {
                continue;
            }

            if let Some(ts) = self.timestamp_value(id) {
                if let Some(start) = current_start {
                    // We have a start timestamp and now an end timestamp.
                    let text = self.decode(&current_tokens)?;
                    segments.push(DecodedSegment {
                        text,
                        start: Some(start),
                        end: Some(ts),
                    });
                    current_tokens.clear();
                    current_start = None;
                } else {
                    // First timestamp in a pair — flush any preceding text.
                    if !current_tokens.is_empty() {
                        let text = self.decode(&current_tokens)?;
                        segments.push(DecodedSegment {
                            text,
                            start: None,
                            end: None,
                        });
                        current_tokens.clear();
                    }
                    current_start = Some(ts);
                }
            } else if !self.is_special(id) {
                current_tokens.push(id);
            }
        }

        // Flush remaining tokens.
        if !current_tokens.is_empty() {
            let text = self.decode(&current_tokens)?;
            segments.push(DecodedSegment {
                text,
                start: current_start,
                end: None,
            });
        }

        Ok(segments)
    }

    /// Check whether a token ID is a special token.
    ///
    /// Special tokens include: EOT, SOT, language, task, timestamps,
    /// and other Whisper control tokens (IDs >= 50257).
    #[must_use]
    pub fn is_special(&self, token_id: usize) -> bool {
        token_id >= EOT_TOKEN
    }

    /// Check whether a token ID is a timestamp token.
    #[must_use]
    pub fn is_timestamp(&self, token_id: usize) -> bool {
        token_id >= TIMESTAMP_BEGIN
    }

    /// Get the timestamp value in seconds for a timestamp token.
    ///
    /// Returns `None` if the token is not a timestamp token.
    /// Timestamp resolution is 0.02 seconds.
    #[must_use]
    pub fn timestamp_value(&self, token_id: usize) -> Option<f64> {
        // Use checked_sub for defense-in-depth against future refactoring.
        token_id
            .checked_sub(TIMESTAMP_BEGIN)
            .map(|offset| offset as f64 * 0.02)
    }

    /// Look up the token ID for a language tag.
    ///
    /// The language code should be a 2-letter ISO 639-1 code (e.g., "en", "fr", "de").
    /// Returns the token ID for `<|{lang}|>`.
    #[must_use]
    pub fn language_token(&self, lang: &str) -> Option<usize> {
        let key = format!("<|{lang}|>");
        self.token_to_id.get(&key).copied()
    }

    /// Look up the token string for a token ID.
    ///
    /// Returns `None` if the ID is out of range.
    #[must_use]
    pub fn token_str(&self, token_id: usize) -> Option<&str> {
        self.id_to_token.get(token_id).map(String::as_str)
    }

    /// Look up the token ID for a token string.
    #[must_use]
    pub fn token_id(&self, token: &str) -> Option<usize> {
        self.token_to_id.get(token).copied()
    }

    /// Vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Whether BPE merges are loaded (encoding is available).
    #[must_use]
    pub fn can_encode(&self) -> bool {
        !self.bpe_ranks.is_empty()
    }
}

// encode() and bpe() methods extracted to tokenizer_encode.rs (#1667).
#[path = "tokenizer_encode.rs"]
mod encode;

#[path = "tokenizer_bpe.rs"]
mod bpe;
use bpe::{bpe_pair_key, parse_merges, pre_tokenize};

#[path = "tokenizer_byte_map.rs"]
mod byte_map;
use byte_map::{build_byte_decoder, build_byte_encoder};

#[cfg(test)]
#[path = "tokenizer_tests.rs"]
mod tests;

#[cfg(kani)]
#[path = "kani_tokenizer_proofs.rs"]
mod kani_tokenizer_proofs;

#[cfg(kani)]
#[path = "kani_byte_map_proofs.rs"]
mod kani_byte_map_proofs;
