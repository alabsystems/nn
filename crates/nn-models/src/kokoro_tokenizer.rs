// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro phoneme tokenizer — maps phoneme strings to token IDs.
//!
//! The tokenizer is data-driven: the phoneme→token vocabulary is loaded from
//! JSON (matching Kokoro's `config.json` format) and can be extended at runtime.
//!
//! # Pipeline
//!
//! ```text
//! Text → G2P (espeak/misaki) → phoneme string → KokoroTokenizer::encode() → token IDs
//! ```
//!
//! Token ID 0 is the padding token (added at start and end of every sequence).
//! The maximum sequence length is 512 (PlBert `max_position_embeddings`), so
//! the maximum phoneme token count per chunk is 510.
//!
//! # Chunking
//!
//! [`KokoroTokenizer::chunk_and_encode`] splits long phoneme strings at
//! punctuation boundaries using a waterfall strategy (prefer `!.?…`, then
//! `:;`, then `,—`) to stay within the 510-token limit.

use std::collections::HashMap;

use crate::kokoro_error::KokoroError;

/// Maximum phoneme tokens per chunk (PlBert context_length=512 minus 2 padding).
pub const MAX_PHONEME_TOKENS: usize = 510;

/// Padding token ID (added at start and end of every encoded sequence).
pub const PAD_TOKEN_ID: u32 = 0;

/// Phoneme-to-token vocabulary, loadable from JSON.
///
/// The vocabulary maps individual phoneme characters (Unicode code points) to
/// sparse token IDs. Kokoro's default vocabulary has 178 tokens (IDs 0–177,
/// with gaps). The mapping is extensible — call [`KokoroVocab::insert`] to
/// add custom phonemes.
#[derive(Debug, Clone)]
pub struct KokoroVocab {
    /// Forward map: phoneme char → token ID.
    char_to_id: HashMap<char, u32>,
    /// Reverse map: token ID → phoneme char (for debugging/decode).
    id_to_char: HashMap<u32, char>,
    /// Total token count (including padding token 0).
    n_tokens: u32,
}

impl KokoroVocab {
    /// Create an empty vocabulary.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            char_to_id: HashMap::new(),
            id_to_char: HashMap::new(),
            n_tokens: 1, // token 0 = padding
        }
    }

    /// Load vocabulary from a JSON object mapping phoneme strings to integer IDs.
    ///
    /// Matches the `vocab` field in Kokoro's `config.json`:
    /// ```json
    /// { ";": 1, ":": 2, ",": 3, "ˈ": 156, ... }
    /// ```
    ///
    /// Each key must be exactly one Unicode character. Multi-char keys are
    /// skipped with a warning (the caller can log these).
    ///
    /// # Errors
    /// Returns `KokoroError::InvalidConfig` if the JSON is malformed.
    pub fn from_json_map(
        map: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<Self, KokoroError> {
        let mut vocab = Self::empty();
        let mut max_id: u32 = 0;
        for (key, value) in map {
            let id = value.as_u64().ok_or_else(|| KokoroError::InvalidConfig {
                field: "vocab",
                reason: format!("expected integer token ID for key '{key}', got {value}"),
            })? as u32;
            let mut chars = key.chars();
            let ch = chars.next().ok_or_else(|| KokoroError::InvalidConfig {
                field: "vocab",
                reason: format!("empty key in vocab map (ID {id})"),
            })?;
            if chars.next().is_some() {
                // Multi-char keys not supported — skip silently.
                // Kokoro's vocab is all single-char, but be defensive.
                continue;
            }
            vocab.char_to_id.insert(ch, id);
            vocab.id_to_char.insert(id, ch);
            if id > max_id {
                max_id = id;
            }
        }
        vocab.n_tokens = max_id + 1;
        Ok(vocab)
    }

    /// Load vocabulary from a complete Kokoro config JSON string.
    ///
    /// Extracts the `vocab` field from the top-level object.
    ///
    /// # Errors
    /// Returns `KokoroError::InvalidConfig` if JSON parsing fails or `vocab` is missing.
    pub fn from_config_json(json: &str) -> Result<Self, KokoroError> {
        let parsed: serde_json::Value =
            serde_json::from_str(json).map_err(|e| KokoroError::InvalidConfig {
                field: "config.json",
                reason: format!("JSON parse error: {e}"),
            })?;
        let map = parsed
            .get("vocab")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| KokoroError::InvalidConfig {
                field: "vocab",
                reason: "missing or non-object 'vocab' field in config".into(),
            })?;
        Self::from_json_map(map)
    }

    /// Insert or update a phoneme→token mapping.
    pub fn insert(&mut self, ch: char, id: u32) {
        self.char_to_id.insert(ch, id);
        self.id_to_char.insert(id, ch);
        if id >= self.n_tokens {
            self.n_tokens = id + 1;
        }
    }

    /// Remove a phoneme mapping. Returns the old token ID if present.
    pub fn remove(&mut self, ch: char) -> Option<u32> {
        if let Some(id) = self.char_to_id.remove(&ch) {
            self.id_to_char.remove(&id);
            Some(id)
        } else {
            None
        }
    }

    /// Look up a phoneme character's token ID.
    #[must_use]
    pub fn get(&self, ch: char) -> Option<u32> {
        self.char_to_id.get(&ch).copied()
    }

    /// Reverse lookup: token ID → phoneme character.
    #[must_use]
    pub fn decode_id(&self, id: u32) -> Option<char> {
        self.id_to_char.get(&id).copied()
    }

    /// Total vocabulary size (including padding token 0).
    #[must_use]
    pub fn n_tokens(&self) -> u32 {
        self.n_tokens
    }

    /// Number of mapped phoneme characters (excluding padding).
    #[must_use]
    pub fn len(&self) -> usize {
        self.char_to_id.len()
    }

    /// Whether the vocabulary has no phoneme mappings.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.char_to_id.is_empty()
    }

    /// Iterate over all (char, token_id) mappings.
    pub fn iter(&self) -> impl Iterator<Item = (char, u32)> + '_ {
        self.char_to_id.iter().map(|(&ch, &id)| (ch, id))
    }

    /// Validate that all token IDs are within the embedding table size.
    ///
    /// Returns `Err` if any mapped token ID is >= `embedding_vocab_size`.
    /// Call this after loading or extending a vocab to ensure forward passes
    /// won't hit `EmbeddingIndexOutOfRange`.
    pub fn validate(&self, embedding_vocab_size: usize) -> Result<(), KokoroError> {
        for (&ch, &id) in &self.char_to_id {
            if (id as usize) >= embedding_vocab_size {
                return Err(KokoroError::InvalidConfig {
                    field: "vocab",
                    reason: format!(
                        "token ID {id} for char '{}' (U+{:04X}) exceeds embedding vocab size {embedding_vocab_size}",
                        ch, ch as u32,
                    ),
                });
            }
        }
        Ok(())
    }

    /// Insert a new phoneme with the next available sequential token ID.
    ///
    /// Returns the assigned token ID. This is the safe path for dynamic
    /// extension — avoids gaps and collisions with existing IDs.
    pub fn insert_auto(&mut self, ch: char) -> u32 {
        let id = self.n_tokens;
        self.char_to_id.insert(ch, id);
        self.id_to_char.insert(id, ch);
        self.n_tokens = id + 1;
        id
    }

    /// Extend the vocabulary from a JSON string mapping chars to IDs.
    ///
    /// Accepts the same format as `from_json_map`: `{"ɡ": 92, "ʃ": 131}`.
    /// Returns the list of newly added `(char, id)` pairs. Existing mappings
    /// for the same char are overwritten.
    ///
    /// # Errors
    /// Returns `KokoroError::InvalidConfig` if the JSON is malformed.
    pub fn extend_from_json(&mut self, json: &str) -> Result<Vec<(char, u32)>, KokoroError> {
        let parsed: serde_json::Value =
            serde_json::from_str(json).map_err(|e| KokoroError::InvalidConfig {
                field: "supplementary_vocab",
                reason: format!("JSON parse error: {e}"),
            })?;
        let map = parsed
            .as_object()
            .ok_or_else(|| KokoroError::InvalidConfig {
                field: "supplementary_vocab",
                reason: "expected a JSON object".into(),
            })?;
        let mut added = Vec::new();
        for (key, value) in map {
            let id = value.as_u64().ok_or_else(|| KokoroError::InvalidConfig {
                field: "supplementary_vocab",
                reason: format!("expected integer token ID for key '{key}', got {value}"),
            })? as u32;
            let mut chars = key.chars();
            let ch = chars.next().ok_or_else(|| KokoroError::InvalidConfig {
                field: "supplementary_vocab",
                reason: format!("empty key in supplementary vocab (ID {id})"),
            })?;
            if chars.next().is_some() {
                continue; // skip multi-char keys
            }
            self.insert(ch, id);
            added.push((ch, id));
        }
        Ok(added)
    }

    /// Build the default Kokoro-82M vocabulary (178 tokens).
    ///
    /// This is the vocabulary from `hexgrad/Kokoro-82M/config.json`.
    /// Prefer loading from JSON for extensibility; this method provides
    /// a zero-dependency fallback.
    #[must_use]
    pub fn kokoro_default() -> Self {
        let mut vocab = Self::empty();
        // Punctuation and special characters
        for (ch, id) in [
            (';', 1),
            (':', 2),
            (',', 3),
            ('.', 4),
            ('!', 5),
            ('?', 6),
            ('\u{2014}', 9),  // — em dash
            ('\u{2026}', 10), // … ellipsis
            ('"', 11),
            ('(', 12),
            (')', 13),
            ('\u{201C}', 14), // " left double quote
            ('\u{201D}', 15), // " right double quote
            (' ', 16),
            ('\u{0303}', 17), // combining tilde
        ] {
            vocab.insert(ch, id);
        }
        // Affricate/special IPA
        for (ch, id) in [
            ('\u{02A3}', 18), // ʣ
            ('\u{02A5}', 19), // ʥ
            ('\u{02A6}', 20), // ʦ
            ('\u{02A8}', 21), // ʨ
            ('\u{1D5D}', 22), // ᵝ
            ('\u{AB67}', 23), // ꭧ
        ] {
            vocab.insert(ch, id);
        }
        // Uppercase diphthong/special phonemes
        for (ch, id) in [
            ('A', 24),
            ('I', 25),
            ('O', 31),
            ('Q', 33),
            ('S', 35),
            ('T', 36),
            ('W', 39),
            ('Y', 41),
        ] {
            vocab.insert(ch, id);
        }
        // Modified IPA
        vocab.insert('\u{1D4A}', 42); // ᵊ
                                      // Lowercase Latin
        for (ch, id) in [
            ('a', 43),
            ('b', 44),
            ('c', 45),
            ('d', 46),
            ('e', 47),
            ('f', 48),
            ('h', 50),
            ('i', 51),
            ('j', 52),
            ('k', 53),
            ('l', 54),
            ('m', 55),
            ('n', 56),
            ('o', 57),
            ('p', 58),
            ('q', 59),
            ('r', 60),
            ('s', 61),
            ('t', 62),
            ('u', 63),
            ('v', 64),
            ('w', 65),
            ('x', 66),
            ('y', 67),
            ('z', 68),
        ] {
            vocab.insert(ch, id);
        }
        // IPA vowels and consonants
        for (ch, id) in [
            ('\u{0251}', 69),  // ɑ
            ('\u{0250}', 70),  // ɐ
            ('\u{0252}', 71),  // ɒ
            ('\u{00E6}', 72),  // æ
            ('\u{03B2}', 75),  // β
            ('\u{0254}', 76),  // ɔ
            ('\u{0255}', 77),  // ɕ
            ('\u{00E7}', 78),  // ç
            ('\u{0256}', 80),  // ɖ
            ('\u{00F0}', 81),  // ð
            ('\u{02A4}', 82),  // ʤ
            ('\u{0259}', 83),  // ə
            ('\u{025A}', 85),  // ɚ
            ('\u{025B}', 86),  // ɛ
            ('\u{025C}', 87),  // ɜ
            ('\u{025F}', 90),  // ɟ
            ('\u{0261}', 92),  // ɡ
            ('\u{0265}', 99),  // ɥ
            ('\u{0268}', 101), // ɨ
            ('\u{026A}', 102), // ɪ
            ('\u{029D}', 103), // ʝ
            ('\u{026F}', 110), // ɯ
            ('\u{0270}', 111), // ɰ
            ('\u{014B}', 112), // ŋ
            ('\u{0273}', 113), // ɳ
            ('\u{0272}', 114), // ɲ
            ('\u{0274}', 115), // ɴ
            ('\u{00F8}', 116), // ø
            ('\u{0278}', 118), // ɸ
            ('\u{03B8}', 119), // θ
            ('\u{0153}', 120), // œ
            ('\u{0279}', 123), // ɹ
            ('\u{027E}', 125), // ɾ
            ('\u{027B}', 126), // ɻ
            ('\u{0281}', 128), // ʁ
            ('\u{027D}', 129), // ɽ
            ('\u{0282}', 130), // ʂ
            ('\u{0283}', 131), // ʃ
            ('\u{0288}', 132), // ʈ
            ('\u{02A7}', 133), // ʧ
            ('\u{028A}', 135), // ʊ
            ('\u{028B}', 136), // ʋ
            ('\u{028C}', 138), // ʌ
            ('\u{0263}', 139), // ɣ
            ('\u{0264}', 140), // ɤ
            ('\u{03C7}', 142), // χ
            ('\u{028E}', 143), // ʎ
            ('\u{0292}', 147), // ʒ
            ('\u{0294}', 148), // ʔ
        ] {
            vocab.insert(ch, id);
        }
        // Prosodic markers and modifiers
        for (ch, id) in [
            ('\u{02C8}', 156), // ˈ primary stress
            ('\u{02CC}', 157), // ˌ secondary stress
            ('\u{02D0}', 158), // ː length
            ('\u{02B0}', 162), // ʰ aspiration
            ('\u{02B2}', 164), // ʲ palatalization
        ] {
            vocab.insert(ch, id);
        }
        // Tone markers
        for (ch, id) in [
            ('\u{2193}', 169), // ↓
            ('\u{2192}', 171), // →
            ('\u{2197}', 172), // ↗
            ('\u{2198}', 173), // ↘
        ] {
            vocab.insert(ch, id);
        }
        // Final special
        vocab.insert('\u{1D7B}', 177); // ᵻ
        vocab
    }
}

// -- KokoroTokenizer ----------------------------------------------------------

/// Kokoro tokenizer: phoneme string → padded token IDs.
///
/// Wraps a [`KokoroVocab`] and provides encoding, chunking, and the complete
/// text→tokens pipeline when combined with a G2P backend.
#[derive(Debug, Clone)]
pub struct KokoroTokenizer {
    vocab: KokoroVocab,
    max_tokens: usize,
}

impl KokoroTokenizer {
    /// Create a tokenizer with the given vocabulary.
    #[must_use]
    pub fn new(vocab: KokoroVocab) -> Self {
        Self {
            vocab,
            max_tokens: MAX_PHONEME_TOKENS,
        }
    }

    /// Create a tokenizer with the given vocabulary, validating token IDs
    /// against the model's embedding table size.
    ///
    /// Returns `Err` if any token ID in the vocab is >= `embedding_vocab_size`.
    pub fn with_validated_vocab(
        vocab: KokoroVocab,
        embedding_vocab_size: usize,
    ) -> Result<Self, KokoroError> {
        vocab.validate(embedding_vocab_size)?;
        Ok(Self {
            vocab,
            max_tokens: MAX_PHONEME_TOKENS,
        })
    }

    /// Create a tokenizer with the default Kokoro-82M vocabulary.
    #[must_use]
    pub fn kokoro_default() -> Self {
        Self::new(KokoroVocab::kokoro_default())
    }

    /// Access the underlying vocabulary.
    #[must_use]
    pub fn vocab(&self) -> &KokoroVocab {
        &self.vocab
    }

    /// Mutable access to the vocabulary for runtime extension.
    pub fn vocab_mut(&mut self) -> &mut KokoroVocab {
        &mut self.vocab
    }

    /// Maximum phoneme tokens per chunk.
    #[must_use]
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Encode a phoneme string to token IDs with padding.
    ///
    /// Each character in `phonemes` is looked up in the vocabulary. Characters
    /// not in the vocabulary are silently dropped (matching Python behavior).
    /// The result is padded with token 0 at start and end: `[0, ...ids, 0]`.
    ///
    /// # Errors
    /// Returns `KokoroError::InvalidConfig` if the encoded sequence exceeds
    /// the maximum context length (512 = 510 tokens + 2 padding).
    pub fn encode(&self, phonemes: &str) -> Result<Vec<u32>, KokoroError> {
        let ids: Vec<u32> = phonemes
            .chars()
            .filter_map(|ch| self.vocab.get(ch))
            .collect();
        if ids.len() > self.max_tokens {
            return Err(KokoroError::InvalidConfig {
                field: "phoneme_length",
                reason: format!(
                    "phoneme sequence produces {} tokens, max is {} (use chunk_and_encode for long text)",
                    ids.len(),
                    self.max_tokens,
                ),
            });
        }
        let mut result = Vec::with_capacity(ids.len() + 2);
        result.push(PAD_TOKEN_ID);
        result.extend_from_slice(&ids);
        result.push(PAD_TOKEN_ID);
        Ok(result)
    }

    /// Encode without length checking (for internal use after chunking).
    fn encode_unchecked(&self, phonemes: &str) -> Vec<u32> {
        let ids: Vec<u32> = phonemes
            .chars()
            .filter_map(|ch| self.vocab.get(ch))
            .collect();
        let mut result = Vec::with_capacity(ids.len() + 2);
        result.push(PAD_TOKEN_ID);
        result.extend_from_slice(&ids);
        result.push(PAD_TOKEN_ID);
        result
    }

    /// Count how many token IDs a phoneme string would produce (without padding).
    #[must_use]
    pub fn count_tokens(&self, phonemes: &str) -> usize {
        phonemes
            .chars()
            .filter(|ch| self.vocab.get(*ch).is_some())
            .count()
    }

    /// Split a phoneme string into chunks that fit within the token limit,
    /// then encode each chunk.
    ///
    /// Uses a waterfall splitting strategy at punctuation boundaries:
    /// 1. Prefer splitting at sentence-ending punctuation (`!`, `.`, `?`, `…`)
    /// 2. Then at clause boundaries (`:`, `;`)
    /// 3. Then at phrase boundaries (`,`, `—`)
    /// 4. Last resort: split at space
    /// 5. Hard truncation if no split point found
    ///
    /// Returns `(chunk_phonemes, chunk_token_ids)` pairs.
    pub fn chunk_and_encode(&self, phonemes: &str) -> Vec<(String, Vec<u32>)> {
        if phonemes.is_empty() {
            return Vec::new();
        }
        // Fast path: fits in one chunk
        if self.count_tokens(phonemes) <= self.max_tokens {
            return vec![(phonemes.to_owned(), self.encode_unchecked(phonemes))];
        }
        let mut results = Vec::new();
        let mut remaining = phonemes;
        while !remaining.is_empty() {
            let token_count = self.count_tokens(remaining);
            if token_count <= self.max_tokens {
                results.push((remaining.to_owned(), self.encode_unchecked(remaining)));
                break;
            }
            // Find split point using waterfall strategy
            let split_idx = self.find_split_point(remaining);
            let (chunk, rest) = remaining.split_at(split_idx);
            let chunk = chunk.trim_end();
            let rest = rest.trim_start();
            if !chunk.is_empty() {
                results.push((chunk.to_owned(), self.encode_unchecked(chunk)));
            }
            remaining = rest;
        }
        results
    }

    /// Find the best split point in a phoneme string that exceeds the limit.
    ///
    /// Walks forward to find how many characters fit in max_tokens, then
    /// searches backward for punctuation boundaries.
    fn find_split_point(&self, phonemes: &str) -> usize {
        // Find the byte index where we exceed the token limit
        let mut token_count = 0usize;
        let mut limit_byte_idx = phonemes.len();
        for (byte_idx, ch) in phonemes.char_indices() {
            if self.vocab.get(ch).is_some() {
                token_count += 1;
            }
            if token_count > self.max_tokens {
                limit_byte_idx = byte_idx;
                break;
            }
        }
        let search_region = &phonemes[..limit_byte_idx];
        // Waterfall: prefer splitting at punctuation boundaries
        static WATERFALL: &[&[char]] = &[
            &['!', '.', '?', '\u{2026}'], // sentence-ending
            &[':', ';'],                  // clause boundary
            &[',', '\u{2014}'],           // phrase boundary (— em dash)
        ];
        for punct_set in WATERFALL {
            if let Some(pos) = search_region.rfind(|c: char| punct_set.contains(&c)) {
                // Include the punctuation character in the chunk
                let split = pos + phonemes[pos..].chars().next().map_or(1, char::len_utf8);
                if split > 0 {
                    return split;
                }
            }
        }
        // Fall back to splitting at last space
        if let Some(pos) = search_region.rfind(' ') {
            if pos > 0 {
                return pos;
            }
        }
        // Hard truncation at the limit
        limit_byte_idx
    }
}

#[cfg(test)]
#[path = "kokoro_tokenizer_tests.rs"]
mod tests;

#[cfg(kani)]
#[path = "kokoro_tokenizer_kani_tests.rs"]
mod kani_proofs;
