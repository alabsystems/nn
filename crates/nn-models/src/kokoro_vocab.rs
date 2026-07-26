// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro phoneme vocabulary — maps individual phoneme characters to token IDs.
//!
//! The vocabulary is data-driven: load from JSON (matching Kokoro's `config.json`
//! format) or use [`KokoroVocab::kokoro_default()`] for the built-in 178-token
//! vocabulary from `hexgrad/Kokoro-82M`.
//!
//! Each entry maps a single Unicode code point (IPA character, punctuation, or
//! prosodic marker) to a sparse integer token ID. The mapping is extensible at
//! runtime via [`KokoroVocab::insert`].

use std::collections::HashMap;

use crate::kokoro_error::KokoroError;

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
