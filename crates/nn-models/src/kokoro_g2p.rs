// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Espeak IPA → Kokoro phoneme remapping.
//!
//! Converts espeak-ng IPA output to Kokoro's simplified phoneme inventory.
//! This is a pure-Rust port of the misaki `espeak.py` remapping tables.
//!
//! # Architecture
//!
//! The remapping is split into two layers:
//! 1. **Multi-char substitutions** (`E2M`): affricate/diphthong sequences
//!    like `aɪ→I`, `dʒ→ʤ` that must be matched longest-first.
//! 2. **Single-char cleanup**: diacritics removal, version-specific adjustments.
//!
//! The tables are data-driven via [`RemapTable`] and can be extended at runtime.

use std::collections::HashMap;

/// A sorted substitution table (longest-match-first).
///
/// Each entry maps a multi-character IPA sequence to a Kokoro phoneme string.
/// Entries are sorted by descending key length so that longer sequences match
/// before shorter prefixes (e.g., `aɪ` before `a`).
#[derive(Debug, Clone)]
pub struct RemapTable {
    entries: Vec<(String, String)>,
}

impl RemapTable {
    /// Create a remap table from (source, target) pairs.
    ///
    /// Entries are automatically sorted by descending source length.
    pub fn new(mut entries: Vec<(String, String)>) -> Self {
        entries.sort_by_key(|e| std::cmp::Reverse(e.0.len()));
        Self { entries }
    }

    /// Add a new mapping. Re-sorts the table.
    pub fn insert(&mut self, source: impl Into<String>, target: impl Into<String>) {
        self.entries.push((source.into(), target.into()));
        self.entries.sort_by_key(|e| std::cmp::Reverse(e.0.len()));
    }

    /// Remove a mapping by source key. Returns the old target if present.
    pub fn remove(&mut self, source: &str) -> Option<String> {
        if let Some(pos) = self.entries.iter().position(|(s, _)| s == source) {
            Some(self.entries.remove(pos).1)
        } else {
            None
        }
    }

    /// Apply all substitutions to the input string (longest-match-first).
    #[must_use]
    pub fn apply(&self, input: &str) -> String {
        let mut result = input.to_owned();
        for (from, to) in &self.entries {
            // Replace all occurrences
            if result.contains(from.as_str()) {
                result = result.replace(from.as_str(), to.as_str());
            }
        }
        result
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// -- EspeakRemapper -----------------------------------------------------------

/// Remaps espeak-ng IPA output to Kokoro's phoneme inventory.
///
/// Implements the misaki `EspeakFallback` (English) and `EspeakG2P` (multilingual)
/// conversion logic in pure Rust. The remap tables can be extended at runtime.
///
/// # Usage
///
/// ```ignore
/// let remapper = EspeakRemapper::english_us();
/// let kokoro_phonemes = remapper.remap("hɛˈloʊ wɜːɹld");
/// ```
#[derive(Debug, Clone)]
pub struct EspeakRemapper {
    /// Multi-character substitution table (longest-match-first).
    remap: RemapTable,
    /// Characters to strip (diacritics, tie bars, etc.).
    strip_chars: Vec<char>,
}

impl EspeakRemapper {
    /// Create a remapper with custom tables.
    pub fn new(remap: RemapTable, strip_chars: Vec<char>) -> Self {
        Self { remap, strip_chars }
    }

    /// English (US) remapper matching misaki's `EspeakFallback` + US adjustments.
    ///
    /// Handles: diphthongs, affricates, r-colored vowels, and US-specific mappings.
    #[must_use]
    pub fn english_us() -> Self {
        let entries = vec![
            // Multi-char: glottal + syllabic nasal
            ("ʔˌn\u{0329}".into(), "ʔn".into()),
            ("ʔn\u{0329}".into(), "ʔn".into()),
            // Diphthongs
            ("aɪ".into(), "I".into()),
            ("aʊ".into(), "W".into()),
            ("eɪ".into(), "A".into()),
            ("e".into(), "A".into()),
            ("ɔɪ".into(), "Y".into()),
            // US-specific
            ("oʊ".into(), "O".into()),
            ("ɜːɹ".into(), "ɜɹ".into()),
            // Affricates
            ("dʒ".into(), "ʤ".into()),
            ("tʃ".into(), "ʧ".into()),
            // Schwa + lateral
            ("əl".into(), "ᵊl".into()),
            // Palatalized sequences → plain
            ("ʲo".into(), "jo".into()),
            ("ʲə".into(), "jə".into()),
            ("ʲ".into(), String::new()),
            // R-colored vowel
            ("ɚ".into(), "əɹ".into()),
            // Single-char substitutions
            ("r".into(), "ɹ".into()),
            ("x".into(), "k".into()),
            ("ç".into(), "k".into()),
            ("ɐ".into(), "ə".into()),
            ("ɬ".into(), "l".into()),
            ("\u{0303}".into(), String::new()), // combining tilde → remove
        ];
        let strip_chars = vec![
            '\u{0329}', // combining vertical line below (syllabic)
            '^',        // tie bar (espeak uses with tie='^')
            '-',        // hyphen separator
        ];
        Self::new(RemapTable::new(entries), strip_chars)
    }

    /// English (British) remapper with GB-specific adjustments.
    #[must_use]
    pub fn english_gb() -> Self {
        let entries = vec![
            ("ʔˌn\u{0329}".into(), "ʔn".into()),
            ("ʔn\u{0329}".into(), "ʔn".into()),
            ("aɪ".into(), "I".into()),
            ("aʊ".into(), "W".into()),
            ("eɪ".into(), "A".into()),
            ("e".into(), "A".into()),
            ("ɔɪ".into(), "Y".into()),
            // GB-specific
            ("eə".into(), "ɛː".into()),
            ("əʊ".into(), "Q".into()),
            // Affricates
            ("dʒ".into(), "ʤ".into()),
            ("tʃ".into(), "ʧ".into()),
            ("əl".into(), "ᵊl".into()),
            ("ʲo".into(), "jo".into()),
            ("ʲə".into(), "jə".into()),
            ("ʲ".into(), String::new()),
            ("ɚ".into(), "əɹ".into()),
            ("r".into(), "ɹ".into()),
            ("x".into(), "k".into()),
            ("ç".into(), "k".into()),
            ("ɐ".into(), "ə".into()),
            ("ɬ".into(), "l".into()),
            ("\u{0303}".into(), String::new()),
        ];
        let strip_chars = vec!['\u{0329}', '^', '-'];
        Self::new(RemapTable::new(entries), strip_chars)
    }

    /// Multilingual remapper matching misaki's `EspeakG2P` (non-English/CJK).
    ///
    /// Includes nasal vowel mappings for French, Portuguese, etc.
    #[must_use]
    pub fn multilingual() -> Self {
        let entries = vec![
            // Diphthongs
            ("aɪ".into(), "I".into()),
            ("aʊ".into(), "W".into()),
            ("eɪ".into(), "A".into()),
            ("oʊ".into(), "O".into()),
            ("əʊ".into(), "Q".into()),
            ("ɔɪ".into(), "Y".into()),
            // Affricates
            ("dz".into(), "ʣ".into()),
            ("dʒ".into(), "ʤ".into()),
            ("ss".into(), "S".into()),
            ("ts".into(), "ʦ".into()),
            ("tʃ".into(), "ʧ".into()),
            // Nasal vowels (v2.0)
            ("œ\u{0303}".into(), "B".into()),
            ("ɔ\u{0303}".into(), "C".into()),
            ("ɑ\u{0303}".into(), "D".into()),
            ("ɛ\u{0303}".into(), "E".into()),
            ("ʊ\u{0303}".into(), "V".into()),
            ("u\u{0303}".into(), "U".into()),
            ("o\u{0303}".into(), "X".into()),
            ("ɐ\u{0303}".into(), "Z".into()),
        ];
        let strip_chars = vec!['^', '-'];
        Self::new(RemapTable::new(entries), strip_chars)
    }

    /// Access the underlying remap table for runtime modification.
    pub fn remap_table_mut(&mut self) -> &mut RemapTable {
        &mut self.remap
    }

    /// Access the strip characters list for runtime modification.
    pub fn strip_chars_mut(&mut self) -> &mut Vec<char> {
        &mut self.strip_chars
    }

    /// Remap an espeak IPA string to Kokoro phonemes.
    ///
    /// 1. Strip tie bars and diacritics
    /// 2. Apply multi-char substitutions (longest-match-first)
    /// 3. Return the remapped string
    #[must_use]
    pub fn remap(&self, ipa: &str) -> String {
        // Step 1: strip unwanted characters
        let cleaned: String = ipa
            .chars()
            .filter(|c| !self.strip_chars.contains(c))
            .collect();
        // Step 2: apply multi-char substitutions
        self.remap.apply(&cleaned)
    }
}

// -- Lexicon support ----------------------------------------------------------

/// A simple phoneme lexicon for high-frequency word overrides.
///
/// Maps lowercase words to their pre-computed phoneme strings. Loaded from
/// a simple text format (one `word\tphonemes` per line) or built programmatically.
#[derive(Debug, Clone, Default)]
pub struct PhonemeLexicon {
    entries: HashMap<String, String>,
}

impl PhonemeLexicon {
    /// Create an empty lexicon.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Load from tab-separated text: `word\tphonemes\n`.
    ///
    /// Lines starting with `#` are comments. Empty lines are skipped.
    pub fn from_tsv(text: &str) -> Self {
        let mut entries = HashMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((word, phonemes)) = line.split_once('\t') {
                entries.insert(word.to_lowercase(), phonemes.to_owned());
            }
        }
        Self { entries }
    }

    /// Look up a word's phonemes. The word is lowercased before lookup.
    #[must_use]
    pub fn get(&self, word: &str) -> Option<&str> {
        self.entries.get(&word.to_lowercase()).map(String::as_str)
    }

    /// Insert or update a word mapping.
    pub fn insert(&mut self, word: impl Into<String>, phonemes: impl Into<String>) {
        self.entries
            .insert(word.into().to_lowercase(), phonemes.into());
    }

    /// Remove a word mapping.
    pub fn remove(&mut self, word: &str) -> Option<String> {
        self.entries.remove(&word.to_lowercase())
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the lexicon is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// -- Post-processing (misaki version < 2.0) -----------------------------------

/// Apply misaki's post-processing for versions < 2.0.
///
/// Replaces `ɾ` (alveolar tap) with `T` and `ʔ` (glottal stop) with `t`.
/// This matches the Python: `ps.replace('ɾ', 'T').replace('ʔ', 't')`.
#[must_use]
pub fn postprocess_v1(phonemes: &str) -> String {
    phonemes
        .replace('\u{027E}', "T") // ɾ → T
        .replace('\u{0294}', "t") // ʔ → t
}

#[cfg(test)]
#[path = "kokoro_g2p_tests.rs"]
mod tests;
