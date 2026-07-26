// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Text preprocessing for Kokoro TTS — misaki-equivalent text normalization.
//!
//! Converts raw English text into a normalized form suitable for espeak-ng G2P.
//! This is a pure-Rust port of misaki's English text preprocessing pipeline:
//! - Number expansion ("42" → "forty two")
//! - Ordinal expansion ("1st" → "first")
//! - Currency expansion ("$3.50" → "three dollars and fifty cents")
//! - Abbreviation expansion ("Dr." → "Doctor")
//! - Punctuation normalization (smart quotes → ASCII, repeated marks)
//!
//! # Pipeline Position
//!
//! ```text
//! Raw text → [TextPreprocessor] → cleaned text → [espeak-ng] → IPA → ...
//! ```

use std::collections::HashMap;

use crate::kokoro_number_words::expand_numbers_in_text;

/// Text preprocessor for English text normalization before G2P.
///
/// Applies number expansion, abbreviation lookup, and punctuation normalization
/// in sequence. Configurable via the abbreviation table and options.
#[derive(Debug, Clone)]
pub struct TextPreprocessor {
    abbreviations: HashMap<String, String>,
    expand_numbers: bool,
    normalize_punctuation: bool,
}

impl Default for TextPreprocessor {
    fn default() -> Self {
        Self {
            abbreviations: default_abbreviations(),
            expand_numbers: true,
            normalize_punctuation: true,
        }
    }
}

impl TextPreprocessor {
    /// Create a preprocessor with default English abbreviations and all features enabled.
    #[must_use]
    pub fn english() -> Self {
        Self::default()
    }

    /// Create a preprocessor with no abbreviations and all features enabled.
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            abbreviations: HashMap::new(),
            expand_numbers: true,
            normalize_punctuation: true,
        }
    }

    /// Add or override an abbreviation mapping.
    pub fn add_abbreviation(&mut self, abbrev: impl Into<String>, expansion: impl Into<String>) {
        self.abbreviations
            .insert(abbrev.into().to_lowercase(), expansion.into());
    }

    /// Remove an abbreviation mapping.
    pub fn remove_abbreviation(&mut self, abbrev: &str) -> Option<String> {
        self.abbreviations.remove(&abbrev.to_lowercase())
    }

    /// Enable or disable number expansion.
    pub fn set_expand_numbers(&mut self, enabled: bool) {
        self.expand_numbers = enabled;
    }

    /// Enable or disable punctuation normalization.
    pub fn set_normalize_punctuation(&mut self, enabled: bool) {
        self.normalize_punctuation = enabled;
    }

    /// Preprocess text for G2P input.
    ///
    /// Applies in order:
    /// 1. Punctuation normalization (smart quotes, repeated marks)
    /// 2. Abbreviation expansion
    /// 3. Number/currency/ordinal expansion
    /// 4. Whitespace normalization
    #[must_use]
    pub fn preprocess(&self, text: &str) -> String {
        let mut result = text.to_owned();

        if self.normalize_punctuation {
            result = normalize_punctuation(&result);
        }

        result = expand_abbreviations(&result, &self.abbreviations);

        if self.expand_numbers {
            result = expand_numbers_in_text(&result);
        }

        normalize_whitespace(&result)
    }

    /// Split text into sentences for separate synthesis.
    ///
    /// Splits on sentence-ending punctuation (`.`, `!`, `?`, `…`) while
    /// preserving the punctuation with its sentence. Handles common
    /// abbreviations (Mr., Dr., etc.) to avoid false splits.
    #[must_use]
    pub fn split_sentences(&self, text: &str) -> Vec<String> {
        split_sentences_inner(text, &self.abbreviations)
    }
}

// -- Punctuation normalization ------------------------------------------------

/// Normalize Unicode punctuation to ASCII equivalents.
#[must_use]
pub(crate) fn normalize_punctuation(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            // Smart quotes → ASCII
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => result.push('\''),
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => result.push('"'),
            // En dash → comma (misaki behavior for parenthetical)
            '\u{2013}' => result.push(','),
            // Repeated punctuation: collapse "!!!" → "!", "???" → "?"
            '!' | '?' => {
                result.push(ch);
                while chars.peek() == Some(&ch) {
                    chars.next();
                }
            }
            // Ellipsis character → three dots
            '\u{2026}' => result.push_str("..."),
            // Other Unicode → pass through
            _ => result.push(ch),
        }
    }
    result
}

// -- Whitespace normalization -------------------------------------------------

/// Collapse runs of whitespace into single spaces, trim.
#[must_use]
pub(crate) fn normalize_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut last_was_space = true; // trim leading
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !last_was_space {
                result.push(' ');
                last_was_space = true;
            }
        } else {
            result.push(ch);
            last_was_space = false;
        }
    }
    // trim trailing
    if result.ends_with(' ') {
        result.pop();
    }
    result
}

// -- Abbreviation expansion ---------------------------------------------------

/// Expand abbreviations in text, matching whole words only.
#[must_use]
pub(crate) fn expand_abbreviations(text: &str, table: &HashMap<String, String>) -> String {
    if table.is_empty() {
        return text.to_owned();
    }

    let mut result = String::with_capacity(text.len());
    let mut i = 0;
    let bytes = text.as_bytes();

    while i < bytes.len() {
        // Try to match a word starting at position i
        if i == 0 || !bytes[i - 1].is_ascii_alphanumeric() {
            if let Some((matched_len, expansion)) = find_abbreviation(text, i, table) {
                result.push_str(expansion);
                i += matched_len;
                continue;
            }
        }
        // Safe: we only advance by one byte for ASCII, or by char width for multi-byte
        let ch = text[i..].chars().next().unwrap();
        result.push(ch);
        i += ch.len_utf8();
    }
    result
}

/// Try to match an abbreviation starting at `pos` in `text`.
///
/// Returns (matched byte length, expansion) if found.
fn find_abbreviation<'a>(
    text: &str,
    pos: usize,
    table: &'a HashMap<String, String>,
) -> Option<(usize, &'a str)> {
    let remaining = &text[pos..];

    // Collect the word (letters + optional trailing period)
    let mut word_end = 0;
    for ch in remaining.chars() {
        if ch.is_alphabetic() || ch == '.' {
            word_end += ch.len_utf8();
        } else {
            break;
        }
    }
    if word_end == 0 {
        return None;
    }

    let candidate = &remaining[..word_end];
    let lower = candidate.to_lowercase();

    // Check if followed by a word boundary
    let after = &remaining[word_end..];
    let at_boundary =
        after.is_empty() || after.chars().next().map_or(true, |c| !c.is_alphanumeric());

    if at_boundary {
        if let Some(expansion) = table.get(&lower) {
            return Some((word_end, expansion.as_str()));
        }
        // Try without trailing period
        let stripped = lower.strip_suffix('.').unwrap_or(&lower);
        if stripped != lower {
            if let Some(_expansion) = table.get(stripped) {
                // Only match if the abbreviation entry expects the period
                // (e.g., "dr." is in the table, not just "dr")
                return None;
            }
        }
    }
    None
}

// -- Sentence splitting -------------------------------------------------------

/// Split text into sentences, preserving punctuation.
fn split_sentences_inner(text: &str, abbreviations: &HashMap<String, String>) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        current.push(chars[i]);

        if chars[i] == '!' || chars[i] == '?' {
            // Definite sentence end
            let trimmed = current.trim().to_owned();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
            i += 1;
            continue;
        }

        if chars[i] == '.' {
            // Check if this period is part of an abbreviation
            let word_before = extract_word_before(&chars, i);
            let with_period = format!("{}.", word_before.to_lowercase());
            let is_abbreviation = abbreviations.contains_key(&with_period);

            // Check for ellipsis (...)
            let is_ellipsis = i + 2 < len && chars[i + 1] == '.' && chars[i + 2] == '.';
            if is_ellipsis {
                current.push('.');
                current.push('.');
                i += 3;
                let trimmed = current.trim().to_owned();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current.clear();
                continue;
            }

            if !is_abbreviation {
                // Check if followed by a space and uppercase (sentence boundary)
                let next_idx = i + 1;
                let at_end = next_idx >= len;
                let followed_by_cap = !at_end
                    && next_idx + 1 < len
                    && chars[next_idx].is_whitespace()
                    && chars[next_idx + 1].is_uppercase();

                if at_end || followed_by_cap {
                    let trimmed = current.trim().to_owned();
                    if !trimmed.is_empty() {
                        sentences.push(trimmed);
                    }
                    current.clear();
                }
            }
        }

        i += 1;
    }

    let trimmed = current.trim().to_owned();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

/// Extract the word immediately before position `i` in the char array.
fn extract_word_before(chars: &[char], i: usize) -> String {
    let mut word = String::new();
    let mut j = i;
    while j > 0 {
        j -= 1;
        if chars[j].is_alphabetic() {
            word.push(chars[j]);
        } else {
            break;
        }
    }
    word.chars().rev().collect()
}

// -- Default abbreviation table -----------------------------------------------

/// Build the default English abbreviation table matching misaki's defaults.
///
/// Entries are stored as lowercase keys (with trailing period where applicable).
#[must_use]
pub(crate) fn default_abbreviations() -> HashMap<String, String> {
    let entries = [
        // Titles
        ("mr.", "Mister"),
        ("mrs.", "Misses"),
        ("ms.", "Miss"),
        ("dr.", "Doctor"),
        ("prof.", "Professor"),
        ("sr.", "Senior"),
        ("jr.", "Junior"),
        ("rev.", "Reverend"),
        ("hon.", "Honorable"),
        ("sgt.", "Sergeant"),
        ("cpl.", "Corporal"),
        ("pvt.", "Private"),
        ("lt.", "Lieutenant"),
        ("cpt.", "Captain"),
        ("capt.", "Captain"),
        ("maj.", "Major"),
        ("col.", "Colonel"),
        ("gen.", "General"),
        ("gov.", "Governor"),
        ("pres.", "President"),
        ("rep.", "Representative"),
        ("sen.", "Senator"),
        // Common abbreviations
        ("st.", "Street"),
        ("ave.", "Avenue"),
        ("blvd.", "Boulevard"),
        ("rd.", "Road"),
        ("ln.", "Lane"),
        ("ct.", "Court"),
        ("apt.", "Apartment"),
        ("dept.", "Department"),
        ("bldg.", "Building"),
        ("fl.", "Floor"),
        ("ste.", "Suite"),
        ("no.", "Number"),
        ("vol.", "Volume"),
        ("pg.", "Page"),
        ("ch.", "Chapter"),
        ("sec.", "Section"),
        ("fig.", "Figure"),
        ("eq.", "Equation"),
        // Units and measures (without period in typical use)
        ("ft.", "feet"),
        ("in.", "inches"),
        ("oz.", "ounces"),
        ("lb.", "pounds"),
        ("lbs.", "pounds"),
        ("yr.", "year"),
        ("yrs.", "years"),
        ("hr.", "hour"),
        ("hrs.", "hours"),
        ("min.", "minutes"),
        ("approx.", "approximately"),
        // Latin abbreviations
        ("etc.", "et cetera"),
        ("vs.", "versus"),
        ("e.g.", "for example"),
        ("i.e.", "that is"),
        ("cf.", "compare"),
        ("al.", "and others"),
        ("viz.", "namely"),
        // Common informal
        ("govt.", "government"),
        ("corp.", "corporation"),
        ("inc.", "incorporated"),
        ("ltd.", "limited"),
        ("assn.", "association"),
        ("intl.", "international"),
        ("natl.", "national"),
    ];

    entries
        .iter()
        .map(|&(k, v)| (k.to_lowercase(), v.to_owned()))
        .collect()
}

#[cfg(test)]
#[path = "kokoro_text_preprocess_tests.rs"]
mod tests;
