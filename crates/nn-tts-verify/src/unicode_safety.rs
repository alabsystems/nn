// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unicode input validation for TTS pipelines.
//!
//! Defensive layer before G2P that detects adversarial Unicode attacks:
//! homoglyphs (Cyrillic "а" vs Latin "a"), invisible characters (zero-width
//! space, soft hyphen), bidirectional override characters, and unexpected
//! scripts for the target language.
//!
//! Part of #1740: Adversarial Robustness of TTS.
//!
//! # References
//!
//! - Unicode Consortium. "Unicode Technical Report #39: Unicode Security
//!   Mechanisms." Confusable character detection.
//! - Davis, M. & Suignard, M. (2021). "Unicode Security Considerations."
//!   Unicode Technical Report #36.

/// Known Unicode attack categories relevant to TTS.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum UnicodeAttack {
    /// Confusable characters (Cyrillic "а" vs Latin "a").
    Homoglyph {
        /// The character found in the input.
        original: char,
        /// The expected (canonical) character it could be confused with.
        confusable: char,
        /// Byte offset in the input string.
        byte_offset: usize,
    },
    /// Invisible characters (zero-width space, soft hyphen, etc.).
    InvisibleChar {
        /// The Unicode codepoint of the invisible character.
        codepoint: u32,
        /// Byte offset in the input string.
        byte_offset: usize,
    },
    /// Bidirectional override characters (RLO, LRO, etc.).
    BidiOverride {
        /// The Unicode codepoint of the bidi override.
        codepoint: u32,
        /// Byte offset in the input string.
        byte_offset: usize,
    },
    /// Characters outside expected script for the language.
    UnexpectedScript {
        /// The unexpected character.
        char: char,
        /// The detected script of the character.
        detected_script: &'static str,
        /// Byte offset in the input string.
        byte_offset: usize,
    },
}

/// Configuration for Unicode safety checking.
#[derive(Debug, Clone)]
pub struct UnicodeSafetyConfig {
    /// Expected scripts (e.g., `["Latin", "Common"]` for English TTS).
    /// Characters outside these scripts trigger `UnexpectedScript` alerts.
    /// Empty = skip script checking.
    pub allowed_scripts: Vec<&'static str>,
    /// Strip invisible characters from the output? Default: true.
    pub strip_invisible: bool,
    /// Whether to also strip homoglyph characters (replace with canonical).
    /// Default: false (report only).
    pub normalize_homoglyphs: bool,
}

impl Default for UnicodeSafetyConfig {
    fn default() -> Self {
        Self {
            allowed_scripts: vec!["Latin", "Common"],
            strip_invisible: true,
            normalize_homoglyphs: false,
        }
    }
}

/// Result of scanning text for Unicode attacks.
#[derive(Debug, Clone)]
pub struct UnicodeScanResult {
    /// The sanitized text (invisible chars removed, optionally homoglyphs normalized).
    pub sanitized: String,
    /// Detected attacks, ordered by byte offset.
    pub attacks: Vec<UnicodeAttack>,
    /// Whether the text was modified during sanitization.
    pub was_modified: bool,
}

/// Scan text for Unicode attacks before G2P.
///
/// Returns the sanitized text and any detected attacks. The sanitized text
/// has invisible characters removed (if configured) and optionally has
/// homoglyphs normalized to their Latin equivalents.
pub fn scan_unicode(text: &str, config: &UnicodeSafetyConfig) -> UnicodeScanResult {
    let mut attacks = Vec::new();
    let mut sanitized = String::with_capacity(text.len());
    let mut was_modified = false;

    for (byte_offset, ch) in text.char_indices() {
        // Check for invisible characters.
        if is_invisible_char(ch) {
            attacks.push(UnicodeAttack::InvisibleChar {
                codepoint: ch as u32,
                byte_offset,
            });
            if config.strip_invisible {
                was_modified = true;
                continue; // Skip this character in output.
            }
            sanitized.push(ch);
            continue;
        }

        // Check for bidi override characters.
        if is_bidi_override(ch) {
            attacks.push(UnicodeAttack::BidiOverride {
                codepoint: ch as u32,
                byte_offset,
            });
            // Always strip bidi overrides — they are never legitimate in TTS input.
            was_modified = true;
            continue;
        }

        // Check for homoglyphs.
        if let Some(canonical) = homoglyph_canonical(ch) {
            attacks.push(UnicodeAttack::Homoglyph {
                original: ch,
                confusable: canonical,
                byte_offset,
            });
            if config.normalize_homoglyphs {
                sanitized.push(canonical);
                was_modified = true;
                continue;
            }
            sanitized.push(ch);
            continue;
        }

        // Check for unexpected script.
        if !config.allowed_scripts.is_empty() {
            let script = detect_script(ch);
            if script != "Common" && script != "Inherited" {
                if !config.allowed_scripts.contains(&script) {
                    attacks.push(UnicodeAttack::UnexpectedScript {
                        char: ch,
                        detected_script: script,
                        byte_offset,
                    });
                }
            }
        }

        sanitized.push(ch);
    }

    UnicodeScanResult {
        sanitized,
        attacks,
        was_modified,
    }
}

/// Canonical confusable pairs for TTS-relevant characters.
///
/// Returns pairs of `(confusable, canonical)` where `confusable` is a non-Latin
/// character that visually resembles the Latin `canonical` character.
///
/// Based on Unicode TR39 confusables.txt (subset relevant to Latin script TTS).
/// Covers Cyrillic, Greek, and other scripts with Latin look-alikes.
pub fn tts_confusables() -> Vec<(char, char)> {
    vec![
        // Cyrillic → Latin confusables (most common in adversarial attacks).
        ('\u{0430}', 'a'), // Cyrillic а → Latin a
        ('\u{0435}', 'e'), // Cyrillic е → Latin e
        ('\u{043E}', 'o'), // Cyrillic о → Latin o
        ('\u{0440}', 'p'), // Cyrillic р → Latin p
        ('\u{0441}', 'c'), // Cyrillic с → Latin c
        ('\u{0443}', 'y'), // Cyrillic у → Latin y
        ('\u{0445}', 'x'), // Cyrillic х → Latin x
        ('\u{0410}', 'A'), // Cyrillic А → Latin A
        ('\u{0412}', 'B'), // Cyrillic В → Latin B
        ('\u{0415}', 'E'), // Cyrillic Е → Latin E
        ('\u{041A}', 'K'), // Cyrillic К → Latin K
        ('\u{041C}', 'M'), // Cyrillic М → Latin M
        ('\u{041D}', 'H'), // Cyrillic Н → Latin H
        ('\u{041E}', 'O'), // Cyrillic О → Latin O
        ('\u{0420}', 'P'), // Cyrillic Р → Latin P
        ('\u{0421}', 'C'), // Cyrillic С → Latin C
        ('\u{0422}', 'T'), // Cyrillic Т → Latin T
        ('\u{0425}', 'X'), // Cyrillic Х → Latin X
        // Greek → Latin confusables.
        ('\u{03B1}', 'a'), // Greek α (alpha) — visually similar in some fonts
        ('\u{03BF}', 'o'), // Greek ο (omicron) → Latin o
        ('\u{03C1}', 'p'), // Greek ρ (rho) — visually similar to p
        ('\u{0391}', 'A'), // Greek Α → Latin A
        ('\u{0392}', 'B'), // Greek Β → Latin B
        ('\u{0395}', 'E'), // Greek Ε → Latin E
        ('\u{0397}', 'H'), // Greek Η → Latin H
        ('\u{039A}', 'K'), // Greek Κ → Latin K
        ('\u{039C}', 'M'), // Greek Μ → Latin M
        ('\u{039D}', 'N'), // Greek Ν → Latin N
        ('\u{039F}', 'O'), // Greek Ο → Latin O
        ('\u{03A1}', 'P'), // Greek Ρ → Latin P
        ('\u{03A4}', 'T'), // Greek Τ → Latin T
        ('\u{03A7}', 'X'), // Greek Χ → Latin X
        ('\u{03A5}', 'Y'), // Greek Υ → Latin Y
        ('\u{0396}', 'Z'), // Greek Ζ → Latin Z
        // Fullwidth Latin (CJK compatibility).
        ('\u{FF21}', 'A'), // Fullwidth A
        ('\u{FF22}', 'B'), // Fullwidth B
        ('\u{FF23}', 'C'), // Fullwidth C
        ('\u{FF41}', 'a'), // Fullwidth a
        ('\u{FF42}', 'b'), // Fullwidth b
        ('\u{FF43}', 'c'), // Fullwidth c
    ]
}

/// Check if a character is an invisible Unicode character.
///
/// Covers zero-width characters, soft hyphens, formatting characters,
/// and other invisible codepoints used in adversarial text attacks.
pub(crate) fn is_invisible_char(ch: char) -> bool {
    matches!(
        ch as u32,
        0x200B // Zero-width space
        | 0x200C // Zero-width non-joiner
        | 0x200D // Zero-width joiner
        | 0x200E // Left-to-right mark
        | 0x200F // Right-to-left mark
        | 0x00AD // Soft hyphen
        | 0xFEFF // Byte order mark (zero-width no-break space)
        | 0x2060 // Word joiner
        | 0x2061 // Function application
        | 0x2062 // Invisible times
        | 0x2063 // Invisible separator
        | 0x2064 // Invisible plus
        | 0x180E // Mongolian vowel separator
        | 0x034F // Combining grapheme joiner
    )
}

/// Check if a character is a bidi override character.
///
/// These characters can reorder text display and are never legitimate
/// in TTS input text.
pub(crate) fn is_bidi_override(ch: char) -> bool {
    matches!(
        ch as u32,
        0x202A // Left-to-right embedding
        | 0x202B // Right-to-left embedding
        | 0x202C // Pop directional formatting
        | 0x202D // Left-to-right override
        | 0x202E // Right-to-left override
        | 0x2066 // Left-to-right isolate
        | 0x2067 // Right-to-left isolate
        | 0x2068 // First strong isolate
        | 0x2069 // Pop directional isolate
    )
}

/// Look up the canonical Latin character for a known homoglyph.
///
/// Returns `Some(canonical)` if the character is a known confusable,
/// `None` otherwise.
pub(crate) fn homoglyph_canonical(ch: char) -> Option<char> {
    // Use a match statement for O(1) lookup on the most common confusables.
    match ch as u32 {
        // Cyrillic lowercase → Latin
        0x0430 => Some('a'),
        0x0435 => Some('e'),
        0x043E => Some('o'),
        0x0440 => Some('p'),
        0x0441 => Some('c'),
        0x0443 => Some('y'),
        0x0445 => Some('x'),
        // Cyrillic uppercase → Latin
        0x0410 => Some('A'),
        0x0412 => Some('B'),
        0x0415 => Some('E'),
        0x041A => Some('K'),
        0x041C => Some('M'),
        0x041D => Some('H'),
        0x041E => Some('O'),
        0x0420 => Some('P'),
        0x0421 => Some('C'),
        0x0422 => Some('T'),
        0x0425 => Some('X'),
        // Greek lowercase → Latin
        0x03BF => Some('o'),
        0x03C1 => Some('p'),
        // Greek uppercase → Latin
        0x0391 => Some('A'),
        0x0392 => Some('B'),
        0x0395 => Some('E'),
        0x0397 => Some('H'),
        0x039A => Some('K'),
        0x039C => Some('M'),
        0x039D => Some('N'),
        0x039F => Some('O'),
        0x03A1 => Some('P'),
        0x03A4 => Some('T'),
        0x03A7 => Some('X'),
        0x03A5 => Some('Y'),
        0x0396 => Some('Z'),
        _ => None,
    }
}

/// Detect the Unicode script of a character.
///
/// Returns a static string naming the script. Covers the scripts most
/// relevant to TTS adversarial attacks: Latin, Cyrillic, Greek, CJK,
/// Arabic, and Common (punctuation, digits, symbols).
pub(crate) fn detect_script(ch: char) -> &'static str {
    let cp = ch as u32;
    match cp {
        // Common: ASCII control, digits, punctuation, symbols (before Latin to avoid overlap).
        0x0000..=0x0040 | 0x005B..=0x0060 | 0x007B..=0x00BF => "Common",
        // General punctuation, currency, math, misc symbols
        0x2000..=0x206F | 0x20A0..=0x20CF | 0x2100..=0x214F => "Common",
        // Latin letters: A-Z, a-z, and Latin Extended blocks.
        0x0041..=0x005A | 0x0061..=0x007A | 0x00C0..=0x024F => "Latin",
        // Latin Extended Additional, Latin Extended-B/C/D/E
        0x1E00..=0x1EFF | 0x2C60..=0x2C7F | 0xA720..=0xA7FF => "Latin",
        // Inherited (combining marks)
        0x0300..=0x036F => "Inherited",
        // Greek
        0x0370..=0x03FF | 0x1F00..=0x1FFF => "Greek",
        // Cyrillic
        0x0400..=0x052F => "Cyrillic",
        // Arabic
        0x0600..=0x06FF | 0x0750..=0x077F | 0x08A0..=0x08FF => "Arabic",
        // Devanagari
        0x0900..=0x097F => "Devanagari",
        // Hangul (Korean)
        0xAC00..=0xD7AF | 0x1100..=0x11FF | 0x3130..=0x318F => "Hangul",
        // Hiragana
        0x3040..=0x309F => "Hiragana",
        // Katakana
        0x30A0..=0x30FF | 0x31F0..=0x31FF => "Katakana",
        // CJK Unified Ideographs
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF => "CJK",
        // Fullwidth forms
        0xFF00..=0xFFEF => "Fullwidth",
        // Default: unknown
        _ => "Unknown",
    }
}

#[cfg(test)]
#[path = "unicode_safety_tests.rs"]
mod tests;
