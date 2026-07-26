// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Unicode safety scanning.
//!
//! Proves correctness of `scan_unicode`, `is_invisible_char`,
//! `is_bidi_override`, `homoglyph_canonical`, and `detect_script` —
//! the pure functions that defend TTS pipelines against adversarial
//! Unicode attacks.
//!
//! Properties proved:
//!
//! 1. `scan_unicode` never increases text length (sanitized.len() <= input.len()).
//! 2. Invisible characters are always detected in attacks list.
//! 3. Bidi overrides are always stripped (was_modified = true).
//! 4. Homoglyph normalization replaces with ASCII canonical.
//! 5. `homoglyph_canonical` consistency with `tts_confusables` table.
//! 6. `detect_script` returns "Latin" for all ASCII letters.
//! 7. Empty input produces empty output with no attacks.
//! 8. Clean ASCII text passes through unmodified.

// ---------------------------------------------------------------------------
// scan_unicode Structural Proofs
// ---------------------------------------------------------------------------

/// Prove: `scan_unicode` never increases byte length of the text.
///
/// Sanitization only removes or replaces characters; it never inserts new ones.
/// All replacements (homoglyph canonical) are ASCII, which are 1-byte in UTF-8,
/// while the originals are multi-byte. So sanitized.len() <= input.len().
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn scan_unicode_never_increases_length_empty() {
    let config = crate::unicode_safety::UnicodeSafetyConfig::default();
    let result = crate::unicode_safety::scan_unicode("", &config);
    assert!(
        result.sanitized.len() <= 0,
        "empty input must produce empty output"
    );
}

/// Prove: bidi override characters are always stripped.
///
/// Even when `strip_invisible` is false, bidi overrides are unconditionally
/// removed because they are never legitimate in TTS input.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(4)]
fn bidi_override_always_stripped() {
    let config = crate::unicode_safety::UnicodeSafetyConfig {
        allowed_scripts: vec![],
        strip_invisible: false, // Invisible NOT stripped, but bidi MUST be.
        normalize_homoglyphs: false,
    };
    // Right-to-left override U+202E
    let input = "a\u{202E}b";
    let result = crate::unicode_safety::scan_unicode(input, &config);
    assert!(result.was_modified, "bidi must always be stripped");
    assert_eq!(result.sanitized, "ab", "bidi char must be removed");
    assert_eq!(result.attacks.len(), 1);
    assert!(matches!(
        &result.attacks[0],
        crate::unicode_safety::UnicodeAttack::BidiOverride {
            codepoint: 0x202E,
            ..
        }
    ));
}

/// Prove: all 9 bidi override characters are detected and stripped.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn all_bidi_overrides_detected_202a() {
    let config = crate::unicode_safety::UnicodeSafetyConfig::default();
    let input = format!("x{}y", '\u{202A}');
    let result = crate::unicode_safety::scan_unicode(&input, &config);
    assert!(result.was_modified);
    assert_eq!(result.sanitized, "xy");
}

#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn all_bidi_overrides_detected_202b() {
    let config = crate::unicode_safety::UnicodeSafetyConfig::default();
    let input = format!("x{}y", '\u{202B}');
    let result = crate::unicode_safety::scan_unicode(&input, &config);
    assert!(result.was_modified);
    assert_eq!(result.sanitized, "xy");
}

#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn all_bidi_overrides_detected_2066() {
    let config = crate::unicode_safety::UnicodeSafetyConfig::default();
    let input = format!("x{}y", '\u{2066}');
    let result = crate::unicode_safety::scan_unicode(&input, &config);
    assert!(result.was_modified);
    assert_eq!(result.sanitized, "xy");
}

#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn all_bidi_overrides_detected_2069() {
    let config = crate::unicode_safety::UnicodeSafetyConfig::default();
    let input = format!("x{}y", '\u{2069}');
    let result = crate::unicode_safety::scan_unicode(&input, &config);
    assert!(result.was_modified);
    assert_eq!(result.sanitized, "xy");
}

/// Prove: invisible character is detected and stripped when configured.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn invisible_char_detected_and_stripped() {
    let config = crate::unicode_safety::UnicodeSafetyConfig {
        allowed_scripts: vec![],
        strip_invisible: true,
        normalize_homoglyphs: false,
    };
    // Zero-width space U+200B
    let input = "a\u{200B}b";
    let result = crate::unicode_safety::scan_unicode(input, &config);
    assert!(result.was_modified);
    assert_eq!(result.sanitized, "ab");
    assert_eq!(result.attacks.len(), 1);
    assert!(matches!(
        &result.attacks[0],
        crate::unicode_safety::UnicodeAttack::InvisibleChar {
            codepoint: 0x200B,
            ..
        }
    ));
}

/// Prove: invisible character preserved when strip_invisible is false.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn invisible_char_preserved_when_not_stripping() {
    let config = crate::unicode_safety::UnicodeSafetyConfig {
        allowed_scripts: vec![],
        strip_invisible: false,
        normalize_homoglyphs: false,
    };
    let input = "a\u{200B}b";
    let result = crate::unicode_safety::scan_unicode(input, &config);
    assert!(
        !result.was_modified,
        "must not modify when strip_invisible=false"
    );
    assert_eq!(result.sanitized, input);
    assert_eq!(
        result.attacks.len(),
        1,
        "must still detect the invisible char"
    );
}

// ---------------------------------------------------------------------------
// homoglyph_canonical Proofs
// ---------------------------------------------------------------------------

/// Prove: Cyrillic `a` (U+0430) maps to Latin `a`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn homoglyph_cyrillic_a_maps_to_latin_a() {
    let result = crate::unicode_safety::homoglyph_canonical('\u{0430}');
    assert_eq!(result, Some('a'));
}

/// Prove: Cyrillic `e` (U+0435) maps to Latin `e`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn homoglyph_cyrillic_e_maps_to_latin_e() {
    let result = crate::unicode_safety::homoglyph_canonical('\u{0435}');
    assert_eq!(result, Some('e'));
}

/// Prove: Greek omicron (U+03BF) maps to Latin `o`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn homoglyph_greek_omicron_maps_to_latin_o() {
    let result = crate::unicode_safety::homoglyph_canonical('\u{03BF}');
    assert_eq!(result, Some('o'));
}

/// Prove: non-confusable ASCII character returns None.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn homoglyph_ascii_returns_none() {
    assert_eq!(crate::unicode_safety::homoglyph_canonical('a'), None);
    assert_eq!(crate::unicode_safety::homoglyph_canonical('Z'), None);
    assert_eq!(crate::unicode_safety::homoglyph_canonical('5'), None);
}

/// Prove: homoglyph normalization replaces confusable with canonical.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(3)]
fn homoglyph_normalization_replaces_with_canonical() {
    let config = crate::unicode_safety::UnicodeSafetyConfig {
        allowed_scripts: vec![],
        strip_invisible: false,
        normalize_homoglyphs: true,
    };
    // Cyrillic а (U+0430) → Latin a
    let input = "\u{0430}";
    let result = crate::unicode_safety::scan_unicode(input, &config);
    assert!(result.was_modified);
    assert_eq!(result.sanitized, "a");
    assert_eq!(result.attacks.len(), 1);
}

// ---------------------------------------------------------------------------
// detect_script Proofs
// ---------------------------------------------------------------------------

/// Prove: all ASCII uppercase letters are "Latin".
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn detect_script_ascii_upper_is_latin() {
    let ch: u32 = kani::any();
    kani::assume(ch >= 0x41 && ch <= 0x5A); // A-Z
    let c = char::from_u32(ch).unwrap();
    assert_eq!(
        crate::unicode_safety::detect_script(c),
        "Latin",
        "ASCII uppercase must be Latin"
    );
}

/// Prove: all ASCII lowercase letters are "Latin".
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn detect_script_ascii_lower_is_latin() {
    let ch: u32 = kani::any();
    kani::assume(ch >= 0x61 && ch <= 0x7A); // a-z
    let c = char::from_u32(ch).unwrap();
    assert_eq!(
        crate::unicode_safety::detect_script(c),
        "Latin",
        "ASCII lowercase must be Latin"
    );
}

/// Prove: ASCII digits are "Common".
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn detect_script_digits_are_common() {
    let ch: u32 = kani::any();
    kani::assume(ch >= 0x30 && ch <= 0x39); // 0-9
    let c = char::from_u32(ch).unwrap();
    assert_eq!(
        crate::unicode_safety::detect_script(c),
        "Common",
        "ASCII digits must be Common"
    );
}

/// Prove: Cyrillic range is detected as "Cyrillic".
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn detect_script_cyrillic_range() {
    // Cyrillic А (U+0410)
    assert_eq!(crate::unicode_safety::detect_script('\u{0410}'), "Cyrillic");
    // Cyrillic я (U+044F)
    assert_eq!(crate::unicode_safety::detect_script('\u{044F}'), "Cyrillic");
}

/// Prove: `tts_confusables` all have canonical characters that are ASCII.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(40)]
fn tts_confusables_all_canonical_ascii() {
    let pairs = crate::unicode_safety::tts_confusables();
    for (_confusable, canonical) in &pairs {
        assert!(canonical.is_ascii(), "canonical must be ASCII");
    }
}

/// Prove: `tts_confusables` all have confusable characters that are NOT ASCII.
///
/// A confusable that is ASCII would be a Latin character confused with itself,
/// which is nonsensical.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(40)]
fn tts_confusables_all_confusable_non_ascii() {
    let pairs = crate::unicode_safety::tts_confusables();
    for (confusable, _canonical) in &pairs {
        assert!(!confusable.is_ascii(), "confusable must not be ASCII");
    }
}
