// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_scan_unicode_clean_text() {
    let config = UnicodeSafetyConfig::default();
    let result = scan_unicode("Hello, world!", &config);
    assert!(result.attacks.is_empty());
    assert!(!result.was_modified);
    assert_eq!(result.sanitized, "Hello, world!");
}

#[test]
fn test_scan_unicode_invisible_chars_stripped() {
    let config = UnicodeSafetyConfig::default();
    // Insert zero-width space between "He" and "llo".
    let input = "He\u{200B}llo";
    let result = scan_unicode(input, &config);
    assert_eq!(result.attacks.len(), 1);
    assert!(result.was_modified);
    assert_eq!(result.sanitized, "Hello");
    assert!(matches!(
        &result.attacks[0],
        UnicodeAttack::InvisibleChar {
            codepoint: 0x200B,
            ..
        }
    ));
}

#[test]
fn test_scan_unicode_invisible_chars_preserved_when_configured() {
    let config = UnicodeSafetyConfig {
        strip_invisible: false,
        ..UnicodeSafetyConfig::default()
    };
    let input = "He\u{200B}llo";
    let result = scan_unicode(input, &config);
    assert_eq!(result.attacks.len(), 1); // Still detected.
    assert!(!result.was_modified); // But not stripped.
    assert_eq!(result.sanitized, input);
}

#[test]
fn test_scan_unicode_multiple_invisible_chars() {
    let config = UnicodeSafetyConfig::default();
    // Zero-width space + soft hyphen + ZW joiner.
    let input = "a\u{200B}b\u{00AD}c\u{200D}d";
    let result = scan_unicode(input, &config);
    assert_eq!(result.attacks.len(), 3);
    assert!(result.was_modified);
    assert_eq!(result.sanitized, "abcd");
}

#[test]
fn test_scan_unicode_bidi_override_always_stripped() {
    let config = UnicodeSafetyConfig {
        strip_invisible: false, // Even with strip_invisible off...
        ..UnicodeSafetyConfig::default()
    };
    // Right-to-left override.
    let input = "Hello\u{202E}World";
    let result = scan_unicode(input, &config);
    assert_eq!(result.attacks.len(), 1);
    assert!(result.was_modified); // Bidi overrides always stripped.
    assert_eq!(result.sanitized, "HelloWorld");
    assert!(matches!(
        &result.attacks[0],
        UnicodeAttack::BidiOverride {
            codepoint: 0x202E,
            ..
        }
    ));
}

#[test]
fn test_scan_unicode_homoglyph_detected() {
    let config = UnicodeSafetyConfig::default();
    // Cyrillic "а" (U+0430) looks identical to Latin "a".
    let input = "H\u{0435}llo"; // Cyrillic е in place of Latin e
    let result = scan_unicode(input, &config);
    assert_eq!(result.attacks.len(), 1);
    assert!(!result.was_modified); // Not normalized by default.
    assert!(matches!(
        &result.attacks[0],
        UnicodeAttack::Homoglyph {
            original: '\u{0435}',
            confusable: 'e',
            ..
        }
    ));
}

#[test]
fn test_scan_unicode_homoglyph_normalized() {
    let config = UnicodeSafetyConfig {
        normalize_homoglyphs: true,
        ..UnicodeSafetyConfig::default()
    };
    // Cyrillic "а" (U+0430) normalized to Latin "a".
    let input = "H\u{0430}ppy"; // Cyrillic а in place of Latin a
    let result = scan_unicode(input, &config);
    assert_eq!(result.attacks.len(), 1);
    assert!(result.was_modified);
    assert_eq!(result.sanitized, "Happy");
}

#[test]
fn test_scan_unicode_unexpected_script() {
    let config = UnicodeSafetyConfig {
        allowed_scripts: vec!["Latin", "Common"],
        ..UnicodeSafetyConfig::default()
    };
    // Arabic character in otherwise Latin text.
    let input = "Hello \u{0639}"; // Arabic ع
    let result = scan_unicode(input, &config);
    let script_attacks: Vec<_> = result
        .attacks
        .iter()
        .filter(|a| matches!(a, UnicodeAttack::UnexpectedScript { .. }))
        .collect();
    assert_eq!(script_attacks.len(), 1);
    assert!(matches!(
        script_attacks[0],
        UnicodeAttack::UnexpectedScript {
            detected_script: "Arabic",
            ..
        }
    ));
}

#[test]
fn test_scan_unicode_no_script_check_when_empty() {
    let config = UnicodeSafetyConfig {
        allowed_scripts: vec![], // Skip script checking.
        ..UnicodeSafetyConfig::default()
    };
    // Arabic character should NOT be flagged.
    let input = "Hello \u{0639}";
    let result = scan_unicode(input, &config);
    let script_attacks: Vec<_> = result
        .attacks
        .iter()
        .filter(|a| matches!(a, UnicodeAttack::UnexpectedScript { .. }))
        .collect();
    assert!(script_attacks.is_empty());
}

#[test]
fn test_tts_confusables_nonempty() {
    let pairs = tts_confusables();
    assert!(
        pairs.len() >= 30,
        "expected at least 30 confusable pairs, got {}",
        pairs.len()
    );
}

#[test]
fn test_tts_confusables_cyrillic_a() {
    let pairs = tts_confusables();
    assert!(
        pairs.contains(&('\u{0430}', 'a')),
        "Cyrillic а → Latin a should be in confusables"
    );
}

#[test]
fn test_tts_confusables_all_canonical_are_ascii() {
    // All canonical characters (second element) should be ASCII Latin.
    let pairs = tts_confusables();
    for (confusable, canonical) in &pairs {
        assert!(
            canonical.is_ascii(),
            "canonical '{}' (U+{:04X}) for confusable '{}' (U+{:04X}) should be ASCII",
            canonical,
            *canonical as u32,
            confusable,
            *confusable as u32
        );
    }
}

#[test]
fn test_scan_unicode_combined_attacks() {
    let config = UnicodeSafetyConfig {
        normalize_homoglyphs: true,
        ..UnicodeSafetyConfig::default()
    };
    // Combine: zero-width space + Cyrillic homoglyph + bidi override.
    let input = "H\u{200B}\u{0435}ll\u{202E}o"; // ZWS + Cyrillic е + RLO
    let result = scan_unicode(input, &config);
    assert_eq!(result.attacks.len(), 3);
    assert!(result.was_modified);
    assert_eq!(result.sanitized, "Hello"); // ZWS stripped, е→e, RLO stripped
}

#[test]
fn test_scan_unicode_empty_input() {
    let config = UnicodeSafetyConfig::default();
    let result = scan_unicode("", &config);
    assert!(result.attacks.is_empty());
    assert!(!result.was_modified);
    assert_eq!(result.sanitized, "");
}

#[test]
fn test_scan_unicode_bom_stripped() {
    let config = UnicodeSafetyConfig::default();
    let input = "\u{FEFF}Hello"; // BOM at start.
    let result = scan_unicode(input, &config);
    assert_eq!(result.attacks.len(), 1);
    assert!(result.was_modified);
    assert_eq!(result.sanitized, "Hello");
    assert!(matches!(
        &result.attacks[0],
        UnicodeAttack::InvisibleChar {
            codepoint: 0xFEFF,
            ..
        }
    ));
}

#[test]
fn test_detect_script_basic_coverage() {
    // Verify detect_script covers the main categories.
    assert_eq!(detect_script('A'), "Latin");
    assert_eq!(detect_script('z'), "Latin");
    assert_eq!(detect_script('\u{0410}'), "Cyrillic");
    assert_eq!(detect_script('\u{03B1}'), "Greek");
    assert_eq!(detect_script('\u{0639}'), "Arabic");
    assert_eq!(detect_script('\u{4E00}'), "CJK");
    assert_eq!(detect_script('!'), "Common");
    assert_eq!(detect_script('5'), "Common");
}

#[test]
fn test_is_invisible_comprehensive() {
    // Test all invisible chars are detected.
    let invisible_chars = [
        '\u{200B}', '\u{200C}', '\u{200D}', '\u{200E}', '\u{200F}', '\u{00AD}', '\u{FEFF}',
        '\u{2060}', '\u{2061}', '\u{2062}', '\u{2063}', '\u{2064}', '\u{180E}', '\u{034F}',
    ];
    for ch in &invisible_chars {
        assert!(
            is_invisible_char(*ch),
            "U+{:04X} should be detected as invisible",
            *ch as u32
        );
    }
    // Normal characters should not be invisible.
    assert!(!is_invisible_char('a'));
    assert!(!is_invisible_char(' ')); // Regular space is visible.
    assert!(!is_invisible_char('\n')); // Newline is not invisible (it's whitespace).
}

#[test]
fn test_homoglyph_canonical_cyrillic_coverage() {
    // Verify all lowercase Cyrillic homoglyphs are detected.
    assert_eq!(homoglyph_canonical('\u{0430}'), Some('a'));
    assert_eq!(homoglyph_canonical('\u{0435}'), Some('e'));
    assert_eq!(homoglyph_canonical('\u{043E}'), Some('o'));
    assert_eq!(homoglyph_canonical('\u{0440}'), Some('p'));
    assert_eq!(homoglyph_canonical('\u{0441}'), Some('c'));
    assert_eq!(homoglyph_canonical('\u{0443}'), Some('y'));
    assert_eq!(homoglyph_canonical('\u{0445}'), Some('x'));
    // Non-homoglyph should return None.
    assert_eq!(homoglyph_canonical('a'), None);
    assert_eq!(homoglyph_canonical('Z'), None);
}
