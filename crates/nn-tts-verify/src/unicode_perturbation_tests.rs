// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Unicode-to-embedding perturbation bridge.

use super::*;
use crate::adversarial::english_confusion_sets;

/// Simple G2P stub: maps character index to a token ID.
/// Lowercase letters map to their ASCII value minus 'a' + 20 (so 'a' → 20, 'b' → 21, etc.).
/// Non-letter positions return None.
fn stub_char_to_token(text: &str) -> impl Fn(usize) -> Option<u32> + '_ {
    move |idx: usize| {
        let ch = text.chars().nth(idx)?;
        if ch.is_ascii_lowercase() {
            Some((ch as u32) - ('a' as u32) + 20)
        } else if ch.is_ascii_uppercase() {
            Some((ch as u32) - ('A' as u32) + 20)
        } else {
            None // Space, punctuation → no phoneme.
        }
    }
}

#[test]
fn test_identify_vulnerable_clean_latin_text() {
    let config = UnicodeSafetyConfig::default();
    let vuln = identify_vulnerable_positions("hello world", &config);

    // Latin chars with homoglyph confusables: h, e, o in "hello", o in "world"
    // 'h' has Cyrillic Н (uppercase only), 'e' has Cyrillic е, 'o' has Cyrillic о
    assert!(
        !vuln.is_empty(),
        "Clean Latin text should still identify vulnerable positions"
    );

    // Verify all identified positions are homoglyph type.
    for v in &vuln {
        assert_eq!(v.attack_type, VulnerabilityType::Homoglyph);
        assert!(v.canonical.is_some());
    }
}

#[test]
fn test_identify_vulnerable_cyrillic_attack() {
    let config = UnicodeSafetyConfig::default();
    // "hеllo" with Cyrillic е (U+0435) instead of Latin e.
    let text = "h\u{0435}llo";
    let vuln = identify_vulnerable_positions(text, &config);

    // Should detect the Cyrillic е as a homoglyph attack.
    let cyrillic_positions: Vec<_> = vuln
        .iter()
        .filter(|v| v.original_char == '\u{0435}')
        .collect();
    assert!(
        !cyrillic_positions.is_empty(),
        "Should detect Cyrillic е in 'hеllo'"
    );
}

#[test]
fn test_identify_vulnerable_invisible_chars() {
    let config = UnicodeSafetyConfig::default();
    // "he\u{200B}llo" with zero-width space between 'e' and 'l'.
    let text = "he\u{200B}llo";
    let vuln = identify_vulnerable_positions(text, &config);

    let invisible_positions: Vec<_> = vuln
        .iter()
        .filter(|v| v.attack_type == VulnerabilityType::InvisibleInsertion)
        .collect();
    assert!(
        !invisible_positions.is_empty(),
        "Should detect invisible character insertion"
    );
}

#[test]
fn test_identify_vulnerable_mixed_script() {
    let config = UnicodeSafetyConfig {
        allowed_scripts: vec!["Latin", "Common"],
        ..Default::default()
    };

    // Text with a Chinese character.
    let text = "hello\u{4E16}world";
    let vuln = identify_vulnerable_positions(text, &config);

    let mixed_script: Vec<_> = vuln
        .iter()
        .filter(|v| v.attack_type == VulnerabilityType::MixedScript)
        .collect();
    assert!(
        !mixed_script.is_empty(),
        "Should detect CJK character as unexpected script"
    );
}

#[test]
fn test_map_to_phoneme_coverage() {
    let config = UnicodeSafetyConfig::default();
    let text = "hello";
    let vuln = identify_vulnerable_positions(text, &config);
    let char_to_token = stub_char_to_token(text);
    let sets = english_confusion_sets();

    let derived = map_to_phoneme_confusion_sets(&vuln, &char_to_token, &sets);

    // Each vulnerable position should have a phoneme token mapping.
    for d in &derived {
        assert!(
            d.phoneme_token_id > 0,
            "Token ID should be positive for letter positions"
        );
    }
}

#[test]
fn test_analyze_coverage_report() {
    let config = UnicodeSafetyConfig::default();
    let text = "hello";
    let char_to_token = stub_char_to_token(text);
    let sets = english_confusion_sets();

    let report = analyze_unicode_coverage(text, &config, &char_to_token, &sets);

    assert!(report.total_vulnerable > 0);
    assert!(report.coverage_ratio >= 0.0 && report.coverage_ratio <= 1.0);
    assert_eq!(
        report.covered_by_linguistic + report.uncovered,
        report.total_vulnerable
    );
}

#[test]
fn test_expand_confusion_sets() {
    let config = UnicodeSafetyConfig::default();
    let text = "hello";
    let char_to_token = stub_char_to_token(text);
    let sets = english_confusion_sets();

    let report = analyze_unicode_coverage(text, &config, &char_to_token, &sets);
    let expanded = expand_confusion_sets_for_unicode(&sets, &report);

    // Expanded should include original sets plus any unicode-derived.
    assert!(expanded.len() >= sets.len());

    // Unicode-derived sets should have names starting with "unicode_derived_".
    let unicode_derived: Vec<_> = expanded
        .iter()
        .filter(|s| s.name.starts_with("unicode_derived_"))
        .collect();
    assert_eq!(unicode_derived.len(), report.uncovered);
}

#[test]
fn test_empty_text_no_vulnerabilities() {
    let config = UnicodeSafetyConfig::default();
    let vuln = identify_vulnerable_positions("", &config);
    assert!(vuln.is_empty());
}

#[test]
fn test_pure_digits_no_vulnerabilities() {
    let config = UnicodeSafetyConfig::default();
    let vuln = identify_vulnerable_positions("12345", &config);
    // Digits have no homoglyph confusables in our table.
    let homoglyphs: Vec<_> = vuln
        .iter()
        .filter(|v| v.attack_type == VulnerabilityType::Homoglyph)
        .collect();
    assert!(homoglyphs.is_empty(), "Digits should have no homoglyphs");
}

#[test]
fn test_coverage_empty_sets_all_uncovered() {
    let config = UnicodeSafetyConfig::default();
    let text = "hello";
    let char_to_token = stub_char_to_token(text);
    let no_sets: Vec<ConfusionSet> = vec![];

    let report = analyze_unicode_coverage(text, &config, &char_to_token, &no_sets);

    // With no existing confusion sets, all positions should be uncovered.
    assert_eq!(report.covered_by_linguistic, 0);
    assert_eq!(report.uncovered, report.total_vulnerable);
    if report.total_vulnerable > 0 {
        assert_eq!(report.coverage_ratio, 0.0);
    }
}

#[test]
fn test_positions_sorted_and_deduped() {
    let config = UnicodeSafetyConfig::default();
    // Multiple attack types at different positions.
    let text = "h\u{0435}ll\u{200B}o"; // Cyrillic е + zero-width space.
    let vuln = identify_vulnerable_positions(text, &config);

    // Verify sorted by char_index.
    for window in vuln.windows(2) {
        assert!(
            window[0].char_index <= window[1].char_index,
            "Positions should be sorted"
        );
    }

    // Verify no duplicate char_index.
    let mut indices: Vec<_> = vuln.iter().map(|v| v.char_index).collect();
    indices.dedup();
    assert_eq!(
        indices.len(),
        vuln.len(),
        "Positions should be deduplicated"
    );
}
