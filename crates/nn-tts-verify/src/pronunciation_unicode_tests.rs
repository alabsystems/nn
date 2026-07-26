// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for pronunciation defect + Unicode vulnerability bridge.

use super::*;
use crate::phoneme::{PhonemeResult, PhonemeVerifyConfig};
use crate::quality::QualityMetric;
use crate::unicode_perturbation::{
    UnicodeCoverageReport, UnicodeDerivedConfusionSet, VulnerabilityType,
};

fn default_config() -> PhonemeVerifyConfig {
    PhonemeVerifyConfig {
        min_duration_ms: 30.0,
        max_duration_ms: 500.0,
        min_voiced_hnr_db: 5.0,
        f0_range_hz: (50.0, 600.0),
        min_energy_ratio: 0.1,
        max_mcd_db: 8.0,
    }
}

fn make_phoneme_result(label: &str, duration_ms: f64, passed: bool) -> PhonemeResult {
    PhonemeResult {
        label: label.to_string(),
        duration_ms,
        metrics: vec![QualityMetric {
            name: "duration_ms",
            value: duration_ms,
            threshold: 30.0,
            passed,
            citation: "test",
        }],
        passed,
    }
}

fn make_coverage(positions: Vec<UnicodeDerivedConfusionSet>) -> UnicodeCoverageReport {
    let total = positions.len();
    let covered = positions
        .iter()
        .filter(|p| p.covered_by_linguistic_set.is_some())
        .count();
    UnicodeCoverageReport {
        total_vulnerable: total,
        covered_by_linguistic: covered,
        uncovered: total - covered,
        coverage_ratio: if total > 0 {
            covered as f64 / total as f64
        } else {
            1.0
        },
        positions,
    }
}

fn make_unicode_pos(
    pos: usize,
    ch: char,
    vuln: VulnerabilityType,
    token_id: u32,
    covered: Option<&str>,
) -> UnicodeDerivedConfusionSet {
    UnicodeDerivedConfusionSet {
        source_position: pos,
        source_char: ch,
        vulnerability: vuln,
        phoneme_token_id: token_id,
        covered_by_linguistic_set: covered.map(ToString::to_string),
    }
}

#[test]
fn test_no_defects_no_vulnerabilities() {
    let results = vec![make_phoneme_result("AH", 100.0, true)];
    let coverage = make_coverage(vec![]);
    let config = default_config();

    let analysis = analyze_defects_with_unicode(&results, &config, &coverage, &|_, _| Some(0));

    assert_eq!(analysis.total_defects, 0);
    assert!(analysis.double_vulnerabilities.is_empty());
    assert_eq!(analysis.critical_count, 0);
    assert_eq!(analysis.high_count, 0);
    assert_eq!(analysis.medium_count, 0);
    assert_eq!(analysis.defect_vulnerability_overlap, 0.0);
}

#[test]
fn test_defect_at_vulnerable_uncovered_position_is_critical() {
    // Phoneme at position 1 has a deletion defect (too short).
    let results = vec![
        make_phoneme_result("HH", 100.0, true),
        make_phoneme_result("EH", 10.0, false), // Too short → deletion defect.
        make_phoneme_result("L", 80.0, true),
    ];

    // Position 1 is Unicode-vulnerable (homoglyph), NOT covered by any set.
    let coverage = make_coverage(vec![make_unicode_pos(
        1,
        'е', // Cyrillic е
        VulnerabilityType::Homoglyph,
        42,
        None,
    )]);

    let config = default_config();
    let analysis = analyze_defects_with_unicode(&results, &config, &coverage, &|idx, _| Some(idx));

    assert_eq!(analysis.total_defects, 1);
    assert_eq!(analysis.double_vulnerabilities.len(), 1);
    assert_eq!(analysis.critical_count, 1);
    assert_eq!(analysis.high_count, 0);
    assert_eq!(analysis.defect_vulnerability_overlap, 1.0);

    let dv = &analysis.double_vulnerabilities[0];
    assert_eq!(dv.char_index, 1);
    assert_eq!(dv.risk_level, RiskLevel::Critical);
    assert_eq!(dv.unicode_vulnerability, VulnerabilityType::Homoglyph);
}

#[test]
fn test_defect_at_covered_position_is_high() {
    let results = vec![make_phoneme_result("EH", 10.0, false)];

    // Position 0 is Unicode-vulnerable BUT covered by a linguistic confusion set.
    let coverage = make_coverage(vec![make_unicode_pos(
        0,
        'e',
        VulnerabilityType::Homoglyph,
        42,
        Some("front_vowels"),
    )]);

    let config = default_config();
    let analysis = analyze_defects_with_unicode(&results, &config, &coverage, &|_, _| Some(0));

    assert_eq!(analysis.total_defects, 1);
    assert_eq!(analysis.double_vulnerabilities.len(), 1);
    assert_eq!(analysis.critical_count, 0);
    assert_eq!(analysis.high_count, 1);
    assert_eq!(
        analysis.double_vulnerabilities[0].risk_level,
        RiskLevel::High
    );
}

#[test]
fn test_vulnerability_without_defect_is_medium() {
    // All phonemes pass — no defects.
    let results = vec![
        make_phoneme_result("HH", 100.0, true),
        make_phoneme_result("EH", 100.0, true),
    ];

    // But position 1 is Unicode-vulnerable.
    let coverage = make_coverage(vec![make_unicode_pos(
        1,
        'е',
        VulnerabilityType::Homoglyph,
        42,
        None,
    )]);

    let config = default_config();
    let analysis = analyze_defects_with_unicode(&results, &config, &coverage, &|idx, _| Some(idx));

    assert_eq!(analysis.total_defects, 0);
    assert!(analysis.double_vulnerabilities.is_empty());
    assert_eq!(analysis.medium_count, 1); // Vulnerable but no defect.
}

#[test]
fn test_defect_at_non_vulnerable_position_is_not_double() {
    // Phoneme at position 0 has a defect.
    let results = vec![make_phoneme_result("EH", 10.0, false)];

    // But no Unicode vulnerability at position 0 — vulnerability is at position 2.
    let coverage = make_coverage(vec![make_unicode_pos(
        2,
        'о',
        VulnerabilityType::MixedScript,
        50,
        None,
    )]);

    let config = default_config();
    let analysis = analyze_defects_with_unicode(&results, &config, &coverage, &|_, _| Some(0));

    assert_eq!(analysis.total_defects, 1);
    assert!(analysis.double_vulnerabilities.is_empty());
    assert_eq!(analysis.defect_vulnerability_overlap, 0.0);
    assert_eq!(analysis.medium_count, 1); // Position 2 is vulnerable but no defect.
}

#[test]
fn test_multiple_vulnerabilities_mixed_risk() {
    let results = vec![
        make_phoneme_result("HH", 100.0, true), // passes
        make_phoneme_result("EH", 10.0, false), // deletion defect
        make_phoneme_result("L", 80.0, true),   // passes
        make_phoneme_result("OW", 5.0, false),  // deletion defect
    ];

    let coverage = make_coverage(vec![
        // Position 1: uncovered homoglyph → critical (EH has defect)
        make_unicode_pos(1, 'е', VulnerabilityType::Homoglyph, 42, None),
        // Position 2: covered invisible → medium (L passes)
        make_unicode_pos(
            2,
            'l',
            VulnerabilityType::InvisibleInsertion,
            43,
            Some("liquids"),
        ),
        // Position 3: covered mixed script → high (OW has defect)
        make_unicode_pos(
            3,
            'о',
            VulnerabilityType::MixedScript,
            44,
            Some("back_vowels"),
        ),
    ]);

    let config = default_config();
    let analysis = analyze_defects_with_unicode(&results, &config, &coverage, &|idx, _| Some(idx));

    assert_eq!(analysis.total_defects, 2);
    assert_eq!(analysis.double_vulnerabilities.len(), 2);
    assert_eq!(analysis.critical_count, 1);
    assert_eq!(analysis.high_count, 1);
    assert_eq!(analysis.medium_count, 1); // Position 2 (no defect).
    assert!((analysis.defect_vulnerability_overlap - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_label_to_char_index_returns_none_skips() {
    let results = vec![make_phoneme_result("EH", 10.0, false)];

    let coverage = make_coverage(vec![make_unicode_pos(
        0,
        'е',
        VulnerabilityType::Homoglyph,
        42,
        None,
    )]);

    let config = default_config();
    // G2P mapping returns None → no character index for this phoneme.
    let analysis = analyze_defects_with_unicode(&results, &config, &coverage, &|_, _| None);

    assert_eq!(analysis.total_defects, 1);
    assert!(analysis.double_vulnerabilities.is_empty());
}

#[test]
fn test_classify_pronunciation_impact() {
    let homoglyph_pos = make_unicode_pos(0, 'а', VulnerabilityType::Homoglyph, 1, None);
    assert_eq!(
        classify_pronunciation_impact(&homoglyph_pos),
        PronunciationImpact::PhonemeSubstitution
    );

    let invisible_pos = make_unicode_pos(0, 'a', VulnerabilityType::InvisibleInsertion, 1, None);
    assert_eq!(
        classify_pronunciation_impact(&invisible_pos),
        PronunciationImpact::BoundaryShift
    );

    let mixed_pos = make_unicode_pos(0, 'а', VulnerabilityType::MixedScript, 1, None);
    assert_eq!(
        classify_pronunciation_impact(&mixed_pos),
        PronunciationImpact::ModelConfusion
    );
}

#[test]
fn test_risk_level_ordering() {
    // Critical > High > Medium.
    assert!(RiskLevel::Critical < RiskLevel::High);
    assert!(RiskLevel::High < RiskLevel::Medium);
}

#[test]
fn test_insertion_defect_at_invisible_char_position() {
    // Long phoneme (insertion defect) at a position vulnerable to invisible char insertion.
    let results = vec![make_phoneme_result("AH", 600.0, false)]; // Too long → insertion.

    let coverage = make_coverage(vec![make_unicode_pos(
        0,
        'a',
        VulnerabilityType::InvisibleInsertion,
        30,
        None,
    )]);

    let config = default_config();
    let analysis = analyze_defects_with_unicode(&results, &config, &coverage, &|_, _| Some(0));

    assert_eq!(analysis.total_defects, 1);
    assert_eq!(analysis.double_vulnerabilities.len(), 1);
    assert_eq!(analysis.critical_count, 1);

    let dv = &analysis.double_vulnerabilities[0];
    assert_eq!(
        dv.unicode_vulnerability,
        VulnerabilityType::InvisibleInsertion
    );
    matches!(&dv.defect, PronunciationDefect::Insertion { .. });
}
