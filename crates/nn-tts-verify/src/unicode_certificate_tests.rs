// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `UnicodeSafetyCertificate` and associated functions.

use super::*;
use crate::phoneme_defects::PronunciationDefect;
use crate::pronunciation_unicode::{DoubleVulnerability, RiskLevel, UnicodeDefectAnalysis};
use crate::unicode_perturbation::VulnerabilityType;

fn make_dv(idx: usize, ch: char, vuln: VulnerabilityType, risk: RiskLevel) -> DoubleVulnerability {
    DoubleVulnerability {
        char_index: idx,
        source_char: ch,
        unicode_vulnerability: vuln,
        defect: PronunciationDefect::Deletion {
            label: "EH".to_string(),
            duration_ms: 10.0,
        },
        phoneme_token_id: 42,
        risk_level: risk,
    }
}

fn empty_analysis() -> UnicodeDefectAnalysis {
    UnicodeDefectAnalysis {
        total_defects: 0,
        double_vulnerabilities: vec![],
        critical_count: 0,
        high_count: 0,
        medium_count: 0,
        defect_vulnerability_overlap: 0.0,
    }
}

#[test]
fn test_safe_certificate_no_vulnerabilities() {
    let cert = UnicodeSafetyCertificate::from_analysis(empty_analysis());
    assert!(cert.is_safe());
    assert!(cert.passed);
    assert_eq!(cert.total_vulnerable_positions, 0);
    assert_eq!(cert.double_vulnerability_count, 0);
}

#[test]
fn test_safe_certificate_medium_only() {
    // Medium risk = Unicode-vulnerable but no pronunciation defect.
    // This is safe (no critical).
    let analysis = UnicodeDefectAnalysis {
        total_defects: 0,
        double_vulnerabilities: vec![],
        critical_count: 0,
        high_count: 0,
        medium_count: 3,
        defect_vulnerability_overlap: 0.0,
    };

    let cert = UnicodeSafetyCertificate::from_analysis(analysis);
    assert!(cert.is_safe());
    assert_eq!(cert.total_vulnerable_positions, 3);
    assert_eq!(cert.double_vulnerability_count, 0);
}

#[test]
fn test_unsafe_certificate_with_critical() {
    let analysis = UnicodeDefectAnalysis {
        total_defects: 1,
        double_vulnerabilities: vec![make_dv(
            1,
            'е',
            VulnerabilityType::Homoglyph,
            RiskLevel::Critical,
        )],
        critical_count: 1,
        high_count: 0,
        medium_count: 0,
        defect_vulnerability_overlap: 1.0,
    };

    let cert = UnicodeSafetyCertificate::from_analysis(analysis);
    assert!(!cert.is_safe());
    assert!(!cert.passed);
    assert_eq!(cert.total_vulnerable_positions, 1);
    assert_eq!(cert.double_vulnerability_count, 1);
}

#[test]
fn test_safe_with_high_risk_only() {
    // High risk = defect at covered Unicode position. Still "safe" per our
    // definition (no critical), but the certificate carries the high count.
    let analysis = UnicodeDefectAnalysis {
        total_defects: 1,
        double_vulnerabilities: vec![make_dv(
            0,
            'e',
            VulnerabilityType::Homoglyph,
            RiskLevel::High,
        )],
        critical_count: 0,
        high_count: 1,
        medium_count: 0,
        defect_vulnerability_overlap: 1.0,
    };

    let cert = UnicodeSafetyCertificate::from_analysis(analysis);
    assert!(cert.is_safe());
    assert_eq!(cert.double_vulnerability_count, 1);
    assert_eq!(cert.analysis.high_count, 1);
}

#[test]
fn test_report_contains_status() {
    let cert = UnicodeSafetyCertificate::from_analysis(empty_analysis());
    let report = cert.report();
    assert!(report.contains("Status: SAFE"));
    assert!(report.contains("Unicode Safety Certificate"));
    assert!(report.contains("Critical: 0"));
}

#[test]
fn test_report_unsafe_contains_critical() {
    let analysis = UnicodeDefectAnalysis {
        total_defects: 2,
        double_vulnerabilities: vec![
            make_dv(1, 'е', VulnerabilityType::Homoglyph, RiskLevel::Critical),
            make_dv(3, 'о', VulnerabilityType::MixedScript, RiskLevel::High),
        ],
        critical_count: 1,
        high_count: 1,
        medium_count: 1,
        defect_vulnerability_overlap: 1.0,
    };

    let cert = UnicodeSafetyCertificate::from_analysis(analysis);
    let report = cert.report();
    assert!(report.contains("Status: UNSAFE"));
    assert!(report.contains("Critical: 1"));
    assert!(report.contains("High: 1"));
    assert!(report.contains("Medium: 1"));
    assert!(report.contains("Double Vulnerabilities"));
    assert!(report.contains("[CRITICAL]"));
    assert!(report.contains("homoglyph"));
    assert!(report.contains("[HIGH]"));
    assert!(report.contains("mixed_script"));
    assert!(report.contains("Defect-vulnerability overlap: 100.0%"));
}

#[test]
fn test_report_no_overlap_section_when_zero_defects() {
    let cert = UnicodeSafetyCertificate::from_analysis(empty_analysis());
    let report = cert.report();
    assert!(!report.contains("overlap"));
}

#[test]
fn test_summary_safe() {
    let cert = UnicodeSafetyCertificate::from_analysis(empty_analysis());
    let summary = unicode_safety_summary(&cert);
    assert!(summary.starts_with("Unicode safety: SAFE"));
    assert!(summary.contains("0 critical"));
}

#[test]
fn test_summary_unsafe() {
    let analysis = UnicodeDefectAnalysis {
        total_defects: 1,
        double_vulnerabilities: vec![make_dv(
            0,
            'а',
            VulnerabilityType::Homoglyph,
            RiskLevel::Critical,
        )],
        critical_count: 1,
        high_count: 0,
        medium_count: 2,
        defect_vulnerability_overlap: 1.0,
    };

    let cert = UnicodeSafetyCertificate::from_analysis(analysis);
    let summary = unicode_safety_summary(&cert);
    assert!(summary.starts_with("Unicode safety: UNSAFE"));
    assert!(summary.contains("1 critical"));
}

#[test]
fn test_dominant_attack_vector_empty() {
    assert_eq!(dominant_attack_vector(&[]), None);
}

#[test]
fn test_dominant_attack_vector_homoglyph() {
    let vulns = vec![
        make_dv(0, 'а', VulnerabilityType::Homoglyph, RiskLevel::Critical),
        make_dv(1, 'е', VulnerabilityType::Homoglyph, RiskLevel::Critical),
        make_dv(
            2,
            'a',
            VulnerabilityType::InvisibleInsertion,
            RiskLevel::High,
        ),
    ];
    assert_eq!(
        dominant_attack_vector(&vulns),
        Some(VulnerabilityType::Homoglyph)
    );
}

#[test]
fn test_dominant_attack_vector_invisible() {
    let vulns = vec![
        make_dv(
            0,
            'a',
            VulnerabilityType::InvisibleInsertion,
            RiskLevel::Critical,
        ),
        make_dv(
            1,
            'b',
            VulnerabilityType::InvisibleInsertion,
            RiskLevel::High,
        ),
        make_dv(2, 'c', VulnerabilityType::MixedScript, RiskLevel::Medium),
    ];
    assert_eq!(
        dominant_attack_vector(&vulns),
        Some(VulnerabilityType::InvisibleInsertion)
    );
}

#[test]
fn test_dominant_attack_vector_tiebreak_prefers_homoglyph() {
    let vulns = vec![
        make_dv(0, 'а', VulnerabilityType::Homoglyph, RiskLevel::Critical),
        make_dv(
            1,
            'a',
            VulnerabilityType::InvisibleInsertion,
            RiskLevel::Critical,
        ),
    ];
    // Tiebreak: homoglyph > invisible.
    assert_eq!(
        dominant_attack_vector(&vulns),
        Some(VulnerabilityType::Homoglyph)
    );
}

#[test]
fn test_total_vulnerable_positions_sums_all_risk_levels() {
    let analysis = UnicodeDefectAnalysis {
        total_defects: 3,
        double_vulnerabilities: vec![
            make_dv(0, 'а', VulnerabilityType::Homoglyph, RiskLevel::Critical),
            make_dv(1, 'е', VulnerabilityType::MixedScript, RiskLevel::High),
        ],
        critical_count: 1,
        high_count: 1,
        medium_count: 5,
        defect_vulnerability_overlap: 0.67,
    };

    let cert = UnicodeSafetyCertificate::from_analysis(analysis);
    // total = critical + high + medium = 1 + 1 + 5 = 7
    assert_eq!(cert.total_vulnerable_positions, 7);
    assert_eq!(cert.double_vulnerability_count, 2);
}
