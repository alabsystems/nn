// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unicode adversarial safety certificate for TTS pipelines.
//!
//! Wraps the combined pronunciation-defect + Unicode-vulnerability analysis
//! (from [`pronunciation_unicode`](crate::pronunciation_unicode)) into a
//! composable certificate with a human-readable report.
//!
//! This is AC4 of #1740: certificate enrichment with Unicode vulnerability metadata.
//!
//! # Design
//!
//! Follows the codebase pattern of composable certificates (`HybridCertificate`,
//! `MoonshotCertificate`, `CoupledTimingCertificate`) rather than extending
//! the base `Certificate` struct. This avoids breaking 74 construction sites
//! and keeps the certificate hierarchy clean.

use crate::pronunciation_unicode::{DoubleVulnerability, RiskLevel, UnicodeDefectAnalysis};
use crate::unicode_perturbation::VulnerabilityType;

/// Unicode adversarial safety certificate for a TTS utterance.
///
/// Combines the pronunciation-defect analysis with Unicode vulnerability
/// data to produce a composable certificate with risk assessment.
///
/// # Example
///
/// ```rust,ignore
/// use nn_tts_verify::unicode_certificate::UnicodeSafetyCertificate;
///
/// let cert = UnicodeSafetyCertificate::from_analysis(analysis);
/// assert!(cert.is_safe());
/// let _report = cert.report();
/// ```
#[derive(Debug, Clone)]
pub struct UnicodeSafetyCertificate {
    /// The underlying defect + vulnerability analysis.
    pub analysis: UnicodeDefectAnalysis,
    /// Whether the utterance passes Unicode safety checks.
    ///
    /// True when there are zero critical-risk double vulnerabilities.
    pub passed: bool,
    /// Total number of Unicode-vulnerable character positions examined.
    pub total_vulnerable_positions: usize,
    /// Number of double-vulnerability positions (defect + Unicode exposure).
    pub double_vulnerability_count: usize,
}

impl UnicodeSafetyCertificate {
    /// Construct a certificate from a completed `UnicodeDefectAnalysis`.
    ///
    /// The certificate passes if there are zero critical-risk positions
    /// (i.e., no positions where both a pronunciation defect AND an
    /// uncovered Unicode vulnerability exist).
    pub fn from_analysis(analysis: UnicodeDefectAnalysis) -> Self {
        let total_vulnerable =
            analysis.critical_count + analysis.high_count + analysis.medium_count;
        let double_count = analysis.double_vulnerabilities.len();
        let passed = analysis.critical_count == 0;

        Self {
            analysis,
            passed,
            total_vulnerable_positions: total_vulnerable,
            double_vulnerability_count: double_count,
        }
    }

    /// Whether the certificate passes (no critical double vulnerabilities).
    pub fn is_safe(&self) -> bool {
        self.passed
    }

    /// Generate a human-readable report of Unicode safety findings.
    pub fn report(&self) -> String {
        let mut r = String::with_capacity(512);
        r.push_str("=== Unicode Safety Certificate ===\n\n");

        r.push_str(&format!(
            "Status: {}\n",
            if self.passed { "SAFE" } else { "UNSAFE" }
        ));
        r.push_str(&format!(
            "Total pronunciation defects: {}\n",
            self.analysis.total_defects
        ));
        r.push_str(&format!(
            "Unicode-vulnerable positions: {}\n",
            self.total_vulnerable_positions
        ));
        r.push_str(&format!(
            "Double vulnerabilities: {}\n",
            self.double_vulnerability_count
        ));

        // Risk breakdown.
        r.push_str(&format!(
            "\n-- Risk Breakdown --\n  Critical: {}\n  High: {}\n  Medium: {}\n",
            self.analysis.critical_count, self.analysis.high_count, self.analysis.medium_count,
        ));

        if self.analysis.total_defects > 0 {
            r.push_str(&format!(
                "\nDefect-vulnerability overlap: {:.1}%\n",
                self.analysis.defect_vulnerability_overlap * 100.0
            ));
        }

        // Detail each double vulnerability.
        if !self.analysis.double_vulnerabilities.is_empty() {
            r.push_str("\n-- Double Vulnerabilities --\n");
            for dv in &self.analysis.double_vulnerabilities {
                r.push_str(&format!(
                    "  [{}] pos={}, char='{}', type={}, token_id={}\n",
                    risk_label(dv.risk_level),
                    dv.char_index,
                    dv.source_char,
                    vulnerability_label(&dv.unicode_vulnerability),
                    dv.phoneme_token_id,
                ));
            }
        }

        r
    }
}

/// Human-readable label for a risk level.
fn risk_label(level: RiskLevel) -> &'static str {
    match level {
        RiskLevel::Critical => "CRITICAL",
        RiskLevel::High => "HIGH",
        RiskLevel::Medium => "MEDIUM",
    }
}

/// Human-readable label for a vulnerability type.
fn vulnerability_label(v: &VulnerabilityType) -> &'static str {
    match v {
        VulnerabilityType::Homoglyph => "homoglyph",
        VulnerabilityType::InvisibleInsertion => "invisible_insertion",
        VulnerabilityType::MixedScript => "mixed_script",
    }
}

/// Summarize double vulnerabilities for embedding in other certificate reports.
///
/// Returns a compact one-line summary suitable for including in a
/// `MoonshotCertificate` or `HybridCertificate` report section.
pub fn unicode_safety_summary(cert: &UnicodeSafetyCertificate) -> String {
    if cert.passed {
        format!(
            "Unicode safety: SAFE ({} vulnerable positions, 0 critical)",
            cert.total_vulnerable_positions,
        )
    } else {
        format!(
            "Unicode safety: UNSAFE ({} critical, {} high, {} medium)",
            cert.analysis.critical_count, cert.analysis.high_count, cert.analysis.medium_count,
        )
    }
}

/// Classify a set of double vulnerabilities by their dominant attack vector.
///
/// Returns the most common vulnerability type among the double vulnerabilities,
/// useful for prioritizing which Unicode normalization defense to apply first.
pub fn dominant_attack_vector(vulns: &[DoubleVulnerability]) -> Option<VulnerabilityType> {
    if vulns.is_empty() {
        return None;
    }
    let mut homoglyph = 0usize;
    let mut invisible = 0usize;
    let mut mixed = 0usize;

    for v in vulns {
        match v.unicode_vulnerability {
            VulnerabilityType::Homoglyph => homoglyph += 1,
            VulnerabilityType::InvisibleInsertion => invisible += 1,
            VulnerabilityType::MixedScript => mixed += 1,
        }
    }

    let max = homoglyph.max(invisible).max(mixed);
    if max == 0 {
        return None;
    }
    // Tiebreak: homoglyph > invisible > mixed (by severity ordering).
    if homoglyph == max {
        Some(VulnerabilityType::Homoglyph)
    } else if invisible == max {
        Some(VulnerabilityType::InvisibleInsertion)
    } else {
        Some(VulnerabilityType::MixedScript)
    }
}

#[cfg(test)]
#[path = "unicode_certificate_tests.rs"]
mod tests;
