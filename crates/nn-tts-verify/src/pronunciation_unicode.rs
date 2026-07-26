// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bridge between Unicode adversarial analysis and pronunciation defect detection.
//!
//! Given a Unicode coverage report (from [`unicode_perturbation`](crate::unicode_perturbation))
//! and phoneme verification results (from [`phoneme_verify`](crate::phoneme_verify)),
//! produces a combined analysis showing which pronunciation defects are attributable
//! to Unicode attack surfaces and which positions are doubly vulnerable (Unicode-exposed
//! AND pronunciation-weak).
//!
//! This is AC3 of #1740: pronunciation defect detection with Unicode-aware confusion sets.

use crate::phoneme::{PhonemeResult, PhonemeVerifyConfig};
use crate::phoneme_defects::{detect_defects, PronunciationDefect};
use crate::unicode_perturbation::{
    UnicodeCoverageReport, UnicodeDerivedConfusionSet, VulnerabilityType,
};

/// A position that is both Unicode-vulnerable and has a pronunciation defect.
///
/// These are the highest-risk positions: an adversarial Unicode substitution
/// at this position could cause a pronunciation defect that is ALREADY present
/// in the clean output, making it harder to distinguish attacks from natural errors.
#[derive(Debug, Clone)]
pub struct DoubleVulnerability {
    /// Character position in the source text.
    pub char_index: usize,
    /// The original character.
    pub source_char: char,
    /// Type of Unicode vulnerability.
    pub unicode_vulnerability: VulnerabilityType,
    /// The pronunciation defect detected at this phoneme.
    pub defect: PronunciationDefect,
    /// The phoneme token ID (from G2P mapping).
    pub phoneme_token_id: u32,
    /// Risk level: combination of vulnerability type and defect severity.
    pub risk_level: RiskLevel,
}

/// Risk classification for double vulnerabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum RiskLevel {
    /// Pronunciation defect at an uncovered Unicode-vulnerable position.
    /// Highest risk: no linguistic confusion set protects this position.
    Critical,
    /// Pronunciation defect at a Unicode-vulnerable position covered by
    /// a linguistic confusion set. The perturbation bounds may still be too wide.
    High,
    /// Unicode-vulnerable position without a pronunciation defect.
    /// The clean output is correct, but an attack could introduce a defect.
    Medium,
}

/// Combined analysis of Unicode vulnerabilities and pronunciation defects.
#[derive(Debug, Clone)]
pub struct UnicodeDefectAnalysis {
    /// Total pronunciation defects detected (independent of Unicode analysis).
    pub total_defects: usize,
    /// Unicode-vulnerable positions with pronunciation defects (double vulnerabilities).
    pub double_vulnerabilities: Vec<DoubleVulnerability>,
    /// Count of defects at uncovered Unicode positions (highest risk).
    pub critical_count: usize,
    /// Count of defects at covered Unicode positions.
    pub high_count: usize,
    /// Unicode-vulnerable positions without pronunciation defects.
    pub medium_count: usize,
    /// Fraction of pronunciation defects that coincide with Unicode vulnerabilities.
    /// High values suggest the model is weakest exactly where attacks are most likely.
    pub defect_vulnerability_overlap: f64,
}

/// Analyze pronunciation defects in the context of Unicode vulnerability.
///
/// Combines:
/// - Per-phoneme verification results (from `verify_phonemes`)
/// - Unicode coverage report (from `analyze_unicode_coverage`)
/// - A G2P mapping from phoneme labels to character positions
///
/// The `label_to_char_index` function maps a phoneme label at a given phoneme
/// index to the character index in the source text. This reverses the G2P mapping
/// to connect phoneme-level defects back to text-level vulnerabilities.
///
/// # Example
///
/// ```rust,ignore
/// use nn_tts_verify::pronunciation_unicode::*;
///
/// let analysis = analyze_defects_with_unicode(
///     &phoneme_results,
///     &verify_config,
///     &coverage_report,
///     &|phoneme_idx, _label| Some(phoneme_idx), // trivial 1:1 mapping
/// );
/// assert!(analysis.critical_count == 0, "no critical double vulnerabilities");
/// ```
pub fn analyze_defects_with_unicode(
    phoneme_results: &[PhonemeResult],
    config: &PhonemeVerifyConfig,
    coverage: &UnicodeCoverageReport,
    label_to_char_index: &dyn Fn(usize, &str) -> Option<usize>,
) -> UnicodeDefectAnalysis {
    let defects = detect_defects(phoneme_results, config);
    let total_defects = defects.len();

    let mut double_vulns = Vec::new();

    // For each defect, check if the phoneme position maps to a Unicode-vulnerable position.
    for (phoneme_idx, result) in phoneme_results.iter().enumerate() {
        if result.passed {
            continue;
        }

        let Some(char_idx) = label_to_char_index(phoneme_idx, &result.label) else {
            continue;
        };

        // Find the corresponding Unicode vulnerability at this character position.
        let unicode_pos = coverage
            .positions
            .iter()
            .find(|p| p.source_position == char_idx);

        let Some(unicode_pos) = unicode_pos else {
            continue; // Defect at a non-vulnerable position — not a double vulnerability.
        };

        // Find the matching defect from our detection.
        let matching_defect = defects.iter().find(|d| defect_label(d) == result.label);

        let Some(defect) = matching_defect else {
            continue;
        };

        let risk = if unicode_pos.covered_by_linguistic_set.is_some() {
            RiskLevel::High
        } else {
            RiskLevel::Critical
        };

        double_vulns.push(DoubleVulnerability {
            char_index: char_idx,
            source_char: unicode_pos.source_char,
            unicode_vulnerability: unicode_pos.vulnerability.clone(),
            defect: defect.clone(),
            phoneme_token_id: unicode_pos.phoneme_token_id,
            risk_level: risk,
        });
    }

    // Count Unicode-vulnerable positions WITHOUT pronunciation defects (medium risk).
    let medium_count = count_medium_risk(coverage, &double_vulns);
    let critical_count = double_vulns
        .iter()
        .filter(|v| v.risk_level == RiskLevel::Critical)
        .count();
    let high_count = double_vulns
        .iter()
        .filter(|v| v.risk_level == RiskLevel::High)
        .count();

    let defect_vulnerability_overlap = if total_defects > 0 {
        double_vulns.len() as f64 / total_defects as f64
    } else {
        0.0
    };

    UnicodeDefectAnalysis {
        total_defects,
        double_vulnerabilities: double_vulns,
        critical_count,
        high_count,
        medium_count,
        defect_vulnerability_overlap,
    }
}

/// Count Unicode-vulnerable positions that have NO pronunciation defect.
fn count_medium_risk(
    coverage: &UnicodeCoverageReport,
    double_vulns: &[DoubleVulnerability],
) -> usize {
    coverage
        .positions
        .iter()
        .filter(|pos| {
            !double_vulns
                .iter()
                .any(|dv| dv.char_index == pos.source_position)
        })
        .count()
}

/// Extract the phoneme label from a pronunciation defect.
fn defect_label(defect: &PronunciationDefect) -> &str {
    match defect {
        PronunciationDefect::Deletion { label, .. }
        | PronunciationDefect::Insertion { label, .. }
        | PronunciationDefect::Devoicing { label, .. }
        | PronunciationDefect::Substitution { label, .. }
        | PronunciationDefect::WeakArticulation { label, .. } => label,
    }
}

/// Classify a `UnicodeDerivedConfusionSet` by its expected pronunciation impact.
///
/// Different Unicode attack types have different expected effects on pronunciation:
/// - Homoglyphs: may cause phoneme substitution (different G2P output)
/// - Invisible insertion: may cause word boundary shifts or extra phonemes
/// - Mixed script: may cause G2P model confusion or OOV behavior
pub fn classify_pronunciation_impact(
    unicode_set: &UnicodeDerivedConfusionSet,
) -> PronunciationImpact {
    match unicode_set.vulnerability {
        VulnerabilityType::Homoglyph => PronunciationImpact::PhonemeSubstitution,
        VulnerabilityType::InvisibleInsertion => PronunciationImpact::BoundaryShift,
        VulnerabilityType::MixedScript => PronunciationImpact::ModelConfusion,
    }
}

/// Expected pronunciation impact from a Unicode attack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PronunciationImpact {
    /// Homoglyph substitution → different G2P output → different phoneme.
    PhonemeSubstitution,
    /// Invisible character insertion → word boundary shift → extra/missing phonemes.
    BoundaryShift,
    /// Mixed-script character → G2P model confusion → unpredictable output.
    ModelConfusion,
}

#[cfg(test)]
#[path = "pronunciation_unicode_tests.rs"]
mod tests;
