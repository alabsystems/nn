// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bridge between Unicode safety scanning and formal adversarial robustness verification.
//!
//! Maps text-level Unicode attacks (homoglyphs, invisible characters) to
//! embedding-space perturbation bounds that can be CROWN-verified. This
//! connects AC1 (perturbation sets) to AC2 (CROWN phoneme stability) of #1740.
//!
//! The key insight: homoglyph attacks operate at the grapheme level (before G2P),
//! but formal verification operates at the embedding level (after G2P). This module
//! bridges the gap by:
//! 1. Identifying which text positions are vulnerable to Unicode substitution
//! 2. Mapping those positions to phoneme confusion sets
//! 3. Computing embedding bounds that cover all possible Unicode attacks
//!
//! # Example
//!
//! ```rust,ignore
//! use nn_tts_verify::{scan_unicode, UnicodeSafetyConfig};
//! use nn_tts_verify::unicode_perturbation::*;
//!
//! let config = UnicodeSafetyConfig::default();
//! let scan = scan_unicode("hеllo", &config); // 'е' is Cyrillic
//! let vuln = identify_vulnerable_positions("hеllo", &config);
//! assert_eq!(vuln.len(), 1); // position 1 is vulnerable
//! ```

use crate::adversarial::{ConfusionCategory, ConfusionSet};
use crate::unicode_safety::{scan_unicode, UnicodeAttack, UnicodeSafetyConfig};

/// A text position vulnerable to Unicode attack, with its perturbation context.
#[derive(Debug, Clone)]
pub struct VulnerablePosition {
    /// Character index in the text (0-based).
    pub char_index: usize,
    /// The original character at this position.
    pub original_char: char,
    /// The attack type detected.
    pub attack_type: VulnerabilityType,
    /// The canonical (safe) character, if applicable.
    pub canonical: Option<char>,
}

/// Type of Unicode vulnerability at a text position.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum VulnerabilityType {
    /// Character could be replaced with a visually identical homoglyph.
    Homoglyph,
    /// Position could have invisible characters inserted adjacent to it.
    InvisibleInsertion,
    /// Character is from an unexpected script (mixed-script attack).
    MixedScript,
}

/// Identify text positions vulnerable to Unicode adversarial attacks.
///
/// Scans the text for:
/// 1. Characters that have known homoglyph confusables (even if the input is clean)
/// 2. Positions where invisible characters could be inserted
/// 3. Characters from unexpected scripts
///
/// Returns positions sorted by char_index.
pub fn identify_vulnerable_positions(
    text: &str,
    config: &UnicodeSafetyConfig,
) -> Vec<VulnerablePosition> {
    let scan = scan_unicode(text, config);
    let mut positions = Vec::new();

    // First, record detected attacks from the scan.
    for attack in &scan.attacks {
        match attack {
            UnicodeAttack::Homoglyph {
                original,
                confusable,
                byte_offset,
            } => {
                let char_idx = text[..*byte_offset].chars().count();
                positions.push(VulnerablePosition {
                    char_index: char_idx,
                    original_char: *original,
                    attack_type: VulnerabilityType::Homoglyph,
                    canonical: Some(*confusable),
                });
            }
            UnicodeAttack::InvisibleChar { byte_offset, .. } => {
                let char_idx = text[..*byte_offset].chars().count();
                // The character before the invisible char is the vulnerable position.
                if char_idx > 0 {
                    let prev_char = text.chars().nth(char_idx.saturating_sub(1)).unwrap_or(' ');
                    positions.push(VulnerablePosition {
                        char_index: char_idx.saturating_sub(1),
                        original_char: prev_char,
                        attack_type: VulnerabilityType::InvisibleInsertion,
                        canonical: None,
                    });
                }
            }
            UnicodeAttack::UnexpectedScript {
                char: ch,
                byte_offset,
                ..
            } => {
                let char_idx = text[..*byte_offset].chars().count();
                positions.push(VulnerablePosition {
                    char_index: char_idx,
                    original_char: *ch,
                    attack_type: VulnerabilityType::MixedScript,
                    canonical: None,
                });
            }
            UnicodeAttack::BidiOverride { .. } => {
                // Bidi overrides are always stripped — not a perturbation, but a removal.
            }
        }
    }

    // Additionally, identify Latin characters that COULD be attacked with homoglyphs.
    // Even in clean text, these positions are vulnerable to future substitution.
    let confusables = crate::unicode_safety::tts_confusables();
    for (char_idx, ch) in text.chars().enumerate() {
        // Check if this Latin character has a known homoglyph.
        let has_confusable = confusables.iter().any(|(_, canonical)| *canonical == ch);
        if has_confusable && !positions.iter().any(|p| p.char_index == char_idx) {
            positions.push(VulnerablePosition {
                char_index: char_idx,
                original_char: ch,
                attack_type: VulnerabilityType::Homoglyph,
                canonical: Some(ch), // Already canonical.
            });
        }
    }

    positions.sort_by_key(|p| p.char_index);
    positions.dedup_by_key(|p| p.char_index);
    positions
}

/// Map vulnerable text positions to phoneme-level confusion sets.
///
/// Given a G2P mapping (character index → phoneme token ID) and the vulnerable
/// positions, returns confusion sets that cover the phoneme tokens at risk.
///
/// The `char_to_token` mapping should provide the phoneme token ID for each
/// character position. Characters that don't map to phonemes (spaces, punctuation)
/// return `None`.
///
/// This is the bridge function: it connects text-level vulnerability analysis
/// (Unicode attacks) to embedding-level perturbation bounds (confusion sets).
pub fn map_to_phoneme_confusion_sets(
    vulnerable: &[VulnerablePosition],
    char_to_token: &dyn Fn(usize) -> Option<u32>,
    existing_sets: &[ConfusionSet],
) -> Vec<UnicodeDerivedConfusionSet> {
    let mut derived_sets = Vec::new();

    for vuln in vulnerable {
        let Some(token_id) = char_to_token(vuln.char_index) else {
            continue; // Position doesn't map to a phoneme (space, punctuation).
        };

        // Check if this token is already covered by a linguistic confusion set.
        let existing_coverage = existing_sets
            .iter()
            .find(|cs| cs.token_ids.contains(&token_id));

        derived_sets.push(UnicodeDerivedConfusionSet {
            source_position: vuln.char_index,
            source_char: vuln.original_char,
            vulnerability: vuln.attack_type.clone(),
            phoneme_token_id: token_id,
            covered_by_linguistic_set: existing_coverage.map(|cs| cs.name.clone()),
        });
    }

    derived_sets
}

/// A confusion set derived from Unicode vulnerability analysis.
#[derive(Debug, Clone)]
pub struct UnicodeDerivedConfusionSet {
    /// Character position in the source text.
    pub source_position: usize,
    /// The character at this position.
    pub source_char: char,
    /// Type of Unicode vulnerability.
    pub vulnerability: VulnerabilityType,
    /// The phoneme token ID at this position (after G2P).
    pub phoneme_token_id: u32,
    /// If non-None, this position is already covered by an existing linguistic
    /// confusion set (name of the set). No additional perturbation bounds needed.
    pub covered_by_linguistic_set: Option<String>,
}

/// Summary of Unicode perturbation coverage analysis.
#[derive(Debug, Clone)]
pub struct UnicodeCoverageReport {
    /// Total vulnerable positions in the text.
    pub total_vulnerable: usize,
    /// Positions covered by existing linguistic confusion sets.
    pub covered_by_linguistic: usize,
    /// Positions needing additional perturbation bounds.
    pub uncovered: usize,
    /// Coverage ratio (covered / total).
    pub coverage_ratio: f64,
    /// Per-position details.
    pub positions: Vec<UnicodeDerivedConfusionSet>,
}

/// Analyze how well existing phoneme confusion sets cover Unicode attack surfaces.
///
/// Returns a coverage report showing which vulnerable positions are already
/// protected by linguistic confusion sets and which need additional bounds.
pub fn analyze_unicode_coverage(
    text: &str,
    config: &UnicodeSafetyConfig,
    char_to_token: &dyn Fn(usize) -> Option<u32>,
    existing_sets: &[ConfusionSet],
) -> UnicodeCoverageReport {
    let vulnerable = identify_vulnerable_positions(text, config);
    let derived = map_to_phoneme_confusion_sets(&vulnerable, char_to_token, existing_sets);

    let total = derived.len();
    let covered = derived
        .iter()
        .filter(|d| d.covered_by_linguistic_set.is_some())
        .count();
    let uncovered = total - covered;

    UnicodeCoverageReport {
        total_vulnerable: total,
        covered_by_linguistic: covered,
        uncovered,
        coverage_ratio: if total > 0 {
            covered as f64 / total as f64
        } else {
            1.0
        },
        positions: derived,
    }
}

/// Build expanded confusion sets that include Unicode-derived perturbations.
///
/// For each uncovered vulnerable position, creates a single-token "confusion set"
/// representing the point bounds at that token. For covered positions, the existing
/// linguistic confusion set already provides the perturbation bounds.
///
/// Returns the original sets plus any new Unicode-derived sets.
pub fn expand_confusion_sets_for_unicode(
    existing_sets: &[ConfusionSet],
    coverage: &UnicodeCoverageReport,
) -> Vec<ConfusionSet> {
    let mut expanded = existing_sets.to_vec();

    for pos in &coverage.positions {
        if pos.covered_by_linguistic_set.is_some() {
            continue; // Already covered.
        }

        // Create a minimal confusion set for this uncovered position.
        // Single-token set = point bounds (tightest possible).
        expanded.push(ConfusionSet {
            name: format!("unicode_derived_pos{}", pos.source_position),
            token_ids: vec![pos.phoneme_token_id],
            labels: vec![format!(
                "unicode_{}",
                match pos.vulnerability {
                    VulnerabilityType::Homoglyph => "homoglyph",
                    VulnerabilityType::InvisibleInsertion => "invisible",
                    VulnerabilityType::MixedScript => "mixed_script",
                }
            )],
            category: ConfusionCategory::EmbeddingSimilar,
        });
    }

    expanded
}

#[cfg(test)]
#[path = "unicode_perturbation_tests.rs"]
mod tests;
