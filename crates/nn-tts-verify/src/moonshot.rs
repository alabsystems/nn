// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Moonshot verification tracker — First Provably Correct Voice.
//!
//! Tracks verification status for 8 formal properties of the Kokoro TTS pipeline:
//! non-silence, non-clipping, intelligibility (attention monotonicity),
//! speaker consistency, temporal boundedness, streaming safety, memory safety,
//! and implementation correctness.
//!
//! Each property maps to concrete verification artifacts (CROWN bounds, Kani
//! harnesses, ay SMT proofs) already present in the nn workspace.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Verification evidence level for a moonshot property.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum VerificationLevel {
    /// No verification evidence.
    None,
    /// Empirical tests pass (hard bounds, quality metrics).
    Empirical,
    /// CROWN bounds exist but may be vacuously wide.
    CrownPartial,
    /// CROWN range + Hoeffding concentration: 99% confidence probabilistic bound.
    ///
    /// When deterministic CROWN bounds are too wide to prove a property, combine
    /// the CROWN range with empirical Monte Carlo samples via Hoeffding inequality.
    /// Stronger than CrownPartial (has confidence guarantee), weaker than CrownProven
    /// (probabilistic, not deterministic). Counts as "verified" for #2463 acceptance.
    CrownProbabilistic,
    /// CROWN bounds are tight and the property is proven.
    CrownProven,
    /// Kani model-checks the Rust implementation.
    KaniProven,
    /// ay SMT proves the mathematical specification.
    SmtProven,
}

impl fmt::Display for VerificationLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "NONE"),
            Self::Empirical => write!(f, "EMPIRICAL"),
            Self::CrownPartial => write!(f, "CROWN_PARTIAL"),
            Self::CrownProbabilistic => write!(f, "CROWN_PROBABILISTIC"),
            Self::CrownProven => write!(f, "CROWN_PROVEN"),
            Self::KaniProven => write!(f, "KANI_PROVEN"),
            Self::SmtProven => write!(f, "SMT_PROVEN"),
        }
    }
}

/// Verification artifact that provides evidence for one or more properties.
#[derive(Debug, Clone)]
pub struct VerificationArtifact {
    /// Human-readable description.
    pub description: &'static str,
    /// File path relative to repo root.
    pub file: &'static str,
    /// Which moonshot properties this covers (0-indexed).
    pub properties: &'static [usize],
    /// What level of verification this provides.
    pub level: VerificationLevel,
}

/// Status of a single moonshot property.
#[derive(Debug, Clone)]
pub struct PropertyStatus {
    /// Property name (e.g., "Non-silent").
    pub name: &'static str,
    /// Best verification level achieved.
    pub verified: VerificationLevel,
    /// Verification artifacts providing evidence.
    pub evidence: Vec<&'static str>,
    /// Remaining gaps to close.
    pub gaps: Vec<&'static str>,
}

/// Aggregate status of all 8 moonshot properties.
#[derive(Debug, Clone)]
pub struct MoonshotStatus {
    /// Status of each of the 8 properties.
    pub properties: [PropertyStatus; 8],
}

#[path = "moonshot_artifacts.rs"]
mod artifacts;
use artifacts::*;

/// All known verification artifacts in the nn workspace mapped to moonshot
/// properties. This is the cross-reference matrix (D3).
pub fn artifact_registry() -> Vec<VerificationArtifact> {
    let mut artifacts = Vec::with_capacity(25);
    artifacts.extend(audio_quality_artifacts());
    artifacts.extend(intelligibility_artifacts());
    artifacts.extend(design_phase_artifacts());
    artifacts.extend(memory_safety_artifacts());
    artifacts.extend(correctness_artifacts());
    artifacts.extend(cross_cutting_artifacts());
    artifacts.extend(full_model_artifacts());
    artifacts
}

/// Property names for each of the 8 moonshot properties.
pub(crate) const PROPERTY_NAMES: [&str; 8] = [
    "Non-silent (RMS > 0.01)",
    "Non-clipping (samples in [-1, 1])",
    "Intelligible (attention monotonic)",
    "Speaker-consistent (embedding distance < ε)",
    "Temporally bounded (< 100ms on M4 Max)",
    "Streaming-safe (bounded chunk discontinuity)",
    "Memory-safe (Kani verified)",
    "Correct implementation (ay/tRust verified)",
];

/// Gaps remaining for each property.
///
/// Updated 2026-03-10: D=192 composed pipeline tests prove P1-P6 at production
/// dimension for synthetic multi-stage pipelines. P3 achieves CrownProven when
/// attention monotonicity certificate is available. Remaining gaps are extending
/// to full Kokoro/production model graphs.
const PROPERTY_GAPS: [&[&str]; 8] = [
    // 0: Non-silent — D=192 CrownProven via composed 3-stage pipeline
    &["Extend D=192 non-silence proof to full Kokoro vocoder graph (not just synthetic 3-stage)"],
    // 1: Non-clipping — D=192 CrownProven via composed 3-stage pipeline
    &["Extend tanh-bounded proof to full Kokoro vocoder graph (not just synthetic 3-stage)"],
    // 2: Intelligible — CrownProven via attention monotonicity certificate (diagonal dominance)
    &[
        "Extend attention monotonicity proof to full Kokoro attention layers (not just synthetic)",
        "Duration positivity for full Kokoro model (not just isolated blocks)",
    ],
    // 3: Speaker-consistent — D=192 CrownProven via 4-stage composed pipeline
    &["Extend D=192 composed speaker proof to full Kokoro+ECAPA-TDNN model graph (not just synthetic 4-stage)"],
    // 4: Temporally bounded — D=192 CrownProven via composed 3-stage pipeline + timing
    &["Extend composed CROWN-coupled timing proof to full Kokoro model (not just synthetic 3-stage)"],
    // 5: Streaming-safe — D=192 CrownProven via composed 3-stage pipeline
    &["Extend D=192 CROWN composition to full Kokoro decoder crossfade graph"],
    // 6: Memory-safe
    &["Coverage audit — some backward rules lack Kani harnesses"],
    // 7: Correct implementation
    &["ay coverage of non-linear ops (only linear kernels currently proven)"],
];

impl MoonshotStatus {
    /// Construct status by scanning the artifact registry.
    pub fn from_repo() -> Self {
        let artifacts = artifact_registry();
        let mut properties = PROPERTY_NAMES.map(|name| PropertyStatus {
            name,
            verified: VerificationLevel::None,
            evidence: Vec::new(),
            gaps: Vec::new(),
        });

        // Map artifacts to properties, tracking best verification level.
        for artifact in &artifacts {
            for &prop_idx in artifact.properties {
                if prop_idx < 8 {
                    let prop = &mut properties[prop_idx];
                    prop.evidence.push(artifact.file);
                    if artifact.level > prop.verified {
                        prop.verified = artifact.level;
                    }
                }
            }
        }

        // Add known gaps.
        for (i, gaps) in PROPERTY_GAPS.iter().enumerate() {
            properties[i].gaps = gaps.to_vec();
        }

        Self { properties }
    }

    /// Check if all 8 properties are at least CrownPartial.
    pub fn all_at_least_crown_partial(&self) -> bool {
        self.properties
            .iter()
            .all(|p| p.verified >= VerificationLevel::CrownPartial)
    }

    /// Check if all 8 properties have any verification evidence.
    pub fn all_have_evidence(&self) -> bool {
        self.properties
            .iter()
            .all(|p| p.verified > VerificationLevel::None)
    }

    /// Count properties at each verification level.
    pub fn level_counts(&self) -> [(VerificationLevel, usize); 7] {
        let levels = [
            VerificationLevel::None,
            VerificationLevel::Empirical,
            VerificationLevel::CrownPartial,
            VerificationLevel::CrownProbabilistic,
            VerificationLevel::CrownProven,
            VerificationLevel::KaniProven,
            VerificationLevel::SmtProven,
        ];
        levels.map(|level| {
            let count = self
                .properties
                .iter()
                .filter(|p| p.verified == level)
                .count();
            (level, count)
        })
    }

    /// Generate a human-readable gap analysis report.
    pub fn report(&self) -> String {
        let mut out = String::with_capacity(2048);
        out.push_str("=== Moonshot Status: First Provably Correct Voice ===\n\n");

        for (i, prop) in self.properties.iter().enumerate() {
            out.push_str(&format!(
                "Property {}: {} [{}]\n",
                i + 1,
                prop.name,
                prop.verified
            ));
            if !prop.evidence.is_empty() {
                out.push_str("  Evidence:\n");
                for e in &prop.evidence {
                    out.push_str(&format!("    - {e}\n"));
                }
            }
            if !prop.gaps.is_empty() {
                out.push_str("  Gaps:\n");
                for g in &prop.gaps {
                    out.push_str(&format!("    - {g}\n"));
                }
            }
            out.push('\n');
        }

        let counts = self.level_counts();
        out.push_str("Summary:\n");
        for (level, count) in &counts {
            if *count > 0 {
                out.push_str(&format!("  {level}: {count} properties\n"));
            }
        }
        out.push_str(&format!(
            "  All at CrownPartial+: {}\n",
            self.all_at_least_crown_partial()
        ));

        out
    }
}

impl fmt::Display for MoonshotStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, prop) in self.properties.iter().enumerate() {
            writeln!(f, "  P{}: [{}] {}", i + 1, prop.verified, prop.name)?;
        }
        Ok(())
    }
}

#[path = "moonshot_certificate.rs"]
mod certificate;
pub use certificate::{
    build_certificate_from_workspace, build_full_certificate,
    build_full_certificate_with_all_evidence, compute_workspace_source_hash, is_valid_sha256_hex,
    validate_certificate, CertificateDeserializeError, CertificateEvidenceSummary,
    CertificateValidation, FindingSeverity, FullCertificateBuilder, KaniVerificationEvidence,
    MoonshotCertificate, PropertyCertificate, SmtVerificationEvidence, SourceHashError,
    ValidationFinding, CERTIFICATE_SCHEMA_VERSION,
};

#[cfg(test)]
#[path = "moonshot_tests.rs"]
mod tests;
