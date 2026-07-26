// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! TTS verification certificate — aggregate result of all checks.

use crate::bounds::HardBound;
use crate::crown_junction::JunctionCheckSummary;
use crate::moonshot::MoonshotCertificate;
use crate::phoneme::PhonemeResult;
use crate::quality::QualityMetric;

#[cfg(feature = "ny")]
use nn_verify::DeadNeuronEliminationProof;

/// Aggregate certificate from TTS quality verification.
///
/// Contains results from all hard bound checks and quality metrics.
/// `overall_passed` is true only if all hard bounds pass AND all
/// quality metrics (if present) pass.
///
/// When CROWN verification is enabled (via `with_crown_verification(true)`
/// on [`CompiledKokoro`]), `crown_evidence` contains a [`MoonshotCertificate`]
/// with formal property verification results derived from the synthesis output.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Certificate {
    /// Results of hard bound checks.
    pub hard_bounds: Vec<HardBound>,
    /// Results of quality metric evaluations (empty if no reference provided).
    pub quality_metrics: Vec<QualityMetric>,
    /// Per-phoneme verification results (None if no alignment provided).
    pub phoneme_results: Option<Vec<PhonemeResult>>,
    /// True if all checks passed.
    pub overall_passed: bool,
    /// Optional deterministic synthesis hash for zero-tolerance regression testing.
    pub deterministic_hash: Option<String>,
    /// Optional CROWN-based moonshot property verification evidence.
    ///
    /// When present, contains formal verification results for up to 8 moonshot
    /// properties (non-silence, non-clipping, intelligibility, speaker consistency,
    /// temporal boundedness, streaming safety, memory safety, implementation
    /// correctness). Populated when CROWN verification is enabled on the
    /// synthesis pipeline.
    pub crown_evidence: Option<MoonshotCertificate>,
    /// Optional junction contract check results (J2-J5).
    ///
    /// When present, contains per-junction pass/fail for the Kokoro pipeline
    /// zone crossings. Populated when CROWN verification is enabled with
    /// `check_junction_contracts = true` in [`CrownCertificateConfig`].
    ///
    /// Part of #4254.
    pub junction_summary: Option<JunctionCheckSummary>,
    /// Optional dead-neuron elimination equivalence proof (upstream
    /// NY commit `1ed64542f`; integration per design
    /// `designs/2026-04-19-NY-f57811-adoption.md` §3).
    ///
    /// When present, certifies that a dead-neuron-optimized sub-network of
    /// the TTS pipeline is formally equivalent to its original form within a
    /// CROWN-verified epsilon bound. This upgrades the moonshot bundle's
    /// claim from "bounds propagated" to
    /// "bounds propagated + dead-neuron equivalence certified".
    ///
    /// Populated by attaching the output of
    /// `nn_verify::run_dead_neuron_elimination` via
    /// [`Certificate::with_dead_neuron_eq_proof`].
    ///
    /// Part of #3874.
    #[cfg(feature = "ny")]
    pub dead_neuron_eq_proof: Option<DeadNeuronEliminationProof>,
}

impl Certificate {
    /// Whether all hard bounds passed.
    #[must_use]
    pub fn passes_hard_bounds(&self) -> bool {
        self.hard_bounds.iter().all(|b| b.passed)
    }

    /// Whether all quality metrics passed.
    ///
    /// Returns `true` if no quality metrics were evaluated (vacuously true).
    #[must_use]
    pub fn passes_quality(&self) -> bool {
        self.quality_metrics.iter().all(|m| m.passed)
    }

    /// Generate a human-readable verification report.
    #[must_use]
    pub fn report(&self) -> String {
        let mut report = String::with_capacity(512);
        report.push_str("=== TTS Verification Certificate ===\n\n");

        report.push_str("-- Hard Bounds --\n");
        for b in &self.hard_bounds {
            let status = if b.passed { "PASS" } else { "FAIL" };
            report.push_str(&format!(
                "  [{status}] {name}: value={value:.4}, threshold={threshold:.4}\n",
                name = b.name,
                value = b.value,
                threshold = b.threshold,
            ));
        }

        if !self.quality_metrics.is_empty() {
            report.push_str("\n-- Quality Metrics --\n");
            for m in &self.quality_metrics {
                let status = if m.passed { "PASS" } else { "FAIL" };
                report.push_str(&format!(
                    "  [{status}] {name}: value={value:.4}, threshold={threshold:.4} ({citation})\n",
                    name = m.name,
                    value = m.value,
                    threshold = m.threshold,
                    citation = m.citation,
                ));
            }
        }

        if let Some(ref phonemes) = self.phoneme_results {
            report.push_str("\n-- Per-Phoneme Verification --\n");
            let total = phonemes.len();
            let passed = phonemes.iter().filter(|p| p.passed).count();
            report.push_str(&format!("  {passed}/{total} phonemes passed\n"));
            for p in phonemes {
                let status = if p.passed { "PASS" } else { "FAIL" };
                report.push_str(&format!(
                    "  [{status}] /{}/: {:.1}ms\n",
                    p.label, p.duration_ms,
                ));
            }
        }

        if let Some(ref hash) = self.deterministic_hash {
            report.push_str(&format!("\n-- Deterministic Hash --\n  SHA-256: {hash}\n"));
        }

        if let Some(ref crown) = self.crown_evidence {
            report.push_str("\n-- CROWN Verification Evidence --\n");
            for prop in &crown.properties {
                let status = if prop.level >= crate::moonshot::VerificationLevel::CrownPartial {
                    "PROVEN"
                } else {
                    "EMPIRICAL"
                };
                report.push_str(&format!(
                    "  [{status}] P{}: {} [{}]\n",
                    prop.property_index + 1,
                    prop.property_name,
                    prop.level,
                ));
            }
            report.push_str(&format!(
                "  All at least CrownPartial: {}\n",
                crown.all_at_least_partial
            ));
            report.push_str(&format!("  All proven: {}\n", crown.all_proven));
        }

        if let Some(ref junctions) = self.junction_summary {
            report.push_str("\n-- Junction Contract Checks --\n");
            report.push_str(&format!(
                "  {}/{} contracts passed\n",
                junctions.total_passed,
                junctions.total_passed + junctions.total_failed,
            ));
            for check in &junctions.checks {
                report.push_str(&format!("  {check}\n"));
            }
        }

        report.push_str(&format!(
            "\nOverall: {}\n",
            if self.overall_passed {
                "PASSED"
            } else {
                "FAILED"
            }
        ));

        report
    }

    /// Attach CROWN verification evidence to this certificate.
    ///
    /// Returns a new certificate with the `crown_evidence` field populated.
    /// This is the primary way to enrich a runtime Certificate with formal
    /// verification results from the moonshot certification system.
    #[must_use]
    pub fn with_crown_evidence(mut self, evidence: MoonshotCertificate) -> Self {
        self.crown_evidence = Some(evidence);
        self
    }

    /// Returns true if CROWN verification evidence is attached.
    #[must_use]
    pub fn has_crown_evidence(&self) -> bool {
        self.crown_evidence.is_some()
    }

    /// Attach junction contract check results to this certificate.
    ///
    /// Returns a new certificate with the `junction_summary` field populated.
    /// Junction contracts (J2-J5) validate intermediate tensor bounds at
    /// Kokoro pipeline zone crossings.
    ///
    /// Part of #4254.
    #[must_use]
    pub fn with_junction_summary(mut self, summary: JunctionCheckSummary) -> Self {
        self.junction_summary = Some(summary);
        self
    }

    /// Returns true if junction contract results are attached.
    #[must_use]
    pub fn has_junction_summary(&self) -> bool {
        self.junction_summary.is_some()
    }

    /// Returns true if all junction contracts passed (or no contracts were checked).
    #[must_use]
    pub fn passes_junction_contracts(&self) -> bool {
        self.junction_summary
            .as_ref()
            .map_or(true, |s| s.total_failed == 0)
    }

    /// Attach a dead-neuron elimination equivalence proof to this certificate.
    ///
    /// Returns a new certificate with the `dead_neuron_eq_proof` field
    /// populated. The proof is produced by
    /// `nn_verify::run_dead_neuron_elimination` (upstream
    /// `eliminate_and_verify`, commit `1ed64542f`).
    ///
    /// Part of #3874.
    #[cfg(feature = "ny")]
    #[must_use]
    pub fn with_dead_neuron_eq_proof(mut self, proof: DeadNeuronEliminationProof) -> Self {
        self.dead_neuron_eq_proof = Some(proof);
        self
    }

    /// Returns true if a dead-neuron elimination equivalence proof is
    /// attached.
    #[cfg(feature = "ny")]
    #[must_use]
    pub fn has_dead_neuron_eq_proof(&self) -> bool {
        self.dead_neuron_eq_proof.is_some()
    }

    /// Returns true if an attached dead-neuron elimination proof attests
    /// deployment-safe equivalence (or if no proof is attached, which is
    /// vacuously true).
    #[cfg(feature = "ny")]
    #[must_use]
    pub fn passes_dead_neuron_equivalence(&self) -> bool {
        self.dead_neuron_eq_proof
            .as_ref()
            .map_or(true, DeadNeuronEliminationProof::is_deployment_safe)
    }
}
