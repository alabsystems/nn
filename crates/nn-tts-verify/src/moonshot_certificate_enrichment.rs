// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Enrichment methods for [`MoonshotCertificate`].
//!
//! These methods add verification evidence from CROWN, timing, speaker,
//! Kani, ay SMT, and dispatch plan analysis to a certificate.
//!
//! Extracted from `moonshot_certificate.rs` (#1940) to keep the parent
//! file under the 500-line limit.

use super::{
    KaniVerificationEvidence, MoonshotCertificate, SmtVerificationEvidence, SubCondition,
    VerificationLevel,
};

impl MoonshotCertificate {
    /// Enrich with CROWN bundle results for properties 1-3, optionally 5 and 6.
    ///
    /// After running CROWN verification via `moonshot_crown::verify_properties_from_pipeline`
    /// or `moonshot_crown::verify_properties_with_timing`, this updates the certificate
    /// with actual bound values and thresholds.
    pub fn with_crown_results(
        mut self,
        bundle: &crate::moonshot_crown::MoonshotCrownBundle,
    ) -> Self {
        self.verification_dim = Some(bundle.verification_dim);

        // Track which property indices have already been written in this
        // iteration. When multiple results share a property_index (e.g.,
        // check_temporal_boundedness and check_memory_boundedness both
        // target index 4), store subsequent results as sub-conditions
        // (#1925) instead of overwriting bound_value/threshold — these
        // quantities may be incomparable (e.g., microseconds vs bytes).
        let mut written: std::collections::HashSet<usize> = std::collections::HashSet::new();

        for result in &bundle.results {
            if result.property_index < self.properties.len() {
                let cert = &mut self.properties[result.property_index];

                if written.contains(&result.property_index) {
                    // Sub-condition: store separately to preserve the
                    // primary result's bound_value/threshold (#1925).
                    // AND proven: take the weaker (lower) level.
                    if !result.proven || result.level < cert.level {
                        cert.level = result.level;
                    }
                    cert.sub_results.push(SubCondition {
                        name: result.property_name.to_string(),
                        bound_value: result.bound_value,
                        threshold: result.threshold,
                        proven: result.proven,
                        explanation: result.explanation.clone(),
                    });
                    cert.assumptions
                        .push(format!("Sub-condition: {}", result.explanation));
                } else {
                    cert.level = result.level;
                    cert.bound_value = Some(result.bound_value);
                    cert.threshold = Some(result.threshold);
                    cert.assumptions = if result.is_sound {
                        vec!["CROWN propagation sound (not IBP fallback)".to_string()]
                    } else {
                        vec!["IBP fallback — bounds may be vacuously wide".to_string()]
                    };
                }
                written.insert(result.property_index);
            }
        }

        self.recompute_aggregate_flags();
        self
    }

    /// Enrich with timing certificate results for Property 5 (temporal boundedness).
    ///
    /// After running timing verification via `verify_pipeline_with_timing`, this
    /// updates the certificate with the worst-case timing bound and threshold.
    /// The timing certificate couples CROWN bounds verification with the roofline
    /// cost model, providing formal evidence for Property 5.
    pub fn with_timing_results(mut self, timing_cert: &crate::pipeline::TimingCertificate) -> Self {
        let result = crate::moonshot_crown::check_temporal_boundedness(timing_cert);
        if result.property_index < self.properties.len() {
            let cert = &mut self.properties[result.property_index];
            cert.level = result.level;
            cert.bound_value = Some(result.bound_value);
            cert.threshold = Some(result.threshold);
            cert.assumptions = if timing_cert.bounds_cert.is_sound {
                vec![
                    "CROWN-coupled timing: bounds and cost from same pipeline".to_string(),
                    format!("Hardware: {}", timing_cert.hardware_name),
                ]
            } else {
                vec![
                    "IBP fallback timing — bounds may be vacuously wide".to_string(),
                    format!("Hardware: {}", timing_cert.hardware_name),
                ]
            };
        }

        self.recompute_aggregate_flags();
        self
    }

    /// Enrich with speaker consistency results for Property 4.
    ///
    /// After running speaker verification via [`check_speaker_consistency`],
    /// this updates the certificate with the worst-case embedding distance
    /// and threshold.
    pub fn with_speaker_results(
        mut self,
        evidence: &crate::moonshot_crown::SpeakerConsistencyEvidence,
    ) -> Self {
        let result = crate::moonshot_crown::check_speaker_consistency(evidence);
        if result.property_index < self.properties.len() {
            let cert = &mut self.properties[result.property_index];
            cert.level = result.level;
            cert.bound_value = Some(result.bound_value);
            cert.threshold = Some(result.threshold);
            cert.assumptions = if evidence.is_sound {
                vec![
                    "CROWN bounds through ECAPA-TDNN speaker encoder".to_string(),
                    format!("Embedding dim: {}", evidence.embed_dim),
                ]
            } else {
                vec![
                    "IBP fallback — speaker embedding bounds may be vacuously wide".to_string(),
                    format!("Embedding dim: {}", evidence.embed_dim),
                ]
            };
        }

        self.recompute_aggregate_flags();
        self
    }

    /// Enrich with Kani verification results for Property 7 (memory safety).
    ///
    /// After scanning Kani harness results (e.g., from `kani_status.json` or
    /// `grep -rc '#[kani::proof]'`), this updates the certificate with the
    /// harness pass count and artifact paths.
    ///
    /// Property 7 achieves `KaniProven` when all harnesses pass.
    pub fn with_kani_results(mut self, evidence: &KaniVerificationEvidence) -> Self {
        const P7_INDEX: usize = 6;
        if P7_INDEX < self.properties.len() {
            let cert = &mut self.properties[P7_INDEX];

            cert.level = if evidence.all_passed && evidence.harnesses_total > 0 {
                VerificationLevel::KaniProven
            } else if evidence.harnesses_passed > 0 {
                VerificationLevel::Empirical
            } else {
                VerificationLevel::None
            };

            cert.proof_artifacts = evidence.harness_files.clone();
            cert.bound_value = Some(evidence.harnesses_passed as f64);
            cert.threshold = Some(evidence.harnesses_total as f64);
            cert.assumptions = if evidence.all_passed {
                vec![
                    format!(
                        "All {} Kani harnesses pass (no UB, no panics in verified paths)",
                        evidence.harnesses_total
                    ),
                    "Kani unwind bounds may limit exploration depth".to_string(),
                ]
            } else {
                vec![format!(
                    "{}/{} Kani harnesses pass — incomplete memory safety coverage",
                    evidence.harnesses_passed, evidence.harnesses_total
                )]
            };
        }

        self.recompute_aggregate_flags();
        self
    }

    /// Enrich with ay SMT verification results for Property 8 (correct
    /// implementation).
    ///
    /// After running `verify_all` or scanning `nn_verify_status.json`,
    /// this updates the certificate with the proven kernel count and names.
    ///
    /// Property 8 achieves `SmtProven` when all kernel proofs reach `Proven`.
    pub fn with_smt_results(mut self, evidence: &SmtVerificationEvidence) -> Self {
        const P8_INDEX: usize = 7;
        if P8_INDEX < self.properties.len() {
            let cert = &mut self.properties[P8_INDEX];

            cert.level = if evidence.all_proven && evidence.kernels_total > 0 {
                VerificationLevel::SmtProven
            } else if evidence.kernels_proven > 0 {
                VerificationLevel::Empirical
            } else {
                VerificationLevel::None
            };

            cert.proof_artifacts = evidence
                .proven_kernel_names
                .iter()
                .map(|name| format!("crates/nn-verify/src/ay/{name}"))
                .collect();
            cert.bound_value = Some(evidence.kernels_proven as f64);
            cert.threshold = Some(evidence.kernels_total as f64);
            cert.assumptions = if evidence.all_proven {
                vec![
                    format!(
                        "All {} kernel proofs reach Proven via ay QF_LRA",
                        evidence.kernels_total
                    ),
                    "SMT quantization margin (1e-4) applied to analytical bounds".to_string(),
                ]
            } else {
                vec![format!(
                    "{}/{} kernel proofs reach Proven — non-linear kernels may reach Unknown",
                    evidence.kernels_proven, evidence.kernels_total
                )]
            };
        }

        self.recompute_aggregate_flags();
        self
    }

    /// Enrich with dispatch plan implementation correctness evidence for
    /// Property 8.
    ///
    /// After analyzing a dispatch plan via [`analyze_dispatch_plan()`],
    /// this updates P8 based on what fraction of numerical operations have
    /// ay-proven kernel correctness proofs. This is complementary to
    /// [`with_smt_results()`] — the SMT path checks the ay proof database,
    /// while this checks the pipeline's specific operation mix.
    ///
    /// If `with_smt_results()` has already set P8 to a higher level, this
    /// method will only upgrade, never downgrade.
    pub fn with_dispatch_plan_correctness(
        mut self,
        evidence: &crate::moonshot_crown::ImplementationCorrectnessEvidence,
    ) -> Self {
        let result = crate::moonshot_crown::check_implementation_correctness(evidence);
        const P8_INDEX: usize = 7;
        if P8_INDEX < self.properties.len() {
            let cert = &mut self.properties[P8_INDEX];
            // Only upgrade, never downgrade
            if result.level > cert.level {
                cert.level = result.level;
                cert.bound_value = Some(result.bound_value);
                cert.threshold = Some(result.threshold);
                cert.assumptions = vec![
                    format!(
                        "{}/{} dispatch ops ay-proven",
                        evidence.proven_steps, evidence.total_steps
                    ),
                    format!("Proven: [{}]", evidence.proven_categories.join(", ")),
                    format!("Gaps: [{}]", evidence.unproven_categories.join(", ")),
                ];
            }
        }

        self.recompute_aggregate_flags();
        self
    }

    /// Attach constructive proof Lean4 exports to CROWN-based properties.
    ///
    /// For properties P1-P4 that use CROWN bounds, this attaches a
    /// self-contained Lean4 module that an auditor can check independently.
    /// The Lean4 source is typically generated by NY's
    /// `compose_crown_proofs()` or `CrownProofExport`.
    ///
    /// `proof_map` maps property index (0-based) to Lean4 source text.
    /// Only CROWN-based properties (P1-P4, indices 0-3) are accepted.
    pub fn with_constructive_proofs(mut self, proof_map: &[(usize, String)]) -> Self {
        for (property_index, lean4_source) in proof_map {
            if *property_index < self.properties.len() {
                self.properties[*property_index].constructive_proof_lean4 =
                    Some(lean4_source.clone());
            }
        }
        self.recompute_aggregate_flags();
        self
    }

    /// Recompute aggregate flags (all_at_least_partial, all_proven,
    /// constructive_proof_count) from the current property levels.
    pub(crate) fn recompute_aggregate_flags(&mut self) {
        self.all_at_least_partial = self
            .properties
            .iter()
            .all(|p| p.level >= VerificationLevel::CrownPartial);
        self.all_proven = self.properties.iter().all(|p| {
            matches!(
                p.level,
                VerificationLevel::CrownProven
                    | VerificationLevel::KaniProven
                    | VerificationLevel::SmtProven
            )
        });
        self.constructive_proof_count = self
            .properties
            .iter()
            .filter(|p| p.constructive_proof_lean4.is_some())
            .count();
    }
}
