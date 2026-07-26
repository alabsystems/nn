// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Model-level certification orchestrator.
//!
//! Single entry point that wires trace → verify → certify → bundle:
//!
//! ```rust,ignore
//! use nn_verify::{certify_model, CertifyConfig};
//! use nn_core::dyn_tensor::trace::trace_graph;
//!
//! let (_output, graph) = trace_graph(|| model.forward(&input))?;
//! let bounds = uniform_bounds(&[1, 3, 8], 1.0);
//! let config = CertifyConfig::new("nn_model");
//! let result = certify_model(&graph, &bounds, &config)?;
//! result.bundle.save(Path::new("model.proof.json"))?;
//! ```
//!
//! Part of #3020 (Proof Certificates), #3030 (VerifiedCompiledModel), #2218.

use ny_api::BoundedTensor;
use nn_core::dyn_tensor::trace::ComputationGraph;
use nn_dsl::verifiability::{classify_op, VerifiabilityClass, VerifiabilitySummary};

use crate::bound_analysis::{analyze_layer_bounds, AnalysisConfig, BoundAnalysisReport};
use crate::certificate::integrity::sign_bundle;
use crate::certificate::{
    certificate_from_pipeline_enriched, CertificateBundle, CertificateEnrichment,
};
use crate::certificate_types::{
    ConstructiveProofData, ConstructiveProofSummary, LayerBoundRecord, PrecisionModel,
};
use crate::error::VerifyError;
use crate::fusion_certificate::FusionEquivalenceCertificate;
use crate::layer_bounds::extract_layer_bounds;
use crate::pipeline::certify_auto_fusion_from_graph;
use crate::signing_config::SigningKey;
use crate::status::ParamInputRecord;
use crate::trace_to_graph::trace_to_graph_model;
use crate::verify::run_escalation;
use crate::verify_types::VerifyConfig;

/// Configuration for model-level certification.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CertifyConfig {
    /// NY verification config.
    pub verify: VerifyConfig,
    /// Model name for the certificate bundle.
    pub model_name: String,
    /// Optional enrichment for source/weight hashes.
    pub enrichment: Option<CertificateEnrichment>,
    /// Epsilon for fusion equivalence verification.
    pub fusion_epsilon: f32,
    /// Production dimension for fusion certificates.
    pub production_dim: usize,
    /// HMAC signing key for certificate integrity. `SigningKey::None` = unsigned.
    pub signing_key: SigningKey,
    /// Whether to generate constructive proof certificates (#4315).
    ///
    /// When `true`, the certify pipeline attempts to generate a constructive
    /// proof certificate from the NY verification results. The
    /// constructive proof contains machine-checkable data (IBP recomputation
    /// data, Lean4 export) that an auditor can verify independently.
    ///
    /// Default: `true`.
    pub generate_constructive_proof: bool,
    /// Optional path to a verification status file (e.g.,
    /// `nn_verify_status_kokoro.json`) for proof strength aggregation (#4315).
    ///
    /// When provided, the certify pipeline reads active (non-stale) status
    /// entries, extracts their `proof_strength` and `method`, and produces a
    /// [`ConstructiveProofSummary`] reporting how many entries are sound,
    /// heuristic, or vacuous, and which verification methods were used.
    ///
    /// Default: `None` (no status-file aggregation).
    pub status_path: Option<std::path::PathBuf>,

    /// Pre-computed transform proof bundle from the compilation pipeline (#4311).
    ///
    /// When provided, the certify pipeline includes these transform proofs in
    /// the `CertifyResult`. This is how the compilation pipeline (peephole
    /// passes) reports equivalence proofs for each transform it applied.
    ///
    /// The typical flow:
    /// 1. Compilation pipeline applies peephole passes (FusedResBlock, style
    ///    absorption, batched style).
    /// 2. Each pass runs its equivalence proof and produces a
    ///    [`TransformProofEntry`].
    /// 3. Entries are collected into a [`TransformProofBundle`] and passed here.
    /// 4. `certify_model()` includes the bundle in `CertifyResult::transform_proofs`.
    ///
    /// Default: `None` (no peephole transforms applied).
    ///
    /// [`TransformProofEntry`]: crate::certificate_types::TransformProofEntry
    /// [`TransformProofBundle`]: crate::certificate_types::TransformProofBundle
    pub transform_proofs: Option<crate::certificate_types::TransformProofBundle>,
}

impl CertifyConfig {
    /// Create a config with sensible defaults.
    #[must_use]
    pub fn new(model_name: &str) -> Self {
        Self {
            verify: VerifyConfig::default(),
            model_name: model_name.to_string(),
            enrichment: None,
            fusion_epsilon: 1e-5,
            production_dim: 256,
            signing_key: SigningKey::None,
            generate_constructive_proof: true,
            status_path: None,
            transform_proofs: None,
        }
    }

    /// Set the status file path for proof strength aggregation.
    #[must_use]
    pub fn with_status_path(mut self, path: std::path::PathBuf) -> Self {
        self.status_path = Some(path);
        self
    }

    /// Attach a pre-computed `TransformProofBundle` from the compilation pipeline.
    ///
    /// When a model is compiled through the peephole pipeline, each transform
    /// generates an equivalence proof. This method wires the collection of
    /// proofs into the certification config so they appear in the final
    /// `CertifyResult::transform_proofs`.
    #[must_use]
    pub fn with_transform_proofs(
        mut self,
        bundle: crate::certificate_types::TransformProofBundle,
    ) -> Self {
        self.transform_proofs = Some(bundle);
        self
    }
}

/// Result of model-level certification.
#[derive(Debug)]
#[non_exhaustive]
pub struct CertifyResult {
    /// Certificate bundle containing proof certificates.
    pub bundle: CertificateBundle,
    /// Fusion equivalence certificates (separate from bounds certificates).
    pub fusion_certificates: Vec<FusionEquivalenceCertificate>,
    /// Output bounds from NY IBP/CROWN.
    pub output_bounds: BoundedTensor,
    /// Bound analysis report (progressive tightening diagnostics).
    pub bound_analysis: Option<BoundAnalysisReport>,
    /// Verifiability summary for the computation graph.
    pub verifiability: VerifiabilitySummary,
    /// Diagnostic: layer bounds extraction failure reason (None = success).
    /// When present, the certificate lacks trace consistency data (#3200 F3).
    pub layer_bounds_warning: Option<String>,
    /// Diagnostic: fusion certification failure reason (None = success or no fusions).
    /// When present, fusion_certificates will be empty (#3200 F2).
    pub fusion_warning: Option<String>,
    /// Constructive proof data from NY (#4315).
    /// `None` when constructive proof generation was disabled or failed.
    pub constructive_proof: Option<ConstructiveProofData>,
    /// Transform proof bundle from the certifying compiler (#4311).
    /// Contains per-transform equivalence proofs for all peephole passes.
    /// `None` when the model was not compiled through the peephole pipeline.
    pub transform_proofs: Option<crate::certificate_types::TransformProofBundle>,
    /// Aggregated proof strength summary from status file entries (#4315).
    ///
    /// When a `status_path` is provided in [`CertifyConfig`], this contains
    /// per-entry proof strength and method aggregation. Reports how many
    /// entries are constructively proved (sound) versus heuristic or vacuous,
    /// and what verification methods were used across the model's pipeline.
    ///
    /// `None` when no status path was configured or the status file could not
    /// be loaded.
    pub proof_summary: Option<ConstructiveProofSummary>,
}

impl CertifyResult {
    /// Whether a constructive proof certificate was generated.
    #[must_use]
    pub fn has_constructive_proof(&self) -> bool {
        self.constructive_proof.is_some()
    }

    /// Access the constructive proof data, if present.
    #[must_use]
    pub fn constructive_proof(&self) -> Option<&ConstructiveProofData> {
        self.constructive_proof.as_ref()
    }

    /// Serialize the constructive proof certificate to JSON for deployment.
    ///
    /// Returns `None` if no constructive proof was generated. Otherwise
    /// returns the JSON string that can be written to a file and later
    /// loaded by [`ConstructiveProofData::from_json`] for independent
    /// auditor verification.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError::Serialization` if JSON serialization fails.
    pub fn constructive_proof_json(&self) -> Result<Option<String>, VerifyError> {
        match &self.constructive_proof {
            Some(proof) => Ok(Some(proof.to_json()?)),
            None => Ok(None),
        }
    }

    /// Save the constructive proof certificate to a standalone JSON file.
    ///
    /// This writes the constructive proof as a separate artifact alongside
    /// the model binary. An auditor can load it with
    /// [`ConstructiveProofData::load`] for independent verification without
    /// needing the full `CertificateBundle`.
    ///
    /// Returns `Ok(true)` if a constructive proof was saved, `Ok(false)` if
    /// no constructive proof was generated (nothing to save).
    ///
    /// # Errors
    ///
    /// Returns `VerifyError::Serialization` or `VerifyError::Io` on failure.
    pub fn save_constructive_proof(&self, path: &std::path::Path) -> Result<bool, VerifyError> {
        match &self.constructive_proof {
            Some(proof) => {
                proof.save(path)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Validate the constructive proof certificate's structural consistency.
    ///
    /// Runs structural validation on the constructive proof (if present),
    /// checking that bounds are finite, non-inverted, and dimensionally
    /// consistent. Also validates the certificate bundle.
    ///
    /// Returns `Ok(true)` if a constructive proof is present and valid,
    /// `Ok(false)` if no constructive proof was generated, or
    /// `Err` if validation fails (structurally invalid proof or bundle).
    pub fn validate_constructive_proof(&self) -> Result<bool, String> {
        // Validate bundle certificates.
        self.bundle
            .validate_all()
            .map_err(|(idx, e)| format!("bundle certificate[{idx}] validation failed: {e}"))?;

        match &self.constructive_proof {
            Some(proof) => {
                proof.validate()?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Replay-verify the constructive proof certificate.
    ///
    /// Performs structural validation and checks bound-chain containment
    /// for per-layer proofs. Returns `true` if the proof passes replay
    /// verification, `false` if no proof is present or verification fails.
    ///
    /// This does NOT re-run NY; it checks the proof's internal
    /// consistency. For full independent verification, use the Lean4 export.
    #[must_use]
    pub fn replay_verify_constructive_proof(&self) -> bool {
        match &self.constructive_proof {
            Some(proof) => proof.replay_verify(),
            None => false,
        }
    }

    /// Whether a proof strength summary was aggregated from status entries.
    #[must_use]
    pub fn has_proof_summary(&self) -> bool {
        self.proof_summary.is_some()
    }

    /// Access the aggregated proof strength summary, if present.
    #[must_use]
    pub fn proof_summary(&self) -> Option<&ConstructiveProofSummary> {
        self.proof_summary.as_ref()
    }

    /// Whether both constructive proof and proof summary indicate
    /// deployment-ready quality.
    ///
    /// Returns `true` when the constructive proof is machine-checkable AND
    /// the proof summary (if present) shows all entries are sound with no
    /// vacuous bounds. When no summary is present, only the constructive
    /// proof is checked.
    #[must_use]
    pub fn is_deployment_certifiable(&self) -> bool {
        let proof_ok = self
            .constructive_proof
            .as_ref()
            .map_or(false, ConstructiveProofData::is_machine_checkable);
        let summary_ok = self
            .proof_summary
            .as_ref()
            .map_or(true, ConstructiveProofSummary::is_deployment_ready);
        proof_ok && summary_ok
    }
}

/// Error type for the certification orchestrator.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CertifyError {
    /// Model contains ops that cannot be verified by NY.
    #[error("model contains unverifiable ops: {}", ops.join(", "))]
    UnverifiableOps {
        /// Names of ops classified as `UnverifiableLearned`.
        ops: Vec<String>,
    },

    /// Verification or translation error from NY.
    #[error(transparent)]
    Verify(#[from] VerifyError),
}

/// Certify a traced model, producing a `CertificateBundle`.
///
/// Pipeline: verifiability check → graph translation → IBP/CROWN →
/// layer bounds → certificate → fusion certification → bundle.
///
/// # Errors
///
/// Returns [`CertifyError::UnverifiableOps`] if any op is classified as
/// `UnverifiableLearned` (with the list of op names for diagnostics).
/// Returns [`CertifyError::Verify`] for translation or propagation failures.
pub fn certify_model(
    graph: &ComputationGraph,
    input_bounds: &BoundedTensor,
    config: &CertifyConfig,
) -> Result<CertifyResult, CertifyError> {
    // 1. Verifiability pre-check
    let summary = classify_graph(graph);
    if !summary.is_fully_compilable() {
        return Err(CertifyError::UnverifiableOps {
            ops: summary.unverifiable_learned_ops,
        });
    }

    // 2. Translate trace → NY GraphNetwork
    let translate_result = trace_to_graph_model(graph)?;
    let network = translate_result.graph;

    // 3. IBP/CROWN escalation
    let (verification, output_bounds) = run_escalation(
        &network,
        input_bounds,
        &config.model_name,
        &config.verify,
        false,
    )?;

    // 4. Extract per-layer bounds for certificate enrichment
    let (layer_bounds, layer_bounds_warning) = match extract_layer_bounds(&network, input_bounds) {
        Ok(lb) => (Some(lb), None),
        Err(e) => (None, Some(format!("layer bounds extraction failed: {e}"))),
    };

    // 5. Generate proof certificate
    //    Scalar summary: min(lower) / max(upper) across all input elements.
    let (lower, upper) = input_bounds.lower_upper();
    let lo_min = lower.iter().copied().fold(f32::INFINITY, f32::min);
    let hi_max = upper.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let variable_inputs = vec![ParamInputRecord {
        param_index: 0,
        lower: lo_min,
        upper: hi_max,
    }];
    let mut cert = certificate_from_pipeline_enriched(
        &verification,
        &variable_inputs,
        &[],
        None, // SMT outcome — not run at model level
        config.enrichment.as_ref(),
    );
    if let Some(lb) = &layer_bounds {
        cert = cert.with_layer_bounds(lb.clone());
    }

    // 5b. Auto-populate PrecisionModel from dtype cast count (#3023).
    let precision = if translate_result.dtype_cast_count > 0 {
        PrecisionModel::F16Aware {
            cast_count: translate_result.dtype_cast_count,
            total_epsilon: 0.0, // Phase 2: ULP widening
        }
    } else {
        PrecisionModel::F32Only
    };
    cert = cert.with_precision_model(precision);

    // 5c. Generate constructive proof certificate (#4315).
    let constructive_proof = if config.generate_constructive_proof {
        generate_constructive_proof(
            &network,
            input_bounds,
            &output_bounds,
            layer_bounds.as_deref(),
            verification.method,
        )
    } else {
        None
    };
    if let Some(ref proof) = constructive_proof {
        cert = cert.with_constructive_proof(proof.clone());
    }

    // 6. Self-validate certificate before bundling — catch NaN/Inf/inverted
    //    bounds that would produce an invalid certificate for the caller.
    if let Err(e) = cert.validate() {
        return Err(CertifyError::Verify(VerifyError::InvalidCertificate {
            reason: format!("generated certificate failed self-validation: {e}"),
        }));
    }

    // 7. Assemble bundle
    let mut bundle = CertificateBundle::new(&config.model_name).with_certificate(cert);

    // 8. Sign bundle if key is configured
    if let Some(key) = config.signing_key.as_bytes() {
        sign_bundle(&mut bundle, key).map_err(|e| {
            CertifyError::Verify(VerifyError::InvalidCertificate {
                reason: format!("certificate signing failed: {e}"),
            })
        })?;
    }

    // 9. Fusion certification (optional — failure recorded, not fatal)
    //    Use IBP/CROWN-derived bounds when available instead of hardcoded (-3, 3).
    let fusion_variable_bounds = derive_fusion_bounds(&layer_bounds, input_bounds);
    let (fusion_certificates, fusion_warning) = match certify_auto_fusion_from_graph(
        graph,
        &fusion_variable_bounds,
        config.fusion_epsilon,
        config.production_dim,
    ) {
        Ok(result) => (result.certificates, None),
        Err(e) => (
            Vec::new(),
            Some(format!("fusion certification failed: {e}")),
        ),
    };

    // 10. Bound analysis
    let bound_analysis = layer_bounds
        .as_ref()
        .map(|lb| analyze_layer_bounds(&config.model_name, lb, &AnalysisConfig::default()));

    // 11. Proof strength aggregation from status file (#4315).
    //     Non-fatal: if the status file is missing or unreadable, we proceed
    //     without the summary. The constructive proof from the live run is the
    //     primary artifact; the summary is supplementary deployment metadata.
    let proof_summary = config.status_path.as_ref().and_then(|path| {
        crate::status::VerifyStatus::load(path)
            .ok()
            .and_then(|status| aggregate_proof_summary(&status))
    });

    Ok(CertifyResult {
        bundle,
        fusion_certificates,
        output_bounds,
        bound_analysis,
        verifiability: summary,
        layer_bounds_warning,
        fusion_warning,
        constructive_proof,
        transform_proofs: config.transform_proofs.clone(),
        proof_summary,
    })
}

// Constructive proof generation and proof strength aggregation extracted
// to certify_constructive.rs (500-line limit, #4315).
#[path = "certify_constructive.rs"]
mod certify_constructive;
use certify_constructive::{aggregate_proof_summary, generate_constructive_proof};
pub use certify_constructive::{
    verify_and_certify, ProofStrengthClassification, VerifyAndCertifyResult,
};

/// Classify all ops in a computation graph, building a `VerifiabilitySummary`.
fn classify_graph(graph: &ComputationGraph) -> VerifiabilitySummary {
    let mut summary = VerifiabilitySummary::default();
    for node in graph.nodes() {
        let class = classify_op(node.op());
        match class {
            VerifiabilityClass::Verifiable => summary.verifiable += 1,
            VerifiabilityClass::VerifiableBounded { .. } => summary.bounded += 1,
            VerifiabilityClass::ShapeOnly => summary.shape_only += 1,
            VerifiabilityClass::Passthrough => summary.passthrough += 1,
            VerifiabilityClass::UnverifiableSafe => summary.unverifiable_safe += 1,
            VerifiabilityClass::UnverifiableLearned => {
                summary.unverifiable_learned += 1;
                summary
                    .unverifiable_learned_ops
                    .push(node.op().canonical_name().to_string());
            }
            // Future variants — conservative: count as unverifiable safe
            _ => summary.unverifiable_safe += 1,
        }
    }
    summary.unverifiable_learned_ops.sort();
    summary.unverifiable_learned_ops.dedup();
    summary
}

/// Derive per-variable activation bounds for fusion verification.
///
/// Uses IBP/CROWN-computed layer bounds when available, falls back to
/// input bounds scalar range when layer bounds are absent.
pub(crate) fn derive_fusion_bounds(
    layer_bounds: &Option<Vec<LayerBoundRecord>>,
    input_bounds: &BoundedTensor,
) -> Vec<(f32, f32)> {
    if let Some(lb) = layer_bounds {
        if let Some((lo, hi)) = tightest_enclosing_interval(lb) {
            return vec![(lo, hi)];
        }
    }
    // Fallback: use input bounds range (wider than necessary but sound).
    let (lower, upper) = input_bounds.lower_upper();
    let lo = lower.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = upper.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    vec![(lo.min(-3.0), hi.max(3.0))]
}

/// Find the tightest interval enclosing all layer output bounds.
pub(crate) fn tightest_enclosing_interval(layer_bounds: &[LayerBoundRecord]) -> Option<(f32, f32)> {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    let mut found = false;
    for lb in layer_bounds {
        for &(l, u) in &lb.output_bounds {
            if l.is_finite() && u.is_finite() {
                lo = lo.min(l);
                hi = hi.max(u);
                found = true;
            }
        }
    }
    if found {
        Some((lo, hi))
    } else {
        None
    }
}

#[cfg(kani)]
#[path = "certify_kani.rs"]
mod kani_proofs;

#[cfg(test)]
#[path = "certify_tests_inline.rs"]
mod tests;

#[cfg(test)]
#[path = "certify_tests.rs"]
mod tests_coverage;

#[cfg(test)]
#[path = "certify_certificate_tests.rs"]
mod certify_certificate_tests;
