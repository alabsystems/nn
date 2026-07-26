// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CROWN certificate deployment wiring for Kokoro TTS.
//!
//! Bridges the per-entry verification status (`nn_verify_status_kokoro.json`)
//! with pipeline stage coverage from the gap detector to produce a
//! deployment-ready [`KokoroCrownCertificate`].
//!
//! The deployment certificate answers one question: **is this Kokoro build
//! certifiably correct for production deployment?** It combines:
//!
//! - Per-stage CROWN coverage mapping (which pipeline stages have CROWN bounds)
//! - Soundness gate (minimum sound entry threshold for deployment)
//! - Per-entry constructive proofs (machine-checkable bound data from #4254)
//! - Gap detection integration (no uncovered pipeline stages)
//!
//! # Usage
//!
//! ```rust,ignore
//! use nn_verify::kokoro_crown_certificate::{
//!     generate_deployment_certificate, DeploymentConfig, DeploymentGate,
//! };
//!
//! let config = DeploymentConfig::new("sha256_hash", status_path);
//! let cert = generate_deployment_certificate(&config)?;
//! assert!(cert.gate.is_deployable());
//! cert.save(Path::new("kokoro_deployment.proof.json"))?;
//! ```
//!
//! Part of #4254.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::certificate_types::{
    compute_bytes_hash, ConstructiveProofData, ConstructiveProofMethod,
};
use crate::error::VerifyError;
use crate::gap_detector::kokoro_pipeline_stages;
use crate::kokoro_certificate::{default_gamma_crown_rev, generate_from_status, KokoroCertificate};
use crate::status::{KernelStatus, VerifyStatus};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for Kokoro deployment certificate generation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DeploymentConfig {
    /// SHA-256 hash of the Kokoro model weights.
    pub model_hash: String,
    /// Path to `nn_verify_status_kokoro.json`.
    pub status_path: std::path::PathBuf,
    /// NY git revision string.
    pub gamma_crown_rev: String,
    /// Minimum fraction of active entries that must be sound (0.0..=1.0).
    /// Default: 0.90 (90%).
    pub min_sound_ratio: f64,
    /// Minimum number of pipeline stages with CROWN coverage.
    /// Default: 3.
    pub min_crown_stages: usize,
    /// Maximum allowed number of vacuous entries. Default: 0.
    pub max_vacuous: usize,
    /// Maximum allowed uncovered pipeline stages (gaps). Default: 0.
    pub max_gaps: usize,
}

impl DeploymentConfig {
    /// Create a config with production defaults.
    #[must_use]
    pub fn new(model_hash: &str, status_path: &Path) -> Self {
        Self {
            model_hash: model_hash.to_string(),
            status_path: status_path.to_path_buf(),
            gamma_crown_rev: default_gamma_crown_rev().to_string(),
            min_sound_ratio: 0.90,
            min_crown_stages: 3,
            max_vacuous: 0,
            max_gaps: 0,
        }
    }

    /// Override the NY revision.
    #[must_use]
    pub fn with_gamma_crown_rev(mut self, rev: &str) -> Self {
        self.gamma_crown_rev = rev.to_string();
        self
    }

    /// Override the minimum sound ratio threshold.
    #[must_use]
    pub fn with_min_sound_ratio(mut self, ratio: f64) -> Self {
        self.min_sound_ratio = ratio;
        self
    }
}

// ---------------------------------------------------------------------------
// Per-stage CROWN coverage
// ---------------------------------------------------------------------------

/// CROWN coverage status for a single Kokoro pipeline stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageCrownCoverage {
    /// Pipeline stage name (e.g., "PlBert + bert_encoder").
    pub stage_name: String,
    /// Status key used to look up the entry in the verification status file.
    pub status_key: String,
    /// Whether a primary (IBP or any method) entry exists for this stage.
    pub has_primary: bool,
    /// Whether a CROWN-family entry exists (CROWN, AlphaCrown, BetaCrown).
    pub has_crown: bool,
    /// The method used for the primary entry, if present.
    pub primary_method: Option<String>,
    /// Output width from the primary entry, if present.
    pub output_width: Option<f32>,
    /// Soundness mode of the primary entry.
    pub soundness: Option<String>,
    /// Whether this stage is a compiled GPU segment.
    pub is_compiled_segment: bool,
}

/// Map pipeline stages to their CROWN coverage using the verification status.
fn compute_stage_coverage(status: &VerifyStatus) -> Vec<StageCrownCoverage> {
    let stages = kokoro_pipeline_stages();
    let kernels = status.kernels();

    stages
        .into_iter()
        .map(|stage| {
            let primary = kernels.get(stage.status_key);
            let crown_key = format!("{}_crown", stage.status_key);
            let crown_entry = kernels.get(&crown_key);

            let has_primary = primary.map_or(false, |ks| {
                matches!(
                    ks.status,
                    crate::status::VerifyOutcome::Verified
                        | crate::status::VerifyOutcome::BoundsComputed
                        | crate::status::VerifyOutcome::IbpFallback
                )
            });

            let primary_method_is_crown = primary.map_or(false, |ks| {
                matches!(
                    ks.method,
                    crate::verify_types::PropMethod::Crown
                        | crate::verify_types::PropMethod::AlphaCrown
                        | crate::verify_types::PropMethod::BetaCrown
                        | crate::verify_types::PropMethod::MixedIbpCrown
                )
            });

            let has_crown = crown_entry.is_some() || primary_method_is_crown;

            let primary_method = primary.map(|ks| format!("{:?}", ks.method));
            let output_width = primary.map(|ks| ks.output_width);
            let soundness = primary.map(|ks| {
                serde_json::to_value(ks.soundness_mode)
                    .ok()
                    .and_then(|v| v.as_str().map(String::from))
                    .unwrap_or_else(|| format!("{:?}", ks.soundness_mode).to_lowercase())
            });

            StageCrownCoverage {
                stage_name: stage.name.to_string(),
                status_key: stage.status_key.to_string(),
                has_primary,
                has_crown,
                primary_method,
                output_width,
                soundness,
                is_compiled_segment: stage.is_compiled_segment,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Deployment gate
// ---------------------------------------------------------------------------

/// Deployment gate: pass/fail evaluation for certified deployment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DeploymentGate {
    /// Whether all gate criteria are met.
    pub deployable: bool,
    /// Sound entry ratio (sound / active).
    pub sound_ratio: f64,
    /// Whether the sound ratio meets the threshold.
    pub sound_ratio_pass: bool,
    /// Number of pipeline stages with CROWN coverage.
    pub crown_stage_count: usize,
    /// Whether the CROWN stage count meets the threshold.
    pub crown_stage_pass: bool,
    /// Number of vacuous entries.
    pub vacuous_count: usize,
    /// Whether vacuous count is within the threshold.
    pub vacuous_pass: bool,
    /// Number of uncovered pipeline stages (gaps).
    pub gap_count: usize,
    /// Whether gap count is within the threshold.
    pub gap_pass: bool,
    /// Per-check detail messages for diagnostics.
    pub details: Vec<String>,
}

impl DeploymentGate {
    /// Whether all deployment criteria are met.
    #[must_use]
    pub fn is_deployable(&self) -> bool {
        self.deployable
    }
}

/// Evaluate the deployment gate from status data and coverage.
fn evaluate_gate(
    status: &VerifyStatus,
    stage_coverage: &[StageCrownCoverage],
    config: &DeploymentConfig,
) -> DeploymentGate {
    let kernels = status.kernels();
    let active_entries: Vec<_> = kernels.values().filter(|ks| !ks.stale).collect();
    let active_count = active_entries.len();

    let sound_count = active_entries
        .iter()
        .filter(|ks| {
            matches!(
                ks.soundness_mode,
                crate::soundness_compat::VerificationSoundnessMode::Sound
            )
        })
        .count();

    let sound_ratio = if active_count > 0 {
        sound_count as f64 / active_count as f64
    } else {
        0.0
    };
    let sound_ratio_pass = sound_ratio >= config.min_sound_ratio;

    let crown_stage_count = stage_coverage.iter().filter(|s| s.has_crown).count();
    let crown_stage_pass = crown_stage_count >= config.min_crown_stages;

    let vacuous_count = active_entries
        .iter()
        .filter(|ks| {
            ks.proof_strength
                .map_or(false, |ps| ps == crate::status::ProofStrength::Vacuous)
        })
        .count();
    let vacuous_pass = vacuous_count <= config.max_vacuous;

    let gap_count = stage_coverage.iter().filter(|s| !s.has_primary).count();
    let gap_pass = gap_count <= config.max_gaps;

    let deployable = sound_ratio_pass && crown_stage_pass && vacuous_pass && gap_pass;

    let mut details = Vec::new();
    details.push(format!(
        "sound_ratio: {sound_count}/{active_count} = {sound_ratio:.2} (threshold: {:.2}) [{}]",
        config.min_sound_ratio,
        if sound_ratio_pass { "PASS" } else { "FAIL" }
    ));
    details.push(format!(
        "crown_stages: {crown_stage_count} (min: {}) [{}]",
        config.min_crown_stages,
        if crown_stage_pass { "PASS" } else { "FAIL" }
    ));
    details.push(format!(
        "vacuous: {vacuous_count} (max: {}) [{}]",
        config.max_vacuous,
        if vacuous_pass { "PASS" } else { "FAIL" }
    ));
    details.push(format!(
        "gaps: {gap_count} (max: {}) [{}]",
        config.max_gaps,
        if gap_pass { "PASS" } else { "FAIL" }
    ));

    DeploymentGate {
        deployable,
        sound_ratio,
        sound_ratio_pass,
        crown_stage_count,
        crown_stage_pass,
        vacuous_count,
        vacuous_pass,
        gap_count,
        gap_pass,
        details,
    }
}

// ---------------------------------------------------------------------------
// Per-entry constructive proof extraction (#4254)
// ---------------------------------------------------------------------------

/// A constructive proof derived from a single Kokoro status entry.
///
/// Bridges the gap between the per-entry verification status
/// (`nn_verify_status_kokoro.json`) and machine-checkable proof
/// artifacts. Each `EntryConstructiveProof` contains the input/output
/// bounds and verification method from one status entry, structured
/// as a [`ConstructiveProofData`] that auditors can independently verify.
///
/// Part of #4254.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntryConstructiveProof {
    /// Kernel entry name from the status file.
    pub kernel_name: String,
    /// The constructive proof data derived from this entry's bounds.
    pub proof: ConstructiveProofData,
    /// Whether this entry uses a CROWN-family method (tighter bounds).
    pub is_crown: bool,
    /// Whether this entry's soundness mode is Sound.
    pub is_sound: bool,
}

/// Extract a constructive proof from a single [`KernelStatus`] entry.
///
/// Uses the entry's input/output bounds (including per-element tensor
/// bounds when available) to produce a [`ConstructiveProofData`] that
/// can be independently verified.
///
/// Returns `None` if the entry has non-finite bounds or is infeasible.
fn extract_entry_proof(name: &str, ks: &KernelStatus) -> Option<EntryConstructiveProof> {
    // Skip infeasible entries — their bounds are sentinels.
    if ks.output_bounds.is_infeasible {
        return None;
    }

    // Determine input bounds: use per-variable tensor data when available,
    // fall back to scalar input_range.
    let (input_lower, input_upper) = if !ks.input_bounds.variable_inputs.is_empty() {
        let lo: Vec<f32> = ks
            .input_bounds
            .variable_inputs
            .iter()
            .map(|p| p.lower)
            .collect();
        let hi: Vec<f32> = ks
            .input_bounds
            .variable_inputs
            .iter()
            .map(|p| p.upper)
            .collect();
        (lo, hi)
    } else if let Some((lo, hi)) = ks.input_bounds.input_range {
        (vec![lo], vec![hi])
    } else {
        return None;
    };

    // Check all input bounds are finite.
    if input_lower.iter().any(|v| !v.is_finite()) || input_upper.iter().any(|v| !v.is_finite()) {
        return None;
    }

    // Determine output bounds: prefer per-element tensor bounds, fall back
    // to scalar lower/upper.
    let (output_lower, output_upper) = match (
        &ks.output_bounds.tensor_lower,
        &ks.output_bounds.tensor_upper,
    ) {
        (Some(tl), Some(tu)) if !tl.is_empty() && tl.len() == tu.len() => (tl.clone(), tu.clone()),
        _ => {
            if !ks.output_bounds.lower.is_finite() || !ks.output_bounds.upper.is_finite() {
                return None;
            }
            (vec![ks.output_bounds.lower], vec![ks.output_bounds.upper])
        }
    };

    // Check all output bounds are finite.
    if output_lower.iter().any(|v| !v.is_finite()) || output_upper.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let method = ConstructiveProofMethod::from_prop_method(ks.method);
    let is_crown = ks.method.is_tight();
    let is_sound = matches!(
        ks.soundness_mode,
        crate::soundness_compat::VerificationSoundnessMode::Sound
    );

    let proof = ConstructiveProofData::new(
        method,
        output_lower,
        output_upper,
        input_lower,
        input_upper,
        0, // Layer count not available from status entries
        true,
    );

    Some(EntryConstructiveProof {
        kernel_name: name.to_string(),
        proof,
        is_crown,
        is_sound,
    })
}

/// Extract constructive proofs from all active (non-stale) entries in a
/// [`VerifyStatus`].
fn extract_all_entry_proofs(status: &VerifyStatus) -> Vec<EntryConstructiveProof> {
    let kernels = status.kernels();
    let mut proofs = Vec::with_capacity(kernels.len());
    for (name, ks) in kernels {
        if ks.stale {
            continue;
        }
        if let Some(entry_proof) = extract_entry_proof(name, ks) {
            proofs.push(entry_proof);
        }
    }
    proofs.sort_by(|a, b| a.kernel_name.cmp(&b.kernel_name));
    proofs
}

// ---------------------------------------------------------------------------
// Deployment certificate
// ---------------------------------------------------------------------------

/// Deployment-ready CROWN certificate for the Kokoro TTS pipeline.
///
/// Extends [`KokoroCertificate`] with:
/// - Per-stage CROWN coverage mapping
/// - Deployment gate (pass/fail)
/// - Per-entry constructive proofs (machine-checkable bound data)
/// - Gap detection summary
///
/// This is the artifact that a deployment system checks before promoting
/// a Kokoro build to production. The `entry_proofs` field contains
/// machine-checkable proof data for each verified status entry, enabling
/// independent auditing without re-running NY.
///
/// Part of #4254.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct KokoroCrownCertificate {
    /// The underlying verification certificate.
    pub base_certificate: KokoroCertificate,
    /// Per-pipeline-stage CROWN coverage.
    pub stage_coverage: Vec<StageCrownCoverage>,
    /// Deployment gate evaluation.
    pub gate: DeploymentGate,
    /// Per-entry constructive proofs extracted from verification status.
    ///
    /// Each entry contains the input/output bounds and verification method
    /// from one kernel status entry, structured as a [`ConstructiveProofData`].
    /// An auditor can recompute interval arithmetic on each entry to
    /// independently confirm the claimed bounds.
    ///
    /// `None` for certificates generated before #4254 (backward compat).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub entry_proofs: Option<Vec<EntryConstructiveProof>>,
    /// SHA-256 content hash of the deployment certificate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
}

impl KokoroCrownCertificate {
    /// Serialize to pretty-printed JSON.
    pub fn to_json(&self) -> Result<String, VerifyError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, VerifyError> {
        Ok(serde_json::from_str(json)?)
    }

    /// Save the deployment certificate to a JSON file (atomic write).
    pub fn save(&self, path: &Path) -> Result<(), VerifyError> {
        use std::io::Write;

        let json = self.to_json()?;
        let tmp_path = {
            let mut s = path.as_os_str().to_owned();
            s.push(".tmp");
            std::path::PathBuf::from(s)
        };
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&tmp_path)?;
        file.write_all(json.as_bytes())?;
        file.sync_all()?;
        drop(file);

        if let Err(e) = std::fs::rename(&tmp_path, path) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(VerifyError::Io(e));
        }
        Ok(())
    }

    /// Load a deployment certificate from a JSON file.
    pub fn load(path: &Path) -> Result<Self, VerifyError> {
        let contents = std::fs::read_to_string(path)?;
        Self::from_json(&contents)
    }

    /// Verify content hash integrity.
    #[must_use]
    pub fn verify_integrity(&self) -> bool {
        match &self.content_hash {
            Some(stored) => {
                let recomputed = compute_deployment_content_hash(self);
                *stored == recomputed
            }
            None => true, // No hash = unsigned, accepted
        }
    }

    /// Number of pipeline stages with CROWN coverage.
    #[must_use]
    pub fn crown_covered_count(&self) -> usize {
        self.stage_coverage.iter().filter(|s| s.has_crown).count()
    }

    /// Names of pipeline stages without any verified bounds.
    #[must_use]
    pub fn uncovered_stages(&self) -> Vec<&str> {
        self.stage_coverage
            .iter()
            .filter(|s| !s.has_primary)
            .map(|s| s.stage_name.as_str())
            .collect()
    }

    /// Number of per-entry constructive proofs in this certificate.
    #[must_use]
    pub fn entry_proof_count(&self) -> usize {
        self.entry_proofs.as_ref().map_or(0, Vec::len)
    }

    /// Number of per-entry proofs using CROWN-family methods.
    #[must_use]
    pub fn crown_proof_count(&self) -> usize {
        self.entry_proofs
            .as_ref()
            .map_or(0, |proofs| proofs.iter().filter(|p| p.is_crown).count())
    }

    /// Number of per-entry proofs that are sound.
    #[must_use]
    pub fn sound_proof_count(&self) -> usize {
        self.entry_proofs
            .as_ref()
            .map_or(0, |proofs| proofs.iter().filter(|p| p.is_sound).count())
    }

    /// Whether all per-entry constructive proofs pass structural validation.
    ///
    /// Returns `true` if no entry proofs are present (backward compat) or
    /// if all proofs validate successfully.
    #[must_use]
    pub fn all_proofs_valid(&self) -> bool {
        self.entry_proofs.as_ref().map_or(true, |proofs| {
            proofs.iter().all(|ep| ep.proof.validate().is_ok())
        })
    }

    /// Look up a specific entry proof by kernel name.
    #[must_use]
    pub fn entry_proof_for(&self, kernel_name: &str) -> Option<&EntryConstructiveProof> {
        self.entry_proofs
            .as_ref()
            .and_then(|proofs| proofs.iter().find(|ep| ep.kernel_name == kernel_name))
    }

    /// Produce a summary of per-entry proof statistics.
    #[must_use]
    pub fn proof_summary(&self) -> ProofSummary {
        let total = self.entry_proof_count();
        let crown = self.crown_proof_count();
        let sound = self.sound_proof_count();
        let ibp = total.saturating_sub(crown);
        let machine_checkable = self.entry_proofs.as_ref().map_or(0, |proofs| {
            proofs
                .iter()
                .filter(|ep| ep.proof.is_machine_checkable())
                .count()
        });
        ProofSummary {
            total_proofs: total,
            crown_proofs: crown,
            ibp_proofs: ibp,
            sound_proofs: sound,
            machine_checkable,
        }
    }
}

/// Summary statistics for per-entry constructive proofs in a deployment
/// certificate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofSummary {
    /// Total number of per-entry proofs.
    pub total_proofs: usize,
    /// Number of proofs using CROWN-family methods.
    pub crown_proofs: usize,
    /// Number of proofs using IBP only.
    pub ibp_proofs: usize,
    /// Number of proofs from Sound entries.
    pub sound_proofs: usize,
    /// Number of proofs that are machine-checkable.
    pub machine_checkable: usize,
}

/// Compute SHA-256 content hash of the deployment certificate (excludes
/// `content_hash` field itself for deterministic hashing).
fn compute_deployment_content_hash(cert: &KokoroCrownCertificate) -> String {
    let mut hashable = cert.clone();
    hashable.content_hash = None;
    let json = serde_json::to_string(&hashable).unwrap_or_default();
    compute_bytes_hash(json.as_bytes())
}

// ---------------------------------------------------------------------------
// Certificate generation
// ---------------------------------------------------------------------------

/// Generate a deployment-ready CROWN certificate for Kokoro.
///
/// Pipeline:
/// 1. Load verification status from `nn_verify_status_kokoro.json`
/// 2. Generate the base `KokoroCertificate` with per-entry proofs
/// 3. Compute per-stage CROWN coverage from the gap detector registry
/// 4. Extract per-entry constructive proofs from status bounds
/// 5. Evaluate the deployment gate
/// 6. Sign with content hash
///
/// # Errors
///
/// Returns `VerifyError` if the status file cannot be loaded.
pub fn generate_deployment_certificate(
    config: &DeploymentConfig,
) -> Result<KokoroCrownCertificate, VerifyError> {
    let status = VerifyStatus::load(&config.status_path)?;
    generate_deployment_from_status(&status, config)
}

/// Generate a deployment certificate from an already-loaded status.
///
/// Testable core that separates I/O from logic.
pub(crate) fn generate_deployment_from_status(
    status: &VerifyStatus,
    config: &DeploymentConfig,
) -> Result<KokoroCrownCertificate, VerifyError> {
    // 1. Generate base certificate
    let base_config = crate::kokoro_certificate::CertificateConfig {
        model_hash: config.model_hash.clone(),
        status_path: config.status_path.clone(),
        gamma_crown_rev: config.gamma_crown_rev.clone(),
        include_stale: false,
    };
    let base_certificate = generate_from_status(status, &base_config)?;

    // 2. Compute per-stage CROWN coverage
    let stage_coverage = compute_stage_coverage(status);

    // 3. Extract per-entry constructive proofs from status bounds (#4254)
    let entry_proofs = extract_all_entry_proofs(status);
    let entry_proofs = if entry_proofs.is_empty() {
        None
    } else {
        Some(entry_proofs)
    };

    // 4. Evaluate deployment gate
    let gate = evaluate_gate(status, &stage_coverage, config);

    // 5. Assemble and sign
    let mut cert = KokoroCrownCertificate {
        base_certificate,
        stage_coverage,
        gate,
        entry_proofs,
        content_hash: None,
    };
    cert.content_hash = Some(compute_deployment_content_hash(&cert));

    Ok(cert)
}

#[cfg(test)]
#[path = "kokoro_crown_certificate_tests.rs"]
mod tests;
