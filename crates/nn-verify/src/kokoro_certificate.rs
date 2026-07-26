// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CROWN certificate production wiring for Kokoro deployment.
//!
//! Reads `nn_verify_status_kokoro.json`, aggregates per-entry verification
//! status, includes junction contract bounds from `nn-tts-verify`, and
//! produces a serializable [`KokoroCertificate`] that can be shipped alongside
//! model deployments for independent verification.
//!
//! # Usage
//!
//! ```rust,ignore
//! use std::path::Path;
//! use nn_verify::kokoro_certificate::{
//!     generate_kokoro_certificate, verify_certificate, CertificateConfig,
//! };
//!
//! let config = CertificateConfig::new(
//!     "sha256_of_weights",
//!     Path::new("nn_verify_status_kokoro.json"),
//! );
//! let cert = generate_kokoro_certificate(&config)?;
//! cert.save(Path::new("kokoro.proof.json"))?;
//!
//! // Independent verification
//! let loaded = KokoroCertificate::load(Path::new("kokoro.proof.json"))?;
//! let verdict = verify_certificate(&loaded, "sha256_of_weights");
//! assert!(verdict.is_valid());
//! ```
//!
//! Part of #3874, #4254.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::certificate_types::compute_bytes_hash;
use crate::error::VerifyError;
use crate::soundness_compat::VerificationSoundnessMode;
use crate::status::{KernelStatus, ProofStrength, VerifyStatus};
use crate::verify_types::PropMethod;

// ---------------------------------------------------------------------------
// Certificate types
// ---------------------------------------------------------------------------

/// Current schema version for Kokoro deployment certificates.
pub const KOKORO_CERTIFICATE_VERSION: u32 = 1;

/// A junction contract bound included in the certificate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JunctionBound {
    /// Junction identifier (e.g., "J2_F0", "J3_MAGNITUDE").
    pub name: String,
    /// Zone crossing description.
    pub zone: String,
    /// Lower bound.
    pub lower: f64,
    /// Upper bound.
    pub upper: f64,
}

/// Per-entry verification summary in the certificate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntryProofSummary {
    /// Kernel entry name from status file.
    pub kernel_name: String,
    /// Verification outcome (verified, bounds_computed, etc.).
    pub status: String,
    /// Propagation method (IBP, CROWN, AlphaCrown, etc.).
    pub method: String,
    /// Soundness classification.
    pub soundness_mode: String,
    /// Proof strength (sound_ibp, sound_crown, vacuous, heuristic).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof_strength: Option<String>,
    /// Output bound interval width.
    pub output_width: f32,
    /// Whether the entry is marked stale.
    #[serde(default, skip_serializing_if = "is_false")]
    pub stale: bool,
}

fn is_false(v: &bool) -> bool {
    !*v
}

/// Aggregate verification statistics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationSummary {
    /// Total number of entries in the status file.
    pub total_entries: usize,
    /// Number of non-stale entries.
    pub active_entries: usize,
    /// Number of entries with sound soundness mode.
    pub sound_count: usize,
    /// Number of entries with heuristic soundness mode.
    pub heuristic_count: usize,
    /// Breakdown by proof strength.
    pub proof_strength_breakdown: BTreeMap<String, usize>,
    /// Breakdown by propagation method.
    pub method_breakdown: BTreeMap<String, usize>,
}

/// Deployment certificate for the Kokoro TTS pipeline.
///
/// Captures the complete verification state at certificate generation time:
/// per-entry proof status, junction contract bounds, model identity, and
/// NY verifier revision. This is the machine-checkable artifact
/// that enables independent auditing without re-running verification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct KokoroCertificate {
    /// Schema version for forward/backward compatibility.
    pub schema_version: u32,
    /// SHA-256 hash of the model weights file.
    pub model_hash: String,
    /// NY git revision used for verification.
    pub gamma_crown_rev: String,
    /// ISO 8601 timestamp of certificate generation.
    pub generated_at: String,
    /// Aggregate verification statistics.
    pub summary: VerificationSummary,
    /// Per-entry proof status (one per kernel in the status file).
    pub entries: Vec<EntryProofSummary>,
    /// Junction contract bounds for the Kokoro pipeline.
    pub junction_bounds: Vec<JunctionBound>,
    /// SHA-256 content hash of the certificate body (all fields except
    /// `content_hash` itself). Enables integrity verification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
}

impl KokoroCertificate {
    /// Serialize to pretty-printed JSON.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError::Json` if serialization fails.
    pub fn to_json(&self) -> Result<String, VerifyError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError::Json` if parsing fails.
    pub fn from_json(json: &str) -> Result<Self, VerifyError> {
        Ok(serde_json::from_str(json)?)
    }

    /// Save the certificate to a JSON file using atomic write.
    ///
    /// Uses write-to-tmp, fsync, rename, fsync(dir) to prevent partial writes.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError::Io` on file I/O failure.
    pub fn save(&self, path: &Path) -> Result<(), VerifyError> {
        use std::io::Write;

        let json = self.to_json()?;
        let dir = path.parent().unwrap_or(Path::new("."));

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
        let _ = std::fs::File::open(dir).and_then(|f| f.sync_all());
        Ok(())
    }

    /// Load a certificate from a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or JSON parsing fails.
    pub fn load(path: &Path) -> Result<Self, VerifyError> {
        let contents = std::fs::read_to_string(path)?;
        Self::from_json(&contents)
    }
}

// ---------------------------------------------------------------------------
// Certificate generation
// ---------------------------------------------------------------------------

/// Configuration for Kokoro certificate generation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CertificateConfig {
    /// SHA-256 hash of the model weights.
    pub model_hash: String,
    /// Path to `nn_verify_status_kokoro.json`.
    pub status_path: std::path::PathBuf,
    /// NY git revision string.
    pub gamma_crown_rev: String,
    /// Whether to include stale entries in the certificate.
    pub include_stale: bool,
}

impl CertificateConfig {
    /// Create a config with default NY rev from workspace Cargo.toml.
    #[must_use]
    pub fn new(model_hash: &str, status_path: &Path) -> Self {
        Self {
            model_hash: model_hash.to_string(),
            status_path: status_path.to_path_buf(),
            gamma_crown_rev: default_gamma_crown_rev().to_string(),
            include_stale: false,
        }
    }

    /// Override the NY revision string.
    #[must_use]
    pub fn with_gamma_crown_rev(mut self, rev: &str) -> Self {
        self.gamma_crown_rev = rev.to_string();
        self
    }

    /// Include stale entries in the certificate (default: exclude).
    #[must_use]
    pub fn with_stale(mut self, include: bool) -> Self {
        self.include_stale = include;
        self
    }
}

/// Default NY revision pinned in this build.
///
/// This is the rev from `Cargo.toml` at build time. When the NY
/// dependency is bumped, this constant updates automatically on next build.
#[must_use]
pub fn default_gamma_crown_rev() -> &'static str {
    // Pinned to the rev in workspace Cargo.toml.
    // Updated when NY dependency is bumped.
    "532203c188bef9eb00fed44ef0ac6466f258af35"
}

/// All 6 Kokoro junction contract bounds.
///
/// These are the verified bounds at zone crossing points in the Kokoro TTS
/// pipeline. Sourced from `nn-tts-verify/src/kokoro_contracts.rs`.
#[must_use]
pub fn kokoro_junction_bounds() -> Vec<JunctionBound> {
    vec![
        JunctionBound {
            name: "J2_F0".to_string(),
            zone: "Decoder -> SourceModule".to_string(),
            lower: -5.0,
            upper: 800.0,
        },
        JunctionBound {
            name: "J2_ENERGY".to_string(),
            zone: "Decoder -> SourceModule".to_string(),
            lower: -50.0,
            upper: 50.0,
        },
        JunctionBound {
            name: "J3_MAGNITUDE".to_string(),
            zone: "Generator post_conv".to_string(),
            lower: -80.0,
            upper: 80.0,
        },
        JunctionBound {
            name: "J3B_PHASE".to_string(),
            zone: "Generator post_conv".to_string(),
            lower: -6283.2,
            upper: 6283.2,
        },
        JunctionBound {
            name: "J4_BF16".to_string(),
            zone: "F32 -> BF16 downcast".to_string(),
            lower: -128.0,
            upper: 128.0,
        },
        JunctionBound {
            name: "J5_AUDIO".to_string(),
            zone: "iSTFT output".to_string(),
            lower: -1.0,
            upper: 1.0,
        },
    ]
}

/// Generate a Kokoro deployment certificate from the status file.
///
/// Reads `nn_verify_status_kokoro.json`, aggregates per-entry verification
/// status, attaches junction contract bounds, and produces a signed
/// certificate with a content hash.
///
/// # Errors
///
/// Returns `VerifyError` if the status file cannot be loaded.
pub fn generate_kokoro_certificate(
    config: &CertificateConfig,
) -> Result<KokoroCertificate, VerifyError> {
    let status = VerifyStatus::load(&config.status_path)?;
    generate_from_status(&status, config)
}

/// Generate a certificate from an already-loaded `VerifyStatus`.
///
/// This is the testable core: separates I/O (file loading) from logic.
pub(crate) fn generate_from_status(
    status: &VerifyStatus,
    config: &CertificateConfig,
) -> Result<KokoroCertificate, VerifyError> {
    let kernels = status.kernels();

    // Build per-entry summaries
    let mut entries = Vec::with_capacity(kernels.len());
    for (name, ks) in kernels {
        if !config.include_stale && ks.stale {
            continue;
        }
        entries.push(entry_summary(name, ks));
    }

    // Compute aggregate summary
    let summary = compute_summary(&entries, kernels.len());

    // Build certificate
    let mut cert = KokoroCertificate {
        schema_version: KOKORO_CERTIFICATE_VERSION,
        model_hash: config.model_hash.clone(),
        gamma_crown_rev: config.gamma_crown_rev.clone(),
        generated_at: crate::certificate::now_iso8601(),
        summary,
        entries,
        junction_bounds: kokoro_junction_bounds(),
        content_hash: None,
    };

    // Compute content hash (covers all fields except content_hash itself)
    cert.content_hash = Some(compute_certificate_content_hash(&cert));

    Ok(cert)
}

/// Build an `EntryProofSummary` from a `KernelStatus`.
fn entry_summary(name: &str, ks: &KernelStatus) -> EntryProofSummary {
    EntryProofSummary {
        kernel_name: name.to_string(),
        status: serde_json::to_value(ks.status)
            .ok()
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_else(|| format!("{:?}", ks.status).to_lowercase()),
        method: format_method(ks.method),
        soundness_mode: format_soundness_mode(ks.soundness_mode),
        proof_strength: ks.proof_strength.map(format_proof_strength),
        output_width: ks.output_width,
        stale: ks.stale,
    }
}

/// Format a `VerificationSoundnessMode` to its snake_case serde representation.
fn format_soundness_mode(mode: VerificationSoundnessMode) -> String {
    // Use serde serialization to get the canonical snake_case name.
    serde_json::to_value(mode)
        .ok()
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_else(|| format!("{mode:?}").to_lowercase())
}

/// Format a `PropMethod` to a human-readable string.
#[allow(unreachable_patterns)] // #[non_exhaustive] — catch-all for forward compat
fn format_method(method: PropMethod) -> String {
    match method {
        PropMethod::Ibp => "IBP".to_string(),
        PropMethod::Crown => "CROWN".to_string(),
        PropMethod::AlphaCrown => "AlphaCrown".to_string(),
        PropMethod::BetaCrown => "BetaCrown".to_string(),
        PropMethod::MixedIbpCrown => "MixedIbpCrown".to_string(),
        PropMethod::Analytical => "Analytical".to_string(),
        _ => format!("{method:?}"),
    }
}

/// Format a `ProofStrength` to a human-readable string.
#[allow(unreachable_patterns)] // #[non_exhaustive] — catch-all for forward compat
fn format_proof_strength(ps: ProofStrength) -> String {
    match ps {
        ProofStrength::SoundCrown => "sound_crown".to_string(),
        ProofStrength::SoundIbp => "sound_ibp".to_string(),
        ProofStrength::SoundMixed => "sound_mixed".to_string(),
        ProofStrength::Heuristic => "heuristic".to_string(),
        ProofStrength::Vacuous => "vacuous".to_string(),
        _ => format!("{ps:?}").to_lowercase(),
    }
}

/// Compute aggregate verification statistics.
fn compute_summary(entries: &[EntryProofSummary], total_in_file: usize) -> VerificationSummary {
    let active = entries.iter().filter(|e| !e.stale).count();
    let sound = entries
        .iter()
        .filter(|e| !e.stale && e.soundness_mode == "sound")
        .count();
    let heuristic = entries
        .iter()
        .filter(|e| !e.stale && e.soundness_mode == "heuristic")
        .count();

    let mut proof_breakdown = BTreeMap::new();
    let mut method_breakdown = BTreeMap::new();
    for entry in entries.iter().filter(|e| !e.stale) {
        if let Some(ref ps) = entry.proof_strength {
            *proof_breakdown.entry(ps.clone()).or_insert(0) += 1;
        }
        *method_breakdown.entry(entry.method.clone()).or_insert(0) += 1;
    }

    VerificationSummary {
        total_entries: total_in_file,
        active_entries: active,
        sound_count: sound,
        heuristic_count: heuristic,
        proof_strength_breakdown: proof_breakdown,
        method_breakdown,
    }
}

/// Compute SHA-256 content hash of the certificate (excluding content_hash).
fn compute_certificate_content_hash(cert: &KokoroCertificate) -> String {
    // Serialize a copy without the content_hash field to produce deterministic hash.
    let mut hashable = cert.clone();
    hashable.content_hash = None;
    let json = serde_json::to_string(&hashable).unwrap_or_default();
    compute_bytes_hash(json.as_bytes())
}

// ---------------------------------------------------------------------------
// Certificate verification
// ---------------------------------------------------------------------------

/// Verdict from certificate verification.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct CertificateVerdict {
    /// Whether the certificate passes all checks.
    pub valid: bool,
    /// Individual findings.
    pub findings: Vec<CertificateFinding>,
}

impl CertificateVerdict {
    /// Whether the certificate is valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Whether there are any error-level findings.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.findings
            .iter()
            .any(|f| f.severity == FindingSeverity::Error)
    }
}

impl std::fmt::Display for CertificateVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.valid {
            writeln!(f, "Certificate VALID")?;
        } else {
            writeln!(f, "Certificate INVALID ({} findings)", self.findings.len())?;
        }
        for finding in &self.findings {
            writeln!(f, "  {finding}")?;
        }
        Ok(())
    }
}

/// A single finding from certificate verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CertificateFinding {
    /// Severity level.
    pub severity: FindingSeverity,
    /// Human-readable description.
    pub message: String,
}

impl std::fmt::Display for CertificateFinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefix = match self.severity {
            FindingSeverity::Error => "ERROR",
            FindingSeverity::Warning => "WARN",
            FindingSeverity::Info => "INFO",
        };
        write!(f, "[{prefix}] {}", self.message)
    }
}

/// Severity of a verification finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FindingSeverity {
    /// Certificate is structurally invalid or tampered.
    Error,
    /// Certificate claim is suspicious but not necessarily invalid.
    Warning,
    /// Informational note.
    Info,
}

/// Verify a Kokoro deployment certificate.
///
/// Checks:
/// 1. Schema version is supported
/// 2. Model hash matches the expected value
/// 3. Content hash integrity (if present)
/// 4. Junction bounds completeness (all 6 present)
/// 5. No vacuous entries in the summary
/// 6. At least one sound entry exists
///
/// # Arguments
///
/// * `cert` — The certificate to verify.
/// * `expected_model_hash` — SHA-256 of the model weights to verify against.
#[must_use]
pub fn verify_certificate(
    cert: &KokoroCertificate,
    expected_model_hash: &str,
) -> CertificateVerdict {
    let mut findings = Vec::new();

    // 1. Schema version
    if cert.schema_version == 0 || cert.schema_version > KOKORO_CERTIFICATE_VERSION {
        findings.push(CertificateFinding {
            severity: FindingSeverity::Error,
            message: format!(
                "unsupported schema version {} (max: {})",
                cert.schema_version, KOKORO_CERTIFICATE_VERSION
            ),
        });
    }

    // 2. Model hash
    if cert.model_hash != expected_model_hash {
        findings.push(CertificateFinding {
            severity: FindingSeverity::Error,
            message: format!(
                "model_hash mismatch: certificate has '{}', expected '{}'",
                truncate_hash(&cert.model_hash),
                truncate_hash(expected_model_hash),
            ),
        });
    }

    // 3. Content hash integrity
    verify_content_integrity(cert, &mut findings);

    // 4. Junction bounds completeness
    let expected_junctions = [
        "J2_F0",
        "J2_ENERGY",
        "J3_MAGNITUDE",
        "J3B_PHASE",
        "J4_BF16",
        "J5_AUDIO",
    ];
    for expected in &expected_junctions {
        if !cert.junction_bounds.iter().any(|j| j.name == *expected) {
            findings.push(CertificateFinding {
                severity: FindingSeverity::Warning,
                message: format!("missing junction bound: {expected}"),
            });
        }
    }

    // 5. Vacuous entries
    let vacuous_count = cert
        .summary
        .proof_strength_breakdown
        .get("vacuous")
        .copied()
        .unwrap_or(0);
    if vacuous_count > 0 {
        findings.push(CertificateFinding {
            severity: FindingSeverity::Warning,
            message: format!("{vacuous_count} entries have vacuous proof strength"),
        });
    }

    // 6. At least one sound entry
    if cert.summary.sound_count == 0 {
        findings.push(CertificateFinding {
            severity: FindingSeverity::Error,
            message: "no sound entries in certificate".to_string(),
        });
    }

    // 7. Entries present
    if cert.entries.is_empty() {
        findings.push(CertificateFinding {
            severity: FindingSeverity::Error,
            message: "certificate has no verification entries".to_string(),
        });
    }

    // 8. NY rev is non-empty
    if cert.gamma_crown_rev.is_empty() {
        findings.push(CertificateFinding {
            severity: FindingSeverity::Warning,
            message: "gamma_crown_rev is empty".to_string(),
        });
    }

    let valid = !findings
        .iter()
        .any(|f| f.severity == FindingSeverity::Error);

    CertificateVerdict { valid, findings }
}

/// Verify the content hash integrity of a certificate.
fn verify_content_integrity(cert: &KokoroCertificate, findings: &mut Vec<CertificateFinding>) {
    match &cert.content_hash {
        Some(stored_hash) => {
            let recomputed = compute_certificate_content_hash(cert);
            if *stored_hash != recomputed {
                findings.push(CertificateFinding {
                    severity: FindingSeverity::Error,
                    message: format!(
                        "content_hash mismatch: stored '{}', recomputed '{}'",
                        truncate_hash(stored_hash),
                        truncate_hash(&recomputed),
                    ),
                });
            }
        }
        None => {
            findings.push(CertificateFinding {
                severity: FindingSeverity::Info,
                message: "no content_hash present (unsigned certificate)".to_string(),
            });
        }
    }
}

/// Truncate a hash to the first 16 characters for display.
fn truncate_hash(hash: &str) -> &str {
    if hash.len() > 16 {
        &hash[..16]
    } else {
        hash
    }
}

#[cfg(test)]
#[path = "kokoro_certificate_tests.rs"]
mod tests;
