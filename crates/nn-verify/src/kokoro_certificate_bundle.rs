// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro CROWN certificate bundle for deployment gating.
//!
//! Aggregates per-stage CROWN certificates from `nn_verify_status_kokoro.json`
//! into a single [`KokoroCertificateBundle`] with overall soundness computation
//! and a deployment-readiness gate.
//!
//! The bundle is the top-level deployment artifact: it wraps the
//! [`KokoroCrownCertificate`] (which handles per-entry proofs and stage
//! coverage) and adds soundness breakdown aggregation, deployment-readiness
//! evaluation, and JSON serialization for CI/CD auditing.
//!
//! # Usage
//!
//! ```rust,ignore
//! use nn_verify::kokoro_certificate_bundle::{
//!     KokoroCertificateBundle, BundleConfig,
//! };
//!
//! let config = BundleConfig::new(status_path);
//! let bundle = KokoroCertificateBundle::from_status_file(&config)?;
//! assert!(bundle.is_deployment_ready());
//! bundle.save(Path::new("kokoro_bundle.proof.json"))?;
//! ```
//!
//! Part of #4254.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::certificate_types::compute_bytes_hash;
use crate::error::VerifyError;
use crate::kokoro_crown_certificate::{
    generate_deployment_from_status, DeploymentConfig, KokoroCrownCertificate,
};
use crate::status::{ProofStrength, VerifyStatus};
use crate::verify_types::PropMethod;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for Kokoro certificate bundle generation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BundleConfig {
    /// Path to `nn_verify_status_kokoro.json`.
    pub status_path: std::path::PathBuf,
    /// SHA-256 hash of the Kokoro model weights.
    pub model_hash: String,
    /// NY git revision string.
    pub gamma_crown_rev: String,
    /// Minimum fraction of active entries that must be sound (0.0..=1.0).
    /// Default: 0.90 (90%).
    pub min_sound_ratio: f64,
    /// Maximum allowed number of vacuous entries. Default: 0.
    pub max_vacuous: usize,
    /// Maximum allowed number of heuristic entries. Default: no limit.
    pub max_heuristic: Option<usize>,
    /// Minimum number of pipeline stages with CROWN coverage. Default: 3.
    pub min_crown_stages: usize,
    /// Maximum allowed uncovered pipeline stages (gaps). Default: 0.
    pub max_gaps: usize,
}

impl BundleConfig {
    /// Create a config with production defaults.
    #[must_use]
    pub fn new(status_path: &Path) -> Self {
        Self {
            status_path: status_path.to_path_buf(),
            model_hash: String::new(),
            gamma_crown_rev: crate::kokoro_certificate::default_gamma_crown_rev().to_string(),
            min_sound_ratio: 0.90,
            max_vacuous: 0,
            max_heuristic: None,
            min_crown_stages: 3,
            max_gaps: 0,
        }
    }

    /// Set the model hash.
    #[must_use]
    pub fn with_model_hash(mut self, hash: &str) -> Self {
        self.model_hash = hash.to_string();
        self
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

    /// Set the maximum allowed heuristic entry count.
    #[must_use]
    pub fn with_max_heuristic(mut self, max: usize) -> Self {
        self.max_heuristic = Some(max);
        self
    }
}

// ---------------------------------------------------------------------------
// Soundness breakdown
// ---------------------------------------------------------------------------

/// Per-entry soundness classification for the bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EntrySoundness {
    /// Sound with CROWN-family method (tight bounds).
    SoundCrown,
    /// Sound with IBP method.
    SoundIbp,
    /// Sound with mixed IBP+CROWN.
    SoundMixed,
    /// Heuristic bounds (approximations used).
    Heuristic,
    /// Bounds too wide to be useful.
    Vacuous,
}

impl EntrySoundness {
    /// Whether this classification is sound (non-heuristic, non-vacuous).
    #[must_use]
    pub fn is_sound(self) -> bool {
        matches!(self, Self::SoundCrown | Self::SoundIbp | Self::SoundMixed)
    }

    /// Convert from [`ProofStrength`].
    #[must_use]
    pub(crate) fn from_proof_strength(ps: ProofStrength) -> Self {
        match ps {
            ProofStrength::SoundCrown => Self::SoundCrown,
            ProofStrength::SoundIbp => Self::SoundIbp,
            ProofStrength::SoundMixed => Self::SoundMixed,
            ProofStrength::Heuristic => Self::Heuristic,
            ProofStrength::Vacuous => Self::Vacuous,
            // Forward compat for future variants.
            _ => Self::Heuristic,
        }
    }
}

/// Per-entry soundness record in the certificate bundle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntrySoundnessRecord {
    /// Kernel entry name from the status file.
    pub kernel_name: String,
    /// Soundness classification.
    pub soundness: EntrySoundness,
    /// Propagation method used.
    pub method: String,
    /// Output bound width.
    pub output_width: f32,
}

/// Aggregate soundness breakdown across all active entries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoundnessBreakdown {
    /// Total number of active (non-stale) entries.
    pub total: usize,
    /// Number of sound entries (SoundCrown + SoundIbp + SoundMixed).
    pub sound: usize,
    /// Number of heuristic entries.
    pub heuristic: usize,
    /// Number of vacuous entries.
    pub vacuous: usize,
    /// Ratio of sound entries to total (0.0..=1.0).
    pub sound_ratio: f64,
    /// Per-method entry counts.
    pub method_counts: BTreeMap<String, usize>,
    /// Per-entry soundness records (sorted by kernel name).
    pub entries: Vec<EntrySoundnessRecord>,
}

impl SoundnessBreakdown {
    /// Whether all entries are sound and none are vacuous.
    #[must_use]
    pub fn all_sound(&self) -> bool {
        self.total > 0 && self.sound == self.total && self.vacuous == 0
    }
}

/// Compute the soundness breakdown from a verification status.
fn compute_soundness_breakdown(status: &VerifyStatus) -> SoundnessBreakdown {
    let kernels = status.kernels();
    let mut entries = Vec::new();
    let mut sound = 0usize;
    let mut heuristic = 0usize;
    let mut vacuous = 0usize;
    let mut method_counts: BTreeMap<String, usize> = BTreeMap::new();

    for (name, ks) in kernels {
        if ks.stale {
            continue;
        }
        let ps = ks.proof_strength.unwrap_or_else(|| {
            crate::status::compute_proof_strength(ks.soundness_mode, ks.method, ks.output_width)
        });
        let es = EntrySoundness::from_proof_strength(ps);

        match es {
            EntrySoundness::SoundCrown | EntrySoundness::SoundIbp | EntrySoundness::SoundMixed => {
                sound += 1;
            }
            EntrySoundness::Heuristic => {
                heuristic += 1;
            }
            EntrySoundness::Vacuous => {
                vacuous += 1;
            }
        }

        *method_counts.entry(format_method(ks.method)).or_insert(0) += 1;

        entries.push(EntrySoundnessRecord {
            kernel_name: name.clone(),
            soundness: es,
            method: format_method(ks.method),
            output_width: ks.output_width,
        });
    }

    entries.sort_by(|a, b| a.kernel_name.cmp(&b.kernel_name));
    let total = sound + heuristic + vacuous;
    let sound_ratio = if total > 0 {
        sound as f64 / total as f64
    } else {
        0.0
    };

    SoundnessBreakdown {
        total,
        sound,
        heuristic,
        vacuous,
        sound_ratio,
        method_counts,
        entries,
    }
}

/// Format a `PropMethod` to a human-readable string.
#[allow(unreachable_patterns)]
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

// ---------------------------------------------------------------------------
// Certificate bundle
// ---------------------------------------------------------------------------

/// Aggregated CROWN certificate bundle for Kokoro deployment.
///
/// This is the top-level deployment artifact that combines:
/// - The underlying [`KokoroCrownCertificate`] (per-entry proofs, stage
///   coverage, deployment gate)
/// - A [`SoundnessBreakdown`] with per-entry sound/heuristic/vacuous
///   classification
/// - An `is_deployment_ready()` check combining all gate criteria
///
/// Serializes to JSON for CI/CD pipeline integration and audit trails.
///
/// Part of #4254.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct KokoroCertificateBundle {
    /// The underlying CROWN deployment certificate.
    pub certificate: KokoroCrownCertificate,
    /// Aggregate soundness breakdown across all active entries.
    pub soundness: SoundnessBreakdown,
    /// Configuration thresholds used for deployment gating.
    pub thresholds: DeploymentThresholds,
    /// ISO 8601 timestamp of bundle generation.
    pub generated_at: String,
    /// SHA-256 content hash of the bundle (all fields except this one).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
}

/// Deployment threshold configuration recorded in the bundle for auditability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeploymentThresholds {
    /// Minimum sound ratio required.
    pub min_sound_ratio: f64,
    /// Maximum vacuous entries allowed.
    pub max_vacuous: usize,
    /// Maximum heuristic entries allowed (`None` = no limit).
    pub max_heuristic: Option<usize>,
    /// Minimum CROWN-covered pipeline stages.
    pub min_crown_stages: usize,
    /// Maximum uncovered pipeline stages.
    pub max_gaps: usize,
}

impl KokoroCertificateBundle {
    /// Whether this bundle meets all deployment-readiness criteria.
    ///
    /// Checks:
    /// 1. The underlying deployment gate passes
    /// 2. Sound ratio meets the threshold
    /// 3. Vacuous count is within the threshold
    /// 4. Heuristic count is within the threshold (if configured)
    /// 5. At least one entry exists
    #[must_use]
    pub fn is_deployment_ready(&self) -> bool {
        if !self.certificate.gate.is_deployable() {
            return false;
        }
        if self.soundness.total == 0 {
            return false;
        }
        if self.soundness.sound_ratio < self.thresholds.min_sound_ratio {
            return false;
        }
        if self.soundness.vacuous > self.thresholds.max_vacuous {
            return false;
        }
        if let Some(max_h) = self.thresholds.max_heuristic {
            if self.soundness.heuristic > max_h {
                return false;
            }
        }
        true
    }

    /// Number of sound entries.
    #[must_use]
    pub fn sound_count(&self) -> usize {
        self.soundness.sound
    }

    /// Number of heuristic entries.
    #[must_use]
    pub fn heuristic_count(&self) -> usize {
        self.soundness.heuristic
    }

    /// Number of vacuous entries.
    #[must_use]
    pub fn vacuous_count(&self) -> usize {
        self.soundness.vacuous
    }

    /// Total active (non-stale) entries.
    #[must_use]
    pub fn total_entries(&self) -> usize {
        self.soundness.total
    }

    /// Sound ratio (sound / total).
    #[must_use]
    pub fn sound_ratio(&self) -> f64 {
        self.soundness.sound_ratio
    }

    /// Per-entry soundness records.
    #[must_use]
    pub fn entry_records(&self) -> &[EntrySoundnessRecord] {
        &self.soundness.entries
    }

    /// Look up the soundness classification for a specific kernel entry.
    #[must_use]
    pub fn soundness_for(&self, kernel_name: &str) -> Option<EntrySoundness> {
        self.soundness
            .entries
            .iter()
            .find(|e| e.kernel_name == kernel_name)
            .map(|e| e.soundness)
    }

    /// Serialize to pretty-printed JSON.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError` if serialization fails.
    pub fn to_json(&self) -> Result<String, VerifyError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError` if the JSON is malformed.
    pub fn from_json(json: &str) -> Result<Self, VerifyError> {
        Ok(serde_json::from_str(json)?)
    }

    /// Save the bundle to a JSON file (atomic write).
    ///
    /// # Errors
    ///
    /// Returns `VerifyError` on I/O or serialization failure.
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

    /// Load a bundle from a JSON file.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError` on I/O or deserialization failure.
    pub fn load(path: &Path) -> Result<Self, VerifyError> {
        let contents = std::fs::read_to_string(path)?;
        Self::from_json(&contents)
    }

    /// Verify content hash integrity.
    #[must_use]
    pub fn verify_integrity(&self) -> bool {
        match &self.content_hash {
            Some(stored) => {
                let recomputed = compute_bundle_content_hash(self);
                *stored == recomputed
            }
            None => true,
        }
    }

    /// Generate a bundle from the status file specified in the config.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError` if the status file cannot be loaded.
    pub fn from_status_file(config: &BundleConfig) -> Result<Self, VerifyError> {
        let status = VerifyStatus::load(&config.status_path)?;
        Self::from_status(&status, config)
    }

    /// Generate a bundle from an already-loaded status.
    ///
    /// Testable core that separates I/O from logic.
    pub fn from_status(status: &VerifyStatus, config: &BundleConfig) -> Result<Self, VerifyError> {
        // Build the underlying deployment certificate.
        let deploy_config = DeploymentConfig::new(&config.model_hash, &config.status_path);
        let deploy_config = deploy_config
            .with_gamma_crown_rev(&config.gamma_crown_rev)
            .with_min_sound_ratio(config.min_sound_ratio);
        let certificate = generate_deployment_from_status(status, &deploy_config)?;

        // Compute soundness breakdown.
        let soundness = compute_soundness_breakdown(status);

        let thresholds = DeploymentThresholds {
            min_sound_ratio: config.min_sound_ratio,
            max_vacuous: config.max_vacuous,
            max_heuristic: config.max_heuristic,
            min_crown_stages: config.min_crown_stages,
            max_gaps: config.max_gaps,
        };

        let mut bundle = Self {
            certificate,
            soundness,
            thresholds,
            generated_at: crate::certificate::now_iso8601(),
            content_hash: None,
        };
        bundle.content_hash = Some(compute_bundle_content_hash(&bundle));

        Ok(bundle)
    }
}

/// Compute SHA-256 content hash of the bundle (excludes `content_hash`).
fn compute_bundle_content_hash(bundle: &KokoroCertificateBundle) -> String {
    let mut hashable = bundle.clone();
    hashable.content_hash = None;
    let json = serde_json::to_string(&hashable).unwrap_or_default();
    compute_bytes_hash(json.as_bytes())
}

#[cfg(test)]
#[path = "kokoro_certificate_bundle_tests.rs"]
mod tests;
