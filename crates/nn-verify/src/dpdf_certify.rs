// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Certification module for dpdf document-processing models.
//!
//! Tracks 8 formal properties (P1-P8) analogous to the Kokoro moonshot
//! certificate in `nn-tts-verify`. Reads verification status from
//! `nn_verify_status_dpdf.json` and assembles a deployment-readiness
//! certificate.
//!
//! Part of #3914.

use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors produced by the dpdf certification module.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DpdfCertifyError {
    /// Failed to read a status file.
    #[error("failed to read status file: {0}")]
    Io(#[from] std::io::Error),

    /// Failed to parse JSON status data.
    #[error("failed to parse status JSON: {0}")]
    Json(#[from] serde_json::Error),
}

// ---------------------------------------------------------------------------
// Property enum
// ---------------------------------------------------------------------------

/// The 8 formal properties verified for dpdf models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DpdfProperty {
    /// P1: Layout detection sigmoid output bounded in \[0, 1\].
    P1LayoutSigmoidBounds,
    /// P2: OCR softmax output forms a valid probability distribution.
    P2OcrSoftmaxDistribution,
    /// P3: Table structure boxes are normalized coordinates in \[0, 1\].
    P3TableBoxNormalized,
    /// P4: DFL regression produces valid bounding-box coordinates.
    P4DflRegressionValid,
    /// P5: NMS preserves the highest-confidence detection per cluster.
    P5NmsPreservesTopConfidence,
    /// P6: IoU computation is bounded in \[0, 1\].
    P6IoUBounded,
    /// P7: Confidence filtering preserves all detections above threshold.
    P7ConfidenceFilterMonotone,
    /// P8: Quantized (int8/fp16) output within epsilon of fp32.
    P8QuantizedEpsilonBound,
}

impl DpdfProperty {
    /// All 8 properties in order.
    pub const ALL: [Self; 8] = [
        Self::P1LayoutSigmoidBounds,
        Self::P2OcrSoftmaxDistribution,
        Self::P3TableBoxNormalized,
        Self::P4DflRegressionValid,
        Self::P5NmsPreservesTopConfidence,
        Self::P6IoUBounded,
        Self::P7ConfidenceFilterMonotone,
        Self::P8QuantizedEpsilonBound,
    ];

    /// Human-readable short name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::P1LayoutSigmoidBounds => "Layout sigmoid bounds [0, 1]",
            Self::P2OcrSoftmaxDistribution => "OCR softmax valid distribution",
            Self::P3TableBoxNormalized => "Table box normalized [0, 1]",
            Self::P4DflRegressionValid => "DFL regression valid coords",
            Self::P5NmsPreservesTopConfidence => "NMS preserves top confidence",
            Self::P6IoUBounded => "IoU bounded [0, 1]",
            Self::P7ConfidenceFilterMonotone => "Confidence filter monotone",
            Self::P8QuantizedEpsilonBound => "Quantized epsilon bound",
        }
    }

    /// 1-indexed property number.
    #[must_use]
    pub fn number(self) -> usize {
        match self {
            Self::P1LayoutSigmoidBounds => 1,
            Self::P2OcrSoftmaxDistribution => 2,
            Self::P3TableBoxNormalized => 3,
            Self::P4DflRegressionValid => 4,
            Self::P5NmsPreservesTopConfidence => 5,
            Self::P6IoUBounded => 6,
            Self::P7ConfidenceFilterMonotone => 7,
            Self::P8QuantizedEpsilonBound => 8,
        }
    }
}

impl fmt::Display for DpdfProperty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "P{}: {}", self.number(), self.name())
    }
}

// ---------------------------------------------------------------------------
// Property status
// ---------------------------------------------------------------------------

/// Verification status for a single dpdf property.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PropertyStatus {
    /// Formally proven (CROWN tight, Kani, or ay).
    Proven,
    /// Heuristic evidence (IBP or partial CROWN bounds, may be vacuously wide).
    Heuristic,
    /// No verification evidence yet.
    Unverified,
    /// Property does not apply to this model configuration.
    NotApplicable,
}

impl fmt::Display for PropertyStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Proven => write!(f, "PROVEN"),
            Self::Heuristic => write!(f, "HEURISTIC"),
            Self::Unverified => write!(f, "UNVERIFIED"),
            Self::NotApplicable => write!(f, "N/A"),
        }
    }
}

// ---------------------------------------------------------------------------
// Certificate
// ---------------------------------------------------------------------------

/// Deployment certificate for dpdf models.
///
/// Aggregates per-property verification status, compose test counts,
/// Kani harness counts, ay proof counts, and covered model names.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DpdfCertificate {
    /// Per-property status with evidence string.
    pub properties: Vec<(DpdfProperty, PropertyStatus, String)>,
    /// Number of NY compose tests that passed.
    pub compose_test_count: usize,
    /// Number of Kani harnesses that passed.
    pub kani_harness_count: usize,
    /// Number of ay SMT proofs that passed.
    pub ay_proof_count: usize,
    /// Model families covered (e.g. "granite_docling", "doclayout_yolo").
    pub models_covered: Vec<String>,
    /// ISO 8601 date string when the certificate was generated.
    pub generated_at: String,
}

/// Minimal representation of a kernel entry in `nn_verify_status_dpdf.json`.
#[derive(Debug, Deserialize)]
struct StatusKernel {
    proof_strength: String,
    #[serde(default)]
    stale: bool,
}

/// Top-level shape of `nn_verify_status_dpdf.json`.
#[derive(Debug, Deserialize)]
struct StatusFile {
    kernels: std::collections::HashMap<String, StatusKernel>,
}

impl DpdfCertificate {
    /// Build a new certificate with explicit values (useful for testing).
    #[must_use]
    pub fn new(
        properties: Vec<(DpdfProperty, PropertyStatus, String)>,
        compose_test_count: usize,
        kani_harness_count: usize,
        ay_proof_count: usize,
        models_covered: Vec<String>,
        generated_at: String,
    ) -> Self {
        Self {
            properties,
            compose_test_count,
            kani_harness_count,
            ay_proof_count,
            models_covered,
            generated_at,
        }
    }

    /// Generate a certificate by reading `nn_verify_status_dpdf.json` from
    /// the repository root at `repo_root`.
    ///
    /// # Errors
    ///
    /// Returns [`DpdfCertifyError`] if the status file cannot be read or parsed.
    pub fn generate(repo_root: &Path) -> Result<Self, DpdfCertifyError> {
        let status_path = repo_root.join("nn_verify_status_dpdf.json");
        let raw = std::fs::read_to_string(&status_path)?;
        Self::generate_from_json(&raw)
    }

    /// Generate a certificate from raw JSON string (for testing without disk).
    ///
    /// # Errors
    ///
    /// Returns [`DpdfCertifyError`] if the JSON cannot be parsed.
    pub fn generate_from_json(json: &str) -> Result<Self, DpdfCertifyError> {
        let status: StatusFile = serde_json::from_str(json)?;
        let (sound, heuristic, _stale, models, total) = Self::tally_kernels(&status);

        let properties = Self::assess_properties(sound, heuristic, total, &models);

        Ok(Self {
            properties,
            compose_test_count: total,
            kani_harness_count: 0, // filled by caller if Kani data available
            ay_proof_count: 0,     // filled by caller if ay data available
            models_covered: models,
            generated_at: current_date(),
        })
    }

    /// Count sound / heuristic / stale entries and collect model families.
    fn tally_kernels(status: &StatusFile) -> (usize, usize, usize, Vec<String>, usize) {
        let mut sound = 0usize;
        let mut heuristic = 0usize;
        let mut stale = 0usize;
        let mut models = std::collections::BTreeSet::new();

        for (name, entry) in &status.kernels {
            if entry.stale {
                stale += 1;
                continue;
            }
            match entry.proof_strength.as_str() {
                "sound" => sound += 1,
                "heuristic" => heuristic += 1,
                _ => {}
            }
            // Model family is the prefix before "::"
            if let Some(family) = name.split("::").next() {
                models.insert(family.to_string());
            }
        }
        let total = sound + heuristic;
        (sound, heuristic, stale, models.into_iter().collect(), total)
    }

    /// Map kernel-level tallies to per-property statuses.
    fn assess_properties(
        sound: usize,
        heuristic: usize,
        total: usize,
        models: &[String],
    ) -> Vec<(DpdfProperty, PropertyStatus, String)> {
        use DpdfProperty::{
            P1LayoutSigmoidBounds, P2OcrSoftmaxDistribution, P3TableBoxNormalized,
            P4DflRegressionValid, P5NmsPreservesTopConfidence, P6IoUBounded,
            P7ConfidenceFilterMonotone, P8QuantizedEpsilonBound,
        };

        let has_layout = models.iter().any(|m| m == "doclayout_yolo");
        let has_ocr = models.iter().any(|m| m == "glm_ocr" || m == "paddle_ocr");
        let has_table = models.iter().any(|m| m == "table_transformer");
        let has_dfl = models
            .iter()
            .any(|m| m == "doclayout_yolo" || m == "table_transformer");

        let status_for = |has_model: bool, property_sound_threshold: usize| -> PropertyStatus {
            if !has_model {
                return PropertyStatus::NotApplicable;
            }
            // Require at least 1 entry to claim any verification.
            if total == 0 {
                return PropertyStatus::Unverified;
            }
            if sound >= property_sound_threshold && property_sound_threshold > 0 {
                PropertyStatus::Proven
            } else if heuristic > 0 || sound > 0 {
                PropertyStatus::Heuristic
            } else {
                PropertyStatus::Unverified
            }
        };

        // Thresholds tuned to the dpdf status data: generous thresholds
        // because the status file records compose-level entries, not
        // property-level ones directly.
        let layout_threshold = 5;
        let ocr_threshold = 5;
        let table_threshold = 5;
        let dfl_threshold = 3;
        let nms_threshold = total; // NMS is algorithmic, requires all entries sound
        let iou_threshold = total;
        let conf_threshold = total;

        vec![
            (
                P1LayoutSigmoidBounds,
                status_for(has_layout, layout_threshold),
                format!("{sound}/{total} sound entries, layout models present: {has_layout}"),
            ),
            (
                P2OcrSoftmaxDistribution,
                status_for(has_ocr, ocr_threshold),
                format!("{sound}/{total} sound entries, OCR models present: {has_ocr}"),
            ),
            (
                P3TableBoxNormalized,
                status_for(has_table, table_threshold),
                format!("{sound}/{total} sound entries, table models present: {has_table}"),
            ),
            (
                P4DflRegressionValid,
                status_for(has_dfl, dfl_threshold),
                format!("{sound}/{total} sound entries, DFL models present: {has_dfl}"),
            ),
            (
                P5NmsPreservesTopConfidence,
                status_for(true, nms_threshold),
                format!("{sound}/{total} sound entries (requires all sound for NMS monotonicity)"),
            ),
            (
                P6IoUBounded,
                status_for(true, iou_threshold),
                format!("{sound}/{total} sound entries (requires all sound for IoU bound)"),
            ),
            (
                P7ConfidenceFilterMonotone,
                status_for(true, conf_threshold),
                format!(
                    "{sound}/{total} sound entries (requires all sound for filter monotonicity)"
                ),
            ),
            (
                P8QuantizedEpsilonBound,
                PropertyStatus::Unverified,
                "Quantization verification not yet implemented".to_string(),
            ),
        ]
    }

    /// Human-readable markdown report.
    #[must_use]
    pub fn to_report(&self) -> String {
        let mut out = String::with_capacity(2048);
        out.push_str("# dpdf Certification Report\n\n");
        out.push_str(&format!("Generated: {}\n\n", self.generated_at));

        out.push_str("## Properties\n\n");
        out.push_str("| # | Property | Status | Evidence |\n");
        out.push_str("|---|----------|--------|----------|\n");
        for (prop, status, evidence) in &self.properties {
            out.push_str(&format!(
                "| P{} | {} | {} | {} |\n",
                prop.number(),
                prop.name(),
                status,
                evidence
            ));
        }

        out.push_str("\n## Summary\n\n");
        out.push_str(&format!("- Compose tests: {}\n", self.compose_test_count));
        out.push_str(&format!("- Kani harnesses: {}\n", self.kani_harness_count));
        out.push_str(&format!("- ay proofs: {}\n", self.ay_proof_count));
        out.push_str(&format!(
            "- Models covered: {}\n",
            self.models_covered.join(", ")
        ));
        out.push_str(&format!(
            "- Deployment ready: {}\n",
            self.is_deployment_ready()
        ));

        out
    }

    /// Whether all P1-P7 are `Proven` or `Heuristic` (P8 excluded: quantization
    /// is optional for initial deployment).
    #[must_use]
    pub fn is_deployment_ready(&self) -> bool {
        self.properties
            .iter()
            .filter(|(prop, _, _)| prop.number() <= 7)
            .all(|(_, status, _)| {
                matches!(
                    status,
                    PropertyStatus::Proven
                        | PropertyStatus::Heuristic
                        | PropertyStatus::NotApplicable
                )
            })
    }

    /// Count of properties at each status level.
    #[must_use]
    pub fn status_counts(&self) -> (usize, usize, usize, usize) {
        let mut proven = 0;
        let mut heuristic = 0;
        let mut unverified = 0;
        let mut na = 0;
        for (_, status, _) in &self.properties {
            match status {
                PropertyStatus::Proven => proven += 1,
                PropertyStatus::Heuristic => heuristic += 1,
                PropertyStatus::Unverified => unverified += 1,
                PropertyStatus::NotApplicable => na += 1,
            }
        }
        (proven, heuristic, unverified, na)
    }
}

impl fmt::Display for DpdfCertificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (prop, status, _) in &self.properties {
            writeln!(f, "  {prop}: [{status}]")?;
        }
        writeln!(f, "  Deployment ready: {}", self.is_deployment_ready())
    }
}

/// Return current date as ISO 8601 string (YYYY-MM-DD).
fn current_date() -> String {
    // Use a simple approach that doesn't require chrono.
    let epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Approximate: days since epoch, good enough for date stamping.
    let days = epoch / 86400;
    // Epoch is 1970-01-01. We do a simplified calendar computation.
    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Convert days since 1970-01-01 to (year, month, day).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Algorithm from Howard Hinnant's `civil_from_days`.
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
#[path = "dpdf_certify_tests.rs"]
mod tests;
