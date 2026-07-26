// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Verification status report generator for the nn dashboard.
//!
//! Aggregates verification status across all models from per-model
//! `nn_verify_status_*.json` files and `operational_state.json` into
//! a unified [`StatusReport`] with per-model breakdowns, Kani harness
//! counts, gap summaries, and trend comparisons.
//!
//! Part of #3942.

use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::VerifyError;
use crate::status::{
    model_status_path, KernelStatus, ProofStrength, VerifyStatus, MODEL_CATEGORIES,
};

/// Breakdown of verification entries by soundness and proof strength.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationBreakdown {
    /// Total non-stale entries.
    pub total: usize,
    /// Entries with `soundness_mode == Sound`.
    pub sound: usize,
    /// Entries with `soundness_mode == Heuristic`.
    pub heuristic: usize,
    /// Stale entries (excluded from sound/heuristic counts).
    pub stale: usize,
    /// Entries with `proof_strength == SoundCrown`.
    pub sound_crown: usize,
    /// Entries with `proof_strength == SoundIbp`.
    pub sound_ibp: usize,
    /// Entries with `proof_strength == SoundMixed`.
    pub sound_mixed: usize,
    /// Entries with `proof_strength == Heuristic`.
    pub heuristic_non_vacuous: usize,
    /// Entries with `proof_strength == Vacuous`.
    pub vacuous: usize,
}

impl VerificationBreakdown {
    /// Build a breakdown from a slice of kernel status entries.
    ///
    /// Includes all entries (stale and non-stale); stale entries are counted
    /// separately and excluded from soundness/proof-strength totals.
    #[must_use]
    pub fn from_entries(entries: &[&KernelStatus]) -> Self {
        let mut b = Self::default();
        for entry in entries {
            if entry.stale {
                b.stale += 1;
                continue;
            }
            b.total += 1;
            match entry.soundness_mode {
                crate::soundness_compat::VerificationSoundnessMode::Sound => b.sound += 1,
                crate::soundness_compat::VerificationSoundnessMode::Heuristic => {
                    b.heuristic += 1;
                }
            }
            let strength = entry.proof_strength.unwrap_or_else(|| {
                crate::status::compute_proof_strength(
                    entry.soundness_mode,
                    entry.method,
                    entry.output_width,
                )
            });
            match strength {
                ProofStrength::SoundCrown => b.sound_crown += 1,
                ProofStrength::SoundIbp => b.sound_ibp += 1,
                ProofStrength::SoundMixed => b.sound_mixed += 1,
                ProofStrength::Heuristic => b.heuristic_non_vacuous += 1,
                ProofStrength::Vacuous => b.vacuous += 1,
            }
        }
        b
    }

    /// Fraction of non-stale entries that are sound (0.0..=1.0).
    /// Returns 0.0 if there are no non-stale entries.
    #[must_use]
    pub fn sound_fraction(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.sound as f64 / self.total as f64
    }
}

/// Per-model verification summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSummary {
    /// Model category name (e.g., "kokoro", "demucs").
    pub model: String,
    /// Verification breakdown for this model.
    pub breakdown: VerificationBreakdown,
}

/// Summary of verification gaps from the gap detector.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GapSummary {
    /// Number of pipeline stages checked.
    pub stages_checked: usize,
    /// Number of stages with no verification bounds.
    pub gaps: usize,
    /// Number of stages with vacuous bounds.
    pub vacuous: usize,
}

/// Trend comparison between current and previous operational state.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Trend {
    /// Previous Kani harness count (from operational_state.json).
    pub prev_kani_harnesses: Option<u64>,
    /// Current Kani harness count.
    pub current_kani_harnesses: Option<u64>,
    /// Delta in Kani harnesses (current - previous). Positive means growth.
    pub kani_delta: Option<i64>,
    /// Previous total sound count across all models.
    pub prev_total_sound: Option<usize>,
    /// Current total sound count across all models.
    pub current_total_sound: Option<usize>,
    /// Delta in total sound count.
    pub sound_delta: Option<i64>,
}

/// Aggregated verification status report across all models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusReport {
    /// Overall verification breakdown (all models combined).
    pub summary: VerificationBreakdown,
    /// Per-model breakdowns.
    pub models: Vec<ModelSummary>,
    /// Kani harness count from the codebase, if computed.
    pub kani_count: Option<u64>,
    /// Gap detector summary, if computed.
    pub gap_summary: Option<GapSummary>,
    /// Trend comparison, if operational state was loaded.
    pub trend: Option<Trend>,
}

impl StatusReport {
    /// Load all per-model status files from `workspace_root` and build a report.
    ///
    /// Reads each `nn_verify_status_{model}.json` that exists in
    /// `workspace_root`. Models with no status file are silently skipped.
    ///
    /// This does not compute Kani counts, gap reports, or trends. Use the
    /// builder methods [`with_kani_count`], [`with_gap_summary`], and
    /// [`with_trend`] to add those.
    pub fn from_status_files(workspace_root: &Path) -> Result<Self, VerifyError> {
        let mut all_entries: Vec<&KernelStatus> = Vec::new();
        let mut models = Vec::new();
        let mut all_statuses: Vec<(String, VerifyStatus)> = Vec::new();

        // Discover status files: use MODEL_CATEGORIES plus any extra
        // nn_verify_status_*.json files in the directory.
        let mut model_names: Vec<String> =
            MODEL_CATEGORIES.iter().map(|s| (*s).to_string()).collect();

        if let Ok(entries) = std::fs::read_dir(workspace_root) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if let Some(model) = name_str
                    .strip_prefix("nn_verify_status_")
                    .and_then(|s| s.strip_suffix(".json"))
                {
                    if !model_names.contains(&model.to_string()) {
                        model_names.push(model.to_string());
                    }
                }
            }
        }

        for model in &model_names {
            let path = model_status_path(workspace_root, model);
            if !path.exists() {
                continue;
            }
            let status = VerifyStatus::load(&path)?;
            all_statuses.push((model.clone(), status));
        }

        // Build per-model summaries. We need to collect references from
        // the loaded statuses, so iterate over the stored data.
        for (model, status) in &all_statuses {
            let entries: Vec<&KernelStatus> = status.kernels().values().collect();
            let breakdown = VerificationBreakdown::from_entries(&entries);
            if breakdown.total > 0 || breakdown.stale > 0 {
                models.push(ModelSummary {
                    model: model.clone(),
                    breakdown,
                });
            }
        }

        // Build overall summary from all entries across all models.
        for (_, status) in &all_statuses {
            for entry in status.kernels().values() {
                all_entries.push(entry);
            }
        }
        let summary = VerificationBreakdown::from_entries(&all_entries);

        Ok(Self {
            summary,
            models,
            kani_count: None,
            gap_summary: None,
            trend: None,
        })
    }

    /// Build a report from a single `VerifyStatus` (useful for testing).
    #[must_use]
    pub fn from_verify_status(model_name: &str, status: &VerifyStatus) -> Self {
        let entries: Vec<&KernelStatus> = status.kernels().values().collect();
        let breakdown = VerificationBreakdown::from_entries(&entries);
        let summary = breakdown.clone();
        let models = vec![ModelSummary {
            model: model_name.to_string(),
            breakdown,
        }];
        Self {
            summary,
            models,
            kani_count: None,
            gap_summary: None,
            trend: None,
        }
    }

    /// Set the Kani harness count.
    #[must_use]
    pub fn with_kani_count(mut self, count: u64) -> Self {
        self.kani_count = Some(count);
        self
    }

    /// Set the gap summary.
    #[must_use]
    pub fn with_gap_summary(mut self, gap: GapSummary) -> Self {
        self.gap_summary = Some(gap);
        self
    }

    /// Compute a trend by comparing this report against operational_state.json.
    ///
    /// Reads `operational_state.json` from `workspace_root` and extracts
    /// the previous `kani_harnesses` count and `kokoro_soundness.sound` for
    /// comparison against the current report values.
    pub fn with_trend(mut self, workspace_root: &Path) -> Self {
        let ops_path = workspace_root.join("operational_state.json");
        if let Ok(contents) = std::fs::read_to_string(&ops_path) {
            if let Ok(ops) = serde_json::from_str::<OperationalState>(&contents) {
                let mut trend = Trend::default();

                // Kani harness trend
                if let Some(prev) = ops.kani_harnesses() {
                    trend.prev_kani_harnesses = Some(prev);
                    if let Some(current) = self.kani_count {
                        trend.current_kani_harnesses = Some(current);
                        trend.kani_delta = Some(current as i64 - prev as i64);
                    }
                }

                // Sound count trend (sum from operational_state vs current report)
                let prev_sound = ops.total_sound_count();
                if prev_sound > 0 {
                    trend.prev_total_sound = Some(prev_sound);
                    trend.current_total_sound = Some(self.summary.sound);
                    trend.sound_delta = Some(self.summary.sound as i64 - prev_sound as i64);
                }

                self.trend = Some(trend);
            }
        }
        self
    }

    /// Total number of non-stale entries across all models.
    #[must_use]
    pub fn total_entries(&self) -> usize {
        self.summary.total
    }

    /// Total stale entries across all models.
    #[must_use]
    pub fn total_stale(&self) -> usize {
        self.summary.stale
    }

    /// Per-model summaries, sorted by model name.
    #[must_use]
    pub fn per_model_summary(&self) -> &[ModelSummary] {
        &self.models
    }

    /// Generate a human-readable text report.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        out.push_str("=== nn Verification Status Report ===\n\n");

        // Overall summary
        out.push_str(&format!(
            "Total entries: {} (+ {} stale)\n",
            self.summary.total, self.summary.stale
        ));
        out.push_str(&format!(
            "Soundness: {} sound, {} heuristic ({:.1}% sound)\n",
            self.summary.sound,
            self.summary.heuristic,
            self.summary.sound_fraction() * 100.0,
        ));
        out.push_str(&format!(
            "Proof strength: {} crown, {} ibp, {} mixed, {} heuristic, {} vacuous\n",
            self.summary.sound_crown,
            self.summary.sound_ibp,
            self.summary.sound_mixed,
            self.summary.heuristic_non_vacuous,
            self.summary.vacuous,
        ));

        if let Some(kani) = self.kani_count {
            out.push_str(&format!("Kani harnesses: {kani}\n"));
        }

        // Per-model breakdown
        if !self.models.is_empty() {
            out.push_str("\n--- Per-Model Breakdown ---\n");
            for m in &self.models {
                let b = &m.breakdown;
                out.push_str(&format!(
                    "\n{}: {} entries ({} sound, {} heuristic, {} vacuous, {} stale)\n",
                    m.model, b.total, b.sound, b.heuristic, b.vacuous, b.stale,
                ));
            }
        }

        // Gap summary
        if let Some(ref gap) = self.gap_summary {
            out.push_str(&format!(
                "\n--- Gap Summary ---\nStages checked: {}, Gaps: {}, Vacuous: {}\n",
                gap.stages_checked, gap.gaps, gap.vacuous,
            ));
        }

        // Trend
        if let Some(ref trend) = self.trend {
            out.push_str("\n--- Trend ---\n");
            if let Some(delta) = trend.kani_delta {
                let sign = if delta >= 0 { "+" } else { "" };
                out.push_str(&format!(
                    "Kani harnesses: {} -> {} ({sign}{delta})\n",
                    trend.prev_kani_harnesses.unwrap_or(0),
                    trend.current_kani_harnesses.unwrap_or(0),
                ));
            }
            if let Some(delta) = trend.sound_delta {
                let sign = if delta >= 0 { "+" } else { "" };
                out.push_str(&format!(
                    "Sound entries: {} -> {} ({sign}{delta})\n",
                    trend.prev_total_sound.unwrap_or(0),
                    trend.current_total_sound.unwrap_or(0),
                ));
            }
        }

        out
    }
}

impl fmt::Display for StatusReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_text())
    }
}

/// Minimal parser for `operational_state.json` — extracts only the fields
/// needed for trend comparison without requiring a full schema definition.
#[derive(Debug, Deserialize)]
struct OperationalState {
    #[serde(default)]
    verification_counts: Option<OperationalVerificationCounts>,
    #[serde(default)]
    kani_harnesses: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct OperationalVerificationCounts {
    #[serde(default)]
    kani_harnesses: Option<OperationalCountValue>,
    #[serde(default)]
    kokoro_soundness: Option<OperationalSoundness>,
    #[serde(default)]
    dpdf_proof_strength: Option<OperationalProofStrength>,
}

#[derive(Debug, Deserialize)]
struct OperationalCountValue {
    #[serde(default)]
    value: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct OperationalSoundness {
    #[serde(default)]
    sound: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct OperationalProofStrength {
    #[serde(default)]
    sound: Option<usize>,
}

impl OperationalState {
    fn kani_harnesses(&self) -> Option<u64> {
        // Try verification_counts.kani_harnesses.value first, then top-level
        if let Some(ref vc) = self.verification_counts {
            if let Some(ref kh) = vc.kani_harnesses {
                if let Some(v) = kh.value {
                    return Some(v);
                }
            }
        }
        // Top-level kani_harnesses (plain number)
        self.kani_harnesses
            .as_ref()
            .and_then(serde_json::Value::as_u64)
    }

    fn total_sound_count(&self) -> usize {
        let mut total = 0usize;
        if let Some(ref vc) = self.verification_counts {
            if let Some(ref ks) = vc.kokoro_soundness {
                total += ks.sound.unwrap_or(0);
            }
            if let Some(ref dp) = vc.dpdf_proof_strength {
                total += dp.sound.unwrap_or(0);
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soundness_compat::VerificationSoundnessMode;
    use crate::status::{InputBoundsRecord, OutputBoundsRecord, ParamInputRecord, VerifyOutcome};
    use crate::verify_types::PropMethod;

    fn make_entry(
        soundness: VerificationSoundnessMode,
        method: PropMethod,
        output_width: f32,
        stale: bool,
    ) -> KernelStatus {
        let mut ks = KernelStatus::new(
            VerifyOutcome::Verified,
            method,
            InputBoundsRecord {
                variable_inputs: vec![ParamInputRecord {
                    param_index: 0,
                    lower: -1.0,
                    upper: 1.0,
                }],
                constant_params: vec![],
                input_shape: Some(vec![1]),
                input_range: Some((-1.0, 1.0)),
            },
            OutputBoundsRecord {
                lower: -output_width / 2.0,
                upper: output_width / 2.0,
                tensor_lower: None,
                tensor_upper: None,
                shape: None,
                is_infeasible: false,
            },
            output_width,
            soundness,
        );
        ks.stale = stale;
        ks
    }

    #[test]
    fn test_breakdown_empty() {
        let b = VerificationBreakdown::from_entries(&[]);
        assert_eq!(b.total, 0);
        assert_eq!(b.sound, 0);
        assert_eq!(b.heuristic, 0);
        assert_eq!(b.stale, 0);
        assert!((b.sound_fraction() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_breakdown_mixed_entries() {
        let sound_ibp = make_entry(
            VerificationSoundnessMode::Sound,
            PropMethod::Ibp,
            2.0,
            false,
        );
        let sound_crown = make_entry(
            VerificationSoundnessMode::Sound,
            PropMethod::Crown,
            1.5,
            false,
        );
        let heuristic = make_entry(
            VerificationSoundnessMode::Heuristic,
            PropMethod::Ibp,
            5.0,
            false,
        );
        let vacuous = make_entry(
            VerificationSoundnessMode::Heuristic,
            PropMethod::Ibp,
            200.0,
            false,
        );
        let stale = make_entry(VerificationSoundnessMode::Sound, PropMethod::Ibp, 2.0, true);

        let entries: Vec<&KernelStatus> =
            vec![&sound_ibp, &sound_crown, &heuristic, &vacuous, &stale];
        let b = VerificationBreakdown::from_entries(&entries);

        assert_eq!(b.total, 4);
        assert_eq!(b.sound, 2);
        assert_eq!(b.heuristic, 2);
        assert_eq!(b.stale, 1);
        assert_eq!(b.sound_ibp, 1);
        assert_eq!(b.sound_crown, 1);
        assert_eq!(b.heuristic_non_vacuous, 1);
        assert_eq!(b.vacuous, 1);
        assert!((b.sound_fraction() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_report_from_verify_status() {
        let entry = make_entry(
            VerificationSoundnessMode::Sound,
            PropMethod::Ibp,
            2.0,
            false,
        );
        let entry2 = make_entry(
            VerificationSoundnessMode::Heuristic,
            PropMethod::Crown,
            3.0,
            false,
        );

        // Build a status with kernels via JSON roundtrip.
        let json = serde_json::json!({
            "kernels": {
                "test_kernel_1": serde_json::to_value(&entry).unwrap(),
                "test_kernel_2": serde_json::to_value(&entry2).unwrap(),
            }
        });
        let status: VerifyStatus = serde_json::from_value(json).expect("deserialize test status");

        let report = StatusReport::from_verify_status("test_model", &status);
        assert_eq!(report.total_entries(), 2);
        assert_eq!(report.summary.sound, 1);
        assert_eq!(report.summary.heuristic, 1);
        assert_eq!(report.models.len(), 1);
        assert_eq!(report.models[0].model, "test_model");
    }

    #[test]
    fn test_report_from_status_files_nonexistent_dir() {
        // Loading from a nonexistent directory should produce an empty report.
        let report = StatusReport::from_status_files(Path::new("/nonexistent/path/42"))
            .expect("should not error on missing dir");
        assert_eq!(report.total_entries(), 0);
        assert!(report.models.is_empty());
    }

    #[test]
    fn test_report_with_kani_count() {
        let report = StatusReport::from_status_files(Path::new("/nonexistent/path/42"))
            .expect("empty report")
            .with_kani_count(8072);
        assert_eq!(report.kani_count, Some(8072));
    }

    #[test]
    fn test_report_with_gap_summary() {
        let gap = GapSummary {
            stages_checked: 8,
            gaps: 0,
            vacuous: 2,
        };
        let report = StatusReport::from_status_files(Path::new("/nonexistent/path/42"))
            .expect("empty report")
            .with_gap_summary(gap.clone());
        assert_eq!(report.gap_summary, Some(gap));
    }

    #[test]
    fn test_report_text_output() {
        let entry = make_entry(
            VerificationSoundnessMode::Sound,
            PropMethod::Ibp,
            2.0,
            false,
        );
        let json = serde_json::json!({
            "kernels": {
                "test_k1": serde_json::to_value(&entry).unwrap(),
            }
        });
        let status: VerifyStatus = serde_json::from_value(json).expect("deserialize");
        let report = StatusReport::from_verify_status("test_model", &status)
            .with_kani_count(100)
            .with_gap_summary(GapSummary {
                stages_checked: 5,
                gaps: 1,
                vacuous: 0,
            });

        let text = report.to_text();
        assert!(text.contains("Total entries: 1"));
        assert!(text.contains("sound"));
        assert!(text.contains("Kani harnesses: 100"));
        assert!(text.contains("Gaps: 1"));
    }

    #[test]
    fn test_operational_state_parsing() {
        let json = serde_json::json!({
            "verification_counts": {
                "kani_harnesses": { "value": 8072 },
                "kokoro_soundness": { "sound": 33 },
                "dpdf_proof_strength": { "sound": 692 }
            }
        });
        let ops: OperationalState = serde_json::from_value(json).expect("parse ops");
        assert_eq!(ops.kani_harnesses(), Some(8072));
        assert_eq!(ops.total_sound_count(), 725);
    }

    #[test]
    fn test_operational_state_top_level_kani() {
        let json = serde_json::json!({
            "kani_harnesses": 500
        });
        let ops: OperationalState = serde_json::from_value(json).expect("parse ops");
        assert_eq!(ops.kani_harnesses(), Some(500));
    }

    #[test]
    fn test_trend_computation() {
        let trend = Trend {
            prev_kani_harnesses: Some(8000),
            current_kani_harnesses: Some(8072),
            kani_delta: Some(72),
            prev_total_sound: Some(700),
            current_total_sound: Some(725),
            sound_delta: Some(25),
        };
        assert_eq!(trend.kani_delta, Some(72));
        assert_eq!(trend.sound_delta, Some(25));
    }

    #[test]
    fn test_display_impl() {
        let entry = make_entry(
            VerificationSoundnessMode::Sound,
            PropMethod::Ibp,
            2.0,
            false,
        );
        let json = serde_json::json!({
            "kernels": {
                "k1": serde_json::to_value(&entry).unwrap(),
            }
        });
        let status: VerifyStatus = serde_json::from_value(json).expect("deserialize");
        let report = StatusReport::from_verify_status("m", &status);
        let display = format!("{report}");
        assert!(display.contains("nn Verification Status Report"));
    }

    #[test]
    fn test_model_summary_serialization_roundtrip() {
        let summary = ModelSummary {
            model: "kokoro".to_string(),
            breakdown: VerificationBreakdown {
                total: 41,
                sound: 33,
                heuristic: 8,
                stale: 10,
                sound_crown: 7,
                sound_ibp: 25,
                sound_mixed: 1,
                heuristic_non_vacuous: 8,
                vacuous: 0,
            },
        };
        let json = serde_json::to_string(&summary).expect("serialize");
        let roundtrip: ModelSummary = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(roundtrip, summary);
    }
}
