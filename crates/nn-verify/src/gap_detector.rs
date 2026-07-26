// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Automated bound propagation gap detector for the Kokoro pipeline.
//!
//! Compares a static registry of pipeline stages against `nn_verify_status_kokoro.json`
//! to find stages with no bounds, vacuous bounds, or missing CROWN coverage.
//!
//! Part of #2930 (Automated bound propagation gap detector).
//! Part of #2218 (Perfect Kokoro epic).

/// A stage in the CompiledKokoro pipeline that needs bound propagation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineStage {
    pub name: &'static str,
    /// Status file key prefix (e.g. "kokoro_production_bert_encoder").
    pub status_key: &'static str,
    /// True if this stage is a compiled GPU segment with a trace graph.
    pub is_compiled_segment: bool,
    /// True if this stage involves a CPU bridge or non-traced GPU ops.
    pub is_bridge: bool,
    /// Source file implementing this stage.
    pub source_file: &'static str,
    /// Known CPU bridges within this stage (empty for pure GPU stages).
    pub cpu_bridges: &'static [&'static str],
}

/// Complete registry of Kokoro pipeline stages.
///
/// Source: `compiled_kokoro_steps.rs` step methods + `compiled_kokoro_bridges.rs`.
pub fn kokoro_pipeline_stages() -> Vec<PipelineStage> {
    vec![
        // Compiled GPU segments (traced, have GraphNetwork verification)
        PipelineStage {
            name: "PlBert + bert_encoder",
            status_key: "kokoro_production_bert_encoder",
            is_compiled_segment: true,
            is_bridge: false,
            source_file: "compiled_kokoro_segments.rs (seg_plbert + seg_text via step_encode)",
            cpu_bridges: &[],
        },
        PipelineStage {
            name: "TextEncoder",
            status_key: "kokoro_production_text_encoder",
            is_compiled_segment: true,
            is_bridge: false,
            source_file: "compiled_kokoro_segments.rs (seg_text via step_encode)",
            cpu_bridges: &[],
        },
        PipelineStage {
            name: "ProsodyPredictor",
            status_key: "kokoro_production_prosody_predictor",
            is_compiled_segment: true,
            is_bridge: false,
            source_file: "compiled_kokoro_segments.rs (seg_prosody via step_predict_prosody)",
            cpu_bridges: &[],
        },
        PipelineStage {
            name: "F0EnergyPredictor",
            status_key: "kokoro_production_f0_predictor",
            is_compiled_segment: true,
            is_bridge: false,
            source_file: "compiled_kokoro_segments.rs (seg_f0 via step_predict_f0_energy)",
            cpu_bridges: &[],
        },
        PipelineStage {
            name: "Generator",
            status_key: "kokoro_production_generator",
            is_compiled_segment: true,
            is_bridge: false,
            source_file: "compiled_kokoro_segments.rs (seg_generator via step_generate)",
            cpu_bridges: &[],
        },
        // Bridge stages (non-compiled, potential verification gaps)
        PipelineStage {
            name: "length_regulate (sigmoid+sum+floor+clamp+repeat_interleave)",
            status_key: "kokoro_production_length_regulate",
            is_compiled_segment: false,
            is_bridge: true,
            source_file: "compiled_kokoro_steps.rs (step_regulate)",
            cpu_bridges: &[],
        },
        PipelineStage {
            name: "harmonic_source (SineGen + forward STFT)",
            status_key: "kokoro_production_harmonic_source",
            is_compiled_segment: false,
            is_bridge: true,
            source_file: "compiled_kokoro_bridges.rs (build_harmonic_source)",
            cpu_bridges: &[],
        },
        PipelineStage {
            name: "iSTFT (polar_to_rect + frequency-to-time)",
            status_key: "kokoro_production_istft",
            is_compiled_segment: false,
            is_bridge: true,
            source_file: "compiled_kokoro_bridges.rs:88 (gpu_istft)",
            cpu_bridges: &["istft_terminal_readback — compiled_kokoro_bridges.rs:137"],
        },
    ]
}

/// Result of checking one pipeline stage against verification status.
#[derive(Debug)]
pub struct StageGapResult {
    pub stage: PipelineStage,
    pub has_ibp_bounds: bool,
    pub has_crown_bounds: bool,
    pub has_analytical_bounds: bool,
    pub is_vacuous: bool,
    pub bound_width: Option<f64>,
    /// Proof strength from the status entry (e.g., "sound", "heuristic", "vacuous").
    pub proof_strength: Option<String>,
    /// Soundness mode from the status entry.
    pub soundness_mode: Option<String>,
    /// Whether a constructive proof certificate is available for this stage (#4315).
    ///
    /// `true` when the status entry has `"has_constructive_certificate": true`,
    /// indicating that a machine-checkable proof artifact (IBP recomputation data
    /// or Lean4 export) was generated during verification.
    pub has_constructive_certificate: bool,
}

impl StageGapResult {
    /// Whether this stage has any form of verified bounds.
    pub fn has_any_bounds(&self) -> bool {
        self.has_ibp_bounds || self.has_crown_bounds || self.has_analytical_bounds
    }

    /// Whether this stage is fully certified: has non-vacuous bounds AND a
    /// constructive proof certificate (#4315).
    pub fn is_fully_certified(&self) -> bool {
        self.has_any_bounds() && !self.is_vacuous && self.has_constructive_certificate
    }
}

/// Summary of all gap detection results.
#[derive(Debug)]
pub struct GapReport {
    pub stages: Vec<StageGapResult>,
    pub total_gaps: usize,
    pub vacuous_count: usize,
    /// Number of stages with constructive proof certificates (#4315).
    pub certified_count: usize,
}

/// Width threshold above which IBP bounds are considered vacuous.
/// ProsodyPredictor is 345.2 — anything wider than 1000 is meaningless.
const VACUOUS_WIDTH_THRESHOLD: f64 = 1000.0;

/// Returns `true` if `method` (case-insensitive, whitespace-trimmed) names a
/// CROWN-family verification method: CROWN, AlphaCROWN, Alpha-CROWN,
/// BetaCROWN, Beta-CROWN, Mixed_IBP_CROWN, Mixed-IBP-CROWN.
pub(crate) fn method_is_crown(method: &str) -> bool {
    matches!(
        method.trim().to_ascii_uppercase().as_str(),
        "CROWN"
            | "ALPHACROWN"
            | "ALPHA-CROWN"
            | "BETACROWN"
            | "BETA-CROWN"
            | "MIXED_IBP_CROWN"
            | "MIXED-IBP-CROWN"
    )
}

/// Extracted classification logic for a single pipeline stage.
///
/// This function captures the exact decision rules used inside `detect_gaps`
/// without requiring `serde_json::Value`, making it amenable to Kani
/// model-checking with symbolic inputs.
///
/// # Arguments
/// * `primary_valid` — whether the primary status entry has status "verified" or "bounds_computed"
/// * `crown_valid` — whether the `_crown` suffix entry has status "verified" or "bounds_computed"
/// * `primary_method` — the `method` field from the primary entry (empty string if absent)
/// * `crown_method` — the `method` field from the crown entry (empty string if absent)
/// * `width` — the `output_width` field from the primary entry (None if absent)
/// * `proof_strength` — the `proof_strength` field from the primary entry (None if absent)
///
/// Returns `(has_ibp, has_crown, has_analytical, is_vacuous, has_any_bounds)`.
pub(crate) fn classify_entry(
    primary_valid: bool,
    crown_valid: bool,
    primary_method: &str,
    crown_method: &str,
    width: Option<f64>,
    proof_strength: Option<&str>,
) -> (bool, bool, bool, bool, bool) {
    let has_ibp = (primary_valid && (primary_method == "IBP" || primary_method.is_empty()))
        || (crown_valid && (crown_method == "IBP" || crown_method.is_empty()));
    let has_crown = (crown_valid && method_is_crown(crown_method))
        || (primary_valid && method_is_crown(primary_method));
    let has_analytical = (primary_valid && primary_method == "ANALYTICAL")
        || (crown_valid && crown_method == "ANALYTICAL");

    let has_vacuous_label = proof_strength == Some("vacuous");
    let is_vacuous = has_vacuous_label || width.map_or(false, |w| w > VACUOUS_WIDTH_THRESHOLD);

    let has_any_bounds = primary_valid || crown_valid;

    (
        has_ibp,
        has_crown,
        has_analytical,
        is_vacuous,
        has_any_bounds,
    )
}

/// Compute gap and vacuous counts from a slice of gap results.
///
/// Extracted from `detect_gaps` to enable Kani proof that the counts
/// match the per-entry predicates.
#[cfg_attr(not(kani), allow(dead_code))]
pub(crate) fn count_gaps_and_vacuous(results: &[StageGapResult]) -> (usize, usize) {
    let total_gaps = results.iter().filter(|r| !r.has_any_bounds()).count();
    let vacuous_count = results.iter().filter(|r| r.is_vacuous).count();
    (total_gaps, vacuous_count)
}

/// Check all pipeline stages against the verification status file.
///
/// The status file stores all entries under the `"kernels"` key.
/// Production segments use the `"kokoro_production_*"` prefix.
/// Entries may have `"status": "verified"` or `"status": "bounds_computed"`.
pub fn detect_gaps(status: &serde_json::Value) -> GapReport {
    let stages = kokoro_pipeline_stages();
    let segments = status.get("kernels").and_then(|s| s.as_object());

    let mut results = Vec::new();
    let mut total_gaps = 0;
    let mut vacuous_count = 0;
    let mut certified_count = 0;

    for stage in stages {
        let primary_key = stage.status_key;
        let crown_key = format!("{}_crown", stage.status_key);

        let primary_entry = segments.and_then(|s| s.get(primary_key));
        let crown_entry = segments.and_then(|s| s.get(&crown_key));

        let is_valid_entry = |e: Option<&serde_json::Value>| -> bool {
            e.and_then(|e| e.get("status"))
                .and_then(|s| s.as_str())
                .map_or(false, |s| s == "verified" || s == "bounds_computed")
        };

        // Check primary entry method — some entries (e.g., iSTFT) use CROWN directly.
        let primary_method = primary_entry
            .and_then(|e| e.get("method"))
            .and_then(|m| m.as_str())
            .unwrap_or("");
        let crown_method = crown_entry
            .and_then(|e| e.get("method"))
            .and_then(|m| m.as_str())
            .unwrap_or("");

        // Use primary entry for width (regardless of method).
        let width = primary_entry
            .and_then(|e| e.get("output_width"))
            .and_then(serde_json::Value::as_f64);

        let proof_strength = primary_entry
            .and_then(|e| e.get("proof_strength"))
            .and_then(|s| s.as_str())
            .map(String::from);

        let soundness_mode = primary_entry
            .and_then(|e| e.get("soundness_mode"))
            .and_then(|s| s.as_str())
            .map(String::from);

        // Check for constructive proof certificate presence (#4315).
        // Either the primary or crown entry having the flag counts.
        let has_constructive_certificate = primary_entry
            .and_then(|e| e.get("has_constructive_certificate"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
            || crown_entry
                .and_then(|e| e.get("has_constructive_certificate"))
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);

        // Delegate to classify_entry — the Kani-provable core logic.
        let (has_ibp, has_crown, has_analytical, is_vacuous, has_any_bounds) = classify_entry(
            is_valid_entry(primary_entry),
            is_valid_entry(crown_entry),
            primary_method,
            crown_method,
            width,
            proof_strength.as_deref(),
        );

        if !has_any_bounds {
            total_gaps += 1;
        }
        if is_vacuous {
            vacuous_count += 1;
        }
        if has_constructive_certificate {
            certified_count += 1;
        }

        results.push(StageGapResult {
            stage,
            has_ibp_bounds: has_ibp,
            has_crown_bounds: has_crown,
            has_analytical_bounds: has_analytical,
            is_vacuous,
            bound_width: width,
            proof_strength,
            soundness_mode,
            has_constructive_certificate,
        });
    }

    GapReport {
        stages: results,
        total_gaps,
        vacuous_count,
        certified_count,
    }
}

/// Format a gap report as a human-readable string with file:line locations.
///
/// Each stage is listed with its verification status, bound width,
/// source file, and CPU bridge information.
pub fn format_gap_report(report: &GapReport) -> String {
    let mut out = String::new();
    out.push_str("=== Kokoro Pipeline Bound Propagation Gap Report ===\n\n");

    for result in &report.stages {
        let status = if !result.has_any_bounds() {
            "GAP"
        } else if result.is_vacuous {
            "VACUOUS"
        } else if result.has_crown_bounds {
            "CROWN"
        } else if result.has_analytical_bounds {
            "ANALYTICAL"
        } else {
            "IBP"
        };

        let emoji = match status {
            "GAP" => "[!!]",
            "VACUOUS" => "[~~]",
            "CROWN" | "ANALYTICAL" => "[OK]",
            _ => "[ok]",
        };

        out.push_str(&format!(
            "{emoji} {name} ({status})\n",
            name = result.stage.name,
        ));
        out.push_str(&format!("    source: {}\n", result.stage.source_file));
        out.push_str(&format!("    status_key: {}\n", result.stage.status_key));

        if let Some(w) = result.bound_width {
            out.push_str(&format!("    bound_width: {w:.4}\n"));
        }
        if let Some(ref ps) = result.proof_strength {
            out.push_str(&format!("    proof_strength: {ps}\n"));
        }
        if let Some(ref sm) = result.soundness_mode {
            out.push_str(&format!("    soundness_mode: {sm}\n"));
        }
        if result.has_constructive_certificate {
            out.push_str("    constructive_certificate: yes\n");
        }
        if !result.stage.cpu_bridges.is_empty() {
            out.push_str("    cpu_bridges:\n");
            for bridge in result.stage.cpu_bridges {
                out.push_str(&format!("      - {bridge}\n"));
            }
        }
        out.push('\n');
    }

    out.push_str(&format!(
        "Summary: {gaps} gaps, {vacuous} vacuous, {certified} certified, {total} total stages\n",
        gaps = report.total_gaps,
        vacuous = report.vacuous_count,
        certified = report.certified_count,
        total = report.stages.len(),
    ));

    let verified_count = report
        .stages
        .iter()
        .filter(|r| r.has_any_bounds() && !r.is_vacuous)
        .count();
    out.push_str(&format!(
        "Verified (non-vacuous): {verified_count}/{total}\n",
        total = report.stages.len(),
    ));

    let fully_certified_count = report
        .stages
        .iter()
        .filter(|r| r.is_fully_certified())
        .count();
    out.push_str(&format!(
        "Fully certified (non-vacuous + certificate): {fully_certified_count}/{total}\n",
        total = report.stages.len(),
    ));

    out
}

#[cfg(test)]
#[path = "gap_detector_tests.rs"]
mod tests;
