// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Structured optimization report for the progressive tightening loop.
//!
//! [`OptimizationReport`] assembles performance, bounds, parity, and
//! certificate data from across the crate ecosystem into a single JSON
//! document. Each section is a `serde_json::Value` to avoid circular
//! cross-crate type dependencies:
//!
//! - **performance**: [`PerformanceReport`](nn_dsl::PerformanceReport) from nn-dsl
//! - **bounds**: [`BoundAnalysisReport`](nn_verify::BoundAnalysisReport) from nn-verify
//! - **parity**: [`DivergenceReport`](nn_reftest::DivergenceReport) from nn-reftest
//! - **certificate**: [`Certificate`](nn_tts_verify::Certificate) from nn-tts-verify
//!
//! An LLM or human reads this report to decide the next optimization action.
//! The proofs are the behavioral contract — any change must satisfy proven bounds.

use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Full optimization report for one iteration of the progressive tightening loop.
///
/// Each field after `performance` is `Option` because not all sections are
/// available at every iteration. The first iteration (iteration 0) typically
/// has only performance data; bounds and parity fill in as verification runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OptimizationReport {
    /// Schema version for forward compatibility.
    pub version: u32,
    /// Iteration number (0 = initial PyTorch conversion).
    pub iteration: usize,
    /// Model name (e.g., "kokoro", "whisper").
    pub model_name: String,
    /// ISO 8601 timestamp of report generation.
    pub generated_at: String,
    /// Performance metrics (dispatch counts, memory, latency).
    pub performance: serde_json::Value,
    /// Bound analysis from NY (explosion points, recommendations).
    pub bounds: Option<serde_json::Value>,
    /// Parity comparison against PyTorch reference (per-layer accuracy).
    pub parity: Option<serde_json::Value>,
    /// TTS audio quality certificate (hard bounds + quality metrics).
    pub certificate: Option<serde_json::Value>,
    /// Fusion equivalence certificates for fused kernel pairs.
    pub fusion_certificates: Vec<serde_json::Value>,
    /// Flattened recommendation summaries from all sources.
    pub recommendations: Vec<String>,
    /// Behavioral contract status (bounds satisfied, violations, tightened).
    pub contract_status: Option<ContractStatus>,
}

/// Status of the behavioral contract after this iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ContractStatus {
    /// Whether all proven bounds from the contract are satisfied.
    pub all_bounds_satisfied: bool,
    /// List of bound violations (empty if all satisfied).
    pub violations: Vec<String>,
    /// Bounds that got tighter this iteration (ratchet improvements).
    pub tightened_bounds: Vec<String>,
}

/// Error type for optimization report I/O.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ReportError {
    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// File I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
}

impl OptimizationReport {
    /// Current schema version.
    pub const CURRENT_VERSION: u32 = 1;

    /// Create a new report with performance data only.
    ///
    /// Other sections default to `None`/empty. Use builder methods to add them.
    pub fn new(
        iteration: usize,
        model_name: impl Into<String>,
        performance: &nn_dsl::PerformanceReport,
    ) -> Result<Self, ReportError> {
        Ok(Self {
            version: Self::CURRENT_VERSION,
            iteration,
            model_name: model_name.into(),
            generated_at: performance.generated_at.clone(),
            performance: serde_json::to_value(performance)?,
            bounds: None,
            parity: None,
            certificate: None,
            fusion_certificates: Vec::new(),
            recommendations: Vec::new(),
            contract_status: None,
        })
    }

    /// Add bound analysis data.
    pub fn with_bounds(mut self, bounds: &serde_json::Value) -> Self {
        self.bounds = Some(bounds.clone());
        self
    }

    /// Add parity data.
    pub fn with_parity(mut self, parity: &serde_json::Value) -> Self {
        self.parity = Some(parity.clone());
        self
    }

    /// Add TTS certificate data.
    pub fn with_certificate(mut self, cert: &serde_json::Value) -> Self {
        self.certificate = Some(cert.clone());
        self
    }

    /// Add a fusion equivalence certificate.
    pub fn add_fusion_certificate(&mut self, cert: serde_json::Value) {
        self.fusion_certificates.push(cert);
    }

    /// Add a recommendation string.
    pub fn add_recommendation(&mut self, rec: impl Into<String>) {
        self.recommendations.push(rec.into());
    }

    /// Set the behavioral contract status.
    pub fn with_contract_status(mut self, status: ContractStatus) -> Self {
        self.contract_status = Some(status);
        self
    }

    /// Serialize to pretty-printed JSON.
    pub fn to_json(&self) -> Result<String, ReportError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Save to a JSON file.
    pub fn save(&self, path: &Path) -> Result<(), ReportError> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load from a JSON file.
    pub fn load(path: &Path) -> Result<Self, ReportError> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    /// Surface norm chain explosion warnings from the bounds section.
    ///
    /// Parses the JSON bounds data (if present) to find `NormChainExplosion`
    /// recommendations and adds human-readable summaries to the report's
    /// recommendation list. This enables detection without importing
    /// nn-verify types directly (bounds are `serde_json::Value`).
    ///
    /// Part of #2708.
    pub fn generate_bounds_recommendations(&mut self) {
        let bounds = match &self.bounds {
            Some(v) => v,
            None => return,
        };
        let recs = match bounds
            .get("recommendations")
            .and_then(serde_json::Value::as_array)
        {
            Some(arr) => arr,
            None => return,
        };
        for rec in recs {
            // TighteningRecommendation is tagged enum — NormChainExplosion
            // serializes as {"NormChainExplosion": {...}}.
            if let Some(chain) = rec.get("NormChainExplosion") {
                let depth = chain
                    .get("chain_depth")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                let expansion = chain
                    .get("total_expansion")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0);
                self.recommendations.push(format!(
                    "NORM_CHAIN_EXPLOSION: {depth}-layer norm chain with {expansion:.1}x \
                     total expansion — bounds drift exceeds threshold",
                ));
            }
            if let Some(prisk) = rec.get("PrecisionRisk") {
                let depth = prisk
                    .get("chained_norm_depth")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                let ratio = prisk
                    .get("precision_drift_ratio")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0);
                let per_layer = prisk
                    .get("drift_per_layer")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0);
                self.recommendations.push(format!(
                    "PRECISION_RISK: {depth}-layer norm chain, F32/F64 ratio={ratio:.4}, \
                     per-layer drift={per_layer:.6} — exceeds precision threshold",
                ));
            }
        }
    }

    /// Surface GPU flush budget warnings from the performance section (#2739).
    ///
    /// If `total_flushes` exceeds the budget (3) or `total_submits` exceeds
    /// the target (0), adds actionable recommendations.
    ///
    /// Reads from the JSON performance value directly so that this method
    /// compiles regardless of whether `PerformanceReport` has been extended
    /// with flush/submit fields yet (cross-crate atomic change safety).
    pub fn generate_flush_recommendations(&mut self) {
        let total_flushes = self
            .performance
            .get("total_flushes")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let total_submits = self
            .performance
            .get("total_submits")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        // Only generate if stats were actually measured (non-zero).
        if total_flushes == 0 && total_submits == 0 {
            return;
        }
        if total_flushes > 3 {
            self.recommendations.push(format!(
                "FLUSH_BUDGET_EXCEEDED: {total_flushes} GPU flushes (budget: ≤3). \
                 Check for: scope exits, unexpected check_output_finite flushes, \
                 or to_device(&Cpu) in pipeline. See #2739.",
            ));
        }
        if total_submits > 0 {
            self.recommendations.push(format!(
                "SUBMIT_REGRESSION: {total_submits} non-blocking submits (target: 0). \
                 Segment fusion should eliminate all mid-pipeline submits. See #2739.",
            ));
        }
    }

    /// Generate dispatch-proportion recommendations from the performance section.
    ///
    /// Identifies segments that account for a disproportionate share of
    /// total dispatches and adds actionable recommendations.
    pub fn generate_dispatch_recommendations(&mut self) {
        let perf: nn_dsl::PerformanceReport =
            match serde_json::from_value(self.performance.clone()) {
                Ok(p) => p,
                Err(_) => return,
            };
        let total = perf.total_dispatches;
        if total == 0 {
            return;
        }
        for seg in &perf.segments {
            let pct = (seg.dispatches as f64 / total as f64) * 100.0;
            if pct > 40.0 {
                self.recommendations.push(format!(
                    "{} is {:.0}% of dispatches ({}/{}) — primary reduction target",
                    seg.name, pct, seg.dispatches, total,
                ));
            }
        }
    }
}

impl ContractStatus {
    /// Create a passing contract status with no violations.
    #[must_use]
    pub fn passing() -> Self {
        Self {
            all_bounds_satisfied: true,
            violations: Vec::new(),
            tightened_bounds: Vec::new(),
        }
    }
}

#[cfg(test)]
#[path = "optimization_report_tests.rs"]
mod tests;
