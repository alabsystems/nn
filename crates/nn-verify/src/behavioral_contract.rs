// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Behavioral contract for verified model synthesis.
//!
//! A [`BehavioralContract`] captures the proven bounds from a verification pass
//! and serves as the invariant that any subsequent optimization iteration must
//! satisfy. The contract is the bridge between "prove once" and "optimize many":
//!
//! 1. Iteration 0: convert model, verify bounds, create contract.
//! 2. Iteration N: optimize, re-verify, [`validate`](BehavioralContract::validate)
//!    against contract.
//! 3. On success: [`tighten`](BehavioralContract::tighten) the contract (ratchet).
//!
//! Part of #2456, #2218.

use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::bound_analysis::BoundAnalysisReport;

/// A behavioral contract preserving proven bounds across optimization iterations.
///
/// Created from a [`BoundAnalysisReport`] after a successful verification pass.
/// Subsequent iterations validate against this contract, ensuring no regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BehavioralContract {
    /// Schema version for forward compatibility.
    pub version: u32,
    /// Model name (e.g., "kokoro", "whisper").
    pub model_name: String,
    /// Per-output interval bounds `(lower, upper)` from the final layer.
    pub output_bounds: Vec<(f32, f32)>,
    /// Named properties with proven bound values.
    pub properties: Vec<ContractProperty>,
    /// Maximum acceptable parity deviation (L3 threshold).
    pub max_parity_deviation: f32,
    /// ISO 8601 timestamp of contract creation.
    pub created_at: String,
    /// Iteration number that produced this contract.
    pub source_iteration: usize,
}

/// A named property with a proven bound value and threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ContractProperty {
    /// Property name (e.g., "non_silence", "non_clipping", "output_is_finite").
    pub name: String,
    /// Proven bound value.
    pub bound_value: f64,
    /// Threshold that must be satisfied.
    pub threshold: f64,
}

/// Result of validating a new analysis against the contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ContractValidation {
    /// Whether all contract bounds are satisfied.
    pub all_satisfied: bool,
    /// List of violated bounds.
    pub violations: Vec<String>,
    /// Bounds that got tighter (improvements).
    pub tightened: Vec<String>,
}

/// Error type for behavioral contract I/O.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ContractError {
    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// File I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
}

impl BehavioralContract {
    /// Current schema version.
    pub const CURRENT_VERSION: u32 = 1;

    /// Create a contract from a bound analysis report.
    ///
    /// Extracts output bounds and key properties (finiteness, explosion points)
    /// from the report. `parity_threshold` sets the maximum acceptable L3
    /// deviation for the contract.
    #[must_use]
    pub fn from_bound_analysis(
        report: &BoundAnalysisReport,
        parity_threshold: f32,
        iteration: usize,
    ) -> Self {
        let mut properties = Vec::new();

        // Record output finiteness as a property.
        properties.push(ContractProperty {
            name: "output_is_finite".into(),
            bound_value: if report.output_is_finite { 1.0 } else { 0.0 },
            threshold: 1.0,
        });

        // Record output width as a property (tighter = better).
        if report.output_width.is_finite() {
            properties.push(ContractProperty {
                name: "output_width".into(),
                bound_value: f64::from(report.output_width),
                threshold: f64::from(report.output_width),
            });
        }

        // Record CROWN coverage.
        properties.push(ContractProperty {
            name: "crown_coverage".into(),
            bound_value: f64::from(report.crown_coverage),
            threshold: f64::from(report.crown_coverage),
        });

        // Record explosion point count (fewer = better).
        properties.push(ContractProperty {
            name: "explosion_points".into(),
            bound_value: report.explosion_points.len() as f64,
            threshold: report.explosion_points.len() as f64,
        });

        // Record precision drift ratio (higher = better, closer to 1.0).
        if let Some(ratio) = report.precision_drift_ratio {
            if ratio.is_finite() {
                properties.push(ContractProperty {
                    name: "precision_drift_ratio".into(),
                    bound_value: f64::from(ratio),
                    threshold: f64::from(ratio),
                });
            }
        }

        Self {
            version: Self::CURRENT_VERSION,
            model_name: report.model_name.clone(),
            output_bounds: Vec::new(), // Filled by caller with per-element bounds
            properties,
            max_parity_deviation: parity_threshold,
            created_at: report.analyzed_at.clone(),
            source_iteration: iteration,
        }
    }

    /// Validate that a new analysis satisfies this contract.
    ///
    /// Checks each property bound. Returns a [`ContractValidation`] with
    /// violations and tightened bounds.
    #[must_use]
    pub fn validate(&self, report: &BoundAnalysisReport) -> ContractValidation {
        let mut violations = Vec::new();
        let mut tightened = Vec::new();

        self.check_finiteness(report, &mut violations);
        self.check_output_width(report, &mut violations, &mut tightened);
        self.check_explosion_points(report, &mut violations, &mut tightened);
        self.check_precision_drift(report, &mut violations, &mut tightened);

        ContractValidation {
            all_satisfied: violations.is_empty(),
            violations,
            tightened,
        }
    }

    fn check_finiteness(&self, report: &BoundAnalysisReport, violations: &mut Vec<String>) {
        if !report.output_is_finite {
            let had_finite = self
                .properties
                .iter()
                .any(|p| p.name == "output_is_finite" && p.bound_value >= 1.0);
            if had_finite {
                violations.push("output_is_finite: was finite, now non-finite".into());
            }
        }
    }

    fn check_output_width(
        &self,
        report: &BoundAnalysisReport,
        violations: &mut Vec<String>,
        tightened: &mut Vec<String>,
    ) {
        if let Some(contract_width) = self.properties.iter().find(|p| p.name == "output_width") {
            let threshold = contract_width.threshold as f32;
            if report.output_width.is_finite() && threshold.is_finite() {
                if report.output_width > threshold * 1.1 {
                    violations.push(format!(
                        "output_width: contract={:.2}, actual={:.2} (>{:.0}% regression)",
                        threshold,
                        report.output_width,
                        ((report.output_width / threshold) - 1.0) * 100.0,
                    ));
                } else if report.output_width < threshold * 0.99 {
                    tightened.push(format!(
                        "output_width: {:.2} -> {:.2}",
                        threshold, report.output_width,
                    ));
                }
            }
        }
    }

    fn check_explosion_points(
        &self,
        report: &BoundAnalysisReport,
        violations: &mut Vec<String>,
        tightened: &mut Vec<String>,
    ) {
        if let Some(contract_ep) = self
            .properties
            .iter()
            .find(|p| p.name == "explosion_points")
        {
            // Validate f64 threshold before cast: NaN saturates to 0, negative wraps.
            if !contract_ep.threshold.is_finite() || contract_ep.threshold < 0.0 {
                violations.push(format!(
                    "explosion_points: contract threshold {} is invalid",
                    contract_ep.threshold
                ));
                return;
            }
            let contract_count = contract_ep.threshold.round() as usize;
            let actual_count = report.explosion_points.len();
            if actual_count > contract_count {
                violations.push(format!(
                    "explosion_points: contract={contract_count}, actual={actual_count}",
                ));
            } else if actual_count < contract_count {
                tightened.push(format!(
                    "explosion_points: {contract_count} -> {actual_count}",
                ));
            }
        }
    }

    fn check_precision_drift(
        &self,
        report: &BoundAnalysisReport,
        violations: &mut Vec<String>,
        tightened: &mut Vec<String>,
    ) {
        if let Some(contract_pdr) = self
            .properties
            .iter()
            .find(|p| p.name == "precision_drift_ratio")
        {
            if let Some(actual_ratio) = report.precision_drift_ratio {
                let threshold = contract_pdr.threshold as f32;
                if actual_ratio.is_finite() && threshold.is_finite() {
                    if actual_ratio < threshold * 0.99 {
                        violations.push(format!(
                            "precision_drift_ratio: contract={threshold:.4}, actual={actual_ratio:.4} (regression)",
                        ));
                    } else if actual_ratio > threshold * 1.01 {
                        tightened.push(format!(
                            "precision_drift_ratio: {threshold:.4} -> {actual_ratio:.4}",
                        ));
                    }
                }
            }
        }
    }

    /// Create a tightened contract from a new analysis.
    ///
    /// Takes the tighter of (contract bound, new bound) for each property.
    /// This is the ratchet: once a bound improves, it can never regress.
    #[must_use]
    pub fn tighten(&self, report: &BoundAnalysisReport, new_iteration: usize) -> Self {
        let mut new_contract = self.clone();
        new_contract.source_iteration = new_iteration;
        new_contract.created_at = report.analyzed_at.clone();

        // Tighten output width.
        if let Some(prop) = new_contract
            .properties
            .iter_mut()
            .find(|p| p.name == "output_width")
        {
            if report.output_width.is_finite() && (f64::from(report.output_width)) < prop.threshold
            {
                prop.bound_value = f64::from(report.output_width);
                prop.threshold = f64::from(report.output_width);
            }
        }

        // Tighten explosion points.
        if let Some(prop) = new_contract
            .properties
            .iter_mut()
            .find(|p| p.name == "explosion_points")
        {
            let new_count = report.explosion_points.len() as f64;
            if new_count < prop.threshold {
                prop.bound_value = new_count;
                prop.threshold = new_count;
            }
        }

        // Tighten CROWN coverage (higher = better, so threshold goes up).
        if let Some(prop) = new_contract
            .properties
            .iter_mut()
            .find(|p| p.name == "crown_coverage")
        {
            let new_cov = f64::from(report.crown_coverage);
            if new_cov > prop.threshold {
                prop.bound_value = new_cov;
                prop.threshold = new_cov;
            }
        }

        // Tighten precision drift ratio (higher = better, closer to 1.0).
        if let Some(prop) = new_contract
            .properties
            .iter_mut()
            .find(|p| p.name == "precision_drift_ratio")
        {
            if let Some(new_ratio) = report.precision_drift_ratio {
                let new_val = f64::from(new_ratio);
                if new_val > prop.threshold {
                    prop.bound_value = new_val;
                    prop.threshold = new_val;
                }
            }
        }

        new_contract
    }

    /// Serialize to pretty-printed JSON.
    pub fn to_json(&self) -> Result<String, ContractError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Save to a JSON file.
    pub fn save(&self, path: &Path) -> Result<(), ContractError> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load from a JSON file.
    pub fn load(path: &Path) -> Result<Self, ContractError> {
        let data = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }
}

impl ContractProperty {
    /// Create a new contract property.
    #[must_use]
    pub fn new(name: impl Into<String>, bound_value: f64, threshold: f64) -> Self {
        Self {
            name: name.into(),
            bound_value,
            threshold,
        }
    }
}

impl ContractValidation {
    /// Create a passing validation with no violations.
    #[must_use]
    pub fn passing() -> Self {
        Self {
            all_satisfied: true,
            violations: Vec::new(),
            tightened: Vec::new(),
        }
    }
}

#[cfg(test)]
#[path = "behavioral_contract_tests.rs"]
mod tests;
