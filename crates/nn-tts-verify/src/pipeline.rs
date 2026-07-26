// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pipeline composition verification — end-to-end CROWN bounds for TTS.
//!
//! Verifies that individually-verified model stages compose correctly:
//! each stage's proven output bounds must fall within the next stage's
//! verified input bounds. This provides end-to-end guarantees without
//! building monolithic multi-million-node NY graphs.
//!
//! # Example
//!
//! ```text
//! let stages = vec![prosody_stage, decoder_stage];
//! let cert = verify_pipeline(&stages)?;
//! assert!(cert.is_valid);
//! ```

use std::fmt;

use crate::error::TtsVerifyError;

/// A verified stage in a pipeline with proven input/output bounds.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct VerifiedStage {
    /// Human-readable stage name (e.g., "kokoro_decoder", "silero_vad").
    pub name: String,
    /// Proven input bounds (per-element lower bounds).
    pub input_lower: Vec<f64>,
    /// Proven input bounds (per-element upper bounds).
    pub input_upper: Vec<f64>,
    /// Proven output bounds (per-element lower bounds).
    pub output_lower: Vec<f64>,
    /// Proven output bounds (per-element upper bounds).
    pub output_upper: Vec<f64>,
    /// Shape of the input tensor (for dimension compatibility check).
    pub input_shape: Vec<usize>,
    /// Shape of the output tensor.
    pub output_shape: Vec<usize>,
    /// Verification method used ("CROWN", "IBP", "alpha-CROWN").
    pub method: String,
    /// Whether this stage's verification is sound (not heuristic).
    pub is_sound: bool,
}

impl VerifiedStage {
    /// Create a new verified stage with the given bounds and metadata.
    pub fn new(
        name: impl Into<String>,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
        input_lower: Vec<f64>,
        input_upper: Vec<f64>,
        output_lower: Vec<f64>,
        output_upper: Vec<f64>,
        method: impl Into<String>,
        is_sound: bool,
    ) -> Self {
        Self {
            name: name.into(),
            input_lower,
            input_upper,
            output_lower,
            output_upper,
            input_shape,
            output_shape,
            method: method.into(),
            is_sound,
        }
    }
}

/// Result of checking compatibility between two adjacent stages.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct JunctionResult {
    /// Index of the junction (0 = between stage 0 and stage 1).
    pub junction_index: usize,
    /// Name of the output stage.
    pub from_stage: String,
    /// Name of the input stage.
    pub to_stage: String,
    /// Whether output shape is compatible with input shape.
    pub shape_compatible: bool,
    /// Per-element bound containment check.
    /// True if output_bounds ⊆ input_bounds for all elements.
    pub bounds_contained: bool,
    /// Maximum bound violation (0.0 if contained, positive otherwise).
    /// max(output_upper - input_upper, input_lower - output_lower) over all elements.
    pub max_violation: f64,
    /// Number of elements where bounds are violated.
    pub violation_count: usize,
}

/// Result of composing a pipeline of verified stages.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PipelineCertificate {
    /// The pipeline stages, in order.
    pub stages: Vec<VerifiedStage>,
    /// Per-junction compatibility results.
    pub junctions: Vec<JunctionResult>,
    /// End-to-end input bounds (from first stage).
    pub e2e_input_lower: Vec<f64>,
    pub e2e_input_upper: Vec<f64>,
    /// End-to-end output bounds (from last stage).
    pub e2e_output_lower: Vec<f64>,
    pub e2e_output_upper: Vec<f64>,
    /// Whether the entire pipeline composition is valid.
    pub is_valid: bool,
    /// Whether all stages used sound verification.
    pub is_sound: bool,
}

/// Compose a pipeline of verified stages and check end-to-end bounds.
///
/// Verifies that each stage's output bounds fall within the next stage's
/// input bounds. Returns a [`PipelineCertificate`] with per-junction results
/// and end-to-end bounds.
///
/// # Errors
///
/// Returns [`TtsVerifyError::InsufficientStages`] if fewer than 2 stages.
pub fn verify_pipeline(stages: &[VerifiedStage]) -> Result<PipelineCertificate, TtsVerifyError> {
    if stages.len() < 2 {
        return Err(TtsVerifyError::InsufficientStages {
            count: stages.len(),
        });
    }

    let mut junctions = Vec::with_capacity(stages.len() - 1);
    let mut all_contained = true;
    let mut all_sound = true;

    for i in 0..stages.len() - 1 {
        let junction = check_junction(&stages[i], &stages[i + 1], i);
        if !junction.bounds_contained || !junction.shape_compatible {
            all_contained = false;
        }
        if !stages[i].is_sound {
            all_sound = false;
        }
        junctions.push(junction);
    }
    // Check soundness of the last stage too.
    if let Some(last) = stages.last() {
        if !last.is_sound {
            all_sound = false;
        }
    }

    let first = &stages[0];
    let last = &stages[stages.len() - 1];

    Ok(PipelineCertificate {
        e2e_input_lower: first.input_lower.clone(),
        e2e_input_upper: first.input_upper.clone(),
        e2e_output_lower: last.output_lower.clone(),
        e2e_output_upper: last.output_upper.clone(),
        stages: stages.to_vec(),
        junctions,
        is_valid: all_contained,
        is_sound: all_sound && all_contained,
    })
}

/// Check compatibility between two adjacent stages.
///
/// Verifies:
/// 1. Output shape of `from` is compatible with input shape of `to`
///    (same total element count).
/// 2. Output bounds of `from` are contained within input bounds of `to`.
///    Element-wise: `from.output_lower >= to.input_lower` AND
///    `from.output_upper <= to.input_upper`.
///
/// NaN or non-finite values in bounds are treated as violations (defense-in-depth,
/// matching the IEEE 754 NaN guard pattern used throughout nn).
///
/// If the bounds vectors have different lengths, unmatched trailing elements are
/// counted as violations rather than silently skipped.
#[must_use]
pub fn check_junction(
    from: &VerifiedStage,
    to: &VerifiedStage,
    junction_index: usize,
) -> JunctionResult {
    let from_elements: usize = from.output_shape.iter().product();
    let to_elements: usize = to.input_shape.iter().product();
    let shape_compatible = from_elements == to_elements;

    let from_len = from.output_lower.len();
    let to_len = to.input_lower.len();
    let n = from_len.min(to_len);
    let mut max_violation = 0.0_f64;
    let mut violation_count = 0;

    for i in 0..n {
        let from_lo = from.output_lower[i];
        let from_hi = from.output_upper[i];
        let to_lo = to.input_lower[i];
        let to_hi = to.input_upper[i];

        // NaN/Inf guard: non-finite values in bounds are always violations.
        if !from_lo.is_finite() || !from_hi.is_finite() || !to_lo.is_finite() || !to_hi.is_finite()
        {
            violation_count += 1;
            // Use a large sentinel for max_violation when NaN is involved.
            max_violation = max_violation.max(f64::MAX);
            continue;
        }

        // Lower bound violation: from's output is below to's expected minimum.
        let lower_gap = to_lo - from_lo;
        // Upper bound violation: from's output exceeds to's expected maximum.
        let upper_gap = from_hi - to_hi;

        let violation = lower_gap.max(upper_gap).max(0.0);
        if violation > 0.0 {
            violation_count += 1;
            max_violation = max_violation.max(violation);
        }
    }

    // Unmatched trailing elements are counted as violations.
    let length_mismatch = from_len.abs_diff(to_len);
    violation_count += length_mismatch;
    if length_mismatch > 0 {
        // Length mismatch is a structural violation — use MAX sentinel.
        max_violation = max_violation.max(f64::MAX);
    }

    JunctionResult {
        junction_index,
        from_stage: from.name.clone(),
        to_stage: to.name.clone(),
        shape_compatible,
        bounds_contained: violation_count == 0,
        max_violation,
        violation_count,
    }
}

impl PipelineCertificate {
    /// Generate a human-readable verification report.
    #[must_use]
    pub fn report(&self) -> String {
        let mut out = String::with_capacity(512);

        out.push_str("=== Pipeline Verification Report ===\n\n");
        out.push_str(&format!("Stages: {}\n", self.stages.len()));
        out.push_str(&format!("Valid: {}\n", self.is_valid));
        out.push_str(&format!("Sound: {}\n\n", self.is_sound));

        for (i, stage) in self.stages.iter().enumerate() {
            out.push_str(&format!(
                "Stage {}: {} (method={}, sound={})\n",
                i, stage.name, stage.method, stage.is_sound,
            ));
            out.push_str(&format!(
                "  Input shape: {:?}, bounds: [{:.4}, {:.4}]\n",
                stage.input_shape,
                bounds_min(&stage.input_lower),
                bounds_max(&stage.input_upper),
            ));
            out.push_str(&format!(
                "  Output shape: {:?}, bounds: [{:.4}, {:.4}]\n",
                stage.output_shape,
                bounds_min(&stage.output_lower),
                bounds_max(&stage.output_upper),
            ));
        }

        out.push('\n');
        for junction in &self.junctions {
            out.push_str(&format!(
                "Junction {}: {} → {}\n",
                junction.junction_index, junction.from_stage, junction.to_stage,
            ));
            out.push_str(&format!(
                "  Shape compatible: {}\n",
                junction.shape_compatible,
            ));
            out.push_str(&format!(
                "  Bounds contained: {}\n",
                junction.bounds_contained,
            ));
            if junction.max_violation > 0.0 {
                out.push_str(&format!(
                    "  Max violation: {:.6} ({} elements)\n",
                    junction.max_violation, junction.violation_count,
                ));
            }
        }

        out.push_str(&format!(
            "\nEnd-to-end bounds:\n  Input: [{:.4}, {:.4}]\n  Output: [{:.4}, {:.4}]\n",
            bounds_min(&self.e2e_input_lower),
            bounds_max(&self.e2e_input_upper),
            bounds_min(&self.e2e_output_lower),
            bounds_max(&self.e2e_output_upper),
        ));

        out
    }
}

impl fmt::Display for PipelineCertificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PipelineCertificate({} stages, valid={}, sound={})",
            self.stages.len(),
            self.is_valid,
            self.is_sound,
        )
    }
}

#[cfg(feature = "ny")]
#[path = "pipeline_crown.rs"]
mod crown;
#[cfg(feature = "ny")]
pub use crown::{
    stage_from_bounds, stage_from_propagation, stage_from_propagation_with_soundness,
    verify_layerwise, verify_layerwise_from_graphs, verify_layerwise_grouped,
    verify_layerwise_mixed, GroupVerifyMode, LayerwiseGrouping,
};

#[path = "pipeline_hybrid.rs"]
mod hybrid;
#[cfg(feature = "ny")]
pub use hybrid::verify_layerwise_with_timing;
pub use hybrid::{verify_pipeline_with_timing, HybridCertificate, TimingCertificate};

/// Minimum value in a bounds vector (for report display).
/// Uses NaN-propagating fold to avoid IEEE 754 minNum silently discarding NaN.
fn bounds_min(v: &[f64]) -> f64 {
    crate::stats::fold_min_propagate_nan(v.iter().copied(), f64::INFINITY)
}

/// Maximum value in a bounds vector (for report display).
/// Uses NaN-propagating fold to avoid IEEE 754 maxNum silently discarding NaN.
fn bounds_max(v: &[f64]) -> f64 {
    crate::stats::fold_max_propagate_nan(v.iter().copied(), f64::NEG_INFINITY)
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod tests;
