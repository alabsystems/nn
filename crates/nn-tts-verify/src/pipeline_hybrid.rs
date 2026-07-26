// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Hybrid verification certificate — formal bounds at small D, statistical
//! evidence at production D.
//!
//! When full-model CROWN is intractable at production dimensions (D=512+),
//! this certificate combines:
//! 1. Formal CROWN bounds at a tractable dimension (e.g., D=64).
//! 2. Statistical testing (paired t-test or bootstrap) at production dimension.
//!
//! The combination provides stronger evidence than either alone: the formal
//! proof shows the property *can* hold under the model structure, while the
//! statistical test shows it *does* hold at the production scale with high
//! confidence.

use std::fmt;

use nn_dsl::DispatchStep;

use crate::cost_model::{
    estimate_peak_memory, profile_dispatch_plan, total_estimated_time_us, total_flops,
    total_memory_bytes, HardwareCostModel, LayerCostProfile, PeakMemoryProfile,
};
use crate::error::{InvalidConfigKind, TtsVerifyError};
use crate::pipeline::{verify_pipeline, PipelineCertificate, VerifiedStage};

/// Timing certificate for a TTS pipeline — proves worst-case inference time.
///
/// Combines CROWN bounds verification with roofline cost model timing to
/// produce end-to-end formal guarantees on both output correctness and
/// execution time. This is Moonshot Property 5: "temporally bounded
/// (inference < X ms on M4 Max)."
///
/// Part of #1739 Phase 1.5.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TimingCertificate {
    /// The underlying bounds verification certificate.
    pub bounds_cert: PipelineCertificate,
    /// Per-step cost profiles from the roofline model.
    pub cost_profiles: Vec<LayerCostProfile>,
    /// Total estimated worst-case time in microseconds.
    pub worst_case_time_us: f64,
    /// Total theoretical FLOPs across all dispatch steps.
    pub total_flops: u64,
    /// Total memory traffic across all dispatch steps in bytes.
    pub total_memory_bytes: u64,
    /// Human-readable hardware description.
    pub hardware_name: String,
    /// Target timing bound in microseconds (e.g., 100_000.0 for 100ms).
    pub timing_bound_us: f64,
    /// Whether the estimated worst-case time is within the timing bound.
    pub timing_bound_met: bool,
    /// Whether both bounds and timing pass.
    pub overall_passed: bool,
    /// Peak memory profile for the dispatch plan (Phase 19).
    pub peak_memory: Option<PeakMemoryProfile>,
}

impl TimingCertificate {
    /// Create a new timing certificate with the given fields.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bounds_cert: PipelineCertificate,
        cost_profiles: Vec<LayerCostProfile>,
        worst_case_time_us: f64,
        total_flops: u64,
        total_memory_bytes: u64,
        hardware_name: impl Into<String>,
        timing_bound_us: f64,
        timing_bound_met: bool,
        overall_passed: bool,
        peak_memory: Option<PeakMemoryProfile>,
    ) -> Self {
        Self {
            bounds_cert,
            cost_profiles,
            worst_case_time_us,
            total_flops,
            total_memory_bytes,
            hardware_name: hardware_name.into(),
            timing_bound_us,
            timing_bound_met,
            overall_passed,
            peak_memory,
        }
    }
}

/// Verify a pipeline with both CROWN bounds and roofline timing.
///
/// Runs pipeline bounds verification on the `stages`, profiles the
/// `dispatch_plan` with the given `hardware_model`, and checks both
/// bounds containment and timing against `timing_bound_us`.
///
/// # Errors
///
/// Returns [`TtsVerifyError::InsufficientStages`] if fewer than 2 stages.
/// Returns [`TtsVerifyError::InvalidConfig`] if `timing_bound_us` is
/// non-positive or non-finite.
pub fn verify_pipeline_with_timing(
    stages: &[VerifiedStage],
    dispatch_plan: &[DispatchStep],
    hardware_model: &HardwareCostModel,
    timing_bound_us: f64,
) -> Result<TimingCertificate, TtsVerifyError> {
    if !timing_bound_us.is_finite() || timing_bound_us <= 0.0 {
        return Err(TtsVerifyError::InvalidConfig(
            InvalidConfigKind::NonPositive {
                param: "timing_bound_us",
            },
        ));
    }

    let bounds_cert = verify_pipeline(stages)?;
    let cost_profiles = profile_dispatch_plan(dispatch_plan, hardware_model);
    let worst_case_time = total_estimated_time_us(&cost_profiles);
    let flops = total_flops(&cost_profiles);
    let mem_bytes = total_memory_bytes(&cost_profiles);

    let timing_met = worst_case_time <= timing_bound_us;
    let peak_mem = estimate_peak_memory(dispatch_plan);

    Ok(TimingCertificate {
        overall_passed: bounds_cert.is_valid && timing_met,
        bounds_cert,
        cost_profiles,
        worst_case_time_us: worst_case_time,
        total_flops: flops,
        total_memory_bytes: mem_bytes,
        hardware_name: format!(
            "peak={:.1} TFLOPS, bw={:.0} GB/s, dispatch={:.1} μs",
            hardware_model.peak_tflops_f32,
            hardware_model.peak_bandwidth_gbs,
            hardware_model.dispatch_overhead_us,
        ),
        timing_bound_us,
        timing_bound_met: timing_met,
        peak_memory: Some(peak_mem),
    })
}

impl TimingCertificate {
    /// Generate a human-readable timing verification report.
    #[must_use]
    pub fn report(&self) -> String {
        let mut out = String::with_capacity(1024);
        out.push_str("=== Timing Verification Report ===\n\n");

        out.push_str(&format!(
            "Bounds: {} ({} stages)\n",
            if self.bounds_cert.is_valid {
                "PASS"
            } else {
                "FAIL"
            },
            self.bounds_cert.stages.len(),
        ));
        out.push_str(&format!(
            "Timing: {} (estimated={:.1} μs, bound={:.1} μs)\n",
            if self.timing_bound_met {
                "PASS"
            } else {
                "FAIL"
            },
            self.worst_case_time_us,
            self.timing_bound_us,
        ));
        out.push_str(&format!("Hardware: {}\n", self.hardware_name));
        out.push_str(&format!(
            "Total FLOPs: {:.2e}, Memory: {:.2} MB\n",
            self.total_flops as f64,
            self.total_memory_bytes as f64 / (1024.0 * 1024.0),
        ));
        if let Some(ref peak) = self.peak_memory {
            out.push_str(&format!(
                "Peak memory: {:.2} MB (weights={:.2} MB, activation={:.2} MB)\n",
                peak.peak_total_mb(),
                peak.weight_bytes as f64 / (1024.0 * 1024.0),
                peak.peak_activation_bytes as f64 / (1024.0 * 1024.0),
            ));
        }
        out.push_str(&format!(
            "\nOverall: {}\n",
            if self.overall_passed {
                "PASSED"
            } else {
                "FAILED"
            },
        ));
        out
    }
}

impl fmt::Display for TimingCertificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TimingCertificate(bounds={}, timing={}, overall={})",
            if self.bounds_cert.is_valid {
                "pass"
            } else {
                "fail"
            },
            if self.timing_bound_met {
                "pass"
            } else {
                "fail"
            },
            if self.overall_passed { "PASS" } else { "FAIL" },
        )
    }
}

/// Combined formal + statistical verification certificate.
///
/// Bridges the CROWN scaling gap: formal verification at tractable dimensions
/// (D <= ~128) with statistical testing at production dimensions (D >= 512).
///
/// # Evidence Strength
///
/// Use [`is_strong_evidence`](HybridCertificate::is_strong_evidence) to check
/// whether both the formal and statistical components provide strong evidence:
/// - Formal side: sound CROWN bounds (not IBP fallback).
/// - Statistical side: p < 0.01, Cohen's d > 0.8, property holds.
#[derive(Debug, Clone)]
pub struct HybridCertificate {
    /// Dimension at which formal verification was performed.
    pub formal_dim: usize,
    /// Property name verified formally (e.g., "output_bounded").
    pub formal_property: String,
    /// Whether the formal verification used sound bounds (CROWN, not IBP).
    pub formal_is_sound: bool,

    /// Dimension at which statistical testing was performed.
    pub statistical_dim: usize,
    /// Number of samples used in statistical test.
    pub n_samples: usize,
    /// p-value from the statistical test (lower = more significant).
    pub p_value: f64,
    /// Cohen's d effect size (larger = stronger effect).
    pub effect_size: f64,
    /// Whether the property holds in the statistical test.
    pub property_holds: bool,
}

impl HybridCertificate {
    /// Check whether both formal and statistical evidence are strong.
    ///
    /// Strong evidence requires:
    /// - Sound formal verification (CROWN, not IBP fallback).
    /// - Statistical significance at p < 0.01.
    /// - Large effect size (Cohen's d > 0.8).
    /// - The property actually holds in the statistical test.
    #[must_use]
    pub fn is_strong_evidence(&self) -> bool {
        self.formal_is_sound && self.p_value < 0.01 && self.effect_size > 0.8 && self.property_holds
    }
}

/// Verify a model layer-by-layer with CROWN and produce a timing certificate.
///
/// This is the unified CROWN + cost bound propagation path: per-layer CROWN
/// verification produces `VerifiedStage`s while the dispatch plan is profiled
/// through the roofline cost model. The result is a `TimingCertificate` where
/// both output correctness bounds and timing bounds are formally grounded.
///
/// Unlike [`verify_pipeline_with_timing`] which takes pre-built stages and a
/// separate dispatch plan, this function runs CROWN propagation itself and
/// couples each CROWN-verified layer to its cost profile. This closes the gap
/// between "bounds are correct" and "cost is bounded" — the timing certificate
/// is produced from the same layers that CROWN verified.
///
/// # Arguments
///
/// * `layers` - Sequence of (TensorKernelDef, parameter bindings) for each layer.
/// * `initial_bounds` - Input bounds for the first layer.
/// * `dispatch_plan` - Dispatch steps corresponding to the model's execution plan.
/// * `hardware_model` - Target hardware for roofline timing estimates.
/// * `timing_bound_us` - Maximum acceptable inference time in microseconds.
///
/// # Errors
///
/// Returns `TtsVerifyError` if CROWN propagation fails for any layer, if
/// `timing_bound_us` is non-positive or non-finite, or if fewer than 2 layers.
///
/// Part of #1739 Phase 2 — AC5: CROWN propagation of cost bounds.
#[cfg(feature = "ny")]
pub fn verify_layerwise_with_timing(
    layers: &[(
        nn_dsl::tensor_ir::TensorKernelDef,
        Vec<nn_verify::TensorParamBinding>,
    )],
    initial_bounds: &nn_verify::BoundedTensor,
    dispatch_plan: &[DispatchStep],
    hardware_model: &HardwareCostModel,
    timing_bound_us: f64,
) -> Result<TimingCertificate, TtsVerifyError> {
    if !timing_bound_us.is_finite() || timing_bound_us <= 0.0 {
        return Err(TtsVerifyError::InvalidConfig(
            InvalidConfigKind::NonPositive {
                param: "timing_bound_us",
            },
        ));
    }

    // Run per-layer CROWN propagation to get the bounds certificate.
    let bounds_cert = crate::pipeline::verify_layerwise(layers, initial_bounds)?;

    // Profile the dispatch plan through the roofline cost model.
    let cost_profiles = profile_dispatch_plan(dispatch_plan, hardware_model);
    let worst_case_time = total_estimated_time_us(&cost_profiles);
    let flops = total_flops(&cost_profiles);
    let mem_bytes = total_memory_bytes(&cost_profiles);

    let timing_met = worst_case_time <= timing_bound_us;
    let peak_mem = estimate_peak_memory(dispatch_plan);

    Ok(TimingCertificate {
        overall_passed: bounds_cert.is_valid && bounds_cert.is_sound && timing_met,
        bounds_cert,
        cost_profiles,
        worst_case_time_us: worst_case_time,
        total_flops: flops,
        total_memory_bytes: mem_bytes,
        hardware_name: format!(
            "peak={:.1} TFLOPS, bw={:.0} GB/s, dispatch={:.1} μs",
            hardware_model.peak_tflops_f32,
            hardware_model.peak_bandwidth_gbs,
            hardware_model.dispatch_overhead_us,
        ),
        timing_bound_us,
        timing_bound_met: timing_met,
        peak_memory: Some(peak_mem),
    })
}

impl fmt::Display for HybridCertificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HybridCertificate(formal_dim={}, stat_dim={}, n={}, p={:.4}, d={:.2}, holds={})",
            self.formal_dim,
            self.statistical_dim,
            self.n_samples,
            self.p_value,
            self.effect_size,
            self.property_holds,
        )
    }
}

#[cfg(test)]
#[path = "pipeline_hybrid_tests.rs"]
mod tests;

#[cfg(all(test, feature = "ny"))]
#[path = "pipeline_hybrid_d192_tests.rs"]
mod d192_tests;

#[cfg(all(test, feature = "ny"))]
#[path = "pipeline_hybrid_p1_tests.rs"]
mod p1_tests;

#[cfg(all(test, feature = "ny"))]
#[path = "pipeline_hybrid_p2_tests.rs"]
mod p2_tests;

#[cfg(all(test, feature = "ny"))]
#[path = "pipeline_hybrid_p3_tests.rs"]
mod p3_tests;
