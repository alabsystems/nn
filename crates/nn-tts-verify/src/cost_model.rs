// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Computational cost model for TTS inference — FLOP counting and roofline timing.
//!
//! Maps tensor operations (via [`DispatchStep`]) to estimated wall-clock time on
//! specific hardware targets using a roofline model. This is the foundation for
//! provable computational boundedness: proving worst-case inference time for TTS
//! pipelines.
//!
//! # Architecture
//!
//! 1. **FLOP counting** — each [`DispatchStep`] variant maps to a theoretical
//!    FLOP count via [`step_flops`].
//! 2. **Memory traffic** — each step's read/write byte count via [`step_memory_bytes`].
//! 3. **Roofline model** — [`HardwareCostModel`] maps (FLOPs, bytes) to estimated
//!    time via `max(compute_time, memory_time) + dispatch_overhead`.
//! 4. **Dispatch plan profiling** — [`profile_dispatch_plan`] produces per-layer
//!    cost profiles for a full model.
//!
//! Part of #1739.

#[path = "cost_model_ops.rs"]
mod ops;

pub use ops::{step_flops, step_memory_bytes, step_output_bytes, step_weight_bytes};

#[path = "cost_model_calibration.rs"]
mod calibration;

pub use calibration::{
    calibrate_profiles, fill_measured, CalibrationReport, Measurement, StepCalibration,
};

#[path = "cost_model_peak_memory.rs"]
mod peak_memory;
pub use peak_memory::{estimate_peak_memory, PeakMemoryProfile};

#[path = "cost_model_autoregressive.rs"]
mod autoregressive;
pub use autoregressive::{bound_autoregressive_inference, AutoregressiveCostBound};

use crate::error::{validate_finite_positive, TtsVerifyError};
use nn_dsl::DispatchStep;

/// Per-layer cost profile produced by the roofline model.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LayerCostProfile {
    /// Human-readable layer name (kernel name from dispatch step).
    pub layer_name: String,
    /// Theoretical floating-point operations.
    pub flops: u64,
    /// Estimated memory traffic in bytes (reads + writes).
    pub memory_bytes: u64,
    /// Estimated wall-clock time in microseconds from roofline model.
    pub estimated_time_us: f64,
    /// Measured wall-clock time (filled by actual GPU profiling, if available).
    pub measured_time_us: Option<f64>,
}

impl LayerCostProfile {
    /// Create a new layer cost profile.
    pub fn new(
        layer_name: impl Into<String>,
        flops: u64,
        memory_bytes: u64,
        estimated_time_us: f64,
        measured_time_us: Option<f64>,
    ) -> Self {
        Self {
            layer_name: layer_name.into(),
            flops,
            memory_bytes,
            estimated_time_us,
            measured_time_us,
        }
    }
}

/// Hardware cost model for roofline-based timing estimates.
///
/// The roofline model: `time = max(flops / peak_tflops, bytes / peak_bandwidth) + overhead`.
/// This provides a conservative upper bound on execution time.
#[derive(Debug, Clone)]
pub struct HardwareCostModel {
    /// Peak compute throughput in TFLOPS (fp32).
    pub peak_tflops_f32: f64,
    /// Peak memory bandwidth in GB/s.
    pub peak_bandwidth_gbs: f64,
    /// Per-kernel dispatch overhead in microseconds.
    pub dispatch_overhead_us: f64,
}

impl HardwareCostModel {
    /// Apple M4 Max hardware model.
    ///
    /// Sources:
    /// - Apple M4 Max: 14.2 TFLOPS fp32 (40-core GPU)
    /// - Unified memory bandwidth: ~400 GB/s (546 GB/s peak, ~75% sustained)
    /// - Metal dispatch overhead: ~5 μs measured (nn-metal benchmarks)
    pub fn m4_max() -> Self {
        Self {
            peak_tflops_f32: 14.2,
            peak_bandwidth_gbs: 400.0,
            dispatch_overhead_us: 5.0,
        }
    }

    /// Apple M4 Max hardware model with conservative correction factors.
    ///
    /// The theoretical roofline model (`m4_max()`) assumes peak hardware
    /// utilization. Real GPU execution achieves ~25% of peak compute throughput
    /// and ~50% of peak memory bandwidth due to:
    /// - Thread scheduling overhead and occupancy limits
    /// - Cache miss penalties (threadgroup memory → register spills)
    /// - Non-coalesced memory access patterns
    /// - Metal command buffer encoding and GPU idle gaps
    ///
    /// This model divides peak throughput by empirically-determined safety
    /// factors to produce estimates that are **guaranteed conservative**
    /// (estimated >= measured) while remaining non-vacuous (< 10x measured).
    ///
    /// Calibration data (M4 Max, simdgroup GEMM benchmark #1518 AC3):
    /// - FFN matmul [512,768]×[768,3072]: measured 784 μs, theoretical 175 μs
    /// - Ratio: 4.5x → use 5x compute factor (rounds up for safety)
    /// - Bandwidth: unified memory achieves ~50% of peak → use 2x factor
    /// - Dispatch overhead: measured ~5 μs but GPU idle gaps add ~5 μs → use 2x
    ///
    /// # Guarantee
    ///
    /// For any dispatch step s: `conservative.estimate_time_us(s) >= measured_time(s)`
    /// provided the GPU executes at no worse than 20% of peak throughput.
    pub fn m4_max_conservative() -> Self {
        Self {
            // 14.2 TFLOPS / 5.0 = 2.84 TFLOPS effective
            // Assumes worst-case 20% peak utilization (measured: 22% for FFN matmul)
            peak_tflops_f32: 14.2 / 5.0,
            // 400 GB/s / 2.0 = 200 GB/s effective
            // Assumes worst-case 50% bandwidth utilization (measured: ~55%)
            peak_bandwidth_gbs: 400.0 / 2.0,
            // 5 μs × 2.0 = 10 μs effective
            // Accounts for GPU idle gaps between dispatches
            dispatch_overhead_us: 5.0 * 2.0,
        }
    }

    /// Validate that all f64 fields are finite and positive.
    pub fn validate(&self) -> Result<(), TtsVerifyError> {
        validate_finite_positive(self.peak_tflops_f32, "peak_tflops_f32")?;
        validate_finite_positive(self.peak_bandwidth_gbs, "peak_bandwidth_gbs")?;
        validate_finite_positive(self.dispatch_overhead_us, "dispatch_overhead_us")?;
        Ok(())
    }

    /// Estimate execution time in microseconds using the roofline model.
    ///
    /// `time = max(compute_time, memory_time) + dispatch_overhead`
    ///
    /// where:
    /// - `compute_time = flops / (peak_tflops * 1e6)` (μs)
    /// - `memory_time = bytes / (peak_bandwidth * 1e3)` (μs)
    pub fn estimate_time_us(&self, flops: u64, memory_bytes: u64) -> f64 {
        // Convert TFLOPS to FLOPS/μs: 1 TFLOPS = 1e12 FLOPS/s = 1e6 FLOPS/μs
        let compute_time_us = flops as f64 / (self.peak_tflops_f32 * 1e6);
        // Convert GB/s to bytes/μs: 1 GB/s = 1e9 bytes/s = 1e3 bytes/μs
        let memory_time_us = memory_bytes as f64 / (self.peak_bandwidth_gbs * 1e3);
        f64::max(compute_time_us, memory_time_us) + self.dispatch_overhead_us
    }
}

/// Extract a human-readable name from a dispatch step.
pub fn step_name(step: &DispatchStep) -> String {
    match step {
        DispatchStep::Reduce { kernel_name, .. }
        | DispatchStep::Elementwise { kernel_name, .. }
        | DispatchStep::Broadcast { kernel_name, .. }
        | DispatchStep::Linear { kernel_name, .. }
        | DispatchStep::MatMul { kernel_name, .. }
        | DispatchStep::BinaryAdd { kernel_name, .. }
        | DispatchStep::BinaryMul { kernel_name, .. }
        | DispatchStep::Sigmoid { kernel_name, .. }
        | DispatchStep::Gelu { kernel_name, .. }
        | DispatchStep::Relu { kernel_name, .. }
        | DispatchStep::Tanh { kernel_name, .. }
        | DispatchStep::AxisSelect { kernel_name, .. }
        | DispatchStep::Stack { kernel_name, .. }
        | DispatchStep::Narrow { kernel_name, .. }
        | DispatchStep::Softmax { kernel_name, .. }
        | DispatchStep::ZeroPad1d { kernel_name, .. }
        | DispatchStep::Transpose { kernel_name, .. }
        | DispatchStep::Embedding { kernel_name, .. }
        | DispatchStep::Concat { kernel_name, .. } => kernel_name.clone(),
        DispatchStep::Conv1d(p) => p.kernel_name.clone(),
        DispatchStep::Conv2d(p) => p.kernel_name.clone(),
        DispatchStep::ConvTranspose1d(p) => p.kernel_name.clone(),
        DispatchStep::Reshape { .. } => "reshape".to_string(),
        _ => "unknown".to_string(),
    }
}

/// Profile a dispatch plan, producing per-step cost estimates.
pub fn profile_dispatch_plan(
    plan: &[DispatchStep],
    model: &HardwareCostModel,
) -> Vec<LayerCostProfile> {
    plan.iter()
        .map(|step| {
            let flops = step_flops(step);
            let memory_bytes = step_memory_bytes(step);
            let estimated_time_us = model.estimate_time_us(flops, memory_bytes);
            LayerCostProfile {
                layer_name: step_name(step),
                flops,
                memory_bytes,
                estimated_time_us,
                measured_time_us: None,
            }
        })
        .collect()
}

/// Total estimated time for a full model dispatch plan in microseconds.
pub fn total_estimated_time_us(profiles: &[LayerCostProfile]) -> f64 {
    profiles.iter().map(|p| p.estimated_time_us).sum()
}

/// Total FLOPs across all layers in a profile.
pub fn total_flops(profiles: &[LayerCostProfile]) -> u64 {
    profiles.iter().map(|p| p.flops).sum()
}

/// Total memory traffic across all layers in bytes.
pub fn total_memory_bytes(profiles: &[LayerCostProfile]) -> u64 {
    profiles.iter().map(|p| p.memory_bytes).sum()
}

#[cfg(test)]
#[path = "cost_model_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "cost_model_tests_autoregressive.rs"]
mod tests_autoregressive;

#[cfg(test)]
#[path = "cost_model_tests_peak_memory.rs"]
mod tests_peak_memory;

#[cfg(test)]
#[path = "cost_model_tests_peak_estimate.rs"]
mod tests_peak_estimate;

#[cfg(test)]
#[path = "cost_model_tests_profiling.rs"]
mod tests_profiling;
