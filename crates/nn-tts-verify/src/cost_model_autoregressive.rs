// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Autoregressive cost bounds for variable-length generation.
//!
//! Extracted from `cost_model.rs` to stay under 500-line limit.

use crate::error::{InvalidConfigKind, TtsVerifyError};
use nn_dsl::DispatchStep;

use super::{
    profile_dispatch_plan, total_estimated_time_us, total_flops, total_memory_bytes,
    HardwareCostModel, LayerCostProfile,
};

/// Worst-case cost bound for autoregressive (variable-length) generation.
///
/// For models like Qwen3-TTS where the number of decode steps depends on
/// input, the total cost is bounded by `max_steps × per_step_cost_upper`.
/// Each decode step has fixed per-step cost determined by the decoder
/// architecture (attention + FFN + KV cache).
///
/// The KV cache grows linearly with sequence length, so per-step memory
/// traffic increases. We model worst-case memory at `max_steps` (the last
/// step is the most expensive).
///
/// Part of #1739 D5.
#[derive(Debug, Clone)]
pub struct AutoregressiveCostBound {
    /// Maximum number of decode steps (tokens to generate).
    pub max_steps: usize,
    /// Dispatch plan for a SINGLE decode step at worst-case (max) KV length.
    pub per_step_plan: Vec<DispatchStep>,
    /// Per-step cost profile at worst-case KV length.
    pub per_step_profiles: Vec<LayerCostProfile>,
    /// Total worst-case time = max_steps × per_step_time_us.
    pub worst_case_total_us: f64,
    /// Total worst-case FLOPs = max_steps × per_step_flops.
    pub worst_case_total_flops: u64,
    /// Total worst-case memory = max_steps × per_step_memory_bytes.
    pub worst_case_total_memory_bytes: u64,
    /// Hardware model used for the estimate.
    pub hardware_name: String,
}

/// Compute worst-case timing for autoregressive generation.
///
/// Models variable-length decode loops (Qwen3-TTS, GPT-style):
/// - Each step runs the full decoder with increasing KV cache
/// - Worst-case cost = max_steps × per_step_cost(max_kv_length)
/// - The last step has the largest KV cache → highest per-step cost
///
/// # Arguments
///
/// * `per_step_plan` - Dispatch steps for a single decode step at max KV length.
/// * `max_steps` - Maximum number of tokens to generate.
/// * `hardware_model` - Target hardware for roofline timing.
///
/// # Errors
///
/// Returns `TtsVerifyError::InvalidConfig` if `max_steps` is 0.
pub fn bound_autoregressive_inference(
    per_step_plan: &[DispatchStep],
    max_steps: usize,
    hardware_model: &HardwareCostModel,
) -> Result<AutoregressiveCostBound, TtsVerifyError> {
    if max_steps == 0 {
        return Err(TtsVerifyError::InvalidConfig(
            InvalidConfigKind::NonPositive { param: "max_steps" },
        ));
    }

    let per_step_profiles = profile_dispatch_plan(per_step_plan, hardware_model);
    let per_step_time_us = total_estimated_time_us(&per_step_profiles);
    let per_step_flops = total_flops(&per_step_profiles);
    let per_step_memory = total_memory_bytes(&per_step_profiles);

    Ok(AutoregressiveCostBound {
        max_steps,
        per_step_plan: per_step_plan.to_vec(),
        per_step_profiles,
        worst_case_total_us: per_step_time_us * max_steps as f64,
        worst_case_total_flops: per_step_flops.saturating_mul(max_steps as u64),
        worst_case_total_memory_bytes: per_step_memory.saturating_mul(max_steps as u64),
        hardware_name: format!(
            "peak={:.1} TFLOPS, bw={:.0} GB/s, dispatch={:.1} μs",
            hardware_model.peak_tflops_f32,
            hardware_model.peak_bandwidth_gbs,
            hardware_model.dispatch_overhead_us,
        ),
    })
}

impl AutoregressiveCostBound {
    /// Per-step estimated time in microseconds.
    pub fn per_step_time_us(&self) -> f64 {
        if self.max_steps == 0 {
            return 0.0;
        }
        self.worst_case_total_us / self.max_steps as f64
    }

    /// Check if worst-case total is within a timing bound.
    pub fn within_bound(&self, timing_bound_us: f64) -> bool {
        self.worst_case_total_us <= timing_bound_us
    }

    /// Generate a human-readable report.
    pub fn report(&self) -> String {
        let mut out = String::with_capacity(512);
        out.push_str("=== Autoregressive Cost Bound ===\n\n");
        out.push_str(&format!("Max decode steps: {}\n", self.max_steps));
        out.push_str(&format!(
            "Per-step time: {:.1} μs ({:.3} ms)\n",
            self.per_step_time_us(),
            self.per_step_time_us() / 1000.0,
        ));
        out.push_str(&format!(
            "Worst-case total: {:.1} μs ({:.1} ms)\n",
            self.worst_case_total_us,
            self.worst_case_total_us / 1000.0,
        ));
        out.push_str(&format!(
            "Worst-case FLOPs: {:.2e}\n",
            self.worst_case_total_flops as f64,
        ));
        out.push_str(&format!(
            "Worst-case memory: {:.2} MB\n",
            self.worst_case_total_memory_bytes as f64 / (1024.0 * 1024.0),
        ));
        out.push_str(&format!("Hardware: {}\n", self.hardware_name));
        out.push_str(&format!(
            "Dispatch steps per decode step: {}\n",
            self.per_step_plan.len(),
        ));
        out
    }
}
