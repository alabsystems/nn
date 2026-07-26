// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Peak memory estimation for dispatch plans.
//!
//! Extracted from `cost_model.rs` to stay under 500-line limit.

use nn_dsl::DispatchStep;

use super::{step_name, step_output_bytes, step_weight_bytes};

/// Peak memory profile for a dispatch plan.
///
/// Tracks the high-water mark of simultaneously-allocated GPU/CPU buffers
/// during inference. The model assumes a sequential pipeline: at step `i`,
/// live activation memory = `output[i-1]` (input) + `output[i]` (output).
/// Weight memory persists for the model lifetime and is summed across all
/// steps. Peak total = weight_bytes + peak_activation_bytes.
///
/// Part of #1739 Phase 19.
#[derive(Debug, Clone)]
pub struct PeakMemoryProfile {
    /// Total weight memory in bytes (summed across all steps).
    pub weight_bytes: u64,
    /// Peak activation memory in bytes (high-water mark of live buffers).
    pub peak_activation_bytes: u64,
    /// Peak total memory = weight_bytes + peak_activation_bytes.
    pub peak_total_bytes: u64,
    /// Index of the step where peak activation memory occurs.
    pub peak_step_index: usize,
    /// Name of the step where peak activation memory occurs.
    pub peak_step_name: String,
    /// Per-step output buffer sizes in bytes.
    pub per_step_output_bytes: Vec<u64>,
}

impl PeakMemoryProfile {
    /// Peak total memory in megabytes.
    pub fn peak_total_mb(&self) -> f64 {
        self.peak_total_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Check if peak memory is within a byte-count bound.
    pub fn within_bound(&self, memory_bound_bytes: u64) -> bool {
        self.peak_total_bytes <= memory_bound_bytes
    }

    /// Generate a human-readable peak memory report.
    pub fn report(&self) -> String {
        let mut out = String::with_capacity(256);
        out.push_str("=== Peak Memory Profile ===\n\n");
        out.push_str(&format!(
            "Weight memory: {:.2} MB ({} bytes)\n",
            self.weight_bytes as f64 / (1024.0 * 1024.0),
            self.weight_bytes,
        ));
        out.push_str(&format!(
            "Peak activation: {:.2} MB ({} bytes)\n",
            self.peak_activation_bytes as f64 / (1024.0 * 1024.0),
            self.peak_activation_bytes,
        ));
        out.push_str(&format!(
            "Peak total: {:.2} MB ({} bytes)\n",
            self.peak_total_mb(),
            self.peak_total_bytes,
        ));
        out.push_str(&format!(
            "Peak step: #{} ({})\n",
            self.peak_step_index, self.peak_step_name,
        ));
        out
    }
}

/// Estimate peak memory usage for a dispatch plan.
///
/// Uses a sequential pipeline model:
/// - Weight memory: sum of all steps' weight bytes (persists for model lifetime).
/// - Activation memory: at each step, live = previous_output (input) + current_output.
/// - Peak activation: maximum live activation across all steps.
/// - Peak total: weight_bytes + peak_activation_bytes.
///
/// Reshape steps produce 0 output bytes (they alias the input buffer).
pub fn estimate_peak_memory(plan: &[DispatchStep]) -> PeakMemoryProfile {
    if plan.is_empty() {
        return PeakMemoryProfile {
            weight_bytes: 0,
            peak_activation_bytes: 0,
            peak_total_bytes: 0,
            peak_step_index: 0,
            peak_step_name: String::new(),
            per_step_output_bytes: vec![],
        };
    }

    let per_step_output: Vec<u64> = plan.iter().map(step_output_bytes).collect();
    let weight_total: u64 = plan.iter().map(step_weight_bytes).sum();

    let mut peak_activation: u64 = 0;
    let mut peak_index: usize = 0;
    let mut prev_output: u64 = 0;

    for (i, &cur_output) in per_step_output.iter().enumerate() {
        let live = prev_output + cur_output;
        if live > peak_activation {
            peak_activation = live;
            peak_index = i;
        }
        prev_output = cur_output;
    }

    PeakMemoryProfile {
        weight_bytes: weight_total,
        peak_activation_bytes: peak_activation,
        peak_total_bytes: weight_total + peak_activation,
        peak_step_index: peak_index,
        peak_step_name: step_name(&plan[peak_index]),
        per_step_output_bytes: per_step_output,
    }
}
