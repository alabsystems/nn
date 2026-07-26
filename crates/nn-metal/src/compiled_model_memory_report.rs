// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Memory report for compiled model execution plans.
//!
//! Analyzes a [`CompiledModelDef`] to produce a structured breakdown of
//! weight and intermediate buffer memory usage. Useful for understanding
//! GPU memory footprint before execution.
//!
//! # Usage
//!
//! ```rust,ignore
//! use nn_metal::compiled_model_memory_report::{generate_memory_report, format_memory_report};
//!
//! let report = generate_memory_report(&compiled_model);
//! println!("{}", format_memory_report(&report));
//! ```
//!
//! Part of #3828.

use std::fmt;

use nn_dsl::trace_compile::CompiledStep;

use crate::compiled_model::CompiledModelDef;

/// Per-step memory breakdown in a compiled model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepMemoryReport {
    /// Index of this step in the compiled plan.
    pub step_index: usize,
    /// Human-readable step name (variant + op detail).
    pub step_name: String,
    /// Bytes consumed by this step's input buffers (from buffer plan).
    pub input_bytes: usize,
    /// Bytes produced by this step's output buffer (from buffer plan).
    pub output_bytes: usize,
    /// Bytes of weight data bound to this step.
    pub weight_bytes: usize,
    /// Whether this step operates in-place (zero output allocation).
    pub is_in_place: bool,
}

/// Aggregated memory report for a compiled model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryReport {
    /// Total bytes occupied by all weight buffers.
    pub total_weight_bytes: usize,
    /// Total bytes of all intermediate (non-weight) buffers (naive sum).
    pub total_intermediate_bytes: usize,
    /// Peak intermediate bytes (from buffer planner's reuse analysis).
    pub peak_intermediate_bytes: usize,
    /// Number of distinct weight buffers.
    pub num_weight_buffers: usize,
    /// Number of intermediate buffer allocations.
    pub num_intermediate_buffers: usize,
    /// Per-step memory breakdown.
    pub per_step_breakdown: Vec<StepMemoryReport>,
}

/// Generate a memory report from a compiled model definition.
///
/// Walks the compiled steps and buffer plan to compute per-step and
/// aggregate memory statistics. Weight bytes are summed from the
/// pre-uploaded GPU weight buffers. Intermediate bytes come from the
/// buffer planner's `step_sizes` and `total_bytes`.
///
/// For external callers, use [`CompiledModel::memory_report()`] instead.
#[must_use]
pub(crate) fn generate_memory_report(def: &CompiledModelDef) -> MemoryReport {
    let num_steps = def.steps.len();
    let plan = &def.buffer_plan;

    // Accumulate weight bytes from the per-step weight buffer maps.
    let mut total_weight_bytes: usize = 0;
    let mut num_weight_buffers: usize = 0;
    for step_weights in &def.weight_buffers {
        for buf in step_weights.values() {
            total_weight_bytes = total_weight_bytes.saturating_add(buf.len());
            num_weight_buffers += 1;
        }
    }
    // Also count constant buffers as weights (they are uploaded once).
    for buf in def.constant_buffers.values() {
        total_weight_bytes = total_weight_bytes.saturating_add(buf.len());
        num_weight_buffers += 1;
    }

    let total_intermediate_bytes = plan.naive_total;
    let peak_intermediate_bytes = plan.total_bytes;

    let num_intermediate_buffers = plan
        .step_sizes
        .iter()
        .filter(|&&sz| sz > 0)
        .count();

    let mut per_step = Vec::with_capacity(num_steps);
    for i in 0..num_steps {
        let step = &def.steps[i];
        let output_bytes = if i < plan.step_sizes.len() {
            plan.step_sizes[i]
        } else {
            0
        };

        // Compute input bytes: sum of output_bytes of this step's edges.
        let input_bytes = if i < def.step_metas.len() {
            def.step_metas[i]
                .edges
                .iter()
                .map(|&src| {
                    if src < plan.step_sizes.len() {
                        plan.step_sizes[src]
                    } else {
                        0
                    }
                })
                .fold(0usize, usize::saturating_add)
        } else {
            0
        };

        // Weight bytes for this step.
        let weight_bytes = if i < def.weight_buffers.len() {
            def.weight_buffers[i]
                .values()
                .map(super::buffer::MetalBuffer::len)
                .fold(0usize, usize::saturating_add)
        } else {
            0
        };

        // In-place: output_bytes == 0 and step is not an InputForward.
        let is_in_place = output_bytes == 0
            && !matches!(step, CompiledStep::InputForward);

        per_step.push(StepMemoryReport {
            step_index: i,
            step_name: step_display_name(step),
            input_bytes,
            output_bytes,
            weight_bytes,
            is_in_place,
        });
    }

    MemoryReport {
        total_weight_bytes,
        total_intermediate_bytes,
        peak_intermediate_bytes,
        num_weight_buffers,
        num_intermediate_buffers,
        per_step_breakdown: per_step,
    }
}

/// Format a memory report as a human-readable string.
#[must_use]
pub fn format_memory_report(report: &MemoryReport) -> String {
    let mut out = String::new();
    out.push_str("=== Compiled Model Memory Report ===\n\n");
    out.push_str(&format!(
        "Weight memory:        {}\n",
        bytes_to_human(report.total_weight_bytes)
    ));
    out.push_str(&format!(
        "Weight buffers:       {}\n",
        report.num_weight_buffers
    ));
    out.push_str(&format!(
        "Intermediate (naive): {}\n",
        bytes_to_human(report.total_intermediate_bytes)
    ));
    out.push_str(&format!(
        "Intermediate (peak):  {}\n",
        bytes_to_human(report.peak_intermediate_bytes)
    ));
    out.push_str(&format!(
        "Intermediate buffers: {}\n",
        report.num_intermediate_buffers
    ));
    if report.total_intermediate_bytes > 0 {
        let savings_pct = if report.total_intermediate_bytes > report.peak_intermediate_bytes {
            let saved = report.total_intermediate_bytes - report.peak_intermediate_bytes;
            (saved as f64 / report.total_intermediate_bytes as f64) * 100.0
        } else {
            0.0
        };
        out.push_str(&format!("Buffer reuse savings:  {savings_pct:.1}%\n"));
    }

    if !report.per_step_breakdown.is_empty() {
        out.push_str("\n--- Per-Step Breakdown ---\n");
        out.push_str(&format!(
            "{:>5}  {:>10}  {:>10}  {:>10}  {:>8}  {}\n",
            "Step", "Input", "Output", "Weights", "InPlace", "Name"
        ));
        for step in &report.per_step_breakdown {
            out.push_str(&format!(
                "{:>5}  {:>10}  {:>10}  {:>10}  {:>8}  {}\n",
                step.step_index,
                bytes_to_human(step.input_bytes),
                bytes_to_human(step.output_bytes),
                bytes_to_human(step.weight_bytes),
                if step.is_in_place { "yes" } else { "no" },
                step.step_name,
            ));
        }
    }

    out
}

/// Convert a byte count to a human-readable string.
///
/// Uses binary units: KB (1024), MB (1024^2), GB (1024^3).
/// Values below 1 KB are shown as bytes. Fractional values
/// are displayed with one decimal place.
#[must_use]
pub fn bytes_to_human(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = 1024 * 1024;
    const GB: usize = 1024 * 1024 * 1024;

    if bytes == 0 {
        return "0 B".to_string();
    }
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Derive a human-readable display name for a compiled step.
fn step_display_name(step: &CompiledStep) -> String {
    #[allow(unreachable_patterns)] // non_exhaustive: catch-all for future variants
    match step {
        CompiledStep::Dispatch { kernel, .. } => {
            format!("Dispatch({})", kernel.name())
        }
        CompiledStep::Passthrough { op_name, .. } => {
            format!("Passthrough({op_name})")
        }
        CompiledStep::NarrowView { byte_offset, .. } => {
            format!("NarrowView(offset={byte_offset})")
        }
        CompiledStep::InputForward => "InputForward".to_string(),
        CompiledStep::IdentityPassthrough => "IdentityPassthrough".to_string(),
        CompiledStep::ConstantValue { value, shape, .. } => {
            format!("ConstantValue({value}, shape={shape:?})")
        }
        CompiledStep::NativeOp { op, .. } => {
            format!("NativeOp({})", op.variant_name())
        }
        CompiledStep::RuntimeOp { op } => {
            format!("RuntimeOp({op:?})")
        }
        _ => "Unknown".to_string(),
    }
}

impl fmt::Display for MemoryReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", format_memory_report(self))
    }
}

impl fmt::Display for StepMemoryReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Step {} ({}): input={}, output={}, weights={}, in_place={}",
            self.step_index,
            self.step_name,
            bytes_to_human(self.input_bytes),
            bytes_to_human(self.output_bytes),
            bytes_to_human(self.weight_bytes),
            self.is_in_place,
        )
    }
}

#[cfg(test)]
#[path = "compiled_model_memory_report_tests.rs"]
mod tests;
