// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-step profiling types for `CompiledModel`.
//!
//! Provides [`StepProfile`] and [`ExecutionProfile`] for identifying
//! execution bottlenecks. Use [`CompiledModel::execute_dyn_profiled()`]
//! to get a profile alongside the output tensor.
//!
//! Part of #2257.

use std::fmt;

use nn_dsl::trace_compile::CompiledStep;

/// Timing and metadata for a single compiled step.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StepProfile {
    /// Step index in the compiled plan.
    pub step_idx: usize,
    /// Human-readable step name (e.g., "matmul_fused_add", "LstmSequence").
    pub step_name: String,
    /// Wall-clock time for this step in microseconds.
    ///
    /// Measured with GPU flush after each dispatch step, so this includes
    /// both GPU execution and flush overhead. Non-dispatch steps (passthrough,
    /// narrow_view, identity) have near-zero timing.
    pub wall_time_us: f64,
    /// Whether this step dispatches GPU work.
    pub is_gpu_dispatch: bool,
    /// Output buffer size in bytes for this step.
    ///
    /// Computed from `step_numels * elem_bytes` where elem_bytes depends on
    /// the step's scalar type (4 for F32, 2 for F16/BF16). Zero for steps
    /// that produce no output buffer (e.g., identity passthrough aliases).
    pub output_bytes: usize,
}

impl StepProfile {
    /// Construct a step profile.
    pub fn new(
        step_idx: usize,
        step_name: String,
        wall_time_us: f64,
        is_gpu_dispatch: bool,
        output_bytes: usize,
    ) -> Self {
        Self {
            step_idx,
            step_name,
            wall_time_us,
            is_gpu_dispatch,
            output_bytes,
        }
    }
}

/// Aggregate execution profile from a single model forward pass.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ExecutionProfile {
    /// Per-step profiles in execution order.
    pub steps: Vec<StepProfile>,
    /// Total wall-clock time in microseconds (sum of all steps).
    pub total_wall_time_us: f64,
}

impl ExecutionProfile {
    /// Construct from a list of step profiles.
    pub fn new(steps: Vec<StepProfile>) -> Self {
        let total_wall_time_us = steps.iter().map(|s| s.wall_time_us).sum();
        Self {
            steps,
            total_wall_time_us,
        }
    }

    /// Returns the top N slowest steps, sorted by wall time descending.
    pub fn slowest_steps(&self, n: usize) -> Vec<&StepProfile> {
        let mut sorted: Vec<&StepProfile> = self.steps.iter().collect();
        sorted.sort_by(|a, b| {
            b.wall_time_us
                .partial_cmp(&a.wall_time_us)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(n);
        sorted
    }

    /// Returns the fraction of total time spent in GPU dispatch steps.
    pub fn gpu_time_fraction(&self) -> f64 {
        if self.total_wall_time_us == 0.0 {
            return 0.0;
        }
        let gpu_us: f64 = self
            .steps
            .iter()
            .filter(|s| s.is_gpu_dispatch)
            .map(|s| s.wall_time_us)
            .sum();
        gpu_us / self.total_wall_time_us
    }

    /// Number of GPU dispatch steps.
    pub fn num_gpu_dispatches(&self) -> usize {
        self.steps.iter().filter(|s| s.is_gpu_dispatch).count()
    }
}

impl ExecutionProfile {
    /// Total output bytes across all steps.
    pub fn total_output_bytes(&self) -> usize {
        self.steps.iter().map(|s| s.output_bytes).sum()
    }
}

impl fmt::Display for ExecutionProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "ExecutionProfile: {:.1} ms total, {} steps ({} GPU dispatches, {:.0}% GPU)",
            self.total_wall_time_us / 1000.0,
            self.steps.len(),
            self.num_gpu_dispatches(),
            self.gpu_time_fraction() * 100.0,
        )?;
        writeln!(f, "  Top 10 slowest:")?;
        for sp in self.slowest_steps(10) {
            let mem = format_bytes(sp.output_bytes);
            writeln!(
                f,
                "    [{:3}] {:>8.1} us  {:>8}  {}{}",
                sp.step_idx,
                sp.wall_time_us,
                mem,
                sp.step_name,
                if sp.is_gpu_dispatch { " [GPU]" } else { "" },
            )?;
        }
        Ok(())
    }
}

/// Format byte count as human-readable string.
fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

/// Derive a human-readable name from a [`CompiledStep`].
pub(crate) fn step_name(step: &CompiledStep) -> String {
    match step {
        CompiledStep::InputForward => "input".into(),
        CompiledStep::IdentityPassthrough => "identity".into(),
        CompiledStep::Passthrough { op_name, .. } => op_name.clone(),
        CompiledStep::NarrowView { .. } => "narrow_view".into(),
        CompiledStep::ConstantValue { .. } => "constant".into(),
        CompiledStep::Dispatch { kernel, .. } => kernel.name().to_string(),
        CompiledStep::NativeOp { op, .. } => op.variant_name().to_string(),
        CompiledStep::RuntimeOp { op, .. } => {
            let debug_str = format!("{op:?}");
            debug_str
                .split_once([' ', '{'])
                .map(|(n, _)| n)
                .unwrap_or(&debug_str)
                .to_string()
        }
        _ => "unknown".into(),
    }
}

/// Whether a compiled step dispatches GPU work.
pub(crate) fn is_gpu_dispatch(step: &CompiledStep) -> bool {
    matches!(
        step,
        CompiledStep::Dispatch { .. }
            | CompiledStep::NativeOp { .. }
            | CompiledStep::RuntimeOp { .. }
    )
}
