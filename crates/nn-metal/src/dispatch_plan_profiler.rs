// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Dispatch plan profiling infrastructure for RTF optimization.
//!
//! Provides [`DispatchPlanProfiler`] for analyzing where time goes in a
//! compiled dispatch plan. Groups steps by category (matmul, conv,
//! elementwise, normalization, etc.) to identify optimization targets.
//!
//! Part of #4264.

use std::collections::BTreeMap;
use std::fmt;

use nn_dsl::trace_compile::CompiledPlan;

use crate::compiled_model::profile::{is_gpu_dispatch, step_name};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single profiled step within a dispatch plan.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub(crate) struct ProfileEntry {
    /// Index of this step in the compiled plan.
    pub(crate) step_index: usize,
    /// Human-readable name derived from the compiled step.
    pub(crate) step_name: String,
    /// Wall-clock duration for this step in microseconds.
    pub(crate) duration_us: f64,
    /// Output buffer size in bytes for this step.
    pub(crate) memory_bytes: usize,
}

/// Aggregate profile of a full dispatch plan execution.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub(crate) struct DispatchProfile {
    /// Per-step profile entries in execution order.
    pub(crate) entries: Vec<ProfileEntry>,
    /// Total wall-clock time in microseconds (sum of all entries).
    pub(crate) total_us: f64,
    /// Number of steps that dispatch GPU work.
    pub(crate) dispatch_count: usize,
}

/// Error type for profiling operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub(crate) enum ProfileError {
    /// The provided step timings length does not match the plan step count.
    #[error("timing count mismatch: plan has {plan_steps} steps but {timing_count} timings provided")]
    TimingCountMismatch {
        plan_steps: usize,
        timing_count: usize,
    },
}

// ---------------------------------------------------------------------------
// Profiler
// ---------------------------------------------------------------------------

/// Profiler that wraps a [`CompiledPlan`] and produces [`DispatchProfile`]s
/// from externally-measured per-step timings.
///
/// The profiler does not perform GPU execution itself — it takes timing
/// data collected by the caller (e.g., from `execute_dyn_profiled()`) and
/// organizes it into an analyzable [`DispatchProfile`].
#[derive(Debug)]
pub(crate) struct DispatchPlanProfiler<'a> {
    plan: &'a CompiledPlan,
}

impl<'a> DispatchPlanProfiler<'a> {
    /// Create a new profiler wrapping the given compiled plan.
    pub(crate) fn new(plan: &'a CompiledPlan) -> Self {
        Self { plan }
    }

    /// Build a [`DispatchProfile`] from per-step wall-clock timings (in
    /// microseconds) and per-step output byte counts.
    ///
    /// Both slices must have exactly `plan.steps.len()` entries.
    pub(crate) fn profile(
        &self,
        timings_us: &[f64],
        output_bytes: &[usize],
    ) -> Result<DispatchProfile, ProfileError> {
        let n = self.plan.steps.len();
        if timings_us.len() != n {
            return Err(ProfileError::TimingCountMismatch {
                plan_steps: n,
                timing_count: timings_us.len(),
            });
        }
        // Allow output_bytes to also mismatch in length — pad with 0.
        let entries: Vec<ProfileEntry> = self
            .plan
            .steps
            .iter()
            .enumerate()
            .map(|(i, step)| ProfileEntry {
                step_index: i,
                step_name: step_name(step),
                duration_us: timings_us[i],
                memory_bytes: output_bytes.get(i).copied().unwrap_or(0),
            })
            .collect();

        let total_us = entries.iter().map(|e| e.duration_us).sum();
        let dispatch_count = self
            .plan
            .steps
            .iter()
            .filter(|s| is_gpu_dispatch(s))
            .count();

        Ok(DispatchProfile {
            entries,
            total_us,
            dispatch_count,
        })
    }
}

// ---------------------------------------------------------------------------
// DispatchProfile analysis methods
// ---------------------------------------------------------------------------

impl DispatchProfile {
    /// Construct a profile directly from entries (for testing or external use).
    pub(crate) fn from_entries(entries: Vec<ProfileEntry>) -> Self {
        let total_us = entries.iter().map(|e| e.duration_us).sum();
        let dispatch_count = entries.len(); // caller-provided entries are assumed dispatches
        Self {
            entries,
            total_us,
            dispatch_count,
        }
    }

    /// Return the top N most expensive steps, sorted by duration descending.
    pub(crate) fn top_n(&self, n: usize) -> Vec<&ProfileEntry> {
        let mut sorted: Vec<&ProfileEntry> = self.entries.iter().collect();
        sorted.sort_by(|a, b| {
            b.duration_us
                .partial_cmp(&a.duration_us)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(n);
        sorted
    }

    /// Group total duration by step category.
    ///
    /// Categories are derived from step names: `matmul`, `conv`, `elementwise`,
    /// `normalization`, `lstm`, `attention`, `passthrough`, and `other`.
    pub(crate) fn by_category(&self) -> BTreeMap<String, f64> {
        let mut categories: BTreeMap<String, f64> = BTreeMap::new();
        for entry in &self.entries {
            let cat = categorize_step(&entry.step_name);
            *categories.entry(cat).or_insert(0.0) += entry.duration_us;
        }
        categories
    }

    /// Human-readable summary sorted by duration descending.
    pub(crate) fn summary(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "DispatchProfile: {:.1} ms total, {} steps ({} dispatches)\n",
            self.total_us / 1000.0,
            self.entries.len(),
            self.dispatch_count,
        ));

        // Category breakdown
        out.push_str("\n  By category:\n");
        let cats = self.by_category();
        let mut cat_vec: Vec<(&String, &f64)> = cats.iter().collect();
        cat_vec.sort_by(|a, b| {
            b.1.partial_cmp(a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (cat, us) in &cat_vec {
            let pct = if self.total_us > 0.0 {
                *us / self.total_us * 100.0
            } else {
                0.0
            };
            out.push_str(&format!("    {cat:<16} {us:>8.1} us  ({pct:>5.1}%)\n"));
        }

        // Top 10 individual steps
        out.push_str("\n  Top 10 slowest steps:\n");
        for entry in self.top_n(10) {
            let mem = format_bytes(entry.memory_bytes);
            out.push_str(&format!(
                "    [{:3}] {:>8.1} us  {:>8}  {}\n",
                entry.step_index, entry.duration_us, mem, entry.step_name,
            ));
        }

        out
    }

    /// Total output memory across all profiled steps.
    pub(crate) fn total_memory_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.memory_bytes).sum()
    }
}

impl fmt::Display for DispatchProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ---------------------------------------------------------------------------
// Category classification
// ---------------------------------------------------------------------------

/// Classify a step name into a high-level category for grouping.
pub(crate) fn categorize_step(name: &str) -> String {
    let lower = name.to_lowercase();

    if lower.contains("matmul") || lower.contains("gemm") || lower.contains("linear") {
        return "matmul".to_string();
    }
    if lower.contains("conv") {
        return "conv".to_string();
    }
    if lower.contains("lstm") {
        return "lstm".to_string();
    }
    if lower.contains("attention") || lower.contains("sdpa") {
        return "attention".to_string();
    }
    if lower.contains("norm") || lower.contains("layernorm") || lower.contains("rmsnorm") {
        return "normalization".to_string();
    }
    if lower.contains("softmax") || lower.contains("log_softmax") {
        return "softmax".to_string();
    }
    if lower.contains("embedding") || lower.contains("gather") {
        return "embedding".to_string();
    }
    if lower.contains("input")
        || lower.contains("identity")
        || lower.contains("reshape")
        || lower.contains("narrow_view")
        || lower.contains("passthrough")
        || lower.contains("squeeze")
        || lower.contains("unsqueeze")
        || lower.contains("constant")
    {
        return "passthrough".to_string();
    }
    // Element-wise ops: activations, basic arithmetic
    if lower.contains("snake")
        || lower.contains("relu")
        || lower.contains("gelu")
        || lower.contains("silu")
        || lower.contains("swiglu")
        || lower.contains("sigmoid")
        || lower.contains("tanh")
        || lower.contains("add")
        || lower.contains("sub")
        || lower.contains("mul")
        || lower.contains("div")
        || lower.contains("fused_")
        || lower.contains("elementwise")
    {
        return "elementwise".to_string();
    }

    "other".to_string()
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

// ---------------------------------------------------------------------------
// Convenience: build a profile from a CompiledPlan + NativeOpKind metadata
// ---------------------------------------------------------------------------

/// Estimate output bytes for a compiled step given its element count and
/// scalar type byte size.
pub(crate) fn estimate_step_bytes(numel: usize, elem_bytes: usize) -> usize {
    numel.saturating_mul(elem_bytes)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "dispatch_plan_profiler_tests.rs"]
mod tests;
