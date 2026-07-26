// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compiled plan diagnostics: summary and diff for regression investigation.
//!
//! `PlanSummary` captures the step-type breakdown, NativeOp variant
//! distribution, fusion stats, and buffer plan metrics for a compiled plan.
//! `PlanDiff` compares two summaries and highlights the differences.
//!
//! Part of #3348.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use nn_core::dyn_tensor::trace::ComputationGraph;

use super::trace_compile_types::{CompiledPlan, CompiledStep, FusionStats, PeepholeStats};

/// Buffer allocation metrics from the buffer planner.
///
/// Computed by running `plan_buffers` on the compiled plan. These metrics
/// are critical for diagnosing flush count regressions: changes in
/// `total_bytes` or `reuse_ratio` directly affect how GPU arenas overflow
/// and trigger command buffer commits.
#[derive(Clone, Debug)]
pub struct BufferPlanMetrics {
    /// Total bytes needed for the backing allocation (after reuse).
    pub total_bytes: usize,
    /// Sum of all individual buffer sizes (before reuse).
    pub naive_total: usize,
    /// Buffer reuse ratio: `1.0 - (total_bytes / naive_total)`. Higher = better.
    pub reuse_ratio: f64,
    /// Number of steps with dedicated buffer allocations.
    pub allocating_steps: usize,
    /// Largest single buffer allocation in bytes.
    pub max_buffer_bytes: usize,
}

/// Aggregate summary of a compiled execution plan.
///
/// Captures all diagnostically relevant metrics in one snapshot, suitable
/// for comparison across nn revisions or peephole pass configurations.
#[derive(Clone, Debug)]
pub struct PlanSummary {
    /// Total steps in the plan.
    pub total_steps: usize,
    /// Steps by variant: Dispatch, Passthrough, NarrowView, InputForward,
    /// IdentityPassthrough, ConstantValue, NativeOp, RuntimeOp.
    pub step_counts: BTreeMap<&'static str, usize>,
    /// NativeOp variant distribution (e.g., LstmSequence: 8, FusedResBlock: 4).
    pub native_op_variants: BTreeMap<&'static str, usize>,
    /// Elementwise chain fusion stats.
    pub fusion: FusionStats,
    /// Peephole NativeOp stats.
    pub peephole: PeepholeStats,
    /// Number of weight names referenced.
    pub weight_count: usize,
    /// Number of input tensors.
    pub input_count: usize,
    /// Buffer allocation metrics. `None` if computed without a graph.
    pub buffer_metrics: Option<BufferPlanMetrics>,
    /// Fusion gap analysis summary. `None` if not computed.
    /// Populated by [`PlanSummary::with_fusion_gap`].
    pub fusion_gap_summary: Option<String>,
}

/// Difference between two `PlanSummary` snapshots.
///
/// Positive deltas mean "new plan has more", negative means "new plan has fewer".
#[derive(Clone, Debug)]
pub struct PlanDiff {
    /// Step count delta (new - old).
    pub step_delta: i64,
    /// Per-variant step count deltas. Only non-zero entries included.
    pub step_count_deltas: BTreeMap<&'static str, i64>,
    /// NativeOp variant deltas. Only non-zero entries included.
    pub native_op_deltas: BTreeMap<&'static str, i64>,
    /// Dispatch savings delta (new - old).
    pub fusion_savings_delta: i64,
    /// NativeOp count delta (new - old).
    pub native_op_count_delta: i64,
    /// Weight count delta.
    pub weight_delta: i64,
    /// Buffer total_bytes delta. `None` if either summary lacks buffer metrics.
    pub buffer_bytes_delta: Option<i64>,
    /// Buffer reuse ratio delta. Positive = better reuse in new plan.
    pub buffer_reuse_delta: Option<f64>,
}

fn step_variant_name(step: &CompiledStep) -> &'static str {
    #[allow(unreachable_patterns)] // non_exhaustive: catch-all for future variants
    match step {
        CompiledStep::Dispatch { .. } => "Dispatch",
        CompiledStep::Passthrough { .. } => "Passthrough",
        CompiledStep::NarrowView { .. } => "NarrowView",
        CompiledStep::InputForward => "InputForward",
        CompiledStep::IdentityPassthrough => "IdentityPassthrough",
        CompiledStep::ConstantValue { .. } => "ConstantValue",
        CompiledStep::NativeOp { .. } => "NativeOp",
        CompiledStep::RuntimeOp { .. } => "RuntimeOp",
        // non_exhaustive catch-all — future variants show as Unknown
        _ => "Unknown",
    }
}

impl CompiledPlan {
    /// Build a diagnostic summary of this compiled plan.
    #[must_use]
    pub fn summary(&self) -> PlanSummary {
        let mut step_counts = BTreeMap::new();
        let mut native_op_variants = BTreeMap::new();

        for step in &self.steps {
            *step_counts.entry(step_variant_name(step)).or_insert(0usize) += 1;
            if let CompiledStep::NativeOp { op, .. } = step {
                *native_op_variants
                    .entry(op.variant_name())
                    .or_insert(0usize) += 1;
            }
        }

        PlanSummary {
            total_steps: self.steps.len(),
            step_counts,
            native_op_variants,
            fusion: self.fusion_stats(),
            peephole: self.peephole_stats(),
            weight_count: self.weight_names.len(),
            input_count: self.input_shapes.len(),
            buffer_metrics: None,
            fusion_gap_summary: None,
        }
    }

    /// Build a diagnostic summary including buffer allocation metrics.
    ///
    /// Runs the buffer planner to compute `total_bytes`, reuse ratio, and
    /// per-step allocation counts. Use this when comparing plans across
    /// revisions to diagnose flush count regressions (#3348).
    #[must_use]
    pub fn summary_with_graph(&self, graph: &ComputationGraph) -> PlanSummary {
        let mut summary = self.summary();
        let buffer_plan = crate::buffer_planner::plan_buffers(self, graph);
        let allocating_steps = buffer_plan
            .step_offsets
            .iter()
            .filter(|o| o.is_some())
            .count();
        let max_buffer_bytes = buffer_plan.step_sizes.iter().copied().max().unwrap_or(0);
        let reuse_ratio = if buffer_plan.naive_total > 0 {
            1.0 - (buffer_plan.total_bytes as f64 / buffer_plan.naive_total as f64)
        } else {
            0.0
        };
        summary.buffer_metrics = Some(BufferPlanMetrics {
            total_bytes: buffer_plan.total_bytes,
            naive_total: buffer_plan.naive_total,
            reuse_ratio,
            allocating_steps,
            max_buffer_bytes,
        });
        summary
    }
}

impl PlanSummary {
    /// Attach a fusion gap analysis summary to this plan summary.
    ///
    /// Consumes and returns `self` for builder-style chaining:
    /// ```ignore
    /// let summary = plan.summary().with_fusion_gap(&analysis);
    /// ```
    #[must_use]
    pub fn with_fusion_gap(
        mut self,
        analysis: &super::fusion_gap_analyzer::FusionGapAnalysis,
    ) -> Self {
        self.fusion_gap_summary = Some(analysis.summarize());
        self
    }

    /// Compare this summary (old) against another (new), producing a diff.
    #[must_use]
    pub fn diff(&self, new: &Self) -> PlanDiff {
        let step_delta = new.total_steps as i64 - self.total_steps as i64;

        let mut step_count_deltas = BTreeMap::new();
        let all_keys: BTreeSet<&str> = self
            .step_counts
            .keys()
            .chain(new.step_counts.keys())
            .copied()
            .collect();
        for &key in &all_keys {
            let old_val = self.step_counts.get(key).copied().unwrap_or(0) as i64;
            let new_val = new.step_counts.get(key).copied().unwrap_or(0) as i64;
            let delta = new_val - old_val;
            if delta != 0 {
                step_count_deltas.insert(key, delta);
            }
        }

        let mut native_op_deltas = BTreeMap::new();
        let all_native: BTreeSet<&str> = self
            .native_op_variants
            .keys()
            .chain(new.native_op_variants.keys())
            .copied()
            .collect();
        for &key in &all_native {
            let old_val = self.native_op_variants.get(key).copied().unwrap_or(0) as i64;
            let new_val = new.native_op_variants.get(key).copied().unwrap_or(0) as i64;
            let delta = new_val - old_val;
            if delta != 0 {
                native_op_deltas.insert(key, delta);
            }
        }

        let (buffer_bytes_delta, buffer_reuse_delta) =
            match (&self.buffer_metrics, &new.buffer_metrics) {
                (Some(old_buf), Some(new_buf)) => (
                    Some(new_buf.total_bytes as i64 - old_buf.total_bytes as i64),
                    Some(new_buf.reuse_ratio - old_buf.reuse_ratio),
                ),
                _ => (None, None),
            };

        PlanDiff {
            step_delta,
            step_count_deltas,
            native_op_deltas,
            fusion_savings_delta: new.fusion.dispatches_saved as i64
                - self.fusion.dispatches_saved as i64,
            native_op_count_delta: new.peephole.native_ops as i64 - self.peephole.native_ops as i64,
            weight_delta: new.weight_count as i64 - self.weight_count as i64,
            buffer_bytes_delta,
            buffer_reuse_delta,
        }
    }
}

impl fmt::Display for PlanSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Plan Summary ===")?;
        writeln!(f, "Total steps: {}", self.total_steps)?;
        writeln!(
            f,
            "Inputs: {}, Weights: {}",
            self.input_count, self.weight_count
        )?;
        writeln!(f)?;
        writeln!(f, "Step breakdown:")?;
        for (name, count) in &self.step_counts {
            writeln!(f, "  {name}: {count}")?;
        }
        if !self.native_op_variants.is_empty() {
            writeln!(f)?;
            writeln!(f, "NativeOp variants:")?;
            for (name, count) in &self.native_op_variants {
                writeln!(f, "  {name}: {count}")?;
            }
        }
        writeln!(f)?;
        writeln!(
            f,
            "Fusion: {} chains, {} ops fused, {} dispatches saved",
            self.fusion.fused_chains, self.fusion.fused_ops, self.fusion.dispatches_saved
        )?;
        writeln!(
            f,
            "Peephole: {} native ops, {} metal dispatches, {} passthroughs",
            self.peephole.native_ops,
            self.peephole.native_dispatches,
            self.peephole.passthrough_count
        )?;
        if let Some(buf) = &self.buffer_metrics {
            writeln!(f)?;
            writeln!(
                f,
                "Buffers: {} total bytes ({} naive), {:.1}% reuse",
                buf.total_bytes,
                buf.naive_total,
                buf.reuse_ratio * 100.0,
            )?;
            writeln!(
                f,
                "  {} allocating steps, {} max buffer bytes",
                buf.allocating_steps, buf.max_buffer_bytes,
            )?;
        }
        if let Some(gap_summary) = &self.fusion_gap_summary {
            writeln!(f)?;
            writeln!(f, "{gap_summary}")?;
        }
        Ok(())
    }
}

impl fmt::Display for PlanDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Plan Diff (new - old) ===")?;
        writeln!(f, "Steps: {:+}", self.step_delta)?;
        if !self.step_count_deltas.is_empty() {
            for (name, delta) in &self.step_count_deltas {
                writeln!(f, "  {name}: {delta:+}")?;
            }
        }
        if !self.native_op_deltas.is_empty() {
            writeln!(f, "NativeOp changes:")?;
            for (name, delta) in &self.native_op_deltas {
                writeln!(f, "  {name}: {delta:+}")?;
            }
        }
        writeln!(f, "Fusion savings: {:+}", self.fusion_savings_delta)?;
        writeln!(f, "Native ops: {:+}", self.native_op_count_delta)?;
        writeln!(f, "Weights: {:+}", self.weight_delta)?;
        if let Some(delta) = self.buffer_bytes_delta {
            writeln!(f, "Buffer bytes: {delta:+}")?;
        }
        if let Some(delta) = self.buffer_reuse_delta {
            writeln!(f, "Buffer reuse: {:+.1}%", delta * 100.0)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_plan_summary() {
        let plan = CompiledPlan {
            steps: vec![],
            input_shapes: vec![],
            output_step: 0,
            weight_names: vec![],
        };
        let summary = plan.summary();
        assert_eq!(summary.total_steps, 0);
        assert!(summary.step_counts.is_empty());
        assert!(summary.native_op_variants.is_empty());
    }

    #[test]
    fn test_summary_diff_zero_delta() {
        let plan = CompiledPlan {
            steps: vec![CompiledStep::InputForward],
            input_shapes: vec![vec![1, 3]],
            output_step: 0,
            weight_names: vec![],
        };
        let s = plan.summary();
        let diff = s.diff(&s);
        assert_eq!(diff.step_delta, 0);
        assert!(diff.step_count_deltas.is_empty());
        assert!(diff.buffer_bytes_delta.is_none());
    }

    #[test]
    fn test_summary_without_graph_has_no_buffer_metrics() {
        let plan = CompiledPlan {
            steps: vec![CompiledStep::InputForward],
            input_shapes: vec![vec![1, 3]],
            output_step: 0,
            weight_names: vec![],
        };
        let summary = plan.summary();
        assert!(summary.buffer_metrics.is_none());
    }

    #[test]
    fn test_display_roundtrip() {
        let plan = CompiledPlan {
            steps: vec![
                CompiledStep::InputForward,
                CompiledStep::IdentityPassthrough,
            ],
            input_shapes: vec![vec![1, 3]],
            output_step: 1,
            weight_names: vec!["w1".into()],
        };
        let summary = plan.summary();
        let output = format!("{summary}");
        assert!(output.contains("Total steps: 2"));
        assert!(output.contains("InputForward: 1"));
    }

    #[test]
    fn test_summary_default_no_fusion_gap() {
        let plan = CompiledPlan {
            steps: vec![CompiledStep::InputForward],
            input_shapes: vec![vec![1, 3]],
            output_step: 0,
            weight_names: vec![],
        };
        let summary = plan.summary();
        assert!(summary.fusion_gap_summary.is_none());
        let output = format!("{summary}");
        assert!(!output.contains("Fusion Gap Analysis"));
    }

    #[test]
    fn test_summary_with_fusion_gap() {
        use super::super::fusion_gap_analyzer::{FusionBlocker, FusionGap, FusionGapAnalysis};

        let plan = CompiledPlan {
            steps: vec![CompiledStep::InputForward],
            input_shapes: vec![vec![1, 3]],
            output_step: 0,
            weight_names: vec![],
        };
        let analysis = FusionGapAnalysis {
            gaps: vec![FusionGap {
                step_a: 0,
                step_b: 1,
                kernel_a: "relu".into(),
                kernel_b: "exp".into(),
                reason: FusionBlocker::FanOut,
                savings: 1,
            }],
            total_dispatches: 10,
            theoretical_minimum: 9,
        };
        let summary = plan.summary().with_fusion_gap(&analysis);
        assert!(summary.fusion_gap_summary.is_some());
        let output = format!("{summary}");
        assert!(output.contains("Fusion Gap Analysis"));
        assert!(output.contains("10 dispatches"));
        assert!(output.contains("theoretical min: 9"));
    }
}
