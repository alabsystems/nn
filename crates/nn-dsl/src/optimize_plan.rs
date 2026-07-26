// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Exhaustive PeepholeConfig search for optimal dispatch count.
//!
//! Enumerates all 2^28 boolean toggle combinations of
//! [`PeepholeConfig`] and compiles each to find the configuration
//! that minimizes dispatch count (Dispatch + NativeOp steps).
//!
//! Part of Phase 4: Self-Optimizing ML Compiler (#3828).

use std::time::{Duration, Instant};

use nn_core::dyn_tensor::trace::ComputationGraph;

use crate::cost_model::CostModel;
use crate::tensor_ir::TensorIRError;

use super::{compile_trace_to_plan_configured, CompiledPlan, CompiledStep, PeepholeConfig};

/// Number of boolean fields in [`PeepholeConfig`].
pub(crate) const PEEPHOLE_FIELD_COUNT: u32 = 28;

/// Result of an exhaustive PeepholeConfig optimization search.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "plan-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OptimizationResult {
    /// The best plan found (lowest dispatch count).
    pub plan: CompiledPlan,
    /// The [`PeepholeConfig`] that produced the best plan.
    pub config: PeepholeConfig,
    /// Dispatch count of the best plan.
    pub dispatch_count: usize,
    /// Number of configurations explored.
    pub configs_explored: usize,
    /// Baseline dispatch count (all passes enabled, i.e., default config).
    pub baseline_dispatch_count: usize,
    /// Estimated cost of the best plan (nanoseconds).
    pub best_cost_ns: f64,
    /// Estimated cost of the baseline plan (nanoseconds).
    pub baseline_cost_ns: f64,
}

/// Result for a single segment in a multi-segment optimization.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "plan-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SegmentOptimizationResult {
    /// Name of the segment (e.g., model stage or layer group).
    pub segment_name: String,
    /// Optimization result for this segment.
    pub result: OptimizationResult,
}

impl OptimizationResult {
    /// Human-readable summary of the optimization result.
    #[must_use]
    pub fn summarize(&self) -> String {
        let total = 1u32 << PEEPHOLE_FIELD_COUNT;
        let improvement = if self.baseline_dispatch_count > 0 {
            let saved = self
                .baseline_dispatch_count
                .saturating_sub(self.dispatch_count);
            let pct = (saved as f64 / self.baseline_dispatch_count as f64) * 100.0;
            format!("{saved} fewer dispatches ({pct:.1}% reduction)")
        } else {
            "baseline is 0 dispatches".to_string()
        };

        let cost_info = if self.baseline_cost_ns > 0.0 {
            let cost_saved = self.baseline_cost_ns - self.best_cost_ns;
            let cost_pct = (cost_saved / self.baseline_cost_ns) * 100.0;
            format!(
                "\n- Baseline cost: {:.1} us\n\
                 - Best cost: {:.1} us\n\
                 - Cost reduction: {:.1}%",
                self.baseline_cost_ns / 1e3,
                self.best_cost_ns / 1e3,
                cost_pct,
            )
        } else {
            String::new()
        };

        format!(
            "Optimization result:\n\
             - Baseline dispatches (default config): {}\n\
             - Best dispatches: {}\n\
             - Improvement: {}\n\
             - Configs explored: {} / {}\n\
             - Best config: {:?}{cost_info}",
            self.baseline_dispatch_count,
            self.dispatch_count,
            improvement,
            self.configs_explored,
            total,
            self.config,
        )
    }
}

/// Enumerate all 2^28 [`PeepholeConfig`] toggle combinations *lazily*.
///
/// Each bit in a `u32` bitmask maps to one boolean field. Bit 0 corresponds
/// to `norm_activ_conv1d`, bit 18 to `fuse_conv1d_snake_norm`.
///
/// Returns a lazy iterator rather than a materialized `Vec`: collecting all
/// 2^28 = 268M configs would allocate ~13 GB. Callers that need a specific
/// config should use [`config_from_bitmask`] directly (O(1)); callers that
/// need to scan should iterate (and `.count()`/`.filter()`) without
/// collecting. The returned iterator is `Clone` (it wraps a function pointer).
#[must_use]
pub fn enumerate_peephole_configs() -> impl Iterator<Item = PeepholeConfig> + Clone {
    let total = 1u32 << PEEPHOLE_FIELD_COUNT;
    (0..total).map(config_from_bitmask)
}

/// Build a [`PeepholeConfig`] from a bitmask.
///
/// Bit assignment (matches struct field order):
/// - bit 0:  `norm_activ_conv1d`
/// - bit 1:  `fused_resblock`
/// - bit 2:  `linear_activation`
/// - bit 3:  `add_layer_norm`
/// - bit 4:  `norm_linear`
/// - bit 5:  `attention_transpose`
/// - bit 6:  `flip_lstm`
/// - bit 7:  `batched_linear_projection`
/// - bit 8:  `channels_first_layer_norm`
/// - bit 9:  `silu_mul`
/// - bit 10: `auto_fuse_elementwise`
/// - bit 11: `bilstm_cat`
/// - bit 12: `add_norm_linear`
/// - bit 13: `fuse_adain_snake`
/// - bit 14: `fuse_upsample_conv1d`
/// - bit 15: `fuse_instance_norm_mul_add`
/// - bit 16: `fuse_conv1d_activation`
/// - bit 17: `fuse_snake_instance_norm`
/// - bit 18: `fuse_conv1d_snake_norm`
/// - bit 19: `fuse_conv1d_snake_norm_resblock`
/// - bit 20: `fuse_add_instance_norm_conv1x1`
/// - bit 21: `fuse_conv_transpose1d_activation`
/// - bit 22: `norm_activ_conv_transpose1d`
/// - bit 23: `fuse_instance_norm_conv1d`
/// - bit 24: `fuse_conv1d_instance_norm`
/// - bit 25: `fuse_linear_layer_norm`
/// - bit 26: `fuse_resblock_chain`
/// - bit 27: `fuse_activation_conv1d`
#[must_use]
pub(crate) fn config_from_bitmask(mask: u32) -> PeepholeConfig {
    PeepholeConfig {
        norm_activ_conv1d: mask & (1 << 0) != 0,
        fused_resblock: mask & (1 << 1) != 0,
        linear_activation: mask & (1 << 2) != 0,
        add_layer_norm: mask & (1 << 3) != 0,
        norm_linear: mask & (1 << 4) != 0,
        attention_transpose: mask & (1 << 5) != 0,
        flip_lstm: mask & (1 << 6) != 0,
        batched_linear_projection: mask & (1 << 7) != 0,
        channels_first_layer_norm: mask & (1 << 8) != 0,
        silu_mul: mask & (1 << 9) != 0,
        auto_fuse_elementwise: mask & (1 << 10) != 0,
        bilstm_cat: mask & (1 << 11) != 0,
        add_norm_linear: mask & (1 << 12) != 0,
        fuse_adain_snake: mask & (1 << 13) != 0,
        fuse_upsample_conv1d: mask & (1 << 14) != 0,
        fuse_instance_norm_mul_add: mask & (1 << 15) != 0,
        fuse_conv1d_activation: mask & (1 << 16) != 0,
        fuse_snake_instance_norm: mask & (1 << 17) != 0,
        fuse_conv1d_snake_norm: mask & (1 << 18) != 0,
        fuse_conv1d_snake_norm_resblock: mask & (1 << 19) != 0,
        fuse_add_instance_norm_conv1x1: mask & (1 << 20) != 0,
        fuse_conv_transpose1d_activation: mask & (1 << 21) != 0,
        norm_activ_conv_transpose1d: mask & (1 << 22) != 0,
        fuse_instance_norm_conv1d: mask & (1 << 23) != 0,
        fuse_conv1d_instance_norm: mask & (1 << 24) != 0,
        fuse_linear_layer_norm: mask & (1 << 25) != 0,
        fuse_resblock_chain: mask & (1 << 26) != 0,
        fuse_activation_conv1d: mask & (1 << 27) != 0,
    }
}

/// Count dispatch steps (Dispatch + NativeOp) in a compiled plan.
///
/// These are the steps that translate to actual GPU kernel launches.
/// Passthrough, InputForward, IdentityPassthrough, ConstantValue,
/// NarrowView, and RuntimeOp steps are excluded.
#[must_use]
pub fn count_dispatches(plan: &CompiledPlan) -> usize {
    plan.steps
        .iter()
        .filter(|step| {
            matches!(
                step,
                CompiledStep::Dispatch { .. } | CompiledStep::NativeOp { .. }
            )
        })
        .count()
}

/// Exhaustively search PeepholeConfig space for the plan with fewest dispatches.
///
/// Compiles the graph with each of the 32768 toggle combinations and returns
/// the one with the lowest dispatch count. The search respects the given
/// `budget` duration — if time runs out, returns the best result found so far.
///
/// A zero-duration budget still compiles the baseline (default config).
///
/// # Errors
///
/// Returns `TensorIRError` if the baseline compilation fails. Compilation
/// failures for non-default configs are silently skipped (the config is
/// not viable).
pub fn optimize_plan(
    graph: &ComputationGraph,
    budget: Duration,
) -> Result<OptimizationResult, TensorIRError> {
    let start = Instant::now();

    // Always compile baseline first (default = all passes enabled).
    let default_config = PeepholeConfig::default();
    let baseline_plan = compile_trace_to_plan_configured(graph, &default_config)?;
    let baseline_dispatches = count_dispatches(&baseline_plan);

    // Use default cost model for cost estimates.
    let default_cost_model = CostModel::apple_m4();
    let baseline_cost_ns = default_cost_model.estimate(&baseline_plan).total_ns;

    let mut best_plan = baseline_plan;
    let mut best_config = default_config;
    let mut best_dispatches = baseline_dispatches;
    let mut best_cost_ns = baseline_cost_ns;
    let mut configs_explored: usize = 1;

    // Iterate the bitmask space lazily rather than materializing all 2^N
    // configs up front. At 28 fields the full enumeration is 2^28 entries
    // (~7.5 GB if collected into a Vec), so the search must be streamed and
    // bounded by the time budget.
    let total_masks = 1u32 << PEEPHOLE_FIELD_COUNT;
    for mask in 0..total_masks {
        // Check time budget before each compilation.
        if start.elapsed() >= budget {
            break;
        }

        let config = config_from_bitmask(mask);

        // Skip the default config — already compiled as baseline.
        if is_default_config(&config) {
            continue;
        }

        configs_explored += 1;

        // Compilation may fail for certain config combinations (e.g., if
        // disabling a pass creates an unsupported pattern). Skip failures.
        let plan = match compile_trace_to_plan_configured(graph, &config) {
            Ok(p) => p,
            Err(_) => continue,
        };

        let dispatches = count_dispatches(&plan);
        if dispatches < best_dispatches {
            best_dispatches = dispatches;
            best_cost_ns = default_cost_model.estimate(&plan).total_ns;
            best_plan = plan;
            best_config = config;
        }
    }

    Ok(OptimizationResult {
        plan: best_plan,
        config: best_config,
        dispatch_count: best_dispatches,
        configs_explored,
        baseline_dispatch_count: baseline_dispatches,
        best_cost_ns,
        baseline_cost_ns,
    })
}

/// Optimize plan considering both dispatch count and estimated cost.
///
/// Like [`optimize_plan()`], but uses the cost model to break ties when
/// two configs produce the same dispatch count. Among configs with equal
/// dispatch count, the one with the lowest estimated cost wins.
///
/// # Errors
///
/// Returns `TensorIRError` if the baseline compilation fails.
pub fn optimize_plan_with_cost(
    graph: &ComputationGraph,
    cost_model: &CostModel,
    budget: Duration,
) -> Result<OptimizationResult, TensorIRError> {
    let start = Instant::now();

    // Always compile baseline first (default = all passes enabled).
    let default_config = PeepholeConfig::default();
    let baseline_plan = compile_trace_to_plan_configured(graph, &default_config)?;
    let baseline_dispatches = count_dispatches(&baseline_plan);
    let baseline_cost_ns = cost_model.estimate(&baseline_plan).total_ns;

    let mut best_plan = baseline_plan;
    let mut best_config = default_config;
    let mut best_dispatches = baseline_dispatches;
    let mut best_cost_ns = baseline_cost_ns;
    let mut configs_explored: usize = 1;

    // Stream the bitmask space lazily (see `optimize_plan` for rationale).
    let total_masks = 1u32 << PEEPHOLE_FIELD_COUNT;
    for mask in 0..total_masks {
        if start.elapsed() >= budget {
            break;
        }

        let config = config_from_bitmask(mask);

        if is_default_config(&config) {
            continue;
        }

        configs_explored += 1;

        let plan = match compile_trace_to_plan_configured(graph, &config) {
            Ok(p) => p,
            Err(_) => continue,
        };

        let dispatches = count_dispatches(&plan);
        let cost_ns = cost_model.estimate(&plan).total_ns;

        // Primary: minimize dispatch count. Secondary: minimize cost.
        let is_better = dispatches < best_dispatches
            || (dispatches == best_dispatches && cost_ns < best_cost_ns);

        if is_better {
            best_dispatches = dispatches;
            best_cost_ns = cost_ns;
            best_plan = plan;
            best_config = config;
        }
    }

    Ok(OptimizationResult {
        plan: best_plan,
        config: best_config,
        dispatch_count: best_dispatches,
        configs_explored,
        baseline_dispatch_count: baseline_dispatches,
        best_cost_ns,
        baseline_cost_ns,
    })
}

/// Optimize multiple segments and return per-segment results.
///
/// Useful for analyzing all segments of a model like Kokoro. Each segment
/// is optimized independently with its own time budget.
///
/// Segments that fail compilation are skipped (not included in the output).
#[must_use]
pub fn optimize_segments(
    segments: &[(&str, &ComputationGraph)],
    cost_model: &CostModel,
    per_segment_budget: Duration,
) -> Vec<SegmentOptimizationResult> {
    segments
        .iter()
        .filter_map(|(name, graph)| {
            optimize_plan_with_cost(graph, cost_model, per_segment_budget)
                .ok()
                .map(|result| SegmentOptimizationResult {
                    segment_name: (*name).to_string(),
                    result,
                })
        })
        .collect()
}

/// Names of PeepholeConfig fields in bit-order (bit 0 .. bit 18).
///
/// Must stay in sync with [`config_from_bitmask`] and [`PeepholeConfig`].
pub(crate) const PEEPHOLE_FIELD_NAMES: [&str; 28] = [
    "norm_activ_conv1d",
    "fused_resblock",
    "linear_activation",
    "add_layer_norm",
    "norm_linear",
    "attention_transpose",
    "flip_lstm",
    "batched_linear_projection",
    "channels_first_layer_norm",
    "silu_mul",
    "auto_fuse_elementwise",
    "bilstm_cat",
    "add_norm_linear",
    "fuse_adain_snake",
    "fuse_upsample_conv1d",
    "fuse_instance_norm_mul_add",
    "fuse_conv1d_activation",
    "fuse_snake_instance_norm",
    "fuse_conv1d_snake_norm",
    "fuse_conv1d_snake_norm_resblock",
    "fuse_add_instance_norm_conv1x1",
    "fuse_conv_transpose1d_activation",
    "norm_activ_conv_transpose1d",
    "fuse_instance_norm_conv1d",
    "fuse_conv1d_instance_norm",
    "fuse_linear_layer_norm",
    "fuse_resblock_chain",
    "fuse_activation_conv1d",
];

/// Result of single-pass ablation analysis.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "plan-serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PassImpactEntry {
    /// Name of the PeepholeConfig field (e.g. `"silu_mul"`).
    pub pass_name: String,
    /// Dispatch count when this pass is enabled (baseline = all enabled).
    pub enabled_dispatch_count: usize,
    /// Dispatch count when this pass alone is disabled (all others enabled).
    pub disabled_dispatch_count: usize,
    /// Impact: `disabled - enabled`. Positive means the pass reduces dispatches.
    pub impact: i64,
}

/// Analyze the impact of each peephole pass by toggling it individually.
///
/// For each of the 15 passes: compile with all-enabled baseline, then
/// compile with that single pass disabled. The delta is the pass's impact.
/// Results are sorted by impact descending (highest impact first).
///
/// # Errors
///
/// Returns `TensorIRError` if the baseline compilation fails. If disabling
/// a single pass causes a compilation failure, that pass is reported with
/// the baseline count as its disabled count (impact = 0) rather than
/// failing the whole analysis.
pub fn analyze_pass_impact(
    graph: &ComputationGraph,
) -> Result<Vec<PassImpactEntry>, TensorIRError> {
    let default_config = PeepholeConfig::default();
    let baseline_plan = compile_trace_to_plan_configured(graph, &default_config)?;
    let baseline_dispatches = count_dispatches(&baseline_plan);

    let all_on_mask = (1u32 << PEEPHOLE_FIELD_COUNT) - 1;

    let mut entries: Vec<PassImpactEntry> = (0..PEEPHOLE_FIELD_COUNT)
        .map(|bit| {
            let name = PEEPHOLE_FIELD_NAMES[bit as usize].to_string();

            // Flip one bit off in the all-enabled mask.
            let mask = all_on_mask ^ (1u32 << bit);
            let config = config_from_bitmask(mask);

            let disabled_dispatches = match compile_trace_to_plan_configured(graph, &config) {
                Ok(plan) => count_dispatches(&plan),
                // If compilation fails with this pass disabled, treat as
                // no impact (the pass is required for correct compilation).
                Err(_) => baseline_dispatches,
            };

            let impact = disabled_dispatches as i64 - baseline_dispatches as i64;

            PassImpactEntry {
                pass_name: name,
                enabled_dispatch_count: baseline_dispatches,
                disabled_dispatch_count: disabled_dispatches,
                impact,
            }
        })
        .collect();

    // Sort by impact descending (most impactful pass first).
    entries.sort_by_key(|e| std::cmp::Reverse(e.impact));

    Ok(entries)
}

/// Check whether a config matches the default (all passes enabled).
pub(crate) fn is_default_config(config: &PeepholeConfig) -> bool {
    config.norm_activ_conv1d
        && config.fused_resblock
        && config.linear_activation
        && config.add_layer_norm
        && config.norm_linear
        && config.attention_transpose
        && config.flip_lstm
        && config.batched_linear_projection
        && config.channels_first_layer_norm
        && config.silu_mul
        && config.auto_fuse_elementwise
        && config.bilstm_cat
        && config.add_norm_linear
        && config.fuse_adain_snake
        && config.fuse_upsample_conv1d
        && config.fuse_instance_norm_mul_add
        && config.fuse_conv1d_activation
        && config.fuse_snake_instance_norm
        && config.fuse_conv1d_snake_norm
        && config.fuse_conv1d_snake_norm_resblock
        && config.fuse_add_instance_norm_conv1x1
        && config.fuse_conv_transpose1d_activation
        && config.norm_activ_conv_transpose1d
        && config.fuse_instance_norm_conv1d
        && config.fuse_conv1d_instance_norm
        && config.fuse_linear_layer_norm
        && config.fuse_resblock_chain
        && config.fuse_activation_conv1d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enumerate_produces_correct_count() {
        // Count lazily (O(1) memory) rather than materializing 2^28 configs.
        assert_eq!(
            enumerate_peephole_configs().count(),
            1usize << PEEPHOLE_FIELD_COUNT
        );
        assert_eq!(enumerate_peephole_configs().count(), 268_435_456);
    }

    #[test]
    fn test_enumerate_includes_all_disabled() {
        // Bitmask 0 = all disabled (O(1), no full enumeration).
        let all_off = config_from_bitmask(0);
        let all_off = &all_off;
        assert!(!all_off.norm_activ_conv1d);
        assert!(!all_off.fused_resblock);
        assert!(!all_off.linear_activation);
        assert!(!all_off.add_layer_norm);
        assert!(!all_off.norm_linear);
        assert!(!all_off.attention_transpose);
        assert!(!all_off.flip_lstm);
        assert!(!all_off.batched_linear_projection);
        assert!(!all_off.channels_first_layer_norm);
        assert!(!all_off.silu_mul);
        assert!(!all_off.auto_fuse_elementwise);
        assert!(!all_off.bilstm_cat);
        assert!(!all_off.add_norm_linear);
        assert!(!all_off.fuse_adain_snake);
        assert!(!all_off.fuse_upsample_conv1d);
        assert!(!all_off.fuse_instance_norm_mul_add);
        assert!(!all_off.fuse_conv1d_activation);
        assert!(!all_off.fuse_snake_instance_norm);
        assert!(!all_off.fuse_conv1d_snake_norm);
        assert!(!all_off.fuse_conv1d_snake_norm_resblock);
        assert!(!all_off.fuse_add_instance_norm_conv1x1);
    }

    #[test]
    fn test_enumerate_includes_all_enabled() {
        // Last entry (all bits set) = all enabled = default (O(1)).
        let all_on = config_from_bitmask((1u32 << PEEPHOLE_FIELD_COUNT) - 1);
        let all_on = &all_on;
        assert!(all_on.norm_activ_conv1d);
        assert!(all_on.fused_resblock);
        assert!(all_on.linear_activation);
        assert!(all_on.add_layer_norm);
        assert!(all_on.norm_linear);
        assert!(all_on.attention_transpose);
        assert!(all_on.flip_lstm);
        assert!(all_on.batched_linear_projection);
        assert!(all_on.channels_first_layer_norm);
        assert!(all_on.silu_mul);
        assert!(all_on.auto_fuse_elementwise);
        assert!(all_on.bilstm_cat);
        assert!(all_on.add_norm_linear);
        assert!(all_on.fuse_adain_snake);
        assert!(all_on.fuse_upsample_conv1d);
        assert!(all_on.fuse_instance_norm_mul_add);
        assert!(all_on.fuse_conv1d_activation);
        assert!(all_on.fuse_snake_instance_norm);
        assert!(all_on.fuse_conv1d_snake_norm);
        assert!(all_on.fuse_conv1d_snake_norm_resblock);
        assert!(all_on.fuse_add_instance_norm_conv1x1);
    }

    #[test]
    fn test_default_config_detection() {
        assert!(is_default_config(&PeepholeConfig::default()));

        let non_default = PeepholeConfig {
            silu_mul: false,
            ..Default::default()
        };
        assert!(!is_default_config(&non_default));
    }

    #[test]
    fn test_count_dispatches_empty_plan() {
        let plan = CompiledPlan {
            steps: vec![],
            input_shapes: vec![],
            output_step: 0,
            weight_names: vec![],
        };
        assert_eq!(count_dispatches(&plan), 0);
    }

    #[test]
    fn test_count_dispatches_mixed_steps() {
        let plan = CompiledPlan {
            steps: vec![
                CompiledStep::InputForward,
                CompiledStep::Passthrough {
                    op_name: "reshape".to_string(),
                    output_shape: vec![1, 2],
                },
                CompiledStep::IdentityPassthrough,
                CompiledStep::ConstantValue {
                    value: 1.0,
                    shape: vec![1],
                },
            ],
            input_shapes: vec![vec![1, 2]],
            output_step: 3,
            weight_names: vec![],
        };
        // None of these are Dispatch or NativeOp
        assert_eq!(count_dispatches(&plan), 0);
    }

    #[test]
    fn test_optimize_plan_with_simple_graph() {
        use nn_core::dyn_tensor::trace::ComputationGraph;

        // Empty graph — produces empty plan, 0 dispatches.
        let graph = ComputationGraph::from_nodes(vec![]);
        let result = optimize_plan(&graph, Duration::from_secs(10))
            .expect("optimize_plan should succeed on empty graph");
        assert_eq!(result.dispatch_count, 0);
        assert_eq!(result.baseline_dispatch_count, 0);
        assert!(
            result.configs_explored >= 1,
            "should explore at least baseline"
        );
    }

    #[test]
    fn test_optimize_plan_zero_budget_returns_baseline() {
        let graph = ComputationGraph::from_nodes(vec![]);
        let result = optimize_plan(&graph, Duration::ZERO)
            .expect("optimize_plan should succeed with zero budget");
        // With zero budget, should still have baseline.
        assert_eq!(result.configs_explored, 1);
        assert_eq!(result.baseline_dispatch_count, 0);
    }

    #[test]
    fn test_summarize_output() {
        let result = OptimizationResult {
            plan: CompiledPlan {
                steps: vec![],
                input_shapes: vec![],
                output_step: 0,
                weight_names: vec![],
            },
            config: PeepholeConfig::default(),
            dispatch_count: 180,
            configs_explored: 4096,
            baseline_dispatch_count: 200,
            best_cost_ns: 9000.0,
            baseline_cost_ns: 10000.0,
        };
        let summary = result.summarize();
        assert!(summary.contains("200"), "should mention baseline count");
        assert!(summary.contains("180"), "should mention best count");
        assert!(
            summary.contains("10.0%"),
            "should show percentage reduction"
        );
        assert!(summary.contains("4096"), "should mention configs explored");
        assert!(
            summary.contains("Baseline cost"),
            "should include cost info"
        );
        assert!(summary.contains("Best cost"), "should include best cost");
    }
}

#[cfg(test)]
#[path = "optimize_plan_tests.rs"]
mod optimize_plan_tests;

#[cfg(test)]
#[path = "peephole_search_tests.rs"]
mod peephole_search_tests;
