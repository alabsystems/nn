// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Dispatch plan optimizer report for compiled models.
//!
//! Compares baseline (default PeepholeConfig) against an optimized config
//! to produce structured analysis of dispatch reduction, cost savings, and
//! per-pass impact. Complements [`compiled_model_memory_report`] with
//! optimization-focused diagnostics.
//!
//! # Usage
//!
//! ```rust,ignore
//! use nn_metal::compiled_model_optimizer_report::{
//!     generate_optimizer_report, format_optimizer_report, diff_peephole_configs,
//! };
//!
//! let report = generate_optimizer_report(&compiled_model_def, &optimized_config);
//! println!("{}", format_optimizer_report(&report));
//! ```
//!
//! Part of #3828.

use std::fmt;

use nn_dsl::trace_compile::CompiledStep;
use nn_dsl::PeepholeConfig;

use crate::compiled_model::CompiledModelDef;

/// Names of PeepholeConfig fields in declaration order (bit 0 .. bit 14).
///
/// Must stay in sync with [`PeepholeConfig`] and
/// [`nn_dsl::optimize_plan::config_from_bitmask`].
const FIELD_NAMES: [&str; 26] = [
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
];

/// Structured optimizer report comparing baseline vs optimized dispatch plans.
#[derive(Debug, Clone)]
pub struct OptimizerReport {
    /// Number of dispatch steps (Dispatch + NativeOp) with default config.
    pub baseline_dispatches: usize,
    /// Number of dispatch steps with the optimized config.
    pub optimized_dispatches: usize,
    /// Percentage reduction: `(baseline - optimized) / baseline * 100`.
    /// Zero when baseline is zero.
    pub dispatch_reduction_pct: f64,
    /// Estimated cost of the baseline plan in nanoseconds.
    /// Zero when no cost model data is available.
    pub baseline_cost_estimate: f64,
    /// Estimated cost of the optimized plan in nanoseconds.
    /// Zero when no cost model data is available.
    pub optimized_cost_estimate: f64,
    /// Estimated speedup: `baseline_cost / optimized_cost`.
    /// 1.0 when costs are equal or unavailable; f64::INFINITY when optimized
    /// cost is zero but baseline is non-zero.
    pub speedup_estimate: f64,
    /// The PeepholeConfig that produced the optimized result.
    pub config_used: PeepholeConfig,
    /// Names of passes that are enabled in `config_used`.
    pub passes_enabled: Vec<String>,
    /// Names of passes that are disabled in `config_used`.
    pub passes_disabled: Vec<String>,
}

/// Generate an optimizer report comparing the current compiled model against
/// the given optimized config.
///
/// Counts dispatch steps (Dispatch + NativeOp) from the compiled model
/// definition. Since `CompiledModelDef` stores already-compiled steps,
/// both baseline and optimized dispatch counts reflect the current plan.
/// For a full before/after comparison with cost estimates, use
/// [`generate_optimizer_report_with_metrics`] instead.
#[must_use]
pub(crate) fn generate_optimizer_report(
    def: &CompiledModelDef,
    optimized_config: &PeepholeConfig,
) -> OptimizerReport {
    let baseline_dispatches = count_dispatches_from_steps(&def.steps);

    // Without a recompilation, optimized == baseline.
    let optimized_dispatches = baseline_dispatches;

    let dispatch_reduction_pct = compute_reduction_pct(baseline_dispatches, optimized_dispatches);
    let speedup_estimate = 1.0; // No cost data without recompilation.

    let (passes_enabled, passes_disabled) = classify_passes(optimized_config);

    OptimizerReport {
        baseline_dispatches,
        optimized_dispatches,
        dispatch_reduction_pct,
        baseline_cost_estimate: 0.0,
        optimized_cost_estimate: 0.0,
        speedup_estimate,
        config_used: optimized_config.clone(),
        passes_enabled,
        passes_disabled,
    }
}

/// Generate an optimizer report with explicit baseline and optimized metrics.
///
/// Accepts pre-computed dispatch counts and cost estimates from an optimization
/// search (e.g., from [`nn_dsl::OptimizationResult`]). This variant produces
/// a full comparison with cost-based speedup estimates.
#[must_use]
pub fn generate_optimizer_report_with_metrics(
    def: &CompiledModelDef,
    optimized_config: &PeepholeConfig,
    optimized_dispatches: usize,
    optimized_cost_ns: f64,
) -> OptimizerReport {
    let baseline_dispatches = count_dispatches_from_steps(&def.steps);

    // Estimate baseline cost from dispatch count using launch overhead.
    // This is a rough proxy; for accurate cost, the caller should supply
    // the full CostEstimate from the optimization search result.
    let cost_model = nn_dsl::CostModel::apple_m4();
    let baseline_cost_estimate = baseline_dispatches as f64 * cost_model.launch_overhead_ns;

    let dispatch_reduction_pct = compute_reduction_pct(baseline_dispatches, optimized_dispatches);

    let speedup_estimate = if optimized_cost_ns > 0.0 {
        baseline_cost_estimate / optimized_cost_ns
    } else if baseline_cost_estimate > 0.0 {
        f64::INFINITY
    } else {
        1.0
    };

    let (passes_enabled, passes_disabled) = classify_passes(optimized_config);

    OptimizerReport {
        baseline_dispatches,
        optimized_dispatches,
        dispatch_reduction_pct,
        baseline_cost_estimate,
        optimized_cost_estimate: optimized_cost_ns,
        speedup_estimate,
        config_used: optimized_config.clone(),
        passes_enabled,
        passes_disabled,
    }
}

/// Format an optimizer report as a human-readable string.
#[must_use]
pub fn format_optimizer_report(report: &OptimizerReport) -> String {
    let mut out = String::new();
    out.push_str("=== Dispatch Plan Optimizer Report ===\n\n");

    // Dispatch counts
    out.push_str("--- Dispatch Counts ---\n");
    out.push_str(&format!(
        "Baseline dispatches:  {}\n",
        report.baseline_dispatches
    ));
    out.push_str(&format!(
        "Optimized dispatches: {}\n",
        report.optimized_dispatches
    ));
    let saved = report
        .baseline_dispatches
        .saturating_sub(report.optimized_dispatches);
    out.push_str(&format!(
        "Reduction:            {} dispatches ({:.1}%)\n",
        saved, report.dispatch_reduction_pct
    ));

    // Cost estimates
    out.push_str("\n--- Cost Estimates ---\n");
    out.push_str(&format!(
        "Baseline cost:  {:.1} us\n",
        report.baseline_cost_estimate / 1e3
    ));
    out.push_str(&format!(
        "Optimized cost: {:.1} us\n",
        report.optimized_cost_estimate / 1e3
    ));
    out.push_str(&format!(
        "Speedup:        {:.2}x\n",
        report.speedup_estimate
    ));

    // Pass configuration
    out.push_str("\n--- Pass Configuration ---\n");
    if report.passes_enabled.is_empty() {
        out.push_str("Enabled:  (none)\n");
    } else {
        out.push_str(&format!(
            "Enabled:  {} passes\n",
            report.passes_enabled.len()
        ));
        for pass in &report.passes_enabled {
            out.push_str(&format!("  + {pass}\n"));
        }
    }
    if report.passes_disabled.is_empty() {
        out.push_str("Disabled: (none)\n");
    } else {
        out.push_str(&format!(
            "Disabled: {} passes\n",
            report.passes_disabled.len()
        ));
        for pass in &report.passes_disabled {
            out.push_str(&format!("  - {pass}\n"));
        }
    }

    out
}

/// Compare two PeepholeConfigs and return a list of differing fields.
///
/// Each entry is `(field_name, value_in_a, value_in_b)`. Only fields
/// where `a` and `b` differ are included. An empty result means the
/// configs are identical.
#[must_use]
pub fn diff_peephole_configs(
    a: &PeepholeConfig,
    b: &PeepholeConfig,
) -> Vec<(String, bool, bool)> {
    let a_vals = config_to_bools(a);
    let b_vals = config_to_bools(b);

    FIELD_NAMES
        .iter()
        .enumerate()
        .filter_map(|(i, name)| {
            if a_vals[i] != b_vals[i] {
                Some((name.to_string(), a_vals[i], b_vals[i]))
            } else {
                None
            }
        })
        .collect()
}

/// Count dispatch steps (Dispatch + NativeOp) in compiled steps.
fn count_dispatches_from_steps(steps: &[CompiledStep]) -> usize {
    steps
        .iter()
        .filter(|step| {
            matches!(
                step,
                CompiledStep::Dispatch { .. } | CompiledStep::NativeOp { .. }
            )
        })
        .count()
}

/// Compute dispatch reduction percentage, handling zero baseline.
fn compute_reduction_pct(baseline: usize, optimized: usize) -> f64 {
    if baseline > 0 {
        let saved = baseline.saturating_sub(optimized);
        (saved as f64 / baseline as f64) * 100.0
    } else {
        0.0
    }
}

/// Classify passes as enabled or disabled.
fn classify_passes(config: &PeepholeConfig) -> (Vec<String>, Vec<String>) {
    let vals = config_to_bools(config);
    let mut enabled = Vec::new();
    let mut disabled = Vec::new();
    for (i, name) in FIELD_NAMES.iter().enumerate() {
        if vals[i] {
            enabled.push(name.to_string());
        } else {
            disabled.push(name.to_string());
        }
    }
    (enabled, disabled)
}

/// Extract boolean field values from a PeepholeConfig in field order.
fn config_to_bools(config: &PeepholeConfig) -> [bool; 26] {
    [
        config.norm_activ_conv1d,
        config.fused_resblock,
        config.linear_activation,
        config.add_layer_norm,
        config.norm_linear,
        config.attention_transpose,
        config.flip_lstm,
        config.batched_linear_projection,
        config.channels_first_layer_norm,
        config.silu_mul,
        config.auto_fuse_elementwise,
        config.bilstm_cat,
        config.add_norm_linear,
        config.fuse_adain_snake,
        config.fuse_upsample_conv1d,
        config.fuse_instance_norm_mul_add,
        config.fuse_conv1d_activation,
        config.fuse_snake_instance_norm,
        config.fuse_conv1d_snake_norm,
        config.fuse_conv1d_snake_norm_resblock,
        config.fuse_add_instance_norm_conv1x1,
        config.fuse_conv_transpose1d_activation,
        config.norm_activ_conv_transpose1d,
        config.fuse_instance_norm_conv1d,
        config.fuse_conv1d_instance_norm,
        config.fuse_linear_layer_norm,
    ]
}

impl fmt::Display for OptimizerReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", format_optimizer_report(self))
    }
}

#[cfg(test)]
#[path = "compiled_model_optimizer_report_tests.rs"]
mod tests;
