// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! NativeOp fusion opportunity analyzer for compiled execution plans.
//!
//! Scans consecutive dispatch steps in a [`CompiledPlan`] for recurring
//! multi-step patterns that could be fused into new NativeOp variants.
//! Returns opportunities sorted by estimated savings (frequency × cost).
//!
//! Unlike [`fusion_gap_analyzer`](super::fusion_gap_analyzer) which diagnoses
//! WHY adjacent elementwise ops weren't fused, this module identifies
//! WHICH multi-step sequences appear most frequently and would benefit
//! most from a new fused NativeOp — guiding prioritization of new
//! peephole passes.
//!
//! Part of #4252.

use std::collections::HashMap;

use super::trace_compile_types::{CompiledPlan, CompiledStep};

/// An operation category for fusion pattern matching.
///
/// Classifies each compiled step into a coarse-grained category used to
/// identify fusible sequences. Two adjacent steps in the same category
/// (e.g., Elementwise+Elementwise) or in known-fusible category pairs
/// (e.g., Norm+Activation, Activation+Conv) form fusion opportunities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum OpCategory {
    /// Elementwise unary or binary ops (relu, gelu, silu, exp, mul, add, etc.).
    Elementwise,
    /// Normalization ops (layer_norm, instance_norm, rms_norm, group_norm).
    Norm,
    /// Activation functions (snake, leaky_relu, silu, tanh, sigmoid).
    Activation,
    /// Convolution ops (conv1d, conv2d, conv_transpose_1d).
    Conv,
    /// Linear / matmul / GEMM ops.
    Linear,
    /// Reduction ops (softmax, log_softmax, mean, sum).
    Reduction,
    /// Already a NativeOp (fused kernel).
    NativeOp,
    /// Other (passthrough, input, constant, runtime, etc.).
    Other,
}

/// Classify a kernel name into an [`OpCategory`].
fn classify_kernel_name(name: &str) -> OpCategory {
    // Normalize: strip "fused_" prefix and "_xN" suffix for chain-fused kernels.
    let base = name.strip_prefix("fused_").unwrap_or(name);

    // Norm ops
    if base.contains("layer_norm")
        || base.contains("instance_norm")
        || base.contains("rms_norm")
        || base.contains("group_norm")
        || base.contains("batch_norm")
    {
        return OpCategory::Norm;
    }

    // Activation ops (check before elementwise since some overlap)
    if base == "snake"
        || base == "leaky_relu"
        || base == "silu"
        || base.starts_with("tanh")
        || base == "sigmoid"
        || base == "gelu"
        || base == "relu"
    {
        return OpCategory::Activation;
    }

    // Conv ops
    if base.starts_with("conv1d")
        || base.starts_with("conv2d")
        || base.starts_with("conv_transpose")
    {
        return OpCategory::Conv;
    }

    // Linear / matmul
    if base == "linear" || base.starts_with("matmul") || base == "gemm" {
        return OpCategory::Linear;
    }

    // Reduction ops
    if base == "softmax" || base == "log_softmax" || base == "mean" || base == "sum" {
        return OpCategory::Reduction;
    }

    // Elementwise binary/unary
    if base == "add"
        || base == "sub"
        || base == "mul"
        || base == "div"
        || base == "neg"
        || base == "exp"
        || base == "log"
        || base == "abs"
        || base == "clamp"
        || base == "pow"
        || base == "sqrt"
        || base == "rsqrt"
        || base == "reciprocal"
        || base == "where_cond"
        || base == "binary_add"
        || base == "binary_sub"
        || base == "binary_mul"
        || base == "binary_div"
    {
        return OpCategory::Elementwise;
    }

    OpCategory::Other
}

/// Extract a human-readable operation label from a compiled step.
///
/// Returns `Some(label)` for Dispatch and NativeOp steps that represent
/// actual GPU work. Returns `None` for non-dispatch steps (Passthrough,
/// InputForward, IdentityPassthrough, etc.).
fn step_op_label(step: &CompiledStep) -> Option<String> {
    match step {
        CompiledStep::Dispatch { kernel, .. } => Some(kernel.name().to_string()),
        CompiledStep::NativeOp { op, .. } => Some(op.variant_name().to_string()),
        _ => None,
    }
}

/// Classify a compiled step into an [`OpCategory`].
fn classify_step(step: &CompiledStep) -> OpCategory {
    match step {
        CompiledStep::Dispatch { kernel, .. } => classify_kernel_name(kernel.name()),
        CompiledStep::NativeOp { .. } => OpCategory::NativeOp,
        _ => OpCategory::Other,
    }
}

/// Known fusible category pairs: `(first, second)` where fusing the pair
/// into a single kernel would eliminate one dispatch.
const FUSIBLE_PAIRS: &[(OpCategory, OpCategory)] = &[
    (OpCategory::Elementwise, OpCategory::Elementwise),
    (OpCategory::Norm, OpCategory::Activation),
    (OpCategory::Norm, OpCategory::Linear),
    (OpCategory::Activation, OpCategory::Conv),
    (OpCategory::Activation, OpCategory::Linear),
    (OpCategory::Linear, OpCategory::Activation),
    (OpCategory::Norm, OpCategory::Conv),
    (OpCategory::Elementwise, OpCategory::Activation),
    (OpCategory::Activation, OpCategory::Elementwise),
    (OpCategory::Norm, OpCategory::Elementwise),
    (OpCategory::Reduction, OpCategory::Elementwise),
    (OpCategory::Conv, OpCategory::Activation),
    (OpCategory::Conv, OpCategory::Norm),
    (OpCategory::Linear, OpCategory::Norm),
];

/// Returns `true` if the `(first, second)` category pair is fusible.
fn is_fusible_pair(first: OpCategory, second: OpCategory) -> bool {
    FUSIBLE_PAIRS
        .iter()
        .any(|&(a, b)| a == first && b == second)
}

/// A detected fusion opportunity: a recurring multi-step sequence that
/// could be replaced by a single NativeOp.
#[derive(Clone, Debug)]
pub struct FusionOpportunity {
    /// The operation names in sequence (e.g., `["instance_norm", "mul", "add", "snake"]`).
    pub sequence: Vec<String>,
    /// Number of times this exact sequence appears in the plan.
    pub count: usize,
    /// Estimated total cost in nanoseconds across all occurrences.
    ///
    /// Computed as `count * (sequence.len() - 1) * DISPATCH_OVERHEAD_NS`.
    /// Higher values indicate more impactful fusion targets.
    pub total_cost_ns: f64,
}

/// Estimated per-dispatch overhead in nanoseconds (Metal kernel launch).
///
/// Conservative estimate based on observed Metal dispatch latency on M4.
/// The actual savings from fusion include both launch overhead elimination
/// and reduced memory traffic from avoiding intermediate buffers.
const DISPATCH_OVERHEAD_NS: f64 = 2000.0;

/// Analyze a compiled plan for NativeOp fusion opportunities.
///
/// Scans consecutive dispatch steps (Dispatch + NativeOp) for fusible
/// 2-step and 3-step sequences. Returns opportunities sorted by
/// estimated savings (highest first).
///
/// # Arguments
///
/// * `plan` - The compiled execution plan to analyze.
///
/// # Returns
///
/// A list of [`FusionOpportunity`] structs, sorted descending by
/// `total_cost_ns` (most impactful fusion targets first).
#[must_use]
pub fn analyze_fusion_opportunities(plan: &CompiledPlan) -> Vec<FusionOpportunity> {
    // Collect dispatch-step labels and categories (skip non-dispatch steps).
    let dispatch_steps: Vec<(String, OpCategory)> = plan
        .steps
        .iter()
        .filter_map(|step| {
            let label = step_op_label(step)?;
            let cat = classify_step(step);
            // Skip Other (non-compute steps) and NativeOp (already fused).
            if cat == OpCategory::Other || cat == OpCategory::NativeOp {
                return None;
            }
            Some((label, cat))
        })
        .collect();

    if dispatch_steps.len() < 2 {
        return Vec::new();
    }

    let mut pair_counts: HashMap<Vec<String>, usize> = HashMap::new();
    let mut triple_counts: HashMap<Vec<String>, usize> = HashMap::new();

    // Scan for fusible 2-step pairs.
    for window in dispatch_steps.windows(2) {
        let (ref name_a, cat_a) = window[0];
        let (ref name_b, cat_b) = window[1];
        if is_fusible_pair(cat_a, cat_b) {
            let key = vec![name_a.clone(), name_b.clone()];
            *pair_counts.entry(key).or_insert(0) += 1;
        }
    }

    // Scan for fusible 3-step triples (both adjacent pairs must be fusible).
    for window in dispatch_steps.windows(3) {
        let (ref name_a, cat_a) = window[0];
        let (ref name_b, cat_b) = window[1];
        let (ref name_c, cat_c) = window[2];
        if is_fusible_pair(cat_a, cat_b) && is_fusible_pair(cat_b, cat_c) {
            let key = vec![name_a.clone(), name_b.clone(), name_c.clone()];
            *triple_counts.entry(key).or_insert(0) += 1;
        }
    }

    let mut opportunities: Vec<FusionOpportunity> = Vec::new();

    for (sequence, count) in pair_counts {
        let dispatches_saved = count; // Each pair fusion saves 1 dispatch.
        let total_cost_ns = dispatches_saved as f64 * DISPATCH_OVERHEAD_NS;
        opportunities.push(FusionOpportunity {
            sequence,
            count,
            total_cost_ns,
        });
    }

    for (sequence, count) in triple_counts {
        let dispatches_saved = count * 2; // Each triple fusion saves 2 dispatches.
        let total_cost_ns = dispatches_saved as f64 * DISPATCH_OVERHEAD_NS;
        opportunities.push(FusionOpportunity {
            sequence,
            count,
            total_cost_ns,
        });
    }

    // Sort descending by total_cost_ns (highest savings first).
    opportunities.sort_by(|a, b| {
        b.total_cost_ns
            .partial_cmp(&a.total_cost_ns)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    opportunities
}

#[cfg(test)]
#[path = "fusion_opportunity_analyzer_tests.rs"]
mod tests;
