// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Data-driven fusion opportunity scanner for compiled execution plans.
//!
//! Unlike [`fusion_opportunity_analyzer`](super::fusion_opportunity_analyzer) which
//! scans category-based 2/3-step windows, this module works on actual plan step
//! indices, considers data dependencies and fan-out, and identifies recurring
//! multi-step patterns with precise dispatch savings estimates.
//!
//! The scanner answers: "Given the production compiled plan, which consecutive
//! op sequences appear most frequently, and how many dispatches would we save
//! by fusing each pattern into a single NativeOp?"
//!
//! # Usage
//!
//! ```rust,ignore
//! use nn_dsl::trace_compile_fusion_scanner::{scan_fusion_opportunities, FusionCategory};
//!
//! let plan = compile_trace_to_plan_with_fusion(&graph)?;
//! let opportunities = scan_fusion_opportunities(&plan);
//! for opp in &opportunities {
//!     println!("{}: {} instances, saves {} dispatches",
//!         opp.pattern_name(), opp.instances, opp.dispatch_savings);
//! }
//! ```
//!
//! Part of #4264 (RTF optimization).

use std::collections::HashMap;
use std::fmt;

use super::trace_compile_types::{CompiledPlan, CompiledStep};

/// Fusion category for classifying compiled steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FusionCategory {
    /// Elementwise ops on same-shape tensors (add, mul, relu, exp, etc.).
    ElementwiseChain,
    /// Scalar broadcast folded into elementwise (add_scalar, mul_scalar).
    BroadcastFold,
    /// Convolution followed by activation (conv1d + snake, conv1d + relu).
    ConvActivation,
    /// Normalization + activation + convolution (already fused via NativeOp).
    NormActivConv,
    /// Consecutive linear projections that could be batched.
    LinearChain,
    /// Normalization followed by linear/GEMM.
    NormLinear,
    /// Activation followed by convolution.
    ActivationConv,
    /// Convolution followed by normalization.
    ConvNorm,
    /// Linear followed by normalization.
    LinearNorm,
    /// Linear followed by activation.
    LinearActivation,
    /// Add followed by normalization (residual connection patterns).
    AddNorm,
    /// Custom or unclassified fusible pattern.
    Custom(u64),
}

impl fmt::Display for FusionCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ElementwiseChain => write!(f, "ElementwiseChain"),
            Self::BroadcastFold => write!(f, "BroadcastFold"),
            Self::ConvActivation => write!(f, "ConvActivation"),
            Self::NormActivConv => write!(f, "NormActivConv"),
            Self::LinearChain => write!(f, "LinearChain"),
            Self::NormLinear => write!(f, "NormLinear"),
            Self::ActivationConv => write!(f, "ActivationConv"),
            Self::ConvNorm => write!(f, "ConvNorm"),
            Self::LinearNorm => write!(f, "LinearNorm"),
            Self::LinearActivation => write!(f, "LinearActivation"),
            Self::AddNorm => write!(f, "AddNorm"),
            Self::Custom(id) => write!(f, "Custom({id})"),
        }
    }
}

/// A detected fusion opportunity with step-level detail.
#[derive(Clone, Debug)]
pub struct FusionOpportunity {
    /// Step indices in the compiled plan that form this pattern.
    pub steps: Vec<usize>,
    /// Operation names for each step (kernel name or NativeOp variant).
    pub ops: Vec<String>,
    /// Fusion category of this pattern.
    pub category: FusionCategory,
    /// How many times this exact op-name pattern appears in the plan.
    pub instances: usize,
    /// Total dispatch savings if all instances were fused.
    /// Each instance saves `(ops.len() - 1)` dispatches.
    pub dispatch_savings: usize,
}

impl FusionOpportunity {
    /// Human-readable name for this pattern.
    #[must_use]
    pub fn pattern_name(&self) -> String {
        self.ops.join(" -> ")
    }

    /// Priority score: instances * dispatch_savings.
    /// Higher = more impactful to implement.
    #[must_use]
    pub fn priority_score(&self) -> usize {
        self.instances * self.dispatch_savings
    }
}

impl fmt::Display for FusionOpportunity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} ({}x, saves {} dispatches, priority={})",
            self.category,
            self.pattern_name(),
            self.instances,
            self.dispatch_savings,
            self.priority_score(),
        )
    }
}

/// Result of a fusion scan.
#[derive(Clone, Debug)]
pub struct FusionScanResult {
    /// All detected opportunities, sorted by priority score descending.
    pub opportunities: Vec<FusionOpportunity>,
    /// Total dispatch steps in the plan.
    pub total_dispatches: usize,
    /// Total potential dispatch savings (sum across all opportunities).
    pub total_potential_savings: usize,
}

impl FusionScanResult {
    /// Human-readable summary.
    #[must_use]
    pub fn summarize(&self) -> String {
        let mut s = format!(
            "Fusion Scan: {} dispatches, {} opportunities, {} potential savings\n",
            self.total_dispatches,
            self.opportunities.len(),
            self.total_potential_savings,
        );
        for (i, opp) in self.opportunities.iter().enumerate().take(20) {
            s.push_str(&format!("  {}: {opp}\n", i + 1));
        }
        if self.opportunities.len() > 20 {
            s.push_str(&format!(
                "  ... and {} more\n",
                self.opportunities.len() - 20
            ));
        }
        s
    }
}

impl fmt::Display for FusionScanResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summarize())
    }
}

// ---------------------------------------------------------------------------
// Step classification
// ---------------------------------------------------------------------------

/// Coarse op class for a compiled step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum OpClass {
    Elementwise,
    Activation,
    Norm,
    Conv,
    Linear,
    Reduction,
    NativeOp,
    Other,
}

/// Classify a kernel name into an [`OpClass`].
fn classify_kernel(name: &str) -> OpClass {
    let base = name.strip_prefix("fused_").unwrap_or(name);

    // Normalization
    if base.contains("layer_norm")
        || base.contains("instance_norm")
        || base.contains("rms_norm")
        || base.contains("group_norm")
        || base.contains("batch_norm")
    {
        return OpClass::Norm;
    }

    // Activation (before elementwise — some overlap)
    if base == "snake"
        || base == "snake_tensor"
        || base == "leaky_relu"
        || base == "silu"
        || base.starts_with("tanh")
        || base == "sigmoid"
        || base == "gelu"
        || base == "relu"
    {
        return OpClass::Activation;
    }

    // Convolution
    if base.starts_with("conv1d")
        || base.starts_with("conv2d")
        || base.starts_with("conv_transpose")
    {
        return OpClass::Conv;
    }

    // Linear / matmul
    if base == "linear" || base.starts_with("matmul") || base == "gemm" {
        return OpClass::Linear;
    }

    // Reduction
    if base == "softmax" || base == "log_softmax" || base == "mean" || base == "sum" {
        return OpClass::Reduction;
    }

    // Elementwise
    if matches!(
        base,
        "add"
            | "sub"
            | "mul"
            | "div"
            | "neg"
            | "exp"
            | "log"
            | "abs"
            | "clamp"
            | "pow"
            | "sqrt"
            | "rsqrt"
            | "reciprocal"
            | "where_cond"
            | "binary_add"
            | "binary_sub"
            | "binary_mul"
            | "binary_div"
            | "add_scalar"
            | "mul_scalar"
            | "maximum"
            | "minimum"
            | "sin"
            | "cos"
    ) {
        return OpClass::Elementwise;
    }

    OpClass::Other
}

/// Extract op label and class from a compiled step.
/// Returns `None` for non-compute steps.
fn step_info(step: &CompiledStep) -> Option<(String, OpClass)> {
    match step {
        CompiledStep::Dispatch { kernel, .. } => {
            let name = kernel.name().to_string();
            let class = classify_kernel(&name);
            Some((name, class))
        }
        CompiledStep::NativeOp { op, .. } => {
            Some((op.variant_name().to_string(), OpClass::NativeOp))
        }
        _ => None,
    }
}

/// Determine the [`FusionCategory`] from a sequence of [`OpClass`] values.
fn classify_pattern(classes: &[OpClass]) -> FusionCategory {
    if classes.len() < 2 {
        return FusionCategory::Custom(0);
    }

    // Check for specific 2-step patterns.
    if classes.len() == 2 {
        return match (classes[0], classes[1]) {
            (OpClass::Conv, OpClass::Activation) => FusionCategory::ConvActivation,
            (OpClass::Activation, OpClass::Conv) => FusionCategory::ActivationConv,
            (OpClass::Conv, OpClass::Norm) => FusionCategory::ConvNorm,
            (OpClass::Norm, OpClass::Conv) => FusionCategory::NormActivConv,
            (OpClass::Norm, OpClass::Linear) => FusionCategory::NormLinear,
            (OpClass::Linear, OpClass::Norm) => FusionCategory::LinearNorm,
            (OpClass::Linear, OpClass::Activation) => FusionCategory::LinearActivation,
            (OpClass::Norm, OpClass::Activation) => FusionCategory::AddNorm,
            (OpClass::Elementwise, OpClass::Elementwise) => FusionCategory::ElementwiseChain,
            (OpClass::Elementwise, OpClass::Activation) => FusionCategory::ElementwiseChain,
            (OpClass::Activation, OpClass::Elementwise) => FusionCategory::ElementwiseChain,
            (OpClass::Elementwise, OpClass::Norm) => FusionCategory::AddNorm,
            (OpClass::Linear, OpClass::Linear) => FusionCategory::LinearChain,
            _ => FusionCategory::Custom(0),
        };
    }

    // For 3+ step patterns, classify by dominant category.
    let has_conv = classes.contains(&OpClass::Conv);
    let has_norm = classes.contains(&OpClass::Norm);
    let has_activation = classes.contains(&OpClass::Activation);
    let has_linear = classes.contains(&OpClass::Linear);
    let all_elementwise = classes
        .iter()
        .all(|c| matches!(c, OpClass::Elementwise | OpClass::Activation));

    if all_elementwise {
        return FusionCategory::ElementwiseChain;
    }
    if has_norm && has_activation && has_conv {
        return FusionCategory::NormActivConv;
    }
    if has_conv && has_activation {
        return FusionCategory::ConvActivation;
    }
    if has_norm && has_linear {
        return FusionCategory::NormLinear;
    }
    if has_linear && has_activation {
        return FusionCategory::LinearActivation;
    }
    if has_conv && has_norm {
        return FusionCategory::ConvNorm;
    }

    FusionCategory::Custom(0)
}

/// Returns `true` if two [`OpClass`] values represent a fusible pair.
fn is_fusible_adjacent(a: OpClass, b: OpClass) -> bool {
    matches!(
        (a, b),
        (OpClass::Elementwise, OpClass::Elementwise)
            | (OpClass::Elementwise, OpClass::Activation)
            | (OpClass::Activation, OpClass::Elementwise)
            | (OpClass::Activation, OpClass::Activation)
            | (OpClass::Norm, OpClass::Activation)
            | (OpClass::Norm, OpClass::Linear)
            | (OpClass::Norm, OpClass::Conv)
            | (OpClass::Norm, OpClass::Elementwise)
            | (OpClass::Activation, OpClass::Conv)
            | (OpClass::Activation, OpClass::Linear)
            | (OpClass::Linear, OpClass::Activation)
            | (OpClass::Linear, OpClass::Norm)
            | (OpClass::Linear, OpClass::Linear)
            | (OpClass::Conv, OpClass::Activation)
            | (OpClass::Conv, OpClass::Norm)
            | (OpClass::Conv, OpClass::Elementwise)
            | (OpClass::Elementwise, OpClass::Norm)
            | (OpClass::Elementwise, OpClass::Conv)
            | (OpClass::Elementwise, OpClass::Linear)
            | (OpClass::Reduction, OpClass::Elementwise)
    )
}

// ---------------------------------------------------------------------------
// Core scanner
// ---------------------------------------------------------------------------

/// Scan a compiled plan for fusion opportunities.
///
/// Walks the plan's steps, identifies consecutive compute operations (Dispatch
/// or NativeOp) that could be fused, groups them into patterns, counts pattern
/// frequency, and returns opportunities sorted by priority (instances x savings).
///
/// The scanner:
/// 1. Extracts compute steps with their op names and classes.
/// 2. Identifies maximal fusible runs of consecutive compute steps.
/// 3. Within each run, extracts all 2-step and 3-step sub-patterns.
/// 4. Deduplicates patterns by op-name sequence.
/// 5. Computes dispatch savings: each fused pattern of length N saves N-1 dispatches.
/// 6. Sorts by priority score (instances * savings per instance).
#[must_use]
pub fn scan_fusion_opportunities(plan: &CompiledPlan) -> FusionScanResult {
    // Step 1: extract compute step info.
    let step_infos: Vec<Option<(String, OpClass)>> =
        plan.steps.iter().map(step_info).collect();

    // Count total dispatches.
    let total_dispatches = plan
        .steps
        .iter()
        .filter(|s| {
            matches!(
                s,
                CompiledStep::Dispatch { .. } | CompiledStep::NativeOp { .. }
            )
        })
        .count();

    // Step 2: find maximal runs of consecutive compute steps that are
    // individually fusible (non-NativeOp, non-Other).
    let runs = find_fusible_runs(&step_infos);

    // Step 3: extract 2-step and 3-step patterns from each run.
    let mut pattern_counts: HashMap<Vec<String>, Vec<Vec<usize>>> = HashMap::new();
    let mut pattern_classes: HashMap<Vec<String>, Vec<OpClass>> = HashMap::new();

    for run in &runs {
        // 2-step windows.
        for window in run.windows(2) {
            let (i, j) = (window[0], window[1]);
            if let (Some((name_a, class_a)), Some((name_b, class_b))) =
                (&step_infos[i], &step_infos[j])
            {
                if is_fusible_adjacent(*class_a, *class_b) {
                    let key = vec![name_a.clone(), name_b.clone()];
                    pattern_counts
                        .entry(key.clone())
                        .or_default()
                        .push(vec![i, j]);
                    pattern_classes
                        .entry(key)
                        .or_insert_with(|| vec![*class_a, *class_b]);
                }
            }
        }

        // 3-step windows.
        for window in run.windows(3) {
            let (i, j, k) = (window[0], window[1], window[2]);
            if let (Some((name_a, class_a)), Some((name_b, class_b)), Some((name_c, class_c))) =
                (&step_infos[i], &step_infos[j], &step_infos[k])
            {
                if is_fusible_adjacent(*class_a, *class_b)
                    && is_fusible_adjacent(*class_b, *class_c)
                {
                    let key = vec![name_a.clone(), name_b.clone(), name_c.clone()];
                    pattern_counts
                        .entry(key.clone())
                        .or_default()
                        .push(vec![i, j, k]);
                    pattern_classes
                        .entry(key)
                        .or_insert_with(|| vec![*class_a, *class_b, *class_c]);
                }
            }
        }

        // 4-step windows (for deeper patterns like norm+activ+conv+norm).
        for window in run.windows(4) {
            let indices: Vec<usize> = window.to_vec();
            let infos: Vec<&(String, OpClass)> = indices
                .iter()
                .filter_map(|&idx| step_infos[idx].as_ref())
                .collect();
            if infos.len() == 4 {
                let all_fusible = infos
                    .windows(2)
                    .all(|pair| is_fusible_adjacent(pair[0].1, pair[1].1));
                if all_fusible {
                    let key: Vec<String> = infos.iter().map(|(n, _)| n.clone()).collect();
                    let classes: Vec<OpClass> = infos.iter().map(|(_, c)| *c).collect();
                    pattern_counts.entry(key.clone()).or_default().push(indices);
                    pattern_classes.entry(key).or_insert(classes);
                }
            }
        }
    }

    // Step 4: build FusionOpportunity for each pattern.
    let mut opportunities: Vec<FusionOpportunity> = Vec::new();

    for (op_names, occurrences) in &pattern_counts {
        let instances = occurrences.len();
        let pattern_len = op_names.len();
        let savings_per_instance = pattern_len.saturating_sub(1);
        let dispatch_savings = instances * savings_per_instance;

        let classes = pattern_classes.get(op_names).cloned().unwrap_or_default();
        let category = classify_pattern(&classes);

        // Use the first occurrence's step indices as representative.
        let representative_steps = occurrences[0].clone();

        opportunities.push(FusionOpportunity {
            steps: representative_steps,
            ops: op_names.clone(),
            category,
            instances,
            dispatch_savings,
        });
    }

    // Step 5: sort by priority score descending.
    opportunities.sort_by_key(|o| std::cmp::Reverse(o.priority_score()));

    let total_potential_savings: usize = opportunities.iter().map(|o| o.dispatch_savings).sum();

    FusionScanResult {
        opportunities,
        total_dispatches,
        total_potential_savings,
    }
}

/// Find maximal runs of consecutive step indices that are fusible compute ops.
///
/// A step is "fusible" if it is a Dispatch with a non-Other, non-NativeOp class.
/// NativeOps break runs because they are already optimally fused.
fn find_fusible_runs(step_infos: &[Option<(String, OpClass)>]) -> Vec<Vec<usize>> {
    let mut runs: Vec<Vec<usize>> = Vec::new();
    let mut current_run: Vec<usize> = Vec::new();

    for (idx, info) in step_infos.iter().enumerate() {
        let is_fusible_compute = match info {
            Some((_, class)) => !matches!(class, OpClass::NativeOp | OpClass::Other),
            None => false,
        };

        if is_fusible_compute {
            current_run.push(idx);
        } else {
            if current_run.len() >= 2 {
                runs.push(std::mem::take(&mut current_run));
            } else {
                current_run.clear();
            }
        }
    }

    // Don't forget the last run.
    if current_run.len() >= 2 {
        runs.push(current_run);
    }

    runs
}

#[cfg(test)]
#[path = "trace_compile_fusion_scanner_tests.rs"]
mod tests;
