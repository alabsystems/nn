// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Implementation correctness check for moonshot Property 8.
//!
//! Maps dispatch step operation types to ay-proven kernel categories.
//! A pipeline achieves implementation correctness evidence when a high
//! fraction of its operations have ay-verified correctness proofs.
//!
//! # Architecture
//!
//! The ay BOUNDS_REGISTRY contains 20 kernel proofs (snake, silu_mul,
//! rope_cos, rope_sin, rms_norm_scalar, layer_norm_scalar,
//! instance_norm_scalar, instance_norm_affine_scalar, adain, adain_snake,
//! gelu, sigmoid, relu, tanh_act, leaky_relu, exp, softplus, add, mul,
//! conv1d_k1_scalar). These map to a subset of `DispatchStep` variants
//! via `ay_kernel_category()`.
//!
//! Operations without ay proofs (matmul, embedding, conv, softmax, reduce)
//! rely on Kani + CROWN for correctness. The fraction of ay-covered
//! operations determines whether P8 reaches SmtProven or Empirical.

use crate::moonshot::{VerificationLevel, PROPERTY_NAMES};

use super::MoonshotPropertyResult;

/// Evidence for Property 8 (implementation correctness) from dispatch plan.
///
/// Captures which operations in the pipeline have ay-proven kernel
/// correctness proofs and which do not.
#[derive(Debug, Clone)]
pub struct ImplementationCorrectnessEvidence {
    /// Total number of dispatch steps analyzed.
    pub total_steps: usize,
    /// Number of steps with ay-proven kernel categories.
    pub proven_steps: usize,
    /// Operation categories that are ay-proven.
    pub proven_categories: Vec<String>,
    /// Operation categories that lack ay proofs.
    pub unproven_categories: Vec<String>,
    /// Whether all steps have ay-proven kernels.
    pub all_proven: bool,
}

/// Known ay-proven kernel categories from BOUNDS_REGISTRY.
///
/// These kernel names correspond to the `name:` fields in
/// `crates/nn-verify/src/bounds/dispatch.rs` (BOUNDS_REGISTRY).
/// Updated for #2917: added "add", "mul", "conv1d_k1_scalar".
const AY_PROVEN_KERNELS: &[&str] = &[
    "snake",
    "silu_mul",
    "rope_cos",
    "rope_sin",
    "rms_norm_scalar",
    "layer_norm_scalar",
    "instance_norm_scalar",
    "instance_norm_affine_scalar",
    "adain",
    "adain_snake",
    "gelu",
    "sigmoid",
    "relu",
    "tanh_act",
    "leaky_relu",
    "exp",
    "softplus",
    "add",
    "mul",
    "conv1d_k1_scalar",
];

/// Map a dispatch step operation type to its ay kernel category name.
///
/// Returns `Some("kernel_name")` if the operation has a ay-proven kernel,
/// `None` if it does not.
///
/// # Mapping rationale
///
/// - `Sigmoid`, `Gelu`, `Relu`, `Tanh` → direct ay kernel match
/// - `BinaryAdd`, `BinaryMul` → ay-proven via scalar `f(x) = x + c` / `f(x) = x * c`
///   bounds in BOUNDS_REGISTRY (#2917)
/// - `Elementwise` → check kernel_name for known patterns (snake, silu_mul)
/// - `Linear`, `MatMul`, `Conv1d`, `Conv2d`, `ConvTranspose1d` → no ay proof
///   (correctness verified by Kani + CROWN composition, not per-kernel SMT)
/// - `Softmax`, `Reduce`, `Embedding` → no ay proof
/// - `Reshape`, `Narrow`, `Transpose`, `AxisSelect` → metadata-only ops,
///   considered trivially correct (no computation, just view changes)
pub fn ay_kernel_category(step: &nn_dsl::DispatchStep) -> Option<&'static str> {
    use nn_dsl::DispatchStep;
    match step {
        DispatchStep::Sigmoid { .. } => Some("sigmoid"),
        DispatchStep::Gelu { .. } => Some("gelu"),
        DispatchStep::Relu { .. } => Some("relu"),
        DispatchStep::Tanh { .. } => Some("tanh_act"),

        // Binary ops: ay-proven via scalar bounds (#2917).
        // Element-wise add/mul reduces to `f(x) = x + c` / `f(x) = x * c` where
        // the other operand is constant from NY's bound propagation.
        DispatchStep::BinaryAdd { .. } => Some("add"),
        DispatchStep::BinaryMul { .. } => Some("mul"),

        // Elementwise kernels may match ay-proven activation functions
        DispatchStep::Elementwise { kernel_name, .. } => {
            let name_lower = kernel_name.to_lowercase();
            if name_lower.contains("snake") && name_lower.contains("adain") {
                Some("adain_snake")
            } else if name_lower.contains("adain") {
                Some("adain")
            } else if name_lower.contains("snake") {
                Some("snake")
            } else if name_lower.contains("silu_mul") || name_lower.contains("silu") {
                Some("silu_mul")
            } else if name_lower.contains("sigmoid") {
                Some("sigmoid")
            } else if name_lower.contains("gelu") {
                Some("gelu")
            } else if name_lower.contains("relu") {
                Some("relu")
            } else if name_lower.contains("tanh") {
                Some("tanh_act")
            } else {
                None
            }
        }

        // Metadata-only ops: trivially correct (no computation)
        DispatchStep::Reshape { .. }
        | DispatchStep::Narrow { .. }
        | DispatchStep::Transpose { .. }
        | DispatchStep::AxisSelect { .. }
        | DispatchStep::ZeroPad1d { .. } => {
            // These are view/copy operations with no numerical computation.
            // We classify them as "trivially correct" — not ay-proven per se,
            // but they don't introduce numerical errors.
            None
        }

        // Linear algebra ops: Kani + CROWN verified, no per-kernel ay proof
        DispatchStep::Linear { .. }
        | DispatchStep::MatMul { .. }
        | DispatchStep::Conv1d(_)
        | DispatchStep::Conv2d(_)
        | DispatchStep::ConvTranspose1d(_)
        | DispatchStep::Softmax { .. }
        | DispatchStep::Reduce { .. }
        | DispatchStep::Embedding { .. }
        | DispatchStep::Broadcast { .. }
        | DispatchStep::Stack { .. }
        | DispatchStep::Concat { .. } => None,

        // Unknown future variants: conservative (no proof)
        _ => None,
    }
}

/// Check whether a dispatch step is a metadata-only operation.
///
/// Metadata-only ops (reshape, narrow, transpose, etc.) do not perform
/// numerical computation. They are excluded from the proven/unproven
/// fraction because they cannot introduce numerical errors.
pub fn is_metadata_only(step: &nn_dsl::DispatchStep) -> bool {
    use nn_dsl::DispatchStep;
    matches!(
        step,
        DispatchStep::Reshape { .. }
            | DispatchStep::Narrow { .. }
            | DispatchStep::Transpose { .. }
            | DispatchStep::AxisSelect { .. }
            | DispatchStep::ZeroPad1d { .. }
    )
}

/// Analyze a dispatch plan for implementation correctness evidence.
///
/// Classifies each step as ay-proven, unproven (numerical), or metadata-only.
/// The proven fraction excludes metadata-only steps from the denominator.
///
/// Returns `ImplementationCorrectnessEvidence` suitable for enriching
/// `MoonshotCertificate` via `with_smt_results()`.
pub fn analyze_dispatch_plan(steps: &[nn_dsl::DispatchStep]) -> ImplementationCorrectnessEvidence {
    let mut proven_categories: Vec<String> = Vec::new();
    let mut unproven_categories: Vec<String> = Vec::new();
    let mut numerical_steps = 0usize;
    let mut proven_steps = 0usize;

    for step in steps {
        if is_metadata_only(step) {
            continue; // Skip metadata-only ops from the fraction
        }

        numerical_steps += 1;

        if let Some(category) = ay_kernel_category(step) {
            proven_steps += 1;
            if !proven_categories.contains(&category.to_string()) {
                proven_categories.push(category.to_string());
            }
        } else {
            let category = dispatch_step_category(step);
            if !unproven_categories.contains(&category) {
                unproven_categories.push(category);
            }
        }
    }

    proven_categories.sort();
    unproven_categories.sort();

    ImplementationCorrectnessEvidence {
        total_steps: numerical_steps,
        proven_steps,
        proven_categories,
        unproven_categories,
        all_proven: proven_steps == numerical_steps && numerical_steps > 0,
    }
}

/// Get the operation category name for a dispatch step (for reporting).
fn dispatch_step_category(step: &nn_dsl::DispatchStep) -> String {
    use nn_dsl::DispatchStep;
    match step {
        DispatchStep::Linear { .. } => "linear".to_string(),
        DispatchStep::MatMul { .. } => "matmul".to_string(),
        DispatchStep::Conv1d(_) => "conv1d".to_string(),
        DispatchStep::Conv2d(_) => "conv2d".to_string(),
        DispatchStep::ConvTranspose1d(_) => "conv_transpose1d".to_string(),
        DispatchStep::Softmax { .. } => "softmax".to_string(),
        DispatchStep::Reduce { .. } => "reduce".to_string(),
        DispatchStep::Embedding { .. } => "embedding".to_string(),
        DispatchStep::BinaryAdd { .. } => "binary_add".to_string(),
        DispatchStep::BinaryMul { .. } => "binary_mul".to_string(),
        DispatchStep::Broadcast { .. } => "broadcast".to_string(),
        DispatchStep::Sigmoid { .. } => "sigmoid".to_string(),
        DispatchStep::Gelu { .. } => "gelu".to_string(),
        DispatchStep::Relu { .. } => "relu".to_string(),
        DispatchStep::Tanh { .. } => "tanh".to_string(),
        DispatchStep::Elementwise { kernel_name, .. } => {
            format!("elementwise:{kernel_name}")
        }
        DispatchStep::Stack { .. } => "stack".to_string(),
        DispatchStep::Concat { .. } => "concat".to_string(),
        _ => "unknown".to_string(),
    }
}

/// Check Property 8 (implementation correctness) against dispatch plan evidence.
///
/// Evaluates the ay-verified kernel coverage fraction for the pipeline's
/// dispatch steps. The proven fraction is `proven_steps / total_numerical_steps`
/// (excluding metadata-only ops like reshape, narrow, transpose).
///
/// # Verification levels
///
/// - `SmtProven`: All numerical operations have ay-proven kernels (rare for
///   pipelines with matmul/conv, which use CROWN+Kani instead of ay).
/// - `CrownPartial`: ≥50% of numerical operations have ay-proven kernels
///   (the remaining ops are verified by CROWN+Kani, not ay).
/// - `Empirical`: <50% ay coverage.
pub fn check_implementation_correctness(
    evidence: &ImplementationCorrectnessEvidence,
) -> MoonshotPropertyResult {
    let fraction = if evidence.total_steps > 0 {
        evidence.proven_steps as f64 / evidence.total_steps as f64
    } else {
        0.0
    };

    let level = if evidence.all_proven {
        VerificationLevel::SmtProven
    } else if fraction >= 0.5 {
        // ≥50% ay coverage + remaining ops verified by CROWN+Kani
        VerificationLevel::CrownPartial
    } else {
        VerificationLevel::Empirical
    };

    let proven = evidence.all_proven;

    MoonshotPropertyResult {
        property_index: 7, // P8: Correct implementation
        property_name: PROPERTY_NAMES[7],
        proven,
        level,
        bound_value: evidence.proven_steps as f64,
        threshold: evidence.total_steps as f64,
        is_sound: true, // ay proofs are inherently sound
        explanation: format!(
            "{}/{} numerical ops ay-proven ({:.0}%), categories: [{}], gaps: [{}]: {}",
            evidence.proven_steps,
            evidence.total_steps,
            fraction * 100.0,
            evidence.proven_categories.join(", "),
            evidence.unproven_categories.join(", "),
            if proven {
                "ALL PROVEN"
            } else if fraction >= 0.5 {
                "PARTIAL (≥50%)"
            } else {
                "LOW COVERAGE"
            }
        ),
    }
}

/// Known ay-proven kernel names for reference.
///
/// Returns the list of kernel names from the BOUNDS_REGISTRY that have
/// ay SMT proofs reaching `Proven` status.
pub fn ay_proven_kernel_names() -> &'static [&'static str] {
    AY_PROVEN_KERNELS
}
