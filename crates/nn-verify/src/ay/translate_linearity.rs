// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Non-linearity detection for translated ay programs.
//!
//! Determines whether a kernel translation contains non-linear operations
//! (e.g., multiplication of two symbolic variables) that ay's QF_LRA solver
//! cannot handle. Used by `prove.rs` to skip direct execution for NRA programs.

use nn_dsl::ir::{BinOpKind, IRNodeKind, KernelDef};

use crate::graph::ParamBinding;

/// Analyze a kernel's IR to detect non-linear operations.
///
/// Returns `true` if the kernel contains any multiplication or division where
/// both operands depend on symbolic variables, or `powi` with a symbolic base
/// and `|exp| > 1`.
///
/// `bindings` determines which parameters are constant (ground) vs symbolic.
pub(crate) fn kernel_uses_nonlinear(kernel: &KernelDef, bindings: &[ParamBinding]) -> bool {
    // Track which nodes are "ground" (contain no symbolic variables).
    let mut node_is_ground: Vec<bool> = Vec::with_capacity(kernel.nodes.len());
    let param_is_ground: Vec<bool> = (0..kernel.params.len())
        .map(|i| matches!(bindings.get(i), Some(ParamBinding::Constant(_))))
        .collect();

    for node in &kernel.nodes {
        let ground = node_is_ground_check(&node.kind, &node_is_ground, &param_is_ground);

        // Detect non-linear operations.
        match &node.kind {
            IRNodeKind::BinOp {
                op: BinOpKind::Mul | BinOpKind::Div,
                lhs,
                rhs,
            } => {
                let lhs_ground = node_is_ground.get(lhs.index()).copied().unwrap_or(false);
                let rhs_ground = node_is_ground.get(rhs.index()).copied().unwrap_or(false);
                if !lhs_ground && !rhs_ground {
                    return true;
                }
            }
            IRNodeKind::Powi { base, exp } if exp.unsigned_abs() > 1 => {
                if !node_is_ground.get(base.index()).copied().unwrap_or(false) {
                    return true;
                }
            }
            // These variants are known-linear (single-variable or structural):
            IRNodeKind::Param(_)
            | IRNodeKind::Literal(_)
            | IRNodeKind::BinOp { .. } // Add/Sub are linear; Mul/Div caught above
            | IRNodeKind::Compare { .. }
            | IRNodeKind::UnaryFn { .. }
            | IRNodeKind::Powi { .. } // exp <= 1 caught by guard above
            | IRNodeKind::Clamp { .. }
            | IRNodeKind::MinMax { .. }
            | IRNodeKind::Select { .. }
            | IRNodeKind::SumReduce { .. } => {}
            // SAFETY: IRNodeKind is #[non_exhaustive]. Unknown future variants
            // are conservatively assumed non-linear to prevent false `Proven`
            // results from ay's QF_LRA solver. Worst case: ay falls back to
            // heuristic bounds for a kernel that is actually linear.
            _ => {
                return true;
            }
        }

        node_is_ground.push(ground);
    }

    false
}

/// Check if a single IR node is "ground" (depends only on constants).
fn node_is_ground_check(
    kind: &IRNodeKind,
    node_is_ground: &[bool],
    param_is_ground: &[bool],
) -> bool {
    match kind {
        IRNodeKind::Param(idx) => param_is_ground.get(*idx).copied().unwrap_or(false),
        IRNodeKind::Literal(_) => true,
        IRNodeKind::BinOp { lhs, rhs, .. } => {
            node_is_ground.get(lhs.index()).copied().unwrap_or(false)
                && node_is_ground.get(rhs.index()).copied().unwrap_or(false)
        }
        IRNodeKind::Compare { lhs, rhs, .. } => {
            node_is_ground.get(lhs.index()).copied().unwrap_or(false)
                && node_is_ground.get(rhs.index()).copied().unwrap_or(false)
        }
        IRNodeKind::UnaryFn { input, .. } => {
            node_is_ground.get(input.index()).copied().unwrap_or(false)
        }
        IRNodeKind::Powi { base, .. } => node_is_ground.get(base.index()).copied().unwrap_or(false),
        IRNodeKind::Clamp { input, min, max } => {
            node_is_ground.get(input.index()).copied().unwrap_or(false)
                && node_is_ground.get(min.index()).copied().unwrap_or(false)
                && node_is_ground.get(max.index()).copied().unwrap_or(false)
        }
        IRNodeKind::MinMax { lhs, rhs, .. } => {
            node_is_ground.get(lhs.index()).copied().unwrap_or(false)
                && node_is_ground.get(rhs.index()).copied().unwrap_or(false)
        }
        IRNodeKind::Select {
            cond,
            then_val,
            else_val,
        } => {
            node_is_ground.get(cond.index()).copied().unwrap_or(false)
                && node_is_ground
                    .get(then_val.index())
                    .copied()
                    .unwrap_or(false)
                && node_is_ground
                    .get(else_val.index())
                    .copied()
                    .unwrap_or(false)
        }
        IRNodeKind::SumReduce { inputs } => inputs
            .iter()
            .all(|nid| node_is_ground.get(nid.index()).copied().unwrap_or(false)),
        // SAFETY: IRNodeKind is #[non_exhaustive]. Unknown future variants are
        // conservatively assumed non-ground (variable-dependent). This may cause
        // false negatives (treating a ground kernel as non-ground) but never
        // false positives (treating a symbolic kernel as ground).
        _ => false,
    }
}

#[cfg(test)]
#[path = "translate_linearity_tests.rs"]
mod tests;
