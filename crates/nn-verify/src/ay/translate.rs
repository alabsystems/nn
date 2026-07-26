// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Translate `KernelDef` IR nodes to ay `AYProgram` expressions.
//!
//! Uses Real arithmetic for all scalar operations. Transcendental ops
//! (sin, cos, exp, sqrt, rsqrt) are encoded as uninterpreted functions
//! with axiomatic range constraints.

use nn_dsl::ir::KernelDef;
use ay_bindings::{Expr, Sort, AYProgram};

use crate::graph::ParamBinding;

use super::error::SmtError;
use super::translate_node::translate_node;
use super::translate_real::real_from_f64;

// Re-import IR types used by test submodules (accessed via `use super::*`).
#[cfg(test)]
#[allow(unused_imports)]
use nn_dsl::ir::{BinOpKind, CompareOpKind, IRNodeKind, MinMaxKind, UnaryFnKind};

/// Result of translating a kernel to a ay program.
///
/// Contains the program (with assertions not yet added — the caller
/// adds property assertions), the output expression, and per-parameter
/// variable expressions for adding input-bound assumptions.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub(crate) struct TranslationResult {
    /// The ay program with variable/function declarations and UF axioms.
    pub program: AYProgram,
    /// Expression representing the kernel output.
    pub output: Expr,
    /// One `Expr` per kernel parameter (in parameter order).
    /// Variable parameters are `declare_const` Real symbols.
    /// Constant parameters are `Expr::real(value)` literals.
    pub param_exprs: Vec<Expr>,
    /// Whether any UF approximation was used (true → encoding is UfApprox).
    pub uses_uf_approx: bool,
    /// Whether any non-linear operation was used (mul/div of two symbolic variables).
    /// True means ay's QF_LRA solver cannot handle this program.
    pub uses_nonlinear: bool,
}

/// Translate a `KernelDef` to a ay `AYProgram` in Real arithmetic.
///
/// `bindings` maps each kernel parameter to `Variable` (symbolic) or
/// `Constant(val)` (ground), matching the NY convention used by
/// `kernel_to_graph_multi` in `graph.rs`. This eliminates the ambiguous
/// constant-first positional convention that previously reversed param
/// assignments (#448).
///
/// # Errors
///
/// Returns `SmtError::UnsupportedOp` for IR nodes that cannot be encoded
/// without UF approximation and UF is not applicable (none currently).
/// Returns `SmtError::NonFiniteLiteral` for NaN/Inf literals.
#[must_use = "returns a Result that may contain an error"]
pub(crate) fn translate_kernel(
    kernel: &KernelDef,
    bindings: &[ParamBinding],
) -> Result<TranslationResult, SmtError> {
    // Validate IR structure first — ensures all node references are in-bounds.
    kernel.validate()?;

    if kernel.params.is_empty() {
        return Err(SmtError::NoParameters);
    }

    // Validate bindings count matches kernel params.
    if bindings.len() != kernel.params.len() {
        return Err(SmtError::ParamCountMismatch {
            ir_count: kernel.params.len(),
            expected: kernel.params.len(),
            provided: bindings.len(),
        });
    }

    // Must have at least one symbolic variable (otherwise the solver has
    // nothing to explore — trivial result).
    let num_variables = bindings
        .iter()
        .filter(|b| matches!(b, ParamBinding::Variable))
        .count();
    if num_variables == 0 {
        return Err(SmtError::ParamCountMismatch {
            ir_count: kernel.params.len(),
            expected: kernel.params.len().saturating_sub(1),
            provided: kernel.params.len(),
        });
    }

    // Validate constant bindings are finite (matches NY path in graph.rs:134).
    for (i, binding) in bindings.iter().enumerate() {
        if let ParamBinding::Constant(val) = binding {
            if !val.is_finite() {
                return Err(SmtError::NonFiniteConstantParam {
                    index: i,
                    value: f64::from(*val),
                });
            }
        }
    }

    // Use QF_UFLRA: quantifier-free linear real arithmetic with
    // uninterpreted functions (for sin, cos, etc. approximations).
    // Non-linear operations (mul of two variables) may push this
    // beyond LRA, but ay handles NRA-like queries in direct mode.
    let mut program = AYProgram::new();
    program.set_logic("QF_UFNRA");
    program.produce_models();

    let real_sort = Sort::real();
    let mut uses_uf_approx = false;

    // Declare parameters: Variable params become symbolic consts,
    // Constant params become literal expressions. Binding position matches
    // the kernel's parameter order exactly — no reordering (#448).
    let mut param_exprs = Vec::with_capacity(kernel.params.len());
    for (i, param) in kernel.params.iter().enumerate() {
        match &bindings[i] {
            ParamBinding::Constant(val) => {
                param_exprs.push(real_from_f64(f64::from(*val))?);
            }
            ParamBinding::Variable => {
                let expr = program.declare_const(&param.name, real_sort.clone());
                param_exprs.push(expr);
            }
        }
    }

    // Track declared UF names to avoid re-declaring.
    let mut declared_ufs: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Detect non-linear operations before translation (extracted to translate_linearity.rs).
    let uses_nonlinear = super::translate_linearity::kernel_uses_nonlinear(kernel, bindings);

    // Track ground (constant) f64 values alongside SMT expressions (#376).
    // When a node is fully ground (depends only on constants), we can compute
    // its value at translation time and avoid UF approximation for
    // transcendentals applied to constant sub-expressions.
    let param_ground_values: Vec<Option<f64>> = bindings
        .iter()
        .map(|b| match b {
            ParamBinding::Constant(val) => Some(f64::from(*val)),
            ParamBinding::Variable => None,
        })
        .collect();

    // Translate each IR node in topological order.
    let mut node_exprs: Vec<Expr> = Vec::with_capacity(kernel.nodes.len());
    let mut node_ground_values: Vec<Option<f64>> = Vec::with_capacity(kernel.nodes.len());

    for node in &kernel.nodes {
        let (expr, ground_val) = translate_node(
            &node.kind,
            &node_exprs,
            &param_exprs,
            &node_ground_values,
            &param_ground_values,
            &mut program,
            &real_sort,
            &mut declared_ufs,
            &mut uses_uf_approx,
        )?;
        node_exprs.push(expr);
        node_ground_values.push(ground_val);
    }

    let output =
        node_exprs
            .get(kernel.output.index())
            .cloned()
            .ok_or(SmtError::IndexOutOfBounds {
                context: "kernel output",
                index: kernel.output.index(),
                length: node_exprs.len(),
            })?;

    Ok(TranslationResult {
        program,
        output,
        param_exprs,
        uses_uf_approx,
        uses_nonlinear,
    })
}

#[cfg(test)]
#[path = "translate_tests.rs"]
mod tests;
