// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-variable ay SMT verification (#411).
//!
//! Extracted from `prove.rs` to keep it under 500 lines.
//! Provides `verify_kernel_smt_multi` for multi-variable kernels
//! with per-variable bounds.
//!
//! Uses shared `finalize_query` and `dispatch_query` from `prove.rs`
//! to avoid duplicating output bounds validation, violation assertion
//! construction, and execution dispatch logic (#442).

use nn_dsl::ir::KernelDef;

use crate::error::VerifyError;
use crate::graph::ParamBinding;
use crate::status::SmtStatusRecord;

use super::snake_uf;
use super::translate::translate_kernel;

use super::prove::{compute_output_bounds_heuristic, dispatch_query, finalize_query, SmtQuery};

/// Build an SMT query with per-variable bounds for multi-variable kernels.
///
/// `bindings` maps each kernel parameter to `Variable` or `Constant(val)`.
/// Passed directly to `translate_kernel` so positional binding is preserved
/// exactly — no reordering or constant extraction (#448).
/// `variable_bounds` provides `(lower, upper)` for each `Variable` binding,
/// in the order they appear in `bindings`.
fn build_smt_query_multi(
    kernel: &KernelDef,
    bindings: &[ParamBinding],
    variable_bounds: &[(f32, f32)],
    expected_output_bounds: Option<(f64, f64)>,
) -> Result<SmtQuery, VerifyError> {
    // Validate variable_bounds length matches the number of Variable bindings.
    // Without this, trailing variables would get no input bounds assertion,
    // making the SMT query unsound (unbounded variables).
    let num_variables = bindings
        .iter()
        .filter(|b| matches!(b, ParamBinding::Variable))
        .count();
    if variable_bounds.len() != num_variables {
        return Err(super::error::SmtError::VariableBoundsMismatch {
            num_variables,
            bounds_len: variable_bounds.len(),
        }
        .into());
    }

    let mut tr = translate_kernel(kernel, bindings)?;

    // Assert per-variable input bounds on symbolic parameters.
    let mut var_idx = 0;
    for (i, expr) in tr.param_exprs.iter().enumerate() {
        if matches!(bindings.get(i), Some(ParamBinding::Variable)) {
            let (lo, hi) = variable_bounds[var_idx];
            snake_uf::assert_input_bounds(&mut tr.program, expr, f64::from(lo), f64::from(hi))?;
            var_idx += 1;
        }
    }

    // Delegate output bounds validation, violation assertion, and SMT-LIB2
    // generation to shared finalize_query (#442).
    //
    // When no explicit output bounds are provided, dispatch through
    // compute_output_bounds_heuristic for analytical bounds (#514).
    // For single-variable kernels (the common case), extract the constant
    // params and variable bounds to match the single-variable convention.
    // Multi-variable kernels with >1 variable fall back to ±1e6 (#385).
    let constant_params: Vec<f32> = bindings
        .iter()
        .filter_map(|b| match b {
            ParamBinding::Constant(v) => Some(*v),
            ParamBinding::Variable => None,
        })
        .collect();
    let num_vars = variable_bounds.len();
    // Check that the single Variable is at index 0 (#448 convention).
    // compute_output_bounds_heuristic expects constant_params[i] = kernel param i+1,
    // which only holds when Variable is at position 0 and all Constants follow.
    let var_at_zero = num_vars == 1
        && bindings
            .first()
            .is_some_and(|b| matches!(b, ParamBinding::Variable));
    finalize_query(tr, expected_output_bounds, || {
        if var_at_zero {
            let (lo, hi) = variable_bounds[0];
            compute_output_bounds_heuristic(kernel, &constant_params, lo, hi)
        } else {
            // Multi-variable or non-standard binding order: analytical bounds
            // functions only handle #448 variable-first convention.
            // Fall back to ±1e6 (#385).
            Ok((-1e6, 1e6, true))
        }
    })
}

/// Verify a multi-variable kernel's output-bounded property via ay SMT.
///
/// Like [`super::verify_kernel_smt_with_bounds`] but accepts per-variable
/// bindings and bounds instead of a single `ScalarInputBounds`. This enables
/// the unified multi-variable pipeline (#411).
#[must_use = "returns a Result that may contain an error"]
pub fn verify_kernel_smt_multi(
    kernel: &KernelDef,
    bindings: &[ParamBinding],
    variable_bounds: &[(f32, f32)],
    expected_output_bounds: Option<(f64, f64)>,
) -> Result<SmtStatusRecord, VerifyError> {
    let query = build_smt_query_multi(kernel, bindings, variable_bounds, expected_output_bounds)?;
    Ok(dispatch_query(query))
}

#[cfg(test)]
#[path = "prove_multi_tests.rs"]
mod tests;
