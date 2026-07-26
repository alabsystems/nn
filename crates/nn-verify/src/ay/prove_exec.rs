// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SMT query execution dispatch — direct solver invocation and fallback logic.
//!
//! Extracted from `prove.rs` to keep it under the 500-line limit (#557).

use ay_bindings::execute_direct::{self, ExecuteResult};

use crate::error::VerifyError;
use crate::status::{SmtEncodingKind, SmtOutcome, SmtStatusRecord};

use super::SmtQuery;
use crate::ay::SmtError;

/// Dispatch an SMT query: handle heuristic bounds gate, attempt direct
/// execution, and produce the final `SmtStatusRecord`.
///
/// Shared by `verify_kernel_smt_with_bounds` (single-variable) and
/// `verify_kernel_smt_multi` (multi-variable). Both paths have identical
/// execution dispatch logic after query construction (#442).
pub(in crate::ay) fn dispatch_query(mut query: SmtQuery) -> SmtStatusRecord {
    // Heuristic bounds (±1e6 fallback) produce vacuous proofs (#385).
    if query.uses_heuristic_bounds {
        return SmtStatusRecord {
            solver: "ay".to_string(),
            encoding: query.encoding,
            property: "output_bounded".to_string(),
            outcome: SmtOutcome::Unexecuted,
            detail: Some(format!(
                "SMT-LIB2 generated ({} bytes), solver not invoked \
                 (heuristic ±1e6 fallback bounds — proof would be vacuous, #385)",
                query.smt2.len(),
            )),
            bounds_source: query.bounds_source,
            expected_bounds: Some(query.expected_bounds),
            proof_alethe: None,
            proof_verdict: None,
        };
    }

    // Try direct execution for ALL encodings (#2640).
    // Linear: QF_LRA (Exact) or QF_UFLRA (UfApprox) — as before (#2617).
    // Non-linear: ALL logic triggers ay auto-detection, which routes NRA
    // programs to ay's NRA theory solver (McCormick envelopes, sign lemmas,
    // tangent plane lemmas). This replaces the uses_nonlinear gate (#203).
    match try_direct_execution(&mut query) {
        Ok(record) => record,
        Err(e) => {
            // Direct execution failed (#395). NeedsFallback, solver error, etc.
            SmtStatusRecord {
                solver: "ay".to_string(),
                encoding: query.encoding,
                property: "output_bounded".to_string(),
                outcome: SmtOutcome::ExecutionFailed,
                detail: Some(format!(
                    "SMT-LIB2 generated ({} bytes), direct execution failed: {}",
                    query.smt2.len(),
                    e,
                )),
                bounds_source: query.bounds_source,
                expected_bounds: Some(query.expected_bounds),
                proof_alethe: None,
                proof_verdict: None,
            }
        }
    }
}

/// Attempt direct in-process solving of an SMT query.
///
/// Uses `ay_bindings::execute_direct::execute` to solve the program
/// without serialization. Maps `ExecuteResult` variants to `SmtStatusRecord`.
///
/// Logic routing (#2640):
/// - Linear `Exact` → `QF_LRA`
/// - Linear `UfApprox` → `QF_UFLRA`
/// - Non-linear (any encoding) → `ALL` for ay auto-detection, which routes
///   NRA programs to ay's NRA theory solver via `StaticFeatures::infer_logic()`.
///   Using `ALL` instead of `QF_NRA` because ay-bindings `Logic` enum lacks a
///   `QfNra` variant — `QF_NRA` would be misrouted to LIRA. `ALL` triggers
///   the auto-detection path which correctly identifies nonlinear real terms.
fn try_direct_execution(query: &mut SmtQuery) -> Result<SmtStatusRecord, VerifyError> {
    // Override logic in place — caller owns the query and does not need
    // the original logic after this call. Avoids a full AYProgram clone.
    let logic = if query.uses_nonlinear {
        // Non-linear: use ALL for ay auto-detection → routes to NRA solver (#2640).
        "ALL"
    } else {
        match query.encoding {
            SmtEncodingKind::UfApprox => "QF_UFLRA",
            SmtEncodingKind::Exact => "QF_LRA",
        }
    };
    query.program.set_logic(logic);

    match execute_direct::execute(&query.program) {
        Ok(ExecuteResult::Verified) => Ok(SmtStatusRecord {
            solver: "ay-direct".to_string(),
            encoding: query.encoding,
            property: "output_bounded".to_string(),
            outcome: SmtOutcome::Proven,
            detail: Some("direct execution: UNSAT (property holds for all inputs)".to_string()),
            bounds_source: query.bounds_source,
            expected_bounds: Some(query.expected_bounds),
            proof_alethe: None,
            proof_verdict: None,
        }),
        Ok(ExecuteResult::Counterexample { model, .. }) => Ok(SmtStatusRecord {
            solver: "ay-direct".to_string(),
            encoding: query.encoding,
            property: "output_bounded".to_string(),
            outcome: SmtOutcome::Counterexample,
            detail: Some(format!(
                "direct execution: SAT (counterexample: {:?})",
                model
            )),
            bounds_source: query.bounds_source,
            expected_bounds: Some(query.expected_bounds),
            proof_alethe: None,
            proof_verdict: None,
        }),
        Ok(ExecuteResult::Unknown(reason)) => Ok(SmtStatusRecord {
            solver: "ay-direct".to_string(),
            encoding: query.encoding,
            property: "output_bounded".to_string(),
            outcome: SmtOutcome::Unknown,
            detail: Some(format!("direct execution: unknown ({})", reason)),
            bounds_source: query.bounds_source,
            expected_bounds: Some(query.expected_bounds),
            proof_alethe: None,
            proof_verdict: None,
        }),
        Ok(ExecuteResult::NeedsFallback(reason)) => {
            // Direct execution can't handle this program — caller should
            // fall back to Unexecuted.
            Err(SmtError::SolverError {
                reason: format!("needs fallback: {}", reason),
            }
            .into())
        }
        Ok(other) => Err(SmtError::SolverError {
            reason: format!("unexpected ExecuteResult variant: {:?}", other),
        }
        .into()),
        Err(e) => Err(SmtError::SolverError {
            reason: e.to_string(),
        }
        .into()),
    }
}
