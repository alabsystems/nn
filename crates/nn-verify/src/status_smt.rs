// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SMT verification status types.
//!
//! Split from `status.rs` to keep file sizes manageable.
//! Re-exported from `status` and `lib.rs` so the public API is unchanged.

use serde::{Deserialize, Serialize};

/// How the SMT encoding represents transcendental operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum SmtEncodingKind {
    /// All operations are exactly modeled in the SMT logic.
    Exact,
    /// Transcendental/unsupported ops use uninterpreted function approximations
    /// with axiomatic range constraints (e.g. -1 <= sin(x) <= 1).
    UfApprox,
}

/// Outcome of an SMT verification query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum SmtOutcome {
    /// The property was proved to hold for all inputs in the declared domain.
    Proven,
    /// A counterexample was found violating the property.
    Counterexample,
    /// The solver could not determine satisfiability.
    Unknown,
    /// The SMT query was generated but NOT executed by a solver.
    ///
    /// This is distinct from `Unknown` (which means the solver ran but
    /// couldn't decide). `Unexecuted` means the solver was never invoked.
    /// Phase A of ay integration produces this outcome — the SMT-LIB2 text
    /// is well-formed but no solver backend is available.
    ///
    /// Also used when the solver is intentionally skipped (e.g., heuristic
    /// bounds producing vacuous proofs, #385).
    ///
    /// # Phase A → B transition
    ///
    /// This variant is eliminated when the ay `direct` feature is enabled,
    /// which requires the upstream `check_sat_with_details` API to be
    /// available (blocked on ay-bindings update, tracked in #141).
    Unexecuted,
    /// The solver was invoked but execution failed.
    ///
    /// This is distinct from `Unexecuted` (solver never invoked) and
    /// `Unknown` (solver ran to completion but couldn't decide). The
    /// `ExecutionFailed` variant means the solver was attempted but
    /// returned an error or required fallback (#395).
    ///
    /// The `detail` field contains the error message from the failed
    /// execution attempt.
    ExecutionFailed,
}

/// Verdict from an independently-checkable SMT proof artifact (e.g. Alethe).
///
/// Distinct from `SmtOutcome` which records the solver's claim. `SmtProofVerdict`
/// records whether the proof artifact itself was validated by a proof checker.
/// When `SmtOutcome::Proven` and `SmtProofVerdict::Verified` agree, the proof
/// is machine-checkable — not just solver-asserted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum SmtProofVerdict {
    /// The proof artifact was checked and is valid.
    Verified,
    /// The proof artifact was checked and found invalid.
    Invalid,
    /// A proof artifact was produced but has not been checked.
    Unchecked,
}

/// Source of the expected output bounds used in the SMT verification query.
///
/// Distinguishes meaningful tight proofs (analytical or caller-provided) from
/// vacuous proofs against the conservative ±1e6 heuristic fallback (#383).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum BoundsSource {
    /// Tight bounds computed analytically for this specific kernel.
    Analytical,
    /// Conservative ±1e6 heuristic (Phase A stopgap). Proofs are vacuous.
    Heuristic,
    /// Bounds provided by the caller (e.g., NY cross-verification).
    CallerProvided,
}

/// Record of a single SMT verification run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SmtStatusRecord {
    /// Solver name (e.g. "ay").
    pub solver: String,
    /// Whether the encoding uses exact or approximated transcendental ops.
    pub encoding: SmtEncodingKind,
    /// The property checked (e.g. "output_finite", "bound_check").
    pub property: String,
    /// Outcome of the SMT check.
    pub outcome: SmtOutcome,
    /// Optional detail string (counterexample, error message, etc.).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub detail: Option<String>,
    /// Source of the expected output bounds used in the SMT query (#383).
    #[serde(default = "default_bounds_source")]
    pub bounds_source: BoundsSource,
    /// The expected output bounds used in the SMT query, for auditability.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub expected_bounds: Option<(f64, f64)>,
    /// Alethe proof text from the solver (machine-checkable UNSAT proof).
    ///
    /// Populated when the solver produces an Alethe-format proof artifact.
    /// This is the raw proof text that a proof checker can validate independently
    /// of the solver. Phase 1 (ay#7502) captures this from ay's proof output.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub proof_alethe: Option<String>,
    /// Verdict from proof artifact validation.
    ///
    /// `Some(Verified)` means an independent checker confirmed the proof.
    /// `Some(Unchecked)` means a proof was produced but not yet validated.
    /// `None` means no proof artifact was generated (pre-Phase 1 records).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub proof_verdict: Option<SmtProofVerdict>,
}

impl SmtStatusRecord {
    /// Construct a new SMT status record with required fields.
    ///
    /// Optional fields (`detail`, `expected_bounds`, `proof_alethe`,
    /// `proof_verdict`) default to `None`.
    /// `bounds_source` defaults to `Heuristic`.
    #[must_use]
    pub fn new(
        solver: String,
        encoding: SmtEncodingKind,
        property: String,
        outcome: SmtOutcome,
    ) -> Self {
        Self {
            solver,
            encoding,
            property,
            outcome,
            detail: None,
            bounds_source: BoundsSource::Heuristic,
            expected_bounds: None,
            proof_alethe: None,
            proof_verdict: None,
        }
    }

    /// Construct a record for a failed SMT execution attempt.
    ///
    /// Used when the pipeline encounters an error after NY has
    /// already recorded its result but before ay completes. This makes
    /// partial entries distinguishable from "SMT never attempted" (`smt: None`)
    /// in the persisted status file (#481).
    #[must_use]
    pub fn execution_failed(reason: &str) -> Self {
        Self {
            solver: "ay".to_string(),
            encoding: SmtEncodingKind::UfApprox,
            property: "pipeline_failure".to_string(),
            outcome: SmtOutcome::ExecutionFailed,
            detail: Some(reason.to_string()),
            bounds_source: BoundsSource::Heuristic,
            expected_bounds: None,
            proof_alethe: None,
            proof_verdict: None,
        }
    }
}

fn default_bounds_source() -> BoundsSource {
    BoundsSource::Heuristic
}

#[cfg(test)]
#[path = "status_smt_tests.rs"]
mod tests;
