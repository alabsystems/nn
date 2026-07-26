// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SMT property verification for kernels.
//!
//! Translates a `KernelDef` to a ay program, adds property assertions,
//! and produces an `SmtStatusRecord`.
//!
//! # Direct execution
//!
//! The `ay-bindings` `direct` feature enables in-process solving for
//! all kernel encodings:
//!
//! - **Exact linear** (pure arithmetic, no transcendentals): `QF_LRA`
//! - **UF approximation linear** (transcendental kernels like
//!   tanh, sigmoid, exp, softplus): `QF_UFLRA` (ay#5406, #2617)
//! - **Non-linear** (mul/div of two symbolic variables, powi): `ALL`
//!   logic triggers ay auto-detection, routing to ay's NRA theory solver
//!   (McCormick envelopes, sign lemmas, tangent plane lemmas) (#2640).
//!
//! Previously, non-linear programs returned `Unexecuted` (#203). With #2640,
//! all 17 BOUNDS_REGISTRY kernels are now submitted to ay for solving.
//!
//! # ay#5357 + ay#5605: RESOLVED
//!
//! ay#5357 (root cause ay#5405) fix landed in ay commit 6aac039. ay-direct
//! now correctly returns `Proven` and `Counterexample` for QF_LRA programs.
//!
//! ay#5605 fix landed: `real_mul` with fractional constant coefficients
//! (e.g., `real_mul(x, (/ 2000000 1000000))` from `real_from_f64` encoding)
//! now works correctly. All linear kernels — including those with `real_mul`
//! by constants — reach `Proven` via ay-direct. All test assertions are
//! strict `assert_eq!(outcome, SmtOutcome::Proven)`.
//!
//! **Identity elimination:** `translate_node.rs` eliminates `*1.0`, `+0.0`,
//! `-0.0`, `/1.0`, and `*0.0` at translation time. This is an optimization
//! (fewer SMT operations) but no longer required for correctness.
//!
//! # Phase B: UF direct execution (landed, #2617)
//!
//! ay#5406 added FuncApp support to ay direct execution. UF-approximated
//! kernels now use `QF_UFLRA` and are solved directly.
//!
//! # Phase C: NRA direct execution (landed, #2640)
//!
//! Removed the `uses_nonlinear` gate that prevented 10/17 BOUNDS_REGISTRY
//! kernels from reaching ay. Non-linear programs now use `ALL` logic for
//! ay auto-detection, routing to ay's NRA theory solver. ay-bindings `Logic`
//! enum lacks `QfNra`, so `QF_NRA` would misroute to LIRA — `ALL` bypasses
//! this via `StaticFeatures::infer_logic()`.

use nn_dsl::ir::KernelDef;

use crate::error::VerifyError;
use crate::graph::ParamBinding;
use crate::status::{BoundsSource, SmtEncodingKind, SmtStatusRecord};
use crate::verify_input::ScalarInputBounds;

// SmtOutcome used by #[path]-linked test sub-modules, not by production code.
#[cfg(test)]
use crate::status::SmtOutcome;

use super::error::SmtError;
use super::snake_uf;
use super::translate::translate_kernel;
use super::translate_real::real_from_f64;

// Execution dispatch extracted to prove_exec.rs (#557, 500-line limit).
#[path = "prove_exec.rs"]
mod prove_exec;
pub(super) use prove_exec::dispatch_query;

/// Result of building an SMT query from a kernel.
pub(super) struct SmtQuery {
    /// The ay program (used for direct execution when available).
    pub(super) program: ay_bindings::AYProgram,
    pub(super) smt2: String,
    pub(super) encoding: SmtEncodingKind,
    /// Whether the kernel contains non-linear operations (mul/div of two
    /// symbolic variables). Used by `try_direct_execution` to select `ALL`
    /// logic (ay auto-detection → NRA solver) instead of `QF_LRA`/`QF_UFLRA`
    /// (#2640). Previously gated direct execution entirely (#203).
    pub(super) uses_nonlinear: bool,
    /// Whether the output bounds were computed via the conservative ±1e6
    /// heuristic fallback rather than analytical bounds. Proofs against
    /// heuristic bounds are vacuous (#385) and should not produce
    /// `SmtOutcome::Proven`.
    pub(super) uses_heuristic_bounds: bool,
    /// Source of the expected output bounds (#383).
    pub(super) bounds_source: BoundsSource,
    /// The expected output bounds used in the query.
    pub(super) expected_bounds: (f64, f64),
}

/// Conservative margin to account for `real_from_f64` quantization error
/// in ground-folded kernel constants (#539).
///
/// When bounds are computed with exact arithmetic (f64 analytical or f32
/// NY IBP/CROWN) but the SMT kernel body uses `real_from_f64`-encoded
/// constants (denominator 1e6, max rounding error 5e-7 per constant), the SMT
/// kernel can produce outputs slightly beyond the exact bounds. This margin
/// widens bounds before encoding to keep the two precision domains consistent.
///
/// Applied to ALL bounds sources — Analytical, CallerProvided (NY),
/// and Heuristic — because the quantization error is in the ay kernel
/// encoding, not in the bounds computation (#906).
///
/// The margin is conservative: for a kernel with N ground-folded constants
/// and input magnitude M, the accumulated error is at most N * M * 5e-7.
/// Current worst case is rope_cos (2 constants, |x| ≤ 10): ~5.5e-6.
/// The 1e-4 margin provides ~18× headroom over the current worst case.
pub(super) const SMT_QUANTIZATION_MARGIN: f64 = 1e-4;

/// Validate output bounds, construct the violation assertion, and finalize
/// the SMT query from a translated kernel.
///
/// Shared by `build_smt_query` (single-variable) and `build_smt_query_multi`
/// (multi-variable). The caller is responsible for asserting input bounds
/// on `tr` before calling this function (#442).
///
/// Takes `tr` by value to avoid cloning `AYProgram` — callers do not use
/// `tr` after this call.
pub(super) fn finalize_query(
    mut tr: super::translate::TranslationResult,
    expected_output_bounds: Option<(f64, f64)>,
    heuristic_fallback: impl FnOnce() -> Result<(f64, f64, bool), VerifyError>,
) -> Result<SmtQuery, VerifyError> {
    let encoding = if tr.uses_uf_approx {
        SmtEncodingKind::UfApprox
    } else {
        SmtEncodingKind::Exact
    };

    let (expected_lower, expected_upper, uses_heuristic, bounds_source) =
        match expected_output_bounds {
            Some((lo, hi)) => {
                if !lo.is_finite() {
                    return Err(SmtError::NonFiniteLiteral(lo).into());
                }
                if !hi.is_finite() {
                    return Err(SmtError::NonFiniteLiteral(hi).into());
                }
                if lo > hi {
                    return Err(SmtError::InvertedBounds {
                        lower: lo,
                        upper: hi,
                    }
                    .into());
                }
                (lo, hi, false, BoundsSource::CallerProvided)
            }
            None => {
                let (lo, hi, heuristic) = heuristic_fallback()?;
                let source = if heuristic {
                    BoundsSource::Heuristic
                } else {
                    BoundsSource::Analytical
                };
                (lo, hi, heuristic, source)
            }
        };

    // Widen bounds by the quantization margin (#539, #906).
    // Ground-folded kernel constants are encoded via real_from_f64 with ~6
    // decimal digits, introducing up to 5e-7 rounding error per constant.
    // Accumulated through kernel arithmetic, the output error can reach ~1e-4.
    // Widening prevents spurious Counterexample results from this precision
    // mismatch. The error is in the ay kernel encoding, not in the bounds
    // source, so ALL bounds need widening — analytical, caller-provided
    // (NY), and heuristic alike (#906).
    let smt_lower = expected_lower - SMT_QUANTIZATION_MARGIN;
    let smt_upper = expected_upper + SMT_QUANTIZATION_MARGIN;

    let lo = real_from_f64(smt_lower)?;
    let hi = real_from_f64(smt_upper)?;
    let out_below = tr.output.clone().real_lt(lo);
    let out_above = tr.output.real_gt(hi);
    let violation = out_below.or(out_above);

    tr.program.assert(violation);
    tr.program.check_sat();

    let smt2 = tr.program.to_string();
    Ok(SmtQuery {
        program: tr.program,
        smt2,
        encoding,
        uses_nonlinear: tr.uses_nonlinear,
        uses_heuristic_bounds: uses_heuristic,
        bounds_source,
        expected_bounds: (expected_lower, expected_upper),
    })
}

/// Build the SMT-LIB2 query for a kernel's output-bounded property.
///
/// Shared implementation used by both `verify_kernel_smt` and `kernel_to_smt2`.
///
/// Single-variable convention: param 0 = Variable, params 1..N = Constant,
/// matching the NY `kernel_to_graph` convention (#448).
fn build_smt_query(
    kernel: &KernelDef,
    constant_params: &[f32],
    bounds: ScalarInputBounds,
    expected_output_bounds: Option<(f64, f64)>,
) -> Result<SmtQuery, VerifyError> {
    // Input bounds are validated at ScalarInputBounds construction (finite + ordered).

    // Build bindings matching NY convention: param 0 = Variable,
    // params 1..N = Constant. This matches kernel_to_graph in graph.rs:90-93.
    let mut bindings = vec![ParamBinding::Variable];
    for &val in constant_params {
        bindings.push(ParamBinding::Constant(val));
    }

    let mut tr = translate_kernel(kernel, &bindings)?;

    // Assert input bounds on variable parameters (only param 0 in single-var mode).
    for (i, expr) in tr.param_exprs.iter().enumerate() {
        if matches!(bindings.get(i), Some(ParamBinding::Variable)) {
            snake_uf::assert_input_bounds(
                &mut tr.program,
                expr,
                f64::from(bounds.lower()),
                f64::from(bounds.upper()),
            )?;
        }
    }

    // Delegate output bounds validation, violation assertion, and SMT-LIB2
    // generation to shared finalize_query (#442).
    // Closure borrows kernel/constant_params by reference — both outlive the
    // finalize_query call. lo/hi are Copy f32 values (#497 AC1).
    let lo = bounds.lower();
    let hi = bounds.upper();
    finalize_query(tr, expected_output_bounds, || {
        compute_output_bounds_heuristic(kernel, constant_params, lo, hi)
    })
}

// Per-kernel output bounds dispatch (compute_output_bounds_heuristic and prove_bounds)
// lives in prove_dispatch.rs to keep this file under 500 lines (#359).
#[path = "prove_dispatch.rs"]
mod prove_dispatch;
pub(in crate::ay) use prove_dispatch::compute_output_bounds_heuristic;

/// Verify a kernel's output-bounded property by translating to ay SMT.
///
/// Translates the kernel to Real arithmetic (with UF approximation for
/// transcendentals), asserts input bounds, adds the negated output-bounded
/// property, and either solves directly or reports translation success.
///
/// # Execution behavior
///
/// - **Exact linear encodings** (pure arithmetic, no transcendentals, no
///   mul/div of two symbolic variables): `QF_LRA`. Returns `Proven` (UNSAT)
///   or `Counterexample` (SAT).
/// - **UF approximation linear encodings** (transcendental kernels like
///   tanh, sigmoid, exp, softplus): `QF_UFLRA` (#2617, ay#5406). Returns
///   `Proven`, `Counterexample`, or `Unknown`.
/// - **Non-linear encodings** (mul/div of two symbolic variables): `ALL`
///   logic for ay auto-detection → NRA theory solver (#2640). Returns
///   `Proven`, `Counterexample`, `Unknown`, or `ExecutionFailed`.
#[must_use = "returns a Result that may contain an error"]
pub fn verify_kernel_smt(
    kernel: &KernelDef,
    constant_params: &[f32],
    bounds: ScalarInputBounds,
) -> Result<SmtStatusRecord, VerifyError> {
    verify_kernel_smt_with_bounds(kernel, constant_params, bounds, None)
}

/// Verify a kernel with explicit expected output bounds.
///
/// When `expected_output_bounds` is `Some((lo, hi))`, those bounds are used
/// directly instead of the internal heuristic. This is the intended entry
/// point for NY cross-verification: IBP computes output bounds,
/// then ay cross-checks them.
#[must_use = "returns a Result that may contain an error"]
pub fn verify_kernel_smt_with_bounds(
    kernel: &KernelDef,
    constant_params: &[f32],
    bounds: ScalarInputBounds,
    expected_output_bounds: Option<(f64, f64)>,
) -> Result<SmtStatusRecord, VerifyError> {
    let query = build_smt_query(kernel, constant_params, bounds, expected_output_bounds)?;
    Ok(dispatch_query(query))
}

/// Return the SMT-LIB2 text for a kernel verification query.
///
/// Useful for debugging, external solver invocation, or testing the
/// translation without solving.
#[must_use = "returns a Result that may contain an error"]
pub fn kernel_to_smt2(
    kernel: &KernelDef,
    constant_params: &[f32],
    bounds: ScalarInputBounds,
) -> Result<String, VerifyError> {
    let query = build_smt_query(kernel, constant_params, bounds, None)?;
    Ok(query.smt2)
}

/// Return the SMT-LIB2 text with explicit expected output bounds.
#[must_use = "returns a Result that may contain an error"]
pub fn kernel_to_smt2_with_bounds(
    kernel: &KernelDef,
    constant_params: &[f32],
    bounds: ScalarInputBounds,
    expected_output_bounds: (f64, f64),
) -> Result<String, VerifyError> {
    let query = build_smt_query(
        kernel,
        constant_params,
        bounds,
        Some(expected_output_bounds),
    )?;
    Ok(query.smt2)
}

#[cfg(test)]
#[path = "prove_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "prove_tests_execution.rs"]
mod execution_tests;

#[cfg(test)]
#[path = "prove_tests_silu.rs"]
mod silu_tests;

#[cfg(test)]
#[path = "prove_tests_counterexample.rs"]
mod counterexample_tests;

#[cfg(test)]
#[path = "prove_tests_finiteness.rs"]
mod finiteness_tests;

#[cfg(test)]
#[path = "prove_rope_tests.rs"]
mod rope_tests;

#[cfg(test)]
#[path = "prove_adain_tests.rs"]
mod adain_tests;

#[cfg(test)]
#[path = "prove_norm_tests.rs"]
mod norm_tests;

#[cfg(test)]
#[path = "prove_norm_ln_in_tests.rs"]
mod norm_ln_in_tests;

#[cfg(test)]
#[path = "prove_bounds_tests.rs"]
mod bounds_tests;

#[cfg(test)]
#[path = "bounds_cross_verify_tests.rs"]
mod bounds_cross_verify_tests;

#[cfg(test)]
#[path = "prove_tests_bounds_dispatch.rs"]
mod bounds_dispatch_tests;

#[cfg(test)]
#[path = "prove_tests_bounds_error_paths.rs"]
mod bounds_error_paths_tests;

#[cfg(test)]
#[path = "prove_tests_dispatch_query.rs"]
mod dispatch_query_tests;

#[cfg(test)]
#[path = "prove_istft_tests.rs"]
mod istft_tests;
