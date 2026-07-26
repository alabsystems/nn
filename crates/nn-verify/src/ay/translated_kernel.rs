// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Reusable translated kernel for incremental SMT verification.
//!
//! [`TranslatedKernel`] wraps a ay program containing the kernel translation
//! and input bounds, allowing multiple output-bounded property checks via
//! push/pop without re-translating. This is the foundation for multi-property
//! verification (output bounded, monotonicity, Lipschitz, etc.) on the same
//! translated kernel.
//!
//! # Push/Pop Protocol
//!
//! ```text
//! translate_kernel()   →  base program (declarations + UF axioms)
//! assert_input_bounds()→  input constraints added permanently
//! push()               →  assertion scope
//! assert(violation)    →  output bounds violation
//! check_sat()          →  solve
//! pop()                →  remove violation, ready for next property
//! ```
//!
//! # Usage
//!
//! ```ignore
//! // NOTE: ignore — requires ay kernel/bounds setup not available as standalone example
//! let mut tk = TranslatedKernel::from_kernel(kernel, &[1.0], bounds)?;
//! let r1 = tk.check_output_bounded((-5.0, 15.0))?;
//! let r2 = tk.check_output_bounded((-10.0, 20.0))?;
//! // Both checks reuse the same translation — no rebuild.
//! ```
//!
//! See issue #443 for design context.

use nn_dsl::ir::KernelDef;

use crate::error::VerifyError;
use crate::graph::ParamBinding;
use crate::status::{BoundsSource, SmtEncodingKind, SmtStatusRecord};
use crate::verify_input::ScalarInputBounds;

use super::error::SmtError;
use super::prove::{dispatch_query, SmtQuery, SMT_QUANTIZATION_MARGIN};
use super::snake_uf;
use super::translate::translate_kernel;
use super::translate_real::real_from_f64;

/// A kernel translated to ay with input bounds applied, ready for
/// incremental property checks via push/pop.
///
/// The ay program contains:
/// - Variable/constant declarations
/// - UF axiom constraints (sin/cos/exp range bounds)
/// - Input domain bounds (lower <= x <= upper)
///
/// Output-bounded property assertions are added inside push/pop scopes
/// so the base program can be reused across multiple checks.
pub struct TranslatedKernel {
    /// The ay program with translation + input bounds (no output assertions).
    program: ay_bindings::AYProgram,
    /// The kernel output expression.
    output: ay_bindings::Expr,
    /// Whether UF approximation was used (sin, cos, etc.).
    uses_uf_approx: bool,
    /// Whether the kernel has non-linear symbolic operations.
    uses_nonlinear: bool,
    /// Name of the kernel (for diagnostics).
    kernel_name: String,
}

impl TranslatedKernel {
    /// Create a `TranslatedKernel` from a kernel definition.
    ///
    /// Translates the kernel IR to ay Real arithmetic and asserts input bounds
    /// on variable parameters. The resulting program is ready for incremental
    /// property checks via [`check_output_bounded`](Self::check_output_bounded).
    ///
    /// Uses the standard single-variable convention: param 0 = Variable,
    /// params 1..N = Constant.
    pub fn from_kernel(
        kernel: &KernelDef,
        constant_params: &[f32],
        bounds: ScalarInputBounds,
    ) -> Result<Self, VerifyError> {
        let mut bindings = vec![ParamBinding::Variable];
        for &val in constant_params {
            bindings.push(ParamBinding::Constant(val));
        }

        let mut tr = translate_kernel(kernel, &bindings)?;

        // Assert input bounds on variable parameters.
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

        Ok(Self {
            output: tr.output,
            uses_uf_approx: tr.uses_uf_approx,
            uses_nonlinear: tr.uses_nonlinear,
            kernel_name: kernel.name.clone(),
            program: tr.program,
        })
    }

    /// Check the output-bounded property using push/pop.
    ///
    /// Adds the violation assertion (output < lo OR output > hi) inside a
    /// push/pop scope, solves, then pops to restore the program state.
    /// The kernel translation and input bounds are preserved for reuse.
    ///
    /// Callers must provide explicit expected output bounds. For the
    /// incremental API, bounds come from NY IBP or analytical
    /// computation — the heuristic fallback is not available here (use
    /// `verify_kernel_smt` for the single-shot path with heuristic support).
    pub fn check_output_bounded(
        &mut self,
        expected_output_bounds: (f64, f64),
    ) -> Result<SmtStatusRecord, VerifyError> {
        let (lo, hi) = expected_output_bounds;

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

        let encoding = if self.uses_uf_approx {
            SmtEncodingKind::UfApprox
        } else {
            SmtEncodingKind::Exact
        };

        // Widen bounds by SMT_QUANTIZATION_MARGIN (#906). The ay kernel
        // encoding uses real_from_f64 for constants (~5e-7 error per constant),
        // so the SMT kernel output can differ slightly from the exact output.
        // Without widening, NY CallerProvided bounds produce spurious
        // Counterexample results at the boundary.
        let smt_lo = lo - SMT_QUANTIZATION_MARGIN;
        let smt_hi = hi + SMT_QUANTIZATION_MARGIN;

        // Pre-encode bounds before push() to avoid leaving the program in a
        // pushed state if real_from_f64 fails (i64 overflow for |val| > 9.2e12).
        let lo_expr = real_from_f64(smt_lo)?;
        let hi_expr = real_from_f64(smt_hi)?;

        // Push a new assertion scope — everything below can be rolled back.
        self.program.push();

        // Add violation assertion: output < lo OR output > hi.
        let out_below = self.output.clone().real_lt(lo_expr);
        let out_above = self.output.clone().real_gt(hi_expr);
        let violation = out_below.or(out_above);

        self.program.assert(violation);
        self.program.check_sat();

        let smt2 = self.program.to_string();

        // Build the query for dispatch. Must clone the program since
        // dispatch_query may mutate it (e.g., set_logic for direct execution).
        let query = SmtQuery {
            program: self.program.clone(),
            smt2,
            encoding,
            uses_nonlinear: self.uses_nonlinear,
            uses_heuristic_bounds: false,
            bounds_source: BoundsSource::CallerProvided,
            expected_bounds: (lo, hi),
        };

        let result = dispatch_query(query);

        // Pop to restore program state for reuse.
        self.program.pop(1);

        Ok(result)
    }

    /// Return the SMT encoding kind (Exact or UfApprox).
    #[must_use]
    pub fn encoding(&self) -> SmtEncodingKind {
        if self.uses_uf_approx {
            SmtEncodingKind::UfApprox
        } else {
            SmtEncodingKind::Exact
        }
    }

    /// Whether the kernel uses non-linear operations.
    #[must_use]
    pub fn uses_nonlinear(&self) -> bool {
        self.uses_nonlinear
    }

    /// The kernel name.
    #[must_use]
    pub fn kernel_name(&self) -> &str {
        &self.kernel_name
    }
}

#[cfg(test)]
#[path = "translated_kernel_tests.rs"]
mod tests;
