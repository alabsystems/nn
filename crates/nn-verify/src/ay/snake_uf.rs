// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Snake activation UF approximation helpers.
//!
//! `snake_output_bounds` was extracted to `crate::bounds::snake` (#859) so it
//! runs without the `ay-smt` feature flag. This module re-exports it and keeps
//! the ay-bindings-dependent `assert_input_bounds` here.

use super::error::SmtError;
use super::translate_real::real_from_f64;

// Re-export from always-available bounds module.
// Used by ay test files (bounds_cross_verify_tests, prove_tests_bounds_dispatch).
#[allow(unused_imports)]
pub(crate) use crate::bounds::snake_output_bounds;

/// Assert input-domain bounds on a variable expression.
///
/// Adds: `lower <= expr <= upper` to the program.
///
/// # Errors
///
/// Returns `SmtError::InvertedBounds` if `lower > upper`.
/// Returns `SmtError::NonFiniteLiteral` if bounds are NaN/Inf, or
/// `SmtError::ValueTooLargeForRealEncoding` if bounds overflow the i64 encoding.
pub(crate) fn assert_input_bounds(
    program: &mut ay_bindings::AYProgram,
    expr: &ay_bindings::Expr,
    lower: f64,
    upper: f64,
) -> Result<(), SmtError> {
    // NaN/Inf rejected by real_from_f64, but check ordering first since
    // NaN comparisons silently return false (IEEE 754, see #66).
    if !lower.is_finite() || !upper.is_finite() {
        // Let real_from_f64 produce the specific error.
        let _ = real_from_f64(lower)?;
        let _ = real_from_f64(upper)?;
    }
    if lower > upper {
        return Err(SmtError::InvertedBounds { lower, upper });
    }
    let lo = real_from_f64(lower)?;
    let hi = real_from_f64(upper)?;
    program.assert(expr.clone().real_ge(lo));
    program.assert(expr.clone().real_le(hi));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assert_input_bounds_inverted_rejected() {
        let mut program = ay_bindings::AYProgram::new();
        let sort = ay_bindings::Sort::real();
        let expr = program.declare_const("x", sort);
        let err = assert_input_bounds(&mut program, &expr, 10.0, -10.0).unwrap_err();
        assert!(
            matches!(err, SmtError::InvertedBounds { lower, upper } if lower == 10.0 && upper == -10.0),
            "inverted bounds should be rejected, got: {err}"
        );
    }

    #[test]
    fn test_assert_input_bounds_equal_accepted() {
        // lower == upper is valid (point constraint).
        let mut program = ay_bindings::AYProgram::new();
        let sort = ay_bindings::Sort::real();
        let expr = program.declare_const("x", sort);
        assert!(assert_input_bounds(&mut program, &expr, 5.0, 5.0).is_ok());
    }
}
