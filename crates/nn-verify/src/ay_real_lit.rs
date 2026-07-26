// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Exact rational literals for `ay_bindings::Expr`.

use ay_bindings::Expr;

/// Exact rational literals for [`Expr`].
///
/// `ay-bindings` stores a `RealConst` as a `BigInt`, so it exposes no rational
/// constructor. A rational literal is instead the SMT-LIB term `(/ num den)`,
/// which the solver reads as an exact value. Building the same constant from an
/// `f64` would round it, and a rounded constant in a proof obligation makes the
/// discharged theorem the wrong one.
pub trait RealLit: Sized {
    /// The exact rational `num / den`.
    ///
    /// # Panics
    /// If `den` is zero.
    fn real_ratio(num: i64, den: i64) -> Self;
}

impl RealLit for Expr {
    fn real_ratio(num: i64, den: i64) -> Self {
        assert_ne!(den, 0, "real_ratio: zero denominator");
        Self::real(num).real_div(Self::real(den))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ay_bindings::execute_direct::{self, ExecuteResult};
    use ay_bindings::AYProgram;

    #[test]
    fn ratio_has_real_sort() {
        assert!(Expr::real_ratio(1, 2).sort().is_real());
        assert!(Expr::real_ratio(-1, 2).sort().is_real());
    }

    /// `1/3` must be the exact rational, not a rounded decimal: the solver has
    /// to prove `(1/3) * 3 = 1`, which fails for any float approximation.
    #[test]
    fn one_third_times_three_is_exactly_one() {
        let mut program = AYProgram::new();
        program.set_logic("QF_LRA");
        let third_times_three = Expr::real_ratio(1, 3).real_mul(Expr::real(3));
        // Assert the negation; UNSAT (`Verified`) means equality always holds.
        program.assert(third_times_three.eq(Expr::real(1)).not());
        program.check_sat();

        match execute_direct::execute(&program) {
            Ok(ExecuteResult::Verified) => {}
            other => panic!("(1/3)*3 == 1 should be Verified (UNSAT), got {other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "zero denominator")]
    fn zero_denominator_panics() {
        let _ = Expr::real_ratio(1, 0);
    }
}
