// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unary function translation for ay SMT encoding.
//!
//! Extracted from `translate_node.rs` (#1575) to keep files under 400 lines.
//! Contains `translate_unary_fn` (UF approximation for transcendentals)
//! and `eval_unary_ground` (ground-value folding for constant arguments).

use nn_dsl::ir::UnaryFnKind;
use ay_bindings::{Expr, Sort, AYProgram};

use super::super::error::SmtError;
use super::super::translate_uf::{apply_bounded_uf, apply_nonneg_uf, apply_positive_uf};

/// Compute the ground value of a unary function, if the argument is ground.
/// Returns `None` for symbolic args or if the result is non-finite.
pub(super) fn eval_unary_ground(op: UnaryFnKind, arg: Option<f64>) -> Option<f64> {
    let val = arg?;
    let result = match op {
        UnaryFnKind::Abs => val.abs(),
        UnaryFnKind::Recip => {
            if val == 0.0 {
                return None;
            }
            1.0 / val
        }
        UnaryFnKind::Sin => val.sin(),
        UnaryFnKind::Cos => val.cos(),
        UnaryFnKind::Exp => val.exp(),
        UnaryFnKind::Sqrt => {
            if val < 0.0 {
                return None;
            }
            val.sqrt()
        }
        UnaryFnKind::Rsqrt => {
            if val <= 0.0 {
                return None;
            }
            1.0 / val.sqrt()
        }
        UnaryFnKind::Tanh => val.tanh(),
        UnaryFnKind::Log => {
            if val <= 0.0 {
                return None;
            }
            val.ln()
        }
        // SAFETY: UnaryFnKind is #[non_exhaustive]. Skipping ground-value
        // folding for unknown variants is conservative — they fall through
        // to the UF approximation path in translate_unary_fn() which returns
        // UnsupportedOp for unknown ops (line 126).
        _ => return None,
    };
    if result.is_finite() {
        Some(result)
    } else {
        None
    }
}

/// Translate a unary math function using UF approximation.
///
/// Each transcendental (sin, cos, exp) is declared as an uninterpreted
/// function with axiomatic range constraints. Algebraic ops (abs, sqrt,
/// rsqrt, recip) that can be encoded exactly in Real arithmetic are
/// handled directly where possible.
pub(super) fn translate_unary_fn(
    op: UnaryFnKind,
    arg: Expr,
    program: &mut AYProgram,
    real_sort: &Sort,
    declared_ufs: &mut std::collections::HashSet<String>,
    uses_uf_approx: &mut bool,
) -> Result<Expr, SmtError> {
    match op {
        // Abs can be encoded exactly: ite(x >= 0, x, -x)
        UnaryFnKind::Abs => {
            let zero = Expr::real(0);
            let cond = arg.clone().real_ge(zero);
            let neg = arg.clone().real_neg();
            Ok(Expr::ite(cond, arg, neg))
        }

        // Recip: 1/x — guard against (/ 1 0) which is unspecified in SMT.
        UnaryFnKind::Recip => {
            program.assert(arg.clone().ne(Expr::real(0)));
            let one = Expr::real(1);
            Ok(one.real_div(arg))
        }

        // Transcendental and irrational functions → UF approximation
        UnaryFnKind::Sin => {
            *uses_uf_approx = true;
            apply_bounded_uf("sin_approx", arg, program, real_sort, declared_ufs, -1, 1)
        }
        UnaryFnKind::Cos => {
            *uses_uf_approx = true;
            apply_bounded_uf("cos_approx", arg, program, real_sort, declared_ufs, -1, 1)
        }
        UnaryFnKind::Exp => {
            // exp(x) > 0 for all x. No finite upper bound in general,
            // but we assert the positive range axiom.
            *uses_uf_approx = true;
            apply_positive_uf("exp_approx", arg, program, real_sort, declared_ufs)
        }
        UnaryFnKind::Sqrt => {
            // sqrt(x) >= 0 for x >= 0.
            // Domain precondition: sqrt is only defined for x >= 0 (#388).
            *uses_uf_approx = true;
            program.assert(arg.clone().real_ge(Expr::real(0)));
            apply_nonneg_uf("sqrt_approx", arg, program, real_sort, declared_ufs)
        }
        UnaryFnKind::Rsqrt => {
            // rsqrt(x) = 1/sqrt(x), positive for x > 0.
            // Domain precondition: rsqrt is only defined for x > 0 (#388).
            *uses_uf_approx = true;
            program.assert(arg.clone().real_gt(Expr::real(0)));
            apply_positive_uf("rsqrt_approx", arg, program, real_sort, declared_ufs)
        }

        UnaryFnKind::Tanh => {
            // tanh(x) in [-1, 1] for all x.
            *uses_uf_approx = true;
            apply_bounded_uf("tanh_approx", arg, program, real_sort, declared_ufs, -1, 1)
        }

        _ => Err(SmtError::UnsupportedOp {
            op_description: format!("UnaryFn {:?}", op),
        }),
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // -- CBMC transcendental stubs (f64) --
    // Kani/CBMC cannot model transcendental intrinsics. These nondeterministic
    // stubs provide sound over-approximations for safety proofs.
    // See: nn_engineering.md "CBMC transcendental stubs for Kani" (Source: #708).

    fn sin_f64_stub(_x: f64) -> f64 {
        let r: f64 = kani::any();
        kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
        r
    }

    fn cos_f64_stub(_x: f64) -> f64 {
        let r: f64 = kani::any();
        kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
        r
    }

    fn exp_f64_stub(x: f64) -> f64 {
        let r: f64 = kani::any();
        kani::assume(r.is_finite() && r > 0.0 && r <= 1e20);
        if x <= 0.0 {
            kani::assume(r <= 1.0);
        }
        if x > 0.0 {
            kani::assume(r > 1.0);
        }
        r
    }

    fn sqrt_f64_stub(x: f64) -> f64 {
        let r: f64 = kani::any();
        kani::assume(r.is_finite() && r >= 0.0 && r <= 1e20);
        if x > 0.0 {
            kani::assume(r > 0.0);
            kani::assume(r >= x.min(1.0));
        }
        if x >= 1.0 {
            kani::assume(r >= 1.0);
        }
        r
    }

    fn tanh_f64_stub(_x: f64) -> f64 {
        let r: f64 = kani::any();
        kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
        r
    }

    fn ln_f64_stub(_x: f64) -> f64 {
        let r: f64 = kani::any();
        kani::assume(r.is_finite() && r >= -100.0 && r <= 100.0);
        r
    }

    /// Proves `eval_unary_ground` always returns `None` when the input is `None`
    /// for all known UnaryFnKind variants (exhaustive over 9 handled ops).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::sin, sin_f64_stub)]
    #[kani::stub(f64::cos, cos_f64_stub)]
    #[kani::stub(f64::exp, exp_f64_stub)]
    #[kani::stub(f64::sqrt, sqrt_f64_stub)]
    #[kani::stub(f64::tanh, tanh_f64_stub)]
    #[kani::stub(f64::ln, ln_f64_stub)]
    fn eval_unary_ground_none_returns_none() {
        assert!(eval_unary_ground(UnaryFnKind::Abs, None).is_none());
        assert!(eval_unary_ground(UnaryFnKind::Sin, None).is_none());
        assert!(eval_unary_ground(UnaryFnKind::Cos, None).is_none());
        assert!(eval_unary_ground(UnaryFnKind::Exp, None).is_none());
        assert!(eval_unary_ground(UnaryFnKind::Sqrt, None).is_none());
        assert!(eval_unary_ground(UnaryFnKind::Rsqrt, None).is_none());
        assert!(eval_unary_ground(UnaryFnKind::Recip, None).is_none());
        assert!(eval_unary_ground(UnaryFnKind::Tanh, None).is_none());
        assert!(eval_unary_ground(UnaryFnKind::Log, None).is_none());
    }

    /// Proves `eval_unary_ground(Abs, _)` returns a non-negative value.
    /// |x| >= 0 for all real x.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::sin, sin_f64_stub)]
    #[kani::stub(f64::cos, cos_f64_stub)]
    #[kani::stub(f64::exp, exp_f64_stub)]
    #[kani::stub(f64::sqrt, sqrt_f64_stub)]
    #[kani::stub(f64::tanh, tanh_f64_stub)]
    #[kani::stub(f64::ln, ln_f64_stub)]
    fn eval_unary_ground_abs_nonneg() {
        let val: f64 = kani::any();
        kani::assume(val.is_finite());

        if let Some(r) = eval_unary_ground(UnaryFnKind::Abs, Some(val)) {
            assert!(r >= 0.0, "abs result must be non-negative");
        }
    }

    /// Proves `eval_unary_ground(Sin, _)` returns a value in [-1, 1].
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::sin, sin_f64_stub)]
    #[kani::stub(f64::cos, cos_f64_stub)]
    #[kani::stub(f64::exp, exp_f64_stub)]
    #[kani::stub(f64::sqrt, sqrt_f64_stub)]
    #[kani::stub(f64::tanh, tanh_f64_stub)]
    #[kani::stub(f64::ln, ln_f64_stub)]
    fn eval_unary_ground_sin_in_range() {
        let val: f64 = kani::any();
        kani::assume(val.is_finite());

        if let Some(r) = eval_unary_ground(UnaryFnKind::Sin, Some(val)) {
            assert!(r >= -1.0 && r <= 1.0, "sin must be in [-1, 1]");
        }
    }

    /// Proves `eval_unary_ground(Cos, _)` returns a value in [-1, 1].
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::sin, sin_f64_stub)]
    #[kani::stub(f64::cos, cos_f64_stub)]
    #[kani::stub(f64::exp, exp_f64_stub)]
    #[kani::stub(f64::sqrt, sqrt_f64_stub)]
    #[kani::stub(f64::tanh, tanh_f64_stub)]
    #[kani::stub(f64::ln, ln_f64_stub)]
    fn eval_unary_ground_cos_in_range() {
        let val: f64 = kani::any();
        kani::assume(val.is_finite());

        if let Some(r) = eval_unary_ground(UnaryFnKind::Cos, Some(val)) {
            assert!(r >= -1.0 && r <= 1.0, "cos must be in [-1, 1]");
        }
    }

    /// Proves `eval_unary_ground(Tanh, _)` returns a value in [-1, 1].
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::sin, sin_f64_stub)]
    #[kani::stub(f64::cos, cos_f64_stub)]
    #[kani::stub(f64::exp, exp_f64_stub)]
    #[kani::stub(f64::sqrt, sqrt_f64_stub)]
    #[kani::stub(f64::tanh, tanh_f64_stub)]
    #[kani::stub(f64::ln, ln_f64_stub)]
    fn eval_unary_ground_tanh_in_range() {
        let val: f64 = kani::any();
        kani::assume(val.is_finite());

        if let Some(r) = eval_unary_ground(UnaryFnKind::Tanh, Some(val)) {
            assert!(r >= -1.0 && r <= 1.0, "tanh must be in [-1, 1]");
        }
    }

    /// Proves `eval_unary_ground(Exp, _)` returns a positive value.
    /// exp(x) > 0 for all real x.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::sin, sin_f64_stub)]
    #[kani::stub(f64::cos, cos_f64_stub)]
    #[kani::stub(f64::exp, exp_f64_stub)]
    #[kani::stub(f64::sqrt, sqrt_f64_stub)]
    #[kani::stub(f64::tanh, tanh_f64_stub)]
    #[kani::stub(f64::ln, ln_f64_stub)]
    fn eval_unary_ground_exp_positive() {
        let val: f64 = kani::any();
        kani::assume(val.is_finite());

        if let Some(r) = eval_unary_ground(UnaryFnKind::Exp, Some(val)) {
            assert!(r > 0.0, "exp(x) > 0");
        }
    }

    /// Proves `eval_unary_ground(Sqrt, _)` rejects negative inputs and
    /// returns non-negative for non-negative inputs.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::sin, sin_f64_stub)]
    #[kani::stub(f64::cos, cos_f64_stub)]
    #[kani::stub(f64::exp, exp_f64_stub)]
    #[kani::stub(f64::sqrt, sqrt_f64_stub)]
    #[kani::stub(f64::tanh, tanh_f64_stub)]
    #[kani::stub(f64::ln, ln_f64_stub)]
    fn eval_unary_ground_sqrt_domain() {
        let val: f64 = kani::any();
        kani::assume(val.is_finite());

        let r = eval_unary_ground(UnaryFnKind::Sqrt, Some(val));
        if val < 0.0 {
            assert!(r.is_none(), "sqrt of negative must be None");
        } else if let Some(v) = r {
            assert!(v >= 0.0, "sqrt(x) >= 0");
        }
    }

    /// Proves `eval_unary_ground(Rsqrt, _)` rejects non-positive inputs.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::sin, sin_f64_stub)]
    #[kani::stub(f64::cos, cos_f64_stub)]
    #[kani::stub(f64::exp, exp_f64_stub)]
    #[kani::stub(f64::sqrt, sqrt_f64_stub)]
    #[kani::stub(f64::tanh, tanh_f64_stub)]
    #[kani::stub(f64::ln, ln_f64_stub)]
    fn eval_unary_ground_rsqrt_domain() {
        let val: f64 = kani::any();
        kani::assume(val.is_finite());
        kani::assume(val <= 0.0);

        let r = eval_unary_ground(UnaryFnKind::Rsqrt, Some(val));
        assert!(r.is_none(), "rsqrt of non-positive must be None");
    }

    /// Proves `eval_unary_ground(Recip, _)` rejects zero.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::sin, sin_f64_stub)]
    #[kani::stub(f64::cos, cos_f64_stub)]
    #[kani::stub(f64::exp, exp_f64_stub)]
    #[kani::stub(f64::sqrt, sqrt_f64_stub)]
    #[kani::stub(f64::tanh, tanh_f64_stub)]
    #[kani::stub(f64::ln, ln_f64_stub)]
    fn eval_unary_ground_recip_zero() {
        assert!(
            eval_unary_ground(UnaryFnKind::Recip, Some(0.0)).is_none(),
            "recip(0) must be None"
        );
        assert!(
            eval_unary_ground(UnaryFnKind::Recip, Some(-0.0)).is_none(),
            "recip(-0) must be None"
        );
    }

    /// Proves `eval_unary_ground(Log, _)` rejects non-positive inputs.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::stub(f64::sin, sin_f64_stub)]
    #[kani::stub(f64::cos, cos_f64_stub)]
    #[kani::stub(f64::exp, exp_f64_stub)]
    #[kani::stub(f64::sqrt, sqrt_f64_stub)]
    #[kani::stub(f64::tanh, tanh_f64_stub)]
    #[kani::stub(f64::ln, ln_f64_stub)]
    fn eval_unary_ground_log_domain() {
        let val: f64 = kani::any();
        kani::assume(val.is_finite());
        kani::assume(val <= 0.0);

        let r = eval_unary_ground(UnaryFnKind::Log, Some(val));
        assert!(r.is_none(), "log of non-positive must be None");
    }
}
