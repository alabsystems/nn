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
