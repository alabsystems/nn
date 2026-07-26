// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-IRNodeKind ay translation: BinOp, Compare, UnaryFn, Powi, Clamp, MinMax, Select, SumReduce.
//!
//! Extracted from `translate.rs` to keep all files under 500 lines (#504).
//! Each match arm translates one `IRNodeKind` variant to ay Real arithmetic,
//! with ground-value tracking for constant-folding (#376).

use nn_dsl::ir::{BinOpKind, CompareOpKind, IRNodeKind, MinMaxKind};
use ay_bindings::{Expr, Sort, AYProgram};

use super::error::SmtError;
use super::translate_real::real_from_f64;
use super::translate_uf::translate_powi;

#[path = "translate_node_unary.rs"]
mod unary;
use unary::{eval_unary_ground, translate_unary_fn};

/// Get a node expression by index, returning an error instead of panicking.
pub(super) fn get_node(node_exprs: &[Expr], idx: usize) -> Result<Expr, SmtError> {
    node_exprs
        .get(idx)
        .cloned()
        .ok_or(SmtError::IndexOutOfBounds {
            context: "node_exprs",
            index: idx,
            length: node_exprs.len(),
        })
}

/// Get a param expression by index, returning an error instead of panicking.
pub(super) fn get_param(param_exprs: &[Expr], idx: usize) -> Result<Expr, SmtError> {
    param_exprs
        .get(idx)
        .cloned()
        .ok_or(SmtError::IndexOutOfBounds {
            context: "param_exprs",
            index: idx,
            length: param_exprs.len(),
        })
}

/// Get the ground value for a node index, if it is fully constant.
pub(super) fn get_node_ground(node_ground_values: &[Option<f64>], idx: usize) -> Option<f64> {
    node_ground_values.get(idx).copied().flatten()
}

/// Translate a single IR node to a ay expression, returning both the expression
/// and an optional ground f64 value for constant-folding (#376).
///
/// When a node is fully ground (depends only on constants), the ground value
/// enables downstream nodes to fold transcendental functions on constant
/// arguments into exact Real literals, avoiding UF approximation.
#[allow(clippy::too_many_arguments)]
pub(super) fn translate_node(
    kind: &IRNodeKind,
    node_exprs: &[Expr],
    param_exprs: &[Expr],
    node_ground_values: &[Option<f64>],
    param_ground_values: &[Option<f64>],
    program: &mut AYProgram,
    real_sort: &Sort,
    declared_ufs: &mut std::collections::HashSet<String>,
    uses_uf_approx: &mut bool,
) -> Result<(Expr, Option<f64>), SmtError> {
    match kind {
        IRNodeKind::Param(idx) => {
            let expr = get_param(param_exprs, *idx)?;
            let ground = param_ground_values.get(*idx).copied().flatten();
            Ok((expr, ground))
        }

        IRNodeKind::Literal(val) => {
            let expr = real_from_f64(*val)?;
            let ground = if val.is_finite() { Some(*val) } else { None };
            Ok((expr, ground))
        }

        IRNodeKind::BinOp { op, lhs, rhs } => {
            let l = get_node(node_exprs, lhs.index())?;
            let r = get_node(node_exprs, rhs.index())?;
            let l_g = get_node_ground(node_ground_values, lhs.index());
            let r_g = get_node_ground(node_ground_values, rhs.index());
            let ground = {
                match (l_g, r_g) {
                    (Some(lv), Some(rv)) => {
                        let result = match op {
                            BinOpKind::Add => Some(lv + rv),
                            BinOpKind::Sub => Some(lv - rv),
                            BinOpKind::Mul => Some(lv * rv),
                            BinOpKind::Div => {
                                if rv == 0.0 {
                                    None
                                } else {
                                    Some(lv / rv)
                                }
                            }
                            // SAFETY: BinOpKind is #[non_exhaustive]. Skipping constant
                            // folding for unknown variants is conservative — the symbolic
                            // SMT expression path (below) handles all variants, and the
                            // catch-all at line 144 returns UnsupportedOp for unknown ops.
                            _ => None,
                        };
                        result.filter(|v| v.is_finite())
                    }
                    // At least one operand is non-constant — skip constant folding.
                    _ => None,
                }
            };
            // Identity elimination: avoid emitting `real_mul(x, 1)` or
            // `real_add(x, 0)` which can trigger ay-direct solver
            // incompleteness on what should be trivially provable programs.
            // Ground-folding (#376) produces `*1.0` for rsqrt(1.0)=1.0,
            // `+0.0` for beta=0.0. Eliminating these lets instance_norm
            // and similar kernels reach Proven without ay#5605 fix.
            let expr = match op {
                BinOpKind::Add => match (l_g, r_g) {
                    (Some(0.0), _) => r,
                    (_, Some(0.0)) => l,
                    _ => l.real_add(r),
                },
                BinOpKind::Sub => match r_g {
                    Some(0.0) => l,
                    _ => l.real_sub(r),
                },
                BinOpKind::Mul => match (l_g, r_g) {
                    (Some(1.0), _) => r,
                    (_, Some(1.0)) => l,
                    (Some(0.0), _) => real_from_f64(0.0)?,
                    (_, Some(0.0)) => real_from_f64(0.0)?,
                    _ => l.real_mul(r),
                },
                BinOpKind::Div => {
                    match r_g {
                        Some(1.0) => l,
                        _ => {
                            // Guard: in SMT Real arithmetic, (/ x 0) evaluates to an
                            // unspecified value. Assert divisor != 0 so the solver
                            // cannot exploit this to "prove" unsound properties.
                            program.assert(r.clone().ne(Expr::real(0)));
                            l.real_div(r)
                        }
                    }
                }
                _ => {
                    return Err(SmtError::UnsupportedOp {
                        op_description: format!("BinOp {:?}", op),
                    })
                }
            };
            Ok((expr, ground))
        }

        IRNodeKind::Compare { op, lhs, rhs } => {
            let l = get_node(node_exprs, lhs.index())?;
            let r = get_node(node_exprs, rhs.index())?;
            // Compare nodes produce booleans, not reals — no ground f64.
            let expr = match op {
                CompareOpKind::Lt => l.real_lt(r),
                CompareOpKind::Le => l.real_le(r),
                CompareOpKind::Gt => l.real_gt(r),
                CompareOpKind::Ge => l.real_ge(r),
                CompareOpKind::Eq => l.eq(r),
                CompareOpKind::Ne => l.ne(r),
                _ => {
                    return Err(SmtError::UnsupportedOp {
                        op_description: format!("Compare {:?}", op),
                    })
                }
            };
            Ok((expr, None))
        }

        IRNodeKind::UnaryFn { op, input } => {
            let arg_ground = get_node_ground(node_ground_values, input.index());
            // Ground-fold (#376): when the argument is a known constant,
            // compute the function result at translation time and emit a
            // Real literal instead of UF approximation. This avoids
            // UfApprox encoding for kernels where transcendentals operate
            // only on constant sub-expressions (e.g., rsqrt in layer_norm,
            // instance_norm when x is the variable).
            if let Some(folded) = eval_unary_ground(*op, arg_ground) {
                match real_from_f64(folded) {
                    Ok(expr) => return Ok((expr, Some(folded))),
                    // If the folded value exceeds the Real encoding range
                    // (e.g., exp(30.0) ≈ 1.07e13 > 9.2e12 limit), fall
                    // through to the UF approximation path instead of
                    // failing the entire translation.
                    Err(SmtError::ValueTooLargeForRealEncoding(_)) => {}
                    Err(e) => return Err(e),
                }
            }
            let arg = get_node(node_exprs, input.index())?;
            let expr =
                translate_unary_fn(*op, arg, program, real_sort, declared_ufs, uses_uf_approx)?;
            Ok((expr, None))
        }

        IRNodeKind::Powi { base, exp } => {
            let base_ground = get_node_ground(node_ground_values, base.index());
            // Ground-fold powi on constant base.
            if let Some(base_val) = base_ground {
                let result = base_val.powi(*exp);
                if result.is_finite() {
                    match real_from_f64(result) {
                        Ok(expr) => return Ok((expr, Some(result))),
                        // Fall through to UF path if value exceeds encoding range.
                        Err(SmtError::ValueTooLargeForRealEncoding(_)) => {}
                        Err(e) => return Err(e),
                    }
                }
            }
            let b = get_node(node_exprs, base.index())?;
            let expr = translate_powi(b, *exp, program, real_sort, declared_ufs, uses_uf_approx)?;
            Ok((expr, None))
        }

        IRNodeKind::Clamp { input, min, max } => {
            // clamp(x, lo, hi) = ite(x < lo, lo, ite(x > hi, hi, x))
            let x = get_node(node_exprs, input.index())?;
            let lo = get_node(node_exprs, min.index())?;
            let hi = get_node(node_exprs, max.index())?;
            let x_g = get_node_ground(node_ground_values, input.index());
            let lo_g = get_node_ground(node_ground_values, min.index());
            let hi_g = get_node_ground(node_ground_values, max.index());
            let ground = match (x_g, lo_g, hi_g) {
                (Some(xv), Some(lov), Some(hiv)) => {
                    let r = xv.clamp(lov, hiv);
                    if r.is_finite() {
                        Some(r)
                    } else {
                        None
                    }
                }
                _ => None,
            };
            let below = x.clone().real_lt(lo.clone());
            let above = x.clone().real_gt(hi.clone());
            let inner = Expr::ite(above, hi, x);
            Ok((Expr::ite(below, lo, inner), ground))
        }

        IRNodeKind::MinMax { op, lhs, rhs } => {
            let l = get_node(node_exprs, lhs.index())?;
            let r = get_node(node_exprs, rhs.index())?;
            let l_g = get_node_ground(node_ground_values, lhs.index());
            let r_g = get_node_ground(node_ground_values, rhs.index());
            let ground = match (l_g, r_g, op) {
                (Some(a), Some(b), MinMaxKind::Min) => {
                    let r = a.min(b);
                    if r.is_finite() {
                        Some(r)
                    } else {
                        None
                    }
                }
                (Some(a), Some(b), MinMaxKind::Max) => {
                    let r = a.max(b);
                    if r.is_finite() {
                        Some(r)
                    } else {
                        None
                    }
                }
                // SAFETY: Catches non-constant operands and unknown #[non_exhaustive]
                // MinMaxKind variants. Skipping constant folding is conservative —
                // the symbolic dispatch at line 269 returns UnsupportedOp for unknown ops.
                _ => None,
            };
            let cond = match op {
                MinMaxKind::Min => l.clone().real_le(r.clone()),
                MinMaxKind::Max => l.clone().real_ge(r.clone()),
                _ => {
                    return Err(SmtError::UnsupportedOp {
                        op_description: format!("MinMax {:?}", op),
                    })
                }
            };
            Ok((Expr::ite(cond, l, r), ground))
        }

        IRNodeKind::Select {
            cond,
            then_val,
            else_val,
        } => {
            let c = get_node(node_exprs, cond.index())?;
            let t = get_node(node_exprs, then_val.index())?;
            let e = get_node(node_exprs, else_val.index())?;
            Ok((Expr::ite(c, t, e), None))
        }

        IRNodeKind::SumReduce { inputs } => {
            if inputs.is_empty() {
                return Ok((real_from_f64(0.0)?, Some(0.0)));
            }
            let mut acc = get_node(node_exprs, inputs[0].index())?;
            let mut ground_acc = get_node_ground(node_ground_values, inputs[0].index());
            for nid in &inputs[1..] {
                acc = acc.real_add(get_node(node_exprs, nid.index())?);
                let g = get_node_ground(node_ground_values, nid.index());
                ground_acc = match (ground_acc, g) {
                    (Some(a), Some(b)) => {
                        let r = a + b;
                        if r.is_finite() {
                            Some(r)
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
            }
            Ok((acc, ground_acc))
        }

        _ => Err(SmtError::UnsupportedOp {
            op_description: format!("{:?}", kind),
        }),
    }
}

#[cfg(test)]
#[path = "translate_node_tests.rs"]
mod tests;
