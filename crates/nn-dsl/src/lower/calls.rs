// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Method-call and function-call lowering for the kernel subset.
//!
//! Handles `.sin()`, `.cos()`, `.sqrt()`, `.rsqrt()`, `.exp()`, `.abs()`,
//! `.recip()`, `.powi(n)`, `.clamp(lo, hi)`, `.max(v)`, `.min(v)`,
//! and `sum_reduce([...])`.

use crate::ir::{IRNodeKind, MinMaxKind, NodeId, UnaryFnKind};

use super::{LowerError, Lowerer};

impl Lowerer {
    pub(super) fn lower_method_call(
        &mut self,
        mc: &syn::ExprMethodCall,
    ) -> Result<NodeId, LowerError> {
        let recv = self.lower_expr(&mc.receiver)?;
        let method = mc.method.to_string();

        if let Some(op) = UnaryFnKind::from_method_name(&method) {
            if !mc.args.is_empty() {
                return Err(LowerError::WrongArgCount {
                    method,
                    expected: 0,
                    got: mc.args.len(),
                });
            }
            return Ok(self.add_node(IRNodeKind::UnaryFn { op, input: recv }));
        }

        match method.as_str() {
            "powi" => {
                let exp = self.extract_i32_arg(&mc.args)?;
                Ok(self.add_node(IRNodeKind::Powi { base: recv, exp }))
            }
            "clamp" => {
                if mc.args.len() != 2 {
                    return Err(LowerError::WrongArgCount {
                        method,
                        expected: 2,
                        got: mc.args.len(),
                    });
                }
                let min = self.lower_expr(&mc.args[0])?;
                let max = self.lower_expr(&mc.args[1])?;
                Ok(self.add_node(IRNodeKind::Clamp {
                    input: recv,
                    min,
                    max,
                }))
            }
            "max" | "min" => {
                if mc.args.len() != 1 {
                    return Err(LowerError::WrongArgCount {
                        method,
                        expected: 1,
                        got: mc.args.len(),
                    });
                }
                let other = self.lower_expr(&mc.args[0])?;
                let op = if method == "max" {
                    MinMaxKind::Max
                } else {
                    MinMaxKind::Min
                };
                Ok(self.add_node(IRNodeKind::MinMax {
                    op,
                    lhs: recv,
                    rhs: other,
                }))
            }
            _ => Err(LowerError::UnsupportedMethod(method)),
        }
    }

    pub(super) fn lower_call(&mut self, call: &syn::ExprCall) -> Result<NodeId, LowerError> {
        let func_name = match call.func.as_ref() {
            syn::Expr::Path(path) => path
                .path
                .segments
                .last()
                .map(|segment| segment.ident.to_string()),
            _ => None,
        }
        .ok_or_else(|| LowerError::UnsupportedExpr("unsupported call target".to_string()))?;

        match func_name.as_str() {
            "sum_reduce" => {
                if call.args.len() != 1 {
                    return Err(LowerError::WrongArgCount {
                        method: func_name,
                        expected: 1,
                        got: call.args.len(),
                    });
                }

                let arr = match &call.args[0] {
                    syn::Expr::Array(arr) => arr,
                    _ => {
                        return Err(LowerError::UnsupportedExpr(
                            "sum_reduce expects an array literal argument".to_string(),
                        ));
                    }
                };

                if arr.elems.is_empty() {
                    return Err(LowerError::UnsupportedExpr(
                        "sum_reduce requires at least one element".to_string(),
                    ));
                }

                let mut inputs = Vec::with_capacity(arr.elems.len());
                for elem in &arr.elems {
                    inputs.push(self.lower_expr(elem)?);
                }
                Ok(self.add_node(IRNodeKind::SumReduce { inputs }))
            }
            _ => Err(LowerError::UnsupportedExpr(format!(
                "unsupported call expression: {func_name}"
            ))),
        }
    }

    pub(super) fn extract_i32_arg(
        &self,
        args: &syn::punctuated::Punctuated<syn::Expr, syn::token::Comma>,
    ) -> Result<i32, LowerError> {
        if args.len() != 1 {
            return Err(LowerError::WrongArgCount {
                method: "powi".to_string(),
                expected: 1,
                got: args.len(),
            });
        }
        match &args[0] {
            syn::Expr::Lit(el) => match &el.lit {
                syn::Lit::Int(i) => i.base10_parse().map_err(|_| LowerError::InvalidLiteral),
                _ => Err(LowerError::InvalidLiteral),
            },
            // Rust parses `-1` as Unary(Neg, Lit(Int(1))), not as a negative literal.
            syn::Expr::Unary(u) if matches!(u.op, syn::UnOp::Neg(_)) => match u.expr.as_ref() {
                syn::Expr::Lit(el) => match &el.lit {
                    syn::Lit::Int(i) => {
                        let positive: i32 =
                            i.base10_parse().map_err(|_| LowerError::InvalidLiteral)?;
                        positive.checked_neg().ok_or(LowerError::InvalidLiteral)
                    }
                    _ => Err(LowerError::InvalidLiteral),
                },
                _ => Err(LowerError::InvalidLiteral),
            },
            _ => Err(LowerError::InvalidLiteral),
        }
    }
}
