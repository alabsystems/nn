// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pretty-print a KernelDef as a human-readable IR dump.

use super::{BinOpKind, BinaryFnKind, CompareOpKind, IRNodeKind, KernelDef, MinMaxKind};

/// Pretty-print a KernelDef as a human-readable IR dump.
///
/// Format:
/// ```text
/// kernel snake(x: f32, alpha: f32) -> f32 {
///   %0 = param(x)
///   %1 = param(alpha)
///   %2 = mul(%1, %0)
///   %3 = sin(%2)
///   %4 = powi(%3, 2)
///   ...
///   return %N
/// }
/// ```
#[must_use]
pub fn ir_pretty_print(kernel: &KernelDef) -> String {
    let mut out = String::new();

    let params: Vec<String> = kernel
        .params
        .iter()
        .map(|p| format!("{}: {}", p.name, p.ty))
        .collect();
    out.push_str(&format!(
        "kernel {}({}) -> {} {{\n",
        kernel.name,
        params.join(", "),
        kernel.return_type,
    ));

    for node in &kernel.nodes {
        out.push_str(&format!("  %{} = ", node.id.index()));
        match &node.kind {
            IRNodeKind::Param(idx) => {
                out.push_str(&format!("param({})", kernel.params[*idx].name));
            }
            IRNodeKind::Literal(v) => {
                out.push_str(&format!("const({})", format_literal(*v)));
            }
            IRNodeKind::BinOp { op, lhs, rhs } => {
                out.push_str(&format!("{}(%{}, %{})", op_name(*op), lhs.0, rhs.0));
            }
            IRNodeKind::Compare { op, lhs, rhs } => {
                out.push_str(&format!("{}(%{}, %{})", compare_name(*op), lhs.0, rhs.0));
            }
            IRNodeKind::UnaryFn { op, input } => {
                out.push_str(&format!("{}(%{})", op, input.0));
            }
            IRNodeKind::Powi { base, exp } => {
                out.push_str(&format!("powi(%{}, {})", base.0, exp));
            }
            IRNodeKind::Clamp { input, min, max } => {
                out.push_str(&format!("clamp(%{}, %{}, %{})", input.0, min.0, max.0));
            }
            IRNodeKind::MinMax { op, lhs, rhs } => {
                let name = match op {
                    MinMaxKind::Min => "min",
                    MinMaxKind::Max => "max",
                };
                out.push_str(&format!("{}(%{}, %{})", name, lhs.0, rhs.0));
            }
            IRNodeKind::Select {
                cond,
                then_val,
                else_val,
            } => {
                out.push_str(&format!(
                    "select(%{}, %{}, %{})",
                    cond.0, then_val.0, else_val.0
                ));
            }
            IRNodeKind::SumReduce { inputs } => {
                let refs: Vec<String> = inputs.iter().map(|id| format!("%{}", id.0)).collect();
                out.push_str(&format!("sum_reduce({})", refs.join(", ")));
            }
            IRNodeKind::BinaryFn { op, lhs, rhs } => {
                let name = match op {
                    BinaryFnKind::Atan2 => "atan2",
                };
                out.push_str(&format!("{}(%{}, %{})", name, lhs.0, rhs.0));
            }
        }
        out.push('\n');
    }

    out.push_str(&format!("  return %{}\n", kernel.output.index()));
    out.push_str("}\n");
    out
}

fn op_name(op: BinOpKind) -> &'static str {
    match op {
        BinOpKind::Add => "add",
        BinOpKind::Sub => "sub",
        BinOpKind::Mul => "mul",
        BinOpKind::Div => "div",
    }
}

fn compare_name(op: CompareOpKind) -> &'static str {
    match op {
        CompareOpKind::Eq => "eq",
        CompareOpKind::Ne => "ne",
        CompareOpKind::Lt => "lt",
        CompareOpKind::Le => "le",
        CompareOpKind::Gt => "gt",
        CompareOpKind::Ge => "ge",
    }
}

#[must_use = "formatted literal string is computed but not used"]
pub(super) fn format_literal(v: f64) -> String {
    if v == v.floor() && v.abs() < 1e15 {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}
