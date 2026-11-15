// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compare operation translation: constant folding and continuous approximations.

use ny_propagate::layers::{AbsLayer, MulConstantLayer};
use ny_propagate::{GraphNetwork, Layer};
use nn_dsl::ir::{BinOpKind, CompareOpKind};

use crate::error::VerifyError;
use crate::graph::{add_unary_node, checked_constant, NodeValue};

use super::translate_binop;

/// Evaluate a constant-constant comparison, returning 1.0 (true) or 0.0 (false).
pub(crate) fn evaluate_constant_compare(
    op: CompareOpKind,
    a: f32,
    b: f32,
) -> Result<f32, VerifyError> {
    let cond = match op {
        CompareOpKind::Gt => a > b,
        CompareOpKind::Ge => a >= b,
        CompareOpKind::Lt => a < b,
        CompareOpKind::Le => a <= b,
        CompareOpKind::Eq => a == b,
        CompareOpKind::Ne => a != b,
        _ => {
            return Err(VerifyError::UnsupportedOp(format!(
                "Compare {op:?} with constant operands"
            )))
        }
    };
    Ok(if cond { 1.0 } else { 0.0 })
}

/// Translate a Compare node. Constant-constant folds to 0/1; variable ordered
/// ops use subtraction (positive = true); Eq/Ne use abs-of-difference.
pub(crate) fn translate_compare(
    name: &str,
    op: CompareOpKind,
    lhs: &NodeValue,
    rhs: &NodeValue,
    graph: &mut GraphNetwork,
) -> Result<NodeValue, VerifyError> {
    match (lhs, rhs) {
        (NodeValue::Constant(a), NodeValue::Constant(b)) => {
            let result = evaluate_constant_compare(op, a.get(), b.get())?;
            checked_constant(result, &format!("Compare {op:?} constant fold"))
        }
        _ => {
            // Variable comparison: model as continuous approximation.
            // Positive result means condition is true for WhereLayer branching.
            match op {
                CompareOpKind::Gt | CompareOpKind::Ge => {
                    // Gt/Ge: lhs - rhs, positive when lhs > rhs
                    translate_binop(name, BinOpKind::Sub, lhs, rhs, graph)
                }
                CompareOpKind::Lt | CompareOpKind::Le => {
                    // Lt/Le: rhs - lhs, positive when lhs < rhs
                    translate_binop(name, BinOpKind::Sub, rhs, lhs, graph)
                }
                CompareOpKind::Ne => {
                    // Ne: abs(lhs - rhs), positive when lhs != rhs.
                    // Sound: zero only at exact equality (measure-zero boundary).
                    let diff_name = format!("{name}_diff");
                    translate_binop(&diff_name, BinOpKind::Sub, lhs, rhs, graph)?;
                    add_unary_node(name, Layer::Abs(AbsLayer::new()), &diff_name, graph);
                    Ok(NodeValue::Variable(name.to_string()))
                }
                CompareOpKind::Eq => {
                    // Eq: -(abs(lhs - rhs)), non-positive when lhs != rhs.
                    // Sound over-approximation: zero at exact equality (boundary case
                    // inherent to continuous modeling of discrete comparison).
                    let diff_name = format!("{name}_diff");
                    translate_binop(&diff_name, BinOpKind::Sub, lhs, rhs, graph)?;
                    let abs_name = format!("{name}_abs");
                    add_unary_node(&abs_name, Layer::Abs(AbsLayer::new()), &diff_name, graph);
                    add_unary_node(
                        name,
                        Layer::MulConstant(MulConstantLayer::scalar(-1.0)),
                        &abs_name,
                        graph,
                    );
                    Ok(NodeValue::Variable(name.to_string()))
                }
                _ => Err(VerifyError::UnsupportedOp(format!(
                    "Compare {op:?} with variable operands"
                ))),
            }
        }
    }
}
