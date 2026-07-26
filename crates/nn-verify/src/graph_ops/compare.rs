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

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Proves Ge constant fold returns 1.0 when a >= b and 0.0 otherwise,
    /// matching the convention that positive = true for WhereLayer branching.
    #[kani::unwind(64)]
    #[kani::proof]
    fn compare_ge_constant_fold_correct() {
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        kani::assume(a.is_finite() && b.is_finite());

        let result = evaluate_constant_compare(CompareOpKind::Ge, a, b)
            .expect("Ge on finite values must not fail");
        if a >= b {
            assert_eq!(
                result.to_bits(),
                1.0f32.to_bits(),
                "Ge(a,b) with a >= b must be 1.0"
            );
        } else {
            assert_eq!(
                result.to_bits(),
                0.0f32.to_bits(),
                "Ge(a,b) with a < b must be 0.0"
            );
        }
    }

    /// Proves all 6 compare ops produce only 0.0 or 1.0 for any finite inputs.
    #[kani::unwind(64)]
    #[kani::proof]
    fn compare_constant_fold_output_is_binary() {
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        kani::assume(a.is_finite() && b.is_finite());

        // Test all 6 ops; CompareOpKind is non-exhaustive so we select from known set.
        let op_idx: u8 = kani::any();
        kani::assume(op_idx < 6);
        let op = match op_idx {
            0 => CompareOpKind::Gt,
            1 => CompareOpKind::Ge,
            2 => CompareOpKind::Lt,
            3 => CompareOpKind::Le,
            4 => CompareOpKind::Eq,
            _ => CompareOpKind::Ne,
        };

        let result = evaluate_constant_compare(op, a, b)
            .expect("known compare ops on finite values must not fail");
        assert!(
            result.to_bits() == 0.0f32.to_bits() || result.to_bits() == 1.0f32.to_bits(),
            "compare constant fold must return exactly 0.0 or 1.0"
        );
    }

    /// Proves Lt is the complement of Ge for all finite inputs:
    /// Lt(a,b) == 1.0 iff Ge(a,b) == 0.0, and vice versa.
    /// This verifies the duality of ordered comparisons critical for
    /// Select branching correctness.
    #[kani::unwind(64)]
    #[kani::proof]
    fn compare_lt_ge_complement() {
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        kani::assume(a.is_finite() && b.is_finite());

        let lt_result = evaluate_constant_compare(CompareOpKind::Lt, a, b)
            .expect("Lt on finite values must not fail");
        let ge_result = evaluate_constant_compare(CompareOpKind::Ge, a, b)
            .expect("Ge on finite values must not fail");
        // Lt and Ge are complements: exactly one is 1.0
        assert_eq!(
            (lt_result.to_bits() == 1.0f32.to_bits()) as u8
                + (ge_result.to_bits() == 1.0f32.to_bits()) as u8,
            1,
            "Lt and Ge must be complements: exactly one is 1.0"
        );
    }

    /// Proves Gt is the complement of Le for all finite inputs:
    /// Gt(a,b) == 1.0 iff Le(a,b) == 0.0, and vice versa.
    #[kani::unwind(64)]
    #[kani::proof]
    fn compare_gt_le_complement() {
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        kani::assume(a.is_finite() && b.is_finite());

        let gt_result = evaluate_constant_compare(CompareOpKind::Gt, a, b)
            .expect("Gt on finite values must not fail");
        let le_result = evaluate_constant_compare(CompareOpKind::Le, a, b)
            .expect("Le on finite values must not fail");
        assert_eq!(
            (gt_result.to_bits() == 1.0f32.to_bits()) as u8
                + (le_result.to_bits() == 1.0f32.to_bits()) as u8,
            1,
            "Gt and Le must be complements: exactly one is 1.0"
        );
    }

    /// Proves Eq and Ne are complements: Eq(a,b) == 1.0 iff Ne(a,b) == 0.0.
    #[kani::unwind(64)]
    #[kani::proof]
    fn compare_eq_ne_complement() {
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        kani::assume(a.is_finite() && b.is_finite());

        let eq_result = evaluate_constant_compare(CompareOpKind::Eq, a, b)
            .expect("Eq on finite values must not fail");
        let ne_result = evaluate_constant_compare(CompareOpKind::Ne, a, b)
            .expect("Ne on finite values must not fail");
        assert_eq!(
            (eq_result.to_bits() == 1.0f32.to_bits()) as u8
                + (ne_result.to_bits() == 1.0f32.to_bits()) as u8,
            1,
            "Eq and Ne must be complements: exactly one is 1.0"
        );
    }
}
