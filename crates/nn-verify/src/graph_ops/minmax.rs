// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MinMax translation: ReLU decomposition for var/const, binary layers for var/var.

use ny_propagate::layers::{
    AddConstantLayer, MaxBinaryLayer, MinBinaryLayer, MulConstantLayer, ReLULayer, SubConstantLayer,
};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use nn_dsl::ir::MinMaxKind;

use crate::error::VerifyError;
use crate::graph::{add_unary_node, checked_constant, scalar_array, NodeValue};

/// Evaluate constant-constant min/max: returns `a.min(b)` or `a.max(b)`.
fn evaluate_constant_minmax(op: MinMaxKind, a: f32, b: f32) -> Result<f32, VerifyError> {
    match op {
        MinMaxKind::Min => Ok(a.min(b)),
        MinMaxKind::Max => Ok(a.max(b)),
        _ => Err(VerifyError::UnsupportedOp(format!("{op:?}"))),
    }
}

/// Translate `max(var, const)` or `min(var, const)` using ReLU decomposition.
///
/// - `max(x, c)` -> `relu(x - c) + c`  (special case: `max(x, 0) = relu(x)`)
/// - `min(x, c)` -> `c - relu(c - x)`  (special case: `min(x, 0) = -relu(-x)`)
fn translate_minmax_var_const(
    name: &str,
    op: MinMaxKind,
    var_name: &str,
    c: f32,
    graph: &mut GraphNetwork,
) -> Result<NodeValue, VerifyError> {
    match op {
        MinMaxKind::Max => {
            if c == 0.0 {
                // max(x, 0) = relu(x)
                add_unary_node(name, Layer::ReLU(ReLULayer::new()), var_name, graph);
            } else {
                // max(x, c) = relu(x - c) + c
                let sub_name = format!("{name}_sub");
                add_unary_node(
                    &sub_name,
                    Layer::SubConstant(SubConstantLayer::scalar(c)),
                    var_name,
                    graph,
                );
                let relu_name = format!("{name}_relu");
                add_unary_node(&relu_name, Layer::ReLU(ReLULayer::new()), &sub_name, graph);
                add_unary_node(
                    name,
                    Layer::AddConstant(AddConstantLayer::new(scalar_array(c)?)),
                    &relu_name,
                    graph,
                );
            }
            Ok(NodeValue::Variable(name.to_string()))
        }
        MinMaxKind::Min => {
            if c == 0.0 {
                // min(x, 0) = -relu(-x)
                let neg_name = format!("{name}_neg");
                add_unary_node(
                    &neg_name,
                    Layer::MulConstant(MulConstantLayer::scalar(-1.0)),
                    var_name,
                    graph,
                );
                let relu_name = format!("{name}_relu");
                add_unary_node(&relu_name, Layer::ReLU(ReLULayer::new()), &neg_name, graph);
                add_unary_node(
                    name,
                    Layer::MulConstant(MulConstantLayer::scalar(-1.0)),
                    &relu_name,
                    graph,
                );
            } else {
                // min(x, c) = c - relu(c - x)
                let sub_name = format!("{name}_sub");
                add_unary_node(
                    &sub_name,
                    Layer::SubConstant(SubConstantLayer::new_reverse(scalar_array(c)?)),
                    var_name,
                    graph,
                );
                let relu_name = format!("{name}_relu");
                add_unary_node(&relu_name, Layer::ReLU(ReLULayer::new()), &sub_name, graph);
                add_unary_node(
                    name,
                    Layer::SubConstant(SubConstantLayer::new_reverse(scalar_array(c)?)),
                    &relu_name,
                    graph,
                );
            }
            Ok(NodeValue::Variable(name.to_string()))
        }
        _ => Err(VerifyError::UnsupportedOp(format!("MinMax {op:?}"))),
    }
}

/// Translate a variable MinMax node to the appropriate NY layer(s).
pub(crate) fn translate_minmax(
    name: &str,
    op: MinMaxKind,
    lhs: &NodeValue,
    rhs: &NodeValue,
    graph: &mut GraphNetwork,
) -> Result<NodeValue, VerifyError> {
    match (lhs, rhs) {
        (NodeValue::Constant(a), NodeValue::Constant(b)) => {
            let result = evaluate_constant_minmax(op, a.get(), b.get())?;
            checked_constant(result, &format!("{op:?}({}, {})", a.get(), b.get()))
        }
        // Variable-constant (commutative: handle both orderings)
        (NodeValue::Variable(var_name), NodeValue::Constant(c))
        | (NodeValue::Constant(c), NodeValue::Variable(var_name)) => {
            translate_minmax_var_const(name, op, var_name, c.get(), graph)
        }
        // Both variable
        (NodeValue::Variable(a_name), NodeValue::Variable(b_name)) => {
            let layer = match op {
                MinMaxKind::Max => Layer::MaxBinary(MaxBinaryLayer),
                MinMaxKind::Min => Layer::MinBinary(MinBinaryLayer),
                _ => return Err(VerifyError::UnsupportedOp(format!("{op:?}"))),
            };
            graph.add_node(GraphNode::binary(
                name.to_string(),
                layer,
                a_name.clone(),
                b_name.clone(),
            ));
            Ok(NodeValue::Variable(name.to_string()))
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Proves `evaluate_constant_minmax(Max, a, b)` returns `a.max(b)` for finite inputs.
    #[kani::unwind(64)]
    #[kani::proof]
    fn minmax_max_constant_fold_correct() {
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        kani::assume(a.is_finite() && b.is_finite());

        let result = evaluate_constant_minmax(MinMaxKind::Max, a, b)
            .expect("Max of finite values must succeed");
        assert_eq!(
            result.to_bits(),
            a.max(b).to_bits(),
            "Max must be bit-exact"
        );
        assert!(result >= a && result >= b, "Max must be >= both operands");
    }

    /// Proves `evaluate_constant_minmax(Min, a, b)` returns `a.min(b)` for finite inputs.
    #[kani::unwind(64)]
    #[kani::proof]
    fn minmax_min_constant_fold_correct() {
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        kani::assume(a.is_finite() && b.is_finite());

        let result = evaluate_constant_minmax(MinMaxKind::Min, a, b)
            .expect("Min of finite values must succeed");
        assert_eq!(
            result.to_bits(),
            a.min(b).to_bits(),
            "Min must be bit-exact"
        );
        assert!(result <= a && result <= b, "Min must be <= both operands");
    }

    /// Proves Max is commutative: Max(a,b) == Max(b,a) for any finite inputs.
    #[kani::unwind(64)]
    #[kani::proof]
    fn minmax_max_commutative() {
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        kani::assume(a.is_finite() && b.is_finite());

        let r1 = evaluate_constant_minmax(MinMaxKind::Max, a, b).expect("Max must succeed");
        let r2 = evaluate_constant_minmax(MinMaxKind::Max, b, a).expect("Max must succeed");
        assert_eq!(
            r1.to_bits(),
            r2.to_bits(),
            "Max must be commutative: Max(a,b) == Max(b,a)"
        );
    }

    /// Proves Min is commutative: Min(a,b) == Min(b,a) for any finite inputs.
    #[kani::unwind(64)]
    #[kani::proof]
    fn minmax_min_commutative() {
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        kani::assume(a.is_finite() && b.is_finite());

        let r1 = evaluate_constant_minmax(MinMaxKind::Min, a, b).expect("Min must succeed");
        let r2 = evaluate_constant_minmax(MinMaxKind::Min, b, a).expect("Min must succeed");
        assert_eq!(
            r1.to_bits(),
            r2.to_bits(),
            "Min must be commutative: Min(a,b) == Min(b,a)"
        );
    }

    /// Proves Max(a, Min(a,b)) == a for finite inputs (absorption law).
    /// This is a fundamental lattice property of min/max that ensures
    /// combined min/max operations in ReLU decomposition are consistent.
    #[kani::unwind(64)]
    #[kani::proof]
    fn minmax_absorption_max_min() {
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        kani::assume(a.is_finite() && b.is_finite());

        let inner = evaluate_constant_minmax(MinMaxKind::Min, a, b).expect("Min must succeed");
        // inner is either a or b (both finite), so it's finite.
        let outer = evaluate_constant_minmax(MinMaxKind::Max, a, inner).expect("Max must succeed");
        assert_eq!(
            outer.to_bits(),
            a.to_bits(),
            "Max(a, Min(a,b)) must equal a (absorption law)"
        );
    }
}
