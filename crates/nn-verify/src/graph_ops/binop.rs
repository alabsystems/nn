// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Binary operation translation: Add, Sub, Mul, Div with constant folding.

use ny_propagate::layers::{
    AddConstantLayer, AddLayer, Atan2Layer, DivConstantLayer, DivLayer, MulBinaryLayer,
    MulConstantLayer, ReciprocalLayer, SubConstantLayer, SubLayer,
};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use nn_dsl::ir::{BinOpKind, BinaryFnKind};

use crate::error::VerifyError;
use crate::graph::{add_unary_node, checked_constant, scalar_array, NodeValue};

pub(crate) fn translate_binop(
    name: &str,
    op: BinOpKind,
    lhs: &NodeValue,
    rhs: &NodeValue,
    graph: &mut GraphNetwork,
) -> Result<NodeValue, VerifyError> {
    match (lhs, rhs) {
        // Both constant: evaluate immediately
        (NodeValue::Constant(a), NodeValue::Constant(b)) => {
            let (a, b) = (a.get(), b.get());
            let result = evaluate_constant_binop(op, a, b)?;
            checked_constant(result, &format!("{a} {op:?} {b}"))
        }

        // Variable op Constant
        (NodeValue::Variable(var_name), NodeValue::Constant(c)) => {
            translate_var_const(name, op, var_name, c.get(), graph)
        }

        // Constant op Variable
        (NodeValue::Constant(c), NodeValue::Variable(var_name)) => {
            translate_const_var(name, op, c.get(), var_name, graph)
        }

        // Both variable
        (NodeValue::Variable(a_name), NodeValue::Variable(b_name)) => {
            translate_var_var(name, op, a_name, b_name, graph)
        }
    }
}

fn evaluate_constant_binop(op: BinOpKind, lhs: f32, rhs: f32) -> Result<f32, VerifyError> {
    match op {
        BinOpKind::Add => Ok(lhs + rhs),
        BinOpKind::Sub => Ok(lhs - rhs),
        BinOpKind::Mul => Ok(lhs * rhs),
        BinOpKind::Div => {
            if rhs == 0.0 {
                return Err(VerifyError::InternalTranslationError {
                    context: "constant division by zero".to_string(),
                });
            }
            Ok(lhs / rhs)
        }
        _ => Err(unsupported_binop(op)),
    }
}

fn translate_var_const(
    name: &str,
    op: BinOpKind,
    var_name: &str,
    c: f32,
    graph: &mut GraphNetwork,
) -> Result<NodeValue, VerifyError> {
    let layer = match op {
        BinOpKind::Add => Layer::AddConstant(AddConstantLayer::new(scalar_array(c)?)),
        BinOpKind::Sub => Layer::SubConstant(SubConstantLayer::scalar(c)),
        BinOpKind::Mul => Layer::MulConstant(MulConstantLayer::scalar(c)),
        BinOpKind::Div => {
            if c == 0.0 {
                return Err(VerifyError::InternalTranslationError {
                    context: format!(
                        "division by constant zero at node `{name}`: \
                         var / 0.0 would produce non-finite bounds during propagation"
                    ),
                });
            }
            Layer::DivConstant(DivConstantLayer::scalar(c))
        }
        _ => return Err(unsupported_binop(op)),
    };
    add_unary_node(name, layer, var_name, graph);
    Ok(NodeValue::Variable(name.to_string()))
}

fn translate_const_var(
    name: &str,
    op: BinOpKind,
    c: f32,
    var_name: &str,
    graph: &mut GraphNetwork,
) -> Result<NodeValue, VerifyError> {
    if matches!(op, BinOpKind::Div) {
        // c / var: reciprocal(var) * c
        let recip_name = format!("{name}_recip");
        add_unary_node(
            &recip_name,
            Layer::Reciprocal(ReciprocalLayer::new()),
            var_name,
            graph,
        );
        add_unary_node(
            name,
            Layer::MulConstant(MulConstantLayer::scalar(c)),
            &recip_name,
            graph,
        );
        return Ok(NodeValue::Variable(name.to_string()));
    }

    let layer = match op {
        // c + var = var + c (commutative)
        BinOpKind::Add => Layer::AddConstant(AddConstantLayer::new(scalar_array(c)?)),
        // c - var: use SubConstant reverse mode
        BinOpKind::Sub => Layer::SubConstant(SubConstantLayer::new_reverse(scalar_array(c)?)),
        // c * var = var * c (commutative)
        BinOpKind::Mul => Layer::MulConstant(MulConstantLayer::scalar(c)),
        _ => return Err(unsupported_binop(op)),
    };
    add_unary_node(name, layer, var_name, graph);
    Ok(NodeValue::Variable(name.to_string()))
}

fn translate_var_var(
    name: &str,
    op: BinOpKind,
    lhs_name: &str,
    rhs_name: &str,
    graph: &mut GraphNetwork,
) -> Result<NodeValue, VerifyError> {
    let layer = match op {
        BinOpKind::Add => Layer::Add(AddLayer),
        BinOpKind::Sub => Layer::Sub(SubLayer),
        BinOpKind::Mul => Layer::MulBinary(MulBinaryLayer),
        BinOpKind::Div => Layer::Div(DivLayer),
        _ => return Err(unsupported_binop(op)),
    };
    graph.add_node(GraphNode::binary(
        name.to_string(),
        layer,
        lhs_name.to_string(),
        rhs_name.to_string(),
    ));
    Ok(NodeValue::Variable(name.to_string()))
}

/// Translate a binary function call (e.g., `atan2(y, x)`) to NY.
///
/// Handles `BinaryFnKind` variants. Constant-folds when both inputs are constant.
pub(crate) fn translate_binary_fn(
    name: &str,
    op: BinaryFnKind,
    lhs: &NodeValue,
    rhs: &NodeValue,
    graph: &mut GraphNetwork,
) -> Result<NodeValue, VerifyError> {
    match (lhs, rhs) {
        (NodeValue::Constant(a), NodeValue::Constant(b)) => {
            let result = match op {
                BinaryFnKind::Atan2 => f64::from(a.get()).atan2(f64::from(b.get())) as f32,
                _ => return Err(VerifyError::UnsupportedOp(format!("BinaryFn {op:?}"))),
            };
            checked_constant(result, &format!("{op:?}({}, {})", a.get(), b.get()))
        }
        (NodeValue::Variable(a_name), NodeValue::Variable(b_name)) => {
            let layer = match op {
                BinaryFnKind::Atan2 => Layer::Atan2(Atan2Layer),
                _ => return Err(VerifyError::UnsupportedOp(format!("BinaryFn {op:?}"))),
            };
            graph.add_node(GraphNode::binary(
                name.to_string(),
                layer,
                a_name.clone(),
                b_name.clone(),
            ));
            Ok(NodeValue::Variable(name.to_string()))
        }
        // Mixed constant/variable: emit binary node with constant as AddConstant(0)+const.
        // For Atan2 this is rare in practice — both STFT inputs are variable activations.
        _ => Err(VerifyError::UnsupportedOp(format!(
            "BinaryFn {op:?} with mixed constant/variable inputs"
        ))),
    }
}

fn unsupported_binop(op: BinOpKind) -> VerifyError {
    VerifyError::UnsupportedOp(format!("BinOp {op:?}"))
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Proves `evaluate_constant_binop` rejects division by zero for all finite lhs.
    /// Uses `unwind(8)` to bound loop unwinding depth — CBMC otherwise diverges
    /// unwinding `syn::error::ErrorMessage` Drop impl (pulled in via nn-dsl → syn).
    /// See #608.
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn binop_div_by_zero_rejected() {
        let lhs: f32 = kani::any();
        kani::assume(lhs.is_finite());

        let result = evaluate_constant_binop(BinOpKind::Div, lhs, 0.0);
        assert!(result.is_err(), "division by zero must return Err");
    }

    /// Proves `evaluate_constant_binop(Add)` returns the correct value (lhs + rhs)
    /// for any finite inputs, verified bit-exact.
    #[kani::unwind(64)]
    #[kani::proof]
    fn binop_add_correct() {
        let lhs: f32 = kani::any();
        let rhs: f32 = kani::any();
        kani::assume(lhs.is_finite());
        kani::assume(rhs.is_finite());

        let result = evaluate_constant_binop(BinOpKind::Add, lhs, rhs);
        let val = result.expect("Add of finite values must not return Err");
        assert_eq!(
            val.to_bits(),
            (lhs + rhs).to_bits(),
            "Add result must be bit-exact"
        );
    }

    /// Proves `evaluate_constant_binop(Mul)` is bit-exact with direct multiplication
    /// for a representative set of f32 values chosen by symbolic index.
    ///
    /// CBMC cannot handle fully-symbolic f32 bitvector multiplication (SAT solver
    /// generates 30k+ clauses, 600s+ timeout). This harness uses symbolic index
    /// selection from 8 representative values covering: zero, positive/negative,
    /// small/large, boundary. Proves bit-exactness for 64 value pairs.
    /// Uses `unwind(8)` — syn::ErrorMessage Drop unwinding (#608).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn binop_mul_correct() {
        // Representative f32 values covering zero, sign, magnitude, and boundary.
        const VALS: [f32; 8] = [0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0, f32::MIN_POSITIVE];
        let i: usize = kani::any();
        let j: usize = kani::any();
        kani::assume(i < VALS.len());
        kani::assume(j < VALS.len());

        let lhs = VALS[i];
        let rhs = VALS[j];

        let result = evaluate_constant_binop(BinOpKind::Mul, lhs, rhs);
        let val = result.expect("Mul of finite values must not return Err");
        assert_eq!(
            val.to_bits(),
            (lhs * rhs).to_bits(),
            "Mul result must be bit-exact"
        );
    }

    /// Proves `evaluate_constant_binop(Sub)` returns the correct value (lhs - rhs)
    /// for any finite inputs, verified bit-exact.
    #[kani::unwind(64)]
    #[kani::proof]
    fn binop_sub_correct() {
        let lhs: f32 = kani::any();
        let rhs: f32 = kani::any();
        kani::assume(lhs.is_finite());
        kani::assume(rhs.is_finite());

        let result = evaluate_constant_binop(BinOpKind::Sub, lhs, rhs);
        let val = result.expect("Sub of finite values must not return Err");
        assert_eq!(
            val.to_bits(),
            (lhs - rhs).to_bits(),
            "Sub result must be bit-exact"
        );
    }

    /// Proves `evaluate_constant_binop(Div)` is bit-exact with direct division
    /// for representative non-zero divisors.
    ///
    /// CBMC cannot handle fully-symbolic f32 division (SAT solver timeout).
    /// Uses symbolic index selection from 8 representative value pairs.
    /// Uses `unwind(8)` — syn::ErrorMessage Drop unwinding (#608).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn binop_div_correct() {
        const VALS: [f32; 8] = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 100.0, -100.0];
        let i: usize = kani::any();
        let j: usize = kani::any();
        kani::assume(i < VALS.len());
        kani::assume(j < VALS.len());

        let lhs = VALS[i];
        let rhs = VALS[j];
        // All VALS are non-zero, so division is valid.

        let result = evaluate_constant_binop(BinOpKind::Div, lhs, rhs);
        let val = result.expect("Div of finite non-zero values must not return Err");
        assert_eq!(
            val.to_bits(),
            (lhs / rhs).to_bits(),
            "Div result must be bit-exact"
        );
    }

    /// Proves Add commutativity: Add(a, b) == Add(b, a) for any finite inputs.
    /// This is a semantic property — not just a consequence of f32 arithmetic,
    /// but of the binop dispatch returning the same result regardless of order.
    #[kani::unwind(64)]
    #[kani::proof]
    fn binop_add_commutative() {
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        kani::assume(a.is_finite());
        kani::assume(b.is_finite());

        let r1 = evaluate_constant_binop(BinOpKind::Add, a, b)
            .expect("Add of finite values must not return Err");
        let r2 = evaluate_constant_binop(BinOpKind::Add, b, a)
            .expect("Add of finite values must not return Err");
        assert_eq!(
            r1.to_bits(),
            r2.to_bits(),
            "Add must be commutative: Add(a,b) == Add(b,a)"
        );
    }

    /// Proves Mul commutativity: Mul(a, b) == Mul(b, a) for representative values.
    /// Uses representative values to avoid CBMC f32 multiplication SAT timeout.
    /// Uses `unwind(8)` — syn::ErrorMessage Drop unwinding (#608).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(8)]
    fn binop_mul_commutative() {
        const VALS: [f32; 8] = [0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0, f32::MIN_POSITIVE];
        let i: usize = kani::any();
        let j: usize = kani::any();
        kani::assume(i < VALS.len());
        kani::assume(j < VALS.len());

        let a = VALS[i];
        let b = VALS[j];

        let r1 = evaluate_constant_binop(BinOpKind::Mul, a, b)
            .expect("Mul of finite values must not return Err");
        let r2 = evaluate_constant_binop(BinOpKind::Mul, b, a)
            .expect("Mul of finite values must not return Err");
        assert_eq!(
            r1.to_bits(),
            r2.to_bits(),
            "Mul must be commutative: Mul(a,b) == Mul(b,a)"
        );
    }
}
