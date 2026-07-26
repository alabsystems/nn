// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for KernelIR type checking: Compare, Select, BinOp, conversions.
//!
//! Structural validation tests (non-finite literals, forward refs, node IDs,
//! Clamp/MinMax/SumReduce mismatches, identifiers): ir_tests_validate.rs

use super::*;

fn n(id: usize, kind: IRNodeKind) -> IRNode {
    IRNode::new(NodeId::new(id), kind)
}

fn f32_param(name: &str) -> Param {
    Param::new(name.to_string(), ScalarType::F32)
}

fn f16_param(name: &str) -> Param {
    Param::new(name.to_string(), ScalarType::F16)
}

#[test]
fn test_valid_compare_select_graph_passes() {
    // x > 0.0 ? x : 0.0
    let kernel = KernelDef::new(
        "relu_cond",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Literal(0.0)),
            n(
                2,
                IRNodeKind::Compare {
                    op: CompareOpKind::Gt,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            n(
                3,
                IRNodeKind::Select {
                    cond: NodeId::new(2),
                    then_val: NodeId::new(0),
                    else_val: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(3),
    );
    kernel
        .validate()
        .expect("valid Compare->Select graph should pass");
}

#[test]
fn test_select_cond_must_be_bool() {
    // Select with a float BinOp as cond — must reject
    let kernel = KernelDef::new(
        "bad_select",
        vec![f32_param("x"), f32_param("y")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(
                2,
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            n(
                3,
                IRNodeKind::Select {
                    cond: NodeId::new(2), // float, not bool
                    then_val: NodeId::new(0),
                    else_val: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(3),
    );
    let err = kernel.validate().unwrap_err();
    assert!(
        matches!(
            err,
            IRError::SelectCondNotBool {
                node,
                found: ValueType::F32
            } if node == NodeId::new(3)
        ),
        "expected SelectCondNotBool, got: {err}"
    );
}

#[test]
fn test_select_branch_type_mismatch() {
    // Select where then is f32 param and else is f16 param
    let kernel = KernelDef::new(
        "mixed_select",
        vec![f32_param("x"), f16_param("y")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(2, IRNodeKind::Literal(0.0)),
            n(
                3,
                IRNodeKind::Compare {
                    op: CompareOpKind::Gt,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(2),
                },
            ),
            n(
                4,
                IRNodeKind::Select {
                    cond: NodeId::new(3),
                    then_val: NodeId::new(0), // F32
                    else_val: NodeId::new(1), // F16
                },
            ),
        ],
        NodeId::new(4),
    );
    let err = kernel.validate().unwrap_err();
    assert!(
        matches!(
            err,
            IRError::SelectBranchTypeMismatch {
                node,
                then_type: ValueType::F32,
                else_type: ValueType::F16,
            } if node == NodeId::new(4)
        ),
        "expected SelectBranchTypeMismatch, got: {err}"
    );
}

#[test]
fn test_compare_operand_type_mismatch() {
    // Compare f32 against f16
    let kernel = KernelDef::new(
        "mixed_cmp",
        vec![f32_param("x"), f16_param("y")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(
                2,
                IRNodeKind::Compare {
                    op: CompareOpKind::Lt,
                    lhs: NodeId::new(0), // F32
                    rhs: NodeId::new(1), // F16
                },
            ),
            // Need a numeric output to satisfy return type
            n(3, IRNodeKind::Literal(1.0)),
        ],
        NodeId::new(3),
    );
    let err = kernel.validate().unwrap_err();
    assert!(
        matches!(
            err,
            IRError::OperandTypeMismatch {
                node,
                lhs_type: ValueType::F32,
                rhs_type: ValueType::F16,
            } if node == NodeId::new(2)
        ),
        "expected OperandTypeMismatch on Compare, got: {err}"
    );
}

#[test]
fn test_binop_operand_type_mismatch() {
    // Add f32 + f16
    let kernel = KernelDef::new(
        "mixed_add",
        vec![f32_param("x"), f16_param("y")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(
                2,
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0), // F32
                    rhs: NodeId::new(1), // F16
                },
            ),
        ],
        NodeId::new(2),
    );
    let err = kernel.validate().unwrap_err();
    assert!(
        matches!(
            err,
            IRError::OperandTypeMismatch {
                node,
                lhs_type: ValueType::F32,
                rhs_type: ValueType::F16,
            } if node == NodeId::new(2)
        ),
        "expected OperandTypeMismatch on BinOp, got: {err}"
    );
}

#[test]
fn test_output_type_mismatch_bool() {
    // Output is a Compare node (bool), but return type is f32
    let kernel = KernelDef::new(
        "bool_output",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Literal(0.0)),
            n(
                2,
                IRNodeKind::Compare {
                    op: CompareOpKind::Gt,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    let err = kernel.validate().unwrap_err();
    assert!(
        matches!(
            err,
            IRError::OutputTypeMismatch {
                found: ValueType::Bool,
                expected: ValueType::F32,
            }
        ),
        "expected OutputTypeMismatch, got: {err}"
    );
}

#[test]
fn test_non_numeric_in_unary_fn() {
    // sin(compare_result) — bool in unary fn
    let kernel = KernelDef::new(
        "sin_bool",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Literal(0.0)),
            n(
                2,
                IRNodeKind::Compare {
                    op: CompareOpKind::Gt,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            n(
                3,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Sin,
                    input: NodeId::new(2), // bool
                },
            ),
        ],
        NodeId::new(3),
    );
    let err = kernel.validate().unwrap_err();
    assert!(
        matches!(
            err,
            IRError::NonNumericOperand {
                node,
                operand,
                found: ValueType::Bool,
            } if node == NodeId::new(3) && operand == NodeId::new(2)
        ),
        "expected NonNumericOperand, got: {err}"
    );
}

#[test]
fn test_valuetype_is_numeric() {
    assert!(ValueType::F32.is_numeric());
    assert!(ValueType::F16.is_numeric());
    assert!(ValueType::BF16.is_numeric());
    assert!(!ValueType::Bool.is_numeric());
}

#[test]
fn test_valuetype_from_scalar_type() {
    assert_eq!(ValueType::from(ScalarType::F32), ValueType::F32);
    assert_eq!(ValueType::from(ScalarType::F16), ValueType::F16);
    assert_eq!(ValueType::from(ScalarType::BF16), ValueType::BF16);
}

// --- format_literal tests ---

#[test]
fn test_format_literal_integer_values_get_decimal_point() {
    // Integer-valued floats should format with .0 suffix
    assert_eq!(ir_pretty::format_literal(0.0), "0.0");
    assert_eq!(ir_pretty::format_literal(1.0), "1.0");
    assert_eq!(ir_pretty::format_literal(-1.0), "-1.0");
    assert_eq!(ir_pretty::format_literal(42.0), "42.0");
    assert_eq!(ir_pretty::format_literal(1000.0), "1000.0");
}

#[test]
fn test_format_literal_fractional_values_use_default_display() {
    // Non-integer values should use default f64 Display
    assert_eq!(ir_pretty::format_literal(1.23), "1.23");
    assert_eq!(ir_pretty::format_literal(-0.5), "-0.5");
    assert_eq!(ir_pretty::format_literal(7.654321), "7.654321");
}

#[test]
fn test_format_literal_large_integers_use_default_display() {
    // Integers >= 1e15 should use default Display (no .0 formatting)
    let large = 1e15;
    let result = ir_pretty::format_literal(large);
    // Should NOT use {:.1} formatting for very large integers
    assert!(
        !result.ends_with(".0") || result.contains('e'),
        "large integer {large} should not use .0 format: {result}"
    );
}

#[test]
fn test_format_literal_special_values() {
    // NaN: v != v.floor() because NaN != NaN, so falls through to default Display
    let nan_result = ir_pretty::format_literal(f64::NAN);
    assert_eq!(nan_result, "NaN");

    // Infinity: same reasoning
    let inf_result = ir_pretty::format_literal(f64::INFINITY);
    assert_eq!(inf_result, "inf");

    let neg_inf_result = ir_pretty::format_literal(f64::NEG_INFINITY);
    assert_eq!(neg_inf_result, "-inf");
}

// --- ScalarType / ValueType / DType conversion tests ---

#[test]
fn test_scalar_type_to_dtype() {
    use nn_core::DType;
    assert_eq!(DType::from(ScalarType::F32), DType::F32);
    assert_eq!(DType::from(ScalarType::F16), DType::F16);
}

#[test]
fn test_dtype_to_scalar_type_valid() {
    use nn_core::DType;
    assert_eq!(ScalarType::try_from(DType::F32).unwrap(), ScalarType::F32);
    assert_eq!(ScalarType::try_from(DType::F16).unwrap(), ScalarType::F16);
}

#[test]
fn test_dtype_to_scalar_type_unsupported() {
    use nn_core::DType;
    for dt in [DType::F64, DType::I32, DType::I64, DType::U8, DType::Bool] {
        let err = ScalarType::try_from(dt).unwrap_err();
        assert!(
            matches!(err, IRError::UnsupportedType(_)),
            "expected UnsupportedType for {dt}, got: {err}"
        );
    }
}

#[test]
fn test_scalar_type_roundtrip_through_dtype() {
    use nn_core::DType;
    for st in [ScalarType::F32, ScalarType::F16, ScalarType::BF16] {
        let dt = DType::from(st);
        let roundtrip = ScalarType::try_from(dt).unwrap();
        assert_eq!(roundtrip, st, "round-trip failed for {st}");
    }
}

#[test]
fn test_dtype_to_value_type_valid() {
    use nn_core::DType;
    assert_eq!(ValueType::try_from(DType::F32).unwrap(), ValueType::F32);
    assert_eq!(ValueType::try_from(DType::F16).unwrap(), ValueType::F16);
    assert_eq!(ValueType::try_from(DType::BF16).unwrap(), ValueType::BF16);
    assert_eq!(ValueType::try_from(DType::Bool).unwrap(), ValueType::Bool);
}

#[test]
fn test_dtype_to_value_type_unsupported() {
    use nn_core::DType;
    for dt in [DType::F64, DType::I32, DType::I64, DType::U8] {
        let err = ValueType::try_from(dt).unwrap_err();
        assert!(
            matches!(err, IRError::UnsupportedType(_)),
            "expected UnsupportedType for {dt}, got: {err}"
        );
    }
}
