// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! IR structural validation tests: non-finite literals, forward refs, node IDs,
//! Clamp/MinMax/SumReduce type mismatches, and MSL identifier validation.

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

// --- ir_validate.rs coverage ---

#[test]
fn test_non_finite_literal_nan_rejected() {
    let kernel = KernelDef::new(
        "nan_kernel",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Literal(f64::NAN)),
        ],
        NodeId::new(1),
    );
    let err = kernel.validate().unwrap_err();
    assert!(
        matches!(err, IRError::NonFiniteLiteral(id, v) if id == NodeId::new(1) && v.is_nan()),
        "NaN literal should be rejected, got: {err}"
    );
}

#[test]
fn test_non_finite_literal_inf_rejected() {
    let kernel = KernelDef::new(
        "inf_kernel",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Literal(f64::INFINITY)),
        ],
        NodeId::new(1),
    );
    let err = kernel.validate().unwrap_err();
    assert!(
        matches!(err, IRError::NonFiniteLiteral(id, v) if id == NodeId::new(1) && v.is_infinite()),
        "Inf literal should be rejected, got: {err}"
    );
}

#[test]
fn test_non_finite_literal_neg_inf_rejected() {
    let kernel = KernelDef::new(
        "neg_inf_kernel",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Literal(f64::NEG_INFINITY)),
        ],
        NodeId::new(1),
    );
    let err = kernel.validate().unwrap_err();
    assert!(
        matches!(err, IRError::NonFiniteLiteral(id, v) if id == NodeId::new(1) && v.is_infinite() && v.is_sign_negative()),
        "-Inf literal should be rejected with negative sign, got: {err}"
    );
}

#[test]
fn test_empty_sum_reduce_rejected() {
    let kernel = KernelDef::new(
        "empty_reduce",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::SumReduce { inputs: vec![] }),
        ],
        NodeId::new(1),
    );
    let err = kernel.validate().unwrap_err();
    assert!(
        matches!(err, IRError::EmptySumReduce(id) if id == NodeId::new(1)),
        "empty SumReduce should be rejected, got: {err}"
    );
}

#[test]
fn test_forward_ref_rejected() {
    let kernel = KernelDef::new(
        "forward_ref",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            // Node 1 references node 2 which hasn't been defined yet
            n(
                1,
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(2),
                },
            ),
            n(2, IRNodeKind::Literal(1.0)),
        ],
        NodeId::new(1),
    );
    let err = kernel.validate().unwrap_err();
    assert!(
        matches!(err, IRError::ForwardRef(a, b) if a == NodeId::new(1) && b == NodeId::new(2)),
        "forward reference should be rejected, got: {err}"
    );
}

#[test]
fn test_self_ref_rejected() {
    let kernel = KernelDef::new(
        "self_ref",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(
                1,
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Sin,
                    input: NodeId::new(1), // self-reference
                },
            ),
        ],
        NodeId::new(1),
    );
    let err = kernel.validate().unwrap_err();
    assert!(
        matches!(err, IRError::ForwardRef(a, b) if a == NodeId::new(1) && b == NodeId::new(1)),
        "self-reference should be rejected, got: {err}"
    );
}

#[test]
fn test_mismatched_node_id_rejected() {
    let kernel = KernelDef::new(
        "bad_ids",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(5, IRNodeKind::Literal(1.0)), // id=5 but index=1
        ],
        NodeId::new(1),
    );
    let err = kernel.validate().unwrap_err();
    assert!(
        matches!(
            err,
            IRError::MismatchedNodeId {
                found,
                expected_index: 1,
            } if found == NodeId::new(5)
        ),
        "mismatched node id should be rejected, got: {err}"
    );
}

// --- ir_type_check.rs: Clamp/MinMax/SumReduce type-mismatch coverage (#272) ---

#[test]
fn test_clamp_min_type_mismatch() {
    let kernel = KernelDef::new(
        "clamp_min_mismatch",
        vec![f32_param("x"), f16_param("lo")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(2, IRNodeKind::Literal(10.0)),
            n(
                3,
                IRNodeKind::Clamp {
                    input: NodeId::new(0),
                    min: NodeId::new(1),
                    max: NodeId::new(2),
                },
            ),
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
                rhs_type: ValueType::F16
            } if node == NodeId::new(3)
        ),
        "Clamp min type mismatch should be rejected, got: {err}"
    );
}

#[test]
fn test_clamp_max_type_mismatch() {
    let kernel = KernelDef::new(
        "clamp_max_mismatch",
        vec![f32_param("x"), f16_param("hi")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Literal(0.0)),
            n(2, IRNodeKind::Param(1)),
            n(
                3,
                IRNodeKind::Clamp {
                    input: NodeId::new(0),
                    min: NodeId::new(1),
                    max: NodeId::new(2),
                },
            ),
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
                rhs_type: ValueType::F16
            } if node == NodeId::new(3)
        ),
        "Clamp max type mismatch should be rejected, got: {err}"
    );
}

#[test]
fn test_minmax_operand_type_mismatch() {
    let kernel = KernelDef::new(
        "mixed_max",
        vec![f32_param("x"), f16_param("y")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(
                2,
                IRNodeKind::MinMax {
                    op: MinMaxKind::Max,
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
            IRError::OperandTypeMismatch {
                node,
                lhs_type: ValueType::F32,
                rhs_type: ValueType::F16
            } if node == NodeId::new(2)
        ),
        "MinMax type mismatch should be rejected, got: {err}"
    );
}

#[test]
fn test_sum_reduce_mixed_types() {
    let kernel = KernelDef::new(
        "mixed_reduce",
        vec![f32_param("x"), f16_param("y")],
        ScalarType::F32,
        vec![
            n(0, IRNodeKind::Param(0)),
            n(1, IRNodeKind::Param(1)),
            n(
                2,
                IRNodeKind::SumReduce {
                    inputs: vec![NodeId::new(0), NodeId::new(1)],
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
                rhs_type: ValueType::F16
            } if node == NodeId::new(2)
        ),
        "SumReduce mixed types should be rejected, got: {err}"
    );
}

// --- MSL identifier validation tests (#280) ---

#[test]
fn test_validate_rejects_empty_kernel_name() {
    let kernel = KernelDef::new(
        "",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![n(0, IRNodeKind::Param(0))],
        NodeId::new(0),
    );
    let err = kernel.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("empty"),
        "expected empty name error, got: {msg}"
    );
}

#[test]
fn test_validate_rejects_digit_start_kernel_name() {
    let kernel = KernelDef::new(
        "3abc",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![n(0, IRNodeKind::Param(0))],
        NodeId::new(0),
    );
    let err = kernel.validate().unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("digit"), "expected digit error, got: {msg}");
}

#[test]
fn test_validate_rejects_special_char_kernel_name() {
    let kernel = KernelDef::new(
        "nn-kernel",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![n(0, IRNodeKind::Param(0))],
        NodeId::new(0),
    );
    let err = kernel.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-alphanumeric"),
        "expected special char error, got: {msg}"
    );
}

/// MSL reserved words are now accepted by IR validation (structural-only).
/// Backend-specific reserved word checks run at codegen time. (Part of #586.)
#[test]
fn test_validate_accepts_msl_reserved_word_kernel_name() {
    let kernel = KernelDef::new(
        "kernel",
        vec![f32_param("x")],
        ScalarType::F32,
        vec![n(0, IRNodeKind::Param(0))],
        NodeId::new(0),
    );
    kernel
        .validate()
        .expect("MSL reserved words should pass IR validation (caught at codegen)");
}

/// MSL reserved words in param names are now accepted by IR validation.
/// Backend-specific checks run at codegen time. (Part of #586.)
#[test]
fn test_validate_accepts_msl_reserved_word_param_name() {
    let kernel = KernelDef::new(
        "nn_kernel",
        vec![f32_param("thread")],
        ScalarType::F32,
        vec![n(0, IRNodeKind::Param(0))],
        NodeId::new(0),
    );
    kernel
        .validate()
        .expect("MSL reserved param names should pass IR validation (caught at codegen)");
}

#[test]
fn test_validate_accepts_valid_identifiers() {
    let kernel = KernelDef::new(
        "nn_kernel_v2",
        vec![f32_param("x"), f32_param("alpha_0")],
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
        ],
        NodeId::new(2),
    );
    kernel.validate().expect("valid identifiers should pass");
}

#[test]
fn test_validate_rejects_space_in_param_name() {
    let kernel = KernelDef::new(
        "nn_kernel",
        vec![Param::new("nn param".to_string(), ScalarType::F32)],
        ScalarType::F32,
        vec![n(0, IRNodeKind::Param(0))],
        NodeId::new(0),
    );
    let err = kernel.validate().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-alphanumeric"),
        "expected non-alphanumeric error for spaces, got: {msg}"
    );
}
