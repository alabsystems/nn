// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for codegen edge cases — covers code paths that
//! unit tests in the source modules missed.

use nn_dsl::ir::{CompareOpKind, IRNode, IRNodeKind, NodeId, Param, ScalarType};
use nn_dsl::{emit_msl, emit_scalar_fn, KernelDef, Lowerer};

fn lower(src: &str) -> KernelDef {
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    Lowerer::lower_fn(&func).expect("lower")
}

fn param(name: &str) -> Param {
    Param::new(name, ScalarType::F32)
}

fn node(id: usize, kind: IRNodeKind) -> IRNode {
    IRNode::new(NodeId::new(id), kind)
}

/// powi(2) emits `b * b` (inlined square), powi(3) emits `b * b * b`
/// (repeated multiplication). metal::pow is never used because it
/// produces NaN for negative bases.
#[test]
fn test_powi_non_square_emits_multiplication() {
    let kernel = lower("fn cube(x: f32) -> f32 { x.powi(3) }");
    let msl = emit_msl(&kernel).expect("emit");
    assert!(
        msl.contains("x * x * x"),
        "powi(3) should expand to repeated multiplication, MSL:\n{msl}"
    );
    assert!(
        !msl.contains("metal::pow") && !msl.contains("metal::precise::pow"),
        "powi should never use metal::pow (broken for negative bases), MSL:\n{msl}"
    );
}

/// `recip()` emits `T(1) / x`, not a function call.
#[test]
fn test_recip_emits_division() {
    let kernel = lower("fn inv(x: f32) -> f32 { x.recip() }");
    let msl = emit_msl(&kernel).expect("emit");
    assert!(
        msl.contains("float(1) /"),
        "recip should emit 1/x pattern, MSL:\n{msl}"
    );
}

#[test]
fn test_sum_reduce_intrinsic_emits_add_chain() {
    let kernel = lower(
        "fn sum3(a: f32, b: f32, c: f32) -> f32 {
            nn_dsl::sum_reduce([a, b, c])
        }",
    );
    let msl = emit_msl(&kernel).expect("emit");
    assert!(
        msl.contains("float t3 = a + b + c;"),
        "sum_reduce should emit explicit add chain, MSL:\n{msl}"
    );
}

#[test]
fn test_if_else_emits_compare_and_ternary() {
    let kernel = lower("fn relu_if(x: f32) -> f32 { if x >= 0.0 { x } else { 0.0 } }");
    let msl = emit_msl(&kernel).expect("emit");
    assert!(
        msl.contains(" >= "),
        "comparison operator should be emitted in MSL, MSL:\n{msl}"
    );
    assert!(
        msl.contains(" ? "),
        "if/else should emit ternary select in MSL, MSL:\n{msl}"
    );
}

/// Snake1d shape validation: zero channels must be rejected.
#[test]
fn test_snake_shape_validation_zero_channels() {
    let x = vec![0.0f32; 4];
    let alpha = vec![1.0f32; 0];
    let err = nn_dsl::snake_ref_f32(&x, &alpha, 0, 4).expect_err("zero channels should fail");
    assert!(
        matches!(
            err,
            nn_dsl::KernelError::InvalidDimension {
                name: "channels",
                value: 0
            }
        ),
        "expected InvalidDimension for channels, got {err:?}"
    );
}

/// Snake1d shape validation: zero length must be rejected.
#[test]
fn test_snake_shape_validation_zero_length() {
    let x = vec![0.0f32; 4];
    let alpha = vec![1.0f32; 2];
    let err = nn_dsl::snake_ref_f32(&x, &alpha, 2, 0).expect_err("zero length should fail");
    assert!(
        matches!(
            err,
            nn_dsl::KernelError::InvalidDimension {
                name: "length",
                value: 0
            }
        ),
        "expected InvalidDimension for length, got {err:?}"
    );
}

/// Snake1d shape validation: alpha length must match channel count.
#[test]
fn test_snake_shape_validation_alpha_size_mismatch() {
    let x = vec![0.0f32; 8];
    let alpha = vec![1.0f32; 3]; // 3 alphas but 2 channels
    let err =
        nn_dsl::snake_ref_f32(&x, &alpha, 2, 4).expect_err("alpha size mismatch should fail");
    assert!(
        matches!(
            err,
            nn_dsl::KernelError::ShapeMismatch {
                expected: 2,
                got: 3
            }
        ),
        "expected ShapeMismatch, got {err:?}"
    );
}

// ======================== Compare MSL emission ========================

#[test]
fn test_compare_all_ops_emit_correct_msl_operators() {
    let cases: &[(CompareOpKind, &str)] = &[
        (CompareOpKind::Eq, "=="),
        (CompareOpKind::Ne, "!="),
        (CompareOpKind::Lt, "<"),
        (CompareOpKind::Le, "<="),
        (CompareOpKind::Gt, ">"),
        (CompareOpKind::Ge, ">="),
    ];
    for &(op, expected_sym) in cases {
        let kernel = KernelDef::new(
            format!("cmp_{op:?}").to_lowercase(),
            vec![param("x"), param("y")],
            ScalarType::F32,
            vec![
                node(0, IRNodeKind::Param(0)),
                node(1, IRNodeKind::Param(1)),
                node(
                    2,
                    IRNodeKind::Compare {
                        op,
                        lhs: NodeId::new(0),
                        rhs: NodeId::new(1),
                    },
                ),
                node(
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
        let msl = emit_scalar_fn(&kernel).expect("emit");
        // Compare node emits `bool tN = lhs OP rhs;`
        let cmp_pattern = format!("bool t2 = x {expected_sym} y;");
        assert!(
            msl.contains(&cmp_pattern),
            "Compare::{op:?} should emit '{cmp_pattern}', MSL:\n{msl}"
        );
        // Compare emits `bool`, not `float`
        assert!(
            msl.contains("bool t2"),
            "Compare should emit bool type, MSL:\n{msl}"
        );
    }
}

#[test]
fn test_compare_node_uses_param_names_not_indices() {
    let kernel = KernelDef::new(
        "named_cmp",
        vec![param("alpha"), param("beta")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(1, IRNodeKind::Param(1)),
            node(
                2,
                IRNodeKind::Compare {
                    op: CompareOpKind::Lt,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            node(
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
    let msl = emit_scalar_fn(&kernel).expect("emit");
    assert!(
        msl.contains("bool t2 = alpha < beta;"),
        "Compare should use param names, MSL:\n{msl}"
    );
}

// ======================== Select MSL emission ========================

#[test]
fn test_select_emits_ternary_operator() {
    let kernel = KernelDef::new(
        "select_test",
        vec![param("x"), param("y")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(1, IRNodeKind::Param(1)),
            node(
                2,
                IRNodeKind::Compare {
                    op: CompareOpKind::Gt,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            node(
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
    let msl = emit_scalar_fn(&kernel).expect("emit");
    assert!(
        msl.contains("(t2 ? x : y)"),
        "Select should emit ternary (cond ? then : else), MSL:\n{msl}"
    );
    // Select result type should be the kernel return type, not bool
    assert!(
        msl.contains("float t3 = (t2 ? x : y);"),
        "Select result should be float, MSL:\n{msl}"
    );
}

#[test]
fn test_select_with_literal_branches() {
    let kernel = KernelDef::new(
        "conditional_literal",
        vec![param("x")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(1, IRNodeKind::Literal(0.0)),
            node(2, IRNodeKind::Literal(1.0)),
            node(
                3,
                IRNodeKind::Compare {
                    op: CompareOpKind::Ge,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            node(
                4,
                IRNodeKind::Select {
                    cond: NodeId::new(3),
                    then_val: NodeId::new(2),
                    else_val: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(4),
    );
    let msl = emit_scalar_fn(&kernel).expect("emit");
    // Compare: x >= 0.0
    assert!(
        msl.contains("bool t3 = x >= t1;"),
        "Compare should reference param and literal, MSL:\n{msl}"
    );
    // Select: t3 ? t2 : t1
    assert!(
        msl.contains("(t3 ? t2 : t1)"),
        "Select should use ternary with literal refs, MSL:\n{msl}"
    );
}

#[test]
fn test_compare_select_full_kernel_wrapper() {
    let kernel = KernelDef::new(
        "relu_manual",
        vec![param("x")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(1, IRNodeKind::Literal(0.0)),
            node(
                2,
                IRNodeKind::Compare {
                    op: CompareOpKind::Gt,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            node(
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
    let msl = emit_msl(&kernel).expect("emit");
    // Full MSL should have both scalar fn and kernel wrapper
    assert!(
        msl.contains("#include <metal_stdlib>"),
        "full MSL should have stdlib include"
    );
    assert!(
        msl.contains("[[kernel]] void relu_manual_kernel("),
        "full MSL should have kernel wrapper"
    );
    assert!(
        msl.contains("bool t2"),
        "full MSL should contain Compare emission"
    );
    assert!(msl.contains("? "), "full MSL should contain Select ternary");
}
