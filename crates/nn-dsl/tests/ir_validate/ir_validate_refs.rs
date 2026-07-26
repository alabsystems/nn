// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Additional reference validation tests for KernelDef::validate().
//!
//! Tests for Powi, Clamp, MinMax invalid references, out-of-bounds param
//! indices, and non-finite literal rejection.
//!
//! BinOp, UnaryFn, and valid-kernel tests are in `ir_validate_properties.rs`.

use nn_dsl::ir::{IRError, IRNode, IRNodeKind, KernelDef, MinMaxKind, NodeId, Param, ScalarType};

// ======================== Helpers ========================

fn param(name: &str) -> Param {
    Param::new(name, ScalarType::F32)
}

fn node(id: usize, kind: IRNodeKind) -> IRNode {
    IRNode::new(NodeId::new(id), kind)
}

// ======================== Invalid node references (continued) ========================

#[test]
fn validate_powi_invalid_base_caught() {
    let kernel = KernelDef::new(
        "bad_powi",
        vec![param("x")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(
                1,
                IRNodeKind::Powi {
                    base: NodeId::new(7), // invalid
                    exp: 2,
                },
            ),
        ],
        NodeId::new(1),
    );
    let err = kernel
        .validate()
        .expect_err("invalid Powi base should fail");
    assert!(matches!(err, IRError::InvalidNodeRef(id) if id == NodeId::new(7)));
}

#[test]
fn validate_clamp_invalid_refs_caught() {
    // Test each clamp ref independently
    let test_cases = [
        ("bad_input", NodeId::new(5), NodeId::new(0), NodeId::new(0)),
        ("bad_min", NodeId::new(0), NodeId::new(5), NodeId::new(0)),
        ("bad_max", NodeId::new(0), NodeId::new(0), NodeId::new(5)),
    ];

    for (label, input, min, max) in test_cases {
        let kernel = KernelDef::new(
            label.to_string(),
            vec![param("x")],
            ScalarType::F32,
            vec![
                node(0, IRNodeKind::Param(0)),
                node(1, IRNodeKind::Clamp { input, min, max }),
            ],
            NodeId::new(1),
        );
        let err = kernel
            .validate()
            .expect_err(&format!("Clamp with {label} should fail validation"));
        assert!(
            matches!(err, IRError::InvalidNodeRef(id) if id == NodeId::new(5)),
            "Clamp {label}: expected InvalidNodeRef(5), got: {err:?}"
        );
    }
}

#[test]
fn validate_minmax_invalid_refs_caught() {
    for op in [MinMaxKind::Min, MinMaxKind::Max] {
        // Invalid lhs
        let kernel = KernelDef::new(
            format!("bad_{op:?}_lhs"),
            vec![param("x")],
            ScalarType::F32,
            vec![
                node(0, IRNodeKind::Param(0)),
                node(
                    1,
                    IRNodeKind::MinMax {
                        op,
                        lhs: NodeId::new(9),
                        rhs: NodeId::new(0),
                    },
                ),
            ],
            NodeId::new(1),
        );
        let err = kernel
            .validate()
            .expect_err(&format!("MinMax {op:?} with invalid lhs should fail"));
        assert!(
            matches!(err, IRError::InvalidNodeRef(id) if id == NodeId::new(9)),
            "MinMax {op:?} lhs: expected InvalidNodeRef(9), got: {err:?}"
        );

        // Invalid rhs
        let kernel = KernelDef::new(
            format!("bad_{op:?}_rhs"),
            vec![param("x")],
            ScalarType::F32,
            vec![
                node(0, IRNodeKind::Param(0)),
                node(
                    1,
                    IRNodeKind::MinMax {
                        op,
                        lhs: NodeId::new(0),
                        rhs: NodeId::new(9),
                    },
                ),
            ],
            NodeId::new(1),
        );
        let err = kernel
            .validate()
            .expect_err(&format!("MinMax {op:?} with invalid rhs should fail"));
        assert!(
            matches!(err, IRError::InvalidNodeRef(id) if id == NodeId::new(9)),
            "MinMax {op:?} rhs: expected InvalidNodeRef(9), got: {err:?}"
        );
    }
}

// ======================== Invalid param references ========================

#[test]
fn validate_param_out_of_bounds_caught() {
    let kernel = KernelDef::new(
        "bad_param",
        vec![param("x")],
        ScalarType::F32,
        vec![
            node(0, IRNodeKind::Param(0)),
            node(1, IRNodeKind::Param(1)), // only 1 param, so index 1 is invalid
        ],
        NodeId::new(1),
    );
    let err = kernel
        .validate()
        .expect_err("out-of-bounds param should fail");
    assert!(
        matches!(err, IRError::InvalidParamRef(1, 1)),
        "expected InvalidParamRef(1, 1), got {err:?}"
    );
}

#[test]
fn validate_param_index_large_caught() {
    let kernel = KernelDef::new(
        "huge_param",
        vec![param("x"), param("y")],
        ScalarType::F32,
        vec![node(0, IRNodeKind::Param(100))],
        NodeId::new(0),
    );
    let err = kernel
        .validate()
        .expect_err("param index 100 with 2 params should fail");
    assert!(matches!(err, IRError::InvalidParamRef(100, 2)));
}

// ======================== Exhaustive node-kind coverage ========================

#[test]
fn validate_finite_literal_passes() {
    // Finite literals pass validation — only NaN/Inf are rejected.
    for val in [0.0, 1.0, -1.0, f64::MAX, f64::MIN, f64::MIN_POSITIVE] {
        let kernel = KernelDef::new(
            "lit",
            vec![param("x")],
            ScalarType::F32,
            vec![
                node(0, IRNodeKind::Param(0)),
                node(1, IRNodeKind::Literal(val)),
            ],
            NodeId::new(0),
        );
        assert!(
            kernel.validate().is_ok(),
            "Literal({val}) should pass validation"
        );
    }
}

#[test]
fn validate_non_finite_literal_rejected() {
    // NaN and Infinity are not valid MSL literals — reject during validation.
    for val in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let kernel = KernelDef::new(
            "lit",
            vec![param("x")],
            ScalarType::F32,
            vec![
                node(0, IRNodeKind::Param(0)),
                node(1, IRNodeKind::Literal(val)),
            ],
            NodeId::new(0),
        );
        assert!(
            matches!(kernel.validate(), Err(IRError::NonFiniteLiteral(..))),
            "Literal({val}) should be rejected as non-finite"
        );
    }
}
