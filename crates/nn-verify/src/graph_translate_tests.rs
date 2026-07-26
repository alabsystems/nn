// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for graph translation helpers.
//!
//! Extracted from `graph_translate.rs` to keep production code under 500 lines.

use super::*;
use ny_propagate::GraphNetwork;
use nn_dsl::ir::{IRNode, IRNodeKind, NodeId};
use ndarray::IxDyn;

/// Test that powi exponents exceeding 2^24 are rejected by the precision guard.
///
/// IR validation (POWI_MAX_EXPONENT=64) normally catches large exponents first.
/// This test bypasses IR validation by calling translate_node directly, exercising
/// the defense-in-depth guard for the i32→f32 precision limit (#562 AC4).
#[test]
fn test_powi_f32_precision_limit_rejects_large_exponent() {
    let large_exp: i32 = (1 << 24) + 1; // 16_777_217: exceeds precision limit
    let nodes = vec![
        IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
        IRNode::new(
            NodeId::new(1),
            IRNodeKind::Powi {
                base: NodeId::new(0),
                exp: large_exp,
            },
        ),
    ];
    let bindings = [ParamBinding::Variable];
    let param_node_names = [None];
    let ctx = TranslationContext {
        prefix: "test_",
        bindings: &bindings,
        num_variables: 1,
        param_node_names: &param_node_names,
        all_nodes: &nodes,
    };
    // First node (Param) produces a Variable
    let mut graph = GraphNetwork::new();
    let param_val =
        translate_node(&ctx, 0, &[], &mut graph).expect("Param translation should succeed");

    // Second node (Powi) should fail with precision limit error
    let result = translate_node(&ctx, 1, &[param_val], &mut graph);
    match result {
        Err(VerifyError::InternalTranslationError { context }) => {
            assert!(
                context.contains("precision limit"),
                "error should mention precision limit, got: {context}"
            );
            assert!(
                context.contains(&large_exp.to_string()),
                "error should mention the exponent {large_exp}, got: {context}"
            );
        }
        other => unreachable!(
            "powi with exp={large_exp} should be rejected by precision guard, got {other:?}"
        ),
    }
}

/// Test that powi exponents at the boundary (2^24) are accepted.
#[test]
fn test_powi_f32_precision_limit_accepts_boundary() {
    let boundary_exp: i32 = 1 << 24; // 16_777_216: exactly at limit (should pass)
    let nodes = vec![
        IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
        IRNode::new(
            NodeId::new(1),
            IRNodeKind::Powi {
                base: NodeId::new(0),
                exp: boundary_exp,
            },
        ),
    ];
    let bindings = [ParamBinding::Variable];
    let param_node_names = [None];
    let ctx = TranslationContext {
        prefix: "test_",
        bindings: &bindings,
        num_variables: 1,
        param_node_names: &param_node_names,
        all_nodes: &nodes,
    };
    let mut graph = GraphNetwork::new();
    let param_val =
        translate_node(&ctx, 0, &[], &mut graph).expect("Param translation should succeed");

    // Exponent exactly at 2^24 should be accepted (lossless cast).
    let result = translate_node(&ctx, 1, &[param_val], &mut graph);
    assert!(
        result.is_ok(),
        "powi with exp=2^24 should be accepted, got {result:?}"
    );
}

/// Test that negative powi exponents exceeding -2^24 are rejected.
///
/// Exercises `unsigned_abs()` on the negative side of the precision guard.
#[test]
fn test_powi_f32_precision_limit_rejects_negative_large_exponent() {
    let neg_large_exp: i32 = -((1 << 24) + 1); // -16_777_217
    let nodes = vec![
        IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
        IRNode::new(
            NodeId::new(1),
            IRNodeKind::Powi {
                base: NodeId::new(0),
                exp: neg_large_exp,
            },
        ),
    ];
    let bindings = [ParamBinding::Variable];
    let param_node_names = [None];
    let ctx = TranslationContext {
        prefix: "test_",
        bindings: &bindings,
        num_variables: 1,
        param_node_names: &param_node_names,
        all_nodes: &nodes,
    };
    let mut graph = GraphNetwork::new();
    let param_val =
        translate_node(&ctx, 0, &[], &mut graph).expect("Param translation should succeed");

    let result = translate_node(&ctx, 1, &[param_val], &mut graph);
    match result {
        Err(VerifyError::InternalTranslationError { context }) => {
            assert!(
                context.contains("precision limit"),
                "error should mention precision limit, got: {context}"
            );
            assert!(
                context.contains(&neg_large_exp.to_string()),
                "error should mention the exponent {neg_large_exp}, got: {context}"
            );
        }
        other => {
            unreachable!("powi with exp={neg_large_exp} should be rejected, got {other:?}")
        }
    }
}

/// AC1 regression: scalar_array rejects NaN (#562).
#[test]
fn test_scalar_array_rejects_nan() {
    let result = scalar_array(f32::NAN);
    assert!(result.is_err(), "scalar_array(NaN) must return Err");
    match result.unwrap_err() {
        VerifyError::NonFiniteConstant { value, .. } => {
            assert!(value.is_nan(), "error value should be NaN");
        }
        other => unreachable!("expected NonFiniteConstant, got {other:?}"),
    }
}

/// AC1 regression: scalar_array rejects infinity (#562).
#[test]
fn test_scalar_array_rejects_infinity() {
    let result = scalar_array(f32::INFINITY);
    assert!(result.is_err(), "scalar_array(+Inf) must return Err");

    let result = scalar_array(f32::NEG_INFINITY);
    assert!(result.is_err(), "scalar_array(-Inf) must return Err");
}

/// AC1 regression: scalar_array accepts finite values (#562).
#[test]
fn test_scalar_array_accepts_finite() {
    let result = scalar_array(42.0);
    assert!(result.is_ok(), "scalar_array(42.0) should succeed");
    let arr = result.unwrap();
    assert_eq!(arr.ndim(), 0, "should be 0-dimensional");
    assert_eq!(arr[IxDyn(&[])], 42.0);
}

/// Test that negative powi exponents at the boundary (-2^24) are accepted.
#[test]
fn test_powi_f32_precision_limit_accepts_negative_boundary() {
    let neg_boundary_exp: i32 = -(1 << 24); // -16_777_216: exactly at limit
    let nodes = vec![
        IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
        IRNode::new(
            NodeId::new(1),
            IRNodeKind::Powi {
                base: NodeId::new(0),
                exp: neg_boundary_exp,
            },
        ),
    ];
    let bindings = [ParamBinding::Variable];
    let param_node_names = [None];
    let ctx = TranslationContext {
        prefix: "test_",
        bindings: &bindings,
        num_variables: 1,
        param_node_names: &param_node_names,
        all_nodes: &nodes,
    };
    let mut graph = GraphNetwork::new();
    let param_val =
        translate_node(&ctx, 0, &[], &mut graph).expect("Param translation should succeed");

    let result = translate_node(&ctx, 1, &[param_val], &mut graph);
    assert!(
        result.is_ok(),
        "powi with exp=-2^24 should be accepted, got {result:?}"
    );
}
