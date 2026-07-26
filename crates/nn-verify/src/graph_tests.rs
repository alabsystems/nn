// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for graph translation utilities.

use super::*;
use nn_dsl::lower::Lowerer;

/// `has_variable_comparison` returns true for a kernel with `x > 0.0`
/// where x is Variable.
#[test]
fn test_has_variable_comparison_gt_variable() {
    // if x > 0.0 { x } else { 0.0 }  (ReLU via select)
    let src = "fn relu(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");
    let bindings = [ParamBinding::Variable];
    assert!(
        has_variable_comparison(&kernel, &bindings),
        "relu has variable-operand Compare (x > 0.0)"
    );
}

/// `has_variable_comparison` returns false when the Compare node only has
/// constant operands (all params are constant).
#[test]
fn test_has_variable_comparison_constant_only() {
    // Same kernel, but x is bound as Constant — no variable comparison.
    let src = "fn relu(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");
    let bindings = [ParamBinding::Constant(5.0)];
    assert!(
        !has_variable_comparison(&kernel, &bindings),
        "constant-only Compare should not flag"
    );
}

/// `has_variable_comparison` returns false for a kernel with no Compare nodes.
#[test]
fn test_has_variable_comparison_no_compare() {
    let src = "fn add1(x: f32) -> f32 { x + 1.0 }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");
    let bindings = [ParamBinding::Variable];
    assert!(
        !has_variable_comparison(&kernel, &bindings),
        "no Compare nodes means no comparison approximation"
    );
}

/// `has_variable_comparison` returns true for multi-variable Eq comparison.
#[test]
fn test_has_variable_comparison_eq_multi_variable() {
    // if x == y { 1.0 } else { 0.0 }
    let src = "fn eq_select(x: f32, y: f32) -> f32 { if x == y { 1.0 } else { 0.0 } }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");
    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    assert!(
        has_variable_comparison(&kernel, &bindings),
        "Eq(x, y) with both variable should flag"
    );
}

/// Malformed KernelDef with out-of-bounds output node returns
/// InternalTranslationError instead of panicking (#313 AC5).
#[test]
fn test_kernel_to_graph_oob_output_returns_error() {
    use nn_dsl::ir::{IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType};
    // Build a valid-looking kernel but point output to non-existent node ID 99.
    let kernel = KernelDef::new(
        "oob_test",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![IRNode::new(NodeId::new(0), IRNodeKind::Param(0))],
        NodeId::new(99), // out of bounds: only 1 node exists
    );
    let result = kernel_to_graph(&kernel, &[]);
    assert!(
        result.is_err(),
        "out-of-bounds output should return error, not panic"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("out-of-bounds"),
        "error should mention 'out-of-bounds', got: {err}"
    );
}
