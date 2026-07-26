// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Proof coverage tests for the nn-dsl kernel lowering pipeline.
//!
//! Covers error paths, unsupported operations, and lowering validation
//! identified during the Prover proof_coverage audit.
//!
//! IR validation, Display, MSL codegen, and precision tests are in
//! `proof_coverage_codegen.rs`.

use nn_dsl::ir::{IRNodeKind, MinMaxKind, UnaryFnKind};
use nn_dsl::{LowerError, Lowerer};

use nn_dsl::ir::KernelDef;

// ======================== Helpers ========================

fn parse_and_lower(src: &str) -> KernelDef {
    let func: syn::ItemFn = syn::parse_str(src).expect("parse source");
    Lowerer::lower_fn(&func).expect("lower to IR")
}

fn parse_and_expect_err(src: &str) -> LowerError {
    let func: syn::ItemFn = syn::parse_str(src).expect("parse source");
    Lowerer::lower_fn(&func).expect_err("should fail to lower")
}

// ======================== lower.rs error path tests ========================

#[test]
fn test_unsupported_binop_modulo() {
    let err = parse_and_expect_err("fn bad(x: f32, y: f32) -> f32 { x % y }");
    assert!(matches!(err, LowerError::UnsupportedBinOp));
}

#[test]
fn test_unsupported_type_i32() {
    let err = parse_and_expect_err("fn bad(x: i32) -> i32 { x }");
    assert!(matches!(err, LowerError::UnsupportedType(ref t) if t == "i32"));
}

#[test]
fn test_unsupported_type_bool() {
    let err = parse_and_expect_err("fn bad(x: bool) -> bool { x }");
    assert!(matches!(err, LowerError::UnsupportedType(ref t) if t == "bool"));
}

#[test]
fn test_unsupported_literal_string() {
    let err = parse_and_expect_err(r#"fn bad(x: f32) -> f32 { "hello"; x }"#);
    assert!(matches!(err, LowerError::UnsupportedLiteral));
}

#[test]
fn test_unsupported_path_multi_segment() {
    let err = parse_and_expect_err("fn bad(x: f32) -> f32 { std::f32::consts::PI }");
    assert!(matches!(err, LowerError::UnsupportedPath));
}

#[test]
fn test_self_param_rejected() {
    let func: syn::ItemFn = syn::parse_str("fn bad(&self) -> f32 { 1.0 }").expect("parse");
    let err = Lowerer::lower_fn(&func).expect_err("self param should fail");
    assert!(matches!(err, LowerError::SelfParam));
}

#[test]
fn test_uninitialized_let_rejected() {
    let err = parse_and_expect_err("fn bad(x: f32) -> f32 { let y: f32; x }");
    assert!(matches!(err, LowerError::UninitializedLet));
}

#[test]
fn test_empty_body_rejected() {
    let func: syn::ItemFn = syn::parse_str("fn bad(x: f32) -> f32 { let _y = x; }").expect("parse");
    let err = Lowerer::lower_fn(&func).expect_err("empty body should fail");
    assert!(matches!(err, LowerError::EmptyBody));
}

#[test]
fn test_if_expr_lowers_to_select() {
    let kernel = parse_and_lower("fn relu_if(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }");
    assert!(
        matches!(
            kernel.nodes[kernel.output.index()].kind,
            IRNodeKind::Select { .. }
        ),
        "if/else should lower to Select, got: {:?}",
        kernel.nodes[kernel.output.index()].kind
    );
}

#[test]
fn test_unsupported_unary_not() {
    let func: syn::ItemFn = syn::parse_str("fn bad(x: f32) -> f32 { !x }").expect("parse");
    let err = Lowerer::lower_fn(&func).expect_err("unary not should fail");
    assert!(matches!(err, LowerError::UnsupportedUnaryOp));
}

#[test]
fn test_missing_return_type_rejected() {
    let err = parse_and_expect_err("fn bad(x: f32) { x }");
    assert!(matches!(err, LowerError::MissingReturnType));
}

#[test]
fn test_unsupported_pattern_in_param() {
    let err = parse_and_expect_err("fn bad((x, y): (f32, f32)) -> f32 { x }");
    assert!(matches!(err, LowerError::UnsupportedPattern));
}

#[test]
fn test_unsupported_pattern_in_let() {
    let err = parse_and_expect_err("fn bad(x: f32) -> f32 { let (a, b) = (x, x); a }");
    assert!(matches!(err, LowerError::UnsupportedPattern));
}

#[test]
fn test_unsupported_pattern_in_typed_let() {
    // Exercises lower/mod.rs:164 — Pat::Type with non-Ident inner pattern
    let err = parse_and_expect_err("fn bad(x: f32) -> f32 { let (a, _b): (f32, f32) = (x, x); a }");
    assert!(matches!(err, LowerError::UnsupportedPattern));
}

#[test]
fn test_unsupported_statement_item() {
    let err = parse_and_expect_err("fn bad(x: f32) -> f32 { struct S; x }");
    assert!(matches!(err, LowerError::UnsupportedStatement));
}

#[test]
fn test_unknown_variable_rejected() {
    let err = parse_and_expect_err("fn bad(x: f32) -> f32 { y }");
    assert!(matches!(err, LowerError::UnknownVariable(ref name) if name == "y"));
}

#[test]
fn test_if_without_else_rejected() {
    let err = parse_and_expect_err("fn bad(x: f32) -> f32 { if x > 0.0 { x }; 0.0 }");
    assert!(matches!(err, LowerError::UnsupportedExpr(ref msg) if msg.contains("requires else")));
}

#[test]
fn test_wrong_arg_count_for_max() {
    let err = parse_and_expect_err("fn bad(x: f32) -> f32 { x.max(1.0, 2.0) }");
    assert!(matches!(
        err,
        LowerError::WrongArgCount {
            method,
            expected: 1,
            got: 2
        } if method == "max"
    ));
}

#[test]
fn test_wrong_arg_count_for_rsqrt() {
    let err = parse_and_expect_err("fn bad(x: f32) -> f32 { x.rsqrt(1.0) }");
    assert!(matches!(
        err,
        LowerError::WrongArgCount {
            method,
            expected: 0,
            got: 1
        } if method == "rsqrt"
    ));
}

// ======================== lower.rs operation tests ========================

#[test]
fn test_lower_min() {
    let kernel = parse_and_lower("fn capped(x: f32) -> f32 { x.min(1.0) }");
    assert_eq!(kernel.name, "capped");
    assert!(
        matches!(
            kernel.nodes[kernel.output.index()].kind,
            IRNodeKind::MinMax {
                op: MinMaxKind::Min,
                ..
            }
        ),
        "min output must be MinMax::Min, got: {:?}",
        kernel.nodes[kernel.output.index()].kind
    );
    kernel.validate().expect("valid IR");
}

#[test]
fn test_lower_cos() {
    let kernel = parse_and_lower("fn cosine(x: f32) -> f32 { x.cos() }");
    assert!(matches!(
        kernel.nodes[kernel.output.index()].kind,
        IRNodeKind::UnaryFn {
            op: UnaryFnKind::Cos,
            ..
        }
    ));
    kernel.validate().expect("valid IR");
}

#[test]
fn test_lower_sqrt() {
    let kernel = parse_and_lower("fn root(x: f32) -> f32 { x.sqrt() }");
    assert!(matches!(
        kernel.nodes[kernel.output.index()].kind,
        IRNodeKind::UnaryFn {
            op: UnaryFnKind::Sqrt,
            ..
        }
    ));
    kernel.validate().expect("valid IR");
}

#[test]
fn test_lower_exp() {
    let kernel = parse_and_lower("fn exponential(x: f32) -> f32 { x.exp() }");
    assert!(matches!(
        kernel.nodes[kernel.output.index()].kind,
        IRNodeKind::UnaryFn {
            op: UnaryFnKind::Exp,
            ..
        }
    ));
    kernel.validate().expect("valid IR");
}

#[test]
fn test_lower_abs() {
    let kernel = parse_and_lower("fn absolute(x: f32) -> f32 { x.abs() }");
    assert!(matches!(
        kernel.nodes[kernel.output.index()].kind,
        IRNodeKind::UnaryFn {
            op: UnaryFnKind::Abs,
            ..
        }
    ));
    kernel.validate().expect("valid IR");
}

#[test]
fn test_lower_rsqrt() {
    let kernel = parse_and_lower("fn inv_sqrt(x: f32) -> f32 { x.rsqrt() }");
    assert!(matches!(
        kernel.nodes[kernel.output.index()].kind,
        IRNodeKind::UnaryFn {
            op: UnaryFnKind::Rsqrt,
            ..
        }
    ));
    kernel.validate().expect("valid IR");
}

#[test]
fn test_lower_recip() {
    let kernel = parse_and_lower("fn inverse(x: f32) -> f32 { x.recip() }");
    assert!(matches!(
        kernel.nodes[kernel.output.index()].kind,
        IRNodeKind::UnaryFn {
            op: UnaryFnKind::Recip,
            ..
        }
    ));
    kernel.validate().expect("valid IR");
}

#[test]
fn test_lower_let_with_type_annotation() {
    let kernel = parse_and_lower("fn annotated(x: f32) -> f32 { let y: f32 = x * 2.0; y }");
    assert_eq!(kernel.name, "annotated");
    kernel.validate().expect("valid IR");
}
