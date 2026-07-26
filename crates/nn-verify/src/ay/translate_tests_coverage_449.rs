// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Structural SMT encoding tests for Exp, Cos, Sub, and Compare variants (#449).
//!
//! Covers translation paths that had zero or only indirect test coverage:
//! - `UnaryFn::Exp` — `apply_positive_uf("exp_approx", ...)` with `result > 0` axiom
//! - `UnaryFn::Cos` — `apply_bounded_uf("cos_approx", ...)` with `[-1, 1]` axiom
//! - `BinOp::Sub` — `real_sub` encoding
//! - `Compare::{Lt, Le, Ge, Eq, Ne}` — 5 previously untested variants
//!
//! Semantic round-trip tests extracted to `translate_tests_coverage_449_semantic.rs`.

use super::*;
use nn_dsl::ir::{BinOpKind, CompareOpKind, UnaryFnKind};

// --- Exp encoding tests ---

/// exp(x) should declare exp_approx UF with positive range axiom (result > 0).
#[test]
fn test_smt_exp_encoding_declares_uf_with_positive_axiom() {
    let kernel = KernelDef::new(
        "exp_kernel",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Exp,
                    input: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(1),
    );
    let result =
        translate_kernel(&kernel, &all_variable(&kernel)).expect("exp kernel should translate");

    // Exp uses UF approximation.
    assert!(result.uses_uf_approx, "exp should use UF approximation");

    let smt2 = result.program.to_string();

    // Must declare exp_approx as a function Real -> Real.
    assert!(
        smt2.contains("exp_approx"),
        "exp kernel should declare exp_approx UF, got: {smt2}"
    );
    assert!(
        smt2.contains("declare-fun"),
        "exp kernel should use declare-fun for exp_approx, got: {smt2}"
    );

    // Positive range axiom: (> (exp_approx ...) 0) — exp(x) > 0 for all x.
    assert!(
        smt2.contains("(> (exp_approx"),
        "exp_approx should have positive range axiom (> (exp_approx ...) 0), got: {smt2}"
    );

    // Output expression should reference exp_approx.
    let output_smt2 = format!("{}", result.output);
    assert!(
        output_smt2.contains("exp_approx"),
        "exp kernel output should be exp_approx application, got: {output_smt2}"
    );
}

// --- Cos encoding tests ---

/// cos(x) should declare cos_approx UF with bounded range axiom [-1, 1].
#[test]
fn test_smt_cos_encoding_declares_uf_with_bounded_axiom() {
    let kernel = KernelDef::new(
        "cos_kernel",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Cos,
                    input: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(1),
    );
    let result =
        translate_kernel(&kernel, &all_variable(&kernel)).expect("cos kernel should translate");

    // Cos uses UF approximation.
    assert!(result.uses_uf_approx, "cos should use UF approximation");

    let smt2 = result.program.to_string();

    // Must declare cos_approx as a function Real -> Real.
    assert!(
        smt2.contains("cos_approx"),
        "cos kernel should declare cos_approx UF, got: {smt2}"
    );
    assert!(
        smt2.contains("declare-fun"),
        "cos kernel should use declare-fun for cos_approx, got: {smt2}"
    );

    // Bounded range axiom: cos_approx result in [-1, 1].
    // Should have (>= (cos_approx ...) -1) and (<= (cos_approx ...) 1).
    assert!(
        smt2.contains("(>= (cos_approx"),
        "cos_approx should have lower bound axiom (>= ... -1), got: {smt2}"
    );
    assert!(
        smt2.contains("(<= (cos_approx"),
        "cos_approx should have upper bound axiom (<= ... 1), got: {smt2}"
    );

    // Output expression should reference cos_approx.
    let output_smt2 = format!("{}", result.output);
    assert!(
        output_smt2.contains("cos_approx"),
        "cos kernel output should be cos_approx application, got: {output_smt2}"
    );
}

// --- BinOp::Sub encoding tests ---

/// x - 1.0 should produce subtraction in SMT encoding.
#[test]
fn test_smt_sub_encoding_contains_subtraction() {
    let kernel = KernelDef::new(
        "sub_one",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(1.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinOp {
                    op: BinOpKind::Sub,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    let result =
        translate_kernel(&kernel, &all_variable(&kernel)).expect("sub kernel should translate");

    // Sub is exact in Real arithmetic — no UF needed.
    assert!(
        !result.uses_uf_approx,
        "sub should not use UF approximation"
    );

    let output_smt2 = format!("{}", result.output);

    // Must contain subtraction operator.
    assert!(
        output_smt2.contains("(- "),
        "sub output should contain subtraction (- operator), got: {output_smt2}"
    );

    // Must reference the literal 1.0.
    assert!(
        output_smt2.contains("1.0"),
        "sub output should reference literal 1.0, got: {output_smt2}"
    );
}

// --- Compare variant encoding tests ---

/// Helper: build a kernel `fn f(x, y) -> f32 { if x <op> y { 1.0 } else { 0.0 } }`.
/// Uses Select to make the boolean comparison observable as an f32 output.
fn compare_select_kernel(name: &str, op: CompareOpKind) -> KernelDef {
    KernelDef::new(
        name,
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("y", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::Compare {
                    op,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            IRNode::new(NodeId::new(3), IRNodeKind::Literal(1.0)),
            IRNode::new(NodeId::new(4), IRNodeKind::Literal(0.0)),
            IRNode::new(
                NodeId::new(5),
                IRNodeKind::Select {
                    cond: NodeId::new(2),
                    then_val: NodeId::new(3),
                    else_val: NodeId::new(4),
                },
            ),
        ],
        NodeId::new(5),
    )
}

/// Compare::Lt should encode as (< x y).
#[test]
fn test_smt_compare_lt_encoding() {
    let kernel = compare_select_kernel("cmp_lt", CompareOpKind::Lt);
    let result =
        translate_kernel(&kernel, &all_variable(&kernel)).expect("Lt kernel should translate");
    let smt2 = result.program.to_string();
    let output_smt2 = format!("{}", result.output);

    assert!(
        smt2.contains("(< ") || output_smt2.contains("(< "),
        "Lt compare should produce (< operator, got SMT: {smt2}"
    );
    assert!(
        output_smt2.contains("ite"),
        "Lt select should produce ite, got: {output_smt2}"
    );
}

/// Compare::Le should encode as (<= x y).
#[test]
fn test_smt_compare_le_encoding() {
    let kernel = compare_select_kernel("cmp_le", CompareOpKind::Le);
    let result =
        translate_kernel(&kernel, &all_variable(&kernel)).expect("Le kernel should translate");
    let smt2 = result.program.to_string();
    let output_smt2 = format!("{}", result.output);

    assert!(
        smt2.contains("(<= ") || output_smt2.contains("(<= "),
        "Le compare should produce (<= operator, got SMT: {smt2}"
    );
    assert!(
        output_smt2.contains("ite"),
        "Le select should produce ite, got: {output_smt2}"
    );
}

/// Compare::Ge should encode as (>= x y).
#[test]
fn test_smt_compare_ge_encoding() {
    let kernel = compare_select_kernel("cmp_ge", CompareOpKind::Ge);
    let result =
        translate_kernel(&kernel, &all_variable(&kernel)).expect("Ge kernel should translate");
    let smt2 = result.program.to_string();
    let output_smt2 = format!("{}", result.output);

    assert!(
        smt2.contains("(>= ") || output_smt2.contains("(>= "),
        "Ge compare should produce (>= operator, got SMT: {smt2}"
    );
    assert!(
        output_smt2.contains("ite"),
        "Ge select should produce ite, got: {output_smt2}"
    );
}

/// Compare::Eq should encode as (= x y).
#[test]
fn test_smt_compare_eq_encoding() {
    let kernel = compare_select_kernel("cmp_eq", CompareOpKind::Eq);
    let result =
        translate_kernel(&kernel, &all_variable(&kernel)).expect("Eq kernel should translate");
    let smt2 = result.program.to_string();
    let output_smt2 = format!("{}", result.output);

    assert!(
        smt2.contains("(= ") || output_smt2.contains("(= "),
        "Eq compare should produce (= operator, got SMT: {smt2}"
    );
    assert!(
        output_smt2.contains("ite"),
        "Eq select should produce ite, got: {output_smt2}"
    );
}

/// Compare::Ne should encode as (not (= x y)) or (distinct x y).
#[test]
fn test_smt_compare_ne_encoding() {
    let kernel = compare_select_kernel("cmp_ne", CompareOpKind::Ne);
    let result =
        translate_kernel(&kernel, &all_variable(&kernel)).expect("Ne kernel should translate");
    let smt2 = result.program.to_string();
    let output_smt2 = format!("{}", result.output);

    let has_ne_encoding = smt2.contains("distinct")
        || output_smt2.contains("distinct")
        || smt2.contains("(not (= ")
        || output_smt2.contains("(not (= ");
    assert!(
        has_ne_encoding,
        "Ne compare should produce (distinct ...) or (not (= ...)), got SMT: {smt2}, output: {output_smt2}"
    );
    assert!(
        output_smt2.contains("ite"),
        "Ne select should produce ite, got: {output_smt2}"
    );
}
