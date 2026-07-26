// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Structural SMT encoding tests for Clamp, MinMax, Select, SumReduce (#396).
//!
//! Each test builds a minimal kernel IR, translates to SMT, and verifies
//! the output contains the expected encoding pattern (substring matching).
//! Semantic round-trip tests live in `translate_tests_node_kinds_semantic.rs`.

use super::*;
use nn_dsl::ir::{BinOpKind, CompareOpKind, MinMaxKind};

// --- Clamp encoding tests ---

/// Clamp(x, 0.0, 1.0) should produce nested ite: if x < 0 then 0 else if x > 1 then 1 else x.
#[test]
fn test_smt_clamp_encoding_contains_ite() {
    let kernel = KernelDef::new(
        "clamp_01",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(0.0)),
            IRNode::new(NodeId::new(2), IRNodeKind::Literal(1.0)),
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::Clamp {
                    input: NodeId::new(0),
                    min: NodeId::new(1),
                    max: NodeId::new(2),
                },
            ),
        ],
        NodeId::new(3),
    );
    let result =
        translate_kernel(&kernel, &all_variable(&kernel)).expect("clamp kernel should translate");
    let output_smt2 = format!("{}", result.output);

    // Clamp encodes as nested ite: (ite (< x 0.0) 0.0 (ite (> x 1.0) 1.0 x))
    // Must have exactly 2 ite operations (outer: below check, inner: above check).
    let ite_count = output_smt2.matches("(ite ").count();
    assert_eq!(
        ite_count, 2,
        "clamp should have exactly 2 nested ite operations, found {ite_count} in: {output_smt2}"
    );

    // Must use (< for the lower bound check (x < min).
    assert!(
        output_smt2.contains("(< "),
        "clamp should use (< for lower bound check, got: {output_smt2}"
    );

    // Must use (> for the upper bound check (x > max).
    assert!(
        output_smt2.contains("(> "),
        "clamp should use (> for upper bound check, got: {output_smt2}"
    );

    // Both 0.0 and 1.0 literals must appear in the encoding.
    assert!(
        output_smt2.contains("0.0") && output_smt2.contains("1.0"),
        "clamp output should reference both 0.0 and 1.0 literals, got: {output_smt2}"
    );

    // Verify bound ordering: the (< comparison should reference 0.0 (lo),
    // the (> comparison should reference 1.0 (hi).
    let lt_pos = output_smt2.find("(< ").expect("should have (<");
    let gt_pos = output_smt2.find("(> ").expect("should have (>");
    let zero_after_lt = output_smt2[lt_pos..].contains("0.0");
    let one_after_gt = output_smt2[gt_pos..].contains("1.0");
    assert!(
        zero_after_lt,
        "lower-bound (< should reference 0.0, got: {output_smt2}"
    );
    assert!(
        one_after_gt,
        "upper-bound (> should reference 1.0, got: {output_smt2}"
    );
}

// --- MinMax encoding tests ---

/// min(x, y) should produce ite(x <= y, x, y).
#[test]
fn test_smt_min_encoding_contains_ite() {
    let kernel = KernelDef::new(
        "min_xy",
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
                IRNodeKind::MinMax {
                    op: MinMaxKind::Min,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    let result =
        translate_kernel(&kernel, &all_variable(&kernel)).expect("min kernel should translate");
    let output_smt2 = format!("{}", result.output);

    // Min encodes as (ite (<= x y) x y) — exactly 1 ite with <= comparison.
    assert!(
        output_smt2.contains("(ite "),
        "min output should contain (ite, got: {output_smt2}"
    );
    assert!(
        output_smt2.contains("(<= "),
        "min should use (<= comparison, got: {output_smt2}"
    );
    // Must NOT use >= (that would be max).
    assert!(
        !output_smt2.contains("(>= "),
        "min should NOT use (>= comparison (that's max), got: {output_smt2}"
    );
    // Exactly 1 ite (not nested like clamp).
    let ite_count = output_smt2.matches("(ite ").count();
    assert_eq!(
        ite_count, 1,
        "min should have exactly 1 ite, found {ite_count} in: {output_smt2}"
    );
}

/// max(x, y) should produce ite(x >= y, x, y).
#[test]
fn test_smt_max_encoding_contains_ite() {
    let kernel = KernelDef::new(
        "max_xy",
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
                IRNodeKind::MinMax {
                    op: MinMaxKind::Max,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    let result =
        translate_kernel(&kernel, &all_variable(&kernel)).expect("max kernel should translate");
    let output_smt2 = format!("{}", result.output);

    // Max encodes as (ite (>= x y) x y) — exactly 1 ite with >= comparison.
    assert!(
        output_smt2.contains("(ite "),
        "max output should contain (ite, got: {output_smt2}"
    );
    assert!(
        output_smt2.contains("(>= "),
        "max should use (>= comparison, got: {output_smt2}"
    );
    // Must NOT use <= (that would be min).
    assert!(
        !output_smt2.contains("(<= "),
        "max should NOT use (<= comparison (that's min), got: {output_smt2}"
    );
    // Exactly 1 ite (not nested like clamp).
    let ite_count = output_smt2.matches("(ite ").count();
    assert_eq!(
        ite_count, 1,
        "max should have exactly 1 ite, found {ite_count} in: {output_smt2}"
    );
}

// --- Select encoding tests ---

/// select(x > 0, x, -x) (manual abs) should produce ite with the condition.
#[test]
fn test_smt_select_encoding_contains_ite() {
    // Build: select(x > 0, x, 0 - x) — manual abs via select.
    let kernel = KernelDef::new(
        "select_abs",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(0.0)),
            // cond: x > 0
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::Compare {
                    op: CompareOpKind::Gt,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            // else: 0 - x (negation)
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::BinOp {
                    op: BinOpKind::Sub,
                    lhs: NodeId::new(1),
                    rhs: NodeId::new(0),
                },
            ),
            // select(cond, x, 0-x)
            IRNode::new(
                NodeId::new(4),
                IRNodeKind::Select {
                    cond: NodeId::new(2),
                    then_val: NodeId::new(0),
                    else_val: NodeId::new(3),
                },
            ),
        ],
        NodeId::new(4),
    );
    let result =
        translate_kernel(&kernel, &all_variable(&kernel)).expect("select kernel should translate");
    let output_smt2 = format!("{}", result.output);
    // Select encodes as ite(cond, then, else).
    assert!(
        output_smt2.contains("ite"),
        "select output should contain ite, got: {output_smt2}"
    );
    // The condition uses > comparison.
    let smt2 = result.program.to_string();
    assert!(
        smt2.contains("(> ") || output_smt2.contains("(> "),
        "select condition should contain > comparison, got SMT: {smt2}"
    );
}

// --- SumReduce encoding tests ---

/// SumReduce of two inputs should produce addition.
#[test]
fn test_smt_sum_reduce_encoding_contains_addition() {
    let kernel = KernelDef::new(
        "sum_xy",
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
                IRNodeKind::SumReduce {
                    inputs: vec![NodeId::new(0), NodeId::new(1)],
                },
            ),
        ],
        NodeId::new(2),
    );
    let result = translate_kernel(&kernel, &all_variable(&kernel))
        .expect("sum_reduce kernel should translate");
    let output_smt2 = format!("{}", result.output);
    // SumReduce of 2 inputs encodes as (+ x y).
    assert!(
        output_smt2.contains("(+ "),
        "sum_reduce output should contain addition (+), got: {output_smt2}"
    );
}

/// SumReduce of three inputs should chain additions.
#[test]
fn test_smt_sum_reduce_three_inputs() {
    let kernel = KernelDef::new(
        "sum_xyz",
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("y", ScalarType::F32),
            Param::new("z", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(NodeId::new(2), IRNodeKind::Param(2)),
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::SumReduce {
                    inputs: vec![NodeId::new(0), NodeId::new(1), NodeId::new(2)],
                },
            ),
        ],
        NodeId::new(3),
    );
    let result = translate_kernel(&kernel, &all_variable(&kernel))
        .expect("sum_reduce 3 kernel should translate");
    let output_smt2 = format!("{}", result.output);
    // Sum of 3 inputs: (+ (+ x y) z) — should contain nested addition.
    let add_count = output_smt2.matches("(+ ").count();
    assert!(
        add_count >= 2,
        "sum_reduce of 3 inputs should have >= 2 additions, found {add_count} in: {output_smt2}"
    );
}

/// SumReduce with a single input should produce just that input (no addition).
#[test]
fn test_smt_sum_reduce_single_input() {
    let kernel = KernelDef::new(
        "sum_x",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::SumReduce {
                    inputs: vec![NodeId::new(0)],
                },
            ),
        ],
        NodeId::new(1),
    );
    let result = translate_kernel(&kernel, &all_variable(&kernel))
        .expect("sum_reduce single kernel should translate");
    let output_smt2 = format!("{}", result.output);
    // Single-input sum should just be the input itself — no addition operator.
    assert!(
        !output_smt2.contains("(+ "),
        "sum_reduce of 1 input should not contain addition, got: {output_smt2}"
    );
}
