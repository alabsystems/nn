// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Semantic round-trip tests for Clamp, MinMax, Select, SumReduce (#415).
//!
//! Unlike the structural tests in `translate_tests_node_kinds.rs` (which check
//! SMT-LIB2 substring patterns), these tests verify **semantic correctness**:
//! they pin symbolic inputs to concrete values and use ay direct execution to
//! prove the SMT encoding evaluates to the expected output.
//!
//! Pattern: translate → pin inputs → assert(output ≠ expected) → UNSAT.
//! UNSAT means: no assignment exists where output differs from expected,
//! i.e., the encoding is semantically correct for those inputs.

use super::*;
use nn_dsl::ir::{BinOpKind, CompareOpKind, MinMaxKind};
use ay_bindings::execute_direct::{self, ExecuteResult};

/// Helper: translate a kernel, pin each symbolic param to `inputs[i]`,
/// assert output ≠ `expected`, run ay direct, and verify UNSAT.
///
/// Tolerates `Unknown` due to ay#5357 (execute_direct regression).
fn assert_semantic_roundtrip(kernel: &KernelDef, inputs: &[f64], expected: f64) {
    let mut tr = translate_kernel(kernel, &all_variable(kernel)).expect("kernel should translate");
    assert_eq!(
        tr.param_exprs.len(),
        inputs.len(),
        "input count must match param count"
    );

    // Pin each symbolic param to its concrete value.
    for (param_expr, &val) in tr.param_exprs.iter().zip(inputs) {
        let concrete = real_from_f64(val).expect("input value should encode");
        tr.program.assert(param_expr.clone().eq(concrete));
    }

    // Assert output ≠ expected. If encoding is correct, this is UNSAT.
    let expected_expr = real_from_f64(expected).expect("expected value should encode");
    tr.program.assert(tr.output.clone().ne(expected_expr));

    // Override logic for direct execution compatibility (same as prove.rs).
    tr.program.set_logic("QF_LRA");
    tr.program.check_sat();

    let result = execute_direct::execute(&tr.program);
    assert!(
        matches!(
            result,
            Ok(ExecuteResult::Verified) | Ok(ExecuteResult::Unknown(_))
        ),
        "semantic roundtrip: expected Verified or Unknown (ay#5357), got: {result:?}\n\
         kernel={}, inputs={inputs:?}, expected={expected}",
        kernel.name
    );
}

// --- Clamp semantic tests ---

/// Helper: build clamp(x, 0.0, 1.0) kernel.
fn clamp_01_kernel() -> KernelDef {
    KernelDef::new(
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
    )
}

/// Clamp(3.0, 0.0, 1.0) → 1.0 (clamped to upper bound).
#[test]
fn test_semantic_clamp_above_upper() {
    assert_semantic_roundtrip(&clamp_01_kernel(), &[3.0], 1.0);
}

/// Clamp(-0.5, 0.0, 1.0) → 0.0 (clamped to lower bound).
#[test]
fn test_semantic_clamp_below_lower() {
    assert_semantic_roundtrip(&clamp_01_kernel(), &[-0.5], 0.0);
}

/// Clamp(0.5, 0.0, 1.0) → 0.5 (within range, passthrough).
#[test]
fn test_semantic_clamp_within_range() {
    assert_semantic_roundtrip(&clamp_01_kernel(), &[0.5], 0.5);
}

// --- MinMax semantic tests ---

/// min(3.0, 5.0) → 3.0.
#[test]
fn test_semantic_min_known_answer() {
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
    assert_semantic_roundtrip(&kernel, &[3.0, 5.0], 3.0);
}

/// max(3.0, 5.0) → 5.0.
#[test]
fn test_semantic_max_known_answer() {
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
    assert_semantic_roundtrip(&kernel, &[3.0, 5.0], 5.0);
}

// --- Select semantic tests ---

/// Helper: build select(x > 0, x, 0-x) kernel (manual abs).
fn select_abs_kernel() -> KernelDef {
    KernelDef::new(
        "select_abs",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(0.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::Compare {
                    op: CompareOpKind::Gt,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::BinOp {
                    op: BinOpKind::Sub,
                    lhs: NodeId::new(1),
                    rhs: NodeId::new(0),
                },
            ),
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
    )
}

/// select(x > 0, x, 0-x) with x=3.0 → 3.0 (true branch).
#[test]
fn test_semantic_select_true_branch() {
    assert_semantic_roundtrip(&select_abs_kernel(), &[3.0], 3.0);
}

/// select(x > 0, x, 0-x) with x=-2.0 → 2.0 (false branch, negation).
#[test]
fn test_semantic_select_false_branch() {
    assert_semantic_roundtrip(&select_abs_kernel(), &[-2.0], 2.0);
}

// --- SumReduce semantic tests ---

/// sum_reduce(3.0, 5.0) → 8.0.
#[test]
fn test_semantic_sum_reduce_known_answer() {
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
    assert_semantic_roundtrip(&kernel, &[3.0, 5.0], 8.0);
}
