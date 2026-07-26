// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end correctness tests for SMT param binding and verification (#451).
//!
//! These tests verify the **semantic correctness** of the translation pipeline,
//! not just structural SMT-LIB2 properties. They would have caught #448
//! (param binding reversal) where constant and symbolic params were swapped.
//!
//! Test categories:
//! - Param assignment: correct Variable/Constant binding in SMT output
//! - Known-answer: exact-arithmetic kernel verifies correct output bounds
//! - Param reversal regression: non-commutative kernel detects swapped params

use super::*;
use crate::graph::ParamBinding;
use nn_dsl::ir::{BinOpKind, IRNode, IRNodeKind, NodeId, Param, ScalarType};
use ay_bindings::execute_direct::{self, ExecuteResult};

/// Build `fn sub_xy(x: f32, y: f32) -> f32 { x - y }`.
/// Non-commutative: swapping x and y changes the result.
fn sub_kernel() -> KernelDef {
    KernelDef::new(
        "sub_xy",
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
                IRNodeKind::BinOp {
                    op: BinOpKind::Sub,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    )
}

/// Build `fn add_xy(x: f32, y: f32) -> f32 { x + y }`.
fn add_xy_kernel() -> KernelDef {
    KernelDef::new(
        "add_xy",
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
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    )
}

/// Build `fn scale(x: f32, s: f32) -> f32 { x * s }`.
fn scale_kernel() -> KernelDef {
    KernelDef::new(
        "scale",
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("s", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    )
}

// --- AC1: Param assignment correctness ---

/// Verify that Variable params become symbolic and Constant params become literals.
///
/// For `add_xy(x, y)` with x=Variable, y=Constant(5.0):
/// - SMT output must contain `declare-const x` (symbolic)
/// - SMT output must NOT contain `declare-const y` (constant, not declared)
/// - param_exprs has exactly 2 entries (one per kernel param)
#[test]
fn test_param_binding_variable_vs_constant() {
    let kernel = add_xy_kernel();
    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(5.0)];
    let tr = translate_kernel(&kernel, &bindings).expect("should translate");

    let smt2 = tr.program.to_string();
    // x (param 0) is Variable → must be declared as symbolic
    assert!(
        smt2.contains("declare-const x"),
        "Variable param 'x' should be declared as symbolic in SMT.\nSMT:\n{smt2}"
    );
    // y (param 1) is Constant(5.0) → must NOT be declared
    assert!(
        !smt2.contains("declare-const y"),
        "Constant param 'y' should NOT be declared as symbolic.\nSMT:\n{smt2}"
    );
    // Both params produce expressions (symbolic for Variable, literal for Constant)
    assert_eq!(tr.param_exprs.len(), 2, "should have 2 param exprs");
    // No UF approximation needed (pure addition)
    assert!(!tr.uses_uf_approx, "add should not use UF approximation");
}

/// Semantic check: x=Variable, y=Constant(5.0), pin x=3.0 → output=8.0.
#[test]
fn test_param_binding_semantic_constant_substituted() {
    let kernel = add_xy_kernel();
    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(5.0)];
    let mut tr = translate_kernel(&kernel, &bindings).expect("should translate");

    // Pin x (the only symbolic param) to 3.0
    let x_expr = &tr.param_exprs[0];
    let three = real_from_f64(3.0).expect("encode 3.0");
    tr.program.assert(x_expr.clone().eq(three));

    // Assert output ≠ 8.0. If encoding is correct, UNSAT (output must be 8.0).
    let eight = real_from_f64(8.0).expect("encode 8.0");
    tr.program.assert(tr.output.clone().ne(eight));
    tr.program.set_logic("QF_LRA");
    tr.program.check_sat();

    let result = execute_direct::execute(&tr.program);
    assert!(
        matches!(
            result,
            Ok(ExecuteResult::Verified) | Ok(ExecuteResult::Unknown(_))
        ),
        "x=3, y=Constant(5): output should be 8.0, got: {result:?}"
    );
}

// --- AC2: Known-answer exact-arithmetic ---

/// Verify output bounds for `scale(x, s)` with s=Constant(2.0), x ∈ [-1, 1].
/// Expected output: [-2, 2]. SMT query: ∃x ∈ [-1,1] s.t. 2x < -2 ∨ 2x > 2.
/// Should be UNSAT (Proven) or Unknown (ay#5357).
#[test]
fn test_known_answer_scale_bounds() {
    let kernel = scale_kernel();
    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(2.0)];
    let mut tr = translate_kernel(&kernel, &bindings).expect("should translate");

    // Assert input bounds: x ∈ [-1, 1]
    let x = &tr.param_exprs[0];
    let lo = real_from_f64(-1.0).expect("encode -1");
    let hi = real_from_f64(1.0).expect("encode 1");
    tr.program.assert(x.clone().real_ge(lo));
    tr.program.assert(x.clone().real_le(hi));

    // Assert violation: output < -2 ∨ output > 2
    let out_lo = real_from_f64(-2.0).expect("encode -2");
    let out_hi = real_from_f64(2.0).expect("encode 2");
    let violation = tr
        .output
        .clone()
        .real_lt(out_lo)
        .or(tr.output.clone().real_gt(out_hi));
    tr.program.assert(violation);
    tr.program.set_logic("QF_LRA");
    tr.program.check_sat();

    let result = execute_direct::execute(&tr.program);
    assert!(
        matches!(
            result,
            Ok(ExecuteResult::Verified) | Ok(ExecuteResult::Unknown(_))
        ),
        "scale(x,2) with x∈[-1,1]: bounds [-2,2] should hold, got: {result:?}"
    );
}

/// Known-answer with intentionally wrong bounds: scale(x,2) with x ∈ [-1,1]
/// but bounds [-0.5, 0.5]. Should produce a counterexample (SAT).
#[test]
fn test_known_answer_scale_wrong_bounds_counterexample() {
    let kernel = scale_kernel();
    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(2.0)];
    let mut tr = translate_kernel(&kernel, &bindings).expect("should translate");

    let x = &tr.param_exprs[0];
    let lo = real_from_f64(-1.0).expect("encode -1");
    let hi = real_from_f64(1.0).expect("encode 1");
    tr.program.assert(x.clone().real_ge(lo));
    tr.program.assert(x.clone().real_le(hi));

    // Wrong bounds: [-0.5, 0.5] — too tight for 2x with x ∈ [-1,1]
    let out_lo = real_from_f64(-0.5).expect("encode -0.5");
    let out_hi = real_from_f64(0.5).expect("encode 0.5");
    let violation = tr
        .output
        .clone()
        .real_lt(out_lo)
        .or(tr.output.clone().real_gt(out_hi));
    tr.program.assert(violation);
    tr.program.set_logic("QF_LRA");
    tr.program.check_sat();

    let result = execute_direct::execute(&tr.program);
    // Solver should find a counterexample (x=1 → output=2 > 0.5)
    // Or Unknown due to ay#5357
    assert!(
        matches!(
            result,
            Ok(ExecuteResult::Counterexample { .. }) | Ok(ExecuteResult::Unknown(_))
        ),
        "scale(x,2) with x∈[-1,1]: bounds [-0.5,0.5] should fail, got: {result:?}"
    );
}

// --- AC3: Param reversal regression test ---

/// Regression test for #448: non-commutative kernel detects swapped params.
///
/// `sub_xy(x, y) = x - y`. With x=Variable, y=Constant(3.0):
/// - Pin x=10.0 → output should be 7.0 (10 - 3)
/// - If params were swapped (#448 bug), output would be -7.0 (3 - 10).
#[test]
fn test_param_reversal_sub_xy_catches_swap() {
    let kernel = sub_kernel();
    // x=Variable, y=Constant(3.0): output = x - 3
    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(3.0)];
    let mut tr = translate_kernel(&kernel, &bindings).expect("should translate");

    // Pin x=10
    let x_expr = &tr.param_exprs[0];
    let ten = real_from_f64(10.0).expect("encode 10");
    tr.program.assert(x_expr.clone().eq(ten));

    // Assert output ≠ 7. Correct encoding: 10 - 3 = 7 → UNSAT.
    // Swapped encoding would give 3 - 10 = -7 → SAT (counterexample).
    let seven = real_from_f64(7.0).expect("encode 7");
    tr.program.assert(tr.output.clone().ne(seven));
    tr.program.set_logic("QF_LRA");
    tr.program.check_sat();

    let result = execute_direct::execute(&tr.program);
    assert!(
        matches!(
            result,
            Ok(ExecuteResult::Verified) | Ok(ExecuteResult::Unknown(_))
        ),
        "sub(x=10, y=Const(3)): output should be 7, got: {result:?}\n\
         If SAT/Counterexample, params are likely swapped (#448)"
    );
}

/// Verify the swapped case would NOT prove: if we swap bindings for sub_xy,
/// pinning the constant to 10 and the variable to the symbolic one,
/// then output should be 3 - x, not x - 3.
#[test]
fn test_param_reversal_swapped_bindings_differ() {
    let kernel = sub_kernel();
    // Normal: x=Variable, y=Constant(3.0)
    let normal = vec![ParamBinding::Variable, ParamBinding::Constant(3.0)];
    let tr_normal = translate_kernel(&kernel, &normal).expect("normal");
    let smt_normal = tr_normal.program.to_string();

    // Swapped: x=Constant(3.0), y=Variable
    let swapped = vec![ParamBinding::Constant(3.0), ParamBinding::Variable];
    let tr_swapped = translate_kernel(&kernel, &swapped).expect("swapped");
    let smt_swapped = tr_swapped.program.to_string();

    // Normal declares x as symbolic; swapped declares y as symbolic
    assert!(
        smt_swapped.contains("declare-const y"),
        "Swapped bindings should declare y as symbolic.\nSMT:\n{smt_swapped}"
    );
    assert!(
        !smt_swapped.contains("declare-const x"),
        "Swapped bindings should NOT declare x (it's constant).\nSMT:\n{smt_swapped}"
    );
    assert!(
        smt_normal.contains("declare-const x"),
        "Normal bindings should declare x as symbolic.\nSMT:\n{smt_normal}"
    );
    assert!(
        !smt_normal.contains("declare-const y"),
        "Normal bindings should NOT declare y (it's constant).\nSMT:\n{smt_normal}"
    );
}
