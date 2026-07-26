// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Ground-folding tests (#376): verify that unary transcendental functions
//! applied to constant (ground) arguments are evaluated at translation time
//! and emitted as Real literals, not UF approximations.

use super::*;

/// Kernel: `fn rsqrt_k(x: f32, c: f32) -> f32 { rsqrt(c) + x }`
/// When c=4.0 (constant), rsqrt(4.0) = 0.5 → ground-folded to literal.
fn rsqrt_plus_x_kernel() -> KernelDef {
    KernelDef::new(
        "rsqrt_plus_x",
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("c", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Rsqrt,
                    input: NodeId::new(1),
                },
            ),
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(2),
                    rhs: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(3),
    )
}

/// Kernel: `fn sin_of_c(x: f32, c: f32) -> f32 { sin(c) + x }`
fn sin_plus_x_kernel() -> KernelDef {
    KernelDef::new(
        "sin_plus_x",
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("c", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Sin,
                    input: NodeId::new(1),
                },
            ),
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(2),
                    rhs: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(3),
    )
}

// --- Ground-folding: rsqrt on constant param ---

#[test]
fn test_rsqrt_on_constant_param_is_ground_folded() {
    // param 0 (x) = Variable, param 1 (c) = Constant(4.0).
    // rsqrt(4.0) = 0.5 → should be folded to Real literal, no UF.
    let kernel = rsqrt_plus_x_kernel();
    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(4.0)];
    let result = translate_kernel(&kernel, &bindings).expect("should translate");
    let smt2 = result.program.to_string();
    assert!(
        !smt2.contains("rsqrt_approx"),
        "rsqrt on constant arg should be ground-folded, not UF. SMT2:\n{smt2}"
    );
    assert!(
        !result.uses_uf_approx,
        "uses_uf_approx should be false when rsqrt is ground-folded"
    );
}

#[test]
fn test_rsqrt_on_symbolic_param_uses_uf() {
    // Both params Variable → rsqrt(x) is symbolic, must use UF.
    let kernel = rsqrt_plus_x_kernel();
    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    let result = translate_kernel(&kernel, &bindings).expect("should translate");
    let smt2 = result.program.to_string();
    assert!(
        smt2.contains("rsqrt_approx"),
        "rsqrt on symbolic arg should use UF approximation. SMT2:\n{smt2}"
    );
    assert!(
        result.uses_uf_approx,
        "uses_uf_approx should be true when rsqrt operates on symbolic arg"
    );
}

// --- Ground-folding: sin on constant param ---

#[test]
fn test_sin_on_constant_param_is_ground_folded() {
    // param 0 (x) = Variable, param 1 (c) = Constant(0.0).
    // sin(0.0) = 0.0 → should be folded to Real literal, no UF.
    let kernel = sin_plus_x_kernel();
    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(0.0)];
    let result = translate_kernel(&kernel, &bindings).expect("should translate");
    let smt2 = result.program.to_string();
    assert!(
        !smt2.contains("sin_approx"),
        "sin on constant arg should be ground-folded, not UF. SMT2:\n{smt2}"
    );
    assert!(
        !result.uses_uf_approx,
        "uses_uf_approx should be false when sin is ground-folded"
    );
}

#[test]
fn test_sin_on_symbolic_param_uses_uf() {
    // Both params Variable → sin(c) has c symbolic, must use UF.
    let kernel = sin_plus_x_kernel();
    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    let result = translate_kernel(&kernel, &bindings).expect("should translate");
    let smt2 = result.program.to_string();
    assert!(
        smt2.contains("sin_approx"),
        "sin on symbolic arg should use UF approximation. SMT2:\n{smt2}"
    );
    assert!(
        result.uses_uf_approx,
        "uses_uf_approx should be true when sin operates on symbolic arg"
    );
}

/// Generic kernel: `fn op_of_c(x: f32, c: f32) -> f32 { unary_op(c) + x }`
fn unary_plus_x_kernel(name: &str, op: UnaryFnKind) -> KernelDef {
    KernelDef::new(
        name,
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("c", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::UnaryFn {
                    op,
                    input: NodeId::new(1),
                },
            ),
            IRNode::new(
                NodeId::new(3),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(2),
                    rhs: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(3),
    )
}

// --- Ground-folding: cos on constant param ---

#[test]
fn test_cos_on_constant_param_is_ground_folded() {
    let kernel = unary_plus_x_kernel("cos_plus_x", UnaryFnKind::Cos);
    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(1.0)];
    let result = translate_kernel(&kernel, &bindings).expect("should translate");
    let smt2 = result.program.to_string();
    assert!(
        !smt2.contains("cos_approx"),
        "cos on constant arg should be ground-folded, not UF. SMT2:\n{smt2}"
    );
    assert!(!result.uses_uf_approx);
}

#[test]
fn test_cos_on_symbolic_param_uses_uf() {
    let kernel = unary_plus_x_kernel("cos_plus_x", UnaryFnKind::Cos);
    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    let result = translate_kernel(&kernel, &bindings).expect("should translate");
    let smt2 = result.program.to_string();
    assert!(
        smt2.contains("cos_approx"),
        "cos on symbolic arg should use UF. SMT2:\n{smt2}"
    );
    assert!(result.uses_uf_approx);
}

// --- Ground-folding: exp on constant param ---

#[test]
fn test_exp_on_constant_param_is_ground_folded() {
    // exp(1.0) ≈ 2.718 — well within real_from_f64 range.
    let kernel = unary_plus_x_kernel("exp_plus_x", UnaryFnKind::Exp);
    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(1.0)];
    let result = translate_kernel(&kernel, &bindings).expect("should translate");
    let smt2 = result.program.to_string();
    assert!(
        !smt2.contains("exp_approx"),
        "exp on constant arg should be ground-folded, not UF. SMT2:\n{smt2}"
    );
    assert!(!result.uses_uf_approx);
}

#[test]
fn test_exp_on_symbolic_param_uses_uf() {
    let kernel = unary_plus_x_kernel("exp_plus_x", UnaryFnKind::Exp);
    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    let result = translate_kernel(&kernel, &bindings).expect("should translate");
    let smt2 = result.program.to_string();
    assert!(
        smt2.contains("exp_approx"),
        "exp on symbolic arg should use UF. SMT2:\n{smt2}"
    );
    assert!(result.uses_uf_approx);
}

#[test]
fn test_exp_large_constant_falls_back_to_uf() {
    // exp(30.0) ≈ 1.07e13 — exceeds real_from_f64 safe range (~9.2e12).
    // Should fall through to UF approximation, not fail translation.
    let kernel = unary_plus_x_kernel("exp_plus_x", UnaryFnKind::Exp);
    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(30.0)];
    let result = translate_kernel(&kernel, &bindings).expect("should translate (UF fallback)");
    let smt2 = result.program.to_string();
    assert!(
        smt2.contains("exp_approx"),
        "exp(30.0) exceeds Real encoding range; should fall back to UF. SMT2:\n{smt2}"
    );
    assert!(result.uses_uf_approx);
}

// --- Ground-folding: sqrt on constant param ---

#[test]
fn test_sqrt_on_constant_param_is_ground_folded() {
    let kernel = unary_plus_x_kernel("sqrt_plus_x", UnaryFnKind::Sqrt);
    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(9.0)];
    let result = translate_kernel(&kernel, &bindings).expect("should translate");
    let smt2 = result.program.to_string();
    assert!(
        !smt2.contains("sqrt_approx"),
        "sqrt on constant arg should be ground-folded. SMT2:\n{smt2}"
    );
    assert!(!result.uses_uf_approx);
}

#[test]
fn test_sqrt_on_symbolic_param_uses_uf() {
    let kernel = unary_plus_x_kernel("sqrt_plus_x", UnaryFnKind::Sqrt);
    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    let result = translate_kernel(&kernel, &bindings).expect("should translate");
    let smt2 = result.program.to_string();
    assert!(
        smt2.contains("sqrt_approx"),
        "sqrt on symbolic arg should use UF. SMT2:\n{smt2}"
    );
    assert!(result.uses_uf_approx);
}

#[test]
fn test_sqrt_on_negative_constant_not_folded() {
    // sqrt(-1.0) is undefined → falls through to UF.
    let kernel = unary_plus_x_kernel("sqrt_plus_x", UnaryFnKind::Sqrt);
    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(-1.0)];
    let result = translate_kernel(&kernel, &bindings).expect("should translate");
    let smt2 = result.program.to_string();
    assert!(
        smt2.contains("sqrt_approx"),
        "sqrt(-1.0) is undefined; should fall through to UF. SMT2:\n{smt2}"
    );
    assert!(result.uses_uf_approx);
}

// --- Ground-folding: abs on constant param (exact, not UF) ---

#[test]
fn test_abs_on_constant_param_is_ground_folded() {
    let kernel = unary_plus_x_kernel("abs_plus_x", UnaryFnKind::Abs);
    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(-3.0)];
    let result = translate_kernel(&kernel, &bindings).expect("should translate");
    let smt2 = result.program.to_string();
    // Abs is exact (ite), not UF. When ground-folded, no ite needed.
    assert!(
        !result.uses_uf_approx,
        "abs should never set uses_uf_approx"
    );
    // Ground-folded abs(-3.0) = 3.0 → Real literal, no ite.
    assert!(
        !smt2.contains("ite") || smt2.contains("3"),
        "ground-folded abs(-3.0) should produce 3.0 literal"
    );
}

// --- Ground-folding: recip on constant param (exact, not UF) ---

#[test]
fn test_recip_on_constant_param_is_ground_folded() {
    let kernel = unary_plus_x_kernel("recip_plus_x", UnaryFnKind::Recip);
    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(4.0)];
    let result = translate_kernel(&kernel, &bindings).expect("should translate");
    assert!(!result.uses_uf_approx);
}

#[test]
fn test_recip_on_zero_constant_not_folded() {
    // recip(0.0) is undefined → falls through to exact encoding with x != 0 guard.
    let kernel = unary_plus_x_kernel("recip_plus_x", UnaryFnKind::Recip);
    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(0.0)];
    let result = translate_kernel(&kernel, &bindings).expect("should translate");
    assert!(!result.uses_uf_approx);
}

// --- Ground-folding: rsqrt domain guard ---

#[test]
fn test_rsqrt_on_zero_constant_not_folded() {
    // rsqrt(0.0) is undefined → eval_unary_ground returns None → falls through to UF path.
    let kernel = rsqrt_plus_x_kernel();
    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(0.0)];
    let result = translate_kernel(&kernel, &bindings).expect("should translate");
    let smt2 = result.program.to_string();
    assert!(
        smt2.contains("rsqrt_approx"),
        "rsqrt(0.0) is undefined; should fall through to UF, not fold. SMT2:\n{smt2}"
    );
    assert!(
        result.uses_uf_approx,
        "uses_uf_approx should be true when rsqrt(0.0) falls through to UF"
    );
}

#[test]
fn test_rsqrt_on_negative_constant_not_folded() {
    // rsqrt(-1.0) is undefined → eval_unary_ground returns None → UF path.
    let kernel = rsqrt_plus_x_kernel();
    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(-1.0)];
    let result = translate_kernel(&kernel, &bindings).expect("should translate");
    let smt2 = result.program.to_string();
    assert!(
        smt2.contains("rsqrt_approx"),
        "rsqrt(-1.0) is undefined; should fall through to UF. SMT2:\n{smt2}"
    );
}
