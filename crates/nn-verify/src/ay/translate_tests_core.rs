// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Core translation tests: identity, add_one, scale, sin UF, no params error,
//! powi binary exponentiation, abs.

use super::*;

#[test]
fn test_translate_identity() {
    let kernel = identity_kernel();
    let result = translate_kernel(&kernel, &all_variable(&kernel))
        .expect("identity kernel should translate");
    assert!(!result.uses_uf_approx);
    assert_eq!(result.param_exprs.len(), 1);
    // Identity kernel output should be the same expression as the input parameter.
    // Both should Display as the declared const name "x".
    let output_smt2 = format!("{}", result.output);
    let param_smt2 = format!("{}", result.param_exprs[0]);
    assert_eq!(
        output_smt2, param_smt2,
        "identity output should equal its input param, got output={output_smt2}, param={param_smt2}"
    );
    // The program should declare x as a Real const
    let prog_smt2 = result.program.to_string();
    assert!(
        prog_smt2.contains("declare-const"),
        "identity should declare x as a const, got: {prog_smt2}"
    );
}

#[test]
fn test_translate_add_one() {
    let kernel = add_one_kernel();
    let result =
        translate_kernel(&kernel, &all_variable(&kernel)).expect("add_one kernel should translate");
    assert!(!result.uses_uf_approx);
    // Should produce a program with Real arithmetic
    let smt2 = result.program.to_string();
    assert!(smt2.contains("declare-const"));
    assert!(smt2.contains("QF_UFNRA"));
}

#[test]
fn test_translate_with_constant_param() {
    let kernel = KernelDef::new(
        "scale",
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("alpha", ScalarType::F32),
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
    );
    // x=2.0 is constant (index 0), alpha is variable (index 1)
    let result = translate_kernel(
        &kernel,
        &[ParamBinding::Constant(2.0), ParamBinding::Variable],
    )
    .expect("scale kernel should translate");
    assert!(!result.uses_uf_approx);
    assert_eq!(result.param_exprs.len(), 2);
    // alpha (index 1) should be declared as a symbolic const
    let smt2 = result.program.to_string();
    assert!(
        smt2.contains("alpha"),
        "alpha should be declared in SMT-LIB2, got: {smt2}"
    );
    // The output expression should contain multiplication and the constant 2.0
    let output_smt2 = format!("{}", result.output);
    assert!(
        output_smt2.contains("(* "),
        "scale output should contain multiplication, got: {output_smt2}"
    );
    assert!(
        output_smt2.contains("2.0"),
        "scale output should contain the constant value 2.0, got: {output_smt2}"
    );
}

#[test]
fn test_translate_with_sin_uses_uf() {
    let kernel = KernelDef::new(
        "sin_kernel",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Sin,
                    input: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(1),
    );
    let result =
        translate_kernel(&kernel, &all_variable(&kernel)).expect("sin kernel should translate");
    assert!(result.uses_uf_approx);
    let smt2 = result.program.to_string();
    assert!(
        smt2.contains("sin_approx"),
        "sin kernel should declare sin_approx UF, got: {smt2}"
    );
    // The output expression should reference sin_approx applied to x
    let output_smt2 = format!("{}", result.output);
    assert!(
        output_smt2.contains("sin_approx"),
        "sin kernel output should be a sin_approx application, got: {output_smt2}"
    );
    // sin_approx should be declared as a function Real -> Real
    assert!(
        smt2.contains("declare-fun"),
        "sin kernel should declare sin_approx as a function, got: {smt2}"
    );
}

#[test]
fn test_translate_no_params_error() {
    let kernel = KernelDef::new(
        "empty",
        vec![],
        ScalarType::F32,
        vec![IRNode::new(NodeId::new(0), IRNodeKind::Literal(1.0))],
        NodeId::new(0),
    );
    let result = translate_kernel(&kernel, &all_variable(&kernel));
    assert!(result.is_err(), "zero-param kernel should return Err");
    let err = result.expect_err("already checked is_err");
    assert!(
        matches!(err, SmtError::NoParameters),
        "zero-param kernel should produce NoParameters, got: {err}"
    );
}

#[test]
fn test_translate_powi_small() {
    let kernel = powi_kernel("square", 2);
    let result =
        translate_kernel(&kernel, &all_variable(&kernel)).expect("powi(2) kernel should translate");
    // powi(2) is expanded to x*x, no UF needed
    assert!(!result.uses_uf_approx);
    // Output should contain multiplication (binary exponentiation of x^2 = x*x)
    let output_smt2 = format!("{}", result.output);
    assert!(
        output_smt2.contains("(* "),
        "powi(2) output should contain multiplication, got: {output_smt2}"
    );
    // No declare-fun needed (no UF)
    let prog_smt2 = result.program.to_string();
    assert!(
        !prog_smt2.contains("declare-fun"),
        "powi(2) should not declare any UFs, got: {prog_smt2}"
    );
}

#[test]
fn test_translate_powi_binary_exp_16() {
    let kernel = powi_kernel("pow16", 16);
    let result = translate_kernel(&kernel, &all_variable(&kernel))
        .expect("powi(16) kernel should translate");
    // powi(16) uses binary exponentiation — exact, no UF
    assert!(!result.uses_uf_approx);
    let smt2 = result.program.to_string();
    // Should not use a UF call
    assert!(
        !smt2.contains("powi_16_approx"),
        "powi(16) should not use UF, got: {smt2}"
    );
    // Binary exponentiation of x^16 produces a chain of squarings:
    // x^2, x^4, x^8, x^16 — so multiple multiplication ops in output.
    let output_smt2 = format!("{}", result.output);
    let mul_count = output_smt2.matches("(* ").count();
    assert!(
        mul_count >= 4,
        "powi(16) via binary exp should produce >= 4 multiplications (x^2,x^4,x^8,x^16), got {mul_count}: {output_smt2}"
    );
}

#[test]
fn test_translate_powi_binary_exp_32_boundary() {
    let kernel = powi_kernel("pow32", 32);
    let result = translate_kernel(&kernel, &all_variable(&kernel))
        .expect("powi(32) kernel should translate");
    // powi(32) is the max exact exponent — no UF
    assert!(!result.uses_uf_approx);
    let smt2 = result.program.to_string();
    assert!(
        !smt2.contains("powi_32_approx"),
        "powi(32) should not use UF, got: {smt2}"
    );
    // Binary exponentiation of x^32 produces squarings: x^2,x^4,x^8,x^16,x^32
    let output_smt2 = format!("{}", result.output);
    let mul_count = output_smt2.matches("(* ").count();
    assert!(
        mul_count >= 5,
        "powi(32) via binary exp should produce >= 5 multiplications, got {mul_count}: {output_smt2}"
    );
}

#[test]
fn test_translate_powi_33_falls_back_to_uf() {
    let kernel = powi_kernel("pow33", 33);
    let result = translate_kernel(&kernel, &all_variable(&kernel))
        .expect("powi(33) kernel should translate");
    // powi(33) exceeds MAX_EXACT_POWI_EXP — UF fallback
    assert!(result.uses_uf_approx);
    assert!(result.program.to_string().contains("powi_33_approx"));
}

#[test]
fn test_translate_powi_negative_16_exact() {
    let kernel = powi_kernel("pow_neg16", -16);
    let result = translate_kernel(&kernel, &all_variable(&kernel))
        .expect("powi(-16) kernel should translate");
    // powi(-16) uses binary exponentiation + reciprocal — exact, no UF
    assert!(!result.uses_uf_approx);
    assert!(
        !result.program.to_string().contains("approx"),
        "powi(-16) should not use any UF approximation"
    );
    // Negative exponent: output = 1 / (x^16), so division wrapping multiplications
    let output_smt2 = format!("{}", result.output);
    assert!(
        output_smt2.contains("(/ "),
        "powi(-16) output should contain division (reciprocal), got: {output_smt2}"
    );
    assert!(
        output_smt2.contains("(* "),
        "powi(-16) output should contain multiplication (binary exp), got: {output_smt2}"
    );
    assert!(
        output_smt2.contains("1.0"),
        "powi(-16) output should contain numerator 1.0 for reciprocal, got: {output_smt2}"
    );
}

#[test]
fn test_translate_powi_negative_33_falls_back_to_uf() {
    let kernel = powi_kernel("pow_neg33", -33);
    let result = translate_kernel(&kernel, &all_variable(&kernel))
        .expect("powi(-33) kernel should translate");
    // powi(-33) exceeds limit — UF fallback
    assert!(result.uses_uf_approx);
    assert!(result.program.to_string().contains("powi_-33_approx"));
}

#[test]
fn test_translate_powi_even_uf_has_nonneg_axiom() {
    // Even exponent UF (e.g., powi(34)) should have >= 0 axiom
    let kernel = powi_kernel("pow34", 34);
    let result = translate_kernel(&kernel, &all_variable(&kernel))
        .expect("powi(34) kernel should translate");
    assert!(result.uses_uf_approx);
    let smt2 = result.program.to_string();
    assert!(
        smt2.contains("powi_34_approx"),
        "powi(34) should use UF approximation"
    );
    // Non-negative axiom: (>= (powi_34_approx ...) 0.0)
    assert!(
        smt2.contains("(>= (powi_34_approx"),
        "even powi UF should have >= 0 range axiom, got: {smt2}"
    );
}

#[test]
fn test_translate_powi_odd_uf_has_no_range_axiom() {
    // Odd exponent UF (e.g., powi(33)) should NOT have a range axiom
    let kernel = powi_kernel("pow33_odd", 33);
    let result = translate_kernel(&kernel, &all_variable(&kernel))
        .expect("powi(33) odd kernel should translate");
    assert!(result.uses_uf_approx);
    let smt2 = result.program.to_string();
    assert!(
        !smt2.contains("(>= (powi_33_approx"),
        "odd powi UF should not have non-negative axiom, got: {smt2}"
    );
    assert!(
        !smt2.contains("(> (powi_33_approx"),
        "odd powi UF should not have positive axiom, got: {smt2}"
    );
}

#[test]
fn test_translate_powi_neg_even_uf_has_positive_axiom() {
    // Negative even exponent UF (e.g., powi(-34)) is 1/x^34 > 0
    let kernel = powi_kernel("pow_neg34", -34);
    let result = translate_kernel(&kernel, &all_variable(&kernel))
        .expect("powi(-34) kernel should translate");
    assert!(result.uses_uf_approx);
    let smt2 = result.program.to_string();
    assert!(
        smt2.contains("(> (powi_-34_approx"),
        "negative even powi UF should have > 0 range axiom, got: {smt2}"
    );
}

#[test]
fn test_translate_abs_exact() {
    let kernel = KernelDef::new(
        "abs_kernel",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Abs,
                    input: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(1),
    );
    let result =
        translate_kernel(&kernel, &all_variable(&kernel)).expect("abs kernel should translate");
    // Abs is exact in Real arithmetic, no UF
    assert!(!result.uses_uf_approx);
}
