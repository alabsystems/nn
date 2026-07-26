// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SMT content verification tests: powi content, add_one content, sin range
//! axioms, abs encoding, divisor non-zero guards (#243), constant count
//! validation (#254).

use super::*;

// --- powi content verification tests ---

#[test]
fn test_translate_powi_2_produces_multiply() {
    // powi(2) should expand to x*x — verify the output expression contains multiplication
    let kernel = powi_kernel("square", 2);
    let result =
        translate_kernel(&kernel, &all_variable(&kernel)).expect("powi(2) kernel should translate");
    let output_smt2 = format!("{}", result.output);
    // The output expression should be (* ... (* x x) ...) pattern
    assert!(
        output_smt2.contains("(* "),
        "powi(2) output expression should contain Real multiplication, got: {output_smt2}"
    );
}

#[test]
fn test_translate_powi_neg2_produces_reciprocal() {
    // powi(-2) should be 1/(x*x) — verify division and multiplication present
    let kernel = powi_kernel("inv_square", -2);
    let result = translate_kernel(&kernel, &all_variable(&kernel))
        .expect("powi(-2) kernel should translate");
    let output_smt2 = format!("{}", result.output);
    assert!(
        output_smt2.contains("(/ ") && output_smt2.contains("(* "),
        "powi(-2) output should contain (/ 1.0 (* ...)), got: {output_smt2}"
    );
}

// --- add_one SMT content verification ---

#[test]
fn test_translate_add_one_output_is_x_plus_literal() {
    // add_one(x) = x + 1.0 — the output expression should contain (+ x 1.0)
    let kernel = add_one_kernel();
    let result =
        translate_kernel(&kernel, &all_variable(&kernel)).expect("add_one kernel should translate");
    let output_smt2 = format!("{}", result.output);
    assert!(
        output_smt2.contains("(+ ") && output_smt2.contains("1.0"),
        "add_one output should encode as (+ x 1.0), got: {output_smt2}"
    );
}

// --- sin UF range axiom verification ---

#[test]
fn test_translate_sin_has_range_axioms() {
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
    let smt2 = result.program.to_string();
    // sin_approx should have range axioms: result >= -1 and result <= 1
    assert!(
        smt2.contains("(>= (sin_approx") || smt2.contains("(assert (>= (sin_approx"),
        "sin should have lower range axiom (>= sin_approx ... -1), got: {smt2}"
    );
    assert!(
        smt2.contains("(<= (sin_approx") || smt2.contains("(assert (<= (sin_approx"),
        "sin should have upper range axiom (<= sin_approx ... 1), got: {smt2}"
    );
}

// --- abs encoding verification ---

#[test]
fn test_translate_abs_encodes_as_ite() {
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
    // Abs should encode as ite(x >= 0, x, -x)
    let output_smt2 = format!("{}", result.output);
    assert!(
        output_smt2.contains("ite") && output_smt2.contains(">="),
        "abs should encode as ite(x >= 0, x, -x), got: {output_smt2}"
    );
}

// --- divisor non-zero guard tests (#243) ---

#[test]
fn test_translate_div_asserts_divisor_nonzero() {
    let kernel = div_kernel();
    let result =
        translate_kernel(&kernel, &all_variable(&kernel)).expect("div kernel should translate");
    let smt2 = result.program.to_string();
    // The program must assert (not (= y 0.0)) to guard against (/ x 0).
    // .ne() produces (not (= a b)) in SMT-LIB2.
    let has_nonzero_guard = smt2.contains("(not (=") && smt2.contains("0.0");
    assert!(
        has_nonzero_guard,
        "div kernel must assert divisor != 0 via (not (= y 0.0)), got: {smt2}"
    );
}

#[test]
fn test_translate_recip_asserts_arg_nonzero() {
    let kernel = KernelDef::new(
        "recip_x",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Recip,
                    input: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(1),
    );
    let result =
        translate_kernel(&kernel, &all_variable(&kernel)).expect("recip kernel should translate");
    let smt2 = result.program.to_string();
    // recip(x) = 1/x must assert x != 0
    let has_nonzero_guard = smt2.contains("(not (=") && smt2.contains("0.0");
    assert!(
        has_nonzero_guard,
        "recip kernel must assert arg != 0, got: {smt2}"
    );
}

#[test]
fn test_translate_powi_negative_asserts_base_nonzero() {
    let kernel = powi_kernel("inv_square", -2);
    let result = translate_kernel(&kernel, &all_variable(&kernel))
        .expect("powi(-2) kernel should translate");
    let smt2 = result.program.to_string();
    // powi(-2) computes 1/(x*x), must assert x*x != 0
    let has_nonzero_guard = smt2.contains("(not (=") && smt2.contains("0.0");
    assert!(
        has_nonzero_guard,
        "negative powi must assert base^n != 0, got: {smt2}"
    );
}

// --- constant_params count validation tests (#254) ---

#[test]
fn test_translate_all_constants_no_variables_rejected() {
    // identity_kernel has 1 param (x). All-Constant bindings → no symbolic vars.
    let kernel = identity_kernel();
    let err = translate_kernel(&kernel, &[ParamBinding::Constant(1.0)]).unwrap_err();
    assert!(
        matches!(err, SmtError::ParamCountMismatch { ir_count: 1, .. }),
        "expected ParamCountMismatch for 1-param kernel with 0 variables, got: {err:?}"
    );
}

#[test]
fn test_translate_all_params_constant_rejected() {
    // 2-param kernel with 2 constants = 0 symbolic variables.
    let kernel = div_kernel(); // fn f(x, y) -> f32 { x / y } — 2 params
    let err = translate_kernel(
        &kernel,
        &[ParamBinding::Constant(1.0), ParamBinding::Constant(2.0)],
    )
    .unwrap_err();
    assert!(
        matches!(err, SmtError::ParamCountMismatch { ir_count: 2, .. }),
        "expected ParamCountMismatch for 2-param kernel with 0 variables, got: {err:?}"
    );
}

#[test]
fn test_translate_fewer_constants_allowed() {
    // 2-param kernel with 0 constants = both symbolic (multi-variable mode).
    // This should succeed — ay supports multi-variable SMT unlike NY.
    let kernel = div_kernel();
    let result = translate_kernel(&kernel, &all_variable(&kernel));
    assert!(
        result.is_ok(),
        "0 constants for 2-param kernel should be allowed, got: {result:?}"
    );
}

#[test]
fn test_translate_exact_constants_allowed() {
    // 2-param kernel with 1 constant = 1 symbolic (single-variable mode).
    // Build a simple 2-param add kernel: fn f(x, alpha) -> f32 { x + alpha }
    let kernel = KernelDef::new(
        "add_alpha",
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
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    let result = translate_kernel(
        &kernel,
        &[ParamBinding::Variable, ParamBinding::Constant(2.0)],
    );
    assert!(
        result.is_ok(),
        "1 constant + 1 variable for 2-param kernel should succeed, got: {result:?}"
    );
}

// --- malformed IR rejection tests (#282) ---

#[test]
fn test_translate_out_of_bounds_node_ref_returns_err() {
    // Construct a kernel with a BinOp referencing NodeId::new(99) which doesn't exist.
    // translate_kernel should return IrValidation error (not panic).
    let kernel = KernelDef::new(
        "bad_ref",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(99), // out of bounds
                },
            ),
        ],
        NodeId::new(1),
    );
    let err = translate_kernel(&kernel, &all_variable(&kernel)).unwrap_err();
    assert!(
        matches!(err, SmtError::IrValidation(_)),
        "out-of-bounds NodeId should produce IrValidation error, got: {err:?}"
    );
}

#[test]
fn test_translate_out_of_bounds_output_returns_err() {
    // Construct a kernel with output pointing to non-existent node.
    let kernel = KernelDef::new(
        "bad_output",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![IRNode::new(NodeId::new(0), IRNodeKind::Param(0))],
        NodeId::new(99), // out of bounds
    );
    let err = translate_kernel(&kernel, &all_variable(&kernel)).unwrap_err();
    assert!(
        matches!(err, SmtError::IrValidation(_)),
        "out-of-bounds output NodeId should produce IrValidation error, got: {err:?}"
    );
}

#[test]
fn test_translate_out_of_bounds_param_idx_returns_err() {
    // Construct a kernel with Param(5) but only 1 parameter declared.
    let kernel = KernelDef::new(
        "bad_param",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![IRNode::new(NodeId::new(0), IRNodeKind::Param(5))],
        NodeId::new(0),
    );
    let err = translate_kernel(&kernel, &all_variable(&kernel)).unwrap_err();
    assert!(
        matches!(err, SmtError::IrValidation(_)),
        "out-of-bounds param index should produce IrValidation error, got: {err:?}"
    );
}

// --- UF domain precondition tests (#388) ---

#[test]
fn test_sqrt_uf_domain_precondition_asserted() {
    // sqrt_approx UF must assert arg >= 0 as domain precondition.
    let kernel = KernelDef::new(
        "sqrt_kernel",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Sqrt,
                    input: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(1),
    );
    let result =
        translate_kernel(&kernel, &all_variable(&kernel)).expect("sqrt kernel should translate");
    let smt2 = result.program.to_string();
    // Domain: arg >= 0 (assert (>= x 0.0))
    // Plus range axiom: result >= 0 (from apply_nonneg_uf)
    // The SMT output should contain ">=" with "0.0" appearing for the domain guard.
    // We verify both domain and range axioms are present by counting >= assertions.
    let ge_count = smt2.matches("(>= ").count();
    assert!(
        ge_count >= 2,
        "sqrt should have both domain (arg >= 0) and range (result >= 0) axioms, \
         found {ge_count} >= assertion(s) in: {smt2}"
    );
}

#[test]
fn test_rsqrt_uf_domain_precondition_asserted() {
    // rsqrt_approx UF must assert arg > 0 as domain precondition.
    let kernel = KernelDef::new(
        "rsqrt_kernel",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::UnaryFn {
                    op: UnaryFnKind::Rsqrt,
                    input: NodeId::new(0),
                },
            ),
        ],
        NodeId::new(1),
    );
    let result =
        translate_kernel(&kernel, &all_variable(&kernel)).expect("rsqrt kernel should translate");
    let smt2 = result.program.to_string();
    // Domain: arg > 0 (assert (> x 0.0))
    // Range: result > 0 (from apply_positive_uf)
    // Both produce (> ... 0.0) assertions.
    let gt_count = smt2.matches("(> ").count();
    assert!(
        gt_count >= 2,
        "rsqrt should have both domain (arg > 0) and range (result > 0) axioms, \
         found {gt_count} > assertion(s) in: {smt2}"
    );
}

#[test]
fn test_powi_large_negative_odd_domain_precondition() {
    // powi(-33) falls through to UF path (|exp| > 32). Odd negative exponent
    // must assert base != 0 since x^(-33) = 1/x^33 is undefined at x = 0.
    let kernel = powi_kernel("powi_neg33", -33);
    let result = translate_kernel(&kernel, &all_variable(&kernel))
        .expect("powi(-33) kernel should translate");
    let smt2 = result.program.to_string();
    // Domain: base != 0 encoded as (assert (not (= x 0.0)))
    let has_nonzero_guard = smt2.contains("(not (=") && smt2.contains("0.0");
    assert!(
        has_nonzero_guard,
        "powi(-33) UF must assert base != 0, got: {smt2}"
    );
    // Also verify UF was used (not exact expansion)
    assert!(
        smt2.contains("powi_-33_approx"),
        "powi(-33) should use UF approximation, got: {smt2}"
    );
}
