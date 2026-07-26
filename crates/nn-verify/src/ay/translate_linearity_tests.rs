// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `translate_linearity.rs` — non-linearity detection.

use super::*;
use crate::graph::ParamBinding;
use nn_dsl::ir::BinOpKind;
use nn_dsl::test_kernels::{
    binop_var_var_kernel, parse_kernel, square_kernel, sub_kernel, unary_fn_kernel,
};

// ---------------------------------------------------------------------------
// Linear kernels (should return false)
// ---------------------------------------------------------------------------

#[test]
fn test_linear_addition() {
    // fn f(x, y) -> f32 { x + y }
    let kernel = binop_var_var_kernel(BinOpKind::Add);
    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    assert!(
        !kernel_uses_nonlinear(&kernel, &bindings),
        "addition of two variables is linear"
    );
}

#[test]
fn test_linear_subtraction() {
    // fn f(a, b) -> f32 { a - b }
    let kernel = sub_kernel();
    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    assert!(
        !kernel_uses_nonlinear(&kernel, &bindings),
        "subtraction of two variables is linear"
    );
}

#[test]
fn test_linear_mul_by_constant() {
    // fn f(x, y) -> f32 { x * y } with y constant
    let kernel = binop_var_var_kernel(BinOpKind::Mul);
    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(2.0)];
    assert!(
        !kernel_uses_nonlinear(&kernel, &bindings),
        "variable * constant is linear"
    );
}

#[test]
fn test_linear_div_by_constant() {
    // fn f(x, y) -> f32 { x / y } with y constant
    let kernel = binop_var_var_kernel(BinOpKind::Div);
    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(3.0)];
    assert!(
        !kernel_uses_nonlinear(&kernel, &bindings),
        "variable / constant is linear"
    );
}

#[test]
fn test_linear_constant_mul_variable() {
    // fn f(x, y) -> f32 { x * y } with x constant
    let kernel = binop_var_var_kernel(BinOpKind::Mul);
    let bindings = vec![ParamBinding::Constant(5.0), ParamBinding::Variable];
    assert!(
        !kernel_uses_nonlinear(&kernel, &bindings),
        "constant * variable is linear"
    );
}

#[test]
fn test_linear_all_constants() {
    // fn f(x, y) -> f32 { x * y } with both constant
    let kernel = binop_var_var_kernel(BinOpKind::Mul);
    let bindings = vec![ParamBinding::Constant(2.0), ParamBinding::Constant(3.0)];
    assert!(
        !kernel_uses_nonlinear(&kernel, &bindings),
        "constant * constant is ground"
    );
}

#[test]
fn test_linear_unary_abs() {
    let kernel = unary_fn_kernel(nn_dsl::ir::UnaryFnKind::Abs);
    let bindings = vec![ParamBinding::Variable];
    // Abs of a single variable does not create a non-linear Mul/Div/Powi.
    // (It's a non-linear function conceptually, but translate_linearity only
    // checks for symbolic Mul/Div and Powi, not unary functions.)
    assert!(!kernel_uses_nonlinear(&kernel, &bindings));
}

// ---------------------------------------------------------------------------
// Non-linear kernels (should return true)
// ---------------------------------------------------------------------------

#[test]
fn test_nonlinear_var_times_var() {
    // fn f(x, y) -> f32 { x * y } — both symbolic
    let kernel = binop_var_var_kernel(BinOpKind::Mul);
    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    assert!(
        kernel_uses_nonlinear(&kernel, &bindings),
        "variable * variable is non-linear"
    );
}

#[test]
fn test_nonlinear_var_div_var() {
    // fn f(x, y) -> f32 { x / y } — both symbolic
    let kernel = binop_var_var_kernel(BinOpKind::Div);
    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    assert!(
        kernel_uses_nonlinear(&kernel, &bindings),
        "variable / variable is non-linear"
    );
}

#[test]
fn test_nonlinear_square() {
    // fn square(x) -> f32 { x * x }
    let kernel = square_kernel();
    let bindings = vec![ParamBinding::Variable];
    assert!(
        kernel_uses_nonlinear(&kernel, &bindings),
        "x * x is non-linear"
    );
}

#[test]
fn test_nonlinear_powi_exp_2() {
    // fn f(x) -> f32 { x.powi(2) }
    let kernel = parse_kernel("fn f(x: f32) -> f32 { x.powi(2) }");
    let bindings = vec![ParamBinding::Variable];
    assert!(
        kernel_uses_nonlinear(&kernel, &bindings),
        "powi(2) on symbolic base is non-linear"
    );
}

#[test]
fn test_nonlinear_powi_exp_3() {
    // fn f(x) -> f32 { x.powi(3) }
    let kernel = parse_kernel("fn f(x: f32) -> f32 { x.powi(3) }");
    let bindings = vec![ParamBinding::Variable];
    assert!(kernel_uses_nonlinear(&kernel, &bindings));
}

#[test]
fn test_nonlinear_snake_kernel() {
    // Snake: x + (1/alpha) * sin²(alpha*x) — contains alpha*x Mul (both symbolic)
    let kernel = parse_kernel(
        "fn snake(x: f32, alpha: f32) -> f32 { x + (1.0 / alpha) * (alpha * x).sin().powi(2) }",
    );
    let bindings = vec![ParamBinding::Variable, ParamBinding::Variable];
    assert!(
        kernel_uses_nonlinear(&kernel, &bindings),
        "snake with both params symbolic is non-linear"
    );
}

// ---------------------------------------------------------------------------
// Powi edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_powi_exp_1_is_linear() {
    // powi(1) is the identity function — not detected as non-linear.
    let kernel = parse_kernel("fn f(x: f32) -> f32 { x.powi(1) }");
    let bindings = vec![ParamBinding::Variable];
    assert!(
        !kernel_uses_nonlinear(&kernel, &bindings),
        "powi(1) is linear"
    );
}

#[test]
fn test_powi_exp_0_is_linear() {
    // powi(0) is always 1 — not detected as non-linear.
    let kernel = parse_kernel("fn f(x: f32) -> f32 { x.powi(0) }");
    let bindings = vec![ParamBinding::Variable];
    assert!(
        !kernel_uses_nonlinear(&kernel, &bindings),
        "powi(0) is constant"
    );
}

#[test]
fn test_powi_exp_negative_1_is_linear() {
    // powi(-1) has |exp| = 1 — the threshold is > 1, so this is linear.
    let kernel = parse_kernel("fn f(x: f32) -> f32 { x.powi(-1) }");
    let bindings = vec![ParamBinding::Variable];
    assert!(
        !kernel_uses_nonlinear(&kernel, &bindings),
        "powi(-1) has |exp| == 1, not detected as non-linear"
    );
}

#[test]
fn test_powi_exp_negative_2_is_nonlinear() {
    // powi(-2) has |exp| = 2 > 1 — non-linear.
    let kernel = parse_kernel("fn f(x: f32) -> f32 { x.powi(-2) }");
    let bindings = vec![ParamBinding::Variable];
    assert!(kernel_uses_nonlinear(&kernel, &bindings));
}

#[test]
fn test_powi_constant_base_is_ground() {
    // powi on a constant base is ground (not non-linear).
    let kernel = parse_kernel("fn f(x: f32, a: f32) -> f32 { x + a.powi(3) }");
    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(2.0)];
    assert!(
        !kernel_uses_nonlinear(&kernel, &bindings),
        "powi on constant base is ground"
    );
}

// ---------------------------------------------------------------------------
// Edge cases: empty and constant-only kernels
// ---------------------------------------------------------------------------

#[test]
fn test_empty_bindings_treats_params_as_non_ground() {
    // No bindings provided — all params are treated as non-ground.
    let kernel = binop_var_var_kernel(BinOpKind::Mul);
    let bindings: Vec<ParamBinding> = vec![];
    // With empty bindings, bindings.get(i) returns None → treated as non-ground.
    // So x * y with both non-ground → non-linear.
    assert!(kernel_uses_nonlinear(&kernel, &bindings));
}

#[test]
fn test_snake_with_alpha_constant_is_still_nonlinear() {
    // Snake with alpha constant: sin²(alpha*x) still involves powi(2) on a symbolic base.
    let kernel = parse_kernel(
        "fn snake(x: f32, alpha: f32) -> f32 { x + (1.0 / alpha) * (alpha * x).sin().powi(2) }",
    );
    let bindings = vec![ParamBinding::Variable, ParamBinding::Constant(1.0)];
    // Even with alpha constant, sin(alpha*x) depends on x (symbolic),
    // and sin(...).powi(2) has a symbolic base with |exp| > 1.
    assert!(kernel_uses_nonlinear(&kernel, &bindings));
}
