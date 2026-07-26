// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `translate_uf.rs` — powi translation and UF approximation.

use super::*;
use std::collections::HashSet;
use ay_bindings::{Expr, Sort, AYProgram};

fn setup() -> (AYProgram, Sort, HashSet<String>, bool) {
    let program = AYProgram::new();
    let real_sort = Sort::real();
    let declared_ufs = HashSet::new();
    let uses_uf_approx = false;
    (program, real_sort, declared_ufs, uses_uf_approx)
}

// ---------------------------------------------------------------------------
// translate_powi — exact binary exponentiation
// ---------------------------------------------------------------------------

#[test]
fn test_powi_exp_0_returns_one() {
    let (mut program, real_sort, mut declared_ufs, mut uses_uf_approx) = setup();
    let base = Expr::real(5);
    let result = translate_powi(
        base,
        0,
        &mut program,
        &real_sort,
        &mut declared_ufs,
        &mut uses_uf_approx,
    );
    assert!(result.is_ok());
    assert!(!uses_uf_approx, "exp=0 should not use UF");
}

#[test]
fn test_powi_exp_1_returns_base() {
    let (mut program, real_sort, mut declared_ufs, mut uses_uf_approx) = setup();
    let base = Expr::real(7);
    let result = translate_powi(
        base,
        1,
        &mut program,
        &real_sort,
        &mut declared_ufs,
        &mut uses_uf_approx,
    );
    assert!(result.is_ok());
    assert!(!uses_uf_approx, "exp=1 should not use UF");
}

#[test]
fn test_powi_exp_2_exact() {
    let (mut program, real_sort, mut declared_ufs, mut uses_uf_approx) = setup();
    let base = Expr::real(3);
    let result = translate_powi(
        base,
        2,
        &mut program,
        &real_sort,
        &mut declared_ufs,
        &mut uses_uf_approx,
    );
    assert!(result.is_ok());
    assert!(!uses_uf_approx, "exp=2 should use binary exponentiation");
}

#[test]
fn test_powi_exp_32_still_exact() {
    let (mut program, real_sort, mut declared_ufs, mut uses_uf_approx) = setup();
    let base = Expr::real(2);
    let result = translate_powi(
        base,
        32,
        &mut program,
        &real_sort,
        &mut declared_ufs,
        &mut uses_uf_approx,
    );
    assert!(result.is_ok());
    assert!(!uses_uf_approx, "exp=32 should use binary exponentiation");
}

#[test]
fn test_powi_exp_33_uses_uf() {
    let (mut program, real_sort, mut declared_ufs, mut uses_uf_approx) = setup();
    let base = Expr::real(2);
    let result = translate_powi(
        base,
        33,
        &mut program,
        &real_sort,
        &mut declared_ufs,
        &mut uses_uf_approx,
    );
    assert!(result.is_ok());
    assert!(uses_uf_approx, "exp=33 should fall back to UF");
    assert!(declared_ufs.contains("powi_33_approx"));
}

// ---------------------------------------------------------------------------
// translate_powi — negative exponents
// ---------------------------------------------------------------------------

#[test]
fn test_powi_exp_neg_1_exact() {
    let (mut program, real_sort, mut declared_ufs, mut uses_uf_approx) = setup();
    let base = Expr::real(4);
    let result = translate_powi(
        base,
        -1,
        &mut program,
        &real_sort,
        &mut declared_ufs,
        &mut uses_uf_approx,
    );
    assert!(result.is_ok());
    assert!(!uses_uf_approx, "exp=-1 should use exact reciprocal");
}

#[test]
fn test_powi_exp_neg_2_exact() {
    let (mut program, real_sort, mut declared_ufs, mut uses_uf_approx) = setup();
    let base = Expr::real(3);
    let result = translate_powi(
        base,
        -2,
        &mut program,
        &real_sort,
        &mut declared_ufs,
        &mut uses_uf_approx,
    );
    assert!(result.is_ok());
    assert!(
        !uses_uf_approx,
        "exp=-2 should use exact binary exp + reciprocal"
    );
}

#[test]
fn test_powi_exp_neg_33_uses_uf() {
    let (mut program, real_sort, mut declared_ufs, mut uses_uf_approx) = setup();
    let base = Expr::real(2);
    let result = translate_powi(
        base,
        -33,
        &mut program,
        &real_sort,
        &mut declared_ufs,
        &mut uses_uf_approx,
    );
    assert!(result.is_ok());
    assert!(uses_uf_approx, "exp=-33 should fall back to UF");
    assert!(declared_ufs.contains("powi_-33_approx"));
}

// ---------------------------------------------------------------------------
// declare_uf_if_needed — idempotency
// ---------------------------------------------------------------------------

#[test]
fn test_declare_uf_first_call_inserts() {
    let (mut program, real_sort, mut declared_ufs, _) = setup();
    declare_uf_if_needed("sin_approx", &mut program, &real_sort, &mut declared_ufs);
    assert!(declared_ufs.contains("sin_approx"));
}

#[test]
fn test_declare_uf_second_call_is_noop() {
    let (mut program, real_sort, mut declared_ufs, _) = setup();
    declare_uf_if_needed("cos_approx", &mut program, &real_sort, &mut declared_ufs);
    assert_eq!(declared_ufs.len(), 1);
    // Second call should not add again.
    declare_uf_if_needed("cos_approx", &mut program, &real_sort, &mut declared_ufs);
    assert_eq!(declared_ufs.len(), 1);
}

#[test]
fn test_declare_uf_different_names() {
    let (mut program, real_sort, mut declared_ufs, _) = setup();
    declare_uf_if_needed("sin_approx", &mut program, &real_sort, &mut declared_ufs);
    declare_uf_if_needed("cos_approx", &mut program, &real_sort, &mut declared_ufs);
    assert_eq!(declared_ufs.len(), 2);
}

// ---------------------------------------------------------------------------
// apply_bounded_uf
// ---------------------------------------------------------------------------

#[test]
fn test_apply_bounded_uf_succeeds() {
    let (mut program, real_sort, mut declared_ufs, _) = setup();
    let arg = Expr::real(1);
    let result = apply_bounded_uf(
        "sin_approx",
        arg,
        &mut program,
        &real_sort,
        &mut declared_ufs,
        -1,
        1,
    );
    assert!(result.is_ok());
    assert!(declared_ufs.contains("sin_approx"));
}

// ---------------------------------------------------------------------------
// apply_positive_uf
// ---------------------------------------------------------------------------

#[test]
fn test_apply_positive_uf_succeeds() {
    let (mut program, real_sort, mut declared_ufs, _) = setup();
    let arg = Expr::real(1);
    let result = apply_positive_uf(
        "exp_approx",
        arg,
        &mut program,
        &real_sort,
        &mut declared_ufs,
    );
    assert!(result.is_ok());
    assert!(declared_ufs.contains("exp_approx"));
}

// ---------------------------------------------------------------------------
// apply_nonneg_uf
// ---------------------------------------------------------------------------

#[test]
fn test_apply_nonneg_uf_succeeds() {
    let (mut program, real_sort, mut declared_ufs, _) = setup();
    let arg = Expr::real(1);
    let result = apply_nonneg_uf(
        "sqrt_approx",
        arg,
        &mut program,
        &real_sort,
        &mut declared_ufs,
    );
    assert!(result.is_ok());
    assert!(declared_ufs.contains("sqrt_approx"));
}

// ---------------------------------------------------------------------------
// translate_powi — even vs odd UF exponents
// ---------------------------------------------------------------------------

#[test]
fn test_powi_large_even_positive_uf_nonneg_constraint() {
    // Even positive exponent: x^34 >= 0 for all x.
    let (mut program, real_sort, mut declared_ufs, mut uses_uf_approx) = setup();
    let base = Expr::real(2);
    let result = translate_powi(
        base,
        34,
        &mut program,
        &real_sort,
        &mut declared_ufs,
        &mut uses_uf_approx,
    );
    assert!(result.is_ok());
    assert!(uses_uf_approx);
    assert!(declared_ufs.contains("powi_34_approx"));
}

#[test]
fn test_powi_large_even_negative_uf_positive_constraint() {
    // Even negative exponent: x^-34 = 1/x^34 > 0 (strictly positive).
    let (mut program, real_sort, mut declared_ufs, mut uses_uf_approx) = setup();
    let base = Expr::real(2);
    let result = translate_powi(
        base,
        -34,
        &mut program,
        &real_sort,
        &mut declared_ufs,
        &mut uses_uf_approx,
    );
    assert!(result.is_ok());
    assert!(uses_uf_approx);
    assert!(declared_ufs.contains("powi_-34_approx"));
}

#[test]
fn test_powi_large_odd_positive_no_sign_constraint() {
    // Odd positive exponent: x^33 can be negative, so no range constraint.
    let (mut program, real_sort, mut declared_ufs, mut uses_uf_approx) = setup();
    let base = Expr::real(2);
    let result = translate_powi(
        base,
        33,
        &mut program,
        &real_sort,
        &mut declared_ufs,
        &mut uses_uf_approx,
    );
    assert!(result.is_ok());
    assert!(uses_uf_approx);
}

// ---------------------------------------------------------------------------
// encode_f16_cast — F16 downcast UF encoding (#3023)
// ---------------------------------------------------------------------------

#[test]
fn test_encode_f16_cast_declares_uf() {
    let (mut program, real_sort, mut declared_ufs, _) = setup();
    let input = Expr::real(42);
    let _result = encode_f16_cast(input, &mut program, &real_sort, &mut declared_ufs);
    assert!(declared_ufs.contains("f16_cast"));
}

#[test]
fn test_encode_f16_cast_idempotent_declaration() {
    let (mut program, real_sort, mut declared_ufs, _) = setup();
    let input1 = Expr::real(1);
    let input2 = Expr::real(2);
    let _r1 = encode_f16_cast(input1, &mut program, &real_sort, &mut declared_ufs);
    let _r2 = encode_f16_cast(input2, &mut program, &real_sort, &mut declared_ufs);
    // Should still have exactly 1 UF declared (idempotent).
    assert_eq!(declared_ufs.len(), 1);
}

// ---------------------------------------------------------------------------
// encode_bf16_cast — BF16 downcast UF encoding (#3023)
// ---------------------------------------------------------------------------

#[test]
fn test_encode_bf16_cast_declares_uf() {
    let (mut program, real_sort, mut declared_ufs, _) = setup();
    let input = Expr::real(42);
    let _result = encode_bf16_cast(input, &mut program, &real_sort, &mut declared_ufs);
    assert!(declared_ufs.contains("bf16_cast"));
}

#[test]
fn test_encode_f16_and_bf16_cast_separate_ufs() {
    let (mut program, real_sort, mut declared_ufs, _) = setup();
    let input = Expr::real(1);
    let _f16 = encode_f16_cast(input.clone(), &mut program, &real_sort, &mut declared_ufs);
    let _bf16 = encode_bf16_cast(input, &mut program, &real_sort, &mut declared_ufs);
    assert_eq!(declared_ufs.len(), 2);
    assert!(declared_ufs.contains("f16_cast"));
    assert!(declared_ufs.contains("bf16_cast"));
}
