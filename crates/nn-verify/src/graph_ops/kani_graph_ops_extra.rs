// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extra Kani proof harnesses for graph_ops numerical correctness.
//!
//! These harnesses prove cross-cutting properties that span multiple
//! graph_ops submodules: FiniteF32 type safety, checked_constant
//! NaN/Inf rejection, compare output bounds, select-compare convention
//! alignment, and numerical edge cases.
//!
//! Part of #3603.

use crate::error::VerifyError;
use crate::graph::{checked_constant, FiniteF32, NodeValue};
use nn_dsl::ir::CompareOpKind;

use super::evaluate_constant_compare;

// ---------------------------------------------------------------------------
// FiniteF32 type-safety harnesses
// ---------------------------------------------------------------------------

/// Proves `FiniteF32::new` rejects NaN.
/// IEEE 754: NaN bypasses relational comparisons, so explicit `is_finite()`
/// check is the only safe guard. Source: #3356.
#[kani::unwind(1)]
#[kani::proof]
fn finite_f32_rejects_nan() {
    let result = FiniteF32::new(f32::NAN);
    assert!(result.is_err(), "FiniteF32::new must reject NaN");
}

/// Proves `FiniteF32::new` rejects positive infinity.
#[kani::unwind(1)]
#[kani::proof]
fn finite_f32_rejects_pos_infinity() {
    let result = FiniteF32::new(f32::INFINITY);
    assert!(result.is_err(), "FiniteF32::new must reject +Inf");
}

/// Proves `FiniteF32::new` rejects negative infinity.
#[kani::unwind(1)]
#[kani::proof]
fn finite_f32_rejects_neg_infinity() {
    let result = FiniteF32::new(f32::NEG_INFINITY);
    assert!(result.is_err(), "FiniteF32::new must reject -Inf");
}

/// Proves `FiniteF32::new` accepts any finite f32 and round-trips via `get()`.
#[kani::unwind(1)]
#[kani::proof]
fn finite_f32_accepts_and_roundtrips_finite() {
    let val: f32 = kani::any();
    kani::assume(val.is_finite());

    let f = FiniteF32::new(val).expect("FiniteF32::new must accept finite values");
    assert_eq!(
        f.get().to_bits(),
        val.to_bits(),
        "FiniteF32::get must return the original value bit-exact"
    );
}

/// Proves `FiniteF32::new` rejects every non-finite f32 value.
/// Exhaustive: covers NaN, +Inf, -Inf via symbolic f32.
#[kani::unwind(1)]
#[kani::proof]
fn finite_f32_rejects_all_non_finite() {
    let val: f32 = kani::any();
    kani::assume(!val.is_finite());

    let result = FiniteF32::new(val);
    assert!(
        result.is_err(),
        "FiniteF32::new must reject all non-finite f32 values"
    );
}

// ---------------------------------------------------------------------------
// checked_constant harnesses
// ---------------------------------------------------------------------------

/// Proves `checked_constant` rejects NaN, producing `NonFiniteConstant` error.
/// Uses `unwind(8)` to bound syn::ErrorMessage Drop unwinding (#608).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn checked_constant_rejects_nan() {
    let result = checked_constant(f32::NAN, "kani_nan_test");
    assert!(result.is_err(), "checked_constant must reject NaN");
}

/// Proves `checked_constant` rejects positive infinity.
/// Uses `unwind(8)` — syn::ErrorMessage Drop unwinding (#608).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn checked_constant_rejects_pos_inf() {
    let result = checked_constant(f32::INFINITY, "kani_inf_test");
    assert!(result.is_err(), "checked_constant must reject +Inf");
}

/// Proves `checked_constant` rejects negative infinity.
/// Uses `unwind(8)` — syn::ErrorMessage Drop unwinding (#608).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn checked_constant_rejects_neg_inf() {
    let result = checked_constant(f32::NEG_INFINITY, "kani_neg_inf_test");
    assert!(result.is_err(), "checked_constant must reject -Inf");
}

/// Proves `checked_constant` rejects every non-finite f32 value (symbolic).
/// Uses `unwind(8)` — syn::ErrorMessage Drop unwinding (#608).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn checked_constant_rejects_all_non_finite() {
    let val: f32 = kani::any();
    kani::assume(!val.is_finite());

    let result = checked_constant(val, "kani_non_finite_test");
    assert!(
        result.is_err(),
        "checked_constant must reject all non-finite f32 values"
    );
}

// ---------------------------------------------------------------------------
// Compare output bounds harnesses
// ---------------------------------------------------------------------------

/// Proves `evaluate_constant_compare` with Gt returns exactly 0.0 or 1.0
/// and correctly identifies strict ordering: Gt(a,b) == 1.0 iff a > b.
#[kani::unwind(1)]
#[kani::proof]
fn compare_gt_correct() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());

    let result = evaluate_constant_compare(CompareOpKind::Gt, a, b)
        .expect("Gt on finite values must not fail");
    if a > b {
        assert_eq!(
            result.to_bits(),
            1.0f32.to_bits(),
            "Gt(a,b) with a > b must be 1.0"
        );
    } else {
        assert_eq!(
            result.to_bits(),
            0.0f32.to_bits(),
            "Gt(a,b) with a <= b must be 0.0"
        );
    }
}

/// Proves `evaluate_constant_compare` with Lt returns exactly 0.0 or 1.0
/// and correctly identifies strict ordering: Lt(a,b) == 1.0 iff a < b.
#[kani::unwind(1)]
#[kani::proof]
fn compare_lt_correct() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());

    let result = evaluate_constant_compare(CompareOpKind::Lt, a, b)
        .expect("Lt on finite values must not fail");
    if a < b {
        assert_eq!(
            result.to_bits(),
            1.0f32.to_bits(),
            "Lt(a,b) with a < b must be 1.0"
        );
    } else {
        assert_eq!(
            result.to_bits(),
            0.0f32.to_bits(),
            "Lt(a,b) with a >= b must be 0.0"
        );
    }
}

/// Proves `evaluate_constant_compare` with Le returns exactly 0.0 or 1.0
/// and correctly identifies non-strict ordering: Le(a,b) == 1.0 iff a <= b.
#[kani::unwind(1)]
#[kani::proof]
fn compare_le_correct() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());

    let result = evaluate_constant_compare(CompareOpKind::Le, a, b)
        .expect("Le on finite values must not fail");
    if a <= b {
        assert_eq!(
            result.to_bits(),
            1.0f32.to_bits(),
            "Le(a,b) with a <= b must be 1.0"
        );
    } else {
        assert_eq!(
            result.to_bits(),
            0.0f32.to_bits(),
            "Le(a,b) with a > b must be 0.0"
        );
    }
}

/// Proves `evaluate_constant_compare` with Eq returns exactly 0.0 or 1.0
/// and correctly identifies equality: Eq(a,b) == 1.0 iff a == b.
#[kani::unwind(1)]
#[kani::proof]
fn compare_eq_correct() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());

    let result = evaluate_constant_compare(CompareOpKind::Eq, a, b)
        .expect("Eq on finite values must not fail");
    if a == b {
        assert_eq!(
            result.to_bits(),
            1.0f32.to_bits(),
            "Eq(a,b) with a == b must be 1.0"
        );
    } else {
        assert_eq!(
            result.to_bits(),
            0.0f32.to_bits(),
            "Eq(a,b) with a != b must be 0.0"
        );
    }
}

/// Proves `evaluate_constant_compare` with Ne returns exactly 0.0 or 1.0
/// and correctly identifies inequality: Ne(a,b) == 1.0 iff a != b.
#[kani::unwind(1)]
#[kani::proof]
fn compare_ne_correct() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());

    let result = evaluate_constant_compare(CompareOpKind::Ne, a, b)
        .expect("Ne on finite values must not fail");
    if a != b {
        assert_eq!(
            result.to_bits(),
            1.0f32.to_bits(),
            "Ne(a,b) with a != b must be 1.0"
        );
    } else {
        assert_eq!(
            result.to_bits(),
            0.0f32.to_bits(),
            "Ne(a,b) with a == b must be 0.0"
        );
    }
}

/// Proves Gt anti-symmetry: Gt(a,b) == 1.0 implies Gt(b,a) == 0.0,
/// and vice versa. The only exception is a == b, where both are 0.0.
/// This is critical for consistent Select branching when operands are swapped.
#[kani::unwind(1)]
#[kani::proof]
fn compare_gt_antisymmetric() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());

    let gt_ab = evaluate_constant_compare(CompareOpKind::Gt, a, b)
        .expect("Gt on finite values must not fail");
    let gt_ba = evaluate_constant_compare(CompareOpKind::Gt, b, a)
        .expect("Gt on finite values must not fail");

    // At most one can be 1.0 (anti-symmetry: not both a > b and b > a)
    assert!(
        !(gt_ab.to_bits() == 1.0f32.to_bits() && gt_ba.to_bits() == 1.0f32.to_bits()),
        "Gt must be anti-symmetric: cannot have both Gt(a,b)==1 and Gt(b,a)==1"
    );
}

/// Proves reflexivity of Ge: Ge(x, x) == 1.0 for all finite x.
/// Required for Select branching correctness.
#[kani::unwind(1)]
#[kani::proof]
fn compare_ge_reflexive() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());

    let result = evaluate_constant_compare(CompareOpKind::Ge, x, x)
        .expect("Ge on finite values must not fail");
    assert_eq!(
        result.to_bits(),
        1.0f32.to_bits(),
        "Ge(x, x) must return 1.0 (reflexive)"
    );
}

/// Proves irreflexivity of Gt: Gt(x, x) == 0.0 for all finite x.
/// x is never strictly greater than itself.
#[kani::unwind(1)]
#[kani::proof]
fn compare_gt_irreflexive() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());

    let result = evaluate_constant_compare(CompareOpKind::Gt, x, x)
        .expect("Gt on finite values must not fail");
    assert_eq!(
        result.to_bits(),
        0.0f32.to_bits(),
        "Gt(x, x) must return 0.0 (irreflexive)"
    );
}

/// Proves reflexivity of Le: Le(x, x) == 1.0 for all finite x.
#[kani::unwind(1)]
#[kani::proof]
fn compare_le_reflexive() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());

    let result = evaluate_constant_compare(CompareOpKind::Le, x, x)
        .expect("Le on finite values must not fail");
    assert_eq!(
        result.to_bits(),
        1.0f32.to_bits(),
        "Le(x, x) must return 1.0 (reflexive)"
    );
}

/// Proves irreflexivity of Lt: Lt(x, x) == 0.0 for all finite x.
#[kani::unwind(1)]
#[kani::proof]
fn compare_lt_irreflexive() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());

    let result = evaluate_constant_compare(CompareOpKind::Lt, x, x)
        .expect("Lt on finite values must not fail");
    assert_eq!(
        result.to_bits(),
        0.0f32.to_bits(),
        "Lt(x, x) must return 0.0 (irreflexive)"
    );
}

/// Proves reflexivity of Eq: Eq(x, x) == 1.0 for all finite x.
/// Every finite value equals itself.
#[kani::unwind(1)]
#[kani::proof]
fn compare_eq_reflexive() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());

    let result = evaluate_constant_compare(CompareOpKind::Eq, x, x)
        .expect("Eq on finite values must not fail");
    assert_eq!(
        result.to_bits(),
        1.0f32.to_bits(),
        "Eq(x, x) must return 1.0 (reflexive)"
    );
}

/// Proves irreflexivity of Ne: Ne(x, x) == 0.0 for all finite x.
#[kani::unwind(1)]
#[kani::proof]
fn compare_ne_irreflexive() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());

    let result = evaluate_constant_compare(CompareOpKind::Ne, x, x)
        .expect("Ne on finite values must not fail");
    assert_eq!(
        result.to_bits(),
        0.0f32.to_bits(),
        "Ne(x, x) must return 0.0 (irreflexive)"
    );
}

// ---------------------------------------------------------------------------
// Cross-module integration: checked_constant + compare
// ---------------------------------------------------------------------------

/// Proves that the output of `evaluate_constant_compare` is always accepted
/// by `checked_constant` (since compare always returns 0.0 or 1.0, both
/// are finite). This validates that the compare → checked_constant pipeline
/// never produces a NonFiniteConstant error.
/// Uses `unwind(8)` — syn::ErrorMessage Drop unwinding (#608).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn compare_output_always_accepted_by_checked_constant() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());

    let op_idx: u8 = kani::any();
    kani::assume(op_idx < 6);
    let op = match op_idx {
        0 => CompareOpKind::Gt,
        1 => CompareOpKind::Ge,
        2 => CompareOpKind::Lt,
        3 => CompareOpKind::Le,
        4 => CompareOpKind::Eq,
        _ => CompareOpKind::Ne,
    };

    let cmp_result =
        evaluate_constant_compare(op, a, b).expect("compare on finite values must not fail");

    // The compare result must be accepted by checked_constant (finite).
    let cc_result = checked_constant(cmp_result, "compare_output_test");
    assert!(
        cc_result.is_ok(),
        "checked_constant must accept compare output (0.0 or 1.0)"
    );
}

/// Proves that `FiniteF32::new` accepts the output of
/// `evaluate_constant_compare`, verifying type-level compatibility.
#[kani::unwind(1)]
#[kani::proof]
fn compare_output_accepted_by_finite_f32() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());

    let op_idx: u8 = kani::any();
    kani::assume(op_idx < 6);
    let op = match op_idx {
        0 => CompareOpKind::Gt,
        1 => CompareOpKind::Ge,
        2 => CompareOpKind::Lt,
        3 => CompareOpKind::Le,
        4 => CompareOpKind::Eq,
        _ => CompareOpKind::Ne,
    };

    let cmp_result =
        evaluate_constant_compare(op, a, b).expect("compare on finite values must not fail");

    let f = FiniteF32::new(cmp_result);
    assert!(
        f.is_ok(),
        "FiniteF32::new must accept compare output (0.0 or 1.0)"
    );
}

// ---------------------------------------------------------------------------
// Numerical edge cases at zero
// ---------------------------------------------------------------------------

/// Proves that compare operations correctly handle the zero boundary.
/// Gt(0.0, 0.0) == 0.0, Ge(0.0, 0.0) == 1.0, distinguishing strict
/// vs non-strict at the critical zero boundary used by ReLU decomposition.
#[kani::unwind(1)]
#[kani::proof]
fn compare_zero_boundary_strict_vs_nonstrict() {
    // Gt(0, 0) must be 0.0 (strict: 0 is not > 0)
    let gt_result =
        evaluate_constant_compare(CompareOpKind::Gt, 0.0, 0.0).expect("Gt on zero must not fail");
    assert_eq!(
        gt_result.to_bits(),
        0.0f32.to_bits(),
        "Gt(0, 0) must be 0.0 (strict inequality)"
    );

    // Ge(0, 0) must be 1.0 (non-strict: 0 >= 0 is true)
    let ge_result =
        evaluate_constant_compare(CompareOpKind::Ge, 0.0, 0.0).expect("Ge on zero must not fail");
    assert_eq!(
        ge_result.to_bits(),
        1.0f32.to_bits(),
        "Ge(0, 0) must be 1.0 (non-strict inequality)"
    );

    // Lt(0, 0) must be 0.0
    let lt_result =
        evaluate_constant_compare(CompareOpKind::Lt, 0.0, 0.0).expect("Lt on zero must not fail");
    assert_eq!(
        lt_result.to_bits(),
        0.0f32.to_bits(),
        "Lt(0, 0) must be 0.0"
    );

    // Le(0, 0) must be 1.0
    let le_result =
        evaluate_constant_compare(CompareOpKind::Le, 0.0, 0.0).expect("Le on zero must not fail");
    assert_eq!(
        le_result.to_bits(),
        1.0f32.to_bits(),
        "Le(0, 0) must be 1.0"
    );

    // Eq(0, 0) must be 1.0
    let eq_result =
        evaluate_constant_compare(CompareOpKind::Eq, 0.0, 0.0).expect("Eq on zero must not fail");
    assert_eq!(
        eq_result.to_bits(),
        1.0f32.to_bits(),
        "Eq(0, 0) must be 1.0"
    );

    // Ne(0, 0) must be 0.0
    let ne_result =
        evaluate_constant_compare(CompareOpKind::Ne, 0.0, 0.0).expect("Ne on zero must not fail");
    assert_eq!(
        ne_result.to_bits(),
        0.0f32.to_bits(),
        "Ne(0, 0) must be 0.0"
    );
}

/// Proves checked_constant correctly distinguishes +0.0 and -0.0.
/// IEEE 754: -0.0 == +0.0 by comparison but differs in bit pattern.
/// `checked_constant` must accept both (both are finite) but preserve
/// the bit pattern, including sign of zero.
/// Uses `unwind(8)` — syn::ErrorMessage Drop unwinding (#608).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn checked_constant_preserves_zero_sign() {
    let pos_zero = checked_constant(0.0f32, "pos_zero");
    let neg_zero = checked_constant(-0.0f32, "neg_zero");

    let pv = pos_zero.expect("checked_constant must accept +0.0");
    let nv = neg_zero.expect("checked_constant must accept -0.0");

    match (pv, nv) {
        (NodeValue::Constant(pf), NodeValue::Constant(nf)) => {
            assert_eq!(
                pf.get().to_bits(),
                0.0f32.to_bits(),
                "+0.0 bit pattern must be preserved"
            );
            assert_eq!(
                nf.get().to_bits(),
                (-0.0f32).to_bits(),
                "-0.0 bit pattern must be preserved"
            );
        }
        _ => panic!("checked_constant must return Constant"),
    }
}
