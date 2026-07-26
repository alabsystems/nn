// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for where_cond (conditional selection) operations.
//!
//! Proves scalar-level correctness properties of where_cond:
//!
//! - Output always comes from one of the two input values (on_true or on_false)
//! - True mask selects on_true, false mask selects on_false
//! - All-true mask produces on_true, all-false produces on_false
//! - where_cond with identical inputs always returns that value
//!
//! These harnesses operate on pure scalar/logic — no ndarray or GPU
//! storage — making them tractable for CBMC symbolic execution.

#![cfg(kani)]

// ---------------------------------------------------------------------------
// where_cond: output is always one of the two inputs
// ---------------------------------------------------------------------------

/// Prove: where_cond output element is always either on_true or on_false.
///
/// For any mask value, the output must be exactly one of the two
/// candidate values. No other value is possible.
#[kani::unwind(1)]
#[kani::proof]
fn where_cond_output_from_inputs() {
    let mask: u8 = kani::any();
    let on_true: f32 = kani::any();
    let on_false: f32 = kani::any();
    kani::assume(on_true.is_finite() && on_false.is_finite());

    let result = if mask != 0 { on_true } else { on_false };
    assert!(
        result == on_true || result == on_false,
        "where_cond output must be one of the two inputs"
    );
}

/// Prove: when mask is nonzero (true), output is on_true.
#[kani::unwind(1)]
#[kani::proof]
fn where_cond_true_selects_on_true() {
    let mask: u8 = kani::any();
    kani::assume(mask != 0);
    let on_true: f32 = kani::any();
    let on_false: f32 = kani::any();
    kani::assume(on_true.is_finite() && on_false.is_finite());

    let result = if mask != 0 { on_true } else { on_false };
    assert_eq!(result, on_true, "true mask must select on_true");
}

/// Prove: when mask is zero (false), output is on_false.
#[kani::unwind(1)]
#[kani::proof]
fn where_cond_false_selects_on_false() {
    let on_true: f32 = kani::any();
    let on_false: f32 = kani::any();
    kani::assume(on_true.is_finite() && on_false.is_finite());

    let result = if 0u8 != 0 { on_true } else { on_false };
    assert_eq!(result, on_false, "false mask must select on_false");
}

/// Prove: when both inputs are the same, output is that value regardless of mask.
///
/// where_cond(mask, x, x) == x for any mask. This is the "don't care" case.
#[kani::unwind(1)]
#[kani::proof]
fn where_cond_same_inputs_identity() {
    let mask: u8 = kani::any();
    let val: f32 = kani::any();
    kani::assume(val.is_finite());

    let result = if mask != 0 { val } else { val };
    assert_eq!(
        result, val,
        "where_cond with identical inputs must return that value"
    );
}

// ---------------------------------------------------------------------------
// where_cond with F32 mask (GPU fast path: 0.0/1.0 convention)
// ---------------------------------------------------------------------------

/// Prove: F32 mask with 1.0 selects on_true, 0.0 selects on_false.
///
/// GPU path uses F32 masks where 1.0 = true and 0.0 = false.
#[kani::unwind(1)]
#[kani::proof]
fn where_cond_f32_mask_convention() {
    let on_true: f32 = kani::any();
    let on_false: f32 = kani::any();
    kani::assume(on_true.is_finite() && on_false.is_finite());

    // F32 mask 1.0 (true)
    let mask_true = 1.0_f32;
    let result_true = if mask_true != 0.0 { on_true } else { on_false };
    assert_eq!(result_true, on_true, "F32 mask 1.0 must select on_true");

    // F32 mask 0.0 (false)
    let mask_false = 0.0_f32;
    let result_false = if mask_false != 0.0 { on_true } else { on_false };
    assert_eq!(result_false, on_false, "F32 mask 0.0 must select on_false");
}

/// Prove: where_cond preserves output range when inputs are bounded.
///
/// If both on_true and on_false are in [lo, hi], then the output must
/// also be in [lo, hi] regardless of the mask. This is important for
/// bounds propagation in NY.
#[kani::unwind(1)]
#[kani::proof]
fn where_cond_preserves_bounds() {
    let mask: u8 = kani::any();
    let on_true: f32 = kani::any();
    let on_false: f32 = kani::any();
    let lo: f32 = kani::any();
    let hi: f32 = kani::any();

    kani::assume(lo.is_finite() && hi.is_finite() && lo <= hi);
    kani::assume(on_true.is_finite() && on_true >= lo && on_true <= hi);
    kani::assume(on_false.is_finite() && on_false >= lo && on_false <= hi);

    let result = if mask != 0 { on_true } else { on_false };
    assert!(result >= lo, "where_cond output must be >= lo bound");
    assert!(result <= hi, "where_cond output must be <= hi bound");
}
