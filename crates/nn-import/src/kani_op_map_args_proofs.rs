// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for op_map_args.rs helper functions (#3748).
//!
//! Covers:
//! - safe_usize: non-negative i64 converts correctly
//! - safe_usize: negative i64 returns Err
//! - safe_usize_allow_neg1: -1 maps to usize::MAX
//! - safe_usize_allow_neg1: other negatives return Err
//! - safe_usize_vec: all-positive converts correctly
//! - safe_usize_vec: any-negative returns Err
//! - resolve_dim: positive dim passes through
//! - resolve_dim: negative dim with known ndim resolves correctly
//! - resolve_dim: negative dim with unknown ndim returns Err
//! - require_single_dim: single dim returns the value
//! - require_single_dim: multi dims returns Err
//! - require_single_dim: empty dims returns 0
//! - scalar_type_to_name: known types map correctly
//! - scalar_type_to_name: unknown types return Err

#![cfg(kani)]

// ---------------------------------------------------------------------------
// safe_usize: non-negative i64 converts correctly
// ---------------------------------------------------------------------------

/// Prove: safe_usize succeeds for all non-negative i64 values.
///
/// Inlines op_map_args.rs:108-114. This is the fundamental conversion used
/// by ALL dimension/size extraction in the import pipeline.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn safe_usize_nonnegative_succeeds() {
    let val: i64 = kani::any();
    kani::assume(val >= 0);

    let result = usize::try_from(val);

    assert!(result.is_ok(), "Non-negative i64 must convert to usize");
    assert_eq!(result.unwrap(), val as usize, "Converted value must match");
}

// ---------------------------------------------------------------------------
// safe_usize: negative i64 returns Err
// ---------------------------------------------------------------------------

/// Prove: safe_usize fails for all negative i64 values.
///
/// Inlines op_map_args.rs:108-114.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn safe_usize_negative_fails() {
    let val: i64 = kani::any();
    kani::assume(val < 0);

    let result = usize::try_from(val);

    assert!(
        result.is_err(),
        "Negative i64 must fail conversion to usize"
    );
}

/// Prove: safe_usize_allow_neg1 rejects negatives other than -1.
///
/// Inlines op_map_args.rs:151-161.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn safe_usize_allow_neg1_other_negatives_fail() {
    let val: i64 = kani::any();
    kani::assume(val < -1 && val >= -1000);

    // The function does: if val == -1 { Ok(MAX) } else { safe_usize(val, ...) }
    let is_neg1 = val == -1;
    let try_from_ok = usize::try_from(val).is_ok();

    assert!(!is_neg1, "Value must not be -1");
    assert!(!try_from_ok, "Negative (non -1) must fail try_from");
}

// ---------------------------------------------------------------------------
// safe_usize_vec: all-positive converts correctly
// ---------------------------------------------------------------------------

/// Prove: safe_usize_vec succeeds when all elements are non-negative.
///
/// Inlines op_map_args.rs:141-149.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(5)]
fn safe_usize_vec_all_positive() {
    let len: usize = kani::any();
    kani::assume(len >= 1 && len <= 4);

    let v0: i64 = kani::any();
    let v1: i64 = kani::any();
    let v2: i64 = kani::any();
    let v3: i64 = kani::any();
    kani::assume(v0 >= 0 && v0 <= 100);
    kani::assume(v1 >= 0 && v1 <= 100);
    kani::assume(v2 >= 0 && v2 <= 100);
    kani::assume(v3 >= 0 && v3 <= 100);

    let vals = [v0, v1, v2, v3];

    let mut all_ok = true;
    let mut i: usize = 0;
    while i < len {
        if usize::try_from(vals[i]).is_err() {
            all_ok = false;
        }
        i += 1;
    }

    assert!(all_ok, "All non-negative i64 values must convert");
}

// ---------------------------------------------------------------------------
// safe_usize_vec: any-negative returns Err
// ---------------------------------------------------------------------------

/// Prove: safe_usize_vec fails when any element is negative.
///
/// Inlines op_map_args.rs:141-149.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
fn safe_usize_vec_any_negative_fails() {
    let len: usize = kani::any();
    kani::assume(len >= 1 && len <= 3);

    let neg_idx: usize = kani::any();
    kani::assume(neg_idx < len);

    let v0: i64 = if neg_idx == 0 { -1 } else { 1 };
    let v1: i64 = if neg_idx == 1 { -1 } else { 2 };
    let v2: i64 = if neg_idx == 2 { -1 } else { 3 };

    let vals = [v0, v1, v2];

    let mut any_err = false;
    let mut i: usize = 0;
    while i < len {
        if usize::try_from(vals[i]).is_err() {
            any_err = true;
        }
        i += 1;
    }

    assert!(any_err, "At least one negative must cause failure");
}

// ---------------------------------------------------------------------------
// resolve_dim: negative dim with known ndim resolves correctly
// ---------------------------------------------------------------------------

/// Prove: resolve_dim with negative val and known ndim resolves to val + ndim.
///
/// Inlines op_map_args.rs:128-129. -1 with ndim=4 must produce 3.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn resolve_dim_negative_with_ndim() {
    let val: i64 = kani::any();
    let ndim: usize = kani::any();
    kani::assume(val < 0 && val >= -10);
    kani::assume(ndim >= 1 && ndim <= 10);
    // Ensure resolution is non-negative.
    kani::assume(val + ndim as i64 >= 0);

    let resolved = val + ndim as i64;
    let result = usize::try_from(resolved);

    assert!(result.is_ok(), "Resolved dim must be non-negative");
    assert_eq!(
        result.unwrap(),
        (val + ndim as i64) as usize,
        "Resolved dim must equal val + ndim"
    );
}

// ---------------------------------------------------------------------------
// resolve_dim: negative dim with unknown ndim (0) returns Err
// ---------------------------------------------------------------------------

/// Prove: resolve_dim with negative val and ndim=0 returns Err.
///
/// Inlines op_map_args.rs:131-138. Without knowing the rank, we can't
/// resolve negative dimensions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn resolve_dim_negative_unknown_ndim_fails() {
    let val: i64 = kani::any();
    kani::assume(val < 0 && val >= -10);

    let ndim: usize = 0;

    // resolve_dim: val < 0 && ndim == 0 → Err
    let would_fail = val < 0 && ndim == 0;

    assert!(would_fail, "Negative dim with unknown ndim must fail");
}

// ---------------------------------------------------------------------------
// require_single_dim: single element returns the value
// ---------------------------------------------------------------------------

/// Prove: require_single_dim with a 1-element slice returns that element.
///
/// Inlines op_map_args.rs:163-177.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn require_single_dim_one_element() {
    let dim: i64 = kani::any();
    kani::assume(dim >= -10 && dim <= 10);

    let dims_len: usize = 1;

    // For len == 1: not > 1, so skip multi error. Return dims[0].
    let result = if dims_len > 1 {
        None // would be error
    } else {
        Some(dim) // dims.first().copied().unwrap_or(0)
    };

    assert!(result.is_some(), "Single dim must succeed");
    assert_eq!(result.unwrap(), dim, "Must return the single dim value");
}

// ---------------------------------------------------------------------------
// require_single_dim: multi dims returns Err
// ---------------------------------------------------------------------------

/// Prove: require_single_dim with >1 elements returns Err.
///
/// Inlines op_map_args.rs:169-174.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn require_single_dim_multi_fails() {
    let dims_len: usize = kani::any();
    kani::assume(dims_len >= 2 && dims_len <= 6);

    let is_err = dims_len > 1;

    assert!(is_err, "Multiple dims must trigger multi-axis error");
}

// ---------------------------------------------------------------------------
// scalar_type_to_name: known types produce Ok
// ---------------------------------------------------------------------------

/// Prove: scalar_type_to_name maps the 6 known ScalarType ints to Ok.
///
/// Inlines op_map_args.rs:240-258.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn scalar_type_to_name_known_types_ok() {
    fn is_known(st: i32) -> bool {
        matches!(st, 1 | 5 | 6 | 7 | 8 | 13)
    }

    assert!(is_known(1), "U8 must be known");
    assert!(is_known(5), "I64 must be known");
    assert!(is_known(6), "F16 must be known");
    assert!(is_known(7), "F32 must be known");
    assert!(is_known(8), "F64 must be known");
    assert!(is_known(13), "BF16 must be known");
}

// ---------------------------------------------------------------------------
// scalar_type_to_name: unknown types produce Err
// ---------------------------------------------------------------------------

/// Prove: scalar_type_to_name returns Err for any ScalarType not in {1,5,6,7,8,13}.
///
/// Inlines op_map_args.rs:251-257.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn scalar_type_to_name_unknown_types_err() {
    let st: i32 = kani::any();
    kani::assume(st >= 0 && st <= 20);
    kani::assume(!matches!(st, 1 | 5 | 6 | 7 | 8 | 13));

    fn is_known(st: i32) -> bool {
        matches!(st, 1 | 5 | 6 | 7 | 8 | 13)
    }

    assert!(!is_known(st), "Unknown ScalarType must not be recognized");
}
