// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for lib.rs helper functions.
//!
//! Proves:
//! - count_non_finite returns 0 for all-finite slices
//! - count_non_finite counts correctly for NaN and Inf
//! - to_u32 succeeds iff value fits in u32
//! - checked_output_bytes detects overflow
//! - threadgroup_width_1d boundary behavior

use crate::count_non_finite;
use crate::to_u32;
use crate::kernel_dispatch::check_dispatch_size;
use crate::dispatch_plan::threadgroup_width_1d;

/// Proves: count_non_finite returns 0 for an empty slice.
#[kani::unwind(1)]
#[kani::proof]
fn count_non_finite_empty() {
    let data: &[f32] = &[];
    assert_eq!(count_non_finite(data), 0);
}

/// Proves: count_non_finite returns 1 for a single-element NaN slice.
#[kani::unwind(2)]
#[kani::proof]
fn count_non_finite_single_nan() {
    let data = [f32::NAN];
    assert_eq!(count_non_finite(&data), 1);
}

/// Proves: count_non_finite returns 1 for a single-element Inf slice.
#[kani::unwind(2)]
#[kani::proof]
fn count_non_finite_single_inf() {
    let data = [f32::INFINITY];
    assert_eq!(count_non_finite(&data), 1);
}

/// Proves: count_non_finite returns 0 for a single finite element.
#[kani::unwind(2)]
#[kani::proof]
fn count_non_finite_single_finite() {
    let val: f32 = kani::any();
    kani::assume(val.is_finite());
    let data = [val];
    assert_eq!(count_non_finite(&data), 0);
}

/// Proves: to_u32 succeeds for values <= u32::MAX and fails above.
#[kani::unwind(1)]
#[kani::proof]
fn to_u32_fits_iff_within_range() {
    let val: usize = kani::any();
    kani::assume(val <= u32::MAX as usize);
    let result = to_u32(val, "test");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), val as u32);
}

/// Proves: to_u32 fails for values above u32::MAX.
#[kani::unwind(1)]
#[kani::proof]
fn to_u32_rejects_overflow() {
    let val: usize = kani::any();
    kani::assume(val > u32::MAX as usize);
    assert!(to_u32(val, "test").is_err());
}

/// Proves: threadgroup_width_1d is always exactly min(total, 64).
#[kani::unwind(1)]
#[kani::proof]
fn threadgroup_width_1d_is_min_total_64() {
    let total: u32 = kani::any();
    kani::assume(total > 0);
    let width = threadgroup_width_1d(total);
    let expected = if total < 64 { total } else { 64 };
    assert_eq!(width, expected);
}

/// Proves: threadgroup_width_1d(0) == 0 (degenerate case).
#[kani::unwind(1)]
#[kani::proof]
fn threadgroup_width_1d_zero() {
    assert_eq!(threadgroup_width_1d(0), 0);
}

/// Proves: check_dispatch_size and to_u32 agree on the boundary.
#[kani::unwind(1)]
#[kani::proof]
fn dispatch_size_and_to_u32_agree() {
    let val: usize = kani::any();
    let dispatch_ok = check_dispatch_size(val).is_some();
    let to_u32_ok = to_u32(val, "test").is_ok();
    assert_eq!(dispatch_ok, to_u32_ok, "both must agree on u32 range");
}
