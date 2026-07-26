// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for DynTensor operations.
//!
//! Proves key arithmetic invariants in tensor creation, shape manipulation,
//! and embedding index computation. These complement the IntervalBounds
//! harnesses (kani_bounds*.rs) and conv harnesses (kani_conv.rs) by covering
//! the DynTensor imperative API used by dvoice model forward paths.
//!
//! All harnesses use scalar arithmetic (inlined from source) rather than
//! calling DynTensor methods directly, since Kani cannot model ndarray or
//! GPU storage. The properties proved are the arithmetic invariants that
//! the runtime code depends on.

#![cfg(kani)]

// -- Kani transcendental stubs (CBMC #239, #329, #708) --

fn ceil_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite());
    r
}

// ---------------------------------------------------------------------------
// AC1: arange — prove ceil()-as-usize correctness for bounded ranges
// ---------------------------------------------------------------------------

/// Prove: for integer-valued start/end in [-255, 255], arange element count
/// equals the mathematical expected count (end - start) when end > start.
///
/// Inlines: `n = (end - start).ceil() as usize` from dyn_tensor.rs:231
///
/// For integer-valued f64 inputs, (end - start).ceil() == (end - start),
/// so n should exactly equal the integer difference.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::ceil, ceil_f32_stub)]
fn arange_count_correct_integer_range() {
    let start_i: i16 = kani::any();
    let end_i: i16 = kani::any();

    // Restrict to [-255, 255] to keep symbolic exploration bounded
    kani::assume(start_i >= -255 && start_i <= 255);
    kani::assume(end_i >= -255 && end_i <= 255);
    kani::assume(end_i > start_i);

    let start = start_i as f64;
    let end = end_i as f64;

    // This matches dyn_tensor.rs:231
    let n = (end - start).ceil() as usize;

    let expected = (end_i - start_i) as usize;
    assert_eq!(n, expected, "arange count must match integer difference");
}

/// Prove: arange with end <= start produces n == 0 (empty range guard).
///
/// Inlines the guard at dyn_tensor.rs:228
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn arange_empty_when_end_leq_start() {
    let start_i: i16 = kani::any();
    let end_i: i16 = kani::any();
    kani::assume(start_i >= -255 && start_i <= 255);
    kani::assume(end_i >= -255 && end_i <= 255);
    kani::assume(end_i <= start_i);

    let start = start_i as f64;
    let end = end_i as f64;

    // The guard at dyn_tensor.rs:228 returns empty when end <= start.
    // Verify: the f64 comparison agrees with the integer comparison.
    assert!(
        end <= start,
        "f64 ordering must match i16 ordering for integers"
    );
}

/// Prove: arange element values are monotonically non-decreasing in f32.
///
/// For small integer ranges, each successive element (start + i) as f32
/// must be >= the previous element.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arange_monotone_small_range() {
    let start_i: i8 = kani::any();
    let len: u8 = kani::any();

    kani::assume(start_i >= -100 && start_i <= 100);
    kani::assume(len >= 2 && len <= 20);

    let start = start_i as f64;
    let i: u8 = kani::any();
    kani::assume(i < len - 1);

    // Two consecutive elements from arange (dyn_tensor.rs:232)
    let val_i = (start + i as f64) as f32;
    let val_next = (start + (i + 1) as f64) as f32;

    assert!(
        val_next >= val_i,
        "arange values must be monotonically non-decreasing"
    );
}

/// Prove: reshape validation inlined from dyn_tensor_shape.rs:90-97.
/// Exercises checked_dim_product (tensor.rs:101-108) for both shapes.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn reshape_validation_numel_check() {
    let a: u16 = kani::any();
    let b: u16 = kani::any();
    let c: u16 = kani::any();
    let d: u16 = kani::any();
    kani::assume(a >= 1 && a <= 64);
    kani::assume(b >= 1 && b <= 64);
    kani::assume(c >= 1 && c <= 64);
    kani::assume(d >= 1 && d <= 64);
    let orig_dims = [a as usize, b as usize];
    let new_dims = [c as usize, d as usize];
    let orig_checked = orig_dims
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim));
    let new_checked = new_dims
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim));
    if let (Some(orig_numel), Some(new_numel)) = (orig_checked, new_checked) {
        if orig_numel == new_numel {
            let orig_unchecked: usize = orig_dims.iter().copied().product();
            let new_unchecked: usize = new_dims.iter().copied().product();
            assert_eq!(orig_numel, orig_unchecked, "checked == unchecked (orig)");
            assert_eq!(new_numel, new_unchecked, "checked == unchecked (new)");
            assert_eq!(orig_unchecked, new_unchecked, "reshape numel preserved");
        }
    }
}

/// Prove: squeeze algorithm from dyn_tensor_shape.rs:190-205.
/// Validates dim < rank, dims[dim] == 1, then verifies numel preservation.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn squeeze_shape_validation() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    let d2: u16 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 32);
    kani::assume(d1 >= 1 && d1 <= 32);
    kani::assume(d2 >= 1 && d2 <= 32);
    let dims = [d0 as usize, d1 as usize, d2 as usize];
    let rank = dims.len();
    let squeeze_dim: usize = kani::any();
    kani::assume(squeeze_dim < rank);
    if dims[squeeze_dim] == 1 {
        let new_dims: [usize; 2] = match squeeze_dim {
            0 => [dims[1], dims[2]],
            1 => [dims[0], dims[2]],
            _ => [dims[0], dims[1]],
        };
        let orig_numel = dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
        let new_numel = new_dims
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d));
        if let (Some(on), Some(nn)) = (orig_numel, new_numel) {
            assert_eq!(on, nn, "squeeze must preserve numel");
        }
        assert_eq!(new_dims.len(), rank - 1, "squeeze must reduce rank by 1");
    }
}

/// Prove: unsqueeze algorithm from dyn_tensor_shape.rs:177-186.
/// Validates dim <= rank, verifies numel preservation and dim ordering.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn unsqueeze_shape_validation() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 32);
    kani::assume(d1 >= 1 && d1 <= 32);
    let dims = [d0 as usize, d1 as usize];
    let rank = dims.len();
    let insert_dim: usize = kani::any();
    kani::assume(insert_dim <= rank);
    let new_dims: [usize; 3] = match insert_dim {
        0 => [1, dims[0], dims[1]],
        1 => [dims[0], 1, dims[1]],
        _ => [dims[0], dims[1], 1],
    };
    assert_eq!(new_dims[insert_dim], 1, "inserted dim must be 1");
    assert_eq!(
        new_dims.len(),
        rank + 1,
        "unsqueeze must increase rank by 1"
    );
    let orig_numel = dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
    let new_numel = new_dims
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d));
    if let (Some(on), Some(nn)) = (orig_numel, new_numel) {
        assert_eq!(on, nn, "unsqueeze must preserve numel");
    }
    let mut orig_idx = 0;
    for (i, &nd) in new_dims.iter().enumerate() {
        if i == insert_dim {
            continue;
        }
        assert_eq!(nd, dims[orig_idx], "original dims must be preserved");
        orig_idx += 1;
    }
    assert_eq!(orig_idx, rank, "all original dims must be accounted for");
}
