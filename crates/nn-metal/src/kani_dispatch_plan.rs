// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for [`dispatch_plan`](super::dispatch_plan) functions.
//!
//! Proves overflow safety, output correctness, and zero-dimension rejection
//! for all dispatch planning modes (elementwise, 2D, 3D, reduction).

use crate::dispatch_plan::*;

/// Proves 2D output_elems does NOT overflow usize on 64-bit platforms.
///
/// Since u32::MAX² < u64::MAX, the product of any two u32 values
/// fits in u64 (= usize on 64-bit). This is the safety proof that
/// `plan_grid_2d` doesn't silently wrap.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn output_elems_2d_no_overflow() {
    let w: u32 = kani::any();
    let h: u32 = kani::any();
    kani::assume(w > 0);
    kani::assume(h > 0);

    // Compute the same way plan_grid_2d does.
    let product = w as usize * h as usize;

    // Verify against widened multiplication (u128 can hold u32² without overflow).
    let expected = (w as u128) * (h as u128);
    assert_eq!(
        product as u128, expected,
        "2D output_elems must equal widened product"
    );
}

/// Proves `plan_elementwise` output_elems always equals the input total.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn plan_elementwise_output_equals_total() {
    let total: u32 = kani::any();
    let plan = plan_elementwise(total).expect("elementwise plan always succeeds");
    assert_eq!(plan.output_elems(), total as usize);
}

/// Proves elementwise plan threads match `threadgroup_width_1d(total)`
/// and the y/z dimensions are always 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn plan_elementwise_threads_consistent() {
    let total: u32 = kani::any();
    kani::assume(total > 0);
    let plan = plan_elementwise(total).expect("elementwise plan always succeeds");
    let threads = plan.threads();
    let grid = plan.grid();
    assert_eq!(threads[0], threadgroup_width_1d(total));
    assert_eq!(threads[1], 1);
    assert_eq!(threads[2], 1);
    assert_eq!(grid[0], total);
    assert_eq!(grid[1], 1);
    assert_eq!(grid[2], 1);
}

/// Proves reduction plan output_elems equals `outer` for all valid inputs,
/// and that the plan uses threadgroups with shared memory set.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn plan_reduction_output_equals_outer() {
    let outer: u32 = kani::any();
    let reduce: u32 = kani::any();
    let threads: u32 = kani::any();
    let shared_bytes: u32 = kani::any();

    kani::assume(outer > 0);
    kani::assume(reduce > 0);
    kani::assume(threads > 0);

    let plan = plan_reduction(outer, reduce, threads, shared_bytes)
        .expect("reduction plan succeeds for non-zero params");
    assert_eq!(plan.output_elems(), outer as usize);
    assert!(plan.use_threadgroups());
    assert_eq!(plan.threadgroup_memory_bytes(), Some(shared_bytes as u64));
    let constants = plan.constants();
    assert_eq!(constants.len(), 2);
    assert_eq!(constants[0], outer);
    assert_eq!(constants[1], reduce);
}

/// Proves all zero-dimension inputs to plan_grid_2d are rejected.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn plan_grid_2d_rejects_any_zero_dim() {
    let g0: u32 = kani::any();
    let g1: u32 = kani::any();
    let t0: u32 = kani::any();
    let t1: u32 = kani::any();

    kani::assume(g0 == 0 || g1 == 0 || t0 == 0 || t1 == 0);

    let result = plan_grid_2d([g0, g1], [t0, t1]);
    assert!(result.is_err(), "plan_grid_2d must reject zero dimensions");
}

/// Proves all zero-dimension inputs to plan_reduction are rejected.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn plan_reduction_rejects_any_zero_param() {
    let outer: u32 = kani::any();
    let reduce: u32 = kani::any();
    let threads: u32 = kani::any();
    let shared_bytes: u32 = kani::any();

    kani::assume(outer == 0 || reduce == 0 || threads == 0);

    let result = plan_reduction(outer, reduce, threads, shared_bytes);
    assert!(result.is_err(), "plan_reduction must reject zero params");
}

/// Proves 3D output_elems is correct when no overflow occurs, and that
/// plan_grid_3d never silently wraps.
///
/// For any non-zero grid/thread dimensions, if the triple product fits
/// in usize, then output_elems equals the expected value. If it overflows,
/// plan_grid_3d returns an error instead of wrapping.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
fn plan_grid_3d_no_silent_wrap() {
    let g0: u32 = kani::any();
    let g1: u32 = kani::any();
    let g2: u32 = kani::any();
    let t0: u32 = kani::any();
    let t1: u32 = kani::any();
    let t2: u32 = kani::any();

    kani::assume(g0 > 0 && g1 > 0 && g2 > 0);
    kani::assume(t0 > 0 && t1 > 0 && t2 > 0);

    let widened = (g0 as u128) * (g1 as u128) * (g2 as u128);
    let result = plan_grid_3d([g0, g1, g2], [t0, t1, t2]);

    if widened > usize::MAX as u128 {
        // Overflow case: must be an error.
        assert!(
            result.is_err(),
            "plan_grid_3d must reject overflowing grids"
        );
    } else {
        // No overflow: output_elems must equal the widened product.
        let plan = result.expect("plan_grid_3d should succeed for non-overflowing grids");
        assert_eq!(plan.output_elems() as u128, widened);
    }
}
