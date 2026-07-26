// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![cfg(kani)]

use crate::IntervalBounds;
use ndarray::arr1;

fn finite_singleton_bounds(lower: f32, upper: f32) -> IntervalBounds {
    IntervalBounds::new(arr1(&[lower]).into_dyn(), arr1(&[upper]).into_dyn())
        .expect("finite ordered singleton bounds")
}

// Shape proofs (shape_numel, broadcast_none, broadcast_compatible) removed:
// Shape enum deleted in #24 Stage 2 migration. Broadcasting may be re-added
// as a Tensor method in a future iteration.

// Structural harnesses (shape/type validation)
mod structural;

// IBP arithmetic harnesses (add/mul/scale/shift) and their scalar equivalents
// removed in #2005 — arithmetic is provided by ny_tensor::BoundedTensor.

#[kani::unwind(1)]
#[kani::proof]
fn bounded_tensor_repair_invalid_inplace_repairs_infeasible_sentinel() {
    let mut bounds = finite_singleton_bounds(-1.0, 1.0);
    bounds.mark_infeasible_all();
    let repaired = bounds.repair_invalid_inplace();
    assert_eq!(repaired, 1);
    assert_eq!(bounds.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(bounds.upper()[[0]], f32::INFINITY);
}

#[kani::unwind(1)]
#[kani::proof]
fn bounded_tensor_repair_invalid_inplace_noops_for_valid_bounds() {
    let mut bounds = finite_singleton_bounds(-1.0, 1.0);
    let repaired = bounds.repair_invalid_inplace();
    assert_eq!(repaired, 0);
    assert_eq!(bounds.lower()[[0]], -1.0);
    assert_eq!(bounds.upper()[[0]], 1.0);
}

// ULP rounding and repair harnesses
mod ulp;
