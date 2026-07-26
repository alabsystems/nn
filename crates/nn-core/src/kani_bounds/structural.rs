// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Structural Kani proof harnesses for BoundedTensor: shape matching, type
//! validation, dimension utilities, and error-path coverage.
//!
//! Extracted from `kani_bounds.rs` to stay under the 500-line file limit (#542).
//! IBP arithmetic harnesses removed in #2005.

use crate::tensor::checked_dim_product;
use crate::{IntervalBounds, Tensor, TensorError};
use ndarray::{ArrayD, IxDyn};

#[kani::unwind(1)]
#[kani::proof]
fn tensor_numel_matches_two_dim_product() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();

    kani::assume((1..=16).contains(&d0));
    kani::assume((1..=16).contains(&d1));

    let tensor: Tensor<2> =
        Tensor::zeros([usize::from(d0), usize::from(d1)]).expect("CPU allocation");
    assert_eq!(tensor.numel(), usize::from(d0) * usize::from(d1));
}

#[kani::unwind(1)]
#[kani::proof]
fn tensor_with_bounds_accepts_matching_shape() {
    let tensor: Tensor<2> = Tensor::zeros([2, 2]).expect("CPU allocation");
    let concrete = ArrayD::from_elem(IxDyn(&[2, 2]), 0.5f32);
    let bounds = IntervalBounds::concrete(concrete).expect("valid concrete bounds");

    let tensor_with_bounds = tensor
        .with_bounds(bounds)
        .expect("matching bounds shape should be accepted");
    let attached_bounds = tensor_with_bounds
        .bounds()
        .expect("accepted bounds must be attached");
    assert_eq!(attached_bounds.shape(), &[2, 2]);
    assert_eq!(attached_bounds.lower()[[0, 0]], 0.5);
    assert_eq!(attached_bounds.upper()[[0, 0]], 0.5);
}

// Shape-mismatch harnesses for tensor_with_bounds, add, mul deleted in #767:
// CBMC cannot efficiently model ndarray IxDyn heap allocation even for constant
// inputs (~121s timeout per harness). These error paths are covered by runtime
// tests in bounds_tests.rs and bounds_tests_scalar_drift.rs.

// from_epsilon ndarray harness deleted in #767:
// Superseded by scalar_from_epsilon_always_produces_lower_le_upper in
// kani_bounds/scalar_scale_shift.rs which proves the same property without
// ndarray overhead.

/// Proves `checked_dim_product` matches unchecked multiplication for small dims.
///
/// For any two u16 dimensions (cast to usize), the product always fits in
/// usize on 64-bit, so `checked_dim_product` must return `Ok(d0 * d1)`.
#[kani::unwind(1)]
#[kani::proof]
fn checked_dim_product_matches_unchecked_for_small_dims() {
    let d0: u16 = kani::any();
    let d1: u16 = kani::any();
    kani::assume(d0 >= 1);
    kani::assume(d1 >= 1);

    let dims = [d0 as usize, d1 as usize];
    let result = checked_dim_product(&dims);

    // u16 × u16 max = 65535² = 4_294_836_225 < usize::MAX on 64-bit
    let product = result.expect("u16 × u16 must not overflow usize");
    assert_eq!(product, (d0 as usize) * (d1 as usize));
}

/// Proves `checked_dim_product` returns `Err(DimensionOverflow)` when overflow.
#[kani::unwind(1)]
#[kani::proof]
fn checked_dim_product_detects_overflow() {
    let d: usize = kani::any();
    kani::assume(d >= 2);

    let dims = [usize::MAX, d];
    let result = checked_dim_product(&dims);

    assert!(result.is_err(), "usize::MAX × d (d >= 2) must overflow");
    assert!(
        matches!(result, Err(TensorError::DimensionOverflow { .. })),
        "must return DimensionOverflow variant"
    );
}
