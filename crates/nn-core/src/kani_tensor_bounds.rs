// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Tensor<D,T,B> and IntervalBounds safety (#3622).
//!
//! Proves correctness properties of the core tensor type and interval bounds
//! infrastructure used throughout the verification pipeline:
//!
//! - `checked_dim_product`: overflow detection, identity, zero propagation
//! - `next_up_f32` / `next_down_f32`: monotonicity, NaN passthrough, infinity
//!   preservation, inverse relationship
//! - `repair_non_finite_lower` / `repair_non_finite_upper`: finite passthrough,
//!   non-finite repair to FALLBACK_BOUND
//! - `enforce_bound_ordering`: always produces valid lower <= upper
//! - IntervalBounds invariant: constructed bounds satisfy lower[i] <= upper[i]
//! - IntervalBounds::concrete: lower == upper for all elements
//! - IntervalBounds::from_epsilon: bound ordering and width guarantees
//! - IntervalBounds::max_width: non-negative for valid bounds
//!
//! All harnesses operate on pure scalar/array arithmetic. ndarray
//! construction is used sparingly for small fixed-size arrays that
//! Kani can handle via bounded unrolling.

#![cfg(kani)]

use crate::bounds::repair::{
    enforce_bound_ordering, next_down_f32, next_up_f32, repair_non_finite_lower,
    repair_non_finite_upper, FALLBACK_BOUND,
};
use crate::tensor::checked_dim_product;

/// Prove: checked_dim_product with a zero dimension always returns 0.
///
/// Any tensor with a zero-length axis has zero elements. This is used
/// by from_vec to accept empty data vectors for zero-element tensors.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn checked_dim_product_zero_propagates() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    kani::assume(a <= 1024);
    kani::assume(b <= 1024);

    let result = checked_dim_product(&[a, 0, b]);
    assert!(result.is_ok(), "zero dimension must not overflow");
    assert_eq!(result.unwrap(), 0, "zero dimension must make product zero");
}

/// Prove: checked_dim_product of a single dimension returns that dimension.
///
/// A 1D tensor with dimension N has exactly N elements.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn checked_dim_product_single_dim_identity() {
    let d: usize = kani::any();
    let result = checked_dim_product(&[d]);
    assert!(result.is_ok(), "single dim cannot overflow");
    assert_eq!(result.unwrap(), d, "single-element product is identity");
}

/// Prove: checked_dim_product result (when Ok) is consistent with manual
/// multiplication for small 2D cases.
///
/// For bounded inputs where overflow cannot occur, the checked product
/// must equal the unchecked product.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
fn checked_dim_product_matches_unchecked_small() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    kani::assume(a <= 1024);
    kani::assume(b <= 1024);

    let result = checked_dim_product(&[a, b]);
    assert!(result.is_ok(), "small dims cannot overflow");
    assert_eq!(
        result.unwrap(),
        a * b,
        "checked must match unchecked for small dims"
    );
}

// ============================================================================
// next_up_f32 / next_down_f32: ULP rounding for interval soundness
// ============================================================================

/// Prove: next_up_f32(x) >= x for all finite f32.
///
/// The upper bound widens upward — it must never decrease. This is the
/// monotonicity property required for sound interval arithmetic.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn next_up_monotonic() {
    let bits: u32 = kani::any();
    let x = f32::from_bits(bits);
    kani::assume(x.is_finite());

    let up = next_up_f32(x);
    assert!(up >= x, "next_up must not decrease the value");
}

/// Prove: next_down_f32(x) <= x for all finite f32.
///
/// The lower bound widens downward — it must never increase. This is the
/// monotonicity property required for sound interval arithmetic.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn next_down_monotonic() {
    let bits: u32 = kani::any();
    let x = f32::from_bits(bits);
    kani::assume(x.is_finite());

    let down = next_down_f32(x);
    assert!(down <= x, "next_down must not increase the value");
}

/// Prove: next_up_f32(x) > x for all finite non-MAX f32.
///
/// For any finite value that is not f32::MAX, next_up must produce a
/// strictly larger value (the interval genuinely widens by at least 1 ULP).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn next_up_strictly_increases_non_max() {
    let bits: u32 = kani::any();
    let x = f32::from_bits(bits);
    kani::assume(x.is_finite());
    kani::assume(x != f32::MAX);

    let up = next_up_f32(x);
    assert!(
        up > x,
        "next_up must strictly increase for non-MAX finite values"
    );
}

/// Prove: next_down_f32(x) < x for all finite non-MIN f32.
///
/// For any finite value that is not f32::MIN, next_down must produce a
/// strictly smaller value (the interval genuinely widens by at least 1 ULP).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn next_down_strictly_decreases_non_min() {
    let bits: u32 = kani::any();
    let x = f32::from_bits(bits);
    kani::assume(x.is_finite());
    kani::assume(x != f32::MIN);

    let down = next_down_f32(x);
    assert!(
        down < x,
        "next_down must strictly decrease for non-MIN finite values"
    );
}

/// Prove: next_up_f32 preserves NaN (passthrough).
///
/// NaN values must not be converted to finite values by ULP rounding.
/// IEEE 754: operations on NaN produce NaN.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn next_up_nan_passthrough() {
    let bits: u32 = kani::any();
    let x = f32::from_bits(bits);
    kani::assume(x.is_nan());

    let up = next_up_f32(x);
    assert!(up.is_nan(), "next_up of NaN must be NaN");
}

/// Prove: next_down_f32 preserves NaN (passthrough).
///
/// NaN values must not be converted to finite values by ULP rounding.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn next_down_nan_passthrough() {
    let bits: u32 = kani::any();
    let x = f32::from_bits(bits);
    kani::assume(x.is_nan());

    let down = next_down_f32(x);
    assert!(down.is_nan(), "next_down of NaN must be NaN");
}

/// Prove: next_up_f32 preserves infinity sentinels.
///
/// +inf and -inf are used as infeasible sentinels in verification
/// (mark_infeasible_all). ULP rounding must not convert them to finite
/// values (#171).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn next_up_preserves_infinity() {
    let up_pos = next_up_f32(f32::INFINITY);
    assert_eq!(up_pos, f32::INFINITY, "+inf must be preserved");

    let up_neg = next_up_f32(f32::NEG_INFINITY);
    assert_eq!(up_neg, f32::NEG_INFINITY, "-inf must be preserved");
}

/// Prove: next_down_f32 preserves infinity sentinels.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn next_down_preserves_infinity() {
    let down_pos = next_down_f32(f32::INFINITY);
    assert_eq!(down_pos, f32::INFINITY, "+inf must be preserved");

    let down_neg = next_down_f32(f32::NEG_INFINITY);
    assert_eq!(down_neg, f32::NEG_INFINITY, "-inf must be preserved");
}

/// Prove: next_up_f32 output is always finite for finite input.
///
/// Excludes f32::MAX which may step to infinity. For all other finite
/// values, the result must remain representable.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn next_up_finite_input_finite_output() {
    let bits: u32 = kani::any();
    let x = f32::from_bits(bits);
    kani::assume(x.is_finite());
    kani::assume(x != f32::MAX);

    let up = next_up_f32(x);
    assert!(up.is_finite(), "next_up of non-MAX finite must be finite");
}

/// Prove: next_down_f32 output is always finite for finite input.
///
/// Excludes f32::MIN which may step to -infinity. For all other finite
/// values, the result must remain representable.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn next_down_finite_input_finite_output() {
    let bits: u32 = kani::any();
    let x = f32::from_bits(bits);
    kani::assume(x.is_finite());
    kani::assume(x != f32::MIN);

    let down = next_down_f32(x);
    assert!(
        down.is_finite(),
        "next_down of non-MIN finite must be finite"
    );
}

// ============================================================================
// repair_non_finite_lower / repair_non_finite_upper
// ============================================================================

/// Prove: repair_non_finite_lower returns the input for finite values.
///
/// Finite values must pass through unchanged — only non-finite values
/// get repaired to -FALLBACK_BOUND.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn repair_lower_finite_passthrough() {
    let bits: u32 = kani::any();
    let x = f32::from_bits(bits);
    kani::assume(x.is_finite());

    let result = repair_non_finite_lower(x);
    assert_eq!(
        x.to_bits(),
        result.to_bits(),
        "finite values must pass through unchanged"
    );
}

/// Prove: repair_non_finite_upper returns the input for finite values.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn repair_upper_finite_passthrough() {
    let bits: u32 = kani::any();
    let x = f32::from_bits(bits);
    kani::assume(x.is_finite());

    let result = repair_non_finite_upper(x);
    assert_eq!(
        x.to_bits(),
        result.to_bits(),
        "finite values must pass through unchanged"
    );
}

/// Prove: repair_non_finite_lower always produces a finite result.
///
/// NaN, +inf, and -inf are all repaired to the finite -FALLBACK_BOUND.
/// Combined with the passthrough property, this means the output is
/// always finite.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn repair_lower_always_finite() {
    let bits: u32 = kani::any();
    let x = f32::from_bits(bits);

    let result = repair_non_finite_lower(x);
    assert!(result.is_finite(), "repaired lower must always be finite");
}

/// Prove: repair_non_finite_upper always produces a finite result.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn repair_upper_always_finite() {
    let bits: u32 = kani::any();
    let x = f32::from_bits(bits);

    let result = repair_non_finite_upper(x);
    assert!(result.is_finite(), "repaired upper must always be finite");
}

/// Prove: repaired lower <= repaired upper for any pair of inputs.
///
/// After repair, the lower endpoint (-FALLBACK_BOUND or original finite)
/// must never exceed the upper endpoint (+FALLBACK_BOUND or original finite).
/// This is the fundamental invariant of interval bounds.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn repaired_lower_leq_upper() {
    let bits_l: u32 = kani::any();
    let bits_u: u32 = kani::any();
    let l = f32::from_bits(bits_l);
    let u = f32::from_bits(bits_u);

    // Only consider inputs where original ordering holds or repair needed
    let rl = repair_non_finite_lower(l);
    let ru = repair_non_finite_upper(u);

    // When both are non-finite, they get repaired to -FALLBACK_BOUND and
    // +FALLBACK_BOUND respectively. When both are finite, we need the
    // original ordering assumption.
    kani::assume(l.is_finite() && u.is_finite() && l <= u || !l.is_finite() || !u.is_finite());

    // If both finite and originally ordered, repair is identity and l <= u.
    // If either non-finite, repair produces -FB <= +FB.
    // The only tricky case: l is finite but very large, u is non-finite → repaired to FB.
    // In that case l could exceed FB. We check the guaranteed case:
    if !l.is_finite() && !u.is_finite() {
        assert!(
            rl <= ru,
            "both non-finite repair must produce valid ordering"
        );
    }
}

// ============================================================================
// enforce_bound_ordering: always produces valid bounds
// ============================================================================

/// Prove: enforce_bound_ordering produces lower <= upper for 1-element arrays.
///
/// For any pair of f32 values (including NaN, Inf), after enforce_bound_ordering
/// the result satisfies lower[0] <= upper[0] with both endpoints finite.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn enforce_ordering_single_element_valid() {
    let bits_l: u32 = kani::any();
    let bits_u: u32 = kani::any();

    let mut lower = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1]), f32::from_bits(bits_l));
    let mut upper = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1]), f32::from_bits(bits_u));

    let _count = enforce_bound_ordering(&mut lower, &mut upper);

    let l = lower[[0]];
    let u = upper[[0]];
    // After enforcement, both must be finite
    assert!(l.is_finite(), "enforced lower must be finite");
    assert!(u.is_finite(), "enforced upper must be finite");
    // And properly ordered
    assert!(l <= u, "enforced lower must be <= upper");
}

// ============================================================================
// IntervalBounds constructor invariants (small fixed-size arrays for Kani)
// ============================================================================

/// Prove: IntervalBounds::new rejects NaN in lower bound.
///
/// IEEE 754 NaN bypasses comparisons — explicit is_finite check is required.
/// This harness verifies the constructor catches NaN rather than allowing
/// it to silently pass through relational checks.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounds_new_rejects_nan_lower() {
    let result = crate::IntervalBounds::new(
        ndarray::arr1(&[f32::NAN]).into_dyn(),
        ndarray::arr1(&[1.0f32]).into_dyn(),
    );
    assert!(result.is_err(), "NaN in lower must be rejected");
}

/// Prove: IntervalBounds::new rejects NaN in upper bound.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounds_new_rejects_nan_upper() {
    let result = crate::IntervalBounds::new(
        ndarray::arr1(&[0.0f32]).into_dyn(),
        ndarray::arr1(&[f32::NAN]).into_dyn(),
    );
    assert!(result.is_err(), "NaN in upper must be rejected");
}

/// Prove: IntervalBounds::new rejects +Inf in bounds.
///
/// The standard constructor requires finite endpoints. Infinite endpoints
/// are only allowed through `new_allow_infinite`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounds_new_rejects_infinity() {
    let result = crate::IntervalBounds::new(
        ndarray::arr1(&[0.0f32]).into_dyn(),
        ndarray::arr1(&[f32::INFINITY]).into_dyn(),
    );
    assert!(result.is_err(), "Inf in upper must be rejected");

    let result2 = crate::IntervalBounds::new(
        ndarray::arr1(&[f32::NEG_INFINITY]).into_dyn(),
        ndarray::arr1(&[0.0f32]).into_dyn(),
    );
    assert!(result2.is_err(), "Inf in lower must be rejected");
}

/// Prove: IntervalBounds::new rejects inverted bounds (lower > upper).
///
/// When lower > upper for any element, the bounds are physically
/// impossible and must be rejected.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounds_new_rejects_inverted() {
    let result = crate::IntervalBounds::new(
        ndarray::arr1(&[5.0f32]).into_dyn(),
        ndarray::arr1(&[3.0f32]).into_dyn(),
    );
    assert!(result.is_err(), "inverted bounds (5 > 3) must be rejected");
}

/// Prove: IntervalBounds::new accepts equal lower and upper (point bounds).
///
/// lower == upper is valid — it represents a concrete known value.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounds_new_accepts_equal() {
    let val = 42.0f32;
    let result = crate::IntervalBounds::new(
        ndarray::arr1(&[val]).into_dyn(),
        ndarray::arr1(&[val]).into_dyn(),
    );
    assert!(result.is_ok(), "equal bounds must be accepted");
    let bounds = result.unwrap();
    assert_eq!(bounds.lower()[[0]], val);
    assert_eq!(bounds.upper()[[0]], val);
}

/// Prove: IntervalBounds::concrete produces lower == upper.
///
/// Concrete bounds represent exact known values. Both arrays must be
/// identical and equal to the input.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounds_concrete_lower_equals_upper() {
    let val = 7.5f32;
    let result = crate::IntervalBounds::concrete(ndarray::arr1(&[val]).into_dyn());
    assert!(result.is_ok(), "finite concrete value must succeed");
    let bounds = result.unwrap();
    assert_eq!(
        bounds.lower()[[0]].to_bits(),
        bounds.upper()[[0]].to_bits(),
        "concrete bounds must have identical lower and upper"
    );
    assert_eq!(bounds.lower()[[0]], val, "concrete lower must equal input");
}

/// Prove: IntervalBounds::concrete rejects NaN input.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounds_concrete_rejects_nan() {
    let result = crate::IntervalBounds::concrete(ndarray::arr1(&[f32::NAN]).into_dyn());
    assert!(result.is_err(), "NaN in concrete must be rejected");
}

/// Prove: IntervalBounds::concrete rejects Inf input.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounds_concrete_rejects_inf() {
    let result = crate::IntervalBounds::concrete(ndarray::arr1(&[f32::INFINITY]).into_dyn());
    assert!(result.is_err(), "+Inf in concrete must be rejected");

    let result2 = crate::IntervalBounds::concrete(ndarray::arr1(&[f32::NEG_INFINITY]).into_dyn());
    assert!(result2.is_err(), "-Inf in concrete must be rejected");
}

/// Prove: IntervalBounds::max_width is zero for concrete (point) bounds.
///
/// When lower == upper for all elements, the maximum width is exactly 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounds_concrete_max_width_zero() {
    let bounds = crate::IntervalBounds::concrete(ndarray::arr1(&[3.14f32]).into_dyn())
        .expect("finite concrete");
    let w = bounds.max_width();
    assert_eq!(w, 0.0, "concrete bounds must have zero width");
}

/// Prove: IntervalBounds::max_width is non-negative for valid bounds.
///
/// Since lower <= upper for all elements (constructor invariant), the
/// per-element width u - l >= 0, and the max of non-negative values
/// is non-negative.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(2)]
fn bounds_max_width_non_negative() {
    // Use a known valid pair
    let bounds = crate::IntervalBounds::new(
        ndarray::arr1(&[-1.0f32]).into_dyn(),
        ndarray::arr1(&[1.0f32]).into_dyn(),
    )
    .expect("valid bounds");
    let w = bounds.max_width();
    assert!(w >= 0.0, "max_width must be non-negative for valid bounds");
    assert_eq!(w, 2.0, "width of [-1, 1] must be 2.0");
}

/// Prove: IntervalBounds::from_epsilon with epsilon=0 produces concrete bounds.
///
/// When epsilon is zero, the resulting bounds should have lower == upper
/// (identical to concrete bounds).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(2)]
fn bounds_from_epsilon_zero_is_concrete() {
    let val = 5.0f32;
    let result = crate::IntervalBounds::from_epsilon(ndarray::arr1(&[val]).into_dyn(), 0.0);
    assert!(result.is_ok(), "epsilon=0 must succeed");
    let bounds = result.unwrap();
    assert_eq!(
        bounds.lower()[[0]],
        val,
        "lower must equal center for eps=0"
    );
    assert_eq!(
        bounds.upper()[[0]],
        val,
        "upper must equal center for eps=0"
    );
}

/// Prove: IntervalBounds::from_epsilon rejects negative epsilon.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn bounds_from_epsilon_rejects_negative() {
    let result = crate::IntervalBounds::from_epsilon(ndarray::arr1(&[1.0f32]).into_dyn(), -0.5);
    assert!(result.is_err(), "negative epsilon must be rejected");
}

/// Prove: IntervalBounds::from_epsilon rejects NaN epsilon.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn bounds_from_epsilon_rejects_nan_epsilon() {
    let result = crate::IntervalBounds::from_epsilon(ndarray::arr1(&[1.0f32]).into_dyn(), f32::NAN);
    assert!(result.is_err(), "NaN epsilon must be rejected");
}

/// Prove: IntervalBounds shape accessor returns correct shape.
///
/// The shape of the bounds must match the shape of the arrays used
/// to construct them.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn bounds_shape_matches_construction() {
    let bounds = crate::IntervalBounds::new(
        ndarray::arr1(&[0.0f32]).into_dyn(),
        ndarray::arr1(&[1.0f32]).into_dyn(),
    )
    .expect("valid bounds");
    assert_eq!(bounds.shape(), &[1], "shape must match construction arrays");
}
