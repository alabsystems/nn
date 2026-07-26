// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conversion between nn-core [`IntervalBounds`] and NY
//! [`BoundedTensor`](ny_api::BoundedTensor).
//!
//! These two types represent the same concept (per-element interval bounds)
//! but live in separate crates to break the circular dependency between
//! nn-core and NY. This module bridges them.
//!
//! Rust's orphan rule prevents implementing `From`/`Into` across foreign
//! crates, so these are free functions rather than trait impls.

use crate::error::StructuralError;
use crate::VerifyError;
use ny_api::BoundedTensor;
use nn_core::IntervalBounds;

/// Convert NY `BoundedTensor` to nn-core `IntervalBounds`.
///
/// Uses `new_allow_infinite` since `BoundedTensor` may contain infinite bounds
/// from conservative propagation.
#[must_use = "returns a Result that may contain an error"]
pub fn to_interval_bounds(bt: BoundedTensor) -> Result<IntervalBounds, VerifyError> {
    let (lower, upper) = bt.into_parts();
    IntervalBounds::new_allow_infinite(lower, upper).map_err(|e| {
        VerifyError::from(StructuralError::BoundsConversion(format!(
            "BoundedTensor -> IntervalBounds: {e}"
        )))
    })
}

/// Convert nn-core `IntervalBounds` to NY `BoundedTensor`.
#[must_use = "returns a Result that may contain an error"]
pub fn to_bounded_tensor(ib: IntervalBounds) -> Result<BoundedTensor, VerifyError> {
    let (lower, upper) = ib.into_parts();
    BoundedTensor::new_allow_infinite(lower, upper).map_err(|e| {
        VerifyError::from(StructuralError::BoundsConversion(format!(
            "IntervalBounds -> BoundedTensor: {e}"
        )))
    })
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    /// Proves bounds_bridge roundtrip preserves values for a single-element
    /// IntervalBounds → BoundedTensor → IntervalBounds conversion.
    ///
    /// For any pair of finite f32 values where lower <= upper, the roundtrip
    /// must produce identical bit patterns.
    #[kani::unwind(64)]
    #[kani::proof]
    fn bounds_bridge_scalar_roundtrip_lossless() {
        let lower_val: f32 = kani::any();
        let upper_val: f32 = kani::any();

        kani::assume(lower_val.is_finite());
        kani::assume(upper_val.is_finite());
        kani::assume(lower_val <= upper_val);

        let lower = ArrayD::from_elem(IxDyn(&[1]), lower_val);
        let upper = ArrayD::from_elem(IxDyn(&[1]), upper_val);

        let ib = IntervalBounds::new(lower, upper).expect("finite lower <= upper must succeed");

        let bt = to_bounded_tensor(ib).expect("IntervalBounds -> BoundedTensor must succeed");
        let ib2 = to_interval_bounds(bt).expect("BoundedTensor -> IntervalBounds must succeed");

        let (rt_lower, rt_upper) = ib2.lower_upper();
        assert_eq!(
            rt_lower[[0]].to_bits(),
            lower_val.to_bits(),
            "lower must round-trip exactly"
        );
        assert_eq!(
            rt_upper[[0]].to_bits(),
            upper_val.to_bits(),
            "upper must round-trip exactly"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    /// Contract test: nn-core and gamma-core FALLBACK_BOUND must be identical.
    ///
    /// Both crates define FALLBACK_BOUND independently (nn-core for
    /// self-containment). If they diverge, bounds arithmetic in nn-core
    /// would produce different repair values than gamma-core, and
    /// conversions via this bridge would silently corrupt verification
    /// results. This test catches divergence at build time.
    #[test]
    fn test_fallback_bound_synchronized() {
        let nn_core_fallback = nn_core::bounds::FALLBACK_BOUND;
        let gamma_core_fallback = ny_core::FALLBACK_BOUND;
        assert_eq!(
            nn_core_fallback, gamma_core_fallback,
            "nn-core FALLBACK_BOUND ({nn_core_fallback}) != gamma-core FALLBACK_BOUND \
             ({gamma_core_fallback}) — bounds bridge conversions will produce incorrect results"
        );
    }

    /// Contract test: nn-core ULP functions match the NY algorithm
    /// for finite inputs.
    ///
    /// Reference: `NY/crates/gamma-tensor/src/rounding.rs` (next_up_f32,
    /// next_down_f32). The shared algorithm is:
    ///   - NaN → NaN, ±inf → ±inf (or step for NY)
    ///   - 0.0 → smallest subnormal in the requested direction
    ///   - positive: bits ± 1, negative: bits ∓ 1
    ///
    /// We inline the NY algorithm (without nn's infinity guards) and
    /// verify it matches nn-core for all finite inputs. If NY changes
    /// the bit-manipulation logic, this test must be updated. Part of #680.
    #[test]
    fn test_ulp_functions_match_gamma_crown_for_finite_inputs() {
        /// NY's next_up_f32 (no infinity guards).
        /// Source: gamma-tensor/src/rounding.rs:16-31.
        fn gc_next_up(x: f32) -> f32 {
            if x.is_nan() || x == f32::INFINITY {
                return x;
            }
            if x == 0.0 {
                return f32::from_bits(1);
            }
            let bits = x.to_bits();
            if x.is_sign_positive() {
                f32::from_bits(bits + 1)
            } else {
                f32::from_bits(bits - 1)
            }
        }

        /// NY's next_down_f32 (no infinity guards).
        /// Source: gamma-tensor/src/rounding.rs:44-59.
        fn gc_next_down(x: f32) -> f32 {
            if x.is_nan() || x == f32::NEG_INFINITY {
                return x;
            }
            if x == 0.0 {
                return f32::from_bits(0x8000_0001);
            }
            let bits = x.to_bits();
            if x.is_sign_positive() {
                f32::from_bits(bits - 1)
            } else {
                f32::from_bits(bits + 1)
            }
        }

        // Representative finite values including edge cases.
        let test_values: &[f32] = &[
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::MAX,
            f32::MIN,
            0.5,
            -0.5,
            1e-38,
            -1e-38,
            1e38,
            -1e38,
            f32::from_bits(1),           // smallest positive subnormal
            f32::from_bits(0x8000_0001), // smallest negative subnormal
            std::f32::consts::PI,
            -std::f32::consts::E,
        ];

        for &v in test_values {
            assert_eq!(
                nn_core::next_down_f32(v).to_bits(),
                gc_next_down(v).to_bits(),
                "next_down_f32 diverged from NY for {v} (bits: {:#010X})",
                v.to_bits()
            );
            assert_eq!(
                nn_core::next_up_f32(v).to_bits(),
                gc_next_up(v).to_bits(),
                "next_up_f32 diverged from NY for {v} (bits: {:#010X})",
                v.to_bits()
            );
        }
    }

    /// Document intentional divergence: nn preserves infinity sentinels (#171),
    /// NY steps toward finite values.
    ///
    /// NY preserves +inf through `next_up` and -inf through `next_down`
    /// (rounding.rs:17, :44), but for infeasible sentinels (+inf lower, -inf upper)
    /// the *opposite* functions apply: `next_down(+inf)` = MAX on lower,
    /// `next_up(-inf)` = -MAX on upper. So NY's round_for_soundness
    /// converts (+inf, -inf) to (MAX, -MAX) — still inverted, but finite.
    /// nn preserves the exact (+inf, -inf) pattern (#171). Part of #680.
    #[test]
    fn test_ulp_infinity_divergence_documented() {
        // nn: preserves +inf sentinel through next_down (returns +inf, per #171)
        assert_eq!(nn_core::next_down_f32(f32::INFINITY), f32::INFINITY);
        // nn: preserves -inf sentinel through next_up (returns -inf, per #171)
        assert_eq!(nn_core::next_up_f32(f32::NEG_INFINITY), f32::NEG_INFINITY);

        // NY: next_down(+inf) = f32::MAX, next_up(-inf) = f32::MIN.
        // The divergence is intentional — nn uses ±inf as infeasible sentinels
        // that must survive round_for_soundness unchanged.
    }

    /// Round-trip: nn-core IntervalBounds → BoundedTensor → IntervalBounds.
    #[test]
    fn test_roundtrip_interval_bounds_to_bounded_tensor() {
        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.0, 0.5]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let original = IntervalBounds::new(lower.clone(), upper.clone()).unwrap();

        let bt = to_bounded_tensor(original).expect("conversion should succeed");
        let roundtripped = to_interval_bounds(bt).expect("roundtrip should succeed");

        let (rt_lower, rt_upper) = roundtripped.lower_upper();
        assert_eq!(rt_lower, lower.view());
        assert_eq!(rt_upper, upper.view());
    }

    /// Round-trip with infinite bounds (allowed by new_allow_infinite).
    #[test]
    fn test_roundtrip_infinite_bounds() {
        let lower = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, -1.0]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, f32::INFINITY]).unwrap();
        let original = IntervalBounds::new_allow_infinite(lower.clone(), upper.clone()).unwrap();

        let bt = to_bounded_tensor(original).expect("conversion should succeed");
        let roundtripped = to_interval_bounds(bt).expect("roundtrip should succeed");

        let (rt_lower, rt_upper) = roundtripped.lower_upper();
        assert_eq!(rt_lower, lower.view());
        assert_eq!(rt_upper, upper.view());
    }
}
