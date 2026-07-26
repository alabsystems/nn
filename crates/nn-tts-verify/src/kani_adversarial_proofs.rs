// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for adversarial robustness verification.
//!
//! Proves correctness of `compute_mean_width`, `check_property` variants
//! (NY gated) and `embedding_bounds_for_token_set` (always available).
//!
//! Properties proved:
//!
//! 1. `compute_mean_width` non-negativity for valid (upper >= lower) bounds.
//! 2. `compute_mean_width` length mismatch rejection.
//! 3. `compute_mean_width` empty-input returns 0.
//! 4. `compute_mean_width` rejects non-finite widths.
//! 5. `check_property(DurationPositive)` correctness.
//! 6. `check_property(F0Bounded)` correctness.
//! 7. `check_property(OutputStable)` correctness.
//! 8. `embedding_bounds_for_token_set` produces lower <= upper.
//! 9. `embedding_bounds_for_token_set` rejects empty token_ids.
//! 10. `embedding_bounds_for_token_set` rejects out-of-range tokens.
//! 11. `embedding_bounds_for_token_set` rejects weight/dimension mismatch.
//! 12. Single-token bounds are point bounds (lower == upper).

// ---------------------------------------------------------------------------
// embedding_bounds_for_token_set Proofs (always available, no feature gate)
// ---------------------------------------------------------------------------

/// Prove: `embedding_bounds_for_token_set` produces lower <= upper for
/// every dimension when given valid inputs.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(5)]
fn embedding_bounds_lower_leq_upper() {
    // 2 tokens, 2-dim embeddings.
    let e00: f64 = kani::any();
    let e01: f64 = kani::any();
    let e10: f64 = kani::any();
    let e11: f64 = kani::any();
    kani::assume(e00.is_finite() && e01.is_finite());
    kani::assume(e10.is_finite() && e11.is_finite());
    kani::assume(e00.abs() <= 1e6 && e01.abs() <= 1e6);
    kani::assume(e10.abs() <= 1e6 && e11.abs() <= 1e6);

    let weights = [e00, e01, e10, e11]; // [vocab=2, dim=2]
    let token_ids = [0u32, 1];
    let (lo, hi) =
        crate::adversarial::embedding_bounds_for_token_set(&weights, 2, 2, &token_ids).unwrap();

    assert_eq!(lo.len(), 2);
    assert_eq!(hi.len(), 2);
    for d in 0..2 {
        assert!(lo[d] <= hi[d], "lower must be <= upper in dimension {d}");
    }
}

/// Prove: single-token bounds are point bounds (lower == upper == embedding).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn embedding_bounds_single_token_is_point() {
    let e0: f64 = kani::any();
    let e1: f64 = kani::any();
    kani::assume(e0.is_finite() && e1.is_finite());

    let weights = [e0, e1, 0.0, 0.0]; // vocab=2, dim=2, only token 0 used
    let (lo, hi) =
        crate::adversarial::embedding_bounds_for_token_set(&weights, 2, 2, &[0]).unwrap();

    assert_eq!(lo[0], e0, "single token lower must equal embedding");
    assert_eq!(hi[0], e0, "single token upper must equal embedding");
    assert_eq!(lo[1], e1);
    assert_eq!(hi[1], e1);
}

/// Prove: `embedding_bounds_for_token_set` rejects empty token_ids.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn embedding_bounds_rejects_empty_tokens() {
    let weights = [1.0, 2.0];
    let result = crate::adversarial::embedding_bounds_for_token_set(&weights, 1, 2, &[]);
    assert!(result.is_err(), "empty token_ids must produce error");
}

/// Prove: `embedding_bounds_for_token_set` rejects out-of-range token.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn embedding_bounds_rejects_out_of_range_token() {
    let weights = [1.0, 2.0, 3.0, 4.0]; // vocab=2, dim=2
    let result = crate::adversarial::embedding_bounds_for_token_set(
        &weights,
        2,
        2,
        &[5], // token 5 >= vocab_size 2
    );
    assert!(result.is_err(), "out-of-range token must produce error");
}

/// Prove: `embedding_bounds_for_token_set` rejects weight/dimension mismatch.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn embedding_bounds_rejects_weight_size_mismatch() {
    let weights = [1.0, 2.0, 3.0]; // 3 elements, but vocab=2 * dim=2 = 4
    let result = crate::adversarial::embedding_bounds_for_token_set(&weights, 2, 2, &[0]);
    assert!(result.is_err(), "weight size mismatch must produce error");
}

// ---------------------------------------------------------------------------
// compute_mean_width Proofs (NY gated)
// ---------------------------------------------------------------------------

/// Prove: `compute_mean_width` returns non-negative for valid bounds.
///
/// When upper[i] >= lower[i] for all i, the mean width must be >= 0.
#[cfg(feature = "ny")]
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
fn mean_width_non_negative_for_valid_bounds() {
    let lo0: f32 = kani::any();
    let hi0: f32 = kani::any();
    let lo1: f32 = kani::any();
    let hi1: f32 = kani::any();
    kani::assume(lo0.is_finite() && hi0.is_finite());
    kani::assume(lo1.is_finite() && hi1.is_finite());
    kani::assume(lo0.abs() <= 1e6 && hi0.abs() <= 1e6);
    kani::assume(lo1.abs() <= 1e6 && hi1.abs() <= 1e6);
    kani::assume(hi0 >= lo0 && hi1 >= lo1);

    let lower = [lo0, lo1];
    let upper = [hi0, hi1];
    let result = crate::adversarial_robustness::compute_mean_width(&lower, &upper);
    assert!(result.is_ok());
    let width = result.unwrap();
    assert!(
        width >= -1e-10,
        "mean width must be non-negative for valid bounds, got {width}"
    );
}

/// Prove: `compute_mean_width` rejects length mismatch.
#[cfg(feature = "ny")]
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn mean_width_rejects_length_mismatch() {
    let lower = [0.0f32, 1.0];
    let upper = [1.0f32];
    let result = crate::adversarial_robustness::compute_mean_width(&lower, &upper);
    assert!(result.is_err(), "length mismatch must produce error");
}

/// Prove: `compute_mean_width` returns 0.0 for empty inputs.
#[cfg(feature = "ny")]
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn mean_width_empty_returns_zero() {
    let result = crate::adversarial_robustness::compute_mean_width(&[], &[]);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0.0, "empty inputs must return 0.0");
}

/// Prove: `compute_mean_width` rejects non-finite bound width.
///
/// If upper - lower is non-finite (Inf inputs), the function must reject it.
#[cfg(feature = "ny")]
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn mean_width_rejects_inf_width() {
    let lower_inf = [-f32::INFINITY];
    let upper_inf = [f32::INFINITY];
    let result = crate::adversarial_robustness::compute_mean_width(&lower_inf, &upper_inf);
    assert!(result.is_err(), "non-finite width must be rejected");
}

// ---------------------------------------------------------------------------
// check_property Proofs (NY gated)
// ---------------------------------------------------------------------------

/// Prove: `DurationPositive` holds iff ALL lower bounds > 0.
#[cfg(feature = "ny")]
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
fn check_property_duration_positive_correctness() {
    use crate::adversarial_robustness::RobustnessProperty;

    let lo0: f32 = kani::any();
    let lo1: f32 = kani::any();
    kani::assume(lo0.is_finite() && lo1.is_finite());
    kani::assume(lo0.abs() <= 1e6 && lo1.abs() <= 1e6);

    let lower = [lo0, lo1];
    let upper = [1.0f32, 1.0]; // upper doesn't matter for DurationPositive
    let result = crate::adversarial_robustness::check_property(
        &RobustnessProperty::DurationPositive,
        &lower,
        &upper,
    );
    assert!(result.is_ok());
    let holds = result.unwrap();
    let expected = lo0 > 0.0 && lo1 > 0.0;
    assert_eq!(
        holds, expected,
        "DurationPositive must hold iff all lower > 0"
    );
}

/// Prove: `F0Bounded` holds iff all lower >= min_hz AND all upper <= max_hz.
#[cfg(feature = "ny")]
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn check_property_f0_bounded_correctness() {
    use crate::adversarial_robustness::RobustnessProperty;

    let lo: f32 = kani::any();
    let hi: f32 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo.abs() <= 1e6 && hi.abs() <= 1e6);

    let prop = RobustnessProperty::F0Bounded {
        min_hz: 80.0,
        max_hz: 400.0,
    };
    let result = crate::adversarial_robustness::check_property(&prop, &[lo], &[hi]);
    assert!(result.is_ok());
    let holds = result.unwrap();
    let expected = lo >= 80.0f32 && hi <= 400.0f32;
    assert_eq!(
        holds, expected,
        "F0Bounded must check lower >= min and upper <= max"
    );
}

/// Prove: `OutputStable` holds iff mean width <= max_width.
#[cfg(feature = "ny")]
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn check_property_output_stable_correctness() {
    use crate::adversarial_robustness::RobustnessProperty;

    let lo: f32 = kani::any();
    let hi: f32 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo.abs() <= 1e4 && hi.abs() <= 1e4);
    kani::assume(hi >= lo);

    let prop = RobustnessProperty::OutputStable { max_width: 1.0 };
    let result = crate::adversarial_robustness::check_property(&prop, &[lo], &[hi]);
    assert!(result.is_ok());
    let holds = result.unwrap();
    let width = f64::from(hi) - f64::from(lo);
    let expected = width <= 1.0;
    assert_eq!(
        holds, expected,
        "OutputStable must hold iff mean width <= max_width"
    );
}
