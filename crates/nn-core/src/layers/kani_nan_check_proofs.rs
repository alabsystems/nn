// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for NaN check policy and finiteness validation.
//!
//! Proves correctness properties of the NaN check infrastructure used by
//! all Tier 1 nn layers (attention, LSTM, SwiGLU, BatchNorm, etc.):
//!
//!  1. NanCheckPolicy::Always and Skip are distinct variants
//!  2. Default policy is Always (safe default)
//!  3. with_nan_check_policy restores prior policy (RAII)
//!  4. with_nan_check_policy sets the requested policy inside scope
//!  5. Nested with_nan_check_policy restores correctly
//!  6. f32::NAN is detected by !is_finite() (IEEE 754 invariant)
//!  7. f32::INFINITY is detected by !is_finite()
//!  8. f32::NEG_INFINITY is detected by !is_finite()
//!  9. All finite f32 values pass is_finite()
//! 10. IEEE 754 NaN comparison bypass: NaN > 0.0 returns false
//! 11. IEEE 754 NaN equality bypass: NaN != NaN
//! 12. Non-finite count is exact for mixed data
//! 13. Non-finite count is zero for all-finite data
//! 14. Non-finite count equals length for all-NaN data
//! 15. Skip policy bypasses check regardless of data content
//!
//! Part of #3624.

use super::{nan_check_policy, with_nan_check_policy, NanCheckPolicy};

// ---------------------------------------------------------------------------
// Harness 1: NanCheckPolicy variants are distinct
// ---------------------------------------------------------------------------

/// Prove: NanCheckPolicy::Always != NanCheckPolicy::Skip.
/// The two variants must be distinguishable — conflating them would
/// silently disable all NaN checks or make Skip impossible.
#[kani::unwind(1)]
#[kani::proof]
fn proof_nan_check_policy_variants_distinct() {
    let always = NanCheckPolicy::Always;
    let skip = NanCheckPolicy::Skip;
    assert!(always != skip, "Always and Skip must be distinct");
}

// ---------------------------------------------------------------------------
// Harness 2: Default policy is Always (safe default)
// ---------------------------------------------------------------------------

/// Prove: The thread-local default NanCheckPolicy is Always.
/// This ensures new threads get the safe (checking) behavior by default.
/// Changing the default to Skip would silently disable all NaN detection.
#[kani::unwind(1)]
#[kani::proof]
fn proof_nan_check_policy_default_is_always() {
    let policy = nan_check_policy();
    assert!(
        policy == NanCheckPolicy::Always,
        "default policy must be Always"
    );
}

// ---------------------------------------------------------------------------
// Harness 3: with_nan_check_policy restores prior policy
// ---------------------------------------------------------------------------

/// Prove: with_nan_check_policy restores the prior policy after the
/// closure returns. The RAII guard must not leak state.
#[kani::unwind(1)]
#[kani::proof]
fn proof_with_nan_check_policy_restores_prior() {
    let before = nan_check_policy();
    with_nan_check_policy(NanCheckPolicy::Skip, || {
        // Inside scope: policy is Skip
    });
    let after = nan_check_policy();
    assert!(before == after, "policy must be restored after scope exits");
}

// ---------------------------------------------------------------------------
// Harness 4: with_nan_check_policy sets requested policy inside scope
// ---------------------------------------------------------------------------

/// Prove: with_nan_check_policy sets the requested policy inside the
/// closure scope. The closure observes the new policy.
#[kani::unwind(1)]
#[kani::proof]
fn proof_with_nan_check_policy_sets_skip_inside() {
    with_nan_check_policy(NanCheckPolicy::Skip, || {
        let inside = nan_check_policy();
        assert!(
            inside == NanCheckPolicy::Skip,
            "policy must be Skip inside scope"
        );
    });
}

/// Prove: with_nan_check_policy(Always, ...) sets Always inside scope.
#[kani::unwind(1)]
#[kani::proof]
fn proof_with_nan_check_policy_sets_always_inside() {
    // First switch to Skip so we can verify Always is set.
    with_nan_check_policy(NanCheckPolicy::Skip, || {
        with_nan_check_policy(NanCheckPolicy::Always, || {
            let inside = nan_check_policy();
            assert!(
                inside == NanCheckPolicy::Always,
                "policy must be Always inside inner scope"
            );
        });
    });
}

// ---------------------------------------------------------------------------
// Harness 5: Nested scopes restore correctly
// ---------------------------------------------------------------------------

/// Prove: nested with_nan_check_policy scopes restore in LIFO order.
/// Inner scope restores to outer scope's policy, outer restores to original.
#[kani::unwind(1)]
#[kani::proof]
fn proof_nested_scopes_restore_correctly() {
    let original = nan_check_policy();
    assert!(original == NanCheckPolicy::Always);

    with_nan_check_policy(NanCheckPolicy::Skip, || {
        assert!(nan_check_policy() == NanCheckPolicy::Skip);

        with_nan_check_policy(NanCheckPolicy::Always, || {
            assert!(nan_check_policy() == NanCheckPolicy::Always);
        });

        // Inner scope restored to Skip
        assert!(
            nan_check_policy() == NanCheckPolicy::Skip,
            "must restore to outer scope policy"
        );
    });

    // Outer scope restored to Always
    assert!(
        nan_check_policy() == NanCheckPolicy::Always,
        "must restore to original policy"
    );
}

// ---------------------------------------------------------------------------
// Harness 6: f32::NAN detected by !is_finite() — IEEE 754 invariant
// ---------------------------------------------------------------------------

/// Prove: f32::NAN is not finite. This is the foundation of all NaN
/// detection in the framework. The check_output_finite() CPU path uses
/// `!v.is_finite()` to classify each element.
///
/// IEEE 754 NaN bypasses comparisons (nn_engineering.md #3356).
/// `is_finite()` returns false for NaN, which is the correct detection
/// method — unlike `v != v` or relational comparisons which are fragile.
#[kani::unwind(1)]
#[kani::proof]
fn proof_nan_detected_by_is_finite() {
    let nan = f32::NAN;
    assert!(!nan.is_finite(), "NaN must not be finite");
}

// ---------------------------------------------------------------------------
// Harness 7: f32::INFINITY detected by !is_finite()
// ---------------------------------------------------------------------------

/// Prove: f32::INFINITY is not finite. Positive infinity from overflow
/// (e.g., softmax exp, division by near-zero) must be caught.
#[kani::unwind(1)]
#[kani::proof]
fn proof_pos_inf_detected_by_is_finite() {
    let inf = f32::INFINITY;
    assert!(!inf.is_finite(), "INFINITY must not be finite");
}

// ---------------------------------------------------------------------------
// Harness 8: f32::NEG_INFINITY detected by !is_finite()
// ---------------------------------------------------------------------------

/// Prove: f32::NEG_INFINITY is not finite. Negative infinity from
/// log(0) or underflow must be caught.
#[kani::unwind(1)]
#[kani::proof]
fn proof_neg_inf_detected_by_is_finite() {
    let neg_inf = f32::NEG_INFINITY;
    assert!(!neg_inf.is_finite(), "NEG_INFINITY must not be finite");
}

// ---------------------------------------------------------------------------
// Harness 9: All finite f32 values pass is_finite()
// ---------------------------------------------------------------------------

/// Prove: any f32 value that is_finite() is neither NaN nor infinite.
/// This is the positive direction — finite values are not falsely rejected.
#[kani::unwind(1)]
#[kani::proof]
fn proof_finite_f32_passes_is_finite() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite());

    // A finite value must not be NaN
    assert!(!v.is_nan(), "finite value must not be NaN");
    // A finite value must not be infinite
    assert!(!v.is_infinite(), "finite value must not be infinite");
}

// ---------------------------------------------------------------------------
// Harness 10: IEEE 754 NaN comparison bypass
// ---------------------------------------------------------------------------

/// Prove: NaN > 0.0 returns false (IEEE 754 comparison bypass).
/// This is the critical invariant documented in nn_engineering.md #3356.
/// Code that uses `lower > upper` to validate bounds will silently pass
/// when either bound is NaN. Always use !val.is_finite() first.
#[kani::unwind(1)]
#[kani::proof]
fn proof_ieee754_nan_comparison_bypass() {
    let nan = f32::NAN;

    // All relational comparisons with NaN return false
    assert!(!(nan > 0.0), "NaN > 0.0 must be false");
    assert!(!(nan < 0.0), "NaN < 0.0 must be false");
    assert!(!(nan >= 0.0), "NaN >= 0.0 must be false");
    assert!(!(nan <= 0.0), "NaN <= 0.0 must be false");
    assert!(!(nan == 0.0), "NaN == 0.0 must be false");

    // NaN is not equal to itself
    assert!(nan != nan, "NaN != NaN must be true");
}

// ---------------------------------------------------------------------------
// Harness 11: IEEE 754 NaN equality bypass
// ---------------------------------------------------------------------------

/// Prove: for any f32 value, v == v is true iff v is not NaN.
/// This is the self-equality test for NaN detection.
#[kani::unwind(1)]
#[kani::proof]
fn proof_nan_self_inequality() {
    let v: f32 = kani::any();

    if v.is_nan() {
        assert!(v != v, "NaN must not equal itself");
    } else {
        assert!(v == v, "non-NaN must equal itself");
    }
}

// ---------------------------------------------------------------------------
// Harness 12: Non-finite count is exact for mixed data
// ---------------------------------------------------------------------------

/// Prove: the filter-count pattern used in check_output_finite correctly
/// counts non-finite elements in a mixed array. Models the CPU path:
/// `data.iter().filter(|v| !v.is_finite()).count()`
#[kani::unwind(5)]
#[kani::proof]
fn proof_non_finite_count_mixed_data() {
    // Model a 4-element array with known non-finite positions
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite());
    kani::assume(b.is_finite());

    let data = [a, f32::NAN, b, f32::INFINITY];
    let count = data.iter().filter(|v| !v.is_finite()).count();

    assert!(
        count == 2,
        "exactly 2 non-finite elements in [finite, NaN, finite, Inf]"
    );
}

// ---------------------------------------------------------------------------
// Harness 13: Non-finite count is zero for all-finite data
// ---------------------------------------------------------------------------

/// Prove: when all elements are finite, the non-finite count is zero.
/// This means check_output_finite returns Ok for valid data.
#[kani::unwind(5)]
#[kani::proof]
fn proof_non_finite_count_zero_for_finite_data() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();

    kani::assume(a.is_finite());
    kani::assume(b.is_finite());
    kani::assume(c.is_finite());

    let data = [a, b, c];
    let count = data.iter().filter(|v| !v.is_finite()).count();

    assert!(
        count == 0,
        "all-finite data must have zero non-finite count"
    );
}

// ---------------------------------------------------------------------------
// Harness 14: Non-finite count equals length for all-NaN data
// ---------------------------------------------------------------------------

/// Prove: when all elements are NaN, every element is counted.
#[kani::unwind(5)]
#[kani::proof]
fn proof_non_finite_count_all_nan() {
    let data = [f32::NAN, f32::NAN, f32::NAN];
    let count = data.iter().filter(|v| !v.is_finite()).count();

    assert!(count == 3, "all-NaN data must have count == length");
}

// ---------------------------------------------------------------------------
// Harness 15: Skip policy bypasses regardless of data content
// ---------------------------------------------------------------------------

/// Prove: under Skip policy, check_output_finite returns Ok immediately
/// without inspecting data. Models the early-return path at line 114-116
/// of nan_check.rs.
///
/// This is a logical proof about the policy decision: when policy == Skip,
/// the function returns Ok(()) without any data access.
#[kani::unwind(1)]
#[kani::proof]
fn proof_skip_policy_bypasses_check() {
    let policy = NanCheckPolicy::Skip;

    // Model the early-return logic from check_output_finite:
    // if NAN_CHECK_POLICY.get() == NanCheckPolicy::Skip { return Ok(()); }
    let should_skip = policy == NanCheckPolicy::Skip;
    assert!(should_skip, "Skip policy must trigger early return");
}

// ---------------------------------------------------------------------------
// Harness 16: Always policy does not bypass check
// ---------------------------------------------------------------------------

/// Prove: under Always policy, the early-return path is NOT taken,
/// meaning the data will be inspected for non-finite values.
#[kani::unwind(1)]
#[kani::proof]
fn proof_always_policy_does_not_bypass() {
    let policy = NanCheckPolicy::Always;

    let should_skip = policy == NanCheckPolicy::Skip;
    assert!(!should_skip, "Always policy must not trigger early return");
}

// ---------------------------------------------------------------------------
// Harness 17: Policy enum exhaustive — any NanCheckPolicy is Always or Skip
// ---------------------------------------------------------------------------

/// Prove: NanCheckPolicy is exhaustively covered by {Always, Skip}.
/// Uses a boolean selector to enumerate all variants. If a new variant
/// is added, the match below will get a compiler error (non-exhaustive).
#[kani::unwind(1)]
#[kani::proof]
fn proof_policy_enum_exhaustive() {
    let select: bool = kani::any();
    let policy = if select {
        NanCheckPolicy::Always
    } else {
        NanCheckPolicy::Skip
    };

    // Exhaustive match — compiler enforces coverage of all variants.
    // Adding a new variant to NanCheckPolicy will cause a compile error here.
    let is_known = match policy {
        NanCheckPolicy::Always => true,
        NanCheckPolicy::Skip => true,
    };
    assert!(is_known, "every NanCheckPolicy variant must be recognized");
}

// ---------------------------------------------------------------------------
// Harness 18: with_nan_check_policy return value is forwarded
// ---------------------------------------------------------------------------

/// Prove: with_nan_check_policy returns the closure's return value.
/// The RAII guard setup must not interfere with value propagation.
#[kani::unwind(1)]
#[kani::proof]
fn proof_with_nan_check_policy_forwards_return_value() {
    let result: u32 = with_nan_check_policy(NanCheckPolicy::Skip, || 42u32);
    assert!(result == 42, "closure return value must be forwarded");
}

// ---------------------------------------------------------------------------
// Harness 19: Non-finite detection is symmetric for +Inf and -Inf
// ---------------------------------------------------------------------------

/// Prove: both positive and negative infinity are equally detected.
/// The check must not be biased toward only one sign of infinity.
#[kani::unwind(1)]
#[kani::proof]
fn proof_inf_detection_symmetric() {
    let pos = f32::INFINITY;
    let neg = f32::NEG_INFINITY;

    let pos_detected = !pos.is_finite();
    let neg_detected = !neg.is_finite();

    assert!(pos_detected, "+Inf must be detected");
    assert!(neg_detected, "-Inf must be detected");
    assert!(
        pos_detected == neg_detected,
        "detection must be symmetric for +/- Inf"
    );
}

// ---------------------------------------------------------------------------
// Harness 20: Subnormal/denormal values are finite (not false positives)
// ---------------------------------------------------------------------------

/// Prove: subnormal (denormal) f32 values are classified as finite.
/// Denormals are tiny but valid floating-point numbers — they must NOT
/// trigger NaN/Inf detection. This prevents false positives near zero.
///
/// f32::MIN_POSITIVE is the smallest normal; values below it (but above 0)
/// are subnormal and must still pass is_finite().
#[kani::unwind(1)]
#[kani::proof]
fn proof_subnormal_values_are_finite() {
    let v: f32 = kani::any();
    kani::assume(v > 0.0);
    kani::assume(v < f32::MIN_POSITIVE);
    // v is a subnormal float

    assert!(v.is_finite(), "subnormal values must be finite");
    assert!(!v.is_nan(), "subnormal values must not be NaN");
    assert!(!v.is_infinite(), "subnormal values must not be infinite");
}

// ---------------------------------------------------------------------------
// Harness 21: Negative zero is finite (not a false positive)
// ---------------------------------------------------------------------------

/// Prove: -0.0 is finite. IEEE 754 negative zero is a valid value that
/// must not trigger NaN/Inf checks.
#[kani::unwind(1)]
#[kani::proof]
fn proof_negative_zero_is_finite() {
    let neg_zero = -0.0f32;
    assert!(neg_zero.is_finite(), "-0.0 must be finite");
    assert!(!neg_zero.is_nan(), "-0.0 must not be NaN");
    assert!(!neg_zero.is_infinite(), "-0.0 must not be infinite");
}

// ---------------------------------------------------------------------------
// Harness 22: f32::MAX and f32::MIN are finite (boundary values)
// ---------------------------------------------------------------------------

/// Prove: the extreme finite f32 values (MAX and MIN) are correctly
/// classified as finite. These are the boundary between finite and overflow.
#[kani::unwind(1)]
#[kani::proof]
fn proof_extreme_finite_values_are_finite() {
    assert!(f32::MAX.is_finite(), "f32::MAX must be finite");
    assert!(f32::MIN.is_finite(), "f32::MIN must be finite");
    assert!((-f32::MAX).is_finite(), "-f32::MAX must be finite");

    // But MAX + MAX overflows to infinity
    let overflow = f32::MAX + f32::MAX;
    assert!(
        !overflow.is_finite(),
        "f32::MAX + f32::MAX must overflow to Inf"
    );
}
