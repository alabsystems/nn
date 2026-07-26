// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for quality metric evaluation functions.
//!
//! Proves properties of the pure functions in `quality.rs`:
//! `check_f0_range` and `QualityMetric` consistency.
//!
//! Note: `compute_mcd`, `compute_hnr`, `compute_snr`, etc. depend on DSP
//! (FFT, autocorrelation) which CBMC cannot model. Those are tested via
//! unit tests. This file covers the pure-logic functions that determine
//! pass/fail from computed values.
//!
//! Properties proved:
//! 1. `check_f0_range` returns passed=false for empty (all-unvoiced) contour.
//! 2. `check_f0_range` returns passed=true when all voiced frames are in range.
//! 3. `check_f0_range` value is in [0.0, 1.0] for any input.
//! 4. `check_f0_range` threshold is always 0.8.
//! 5. `compute_rms` rejects empty input.
//! 6. `QualityMetric` passed field is consistent with value vs threshold
//!    for higher-is-better metrics.

use super::{check_f0_range, compute_rms};

// ---------- CBMC transcendental stubs for Kani (#708) -----------------------

/// Nondeterministic stub for `f64::sqrt`.
/// CBMC cannot handle the sqrt intrinsic. Returns a finite non-negative f64.
fn sqrt_f64_stub(x: f64) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e20);
    if x > 0.0 {
        kani::assume(r > 0.0);
        kani::assume(r >= x.min(1.0));
    }
    if x >= 1.0 {
        kani::assume(r >= 1.0);
    }
    r
}

// ---------------------------------------------------------------------------
// check_f0_range proofs
// ---------------------------------------------------------------------------

/// Prove: check_f0_range returns passed=false for empty contour.
///
/// An empty F0 contour means no voiced frames were detected, which
/// indicates silence or noise. The metric must fail.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn check_f0_range_empty_contour_fails() {
    let result = check_f0_range(&[], 80.0, 400.0);
    assert!(!result.passed, "empty contour must not pass");
    assert_eq!(result.value, 0.0, "empty contour value must be 0.0");
}

/// Prove: check_f0_range returns passed=false for all-unvoiced contour.
///
/// Unvoiced frames have F0=0.0 and are filtered out. If all frames are
/// unvoiced, the metric must fail (same as empty).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn check_f0_range_all_unvoiced_fails() {
    let contour = [0.0, 0.0];
    let result = check_f0_range(&contour, 80.0, 400.0);
    assert!(!result.passed, "all-unvoiced contour must not pass");
}

/// Prove: check_f0_range returns passed=true when all voiced frames are in range.
///
/// If every voiced frame has F0 in [min_hz, max_hz], the ratio is 1.0
/// which exceeds the 0.8 threshold.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn check_f0_range_all_in_range_passes() {
    let min_hz: f64 = kani::any();
    let max_hz: f64 = kani::any();
    let f0: f64 = kani::any();
    kani::assume(min_hz.is_finite() && max_hz.is_finite() && f0.is_finite());
    kani::assume(min_hz > 0.0 && max_hz > min_hz);
    kani::assume(f0 >= min_hz && f0 <= max_hz);
    kani::assume(min_hz <= 1e4 && max_hz <= 1e4);

    let contour = [f0];
    let result = check_f0_range(&contour, min_hz, max_hz);
    assert_eq!(result.value, 1.0, "all-in-range ratio must be 1.0");
    assert!(result.passed, "all-in-range contour must pass");
}

/// Prove: check_f0_range value is in [0.0, 1.0] for any finite input.
///
/// The value is a ratio: in_range_count / voiced_count. Since
/// in_range_count <= voiced_count, the ratio must be in [0, 1].
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn check_f0_range_value_bounded() {
    let f1: f64 = kani::any();
    let f2: f64 = kani::any();
    kani::assume(f1.is_finite() && f2.is_finite());
    kani::assume(f1 >= 0.0 && f2 >= 0.0);
    kani::assume(f1 <= 1e4 && f2 <= 1e4);
    // At least one voiced frame
    kani::assume(f1 > 0.0 || f2 > 0.0);

    let contour = [f1, f2];
    let result = check_f0_range(&contour, 80.0, 400.0);
    assert!(
        result.value >= 0.0 && result.value <= 1.0,
        "f0_range value must be in [0.0, 1.0], got {}",
        result.value
    );
}

/// Prove: check_f0_range threshold is always 0.8.
///
/// The function hardcodes threshold=0.8 (80% of voiced frames must be
/// in range). This harness verifies the threshold is not accidentally
/// changed to a different value.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn check_f0_range_threshold_is_point_eight() {
    let contour = [200.0]; // any voiced frame
    let result = check_f0_range(&contour, 80.0, 400.0);
    assert_eq!(result.threshold, 0.8, "f0_range threshold must be 0.8");
}

/// Prove: check_f0_range name is always "f0_range".
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn check_f0_range_name_correct() {
    let contour = [200.0];
    let result = check_f0_range(&contour, 80.0, 400.0);
    assert_eq!(result.name, "f0_range", "metric name must be f0_range");
}

// ---------------------------------------------------------------------------
// compute_rms proofs
// ---------------------------------------------------------------------------

/// Prove: compute_rms rejects empty input.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn compute_rms_rejects_empty() {
    let result = compute_rms(&[], 0.01);
    assert!(result.is_err(), "empty input must be rejected");
}

/// Prove: compute_rms metric name is "rms_energy".
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
fn compute_rms_name_correct() {
    let result = compute_rms(&[0.5], 0.01);
    if let Ok(metric) = result {
        assert_eq!(metric.name, "rms_energy", "metric name must be rms_energy");
    }
}

/// Prove: compute_rms value is non-negative for any finite input.
///
/// RMS = sqrt(mean(x^2)). Since x^2 >= 0, mean(x^2) >= 0, sqrt >= 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
fn compute_rms_value_non_negative() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a.abs() <= 1.0 && b.abs() <= 1.0);

    let result = compute_rms(&[a, b], 0.0);
    if let Ok(metric) = result {
        assert!(
            metric.value >= 0.0,
            "RMS must be non-negative, got {}",
            metric.value
        );
    }
}

/// Prove: compute_rms passed is consistent with value >= threshold.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
fn compute_rms_passed_consistency() {
    let threshold: f64 = kani::any();
    kani::assume(threshold.is_finite() && threshold >= 0.0 && threshold <= 1.0);

    let result = compute_rms(&[0.5, 0.5], threshold);
    if let Ok(metric) = result {
        assert_eq!(
            metric.passed,
            metric.value >= threshold,
            "passed must equal (value >= threshold)"
        );
    }
}
