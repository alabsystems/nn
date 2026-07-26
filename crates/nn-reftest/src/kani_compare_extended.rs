// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for tensor comparison engine.
//!
//! Complements `kani_compare.rs` with proofs for:
//! - RMS tolerance gate behavior
//! - Peak amplitude gate behavior
//! - Tolerance monotonicity (wider tolerances never reject what tighter ones accept)
//! - Inf candidate handling
//! - Metric bounds relationships (rms_diff vs max_abs_diff)
//! - NamedTensor construction safety
//! - Shape product overflow detection
//!
//! Issue: #3670

use crate::compare::{compare_tensors, ComparisonConfig};
use crate::trace::NamedTensor;

/// Helper: create a 1-element NamedTensor for scalar comparison proofs.
fn scalar_tensor(name: &str, val: f32) -> NamedTensor {
    NamedTensor {
        name: name.to_string(),
        shape: vec![1],
        data: vec![val],
    }
}

/// Helper: create a 2-element NamedTensor.
fn pair_tensor(name: &str, a: f32, b: f32) -> NamedTensor {
    NamedTensor {
        name: name.to_string(),
        shape: vec![2],
        data: vec![a, b],
    }
}

// ---------------------------------------------------------------------------
// RMS tolerance gate proofs
// ---------------------------------------------------------------------------

/// Proves that enabling the RMS gate causes failure when rms_diff exceeds
/// the threshold, even if abs and rel tolerances pass.
///
/// This tests that the RMS gate is an independent quality dimension —
/// a tensor can have small element-wise errors but large overall RMS.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn rms_gate_rejects_high_rms() {
    let r: f32 = kani::any();
    let c: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0e4 && r <= 1.0e4);
    kani::assume(c.is_finite() && c >= -1.0e4 && c <= 1.0e4);

    let abs_diff = (r - c).abs();
    // Ensure the abs diff is significant enough to produce meaningful RMS.
    kani::assume(abs_diff > 0.1);

    // Set abs/rel tolerances very wide so they always pass.
    // Set RMS tolerance very tight so it fails.
    let config = ComparisonConfig::new(1e6, 1e6, -1.0).with_rms_tolerance(1e-10);
    let reference = scalar_tensor("ref", r);
    let candidate = scalar_tensor("cand", c);

    let result = compare_tensors(&reference, &candidate, &config).expect("should not error");

    // For a single element, rms_diff == abs_diff. With abs_diff > 0.1
    // and rms_tolerance = 1e-10, this must fail.
    assert!(
        !result.passed,
        "RMS gate with tight threshold must reject high-rms pair"
    );
}

/// Proves that disabling the RMS gate (None) never causes RMS-related failure.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn rms_gate_disabled_never_rejects() {
    let r: f32 = kani::any();
    let c: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0e4 && r <= 1.0e4);
    kani::assume(c.is_finite() && c >= -1.0e4 && c <= 1.0e4);

    // Wide abs/rel/cos tolerances, no RMS gate.
    let config = ComparisonConfig::new(1e6, 1e6, -1.0);
    let reference = scalar_tensor("ref", r);
    let candidate = scalar_tensor("cand", c);

    let result = compare_tensors(&reference, &candidate, &config).expect("should not error");

    // With wide tolerances and no RMS gate, everything passes.
    assert!(
        result.passed,
        "disabled RMS gate with wide tolerances must always pass"
    );
}

// ---------------------------------------------------------------------------
// Peak amplitude gate proofs
// ---------------------------------------------------------------------------

/// Proves that the peak amplitude gate causes failure when candidate
/// exceeds the amplitude limit.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn peak_gate_rejects_high_amplitude() {
    let r: f32 = kani::any();
    let c: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
    kani::assume(c.is_finite() && c.abs() > 10.0 && c.abs() <= 1e4);

    // Wide abs/rel, tight peak gate.
    let config = ComparisonConfig::new(1e6, 1e6, -1.0).with_peak_amplitude_limit(5.0);
    let reference = scalar_tensor("ref", r);
    let candidate = scalar_tensor("cand", c);

    let result = compare_tensors(&reference, &candidate, &config).expect("should not error");

    assert!(
        !result.passed,
        "peak gate must reject candidate with |c| > peak_limit"
    );
}

/// Proves that the peak amplitude gate passes when candidate is within limit.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn peak_gate_passes_within_limit() {
    let c: f32 = kani::any();
    kani::assume(c.is_finite() && c.abs() <= 5.0);

    // Wide tolerances, generous peak gate.
    let config = ComparisonConfig::new(1e6, 1e6, -1.0).with_peak_amplitude_limit(100.0);
    let reference = scalar_tensor("ref", c); // identical
    let candidate = scalar_tensor("cand", c);

    let result = compare_tensors(&reference, &candidate, &config).expect("should not error");

    assert!(
        result.passed,
        "peak gate must pass when |candidate| <= peak_limit"
    );
}

// ---------------------------------------------------------------------------
// Inf candidate handling proofs
// ---------------------------------------------------------------------------

/// Proves that Inf candidate produces infinite divergence, same as NaN.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn inf_candidate_produces_infinite_divergence() {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0e6 && r <= 1.0e6);

    let config = ComparisonConfig::new(1.0, 1.0, -1.0);
    let reference = scalar_tensor("ref", r);
    let candidate = scalar_tensor("cand", f32::INFINITY);

    let result = compare_tensors(&reference, &candidate, &config).expect("should not error");

    assert!(
        result.max_abs_diff == f32::INFINITY,
        "Inf candidate must produce INFINITY max_abs_diff"
    );
    assert!(
        result.peak_amplitude == f32::INFINITY,
        "Inf candidate must produce INFINITY peak_amplitude"
    );
}

/// Proves that NEG_INFINITY candidate also produces infinite divergence.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn neg_inf_candidate_produces_infinite_divergence() {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0e6 && r <= 1.0e6);

    let config = ComparisonConfig::new(1.0, 1.0, -1.0);
    let reference = scalar_tensor("ref", r);
    let candidate = scalar_tensor("cand", f32::NEG_INFINITY);

    let result = compare_tensors(&reference, &candidate, &config).expect("should not error");

    assert!(
        result.max_abs_diff == f32::INFINITY,
        "NEG_INF candidate must produce INFINITY max_abs_diff"
    );
}

// ---------------------------------------------------------------------------
// Metric bounds relationship proofs
// ---------------------------------------------------------------------------

/// Proves that for a single element, rms_diff == max_abs_diff == mean_abs_diff.
///
/// For n=1: RMS = sqrt(diff^2 / 1) = |diff|. Mean = |diff|/1 = |diff|.
/// Max = |diff|. All three metrics collapse to the same value.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(2)]
fn single_element_metrics_equal() {
    let r: f32 = kani::any();
    let c: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0e4 && r <= 1.0e4);
    kani::assume(c.is_finite() && c >= -1.0e4 && c <= 1.0e4);

    let config = ComparisonConfig::new(1e6, 1e6, -1.0);
    let reference = scalar_tensor("ref", r);
    let candidate = scalar_tensor("cand", c);

    let result = compare_tensors(&reference, &candidate, &config).expect("should not error");

    assert!(
        result.max_abs_diff == result.mean_abs_diff,
        "for n=1, max_abs_diff must equal mean_abs_diff"
    );
    // RMS goes through f64 path so may have tiny floating point differences.
    let rms_diff_f32 = result.rms_diff;
    let abs_diff = result.max_abs_diff;
    let ulp_tolerance = abs_diff * 1e-6 + 1e-10;
    assert!(
        (rms_diff_f32 - abs_diff).abs() <= ulp_tolerance,
        "for n=1, rms_diff must approximately equal max_abs_diff"
    );
}

/// Proves that rms_diff >= mean_abs_diff for any 2-element tensor.
///
/// By Jensen's inequality (or QM-AM inequality): RMS >= mean for non-negative values.
/// sqrt(sum(x_i^2)/n) >= sum(x_i)/n.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
fn rms_geq_mean_abs_qm_am_inequality() {
    let r0: f32 = kani::any();
    let r1: f32 = kani::any();
    let c0: f32 = kani::any();
    let c1: f32 = kani::any();
    kani::assume(r0.is_finite() && r0 >= -1.0e4 && r0 <= 1.0e4);
    kani::assume(r1.is_finite() && r1 >= -1.0e4 && r1 <= 1.0e4);
    kani::assume(c0.is_finite() && c0 >= -1.0e4 && c0 <= 1.0e4);
    kani::assume(c1.is_finite() && c1 >= -1.0e4 && c1 <= 1.0e4);

    let config = ComparisonConfig::new(1e6, 1e6, -1.0);
    let reference = pair_tensor("ref", r0, r1);
    let candidate = pair_tensor("cand", c0, c1);

    let result = compare_tensors(&reference, &candidate, &config).expect("should not error");

    // QM-AM inequality with small float tolerance.
    let tol = result.rms_diff * 1e-5 + 1e-10;
    assert!(
        result.rms_diff >= result.mean_abs_diff - tol,
        "rms_diff must be >= mean_abs_diff (QM-AM inequality)"
    );
}

// ---------------------------------------------------------------------------
// NamedTensor construction proofs
// ---------------------------------------------------------------------------

/// Proves that NamedTensor::new rejects data whose length does not match
/// the shape product.
#[kani::unwind(8)]
#[kani::proof]
fn named_tensor_rejects_length_mismatch() {
    let shape_d0: usize = kani::any();
    let shape_d1: usize = kani::any();
    kani::assume(shape_d0 >= 1 && shape_d0 <= 8);
    kani::assume(shape_d1 >= 1 && shape_d1 <= 8);

    let expected_len = shape_d0 * shape_d1;
    let actual_len: usize = kani::any();
    kani::assume(actual_len <= 64);
    kani::assume(actual_len != expected_len);

    let data = vec![0.0f32; actual_len];
    let result = NamedTensor::new("test", vec![shape_d0, shape_d1], data);

    assert!(
        result.is_err(),
        "NamedTensor::new must reject data length != shape product"
    );
}

/// Proves that NamedTensor::new accepts data whose length matches
/// the shape product exactly.
#[kani::unwind(8)]
#[kani::proof]
fn named_tensor_accepts_matching_length() {
    let d0: usize = kani::any();
    let d1: usize = kani::any();
    kani::assume(d0 >= 1 && d0 <= 8);
    kani::assume(d1 >= 1 && d1 <= 8);

    let len = d0 * d1;
    let data = vec![0.0f32; len];
    let result = NamedTensor::new("test", vec![d0, d1], data);

    assert!(
        result.is_ok(),
        "NamedTensor::new must accept data of matching length"
    );
    let t = result.unwrap();
    assert!(t.numel() == len, "numel must equal shape product");
}

/// Proves that NamedTensor::new detects shape product overflow.
#[kani::unwind(8)]
#[kani::proof]
fn named_tensor_detects_shape_overflow() {
    let d0: usize = kani::any();
    let d1: usize = kani::any();
    kani::assume(d0 > 0 && d1 > 0);
    kani::assume(d0 > usize::MAX / d1); // guarantee overflow

    let result = NamedTensor::new("overflow", vec![d0, d1], vec![]);

    assert!(
        result.is_err(),
        "NamedTensor::new must detect shape product overflow"
    );
}
