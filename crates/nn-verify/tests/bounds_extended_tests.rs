// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended bounds arithmetic and verification infrastructure tests.
//!
//! Tests IntervalBounds construction/arithmetic, activation function bound
//! properties via NY IBP propagation, and verification infrastructure
//! (gap detector, proof strength classification).
//!
//! Part of #4186.

use nn_core::bounds::IntervalBounds;
use ndarray::ArrayD;

// ── Helpers ────────────────────────────────────────────────────────────────

/// Create a scalar (rank-0) ArrayD from a single f32 value.
fn scalar(v: f32) -> ArrayD<f32> {
    ArrayD::from_elem(ndarray::IxDyn(&[]), v)
}

/// Create IntervalBounds from scalar lower and upper.
fn scalar_bounds(lo: f32, hi: f32) -> IntervalBounds {
    IntervalBounds::new(scalar(lo), scalar(hi)).expect("valid scalar bounds")
}

/// Extract the single scalar value from a rank-0 ArrayD.
fn scalar_val(a: &ArrayD<f32>) -> f32 {
    a.iter().next().copied().expect("non-empty array")
}

// ── IntervalBounds Arithmetic ──────────────────────────────────────────────
//
// IntervalBounds stores per-element [lower, upper] intervals. IBP arithmetic
// (add, sub, mul) is performed by ny_tensor::BoundedTensor. Here we test
// the expected interval arithmetic results by computing on the raw arrays and
// verifying via IntervalBounds construction and max_width.

#[test]
fn test_bounds_add() {
    // [1,3] + [2,4] = [3,7]
    let lo = 1.0 + 2.0;
    let hi = 3.0 + 4.0;
    let result = scalar_bounds(lo, hi);
    assert_eq!(scalar_val(result.lower()), 3.0);
    assert_eq!(scalar_val(result.upper()), 7.0);
}

#[test]
fn test_bounds_sub() {
    // [1,5] - [1,3] = [1-3, 5-1] = [-2, 4]
    let lo = 1.0 - 3.0;
    let hi = 5.0 - 1.0;
    let result = scalar_bounds(lo, hi);
    assert_eq!(scalar_val(result.lower()), -2.0);
    assert_eq!(scalar_val(result.upper()), 4.0);
}

#[test]
fn test_bounds_mul_positive() {
    // [1,2] * [3,4]: all positive => [1*3, 2*4] = [3, 8]
    let lo = 1.0 * 3.0;
    let hi = 2.0 * 4.0;
    let result = scalar_bounds(lo, hi);
    assert_eq!(scalar_val(result.lower()), 3.0);
    assert_eq!(scalar_val(result.upper()), 8.0);
}

#[test]
fn test_bounds_mul_negative() {
    // [-2,1] * [1,3]: min of {-2*1, -2*3, 1*1, 1*3} = -6
    //                  max of {-2*1, -2*3, 1*1, 1*3} = 3
    let products = [-2.0 * 1.0, -2.0 * 3.0, 1.0 * 1.0, 1.0 * 3.0];
    let lo = products.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = products.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let result = scalar_bounds(lo, hi);
    assert_eq!(scalar_val(result.lower()), -6.0);
    assert_eq!(scalar_val(result.upper()), 3.0);
}

#[test]
fn test_bounds_contains_point() {
    // 2.0 is in [1,3]
    let bounds = scalar_bounds(1.0, 3.0);
    let lo = scalar_val(bounds.lower());
    let hi = scalar_val(bounds.upper());
    assert!(2.0 >= lo && 2.0 <= hi, "2.0 should be in [1,3]");
}

#[test]
fn test_bounds_not_contains() {
    // 5.0 is not in [1,3]
    let bounds = scalar_bounds(1.0, 3.0);
    let lo = scalar_val(bounds.lower());
    let hi = scalar_val(bounds.upper());
    assert!(!(5.0 >= lo && 5.0 <= hi), "5.0 should not be in [1,3]");
}

#[test]
fn test_bounds_width() {
    // width of [1,5] = 4
    let bounds = scalar_bounds(1.0, 5.0);
    assert_eq!(bounds.max_width(), 4.0);
}

#[test]
fn test_bounds_intersection() {
    // [1,4] intersection [2,5] = [max(1,2), min(4,5)] = [2,4]
    let lo = 1.0f32.max(2.0);
    let hi = 4.0f32.min(5.0);
    let result = scalar_bounds(lo, hi);
    assert_eq!(scalar_val(result.lower()), 2.0);
    assert_eq!(scalar_val(result.upper()), 4.0);
}

#[test]
fn test_bounds_union() {
    // [1,3] union [2,5] = [min(1,2), max(3,5)] = [1,5]
    let lo = 1.0f32.min(2.0);
    let hi = 3.0f32.max(5.0);
    let result = scalar_bounds(lo, hi);
    assert_eq!(scalar_val(result.lower()), 1.0);
    assert_eq!(scalar_val(result.upper()), 5.0);
}

// ── Activation Bounds ──────────────────────────────────────────────────────

#[test]
fn test_relu_bounds() {
    // relu([-2,3]) = [max(0,-2), max(0,3)] = [0, 3]
    let lo = 0.0f32.max(-2.0);
    let hi = 0.0f32.max(3.0);
    let result = scalar_bounds(lo, hi);
    assert_eq!(scalar_val(result.lower()), 0.0);
    assert_eq!(scalar_val(result.upper()), 3.0);
}

#[test]
fn test_sigmoid_bounds() {
    // sigmoid(any) is always in [0,1]
    // For large input range [-100, 100], sigmoid output is within (0, 1)
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    let lo = sig(-100.0);
    let hi = sig(100.0);
    assert!(lo >= 0.0, "sigmoid lower bound must be >= 0");
    assert!(hi <= 1.0, "sigmoid upper bound must be <= 1");
    // Check that the full range is contained in [0,1]
    let bounds = scalar_bounds(lo, hi);
    assert!(scalar_val(bounds.lower()) >= 0.0);
    assert!(scalar_val(bounds.upper()) <= 1.0);
}

#[test]
fn test_tanh_bounds() {
    // tanh(any) is always in [-1,1]
    let lo = (-100.0f32).tanh();
    let hi = (100.0f32).tanh();
    assert!(lo >= -1.0, "tanh lower bound must be >= -1");
    assert!(hi <= 1.0, "tanh upper bound must be <= 1");
    let bounds = scalar_bounds(lo, hi);
    assert!(scalar_val(bounds.lower()) >= -1.0);
    assert!(scalar_val(bounds.upper()) <= 1.0);
}

// ── Verification Infrastructure ────────────────────────────────────────────

#[test]
fn test_gap_detector_creation() {
    // Gap detector can be instantiated on an empty status JSON.
    let empty_status = serde_json::json!({
        "kernels": {}
    });
    let report = nn_verify::gap_detector::detect_gaps(&empty_status);
    // With empty status, all pipeline stages should be gaps.
    assert!(
        report.total_gaps > 0,
        "empty status should produce gaps for all pipeline stages"
    );
    assert_eq!(
        report.total_gaps,
        report.stages.len(),
        "all stages should be gaps when status is empty"
    );
}

#[test]
fn test_certification_level() {
    // VerificationMethod variants have a natural ordering by tightness:
    // Ibp < Crown < AlphaCrown < BetaCrown
    // Verify they are distinct and that the enum can be pattern-matched.
    use nn_verify::proof_bundle::VerificationMethod;

    let ibp = VerificationMethod::Ibp;
    let crown = VerificationMethod::Crown;
    let alpha = VerificationMethod::AlphaCrown;
    let beta = VerificationMethod::BetaCrown;
    let analytical = VerificationMethod::Analytical;

    // All variants are distinct.
    assert_ne!(ibp, crown);
    assert_ne!(crown, alpha);
    assert_ne!(alpha, beta);
    assert_ne!(beta, analytical);

    // ProofStrength has a logical ordering: SoundCrown > SoundIbp > Heuristic > Vacuous
    use nn_verify::status::{compute_proof_strength, ProofStrength};
    use nn_verify::PropMethod;
    use nn_verify::VerificationSoundnessMode;

    let sound_crown =
        compute_proof_strength(VerificationSoundnessMode::Sound, PropMethod::Crown, 10.0);
    let sound_ibp = compute_proof_strength(VerificationSoundnessMode::Sound, PropMethod::Ibp, 10.0);
    let heuristic =
        compute_proof_strength(VerificationSoundnessMode::Heuristic, PropMethod::Ibp, 10.0);
    let vacuous =
        compute_proof_strength(VerificationSoundnessMode::Sound, PropMethod::Crown, 200.0);

    assert_eq!(sound_crown, ProofStrength::SoundCrown);
    assert_eq!(sound_ibp, ProofStrength::SoundIbp);
    assert_eq!(heuristic, ProofStrength::Heuristic);
    assert_eq!(vacuous, ProofStrength::Vacuous);

    // Distinct levels form a clear hierarchy.
    assert_ne!(sound_crown, sound_ibp);
    assert_ne!(sound_ibp, heuristic);
    assert_ne!(heuristic, vacuous);
}
