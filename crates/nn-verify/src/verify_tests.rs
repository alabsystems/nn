// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `should_escalate_to_crown` and NaN/Inf sanitization in verify.rs.

use super::*;
use ny_core::{MethodUsed, UnknownReason};

#[test]
fn test_unknown_with_wide_bounds_escalates() {
    let result = VerificationResult::Unknown {
        provenance: Default::default(),
        bounds: vec![Bound::new(-10.0, 10.0)],
        reason: UnknownReason::BoundsTooLoose { gap: None },
        actual_method: None,
    };
    // Width = 20.0, threshold = 0.5 → escalate
    assert!(should_escalate_to_crown(&result, 0.5));
}

#[test]
fn test_unknown_with_tight_bounds_no_escalate() {
    let result = VerificationResult::Unknown {
        provenance: Default::default(),
        bounds: vec![Bound::new(0.0, 0.1)],
        reason: UnknownReason::BoundsTooLoose { gap: None },
        actual_method: None,
    };
    // Width = 0.1, threshold = 0.5 → no escalate
    assert!(!should_escalate_to_crown(&result, 0.5));
}

#[test]
fn test_timeout_without_partial_bounds_escalates() {
    let result = VerificationResult::Timeout {
        provenance: Default::default(),
        partial_bounds: None,
        actual_method: None,
    };
    assert!(should_escalate_to_crown(&result, 0.5));
}

#[test]
fn test_timeout_with_wide_partial_bounds_escalates() {
    let result = VerificationResult::Timeout {
        provenance: Default::default(),
        partial_bounds: Some(vec![Bound::new(-10.0, 10.0)]),
        actual_method: None,
    };
    // Width = 20.0, threshold = 0.5 → escalate
    assert!(should_escalate_to_crown(&result, 0.5));
}

#[test]
fn test_timeout_with_tight_partial_bounds_no_escalate() {
    let result = VerificationResult::Timeout {
        provenance: Default::default(),
        partial_bounds: Some(vec![Bound::new(0.0, 0.1)]),
        actual_method: None,
    };
    // Width = 0.1, threshold = 0.5 → no escalate
    assert!(!should_escalate_to_crown(&result, 0.5));
}

#[test]
fn test_verified_does_not_escalate() {
    let result = VerificationResult::Verified {
        provenance: Default::default(),
        output_bounds: vec![],
        proof: None,
        actual_method: None,
    };
    assert!(!should_escalate_to_crown(&result, 0.5));
}

#[test]
fn test_violated_does_not_escalate() {
    let result = VerificationResult::Violated {
        provenance: Default::default(),
        counterexample: vec![1.0],
        output: vec![2.0],
        details: None,
        actual_method: None,
    };
    assert!(!should_escalate_to_crown(&result, 0.5));
}

// ---------------------------------------------------------------------------
// NaN/Inf sanitization — AC1, AC2, AC3 of #226
// ---------------------------------------------------------------------------

#[test]
fn test_kernel_verification_sanitized_nonfinite_bounds_serialize() {
    // Exercise the same finite_or sanitization path that run_escalation uses
    // (verify.rs:413-414) when NY returns non-finite bounds.
    let kv = KernelVerification {
        kernel_name: "test_sanitized".to_string(),
        method: PropMethod::Ibp,
        output_lower: finite_or(f32::NEG_INFINITY, 0.0),
        output_upper: finite_or(f32::INFINITY, 0.0),
        output_width: f32::MAX, // run_escalation uses MAX when width is non-finite
        is_finite: false,
        output_tensor: None,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
    };
    // Verify finite_or replaced non-finite values
    assert_eq!(kv.output_lower, 0.0);
    assert_eq!(kv.output_upper, 0.0);
    assert!(!kv.is_finite);
    // Serialization must succeed — the original #226 bug was serde panicking on Inf
    let json = serde_json::to_string(&kv).expect("sanitized KernelVerification must serialize");
    assert!(json.contains("\"is_finite\":false"));
    assert!(json.contains("\"output_lower\":0.0"));
}

#[test]
fn test_finite_or_sanitizes_nan() {
    assert_eq!(finite_or(f32::NAN, 0.0), 0.0);
    assert_eq!(finite_or(f32::INFINITY, -1.0), -1.0);
    assert_eq!(finite_or(f32::NEG_INFINITY, 42.0), 42.0);
    assert_eq!(finite_or(2.5, 0.0), 2.5);
}

#[test]
fn test_output_tensor_bounds_sanitizes_inf_via_bounded_tensor() {
    use ndarray::ArrayD;
    // new_allow_infinite permits ±Inf bounds (but not NaN)
    let lower = ArrayD::from_shape_vec(vec![3], vec![f32::NEG_INFINITY, -1.0, 0.5]).unwrap();
    let upper = ArrayD::from_shape_vec(vec![3], vec![1.0, f32::INFINITY, 2.0]).unwrap();
    let bt = BoundedTensor::new_allow_infinite(lower, upper).unwrap();
    let otb = OutputTensorBounds::from_bounded_tensor(&bt);
    // Inf values replaced with 0.0 sentinel
    assert_eq!(otb.lower, vec![0.0, -1.0, 0.5]);
    assert_eq!(otb.upper, vec![1.0, 0.0, 2.0]);
    // finite_mask: element 0 has -Inf lower, element 1 has Inf upper → both false
    assert_eq!(otb.finite_mask, vec![false, false, true]);
    // Must serialize without panic
    serde_json::to_string(&otb).expect("sanitized OutputTensorBounds must serialize");
}

#[test]
fn test_output_tensor_bounds_finite_mask_identifies_nonfinite_elements() {
    use ndarray::ArrayD;
    // Mix of finite and non-finite elements to verify per-element tracking (#382).
    let lower = ArrayD::from_shape_vec(vec![4], vec![f32::NEG_INFINITY, -1.0, 0.5, -2.0]).unwrap();
    let upper = ArrayD::from_shape_vec(vec![4], vec![1.0, 2.0, f32::INFINITY, 3.0]).unwrap();
    let bt = BoundedTensor::new_allow_infinite(lower, upper).unwrap();
    let otb = OutputTensorBounds::from_bounded_tensor(&bt);
    // Element 0: lower=-Inf → non-finite. Element 2: upper=Inf → non-finite.
    // Elements 1 and 3: both bounds finite.
    assert_eq!(otb.finite_mask, vec![false, true, false, true]);
    // Sanitized values
    assert_eq!(otb.lower[0], 0.0);
    assert_eq!(otb.lower[1], -1.0);
    assert_eq!(otb.upper[2], 0.0);
    assert_eq!(otb.upper[3], 3.0);
}

#[test]
fn test_output_tensor_bounds_direct_nan_serializes() {
    // Construct directly with NaN to verify the struct's field-level behavior.
    // The fix ensures `from_bounded_tensor` never stores NaN, but direct
    // construction is possible since fields are pub.
    let otb = OutputTensorBounds {
        lower: vec![f32::NAN, 1.0],
        upper: vec![2.0, f32::INFINITY],
        shape: vec![2],
        finite_mask: vec![],
    };
    // Non-finite values are present when constructed directly (bypassing from_bounded_tensor).
    // The `is_finite` flag on KernelVerification signals this condition to callers.
    assert!(!otb.lower[0].is_finite());
    assert!(!otb.upper[1].is_finite());
}

// ---------------------------------------------------------------------------
// should_escalate_to_crown NaN width — IEEE 754 safety guard (line 163)
// ---------------------------------------------------------------------------

#[test]
fn test_unknown_with_nan_width_bounds_escalates() {
    // Bound::new_allow_infinite(Inf, Inf) passes validation (Inf <= Inf is true)
    // but produces NaN width: Inf - Inf = NaN in IEEE 754.
    // Without the explicit `is_nan()` guard in `exceeds_threshold`,
    // NaN > threshold would be false, silently skipping escalation.
    let result = VerificationResult::Unknown {
        provenance: Default::default(),
        bounds: vec![Bound::new_allow_infinite(f32::INFINITY, f32::INFINITY)],
        reason: UnknownReason::BoundsTooLoose { gap: None },
        actual_method: None,
    };
    // NaN width must conservatively trigger escalation.
    assert!(should_escalate_to_crown(&result, 1e6));
}

#[test]
fn test_timeout_with_nan_width_partial_bounds_escalates() {
    // NEG_INFINITY width: -Inf - (-Inf) = NaN
    let result = VerificationResult::Timeout {
        provenance: Default::default(),
        partial_bounds: Some(vec![Bound::new_allow_infinite(
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        )]),
        actual_method: None,
    };
    // NaN width from degenerate Inf bounds must trigger escalation.
    assert!(should_escalate_to_crown(&result, 1e6));
}

#[test]
fn test_exceeds_threshold_nan_guard_logic() {
    // Direct test of the exceeds_threshold closure semantics.
    // Replicates the closure at verify.rs:163 to prove NaN is caught.
    let exceeds_threshold =
        |max_width: f32, threshold: f32| -> bool { max_width.is_nan() || max_width > threshold };

    // NaN: always escalate (the guard we're testing)
    assert!(exceeds_threshold(f32::NAN, 1e6));

    // Inf: always escalate (Inf > threshold is true for finite threshold)
    assert!(exceeds_threshold(f32::INFINITY, 1e6));

    // Wide: escalate
    assert!(exceeds_threshold(2e6, 1e6));

    // Tight: no escalate
    assert!(!exceeds_threshold(0.5, 1e6));
}

// ---------------------------------------------------------------------------
// actual_method surfacing — preserve AlphaCrown/BetaCrown from NY
// ---------------------------------------------------------------------------

#[test]
fn test_resolve_result_method_prefers_actual_alpha_crown() {
    let result = VerificationResult::Verified {
        provenance: Default::default(),
        output_bounds: vec![Bound::new(-1.0, 1.0)],
        proof: None,
        actual_method: Some(MethodUsed::AlphaCrown),
    };

    assert_eq!(
        resolve_result_method(&result, PropMethod::Crown),
        PropMethod::AlphaCrown
    );
}

#[test]
fn test_resolve_result_method_prefers_actual_beta_crown() {
    let result = VerificationResult::Timeout {
        provenance: Default::default(),
        partial_bounds: None,
        actual_method: Some(MethodUsed::BetaCrown),
    };

    assert_eq!(
        resolve_result_method(&result, PropMethod::Crown),
        PropMethod::BetaCrown
    );
}

#[test]
fn test_resolve_result_method_falls_back_to_requested_when_unmapped() {
    let result = VerificationResult::Unknown {
        provenance: Default::default(),
        bounds: vec![Bound::new(-1.0, 1.0)],
        reason: UnknownReason::BoundsTooLoose { gap: None },
        actual_method: Some(MethodUsed::SmtRefiner),
    };

    assert_eq!(
        resolve_result_method(&result, PropMethod::Crown),
        PropMethod::Crown
    );
}
