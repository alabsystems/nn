// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! #1692 F3: Infeasible bounds sentinel tests.
//!
//! Verifies that `OutputBoundsRecord::from_verification()` sets
//! `is_infeasible = true` when bounds are non-finite or inverted,
//! and that legacy JSON backward compatibility is preserved.

use super::*;

/// Infeasible bounds (+Inf, -Inf) from mark_infeasible_all() must set
/// `is_infeasible = true` so consumers don't misinterpret `(0.0, 0.0)`
/// as a verified tight bound.
#[test]
fn test_f3_infeasible_bounds_sets_is_infeasible_true() {
    let result = KernelVerification {
        kernel_name: "infeasible_kernel".to_string(),
        method: PropMethod::Ibp,
        output_lower: f32::INFINITY, // mark_infeasible pattern
        output_upper: f32::NEG_INFINITY,
        output_width: f32::NAN,
        is_finite: false,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    };

    let record = OutputBoundsRecord::from_verification(&result);
    assert_eq!(record.lower, 0.0, "infeasible lower sanitized to 0.0");
    assert_eq!(record.upper, 0.0, "infeasible upper sanitized to 0.0");
    assert!(
        record.is_infeasible,
        "infeasible bounds must set is_infeasible = true"
    );
}

/// Normal finite bounds must NOT set is_infeasible.
#[test]
fn test_f3_finite_bounds_is_not_infeasible() {
    let result = KernelVerification {
        kernel_name: "finite_kernel".to_string(),
        method: PropMethod::Ibp,
        output_lower: -5.0,
        output_upper: 5.0,
        output_width: 10.0,
        is_finite: true,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    };

    let record = OutputBoundsRecord::from_verification(&result);
    assert!(
        !record.is_infeasible,
        "finite bounds must not be infeasible"
    );
}

/// NaN bounds (both non-finite) must also be marked infeasible.
#[test]
fn test_f3_nan_bounds_sets_is_infeasible_true() {
    let result = KernelVerification {
        kernel_name: "nan_kernel".to_string(),
        method: PropMethod::Ibp,
        output_lower: f32::NAN,
        output_upper: f32::NAN,
        output_width: f32::NAN,
        is_finite: false,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    };

    let record = OutputBoundsRecord::from_verification(&result);
    assert!(
        record.is_infeasible,
        "NaN bounds (both non-finite) must be marked infeasible"
    );
}

/// Inverted finite bounds (lower > upper) must be marked infeasible.
#[test]
fn test_f3_inverted_finite_bounds_is_infeasible() {
    let result = KernelVerification {
        kernel_name: "inverted_kernel".to_string(),
        method: PropMethod::Ibp,
        output_lower: 100.0,
        output_upper: -100.0,
        output_width: 200.0,
        is_finite: true,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: None,
    };

    let record = OutputBoundsRecord::from_verification(&result);
    assert!(
        record.is_infeasible,
        "inverted finite bounds must be marked infeasible"
    );
}

/// Legacy JSON without is_infeasible field defaults to false.
#[test]
fn test_f3_legacy_json_defaults_is_infeasible_false() {
    let json = r#"{"lower": -5.0, "upper": 5.0}"#;
    let record: OutputBoundsRecord = serde_json::from_str(json).expect("deserialize legacy");
    assert!(
        !record.is_infeasible,
        "legacy JSON must default is_infeasible to false"
    );
}

/// is_infeasible survives serialization roundtrip.
#[test]
fn test_f3_is_infeasible_serialization_roundtrip() {
    let record = OutputBoundsRecord {
        lower: 0.0,
        upper: 0.0,
        tensor_lower: None,
        tensor_upper: None,
        shape: None,
        is_infeasible: true,
    };
    let json = serde_json::to_string(&record).expect("serialize");
    assert!(json.contains("\"is_infeasible\":true"));
    let deserialized: OutputBoundsRecord = serde_json::from_str(&json).expect("deserialize");
    assert!(deserialized.is_infeasible);
}
