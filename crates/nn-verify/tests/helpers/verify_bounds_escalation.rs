// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CROWN escalation tests, serde roundtrip, and tensor bounds persistence.
//!
//! Split from verify_bounds.rs to stay under the 500-line file limit.
//! Core IBP tests: verify_bounds.rs
//! Multi-variable tests: verify_bounds_multi.rs
//! Validation/edge-case tests: verify_bounds_validation.rs
//! Soundness provenance tests: verify_bounds_soundness.rs

use nn_dsl::lower::Lowerer;
use nn_verify::{
    scalar_input_bounds, KernelVerification, PropMethod, ScalarInputBounds,
    VerificationSoundnessMode, VerifyConfig, VerifyRequest, VerifyStatus,
};

use super::common::{exp_kernel, snake_kernel};

// --- CROWN escalation tests ---

#[test]
fn test_crown_escalation_triggered_on_low_threshold() {
    let kernel = snake_kernel();
    let input_bounds = scalar_input_bounds(-10.0, 10.0).expect("bounds");
    let config = VerifyConfig::with_threshold(5.0).expect("valid threshold");

    let result = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&input_bounds)
        .config(config)
        .verify_bounds()
        .expect("verification should pass");

    assert!(result.is_finite, "Snake bounds should be finite");
    assert!(
        result.output_lower <= -9.0,
        "lower bound must be sound: got {}",
        result.output_lower,
    );
    assert!(
        result.output_upper >= 10.0,
        "upper bound must be sound: got {}",
        result.output_upper,
    );
}

#[test]
fn test_crown_not_triggered_on_high_threshold() {
    let kernel = snake_kernel();
    let input_bounds = scalar_input_bounds(-10.0, 10.0).expect("bounds");
    let config = VerifyConfig::with_threshold(1000.0).expect("valid threshold");

    let result = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&input_bounds)
        .config(config)
        .verify_bounds()
        .expect("verification should pass");

    assert_eq!(
        result.method,
        PropMethod::Ibp,
        "should stay IBP when threshold is above IBP width"
    );
    assert!(result.is_finite);
}

#[test]
fn test_crown_escalation_exp_kernel() {
    let kernel = exp_kernel();
    let input_bounds = scalar_input_bounds(-50.0, 50.0).expect("bounds");
    let config = VerifyConfig::with_threshold(1.0).expect("valid threshold");

    let result = VerifyRequest::new(&kernel)
        .constant_params(&[])
        .input_bounds(&input_bounds)
        .config(config)
        .verify_bounds()
        .expect("verification should pass");

    assert!(result.is_finite, "exp bounds should be finite");
    assert!(
        result.output_lower <= f32::exp(-50.0) + 1e-10,
        "lower must contain exp(-50): got {}",
        result.output_lower,
    );
}

#[test]
fn test_crown_vs_ibp_bounds_are_sound() {
    let kernel = snake_kernel();
    let input_bounds = scalar_input_bounds(-5.0, 5.0).expect("bounds");

    let ibp_result = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&input_bounds)
        .config(VerifyConfig::with_threshold(1e10).expect("valid threshold"))
        .verify_bounds()
        .expect("IBP pass");
    assert_eq!(ibp_result.method, PropMethod::Ibp);

    let crown_result = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&input_bounds)
        .config(VerifyConfig::with_threshold(0.0).expect("valid threshold"))
        .verify_bounds()
        .expect("CROWN pass");

    assert!(ibp_result.is_finite, "IBP bounds finite");
    assert!(crown_result.is_finite, "CROWN bounds finite");

    assert!(
        crown_result.output_lower >= ibp_result.output_lower - 1e-6,
        "CROWN lower {} should be >= IBP lower {}",
        crown_result.output_lower,
        ibp_result.output_lower,
    );
    assert!(
        crown_result.output_upper <= ibp_result.output_upper + 1e-6,
        "CROWN upper {} should be <= IBP upper {}",
        crown_result.output_upper,
        ibp_result.output_upper,
    );
}

#[test]
fn test_verify_config_default() {
    let config = VerifyConfig::default();
    assert_eq!(
        config.escalation_threshold(),
        1e6,
        "default threshold should be 1e6"
    );
}

#[test]
fn test_crown_escalation_persists_method_in_status() {
    let kernel = snake_kernel();
    let input_bounds = scalar_input_bounds(-10.0, 10.0).expect("bounds");
    let config = VerifyConfig::with_threshold(0.0).expect("valid threshold");

    let result = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&input_bounds)
        .config(config)
        .verify_bounds()
        .expect("verification");

    let mut status = VerifyStatus::default();
    status
        .record(
            &result,
            ScalarInputBounds::new(-10.0, 10.0).expect("valid test bounds"),
            &[1.0],
            None,
        )
        .expect("record");

    let entry = status.kernel("snake").expect("snake kernel recorded");
    assert_eq!(entry.method, result.method);
}

// --- Serde roundtrip tests ---

#[test]
fn test_kernel_verification_serde_roundtrip() {
    let kernel = snake_kernel();
    let ib = scalar_input_bounds(-10.0, 10.0).expect("input bounds");
    let result = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&ib)
        .verify_bounds()
        .expect("verification should pass");

    let json = serde_json::to_string(&result).expect("serialize");
    let deserialized: KernelVerification = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(deserialized.kernel_name, result.kernel_name);
    assert_eq!(deserialized.method, result.method);
    assert_eq!(deserialized.output_lower, result.output_lower);
    assert_eq!(deserialized.output_upper, result.output_upper);
    assert_eq!(deserialized.is_finite, result.is_finite);
}

#[test]
fn test_kernel_verification_legacy_deserialize_defaults_soundness_mode() {
    let legacy_json = r#"{
  "kernel_name": "snake",
  "method": "IBP",
  "output_lower": -10.0,
  "output_upper": 11.0,
  "output_width": 21.0,
  "is_finite": true
}"#;

    let parsed: KernelVerification = serde_json::from_str(legacy_json)
        .expect("legacy KernelVerification JSON should deserialize");
    // Legacy JSON without soundness_mode field defaults to Heuristic (fail-closed, #201).
    assert_eq!(parsed.soundness_mode, VerificationSoundnessMode::Heuristic);
}

#[test]
fn test_verify_kernel_bounds_tensor_output_roundtrip_through_status() {
    let src = "fn affine(x: f32) -> f32 { x + 1.0 }";
    let func: syn::ItemFn = syn::parse_str(src).expect("parse");
    let kernel = Lowerer::lower_fn(&func).expect("lower");

    let lower =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![-2.0f32, -1.0, 0.0, 1.0])
            .expect("lower shape");
    let upper =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0f32, 2.0, 3.0, 4.0])
            .expect("upper shape");
    let input_bounds = nn_verify::BoundedTensor::new(lower, upper).expect("input bounds");

    let result = VerifyRequest::new(&kernel)
        .constant_params(&[])
        .input_bounds(&input_bounds)
        .verify_bounds()
        .expect("tensor verification should pass");
    assert!(result.is_finite);
    assert_eq!(result.output_lower, -1.0);
    assert_eq!(result.output_upper, 5.0);

    let output_tensor = result
        .output_tensor
        .as_ref()
        .expect("tensor output should be preserved");
    assert_eq!(output_tensor.shape, vec![2, 2]);
    assert_eq!(output_tensor.lower, vec![-1.0, 0.0, 1.0, 2.0]);
    assert_eq!(output_tensor.upper, vec![2.0, 3.0, 4.0, 5.0]);

    let json = serde_json::to_string(&result).expect("serialize result");
    let deserialized: KernelVerification = serde_json::from_str(&json).expect("deserialize result");
    assert_eq!(
        deserialized.output_tensor, result.output_tensor,
        "kernel result JSON must preserve full tensor bounds"
    );

    let mut status = VerifyStatus::default();
    status
        .record(
            &result,
            ScalarInputBounds::new(-2.0, 4.0).expect("valid test bounds"),
            &[],
            None,
        )
        .expect("record");
    let path = std::env::temp_dir().join(format!(
        "nn_verify_tensor_bounds_{}_{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before UNIX epoch")
            .as_nanos()
    ));
    status.save(&path).expect("save status");
    let loaded = VerifyStatus::load(&path).expect("load status");
    let _ = std::fs::remove_file(&path);

    let persisted = &loaded
        .kernel("affine")
        .expect("affine kernel recorded")
        .output_bounds;
    assert_eq!(persisted.lower, -1.0);
    assert_eq!(persisted.upper, 5.0);
    assert_eq!(persisted.shape, Some(vec![2, 2]));
    assert_eq!(persisted.tensor_lower, Some(vec![-1.0, 0.0, 1.0, 2.0]));
    assert_eq!(persisted.tensor_upper, Some(vec![2.0, 3.0, 4.0, 5.0]));
}
