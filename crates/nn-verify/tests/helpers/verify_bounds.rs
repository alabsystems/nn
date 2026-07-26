// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Core integration tests for kernel bounds verification (IBP).
//!
//! CROWN escalation + serde/tensor roundtrip: verify_bounds_escalation.rs
//! Multi-variable tests: verify_bounds_multi.rs
//! Validation/edge-case tests: verify_bounds_validation.rs
//! Soundness provenance tests (#189, #194): verify_bounds_soundness.rs

use nn_dsl::snake_scalar_bounds;
use nn_verify::{
    scalar_input_bounds, Bound, ScalarInputBounds, VerificationResult, VerifyConfig, VerifyRequest,
    VerifyStatus,
};

use super::common::snake_kernel;

#[test]
fn test_verify_snake_ibp() {
    let kernel = snake_kernel();
    let ib = scalar_input_bounds(-10.0, 10.0).expect("input bounds");
    let result = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&ib)
        .verify_bounds()
        .expect("verification should pass");

    assert_eq!(result.kernel_name, "snake");
    assert!(result.is_finite, "Snake output bounds should be finite");
    // Native SnakeLayer exploits monotonicity for exact IBP bounds:
    // snake(x, 1) = x + sin²(x), f'(x) = 1 + sin(2x) >= 0
    // f([-10, 10]) = [f(-10), f(10)] ≈ [-9.70, 10.30]
    assert!(
        result.output_lower >= -10.0,
        "Snake is monotone: lower bound >= input lower, got {}",
        result.output_lower
    );
    assert!(
        result.output_lower <= -9.0,
        "lower bound should be near snake(-10) ≈ -9.7, got {}",
        result.output_lower
    );
    assert!(
        result.output_upper >= 10.0,
        "upper bound should be >= 10: got {}",
        result.output_upper
    );
    assert!(
        result.output_upper <= 12.0,
        "Native SnakeLayer should give tight upper bound, got {}",
        result.output_upper
    );
}

#[test]
fn test_verify_snake_tight_bounds() {
    let kernel = snake_kernel();
    let ib = scalar_input_bounds(-1.0, 1.0).expect("input bounds");
    let result = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&ib)
        .verify_bounds()
        .expect("verification should pass");

    assert!(result.is_finite);
    assert!(
        result.output_lower >= -2.0,
        "lower bound too loose: {}",
        result.output_lower
    );
    assert!(
        result.output_upper <= 3.0,
        "upper bound too loose: {}",
        result.output_upper
    );
}

#[test]
fn test_verify_snake_high_alpha() {
    let kernel = snake_kernel();
    let ib = scalar_input_bounds(-10.0, 10.0).expect("input bounds");
    let result = VerifyRequest::new(&kernel)
        .constant_params(&[100.0])
        .input_bounds(&ib)
        .verify_bounds()
        .expect("verification should pass");

    assert!(result.is_finite);
    assert!(
        result.output_upper <= 12.0,
        "upper bound should be tight with high alpha: got {}",
        result.output_upper
    );
}

#[test]
fn test_verify_kernel_spec_snake_domain_verified() {
    let kernel = snake_kernel();
    let input_bounds = scalar_input_bounds(-10.0, 10.0).expect("input bounds");
    let (out_lower, out_upper) =
        snake_scalar_bounds(-10.0, 10.0, 0.01, 100.0).expect("finite bounds");
    let output_spec = vec![Bound::new(out_lower, out_upper)];
    let config = VerifyConfig::with_threshold(5.0).expect("valid threshold");

    for alpha in [0.01f32, 0.1, 1.0, 10.0, 100.0] {
        let spec_v = VerifyRequest::new(&kernel)
            .constant_params(&[alpha])
            .input_bounds(&input_bounds)
            .required_output_bounds(&output_spec)
            .config(config.clone())
            .verify_spec()
            .unwrap_or_else(|e| panic!("spec verification failed for alpha={alpha}: {e}"));

        assert!(
            matches!(spec_v.result, VerificationResult::Verified { .. }),
            "expected Verified for alpha={alpha}, got {:?}",
            spec_v.result
        );
        assert!(
            spec_v.crown_fallback_reason.is_none(),
            "CROWN should not have failed for alpha={alpha}"
        );
    }
}

#[test]
fn test_verify_kernel_spec_tight_property_unknown() {
    let kernel = snake_kernel();
    let input_bounds = scalar_input_bounds(-10.0, 10.0).expect("input bounds");
    let tight_spec = vec![Bound::new(-10.0, 10.0)];
    let config = VerifyConfig::with_threshold(0.0).expect("valid threshold");

    let spec_v = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&input_bounds)
        .required_output_bounds(&tight_spec)
        .config(config)
        .verify_spec()
        .expect("spec verification should return a result");
    assert!(
        matches!(spec_v.result, VerificationResult::Unknown { .. }),
        "tight spec should remain unknown, got {:?}",
        spec_v.result
    );
}

#[test]
fn test_verify_and_persist_snake() {
    let kernel = snake_kernel();
    let mut status = VerifyStatus::default();
    let ib = scalar_input_bounds(-10.0, 10.0).expect("input bounds");

    for alpha in &[0.1f32, 1.0, 10.0, 100.0] {
        let result = VerifyRequest::new(&kernel)
            .constant_params(&[*alpha])
            .input_bounds(&ib)
            .verify_bounds()
            .expect("verification should pass");

        assert!(
            result.is_finite,
            "Snake bounds should be finite for alpha={alpha}"
        );

        let key_name = format!("snake_alpha_{alpha}");
        let mut named_result = result.clone();
        named_result.kernel_name = key_name;
        status
            .record(
                &named_result,
                ScalarInputBounds::new(-10.0, 10.0).expect("valid test bounds"),
                &[*alpha],
                None,
            )
            .expect("record");
    }

    let canonical = VerifyRequest::new(&kernel)
        .constant_params(&[1.0])
        .input_bounds(&ib)
        .verify_bounds()
        .expect("verification should pass");
    status
        .record(
            &canonical,
            ScalarInputBounds::new(-10.0, 10.0).expect("valid test bounds"),
            &[1.0],
            None,
        )
        .expect("record");

    let path = std::env::temp_dir().join(format!(
        "nn_verify_bounds_test_{}_{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before UNIX epoch")
            .as_nanos()
    ));
    status.save(&path).expect("save status");

    let loaded = VerifyStatus::load(&path).expect("load status");
    assert_eq!(loaded.kernel_count(), 5);
    assert!(loaded.has_kernel("snake"));
    assert!(loaded.has_kernel("snake_alpha_1"));

    let _ = std::fs::remove_file(&path);
}
