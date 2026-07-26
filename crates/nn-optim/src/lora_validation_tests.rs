// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Validation regression tests for LoRA adapter (#1503).
//!
//! Tests that construction-time validation rejects invalid parameters:
//! rank=0, alpha=NaN/Inf, scaling f32 overflow.

use super::*;
use nn_core::test_utils::make_linear;

#[test]
fn test_lora_rank_zero_rejected() {
    let linear = make_linear(4, 3);
    let err = LoraLinear::from_linear(&linear, 0, 4.0).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("rank must be > 0"),
        "expected rank error, got: {msg}"
    );
}

#[test]
fn test_lora_alpha_nan_rejected() {
    let linear = make_linear(4, 3);
    let err = LoraLinear::from_linear(&linear, 4, f64::NAN).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("alpha must be finite"),
        "expected alpha error, got: {msg}"
    );
}

#[test]
fn test_lora_alpha_inf_rejected() {
    let linear = make_linear(4, 3);
    let err = LoraLinear::from_linear(&linear, 4, f64::INFINITY).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("alpha must be finite"),
        "expected alpha error, got: {msg}"
    );
}

#[test]
fn test_lora_alpha_neg_inf_rejected() {
    let linear = make_linear(4, 3);
    let err = LoraLinear::from_linear(&linear, 4, f64::NEG_INFINITY).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("alpha must be finite"),
        "expected alpha error, got: {msg}"
    );
}

#[test]
fn test_trainable_lora_rank_zero_rejected() {
    let linear = make_linear(4, 3);
    let err = TrainableLoraLinear::from_linear(&linear, 0, 4.0).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("rank must be > 0"),
        "expected rank error, got: {msg}"
    );
}

#[test]
fn test_trainable_lora_alpha_nan_rejected() {
    let linear = make_linear(4, 3);
    let err = TrainableLoraLinear::from_linear(&linear, 4, f64::NAN).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("alpha must be finite"),
        "expected alpha error, got: {msg}"
    );
}

#[test]
fn test_lora_alpha_zero_accepted() {
    // alpha=0 is valid (disables LoRA contribution), produces scaling=0 (finite)
    let linear = make_linear(4, 3);
    let lora = LoraLinear::from_linear(&linear, 4, 0.0).unwrap();
    assert_eq!(lora.scaling(), 0.0);
}

#[test]
fn test_lora_scaling_f32_overflow_rejected() {
    // Finite f64 alpha that produces scaling overflowing f32::MAX
    let linear = make_linear(4, 3);
    let result = LoraLinear::from_linear(&linear, 1, 1e39);
    assert!(result.is_err(), "scaling 1e39 overflows f32");
    let err = result.unwrap_err().to_string();
    assert!(err.contains("overflows f32"), "error message: {err}");
}

#[test]
fn test_trainable_lora_scaling_f32_overflow_rejected() {
    let linear = make_linear(4, 3);
    let result = TrainableLoraLinear::from_linear(&linear, 1, 1e39);
    assert!(result.is_err(), "scaling 1e39 overflows f32");
    let err = result.unwrap_err().to_string();
    assert!(err.contains("overflows f32"), "error message: {err}");
}

#[test]
fn test_trainable_lora_boundary_f32_overflow() {
    // f32::MAX ≈ 3.4028235e38. With rank=1, scaling = alpha/rank = alpha.
    // Values just below f32::MAX should be accepted; values above should be rejected.
    let linear = make_linear(4, 3);
    // 3.4e38 is within f32 range
    assert!(TrainableLoraLinear::from_linear(&linear, 1, 3.4e38).is_ok());
    // 3.5e38 exceeds f32::MAX
    assert!(TrainableLoraLinear::from_linear(&linear, 1, 3.5e38).is_err());
}

#[test]
fn test_lora_large_but_f32_finite_accepted() {
    // alpha that produces large but f32-representable scaling
    let linear = make_linear(4, 3);
    let result = LoraLinear::from_linear(&linear, 1, 1e30);
    assert!(result.is_ok(), "1e30 is within f32 range");
}
