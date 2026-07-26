// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `error.rs` — error types and conversions.

use super::*;

// ---------------------------------------------------------------------------
// StructuralError Display formatting
// ---------------------------------------------------------------------------

#[test]
fn test_structural_error_fusion_param_display() {
    let err = StructuralError::FusionParam {
        context: "alpha mismatch".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("fusion parameter mismatch"));
    assert!(msg.contains("alpha mismatch"));
}

#[test]
fn test_structural_error_shape_constraint_display() {
    let err = StructuralError::ShapeConstraint {
        context: "axis out of range".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("shape/axis constraint violation"));
    assert!(msg.contains("axis out of range"));
}

#[test]
fn test_structural_error_shape_display() {
    let err = StructuralError::Shape {
        reason: "ndarray shape mismatch".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("array shape construction failed"));
}

#[test]
fn test_structural_error_missing_node_name_display() {
    let err = StructuralError::MissingNodeName { input_idx: 3 };
    let msg = format!("{err}");
    assert!(msg.contains("input index 3"));
    assert!(msg.contains("missing a node name"));
}

#[test]
fn test_structural_error_non_finite_bounds_display() {
    let err = StructuralError::NonFiniteBounds {
        lower: f32::NEG_INFINITY,
        upper: f32::INFINITY,
    };
    let msg = format!("{err}");
    assert!(msg.contains("non-finite diff bounds"));
}

#[test]
fn test_structural_error_bounds_conversion_display() {
    let err = StructuralError::BoundsConversion("shape mismatch".to_string());
    let msg = format!("{err}");
    assert!(msg.contains("bounds type conversion failed"));
    assert!(msg.contains("shape mismatch"));
}

// ---------------------------------------------------------------------------
// VerifyError Display formatting
// ---------------------------------------------------------------------------

#[test]
fn test_verify_error_unsupported_op_display() {
    let err = VerifyError::UnsupportedOp("BinOp Mod".to_string());
    let msg = format!("{err}");
    assert!(msg.contains("unsupported IR operation"));
    assert!(msg.contains("BinOp Mod"));
}

#[test]
fn test_verify_error_internal_translation_display() {
    let err = VerifyError::InternalTranslationError {
        context: "constant division by zero".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("internal graph translation error"));
    assert!(msg.contains("constant division by zero"));
}

#[test]
fn test_verify_error_param_count_mismatch_display() {
    let err = VerifyError::ParamCountMismatch {
        ir_count: 3,
        provided: 2,
    };
    let msg = format!("{err}");
    assert!(msg.contains("3 params"));
    assert!(msg.contains("2 values"));
}

#[test]
fn test_verify_error_invalid_input_bounds_display() {
    let err = VerifyError::InvalidInputBounds {
        lower: 5.0,
        upper: -5.0,
    };
    let msg = format!("{err}");
    assert!(msg.contains("5"));
    assert!(msg.contains("-5"));
}

#[test]
fn test_verify_error_non_finite_constant_display() {
    let err = VerifyError::NonFiniteConstant {
        value: f32::NAN,
        context: "exp overflow".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("non-finite value"));
    assert!(msg.contains("exp overflow"));
}

#[test]
fn test_verify_error_variable_bounds_mismatch_display() {
    let err = VerifyError::VariableBoundsMismatch {
        variable_count: 3,
        bounds_count: 1,
    };
    let msg = format!("{err}");
    assert!(msg.contains("3 Variable bindings"));
    assert!(msg.contains("1 bounds"));
}

#[test]
fn test_verify_error_no_variable_bindings_display() {
    let err = VerifyError::NoVariableBindings;
    let msg = format!("{err}");
    assert!(msg.contains("at least one Variable binding"));
}

#[test]
fn test_verify_error_invalid_threshold_display() {
    let err = VerifyError::InvalidThreshold { value: -1.0 };
    let msg = format!("{err}");
    assert!(msg.contains("invalid threshold"));
    assert!(msg.contains("-1"));
}

#[test]
fn test_verify_error_soundness_required_display() {
    let err = VerifyError::SoundnessRequired {
        kernel_name: "snake".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("soundness required"));
    assert!(msg.contains("snake"));
}

#[test]
fn test_verify_error_non_finite_input_metadata_display() {
    let err = VerifyError::NonFiniteInputMetadata {
        context: "alpha is NaN".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("non-finite value in input metadata"));
    assert!(msg.contains("alpha is NaN"));
}

#[test]
fn test_verify_error_invalid_input_display() {
    let err = VerifyError::InvalidInput("empty kernel".to_string());
    let msg = format!("{err}");
    assert!(msg.contains("invalid verification input"));
    assert!(msg.contains("empty kernel"));
}

// ---------------------------------------------------------------------------
// VerifyError::Structural wrapping
// ---------------------------------------------------------------------------

#[test]
fn test_structural_error_converts_to_verify_error() {
    let structural = StructuralError::Shape {
        reason: "test".to_string(),
    };
    let verify: VerifyError = structural.into();
    let msg = format!("{verify}");
    assert!(msg.contains("structural error"));
    assert!(msg.contains("test"));
}

// ---------------------------------------------------------------------------
// From<VerifyError> for TensorError
// ---------------------------------------------------------------------------

#[test]
fn test_verify_error_to_tensor_error_domain() {
    let verify_err = VerifyError::UnsupportedOp("test".to_string());
    let tensor_err: nn_core::TensorError = verify_err.into();
    match tensor_err {
        nn_core::TensorError::BackendFailure {
            domain, message, ..
        } => {
            assert_eq!(domain, nn_core::BackendDomain::Verification);
            assert!(message.contains("unsupported IR operation"));
            assert!(message.contains("test"));
        }
        other => panic!("expected BackendFailure, got: {other:?}"),
    }
}

#[test]
fn test_verify_error_to_tensor_error_preserves_message() {
    let verify_err = VerifyError::InvalidThreshold { value: -42.0 };
    let tensor_err: nn_core::TensorError = verify_err.into();
    match tensor_err {
        nn_core::TensorError::BackendFailure { message, .. } => {
            assert!(message.contains("-42"));
            assert!(message.contains("invalid threshold"));
        }
        other => panic!("expected BackendFailure, got: {other:?}"),
    }
}
