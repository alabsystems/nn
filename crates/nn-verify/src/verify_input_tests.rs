// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `verify_input.rs` — input bounds construction and validation.

use super::*;
use crate::graph::ParamBinding;
use ny_api::Bound;

// ---------------------------------------------------------------------------
// ScalarInputBounds::new
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_input_bounds_valid() {
    let b = ScalarInputBounds::new(-1.0, 1.0).unwrap();
    assert_eq!(b.lower(), -1.0);
    assert_eq!(b.upper(), 1.0);
}

#[test]
fn test_scalar_input_bounds_equal_lower_upper() {
    // Point bounds (lower == upper) should be valid.
    let b = ScalarInputBounds::new(3.0, 3.0).unwrap();
    assert_eq!(b.lower(), 3.0);
    assert_eq!(b.upper(), 3.0);
}

#[test]
fn test_scalar_input_bounds_zero() {
    let b = ScalarInputBounds::new(0.0, 0.0).unwrap();
    assert_eq!(b.lower(), 0.0);
    assert_eq!(b.upper(), 0.0);
}

#[test]
fn test_scalar_input_bounds_boundary_f32_values() {
    // f32::MAX and f32::MIN are finite extremes — should be valid.
    let b = ScalarInputBounds::new(f32::MIN, f32::MAX).unwrap();
    assert_eq!(b.lower(), f32::MIN);
    assert_eq!(b.upper(), f32::MAX);
}

#[test]
fn test_scalar_input_bounds_rejects_nan_lower() {
    let err = ScalarInputBounds::new(f32::NAN, 1.0).unwrap_err();
    assert!(matches!(err, VerifyError::InvalidInputBounds { .. }));
}

#[test]
fn test_scalar_input_bounds_rejects_nan_upper() {
    let err = ScalarInputBounds::new(-1.0, f32::NAN).unwrap_err();
    assert!(matches!(err, VerifyError::InvalidInputBounds { .. }));
}

#[test]
fn test_scalar_input_bounds_rejects_infinity_lower() {
    let err = ScalarInputBounds::new(f32::NEG_INFINITY, 1.0).unwrap_err();
    assert!(matches!(err, VerifyError::InvalidInputBounds { .. }));
}

#[test]
fn test_scalar_input_bounds_rejects_infinity_upper() {
    let err = ScalarInputBounds::new(-1.0, f32::INFINITY).unwrap_err();
    assert!(matches!(err, VerifyError::InvalidInputBounds { .. }));
}

#[test]
fn test_scalar_input_bounds_rejects_inverted() {
    let err = ScalarInputBounds::new(5.0, -5.0).unwrap_err();
    assert!(matches!(err, VerifyError::InvalidInputBounds { .. }));
}

#[test]
fn test_scalar_input_bounds_rejects_both_nan() {
    // IEEE 754: NaN > NaN is false, but !NaN.is_finite() catches it.
    let err = ScalarInputBounds::new(f32::NAN, f32::NAN).unwrap_err();
    assert!(matches!(err, VerifyError::InvalidInputBounds { .. }));
}

// ---------------------------------------------------------------------------
// ScalarInputBounds::to_bounded_tensor
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_input_bounds_to_bounded_tensor() {
    let b = ScalarInputBounds::new(-2.0, 3.0).unwrap();
    let bt = b.to_bounded_tensor().unwrap();
    assert_eq!(bt.lower()[[0]], -2.0);
    assert_eq!(bt.upper()[[0]], 3.0);
    assert_eq!(bt.lower().shape(), &[1]);
}

// ---------------------------------------------------------------------------
// scalar_input_bounds (convenience wrapper)
// ---------------------------------------------------------------------------

#[test]
fn test_scalar_input_bounds_fn_valid() {
    let bt = scalar_input_bounds(-1.0, 1.0).unwrap();
    assert_eq!(bt.lower()[[0]], -1.0);
    assert_eq!(bt.upper()[[0]], 1.0);
}

#[test]
fn test_scalar_input_bounds_fn_rejects_nan() {
    assert!(scalar_input_bounds(f32::NAN, 1.0).is_err());
}

// ---------------------------------------------------------------------------
// multi_scalar_input_bounds
// ---------------------------------------------------------------------------

#[test]
fn test_multi_scalar_single_element() {
    let bt = multi_scalar_input_bounds(&[(-1.0, 1.0)]).unwrap();
    assert_eq!(bt.lower().shape(), &[1]);
    assert_eq!(bt.lower()[[0]], -1.0);
    assert_eq!(bt.upper()[[0]], 1.0);
}

#[test]
fn test_multi_scalar_multiple_elements() {
    let bt = multi_scalar_input_bounds(&[(-1.0, 1.0), (0.0, 10.0), (-5.0, 5.0)]).unwrap();
    assert_eq!(bt.lower().shape(), &[3]);
    assert_eq!(bt.lower()[[0]], -1.0);
    assert_eq!(bt.lower()[[1]], 0.0);
    assert_eq!(bt.lower()[[2]], -5.0);
    assert_eq!(bt.upper()[[0]], 1.0);
    assert_eq!(bt.upper()[[1]], 10.0);
    assert_eq!(bt.upper()[[2]], 5.0);
}

#[test]
fn test_multi_scalar_rejects_empty() {
    let err = multi_scalar_input_bounds(&[]).unwrap_err();
    assert!(matches!(
        err,
        VerifyError::InvalidInputBounds {
            lower: 0.0,
            upper: 0.0
        }
    ));
}

#[test]
fn test_multi_scalar_rejects_nan_in_any_pair() {
    // NaN in the second pair — should fail.
    let err = multi_scalar_input_bounds(&[(-1.0, 1.0), (f32::NAN, 2.0)]).unwrap_err();
    assert!(matches!(err, VerifyError::InvalidInputBounds { .. }));
}

#[test]
fn test_multi_scalar_rejects_infinity_in_any_pair() {
    let err = multi_scalar_input_bounds(&[(-1.0, 1.0), (0.0, f32::INFINITY)]).unwrap_err();
    assert!(matches!(err, VerifyError::InvalidInputBounds { .. }));
}

#[test]
fn test_multi_scalar_rejects_inverted_pair() {
    let err = multi_scalar_input_bounds(&[(-1.0, 1.0), (5.0, -5.0)]).unwrap_err();
    assert!(matches!(err, VerifyError::InvalidInputBounds { .. }));
}

#[test]
fn test_multi_scalar_point_bounds() {
    // All pairs with lower == upper — should be valid.
    let bt = multi_scalar_input_bounds(&[(1.0, 1.0), (2.0, 2.0)]).unwrap();
    assert_eq!(bt.lower()[[0]], 1.0);
    assert_eq!(bt.upper()[[0]], 1.0);
    assert_eq!(bt.lower()[[1]], 2.0);
    assert_eq!(bt.upper()[[1]], 2.0);
}

// ---------------------------------------------------------------------------
// count_variable_bindings
// ---------------------------------------------------------------------------

#[test]
fn test_count_variable_bindings_all_variable() {
    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    assert_eq!(count_variable_bindings(&bindings), 2);
}

#[test]
fn test_count_variable_bindings_all_constant() {
    let bindings = [ParamBinding::Constant(1.0), ParamBinding::Constant(2.0)];
    assert_eq!(count_variable_bindings(&bindings), 0);
}

#[test]
fn test_count_variable_bindings_mixed() {
    let bindings = [
        ParamBinding::Variable,
        ParamBinding::Constant(1.0),
        ParamBinding::Variable,
        ParamBinding::Constant(3.0),
    ];
    assert_eq!(count_variable_bindings(&bindings), 2);
}

#[test]
fn test_count_variable_bindings_empty() {
    let bindings: Vec<ParamBinding> = vec![];
    assert_eq!(count_variable_bindings(&bindings), 0);
}

// ---------------------------------------------------------------------------
// validate_variable_bounds
// ---------------------------------------------------------------------------

#[test]
fn test_validate_variable_bounds_matching_count() {
    let bindings = [
        ParamBinding::Variable,
        ParamBinding::Constant(1.0),
        ParamBinding::Variable,
    ];
    let bounds = [(0.0, 1.0), (-1.0, 1.0)]; // 2 bounds for 2 variables
    assert!(validate_variable_bounds(&bindings, &bounds).is_ok());
}

#[test]
fn test_validate_variable_bounds_count_mismatch() {
    let bindings = [ParamBinding::Variable, ParamBinding::Variable];
    let bounds = [(0.0, 1.0)]; // 1 bound for 2 variables
    let err = validate_variable_bounds(&bindings, &bounds).unwrap_err();
    assert!(matches!(
        err,
        VerifyError::VariableBoundsMismatch {
            variable_count: 2,
            bounds_count: 1
        }
    ));
}

#[test]
fn test_validate_variable_bounds_zero_variables() {
    let bindings = [ParamBinding::Constant(1.0), ParamBinding::Constant(2.0)];
    let bounds: [(f32, f32); 0] = [];
    let err = validate_variable_bounds(&bindings, &bounds).unwrap_err();
    assert!(matches!(err, VerifyError::NoVariableBindings));
}

#[test]
fn test_validate_variable_bounds_empty_bindings() {
    let bindings: Vec<ParamBinding> = vec![];
    let bounds: [(f32, f32); 0] = [];
    let err = validate_variable_bounds(&bindings, &bounds).unwrap_err();
    assert!(matches!(err, VerifyError::NoVariableBindings));
}

// ---------------------------------------------------------------------------
// verification_spec_from_tensors
// ---------------------------------------------------------------------------

#[test]
fn test_verification_spec_from_tensors_valid() {
    use ny_api::Bound;
    let bt = scalar_input_bounds(-1.0, 1.0).unwrap();
    let output_bounds = [Bound::try_new(-2.0, 2.0).unwrap()];
    let spec = verification_spec_from_tensors(&bt, &output_bounds).unwrap();
    assert_eq!(spec.input_shape(), Some(&vec![1usize][..]));
    // Verify input bounds are preserved.
    assert_eq!(spec.input_bounds().len(), 1);
    assert_eq!(spec.input_bounds()[0].lower(), -1.0);
    assert_eq!(spec.input_bounds()[0].upper(), 1.0);
    // Verify output bounds are preserved.
    assert_eq!(spec.output_bounds().len(), 1);
    assert_eq!(spec.output_bounds()[0].lower(), -2.0);
    assert_eq!(spec.output_bounds()[0].upper(), 2.0);
}

#[test]
fn test_verification_spec_from_tensors_multi_variable() {
    use ny_api::Bound;
    let bt = multi_scalar_input_bounds(&[(-1.0, 1.0), (0.0, 5.0), (-3.0, 3.0)]).unwrap();
    let output_bounds = [Bound::try_new(-10.0, 10.0).unwrap()];
    let spec = verification_spec_from_tensors(&bt, &output_bounds).unwrap();
    assert_eq!(spec.input_shape(), Some(&vec![3usize][..]));
    assert_eq!(spec.input_bounds().len(), 3);
    // Verify each input bound pair.
    assert_eq!(spec.input_bounds()[0].lower(), -1.0);
    assert_eq!(spec.input_bounds()[0].upper(), 1.0);
    assert_eq!(spec.input_bounds()[1].lower(), 0.0);
    assert_eq!(spec.input_bounds()[1].upper(), 5.0);
    assert_eq!(spec.input_bounds()[2].lower(), -3.0);
    assert_eq!(spec.input_bounds()[2].upper(), 3.0);
}

#[test]
fn test_verification_spec_from_tensors_multiple_output_bounds() {
    use ny_api::Bound;
    let bt = scalar_input_bounds(-1.0, 1.0).unwrap();
    let output_bounds = [
        Bound::try_new(-2.0, 2.0).unwrap(),
        Bound::try_new(0.0, 1.0).unwrap(),
    ];
    let spec = verification_spec_from_tensors(&bt, &output_bounds).unwrap();
    assert_eq!(spec.output_bounds().len(), 2);
    assert_eq!(spec.output_bounds()[0].lower(), -2.0);
    assert_eq!(spec.output_bounds()[1].lower(), 0.0);
}

#[test]
fn test_verification_spec_from_tensors_empty_output_bounds_rejected() {
    let bt = scalar_input_bounds(-1.0, 1.0).unwrap();
    let output_bounds: &[Bound] = &[];
    let result = verification_spec_from_tensors(&bt, output_bounds);
    assert!(
        result.is_err(),
        "empty output bounds should be rejected by VerificationSpec"
    );
}

#[test]
fn test_verification_spec_from_tensors_point_bounds() {
    use ny_api::Bound;
    let bt = scalar_input_bounds(3.0, 3.0).unwrap();
    let output_bounds = [Bound::try_new(3.0, 3.0).unwrap()];
    let spec = verification_spec_from_tensors(&bt, &output_bounds).unwrap();
    assert_eq!(spec.input_bounds()[0].lower(), 3.0);
    assert_eq!(spec.input_bounds()[0].upper(), 3.0);
}

// ---------------------------------------------------------------------------
// uniform_bounds
// ---------------------------------------------------------------------------

#[test]
fn test_uniform_bounds_basic() {
    let bt = uniform_bounds(&[1, 128], 1.0).unwrap();
    assert_eq!(bt.lower().shape(), &[1, 128]);
    assert_eq!(bt.upper().shape(), &[1, 128]);
    for &v in bt.lower().iter() {
        assert_eq!(v, -1.0);
    }
    for &v in bt.upper().iter() {
        assert_eq!(v, 1.0);
    }
}

#[test]
fn test_uniform_bounds_zero_range() {
    let bt = uniform_bounds(&[2, 3], 0.0).unwrap();
    for &v in bt.lower().iter() {
        assert_eq!(v, 0.0);
    }
    for &v in bt.upper().iter() {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn test_uniform_bounds_rejects_nan() {
    assert!(uniform_bounds(&[1], f32::NAN).is_err());
}

#[test]
fn test_uniform_bounds_rejects_infinity() {
    assert!(uniform_bounds(&[1], f32::INFINITY).is_err());
}

#[test]
fn test_uniform_bounds_rejects_negative_range() {
    assert!(uniform_bounds(&[1], -1.0).is_err());
}
