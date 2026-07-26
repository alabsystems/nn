// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::kernel_error::KernelError;
use crate::lower::LowerError;

// --- validate_finite_inputs ---

#[test]
fn test_validate_finite_inputs_accepts_finite() {
    let result = validate_finite_inputs(&[("x", 1.0), ("y", -3.5), ("z", 0.0)]);
    assert!(result.is_ok());
}

#[test]
fn test_validate_finite_inputs_rejects_nan_first() {
    let result = validate_finite_inputs(&[("x", f32::NAN), ("y", 1.0)]);
    assert!(matches!(
        result,
        Err(KernelError::NonFiniteInput { name: "x", .. })
    ));
}

#[test]
fn test_validate_finite_inputs_rejects_nan_middle() {
    let result = validate_finite_inputs(&[("a", 1.0), ("b", f32::NAN), ("c", 2.0)]);
    assert!(matches!(
        result,
        Err(KernelError::NonFiniteInput { name: "b", .. })
    ));
}

#[test]
fn test_validate_finite_inputs_rejects_pos_inf() {
    let result = validate_finite_inputs(&[("x", f32::INFINITY)]);
    assert!(matches!(
        result,
        Err(KernelError::NonFiniteInput { name: "x", .. })
    ));
}

#[test]
fn test_validate_finite_inputs_rejects_neg_inf() {
    let result = validate_finite_inputs(&[("x", f32::NEG_INFINITY)]);
    assert!(matches!(
        result,
        Err(KernelError::NonFiniteInput { name: "x", .. })
    ));
}

#[test]
fn test_validate_finite_inputs_empty_is_ok() {
    let result = validate_finite_inputs(&[]);
    assert!(result.is_ok());
}

// --- validate_finite_slice ---

#[test]
fn test_validate_finite_slice_accepts_finite() {
    let result = validate_finite_slice("data", &[1.0, -2.5, 0.0, 100.0]);
    assert!(result.is_ok());
}

#[test]
fn test_validate_finite_slice_rejects_nan_with_index() {
    let result = validate_finite_slice("data", &[1.0, 2.0, f32::NAN, 4.0]);
    match result {
        Err(KernelError::NonFiniteSliceElement { name, index, .. }) => {
            assert_eq!(name, "data");
            assert_eq!(index, 2);
        }
        other => panic!("expected NonFiniteSliceElement, got {other:?}"),
    }
}

#[test]
fn test_validate_finite_slice_rejects_inf_with_index() {
    let result = validate_finite_slice("weights", &[0.5, f32::INFINITY]);
    match result {
        Err(KernelError::NonFiniteSliceElement { name, index, .. }) => {
            assert_eq!(name, "weights");
            assert_eq!(index, 1);
        }
        other => panic!("expected NonFiniteSliceElement, got {other:?}"),
    }
}

#[test]
fn test_validate_finite_slice_empty_is_ok() {
    let result = validate_finite_slice("empty", &[]);
    assert!(result.is_ok());
}

// --- checked_scalar_output ---

#[test]
fn test_checked_scalar_output_accepts_finite() {
    let result = checked_scalar_output(42.0);
    assert_eq!(result.unwrap(), 42.0);
}

#[test]
fn test_checked_scalar_output_rejects_nan() {
    let result = checked_scalar_output(f32::NAN);
    assert!(matches!(result, Err(KernelError::NonFiniteOutput { .. })));
}

#[test]
fn test_checked_scalar_output_rejects_pos_inf() {
    let result = checked_scalar_output(f32::INFINITY);
    assert!(matches!(result, Err(KernelError::NonFiniteOutput { .. })));
}

#[test]
fn test_checked_scalar_output_rejects_neg_inf() {
    let result = checked_scalar_output(f32::NEG_INFINITY);
    assert!(matches!(result, Err(KernelError::NonFiniteOutput { .. })));
}

// --- checked_slice_output ---

#[test]
fn test_checked_slice_output_accepts_finite() {
    let result = checked_slice_output(&[1.0, -2.0, 0.0]);
    assert!(result.is_ok());
}

#[test]
fn test_checked_slice_output_rejects_nan_with_index() {
    let result = checked_slice_output(&[1.0, 2.0, f32::NAN]);
    match result {
        Err(KernelError::NonFiniteSliceOutput { index, .. }) => {
            assert_eq!(index, 2);
        }
        other => panic!("expected NonFiniteSliceOutput, got {other:?}"),
    }
}

#[test]
fn test_checked_slice_output_rejects_inf_with_index() {
    let result = checked_slice_output(&[f32::INFINITY, 2.0]);
    match result {
        Err(KernelError::NonFiniteSliceOutput { index, .. }) => {
            assert_eq!(index, 0);
        }
        other => panic!("expected NonFiniteSliceOutput, got {other:?}"),
    }
}

#[test]
fn test_checked_slice_output_empty_is_ok() {
    let result = checked_slice_output(&[]);
    assert!(result.is_ok());
}

// --- validate_bounds_pairs ---

#[test]
fn test_validate_bounds_pairs_accepts_valid() {
    let result = validate_bounds_pairs(&[(-1.0, 1.0), (0.0, 100.0)]);
    assert!(result.is_ok());
}

#[test]
fn test_validate_bounds_pairs_rejects_nan_lower() {
    let result = validate_bounds_pairs(&[(f32::NAN, 1.0)]);
    assert!(matches!(result, Err(KernelError::NonFiniteBound { .. })));
}

#[test]
fn test_validate_bounds_pairs_rejects_nan_upper() {
    let result = validate_bounds_pairs(&[(0.0, f32::NAN)]);
    assert!(matches!(result, Err(KernelError::NonFiniteBound { .. })));
}

#[test]
fn test_validate_bounds_pairs_rejects_inverted() {
    let result = validate_bounds_pairs(&[(5.0, -5.0)]);
    match result {
        Err(KernelError::InvertedBounds { lower, upper }) => {
            assert_eq!(lower, 5.0);
            assert_eq!(upper, -5.0);
        }
        other => panic!("expected InvertedBounds, got {other:?}"),
    }
}

#[test]
fn test_validate_bounds_pairs_accepts_equal() {
    let result = validate_bounds_pairs(&[(3.0, 3.0)]);
    assert!(result.is_ok());
}

// --- validate_bounds_output ---

#[test]
fn test_validate_bounds_output_accepts_finite() {
    let result = validate_bounds_output(-1.0, 1.0);
    assert_eq!(result.unwrap(), (-1.0, 1.0));
}

#[test]
fn test_validate_bounds_output_rejects_nan_lower() {
    let result = validate_bounds_output(f32::NAN, 1.0);
    assert!(matches!(result, Err(KernelError::NonFiniteBound { .. })));
}

#[test]
fn test_validate_bounds_output_rejects_inf_upper() {
    let result = validate_bounds_output(0.0, f32::INFINITY);
    assert!(matches!(result, Err(KernelError::NonFiniteBound { .. })));
}

#[test]
fn test_validate_bounds_output_accepts_inverted() {
    let result = validate_bounds_output(5.0, -5.0);
    assert_eq!(result.unwrap(), (5.0, -5.0));
}

// --- validate_eps ---

#[test]
fn test_validate_eps_accepts_valid() {
    let result = validate_eps(1e-5);
    assert!(result.is_ok());
}

#[test]
fn test_validate_eps_rejects_zero() {
    let result = validate_eps(0.0);
    assert!(matches!(result, Err(KernelError::InvalidEps { .. })));
}

#[test]
fn test_validate_eps_rejects_negative() {
    let result = validate_eps(-1e-5);
    assert!(matches!(result, Err(KernelError::InvalidEps { .. })));
}

#[test]
fn test_validate_eps_rejects_nan() {
    let result = validate_eps(f32::NAN);
    assert!(matches!(result, Err(KernelError::InvalidEps { .. })));
}

#[test]
fn test_validate_eps_rejects_infinity() {
    let result = validate_eps(f32::INFINITY);
    assert!(matches!(result, Err(KernelError::InvalidEps { .. })));
}

// --- affine_normalize_scalar ---

#[test]
fn test_affine_normalize_identity() {
    let result = affine_normalize_scalar(2.0, 0.0, 1.0, 1e-5, 1.0, 0.0).unwrap();
    assert!((result - 2.0).abs() < 1e-3, "expected ~2.0, got {result}");
}

#[test]
fn test_affine_normalize_known_values() {
    let result = affine_normalize_scalar(4.0, 2.0, 4.0, 1e-5, 3.0, 1.0).unwrap();
    assert!((result - 4.0).abs() < 1e-3, "expected ~4.0, got {result}");
}

#[test]
fn test_affine_normalize_with_beta_shift() {
    let result = affine_normalize_scalar(0.0, 0.0, 1.0, 1e-5, 1.0, 5.0).unwrap();
    assert!((result - 5.0).abs() < 1e-3, "expected ~5.0, got {result}");
}

#[test]
fn test_affine_normalize_rejects_nan_input() {
    let result = affine_normalize_scalar(f32::NAN, 0.0, 1.0, 1e-5, 1.0, 0.0);
    assert!(matches!(
        result,
        Err(KernelError::NonFiniteInput { name: "x", .. })
    ));
}

#[test]
fn test_affine_normalize_rejects_var_plus_eps_nonpositive() {
    let result = affine_normalize_scalar(1.0, 0.0, -1.0, 0.5, 1.0, 0.0);
    assert!(matches!(result, Err(KernelError::InvalidEps { .. })));
}

#[test]
fn test_affine_normalize_overflow_produces_non_finite_output() {
    let result = affine_normalize_scalar(f32::MAX, -f32::MAX, 1e-10, 1e-5, f32::MAX, 0.0);
    assert!(matches!(result, Err(KernelError::NonFiniteOutput { .. })));
}

// --- validate_nonzero_dims ---

#[test]
fn test_validate_nonzero_dims_accepts_nonzero() {
    let result = validate_nonzero_dims(&[("channels", 64), ("length", 256)]);
    assert!(result.is_ok());
}

#[test]
fn test_validate_nonzero_dims_rejects_zero_with_name() {
    let result = validate_nonzero_dims(&[("channels", 64), ("length", 0)]);
    match result {
        Err(KernelError::InvalidDimension { name, value }) => {
            assert_eq!(name, "length");
            assert_eq!(value, 0);
        }
        other => panic!("expected InvalidDimension, got {other:?}"),
    }
}

#[test]
fn test_validate_nonzero_dims_empty_is_ok() {
    let result = validate_nonzero_dims(&[]);
    assert!(result.is_ok());
}

// --- build_scalar_kernel ---

#[test]
fn test_build_scalar_kernel_valid_source() {
    let src = "fn add(x: f32, y: f32) -> f32 { x + y }";
    let result = build_scalar_kernel(src);
    assert!(result.is_ok());
    let kdef = result.unwrap();
    assert_eq!(kdef.name, "add");
    assert_eq!(kdef.params.len(), 2);
}

#[test]
fn test_build_scalar_kernel_invalid_source() {
    let src = "not valid rust";
    let result = build_scalar_kernel(src);
    assert!(matches!(result, Err(LowerError::ParseError(_))));
}
