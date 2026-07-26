// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `TensorError` and helper functions.

use std::error::Error;

use super::{BackendDomain, BackendErrorKind, TensorError};
use crate::DType;

#[test]
fn test_dtype_mismatch_uses_typed_dtype_fields() {
    let error = TensorError::dtype_mismatch(DType::F32, DType::BF16);

    assert_eq!(
        error.to_string(),
        "Data type mismatch: expected f32, got bf16"
    );
    match error {
        TensorError::DTypeMismatch {
            expected, actual, ..
        } => {
            assert_eq!(expected, DType::F32);
            assert_eq!(actual, DType::BF16);
        }
        other => panic!("unexpected variant: {other:?}"),
    }
}

#[test]
fn test_backend_failure_preserves_domain() {
    let error = TensorError::backend_failure(
        BackendDomain::Metal,
        BackendErrorKind::Other,
        "kernel expects 3 parameters but got 5".into(),
    );

    assert_eq!(
        error.to_string(),
        "Metal error: kernel expects 3 parameters but got 5"
    );
    match error {
        TensorError::BackendFailure {
            domain, message, ..
        } => {
            assert_eq!(domain, BackendDomain::Metal);
            assert!(message.contains("3 parameters"));
        }
        other => panic!("unexpected variant: {other:?}"),
    }
}

#[test]
fn test_backend_failure_device_domain() {
    let error = TensorError::backend_failure(
        BackendDomain::Device,
        BackendErrorKind::Other,
        "Tensor not on CPU".into(),
    );

    assert_eq!(error.to_string(), "Device error: Tensor not on CPU");
}

#[test]
fn test_dimension_out_of_range_fields() {
    let error = TensorError::DimensionOutOfRange { dim: 5, rank: 3 };
    assert_eq!(error.to_string(), "Dimension 5 out of range for rank 3");
    match error {
        TensorError::DimensionOutOfRange { dim, rank } => {
            assert_eq!(dim, 5);
            assert_eq!(rank, 3);
        }
        other => panic!("unexpected variant: {other:?}"),
    }
}

#[test]
fn test_check_dim_valid() {
    assert!(super::check_dim(0, 3).is_ok());
    assert!(super::check_dim(2, 3).is_ok());
}

#[test]
fn test_check_dim_out_of_range() {
    let err = super::check_dim(3, 3).unwrap_err();
    assert!(matches!(
        err,
        TensorError::DimensionOutOfRange { dim: 3, rank: 3 }
    ));
    let err = super::check_dim(5, 2).unwrap_err();
    assert!(matches!(
        err,
        TensorError::DimensionOutOfRange { dim: 5, rank: 2 }
    ));
}

// D3: Backtrace capture tests

#[test]
fn test_backtrace_accessor_returns_some_for_dtype_mismatch() {
    let error = TensorError::dtype_mismatch(DType::F32, DType::BF16);
    assert!(error.backtrace().is_some());
}

#[test]
fn test_backtrace_accessor_returns_some_for_shape_mismatch() {
    let error = TensorError::shape_mismatch(vec![2, 3], vec![4, 5]);
    assert!(error.backtrace().is_some());
}

#[test]
fn test_backtrace_accessor_returns_some_for_backend_failure() {
    let error = TensorError::backend_failure(
        BackendDomain::Metal,
        BackendErrorKind::Other,
        "test error".into(),
    );
    assert!(error.backtrace().is_some());
}

#[test]
fn test_backtrace_accessor_returns_some_for_device_transfer() {
    let error = TensorError::device_transfer(crate::Device::Cpu, crate::Device::Cpu);
    assert!(error.backtrace().is_some());
}

#[test]
fn test_backtrace_accessor_returns_none_for_other_variants() {
    let error = TensorError::InvalidShape("test".into());
    assert!(error.backtrace().is_none());

    let error = TensorError::DimensionOutOfRange { dim: 0, rank: 0 };
    assert!(error.backtrace().is_none());

    let error = TensorError::Unsupported("test".into());
    assert!(error.backtrace().is_none());
}

#[test]
fn test_backtrace_display_unchanged() {
    // Backtrace must NOT appear in the Display output (error message).
    let error = TensorError::dtype_mismatch(DType::F32, DType::BF16);
    let msg = error.to_string();
    assert_eq!(msg, "Data type mismatch: expected f32, got bf16");
    // No "Backtrace" or stack frame text in the message.
    assert!(!msg.contains("Backtrace"));
    assert!(!msg.contains("backtrace"));
}

#[test]
fn test_backend_error_kind_accessor() {
    let error = TensorError::backend_failure(
        BackendDomain::Metal,
        BackendErrorKind::OutOfMemory,
        "buffer alloc failed".into(),
    );
    assert_eq!(
        error.backend_error_kind(),
        Some(BackendErrorKind::OutOfMemory)
    );

    let error = TensorError::backend_failure(
        BackendDomain::Metal,
        BackendErrorKind::KernelCompile,
        "MSL compile failed".into(),
    );
    assert_eq!(
        error.backend_error_kind(),
        Some(BackendErrorKind::KernelCompile)
    );

    // Non-BackendFailure variants return None.
    let error = TensorError::Unsupported("test".into());
    assert_eq!(error.backend_error_kind(), None);
}

// Source chain tests for BackendFailure (#2471 Category 5)

/// Helper error type for testing source chain preservation.
#[derive(Debug, thiserror::Error)]
#[error("test backend error: {msg}")]
struct TestBackendError {
    msg: String,
}

#[test]
fn test_backend_failure_no_source_returns_none() {
    let error = TensorError::backend_failure(
        BackendDomain::Metal,
        BackendErrorKind::Other,
        "no source".into(),
    );
    assert!(error.source().is_none());
}

#[test]
fn test_backend_failure_with_source_returns_some() {
    let inner = TestBackendError {
        msg: "buffer alloc failed".into(),
    };
    let error = TensorError::backend_failure_with_source(
        BackendDomain::Metal,
        BackendErrorKind::OutOfMemory,
        inner.to_string(),
        inner,
    );
    assert!(error.source().is_some());
}

#[test]
fn test_backend_failure_source_downcast() {
    let inner = TestBackendError {
        msg: "kernel compile error".into(),
    };
    let error = TensorError::backend_failure_with_source(
        BackendDomain::Metal,
        BackendErrorKind::KernelCompile,
        inner.to_string(),
        inner,
    );
    let source = error.source().expect("source should be present");
    let downcast = source
        .downcast_ref::<TestBackendError>()
        .expect("should downcast to TestBackendError");
    assert_eq!(downcast.msg, "kernel compile error");
}

#[test]
fn test_backend_failure_with_source_display_unchanged() {
    let inner = TestBackendError { msg: "oom".into() };
    let error = TensorError::backend_failure_with_source(
        BackendDomain::Metal,
        BackendErrorKind::OutOfMemory,
        "test backend error: oom".into(),
        inner,
    );
    // Display uses the message field, not the source.
    assert_eq!(error.to_string(), "Metal error: test backend error: oom");
}

#[test]
fn test_backend_failure_with_source_preserves_kind_and_domain() {
    let inner = TestBackendError {
        msg: "dispatch failed".into(),
    };
    let error = TensorError::backend_failure_with_source(
        BackendDomain::Cuda,
        BackendErrorKind::DispatchFailed,
        inner.to_string(),
        inner,
    );
    assert_eq!(
        error.backend_error_kind(),
        Some(BackendErrorKind::DispatchFailed)
    );
    match &error {
        TensorError::BackendFailure { domain, .. } => {
            assert_eq!(*domain, BackendDomain::Cuda);
        }
        other => panic!("unexpected variant: {other:?}"),
    }
}
