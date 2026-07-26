// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `status_helpers.rs` — input metadata validation and helpers.

use super::*;
use crate::error::VerifyError;
use crate::status::ParamInputRecord;

// ---------------------------------------------------------------------------
// validate_input_metadata — valid inputs
// ---------------------------------------------------------------------------

#[test]
fn test_validate_single_variable_valid() {
    let inputs = [ParamInputRecord {
        param_index: 0,
        lower: -1.0,
        upper: 1.0,
    }];
    assert!(validate_input_metadata(&inputs, &[]).is_ok());
}

#[test]
fn test_validate_multiple_variables_valid() {
    let inputs = [
        ParamInputRecord {
            param_index: 0,
            lower: 0.0,
            upper: 10.0,
        },
        ParamInputRecord {
            param_index: 1,
            lower: -5.0,
            upper: 5.0,
        },
    ];
    assert!(validate_input_metadata(&inputs, &[1.0, 2.0]).is_ok());
}

#[test]
fn test_validate_point_bounds_valid() {
    let inputs = [ParamInputRecord {
        param_index: 0,
        lower: 3.0,
        upper: 3.0,
    }];
    assert!(validate_input_metadata(&inputs, &[]).is_ok());
}

#[test]
fn test_validate_empty_inputs_valid() {
    assert!(validate_input_metadata(&[], &[]).is_ok());
}

#[test]
fn test_validate_only_constants_valid() {
    assert!(validate_input_metadata(&[], &[1.0, -2.5, 0.0]).is_ok());
}

// ---------------------------------------------------------------------------
// validate_input_metadata — NaN rejection in variable inputs
// ---------------------------------------------------------------------------

#[test]
fn test_validate_nan_lower_rejected() {
    let inputs = [ParamInputRecord {
        param_index: 0,
        lower: f32::NAN,
        upper: 1.0,
    }];
    let err = validate_input_metadata(&inputs, &[]).unwrap_err();
    match err {
        VerifyError::NonFiniteInputMetadata { context } => {
            assert!(
                context.contains("lower"),
                "error should mention 'lower': {context}"
            );
            assert!(
                context.contains("[0]"),
                "error should mention index: {context}"
            );
        }
        other => panic!("expected NonFiniteInputMetadata, got: {other:?}"),
    }
}

#[test]
fn test_validate_nan_upper_rejected() {
    let inputs = [ParamInputRecord {
        param_index: 0,
        lower: -1.0,
        upper: f32::NAN,
    }];
    let err = validate_input_metadata(&inputs, &[]).unwrap_err();
    match err {
        VerifyError::NonFiniteInputMetadata { context } => {
            assert!(context.contains("upper"));
        }
        other => panic!("expected NonFiniteInputMetadata, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// validate_input_metadata — infinity rejection
// ---------------------------------------------------------------------------

#[test]
fn test_validate_positive_infinity_lower_rejected() {
    let inputs = [ParamInputRecord {
        param_index: 0,
        lower: f32::INFINITY,
        upper: f32::INFINITY,
    }];
    assert!(validate_input_metadata(&inputs, &[]).is_err());
}

#[test]
fn test_validate_negative_infinity_upper_rejected() {
    let inputs = [ParamInputRecord {
        param_index: 0,
        lower: f32::NEG_INFINITY,
        upper: 0.0,
    }];
    assert!(validate_input_metadata(&inputs, &[]).is_err());
}

// ---------------------------------------------------------------------------
// validate_input_metadata — inverted bounds rejection
// ---------------------------------------------------------------------------

#[test]
fn test_validate_inverted_bounds_rejected() {
    let inputs = [ParamInputRecord {
        param_index: 0,
        lower: 5.0,
        upper: -5.0,
    }];
    let err = validate_input_metadata(&inputs, &[]).unwrap_err();
    match err {
        VerifyError::NonFiniteInputMetadata { context } => {
            assert!(
                context.contains("lower"),
                "error should mention 'lower': {context}"
            );
            assert!(
                context.contains("upper"),
                "error should mention 'upper': {context}"
            );
        }
        other => panic!("expected NonFiniteInputMetadata, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// validate_input_metadata — NaN in constant params
// ---------------------------------------------------------------------------

#[test]
fn test_validate_nan_constant_rejected() {
    let err = validate_input_metadata(&[], &[1.0, f32::NAN]).unwrap_err();
    match err {
        VerifyError::NonFiniteInputMetadata { context } => {
            assert!(context.contains("constant_params[1]"));
        }
        other => panic!("expected NonFiniteInputMetadata, got: {other:?}"),
    }
}

#[test]
fn test_validate_infinity_constant_rejected() {
    let err = validate_input_metadata(&[], &[f32::INFINITY]).unwrap_err();
    match err {
        VerifyError::NonFiniteInputMetadata { context } => {
            assert!(context.contains("constant_params[0]"));
        }
        other => panic!("expected NonFiniteInputMetadata, got: {other:?}"),
    }
}

#[test]
fn test_validate_neg_infinity_constant_rejected() {
    let err = validate_input_metadata(&[], &[f32::NEG_INFINITY]).unwrap_err();
    assert!(matches!(err, VerifyError::NonFiniteInputMetadata { .. }));
}

// ---------------------------------------------------------------------------
// validate_input_metadata — error ordering (first error wins)
// ---------------------------------------------------------------------------

#[test]
fn test_validate_first_bad_variable_reported() {
    let inputs = [
        ParamInputRecord {
            param_index: 0,
            lower: 0.0,
            upper: 1.0,
        },
        ParamInputRecord {
            param_index: 1,
            lower: f32::NAN,
            upper: 2.0,
        },
    ];
    let err = validate_input_metadata(&inputs, &[]).unwrap_err();
    match err {
        VerifyError::NonFiniteInputMetadata { context } => {
            assert!(context.contains("[1]"), "should report index 1: {context}");
        }
        other => panic!("expected NonFiniteInputMetadata, got: {other:?}"),
    }
}

#[test]
fn test_validate_variable_errors_before_constant_errors() {
    // First variable input is bad, first constant is also bad.
    // Variable errors should be reported first.
    let inputs = [ParamInputRecord {
        param_index: 0,
        lower: f32::NAN,
        upper: 1.0,
    }];
    let err = validate_input_metadata(&inputs, &[f32::NAN]).unwrap_err();
    match err {
        VerifyError::NonFiniteInputMetadata { context } => {
            assert!(
                context.contains("variable_inputs"),
                "variable error should come first: {context}"
            );
        }
        other => panic!("expected NonFiniteInputMetadata, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// atomic_tmp_path
// ---------------------------------------------------------------------------

#[test]
fn test_atomic_tmp_path_ends_with_tmp() {
    let status_path = std::env::temp_dir().join("nn_verify_status.json");
    let tmp_path = atomic_tmp_path(&status_path);
    assert!(
        tmp_path.to_string_lossy().ends_with(".tmp"),
        "temp path should end with .tmp: {}",
        tmp_path.display()
    );
}

#[test]
fn test_atomic_tmp_path_in_same_directory() {
    let status_path = std::env::temp_dir().join("nn_verify_status.json");
    let tmp_path = atomic_tmp_path(&status_path);
    assert_eq!(tmp_path.parent(), Some(std::env::temp_dir().as_path()));
}

#[test]
fn test_atomic_tmp_path_unique_per_call() {
    let status_path = std::env::temp_dir().join("status.json");
    let path1 = atomic_tmp_path(&status_path);
    let path2 = atomic_tmp_path(&status_path);
    assert_ne!(
        path1, path2,
        "consecutive calls should produce different paths"
    );
}

// ---------------------------------------------------------------------------
// StatusFileLock — acquire, release, stale cleanup
// ---------------------------------------------------------------------------

/// Helper: create a unique temp directory for lock tests.
fn lock_test_dir(suffix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("nn_lock_test_{}_{}", std::process::id(), suffix));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create test dir");
    dir
}

#[test]
fn test_lock_acquire_creates_lock_file() {
    let dir = lock_test_dir("acquire");
    let status_path = dir.join("status.json");
    let lock_path = dir.join(".status.json.lock");

    let guard = StatusFileLock::acquire(&status_path).expect("acquire should succeed");
    assert!(lock_path.exists(), "lock file should exist after acquire");
    drop(guard);
    assert!(
        !lock_path.exists(),
        "lock file should be removed after drop"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_lock_held_prevents_second_acquire_file_exists() {
    let dir = lock_test_dir("held");
    let status_path = dir.join("status.json");
    let lock_path = dir.join(".status.json.lock");

    let _guard = StatusFileLock::acquire(&status_path).expect("first acquire");
    assert!(lock_path.exists(), "lock file should exist while held");

    // We can't easily test the full retry loop (LOCK_MAX_RETRIES×100ms), but we
    // can verify the lock file exists and would block create_new.
    let result = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&lock_path);
    assert!(result.is_err(), "create_new should fail when lock is held");

    drop(_guard);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_lock_drop_is_idempotent() {
    let dir = lock_test_dir("idempotent");
    let status_path = dir.join("status.json");
    let lock_path = dir.join(".status.json.lock");

    let guard = StatusFileLock::acquire(&status_path).expect("acquire");
    assert!(lock_path.exists());

    // Manually remove the lock file before drop (simulating external cleanup).
    std::fs::remove_file(&lock_path).expect("manual remove");
    // Drop should not panic even though the file is already gone.
    drop(guard);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_lock_reacquire_after_release() {
    let dir = lock_test_dir("reacquire");
    let status_path = dir.join("status.json");

    // First acquire + release.
    let guard = StatusFileLock::acquire(&status_path).expect("first acquire");
    drop(guard);

    // Second acquire should succeed immediately.
    let guard2 = StatusFileLock::acquire(&status_path).expect("second acquire");
    drop(guard2);
    let _ = std::fs::remove_dir_all(&dir);
}

/// StatusFileLock Drop must remove the lock file even during stack unwinding.
///
/// Rust guarantees Drop runs during unwind unless the destructor itself
/// panics (double-panic = abort). This test verifies the RAII cleanup
/// contract holds through panic propagation.
///
/// Part of #3020 (certificate pipeline verification), memory_verification phase.
#[test]
fn test_lock_drop_during_panic_unwind() {
    let dir = lock_test_dir("panic_unwind");
    let status_path = dir.join("status.json");
    let lock_path = dir.join(".status.json.lock");

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = StatusFileLock::acquire(&status_path).expect("should acquire lock");
        assert!(lock_path.exists(), "lock should exist before panic");
        panic!("intentional panic to test RAII cleanup");
    }));

    assert!(result.is_err(), "should have caught panic");
    assert!(
        !lock_path.exists(),
        "lock file must be removed after panic unwind: {}",
        lock_path.display()
    );
    let _ = std::fs::remove_dir_all(&dir);
}
