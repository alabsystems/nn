// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for advisory file locking in VerifyStatus save/load_locked.

use super::*;
use crate::verify_input::ScalarInputBounds;

#[test]
fn test_save_creates_and_removes_lock_file() {
    let dir = std::env::temp_dir().join(format!("nn_test_lock_save_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let status_path = dir.join("test_status.json");
    let lock_path = dir.join(".test_status.json.lock");

    // Clean up from any previous test run.
    let _ = std::fs::remove_file(&status_path);
    let _ = std::fs::remove_file(&lock_path);

    let status = VerifyStatus::default();
    status.save(&status_path).expect("save should succeed");

    // Lock file should be cleaned up after save completes.
    assert!(
        !lock_path.exists(),
        "lock file should be removed after save"
    );
    // Status file should exist.
    assert!(status_path.exists(), "status file should be written");

    // Clean up.
    let _ = std::fs::remove_file(&status_path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_load_locked_roundtrip() {
    let dir = std::env::temp_dir().join(format!("nn_test_load_locked_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let status_path = dir.join("test_locked_status.json");
    let lock_path = dir.join(".test_locked_status.json.lock");

    // Clean up from any previous test run.
    let _ = std::fs::remove_file(&status_path);
    let _ = std::fs::remove_file(&lock_path);

    // Create initial status.
    let mut status = VerifyStatus::default();
    status
        .record_failure(
            "test_kernel",
            PropMethod::Ibp,
            ScalarInputBounds::new(-1.0, 1.0).expect("valid bounds"),
            &[],
        )
        .expect("record failure");
    status.save(&status_path).expect("initial save");

    // Load with lock, modify, save.
    {
        let mut locked = VerifyStatus::load_locked(&status_path).expect("load_locked");
        assert_eq!(locked.status.kernel_count(), 1);
        assert!(locked.status.has_kernel("test_kernel"));

        // Lock file should exist while guard is alive.
        assert!(lock_path.exists(), "lock file should exist during guard");

        // Add another kernel via record_failure.
        locked
            .status
            .record_failure(
                "test_kernel_2",
                PropMethod::Crown,
                ScalarInputBounds::new(-2.0, 2.0).expect("valid bounds"),
                &[0.5],
            )
            .expect("record failure 2");
        locked.save().expect("locked save");
    }

    // Lock file should be removed after guard drop.
    assert!(
        !lock_path.exists(),
        "lock file should be removed after guard drop"
    );

    // Verify the saved data.
    let reloaded = VerifyStatus::load(&status_path).expect("reload");
    assert_eq!(reloaded.kernel_count(), 2);
    assert!(reloaded.has_kernel("test_kernel"));
    assert!(reloaded.has_kernel("test_kernel_2"));

    // Clean up.
    let _ = std::fs::remove_file(&status_path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_save_load_roundtrip_with_locking() {
    let dir = std::env::temp_dir().join(format!("nn_test_lock_roundtrip_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let status_path = dir.join("test_roundtrip.json");
    let lock_path = dir.join(".test_roundtrip.json.lock");

    let _ = std::fs::remove_file(&status_path);
    let _ = std::fs::remove_file(&lock_path);

    // Save, then load — verifies locking doesn't break basic roundtrip.
    let mut status = VerifyStatus::default();
    status
        .record_failure(
            "roundtrip_kernel",
            PropMethod::Ibp,
            ScalarInputBounds::new(-5.0, 5.0).expect("valid bounds"),
            &[1.0, 2.0],
        )
        .expect("record");
    status.save(&status_path).expect("save");

    let loaded = VerifyStatus::load(&status_path).expect("load");
    assert_eq!(loaded.kernel_count(), 1);
    assert!(loaded.has_kernel("roundtrip_kernel"));

    // Clean up.
    let _ = std::fs::remove_file(&status_path);
    let _ = std::fs::remove_dir(&dir);
}
