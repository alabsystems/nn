// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `hip_runtime.rs` — HIP/ROCm runtime abstraction.

use super::*;

#[test]
fn test_runtime_not_available_on_macos() {
    // On macOS, HIP is never available.
    if cfg!(target_os = "macos") {
        assert!(!is_hip_available());
        assert!(hip_device_count().is_err());
    }
}

#[test]
fn test_runtime_init_fails_gracefully() {
    if is_hip_available() {
        return; // Skip — HIP is available, test the real runtime.
    }
    let result = HipRuntime::init(0);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), HipRuntimeError::NotAvailable));
}

#[test]
fn test_validate_launch_config_valid() {
    let cfg = LaunchConfig::for_elementwise(1024, 256);
    assert!(validate_launch_config(&cfg).is_ok());
}

#[test]
fn test_validate_launch_config_zero_block() {
    let cfg = LaunchConfig {
        grid: Dim3::d1(1),
        block: Dim3::d1(0),
        shared_mem_bytes: 0,
    };
    let err = validate_launch_config(&cfg).unwrap_err();
    assert!(matches!(err, HipRuntimeError::InvalidLaunchConfig { .. }));
}

#[test]
fn test_validate_launch_config_too_many_threads() {
    let cfg = LaunchConfig {
        grid: Dim3::d1(1),
        block: Dim3::new(32, 32, 2), // 2048 > 1024
        shared_mem_bytes: 0,
    };
    let err = validate_launch_config(&cfg).unwrap_err();
    match err {
        HipRuntimeError::InvalidLaunchConfig { reason } => {
            assert!(reason.contains("1024"), "got: {reason}");
        }
        other => panic!("expected InvalidLaunchConfig, got: {other}"),
    }
}

#[test]
fn test_validate_launch_config_zero_grid() {
    let cfg = LaunchConfig {
        grid: Dim3::d1(0),
        block: Dim3::d1(256),
        shared_mem_bytes: 0,
    };
    assert!(validate_launch_config(&cfg).is_err());
}
