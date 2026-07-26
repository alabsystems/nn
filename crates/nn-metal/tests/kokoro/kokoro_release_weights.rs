// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for [`CompiledKokoro::release_model_weights`] (#3079 D1b).
//!
//! Validates the weight release API contract:
//! - After release, `weights_released()` returns true.
//! - After release, `config()` still works.
//! - Cloned instances block release (`SharedOwnership` error).
//! - Release after clone dropped succeeds.
//! - Idempotent double release.

use super::kokoro_test_weights as kw;

/// After release, `weights_released()` returns true and `config()` still works.
#[test]
fn test_release_model_weights_basic() {
    let config = kw::mini_test_config();
    let (mut kokoro, _cache) = kw::build_kokoro_with_config(&config);

    assert!(!kokoro.weights_released(), "initially weights should exist");
    assert_eq!(
        kokoro.config().d_en,
        config.d_en,
        "config accessible before release"
    );

    kokoro
        .release_model_weights()
        .expect("release should succeed on sole owner");

    assert!(
        kokoro.weights_released(),
        "weights should be released after call"
    );
    assert_eq!(
        kokoro.config().d_en,
        config.d_en,
        "config must remain accessible after release"
    );
}

/// Double release is idempotent — setting model to None twice is fine.
#[test]
fn test_release_model_weights_idempotent() {
    let (mut kokoro, _cache) = kw::build_kokoro_mini();

    kokoro.release_model_weights().expect("first release");
    assert!(kokoro.weights_released());

    kokoro
        .release_model_weights()
        .expect("second release should be idempotent");
    assert!(kokoro.weights_released());
}

/// Release blocked when clone_dispatch instances exist (SharedOwnership error).
#[test]
fn test_release_blocked_by_clone_dispatch() {
    let (mut kokoro, _cache) = kw::build_kokoro_mini();
    let _clone = kokoro.clone_dispatch();

    let result = kokoro.release_model_weights();
    assert!(
        result.is_err(),
        "release should fail with outstanding clone"
    );

    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("shared") || msg.contains("sole ownership"),
        "error should mention shared ownership: {msg}"
    );
}

/// Release succeeds after all clones are dropped.
#[test]
fn test_release_after_clone_dropped() {
    let (mut kokoro, _cache) = kw::build_kokoro_mini();
    let clone = kokoro.clone_dispatch();

    assert!(kokoro.release_model_weights().is_err());

    drop(clone);

    kokoro
        .release_model_weights()
        .expect("release after clone dropped");
    assert!(kokoro.weights_released());
}

/// After release, GPU weight bytes are still available (compiled segments unaffected).
#[test]
fn test_gpu_weights_survive_release() {
    let (mut kokoro, _cache) = kw::build_kokoro_mini();

    let bytes_before = kokoro.gpu_weight_bytes();

    kokoro.release_model_weights().expect("release");

    let bytes_after = kokoro.gpu_weight_bytes();
    // GPU weight bytes should be unchanged — compiled segment buffers are independent.
    assert_eq!(
        bytes_before, bytes_after,
        "GPU weight bytes should survive weight release"
    );
}
