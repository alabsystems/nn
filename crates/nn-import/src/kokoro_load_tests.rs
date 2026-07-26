// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for load_kokoro().
//!
//! The full end-to-end test requires KOKORO_WEIGHTS env var pointing to
//! kokoro_v1_0.safetensors. Tests without weights validate error paths.
//!
//! Part of #2465.

use super::*;

#[test]
fn test_load_kokoro_missing_file() {
    let result = load_kokoro("/nonexistent/kokoro.safetensors");
    match result {
        Ok(_) => panic!("expected error for missing file"),
        Err(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("I/O error") || msg.contains("No such file"),
                "unexpected error: {msg}"
            );
        }
    }
}

#[test]
fn test_load_kokoro_invalid_safetensors() {
    // Create a temp file with invalid safetensors content.
    let dir = std::env::temp_dir().join(format!("nn_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bad.safetensors");
    std::fs::write(&path, b"not a safetensors file").unwrap();

    assert!(
        load_kokoro(&path).is_err(),
        "expected error for invalid safetensors"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// Full load test — only runs when KOKORO_WEIGHTS is set.
#[test]
fn test_load_kokoro_real_weights() {
    let weights = match std::env::var("KOKORO_WEIGHTS") {
        Ok(v) if !v.is_empty() && Path::new(&v).exists() => v,
        _ => {
            // KOKORO_WEIGHTS not set — skip.
            return;
        }
    };

    match load_kokoro(&weights) {
        Ok(_kokoro) => { /* load succeeded */ }
        Err(e) => panic!("load_kokoro failed: {e}"),
    }
}
