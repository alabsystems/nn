// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared env-var gating for Kokoro production-asset tests.
//!
//! Local development can skip production-weight tests when the large external
//! assets are absent. In CI, missing assets must fail loudly instead of
//! returning early and creating false confidence.

use std::path::Path;

fn running_in_ci() -> bool {
    ["CI", "BUILDKITE", "GITHUB_ACTIONS"]
        .iter()
        .any(|name| std::env::var_os(name).is_some())
}

fn skip_or_fail(message: String) -> Option<String> {
    assert!(!running_in_ci(), "CI misconfiguration: {message}");
    eprintln!("SKIP: {message}");
    None
}

fn require_existing_env_path(var: &str, skip_reason: &str) -> Option<String> {
    let value = match std::env::var(var) {
        Ok(v) if !v.is_empty() => v,
        _ => return skip_or_fail(format!("{var} not set — {skip_reason}")),
    };
    if !Path::new(&value).exists() {
        return skip_or_fail(format!("{var} does not exist: {value} — {skip_reason}"));
    }
    Some(value)
}

pub(crate) fn require_kokoro_weights(skip_reason: &str) -> Option<String> {
    require_existing_env_path("KOKORO_WEIGHTS", skip_reason)
}

pub(crate) fn require_kokoro_reference(skip_reason: &str) -> Option<String> {
    require_existing_env_path("KOKORO_REFERENCE", skip_reason)
}

pub(crate) fn require_kokoro_weights_and_reference(skip_reason: &str) -> Option<(String, String)> {
    let weights = require_kokoro_weights(skip_reason)?;
    let reference = require_kokoro_reference(skip_reason)?;
    Some((weights, reference))
}
