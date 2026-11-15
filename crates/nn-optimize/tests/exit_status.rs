// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! The binary's exit contract: 0 on success, 1 with `error: <chain>` on
//! stderr on failure. Pinned here because `main` returns an `ExitCode`
//! instead of calling `process::exit(1)`; the observable status and message
//! must not change.

use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_nn-optimize"))
}

#[test]
fn failure_exits_one_with_error_chain_on_stderr() {
    let missing = std::env::temp_dir()
        .join("nn_optimize_exit_status_missing")
        .join("does-not-exist.json");
    let out = bin()
        .args(["report", "--input"])
        .arg(&missing)
        .output()
        .expect("spawn nn-optimize");
    assert_eq!(out.status.code(), Some(1), "status: {:?}", out.status);
    assert!(out.stdout.is_empty());
    let stderr = String::from_utf8(out.stderr).expect("utf-8 stderr");
    assert!(
        stderr.starts_with("error: failed to read "),
        "stderr: {stderr:?}"
    );
    // `{e:#}` keeps the source chain on one line.
    assert!(stderr.ends_with(")\n"), "stderr: {stderr:?}");
    assert_eq!(stderr.lines().count(), 1, "stderr: {stderr:?}");
}

#[test]
fn invalid_tier_exits_one() {
    let out = bin()
        .args(["--tier", "9"])
        .output()
        .expect("spawn nn-optimize");
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8(out.stderr).expect("utf-8 stderr");
    assert!(
        stderr.starts_with("error: invalid tier: 9."),
        "stderr: {stderr:?}"
    );
}

#[test]
fn success_exits_zero() {
    let out = bin().arg("--schema").output().expect("spawn nn-optimize");
    assert_eq!(out.status.code(), Some(0));
    let schema: serde_json::Value = serde_json::from_slice(&out.stdout).expect("schema is JSON");
    assert!(schema.is_object());
}
