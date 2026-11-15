// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! The `nn` binary's exit contract: 0 on success, 1 with a single
//! `error: <chain>` line on stderr when a subcommand fails. Pinned here
//! because `main` returns an `ExitCode` instead of calling
//! `process::exit(1)`; the observable status and message must not change.

use std::process::Command;

fn nn() -> Command {
    Command::new(env!("CARGO_BIN_EXE_nn"))
}

#[test]
fn failing_subcommand_exits_one_with_error_line() {
    let dir = std::env::temp_dir().join("nn_cli_exit_status_missing");
    let plan = dir.join("plan.nnc");
    let out = nn()
        .arg("optimize")
        .arg("--graph")
        .arg(dir.join("graph.json"))
        .arg("--weights")
        .arg(dir.join("weights.safetensors"))
        .arg(&plan)
        .output()
        .expect("spawn nn");
    assert_eq!(out.status.code(), Some(1), "status: {:?}", out.status);
    assert!(out.stdout.is_empty());
    let stderr = String::from_utf8(out.stderr).expect("utf-8 stderr");
    assert_eq!(
        stderr,
        format!("error: plan file not found: {}\n", plan.display())
    );
}

#[cfg(target_os = "macos")]
#[test]
fn successful_subcommand_exits_zero() {
    let out = nn().args(["device", "--json"]).output().expect("spawn nn");
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&out.stdout).expect("device JSON");
    assert!(report.is_object());
}
