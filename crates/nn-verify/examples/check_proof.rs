// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Independent certificate checker CLI for `.proof.json` files.
//!
//! Run with: `cargo run -p nn-verify --example check_proof -- <proof.json> [--weights <path>] [--source <path>]`
//!
//! Validates proof certificates without access to NY or the original
//! model. See `certificate_checker.rs` for the checks performed.
//!
//! Addresses #802 AC5: standalone CLI checker binary.

use std::path::PathBuf;
use std::process;

use nn_verify::check_bundle_file;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: check_proof <proof.json> [--weights <path>] [--source <path>]");
        process::exit(2);
    }

    let proof_path = PathBuf::from(&args[1]);
    let weight_path = find_arg(&args, "--weights");
    let source_path = find_arg(&args, "--source");

    println!("Checking {}", proof_path.display());
    if let Some(ref p) = weight_path {
        println!("  Weight file: {}", p.display());
    }
    if let Some(ref p) = source_path {
        println!("  Source file: {}", p.display());
    }
    println!();

    let results =
        match check_bundle_file(&proof_path, weight_path.as_deref(), source_path.as_deref()) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Error loading certificate bundle: {e}");
                process::exit(1);
            }
        };

    let mut all_valid = true;
    for result in &results {
        if result.is_valid() {
            println!("  {} ... OK", result.kernel_name);
        } else {
            all_valid = false;
            eprintln!("  {} ... FAIL", result.kernel_name);
            for issue in &result.issues {
                eprintln!("    - {issue}");
            }
        }
    }

    println!();
    let total = results.len();
    let passed = results.iter().filter(|r| r.is_valid()).count();
    let failed = total - passed;
    println!("{passed}/{total} certificates valid, {failed} failed");

    if !all_valid {
        process::exit(1);
    }
}

/// Find the value of a flag like `--weights <path>` in the arg list.
fn find_arg(args: &[String], flag: &str) -> Option<PathBuf> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
}
