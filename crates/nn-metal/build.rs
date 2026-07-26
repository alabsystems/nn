// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Build script for nn-metal: compile `.metal` sources to `.metallib`.
//!
//! If `NN_METALLIB_DIR` is set or `target/nn-compiled/` contains `.metal`
//! files, compiles them to a single `.metallib` in `OUT_DIR`. The resulting
//! metallib bytes are **embedded into the binary** at compile time via
//! `include_bytes!(env!("NN_EMBEDDED_METALLIB"))` in `metallib_loader.rs` —
//! the proof-closed default is that shaders ship inside the binary, never
//! loaded from the filesystem at runtime.
//!
//! When no `.metal` files exist, an empty placeholder is embedded and all
//! shader compilation happens at runtime via `PipelineCache` (from MSL
//! sources that are themselves embedded string constants).

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let metal_dir = metal_source_dir();
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("invariant: OUT_DIR set by Cargo"));

    let metal_files = find_metal_files(&metal_dir);
    if metal_files.is_empty() {
        // No precompiled shaders — embed an empty placeholder so
        // `include_bytes!(env!("NN_EMBEDDED_METALLIB"))` still compiles.
        // Kernels compile at runtime via PipelineCache from embedded MSL.
        let placeholder = out_dir.join("embedded-empty.metallib");
        std::fs::write(&placeholder, [])
            .expect("invariant: OUT_DIR is writable during build scripts");
        emit_embedded_metallib_env(&placeholder);
        return;
    }

    // Compile each .metal → .air, then link all .air → single .metallib.
    let mut air_files = Vec::with_capacity(metal_files.len());
    for metal_path in &metal_files {
        let air_path = compile_metal_to_air(metal_path, &out_dir);
        air_files.push(air_path);

        // Re-run build if any .metal file changes.
        // NOTE: cargo:: directives require println — this is Cargo build script API.
        #[allow(clippy::print_stdout)]
        {
            println!("cargo::rerun-if-changed={}", metal_path.display());
        }
    }

    let metallib_path = out_dir.join("precompiled.metallib");
    link_air_to_metallib(&air_files, &metallib_path);

    // Export the metallib path. This is informational (and consumed only by
    // the explicit, double-opt-in runtime loading path); the default delivery
    // path is the compile-time embedding below.
    // NOTE: cargo:: directives require println — this is Cargo build script API.
    #[allow(clippy::print_stdout)]
    {
        println!(
            "cargo::rustc-env=NN_PRECOMPILED_METALLIB={}",
            metallib_path.display()
        );
    }

    // Embed the metallib bytes into the binary (proof-closed default).
    emit_embedded_metallib_env(&metallib_path);
}

/// Point `include_bytes!(env!("NN_EMBEDDED_METALLIB"))` at `path`.
///
/// `metallib_loader::embedded_metallib()` treats an empty file as "no
/// precompiled metallib".
fn emit_embedded_metallib_env(path: &Path) {
    // NOTE: cargo:: directives require println — this is Cargo build script API.
    #[allow(clippy::print_stdout)]
    {
        println!("cargo::rustc-env=NN_EMBEDDED_METALLIB={}", path.display());
    }
}

/// Determine the directory to search for `.metal` files.
///
/// Priority:
/// 1. `NN_METALLIB_DIR` env var (explicit override)
/// 2. `target/nn-compiled/` relative to workspace root
fn metal_source_dir() -> PathBuf {
    // NOTE: cargo:: directives require println — this is Cargo build script API.
    #[allow(clippy::print_stdout)]
    {
        println!("cargo::rerun-if-env-changed=NN_METALLIB_DIR");
    }

    if let Ok(dir) = std::env::var("NN_METALLIB_DIR") {
        return PathBuf::from(dir);
    }

    // Default: look for nn-compiled/ in the workspace target dir.
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("invariant: CARGO_MANIFEST_DIR set by Cargo"),
    );
    manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map(|workspace| workspace.join("target").join("nn-compiled"))
        .unwrap_or_else(|| PathBuf::from("target/nn-compiled"))
}

/// Find all `.metal` files in the given directory (non-recursive).
fn find_metal_files(dir: &Path) -> Vec<PathBuf> {
    if !dir.is_dir() {
        return Vec::new();
    }

    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "metal"))
        .collect();

    files.sort();
    files
}

/// Compile a `.metal` file to `.air` (Apple Intermediate Representation).
fn compile_metal_to_air(metal_path: &Path, out_dir: &Path) -> PathBuf {
    let stem = metal_path
        .file_stem()
        .expect("invariant: .metal file has a stem")
        .to_str()
        .expect("invariant: .metal filename is valid UTF-8");
    let air_path = out_dir.join(format!("{stem}.air"));

    let metal_str = metal_path
        .to_str()
        .expect("invariant: .metal path is valid UTF-8");
    let air_str = air_path
        .to_str()
        .expect("invariant: .air path is valid UTF-8");

    let output = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-c", metal_str, "-o", air_str])
        .output()
        .unwrap_or_else(|e| {
            panic!("invariant: xcrun metal must be available on macOS build host: {e}")
        });

    assert!(
        output.status.success(),
        "xcrun metal failed for {}:\n  exit code: {:?}\n  stderr: {}",
        metal_path.display(),
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    air_path
}

/// Link multiple `.air` files into a single `.metallib`.
fn link_air_to_metallib(air_files: &[PathBuf], metallib_path: &Path) {
    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib"]);
    for air in air_files {
        cmd.arg(air.to_str().expect("invariant: .air path is valid UTF-8"));
    }
    cmd.args([
        "-o",
        metallib_path
            .to_str()
            .expect("invariant: metallib path is valid UTF-8"),
    ]);

    let output = cmd.output().unwrap_or_else(|e| {
        panic!("invariant: xcrun metallib must be available on macOS build host: {e}")
    });

    assert!(
        output.status.success(),
        "xcrun metallib failed:\n  exit code: {:?}\n  stderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
}
