// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HIP compilation pipeline: source → `.hsaco` code object.
//!
//! Compiles generated HIP C++ source to AMD GPU Code Objects (`.hsaco`)
//! via `hipcc --genco`. The `.hsaco` files are ELF shared objects that
//! can be loaded at runtime via `hipModuleLoad`.
//!
//! # Pipeline
//!
//! ```text
//! HIP C++ source (String)
//!   → write to temp .cpp file
//!   → hipcc --genco --offload-arch=<target> -o output.hsaco source.cpp
//!   → .hsaco code object (cached by content hash + arch)
//! ```
//!
//! # Platform support
//!
//! `hipcc` is only available on systems with ROCm installed (Linux with
//! AMD GPUs). On macOS or other platforms, `check_hipcc()` returns `false`
//! and compilation attempts return `HipCompileError::HipccNotFound`.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::hip_cache::HipCache;

/// Errors from the HIP compilation pipeline.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HipCompileError {
    #[error("`hipcc` not found — ROCm must be installed")]
    HipccNotFound,

    #[error("hipcc compilation failed (exit code {exit_code:?}):\n{stderr}")]
    CompilationFailed {
        exit_code: Option<i32>,
        stderr: String,
    },

    #[error("I/O error during HIP compilation: {0}")]
    Io(#[from] std::io::Error),

    #[error("codegen error: {0}")]
    Codegen(#[from] crate::HipCodegenError),
}

/// Result of a successful HIP compilation.
#[derive(Debug, Clone)]
pub struct HipModule {
    /// Path to the compiled `.hsaco` code object.
    pub hsaco_path: PathBuf,
    /// Target GPU architecture (e.g., `"gfx90a"`, `"gfx1100"`).
    pub target_arch: String,
    /// Whether this was a cache hit (no compilation needed).
    pub cache_hit: bool,
}

/// Common AMD GPU architecture targets.
///
/// Used with `hipcc --offload-arch=<target>`.
pub mod target {
    /// MI200 / MI210 / MI250 (CDNA2, datacenter).
    pub const GFX90A: &str = "gfx90a";
    /// MI300X (CDNA3, datacenter).
    pub const GFX942: &str = "gfx942";
    /// MI350X / MI355X (CDNA4, datacenter). AMD x GPU MODE competition target.
    pub const GFX950: &str = "gfx950";
    /// RX 7900 XTX (RDNA3, consumer).
    pub const GFX1100: &str = "gfx1100";
    /// RX 7600 (RDNA3, consumer).
    pub const GFX1102: &str = "gfx1102";
}

/// Check whether `hipcc` is available on this system.
///
/// Returns `true` if `hipcc --version` exits successfully.
pub fn check_hipcc() -> bool {
    Command::new("hipcc")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Compile HIP C++ source to a `.hsaco` code object.
///
/// If a [`HipCache`] is provided and the source has been compiled before
/// (same content hash + target arch), returns the cached `.hsaco` path
/// without recompilation.
///
/// # Arguments
///
/// * `source` — Complete HIP C++ source code (including `#include` directives).
/// * `target_arch` — GPU architecture target (e.g., `"gfx90a"`). See [`target`].
/// * `cache` — Optional filesystem cache. `None` compiles to a temp directory.
///
/// # Errors
///
/// Returns `HipCompileError::HipccNotFound` if `hipcc` is not installed.
/// Returns `HipCompileError::CompilationFailed` if `hipcc` exits with an error.
pub fn compile_hip_source(
    source: &str,
    target_arch: &str,
    cache: Option<&HipCache>,
) -> Result<HipModule, HipCompileError> {
    // Check cache first.
    if let Some(c) = cache {
        if let Some(cached_path) = c.lookup(source, target_arch) {
            return Ok(HipModule {
                hsaco_path: cached_path,
                target_arch: target_arch.to_owned(),
                cache_hit: true,
            });
        }
    }

    if !check_hipcc() {
        return Err(HipCompileError::HipccNotFound);
    }

    // Determine output directory.
    // When no cache is provided, create a temp directory and use `keep()`
    // to prevent auto-cleanup. The caller owns the returned hsaco_path and is
    // responsible for cleanup of the parent directory.
    let out_dir = match cache {
        Some(c) => c.dir().to_owned(),
        None => {
            let tmp = tempfile::tempdir()?;
            tmp.keep() // Prevents auto-cleanup — caller owns the path.
        }
    };

    // Write source to a .cpp file.
    let source_hash = HipCache::content_hash(source, target_arch);
    let source_path = out_dir.join(format!("{source_hash}.hip.cpp"));
    {
        let mut f = std::fs::File::create(&source_path)?;
        f.write_all(source.as_bytes())?;
    }

    let hsaco_path = out_dir.join(format!("{source_hash}.hsaco"));

    // hipcc --genco --offload-arch=<target> -o output.hsaco source.cpp
    let output = Command::new("hipcc")
        .arg("--genco")
        .arg(format!("--offload-arch={target_arch}"))
        .arg("-O3")
        .arg("-o")
        .arg(&hsaco_path)
        .arg(&source_path)
        .output()?;

    if !output.status.success() {
        return Err(HipCompileError::CompilationFailed {
            exit_code: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }

    // Register in cache.
    if let Some(c) = cache {
        c.register(source, target_arch, &hsaco_path);
    }

    Ok(HipModule {
        hsaco_path,
        target_arch: target_arch.to_owned(),
        cache_hit: false,
    })
}

/// Generate the `hipcc` command line for a given source, without executing it.
///
/// Useful for CI scripts, logging, and environments without `hipcc`.
/// Returns the command as a `Vec<String>` suitable for `Command::new(args[0]).args(&args[1..])`.
pub fn hipcc_command(source_path: &Path, hsaco_path: &Path, target_arch: &str) -> Vec<String> {
    vec![
        "hipcc".to_owned(),
        "--genco".to_owned(),
        format!("--offload-arch={target_arch}"),
        "-O3".to_owned(),
        "-o".to_owned(),
        hsaco_path.display().to_string(),
        source_path.display().to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hipcc_command_format() {
        let cmd = hipcc_command(
            Path::new("/tmp/kernel.hip.cpp"),
            Path::new("/tmp/kernel.hsaco"),
            "gfx90a",
        );
        assert_eq!(cmd[0], "hipcc");
        assert_eq!(cmd[1], "--genco");
        assert_eq!(cmd[2], "--offload-arch=gfx90a");
        assert_eq!(cmd[3], "-O3");
        assert_eq!(cmd[4], "-o");
        assert_eq!(cmd[5], "/tmp/kernel.hsaco");
        assert_eq!(cmd[6], "/tmp/kernel.hip.cpp");
    }

    #[test]
    fn test_target_arch_constants() {
        assert_eq!(target::GFX90A, "gfx90a");
        assert_eq!(target::GFX942, "gfx942");
        assert_eq!(target::GFX950, "gfx950");
        assert_eq!(target::GFX1100, "gfx1100");
        assert_eq!(target::GFX1102, "gfx1102");
    }

    #[test]
    fn test_compile_without_hipcc() {
        // On macOS (where we develop), hipcc is not available.
        // This test verifies graceful fallback.
        if check_hipcc() {
            return; // Skip — hipcc is available, cannot test absence.
        }
        let result = compile_hip_source("__global__ void k() {}", "gfx90a", None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, HipCompileError::HipccNotFound),
            "expected HipccNotFound, got: {err}"
        );
    }
}
