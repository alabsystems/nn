// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! NVIDIA CUDA compilation pipeline: CUDA C++ source -> PTX -> cubin.
//!
//! Parallel to [`compile_hip`](super::compile_hip). Compiles generated CUDA C++
//! source to PTX assembly via `nvcc`, and optionally to cubin (device binary)
//! via `ptxas`.
//!
//! # Pipeline
//!
//! ```text
//! CUDA C++ source (String)
//!   -> write to temp .cu file
//!   -> nvcc --ptx --gpu-architecture=<sm_target> -o output.ptx source.cu
//!   -> .ptx assembly (can be JIT-compiled by CUDA driver)
//!
//! Optionally:
//!   .ptx -> ptxas --gpu-name=<sm_target> -o output.cubin input.ptx
//!   -> .cubin device binary (no JIT needed at runtime)
//! ```
//!
//! # Platform support
//!
//! `nvcc` requires the CUDA Toolkit. On systems without it, `check_nvcc()`
//! returns `false` and compilation returns `PtxCompileError::NvccNotFound`.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::hip_cache::HipCache;

/// Errors from the PTX compilation pipeline.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PtxCompileError {
    #[error("`nvcc` not found — CUDA Toolkit must be installed")]
    NvccNotFound,

    #[error("`ptxas` not found — CUDA Toolkit must be installed")]
    PtxasNotFound,

    #[error("nvcc compilation failed (exit code {exit_code:?}):\n{stderr}")]
    CompilationFailed {
        exit_code: Option<i32>,
        stderr: String,
    },

    #[error("ptxas assembly failed (exit code {exit_code:?}):\n{stderr}")]
    AssemblyFailed {
        exit_code: Option<i32>,
        stderr: String,
    },

    #[error("I/O error during PTX compilation: {0}")]
    Io(#[from] std::io::Error),

    #[error("codegen error: {0}")]
    Codegen(#[from] crate::codegen_ptx::PtxCodegenError),
}

/// Result of a successful CUDA compilation to PTX.
#[derive(Debug, Clone)]
pub struct PtxModule {
    /// Path to the compiled `.ptx` assembly file.
    pub ptx_path: PathBuf,
    /// Path to the `.cubin` device binary (if assembled via `ptxas`).
    pub cubin_path: Option<PathBuf>,
    /// Target SM architecture (e.g., `"sm_80"`).
    pub sm_target: String,
    /// Whether this was a cache hit (no compilation needed).
    pub cache_hit: bool,
}

/// Check whether `nvcc` is available on this system.
///
/// Returns `true` if `nvcc --version` exits successfully.
pub fn check_nvcc() -> bool {
    Command::new("nvcc")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Check whether `ptxas` is available on this system.
pub fn check_ptxas() -> bool {
    Command::new("ptxas")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Compile CUDA C++ source to PTX assembly via `nvcc`.
///
/// If a [`HipCache`] is provided (reused for CUDA; keyed by content hash +
/// target), returns the cached `.ptx` path without recompilation.
///
/// # Arguments
///
/// * `source` — Complete CUDA C++ source code (including `#include` directives).
/// * `sm_target` — GPU architecture target (e.g., `"sm_80"`). See [`super::cuda_ffi::sm_target`].
/// * `cache` — Optional filesystem cache. `None` compiles to a temp directory.
///
/// # Errors
///
/// Returns `PtxCompileError::NvccNotFound` if `nvcc` is not installed.
/// Returns `PtxCompileError::CompilationFailed` if `nvcc` exits with an error.
pub fn compile_cuda_to_ptx(
    source: &str,
    sm_target: &str,
    cache: Option<&HipCache>,
) -> Result<PtxModule, PtxCompileError> {
    // Check cache first.
    let cache_key = format!("ptx_{sm_target}");
    if let Some(c) = cache {
        if let Some(cached_path) = c.lookup(source, &cache_key) {
            return Ok(PtxModule {
                ptx_path: cached_path,
                cubin_path: None,
                sm_target: sm_target.to_owned(),
                cache_hit: true,
            });
        }
    }

    if !check_nvcc() {
        return Err(PtxCompileError::NvccNotFound);
    }

    // Determine output directory.
    let out_dir = match cache {
        Some(c) => c.dir().to_owned(),
        None => {
            let tmp = tempfile::tempdir()?;
            tmp.keep()
        }
    };

    // Write source to a .cu file.
    let source_hash = HipCache::content_hash(source, &cache_key);
    let source_path = out_dir.join(format!("{source_hash}.cu"));
    {
        let mut f = std::fs::File::create(&source_path)?;
        f.write_all(source.as_bytes())?;
    }

    let ptx_path = out_dir.join(format!("{source_hash}.ptx"));

    // nvcc --ptx --gpu-architecture=<target> -O3 -o output.ptx source.cu
    let output = Command::new("nvcc")
        .arg("--ptx")
        .arg(format!("--gpu-architecture={sm_target}"))
        .arg("-O3")
        .arg("-o")
        .arg(&ptx_path)
        .arg(&source_path)
        .output()?;

    if !output.status.success() {
        return Err(PtxCompileError::CompilationFailed {
            exit_code: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }

    // Register in cache.
    if let Some(c) = cache {
        c.register(source, &cache_key, &ptx_path);
    }

    Ok(PtxModule {
        ptx_path,
        cubin_path: None,
        sm_target: sm_target.to_owned(),
        cache_hit: false,
    })
}

/// Assemble PTX to cubin via `ptxas`.
///
/// Takes a `.ptx` file and produces a `.cubin` device binary. The cubin
/// can be loaded via `cuModuleLoad` without JIT compilation at runtime.
pub fn assemble_ptx_to_cubin(ptx_path: &Path, sm_target: &str) -> Result<PathBuf, PtxCompileError> {
    if !check_ptxas() {
        return Err(PtxCompileError::PtxasNotFound);
    }

    let cubin_path = ptx_path.with_extension("cubin");

    let output = Command::new("ptxas")
        .arg(format!("--gpu-name={sm_target}"))
        .arg("-O3")
        .arg("-o")
        .arg(&cubin_path)
        .arg(ptx_path)
        .output()?;

    if !output.status.success() {
        return Err(PtxCompileError::AssemblyFailed {
            exit_code: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }

    Ok(cubin_path)
}

/// Generate the `nvcc` command line for compilation, without executing it.
///
/// Useful for CI scripts, logging, and environments without `nvcc`.
pub fn nvcc_command(source_path: &Path, ptx_path: &Path, sm_target: &str) -> Vec<String> {
    vec![
        "nvcc".to_owned(),
        "--ptx".to_owned(),
        format!("--gpu-architecture={sm_target}"),
        "-O3".to_owned(),
        "-o".to_owned(),
        ptx_path.display().to_string(),
        source_path.display().to_string(),
    ]
}

/// Generate the `ptxas` command line for assembly, without executing it.
pub fn ptxas_command(ptx_path: &Path, cubin_path: &Path, sm_target: &str) -> Vec<String> {
    vec![
        "ptxas".to_owned(),
        format!("--gpu-name={sm_target}"),
        "-O3".to_owned(),
        "-o".to_owned(),
        cubin_path.display().to_string(),
        ptx_path.display().to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nvcc_command_format() {
        let cmd = nvcc_command(
            Path::new("/tmp/kernel.cu"),
            Path::new("/tmp/kernel.ptx"),
            "sm_80",
        );
        assert_eq!(cmd[0], "nvcc");
        assert_eq!(cmd[1], "--ptx");
        assert_eq!(cmd[2], "--gpu-architecture=sm_80");
        assert_eq!(cmd[3], "-O3");
        assert_eq!(cmd[4], "-o");
        assert_eq!(cmd[5], "/tmp/kernel.ptx");
        assert_eq!(cmd[6], "/tmp/kernel.cu");
    }

    #[test]
    fn test_ptxas_command_format() {
        let cmd = ptxas_command(
            Path::new("/tmp/kernel.ptx"),
            Path::new("/tmp/kernel.cubin"),
            "sm_80",
        );
        assert_eq!(cmd[0], "ptxas");
        assert_eq!(cmd[1], "--gpu-name=sm_80");
        assert_eq!(cmd[2], "-O3");
        assert_eq!(cmd[3], "-o");
        assert_eq!(cmd[4], "/tmp/kernel.cubin");
        assert_eq!(cmd[5], "/tmp/kernel.ptx");
    }

    #[test]
    fn test_compile_without_nvcc() {
        // On macOS (where we develop), nvcc is not available.
        if check_nvcc() {
            return; // Skip — nvcc is available, cannot test absence.
        }
        let result = compile_cuda_to_ptx("__global__ void k() {}", "sm_80", None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, PtxCompileError::NvccNotFound),
            "expected NvccNotFound, got: {err}"
        );
    }
}
