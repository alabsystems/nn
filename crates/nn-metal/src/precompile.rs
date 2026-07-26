// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Collect native Metal kernel MSL sources for pre-compilation to `.metallib`.
//!
//! [`collect_native_kernel_sources`] returns all fixed-name kernel MSL sources
//! used by the Kokoro pipeline and DynTensor dispatch. These can be written
//! to `.metal` files and compiled at build time via `build.rs`, eliminating
//! first-inference MSL compilation overhead (~1ms/kernel → ~10µs/kernel).
//!
//! # Usage
//!
//! ```rust,ignore
//! use nn_metal::precompile::{collect_native_kernel_sources, write_metal_sources};
//!
//! let sources = collect_native_kernel_sources();
//! write_metal_sources(&sources, "target/nn-compiled/")?;
//! ```

use std::io;
use std::path::Path;

/// A pre-compilable Metal kernel: entry point name + MSL source.
#[derive(Debug, Clone)]
pub struct KernelMslSource {
    /// MSL function name (e.g., `"fused_adain_snake_f32"`).
    pub entry_point: &'static str,
    /// Complete MSL source text for the kernel.
    pub msl_source: String,
}

/// Collect all native kernel MSL sources suitable for pre-compilation.
///
/// Returns sources for all fixed-name Metal kernels (those with static MSL
/// that doesn't depend on model dimensions). Parameterized kernels (LSTM
/// sequence, cumsum block scan) are excluded — they require model-specific
/// dimensions and are compiled at runtime.
///
/// The returned kernels cover:
/// - Fused normalization+activation: AdaIN+Snake, AdaIN+LeakyRelu,
///   AdaLayerNorm, InstanceNorm
/// - Compute: simdgroup GEMM (f32, f16), Flash Attention (f32, f16)
/// - Signal processing: forward STFT DFT, iSTFT (IDFT + overlap-add),
///   polar-to-rect conversion
/// - Prefix scan: cumsum single-pass, cumsum propagate
#[must_use]
pub fn collect_native_kernel_sources() -> Vec<KernelMslSource> {
    let mut sources = Vec::new();

    // Native fused kernels from dyn_tensor_metal submodules.
    for (name, msl) in crate::dyn_tensor_metal::collect_native_msl_sources() {
        sources.push(KernelMslSource {
            entry_point: name,
            msl_source: msl,
        });
    }

    // STFT/iSTFT kernels (crate-root modules).
    sources.push(KernelMslSource {
        entry_point: "stft_dft_f32",
        msl_source: crate::stft_gpu::stft_dft_msl_source(),
    });
    sources.push(KernelMslSource {
        entry_point: "stft_fft_f32",
        msl_source: crate::stft_gpu::stft_fft_msl_source(),
    });
    sources.push(KernelMslSource {
        entry_point: "istft_idft_f32",
        msl_source: crate::istft_gpu::istft_idft_msl_source(),
    });
    sources.push(KernelMslSource {
        entry_point: "istft_overlap_add_f32",
        msl_source: crate::istft_gpu::istft_overlap_add_msl_source(),
    });
    sources.push(KernelMslSource {
        entry_point: "istft_fused_polar_f32",
        msl_source: crate::istft_gpu::istft_fused_polar_msl_source(),
    });

    sources
}

/// Write collected MSL sources as `.metal` files to a directory.
///
/// Creates one `.metal` file per kernel entry point. The directory is created
/// if it doesn't exist. `build.rs` picks up files from this directory and
/// compiles them to a single `.metallib`.
///
/// # Errors
///
/// Returns `io::Error` if directory creation or file writing fails.
pub fn write_metal_sources(sources: &[KernelMslSource], dir: &Path) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let expected_files: std::collections::HashSet<_> = sources
        .iter()
        .map(|source| format!("{}.metal", source.entry_point))
        .collect();

    // Remove stale generated .metal files so deleted kernels don't linger in
    // target/nn-compiled and get pulled back into the build-time metallib.
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "metal") {
            let file_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid UTF-8 filename: {}", path.display()),
                    )
                })?;
            if !expected_files.contains(file_name) {
                std::fs::remove_file(&path)?;
            }
        }
    }

    for source in sources {
        let file_path = dir.join(format!("{}.metal", source.entry_point));
        std::fs::write(&file_path, &source.msl_source)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_native_kernel_sources_nonempty() {
        let sources = collect_native_kernel_sources();
        assert!(
            sources.len() >= 14,
            "expected at least 14 native kernels, got {}",
            sources.len()
        );
    }

    #[test]
    fn test_all_sources_have_kernel_entry_point() {
        let sources = collect_native_kernel_sources();
        for source in &sources {
            assert!(
                source
                    .msl_source
                    .contains(&format!("kernel void {}", source.entry_point)),
                "MSL for '{}' does not contain expected kernel function declaration",
                source.entry_point,
            );
        }
    }

    #[test]
    fn test_unique_entry_points() {
        let sources = collect_native_kernel_sources();
        let mut names: Vec<_> = sources.iter().map(|s| s.entry_point).collect();
        names.sort_unstable();
        let len_before = names.len();
        names.dedup();
        assert_eq!(
            names.len(),
            len_before,
            "duplicate kernel entry point names found"
        );
    }

    #[test]
    fn test_write_metal_sources_creates_files() {
        let sources = collect_native_kernel_sources();
        let dir = std::env::temp_dir().join(format!("nn_precompile_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        write_metal_sources(&sources, &dir).expect("write_metal_sources");

        for source in &sources {
            let path = dir.join(format!("{}.metal", source.entry_point));
            assert!(
                path.exists(),
                "missing .metal file for {}",
                source.entry_point
            );
            let content = std::fs::read_to_string(&path).expect("read");
            assert_eq!(content, source.msl_source);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_write_metal_sources_removes_stale_files() {
        let dir =
            std::env::temp_dir().join(format!("nn_precompile_stale_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");

        let stale = dir.join("stale_kernel.metal");
        std::fs::write(&stale, "kernel void stale() {}").expect("write stale file");
        assert!(stale.exists(), "stale file should exist before rewrite");

        let sources = [
            KernelMslSource {
                entry_point: "fresh_kernel",
                msl_source: "kernel void fresh_kernel() {}".to_string(),
            },
            KernelMslSource {
                entry_point: "other_kernel",
                msl_source: "kernel void other_kernel() {}".to_string(),
            },
        ];

        write_metal_sources(&sources, &dir).expect("write_metal_sources");

        assert!(
            !stale.exists(),
            "stale .metal file should be removed on rewrite"
        );
        assert!(
            dir.join("fresh_kernel.metal").exists(),
            "fresh file should be written"
        );
        assert!(
            dir.join("other_kernel.metal").exists(),
            "second fresh file should be written"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
