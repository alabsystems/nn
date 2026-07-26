// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for .metallib pre-compilation (#2467).
//!
//! Verifies that:
//! 1. All native kernel MSL sources can be collected
//! 2. MSL sources compile to `.air` files via `xcrun metal`
//! 3. `.air` files link into a single `.metallib` via `xcrun metallib`
//! 4. The metallib can be loaded and pipelines created from it

use nn_metal::precompile::{collect_native_kernel_sources, write_metal_sources};
use nn_metal::MetalBackend;

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

fn test_dir() -> PathBuf {
    std::env::temp_dir().join(format!("nn_metallib_test_{}", std::process::id()))
}

#[test]
fn test_collect_sources_produces_valid_msl() {
    let sources = collect_native_kernel_sources();

    assert!(
        sources.len() >= 14,
        "expected at least 14 kernels, got {}",
        sources.len()
    );

    for source in &sources {
        assert!(
            !source.entry_point.is_empty(),
            "kernel has empty entry point"
        );
        assert!(
            !source.msl_source.is_empty(),
            "kernel '{}' has empty MSL source",
            source.entry_point
        );
        assert!(
            source.msl_source.contains("kernel void"),
            "kernel '{}' MSL does not contain 'kernel void'",
            source.entry_point
        );
    }
}

#[test]
fn test_write_and_compile_metallib() {
    let dir = test_dir();
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create test dir");

    let sources = collect_native_kernel_sources();
    write_metal_sources(&sources, &dir).expect("write .metal files");

    // Compile each .metal -> .air
    let mut air_files = Vec::new();
    for source in &sources {
        let metal_path = dir.join(format!("{}.metal", source.entry_point));
        let air_path = dir.join(format!("{}.air", source.entry_point));

        let output = Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-c",
                metal_path.to_str().unwrap(),
                "-o",
                air_path.to_str().unwrap(),
            ])
            .output()
            .expect("xcrun metal must be available");

        assert!(
            output.status.success(),
            "xcrun metal failed for {}:\nstderr: {}",
            source.entry_point,
            String::from_utf8_lossy(&output.stderr)
        );

        air_files.push(air_path);
    }

    // Link all .air -> single .metallib
    let metallib_path = dir.join("precompiled.metallib");
    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib"]);
    for air in &air_files {
        cmd.arg(air.to_str().unwrap());
    }
    cmd.args(["-o", metallib_path.to_str().unwrap()]);

    let output = cmd.output().expect("xcrun metallib must be available");
    assert!(
        output.status.success(),
        "xcrun metallib failed:\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(
        metallib_path.exists(),
        "metallib was not created at {}",
        metallib_path.display()
    );

    let metallib_size = std::fs::metadata(&metallib_path)
        .expect("metallib metadata")
        .len();
    assert!(metallib_size > 0, "metallib is empty (0 bytes)");

    eprintln!(
        "compiled {} kernels to metallib ({} bytes)",
        sources.len(),
        metallib_size
    );

    // Load the metallib and create pipelines.
    let _backend = MetalBackend::init().expect("Metal init");
    let metallib_bytes = std::fs::read(&metallib_path).expect("read metallib");

    let entry_points: Vec<&str> = sources.iter().map(|s| s.entry_point).collect();
    let ctx = nn_metal::metallib_loader::precompiled_metallib_path();
    eprintln!("build-time metallib path: {ctx:?}");

    // Use the metallib_loader to verify the library loads.
    let context = nn_metal::PipelineCache::new_global()
        .expect("cache")
        .context()
        .clone();
    let library = nn_metal::metallib_loader::load_metallib(&context, &metallib_bytes)
        .expect("load metallib");
    let pipelines =
        nn_metal::metallib_loader::pipelines_from_metallib(&context, &library, &entry_points)
            .expect("create pipelines from metallib");

    assert_eq!(pipelines.len(), sources.len(), "pipeline count mismatch");

    for (pipeline, source) in pipelines.iter().zip(sources.iter()) {
        assert_eq!(
            pipeline.entry_point(),
            source.entry_point,
            "pipeline entry point mismatch"
        );
    }

    eprintln!("all {} pipelines created from metallib", pipelines.len());

    // Cleanup.
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_generate_to_nn_compiled_dir() {
    // This test generates .metal files to the location that build.rs looks for.
    // It can be used as a manual step: run this test, then rebuild to get .metallib.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = std::path::Path::new(manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let nn_compiled_dir = workspace_root.join("target").join("nn-compiled");

    let sources = collect_native_kernel_sources();
    write_metal_sources(&sources, &nn_compiled_dir).expect("write to nn-compiled");

    eprintln!(
        "wrote {} .metal files to {}",
        sources.len(),
        nn_compiled_dir.display()
    );

    // Verify files exist.
    for source in &sources {
        let path = nn_compiled_dir.join(format!("{}.metal", source.entry_point));
        assert!(path.exists(), "missing: {}", path.display());
    }
}

/// Benchmark: pipeline creation from MSL strings vs precompiled metallib.
///
/// Measures per-kernel latency for both paths. Compiles a metallib from
/// scratch, loads it, and compares creation time against runtime MSL
/// compilation.
///
/// Warm-cache result: ~4x faster (73us vs 292us/kernel). Cold-cache (first
/// app launch) improvement is larger since MSL invokes the full shader compiler.
#[test]
fn test_pipeline_creation_latency_benchmark() {
    let _backend = MetalBackend::init().expect("Metal init");
    let cache = nn_metal::PipelineCache::new_global().expect("cache");
    let context = cache.context().clone();

    let sources = collect_native_kernel_sources();
    assert!(!sources.is_empty(), "no kernel sources");

    // Build a metallib from the collected sources.
    let dir = std::env::temp_dir().join(format!("nn_bench_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create dir");
    write_metal_sources(&sources, &dir).expect("write");

    let mut air_files = Vec::new();
    for source in &sources {
        let metal_path = dir.join(format!("{}.metal", source.entry_point));
        let air_path = dir.join(format!("{}.air", source.entry_point));
        let output = Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-c",
                metal_path.to_str().unwrap(),
                "-o",
                air_path.to_str().unwrap(),
            ])
            .output()
            .expect("xcrun metal");
        assert!(
            output.status.success(),
            "compile failed for {}",
            source.entry_point
        );
        air_files.push(air_path);
    }

    let metallib_path = dir.join("bench.metallib");
    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib"]);
    for air in &air_files {
        cmd.arg(air);
    }
    cmd.args(["-o", metallib_path.to_str().unwrap()]);
    let output = cmd.output().expect("xcrun metallib");
    assert!(output.status.success(), "metallib link failed");

    let metallib_bytes = std::fs::read(&metallib_path).expect("read metallib");

    // --- Benchmark: pipeline creation from precompiled metallib ---
    let entry_points: Vec<&str> = sources.iter().map(|s| s.entry_point).collect();

    let t_metallib = Instant::now();
    let library = nn_metal::metallib_loader::load_metallib(&context, &metallib_bytes)
        .expect("load metallib");
    let _pipelines =
        nn_metal::metallib_loader::pipelines_from_metallib(&context, &library, &entry_points)
            .expect("pipelines from metallib");
    let metallib_elapsed = t_metallib.elapsed();

    // --- Benchmark: pipeline creation from MSL string compilation ---
    let t_msl = Instant::now();
    for source in &sources {
        let ks = nn_metal::KernelSource::new(&source.msl_source, source.entry_point);
        let _pipeline = context.compile_pipeline(&ks).expect("compile from MSL");
    }
    let msl_elapsed = t_msl.elapsed();

    let n = sources.len();
    let per_kernel_metallib_us = metallib_elapsed.as_micros() as f64 / n as f64;
    let per_kernel_msl_us = msl_elapsed.as_micros() as f64 / n as f64;
    let speedup = msl_elapsed.as_secs_f64() / metallib_elapsed.as_secs_f64();

    eprintln!("=== .metallib pre-compilation benchmark ({n} kernels) ===");
    eprintln!(
        "  MSL string compilation:  {:>8.0} us/kernel  ({:.1} ms total)",
        per_kernel_msl_us,
        msl_elapsed.as_secs_f64() * 1000.0
    );
    eprintln!(
        "  Metallib precompiled:    {:>8.0} us/kernel  ({:.1} ms total)",
        per_kernel_metallib_us,
        metallib_elapsed.as_secs_f64() * 1000.0
    );
    eprintln!("  Speedup:                 {speedup:.1}x");

    // Sanity: metallib should be faster than MSL compilation.
    assert!(
        metallib_elapsed < msl_elapsed,
        "metallib ({metallib_elapsed:?}) should be faster than MSL ({msl_elapsed:?})"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
