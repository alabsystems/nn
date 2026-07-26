// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the HIP compilation pipeline.
//!
//! These tests exercise the end-to-end path: TensorOpKind IR → HIP C++ source
//! → hipcc compilation (or graceful fallback when hipcc is absent).

use nn_cuda::{
    check_hipcc, compile_hip_source, emit_gemm_hip, hipcc_command, HipCache, HipCompileError,
};
use nn_dsl::ScalarType;
use std::path::Path;

#[test]
fn test_gemm_generates_valid_hip_source() {
    let source = emit_gemm_hip("test_gemm", ScalarType::F32, 64, 128, 64).unwrap();

    // Basic structural checks on generated HIP C++.
    assert!(source.contains("#include <hip/hip_runtime.h>"));
    assert!(source.contains("__global__"));
    assert!(source.contains("test_gemm"));
}

#[test]
fn test_cache_roundtrip_with_codegen() {
    let tmp = tempfile::tempdir().unwrap();
    let cache = HipCache::new(tmp.path()).unwrap();

    let source = emit_gemm_hip("cached_gemm", ScalarType::F32, 32, 32, 32).unwrap();

    // No cache entry yet.
    assert!(cache.lookup(&source, "gfx90a").is_none());

    // Simulate registration (as if hipcc compiled it).
    let fake_hsaco = tmp.path().join("fake.hsaco");
    std::fs::write(&fake_hsaco, b"ELF_FAKE_HSACO_DATA").unwrap();
    cache.register(&source, "gfx90a", &fake_hsaco);

    // Cache hit.
    let cached = cache.lookup(&source, "gfx90a");
    assert!(cached.is_some());

    // Different arch = miss.
    assert!(cache.lookup(&source, "gfx1100").is_none());
}

#[test]
fn test_hipcc_command_generation() {
    let cmd = hipcc_command(
        Path::new("/src/kernel.hip.cpp"),
        Path::new("/out/kernel.hsaco"),
        "gfx942",
    );
    assert_eq!(cmd.len(), 7);
    assert_eq!(cmd[0], "hipcc");
    assert!(cmd[2].contains("gfx942"));
}

#[test]
fn test_compile_graceful_fallback() {
    // On macOS (no ROCm), hipcc is not available.
    // Verify the pipeline returns a structured error, not a panic.
    if check_hipcc() {
        // hipcc is available — run actual compilation test instead.
        let source = emit_gemm_hip("compile_test", ScalarType::F32, 16, 16, 16).unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let cache = HipCache::new(tmp.path()).unwrap();
        let result = compile_hip_source(&source, "gfx90a", Some(&cache));
        // If hipcc is available, compilation should succeed.
        assert!(
            result.is_ok(),
            "hipcc available but compilation failed: {result:?}"
        );
        let module = result.unwrap();
        assert!(!module.cache_hit);
        assert!(module.hsaco_path.exists());
        return;
    }

    let source = "__global__ void test_kernel() {}";
    let result = compile_hip_source(source, "gfx90a", None);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        HipCompileError::HipccNotFound
    ));
}

#[test]
fn test_compile_with_cache_hit() {
    if check_hipcc() {
        // Full pipeline test: compile, then verify cache hit on second call.
        let source = emit_gemm_hip("cache_hit_test", ScalarType::F32, 8, 8, 8).unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let cache = HipCache::new(tmp.path()).unwrap();

        let first = compile_hip_source(&source, "gfx90a", Some(&cache)).unwrap();
        assert!(!first.cache_hit);

        let second = compile_hip_source(&source, "gfx90a", Some(&cache)).unwrap();
        assert!(second.cache_hit);
    }
}
