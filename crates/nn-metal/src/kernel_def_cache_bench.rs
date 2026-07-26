// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Benchmark tests for [`KernelDefCache`] IR build cost savings.
//!
//! Validates that the cache eliminates redundant `TensorBlockBuilder` IR
//! construction for DynTensor GPU ops. Uses real IR graphs (matmul, add,
//! gelu) to measure cold-build vs warm-cache-hit performance.
//!
//! Design: `designs/2026-03-07-gpu-dispatch-unification.md` (Direction 3).

use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::TensorBlockBuilder;

use nn_core::DType;

use super::kernel_def_cache::{cache_len, clear_cache, get_or_build};

/// Build a real matmul TensorKernelDef via TensorBlockBuilder.
fn build_real_matmul_def(m: usize, k: usize, n: usize) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("bench_matmul");
    let lhs = b.add_input("lhs", &[m, k]);
    let rhs = b.add_input("rhs", &[k, n]);
    let out = b.add_matmul(lhs, rhs, false, None, &[m, n]);
    b.build(out).expect("invariant: valid matmul graph")
}

#[test]
fn test_cache_eliminates_ir_rebuild_for_real_matmul() {
    clear_cache();

    let shapes: &[&[usize]] = &[&[64, 128], &[128, 256]];

    let mut build_count = 0u32;

    // First call: cache miss, builds full IR
    let def1 = get_or_build("matmul", shapes, &[], DType::F32, || {
        build_count += 1;
        Ok(build_real_matmul_def(64, 128, 256))
    })
    .expect("invariant: matmul build succeeds");
    assert_eq!(build_count, 1, "first call should build IR");
    assert_eq!(cache_len(), 1, "one entry cached");

    // Second call with same key: cache hit, no rebuild
    let def2 = get_or_build("matmul", shapes, &[], DType::F32, || {
        build_count += 1;
        Ok(build_real_matmul_def(64, 128, 256))
    })
    .expect("invariant: cache hit succeeds");
    assert_eq!(build_count, 1, "second call should hit cache, no rebuild");
    assert_eq!(def1.name, def2.name, "same def returned from cache");
    assert_eq!(cache_len(), 1, "still one entry");
}

#[test]
fn test_cache_handles_multiple_real_op_types() {
    clear_cache();

    let mut total_builds = 0u32;

    // 1. Matmul
    get_or_build("matmul", &[&[32, 64], &[64, 32]], &[], DType::F32, || {
        total_builds += 1;
        Ok(build_real_matmul_def(32, 64, 32))
    })
    .expect("invariant: matmul build succeeds");

    // 2. Add (binary element-wise)
    get_or_build("add", &[&[32, 64], &[32, 64]], &[], DType::F32, || {
        total_builds += 1;
        let mut b = TensorBlockBuilder::new("bench_add");
        let lhs = b.add_input("lhs", &[32, 64]);
        let rhs = b.add_input("rhs", &[32, 64]);
        let out = b.add_binary_add(lhs, rhs, &[32, 64]);
        Ok(b.build(out).expect("invariant: valid add graph"))
    })
    .expect("invariant: add build succeeds");

    // 3. GELU (unary activation)
    get_or_build("gelu", &[&[32, 64]], &[], DType::F32, || {
        total_builds += 1;
        let mut b = TensorBlockBuilder::new("bench_gelu");
        let inp = b.add_input("x", &[32, 64]);
        let out = b.add_gelu(inp, &[32, 64]);
        Ok(b.build(out).expect("invariant: valid gelu graph"))
    })
    .expect("invariant: gelu build succeeds");

    assert_eq!(total_builds, 3, "3 unique ops built");
    assert_eq!(cache_len(), 3, "3 entries cached");

    // Hit all 3 caches again — zero rebuilds expected
    get_or_build("matmul", &[&[32, 64], &[64, 32]], &[], DType::F32, || {
        total_builds += 1;
        Ok(build_real_matmul_def(32, 64, 32))
    })
    .expect("invariant: cache hit");
    get_or_build("add", &[&[32, 64], &[32, 64]], &[], DType::F32, || {
        total_builds += 1;
        Ok(build_real_matmul_def(32, 64, 32))
    })
    .expect("invariant: cache hit");
    get_or_build("gelu", &[&[32, 64]], &[], DType::F32, || {
        total_builds += 1;
        Ok(build_real_matmul_def(32, 64, 32))
    })
    .expect("invariant: cache hit");

    assert_eq!(total_builds, 3, "all 3 repeat calls should hit cache");
}

#[test]
fn bench_ir_build_cost_vs_cache_hit() {
    use std::time::Instant;

    clear_cache();

    let shapes: &[&[usize]] = &[&[512, 768], &[768, 3072]];

    // Cold: first call builds IR from TensorBlockBuilder
    let cold_start = Instant::now();
    let _def = get_or_build("matmul_bench", shapes, &[], DType::F32, || {
        Ok(build_real_matmul_def(512, 768, 3072))
    })
    .expect("invariant: matmul build succeeds");
    let cold_elapsed = cold_start.elapsed();

    // Warm: subsequent calls hit the cache (Arc::clone only)
    let runs: i32 = 100;
    let warm_start = Instant::now();
    for _ in 0..runs {
        let _def = get_or_build("matmul_bench", shapes, &[], DType::F32, || {
            Err(nn_core::TensorError::InvalidShape(
                "cache miss in warm loop — should never happen".into(),
            ))
        })
        .expect("invariant: cache hit in warm loop");
    }
    let warm_elapsed = warm_start.elapsed();

    let cold_us = cold_elapsed.as_micros();
    let warm_avg_us = warm_elapsed.as_micros() as f64 / f64::from(runs);

    // The cache hit should be faster than building IR
    assert!(
        warm_avg_us < cold_us as f64,
        "cache hit ({warm_avg_us:.1}us) should be faster than cold build ({cold_us}us)"
    );
}
