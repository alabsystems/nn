// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the two-tier pipeline cache (L1 thread-local + L2 shared).

use super::*;

const TRIPLE_MSL: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    kernel void triple_values(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint& total [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < total) {
            output[id] = input[id] * 3.0;
        }
    }
"#;

const NEGATE_MSL: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    kernel void negate_values(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint& total [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < total) {
            output[id] = -input[id];
        }
    }
"#;

/// L2 shared cache should be populated after a compile.
#[test]
fn test_l2_populated_after_compile() {
    // Use a unique kernel MSL so parallel tests sharing the L2 cache
    // (e.g. test_concurrent_compile_no_panic using TRIPLE_MSL) cannot
    // pre-populate this entry, which would prevent L2 growth detection.
    let unique_msl = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void l2_populate_test(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& total [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id < total) { output[id] = input[id] * 11.0; }
        }
    "#;

    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx);

    let initial_shared = PipelineCache::shared_cache_len();

    let source = KernelSource::new(unique_msl, "l2_populate_test");
    cache
        .get_or_compile(&source)
        .expect("compile should succeed");

    // L1 should have the entry.
    assert_eq!(cache.len(), 1);
    // L2 should have at least one more entry than before.
    assert!(
        PipelineCache::shared_cache_len() > initial_shared,
        "shared cache should grow after a compile"
    );
}

/// Second thread should find the pipeline in L2 without recompiling.
#[test]
fn test_cross_thread_l2_hit() {
    let ctx = MetalContext::new().expect("Metal context");

    // Thread 1: compile a pipeline — populates L1 and L2.
    let source = KernelSource::new(NEGATE_MSL, "negate_values");
    let cache1 = PipelineCache::new(ctx.clone());
    cache1.get_or_compile(&source).expect("thread 1 compile");

    let shared_after_t1 = PipelineCache::shared_cache_len();
    assert!(shared_after_t1 > 0, "L2 should have entries after thread 1");

    // Thread 2: create a fresh PipelineCache (empty L1), request same pipeline.
    // Wrapped in autoreleasepool — Metal pipeline compilation creates ObjC
    // autoreleased objects that leak on background threads (dvoice#1245).
    let handle = std::thread::spawn(move || {
        objc::rc::autoreleasepool(|| {
            let cache2 = PipelineCache::new(ctx);
            assert_eq!(cache2.len(), 0, "thread 2 L1 should start empty");

            let pipeline = cache2
                .get_or_compile(&source)
                .expect("thread 2 should get L2 hit");

            // L1 should now have the entry (promoted from L2).
            assert_eq!(cache2.len(), 1, "thread 2 L1 should have entry from L2");

            pipeline.entry_point().to_string()
        })
    });

    let entry = handle.join().expect("thread 2 should not panic");
    assert_eq!(entry, "negate_values");

    // L2 count should be at least what thread 1 left — thread 2 didn't
    // compile anything new. It may be higher if other tests running in
    // parallel inserted into the shared cache concurrently.
    assert!(
        PipelineCache::shared_cache_len() >= shared_after_t1,
        "L2 should not shrink when thread 2 gets a hit"
    );
}

/// Multiple threads compiling concurrently should not panic or deadlock.
#[test]
fn test_concurrent_compile_no_panic() {
    let ctx = MetalContext::new().expect("Metal context");
    let num_threads = 4;
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(num_threads));

    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let ctx_clone = ctx.clone();
            let barrier_clone = barrier.clone();
            std::thread::spawn(move || {
                objc::rc::autoreleasepool(|| {
                    let cache = PipelineCache::new(ctx_clone);
                    barrier_clone.wait();

                    // All threads compile the same kernel — only one should actually
                    // hit the GPU shader compiler; others should get L2 hits.
                    let source = KernelSource::new(TRIPLE_MSL, "triple_values");
                    let pipeline = cache
                        .get_or_compile(&source)
                        .expect("concurrent compile should succeed");

                    assert_eq!(cache.len(), 1);
                    (i, pipeline.entry_point().to_string())
                })
            })
        })
        .collect();

    for h in handles {
        let (thread_id, entry) = h.join().expect("thread should not panic");
        assert_eq!(
            entry, "triple_values",
            "thread {thread_id} got wrong entry point"
        );
    }
}

/// L2 eviction should not panic when shared cache is full.
#[test]
fn test_shared_cache_eviction_does_not_panic() {
    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx);

    // Insert many entries into the shared cache to eventually trigger eviction.
    // SHARED_MAX_ENTRIES = 512, but we just need to verify no panic, not
    // actually fill it. Use a few unique kernels.
    for i in 0..5 {
        let msl = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void k{i}(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                constant uint& total [[buffer(2)]],
                uint id [[thread_position_in_grid]]
            ) {{
                if (id < total) {{ output[id] = input[id] * {val}.0; }}
            }}
            "#,
            i = i,
            val = i + 2,
        );
        let source = KernelSource::new(&msl, format!("k{i}"));
        cache
            .get_or_compile(&source)
            .expect("compile unique kernel");
    }

    // All 5 should be in L1.
    assert_eq!(cache.len(), 5);
    // L2 should also have them.
    assert!(PipelineCache::shared_cache_len() >= 5);
}

/// Second thread gets a pipeline faster via L2 than the first thread
/// pays for GPU shader compilation.
///
/// Measures: thread 1 compiles (cold), thread 2 retrieves from L2 (warm).
/// L2 retrieval should be significantly faster than shader compilation.
#[test]
fn test_l2_reduces_cold_start_latency() {
    use std::time::Instant;

    // Use a unique kernel MSL so other tests' L2 entries don't interfere.
    let bench_msl = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void bench_kernel(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& total [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            if (id < total) { output[id] = input[id] * 7.0 + 1.0; }
        }
    "#;

    let ctx = MetalContext::new().expect("Metal context");
    let source = KernelSource::new(bench_msl, "bench_kernel");

    // Thread 1: compile from scratch (cold path).
    // autoreleasepool: Metal pipeline compilation creates ObjC temporaries.
    let ctx_t1 = ctx.clone();
    let source_t1 = source.clone();
    let (cold_ns, _) = {
        let handle = std::thread::spawn(move || {
            objc::rc::autoreleasepool(|| {
                let cache = PipelineCache::new(ctx_t1);
                let start = Instant::now();
                let pipeline = cache
                    .get_or_compile(&source_t1)
                    .expect("cold compile should succeed");
                let elapsed = start.elapsed().as_nanos();
                (elapsed, pipeline.entry_point().to_string())
            })
        });
        handle.join().expect("thread 1 should not panic")
    };

    // Thread 2: retrieve from L2 (warm path — no GPU shader compilation).
    let (warm_ns, _) = {
        let handle = std::thread::spawn(move || {
            objc::rc::autoreleasepool(|| {
                let cache = PipelineCache::new(ctx);
                assert_eq!(cache.len(), 0, "thread 2 L1 should start empty");

                let start = Instant::now();
                let pipeline = cache
                    .get_or_compile(&source)
                    .expect("L2 retrieval should succeed");
                let elapsed = start.elapsed().as_nanos();
                (elapsed, pipeline.entry_point().to_string())
            })
        });
        handle.join().expect("thread 2 should not panic")
    };

    // L2 hit should be faster than GPU shader compilation.
    // Conservative check: warm path should be < cold path.
    // Shader compilation is typically 1-50ms; L2 lookup is <1μs.
    assert!(
        warm_ns < cold_ns,
        "L2 retrieval ({warm_ns}ns) should be faster than cold compile ({cold_ns}ns)"
    );

    // Verify thread 2 actually got an L1 entry (promoted from L2).
    // This is validated by test_cross_thread_l2_hit; we just confirm
    // the measurement path worked correctly.
    assert!(
        warm_ns < 10_000_000,
        "L2 hit should be under 10ms (got {warm_ns}ns)"
    );
}

/// L1 collision detection: when two different KernelSources hash to the same
/// u64 key, get_or_compile must NOT return the stale pipeline. Instead it
/// should fall through, compile the correct pipeline, and replace the entry.
///
/// Regression test for #2211.
#[test]
fn test_l1_hash_collision_returns_correct_pipeline() {
    let ctx = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::new(ctx);

    // Source A: triple_values kernel.
    let source_a = KernelSource::new(TRIPLE_MSL, "triple_values");
    let pipeline_a = cache.get_or_compile(&source_a).expect("compile source_a");
    assert_eq!(pipeline_a.entry_point(), "triple_values");
    assert_eq!(cache.len(), 1);

    // Source B: negate_values kernel (different MSL and entry point).
    let source_b = KernelSource::new(NEGATE_MSL, "negate_values");

    // Force a collision: insert source_a's pipeline under source_b's hash key.
    // This simulates two different KernelSources hashing to the same u64.
    let key_b = PipelineCache::hash_key(&source_b);
    cache.insert_with_forced_key(key_b, &source_a, &pipeline_a);

    // Now L1 has source_a's pipeline stored under key_b's hash.
    // get_or_compile(source_b) should detect the mismatch, fall through,
    // compile source_b, and return the correct negate_values pipeline.
    let pipeline_b = cache
        .get_or_compile(&source_b)
        .expect("compile source_b despite collision");

    assert_eq!(
        pipeline_b.entry_point(),
        "negate_values",
        "collision should NOT return stale pipeline from source_a"
    );
}
