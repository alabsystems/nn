// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LRU eviction tests for `PipelineCache`.
//!
//! Split from `dispatch_cache.rs` to stay under 500-line limit.
//! Tests `with_capacity()`, eviction ordering, and promote behavior.
//!
//! Part of #874 AC6.

use nn_metal::{KernelSource, MetalContext, PipelineCache};

/// Three distinct MSL kernels for LRU eviction tests. Each has a unique entry
/// point so they produce distinct `KernelSource` keys.
const TRIPLE_MSL: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    kernel void kernel_a(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint& total [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < total) { output[id] = input[id] + 1.0; }
    }

    kernel void kernel_b(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint& total [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < total) { output[id] = input[id] + 2.0; }
    }

    kernel void kernel_c(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant uint& total [[buffer(2)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id < total) { output[id] = input[id] + 3.0; }
    }
"#;

/// Verify with_capacity creates a cache with the specified max entries.
#[test]
fn test_cache_with_capacity() {
    let context = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::with_capacity(context, 2);
    assert_eq!(cache.max_entries(), 2);
    assert!(cache.is_empty());
}

/// Verify that inserting beyond capacity evicts the oldest entry.
///
/// Insert A, B (capacity=2, both cached). Insert C — A should be evicted
/// because it was inserted first and never promoted.
#[test]
fn test_cache_evicts_oldest_at_capacity() {
    let context = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::with_capacity(context, 2);
    let src_a = KernelSource::new(TRIPLE_MSL, "kernel_a");
    let src_b = KernelSource::new(TRIPLE_MSL, "kernel_b");
    let src_c = KernelSource::new(TRIPLE_MSL, "kernel_c");

    // Insert A and B.
    cache.get_or_compile(&src_a).expect("compile A");
    cache.get_or_compile(&src_b).expect("compile B");
    assert_eq!(cache.len(), 2, "cache should have 2 entries at capacity");

    // Insert C — should evict A (oldest).
    cache.get_or_compile(&src_c).expect("compile C");
    assert_eq!(
        cache.len(),
        2,
        "cache should stay at capacity after eviction"
    );

    // Re-insert A — recompiles (was evicted). Evicts B (now oldest).
    cache.get_or_compile(&src_a).expect("recompile A");
    assert_eq!(cache.len(), 2, "cache should still be at capacity");
}

/// Verify that promote() changes eviction order: accessing A after B makes B
/// the eviction candidate instead of A.
///
/// Sequence: insert A, insert B, access A (promote), insert C.
/// Expected: B is evicted (oldest non-promoted), A and C remain.
#[test]
fn test_cache_promote_prevents_eviction() {
    let context = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::with_capacity(context, 2);
    let src_a = KernelSource::new(TRIPLE_MSL, "kernel_a");
    let src_b = KernelSource::new(TRIPLE_MSL, "kernel_b");
    let src_c = KernelSource::new(TRIPLE_MSL, "kernel_c");

    // Insert A, then B.
    cache.get_or_compile(&src_a).expect("compile A");
    cache.get_or_compile(&src_b).expect("compile B");

    // Access A — promotes it to most-recently-used.
    cache.get_or_compile(&src_a).expect("hit A (promote)");
    assert_eq!(cache.len(), 2, "promote should not change size");

    // Insert C — B should be evicted (it is now the oldest).
    cache.get_or_compile(&src_c).expect("compile C");
    assert_eq!(cache.len(), 2);

    // Insert B again — A is now oldest (C was just inserted, A was promoted
    // before C). A should be evicted.
    cache.get_or_compile(&src_b).expect("recompile B");
    assert_eq!(cache.len(), 2, "eviction should maintain capacity");
}

/// Verify that a capacity-1 cache correctly evicts on every new insert.
#[test]
fn test_cache_capacity_one_evicts_on_every_insert() {
    let context = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::with_capacity(context, 1);
    let src_a = KernelSource::new(TRIPLE_MSL, "kernel_a");
    let src_b = KernelSource::new(TRIPLE_MSL, "kernel_b");

    cache.get_or_compile(&src_a).expect("compile A");
    assert_eq!(cache.len(), 1);

    cache.get_or_compile(&src_b).expect("compile B");
    assert_eq!(
        cache.len(),
        1,
        "capacity-1 cache should evict A when B is inserted"
    );

    // Re-insert A — B should be evicted.
    cache.get_or_compile(&src_a).expect("recompile A");
    assert_eq!(
        cache.len(),
        1,
        "capacity-1 cache should evict B when A is re-inserted"
    );
}

/// Verify that repeated access to the same key does not grow the cache.
#[test]
fn test_cache_repeated_access_same_key() {
    let context = MetalContext::new().expect("Metal context");
    let cache = PipelineCache::with_capacity(context, 2);
    let src_a = KernelSource::new(TRIPLE_MSL, "kernel_a");

    for _ in 0..10 {
        cache.get_or_compile(&src_a).expect("compile/hit A");
    }
    assert_eq!(
        cache.len(),
        1,
        "repeated access to same key should not grow cache"
    );
}
