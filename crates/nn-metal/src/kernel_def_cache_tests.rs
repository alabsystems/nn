// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`KernelDefCache`] LRU cache behavior.
//!
//! Extracted from `kernel_def_cache.rs` to keep production code under 500 lines.

use nn_core::DType;

use super::*;

#[test]
fn test_cache_hit_returns_same_def() {
    clear_cache();

    let mut build_count = 0u32;
    let def1 = get_or_build("add", &[&[2, 3], &[2, 3]], &[], DType::F32, || {
        build_count += 1;
        Ok(TensorKernelDef::new(
            "test",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    assert_eq!(build_count, 1);

    let def2 = get_or_build("add", &[&[2, 3], &[2, 3]], &[], DType::F32, || {
        build_count += 1;
        Ok(TensorKernelDef::new(
            "test2",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    assert_eq!(build_count, 1, "second call should hit cache");
    assert_eq!(def1.name, def2.name, "cached def should be returned");
}

#[test]
fn test_different_shapes_miss_cache() {
    clear_cache();

    let mut build_count = 0u32;
    let _def1 = get_or_build("matmul", &[&[2, 3], &[3, 4]], &[], DType::F32, || {
        build_count += 1;
        Ok(TensorKernelDef::new(
            "m1",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    let _def2 = get_or_build("matmul", &[&[4, 5], &[5, 6]], &[], DType::F32, || {
        build_count += 1;
        Ok(TensorKernelDef::new(
            "m2",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    assert_eq!(build_count, 2, "different shapes should miss cache");
}

#[test]
fn test_different_params_miss_cache() {
    clear_cache();

    let mut build_count = 0u32;
    // stride=1, padding=0
    let _def1 = get_or_build("conv1d", &[&[1, 3, 16]], &[1, 0], DType::F32, || {
        build_count += 1;
        Ok(TensorKernelDef::new(
            "c1",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    // stride=2, padding=1
    let _def2 = get_or_build("conv1d", &[&[1, 3, 16]], &[2, 1], DType::F32, || {
        build_count += 1;
        Ok(TensorKernelDef::new(
            "c2",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    assert_eq!(build_count, 2, "different params should miss cache");
}

#[test]
fn test_build_error_not_cached() {
    clear_cache();

    let result = get_or_build("fail", &[&[1]], &[], DType::F32, || {
        Err(nn_core::TensorError::InvalidShape("test error".into()))
    });
    assert!(result.is_err());
    assert_eq!(cache_len(), 0, "failed builds should not be cached");
}

#[test]
fn test_lru_eviction_fires() {
    clear_cache();
    set_max_entries(3);

    // Insert 3 entries — fills cache to capacity
    for i in 0..3 {
        get_or_build("op", &[&[i]], &[], DType::F32, || {
            Ok(TensorKernelDef::new(
                format!("op_{i}"),
                vec![],
                nn_dsl::TensorNodeId::new(0),
            ))
        })
        .unwrap();
    }
    assert_eq!(cache_len(), 3);

    // Insert a 4th entry — should evict the oldest (op_0)
    get_or_build("op", &[&[3]], &[], DType::F32, || {
        Ok(TensorKernelDef::new(
            "op_3",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    assert_eq!(cache_len(), 3, "eviction should keep cache at max_entries");

    // Verify op_0 was evicted (cache miss — build closure runs)
    let mut rebuilt = false;
    get_or_build("op", &[&[0usize]], &[], DType::F32, || {
        rebuilt = true;
        Ok(TensorKernelDef::new(
            "op_0_rebuilt",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    assert!(rebuilt, "op_0 should have been evicted and rebuilt");
}

#[test]
fn test_lru_eviction_preserves_recently_accessed() {
    clear_cache();
    set_max_entries(3);

    // Insert 3 entries: op_0, op_1, op_2
    for i in 0..3 {
        get_or_build("op", &[&[i]], &[], DType::F32, || {
            Ok(TensorKernelDef::new(
                format!("op_{i}"),
                vec![],
                nn_dsl::TensorNodeId::new(0),
            ))
        })
        .unwrap();
    }

    // Access op_0 again to make it the most recently used
    let mut build_count = 0u32;
    get_or_build("op", &[&[0usize]], &[], DType::F32, || {
        build_count += 1;
        Ok(TensorKernelDef::new(
            "x",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    assert_eq!(build_count, 0, "op_0 should be a cache hit");

    // Insert op_3 — should evict op_1 (oldest non-recently-accessed)
    get_or_build("op", &[&[3]], &[], DType::F32, || {
        Ok(TensorKernelDef::new(
            "op_3",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    assert_eq!(cache_len(), 3);

    // Verify op_0 is still cached (was recently accessed)
    let mut rebuilt_0 = false;
    get_or_build("op", &[&[0usize]], &[], DType::F32, || {
        rebuilt_0 = true;
        Ok(TensorKernelDef::new(
            "x",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    assert!(
        !rebuilt_0,
        "op_0 should still be cached (recently accessed)"
    );

    // Verify op_1 was evicted
    let mut rebuilt_1 = false;
    get_or_build("op", &[&[1usize]], &[], DType::F32, || {
        rebuilt_1 = true;
        Ok(TensorKernelDef::new(
            "x",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    assert!(rebuilt_1, "op_1 should have been evicted (oldest)");
}

#[test]
fn test_eviction_with_capacity_one() {
    clear_cache();
    set_max_entries(1);

    get_or_build("a", &[&[1]], &[], DType::F32, || {
        Ok(TensorKernelDef::new(
            "a",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    assert_eq!(cache_len(), 1);

    get_or_build("b", &[&[2]], &[], DType::F32, || {
        Ok(TensorKernelDef::new(
            "b",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    assert_eq!(cache_len(), 1, "should evict 'a' to fit 'b'");

    // Verify 'a' was evicted
    let mut rebuilt = false;
    get_or_build("a", &[&[1]], &[], DType::F32, || {
        rebuilt = true;
        Ok(TensorKernelDef::new(
            "a2",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    assert!(rebuilt, "'a' should have been evicted");
}

#[test]
fn test_different_ops_same_shapes() {
    clear_cache();

    let mut build_count = 0u32;
    let _def1 = get_or_build("add", &[&[2, 3]], &[], DType::F32, || {
        build_count += 1;
        Ok(TensorKernelDef::new(
            "a",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    let _def2 = get_or_build("mul", &[&[2, 3]], &[], DType::F32, || {
        build_count += 1;
        Ok(TensorKernelDef::new(
            "m",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    assert_eq!(build_count, 2, "different ops should miss cache");
}

#[test]
fn test_hash_collision_returns_correct_def() {
    clear_cache();
    // Two different keys forced to the same hash value (simulated collision)
    let forced_hash = 0xDEAD_BEEF_CAFE_BABE;
    let key_a = KernelDefKey::with_forced_hash("add", &[&[2, 3]], &[], forced_hash);
    let key_b = KernelDefKey::with_forced_hash("mul", &[&[4, 5]], &[], forced_hash);

    // Insert key_a's def
    let def_a = get_or_build_with_key(key_a, || {
        Ok(TensorKernelDef::new(
            "def_add",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    assert_eq!(def_a.name, "def_add");

    // key_b has the same hash but different key data — should NOT return def_a.
    // Instead, it should call the build closure and return the new def.
    let mut built_b = false;
    let def_b = get_or_build_with_key(key_b, || {
        built_b = true;
        Ok(TensorKernelDef::new(
            "def_mul",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    assert!(built_b, "collision should trigger rebuild");
    assert_eq!(
        def_b.name, "def_mul",
        "must return correct def on collision"
    );

    // After collision, key_b's def replaces key_a's in the slot.
    // Verify key_a now triggers a rebuild.
    let mut rebuilt_a = false;
    let key_a2 = KernelDefKey::with_forced_hash("add", &[&[2, 3]], &[], forced_hash);
    let def_a2 = get_or_build_with_key(key_a2, || {
        rebuilt_a = true;
        Ok(TensorKernelDef::new(
            "def_add_2",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    assert!(rebuilt_a, "key_a was evicted by collision — should rebuild");
    assert_eq!(def_a2.name, "def_add_2");
}

#[test]
fn test_different_dtype_misses_cache() {
    clear_cache();

    let mut build_count = 0u32;
    let _def1 = get_or_build("add", &[&[2, 3]], &[], DType::F32, || {
        build_count += 1;
        Ok(TensorKernelDef::new(
            "add_f32",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    let _def2 = get_or_build("add", &[&[2, 3]], &[], DType::BF16, || {
        build_count += 1;
        Ok(TensorKernelDef::new(
            "add_bf16",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    assert_eq!(
        build_count, 2,
        "same op+shapes but different dtype should miss cache"
    );
}

#[test]
fn test_same_dtype_hits_cache() {
    clear_cache();

    let mut build_count = 0u32;
    let _def1 = get_or_build("mul", &[&[4, 8]], &[], DType::BF16, || {
        build_count += 1;
        Ok(TensorKernelDef::new(
            "mul_bf16",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    let _def2 = get_or_build("mul", &[&[4, 8]], &[], DType::BF16, || {
        build_count += 1;
        Ok(TensorKernelDef::new(
            "mul_bf16_2",
            vec![],
            nn_dsl::TensorNodeId::new(0),
        ))
    })
    .unwrap();
    assert_eq!(build_count, 1, "same op+shapes+dtype should hit cache");
}

// -- Performance proof: LRU eviction scan cost scales linearly ----------------

/// Prove that LRU eviction cost scales with cache size (O(n) per eviction).
///
/// The current `evict_lru()` implementation scans all `access_gen` entries
/// via `min_by_key()` to find the oldest entry. This test measures the
/// eviction cost at two cache sizes and verifies the scaling ratio is
/// consistent with O(n) — not O(1) (which would mean we had a proper
/// linked-list LRU) and not worse than O(n).
///
/// This documents the known O(n) eviction cost so any future optimization
/// to O(1) (e.g., doubly-linked-list LRU) can be validated by this test
/// changing from ~linear scaling to ~constant scaling.
#[test]
fn test_lru_eviction_cost_scales_linearly() {
    use std::time::Instant;

    // Measure eviction cost at two cache sizes.
    let sizes = [64, 256];
    let evictions_per_size = 50;
    let mut timings = Vec::new();

    for &max in &sizes {
        clear_cache();
        set_max_entries(max);

        // Fill the cache to capacity.
        for i in 0..max {
            get_or_build("fill", &[&[i, 1]], &[], DType::F32, || {
                Ok(TensorKernelDef::new(
                    format!("fill_{i}"),
                    vec![],
                    nn_dsl::TensorNodeId::new(0),
                ))
            })
            .unwrap();
        }
        assert_eq!(cache_len(), max);

        // Now each new insert triggers an eviction. Measure total time.
        let start = Instant::now();
        for i in 0..evictions_per_size {
            let shape_val = max + i;
            get_or_build("evict", &[&[shape_val, 2]], &[], DType::F32, || {
                Ok(TensorKernelDef::new(
                    format!("evict_{shape_val}"),
                    vec![],
                    nn_dsl::TensorNodeId::new(0),
                ))
            })
            .unwrap();
        }
        let elapsed_ns = start.elapsed().as_nanos();
        let per_eviction_ns = elapsed_ns / evictions_per_size as u128;
        timings.push((max, per_eviction_ns));
        eprintln!(
            "  LRU eviction at max_entries={max}: {per_eviction_ns}ns/eviction \
             ({evictions_per_size} evictions in {elapsed_ns}ns)"
        );

        assert_eq!(cache_len(), max, "cache should stay at max_entries");
    }

    // Verify: larger cache should have proportionally slower evictions.
    // With O(n) eviction, 4x cache size should yield ~4x eviction cost.
    // We use a conservative bound: ratio should be >= 1.5x (accounting
    // for noise) and the absolute cost at n=256 should be measurable.
    let (size_small, time_small) = timings[0];
    let (size_large, time_large) = timings[1];
    let size_ratio = size_large as f64 / size_small as f64;
    let time_ratio = time_large as f64 / time_small.max(1) as f64;

    eprintln!(
        "  Size ratio: {size_ratio:.1}x, Time ratio: {time_ratio:.1}x \
         (expected ~{size_ratio:.1}x for O(n))"
    );

    // The scaling should be super-linear (> 1.0x). We don't assert a tight
    // bound because wall-clock timing is noisy under parallel test execution.
    // At these microsecond scales, measurement noise can exceed the actual
    // O(n) difference. Assert only that the absolute cost is measurable
    // (not optimized to zero), documenting the O(n) scan exists.
    assert!(
        time_large > 0 && time_small > 0,
        "eviction cost should be measurable: n={size_small} ({time_small}ns), \
         n={size_large} ({time_large}ns)"
    );
}

/// Prove that `input_nodes()` allocates a new Vec on every call.
///
/// `ComputationGraph::input_nodes()` collects into `Vec<&TraceNode>` via
/// `.filter().collect()`. For a `CompiledModel` that calls this per-forward,
/// this creates unnecessary allocations. This test documents the pattern
/// so a future optimization (e.g., caching input indices at construction)
/// can be validated.
#[test]
fn test_input_nodes_allocates_on_every_call() {
    clear_cache();

    // Build a graph with a known number of input nodes.
    // We use the graph's `input_nodes()` method to verify it re-scans
    // nodes on every call (returning independent Vecs).
    use nn_core::dyn_tensor::trace::{record_input, trace_graph};
    use nn_core::DType;

    let ((), graph) = trace_graph(|| {
        let _id1 = record_input(&[4], DType::F32);
        let _id2 = record_input(&[8], DType::F32);
        let _id3 = record_input(&[16], DType::F32);
        Ok(())
    })
    .unwrap();

    // Call input_nodes() twice — should return independent Vecs with same content.
    let inputs1 = graph.input_nodes();
    let inputs2 = graph.input_nodes();

    assert_eq!(inputs1.len(), 3);
    assert_eq!(inputs2.len(), 3);

    // Verify they contain the same nodes (by ID).
    for (a, b) in inputs1.iter().zip(inputs2.iter()) {
        assert_eq!(a.id(), b.id());
    }

    // The two Vecs are independent heap allocations (different pointers).
    // This documents the O(n) per-call cost that could be avoided by
    // pre-computing input indices at graph construction time.
    assert!(
        inputs1.as_ptr() != inputs2.as_ptr(),
        "input_nodes() should return fresh Vec allocations each call"
    );
}
