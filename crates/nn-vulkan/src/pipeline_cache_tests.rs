// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for the two-tier Vulkan pipeline cache.
//!
//! Tests cover: creation, stats tracking, L1/L2 behavior, LRU eviction,
//! cache key construction, hit rate calculation, thread safety, and
//! the `compile_or_cache` helper.

use super::*;
use crate::dispatch::{DescriptorBinding, DescriptorType};
use crate::spirv_emit::{SPIRV_MAGIC, SPIRV_VERSION_1_5};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn make_test_spirv() -> Vec<u32> {
    vec![SPIRV_MAGIC, SPIRV_VERSION_1_5, 0, 0, 0]
}

fn make_test_ds_layout() -> DescriptorSetLayout {
    DescriptorSetLayout::new(vec![DescriptorBinding {
        binding: 0,
        descriptor_type: DescriptorType::StorageBuffer,
        count: 1,
    }])
    .expect("ds layout")
}

fn make_test_pipeline(entry: &str) -> ComputePipeline {
    let spirv = make_test_spirv();
    let ds = make_test_ds_layout();
    let pl = PipelineLayout::new(&ds, 4).expect("pl");
    ComputePipeline::new(&spirv, entry, &pl).expect("pipeline")
}

fn make_cached_pipeline(src_hash: u64) -> CachedPipeline {
    CachedPipeline {
        pipeline: make_test_pipeline("main"),
        glsl_source_hash: src_hash,
    }
}

// ---------------------------------------------------------------------------
// Cache creation
// ---------------------------------------------------------------------------

#[test]
fn test_new_cache_is_empty() {
    let cache = PipelineCache::new();
    assert_eq!(cache.l1_len(), 0, "new cache should have 0 L1 entries");
}

#[test]
fn test_default_cache_is_empty() {
    let cache = PipelineCache::default();
    assert_eq!(cache.l1_len(), 0);
}

#[test]
fn test_new_and_default_produce_same_stats() {
    let a = PipelineCache::new();
    let b = PipelineCache::default();
    assert_eq!(a.stats(), b.stats());
}

// ---------------------------------------------------------------------------
// Stats initially zero
// ---------------------------------------------------------------------------

#[test]
fn test_stats_initially_zero() {
    let stats = PipelineCacheStats::default();
    assert_eq!(stats.l1_hits, 0);
    assert_eq!(stats.l2_hits, 0);
    assert_eq!(stats.misses, 0);
    assert_eq!(stats.inserts, 0);
    assert_eq!(stats.l1_evictions, 0);
}

#[test]
fn test_fresh_cache_stats_are_zero() {
    let cache = PipelineCache::new();
    let s = cache.stats();
    assert_eq!(s.l1_hits + s.l2_hits + s.misses + s.inserts + s.l1_evictions, 0);
}

// ---------------------------------------------------------------------------
// Cache miss counting
// ---------------------------------------------------------------------------

#[test]
fn test_miss_on_empty_cache() {
    let mut cache = PipelineCache::new();
    let key = pipeline_cache_key("missing_shader", "main");
    assert!(cache.get(key).is_none());
    assert_eq!(cache.stats().misses, 1);
    assert_eq!(cache.stats().l1_hits, 0);
    assert_eq!(cache.stats().l2_hits, 0);
}

#[test]
fn test_multiple_misses_accumulate() {
    let mut cache = PipelineCache::new();
    for i in 0..10 {
        let key = pipeline_cache_key(&format!("absent_{i}"), "main");
        assert!(cache.get(key).is_none());
    }
    assert_eq!(cache.stats().misses, 10);
}

// ---------------------------------------------------------------------------
// Cache hit counting (L1)
// ---------------------------------------------------------------------------

#[test]
fn test_l1_hit_after_insert() {
    let mut cache = PipelineCache::new();
    let key = pipeline_cache_key("nn_shader", "main");
    let pipeline = make_test_pipeline("main");
    cache.insert(key, pipeline, 42);

    assert!(cache.get(key).is_some());
    assert_eq!(cache.stats().l1_hits, 1);
    assert_eq!(cache.stats().misses, 0);
}

#[test]
fn test_repeated_l1_hits_accumulate() {
    let mut cache = PipelineCache::new();
    let key = pipeline_cache_key("shader_a", "main");
    cache.insert(key, make_test_pipeline("main"), 1);

    for _ in 0..5 {
        assert!(cache.get(key).is_some());
    }
    assert_eq!(cache.stats().l1_hits, 5);
}

// ---------------------------------------------------------------------------
// Hit rate calculation from stats
// ---------------------------------------------------------------------------

#[test]
fn test_hit_rate_from_stats() {
    let mut cache = PipelineCache::new();
    let key = pipeline_cache_key("rate_shader", "main");

    // 1 miss (initial lookup)
    assert!(cache.get(key).is_none());

    // Insert
    cache.insert(key, make_test_pipeline("main"), 99);

    // 3 hits
    for _ in 0..3 {
        assert!(cache.get(key).is_some());
    }

    let s = cache.stats();
    let total_lookups = s.l1_hits + s.l2_hits + s.misses;
    assert_eq!(total_lookups, 4); // 1 miss + 3 hits
    let hit_rate = (s.l1_hits + s.l2_hits) as f64 / total_lookups as f64;
    assert!((hit_rate - 0.75).abs() < 1e-9, "hit rate should be 75%");
}

#[test]
fn test_hit_rate_zero_when_all_misses() {
    let mut cache = PipelineCache::new();
    for i in 0..5 {
        let key = pipeline_cache_key(&format!("miss_{i}"), "main");
        cache.get(key);
    }
    let s = cache.stats();
    let total = s.l1_hits + s.l2_hits + s.misses;
    assert_eq!(total, 5);
    assert_eq!(s.l1_hits + s.l2_hits, 0);
}

// ---------------------------------------------------------------------------
// L1 (thread-local) cache behavior
// ---------------------------------------------------------------------------

#[test]
fn test_l1_len_tracks_inserts() {
    let mut cache = PipelineCache::new();
    for i in 0..5 {
        let key = pipeline_cache_key(&format!("len_shader_{i}"), "main");
        cache.insert(key, make_test_pipeline("main"), i as u64);
    }
    assert_eq!(cache.l1_len(), 5);
}

#[test]
fn test_l1_duplicate_insert_does_not_grow() {
    let mut cache = PipelineCache::new();
    let key = pipeline_cache_key("dup", "main");
    cache.insert(key, make_test_pipeline("main"), 1);
    cache.insert(key, make_test_pipeline("main"), 2);
    assert_eq!(cache.l1_len(), 1, "duplicate key should overwrite, not grow");
    assert_eq!(cache.stats().inserts, 2, "inserts counter still increments");
}

#[test]
fn test_l1_returns_correct_entry() {
    let mut cache = PipelineCache::new();
    let key = pipeline_cache_key("correct_entry", "main");
    cache.insert(key, make_test_pipeline("main"), 777);

    let cached = cache.get(key).expect("should hit");
    assert_eq!(cached.glsl_source_hash, 777);
    assert_eq!(cached.pipeline.entry_point(), "main");
}

// ---------------------------------------------------------------------------
// L2 (shared) cache behavior
// ---------------------------------------------------------------------------

#[test]
fn test_l2_receives_inserts() {
    let mut cache = PipelineCache::new();
    let key = pipeline_cache_key("l2_test_unique_34872", "main");
    cache.insert(key, make_test_pipeline("main"), 10);

    // L2 should have at least one entry (it's a global singleton, so it
    // accumulates from all tests -- we just verify it's non-empty after insert).
    assert!(cache.l2_len() > 0, "L2 should have entries after insert");
}

#[test]
fn test_l2_promotion_to_l1() {
    // Thread A inserts, then a fresh L1 cache (simulating thread B) should
    // find it in L2 and promote to L1.
    let key = pipeline_cache_key("l2_promote_unique_99123", "main");

    // Insert via one cache instance (populates L2).
    {
        let mut cache_a = PipelineCache::new();
        cache_a.insert(key, make_test_pipeline("main"), 55);
    }

    // Fresh cache (empty L1) should find it in L2.
    let mut cache_b = PipelineCache::new();
    let result = cache_b.get(key);
    assert!(result.is_some(), "L2 should serve the entry to a fresh L1");
    assert_eq!(cache_b.stats().l2_hits, 1, "should count as L2 hit");
    assert_eq!(cache_b.stats().l1_hits, 0, "first access should not be L1 hit");

    // Second access should be L1 hit (promoted).
    let result2 = cache_b.get(key);
    assert!(result2.is_some());
    assert_eq!(cache_b.stats().l1_hits, 1, "second access should be L1 hit after promotion");
}

// ---------------------------------------------------------------------------
// Cache key construction
// ---------------------------------------------------------------------------

#[test]
fn test_pipeline_cache_key_deterministic() {
    let k1 = pipeline_cache_key("void main() {}", "main");
    let k2 = pipeline_cache_key("void main() {}", "main");
    assert_eq!(k1, k2);
}

#[test]
fn test_pipeline_cache_key_different_source() {
    let k1 = pipeline_cache_key("void main() { float x = 1.0; }", "main");
    let k2 = pipeline_cache_key("void main() { float x = 2.0; }", "main");
    assert_ne!(k1, k2);
}

#[test]
fn test_pipeline_cache_key_different_entry_point() {
    let k1 = pipeline_cache_key("void main() {}", "main");
    let k2 = pipeline_cache_key("void main() {}", "kernel_main");
    assert_ne!(k1, k2);
}

#[test]
fn test_spirv_cache_key_deterministic() {
    let w = make_test_spirv();
    let k1 = spirv_cache_key(&w, "main");
    let k2 = spirv_cache_key(&w, "main");
    assert_eq!(k1, k2);
}

#[test]
fn test_spirv_cache_key_different_words() {
    let w1 = vec![SPIRV_MAGIC, SPIRV_VERSION_1_5, 0, 0, 0];
    let w2 = vec![SPIRV_MAGIC, SPIRV_VERSION_1_5, 0, 0, 1];
    let k1 = spirv_cache_key(&w1, "main");
    let k2 = spirv_cache_key(&w2, "main");
    assert_ne!(k1, k2);
}

#[test]
fn test_spirv_cache_key_different_entry() {
    let w = make_test_spirv();
    let k1 = spirv_cache_key(&w, "main");
    let k2 = spirv_cache_key(&w, "alt");
    assert_ne!(k1, k2);
}

#[test]
fn test_pipeline_and_spirv_keys_differ_for_same_content() {
    // Even if we hash the same logical content, the two key functions
    // hash different types (str vs [u32]) so they should differ.
    let k_pipeline = pipeline_cache_key("test", "main");
    let k_spirv = spirv_cache_key(&[0x74657374], "main"); // "test" as u32 -- different
    assert_ne!(k_pipeline, k_spirv);
}

// ---------------------------------------------------------------------------
// Different shader keys produce different entries
// ---------------------------------------------------------------------------

#[test]
fn test_different_shaders_stored_independently() {
    let mut cache = PipelineCache::new();
    let key_a = pipeline_cache_key("shader_alpha", "main");
    let key_b = pipeline_cache_key("shader_beta", "main");

    cache.insert(key_a, make_test_pipeline("main"), 100);
    cache.insert(key_b, make_test_pipeline("main"), 200);

    let a = cache.get(key_a).expect("key_a present");
    assert_eq!(a.glsl_source_hash, 100);

    let b = cache.get(key_b).expect("key_b present");
    assert_eq!(b.glsl_source_hash, 200);
}

#[test]
fn test_many_distinct_keys_all_retrievable() {
    let mut cache = PipelineCache::new();
    let n = 30;
    let keys: Vec<u64> = (0..n)
        .map(|i| pipeline_cache_key(&format!("unique_shader_{i}"), "main"))
        .collect();

    for (i, &k) in keys.iter().enumerate() {
        cache.insert(k, make_test_pipeline("main"), i as u64);
    }

    for (i, &k) in keys.iter().enumerate() {
        let cached = cache.get(k).expect("should be cached");
        assert_eq!(cached.glsl_source_hash, i as u64);
    }
}

// ---------------------------------------------------------------------------
// Eviction behavior (LRU on L1, capacity on L2)
// ---------------------------------------------------------------------------

#[test]
fn test_l1_eviction_at_capacity() {
    let mut cache = PipelineCache::new();

    // Fill L1 to capacity + extra.
    for i in 0..(LOCAL_MAX_ENTRIES + 10) {
        let key = pipeline_cache_key(&format!("evict_shader_{i}"), "main");
        cache.insert(key, make_test_pipeline("main"), i as u64);
    }

    assert!(
        cache.l1_len() <= LOCAL_MAX_ENTRIES,
        "L1 should not exceed LOCAL_MAX_ENTRIES ({}), got {}",
        LOCAL_MAX_ENTRIES,
        cache.l1_len()
    );
    assert!(
        cache.stats().l1_evictions >= 10,
        "should have evicted at least 10 entries, got {}",
        cache.stats().l1_evictions
    );
}

#[test]
fn test_l1_lru_evicts_oldest() {
    let mut cache = PipelineCache::new();

    // Insert entries 0..LOCAL_MAX_ENTRIES, filling L1 exactly.
    let mut keys = Vec::new();
    for i in 0..LOCAL_MAX_ENTRIES {
        let key = pipeline_cache_key(&format!("lru_shader_{i}"), "main");
        cache.insert(key, make_test_pipeline("main"), i as u64);
        keys.push(key);
    }
    assert_eq!(cache.l1_len(), LOCAL_MAX_ENTRIES);
    assert_eq!(cache.stats().l1_evictions, 0);

    // Insert one more -- should evict the LRU (first inserted, key 0).
    let extra_key = pipeline_cache_key("lru_shader_extra", "main");
    cache.insert(extra_key, make_test_pipeline("main"), 999);
    assert_eq!(cache.stats().l1_evictions, 1);

    // The extra entry should be in L1.
    assert!(cache.get(extra_key).is_some());
}

#[test]
fn test_l1_lru_touch_prevents_eviction() {
    let mut cache = PipelineCache::new();

    // Insert entries 0..LOCAL_MAX_ENTRIES.
    let mut keys = Vec::new();
    for i in 0..LOCAL_MAX_ENTRIES {
        let key = pipeline_cache_key(&format!("touch_shader_{i}"), "main");
        cache.insert(key, make_test_pipeline("main"), i as u64);
        keys.push(key);
    }

    // Touch entry 0 (oldest) to make it most-recently-used.
    cache.get(keys[0]);

    // Insert a new entry, which should evict entry 1 (now the LRU), not entry 0.
    let new_key = pipeline_cache_key("touch_shader_new", "main");
    cache.insert(new_key, make_test_pipeline("main"), 888);

    // Entry 0 should still be in L1 (was touched).
    // We can verify via L1 hit -- if it's there, we get l1_hit increment.
    let stats_before = cache.stats().l1_hits;
    let result = cache.get(keys[0]);
    assert!(result.is_some(), "touched entry should survive eviction");
    assert_eq!(
        cache.stats().l1_hits,
        stats_before + 1,
        "should be L1 hit, not L2"
    );
}

// ---------------------------------------------------------------------------
// Stats after many operations
// ---------------------------------------------------------------------------

#[test]
fn test_stats_consistency_after_mixed_operations() {
    let mut cache = PipelineCache::new();

    // 3 misses
    for i in 0..3 {
        cache.get(pipeline_cache_key(&format!("miss_{i}"), "main"));
    }

    // 2 inserts
    let key_x = pipeline_cache_key("shader_x", "main");
    let key_y = pipeline_cache_key("shader_y", "main");
    cache.insert(key_x, make_test_pipeline("main"), 1);
    cache.insert(key_y, make_test_pipeline("main"), 2);

    // 4 L1 hits
    for _ in 0..2 {
        cache.get(key_x);
        cache.get(key_y);
    }

    let s = cache.stats();
    assert_eq!(s.misses, 3);
    assert_eq!(s.inserts, 2);
    assert_eq!(s.l1_hits, 4);
    let total = s.l1_hits + s.l2_hits + s.misses;
    assert_eq!(total, 7, "total lookups = 3 misses + 4 hits");
}

#[test]
fn test_insert_count_independent_of_lookup_count() {
    let mut cache = PipelineCache::new();
    let key = pipeline_cache_key("count_shader", "main");

    cache.insert(key, make_test_pipeline("main"), 0);
    cache.insert(key, make_test_pipeline("main"), 0);
    cache.insert(key, make_test_pipeline("main"), 0);

    assert_eq!(cache.stats().inserts, 3);
    assert_eq!(cache.stats().l1_hits, 0, "no lookups performed");
    assert_eq!(cache.stats().misses, 0);
}

// ---------------------------------------------------------------------------
// compile_or_cache integration
// ---------------------------------------------------------------------------

#[test]
fn test_compile_or_cache_miss_then_hit() {
    let mut cache = PipelineCache::new();
    let glsl = "void main() { compile_or_cache_test; }";
    let spirv = make_test_spirv();
    let ds_layout = make_test_ds_layout();

    // First: miss + compile.
    let p1 = compile_or_cache(&mut cache, glsl, "main", &spirv, &ds_layout, 4)
        .expect("first compile");
    assert_eq!(cache.stats().misses, 1);
    assert_eq!(cache.stats().inserts, 1);

    // Second: hit.
    let p2 = compile_or_cache(&mut cache, glsl, "main", &spirv, &ds_layout, 4)
        .expect("cached");
    assert_eq!(cache.stats().l1_hits, 1);
    assert_eq!(p1.entry_point(), p2.entry_point());
}

#[test]
fn test_compile_or_cache_different_sources_produce_different_entries() {
    let mut cache = PipelineCache::new();
    let spirv = make_test_spirv();
    let ds_layout = make_test_ds_layout();

    let _p1 = compile_or_cache(&mut cache, "source_A", "main", &spirv, &ds_layout, 4)
        .expect("compile A");
    let _p2 = compile_or_cache(&mut cache, "source_B", "main", &spirv, &ds_layout, 4)
        .expect("compile B");

    assert_eq!(cache.stats().inserts, 2);
    assert_eq!(cache.stats().misses, 2);
    assert_eq!(cache.l1_len(), 2);
}

// ---------------------------------------------------------------------------
// Thread safety: concurrent lookups
// ---------------------------------------------------------------------------

#[test]
fn test_concurrent_inserts_via_shared_cache() {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let barrier = Arc::new(Barrier::new(4));
    let handles: Vec<_> = (0..4)
        .map(|t| {
            let bar = Arc::clone(&barrier);
            thread::spawn(move || {
                let mut cache = PipelineCache::new();
                bar.wait(); // synchronize start

                // Each thread inserts 10 unique pipelines.
                for i in 0..10 {
                    let key = pipeline_cache_key(
                        &format!("concurrent_shader_t{t}_i{i}"),
                        "main",
                    );
                    cache.insert(key, make_test_pipeline("main"), (t * 10 + i) as u64);
                }

                // Each thread should see its own entries.
                for i in 0..10 {
                    let key = pipeline_cache_key(
                        &format!("concurrent_shader_t{t}_i{i}"),
                        "main",
                    );
                    assert!(cache.get(key).is_some(), "thread {t} should find its own key {i}");
                }

                cache.stats()
            })
        })
        .collect();

    let all_stats: Vec<_> = handles.into_iter().map(|h| h.join().expect("thread")).collect();

    // Each thread should have 10 inserts.
    for (t, s) in all_stats.iter().enumerate() {
        assert_eq!(s.inserts, 10, "thread {t} should have 10 inserts");
    }
}

#[test]
fn test_cross_thread_l2_sharing() {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let unique = "cross_thread_l2_sharing_xz9012";
    let key = pipeline_cache_key(unique, "main");

    // Thread A inserts.
    {
        let mut cache = PipelineCache::new();
        cache.insert(key, make_test_pipeline("main"), 42);
    }

    // Threads B, C, D read from L2.
    let barrier = Arc::new(Barrier::new(3));
    let handles: Vec<_> = (0..3)
        .map(|_| {
            let bar = Arc::clone(&barrier);
            thread::spawn(move || {
                let mut cache = PipelineCache::new();
                bar.wait();
                let result = cache.get(key);
                (result.is_some(), cache.stats())
            })
        })
        .collect();

    for h in handles {
        let (found, stats) = h.join().expect("thread");
        assert!(found, "each thread should find the entry via L2");
        assert_eq!(stats.l2_hits, 1);
    }
}

// ---------------------------------------------------------------------------
// SharedPipelineCache eviction (L2 max capacity)
// ---------------------------------------------------------------------------

#[test]
fn test_shared_cache_eviction_under_pressure() {
    // We cannot easily isolate the global shared cache, but we can verify
    // it doesn't grow unbounded by inserting many entries.
    let mut cache = PipelineCache::new();
    for i in 0..600 {
        let key = pipeline_cache_key(&format!("shared_pressure_{i}"), "main");
        cache.insert(key, make_test_pipeline("main"), i as u64);
    }
    // SHARED_MAX_ENTRIES = 512. Even accounting for entries from other tests,
    // the shared cache should be bounded.
    assert!(
        cache.l2_len() <= SHARED_MAX_ENTRIES + 50,
        "L2 should be bounded near SHARED_MAX_ENTRIES, got {}",
        cache.l2_len()
    );
}

// ---------------------------------------------------------------------------
// CachedPipeline Clone
// ---------------------------------------------------------------------------

#[test]
fn test_cached_pipeline_clone_preserves_fields() {
    let original = make_cached_pipeline(12345);
    let cloned = original.clone();
    assert_eq!(cloned.glsl_source_hash, original.glsl_source_hash);
    assert_eq!(cloned.pipeline.entry_point(), original.pipeline.entry_point());
}

// ---------------------------------------------------------------------------
// PipelineCacheStats equality
// ---------------------------------------------------------------------------

#[test]
fn test_stats_equality() {
    let a = PipelineCacheStats {
        l1_hits: 5,
        l2_hits: 3,
        misses: 2,
        inserts: 4,
        l1_evictions: 1,
    };
    let b = PipelineCacheStats {
        l1_hits: 5,
        l2_hits: 3,
        misses: 2,
        inserts: 4,
        l1_evictions: 1,
    };
    assert_eq!(a, b);
}

#[test]
fn test_stats_inequality() {
    let a = PipelineCacheStats::default();
    let b = PipelineCacheStats {
        l1_hits: 1,
        ..PipelineCacheStats::default()
    };
    assert_ne!(a, b);
}

#[test]
fn test_stats_copy_semantics() {
    let a = PipelineCacheStats {
        l1_hits: 10,
        l2_hits: 5,
        misses: 3,
        inserts: 8,
        l1_evictions: 2,
    };
    let b = a; // Copy
    assert_eq!(a, b); // a still usable (Copy)
}
