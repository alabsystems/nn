// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`SharedSegmentStore`].

use std::sync::{Arc, Mutex};

use super::*;
use crate::compiled_model::CompiledModel;

/// Helper: create a `SharedCompiledSegment` from an empty `CompiledModel`.
fn empty_segment() -> SharedCompiledSegment {
    let model = CompiledModel::empty();
    SharedCompiledSegment::new(model.share_def())
}

/// Helper: create a segment with a specific tracked byte size.
/// Uses `CompiledModel::empty()` (which has `buffer_plan.total_bytes = 0`)
/// then overrides `tracked_bytes` for budget-testing purposes.
fn segment_with_bytes(bytes: usize) -> SharedCompiledSegment {
    let model = CompiledModel::empty();
    SharedCompiledSegment {
        def: model.share_def(),
        tracked_bytes: bytes,
    }
}

// -- Construction tests --

#[test]
fn test_new_store_is_empty() {
    let store = SharedSegmentStore::new(1024);
    assert_eq!(store.len(), 0);
    assert!(store.is_empty());
    assert_eq!(store.total_bytes(), 0);
    assert_eq!(store.byte_budget(), 1024);
    assert_eq!(store.max_entries(), DEFAULT_MAX_ENTRIES);
}

#[test]
fn test_with_capacity_custom() {
    let store = SharedSegmentStore::with_capacity(2048, 16);
    assert_eq!(store.byte_budget(), 2048);
    assert_eq!(store.max_entries(), 16);
    assert!(store.is_empty());
}

#[test]
#[should_panic(expected = "byte_budget must be > 0")]
fn test_new_zero_budget_panics() {
    let _store = SharedSegmentStore::new(0);
}

#[test]
#[should_panic(expected = "max_entries must be > 0")]
fn test_with_capacity_zero_entries_panics() {
    let _store = SharedSegmentStore::with_capacity(1024, 0);
}

// -- Insert and retrieve tests --

#[test]
fn test_insert_and_get() {
    let mut store = SharedSegmentStore::new(1024 * 1024);
    let key = SegmentKey::new(4, vec![1, 48000]);
    let segment = empty_segment();
    store.insert(key.clone(), segment);

    assert_eq!(store.len(), 1);
    assert!(!store.is_empty());

    let result = store.get(&key);
    assert!(result.is_some(), "should find inserted key");
}

#[test]
fn test_get_miss_returns_none() {
    let mut store = SharedSegmentStore::new(1024 * 1024);
    let key = SegmentKey::new(4, vec![1, 48000]);
    store.insert(key, empty_segment());

    let missing = SegmentKey::new(4, vec![1, 96000]);
    assert!(store.get(&missing).is_none(), "different shape should miss");
}

#[test]
fn test_different_segment_kinds_coexist() {
    let mut store = SharedSegmentStore::new(1024 * 1024);
    let key_a = SegmentKey::new(0, vec![1, 128]);
    let key_b = SegmentKey::new(4, vec![1, 128]);
    store.insert(key_a.clone(), empty_segment());
    store.insert(key_b.clone(), empty_segment());

    assert_eq!(store.len(), 2);
    assert!(store.get(&key_a).is_some());
    assert!(store.get(&key_b).is_some());
}

#[test]
fn test_same_kind_different_shapes_coexist() {
    let mut store = SharedSegmentStore::new(1024 * 1024);
    let key_a = SegmentKey::new(4, vec![1, 48000]);
    let key_b = SegmentKey::new(4, vec![1, 96000]);
    store.insert(key_a.clone(), empty_segment());
    store.insert(key_b.clone(), empty_segment());

    assert_eq!(store.len(), 2);
    assert!(store.get(&key_a).is_some());
    assert!(store.get(&key_b).is_some());
}

#[test]
fn test_insert_same_key_replaces() {
    let mut store = SharedSegmentStore::new(1024 * 1024);
    let key = SegmentKey::new(4, vec![1, 48000]);
    let seg1 = empty_segment();
    let seg2 = empty_segment();

    let def1 = store.insert(key.clone(), seg1);
    let def2 = store.insert(key, seg2);

    assert_eq!(store.len(), 1, "duplicate key should not increase count");
    // defs should be different Arc allocations (from different empty() calls)
    assert!(
        !Arc::ptr_eq(&def1, &def2),
        "replacement should use the new segment"
    );
}

// -- LRU eviction tests --

#[test]
fn test_lru_eviction_at_max_entries() {
    let mut store = SharedSegmentStore::with_capacity(1024 * 1024 * 1024, 2);
    let key_a = SegmentKey::new(0, vec![10]);
    let key_b = SegmentKey::new(1, vec![20]);
    let key_c = SegmentKey::new(2, vec![30]);

    store.insert(key_a.clone(), empty_segment());
    store.insert(key_b.clone(), empty_segment());
    assert_eq!(store.len(), 2);

    // Insert third — should evict key_a (LRU = back)
    store.insert(key_c.clone(), empty_segment());
    assert_eq!(store.len(), 2);
    assert!(store.get(&key_a).is_none(), "key_a should be evicted");
    assert!(store.get(&key_b).is_some());
    assert!(store.get(&key_c).is_some());
}

#[test]
fn test_lru_promotion_on_get() {
    let mut store = SharedSegmentStore::with_capacity(1024 * 1024 * 1024, 2);
    let key_a = SegmentKey::new(0, vec![10]);
    let key_b = SegmentKey::new(1, vec![20]);
    let key_c = SegmentKey::new(2, vec![30]);

    store.insert(key_a.clone(), empty_segment());
    store.insert(key_b.clone(), empty_segment());

    // Access key_a to promote it to MRU
    store.get(&key_a);

    // Now key_b is LRU. Insert key_c should evict key_b.
    store.insert(key_c, empty_segment());
    assert!(
        store.get(&key_a).is_some(),
        "key_a was promoted, should survive"
    );
    assert!(store.get(&key_b).is_none(), "key_b was LRU, should be evicted");
}

#[test]
fn test_byte_budget_eviction() {
    // Budget = 100 bytes. Insert entries that are 60 bytes each.
    let mut store = SharedSegmentStore::with_capacity(100, 64);
    let key_a = SegmentKey::new(0, vec![10]);
    let key_b = SegmentKey::new(1, vec![20]);
    let key_c = SegmentKey::new(2, vec![30]);

    store.insert(key_a.clone(), segment_with_bytes(60));
    assert_eq!(store.total_bytes(), 60);
    assert_eq!(store.len(), 1);

    // Second insert: 60 + 60 = 120 > 100 → evict key_a
    store.insert(key_b.clone(), segment_with_bytes(60));
    assert_eq!(store.len(), 1, "byte budget should force eviction");
    assert_eq!(store.total_bytes(), 60);
    assert!(store.get(&key_a).is_none(), "key_a should be evicted by budget");
    assert!(store.get(&key_b).is_some());

    // Third insert: still within budget after eviction
    store.insert(key_c, segment_with_bytes(30));
    assert_eq!(store.len(), 2);
    assert_eq!(store.total_bytes(), 90);
}

#[test]
fn test_single_oversized_entry_still_stored() {
    // Budget = 10 bytes, but a single 100-byte entry should still be stored.
    let mut store = SharedSegmentStore::with_capacity(10, 64);
    let key = SegmentKey::new(0, vec![1]);
    store.insert(key.clone(), segment_with_bytes(100));

    assert_eq!(store.len(), 1, "single oversized entry should still be stored");
    assert_eq!(store.total_bytes(), 100);
    assert!(store.get(&key).is_some());
}

// -- evict_lru tests --

#[test]
fn test_evict_lru_empty_store() {
    let mut store = SharedSegmentStore::new(1024);
    assert!(!store.evict_lru(), "evict on empty store should return false");
}

#[test]
fn test_evict_lru_removes_oldest() {
    let mut store = SharedSegmentStore::new(1024 * 1024);
    let key_a = SegmentKey::new(0, vec![10]);
    let key_b = SegmentKey::new(1, vec![20]);

    store.insert(key_a.clone(), segment_with_bytes(50));
    store.insert(key_b.clone(), segment_with_bytes(30));

    assert!(store.evict_lru(), "should evict successfully");
    assert_eq!(store.len(), 1);
    assert!(store.get(&key_a).is_none(), "key_a (LRU) should be evicted");
    assert!(store.get(&key_b).is_some(), "key_b (MRU) should survive");
    assert_eq!(store.total_bytes(), 30);
}

// -- Stats tests --

#[test]
fn test_stats_initial() {
    let store = SharedSegmentStore::new(1024);
    let stats = store.stats();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
    assert_eq!(stats.evictions, 0);
    assert_eq!(stats.total_bytes, 0);
    assert_eq!(stats.entry_count, 0);
    assert_eq!(stats.hit_rate(), 0.0);
}

#[test]
fn test_stats_after_operations() {
    let mut store = SharedSegmentStore::with_capacity(100, 2);
    let key_a = SegmentKey::new(0, vec![10]);
    let key_b = SegmentKey::new(1, vec![20]);
    let key_c = SegmentKey::new(2, vec![30]);

    store.insert(key_a.clone(), empty_segment());
    store.insert(key_b, empty_segment());

    // Hit
    store.get(&key_a);
    // Miss
    store.get(&key_c);
    // Eviction (key_b is LRU after key_a was promoted)
    store.insert(key_c.clone(), empty_segment());

    let stats = store.stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.evictions, 1);
    assert_eq!(stats.entry_count, 2);
    assert_eq!(stats.hit_rate(), 0.5);
}

#[test]
fn test_stats_eviction_count_from_byte_budget() {
    let mut store = SharedSegmentStore::with_capacity(50, 64);
    store.insert(SegmentKey::new(0, vec![1]), segment_with_bytes(30));
    store.insert(SegmentKey::new(1, vec![2]), segment_with_bytes(30));
    // Second insert evicts first due to byte budget

    let stats = store.stats();
    assert_eq!(stats.evictions, 1);
}

// -- Concurrent access test (Arc<Mutex> safety) --

#[test]
fn test_concurrent_access_arc_mutex() {
    let store = Arc::new(Mutex::new(SharedSegmentStore::new(1024 * 1024)));

    // Insert from "thread 1" context
    {
        let mut guard = store.lock().expect("mutex should not be poisoned");
        guard.insert(SegmentKey::new(0, vec![128]), empty_segment());
    }

    // Read from "thread 2" context
    {
        let mut guard = store.lock().expect("mutex should not be poisoned");
        let result = guard.get(&SegmentKey::new(0, vec![128]));
        assert!(result.is_some(), "should find entry inserted by other lock scope");
    }

    // Stats should reflect both operations
    {
        let guard = store.lock().expect("mutex should not be poisoned");
        let stats = guard.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.entry_count, 1);
    }
}

// -- contains_key tests --

#[test]
fn test_contains_key() {
    let mut store = SharedSegmentStore::new(1024 * 1024);
    let key = SegmentKey::new(4, vec![1, 48000]);
    assert!(!store.contains_key(&key));

    store.insert(key.clone(), empty_segment());
    assert!(store.contains_key(&key));
}

// -- clear tests --

#[test]
fn test_clear() {
    let mut store = SharedSegmentStore::new(1024 * 1024);
    store.insert(SegmentKey::new(0, vec![1]), segment_with_bytes(50));
    store.insert(SegmentKey::new(1, vec![2]), segment_with_bytes(30));

    // Generate some stats
    let _ = store.get(&SegmentKey::new(0, vec![1]));
    let _ = store.get(&SegmentKey::new(99, vec![99])); // miss

    store.clear();
    assert_eq!(store.len(), 0);
    assert!(store.is_empty());
    assert_eq!(store.total_bytes(), 0);
    let stats = store.stats();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
    assert_eq!(stats.evictions, 0);
}

// -- as_cache_stats compatibility --

#[test]
fn test_as_cache_stats_compatibility() {
    let mut store = SharedSegmentStore::new(1024 * 1024);
    store.insert(SegmentKey::new(0, vec![1]), segment_with_bytes(42));
    let _ = store.get(&SegmentKey::new(0, vec![1])); // hit
    let _ = store.get(&SegmentKey::new(99, vec![1])); // miss

    let stats = store.as_cache_stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.evictions, 0);
    assert_eq!(stats.total_bytes, 42);
}

// -- SegmentKey tests --

#[test]
fn test_segment_key_equality() {
    let a = SegmentKey::new(4, vec![1, 48000]);
    let b = SegmentKey::new(4, vec![1, 48000]);
    let c = SegmentKey::new(4, vec![1, 96000]);
    let d = SegmentKey::new(5, vec![1, 48000]);

    assert_eq!(a, b, "same kind and shape should be equal");
    assert_ne!(a, c, "different shape should not be equal");
    assert_ne!(a, d, "different kind should not be equal");
}

#[test]
fn test_segment_key_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(SegmentKey::new(0, vec![1, 128]));
    set.insert(SegmentKey::new(0, vec![1, 128])); // duplicate
    set.insert(SegmentKey::new(0, vec![1, 256]));
    assert_eq!(set.len(), 2, "HashSet should deduplicate equal keys");
}

// -- Debug format --

#[test]
fn test_debug_format() {
    let store = SharedSegmentStore::new(1024);
    let debug = format!("{store:?}");
    assert!(debug.contains("SharedSegmentStore"));
    assert!(debug.contains("entries"));
    assert!(debug.contains("byte_budget"));
}

// -- SharedCompiledSegment tests --

#[test]
fn test_shared_compiled_segment_from_empty() {
    let seg = empty_segment();
    assert_eq!(seg.tracked_bytes(), 0, "empty model has 0 buffer_plan bytes");
    assert!(Arc::strong_count(seg.def()) >= 1);
}

// -- Arc sharing verification --

#[test]
fn test_insert_returns_shared_arc() {
    let mut store = SharedSegmentStore::new(1024 * 1024);
    let key = SegmentKey::new(4, vec![1, 48000]);
    let segment = empty_segment();
    let original_def = Arc::clone(segment.def());

    let returned_def = store.insert(key, segment);
    assert!(
        Arc::ptr_eq(&returned_def, &original_def),
        "insert should return Arc pointing to the same allocation"
    );
}

// -- Multiple keys same segment kind with different shapes --

#[test]
fn test_multiple_shapes_per_segment_kind() {
    let mut store = SharedSegmentStore::new(1024 * 1024);

    // Generator segment compiled for 3 different sample lengths
    let shapes = vec![vec![1, 48000], vec![1, 96000], vec![1, 24000]];
    for shape in &shapes {
        let key = SegmentKey::new(4, shape.clone());
        store.insert(key, empty_segment());
    }

    assert_eq!(store.len(), 3);
    for shape in &shapes {
        let key = SegmentKey::new(4, shape.clone());
        assert!(
            store.get(&key).is_some(),
            "should find segment for shape {shape:?}"
        );
    }
}
