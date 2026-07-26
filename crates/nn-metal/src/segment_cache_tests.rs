// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for [`ShapeKeyedCache`] and [`SegmentCacheConfig`].
//!
//! These tests exercise pure cache logic (no GPU, no weights required).

use super::*;

// ========================================================================
// SegmentCacheConfig defaults and construction
// ========================================================================

#[test]
fn test_config_default_max_segments() {
    let config = SegmentCacheConfig::default();
    assert_eq!(config.max_segments_per_step, 4);
}

#[test]
fn test_config_default_eviction_policy() {
    let config = SegmentCacheConfig::default();
    assert_eq!(config.eviction, EvictionPolicy::Lru);
}

#[test]
fn test_config_default_byte_budget_is_none() {
    let config = SegmentCacheConfig::default();
    assert_eq!(
        config.byte_budget, None,
        "default byte_budget should be None (defer to compiled cache default)"
    );
}

#[test]
fn test_config_custom_max_segments() {
    let config = SegmentCacheConfig {
        max_segments_per_step: 32,
        ..SegmentCacheConfig::default()
    };
    assert_eq!(config.max_segments_per_step, 32);
    // Other fields remain at defaults.
    assert_eq!(config.eviction, EvictionPolicy::Lru);
    assert_eq!(config.byte_budget, None);
}

#[test]
fn test_config_custom_byte_budget() {
    let config = SegmentCacheConfig {
        byte_budget: Some(256 * 1024 * 1024),
        ..SegmentCacheConfig::default()
    };
    assert_eq!(config.byte_budget, Some(256 * 1024 * 1024));
    assert_eq!(config.max_segments_per_step, 4);
}

#[test]
fn test_config_single_entry_capacity() {
    // max=1: one compiled segment per step, thrashes on shape change.
    // Pre-#2626 behavior.
    let config = SegmentCacheConfig {
        max_segments_per_step: 1,
        ..SegmentCacheConfig::default()
    };
    assert_eq!(config.max_segments_per_step, 1);
}

#[test]
fn test_config_clone() {
    let config = SegmentCacheConfig {
        max_segments_per_step: 7,
        eviction: EvictionPolicy::Lru,
        byte_budget: Some(1024),
        shared_store: None,
    };
    let cloned = config;
    assert_eq!(cloned.max_segments_per_step, 7);
    assert_eq!(cloned.eviction, EvictionPolicy::Lru);
    assert_eq!(cloned.byte_budget, Some(1024));
}

#[test]
fn test_config_debug_format() {
    let config = SegmentCacheConfig::default();
    let debug = format!("{config:?}");
    assert!(debug.contains("SegmentCacheConfig"));
    assert!(debug.contains("max_segments_per_step"));
}

// ========================================================================
// EvictionPolicy properties
// ========================================================================

#[test]
fn test_eviction_policy_eq() {
    assert_eq!(EvictionPolicy::Lru, EvictionPolicy::Lru);
}

#[test]
fn test_eviction_policy_copy() {
    let a = EvictionPolicy::Lru;
    let b = a; // Copy
    assert_eq!(a, b);
}

#[test]
fn test_eviction_policy_debug() {
    let debug = format!("{:?}", EvictionPolicy::Lru);
    assert_eq!(debug, "Lru");
}

// ========================================================================
// ShapeKeyedCache: construction and empty state
// ========================================================================

#[test]
fn test_new_cache_is_empty() {
    let cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(4);
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.capacity(), 4);
}

#[test]
fn test_new_cache_various_capacities() {
    for cap in [1, 2, 5, 10, 100, 1000] {
        let cache: ShapeKeyedCache<()> = ShapeKeyedCache::new(cap);
        assert_eq!(cache.capacity(), cap);
        assert!(cache.is_empty());
    }
}

#[test]
#[should_panic(expected = "capacity must be > 0")]
fn test_zero_capacity_panics() {
    let _cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(0);
}

#[test]
fn test_get_on_empty_cache_returns_none() {
    let mut cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(4);
    assert_eq!(cache.get(&[1, 128]), None);
    assert_eq!(cache.get(&[]), None);
    assert_eq!(cache.get(&[0]), None);
}

// ========================================================================
// ShapeKeyedCache: basic insertion and retrieval
// ========================================================================

#[test]
fn test_insert_and_get_single() {
    let mut cache: ShapeKeyedCache<String> = ShapeKeyedCache::new(4);
    cache.insert(vec![1, 128], "model_128".to_string());
    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());
    assert_eq!(cache.get(&[1, 128]), Some(&"model_128".to_string()));
}

#[test]
fn test_get_miss_returns_none() {
    let mut cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(4);
    cache.insert(vec![1, 128], 42);
    assert_eq!(cache.get(&[1, 256]), None);
    assert_eq!(cache.get(&[2, 128]), None);
    assert_eq!(cache.get(&[1]), None);
    assert_eq!(cache.get(&[1, 128, 0]), None);
}

#[test]
fn test_multiple_shapes_coexist() {
    let mut cache: ShapeKeyedCache<&str> = ShapeKeyedCache::new(8);
    cache.insert(vec![1, 64], "a");
    cache.insert(vec![1, 128], "b");
    cache.insert(vec![1, 256], "c");
    cache.insert(vec![2, 64], "d");
    assert_eq!(cache.len(), 4);
    assert_eq!(cache.get(&[1, 64]), Some(&"a"));
    assert_eq!(cache.get(&[1, 128]), Some(&"b"));
    assert_eq!(cache.get(&[1, 256]), Some(&"c"));
    assert_eq!(cache.get(&[2, 64]), Some(&"d"));
}

#[test]
fn test_insert_same_shape_replaces_value() {
    let mut cache: ShapeKeyedCache<&str> = ShapeKeyedCache::new(4);
    cache.insert(vec![1, 128], "v1");
    cache.insert(vec![1, 128], "v2");
    assert_eq!(cache.len(), 1, "duplicate shape should not increase count");
    assert_eq!(cache.get(&[1, 128]), Some(&"v2"), "value should be updated");
}

#[test]
fn test_insert_same_shape_three_times() {
    let mut cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(4);
    cache.insert(vec![3, 5], 1);
    cache.insert(vec![3, 5], 2);
    cache.insert(vec![3, 5], 3);
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.get(&[3, 5]), Some(&3));
}

// ========================================================================
// ShapeKeyedCache: LRU eviction
// ========================================================================

#[test]
fn test_lru_eviction_at_capacity() {
    let mut cache: ShapeKeyedCache<&str> = ShapeKeyedCache::new(2);
    cache.insert(vec![1, 10], "first");
    cache.insert(vec![1, 20], "second");
    assert_eq!(cache.len(), 2);

    // Insert third -- should evict "first" (LRU = back).
    cache.insert(vec![1, 30], "third");
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.get(&[1, 10]), None, "first should be evicted");
    assert_eq!(cache.get(&[1, 20]), Some(&"second"));
    assert_eq!(cache.get(&[1, 30]), Some(&"third"));
}

#[test]
fn test_lru_eviction_chain() {
    // Capacity 2: insert A, B, C, D. Only C and D should survive.
    let mut cache: ShapeKeyedCache<char> = ShapeKeyedCache::new(2);
    cache.insert(vec![1], 'A');
    cache.insert(vec![2], 'B');
    cache.insert(vec![3], 'C'); // evicts A
    cache.insert(vec![4], 'D'); // evicts B
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.get(&[1]), None);
    assert_eq!(cache.get(&[2]), None);
    assert_eq!(cache.get(&[3]), Some(&'C'));
    assert_eq!(cache.get(&[4]), Some(&'D'));
}

#[test]
fn test_lru_promotion_on_get() {
    let mut cache: ShapeKeyedCache<&str> = ShapeKeyedCache::new(2);
    cache.insert(vec![1, 10], "first");
    cache.insert(vec![1, 20], "second");

    // Access "first" to promote it to MRU.
    cache.get(&[1, 10]);

    // Now "second" is LRU. Inserting a third should evict "second".
    cache.insert(vec![1, 30], "third");
    assert_eq!(
        cache.get(&[1, 10]),
        Some(&"first"),
        "first was promoted, should survive"
    );
    assert_eq!(
        cache.get(&[1, 20]),
        None,
        "second was LRU, should be evicted"
    );
    assert_eq!(cache.get(&[1, 30]), Some(&"third"));
}

#[test]
fn test_lru_promotion_complex_sequence() {
    // Capacity 3: insert A, B, C. Access A. Insert D. B should be evicted.
    let mut cache: ShapeKeyedCache<&str> = ShapeKeyedCache::new(3);
    cache.insert(vec![1], "A"); // Order: [A]
    cache.insert(vec![2], "B"); // Order: [B, A]
    cache.insert(vec![3], "C"); // Order: [C, B, A]

    // Access A -- promotes to MRU.
    cache.get(&[1]); // Order: [A, C, B]

    // Insert D -- evicts LRU (B).
    cache.insert(vec![4], "D"); // Order: [D, A, C]
    assert_eq!(cache.get(&[1]), Some(&"A"), "A was promoted");
    assert_eq!(cache.get(&[2]), None, "B was LRU and evicted");
    assert_eq!(cache.get(&[3]), Some(&"C"));
    assert_eq!(cache.get(&[4]), Some(&"D"));
}

#[test]
fn test_insert_same_shape_promotes_to_mru() {
    let mut cache: ShapeKeyedCache<&str> = ShapeKeyedCache::new(2);
    cache.insert(vec![1, 10], "a");
    cache.insert(vec![1, 20], "b");

    // Re-insert shape [1, 10] with new value -- promotes to MRU.
    cache.insert(vec![1, 10], "a_updated");

    // Now [1, 20] is LRU. Inserting new shape should evict it.
    cache.insert(vec![1, 30], "c");
    assert_eq!(
        cache.get(&[1, 10]),
        Some(&"a_updated"),
        "re-inserted shape should survive"
    );
    assert_eq!(cache.get(&[1, 20]), None, "LRU entry should be evicted");
}

#[test]
fn test_capacity_one_always_has_latest() {
    let mut cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(1);
    cache.insert(vec![1, 10], 1);
    assert_eq!(cache.len(), 1);
    cache.insert(vec![1, 20], 2);
    assert_eq!(cache.len(), 1);
    cache.insert(vec![1, 30], 3);
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.get(&[1, 30]), Some(&3));
    assert_eq!(cache.get(&[1, 10]), None);
    assert_eq!(cache.get(&[1, 20]), None);
}

#[test]
fn test_capacity_one_get_then_insert_new() {
    let mut cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(1);
    cache.insert(vec![10], 100);
    // get promotes the only entry (no-op since it is already MRU).
    assert_eq!(cache.get(&[10]), Some(&100));
    // Insert a new shape -- must evict the only entry.
    cache.insert(vec![20], 200);
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.get(&[10]), None);
    assert_eq!(cache.get(&[20]), Some(&200));
}

#[test]
fn test_fill_to_exact_capacity() {
    let mut cache: ShapeKeyedCache<usize> = ShapeKeyedCache::new(5);
    for i in 0..5 {
        cache.insert(vec![i], i);
    }
    assert_eq!(cache.len(), 5);
    // All entries still present.
    for i in 0..5 {
        assert_eq!(cache.get(&[i]), Some(&i));
    }
}

#[test]
fn test_eviction_at_exact_capacity_plus_one() {
    let mut cache: ShapeKeyedCache<usize> = ShapeKeyedCache::new(5);
    for i in 0..5 {
        cache.insert(vec![i], i);
    }
    // Insert 6th -- evicts the LRU (key 0, inserted first, never accessed).
    // But note: we accessed all of them via get in previous test, here we do not.
    // After inserts 0..5, the LRU is 0 (inserted first, never promoted since).
    cache.insert(vec![99], 99);
    assert_eq!(cache.len(), 5);
    assert_eq!(cache.get(&[0]), None, "key 0 was LRU and should be evicted");
    assert_eq!(cache.get(&[99]), Some(&99));
    // Keys 1..5 still present.
    for i in 1..5 {
        assert_eq!(cache.get(&[i]), Some(&i));
    }
}

// ========================================================================
// ShapeKeyedCache: clear and reset
// ========================================================================

#[test]
fn test_clear_populated_cache() {
    let mut cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(4);
    cache.insert(vec![1], 10);
    cache.insert(vec![2], 20);
    assert_eq!(cache.len(), 2);
    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    // Capacity is preserved.
    assert_eq!(cache.capacity(), 4);
}

#[test]
fn test_clear_empty_cache() {
    let mut cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(4);
    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}

#[test]
fn test_insert_after_clear() {
    let mut cache: ShapeKeyedCache<&str> = ShapeKeyedCache::new(2);
    cache.insert(vec![1], "old_a");
    cache.insert(vec![2], "old_b");
    cache.clear();

    // Cache should behave as fresh after clear.
    cache.insert(vec![3], "new_a");
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.get(&[3]), Some(&"new_a"));
    assert_eq!(cache.get(&[1]), None, "cleared entries should not reappear");
}

#[test]
fn test_clear_then_fill_to_capacity() {
    let mut cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(3);
    cache.insert(vec![1], 1);
    cache.insert(vec![2], 2);
    cache.insert(vec![3], 3);
    cache.clear();

    // Refill.
    cache.insert(vec![10], 10);
    cache.insert(vec![20], 20);
    cache.insert(vec![30], 30);
    assert_eq!(cache.len(), 3);
    // Insert one more -- should evict LRU (10).
    cache.insert(vec![40], 40);
    assert_eq!(cache.len(), 3);
    assert_eq!(cache.get(&[10]), None);
    assert_eq!(cache.get(&[40]), Some(&40));
}

// ========================================================================
// ShapeKeyedCache: edge cases with shape keys
// ========================================================================

#[test]
fn test_empty_shape_key() {
    let mut cache: ShapeKeyedCache<&str> = ShapeKeyedCache::new(4);
    cache.insert(vec![], "scalar");
    assert_eq!(cache.get(&[]), Some(&"scalar"));
    assert_eq!(cache.len(), 1);
}

#[test]
fn test_high_rank_shape_key() {
    let mut cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(4);
    let shape = vec![2, 3, 4, 5, 6, 7, 8];
    cache.insert(shape.clone(), 42);
    assert_eq!(cache.get(&shape), Some(&42));
}

#[test]
fn test_shape_keys_distinguish_by_length() {
    // [1, 128] and [1, 128, 1] are different shapes.
    let mut cache: ShapeKeyedCache<&str> = ShapeKeyedCache::new(4);
    cache.insert(vec![1, 128], "2d");
    cache.insert(vec![1, 128, 1], "3d");
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.get(&[1, 128]), Some(&"2d"));
    assert_eq!(cache.get(&[1, 128, 1]), Some(&"3d"));
}

#[test]
fn test_shape_keys_distinguish_by_values() {
    // [1, 128] and [128, 1] are different shapes.
    let mut cache: ShapeKeyedCache<&str> = ShapeKeyedCache::new(4);
    cache.insert(vec![1, 128], "narrow");
    cache.insert(vec![128, 1], "wide");
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.get(&[1, 128]), Some(&"narrow"));
    assert_eq!(cache.get(&[128, 1]), Some(&"wide"));
}

#[test]
fn test_large_dimension_values() {
    let mut cache: ShapeKeyedCache<&str> = ShapeKeyedCache::new(2);
    cache.insert(vec![usize::MAX, 0], "max_first");
    cache.insert(vec![0, usize::MAX], "max_second");
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.get(&[usize::MAX, 0]), Some(&"max_first"));
    assert_eq!(cache.get(&[0, usize::MAX]), Some(&"max_second"));
}

// ========================================================================
// ShapeKeyedCache: get does not insert
// ========================================================================

#[test]
fn test_get_miss_does_not_modify_cache() {
    let mut cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(4);
    cache.insert(vec![1], 10);
    assert_eq!(cache.get(&[99]), None);
    assert_eq!(cache.len(), 1, "miss should not alter length");
}

#[test]
fn test_repeated_misses_do_not_grow_cache() {
    let mut cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(2);
    for i in 0..100 {
        assert_eq!(cache.get(&[i]), None);
    }
    assert_eq!(cache.len(), 0);
}

// ========================================================================
// ShapeKeyedCache: promotion idempotence
// ========================================================================

#[test]
fn test_get_already_mru_is_idempotent() {
    // Accessing the MRU entry should not change ordering.
    let mut cache: ShapeKeyedCache<&str> = ShapeKeyedCache::new(3);
    cache.insert(vec![1], "A");
    cache.insert(vec![2], "B");
    cache.insert(vec![3], "C"); // MRU = C

    // Access C (already MRU). Order should remain [C, B, A].
    assert_eq!(cache.get(&[3]), Some(&"C"));

    // Insert D. Should evict A (still LRU).
    cache.insert(vec![4], "D");
    assert_eq!(cache.get(&[1]), None, "A should be evicted (was LRU)");
    assert_eq!(cache.get(&[2]), Some(&"B"));
    assert_eq!(cache.get(&[3]), Some(&"C"));
    assert_eq!(cache.get(&[4]), Some(&"D"));
}

#[test]
fn test_double_promotion_preserves_order() {
    let mut cache: ShapeKeyedCache<&str> = ShapeKeyedCache::new(3);
    cache.insert(vec![1], "A");
    cache.insert(vec![2], "B");
    cache.insert(vec![3], "C");

    // Promote A twice. Should have no additional effect.
    cache.get(&[1]);
    cache.get(&[1]);

    // Insert D. B should be evicted (LRU after A was promoted).
    cache.insert(vec![4], "D");
    assert_eq!(cache.get(&[1]), Some(&"A"));
    assert_eq!(cache.get(&[2]), None, "B should be evicted");
    assert_eq!(cache.get(&[3]), Some(&"C"));
}

// ========================================================================
// ShapeKeyedCache: stress / many entries
// ========================================================================

#[test]
fn test_many_inserts_with_small_capacity() {
    let mut cache: ShapeKeyedCache<usize> = ShapeKeyedCache::new(3);
    for i in 0..100 {
        cache.insert(vec![i], i);
    }
    assert_eq!(cache.len(), 3);
    // Only the last 3 entries should survive.
    assert_eq!(cache.get(&[97]), Some(&97));
    assert_eq!(cache.get(&[98]), Some(&98));
    assert_eq!(cache.get(&[99]), Some(&99));
    // Older entries are gone.
    assert_eq!(cache.get(&[96]), None);
}

#[test]
fn test_interleaved_get_and_insert() {
    let mut cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(2);
    cache.insert(vec![1], 1);
    cache.insert(vec![2], 2);
    // Access key 1 to promote.
    assert_eq!(cache.get(&[1]), Some(&1));
    // Insert key 3 -- evicts key 2.
    cache.insert(vec![3], 3);
    assert_eq!(cache.get(&[1]), Some(&1));
    assert_eq!(cache.get(&[2]), None);
    // Access key 3 to promote.
    assert_eq!(cache.get(&[3]), Some(&3));
    // Insert key 4 -- evicts key 1 (now LRU).
    cache.insert(vec![4], 4);
    assert_eq!(cache.get(&[1]), None);
    assert_eq!(cache.get(&[3]), Some(&3));
    assert_eq!(cache.get(&[4]), Some(&4));
}

// ========================================================================
// ShapeKeyedCache: capacity is immutable
// ========================================================================

#[test]
fn test_capacity_unchanged_after_operations() {
    let mut cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(5);
    assert_eq!(cache.capacity(), 5);
    cache.insert(vec![1], 1);
    assert_eq!(cache.capacity(), 5);
    cache.get(&[1]);
    assert_eq!(cache.capacity(), 5);
    cache.clear();
    assert_eq!(cache.capacity(), 5);
    for i in 0..20 {
        cache.insert(vec![i], i as i32);
    }
    assert_eq!(cache.capacity(), 5);
}

// ========================================================================
// ShapeKeyedCache: len/is_empty consistency
// ========================================================================

#[test]
fn test_len_tracks_inserts_and_evictions() {
    let mut cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(2);
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());

    cache.insert(vec![1], 1);
    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());

    cache.insert(vec![2], 2);
    assert_eq!(cache.len(), 2);

    // At capacity. Next insert evicts one.
    cache.insert(vec![3], 3);
    assert_eq!(cache.len(), 2, "eviction keeps len at capacity");

    // Replace existing (no net change).
    cache.insert(vec![3], 33);
    assert_eq!(cache.len(), 2);
}

// ========================================================================
// ShapeKeyedCache: generic value types
// ========================================================================

#[test]
fn test_cache_with_integer_values() {
    let mut cache: ShapeKeyedCache<u64> = ShapeKeyedCache::new(2);
    cache.insert(vec![1], u64::MAX);
    assert_eq!(cache.get(&[1]), Some(&u64::MAX));
}

#[test]
fn test_cache_with_vec_values() {
    let mut cache: ShapeKeyedCache<Vec<f32>> = ShapeKeyedCache::new(2);
    cache.insert(vec![1, 128], vec![1.0, 2.0, 3.0]);
    assert_eq!(cache.get(&[1, 128]), Some(&vec![1.0, 2.0, 3.0]));
}

#[test]
fn test_cache_with_unit_values() {
    // Useful for "set-like" caches where presence is all that matters.
    let mut cache: ShapeKeyedCache<()> = ShapeKeyedCache::new(4);
    cache.insert(vec![1, 64], ());
    cache.insert(vec![1, 128], ());
    assert_eq!(cache.get(&[1, 64]), Some(&()));
    assert_eq!(cache.get(&[1, 256]), None);
}

#[test]
fn test_cache_with_bool_values() {
    let mut cache: ShapeKeyedCache<bool> = ShapeKeyedCache::new(2);
    cache.insert(vec![1], true);
    cache.insert(vec![2], false);
    assert_eq!(cache.get(&[1]), Some(&true));
    assert_eq!(cache.get(&[2]), Some(&false));
}
