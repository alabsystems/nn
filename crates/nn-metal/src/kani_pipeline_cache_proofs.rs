// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for [`PipelineCache`](crate::PipelineCache) LRU properties.
//!
//! `PipelineCache` is the L3 tier in the 3-level GPU dispatch cache hierarchy.
//! It caches compiled Metal compute pipelines with LRU eviction based on a
//! monotonically increasing generation counter. Each access stamps the entry
//! with the current generation; eviction scans for the minimum generation.
//!
//! These harnesses model the generation-counter LRU algorithm abstractly
//! (without real Metal pipelines) and prove:
//!
//! 1. **LRU eviction order** — entry with minimum generation is evicted first
//! 2. **Hit updates recency** — accessing an entry assigns a new (higher) generation
//! 3. **Capacity bound** — cache size never exceeds `max_entries`
//! 4. **Insert-then-lookup** — inserted entry is immediately retrievable
//! 5. **Eviction frees exactly one** — when full, insert evicts exactly one entry
//! 6. **Counter matches cache size** — access_gen and pipelines maps stay in sync
//! 7. **Empty cache** — fresh cache has 0 entries

use std::collections::HashMap;

/// Model of the PipelineCache LRU state machine.
///
/// Mirrors the real `PipelineCache` fields: `pipelines` (HashMap of entries),
/// `access_gen` (HashMap of key -> generation), `gen_counter` (monotonic),
/// and `max_entries` (capacity bound). Uses `u64` keys and `u32` values
/// as lightweight stand-ins for `KernelSource` / `ComputePipeline`.
struct CacheModel {
    entries: HashMap<u64, u32>,
    generations: HashMap<u64, u64>,
    gen_counter: u64,
    max_entries: usize,
}

impl CacheModel {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            generations: HashMap::new(),
            gen_counter: 0,
            max_entries,
        }
    }

    /// Model of `PipelineCache::stamp` — increment gen_counter, assign to key.
    fn stamp(&mut self, key: u64) {
        self.gen_counter += 1;
        self.generations.insert(key, self.gen_counter);
    }

    /// Model of `PipelineCache::evict_lru` — remove entry with minimum generation.
    fn evict_lru(&mut self) {
        if let Some((&oldest_key, _)) = self.generations.iter().min_by_key(|(_, &g)| g) {
            self.generations.remove(&oldest_key);
            self.entries.remove(&oldest_key);
        }
    }

    /// Model of `PipelineCache::insert_l1` — evict if full, then insert + stamp.
    fn insert(&mut self, key: u64, value: u32) {
        if self.entries.len() >= self.max_entries {
            self.evict_lru();
        }
        self.entries.insert(key, value);
        self.stamp(key);
    }

    /// Model of cache lookup — returns value if present, stamps on hit.
    fn get(&mut self, key: u64) -> Option<u32> {
        if let Some(&val) = self.entries.get(&key) {
            self.stamp(key);
            Some(val)
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

// ============================================================================
// 1. LRU eviction order: entry with minimum generation is evicted first
// ============================================================================

/// Prove: when the cache is full and a new entry is inserted, the evicted
/// entry is the one with the smallest generation value (least recently used).
///
/// Models a cache of capacity 3 filled with entries A, B, C inserted in
/// order (A first = oldest). On next insert, A (lowest generation) is evicted.
#[kani::unwind(5)]
#[kani::proof]
fn proof_lru_eviction_order() {
    let max_entries: usize = 3;
    let mut cache = CacheModel::new(max_entries);

    // Insert A, B, C in order. A has the lowest generation.
    let key_a: u64 = 1;
    let key_b: u64 = 2;
    let key_c: u64 = 3;
    cache.insert(key_a, 10);
    cache.insert(key_b, 20);
    cache.insert(key_c, 30);

    let gen_a = *cache.generations.get(&key_a).unwrap();
    let gen_b = *cache.generations.get(&key_b).unwrap();
    let gen_c = *cache.generations.get(&key_c).unwrap();

    // A was inserted first, so it has the lowest generation.
    assert!(gen_a < gen_b, "A must have lower generation than B");
    assert!(gen_b < gen_c, "B must have lower generation than C");

    assert_eq!(cache.len(), 3, "cache must be full");

    // Insert D — triggers eviction of A (lowest generation).
    let key_d: u64 = 4;
    cache.insert(key_d, 40);

    // A must be evicted.
    assert!(
        cache.entries.get(&key_a).is_none(),
        "entry A (lowest generation) must be evicted"
    );
    // B, C, D remain.
    assert!(cache.entries.get(&key_b).is_some(), "B must remain");
    assert!(cache.entries.get(&key_c).is_some(), "C must remain");
    assert!(cache.entries.get(&key_d).is_some(), "D must remain");
    assert_eq!(cache.len(), max_entries, "cache size must equal max_entries");
}

// ============================================================================
// 2. Hit updates recency: accessing an entry assigns a higher generation
// ============================================================================

/// Prove: when `get` hits a cached entry, that entry's generation is updated
/// to the current (highest) generation counter value, making it the most
/// recently used.
///
/// Models: insert A, B, C (A is LRU). Access A — its generation becomes
/// the highest. Now B is LRU instead of A.
#[kani::unwind(5)]
#[kani::proof]
fn proof_hit_updates_recency() {
    let max_entries: usize = 3;
    let mut cache = CacheModel::new(max_entries);

    let key_a: u64 = 1;
    let key_b: u64 = 2;
    let key_c: u64 = 3;
    cache.insert(key_a, 10);
    cache.insert(key_b, 20);
    cache.insert(key_c, 30);

    // Before access: A has lowest gen, C has highest.
    let gen_a_before = *cache.generations.get(&key_a).unwrap();
    let gen_c_before = *cache.generations.get(&key_c).unwrap();
    assert!(
        gen_a_before < gen_c_before,
        "A must be LRU before access"
    );

    // Access A — promotes it to MRU.
    let result = cache.get(key_a);
    assert_eq!(result, Some(10), "get must return A's value");

    let gen_a_after = *cache.generations.get(&key_a).unwrap();
    let gen_b = *cache.generations.get(&key_b).unwrap();
    let gen_c = *cache.generations.get(&key_c).unwrap();

    // A's generation must now exceed B and C.
    assert!(
        gen_a_after > gen_b,
        "A's generation must exceed B after access"
    );
    assert!(
        gen_a_after > gen_c,
        "A's generation must exceed C after access"
    );

    // Now B is LRU (smallest generation). Inserting D should evict B, not A.
    let key_d: u64 = 4;
    cache.insert(key_d, 40);
    assert!(
        cache.entries.get(&key_b).is_none(),
        "B must be evicted (it is now LRU)"
    );
    assert!(
        cache.entries.get(&key_a).is_some(),
        "A must remain (it was promoted to MRU)"
    );
}

// ============================================================================
// 3. Capacity bound: cache size never exceeds max_entries
// ============================================================================

/// Prove: after any sequence of inserts, `cache.len() <= max_entries`.
///
/// Uses symbolic choices for keys and values over a bounded operation
/// sequence, asserting the capacity invariant after each insert.
#[kani::unwind(8)]
#[kani::proof]
fn proof_capacity_bound() {
    let max_entries: usize = kani::any();
    kani::assume(max_entries >= 1 && max_entries <= 3);

    let mut cache = CacheModel::new(max_entries);

    // Perform up to 5 inserts with symbolic keys.
    let num_ops: usize = kani::any();
    kani::assume(num_ops <= 5);

    let mut i: usize = 0;
    while i < num_ops {
        let key: u64 = kani::any();
        kani::assume(key <= 10); // bound key space for tractability
        let val: u32 = kani::any();
        cache.insert(key, val);

        // Invariant: size never exceeds capacity.
        assert!(
            cache.len() <= max_entries,
            "cache size must never exceed max_entries"
        );
        i += 1;
    }
}

// ============================================================================
// 4. Insert-then-lookup: inserted entry is immediately retrievable
// ============================================================================

/// Prove: after `insert(key, value)`, a subsequent `get(key)` returns
/// `Some(value)`.
///
/// Uses symbolic key and value. The cache starts empty with capacity >= 1,
/// so no eviction can remove the just-inserted entry before the get.
#[kani::unwind(3)]
#[kani::proof]
fn proof_insert_then_lookup() {
    let max_entries: usize = kani::any();
    kani::assume(max_entries >= 1 && max_entries <= 4);

    let mut cache = CacheModel::new(max_entries);

    let key: u64 = kani::any();
    let value: u32 = kani::any();

    cache.insert(key, value);

    // Immediately retrievable.
    let result = cache.get(key);
    assert_eq!(
        result,
        Some(value),
        "inserted entry must be immediately retrievable"
    );

    // Still present after get.
    assert!(
        cache.entries.contains_key(&key),
        "entry must remain in cache after get"
    );
}

// ============================================================================
// 5. Eviction frees exactly one: when full, insert evicts exactly one entry
// ============================================================================

/// Prove: when the cache is at capacity and a new (distinct) key is inserted,
/// exactly one existing entry is removed and replaced by the new entry.
/// The final size remains equal to `max_entries`.
#[kani::unwind(6)]
#[kani::proof]
fn proof_eviction_frees_exactly_one() {
    let max_entries: usize = kani::any();
    kani::assume(max_entries >= 2 && max_entries <= 4);

    let mut cache = CacheModel::new(max_entries);

    // Fill the cache with distinct keys 0..max_entries.
    let mut i: usize = 0;
    while i < max_entries {
        cache.insert(i as u64, i as u32);
        i += 1;
    }
    let size_before = cache.len();
    assert_eq!(size_before, max_entries, "cache must be full");

    // Insert a new key that does not collide with existing keys.
    let new_key: u64 = max_entries as u64; // guaranteed distinct from 0..max_entries
    let new_val: u32 = 99;

    cache.insert(new_key, new_val);

    // Size remains at max_entries (one evicted, one inserted).
    assert_eq!(
        cache.len(),
        max_entries,
        "cache size must remain max_entries after eviction + insert"
    );

    // The new entry is present.
    assert!(
        cache.entries.contains_key(&new_key),
        "new entry must be present"
    );

    // Exactly one old entry is missing.
    let mut missing_count: usize = 0;
    let mut j: usize = 0;
    while j < max_entries {
        if !cache.entries.contains_key(&(j as u64)) {
            missing_count += 1;
        }
        j += 1;
    }
    assert_eq!(
        missing_count, 1,
        "exactly one old entry must have been evicted"
    );

    // The evicted entry must be key 0 (first inserted = lowest generation).
    assert!(
        !cache.entries.contains_key(&0),
        "key 0 (oldest) must be the evicted entry"
    );
}

// ============================================================================
// 6. Generation map matches entry map: maps stay in sync
// ============================================================================

/// Prove: after any sequence of inserts and gets, every key in `entries`
/// has a corresponding entry in `generations`, and vice versa.
///
/// This models the invariant that `PipelineCache::pipelines` and
/// `PipelineCache::access_gen` are always synchronized — `insert_l1` and
/// `evict_lru` modify both maps atomically.
#[kani::unwind(7)]
#[kani::proof]
fn proof_generation_map_matches_entries() {
    let max_entries: usize = kani::any();
    kani::assume(max_entries >= 1 && max_entries <= 3);

    let mut cache = CacheModel::new(max_entries);

    // Perform a mixed sequence of inserts and gets.
    let num_ops: usize = kani::any();
    kani::assume(num_ops <= 4);

    let mut i: usize = 0;
    while i < num_ops {
        let is_insert: bool = kani::any();
        let key: u64 = kani::any();
        kani::assume(key <= 6); // bounded key space

        if is_insert {
            let val: u32 = kani::any();
            cache.insert(key, val);
        } else {
            let _ = cache.get(key);
        }

        // Invariant: entries and generations have the same key set.
        assert_eq!(
            cache.entries.len(),
            cache.generations.len(),
            "entries and generations maps must have equal size"
        );

        i += 1;
    }

    // Final check: every key in entries exists in generations.
    for &key in cache.entries.keys() {
        assert!(
            cache.generations.contains_key(&key),
            "every entry must have a generation"
        );
    }

    // Every key in generations exists in entries.
    for &key in cache.generations.keys() {
        assert!(
            cache.entries.contains_key(&key),
            "every generation must correspond to an entry"
        );
    }
}

// ============================================================================
// 7. Empty cache: fresh cache has 0 entries
// ============================================================================

/// Prove: a newly constructed `CacheModel` (modeling `PipelineCache::new`)
/// has 0 entries, 0 generations, and generation counter at 0.
///
/// Also proves that `get` on an empty cache returns `None` for any key,
/// and that the capacity is set correctly.
#[kani::unwind(1)]
#[kani::proof]
fn proof_empty_cache() {
    let max_entries: usize = kani::any();
    kani::assume(max_entries >= 1 && max_entries <= 256);

    let mut cache = CacheModel::new(max_entries);

    // Fresh cache has 0 entries.
    assert_eq!(cache.len(), 0, "fresh cache must have 0 entries");
    assert!(cache.entries.is_empty(), "entries map must be empty");
    assert!(cache.generations.is_empty(), "generations map must be empty");
    assert_eq!(cache.gen_counter, 0, "generation counter must start at 0");
    assert_eq!(cache.max_entries, max_entries, "max_entries must be set");

    // get on empty cache returns None for any key.
    let any_key: u64 = kani::any();
    let result = cache.get(any_key);
    assert!(
        result.is_none(),
        "get on empty cache must return None"
    );

    // Cache remains empty after a miss (get does not insert).
    assert_eq!(cache.len(), 0, "cache must remain empty after a miss");
    assert_eq!(
        cache.gen_counter, 0,
        "gen counter must remain 0 after miss on empty cache"
    );
}
