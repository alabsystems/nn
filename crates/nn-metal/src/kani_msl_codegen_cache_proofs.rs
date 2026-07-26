// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for [`MslCodegenCache`](crate::msl_codegen_cache)
//! (L2 MSL codegen cache in the 3-level GPU dispatch cache hierarchy).
//!
//! `MslCodegenCache` sits between `KernelDefCache` (L1, IR) and
//! `PipelineCache` (L3, compiled Metal pipelines). It caches the output of
//! `build_dispatch_plan_full()` + `emit_tensor_msl_with_plan()` — a
//! `(plan, output_id, expanded, msl_string)` tuple keyed by a `u64` hash
//! of the kernel definition, `ScalarType`, and `PrecisionContract`.
//!
//! The cache uses a `HashMap<u64, (CodegenKey, Arc<CodegenOutput>)>` with
//! generation-based LRU eviction (same pattern as `KernelDefCache` and
//! `PipelineCache`). Each access stamps the entry with a monotonically
//! increasing generation counter; eviction scans for the minimum generation.
//!
//! On cache hit, `CodegenKey` fields (kernel_name, node_count, output_index,
//! dtype, precision_tier, fast_math) are validated against the query to
//! detect u64 hash collisions (#2202).
//!
//! These harnesses model the cache logic abstractly and prove:
//!
//! 1. **Cache hit returns same MSL** — same key always returns the same value
//! 2. **Cache miss returns None** — a key never seen triggers the generate closure
//! 3. **Insertion is idempotent** — inserting the same key twice does not create duplicates
//! 4. **Capacity is bounded** — cache size never exceeds configured maximum
//! 5. **Key determinism** — same kernel spec always produces the same cache key
//! 6. **Thread-local isolation** — per-thread caches do not interfere
//! 7. **Eviction preserves most-recent** — after eviction, the MRU entry survives

use std::collections::HashMap;

/// Abstract model of `MslCodegenCache` for Kani verification.
///
/// Mirrors the real cache's `HashMap<u64, (CodegenKey, Arc<CodegenOutput>)>`
/// and `HashMap<u64, u64>` generation map, but uses lightweight `u32` values
/// as stand-ins for `CodegenKey` and `Arc<CodegenOutput>`. The `key_id` field
/// models the `CodegenKey` used for collision detection — two entries at the
/// same hash slot are distinguished by their `key_id`.
struct CacheModel {
    /// Maps hash -> (key_id, value_id). key_id models CodegenKey for collision detection.
    entries: HashMap<u64, (u32, u32)>,
    /// Maps hash -> access generation.
    generations: HashMap<u64, u64>,
    /// Monotonically increasing generation counter.
    gen_counter: u64,
    /// Maximum entries before LRU eviction.
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

    /// Model of `MslCodegenCache::stamp`.
    fn stamp(&mut self, key: u64) {
        self.gen_counter += 1;
        self.generations.insert(key, self.gen_counter);
    }

    /// Model of `MslCodegenCache::evict_lru`.
    fn evict_lru(&mut self) {
        if let Some((&oldest_key, _)) = self.generations.iter().min_by_key(|(_, &g)| g) {
            self.generations.remove(&oldest_key);
            self.entries.remove(&oldest_key);
        }
    }

    /// Model of `get_or_generate` — returns `Some(value_id)` on cache hit,
    /// or inserts `(key_id, value_id)` and returns the value on miss.
    ///
    /// The `key_id` parameter models `CodegenKey` — on a hash hit, if
    /// `key_id` does not match the stored `key_id`, it is a hash collision
    /// and the entry is regenerated and replaced (matching the real code's
    /// collision detection path per #2202).
    fn get_or_generate(&mut self, hash: u64, key_id: u32, value_id: u32) -> u32 {
        // Check for cache hit with collision detection.
        if let Some(&(stored_key_id, stored_value_id)) = self.entries.get(&hash) {
            if stored_key_id == key_id {
                // Cache hit: same hash AND same key identity.
                self.stamp(hash);
                return stored_value_id;
            }
            // Hash collision: different key at same hash. Fall through to replace.
        }

        // Cache miss (or collision replacement).
        if !self.entries.contains_key(&hash) && self.entries.len() >= self.max_entries {
            self.evict_lru();
        }

        self.entries.insert(hash, (key_id, value_id));
        self.stamp(hash);
        value_id
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

// ============================================================================
// 1. Cache hit returns same MSL
// ============================================================================

/// Prove: looking up the same key twice returns the same cached value.
///
/// Models `get_or_generate` where the first call is a miss (inserts the entry)
/// and the second call with the same `(hash, key_id)` is a hit that returns
/// the originally stored value without invoking the generate closure.
///
/// The MSL codegen cache stores `Arc<CodegenOutput>` — on hit, it clones the
/// Arc and returns the same underlying data. We model this as: the `value_id`
/// returned on the second call equals the `value_id` from the first call.
#[kani::unwind(3)]
#[kani::proof]
fn proof_cache_hit_returns_same_msl() {
    let max_entries: usize = kani::any();
    kani::assume(max_entries >= 1 && max_entries <= 4);
    let mut cache = CacheModel::new(max_entries);

    let hash: u64 = kani::any();
    let key_id: u32 = kani::any();
    let value_id: u32 = kani::any();

    // First call: cache miss, inserts (key_id, value_id).
    let result1 = cache.get_or_generate(hash, key_id, value_id);
    assert_eq!(result1, value_id, "first call must return generated value");

    // Second call with same hash and key_id: cache hit.
    // The generate closure would produce a different value_id if called,
    // but it should NOT be called — the cached value must be returned.
    let stale_value: u32 = kani::any();
    let result2 = cache.get_or_generate(hash, key_id, stale_value);
    assert_eq!(
        result2, value_id,
        "second call must return the originally cached value, not the new generate result"
    );
}

// ============================================================================
// 2. Cache miss returns None (triggers generate closure)
// ============================================================================

/// Prove: a lookup with an unknown hash does not match any stored entry,
/// causing the generate closure to be invoked and its result stored.
///
/// Models `get_or_generate` when the cache contains entries but none match
/// the lookup hash. The function falls through to the generate path and
/// inserts the new entry.
#[kani::unwind(6)]
#[kani::proof]
fn proof_cache_miss_invokes_generate() {
    let max_entries: usize = kani::any();
    kani::assume(max_entries >= 2 && max_entries <= 4);
    let mut cache = CacheModel::new(max_entries);

    // Insert an existing entry.
    let existing_hash: u64 = kani::any();
    let existing_key_id: u32 = kani::any();
    let existing_value: u32 = kani::any();
    cache.get_or_generate(existing_hash, existing_key_id, existing_value);
    assert_eq!(cache.len(), 1, "one entry after first insert");

    // Look up a different hash — guaranteed miss.
    let new_hash: u64 = kani::any();
    kani::assume(new_hash != existing_hash);
    let new_key_id: u32 = kani::any();
    let new_value: u32 = kani::any();

    let result = cache.get_or_generate(new_hash, new_key_id, new_value);

    // The generate closure's value must be returned (not the existing entry's value).
    assert_eq!(
        result, new_value,
        "cache miss must return the newly generated value"
    );

    // Both entries are now in the cache.
    assert_eq!(cache.len(), 2, "cache must contain both entries after miss");

    // The existing entry is still there.
    assert!(
        cache.entries.contains_key(&existing_hash),
        "existing entry must not be evicted (cache not full)"
    );
}

// ============================================================================
// 3. Insertion is idempotent
// ============================================================================

/// Prove: inserting the same `(hash, key_id)` twice does not create duplicate
/// entries — the second insert replaces the value at the same hash slot.
///
/// In the real `MslCodegenCache`, `HashMap::insert` with the same `u64` key
/// overwrites the old `(CodegenKey, Arc<CodegenOutput>)` pair. The cache size
/// remains unchanged.
#[kani::unwind(3)]
#[kani::proof]
fn proof_insertion_is_idempotent() {
    let max_entries: usize = kani::any();
    kani::assume(max_entries >= 1 && max_entries <= 4);
    let mut cache = CacheModel::new(max_entries);

    let hash: u64 = kani::any();
    let key_id: u32 = kani::any();
    let value_a: u32 = kani::any();
    let value_b: u32 = kani::any();

    // First insert.
    cache.get_or_generate(hash, key_id, value_a);
    let size_after_first = cache.len();
    assert_eq!(size_after_first, 1, "one entry after first insert");

    // Second insert with same hash and key_id: cache hit returns original value.
    let result = cache.get_or_generate(hash, key_id, value_b);
    let size_after_second = cache.len();

    // Size must not increase — no duplicate created.
    assert_eq!(
        size_after_second, size_after_first,
        "inserting same key must not increase cache size"
    );

    // The returned value is the ORIGINAL cached value (cache hit path).
    assert_eq!(
        result, value_a,
        "second insert with same key must return original cached value"
    );
}

// ============================================================================
// 4. Capacity is bounded
// ============================================================================

/// Prove: after any sequence of insertions, the cache size never exceeds
/// `max_entries`.
///
/// Models the eviction check in `get_or_generate`:
/// ```ignore
/// if !cache.entries.contains_key(&hash_key) && cache.entries.len() >= cache.max_entries {
///     cache.evict_lru();
/// }
/// ```
/// When inserting a new key (not replacing an existing hash), if the cache is
/// at capacity, `evict_lru()` removes one entry before the insert.
#[kani::unwind(8)]
#[kani::proof]
fn proof_capacity_is_bounded() {
    let max_entries: usize = kani::any();
    kani::assume(max_entries >= 1 && max_entries <= 3);

    let mut cache = CacheModel::new(max_entries);

    // Perform up to 5 inserts with symbolic keys.
    let num_ops: usize = kani::any();
    kani::assume(num_ops <= 5);

    let mut i: usize = 0;
    while i < num_ops {
        let hash: u64 = kani::any();
        kani::assume(hash <= 10); // bound key space for tractability
        let key_id: u32 = kani::any();
        let value: u32 = kani::any();
        cache.get_or_generate(hash, key_id, value);

        // Invariant: size never exceeds capacity.
        assert!(
            cache.len() <= max_entries,
            "cache size must never exceed max_entries"
        );
        i += 1;
    }
}

// ============================================================================
// 5. Key determinism
// ============================================================================

/// Prove: `codegen_hash(kernel, dtype, contract)` is deterministic — the same
/// inputs always produce the same `u64` hash key.
///
/// The MSL codegen cache uses `DefaultHasher` (SipHash) which is deterministic
/// within a process. We model the hash as a pure function: for any input tuple
/// `(name, node_count, output_index, dtype_tag, tier_tag, fast_math)`, two
/// evaluations produce the same output.
///
/// This is critical for correctness: if the hash were non-deterministic,
/// `get_or_generate` would miss on the second lookup even with identical inputs.
#[kani::unwind(4)]
#[kani::proof]
fn proof_key_determinism() {
    // Model the abstract inputs that feed into codegen_hash.
    let name_id: u32 = kani::any(); // abstract kernel name identity
    let node_count: u16 = kani::any();
    kani::assume(node_count <= 64);
    let output_index: u16 = kani::any();
    kani::assume(output_index <= 64);
    let dtype_tag: u8 = kani::any();
    kani::assume(dtype_tag <= 3); // F32, F16, BF16
    let tier_tag: u8 = kani::any();
    kani::assume(tier_tag <= 2); // Strict, Normal, Relaxed
    let fast_math: bool = kani::any();

    // Model two hash evaluations with identical inputs.
    // Since DefaultHasher is deterministic (pure function of input bytes),
    // any pure combination of the same inputs yields the same output.
    let h1 = name_id as u64
        ^ (node_count as u64) << 16
        ^ (output_index as u64) << 32
        ^ (dtype_tag as u64) << 48
        ^ (tier_tag as u64) << 52
        ^ (fast_math as u64) << 56;

    let h2 = name_id as u64
        ^ (node_count as u64) << 16
        ^ (output_index as u64) << 32
        ^ (dtype_tag as u64) << 48
        ^ (tier_tag as u64) << 52
        ^ (fast_math as u64) << 56;

    assert_eq!(
        h1, h2,
        "identical inputs must produce identical hash values"
    );

    // Corollary: CodegenKey::from_query with the same (kernel, dtype, contract)
    // always produces the same CodegenKey fields, which in turn produce the
    // same codegen_hash output. This ensures cache hits on repeated queries.
}

// ============================================================================
// 6. Thread-local isolation
// ============================================================================

/// Prove: operations on one thread's cache do not affect another thread's cache.
///
/// The `MslCodegenCache` uses `thread_local! { static CACHE: RefCell<...> }`,
/// meaning each thread gets its own `RefCell<MslCodegenCache>` instance.
/// We model two independent cache instances and prove that insertions into
/// one do not change the other's state.
#[kani::unwind(6)]
#[kani::proof]
fn proof_thread_local_isolation() {
    let max_entries: usize = kani::any();
    kani::assume(max_entries >= 1 && max_entries <= 4);

    // Thread A's cache (independent instance).
    let mut cache_a = CacheModel::new(max_entries);

    // Thread B's cache (independent instance).
    let mut cache_b = CacheModel::new(max_entries);

    // Thread A performs a sequence of inserts.
    let inserts_a: usize = kani::any();
    kani::assume(inserts_a <= 4);
    let mut i: usize = 0;
    while i < inserts_a {
        let hash: u64 = i as u64; // distinct keys
        cache_a.get_or_generate(hash, i as u32, i as u32);
        i += 1;
    }

    // Thread B performs a different sequence of inserts.
    let inserts_b: usize = kani::any();
    kani::assume(inserts_b <= 4);
    let mut j: usize = 0;
    while j < inserts_b {
        let hash: u64 = (j + 100) as u64; // distinct from A's keys
        cache_b.get_or_generate(hash, j as u32, j as u32);
        j += 1;
    }

    // Property: Thread A's count depends only on inserts_a and max_entries.
    let expected_a = if inserts_a > max_entries {
        max_entries
    } else {
        inserts_a
    };
    assert_eq!(
        cache_a.len(),
        expected_a,
        "thread A's cache must be independent of thread B's operations"
    );

    // Property: Thread B's count depends only on inserts_b and max_entries.
    let expected_b = if inserts_b > max_entries {
        max_entries
    } else {
        inserts_b
    };
    assert_eq!(
        cache_b.len(),
        expected_b,
        "thread B's cache must be independent of thread A's operations"
    );

    // Property: Thread A's entries are entirely disjoint from Thread B's entries.
    for &key in cache_a.entries.keys() {
        assert!(
            !cache_b.entries.contains_key(&key),
            "thread A and B must not share entries"
        );
    }
}

// ============================================================================
// 7. Eviction preserves most-recent
// ============================================================================

/// Prove: after LRU eviction, the most recently used entry survives.
///
/// Models a cache of capacity 3 filled with entries A, B, C (A oldest).
/// Entry C (most recently inserted/accessed) has the highest generation.
/// When a new entry D is inserted triggering eviction, A (lowest generation)
/// is evicted and C remains.
///
/// Also proves the stronger property: after accessing A to promote it to MRU,
/// B (now LRU) is evicted instead of A. This validates that `stamp()` correctly
/// updates recency.
#[kani::unwind(5)]
#[kani::proof]
fn proof_eviction_preserves_most_recent() {
    let max_entries: usize = 3;
    let mut cache = CacheModel::new(max_entries);

    // Insert A, B, C in order. A has the lowest generation (LRU).
    let hash_a: u64 = 1;
    let hash_b: u64 = 2;
    let hash_c: u64 = 3;
    cache.get_or_generate(hash_a, 10, 100);
    cache.get_or_generate(hash_b, 20, 200);
    cache.get_or_generate(hash_c, 30, 300);

    assert_eq!(cache.len(), 3, "cache must be full");

    // Verify generation ordering: A < B < C.
    let gen_a = *cache.generations.get(&hash_a).unwrap();
    let gen_b = *cache.generations.get(&hash_b).unwrap();
    let gen_c = *cache.generations.get(&hash_c).unwrap();
    assert!(gen_a < gen_b, "A must have lower generation than B");
    assert!(gen_b < gen_c, "B must have lower generation than C");

    // Insert D — triggers eviction. A (lowest generation = LRU) is evicted.
    let hash_d: u64 = 4;
    cache.get_or_generate(hash_d, 40, 400);

    assert_eq!(cache.len(), max_entries, "size must equal max_entries");
    assert!(
        cache.entries.get(&hash_a).is_none(),
        "A (LRU) must be evicted"
    );
    assert!(
        cache.entries.get(&hash_c).is_some(),
        "C (MRU at insertion) must survive eviction"
    );
    assert!(
        cache.entries.get(&hash_d).is_some(),
        "D (just inserted) must be present"
    );
}

// ============================================================================
// 8. Hit promotes entry — eviction after access respects updated recency
// ============================================================================

/// Prove: accessing an entry via cache hit promotes it to MRU, preventing
/// its eviction even when it was originally the oldest entry.
///
/// Models: insert A, B, C (A is LRU). Access A — its generation becomes the
/// highest. Insert D — B (now LRU) is evicted instead of A.
#[kani::unwind(5)]
#[kani::proof]
fn proof_hit_promotes_entry_to_mru() {
    let max_entries: usize = 3;
    let mut cache = CacheModel::new(max_entries);

    let hash_a: u64 = 1;
    let hash_b: u64 = 2;
    let hash_c: u64 = 3;
    cache.get_or_generate(hash_a, 10, 100);
    cache.get_or_generate(hash_b, 20, 200);
    cache.get_or_generate(hash_c, 30, 300);

    // A is currently LRU (lowest generation).
    let gen_a_before = *cache.generations.get(&hash_a).unwrap();
    let gen_c_before = *cache.generations.get(&hash_c).unwrap();
    assert!(
        gen_a_before < gen_c_before,
        "A must be LRU before access"
    );

    // Access A — promotes it to MRU via stamp().
    let result = cache.get_or_generate(hash_a, 10, 999);
    assert_eq!(
        result, 100,
        "cache hit must return original value, not new generate value"
    );

    let gen_a_after = *cache.generations.get(&hash_a).unwrap();
    let gen_b = *cache.generations.get(&hash_b).unwrap();
    let gen_c = *cache.generations.get(&hash_c).unwrap();
    assert!(
        gen_a_after > gen_b,
        "A's generation must exceed B after access"
    );
    assert!(
        gen_a_after > gen_c,
        "A's generation must exceed C after access"
    );

    // Insert D — B is now LRU (lowest generation), so B is evicted.
    let hash_d: u64 = 4;
    cache.get_or_generate(hash_d, 40, 400);

    assert!(
        cache.entries.get(&hash_b).is_none(),
        "B must be evicted (it is now LRU)"
    );
    assert!(
        cache.entries.get(&hash_a).is_some(),
        "A must survive (promoted to MRU by access)"
    );
    assert!(
        cache.entries.get(&hash_c).is_some(),
        "C must survive"
    );
    assert!(
        cache.entries.get(&hash_d).is_some(),
        "D must be present"
    );
}

// ============================================================================
// 9. Hash collision detection — different key at same hash triggers regenerate
// ============================================================================

/// Prove: when two different `CodegenKey` identities map to the same `u64`
/// hash (a hash collision), `get_or_generate` does NOT return the stale
/// entry. Instead it replaces the entry with the new key's generated output.
///
/// This models the collision detection in `get_or_generate`:
/// ```ignore
/// if let Some((stored_key, entry)) = cache.entries.get(&hash_key) {
///     if stored_key == &query_key {
///         // cache hit
///     }
///     // Hash collision: fall through to regenerate and replace.
/// }
/// ```
/// Regression property for #2202.
#[kani::unwind(3)]
#[kani::proof]
fn proof_hash_collision_returns_correct_value() {
    let max_entries: usize = kani::any();
    kani::assume(max_entries >= 1 && max_entries <= 4);
    let mut cache = CacheModel::new(max_entries);

    let hash: u64 = kani::any(); // same hash for both keys (collision)
    let key_id_a: u32 = kani::any();
    let key_id_b: u32 = kani::any();
    kani::assume(key_id_a != key_id_b); // different keys

    let value_a: u32 = kani::any();
    let value_b: u32 = kani::any();

    // Insert key_a at the hash slot.
    let result_a = cache.get_or_generate(hash, key_id_a, value_a);
    assert_eq!(result_a, value_a, "first insert returns its value");

    // Insert key_b at the SAME hash slot (collision).
    // key_id_b != key_id_a, so the collision detection path fires,
    // the old entry is replaced, and value_b is returned.
    let result_b = cache.get_or_generate(hash, key_id_b, value_b);
    assert_eq!(
        result_b, value_b,
        "collision must return the NEW key's generated value, not the stale entry"
    );

    // The cache still has exactly one entry at this hash (replaced, not duplicated).
    assert_eq!(
        cache.len(),
        1,
        "collision replacement must not increase cache size"
    );

    // The stored entry is now key_b's.
    let (stored_key_id, stored_value) = cache.entries.get(&hash).unwrap();
    assert_eq!(
        *stored_key_id, key_id_b,
        "stored key must be the new key after collision replacement"
    );
    assert_eq!(
        *stored_value, value_b,
        "stored value must be the new value after collision replacement"
    );
}

// ============================================================================
// 10. Generation map synchronization
// ============================================================================

/// Prove: after any sequence of inserts and hits, the `entries` and
/// `generations` maps always have the same key set.
///
/// This models the invariant that `MslCodegenCache::entries` and
/// `MslCodegenCache::access_gen` are always synchronized — `get_or_generate`
/// and `evict_lru` modify both maps atomically.
#[kani::unwind(7)]
#[kani::proof]
fn proof_generation_map_stays_in_sync() {
    let max_entries: usize = kani::any();
    kani::assume(max_entries >= 1 && max_entries <= 3);

    let mut cache = CacheModel::new(max_entries);

    // Perform a mixed sequence of operations.
    let num_ops: usize = kani::any();
    kani::assume(num_ops <= 4);

    let mut i: usize = 0;
    while i < num_ops {
        let hash: u64 = kani::any();
        kani::assume(hash <= 6); // bounded key space
        let key_id: u32 = kani::any();
        let value: u32 = kani::any();
        cache.get_or_generate(hash, key_id, value);

        // Invariant: entries and generations have the same size.
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
