// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

//! LRU segment cache for [`CompiledKokoro`].
//!
//! Each compiled segment is keyed by a shape dimension (e.g. `seq_len`,
//! `t_mel`, `total_samples`). Without caching, every new text length
//! triggers a full recompile (trace + compile + GPU weight upload).
//! This LRU cache keeps the last N compiled models alive, eliminating
//! recompilation for frequently-seen shapes.
//!
//! Part of #2626, #2218.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::buffer::MetalBuffer;
use crate::compiled_model::{CompiledModel, CompiledModelDef};
use crate::segment_cache::{SegmentCacheConfig, SegmentCacheStats};

/// Default number of compiled models to cache per segment.
const DEFAULT_CAPACITY: usize = 4;

/// Default byte budget for planned buffers per segment (512 MB).
///
/// Each cached `CompiledModel` holds a `cached_planned_buf` GPU buffer sized to
/// `BufferPlan.total_bytes`. At ~80 MB per generator model, 512 MB fits ~6
/// cached segments — enough for typical chorus synthesis where text lengths vary.
/// The previous 128 MB budget only fit 1-2 generator models, causing LRU
/// eviction thrashing when text lengths alternated.
///
/// Override via `SegmentCacheConfig::byte_budget` for workload-specific tuning.
/// Part of #3079 (RSS optimization), #4187 (cache thrashing fix).
const DEFAULT_BYTE_BUDGET: usize = 512 * 1024 * 1024;

/// LRU cache mapping a shape key (single `usize` dimension) to a
/// [`CompiledModel`] with pre-uploaded GPU weights.
///
/// Capacity-bounded: when full, the least-recently-used entry is evicted.
/// Access promotes entries to MRU position.
///
/// GPU weight buffers are shared across all cached models via
/// `MetalBuffer::alias()` (ARC reference counting, zero-copy). The first
/// model inserted populates the shared weight store; subsequent models alias
/// from it instead of re-uploading identical weights to GPU. Evicting an
/// entry drops its dispatch plan but GPU weight memory is retained by the
/// shared store until the `SegmentCache` itself is dropped. Part of #2630.
pub(crate) struct SegmentCache {
    entries: VecDeque<(usize, CompiledModel)>,
    capacity: usize,
    /// Maximum total `buffer_plan.total_bytes` across all cached entries.
    /// Evicts LRU entries when inserting would exceed this budget.
    /// Part of #3079 (RSS optimization).
    byte_budget: usize,
    /// Running total of `buffer_plan.total_bytes` across all entries.
    /// Maintained incrementally to avoid O(N) recomputation per eviction
    /// iteration (was O(N²) in the eviction loop before this fix).
    tracked_total_bytes: usize,
    /// Shared GPU weight buffers populated from the first compiled model.
    /// Subsequent compilations alias from this store instead of uploading.
    shared_weights: Option<HashMap<(usize, String), MetalBuffer>>,
    /// Cumulative hit/miss/eviction counters.
    stats: SegmentCacheStats,
}

impl SegmentCache {
    /// Create an empty cache with the default capacity and byte budget.
    pub(super) fn new() -> Self {
        Self {
            entries: VecDeque::with_capacity(DEFAULT_CAPACITY),
            capacity: DEFAULT_CAPACITY,
            byte_budget: DEFAULT_BYTE_BUDGET,
            tracked_total_bytes: 0,
            shared_weights: None,
            stats: SegmentCacheStats::default(),
        }
    }

    /// Create an empty cache configured by a [`SegmentCacheConfig`].
    ///
    /// Uses the config's `max_segments_per_step` as the capacity (clamped
    /// to a minimum of 1). The byte budget uses the config's `byte_budget`
    /// if set, otherwise falls back to `DEFAULT_BYTE_BUDGET` (512 MB).
    ///
    /// Part of #3634, #4187.
    pub(super) fn with_config(config: &SegmentCacheConfig) -> Self {
        let cap = config.max_segments_per_step.max(1);
        Self {
            entries: VecDeque::with_capacity(cap),
            capacity: cap,
            byte_budget: config.byte_budget.unwrap_or(DEFAULT_BYTE_BUDGET),
            tracked_total_bytes: 0,
            shared_weights: None,
            stats: SegmentCacheStats::default(),
        }
    }

    /// Create a cache pre-seeded with shared weights and custom config.
    ///
    /// Combines [`with_config`](Self::with_config) and
    /// [`with_shared_weights`](Self::with_shared_weights): uses the config's
    /// capacity and byte budget while seeding GPU weight buffers for
    /// cross-instance sharing.
    ///
    /// Part of #3634, #2740, #4187.
    pub(super) fn with_config_and_shared_weights(
        config: &SegmentCacheConfig,
        weights: HashMap<(usize, String), MetalBuffer>,
    ) -> Self {
        let cap = config.max_segments_per_step.max(1);
        Self {
            entries: VecDeque::with_capacity(cap),
            capacity: cap,
            byte_budget: config.byte_budget.unwrap_or(DEFAULT_BYTE_BUDGET),
            tracked_total_bytes: 0,
            shared_weights: Some(weights),
            stats: SegmentCacheStats::default(),
        }
    }

    /// Create an empty cache pre-seeded with shared GPU weight buffers.
    ///
    /// The new cache has no compiled models, but when the first model is
    /// compiled it will alias weight buffers from `weights` instead of
    /// re-uploading to GPU. This enables cross-instance weight sharing
    /// for [`clone_dispatch()`](super::CompiledKokoro::clone_dispatch).
    ///
    /// Part of #2740.
    pub(super) fn with_shared_weights(weights: HashMap<(usize, String), MetalBuffer>) -> Self {
        Self {
            entries: VecDeque::with_capacity(DEFAULT_CAPACITY),
            capacity: DEFAULT_CAPACITY,
            byte_budget: DEFAULT_BYTE_BUDGET,
            tracked_total_bytes: 0,
            shared_weights: Some(weights),
            stats: SegmentCacheStats::default(),
        }
    }

    /// Returns the shared weight buffers, if any model has been compiled.
    ///
    /// Callers pass this to `CompiledModel::builder(..).shared_weights(w).build()`
    /// to alias existing GPU buffers instead of re-uploading.
    pub(super) fn shared_weights(&self) -> Option<&HashMap<(usize, String), MetalBuffer>> {
        self.shared_weights.as_ref()
    }

    /// Total bytes of shared GPU weight buffers in this cache.
    ///
    /// Returns 0 if no segment has been compiled yet (no shared weights).
    /// Used by [`CompiledKokoro::gpu_weight_bytes`] for memory diagnostics.
    pub(super) fn shared_weight_bytes(&self) -> usize {
        self.shared_weights
            .as_ref()
            .map(|w| w.values().map(MetalBuffer::len).sum())
            .unwrap_or(0)
    }

    /// Number of shared GPU weight buffers in this cache.
    pub(super) fn shared_weight_count(&self) -> usize {
        self.shared_weights.as_ref().map(HashMap::len).unwrap_or(0)
    }

    /// Total bytes of `cached_planned_buf` across all cached entries.
    ///
    /// Each cached `CompiledModel` holds a contiguous GPU buffer sized to
    /// `BufferPlan.total_bytes` for intermediate sub-allocation. With up to
    /// `capacity` entries per segment × 8 segments, these buffers can sum to
    /// hundreds of MB. Used by [`MemoryBreakdown`] for memory attribution.
    pub(super) fn total_planned_buf_bytes(&self) -> usize {
        // Debug invariant: tracked total must match actual sum. This catches
        // any insert/evict path that fails to update tracked_total_bytes.
        #[cfg(debug_assertions)]
        {
            let actual: usize = self
                .entries
                .iter()
                .map(|(_, model)| model.buffer_plan().total_bytes)
                .sum();
            debug_assert_eq!(
                self.tracked_total_bytes, actual,
                "tracked_total_bytes desync: tracked={}, actual={}",
                self.tracked_total_bytes, actual,
            );
        }
        self.tracked_total_bytes
    }

    /// Check whether `key` is cached (does NOT promote to MRU).
    ///
    /// For hot paths, prefer `get().is_none()` to avoid a separate scan.
    /// This method is useful for read-only diagnostic checks that must not
    /// disturb LRU ordering.
    pub(super) fn contains_key(&self, key: usize) -> bool {
        self.entries.iter().any(|(k, _)| *k == key)
    }

    /// Look up `key`, promoting it to MRU position on hit.
    ///
    /// Returns a reference to the cached [`CompiledModel`], or `None` on miss.
    pub(super) fn get(&mut self, key: usize) -> Option<&CompiledModel> {
        let idx = self.entries.iter().position(|(k, _)| *k == key);
        match idx {
            Some(i) => {
                self.stats.hits += 1;
                if i > 0 {
                    if let Some(entry) = self.entries.remove(i) {
                        self.entries.push_front(entry);
                    }
                }
                self.entries.front().map(|(_, v)| v)
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Look up `key` and return a shared `Arc<CompiledModelDef>` if cached.
    ///
    /// Unlike [`get()`](Self::get), this does NOT promote the entry to MRU
    /// position. The caller receives a cheap `Arc::clone` of the compiled
    /// model definition, which can be used to create new `CompiledModel`
    /// execution instances via [`CompiledModel::from_shared()`].
    ///
    /// This enables cross-instance sharing: multiple chorus voices can
    /// each hold their own `CompiledModel` (with independent execution
    /// caches) while sharing the same immutable compiled pipeline, GPU
    /// weight buffers, and step metadata.
    ///
    /// Returns `None` if `key` is not in the cache.
    ///
    /// Part of #4104.
    pub(super) fn get_shared_def(&self, key: usize) -> Option<Arc<CompiledModelDef>> {
        self.entries
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, model)| model.share_def())
    }

    /// Returns the number of distinct `CompiledModelDef` allocations in
    /// the cache.
    ///
    /// When entries share a `CompiledModelDef` via `Arc` (e.g., after
    /// `clone_warm`), this counts unique `Arc` allocations, not entries.
    /// For an un-warmed cache, this equals `len()`. For a warmed cache
    /// cloned from another, it may be less than `len()` if entries share
    /// defs across caches (though typical usage creates independent Arcs).
    ///
    /// Part of #4104.
    pub(super) fn distinct_def_count(&self) -> usize {
        let ptrs: std::collections::HashSet<*const CompiledModelDef> = self
            .entries
            .iter()
            .map(|(_, model)| Arc::as_ptr(&model.def))
            .collect();
        ptrs.len()
    }

    /// Insert a compiled model for `key`, evicting LRU entries if at capacity
    /// or over the byte budget for planned buffers.
    ///
    /// On first insert, extracts weight buffer aliases into the shared store
    /// so subsequent compilations can skip GPU weight upload (#2630).
    /// If `key` already exists, the old entry is replaced and promoted to MRU.
    ///
    /// Byte-budget eviction (#3079): evicts LRU entries until total
    /// `buffer_plan.total_bytes` + new entry fits within `byte_budget`.
    /// A single model exceeding the budget is still cached (alone).
    pub(super) fn insert(&mut self, key: usize, model: CompiledModel) {
        // Populate shared weight store from the first model's GPU buffers.
        // Subsequent models already use aliases from this store, so their
        // weight_buffer_aliases() would just re-alias the same buffers.
        if self.shared_weights.is_none() {
            self.shared_weights = Some(model.weight_buffer_aliases());
        }
        // Remove existing entry with same key (will be re-inserted at front).
        if let Some(idx) = self.entries.iter().position(|(k, _)| *k == key) {
            if let Some((_, evicted)) = self.entries.remove(idx) {
                self.tracked_total_bytes -= evicted.buffer_plan().total_bytes;
            }
        }
        let new_bytes = model.buffer_plan().total_bytes;
        // Evict LRU entries until both count and byte budget are satisfied.
        // tracked_total_bytes is O(1); was O(N) per iteration before this fix.
        // The !is_empty() guard ensures a single oversized model is still cached.
        while self.entries.len() >= self.capacity
            || (self.tracked_total_bytes + new_bytes > self.byte_budget && !self.entries.is_empty())
        {
            if let Some((_, evicted)) = self.entries.pop_back() {
                self.tracked_total_bytes -= evicted.buffer_plan().total_bytes;
                self.stats.evictions += 1;
            }
        }
        self.tracked_total_bytes += new_bytes;
        self.entries.push_front((key, model));
    }

    /// Return the most-recently-used entry (front of queue), if any.
    pub(super) fn most_recent(&self) -> Option<&(usize, CompiledModel)> {
        self.entries.front()
    }

    /// Returns the number of cached entries.
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns the maximum capacity of this cache.
    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the largest `buffer_plan.total_bytes` across all cached entries.
    ///
    /// Used by [`CompiledKokoro::estimate_arena_bytes`] to right-size the
    /// thread-local arena before synthesis, preventing growth events during
    /// the hot path.
    ///
    /// Part of #4289.
    pub(super) fn max_entry_buffer_plan_bytes(&self) -> usize {
        self.entries
            .iter()
            .map(|(_, model)| model.buffer_plan().total_bytes)
            .max()
            .unwrap_or(0)
    }

    /// Remove all cached entries, preserving shared weights and capacity.
    ///
    /// After clearing, subsequent `ensure_seg_*` calls will recompile segments
    /// (using the shared weights for zero-copy GPU weight aliasing). This is
    /// used when peephole configs change and previously-compiled entries need
    /// to be recompiled with the new config.
    ///
    /// Part of #3828.
    pub(super) fn clear(&mut self) {
        self.entries.clear();
        self.tracked_total_bytes = 0;
    }

    /// Clone this cache into a new cache with pre-populated compiled entries.
    ///
    /// For each cached `(key, CompiledModel)` entry, the new cache gets a
    /// `CompiledModel` that shares the immutable `Arc<CompiledModelDef>`
    /// (compiled steps, pipeline states, GPU weight buffers) but has its own
    /// fresh execution caches (`cached_planned_buf`, `cached_icbs`). This
    /// eliminates recompilation (~1.2s per segment per shape) while keeping
    /// execution state independent.
    ///
    /// Weight buffers are aliased (zero-copy via `MetalBuffer::alias()`).
    /// LRU ordering is preserved from the source cache.
    ///
    /// Part of #4104.
    pub(super) fn clone_warm(&self, config: &SegmentCacheConfig) -> Self {
        // Start with weight aliases for future compilations of unseen shapes.
        let mut cache = match &self.shared_weights {
            Some(w) => {
                let aliases = w.iter().map(|(k, b)| (k.clone(), b.alias())).collect();
                Self::with_config_and_shared_weights(config, aliases)
            }
            None => Self::with_config(config),
        };

        // Pre-populate with shared compiled models. Insert in reverse order
        // (back to front) so that MRU entries end up at the front of the
        // new cache (insert() pushes to front).
        for &(key, ref model) in self.entries.iter().rev() {
            let shared_def = model.share_def();
            let warm_model = CompiledModel::from_shared(shared_def);
            cache.insert(key, warm_model);
        }

        cache
    }

    /// Returns the byte budget for planned GPU buffers.
    ///
    /// Part of #4187.
    #[cfg(test)]
    fn byte_budget(&self) -> usize {
        self.byte_budget
    }

    /// Returns cumulative cache statistics (hits, misses, evictions, total_bytes).
    ///
    /// `total_bytes` reflects the current `tracked_total_bytes` (sum of
    /// `buffer_plan.total_bytes` across all cached entries).
    pub(super) fn stats(&self) -> SegmentCacheStats {
        SegmentCacheStats {
            total_bytes: self.tracked_total_bytes,
            ..self.stats
        }
    }

    /// Reset all statistics counters to zero.
    ///
    /// Does not affect `total_bytes` (which reflects current cache state,
    /// not cumulative history).
    pub(super) fn reset_stats(&mut self) {
        self.stats = SegmentCacheStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_cache_is_empty() {
        let cache = SegmentCache::new();
        assert_eq!(cache.len(), 0);
        assert!(!cache.contains_key(42));
        assert!(cache.most_recent().is_none());
    }

    #[test]
    fn test_new_cache_default_capacity() {
        let cache = SegmentCache::new();
        assert_eq!(cache.capacity(), DEFAULT_CAPACITY);
    }

    #[test]
    fn test_with_config_custom_capacity() {
        let config = SegmentCacheConfig {
            max_segments_per_step: 8,
            ..SegmentCacheConfig::default()
        };
        let cache = SegmentCache::with_config(&config);
        assert_eq!(cache.capacity(), 8);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_with_config_clamps_to_one() {
        let config = SegmentCacheConfig {
            max_segments_per_step: 0,
            ..SegmentCacheConfig::default()
        };
        let cache = SegmentCache::with_config(&config);
        assert_eq!(
            cache.capacity(),
            1,
            "capacity should be clamped to minimum 1"
        );
    }

    #[test]
    fn test_with_config_and_shared_weights() {
        let config = SegmentCacheConfig {
            max_segments_per_step: 16,
            ..SegmentCacheConfig::default()
        };
        let cache = SegmentCache::with_config_and_shared_weights(&config, HashMap::new());
        assert_eq!(cache.capacity(), 16);
        assert!(cache.shared_weights().is_some());
    }

    #[test]
    fn test_with_shared_weights_empty_map() {
        let cache = SegmentCache::with_shared_weights(HashMap::new());
        assert_eq!(cache.len(), 0, "cache should start empty");
        assert!(
            cache.shared_weights().is_some(),
            "shared_weights should be pre-seeded (even if empty)"
        );
    }

    // -- byte_budget wiring tests (Part of #4187) --

    #[test]
    fn test_default_byte_budget_is_512mb() {
        let cache = SegmentCache::new();
        assert_eq!(
            cache.byte_budget(),
            512 * 1024 * 1024,
            "DEFAULT_BYTE_BUDGET should be 512 MB"
        );
    }

    #[test]
    fn test_byte_budget_from_config() {
        let config = SegmentCacheConfig {
            byte_budget: Some(256 * 1024 * 1024),
            ..SegmentCacheConfig::default()
        };
        let cache = SegmentCache::with_config(&config);
        assert_eq!(
            cache.byte_budget(),
            256 * 1024 * 1024,
            "with_config should use config byte_budget when set"
        );
    }

    #[test]
    fn test_byte_budget_from_config_none_uses_default() {
        let config = SegmentCacheConfig {
            byte_budget: None,
            ..SegmentCacheConfig::default()
        };
        let cache = SegmentCache::with_config(&config);
        assert_eq!(
            cache.byte_budget(),
            512 * 1024 * 1024,
            "with_config should use DEFAULT_BYTE_BUDGET when config is None"
        );
    }

    #[test]
    fn test_byte_budget_from_config_and_shared_weights() {
        let config = SegmentCacheConfig {
            byte_budget: Some(1024 * 1024 * 1024),
            ..SegmentCacheConfig::default()
        };
        let cache = SegmentCache::with_config_and_shared_weights(&config, HashMap::new());
        assert_eq!(
            cache.byte_budget(),
            1024 * 1024 * 1024,
            "with_config_and_shared_weights should use config byte_budget when set"
        );
    }

    // -- clone_warm tests (Part of #4104) --

    #[test]
    fn test_clone_warm_empty_cache_produces_empty() {
        let cache = SegmentCache::new();
        let config = SegmentCacheConfig::default();
        let warm = cache.clone_warm(&config);
        assert_eq!(warm.len(), 0, "clone_warm of empty cache should be empty");
        assert!(
            warm.shared_weights().is_none(),
            "empty cache has no shared weights to propagate"
        );
    }

    #[test]
    fn test_clone_warm_with_shared_weights_propagates() {
        // Empty HashMap is enough — shared_weights being Some(empty_map) still
        // signals "weight sharing is active" vs None (not yet compiled).
        let cache = SegmentCache::with_shared_weights(HashMap::new());
        let config = SegmentCacheConfig::default();
        let warm = cache.clone_warm(&config);
        assert!(
            warm.shared_weights().is_some(),
            "clone_warm should propagate shared weights"
        );
    }

    #[test]
    fn test_clone_warm_uses_config_capacity() {
        let cache = SegmentCache::new();
        let config = SegmentCacheConfig {
            max_segments_per_step: 16,
            ..SegmentCacheConfig::default()
        };
        let warm = cache.clone_warm(&config);
        assert_eq!(
            warm.capacity(),
            16,
            "clone_warm should use provided config capacity"
        );
    }

    #[test]
    fn test_clone_warm_uses_config_byte_budget() {
        let cache = SegmentCache::new();
        let config = SegmentCacheConfig {
            byte_budget: Some(1024 * 1024 * 1024),
            ..SegmentCacheConfig::default()
        };
        let warm = cache.clone_warm(&config);
        assert_eq!(
            warm.byte_budget(),
            1024 * 1024 * 1024,
            "clone_warm should propagate byte_budget from config"
        );
    }

    #[test]
    fn test_clone_warm_uses_default_byte_budget_when_none() {
        let cache = SegmentCache::new();
        let config = SegmentCacheConfig {
            byte_budget: None,
            ..SegmentCacheConfig::default()
        };
        let warm = cache.clone_warm(&config);
        assert_eq!(
            warm.byte_budget(),
            512 * 1024 * 1024,
            "clone_warm with byte_budget=None should use DEFAULT_BYTE_BUDGET"
        );
    }

    // -- clear tests (Part of #3828) --

    #[test]
    fn test_clear_empties_entries() {
        let mut cache = SegmentCache::new();
        // Shared weights start as None; clear preserves that.
        assert!(cache.shared_weights().is_none());
        cache.clear();
        assert_eq!(cache.len(), 0, "clear on empty cache is a no-op");
        assert!(cache.shared_weights().is_none(), "clear preserves None shared_weights");
    }

    #[test]
    fn test_clear_preserves_shared_weights() {
        // Pre-seed with an empty HashMap — Some(empty) signals "weight sharing
        // is active" vs None (not yet compiled). The key invariant: clear()
        // must NOT reset shared_weights to None.
        let mut cache = SegmentCache::with_shared_weights(HashMap::new());
        assert!(cache.shared_weights().is_some(), "pre-seeded shared weights");

        cache.clear();
        assert_eq!(cache.len(), 0, "entries should be empty after clear");
        assert_eq!(cache.total_planned_buf_bytes(), 0, "tracked bytes should be 0 after clear");
        assert!(
            cache.shared_weights().is_some(),
            "shared_weights must survive clear for zero-copy aliasing"
        );
    }

    #[test]
    fn test_clear_preserves_capacity() {
        let config = SegmentCacheConfig {
            max_segments_per_step: 12,
            ..SegmentCacheConfig::default()
        };
        let mut cache = SegmentCache::with_config(&config);
        assert_eq!(cache.capacity(), 12);
        cache.clear();
        assert_eq!(
            cache.capacity(),
            12,
            "capacity should be preserved after clear"
        );
    }

    #[test]
    fn test_clear_resets_tracked_bytes() {
        let mut cache = SegmentCache::new();
        // tracked_total_bytes starts at 0 and clear should keep it at 0.
        assert_eq!(cache.total_planned_buf_bytes(), 0);
        cache.clear();
        assert_eq!(
            cache.total_planned_buf_bytes(),
            0,
            "tracked_total_bytes should be 0 after clear"
        );
    }

    // -- get_shared_def tests (Part of #4104) --

    #[test]
    fn test_get_shared_def_empty_cache_returns_none() {
        let cache = SegmentCache::new();
        assert!(
            cache.get_shared_def(42).is_none(),
            "empty cache should return None for any key"
        );
    }

    #[test]
    fn test_get_shared_def_miss_returns_none() {
        let mut cache = SegmentCache::new();
        let model = CompiledModel::empty();
        cache.insert(10, model);
        assert!(
            cache.get_shared_def(99).is_none(),
            "uncached key should return None"
        );
    }

    #[test]
    fn test_get_shared_def_hit_returns_arc() {
        let mut cache = SegmentCache::new();
        let model = CompiledModel::empty();
        let original_def = model.share_def();
        cache.insert(42, model);

        let shared = cache
            .get_shared_def(42)
            .expect("cached key should return Some");

        // The returned Arc should point to the same allocation as the
        // model's internal def.
        assert!(
            Arc::ptr_eq(&shared, &original_def),
            "get_shared_def should return the same Arc as share_def"
        );
    }

    #[test]
    fn test_get_shared_def_same_key_returns_same_arc() {
        let mut cache = SegmentCache::new();
        cache.insert(7, CompiledModel::empty());

        let arc1 = cache.get_shared_def(7).unwrap();
        let arc2 = cache.get_shared_def(7).unwrap();
        assert!(
            Arc::ptr_eq(&arc1, &arc2),
            "repeated get_shared_def for the same key should return the same Arc"
        );
    }

    #[test]
    fn test_get_shared_def_does_not_promote_to_mru() {
        let mut cache = SegmentCache::new();
        cache.insert(1, CompiledModel::empty());
        cache.insert(2, CompiledModel::empty());
        // MRU order after inserts: [2, 1]

        // get_shared_def(1) should NOT promote key=1.
        let _ = cache.get_shared_def(1);

        // MRU should still be key=2 (most_recent() returns the front).
        let (mru_key, _) = cache.most_recent().expect("cache is not empty");
        assert_eq!(
            *mru_key, 2,
            "get_shared_def should not change LRU ordering"
        );
    }

    #[test]
    fn test_get_shared_def_does_not_affect_stats() {
        let mut cache = SegmentCache::new();
        cache.insert(5, CompiledModel::empty());

        let stats_before = cache.stats();
        let _ = cache.get_shared_def(5);
        let _ = cache.get_shared_def(999); // miss
        let stats_after = cache.stats();

        assert_eq!(
            stats_before.hits, stats_after.hits,
            "get_shared_def should not increment hits"
        );
        assert_eq!(
            stats_before.misses, stats_after.misses,
            "get_shared_def should not increment misses"
        );
    }

    #[test]
    fn test_get_shared_def_enables_independent_instances() {
        let mut cache = SegmentCache::new();
        cache.insert(42, CompiledModel::empty());

        let shared = cache.get_shared_def(42).unwrap();
        let instance_a = CompiledModel::from_shared(Arc::clone(&shared));
        let instance_b = CompiledModel::from_shared(Arc::clone(&shared));

        // Both instances share the same definition (same step count, inputs).
        assert_eq!(instance_a.num_steps(), instance_b.num_steps());
        assert_eq!(instance_a.num_inputs(), instance_b.num_inputs());
        assert_eq!(instance_a.num_steps(), 0); // empty model

        // The definition Arc is shared (same allocation).
        let def_a = instance_a.share_def();
        let def_b = instance_b.share_def();
        assert!(
            Arc::ptr_eq(&def_a, &def_b),
            "instances from same shared def should have the same Arc"
        );
    }

    // -- distinct_def_count tests (Part of #4104) --

    #[test]
    fn test_distinct_def_count_empty() {
        let cache = SegmentCache::new();
        assert_eq!(cache.distinct_def_count(), 0);
    }

    #[test]
    fn test_distinct_def_count_each_insert_is_distinct() {
        let mut cache = SegmentCache::new();
        cache.insert(1, CompiledModel::empty());
        cache.insert(2, CompiledModel::empty());
        cache.insert(3, CompiledModel::empty());
        // Each CompiledModel::empty() creates its own Arc<CompiledModelDef>.
        assert_eq!(
            cache.distinct_def_count(),
            3,
            "each inserted model should have a distinct def"
        );
    }

    #[test]
    fn test_distinct_def_count_shared_defs_after_clone_warm() {
        let mut source = SegmentCache::new();
        source.insert(1, CompiledModel::empty());
        source.insert(2, CompiledModel::empty());

        let config = SegmentCacheConfig::default();
        let warm = source.clone_warm(&config);

        // clone_warm creates new CompiledModel instances via from_shared,
        // each with a cloned Arc. The warm cache entries share defs with
        // source entries, but within the warm cache each entry has its own
        // Arc (from a different source entry).
        assert_eq!(warm.len(), 2);
        assert_eq!(
            warm.distinct_def_count(),
            2,
            "warm clone should have 2 distinct defs (one per source entry)"
        );
    }

    // -- clone_warm Arc-sharing tests (Part of #4104) --

    #[test]
    fn test_clone_warm_shares_defs_with_source() {
        let mut source = SegmentCache::new();
        let model = CompiledModel::empty();
        let source_def = model.share_def();
        source.insert(42, model);

        let config = SegmentCacheConfig::default();
        let warm = source.clone_warm(&config);

        // The warm clone's entry for key=42 should share the same
        // Arc<CompiledModelDef> as the source entry.
        let warm_def = warm
            .get_shared_def(42)
            .expect("warm clone should have key=42");
        assert!(
            Arc::ptr_eq(&warm_def, &source_def),
            "clone_warm entry should share Arc<CompiledModelDef> with source"
        );
    }

    #[test]
    fn test_clone_warm_preserves_lru_order() {
        let mut source = SegmentCache::new();
        // Insert in order: 1, 2, 3. MRU = 3 (front), LRU = 1 (back).
        source.insert(1, CompiledModel::empty());
        source.insert(2, CompiledModel::empty());
        source.insert(3, CompiledModel::empty());

        let config = SegmentCacheConfig::default();
        let warm = source.clone_warm(&config);

        // MRU in warm should be 3 (same as source).
        let (mru_key, _) = warm.most_recent().expect("warm cache not empty");
        assert_eq!(
            *mru_key, 3,
            "clone_warm should preserve MRU ordering from source"
        );
    }

    #[test]
    fn test_clone_warm_independent_mutation() {
        let mut source = SegmentCache::new();
        source.insert(1, CompiledModel::empty());
        source.insert(2, CompiledModel::empty());

        let config = SegmentCacheConfig::default();
        let mut warm = source.clone_warm(&config);

        // Mutating the warm clone should not affect the source.
        warm.insert(3, CompiledModel::empty());
        assert_eq!(warm.len(), 3);
        assert_eq!(source.len(), 2, "source should not be affected by warm clone mutation");
    }

    #[test]
    fn test_shared_def_memory_reduction() {
        // Verify that shared defs use the same underlying allocation.
        let mut cache = SegmentCache::new();
        let model = CompiledModel::empty();
        cache.insert(42, model);

        let def1 = cache.get_shared_def(42).unwrap();
        let def2 = cache.get_shared_def(42).unwrap();

        // Arc strong_count should reflect the sharing: the cache entry
        // holds one, and we hold two more.
        assert!(
            Arc::strong_count(&def1) >= 3,
            "Arc strong count should reflect shared references: got {}",
            Arc::strong_count(&def1)
        );
        assert!(
            Arc::ptr_eq(&def1, &def2),
            "same allocation should be shared, not duplicated"
        );
    }
}
