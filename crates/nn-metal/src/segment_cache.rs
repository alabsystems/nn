// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-shape segment caching with configurable LRU eviction.
//!
//! [`ShapeKeyedCache`] is a generic LRU cache keyed by tensor shape
//! (`Vec<usize>`). It supports configurable capacity and eviction policy,
//! designed for TTS pipelines where variable text lengths cause cache
//! thrashing with single-entry caches.
//!
//! [`SegmentCacheConfig`] controls per-step cache capacity and eviction
//! policy. Default: `max_segments_per_step = 4`, `eviction = Lru`.

/// Cache hit/miss/eviction statistics for segment caches.
///
/// Tracks how often cache lookups succeed (hits) vs trigger recompilation
/// (misses), and how many entries have been evicted due to capacity or byte
/// budget constraints.
///
/// These counters are cumulative from cache creation (or last reset). Use
/// [`hit_rate()`](Self::hit_rate) for a derived metric.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SegmentCacheStats {
    /// Number of `get()` calls that found the requested key.
    pub hits: usize,
    /// Number of `get()` calls that did not find the requested key.
    pub misses: usize,
    /// Number of entries evicted (LRU eviction or byte-budget eviction).
    pub evictions: usize,
    /// Current total bytes tracked across all cached entries.
    /// For `ShapeKeyedCache` this is always 0 (no byte tracking).
    /// For `SegmentCache` this is `tracked_total_bytes`.
    pub total_bytes: usize,
}

impl SegmentCacheStats {
    /// Cache hit rate as a fraction in `[0.0, 1.0]`.
    ///
    /// Returns 0.0 if no lookups have been recorded.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    /// Total number of lookups (hits + misses).
    #[must_use]
    pub fn lookups(&self) -> usize {
        self.hits + self.misses
    }
}

/// Configuration for per-shape segment caching.
///
/// Controls how many compiled segments are cached per pipeline step and
/// which eviction policy is used when the cache is full.
///
/// # Example
///
/// ```rust,ignore
/// use nn_metal::segment_cache::{SegmentCacheConfig, EvictionPolicy};
///
/// // Cache up to 8 compiled segments per step with LRU eviction.
/// let config = SegmentCacheConfig {
///     max_segments_per_step: 8,
///     eviction: EvictionPolicy::Lru,
///     ..SegmentCacheConfig::default()
/// };
/// let kokoro = CompiledKokoro::new(model)?
///     .with_segment_cache_config(config);
/// ```
#[derive(Debug, Clone)]
pub struct SegmentCacheConfig {
    /// Maximum compiled segments cached per pipeline step.
    /// Each unique input shape gets its own cached segment.
    /// Default: 4.
    pub max_segments_per_step: usize,
    /// Eviction policy when capacity is reached.
    /// Default: [`EvictionPolicy::Lru`].
    pub eviction: EvictionPolicy,
    /// Maximum total bytes for planned GPU buffers per segment cache.
    /// Default: `None` (uses the compiled segment cache's built-in default,
    /// currently 512 MB — fits ~6 generator models at ~80 MB each).
    /// Set higher for chorus synthesis where multiple text lengths coexist.
    /// Part of #4187.
    pub byte_budget: Option<usize>,
    /// Optional shared segment store for cross-instance compiled segment sharing.
    ///
    /// When `Some`, multiple `CompiledKokoro` instances can share compiled
    /// segment definitions through this store, avoiding redundant compilations
    /// for the same segment kind + shape combination across chorus voices.
    ///
    /// Wrap the store in `Arc<Mutex<_>>` and pass the same clone to each
    /// instance's `SegmentCacheConfig`.
    ///
    /// Default: `None` (each instance maintains its own compiled segments).
    /// Part of #4104.
    pub shared_store: Option<std::sync::Arc<std::sync::Mutex<crate::shared_segment_store::SharedSegmentStore>>>,
}

impl Default for SegmentCacheConfig {
    fn default() -> Self {
        Self {
            max_segments_per_step: 4,
            eviction: EvictionPolicy::Lru,
            byte_budget: None,
            shared_store: None,
        }
    }
}

impl SegmentCacheConfig {
    /// Interactive TTS preset -- keep few shapes cached, minimal memory.
    ///
    /// Suitable for single-voice, low-latency scenarios where the user
    /// sends short utterances with varying lengths. Caches 2 shapes per
    /// step within a 256 MB byte budget.
    #[must_use]
    pub fn interactive() -> Self {
        Self {
            max_segments_per_step: 2,
            eviction: EvictionPolicy::Lru,
            byte_budget: Some(256 * 1024 * 1024),
            shared_store: None,
        }
    }

    /// Batch processing preset -- cache many shapes for throughput.
    ///
    /// Suitable for offline TTS rendering where many different text
    /// lengths are processed sequentially. Caches 8 shapes per step
    /// within a 1024 MB byte budget to minimise recompilation.
    #[must_use]
    pub fn batch() -> Self {
        Self {
            max_segments_per_step: 8,
            eviction: EvictionPolicy::Lru,
            byte_budget: Some(1024 * 1024 * 1024),
            shared_store: None,
        }
    }

    /// Chorus synthesis preset -- many voices share compiled segments.
    ///
    /// Suitable for multi-voice chorus where several voices with
    /// different text lengths run concurrently on shared compiled
    /// segments. Caches 6 shapes per step within a 768 MB byte budget.
    #[must_use]
    pub fn chorus() -> Self {
        Self {
            max_segments_per_step: 6,
            eviction: EvictionPolicy::Lru,
            byte_budget: Some(768 * 1024 * 1024),
            shared_store: None,
        }
    }

    /// Memory-constrained preset -- minimal cache footprint.
    ///
    /// Suitable for environments with tight memory budgets (e.g. mobile
    /// or embedded). Caches only 1 shape per step within a 128 MB byte
    /// budget. Expect frequent recompilation on shape changes.
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            max_segments_per_step: 1,
            eviction: EvictionPolicy::Lru,
            byte_budget: Some(128 * 1024 * 1024),
            shared_store: None,
        }
    }
}

/// Eviction policy for segment caches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used -- natural for TTS where recent lengths predict
    /// future lengths.
    Lru,
}

/// Generic LRU cache keyed by tensor shape (`Vec<usize>`).
///
/// Entries are ordered from most-recently-used (front) to least-recently-used
/// (back). On cache hit, the entry is promoted to the front. On insert at
/// capacity, the least-recently-used entry (back) is evicted.
///
/// This cache is type-generic over the cached value `V`, making it reusable
/// for compiled models, intermediate buffers, or any shape-dependent data.
///
/// # Example
///
/// ```rust,ignore
/// use nn_metal::segment_cache::ShapeKeyedCache;
///
/// let mut cache: ShapeKeyedCache<String> = ShapeKeyedCache::new(3);
/// cache.insert(vec![1, 128], "compiled_128".to_string());
/// cache.insert(vec![1, 256], "compiled_256".to_string());
///
/// assert_eq!(cache.get(&[1, 128]), Some(&"compiled_128".to_string()));
/// assert_eq!(cache.len(), 2);
/// ```
pub struct ShapeKeyedCache<V> {
    /// Ordered entries: front = MRU, back = LRU.
    entries: Vec<(Vec<usize>, V)>,
    /// Maximum number of entries before LRU eviction.
    max_entries: usize,
    /// Cumulative hit/miss/eviction counters.
    stats: SegmentCacheStats,
}

impl<V> ShapeKeyedCache<V> {
    /// Create an empty cache with the given maximum capacity.
    ///
    /// # Panics
    ///
    /// Panics if `max_entries` is 0. A zero-capacity cache cannot store
    /// anything and would silently discard all inserts.
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        assert!(max_entries > 0, "ShapeKeyedCache capacity must be > 0");
        Self {
            entries: Vec::with_capacity(max_entries),
            max_entries,
            stats: SegmentCacheStats::default(),
        }
    }

    /// Look up `shape`, promoting it to MRU position on hit.
    ///
    /// Returns a reference to the cached value, or `None` on miss.
    pub fn get(&mut self, shape: &[usize]) -> Option<&V> {
        let idx = self
            .entries
            .iter()
            .position(|(k, _)| k.as_slice() == shape);
        match idx {
            Some(i) => {
                self.stats.hits += 1;
                if i > 0 {
                    // Promote to MRU (front): remove from current position, push to front.
                    let entry = self.entries.remove(i);
                    self.entries.insert(0, entry);
                }
                self.entries.first().map(|(_, v)| v)
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Insert a value for `shape`, evicting the LRU entry if at capacity.
    ///
    /// If `shape` already exists, the old entry is replaced and promoted to MRU.
    pub fn insert(&mut self, shape: Vec<usize>, value: V) {
        // Remove existing entry with same shape (will be re-inserted at front).
        if let Some(idx) = self
            .entries
            .iter()
            .position(|(k, _)| k.as_slice() == shape.as_slice())
        {
            self.entries.remove(idx);
        }
        // Evict LRU (back) if at capacity.
        while self.entries.len() >= self.max_entries {
            self.entries.pop();
            self.stats.evictions += 1;
        }
        // Insert at front (MRU position).
        self.entries.insert(0, (shape, value));
    }

    /// Returns the number of cached entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the cache contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all entries from the cache.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Maximum number of entries this cache can hold.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.max_entries
    }

    /// Returns cumulative cache statistics (hits, misses, evictions).
    #[must_use]
    pub fn stats(&self) -> SegmentCacheStats {
        self.stats
    }

    /// Reset all statistics counters to zero.
    pub fn reset_stats(&mut self) {
        self.stats = SegmentCacheStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- SegmentCacheConfig tests --

    #[test]
    fn test_default_config_backwards_compatible() {
        let config = SegmentCacheConfig::default();
        assert_eq!(config.max_segments_per_step, 4);
        assert_eq!(config.eviction, EvictionPolicy::Lru);
    }

    #[test]
    fn test_config_custom_capacity() {
        let config = SegmentCacheConfig {
            max_segments_per_step: 16,
            ..SegmentCacheConfig::default()
        };
        assert_eq!(config.max_segments_per_step, 16);
    }

    #[test]
    fn test_config_single_entry_preserves_old_behaviour() {
        let config = SegmentCacheConfig {
            max_segments_per_step: 1,
            ..SegmentCacheConfig::default()
        };
        // max=1 means: one compiled segment per step, thrashes on shape change.
        // This was the pre-#2626 behaviour before default was raised to 4.
        assert_eq!(config.max_segments_per_step, 1);
    }

    // -- ShapeKeyedCache basic tests --

    #[test]
    fn test_new_cache_is_empty() {
        let cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(4);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.capacity(), 4);
    }

    #[test]
    #[should_panic(expected = "capacity must be > 0")]
    fn test_zero_capacity_panics() {
        let _cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(0);
    }

    #[test]
    fn test_insert_and_get() {
        let mut cache: ShapeKeyedCache<String> = ShapeKeyedCache::new(4);
        cache.insert(vec![1, 128], "model_128".to_string());
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&[1, 128]), Some(&"model_128".to_string()));
    }

    #[test]
    fn test_get_miss_returns_none() {
        let mut cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(4);
        cache.insert(vec![1, 128], 42);
        assert_eq!(cache.get(&[1, 256]), None);
    }

    #[test]
    fn test_multiple_shapes_coexist() {
        let mut cache: ShapeKeyedCache<&str> = ShapeKeyedCache::new(4);
        cache.insert(vec![1, 64], "a");
        cache.insert(vec![1, 128], "b");
        cache.insert(vec![1, 256], "c");
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(&[1, 64]), Some(&"a"));
        assert_eq!(cache.get(&[1, 128]), Some(&"b"));
        assert_eq!(cache.get(&[1, 256]), Some(&"c"));
    }

    // -- LRU eviction tests --

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
    fn test_insert_same_shape_replaces_value() {
        let mut cache: ShapeKeyedCache<&str> = ShapeKeyedCache::new(4);
        cache.insert(vec![1, 128], "v1");
        cache.insert(vec![1, 128], "v2");
        assert_eq!(cache.len(), 1, "duplicate shape should not increase count");
        assert_eq!(cache.get(&[1, 128]), Some(&"v2"), "value should be updated");
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
        cache.insert(vec![1, 20], 2);
        cache.insert(vec![1, 30], 3);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&[1, 30]), Some(&3));
        assert_eq!(cache.get(&[1, 10]), None);
        assert_eq!(cache.get(&[1, 20]), None);
    }

    #[test]
    fn test_clear() {
        let mut cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(4);
        cache.insert(vec![1], 10);
        cache.insert(vec![2], 20);
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_empty_shape_key() {
        let mut cache: ShapeKeyedCache<&str> = ShapeKeyedCache::new(4);
        cache.insert(vec![], "scalar");
        assert_eq!(cache.get(&[]), Some(&"scalar"));
    }

    #[test]
    fn test_high_rank_shape_key() {
        let mut cache: ShapeKeyedCache<i32> = ShapeKeyedCache::new(4);
        let shape = vec![2, 3, 4, 5, 6];
        cache.insert(shape.clone(), 42);
        assert_eq!(cache.get(&shape), Some(&42));
    }

    // -- byte_budget config tests (Part of #4187) --

    #[test]
    fn test_default_config_byte_budget_is_none() {
        let config = SegmentCacheConfig::default();
        assert_eq!(
            config.byte_budget, None,
            "default byte_budget should be None (use compiled cache default)"
        );
    }

    #[test]
    fn test_config_custom_byte_budget() {
        let config = SegmentCacheConfig {
            byte_budget: Some(256 * 1024 * 1024),
            ..SegmentCacheConfig::default()
        };
        assert_eq!(config.byte_budget, Some(256 * 1024 * 1024));
        // Other fields remain at defaults.
        assert_eq!(config.max_segments_per_step, 4);
        assert_eq!(config.eviction, EvictionPolicy::Lru);
    }

    // -- Production preset tests --

    #[test]
    fn test_preset_interactive() {
        let config = SegmentCacheConfig::interactive();
        assert_eq!(config.max_segments_per_step, 2);
        assert_eq!(config.eviction, EvictionPolicy::Lru);
        assert_eq!(config.byte_budget, Some(256 * 1024 * 1024));
    }

    #[test]
    fn test_preset_batch() {
        let config = SegmentCacheConfig::batch();
        assert_eq!(config.max_segments_per_step, 8);
        assert_eq!(config.eviction, EvictionPolicy::Lru);
        assert_eq!(config.byte_budget, Some(1024 * 1024 * 1024));
    }

    #[test]
    fn test_preset_chorus() {
        let config = SegmentCacheConfig::chorus();
        assert_eq!(config.max_segments_per_step, 6);
        assert_eq!(config.eviction, EvictionPolicy::Lru);
        assert_eq!(config.byte_budget, Some(768 * 1024 * 1024));
    }

    #[test]
    fn test_preset_minimal() {
        let config = SegmentCacheConfig::minimal();
        assert_eq!(config.max_segments_per_step, 1);
        assert_eq!(config.eviction, EvictionPolicy::Lru);
        assert_eq!(config.byte_budget, Some(128 * 1024 * 1024));
    }

    #[test]
    fn test_presets_have_distinct_budgets() {
        let presets = [
            SegmentCacheConfig::minimal(),
            SegmentCacheConfig::interactive(),
            SegmentCacheConfig::chorus(),
            SegmentCacheConfig::batch(),
        ];
        // Budgets should be strictly increasing in this order.
        for pair in presets.windows(2) {
            assert!(
                pair[0].byte_budget.unwrap() < pair[1].byte_budget.unwrap(),
                "presets should have strictly increasing byte budgets: {:?} vs {:?}",
                pair[0].byte_budget,
                pair[1].byte_budget,
            );
        }
    }

    #[test]
    fn test_presets_have_distinct_entries() {
        let presets = [
            SegmentCacheConfig::minimal(),
            SegmentCacheConfig::interactive(),
            SegmentCacheConfig::chorus(),
            SegmentCacheConfig::batch(),
        ];
        // Entry counts should be strictly increasing in this order.
        for pair in presets.windows(2) {
            assert!(
                pair[0].max_segments_per_step < pair[1].max_segments_per_step,
                "presets should have strictly increasing entry counts: {} vs {}",
                pair[0].max_segments_per_step,
                pair[1].max_segments_per_step,
            );
        }
    }
}

#[cfg(test)]
#[path = "segment_cache_tests.rs"]
mod segment_cache_tests;
