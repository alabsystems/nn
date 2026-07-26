// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Generic cross-instance compiled segment store with LRU eviction.
//!
//! [`SharedSegmentStore`] holds `Arc<CompiledModelDef>` entries keyed by
//! [`SegmentKey`] (segment kind + input shape). Multiple `CompiledKokoro`
//! instances (e.g., chorus voices) can share a single store to avoid
//! redundant compilations of the same segment for the same shape.
//!
//! Unlike the Kokoro-specific [`SharedSegmentCache`](super::compiled_kokoro::SharedSegmentCache),
//! this store is model-agnostic: any compiled segment keyed by kind + shape
//! can be stored and retrieved. It is designed to be wrapped in
//! `Arc<Mutex<SharedSegmentStore>>` and shared across instances.
//!
//! # Eviction Policy
//!
//! LRU eviction triggers when either:
//! - The entry count exceeds `max_entries`
//! - Inserting a new entry would push `total_bytes` over `byte_budget`
//!
//! A single entry exceeding the budget is still stored (alone), matching
//! the behavior of [`SegmentCache`](super::compiled_kokoro_segment_cache::SegmentCache).
//!
//! # Thread Safety
//!
//! The store itself is `Send + Sync` (all fields are owned or `Arc`).
//! Callers are expected to wrap it in `Arc<Mutex<_>>` for concurrent access.
//!
//! Part of #4104.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::compiled_model::CompiledModelDef;
use crate::segment_cache::SegmentCacheStats;

/// Default maximum entries before LRU eviction.
const DEFAULT_MAX_ENTRIES: usize = 64;

/// Default byte budget (768 MB) — enough for chorus workloads with
/// multiple segment kinds and shape variants.
const DEFAULT_BYTE_BUDGET: usize = 768 * 1024 * 1024;

/// Composite key identifying a compiled segment: kind (pipeline stage) +
/// input shape dimensions.
///
/// Two entries with the same `segment_kind` but different `input_shape`
/// represent the same pipeline stage compiled for different tensor shapes
/// (e.g., generator compiled for 48000 samples vs 96000 samples).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SegmentKey {
    /// Pipeline segment identifier (e.g., 0=plbert, 4=generator).
    pub segment_kind: u8,
    /// Input tensor shape dimensions that parameterize compilation.
    pub input_shape: Vec<usize>,
}

impl SegmentKey {
    /// Create a new segment key.
    #[must_use]
    pub fn new(segment_kind: u8, input_shape: Vec<usize>) -> Self {
        Self {
            segment_kind,
            input_shape,
        }
    }
}

/// A shared compiled segment: immutable model definition wrapped in `Arc`.
///
/// The `Arc<CompiledModelDef>` is the established shareable unit in nn-metal.
/// It contains compiled steps, GPU weight buffers, pipeline states, and buffer
/// plans. Multiple `CompiledModel` execution instances can be created from a
/// single `Arc<CompiledModelDef>` via `CompiledModel::from_shared()`.
#[derive(Debug, Clone)]
pub struct SharedCompiledSegment {
    /// Immutable compiled model definition (steps, weights, pipelines).
    pub(crate) def: Arc<CompiledModelDef>,
    /// Tracked byte size for budget accounting.
    /// Uses `buffer_plan.total_bytes` from the definition.
    pub(crate) tracked_bytes: usize,
}

impl SharedCompiledSegment {
    /// Create a shared segment from a compiled model definition.
    #[must_use]
    pub(crate) fn new(def: Arc<CompiledModelDef>) -> Self {
        let tracked_bytes = def.buffer_plan.total_bytes;
        Self { def, tracked_bytes }
    }

    /// The compiled model definition.
    #[must_use]
    pub(crate) fn def(&self) -> &Arc<CompiledModelDef> {
        &self.def
    }

    /// Byte size tracked for budget accounting.
    #[must_use]
    pub fn tracked_bytes(&self) -> usize {
        self.tracked_bytes
    }
}

/// Statistics for the shared segment store.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SharedSegmentStats {
    /// Number of `get()` calls that found the requested key.
    pub hits: usize,
    /// Number of `get()` calls that did not find the requested key.
    pub misses: usize,
    /// Number of entries evicted (LRU eviction or byte-budget eviction).
    pub evictions: usize,
    /// Current total tracked bytes across all entries.
    pub total_bytes: usize,
    /// Current number of entries in the store.
    pub entry_count: usize,
}

impl SharedSegmentStats {
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
}

/// Generic cross-instance store for compiled segment definitions.
///
/// Stores `Arc<CompiledModelDef>` entries keyed by [`SegmentKey`] with LRU
/// eviction. Designed to be wrapped in `Arc<Mutex<SharedSegmentStore>>` for
/// concurrent access by multiple `CompiledKokoro` instances.
///
/// # Example
///
/// ```rust,ignore
/// use std::sync::{Arc, Mutex};
/// use nn_metal::shared_segment_store::{SharedSegmentStore, SegmentKey, SharedCompiledSegment};
///
/// let store = Arc::new(Mutex::new(SharedSegmentStore::new(768 * 1024 * 1024)));
///
/// // Insert a compiled segment
/// let key = SegmentKey::new(4, vec![1, 48000]);
/// let segment = SharedCompiledSegment::new(compiled_model.share_def());
/// store.lock().unwrap().insert(key.clone(), segment);
///
/// // Retrieve from another instance
/// if let Some(shared) = store.lock().unwrap().get(&key) {
///     let model = CompiledModel::from_shared(Arc::clone(shared.def()));
/// }
/// ```
pub struct SharedSegmentStore {
    /// LRU-ordered entries: front = MRU, back = LRU.
    entries: VecDeque<(SegmentKey, SharedCompiledSegment)>,
    /// Current total tracked bytes across all entries.
    total_bytes: usize,
    /// Maximum total bytes before LRU eviction.
    byte_budget: usize,
    /// Maximum number of entries before LRU eviction.
    max_entries: usize,
    /// Cumulative hit/miss/eviction counters.
    hits: usize,
    misses: usize,
    evictions: usize,
}

impl SharedSegmentStore {
    /// Create a new store with the given byte budget and default max entries.
    ///
    /// # Panics
    ///
    /// Panics if `byte_budget` is 0.
    #[must_use]
    pub fn new(byte_budget: usize) -> Self {
        assert!(byte_budget > 0, "SharedSegmentStore byte_budget must be > 0");
        Self {
            entries: VecDeque::with_capacity(DEFAULT_MAX_ENTRIES.min(64)),
            total_bytes: 0,
            byte_budget,
            max_entries: DEFAULT_MAX_ENTRIES,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Create a new store with custom byte budget and max entry count.
    ///
    /// # Panics
    ///
    /// Panics if `byte_budget` is 0 or `max_entries` is 0.
    #[must_use]
    pub fn with_capacity(byte_budget: usize, max_entries: usize) -> Self {
        assert!(byte_budget > 0, "SharedSegmentStore byte_budget must be > 0");
        assert!(
            max_entries > 0,
            "SharedSegmentStore max_entries must be > 0"
        );
        Self {
            entries: VecDeque::with_capacity(max_entries.min(128)),
            total_bytes: 0,
            byte_budget,
            max_entries,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Look up a key, promoting it to MRU position on hit.
    ///
    /// Returns a reference to the cached segment, or `None` on miss.
    pub(crate) fn get(&mut self, key: &SegmentKey) -> Option<&SharedCompiledSegment> {
        let idx = self.entries.iter().position(|(k, _)| k == key);
        match idx {
            Some(i) => {
                self.hits += 1;
                if i > 0 {
                    if let Some(entry) = self.entries.remove(i) {
                        self.entries.push_front(entry);
                    }
                }
                self.entries.front().map(|(_, v)| v)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Insert a compiled segment, evicting LRU entries if at capacity or
    /// over the byte budget.
    ///
    /// If the key already exists, the old entry is replaced and promoted to MRU.
    /// Returns a clone of the `Arc<CompiledModelDef>` for the inserted segment.
    pub(crate) fn insert(
        &mut self,
        key: SegmentKey,
        segment: SharedCompiledSegment,
    ) -> Arc<CompiledModelDef> {
        // Remove existing entry with same key.
        if let Some(idx) = self.entries.iter().position(|(k, _)| *k == key) {
            if let Some((_, evicted)) = self.entries.remove(idx) {
                self.total_bytes -= evicted.tracked_bytes;
            }
        }

        let new_bytes = segment.tracked_bytes;

        // Evict LRU entries until both count and byte budget are satisfied.
        // The !is_empty() guard ensures a single oversized entry is still stored.
        while self.entries.len() >= self.max_entries
            || (self.total_bytes + new_bytes > self.byte_budget && !self.entries.is_empty())
        {
            if let Some((_, evicted)) = self.entries.pop_back() {
                self.total_bytes -= evicted.tracked_bytes;
                self.evictions += 1;
            }
        }

        let def = Arc::clone(&segment.def);
        self.total_bytes += new_bytes;
        self.entries.push_front((key, segment));
        def
    }

    /// Evict the least-recently-used entry, if any.
    ///
    /// Returns `true` if an entry was evicted, `false` if the store was empty.
    pub fn evict_lru(&mut self) -> bool {
        if let Some((_, evicted)) = self.entries.pop_back() {
            self.total_bytes -= evicted.tracked_bytes;
            self.evictions += 1;
            true
        } else {
            false
        }
    }

    /// Returns the current store statistics.
    #[must_use]
    pub fn stats(&self) -> SharedSegmentStats {
        SharedSegmentStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            total_bytes: self.total_bytes,
            entry_count: self.entries.len(),
        }
    }

    /// Number of entries currently in the store.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the store contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Current total tracked bytes across all entries.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// The byte budget for this store.
    #[must_use]
    pub fn byte_budget(&self) -> usize {
        self.byte_budget
    }

    /// Maximum entry count for this store.
    #[must_use]
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// Remove all entries, resetting tracked bytes and stats.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.total_bytes = 0;
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
    }

    /// Check whether a key exists (does NOT promote to MRU).
    #[must_use]
    pub fn contains_key(&self, key: &SegmentKey) -> bool {
        self.entries.iter().any(|(k, _)| k == key)
    }

    /// Convert to a [`SegmentCacheStats`] for compatibility with the existing
    /// segment cache stats infrastructure.
    #[must_use]
    pub fn as_cache_stats(&self) -> SegmentCacheStats {
        SegmentCacheStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            total_bytes: self.total_bytes,
        }
    }
}

impl std::fmt::Debug for SharedSegmentStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedSegmentStore")
            .field("entries", &self.entries.len())
            .field("total_bytes", &self.total_bytes)
            .field("byte_budget", &self.byte_budget)
            .field("max_entries", &self.max_entries)
            .field("hits", &self.hits)
            .field("misses", &self.misses)
            .field("evictions", &self.evictions)
            .finish()
    }
}

// SharedSegmentStore is automatically Send+Sync because all its fields are:
// - VecDeque<(SegmentKey, SharedCompiledSegment)>: Send+Sync (SegmentKey is
//   owned Vec<usize> + u8; SharedCompiledSegment is Arc<CompiledModelDef> + usize)
// - usize: Send+Sync
// CompiledModelDef's Send+Sync is verified in compiled_model_def_tests.rs.

#[cfg(test)]
#[path = "shared_segment_store_tests.rs"]
mod tests;
