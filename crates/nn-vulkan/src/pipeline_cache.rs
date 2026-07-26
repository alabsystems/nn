// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Two-tier pipeline cache for Vulkan compute pipelines.
//!
//! Mirrors the Metal backend's [`PipelineCache`](nn-metal) design:
//!
//! - **L1 (thread-local):** Per-thread cache with LRU eviction.
//!   Zero synchronization overhead on the hot path.
//! - **L2 (shared):** Process-global `RwLock`-backed cache. When a pipeline
//!   is compiled on one thread, other threads can reuse it without
//!   recompiling (one `RwLock::read()` on L1 miss).
//!
//! Each thread creates its own `PipelineCache` (L1), but all instances
//! share the same L2 backing store via [`shared_cache()`].

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{OnceLock, RwLock};

use crate::dispatch::{ComputePipeline, DescriptorSetLayout, PipelineLayout};
use crate::error::VulkanError;

// ---------------------------------------------------------------------------
// Shared (cross-thread) pipeline cache -- L2
// ---------------------------------------------------------------------------

/// Maximum number of pipelines in the shared cross-thread cache.
const SHARED_MAX_ENTRIES: usize = 512;

/// Maximum number of pipelines in the thread-local L1 cache.
const LOCAL_MAX_ENTRIES: usize = 64;

/// Process-global shared pipeline cache.
///
/// Compiled Vulkan pipelines are expensive to create (SPIR-V shader
/// compilation) but cheap to share across threads when using the same
/// logical device. This cache stores compiled pipeline metadata behind a
/// [`RwLock`] so that thread B can reuse a pipeline compiled by thread A.
///
/// Access pattern:
/// - Hot path (L1 hit in thread-local [`PipelineCache`]): zero synchronization.
/// - Warm path (L1 miss, L2 hit here): one `RwLock::read()` acquisition.
/// - Cold path (L1 + L2 miss): compile, then `RwLock::write()` to insert.
#[derive(Debug)]
struct SharedPipelineCache {
    pipelines: RwLock<HashMap<u64, CachedPipeline>>,
}

impl SharedPipelineCache {
    fn new() -> Self {
        Self {
            pipelines: RwLock::new(HashMap::new()),
        }
    }

    /// Look up a cached pipeline by its hash key (read lock).
    fn get(&self, key: u64) -> Option<CachedPipeline> {
        let map = self.pipelines.read().ok()?;
        map.get(&key).cloned()
    }

    /// Insert a cached pipeline (write lock). If the cache exceeds
    /// `SHARED_MAX_ENTRIES`, evict a random entry (HashMap iteration
    /// order) to bound memory. Eviction is rare and O(1).
    fn insert(&self, key: u64, pipeline: CachedPipeline) {
        let Ok(mut map) = self.pipelines.write() else {
            return; // Poisoned lock -- skip insertion, not fatal.
        };
        if map.len() >= SHARED_MAX_ENTRIES {
            if let Some(&evict_key) = map.keys().next() {
                map.remove(&evict_key);
            }
        }
        map.insert(key, pipeline);
    }

    /// Number of entries in the shared cache (for diagnostics).
    fn len(&self) -> usize {
        self.pipelines.read().map_or(0, |m| m.len())
    }
}

/// Global singleton for the shared pipeline cache.
fn shared_cache() -> &'static SharedPipelineCache {
    static CACHE: OnceLock<SharedPipelineCache> = OnceLock::new();
    CACHE.get_or_init(SharedPipelineCache::new)
}

// ---------------------------------------------------------------------------
// Cached pipeline entry
// ---------------------------------------------------------------------------

/// A cached pipeline entry containing the compiled pipeline and its metadata.
#[derive(Debug, Clone)]
pub struct CachedPipeline {
    /// The compiled compute pipeline.
    pub pipeline: ComputePipeline,
    /// GLSL source that was compiled (for debugging / cache key verification).
    pub glsl_source_hash: u64,
}

// ---------------------------------------------------------------------------
// Pipeline cache key
// ---------------------------------------------------------------------------

/// Compute a cache key from GLSL source and entry point.
///
/// Uses FNV-1a for fast, deterministic hashing. Not cryptographic --
/// collisions are acceptable (worst case: recompile).
#[must_use]
pub fn pipeline_cache_key(glsl_source: &str, entry_point: &str) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    glsl_source.hash(&mut hasher);
    entry_point.hash(&mut hasher);
    hasher.finish()
}

/// Compute a cache key from SPIR-V binary words and entry point.
#[must_use]
pub fn spirv_cache_key(spirv_words: &[u32], entry_point: &str) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    spirv_words.hash(&mut hasher);
    entry_point.hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------
// Thread-local pipeline cache -- L1
// ---------------------------------------------------------------------------

/// Thread-local pipeline cache with L2 shared backing store.
///
/// # Usage
///
/// ```no_run
/// use nn_vulkan::pipeline_cache::PipelineCache;
///
/// let mut cache = PipelineCache::new();
///
/// // First lookup: L1 miss, L2 miss -> compile
/// let key = nn_vulkan::pipeline_cache::pipeline_cache_key("...", "main");
/// let hit = cache.get(key);
/// assert!(hit.is_none());
///
/// // After compilation, insert into cache
/// // cache.insert(key, pipeline);
///
/// // Subsequent lookup: L1 hit -> zero synchronization
/// // let hit = cache.get(key);
/// ```
pub struct PipelineCache {
    /// L1 thread-local cache.
    local: HashMap<u64, CachedPipeline>,
    /// L1 access order for LRU eviction (most recent at end).
    access_order: Vec<u64>,
    /// Cache statistics.
    stats: PipelineCacheStats,
}

/// Pipeline cache statistics for diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipelineCacheStats {
    /// Lookups that hit L1 (thread-local).
    pub l1_hits: usize,
    /// Lookups that missed L1 but hit L2 (shared).
    pub l2_hits: usize,
    /// Lookups that missed both L1 and L2.
    pub misses: usize,
    /// Total insert operations.
    pub inserts: usize,
    /// L1 evictions due to capacity limit.
    pub l1_evictions: usize,
}

impl PipelineCache {
    /// Create a new thread-local pipeline cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            local: HashMap::new(),
            access_order: Vec::new(),
            stats: PipelineCacheStats::default(),
        }
    }

    /// Look up a cached pipeline by key.
    ///
    /// Checks L1 (thread-local) first, then L2 (shared). On L2 hit,
    /// promotes the entry to L1 for subsequent fast access.
    pub fn get(&mut self, key: u64) -> Option<&CachedPipeline> {
        // L1 check.
        if self.local.contains_key(&key) {
            self.stats.l1_hits += 1;
            self.touch_access_order(key);
            return self.local.get(&key);
        }

        // L2 check.
        if let Some(entry) = shared_cache().get(key) {
            self.stats.l2_hits += 1;
            self.insert_local(key, entry);
            return self.local.get(&key);
        }

        self.stats.misses += 1;
        None
    }

    /// Insert a compiled pipeline into both L1 and L2 caches.
    pub fn insert(&mut self, key: u64, pipeline: ComputePipeline, glsl_source_hash: u64) {
        let entry = CachedPipeline {
            pipeline,
            glsl_source_hash,
        };
        self.stats.inserts += 1;
        self.insert_local(key, entry.clone());
        shared_cache().insert(key, entry);
    }

    /// Insert into L1 with LRU eviction.
    fn insert_local(&mut self, key: u64, entry: CachedPipeline) {
        if self.local.len() >= LOCAL_MAX_ENTRIES && !self.local.contains_key(&key) {
            // Evict LRU entry.
            if let Some(evict_key) = self.access_order.first().copied() {
                self.local.remove(&evict_key);
                self.access_order.retain(|&k| k != evict_key);
                self.stats.l1_evictions += 1;
            }
        }
        self.local.insert(key, entry);
        self.touch_access_order(key);
    }

    /// Move a key to the end of the access order (most recently used).
    fn touch_access_order(&mut self, key: u64) {
        self.access_order.retain(|&k| k != key);
        self.access_order.push(key);
    }

    /// Number of entries in the L1 (thread-local) cache.
    #[must_use]
    pub fn l1_len(&self) -> usize {
        self.local.len()
    }

    /// Number of entries in the L2 (shared) cache.
    #[must_use]
    pub fn l2_len(&self) -> usize {
        shared_cache().len()
    }

    /// Cache statistics snapshot.
    #[must_use]
    pub fn stats(&self) -> PipelineCacheStats {
        self.stats
    }
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Compile helper
// ---------------------------------------------------------------------------

/// Compile a GLSL source string to a `ComputePipeline` via SPIR-V, using
/// the pipeline cache for deduplication.
///
/// This is the primary entry point for kernel compilation. It:
/// 1. Computes the cache key from GLSL source + entry point.
/// 2. Checks the pipeline cache (L1 then L2).
/// 3. On miss, compiles GLSL to SPIR-V and creates a `ComputePipeline`.
/// 4. Inserts the result into both cache tiers.
///
/// # Arguments
///
/// * `cache` -- Mutable reference to the thread-local pipeline cache.
/// * `glsl_source` -- GLSL 450 compute shader source string.
/// * `entry_point` -- Shader entry point name (usually `"main"`).
/// * `spirv_words` -- Pre-compiled SPIR-V binary words. In production this
///   comes from `glslangValidator` or `shaderc`. For testing, pass a valid
///   SPIR-V header.
/// * `descriptor_layout` -- Descriptor set layout for buffer bindings.
/// * `push_constant_size` -- Push constant block size in bytes.
///
/// # Errors
///
/// Returns [`VulkanError`] if pipeline creation fails.
pub fn compile_or_cache(
    cache: &mut PipelineCache,
    glsl_source: &str,
    entry_point: &str,
    spirv_words: &[u32],
    descriptor_layout: &DescriptorSetLayout,
    push_constant_size: u32,
) -> Result<ComputePipeline, VulkanError> {
    let key = pipeline_cache_key(glsl_source, entry_point);

    // Cache hit path.
    if let Some(cached) = cache.get(key) {
        return Ok(cached.pipeline.clone());
    }

    // Cache miss: compile.
    let pl = PipelineLayout::new(descriptor_layout, push_constant_size)?;
    let pipeline = ComputePipeline::new(spirv_words, entry_point, &pl)?;

    let glsl_hash = {
        let mut h = std::hash::DefaultHasher::new();
        glsl_source.hash(&mut h);
        h.finish()
    };
    cache.insert(key, pipeline.clone(), glsl_hash);
    Ok(pipeline)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::{DescriptorBinding, DescriptorType};
    use crate::spirv_emit::{SPIRV_MAGIC, SPIRV_VERSION_1_5};

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

    #[test]
    fn test_cache_key_deterministic() {
        let k1 = pipeline_cache_key("void main() {}", "main");
        let k2 = pipeline_cache_key("void main() {}", "main");
        assert_eq!(k1, k2, "same source + entry should produce same key");
    }

    #[test]
    fn test_cache_key_different_source() {
        let k1 = pipeline_cache_key("void main() { float x = 1.0; }", "main");
        let k2 = pipeline_cache_key("void main() { float x = 2.0; }", "main");
        assert_ne!(k1, k2, "different source should produce different keys");
    }

    #[test]
    fn test_cache_key_different_entry_point() {
        let k1 = pipeline_cache_key("void main() {}", "main");
        let k2 = pipeline_cache_key("void main() {}", "alt_main");
        assert_ne!(
            k1, k2,
            "different entry points should produce different keys"
        );
    }

    #[test]
    fn test_spirv_cache_key_deterministic() {
        let words = make_test_spirv();
        let k1 = spirv_cache_key(&words, "main");
        let k2 = spirv_cache_key(&words, "main");
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_cache_l1_miss_then_hit() {
        let mut cache = PipelineCache::new();
        let key = pipeline_cache_key("test_shader", "main");

        // Miss.
        assert!(cache.get(key).is_none());
        assert_eq!(cache.stats().misses, 1);

        // Insert.
        let spirv = make_test_spirv();
        let ds_layout = make_test_ds_layout();
        let pl = PipelineLayout::new(&ds_layout, 4).expect("pl");
        let pipeline = ComputePipeline::new(&spirv, "main", &pl).expect("pipeline");
        cache.insert(key, pipeline, 12345);
        assert_eq!(cache.stats().inserts, 1);

        // Hit.
        assert!(cache.get(key).is_some());
        assert_eq!(cache.stats().l1_hits, 1);
    }

    #[test]
    fn test_cache_l1_eviction() {
        let mut cache = PipelineCache::new();
        let spirv = make_test_spirv();
        let ds_layout = make_test_ds_layout();
        let pl = PipelineLayout::new(&ds_layout, 4).expect("pl");

        // Fill L1 beyond capacity.
        for i in 0..LOCAL_MAX_ENTRIES + 5 {
            let key = pipeline_cache_key(&format!("shader_{i}"), "main");
            let pipeline = ComputePipeline::new(&spirv, "main", &pl).expect("pipeline");
            cache.insert(key, pipeline, i as u64);
        }

        // L1 should be at capacity.
        assert!(cache.l1_len() <= LOCAL_MAX_ENTRIES);
        assert!(cache.stats().l1_evictions > 0);
    }

    #[test]
    fn test_compile_or_cache_miss_then_hit() {
        let mut cache = PipelineCache::new();
        let glsl = "void main() { test; }";
        let spirv = make_test_spirv();
        let ds_layout = make_test_ds_layout();

        // First call: cache miss, compiles.
        let p1 = compile_or_cache(&mut cache, glsl, "main", &spirv, &ds_layout, 4)
            .expect("first compile");
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().inserts, 1);

        // Second call: cache hit.
        let p2 = compile_or_cache(&mut cache, glsl, "main", &spirv, &ds_layout, 4)
            .expect("cached compile");
        assert_eq!(cache.stats().l1_hits, 1);
        assert_eq!(p1.entry_point(), p2.entry_point());
    }

    #[test]
    fn test_stats_default() {
        let stats = PipelineCacheStats::default();
        assert_eq!(stats.l1_hits, 0);
        assert_eq!(stats.l2_hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.inserts, 0);
        assert_eq!(stats.l1_evictions, 0);
    }

    #[test]
    fn test_cache_default_constructor() {
        let cache = PipelineCache::default();
        assert_eq!(cache.l1_len(), 0);
    }
}
