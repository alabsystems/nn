// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Two-tier pipeline cache for Metal compute kernels.
//!
//! [`PipelineCache`] deduplicates MSL compilation with a two-tier design:
//!
//! - **L1 (thread-local):** Per-thread `RefCell` cache with LRU eviction.
//!   Zero synchronization overhead on the hot path.
//! - **L2 (shared):** Process-global `RwLock`-backed cache. When a pipeline
//!   is compiled on one thread, other threads can reuse it without
//!   recompiling (one `RwLock::read()` on L1 miss).
//!
//! Each thread still creates its own `PipelineCache` (L1), but all instances
//! share the same L2 backing store via [`shared_cache()`].

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{OnceLock, RwLock};

use crate::context::MetalContext;
use crate::error::MetalError;
use crate::kernel_source::KernelSource;
use crate::pipeline::ComputePipeline;

// ---------------------------------------------------------------------------
// Shared (cross-thread) pipeline cache — L2
// ---------------------------------------------------------------------------

/// Maximum number of pipelines in the shared cross-thread cache.
const SHARED_MAX_ENTRIES: usize = 512;

/// Process-global shared pipeline cache.
///
/// Compiled Metal pipelines are expensive to create (GPU shader compilation)
/// but cheap to clone (`ComputePipelineState` is `Send + Sync` — an Obj-C
/// refcounted pointer). This cache stores compiled pipelines behind a
/// [`RwLock`] so that when thread B compiles a pipeline, thread A can reuse
/// it on its next cache miss without recompiling.
///
/// Access pattern:
/// - Hot path (L1 hit in thread-local [`PipelineCache`]): zero synchronization.
/// - Warm path (L1 miss, L2 hit here): one `RwLock::read()` acquisition.
/// - Cold path (L1 + L2 miss): compile, then `RwLock::write()` to insert.
#[derive(Debug)]
struct SharedPipelineCache {
    pipelines: RwLock<HashMap<u64, ComputePipeline>>,
}

impl SharedPipelineCache {
    fn new() -> Self {
        Self {
            pipelines: RwLock::new(HashMap::new()),
        }
    }

    /// Look up a compiled pipeline by its hash key (read lock).
    fn get(&self, key: u64) -> Option<ComputePipeline> {
        let map = self.pipelines.read().ok()?;
        map.get(&key).cloned()
    }

    /// Insert a compiled pipeline (write lock). If the cache exceeds
    /// `SHARED_MAX_ENTRIES`, evict a random entry (HashMap iteration
    /// order) to bound memory. Eviction is rare and O(1).
    fn insert(&self, key: u64, pipeline: ComputePipeline) {
        let Ok(mut map) = self.pipelines.write() else {
            return; // Poisoned lock — skip insertion, not fatal.
        };
        if map.len() >= SHARED_MAX_ENTRIES {
            // Evict one arbitrary entry to stay within bounds.
            if let Some(&evict_key) = map.keys().next() {
                map.remove(&evict_key);
            }
        }
        map.insert(key, pipeline);
    }

    /// Number of entries (for testing/diagnostics).
    fn len(&self) -> usize {
        self.pipelines.read().map_or(0, |m| m.len())
    }
}

/// Global shared pipeline cache singleton.
fn shared_cache() -> &'static SharedPipelineCache {
    static INSTANCE: OnceLock<SharedPipelineCache> = OnceLock::new();
    INSTANCE.get_or_init(SharedPipelineCache::new)
}

// ---------------------------------------------------------------------------
// Pre-compiled metallib pipeline store
// ---------------------------------------------------------------------------

/// Process-global store for pipelines loaded from a pre-compiled `.metallib`.
///
/// Keyed by entry point name (e.g., `"fused_adain_snake_f32"`). When
/// `get_or_compile()` misses both L1 and L2, it checks this store before
/// falling through to runtime MSL compilation.
///
/// Populated by [`load_precompiled_metallib`] at application startup.
struct PrecompiledStore {
    pipelines: HashMap<String, ComputePipeline>,
}

impl PrecompiledStore {
    fn get(&self, entry_point: &str) -> Option<ComputePipeline> {
        self.pipelines.get(entry_point).cloned()
    }

    fn len(&self) -> usize {
        self.pipelines.len()
    }
}

/// Global precompiled pipeline store singleton.
///
/// `None` if no metallib was loaded; `Some(store)` after successful loading.
static PRECOMPILED: OnceLock<Option<PrecompiledStore>> = OnceLock::new();

fn precompiled_store() -> Option<&'static PrecompiledStore> {
    PRECOMPILED.get().and_then(Option::as_ref)
}

/// Load a pre-compiled `.metallib` and populate the precompiled pipeline store.
///
/// Call this once at startup (typically from `MetalBackend::init`). The
/// function parses the provided metallib bytes, creates `ComputePipeline`
/// objects for all known kernel entry points, and stores them in the
/// global singleton. The caller decides where the bytes come from; the
/// default is the compile-time embedded metallib
/// (`metallib_loader::embedded_metallib`).
///
/// Subsequent calls are no-ops (the store is initialized exactly once).
///
/// # Arguments
///
/// * `context` — The Metal context to create pipelines from.
/// * `metallib_bytes` — Raw bytes of the `.metallib` file.
/// * `entry_points` — List of kernel function names to extract from the library.
///
/// # Errors
///
/// Returns `MetalError::MetallibLoad` if the metallib bytes cannot be
/// parsed into a Metal library — invalid or corrupted data is a hard
/// error, never a silent fallthrough to runtime MSL compilation.
///
/// Individual entry points missing from an otherwise valid metallib are
/// logged and skipped; those kernels compile at runtime from embedded MSL
/// sources (an integrity non-issue: the sources are string constants in
/// the binary).
#[cfg(target_os = "macos")]
pub(crate) fn load_precompiled_metallib(
    context: &MetalContext,
    metallib_bytes: &[u8],
    entry_points: &[&str],
) -> Result<usize, MetalError> {
    let mut load_error: Option<MetalError> = None;
    PRECOMPILED.get_or_init(|| {
        let library = match context.device().new_library_with_data(metallib_bytes) {
            Ok(lib) => lib,
            Err(e) => {
                load_error = Some(MetalError::MetallibLoad(e));
                return None;
            }
        };

        let mut pipelines = HashMap::with_capacity(entry_points.len());
        for &name in entry_points {
            let function = match library.get_function(name, None) {
                Ok(f) => f,
                Err(_) => {
                    eprintln!("[nn-metal] metallib missing entry point: {name}");
                    continue;
                }
            };

            let pipeline_state = match context
                .device()
                .new_compute_pipeline_state_with_function(&function)
            {
                Ok(ps) => ps,
                Err(e) => {
                    eprintln!("[nn-metal] metallib pipeline create failed for {name}: {e}");
                    continue;
                }
            };

            pipelines.insert(
                name.to_owned(),
                ComputePipeline::from_raw(pipeline_state, name, false),
            );
        }

        Some(PrecompiledStore { pipelines })
    });

    if let Some(e) = load_error {
        return Err(e);
    }
    Ok(precompiled_pipeline_count())
}

/// Non-macOS stub.
#[cfg(not(target_os = "macos"))]
pub(crate) fn load_precompiled_metallib(
    _context: &MetalContext,
    _metallib_bytes: &[u8],
    _entry_points: &[&str],
) -> Result<usize, MetalError> {
    PRECOMPILED.get_or_init(|| None);
    Ok(0)
}

/// Number of pre-compiled pipelines loaded, or 0 if none.
#[must_use]
pub fn precompiled_pipeline_count() -> usize {
    precompiled_store().map_or(0, PrecompiledStore::len)
}

/// Default maximum number of cached pipelines before LRU eviction.
const DEFAULT_MAX_ENTRIES: usize = 256;

/// Compile-once cache for Metal pipelines with LRU eviction.
///
/// Internally keyed on `u64` hashes of [`KernelSource`] to avoid cloning
/// full MSL source strings on every cache hit.
///
/// LRU tracking uses a generation counter: each access increments a global
/// counter and stamps the entry. Promote is O(1) (HashMap lookup + counter
/// increment). Eviction scans for the minimum generation — O(n) but only
/// triggered when the cache is full and a new entry is needed.
///
/// When the cache exceeds `max_entries`, the least-recently-used entry is
/// evicted. "Recently used" tracks both insertions and lookups.
#[derive(Debug)]
#[non_exhaustive]
pub struct PipelineCache {
    context: MetalContext,
    /// Maps `u64` hash of `KernelSource` → `(KernelSource, ComputePipeline)`.
    /// The `KernelSource` is retained for eviction and debugging.
    pipelines: RefCell<HashMap<u64, (KernelSource, ComputePipeline)>>,
    /// Maps `u64` hash key → last-access generation for LRU eviction.
    /// The entry with the lowest generation is the least-recently-used.
    access_gen: RefCell<HashMap<u64, u64>>,
    /// Monotonically increasing generation counter for LRU ordering.
    gen_counter: Cell<u64>,
    max_entries: usize,
}

impl PipelineCache {
    /// Create a new empty pipeline cache backed by the given Metal context.
    ///
    /// Uses the default capacity of 256 entries.
    #[must_use]
    pub fn new(context: MetalContext) -> Self {
        Self {
            context,
            pipelines: RefCell::new(HashMap::new()),
            access_gen: RefCell::new(HashMap::new()),
            gen_counter: Cell::new(0),
            max_entries: DEFAULT_MAX_ENTRIES,
        }
    }

    /// Create a pipeline cache using the global Metal context.
    ///
    /// Requires [`MetalBackend::init`] to have been called first. This avoids
    /// the boilerplate of manually cloning the context from the backend.
    ///
    /// # Errors
    ///
    /// Returns `MetalError::UninitializedBackend` if the global context is not initialized.
    pub fn new_global() -> Result<Self, MetalError> {
        let ctx = crate::metal_backend::global_metal_context()?;
        Ok(Self::new(ctx.clone()))
    }

    /// Create a pipeline cache with a custom maximum capacity.
    #[must_use]
    pub fn with_capacity(context: MetalContext, max_entries: usize) -> Self {
        Self {
            context,
            pipelines: RefCell::new(HashMap::new()),
            access_gen: RefCell::new(HashMap::new()),
            gen_counter: Cell::new(0),
            max_entries,
        }
    }

    /// Reference to the underlying Metal context.
    #[must_use]
    pub fn context(&self) -> &MetalContext {
        &self.context
    }

    /// Maximum number of cached pipelines before eviction.
    #[must_use]
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// Return a cached pipeline or compile one from `source`.
    ///
    /// Uses a three-tier lookup:
    /// - **L1 (thread-local):** Zero-synchronization `RefCell` lookup. Hot path.
    /// - **L2 (shared):** `RwLock`-backed global cache. Checked on L1 miss.
    /// - **Precompiled:** Entry-point lookup in pre-loaded `.metallib` (#2467).
    ///   Checked on L1+L2 miss before runtime MSL compilation.
    /// - **Compile:** GPU shader compilation as fallback. Result stored
    ///   in both L1 and L2.
    ///
    /// On cache hit, the entry is promoted to most-recently-used (O(1)).
    /// On cache miss, the pipeline is compiled and inserted; if the cache
    /// is at capacity, the least-recently-used entry is evicted first.
    #[must_use = "returns a Result that may contain an error"]
    pub fn get_or_compile(&self, source: &KernelSource) -> Result<ComputePipeline, MetalError> {
        let key = Self::hash_key(source);

        // L1: thread-local cache — zero synchronization.
        // Validate KernelSource on hit to detect hash collisions (#2211).
        if let Some((stored_source, pipeline)) = self.pipelines.borrow().get(&key) {
            if stored_source == source {
                self.stamp(key);
                return Ok(pipeline.clone());
            }
            // Hash collision: different KernelSource mapped to the same u64.
            // Fall through to L2 / compile and replace.
        }

        // L2: shared cross-thread cache — RwLock read.
        let shared = shared_cache();
        if let Some(pipeline) = shared.get(key) {
            // Promote into L1 so subsequent lookups on this thread are free.
            self.insert_l1(key, source, &pipeline);
            return Ok(pipeline);
        }

        // Precompiled metallib lookup by entry point name (#2467).
        // Faster than runtime MSL compilation — the pipeline was compiled
        // at build time via `xcrun metal` + `xcrun metallib`.
        // Skip for sources with function constants — precompiled pipelines
        // are unspecialized and would have wrong constant values (#3449).
        if source.function_constants().is_empty() {
            if let Some(store) = precompiled_store() {
                if let Some(pipeline) = store.get(source.entry_point()) {
                    // Promote into L1 and L2 for future lookups.
                    self.insert_l1(key, source, &pipeline);
                    shared.insert(key, pipeline.clone());
                    return Ok(pipeline);
                }
            }
        }

        // Cold path: compile the pipeline from MSL source.
        let compiled = self.context.compile_pipeline(source)?;

        // Insert into L1 (thread-local) and L2 (shared).
        self.insert_l1(key, source, &compiled);
        shared.insert(key, compiled.clone());
        Ok(compiled)
    }

    /// Like [`get_or_compile`], but compiles with ICB support enabled.
    ///
    /// ICB-compatible pipelines use a separate key namespace (XOR with a
    /// constant) so they don't collide with regular pipelines in the cache.
    /// Part of #3259 (D3).
    #[must_use = "returns a Result that may contain an error"]
    pub fn get_or_compile_icb(&self, source: &KernelSource) -> Result<ComputePipeline, MetalError> {
        const ICB_KEY_XOR: u64 = 0x1CB0_0000_0000_1CB0;
        let key = Self::hash_key(source) ^ ICB_KEY_XOR;
        if let Some((stored_source, pipeline)) = self.pipelines.borrow().get(&key) {
            if stored_source == source {
                self.stamp(key);
                return Ok(pipeline.clone());
            }
        }
        let shared = shared_cache();
        if let Some(pipeline) = shared.get(key) {
            self.insert_l1(key, source, &pipeline);
            return Ok(pipeline);
        }
        let compiled = self.context.compile_pipeline_icb(source)?;
        self.insert_l1(key, source, &compiled);
        shared.insert(key, compiled.clone());
        Ok(compiled)
    }

    /// Number of compiled pipelines in the thread-local L1 cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pipelines.borrow().len()
    }

    /// Whether L1 is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pipelines.borrow().is_empty()
    }

    /// Size of shared L2 cache. Acquires a read lock.
    #[must_use]
    pub fn shared_cache_len() -> usize {
        shared_cache().len()
    }

    /// Compute a stable `u64` hash of a [`KernelSource`].
    fn hash_key(source: &KernelSource) -> u64 {
        let mut hasher = std::hash::DefaultHasher::new();
        source.hash(&mut hasher);
        hasher.finish()
    }

    /// Insert a compiled pipeline into the thread-local L1 cache.
    fn insert_l1(&self, key: u64, source: &KernelSource, pipeline: &ComputePipeline) {
        if self.pipelines.borrow().len() >= self.max_entries {
            self.evict_lru();
        }
        self.pipelines
            .borrow_mut()
            .insert(key, (source.clone(), pipeline.clone()));
        self.stamp(key);
    }

    /// Stamp `key` with the next generation counter — O(1).
    fn stamp(&self, key: u64) {
        let next = self.gen_counter.get() + 1;
        self.gen_counter.set(next);
        self.access_gen.borrow_mut().insert(key, next);
    }

    /// Remove the least-recently-used entry from the cache.
    ///
    /// Scans `access_gen` for the entry with the lowest generation — O(n)
    /// but only called when the cache is full and a new entry is needed.
    fn evict_lru(&self) {
        let gens = self.access_gen.borrow();
        if let Some((&oldest_key, _)) = gens.iter().min_by_key(|(_, &g)| g) {
            drop(gens);
            self.access_gen.borrow_mut().remove(&oldest_key);
            self.pipelines.borrow_mut().remove(&oldest_key);
        }
    }

    /// Insert a pipeline into L1 under a forced hash key (test-only).
    ///
    /// Used by collision detection tests to simulate two different
    /// `KernelSource`s mapping to the same `u64` hash (#2211).
    #[cfg(test)]
    fn insert_with_forced_key(&self, key: u64, source: &KernelSource, pipeline: &ComputePipeline) {
        self.pipelines
            .borrow_mut()
            .insert(key, (source.clone(), pipeline.clone()));
        self.stamp(key);
    }
}

#[cfg(test)]
#[path = "cache_tests.rs"]
mod tests;
