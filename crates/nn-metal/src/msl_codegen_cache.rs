// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Thread-local LRU cache for MSL codegen output.
//!
//! Sits between [`KernelDefCache`](crate::kernel_def_cache) (Layer 1, caches
//! `TensorKernelDef` IR) and [`PipelineCache`](crate::PipelineCache) (Layer 3,
//! caches compiled Metal pipelines). This Layer 2 caches the output of
//! `build_dispatch_plan_full()` + `emit_tensor_msl_with_plan()`:
//!
//! ```text
//! Layer 1: KernelDefCache  →  TensorKernelDef (IR)
//! Layer 2: MslCodegenCache →  (plan, output_id, expanded, msl_string)  ← THIS
//! Layer 3: PipelineCache   →  ComputePipeline (compiled Metal)
//! ```
//!
//! For HTDemucs temporal inference (~130 dispatches per forward), this
//! eliminates ~30-150µs of repeated MSL string generation per dispatch.
//! The cache key hashes the kernel definition structure, `ScalarType`, and
//! `PrecisionContract` — deterministic for identical inputs.
//!
//! See #2032 for tracking.

use std::cell::RefCell;
use std::fmt::Write as FmtWrite;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use nn_dsl::ir::ScalarType;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::{DispatchStep, PrecisionContract, TensorNodeId};

/// Default maximum number of cached codegen entries before LRU eviction.
const DEFAULT_MAX_ENTRIES: usize = 256;

thread_local! {
    static CACHE: RefCell<MslCodegenCache> = RefCell::new(MslCodegenCache::new());
}

/// Cached output of `build_dispatch_plan_full()` + `emit_tensor_msl_with_plan()`.
///
/// Stored as `Arc<CodegenOutput>` in the cache to avoid cloning the MSL string
/// (~1-12KB) and dispatch plan on every cache hit. Callers borrow through the
/// Arc instead. See D1 in `designs/2026-03-12-msl-codegen-cache-elimination.md`.
pub(crate) struct CodegenOutput {
    pub(crate) plan: Vec<DispatchStep>,
    pub(crate) effective_output: TensorNodeId,
    pub(crate) expanded: TensorKernelDef,
    pub(crate) msl: String,
}

/// Lightweight key stored alongside each cache entry for collision detection.
///
/// On a hash hit, we validate these fields against the query to detect
/// u64 hash collisions. This avoids storing the full `TensorKernelDef`
/// (which contains the entire IR graph) while providing high-discrimination
/// validation. See #2202.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CodegenKey {
    kernel_name: String,
    node_count: usize,
    output_index: usize,
    dtype: ScalarType,
    precision_tier: nn_dsl::PrecisionTier,
    fast_math: bool,
}

impl CodegenKey {
    fn from_query(
        kernel: &TensorKernelDef,
        dtype: ScalarType,
        contract: PrecisionContract,
    ) -> Self {
        Self {
            kernel_name: kernel.name.clone(),
            node_count: kernel.nodes.len(),
            output_index: kernel.output.index(),
            dtype,
            precision_tier: contract.tier,
            fast_math: contract.fast_math,
        }
    }
}

/// Thread-local LRU cache for MSL codegen results.
///
/// Mirrors the [`KernelDefCache`](crate::kernel_def_cache) pattern:
/// `HashMap` keyed on `u64` hashes with generation-based LRU eviction.
/// Each entry stores a [`CodegenKey`] alongside the output for collision
/// detection on cache hits (#2202).
struct MslCodegenCache {
    entries: std::collections::HashMap<u64, (CodegenKey, Arc<CodegenOutput>)>,
    access_gen: std::collections::HashMap<u64, u64>,
    gen_counter: u64,
    max_entries: usize,
}

impl MslCodegenCache {
    fn new() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
            access_gen: std::collections::HashMap::new(),
            gen_counter: 0,
            max_entries: DEFAULT_MAX_ENTRIES,
        }
    }

    fn stamp(&mut self, key: u64) {
        self.gen_counter += 1;
        self.access_gen.insert(key, self.gen_counter);
    }

    fn evict_lru(&mut self) {
        if let Some((&oldest_key, _)) = self.access_gen.iter().min_by_key(|(_, &g)| g) {
            self.access_gen.remove(&oldest_key);
            self.entries.remove(&oldest_key);
        }
    }
}

/// Zero-alloc adapter that feeds `fmt::Write` output directly into a `Hasher`.
///
/// Replaces the previous `format!("{:?}", val).hash(&mut hasher)` pattern,
/// which allocated a temporary `String` per IR node on every hash computation.
/// This adapter hashes the Debug representation byte-by-byte without allocation.
struct HashWriter<'a, H: Hasher>(&'a mut H);

impl<H: Hasher> FmtWrite for HashWriter<'_, H> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.write(s.as_bytes());
        Ok(())
    }
}

/// Hash a `Debug`-formatted value directly into the hasher without allocation.
fn hash_debug<H: Hasher>(hasher: &mut H, val: &impl std::fmt::Debug) {
    let _ = write!(HashWriter(hasher), "{val:?}");
}

/// Compute a stable `u64` hash of a `TensorKernelDef` + codegen parameters.
///
/// The hash covers the kernel name, all node shapes and output IDs,
/// `ScalarType`, and `PrecisionTier` — everything that determines the
/// codegen output. Two kernels with identical structure, dtype, and
/// precision contract always produce the same MSL.
///
/// Uses zero-alloc `HashWriter` adapter for Debug representations instead
/// of `format!("{:?}")` which allocated a `String` per node. See D3 in
/// `designs/2026-03-12-msl-codegen-cache-elimination.md`.
fn codegen_hash(kernel: &TensorKernelDef, dtype: ScalarType, contract: PrecisionContract) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    kernel.name.hash(&mut hasher);
    kernel.nodes.len().hash(&mut hasher);
    for node in &kernel.nodes {
        node.id.index().hash(&mut hasher);
        node.shape.len().hash(&mut hasher);
        for &d in &node.shape {
            d.hash(&mut hasher);
        }
        // Hash the full op kind structure (discriminant + parameters) via
        // Debug representation, but without allocating a temporary String.
        hash_debug(&mut hasher, &node.kind);
    }
    kernel.output.index().hash(&mut hasher);
    // ScalarType does not derive Hash; use Debug discriminant.
    hash_debug(&mut hasher, &dtype);
    contract.tier.hash(&mut hasher);
    contract.fast_math.hash(&mut hasher);
    hasher.finish()
}

/// Look up cached codegen output, or generate and cache it.
///
/// On cache hit, returns an `Arc` reference to the cached output (O(1) hash
/// lookup + Arc clone). On miss, calls `generate` to produce the codegen
/// output, wraps it in `Arc`, caches it, and returns a shared reference.
///
/// Using `Arc` instead of cloning eliminates ~1-12KB of MSL string allocation
/// per cache hit, plus `Vec<DispatchStep>` and expanded kernel clones.
pub(crate) fn get_or_generate<F>(
    kernel: &TensorKernelDef,
    dtype: ScalarType,
    contract: PrecisionContract,
    generate: F,
) -> Result<Arc<CodegenOutput>, super::TensorDispatchError>
where
    F: FnOnce() -> Result<CodegenOutput, super::TensorDispatchError>,
{
    let hash_key = codegen_hash(kernel, dtype, contract);
    let query_key = CodegenKey::from_query(kernel, dtype, contract);

    CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();

        // Validate CodegenKey on hit to detect hash collisions (#2202).
        if let Some((stored_key, entry)) = cache.entries.get(&hash_key) {
            if stored_key == &query_key {
                let output = Arc::clone(entry);
                cache.stamp(hash_key);
                return Ok(output);
            }
            // Hash collision: different kernel/dtype/contract mapped to the
            // same u64. Fall through to regenerate and replace.
        }

        let output = Arc::new(generate()?);

        if cache.entries.len() >= cache.max_entries {
            cache.evict_lru();
        }

        cache
            .entries
            .insert(hash_key, (query_key, Arc::clone(&output)));
        cache.stamp(hash_key);
        Ok(output)
    })
}

/// Number of cached codegen entries (for testing/diagnostics).
#[cfg(test)]
pub(crate) fn cache_len() -> usize {
    CACHE.with(|cell| cell.borrow().entries.len())
}

/// Clear the thread-local cache (for testing).
#[cfg(test)]
pub(crate) fn clear_cache() {
    CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        cache.entries.clear();
        cache.access_gen.clear();
        cache.gen_counter = 0;
    });
}

/// Compute the hash key for a kernel/dtype/contract query (test-only).
///
/// Exposed for collision detection tests that need to force two different
/// queries to the same hash key (#2202).
#[cfg(test)]
pub(crate) fn test_codegen_hash(
    kernel: &TensorKernelDef,
    dtype: ScalarType,
    contract: PrecisionContract,
) -> u64 {
    codegen_hash(kernel, dtype, contract)
}

/// Insert an entry with a forced hash key (test-only).
///
/// Used by collision detection tests to simulate two different
/// kernel/dtype/contract queries mapping to the same u64 hash (#2202).
#[cfg(test)]
pub(crate) fn insert_with_forced_key(
    key: u64,
    kernel: &TensorKernelDef,
    dtype: ScalarType,
    contract: PrecisionContract,
    output: Arc<CodegenOutput>,
) {
    let codegen_key = CodegenKey::from_query(kernel, dtype, contract);
    CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        cache.entries.insert(key, (codegen_key, output));
        cache.stamp(key);
    });
}

#[cfg(test)]
#[path = "msl_codegen_cache_tests.rs"]
mod tests;
