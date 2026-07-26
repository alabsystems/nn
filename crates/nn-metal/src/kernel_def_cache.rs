// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Thread-local LRU cache for [`TensorKernelDef`] IR definitions.
//!
//! DynTensor GPU ops rebuild `TensorKernelDef` on every call via
//! `TensorBlockBuilder`. The resulting IR is deterministic given the same
//! `(operation, input_shapes, parameters)` tuple. This cache eliminates
//! redundant IR construction by caching definitions keyed on a `u64` hash
//! of those inputs.
//!
//! The cache sits between the DynTensor `GpuBackend` trait methods and the
//! existing `dispatch_def()` → `execute_tensor_dispatch_to_buffer()` path.
//! Pre-built models (SileroVad, HTDemucs) are unaffected — they construct
//! `TensorKernelDef` once at model creation time.
//!
//! ## Cache hit path (zero heap allocations)
//!
//! [`get_or_build`] accepts borrowed data (`&str`, `&[&[usize]]`, `&[u64]`,
//! `DType`) and computes only a `u64` hash on the stack. On cache hit, the
//! stored owned key is compared against the borrowed references — no `String`,
//! `Vec<Vec<usize>>`, or `Vec<u64>` allocations occur. Owned `KernelDefKey`
//! is only constructed on cache miss (to store alongside the new entry).
//!
//! Design: `designs/2026-03-07-gpu-dispatch-unification.md` (Direction 1).

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use nn_core::DType;
use nn_dsl::tensor_ir::TensorKernelDef;

/// Default maximum number of cached kernel definitions before LRU eviction.
const DEFAULT_MAX_ENTRIES: usize = 512;

thread_local! {
    static CACHE: RefCell<KernelDefCache> = RefCell::new(KernelDefCache::new());
}

/// Cache key combining operation tag, input shapes, scalar parameters, and dtype.
///
/// Stored inside the cache for collision detection. Constructed only on cache
/// miss — the hot path uses [`compute_hash`] and [`KernelDefKey::eq_ref`] to
/// avoid allocating these owned fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct KernelDefKey {
    hash: u64,
    op: String,
    shapes: Vec<Vec<usize>>,
    params: Vec<u64>,
    dtype: DType,
}

impl KernelDefKey {
    /// Build a cache key from operation name, input shapes, scalar parameters, and dtype.
    ///
    /// Allocates owned copies of all fields. Prefer passing borrowed data to
    /// [`get_or_build`] directly — this constructor is only needed for tests
    /// and the cache-miss path.
    pub(crate) fn new(op: &str, shapes: &[&[usize]], params: &[u64], dtype: DType) -> Self {
        Self {
            hash: compute_hash(op, shapes, params, dtype),
            op: op.to_owned(),
            shapes: shapes.iter().map(|s| s.to_vec()).collect(),
            params: params.to_vec(),
            dtype,
        }
    }

    /// Compare this owned key against borrowed references (zero allocation).
    fn eq_ref(&self, op: &str, shapes: &[&[usize]], params: &[u64], dtype: DType) -> bool {
        self.dtype == dtype
            && self.op == op
            && self.params == params
            && self.shapes.len() == shapes.len()
            && self
                .shapes
                .iter()
                .zip(shapes.iter())
                .all(|(owned, borrowed)| owned.as_slice() == *borrowed)
    }

    /// Create a key with a forced hash value (for testing hash collisions).
    #[cfg(test)]
    pub(crate) fn with_forced_hash(
        op: &str,
        shapes: &[&[usize]],
        params: &[u64],
        hash: u64,
    ) -> Self {
        Self {
            hash,
            op: op.to_owned(),
            shapes: shapes.iter().map(|s| s.to_vec()).collect(),
            params: params.to_vec(),
            dtype: DType::F32,
        }
    }
}

/// Compute the cache lookup hash from borrowed data (zero allocation).
fn compute_hash(op: &str, shapes: &[&[usize]], params: &[u64], dtype: DType) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    op.hash(&mut hasher);
    for shape in shapes {
        shape.len().hash(&mut hasher);
        for &d in *shape {
            d.hash(&mut hasher);
        }
    }
    for &p in params {
        p.hash(&mut hasher);
    }
    dtype.hash(&mut hasher);
    hasher.finish()
}

/// Thread-local LRU cache for `TensorKernelDef` IR definitions.
///
/// Mirrors the [`PipelineCache`](crate::PipelineCache) pattern: `HashMap` keyed
/// on `u64` hashes with generation-based LRU eviction. Stores the full
/// `KernelDefKey` alongside each entry for collision detection.
pub(crate) struct KernelDefCache {
    entries: HashMap<u64, (KernelDefKey, Arc<TensorKernelDef>)>,
    access_gen: HashMap<u64, u64>,
    gen_counter: u64,
    max_entries: usize,
}

impl KernelDefCache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            access_gen: HashMap::new(),
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

/// Look up a cached `TensorKernelDef` or build and cache one.
///
/// Accepts borrowed data directly — zero heap allocations on cache hit.
/// On cache miss, constructs an owned `KernelDefKey` to store alongside
/// the newly built def.
///
/// # Example
///
/// ```ignore
/// // NOTE: ignore — pub(crate) API using crate-internal paths
/// let def = get_or_build("matmul", &[l_shape, r_shape], &[], DType::F32, || {
///     let mut b = TensorBlockBuilder::new("dyn_matmul");
///     let lhs = b.add_input("lhs", l_shape);
///     let rhs = b.add_input("rhs", r_shape);
///     let out = b.add_matmul(lhs, rhs, false, None, &out_shape);
///     crate::build_kernel(b, out)
/// })?;
/// ```
pub(crate) fn get_or_build<F>(
    op: &str,
    shapes: &[&[usize]],
    params: &[u64],
    dtype: DType,
    build: F,
) -> nn_core::Result<Arc<TensorKernelDef>>
where
    F: FnOnce() -> nn_core::Result<TensorKernelDef>,
{
    CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        let h = compute_hash(op, shapes, params, dtype);

        if let Some((stored_key, def)) = cache.entries.get(&h) {
            if stored_key.eq_ref(op, shapes, params, dtype) {
                let def = Arc::clone(def);
                cache.stamp(h);
                return Ok(def);
            }
            // Hash collision: different keys mapped to the same hash.
            // Fall through to rebuild and replace.
        }

        let def = Arc::new(build()?);

        if !cache.entries.contains_key(&h) && cache.entries.len() >= cache.max_entries {
            cache.evict_lru();
        }

        let ret = Arc::clone(&def);
        let key = KernelDefKey::new(op, shapes, params, dtype);
        cache.entries.insert(h, (key, def));
        cache.stamp(h);
        Ok(ret)
    })
}

/// Look up a cached def using a pre-constructed key (for testing with forced hashes).
#[cfg(test)]
pub(crate) fn get_or_build_with_key<F>(
    key: KernelDefKey,
    build: F,
) -> nn_core::Result<Arc<TensorKernelDef>>
where
    F: FnOnce() -> nn_core::Result<TensorKernelDef>,
{
    CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        let h = key.hash;

        if let Some((stored_key, def)) = cache.entries.get(&h) {
            if *stored_key == key {
                let def = Arc::clone(def);
                cache.stamp(h);
                return Ok(def);
            }
        }

        let def = Arc::new(build()?);

        if !cache.entries.contains_key(&h) && cache.entries.len() >= cache.max_entries {
            cache.evict_lru();
        }

        let ret = Arc::clone(&def);
        cache.entries.insert(h, (key, def));
        cache.stamp(h);
        Ok(ret)
    })
}

/// Number of cached kernel definitions (for testing/diagnostics).
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
        cache.max_entries = DEFAULT_MAX_ENTRIES;
    });
}

/// Set the maximum number of cached entries (for testing eviction).
#[cfg(test)]
pub(crate) fn set_max_entries(n: usize) {
    CACHE.with(|cell| {
        cell.borrow_mut().max_entries = n;
    });
}

#[cfg(test)]
#[path = "kernel_def_cache_tests.rs"]
mod tests;
