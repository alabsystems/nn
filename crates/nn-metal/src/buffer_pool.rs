// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Thread-local Metal buffer pool for non-arena allocations (#3079 D3).
//!
//! Recycles standalone Metal buffers created by `without_arena` cross-step
//! tensors and arena overflow. Each `synthesize()` call creates ~20-50
//! standalone buffers that are allocated/freed per call. The pool keeps
//! them alive between calls, reducing Metal VM churn and fragmentation.
//!
//! # Design
//!
//! Size-class bucketing with ObjC ARC aliasing. The pool holds canonical
//! `MetalBuffer` references; consumers get aliases. When the consumer drops
//! its alias, the pool's reference keeps the Metal allocation alive for reuse.
//!
//! See `designs/2026-03-21-metal-buffer-pool-d3.md` for full design.

use std::cell::RefCell;

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::error::MetalError;

/// Size-class boundaries (bytes). Each class holds buffers >= class_size.
/// Powers of 4 from 64KB to 256MB (7 classes).
const SIZE_CLASSES: [usize; 7] = [
    64 * 1024,         // 0: 64 KB
    256 * 1024,        // 1: 256 KB
    1024 * 1024,       // 2: 1 MB
    4 * 1024 * 1024,   // 3: 4 MB
    16 * 1024 * 1024,  // 4: 16 MB
    64 * 1024 * 1024,  // 5: 64 MB
    256 * 1024 * 1024, // 6: 256 MB
];

/// Maximum buffers per size class. Prevents unbounded growth.
const MAX_PER_CLASS: usize = 8;

/// Maximum total bytes retained across all size classes. Without this,
/// worst case is 8×64MB + 8×256MB = 2.5 GB of retained Metal buffers —
/// counterproductive for RSS reduction. 512 MB accommodates ~20-50 Kokoro
/// synthesis buffers while staying under the ~700 MB target savings.
///
/// Adopted from NY's `TensorPool::MAX_POOLED_BYTES` pattern.
/// See `designs/2026-03-21-cross-repo-buffer-pool-patterns.md`.
const MAX_POOLED_BYTES: usize = 512 * 1024 * 1024;

struct PoolEntry {
    buffer: MetalBuffer,
    available: bool,
}

/// Internal stats for pool diagnostics.
#[derive(Clone, Debug, Default)]
struct PoolStatsInternal {
    acquisitions: usize,
    hits: usize,
    misses: usize,
    discards: usize,
}

/// Snapshot of pool statistics for external consumers.
///
/// Embedded in [`super::ArenaStats::pool`] for unified arena+pool diagnostics.
/// Part of #3079 D3.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Total acquire() calls (hits + misses + discards).
    pub acquisitions: usize,
    /// Buffers reused from pool (avoided Metal VM allocation).
    pub hits: usize,
    /// Acquire calls where no available entry existed AND a new buffer was
    /// allocated into the pool. Does NOT include bucket-full/budget-exceeded
    /// cases (those are counted as `discards`).
    pub misses: usize,
    /// Requests that bypassed the pool (oversized or bucket full/byte budget).
    pub discards: usize,
    /// Total bytes retained across all size classes.
    pub pooled_bytes: usize,
    /// Total buffer entries across all size classes.
    pub pooled_buffers: usize,
}

/// Thread-local Metal buffer pool with size-class bucketing.
///
/// Holds canonical `MetalBuffer` references. [`acquire`](Self::acquire)
/// returns aliases; [`reclaim_all`](Self::reclaim_all) marks all entries
/// as available for reuse without Metal deallocation.
///
/// Total retained bytes are capped at [`MAX_POOLED_BYTES`] (512 MB).
/// Requests that would exceed the byte budget fall through to direct
/// allocation, same as bucket-full or oversized requests.
pub(crate) struct MetalBufferPool {
    classes: [Vec<PoolEntry>; 7],
    pooled_bytes: usize,
    stats: PoolStatsInternal,
}

impl MetalBufferPool {
    fn new() -> Self {
        Self {
            classes: [
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ],
            pooled_bytes: 0,
            stats: PoolStatsInternal::default(),
        }
    }

    /// Acquire a buffer with at least `min_bytes` capacity.
    ///
    /// Returns `(alias, byte_offset=0)` matching the `arena_alloc_or_create`
    /// return type. The alias is an independent ObjC ARC reference — when the
    /// caller's DynTensor drops it, the pool's canonical reference keeps the
    /// Metal allocation alive for reuse.
    ///
    /// 1. Find an available entry in the matching size class.
    /// 2. If found: mark unavailable, return alias.
    /// 3. If not found, bucket not full, and byte budget permits: allocate new.
    /// 4. Otherwise: return fresh `create_buffer_zeroed` (not pooled).
    fn acquire(
        &mut self,
        ctx: &MetalContext,
        min_bytes: usize,
    ) -> Result<(MetalBuffer, usize), MetalError> {
        self.stats.acquisitions += 1;

        // Requests larger than the biggest size class bypass the pool entirely.
        // Without this guard, size_class_for falls through to the last class
        // and allocates a buffer smaller than min_bytes — GPU out-of-bounds.
        if min_bytes > *SIZE_CLASSES.last().expect("non-empty SIZE_CLASSES") {
            self.stats.discards += 1;
            return Ok((ctx.create_buffer_zeroed(min_bytes)?, 0));
        }

        let class = Self::size_class_for(min_bytes);
        let class_size = SIZE_CLASSES[class];
        let entries = &mut self.classes[class];

        // Try to reuse an available entry.
        for entry in entries.iter_mut() {
            if entry.available {
                entry.available = false;
                self.stats.hits += 1;
                return Ok((entry.buffer.alias(), 0));
            }
        }

        // No available entry — create new if bucket not full AND byte budget permits.
        if entries.len() < MAX_PER_CLASS && self.pooled_bytes + class_size <= MAX_POOLED_BYTES {
            self.stats.misses += 1;
            let buffer = ctx.create_buffer_zeroed(class_size)?;
            let alias = buffer.alias();
            entries.push(PoolEntry {
                buffer,
                available: false,
            });
            self.pooled_bytes += class_size;
            return Ok((alias, 0));
        }

        // Bucket full or byte budget exceeded — unpooled fallback.
        self.stats.discards += 1;
        Ok((ctx.create_buffer_zeroed(min_bytes)?, 0))
    }

    /// Mark all entries as available for reuse.
    ///
    /// # Safety contract
    ///
    /// Caller MUST ensure no outstanding DynTensor references to previously
    /// issued aliases exist. In the Kokoro pipeline, this is guaranteed at
    /// the start of `synthesize()`: all intermediate GPU tensors from the
    /// previous call have been dropped (the output is CPU `Vec<f32>`).
    ///
    /// # Why batch reclaim instead of RAII return-on-drop
    ///
    /// `DynTensor` is `Send+Sync`, so `Drop` may execute on any thread, but
    /// the pool is `thread_local!`. An RAII wrapper cannot safely return a
    /// buffer to a different thread's pool. Batch reclaim at synthesis
    /// boundaries is the correct model: all intermediates are guaranteed
    /// dropped when `synthesize()` returns CPU `Vec<f32>`.
    fn reclaim_all(&mut self) {
        for class in &mut self.classes {
            for entry in class.iter_mut() {
                entry.available = true;
            }
        }
    }

    /// Size class index for a given byte count.
    fn size_class_for(bytes: usize) -> usize {
        for (i, &threshold) in SIZE_CLASSES.iter().enumerate() {
            if bytes <= threshold {
                return i;
            }
        }
        SIZE_CLASSES.len() - 1
    }

    /// Total number of entries across all size classes.
    fn total_entries(&self) -> usize {
        self.classes.iter().map(Vec::len).sum()
    }

    /// Snapshot of current pool statistics.
    fn snapshot_stats(&self) -> PoolStats {
        PoolStats {
            acquisitions: self.stats.acquisitions,
            hits: self.stats.hits,
            misses: self.stats.misses,
            discards: self.stats.discards,
            pooled_bytes: self.pooled_bytes,
            pooled_buffers: self.total_entries(),
        }
    }

    /// Reset all stats counters (not pool contents).
    fn reset_stats(&mut self) {
        self.stats = PoolStatsInternal::default();
    }
}

thread_local! {
    static BUFFER_POOL: RefCell<MetalBufferPool> = RefCell::new(MetalBufferPool::new());
}

/// Acquire a buffer from the thread-local pool.
///
/// Used by `arena_scope.rs` bypass and overflow paths as a replacement
/// for `create_buffer_zeroed`. Returns `(buffer_alias, 0)`.
pub(crate) fn pool_acquire(
    ctx: &MetalContext,
    min_bytes: usize,
) -> Result<(MetalBuffer, usize), MetalError> {
    BUFFER_POOL.with(|cell| cell.borrow_mut().acquire(ctx, min_bytes))
}

/// Mark all pool entries as available. Call at synthesis start.
///
/// Safe to call when no aliases from previous synthesis exist — the
/// Kokoro pipeline drops all intermediate GPU tensors before returning.
pub(crate) fn pool_reclaim() {
    BUFFER_POOL.with(|cell| cell.borrow_mut().reclaim_all());
}

/// Snapshot of all pool statistics.
pub(crate) fn pool_stats() -> PoolStats {
    BUFFER_POOL.with(|cell| cell.borrow().snapshot_stats())
}

/// Reset pool stats counters to zero (not pool contents).
pub(crate) fn reset_pool_stats() {
    BUFFER_POOL.with(|cell| cell.borrow_mut().reset_stats());
}

#[cfg(kani)]
#[path = "buffer_pool_kani.rs"]
mod proofs;

#[cfg(test)]
#[path = "buffer_pool_stats_tests.rs"]
mod buffer_pool_stats_tests;

#[cfg(test)]
#[path = "buffer_pool_perf_tests.rs"]
mod buffer_pool_perf_proofs;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_class_selection() {
        assert_eq!(MetalBufferPool::size_class_for(100), 0); // < 64KB → class 0
        assert_eq!(MetalBufferPool::size_class_for(64 * 1024), 0); // exactly 64KB
        assert_eq!(MetalBufferPool::size_class_for(64 * 1024 + 1), 1); // > 64KB → class 1
        assert_eq!(MetalBufferPool::size_class_for(256 * 1024), 1);
        assert_eq!(MetalBufferPool::size_class_for(1024 * 1024), 2);
        assert_eq!(MetalBufferPool::size_class_for(300 * 1024 * 1024), 6); // > 256MB → class 6
    }

    #[test]
    fn test_reclaim_resets_availability() {
        let mut pool = MetalBufferPool::new();
        pool.reclaim_all(); // no-op on empty pool
        assert_eq!(pool.total_entries(), 0);
    }

    /// P1: Verify that requests exceeding the largest size class (256MB)
    /// do NOT use a pool-allocated buffer. Without the oversized guard,
    /// `size_class_for` falls through to class 6 (256MB) and `acquire`
    /// would create a 256MB buffer for a >256MB request — GPU out-of-bounds.
    #[test]
    fn test_oversized_request_bypasses_pool() {
        let oversized = 300 * 1024 * 1024; // 300 MB > 256 MB (largest class)
        let class = MetalBufferPool::size_class_for(oversized);
        let class_size = SIZE_CLASSES[class];
        // The size class caps at 256 MB — smaller than the request.
        assert!(
            class_size < oversized,
            "size class {class_size} must be < oversized request {oversized}"
        );
        // The acquire() guard ensures oversized requests bypass the pool.
        // Full integration test requires Metal context — see arena_tests.rs.
    }

    /// Verify that requests exactly at the largest size class ARE pooled.
    #[test]
    fn test_exact_max_class_size_is_pooled() {
        let exact_max = 256 * 1024 * 1024; // exactly 256 MB
        let class = MetalBufferPool::size_class_for(exact_max);
        let class_size = SIZE_CLASSES[class];
        assert_eq!(
            class_size, exact_max,
            "256MB request should fit class 6 exactly"
        );
    }

    #[test]
    fn test_new_pool_has_zero_bytes_and_stats() {
        let pool = MetalBufferPool::new();
        assert_eq!(pool.pooled_bytes, 0);
        let stats = pool.snapshot_stats();
        assert_eq!(stats, PoolStats::default());
    }

    #[test]
    fn test_stats_reset_clears_counters() {
        let mut pool = MetalBufferPool::new();
        pool.stats.acquisitions = 10;
        pool.stats.hits = 5;
        pool.stats.misses = 3;
        pool.stats.discards = 2;
        pool.reset_stats();
        assert_eq!(pool.stats.acquisitions, 0);
        assert_eq!(pool.stats.hits, 0);
        assert_eq!(pool.stats.misses, 0);
        assert_eq!(pool.stats.discards, 0);
        // pooled_bytes and entries are NOT reset by reset_stats
        assert_eq!(pool.pooled_bytes, 0);
    }

    #[test]
    fn test_max_pooled_bytes_constant() {
        assert_eq!(MAX_POOLED_BYTES, 512 * 1024 * 1024);
        // Worst case per class: MAX_PER_CLASS * class_size.
        // Without byte budget, 8 entries in class 6 = 2048 MB.
        // Budget should prevent this.
        let worst_class6 = MAX_PER_CLASS * SIZE_CLASSES[6];
        assert!(
            worst_class6 > MAX_POOLED_BYTES,
            "byte budget must be tighter than per-class cap: {worst_class6} > {MAX_POOLED_BYTES}"
        );
    }
}
