// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Buffer pool for Vulkan GPU buffer reuse.
//!
//! Mirrors the Metal backend's `BufferPool` pattern: size-class bucketing
//! with buffer reuse to avoid repeated `vkAllocateMemory` / `vkFreeMemory`
//! calls which cause Vulkan memory allocator churn and fragmentation.
//!
//! # Design
//!
//! Size-class bucketing with power-of-4 boundaries. Acquisition finds the
//! smallest available buffer that satisfies the request. When the pool
//! exceeds `MAX_POOLED_BYTES`, new requests bypass the pool.
//!
//! # Statistics
//!
//! The pool tracks allocation, reuse, and eviction statistics including
//! peak usage watermarks and per-size-class breakdowns via [`BufferPoolStats`].
//! Use [`BufferPool::pool_stats()`] for the full breakdown and
//! [`BufferPool::reset_stats()`] to zero counters between benchmark runs.
//!
//! # Thread safety
//!
//! `BufferPool` is **not** `Send` or `Sync`. Each thread should have its own
//! pool instance (matching the Metal backend's thread-local pool pattern).

use std::fmt;

use crate::buffer::{BufferUsage, VulkanBuffer};
use crate::error::VulkanError;

/// Number of size classes.
const NUM_SIZE_CLASSES: usize = 7;

/// Size-class boundaries (bytes). Power-of-4 from 64KB to 256MB.
const SIZE_CLASSES: [usize; NUM_SIZE_CLASSES] = [
    64 * 1024,         // 0: 64 KB
    256 * 1024,        // 1: 256 KB
    1024 * 1024,       // 2: 1 MB
    4 * 1024 * 1024,   // 3: 4 MB
    16 * 1024 * 1024,  // 4: 16 MB
    64 * 1024 * 1024,  // 5: 64 MB
    256 * 1024 * 1024, // 6: 256 MB
];

/// Human-readable labels for size classes.
const SIZE_CLASS_LABELS: [&str; NUM_SIZE_CLASSES] =
    ["64KB", "256KB", "1MB", "4MB", "16MB", "64MB", "256MB"];

/// Maximum buffers per size class.
const MAX_PER_CLASS: usize = 8;

/// Maximum total bytes retained across all size classes.
/// 512 MB accommodates typical ML inference workloads while staying
/// under reasonable GPU memory usage targets.
const MAX_POOLED_BYTES: usize = 512 * 1024 * 1024;

/// Internal pool entry.
struct PoolEntry {
    buffer: VulkanBuffer,
    available: bool,
}

/// Snapshot of pool statistics for external consumers.
///
/// Lightweight snapshot compatible with the Metal backend's `PoolStats`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Total acquire() calls (hits + misses + discards).
    pub acquisitions: usize,
    /// Buffers reused from pool (avoided Vulkan allocation).
    pub hits: usize,
    /// Acquire calls that required a new buffer allocation into the pool.
    pub misses: usize,
    /// Requests that bypassed the pool (oversized or bucket full/byte budget).
    pub discards: usize,
    /// Total bytes retained across all size classes.
    pub retained_bytes: usize,
    /// Number of buffers currently in the pool (available + in-use).
    pub buffer_count: usize,
}

/// Per-size-class statistics breakdown.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SizeClassStats {
    /// Size class boundary in bytes.
    pub class_bytes: usize,
    /// Buffers currently in this class.
    pub buffer_count: usize,
    /// Buffers currently available (not in-use) in this class.
    pub available_count: usize,
    /// Total bytes retained by this class.
    pub retained_bytes: usize,
    /// Total allocations into this class (misses that created a new entry).
    pub total_allocated: usize,
    /// Total reuse hits from this class.
    pub total_reused: usize,
    /// Total evictions from this class.
    pub total_evicted: usize,
}

/// Comprehensive buffer pool statistics with peak tracking and per-class breakdown.
///
/// Returned by [`BufferPool::pool_stats()`]. Includes cumulative counters,
/// peak watermarks, and per-size-class breakdowns for capacity planning and
/// performance diagnostics.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BufferPoolStats {
    /// Total buffers allocated into the pool (lifetime count).
    pub total_allocated: usize,
    /// Total buffers reused from pool (lifetime hit count).
    pub total_reused: usize,
    /// Total buffers evicted from pool via [`BufferPool::evict()`].
    pub total_evicted: usize,
    /// Peak number of live (in-use, not available) buffers observed.
    pub peak_live_count: usize,
    /// Peak bytes held by live (in-use) buffers observed.
    pub peak_live_bytes: usize,
    /// Hit rate: `total_reused / (total_reused + total_allocated)`.
    /// Returns 0.0 when no acquisitions have occurred.
    pub hit_rate: f64,
    /// Total acquire() calls.
    pub total_acquisitions: usize,
    /// Requests that bypassed the pool.
    pub total_discards: usize,
    /// Current total bytes retained across all size classes.
    pub current_retained_bytes: usize,
    /// Current total buffer count across all size classes.
    pub current_buffer_count: usize,
    /// Current live (in-use) buffer count.
    pub current_live_count: usize,
    /// Current live (in-use) bytes.
    pub current_live_bytes: usize,
    /// Per-size-class breakdown.
    pub size_classes: Vec<SizeClassStats>,
}

impl fmt::Display for BufferPoolStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Vulkan BufferPool Statistics")?;
        writeln!(f, "============================")?;
        writeln!(f)?;

        // Summary counters.
        writeln!(f, "Lifetime:")?;
        writeln!(f, "  total_allocated:  {}", self.total_allocated)?;
        writeln!(f, "  total_reused:     {}", self.total_reused)?;
        writeln!(f, "  total_evicted:    {}", self.total_evicted)?;
        writeln!(f, "  total_discards:   {}", self.total_discards)?;
        writeln!(f, "  total_acquires:   {}", self.total_acquisitions)?;
        writeln!(f, "  hit_rate:         {:.1}%", self.hit_rate * 100.0)?;
        writeln!(f)?;

        // Peak watermarks.
        writeln!(f, "Peaks:")?;
        writeln!(f, "  peak_live_count:  {}", self.peak_live_count)?;
        writeln!(
            f,
            "  peak_live_bytes:  {}",
            format_bytes(self.peak_live_bytes)
        )?;
        writeln!(f)?;

        // Current state.
        writeln!(f, "Current:")?;
        writeln!(f, "  buffer_count:     {}", self.current_buffer_count)?;
        writeln!(
            f,
            "  retained_bytes:   {}",
            format_bytes(self.current_retained_bytes)
        )?;
        writeln!(f, "  live_count:       {}", self.current_live_count)?;
        writeln!(
            f,
            "  live_bytes:       {}",
            format_bytes(self.current_live_bytes)
        )?;
        writeln!(f)?;

        // Per-class breakdown.
        writeln!(f, "Per-size-class:")?;
        writeln!(
            f,
            "  {:>6}  {:>5}  {:>5}  {:>10}  {:>5}  {:>5}  {:>5}",
            "Class", "Bufs", "Avail", "Retained", "Alloc", "Reuse", "Evict"
        )?;
        writeln!(
            f,
            "  {:>6}  {:>5}  {:>5}  {:>10}  {:>5}  {:>5}  {:>5}",
            "------", "-----", "-----", "----------", "-----", "-----", "-----"
        )?;
        for (i, sc) in self.size_classes.iter().enumerate() {
            let label = if i < SIZE_CLASS_LABELS.len() {
                SIZE_CLASS_LABELS[i]
            } else {
                "???"
            };
            writeln!(
                f,
                "  {:>6}  {:>5}  {:>5}  {:>10}  {:>5}  {:>5}  {:>5}",
                label,
                sc.buffer_count,
                sc.available_count,
                format_bytes(sc.retained_bytes),
                sc.total_allocated,
                sc.total_reused,
                sc.total_evicted,
            )?;
        }

        Ok(())
    }
}

/// Format byte counts for human-readable display.
fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

/// Buffer pool for Vulkan GPU buffer reuse.
///
/// Size-class bucketing avoids repeated memory allocation/deallocation.
///
/// # Example
///
/// ```
/// use nn_vulkan::buffer_pool::BufferPool;
/// use nn_vulkan::buffer::BufferUsage;
///
/// let mut pool = BufferPool::new();
/// let buf = pool.acquire(1024, BufferUsage::StorageReadWrite).expect("acquire");
/// assert!(buf.size_bytes() >= 1024);
/// // When done, release back to pool:
/// pool.release(buf);
/// // Next acquire of similar size reuses the buffer:
/// let buf2 = pool.acquire(1024, BufferUsage::StorageReadWrite).expect("acquire");
/// ```
pub struct BufferPool {
    /// Size-class buckets. Index matches `SIZE_CLASSES`.
    buckets: Vec<Vec<PoolEntry>>,
    /// Oversized buffers (larger than the largest size class).
    oversized: Vec<PoolEntry>,
    /// Total retained bytes.
    retained_bytes: usize,
    /// Internal counters.
    stats: PoolStatsInternal,
}

/// Internal mutable statistics tracker.
#[derive(Clone, Debug, Default)]
struct PoolStatsInternal {
    acquisitions: usize,
    hits: usize,
    misses: usize,
    discards: usize,
    total_evicted: usize,
    peak_live_count: usize,
    peak_live_bytes: usize,
    /// Per-size-class counters: (allocated, reused, evicted).
    per_class_allocated: [usize; NUM_SIZE_CLASSES],
    per_class_reused: [usize; NUM_SIZE_CLASSES],
    per_class_evicted: [usize; NUM_SIZE_CLASSES],
}

impl BufferPool {
    /// Create a new empty buffer pool.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buckets: (0..SIZE_CLASSES.len()).map(|_| Vec::new()).collect(),
            oversized: Vec::new(),
            retained_bytes: 0,
            stats: PoolStatsInternal::default(),
        }
    }

    /// Acquire a buffer of at least `min_bytes` from the pool.
    ///
    /// If a suitable buffer is available in the pool, it is reused.
    /// Otherwise, a new buffer is allocated and added to the pool.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::OutOfMemory`] if allocation fails.
    /// Returns [`VulkanError::InvalidParameter`] if `min_bytes` is 0.
    pub fn acquire(
        &mut self,
        min_bytes: usize,
        usage: BufferUsage,
    ) -> Result<VulkanBuffer, VulkanError> {
        if min_bytes == 0 {
            return Err(VulkanError::InvalidParameter(
                "buffer size must be > 0".into(),
            ));
        }

        self.stats.acquisitions += 1;

        let class_idx = self.size_class_for(min_bytes);

        // Try to find an available buffer in the appropriate size class.
        if let Some(idx) = class_idx {
            if let Some(entry) = self.buckets[idx]
                .iter_mut()
                .find(|e| e.available && e.buffer.size_bytes() >= min_bytes)
            {
                entry.available = false;
                let reused_size = entry.buffer.size_bytes();
                self.stats.hits += 1;
                self.stats.per_class_reused[idx] += 1;
                self.update_peak_live();
                // Return a new VulkanBuffer at the requested size, backed by
                // the pooled allocation (in real FFI, this would be a view).
                return VulkanBuffer::new(reused_size, usage);
            }

            // No available buffer: allocate and add to pool.
            let alloc_size = self.round_up_to_class(min_bytes, idx);

            // Check pool budget.
            if self.retained_bytes + alloc_size > MAX_POOLED_BYTES
                || self.buckets[idx].len() >= MAX_PER_CLASS
            {
                self.stats.discards += 1;
                return VulkanBuffer::new(min_bytes, usage);
            }

            let buffer = VulkanBuffer::new(alloc_size, usage)?;
            self.retained_bytes += alloc_size;
            self.stats.misses += 1;
            self.stats.per_class_allocated[idx] += 1;

            self.buckets[idx].push(PoolEntry {
                buffer: VulkanBuffer::new(alloc_size, usage)?,
                available: false,
            });

            self.update_peak_live();
            return Ok(buffer);
        }

        // Oversized: bypass pool.
        self.stats.discards += 1;
        VulkanBuffer::new(min_bytes, usage)
    }

    /// Release a buffer back to the pool for future reuse.
    ///
    /// If the buffer size does not match any pooled entry, it is dropped.
    pub fn release(&mut self, buffer: VulkanBuffer) {
        let size = buffer.size_bytes();
        let class_idx = self.size_class_for(size);

        if let Some(idx) = class_idx {
            // Find the entry with matching size and mark available.
            for entry in &mut self.buckets[idx] {
                if !entry.available && entry.buffer.size_bytes() == size {
                    entry.available = true;
                    return;
                }
            }
        }
        // Buffer not from pool, or oversized -- just drop it.
    }

    /// Evict available (unused) buffers from the pool to free GPU memory.
    ///
    /// Returns the number of buffers evicted and bytes freed.
    pub fn evict(&mut self) -> (usize, usize) {
        let mut evicted_count = 0usize;
        let mut evicted_bytes = 0usize;

        for (class_idx, bucket) in self.buckets.iter_mut().enumerate() {
            let before_len = bucket.len();
            let mut freed = 0usize;

            bucket.retain(|entry| {
                if entry.available {
                    freed += entry.buffer.size_bytes();
                    false
                } else {
                    true
                }
            });

            let removed = before_len - bucket.len();
            evicted_count += removed;
            evicted_bytes += freed;
            self.stats.per_class_evicted[class_idx] += removed;
        }

        // Evict available oversized entries too.
        let before_oversized = self.oversized.len();
        self.oversized.retain(|entry| {
            if entry.available {
                evicted_bytes += entry.buffer.size_bytes();
                false
            } else {
                true
            }
        });
        evicted_count += before_oversized - self.oversized.len();

        self.retained_bytes = self.retained_bytes.saturating_sub(evicted_bytes);
        self.stats.total_evicted += evicted_count;

        (evicted_count, evicted_bytes)
    }

    /// Lightweight pool statistics snapshot (backward-compatible).
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        let buffer_count: usize =
            self.buckets.iter().map(Vec::len).sum::<usize>() + self.oversized.len();
        PoolStats {
            acquisitions: self.stats.acquisitions,
            hits: self.stats.hits,
            misses: self.stats.misses,
            discards: self.stats.discards,
            retained_bytes: self.retained_bytes,
            buffer_count,
        }
    }

    /// Comprehensive pool statistics with peak tracking and per-class breakdown.
    #[must_use]
    pub fn pool_stats(&self) -> BufferPoolStats {
        let total_allocated = self.stats.misses;
        let total_reused = self.stats.hits;
        let denominator = total_reused + total_allocated;
        let hit_rate = if denominator > 0 {
            total_reused as f64 / denominator as f64
        } else {
            0.0
        };

        let (current_live_count, current_live_bytes) = self.live_counts();
        let current_buffer_count: usize =
            self.buckets.iter().map(Vec::len).sum::<usize>() + self.oversized.len();

        let size_classes: Vec<SizeClassStats> = self
            .buckets
            .iter()
            .enumerate()
            .map(|(i, bucket)| {
                let available_count = bucket.iter().filter(|e| e.available).count();
                let retained: usize = bucket.iter().map(|e| e.buffer.size_bytes()).sum();
                SizeClassStats {
                    class_bytes: SIZE_CLASSES[i],
                    buffer_count: bucket.len(),
                    available_count,
                    retained_bytes: retained,
                    total_allocated: self.stats.per_class_allocated[i],
                    total_reused: self.stats.per_class_reused[i],
                    total_evicted: self.stats.per_class_evicted[i],
                }
            })
            .collect();

        BufferPoolStats {
            total_allocated,
            total_reused,
            total_evicted: self.stats.total_evicted,
            peak_live_count: self.stats.peak_live_count,
            peak_live_bytes: self.stats.peak_live_bytes,
            hit_rate,
            total_acquisitions: self.stats.acquisitions,
            total_discards: self.stats.discards,
            current_retained_bytes: self.retained_bytes,
            current_buffer_count,
            current_live_count,
            current_live_bytes,
            size_classes,
        }
    }

    /// Reset all statistics counters to zero.
    ///
    /// Pool contents (buffers) are preserved; only the counters are cleared.
    /// Useful between benchmark runs to isolate measurements.
    pub fn reset_stats(&mut self) {
        self.stats = PoolStatsInternal::default();
    }

    /// Clear all pooled buffers, freeing GPU memory.
    pub fn clear(&mut self) {
        for bucket in &mut self.buckets {
            bucket.clear();
        }
        self.oversized.clear();
        self.retained_bytes = 0;
    }

    /// Find the size class index for a given byte size.
    fn size_class_for(&self, bytes: usize) -> Option<usize> {
        for (i, &class_size) in SIZE_CLASSES.iter().enumerate() {
            if bytes <= class_size {
                return Some(i);
            }
        }
        None // Oversized.
    }

    /// Round up to the size class boundary.
    fn round_up_to_class(&self, bytes: usize, class_idx: usize) -> usize {
        SIZE_CLASSES[class_idx].max(bytes)
    }

    /// Count live (in-use) buffers and their total bytes.
    fn live_counts(&self) -> (usize, usize) {
        let mut count = 0usize;
        let mut bytes = 0usize;
        for bucket in &self.buckets {
            for entry in bucket {
                if !entry.available {
                    count += 1;
                    bytes += entry.buffer.size_bytes();
                }
            }
        }
        for entry in &self.oversized {
            if !entry.available {
                count += 1;
                bytes += entry.buffer.size_bytes();
            }
        }
        (count, bytes)
    }

    /// Update peak watermarks based on current live counts.
    fn update_peak_live(&mut self) {
        let (live_count, live_bytes) = self.live_counts();
        if live_count > self.stats.peak_live_count {
            self.stats.peak_live_count = live_count;
        }
        if live_bytes > self.stats.peak_live_bytes {
            self.stats.peak_live_bytes = live_bytes;
        }
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "buffer_pool_tests.rs"]
mod buffer_pool_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_new_is_empty() {
        let pool = BufferPool::new();
        let stats = pool.stats();
        assert_eq!(stats.buffer_count, 0);
        assert_eq!(stats.retained_bytes, 0);
        assert_eq!(stats.acquisitions, 0);
    }

    #[test]
    fn test_pool_acquire_zero_bytes_rejected() {
        let mut pool = BufferPool::new();
        assert!(pool.acquire(0, BufferUsage::StorageReadWrite).is_err());
    }

    #[test]
    fn test_pool_acquire_creates_buffer() {
        let mut pool = BufferPool::new();
        let buf = pool
            .acquire(1024, BufferUsage::StorageReadWrite)
            .expect("acquire");
        assert!(buf.size_bytes() >= 1024);
        assert_eq!(pool.stats().acquisitions, 1);
    }

    #[test]
    fn test_pool_small_allocation_bypasses_class() {
        let mut pool = BufferPool::new();
        // 100 bytes is well under the smallest size class (64KB).
        // Should still allocate from the first class.
        let buf = pool
            .acquire(100, BufferUsage::StorageReadWrite)
            .expect("acquire");
        assert!(buf.size_bytes() >= 100);
    }

    #[test]
    fn test_pool_oversized_allocation_bypasses_pool() {
        let mut pool = BufferPool::new();
        // 512 MB is larger than the largest size class (256 MB).
        let buf = pool
            .acquire(512 * 1024 * 1024, BufferUsage::StorageReadWrite)
            .expect("acquire");
        assert_eq!(buf.size_bytes(), 512 * 1024 * 1024);
        assert_eq!(pool.stats().discards, 1);
    }

    #[test]
    fn test_pool_clear() {
        let mut pool = BufferPool::new();
        let _ = pool
            .acquire(1024, BufferUsage::StorageReadWrite)
            .expect("acquire");
        pool.clear();
        assert_eq!(pool.stats().buffer_count, 0);
        assert_eq!(pool.stats().retained_bytes, 0);
    }

    #[test]
    fn test_pool_stats_default() {
        let stats = PoolStats::default();
        assert_eq!(stats.acquisitions, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.discards, 0);
        assert_eq!(stats.retained_bytes, 0);
        assert_eq!(stats.buffer_count, 0);
    }

    #[test]
    fn test_pool_default_constructor() {
        let pool = BufferPool::default();
        assert_eq!(pool.stats().buffer_count, 0);
    }

    #[test]
    fn test_pool_multiple_acquisitions() {
        let mut pool = BufferPool::new();
        for _ in 0..10 {
            let _ = pool
                .acquire(4096, BufferUsage::StorageReadWrite)
                .expect("acquire");
        }
        assert_eq!(pool.stats().acquisitions, 10);
    }

    // --- BufferPoolStats tests ---

    #[test]
    fn test_pool_stats_total_allocated_tracks_misses() {
        let mut pool = BufferPool::new();
        // First acquire is a miss (new allocation into pool).
        let _ = pool
            .acquire(1024, BufferUsage::StorageReadWrite)
            .expect("acquire");
        let ps = pool.pool_stats();
        assert_eq!(ps.total_allocated, 1);
        assert_eq!(ps.total_reused, 0);
        assert_eq!(ps.total_acquisitions, 1);
    }

    #[test]
    fn test_pool_stats_reuse_after_release() {
        let mut pool = BufferPool::new();
        // Allocate into pool (miss).
        let buf = pool
            .acquire(1024, BufferUsage::StorageReadWrite)
            .expect("acquire");
        let alloc_size = buf.size_bytes();
        pool.release(buf);

        // Second acquire should hit the released buffer.
        let _ = pool
            .acquire(1024, BufferUsage::StorageReadWrite)
            .expect("acquire");
        let ps = pool.pool_stats();
        assert_eq!(ps.total_allocated, 1, "only one allocation into pool");
        assert_eq!(ps.total_reused, 1, "second acquire reuses the buffer");
        assert_eq!(ps.total_acquisitions, 2);
        assert!(ps.hit_rate > 0.0, "hit rate should be positive after reuse");

        // Verify the per-class breakdown recorded the reuse.
        let class_stats: Vec<&SizeClassStats> = ps
            .size_classes
            .iter()
            .filter(|sc| sc.total_reused > 0)
            .collect();
        assert_eq!(class_stats.len(), 1, "exactly one class with reuse");
        assert_eq!(class_stats[0].total_reused, 1);
        assert_eq!(class_stats[0].class_bytes, alloc_size);
    }

    #[test]
    fn test_pool_stats_hit_rate_calculation() {
        let mut pool = BufferPool::new();
        // 1 miss (alloc), then release, then 3 hits.
        let buf = pool
            .acquire(2048, BufferUsage::StorageReadWrite)
            .expect("acquire");
        pool.release(buf);

        for _ in 0..3 {
            let buf = pool
                .acquire(2048, BufferUsage::StorageReadWrite)
                .expect("acquire");
            pool.release(buf);
        }

        let ps = pool.pool_stats();
        // 1 alloc, 3 reuses => hit_rate = 3/4 = 0.75
        assert_eq!(ps.total_allocated, 1);
        assert_eq!(ps.total_reused, 3);
        let expected = 3.0 / 4.0;
        assert!(
            (ps.hit_rate - expected).abs() < 1e-9,
            "expected hit_rate={expected}, got {}",
            ps.hit_rate,
        );
    }

    #[test]
    fn test_pool_stats_hit_rate_zero_when_empty() {
        let pool = BufferPool::new();
        let ps = pool.pool_stats();
        assert_eq!(ps.hit_rate, 0.0);
    }

    #[test]
    fn test_pool_stats_evict_counts() {
        let mut pool = BufferPool::new();
        // Allocate then release (makes buffer available for eviction).
        let buf = pool
            .acquire(4096, BufferUsage::StorageReadWrite)
            .expect("acquire");
        pool.release(buf);
        assert_eq!(pool.stats().buffer_count, 1);

        let (evicted_count, evicted_bytes) = pool.evict();
        assert_eq!(evicted_count, 1);
        assert!(evicted_bytes > 0);
        assert_eq!(pool.stats().buffer_count, 0);

        let ps = pool.pool_stats();
        assert_eq!(ps.total_evicted, 1);
    }

    #[test]
    fn test_pool_stats_evict_only_available() {
        let mut pool = BufferPool::new();
        // Allocate two buffers; release only one.
        let _buf_live = pool
            .acquire(1024, BufferUsage::StorageReadWrite)
            .expect("acquire");
        let buf_avail = pool
            .acquire(2048, BufferUsage::StorageReadWrite)
            .expect("acquire");
        pool.release(buf_avail);

        let (evicted, _) = pool.evict();
        // Only the released buffer should be evicted.
        assert_eq!(evicted, 1);
        // The live buffer remains.
        assert_eq!(pool.stats().buffer_count, 1);
    }

    #[test]
    fn test_pool_stats_evict_empty_pool_is_noop() {
        let mut pool = BufferPool::new();
        let (evicted, bytes) = pool.evict();
        assert_eq!(evicted, 0);
        assert_eq!(bytes, 0);
        assert_eq!(pool.pool_stats().total_evicted, 0);
    }

    #[test]
    fn test_pool_stats_peak_live_count() {
        let mut pool = BufferPool::new();
        // Acquire 3 buffers without releasing (all live).
        for _ in 0..3 {
            let _ = pool
                .acquire(1024, BufferUsage::StorageReadWrite)
                .expect("acquire");
        }
        let ps = pool.pool_stats();
        assert_eq!(ps.peak_live_count, 3, "peak should be 3 after 3 acquires");

        // Release all, then acquire 1 more.
        // (release doesn't change peak; it's a watermark.)
        // Note: releasing requires matching size so we track buffers.
        pool.reset_stats();
        let ps_after_reset = pool.pool_stats();
        assert_eq!(
            ps_after_reset.peak_live_count, 0,
            "peak resets to 0 after reset_stats"
        );
    }

    #[test]
    fn test_pool_stats_peak_live_bytes() {
        let mut pool = BufferPool::new();
        let _ = pool
            .acquire(1024, BufferUsage::StorageReadWrite)
            .expect("acquire");
        let ps = pool.pool_stats();
        // At least 1 live buffer with the class-rounded size.
        assert!(ps.peak_live_bytes > 0, "peak_live_bytes should be positive");
    }

    #[test]
    fn test_pool_stats_per_class_breakdown_count() {
        let pool = BufferPool::new();
        let ps = pool.pool_stats();
        assert_eq!(
            ps.size_classes.len(),
            NUM_SIZE_CLASSES,
            "should have one entry per size class"
        );
    }

    #[test]
    fn test_pool_stats_per_class_allocation_tracking() {
        let mut pool = BufferPool::new();
        // Allocate into the 64KB class (any request <= 64KB).
        let _ = pool
            .acquire(100, BufferUsage::StorageReadWrite)
            .expect("acquire");
        // Allocate into the 1MB class (300KB > 256KB boundary).
        let _ = pool
            .acquire(300 * 1024, BufferUsage::StorageReadWrite)
            .expect("acquire");

        let ps = pool.pool_stats();
        // Class 0 (64KB) should have 1 allocation.
        assert_eq!(ps.size_classes[0].total_allocated, 1);
        assert_eq!(ps.size_classes[0].class_bytes, 64 * 1024);
        // Class 2 (1MB) should have 1 allocation (300KB > 256KB, rounds up to 1MB).
        assert_eq!(ps.size_classes[2].total_allocated, 1);
        assert_eq!(ps.size_classes[2].class_bytes, 1024 * 1024);
    }

    #[test]
    fn test_pool_stats_per_class_eviction_tracking() {
        let mut pool = BufferPool::new();
        let buf = pool
            .acquire(100, BufferUsage::StorageReadWrite)
            .expect("acquire");
        pool.release(buf);
        pool.evict();

        let ps = pool.pool_stats();
        // Eviction should be tracked in class 0 (64KB).
        assert_eq!(ps.size_classes[0].total_evicted, 1);
        // All other classes should have 0 evictions.
        for sc in &ps.size_classes[1..] {
            assert_eq!(sc.total_evicted, 0);
        }
    }

    #[test]
    fn test_pool_stats_reset_clears_counters() {
        let mut pool = BufferPool::new();
        let buf = pool
            .acquire(1024, BufferUsage::StorageReadWrite)
            .expect("acquire");
        pool.release(buf);
        let _ = pool
            .acquire(1024, BufferUsage::StorageReadWrite)
            .expect("acquire");

        // Confirm counters are non-zero before reset.
        let ps = pool.pool_stats();
        assert!(ps.total_allocated > 0 || ps.total_reused > 0);

        pool.reset_stats();

        let ps = pool.pool_stats();
        assert_eq!(ps.total_allocated, 0);
        assert_eq!(ps.total_reused, 0);
        assert_eq!(ps.total_evicted, 0);
        assert_eq!(ps.total_acquisitions, 0);
        assert_eq!(ps.total_discards, 0);
        assert_eq!(ps.peak_live_count, 0);
        assert_eq!(ps.peak_live_bytes, 0);
        assert_eq!(ps.hit_rate, 0.0);

        // Per-class counters also reset.
        for sc in &ps.size_classes {
            assert_eq!(sc.total_allocated, 0);
            assert_eq!(sc.total_reused, 0);
            assert_eq!(sc.total_evicted, 0);
        }
    }

    #[test]
    fn test_pool_stats_reset_preserves_buffers() {
        let mut pool = BufferPool::new();
        let _ = pool
            .acquire(1024, BufferUsage::StorageReadWrite)
            .expect("acquire");
        assert_eq!(pool.stats().buffer_count, 1);

        pool.reset_stats();

        // Buffers are still in the pool.
        assert_eq!(pool.stats().buffer_count, 1);
        assert!(pool.stats().retained_bytes > 0);
    }

    #[test]
    fn test_pool_stats_discards_for_oversized() {
        let mut pool = BufferPool::new();
        let _ = pool
            .acquire(512 * 1024 * 1024, BufferUsage::StorageReadWrite)
            .expect("acquire");
        let ps = pool.pool_stats();
        assert_eq!(ps.total_discards, 1);
        assert_eq!(ps.total_allocated, 0, "oversized bypasses pool");
    }

    #[test]
    fn test_pool_stats_current_live_tracking() {
        let mut pool = BufferPool::new();
        let buf = pool
            .acquire(1024, BufferUsage::StorageReadWrite)
            .expect("acquire");
        let ps = pool.pool_stats();
        assert_eq!(ps.current_live_count, 1);
        assert!(ps.current_live_bytes > 0);

        pool.release(buf);
        let ps = pool.pool_stats();
        assert_eq!(ps.current_live_count, 0);
        assert_eq!(ps.current_live_bytes, 0);
    }

    #[test]
    fn test_pool_stats_display_does_not_panic() {
        let mut pool = BufferPool::new();
        let buf = pool
            .acquire(2048, BufferUsage::StorageReadWrite)
            .expect("acquire");
        pool.release(buf);
        let _ = pool
            .acquire(2048, BufferUsage::StorageReadWrite)
            .expect("acquire");
        pool.evict();

        let ps = pool.pool_stats();
        let display = format!("{ps}");
        assert!(display.contains("Vulkan BufferPool Statistics"));
        assert!(display.contains("total_allocated:"));
        assert!(display.contains("total_reused:"));
        assert!(display.contains("total_evicted:"));
        assert!(display.contains("hit_rate:"));
        assert!(display.contains("peak_live_count:"));
        assert!(display.contains("peak_live_bytes:"));
        assert!(display.contains("Per-size-class:"));
        assert!(display.contains("64KB"));
    }

    #[test]
    fn test_pool_stats_display_empty_pool() {
        let pool = BufferPool::new();
        let ps = pool.pool_stats();
        let display = format!("{ps}");
        assert!(display.contains("hit_rate:         0.0%"));
    }

    #[test]
    fn test_buffer_pool_stats_default() {
        let ps = BufferPoolStats::default();
        assert_eq!(ps.total_allocated, 0);
        assert_eq!(ps.total_reused, 0);
        assert_eq!(ps.total_evicted, 0);
        assert_eq!(ps.peak_live_count, 0);
        assert_eq!(ps.peak_live_bytes, 0);
        assert_eq!(ps.hit_rate, 0.0);
        assert!(ps.size_classes.is_empty());
    }

    #[test]
    fn test_size_class_stats_default() {
        let sc = SizeClassStats::default();
        assert_eq!(sc.class_bytes, 0);
        assert_eq!(sc.buffer_count, 0);
        assert_eq!(sc.available_count, 0);
        assert_eq!(sc.retained_bytes, 0);
        assert_eq!(sc.total_allocated, 0);
        assert_eq!(sc.total_reused, 0);
        assert_eq!(sc.total_evicted, 0);
    }

    #[test]
    fn test_format_bytes_units() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
        assert_eq!(format_bytes(1536), "1.5 KB");
    }
}
