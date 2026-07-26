// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Size-class GPU buffer allocator for reduced Metal memory fragmentation.
//!
//! [`SizeClassAllocator`] manages pools of pre-allocated Metal GPU buffers
//! organized by size class. Allocation rounds up to the next size class and
//! returns buffers to the free list on deallocation (rather than releasing
//! them back to the Metal VM), reducing fragmentation during Kokoro inference.
//!
//! # Size classes
//!
//! 8 power-of-4 classes from 4 KB to 64 MB:
//!
//! | Class | Size   |
//! |-------|--------|
//! | 0     | 4 KB   |
//! | 1     | 16 KB  |
//! | 2     | 64 KB  |
//! | 3     | 256 KB |
//! | 4     | 1 MB   |
//! | 5     | 4 MB   |
//! | 6     | 16 MB  |
//! | 7     | 64 MB  |
//!
//! Requests larger than 64 MB are forwarded directly to the Metal allocator
//! and tracked as oversized allocations (not pooled).
//!
//! Part of #4264 (RTF optimization).

/// Size-class boundaries (bytes). Powers of 4 from 4 KB to 64 MB (8 classes).
pub(crate) const SIZE_CLASS_BOUNDARIES: [usize; 8] = [
    4 * 1024,          // 0:  4 KB
    16 * 1024,         // 1: 16 KB
    64 * 1024,         // 2: 64 KB
    256 * 1024,        // 3: 256 KB
    1024 * 1024,       // 4:  1 MB
    4 * 1024 * 1024,   // 5:  4 MB
    16 * 1024 * 1024,  // 6: 16 MB
    64 * 1024 * 1024,  // 7: 64 MB
];

/// Number of size classes.
pub(crate) const NUM_SIZE_CLASSES: usize = SIZE_CLASS_BOUNDARIES.len();

/// Maximum buffers retained per size class.
const MAX_FREE_PER_CLASS: usize = 16;

/// Per-size-class statistics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SizeClassStats {
    /// Number of allocations served from the free list (reuse).
    pub hits: usize,
    /// Number of allocations that required a new buffer (no free entry).
    pub misses: usize,
    /// Current number of buffers on the free list.
    pub free_count: usize,
    /// Current number of buffers in use (allocated, not yet freed).
    pub in_use_count: usize,
    /// Peak number of buffers in use simultaneously.
    pub peak_in_use: usize,
    /// Total bytes currently retained on the free list.
    pub free_bytes: usize,
}

impl SizeClassStats {
    /// Total allocation count (hits + misses).
    #[must_use]
    pub fn total_allocs(&self) -> usize {
        self.hits + self.misses
    }

    /// Hit rate as a fraction in [0.0, 1.0]. Returns 0.0 if no allocations.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_allocs();
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }
}

/// Aggregate statistics across all size classes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BufferPoolSizeClassStats {
    /// Per-class statistics, indexed by size class (0..NUM_SIZE_CLASSES).
    pub per_class: [SizeClassStats; NUM_SIZE_CLASSES],
    /// Number of oversized allocations (> 64 MB) that bypassed the pool.
    pub oversized_allocs: usize,
    /// Total bytes currently on free lists across all classes.
    pub total_free_bytes: usize,
    /// Total bytes currently allocated (in use) across all classes.
    pub total_used_bytes: usize,
    /// Fragmentation ratio: total_free_bytes / (total_free_bytes + total_used_bytes).
    /// 0.0 means no wasted memory; 1.0 means all pooled memory is free (idle).
    pub fragmentation_ratio: f64,
    /// Overall hit rate across all size classes.
    pub hit_rate: f64,
}

/// Size-class GPU buffer allocator.
///
/// This is a pure-logic allocator that does not hold actual Metal buffers.
/// It tracks allocation and deallocation events, per-class statistics, and
/// free-list occupancy. The actual Metal buffer creation/reuse is managed
/// by the caller (or by integration with `MetalBufferPool`).
///
/// # Design rationale
///
/// Separated from `MetalBufferPool` to allow:
/// 1. Unit testing without a Metal context (all tests run on CI).
/// 2. Compositional use — can be integrated into `ActivationArena` or
///    `MetalBufferPool` independently.
/// 3. Clear separation between allocation policy (this module) and Metal
///    buffer lifecycle management (`buffer_pool.rs`, `arena.rs`).
#[derive(Debug)]
pub struct SizeClassAllocator {
    /// Per-class free list counts and stats.
    classes: [ClassState; NUM_SIZE_CLASSES],
    /// Oversized allocation count (requests > largest size class).
    oversized_allocs: usize,
}

/// Per-class internal state.
#[derive(Debug, Clone)]
#[derive(Default)]
struct ClassState {
    /// Number of buffers currently on the free list.
    free_count: usize,
    /// Number of buffers currently allocated (not yet freed).
    in_use_count: usize,
    /// Peak in_use_count.
    peak_in_use: usize,
    /// Total hits (reuse from free list).
    hits: usize,
    /// Total misses (new allocation required).
    misses: usize,
}


impl SizeClassAllocator {
    /// Create a new empty allocator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            classes: Default::default(),
            oversized_allocs: 0,
        }
    }

    /// Determine the size class for a given byte count.
    ///
    /// Returns `Some(class_index)` if the request fits within a size class,
    /// or `None` if it exceeds the largest class (64 MB).
    #[must_use]
    pub fn size_class_for(bytes: usize) -> Option<usize> {
        if bytes == 0 {
            return Some(0); // Zero-byte requests go to smallest class.
        }
        for (i, &boundary) in SIZE_CLASS_BOUNDARIES.iter().enumerate() {
            if bytes <= boundary {
                return Some(i);
            }
        }
        None // Oversized.
    }

    /// Size of the given class in bytes. Panics if `class >= NUM_SIZE_CLASSES`.
    #[must_use]
    pub fn class_size(class: usize) -> usize {
        SIZE_CLASS_BOUNDARIES[class]
    }

    /// Allocate a buffer of at least `requested_bytes`.
    ///
    /// Returns the size class index and the actual allocation size (rounded
    /// up to the class boundary). If the request exceeds 64 MB, returns
    /// `None` (caller must allocate directly from Metal).
    ///
    /// If a free buffer exists in the matching class, it is reused (hit).
    /// Otherwise, a new buffer must be created (miss).
    pub fn allocate(&mut self, requested_bytes: usize) -> Option<AllocResult> {
        let class = Self::size_class_for(requested_bytes)?;
        let class_bytes = SIZE_CLASS_BOUNDARIES[class];
        let state = &mut self.classes[class];

        let reused = if state.free_count > 0 {
            state.free_count -= 1;
            state.hits += 1;
            true
        } else {
            state.misses += 1;
            false
        };

        state.in_use_count += 1;
        if state.in_use_count > state.peak_in_use {
            state.peak_in_use = state.in_use_count;
        }

        Some(AllocResult {
            class,
            alloc_bytes: class_bytes,
            reused,
        })
    }

    /// Return a buffer to its size class's free list.
    ///
    /// `class` must be the same value returned by [`allocate`](Self::allocate).
    /// Returns `false` if the free list is full (caller should release the
    /// Metal buffer to the OS), `true` if the buffer was accepted.
    pub fn deallocate(&mut self, class: usize) -> bool {
        if class >= NUM_SIZE_CLASSES {
            return false;
        }
        let state = &mut self.classes[class];
        if state.in_use_count == 0 {
            return false; // Double-free protection.
        }
        state.in_use_count -= 1;

        if state.free_count >= MAX_FREE_PER_CLASS {
            return false; // Free list full — release to OS.
        }
        state.free_count += 1;
        true
    }

    /// Record an oversized allocation (> 64 MB).
    pub fn record_oversized(&mut self) {
        self.oversized_allocs += 1;
    }

    /// Snapshot of all statistics.
    #[must_use]
    pub fn stats(&self) -> BufferPoolSizeClassStats {
        let mut per_class = [SizeClassStats::default(); NUM_SIZE_CLASSES];
        let mut total_free_bytes: usize = 0;
        let mut total_used_bytes: usize = 0;
        let mut total_hits: usize = 0;
        let mut total_allocs: usize = 0;

        for (i, state) in self.classes.iter().enumerate() {
            let class_bytes = SIZE_CLASS_BOUNDARIES[i];
            let free_bytes = state.free_count.saturating_mul(class_bytes);
            let used_bytes = state.in_use_count.saturating_mul(class_bytes);

            per_class[i] = SizeClassStats {
                hits: state.hits,
                misses: state.misses,
                free_count: state.free_count,
                in_use_count: state.in_use_count,
                peak_in_use: state.peak_in_use,
                free_bytes,
            };

            total_free_bytes = total_free_bytes.saturating_add(free_bytes);
            total_used_bytes = total_used_bytes.saturating_add(used_bytes);
            total_hits = total_hits.saturating_add(state.hits);
            total_allocs = total_allocs.saturating_add(state.hits + state.misses);
        }

        let total_pooled = total_free_bytes.saturating_add(total_used_bytes);
        let fragmentation_ratio = if total_pooled == 0 {
            0.0
        } else {
            total_free_bytes as f64 / total_pooled as f64
        };

        let hit_rate = if total_allocs == 0 {
            0.0
        } else {
            total_hits as f64 / total_allocs as f64
        };

        BufferPoolSizeClassStats {
            per_class,
            oversized_allocs: self.oversized_allocs,
            total_free_bytes,
            total_used_bytes,
            fragmentation_ratio,
            hit_rate,
        }
    }

    /// Reset all statistics and free lists. Does not release Metal buffers
    /// (those are managed externally).
    pub fn reset(&mut self) {
        for state in &mut self.classes {
            *state = ClassState::default();
        }
        self.oversized_allocs = 0;
    }
}

impl Default for SizeClassAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a successful allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocResult {
    /// Size class index (0..NUM_SIZE_CLASSES).
    pub class: usize,
    /// Actual bytes allocated (rounded up to size class boundary).
    pub alloc_bytes: usize,
    /// Whether a buffer was reused from the free list.
    pub reused: bool,
}

#[cfg(test)]
#[path = "buffer_pool_size_class_tests.rs"]
mod tests;
