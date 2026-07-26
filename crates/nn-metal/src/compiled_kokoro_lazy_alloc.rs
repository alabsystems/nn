// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Lazy Metal buffer allocation with size-class pooling for Kokoro inference.
//!
//! [`LazyBufferPool`] allocates GPU buffers just-in-time and returns freed
//! buffers to a size-class pool for reuse. Combined with [`LivenessAnalysis`],
//! this reduces peak memory by reusing buffers whose producers are dead.
//!
//! 7 power-of-4 size classes from 4 KB to 16 MB. Requests > 16 MB are
//! oversized (not pooled). Zero-byte requests return a sentinel.
//!
//! This module provides allocation policy only — no Metal buffer handles.
//! The caller maps [`LazyBufferHandle`] values to real GPU buffers.
//!
//! Part of #4264 (RTF optimization).

use std::fmt;

/// Size-class boundaries (bytes). Powers of 4 from 4 KB to 16 MB (7 classes).
const SIZE_CLASSES: [usize; 7] = [
    4 * 1024,          // 0:  4 KB
    16 * 1024,         // 1: 16 KB
    64 * 1024,         // 2: 64 KB
    256 * 1024,        // 3: 256 KB
    1024 * 1024,       // 4:  1 MB
    4 * 1024 * 1024,   // 5:  4 MB
    16 * 1024 * 1024,  // 6: 16 MB
];

/// Number of size classes.
const NUM_CLASSES: usize = SIZE_CLASSES.len();

/// Maximum buffers retained per size class in the free pool.
const MAX_FREE_PER_CLASS: usize = 32;

/// Handle returned by [`LazyBufferPool::alloc`]. Must be passed to
/// [`LazyBufferPool::free`] when the buffer is no longer needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LazyBufferHandle {
    /// Size class (0..NUM_CLASSES), or `usize::MAX` for oversized.
    class: usize,
    /// Actual allocation size in bytes.
    alloc_bytes: usize,
    /// Unique allocation id for tracking.
    id: u64,
}

impl LazyBufferHandle {
    /// Size class index, or `usize::MAX` for oversized allocations.
    #[must_use]
    pub fn class(&self) -> usize {
        self.class
    }

    /// Actual allocated bytes (rounded up to size-class boundary).
    #[must_use]
    pub fn alloc_bytes(&self) -> usize {
        self.alloc_bytes
    }

    /// Unique allocation id.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }
}

/// Statistics snapshot from [`LazyBufferPool`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LazyPoolStats {
    /// Total allocations served.
    pub total_allocs: usize,
    /// Allocations served from the free pool (reuse).
    pub pool_hits: usize,
    /// Allocations that required new buffers (no free entry).
    pub pool_misses: usize,
    /// Buffers returned to the pool.
    pub total_frees: usize,
    /// Oversized allocations (> 16 MB, not pooled).
    pub oversized_allocs: usize,
    /// Zero-byte allocation requests.
    pub zero_allocs: usize,
    /// Peak number of bytes simultaneously allocated (high-water mark).
    pub peak_bytes: usize,
    /// Current bytes allocated (not yet freed).
    pub current_bytes: usize,
    /// Current bytes retained on free lists.
    pub free_pool_bytes: usize,
    /// Current buffer count on free lists.
    pub free_pool_count: usize,
    /// Per-class free counts.
    pub per_class_free: [usize; NUM_CLASSES],
}

impl LazyPoolStats {
    /// Pool reuse rate as a fraction in [0.0, 1.0].
    #[must_use]
    pub fn reuse_rate(&self) -> f64 {
        if self.total_allocs == 0 {
            return 0.0;
        }
        self.pool_hits as f64 / self.total_allocs as f64
    }

    /// Peak memory in megabytes.
    #[must_use]
    pub fn peak_mb(&self) -> f64 {
        self.peak_bytes as f64 / (1024.0 * 1024.0)
    }
}

impl fmt::Display for LazyPoolStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "LazyBufferPool Stats")?;
        writeln!(
            f,
            "  allocs: {} total ({} hits, {} misses, {} oversized, {} zero)",
            self.total_allocs, self.pool_hits, self.pool_misses,
            self.oversized_allocs, self.zero_allocs,
        )?;
        writeln!(f, "  reuse rate: {:.1}%", self.reuse_rate() * 100.0)?;
        writeln!(
            f,
            "  peak memory: {:.2} MB, current: {:.2} MB",
            self.peak_mb(),
            self.current_bytes as f64 / (1024.0 * 1024.0),
        )?;
        write!(
            f,
            "  free pool: {} buffers ({:.2} MB)",
            self.free_pool_count,
            self.free_pool_bytes as f64 / (1024.0 * 1024.0),
        )
    }
}

/// Lazy GPU buffer allocator with size-class pooling.
///
/// Tracks logical allocations and frees without holding Metal buffers.
/// The caller maps handles to real GPU buffers. Part of #4264.
#[derive(Debug)]
pub struct LazyBufferPool {
    /// Per-class free list count.
    free_counts: [usize; NUM_CLASSES],
    /// Stats counters.
    total_allocs: usize,
    pool_hits: usize,
    pool_misses: usize,
    total_frees: usize,
    oversized_allocs: usize,
    zero_allocs: usize,
    /// Current bytes in use.
    current_bytes: usize,
    /// Peak bytes in use (high-water mark).
    peak_bytes: usize,
    /// Monotonically increasing allocation id.
    next_id: u64,
}

impl LazyBufferPool {
    /// Create a new empty pool.
    #[must_use]
    pub fn new() -> Self {
        Self {
            free_counts: [0; NUM_CLASSES],
            total_allocs: 0,
            pool_hits: 0,
            pool_misses: 0,
            total_frees: 0,
            oversized_allocs: 0,
            zero_allocs: 0,
            current_bytes: 0,
            peak_bytes: 0,
            next_id: 0,
        }
    }

    /// Determine the size class for a given byte count.
    ///
    /// Returns `Some((class_index, class_bytes))` if the request fits,
    /// or `None` if it exceeds the largest class (16 MB).
    #[must_use]
    pub fn size_class_for(bytes: usize) -> Option<(usize, usize)> {
        if bytes == 0 {
            return None;
        }
        for (i, &boundary) in SIZE_CLASSES.iter().enumerate() {
            if bytes <= boundary {
                return Some((i, boundary));
            }
        }
        None
    }

    /// Allocate a buffer of at least `requested_bytes`.
    /// Zero-byte requests return a sentinel. Reuses from free pool when available.
    pub fn alloc(&mut self, requested_bytes: usize) -> LazyBufferHandle {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.total_allocs += 1;

        if requested_bytes == 0 {
            self.zero_allocs += 1;
            return LazyBufferHandle {
                class: usize::MAX,
                alloc_bytes: 0,
                id,
            };
        }

        match Self::size_class_for(requested_bytes) {
            Some((class, class_bytes)) => {
                if self.free_counts[class] > 0 {
                    self.free_counts[class] -= 1;
                    self.pool_hits += 1;
                } else {
                    self.pool_misses += 1;
                }
                self.current_bytes = self.current_bytes.saturating_add(class_bytes);
                if self.current_bytes > self.peak_bytes {
                    self.peak_bytes = self.current_bytes;
                }
                LazyBufferHandle {
                    class,
                    alloc_bytes: class_bytes,
                    id,
                }
            }
            None => {
                self.oversized_allocs += 1;
                self.pool_misses += 1;
                self.current_bytes = self.current_bytes.saturating_add(requested_bytes);
                if self.current_bytes > self.peak_bytes {
                    self.peak_bytes = self.current_bytes;
                }
                LazyBufferHandle {
                    class: usize::MAX,
                    alloc_bytes: requested_bytes,
                    id,
                }
            }
        }
    }

    /// Return a buffer to the pool. Returns `true` if accepted into the free
    /// pool, `false` if oversized, zero-byte, or free list full.
    pub fn free(&mut self, handle: LazyBufferHandle) -> bool {
        self.total_frees += 1;
        self.current_bytes = self.current_bytes.saturating_sub(handle.alloc_bytes);

        if handle.alloc_bytes == 0 || handle.class == usize::MAX {
            return false;
        }

        let class = handle.class;
        if class >= NUM_CLASSES {
            return false;
        }

        if self.free_counts[class] >= MAX_FREE_PER_CLASS {
            return false;
        }

        self.free_counts[class] += 1;
        true
    }

    /// Snapshot current statistics.
    #[must_use]
    pub fn stats(&self) -> LazyPoolStats {
        let mut free_pool_bytes = 0usize;
        let mut free_pool_count = 0usize;
        for (i, &count) in self.free_counts.iter().enumerate() {
            free_pool_bytes = free_pool_bytes
                .saturating_add(count.saturating_mul(SIZE_CLASSES[i]));
            free_pool_count += count;
        }

        LazyPoolStats {
            total_allocs: self.total_allocs,
            pool_hits: self.pool_hits,
            pool_misses: self.pool_misses,
            total_frees: self.total_frees,
            oversized_allocs: self.oversized_allocs,
            zero_allocs: self.zero_allocs,
            peak_bytes: self.peak_bytes,
            current_bytes: self.current_bytes,
            free_pool_bytes,
            free_pool_count,
            per_class_free: self.free_counts,
        }
    }

    /// Reset all statistics and free lists.
    pub fn reset(&mut self) {
        self.free_counts = [0; NUM_CLASSES];
        self.total_allocs = 0;
        self.pool_hits = 0;
        self.pool_misses = 0;
        self.total_frees = 0;
        self.oversized_allocs = 0;
        self.zero_allocs = 0;
        self.current_bytes = 0;
        self.peak_bytes = 0;
        // Preserve next_id to avoid handle collisions after reset.
    }

    /// Peak bytes allocated simultaneously.
    #[must_use]
    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes
    }

    /// Current bytes allocated (not yet freed).
    #[must_use]
    pub fn current_bytes(&self) -> usize {
        self.current_bytes
    }
}

impl Default for LazyBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-buffer liveness information from a dispatch plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferLiveness {
    /// Step index where the buffer is produced (allocated).
    pub produced_at: usize,
    /// Last step index that reads this buffer. After this step, the buffer
    /// can be freed. `None` means the buffer is never read (dead output).
    pub last_use: Option<usize>,
    /// Required buffer size in bytes.
    pub size_bytes: usize,
}

/// Result of liveness analysis on a dispatch plan.
#[derive(Debug, Clone)]
pub struct LivenessAnalysis {
    /// Per-buffer liveness, indexed by buffer id (step index in the plan).
    pub buffers: Vec<BufferLiveness>,
    /// Peak number of simultaneously live buffers.
    pub peak_live_count: usize,
    /// Peak bytes simultaneously live (sum of live buffer sizes).
    pub peak_live_bytes: usize,
    /// Total bytes if all buffers were allocated upfront (no reuse).
    pub total_bytes_no_reuse: usize,
}

impl LivenessAnalysis {
    /// Memory savings ratio from lazy allocation vs. upfront allocation.
    /// Returns a value in [0.0, 1.0] where 1.0 means 100% savings.
    #[must_use]
    pub fn savings_ratio(&self) -> f64 {
        if self.total_bytes_no_reuse == 0 {
            return 0.0;
        }
        let saved = self.total_bytes_no_reuse.saturating_sub(self.peak_live_bytes);
        saved as f64 / self.total_bytes_no_reuse as f64
    }
}

/// Dispatch step descriptor for liveness analysis.
#[derive(Debug, Clone)]
pub struct DispatchStepDesc {
    /// Indices of input buffers consumed by this step.
    pub input_indices: Vec<usize>,
    /// Output buffer size in bytes.
    pub output_bytes: usize,
}

/// Compute liveness analysis for a sequence of dispatch steps.
/// Step `i` produces buffer `i`. Returns per-buffer last-use and peak memory.
pub fn compute_liveness(steps: &[DispatchStepDesc]) -> LivenessAnalysis {
    let n = steps.len();
    if n == 0 {
        return LivenessAnalysis {
            buffers: Vec::new(),
            peak_live_count: 0,
            peak_live_bytes: 0,
            total_bytes_no_reuse: 0,
        };
    }

    let mut buffers: Vec<BufferLiveness> = steps
        .iter()
        .enumerate()
        .map(|(i, step)| BufferLiveness {
            produced_at: i,
            last_use: None,
            size_bytes: step.output_bytes,
        })
        .collect();

    // Forward scan: for each step, update last_use for its inputs.
    for (step_idx, step) in steps.iter().enumerate() {
        for &input_idx in &step.input_indices {
            if input_idx < n {
                match buffers[input_idx].last_use {
                    None => buffers[input_idx].last_use = Some(step_idx),
                    Some(prev) if step_idx > prev => {
                        buffers[input_idx].last_use = Some(step_idx);
                    }
                    _ => {}
                }
            }
        }
    }

    let total_bytes_no_reuse: usize = buffers.iter().map(|b| b.size_bytes).sum();

    // Forward sweep: track which buffers are live at each step.
    let mut peak_live_count = 0usize;
    let mut peak_live_bytes = 0usize;
    let mut live = vec![false; n];
    let mut live_count = 0usize;
    let mut live_bytes = 0usize;

    for step_idx in 0..n {
        live[step_idx] = true;
        live_count += 1;
        live_bytes = live_bytes.saturating_add(buffers[step_idx].size_bytes);

        if live_count > peak_live_count {
            peak_live_count = live_count;
        }
        if live_bytes > peak_live_bytes {
            peak_live_bytes = live_bytes;
        }

        // Free: any buffer whose last_use is this step is now dead.
        for buf_idx in 0..=step_idx {
            if live[buf_idx] {
                if let Some(last) = buffers[buf_idx].last_use {
                    if last <= step_idx {
                        live[buf_idx] = false;
                        live_count -= 1;
                        live_bytes = live_bytes.saturating_sub(buffers[buf_idx].size_bytes);
                    }
                }
            }
        }
    }

    LivenessAnalysis {
        buffers,
        peak_live_count,
        peak_live_bytes,
        total_bytes_no_reuse,
    }
}

/// Simulate lazy allocation with pool reuse. Runs [`compute_liveness`] then
/// allocates/frees through a [`LazyBufferPool`].
pub fn simulate_lazy_alloc(steps: &[DispatchStepDesc]) -> (LivenessAnalysis, LazyPoolStats) {
    let liveness = compute_liveness(steps);
    let n = steps.len();

    if n == 0 {
        let pool = LazyBufferPool::new();
        return (liveness, pool.stats());
    }

    let mut pool = LazyBufferPool::new();
    let mut handles: Vec<Option<LazyBufferHandle>> = vec![None; n];

    for step_idx in 0..n {
        let handle = pool.alloc(steps[step_idx].output_bytes);
        handles[step_idx] = Some(handle);

        for buf_idx in 0..=step_idx {
            if let Some(h) = handles[buf_idx] {
                if let Some(last) = liveness.buffers[buf_idx].last_use {
                    if last <= step_idx {
                        pool.free(h);
                        handles[buf_idx] = None;
                    }
                }
            }
        }
    }

    // Free remaining live buffers (pipeline outputs).
    for handle in handles.into_iter().flatten() {
        pool.free(handle);
    }

    let stats = pool.stats();
    (liveness, stats)
}

#[cfg(test)]
#[path = "compiled_kokoro_lazy_alloc_tests.rs"]
mod tests;
