// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Metal buffer memory profiler for tracking GPU memory usage.
//!
//! Provides thread-local tracking of GPU buffer allocations and deallocations,
//! with snapshots for total/live/peak memory and per-category breakdowns.
//!
//! # Usage
//!
//! ```rust,ignore
//! use nn_metal::memory_profiler::{GpuMemoryProfiler, BufferCategory};
//!
//! GpuMemoryProfiler::reset();
//! GpuMemoryProfiler::record_allocation(4096, "layer_norm_weights", BufferCategory::Weights);
//! GpuMemoryProfiler::record_allocation(1024, "activation_0", BufferCategory::Activations);
//! GpuMemoryProfiler::record_deallocation(1024);
//!
//! let snap = GpuMemoryProfiler::snapshot();
//! assert_eq!(snap.total_allocated, 5120);
//! assert_eq!(snap.total_live, 4096);
//! assert_eq!(snap.peak_live, 5120);
//!
//! let breakdown = GpuMemoryProfiler::breakdown();
//! assert_eq!(breakdown.weights, 4096);
//! ```
//!
//! Part of #4353.

use std::cell::RefCell;
use std::fmt;

/// Category of a GPU buffer allocation for memory attribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferCategory {
    /// Model weight buffers (persisted across forward passes).
    Weights,
    /// Intermediate activation buffers (transient per forward pass).
    Activations,
    /// Scratch/temporary buffers (workspace for kernels).
    Scratch,
    /// Uncategorized allocations.
    Other,
}

impl fmt::Display for BufferCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Weights => write!(f, "weights"),
            Self::Activations => write!(f, "activations"),
            Self::Scratch => write!(f, "scratch"),
            Self::Other => write!(f, "other"),
        }
    }
}

/// A point-in-time snapshot of GPU memory usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuMemorySnapshot {
    /// Cumulative bytes allocated since last reset.
    pub total_allocated: usize,
    /// Currently live (allocated minus deallocated) bytes.
    pub total_live: usize,
    /// Peak live bytes observed since last reset.
    pub peak_live: usize,
    /// Number of currently live buffer allocations.
    pub buffer_count: usize,
    /// Size of the largest single allocation since last reset.
    pub largest_buffer: usize,
}

impl fmt::Display for GpuMemorySnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GPU Memory Snapshot")?;
        writeln!(f, "  total_allocated: {:>10.2} MB", self.total_allocated as f64 / (1024.0 * 1024.0))?;
        writeln!(f, "  total_live:      {:>10.2} MB", self.total_live as f64 / (1024.0 * 1024.0))?;
        writeln!(f, "  peak_live:       {:>10.2} MB", self.peak_live as f64 / (1024.0 * 1024.0))?;
        writeln!(f, "  buffer_count:    {:>10}", self.buffer_count)?;
        write!(f, "  largest_buffer:  {:>10.2} MB", self.largest_buffer as f64 / (1024.0 * 1024.0))
    }
}

/// Per-category memory breakdown for attribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MemoryBreakdownByCategory {
    /// Live bytes attributed to weight buffers.
    pub weights: usize,
    /// Live bytes attributed to activation buffers.
    pub activations: usize,
    /// Live bytes attributed to scratch/workspace buffers.
    pub scratch: usize,
    /// Live bytes attributed to uncategorized buffers.
    pub other: usize,
}

impl MemoryBreakdownByCategory {
    /// Total live bytes across all categories.
    #[must_use]
    pub fn total(&self) -> usize {
        self.weights
            .saturating_add(self.activations)
            .saturating_add(self.scratch)
            .saturating_add(self.other)
    }
}

impl fmt::Display for MemoryBreakdownByCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GPU Memory Breakdown")?;
        writeln!(f, "  weights:     {:>10.2} MB", self.weights as f64 / (1024.0 * 1024.0))?;
        writeln!(f, "  activations: {:>10.2} MB", self.activations as f64 / (1024.0 * 1024.0))?;
        writeln!(f, "  scratch:     {:>10.2} MB", self.scratch as f64 / (1024.0 * 1024.0))?;
        writeln!(f, "  other:       {:>10.2} MB", self.other as f64 / (1024.0 * 1024.0))?;
        write!(f, "  total:       {:>10.2} MB", self.total() as f64 / (1024.0 * 1024.0))
    }
}

/// A single allocation record tracked by the profiler.
#[derive(Debug, Clone)]
struct AllocationRecord {
    size: usize,
    label: String,
    category: BufferCategory,
}

/// Thread-local state for the GPU memory profiler.
#[derive(Debug)]
struct ProfilerState {
    /// All currently live allocations.
    live_allocations: Vec<AllocationRecord>,
    /// Cumulative bytes allocated since last reset.
    total_allocated: usize,
    /// Peak live bytes since last reset.
    peak_live: usize,
    /// Current live bytes.
    current_live: usize,
    /// Largest single allocation since last reset.
    largest_buffer: usize,
}

impl ProfilerState {
    fn new() -> Self {
        Self {
            live_allocations: Vec::new(),
            total_allocated: 0,
            peak_live: 0,
            current_live: 0,
            largest_buffer: 0,
        }
    }
}

impl Default for ProfilerState {
    fn default() -> Self {
        Self::new()
    }
}

thread_local! {
    static PROFILER_STATE: RefCell<ProfilerState> = RefCell::new(ProfilerState::new());
}

/// Thread-local GPU memory profiler for tracking Metal buffer allocations.
///
/// All methods are static and operate on thread-local state, matching the
/// pattern used by [`dispatch_stats`](crate::dispatch_stats) and
/// [`arena_stats`](crate::arena::arena_stats).
///
/// The profiler tracks allocations and deallocations recorded via
/// [`record_allocation`](Self::record_allocation) and
/// [`record_deallocation`](Self::record_deallocation). It does not
/// automatically instrument Metal buffer creation -- callers must
/// explicitly record events at allocation sites.
pub struct GpuMemoryProfiler;

impl GpuMemoryProfiler {
    /// Record a GPU buffer allocation.
    ///
    /// Tracks the allocation by size, label, and category. Updates
    /// cumulative totals, peak live bytes, and largest buffer metrics.
    pub fn record_allocation(size: usize, label: &str, category: BufferCategory) {
        PROFILER_STATE.with(|state| {
            let mut s = state.borrow_mut();
            s.live_allocations.push(AllocationRecord {
                size,
                label: label.to_string(),
                category,
            });
            s.total_allocated = s.total_allocated.saturating_add(size);
            s.current_live = s.current_live.saturating_add(size);
            if s.current_live > s.peak_live {
                s.peak_live = s.current_live;
            }
            if size > s.largest_buffer {
                s.largest_buffer = size;
            }
        });
    }

    /// Record a GPU buffer deallocation.
    ///
    /// Removes the first live allocation matching the given size (FIFO order).
    /// If no matching allocation is found, the deallocation is still subtracted
    /// from `current_live` (clamped to zero) to handle untracked buffers
    /// gracefully.
    pub fn record_deallocation(size: usize) {
        PROFILER_STATE.with(|state| {
            let mut s = state.borrow_mut();
            // Remove the first matching allocation by size.
            if let Some(pos) = s.live_allocations.iter().position(|a| a.size == size) {
                s.live_allocations.swap_remove(pos);
            }
            s.current_live = s.current_live.saturating_sub(size);
        });
    }

    /// Take a point-in-time snapshot of GPU memory usage.
    #[must_use]
    pub fn snapshot() -> GpuMemorySnapshot {
        PROFILER_STATE.with(|state| {
            let s = state.borrow();
            GpuMemorySnapshot {
                total_allocated: s.total_allocated,
                total_live: s.current_live,
                peak_live: s.peak_live,
                buffer_count: s.live_allocations.len(),
                largest_buffer: s.largest_buffer,
            }
        })
    }

    /// Get a per-category breakdown of currently live memory.
    #[must_use]
    pub fn breakdown() -> MemoryBreakdownByCategory {
        PROFILER_STATE.with(|state| {
            let s = state.borrow();
            let mut bd = MemoryBreakdownByCategory::default();
            for alloc in &s.live_allocations {
                match alloc.category {
                    BufferCategory::Weights => {
                        bd.weights = bd.weights.saturating_add(alloc.size);
                    }
                    BufferCategory::Activations => {
                        bd.activations = bd.activations.saturating_add(alloc.size);
                    }
                    BufferCategory::Scratch => {
                        bd.scratch = bd.scratch.saturating_add(alloc.size);
                    }
                    BufferCategory::Other => {
                        bd.other = bd.other.saturating_add(alloc.size);
                    }
                }
            }
            bd
        })
    }

    /// Reset all profiler state to zero.
    ///
    /// Clears all tracked allocations and resets cumulative counters.
    /// Call before a profiling region to isolate measurements.
    pub fn reset() {
        PROFILER_STATE.with(|state| {
            *state.borrow_mut() = ProfilerState::new();
        });
    }

    /// Number of currently live allocations.
    #[must_use]
    pub fn live_allocation_count() -> usize {
        PROFILER_STATE.with(|state| state.borrow().live_allocations.len())
    }

    /// Return labels and sizes of all currently live allocations.
    ///
    /// Useful for diagnostic output and debugging memory leaks.
    /// Sorted by size descending (largest first).
    #[must_use]
    pub fn live_allocations() -> Vec<(String, usize, BufferCategory)> {
        PROFILER_STATE.with(|state| {
            let s = state.borrow();
            let mut entries: Vec<(String, usize, BufferCategory)> = s
                .live_allocations
                .iter()
                .map(|a| (a.label.clone(), a.size, a.category))
                .collect();
            entries.sort_by_key(|x| std::cmp::Reverse(x.1));
            entries
        })
    }
}

#[cfg(test)]
#[path = "memory_profiler_tests.rs"]
mod tests;
