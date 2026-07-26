// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Arena allocation statistics and generation query functions.
//!
//! Extracted from `arena.rs` to keep the parent under the 450-line limit.

use std::cell::Cell;

use super::pool::PoolStats;
use super::scope::{ARENA_HIT_COUNT, ARENA_MISS_COUNT, DEFAULT_ARENA, LAST_ALLOC_GEN};

/// Arena allocation statistics since the last [`reset_arena_stats`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ArenaStats {
    /// Number of allocations served from the arena.
    pub hits: usize,
    /// Number of allocations that fell back to standalone buffer or pool.
    pub misses: usize,
    /// Full buffer pool statistics snapshot. Replaces the former individual
    /// `pool_hits`/`pool_size`/`pool_retained_bytes` fields with complete
    /// pool diagnostics (acquisitions, hits, misses, discards, bytes, buffers).
    /// Part of #3079 D3.
    pub pool: PoolStats,
    /// Number of auto-grow events in the current arena generation.
    /// Part of #4289.
    pub growth_count: usize,
    /// Total auto-grow events since the default arena was created.
    /// Part of #4289.
    pub total_growth_count: usize,
    /// Number of overflow events in the current arena generation.
    ///
    /// An overflow is an allocation that exceeded the current slab's remaining
    /// capacity and triggered slab growth (with auto-grow) or an error (without).
    /// Part of #4289.
    pub overflow_count: usize,
    /// Total overflow events since the default arena was created.
    /// Part of #4289.
    pub total_overflow_count: usize,
    /// Cumulative bytes allocated via overflow in the current generation.
    /// Part of #4289.
    pub overflow_bytes: usize,
    /// Total bytes allocated via overflow since the default arena was created.
    /// Part of #4289.
    pub total_overflow_bytes: usize,
}

impl ArenaStats {
    /// Arena hit rate as a fraction in [0.0, 1.0]. Returns 0.0 if no allocations.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    /// Number of misses that required a fresh Metal buffer allocation.
    ///
    /// `fresh_allocs = misses - pool.hits`. These are the allocations that
    /// actually increase RSS — pool hits reuse existing Metal VM mappings.
    /// On a warm-cache second synthesis call, `fresh_allocs() == 0` means
    /// the pool captures all buffer reuse. Part of #3079 D4 measurement.
    #[must_use]
    pub fn fresh_allocs(&self) -> usize {
        self.misses.saturating_sub(self.pool.hits)
    }
}

/// Default arena capacity in bytes (mirrors `DEFAULT_ARENA_CAPACITY` in arena_scope.rs).
///
/// Exposed for diagnostic display alongside `default_arena_peak_bytes()`.
/// If `peak >= capacity`, the arena overflowed and D4 right-sizing is needed.
#[must_use]
pub fn arena_capacity() -> usize {
    super::scope::DEFAULT_ARENA_CAPACITY
}

/// Read cumulative arena statistics for the current thread.
pub fn arena_stats() -> ArenaStats {
    ArenaStats {
        hits: ARENA_HIT_COUNT.with(Cell::get),
        misses: ARENA_MISS_COUNT.with(Cell::get),
        pool: super::pool::pool_stats(),
        growth_count: super::scope::default_arena_growth_count(),
        total_growth_count: super::scope::default_arena_total_growth_count(),
        overflow_count: super::scope::default_arena_overflow_count(),
        total_overflow_count: super::scope::default_arena_total_overflow_count(),
        overflow_bytes: super::scope::default_arena_overflow_bytes(),
        total_overflow_bytes: super::scope::default_arena_total_overflow_bytes(),
    }
}

/// Reset arena statistics counters to zero.
pub fn reset_arena_stats() {
    ARENA_HIT_COUNT.with(|c| c.set(0));
    ARENA_MISS_COUNT.with(|c| c.set(0));
    super::pool::reset_pool_stats();
}

/// Arena generation from the most recent [`super::arena_alloc_or_create`] call.
///
/// Returns `Some(gen)` if the last allocation came from an arena, or `None`
/// if it fell back to a fresh `create_buffer_zeroed`. Used by dispatch
/// boundaries to stamp `MetalTensorData` with arena generation info.
pub(crate) fn last_alloc_generation() -> Option<u64> {
    LAST_ALLOC_GEN.with(Cell::get)
}

/// Current generation of the thread-local default arena, if it exists.
///
/// Returns `None` if the default arena has not been initialized on this
/// thread. Used by stale-read detection in `gpu_to_cpu` (#2328).
pub(crate) fn default_arena_generation() -> Option<u64> {
    DEFAULT_ARENA.with(|cell| {
        cell.borrow()
            .as_ref()
            .map(super::ActivationArena::generation)
    })
}

/// Current used bytes of the thread-local default arena, if it exists.
///
/// Returns `None` if the default arena has not been initialized. Used
/// alongside [`default_arena_generation`] for fine-grained stale detection.
pub(crate) fn default_arena_used_bytes() -> Option<usize> {
    DEFAULT_ARENA.with(|cell| {
        cell.borrow()
            .as_ref()
            .map(super::ActivationArena::used_bytes)
    })
}

/// Peak bytes used by the thread-local default arena, if it exists.
///
/// Returns `None` if the default arena has not been initialized.
/// Reports the *default* (64 MB) arena's peak, not any explicit
/// `with_arena` arena. Part of #2914.
pub fn default_arena_peak_bytes() -> Option<usize> {
    DEFAULT_ARENA.with(|cell| {
        cell.borrow()
            .as_ref()
            .map(super::ActivationArena::peak_bytes)
    })
}

#[cfg(test)]
#[path = "arena_stats_tests.rs"]
mod tests;
