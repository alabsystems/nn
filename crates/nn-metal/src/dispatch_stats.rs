// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU dispatch statistics and cumulative counters.
//!
//! Extracted from `gpu_scope.rs` to stay within the 450-line limit (#1815).
//! The four cumulative counters are thread-local and incremented by
//! [`gpu_scope`](crate::gpu_scope) batch operations.

use std::cell::Cell;
use std::sync::OnceLock;

thread_local! {
    /// Cumulative GPU dispatch encodings since last [`reset_counters`].
    pub(crate) static TOTAL_ENCODINGS: Cell<usize> = const { Cell::new(0) };

    /// Cumulative buffer-planner blit copies since last [`reset_counters`].
    /// Separate from `TOTAL_ENCODINGS` (compute-only). Total Metal encodings
    /// = `TOTAL_ENCODINGS + TOTAL_BLITS`. See #1815.
    pub(crate) static TOTAL_BLITS: Cell<usize> = const { Cell::new(0) };

    /// Cumulative number of `commit_and_wait` calls since last counter reset.
    /// Incremented in [`gpu_scope::flush`](crate::gpu_scope::flush) and auto-flush.
    pub(crate) static TOTAL_FLUSHES: Cell<usize> = const { Cell::new(0) };

    /// Cumulative number of non-blocking submit calls since last counter reset.
    /// Incremented in [`gpu_scope::submit`](crate::gpu_scope::submit).
    pub(crate) static TOTAL_SUBMITS: Cell<usize> = const { Cell::new(0) };

    /// Cumulative blits eliminated by the planned-buffer redirect + skip
    /// normalization optimization (#4264). Incremented when a Dispatch or
    /// NativeOp step writes directly into the planned buffer region,
    /// skipping the `relocate_to_planned_buffer` blit.
    pub(crate) static TOTAL_BLITS_ELIMINATED: Cell<usize> = const { Cell::new(0) };
}

/// Increment a counter and optionally trace the event to stderr when
/// `NN_FLUSH_TRACE=1`. Returns the new counter value.
pub(crate) fn record_gpu_event(counter: &Cell<usize>, kind: &str, encodings: usize) -> usize {
    let n = counter.get() + 1;
    counter.set(n);
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if *ENABLED.get_or_init(|| std::env::var("NN_FLUSH_TRACE").as_deref() == Ok("1")) {
        let bt = std::backtrace::Backtrace::force_capture();
        eprintln!("[NN_FLUSH_TRACE] {kind} #{n} ({encodings} encodings)\n{bt}");
    }
    n
}

/// Dispatch statistics since the last [`reset_counters`] call.
///
/// Used for benchmarking lazy batch effectiveness (#2009 AC2).
/// `compute_encodings` counts compute dispatches only (not blits).
/// `blits` counts buffer-planner blit copies only.
/// Total Metal command encodings = `compute_encodings + blits`.
/// `flushes` is the total number of `commit_and_wait` calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DispatchStats {
    /// Compute dispatch encodings only (excludes blits).
    pub compute_encodings: usize,
    /// Buffer-planner blit copies (excludes compute dispatches).
    /// Total Metal encodings = `compute_encodings + blits`.
    pub blits: usize,
    /// Total `commit_and_wait` calls (flushes).
    pub flushes: usize,
    /// Total non-blocking submit calls.
    pub submits: usize,
    /// Blits eliminated by the planned-buffer redirect + skip normalization
    /// optimization (#4264). Each count represents one blit that would have
    /// been required without the optimization.
    pub blits_eliminated: usize,
    /// Arena allocation statistics (hits + misses).
    pub arena: crate::arena::ArenaStats,
}

/// Read the cumulative dispatch statistics for the current thread.
///
/// Returns the number of GPU dispatch encodings and flush (commit_and_wait)
/// calls since the last [`reset_counters`] call, plus arena hit/miss stats.
pub fn dispatch_stats() -> DispatchStats {
    DispatchStats {
        compute_encodings: TOTAL_ENCODINGS.with(Cell::get),
        blits: TOTAL_BLITS.with(Cell::get),
        flushes: TOTAL_FLUSHES.with(Cell::get),
        submits: TOTAL_SUBMITS.with(Cell::get),
        blits_eliminated: TOTAL_BLITS_ELIMINATED.with(Cell::get),
        arena: crate::arena::arena_stats(),
    }
}

/// Reset the dispatch counters to zero.
///
/// Call before a benchmark region to measure dispatch reduction within
/// that region only.
pub fn reset_counters() {
    TOTAL_ENCODINGS.with(|c| c.set(0));
    TOTAL_BLITS.with(|c| c.set(0));
    TOTAL_FLUSHES.with(|c| c.set(0));
    TOTAL_SUBMITS.with(|c| c.set(0));
    TOTAL_BLITS_ELIMINATED.with(|c| c.set(0));
    crate::arena::reset_arena_stats();
}

#[cfg(test)]
#[path = "dispatch_stats_tests.rs"]
mod tests;
