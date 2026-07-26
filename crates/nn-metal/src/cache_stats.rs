// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cache performance statistics for the 3-level Metal dispatch cache.
//!
//! The Metal dispatch infrastructure uses a 3-level cache hierarchy:
//!
//! ```text
//! L1: KernelDefCache   → TensorKernelDef (IR)
//! L2: MslCodegenCache  → (plan, output_id, expanded, msl_string)
//! L3: PipelineCache    → ComputePipeline (compiled Metal)
//! ```
//!
//! [`CacheStats`] provides atomic counters for hit/miss tracking across all
//! three levels, plus compile-time accumulation and dispatch counting. All
//! counters use [`AtomicU64`] with `Relaxed` ordering for minimal overhead
//! on the hot path — exact counts are not required for diagnostics.
//!
//! # Usage
//!
//! ```no_run
//! use nn_metal::CacheStats;
//!
//! // Reset before a benchmark region
//! CacheStats::global().reset();
//!
//! // ... run inference ...
//!
//! let stats = CacheStats::global().snapshot();
//! println!("{}", stats.summary());
//! assert!(stats.hit_rate() > 0.9, "cache hit rate below 90%");
//! ```

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Relaxed ordering — sufficient for diagnostic counters. Exact cross-thread
/// visibility is not required; eventual consistency is fine.
const ORD: Ordering = Ordering::Relaxed;

/// Thread-safe atomic cache performance counters.
///
/// All fields are `AtomicU64` with `Relaxed` ordering for minimal hot-path
/// overhead. Use [`snapshot`](Self::snapshot) to read a consistent point-in-time
/// view, or [`global`](Self::global) to access the process-wide singleton.
pub struct CacheStats {
    kernel_cache_hits: AtomicU64,
    kernel_cache_misses: AtomicU64,
    msl_cache_hits: AtomicU64,
    msl_cache_misses: AtomicU64,
    pipeline_cache_hits: AtomicU64,
    pipeline_cache_misses: AtomicU64,
    total_dispatches: AtomicU64,
    total_compile_time_us: AtomicU64,
}

impl CacheStats {
    /// Create a new zeroed stats instance.
    #[must_use]
    const fn new() -> Self {
        Self {
            kernel_cache_hits: AtomicU64::new(0),
            kernel_cache_misses: AtomicU64::new(0),
            msl_cache_hits: AtomicU64::new(0),
            msl_cache_misses: AtomicU64::new(0),
            pipeline_cache_hits: AtomicU64::new(0),
            pipeline_cache_misses: AtomicU64::new(0),
            total_dispatches: AtomicU64::new(0),
            total_compile_time_us: AtomicU64::new(0),
        }
    }

    /// Access the process-wide singleton.
    ///
    /// All cache layers record into this shared instance. Use [`snapshot`]
    /// to read a consistent view.
    pub fn global() -> &'static Self {
        static INSTANCE: OnceLock<CacheStats> = OnceLock::new();
        INSTANCE.get_or_init(Self::new)
    }

    // -----------------------------------------------------------------
    // Recording (called from cache internals)
    // -----------------------------------------------------------------

    /// Record a kernel def cache (L1) hit.
    pub fn record_kernel_hit(&self) {
        self.kernel_cache_hits.fetch_add(1, ORD);
    }

    /// Record a kernel def cache (L1) miss.
    pub fn record_kernel_miss(&self) {
        self.kernel_cache_misses.fetch_add(1, ORD);
    }

    /// Record an MSL codegen cache (L2) hit.
    pub fn record_msl_hit(&self) {
        self.msl_cache_hits.fetch_add(1, ORD);
    }

    /// Record an MSL codegen cache (L2) miss.
    pub fn record_msl_miss(&self) {
        self.msl_cache_misses.fetch_add(1, ORD);
    }

    /// Record a pipeline cache (L3) hit.
    pub fn record_pipeline_hit(&self) {
        self.pipeline_cache_hits.fetch_add(1, ORD);
    }

    /// Record a pipeline cache (L3) miss.
    pub fn record_pipeline_miss(&self) {
        self.pipeline_cache_misses.fetch_add(1, ORD);
    }

    /// Record a Metal shader compilation with its duration.
    pub fn record_compile(&self, time_us: u64) {
        self.total_compile_time_us.fetch_add(time_us, ORD);
    }

    /// Record a GPU dispatch event.
    pub fn record_dispatch(&self) {
        self.total_dispatches.fetch_add(1, ORD);
    }

    // -----------------------------------------------------------------
    // Reading
    // -----------------------------------------------------------------

    /// Take a point-in-time snapshot of all counters.
    ///
    /// Each atomic load uses `Relaxed` ordering, so the snapshot is not
    /// guaranteed to be mutually consistent across counters (a concurrent
    /// writer may update one counter between reads). For diagnostics and
    /// benchmarking this is acceptable.
    #[must_use]
    pub fn snapshot(&self) -> CacheStatsSnapshot {
        CacheStatsSnapshot {
            kernel_cache_hits: self.kernel_cache_hits.load(ORD),
            kernel_cache_misses: self.kernel_cache_misses.load(ORD),
            msl_cache_hits: self.msl_cache_hits.load(ORD),
            msl_cache_misses: self.msl_cache_misses.load(ORD),
            pipeline_cache_hits: self.pipeline_cache_hits.load(ORD),
            pipeline_cache_misses: self.pipeline_cache_misses.load(ORD),
            total_dispatches: self.total_dispatches.load(ORD),
            total_compile_time_us: self.total_compile_time_us.load(ORD),
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.kernel_cache_hits.store(0, ORD);
        self.kernel_cache_misses.store(0, ORD);
        self.msl_cache_hits.store(0, ORD);
        self.msl_cache_misses.store(0, ORD);
        self.pipeline_cache_hits.store(0, ORD);
        self.pipeline_cache_misses.store(0, ORD);
        self.total_dispatches.store(0, ORD);
        self.total_compile_time_us.store(0, ORD);
    }
}

// CacheStats is Send + Sync because all fields are AtomicU64.
// The compiler infers this automatically, but we assert it for documentation.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CacheStats>();
};

/// Immutable point-in-time snapshot of cache statistics.
///
/// Produced by [`CacheStats::snapshot`]. All derived metrics (hit rates,
/// average compile time) are computed from the snapshot values.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CacheStatsSnapshot {
    /// L1 kernel def cache hits.
    pub kernel_cache_hits: u64,
    /// L1 kernel def cache misses.
    pub kernel_cache_misses: u64,
    /// L2 MSL codegen cache hits.
    pub msl_cache_hits: u64,
    /// L2 MSL codegen cache misses.
    pub msl_cache_misses: u64,
    /// L3 pipeline cache hits.
    pub pipeline_cache_hits: u64,
    /// L3 pipeline cache misses.
    pub pipeline_cache_misses: u64,
    /// Total GPU dispatches recorded.
    pub total_dispatches: u64,
    /// Cumulative Metal shader compile time in microseconds.
    pub total_compile_time_us: u64,
}

impl CacheStatsSnapshot {
    /// Overall hit rate across all three cache levels.
    ///
    /// Computed as `total_hits / (total_hits + total_misses)`.
    /// Returns 0.0 if no lookups have been recorded.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        // Accumulate in f64 to avoid u64 overflow when summing very large
        // per-level counts (sum can exceed u64::MAX even though each field fits).
        let hits = self.kernel_cache_hits as f64
            + self.msl_cache_hits as f64
            + self.pipeline_cache_hits as f64;
        let misses = self.kernel_cache_misses as f64
            + self.msl_cache_misses as f64
            + self.pipeline_cache_misses as f64;
        let total = hits + misses;
        if total == 0.0 {
            return 0.0;
        }
        hits / total
    }

    /// L1 kernel def cache hit rate. Returns 0.0 if no kernel lookups.
    #[must_use]
    pub fn kernel_hit_rate(&self) -> f64 {
        hit_rate(self.kernel_cache_hits, self.kernel_cache_misses)
    }

    /// L2 MSL codegen cache hit rate. Returns 0.0 if no MSL lookups.
    #[must_use]
    pub fn msl_hit_rate(&self) -> f64 {
        hit_rate(self.msl_cache_hits, self.msl_cache_misses)
    }

    /// L3 pipeline cache hit rate. Returns 0.0 if no pipeline lookups.
    #[must_use]
    pub fn pipeline_hit_rate(&self) -> f64 {
        hit_rate(self.pipeline_cache_hits, self.pipeline_cache_misses)
    }

    /// Average Metal shader compile time in microseconds.
    ///
    /// Computed from `total_compile_time_us / pipeline_cache_misses`.
    /// Returns 0.0 if no compilations have occurred.
    #[must_use]
    pub fn avg_compile_time_us(&self) -> f64 {
        if self.pipeline_cache_misses == 0 {
            return 0.0;
        }
        self.total_compile_time_us as f64 / self.pipeline_cache_misses as f64
    }

    /// Human-readable multi-line summary of cache performance.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Cache Statistics:\n\
             \n  L1 Kernel Def:  hits={}, misses={}, rate={:.1}%\
             \n  L2 MSL Codegen: hits={}, misses={}, rate={:.1}%\
             \n  L3 Pipeline:    hits={}, misses={}, rate={:.1}%\
             \n  Overall:        rate={:.1}%\
             \n  Dispatches:     {}\
             \n  Compile time:   {} us total, {:.1} us avg",
            self.kernel_cache_hits,
            self.kernel_cache_misses,
            self.kernel_hit_rate() * 100.0,
            self.msl_cache_hits,
            self.msl_cache_misses,
            self.msl_hit_rate() * 100.0,
            self.pipeline_cache_hits,
            self.pipeline_cache_misses,
            self.pipeline_hit_rate() * 100.0,
            self.hit_rate() * 100.0,
            self.total_dispatches,
            self.total_compile_time_us,
            self.avg_compile_time_us(),
        )
    }
}

impl fmt::Display for CacheStatsSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Compute hit rate from hits and misses. Returns 0.0 if total is 0.
fn hit_rate(hits: u64, misses: u64) -> f64 {
    let total = hits + misses;
    if total == 0 {
        return 0.0;
    }
    hits as f64 / total as f64
}

#[cfg(test)]
#[path = "cache_stats_tests.rs"]
mod cache_stats_tests;
