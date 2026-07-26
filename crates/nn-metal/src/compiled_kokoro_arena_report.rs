// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-synthesis arena utilization report for Kokoro pipelines.
//!
//! [`KokoroArenaReport`] captures before/after arena statistics around a
//! synthesis call, computing delta metrics that show how many GPU buffers
//! were reused via the arena vs. freshly allocated. This is the primary
//! tool for measuring the RTF impact of arena buffer reuse.
//!
//! # Usage
//!
//! ```rust,ignore
//! let (audio, cert, report) = kokoro.synthesize_with_arena_report(
//!     &input_ids, &style, 1.0, &cache,
//! )?;
//! println!("Arena hit rate: {:.0}%", report.hit_rate() * 100.0);
//! println!("Bytes saved: {:.1} MB", report.bytes_saved_mb());
//! ```
//!
//! Part of #4264.

use std::fmt;

use crate::arena::{self, ArenaStats};

/// Per-synthesis arena utilization metrics.
///
/// Computed as the delta between pre-synthesis and post-synthesis
/// [`ArenaStats`] snapshots. All counts are for a single synthesis call.
///
/// Part of #4264.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct KokoroArenaReport {
    /// Number of GPU buffer allocations served from the arena during synthesis.
    pub arena_hits: usize,
    /// Number of allocations that overflowed the arena (standalone or pool).
    pub arena_misses: usize,
    /// Number of pool hits (buffer reuse from the Metal buffer pool).
    pub pool_hits: usize,
    /// Number of fresh Metal allocations (neither arena nor pool served them).
    pub fresh_allocs: usize,
    /// Peak arena bytes used (high-water mark).
    pub peak_bytes: usize,
    /// Arena capacity in bytes (current slab size).
    pub capacity_bytes: usize,
    /// Number of auto-grow slab events during this synthesis call.
    pub growth_events: usize,
    /// Number of overflow events during this synthesis call.
    pub overflow_events: usize,
    /// Bytes allocated via overflow during this synthesis call.
    pub overflow_bytes: usize,
    /// Pool retained bytes at end of synthesis.
    pub pool_retained_bytes: usize,
    /// Pool retained buffer count at end of synthesis.
    pub pool_retained_buffers: usize,
}

impl KokoroArenaReport {
    /// Arena hit rate as a fraction in [0.0, 1.0].
    ///
    /// Returns 0.0 if no allocations occurred.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.arena_hits + self.arena_misses;
        if total == 0 {
            return 0.0;
        }
        self.arena_hits as f64 / total as f64
    }

    /// Total allocations during synthesis (arena + overflow).
    #[must_use]
    pub fn total_allocs(&self) -> usize {
        self.arena_hits + self.arena_misses
    }

    /// Estimated bytes saved by arena reuse.
    ///
    /// Each arena hit avoided a Metal buffer allocation. The average
    /// allocation size is estimated from `peak_bytes / arena_hits` when
    /// hits > 0. This is a rough estimate — actual savings depend on the
    /// size distribution of arena-served allocations.
    #[must_use]
    pub fn estimated_bytes_saved(&self) -> usize {
        if self.arena_hits == 0 {
            return 0;
        }
        // Use peak bytes as proxy for total arena usage across all hits.
        // Each hit reuses arena memory that would otherwise be a fresh alloc.
        self.peak_bytes
    }

    /// Estimated bytes saved in megabytes.
    #[must_use]
    pub fn bytes_saved_mb(&self) -> f64 {
        self.estimated_bytes_saved() as f64 / (1024.0 * 1024.0)
    }

    /// Peak arena utilization as a fraction of capacity.
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.capacity_bytes == 0 {
            return 0.0;
        }
        self.peak_bytes as f64 / self.capacity_bytes as f64
    }

    /// Returns `true` if the arena grew during this synthesis call.
    ///
    /// Growth indicates the initial arena capacity was insufficient.
    /// Consider calling `ensure_default_arena_capacity()` with a larger
    /// value, or using `estimate_arena_bytes()` to pre-size.
    #[must_use]
    pub fn had_growth(&self) -> bool {
        self.growth_events > 0
    }

    /// Returns `true` if any allocations overflowed the arena.
    #[must_use]
    pub fn had_overflow(&self) -> bool {
        self.overflow_events > 0
    }
}

impl fmt::Display for KokoroArenaReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn mb(b: usize) -> f64 {
            b as f64 / (1024.0 * 1024.0)
        }
        writeln!(f, "Kokoro Arena Report")?;
        writeln!(
            f,
            "  allocations: {} total ({} arena hits, {} misses)",
            self.total_allocs(),
            self.arena_hits,
            self.arena_misses,
        )?;
        writeln!(f, "  hit rate:    {:.1}%", self.hit_rate() * 100.0)?;
        writeln!(
            f,
            "  peak usage:  {:.1} MB / {:.1} MB capacity ({:.0}% utilization)",
            mb(self.peak_bytes),
            mb(self.capacity_bytes),
            self.utilization() * 100.0,
        )?;
        writeln!(
            f,
            "  pool reuse:  {} hits, {} fresh allocs",
            self.pool_hits, self.fresh_allocs,
        )?;
        writeln!(
            f,
            "  pool state:  {} buffers ({:.1} MB retained)",
            self.pool_retained_buffers,
            mb(self.pool_retained_bytes),
        )?;
        if self.growth_events > 0 {
            writeln!(f, "  growth:      {} slab events", self.growth_events)?;
        }
        if self.overflow_events > 0 {
            writeln!(
                f,
                "  overflow:    {} events ({:.1} MB)",
                self.overflow_events,
                mb(self.overflow_bytes),
            )?;
        }
        write!(
            f,
            "  bytes saved: ~{:.1} MB (arena reuse)",
            self.bytes_saved_mb(),
        )
    }
}

/// Capture a pre-synthesis arena stats snapshot.
///
/// Call this before synthesis, then call [`build_arena_report`] after
/// synthesis with the returned snapshot to compute delta metrics.
pub(crate) fn snapshot_arena_pre() -> ArenaStats {
    arena::reset_arena_stats();
    arena::arena_stats()
}

/// Build a [`KokoroArenaReport`] from pre/post synthesis arena stats.
pub(crate) fn build_arena_report(_pre: &ArenaStats) -> KokoroArenaReport {
    let post = arena::arena_stats();
    let peak = arena::default_arena_peak_bytes().unwrap_or(0);
    let capacity = arena::arena_capacity();

    KokoroArenaReport {
        arena_hits: post.hits,
        arena_misses: post.misses,
        pool_hits: post.pool.hits,
        fresh_allocs: post.fresh_allocs(),
        peak_bytes: peak,
        capacity_bytes: capacity,
        growth_events: post.growth_count,
        overflow_events: post.overflow_count,
        overflow_bytes: post.overflow_bytes,
        pool_retained_bytes: post.pool.pooled_bytes,
        pool_retained_buffers: post.pool.pooled_buffers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_report_display() {
        let report = KokoroArenaReport {
            arena_hits: 150,
            arena_misses: 10,
            pool_hits: 5,
            fresh_allocs: 5,
            peak_bytes: 32 * 1024 * 1024,
            capacity_bytes: 64 * 1024 * 1024,
            growth_events: 0,
            overflow_events: 0,
            overflow_bytes: 0,
            pool_retained_bytes: 4 * 1024 * 1024,
            pool_retained_buffers: 8,
        };
        let s = format!("{report}");
        assert!(s.contains("150 arena hits"));
        assert!(s.contains("hit rate"));
    }

    #[test]
    fn test_arena_report_hit_rate() {
        let report = KokoroArenaReport {
            arena_hits: 80,
            arena_misses: 20,
            pool_hits: 10,
            fresh_allocs: 10,
            peak_bytes: 0,
            capacity_bytes: 0,
            growth_events: 0,
            overflow_events: 0,
            overflow_bytes: 0,
            pool_retained_bytes: 0,
            pool_retained_buffers: 0,
        };
        assert!((report.hit_rate() - 0.8).abs() < 1e-9);
        assert_eq!(report.total_allocs(), 100);
    }

    #[test]
    fn test_arena_report_zero_allocs() {
        let report = KokoroArenaReport {
            arena_hits: 0,
            arena_misses: 0,
            pool_hits: 0,
            fresh_allocs: 0,
            peak_bytes: 0,
            capacity_bytes: 64 * 1024 * 1024,
            growth_events: 0,
            overflow_events: 0,
            overflow_bytes: 0,
            pool_retained_bytes: 0,
            pool_retained_buffers: 0,
        };
        assert!((report.hit_rate() - 0.0).abs() < 1e-9);
        assert!((report.utilization() - 0.0).abs() < 1e-9);
        assert!(!report.had_growth());
        assert!(!report.had_overflow());
    }

    #[test]
    fn test_arena_report_with_growth() {
        let report = KokoroArenaReport {
            arena_hits: 100,
            arena_misses: 5,
            pool_hits: 3,
            fresh_allocs: 2,
            peak_bytes: 100 * 1024 * 1024,
            capacity_bytes: 128 * 1024 * 1024,
            growth_events: 1,
            overflow_events: 2,
            overflow_bytes: 8 * 1024 * 1024,
            pool_retained_bytes: 2 * 1024 * 1024,
            pool_retained_buffers: 4,
        };
        assert!(report.had_growth());
        assert!(report.had_overflow());
        let s = format!("{report}");
        assert!(s.contains("growth"));
        assert!(s.contains("overflow"));
    }
}
