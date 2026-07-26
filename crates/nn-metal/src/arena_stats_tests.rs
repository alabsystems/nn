// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for [`ArenaStats`], [`PoolStats`] computations, and
//! arena statistics query functions.

use super::super::{arena_capacity, arena_stats, reset_arena_stats, ArenaStats};
use super::super::PoolStats;

// ---------------------------------------------------------------------------
// ArenaStats::hit_rate
// ---------------------------------------------------------------------------

#[test]
fn test_hit_rate_zero_allocations() {
    let stats = ArenaStats {
        hits: 0,
        misses: 0,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert_eq!(stats.hit_rate(), 0.0, "no allocations → 0.0 hit rate");
}

#[test]
fn test_hit_rate_all_hits() {
    let stats = ArenaStats {
        hits: 100,
        misses: 0,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert_eq!(stats.hit_rate(), 1.0, "all hits → 1.0 hit rate");
}

#[test]
fn test_hit_rate_all_misses() {
    let stats = ArenaStats {
        hits: 0,
        misses: 50,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert_eq!(stats.hit_rate(), 0.0, "all misses → 0.0 hit rate");
}

#[test]
fn test_hit_rate_mixed() {
    let stats = ArenaStats {
        hits: 3,
        misses: 1,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert!((stats.hit_rate() - 0.75).abs() < 1e-12, "3/4 = 0.75");
}

#[test]
fn test_hit_rate_single_hit() {
    let stats = ArenaStats {
        hits: 1,
        misses: 0,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert_eq!(stats.hit_rate(), 1.0);
}

#[test]
fn test_hit_rate_single_miss() {
    let stats = ArenaStats {
        hits: 0,
        misses: 1,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert_eq!(stats.hit_rate(), 0.0);
}

// ---------------------------------------------------------------------------
// ArenaStats::fresh_allocs
// ---------------------------------------------------------------------------

#[test]
fn test_fresh_allocs_no_pool_hits() {
    let stats = ArenaStats {
        hits: 10,
        misses: 5,
        pool: PoolStats {
            hits: 0,
            ..Default::default()
        },
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert_eq!(
        stats.fresh_allocs(),
        5,
        "no pool hits → all misses are fresh allocs"
    );
}

#[test]
fn test_fresh_allocs_all_pool_hits() {
    let stats = ArenaStats {
        hits: 10,
        misses: 5,
        pool: PoolStats {
            hits: 5,
            ..Default::default()
        },
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert_eq!(
        stats.fresh_allocs(),
        0,
        "pool caught all misses → 0 fresh allocs"
    );
}

#[test]
fn test_fresh_allocs_partial_pool_coverage() {
    let stats = ArenaStats {
        hits: 10,
        misses: 8,
        pool: PoolStats {
            hits: 3,
            ..Default::default()
        },
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert_eq!(
        stats.fresh_allocs(),
        5,
        "8 misses - 3 pool hits = 5 fresh"
    );
}

#[test]
fn test_fresh_allocs_pool_hits_exceed_misses_saturates() {
    // Defensive: saturating_sub prevents underflow if pool.hits > misses
    // (should not happen in practice, but the API is safe).
    let stats = ArenaStats {
        hits: 10,
        misses: 2,
        pool: PoolStats {
            hits: 5,
            ..Default::default()
        },
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert_eq!(
        stats.fresh_allocs(),
        0,
        "saturating_sub prevents underflow"
    );
}

#[test]
fn test_fresh_allocs_zero_everything() {
    let stats = ArenaStats {
        hits: 0,
        misses: 0,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert_eq!(stats.fresh_allocs(), 0);
}

// ---------------------------------------------------------------------------
// ArenaStats equality and debug
// ---------------------------------------------------------------------------

#[test]
fn test_arena_stats_eq() {
    let a = ArenaStats {
        hits: 5,
        misses: 3,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    let b = ArenaStats {
        hits: 5,
        misses: 3,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert_eq!(a, b);
}

#[test]
fn test_arena_stats_ne_hits() {
    let a = ArenaStats {
        hits: 5,
        misses: 3,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    let b = ArenaStats {
        hits: 6,
        misses: 3,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert_ne!(a, b);
}

#[test]
fn test_arena_stats_debug() {
    let stats = ArenaStats {
        hits: 1,
        misses: 2,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    let dbg = format!("{stats:?}");
    assert!(dbg.contains("ArenaStats"));
    assert!(dbg.contains("hits"));
    assert!(dbg.contains("misses"));
}

// ---------------------------------------------------------------------------
// PoolStats default and equality
// ---------------------------------------------------------------------------

#[test]
fn test_pool_stats_default_all_zero() {
    let ps = PoolStats::default();
    assert_eq!(ps.acquisitions, 0);
    assert_eq!(ps.hits, 0);
    assert_eq!(ps.misses, 0);
    assert_eq!(ps.discards, 0);
    assert_eq!(ps.pooled_bytes, 0);
    assert_eq!(ps.pooled_buffers, 0);
}

#[test]
fn test_pool_stats_copy_semantics() {
    let a = PoolStats {
        acquisitions: 10,
        hits: 4,
        misses: 3,
        discards: 3,
        pooled_bytes: 1024,
        pooled_buffers: 2,
    };
    let b = a; // Copy
    assert_eq!(a, b);
}

#[test]
fn test_pool_stats_debug() {
    let ps = PoolStats::default();
    let dbg = format!("{ps:?}");
    assert!(dbg.contains("PoolStats"));
}

// ---------------------------------------------------------------------------
// Thread-local arena_stats() and reset_arena_stats()
// ---------------------------------------------------------------------------

#[test]
fn test_arena_stats_after_reset() {
    reset_arena_stats();
    let stats = arena_stats();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
}

#[test]
fn test_arena_capacity_is_64mb() {
    let cap = arena_capacity();
    assert_eq!(cap, 64 * 1024 * 1024, "default arena capacity = 64 MB");
}

#[test]
fn test_reset_arena_stats_idempotent() {
    reset_arena_stats();
    reset_arena_stats();
    let stats = arena_stats();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
}

// ---------------------------------------------------------------------------
// ArenaStats Copy/Clone semantics
// ---------------------------------------------------------------------------

#[test]
fn test_arena_stats_copy_semantics() {
    let a = ArenaStats {
        hits: 42,
        misses: 7,
        pool: PoolStats {
            acquisitions: 10,
            hits: 3,
            misses: 4,
            discards: 3,
            pooled_bytes: 2048,
            pooled_buffers: 5,
        },
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    let b = a; // Copy
    // Both `a` and `b` remain usable after copy (no move).
    assert_eq!(a, b);
    assert_eq!(a.hits, b.hits);
    assert_eq!(a.pool.pooled_bytes, b.pool.pooled_bytes);
}

#[test]
fn test_arena_stats_clone() {
    let a = ArenaStats {
        hits: 100,
        misses: 20,
        pool: PoolStats {
            acquisitions: 15,
            hits: 8,
            misses: 5,
            discards: 2,
            pooled_bytes: 4096,
            pooled_buffers: 3,
        },
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    let b = a;
    assert_eq!(a, b);
}

// ---------------------------------------------------------------------------
// ArenaStats inequality — misses and pool fields
// ---------------------------------------------------------------------------

#[test]
fn test_arena_stats_ne_misses() {
    let a = ArenaStats {
        hits: 5,
        misses: 3,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    let b = ArenaStats {
        hits: 5,
        misses: 4,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert_ne!(a, b);
}

#[test]
fn test_arena_stats_ne_pool() {
    let a = ArenaStats {
        hits: 5,
        misses: 3,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    let b = ArenaStats {
        hits: 5,
        misses: 3,
        pool: PoolStats {
            hits: 1,
            ..Default::default()
        },
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert_ne!(a, b, "different pool stats → different ArenaStats");
}

#[test]
fn test_pool_stats_ne() {
    let a = PoolStats {
        acquisitions: 10,
        hits: 4,
        misses: 3,
        discards: 3,
        pooled_bytes: 1024,
        pooled_buffers: 2,
    };
    let b = PoolStats {
        acquisitions: 10,
        hits: 4,
        misses: 3,
        discards: 3,
        pooled_bytes: 2048, // different
        pooled_buffers: 2,
    };
    assert_ne!(a, b);
}

// ---------------------------------------------------------------------------
// hit_rate edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_hit_rate_large_values() {
    // Large counts near usize::MAX / 2 to exercise f64 precision.
    let big = usize::MAX / 2;
    let stats = ArenaStats {
        hits: big,
        misses: big,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    // Equal hits and misses → ~0.5 hit rate.
    assert!(
        (stats.hit_rate() - 0.5).abs() < 1e-9,
        "large equal counts → ~0.5 hit rate, got {}",
        stats.hit_rate()
    );
}

#[test]
fn test_hit_rate_50_50() {
    let stats = ArenaStats {
        hits: 50,
        misses: 50,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert!((stats.hit_rate() - 0.5).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// fresh_allocs edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_fresh_allocs_large_values() {
    let stats = ArenaStats {
        hits: 0,
        misses: usize::MAX,
        pool: PoolStats {
            hits: 1,
            ..Default::default()
        },
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert_eq!(
        stats.fresh_allocs(),
        usize::MAX - 1,
        "large misses minus 1 pool hit"
    );
}

#[test]
fn test_fresh_allocs_pool_discards_dont_reduce_fresh() {
    // Discards are requests that bypassed the pool entirely. They are
    // counted in misses but NOT in pool.hits, so fresh_allocs includes them.
    let stats = ArenaStats {
        hits: 0,
        misses: 10,
        pool: PoolStats {
            acquisitions: 10,
            hits: 2,
            misses: 3,
            discards: 5,
            pooled_bytes: 0,
            pooled_buffers: 0,
        },
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert_eq!(
        stats.fresh_allocs(),
        8,
        "10 misses - 2 pool hits = 8 fresh (discards don't reduce)"
    );
}

// ---------------------------------------------------------------------------
// arena_capacity invariants
// ---------------------------------------------------------------------------

#[test]
fn test_arena_capacity_is_power_of_two() {
    let cap = arena_capacity();
    assert!(cap > 0, "capacity must be positive");
    assert!(
        cap.is_power_of_two(),
        "capacity ({cap}) must be a power of two"
    );
}

#[test]
fn test_arena_capacity_at_least_1mb() {
    let cap = arena_capacity();
    assert!(
        cap >= 1024 * 1024,
        "capacity ({cap}) should be at least 1 MB"
    );
}

// ---------------------------------------------------------------------------
// Thread isolation: each thread has its own stats
// ---------------------------------------------------------------------------

#[test]
fn test_arena_stats_thread_isolation() {
    use std::sync::mpsc;

    // Reset stats on this thread.
    reset_arena_stats();

    let (tx, rx) = mpsc::channel();

    // Spawn a thread and read its stats — they should be independent
    // (zero) because thread-local cells are per-thread.
    std::thread::spawn(move || {
        let stats = arena_stats();
        tx.send((stats.hits, stats.misses)).unwrap();
    })
    .join()
    .expect("thread join");

    let (hits, misses) = rx.recv().unwrap();
    // The spawned thread never allocated anything, so its counters are zero.
    assert_eq!(hits, 0, "spawned thread hits should be zero");
    assert_eq!(misses, 0, "spawned thread misses should be zero");
}

// ---------------------------------------------------------------------------
// ArenaStats overflow fields
// ---------------------------------------------------------------------------

#[test]
fn test_arena_stats_overflow_fields_default_zero() {
    let stats = ArenaStats {
        hits: 0,
        misses: 0,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    assert_eq!(stats.overflow_count, 0);
    assert_eq!(stats.total_overflow_count, 0);
    assert_eq!(stats.overflow_bytes, 0);
    assert_eq!(stats.total_overflow_bytes, 0);
}

#[test]
fn test_arena_stats_ne_overflow_count() {
    let a = ArenaStats {
        hits: 5,
        misses: 3,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 0,
        total_overflow_count: 0,
        overflow_bytes: 0,
        total_overflow_bytes: 0,
    };
    let b = ArenaStats {
        hits: 5,
        misses: 3,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 1,
        total_overflow_count: 1,
        overflow_bytes: 4096,
        total_overflow_bytes: 4096,
    };
    assert_ne!(a, b, "different overflow counts → different ArenaStats");
}

#[test]
fn test_arena_stats_overflow_debug_contains_fields() {
    let stats = ArenaStats {
        hits: 0,
        misses: 0,
        pool: PoolStats::default(),
        growth_count: 0,
        total_growth_count: 0,
        overflow_count: 3,
        total_overflow_count: 7,
        overflow_bytes: 12288,
        total_overflow_bytes: 28672,
    };
    let dbg = format!("{stats:?}");
    assert!(dbg.contains("overflow_count"), "debug shows overflow_count");
    assert!(dbg.contains("overflow_bytes"), "debug shows overflow_bytes");
}

// ---------------------------------------------------------------------------
// estimate_arena_peak_bytes
// ---------------------------------------------------------------------------

#[test]
fn test_estimate_arena_peak_bytes_empty() {
    let est = super::super::estimate_arena_peak_bytes(std::iter::empty());
    assert_eq!(est.peak_bytes, 0);
    assert_eq!(est.total_bytes, 0);
    assert_eq!(est.step_count, 0);
}

#[test]
fn test_estimate_arena_peak_bytes_single() {
    let est = super::super::estimate_arena_peak_bytes(std::iter::once(1024));
    assert_eq!(est.peak_bytes, 1024);
    assert_eq!(est.total_bytes, 1024);
    assert_eq!(est.step_count, 1);
}

#[test]
fn test_estimate_arena_peak_bytes_multiple() {
    // 100 bytes at offset 0 → used 100
    // next aligned to 256, then 200 → 256 + 200 = 456
    // next aligned to 512, then 300 → 512 + 300 = 812
    let est = super::super::estimate_arena_peak_bytes(vec![100, 200, 300]);
    assert_eq!(est.peak_bytes, 812);
    assert_eq!(est.total_bytes, 600);
    assert_eq!(est.step_count, 3);
}

#[test]
fn test_estimate_arena_peak_bytes_skips_zero() {
    let est = super::super::estimate_arena_peak_bytes(vec![0, 1024, 0, 2048]);
    assert_eq!(est.step_count, 2, "zero-size entries skipped");
    assert_eq!(est.total_bytes, 3072);
}

#[test]
fn test_estimate_arena_peak_bytes_alignment() {
    // 1 byte at offset 0 → used 1
    // next aligned to 256, then 1 byte → 256 + 1 = 257
    let est = super::super::estimate_arena_peak_bytes(vec![1, 1]);
    assert_eq!(est.peak_bytes, 257);
}

#[test]
fn test_estimate_arena_peak_bytes_large_allocation() {
    // A single large allocation
    let big = 256 * 1024 * 1024; // 256 MB
    let est = super::super::estimate_arena_peak_bytes(std::iter::once(big));
    assert_eq!(est.peak_bytes, big);
    assert_eq!(est.total_bytes, big);
}

// ---------------------------------------------------------------------------
// estimate_arena_peak_from_shapes
// ---------------------------------------------------------------------------

#[test]
fn test_estimate_arena_peak_from_shapes_basic() {
    let entries: Vec<(&str, &[usize], usize)> = vec![
        ("conv_out", &[1, 256, 100], 4),   // 102400 bytes
        ("norm_out", &[1, 256, 100], 4),    // 102400 bytes
    ];
    let est = super::super::estimate_arena_peak_from_shapes(&entries);
    assert_eq!(est.step_count, 2);
    assert_eq!(est.total_bytes, 204800);
    // First at offset 0: 102400. 102400 is already a multiple of the 256-byte
    // alignment (102400 = 400 * 256), so the second allocation needs no padding:
    // 102400 + 102400 = 204800.
    assert_eq!(est.peak_bytes, 204800);
}

#[test]
fn test_estimate_arena_peak_from_shapes_empty() {
    let entries: Vec<(&str, &[usize], usize)> = vec![];
    let est = super::super::estimate_arena_peak_from_shapes(&entries);
    assert_eq!(est.peak_bytes, 0);
    assert_eq!(est.step_count, 0);
}
