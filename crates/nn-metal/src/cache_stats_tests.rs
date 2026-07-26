// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`CacheStats`] and [`CacheStatsSnapshot`].

use super::*;

// ---------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------

#[test]
fn test_snapshot_default_is_zero() {
    let snap = CacheStatsSnapshot::default();
    assert_eq!(snap.kernel_cache_hits, 0);
    assert_eq!(snap.kernel_cache_misses, 0);
    assert_eq!(snap.msl_cache_hits, 0);
    assert_eq!(snap.msl_cache_misses, 0);
    assert_eq!(snap.pipeline_cache_hits, 0);
    assert_eq!(snap.pipeline_cache_misses, 0);
    assert_eq!(snap.total_dispatches, 0);
    assert_eq!(snap.total_compile_time_us, 0);
}

#[test]
fn test_new_stats_snapshot_is_zero() {
    let stats = CacheStats::new();
    let snap = stats.snapshot();
    assert_eq!(snap, CacheStatsSnapshot::default());
}

// ---------------------------------------------------------------
// Hit / miss counting
// ---------------------------------------------------------------

#[test]
fn test_record_kernel_hit_miss() {
    let stats = CacheStats::new();
    stats.record_kernel_hit();
    stats.record_kernel_hit();
    stats.record_kernel_miss();

    let snap = stats.snapshot();
    assert_eq!(snap.kernel_cache_hits, 2);
    assert_eq!(snap.kernel_cache_misses, 1);
}

#[test]
fn test_record_msl_hit_miss() {
    let stats = CacheStats::new();
    stats.record_msl_hit();
    stats.record_msl_miss();
    stats.record_msl_miss();

    let snap = stats.snapshot();
    assert_eq!(snap.msl_cache_hits, 1);
    assert_eq!(snap.msl_cache_misses, 2);
}

#[test]
fn test_record_pipeline_hit_miss() {
    let stats = CacheStats::new();
    stats.record_pipeline_hit();
    stats.record_pipeline_hit();
    stats.record_pipeline_hit();
    stats.record_pipeline_miss();

    let snap = stats.snapshot();
    assert_eq!(snap.pipeline_cache_hits, 3);
    assert_eq!(snap.pipeline_cache_misses, 1);
}

#[test]
fn test_record_compile_accumulates() {
    let stats = CacheStats::new();
    stats.record_compile(500);
    stats.record_compile(300);
    stats.record_compile(200);

    let snap = stats.snapshot();
    assert_eq!(snap.total_compile_time_us, 1000);
}

#[test]
fn test_record_dispatch_counts() {
    let stats = CacheStats::new();
    for _ in 0..17 {
        stats.record_dispatch();
    }
    let snap = stats.snapshot();
    assert_eq!(snap.total_dispatches, 17);
}

#[test]
fn test_record_and_snapshot_combined() {
    let stats = CacheStats::new();
    stats.record_kernel_hit();
    stats.record_kernel_hit();
    stats.record_kernel_miss();
    stats.record_msl_hit();
    stats.record_msl_miss();
    stats.record_msl_miss();
    stats.record_pipeline_hit();
    stats.record_pipeline_hit();
    stats.record_pipeline_hit();
    stats.record_pipeline_miss();
    stats.record_compile(500);
    stats.record_compile(300);
    stats.record_dispatch();
    stats.record_dispatch();
    stats.record_dispatch();

    let snap = stats.snapshot();
    assert_eq!(snap.kernel_cache_hits, 2);
    assert_eq!(snap.kernel_cache_misses, 1);
    assert_eq!(snap.msl_cache_hits, 1);
    assert_eq!(snap.msl_cache_misses, 2);
    assert_eq!(snap.pipeline_cache_hits, 3);
    assert_eq!(snap.pipeline_cache_misses, 1);
    assert_eq!(snap.total_compile_time_us, 800);
    assert_eq!(snap.total_dispatches, 3);
}

// ---------------------------------------------------------------
// Hit rate calculations
// ---------------------------------------------------------------

#[test]
fn test_hit_rate_zero_lookups() {
    let snap = CacheStatsSnapshot::default();
    assert_eq!(snap.hit_rate(), 0.0);
    assert_eq!(snap.kernel_hit_rate(), 0.0);
    assert_eq!(snap.msl_hit_rate(), 0.0);
    assert_eq!(snap.pipeline_hit_rate(), 0.0);
}

#[test]
fn test_hit_rate_all_hits() {
    let snap = CacheStatsSnapshot {
        kernel_cache_hits: 10,
        kernel_cache_misses: 0,
        msl_cache_hits: 20,
        msl_cache_misses: 0,
        pipeline_cache_hits: 30,
        pipeline_cache_misses: 0,
        total_dispatches: 60,
        total_compile_time_us: 0,
    };
    assert_eq!(snap.hit_rate(), 1.0);
    assert_eq!(snap.kernel_hit_rate(), 1.0);
    assert_eq!(snap.msl_hit_rate(), 1.0);
    assert_eq!(snap.pipeline_hit_rate(), 1.0);
}

#[test]
fn test_hit_rate_all_misses() {
    let snap = CacheStatsSnapshot {
        kernel_cache_hits: 0,
        kernel_cache_misses: 5,
        msl_cache_hits: 0,
        msl_cache_misses: 10,
        pipeline_cache_hits: 0,
        pipeline_cache_misses: 15,
        total_dispatches: 30,
        total_compile_time_us: 100_000,
    };
    assert_eq!(snap.hit_rate(), 0.0);
    assert_eq!(snap.kernel_hit_rate(), 0.0);
    assert_eq!(snap.msl_hit_rate(), 0.0);
    assert_eq!(snap.pipeline_hit_rate(), 0.0);
}

#[test]
fn test_hit_rate_mixed() {
    let snap = CacheStatsSnapshot {
        kernel_cache_hits: 80,
        kernel_cache_misses: 20,
        msl_cache_hits: 0,
        msl_cache_misses: 0,
        pipeline_cache_hits: 0,
        pipeline_cache_misses: 0,
        total_dispatches: 100,
        total_compile_time_us: 0,
    };
    assert!((snap.kernel_hit_rate() - 0.8).abs() < 1e-10);
    // Overall: 80 hits / 100 total
    assert!((snap.hit_rate() - 0.8).abs() < 1e-10);
}

#[test]
fn test_hit_rate_50_percent() {
    let snap = CacheStatsSnapshot {
        kernel_cache_hits: 5,
        kernel_cache_misses: 5,
        msl_cache_hits: 10,
        msl_cache_misses: 10,
        pipeline_cache_hits: 15,
        pipeline_cache_misses: 15,
        ..Default::default()
    };
    assert!((snap.hit_rate() - 0.5).abs() < 1e-10);
    assert!((snap.kernel_hit_rate() - 0.5).abs() < 1e-10);
    assert!((snap.msl_hit_rate() - 0.5).abs() < 1e-10);
    assert!((snap.pipeline_hit_rate() - 0.5).abs() < 1e-10);
}

#[test]
fn test_per_level_hit_rates_independent() {
    // Each level can have a different hit rate
    let snap = CacheStatsSnapshot {
        kernel_cache_hits: 9,
        kernel_cache_misses: 1,
        msl_cache_hits: 1,
        msl_cache_misses: 9,
        pipeline_cache_hits: 5,
        pipeline_cache_misses: 5,
        ..Default::default()
    };
    assert!((snap.kernel_hit_rate() - 0.9).abs() < 1e-10);
    assert!((snap.msl_hit_rate() - 0.1).abs() < 1e-10);
    assert!((snap.pipeline_hit_rate() - 0.5).abs() < 1e-10);
    // Overall: 15 hits / 30 total = 0.5
    assert!((snap.hit_rate() - 0.5).abs() < 1e-10);
}

#[test]
fn test_hit_rate_single_level_only() {
    // Only kernel has data — msl and pipeline are zero
    let snap = CacheStatsSnapshot {
        kernel_cache_hits: 7,
        kernel_cache_misses: 3,
        ..Default::default()
    };
    assert!((snap.kernel_hit_rate() - 0.7).abs() < 1e-10);
    assert_eq!(snap.msl_hit_rate(), 0.0);
    assert_eq!(snap.pipeline_hit_rate(), 0.0);
    // Overall: only kernel contributes
    assert!((snap.hit_rate() - 0.7).abs() < 1e-10);
}

// ---------------------------------------------------------------
// Average compile time
// ---------------------------------------------------------------

#[test]
fn test_avg_compile_time_no_compiles() {
    let snap = CacheStatsSnapshot::default();
    assert_eq!(snap.avg_compile_time_us(), 0.0);
}

#[test]
fn test_avg_compile_time() {
    let snap = CacheStatsSnapshot {
        pipeline_cache_misses: 4,
        total_compile_time_us: 1000,
        ..Default::default()
    };
    assert!((snap.avg_compile_time_us() - 250.0).abs() < 1e-10);
}

#[test]
fn test_avg_compile_time_single_compile() {
    let snap = CacheStatsSnapshot {
        pipeline_cache_misses: 1,
        total_compile_time_us: 4200,
        ..Default::default()
    };
    assert!((snap.avg_compile_time_us() - 4200.0).abs() < 1e-10);
}

// ---------------------------------------------------------------
// Reset / clear
// ---------------------------------------------------------------

#[test]
fn test_reset() {
    let stats = CacheStats::new();
    stats.record_kernel_hit();
    stats.record_kernel_miss();
    stats.record_msl_hit();
    stats.record_msl_miss();
    stats.record_pipeline_hit();
    stats.record_pipeline_miss();
    stats.record_compile(100);
    stats.record_dispatch();
    stats.reset();

    let snap = stats.snapshot();
    assert_eq!(snap, CacheStatsSnapshot::default());
}

#[test]
fn test_reset_then_record() {
    let stats = CacheStats::new();
    stats.record_kernel_hit();
    stats.record_kernel_hit();
    stats.reset();
    stats.record_kernel_hit();

    let snap = stats.snapshot();
    assert_eq!(snap.kernel_cache_hits, 1);
    assert_eq!(snap.kernel_cache_misses, 0);
}

#[test]
fn test_double_reset() {
    let stats = CacheStats::new();
    stats.record_kernel_hit();
    stats.reset();
    stats.reset();
    let snap = stats.snapshot();
    assert_eq!(snap, CacheStatsSnapshot::default());
}

// ---------------------------------------------------------------
// Display / summary formatting
// ---------------------------------------------------------------

#[test]
fn test_summary_contains_all_levels() {
    let snap = CacheStatsSnapshot {
        kernel_cache_hits: 100,
        kernel_cache_misses: 10,
        msl_cache_hits: 90,
        msl_cache_misses: 5,
        pipeline_cache_hits: 80,
        pipeline_cache_misses: 2,
        total_dispatches: 200,
        total_compile_time_us: 5000,
    };
    let summary = snap.summary();
    assert!(summary.contains("L1 Kernel Def"));
    assert!(summary.contains("L2 MSL Codegen"));
    assert!(summary.contains("L3 Pipeline"));
    assert!(summary.contains("Overall"));
    assert!(summary.contains("Dispatches"));
    assert!(summary.contains("Compile time"));
}

#[test]
fn test_display_matches_summary() {
    let snap = CacheStatsSnapshot {
        kernel_cache_hits: 5,
        kernel_cache_misses: 1,
        ..Default::default()
    };
    assert_eq!(format!("{snap}"), snap.summary());
}

#[test]
fn test_summary_zero_stats() {
    let snap = CacheStatsSnapshot::default();
    let summary = snap.summary();
    assert!(summary.contains("hits=0"));
    assert!(summary.contains("misses=0"));
    assert!(summary.contains("rate=0.0%"));
}

#[test]
fn test_summary_shows_correct_hit_counts() {
    let snap = CacheStatsSnapshot {
        kernel_cache_hits: 42,
        kernel_cache_misses: 8,
        ..Default::default()
    };
    let summary = snap.summary();
    assert!(summary.contains("hits=42"));
    assert!(summary.contains("misses=8"));
}

#[test]
fn test_summary_shows_dispatch_count() {
    let snap = CacheStatsSnapshot {
        total_dispatches: 999,
        ..Default::default()
    };
    let summary = snap.summary();
    assert!(summary.contains("999"));
}

#[test]
fn test_summary_shows_compile_time() {
    let snap = CacheStatsSnapshot {
        total_compile_time_us: 12345,
        pipeline_cache_misses: 3,
        ..Default::default()
    };
    let summary = snap.summary();
    assert!(summary.contains("12345 us total"));
}

// ---------------------------------------------------------------
// Multiple independent instances
// ---------------------------------------------------------------

#[test]
fn test_independent_instances() {
    let stats_a = CacheStats::new();
    let stats_b = CacheStats::new();

    stats_a.record_kernel_hit();
    stats_a.record_kernel_hit();
    stats_a.record_kernel_hit();
    stats_b.record_kernel_miss();

    let snap_a = stats_a.snapshot();
    let snap_b = stats_b.snapshot();

    assert_eq!(snap_a.kernel_cache_hits, 3);
    assert_eq!(snap_a.kernel_cache_misses, 0);
    assert_eq!(snap_b.kernel_cache_hits, 0);
    assert_eq!(snap_b.kernel_cache_misses, 1);
}

#[test]
fn test_independent_instances_msl() {
    let stats_a = CacheStats::new();
    let stats_b = CacheStats::new();

    stats_a.record_msl_hit();
    stats_b.record_msl_miss();
    stats_b.record_msl_miss();

    assert_eq!(stats_a.snapshot().msl_cache_hits, 1);
    assert_eq!(stats_a.snapshot().msl_cache_misses, 0);
    assert_eq!(stats_b.snapshot().msl_cache_hits, 0);
    assert_eq!(stats_b.snapshot().msl_cache_misses, 2);
}

#[test]
fn test_independent_instances_pipeline() {
    let stats_a = CacheStats::new();
    let stats_b = CacheStats::new();

    stats_a.record_pipeline_hit();
    stats_a.record_compile(100);
    stats_b.record_pipeline_miss();
    stats_b.record_compile(500);

    let snap_a = stats_a.snapshot();
    let snap_b = stats_b.snapshot();

    assert_eq!(snap_a.pipeline_cache_hits, 1);
    assert_eq!(snap_a.pipeline_cache_misses, 0);
    assert_eq!(snap_a.total_compile_time_us, 100);
    assert_eq!(snap_b.pipeline_cache_hits, 0);
    assert_eq!(snap_b.pipeline_cache_misses, 1);
    assert_eq!(snap_b.total_compile_time_us, 500);
}

#[test]
fn test_independent_instances_dispatch() {
    let stats_a = CacheStats::new();
    let stats_b = CacheStats::new();

    for _ in 0..10 {
        stats_a.record_dispatch();
    }
    for _ in 0..3 {
        stats_b.record_dispatch();
    }

    assert_eq!(stats_a.snapshot().total_dispatches, 10);
    assert_eq!(stats_b.snapshot().total_dispatches, 3);
}

// ---------------------------------------------------------------
// Thread safety: concurrent hit/miss from multiple threads
// ---------------------------------------------------------------

#[test]
fn test_concurrent_kernel_hits() {
    let stats = std::sync::Arc::new(CacheStats::new());
    let mut handles = Vec::new();
    let threads = 8;
    let ops_per_thread = 1000;

    for _ in 0..threads {
        let s = stats.clone();
        handles.push(std::thread::spawn(move || {
            for _ in 0..ops_per_thread {
                s.record_kernel_hit();
            }
        }));
    }
    for h in handles {
        h.join().expect("thread panicked");
    }

    let snap = stats.snapshot();
    assert_eq!(snap.kernel_cache_hits, (threads * ops_per_thread) as u64);
}

#[test]
fn test_concurrent_mixed_operations() {
    let stats = std::sync::Arc::new(CacheStats::new());
    let mut handles = Vec::new();
    let threads = 4;
    let ops_per_thread = 500;

    for _ in 0..threads {
        let s = stats.clone();
        handles.push(std::thread::spawn(move || {
            for _ in 0..ops_per_thread {
                s.record_kernel_hit();
                s.record_kernel_miss();
                s.record_msl_hit();
                s.record_msl_miss();
                s.record_pipeline_hit();
                s.record_pipeline_miss();
                s.record_compile(10);
                s.record_dispatch();
            }
        }));
    }
    for h in handles {
        h.join().expect("thread panicked");
    }

    let snap = stats.snapshot();
    let total = (threads * ops_per_thread) as u64;
    assert_eq!(snap.kernel_cache_hits, total);
    assert_eq!(snap.kernel_cache_misses, total);
    assert_eq!(snap.msl_cache_hits, total);
    assert_eq!(snap.msl_cache_misses, total);
    assert_eq!(snap.pipeline_cache_hits, total);
    assert_eq!(snap.pipeline_cache_misses, total);
    assert_eq!(snap.total_compile_time_us, total * 10);
    assert_eq!(snap.total_dispatches, total);
}

#[test]
fn test_concurrent_dispatch_only() {
    let stats = std::sync::Arc::new(CacheStats::new());
    let mut handles = Vec::new();
    let threads = 16;
    let ops_per_thread = 250;

    for _ in 0..threads {
        let s = stats.clone();
        handles.push(std::thread::spawn(move || {
            for _ in 0..ops_per_thread {
                s.record_dispatch();
            }
        }));
    }
    for h in handles {
        h.join().expect("thread panicked");
    }

    assert_eq!(
        stats.snapshot().total_dispatches,
        (threads * ops_per_thread) as u64
    );
}

#[test]
fn test_concurrent_compile_time_accumulation() {
    let stats = std::sync::Arc::new(CacheStats::new());
    let mut handles = Vec::new();
    let threads = 8;
    let ops_per_thread = 100;
    let time_per_op: u64 = 7;

    for _ in 0..threads {
        let s = stats.clone();
        handles.push(std::thread::spawn(move || {
            for _ in 0..ops_per_thread {
                s.record_compile(time_per_op);
            }
        }));
    }
    for h in handles {
        h.join().expect("thread panicked");
    }

    let expected = threads as u64 * ops_per_thread as u64 * time_per_op;
    assert_eq!(stats.snapshot().total_compile_time_us, expected);
}

// ---------------------------------------------------------------
// Overflow safety for very large counts (AtomicU64)
// ---------------------------------------------------------------

#[test]
fn test_large_counts_no_overflow() {
    // Verify that u64 can hold counts far beyond realistic usage
    let snap = CacheStatsSnapshot {
        kernel_cache_hits: u64::MAX / 2,
        kernel_cache_misses: u64::MAX / 2,
        msl_cache_hits: u64::MAX / 4,
        msl_cache_misses: u64::MAX / 4,
        pipeline_cache_hits: 1_000_000_000_000,
        pipeline_cache_misses: 1_000_000_000_000,
        total_dispatches: u64::MAX,
        total_compile_time_us: u64::MAX,
    };
    // hit_rate should not panic or produce NaN
    let rate = snap.hit_rate();
    assert!(rate.is_finite());
    assert!((0.0..=1.0).contains(&rate));
}

#[test]
fn test_large_kernel_hit_rate() {
    let snap = CacheStatsSnapshot {
        kernel_cache_hits: 1_000_000_000_000,
        kernel_cache_misses: 1,
        ..Default::default()
    };
    let rate = snap.kernel_hit_rate();
    assert!(rate > 0.999_999_999);
    assert!(rate <= 1.0);
}

#[test]
fn test_large_compile_time_avg() {
    let snap = CacheStatsSnapshot {
        pipeline_cache_misses: 1_000_000,
        total_compile_time_us: 1_000_000_000_000,
        ..Default::default()
    };
    let avg = snap.avg_compile_time_us();
    assert!((avg - 1_000_000.0).abs() < 1e-6);
}

#[test]
fn test_max_u64_hit_rate_50_percent() {
    // Both at max/2 should yield ~0.5
    let half = u64::MAX / 2;
    let snap = CacheStatsSnapshot {
        kernel_cache_hits: half,
        kernel_cache_misses: half,
        ..Default::default()
    };
    let rate = snap.kernel_hit_rate();
    assert!((rate - 0.5).abs() < 1e-10);
}

// ---------------------------------------------------------------
// Alternating hit/miss pattern
// ---------------------------------------------------------------

#[test]
fn test_alternating_hit_miss_pattern() {
    let stats = CacheStats::new();
    let iterations = 100;

    for _ in 0..iterations {
        stats.record_kernel_hit();
        stats.record_kernel_miss();
        stats.record_msl_hit();
        stats.record_msl_miss();
        stats.record_pipeline_hit();
        stats.record_pipeline_miss();
    }

    let snap = stats.snapshot();
    assert_eq!(snap.kernel_cache_hits, iterations as u64);
    assert_eq!(snap.kernel_cache_misses, iterations as u64);
    assert_eq!(snap.msl_cache_hits, iterations as u64);
    assert_eq!(snap.msl_cache_misses, iterations as u64);
    assert_eq!(snap.pipeline_cache_hits, iterations as u64);
    assert_eq!(snap.pipeline_cache_misses, iterations as u64);

    // Every level should be exactly 50%
    assert!((snap.kernel_hit_rate() - 0.5).abs() < 1e-10);
    assert!((snap.msl_hit_rate() - 0.5).abs() < 1e-10);
    assert!((snap.pipeline_hit_rate() - 0.5).abs() < 1e-10);
    assert!((snap.hit_rate() - 0.5).abs() < 1e-10);
}

#[test]
fn test_warmup_then_steady_state_pattern() {
    // Simulate: first N calls are misses (cold cache), then all hits
    let stats = CacheStats::new();
    let cold_misses = 10;
    let warm_hits = 90;

    for _ in 0..cold_misses {
        stats.record_kernel_miss();
    }
    for _ in 0..warm_hits {
        stats.record_kernel_hit();
    }

    let snap = stats.snapshot();
    assert_eq!(snap.kernel_cache_hits, warm_hits as u64);
    assert_eq!(snap.kernel_cache_misses, cold_misses as u64);
    assert!((snap.kernel_hit_rate() - 0.9).abs() < 1e-10);
}

// ---------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------

#[test]
fn test_global_singleton_identity() {
    let a = CacheStats::global();
    let b = CacheStats::global();
    assert!(std::ptr::eq(a, b));
}

// ---------------------------------------------------------------
// Snapshot Clone / Eq / Debug
// ---------------------------------------------------------------

#[test]
fn test_snapshot_clone_eq() {
    let snap = CacheStatsSnapshot {
        kernel_cache_hits: 42,
        kernel_cache_misses: 7,
        msl_cache_hits: 3,
        msl_cache_misses: 1,
        pipeline_cache_hits: 99,
        pipeline_cache_misses: 0,
        total_dispatches: 200,
        total_compile_time_us: 12345,
    };
    let cloned = snap.clone();
    assert_eq!(snap, cloned);
}

#[test]
fn test_snapshot_ne() {
    let snap_a = CacheStatsSnapshot {
        kernel_cache_hits: 1,
        ..Default::default()
    };
    let snap_b = CacheStatsSnapshot::default();
    assert_ne!(snap_a, snap_b);
}

#[test]
fn test_snapshot_debug_output() {
    let snap = CacheStatsSnapshot::default();
    let debug = format!("{snap:?}");
    assert!(debug.contains("CacheStatsSnapshot"));
    assert!(debug.contains("kernel_cache_hits"));
}

// ---------------------------------------------------------------
// Snapshot from recorded stats preserves hit rate through reset
// ---------------------------------------------------------------

#[test]
fn test_snapshot_survives_reset() {
    let stats = CacheStats::new();
    stats.record_kernel_hit();
    stats.record_kernel_hit();
    stats.record_kernel_miss();

    let snap_before = stats.snapshot();
    stats.reset();
    let snap_after = stats.snapshot();

    // The snapshot taken before reset retains its values
    assert_eq!(snap_before.kernel_cache_hits, 2);
    assert_eq!(snap_before.kernel_cache_misses, 1);

    // The snapshot after reset is zeroed
    assert_eq!(snap_after, CacheStatsSnapshot::default());
}

// ---------------------------------------------------------------
// Send + Sync compile-time assertions (documentation)
// ---------------------------------------------------------------

#[test]
fn test_cache_stats_is_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<CacheStats>();
    assert_sync::<CacheStats>();
}

// ---------------------------------------------------------------
// Helper function `hit_rate` edge cases
// ---------------------------------------------------------------

#[test]
fn test_helper_hit_rate_zero_total() {
    // Indirectly tested via snapshot methods, but verify the boundary
    let snap = CacheStatsSnapshot {
        kernel_cache_hits: 0,
        kernel_cache_misses: 0,
        ..Default::default()
    };
    assert_eq!(snap.kernel_hit_rate(), 0.0);
}

#[test]
fn test_helper_hit_rate_one_hit() {
    let snap = CacheStatsSnapshot {
        kernel_cache_hits: 1,
        kernel_cache_misses: 0,
        ..Default::default()
    };
    assert_eq!(snap.kernel_hit_rate(), 1.0);
}

#[test]
fn test_helper_hit_rate_one_miss() {
    let snap = CacheStatsSnapshot {
        kernel_cache_hits: 0,
        kernel_cache_misses: 1,
        ..Default::default()
    };
    assert_eq!(snap.kernel_hit_rate(), 0.0);
}

// ---------------------------------------------------------------
// record_compile with zero duration
// ---------------------------------------------------------------

#[test]
fn test_record_compile_zero_duration() {
    let stats = CacheStats::new();
    stats.record_compile(0);
    stats.record_compile(0);
    assert_eq!(stats.snapshot().total_compile_time_us, 0);
}

// ---------------------------------------------------------------
// Multiple snapshots from same stats (non-destructive read)
// ---------------------------------------------------------------

#[test]
fn test_multiple_snapshots_consistent() {
    let stats = CacheStats::new();
    stats.record_kernel_hit();
    stats.record_kernel_hit();

    let snap1 = stats.snapshot();
    let snap2 = stats.snapshot();

    // Without intervening writes, both snapshots should match
    assert_eq!(snap1, snap2);
}

#[test]
fn test_snapshot_does_not_consume() {
    let stats = CacheStats::new();
    stats.record_kernel_hit();

    let _ = stats.snapshot();
    stats.record_kernel_hit();

    // Snapshot is a read — counter should now be 2
    assert_eq!(stats.snapshot().kernel_cache_hits, 2);
}
