// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`LazyBufferPool`] and liveness analysis.

use super::*;

// ---------------------------------------------------------------------------
// Size class selection
// ---------------------------------------------------------------------------

#[test]
fn test_size_class_for_zero_returns_none() {
    assert_eq!(LazyBufferPool::size_class_for(0), None);
}

#[test]
fn test_size_class_for_small_request() {
    let (class, bytes) = LazyBufferPool::size_class_for(100).unwrap();
    assert_eq!(class, 0);
    assert_eq!(bytes, 4 * 1024);
}

#[test]
fn test_size_class_for_exact_boundaries() {
    let expected = [
        (4 * 1024, 0),
        (16 * 1024, 1),
        (64 * 1024, 2),
        (256 * 1024, 3),
        (1024 * 1024, 4),
        (4 * 1024 * 1024, 5),
        (16 * 1024 * 1024, 6),
    ];
    for (size, expected_class) in expected {
        let (class, class_bytes) = LazyBufferPool::size_class_for(size).unwrap();
        assert_eq!(class, expected_class, "size={size}");
        assert_eq!(class_bytes, size, "size={size}");
    }
}

#[test]
fn test_size_class_for_one_over_boundary() {
    // 4 KB + 1 -> class 1 (16 KB).
    let (class, bytes) = LazyBufferPool::size_class_for(4 * 1024 + 1).unwrap();
    assert_eq!(class, 1);
    assert_eq!(bytes, 16 * 1024);
}

#[test]
fn test_size_class_for_oversized() {
    assert!(LazyBufferPool::size_class_for(16 * 1024 * 1024 + 1).is_none());
    assert!(LazyBufferPool::size_class_for(64 * 1024 * 1024).is_none());
    assert!(LazyBufferPool::size_class_for(usize::MAX).is_none());
}

// ---------------------------------------------------------------------------
// Buffer pool alloc/free
// ---------------------------------------------------------------------------

#[test]
fn test_alloc_basic() {
    let mut pool = LazyBufferPool::new();
    let h = pool.alloc(5000);
    assert_eq!(h.class(), 1); // 5000 bytes -> class 1 (16 KB).
    assert_eq!(h.alloc_bytes(), 16 * 1024);
    assert_eq!(h.id(), 0);
}

#[test]
fn test_alloc_zero_bytes() {
    let mut pool = LazyBufferPool::new();
    let h = pool.alloc(0);
    assert_eq!(h.class(), usize::MAX);
    assert_eq!(h.alloc_bytes(), 0);

    let stats = pool.stats();
    assert_eq!(stats.zero_allocs, 1);
    assert_eq!(stats.total_allocs, 1);
    assert_eq!(stats.peak_bytes, 0);
}

#[test]
fn test_alloc_oversized() {
    let mut pool = LazyBufferPool::new();
    let h = pool.alloc(32 * 1024 * 1024);
    assert_eq!(h.class(), usize::MAX);
    assert_eq!(h.alloc_bytes(), 32 * 1024 * 1024);

    let stats = pool.stats();
    assert_eq!(stats.oversized_allocs, 1);
    assert_eq!(stats.peak_bytes, 32 * 1024 * 1024);
}

#[test]
fn test_free_returns_true_for_pooled() {
    let mut pool = LazyBufferPool::new();
    let h = pool.alloc(1000);
    assert!(pool.free(h));

    let stats = pool.stats();
    assert_eq!(stats.total_frees, 1);
    assert_eq!(stats.current_bytes, 0);
}

#[test]
fn test_free_returns_false_for_zero() {
    let mut pool = LazyBufferPool::new();
    let h = pool.alloc(0);
    assert!(!pool.free(h));
}

#[test]
fn test_free_returns_false_for_oversized() {
    let mut pool = LazyBufferPool::new();
    let h = pool.alloc(32 * 1024 * 1024);
    assert!(!pool.free(h));
}

// ---------------------------------------------------------------------------
// Buffer reuse
// ---------------------------------------------------------------------------

#[test]
fn test_reuse_when_sizes_match_class() {
    let mut pool = LazyBufferPool::new();

    // Allocate class 0 (4 KB), free it, allocate again -> pool hit.
    let h1 = pool.alloc(1000);
    assert_eq!(h1.class(), 0);
    pool.free(h1);

    let h2 = pool.alloc(2000); // Still fits class 0.
    assert_eq!(h2.class(), 0);

    let stats = pool.stats();
    assert_eq!(stats.pool_hits, 1);
    assert_eq!(stats.pool_misses, 1); // First alloc was a miss.
    assert_eq!(stats.total_allocs, 2);
}

#[test]
fn test_no_cross_class_reuse() {
    let mut pool = LazyBufferPool::new();

    // Free a class 0 buffer.
    let h = pool.alloc(1000);
    pool.free(h);

    // Allocate class 2 (64 KB) -> should NOT reuse class 0.
    let h2 = pool.alloc(20_000);
    assert_eq!(h2.class(), 2);

    let stats = pool.stats();
    assert_eq!(stats.pool_hits, 0);
    assert_eq!(stats.pool_misses, 2);
}

#[test]
fn test_multiple_reuses_same_class() {
    let mut pool = LazyBufferPool::new();

    // Allocate 3, free all, re-allocate all -> 3 hits.
    let handles: Vec<_> = (0..3).map(|_| pool.alloc(500)).collect();
    for h in handles {
        pool.free(h);
    }

    for _ in 0..3 {
        let h = pool.alloc(500);
        assert_eq!(h.class(), 0);
    }

    let stats = pool.stats();
    assert_eq!(stats.pool_hits, 3);
    assert_eq!(stats.pool_misses, 3);
}

// ---------------------------------------------------------------------------
// Peak memory tracking
// ---------------------------------------------------------------------------

#[test]
fn test_peak_memory_single_alloc() {
    let mut pool = LazyBufferPool::new();
    pool.alloc(100); // 4 KB class.

    assert_eq!(pool.peak_bytes(), 4 * 1024);
    assert_eq!(pool.current_bytes(), 4 * 1024);
}

#[test]
fn test_peak_memory_alloc_free_cycle() {
    let mut pool = LazyBufferPool::new();

    // Allocate 2 x class 4 (1 MB each) = 2 MB peak.
    let h1 = pool.alloc(500_000);
    let h2 = pool.alloc(500_000);
    assert_eq!(pool.peak_bytes(), 2 * 1024 * 1024);

    // Free one -> current drops but peak stays.
    pool.free(h1);
    assert_eq!(pool.peak_bytes(), 2 * 1024 * 1024);
    assert_eq!(pool.current_bytes(), 1024 * 1024);

    // Free second.
    pool.free(h2);
    assert_eq!(pool.peak_bytes(), 2 * 1024 * 1024);
    assert_eq!(pool.current_bytes(), 0);
}

#[test]
fn test_peak_memory_with_reuse() {
    let mut pool = LazyBufferPool::new();

    // Alloc 1 MB, free, alloc 1 MB again -> peak is still 1 MB (reuse).
    let h = pool.alloc(500_000);
    pool.free(h);
    let _h2 = pool.alloc(500_000);

    assert_eq!(pool.peak_bytes(), 1024 * 1024);
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

#[test]
fn test_stats_display() {
    let mut pool = LazyBufferPool::new();
    pool.alloc(100);
    pool.alloc(5000);

    let stats = pool.stats();
    let display = format!("{stats}");
    assert!(display.contains("LazyBufferPool Stats"));
    assert!(display.contains("reuse rate"));
}

#[test]
fn test_stats_reuse_rate() {
    let stats = LazyPoolStats {
        total_allocs: 10,
        pool_hits: 4,
        pool_misses: 6,
        total_frees: 0,
        oversized_allocs: 0,
        zero_allocs: 0,
        peak_bytes: 0,
        current_bytes: 0,
        free_pool_bytes: 0,
        free_pool_count: 0,
        per_class_free: [0; NUM_CLASSES],
    };
    assert!((stats.reuse_rate() - 0.4).abs() < 1e-9);
}

#[test]
fn test_stats_reuse_rate_zero_allocs() {
    let stats = LazyPoolStats {
        total_allocs: 0,
        pool_hits: 0,
        pool_misses: 0,
        total_frees: 0,
        oversized_allocs: 0,
        zero_allocs: 0,
        peak_bytes: 0,
        current_bytes: 0,
        free_pool_bytes: 0,
        free_pool_count: 0,
        per_class_free: [0; NUM_CLASSES],
    };
    assert!((stats.reuse_rate() - 0.0).abs() < 1e-9);
}

#[test]
fn test_reset_clears_stats() {
    let mut pool = LazyBufferPool::new();
    pool.alloc(100);
    pool.alloc(5000);
    pool.reset();

    let stats = pool.stats();
    assert_eq!(stats.total_allocs, 0);
    assert_eq!(stats.pool_hits, 0);
    assert_eq!(stats.pool_misses, 0);
    assert_eq!(stats.peak_bytes, 0);
    assert_eq!(stats.current_bytes, 0);
}

// ---------------------------------------------------------------------------
// Liveness analysis: simple linear chain
// ---------------------------------------------------------------------------

#[test]
fn test_liveness_empty_plan() {
    let analysis = compute_liveness(&[]);
    assert_eq!(analysis.buffers.len(), 0);
    assert_eq!(analysis.peak_live_count, 0);
    assert_eq!(analysis.peak_live_bytes, 0);
    assert_eq!(analysis.total_bytes_no_reuse, 0);
}

#[test]
fn test_liveness_linear_chain() {
    // Linear chain: step 0 -> step 1 -> step 2 -> step 3
    let steps = vec![
        DispatchStepDesc { input_indices: vec![], output_bytes: 1024 },
        DispatchStepDesc { input_indices: vec![0], output_bytes: 2048 },
        DispatchStepDesc { input_indices: vec![1], output_bytes: 4096 },
        DispatchStepDesc { input_indices: vec![2], output_bytes: 1024 },
    ];

    let analysis = compute_liveness(&steps);
    assert_eq!(analysis.buffers.len(), 4);

    // Buffer 0: produced at 0, last used at 1.
    assert_eq!(analysis.buffers[0].last_use, Some(1));
    // Buffer 1: produced at 1, last used at 2.
    assert_eq!(analysis.buffers[1].last_use, Some(2));
    // Buffer 2: produced at 2, last used at 3.
    assert_eq!(analysis.buffers[2].last_use, Some(3));
    // Buffer 3: produced at 3, never read.
    assert_eq!(analysis.buffers[3].last_use, None);

    // In a linear chain, at most 2 buffers are live simultaneously.
    assert_eq!(analysis.peak_live_count, 2);

    // Total without reuse: 1024 + 2048 + 4096 + 1024 = 8192.
    assert_eq!(analysis.total_bytes_no_reuse, 8192);

    // Peak live bytes: step 2 produces 4096 while 2048 is still live = 6144.
    assert_eq!(analysis.peak_live_bytes, 6144);
}

#[test]
fn test_liveness_diamond() {
    // Diamond pattern:
    //   step 0 -> step 1
    //   step 0 -> step 2
    //   step 1, step 2 -> step 3
    let steps = vec![
        DispatchStepDesc { input_indices: vec![], output_bytes: 1024 },
        DispatchStepDesc { input_indices: vec![0], output_bytes: 2048 },
        DispatchStepDesc { input_indices: vec![0], output_bytes: 2048 },
        DispatchStepDesc { input_indices: vec![1, 2], output_bytes: 1024 },
    ];

    let analysis = compute_liveness(&steps);

    // Buffer 0: last used at step 2.
    assert_eq!(analysis.buffers[0].last_use, Some(2));
    // Buffer 1: last used at step 3.
    assert_eq!(analysis.buffers[1].last_use, Some(3));
    // Buffer 2: last used at step 3.
    assert_eq!(analysis.buffers[2].last_use, Some(3));
    // Buffer 3: never read.
    assert_eq!(analysis.buffers[3].last_use, None);

    // At step 2: buffers 0, 1, 2 are all live -> 3 live.
    assert_eq!(analysis.peak_live_count, 3);
}

#[test]
fn test_liveness_savings_ratio() {
    // Linear chain where each step is 1 KB.
    let steps: Vec<_> = (0..10)
        .map(|i| DispatchStepDesc {
            input_indices: if i > 0 { vec![i - 1] } else { vec![] },
            output_bytes: 1024,
        })
        .collect();

    let analysis = compute_liveness(&steps);
    assert_eq!(analysis.total_bytes_no_reuse, 10240);
    assert_eq!(analysis.peak_live_bytes, 2048);
    assert!((analysis.savings_ratio() - 0.8).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// Simulate lazy alloc
// ---------------------------------------------------------------------------

#[test]
fn test_simulate_lazy_alloc_linear() {
    let steps: Vec<_> = (0..5)
        .map(|i| DispatchStepDesc {
            input_indices: if i > 0 { vec![i - 1] } else { vec![] },
            output_bytes: 4 * 1024, // Exactly class 0.
        })
        .collect();

    let (liveness, stats) = simulate_lazy_alloc(&steps);

    assert_eq!(liveness.peak_live_count, 2);
    assert_eq!(stats.total_allocs, 5);
    // Steps 2, 3, 4 reuse freed buffers -> 3 hits.
    assert_eq!(stats.pool_hits, 3);
    assert_eq!(stats.pool_misses, 2);
}

#[test]
fn test_simulate_lazy_alloc_empty() {
    let (liveness, stats) = simulate_lazy_alloc(&[]);
    assert_eq!(liveness.buffers.len(), 0);
    assert_eq!(stats.total_allocs, 0);
}

#[test]
fn test_simulate_lazy_alloc_no_reuse_opportunity() {
    // All buffers read at the last step -> no early frees, no reuse.
    let steps = vec![
        DispatchStepDesc { input_indices: vec![], output_bytes: 1024 },
        DispatchStepDesc { input_indices: vec![], output_bytes: 1024 },
        DispatchStepDesc { input_indices: vec![], output_bytes: 1024 },
        DispatchStepDesc { input_indices: vec![0, 1, 2], output_bytes: 1024 },
    ];

    let (liveness, stats) = simulate_lazy_alloc(&steps);
    assert_eq!(liveness.peak_live_count, 4);
    assert_eq!(stats.pool_hits, 0);
    assert_eq!(stats.pool_misses, 4);
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_alloc_ids_are_unique() {
    let mut pool = LazyBufferPool::new();
    let h1 = pool.alloc(100);
    let h2 = pool.alloc(100);
    let h3 = pool.alloc(0);
    assert_ne!(h1.id(), h2.id());
    assert_ne!(h2.id(), h3.id());
}

#[test]
fn test_free_list_capacity_limit() {
    let mut pool = LazyBufferPool::new();

    // Allocate MAX_FREE_PER_CLASS + 1 buffers first, then free all.
    let handles: Vec<_> = (0..=MAX_FREE_PER_CLASS).map(|_| pool.alloc(100)).collect();
    for h in handles {
        pool.free(h);
    }

    let stats = pool.stats();
    // Free list capped at MAX_FREE_PER_CLASS.
    assert_eq!(stats.per_class_free[0], MAX_FREE_PER_CLASS);
}

#[test]
fn test_liveness_single_step() {
    let steps = vec![DispatchStepDesc {
        input_indices: vec![],
        output_bytes: 4096,
    }];

    let analysis = compute_liveness(&steps);
    assert_eq!(analysis.buffers.len(), 1);
    assert_eq!(analysis.buffers[0].last_use, None);
    assert_eq!(analysis.peak_live_count, 1);
    assert_eq!(analysis.peak_live_bytes, 4096);
    assert_eq!(analysis.total_bytes_no_reuse, 4096);
    assert!((analysis.savings_ratio() - 0.0).abs() < 1e-9);
}

#[test]
fn test_liveness_self_referencing_ignored() {
    // A step referencing a buffer index beyond the plan size is ignored.
    let steps = vec![
        DispatchStepDesc { input_indices: vec![99], output_bytes: 1024 },
        DispatchStepDesc { input_indices: vec![0], output_bytes: 1024 },
    ];

    let analysis = compute_liveness(&steps);
    assert_eq!(analysis.buffers[0].last_use, Some(1));
    assert_eq!(analysis.buffers[1].last_use, None);
}

#[test]
fn test_liveness_dead_output() {
    // Step 0 produces output but no one reads it (dead).
    // Step 1 also produces output, no readers.
    let steps = vec![
        DispatchStepDesc { input_indices: vec![], output_bytes: 1024 },
        DispatchStepDesc { input_indices: vec![], output_bytes: 2048 },
    ];

    let analysis = compute_liveness(&steps);
    assert_eq!(analysis.buffers[0].last_use, None);
    assert_eq!(analysis.buffers[1].last_use, None);
    // Both are never freed in the sweep, so both are live at step 1.
    assert_eq!(analysis.peak_live_count, 2);
    assert_eq!(analysis.peak_live_bytes, 3072);
}

#[test]
fn test_pool_default_trait() {
    let pool = LazyBufferPool::default();
    let stats = pool.stats();
    assert_eq!(stats.total_allocs, 0);
}
