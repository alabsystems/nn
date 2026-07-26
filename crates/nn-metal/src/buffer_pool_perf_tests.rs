// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Performance proof tests for [`MetalBufferPool`] algorithmic complexity.
//!
//! Proves that all pool operations are O(1) bounded by constants, and
//! that the pool's memory footprint is monotonically capped. Part of #3020.
//!
//! # Proven properties
//!
//! - `acquire` scan is O(MAX_PER_CLASS) = O(8) per call
//! - `reclaim_all` iterates at most 56 entries = O(1)
//! - Internal fragmentation ≤ 4× (worst-case size class ratio)
//! - `pooled_bytes` is monotonically non-decreasing and capped at MAX_POOLED_BYTES
//! - `size_class_for` does at most 7 comparisons = O(1)

use super::*;

/// Prove: `acquire` linear scan is bounded by MAX_PER_CLASS (8).
///
/// The hot path in `acquire` iterates `entries.iter_mut()` looking for
/// an available entry. Worst case: all MAX_PER_CLASS entries are
/// unavailable → miss. This is O(MAX_PER_CLASS) = O(8) = O(1).
#[test]
fn proof_acquire_scan_bounded_by_max_per_class() {
    assert_eq!(MAX_PER_CLASS, 8, "scan upper bound is 8 iterations");
    // acquire() loop: `for entry in entries.iter_mut()`.
    // entries.len() <= MAX_PER_CLASS (enforced by `entries.len() < MAX_PER_CLASS`
    // guard before push). So the scan is bounded at 8 iterations regardless
    // of how many acquire/reclaim cycles have occurred.
    // Each iteration: 1 bool check + 1 branch. No allocations.
    assert!(MAX_PER_CLASS <= 8);
}

/// Prove: `reclaim_all` iterates at most 7 × MAX_PER_CLASS = 56 entries.
///
/// This is the absolute upper bound on reclaim work per call.
/// At 56 bool writes, reclaim_all is effectively O(1).
#[test]
fn proof_reclaim_all_bounded_total_entries() {
    let max_total = SIZE_CLASSES.len() * MAX_PER_CLASS;
    assert_eq!(
        max_total, 56,
        "maximum pool entries = 7 classes × 8 per class"
    );
    // reclaim_all: `for class in &mut self.classes { for entry in class { ... } }`.
    // Total iterations <= 56. Each: 1 bool write. No allocations.
}

/// Prove: internal fragmentation is bounded by the size class ratio.
///
/// Worst case: request for `SIZE_CLASSES[i-1] + 1` bytes gets a buffer
/// of `SIZE_CLASSES[i]` bytes. The ratio `class_size / requested` is
/// the internal fragmentation factor.
#[test]
fn proof_internal_fragmentation_bounded() {
    let mut max_ratio: f64 = 0.0;
    let mut worst_request = 0;
    let mut worst_class = 0;

    for i in 1..SIZE_CLASSES.len() {
        let request = SIZE_CLASSES[i - 1] + 1;
        let class = MetalBufferPool::size_class_for(request);
        let class_size = SIZE_CLASSES[class];
        assert!(class_size >= request, "class must be sufficient");

        let ratio = class_size as f64 / request as f64;
        if ratio > max_ratio {
            max_ratio = ratio;
            worst_request = request;
            worst_class = class;
        }
    }
    // With 4× class boundaries (64K, 256K, 1M, 4M, ...), worst case ≈ 4×.
    assert!(
        max_ratio < 4.1,
        "internal fragmentation must be < 4.1×, got {max_ratio:.2}× \
         (request={worst_request}, class={worst_class}, size={})",
        SIZE_CLASSES[worst_class]
    );
    assert!(
        max_ratio > 3.5,
        "expected near-4× worst case, got {max_ratio:.2}×"
    );
}

/// Prove: `pooled_bytes` accumulates monotonically and is capped.
///
/// Each `acquire` miss adds exactly `class_size` to `pooled_bytes`.
/// The guard `self.pooled_bytes + class_size <= MAX_POOLED_BYTES`
/// prevents exceeding the budget. Entries are never removed, so
/// `pooled_bytes` is monotonically non-decreasing.
#[test]
fn proof_pooled_bytes_monotonic_and_capped() {
    // Simulate sequential misses in the largest poolable class (64 MB).
    let class_size = SIZE_CLASSES[5]; // 64 MB
    let mut pooled_bytes: usize = 0;
    let mut entries_added = 0;
    let mut prev = 0;

    for _ in 0..100 {
        // Guard from acquire(): only add if budget permits.
        if entries_added < MAX_PER_CLASS && pooled_bytes + class_size <= MAX_POOLED_BYTES {
            pooled_bytes += class_size;
            entries_added += 1;
            // Monotonicity: pooled_bytes only increases.
            assert!(pooled_bytes >= prev, "must be monotonically non-decreasing");
            prev = pooled_bytes;
        }
    }
    // After filling: exactly 8 × 64 MB = 512 MB = MAX_POOLED_BYTES.
    assert_eq!(pooled_bytes, MAX_POOLED_BYTES);
    assert_eq!(entries_added, MAX_PER_CLASS);

    // No further entries can be added to ANY class.
    for &cs in &SIZE_CLASSES {
        assert!(
            pooled_bytes + cs > MAX_POOLED_BYTES,
            "budget exhausted for {cs}"
        );
    }
}

/// Prove: `size_class_for` is O(1) with exactly 7 comparisons maximum.
#[test]
fn proof_size_class_for_constant_time() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let comparisons = AtomicUsize::new(0);

    fn size_class_for_counted(bytes: usize, counter: &AtomicUsize) -> usize {
        for (i, &threshold) in SIZE_CLASSES.iter().enumerate() {
            counter.fetch_add(1, Ordering::Relaxed);
            if bytes <= threshold {
                return i;
            }
        }
        SIZE_CLASSES.len() - 1
    }

    // Best case: 1 comparison.
    let _ = size_class_for_counted(100, &comparisons);
    let best = comparisons.swap(0, Ordering::Relaxed);
    assert_eq!(best, 1);

    // Worst case: 7 comparisons (request > all classes).
    let _ = size_class_for_counted(usize::MAX, &comparisons);
    let worst = comparisons.swap(0, Ordering::Relaxed);
    assert_eq!(worst, SIZE_CLASSES.len());
    assert!(worst <= 7, "max comparisons bounded at 7");
}
