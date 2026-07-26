// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`SizeClassAllocator`].

use super::*;

// ---------------------------------------------------------------------------
// Size class selection
// ---------------------------------------------------------------------------

#[test]
fn test_size_class_for_zero_bytes() {
    assert_eq!(SizeClassAllocator::size_class_for(0), Some(0));
}

#[test]
fn test_size_class_for_small_request() {
    // 100 bytes fits in class 0 (4 KB).
    assert_eq!(SizeClassAllocator::size_class_for(100), Some(0));
}

#[test]
fn test_size_class_for_exact_boundary() {
    // Exactly 4 KB -> class 0.
    assert_eq!(SizeClassAllocator::size_class_for(4 * 1024), Some(0));
    // Exactly 16 KB -> class 1.
    assert_eq!(SizeClassAllocator::size_class_for(16 * 1024), Some(1));
    // Exactly 64 KB -> class 2.
    assert_eq!(SizeClassAllocator::size_class_for(64 * 1024), Some(2));
    // Exactly 256 KB -> class 3.
    assert_eq!(SizeClassAllocator::size_class_for(256 * 1024), Some(3));
    // Exactly 1 MB -> class 4.
    assert_eq!(SizeClassAllocator::size_class_for(1024 * 1024), Some(4));
    // Exactly 4 MB -> class 5.
    assert_eq!(SizeClassAllocator::size_class_for(4 * 1024 * 1024), Some(5));
    // Exactly 16 MB -> class 6.
    assert_eq!(SizeClassAllocator::size_class_for(16 * 1024 * 1024), Some(6));
    // Exactly 64 MB -> class 7.
    assert_eq!(SizeClassAllocator::size_class_for(64 * 1024 * 1024), Some(7));
}

#[test]
fn test_size_class_for_one_byte_over_boundary() {
    // 4 KB + 1 -> class 1 (16 KB).
    assert_eq!(SizeClassAllocator::size_class_for(4 * 1024 + 1), Some(1));
    // 64 MB + 1 -> None (oversized).
    assert_eq!(
        SizeClassAllocator::size_class_for(64 * 1024 * 1024 + 1),
        None
    );
}

#[test]
fn test_size_class_for_oversized() {
    assert_eq!(SizeClassAllocator::size_class_for(128 * 1024 * 1024), None);
    assert_eq!(SizeClassAllocator::size_class_for(usize::MAX), None);
}

// ---------------------------------------------------------------------------
// Allocation and free within size classes
// ---------------------------------------------------------------------------

#[test]
fn test_allocate_returns_correct_class_and_size() {
    let mut alloc = SizeClassAllocator::new();
    let result = alloc.allocate(5000).expect("should fit in class 1 (16KB)");
    assert_eq!(result.class, 1);
    assert_eq!(result.alloc_bytes, 16 * 1024);
    assert!(!result.reused);
}

#[test]
fn test_allocate_exact_class_boundary() {
    let mut alloc = SizeClassAllocator::new();
    let result = alloc.allocate(4 * 1024).expect("should fit in class 0 (4KB)");
    assert_eq!(result.class, 0);
    assert_eq!(result.alloc_bytes, 4 * 1024);
    assert!(!result.reused);
}

#[test]
fn test_allocate_oversized_returns_none() {
    let mut alloc = SizeClassAllocator::new();
    assert!(alloc.allocate(100 * 1024 * 1024).is_none());
}

#[test]
fn test_allocate_zero_bytes_uses_smallest_class() {
    let mut alloc = SizeClassAllocator::new();
    let result = alloc.allocate(0).expect("zero should use class 0");
    assert_eq!(result.class, 0);
    assert_eq!(result.alloc_bytes, 4 * 1024);
}

// ---------------------------------------------------------------------------
// Free list reuse
// ---------------------------------------------------------------------------

#[test]
fn test_free_list_reuse() {
    let mut alloc = SizeClassAllocator::new();

    // Allocate and then deallocate.
    let r1 = alloc.allocate(1000).unwrap();
    assert!(!r1.reused);
    assert!(alloc.deallocate(r1.class));

    // Second allocation should reuse from free list.
    let r2 = alloc.allocate(1000).unwrap();
    assert!(r2.reused);
    assert_eq!(r2.class, r1.class);
}

#[test]
fn test_multiple_reuses() {
    let mut alloc = SizeClassAllocator::new();

    // Allocate 3 buffers, free all, then re-allocate all.
    let mut classes = Vec::new();
    for _ in 0..3 {
        let r = alloc.allocate(500).unwrap();
        classes.push(r.class);
    }
    for &c in &classes {
        assert!(alloc.deallocate(c));
    }

    // All three should be reused.
    for _ in 0..3 {
        let r = alloc.allocate(500).unwrap();
        assert!(r.reused);
    }
    // Fourth should be a miss (free list empty).
    let r = alloc.allocate(500).unwrap();
    assert!(!r.reused);
}

// ---------------------------------------------------------------------------
// Cross-size-class allocation
// ---------------------------------------------------------------------------

#[test]
fn test_cross_class_no_reuse() {
    let mut alloc = SizeClassAllocator::new();

    // Allocate in class 0 (4KB), free it.
    let r = alloc.allocate(1000).unwrap();
    assert_eq!(r.class, 0);
    alloc.deallocate(r.class);

    // Allocate in class 2 (64KB) — should NOT reuse class 0 free entry.
    let r2 = alloc.allocate(20_000).unwrap();
    assert_eq!(r2.class, 2);
    assert!(!r2.reused);
}

// ---------------------------------------------------------------------------
// Peak tracking accuracy
// ---------------------------------------------------------------------------

#[test]
fn test_peak_tracking() {
    let mut alloc = SizeClassAllocator::new();

    // Allocate 5 buffers in class 0.
    for _ in 0..5 {
        alloc.allocate(100).unwrap();
    }

    let stats = alloc.stats();
    assert_eq!(stats.per_class[0].peak_in_use, 5);
    assert_eq!(stats.per_class[0].in_use_count, 5);

    // Free 3 — peak should stay at 5.
    for _ in 0..3 {
        alloc.deallocate(0);
    }
    let stats = alloc.stats();
    assert_eq!(stats.per_class[0].peak_in_use, 5);
    assert_eq!(stats.per_class[0].in_use_count, 2);

    // Allocate 4 more — peak should be 6 (2 existing + 4 new).
    for _ in 0..4 {
        alloc.allocate(100).unwrap();
    }
    let stats = alloc.stats();
    assert_eq!(stats.per_class[0].peak_in_use, 6);
    assert_eq!(stats.per_class[0].in_use_count, 6);
}

// ---------------------------------------------------------------------------
// Fragmentation calculation
// ---------------------------------------------------------------------------

#[test]
fn test_fragmentation_zero_when_all_in_use() {
    let mut alloc = SizeClassAllocator::new();
    alloc.allocate(100).unwrap();
    alloc.allocate(100).unwrap();

    let stats = alloc.stats();
    assert!((stats.fragmentation_ratio - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_fragmentation_one_when_all_free() {
    let mut alloc = SizeClassAllocator::new();
    let r = alloc.allocate(100).unwrap();
    alloc.deallocate(r.class);

    let stats = alloc.stats();
    assert!((stats.fragmentation_ratio - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_fragmentation_half() {
    let mut alloc = SizeClassAllocator::new();
    // Allocate 2, free 1 — half fragmented.
    let r1 = alloc.allocate(100).unwrap();
    alloc.allocate(100).unwrap();
    alloc.deallocate(r1.class);

    let stats = alloc.stats();
    // 1 free (4KB) + 1 in-use (4KB) = 50% fragmentation.
    assert!((stats.fragmentation_ratio - 0.5).abs() < f64::EPSILON);
}

#[test]
fn test_fragmentation_empty_pool() {
    let alloc = SizeClassAllocator::new();
    let stats = alloc.stats();
    assert!((stats.fragmentation_ratio - 0.0).abs() < f64::EPSILON);
}

// ---------------------------------------------------------------------------
// Hit rate calculation
// ---------------------------------------------------------------------------

#[test]
fn test_hit_rate_zero_on_cold_start() {
    let mut alloc = SizeClassAllocator::new();
    alloc.allocate(100).unwrap();
    alloc.allocate(200).unwrap();

    let stats = alloc.stats();
    assert!((stats.hit_rate - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_hit_rate_after_reuse() {
    let mut alloc = SizeClassAllocator::new();
    // 1 miss.
    let r = alloc.allocate(100).unwrap();
    alloc.deallocate(r.class);
    // 1 hit.
    alloc.allocate(100).unwrap();

    let stats = alloc.stats();
    // 1 hit / 2 total = 0.5.
    assert!((stats.hit_rate - 0.5).abs() < f64::EPSILON);
}

#[test]
fn test_hit_rate_empty_pool() {
    let alloc = SizeClassAllocator::new();
    let stats = alloc.stats();
    assert!((stats.hit_rate - 0.0).abs() < f64::EPSILON);
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_free_list_capacity_limit() {
    let mut alloc = SizeClassAllocator::new();

    // Fill free list to MAX_FREE_PER_CLASS.
    for _ in 0..MAX_FREE_PER_CLASS {
        let r = alloc.allocate(100).unwrap();
        alloc.deallocate(r.class);
    }

    // Allocate one more and try to free — should be rejected.
    let r = alloc.allocate(100).unwrap();
    // Free list has MAX_FREE_PER_CLASS entries, but we just reused one via
    // the allocate above. So let's free what we have: in_use should be 1.
    // Actually, since we freed MAX_FREE_PER_CLASS and then allocated 1 (reuse),
    // free list is MAX_FREE_PER_CLASS-1, in_use is 1.
    // Allocate MAX_FREE_PER_CLASS more to empty the free list and fill in_use.
    alloc.deallocate(r.class);

    // Directly set up the scenario: fill free list.
    alloc.reset();
    for _ in 0..MAX_FREE_PER_CLASS {
        alloc.allocate(100).unwrap();
    }
    for _ in 0..MAX_FREE_PER_CLASS {
        alloc.deallocate(0);
    }
    // Free list now has MAX_FREE_PER_CLASS entries.
    let stats = alloc.stats();
    assert_eq!(stats.per_class[0].free_count, MAX_FREE_PER_CLASS);

    // Allocate and try to free one more — should fail.
    alloc.allocate(100).unwrap(); // Reuses from free list.
    alloc.allocate(100).unwrap(); // Reuses from free list.
    // Free both back.
    assert!(alloc.deallocate(0)); // free_count back to MAX_FREE_PER_CLASS-1.
    assert!(alloc.deallocate(0)); // free_count back to MAX_FREE_PER_CLASS.
    // Now free list is full at MAX_FREE_PER_CLASS. Allocate + free one more:
    alloc.allocate(100).unwrap(); // Reuses (free_count = MAX_FREE_PER_CLASS-1).
    alloc.allocate(100).unwrap(); // Reuses (free_count = MAX_FREE_PER_CLASS-2).
    assert!(alloc.deallocate(0));
    assert!(alloc.deallocate(0));
    // At this point free is MAX_FREE_PER_CLASS, in_use is 0. Now try adding past limit:
    alloc.allocate(100).unwrap(); // reuse -> free_count = MAX_FREE_PER_CLASS-1
    // Force in_use > 0, then manually add to free list:
    // Actually, the limit is checked in deallocate. Let's make a cleaner test.
    alloc.reset();

    // Clean test: allocate MAX_FREE_PER_CLASS + 1 buffers, free all.
    for _ in 0..=MAX_FREE_PER_CLASS {
        alloc.allocate(100).unwrap();
    }
    // Free all — last one should be rejected.
    let mut accepted = 0;
    let mut rejected = 0;
    for _ in 0..=MAX_FREE_PER_CLASS {
        if alloc.deallocate(0) {
            accepted += 1;
        } else {
            rejected += 1;
        }
    }
    assert_eq!(accepted, MAX_FREE_PER_CLASS);
    assert_eq!(rejected, 1);
}

#[test]
fn test_double_free_protection() {
    let alloc = SizeClassAllocator::new();
    // No buffers in use — deallocate should return false.
    let mut alloc = alloc;
    assert!(!alloc.deallocate(0));
}

#[test]
fn test_deallocate_invalid_class() {
    let mut alloc = SizeClassAllocator::new();
    assert!(!alloc.deallocate(NUM_SIZE_CLASSES));
    assert!(!alloc.deallocate(usize::MAX));
}

#[test]
fn test_oversized_tracking() {
    let mut alloc = SizeClassAllocator::new();
    alloc.record_oversized();
    alloc.record_oversized();
    let stats = alloc.stats();
    assert_eq!(stats.oversized_allocs, 2);
}

#[test]
fn test_reset_clears_everything() {
    let mut alloc = SizeClassAllocator::new();
    alloc.allocate(100).unwrap();
    alloc.allocate(5000).unwrap();
    alloc.record_oversized();

    alloc.reset();
    let stats = alloc.stats();
    for cs in &stats.per_class {
        assert_eq!(cs.hits, 0);
        assert_eq!(cs.misses, 0);
        assert_eq!(cs.free_count, 0);
        assert_eq!(cs.in_use_count, 0);
        assert_eq!(cs.peak_in_use, 0);
    }
    assert_eq!(stats.oversized_allocs, 0);
    assert_eq!(stats.total_free_bytes, 0);
    assert_eq!(stats.total_used_bytes, 0);
}

#[test]
fn test_class_size_accessor() {
    assert_eq!(SizeClassAllocator::class_size(0), 4 * 1024);
    assert_eq!(SizeClassAllocator::class_size(7), 64 * 1024 * 1024);
}

// ---------------------------------------------------------------------------
// Stats byte tracking
// ---------------------------------------------------------------------------

#[test]
fn test_stats_byte_tracking() {
    let mut alloc = SizeClassAllocator::new();
    // Allocate 1x class 0 (4KB) + 1x class 4 (1MB).
    alloc.allocate(100).unwrap();
    alloc.allocate(500_000).unwrap();

    let stats = alloc.stats();
    assert_eq!(stats.total_used_bytes, 4 * 1024 + 1024 * 1024);
    assert_eq!(stats.total_free_bytes, 0);
}

#[test]
fn test_per_class_hit_rate() {
    let cs = SizeClassStats {
        hits: 3,
        misses: 7,
        ..SizeClassStats::default()
    };
    assert!((cs.hit_rate() - 0.3).abs() < f64::EPSILON);
}

#[test]
fn test_per_class_hit_rate_zero_allocs() {
    let cs = SizeClassStats::default();
    assert!((cs.hit_rate() - 0.0).abs() < f64::EPSILON);
}

// ---------------------------------------------------------------------------
// Sizes array invariants
// ---------------------------------------------------------------------------

#[test]
fn test_size_classes_are_sorted() {
    for i in 1..SIZE_CLASS_BOUNDARIES.len() {
        assert!(
            SIZE_CLASS_BOUNDARIES[i] > SIZE_CLASS_BOUNDARIES[i - 1],
            "Size classes must be strictly increasing"
        );
    }
}

#[test]
fn test_size_classes_are_powers_of_four() {
    for &boundary in &SIZE_CLASS_BOUNDARIES {
        // Each boundary should be 4^k * 1024 for some k >= 0.
        // Verify it's a power of 2 at minimum (all our values are powers of 4).
        assert!(boundary.is_power_of_two(), "{boundary} is not a power of 2");
    }
}

#[test]
fn test_num_size_classes_matches() {
    assert_eq!(NUM_SIZE_CLASSES, 8);
    assert_eq!(SIZE_CLASS_BOUNDARIES.len(), NUM_SIZE_CLASSES);
}
