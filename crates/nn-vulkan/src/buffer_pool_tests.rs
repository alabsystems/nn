// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for Vulkan buffer pool and buffer management.
//!
//! Tests VulkanBuffer, StagingBuffer, BufferUsage, and BufferPool
//! edge cases beyond the inline unit tests in buffer_pool.rs.

use crate::buffer::{BufferUsage, StagingBuffer, VulkanBuffer};
use crate::buffer_pool::{BufferPool, BufferPoolStats, PoolStats, SizeClassStats};

// ---------------------------------------------------------------------------
// VulkanBuffer tests
// ---------------------------------------------------------------------------

#[test]
fn test_vulkan_buffer_creation_and_accessors() {
    let buf = VulkanBuffer::new(4096, BufferUsage::StorageReadWrite).expect("should create");
    assert_eq!(buf.size_bytes(), 4096);
    assert_eq!(buf.usage(), BufferUsage::StorageReadWrite);
    assert_eq!(buf.handle(), 0); // placeholder handle
}

#[test]
fn test_vulkan_buffer_zero_size_rejected() {
    let result = VulkanBuffer::new(0, BufferUsage::StorageRead);
    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("size must be > 0"), "error: {msg}");
}

#[test]
fn test_vulkan_buffer_each_usage_variant() {
    let usages = [
        BufferUsage::StorageRead,
        BufferUsage::StorageReadWrite,
        BufferUsage::Uniform,
        BufferUsage::TransferSrc,
        BufferUsage::TransferDst,
    ];
    for usage in usages {
        let buf = VulkanBuffer::new(256, usage).expect("should create");
        assert_eq!(buf.usage(), usage);
        assert_eq!(buf.size_bytes(), 256);
    }
}

#[test]
fn test_vulkan_buffer_very_large_size() {
    // 1 GB buffer -- should succeed in placeholder implementation.
    let size = 1024 * 1024 * 1024;
    let buf = VulkanBuffer::new(size, BufferUsage::StorageReadWrite).expect("should create");
    assert_eq!(buf.size_bytes(), size);
}

#[test]
fn test_vulkan_buffer_size_one() {
    let buf = VulkanBuffer::new(1, BufferUsage::StorageRead).expect("should create");
    assert_eq!(buf.size_bytes(), 1);
}

#[test]
fn test_vulkan_buffer_debug_format() {
    let buf = VulkanBuffer::new(512, BufferUsage::Uniform).expect("should create");
    let debug = format!("{buf:?}");
    assert!(debug.contains("VulkanBuffer"));
    assert!(debug.contains("512"));
    assert!(debug.contains("Uniform"));
}

// ---------------------------------------------------------------------------
// BufferUsage tests
// ---------------------------------------------------------------------------

#[test]
fn test_buffer_usage_to_vk_bits_storage() {
    // Both StorageRead and StorageReadWrite map to VK_BUFFER_USAGE_STORAGE_BUFFER_BIT.
    assert_eq!(BufferUsage::StorageRead.to_vk_bits(), 0x0000_0020);
    assert_eq!(BufferUsage::StorageReadWrite.to_vk_bits(), 0x0000_0020);
}

#[test]
fn test_buffer_usage_to_vk_bits_uniform() {
    assert_eq!(BufferUsage::Uniform.to_vk_bits(), 0x0000_0010);
}

#[test]
fn test_buffer_usage_to_vk_bits_transfer() {
    assert_eq!(BufferUsage::TransferSrc.to_vk_bits(), 0x0000_0001);
    assert_eq!(BufferUsage::TransferDst.to_vk_bits(), 0x0000_0002);
}

#[test]
fn test_buffer_usage_all_bits_distinct_or_documented() {
    // TransferSrc, TransferDst, Uniform each have distinct bits.
    let src = BufferUsage::TransferSrc.to_vk_bits();
    let dst = BufferUsage::TransferDst.to_vk_bits();
    let uni = BufferUsage::Uniform.to_vk_bits();
    let storage = BufferUsage::StorageRead.to_vk_bits();

    assert_ne!(src, dst);
    assert_ne!(src, uni);
    assert_ne!(src, storage);
    assert_ne!(dst, uni);
    assert_ne!(dst, storage);
    assert_ne!(uni, storage);
}

#[test]
fn test_buffer_usage_clone_copy_eq() {
    let u1 = BufferUsage::StorageReadWrite;
    let u2 = u1; // Copy
    let u3 = u1; // Clone
    assert_eq!(u1, u2);
    assert_eq!(u1, u3);
    assert_ne!(u1, BufferUsage::Uniform);
}

#[test]
fn test_buffer_usage_debug_format() {
    let debug = format!("{:?}", BufferUsage::TransferSrc);
    assert_eq!(debug, "TransferSrc");
}

// ---------------------------------------------------------------------------
// StagingBuffer tests
// ---------------------------------------------------------------------------

#[test]
fn test_staging_buffer_upload_creation() {
    let sb = StagingBuffer::new_upload(1024).expect("should create");
    assert_eq!(sb.size_bytes(), 1024);
    assert!(sb.is_upload());
}

#[test]
fn test_staging_buffer_download_creation() {
    let sb = StagingBuffer::new_download(2048).expect("should create");
    assert_eq!(sb.size_bytes(), 2048);
    assert!(!sb.is_upload());
}

#[test]
fn test_staging_buffer_zero_size_upload_rejected() {
    let result = StagingBuffer::new_upload(0);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("size must be > 0"), "error: {msg}");
}

#[test]
fn test_staging_buffer_zero_size_download_rejected() {
    let result = StagingBuffer::new_download(0);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("size must be > 0"), "error: {msg}");
}

#[test]
fn test_staging_buffer_write_f32_within_capacity() {
    let mut sb = StagingBuffer::new_upload(16).expect("should create");
    // 4 floats * 4 bytes = 16 bytes, exactly fits.
    let data = [1.0_f32, 2.0, 3.0, 4.0];
    assert!(sb.write_f32(&data).is_ok());
}

#[test]
fn test_staging_buffer_write_f32_exceeds_capacity() {
    let mut sb = StagingBuffer::new_upload(8).expect("should create");
    // 3 floats * 4 bytes = 12 bytes > 8 bytes.
    let data = [1.0_f32, 2.0, 3.0];
    let result = sb.write_f32(&data);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("size mismatch"), "error: {msg}");
}

#[test]
fn test_staging_buffer_write_f32_empty_slice() {
    let mut sb = StagingBuffer::new_upload(64).expect("should create");
    // Empty slice: 0 bytes, always fits.
    assert!(sb.write_f32(&[]).is_ok());
}

#[test]
fn test_staging_buffer_read_f32_within_capacity() {
    let sb = StagingBuffer::new_download(16).expect("should create");
    let data = sb.read_f32(4).expect("should read");
    assert_eq!(data.len(), 4);
    // Placeholder returns zeros.
    assert!(data.iter().all(|&v| v == 0.0));
}

#[test]
fn test_staging_buffer_read_f32_exceeds_capacity() {
    let sb = StagingBuffer::new_download(8).expect("should create");
    // 3 floats * 4 = 12 > 8.
    let result = sb.read_f32(3);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("size mismatch"), "error: {msg}");
}

#[test]
fn test_staging_buffer_read_f32_zero_count() {
    let sb = StagingBuffer::new_download(64).expect("should create");
    let data = sb.read_f32(0).expect("should read zero");
    assert!(data.is_empty());
}

#[test]
fn test_staging_buffer_debug_format() {
    let sb = StagingBuffer::new_upload(512).expect("should create");
    let debug = format!("{sb:?}");
    assert!(debug.contains("StagingBuffer"));
    assert!(debug.contains("512"));
}

// ---------------------------------------------------------------------------
// BufferPool: size class boundary tests
// ---------------------------------------------------------------------------

#[test]
fn test_pool_size_class_exact_64kb_boundary() {
    let mut pool = BufferPool::new();
    // Exactly 64KB should go into class 0.
    let buf = pool
        .acquire(64 * 1024, BufferUsage::StorageReadWrite)
        .expect("acquire");
    assert!(buf.size_bytes() >= 64 * 1024);
    let ps = pool.pool_stats();
    assert_eq!(ps.size_classes[0].total_allocated, 1);
}

#[test]
fn test_pool_size_class_just_over_64kb() {
    let mut pool = BufferPool::new();
    // 64KB + 1 should go into class 1 (256KB).
    let _ = pool
        .acquire(64 * 1024 + 1, BufferUsage::StorageReadWrite)
        .expect("acquire");
    let ps = pool.pool_stats();
    assert_eq!(ps.size_classes[0].total_allocated, 0, "not in 64KB class");
    assert_eq!(
        ps.size_classes[1].total_allocated, 1,
        "should be in 256KB class"
    );
}

#[test]
fn test_pool_size_class_exact_256kb_boundary() {
    let mut pool = BufferPool::new();
    let _ = pool
        .acquire(256 * 1024, BufferUsage::StorageReadWrite)
        .expect("acquire");
    let ps = pool.pool_stats();
    assert_eq!(ps.size_classes[1].total_allocated, 1);
}

#[test]
fn test_pool_size_class_exact_1mb_boundary() {
    let mut pool = BufferPool::new();
    let _ = pool
        .acquire(1024 * 1024, BufferUsage::StorageReadWrite)
        .expect("acquire");
    let ps = pool.pool_stats();
    assert_eq!(ps.size_classes[2].total_allocated, 1);
}

#[test]
fn test_pool_size_class_exact_4mb_boundary() {
    let mut pool = BufferPool::new();
    let _ = pool
        .acquire(4 * 1024 * 1024, BufferUsage::StorageReadWrite)
        .expect("acquire");
    let ps = pool.pool_stats();
    assert_eq!(ps.size_classes[3].total_allocated, 1);
}

#[test]
fn test_pool_size_class_exact_16mb_boundary() {
    let mut pool = BufferPool::new();
    let _ = pool
        .acquire(16 * 1024 * 1024, BufferUsage::StorageReadWrite)
        .expect("acquire");
    let ps = pool.pool_stats();
    assert_eq!(ps.size_classes[4].total_allocated, 1);
}

#[test]
fn test_pool_size_class_exact_64mb_boundary() {
    let mut pool = BufferPool::new();
    let _ = pool
        .acquire(64 * 1024 * 1024, BufferUsage::StorageReadWrite)
        .expect("acquire");
    let ps = pool.pool_stats();
    assert_eq!(ps.size_classes[5].total_allocated, 1);
}

#[test]
fn test_pool_size_class_exact_256mb_boundary() {
    let mut pool = BufferPool::new();
    let _ = pool
        .acquire(256 * 1024 * 1024, BufferUsage::StorageReadWrite)
        .expect("acquire");
    let ps = pool.pool_stats();
    assert_eq!(ps.size_classes[6].total_allocated, 1);
}

#[test]
fn test_pool_size_class_just_over_256mb_is_oversized() {
    let mut pool = BufferPool::new();
    let _ = pool
        .acquire(256 * 1024 * 1024 + 1, BufferUsage::StorageReadWrite)
        .expect("acquire");
    let ps = pool.pool_stats();
    // All classes should have 0 allocations.
    for sc in &ps.size_classes {
        assert_eq!(sc.total_allocated, 0);
    }
    assert_eq!(ps.total_discards, 1, "should be discarded as oversized");
}

#[test]
fn test_pool_tiny_request_goes_to_first_class() {
    let mut pool = BufferPool::new();
    let _ = pool
        .acquire(1, BufferUsage::StorageReadWrite)
        .expect("acquire");
    let ps = pool.pool_stats();
    assert_eq!(ps.size_classes[0].total_allocated, 1);
}

// ---------------------------------------------------------------------------
// BufferPool: capacity limits (MAX_PER_CLASS = 8)
// ---------------------------------------------------------------------------

#[test]
fn test_pool_max_per_class_limit() {
    let mut pool = BufferPool::new();
    // Fill up class 0 (64KB) with 8 buffers, then 9th should be discarded.
    for i in 0..8 {
        let _ = pool
            .acquire(100, BufferUsage::StorageReadWrite)
            .unwrap_or_else(|_| panic!("acquire {i}"));
    }
    let ps = pool.pool_stats();
    assert_eq!(ps.size_classes[0].buffer_count, 8);
    assert_eq!(ps.total_discards, 0, "first 8 should fit");

    // 9th should be discarded (bucket full).
    let _ = pool
        .acquire(100, BufferUsage::StorageReadWrite)
        .expect("acquire 9th");
    let ps = pool.pool_stats();
    assert_eq!(ps.total_discards, 1, "9th should be discarded");
    assert_eq!(ps.size_classes[0].buffer_count, 8, "still 8 in bucket");
}

// ---------------------------------------------------------------------------
// BufferPool: multiple size classes simultaneously
// ---------------------------------------------------------------------------

#[test]
fn test_pool_multiple_classes_independent() {
    let mut pool = BufferPool::new();
    // Allocate into class 0 (64KB), class 2 (1MB), class 4 (16MB).
    let _ = pool
        .acquire(100, BufferUsage::StorageReadWrite)
        .expect("acquire class 0");
    let _ = pool
        .acquire(500 * 1024, BufferUsage::StorageReadWrite)
        .expect("acquire class 2");
    let _ = pool
        .acquire(5 * 1024 * 1024, BufferUsage::StorageReadWrite)
        .expect("acquire class 4");

    let ps = pool.pool_stats();
    assert_eq!(ps.size_classes[0].total_allocated, 1);
    assert_eq!(ps.size_classes[2].total_allocated, 1);
    assert_eq!(ps.size_classes[4].total_allocated, 1);
    assert_eq!(ps.total_allocated, 3);
    assert_eq!(ps.current_buffer_count, 3);
}

// ---------------------------------------------------------------------------
// BufferPool: release and reuse patterns
// ---------------------------------------------------------------------------

#[test]
fn test_pool_release_then_reuse_same_class() {
    let mut pool = BufferPool::new();
    let buf = pool
        .acquire(1024, BufferUsage::StorageReadWrite)
        .expect("acquire");
    let original_size = buf.size_bytes();
    pool.release(buf);

    // Re-acquire same size: should hit.
    let buf2 = pool
        .acquire(1024, BufferUsage::StorageReadWrite)
        .expect("reacquire");
    assert_eq!(buf2.size_bytes(), original_size);

    let ps = pool.pool_stats();
    assert_eq!(ps.total_allocated, 1, "only one real allocation");
    assert_eq!(ps.total_reused, 1, "one reuse");
}

#[test]
fn test_pool_release_buffer_not_from_pool() {
    let mut pool = BufferPool::new();
    // Create a buffer directly (not via pool.acquire).
    let standalone = VulkanBuffer::new(4096, BufferUsage::StorageReadWrite).expect("create");
    // Release it -- should silently drop (no crash, no corruption).
    pool.release(standalone);
    assert_eq!(pool.stats().buffer_count, 0);
}

#[test]
fn test_pool_release_oversized_buffer_silently_dropped() {
    let mut pool = BufferPool::new();
    // Acquire oversized (bypasses pool).
    let big = pool
        .acquire(512 * 1024 * 1024, BufferUsage::StorageReadWrite)
        .expect("acquire oversized");
    pool.release(big);
    // Pool should still be empty.
    assert_eq!(pool.stats().buffer_count, 0);
}

// ---------------------------------------------------------------------------
// BufferPool: eviction patterns
// ---------------------------------------------------------------------------

#[test]
fn test_pool_evict_multiple_classes() {
    let mut pool = BufferPool::new();
    let buf_a = pool
        .acquire(100, BufferUsage::StorageReadWrite)
        .expect("acquire a");
    let buf_b = pool
        .acquire(500 * 1024, BufferUsage::StorageReadWrite)
        .expect("acquire b");
    pool.release(buf_a);
    pool.release(buf_b);

    let (evicted, bytes) = pool.evict();
    assert_eq!(evicted, 2);
    assert!(bytes > 0);
    assert_eq!(pool.stats().buffer_count, 0);
}

#[test]
fn test_pool_evict_preserves_in_use_buffers() {
    let mut pool = BufferPool::new();
    // Acquire two, release only first.
    let buf1 = pool
        .acquire(100, BufferUsage::StorageReadWrite)
        .expect("acquire 1");
    let _buf2_live = pool
        .acquire(200, BufferUsage::StorageReadWrite)
        .expect("acquire 2");
    pool.release(buf1);

    let (evicted, _) = pool.evict();
    assert_eq!(evicted, 1, "only released buffer evicted");
    assert_eq!(pool.stats().buffer_count, 1, "in-use buffer remains");
}

#[test]
fn test_pool_double_evict_is_idempotent() {
    let mut pool = BufferPool::new();
    let buf = pool
        .acquire(100, BufferUsage::StorageReadWrite)
        .expect("acquire");
    pool.release(buf);

    let (evicted1, bytes1) = pool.evict();
    assert_eq!(evicted1, 1);
    assert!(bytes1 > 0);

    // Second evict: nothing left.
    let (evicted2, bytes2) = pool.evict();
    assert_eq!(evicted2, 0);
    assert_eq!(bytes2, 0);
}

// ---------------------------------------------------------------------------
// BufferPool: clear behavior
// ---------------------------------------------------------------------------

#[test]
fn test_pool_clear_resets_retained_bytes() {
    let mut pool = BufferPool::new();
    let _ = pool
        .acquire(100, BufferUsage::StorageReadWrite)
        .expect("acquire");
    assert!(pool.stats().retained_bytes > 0);
    pool.clear();
    assert_eq!(pool.stats().retained_bytes, 0);
    assert_eq!(pool.stats().buffer_count, 0);
}

#[test]
fn test_pool_clear_then_acquire_works() {
    let mut pool = BufferPool::new();
    let _ = pool
        .acquire(100, BufferUsage::StorageReadWrite)
        .expect("first");
    pool.clear();
    // Should still be able to acquire after clear.
    let buf = pool
        .acquire(200, BufferUsage::StorageReadWrite)
        .expect("after clear");
    assert!(buf.size_bytes() >= 200);
}

// ---------------------------------------------------------------------------
// BufferPool: stats snapshot types
// ---------------------------------------------------------------------------

#[test]
fn test_pool_stats_clone_copy_eq() {
    let s1 = PoolStats {
        acquisitions: 5,
        hits: 3,
        misses: 2,
        discards: 0,
        retained_bytes: 1024,
        buffer_count: 2,
    };
    let s2 = s1; // Copy
    let s3 = s1;
    assert_eq!(s1, s2);
    assert_eq!(s1, s3);
}

#[test]
fn test_pool_stats_ne() {
    let s1 = PoolStats::default();
    let s2 = PoolStats {
        acquisitions: 1,
        ..PoolStats::default()
    };
    assert_ne!(s1, s2);
}

#[test]
fn test_size_class_stats_clone_copy() {
    let sc = SizeClassStats {
        class_bytes: 65536,
        buffer_count: 2,
        available_count: 1,
        retained_bytes: 131072,
        total_allocated: 3,
        total_reused: 1,
        total_evicted: 0,
    };
    let sc2 = sc; // Copy
    let sc3 = sc;
    assert_eq!(sc, sc2);
    assert_eq!(sc, sc3);
}

#[test]
fn test_buffer_pool_stats_clone() {
    let mut pool = BufferPool::new();
    let buf = pool
        .acquire(1024, BufferUsage::StorageReadWrite)
        .expect("acquire");
    pool.release(buf);
    let _ = pool
        .acquire(1024, BufferUsage::StorageReadWrite)
        .expect("reacquire");

    let ps = pool.pool_stats();
    let ps2 = ps.clone();
    assert_eq!(ps, ps2);
}

#[test]
fn test_buffer_pool_stats_debug() {
    let ps = BufferPoolStats::default();
    let debug = format!("{ps:?}");
    assert!(debug.contains("BufferPoolStats"));
}

// ---------------------------------------------------------------------------
// BufferPool: stats consistency invariants
// ---------------------------------------------------------------------------

#[test]
fn test_pool_stats_acquisitions_equals_hits_plus_misses_plus_discards() {
    let mut pool = BufferPool::new();

    // miss
    let buf = pool
        .acquire(100, BufferUsage::StorageReadWrite)
        .expect("miss");
    pool.release(buf);
    // hit
    let _ = pool
        .acquire(100, BufferUsage::StorageReadWrite)
        .expect("hit");
    // discard (oversized)
    let _ = pool
        .acquire(512 * 1024 * 1024, BufferUsage::StorageReadWrite)
        .expect("discard");

    let s = pool.stats();
    assert_eq!(
        s.acquisitions,
        s.hits + s.misses + s.discards,
        "acquisitions = hits + misses + discards"
    );
}

#[test]
fn test_pool_stats_comprehensive_invariant_check() {
    let mut pool = BufferPool::new();

    // Allocate several buffers across different classes.
    let b1 = pool
        .acquire(100, BufferUsage::StorageReadWrite)
        .expect("b1");
    let b2 = pool
        .acquire(300 * 1024, BufferUsage::StorageReadWrite)
        .expect("b2");
    let _b3 = pool
        .acquire(2 * 1024 * 1024, BufferUsage::StorageReadWrite)
        .expect("b3");

    // Release some.
    pool.release(b1);
    pool.release(b2);

    // Reuse one.
    let _ = pool
        .acquire(100, BufferUsage::StorageReadWrite)
        .expect("reuse b1");

    let ps = pool.pool_stats();

    // total_acquisitions = total_allocated + total_reused + total_discards
    assert_eq!(
        ps.total_acquisitions,
        ps.total_allocated + ps.total_reused + ps.total_discards,
        "acquisitions invariant"
    );

    // Per-class allocated sum should equal total_allocated.
    let class_alloc_sum: usize = ps.size_classes.iter().map(|sc| sc.total_allocated).sum();
    assert_eq!(
        class_alloc_sum, ps.total_allocated,
        "per-class allocation sum"
    );

    // Per-class reused sum should equal total_reused.
    let class_reuse_sum: usize = ps.size_classes.iter().map(|sc| sc.total_reused).sum();
    assert_eq!(class_reuse_sum, ps.total_reused, "per-class reuse sum");

    // hit_rate check: total_reused / (total_reused + total_allocated).
    if ps.total_reused + ps.total_allocated > 0 {
        let expected_rate = ps.total_reused as f64 / (ps.total_reused + ps.total_allocated) as f64;
        assert!(
            (ps.hit_rate - expected_rate).abs() < 1e-9,
            "hit_rate: {} vs expected: {}",
            ps.hit_rate,
            expected_rate
        );
    }
}

// ---------------------------------------------------------------------------
// BufferPool: reset_stats then acquire cycle
// ---------------------------------------------------------------------------

#[test]
fn test_pool_reset_stats_then_new_acquisitions_tracked() {
    let mut pool = BufferPool::new();
    let buf = pool
        .acquire(100, BufferUsage::StorageReadWrite)
        .expect("first");
    pool.release(buf);

    pool.reset_stats();

    // After reset, counters start fresh.
    let _ = pool
        .acquire(100, BufferUsage::StorageReadWrite)
        .expect("after reset");
    let ps = pool.pool_stats();
    assert_eq!(
        ps.total_acquisitions, 1,
        "only post-reset acquisition counted"
    );
    // This should be a hit (buffer was available from pre-reset).
    assert_eq!(ps.total_reused, 1, "reuse of pre-reset buffer");
}

// ---------------------------------------------------------------------------
// BufferPool: retained_bytes tracking
// ---------------------------------------------------------------------------

#[test]
fn test_pool_retained_bytes_increases_on_miss() {
    let mut pool = BufferPool::new();
    let before = pool.stats().retained_bytes;
    let _ = pool
        .acquire(100, BufferUsage::StorageReadWrite)
        .expect("acquire");
    let after = pool.stats().retained_bytes;
    assert!(after > before, "retained_bytes should increase on miss");
}

#[test]
fn test_pool_retained_bytes_decreases_on_evict() {
    let mut pool = BufferPool::new();
    let buf = pool
        .acquire(100, BufferUsage::StorageReadWrite)
        .expect("acquire");
    pool.release(buf);
    let before = pool.stats().retained_bytes;
    assert!(before > 0);

    pool.evict();
    let after = pool.stats().retained_bytes;
    assert_eq!(after, 0, "retained_bytes should be 0 after full evict");
}

// ---------------------------------------------------------------------------
// BufferPool: pool_stats vs stats consistency
// ---------------------------------------------------------------------------

#[test]
fn test_pool_stats_and_pool_stats_agree() {
    let mut pool = BufferPool::new();
    let buf = pool
        .acquire(100, BufferUsage::StorageReadWrite)
        .expect("acquire");
    pool.release(buf);
    let _ = pool
        .acquire(100, BufferUsage::StorageReadWrite)
        .expect("reacquire");

    let simple = pool.stats();
    let full = pool.pool_stats();

    assert_eq!(simple.acquisitions, full.total_acquisitions);
    assert_eq!(simple.hits, full.total_reused);
    assert_eq!(simple.misses, full.total_allocated);
    assert_eq!(simple.discards, full.total_discards);
    assert_eq!(simple.retained_bytes, full.current_retained_bytes);
    assert_eq!(simple.buffer_count, full.current_buffer_count);
}

// ---------------------------------------------------------------------------
// BufferPool: display format content checks
// ---------------------------------------------------------------------------

#[test]
fn test_pool_stats_display_all_class_labels() {
    let mut pool = BufferPool::new();
    let _ = pool
        .acquire(100, BufferUsage::StorageReadWrite)
        .expect("acquire");
    let ps = pool.pool_stats();
    let display = format!("{ps}");

    let expected_labels = ["64KB", "256KB", "1MB", "4MB", "16MB", "64MB", "256MB"];
    for label in &expected_labels {
        assert!(display.contains(label), "display should contain {label}");
    }
}

#[test]
fn test_pool_stats_display_sections() {
    let ps = BufferPoolStats::default();
    let display = format!("{ps}");
    assert!(display.contains("Lifetime:"));
    assert!(display.contains("Peaks:"));
    assert!(display.contains("Current:"));
    assert!(display.contains("Per-size-class:"));
}

// ---------------------------------------------------------------------------
// BufferPool: rapid acquire-release cycles
// ---------------------------------------------------------------------------

#[test]
fn test_pool_rapid_acquire_release_cycle() {
    let mut pool = BufferPool::new();

    for _ in 0..50 {
        let buf = pool
            .acquire(4096, BufferUsage::StorageReadWrite)
            .expect("acquire");
        pool.release(buf);
    }

    let ps = pool.pool_stats();
    assert_eq!(ps.total_acquisitions, 50);
    // First is a miss, remaining 49 should be hits.
    assert_eq!(ps.total_allocated, 1);
    assert_eq!(ps.total_reused, 49);
    assert!(
        ps.hit_rate > 0.95,
        "hit_rate should be ~0.98, got {}",
        ps.hit_rate
    );
}

#[test]
fn test_pool_alternating_sizes_no_reuse() {
    let mut pool = BufferPool::new();

    // Alternate between different size classes -- no reuse because released
    // buffer is the wrong size for the next acquire.
    for i in 0..4 {
        let size = if i % 2 == 0 { 100 } else { 500 * 1024 };
        let buf = pool
            .acquire(size, BufferUsage::StorageReadWrite)
            .expect("acquire");
        pool.release(buf);
    }

    let ps = pool.pool_stats();
    // 2 allocations into class 0, 2 into class 2.
    // Each size alternates, so after first pair, each class has an available
    // buffer and subsequent requests reuse it.
    // Pattern: miss(100), miss(500K), hit(100), hit(500K).
    assert_eq!(ps.total_allocated, 2);
    assert_eq!(ps.total_reused, 2);
}

// ---------------------------------------------------------------------------
// BufferPool: different usage flags don't affect pooling
// ---------------------------------------------------------------------------

#[test]
fn test_pool_different_usage_flags_share_class() {
    let mut pool = BufferPool::new();
    // Acquire with different usage flags -- pool doesn't differentiate by usage.
    let buf1 = pool
        .acquire(100, BufferUsage::StorageRead)
        .expect("acquire StorageRead");
    pool.release(buf1);

    // Acquire with different usage -- should still reuse.
    let _ = pool
        .acquire(100, BufferUsage::StorageReadWrite)
        .expect("acquire StorageReadWrite");

    let ps = pool.pool_stats();
    assert_eq!(ps.total_reused, 1, "should reuse regardless of usage flag");
}
