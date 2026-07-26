// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the Vulkan command batch and barrier system.

use super::*;
use crate::buffer::BufferUsage;
use crate::dispatch::{DescriptorBinding, DescriptorSetLayout, DescriptorType, PipelineLayout};
use crate::spirv_emit::{SPIRV_MAGIC, SPIRV_VERSION_1_5};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_pipeline() -> ComputePipeline {
    let ds_layout = DescriptorSetLayout::new(vec![DescriptorBinding {
        binding: 0,
        descriptor_type: DescriptorType::StorageBuffer,
        count: 1,
    }])
    .expect("ds layout");
    let pl = PipelineLayout::new(&ds_layout, 4).expect("pl");
    let spirv = vec![SPIRV_MAGIC, SPIRV_VERSION_1_5, 0, 0, 0];
    ComputePipeline::new(&spirv, "main", &pl).expect("pipeline")
}

fn make_buffer(size: usize) -> VulkanBuffer {
    VulkanBuffer::new(size, BufferUsage::StorageReadWrite).expect("buf")
}

// ---------------------------------------------------------------------------
// CommandBatch creation (empty)
// ---------------------------------------------------------------------------

#[test]
fn test_batch_creation_empty_auto() {
    let batch = CommandBatch::new(BarrierStrategy::Auto);
    assert_eq!(batch.dispatch_count(), 0);
    assert_eq!(batch.barrier_count(), 0);
    assert_eq!(batch.strategy(), BarrierStrategy::Auto);
}

#[test]
fn test_batch_creation_empty_manual() {
    let batch = CommandBatch::new(BarrierStrategy::Manual);
    assert_eq!(batch.dispatch_count(), 0);
    assert_eq!(batch.barrier_count(), 0);
    assert_eq!(batch.strategy(), BarrierStrategy::Manual);
}

// ---------------------------------------------------------------------------
// BarrierStrategy enum variants and Default impl
// ---------------------------------------------------------------------------

#[test]
fn test_barrier_strategy_default_is_auto() {
    assert_eq!(BarrierStrategy::default(), BarrierStrategy::Auto);
}

#[test]
fn test_barrier_strategy_variants_distinct() {
    assert_ne!(BarrierStrategy::Auto, BarrierStrategy::Manual);
}

#[test]
fn test_barrier_strategy_clone_copy() {
    let s = BarrierStrategy::Auto;
    let cloned = s;
    let copied = s;
    assert_eq!(s, cloned);
    assert_eq!(s, copied);
}

#[test]
fn test_barrier_strategy_debug() {
    let dbg = format!("{:?}", BarrierStrategy::Auto);
    assert!(dbg.contains("Auto"));
    let dbg_manual = format!("{:?}", BarrierStrategy::Manual);
    assert!(dbg_manual.contains("Manual"));
}

#[test]
fn test_batch_from_default_strategy() {
    let batch = CommandBatch::new(BarrierStrategy::default());
    assert_eq!(batch.strategy(), BarrierStrategy::Auto);
}

// ---------------------------------------------------------------------------
// PendingBatch state tracking
// ---------------------------------------------------------------------------

#[test]
fn test_pending_batch_is_completed() {
    let pipeline = make_pipeline();
    let buf = make_buffer(256);

    let mut batch = CommandBatch::new(BarrierStrategy::Auto);
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
        .expect("record");
    let pending = batch.submit_async().expect("async submit");
    // Placeholder returns true (no real GPU).
    assert!(pending.is_completed());
}

#[test]
fn test_pending_batch_wait_succeeds() {
    let pipeline = make_pipeline();
    let buf = make_buffer(256);

    let mut batch = CommandBatch::new(BarrierStrategy::Auto);
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [2, 2, 1])
        .expect("record");
    let pending = batch.submit_async().expect("async submit");
    pending.wait().expect("wait should succeed");
}

#[test]
fn test_pending_batch_dispatch_count_matches_batch() {
    let pipeline = make_pipeline();
    let buf = make_buffer(256);

    let mut batch = CommandBatch::new(BarrierStrategy::Auto);
    for _ in 0..7 {
        batch
            .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
            .expect("record");
    }
    let pending = batch.submit_async().expect("async submit");
    assert_eq!(pending.dispatch_count(), 7);
}

#[test]
fn test_pending_batch_debug() {
    let pipeline = make_pipeline();
    let buf = make_buffer(256);

    let mut batch = CommandBatch::new(BarrierStrategy::Auto);
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
        .expect("record");
    let pending = batch.submit_async().expect("async submit");
    let dbg = format!("{pending:?}");
    assert!(dbg.contains("PendingBatch"));
}

// ---------------------------------------------------------------------------
// Batch dispatch count tracking
// ---------------------------------------------------------------------------

#[test]
fn test_dispatch_count_increments() {
    let pipeline = make_pipeline();
    let buf = make_buffer(256);
    let mut batch = CommandBatch::new(BarrierStrategy::Auto);

    for expected in 1..=10u32 {
        batch
            .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
            .expect("record");
        assert_eq!(batch.dispatch_count(), expected);
    }
}

#[test]
fn test_dispatch_count_unaffected_by_failed_records() {
    let pipeline = make_pipeline();
    let buf = make_buffer(256);
    let mut batch = CommandBatch::new(BarrierStrategy::Auto);

    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
        .expect("record");
    // This should fail (zero group count) and NOT increment dispatch count.
    let _ = batch.record(&pipeline, &[&buf], &[0u8; 4], [0, 0, 0]);
    assert_eq!(batch.dispatch_count(), 1);
}

// ---------------------------------------------------------------------------
// Memory barrier insertion rules
// ---------------------------------------------------------------------------

#[test]
fn test_auto_barrier_count_equals_dispatch_count() {
    let pipeline = make_pipeline();
    let buf = make_buffer(256);
    let mut batch = CommandBatch::new(BarrierStrategy::Auto);

    for n in 1..=8u32 {
        batch
            .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
            .expect("record");
        assert_eq!(
            batch.barrier_count(),
            n,
            "Auto barrier count should equal dispatch count"
        );
    }
}

#[test]
fn test_manual_barrier_count_zero_without_explicit() {
    let pipeline = make_pipeline();
    let buf = make_buffer(256);
    let mut batch = CommandBatch::new(BarrierStrategy::Manual);

    for _ in 0..5 {
        batch
            .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
            .expect("record");
    }
    assert_eq!(batch.barrier_count(), 0);
}

#[test]
fn test_auto_mode_ignores_explicit_barrier() {
    let pipeline = make_pipeline();
    let buf = make_buffer(256);
    let mut batch = CommandBatch::new(BarrierStrategy::Auto);

    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
        .expect("record");
    batch.barrier(); // Should be a no-op in Auto mode.
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
        .expect("record");

    // barrier_count should be 2 (one per dispatch), not 3.
    assert_eq!(batch.barrier_count(), 2);
    assert_eq!(batch.dispatch_count(), 2);
}

#[test]
fn test_manual_multiple_explicit_barriers() {
    let pipeline = make_pipeline();
    let buf = make_buffer(256);
    let mut batch = CommandBatch::new(BarrierStrategy::Manual);

    // Dispatch A
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
        .expect("record A");
    batch.barrier();
    // Dispatch B
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
        .expect("record B");
    batch.barrier();
    // Dispatch C
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
        .expect("record C");

    assert_eq!(batch.dispatch_count(), 3);
    assert_eq!(batch.barrier_count(), 2);
    assert!(batch.has_barrier_after(0));
    assert!(batch.has_barrier_after(1));
    assert!(!batch.has_barrier_after(2));
}

#[test]
fn test_manual_barrier_on_empty_batch_is_noop() {
    let mut batch = CommandBatch::new(BarrierStrategy::Manual);
    batch.barrier();
    // No dispatches to attach barrier to, but barrier_count still increments.
    assert_eq!(batch.barrier_count(), 1);
    assert_eq!(batch.dispatch_count(), 0);
}

// ---------------------------------------------------------------------------
// Batch submission and completion (mock/logical level)
// ---------------------------------------------------------------------------

#[test]
fn test_batch_empty_submit_rejected() {
    let batch = CommandBatch::new(BarrierStrategy::Auto);
    assert!(batch.submit_and_wait().is_err());
}

#[test]
fn test_batch_empty_async_submit_rejected() {
    let batch = CommandBatch::new(BarrierStrategy::Auto);
    assert!(batch.submit_async().is_err());
}

#[test]
fn test_batch_submit_and_wait() {
    let pipeline = make_pipeline();
    let buf = make_buffer(256);

    let mut batch = CommandBatch::new(BarrierStrategy::Auto);
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [4, 1, 1])
        .expect("record");
    batch.submit_and_wait().expect("submit");
}

#[test]
fn test_batch_submit_async() {
    let pipeline = make_pipeline();
    let buf = make_buffer(256);

    let mut batch = CommandBatch::new(BarrierStrategy::Auto);
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [4, 1, 1])
        .expect("record");
    let pending = batch.submit_async().expect("async submit");
    assert_eq!(pending.dispatch_count(), 1);
    assert!(pending.is_completed());
    pending.wait().expect("wait");
}

#[test]
fn test_batch_manual_submit_succeeds() {
    let pipeline = make_pipeline();
    let buf = make_buffer(256);

    let mut batch = CommandBatch::new(BarrierStrategy::Manual);
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
        .expect("record");
    batch.submit_and_wait().expect("submit");
}

// ---------------------------------------------------------------------------
// Multiple dispatches in sequence
// ---------------------------------------------------------------------------

#[test]
fn test_batch_auto_barrier_single_dispatch() {
    let pipeline = make_pipeline();
    let buf = make_buffer(256);

    let mut batch = CommandBatch::new(BarrierStrategy::Auto);
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
        .expect("record");

    assert_eq!(batch.dispatch_count(), 1);
    assert_eq!(batch.barrier_count(), 1);
    assert!(batch.has_barrier_after(0));
}

#[test]
fn test_batch_auto_barrier_multiple_dispatches() {
    let pipeline = make_pipeline();
    let buf = make_buffer(256);

    let mut batch = CommandBatch::new(BarrierStrategy::Auto);
    for _ in 0..5 {
        batch
            .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
            .expect("record");
    }

    assert_eq!(batch.dispatch_count(), 5);
    assert_eq!(batch.barrier_count(), 5);
    for i in 0..5 {
        assert!(batch.has_barrier_after(i));
    }
}

// ---------------------------------------------------------------------------
// Barrier between read-after-write operations
// ---------------------------------------------------------------------------

#[test]
fn test_batch_manual_barrier_explicit_insert() {
    let pipeline = make_pipeline();
    let buf = make_buffer(256);

    let mut batch = CommandBatch::new(BarrierStrategy::Manual);
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
        .expect("record 1");
    batch.barrier(); // Explicit barrier between dispatches.
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
        .expect("record 2");

    assert_eq!(batch.dispatch_count(), 2);
    assert_eq!(batch.barrier_count(), 1);
    assert!(batch.has_barrier_after(0));
    assert!(!batch.has_barrier_after(1));
}

#[test]
fn test_raw_dependency_chain_manual() {
    // Simulates: kernel_a writes buf -> barrier -> kernel_b reads buf -> barrier -> kernel_c reads buf
    let pipeline = make_pipeline();
    let buf_a = make_buffer(1024);
    let buf_b = make_buffer(1024);

    let mut batch = CommandBatch::new(BarrierStrategy::Manual);

    // kernel_a: writes to buf_a
    batch
        .record(&pipeline, &[&buf_a], &[0u8; 4], [8, 1, 1])
        .expect("kernel_a");
    batch.barrier(); // RAW: kernel_b reads buf_a after kernel_a writes it

    // kernel_b: reads buf_a, writes buf_b
    batch
        .record(&pipeline, &[&buf_a, &buf_b], &[0u8; 4], [8, 1, 1])
        .expect("kernel_b");
    batch.barrier(); // RAW: kernel_c reads buf_b after kernel_b writes it

    // kernel_c: reads buf_b
    batch
        .record(&pipeline, &[&buf_b], &[0u8; 4], [8, 1, 1])
        .expect("kernel_c");

    assert_eq!(batch.dispatch_count(), 3);
    assert_eq!(batch.barrier_count(), 2);
    assert!(batch.has_barrier_after(0));
    assert!(batch.has_barrier_after(1));
    assert!(!batch.has_barrier_after(2));
}

// ---------------------------------------------------------------------------
// Edge cases: empty batch submission
// ---------------------------------------------------------------------------

#[test]
fn test_empty_batch_manual_submit_rejected() {
    let batch = CommandBatch::new(BarrierStrategy::Manual);
    assert!(batch.submit_and_wait().is_err());
}

#[test]
fn test_empty_batch_manual_async_rejected() {
    let batch = CommandBatch::new(BarrierStrategy::Manual);
    assert!(batch.submit_async().is_err());
}

// ---------------------------------------------------------------------------
// Edge cases: single dispatch batch
// ---------------------------------------------------------------------------

#[test]
fn test_single_dispatch_auto_submit() {
    let pipeline = make_pipeline();
    let buf = make_buffer(64);

    let mut batch = CommandBatch::new(BarrierStrategy::Auto);
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
        .expect("record");
    batch.submit_and_wait().expect("submit");
    assert_eq!(batch.dispatch_count(), 1);
    assert_eq!(batch.barrier_count(), 1);
}

#[test]
fn test_single_dispatch_manual_submit() {
    let pipeline = make_pipeline();
    let buf = make_buffer(64);

    let mut batch = CommandBatch::new(BarrierStrategy::Manual);
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
        .expect("record");
    batch.submit_and_wait().expect("submit");
    assert_eq!(batch.dispatch_count(), 1);
    assert_eq!(batch.barrier_count(), 0);
}

#[test]
fn test_single_dispatch_async_pending_count() {
    let pipeline = make_pipeline();
    let buf = make_buffer(64);

    let mut batch = CommandBatch::new(BarrierStrategy::Auto);
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
        .expect("record");
    let pending = batch.submit_async().expect("async");
    assert_eq!(pending.dispatch_count(), 1);
}

// ---------------------------------------------------------------------------
// Batch statistics (dispatch count, barrier count)
// ---------------------------------------------------------------------------

#[test]
fn test_statistics_auto_large_batch() {
    let pipeline = make_pipeline();
    let buf = make_buffer(256);
    let mut batch = CommandBatch::new(BarrierStrategy::Auto);

    let n = 20;
    for _ in 0..n {
        batch
            .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
            .expect("record");
    }

    assert_eq!(batch.dispatch_count(), n);
    assert_eq!(batch.barrier_count(), n);
    // Every dispatch has a barrier after it.
    for i in 0..n as usize {
        assert!(
            batch.has_barrier_after(i),
            "dispatch {i} should have barrier after"
        );
    }
}

#[test]
fn test_statistics_manual_selective_barriers() {
    let pipeline = make_pipeline();
    let buf = make_buffer(256);
    let mut batch = CommandBatch::new(BarrierStrategy::Manual);

    // 5 dispatches with barriers after dispatch 1 and 3 only.
    for i in 0..5 {
        batch
            .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
            .expect("record");
        if i == 1 || i == 3 {
            batch.barrier();
        }
    }

    assert_eq!(batch.dispatch_count(), 5);
    assert_eq!(batch.barrier_count(), 2);
    assert!(!batch.has_barrier_after(0));
    assert!(batch.has_barrier_after(1));
    assert!(!batch.has_barrier_after(2));
    assert!(batch.has_barrier_after(3));
    assert!(!batch.has_barrier_after(4));
}

// ---------------------------------------------------------------------------
// Additional edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_batch_manual_barrier_no_auto_barriers() {
    let pipeline = make_pipeline();
    let buf = make_buffer(256);

    let mut batch = CommandBatch::new(BarrierStrategy::Manual);
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
        .expect("record 1");
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
        .expect("record 2");

    assert_eq!(batch.dispatch_count(), 2);
    assert_eq!(batch.barrier_count(), 0);
    assert!(!batch.has_barrier_after(0));
    assert!(!batch.has_barrier_after(1));
}

#[test]
fn test_batch_zero_group_rejected() {
    let pipeline = make_pipeline();
    let buf = make_buffer(256);

    let mut batch = CommandBatch::new(BarrierStrategy::Auto);
    assert!(batch
        .record(&pipeline, &[&buf], &[0u8; 4], [0, 1, 1])
        .is_err());
    assert!(batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 0, 1])
        .is_err());
    assert!(batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 0])
        .is_err());
}

#[test]
fn test_batch_zero_group_all_zeros() {
    let pipeline = make_pipeline();
    let buf = make_buffer(256);

    let mut batch = CommandBatch::new(BarrierStrategy::Auto);
    assert!(batch
        .record(&pipeline, &[&buf], &[0u8; 4], [0, 0, 0])
        .is_err());
    assert_eq!(batch.dispatch_count(), 0);
}

#[test]
fn test_batch_strategy_getter() {
    let auto_batch = CommandBatch::new(BarrierStrategy::Auto);
    assert_eq!(auto_batch.strategy(), BarrierStrategy::Auto);

    let manual_batch = CommandBatch::new(BarrierStrategy::Manual);
    assert_eq!(manual_batch.strategy(), BarrierStrategy::Manual);
}

#[test]
fn test_has_barrier_after_out_of_bounds() {
    let batch = CommandBatch::new(BarrierStrategy::Auto);
    assert!(!batch.has_barrier_after(0));
    assert!(!batch.has_barrier_after(100));
    assert!(!batch.has_barrier_after(usize::MAX));
}

#[test]
fn test_batch_multiple_buffers_per_dispatch() {
    let pipeline = make_pipeline();
    let buf_a = make_buffer(256);
    let buf_b = make_buffer(512);
    let buf_c = make_buffer(1024);

    let mut batch = CommandBatch::new(BarrierStrategy::Auto);
    batch
        .record(&pipeline, &[&buf_a, &buf_b, &buf_c], &[0u8; 4], [4, 2, 1])
        .expect("record with 3 buffers");

    assert_eq!(batch.dispatch_count(), 1);
    assert_eq!(batch.barrier_count(), 1);
}

#[test]
fn test_batch_empty_buffer_list() {
    let pipeline = make_pipeline();

    let mut batch = CommandBatch::new(BarrierStrategy::Auto);
    // Empty buffer list is allowed at the command batch level
    // (descriptor validation is deferred to actual Vulkan dispatch).
    batch
        .record(&pipeline, &[], &[0u8; 4], [1, 1, 1])
        .expect("record with no buffers");
    assert_eq!(batch.dispatch_count(), 1);
}

#[test]
fn test_batch_3d_workgroup_dispatch() {
    let pipeline = make_pipeline();
    let buf = make_buffer(256);

    let mut batch = CommandBatch::new(BarrierStrategy::Auto);
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [64, 32, 16])
        .expect("3D dispatch");
    assert_eq!(batch.dispatch_count(), 1);
}

#[test]
fn test_batch_debug_format() {
    let batch = CommandBatch::new(BarrierStrategy::Auto);
    let dbg = format!("{batch:?}");
    assert!(dbg.contains("CommandBatch"));
}

#[test]
fn test_batch_consecutive_barriers_manual() {
    let pipeline = make_pipeline();
    let buf = make_buffer(256);

    let mut batch = CommandBatch::new(BarrierStrategy::Manual);
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
        .expect("record");
    // Two consecutive barriers after the same dispatch.
    batch.barrier();
    batch.barrier();

    // Both barrier() calls increment barrier_count.
    assert_eq!(batch.barrier_count(), 2);
    assert!(batch.has_barrier_after(0));
}
