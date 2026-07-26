// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the GPU memory profiler.

use super::*;

#[test]
fn test_profiler_empty_snapshot_is_zero() {
    GpuMemoryProfiler::reset();
    let snap = GpuMemoryProfiler::snapshot();
    assert_eq!(snap.total_allocated, 0);
    assert_eq!(snap.total_live, 0);
    assert_eq!(snap.peak_live, 0);
    assert_eq!(snap.buffer_count, 0);
    assert_eq!(snap.largest_buffer, 0);
}

#[test]
fn test_profiler_single_allocation_tracked() {
    GpuMemoryProfiler::reset();
    GpuMemoryProfiler::record_allocation(4096, "weights_0", BufferCategory::Weights);

    let snap = GpuMemoryProfiler::snapshot();
    assert_eq!(snap.total_allocated, 4096);
    assert_eq!(snap.total_live, 4096);
    assert_eq!(snap.peak_live, 4096);
    assert_eq!(snap.buffer_count, 1);
    assert_eq!(snap.largest_buffer, 4096);
}

#[test]
fn test_profiler_allocation_then_deallocation() {
    GpuMemoryProfiler::reset();
    GpuMemoryProfiler::record_allocation(1024, "scratch_0", BufferCategory::Scratch);
    GpuMemoryProfiler::record_allocation(2048, "scratch_1", BufferCategory::Scratch);
    GpuMemoryProfiler::record_deallocation(1024);

    let snap = GpuMemoryProfiler::snapshot();
    assert_eq!(snap.total_allocated, 3072);
    assert_eq!(snap.total_live, 2048);
    assert_eq!(snap.peak_live, 3072);
    assert_eq!(snap.buffer_count, 1);
    assert_eq!(snap.largest_buffer, 2048);
}

#[test]
fn test_profiler_peak_survives_deallocation() {
    GpuMemoryProfiler::reset();
    GpuMemoryProfiler::record_allocation(8192, "big_buf", BufferCategory::Activations);
    GpuMemoryProfiler::record_deallocation(8192);
    GpuMemoryProfiler::record_allocation(1024, "small_buf", BufferCategory::Activations);

    let snap = GpuMemoryProfiler::snapshot();
    assert_eq!(snap.total_allocated, 9216);
    assert_eq!(snap.total_live, 1024);
    assert_eq!(snap.peak_live, 8192, "peak should reflect the high-water mark");
    assert_eq!(snap.buffer_count, 1);
    assert_eq!(snap.largest_buffer, 8192, "largest_buffer tracks all-time max");
}

#[test]
fn test_profiler_breakdown_by_category() {
    GpuMemoryProfiler::reset();
    GpuMemoryProfiler::record_allocation(4096, "w0", BufferCategory::Weights);
    GpuMemoryProfiler::record_allocation(2048, "act0", BufferCategory::Activations);
    GpuMemoryProfiler::record_allocation(512, "tmp", BufferCategory::Scratch);
    GpuMemoryProfiler::record_allocation(256, "misc", BufferCategory::Other);

    let bd = GpuMemoryProfiler::breakdown();
    assert_eq!(bd.weights, 4096);
    assert_eq!(bd.activations, 2048);
    assert_eq!(bd.scratch, 512);
    assert_eq!(bd.other, 256);
    assert_eq!(bd.total(), 4096 + 2048 + 512 + 256);
}

#[test]
fn test_profiler_breakdown_updates_on_deallocation() {
    GpuMemoryProfiler::reset();
    GpuMemoryProfiler::record_allocation(4096, "w0", BufferCategory::Weights);
    GpuMemoryProfiler::record_allocation(2048, "act0", BufferCategory::Activations);
    GpuMemoryProfiler::record_deallocation(2048); // removes the activation

    let bd = GpuMemoryProfiler::breakdown();
    assert_eq!(bd.weights, 4096);
    assert_eq!(bd.activations, 0);
    assert_eq!(bd.total(), 4096);
}

#[test]
fn test_profiler_reset_clears_everything() {
    GpuMemoryProfiler::reset();
    GpuMemoryProfiler::record_allocation(4096, "w0", BufferCategory::Weights);
    GpuMemoryProfiler::record_allocation(2048, "act0", BufferCategory::Activations);

    GpuMemoryProfiler::reset();

    let snap = GpuMemoryProfiler::snapshot();
    assert_eq!(snap.total_allocated, 0);
    assert_eq!(snap.total_live, 0);
    assert_eq!(snap.peak_live, 0);
    assert_eq!(snap.buffer_count, 0);
    assert_eq!(snap.largest_buffer, 0);

    let bd = GpuMemoryProfiler::breakdown();
    assert_eq!(bd.total(), 0);
}

#[test]
fn test_profiler_deallocation_of_untracked_buffer_clamps_to_zero() {
    GpuMemoryProfiler::reset();
    GpuMemoryProfiler::record_allocation(1024, "buf", BufferCategory::Other);
    // Deallocate more than allocated -- should clamp to zero, not underflow.
    GpuMemoryProfiler::record_deallocation(2048);

    let snap = GpuMemoryProfiler::snapshot();
    assert_eq!(snap.total_live, 0, "saturating_sub prevents underflow");
}

#[test]
fn test_profiler_multiple_same_size_allocations() {
    GpuMemoryProfiler::reset();
    GpuMemoryProfiler::record_allocation(1024, "a", BufferCategory::Activations);
    GpuMemoryProfiler::record_allocation(1024, "b", BufferCategory::Activations);
    GpuMemoryProfiler::record_allocation(1024, "c", BufferCategory::Activations);

    let snap = GpuMemoryProfiler::snapshot();
    assert_eq!(snap.buffer_count, 3);
    assert_eq!(snap.total_live, 3072);

    // Deallocate one -- should remove exactly one.
    GpuMemoryProfiler::record_deallocation(1024);
    let snap = GpuMemoryProfiler::snapshot();
    assert_eq!(snap.buffer_count, 2);
    assert_eq!(snap.total_live, 2048);
}

#[test]
fn test_profiler_live_allocation_count() {
    GpuMemoryProfiler::reset();
    assert_eq!(GpuMemoryProfiler::live_allocation_count(), 0);

    GpuMemoryProfiler::record_allocation(512, "x", BufferCategory::Other);
    GpuMemoryProfiler::record_allocation(256, "y", BufferCategory::Weights);
    assert_eq!(GpuMemoryProfiler::live_allocation_count(), 2);

    GpuMemoryProfiler::record_deallocation(512);
    assert_eq!(GpuMemoryProfiler::live_allocation_count(), 1);
}

#[test]
fn test_snapshot_display_does_not_panic() {
    GpuMemoryProfiler::reset();
    GpuMemoryProfiler::record_allocation(1_048_576, "1mb", BufferCategory::Weights);
    let snap = GpuMemoryProfiler::snapshot();
    let text = format!("{snap}");
    assert!(text.contains("GPU Memory Snapshot"));
    assert!(text.contains("total_allocated"));
}

#[test]
fn test_breakdown_display_does_not_panic() {
    GpuMemoryProfiler::reset();
    GpuMemoryProfiler::record_allocation(1024, "w", BufferCategory::Weights);
    let bd = GpuMemoryProfiler::breakdown();
    let text = format!("{bd}");
    assert!(text.contains("GPU Memory Breakdown"));
    assert!(text.contains("weights"));
}

#[test]
fn test_buffer_category_display() {
    assert_eq!(format!("{}", BufferCategory::Weights), "weights");
    assert_eq!(format!("{}", BufferCategory::Activations), "activations");
    assert_eq!(format!("{}", BufferCategory::Scratch), "scratch");
    assert_eq!(format!("{}", BufferCategory::Other), "other");
}
