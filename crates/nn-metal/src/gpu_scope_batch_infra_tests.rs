#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Command buffer batching infrastructure tests.
//!
//! Covers gaps in lazy evaluation, submit/sync flows, flush idempotency,
//! discard_pending_batch, ensure_batch_for_blit, ScopeExitMode RAII,
//! and arena+flush integration.
//!
//! Part of the GPU lazy evaluation test suite, extracted for 500-line compliance.

use super::*;
use crate::dispatch_stats::{dispatch_stats, reset_counters};
use crate::test_common::init;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// Flush idempotency and empty-batch behavior
// ---------------------------------------------------------------------------

/// Double flush does not error: second flush is a no-op (no pending batch).
#[test]
fn test_flush_idempotent_double_flush() {
    init();

    let device = Device::metal();
    let a = DynTensor::full(&[4], 1.0, DType::F32, &device).unwrap();
    let b = DynTensor::full(&[4], 2.0, DType::F32, &device).unwrap();
    let _c = a.add(&b).unwrap(); // Encodes into lazy batch.

    // First flush commits.
    assert!(flush().is_ok(), "first flush should succeed");
    assert!(!is_lazy_batch_active(), "batch consumed after first flush");

    // Second flush is a no-op (no pending batch).
    assert!(flush().is_ok(), "second flush should succeed (no-op)");
    assert!(!is_lazy_batch_active(), "still no active batch");
}

/// flush() on a fresh thread with no Metal ops is safe.
#[test]
fn test_flush_fresh_thread_no_ops() {
    init();
    let result = flush();
    assert!(result.is_ok(), "flush on empty state should succeed");
    assert_eq!(pending_encoding_count(), 0);
}

// ---------------------------------------------------------------------------
// Submit + sync flow
// ---------------------------------------------------------------------------

/// submit() + sync() happy path: non-blocking commit then wait for completion.
#[test]
fn test_submit_then_sync_produces_correct_result() {
    init();

    let device = Device::metal();
    let a = DynTensor::full(&[4], 5.0, DType::F32, &device).unwrap();
    let b = DynTensor::full(&[4], 3.0, DType::F32, &device).unwrap();
    let c = a.add(&b).unwrap(); // Encodes into lazy batch.

    // submit() non-blocking commit.
    assert!(submit().is_ok(), "submit should succeed");
    assert!(!is_lazy_batch_active(), "batch consumed by submit");

    // sync() waits for GPU completion.
    assert!(sync().is_ok(), "sync should succeed");

    // Now CPU readback should work — GPU work is complete.
    let vals = c
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals, vec![8.0; 4]);
}

/// submit() when no batch is pending is a no-op.
#[test]
fn test_submit_noop_when_no_batch() {
    init();
    flush().unwrap(); // Clear any prior state.

    assert!(!is_lazy_batch_active());
    let result = submit();
    assert!(result.is_ok(), "submit with no batch should succeed (no-op)");
}

/// sync() when no pending submitted batch is a no-op.
#[test]
fn test_sync_noop_when_no_pending() {
    init();
    flush().unwrap(); // Clear any prior state.

    let result = sync();
    assert!(result.is_ok(), "sync with no pending batch should succeed");
}

/// submit() + submit() serializes: second submit waits for first.
#[test]
fn test_submit_serializes_sequential_batches() {
    init();

    let device = Device::metal();
    let a = DynTensor::full(&[4], 1.0, DType::F32, &device).unwrap();

    // First batch.
    let b = a.add_scalar(1.0).unwrap(); // 2.0
    submit().unwrap();

    // Second batch — submit() calls sync() internally for the first.
    let c = b.add_scalar(1.0).unwrap(); // 3.0
    submit().unwrap();

    // sync waits for the second submit.
    sync().unwrap();

    let vals = c
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals, vec![3.0; 4]);
}

/// submit() increments TOTAL_SUBMITS counter.
#[test]
fn test_submit_increments_stats_counter() {
    init();
    flush().unwrap();
    reset_counters();

    let device = Device::metal();
    let a = DynTensor::full(&[4], 1.0, DType::F32, &device).unwrap();
    let _b = a.add_scalar(1.0).unwrap(); // Encode into batch.

    let stats_before = dispatch_stats();
    submit().unwrap();
    let stats_after = dispatch_stats();

    assert_eq!(
        stats_after.submits - stats_before.submits,
        1,
        "submit should increment TOTAL_SUBMITS by 1"
    );

    // Clean up.
    sync().unwrap();
}

// ---------------------------------------------------------------------------
// discard_pending_batch
// ---------------------------------------------------------------------------

/// discard_pending_batch clears lazy batch state without committing.
#[test]
fn test_discard_pending_batch_clears_state() {
    init();

    let device = Device::metal();
    let a = DynTensor::full(&[4], 1.0, DType::F32, &device).unwrap();
    let _b = a.add_scalar(1.0).unwrap(); // Encode into lazy batch.

    assert!(is_lazy_batch_active(), "batch should be active before discard");
    assert!(pending_encoding_count() > 0, "should have pending encodings");

    discard_pending_batch();

    assert!(!is_lazy_batch_active(), "batch cleared after discard");
    assert_eq!(pending_encoding_count(), 0, "encoding count reset after discard");
}

/// discard_pending_batch is a no-op when no batch exists.
#[test]
fn test_discard_pending_batch_noop_when_empty() {
    init();
    flush().unwrap();

    assert!(!is_lazy_batch_active());
    discard_pending_batch(); // Should not panic.
    assert!(!is_lazy_batch_active());
    assert_eq!(pending_encoding_count(), 0);
}

/// After discard, new GPU ops work correctly with a fresh batch.
#[test]
fn test_discard_then_new_ops_succeed() {
    init();

    let device = Device::metal();
    let a = DynTensor::full(&[4], 2.0, DType::F32, &device).unwrap();
    let _b = a.add_scalar(1.0).unwrap(); // Encode into lazy batch.

    // Discard the batch (simulating error recovery).
    discard_pending_batch();
    assert!(!is_lazy_batch_active());

    // New ops should work on a fresh batch.
    let c = a.mul_scalar(3.0).unwrap(); // 6.0
    let vals = c
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals, vec![6.0; 4]);
}

// ---------------------------------------------------------------------------
// ensure_batch_for_blit
// ---------------------------------------------------------------------------

/// ensure_batch_for_blit increments TOTAL_BLITS, not TOTAL_ENCODINGS.
#[test]
fn test_ensure_batch_for_blit_increments_blit_counter() {
    init();
    flush().unwrap();
    reset_counters();

    ensure_batch_for_blit().unwrap();

    let stats = dispatch_stats();
    assert_eq!(stats.blits, 1, "blit counter should be 1");
    assert_eq!(stats.compute_encodings, 0, "compute counter should be 0");

    // Clean up.
    flush().unwrap();
}

/// get_or_create_batch increments TOTAL_ENCODINGS, not TOTAL_BLITS.
#[test]
fn test_get_or_create_batch_increments_compute_counter() {
    init();
    flush().unwrap();
    reset_counters();

    get_or_create_batch().unwrap();

    let stats = dispatch_stats();
    assert_eq!(stats.compute_encodings, 1, "compute counter should be 1");
    assert_eq!(stats.blits, 0, "blit counter should be 0");

    // Clean up.
    flush().unwrap();
}

/// Mixed compute + blit encodings tracked separately.
#[test]
fn test_mixed_compute_and_blit_tracking() {
    init();
    flush().unwrap();
    reset_counters();

    get_or_create_batch().unwrap(); // compute: 1
    get_or_create_batch().unwrap(); // compute: 2
    ensure_batch_for_blit().unwrap(); // blit: 1
    get_or_create_batch().unwrap(); // compute: 3
    ensure_batch_for_blit().unwrap(); // blit: 2

    let stats = dispatch_stats();
    assert_eq!(stats.compute_encodings, 3, "compute count");
    assert_eq!(stats.blits, 2, "blit count");

    // Both kinds count toward encoding_count (auto-flush threshold).
    assert_eq!(pending_encoding_count(), 5, "total encoding count = compute + blits");

    // Clean up.
    flush().unwrap();
}

// ---------------------------------------------------------------------------
// ScopeExitMode RAII behavior
// ---------------------------------------------------------------------------

/// with_scope_exit_mode restores previous mode on normal return.
#[test]
fn test_scope_exit_mode_raii_restores_on_normal_return() {
    init();

    // Default mode is Flush.
    let default_mode = SCOPE_EXIT_MODE.with(Cell::get);
    assert_eq!(default_mode, ScopeExitMode::Flush);

    with_scope_exit_mode(ScopeExitMode::Submit, || {
        let inner = SCOPE_EXIT_MODE.with(Cell::get);
        assert_eq!(inner, ScopeExitMode::Submit, "mode should be Submit inside scope");
    });

    let restored = SCOPE_EXIT_MODE.with(Cell::get);
    assert_eq!(restored, ScopeExitMode::Flush, "mode should be restored after scope");
}

/// with_scope_exit_mode restores previous mode on panic.
#[test]
fn test_scope_exit_mode_raii_restores_on_panic() {
    init();

    let default_mode = SCOPE_EXIT_MODE.with(Cell::get);
    assert_eq!(default_mode, ScopeExitMode::Flush);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_scope_exit_mode(ScopeExitMode::Submit, || {
            panic!("test panic inside scope_exit_mode");
        });
    }));
    assert!(result.is_err(), "panic should propagate");

    let restored = SCOPE_EXIT_MODE.with(Cell::get);
    assert_eq!(restored, ScopeExitMode::Flush, "mode restored after panic");
}

/// Nested with_scope_exit_mode preserves outer mode.
#[test]
fn test_scope_exit_mode_nesting() {
    init();

    with_scope_exit_mode(ScopeExitMode::Submit, || {
        let outer = SCOPE_EXIT_MODE.with(Cell::get);
        assert_eq!(outer, ScopeExitMode::Submit);

        with_scope_exit_mode(ScopeExitMode::Flush, || {
            let inner = SCOPE_EXIT_MODE.with(Cell::get);
            assert_eq!(inner, ScopeExitMode::Flush, "inner overrides to Flush");
        });

        let restored = SCOPE_EXIT_MODE.with(Cell::get);
        assert_eq!(restored, ScopeExitMode::Submit, "outer Submit restored");
    });

    let final_mode = SCOPE_EXIT_MODE.with(Cell::get);
    assert_eq!(final_mode, ScopeExitMode::Flush, "default Flush restored");
}

/// with_gpu_scope in Submit mode calls submit() instead of flush() on success.
#[test]
fn test_gpu_scope_submit_mode_calls_submit() {
    init();
    flush().unwrap();
    reset_counters();

    let device = Device::metal();
    let a = DynTensor::full(&[4], 2.0, DType::F32, &device).unwrap();

    let result = with_scope_exit_mode(ScopeExitMode::Submit, || {
        with_gpu_scope(|| a.add_scalar(1.0))
    });
    assert!(result.is_ok());

    let stats = dispatch_stats();
    assert!(stats.submits >= 1, "Submit mode should increment submits counter");

    // Must sync to wait for GPU completion before readback.
    sync().unwrap();

    let vals = result
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals, vec![3.0; 4]);
}

// ---------------------------------------------------------------------------
// Arena + flush integration
// ---------------------------------------------------------------------------

/// flush() resets the default arena generation (arena reuse after commit).
#[test]
fn test_flush_resets_default_arena() {
    init();

    let device = Device::metal();
    let a = DynTensor::full(&[4], 1.0, DType::F32, &device).unwrap();

    // Perform a GPU op to trigger arena allocation.
    let _b = a.add_scalar(1.0).unwrap();

    let gen_before = crate::arena::default_arena_generation();

    // flush commits and resets the arena.
    flush().unwrap();

    let gen_after = crate::arena::default_arena_generation();

    // Arena generation should advance after flush (reset increments generation).
    if let (Some(before), Some(after)) = (gen_before, gen_after) {
        assert!(
            after > before,
            "arena generation should advance after flush: before={before}, after={after}"
        );
    }
}

/// Arena stats track allocations through flush boundaries.
#[test]
fn test_arena_stats_through_flush_boundaries() {
    init();
    flush().unwrap();
    reset_counters();

    let device = Device::metal();
    let a = DynTensor::full(&[4], 1.0, DType::F32, &device).unwrap();

    // Several ops, each triggering arena allocation.
    let _b = a.add_scalar(1.0).unwrap();
    let _c = a.mul_scalar(2.0).unwrap();
    let _d = a.add_scalar(3.0).unwrap();

    let stats = dispatch_stats();
    // Arena should have been used for these intermediate buffers.
    let total_allocs = stats.arena.hits + stats.arena.misses;
    assert!(total_allocs > 0, "arena should have served allocations: {stats:?}");

    flush().unwrap();
}

/// Multiple GPU op chains with intermediate flushes all produce correct results.
#[test]
fn test_multiple_chains_with_intermediate_flushes() {
    init();

    let device = Device::metal();
    let x = DynTensor::full(&[4], 1.0, DType::F32, &device).unwrap();

    // Chain 1: x + 1 = 2
    let a = x.add_scalar(1.0).unwrap();
    flush().unwrap();
    let vals_a = a
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals_a, vec![2.0; 4]);

    // Chain 2: a * 3 = 6
    let b = a.mul_scalar(3.0).unwrap();
    flush().unwrap();
    let vals_b = b
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals_b, vec![6.0; 4]);

    // Chain 3: b - 1 = 5
    let c = b.add_scalar(-1.0).unwrap();
    flush().unwrap();
    let vals_c = c
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals_c, vec![5.0; 4]);
}

/// Encoding count resets after flush.
#[test]
fn test_encoding_count_resets_after_flush() {
    init();
    flush().unwrap();

    // Accumulate some encodings.
    for _ in 0..5 {
        get_or_create_batch().unwrap();
    }
    assert_eq!(pending_encoding_count(), 5);

    flush().unwrap();
    assert_eq!(pending_encoding_count(), 0, "count resets after flush");

    // New encodings start from 0.
    get_or_create_batch().unwrap();
    assert_eq!(pending_encoding_count(), 1);

    flush().unwrap();
}

/// try_reset_active_arena returns true when default arena exists.
#[test]
fn test_try_reset_active_arena_with_default_arena() {
    init();

    let device = Device::metal();
    let a = DynTensor::full(&[4], 1.0, DType::F32, &device).unwrap();
    let _b = a.add_scalar(1.0).unwrap(); // Triggers default arena creation.
    flush().unwrap();

    // Default arena should exist now.
    let did_reset = crate::arena::try_reset_active_arena();
    assert!(did_reset, "should reset the default arena");

    flush().unwrap();
}
