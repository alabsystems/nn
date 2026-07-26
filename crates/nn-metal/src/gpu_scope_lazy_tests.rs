#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Lazy batching, auto-flush, error drop, dispatch reduction, and flush
//! safety guard tests. Extracted from `gpu_scope_tests.rs` for 500-line
//! compliance.

use super::*;
use crate::dispatch_stats::{dispatch_stats, reset_counters};
use crate::error::MetalError;
use crate::metal_backend::global_metal_context;
use crate::test_common::init;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

// -- Always-on lazy batching (#2009) ------------------------------------------
// GPU ops WITHOUT `with_gpu_scope` — lazy batching is always active and
// `flush()` is called transparently by CPU readback paths.

/// GPU ops without explicit scope produce correct results via auto-flush.
///
/// This is the core #2009 invariant: `a.add(&b)` encodes into the lazy
/// batch, and `to_device(Cpu)` calls `flush()` before reading.
#[test]
fn test_lazy_batch_no_scope_produces_correct_result() {
    init();

    let device = Device::metal();
    let a = DynTensor::full(&[4], 2.0, DType::F32, &device).unwrap();
    let b = DynTensor::full(&[4], 3.0, DType::F32, &device).unwrap();

    // No with_gpu_scope — lazy batch handles everything.
    let c = a.add(&b).unwrap();
    let vals = c
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals, vec![5.0; 4]);
}

/// Multi-op chain without scope — all ops batch, flush on readback.
#[test]
fn test_lazy_batch_multi_op_chain_no_scope() {
    init();

    let device = Device::metal();
    let x = DynTensor::full(&[8], 3.0, DType::F32, &device).unwrap();

    // 3 ops encoded into lazy batch, flushed on to_device(Cpu).
    let a = x.mul_scalar(2.0).unwrap(); // 6.0
    let b = a.add_scalar(1.0).unwrap(); // 7.0
    let c = b.neg().unwrap(); // -7.0

    let vals = c
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals, vec![-7.0; 8]);
}

/// Matmul without scope — verifies the primary production path.
#[test]
fn test_lazy_batch_matmul_no_scope() {
    init();

    let device = Device::metal();
    let a_cpu = DynTensor::new(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(
        &[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0],
        &[3, 2],
        &Device::Cpu,
    )
    .unwrap();
    let a = a_cpu.to_device(&device).unwrap();
    let b = b_cpu.to_device(&device).unwrap();

    let result = a.matmul(&b).unwrap();
    let vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    // [[58, 64], [139, 154]]
    assert_eq!(vals, vec![58.0, 64.0, 139.0, 154.0]);
}

/// Multiple independent readbacks — each flush is independent.
#[test]
fn test_lazy_batch_multiple_readbacks() {
    init();

    let device = Device::metal();
    let a = DynTensor::full(&[4], 2.0, DType::F32, &device).unwrap();
    let b = DynTensor::full(&[4], 3.0, DType::F32, &device).unwrap();

    // First op + readback.
    let c = a.add(&b).unwrap();
    let vals1 = c
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals1, vec![5.0; 4]);

    // Second op + readback — new lazy batch created automatically.
    let d = c.mul(&a).unwrap();
    let vals2 = d
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals2, vec![10.0; 4]);
}

/// flush() is a no-op when no batch is pending.
#[test]
fn test_flush_noop_when_no_batch() {
    init();
    // No GPU ops issued — flush should succeed silently.
    assert!(flush().is_ok());
    assert!(!is_lazy_batch_active());
}

/// CPU readback inside with_gpu_scope flushes and succeeds.
#[test]
fn test_readback_inside_scope_succeeds() {
    init();
    let device = Device::metal();
    let a = DynTensor::full(&[4], 7.0, DType::F32, &device).unwrap();
    let result = with_gpu_scope(|| {
        let b = a.mul_scalar(3.0)?;
        let vals = b.to_device(&Device::Cpu)?.to_flat_vec::<f32>()?;
        assert_eq!(vals, vec![21.0; 4]);
        Ok(vals)
    })
    .unwrap();
    assert_eq!(result, vec![21.0; 4]);
}

// -- Error path drop behavior (#2031 AC3) -------------------------------------

/// When `with_gpu_scope` errors, the pending lazy batch (including any
/// pre-scope encodings) is dropped without committing. This test verifies
/// that GPU work encoded before the scope is discarded when the scope
/// closure returns an error.
///
/// The lazy batch is shared across scope boundaries — pre-scope encodings
/// live in the same command buffer. Dropping the batch on error means
/// Metal discards the uncommitted command buffer via ObjC ARC.
#[test]
fn test_scope_error_drops_pre_scope_encodings() {
    init();

    let device = Device::metal();
    let a = DynTensor::full(&[4], 2.0, DType::F32, &device).unwrap();
    let b = DynTensor::full(&[4], 3.0, DType::F32, &device).unwrap();

    // Encode a GPU op BEFORE the scope — this goes into the lazy batch.
    let pre_scope_result = a.add(&b).unwrap();

    // Now enter a scope that errors. The error path drops the entire lazy
    // batch, including the pre-scope encoding.
    let scope_result: nn_core::Result<DynTensor> = with_gpu_scope(|| {
        // This encoding also goes into the same lazy batch.
        let _inner = a.mul(&b).unwrap();
        // Force an error.
        Err(TensorError::InvalidShape("deliberate error".into()))
    });

    assert!(scope_result.is_err());
    // The batch was dropped — no pending work.
    assert!(!is_lazy_batch_active());

    // The pre-scope tensor still holds a MetalBuffer handle, but the
    // command buffer that would have written its data was dropped.
    // Reading it back should still succeed (flush creates a new empty batch
    // and commits it — the buffer contents are whatever Metal left there).
    // The key invariant: the scope cleanup didn't panic or leave dangling state.
    let readback = pre_scope_result.to_device(&Device::Cpu);
    assert!(readback.is_ok(), "readback after error should not panic");
}

// -- Dispatch reduction benchmark (#2009 AC2) ---------------------------------

/// Measures dispatch reduction ratio for a multi-op chain.
///
/// Without lazy batching, each GPU op issues its own `commit_and_wait`.
/// With lazy batching, N ops share 1 flush on CPU readback.
/// AC2 target: >10x dispatch reduction on a representative workload.
#[test]
fn test_dispatch_reduction_ratio() {
    init();
    let device = Device::metal();

    reset_counters();

    // 12 GPU ops in a chain, single readback at the end.
    let x = DynTensor::full(&[64], 1.0, DType::F32, &device).unwrap();
    let a = x.mul_scalar(2.0).unwrap();
    let b = a.add_scalar(1.0).unwrap();
    let c = b.neg().unwrap();
    let d = c.mul_scalar(0.5).unwrap();
    let e = d.add_scalar(10.0).unwrap();
    let f = e.neg().unwrap();
    let g = f.mul_scalar(3.0).unwrap();
    let h = g.add_scalar(-1.0).unwrap();
    let i = h.neg().unwrap();
    let j = i.mul_scalar(0.1).unwrap();
    let k = j.add_scalar(5.0).unwrap();
    let result = k.neg().unwrap();

    // Single readback triggers flush.
    let vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals.len(), 64);

    let stats = dispatch_stats();
    assert!(
        stats.compute_encodings >= 12,
        "expected >=12 encodings, got {}",
        stats.compute_encodings
    );
    assert!(
        stats.flushes >= 1,
        "expected >=1 flush, got {}",
        stats.flushes
    );

    let ratio = stats.compute_encodings as f64 / stats.flushes as f64;
    assert!(
        ratio >= 10.0,
        "dispatch reduction ratio {ratio:.1}x ({} encodings / {} flushes) < 10x target",
        stats.compute_encodings,
        stats.flushes,
    );
}

// -- Flush safety guards (#2041) ----------------------------------------------

/// Verify `pending_encoding_count()` reflects actual lazy batch state.
#[test]
fn test_pending_encoding_count_tracks_lazy_batch() {
    init();

    // No encodings initially.
    flush().unwrap();
    assert_eq!(pending_encoding_count(), 0);

    // Trigger a GPU dispatch to increment the count.
    let device = Device::metal();
    let _a = DynTensor::full(&[4], 1.0, DType::F32, &device).unwrap();
    // `full` on GPU creates a buffer directly (no dispatch encoding).
    // An actual dispatch (e.g., add) goes through get_or_create_batch.
    let b = DynTensor::full(&[4], 2.0, DType::F32, &device).unwrap();
    let _c = _a.add(&b).unwrap(); // This encodes into lazy batch.

    assert!(
        pending_encoding_count() > 0,
        "expected >0 pending encodings after GPU dispatch"
    );

    // Flush resets the count.
    flush().unwrap();
    assert_eq!(pending_encoding_count(), 0);
}

/// `clone_buffer` after flush succeeds (no debug_assert violation).
#[test]
fn test_clone_buffer_after_flush_succeeds() {
    init();

    let device = Device::metal();
    let a = DynTensor::full(&[4], 42.0, DType::F32, &device).unwrap();
    let _b = a.mul_scalar(2.0).unwrap(); // Encodes into lazy batch.

    // Flush before clone — satisfies the debug_assert contract.
    flush().unwrap();

    let ctx = global_metal_context().unwrap();
    // Access the underlying buffer for clone_buffer test.
    // Use a known-good buffer: create a fresh one directly.
    let src = ctx.create_buffer(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let cloned = ctx.clone_buffer(&src);
    assert!(cloned.is_ok(), "clone_buffer after flush should succeed");
}

/// `clone_buffer` with pending encodings returns `PendingFlushRequired` error.
///
/// This test verifies the guard added in #2041 catches the contract violation
/// that caused P1 bugs #1912 and #1933. `clone_buffer` returns a proper
/// `Result` error in all build modes (debug and release).
#[test]
fn test_clone_buffer_without_flush_returns_error() {
    init();

    let device = Device::metal();
    let a = DynTensor::full(&[4], 1.0, DType::F32, &device).unwrap();
    let b = DynTensor::full(&[4], 2.0, DType::F32, &device).unwrap();
    let _c = a.add(&b).unwrap(); // Encodes into lazy batch — NOT flushed.

    // clone_buffer without flush — should return PendingFlushRequired error.
    let ctx = global_metal_context().unwrap();
    let src = ctx.create_buffer(&[1.0_f32, 2.0]).unwrap();
    let result = ctx.clone_buffer(&src);
    assert!(
        matches!(&result, Err(MetalError::PendingFlushRequired { .. })),
        "expected PendingFlushRequired error, got: {result:?}"
    );
}

/// `encode_custom_dispatch` encodes into the shared lazy batch.
#[test]
fn test_encode_custom_dispatch_shares_lazy_batch() {
    init();
    flush().unwrap();
    reset_counters();

    let device = Device::metal();
    // Native nn op — encodes into lazy batch.
    let a = DynTensor::full(&[4], 1.0, DType::F32, &device).unwrap();
    let b = DynTensor::full(&[4], 2.0, DType::F32, &device).unwrap();
    let _c = a.add(&b).unwrap();

    // Custom dispatch — encodes into the SAME lazy batch.
    let result = encode_custom_dispatch(|_batch| -> Result<(), String> {
        // No-op encoding for test — just verify we get a valid batch ref.
        Ok(())
    });
    assert!(result.is_ok(), "encode_custom_dispatch should succeed");
    assert!(result.unwrap().is_ok(), "callback should succeed");

    // Both encodings share one batch — flush produces exactly 1 commit.
    let stats_before = dispatch_stats();
    flush().unwrap();
    let stats_after = dispatch_stats();
    assert_eq!(
        stats_after.flushes - stats_before.flushes,
        1,
        "native + custom should share one flush"
    );
}

/// `encode_custom_dispatch` propagates callback errors through Ok(Err(E)).
#[test]
fn test_encode_custom_dispatch_propagates_callback_error() {
    init();
    flush().unwrap();

    let result = encode_custom_dispatch(|_batch| -> Result<(), String> {
        Err("custom kernel failed".to_string())
    });
    assert!(result.is_ok(), "outer Result should be Ok (batch exists)");
    let inner = result.unwrap();
    assert!(inner.is_err(), "inner Result should carry callback error");
    assert_eq!(inner.unwrap_err(), "custom kernel failed");

    // Clean up.
    flush().unwrap();
}

// -- Submit-mode error path (#2375) -------------------------------------------

/// Error inside `with_gpu_scope` with `ScopeExitMode::Submit` cleans up both
/// the uncommitted lazy batch AND any pending submitted batch.
///
/// Without the fix from #2375 (W4-1612), the PENDING thread-local would leak
/// a stale `PendingBatch` into the next scope on the same thread. This test
/// verifies that after an error in Submit mode:
/// 1. No lazy batch remains active.
/// 2. No pending submitted batch remains.
/// 3. Subsequent GPU operations succeed (no stale state).
#[test]
fn test_submit_mode_error_cleans_pending_state() {
    init();

    let device = Device::metal();
    let a = DynTensor::full(&[4], 2.0, DType::F32, &device).unwrap();
    let b = DynTensor::full(&[4], 3.0, DType::F32, &device).unwrap();

    // Scope 1 (Submit mode, success): creates a pending batch.
    with_scope_exit_mode(ScopeExitMode::Submit, || with_gpu_scope(|| a.add(&b))).unwrap();

    // Scope 2 (Submit mode, error): must clean up both LAZY_BATCH and PENDING.
    let err_result: nn_core::Result<DynTensor> =
        with_scope_exit_mode(ScopeExitMode::Submit, || {
            with_gpu_scope(|| {
                let _c = a.mul(&b)?; // succeeds, encodes into lazy batch
                Err(TensorError::InvalidShape("deliberate error".into()))
            })
        });

    assert!(err_result.is_err(), "scope should return the error");
    assert!(
        !is_lazy_batch_active(),
        "lazy batch must be cleared after Submit-mode error"
    );

    // Scope 3 (normal operation): must succeed without stale PENDING interference.
    let recovery = with_gpu_scope(|| a.add_scalar(1.0));
    assert!(
        recovery.is_ok(),
        "GPU op after Submit-mode error should succeed, got: {:?}",
        recovery.err()
    );
    let vals = recovery
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals, vec![3.0; 4]);
}

// MAX_LAZY_ENCODINGS auto-flush tests extracted to
// `gpu_scope_autoflush_tests.rs` for 500-line compliance.
#[path = "gpu_scope_autoflush_tests.rs"]
mod autoflush_tests;

// Command buffer batching infrastructure tests (flush idempotency,
// submit/sync, discard, blit counters, ScopeExitMode RAII, arena integration).
#[path = "gpu_scope_batch_infra_tests.rs"]
mod batch_infra_tests;
