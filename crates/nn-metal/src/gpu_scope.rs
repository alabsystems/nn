// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Always-on lazy GPU command buffer batching.
//!
//! GPU dispatch calls are automatically batched into a shared Metal command
//! buffer. The batch is created lazily on the first GPU dispatch and committed
//! (flushed) automatically when:
//!
//! - CPU readback is requested ([`flush`])
//! - A [`with_gpu_scope`] fence exits
//! - The encoding count reaches [`MAX_LAZY_ENCODINGS`]
//!
//! This eliminates per-op `commit_and_wait` barriers, keeping the GPU
//! saturated across producer→consumer chains. See #2009 and
//! `designs/2026-03-12-lazy-graph-execution.md`.
//!
//! **Dispatch pattern:** GPU-resident ops use [`get_or_create_batch`] +
//! [`encode_into_lazy_batch`]. CPU readback calls [`flush`] first.
//! [`with_gpu_scope`] is a flush fence (calls [`flush`] on scope exit).
//! `NanCheckPolicy::Skip` suppresses intermediate flushes. See #1915.
//!
//! # Flush Tracing
//!
//! Set `NN_FLUSH_TRACE=1` to print a backtrace on every flush/submit/auto-flush.
//! Each trace line includes the event type, sequence number, and encoding count.

use std::cell::{Cell, RefCell};

use crate::dispatch::{CommandBatch, PendingBatch};
use crate::dispatch_stats::{
    record_gpu_event, TOTAL_BLITS, TOTAL_ENCODINGS, TOTAL_FLUSHES, TOTAL_SUBMITS,
};
use crate::metal_backend::global_metal_context;
use nn_core::TensorError;

/// Controls [`with_gpu_scope`] exit behavior (#2375). `Flush` (default)
/// commits and waits; `Submit` non-blocking submits for CPU-GPU pipelining.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeExitMode {
    /// Commit and wait at scope exit (synchronous, default).
    Flush,
    /// Submit without waiting at scope exit (non-blocking).
    Submit,
}

/// Maximum encodings per lazy batch before auto-flush. Smaller batches reduce
/// the blast radius of a single hung GPU command buffer and limit the amount
/// of GPU work that blocks on a single `commit_and_wait`. 128 prevented
/// command buffer scheduling pathology on deep models (SigLip2: 12 blocks ×
/// 17 dispatches = 204 total caused Metal GPU timeout at 1024). Prior value
/// of 4096 contributed to a system-level watchdog kernel panic.
///
/// Increased from 128 to 256 (#4264): Kokoro production pipeline has 239
/// total events (181 compute + 58 blits). At 128, Kokoro triggered 1
/// mid-pipeline auto-flush per synthesis — a `commit_and_wait` stall that
/// serialized GPU work unnecessarily. At 256, the entire Kokoro pipeline
/// fits in a single command buffer with no mid-pipeline auto-flush.
/// SigLip2 (204 dispatches) also fits without auto-flush at 256.
///
/// The StaleArenaRead fix (removing arena reset from auto-flush) makes
/// higher limits safe — auto-flush only rotates the Metal command buffer,
/// not the arena. Auto-grow (#4289) handles arena overflow independently.
///
/// Source: #4319 (SigLip2 hang), #4317 (original NaN check fix), #4264
const MAX_LAZY_ENCODINGS: usize = 256;

thread_local! {
    /// Pending GPU command batch. Created on first dispatch, committed on
    /// readback or scope exit.
    static LAZY_BATCH: RefCell<Option<CommandBatch>> = const { RefCell::new(None) };

    /// Number of encodings in the current lazy batch. Reset on flush.
    static ENCODING_COUNT: Cell<usize> = const { Cell::new(0) };

    /// Pending GPU work from the most recent [`submit`] call. At most one
    /// at a time; [`sync`] waits for it, [`flush`] calls sync first.
    static PENDING: RefCell<Option<PendingBatch>> = const { RefCell::new(None) };

    /// Controls [`with_gpu_scope`] exit behavior. See [`ScopeExitMode`].
    static SCOPE_EXIT_MODE: Cell<ScopeExitMode> = const { Cell::new(ScopeExitMode::Flush) };
}

/// Whether an encoding is a compute dispatch or a buffer-planner blit copy.
/// Determines which cumulative counter (`TOTAL_ENCODINGS` vs `TOTAL_BLITS`)
/// is incremented. Both kinds increment `ENCODING_COUNT` (auto-flush threshold).
enum BatchEncoding {
    /// Compute kernel dispatch — increments `TOTAL_ENCODINGS`.
    Compute,
    /// Buffer-planner blit copy — increments `TOTAL_BLITS`.
    Blit,
}

/// Internal: ensure a lazy batch exists, auto-flush if needed, and record
/// the encoding in the appropriate cumulative counter.
fn ensure_batch(kind: BatchEncoding) -> Result<(), TensorError> {
    LAZY_BATCH.with(|cell| {
        // Auto-flush if we've accumulated too many encodings.
        let count = ENCODING_COUNT.with(Cell::get);
        if count >= MAX_LAZY_ENCODINGS && cell.borrow().is_some() {
            // Take the batch, commit it, then let a fresh one be created below.
            if let Some(batch) = cell.borrow_mut().take() {
                TOTAL_FLUSHES.with(|c| record_gpu_event(c, "auto-flush", count));
                batch.commit_and_wait().map_err(|e| {
                    TensorError::backend_failure(
                        nn_core::BackendDomain::Metal,
                        nn_core::BackendErrorKind::DispatchFailed,
                        format!("lazy batch auto-flush failed: {e}"),
                    )
                })?;
            }
            ENCODING_COUNT.with(|c| c.set(0));

            // NOTE: Do NOT reset the default arena here. Auto-flush rotates
            // the Metal command buffer for size management, but arena-resident
            // tensors from before the rotation may still be referenced by
            // subsequent GPU dispatches or the pipeline-exit `gpu_to_cpu()`.
            // With auto-grow (#4289), the arena handles overflow by allocating
            // new slabs — no memory pressure. The arena is reset at the
            // explicit pipeline boundary (`sync()` or `flush()`), which is
            // the only point where all GPU work is known to be complete.
            //
            // Prior to this change, auto-flush reset caused `StaleArenaRead`
            // errors on production Kokoro (D=512, 200+ dispatches) because
            // the 128-encoding auto-flush threshold triggered multiple mid-
            // pipeline arena resets, advancing the generation counter far
            // beyond the allocation generation of tensors still in use.
            // Part of #4264.
        }

        if cell.borrow().is_none() {
            let ctx = global_metal_context().map_err(|e| {
                TensorError::backend_failure(
                    nn_core::BackendDomain::Metal,
                    nn_core::BackendErrorKind::Other,
                    e.to_string(),
                )
            })?;
            let batch = ctx.begin_batch().map_err(|e| {
                TensorError::backend_failure(
                    nn_core::BackendDomain::Metal,
                    nn_core::BackendErrorKind::DispatchFailed,
                    format!("lazy batch: begin_batch failed: {e}"),
                )
            })?;
            *cell.borrow_mut() = Some(batch);
            ENCODING_COUNT.with(|c| c.set(0));
        }

        ENCODING_COUNT.with(|c| c.set(c.get() + 1));
        match kind {
            BatchEncoding::Compute => TOTAL_ENCODINGS.with(|c| c.set(c.get() + 1)),
            BatchEncoding::Blit => TOTAL_BLITS.with(|c| c.set(c.get() + 1)),
        }
        Ok(())
    })
}

/// Get or create the thread-local lazy batch for a compute dispatch.
///
/// On first call, creates a new [`CommandBatch`] from the global Metal context.
/// Subsequent calls within the same batch are no-ops. Auto-flushes when
/// encoding count reaches [`MAX_LAZY_ENCODINGS`].
///
/// Increments `TOTAL_ENCODINGS` (compute-only counter). For buffer-planner
/// blit copies, use [`ensure_batch_for_blit`] instead.
pub(crate) fn get_or_create_batch() -> Result<(), TensorError> {
    ensure_batch(BatchEncoding::Compute)
}

/// Get or create the thread-local lazy batch for a buffer-planner blit copy.
///
/// Same as [`get_or_create_batch`] but increments `TOTAL_BLITS` instead of
/// `TOTAL_ENCODINGS`. This separates compute dispatches from memory copies
/// in the dispatch statistics, allowing gate tests to target compute-only
/// dispatch counts. See #1815.
pub(crate) fn ensure_batch_for_blit() -> Result<(), TensorError> {
    ensure_batch(BatchEncoding::Blit)
}

/// Ensure a lazy batch exists for `count` blit copies.
///
/// Like [`ensure_batch_for_blit`] but increments `TOTAL_BLITS` by `count`
/// and `ENCODING_COUNT` by 1 (one shared blit encoder for all copies).
/// Used by [`relocate_blits_batch`](crate::compiled_model_execute_helpers)
/// to encode multiple relocations via [`CommandBatch::blit_copy_batch`].
///
/// Part of #4264 (R4: blit encoder batching).
pub(crate) fn ensure_batch_for_blit_batch(count: usize) -> Result<(), TensorError> {
    if count == 0 {
        return Ok(());
    }
    // One encoding (one blit encoder), but `count` blit copies.
    ensure_batch(BatchEncoding::Blit)?;
    // ensure_batch already incremented TOTAL_BLITS by 1 and ENCODING_COUNT by 1.
    // Increment TOTAL_BLITS by (count - 1) for the remaining copies.
    if count > 1 {
        TOTAL_BLITS.with(|c| c.set(c.get() + count - 1));
    }
    Ok(())
}

/// Encode dispatch steps into the lazy batch's [`CommandBatch`].
///
/// `f` receives a `&CommandBatch` to encode into. No `commit_and_wait` occurs.
///
/// # Re-entrancy: `f` runs while `LAZY_BATCH` is `RefCell::borrow()`-ed.
/// `f` MUST NOT call [`flush`], [`submit`], [`sync`], [`get_or_create_batch`],
/// or this function — re-entrant borrow panics with "already borrowed".
///
/// Returns `Ok(Ok(()))` on success, `Ok(Err(e))` if encoding fails, or
/// `Err(TensorError)` if no lazy batch exists (caller should call
/// [`get_or_create_batch`] first).
pub(crate) fn encode_into_lazy_batch<F, E>(f: F) -> Result<Result<(), E>, TensorError>
where
    F: FnOnce(&CommandBatch) -> Result<(), E>,
{
    LAZY_BATCH.with(|cell| {
        let guard = cell.borrow();
        match guard.as_ref() {
            Some(batch) => Ok(f(batch)),
            None => Err(TensorError::backend_failure(
                nn_core::BackendDomain::Metal,
                nn_core::BackendErrorKind::DispatchFailed,
                "encode_into_lazy_batch called without active batch".to_string(),
            )),
        }
    })
}

/// Encode a custom GPU dispatch into the thread-local lazy batch.
///
/// This is the public API for integrating non-nn Metal compute kernels
/// (e.g., dvoice custom MSL kernels) into the lazy batch system. The
/// callback `f` receives a [`CommandBatch`] reference to encode into — the
/// same command buffer used by nn-native ops.
///
/// ```rust,ignore
/// use nn_metal::encode_custom_dispatch;
///
/// encode_custom_dispatch(|batch| {
///     batch.encode_compute(pipeline, &buffers, grid_size, group_size)
/// })?;
/// ```
///
/// # Errors
///
/// Returns `Err(TensorError)` if the lazy batch cannot be created (Metal
/// context unavailable). Returns `Ok(Err(E))` if the callback itself fails.
/// Returns `Ok(Ok(()))` on success.
pub fn encode_custom_dispatch<F, E>(f: F) -> Result<Result<(), E>, TensorError>
where
    F: FnOnce(&CommandBatch) -> Result<(), E>,
{
    get_or_create_batch()?;
    encode_into_lazy_batch(f)
}

/// Submit pending GPU work without waiting. CPU continues immediately.
///
/// Commits the current lazy batch to the GPU via [`CommandBatch::commit_no_wait`]
/// and stores a [`PendingBatch`] handle for later synchronization. The GPU starts
/// executing immediately while the CPU can continue encoding a new batch.
///
/// At most one pending batch is tracked (simple sequential pipelining). If a
/// prior pending batch exists, [`sync`] is called first to wait for it.
///
/// Call [`sync`] or [`flush`] when CPU readback of the submitted results is needed.
///
/// No-op when no lazy batch is pending.
///
/// # Performance
///
/// This is the Metal equivalent of PyTorch MPS `COMMIT_AND_CONTINUE` (#2375).
/// Use between compiled model segments where the next segment's encoding can
/// overlap with the current segment's GPU execution.
pub fn submit() -> Result<(), TensorError> {
    // Wait for any prior pending batch first (simple sequential pipelining).
    sync()?;

    LAZY_BATCH.with(|cell| {
        if let Some(batch) = cell.borrow_mut().take() {
            let enc = ENCODING_COUNT.with(Cell::get);
            ENCODING_COUNT.with(|c| c.set(0));
            TOTAL_SUBMITS.with(|c| record_gpu_event(c, "submit", enc));
            let pending = batch.commit_no_wait();
            PENDING.with(|p| *p.borrow_mut() = Some(pending));

            // Note: arena reset is deferred to sync/flush when GPU work is known
            // to be complete. ObjC ARC keeps Metal buffers alive for any DynTensor
            // still holding a reference, so this is safe.
        }
        Ok(())
    })
}

/// Wait for the most recently submitted batch to complete.
///
/// No-op if no pending batch. Called automatically by [`flush`] and before
/// a new [`submit`].
pub fn sync() -> Result<(), TensorError> {
    PENDING.with(|cell| {
        if let Some(pending) = cell.borrow_mut().take() {
            pending.wait().map_err(|e| {
                TensorError::backend_failure(
                    nn_core::BackendDomain::Metal,
                    nn_core::BackendErrorKind::DispatchFailed,
                    format!("sync: pending batch failed: {e}"),
                )
            })?;

            // Now that the submitted GPU work is complete, reset the arena.
            crate::arena::reset_default_arena();
        }
        Ok(())
    })
}

/// Take the current lazy batch, commit it non-blocking, and return a
/// [`GpuFence`] handle. Unlike [`submit`], this does NOT wait for any prior
/// `PENDING` batch and does NOT store the result in `PENDING`.
///
/// Returns `Ok(None)` if no lazy batch is pending.
///
/// This is the internal helper for [`GpuFence::submit_current`].
pub(crate) fn take_lazy_batch_for_fence() -> Result<Option<crate::gpu_fence::GpuFence>, TensorError> {
    LAZY_BATCH.with(|cell| {
        if let Some(batch) = cell.borrow_mut().take() {
            let enc = ENCODING_COUNT.with(Cell::get);
            ENCODING_COUNT.with(|c| c.set(0));
            TOTAL_SUBMITS.with(|c| record_gpu_event(c, "fence-submit", enc));
            let pending = batch.commit_no_wait();
            Ok(Some(crate::gpu_fence::GpuFence::from_pending(pending)))
        } else {
            Ok(None)
        }
    })
}

/// Take the current lazy batch, commit it non-blocking, and return a
/// [`GpuFuture`] handle. Like [`take_lazy_batch_for_fence`] but returns
/// the richer `GpuFuture` type that supports callback-based notification.
///
/// Uses `CommandBatch::submit_async()` which registers a Metal
/// `addCompletedHandler` BEFORE `commit()` (Metal requires handler
/// registration before commit).
///
/// Returns `Ok(None)` if no lazy batch is pending.
///
/// This is the internal helper for [`GpuFuture::submit_current`].
pub(crate) fn take_lazy_batch_for_future() -> Result<Option<crate::gpu_future::GpuFuture>, TensorError> {
    LAZY_BATCH.with(|cell| {
        if let Some(batch) = cell.borrow_mut().take() {
            let enc = ENCODING_COUNT.with(Cell::get);
            ENCODING_COUNT.with(|c| c.set(0));
            TOTAL_SUBMITS.with(|c| record_gpu_event(c, "future-submit", enc));
            let future = batch.submit_async();
            Ok(Some(future))
        } else {
            Ok(None)
        }
    })
}

struct ScopeExitModeGuard {
    prev: ScopeExitMode,
}

impl Drop for ScopeExitModeGuard {
    fn drop(&mut self) {
        SCOPE_EXIT_MODE.with(|c| c.set(self.prev));
    }
}

/// Run `f` with the given [`ScopeExitMode`]. Restores prior mode on return
/// (RAII guard). Optionally combine with `NanCheckPolicy::Skip` for throughput.
pub fn with_scope_exit_mode<T>(mode: ScopeExitMode, f: impl FnOnce() -> T) -> T {
    let _guard = ScopeExitModeGuard {
        prev: SCOPE_EXIT_MODE.with(Cell::get),
    };
    SCOPE_EXIT_MODE.with(|c| c.set(mode));
    f()
}

/// Commit the pending lazy batch (if any) and wait for GPU completion.
///
/// Called automatically before any CPU readback of GPU buffer data. After
/// `flush()`, all previously encoded GPU operations are complete and their
/// output buffers contain valid data.
///
/// Also waits for any prior [`submit`] batch before committing the current
/// batch, ensuring all GPU work is complete on return.
///
/// No-op when no lazy batch is pending.
pub fn flush() -> Result<(), TensorError> {
    // Wait for any prior submit() batch first.
    sync()?;

    LAZY_BATCH.with(|cell| {
        if let Some(batch) = cell.borrow_mut().take() {
            let enc = ENCODING_COUNT.with(Cell::get);
            ENCODING_COUNT.with(|c| c.set(0));
            TOTAL_FLUSHES.with(|c| record_gpu_event(c, "flush", enc));
            batch.commit_and_wait().map_err(|e| {
                TensorError::backend_failure(
                    nn_core::BackendDomain::Metal,
                    nn_core::BackendErrorKind::DispatchFailed,
                    format!("flush: commit failed: {e}"),
                )
            })?;

            // Reset the default arena after GPU work completes. All intermediate
            // arena buffers are no longer referenced by pending encoders. ObjC
            // ARC keeps the Metal buffer alive for any DynTensor still holding
            // a reference.
            crate::arena::reset_default_arena();
        }
        Ok(())
    })
}

/// Returns `true` if a lazy batch is pending on the current thread.
#[cfg(test)]
pub(crate) fn is_lazy_batch_active() -> bool {
    LAZY_BATCH.with(|cell| cell.borrow().is_some())
}

/// Returns the number of pending encodings in the current lazy batch.
///
/// Used by [`MetalContext::clone_buffer`] and [`MetalContext::clone_buffer_range`]
/// to return `Err(MetalError::PendingFlushRequired)` when callers attempt to read
/// GPU buffer contents without flushing first. Two prior P1 bugs (#1912, #1933)
/// were caused by this omission.
pub(crate) fn pending_encoding_count() -> usize {
    ENCODING_COUNT.with(Cell::get)
}

/// Discard the pending lazy batch without committing.
///
/// Used for error recovery when `_no_fence` execution encounters a failure
/// mid-pipeline. Without this, stale GPU commands persist in the thread-local
/// batch and execute alongside the next call's commands.
///
/// No-op when no lazy batch is pending. Metal ObjC ARC handles buffer cleanup.
///
/// # Safety invariant: Metal queue serial execution
///
/// Dropping the `PendingBatch` in `PENDING` releases our ARC reference to
/// the already-committed command buffer. Metal's internal queue retains it
/// until GPU execution completes. Subsequent lazy batches use the same
/// Metal command queue, which guarantees serial execution — the old batch's
/// GPU writes complete before any new batch's GPU writes begin. This makes
/// arena buffer reuse safe: `reset_default_arena()` (called by future
/// `flush()`) reclaims memory only after the old batch has finished.
///
/// If `MetalContext` ever switches to concurrent dispatch queues, this
/// function would need an explicit `wait()` before dropping `PENDING`.
pub(crate) fn discard_pending_batch() {
    LAZY_BATCH.with(|cell| {
        cell.borrow_mut().take();
    });
    ENCODING_COUNT.with(|c| c.set(0));
    PENDING.with(|cell| {
        cell.borrow_mut().take();
    });
}

/// Execute a closure with an explicit flush fence at the end.
///
/// All Metal kernel dispatches within `f` are batched (as always with lazy
/// execution). On success, [`flush`] is called at scope exit to commit all
/// pending GPU work. This matches the original `with_gpu_scope` semantics:
/// callers are guaranteed that all GPU work from within the scope is complete
/// when the function returns.
///
/// # Nesting
///
/// Nested `with_gpu_scope` calls are transparent — the inner scope runs `f`
/// and calls `flush()` at exit. The lazy batch is shared across all nesting
/// levels.
///
/// # Errors
///
/// Returns `Err` if `f` returns an error or if the flush fails. On error
/// from `f`, the pending batch is dropped without committing (Metal discards
/// uncommitted command buffers automatically).
pub fn with_gpu_scope<F, T>(f: F) -> nn_core::Result<T>
where
    F: FnOnce() -> nn_core::Result<T>,
{
    let result = f();

    if result.is_ok() {
        match SCOPE_EXIT_MODE.with(Cell::get) {
            ScopeExitMode::Flush => flush()?,
            ScopeExitMode::Submit => submit()?,
        }
    } else {
        // Drop uncommitted batch + any pending submit. Metal ObjC ARC
        // handles buffer cleanup; we clear state to avoid leaking into
        // the next scope on this thread.
        discard_pending_batch();
    }

    result
}

#[cfg(test)]
#[path = "gpu_scope_tests.rs"]
mod tests;
