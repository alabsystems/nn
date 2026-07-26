// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Non-blocking GPU submit with callback notification.
//!
//! [`GpuFuture`] wraps a committed Metal command buffer and provides three
//! completion APIs:
//!
//! - [`wait()`](GpuFuture::wait) — block until GPU work finishes (synchronous)
//! - [`is_complete()`](GpuFuture::is_complete) — poll without blocking
//! - [`on_complete()`](GpuFuture::on_complete) — register a callback that
//!   fires when the GPU finishes (asynchronous notification)
//!
//! Unlike [`GpuFence`](crate::GpuFence) which is a thin wrapper for segment
//! pipelining, `GpuFuture` supports callback-driven async notification and
//! is designed for streaming and concurrent GPU work patterns.
//!
//! # Architecture
//!
//! Metal's `addCompletedHandler` must be registered BEFORE `commit()`. So
//! `GpuFuture` registers an internal handler at construction time (before
//! commit) that sets a shared `completed` flag and optionally invokes a
//! user callback. The user can register their callback via `on_complete()`
//! at any time — if the GPU work is already done, the callback fires
//! immediately on the calling thread.
//!
//! # Example
//!
//! ```rust,no_run
//! use nn_metal::GpuFuture;
//!
//! // Encode GPU work into the lazy batch, then submit async.
//! let future = GpuFuture::submit_current().unwrap();
//! if let Some(fut) = future {
//!     // CPU is free to do other work while GPU executes.
//!     // Option A: poll
//!     while !fut.is_complete() {
//!         // do CPU work...
//!     }
//!     // Option B: blocking wait
//!     // fut.wait().unwrap();
//! }
//! ```
//!
//! # Callback-based notification
//!
//! ```rust,no_run
//! use nn_metal::GpuFuture;
//! use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
//!
//! let future = GpuFuture::submit_current().unwrap().unwrap();
//! let done = Arc::new(AtomicBool::new(false));
//! let done_clone = done.clone();
//! future.on_complete(move |success| {
//!     done_clone.store(true, Ordering::Release);
//!     if success {
//!         println!("GPU work finished successfully");
//!     }
//! });
//! // CPU continues immediately — callback fires when GPU finishes.
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use block::ConcreteBlock;
use objc::rc::autoreleasepool;

use crate::dispatch::PendingBatch;
use crate::error::MetalError;
use crate::gpu_scope;
use nn_core::TensorError;

/// Shared completion state between the Metal handler and the `GpuFuture`.
///
/// Registered as an `addCompletedHandler` BEFORE the command buffer is
/// committed. The handler sets `completed` and `success`, then invokes
/// the user callback if one is registered.
pub(crate) struct CompletionState {
    /// Whether the GPU work has completed (success or error).
    completed: AtomicBool,
    /// Whether the GPU work completed successfully (`Completed` status).
    success: AtomicBool,
    /// Optional user callback, set via `on_complete()`. If already set when
    /// the Metal handler fires, it is invoked on the Metal callback thread.
    /// If set after the handler fires (GPU already done), it is invoked
    /// immediately on the calling thread.
    user_callback: Mutex<Option<Box<dyn FnOnce(bool) + Send>>>,
}

/// Handle to submitted GPU work with callback-based completion notification.
///
/// Wraps a committed Metal command buffer. The GPU is already executing when
/// this handle is returned. Provides three ways to detect completion:
///
/// 1. **Blocking wait** — [`wait()`](Self::wait)
/// 2. **Non-blocking poll** — [`is_complete()`](Self::is_complete)
/// 3. **Callback** — [`on_complete()`](Self::on_complete) fires when GPU
///    work finishes (on Metal's internal thread or immediately if already done)
///
/// # Arena safety
///
/// Like [`GpuFence`](crate::GpuFence), `GpuFuture` does NOT reset the
/// activation arena on `wait()`. Callers using `GpuFuture` for multi-segment
/// pipelining must manage arena lifetimes explicitly.
pub struct GpuFuture {
    pending: PendingBatch,
    /// Shared state with the Metal completed handler. The handler is
    /// registered before commit, so it fires when the GPU finishes.
    state: Arc<CompletionState>,
}

impl GpuFuture {
    /// Submit the current lazy batch and return a `GpuFuture`.
    ///
    /// Takes the current `LAZY_BATCH` from thread-local storage, registers a
    /// Metal `addCompletedHandler`, commits non-blocking, and returns a
    /// `GpuFuture` handle. The GPU starts executing immediately.
    ///
    /// Returns `Ok(None)` if no lazy batch is pending.
    ///
    /// Unlike `gpu_scope::submit()`, this does NOT wait for any prior pending
    /// batch and does NOT store the handle in thread-local `PENDING`. The
    /// caller owns the `GpuFuture` and is responsible for waiting on it.
    #[must_use = "returns a Result that may contain an error"]
    pub fn submit_current() -> Result<Option<Self>, TensorError> {
        gpu_scope::take_lazy_batch_for_future()
    }

    /// Block until GPU work completes, with timeout protection.
    ///
    /// Uses the same [`wait_with_timeout`](crate::dispatch::wait_with_timeout)
    /// mechanism as `flush()` and `PendingBatch::wait()`.
    ///
    /// Does NOT reset the activation arena — see struct-level docs.
    #[must_use = "returns a Result that may contain an error"]
    pub fn wait(self) -> Result<(), TensorError> {
        self.pending.wait().map_err(|e| {
            TensorError::backend_failure(
                nn_core::BackendDomain::Metal,
                nn_core::BackendErrorKind::DispatchFailed,
                format!("GpuFuture::wait: pending batch failed: {e}"),
            )
        })
    }

    /// Check if GPU work completed without blocking.
    ///
    /// Returns `true` when the command buffer has reached `Completed` status.
    /// Uses the internal completion flag set by the Metal handler, so this
    /// is a single atomic load (no Metal API call).
    pub fn is_complete(&self) -> bool {
        self.state.completed.load(Ordering::Acquire)
    }

    /// Register a callback to fire when GPU work completes.
    ///
    /// The callback receives a `bool`: `true` if the command buffer completed
    /// successfully, `false` if it completed with an error.
    ///
    /// **If the GPU work has already completed**, the callback fires
    /// immediately on the calling thread. Otherwise, it fires on a
    /// Metal-internal thread when the GPU finishes.
    ///
    /// The callback should be lightweight (signal a condvar, set an atomic,
    /// send on a channel). Do NOT call nn-metal APIs from inside the
    /// callback (potential deadlock on thread-local state).
    ///
    /// # Errors
    ///
    /// Returns `Err` if a callback was already registered on this future.
    pub fn on_complete<F>(&self, callback: F) -> Result<(), MetalError>
    where
        F: FnOnce(bool) + Send + 'static,
    {
        let mut guard = self.state.user_callback.lock().map_err(|_| {
            MetalError::InvalidDispatchBindings(
                "GpuFuture::on_complete: completion state mutex poisoned",
            )
        })?;
        if guard.is_some() {
            return Err(MetalError::InvalidDispatchBindings(
                "GpuFuture::on_complete called twice — only one callback per future is allowed",
            ));
        }

        // If GPU work is already done, invoke immediately.
        if self.state.completed.load(Ordering::Acquire) {
            let success = self.state.success.load(Ordering::Acquire);
            drop(guard); // Release lock before calling user code.
            callback(success);
            return Ok(());
        }

        // Not yet done — store the callback for the Metal handler to invoke.
        *guard = Some(Box::new(callback));
        Ok(())
    }

    /// Construct a `GpuFuture` from a `PendingBatch` and pre-registered
    /// completion state.
    ///
    /// Called by `CommandBatch::submit_async` and
    /// `gpu_scope::take_lazy_batch_for_future` which register the handler
    /// before commit. Not public — construction is only via
    /// [`submit_current()`](Self::submit_current).
    pub(crate) fn from_pending_with_state(
        pending: PendingBatch,
        state: Arc<CompletionState>,
    ) -> Self {
        Self { pending, state }
    }
}

impl std::fmt::Debug for GpuFuture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuFuture")
            .field("is_complete", &self.is_complete())
            .finish()
    }
}

/// A `GpuFuture` paired with a completed output that can be retrieved
/// after the GPU work finishes.
///
/// Used by `CompiledModel::execute_dyn_async` to return both the future
/// handle and the (not-yet-valid) output tensor. The caller waits on the
/// future before accessing the tensor data.
pub struct AsyncGpuResult<T> {
    /// The future representing pending GPU work.
    pub future: GpuFuture,
    /// The result value. GPU buffers backing this value contain valid data
    /// only after `future.wait()` or `future.is_complete()` returns `true`.
    pub value: T,
}

impl<T: std::fmt::Debug> std::fmt::Debug for AsyncGpuResult<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncGpuResult")
            .field("future", &self.future)
            .field("value", &self.value)
            .finish()
    }
}

/// Register a Metal `addCompletedHandler` on a command buffer and return the
/// shared completion state. Must be called BEFORE `commit()`.
///
/// This is the shared implementation used by `CommandBatch::submit_async()`
/// and `gpu_scope::take_lazy_batch_for_future()`.
pub(crate) fn register_completion_handler(
    command_buffer: &metal::CommandBufferRef,
) -> Arc<CompletionState> {
    let state = Arc::new(CompletionState {
        completed: AtomicBool::new(false),
        success: AtomicBool::new(false),
        user_callback: Mutex::new(None),
    });

    let handler_state = state.clone();
    let block = ConcreteBlock::new(move |cb: &metal::CommandBufferRef| {
        autoreleasepool(|| {
            let success = cb.status() == metal::MTLCommandBufferStatus::Completed;
            handler_state.success.store(success, Ordering::Release);
            handler_state.completed.store(true, Ordering::Release);

            // Invoke user callback if registered.
            if let Ok(mut guard) = handler_state.user_callback.lock() {
                if let Some(cb_fn) = guard.take() {
                    cb_fn(success);
                }
            }
        });
    });
    let block = block.copy();

    command_buffer.add_completed_handler(&block);
    state
}

#[cfg(test)]
#[path = "gpu_future_tests.rs"]
mod tests;
