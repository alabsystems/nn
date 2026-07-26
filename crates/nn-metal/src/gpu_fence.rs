// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Caller-held fence for non-blocking GPU submit.
//!
//! Unlike the thread-local [`submit()`](crate::gpu_scope::submit) /
//! [`sync()`](crate::gpu_scope::sync) pattern which tracks at most one pending
//! batch, [`GpuFence`] lets callers hold multiple outstanding GPU submissions
//! and wait on them individually. This enables segment-level pipelining:
//! encode segment N+1 on CPU while segment N executes on GPU.
//!
//! # Example
//!
//! ```rust,no_run
//! use nn_metal::GpuFence;
//!
//! // Encode segment 1
//! // encode_segment_1();
//! let fence1 = GpuFence::submit_current().unwrap();
//!
//! // Encode segment 2 while segment 1 executes on GPU
//! // encode_segment_2();
//! let fence2 = GpuFence::submit_current().unwrap();
//!
//! // Wait for segment 1 results
//! if let Some(f) = fence1 {
//!     f.wait().unwrap();
//! }
//! // read_segment_1_output();
//!
//! // Wait for segment 2
//! if let Some(f) = fence2 {
//!     f.wait().unwrap();
//! }
//! ```

use std::time::{Duration, Instant};

use crate::dispatch::PendingBatch;
use crate::gpu_scope;
use nn_core::TensorError;

/// Caller-held handle to submitted GPU work.
///
/// Unlike the thread-local `submit()`/`sync()` pattern which tracks at most
/// one pending batch, `GpuFence` lets callers hold multiple outstanding
/// GPU submissions and wait on them individually. This enables segment-level
/// pipelining: encode segment N+1 on CPU while segment N executes on GPU.
///
/// # Profiling
///
/// Each fence records the time of submission ([`elapsed`](Self::elapsed)),
/// enabling lightweight GPU latency profiling without Metal GPU counters.
///
/// # Safety
///
/// `GpuFence` does NOT reset the activation arena on `wait()`. Callers using
/// `GpuFence` for multi-segment pipelining must manage arena lifetimes
/// explicitly (e.g., via `with_arena` scoping each segment). The thread-local
/// `sync()`/`flush()` path resets the arena automatically because it knows
/// only one batch is outstanding.
pub struct GpuFence {
    pending: PendingBatch,
    /// Timestamp captured at fence creation (submit time).
    submit_time: Instant,
}

impl GpuFence {
    /// Submit the current lazy batch and return a fence.
    ///
    /// Takes the current `LAZY_BATCH` from thread-local storage, commits it
    /// via `commit_no_wait()`, and returns a `GpuFence` handle. Unlike
    /// `gpu_scope::submit()`, this does NOT wait for any prior pending batch
    /// and does NOT store the handle in the thread-local `PENDING`.
    ///
    /// Returns `Ok(None)` if no lazy batch is pending.
    #[must_use = "returns a Result that may contain an error"]
    pub fn submit_current() -> Result<Option<Self>, TensorError> {
        gpu_scope::take_lazy_batch_for_fence()
    }

    /// Block until the submitted GPU work completes.
    ///
    /// Does NOT reset the activation arena — see struct-level docs.
    #[must_use = "returns a Result that may contain an error"]
    pub fn wait(self) -> Result<(), TensorError> {
        self.pending.wait().map_err(|e| {
            TensorError::backend_failure(
                nn_core::BackendDomain::Metal,
                nn_core::BackendErrorKind::DispatchFailed,
                format!("GpuFence::wait: pending batch failed: {e}"),
            )
        })
    }

    /// Block until the submitted GPU work completes or the timeout expires.
    ///
    /// Returns `Ok(true)` if the GPU work completed within the timeout,
    /// `Ok(false)` if the timeout expired before completion. Returns `Err`
    /// if the command buffer entered an error state.
    ///
    /// Unlike [`wait`](Self::wait), this does NOT consume `self` — the caller
    /// retains ownership and can poll again, call `wait()`, or call
    /// `wait_timeout()` with a longer timeout.
    #[must_use = "returns a Result that may contain an error"]
    pub fn wait_timeout(&self, timeout: Duration) -> Result<bool, TensorError> {
        self.pending.wait_timeout(timeout).map_err(|e| {
            TensorError::backend_failure(
                nn_core::BackendDomain::Metal,
                nn_core::BackendErrorKind::DispatchFailed,
                format!("GpuFence::wait_timeout: pending batch failed: {e}"),
            )
        })
    }

    /// Check if GPU work completed without blocking.
    pub fn is_completed(&self) -> bool {
        self.pending.is_completed()
    }

    /// Time elapsed since this fence was submitted.
    ///
    /// Useful for lightweight GPU latency profiling. The returned duration
    /// measures wall-clock time from submit to now (or to `wait()` if called
    /// after waiting).
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.submit_time.elapsed()
    }

    /// The instant at which this fence was submitted.
    #[must_use]
    pub fn submit_time(&self) -> Instant {
        self.submit_time
    }

    /// Construct a `GpuFence` from a `PendingBatch`.
    ///
    /// Called by `gpu_scope::take_lazy_batch_for_fence` after committing the
    /// lazy batch. Not public — construction is only via `submit_current()`.
    pub(crate) fn from_pending(pending: PendingBatch) -> Self {
        Self {
            pending,
            submit_time: Instant::now(),
        }
    }
}

impl std::fmt::Debug for GpuFence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuFence")
            .field("is_completed", &self.is_completed())
            .field("elapsed_ms", &self.elapsed().as_millis())
            .finish()
    }
}

#[cfg(test)]
#[path = "gpu_fence_tests.rs"]
mod tests;
