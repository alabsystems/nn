// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! [`PendingBatch`] — a committed command buffer awaiting completion.

use std::time::Duration;

use objc::rc::autoreleasepool;

use crate::error::MetalError;

/// A committed command buffer that has not been waited on.
///
/// Returned by [`super::CommandBatch::commit_no_wait`]. The GPU is executing the
/// submitted work asynchronously. Call [`wait`](Self::wait) before reading
/// any buffer written by this batch.
#[derive(Debug)]
#[must_use = "dropping PendingBatch without wait() orphans GPU work — arena may overwrite buffers before GPU reads complete"]
pub struct PendingBatch {
    pub(super) command_buffer: metal::CommandBuffer,
}

impl PendingBatch {
    /// Block until GPU work completes, with timeout protection.
    ///
    /// Call before CPU readback of any buffer written by the committed batch.
    /// Uses [`super::wait_with_timeout`] to prevent indefinite blocking from
    /// a hung GPU shader or wedged Metal driver.
    #[must_use = "returns a Result that may contain an error"]
    pub fn wait(self) -> Result<(), MetalError> {
        autoreleasepool(|| super::wait_with_timeout(&self.command_buffer, super::GPU_TIMEOUT))
    }

    /// Block until GPU work completes or the timeout expires.
    ///
    /// Returns `Ok(true)` if the GPU work completed within the timeout,
    /// `Ok(false)` if the timeout expired before completion. Returns `Err`
    /// if the command buffer entered an error state.
    ///
    /// Unlike [`wait`](Self::wait), this does NOT consume `self` — the caller
    /// retains ownership and can poll again or call `wait()` later.
    #[must_use = "returns a Result that may contain an error"]
    pub fn wait_timeout(&self, timeout: Duration) -> Result<bool, MetalError> {
        autoreleasepool(|| {
            let deadline = std::time::Instant::now() + timeout;
            let mut sleep_us = 100u64;
            loop {
                let status = self.command_buffer.status();
                match status {
                    metal::MTLCommandBufferStatus::Completed => return Ok(true),
                    metal::MTLCommandBufferStatus::Error => {
                        return Err(MetalError::DispatchFailed(format!("{status:?}")));
                    }
                    _ => {
                        if std::time::Instant::now() >= deadline {
                            return Ok(false);
                        }
                        std::thread::sleep(Duration::from_micros(sleep_us));
                        sleep_us = (sleep_us * 2).min(10_000); // cap at 10ms
                    }
                }
            }
        })
    }

    /// Check if GPU work completed without blocking.
    pub fn is_completed(&self) -> bool {
        self.command_buffer.status() == metal::MTLCommandBufferStatus::Completed
    }
}
