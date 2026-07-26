// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Helper functions for verification status persistence.
//!
//! Extracted from `status.rs` to stay within the 500-line limit.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::{SystemTime, UNIX_EPOCH};

use super::{ParamInputRecord, SAVE_NONCE};
use crate::error::VerifyError;

/// Validate that all f32 values in variable inputs and constant params are finite.
pub(super) fn validate_input_metadata(
    variable_inputs: &[ParamInputRecord],
    constant_params: &[f32],
) -> Result<(), VerifyError> {
    let non_finite = |ctx: String| VerifyError::NonFiniteInputMetadata { context: ctx };
    for (i, p) in variable_inputs.iter().enumerate() {
        if !p.lower.is_finite() {
            return Err(non_finite(format!(
                "variable_inputs[{i}].lower = {}",
                p.lower
            )));
        }
        if !p.upper.is_finite() {
            return Err(non_finite(format!(
                "variable_inputs[{i}].upper = {}",
                p.upper
            )));
        }
        if p.lower > p.upper {
            return Err(non_finite(format!(
                "variable_inputs[{i}]: lower ({}) > upper ({})",
                p.lower, p.upper
            )));
        }
    }
    for (i, &val) in constant_params.iter().enumerate() {
        if !val.is_finite() {
            return Err(non_finite(format!("constant_params[{i}] = {val}")));
        }
    }
    Ok(())
}

pub(super) fn atomic_tmp_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .map_or_else(|| "nn_verify_status".into(), |n| n.to_string_lossy());
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let nonce = SAVE_NONCE.fetch_add(1, Ordering::Relaxed);
    path.parent().unwrap_or(Path::new(".")).join(format!(
        ".{name}.{}.{nanos}.{nonce}.tmp",
        std::process::id(),
    ))
}

/// Advisory file lock for `VerifyStatus` save operations.
///
/// Uses `OpenOptions::create_new` on a `.lock` file as a cross-platform
/// advisory lock. The lock file is removed on `Drop`. This prevents
/// concurrent `load-modify-save` sequences from silently dropping results
/// when multiple processes write to the same status file.
///
/// This is advisory: processes that don't use `StatusFileLock` can still
/// write to the file. The lock is best-effort protection against the common
/// case of multiple nn verification processes running in parallel.
pub(super) struct StatusFileLock {
    lock_path: PathBuf,
}

/// Maximum number of retry attempts when acquiring the lock file.
///
/// The lock is held for the *entire* load-modify-save sequence (see
/// [`VerifyStatus::load_locked`]), which for large models spans a multi-second
/// verification. When many verifications run concurrently against the same
/// per-model status file (e.g. the full integration-test suite, where a single
/// model's status is contended by all of that model's compose tests running in
/// parallel), a waiter must out-wait the *serial* completion of every prior
/// holder, not just one. `50 * 100ms = 5s` was far too small for that and made
/// the suite fail spuriously under parallelism. The budget below
/// (`1500 * 100ms = 150s`) comfortably exceeds the serial runtime of the
/// heaviest test binary while staying bounded; the stale-lock cleanup
/// (`LOCK_STALE_THRESHOLD_SECS`) remains the safety net against crashed holders.
const LOCK_MAX_RETRIES: u32 = 1500;

/// Delay between retry attempts in milliseconds.
const LOCK_RETRY_DELAY_MS: u64 = 100;

/// Lock files older than this (in seconds) are considered stale and removed.
const LOCK_STALE_THRESHOLD_SECS: u64 = 300;

impl StatusFileLock {
    /// Acquire an advisory lock for the given status file path.
    ///
    /// Creates a `.lock` file alongside the status file. Retries with
    /// backoff if the lock is held. Stale locks (older than 5 minutes)
    /// are automatically cleaned up to prevent deadlocks from crashed
    /// processes.
    ///
    /// # Errors
    ///
    /// Returns `VerifyError::Io` if the lock cannot be acquired after
    /// retries, or if the lock file directory is not writable.
    pub(super) fn acquire(status_path: &Path) -> Result<Self, VerifyError> {
        let lock_path = lock_path_for(status_path);

        for _ in 0..LOCK_MAX_RETRIES {
            match std::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&lock_path)
            {
                Ok(file) => {
                    drop(file);
                    return Ok(Self { lock_path });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    // Check for stale lock from a crashed process.
                    if is_stale_lock(&lock_path) {
                        let _ = std::fs::remove_file(&lock_path);
                        continue;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(LOCK_RETRY_DELAY_MS));
                }
                Err(e) => return Err(VerifyError::Io(e)),
            }
        }

        Err(VerifyError::Io(std::io::Error::new(
            std::io::ErrorKind::WouldBlock,
            format!(
                "could not acquire status file lock after {} retries: {}",
                LOCK_MAX_RETRIES,
                lock_path.display()
            ),
        )))
    }
}

impl Drop for StatusFileLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.lock_path);
    }
}

/// Compute the lock file path for a given status file.
fn lock_path_for(status_path: &Path) -> PathBuf {
    let name = status_path
        .file_name()
        .map_or_else(|| "nn_verify_status".into(), |n| n.to_string_lossy());
    status_path
        .parent()
        .unwrap_or(Path::new("."))
        .join(format!(".{name}.lock"))
}

/// Check if a lock file is stale (older than `LOCK_STALE_THRESHOLD_SECS`).
fn is_stale_lock(lock_path: &Path) -> bool {
    lock_path
        .metadata()
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.elapsed().ok())
        .map_or(false, |age| age.as_secs() > LOCK_STALE_THRESHOLD_SECS)
}

#[cfg(unix)]
pub(super) fn sync_directory(dir: &Path) -> std::io::Result<()> {
    std::fs::File::open(dir)?.sync_all()
}

#[cfg(not(unix))]
pub(super) fn sync_directory(_dir: &Path) -> std::io::Result<()> {
    Ok(())
}

#[cfg(test)]
#[path = "status_helpers_tests.rs"]
mod tests;
