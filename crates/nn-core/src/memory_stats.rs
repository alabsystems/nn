// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Process memory measurement — backend-agnostic RSS helpers.
//!
//! Provides [`get_peak_rss()`] and [`get_current_rss()`] using the Mach kernel
//! API on macOS (`mach_task_basic_info`). These are backend-agnostic and live
//! in nn-core so any crate can measure memory without depending on nn-metal.
//!
//! For Metal GPU allocation tracking, see `nn_metal::rss`.
//!
//! # Platform Support
//!
//! macOS: uses `mach_task_basic_info` (flavor 20).
//! Other platforms: returns `Err(TensorError::BackendFailure)`.
//!
//! Part of #3211.

use crate::error::{BackendDomain, BackendErrorKind, TensorError};

/// A point-in-time snapshot of process memory usage.
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Peak RSS in bytes (lifetime high-water mark).
    pub peak_rss: usize,
    /// Current RSS in bytes.
    pub current_rss: usize,
    /// Timestamp (seconds since UNIX epoch) when this snapshot was taken.
    pub timestamp_secs: f64,
}

impl MemorySnapshot {
    /// Take a snapshot of the current process memory state.
    ///
    /// Returns an error on non-macOS platforms or if the Mach syscall fails.
    pub fn capture() -> crate::Result<Self> {
        let current_rss = get_current_rss()?;
        let peak_rss = get_peak_rss()?;
        let timestamp_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        Ok(Self {
            peak_rss,
            current_rss,
            timestamp_secs,
        })
    }

    /// Peak RSS in megabytes.
    #[must_use]
    pub fn peak_rss_mb(&self) -> f64 {
        self.peak_rss as f64 / (1024.0 * 1024.0)
    }

    /// Current RSS in megabytes.
    #[must_use]
    pub fn current_rss_mb(&self) -> f64 {
        self.current_rss as f64 / (1024.0 * 1024.0)
    }
}

/// Query the peak (high-water-mark) RSS in bytes for the current process.
///
/// On macOS, this reads `resident_size_max` from `mach_task_basic_info`.
///
/// # Errors
///
/// Returns `TensorError::BackendFailure` on non-macOS platforms or if
/// the Mach kernel syscall fails.
pub fn get_peak_rss() -> crate::Result<usize> {
    get_mach_rss_fields().map(|f| f.resident_size_max)
}

/// Query the current RSS in bytes for the current process.
///
/// On macOS, this reads `resident_size` from `mach_task_basic_info`.
///
/// # Errors
///
/// Returns `TensorError::BackendFailure` on non-macOS platforms or if
/// the Mach kernel syscall fails.
pub fn get_current_rss() -> crate::Result<usize> {
    get_mach_rss_fields().map(|f| f.resident_size)
}

// -- Internal implementation -----------------------------------------------

struct MachRssFields {
    resident_size: usize,
    resident_size_max: usize,
}

#[cfg(target_os = "macos")]
fn get_mach_rss_fields() -> crate::Result<MachRssFields> {
    use std::mem;

    // mach_task_basic_info layout (from <mach/task_info.h>):
    //   mach_vm_size_t virtual_size        (8 bytes)
    //   mach_vm_size_t resident_size       (8 bytes)
    //   mach_vm_size_t resident_size_max   (8 bytes)
    //   time_value_t   user_time           (8 bytes)
    //   time_value_t   system_time         (8 bytes)
    //   policy_t       policy              (4 bytes)
    //   integer_t      suspend_count       (4 bytes)
    //   Total: 48 bytes, count = 12
    #[repr(C)]
    struct MachTaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time_seconds: i32,
        user_time_microseconds: i32,
        system_time_seconds: i32,
        system_time_microseconds: i32,
        policy: i32,
        suspend_count: i32,
    }

    const _: () = assert!(size_of::<MachTaskBasicInfo>() == 48);

    const MACH_TASK_BASIC_INFO: u32 = 20;
    const MACH_TASK_BASIC_INFO_COUNT: u32 =
        (size_of::<MachTaskBasicInfo>() / size_of::<i32>()) as u32;

    extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(
            target_task: u32,
            flavor: u32,
            task_info_out: *mut MachTaskBasicInfo,
            task_info_out_cnt: *mut u32,
        ) -> i32;
    }

    let mut info: MachTaskBasicInfo = unsafe { mem::zeroed() };
    let mut count = MACH_TASK_BASIC_INFO_COUNT;

    // SAFETY: Correctly-sized buffer; mach_task_self() returns the current
    // task port. task_info fills the struct on KERN_SUCCESS (return 0).
    let kr = unsafe {
        task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO,
            &raw mut info,
            &raw mut count,
        )
    };

    if kr == 0 {
        Ok(MachRssFields {
            resident_size: info.resident_size as usize,
            resident_size_max: info.resident_size_max as usize,
        })
    } else {
        Err(TensorError::backend_failure(
            BackendDomain::Device,
            BackendErrorKind::Other,
            format!("mach_task_basic_info failed with kern_return_t={kr}"),
        ))
    }
}

#[cfg(not(target_os = "macos"))]
fn get_mach_rss_fields() -> crate::Result<MachRssFields> {
    Err(TensorError::backend_failure(
        BackendDomain::Device,
        BackendErrorKind::Other,
        "memory_stats requires macOS (mach_task_basic_info)".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_current_rss() {
        if cfg!(target_os = "macos") {
            let rss = get_current_rss().expect("should succeed on macOS");
            // Any live process should have > 1 MB RSS.
            assert!(rss > 1_000_000, "RSS too small: {rss}");
            // Sanity: < 100 GB.
            assert!(rss < 100_000_000_000, "RSS too large: {rss}");
        }
    }

    #[test]
    fn test_get_peak_rss() {
        if cfg!(target_os = "macos") {
            let peak = get_peak_rss().expect("should succeed on macOS");
            let current = get_current_rss().expect("should succeed on macOS");
            // Peak must be >= current.
            assert!(peak >= current, "peak {peak} < current {current}");
        }
    }

    #[test]
    fn test_memory_snapshot_capture() {
        if cfg!(target_os = "macos") {
            let snap = MemorySnapshot::capture().expect("should succeed on macOS");
            assert!(snap.peak_rss >= snap.current_rss);
            assert!(snap.peak_rss_mb() > 1.0);
            assert!(snap.current_rss_mb() > 1.0);
            assert!(snap.timestamp_secs > 0.0);
        }
    }
}
