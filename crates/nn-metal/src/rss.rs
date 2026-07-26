// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Process RSS and Metal GPU allocation measurement.
//!
//! Provides [`rss_bytes()`] to query the current process's physical memory
//! footprint, [`metal_allocated_bytes()`] for Metal GPU buffer allocations,
//! and [`RssTracker`] for checkpoint-based dual memory profiling.
//!
//! On Apple Silicon (unified memory), Metal allocations are reflected in
//! process RSS since CPU and GPU share physical memory. Tracking both RSS
//! and Metal allocation separately helps attribute memory overhead: the
//! difference (RSS - Metal) approximates CPU-side allocations, metadata,
//! and OS overhead.
//!
//! # Platform Support
//!
//! Uses `mach_task_basic_info` on macOS. Metal allocation uses
//! `MTLDevice.currentAllocatedSize`. Returns `None` on non-macOS platforms.
//!
//! # Example
//!
//! ```rust,no_run
//! use nn_metal::rss::{rss_bytes, metal_allocated_bytes, RssTracker};
//!
//! let mut tracker = RssTracker::new();
//! tracker.checkpoint("before_load");
//! // ... load weights ...
//! tracker.checkpoint("after_load");
//! tracker.checkpoint("after_inference");
//! println!("{tracker}");
//! ```
//!
//! Part of #3079.

use std::fmt;

/// Query the current process RSS in bytes.
///
/// Uses `mach_task_basic_info` on macOS via the Mach kernel API.
/// Returns `None` on non-macOS platforms or if the syscall fails.
#[cfg(target_os = "macos")]
pub fn rss_bytes() -> Option<usize> {
    use std::mem;

    // mach_task_basic_info struct layout (from <mach/task_info.h>):
    //   mach_vm_size_t virtual_size        (offset 0,  8 bytes)
    //   mach_vm_size_t resident_size       (offset 8,  8 bytes) <-- this is RSS
    //   mach_vm_size_t resident_size_max   (offset 16, 8 bytes)
    //   time_value_t   user_time           (offset 24, 8 bytes)
    //   time_value_t   system_time         (offset 32, 8 bytes)
    //   policy_t       policy              (offset 40, 4 bytes)
    //   integer_t      suspend_count       (offset 44, 4 bytes)
    //   Total: 48 bytes, MACH_TASK_BASIC_INFO_COUNT = 12
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

    // Static assert: struct must be exactly 48 bytes (matching C sizeof).
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

    // SAFETY: MachTaskBasicInfo is a repr(C) struct of integer fields with no
    // padding invariants. All-zeros is a valid bit pattern for every field.
    let mut info: MachTaskBasicInfo = unsafe { mem::zeroed() };
    let mut count = MACH_TASK_BASIC_INFO_COUNT;

    // SAFETY: We pass a correctly-sized buffer and count. mach_task_self()
    // returns the current task port. task_info fills the struct on success
    // (return value 0 = KERN_SUCCESS).
    let kr = unsafe {
        task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO,
            &raw mut info,
            &raw mut count,
        )
    };

    if kr == 0 {
        Some(info.resident_size as usize)
    } else {
        None
    }
}

/// Stub for non-macOS platforms.
#[cfg(not(target_os = "macos"))]
pub fn rss_bytes() -> Option<usize> {
    None
}

/// Query RSS in megabytes (convenience wrapper).
#[must_use]
pub fn rss_mb() -> Option<f64> {
    rss_bytes().map(|b| b as f64 / (1024.0 * 1024.0))
}

/// Query the total bytes currently allocated by the Metal device.
///
/// Uses `MTLDevice.currentAllocatedSize` which tracks all Metal buffer
/// allocations made by this process. On Apple Silicon unified memory,
/// these allocations are also reflected in process RSS.
///
/// Returns `None` on non-macOS platforms or if no Metal device is available.
#[cfg(target_os = "macos")]
#[must_use]
pub fn metal_allocated_bytes() -> Option<usize> {
    let device = metal::Device::system_default()?;
    Some(device.current_allocated_size() as usize)
}

/// Stub for non-macOS platforms.
#[cfg(not(target_os = "macos"))]
#[must_use]
pub fn metal_allocated_bytes() -> Option<usize> {
    None
}

/// Query Metal allocation in megabytes (convenience wrapper).
#[must_use]
pub fn metal_allocated_mb() -> Option<f64> {
    metal_allocated_bytes().map(|b| b as f64 / (1024.0 * 1024.0))
}

/// Query the recommended maximum working set size for the Metal device.
///
/// This is the GPU memory budget suggested by the system. Exceeding it
/// may cause performance degradation due to paging.
///
/// Returns `None` on non-macOS platforms or if no Metal device is available.
#[cfg(target_os = "macos")]
#[must_use]
pub fn metal_budget_bytes() -> Option<u64> {
    let device = metal::Device::system_default()?;
    Some(device.recommended_max_working_set_size())
}

/// Stub for non-macOS platforms.
#[cfg(not(target_os = "macos"))]
#[must_use]
pub fn metal_budget_bytes() -> Option<u64> {
    None
}

/// A single RSS measurement at a named checkpoint.
#[derive(Debug, Clone)]
pub struct RssSnapshot {
    /// Checkpoint label (e.g., "after_weight_load").
    pub label: String,
    /// RSS in bytes at this checkpoint.
    pub rss_bytes: usize,
    /// Metal GPU allocation in bytes at this checkpoint.
    /// `None` if Metal device was unavailable when the checkpoint was taken.
    pub metal_bytes: Option<usize>,
}

impl RssSnapshot {
    /// RSS in megabytes.
    #[must_use]
    pub fn rss_mb(&self) -> f64 {
        self.rss_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Metal allocation in megabytes, if available.
    #[must_use]
    pub fn metal_mb(&self) -> Option<f64> {
        self.metal_bytes.map(|b| b as f64 / (1024.0 * 1024.0))
    }
}

/// Tracks RSS at named checkpoints for memory profiling.
///
/// Records the process RSS at each `checkpoint()` call and computes
/// deltas between consecutive checkpoints for memory attribution.
#[derive(Debug, Clone, Default)]
pub struct RssTracker {
    snapshots: Vec<RssSnapshot>,
}

impl RssTracker {
    /// Create a new empty tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the current RSS and Metal allocation with a label.
    ///
    /// If RSS measurement fails (non-macOS), the checkpoint is skipped.
    /// Metal allocation is captured opportunistically — `metal_bytes` is
    /// `None` if no Metal device is available at checkpoint time.
    pub fn checkpoint(&mut self, label: &str) {
        if let Some(bytes) = rss_bytes() {
            self.snapshots.push(RssSnapshot {
                label: label.to_string(),
                rss_bytes: bytes,
                metal_bytes: metal_allocated_bytes(),
            });
        }
    }

    /// All recorded snapshots.
    #[must_use]
    pub fn snapshots(&self) -> &[RssSnapshot] {
        &self.snapshots
    }

    /// Delta between two named checkpoints in bytes.
    ///
    /// Returns `None` if either label is not found.
    #[must_use]
    pub fn delta_bytes(&self, from: &str, to: &str) -> Option<isize> {
        let from_rss = self.snapshots.iter().find(|s| s.label == from)?.rss_bytes;
        let to_rss = self.snapshots.iter().find(|s| s.label == to)?.rss_bytes;
        Some(to_rss as isize - from_rss as isize)
    }

    /// Total RSS growth from first to last checkpoint.
    #[must_use]
    pub fn total_growth_bytes(&self) -> Option<isize> {
        if self.snapshots.len() < 2 {
            return None;
        }
        let first = self.snapshots.first()?.rss_bytes;
        let last = self.snapshots.last()?.rss_bytes;
        Some(last as isize - first as isize)
    }

    /// Peak RSS across all checkpoints.
    #[must_use]
    pub fn peak_rss_bytes(&self) -> Option<usize> {
        self.snapshots.iter().map(|s| s.rss_bytes).max()
    }

    /// Peak Metal GPU allocation across all checkpoints.
    #[must_use]
    pub fn peak_metal_bytes(&self) -> Option<usize> {
        self.snapshots.iter().filter_map(|s| s.metal_bytes).max()
    }
}

impl fmt::Display for RssTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let has_metal = self.snapshots.iter().any(|s| s.metal_bytes.is_some());
        if has_metal {
            writeln!(
                f,
                "RSS + Metal Memory Profile ({} checkpoints)",
                self.snapshots.len()
            )?;
        } else {
            writeln!(
                f,
                "RSS Memory Profile ({} checkpoints)",
                self.snapshots.len()
            )?;
        }
        writeln!(f, "{:-<72}", "")?;
        let mut prev_rss: Option<usize> = None;
        let mut prev_metal: Option<usize> = None;
        for snap in &self.snapshots {
            let rss_delta = match prev_rss {
                Some(p) => {
                    let delta = snap.rss_bytes as isize - p as isize;
                    let sign = if delta >= 0 { "+" } else { "" };
                    format!("  ({sign}{:.1})", delta as f64 / (1024.0 * 1024.0))
                }
                None => String::new(),
            };
            if has_metal {
                let metal_str = match snap.metal_bytes {
                    Some(mb) => {
                        let metal_delta = match prev_metal {
                            Some(p) => {
                                let delta = mb as isize - p as isize;
                                let sign = if delta >= 0 { "+" } else { "" };
                                format!("  ({sign}{:.1})", delta as f64 / (1024.0 * 1024.0))
                            }
                            None => String::new(),
                        };
                        prev_metal = Some(mb);
                        format!(
                            "  Metal: {:>7.1} MB{}",
                            mb as f64 / (1024.0 * 1024.0),
                            metal_delta,
                        )
                    }
                    None => String::new(),
                };
                writeln!(
                    f,
                    "  {:<26} RSS: {:>7.1} MB{}{}",
                    snap.label,
                    snap.rss_mb(),
                    rss_delta,
                    metal_str,
                )?;
            } else {
                writeln!(
                    f,
                    "  {:<30} {:>8.1} MB{}",
                    snap.label,
                    snap.rss_mb(),
                    rss_delta,
                )?;
            }
            prev_rss = Some(snap.rss_bytes);
        }
        if let Some(growth) = self.total_growth_bytes() {
            writeln!(f, "{:-<72}", "")?;
            writeln!(
                f,
                "  total RSS growth: {:>+.1} MB",
                growth as f64 / (1024.0 * 1024.0)
            )?;
        }
        if let Some(peak) = self.peak_rss_bytes() {
            write!(
                f,
                "  peak RSS:         {:>.1} MB",
                peak as f64 / (1024.0 * 1024.0)
            )?;
        }
        if has_metal {
            if let Some(peak_metal) = self.peak_metal_bytes() {
                write!(
                    f,
                    "\n  peak Metal:       {:>.1} MB",
                    peak_metal as f64 / (1024.0 * 1024.0)
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "rss_tests.rs"]
mod tests;
