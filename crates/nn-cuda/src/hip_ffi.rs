// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Raw FFI type definitions for the HIP runtime API.
//!
//! These are pure Rust type aliases and constants — no linking or dynamic
//! loading. They mirror the C types from `<hip/hip_runtime_api.h>` so that
//! the safe wrapper in [`super::hip_runtime`] has correct type signatures.
//!
//! # Platform support
//!
//! These types compile on all platforms. The actual HIP runtime library
//! (`libamdhip64.so`) is Linux-only and loaded dynamically.

use std::ffi::c_void;

/// HIP error code (mirrors `hipError_t`).
pub type HipError = i32;

/// HIP device ordinal (mirrors `hipDevice_t`).
pub type HipDevice = i32;

/// Opaque GPU device pointer (mirrors `hipDeviceptr_t`).
pub type HipDevicePtr = *mut c_void;

/// Opaque HIP stream handle (mirrors `hipStream_t`).
pub type HipStreamHandle = *mut c_void;

/// Opaque HIP module handle (mirrors `hipModule_t`).
pub type HipModuleHandle = *mut c_void;

/// Opaque HIP function/kernel handle (mirrors `hipFunction_t`).
pub type HipFunctionHandle = *mut c_void;

/// HIP error codes from `hipError_t`.
///
/// Subset of commonly encountered error codes. The full list is in
/// `<hip/hip_runtime_api.h>`.
pub mod error_code {
    use super::HipError;

    pub const HIP_SUCCESS: HipError = 0;
    pub const HIP_ERROR_INVALID_VALUE: HipError = 1;
    pub const HIP_ERROR_OUT_OF_MEMORY: HipError = 2;
    pub const HIP_ERROR_NOT_INITIALIZED: HipError = 3;
    pub const HIP_ERROR_INVALID_DEVICE: HipError = 101;
    pub const HIP_ERROR_FILE_NOT_FOUND: HipError = 301;
    pub const HIP_ERROR_NOT_FOUND: HipError = 500;
    pub const HIP_ERROR_LAUNCH_FAILURE: HipError = 719;
    pub const HIP_ERROR_NO_DEVICE: HipError = 100;
}

/// Memory copy direction for `hipMemcpy`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HipMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
}

/// Grid dimensions for kernel launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    #[must_use]
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// 1D grid/block dimension.
    #[must_use]
    pub const fn d1(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// 2D grid/block dimension.
    #[must_use]
    pub const fn d2(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }

    /// Total number of threads in this dimension.
    #[must_use]
    pub const fn total(&self) -> u64 {
        self.x as u64 * self.y as u64 * self.z as u64
    }
}

/// Kernel launch configuration.
///
/// Bundles grid dimensions, block dimensions, and shared memory size
/// into a single struct for ergonomic kernel launches.
#[derive(Debug, Clone, Copy)]
pub struct LaunchConfig {
    pub grid: Dim3,
    pub block: Dim3,
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    /// Create a 1D launch config for `total_elements` with the given block size.
    ///
    /// Grid size is `ceil(total_elements / block_size)`.
    #[must_use]
    pub fn for_elementwise(total_elements: usize, block_size: u32) -> Self {
        let grid_x = (total_elements as u64)
            .div_ceil(u64::from(block_size))
            .min(u64::from(u32::MAX)) as u32;
        Self {
            grid: Dim3::d1(grid_x),
            block: Dim3::d1(block_size),
            shared_mem_bytes: 0,
        }
    }

    /// Create a 1D launch config with shared memory.
    #[must_use]
    pub fn for_reduction(total_elements: usize, block_size: u32) -> Self {
        let grid_x = (total_elements as u64)
            .div_ceil(u64::from(block_size))
            .min(u64::from(u32::MAX)) as u32;
        Self {
            grid: Dim3::d1(grid_x),
            block: Dim3::d1(block_size),
            shared_mem_bytes: block_size * 4, // sizeof(float) per thread
        }
    }

    /// Create a 2D launch config for matmul (M×N output tiles).
    #[must_use]
    pub fn for_matmul(m: usize, n: usize, tile_m: u32, tile_n: u32) -> Self {
        let grid_x = (n as u64)
            .div_ceil(u64::from(tile_n))
            .min(u64::from(u32::MAX)) as u32;
        let grid_y = (m as u64)
            .div_ceil(u64::from(tile_m))
            .min(u64::from(u32::MAX)) as u32;
        Self {
            grid: Dim3::d2(grid_x, grid_y),
            block: Dim3::d2(tile_n, tile_m),
            shared_mem_bytes: 0,
        }
    }

    /// Create a 3D launch config for rocWMMA tiled GEMM.
    ///
    /// Grid: `(ceil(N/32), ceil(M/32), batch_count)`.
    /// Block: `(256, 1, 1)` — 4 wavefronts of 64 threads each.
    /// Shared memory is statically declared in the kernel (`__shared__`).
    #[must_use]
    pub fn for_rocwmma(m: usize, n: usize, batch_count: usize) -> Self {
        let tile = 32u64;
        let grid_x = (n as u64).div_ceil(tile).min(u64::from(u32::MAX)) as u32;
        let grid_y = (m as u64).div_ceil(tile).min(u64::from(u32::MAX)) as u32;
        let grid_z = (batch_count as u64).min(u64::from(u32::MAX)) as u32;
        Self {
            grid: Dim3::new(grid_x, grid_y, grid_z),
            block: Dim3::d1(256),
            shared_mem_bytes: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dim3_constructors() {
        let d = Dim3::d1(256);
        assert_eq!((d.x, d.y, d.z), (256, 1, 1));
        assert_eq!(d.total(), 256);

        let d = Dim3::d2(16, 16);
        assert_eq!((d.x, d.y, d.z), (16, 16, 1));
        assert_eq!(d.total(), 256);

        let d = Dim3::new(4, 8, 2);
        assert_eq!(d.total(), 64);
    }

    #[test]
    fn test_launch_config_elementwise() {
        let cfg = LaunchConfig::for_elementwise(1024, 256);
        assert_eq!(cfg.grid.x, 4);
        assert_eq!(cfg.block.x, 256);
        assert_eq!(cfg.shared_mem_bytes, 0);
    }

    #[test]
    fn test_launch_config_elementwise_not_multiple() {
        let cfg = LaunchConfig::for_elementwise(1000, 256);
        assert_eq!(cfg.grid.x, 4); // ceil(1000/256) = 4
        assert_eq!(cfg.block.x, 256);
    }

    #[test]
    fn test_launch_config_reduction() {
        let cfg = LaunchConfig::for_reduction(1024, 256);
        assert_eq!(cfg.grid.x, 4);
        assert_eq!(cfg.shared_mem_bytes, 256 * 4);
    }

    #[test]
    fn test_launch_config_matmul() {
        let cfg = LaunchConfig::for_matmul(128, 64, 16, 16);
        assert_eq!(cfg.grid.x, 4); // ceil(64/16)
        assert_eq!(cfg.grid.y, 8); // ceil(128/16)
        assert_eq!(cfg.block.x, 16);
        assert_eq!(cfg.block.y, 16);
    }

    #[test]
    fn test_launch_config_rocwmma() {
        let cfg = LaunchConfig::for_rocwmma(256, 128, 4);
        assert_eq!(cfg.grid.x, 4); // ceil(128/32)
        assert_eq!(cfg.grid.y, 8); // ceil(256/32)
        assert_eq!(cfg.grid.z, 4); // batch_count
        assert_eq!(cfg.block.x, 256);
        assert_eq!(cfg.block.y, 1);
        assert_eq!(cfg.block.z, 1);
        assert_eq!(cfg.shared_mem_bytes, 0);
    }

    #[test]
    fn test_memcpy_kind_values() {
        assert_eq!(HipMemcpyKind::HostToDevice as i32, 1);
        assert_eq!(HipMemcpyKind::DeviceToHost as i32, 2);
    }
}
