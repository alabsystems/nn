// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Raw FFI type definitions for the CUDA driver API.
//!
//! Parallel to [`hip_ffi`](super::hip_ffi) — defines pure Rust type aliases
//! and constants that mirror the C types from `<cuda.h>` (driver API) and
//! `<cuda_runtime.h>` (runtime API).
//!
//! These are data-only definitions with no linking or dynamic loading.
//! The safe wrappers in [`super::cuda_runtime`] use these types.
//!
//! # CUDA Driver vs Runtime API
//!
//! The CUDA Driver API (`cuModuleLoad`, `cuLaunchKernel`) gives explicit
//! control over module loading and JIT compilation — matching the HIP
//! `hipModuleLoad`/`hipModuleLaunchKernel` path. The Runtime API
//! (`cudaMalloc`, `cudaMemcpy`) is used for memory management.

use std::ffi::c_void;

/// CUDA error code (mirrors `CUresult` from the Driver API).
pub type CudaError = i32;

/// CUDA device ordinal.
pub type CudaDevice = i32;

/// Opaque GPU device pointer (mirrors `CUdeviceptr`).
///
/// On 64-bit systems this is a `u64`; we use `*mut c_void` for pointer
/// compatibility with `cudaMemcpy`.
pub type CudaDevicePtr = *mut c_void;

/// Opaque CUDA stream handle (mirrors `cudaStream_t` / `CUstream`).
pub type CudaStreamHandle = *mut c_void;

/// Opaque CUDA module handle (mirrors `CUmodule`).
pub type CudaModuleHandle = *mut c_void;

/// Opaque CUDA function handle (mirrors `CUfunction`).
pub type CudaFunctionHandle = *mut c_void;

/// Opaque CUDA context handle (mirrors `CUcontext`).
pub type CudaContextHandle = *mut c_void;

/// CUDA error codes (subset of `cudaError_t` / `CUresult`).
pub mod error_code {
    use super::CudaError;

    pub const CUDA_SUCCESS: CudaError = 0;
    pub const CUDA_ERROR_INVALID_VALUE: CudaError = 1;
    pub const CUDA_ERROR_OUT_OF_MEMORY: CudaError = 2;
    pub const CUDA_ERROR_NOT_INITIALIZED: CudaError = 3;
    pub const CUDA_ERROR_INVALID_DEVICE: CudaError = 101;
    pub const CUDA_ERROR_FILE_NOT_FOUND: CudaError = 301;
    pub const CUDA_ERROR_NOT_FOUND: CudaError = 500;
    pub const CUDA_ERROR_LAUNCH_FAILED: CudaError = 719;
    pub const CUDA_ERROR_NO_DEVICE: CudaError = 100;
}

/// Memory copy direction for `cudaMemcpy`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
}

/// Grid/block dimensions for kernel launch.
///
/// Identical to HIP's `Dim3` — both CUDA and HIP use 3D grid/block dims.
/// Defined separately to avoid coupling the CUDA and HIP subsystems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaDim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl CudaDim3 {
    #[must_use]
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// 1D dimension.
    #[must_use]
    pub const fn d1(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// 2D dimension.
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

/// CUDA kernel launch configuration.
///
/// Parallel to HIP's [`LaunchConfig`](super::hip_ffi::LaunchConfig).
#[derive(Debug, Clone, Copy)]
pub struct CudaLaunchConfig {
    pub grid: CudaDim3,
    pub block: CudaDim3,
    pub shared_mem_bytes: u32,
}

impl CudaLaunchConfig {
    /// 1D launch config for elementwise kernels.
    #[must_use]
    pub fn for_elementwise(total_elements: usize, block_size: u32) -> Self {
        let grid_x = (total_elements as u64)
            .div_ceil(u64::from(block_size))
            .min(u64::from(u32::MAX)) as u32;
        Self {
            grid: CudaDim3::d1(grid_x),
            block: CudaDim3::d1(block_size),
            shared_mem_bytes: 0,
        }
    }

    /// 1D launch config for reduction with shared memory.
    #[must_use]
    pub fn for_reduction(num_rows: usize, block_size: u32) -> Self {
        let grid_x = num_rows.min(u32::MAX as usize) as u32;
        Self {
            grid: CudaDim3::d1(grid_x),
            block: CudaDim3::d1(block_size),
            shared_mem_bytes: block_size * 4, // sizeof(float) per thread
        }
    }

    /// 2D launch config for tiled matmul.
    #[must_use]
    pub fn for_matmul(m: usize, n: usize, tile_m: u32, tile_n: u32) -> Self {
        let grid_x = (n as u64)
            .div_ceil(u64::from(tile_n))
            .min(u64::from(u32::MAX)) as u32;
        let grid_y = (m as u64)
            .div_ceil(u64::from(tile_m))
            .min(u64::from(u32::MAX)) as u32;
        Self {
            grid: CudaDim3::d2(grid_x, grid_y),
            block: CudaDim3::d2(tile_n, tile_m),
            shared_mem_bytes: 0,
        }
    }

    /// 3D launch config for batched operations.
    #[must_use]
    pub fn for_batched(grid_x: u32, grid_y: u32, batch_count: u32, block_x: u32) -> Self {
        Self {
            grid: CudaDim3::new(grid_x, grid_y, batch_count),
            block: CudaDim3::d1(block_x),
            shared_mem_bytes: 0,
        }
    }
}

/// Common NVIDIA GPU compute capability targets.
///
/// Used with `nvcc --gpu-architecture=<target>`.
pub mod sm_target {
    /// Volta (V100). First tensor core architecture.
    pub const SM_70: &str = "sm_70";
    /// Turing (T4, RTX 2080).
    pub const SM_75: &str = "sm_75";
    /// Ampere (A100, RTX 3090). bf16 tensor cores.
    pub const SM_80: &str = "sm_80";
    /// Ampere consumer (RTX 3060, etc.).
    pub const SM_86: &str = "sm_86";
    /// Ada Lovelace (L40, RTX 4090). FP8 tensor cores.
    pub const SM_89: &str = "sm_89";
    /// Hopper (H100, H200). TMA, wgmma.
    pub const SM_90: &str = "sm_90";
    /// Blackwell (B200, GB200). FP4 tensor cores.
    pub const SM_100: &str = "sm_100";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_dim3_constructors() {
        let d = CudaDim3::d1(256);
        assert_eq!((d.x, d.y, d.z), (256, 1, 1));
        assert_eq!(d.total(), 256);

        let d = CudaDim3::d2(16, 16);
        assert_eq!((d.x, d.y, d.z), (16, 16, 1));
        assert_eq!(d.total(), 256);

        let d = CudaDim3::new(4, 8, 2);
        assert_eq!(d.total(), 64);
    }

    #[test]
    fn test_cuda_launch_config_elementwise() {
        let cfg = CudaLaunchConfig::for_elementwise(1024, 256);
        assert_eq!(cfg.grid.x, 4);
        assert_eq!(cfg.block.x, 256);
        assert_eq!(cfg.shared_mem_bytes, 0);
    }

    #[test]
    fn test_cuda_launch_config_elementwise_not_multiple() {
        let cfg = CudaLaunchConfig::for_elementwise(1000, 256);
        assert_eq!(cfg.grid.x, 4); // ceil(1000/256)
        assert_eq!(cfg.block.x, 256);
    }

    #[test]
    fn test_cuda_launch_config_reduction() {
        let cfg = CudaLaunchConfig::for_reduction(32, 256);
        assert_eq!(cfg.grid.x, 32);
        assert_eq!(cfg.shared_mem_bytes, 256 * 4);
    }

    #[test]
    fn test_cuda_launch_config_matmul() {
        let cfg = CudaLaunchConfig::for_matmul(128, 64, 16, 16);
        assert_eq!(cfg.grid.x, 4); // ceil(64/16)
        assert_eq!(cfg.grid.y, 8); // ceil(128/16)
        assert_eq!(cfg.block.x, 16);
        assert_eq!(cfg.block.y, 16);
    }

    #[test]
    fn test_cuda_memcpy_kind_values() {
        assert_eq!(CudaMemcpyKind::HostToDevice as i32, 1);
        assert_eq!(CudaMemcpyKind::DeviceToHost as i32, 2);
    }

    #[test]
    fn test_sm_target_constants() {
        assert_eq!(sm_target::SM_70, "sm_70");
        assert_eq!(sm_target::SM_80, "sm_80");
        assert_eq!(sm_target::SM_90, "sm_90");
        assert_eq!(sm_target::SM_100, "sm_100");
    }
}
