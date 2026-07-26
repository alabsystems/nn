// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Safe Rust wrapper for the CUDA runtime and driver APIs.
//!
//! Parallel to [`hip_runtime`](super::hip_runtime). Provides [`CudaRuntime`],
//! [`CudaBuffer`], [`CudaStream`], and [`CudaKernel`] for GPU buffer
//! management, kernel loading from `.ptx`/`.cubin` files, and kernel dispatch
//! on NVIDIA GPUs.
//!
//! # Platform support
//!
//! The CUDA runtime (`libcuda.so` / `libcudart.so`) is only available on
//! Linux and Windows with NVIDIA drivers. On macOS and other platforms,
//! [`CudaRuntime::init`] returns [`CudaRuntimeError::NotAvailable`].
//! All codegen and compilation features work cross-platform; only runtime
//! dispatch requires NVIDIA hardware.
//!
//! # Safety model
//!
//! Raw FFI types are defined in [`super::cuda_ffi`]. This module wraps them
//! in safe Rust types with RAII cleanup. GPU buffers are freed on drop,
//! streams are destroyed on drop, and modules are unloaded on drop.

use crate::compile_ptx::PtxModule;
use crate::cuda_ffi::{
    CudaDevicePtr, CudaFunctionHandle, CudaLaunchConfig, CudaModuleHandle, CudaStreamHandle,
};

/// Errors from the CUDA runtime.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CudaRuntimeError {
    #[error("CUDA runtime not available — NVIDIA driver and CUDA Toolkit required")]
    NotAvailable,

    #[error("no NVIDIA GPU devices found")]
    NoDevices,

    #[error("CUDA API call failed: {function}() returned error {code}")]
    ApiError { function: &'static str, code: i32 },

    #[error("GPU out of memory: requested {requested} bytes")]
    OutOfMemory { requested: usize },

    #[error("kernel not found in module: {name}")]
    KernelNotFound { name: String },

    #[error("failed to load module from {path}: {reason}")]
    ModuleLoadFailed { path: String, reason: String },

    #[error("launch config invalid: {reason}")]
    InvalidLaunchConfig { reason: String },

    #[error("buffer size mismatch: expected {expected} bytes, got {actual}")]
    BufferSizeMismatch { expected: usize, actual: usize },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// CUDA runtime context.
///
/// Manages the lifecycle of the CUDA runtime on a single device. On platforms
/// without NVIDIA drivers, [`init`](Self::init) returns
/// [`CudaRuntimeError::NotAvailable`].
#[derive(Debug)]
pub struct CudaRuntime {
    device_id: i32,
    device_name: String,
    compute_capability: (i32, i32),
}

impl CudaRuntime {
    /// Initialize the CUDA runtime and select a device.
    ///
    /// # Errors
    ///
    /// Returns [`CudaRuntimeError::NotAvailable`] on platforms without CUDA.
    /// Returns [`CudaRuntimeError::NoDevices`] if no NVIDIA GPUs are found.
    pub fn init(device_ordinal: i32) -> Result<Self, CudaRuntimeError> {
        if !is_cuda_available() {
            return Err(CudaRuntimeError::NotAvailable);
        }

        let count = cuda_get_device_count()?;
        if count == 0 {
            return Err(CudaRuntimeError::NoDevices);
        }
        if device_ordinal >= count {
            return Err(CudaRuntimeError::ApiError {
                function: "cudaSetDevice",
                code: 101, // cudaErrorInvalidDevice
            });
        }

        cuda_set_device(device_ordinal)?;
        let name = cuda_get_device_name(device_ordinal)?;
        let cc = cuda_get_compute_capability(device_ordinal)?;

        Ok(Self {
            device_id: device_ordinal,
            device_name: name,
            compute_capability: cc,
        })
    }

    /// The selected device ordinal.
    #[must_use]
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// The device name (e.g., "NVIDIA A100-SXM4-80GB").
    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Compute capability as (major, minor) — e.g., (8, 0) for A100.
    #[must_use]
    pub fn compute_capability(&self) -> (i32, i32) {
        self.compute_capability
    }

    /// SM target string for this device (e.g., "sm_80").
    #[must_use]
    pub fn sm_target(&self) -> String {
        format!(
            "sm_{}{}",
            self.compute_capability.0, self.compute_capability.1
        )
    }

    /// Allocate a GPU buffer for `count` f32 elements.
    pub fn alloc_f32(&self, count: usize) -> Result<CudaBuffer, CudaRuntimeError> {
        let bytes = count.checked_mul(4).ok_or(CudaRuntimeError::OutOfMemory {
            requested: usize::MAX,
        })?;
        let ptr = cuda_malloc(bytes)?;
        Ok(CudaBuffer {
            ptr,
            byte_len: bytes,
            elem_count: count,
        })
    }

    /// Copy f32 data from host to a device buffer.
    pub fn copy_to_device(&self, buf: &CudaBuffer, data: &[f32]) -> Result<(), CudaRuntimeError> {
        let bytes = data.len() * 4;
        if bytes > buf.byte_len {
            return Err(CudaRuntimeError::BufferSizeMismatch {
                expected: buf.byte_len,
                actual: bytes,
            });
        }
        cuda_memcpy_htod(buf.ptr, data.as_ptr().cast(), bytes)
    }

    /// Copy f32 data from a device buffer to host.
    pub fn copy_to_host(&self, buf: &CudaBuffer, data: &mut [f32]) -> Result<(), CudaRuntimeError> {
        let bytes = data.len() * 4;
        if bytes > buf.byte_len {
            return Err(CudaRuntimeError::BufferSizeMismatch {
                expected: buf.byte_len,
                actual: bytes,
            });
        }
        cuda_memcpy_dtoh(data.as_mut_ptr().cast(), buf.ptr, bytes)
    }

    /// Load a compiled `.ptx` or `.cubin` module and extract a kernel by name.
    pub fn load_kernel(
        &self,
        module: &PtxModule,
        kernel_name: &str,
    ) -> Result<CudaKernel, CudaRuntimeError> {
        let path = module.cubin_path.as_ref().unwrap_or(&module.ptx_path);
        let mod_handle = cuda_module_load(path)?;
        let func = cuda_module_get_function(mod_handle, kernel_name)?;
        Ok(CudaKernel {
            _module: mod_handle,
            function: func,
            name: kernel_name.to_owned(),
        })
    }

    /// Create a new CUDA stream for asynchronous dispatch.
    pub fn create_stream(&self) -> Result<CudaStream, CudaRuntimeError> {
        let handle = cuda_stream_create()?;
        Ok(CudaStream { handle })
    }

    /// Launch a kernel on a stream with the given configuration and buffer arguments.
    pub fn launch(
        &self,
        kernel: &CudaKernel,
        stream: &CudaStream,
        config: CudaLaunchConfig,
        buffers: &[&CudaBuffer],
    ) -> Result<(), CudaRuntimeError> {
        validate_launch_config(&config)?;

        let mut arg_ptrs: Vec<*mut std::ffi::c_void> = buffers.iter().map(|b| b.ptr).collect();

        cuda_launch_kernel(
            kernel.function,
            config.grid,
            config.block,
            config.shared_mem_bytes,
            stream.handle,
            &mut arg_ptrs,
        )
    }
}

/// GPU memory buffer.
///
/// Freed via `cudaFree` on drop.
pub struct CudaBuffer {
    ptr: CudaDevicePtr,
    byte_len: usize,
    elem_count: usize,
}

impl CudaBuffer {
    /// Size in bytes.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    /// Number of elements (assumes f32).
    #[must_use]
    pub fn elem_count(&self) -> usize {
        self.elem_count
    }

    /// Raw device pointer (for FFI).
    #[must_use]
    pub fn as_device_ptr(&self) -> CudaDevicePtr {
        self.ptr
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = cuda_free(self.ptr);
        }
    }
}

/// CUDA stream for asynchronous kernel dispatch.
///
/// Destroyed via `cudaStreamDestroy` on drop.
pub struct CudaStream {
    handle: CudaStreamHandle,
}

impl CudaStream {
    /// Block until all operations on this stream complete.
    pub fn synchronize(&self) -> Result<(), CudaRuntimeError> {
        cuda_stream_synchronize(self.handle)
    }

    /// Raw stream handle (for FFI).
    #[must_use]
    pub fn as_raw(&self) -> CudaStreamHandle {
        self.handle
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let _ = cuda_stream_destroy(self.handle);
        }
    }
}

/// Loaded kernel function, ready for dispatch.
///
/// Holds both the module handle and the function handle.
pub struct CudaKernel {
    _module: CudaModuleHandle,
    function: CudaFunctionHandle,
    name: String,
}

impl CudaKernel {
    /// The kernel function name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

// ---------------------------------------------------------------------------
// Platform-gated CUDA runtime function wrappers
// ---------------------------------------------------------------------------

/// Check if the CUDA runtime is available on this platform.
#[must_use]
pub fn is_cuda_available() -> bool {
    cfg!(target_os = "linux") && probe_cuda_runtime()
}

/// Number of CUDA-capable devices.
pub fn cuda_device_count() -> Result<i32, CudaRuntimeError> {
    if !is_cuda_available() {
        return Err(CudaRuntimeError::NotAvailable);
    }
    cuda_get_device_count()
}

pub(crate) fn validate_launch_config(config: &CudaLaunchConfig) -> Result<(), CudaRuntimeError> {
    if config.block.x == 0 || config.block.y == 0 || config.block.z == 0 {
        return Err(CudaRuntimeError::InvalidLaunchConfig {
            reason: "block dimensions must be non-zero".into(),
        });
    }
    if config.grid.x == 0 || config.grid.y == 0 || config.grid.z == 0 {
        return Err(CudaRuntimeError::InvalidLaunchConfig {
            reason: "grid dimensions must be non-zero".into(),
        });
    }
    let threads_per_block = config.block.total();
    if threads_per_block > 1024 {
        return Err(CudaRuntimeError::InvalidLaunchConfig {
            reason: format!("threads per block ({threads_per_block}) exceeds max (1024)"),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// FFI dispatch layer — cfg-gated per platform
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
#[path = "cuda_runtime_ffi_linux.rs"]
mod ffi_impl;

#[cfg(not(target_os = "linux"))]
#[path = "cuda_runtime_ffi_stub.rs"]
mod ffi_impl;

use ffi_impl::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_ffi::CudaDim3;

    #[test]
    fn test_validate_launch_config_valid() {
        let config = CudaLaunchConfig {
            grid: CudaDim3::d1(4),
            block: CudaDim3::d1(256),
            shared_mem_bytes: 0,
        };
        assert!(validate_launch_config(&config).is_ok());
    }

    #[test]
    fn test_validate_launch_config_zero_block() {
        let config = CudaLaunchConfig {
            grid: CudaDim3::d1(4),
            block: CudaDim3::d1(0),
            shared_mem_bytes: 0,
        };
        assert!(validate_launch_config(&config).is_err());
    }

    #[test]
    fn test_validate_launch_config_zero_grid() {
        let config = CudaLaunchConfig {
            grid: CudaDim3::d1(0),
            block: CudaDim3::d1(256),
            shared_mem_bytes: 0,
        };
        assert!(validate_launch_config(&config).is_err());
    }

    #[test]
    fn test_validate_launch_config_too_many_threads() {
        let config = CudaLaunchConfig {
            grid: CudaDim3::d1(1),
            block: CudaDim3::d2(64, 32), // 2048 threads
            shared_mem_bytes: 0,
        };
        assert!(validate_launch_config(&config).is_err());
    }

    #[test]
    fn test_cuda_not_available_on_macos() {
        // On macOS (dev machine), CUDA is never available.
        if cfg!(target_os = "macos") {
            assert!(!is_cuda_available());
        }
    }

    #[test]
    fn test_init_returns_not_available_on_macos() {
        if cfg!(target_os = "macos") {
            let result = CudaRuntime::init(0);
            assert!(matches!(result, Err(CudaRuntimeError::NotAvailable)));
        }
    }
}
