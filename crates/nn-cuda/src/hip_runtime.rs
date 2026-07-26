// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Safe Rust wrapper for the HIP runtime API.
//!
//! Provides [`HipRuntime`], [`HipBuffer`], [`HipStream`], and [`HipKernel`]
//! for GPU buffer management, kernel loading from `.hsaco` code objects,
//! and kernel dispatch on AMD GPUs.
//!
//! # Platform support
//!
//! The HIP runtime (`libamdhip64.so`) is only available on Linux with ROCm.
//! On macOS and other platforms, [`HipRuntime::init`] returns
//! [`HipRuntimeError::NotAvailable`]. All codegen and compilation features
//! work cross-platform; only runtime dispatch requires AMD hardware.
//!
//! # Safety model
//!
//! Raw FFI types are defined in [`super::hip_ffi`]. This module wraps them
//! in safe Rust types with RAII cleanup. GPU buffers are freed on drop,
//! streams are destroyed on drop, and modules are unloaded on drop.

use std::path::Path;

use crate::compile_hip::HipModule;
use crate::hip_ffi::{
    Dim3, HipDevicePtr, HipFunctionHandle, HipModuleHandle, HipStreamHandle, LaunchConfig,
};

/// Errors from the HIP runtime.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HipRuntimeError {
    #[error("HIP runtime not available — ROCm is required (Linux + AMD GPU)")]
    NotAvailable,

    #[error("no AMD GPU devices found")]
    NoDevices,

    #[error("HIP API call failed: {function}() returned error {code}")]
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

/// HIP runtime context.
///
/// Manages the lifecycle of the HIP runtime. On platforms without ROCm,
/// [`init`](Self::init) returns [`HipRuntimeError::NotAvailable`].
///
/// # Example (Linux with ROCm)
///
/// ```ignore
/// let rt = HipRuntime::init(0)?;
/// let buf = rt.alloc_f32(1024)?;
/// rt.copy_to_device(&buf, &host_data)?;
/// let kernel = rt.load_kernel(&module, "nn_kernel")?;
/// let stream = rt.create_stream()?;
/// rt.launch(&kernel, &stream, config, &[&buf])?;
/// stream.synchronize()?;
/// rt.copy_to_host(&buf, &mut output)?;
/// ```
#[derive(Debug)]
pub struct HipRuntime {
    device_id: i32,
    device_name: String,
}

impl HipRuntime {
    /// Initialize the HIP runtime and select a device.
    ///
    /// # Errors
    ///
    /// Returns [`HipRuntimeError::NotAvailable`] on platforms without ROCm.
    /// Returns [`HipRuntimeError::NoDevices`] if no AMD GPUs are found.
    pub fn init(device_ordinal: i32) -> Result<Self, HipRuntimeError> {
        if !is_hip_available() {
            return Err(HipRuntimeError::NotAvailable);
        }

        let count = hip_get_device_count()?;
        if count == 0 {
            return Err(HipRuntimeError::NoDevices);
        }
        if device_ordinal >= count {
            return Err(HipRuntimeError::ApiError {
                function: "hipSetDevice",
                code: 101, // hipErrorInvalidDevice
            });
        }

        hip_set_device(device_ordinal)?;
        let name = hip_get_device_name(device_ordinal)?;

        Ok(Self {
            device_id: device_ordinal,
            device_name: name,
        })
    }

    /// The selected device ordinal.
    #[must_use]
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// The device name (e.g., "AMD Instinct MI300X").
    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Allocate a GPU buffer for `count` f32 elements.
    pub fn alloc_f32(&self, count: usize) -> Result<HipBuffer, HipRuntimeError> {
        let bytes = count.checked_mul(4).ok_or(HipRuntimeError::OutOfMemory {
            requested: usize::MAX,
        })?;
        let ptr = hip_malloc(bytes)?;
        Ok(HipBuffer {
            ptr,
            byte_len: bytes,
            elem_count: count,
        })
    }

    /// Copy f32 data from host to a device buffer.
    pub fn copy_to_device(&self, buf: &HipBuffer, data: &[f32]) -> Result<(), HipRuntimeError> {
        let bytes = data.len() * 4;
        if bytes > buf.byte_len {
            return Err(HipRuntimeError::BufferSizeMismatch {
                expected: buf.byte_len,
                actual: bytes,
            });
        }
        hip_memcpy_htod(buf.ptr, data.as_ptr().cast(), bytes)
    }

    /// Copy f32 data from a device buffer to host.
    pub fn copy_to_host(&self, buf: &HipBuffer, data: &mut [f32]) -> Result<(), HipRuntimeError> {
        let bytes = data.len() * 4;
        if bytes > buf.byte_len {
            return Err(HipRuntimeError::BufferSizeMismatch {
                expected: buf.byte_len,
                actual: bytes,
            });
        }
        hip_memcpy_dtoh(data.as_mut_ptr().cast(), buf.ptr, bytes)
    }

    /// Load a compiled `.hsaco` module and extract a kernel by name.
    pub fn load_kernel(
        &self,
        module: &HipModule,
        kernel_name: &str,
    ) -> Result<HipKernel, HipRuntimeError> {
        let mod_handle = hip_module_load(&module.hsaco_path)?;
        let func = hip_module_get_function(mod_handle, kernel_name)?;
        Ok(HipKernel {
            _module: mod_handle,
            function: func,
            name: kernel_name.to_owned(),
        })
    }

    /// Create a new HIP stream for asynchronous dispatch.
    pub fn create_stream(&self) -> Result<HipStream, HipRuntimeError> {
        let handle = hip_stream_create()?;
        Ok(HipStream { handle })
    }

    /// Launch a kernel on a stream with the given configuration and buffer arguments.
    ///
    /// Each buffer pointer is passed as a kernel argument in order.
    pub fn launch(
        &self,
        kernel: &HipKernel,
        stream: &HipStream,
        config: LaunchConfig,
        buffers: &[&HipBuffer],
    ) -> Result<(), HipRuntimeError> {
        validate_launch_config(&config)?;

        let mut arg_ptrs: Vec<*mut std::ffi::c_void> = buffers.iter().map(|b| b.ptr).collect();

        hip_module_launch_kernel(
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
/// Freed via `hipFree` on drop. The buffer is bound to the device that
/// was active when it was allocated.
pub struct HipBuffer {
    ptr: HipDevicePtr,
    byte_len: usize,
    elem_count: usize,
}

impl HipBuffer {
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

    /// Raw device pointer (for advanced use / FFI).
    #[must_use]
    pub fn as_device_ptr(&self) -> HipDevicePtr {
        self.ptr
    }
}

impl Drop for HipBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = hip_free(self.ptr);
        }
    }
}

/// HIP stream for asynchronous kernel dispatch.
///
/// Destroyed via `hipStreamDestroy` on drop.
pub struct HipStream {
    handle: HipStreamHandle,
}

impl HipStream {
    /// Block until all operations on this stream complete.
    pub fn synchronize(&self) -> Result<(), HipRuntimeError> {
        hip_stream_synchronize(self.handle)
    }

    /// Raw stream handle (for advanced use / FFI).
    #[must_use]
    pub fn as_raw(&self) -> HipStreamHandle {
        self.handle
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let _ = hip_stream_destroy(self.handle);
        }
    }
}

/// Loaded kernel function, ready for dispatch.
///
/// Holds both the module handle and the function handle. The module is
/// unloaded when the kernel is dropped.
pub struct HipKernel {
    _module: HipModuleHandle,
    function: HipFunctionHandle,
    name: String,
}

impl HipKernel {
    /// The kernel function name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

// ---------------------------------------------------------------------------
// Platform-gated HIP runtime function wrappers
// ---------------------------------------------------------------------------
//
// On Linux with ROCm, these call the actual HIP runtime via FFI.
// On macOS and other platforms, they return NotAvailable.

/// Check if the HIP runtime library is loadable on this platform.
#[must_use]
pub fn is_hip_available() -> bool {
    cfg!(target_os = "linux") && probe_hip_runtime()
}

/// Number of HIP-capable devices.
pub fn hip_device_count() -> Result<i32, HipRuntimeError> {
    if !is_hip_available() {
        return Err(HipRuntimeError::NotAvailable);
    }
    hip_get_device_count()
}

pub(crate) fn validate_launch_config(config: &LaunchConfig) -> Result<(), HipRuntimeError> {
    if config.block.x == 0 || config.block.y == 0 || config.block.z == 0 {
        return Err(HipRuntimeError::InvalidLaunchConfig {
            reason: "block dimensions must be non-zero".into(),
        });
    }
    if config.grid.x == 0 || config.grid.y == 0 || config.grid.z == 0 {
        return Err(HipRuntimeError::InvalidLaunchConfig {
            reason: "grid dimensions must be non-zero".into(),
        });
    }
    let threads_per_block = config.block.total();
    if threads_per_block > 1024 {
        return Err(HipRuntimeError::InvalidLaunchConfig {
            reason: format!("threads per block ({threads_per_block}) exceeds max (1024)"),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// FFI dispatch layer — cfg-gated per platform (extracted for file length)
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
#[path = "hip_runtime_ffi_linux.rs"]
mod ffi_impl;

#[cfg(not(target_os = "linux"))]
#[path = "hip_runtime_ffi_stub.rs"]
mod ffi_impl;

// Re-export the platform-specific implementations.
use ffi_impl::*;

#[cfg(test)]
#[path = "hip_runtime_tests.rs"]
mod tests;
