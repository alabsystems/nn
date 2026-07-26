// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Linux CUDA FFI implementations using dynamic loading.
//!
//! Loads `libcuda.so` (Driver API) and `libcudart.so` (Runtime API) at
//! runtime via `dlopen`. This avoids a hard link-time dependency on the
//! CUDA Toolkit, allowing the same binary to run on systems with or without
//! NVIDIA drivers.
//!
//! # Current status
//!
//! Scaffolding — function signatures match the CUDA API but the actual
//! `dlopen`/`dlsym` loading is deferred. All functions currently return
//! `NotAvailable` until the dynamic loading infrastructure is implemented.

use std::ffi::c_void;
use std::path::Path;

use super::CudaRuntimeError;
use crate::cuda_ffi::{
    CudaDevicePtr, CudaDim3, CudaFunctionHandle, CudaModuleHandle, CudaStreamHandle,
};

/// Probe for `libcuda.so` availability.
///
/// Attempts `dlopen("libcuda.so.1", RTLD_LAZY)`. Returns `true` if the
/// library loads successfully.
pub(super) fn probe_cuda_runtime() -> bool {
    // TODO: Implement dlopen probe for libcuda.so.1
    // For now, return false — actual runtime dispatch requires the dlopen infra.
    false
}

pub(super) fn cuda_get_device_count() -> Result<i32, CudaRuntimeError> {
    Err(CudaRuntimeError::NotAvailable)
}

pub(super) fn cuda_set_device(_device: i32) -> Result<(), CudaRuntimeError> {
    Err(CudaRuntimeError::NotAvailable)
}

pub(super) fn cuda_get_device_name(_device: i32) -> Result<String, CudaRuntimeError> {
    Err(CudaRuntimeError::NotAvailable)
}

pub(super) fn cuda_get_compute_capability(_device: i32) -> Result<(i32, i32), CudaRuntimeError> {
    Err(CudaRuntimeError::NotAvailable)
}

pub(super) fn cuda_malloc(_bytes: usize) -> Result<CudaDevicePtr, CudaRuntimeError> {
    Err(CudaRuntimeError::NotAvailable)
}

pub(super) fn cuda_free(_ptr: CudaDevicePtr) -> Result<(), CudaRuntimeError> {
    Err(CudaRuntimeError::NotAvailable)
}

pub(super) fn cuda_memcpy_htod(
    _dst: CudaDevicePtr,
    _src: *const c_void,
    _bytes: usize,
) -> Result<(), CudaRuntimeError> {
    Err(CudaRuntimeError::NotAvailable)
}

pub(super) fn cuda_memcpy_dtoh(
    _dst: *mut c_void,
    _src: CudaDevicePtr,
    _bytes: usize,
) -> Result<(), CudaRuntimeError> {
    Err(CudaRuntimeError::NotAvailable)
}

pub(super) fn cuda_module_load(_path: &Path) -> Result<CudaModuleHandle, CudaRuntimeError> {
    Err(CudaRuntimeError::NotAvailable)
}

pub(super) fn cuda_module_get_function(
    _module: CudaModuleHandle,
    _name: &str,
) -> Result<CudaFunctionHandle, CudaRuntimeError> {
    Err(CudaRuntimeError::NotAvailable)
}

pub(super) fn cuda_stream_create() -> Result<CudaStreamHandle, CudaRuntimeError> {
    Err(CudaRuntimeError::NotAvailable)
}

pub(super) fn cuda_stream_synchronize(_stream: CudaStreamHandle) -> Result<(), CudaRuntimeError> {
    Err(CudaRuntimeError::NotAvailable)
}

pub(super) fn cuda_stream_destroy(_stream: CudaStreamHandle) -> Result<(), CudaRuntimeError> {
    Err(CudaRuntimeError::NotAvailable)
}

pub(super) fn cuda_launch_kernel(
    _func: CudaFunctionHandle,
    _grid: CudaDim3,
    _block: CudaDim3,
    _shared_mem: u32,
    _stream: CudaStreamHandle,
    _args: &mut [*mut c_void],
) -> Result<(), CudaRuntimeError> {
    Err(CudaRuntimeError::NotAvailable)
}
