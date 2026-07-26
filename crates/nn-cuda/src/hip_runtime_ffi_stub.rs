// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Stub FFI implementation for non-Linux platforms.
//!
//! All functions return `HipRuntimeError::NotAvailable` since the
//! HIP runtime (`libamdhip64.so`) is Linux-only.

use super::*;
use std::ffi::c_void;

fn not_available<T>() -> Result<T, HipRuntimeError> {
    Err(HipRuntimeError::NotAvailable)
}

pub(super) fn probe_hip_runtime() -> bool {
    false
}

pub(super) fn hip_get_device_count() -> Result<i32, HipRuntimeError> {
    not_available()
}

pub(super) fn hip_set_device(_id: i32) -> Result<(), HipRuntimeError> {
    not_available()
}

pub(super) fn hip_get_device_name(_id: i32) -> Result<String, HipRuntimeError> {
    not_available()
}

pub(super) fn hip_malloc(_bytes: usize) -> Result<HipDevicePtr, HipRuntimeError> {
    not_available()
}

pub(super) fn hip_free(_ptr: HipDevicePtr) -> Result<(), HipRuntimeError> {
    Ok(()) // No-op on non-Linux — buffer was never allocated.
}

pub(super) fn hip_memcpy_htod(
    _dst: HipDevicePtr,
    _src: *const c_void,
    _bytes: usize,
) -> Result<(), HipRuntimeError> {
    not_available()
}

pub(super) fn hip_memcpy_dtoh(
    _dst: *mut c_void,
    _src: HipDevicePtr,
    _bytes: usize,
) -> Result<(), HipRuntimeError> {
    not_available()
}

pub(super) fn hip_module_load(_path: &Path) -> Result<HipModuleHandle, HipRuntimeError> {
    not_available()
}

pub(super) fn hip_module_get_function(
    _module: HipModuleHandle,
    _name: &str,
) -> Result<HipFunctionHandle, HipRuntimeError> {
    not_available()
}

pub(super) fn hip_module_launch_kernel(
    _function: HipFunctionHandle,
    _grid: Dim3,
    _block: Dim3,
    _shared_mem: u32,
    _stream: HipStreamHandle,
    _args: &mut [*mut c_void],
) -> Result<(), HipRuntimeError> {
    not_available()
}

pub(super) fn hip_stream_create() -> Result<HipStreamHandle, HipRuntimeError> {
    not_available()
}

pub(super) fn hip_stream_synchronize(_stream: HipStreamHandle) -> Result<(), HipRuntimeError> {
    not_available()
}

pub(super) fn hip_stream_destroy(_stream: HipStreamHandle) -> Result<(), HipRuntimeError> {
    Ok(()) // No-op — stream was never created.
}
