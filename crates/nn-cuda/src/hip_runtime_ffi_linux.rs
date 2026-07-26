// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Linux FFI implementation for HIP runtime.
//!
//! Links against `libamdhip64.so` (ROCm). This module is only compiled
//! on `target_os = "linux"`.

use super::*;
use crate::hip_ffi::error_code;
use std::ffi::{c_void, CString};

// Link against libamdhip64.so (ROCm).
// This will fail to link if ROCm is not installed, which is the
// intended behavior — the crate compiles but runtime init fails.
extern "C" {
    fn hipInit(flags: u32) -> i32;
    fn hipGetDeviceCount(count: *mut i32) -> i32;
    fn hipSetDevice(device_id: i32) -> i32;
    fn hipGetDeviceProperties(prop: *mut HipDeviceProperties, device_id: i32) -> i32;
    fn hipMalloc(dev_ptr: *mut HipDevicePtr, size: usize) -> i32;
    fn hipFree(dev_ptr: HipDevicePtr) -> i32;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, size_bytes: usize, kind: i32) -> i32;
    fn hipModuleLoad(module: *mut HipModuleHandle, fname: *const std::ffi::c_char) -> i32;
    fn hipModuleGetFunction(
        function: *mut HipFunctionHandle,
        module: HipModuleHandle,
        name: *const std::ffi::c_char,
    ) -> i32;
    fn hipModuleLaunchKernel(
        f: HipFunctionHandle,
        grid_dim_x: u32,
        grid_dim_y: u32,
        grid_dim_z: u32,
        block_dim_x: u32,
        block_dim_y: u32,
        block_dim_z: u32,
        shared_mem_bytes: u32,
        stream: HipStreamHandle,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
    fn hipStreamCreate(stream: *mut HipStreamHandle) -> i32;
    fn hipStreamSynchronize(stream: HipStreamHandle) -> i32;
    fn hipStreamDestroy(stream: HipStreamHandle) -> i32;
}

/// Subset of hipDeviceProperties used for device name.
#[repr(C)]
struct HipDeviceProperties {
    name: [u8; 256],
    // Many more fields follow but we only need the name.
    _padding: [u8; 1024],
}

fn check(code: i32, function: &'static str) -> Result<(), HipRuntimeError> {
    if code == error_code::HIP_SUCCESS {
        Ok(())
    } else if code == error_code::HIP_ERROR_OUT_OF_MEMORY {
        Err(HipRuntimeError::OutOfMemory { requested: 0 })
    } else {
        Err(HipRuntimeError::ApiError { function, code })
    }
}

pub(super) fn probe_hip_runtime() -> bool {
    unsafe { hipInit(0) == error_code::HIP_SUCCESS }
}

pub(super) fn hip_get_device_count() -> Result<i32, HipRuntimeError> {
    let mut count: i32 = 0;
    let code = unsafe { hipGetDeviceCount(&mut count) };
    check(code, "hipGetDeviceCount")?;
    Ok(count)
}

pub(super) fn hip_set_device(device_id: i32) -> Result<(), HipRuntimeError> {
    let code = unsafe { hipSetDevice(device_id) };
    check(code, "hipSetDevice")
}

pub(super) fn hip_get_device_name(device_id: i32) -> Result<String, HipRuntimeError> {
    let mut props = HipDeviceProperties {
        name: [0u8; 256],
        _padding: [0u8; 1024],
    };
    let code = unsafe { hipGetDeviceProperties(&mut props, device_id) };
    check(code, "hipGetDeviceProperties")?;
    let name = props
        .name
        .iter()
        .position(|&b| b == 0)
        .map(|end| String::from_utf8_lossy(&props.name[..end]).into_owned())
        .unwrap_or_default();
    Ok(name)
}

pub(super) fn hip_malloc(bytes: usize) -> Result<HipDevicePtr, HipRuntimeError> {
    let mut ptr: HipDevicePtr = std::ptr::null_mut();
    let code = unsafe { hipMalloc(&mut ptr, bytes) };
    if code == error_code::HIP_ERROR_OUT_OF_MEMORY {
        return Err(HipRuntimeError::OutOfMemory { requested: bytes });
    }
    check(code, "hipMalloc")?;
    Ok(ptr)
}

pub(super) fn hip_free(ptr: HipDevicePtr) -> Result<(), HipRuntimeError> {
    let code = unsafe { hipFree(ptr) };
    check(code, "hipFree")
}

pub(super) fn hip_memcpy_htod(
    dst: HipDevicePtr,
    src: *const c_void,
    bytes: usize,
) -> Result<(), HipRuntimeError> {
    let code = unsafe {
        hipMemcpy(
            dst,
            src,
            bytes,
            crate::hip_ffi::HipMemcpyKind::HostToDevice as i32,
        )
    };
    check(code, "hipMemcpy(HtoD)")
}

pub(super) fn hip_memcpy_dtoh(
    dst: *mut c_void,
    src: HipDevicePtr,
    bytes: usize,
) -> Result<(), HipRuntimeError> {
    let code = unsafe {
        hipMemcpy(
            dst,
            src,
            bytes,
            crate::hip_ffi::HipMemcpyKind::DeviceToHost as i32,
        )
    };
    check(code, "hipMemcpy(DtoH)")
}

pub(super) fn hip_module_load(path: &Path) -> Result<HipModuleHandle, HipRuntimeError> {
    let c_path = CString::new(path.to_string_lossy().as_bytes()).map_err(|_| {
        HipRuntimeError::ModuleLoadFailed {
            path: path.display().to_string(),
            reason: "path contains null byte".into(),
        }
    })?;
    let mut module: HipModuleHandle = std::ptr::null_mut();
    let code = unsafe { hipModuleLoad(&mut module, c_path.as_ptr()) };
    if code == error_code::HIP_ERROR_FILE_NOT_FOUND {
        return Err(HipRuntimeError::ModuleLoadFailed {
            path: path.display().to_string(),
            reason: "file not found".into(),
        });
    }
    check(code, "hipModuleLoad")?;
    Ok(module)
}

pub(super) fn hip_module_get_function(
    module: HipModuleHandle,
    name: &str,
) -> Result<HipFunctionHandle, HipRuntimeError> {
    let c_name = CString::new(name).map_err(|_| HipRuntimeError::KernelNotFound {
        name: name.to_owned(),
    })?;
    let mut func: HipFunctionHandle = std::ptr::null_mut();
    let code = unsafe { hipModuleGetFunction(&mut func, module, c_name.as_ptr()) };
    if code == error_code::HIP_ERROR_NOT_FOUND {
        return Err(HipRuntimeError::KernelNotFound {
            name: name.to_owned(),
        });
    }
    check(code, "hipModuleGetFunction")?;
    Ok(func)
}

pub(super) fn hip_module_launch_kernel(
    function: HipFunctionHandle,
    grid: Dim3,
    block: Dim3,
    shared_mem: u32,
    stream: HipStreamHandle,
    args: &mut [*mut c_void],
) -> Result<(), HipRuntimeError> {
    let code = unsafe {
        hipModuleLaunchKernel(
            function,
            grid.x,
            grid.y,
            grid.z,
            block.x,
            block.y,
            block.z,
            shared_mem,
            stream,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    check(code, "hipModuleLaunchKernel")
}

pub(super) fn hip_stream_create() -> Result<HipStreamHandle, HipRuntimeError> {
    let mut stream: HipStreamHandle = std::ptr::null_mut();
    let code = unsafe { hipStreamCreate(&mut stream) };
    check(code, "hipStreamCreate")?;
    Ok(stream)
}

pub(super) fn hip_stream_synchronize(stream: HipStreamHandle) -> Result<(), HipRuntimeError> {
    let code = unsafe { hipStreamSynchronize(stream) };
    check(code, "hipStreamSynchronize")
}

pub(super) fn hip_stream_destroy(stream: HipStreamHandle) -> Result<(), HipRuntimeError> {
    let code = unsafe { hipStreamDestroy(stream) };
    check(code, "hipStreamDestroy")
}
