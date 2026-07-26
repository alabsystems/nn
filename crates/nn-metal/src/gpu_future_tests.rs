// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`GpuFuture`] — non-blocking GPU submit with callbacks.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::*;
use crate::gpu_scope::{get_or_create_batch, is_lazy_batch_active};
use crate::test_common::init;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

#[test]
fn test_submit_current_returns_none_when_no_batch() {
    assert!(!is_lazy_batch_active());
    let result = GpuFuture::submit_current();
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
}

#[test]
fn test_submit_current_returns_future_when_batch_pending() {
    init();

    get_or_create_batch().unwrap();
    assert!(is_lazy_batch_active());

    let future = GpuFuture::submit_current()
        .expect("submit_current should succeed")
        .expect("should return Some(GpuFuture) when batch is pending");

    assert!(
        !is_lazy_batch_active(),
        "lazy batch should be consumed after future submit"
    );

    future.wait().expect("future wait should succeed");
}

#[test]
fn test_future_is_complete_after_wait() {
    init();

    let device = Device::metal();
    let a = DynTensor::full(&[4], 2.0, DType::F32, &device).unwrap();
    let _b = a.add_scalar(1.0).unwrap();

    let future = GpuFuture::submit_current()
        .expect("submit_current should succeed")
        .expect("should return Some(GpuFuture)");

    future.wait().expect("future wait should succeed");
}

#[test]
fn test_future_on_complete_callback() {
    init();

    let device = Device::metal();
    let a = DynTensor::full(&[8], 3.0, DType::F32, &device).unwrap();
    let _b = a.mul_scalar(2.0).unwrap();

    let future = GpuFuture::submit_current()
        .unwrap()
        .expect("should return GpuFuture");

    let done = Arc::new(AtomicBool::new(false));
    let done_clone = done.clone();
    let success_flag = Arc::new(AtomicBool::new(false));
    let success_clone = success_flag.clone();

    future
        .on_complete(move |success| {
            success_clone.store(success, Ordering::Release);
            done_clone.store(true, Ordering::Release);
        })
        .expect("on_complete registration should succeed");

    // Wait for the GPU work to complete via blocking wait on the future.
    future.wait().expect("future wait should succeed");

    // After wait, the callback should have fired.
    // Give the callback thread a moment to complete.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !done.load(Ordering::Acquire) {
        if std::time::Instant::now() >= deadline {
            panic!("callback did not fire within 5 seconds");
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    assert!(
        success_flag.load(Ordering::Acquire),
        "callback should report success"
    );
}

#[test]
fn test_future_on_complete_rejects_double_registration() {
    init();

    get_or_create_batch().unwrap();
    let future = GpuFuture::submit_current().unwrap().unwrap();

    future
        .on_complete(|_| {})
        .expect("first on_complete should succeed");

    let result = future.on_complete(|_| {});
    assert!(
        result.is_err(),
        "second on_complete should be rejected, got: {result:?}"
    );

    future.wait().unwrap();
}

#[test]
fn test_multiple_futures_pipeline() {
    init();

    let device = Device::metal();

    // Segment 1.
    let a = DynTensor::full(&[8], 3.0, DType::F32, &device).unwrap();
    let _seg1 = a.mul_scalar(2.0).unwrap();

    let fut1 = GpuFuture::submit_current()
        .unwrap()
        .expect("should return future for segment 1");

    // Segment 2 while segment 1 may still execute.
    let b = DynTensor::full(&[8], 5.0, DType::F32, &device).unwrap();
    let _seg2 = b.add_scalar(1.0).unwrap();

    let fut2 = GpuFuture::submit_current()
        .unwrap()
        .expect("should return future for segment 2");

    fut1.wait().expect("future 1 wait should succeed");
    fut2.wait().expect("future 2 wait should succeed");
}

#[test]
fn test_future_debug_format() {
    init();

    get_or_create_batch().unwrap();
    let future = GpuFuture::submit_current().unwrap().unwrap();

    let debug_str = format!("{future:?}");
    assert!(
        debug_str.contains("GpuFuture"),
        "debug format should contain type name, got: {debug_str}"
    );
    assert!(
        debug_str.contains("is_complete"),
        "debug format should show is_complete field, got: {debug_str}"
    );

    future.wait().unwrap();
}

#[test]
fn test_future_does_not_interfere_with_thread_local_submit() {
    init();

    let device = Device::metal();

    // Use GpuFuture path.
    let a = DynTensor::full(&[4], 1.0, DType::F32, &device).unwrap();
    let _r = a.add_scalar(1.0).unwrap();
    let future = GpuFuture::submit_current().unwrap().unwrap();

    // Use thread-local submit() path independently.
    let b = DynTensor::full(&[4], 10.0, DType::F32, &device).unwrap();
    let _s = b.mul_scalar(2.0).unwrap();
    gpu_scope::submit().expect("thread-local submit should work after future");

    future.wait().expect("future wait should succeed");
    gpu_scope::sync().expect("thread-local sync should succeed");
}

#[test]
fn test_command_batch_submit_async() {
    init();

    let ctx = crate::context::MetalContext::new().expect("Metal context");
    let batch = ctx.begin_batch().expect("begin_batch");

    // submit_async returns a GpuFuture (even with empty batch).
    let future = batch.submit_async();
    assert!(
        format!("{future:?}").contains("GpuFuture"),
        "submit_async should return a GpuFuture"
    );
    future.wait().expect("future from submit_async should complete");
}

#[test]
fn test_async_gpu_result_debug() {
    init();

    let device = Device::metal();
    let a = DynTensor::full(&[4], 1.0, DType::F32, &device).unwrap();
    let _r = a.add_scalar(1.0).unwrap();

    let future = GpuFuture::submit_current().unwrap().unwrap();
    let async_result = AsyncGpuResult {
        future,
        value: 42u32,
    };

    let debug_str = format!("{async_result:?}");
    assert!(debug_str.contains("AsyncGpuResult"));
    assert!(debug_str.contains("42"));

    async_result.future.wait().unwrap();
}
