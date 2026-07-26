// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`GpuAudioHandle`].

use super::GpuAudioHandle;
use crate::context::MetalContext;

#[test]
fn test_gpu_audio_handle_sample_count() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 8] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let buffer = ctx.create_buffer(&data).expect("create buffer");

    let handle = GpuAudioHandle::new(buffer, 8, 24000);
    assert_eq!(handle.sample_count(), 8);
}

#[test]
fn test_gpu_audio_handle_sample_rate() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [0.0; 4];
    let buffer = ctx.create_buffer(&data).expect("create buffer");

    let handle = GpuAudioHandle::new(buffer, 4, 24000);
    assert_eq!(handle.sample_rate(), 24000);
}

#[test]
fn test_gpu_audio_handle_sample_rate_custom() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [0.0; 4];
    let buffer = ctx.create_buffer(&data).expect("create buffer");

    let handle = GpuAudioHandle::new(buffer, 4, 44100);
    assert_eq!(handle.sample_rate(), 44100);
}

#[test]
fn test_gpu_audio_handle_duration_secs() {
    let ctx = MetalContext::new().expect("Metal device");
    let sample_count = 24000usize;
    let data = vec![0.0f32; sample_count];
    let buffer = ctx.create_buffer(&data).expect("create buffer");

    let handle = GpuAudioHandle::new(buffer, sample_count, 24000);
    let duration = handle.duration_secs();
    assert!(
        (duration - 1.0).abs() < 1e-6,
        "24000 samples at 24kHz should be ~1.0s, got {duration}"
    );
}

#[test]
fn test_gpu_audio_handle_duration_secs_half_second() {
    let ctx = MetalContext::new().expect("Metal device");
    let sample_count = 12000usize;
    let data = vec![0.0f32; sample_count];
    let buffer = ctx.create_buffer(&data).expect("create buffer");

    let handle = GpuAudioHandle::new(buffer, sample_count, 24000);
    let duration = handle.duration_secs();
    assert!(
        (duration - 0.5).abs() < 1e-6,
        "12000 samples at 24kHz should be ~0.5s, got {duration}"
    );
}

#[test]
fn test_gpu_audio_handle_gpu_buffer_access() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let buffer = ctx.create_buffer(&data).expect("create buffer");
    let expected_len = buffer.len();

    let handle = GpuAudioHandle::new(buffer, 4, 24000);
    assert_eq!(handle.gpu_buffer().len(), expected_len);
}

#[test]
fn test_gpu_audio_handle_to_cpu_roundtrip() {
    crate::test_common::init();

    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 8] = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
    let buffer = ctx.create_buffer(&data).expect("create buffer");

    let handle = GpuAudioHandle::new(buffer, 8, 24000);
    let cpu_data = handle.to_cpu().expect("to_cpu should succeed");

    assert_eq!(cpu_data.len(), 8, "sample count mismatch");
    for (i, (got, expected)) in cpu_data.iter().zip(data.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-7,
            "sample [{i}]: got={got}, expected={expected}"
        );
    }
}

#[test]
fn test_gpu_audio_handle_to_cpu_partial_buffer() {
    // Buffer has capacity for 8 f32s but handle only claims 4 samples.
    crate::test_common::init();

    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let buffer = ctx.create_buffer(&data).expect("create buffer");

    let handle = GpuAudioHandle::new(buffer, 4, 24000);
    let cpu_data = handle.to_cpu().expect("to_cpu should succeed");

    assert_eq!(cpu_data.len(), 4, "should only read sample_count elements");
    assert_eq!(&cpu_data, &[1.0, 2.0, 3.0, 4.0]);
}

// ============================================================================
// Fence-backed GpuAudioHandle tests
// ============================================================================

#[test]
fn test_gpu_audio_handle_no_fence_by_default() {
    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [0.0; 4];
    let buffer = ctx.create_buffer(&data).expect("create buffer");

    let handle = GpuAudioHandle::new(buffer, 4, 24000);
    assert!(!handle.has_fence(), "default handle should not have fence");
    assert!(handle.fence().is_none());
    // is_ready returns false when no fence is attached.
    assert!(!handle.is_ready(), "no-fence handle should report not ready");
}

#[test]
fn test_gpu_audio_handle_with_fence() {
    crate::test_common::init();

    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let buffer = ctx.create_buffer(&data).expect("create buffer");

    // Create a fence by submitting an empty batch.
    crate::gpu_scope::get_or_create_batch().unwrap();
    let fence = crate::gpu_fence::GpuFence::submit_current()
        .unwrap()
        .expect("should return fence");

    let handle = GpuAudioHandle::with_fence(buffer, 4, 24000, fence);
    assert!(handle.has_fence(), "with_fence handle should have fence");
    assert!(handle.fence().is_some());

    // Wait for completion, then check is_ready.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !handle.is_ready() {
        assert!(std::time::Instant::now() < deadline, "timed out waiting for fence");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(handle.is_ready(), "should be ready after GPU completion");
}

#[test]
fn test_gpu_audio_handle_with_fence_to_cpu() {
    crate::test_common::init();

    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [10.0, 20.0, 30.0, 40.0];
    let buffer = ctx.create_buffer(&data).expect("create buffer");

    // Submit empty batch to get a fence.
    crate::gpu_scope::get_or_create_batch().unwrap();
    let fence = crate::gpu_fence::GpuFence::submit_current()
        .unwrap()
        .expect("should return fence");

    let handle = GpuAudioHandle::with_fence(buffer, 4, 24000, fence);
    let cpu_data = handle.to_cpu().expect("to_cpu with fence should succeed");

    assert_eq!(cpu_data.len(), 4);
    assert_eq!(&cpu_data, &[10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_gpu_audio_handle_fence_elapsed() {
    crate::test_common::init();

    let ctx = MetalContext::new().expect("Metal device");
    let data: [f32; 4] = [0.0; 4];
    let buffer = ctx.create_buffer(&data).expect("create buffer");

    crate::gpu_scope::get_or_create_batch().unwrap();
    let fence = crate::gpu_fence::GpuFence::submit_current()
        .unwrap()
        .expect("should return fence");

    let handle = GpuAudioHandle::with_fence(buffer, 4, 24000, fence);
    // Elapsed should return a valid Duration (non-panicking).
    let _elapsed = handle.fence().unwrap().elapsed();

    // Clean up: to_cpu waits on fence.
    let _ = handle.to_cpu();
}
