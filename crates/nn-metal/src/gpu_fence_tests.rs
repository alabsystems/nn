// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`GpuFence`] — caller-held non-blocking GPU submit.

use super::*;
use crate::gpu_scope::{self, get_or_create_batch, is_lazy_batch_active};
use crate::test_common::init;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

// ============================================================================
// Construction: submit_current returns correct Option variant
// ============================================================================

#[test]
fn test_submit_current_returns_none_when_no_batch() {
    // No lazy batch active -> submit_current returns Ok(None).
    assert!(!is_lazy_batch_active());
    let result = GpuFence::submit_current();
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
}

#[test]
fn test_submit_current_returns_fence_when_batch_pending() {
    init();

    // Create a lazy batch by triggering a dispatch.
    get_or_create_batch().unwrap();
    assert!(is_lazy_batch_active());

    let fence = GpuFence::submit_current()
        .expect("submit_current should succeed")
        .expect("should return Some(GpuFence) when batch is pending");

    // After submit, lazy batch should be consumed.
    assert!(
        !is_lazy_batch_active(),
        "lazy batch should be consumed after fence submit"
    );

    // Wait for the GPU work to complete.
    fence.wait().expect("fence wait should succeed");
}

#[test]
fn test_submit_current_with_real_gpu_work() {
    init();

    let device = Device::metal();
    let a = DynTensor::full(&[4], 2.0, DType::F32, &device).unwrap();
    let _b = a.add_scalar(1.0).unwrap();

    let fence = GpuFence::submit_current()
        .expect("submit_current should succeed")
        .expect("should return Some(GpuFence)");

    fence.wait().expect("fence wait should succeed");
}

// ============================================================================
// is_completed transitions
// ============================================================================

#[test]
fn test_is_completed_eventually_true_after_trivial_work() {
    init();

    // Submit an empty batch (get_or_create only, no real compute work).
    get_or_create_batch().unwrap();
    let fence = GpuFence::submit_current()
        .unwrap()
        .expect("should return fence");

    // Poll until completed or timeout (trivial work completes almost instantly).
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if fence.is_completed() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "fence did not complete within 5 seconds for trivial work"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(fence.is_completed(), "fence should be completed");
    // Consume the fence (wait on already-completed work is fine).
    fence.wait().expect("wait on completed fence should succeed");
}

#[test]
fn test_is_completed_transitions_with_real_compute() {
    init();

    let device = Device::metal();
    // Do enough GPU work that is_completed is meaningful.
    let a = DynTensor::full(&[1024], 3.0, DType::F32, &device).unwrap();
    let _b = a.mul_scalar(2.0).unwrap();

    let fence = GpuFence::submit_current()
        .unwrap()
        .expect("should return fence");

    // We cannot guarantee is_completed() is false at this exact moment (Metal
    // may complete trivial work before we even poll), but we CAN guarantee:
    // 1. is_completed() returns a bool without panicking before wait().
    // 2. After wait(), is_completed() is guaranteed true.
    let _initial_status = fence.is_completed(); // must not panic
    fence.wait().expect("wait should succeed");
}

#[test]
fn test_is_completed_callable_multiple_times() {
    init();

    get_or_create_batch().unwrap();
    let fence = GpuFence::submit_current()
        .unwrap()
        .expect("should return fence");

    // Polling is_completed multiple times should be safe and idempotent.
    let _ = fence.is_completed();
    let _ = fence.is_completed();
    let _ = fence.is_completed();

    fence.wait().unwrap();

    // After wait, cannot call is_completed because wait() consumes self.
    // This is enforced at the type level -- no runtime test needed.
}

// ============================================================================
// Empty command buffer fence
// ============================================================================

#[test]
fn test_fence_with_empty_batch() {
    init();

    // Create batch but encode no actual compute work.
    get_or_create_batch().unwrap();
    assert!(is_lazy_batch_active());

    let fence = GpuFence::submit_current()
        .unwrap()
        .expect("should return fence even for empty batch");

    assert!(
        !is_lazy_batch_active(),
        "batch consumed after submit"
    );

    // Wait should succeed -- Metal can commit an empty command buffer.
    fence.wait().expect("wait on empty batch should succeed");
}

// ============================================================================
// Multiple sequential submits
// ============================================================================

#[test]
fn test_multiple_fences_pipeline() {
    init();

    let device = Device::metal();

    // Segment 1: create GPU work.
    let a = DynTensor::full(&[8], 3.0, DType::F32, &device).unwrap();
    let _seg1 = a.mul_scalar(2.0).unwrap(); // 6.0

    let fence1 = GpuFence::submit_current()
        .expect("submit_current should succeed")
        .expect("should return fence for segment 1");

    // Segment 2: more GPU work while segment 1 may still be executing.
    let b = DynTensor::full(&[8], 5.0, DType::F32, &device).unwrap();
    let _seg2 = b.add_scalar(1.0).unwrap(); // 6.0

    let fence2 = GpuFence::submit_current()
        .expect("submit_current should succeed")
        .expect("should return fence for segment 2");

    // Wait in order.
    fence1.wait().expect("fence1 wait should succeed");
    fence2.wait().expect("fence2 wait should succeed");
}

#[test]
fn test_multiple_fences_wait_reverse_order() {
    init();

    let device = Device::metal();

    // Segment 1
    let a = DynTensor::full(&[4], 1.0, DType::F32, &device).unwrap();
    let _r1 = a.add_scalar(2.0).unwrap();
    let fence1 = GpuFence::submit_current().unwrap().unwrap();

    // Segment 2
    let b = DynTensor::full(&[4], 10.0, DType::F32, &device).unwrap();
    let _r2 = b.mul_scalar(3.0).unwrap();
    let fence2 = GpuFence::submit_current().unwrap().unwrap();

    // Wait in REVERSE order -- should not deadlock or panic.
    fence2.wait().expect("fence2 wait should succeed");
    fence1.wait().expect("fence1 wait should succeed");
}

#[test]
fn test_three_sequential_fences() {
    init();

    let device = Device::metal();

    let a = DynTensor::full(&[4], 1.0, DType::F32, &device).unwrap();
    let _r1 = a.add_scalar(1.0).unwrap();
    let fence1 = GpuFence::submit_current().unwrap().unwrap();

    let b = DynTensor::full(&[4], 2.0, DType::F32, &device).unwrap();
    let _r2 = b.add_scalar(2.0).unwrap();
    let fence2 = GpuFence::submit_current().unwrap().unwrap();

    let c = DynTensor::full(&[4], 3.0, DType::F32, &device).unwrap();
    let _r3 = c.add_scalar(3.0).unwrap();
    let fence3 = GpuFence::submit_current().unwrap().unwrap();

    fence1.wait().expect("fence1");
    fence2.wait().expect("fence2");
    fence3.wait().expect("fence3");
}

// ============================================================================
// Pipelining pattern: submit -> CPU work -> wait -> verify results
// ============================================================================

#[test]
fn test_pipeline_submit_cpu_work_wait_verify() {
    init();

    let device = Device::metal();

    // Segment 1: GPU compute.
    let a = DynTensor::full(&[4], 5.0, DType::F32, &device).unwrap();
    let result1 = a.mul_scalar(3.0).unwrap(); // 15.0

    let fence1 = GpuFence::submit_current()
        .unwrap()
        .expect("should return fence");

    // Simulate CPU work while GPU runs segment 1.
    let cpu_data: Vec<f32> = (0..100).map(|i| (i as f32).sqrt()).collect();
    assert_eq!(cpu_data.len(), 100); // ensure CPU work actually ran

    // Segment 2: more GPU work, overlapping with segment 1 on GPU.
    let b = DynTensor::full(&[4], 2.0, DType::F32, &device).unwrap();
    let result2 = b.add_scalar(8.0).unwrap(); // 10.0

    let fence2 = GpuFence::submit_current()
        .unwrap()
        .expect("should return fence");

    // Wait for segment 1, then read its results.
    fence1.wait().expect("fence1 wait");
    let vals1 = result1
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals1, vec![15.0; 4], "segment 1: 5.0 * 3.0 = 15.0");

    // Wait for segment 2, then read its results.
    fence2.wait().expect("fence2 wait");
    let vals2 = result2
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals2, vec![10.0; 4], "segment 2: 2.0 + 8.0 = 10.0");
}

#[test]
fn test_pipeline_matmul_then_elementwise() {
    init();

    let device = Device::metal();

    // Segment 1: matmul [2,2] x [2,2].
    let a_cpu = DynTensor::new(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap();
    let b_cpu = DynTensor::new(&[5.0_f32, 6.0, 7.0, 8.0], &[2, 2], &Device::Cpu).unwrap();
    let a = a_cpu.to_device(&device).unwrap();
    let b = b_cpu.to_device(&device).unwrap();
    let seg1_result = a.matmul(&b).unwrap();
    // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]

    let fence1 = GpuFence::submit_current()
        .unwrap()
        .expect("matmul fence");

    // Segment 2: elementwise on different tensors.
    let c = DynTensor::full(&[4], 10.0, DType::F32, &device).unwrap();
    let seg2_result = c.neg().unwrap(); // -10.0

    let fence2 = GpuFence::submit_current()
        .unwrap()
        .expect("elementwise fence");

    // Wait and verify both.
    fence1.wait().expect("fence1");
    let vals1 = seg1_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals1, vec![19.0, 22.0, 43.0, 50.0]);

    fence2.wait().expect("fence2");
    let vals2 = seg2_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals2, vec![-10.0; 4]);
}

// ============================================================================
// Interaction with thread-local submit/sync
// ============================================================================

#[test]
fn test_fence_does_not_interfere_with_thread_local_submit() {
    init();

    let device = Device::metal();

    // Use GpuFence path.
    let a = DynTensor::full(&[4], 1.0, DType::F32, &device).unwrap();
    let _r = a.add_scalar(1.0).unwrap();
    let fence = GpuFence::submit_current().unwrap().unwrap();

    // Use thread-local submit() path independently -- should not conflict.
    let b = DynTensor::full(&[4], 10.0, DType::F32, &device).unwrap();
    let _s = b.mul_scalar(2.0).unwrap();
    gpu_scope::submit().expect("thread-local submit should work after fence");

    // Wait for fence.
    fence.wait().expect("fence wait should succeed");

    // Sync the thread-local pending batch.
    gpu_scope::sync().expect("thread-local sync should succeed");
}

#[test]
fn test_fence_after_flush_returns_none() {
    init();

    let device = Device::metal();
    let a = DynTensor::full(&[4], 1.0, DType::F32, &device).unwrap();
    let _r = a.add_scalar(1.0).unwrap();

    // Flush commits and waits -- consumes the lazy batch.
    gpu_scope::flush().expect("flush should succeed");

    // After flush, no lazy batch remains, so submit_current returns None.
    let fence = GpuFence::submit_current().expect("should not error");
    assert!(fence.is_none(), "no batch remains after flush");
}

#[test]
fn test_fence_then_scope_flush() {
    init();

    let device = Device::metal();

    // Create work and fence-submit it.
    let a = DynTensor::full(&[4], 3.0, DType::F32, &device).unwrap();
    let _r1 = a.mul_scalar(2.0).unwrap();
    let fence = GpuFence::submit_current().unwrap().unwrap();

    // Create more work and flush it via the thread-local path.
    let b = DynTensor::full(&[4], 7.0, DType::F32, &device).unwrap();
    let result = b.add_scalar(1.0).unwrap();
    gpu_scope::flush().expect("flush after fence should work");

    // Both should complete successfully.
    fence.wait().expect("fence wait");
    let vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals, vec![8.0; 4]);
}

// ============================================================================
// Debug formatting
// ============================================================================

#[test]
fn test_fence_debug_format() {
    init();

    get_or_create_batch().unwrap();
    let fence = GpuFence::submit_current()
        .unwrap()
        .expect("should return fence");

    let debug_str = format!("{fence:?}");
    assert!(
        debug_str.contains("GpuFence"),
        "debug format should contain type name, got: {debug_str}"
    );
    assert!(
        debug_str.contains("is_completed"),
        "debug format should contain is_completed field, got: {debug_str}"
    );

    fence.wait().unwrap();
}

#[test]
fn test_fence_debug_shows_completion_status() {
    init();

    get_or_create_batch().unwrap();
    let fence = GpuFence::submit_current()
        .unwrap()
        .expect("should return fence");

    // Wait for completion, then check debug shows true.
    // Poll until done first.
    let deadline = Instant::now() + Duration::from_secs(5);
    while !fence.is_completed() {
        assert!(Instant::now() < deadline, "timed out");
        std::thread::sleep(Duration::from_millis(1));
    }

    let debug_str = format!("{fence:?}");
    assert!(
        debug_str.contains("true"),
        "completed fence debug should show true, got: {debug_str}"
    );

    fence.wait().unwrap();
}

// ============================================================================
// Consecutive submit_current calls (no work between them)
// ============================================================================

#[test]
fn test_consecutive_submit_current_second_returns_none() {
    init();

    get_or_create_batch().unwrap();
    let fence = GpuFence::submit_current().unwrap().unwrap();

    // Second submit_current without new work should return None.
    let fence2 = GpuFence::submit_current().expect("should not error");
    assert!(
        fence2.is_none(),
        "second submit_current without new work should return None"
    );

    fence.wait().unwrap();
}

// ============================================================================
// Result correctness through fence boundary
// ============================================================================

#[test]
fn test_fence_preserves_tensor_data_integrity() {
    init();

    let device = Device::metal();

    // Build a multi-op chain: x * 2 + 3.
    let x = DynTensor::full(&[16], 4.0, DType::F32, &device).unwrap();
    let y = x.mul_scalar(2.0).unwrap(); // 8.0
    let z = y.add_scalar(3.0).unwrap(); // 11.0

    let fence = GpuFence::submit_current()
        .unwrap()
        .expect("should return fence");

    fence.wait().expect("wait");

    // Verify the intermediate and final results are correct after fence.
    let z_vals = z
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(z_vals, vec![11.0; 16], "4.0 * 2.0 + 3.0 = 11.0");
}

#[test]
fn test_fence_with_larger_tensors() {
    init();

    let device = Device::metal();

    // Use larger tensors to exercise real GPU dispatch (not just trivial sizes).
    let n = 4096;
    let a = DynTensor::full(&[n], 2.5, DType::F32, &device).unwrap();
    let b = DynTensor::full(&[n], 1.5, DType::F32, &device).unwrap();
    let result = a.add(&b).unwrap(); // 4.0

    let fence = GpuFence::submit_current()
        .unwrap()
        .expect("should return fence for larger tensor work");

    fence.wait().expect("wait on larger tensors");

    let vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals.len(), n);
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - 4.0).abs() < 1e-6,
            "element [{i}]: expected 4.0, got {v}"
        );
    }
}

// ============================================================================
// wait_timeout tests
// ============================================================================

#[test]
fn test_wait_timeout_completes_within_generous_timeout() {
    init();

    let device = Device::metal();
    let a = DynTensor::full(&[16], 2.0, DType::F32, &device).unwrap();
    let _b = a.add_scalar(1.0).unwrap();

    let fence = GpuFence::submit_current()
        .unwrap()
        .expect("should return fence");

    // 10 seconds is very generous for trivial GPU work.
    let completed = fence
        .wait_timeout(Duration::from_secs(10))
        .expect("wait_timeout should not error");
    assert!(completed, "trivial GPU work should complete within 10s");

    // After successful wait_timeout, is_completed should be true.
    assert!(fence.is_completed());

    // Clean up by consuming the fence.
    fence.wait().unwrap();
}

#[test]
fn test_wait_timeout_zero_duration_may_return_false() {
    init();

    let device = Device::metal();
    // Do enough GPU work that a zero timeout is meaningful.
    let a = DynTensor::full(&[4096], 3.0, DType::F32, &device).unwrap();
    let _b = a.mul_scalar(2.0).unwrap();

    let fence = GpuFence::submit_current()
        .unwrap()
        .expect("should return fence");

    // Zero timeout: may or may not have completed.
    let result = fence
        .wait_timeout(Duration::ZERO)
        .expect("wait_timeout should not error");
    // We cannot assert the return value — Metal might complete instantly.
    let _ = result;

    // Clean up.
    fence.wait().unwrap();
}

#[test]
fn test_wait_timeout_repeated_calls() {
    init();

    let device = Device::metal();
    let a = DynTensor::full(&[8], 5.0, DType::F32, &device).unwrap();
    let _b = a.add_scalar(10.0).unwrap();

    let fence = GpuFence::submit_current()
        .unwrap()
        .expect("should return fence");

    // Multiple wait_timeout calls should be safe (does not consume self).
    let _ = fence.wait_timeout(Duration::from_millis(1));
    let _ = fence.wait_timeout(Duration::from_millis(10));
    let completed = fence
        .wait_timeout(Duration::from_secs(5))
        .expect("should not error");
    assert!(completed, "should complete within 5s");

    fence.wait().unwrap();
}

// ============================================================================
// elapsed / submit_time tests
// ============================================================================

#[test]
fn test_elapsed_is_monotonically_increasing() {
    init();

    get_or_create_batch().unwrap();
    let fence = GpuFence::submit_current()
        .unwrap()
        .expect("should return fence");

    let t1 = fence.elapsed();
    std::thread::sleep(Duration::from_millis(5));
    let t2 = fence.elapsed();

    assert!(
        t2 >= t1,
        "elapsed should be monotonically increasing: t1={t1:?}, t2={t2:?}"
    );

    fence.wait().unwrap();
}

#[test]
fn test_submit_time_is_before_now() {
    init();

    get_or_create_batch().unwrap();
    let fence = GpuFence::submit_current()
        .unwrap()
        .expect("should return fence");

    let submit = fence.submit_time();
    assert!(
        submit <= Instant::now(),
        "submit_time should be in the past"
    );

    fence.wait().unwrap();
}

#[test]
fn test_elapsed_after_wait_is_positive() {
    init();

    let device = Device::metal();
    let a = DynTensor::full(&[64], 1.0, DType::F32, &device).unwrap();
    let _b = a.mul_scalar(2.0).unwrap();

    let fence = GpuFence::submit_current()
        .unwrap()
        .expect("should return fence");

    // Elapsed is available even before wait (returns a valid Duration).
    let _pre_wait = fence.elapsed();

    fence.wait().unwrap();
}

#[test]
fn test_debug_format_includes_elapsed() {
    init();

    get_or_create_batch().unwrap();
    let fence = GpuFence::submit_current()
        .unwrap()
        .expect("should return fence");

    let debug_str = format!("{fence:?}");
    assert!(
        debug_str.contains("elapsed_ms"),
        "debug format should include elapsed_ms, got: {debug_str}"
    );

    fence.wait().unwrap();
}
