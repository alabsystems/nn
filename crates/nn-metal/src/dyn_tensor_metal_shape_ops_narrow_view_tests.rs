#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for zero-copy GPU narrow contiguous views (#1945, #2007).
//!
//! These tests verify that `gpu_narrow()` returns a zero-copy buffer view
//! (sharing the parent buffer with a byte offset) for contiguous narrows, and
//! that the view is correctly materialized when used in GPU dispatch operations.
//!
//! Contiguous narrow: dim-0 (always), dim-1 when shape[0]==1, etc.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

use crate::test_common::{assert_close, init};

/// Chained dim-0 narrow: narrow a view again. Tests byte offset composition.
#[test]
fn test_gpu_narrow_dim0_chained_view() {
    init();
    // [6, 4] -> narrow(0, 1, 4) -> [4, 4] -> narrow(0, 1, 2) -> [2, 4]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[6, 4], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let view1 = gpu.narrow(0, 1, 4).unwrap(); // rows 1..5
    let view2 = view1.narrow(0, 1, 2).unwrap(); // rows 2..4 of original

    let cpu_result = cpu.narrow(0, 2, 2).unwrap(); // direct: rows 2..4
    let gpu_vals = view2
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "narrow_dim0_chained");
}

/// Zero-copy view used in GPU binary op (add). Tests dispatch_def byte_offset.
#[test]
fn test_gpu_narrow_dim0_view_in_add() {
    init();
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[3, 4], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    // Narrow to rows 1..3 (zero-copy view), then add it to itself.
    let view = gpu.narrow(0, 1, 2).unwrap();
    let gpu_sum = (&view + &view).unwrap();

    let cpu_view = cpu.narrow(0, 1, 2).unwrap();
    let cpu_sum = (&cpu_view + &cpu_view).unwrap();

    let gpu_vals = gpu_sum
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_sum.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "narrow_view_add");
}

/// Sequential narrow(0, t, 1) loop — LSTM timestep pattern.
#[test]
fn test_gpu_narrow_dim0_sequential_lstm_pattern() {
    init();
    // [seq_len=5, batch=2, hidden=3]
    let data: Vec<f32> = (0..30).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[5, 2, 3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    for t in 0..5 {
        let gpu_step = gpu.narrow(0, t, 1).unwrap();
        let cpu_step = cpu.narrow(0, t, 1).unwrap();
        assert_eq!(gpu_step.dims(), &[1, 2, 3]);
        let gpu_vals = gpu_step
            .to_device(&Device::Cpu)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap();
        let cpu_vals = cpu_step.to_flat_vec::<f32>().unwrap();
        assert_close(&gpu_vals, &cpu_vals, 0.0, &format!("lstm_step_{t}"));
    }
}

/// Narrow view used in matmul — tests offset propagation through GEMM dispatch.
#[test]
fn test_gpu_narrow_dim0_view_in_matmul() {
    init();
    // [4, 3] narrow(0, 1, 2) -> [2, 3] matmul [3, 2] -> [2, 2]
    let data: Vec<f32> = (0..12).map(|i| (i + 1) as f32).collect();
    let w_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let cpu = DynTensor::new(&data, &[4, 3], &Device::Cpu).unwrap();
    let w_cpu = DynTensor::new(&w_data, &[3, 2], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();
    let w_gpu = w_cpu.to_device(&Device::metal()).unwrap();

    let view = gpu.narrow(0, 1, 2).unwrap();
    let gpu_result = view.matmul(&w_gpu).unwrap();

    let cpu_view = cpu.narrow(0, 1, 2).unwrap();
    let cpu_result = cpu_view.matmul(&w_cpu).unwrap();

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-5, "narrow_view_matmul");
}

// ===== #1964 regression tests: narrow-view byte_offset through GPU ops =====
//
// These tests verify that GPU dispatch correctly reads from the narrow view's
// byte_offset, not from the start of the parent buffer.

/// Narrow view used in scatter_add — tests byte_offset in atomic scatter kernel.
#[test]
fn test_gpu_narrow_dim0_view_in_scatter_add() {
    init();
    // [4, 3]: rows 0..4. Narrow to rows 1..3 -> [2, 3] as scatter target.
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[4, 3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let gpu_view = gpu.narrow(0, 1, 2).unwrap(); // [2, 3]
    let cpu_view = cpu.narrow(0, 1, 2).unwrap();

    // index: scatter src row 0 → target row 1, src row 1 → target row 0
    let idx_data: Vec<u32> = vec![1, 1, 1, 0, 0, 0];
    let idx = DynTensor::from_vec_u32(idx_data, &[2, 3], &Device::Cpu).unwrap();
    let idx_gpu = idx.to_device(&Device::metal()).unwrap();

    // src: small values to scatter-add
    let src_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let src = DynTensor::new(&src_data, &[2, 3], &Device::Cpu).unwrap();
    let src_gpu = src.to_device(&Device::metal()).unwrap();

    let gpu_result = gpu_view.scatter_add(0, &idx_gpu, &src_gpu).unwrap();
    let cpu_result = cpu_view.scatter_add(0, &idx, &src).unwrap();

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-5, "narrow_view_scatter_add");
}

/// Narrow view used in index_add — tests byte_offset in atomic index_add kernel.
#[test]
fn test_gpu_narrow_dim0_view_in_index_add() {
    init();
    // [4, 3] narrow to rows 1..3 -> [2, 3] as index_add target.
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[4, 3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let gpu_view = gpu.narrow(0, 1, 2).unwrap();
    let cpu_view = cpu.narrow(0, 1, 2).unwrap();

    // 1D index: map src row 0 → target row 1, src row 1 → target row 0
    let idx = DynTensor::from_vec_u32(vec![1u32, 0], &[2], &Device::Cpu).unwrap();
    let idx_gpu = idx.to_device(&Device::metal()).unwrap();

    let src_data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    let src = DynTensor::new(&src_data, &[2, 3], &Device::Cpu).unwrap();
    let src_gpu = src.to_device(&Device::metal()).unwrap();

    let gpu_result = gpu_view.index_add(0, &idx_gpu, &src_gpu).unwrap();
    let cpu_result = cpu_view.index_add(0, &idx, &src).unwrap();

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-5, "narrow_view_index_add");
}

/// Narrow view used in argmax — tests byte_offset in argreduce kernel.
#[test]
fn test_gpu_narrow_dim0_view_in_argmax() {
    init();
    // [4, 5] narrow to rows 1..3 -> [2, 5], argmax along dim 1.
    let data: Vec<f32> = (0..20).map(|i| (i as f32) * 0.7 - 5.0).collect();
    let cpu = DynTensor::new(&data, &[4, 5], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let gpu_view = gpu.narrow(0, 1, 2).unwrap();
    let cpu_view = cpu.narrow(0, 1, 2).unwrap();

    let gpu_result = gpu_view.argmax(1).unwrap();
    let cpu_result = cpu_view.argmax(1).unwrap();

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::U32)
        .unwrap()
        .to_flat_vec::<u32>()
        .unwrap();
    let cpu_vals = cpu_result
        .to_dtype(DType::U32)
        .unwrap()
        .to_flat_vec::<u32>()
        .unwrap();
    assert_eq!(gpu_vals, cpu_vals, "narrow_view_argmax mismatch");
}

/// Narrow view used in argmin — tests byte_offset in argreduce kernel.
#[test]
fn test_gpu_narrow_dim0_view_in_argmin() {
    init();
    let data: Vec<f32> = (0..20).map(|i| (i as f32) * 0.7 - 5.0).collect();
    let cpu = DynTensor::new(&data, &[4, 5], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let gpu_view = gpu.narrow(0, 1, 2).unwrap();
    let cpu_view = cpu.narrow(0, 1, 2).unwrap();

    let gpu_result = gpu_view.argmin(1).unwrap();
    let cpu_result = cpu_view.argmin(1).unwrap();

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::U32)
        .unwrap()
        .to_flat_vec::<u32>()
        .unwrap();
    let cpu_vals = cpu_result
        .to_dtype(DType::U32)
        .unwrap()
        .to_flat_vec::<u32>()
        .unwrap();
    assert_eq!(gpu_vals, cpu_vals, "narrow_view_argmin mismatch");
}

/// Narrow view used in topk — tests byte_offset in topk kernel.
#[test]
fn test_gpu_narrow_dim0_view_in_topk() {
    init();
    // [4, 6] narrow to rows 1..3 -> [2, 6], topk(k=3) along last dim.
    let data: Vec<f32> = (0..24).map(|i| (i as f32) * 1.1 - 10.0).collect();
    let cpu = DynTensor::new(&data, &[4, 6], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let gpu_view = gpu.narrow(0, 1, 2).unwrap();
    let cpu_view = cpu.narrow(0, 1, 2).unwrap();

    let (gpu_vals, gpu_idxs) = gpu_view.topk(1, 3).unwrap();
    let (cpu_vals, cpu_idxs) = cpu_view.topk(1, 3).unwrap();

    let gv = gpu_vals
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cv = cpu_vals.to_flat_vec::<f32>().unwrap();
    assert_close(&gv, &cv, 1e-5, "narrow_view_topk_vals");

    let gi = gpu_idxs
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::U32)
        .unwrap()
        .to_flat_vec::<u32>()
        .unwrap();
    let ci = cpu_idxs
        .to_dtype(DType::U32)
        .unwrap()
        .to_flat_vec::<u32>()
        .unwrap();
    assert_eq!(gi, ci, "narrow_view_topk_idxs mismatch");
}

/// Narrow view used in cumsum — tests byte_offset in Blelloch prefix scan kernel.
#[test]
fn test_gpu_narrow_dim0_view_in_cumsum() {
    init();
    // [4, 5] narrow to rows 1..3 -> [2, 5], cumsum along dim 1.
    let data: Vec<f32> = (0..20).map(|i| (i + 1) as f32).collect();
    let cpu = DynTensor::new(&data, &[4, 5], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let gpu_view = gpu.narrow(0, 1, 2).unwrap();
    let cpu_view = cpu.narrow(0, 1, 2).unwrap();

    let gpu_result = gpu_view.cumsum(1).unwrap();
    let cpu_result = cpu_view.cumsum(1).unwrap();

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 1e-5, "narrow_view_cumsum");
}

/// Verify zero-copy buffer sharing: dim-0 narrow returns a view that shares
/// the parent's Metal buffer (same allocation size) with a non-zero byte offset.
///
/// This is the Prover's P1-140 concern: tests above verify correctness but not
/// the zero-copy property. This test verifies:
/// 1. Parent buffer size == child buffer size (alias shares allocation).
/// 2. Child byte_offset > 0 (adjusted for start position).
/// 3. Chained views accumulate byte_offset correctly.
#[test]
fn test_gpu_narrow_dim0_zero_copy_buffer_sharing() {
    init();
    use super::MetalTensorData;

    // Parent: [6, 4] f32 = 96 bytes.
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[6, 4], &Device::Cpu).unwrap();
    let parent = cpu.to_device(&Device::metal()).unwrap();
    let parent_data = parent.gpu_data::<MetalTensorData>().unwrap();

    assert_eq!(parent_data.byte_offset(), 0, "parent should have offset 0");
    let parent_buf_len = parent_data.buffer().len();
    assert_eq!(parent_buf_len, 96, "parent buffer = 6*4*4 bytes");

    // View 1: narrow(0, 2, 3) -> rows 2..5, shape [3, 4].
    let view1 = parent.narrow(0, 2, 3).unwrap();
    assert_eq!(view1.dims(), &[3, 4]);
    let view1_data = view1.gpu_data::<MetalTensorData>().unwrap();

    // Zero-copy: same underlying buffer (same allocation size).
    assert_eq!(
        view1_data.buffer().len(),
        parent_buf_len,
        "view should share parent's buffer allocation"
    );
    // Byte offset = start * stride_0 * elem_bytes = 2 * 4 * 4 = 32.
    assert_eq!(
        view1_data.byte_offset(),
        32,
        "view byte_offset should be 2 * 4 elements * 4 bytes"
    );

    // View 2: chained narrow(0, 1, 1) on view1 -> row 3 of parent, shape [1, 4].
    let view2 = view1.narrow(0, 1, 1).unwrap();
    assert_eq!(view2.dims(), &[1, 4]);
    let view2_data = view2.gpu_data::<MetalTensorData>().unwrap();

    // Chained: same parent buffer.
    assert_eq!(
        view2_data.buffer().len(),
        parent_buf_len,
        "chained view should share same buffer"
    );
    // Composed offset: parent(0) + view1(32) + 1*4*4 = 48.
    assert_eq!(
        view2_data.byte_offset(),
        48,
        "chained view byte_offset should compose: 32 + 1*4*4 = 48"
    );

    // Verify the view data is correct (row 3 of parent = [12, 13, 14, 15]).
    let vals = view2
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_eq!(vals, vec![12.0, 13.0, 14.0, 15.0]);
}

// Non-dim-0 contiguous narrow tests extracted to
// dyn_tensor_metal_shape_ops_narrow_view_tests_nondim0.rs (#2017).
#[path = "dyn_tensor_metal_shape_ops_narrow_view_tests_nondim0.rs"]
mod nondim0;
