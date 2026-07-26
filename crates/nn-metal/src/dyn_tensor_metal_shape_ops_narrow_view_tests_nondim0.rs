#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Non-dim-0 contiguous narrow zero-copy view tests (#2007).
//!
//! These tests verify that `gpu_narrow()` takes the zero-copy view path
//! for non-dim-0 narrows when all leading dimensions are 1.
//!
//! Extracted from `dyn_tensor_metal_shape_ops_narrow_view_tests.rs` (#2017).

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::{assert_close, init};

/// dim-1 narrow on [1, N, D] — the LSTM batch=1 case.
/// shape[0]=1, so narrow along dim=1 is contiguous.
#[test]
fn test_gpu_narrow_dim1_contiguous_view() {
    init();
    // [1, 6, 4] -> narrow(1, 2, 3) -> [1, 3, 4]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 6, 4], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let gpu_view = gpu.narrow(1, 2, 3).unwrap();
    let cpu_view = cpu.narrow(1, 2, 3).unwrap();

    assert_eq!(gpu_view.dims(), &[1, 3, 4]);
    let gpu_vals = gpu_view
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_view.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "narrow_dim1_contiguous");
}

/// Verify zero-copy property for dim-1 narrow on [1, N, D].
#[test]
fn test_gpu_narrow_dim1_zero_copy_buffer_sharing() {
    init();
    use crate::MetalTensorData;

    // [1, 8, 4] f32 = 128 bytes.
    let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 8, 4], &Device::Cpu).unwrap();
    let parent = cpu.to_device(&Device::metal()).unwrap();
    let parent_data = parent.gpu_data::<MetalTensorData>().unwrap();

    assert_eq!(parent_data.byte_offset(), 0);
    let parent_buf_len = parent_data.buffer().len();
    assert_eq!(parent_buf_len, 128, "parent buffer = 1*8*4*4 bytes");

    // narrow(1, 3, 2) -> [1, 2, 4]. stride_1 = product(shape[2..]) = 4.
    // byte_offset = 3 * 4 * 4 = 48.
    let view = parent.narrow(1, 3, 2).unwrap();
    assert_eq!(view.dims(), &[1, 2, 4]);
    let view_data = view.gpu_data::<MetalTensorData>().unwrap();

    assert_eq!(
        view_data.buffer().len(),
        parent_buf_len,
        "dim-1 view should share parent's buffer allocation"
    );
    assert_eq!(
        view_data.byte_offset(),
        48,
        "byte_offset should be 3 * 4 * 4 = 48"
    );

    // Verify data correctness: elements 12..20.
    let vals = view
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let expected: Vec<f32> = (12..20).map(|i| i as f32).collect();
    assert_eq!(vals, expected);
}

/// dim-1 narrow on [1, N, D] used in matmul — LSTM gate splitting pattern.
/// gates [1, 4*H] -> narrow(1, 0, H), narrow(1, H, H), etc.
#[test]
fn test_gpu_narrow_dim1_contiguous_in_matmul() {
    init();
    let hidden_size = 4;
    // gates: [1, 16] (4*hidden_size)
    let data: Vec<f32> = (0..16).map(|i| (i + 1) as f32 * 0.1).collect();
    let cpu = DynTensor::new(&data, &[1, 16], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    // Simulate LSTM gate split: narrow each gate from [1, 16]
    for gate in 0..4 {
        let start = gate * hidden_size;
        let gpu_gate = gpu.narrow(1, start, hidden_size).unwrap();
        let cpu_gate = cpu.narrow(1, start, hidden_size).unwrap();
        assert_eq!(gpu_gate.dims(), &[1, hidden_size]);

        let gpu_vals = gpu_gate
            .to_device(&Device::Cpu)
            .unwrap()
            .to_flat_vec::<f32>()
            .unwrap();
        let cpu_vals = cpu_gate.to_flat_vec::<f32>().unwrap();
        assert_close(&gpu_vals, &cpu_vals, 0.0, &format!("gate_{gate}"));
    }
}

/// dim-1 narrow on [1, 2, H] — LSTM h/c split pattern (batch=1).
/// This is the exact pattern from gpu_lstm_cell for batch==1:
/// reshape [1, 2, H] -> [2, H], then dim-0 narrow.
/// With #2007, can also do dim-1 narrow directly on [1, 2, H].
#[test]
fn test_gpu_narrow_dim1_lstm_hc_split() {
    init();
    let hidden_size = 8;
    // Simulate LSTM output: [1, 2, 8] (h stacked with c)
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 2, hidden_size], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    // Split h_new and c_new via dim-1 narrow (now zero-copy with #2007)
    let h_gpu = gpu.narrow(1, 0, 1).unwrap();
    let c_gpu = gpu.narrow(1, 1, 1).unwrap();
    let h_cpu = cpu.narrow(1, 0, 1).unwrap();
    let c_cpu = cpu.narrow(1, 1, 1).unwrap();

    assert_eq!(h_gpu.dims(), &[1, 1, hidden_size]);
    assert_eq!(c_gpu.dims(), &[1, 1, hidden_size]);

    let h_vals = h_gpu
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let c_vals = c_gpu
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(
        &h_vals,
        &h_cpu.to_flat_vec::<f32>().unwrap(),
        0.0,
        "lstm_h_split",
    );
    assert_close(
        &c_vals,
        &c_cpu.to_flat_vec::<f32>().unwrap(),
        0.0,
        "lstm_c_split",
    );
}

/// dim-2 narrow on [1, 1, T, D] — contiguous when shape[0]=shape[1]=1.
#[test]
fn test_gpu_narrow_dim2_contiguous_view() {
    init();
    // [1, 1, 6, 3] -> narrow(2, 1, 3) -> [1, 1, 3, 3]
    let data: Vec<f32> = (0..18).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 1, 6, 3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let gpu_view = gpu.narrow(2, 1, 3).unwrap();
    let cpu_view = cpu.narrow(2, 1, 3).unwrap();

    assert_eq!(gpu_view.dims(), &[1, 1, 3, 3]);
    let gpu_vals = gpu_view
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_view.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "narrow_dim2_contiguous");
}

/// Chained non-dim-0 contiguous narrows: compose byte offsets.
#[test]
fn test_gpu_narrow_dim1_chained_contiguous() {
    init();
    use crate::MetalTensorData;

    // [1, 10, 3] -> narrow(1, 2, 6) -> [1, 6, 3] -> narrow(1, 1, 3) -> [1, 3, 3]
    let data: Vec<f32> = (0..30).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 10, 3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let view1 = gpu.narrow(1, 2, 6).unwrap(); // elements starting at index 6
    let view2 = view1.narrow(1, 1, 3).unwrap(); // elements starting at index 9

    // Verify zero-copy property
    let v2_data = view2.gpu_data::<MetalTensorData>().unwrap();
    // Composed offset: (2 + 1) * 3 * 4 = 36 bytes
    assert_eq!(
        v2_data.byte_offset(),
        36,
        "chained dim-1 offset: (2+1) * stride_1(3) * 4 = 36"
    );

    // Verify values match CPU
    let cpu_result = cpu.narrow(1, 3, 3).unwrap();
    let gpu_vals = view2
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "narrow_dim1_chained");
}

/// Non-contiguous dim-1 narrow: [2, 6, 3] with shape[0]=2 > 1.
/// This MUST fall back to GPU kernel dispatch, not zero-copy.
#[test]
fn test_gpu_narrow_dim1_non_contiguous_still_correct() {
    init();
    // [2, 6, 3] narrow(1, 1, 3) -> [2, 3, 3] — NOT contiguous (shape[0]=2)
    let data: Vec<f32> = (0..36).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[2, 6, 3], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    let gpu_view = gpu.narrow(1, 1, 3).unwrap();
    let cpu_view = cpu.narrow(1, 1, 3).unwrap();

    assert_eq!(gpu_view.dims(), &[2, 3, 3]);
    let gpu_vals = gpu_view
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_view.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "narrow_dim1_non_contiguous");
}

/// SigLip2/VitEncoder QKV narrow pattern (#4319): [1, 1024, 2304] dim=2.
///
/// This is the exact pattern from VitEncoder attention: fused QKV projection
/// outputs [B, S, 3*D] and is split into Q, K, V via three narrow(2, ...) ops.
/// With B=1, S=1024, D=768, this is [1, 1024, 2304] narrowed along dim=2.
///
/// This narrow is NOT contiguous because shape[1]=1024 > 1: the 1024 rows
/// each contribute 768 elements but are separated by (2304-768)=1536 element
/// gaps. Zero-copy byte-offset views require contiguous memory, so this falls
/// back to GPU kernel dispatch. Eliminating these dispatches would require
/// strided view support in MetalTensorData (#4319).
///
/// This test verifies correctness of the GPU kernel path for all three splits.
#[test]
fn test_gpu_narrow_qkv_split_dim2_correctness() {
    init();
    let b = 1;
    let s = 1024;
    let d = 768;
    let total = b * s * 3 * d;
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.001).collect();
    let cpu = DynTensor::new(&data, &[b, s, 3 * d], &Device::Cpu).unwrap();
    let gpu = cpu.to_device(&Device::metal()).unwrap();

    // Q: narrow(2, 0, 768)
    let q_gpu = gpu.narrow(2, 0, d).unwrap();
    let q_cpu = cpu.narrow(2, 0, d).unwrap();
    assert_eq!(q_gpu.dims(), &[b, s, d]);
    let q_gpu_vals = q_gpu
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let q_cpu_vals = q_cpu.to_flat_vec::<f32>().unwrap();
    assert_close(&q_gpu_vals, &q_cpu_vals, 1e-5, "qkv_Q");

    // K: narrow(2, 768, 768)
    let k_gpu = gpu.narrow(2, d, d).unwrap();
    let k_cpu = cpu.narrow(2, d, d).unwrap();
    assert_eq!(k_gpu.dims(), &[b, s, d]);
    let k_gpu_vals = k_gpu
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let k_cpu_vals = k_cpu.to_flat_vec::<f32>().unwrap();
    assert_close(&k_gpu_vals, &k_cpu_vals, 1e-5, "qkv_K");

    // V: narrow(2, 1536, 768)
    let v_gpu = gpu.narrow(2, 2 * d, d).unwrap();
    let v_cpu = cpu.narrow(2, 2 * d, d).unwrap();
    assert_eq!(v_gpu.dims(), &[b, s, d]);
    let v_gpu_vals = v_gpu
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let v_cpu_vals = v_cpu.to_flat_vec::<f32>().unwrap();
    assert_close(&v_gpu_vals, &v_cpu_vals, 1e-5, "qkv_V");
}

/// Verify that dim-2 narrow on multi-row tensor uses GPU kernel (not zero-copy).
///
/// The narrow [1, 1024, 2304] dim=2 is NOT contiguous because shape[1]=1024.
/// A zero-copy view would read wrong data: row 1 of the view would map to
/// bytes in the middle of row 0 of the parent. This test verifies the result
/// is a fresh buffer (different allocation) rather than a view into the parent.
#[test]
fn test_gpu_narrow_dim2_multi_row_is_not_zero_copy() {
    init();
    use crate::MetalTensorData;

    // [1, 4, 12] narrow(2, 0, 4) -> [1, 4, 4]
    // shape[1]=4 > 1, so NOT contiguous for dim-2 narrow.
    let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
    let cpu = DynTensor::new(&data, &[1, 4, 12], &Device::Cpu).unwrap();
    let parent = cpu.to_device(&Device::metal()).unwrap();
    let parent_data = parent.gpu_data::<MetalTensorData>().unwrap();

    let parent_buf_len = parent_data.buffer().len();
    assert_eq!(parent_buf_len, 192, "parent buffer = 1*4*12*4 bytes");

    let view = parent.narrow(2, 0, 4).unwrap();
    assert_eq!(view.dims(), &[1, 4, 4]);
    let view_data = view.gpu_data::<MetalTensorData>().unwrap();

    // The result should be a NEW buffer (GPU kernel output), not a view.
    // A view would share the parent buffer (same length); a new buffer has
    // exactly the output size.
    let expected_output_bytes = 1 * 4 * 4 * 4; // 64 bytes
    assert_eq!(
        view_data.buffer().len(),
        expected_output_bytes,
        "dim-2 narrow on multi-row tensor should allocate a new buffer, not share parent"
    );
    assert_eq!(
        view_data.byte_offset(),
        0,
        "new buffer should have zero byte_offset"
    );

    // Verify correctness: should get elements [0..3, 12..15, 24..27, 36..39]
    let vals = view
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_view = cpu.narrow(2, 0, 4).unwrap();
    let cpu_vals = cpu_view.to_flat_vec::<f32>().unwrap();
    assert_close(&vals, &cpu_vals, 0.0, "narrow_dim2_multi_row_correctness");
}
