#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU slice_set and packed cat (>28 inputs) integration tests.
//!
//! Extracted from `dyn_tensor_metal_shape_ops_tests.rs` to stay under 500 LOC.
//! Slice_set tests cover dim-0/dim-1 writes and the KV cache pattern.
//! Packed cat tests exercise the >MAX_DIRECT_BINDING_INPUTS buffer path.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use crate::test_common::{assert_close, init};

// -- Slice_set tests (#1297) --------------------------------------------------

#[test]
fn test_gpu_slice_set_dim1() {
    init();
    // dst: [3, 4], src: [3, 2] -> write at offset=1 along dim=1
    let dst_data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let src_data: Vec<f32> = vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0];

    let cpu_dst = DynTensor::new(&dst_data, &[3, 4], &Device::Cpu).unwrap();
    let cpu_src = DynTensor::new(&src_data, &[3, 2], &Device::Cpu).unwrap();

    let gpu_dst = cpu_dst.to_device(&Device::metal()).unwrap();
    let gpu_src = cpu_src.to_device(&Device::metal()).unwrap();

    let gpu_result = gpu_dst.slice_set(1, 1, &gpu_src).unwrap();
    assert_eq!(gpu_result.dims(), &[3, 4]);
    assert_eq!(gpu_result.device(), Device::metal());

    // CPU reference: reconstruct from same data since slice_set consumes self.
    let cpu_dst = DynTensor::new(&dst_data, &[3, 4], &Device::Cpu).unwrap();
    let cpu_src = DynTensor::new(&src_data, &[3, 2], &Device::Cpu).unwrap();
    let cpu_result = cpu_dst.slice_set(1, 1, &cpu_src).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "slice_set_dim1");
}

#[test]
fn test_gpu_slice_set_kv_cache_pattern() {
    init();
    // KV cache pattern: [B, H, S, D] with slice along dim=2 (sequence dim).
    // dst: [1, 2, 8, 4] (batch=1, heads=2, seq=8, head_dim=4)
    // src: [1, 2, 1, 4] (single new token)
    let dst_data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
    let src_data: Vec<f32> = vec![
        99.0, 98.0, 97.0, 96.0, // head 0
        95.0, 94.0, 93.0, 92.0, // head 1
    ];

    let cpu_dst = DynTensor::new(&dst_data, &[1, 2, 8, 4], &Device::Cpu).unwrap();
    let cpu_src = DynTensor::new(&src_data, &[1, 2, 1, 4], &Device::Cpu).unwrap();

    let gpu_dst = cpu_dst.to_device(&Device::metal()).unwrap();
    let gpu_src = cpu_src.to_device(&Device::metal()).unwrap();

    // Write new token at position 3 in the sequence.
    let gpu_result = gpu_dst.slice_set(2, 3, &gpu_src).unwrap();
    assert_eq!(gpu_result.dims(), &[1, 2, 8, 4]);
    assert_eq!(gpu_result.device(), Device::metal());

    // CPU reference: reconstruct from same data since slice_set consumes self.
    let cpu_dst = DynTensor::new(&dst_data, &[1, 2, 8, 4], &Device::Cpu).unwrap();
    let cpu_src = DynTensor::new(&src_data, &[1, 2, 1, 4], &Device::Cpu).unwrap();
    let cpu_result = cpu_dst.slice_set(2, 3, &cpu_src).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "slice_set_kv_cache");
}

#[test]
fn test_gpu_slice_set_dim0() {
    init();
    // dst: [4, 3], src: [2, 3] -> write 2 rows starting at offset=1
    let dst_data: Vec<f32> = vec![0.0; 12];
    let src_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let cpu_dst = DynTensor::new(&dst_data, &[4, 3], &Device::Cpu).unwrap();
    let cpu_src = DynTensor::new(&src_data, &[2, 3], &Device::Cpu).unwrap();

    let gpu_dst = cpu_dst.to_device(&Device::metal()).unwrap();
    let gpu_src = cpu_src.to_device(&Device::metal()).unwrap();

    let gpu_result = gpu_dst.slice_set(0, 1, &gpu_src).unwrap();
    assert_eq!(gpu_result.dims(), &[4, 3]);

    // CPU reference: reconstruct from same data since slice_set consumes self.
    let cpu_dst = DynTensor::new(&dst_data, &[4, 3], &Device::Cpu).unwrap();
    let cpu_src = DynTensor::new(&src_data, &[2, 3], &Device::Cpu).unwrap();
    let cpu_result = cpu_dst.slice_set(0, 1, &cpu_src).unwrap();
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "slice_set_dim0");
}

/// Regression test for #1969: slice_set on narrow views (byte_offset > 0).
///
/// Both src and dst are narrow views into larger parent tensors. Before the
/// fix, gpu_slice_set ignored byte_offset, reading from byte 0 of each
/// underlying buffer instead of the logical tensor start.
#[test]
fn test_gpu_slice_set_narrow_view_byte_offset() {
    init();
    // Parent tensors on GPU — larger than what we'll narrow to.
    let parent_dst_data: Vec<f32> = (0..24).map(|i| i as f32).collect(); // [6, 4]
    let parent_src_data: Vec<f32> = (100..112).map(|i| i as f32).collect(); // [3, 4]

    let parent_dst = DynTensor::new(&parent_dst_data, &[6, 4], &Device::metal()).unwrap();
    let parent_src = DynTensor::new(&parent_src_data, &[3, 4], &Device::metal()).unwrap();

    // Narrow along dim 0 to create views with byte_offset > 0.
    // dst_view = parent_dst[1..5, :] → shape [4, 4], byte_offset = 1*4*4 = 16
    let dst_view = parent_dst.narrow(0, 1, 4).unwrap();
    assert_eq!(dst_view.dims(), &[4, 4]);
    // src_view = parent_src[1..3, :] → shape [2, 4], byte_offset = 1*4*4 = 16
    let src_view = parent_src.narrow(0, 1, 2).unwrap();
    assert_eq!(src_view.dims(), &[2, 4]);

    // slice_set: write src_view into dst_view at dim=0, offset=1
    // Result should be [4, 4] with rows: dst_row1, src_row0, src_row1, dst_row3
    let gpu_result = dst_view.slice_set(0, 1, &src_view).unwrap();
    assert_eq!(gpu_result.dims(), &[4, 4]);

    // CPU reference with identical narrow views.
    let cpu_parent_dst = DynTensor::new(&parent_dst_data, &[6, 4], &Device::Cpu).unwrap();
    let cpu_parent_src = DynTensor::new(&parent_src_data, &[3, 4], &Device::Cpu).unwrap();
    let cpu_dst_view = cpu_parent_dst.narrow(0, 1, 4).unwrap();
    let cpu_src_view = cpu_parent_src.narrow(0, 1, 2).unwrap();
    let cpu_result = cpu_dst_view.slice_set(0, 1, &cpu_src_view).unwrap();

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(
        &gpu_vals,
        &cpu_vals,
        0.0,
        "slice_set_narrow_view_byte_offset",
    );

    // Verify the actual values to confirm narrow offsets were respected:
    // dst_view rows are [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19]
    // src_view rows are [104,105,106,107], [108,109,110,111]
    // After slice_set(dim=0, offset=1): row0=dst[0], row1=src[0], row2=src[1], row3=dst[3]
    let expected: Vec<f32> = vec![
        4.0, 5.0, 6.0, 7.0, // dst_view row 0 (parent row 1)
        104.0, 105.0, 106.0, 107.0, // src_view row 0 (parent row 1)
        108.0, 109.0, 110.0, 111.0, // src_view row 1 (parent row 2)
        16.0, 17.0, 18.0, 19.0, // dst_view row 3 (parent row 4)
    ];
    assert_close(
        &gpu_vals,
        &expected,
        0.0,
        "slice_set_narrow_view_exact_values",
    );
}

// -- Cat single-input test (#1311) --------------------------------------------

#[test]
fn test_gpu_cat_single_tensor() {
    init();
    let a = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::metal()).unwrap();
    let result = DynTensor::cat(&[&a], 0).unwrap();
    assert_eq!(result.dims(), &[2, 3]);
    assert_eq!(result.device(), Device::metal());
    let vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&vals, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.0, "cat_single");
}

#[test]
fn test_gpu_cat_single_tensor_dim1() {
    init();
    let a = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::metal()).unwrap();
    let result = DynTensor::cat(&[&a], 1).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    assert_eq!(result.device(), Device::metal());
    let vals = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    assert_close(&vals, &[1.0, 2.0, 3.0, 4.0], 0.0, "cat_single_dim1");
}

// -- Packed cat test (>28 inputs triggers packed buffer path) ------------------

#[test]
fn test_gpu_cat_packed_30_inputs_dim0() {
    init();
    // 30 inputs exceeds MAX_DIRECT_BINDING_INPUTS (28), exercising the
    // packed buffer dispatch path end-to-end through DynTensor::cat().
    let n = 30;
    let cols = 4;
    let mut gpu_tensors: Vec<DynTensor> = Vec::with_capacity(n);
    let mut cpu_tensors: Vec<DynTensor> = Vec::with_capacity(n);

    for i in 0..n {
        let data: Vec<f32> = (0..cols).map(|j| (i * cols + j) as f32).collect();
        gpu_tensors.push(DynTensor::new(&data, &[1, cols], &Device::metal()).unwrap());
        cpu_tensors.push(DynTensor::new(&data, &[1, cols], &Device::Cpu).unwrap());
    }

    let gpu_refs: Vec<&DynTensor> = gpu_tensors.iter().collect();
    let cpu_refs: Vec<&DynTensor> = cpu_tensors.iter().collect();

    let gpu_result = DynTensor::cat(&gpu_refs, 0).unwrap();
    assert_eq!(gpu_result.dims(), &[n, cols]);
    assert_eq!(gpu_result.device(), Device::metal());

    let cpu_result = DynTensor::cat(&cpu_refs, 0).unwrap();

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "cat_packed_30_dim0");
}

#[test]
fn test_gpu_cat_packed_30_inputs_dim1() {
    init();
    // 30 inputs along dim=1, packed buffer path.
    let n = 30;
    let rows = 2;
    let mut gpu_tensors: Vec<DynTensor> = Vec::with_capacity(n);
    let mut cpu_tensors: Vec<DynTensor> = Vec::with_capacity(n);

    for i in 0..n {
        let data: Vec<f32> = (0..rows * 3).map(|j| (i * rows * 3 + j) as f32).collect();
        gpu_tensors.push(DynTensor::new(&data, &[rows, 3], &Device::metal()).unwrap());
        cpu_tensors.push(DynTensor::new(&data, &[rows, 3], &Device::Cpu).unwrap());
    }

    let gpu_refs: Vec<&DynTensor> = gpu_tensors.iter().collect();
    let cpu_refs: Vec<&DynTensor> = cpu_tensors.iter().collect();

    let gpu_result = DynTensor::cat(&gpu_refs, 1).unwrap();
    assert_eq!(gpu_result.dims(), &[rows, n * 3]);
    assert_eq!(gpu_result.device(), Device::metal());

    let cpu_result = DynTensor::cat(&cpu_refs, 1).unwrap();

    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_close(&gpu_vals, &cpu_vals, 0.0, "cat_packed_30_dim1");
}
