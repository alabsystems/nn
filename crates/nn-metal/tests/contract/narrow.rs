// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-backend contract tests for standalone Narrow (slice) op:
//! GPU output of Narrow(input, axis, start, length) within NY
//! IBP bounds.
//!
//! Part of #709 AC1.

use super::test_utils::{assert_gpu_within_bounds, metal_setup, rand_f32_vec};

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::ScalarType;
use nn_metal::execute_tensor_dispatch;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

// ===========================================================================
// Narrow GPU contract tests
// ===========================================================================

/// Narrow along axis 1 (temporal): input [4, 16], slice [4, 8] starting at 4.
/// Part of #709 AC1.
#[test]
fn test_narrow_gpu_within_bounds_axis1() {
    let (channels, length) = (4, 16);
    let (axis, start, slice_len) = (1, 4, 8);

    let mut b = TensorBlockBuilder::new("narrow_axis1");
    let x = b.add_input("x", &[channels, length]);
    let out = b.add_narrow(x, axis, start, slice_len, &[channels, slice_len]);
    let def = b.build(out).expect("valid graph");

    let bindings = vec![TensorParamBinding::Variable];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("narrow graph must build");
    // Use 2D bounds matching tensor shape [C, T] — single-variable graphs have
    // axis_offset=0, so input bounds must match the tensor IR shape directly.
    let lower_in = ArrayD::from_elem(IxDyn(&[channels, length]), -1.0f32);
    let upper_in = ArrayD::from_elem(IxDyn(&[channels, length]), 1.0f32);
    let input_bounds = BoundedTensor::new(lower_in, upper_in).expect("valid input bounds");
    let output_bounds = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through narrow");
    let (proved_lo, proved_hi) = output_bounds.lower_upper();

    assert_eq!(
        proved_lo.shape(),
        &[channels, slice_len],
        "output bounds shape"
    );

    // Run on Metal GPU.
    let cache = metal_setup();
    let data = rand_f32_vec(0x0A01_0001, channels * length, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("x", data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("narrow GPU dispatch");
    assert_eq!(gpu_out.len(), channels * slice_len, "output length");

    assert_gpu_within_bounds("narrow_axis1", &gpu_out, proved_lo, proved_hi);
}

/// Narrow along axis 0 (channel): input [8, 16], slice [4, 16] starting at 0.
/// Part of #709 AC1.
#[test]
fn test_narrow_gpu_within_bounds_axis0() {
    let (channels, length) = (8, 16);
    let (axis, start, slice_len) = (0, 0, 4);

    let mut b = TensorBlockBuilder::new("narrow_axis0");
    let x = b.add_input("x", &[channels, length]);
    let out = b.add_narrow(x, axis, start, slice_len, &[slice_len, length]);
    let def = b.build(out).expect("valid graph");

    let bindings = vec![TensorParamBinding::Variable];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("narrow graph must build");
    // Use 2D bounds matching tensor shape [C, T] — single-variable graphs have
    // axis_offset=0, so input bounds must match the tensor IR shape directly.
    let lower_in = ArrayD::from_elem(IxDyn(&[channels, length]), -2.0f32);
    let upper_in = ArrayD::from_elem(IxDyn(&[channels, length]), 2.0f32);
    let input_bounds = BoundedTensor::new(lower_in, upper_in).expect("valid input bounds");
    let output_bounds = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through narrow");
    let (proved_lo, proved_hi) = output_bounds.lower_upper();

    assert_eq!(
        proved_lo.shape(),
        &[slice_len, length],
        "output bounds shape"
    );

    let cache = metal_setup();
    let data = rand_f32_vec(0x0A02_0001, channels * length, -2.0, 2.0);
    let mut inputs = HashMap::new();
    inputs.insert("x", data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("narrow GPU dispatch");
    assert_eq!(gpu_out.len(), slice_len * length, "output length");

    assert_gpu_within_bounds("narrow_axis0", &gpu_out, proved_lo, proved_hi);
}

/// Dvoice-scale narrow: input [48, 32], slice [48, 16] at start=16.
/// Part of #709 AC1.
#[test]
fn test_narrow_gpu_within_bounds_dvoice() {
    let (channels, length) = (48, 32);
    let (axis, start, slice_len) = (1, 16, 16);

    let mut b = TensorBlockBuilder::new("narrow_dvoice");
    let x = b.add_input("x", &[channels, length]);
    let out = b.add_narrow(x, axis, start, slice_len, &[channels, slice_len]);
    let def = b.build(out).expect("valid graph");

    let bindings = vec![TensorParamBinding::Variable];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("narrow graph must build");
    // Use 2D bounds matching tensor shape [C, T] — single-variable graphs have
    // axis_offset=0, so input bounds must match the tensor IR shape directly.
    let lower_in = ArrayD::from_elem(IxDyn(&[channels, length]), -1.0f32);
    let upper_in = ArrayD::from_elem(IxDyn(&[channels, length]), 1.0f32);
    let input_bounds = BoundedTensor::new(lower_in, upper_in).expect("valid input bounds");
    let output_bounds = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through narrow");
    let (proved_lo, proved_hi) = output_bounds.lower_upper();

    assert_eq!(proved_lo.shape(), &[channels, slice_len]);

    let cache = metal_setup();
    let data = rand_f32_vec(0xDA5D_0010, channels * length, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("x", data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("narrow GPU dispatch");
    assert_eq!(gpu_out.len(), channels * slice_len);

    assert_gpu_within_bounds("narrow_dvoice", &gpu_out, proved_lo, proved_hi);
}
