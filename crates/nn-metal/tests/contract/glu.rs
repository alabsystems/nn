// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-backend contract tests for standalone GLU (Gated Linear Unit) op:
//! GPU output of GLU(input, axis) within NY IBP bounds.
//!
//! GLU decomposes into: narrow(data) * sigmoid(narrow(gate)).
//! The builder's `add_glu` produces 4 nodes:
//!   Narrow(data, axis, 0, half) + Narrow(gate, axis, half, half)
//!   + Sigmoid(gate) + BinaryMul(data, sigmoid(gate))
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
// GLU GPU contract tests
// ===========================================================================

/// GLU along axis 0: input [8, 16] → output [4, 16].
/// Exercises: 2× Narrow + Sigmoid + BinaryMul decomposition.
/// Part of #709 AC1.
#[test]
fn test_glu_gpu_within_bounds_axis0() {
    let (channels_2x, length) = (8, 16);
    let half = channels_2x / 2;

    let mut b = TensorBlockBuilder::new("glu_axis0");
    let x = b.add_input("x", &[channels_2x, length]);
    let out = b.add_glu(x, 0, &[channels_2x, length]).expect("even dim");
    let def = b.build(out).expect("valid graph");

    let bindings = vec![TensorParamBinding::Variable];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("GLU graph must build");
    // Use 2D bounds matching tensor shape [C, T] — single-variable graphs have
    // axis_offset=0, so input bounds must match the tensor IR shape directly.
    let lower_in = ArrayD::from_elem(IxDyn(&[channels_2x, length]), -1.0f32);
    let upper_in = ArrayD::from_elem(IxDyn(&[channels_2x, length]), 1.0f32);
    let input_bounds = BoundedTensor::new(lower_in, upper_in).expect("valid input bounds");
    let output_bounds = graph.propagate_ibp(&input_bounds).expect("IBP through GLU");
    let (proved_lo, proved_hi) = output_bounds.lower_upper();

    assert_eq!(proved_lo.shape(), &[half, length], "output bounds shape");

    // Run on Metal GPU.
    let cache = metal_setup();
    let data = rand_f32_vec(0x6100_0001, channels_2x * length, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("x", data);

    let gpu_out =
        execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs).expect("GLU GPU dispatch");
    assert_eq!(gpu_out.len(), half * length, "output length");

    assert_gpu_within_bounds("glu_axis0", &gpu_out, proved_lo, proved_hi);
}

/// GLU along axis 1 (temporal): input [4, 16] → output [4, 8].
/// Tests GLU splitting along the temporal dimension instead of channels.
/// Part of #709 AC1.
#[test]
fn test_glu_gpu_within_bounds_axis1() {
    let (channels, length_2x) = (4, 16);
    let half = length_2x / 2;

    let mut b = TensorBlockBuilder::new("glu_axis1");
    let x = b.add_input("x", &[channels, length_2x]);
    let out = b.add_glu(x, 1, &[channels, length_2x]).expect("even dim");
    let def = b.build(out).expect("valid graph");

    let bindings = vec![TensorParamBinding::Variable];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("GLU graph must build");
    // Use 2D bounds matching tensor shape [C, T] — single-variable graphs have
    // axis_offset=0, so input bounds must match the tensor IR shape directly.
    let lower_in = ArrayD::from_elem(IxDyn(&[channels, length_2x]), -2.0f32);
    let upper_in = ArrayD::from_elem(IxDyn(&[channels, length_2x]), 2.0f32);
    let input_bounds = BoundedTensor::new(lower_in, upper_in).expect("valid input bounds");
    let output_bounds = graph.propagate_ibp(&input_bounds).expect("IBP through GLU");
    let (proved_lo, proved_hi) = output_bounds.lower_upper();

    assert_eq!(proved_lo.shape(), &[channels, half], "output bounds shape");

    let cache = metal_setup();
    let data = rand_f32_vec(0x6100_0002, channels * length_2x, -2.0, 2.0);
    let mut inputs = HashMap::new();
    inputs.insert("x", data);

    let gpu_out =
        execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs).expect("GLU GPU dispatch");
    assert_eq!(gpu_out.len(), channels * half, "output length");

    assert_gpu_within_bounds("glu_axis1", &gpu_out, proved_lo, proved_hi);
}

/// Dvoice-scale GLU: input [96, 16] → output [48, 16].
/// Tests at production channel counts (Conv1d outputs 2C=96 for GLU split).
/// Part of #709 AC1.
#[test]
fn test_glu_gpu_within_bounds_dvoice() {
    let (channels_2x, length) = (96, 16);
    let half = channels_2x / 2;

    let mut b = TensorBlockBuilder::new("glu_dvoice");
    let x = b.add_input("x", &[channels_2x, length]);
    let out = b.add_glu(x, 0, &[channels_2x, length]).expect("even dim");
    let def = b.build(out).expect("valid graph");

    let bindings = vec![TensorParamBinding::Variable];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("GLU graph must build");
    // Use 2D bounds matching tensor shape [C, T] — single-variable graphs have
    // axis_offset=0, so input bounds must match the tensor IR shape directly.
    let lower_in = ArrayD::from_elem(IxDyn(&[channels_2x, length]), -1.0f32);
    let upper_in = ArrayD::from_elem(IxDyn(&[channels_2x, length]), 1.0f32);
    let input_bounds = BoundedTensor::new(lower_in, upper_in).expect("valid input bounds");
    let output_bounds = graph.propagate_ibp(&input_bounds).expect("IBP through GLU");
    let (proved_lo, proved_hi) = output_bounds.lower_upper();

    assert_eq!(proved_lo.shape(), &[half, length]);

    let cache = metal_setup();
    let data = rand_f32_vec(0xDA5D_6100, channels_2x * length, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("x", data);

    let gpu_out =
        execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs).expect("GLU GPU dispatch");
    assert_eq!(gpu_out.len(), half * length);

    assert_gpu_within_bounds("glu_dvoice", &gpu_out, proved_lo, proved_hi);
}
