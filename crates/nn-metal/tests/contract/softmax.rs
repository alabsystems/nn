// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-backend contract tests for standalone Softmax op:
//! GPU output of Softmax(input, axis) within NY IBP bounds.
//!
//! Part of #738 AC5: GPU contract tests for Softmax.

use super::test_utils::{assert_gpu_within_bounds, metal_setup, rand_f32_vec};

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::ScalarType;
use nn_metal::execute_tensor_dispatch;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

// ===========================================================================
// Softmax GPU contract tests
// ===========================================================================

/// Softmax on last axis of 2D tensor: input [4, 8], axis=-1.
/// Part of #738 AC5.
#[test]
fn test_softmax_gpu_within_bounds_2d() {
    let (rows, cols) = (4, 8);

    let mut b = TensorBlockBuilder::new("softmax_2d");
    let x = b.add_input("x", &[rows, cols]);
    let out = b.add_softmax(x, -1, &[rows, cols]);
    let def = b.build(out).expect("valid graph");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("softmax graph must build");

    let lower_in = ArrayD::from_elem(IxDyn(&[rows, cols]), -2.0f32);
    let upper_in = ArrayD::from_elem(IxDyn(&[rows, cols]), 2.0f32);
    let input_bounds = BoundedTensor::new(lower_in, upper_in).expect("valid input bounds");
    let output_bounds = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through softmax");
    let (proved_lo, proved_hi) = output_bounds.lower_upper();

    assert_eq!(proved_lo.shape(), &[rows, cols], "output bounds shape");

    // Run on Metal GPU.
    let cache = metal_setup();
    let data = rand_f32_vec(0x50FD_0001, rows * cols, -2.0, 2.0);
    let mut inputs = HashMap::new();
    inputs.insert("x", data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("softmax GPU dispatch");
    assert_eq!(gpu_out.len(), rows * cols, "output length");

    assert_gpu_within_bounds("softmax_2d", &gpu_out, proved_lo, proved_hi);

    // Extra: verify softmax outputs are non-negative and sum to ~1 per row.
    for row in 0..rows {
        let row_sum: f32 = gpu_out[row * cols..(row + 1) * cols].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-4,
            "softmax row {row} sum = {row_sum}, expected ~1.0"
        );
        for col in 0..cols {
            assert!(
                gpu_out[row * cols + col] >= 0.0,
                "softmax[{row},{col}] = {} should be >= 0",
                gpu_out[row * cols + col]
            );
        }
    }
}

/// Softmax on 1D tensor: input [16], axis=-1.
/// Part of #738 AC5.
#[test]
fn test_softmax_gpu_within_bounds_1d() {
    let size = 16;

    let mut b = TensorBlockBuilder::new("softmax_1d");
    let x = b.add_input("x", &[size]);
    let out = b.add_softmax(x, -1, &[size]);
    let def = b.build(out).expect("valid graph");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("softmax graph must build");

    let lower_in = ArrayD::from_elem(IxDyn(&[size]), -3.0f32);
    let upper_in = ArrayD::from_elem(IxDyn(&[size]), 3.0f32);
    let input_bounds = BoundedTensor::new(lower_in, upper_in).expect("valid input bounds");
    let output_bounds = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through softmax");
    let (proved_lo, proved_hi) = output_bounds.lower_upper();

    assert_eq!(proved_lo.shape(), &[size], "output bounds shape");

    let cache = metal_setup();
    let data = rand_f32_vec(0x50FD_0002, size, -3.0, 3.0);
    let mut inputs = HashMap::new();
    inputs.insert("x", data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("softmax GPU dispatch");
    assert_eq!(gpu_out.len(), size, "output length");

    assert_gpu_within_bounds("softmax_1d", &gpu_out, proved_lo, proved_hi);

    // Verify sum to 1.
    let sum: f32 = gpu_out.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "softmax sum = {sum}, expected ~1.0"
    );
}

/// Dvoice-scale softmax: input [8, 64], axis=-1.
/// Simulates attention weight computation at typical dvoice sequence lengths.
/// Part of #738 AC5.
#[test]
fn test_softmax_gpu_within_bounds_dvoice() {
    let (heads, seq_len) = (8, 64);

    let mut b = TensorBlockBuilder::new("softmax_dvoice");
    let x = b.add_input("logits", &[heads, seq_len]);
    let out = b.add_softmax(x, -1, &[heads, seq_len]);
    let def = b.build(out).expect("valid graph");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("softmax graph must build");

    let lower_in = ArrayD::from_elem(IxDyn(&[heads, seq_len]), -5.0f32);
    let upper_in = ArrayD::from_elem(IxDyn(&[heads, seq_len]), 5.0f32);
    let input_bounds = BoundedTensor::new(lower_in, upper_in).expect("valid input bounds");
    let output_bounds = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through softmax");
    let (proved_lo, proved_hi) = output_bounds.lower_upper();

    assert_eq!(proved_lo.shape(), &[heads, seq_len]);

    let cache = metal_setup();
    let data = rand_f32_vec(0xDA5D_50FD, heads * seq_len, -5.0, 5.0);
    let mut inputs = HashMap::new();
    inputs.insert("logits", data);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("softmax GPU dispatch");
    assert_eq!(gpu_out.len(), heads * seq_len);

    assert_gpu_within_bounds("softmax_dvoice", &gpu_out, proved_lo, proved_hi);

    // Verify per-head softmax properties.
    for h in 0..heads {
        let row_sum: f32 = gpu_out[h * seq_len..(h + 1) * seq_len].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-3,
            "softmax head {h} sum = {row_sum}, expected ~1.0"
        );
    }
}
