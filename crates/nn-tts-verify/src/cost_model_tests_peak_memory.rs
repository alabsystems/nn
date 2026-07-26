// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for step_output_bytes and step_weight_bytes (#1739 Phase 19).

use super::*;
use nn_dsl::ir::ScalarType;
use nn_dsl::tensor_ir::TensorNodeId;
use nn_dsl::{Conv1dParams, ConvTranspose1dParams, DispatchStep, ReduceOp};

fn node(id: usize) -> TensorNodeId {
    TensorNodeId::new(id)
}

// --- step_output_bytes tests ---

#[test]
fn test_output_bytes_linear() {
    let step = DispatchStep::Linear {
        kernel_name: "linear_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        weight: node(1),
        bias: None,
        output: node(2),
        in_features: 768,
        out_features: 3072,
        batch_size: 1,
        total_elements: 3072,
    };
    // 3072 elements × 4 bytes = 12,288
    assert_eq!(step_output_bytes(&step), 3072 * 4);
}

#[test]
fn test_output_bytes_conv1d() {
    let step = DispatchStep::Conv1d(Conv1dParams::new(
        "conv_0".to_string(),
        ScalarType::F32,
        node(0),
        node(1),
        Some(node(2)),
        node(3),
        1,
        48,
        8,
        1000,
        48 * 250, // 12,000
        4,
        0,
        1,
        1,
    ));
    assert_eq!(step_output_bytes(&step), 48 * 250 * 4);
}

#[test]
fn test_output_bytes_reshape_zero() {
    let step = DispatchStep::Reshape {
        input: node(0),
        output: node(1),
    };
    // Reshape: 0 bytes (no new allocation)
    assert_eq!(step_output_bytes(&step), 0);
}

#[test]
fn test_output_bytes_relu() {
    let step = DispatchStep::Relu {
        kernel_name: "relu_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        output: node(1),
        total_elements: 1024,
    };
    assert_eq!(step_output_bytes(&step), 1024 * 4);
}

#[test]
fn test_output_bytes_softmax() {
    let step = DispatchStep::Softmax {
        kernel_name: "softmax_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        output: node(1),
        axis: 1,
        axis_size: 128,
        outer_size: 8,
    };
    // axis_size × outer_size × 4 = 128 × 8 × 4 = 4,096
    assert_eq!(step_output_bytes(&step), 128 * 8 * 4);
}

#[test]
fn test_output_bytes_reduce() {
    let step = DispatchStep::Reduce {
        kernel_name: "reduce_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        output: node(1),
        reduce_dim: 512,
        outer_size: 16,
        op: ReduceOp::Sum,
        keepdim: false,
    };
    // Output: outer_size × 4 = 16 × 4 = 64
    assert_eq!(step_output_bytes(&step), 16 * 4);
}

#[test]
fn test_output_bytes_embedding() {
    let step = DispatchStep::Embedding {
        kernel_name: "embed_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        weight: node(1),
        output: node(2),
        embedding_dim: 768,
        total_elements: 256 * 768, // 256 tokens × 768 dim
        num_indices: 256,
    };
    assert_eq!(step_output_bytes(&step), 256 * 768 * 4);
}

// --- step_weight_bytes tests ---

#[test]
fn test_weight_bytes_linear_no_bias() {
    let step = DispatchStep::Linear {
        kernel_name: "linear_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        weight: node(1),
        bias: None,
        output: node(2),
        in_features: 768,
        out_features: 3072,
        batch_size: 1,
        total_elements: 3072,
    };
    // 768 × 3072 × 4 = 9,437,184
    assert_eq!(step_weight_bytes(&step), 768 * 3072 * 4);
}

#[test]
fn test_weight_bytes_linear_with_bias() {
    let step = DispatchStep::Linear {
        kernel_name: "linear_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        weight: node(1),
        bias: Some(node(3)),
        output: node(2),
        in_features: 768,
        out_features: 3072,
        batch_size: 1,
        total_elements: 3072,
    };
    // Weight: 768 × 3072 × 4 + bias: 3072 × 4
    assert_eq!(step_weight_bytes(&step), 768 * 3072 * 4 + 3072 * 4);
}

#[test]
fn test_weight_bytes_conv1d() {
    let step = DispatchStep::Conv1d(Conv1dParams::new(
        "conv_0".to_string(),
        ScalarType::F32,
        node(0),
        node(1),
        Some(node(2)),
        node(3),
        1,
        48,
        8,
        1000,
        12_000,
        4,
        0,
        1,
        1,
    ));
    // Weight: out_ch × (in_ch/groups) × kernel × sizeof(f32) + bias
    // 48 × (1/1) × 8 × 4 = 1,536 + bias: 48 × 4 = 192 → total 1,728
    assert_eq!(step_weight_bytes(&step), 1_536 + 192);
}

#[test]
fn test_weight_bytes_conv1d_grouped() {
    // Depthwise conv: groups == in_channels
    let step = DispatchStep::Conv1d(Conv1dParams::new(
        "dw_conv".to_string(),
        ScalarType::F32,
        node(0),
        node(1),
        None,
        node(3),
        48,
        48,
        3,
        1000,
        48_000,
        1,
        1,
        1,
        48,
    ));
    // Weight: out_ch × (in_ch/groups) × kernel × sizeof(f32)
    // 48 × (48/48) × 3 × 4 = 48 × 3 × 4 = 576
    assert_eq!(step_weight_bytes(&step), 576);
}

#[test]
fn test_weight_bytes_relu_zero() {
    let step = DispatchStep::Relu {
        kernel_name: "relu_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        output: node(1),
        total_elements: 10_000,
    };
    // Element-wise: 0 weight bytes
    assert_eq!(step_weight_bytes(&step), 0);
}

#[test]
fn test_weight_bytes_conv_transpose1d() {
    let step = DispatchStep::ConvTranspose1d(ConvTranspose1dParams::new(
        "ct_0".to_string(),
        ScalarType::F32,
        node(0),
        node(1),
        Some(node(2)),
        node(3),
        96,
        48,
        8,
        1000,
        48_000,
        4,
        2,
        1,
        1,
        0,
    ));
    // Weight: 48 × 96 × 8 × 4 + bias: 48 × 4
    assert_eq!(step_weight_bytes(&step), 48 * 96 * 8 * 4 + 48 * 4);
}

// --- step_output_bytes: data-movement variants ---

#[test]
fn test_output_bytes_axis_select() {
    // AxisSelect on [2, 8, 4] along axis=0 → output [8, 4] = 32 elements
    let step = DispatchStep::AxisSelect {
        kernel_name: "select_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        output: node(1),
        input_shape: vec![2, 8, 4],
        axis: 0,
        index: 0,
    };
    // 2*8*4 = 64 total, axis dim = 2 → 64/2 = 32 elements → 128 bytes
    assert_eq!(step_output_bytes(&step), 32 * 4);
}

#[test]
fn test_output_bytes_axis_select_middle_axis() {
    // AxisSelect on [2, 8, 4] along axis=1 → output [2, 4] = 8 elements
    let step = DispatchStep::AxisSelect {
        kernel_name: "select_1".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        output: node(1),
        input_shape: vec![2, 8, 4],
        axis: 1,
        index: 3,
    };
    // 64 total, axis dim = 8 → 64/8 = 8 elements → 32 bytes
    assert_eq!(step_output_bytes(&step), 8 * 4);
}

#[test]
fn test_output_bytes_stack() {
    // Stack 3 tensors of shape [4, 8] along new axis → [3, 4, 8] = 96 elements
    let step = DispatchStep::Stack {
        kernel_name: "stack_0".to_string(),
        dtype: ScalarType::F32,
        inputs: vec![node(0), node(1), node(2)],
        output: node(3),
        input_shape: vec![4, 8],
        axis: 0,
    };
    // 3 inputs × 32 per input = 96 elements → 384 bytes
    assert_eq!(step_output_bytes(&step), 96 * 4);
}

#[test]
fn test_output_bytes_narrow() {
    // Narrow on [2, 16, 4] along axis=1, length=8 → [2, 8, 4] = 64 elements
    let step = DispatchStep::Narrow {
        kernel_name: "narrow_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        output: node(1),
        input_shape: vec![2, 16, 4],
        axis: 1,
        start: 4,
        length: 8,
    };
    // total = 128, axis dim = 16 → 128/16 * 8 = 64 elements → 256 bytes
    assert_eq!(step_output_bytes(&step), 64 * 4);
}

#[test]
fn test_output_bytes_zero_pad_1d() {
    // ZeroPad1d: channels=48, out_length=260 → 48*260 = 12,480 elements
    let step = DispatchStep::ZeroPad1d {
        kernel_name: "zp_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        output: node(1),
        channels: 48,
        in_length: 250,
        pad_left: 5,
        out_length: 260,
    };
    assert_eq!(step_output_bytes(&step), 48 * 260 * 4);
}

#[test]
fn test_output_bytes_transpose() {
    let step = DispatchStep::Transpose {
        kernel_name: "trans_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        output: node(1),
        input_shape: vec![4, 8, 16],
        axes: vec![0, 2, 1],
        total_elements: 4 * 8 * 16,
    };
    assert_eq!(step_output_bytes(&step), 4 * 8 * 16 * 4);
}

#[test]
fn test_output_bytes_concat() {
    // Concat two tensors of shape [2, 3, 4] and [2, 5, 4] along axis=1
    // → [2, 8, 4] = 64 elements
    let step = DispatchStep::Concat {
        kernel_name: "cat_0".to_string(),
        dtype: ScalarType::F32,
        inputs: vec![node(0), node(1)],
        output: node(2),
        first_input_shape: vec![2, 3, 4],
        input_axis_sizes: vec![3, 5],
        axis: 1,
    };
    // non_axis = 2 * 4 = 8, total_axis = 3 + 5 = 8 → 64 elements → 256 bytes
    assert_eq!(step_output_bytes(&step), 64 * 4);
}

#[test]
fn test_output_bytes_binary_ops() {
    let step = DispatchStep::BinaryAdd {
        kernel_name: "add_0".to_string(),
        dtype: ScalarType::F32,
        left: node(0),
        right: node(1),
        output: node(2),
        total_elements: 1024,
        broadcast: None,
    };
    assert_eq!(step_output_bytes(&step), 1024 * 4);

    let step = DispatchStep::BinaryMul {
        kernel_name: "mul_0".to_string(),
        dtype: ScalarType::F32,
        left: node(0),
        right: node(1),
        output: node(2),
        total_elements: 512,
        broadcast: None,
    };
    assert_eq!(step_output_bytes(&step), 512 * 4);
}
