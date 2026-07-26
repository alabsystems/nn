// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for computational cost model.
//!
//! Part of #1739.

use super::*;
use nn_dsl::ir::ScalarType;
use nn_dsl::tensor_ir::{ReduceOp, TensorNodeId};
use nn_dsl::{Conv1dParams, Conv2dParams, ConvTranspose1dParams, DispatchStep};

fn node(id: usize) -> TensorNodeId {
    TensorNodeId::new(id)
}

// --- HardwareCostModel tests ---

#[test]
fn test_m4_max_model_sanity() {
    let model = HardwareCostModel::m4_max();
    assert!(model.peak_tflops_f32 > 10.0);
    assert!(model.peak_bandwidth_gbs > 300.0);
    assert!(model.dispatch_overhead_us > 0.0);
}

#[test]
fn test_roofline_compute_bound() {
    let model = HardwareCostModel::m4_max();
    // 1 TFLOP of compute, 1 byte of memory — compute-bound
    let flops = 1_000_000_000_000u64;
    let time = model.estimate_time_us(flops, 1);
    // Compute time = 1e12 / (14.2e6) ≈ 70422 μs
    assert!(time > 70_000.0);
    assert!(time < 75_000.0); // Includes dispatch overhead
}

#[test]
fn test_roofline_memory_bound() {
    let model = HardwareCostModel::m4_max();
    // 1 FLOP, 1 GB of memory — memory-bound
    let bytes = 1_000_000_000u64;
    let time = model.estimate_time_us(1, bytes);
    // Memory time = 1e9 / (400e3) = 2500 μs
    assert!(time > 2500.0);
    assert!(time < 2510.0); // Plus 5 μs dispatch
}

#[test]
fn test_roofline_dispatch_overhead_minimum() {
    let model = HardwareCostModel::m4_max();
    // Zero work — should still have dispatch overhead
    let time = model.estimate_time_us(0, 0);
    assert!((time - model.dispatch_overhead_us).abs() < 1e-10);
}

// --- Linear FLOP counting ---

#[test]
fn test_linear_flops() {
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
    let flops = step_flops(&step);
    // 2 * 1 * 768 * 3072 = 4_718_592
    assert_eq!(flops, 2 * 768 * 3072);
}

#[test]
fn test_linear_flops_with_bias() {
    let step = DispatchStep::Linear {
        kernel_name: "linear_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        weight: node(1),
        bias: Some(node(3)),
        output: node(2),
        in_features: 768,
        out_features: 3072,
        batch_size: 4,
        total_elements: 4 * 3072,
    };
    let flops = step_flops(&step);
    // MatMul: 2 * 4 * 768 * 3072 = 18_874_368
    // Bias: 4 * 3072 = 12_288
    assert_eq!(flops, 2 * 4 * 768 * 3072 + 4 * 3072);
}

#[test]
fn test_linear_memory_bytes() {
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
    let bytes = step_memory_bytes(&step);
    // Input: 1*768*4 = 3072
    // Weight: 3072*768*4 = 9_437_184
    // Output: 3072*4 = 12_288
    assert_eq!(bytes, (768 + 3072 * 768 + 3072) * 4);
}

// --- MatMul FLOP counting ---

#[test]
fn test_matmul_flops_attention_qk() {
    // Q@K^T in attention: [B, H, S, D] @ [B, H, D, S]
    // For B=1, H=12, S=64, D=64
    let step = DispatchStep::MatMul {
        kernel_name: "matmul_qk".to_string(),
        dtype: ScalarType::F32,
        left: node(0),
        right: node(1),
        output: node(2),
        m: 64,          // S
        k: 64,          // D
        n: 64,          // S
        batch_size: 12, // B*H
        transpose_right: true,
        broadcast_right: false,
        scale: Some(0.125), // 1/sqrt(64)
        total_elements: 12 * 64 * 64,
    };
    let flops = step_flops(&step);
    // 2 * 12 * 64 * 64 * 64 = 6_291_456
    assert_eq!(flops, 2 * 12 * 64 * 64 * 64);
}

// --- Conv1d FLOP counting ---

#[test]
fn test_conv1d_flops() {
    let step = DispatchStep::Conv1d(Conv1dParams::new(
        "conv1d_0".to_string(),
        ScalarType::F32,
        node(0),
        node(1),
        None,
        node(2),
        1,
        48,
        8,
        16384,
        48 * 4096, // out_channels * out_len
        4,
        0,
        1,
        1,
    ));
    let flops = step_flops(&step);
    // out_len = total_elements / out_channels = 4096
    // cpg = 1 (ungrouped), 2 * 4096 * 48 * 1 * 8 = 3_145_728
    assert_eq!(flops, 2_u64 * 4096 * 48 * 8);
}

#[test]
fn test_conv1d_flops_grouped() {
    let step = DispatchStep::Conv1d(Conv1dParams::new(
        "dconv_0".to_string(),
        ScalarType::F32,
        node(0),
        node(1),
        None,
        node(2),
        96,
        96,
        3,
        1024,
        96 * 1024,
        1,
        1,
        1,
        96, // Depthwise
    ));
    let flops = step_flops(&step);
    // channels_per_group = 96/96 = 1
    // cpg = 1 (depthwise), 2 * 1024 * 96 * 1 * 3 = 589_824
    assert_eq!(flops, 2_u64 * 1024 * 96 * 3);
}

// --- Conv2d FLOP counting ---

#[test]
fn test_conv2d_flops() {
    let step = DispatchStep::Conv2d(Conv2dParams::new(
        "conv2d_0".to_string(),
        ScalarType::F32,
        node(0),
        node(1),
        None,
        node(2),
        3,
        64,
        3,
        3,
        224,
        224,
        64 * 224 * 224, // Assuming padding preserves spatial dims
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ));
    let flops = step_flops(&step);
    // out_spatial = total_elements / out_channels = 224*224 = 50176
    // 2 * 50176 * 64 * 3 * 3 * 3 = 173_408_256
    assert_eq!(flops, 2 * 50176 * 64 * 3 * 3 * 3);
}

// --- Softmax ---

#[test]
fn test_softmax_flops() {
    let step = DispatchStep::Softmax {
        kernel_name: "softmax_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        output: node(1),
        axis: 2,
        axis_size: 64,
        outer_size: 12, // B*H
    };
    let flops = step_flops(&step);
    // 5 * 12 * 64 = 3840
    assert_eq!(flops, 5 * 12 * 64);
}

// --- Element-wise ops ---

#[test]
fn test_relu_flops() {
    let step = DispatchStep::Relu {
        kernel_name: "relu_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        output: node(1),
        total_elements: 100_000,
    };
    assert_eq!(step_flops(&step), 100_000);
    // Memory: read + write = 2 * 100_000 * 4 = 800_000
    assert_eq!(step_memory_bytes(&step), 800_000);
}

#[test]
fn test_binary_add_memory() {
    let step = DispatchStep::BinaryAdd {
        kernel_name: "add_0".to_string(),
        dtype: ScalarType::F32,
        left: node(0),
        right: node(1),
        output: node(2),
        total_elements: 10_000,
        broadcast: None,
    };
    // 3 * 10_000 * 4 = 120_000
    assert_eq!(step_memory_bytes(&step), 120_000);
}

// --- Data movement ops ---

#[test]
fn test_reshape_zero_cost() {
    let step = DispatchStep::Reshape {
        input: node(0),
        output: node(1),
    };
    assert_eq!(step_flops(&step), 0);
    assert_eq!(step_memory_bytes(&step), 0);
}

// --- Reduce ---

#[test]
fn test_reduce_flops() {
    let step = DispatchStep::Reduce {
        kernel_name: "reduce_sum".to_string(),
        op: ReduceOp::Sum,
        dtype: ScalarType::F32,
        input: node(0),
        output: node(1),
        reduce_dim: 256,
        outer_size: 64,
        keepdim: false,
    };
    let flops = step_flops(&step);
    assert_eq!(flops, 64 * 256);
}

// --- Embedding ---

#[test]
fn test_embedding_zero_flops() {
    let step = DispatchStep::Embedding {
        kernel_name: "embed_0".to_string(),
        dtype: ScalarType::F32,
        input: node(0),
        weight: node(1),
        output: node(2),
        embedding_dim: 256,
        num_indices: 100,
        total_elements: 100 * 256,
    };
    assert_eq!(step_flops(&step), 0);
    // Memory: indices (100*4) + read rows (25600*4) + write (25600*4) = 205200
    assert_eq!(step_memory_bytes(&step), 100 * 4 + 25600 * 4 + 25600 * 4);
}

// --- ConvTranspose1d ---

#[test]
fn test_conv_transpose1d_flops() {
    let step = DispatchStep::ConvTranspose1d(ConvTranspose1dParams::new(
        "convt_0".to_string(),
        ScalarType::F32,
        node(0),
        node(1),
        None,
        node(2),
        48,
        1,
        8,
        4096,
        16384, // out_channels(1) * out_len
        4,
        0,
        1,
        1,
        0,
    ));
    let flops = step_flops(&step);
    // out_len = 16384/1 = 16384
    // out_channels=1, 2 * 16384 * 1 * 48 * 8 = 12_582_912
    assert_eq!(flops, 2_u64 * 16384 * 48 * 8);
}

// --- step_name ---

#[test]
fn test_step_name_variants() {
    assert_eq!(
        step_name(&DispatchStep::Reshape {
            input: node(0),
            output: node(1)
        }),
        "reshape"
    );
    assert_eq!(
        step_name(&DispatchStep::Relu {
            kernel_name: "nn_relu".to_string(),
            dtype: ScalarType::F32,
            input: node(0),
            output: node(1),
            total_elements: 100,
        }),
        "nn_relu"
    );
}
