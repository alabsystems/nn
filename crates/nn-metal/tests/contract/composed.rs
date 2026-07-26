// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-backend contract tests for composed multi-op tensor kernels:
//! GPU output of a Conv1d → Snake chain within NY composed IBP bounds.
//!
//! This bridges the gap between "individual kernel GPU contracts" (contract_conv1d.rs)
//! and "bounds composition verification" (compose_tensor_chain.rs in nn-verify).
//! The test proves that GPU execution of a composed pipeline matches the composed
//! bounds — not just each individual kernel.
//!
//! Note: InstanceNorm1d is a high-level TensorOpKind that the Metal codegen does not
//! support directly (it must be decomposed via build_instance_norm_decomposed).
//! These tests use Conv1d → Snake which exercises 3 Metal dispatch steps:
//! Conv1d kernel, Broadcast, and Snake Elementwise.
//!
//! Part of #637.

use super::test_utils::{assert_gpu_within_bounds, metal_setup, rand_f32_vec};

use nn_dsl::adain::build_snake_scalar_kernel;
use nn_dsl::tensor_ir::{
    BroadcastAlignment, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind,
};
use nn_dsl::ScalarType;
use nn_metal::execute_tensor_dispatch;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Conv1d → Snake composed block builder
// ---------------------------------------------------------------------------

/// Build a composed Conv1d → Snake block.
///
/// Nodes: data [in_ch, in_len], weight [out_ch, in_ch, k], alpha [1],
///        Conv1d, Broadcast(alpha), Snake(Elementwise).
/// Output: Snake [out_ch, out_len].
///
/// This exercises 3 Metal dispatch steps: Conv1d, Broadcast, Elementwise.
fn build_conv1d_snake_block(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    in_length: usize,
    stride: usize,
    padding: usize,
) -> TensorKernelDef {
    let out_length = (in_length + 2 * padding - kernel_size) / stride + 1;
    let out_shape = vec![out_channels, out_length];
    let snake_kernel = build_snake_scalar_kernel().expect("snake kernel must build");

    let nodes = vec![
        TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: "data".to_string(),
                shape: vec![in_channels, in_length],
            },
            vec![in_channels, in_length],
        ),
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Input {
                name: "weight".to_string(),
                shape: vec![out_channels, in_channels, kernel_size],
            },
            vec![out_channels, in_channels, kernel_size],
        ),
        TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::Input {
                name: "alpha".to_string(),
                shape: vec![1],
            },
            vec![1],
        ),
        TensorNode::new(
            TensorNodeId::new(3),
            TensorOpKind::Conv1d {
                input: TensorNodeId::new(0),
                weight: TensorNodeId::new(1),
                bias: None,
                stride,
                padding,
                dilation: 1,
                groups: 1,
            },
            out_shape.clone(),
        ),
        TensorNode::new(
            TensorNodeId::new(4),
            TensorOpKind::Broadcast {
                input: TensorNodeId::new(2),
                target_shape: out_shape.clone(),
                alignment: BroadcastAlignment::Right,
            },
            out_shape.clone(),
        ),
        TensorNode::new(
            TensorNodeId::new(5),
            TensorOpKind::Elementwise {
                kernel: snake_kernel,
                inputs: vec![TensorNodeId::new(3), TensorNodeId::new(4)],
            },
            out_shape,
        ),
    ];

    TensorKernelDef::new("conv1d_snake", nodes, TensorNodeId::new(5))
}

// ---------------------------------------------------------------------------
// Shared verification helpers
// ---------------------------------------------------------------------------

/// Prove IBP bounds for a composed tensor kernel, returning (proved_lo, proved_hi).
fn prove_composed_bounds(
    def: &TensorKernelDef,
    bindings: &[TensorParamBinding],
    in_ch: usize,
    in_len: usize,
) -> (ArrayD<f32>, ArrayD<f32>) {
    let graph = tensor_kernel_to_graph(def, bindings).expect("composed graph");
    let lower_in = ArrayD::from_elem(IxDyn(&[in_ch, in_len]), -1.0f32);
    let upper_in = ArrayD::from_elem(IxDyn(&[in_ch, in_len]), 1.0f32);
    let input_bounds = BoundedTensor::new(lower_in, upper_in).expect("input bounds");
    let output_bounds = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    let (lo, hi) = output_bounds.lower_upper();
    assert!(
        lo.iter().all(|v| v.is_finite()),
        "proved lower must be finite"
    );
    assert!(
        hi.iter().all(|v| v.is_finite()),
        "proved upper must be finite"
    );
    for (l, u) in lo.iter().zip(hi.iter()) {
        assert!(l <= u, "lower {l} must be <= upper {u}");
    }
    (lo.clone(), hi.clone())
}

// ===========================================================================
// Composed multi-op contract tests
// ===========================================================================

/// Composed contract: Conv1d(2→3, k=2) → Snake(α=1).
/// Small config for fast execution. Tests dispatch step ordering and buffer reuse.
/// Part of #637.
#[test]
fn test_conv1d_snake_gpu_output_within_composed_bounds_small() {
    let (in_ch, out_ch, kernel_size, in_len, stride, padding) = (2, 3, 2, 6, 1, 0);
    let out_len = (in_len + 2 * padding - kernel_size) / stride + 1;

    let def = build_conv1d_snake_block(in_ch, out_ch, kernel_size, in_len, stride, padding);

    let weight_data = rand_f32_vec(0xDE0C_0001, out_ch * in_ch * kernel_size, -0.5, 0.5);
    let alpha = 1.0f32;

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[out_ch, in_ch, kernel_size]), weight_data.clone())
                .expect("weight shape"),
        ),
        TensorParamBinding::ConstantScalar(alpha),
    ];

    let (proved_lo, proved_hi) = prove_composed_bounds(&def, &bindings, in_ch, in_len);
    assert_eq!(
        proved_lo.shape(),
        &[out_ch, out_len],
        "composed output bounds shape"
    );

    let cache = metal_setup();
    let data = rand_f32_vec(0xDE0C_0002, in_ch * in_len, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("weight", weight_data);
    inputs.insert("alpha", vec![alpha]);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("composed GPU dispatch");
    assert_eq!(gpu_out.len(), out_ch * out_len, "output length");

    assert_gpu_within_bounds("conv1d_snake_small", &gpu_out, &proved_lo, &proved_hi);
}

/// Composed contract: dvoice-realistic Conv1d(1→48, k=8, stride=4, pad=2) →
/// Snake(α=1).
/// Tests precision accumulation across chained ops at production scale.
/// Part of #637.
#[test]
fn test_conv1d_snake_gpu_output_within_composed_bounds_dvoice() {
    let (in_ch, out_ch, kernel_size, in_len, stride, padding) = (1, 48, 8, 64, 4, 2);
    let out_len = (in_len + 2 * padding - kernel_size) / stride + 1;

    let def = build_conv1d_snake_block(in_ch, out_ch, kernel_size, in_len, stride, padding);

    let weight_data = rand_f32_vec(0xDA5E_0001, out_ch * in_ch * kernel_size, -0.2, 0.2);
    let alpha = 1.0f32;

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[out_ch, in_ch, kernel_size]), weight_data.clone())
                .expect("weight shape"),
        ),
        TensorParamBinding::ConstantScalar(alpha),
    ];

    let (proved_lo, proved_hi) = prove_composed_bounds(&def, &bindings, in_ch, in_len);
    assert_eq!(proved_lo.shape(), &[out_ch, out_len]);

    let cache = metal_setup();
    let data = rand_f32_vec(0xDA5E_0002, in_ch * in_len, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("weight", weight_data);
    inputs.insert("alpha", vec![alpha]);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("dvoice composed GPU dispatch");
    assert_eq!(gpu_out.len(), out_ch * out_len);

    assert_gpu_within_bounds("conv1d_snake_dvoice", &gpu_out, &proved_lo, &proved_hi);
}

/// Composed contract: Conv1d(1→4, k=3) → Snake(α=0.5).
/// Non-unit alpha to exercise Snake non-linearity scaling.
/// Part of #637.
#[test]
fn test_conv1d_snake_gpu_output_within_composed_bounds_alpha_half() {
    let (in_ch, out_ch, kernel_size, in_len, stride, padding) = (1, 4, 3, 16, 1, 1);
    let out_len = (in_len + 2 * padding - kernel_size) / stride + 1;

    let def = build_conv1d_snake_block(in_ch, out_ch, kernel_size, in_len, stride, padding);

    let weight_data = rand_f32_vec(0xA1FA_0001, out_ch * in_ch * kernel_size, -0.3, 0.3);
    let alpha = 0.5f32;

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(
            ArrayD::from_shape_vec(IxDyn(&[out_ch, in_ch, kernel_size]), weight_data.clone())
                .expect("weight shape"),
        ),
        TensorParamBinding::ConstantScalar(alpha),
    ];

    let (proved_lo, proved_hi) = prove_composed_bounds(&def, &bindings, in_ch, in_len);
    assert_eq!(proved_lo.shape(), &[out_ch, out_len]);

    let cache = metal_setup();
    let data = rand_f32_vec(0xA1FA_0002, in_ch * in_len, -1.0, 1.0);
    let mut inputs = HashMap::new();
    inputs.insert("data", data);
    inputs.insert("weight", weight_data);
    inputs.insert("alpha", vec![alpha]);

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("alpha=0.5 composed GPU dispatch");
    assert_eq!(gpu_out.len(), out_ch * out_len);

    assert_gpu_within_bounds("conv1d_snake_alpha_half", &gpu_out, &proved_lo, &proved_hi);
}
