// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end compiled model tests for `NativeOpKind::FusedUpsampleConv1d`.
//!
//! The peephole pass fuses `Upsample1d + Conv1d` into a single
//! `FusedUpsampleConv1d` NativeOp that executes nearest-neighbor upsample
//! followed by conv1d in a single plan step.
//!
//! This test verifies the full pipeline: trace graph with Upsample1d +
//! Conv1d -> peephole fuses into FusedUpsampleConv1d -> GPU execute ->
//! verify against CPU reference.
//!
//! Part of #4310.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use super::helpers::{assert_close, compile_and_run, create_input_buffer, input_node};

// -- CPU reference helpers ----------------------------------------------------

/// CPU nearest-neighbor upsample along the last dimension.
///
/// Input: `[B, C, T]` -> Output: `[B, C, T * factor]`.
/// Each time-step is repeated `factor` times.
fn cpu_upsample_nearest_1d(
    input: &[f32],
    batch: usize,
    channels: usize,
    time: usize,
    factor: usize,
) -> Vec<f32> {
    let t_out = time * factor;
    let mut output = vec![0.0_f32; batch * channels * t_out];
    for b in 0..batch {
        for c in 0..channels {
            let in_offset = (b * channels + c) * time;
            let out_offset = (b * channels + c) * t_out;
            for t in 0..time {
                for f in 0..factor {
                    output[out_offset + t * factor + f] = input[in_offset + t];
                }
            }
        }
    }
    output
}

/// CPU Conv1d: stride/groups=1, dilation=1, with padding.
///
/// Input: `[B, C_in, T]`, Weight: `[C_out, C_in, K]`, Bias: `[C_out]`.
/// Output: `[B, C_out, T_out]` where `T_out = (T + 2*padding - K) / stride + 1`.
fn cpu_conv1d(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch: usize,
    c_in: usize,
    c_out: usize,
    time: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
) -> Vec<f32> {
    let t_out = (time + 2 * padding - kernel_size) / stride + 1;
    let mut output = vec![0.0_f32; batch * c_out * t_out];
    for b in 0..batch {
        for oc in 0..c_out {
            for t in 0..t_out {
                let mut sum = bias[oc];
                for ic in 0..c_in {
                    for k in 0..kernel_size {
                        let t_in = (t * stride + k) as isize - padding as isize;
                        if t_in >= 0 && (t_in as usize) < time {
                            let w_idx = oc * c_in * kernel_size + ic * kernel_size + k;
                            let x_idx = b * c_in * time + ic * time + t_in as usize;
                            sum += weight[w_idx] * input[x_idx];
                        }
                    }
                }
                output[b * c_out * t_out + oc * t_out + t] = sum;
            }
        }
    }
    output
}

/// Full CPU reference: Upsample1d -> Conv1d.
fn cpu_upsample_conv1d(
    x: &[f32],
    conv_weight: &[f32],
    conv_bias: &[f32],
    batch: usize,
    c_in: usize,
    c_out: usize,
    time: usize,
    factor: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
) -> Vec<f32> {
    let upsampled = cpu_upsample_nearest_1d(x, batch, c_in, time, factor);
    let up_time = time * factor;
    cpu_conv1d(
        &upsampled,
        conv_weight,
        conv_bias,
        batch,
        c_in,
        c_out,
        up_time,
        kernel_size,
        padding,
        stride,
    )
}

// -- Tests --------------------------------------------------------------------

/// [1, 4, 16] -> Upsample1d(factor=2) -> Conv1d(k=3, pad=1, stride=1).
///
/// Constructs a trace graph where the peephole pass should detect the
/// Upsample1d + Conv1d pair and fuse into `NativeOpKind::FusedUpsampleConv1d`.
/// Verifies fused GPU output matches decomposed CPU reference.
#[test]
fn test_compiled_fused_upsample_conv1d_basic() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, c_in, time) = (1, 4, 16);
    let c_out = 8;
    let factor = 2;
    let kernel_size = 3;
    let padding = 1;
    let stride = 1;
    let up_time = time * factor;
    let t_out = (up_time + 2 * padding - kernel_size) / stride + 1;

    let x_data = super::test_utils::rand_f32_vec(0xFC01_0001, batch * c_in * time, -1.0, 1.0);
    let conv_w_data =
        super::test_utils::rand_f32_vec(0xFC01_0002, c_out * c_in * kernel_size, -0.5, 0.5);
    let conv_b_data = super::test_utils::rand_f32_vec(0xFC01_0003, c_out, -0.1, 0.1);

    let conv_weight =
        WeightRef::new(conv_w_data.clone(), vec![c_out, c_in, kernel_size]).expect("conv weight");
    let conv_bias = WeightRef::new(conv_b_data.clone(), vec![c_out]).expect("conv bias");

    // Trace graph: input -> Upsample1d -> Conv1d.
    // The peephole should fuse these into FusedUpsampleConv1d.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, c_in, time]),
        TraceNode::new(
            1,
            "upsample1d_0".into(),
            TraceOp::Upsample1d { factor },
            vec![0],
            vec![batch, c_in, up_time],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "conv1d_0".into(),
            TraceOp::Conv1d {
                weight: conv_weight,
                bias: Some(conv_bias),
                stride,
                padding,
                dilation: 1,
                groups: 1,
            },
            vec![1],
            vec![batch, c_out, t_out],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &x_data);

    let result = compile_and_run(&cache, graph, &[&x_buf], batch * c_out * t_out);

    let expected = cpu_upsample_conv1d(
        &x_data,
        &conv_w_data,
        &conv_b_data,
        batch,
        c_in,
        c_out,
        time,
        factor,
        kernel_size,
        padding,
        stride,
    );
    assert_close("fused_upsample_conv1d_basic", &result, &expected, 1e-3);
}

/// [1, 4, 8] -> Upsample1d(factor=4) -> Conv1d(k=3, pad=1, stride=1).
///
/// Tests with a larger upsample factor (4x), matching the Kokoro f0_energy
/// pattern where 6 such pairs exist.
#[test]
fn test_compiled_fused_upsample_conv1d_factor4() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, c_in, time) = (1, 4, 8);
    let c_out = 8;
    let factor = 4;
    let kernel_size = 3;
    let padding = 1;
    let stride = 1;
    let up_time = time * factor;
    let t_out = (up_time + 2 * padding - kernel_size) / stride + 1;

    let x_data = super::test_utils::rand_f32_vec(0xFC02_0001, batch * c_in * time, -1.0, 1.0);
    let conv_w_data =
        super::test_utils::rand_f32_vec(0xFC02_0002, c_out * c_in * kernel_size, -0.5, 0.5);
    let conv_b_data = super::test_utils::rand_f32_vec(0xFC02_0003, c_out, -0.1, 0.1);

    let conv_weight =
        WeightRef::new(conv_w_data.clone(), vec![c_out, c_in, kernel_size]).expect("conv weight");
    let conv_bias = WeightRef::new(conv_b_data.clone(), vec![c_out]).expect("conv bias");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, c_in, time]),
        TraceNode::new(
            1,
            "upsample1d_0".into(),
            TraceOp::Upsample1d { factor },
            vec![0],
            vec![batch, c_in, up_time],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "conv1d_0".into(),
            TraceOp::Conv1d {
                weight: conv_weight,
                bias: Some(conv_bias),
                stride,
                padding,
                dilation: 1,
                groups: 1,
            },
            vec![1],
            vec![batch, c_out, t_out],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &x_data);

    let result = compile_and_run(&cache, graph, &[&x_buf], batch * c_out * t_out);

    let expected = cpu_upsample_conv1d(
        &x_data,
        &conv_w_data,
        &conv_b_data,
        batch,
        c_in,
        c_out,
        time,
        factor,
        kernel_size,
        padding,
        stride,
    );
    assert_close("fused_upsample_conv1d_factor4", &result, &expected, 1e-3);
}

/// [2, 8, 16] -> Upsample1d(factor=2) -> Conv1d(k=3, pad=1): batched.
///
/// Larger dimensions to test threadgroup dispatch with multiple batches.
#[test]
fn test_compiled_fused_upsample_conv1d_batched() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, c_in, time) = (2, 8, 16);
    let c_out = 16;
    let factor = 2;
    let kernel_size = 3;
    let padding = 1;
    let stride = 1;
    let up_time = time * factor;
    let t_out = (up_time + 2 * padding - kernel_size) / stride + 1;

    let x_data = super::test_utils::rand_f32_vec(0xFC03_0001, batch * c_in * time, -2.0, 2.0);
    let conv_w_data =
        super::test_utils::rand_f32_vec(0xFC03_0002, c_out * c_in * kernel_size, -0.3, 0.3);
    let conv_b_data = super::test_utils::rand_f32_vec(0xFC03_0003, c_out, -0.1, 0.1);

    let conv_weight =
        WeightRef::new(conv_w_data.clone(), vec![c_out, c_in, kernel_size]).expect("conv weight");
    let conv_bias = WeightRef::new(conv_b_data.clone(), vec![c_out]).expect("conv bias");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, c_in, time]),
        TraceNode::new(
            1,
            "upsample1d_0".into(),
            TraceOp::Upsample1d { factor },
            vec![0],
            vec![batch, c_in, up_time],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "conv1d_0".into(),
            TraceOp::Conv1d {
                weight: conv_weight,
                bias: Some(conv_bias),
                stride,
                padding,
                dilation: 1,
                groups: 1,
            },
            vec![1],
            vec![batch, c_out, t_out],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &x_data);

    let result = compile_and_run(&cache, graph, &[&x_buf], batch * c_out * t_out);

    let expected = cpu_upsample_conv1d(
        &x_data,
        &conv_w_data,
        &conv_b_data,
        batch,
        c_in,
        c_out,
        time,
        factor,
        kernel_size,
        padding,
        stride,
    );
    assert_close("fused_upsample_conv1d_batched", &result, &expected, 1e-3);
}
