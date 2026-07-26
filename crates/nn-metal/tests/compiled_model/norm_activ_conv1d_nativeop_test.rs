// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end compiled model tests for `NativeOpKind::NormActivConv1d`.
//!
//! The peephole pass fuses `AdainLeakyRelu + Conv1d` into a single
//! `NormActivConv1d` NativeOp that uses the fused MSL kernel
//! (`native_norm_activ_conv1d`). This kernel computes InstanceNorm +
//! style affine + LeakyRelu + Conv1d in a single dispatch, eliminating
//! the intermediate activated tensor.
//!
//! This test verifies the full pipeline: trace graph with AdainLeakyRelu →
//! Conv1d → peephole fuses into NormActivConv1d → GPU execute → verify
//! against CPU reference.
//!
//! **Why this test matters:** The FusedResBlock equivalence test
//! (`compiled_resblock_equivalence.rs`) exercises the FusedResBlock executor
//! which uses decomposed `run_norm_activ` + `run_conv1d` (separate dispatches).
//! The standalone NormActivConv1d code path uses a DIFFERENT fused MSL kernel
//! (`native_norm_activ_conv1d`). Without this test, that fused kernel has no
//! compiled-model-level parity verification.
//!
//! Part of #2218 (Kokoro epic).

use nn_core::dyn_tensor::trace::{ComputationGraph, KokoroFusedOp, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use super::helpers::{assert_close, compile_and_run, create_input_buffer, input_node};

// -- CPU reference helpers ----------------------------------------------------

/// CPU InstanceNorm: normalize each (batch, channel) slice independently.
fn cpu_instance_norm(
    input: &[f32],
    batch: usize,
    channels: usize,
    time: usize,
    eps: f32,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; batch * channels * time];
    for b in 0..batch {
        for c in 0..channels {
            let offset = (b * channels + c) * time;
            let slice = &input[offset..offset + time];
            let mean: f32 = slice.iter().sum::<f32>() / time as f32;
            let var: f32 = slice.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / time as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for t in 0..time {
                output[offset + t] = (slice[t] - mean) * inv_std;
            }
        }
    }
    output
}

/// CPU AdainLeakyRelu: InstanceNorm → affine → leaky_relu.
fn cpu_adain_leaky_relu(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    batch: usize,
    channels: usize,
    time: usize,
    eps: f32,
    slope: f32,
) -> Vec<f32> {
    let normed = cpu_instance_norm(x, batch, channels, time, eps);
    let mut output = vec![0.0_f32; batch * channels * time];
    for b in 0..batch {
        for c in 0..channels {
            let g = gamma[b * channels + c];
            let be = beta[b * channels + c];
            let offset = (b * channels + c) * time;
            for t in 0..time {
                let y = (1.0 + g) * normed[offset + t] + be;
                output[offset + t] = if y >= 0.0 { y } else { slope * y };
            }
        }
    }
    output
}

/// CPU Conv1d: stride=1, groups=1, with padding and dilation.
///
/// Input: `[B, C_in, T]`, Weight: `[C_out, C_in, K]`, Bias: `[C_out]`.
/// Output: `[B, C_out, T_out]` where `T_out = T + 2*padding - dilation*(K-1) - 1 + 1`.
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
    dilation: usize,
) -> Vec<f32> {
    let t_out = time + 2 * padding - dilation * (kernel_size - 1);
    let mut output = vec![0.0_f32; batch * c_out * t_out];
    for b in 0..batch {
        for oc in 0..c_out {
            for t in 0..t_out {
                let mut sum = bias[oc];
                for ic in 0..c_in {
                    for k in 0..kernel_size {
                        let t_in = t as isize + (k * dilation) as isize - padding as isize;
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

/// Full CPU reference: AdainLeakyRelu → Conv1d (the decomposed computation
/// that the fused NormActivConv1d kernel should match).
fn cpu_norm_activ_conv1d(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    conv_weight: &[f32],
    conv_bias: &[f32],
    batch: usize,
    c_in: usize,
    c_out: usize,
    time: usize,
    kernel_size: usize,
    padding: usize,
    dilation: usize,
    eps: f32,
    slope: f32,
) -> Vec<f32> {
    let activated = cpu_adain_leaky_relu(x, gamma, beta, batch, c_in, time, eps, slope);
    cpu_conv1d(
        &activated,
        conv_weight,
        conv_bias,
        batch,
        c_in,
        c_out,
        time,
        kernel_size,
        padding,
        dilation,
    )
}

// -- Tests --------------------------------------------------------------------

/// [1, 4, 16] → AdainLeakyRelu → Conv1d(k=3, pad=1): fused NormActivConv1d.
///
/// Constructs a trace graph where the peephole pass should detect the
/// AdainLeakyRelu + Conv1d pair and fuse into `NativeOpKind::NormActivConv1d`.
/// Verifies fused GPU output matches decomposed CPU reference.
#[test]
fn test_compiled_norm_activ_conv1d_leaky_relu() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, c_in, time) = (1, 4, 16);
    let c_out = 4;
    let kernel_size = 3;
    let padding = 1;
    let dilation = 1;
    let eps = 1e-5_f64;
    let slope = 0.2_f64;
    let t_out = time + 2 * padding - dilation * (kernel_size - 1);

    let x_data = super::test_utils::rand_f32_vec(0xAC01_0001, batch * c_in * time, -1.0, 1.0);
    let gamma_data = super::test_utils::rand_f32_vec(0xAC01_0002, batch * c_in, -0.3, 0.3);
    let beta_data = super::test_utils::rand_f32_vec(0xAC01_0003, batch * c_in, -0.2, 0.2);
    let conv_w_data =
        super::test_utils::rand_f32_vec(0xAC01_0004, c_out * c_in * kernel_size, -0.5, 0.5);
    let conv_b_data = super::test_utils::rand_f32_vec(0xAC01_0005, c_out, -0.1, 0.1);

    let conv_weight =
        WeightRef::new(conv_w_data.clone(), vec![c_out, c_in, kernel_size]).expect("conv weight");
    let conv_bias = WeightRef::new(conv_b_data.clone(), vec![c_out]).expect("conv bias");

    // Trace graph: input → AdainLeakyRelu → Conv1d.
    // The peephole should fuse these into NormActivConv1d.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, c_in, time]), // x
        input_node(1, &[batch, c_in, 1]),    // gamma
        input_node(2, &[batch, c_in, 1]),    // beta
        TraceNode::new(
            3,
            "adain_leaky_relu_0".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdainLeakyRelu { eps, slope }),
            vec![0, 1, 2],
            vec![batch, c_in, time],
            DType::F32,
        ),
        TraceNode::new(
            4,
            "conv1d_0".into(),
            TraceOp::Conv1d {
                weight: conv_weight,
                bias: Some(conv_bias),
                stride: 1,
                padding,
                dilation,
                groups: 1,
            },
            vec![3],
            vec![batch, c_out, t_out],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &x_data);
    let gamma_buf = create_input_buffer(&cache, &gamma_data);
    let beta_buf = create_input_buffer(&cache, &beta_data);

    let result = compile_and_run(
        &cache,
        graph,
        &[&x_buf, &gamma_buf, &beta_buf],
        batch * c_out * t_out,
    );

    let expected = cpu_norm_activ_conv1d(
        &x_data,
        &gamma_data,
        &beta_data,
        &conv_w_data,
        &conv_b_data,
        batch,
        c_in,
        c_out,
        time,
        kernel_size,
        padding,
        dilation,
        eps as f32,
        slope as f32,
    );
    assert_close("norm_activ_conv1d_leaky_relu", &result, &expected, 1e-3);
}

/// [2, 8, 32] → AdainLeakyRelu → Conv1d(k=3, pad=1): batched fused kernel.
///
/// Larger dimensions to test threadgroup dispatch with multiple batches.
#[test]
fn test_compiled_norm_activ_conv1d_leaky_relu_batched() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, c_in, time) = (2, 8, 32);
    let c_out = 8;
    let kernel_size = 3;
    let padding = 1;
    let dilation = 1;
    let eps = 1e-5_f64;
    let slope = 0.2_f64;
    let t_out = time + 2 * padding - dilation * (kernel_size - 1);

    let x_data = super::test_utils::rand_f32_vec(0xAC02_0001, batch * c_in * time, -2.0, 2.0);
    let gamma_data = super::test_utils::rand_f32_vec(0xAC02_0002, batch * c_in, -0.3, 0.3);
    let beta_data = super::test_utils::rand_f32_vec(0xAC02_0003, batch * c_in, -0.2, 0.2);
    let conv_w_data =
        super::test_utils::rand_f32_vec(0xAC02_0004, c_out * c_in * kernel_size, -0.3, 0.3);
    let conv_b_data = super::test_utils::rand_f32_vec(0xAC02_0005, c_out, -0.1, 0.1);

    let conv_weight =
        WeightRef::new(conv_w_data.clone(), vec![c_out, c_in, kernel_size]).expect("conv weight");
    let conv_bias = WeightRef::new(conv_b_data.clone(), vec![c_out]).expect("conv bias");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, c_in, time]),
        input_node(1, &[batch, c_in, 1]),
        input_node(2, &[batch, c_in, 1]),
        TraceNode::new(
            3,
            "adain_leaky_relu_0".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdainLeakyRelu { eps, slope }),
            vec![0, 1, 2],
            vec![batch, c_in, time],
            DType::F32,
        ),
        TraceNode::new(
            4,
            "conv1d_0".into(),
            TraceOp::Conv1d {
                weight: conv_weight,
                bias: Some(conv_bias),
                stride: 1,
                padding,
                dilation,
                groups: 1,
            },
            vec![3],
            vec![batch, c_out, t_out],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &x_data);
    let gamma_buf = create_input_buffer(&cache, &gamma_data);
    let beta_buf = create_input_buffer(&cache, &beta_data);

    let result = compile_and_run(
        &cache,
        graph,
        &[&x_buf, &gamma_buf, &beta_buf],
        batch * c_out * t_out,
    );

    let expected = cpu_norm_activ_conv1d(
        &x_data,
        &gamma_data,
        &beta_data,
        &conv_w_data,
        &conv_b_data,
        batch,
        c_in,
        c_out,
        time,
        kernel_size,
        padding,
        dilation,
        eps as f32,
        slope as f32,
    );
    assert_close(
        "norm_activ_conv1d_leaky_relu_batched",
        &result,
        &expected,
        1e-3,
    );
}

/// [1, 4, 16] → AdainLeakyRelu → Conv1d(k=3, pad=2, dilation=2): dilated conv.
///
/// Tests the fused kernel with dilation > 1 (used in Kokoro DConv layers).
#[test]
fn test_compiled_norm_activ_conv1d_dilated() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, c_in, time) = (1, 4, 16);
    let c_out = 4;
    let kernel_size = 3;
    let padding = 2;
    let dilation = 2;
    let eps = 1e-5_f64;
    let slope = 0.2_f64;
    let t_out = time + 2 * padding - dilation * (kernel_size - 1);

    let x_data = super::test_utils::rand_f32_vec(0xAC03_0001, batch * c_in * time, -1.0, 1.0);
    let gamma_data = super::test_utils::rand_f32_vec(0xAC03_0002, batch * c_in, -0.3, 0.3);
    let beta_data = super::test_utils::rand_f32_vec(0xAC03_0003, batch * c_in, -0.2, 0.2);
    let conv_w_data =
        super::test_utils::rand_f32_vec(0xAC03_0004, c_out * c_in * kernel_size, -0.5, 0.5);
    let conv_b_data = super::test_utils::rand_f32_vec(0xAC03_0005, c_out, -0.1, 0.1);

    let conv_weight =
        WeightRef::new(conv_w_data.clone(), vec![c_out, c_in, kernel_size]).expect("conv weight");
    let conv_bias = WeightRef::new(conv_b_data.clone(), vec![c_out]).expect("conv bias");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, c_in, time]),
        input_node(1, &[batch, c_in, 1]),
        input_node(2, &[batch, c_in, 1]),
        TraceNode::new(
            3,
            "adain_leaky_relu_0".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdainLeakyRelu { eps, slope }),
            vec![0, 1, 2],
            vec![batch, c_in, time],
            DType::F32,
        ),
        TraceNode::new(
            4,
            "conv1d_0".into(),
            TraceOp::Conv1d {
                weight: conv_weight,
                bias: Some(conv_bias),
                stride: 1,
                padding,
                dilation,
                groups: 1,
            },
            vec![3],
            vec![batch, c_out, t_out],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &x_data);
    let gamma_buf = create_input_buffer(&cache, &gamma_data);
    let beta_buf = create_input_buffer(&cache, &beta_data);

    let result = compile_and_run(
        &cache,
        graph,
        &[&x_buf, &gamma_buf, &beta_buf],
        batch * c_out * t_out,
    );

    let expected = cpu_norm_activ_conv1d(
        &x_data,
        &gamma_data,
        &beta_data,
        &conv_w_data,
        &conv_b_data,
        batch,
        c_in,
        c_out,
        time,
        kernel_size,
        padding,
        dilation,
        eps as f32,
        slope as f32,
    );
    assert_close("norm_activ_conv1d_dilated", &result, &expected, 1e-3);
}

// -- CPU reference: Snake activation ------------------------------------------

/// CPU Snake activation: `x + (1/alpha) * sin(alpha * x)^2`.
fn cpu_snake(x: f32, alpha: f32) -> f32 {
    let a = alpha.max(1e-8);
    let sin_val = (a * x).sin();
    x + (1.0 / a) * sin_val * sin_val
}

/// CPU AdainSnake: InstanceNorm → affine → Snake.
fn cpu_adain_snake(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    alpha: &[f32],
    batch: usize,
    channels: usize,
    time: usize,
    eps: f32,
) -> Vec<f32> {
    let normed = cpu_instance_norm(x, batch, channels, time, eps);
    let mut output = vec![0.0_f32; batch * channels * time];
    for b in 0..batch {
        for c in 0..channels {
            let g = gamma[b * channels + c];
            let be = beta[b * channels + c];
            let a = alpha[c]; // per-channel, not per-batch
            let offset = (b * channels + c) * time;
            for t in 0..time {
                let y = (1.0 + g) * normed[offset + t] + be;
                output[offset + t] = cpu_snake(y, a);
            }
        }
    }
    output
}

/// Full CPU reference: AdainSnake → Conv1d (decomposed computation for Snake).
fn cpu_norm_activ_conv1d_snake(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    alpha: &[f32],
    conv_weight: &[f32],
    conv_bias: &[f32],
    batch: usize,
    c_in: usize,
    c_out: usize,
    time: usize,
    kernel_size: usize,
    padding: usize,
    dilation: usize,
    eps: f32,
) -> Vec<f32> {
    let activated = cpu_adain_snake(x, gamma, beta, alpha, batch, c_in, time, eps);
    cpu_conv1d(
        &activated,
        conv_weight,
        conv_bias,
        batch,
        c_in,
        c_out,
        time,
        kernel_size,
        padding,
        dilation,
    )
}

// -- Tests: Snake NormActivConv1d ---------------------------------------------

/// [1, 4, 16] → AdainSnake → Conv1d(k=3, pad=1): fused NormActivConv1d Snake.
///
/// Verifies the Snake variant of the fused kernel produces output matching
/// the decomposed CPU reference (InstanceNorm → style affine → Snake → Conv1d).
/// Part of #2218 (requested by W10 b8c12de).
#[test]
fn test_compiled_norm_activ_conv1d_snake() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, c_in, time) = (1, 4, 16);
    let c_out = 4;
    let kernel_size = 3;
    let padding = 1;
    let dilation = 1;
    let eps = 1e-5_f64;
    let t_out = time + 2 * padding - dilation * (kernel_size - 1);

    let x_data = super::test_utils::rand_f32_vec(0xAC04_0001, batch * c_in * time, -1.0, 1.0);
    let gamma_data = super::test_utils::rand_f32_vec(0xAC04_0002, batch * c_in, -0.3, 0.3);
    let beta_data = super::test_utils::rand_f32_vec(0xAC04_0003, batch * c_in, -0.2, 0.2);
    let alpha_data = super::test_utils::rand_f32_vec(0xAC04_0004, c_in, 0.5, 2.0);
    let conv_w_data =
        super::test_utils::rand_f32_vec(0xAC04_0005, c_out * c_in * kernel_size, -0.5, 0.5);
    let conv_b_data = super::test_utils::rand_f32_vec(0xAC04_0006, c_out, -0.1, 0.1);

    let alpha_weight = WeightRef::new(alpha_data.clone(), vec![c_in]).expect("alpha weight");
    let conv_weight =
        WeightRef::new(conv_w_data.clone(), vec![c_out, c_in, kernel_size]).expect("conv weight");
    let conv_bias = WeightRef::new(conv_b_data.clone(), vec![c_out]).expect("conv bias");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, c_in, time]),
        input_node(1, &[batch, c_in, 1]),
        input_node(2, &[batch, c_in, 1]),
        TraceNode::new(
            3,
            "adain_snake_0".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdainSnake {
                alpha: alpha_weight,
                eps,
            }),
            vec![0, 1, 2],
            vec![batch, c_in, time],
            DType::F32,
        ),
        TraceNode::new(
            4,
            "conv1d_0".into(),
            TraceOp::Conv1d {
                weight: conv_weight,
                bias: Some(conv_bias),
                stride: 1,
                padding,
                dilation,
                groups: 1,
            },
            vec![3],
            vec![batch, c_out, t_out],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &x_data);
    let gamma_buf = create_input_buffer(&cache, &gamma_data);
    let beta_buf = create_input_buffer(&cache, &beta_data);

    let result = compile_and_run(
        &cache,
        graph,
        &[&x_buf, &gamma_buf, &beta_buf],
        batch * c_out * t_out,
    );

    let expected = cpu_norm_activ_conv1d_snake(
        &x_data,
        &gamma_data,
        &beta_data,
        &alpha_data,
        &conv_w_data,
        &conv_b_data,
        batch,
        c_in,
        c_out,
        time,
        kernel_size,
        padding,
        dilation,
        eps as f32,
    );
    assert_close("norm_activ_conv1d_snake", &result, &expected, 2e-3);
}

/// [2, 8, 32] → AdainSnake → Conv1d(k=3, pad=1): batched Snake fused kernel.
#[test]
fn test_compiled_norm_activ_conv1d_snake_batched() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, c_in, time) = (2, 8, 32);
    let c_out = 8;
    let kernel_size = 3;
    let padding = 1;
    let dilation = 1;
    let eps = 1e-5_f64;
    let t_out = time + 2 * padding - dilation * (kernel_size - 1);

    let x_data = super::test_utils::rand_f32_vec(0xAC05_0001, batch * c_in * time, -2.0, 2.0);
    let gamma_data = super::test_utils::rand_f32_vec(0xAC05_0002, batch * c_in, -0.3, 0.3);
    let beta_data = super::test_utils::rand_f32_vec(0xAC05_0003, batch * c_in, -0.2, 0.2);
    let alpha_data = super::test_utils::rand_f32_vec(0xAC05_0004, c_in, 0.5, 2.0);
    let conv_w_data =
        super::test_utils::rand_f32_vec(0xAC05_0005, c_out * c_in * kernel_size, -0.3, 0.3);
    let conv_b_data = super::test_utils::rand_f32_vec(0xAC05_0006, c_out, -0.1, 0.1);

    let alpha_weight = WeightRef::new(alpha_data.clone(), vec![c_in]).expect("alpha weight");
    let conv_weight =
        WeightRef::new(conv_w_data.clone(), vec![c_out, c_in, kernel_size]).expect("conv weight");
    let conv_bias = WeightRef::new(conv_b_data.clone(), vec![c_out]).expect("conv bias");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, c_in, time]),
        input_node(1, &[batch, c_in, 1]),
        input_node(2, &[batch, c_in, 1]),
        TraceNode::new(
            3,
            "adain_snake_0".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdainSnake {
                alpha: alpha_weight,
                eps,
            }),
            vec![0, 1, 2],
            vec![batch, c_in, time],
            DType::F32,
        ),
        TraceNode::new(
            4,
            "conv1d_0".into(),
            TraceOp::Conv1d {
                weight: conv_weight,
                bias: Some(conv_bias),
                stride: 1,
                padding,
                dilation,
                groups: 1,
            },
            vec![3],
            vec![batch, c_out, t_out],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &x_data);
    let gamma_buf = create_input_buffer(&cache, &gamma_data);
    let beta_buf = create_input_buffer(&cache, &beta_data);

    let result = compile_and_run(
        &cache,
        graph,
        &[&x_buf, &gamma_buf, &beta_buf],
        batch * c_out * t_out,
    );

    let expected = cpu_norm_activ_conv1d_snake(
        &x_data,
        &gamma_data,
        &beta_data,
        &alpha_data,
        &conv_w_data,
        &conv_b_data,
        batch,
        c_in,
        c_out,
        time,
        kernel_size,
        padding,
        dilation,
        eps as f32,
    );
    assert_close("norm_activ_conv1d_snake_batched", &result, &expected, 2e-3);
}

// -- Autocast parity tests ----------------------------------------------------

/// Autocast NormActivConv1d produces results within F16 tolerance of F32.
///
/// NormActivConv1d is the dominant compute in Kokoro's decoder (~24 phases).
/// After D1 added it to `is_compute_native_op()`, the autocast pipeline
/// classifies it for F16. This test verifies the F16 kernel path produces
/// correct results. Part of #2981.
#[test]
fn test_autocast_norm_activ_conv1d_parity() {
    use nn_core::mixed_precision::MixedPrecisionPolicy;
    use nn_metal::compiled_model::CompiledModel;

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, c_in, time) = (1, 4, 16);
    let c_out = 4;
    let kernel_size = 3;
    let padding = 1;
    let dilation = 1;
    let eps = 1e-5_f64;
    let slope = 0.2_f64;
    let t_out = time + 2 * padding - dilation * (kernel_size - 1);

    let x_data = super::test_utils::rand_f32_vec(0xAC06_0001, batch * c_in * time, -1.0, 1.0);
    let gamma_data = super::test_utils::rand_f32_vec(0xAC06_0002, batch * c_in, -0.3, 0.3);
    let beta_data = super::test_utils::rand_f32_vec(0xAC06_0003, batch * c_in, -0.2, 0.2);
    let conv_w_data =
        super::test_utils::rand_f32_vec(0xAC06_0004, c_out * c_in * kernel_size, -0.5, 0.5);
    let conv_b_data = super::test_utils::rand_f32_vec(0xAC06_0005, c_out, -0.1, 0.1);

    let conv_weight =
        WeightRef::new(conv_w_data, vec![c_out, c_in, kernel_size]).expect("conv weight");
    let conv_bias = WeightRef::new(conv_b_data, vec![c_out]).expect("conv bias");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, c_in, time]),
        input_node(1, &[batch, c_in, 1]),
        input_node(2, &[batch, c_in, 1]),
        TraceNode::new(
            3,
            "adain_leaky_relu_0".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdainLeakyRelu { eps, slope }),
            vec![0, 1, 2],
            vec![batch, c_in, time],
            DType::F32,
        ),
        TraceNode::new(
            4,
            "conv1d_0".into(),
            TraceOp::Conv1d {
                weight: conv_weight,
                bias: Some(conv_bias),
                stride: 1,
                padding,
                dilation,
                groups: 1,
            },
            vec![3],
            vec![batch, c_out, t_out],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &x_data);
    let gamma_buf = create_input_buffer(&cache, &gamma_data);
    let beta_buf = create_input_buffer(&cache, &beta_data);
    let inputs: &[&nn_metal::MetalBuffer] = &[&x_buf, &gamma_buf, &beta_buf];

    // F32 baseline.
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f32_buf = f32_model.execute(&cache, inputs).expect("f32 exec");
    let f32_result = super::helpers::read_output_n(&f32_buf, batch * c_out * t_out);

    // Autocast (F16 for NormActivConv1d).
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert!(ac_model.is_autocast(), "model should be autocast");
    assert!(
        ac_model.num_autocast_f16_steps() > 0,
        "NormActivConv1d should be classified as F16 step"
    );
    let ac_buf = ac_model.execute(&cache, inputs).expect("autocast exec");
    let ac_result = super::helpers::read_output_n(&ac_buf, batch * c_out * t_out);

    // F16 tolerance: NormActivConv1d uses float accumulators, so error is small.
    assert_close("autocast_norm_activ_conv1d", &ac_result, &f32_result, 1e-2);
}
