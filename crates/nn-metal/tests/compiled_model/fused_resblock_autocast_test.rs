// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end autocast test for `NativeOpKind::FusedResBlock`.
//!
//! FusedResBlock is classified as `is_compute_native_op()` (runs F16 in
//! autocast), but had ZERO F16 E2E tests prior to this file. It is the
//! dominant compute in Kokoro (35 blocks per forward pass). The executor
//! sequences 2× NormActivConv1d + residual add, and buffer handoff or
//! residual accumulation could have F16 bugs that the standalone
//! NormActivConv1d autocast test would not catch.
//!
//! Test builds a trace graph that the peephole fuses into FusedResBlock:
//!   x → AdainSnake(γ1,β1) → Conv1d → AdainSnake(γ2,β2) → Conv1d → Add(x,h)
//!
//! Part of #3299.

use nn_core::dyn_tensor::trace::{ComputationGraph, KokoroFusedOp, TraceNode, TraceOp, WeightRef};
use nn_core::mixed_precision::MixedPrecisionPolicy;
use nn_core::DType;
use nn_metal::compiled_model::CompiledModel;

use super::helpers::{assert_close, create_input_buffer, input_node, read_output_n};
use super::test_utils;

// -- CPU reference helpers ----------------------------------------------------

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

fn cpu_snake(x: f32, alpha: f32) -> f32 {
    let a = alpha.max(1e-8);
    let sin_val = (a * x).sin();
    x + (1.0 / a) * sin_val * sin_val
}

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
            let a = alpha[c];
            let offset = (b * channels + c) * time;
            for t in 0..time {
                let y = (1.0 + g) * normed[offset + t] + be;
                output[offset + t] = cpu_snake(y, a);
            }
        }
    }
    output
}

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

/// CPU reference for the full FusedResBlock: 2× (AdainSnake + Conv1d) + residual add.
fn cpu_fused_resblock(
    x: &[f32],
    gamma1: &[f32],
    beta1: &[f32],
    alpha1: &[f32],
    conv1_weight: &[f32],
    conv1_bias: &[f32],
    gamma2: &[f32],
    beta2: &[f32],
    alpha2: &[f32],
    conv2_weight: &[f32],
    conv2_bias: &[f32],
    batch: usize,
    channels: usize,
    time: usize,
    kernel_size: usize,
    padding: usize,
    dilation: usize,
    eps: f32,
) -> Vec<f32> {
    // Phase 1: AdainSnake → Conv1d
    let h1 = cpu_adain_snake(x, gamma1, beta1, alpha1, batch, channels, time, eps);
    let h1_conv = cpu_conv1d(
        &h1,
        conv1_weight,
        conv1_bias,
        batch,
        channels,
        channels,
        time,
        kernel_size,
        padding,
        dilation,
    );
    let t_out = time + 2 * padding - dilation * (kernel_size - 1);
    // Phase 2: AdainSnake → Conv1d
    let h2 = cpu_adain_snake(&h1_conv, gamma2, beta2, alpha2, batch, channels, t_out, eps);
    let h2_conv = cpu_conv1d(
        &h2,
        conv2_weight,
        conv2_bias,
        batch,
        channels,
        channels,
        t_out,
        kernel_size,
        1, // phase 2 padding
        1, // phase 2 dilation always 1
    );
    let t_final = t_out + 2 - (kernel_size - 1);
    // Residual add: x + h2_conv (requires same shape)
    // For same-shape residual: padding/dilation must preserve time dimension.
    assert_eq!(
        t_final, time,
        "conv params must preserve time for residual add"
    );
    let n = batch * channels * time;
    let mut output = vec![0.0_f32; n];
    for i in 0..n {
        output[i] = x[i] + h2_conv[i];
    }
    output
}

// -- Tests --------------------------------------------------------------------

/// FusedResBlock autocast: F16 vs F32 parity.
///
/// Builds a trace graph that peephole-fuses into FusedResBlock, then
/// verifies autocast F16 execution matches the F32 baseline.
#[test]
fn test_autocast_fused_resblock_snake() {
    test_utils::gpu_init();
    let cache = test_utils::metal_setup();

    let (batch, channels, time) = (1, 4, 16);
    let kernel_size = 3;
    // padding = dilation for phase 1 (Kokoro DConv pattern), padding=1/dilation=1 for phase 2.
    // Both must preserve time dimension for the residual add.
    let p1_dilation = 1;
    let p1_padding = 1;
    let eps = 1e-5_f64;

    // Random data with fixed seeds for reproducibility.
    let x_data = test_utils::rand_f32_vec(0xFB01_0001, batch * channels * time, -1.0, 1.0);
    let gamma1_data = test_utils::rand_f32_vec(0xFB01_0002, batch * channels, -0.3, 0.3);
    let beta1_data = test_utils::rand_f32_vec(0xFB01_0003, batch * channels, -0.2, 0.2);
    let gamma2_data = test_utils::rand_f32_vec(0xFB01_0004, batch * channels, -0.3, 0.3);
    let beta2_data = test_utils::rand_f32_vec(0xFB01_0005, batch * channels, -0.2, 0.2);
    let alpha1_data = test_utils::rand_f32_vec(0xFB01_0006, channels, 0.5, 2.0);
    let alpha2_data = test_utils::rand_f32_vec(0xFB01_0007, channels, 0.5, 2.0);
    let conv1_w_data =
        test_utils::rand_f32_vec(0xFB01_0008, channels * channels * kernel_size, -0.5, 0.5);
    let conv1_b_data = test_utils::rand_f32_vec(0xFB01_0009, channels, -0.1, 0.1);
    let conv2_w_data =
        test_utils::rand_f32_vec(0xFB01_000A, channels * channels * kernel_size, -0.5, 0.5);
    let conv2_b_data = test_utils::rand_f32_vec(0xFB01_000B, channels, -0.1, 0.1);

    // WeightRefs for Conv1d nodes.
    let alpha1_weight = WeightRef::new(alpha1_data.clone(), vec![channels]).expect("alpha1");
    let conv1_weight = WeightRef::new(conv1_w_data.clone(), vec![channels, channels, kernel_size])
        .expect("conv1 weight");
    let conv1_bias = WeightRef::new(conv1_b_data.clone(), vec![channels]).expect("conv1 bias");

    let alpha2_weight = WeightRef::new(alpha2_data.clone(), vec![channels]).expect("alpha2");
    let conv2_weight = WeightRef::new(conv2_w_data.clone(), vec![channels, channels, kernel_size])
        .expect("conv2 weight");
    let conv2_bias = WeightRef::new(conv2_b_data.clone(), vec![channels]).expect("conv2 bias");

    // Trace graph: x → AdainSnake(γ1,β1) → Conv1d → AdainSnake(γ2,β2) → Conv1d → Add(x,h)
    // Peephole pass 1 fuses each AdainSnake+Conv1d into NormActivConv1d.
    // Peephole pass 2 fuses 2× NormActivConv1d + Add into FusedResBlock.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, channels, time]), // x
        input_node(1, &[batch, channels, 1]),    // gamma1
        input_node(2, &[batch, channels, 1]),    // beta1
        input_node(3, &[batch, channels, 1]),    // gamma2
        input_node(4, &[batch, channels, 1]),    // beta2
        // Phase 1: AdainSnake → Conv1d
        TraceNode::new(
            5,
            "adain_snake_p1".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdainSnake {
                alpha: alpha1_weight,
                eps,
            }),
            vec![0, 1, 2],
            vec![batch, channels, time],
            DType::F32,
        ),
        TraceNode::new(
            6,
            "conv1d_p1".into(),
            TraceOp::Conv1d {
                weight: conv1_weight,
                bias: Some(conv1_bias),
                stride: 1,
                padding: p1_padding,
                dilation: p1_dilation,
                groups: 1,
            },
            vec![5],
            vec![batch, channels, time], // padding preserves time
            DType::F32,
        ),
        // Phase 2: AdainSnake → Conv1d
        TraceNode::new(
            7,
            "adain_snake_p2".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdainSnake {
                alpha: alpha2_weight,
                eps,
            }),
            vec![6, 3, 4],
            vec![batch, channels, time],
            DType::F32,
        ),
        TraceNode::new(
            8,
            "conv1d_p2".into(),
            TraceOp::Conv1d {
                weight: conv2_weight,
                bias: Some(conv2_bias),
                stride: 1,
                padding: 1,
                dilation: 1,
                groups: 1,
            },
            vec![7],
            vec![batch, channels, time],
            DType::F32,
        ),
        // Residual add: x + conv1d_p2_output
        TraceNode::new(
            9,
            "add_residual".into(),
            TraceOp::Add,
            vec![0, 8],
            vec![batch, channels, time],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &x_data);
    let gamma1_buf = create_input_buffer(&cache, &gamma1_data);
    let beta1_buf = create_input_buffer(&cache, &beta1_data);
    let gamma2_buf = create_input_buffer(&cache, &gamma2_data);
    let beta2_buf = create_input_buffer(&cache, &beta2_data);
    let inputs: &[&nn_metal::MetalBuffer] =
        &[&x_buf, &gamma1_buf, &beta1_buf, &gamma2_buf, &beta2_buf];

    let output_numel = batch * channels * time;

    // F32 baseline.
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f32_buf = f32_model.execute(&cache, inputs).expect("f32 exec");
    let f32_result = read_output_n(&f32_buf, output_numel);

    // CPU reference for sanity check.
    let cpu_expected = cpu_fused_resblock(
        &x_data,
        &gamma1_data,
        &beta1_data,
        &alpha1_data,
        &conv1_w_data,
        &conv1_b_data,
        &gamma2_data,
        &beta2_data,
        &alpha2_data,
        &conv2_w_data,
        &conv2_b_data,
        batch,
        channels,
        time,
        kernel_size,
        p1_padding,
        p1_dilation,
        eps as f32,
    );
    assert_close(
        "fused_resblock_f32_vs_cpu",
        &f32_result,
        &cpu_expected,
        2e-3,
    );

    // Autocast (F16 for FusedResBlock via is_compute_native_op).
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert!(ac_model.is_autocast(), "model should be autocast");
    assert!(
        ac_model.num_autocast_f16_steps() > 0,
        "FusedResBlock should be classified as F16 step"
    );
    let ac_buf = ac_model.execute(&cache, inputs).expect("autocast exec");
    let ac_result = read_output_n(&ac_buf, output_numel);

    // F16 tolerance: FusedResBlock uses float accumulators in NormActivConv1d
    // kernels, but two sequential phases + residual add accumulate error.
    assert_close("autocast_fused_resblock", &ac_result, &f32_result, 1e-2);
}

/// FusedResBlock autocast with batched input [2, 8, 32].
///
/// Exercises larger dimensions to test threadgroup dispatch and
/// multi-batch buffer indexing under F16.
#[test]
fn test_autocast_fused_resblock_snake_batched() {
    test_utils::gpu_init();
    let cache = test_utils::metal_setup();

    let (batch, channels, time) = (2, 8, 32);
    let kernel_size = 3;
    let eps = 1e-5_f64;

    let x_data = test_utils::rand_f32_vec(0xFB02_0001, batch * channels * time, -2.0, 2.0);
    let gamma1_data = test_utils::rand_f32_vec(0xFB02_0002, batch * channels, -0.3, 0.3);
    let beta1_data = test_utils::rand_f32_vec(0xFB02_0003, batch * channels, -0.2, 0.2);
    let gamma2_data = test_utils::rand_f32_vec(0xFB02_0004, batch * channels, -0.3, 0.3);
    let beta2_data = test_utils::rand_f32_vec(0xFB02_0005, batch * channels, -0.2, 0.2);
    let alpha1_data = test_utils::rand_f32_vec(0xFB02_0006, channels, 0.5, 2.0);
    let alpha2_data = test_utils::rand_f32_vec(0xFB02_0007, channels, 0.5, 2.0);
    let conv1_w_data =
        test_utils::rand_f32_vec(0xFB02_0008, channels * channels * kernel_size, -0.3, 0.3);
    let conv1_b_data = test_utils::rand_f32_vec(0xFB02_0009, channels, -0.1, 0.1);
    let conv2_w_data =
        test_utils::rand_f32_vec(0xFB02_000A, channels * channels * kernel_size, -0.3, 0.3);
    let conv2_b_data = test_utils::rand_f32_vec(0xFB02_000B, channels, -0.1, 0.1);

    let alpha1_weight = WeightRef::new(alpha1_data, vec![channels]).expect("alpha1");
    let conv1_weight =
        WeightRef::new(conv1_w_data, vec![channels, channels, kernel_size]).expect("conv1 weight");
    let conv1_bias = WeightRef::new(conv1_b_data, vec![channels]).expect("conv1 bias");
    let alpha2_weight = WeightRef::new(alpha2_data, vec![channels]).expect("alpha2");
    let conv2_weight =
        WeightRef::new(conv2_w_data, vec![channels, channels, kernel_size]).expect("conv2 weight");
    let conv2_bias = WeightRef::new(conv2_b_data, vec![channels]).expect("conv2 bias");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, channels, time]),
        input_node(1, &[batch, channels, 1]),
        input_node(2, &[batch, channels, 1]),
        input_node(3, &[batch, channels, 1]),
        input_node(4, &[batch, channels, 1]),
        TraceNode::new(
            5,
            "adain_snake_p1".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdainSnake {
                alpha: alpha1_weight,
                eps,
            }),
            vec![0, 1, 2],
            vec![batch, channels, time],
            DType::F32,
        ),
        TraceNode::new(
            6,
            "conv1d_p1".into(),
            TraceOp::Conv1d {
                weight: conv1_weight,
                bias: Some(conv1_bias),
                stride: 1,
                padding: 1,
                dilation: 1,
                groups: 1,
            },
            vec![5],
            vec![batch, channels, time],
            DType::F32,
        ),
        TraceNode::new(
            7,
            "adain_snake_p2".into(),
            TraceOp::KokoroFused(KokoroFusedOp::AdainSnake {
                alpha: alpha2_weight,
                eps,
            }),
            vec![6, 3, 4],
            vec![batch, channels, time],
            DType::F32,
        ),
        TraceNode::new(
            8,
            "conv1d_p2".into(),
            TraceOp::Conv1d {
                weight: conv2_weight,
                bias: Some(conv2_bias),
                stride: 1,
                padding: 1,
                dilation: 1,
                groups: 1,
            },
            vec![7],
            vec![batch, channels, time],
            DType::F32,
        ),
        TraceNode::new(
            9,
            "add_residual".into(),
            TraceOp::Add,
            vec![0, 8],
            vec![batch, channels, time],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &x_data);
    let gamma1_buf = create_input_buffer(&cache, &gamma1_data);
    let beta1_buf = create_input_buffer(&cache, &beta1_data);
    let gamma2_buf = create_input_buffer(&cache, &gamma2_data);
    let beta2_buf = create_input_buffer(&cache, &beta2_data);
    let inputs: &[&nn_metal::MetalBuffer] =
        &[&x_buf, &gamma1_buf, &beta1_buf, &gamma2_buf, &beta2_buf];

    let output_numel = batch * channels * time;

    // F32 baseline.
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f32_buf = f32_model.execute(&cache, inputs).expect("f32 exec");
    let f32_result = read_output_n(&f32_buf, output_numel);

    // Autocast.
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert!(ac_model.is_autocast());
    assert!(
        ac_model.num_autocast_f16_steps() > 0,
        "FusedResBlock should have F16 steps"
    );
    let ac_buf = ac_model.execute(&cache, inputs).expect("autocast exec");
    let ac_result = read_output_n(&ac_buf, output_numel);

    assert_close(
        "autocast_fused_resblock_batched",
        &ac_result,
        &f32_result,
        1e-2,
    );
}
