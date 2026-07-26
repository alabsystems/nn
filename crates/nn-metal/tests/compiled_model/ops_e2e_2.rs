// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end compiled model tests: Conv2d, LogSoftmax, activations, norms.
//!
//! Continuation of `compiled_model_ops_e2e.rs` (tests 12–19).
//! Exercises the full pipeline: build graph → compile → GPU execute → verify
//! against CPU reference for Conv2d, LogSoftmax, Gelu, Sigmoid, Tanh, Silu,
//! GroupNorm, and BatchNorm.
//!
//! Part of #2214.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use super::helpers::{assert_close, compile_and_run, create_input_buffer, input_node};

fn weight(data: Vec<f32>, shape: Vec<usize>) -> WeightRef {
    WeightRef::new(data, shape).expect("weight")
}

// -- CPU reference helpers ----------------------------------------------------

fn cpu_gelu(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

fn cpu_softmax(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0_f32; rows * cols];
    for b in 0..rows {
        let row = &input[b * cols..(b + 1) * cols];
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = row.iter().map(|v| (v - max_val).exp()).sum();
        for d in 0..cols {
            output[b * cols + d] = (row[d] - max_val).exp() / exp_sum;
        }
    }
    output
}

/// CPU reference for Conv2d: [C_in, H, W] x [C_out, C_in, KH, KW] -> [C_out, OH, OW].
fn cpu_conv2d(
    input: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    in_ch: usize,
    out_ch: usize,
    kh: usize,
    kw: usize,
    h: usize,
    w_dim: usize,
    stride: usize,
    pad: usize,
) -> Vec<f32> {
    let out_h = (h + 2 * pad - kh) / stride + 1;
    let out_w = (w_dim + 2 * pad - kw) / stride + 1;
    let mut output = vec![0.0_f32; out_ch * out_h * out_w];
    for oc in 0..out_ch {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut sum = bias.map_or(0.0, |b| b[oc]);
                for ic in 0..in_ch {
                    for fh in 0..kh {
                        for fw in 0..kw {
                            let ih = oh * stride + fh;
                            let iw = ow * stride + fw;
                            let ih_off = ih as isize - pad as isize;
                            let iw_off = iw as isize - pad as isize;
                            if ih_off >= 0
                                && (ih_off as usize) < h
                                && iw_off >= 0
                                && (iw_off as usize) < w_dim
                            {
                                let in_idx =
                                    ic * h * w_dim + ih_off as usize * w_dim + iw_off as usize;
                                let w_idx = oc * in_ch * kh * kw + ic * kh * kw + fh * kw + fw;
                                sum += input[in_idx] * w[w_idx];
                            }
                        }
                    }
                }
                output[oc * out_h * out_w + oh * out_w + ow] = sum;
            }
        }
    }
    output
}

fn cpu_group_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    batch: usize,
    channels: usize,
    time: usize,
    num_groups: usize,
) -> Vec<f32> {
    let eps = 1e-5_f32;
    let ch_per_group = channels / num_groups;
    let mut output = vec![0.0_f32; batch * channels * time];
    for b in 0..batch {
        for g in 0..num_groups {
            let c_start = g * ch_per_group;
            let c_end = c_start + ch_per_group;
            let group_size = ch_per_group * time;
            let mut sum = 0.0_f32;
            for c in c_start..c_end {
                for t in 0..time {
                    sum += input[b * channels * time + c * time + t];
                }
            }
            let mean = sum / group_size as f32;
            let mut var_sum = 0.0_f32;
            for c in c_start..c_end {
                for t in 0..time {
                    let v = input[b * channels * time + c * time + t] - mean;
                    var_sum += v * v;
                }
            }
            let var = var_sum / group_size as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for c in c_start..c_end {
                for t in 0..time {
                    let idx = b * channels * time + c * time + t;
                    output[idx] = (input[idx] - mean) * inv_std * gamma[c] + beta[c];
                }
            }
        }
    }
    output
}

fn cpu_batch_norm(
    input: &[f32],
    bn_weight: &[f32],
    bn_bias: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    batch: usize,
    channels: usize,
    time: usize,
) -> Vec<f32> {
    let eps = 1e-5_f32;
    let mut output = vec![0.0_f32; batch * channels * time];
    for b in 0..batch {
        for c in 0..channels {
            let inv_std = 1.0 / (running_var[c] + eps).sqrt();
            for t in 0..time {
                let idx = b * channels * time + c * time + t;
                output[idx] = bn_weight[c] * (input[idx] - running_mean[c]) * inv_std + bn_bias[c];
            }
        }
    }
    output
}

// -- Test 12: Conv2d + ReLU ---------------------------------------------------

/// Conv2d(1, 4, 3x3, pad=1) + ReLU: 2D convolution compiled end-to-end.
#[test]
fn test_compiled_conv2d_relu() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (in_ch, out_ch, kh, kw, h, w_dim, pad) = (1, 4, 3, 3, 6, 6, 1);
    let out_h = (h + 2 * pad - kh) + 1;
    let out_w = (w_dim + 2 * pad - kw) + 1;

    let w_data = super::test_utils::rand_f32_vec(0xC2D_0001, out_ch * in_ch * kh * kw, -0.3, 0.3);
    let b_data = super::test_utils::rand_f32_vec(0xC2D_0002, out_ch, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0xC2D_0003, in_ch * h * w_dim, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[in_ch, h, w_dim]),
        TraceNode::new(
            1,
            "conv2d_0".into(),
            TraceOp::Conv2d {
                weight: weight(w_data.clone(), vec![out_ch, in_ch, kh, kw]),
                bias: Some(weight(b_data.clone(), vec![out_ch])),
                padding: [pad, pad],
                stride: [1, 1],
                dilation: [1, 1],
                groups: 1,
            },
            vec![0],
            vec![out_ch, out_h, out_w],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "relu_0".into(),
            TraceOp::Relu,
            vec![1],
            vec![out_ch, out_h, out_w],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        out_ch * out_h * out_w,
    );

    let conv_out = cpu_conv2d(
        &input_data,
        &w_data,
        Some(&b_data),
        in_ch,
        out_ch,
        kh,
        kw,
        h,
        w_dim,
        1,
        pad,
    );
    let expected: Vec<f32> = conv_out.iter().map(|v| v.max(0.0)).collect();
    assert_close("conv2d_relu", &result, &expected, 1e-3);
}

// -- Test 13: LogSoftmax ------------------------------------------------------

/// Linear(4, 6) -> LogSoftmax(dim=1): log-softmax in compiled pipeline.
#[test]
fn test_compiled_log_softmax() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (in_f, out_f, batch) = (4, 6, 2);
    let w = super::test_utils::rand_f32_vec(0x1057_0001, out_f * in_f, -0.5, 0.5);
    let b = super::test_utils::rand_f32_vec(0x1057_0002, out_f, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0x1057_0003, batch * in_f, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, in_f]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight(w.clone(), vec![out_f, in_f]),
                bias: Some(weight(b.clone(), vec![out_f])),
            },
            vec![0],
            vec![batch, out_f],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "log_softmax_0".into(),
            TraceOp::LogSoftmax { dim: 1 },
            vec![1],
            vec![batch, out_f],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * out_f,
    );

    let logits = super::test_utils::linear_ref(&input_data, &w, Some(&b), batch, in_f, out_f);
    let sm = cpu_softmax(&logits, batch, out_f);
    let expected: Vec<f32> = sm.iter().map(|v| v.ln()).collect();
    assert_close("log_softmax", &result, &expected, 1e-4);
}

// -- Test 14: Gelu activation -------------------------------------------------

/// Linear(4, 8) -> Gelu: GELU activation in compiled pipeline.
#[test]
fn test_compiled_gelu() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (in_f, out_f, batch) = (4, 8, 2);
    let w = super::test_utils::rand_f32_vec(0x6E10_0001, out_f * in_f, -0.5, 0.5);
    let b = super::test_utils::rand_f32_vec(0x6E10_0002, out_f, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0x6E10_0003, batch * in_f, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, in_f]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight(w.clone(), vec![out_f, in_f]),
                bias: Some(weight(b.clone(), vec![out_f])),
            },
            vec![0],
            vec![batch, out_f],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "gelu_0".into(),
            TraceOp::Gelu,
            vec![1],
            vec![batch, out_f],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * out_f,
    );

    let linear_out = super::test_utils::linear_ref(&input_data, &w, Some(&b), batch, in_f, out_f);
    let expected: Vec<f32> = linear_out.iter().map(|&x| cpu_gelu(x)).collect();
    assert_close("gelu", &result, &expected, 1e-3);
}

// -- Test 15: Sigmoid ---------------------------------------------------------

/// Linear(4, 6) -> Sigmoid: sigmoid activation in compiled pipeline.
#[test]
fn test_compiled_sigmoid() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (in_f, out_f, batch) = (4, 6, 2);
    let w = super::test_utils::rand_f32_vec(0x516D_0001, out_f * in_f, -0.5, 0.5);
    let b = super::test_utils::rand_f32_vec(0x516D_0002, out_f, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0x516D_0003, batch * in_f, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, in_f]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight(w.clone(), vec![out_f, in_f]),
                bias: Some(weight(b.clone(), vec![out_f])),
            },
            vec![0],
            vec![batch, out_f],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "sigmoid_0".into(),
            TraceOp::Sigmoid,
            vec![1],
            vec![batch, out_f],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * out_f,
    );

    let linear_out = super::test_utils::linear_ref(&input_data, &w, Some(&b), batch, in_f, out_f);
    let expected: Vec<f32> = linear_out
        .iter()
        .map(|&x| 1.0 / (1.0 + (-x).exp()))
        .collect();
    assert_close("sigmoid", &result, &expected, 1e-5);
}

// -- Test 16: Tanh ------------------------------------------------------------

/// Linear(4, 6) -> Tanh: tanh activation in compiled pipeline.
#[test]
fn test_compiled_tanh() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (in_f, out_f, batch) = (4, 6, 2);
    let w = super::test_utils::rand_f32_vec(0x7A4E_0001, out_f * in_f, -0.5, 0.5);
    let b = super::test_utils::rand_f32_vec(0x7A4E_0002, out_f, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0x7A4E_0003, batch * in_f, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, in_f]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight(w.clone(), vec![out_f, in_f]),
                bias: Some(weight(b.clone(), vec![out_f])),
            },
            vec![0],
            vec![batch, out_f],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "tanh_0".into(),
            TraceOp::Tanh,
            vec![1],
            vec![batch, out_f],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * out_f,
    );

    let linear_out = super::test_utils::linear_ref(&input_data, &w, Some(&b), batch, in_f, out_f);
    let expected: Vec<f32> = linear_out.iter().map(|&x| x.tanh()).collect();
    assert_close("tanh", &result, &expected, 1e-5);
}

// -- Test 17: Silu (SiLU = x * sigmoid(x)) -----------------------------------

/// Linear(4, 8) -> Silu: SiLU activation in compiled pipeline.
#[test]
fn test_compiled_silu() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (in_f, out_f, batch) = (4, 8, 2);
    let w = super::test_utils::rand_f32_vec(0x5110_0001, out_f * in_f, -0.5, 0.5);
    let b = super::test_utils::rand_f32_vec(0x5110_0002, out_f, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0x5110_0003, batch * in_f, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, in_f]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight(w.clone(), vec![out_f, in_f]),
                bias: Some(weight(b.clone(), vec![out_f])),
            },
            vec![0],
            vec![batch, out_f],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "silu_0".into(),
            TraceOp::Silu,
            vec![1],
            vec![batch, out_f],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * out_f,
    );

    let linear_out = super::test_utils::linear_ref(&input_data, &w, Some(&b), batch, in_f, out_f);
    let expected: Vec<f32> = linear_out
        .iter()
        .map(|&x| x * (1.0 / (1.0 + (-x).exp())))
        .collect();
    assert_close("silu", &result, &expected, 1e-4);
}

// -- Test 18: GroupNorm -------------------------------------------------------

/// [2, 4, 8] -> GroupNorm(num_groups=2) -> [2, 4, 8]: group normalization.
#[test]
fn test_compiled_group_norm() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, channels, time) = (2, 4, 8);
    let num_groups = 2;
    let gamma = super::test_utils::rand_f32_vec(0x64B0_0001, channels, 0.5, 1.5);
    let beta = super::test_utils::rand_f32_vec(0x64B0_0002, channels, -0.1, 0.1);
    let input_data =
        super::test_utils::rand_f32_vec(0x64B0_0003, batch * channels * time, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, channels, time]),
        TraceNode::new(
            1,
            "group_norm_0".into(),
            TraceOp::GroupNorm {
                num_groups,
                eps: 1e-5,
                weight: weight(gamma.clone(), vec![channels]),
                bias: weight(beta.clone(), vec![channels]),
            },
            vec![0],
            vec![batch, channels, time],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * channels * time,
    );

    let expected = cpu_group_norm(
        &input_data,
        &gamma,
        &beta,
        batch,
        channels,
        time,
        num_groups,
    );
    assert_close("group_norm", &result, &expected, 1e-3);
}

// -- Test 19: BatchNorm -------------------------------------------------------

/// [2, 4, 8] -> BatchNorm(eps=1e-5) -> [2, 4, 8]: batch normalization (eval mode).
#[test]
fn test_compiled_batch_norm() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, channels, time) = (2, 4, 8);
    let bn_weight = super::test_utils::rand_f32_vec(0xBA70_0001, channels, 0.5, 1.5);
    let bn_bias = super::test_utils::rand_f32_vec(0xBA70_0002, channels, -0.1, 0.1);
    let running_mean = super::test_utils::rand_f32_vec(0xBA70_0003, channels, -0.5, 0.5);
    let running_var = super::test_utils::rand_f32_vec(0xBA70_0004, channels, 0.5, 2.0);
    let input_data =
        super::test_utils::rand_f32_vec(0xBA70_0005, batch * channels * time, -1.0, 1.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, channels, time]),
        TraceNode::new(
            1,
            "batch_norm_0".into(),
            TraceOp::BatchNorm {
                eps: 1e-5,
                weight: weight(bn_weight.clone(), vec![channels]),
                bias: weight(bn_bias.clone(), vec![channels]),
                running_mean: weight(running_mean.clone(), vec![channels]),
                running_var: weight(running_var.clone(), vec![channels]),
            },
            vec![0],
            vec![batch, channels, time],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * channels * time,
    );

    let expected = cpu_batch_norm(
        &input_data,
        &bn_weight,
        &bn_bias,
        &running_mean,
        &running_var,
        batch,
        channels,
        time,
    );
    assert_close("batch_norm", &result, &expected, 1e-3);
}

// -- Test 20: Cumsum 1D -------------------------------------------------------

/// Cumsum([1, 3, 5, 7], dim=0) -> [1, 4, 9, 16]: prefix sum on GPU via Blelloch.
///
/// Verifies NativeOpKind::Cumsum executes correctly in the compiled model
/// pipeline. The DSL compilation tests (trace_compile_tests_conv.rs) verify
/// the TraceOp compiles to NativeOp; this test verifies GPU execution parity.
///
/// Part of #2228.
#[test]
fn test_compiled_cumsum_1d() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let input_data: Vec<f32> = vec![1.0, 3.0, 5.0, 7.0];
    let n = input_data.len();

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[n]),
        TraceNode::new(
            1,
            "cumsum_0".into(),
            TraceOp::Cumsum { dim: 0 },
            vec![0],
            vec![n],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        n,
    );

    // CPU reference: cumulative sum.
    let expected: Vec<f32> = vec![1.0, 4.0, 9.0, 16.0];
    assert_close("cumsum_1d", &result, &expected, 1e-5);
}

// -- Test 21: Cumsum 2D along dim 1 -------------------------------------------

/// Cumsum([[1, 2, 3], [4, 5, 6]], dim=1) -> [[1, 3, 6], [4, 9, 15]]:
/// 2D prefix sum along last dimension.
///
/// Part of #2228.
#[test]
fn test_compiled_cumsum_2d_dim1() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 3);
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "cumsum_0".into(),
            TraceOp::Cumsum { dim: 1 },
            vec![0],
            vec![rows, cols],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        rows * cols,
    );

    // CPU reference: cumsum along dim 1 (columns).
    let expected: Vec<f32> = vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0];
    assert_close("cumsum_2d_dim1", &result, &expected, 1e-5);
}

// -- Test 22: Cumsum 2D along dim 0 -------------------------------------------

/// Cumsum([[1, 2, 3], [4, 5, 6]], dim=0) -> [[1, 2, 3], [5, 7, 9]]:
/// 2D prefix sum along first dimension.
///
/// Part of #2228.
#[test]
fn test_compiled_cumsum_2d_dim0() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 3);
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "cumsum_0".into(),
            TraceOp::Cumsum { dim: 0 },
            vec![0],
            vec![rows, cols],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        rows * cols,
    );

    // CPU reference: cumsum along dim 0 (rows).
    let expected: Vec<f32> = vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0];
    assert_close("cumsum_2d_dim0", &result, &expected, 1e-5);
}

// -- Test 23: Cumsum with random data -----------------------------------------

/// Cumsum on random [4, 64] data along dim=1: exercises the Blelloch kernel
/// with non-trivial axis length, verifying GPU parity against CPU reference.
///
/// Part of #2228.
#[test]
fn test_compiled_cumsum_random() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (4, 64);
    let input_data = super::test_utils::rand_f32_vec(0xC500_0001, rows * cols, -2.0, 2.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "cumsum_0".into(),
            TraceOp::Cumsum { dim: 1 },
            vec![0],
            vec![rows, cols],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        rows * cols,
    );

    // CPU reference: cumulative sum along dim 1.
    let mut expected = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        let mut acc = 0.0_f32;
        for c in 0..cols {
            acc += input_data[r * cols + c];
            expected[r * cols + c] = acc;
        }
    }
    assert_close("cumsum_random", &result, &expected, 1e-3);
}
