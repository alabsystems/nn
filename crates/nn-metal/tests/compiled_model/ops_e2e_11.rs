// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end compiled model tests: Permute, InstanceNorm, MaxPool1d,
//! ToDtype, Arange.
//!
//! Continuation of `ops_e2e_10.rs` (tests 76+).
//! Fills proof coverage gaps: these ops compile and execute but lacked
//! GPU E2E tests verifying trace → compile → Metal execute → CPU reference.
//!
//! Part of #3020.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use super::helpers::{assert_close, compile_and_run, create_input_buffer, input_node};

// -- Test 76: Permute [0, 2, 1] (GPU transpose kernel) -----------------------

/// Permute: [2, 3, 4] → [2, 4, 3] via axes [0, 2, 1].
/// Compiles to Dispatch (GPU transpose kernel), NOT a passthrough.
#[test]
fn test_compiled_permute_021() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (d0, d1, d2) = (2, 3, 4);
    let n = d0 * d1 * d2;
    let input_data = super::test_utils::rand_f32_vec(0xB010_0001, n, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[d0, d1, d2]),
        TraceNode::new(
            1,
            "permute_021".into(),
            TraceOp::Permute {
                axes: vec![0, 2, 1],
            },
            vec![0],
            vec![d0, d2, d1],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        n,
    );

    let expected = cpu_permute_021(&input_data, d0, d1, d2);
    assert_close("permute_021", &result, &expected, 0.0);
}

/// Permute: [2, 3, 4] → [4, 2, 3] via axes [2, 0, 1].
#[test]
fn test_compiled_permute_201() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (d0, d1, d2) = (2, 3, 4);
    let n = d0 * d1 * d2;
    let input_data = super::test_utils::rand_f32_vec(0xB010_0002, n, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[d0, d1, d2]),
        TraceNode::new(
            1,
            "permute_201".into(),
            TraceOp::Permute {
                axes: vec![2, 0, 1],
            },
            vec![0],
            vec![d2, d0, d1],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        n,
    );

    let expected = cpu_permute_201(&input_data, d0, d1, d2);
    assert_close("permute_201", &result, &expected, 0.0);
}

// -- Test 78: InstanceNorm (NativeOp, fused GPU kernel) -----------------------

/// InstanceNorm: [1, 2, 8] → [1, 2, 8].
/// Normalizes each channel independently (mean=0, variance=1).
/// Compiles to NativeOp::InstanceNorm with fused threadgroup reduction.
#[test]
fn test_compiled_instance_norm() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, ch, time) = (1, 2, 8);
    let n = batch * ch * time;
    let eps = 1e-5_f64;
    let input_data = super::test_utils::rand_f32_vec(0xB020_0001, n, -3.0, 3.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, ch, time]),
        TraceNode::new(
            1,
            "inorm_0".into(),
            TraceOp::InstanceNorm { eps },
            vec![0],
            vec![batch, ch, time],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        n,
    );

    let expected = cpu_instance_norm(&input_data, batch, ch, time, eps as f32);
    // InstanceNorm tolerance: GPU uses threadgroup reduction with float precision.
    assert_close("instance_norm", &result, &expected, 1e-4);
}

// -- Test 79: MaxPool1d (NativeOp, CPU roundtrip) -----------------------------

/// MaxPool1d: [1, 2, 8] with kernel=3, stride=2, padding=1 → [1, 2, 4].
/// Compiles to NativeOp::MaxPool1d.
#[test]
fn test_compiled_max_pool1d() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, ch, time) = (1, 2, 8);
    let (kernel_size, stride, padding) = (3, 2, 1);
    let out_time = (time + 2 * padding - kernel_size) / stride + 1; // 4
    let n_out = batch * ch * out_time;
    let input_data = super::test_utils::rand_f32_vec(0xB030_0001, batch * ch * time, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, ch, time]),
        TraceNode::new(
            1,
            "maxpool1d_0".into(),
            TraceOp::MaxPool1d {
                kernel_size,
                stride,
                padding,
            },
            vec![0],
            vec![batch, ch, out_time],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        n_out,
    );

    let expected = cpu_max_pool1d(&input_data, batch, ch, time, kernel_size, stride, padding);
    assert_close("max_pool1d", &result, &expected, 0.0);
}

/// MaxPool1d: no padding, kernel=2, stride=2.
/// [1, 1, 6] → [1, 1, 3].
#[test]
fn test_compiled_max_pool1d_no_padding() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, ch, time) = (1, 1, 6);
    let (kernel_size, stride, padding) = (2, 2, 0);
    let out_time = (time - kernel_size) / stride + 1; // 3
    let n_out = batch * ch * out_time;
    let input_data = super::test_utils::rand_f32_vec(0xB030_0002, batch * ch * time, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, ch, time]),
        TraceNode::new(
            1,
            "maxpool1d_nopad".into(),
            TraceOp::MaxPool1d {
                kernel_size,
                stride,
                padding,
            },
            vec![0],
            vec![batch, ch, out_time],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        n_out,
    );

    let expected = cpu_max_pool1d(&input_data, batch, ch, time, kernel_size, stride, padding);
    assert_close("max_pool1d_no_padding", &result, &expected, 0.0);
}

// -- Test 81: ToDtype (Passthrough) -------------------------------------------

/// ToDtype: [2, 4] → [2, 4] (F32 → F32 passthrough).
/// In the compiled pipeline, DynTensor uses F32 storage, so ToDtype is a
/// no-op passthrough. Data must be preserved bit-exact.
#[test]
fn test_compiled_to_dtype() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 4);
    let n = rows * cols;
    let input_data = super::test_utils::rand_f32_vec(0xB040_0001, n, -10.0, 10.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "to_dtype_0".into(),
            TraceOp::ToDtype {
                target_dtype: DType::F32,
            },
            vec![0],
            vec![rows, cols],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        n,
    );

    assert_close("to_dtype", &result, &input_data, 0.0);
}

// -- Test 82: Arange (pre-computed constant) ----------------------------------

/// Arange: start=0, end=5, step=1 → [0, 1, 2, 3, 4].
/// Compiles to ConstantValue (pre-computed at compile time).
/// No input tensor needed — output is purely from the constant.
#[test]
fn test_compiled_arange() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (start, end, step) = (0.0_f64, 5.0_f64, 1.0_f64);
    let n = 5;

    // Arange has no inputs — use a dummy input forwarded through.
    // The graph needs at least one Input node for CompiledModel.
    let dummy_data = vec![0.0f32; 1];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1]),
        TraceNode::new(
            1,
            "arange_0".into(),
            TraceOp::Arange { start, end, step },
            vec![], // no inputs — purely constant
            vec![n],
            DType::F32,
        ),
    ]);

    let compiled = nn_metal::compiled_model::CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");
    let out_buf = compiled
        .execute(&cache, &[&create_input_buffer(&cache, &dummy_data)])
        .expect("execute");

    // Arange is the last node but graph output may vary.
    // The compiled model returns the last node's output.
    let result = super::helpers::read_output_n(&out_buf, n);
    let expected: Vec<f32> = (0..n).map(|i| (start + (i as f64) * step) as f32).collect();
    assert_close("arange", &result, &expected, 0.0);
}

/// Arange: start=0.5, end=3.5, step=0.5 → [0.5, 1.0, 1.5, 2.0, 2.5, 3.0].
#[test]
fn test_compiled_arange_fractional() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (start, end, step) = (0.5_f64, 3.5_f64, 0.5_f64);
    let n = 6;
    let dummy_data = vec![0.0f32; 1];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1]),
        TraceNode::new(
            1,
            "arange_frac".into(),
            TraceOp::Arange { start, end, step },
            vec![],
            vec![n],
            DType::F32,
        ),
    ]);

    let compiled = nn_metal::compiled_model::CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");
    let out_buf = compiled
        .execute(&cache, &[&create_input_buffer(&cache, &dummy_data)])
        .expect("execute");

    let result = super::helpers::read_output_n(&out_buf, n);
    let expected: Vec<f32> = (0..n).map(|i| (start + (i as f64) * step) as f32).collect();
    assert_close("arange_fractional", &result, &expected, 0.0);
}

// -- CPU reference helpers ----------------------------------------------------

fn cpu_permute_021(input: &[f32], d0: usize, d1: usize, d2: usize) -> Vec<f32> {
    // [d0, d1, d2] → [d0, d2, d1]
    let mut out = vec![0.0f32; d0 * d1 * d2];
    for i in 0..d0 {
        for j in 0..d1 {
            for k in 0..d2 {
                out[i * d2 * d1 + k * d1 + j] = input[i * d1 * d2 + j * d2 + k];
            }
        }
    }
    out
}

fn cpu_permute_201(input: &[f32], d0: usize, d1: usize, d2: usize) -> Vec<f32> {
    // [d0, d1, d2] → [d2, d0, d1]
    let mut out = vec![0.0f32; d0 * d1 * d2];
    for i in 0..d0 {
        for j in 0..d1 {
            for k in 0..d2 {
                out[k * d0 * d1 + i * d1 + j] = input[i * d1 * d2 + j * d2 + k];
            }
        }
    }
    out
}

fn cpu_instance_norm(input: &[f32], batch: usize, ch: usize, time: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * ch * time];
    for b in 0..batch {
        for c in 0..ch {
            let offset = b * ch * time + c * time;
            let slice = &input[offset..offset + time];
            let mean: f32 = slice.iter().sum::<f32>() / time as f32;
            let var: f32 =
                slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / time as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for t in 0..time {
                out[offset + t] = (slice[t] - mean) * inv_std;
            }
        }
    }
    out
}

fn cpu_max_pool1d(
    input: &[f32],
    batch: usize,
    ch: usize,
    time: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    let out_time = (time + 2 * padding - kernel_size) / stride + 1;
    let mut out = vec![0.0f32; batch * ch * out_time];
    for b in 0..batch {
        for c in 0..ch {
            for ot in 0..out_time {
                let mut max_val = f32::NEG_INFINITY;
                for k in 0..kernel_size {
                    let t = ot * stride + k;
                    if t >= padding && t < time + padding {
                        let idx = b * ch * time + c * time + (t - padding);
                        max_val = max_val.max(input[idx]);
                    }
                }
                out[b * ch * out_time + c * out_time + ot] = max_val;
            }
        }
    }
    out
}
