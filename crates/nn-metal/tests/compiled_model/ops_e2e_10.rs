// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end compiled model tests: Unsqueeze, Squeeze, ReflectionPad1d,
//! ConstantPadNd, Unfold.
//!
//! Continuation of `ops_e2e_9.rs` (tests 65+).
//! Fills proof coverage gaps: these ops have compile dispatch but lacked
//! GPU E2E tests verifying trace → compile → Metal execute → CPU reference.
//!
//! Part of #3020.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use super::helpers::{assert_close, compile_and_run, create_input_buffer, input_node};

// -- Test 65: Unsqueeze (passthrough) -----------------------------------------

/// Unsqueeze: [2, 6] → [1, 2, 6] by inserting dim 0.
/// Passthrough op — verifies buffer aliasing preserves data through reshape.
#[test]
fn test_compiled_unsqueeze_dim0() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let n = rows * cols;
    let input_data = super::test_utils::rand_f32_vec(0xA010_0001, n, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "unsqueeze_0".into(),
            TraceOp::Unsqueeze { dim: 0 },
            vec![0],
            vec![1, rows, cols],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        n,
    );

    // Data should be identical — only shape metadata changes.
    assert_close("unsqueeze_dim0", &result, &input_data, 0.0);
}

/// Unsqueeze: [2, 6] → [2, 1, 6] by inserting dim 1.
#[test]
fn test_compiled_unsqueeze_dim1() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let n = rows * cols;
    let input_data = super::test_utils::rand_f32_vec(0xA010_0002, n, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "unsqueeze_1".into(),
            TraceOp::Unsqueeze { dim: 1 },
            vec![0],
            vec![rows, 1, cols],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        n,
    );

    assert_close("unsqueeze_dim1", &result, &input_data, 0.0);
}

// -- Test 67: Squeeze (passthrough) -------------------------------------------

/// Squeeze: [1, 2, 6] → [2, 6] by removing dim 0.
#[test]
fn test_compiled_squeeze_dim0() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let n = rows * cols;
    let input_data = super::test_utils::rand_f32_vec(0xA020_0001, n, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, rows, cols]),
        TraceNode::new(
            1,
            "squeeze_0".into(),
            TraceOp::Squeeze { dim: 0 },
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

    assert_close("squeeze_dim0", &result, &input_data, 0.0);
}

// NOTE: AvgPool2d and MaxPool2d compile to TensorBlock steps but MSL codegen
// is deferred. They execute via runtime dispatch (NativeOp), not compiled
// Metal shaders. E2E tests for these pool ops are deferred until MSL codegen
// lands or NativeOp routing is implemented in CompiledModel execution.
// MaxPool1d works because it compiles to CompiledStep::NativeOp directly.

// -- Test 72: ReflectionPad1d -------------------------------------------------

/// ReflectionPad1d: [1, 2, 8] → [1, 2, 12] with pad_left=2, pad_right=2.
#[test]
fn test_compiled_reflection_pad1d() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, ch, time) = (1, 2, 8);
    let (pad_left, pad_right) = (2, 2);
    let out_time = time + pad_left + pad_right; // 12

    let input_data = super::test_utils::rand_f32_vec(0xA050_0001, batch * ch * time, -3.0, 3.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, ch, time]),
        TraceNode::new(
            1,
            "refpad_0".into(),
            TraceOp::ReflectionPad1d {
                pad_left,
                pad_right,
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
        batch * ch * out_time,
    );

    let expected = cpu_reflection_pad1d(&input_data, batch, ch, time, pad_left, pad_right);
    assert_close("reflection_pad1d", &result, &expected, 0.0);
}

// -- Test 73: ConstantPadNd ---------------------------------------------------

/// ConstantPadNd: [1, 2, 6] → [1, 2, 10] with padding=[2, 2], value=0.0.
/// Pads last dim left and right with zeros.
#[test]
fn test_compiled_constant_pad_nd() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, ch, time) = (1, 2, 6);
    let (pad_left, pad_right) = (2, 2);
    let out_time = time + pad_left + pad_right; // 10
    let pad_value = 0.0_f64;

    let input_data = super::test_utils::rand_f32_vec(0xA060_0001, batch * ch * time, -3.0, 3.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, ch, time]),
        TraceNode::new(
            1,
            "constpad_0".into(),
            TraceOp::ConstantPadNd {
                padding: vec![pad_left, pad_right],
                value: pad_value,
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
        batch * ch * out_time,
    );

    let expected = cpu_constant_pad_1d(
        &input_data,
        batch,
        ch,
        time,
        pad_left,
        pad_right,
        pad_value as f32,
    );
    assert_close("constant_pad_nd", &result, &expected, 0.0);
}

/// ConstantPadNd with non-zero fill: padding=[3, 1], value=-1.0.
#[test]
fn test_compiled_constant_pad_nd_nonzero() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, ch, time) = (1, 1, 4);
    let (pad_left, pad_right) = (3, 1);
    let out_time = time + pad_left + pad_right; // 8
    let pad_value = -1.0_f64;

    let input_data = super::test_utils::rand_f32_vec(0xA060_0002, batch * ch * time, 0.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, ch, time]),
        TraceNode::new(
            1,
            "constpad_nz".into(),
            TraceOp::ConstantPadNd {
                padding: vec![pad_left, pad_right],
                value: pad_value,
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
        batch * ch * out_time,
    );

    let expected = cpu_constant_pad_1d(
        &input_data,
        batch,
        ch,
        time,
        pad_left,
        pad_right,
        pad_value as f32,
    );
    assert_close("constant_pad_nd_nonzero", &result, &expected, 0.0);
}

// -- Test 75: Unfold ----------------------------------------------------------

/// Unfold: [1, 1, 8] along dim=2 with size=3, step=2 → [1, 1, 3, 3].
/// Extracts sliding windows: positions [0,2,4] each of width 3.
#[test]
fn test_compiled_unfold() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, ch, time) = (1, 1, 8);
    let (size, step) = (3, 2);
    let n_windows = (time - size) / step + 1; // 3

    let input_data = super::test_utils::rand_f32_vec(0xA070_0001, batch * ch * time, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, ch, time]),
        TraceNode::new(
            1,
            "unfold_0".into(),
            TraceOp::Unfold { dim: 2, size, step },
            vec![0],
            vec![batch, ch, n_windows, size],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * ch * n_windows * size,
    );

    let expected = cpu_unfold(&input_data, batch, ch, time, 2, size, step);
    assert_close("unfold", &result, &expected, 0.0);
}

// -- CPU reference helpers ----------------------------------------------------

fn cpu_reflection_pad1d(
    input: &[f32],
    batch: usize,
    ch: usize,
    time: usize,
    pad_left: usize,
    pad_right: usize,
) -> Vec<f32> {
    let out_time = time + pad_left + pad_right;
    let mut out = vec![0.0f32; batch * ch * out_time];
    for b in 0..batch {
        for c in 0..ch {
            for t in 0..out_time {
                let src_t = if t < pad_left {
                    // Reflect left: pad_left - t
                    pad_left - t
                } else if t >= pad_left + time {
                    // Reflect right: time - 2 - (t - pad_left - time)
                    2 * time - 2 - (t - pad_left)
                } else {
                    t - pad_left
                };
                out[b * ch * out_time + c * out_time + t] = input[b * ch * time + c * time + src_t];
            }
        }
    }
    out
}

fn cpu_constant_pad_1d(
    input: &[f32],
    batch: usize,
    ch: usize,
    time: usize,
    pad_left: usize,
    pad_right: usize,
    value: f32,
) -> Vec<f32> {
    let out_time = time + pad_left + pad_right;
    let mut out = vec![value; batch * ch * out_time];
    for b in 0..batch {
        for c in 0..ch {
            for t in 0..time {
                out[b * ch * out_time + c * out_time + pad_left + t] =
                    input[b * ch * time + c * time + t];
            }
        }
    }
    out
}

fn cpu_unfold(
    input: &[f32],
    batch: usize,
    ch: usize,
    time: usize,
    _dim: usize, // always last dim for 3D input
    size: usize,
    step: usize,
) -> Vec<f32> {
    let n_windows = (time - size) / step + 1;
    let mut out = vec![0.0f32; batch * ch * n_windows * size];
    for b in 0..batch {
        for c in 0..ch {
            for wi in 0..n_windows {
                let start = wi * step;
                for s in 0..size {
                    out[b * ch * n_windows * size + c * n_windows * size + wi * size + s] =
                        input[b * ch * time + c * time + start + s];
                }
            }
        }
    }
    out
}
