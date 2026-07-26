// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end compiled model tests: ELU + tiled transpose.
//!
//! Continuation of `compiled_model_ops_e2e_3.rs` (tests 34+).
//! Exercises the full pipeline: build graph -> compile -> GPU execute -> verify
//! against CPU reference.
//!
//! Part of #3230.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use super::helpers::{assert_close, compile_and_run, create_input_buffer, input_node};

fn weight(data: Vec<f32>, shape: Vec<usize>) -> WeightRef {
    WeightRef::new(data, shape).expect("weight")
}

// -- Test 34: ELU (standalone) ------------------------------------------------

/// ELU(alpha=1.0): [2, 6] -> elu -> [2, 6].
/// `x if x >= 0, alpha * (exp(x) - 1) otherwise`.
#[test]
fn test_compiled_elu() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let input_data = super::test_utils::rand_f32_vec(0xE1A0_0001, rows * cols, -3.0, 3.0);
    let alpha = 1.0_f64;

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "elu_0".into(),
            TraceOp::Elu { alpha },
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

    let expected: Vec<f32> = input_data
        .iter()
        .map(|&x| {
            if x >= 0.0 {
                x
            } else {
                (alpha as f32) * x.exp_m1()
            }
        })
        .collect();
    assert_close("elu", &result, &expected, 1e-5);
}

/// ELU(alpha=0.5): non-default alpha, same shape.
#[test]
fn test_compiled_elu_alpha_half() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let input_data = super::test_utils::rand_f32_vec(0xE1A0_0002, rows * cols, -4.0, 4.0);
    let alpha = 0.5_f64;

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "elu_half_0".into(),
            TraceOp::Elu { alpha },
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

    let expected: Vec<f32> = input_data
        .iter()
        .map(|&x| {
            if x >= 0.0 {
                x
            } else {
                (alpha as f32) * x.exp_m1()
            }
        })
        .collect();
    assert_close("elu_alpha_half", &result, &expected, 1e-5);
}

// -- Test 36: Tiled transpose 2D (>= 16x16) -----------------------------------

/// Transpose [32, 64] → [64, 32] exercises the tiled shared-memory kernel.
/// Both dims >= 16, axes = [1, 0] → qualifies for tiled dispatch.
/// Part of #3230 (Gap 4).
#[test]
fn test_compiled_tiled_transpose_2d() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (m, n) = (32, 64);
    let input_data = super::test_utils::rand_f32_vec(0xD1ED_0001, m * n, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[m, n]),
        TraceNode::new(
            1,
            "transpose_0".into(),
            TraceOp::Transpose { dim0: 0, dim1: 1 },
            vec![0],
            vec![n, m],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        m * n,
    );

    // CPU reference: output[col * m + row] = input[row * n + col]
    let mut expected = vec![0.0_f32; m * n];
    for row in 0..m {
        for col in 0..n {
            expected[col * m + row] = input_data[row * n + col];
        }
    }
    assert_close("tiled_transpose_2d", &result, &expected, 0.0);
}

/// Batched transpose [4, 32, 48] → [4, 48, 32] — tiled with batch dim.
/// Validates grid z-dimension batching.
#[test]
fn test_compiled_tiled_transpose_batched() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, m, n) = (4, 32, 48);
    let total = batch * m * n;
    let input_data = super::test_utils::rand_f32_vec(0xD1ED_0002, total, -3.0, 3.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, m, n]),
        TraceNode::new(
            1,
            "transpose_b".into(),
            TraceOp::Transpose { dim0: 1, dim1: 2 },
            vec![0],
            vec![batch, n, m],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        total,
    );

    // CPU reference: per-batch 2D transpose
    let mut expected = vec![0.0_f32; total];
    for b in 0..batch {
        let off = b * m * n;
        for row in 0..m {
            for col in 0..n {
                expected[off + col * m + row] = input_data[off + row * n + col];
            }
        }
    }
    assert_close("tiled_transpose_batched", &result, &expected, 0.0);
}

/// Non-tile-aligned shape [20, 35] → [35, 20] — boundary handling.
/// Dims > 16 but not multiples of 16, verifies bounds checking in MSL kernel.
#[test]
fn test_compiled_tiled_transpose_non_aligned() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (m, n) = (20, 35);
    let input_data = super::test_utils::rand_f32_vec(0xD1ED_0003, m * n, -2.0, 2.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[m, n]),
        TraceNode::new(
            1,
            "transpose_na".into(),
            TraceOp::Transpose { dim0: 0, dim1: 1 },
            vec![0],
            vec![n, m],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        m * n,
    );

    let mut expected = vec![0.0_f32; m * n];
    for row in 0..m {
        for col in 0..n {
            expected[col * m + row] = input_data[row * n + col];
        }
    }
    assert_close("tiled_transpose_non_aligned", &result, &expected, 0.0);
}
