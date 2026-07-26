// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end compiled model tests: unary & binary elementwise ops.
//!
//! Continuation of `compiled_model_ops_e2e_4.rs` (tests 36+).
//! Fills proof coverage gap: these ops have compile dispatch but lacked
//! GPU E2E tests verifying trace → compile → Metal execute → CPU reference.
//!
//! Part of #3020.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use super::helpers::{assert_close, compile_and_run, create_input_buffer, input_node};

// -- Test 36: Sub (binary, two variable inputs) -------------------------------

/// Sub: [2, 4] - [2, 4] -> [2, 4]. Element-wise subtraction.
#[test]
fn test_compiled_binary_sub() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 4);
    let a_data = super::test_utils::rand_f32_vec(0x50B0_0001, rows * cols, -2.0, 2.0);
    let b_data = super::test_utils::rand_f32_vec(0x50B0_0002, rows * cols, -2.0, 2.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        input_node(1, &[rows, cols]),
        TraceNode::new(
            2,
            "sub_0".into(),
            TraceOp::Sub,
            vec![0, 1],
            vec![rows, cols],
            DType::F32,
        ),
    ]);

    let buf_a = create_input_buffer(&cache, &a_data);
    let buf_b = create_input_buffer(&cache, &b_data);
    let result = compile_and_run(&cache, graph, &[&buf_a, &buf_b], rows * cols);

    let expected: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(a, b)| a - b)
        .collect();
    assert_close("binary_sub", &result, &expected, 1e-6);
}

// -- Test 37: Div (binary, two variable inputs) -------------------------------

/// Div: [3, 4] / [3, 4] -> [3, 4]. Element-wise division.
/// Input B range avoids near-zero to prevent inf in reference.
#[test]
fn test_compiled_binary_div() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (3, 4);
    let a_data = super::test_utils::rand_f32_vec(0xD1F0_0001, rows * cols, -2.0, 2.0);
    let b_data = super::test_utils::rand_f32_vec(0xD1F0_0002, rows * cols, 0.5, 2.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        input_node(1, &[rows, cols]),
        TraceNode::new(
            2,
            "div_0".into(),
            TraceOp::Div,
            vec![0, 1],
            vec![rows, cols],
            DType::F32,
        ),
    ]);

    let buf_a = create_input_buffer(&cache, &a_data);
    let buf_b = create_input_buffer(&cache, &b_data);
    let result = compile_and_run(&cache, graph, &[&buf_a, &buf_b], rows * cols);

    let expected: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(a, b)| a / b)
        .collect();
    assert_close("binary_div", &result, &expected, 1e-5);
}

// -- Test 38: Log (unary) -----------------------------------------------------

/// Log: [2, 6] -> log -> [2, 6]. Natural logarithm.
/// Input range (0.1, 5.0) avoids log(0) and log(negative).
#[test]
fn test_compiled_log() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let input_data = super::test_utils::rand_f32_vec(0x10C0_0001, rows * cols, 0.1, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "log_0".into(),
            TraceOp::Log,
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

    let expected: Vec<f32> = input_data.iter().map(|x| x.ln()).collect();
    assert_close("log", &result, &expected, 1e-5);
}

// -- Test 39: Sqrt (unary) ----------------------------------------------------

/// Sqrt: [2, 6] -> sqrt -> [2, 6]. Square root.
/// Input range (0.01, 10.0) avoids sqrt(negative).
#[test]
fn test_compiled_sqrt() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let input_data = super::test_utils::rand_f32_vec(0x5080_0001, rows * cols, 0.01, 10.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "sqrt_0".into(),
            TraceOp::Sqrt,
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

    let expected: Vec<f32> = input_data.iter().map(|x| x.sqrt()).collect();
    assert_close("sqrt", &result, &expected, 1e-6);
}

// -- Test 40: Neg (unary) -----------------------------------------------------

/// Neg: [2, 4] -> neg -> [2, 4]. Element-wise negation.
#[test]
fn test_compiled_neg() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 4);
    let input_data = super::test_utils::rand_f32_vec(0x0EC0_0001, rows * cols, -3.0, 3.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "neg_0".into(),
            TraceOp::Neg,
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

    let expected: Vec<f32> = input_data.iter().map(|x| -x).collect();
    assert_close("neg", &result, &expected, 1e-7);
}

// -- Test 41: Abs (unary) -----------------------------------------------------

/// Abs: [3, 4] -> abs -> [3, 4]. Element-wise absolute value.
#[test]
fn test_compiled_abs() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (3, 4);
    let input_data = super::test_utils::rand_f32_vec(0xAB50_0001, rows * cols, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "abs_0".into(),
            TraceOp::Abs,
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

    let expected: Vec<f32> = input_data.iter().map(|x| x.abs()).collect();
    assert_close("abs", &result, &expected, 1e-7);
}

// -- Test 42: Sqr (unary) -----------------------------------------------------

/// Sqr: [2, 6] -> sqr -> [2, 6]. Element-wise square (x*x).
#[test]
fn test_compiled_sqr() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let input_data = super::test_utils::rand_f32_vec(0x5020_0001, rows * cols, -3.0, 3.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "sqr_0".into(),
            TraceOp::Sqr,
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

    let expected: Vec<f32> = input_data.iter().map(|x| x * x).collect();
    assert_close("sqr", &result, &expected, 1e-6);
}

// -- Test 43: Recip (unary) ---------------------------------------------------

/// Recip: [2, 4] -> recip -> [2, 4]. Element-wise reciprocal (1/x).
/// Input range avoids near-zero to prevent inf.
#[test]
fn test_compiled_recip() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 4);
    let input_data = super::test_utils::rand_f32_vec(0x0EC1_0001, rows * cols, 0.5, 4.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "recip_0".into(),
            TraceOp::Recip,
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

    let expected: Vec<f32> = input_data.iter().map(|x| 1.0 / x).collect();
    assert_close("recip", &result, &expected, 1e-5);
}

// -- Test 44: Sin (unary) -----------------------------------------------------

/// Sin: [2, 6] -> sin -> [2, 6]. Element-wise sine.
#[test]
fn test_compiled_sin() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let input_data = super::test_utils::rand_f32_vec(
        0x51D0_0001,
        rows * cols,
        -std::f32::consts::PI,
        std::f32::consts::PI,
    );

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "sin_0".into(),
            TraceOp::Sin,
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

    let expected: Vec<f32> = input_data.iter().map(|x| x.sin()).collect();
    assert_close("sin", &result, &expected, 1e-5);
}

// -- Test 45: Cos (unary) -----------------------------------------------------

/// Cos: [2, 6] -> cos -> [2, 6]. Element-wise cosine.
#[test]
fn test_compiled_cos() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let input_data = super::test_utils::rand_f32_vec(
        0xC050_0001,
        rows * cols,
        -std::f32::consts::PI,
        std::f32::consts::PI,
    );

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "cos_0".into(),
            TraceOp::Cos,
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

    let expected: Vec<f32> = input_data.iter().map(|x| x.cos()).collect();
    assert_close("cos", &result, &expected, 1e-5);
}

// -- Test 46: ReduceMin (dim=1) -----------------------------------------------

/// ReduceMin(dim=1): [3, 5] -> reduce_min(dim=1) -> [3, 1]. Minimum over columns.
#[test]
fn test_compiled_reduce_min() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (3, 5);
    let input_data = super::test_utils::rand_f32_vec(0x0E10_0001, rows * cols, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "reduce_min_0".into(),
            TraceOp::ReduceMin {
                dim: 1,
                keepdim: true,
            },
            vec![0],
            vec![rows, 1],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        rows, // output is [3, 1] = 3 elements
    );

    let expected: Vec<f32> = (0..rows)
        .map(|r| {
            input_data[r * cols..(r + 1) * cols]
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min)
        })
        .collect();
    assert_close("reduce_min", &result, &expected, 1e-6);
}

// -- Test 47: GeluErf (unary) -------------------------------------------------

/// GeluErf: [2, 6] -> gelu_erf -> [2, 6]. GELU with erf implementation.
/// gelu_erf(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
#[test]
fn test_compiled_gelu_erf() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let input_data = super::test_utils::rand_f32_vec(0xCE0F_0001, rows * cols, -3.0, 3.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "gelu_erf_0".into(),
            TraceOp::GeluErf,
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

    // Reference: 0.5 * x * (1 + erf(x / sqrt(2)))
    // Use the tanh approximation for erf since Rust doesn't have erf in std.
    // erf(z) ≈ tanh(sqrt(2/π) * (z + 0.044715 * z³)) — same as GELU fast approx.
    // Actually for exact erf we use the series. But the GPU uses precise erf, so
    // let's use the same formula the GPU uses: 0.5 * x * (1 + erf(x/sqrt(2))).
    // Rust libm provides erf.
    let sqrt2 = std::f32::consts::SQRT_2;
    let expected: Vec<f32> = input_data
        .iter()
        .map(|&x| {
            // Compute erf via the Horner approximation (Abramowitz and Stegun 7.1.26)
            let z = x / sqrt2;
            let t = 1.0 / (1.0 + 0.3275911 * z.abs());
            let poly = t
                * (0.254_829_6
                    + t * (-0.284_496_72
                        + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
            let erf_z = 1.0 - poly * (-z * z).exp();
            let erf_val = if z >= 0.0 { erf_z } else { -erf_z };
            0.5 * x * (1.0 + erf_val)
        })
        .collect();
    assert_close("gelu_erf", &result, &expected, 1e-4);
}
