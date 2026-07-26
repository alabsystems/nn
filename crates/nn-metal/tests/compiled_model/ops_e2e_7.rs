// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end compiled model tests: binary min/max, clamp, powf, floor.
//!
//! Continuation of `ops_e2e_5.rs` (tests 48+).
//! Fills proof coverage gaps: these ops have compile dispatch but lacked
//! GPU E2E tests verifying trace → compile → Metal execute → CPU reference.
//!
//! Part of #3020.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;

use super::helpers::{assert_close, compile_and_run, create_input_buffer, input_node};

// -- Test 48: Maximum (binary) ------------------------------------------------

/// Maximum: element-wise max([2, 6], [2, 6]) -> [2, 6].
#[test]
fn test_compiled_binary_maximum() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let a_data = super::test_utils::rand_f32_vec(0xBA10_0001, rows * cols, -3.0, 3.0);
    let b_data = super::test_utils::rand_f32_vec(0xBA10_0002, rows * cols, -3.0, 3.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        input_node(1, &[rows, cols]),
        TraceNode::new(
            2,
            "maximum_0".into(),
            TraceOp::Maximum,
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
        .map(|(a, b)| a.max(*b))
        .collect();
    assert_close("binary_maximum", &result, &expected, 0.0);
}

// -- Test 49: Minimum (binary) ------------------------------------------------

/// Minimum: element-wise min([2, 6], [2, 6]) -> [2, 6].
#[test]
fn test_compiled_binary_minimum() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let a_data = super::test_utils::rand_f32_vec(0xB110_0001, rows * cols, -3.0, 3.0);
    let b_data = super::test_utils::rand_f32_vec(0xB110_0002, rows * cols, -3.0, 3.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        input_node(1, &[rows, cols]),
        TraceNode::new(
            2,
            "minimum_0".into(),
            TraceOp::Minimum,
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
        .map(|(a, b)| a.min(*b))
        .collect();
    assert_close("binary_minimum", &result, &expected, 0.0);
}

// -- Test 50: Clamp (min + max) -----------------------------------------------

/// Clamp: [2, 6] -> clamp(-1.0, 1.5) -> [2, 6].
#[test]
fn test_compiled_clamp() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let input_data = super::test_utils::rand_f32_vec(0xC1A0_0001, rows * cols, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "clamp_0".into(),
            TraceOp::Clamp {
                min: Some(-1.0),
                max: Some(1.5),
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
        rows * cols,
    );

    let expected: Vec<f32> = input_data.iter().map(|x| x.clamp(-1.0, 1.5)).collect();
    assert_close("clamp", &result, &expected, 0.0);
}

// -- Test 51: Clamp (min only) ------------------------------------------------

/// Clamp(min=0.0, max=None): equivalent to relu for negative values.
#[test]
fn test_compiled_clamp_min_only() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let input_data = super::test_utils::rand_f32_vec(0xC1A0_0002, rows * cols, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "clamp_min_0".into(),
            TraceOp::Clamp {
                min: Some(0.0),
                max: None,
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
        rows * cols,
    );

    let expected: Vec<f32> = input_data.iter().map(|x| x.max(0.0)).collect();
    assert_close("clamp_min_only", &result, &expected, 0.0);
}

// -- Test 52: Powf (exponent=3.0) ---------------------------------------------

/// Powf: [2, 6] -> x^3 -> [2, 6]. Cubic power.
/// Input range (0.1, 3.0) avoids negative base for fractional exponents.
#[test]
fn test_compiled_powf_cubic() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let input_data = super::test_utils::rand_f32_vec(0xB0F0_0001, rows * cols, 0.1, 3.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "powf_0".into(),
            TraceOp::Powf { exponent: 3.0 },
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

    let expected: Vec<f32> = input_data.iter().map(|x| x.powf(3.0)).collect();
    assert_close("powf_cubic", &result, &expected, 1e-4);
}

// -- Test 53: Powf (exponent=0.5, sqrt) ---------------------------------------

/// Powf(0.5): [2, 6] -> x^0.5 -> [2, 6]. Should match sqrt.
#[test]
fn test_compiled_powf_sqrt() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let input_data = super::test_utils::rand_f32_vec(0xB0F0_0002, rows * cols, 0.01, 10.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "powf_sqrt_0".into(),
            TraceOp::Powf { exponent: 0.5 },
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
    assert_close("powf_sqrt", &result, &expected, 1e-6);
}

// -- Test 54: Floor (unary) ---------------------------------------------------

/// Floor: [2, 6] -> floor -> [2, 6]. Round toward negative infinity.
#[test]
fn test_compiled_floor() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let input_data = super::test_utils::rand_f32_vec(0xF100_0001, rows * cols, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "floor_0".into(),
            TraceOp::Floor,
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

    let expected: Vec<f32> = input_data.iter().map(|x| x.floor()).collect();
    assert_close("floor", &result, &expected, 0.0);
}

// -- Test 55: Round (unary) ---------------------------------------------------

/// Round: [2, 6] -> round -> [2, 6]. Round to nearest integer.
#[test]
fn test_compiled_round() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let input_data = super::test_utils::rand_f32_vec(0x00D0_0001, rows * cols, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "round_0".into(),
            TraceOp::Round,
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

    let expected: Vec<f32> = input_data.iter().map(|x| x.round()).collect();
    assert_close("round", &result, &expected, 0.0);
}

// -- Test 56: Fract (unary) ---------------------------------------------------

/// Fract: [2, 6] -> fract -> [2, 6]. Fractional part.
///
/// Metal `fract(x)` = `x - floor(x)` (always in [0, 1)).
/// Rust `f32::fract()` preserves sign (-1.9.fract() = -0.9).
/// The compiled kernel uses the Metal definition.
#[test]
fn test_compiled_fract() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let input_data = super::test_utils::rand_f32_vec(0xF0AC_0001, rows * cols, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "fract_0".into(),
            TraceOp::Fract,
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

    // Metal fract(x) = x - floor(x), always in [0, 1).
    let expected: Vec<f32> = input_data.iter().map(|x| x - x.floor()).collect();
    assert_close("fract", &result, &expected, 1e-6);
}
