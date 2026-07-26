// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end compiled model tests: WhereCond, Compare, Expand, Flip, Atan2.
//!
//! Continuation of `ops_e2e_7.rs` / `ops_e2e_8.rs`.
//! Fills proof coverage gaps: these ops have compile dispatch but lacked
//! GPU E2E tests verifying trace → compile → Metal execute → CPU reference.
//!
//! Part of #3020.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::dyn_tensor::CompareOp;
use nn_core::DType;

use super::helpers::{assert_close, compile_and_run, create_input_buffer, input_node};

// -- Test 57: WhereCond (ternary select) --------------------------------------

/// WhereCond: select element-wise from two tensors based on mask.
/// mask=1.0 → on_true, mask=0.0 → on_false.
/// Decomposed as `mask * on_true + (1 - mask) * on_false`.
#[test]
fn test_compiled_where_cond() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let n = rows * cols;
    // mask: alternating 0.0 and 1.0
    let mask_data: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
    let true_data = super::test_utils::rand_f32_vec(0xCDE0_0001, n, -5.0, 5.0);
    let false_data = super::test_utils::rand_f32_vec(0xCDE0_0002, n, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]), // mask
        input_node(1, &[rows, cols]), // on_true
        input_node(2, &[rows, cols]), // on_false
        TraceNode::new(
            3,
            "where_0".into(),
            TraceOp::WhereCond,
            vec![0, 1, 2],
            vec![rows, cols],
            DType::F32,
        ),
    ]);

    let buf_mask = create_input_buffer(&cache, &mask_data);
    let buf_true = create_input_buffer(&cache, &true_data);
    let buf_false = create_input_buffer(&cache, &false_data);
    let result = compile_and_run(&cache, graph, &[&buf_mask, &buf_true, &buf_false], n);

    // mask * on_true + (1 - mask) * on_false
    let expected: Vec<f32> = mask_data
        .iter()
        .zip(true_data.iter())
        .zip(false_data.iter())
        .map(|((m, t), f)| m * t + (1.0 - m) * f)
        .collect();
    assert_close("where_cond", &result, &expected, 1e-6);
}

// -- Test 58: Compare (scalar Gt) --------------------------------------------

/// Compare(Gt, 0.0): produces F32 mask — 1.0 where input > 0.0, else 0.0.
#[test]
fn test_compiled_compare_gt() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let n = rows * cols;
    let input_data = super::test_utils::rand_f32_vec(0xC0A0_0001, n, -3.0, 3.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "compare_gt_0".into(),
            TraceOp::Compare {
                op: CompareOp::Gt,
                value: 0.0,
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

    let expected: Vec<f32> = input_data
        .iter()
        .map(|x| if *x > 0.0 { 1.0 } else { 0.0 })
        .collect();
    assert_close("compare_gt", &result, &expected, 0.0);
}

// -- Test 59: Compare (scalar Le) --------------------------------------------

/// Compare(Le, 1.5): produces F32 mask — 1.0 where input <= 1.5.
#[test]
fn test_compiled_compare_le() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let n = rows * cols;
    let input_data = super::test_utils::rand_f32_vec(0xC0A0_0002, n, -3.0, 3.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "compare_le_0".into(),
            TraceOp::Compare {
                op: CompareOp::Le,
                value: 1.5,
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

    let expected: Vec<f32> = input_data
        .iter()
        .map(|x| if *x <= 1.5 { 1.0 } else { 0.0 })
        .collect();
    assert_close("compare_le", &result, &expected, 0.0);
}

// -- Test 60: Compare → WhereCond (pipeline) ---------------------------------

/// Pipeline: Compare(Gt, 0.0) → WhereCond. Equivalent to relu behavior
/// but through the compare+select decomposition.
#[test]
fn test_compiled_compare_then_where() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let n = rows * cols;
    let input_data = super::test_utils::rand_f32_vec(0xCDE1_0001, n, -3.0, 3.0);
    // zeros tensor for the false branch
    let zeros = vec![0.0f32; n];

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]), // input
        input_node(1, &[rows, cols]), // zeros
        TraceNode::new(
            2,
            "compare_gt_0".into(),
            TraceOp::Compare {
                op: CompareOp::Gt,
                value: 0.0,
            },
            vec![0],
            vec![rows, cols],
            DType::F32,
        ),
        // WhereCond(mask=compare, on_true=input, on_false=zeros)
        TraceNode::new(
            3,
            "where_0".into(),
            TraceOp::WhereCond,
            vec![2, 0, 1],
            vec![rows, cols],
            DType::F32,
        ),
    ]);

    let buf_input = create_input_buffer(&cache, &input_data);
    let buf_zeros = create_input_buffer(&cache, &zeros);
    let result = compile_and_run(&cache, graph, &[&buf_input, &buf_zeros], n);

    // relu-like: positive values pass through, negative become 0.0
    let expected: Vec<f32> = input_data.iter().map(|x| x.max(0.0)).collect();
    assert_close("compare_then_where", &result, &expected, 1e-6);
}

// -- Test 61: Expand (broadcast) ----------------------------------------------

/// Expand: [1, 6] → [4, 6]. Broadcast dim 0 from 1 to 4.
#[test]
fn test_compiled_expand() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let cols = 6;
    let target_rows = 4;
    let input_data = super::test_utils::rand_f32_vec(0xE100_0001, cols, -3.0, 3.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, cols]),
        TraceNode::new(
            1,
            "expand_0".into(),
            TraceOp::Expand {
                target_shape: vec![target_rows, cols],
            },
            vec![0],
            vec![target_rows, cols],
            DType::F32,
        ),
    ]);

    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        target_rows * cols,
    );

    // Each row should be a copy of the input row.
    let expected: Vec<f32> = (0..target_rows)
        .flat_map(|_| input_data.iter().copied())
        .collect();
    assert_close("expand", &result, &expected, 0.0);
}

// -- Test 62: Flip (reverse along dim) ----------------------------------------

/// Flip: [2, 6] → reverse along dim 1.
#[test]
fn test_compiled_flip() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let n = rows * cols;
    let input_data = super::test_utils::rand_f32_vec(0xF110_0001, n, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "flip_0".into(),
            TraceOp::Flip { dim: 1 },
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

    // Reverse each row along dim 1.
    let mut expected = vec![0.0f32; n];
    for r in 0..rows {
        for c in 0..cols {
            expected[r * cols + c] = input_data[r * cols + (cols - 1 - c)];
        }
    }
    assert_close("flip_dim1", &result, &expected, 0.0);
}

// -- Test 63: Flip (reverse along dim 0) --------------------------------------

/// Flip: [4, 3] → reverse along dim 0.
#[test]
fn test_compiled_flip_dim0() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (4, 3);
    let n = rows * cols;
    let input_data = super::test_utils::rand_f32_vec(0xF110_0002, n, -5.0, 5.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]),
        TraceNode::new(
            1,
            "flip_0".into(),
            TraceOp::Flip { dim: 0 },
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

    // Reverse rows along dim 0.
    let mut expected = vec![0.0f32; n];
    for r in 0..rows {
        for c in 0..cols {
            expected[r * cols + c] = input_data[(rows - 1 - r) * cols + c];
        }
    }
    assert_close("flip_dim0", &result, &expected, 0.0);
}

// -- Test 64: Atan2 (binary) --------------------------------------------------

/// Atan2: element-wise atan2(y, x) for [2, 6] inputs.
#[test]
fn test_compiled_atan2() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (rows, cols) = (2, 6);
    let n = rows * cols;
    let y_data = super::test_utils::rand_f32_vec(0xA720_0001, n, -3.0, 3.0);
    let x_data = super::test_utils::rand_f32_vec(0xA720_0002, n, -3.0, 3.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[rows, cols]), // y
        input_node(1, &[rows, cols]), // x
        TraceNode::new(
            2,
            "atan2_0".into(),
            TraceOp::Atan2,
            vec![0, 1],
            vec![rows, cols],
            DType::F32,
        ),
    ]);

    let buf_y = create_input_buffer(&cache, &y_data);
    let buf_x = create_input_buffer(&cache, &x_data);
    let result = compile_and_run(&cache, graph, &[&buf_y, &buf_x], n);

    let expected: Vec<f32> = y_data
        .iter()
        .zip(x_data.iter())
        .map(|(y, x)| y.atan2(*x))
        .collect();
    assert_close("atan2", &result, &expected, 1e-5);
}
