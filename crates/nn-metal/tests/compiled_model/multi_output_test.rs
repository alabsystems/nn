// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-output integration tests for `CompiledModel` (#2184).
//!
//! Tests verify that `CompiledModel` can return multiple output buffers
//! via `execute_outputs()` and `execute_dyn_outputs()`, and that
//! single-output models continue to work unchanged (backward compat).

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;
use nn_metal::compiled_model::CompiledModel;

use super::helpers::{assert_close, create_input_buffer, input_node, read_output, unary_node};

fn narrow_node(
    id: u64,
    name: &str,
    input_id: u64,
    shape: &[usize],
    dim: usize,
    start: usize,
    length: usize,
) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        TraceOp::Narrow { dim, start, length },
        vec![input_id],
        shape.to_vec(),
        DType::F32,
    )
}

// -- Tests: multi-output support (#2184 AC2-AC5) -----------------------------

/// AC2+AC3: multi-output graph returns multiple output buffers via
/// `execute_outputs()`.
///
/// Uses a diamond topology (input fans out to relu and sigmoid on
/// independent branches) so fusion does not merge the two output steps
/// into one.
#[test]
fn test_compiled_model_multi_output() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Diamond: input(0) → relu(1), input(0) → sigmoid(2)
    // Both outputs take input directly (no chain between them).
    // from_nodes marks last node (sigmoid, id=2) as output.
    // mark_output(1) adds relu as a second output.
    let mut graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 0, &[4]),
    ]);
    let _ = graph.mark_output(1); // relu is also an output

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile multi-output");
    assert_eq!(compiled.num_outputs(), 2, "sigmoid + relu = 2 outputs");

    let input_data = [1.0_f32, -2.0, 0.0, 3.0];
    let input_buf = create_input_buffer(&cache, &input_data);
    let out_bufs = compiled
        .execute_outputs(&cache, &[&input_buf])
        .expect("execute_outputs");
    assert_eq!(out_bufs.len(), 2, "should return 2 output buffers");

    // Output order: sigmoid (from from_nodes auto-mark), relu (explicit mark).
    let sigmoid_result = read_output(&out_bufs[0]);
    let relu_result = read_output(&out_bufs[1]);

    let expected_relu: Vec<f32> = input_data.iter().map(|x| x.max(0.0)).collect();
    let expected_sigmoid: Vec<f32> = input_data
        .iter()
        .map(|x| 1.0 / (1.0 + (-x).exp()))
        .collect();

    for (i, (r, e)) in relu_result.iter().zip(expected_relu.iter()).enumerate() {
        assert!((r - e).abs() < 1e-6, "relu[{i}]: got {r}, expected {e}");
    }
    for (i, (r, e)) in sigmoid_result
        .iter()
        .zip(expected_sigmoid.iter())
        .enumerate()
    {
        assert!((r - e).abs() < 1e-5, "sigmoid[{i}]: got {r}, expected {e}");
    }

    // execute() returns the primary output (last = relu).
    let primary = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute primary");
    let primary_data = read_output(&primary);
    assert_eq!(
        primary_data, relu_result,
        "primary should be last output (relu)"
    );
}

/// AC5: single-output models work unchanged (backward compat).
#[test]
fn test_compiled_model_single_output_backward_compat() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile single");
    assert_eq!(compiled.num_outputs(), 1);

    let input_data = [1.0_f32, -2.0, 3.0, -4.0];
    let input_buf = create_input_buffer(&cache, &input_data);

    // execute_outputs returns 1-element vec matching execute().
    let out_bufs = compiled
        .execute_outputs(&cache, &[&input_buf])
        .expect("execute_outputs");
    assert_eq!(out_bufs.len(), 1);

    let out_single = compiled.execute(&cache, &[&input_buf]).expect("execute");
    let data_multi = read_output(&out_bufs[0]);
    let data_single = read_output(&out_single);
    assert_eq!(data_multi, data_single, "execute_outputs[0] == execute()");
}

/// Regression guard for internal non-zero-offset NarrowView consumers.
///
/// The narrow is taken from an intermediate GPU buffer (`relu_0`), not from
/// the external input buffer. This exercises the exact path used by Kokoro's
/// phase head: `conv_post -> narrow(start>0) -> sin`.
#[test]
fn test_compiled_model_internal_narrow_view_nonzero_offset_single_output() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 6]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[1, 6]),
        narrow_node(2, "narrow_0", 1, &[1, 3], 1, 3, 3),
        unary_node(3, "sin_0", TraceOp::Sin, 2, &[1, 3]),
    ]);

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile internal narrow-view graph");

    let input_data = [-1.0_f32, 0.5, 1.0, 0.25, -0.5, 1.5];
    let input_buf = create_input_buffer(&cache, &input_data);
    let out_buf = compiled.execute(&cache, &[&input_buf]).expect("execute");
    let result = read_output(&out_buf);

    let relu: Vec<f32> = input_data.iter().map(|x| x.max(0.0)).collect();
    let expected: Vec<f32> = relu[3..].iter().map(|x| x.sin()).collect();
    assert_close("internal_narrow_view_sin", &result, &expected, 1e-6);
}

/// Regression guard for Kokoro-style split heads:
/// shared buffer -> narrow(offset 0) and narrow(offset > 0) -> sin.
///
/// The default output is the phase branch (`sin_0`), while `mark_output`
/// adds the magnitude branch (`narrow_mag`). This matches the way the
/// compiled generator marks magnitude + phase.
#[test]
fn test_compiled_model_multi_output_split_head_with_offset_phase_branch() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let mut graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[1, 6]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[1, 6]),
        narrow_node(2, "narrow_mag", 1, &[1, 3], 1, 0, 3),
        narrow_node(3, "narrow_phase", 1, &[1, 3], 1, 3, 3),
        unary_node(4, "sin_0", TraceOp::Sin, 3, &[1, 3]),
    ]);
    let _ = graph.mark_output(2);

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile split-head multi-output graph");
    assert_eq!(compiled.num_outputs(), 2, "phase + magnitude = 2 outputs");

    let input_data = [-1.0_f32, 0.5, 1.0, 0.25, -0.5, 1.5];
    let input_buf = create_input_buffer(&cache, &input_data);
    let out_bufs = compiled
        .execute_outputs(&cache, &[&input_buf])
        .expect("execute_outputs");
    assert_eq!(out_bufs.len(), 2, "should return 2 output buffers");

    let phase_result = read_output(&out_bufs[0]);
    let mag_result = read_output(&out_bufs[1]);

    let relu: Vec<f32> = input_data.iter().map(|x| x.max(0.0)).collect();
    let expected_mag = relu[..3].to_vec();
    let expected_phase: Vec<f32> = relu[3..].iter().map(|x| x.sin()).collect();

    assert_close("split_head_phase", &phase_result, &expected_phase, 1e-6);
    assert_close("split_head_mag", &mag_result, &expected_mag, 1e-6);
}
