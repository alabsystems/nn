// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared test helpers for `CompiledModel` integration tests.
//!
//! Deduplicated from 7 test files per #2277. Include via
//! `#[path = "compiled_model_test_helpers.rs"] mod helpers;`

#![allow(dead_code, unreachable_pub)]

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;
use nn_metal::compiled_model::CompiledModel;
use nn_metal::MetalElement;

/// Build a `TraceNode` for an input with the given id and shape.
pub fn input_node(id: u64, shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        format!("input_{id}"),
        TraceOp::Input,
        vec![],
        shape.to_vec(),
        DType::F32,
    )
}

/// Build a `TraceNode` for a unary op (single input).
pub fn unary_node(id: u64, name: &str, op: TraceOp, input_id: u64, shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        op,
        vec![input_id],
        shape.to_vec(),
        DType::F32,
    )
}

/// Build a `TraceNode` for a binary op (two inputs).
pub fn binary_node(
    id: u64,
    name: &str,
    op: TraceOp,
    lhs_id: u64,
    rhs_id: u64,
    shape: &[usize],
) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        op,
        vec![lhs_id, rhs_id],
        shape.to_vec(),
        DType::F32,
    )
}

/// Allocate a GPU buffer from the given f32 data.
pub fn create_input_buffer(
    cache: &nn_metal::PipelineCache,
    data: &[f32],
) -> nn_metal::MetalBuffer {
    cache
        .context()
        .create_buffer(data)
        .expect("create input buffer")
}

/// Read all floats from a GPU buffer (assumes offset-0 normalized output).
pub fn read_output(buf: &nn_metal::MetalBuffer) -> Vec<f32> {
    f32::read_buffer(buf).expect("read GPU output")
}

/// Read exactly `count` f32 elements from the start of a GPU buffer.
///
/// `CompiledModel::execute` may return arena-backed buffers whose total size
/// exceeds the logical output. This reads the expected output region only.
pub fn read_output_n(buf: &nn_metal::MetalBuffer, count: usize) -> Vec<f32> {
    f32::read_buffer_at_offset(buf, 0, count).expect("read GPU output")
}

/// Compile a graph, execute with given inputs, and return the output slice.
pub fn compile_and_run(
    cache: &nn_metal::PipelineCache,
    graph: ComputationGraph,
    input_bufs: &[&nn_metal::MetalBuffer],
    output_numel: usize,
) -> Vec<f32> {
    let compiled = CompiledModel::builder(&graph, cache)
        .build()
        .expect("compile");
    let out_buf = compiled.execute(cache, input_bufs).expect("execute");
    read_output_n(&out_buf, output_numel)
}

/// Assert element-wise closeness between two slices.
pub fn assert_close(label: &str, result: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(
        result.len(),
        expected.len(),
        "{label}: output length mismatch"
    );
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() <= tol,
            "{label}[{i}]: gpu={r}, expected={e}, diff={}",
            (r - e).abs()
        );
    }
}
