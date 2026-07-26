// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for NarrowView upper-bound validation (#3266).
//!
//! The runtime validation in `compiled_model_execute_steps.rs` ensures that
//! NarrowView byte_offset + data_bytes does not exceed the source buffer.
//! These tests exercise the error paths that prevent buffer overread.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;
use nn_dsl::trace_compile::CompiledStep;
use nn_metal::compiled_model::CompiledModel;

use super::helpers::{create_input_buffer, input_node, unary_node};

/// Regression test for #3266: NarrowView with byte_offset exceeding the
/// source buffer length must produce a descriptive error, not read out of
/// bounds.
///
/// Strategy: compile a graph that expects input[16] and narrows at start=8.
/// Then execute with a 4-element buffer (16 bytes). The NarrowView needs
/// bytes [32..48] but only 16 bytes exist.
#[test]
fn test_narrow_view_out_of_bounds_small_buffer() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Graph: input[16] → narrow(dim=0, start=8, len=4) → relu → output
    // NarrowView byte_offset = 8 * sizeof(f32) = 32 bytes.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[16]),
        TraceNode::new(
            1,
            "narrow_0".into(),
            TraceOp::Narrow {
                dim: 0,
                start: 8,
                length: 4,
            },
            vec![0],
            vec![4],
            DType::F32,
        ),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[4]),
    ]);

    // Verify compilation produces a NarrowView step.
    let steps = nn_dsl::trace_compile::compile_trace(&graph).expect("compile trace");
    let has_narrow = steps
        .iter()
        .any(|s| matches!(s, CompiledStep::NarrowView { .. }));
    assert!(has_narrow, "graph must produce a NarrowView step");

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile narrow model");

    // Provide a 4-element buffer (16 bytes) instead of the expected 16
    // elements (64 bytes). NarrowView needs bytes [32..48] in a 16-byte
    // buffer → out of bounds.
    let small_buf = create_input_buffer(&cache, &[1.0f32, 2.0, 3.0, 4.0]);
    let result = compiled.execute(&cache, &[&small_buf]);
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("NarrowView out of bounds"),
        "expected 'NarrowView out of bounds' error, got: {err_msg}"
    );
}

/// Boundary test: buffer is exactly 1 element (4 bytes) too small for the
/// narrowed view. Validates tight boundary detection, not just gross
/// overflows.
///
/// Graph: input[8] → narrow(dim=0, start=3, len=4) → relu
/// NarrowView: offset=12, numel=4, data_bytes=16, end=28.
/// Buffer: 6 elements = 24 bytes. end (28) > buf_len (24) → error.
#[test]
fn test_narrow_view_out_of_bounds_off_by_one() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[8]),
        TraceNode::new(
            1,
            "narrow_0".into(),
            TraceOp::Narrow {
                dim: 0,
                start: 3,
                length: 4,
            },
            vec![0],
            vec![4],
            DType::F32,
        ),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[4]),
    ]);

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile narrow model");

    // 6 elements = 24 bytes. NarrowView needs bytes [12..28]. 28 > 24 → error.
    let buf_6 = create_input_buffer(&cache, &[1.0f32; 6]);
    let result = compiled.execute(&cache, &[&buf_6]);
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("NarrowView out of bounds"),
        "expected 'NarrowView out of bounds' for off-by-one, got: {err_msg}"
    );
}

/// Positive test: buffer is exactly large enough for the narrowed view.
/// Confirms the validation does NOT reject valid executions.
///
/// Same graph as off-by-one above, but with 7 elements = 28 bytes.
/// NarrowView needs bytes [12..28]. end (28) == buf_len (28) → OK.
#[test]
fn test_narrow_view_exact_fit_succeeds() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[8]),
        TraceNode::new(
            1,
            "narrow_0".into(),
            TraceOp::Narrow {
                dim: 0,
                start: 3,
                length: 4,
            },
            vec![0],
            vec![4],
            DType::F32,
        ),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[4]),
    ]);

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile narrow model");

    // 7 elements = 28 bytes. NarrowView needs bytes [12..28]. Exactly fits.
    let buf_7 = create_input_buffer(&cache, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = compiled.execute(&cache, &[&buf_7]);
    assert!(
        result.is_ok(),
        "NarrowView with exact-fit buffer should succeed, got: {}",
        result.unwrap_err()
    );
}
