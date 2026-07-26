// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `CompiledModel` input validation — dtype and shape checks.
//!
//! Extracted from `compiled_model_test.rs` to keep that file under the
//! 1000-line test file limit. See #2192 for the dtype validation fix.

use std::sync::Arc;

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use nn_metal::compiled_model::CompiledModel;

use nn_metal::MetalElement;

use super::helpers::{create_input_buffer, input_node, unary_node};

/// Regression test for #2192: a BF16 DynTensor passed to an F32-traced model
/// must be rejected with DtypeMismatch, not silently interpreted as F32.
#[test]
fn test_compiled_model_execute_dyn_dtype_mismatch() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // F32-traced graph: input → relu
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    // Create a GPU tensor claiming BF16 dtype (buffer contents don't matter —
    // validation rejects before execution).
    let buf = create_input_buffer(&cache, &[1.0f32, 2.0, 3.0, 4.0]);
    let storage = nn_metal::MetalTensorData::new(buf);
    let bf16_tensor =
        DynTensor::from_gpu_storage(vec![4], DType::BF16, Arc::new(storage), Device::metal())
            .expect("create BF16 GPU DynTensor");

    let result = compiled.execute_dyn(&cache, &[&bf16_tensor]);
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("dtype mismatch"),
        "expected dtype mismatch error, got: {err_msg}"
    );
}

/// Regression test for #2213: a dispatch step whose kernel declares more
/// non-weight inputs than the graph edge_map provides must produce a
/// descriptive error, not silently drop the unresolvable input.
///
/// Strategy: compile a binary Add graph (2-input kernel) to get a valid
/// `CompiledPlan`, then use `from_plan` with a unary graph (1 edge per
/// non-input node). The kernel expects 2 non-weight inputs but only 1
/// graph edge exists → the else branch fires with a descriptive error.
#[test]
fn test_compiled_model_unresolvable_input_returns_error() {
    use nn_dsl::compile_trace_to_plan_with_fusion;

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Graph A: binary add (2 inputs → kernel expects 2 non-weight inputs).
    // Nodes: 0=input, 1=input, 2=add(0,1)
    let add_graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        TraceNode::new(
            2,
            "add_0".into(),
            TraceOp::Add,
            vec![0, 1],
            vec![4],
            DType::F32,
        ),
    ]);
    let plan = compile_trace_to_plan_with_fusion(&add_graph).expect("compile add plan");

    // Graph B: same node count (3 nodes), same 2 input nodes, but
    // node 2 has only 1 input edge instead of 2.
    // edge_map[2] = [0] — only 1 graph edge, but the Add kernel needs 2.
    let mismatched_graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        unary_node(2, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);

    // from_plan builds edge_map from mismatched_graph: step 2 has 1 edge [0].
    // But the plan's step 2 is an Add dispatch kernel with 2 non-weight
    // inputs. The second input will be unresolvable.
    let compiled = CompiledModel::from_plan(&plan, &mismatched_graph, &cache).expect("from_plan");
    let buf_a = create_input_buffer(&cache, &[1.0f32, 2.0, 3.0, 4.0]);
    let buf_b = create_input_buffer(&cache, &[5.0f32, 6.0, 7.0, 8.0]);
    let result = compiled.execute(&cache, &[&buf_a, &buf_b]);
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("neither a weight nor a graph edge"),
        "expected unresolvable input error, got: {err_msg}"
    );
}

/// Input shape mismatch is also caught before execution.
#[test]
fn test_compiled_model_execute_dyn_shape_mismatch() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Graph expects shape [4]
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    // Create a GPU tensor with shape [8] instead of [4]
    let buf = create_input_buffer(&cache, &[1.0f32; 8]);
    let storage = nn_metal::MetalTensorData::new(buf);
    let wrong_shape_tensor =
        DynTensor::from_gpu_storage(vec![8], DType::F32, Arc::new(storage), Device::metal())
            .expect("create wrong-shape GPU DynTensor");

    let result = compiled.execute_dyn(&cache, &[&wrong_shape_tensor]);
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("shape mismatch"),
        "expected shape mismatch error, got: {err_msg}"
    );
}

/// Regression test for #2344: ConstantValue with f64 that overflows f32
/// must be rejected, not silently produce Infinity.
///
/// Since #2338 (constant pre-upload at construction), the non-finite check
/// triggers at `builder().build()` time rather than execution time.
#[test]
fn test_compiled_model_constant_f64_overflow_rejected() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Graph: input[4] → relu → add(relu, constant(1e308)) → output
    // 1e308 is a valid f64 but overflows to f32::INFINITY.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
        TraceNode::new(
            2,
            "const_overflow".into(),
            TraceOp::Constant { value: 1e308 },
            vec![],
            vec![4],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "add_0".into(),
            TraceOp::Add,
            vec![1, 2],
            vec![4],
            DType::F32,
        ),
    ]);

    // Non-finite constant rejected at construction time (#2338).
    let err = CompiledModel::builder(&graph, &cache)
        .build()
        .err()
        .expect("build should reject non-finite constant");
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("not finite"),
        "expected 'not finite' error for f64→f32 overflow, got: {err_msg}"
    );
}

/// Regression test for #2344: ConstantValue with NaN must be rejected.
///
/// Since #2338 (constant pre-upload at construction), the non-finite check
/// triggers at `builder().build()` time rather than execution time.
#[test]
fn test_compiled_model_constant_nan_rejected() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // A standalone constant NaN node as the entire graph.
    let graph = ComputationGraph::from_nodes(vec![TraceNode::new(
        0,
        "const_nan".into(),
        TraceOp::Constant { value: f64::NAN },
        vec![],
        vec![2],
        DType::F32,
    )]);

    // Non-finite constant rejected at construction time (#2338).
    let err = CompiledModel::builder(&graph, &cache)
        .build()
        .err()
        .expect("build should reject non-finite constant");
    let err_msg = err.to_string();
    assert!(
        err_msg.contains("not finite"),
        "expected 'not finite' error for NaN constant, got: {err_msg}"
    );
}

/// Regression test for #2345: validate_dyn_inputs() used zip() which
/// silently truncated when input count mismatched spec count. Now an
/// explicit InputCountMismatch check fires before any shape/dtype
/// validation.
#[test]
fn test_compiled_model_execute_dyn_too_few_inputs() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Graph with 2 inputs: input0[4] + input1[4] → add → output
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        TraceNode::new(
            2,
            "add_0".into(),
            TraceOp::Add,
            vec![0, 1],
            vec![4],
            DType::F32,
        ),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    // Pass only 1 input — should fail with InputCountMismatch
    let buf = create_input_buffer(&cache, &[1.0f32, 2.0, 3.0, 4.0]);
    let storage = nn_metal::MetalTensorData::new(buf);
    let tensor =
        DynTensor::from_gpu_storage(vec![4], DType::F32, Arc::new(storage), Device::metal())
            .expect("create GPU DynTensor");

    let result = compiled.execute_dyn(&cache, &[&tensor]);
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("expected") && err_msg.contains("inputs, got"),
        "expected input count mismatch error, got: {err_msg}"
    );
}

/// Too many inputs also caught by the early count check.
#[test]
fn test_compiled_model_execute_dyn_too_many_inputs() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Graph with 1 input: input[4] → relu → output
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    // Pass 2 inputs — should fail with InputCountMismatch
    let buf_a = create_input_buffer(&cache, &[1.0f32, 2.0, 3.0, 4.0]);
    let storage_a = nn_metal::MetalTensorData::new(buf_a);
    let tensor_a =
        DynTensor::from_gpu_storage(vec![4], DType::F32, Arc::new(storage_a), Device::metal())
            .expect("create GPU DynTensor a");

    let buf_b = create_input_buffer(&cache, &[5.0f32, 6.0, 7.0, 8.0]);
    let storage_b = nn_metal::MetalTensorData::new(buf_b);
    let tensor_b =
        DynTensor::from_gpu_storage(vec![4], DType::F32, Arc::new(storage_b), Device::metal())
            .expect("create GPU DynTensor b");

    let result = compiled.execute_dyn(&cache, &[&tensor_a, &tensor_b]);
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("expected") && err_msg.contains("inputs, got"),
        "expected input count mismatch error, got: {err_msg}"
    );
}

/// Normal finite constants should still work correctly.
#[test]
fn test_compiled_model_constant_finite_succeeds() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Graph: input[4] + constant(10.0) → output
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        TraceNode::new(
            1,
            "const_ten".into(),
            TraceOp::Constant { value: 10.0 },
            vec![],
            vec![4],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "add_0".into(),
            TraceOp::Add,
            vec![0, 1],
            vec![4],
            DType::F32,
        ),
    ]);

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");
    let buf = create_input_buffer(&cache, &[1.0f32, 2.0, 3.0, 4.0]);
    let output_buf = compiled
        .execute(&cache, &[&buf])
        .expect("execute should succeed");

    // Read back and verify: [1+10, 2+10, 3+10, 4+10] = [11, 12, 13, 14]
    let output_slice: &[f32] = output_buf.contents().expect("read output");
    let expected = [11.0f32, 12.0, 13.0, 14.0];
    for (i, (g, e)) in output_slice.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-5,
            "finite_constant[{i}]: got {g}, expected {e}"
        );
    }
}

/// Regression test for #2268: narrow'd GPU DynTensor input must preserve
/// byte_offset through execute_dyn. A dim-0 narrow creates a zero-copy
/// view with non-zero byte_offset. If the offset is dropped, the compiled
/// model reads from the wrong memory location, producing silent wrong results.
#[test]
fn test_compiled_model_execute_dyn_narrow_input_2268() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Build a simple relu graph: input [2, 4] -> relu -> output [2, 4].
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[2, 4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile relu");

    // Create a [3, 4] GPU tensor with distinctive data per row so we can
    // detect if the wrong rows are read.
    //   Row 0: [100, 200, 300, 400]  -- sentinel values (should NOT appear)
    //   Row 1: [-1,  2,  -3,  4]     -- narrow starts here
    //   Row 2: [5,  -6,   7, -8]     -- narrow ends here
    let full_data: Vec<f32> = vec![
        100.0, 200.0, 300.0, 400.0, // row 0 (sentinel)
        -1.0, 2.0, -3.0, 4.0, // row 1
        5.0, -6.0, 7.0, -8.0, // row 2
    ];
    let full_tensor =
        DynTensor::new(&full_data, &[3, 4], &Device::metal()).expect("create [3,4] GPU tensor");

    // Narrow dim 0, start=1, len=2 → [2, 4] view with byte_offset = 16.
    let narrowed = full_tensor.narrow(0, 1, 2).expect("narrow dim-0");
    assert_eq!(narrowed.dims(), &[2, 4]);

    // Execute via DynTensor interface — the fix in #2268 ensures the
    // byte_offset from the narrow is preserved through to the GPU dispatch.
    let output = compiled
        .execute_dyn(&cache, &[&narrowed])
        .expect("execute_dyn with narrow'd input");

    assert_eq!(output.dims(), &[2, 4]);
    assert_eq!(output.dtype(), DType::F32);

    let result = output.to_flat_vec::<f32>().expect("read output");

    // Expected: relu applied to the narrowed rows (rows 1-2 of original):
    //   relu([-1, 2, -3, 4]) = [0, 2, 0, 4]
    //   relu([5, -6, 7, -8]) = [5, 0, 7, 0]
    let expected: [f32; 8] = [0.0, 2.0, 0.0, 4.0, 5.0, 0.0, 7.0, 0.0];
    for (i, (g, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-5,
            "narrow_input_2268[{i}]: got {g}, expected {e} — \
             byte_offset likely dropped if values are from row 0"
        );
    }
}

/// Test for #2273: F16 input → compiled pipeline → F16 output round-trip.
///
/// Verifies that the compiled pipeline dispatches correctly when graph
/// nodes have DType::F16: step_scalar_types derive F16, MSL kernels
/// emit `half`, and GPU buffers contain f16 data.
#[test]
fn test_compiled_model_f16_relu_round_trip() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // F16-typed graph: input[4] → relu → output[4]
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![4],
            DType::F16,
        ),
        TraceNode::new(
            1,
            "relu_0".into(),
            TraceOp::Relu,
            vec![0],
            vec![4],
            DType::F16,
        ),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile F16 relu");
    assert_eq!(compiled.output_dtype(), DType::F16);

    // Create F16 GPU input buffer: [-1.0, 2.0, -3.0, 4.0]
    let f16_data: Vec<half::f16> = [-1.0f32, 2.0, -3.0, 4.0]
        .iter()
        .map(|&v| half::f16::from_f32(v))
        .collect();
    let input_buf =
        half::f16::create_buffer(cache.context(), &f16_data).expect("create F16 input buffer");

    let output_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute F16 relu");

    // Read back as f16 and verify relu: max(0, x)
    let output_f16 = half::f16::read_buffer_at_offset(&output_buf, 0, 4).expect("read F16 output");
    let result: Vec<f32> = output_f16.iter().map(|v| v.to_f32()).collect();
    let expected = [0.0f32, 2.0, 0.0, 4.0];
    for (i, (g, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() < 1e-3, "f16_relu[{i}]: got {g}, expected {e}");
    }
}

/// Test for #2273: F16 constant upload creates f16 buffer, not f32.
///
/// Verifies that `upload_constants` converts the f64 constant value to
/// f16 when the step's ScalarType is F16. Without the fix, the constant
/// buffer would contain f32 bytes interpreted as f16, producing garbage.
#[test]
fn test_compiled_model_f16_constant_add() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // F16-typed graph: input[4] + constant(10.0) → output[4]
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            vec![4],
            DType::F16,
        ),
        TraceNode::new(
            1,
            "const_ten".into(),
            TraceOp::Constant { value: 10.0 },
            vec![],
            vec![4],
            DType::F16,
        ),
        TraceNode::new(
            2,
            "add_0".into(),
            TraceOp::Add,
            vec![0, 1],
            vec![4],
            DType::F16,
        ),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile F16 add+constant");
    assert_eq!(compiled.output_dtype(), DType::F16);

    // Create F16 GPU input: [1.0, 2.0, 3.0, 4.0]
    let f16_data: Vec<half::f16> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .map(|&v| half::f16::from_f32(v))
        .collect();
    let input_buf =
        half::f16::create_buffer(cache.context(), &f16_data).expect("create F16 input buffer");

    let output_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute F16 add+constant");

    // Read back as f16: [1+10, 2+10, 3+10, 4+10] = [11, 12, 13, 14]
    let output_f16 = half::f16::read_buffer_at_offset(&output_buf, 0, 4).expect("read F16 output");
    let result: Vec<f32> = output_f16.iter().map(|v| v.to_f32()).collect();
    let expected = [11.0f32, 12.0, 13.0, 14.0];
    for (i, (g, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 0.1,
            "f16_constant_add[{i}]: got {g}, expected {e}"
        );
    }
}
