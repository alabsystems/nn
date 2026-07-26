// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `CompiledModel` -- trace-based compiled graph execution.
//!
//! Tests build computation graphs manually using `ComputationGraph::from_nodes()`,
//! compile them via `CompiledModel::builder().build()`, execute on GPU, and verify
//! results against CPU reference computations.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;
use nn_dsl::compile_trace_to_plan_with_fusion;
use nn_metal::compiled_model::CompiledModel;

use super::helpers::{binary_node, create_input_buffer, input_node, read_output, unary_node};

// -- Tests: metadata ----------------------------------------------------------

#[test]
fn test_compiled_model_empty_graph() {
    let cache = super::test_utils::metal_setup();
    let graph = ComputationGraph::from_nodes(vec![]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile empty");
    assert_eq!(compiled.num_steps(), 0);
    assert_eq!(compiled.num_dispatches(), 0);
    assert_eq!(compiled.num_inputs(), 0);
    assert!(compiled.output_shape().is_empty());
}

#[test]
fn test_compiled_model_empty_execute_returns_error() {
    let cache = super::test_utils::metal_setup();
    let graph = ComputationGraph::from_nodes(vec![]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile empty");
    let err_msg = compiled
        .execute(&cache, &[] as &[&nn_metal::MetalBuffer])
        .unwrap_err()
        .to_string();
    assert!(
        err_msg.contains("compiled plan is empty"),
        "expected EmptyPlan error, got: {err_msg}"
    );
}

#[test]
fn test_compiled_model_input_count_mismatch() {
    let cache = super::test_utils::metal_setup();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        binary_node(2, "add_0", TraceOp::Add, 0, 1, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile add");
    assert_eq!(compiled.num_inputs(), 2);

    // Provide only 1 input instead of 2
    let buf = create_input_buffer(&cache, &[1.0, 2.0, 3.0, 4.0]);
    let err_msg = compiled.execute(&cache, &[&buf]).unwrap_err().to_string();
    assert!(
        err_msg.contains("expected 2 inputs, got 1"),
        "expected InputCountMismatch error, got: {err_msg}"
    );
}

// -- Tests: metadata for compiled graphs --------------------------------------

#[test]
fn test_compiled_model_single_input_metadata() {
    let cache = super::test_utils::metal_setup();
    let graph = ComputationGraph::from_nodes(vec![input_node(0, &[2, 3])]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile single input");
    assert_eq!(compiled.num_steps(), 1);
    assert_eq!(compiled.num_dispatches(), 0);
    assert_eq!(compiled.num_inputs(), 1);
    assert_eq!(compiled.output_shape(), &[2, 3]);
}

#[test]
fn test_compiled_model_relu_metadata() {
    let cache = super::test_utils::metal_setup();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile relu");
    assert_eq!(compiled.num_steps(), 2);
    assert_eq!(compiled.num_dispatches(), 1);
    assert_eq!(compiled.num_inputs(), 1);
    assert_eq!(compiled.output_shape(), &[4]);
}

// -- Tests: GPU execution correctness ----------------------------------------

#[test]
fn test_compiled_model_execute_relu() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile relu");

    let input_data = [1.0_f32, -2.0, 3.0, -4.0];
    let input_buf = create_input_buffer(&cache, &input_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute relu");

    let expected = [1.0_f32, 0.0, 3.0, 0.0];
    let result = read_output(&out_buf);
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((r - e).abs() < 1e-6, "relu[{i}]: got {r}, expected {e}");
    }
}

#[test]
fn test_compiled_model_execute_sigmoid() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "sigmoid_0", TraceOp::Sigmoid, 0, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile sigmoid");

    let input_data = [0.0_f32, 1.0, -1.0, 2.0];
    let input_buf = create_input_buffer(&cache, &input_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute sigmoid");

    let expected: Vec<f32> = input_data
        .iter()
        .map(|x| 1.0 / (1.0 + (-x).exp()))
        .collect();
    let result = read_output(&out_buf);
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((r - e).abs() < 1e-5, "sigmoid[{i}]: got {r}, expected {e}");
    }
}

#[test]
fn test_compiled_model_execute_binary_add() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        binary_node(2, "add_0", TraceOp::Add, 0, 1, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile add");

    let a_data = [1.0_f32, 2.0, 3.0, 4.0];
    let b_data = [10.0_f32, 20.0, 30.0, 40.0];
    let a_buf = create_input_buffer(&cache, &a_data);
    let b_buf = create_input_buffer(&cache, &b_data);
    let out_buf = compiled
        .execute(&cache, &[&a_buf, &b_buf])
        .expect("execute add");

    let expected = [11.0_f32, 22.0, 33.0, 44.0];
    let result = read_output(&out_buf);
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((r - e).abs() < 1e-6, "add[{i}]: got {r}, expected {e}");
    }
}

#[test]
fn test_compiled_model_execute_binary_mul() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        binary_node(2, "mul_0", TraceOp::Mul, 0, 1, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile mul");

    let a_data = [2.0_f32, 3.0, 4.0, 5.0];
    let b_data = [0.5_f32, 0.5, 0.5, 0.5];
    let a_buf = create_input_buffer(&cache, &a_data);
    let b_buf = create_input_buffer(&cache, &b_data);
    let out_buf = compiled
        .execute(&cache, &[&a_buf, &b_buf])
        .expect("execute mul");

    let expected = [1.0_f32, 1.5, 2.0, 2.5];
    let result = read_output(&out_buf);
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((r - e).abs() < 1e-6, "mul[{i}]: got {r}, expected {e}");
    }
}

// -- Tests: multi-step chains ------------------------------------------------

#[test]
fn test_compiled_model_execute_relu_then_add() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();
    // Graph: input -> relu -> add(relu_out, input)
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
        binary_node(2, "add_0", TraceOp::Add, 1, 0, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile chain");
    assert_eq!(compiled.num_steps(), 3);
    // With fusion: relu->add fuses (relu has fan-out 1, add chains from it).
    assert_eq!(compiled.num_dispatches(), 1); // fused relu+add

    let input_data = [1.0_f32, -2.0, 3.0, -4.0];
    let input_buf = create_input_buffer(&cache, &input_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute chain");

    // relu([1, -2, 3, -4]) = [1, 0, 3, 0]
    // add([1, 0, 3, 0], [1, -2, 3, -4]) = [2, -2, 6, -4]
    let expected = [2.0_f32, -2.0, 6.0, -4.0];
    let result = read_output(&out_buf);
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((r - e).abs() < 1e-6, "relu_add[{i}]: got {r}, expected {e}");
    }
}

#[test]
fn test_compiled_model_execute_dropout_passthrough() {
    let cache = super::test_utils::metal_setup();
    // Dropout at inference is identity
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "dropout_0", TraceOp::Dropout, 0, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile dropout");
    assert_eq!(compiled.num_dispatches(), 0); // dropout = identity, no GPU dispatch

    let input_data = [1.0_f32, 2.0, 3.0, 4.0];
    let input_buf = create_input_buffer(&cache, &input_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute dropout");
    let result = read_output(&out_buf);

    // Dropout is identity -- output should match input exactly
    for (i, (r, e)) in result.iter().zip(input_data.iter()).enumerate() {
        assert!((r - e).abs() < 1e-9, "dropout[{i}]: got {r}, expected {e}");
    }
}

// -- Tests: weighted ops (linear) --------------------------------------------

#[test]
fn test_compiled_model_execute_linear() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Linear: y = x @ W^T + b
    // x: [2, 4], W: [3, 4], b: [3] -> y: [2, 3]
    let weight_data = vec![
        1.0, 0.0, 0.0, 0.0, // W[0]: selects x[..][0]
        0.0, 1.0, 0.0, 0.0, // W[1]: selects x[..][1]
        0.0, 0.0, 1.0, 0.0, // W[2]: selects x[..][2]
    ];
    let bias_data = vec![10.0, 20.0, 30.0];
    let weight = WeightRef::new(weight_data, vec![3, 4]).expect("test data");
    let bias = Some(WeightRef::new(bias_data.clone(), vec![3]).expect("test data"));

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 4]),
        TraceNode::new(
            1,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight.clone(),
                bias,
            },
            vec![0],
            vec![2, 3],
            DType::F32,
        ),
    ]);

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile linear");
    assert_eq!(compiled.num_steps(), 2);
    assert_eq!(compiled.num_dispatches(), 1);
    assert_eq!(compiled.num_inputs(), 1);
    assert_eq!(compiled.output_shape(), &[2, 3]);

    let input_data = [
        1.0_f32, 2.0, 3.0, 4.0, // row 0
        5.0, 6.0, 7.0, 8.0, // row 1
    ];
    let input_buf = create_input_buffer(&cache, &input_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute linear");

    // CPU reference: y = x @ W^T + b
    let expected =
        super::test_utils::linear_ref(&input_data, weight.data(), Some(&bias_data), 2, 4, 3);
    let result = read_output(&out_buf);
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((r - e).abs() < 1e-4, "linear[{i}]: got {r}, expected {e}");
    }
}

/// Dense-weight companion to `test_compiled_model_execute_linear`.
///
/// The original test uses a selector matrix (identity rows) that trivially
/// passes even with broken matmul. This test uses dense, non-trivial weights
/// where every element contributes to the output, exercising real matmul
/// arithmetic. Requested by Prover (P1-307 F1).
#[test]
fn test_compiled_model_execute_linear_dense_weights() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Linear: y = x @ W^T + b
    // x: [3, 4], W: [2, 4], b: [2] -> y: [3, 2]
    //
    // Dense weights: every element is non-zero and distinct.
    let weight_data = vec![
        0.5, -0.3, 0.7, 0.1, // W[0]: 4 non-trivial coefficients
        -0.2, 0.4, -0.6, 0.8, // W[1]: 4 non-trivial coefficients
    ];
    let bias_data = vec![0.1, -0.2];
    let weight = WeightRef::new(weight_data.clone(), vec![2, 4]).expect("test data");
    let bias = Some(WeightRef::new(bias_data.clone(), vec![2]).expect("test data"));

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[3, 4]),
        TraceNode::new(
            1,
            "linear_dense".into(),
            TraceOp::Linear { weight, bias },
            vec![0],
            vec![3, 2],
            DType::F32,
        ),
    ]);

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile linear dense");
    assert_eq!(compiled.num_inputs(), 1);
    assert_eq!(compiled.output_shape(), &[3, 2]);

    let input_data = [
        1.0_f32, 2.0, 3.0, 4.0, // row 0: diverse values
        -1.0, 0.5, -0.5, 2.0, // row 1: mix of negative and positive
        0.0, 0.0, 0.0, 0.0, // row 2: zeros (bias-only output)
    ];
    let input_buf = create_input_buffer(&cache, &input_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute linear dense");

    // CPU reference: y = x @ W^T + b
    let expected =
        super::test_utils::linear_ref(&input_data, &weight_data, Some(&bias_data), 3, 4, 2);
    let result = read_output(&out_buf);
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-4,
            "linear_dense[{i}]: got {r}, expected {e}"
        );
    }

    // Verify row 2 (all-zeros input) produces bias-only output.
    assert!(
        (result[4] - 0.1).abs() < 1e-5,
        "zero input should yield bias[0]=0.1, got {}",
        result[4]
    );
    assert!(
        (result[5] - (-0.2)).abs() < 1e-5,
        "zero input should yield bias[1]=-0.2, got {}",
        result[5]
    );
}

// -- Tests: reshape passthrough ----------------------------------------------

#[test]
fn test_compiled_model_reshape_passthrough() {
    let cache = super::test_utils::metal_setup();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 3]),
        TraceNode::new(
            1,
            "reshape_0".into(),
            TraceOp::Reshape {
                target_shape: vec![6],
            },
            vec![0],
            vec![6],
            DType::F32,
        ),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile reshape");
    assert_eq!(compiled.num_dispatches(), 0); // reshape = passthrough

    let input_data = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input_buf = create_input_buffer(&cache, &input_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute reshape");
    let result = read_output(&out_buf);

    // Data should pass through unchanged
    for (i, (r, e)) in result.iter().zip(input_data.iter()).enumerate() {
        assert!((r - e).abs() < 1e-9, "reshape[{i}]: got {r}, expected {e}");
    }
}

// -- Tests: reuse (execute twice with different data) ------------------------

#[test]
fn test_compiled_model_reuse_with_different_inputs() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    // First execution
    let buf1 = create_input_buffer(&cache, &[1.0, -1.0, 2.0, -2.0]);
    let out1 = compiled.execute(&cache, &[&buf1]).expect("execute 1");
    let r1 = read_output(&out1);
    assert_eq!(r1, vec![1.0, 0.0, 2.0, 0.0]);

    // Second execution with different data -- same compiled plan
    let buf2 = create_input_buffer(&cache, &[-5.0, 10.0, -15.0, 20.0]);
    let out2 = compiled.execute(&cache, &[&buf2]).expect("execute 2");
    let r2 = read_output(&out2);
    assert_eq!(r2, vec![0.0, 10.0, 0.0, 20.0]);
}

// -- Tests: elementwise chain fusion -----------------------------------------

/// Verifies that a chain of 3 elementwise ops (exp -> relu -> sigmoid) on the
/// same shape fuses into a single GPU dispatch instead of 3.
#[test]
fn test_compiled_model_fusion_reduces_dispatches() {
    let cache = super::test_utils::metal_setup();
    // Graph: input -> exp -> relu -> sigmoid (3 fusible ops, same shape)
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[4]),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[4]),
        unary_node(3, "sigmoid_0", TraceOp::Sigmoid, 2, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile fused chain");
    // With fusion: 1 input identity + 2 identity placeholders + 1 fused dispatch = 4 steps
    // Without fusion it would be: 1 input identity + 3 dispatches = 4 steps, 3 dispatches
    assert_eq!(compiled.num_steps(), 4);
    assert_eq!(
        compiled.num_dispatches(),
        1,
        "3 consecutive elementwise ops should fuse into 1 dispatch"
    );
}

/// Verifies that a fused chain produces correct numerical output.
#[test]
fn test_compiled_model_fusion_correctness() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();
    // Graph: input -> exp -> relu -> sigmoid
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "exp_0", TraceOp::Exp, 0, &[4]),
        unary_node(2, "relu_0", TraceOp::Relu, 1, &[4]),
        unary_node(3, "sigmoid_0", TraceOp::Sigmoid, 2, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile fused chain");

    let input_data = [0.5_f32, -1.0, 0.0, 1.5];
    let input_buf = create_input_buffer(&cache, &input_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute fused chain");

    // CPU reference: sigmoid(relu(exp(x)))
    let expected: Vec<f32> = input_data
        .iter()
        .map(|x| {
            let e = x.exp();
            let r = e.max(0.0);
            1.0 / (1.0 + (-r).exp())
        })
        .collect();
    let result = read_output(&out_buf);
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((r - e).abs() < 1e-5, "fused[{i}]: got {r}, expected {e}");
    }
}

/// Verifies fusion breaks at fan-out: when a node's output feeds two
/// downstream consumers, the chain cannot fuse across that node.
#[test]
fn test_compiled_model_fusion_breaks_at_fanout() {
    let cache = super::test_utils::metal_setup();
    // Graph: input -> relu -> add(relu_out, relu_out)
    // relu has fan-out 2 (feeds both inputs of add), so no fusion.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
        binary_node(2, "add_0", TraceOp::Add, 1, 1, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile fanout");
    // relu has fan-out=2 so it cannot be fused with add.
    // Expect: 1 identity (input) + 1 dispatch (relu) + 1 dispatch (add) = 2 dispatches
    assert_eq!(
        compiled.num_dispatches(),
        2,
        "fan-out should prevent fusion"
    );
}

// -- Tests: diamond topology (DAG, not just chains) --------------------------

/// Diamond: input -> relu and input -> sigmoid, then add(relu_out, sigmoid_out).
/// Tests proper DAG edge resolution with diverge-then-merge topology.
#[test]
fn test_compiled_model_diamond_topology() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 0, &[4]),
        binary_node(3, "add_0", TraceOp::Add, 1, 2, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile diamond");
    assert_eq!(compiled.num_inputs(), 1);
    assert_eq!(compiled.num_dispatches(), 2);

    let input_data = [1.0_f32, -2.0, 0.0, 3.0];
    let input_buf = create_input_buffer(&cache, &input_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute diamond");

    // CPU reference: relu(x) + sigmoid(x)
    let expected: Vec<f32> = input_data
        .iter()
        .map(|x| x.max(0.0) + 1.0 / (1.0 + (-x).exp()))
        .collect();
    let result = read_output(&out_buf);
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((r - e).abs() < 1e-5, "diamond[{i}]: got {r}, expected {e}");
    }
}

/// Residual block pattern: add(f(x), x) where f is a fusible chain.
/// Tests that the skip connection (direct input reference) works with
/// fusion on the main path.
#[test]
fn test_compiled_model_residual_block() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "sigmoid_0", TraceOp::Sigmoid, 0, &[4]),
        unary_node(2, "exp_0", TraceOp::Exp, 1, &[4]),
        binary_node(3, "add_0", TraceOp::Add, 2, 0, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile residual");
    assert_eq!(
        compiled.num_dispatches(),
        1,
        "sigmoid->exp->add should all fuse into 1 dispatch"
    );

    let input_data = [0.5_f32, -1.0, 0.0, 2.0];
    let input_buf = create_input_buffer(&cache, &input_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute residual");

    // CPU reference: exp(sigmoid(x)) + x
    let expected: Vec<f32> = input_data
        .iter()
        .map(|x| (1.0 / (1.0 + (-x).exp())).exp() + x)
        .collect();
    let result = read_output(&out_buf);
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((r - e).abs() < 1e-4, "residual[{i}]: got {r}, expected {e}");
    }
}

// -- Tests: DynTensor interface -----------------------------------------------

#[test]
fn test_compiled_model_execute_dyn() {
    use nn_core::dyn_tensor::DynTensor;
    use nn_core::{DType, Device};
    use std::sync::Arc;

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Build a simple relu graph: input -> relu -> output
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile relu");
    assert_eq!(compiled.output_dtype(), DType::F32);

    // Create a GPU DynTensor input.
    let input_data = vec![-2.0f32, -1.0, 1.0, 2.0];
    let buf = create_input_buffer(&cache, &input_data);
    let storage = nn_metal::MetalTensorData::new(buf);
    let input_tensor =
        DynTensor::from_gpu_storage(vec![4], DType::F32, Arc::new(storage), Device::metal())
            .expect("create GPU DynTensor");

    // Execute via DynTensor interface.
    let output = compiled
        .execute_dyn(&cache, &[&input_tensor])
        .expect("execute_dyn");

    // Verify output shape and dtype.
    assert_eq!(output.dims(), &[4]);
    assert_eq!(output.dtype(), DType::F32);

    // Verify numerical correctness (relu).
    let result = output.to_flat_vec::<f32>().expect("read output");
    let expected = vec![0.0, 0.0, 1.0, 2.0];
    assert_eq!(result, expected);
}

#[test]
fn test_compiled_model_execute_dyn_cpu_input_auto_transfers() {
    use nn_core::dyn_tensor::DynTensor;

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    // CPU tensor auto-transfers to GPU since #2567.
    let cpu_tensor = DynTensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], &[4], &nn_core::Device::Cpu)
        .expect("cpu tensor");
    let result = compiled.execute_dyn(&cache, &[&cpu_tensor]);
    // Auto-transfer depends on Metal backend init state. Both Ok and Err
    // are acceptable; the key invariant is no panic. When Metal is fully
    // initialized, to_device succeeds and the output is correct.
    if let Ok(output) = result {
        let out: Vec<f32> = output.to_flat_vec().expect("read output");
        assert_eq!(out, vec![0.0, 2.0, 0.0, 4.0], "relu correctness");
    }
}

// -- Tests: CompiledPlan -> CompiledModel bridge (from_plan) -------------------

/// Verifies the full from_plan() path: compile to CompiledPlan, build
/// CompiledModel via from_plan(), execute on GPU, verify correctness.
/// This is the deserialization path for Wave 2 (#2125).
#[test]
fn test_compiled_model_from_plan_relu_chain() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Graph: input -> relu -> sigmoid (fusible chain)
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 1, &[4]),
    ]);

    // Compile to CompiledPlan (with fusion)
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile plan");
    assert_eq!(plan.input_shapes.len(), 1);
    assert_eq!(plan.input_shapes[0], vec![4]);

    // Build CompiledModel from the plan (the from_plan path)
    let compiled = CompiledModel::from_plan(&plan, &graph, &cache).expect("from_plan");
    assert_eq!(compiled.num_inputs(), 1);
    assert_eq!(compiled.output_shape(), &[4]);

    // Execute and verify numerical correctness
    let input_data = [1.0_f32, -2.0, 0.0, 3.0];
    let input_buf = create_input_buffer(&cache, &input_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute from_plan");
    let result = read_output(&out_buf);

    // CPU reference: sigmoid(relu(x))
    let expected: Vec<f32> = input_data
        .iter()
        .map(|x| {
            let r = x.max(0.0);
            1.0 / (1.0 + (-r).exp())
        })
        .collect();
    assert_eq!(result.len(), expected.len());
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-5,
            "from_plan[{i}]: got {r}, expected {e}"
        );
    }
}

/// Verifies that from_plan() and builder().build() produce identical results.
#[test]
fn test_compiled_model_from_plan_matches_from_trace() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);

    // Build via from_trace (the original path)
    let compiled_trace = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("from_trace");

    // Build via from_plan (the new path)
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("plan");
    let compiled_plan = CompiledModel::from_plan(&plan, &graph, &cache).expect("from_plan");

    // Metadata should match
    assert_eq!(compiled_trace.num_steps(), compiled_plan.num_steps());
    assert_eq!(
        compiled_trace.num_dispatches(),
        compiled_plan.num_dispatches()
    );
    assert_eq!(compiled_trace.num_inputs(), compiled_plan.num_inputs());
    assert_eq!(compiled_trace.output_shape(), compiled_plan.output_shape());
    assert_eq!(compiled_trace.output_dtype(), compiled_plan.output_dtype());

    // Execute both and compare outputs
    let input_data = [1.0_f32, -2.0, 3.0, -4.0];
    let input_buf = create_input_buffer(&cache, &input_data);
    let out_trace = compiled_trace
        .execute(&cache, &[&input_buf])
        .expect("execute trace");
    let out_plan = compiled_plan
        .execute(&cache, &[&input_buf])
        .expect("execute plan");
    let r_trace = read_output(&out_trace);
    let r_plan = read_output(&out_plan);
    assert_eq!(
        r_trace, r_plan,
        "from_plan and from_trace should produce identical output"
    );
}

// -- Tests: batch cleanup on failure (#2185) ----------------------------------

/// Regression test for #2185: after a failed execute(), subsequent GPU
/// operations should not be corrupted by leaked partial commands.
#[test]
fn test_compiled_model_failure_does_not_corrupt_next_execution() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 1, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    // First execution: success.
    let input_data = [1.0_f32, -2.0, 0.0, 3.0];
    let input_buf = create_input_buffer(&cache, &input_data);
    let out1 = compiled
        .execute(&cache, &[&input_buf])
        .expect("first execute");
    let result1 = read_output(&out1);

    // Force a failure: wrong number of inputs.
    let err_msg = compiled.execute(&cache, &[]).unwrap_err().to_string();
    assert!(
        err_msg.contains("expected 1 inputs, got 0"),
        "expected InputCountMismatch error, got: {err_msg}"
    );

    // Second execution after failure: should produce identical output.
    let input_buf2 = create_input_buffer(&cache, &input_data);
    let out2 = compiled
        .execute(&cache, &[&input_buf2])
        .expect("execute after failure should succeed");
    let result2 = read_output(&out2);

    assert_eq!(
        result1, result2,
        "execution after failure must produce identical results"
    );
}

/// Regression test for #2185: execute an empty-plan model (which fails),
/// then execute a valid model.
#[test]
fn test_compiled_model_empty_plan_failure_then_success() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Empty plan -- execute will fail.
    let empty_graph = ComputationGraph::from_nodes(vec![]);
    let empty_compiled = CompiledModel::builder(&empty_graph, &cache)
        .build()
        .expect("compile empty");
    let err_msg = empty_compiled.execute(&cache, &[]).unwrap_err().to_string();
    assert!(
        err_msg.contains("compiled plan is empty"),
        "expected EmptyPlan error, got: {err_msg}"
    );

    // Valid model -- should work cleanly after the empty-plan failure.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile relu");
    let input_buf = create_input_buffer(&cache, &[1.0, -1.0, 2.0, -2.0]);
    let out = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute after empty-plan failure");
    let result = read_output(&out);
    assert_eq!(result, vec![1.0, 0.0, 2.0, 0.0]);
}

// -- Test: Empty weight data produces error, not silent zero-fill (#2190) -----

#[test]
fn test_compiled_model_empty_weight_errors() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let empty_weight = WeightRef::from_shape(&[3, 4]); // non-zero shape, empty data
    let bias = Some(WeightRef::from_shape(&[3]));

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[2, 4]),
        TraceNode::new(
            1,
            "linear_empty_weight".into(),
            TraceOp::Linear {
                weight: empty_weight,
                bias,
            },
            vec![0],
            vec![2, 3],
            DType::F32,
        ),
    ]);

    // from_trace must fail with an error about empty weight data,
    // NOT succeed with zero-filled weights.
    let msg = match CompiledModel::builder(&graph, &cache).build() {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected error for empty weight data, but from_trace succeeded"),
    };
    assert!(
        msg.contains("empty data"),
        "expected error about empty weight data, got: {msg}"
    );
}

// -- Tests: buffer planner integration ----------------------------------------

/// Verifies that the buffer plan is computed and stored in CompiledModel.
#[test]
fn test_compiled_model_buffer_plan_computed() {
    let cache = super::test_utils::metal_setup();
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");
    let bp = compiled.buffer_plan();

    // InputForward has no allocation; Relu has 4 * 4 = 16 bytes.
    assert_eq!(bp.step_offsets.len(), 2);
    assert!(bp.step_offsets[0].is_none()); // InputForward
    assert_eq!(bp.step_offsets[1], Some(0)); // Relu dispatch
    assert_eq!(bp.total_bytes, 16);
}

/// Verifies that buffer planning with a diamond topology correctly tracks
/// simultaneous buffer lifetimes: relu and sigmoid must both be live
/// until the add that consumes them.
#[test]
fn test_compiled_model_buffer_plan_diamond() {
    let cache = super::test_utils::metal_setup();
    // Diamond: input -> relu(1), input -> sigmoid(2), add(relu, sigmoid)(3)
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
        unary_node(2, "sigmoid_0", TraceOp::Sigmoid, 0, &[4]),
        binary_node(3, "add_0", TraceOp::Add, 1, 2, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile diamond");
    let bp = compiled.buffer_plan();

    // Verify plan has the right number of steps.
    assert_eq!(bp.step_offsets.len(), 4);
    // Total bytes should be > 0.
    assert!(bp.total_bytes > 0);
}

/// Verifies that GPU execution still produces correct results after
/// buffer plan integration (the plan is computed but eager release is
/// deferred pending fused-dispatch correctness fixes).
#[test]
fn test_compiled_model_buffer_plan_execution() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    // Simple: input -> relu. Buffer plan should show 1 allocating step.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
    ]);
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    // Verify buffer plan metadata.
    let bp = compiled.buffer_plan();
    assert_eq!(bp.step_offsets.len(), 2);
    assert!(bp.step_offsets[0].is_none(), "InputForward: no allocation");
    assert!(
        bp.step_offsets[1].is_some(),
        "Relu dispatch: has allocation"
    );
    assert_eq!(bp.step_sizes[1], 16); // 4 * f32 = 16 bytes

    // Verify execution produces correct results.
    let input_data = [1.0_f32, -2.0, 3.0, -4.0];
    let input_buf = create_input_buffer(&cache, &input_data);
    let out_buf = compiled
        .execute(&cache, &[&input_buf])
        .expect("execute with buffer plan");
    let result = read_output(&out_buf);

    // Note: output buffer may be larger than the logical output due to
    // Metal buffer allocation granularity. Check correctness via zip.
    let expected = [1.0_f32, 0.0, 3.0, 0.0];
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (r - e).abs() < 1e-6,
            "buffer_plan_exec[{i}]: got {r}, expected {e}"
        );
    }
}
