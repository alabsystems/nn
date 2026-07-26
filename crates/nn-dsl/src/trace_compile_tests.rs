// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `compile_trace()` in `trace_compile.rs`.
//!
//! Core structural tests: identity ops, shape passthroughs, binary/unary
//! element-wise, activation routing, error paths, matmul, narrow.
//!
//! Reduce/keepdim/softmax/clamp tests → `trace_compile_tests_reduce.rs`.
//! Weighted ops/plan/cumsum/repeat/constant → `trace_compile_tests_conv.rs`.

use nn_core::dyn_tensor::trace::{
    ComputationGraph, TraceActivation, TraceNode, TraceOp, WeightRef,
};
use nn_core::DType;

use crate::trace_compile::{compile_trace, CompiledStep};

// -- Helpers ------------------------------------------------------------------

/// Build a minimal `ComputationGraph` from a slice of `TraceNode`s.
///
/// The first node is the output. Nodes are stored in insertion order
/// (topological for our purposes since we control the DAG).
fn graph_from_nodes(nodes: Vec<TraceNode>) -> ComputationGraph {
    ComputationGraph::from_nodes(nodes)
}

fn input_node(id: u64, shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        format!("input_{id}"),
        TraceOp::Input,
        vec![],
        shape.to_vec(),
        DType::F32,
    )
}

fn unary_node(id: u64, name: &str, op: TraceOp, input_id: u64, shape: &[usize]) -> TraceNode {
    TraceNode::new(
        id,
        name.to_string(),
        op,
        vec![input_id],
        shape.to_vec(),
        DType::F32,
    )
}

fn binary_node(
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

// -- Tests --------------------------------------------------------------------

#[test]
fn test_compile_empty_graph() {
    let graph = graph_from_nodes(vec![]);
    let steps = compile_trace(&graph).expect("empty graph should compile");
    assert!(steps.is_empty());
}

#[test]
fn test_compile_single_input() {
    let graph = graph_from_nodes(vec![input_node(0, &[2, 3])]);
    let steps = compile_trace(&graph).expect("single input should compile");
    assert_eq!(steps.len(), 1);
    assert!(matches!(steps[0], CompiledStep::InputForward));
}

#[test]
fn test_compile_binary_add() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        binary_node(2, "add_0", TraceOp::Add, 0, 1, &[4]),
    ]);
    let steps = compile_trace(&graph).expect("add should compile");
    assert_eq!(steps.len(), 3);
    // First two are InputForward (inputs), third is Dispatch (add)
    assert!(matches!(steps[0], CompiledStep::InputForward));
    assert!(matches!(steps[1], CompiledStep::InputForward));
    assert!(matches!(steps[2], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_binary_mul() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[8]),
        input_node(1, &[8]),
        binary_node(2, "mul_0", TraceOp::Mul, 0, 1, &[8]),
    ]);
    let steps = compile_trace(&graph).expect("mul should compile");
    assert_eq!(steps.len(), 3);
    assert!(matches!(steps[2], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_binary_sub() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        binary_node(2, "sub_0", TraceOp::Sub, 0, 1, &[4]),
    ]);
    let steps = compile_trace(&graph).expect("sub should compile");
    assert_eq!(steps.len(), 3);
    assert!(matches!(steps[2], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_unary_relu() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[2, 4]),
    ]);
    let steps = compile_trace(&graph).expect("relu should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_unary_gelu() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[16]),
        unary_node(1, "gelu_0", TraceOp::Gelu, 0, &[16]),
    ]);
    let steps = compile_trace(&graph).expect("gelu should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_unary_sigmoid() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "sigmoid_0", TraceOp::Sigmoid, 0, &[4]),
    ]);
    let steps = compile_trace(&graph).expect("sigmoid should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_sqrt() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[8]),
        unary_node(1, "sqrt_0", TraceOp::Sqrt, 0, &[8]),
    ]);
    let steps = compile_trace(&graph).expect("sqrt should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_dropout_is_identity() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "dropout_0", TraceOp::Dropout, 0, &[4]),
    ]);
    let steps = compile_trace(&graph).expect("dropout should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::IdentityPassthrough));
}

#[test]
fn test_compile_reshape_passthrough() {
    let graph = graph_from_nodes(vec![
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
    let steps = compile_trace(&graph).expect("reshape should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::Passthrough {
            op_name,
            output_shape,
        } => {
            assert_eq!(op_name, "reshape");
            assert_eq!(output_shape, &[6]);
        }
        other => panic!("expected Passthrough, got {other:?}"),
    }
}

#[test]
fn test_compile_unsqueeze_passthrough() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[4]),
        TraceNode::new(
            1,
            "unsqueeze_0".into(),
            TraceOp::Unsqueeze { dim: 0 },
            vec![0],
            vec![1, 4],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("unsqueeze should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::Passthrough { .. }));
}

#[test]
fn test_compile_permute_produces_dispatch_not_passthrough() {
    // Permute reorders physical data layout — it must produce a Dispatch step,
    // NOT a Passthrough (which just aliases the buffer). This test catches the
    // correctness bug where Permute was grouped with Reshape/Squeeze/Unsqueeze.
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 3, 4]),
        TraceNode::new(
            1,
            "permute_0".into(),
            TraceOp::Permute {
                axes: vec![2, 0, 1],
            },
            vec![0],
            vec![4, 2, 3],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("permute should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::Dispatch { kernel, .. } => {
            assert_eq!(kernel.name(), "permute");
        }
        CompiledStep::Passthrough { .. } => {
            panic!("Permute must produce Dispatch (GPU kernel), not Passthrough (buffer alias)");
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

#[test]
fn test_compile_chain_relu_add() {
    // Input → Relu → Add(relu_out, input) — 2 ops chained
    let graph = graph_from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_0", TraceOp::Relu, 0, &[4]),
        binary_node(2, "add_0", TraceOp::Add, 1, 0, &[4]),
    ]);
    let steps = compile_trace(&graph).expect("chain should compile");
    assert_eq!(steps.len(), 3);
    assert!(matches!(steps[0], CompiledStep::InputForward)); // input
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. })); // relu
    assert!(matches!(steps[2], CompiledStep::Dispatch { .. })); // add
}

#[test]
fn test_compile_unsupported_op_returns_error() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[4]),
        TraceNode::new(
            1,
            "custom_0".into(),
            TraceOp::Custom {
                name: "nn_custom".into(),
            },
            vec![0],
            vec![4],
            DType::F32,
        ),
    ]);
    let result = compile_trace(&graph);
    assert!(result.is_err(), "custom op should fail");
}

#[test]
fn test_compile_matmul() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 4]),
        input_node(1, &[4, 3]),
        binary_node(2, "matmul_0", TraceOp::MatMul, 0, 1, &[2, 3]),
    ]);
    let steps = compile_trace(&graph).expect("matmul should compile");
    assert_eq!(steps.len(), 3);
    assert!(matches!(steps[2], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_narrow() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 8]),
        TraceNode::new(
            1,
            "narrow_0".into(),
            TraceOp::Narrow {
                dim: 1,
                start: 2,
                length: 4,
            },
            vec![0],
            vec![2, 4],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("narrow should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_narrow_open_ended_non_contiguous_clamps_length() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 22, 3001]),
        TraceNode::new(
            1,
            "narrow_0".into(),
            TraceOp::Narrow {
                dim: 1,
                start: 11,
                length: usize::MAX,
            },
            vec![0],
            vec![2, 11, 3001],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("open-ended narrow should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
}

/// Contiguous narrow (leading dims all 1) compiles to NarrowView (zero-copy).
#[test]
fn test_compile_narrow_contiguous_is_narrow_view() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 8]),
        TraceNode::new(
            1,
            "narrow_0".into(),
            TraceOp::Narrow {
                dim: 1,
                start: 2,
                length: 4,
            },
            vec![0],
            vec![1, 4],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("contiguous narrow should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::NarrowView {
            byte_offset,
            output_shape,
            ..
        } => {
            // start=2, trailing=1, sizeof(f32)=4 → byte_offset=8
            assert_eq!(*byte_offset, 8);
            assert_eq!(output_shape, &[1, 4]);
        }
        other => panic!("expected NarrowView, got {other:?}"),
    }
}

#[test]
fn test_compile_narrow_open_ended_contiguous_is_narrow_view() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 22, 3001]),
        TraceNode::new(
            1,
            "narrow_0".into(),
            TraceOp::Narrow {
                dim: 1,
                start: 11,
                length: usize::MAX,
            },
            vec![0],
            vec![1, 11, 3001],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("contiguous open-ended narrow should compile");
    assert_eq!(steps.len(), 2);
    match &steps[1] {
        CompiledStep::NarrowView { output_shape, .. } => assert_eq!(output_shape, &[1, 11, 3001]),
        other => panic!("expected NarrowView, got {other:?}"),
    }
}

// -- Extended unary/binary ops ------------------------------------------------

#[test]
fn test_compile_gelu_erf() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[16]),
        unary_node(1, "gelu_erf_0", TraceOp::GeluErf, 0, &[16]),
    ]);
    let steps = compile_trace(&graph).expect("gelu_erf should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_silu() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[8]),
        unary_node(1, "silu_0", TraceOp::Silu, 0, &[8]),
    ]);
    let steps = compile_trace(&graph).expect("silu should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_activation_named_gelu() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[16]),
        TraceNode::new(
            1,
            "activation_0".into(),
            TraceOp::Activation {
                kind: TraceActivation::Gelu,
            },
            vec![0],
            vec![16],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("activation(gelu) should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_activation_named_gelu_erf() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[16]),
        TraceNode::new(
            1,
            "activation_0".into(),
            TraceOp::Activation {
                kind: TraceActivation::GeluErf,
            },
            vec![0],
            vec![16],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("activation(gelu_erf) should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_activation_named_leaky_relu_rejected() {
    // Generic Activation { name: "LeakyRelu" } rejected to prevent hardcoded slope=0.01
    // from silently producing wrong results. Use TraceOp::LeakyRelu { slope }. #2267
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 8]),
        TraceNode::new(
            1,
            "activation_0".into(),
            TraceOp::Activation {
                kind: TraceActivation::LeakyRelu,
            },
            vec![0],
            vec![2, 8],
            DType::F32,
        ),
    ]);
    let err = compile_trace(&graph).unwrap_err();
    assert!(
        format!("{err:?}").contains("LeakyRelu"),
        "should reject generic LeakyRelu path: {err:?}"
    );
}

#[test]
fn test_compile_activation_leaky_relu_lowercase_rejected() {
    // Same as above for lowercase variant. #2267
    let graph = graph_from_nodes(vec![
        input_node(0, &[16]),
        TraceNode::new(
            1,
            "activation_0".into(),
            TraceOp::Activation {
                kind: TraceActivation::LeakyRelu,
            },
            vec![0],
            vec![16],
            DType::F32,
        ),
    ]);
    let err = compile_trace(&graph).unwrap_err();
    assert!(
        format!("{err:?}").contains("LeakyRelu"),
        "should reject generic leaky_relu path: {err:?}"
    );
}

#[test]
fn test_compile_activation_named_relu() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[16]),
        TraceNode::new(
            1,
            "activation_0".into(),
            TraceOp::Activation {
                kind: TraceActivation::Relu,
            },
            vec![0],
            vec![16],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("activation(relu) should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_activation_named_sigmoid() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[16]),
        TraceNode::new(
            1,
            "activation_0".into(),
            TraceOp::Activation {
                kind: TraceActivation::Sigmoid,
            },
            vec![0],
            vec![16],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("activation(sigmoid) should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_activation_named_silu() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[16]),
        TraceNode::new(
            1,
            "activation_0".into(),
            TraceOp::Activation {
                kind: TraceActivation::Silu,
            },
            vec![0],
            vec![16],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("activation(silu) should compile");
    assert_eq!(steps.len(), 2);
}

#[test]
fn test_compile_activation_named_tanh() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[16]),
        TraceNode::new(
            1,
            "activation_0".into(),
            TraceOp::Activation {
                kind: TraceActivation::Tanh,
            },
            vec![0],
            vec![16],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("activation(tanh) should compile");
    assert_eq!(steps.len(), 2);
}

#[test]
fn test_compile_activation_named_exp() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[16]),
        TraceNode::new(
            1,
            "activation_0".into(),
            TraceOp::Activation {
                kind: TraceActivation::Exp,
            },
            vec![0],
            vec![16],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("activation(exp) should compile");
    assert_eq!(steps.len(), 2);
}

#[test]
fn test_compile_activation_named_log() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[16]),
        TraceNode::new(
            1,
            "activation_0".into(),
            TraceOp::Activation {
                kind: TraceActivation::Log,
            },
            vec![0],
            vec![16],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("activation(log) should compile");
    assert_eq!(steps.len(), 2);
}

#[test]
fn test_compile_activation_named_elu_rejected() {
    // Generic Activation { name: "elu" } is rejected to prevent hardcoded alpha=1.0
    // from silently producing wrong results. Use TraceOp::Elu { alpha } instead. #2267
    let graph = graph_from_nodes(vec![
        input_node(0, &[16]),
        TraceNode::new(
            1,
            "activation_0".into(),
            TraceOp::Activation {
                kind: TraceActivation::Elu,
            },
            vec![0],
            vec![16],
            DType::F32,
        ),
    ]);
    let err = compile_trace(&graph).unwrap_err();
    assert!(
        format!("{err:?}").contains("Elu"),
        "should reject generic Elu path: {err:?}"
    );
}

#[test]
fn test_compile_maximum() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        binary_node(2, "maximum_0", TraceOp::Maximum, 0, 1, &[4]),
    ]);
    let steps = compile_trace(&graph).expect("maximum should compile");
    assert_eq!(steps.len(), 3);
    assert!(matches!(steps[2], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_minimum() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[4]),
        input_node(1, &[4]),
        binary_node(2, "minimum_0", TraceOp::Minimum, 0, 1, &[4]),
    ]);
    let steps = compile_trace(&graph).expect("minimum should compile");
    assert_eq!(steps.len(), 3);
    assert!(matches!(steps[2], CompiledStep::Dispatch { .. }));
}

// -- Error path tests (F4, F5, F7 from #2086) ---------------------------------

#[test]
fn test_compile_transpose_oob_dims_returns_error() {
    // F4: Transpose with dim0 or dim1 >= ndim must return Err, not silent identity.
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 3]),
        TraceNode::new(
            1,
            "transpose_1".into(),
            TraceOp::Transpose { dim0: 0, dim1: 5 }, // dim1=5 is OOB for rank 2
            vec![0],
            vec![2, 3],
            DType::F32,
        ),
    ]);
    let err = compile_trace(&graph).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("out of bounds"),
        "expected TransposeDimOutOfBounds error, got: {msg}"
    );
}

#[test]
fn test_compile_layer_norm_nonfinite_eps_returns_error() {
    // F5: eps that overflows f32 must return Err, not silently produce Inf.
    let weight = WeightRef::new(vec![1.0; 4], vec![4]).expect("test data");
    let bias = WeightRef::new(vec![0.0; 4], vec![4]).expect("test data");
    let graph = graph_from_nodes(vec![
        input_node(0, &[1, 4]),
        TraceNode::new(
            1,
            "layer_norm_1".into(),
            TraceOp::LayerNorm {
                eps: f64::INFINITY,
                weight,
                bias,
            },
            vec![0],
            vec![1, 4],
            DType::F32,
        ),
    ]);
    let err = compile_trace(&graph).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite"),
        "expected NonFiniteConstant error, got: {msg}"
    );
}

#[test]
fn test_compile_clamp_nonfinite_min_returns_error() {
    // F5: clamp min that is NaN must return Err.
    let graph = graph_from_nodes(vec![
        input_node(0, &[8]),
        TraceNode::new(
            1,
            "clamp_1".into(),
            TraceOp::Clamp {
                min: Some(f64::NAN),
                max: None,
            },
            vec![0],
            vec![8],
            DType::F32,
        ),
    ]);
    let err = compile_trace(&graph).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("non-finite"),
        "expected NonFiniteConstant error for NaN clamp_min, got: {msg}"
    );
}

#[test]
fn test_compile_bad_topology_returns_error() {
    // F7: Nodes referencing future nodes must error, not produce wrong results.
    // Node 1 references node 2 which appears after it — out-of-order.
    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[4]),
        unary_node(1, "relu_1", TraceOp::Relu, 2, &[4]), // references node 2 (not yet seen)
        unary_node(2, "relu_2", TraceOp::Relu, 0, &[4]),
    ]);
    let err = compile_trace(&graph).unwrap_err();
    let msg = err.to_string();
    // relu_1's only input (index 0) references node 2 which hasn't been seen.
    assert!(
        msg.contains("missing input at index 0") && msg.contains("input_id=2"),
        "expected MissingInputNode for out-of-order reference at input 0, got: {msg}"
    );
}

// -- Dedicated Elu / LeakyRelu TraceOp variants (#2246) ----------------------

#[test]
fn test_compile_elu_with_alpha() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 8]),
        TraceNode::new(
            1,
            "elu_0".into(),
            TraceOp::Elu { alpha: 0.5 },
            vec![0],
            vec![2, 8],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("elu should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(&steps[1], CompiledStep::Dispatch { .. }));
}

#[test]
fn test_compile_leaky_relu_with_slope() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 8]),
        TraceNode::new(
            1,
            "leaky_relu_0".into(),
            TraceOp::LeakyRelu { slope: 0.2 },
            vec![0],
            vec![2, 8],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("leaky_relu should compile");
    assert_eq!(steps.len(), 2);
    assert!(matches!(&steps[1], CompiledStep::Dispatch { .. }));
}

/// Verify the optimized leaky_relu decomposition (max(x, alpha*x)) produces
/// fewer Metal dispatch steps than the old 7-node decomposition. Part of #1815.
#[test]
fn test_compile_leaky_relu_optimized_plan_steps() {
    use crate::codegen_msl_tensor::build_dispatch_plan;
    use crate::ir::ScalarType;

    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 8]),
        TraceNode::new(
            1,
            "leaky_relu_0".into(),
            TraceOp::LeakyRelu { slope: 0.2 },
            vec![0],
            vec![2, 8],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("leaky_relu should compile");
    let kernel = match &steps[1] {
        CompiledStep::Dispatch { kernel, .. } => kernel,
        other => panic!("expected Dispatch, got {:?}", std::mem::discriminant(other)),
    };
    let (plan, _) =
        build_dispatch_plan(kernel.def(), ScalarType::F32).expect("leaky_relu plan should build");
    // Optimized max(x, alpha*x): broadcast(slope) + mul + max = 3 IR nodes.
    // After broadcast fusion in build_dispatch_plan, broadcast may be absorbed
    // into the mul step, giving 2 plan steps. Assert <= 3 (was 5+ with old path).
    assert!(
        plan.len() <= 3,
        "optimized leaky_relu should produce <= 3 plan steps, got {}",
        plan.len()
    );
    // Verify it's strictly fewer than the old 7-node decomposition (which was 5 plan steps).
    assert!(
        plan.len() < 5,
        "optimized leaky_relu ({}) should be fewer than old decomposition (5)",
        plan.len()
    );
}

// -- IndexSelect / Gather (#2177) --------------------------------------------

#[test]
fn test_compile_index_select() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[10, 4]),
        input_node(1, &[3]),
        TraceNode::new(
            2,
            "index_select_0".into(),
            TraceOp::IndexSelect { dim: 0 },
            vec![0, 1],
            vec![3, 4],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("index_select should compile");
    assert_eq!(steps.len(), 3);
    match &steps[2] {
        CompiledStep::Dispatch { kernel, .. } => {
            assert_eq!(kernel.name(), "index_select");
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}

#[test]
fn test_compile_gather() {
    let graph = graph_from_nodes(vec![
        input_node(0, &[2, 5]),
        input_node(1, &[2, 3]),
        TraceNode::new(
            2,
            "gather_0".into(),
            TraceOp::Gather { dim: 1 },
            vec![0, 1],
            vec![2, 3],
            DType::F32,
        ),
    ]);
    let steps = compile_trace(&graph).expect("gather should compile");
    assert_eq!(steps.len(), 3);
    match &steps[2] {
        CompiledStep::Dispatch { kernel, .. } => {
            assert_eq!(kernel.name(), "gather");
        }
        other => panic!("expected Dispatch, got {other:?}"),
    }
}
